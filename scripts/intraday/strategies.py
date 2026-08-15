"""
Candidate intraday strategies, each long AND short capable. All operate on a
single symbol's intraday bar data:

    DataFrame indexed by tz-naive datetime (IST), columns:
        open, high, low, close, volume
    spanning multiple trading days, one row per bar (e.g. 15-minute).

Each strategy function returns a DataFrame of trades:
    date, direction (+1 long / -1 short), entry_time, entry_price,
    exit_time, exit_price, exit_reason

No costs are applied here — that's the engine's job (costs.py + engine.py).
Entries execute at the *next* bar's open after a signal bar closes (no
look-ahead: the signal is only known once the signal bar itself has closed).
"""

import pandas as pd


def _day_groups(df: pd.DataFrame):
    return df.groupby(df.index.date)


def opening_range_breakout(df: pd.DataFrame, or_bars: int = 2, stop_r: float = 1.0,
                            target_r: float = 2.0) -> pd.DataFrame:
    """Classic ORB: define the opening range as the high/low of the first
    `or_bars` bars of the day. First close that breaks above OR-high after
    the opening range -> long next bar's open. First close that breaks below
    OR-low -> short next bar's open. Stop-loss and target are set in
    "R" multiples of the opening-range width. Square off at day's last bar
    if neither stop nor target is hit.
    """
    trades = []
    for day, day_df in _day_groups(df):
        if len(day_df) <= or_bars + 1:
            continue
        or_high = day_df["high"].iloc[:or_bars].max()
        or_low = day_df["low"].iloc[:or_bars].min()
        or_range = or_high - or_low
        if or_range <= 0:
            continue

        rest = day_df.iloc[or_bars:]
        in_position = False
        direction = 0
        entry_price = entry_time = None
        stop = target = None

        bars = list(rest.itertuples())
        for i, bar in enumerate(bars):
            if not in_position:
                if bar.close > or_high:
                    direction = 1
                elif bar.close < or_low:
                    direction = -1
                else:
                    continue
                if i + 1 >= len(bars):
                    break  # no next bar to enter on
                next_bar = bars[i + 1]
                entry_price = next_bar.open
                entry_time = next_bar.Index
                if direction == 1:
                    stop = entry_price - stop_r * or_range
                    target = entry_price + target_r * or_range
                else:
                    stop = entry_price + stop_r * or_range
                    target = entry_price - target_r * or_range
                in_position = True
                continue

            # in position: check stop/target intrabar using high/low
            if direction == 1:
                hit_stop = bar.low <= stop
                hit_target = bar.high >= target
            else:
                hit_stop = bar.high >= stop
                hit_target = bar.low <= target

            if hit_stop or hit_target:
                exit_price = stop if hit_stop else target
                trades.append({
                    "date": day, "direction": direction,
                    "entry_time": entry_time, "entry_price": entry_price,
                    "exit_time": bar.Index, "exit_price": exit_price,
                    "exit_reason": "stop" if hit_stop else "target",
                })
                in_position = False

        if in_position:
            last_bar = bars[-1]
            trades.append({
                "date": day, "direction": direction,
                "entry_time": entry_time, "entry_price": entry_price,
                "exit_time": last_bar.Index, "exit_price": last_bar.close,
                "exit_reason": "eod",
            })

    return pd.DataFrame(trades)


def vwap_mean_reversion(df: pd.DataFrame, entry_z: float = 1.5, lookback_bars: int = 10) -> pd.DataFrame:
    """Intraday mean reversion to VWAP. Computes a running VWAP per day and a
    rolling std of (close - vwap). Enters when price is entry_z std below
    VWAP (long, betting on reversion up) or above (short). Exits on
    VWAP touch or end of day.
    """
    trades = []
    for day, day_df in _day_groups(df):
        if len(day_df) <= lookback_bars + 1:
            continue
        typical = (day_df["high"] + day_df["low"] + day_df["close"]) / 3
        cum_vol = day_df["volume"].cumsum().replace(0, pd.NA)
        vwap = (typical * day_df["volume"]).cumsum() / cum_vol
        vwap = vwap.ffill().fillna(day_df["close"])
        dev = day_df["close"] - vwap
        dev_std = dev.rolling(lookback_bars, min_periods=lookback_bars).std()

        in_position = False
        direction = 0
        entry_price = entry_time = None

        bars = list(day_df.itertuples())
        for i in range(lookback_bars, len(bars)):
            bar = bars[i]
            z = dev.iloc[i] / dev_std.iloc[i] if dev_std.iloc[i] and dev_std.iloc[i] > 0 else 0

            if not in_position:
                if z <= -entry_z:
                    direction = 1
                elif z >= entry_z:
                    direction = -1
                else:
                    continue
                if i + 1 >= len(bars):
                    break
                next_bar = bars[i + 1]
                entry_price = next_bar.open
                entry_time = next_bar.Index
                in_position = True
                continue

            crossed_vwap = (
                (direction == 1 and bar.close >= vwap.iloc[i]) or
                (direction == -1 and bar.close <= vwap.iloc[i])
            )
            if crossed_vwap:
                trades.append({
                    "date": day, "direction": direction,
                    "entry_time": entry_time, "entry_price": entry_price,
                    "exit_time": bar.Index, "exit_price": bar.close,
                    "exit_reason": "vwap_touch",
                })
                in_position = False

        if in_position:
            last_bar = bars[-1]
            trades.append({
                "date": day, "direction": direction,
                "entry_time": entry_time, "entry_price": entry_price,
                "exit_time": last_bar.Index, "exit_price": last_bar.close,
                "exit_reason": "eod",
            })

    return pd.DataFrame(trades)


def momentum_volume_breakout(df: pd.DataFrame, or_bars: int = 4, vol_mult: float = 1.5,
                              target_r: float = 3.0, stop_r: float = 1.0) -> pd.DataFrame:
    """Like ORB, but only takes the breakout if the breakout bar's volume is
    vol_mult times the average volume of the opening range bars — a filter
    meant to skip low-conviction breakouts.
    """
    trades = []
    for day, day_df in _day_groups(df):
        if len(day_df) <= or_bars + 1:
            continue
        or_high = day_df["high"].iloc[:or_bars].max()
        or_low = day_df["low"].iloc[:or_bars].min()
        or_range = or_high - or_low
        or_avg_vol = day_df["volume"].iloc[:or_bars].mean()
        if or_range <= 0 or or_avg_vol <= 0:
            continue

        rest = day_df.iloc[or_bars:]
        in_position = False
        direction = 0
        entry_price = entry_time = None
        stop = target = None

        bars = list(rest.itertuples())
        for i, bar in enumerate(bars):
            if not in_position:
                strong_volume = bar.volume >= vol_mult * or_avg_vol
                if strong_volume and bar.close > or_high:
                    direction = 1
                elif strong_volume and bar.close < or_low:
                    direction = -1
                else:
                    continue
                if i + 1 >= len(bars):
                    break
                next_bar = bars[i + 1]
                entry_price = next_bar.open
                entry_time = next_bar.Index
                if direction == 1:
                    stop = entry_price - stop_r * or_range
                    target = entry_price + target_r * or_range
                else:
                    stop = entry_price + stop_r * or_range
                    target = entry_price - target_r * or_range
                in_position = True
                continue

            if direction == 1:
                hit_stop = bar.low <= stop
                hit_target = bar.high >= target
            else:
                hit_stop = bar.high >= stop
                hit_target = bar.low <= target

            if hit_stop or hit_target:
                exit_price = stop if hit_stop else target
                trades.append({
                    "date": day, "direction": direction,
                    "entry_time": entry_time, "entry_price": entry_price,
                    "exit_time": bar.Index, "exit_price": exit_price,
                    "exit_reason": "stop" if hit_stop else "target",
                })
                in_position = False

        if in_position:
            last_bar = bars[-1]
            trades.append({
                "date": day, "direction": direction,
                "entry_time": entry_time, "entry_price": entry_price,
                "exit_time": last_bar.Index, "exit_price": last_bar.close,
                "exit_reason": "eod",
            })

    return pd.DataFrame(trades)


STRATEGIES = {
    "opening_range_breakout": opening_range_breakout,
    "vwap_mean_reversion": vwap_mean_reversion,
    "momentum_volume_breakout": momentum_volume_breakout,
}
