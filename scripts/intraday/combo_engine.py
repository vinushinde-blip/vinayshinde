"""
Turns a (trigger, filter, stop/target) combination into a trade list, using
the same output shape as strategies.py (date, direction, entry_time,
entry_price, exit_time, exit_price, exit_reason) so it plugs into the
existing engine.py/metrics.py pipeline unchanged.

Indicators are computed once per symbol on the whole continuous series (not
reset each day — same as any charting platform's intraday RSI/MACD/etc.).
Positions respect market_rules: no new entries within ~12 min of the real
Zerodha MIS square-off (~15:12 IST), and a still-open position is force-
exited at the bar covering that cutoff rather than the day's literal last
bar (~15-20 min later than a live account could actually hold to).

Risk is ATR-based here (stop/target in multiples of ATR at the signal bar)
rather than opening-range-based, since these are generic indicator signals
without a natural range of their own.
"""

import pandas as pd

import indicators as ind
import market_rules as mr


def generate_trades(df: pd.DataFrame, signal: pd.Series, long_ok: pd.Series, short_ok: pd.Series,
                     stop_atr_mult: float = 1.5, target_atr_mult: float = 3.0,
                     warmup_bars: int = 30) -> pd.DataFrame:
    atr = ind.atr(df, 14)
    bar_interval = mr.infer_bar_interval(df)
    squareoff_trigger = mr.squareoff_trigger_time(bar_interval)
    trades = []

    for day, day_df in df.groupby(df.index.date):
        bars = list(day_df.itertuples())
        in_position = False
        direction = 0
        entry_price = entry_time = stop = target = None

        for i, bar in enumerate(bars):
            idx = bar.Index

            if in_position:
                if idx.time() >= squareoff_trigger:
                    trades.append({
                        "date": day, "direction": direction,
                        "entry_time": entry_time, "entry_price": entry_price,
                        "exit_time": idx, "exit_price": bar.open,
                        "exit_reason": "squareoff",
                    })
                    in_position = False
                    continue

                if direction == 1:
                    hit_stop, hit_target = bar.low <= stop, bar.high >= target
                else:
                    hit_stop, hit_target = bar.high >= stop, bar.low <= target
                if hit_stop or hit_target:
                    exit_price = stop if hit_stop else target
                    trades.append({
                        "date": day, "direction": direction,
                        "entry_time": entry_time, "entry_price": entry_price,
                        "exit_time": idx, "exit_price": exit_price,
                        "exit_reason": "stop" if hit_stop else "target",
                    })
                    in_position = False
                continue

            if idx not in signal.index or pd.isna(atr.loc[idx]) or atr.loc[idx] <= 0:
                continue
            sig = signal.loc[idx]
            if sig == 0:
                continue
            if sig == 1 and not bool(long_ok.loc[idx]):
                continue
            if sig == -1 and not bool(short_ok.loc[idx]):
                continue
            if i + 1 >= len(bars):
                continue  # no next bar today to enter on

            next_bar = bars[i + 1]
            if not mr.is_entry_allowed(next_bar.Index):
                continue
            direction = int(sig)
            entry_price = next_bar.open
            entry_time = next_bar.Index
            a = atr.loc[idx]
            if direction == 1:
                stop = entry_price - stop_atr_mult * a
                target = entry_price + target_atr_mult * a
            else:
                stop = entry_price + stop_atr_mult * a
                target = entry_price - target_atr_mult * a
            in_position = True

        if in_position:
            last_bar = bars[-1]
            trades.append({
                "date": day, "direction": direction,
                "entry_time": entry_time, "entry_price": entry_price,
                "exit_time": last_bar.Index, "exit_price": last_bar.close,
                "exit_reason": "eod_fallback",
            })

    return pd.DataFrame(trades)
