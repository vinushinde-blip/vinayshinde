"""
Turns a (trigger, filter, stop/target) combination into a trade list, using
the same output shape as strategies.py (date, direction, entry_time,
entry_price, exit_time, exit_price, exit_reason) so it plugs into the
existing engine.py/metrics.py pipeline unchanged.

Indicators are computed once per symbol on the whole continuous series (not
reset each day — same as any charting platform's intraday RSI/MACD/etc.),
but positions are still forced flat by each day's last bar: no carrying
intraday risk overnight, consistent with how MIS/intraday orders actually
work.

Risk is ATR-based here (stop/target in multiples of ATR at the signal bar)
rather than opening-range-based, since these are generic indicator signals
without a natural range of their own.
"""

import pandas as pd

import indicators as ind


def generate_trades(df: pd.DataFrame, signal: pd.Series, long_ok: pd.Series, short_ok: pd.Series,
                     stop_atr_mult: float = 1.5, target_atr_mult: float = 3.0,
                     warmup_bars: int = 30) -> pd.DataFrame:
    atr = ind.atr(df, 14)
    trades = []

    for day, day_df in df.groupby(df.index.date):
        bars = list(day_df.itertuples())
        in_position = False
        direction = 0
        entry_price = entry_time = stop = target = None

        for i, bar in enumerate(bars):
            idx = bar.Index

            if in_position:
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
                "exit_reason": "eod",
            })

    return pd.DataFrame(trades)
