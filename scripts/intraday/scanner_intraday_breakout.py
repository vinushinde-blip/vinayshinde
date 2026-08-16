"""
Intraday breakout scanner — surfaces stocks CURRENTLY showing the
consolidation + volume-confirmed-breakout setup, for a trader's own review.
Distinct from generate_signals.py: that applies the full execution cap
(concurrency/day limits) to decide what to actually TAKE; this is a wider
visibility scan — every candidate meeting the pattern, not just the capped
selection. Same underlying detector (momentum_volume_breakout) as the rest
of this pipeline, so it's the exact setup already backtested, not a new
untested pattern.

Two lists:
  - "Just broke out": a momentum_volume_breakout signal fired within the
    last --recent-bars bars (default 3, ~15 min on 5-min data) — the move
    described in steps 1-2 of the chart you shared.
  - "Coiling": currently in an unusually tight range relative to its own
    recent volatility (a rolling ATR squeeze) and sitting in the upper
    part of that range — a pre-breakout candidate, not yet triggered.

Safe to run mid-session: pass --as-of to only see bars up to that time.

Usage:
    python3 scanner_intraday_breakout.py
    python3 scanner_intraday_breakout.py --date 2026-08-14 --as-of 11:35
"""

import argparse
import glob
import os

import pandas as pd

import indicators as ind
from strategies import momentum_volume_breakout

BARS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "intraday", "bars")
UNIVERSE_FILE = os.path.join(os.path.dirname(__file__), "..", "..", "data", "intraday", "universe_top500.csv")


def load_bars_and_rank():
    universe = pd.read_csv(UNIVERSE_FILE)
    rank = dict(zip(universe["tradingsymbol"], universe["liquidity_rank"]))
    bars = {}
    for path in sorted(glob.glob(os.path.join(BARS_DIR, "*.parquet"))):
        symbol = os.path.splitext(os.path.basename(path))[0]
        bars[symbol] = pd.read_parquet(path)
    return bars, rank


def find_recent_breakouts(day_df: pd.DataFrame, recent_bars: int) -> pd.DataFrame:
    trades = momentum_volume_breakout(day_df)
    if trades.empty:
        return trades
    cutoff = day_df.index[-1] - (day_df.index[1] - day_df.index[0]) * recent_bars
    return trades[trades["entry_time"] >= cutoff]


def find_coiling(day_df: pd.DataFrame, squeeze_lookback: int = 20, squeeze_pctile: float = 0.25) -> dict:
    """Currently tight vs its own recent range, and sitting in the upper
    half of today's range (leaning toward an upside break, not a random
    quiet stock)."""
    if len(day_df) < squeeze_lookback + 5:
        return None
    atr = ind.atr(day_df, 14)
    recent_atr = atr.iloc[-squeeze_lookback:]
    current_atr = atr.iloc[-1]
    if pd.isna(current_atr) or recent_atr.isna().all():
        return None
    percentile_rank = (recent_atr < current_atr).mean()
    if percentile_rank > squeeze_pctile:
        return None  # not unusually tight

    today_high, today_low = day_df["high"].max(), day_df["low"].min()
    today_range = today_high - today_low
    if today_range <= 0:
        return None
    last_close = day_df["close"].iloc[-1]
    position_in_range = (last_close - today_low) / today_range
    if position_in_range < 0.5:
        return None  # sitting low in the range, not coiled toward a breakout

    return {"atr_percentile": round(percentile_rank * 100, 1),
            "position_in_range_pct": round(position_in_range * 100, 1),
            "last_close": last_close, "today_high": today_high}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default=None)
    parser.add_argument("--as-of", default=None)
    parser.add_argument("--recent-bars", type=int, default=3)
    args = parser.parse_args()

    bars, rank = load_bars_and_rank()
    if not bars:
        raise SystemExit(f"No bar data in {BARS_DIR}.")

    all_dates = sorted({d for df in bars.values() for d in df.index.date})
    target_date = pd.Timestamp(args.date).date() if args.date else all_dates[-1]
    as_of = pd.Timestamp(f"{target_date} {args.as_of}") if args.as_of else None

    breakouts, coiling = [], []
    for symbol, df in bars.items():
        day_df = df[df.index.date == target_date]
        if as_of is not None:
            day_df = day_df[day_df.index <= as_of]
        if len(day_df) < 6:
            continue

        recent = find_recent_breakouts(day_df, args.recent_bars)
        for row in recent.itertuples():
            breakouts.append({
                "symbol": symbol, "liquidity_rank": rank.get(symbol),
                "direction": "long" if row.direction == 1 else "short",
                "entry_time": row.entry_time, "entry_price": round(row.entry_price, 2),
            })

        coil = find_coiling(day_df)
        if coil:
            coil["symbol"] = symbol
            coil["liquidity_rank"] = rank.get(symbol)
            coiling.append(coil)

    print(f"Scan as of {target_date}" + (f" {args.as_of}" if args.as_of else " (full day)") + "\n")

    print(f"=== Just broke out (last {args.recent_bars} bars) — {len(breakouts)} ===")
    if breakouts:
        df_out = pd.DataFrame(breakouts).sort_values("liquidity_rank")
        print(df_out.to_string(index=False))
    else:
        print("(none)")

    print(f"\n=== Coiling — tight range, upper half, no trigger yet — {len(coiling)} ===")
    if coiling:
        df_out = pd.DataFrame(coiling).sort_values("liquidity_rank")
        cols = ["symbol", "liquidity_rank", "last_close", "today_high", "atr_percentile", "position_in_range_pct"]
        print(df_out[cols].to_string(index=False))
    else:
        print("(none)")


if __name__ == "__main__":
    main()
