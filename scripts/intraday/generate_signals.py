"""
Generates today's (or a given day's) actionable trading signals from the
selected strategy — the final, selective, execution-capped output meant to
actually be traded, not a research artifact.

Safe to run intraday, mid-session: pass --as-of "HH:MM" and it only sees
bars up to that time, exactly like a live polling system would. Re-run it
every few minutes during market hours (after refreshing bar data via
kite_fetch_intraday.py) and it returns the up-to-the-moment tradable
signal list — never re-deriving something it "shouldn't" have known yet.

Usage:
    python3 generate_signals.py                        # latest full day in cached data
    python3 generate_signals.py --date 2026-08-14       # a specific day
    python3 generate_signals.py --date 2026-08-14 --as-of 11:30   # simulate mid-session
"""

import argparse
import glob
import os

import pandas as pd

from selected_strategy import generate_signals, UNIVERSE_TOP_N, MAX_CONCURRENT_POSITIONS, \
    MAX_TRADES_PER_SYMBOL_PER_DAY, MAX_TRADES_PER_DAY

BARS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "intraday", "bars")
UNIVERSE_FILE = os.path.join(os.path.dirname(__file__), "..", "..", "data", "intraday", "universe_top500.csv")
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "intraday", "signals")


def load_bars_and_rank():
    universe = pd.read_csv(UNIVERSE_FILE)
    rank = dict(zip(universe["tradingsymbol"], universe["liquidity_rank"]))
    bars = {}
    for path in sorted(glob.glob(os.path.join(BARS_DIR, "*.parquet"))):
        symbol = os.path.splitext(os.path.basename(path))[0]
        bars[symbol] = pd.read_parquet(path)
    return bars, rank


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default=None, help="YYYY-MM-DD; default: latest date in cached data")
    parser.add_argument("--as-of", default=None, help="HH:MM IST — only see bars up to this time (simulates mid-session)")
    args = parser.parse_args()

    bars, rank = load_bars_and_rank()
    if not bars:
        raise SystemExit(f"No bar data in {BARS_DIR}. Run kite_fetch_intraday.py first (needs a fresh Kite login).")

    all_dates = sorted({d for df in bars.values() for d in df.index.date})
    target_date = pd.Timestamp(args.date).date() if args.date else all_dates[-1]
    if target_date not in all_dates:
        raise SystemExit(f"{target_date} not found in cached data. Available range: {all_dates[0]} to {all_dates[-1]}")

    day_bars = {sym: df[df.index.date == target_date] for sym, df in bars.items()}
    day_bars = {sym: df for sym, df in day_bars.items() if not df.empty}

    as_of = None
    if args.as_of:
        as_of = pd.Timestamp(f"{target_date} {args.as_of}")

    print(f"Generating signals for {target_date}" + (f" as of {args.as_of}" if args.as_of else " (full day)"))
    print(f"Constraints: top-{UNIVERSE_TOP_N} liquid universe, max {MAX_CONCURRENT_POSITIONS} concurrent positions, "
          f"max {MAX_TRADES_PER_SYMBOL_PER_DAY}/stock/day, max {MAX_TRADES_PER_DAY}/day overall.\n")

    signals = generate_signals(day_bars, rank, as_of=as_of)

    if signals.empty:
        print("No tradable signals today under the current constraints.")
        return

    display = signals.copy()
    display["net_return_pct"] = (display["net_return"] * 100).round(3)
    cols = ["symbol", "entry_time", "entry_price", "exit_time", "exit_price", "exit_reason", "net_return_pct"]
    print(f"{len(signals)} tradable signal(s):\n")
    print(display[cols].to_string(index=False))

    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, f"signals_{target_date}.csv")
    signals.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
