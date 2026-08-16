"""
Fetches multi-year DAILY OHLCV history for the universe — a separate,
much smaller/faster fetch than kite_fetch_intraday.py's 5-minute pull.
Kite's day-interval history has no meaningful lookback limit (unlike
intraday intervals), so 5+ years fits in a single request per symbol.

Saves to data/daily/bars/ — deliberately separate from data/intraday/bars/
(5-minute candles) so the two don't collide or get mixed up by symbol name.

Usage:
    python3 kite_fetch_daily.py --years 5
"""

import argparse
import os

import pandas as pd

from kite_auth import get_kite
from kite_fetch_intraday import fetch_symbol_history

UNIVERSE_FILE = os.path.join(os.path.dirname(__file__), "..", "..", "data", "intraday", "universe_top500.csv")
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "daily", "bars")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--years", type=float, default=5.0)
    parser.add_argument("--limit", type=int, default=None, help="only fetch the first N symbols (for testing)")
    args = parser.parse_args()

    if not os.path.exists(UNIVERSE_FILE):
        raise SystemExit(f"Universe file not found: {UNIVERSE_FILE}")

    universe = pd.read_csv(UNIVERSE_FILE)
    if args.limit:
        universe = universe.head(args.limit)

    lookback_days = int(args.years * 365.25) + 30  # small buffer past the exact year count
    kite = get_kite()
    os.makedirs(OUT_DIR, exist_ok=True)

    for i, row in enumerate(universe.itertuples(), start=1):
        out_path = os.path.join(OUT_DIR, f"{row.tradingsymbol}.parquet")
        if os.path.exists(out_path):
            continue

        df = fetch_symbol_history(kite, row.instrument_token, "day", lookback_days)
        if df.empty:
            print(f"[{i}/{len(universe)}] {row.tradingsymbol}: no data, skipping")
            continue

        df.to_parquet(out_path)
        print(f"[{i}/{len(universe)}] {row.tradingsymbol}: {len(df)} bars saved "
              f"({df.index.min().date()} to {df.index.max().date()})")

    print(f"\nDone. Daily bar files in {OUT_DIR}")


if __name__ == "__main__":
    main()
