"""
Fetches multi-year daily OHLCV for the NIFTY 50 index (instrument_token
256265, INDICES segment) — the benchmark used for relative-strength and
market-regime filters in swing_backtest.py. Saved separately from the
per-symbol universe fetch since it's a single instrument.

Usage:
    python3 kite_fetch_nifty_index.py --years 10
"""

import argparse
import os

from kite_auth import get_kite
from kite_fetch_intraday import fetch_symbol_history

NIFTY50_TOKEN = 256265
OUT_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "data", "daily", "NIFTY50.parquet")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--years", type=float, default=10.0)
    args = parser.parse_args()

    lookback_days = int(args.years * 365.25) + 30
    kite = get_kite()
    df = fetch_symbol_history(kite, NIFTY50_TOKEN, "day", lookback_days)
    if df.empty:
        raise SystemExit("No data returned for NIFTY 50 index.")

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    df.to_parquet(OUT_PATH)
    print(f"Saved {len(df)} daily bars, {df.index.min().date()} to {df.index.max().date()} -> {OUT_PATH}")


if __name__ == "__main__":
    main()
