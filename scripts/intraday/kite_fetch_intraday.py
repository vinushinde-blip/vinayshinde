"""
Fetches historical intraday OHLCV candles for every symbol in the liquidity
universe (from kite_universe.py) and saves one Parquet file per symbol.

Kite's historical API caps how many days you can request per call, and how
far back data is available at all, and this varies by candle interval.
These are Kite's documented limits as of when this was written — CONFIRM
against https://kite.trade/docs/connect/v3/historical-data/ before a large
run, they do change:
    minute:            60 days total
    3minute/5minute/10minute/15minute/30minute/60minute: ~100-400 days
    day:               no practical limit

Because of this, "rigorous backtesting" on 500 stocks of pure 1-minute data
is not viable via this API (only 60 days of history) — 15-minute bars are
the default here as a reasonable trade-off between granularity and having
enough calendar time to backtest across different market regimes.

Usage:
    python3 kite_fetch_intraday.py --interval 15minute --lookback-days 200
"""

import argparse
import os
import time
from datetime import datetime, timedelta

import pandas as pd

from kite_auth import get_kite

UNIVERSE_FILE = os.path.join(os.path.dirname(__file__), "..", "..", "data", "intraday", "universe_top500.csv")
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "intraday", "bars")

REQUEST_DELAY_SEC = 0.35
MAX_RETRIES = 4

# Max days per single API request, by interval (Kite-documented ceiling).
MAX_DAYS_PER_REQUEST = {
    "minute": 60,
    "3minute": 100, "5minute": 100, "10minute": 100,
    "15minute": 200, "30minute": 200,
    "60minute": 400,
    "day": 2000,
}


def _fetch_with_retry(kite, token, from_date, to_date, interval):
    for attempt in range(MAX_RETRIES):
        try:
            return kite.historical_data(token, from_date, to_date, interval=interval)
        except Exception as e:
            wait = 2 ** attempt
            print(f"  retry {attempt + 1}/{MAX_RETRIES} for token {token} after error: {e} (sleep {wait}s)")
            time.sleep(wait)
    return []


def fetch_symbol_history(kite, token, interval, total_lookback_days):
    chunk_days = MAX_DAYS_PER_REQUEST.get(interval, 60)
    to_date = datetime.today()
    earliest = to_date - timedelta(days=total_lookback_days)

    all_candles = []
    window_end = to_date
    while window_end > earliest:
        window_start = max(window_end - timedelta(days=chunk_days), earliest)
        candles = _fetch_with_retry(kite, token, window_start, window_end, interval)
        if not candles:
            break
        all_candles = candles + all_candles
        window_end = window_start - timedelta(days=1)
        time.sleep(REQUEST_DELAY_SEC)

    if not all_candles:
        return pd.DataFrame()

    df = pd.DataFrame(all_candles).drop_duplicates(subset="date").sort_values("date")
    df = df.rename(columns={"date": "datetime"}).set_index("datetime")
    df.index = pd.to_datetime(df.index).tz_localize(None)
    return df[["open", "high", "low", "close", "volume"]]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--interval", default="15minute", choices=list(MAX_DAYS_PER_REQUEST))
    parser.add_argument("--lookback-days", type=int, default=200,
                         help="total calendar days of history to fetch (capped by what Kite actually retains)")
    parser.add_argument("--limit", type=int, default=None, help="only fetch the first N symbols (for testing)")
    args = parser.parse_args()

    if not os.path.exists(UNIVERSE_FILE):
        raise SystemExit(f"Universe file not found: {UNIVERSE_FILE}\nRun kite_universe.py first.")

    universe = pd.read_csv(UNIVERSE_FILE)
    if args.limit:
        universe = universe.head(args.limit)

    kite = get_kite()
    os.makedirs(OUT_DIR, exist_ok=True)

    for i, row in enumerate(universe.itertuples(), start=1):
        out_path = os.path.join(OUT_DIR, f"{row.tradingsymbol}.parquet")
        if os.path.exists(out_path):
            continue

        df = fetch_symbol_history(kite, row.instrument_token, args.interval, args.lookback_days)
        if df.empty:
            print(f"[{i}/{len(universe)}] {row.tradingsymbol}: no data, skipping")
            continue

        df.to_parquet(out_path)
        print(f"[{i}/{len(universe)}] {row.tradingsymbol}: {len(df)} bars saved "
              f"({df.index.min()} to {df.index.max()})")

    print(f"\nDone. Bar files in {OUT_DIR}")


if __name__ == "__main__":
    main()
