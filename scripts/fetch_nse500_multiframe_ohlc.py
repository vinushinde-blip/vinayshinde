"""
Bulk historical OHLC downloader for the NSE500 - weekly, daily, 2-hour,
1-hour, 15-min and 5-min candles, 2 years back, via Kite's historical
data API.

This is a long-running batch job (roughly 40-50 minutes for the full
NSE500 x all timeframes, dominated by Kite's historical-data rate limit
and the per-interval date-range chunk limits) - run it with nohup on a
box that stays up, not interactively in a foreground shell.

Kite's native intervals only go up to "day"; there's no native 2-hour or
weekly candle. Those two are derived locally by resampling the fetched
60-minute and daily data after download (2-hour bins are aligned to each
day's session start so they read 09:15-11:15, 11:15-13:15, ... rather
than arbitrary clock-hour boundaries).

Setup
-----
  pip install kiteconnect pandas

  export KITE_API_KEY=...
  export KITE_ACCESS_TOKEN=...
  export KITE_WATCHLIST=NSE500      # default; or a comma list of symbols

Run
---
  python scripts/fetch_nse500_multiframe_ohlc.py

Output
------
  data/ohlc/<interval>/<SYMBOL>.csv   - date,open,high,low,close,volume
  data/ohlc/.checkpoint.json          - which (symbol, interval) pairs are done

Resumable: re-running after an interruption (Ctrl-C, SSH drop, crash)
skips any (symbol, interval) pair already recorded in the checkpoint
file instead of re-fetching it.
"""

import csv
import json
import logging
import os
import sys
import time
from datetime import datetime, timedelta

import pandas as pd
from kiteconnect import KiteConnect

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("ohlc_fetch")

YEARS_BACK = 2
RATE_LIMIT_SLEEP = 0.35  # ~3 req/sec, Kite's historical-data limit
MAX_RETRIES = 5

# Kite's max days-per-request per interval (conservative, documented limits).
FETCH_INTERVALS = {
    "day": 2000,
    "60minute": 400,
    "15minute": 200,
    "5minute": 100,
}

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "ohlc")
CHECKPOINT_FILE = os.path.join(DATA_DIR, ".checkpoint.json")

AGG = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}


def load_checkpoint() -> set:
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE) as f:
            return set(tuple(x) for x in json.load(f))
    return set()


def save_checkpoint(done: set) -> None:
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump(sorted(list(done)), f)


def date_chunks(start: datetime, end: datetime, days: int):
    cur = start
    while cur < end:
        nxt = min(cur + timedelta(days=days), end)
        yield cur, nxt
        cur = nxt


def fetch_interval(kite: KiteConnect, token: int, interval: str, days_per_chunk: int) -> pd.DataFrame:
    start = datetime.now() - timedelta(days=365 * YEARS_BACK)
    end = datetime.now()
    rows = []
    for chunk_start, chunk_end in date_chunks(start, end, days_per_chunk):
        for attempt in range(MAX_RETRIES):
            try:
                rows.extend(kite.historical_data(token, chunk_start, chunk_end, interval))
                break
            except Exception as e:
                wait = 2 ** attempt
                log.warning("historical_data retry %d for token=%s interval=%s: %s", attempt + 1, token, interval, e)
                time.sleep(wait)
        else:
            log.error("Giving up on token=%s interval=%s chunk %s->%s", token, interval, chunk_start, chunk_end)
        time.sleep(RATE_LIMIT_SLEEP)

    if not rows:
        return pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume"])
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    df = df.drop_duplicates(subset="date").sort_values("date")
    return df[["date", "open", "high", "low", "close", "volume"]]


def resample_2hour(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    indexed = df.set_index("date")
    parts = []
    for _, group in indexed.groupby(indexed.index.date):
        origin = group.index[0]
        parts.append(group.resample("2h", origin=origin).agg(AGG).dropna())
    return pd.concat(parts).reset_index()


def resample_week(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    return df.set_index("date").resample("W").agg(AGG).dropna().reset_index()


def save_csv(df: pd.DataFrame, interval: str, symbol: str) -> None:
    out_dir = os.path.join(DATA_DIR, interval)
    os.makedirs(out_dir, exist_ok=True)
    df.to_csv(os.path.join(out_dir, f"{symbol}.csv"), index=False)


def fetch_nse500_symbols() -> list:
    from kite_scanner2 import fetch_nse500_symbols as _fetch
    return _fetch()


def main():
    sys.path.insert(0, os.path.dirname(__file__))

    api_key = os.environ.get("KITE_API_KEY")
    access_token = os.environ.get("KITE_ACCESS_TOKEN")
    watchlist_raw = os.environ.get("KITE_WATCHLIST", "NSE500").strip()

    if not api_key or not access_token:
        log.error("Set KITE_API_KEY and KITE_ACCESS_TOKEN environment variables.")
        sys.exit(1)

    kite = KiteConnect(api_key=api_key)
    kite.set_access_token(access_token)

    watchlist = fetch_nse500_symbols() if watchlist_raw.upper() == "NSE500" else \
        [s.strip().upper() for s in watchlist_raw.split(",") if s.strip()]

    instruments = kite.instruments("NSE")
    by_symbol = {i["tradingsymbol"]: i["instrument_token"] for i in instruments}

    done = load_checkpoint()
    total_pairs = len(watchlist) * len(FETCH_INTERVALS)
    log.info("Starting: %d symbols x %d fetched intervals (+2 derived) = %d pairs, %d already done.",
              len(watchlist), len(FETCH_INTERVALS), total_pairs, len(done))

    for i, symbol in enumerate(watchlist, 1):
        token = by_symbol.get(symbol)
        if token is None:
            log.warning("[%d/%d] %s: not found on NSE, skipping.", i, len(watchlist), symbol)
            continue

        daily_df = None
        hourly_df = None

        for interval, days_per_chunk in FETCH_INTERVALS.items():
            key = f"{symbol}:{interval}"
            if (symbol, interval) in done:
                if interval == "day":
                    daily_df = pd.read_csv(os.path.join(DATA_DIR, "day", f"{symbol}.csv"), parse_dates=["date"])
                if interval == "60minute":
                    hourly_df = pd.read_csv(os.path.join(DATA_DIR, "60minute", f"{symbol}.csv"), parse_dates=["date"])
                continue

            df = fetch_interval(kite, token, interval, days_per_chunk)
            save_csv(df, interval, symbol)
            done.add((symbol, interval))
            save_checkpoint(done)
            log.info("[%d/%d] %s %s: %d candles", i, len(watchlist), symbol, interval, len(df))

            if interval == "day":
                daily_df = df
            if interval == "60minute":
                hourly_df = df

        if (symbol, "week") not in done and daily_df is not None:
            save_csv(resample_week(daily_df), "week", symbol)
            done.add((symbol, "week"))
            save_checkpoint(done)

        if (symbol, "2hour") not in done and hourly_df is not None:
            save_csv(resample_2hour(hourly_df), "2hour", symbol)
            done.add((symbol, "2hour"))
            save_checkpoint(done)

    log.info("Done. %d/%d pairs complete.", len(done), total_pairs)


if __name__ == "__main__":
    main()
