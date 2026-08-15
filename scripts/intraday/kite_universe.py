"""
Builds the "top 500 most liquid NSE stocks" universe directly from Kite data
(no NSE website dependency): pulls all NSE equity instruments, fetches
recent daily volume for each, ranks by average daily turnover (price *
volume), and keeps the top 500.

This makes liquidity the actual selection criterion, rather than index
membership (Nifty 500) which is cap-weighted and reconstituted only twice
a year.

Rate-limited to stay well under Kite's API limits; checkpoints progress to
CSV so a failed run can resume instead of refetching everything.

Usage:
    python3 kite_universe.py --lookback-days 60 --top-n 500
"""

import argparse
import os
import time
from datetime import datetime, timedelta

import pandas as pd

from kite_auth import get_kite

CHECKPOINT_FILE = os.path.join(os.path.dirname(__file__), "_universe_checkpoint.csv")
OUTPUT_FILE = os.path.join(os.path.dirname(__file__), "..", "..", "data", "intraday", "universe_top500.csv")

REQUEST_DELAY_SEC = 0.35   # ~2.8 req/sec, under Kite's ~3 req/sec ceiling
MAX_RETRIES = 4


def fetch_nse_equities(kite) -> pd.DataFrame:
    """NSE's Kite instrument dump tags SDLs/G-secs, SME-board stocks, NCDs,
    and REIT/InvIT units all as instrument_type=="EQ" too — they're not the
    liquid mainboard stocks we want. Real NSE cash-equity stocks always
    trade in market lot size 1 (SDLs/SME/NCDs/trusts don't), so filtering on
    that is a much more accurate mainboard-equity filter than instrument_type
    alone. Any oddity that slips through still gets pushed out of the top
    500 by the turnover ranking itself.
    """
    instruments = pd.DataFrame(kite.instruments("NSE"))
    equities = instruments[
        (instruments["instrument_type"] == "EQ") &
        (instruments["segment"] == "NSE") &
        (instruments["lot_size"] == 1)
    ].copy()
    return equities[["instrument_token", "tradingsymbol", "name"]]


def _fetch_with_retry(kite, token, from_date, to_date):
    for attempt in range(MAX_RETRIES):
        try:
            return kite.historical_data(token, from_date, to_date, interval="day")
        except Exception as e:
            wait = 2 ** attempt
            print(f"  retry {attempt + 1}/{MAX_RETRIES} for token {token} after error: {e} (sleep {wait}s)")
            time.sleep(wait)
    return []


def compute_avg_turnover(kite, equities: pd.DataFrame, lookback_days: int) -> pd.DataFrame:
    done = {}
    if os.path.exists(CHECKPOINT_FILE):
        existing = pd.read_csv(CHECKPOINT_FILE)
        done = dict(zip(existing["tradingsymbol"], existing["avg_turnover"]))
        print(f"Resuming: {len(done)} symbols already have turnover computed.")

    to_date = datetime.today()
    from_date = to_date - timedelta(days=int(lookback_days * 1.6))  # pad for weekends/holidays

    rows = list(done.items())
    remaining = equities[~equities["tradingsymbol"].isin(done.keys())]
    print(f"Fetching daily volume for {len(remaining)} remaining symbols...")

    for i, r in enumerate(remaining.itertuples(), start=1):
        candles = _fetch_with_retry(kite, r.instrument_token, from_date, to_date)
        if candles:
            df = pd.DataFrame(candles).tail(lookback_days)
            avg_turnover = (df["close"] * df["volume"]).mean()
        else:
            avg_turnover = 0.0

        rows.append((r.tradingsymbol, avg_turnover))
        time.sleep(REQUEST_DELAY_SEC)

        if i % 50 == 0:
            pd.DataFrame(rows, columns=["tradingsymbol", "avg_turnover"]).to_csv(CHECKPOINT_FILE, index=False)
            print(f"  progress: {i}/{len(remaining)} (checkpoint saved)")

    result = pd.DataFrame(rows, columns=["tradingsymbol", "avg_turnover"])
    result.to_csv(CHECKPOINT_FILE, index=False)
    return result.merge(equities, on="tradingsymbol", how="left")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lookback-days", type=int, default=60,
                         help="trading days of volume history to average for the liquidity ranking")
    parser.add_argument("--top-n", type=int, default=500)
    args = parser.parse_args()

    kite = get_kite()
    equities = fetch_nse_equities(kite)
    print(f"Found {len(equities)} NSE equity instruments.")

    ranked = compute_avg_turnover(kite, equities, args.lookback_days)
    ranked = ranked.sort_values("avg_turnover", ascending=False).reset_index(drop=True)
    ranked["liquidity_rank"] = ranked.index + 1

    top = ranked.head(args.top_n)
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    top.to_csv(OUTPUT_FILE, index=False)
    print(f"Saved top {len(top)} liquid NSE stocks to {OUTPUT_FILE}")
    print(top[["liquidity_rank", "tradingsymbol", "name", "avg_turnover"]].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
