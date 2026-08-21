"""Fetch 5-minute intraday OHLC candles for the NSE500 universe.

Yahoo Finance caps 5m history at ~60 calendar days, so this always pulls the
full available window. Symbols are read from data/nse500_symbols.csv (see
scripts/fetch_nse500_symbols.py); if that file hasn't been generated yet,
falls back to the smaller data/nse500_symbols_sample.csv so the rest of the
pipeline (scripts/orb_vwap_strategy.py) has something to run against.
"""

import os
import time

import pandas as pd
import yfinance as yf

SYMBOLS_FILE = (
    "data/nse500_symbols.csv"
    if os.path.exists("data/nse500_symbols.csv")
    else "data/nse500_symbols_sample.csv"
)
OUTPUT_DIR = "data/intraday_5m"

INTERVAL = "5m"
PERIOD = "60d"  # Yahoo Finance's max lookback for 5m bars
BATCH_SIZE = 40  # keep batches small to avoid Yahoo Finance rate limits
SLEEP_BETWEEN_BATCHES = 5  # seconds
MAX_RETRIES = 3


def load_symbols():
    df = pd.read_csv(SYMBOLS_FILE)
    return [f"{s.strip().upper()}.NS" for s in df["Symbol"]]


def fetch_batch(tickers):
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return yf.download(
                tickers,
                period=PERIOD,
                interval=INTERVAL,
                group_by="ticker",
                auto_adjust=False,
                progress=False,
                threads=True,
            )
        except Exception as exc:
            wait = 2**attempt
            print(f"Batch fetch failed (attempt {attempt}/{MAX_RETRIES}): {exc}; retrying in {wait}s")
            time.sleep(wait)
    print(f"Giving up on batch: {tickers}")
    return None


def save_symbol(ticker, df):
    if df is None or df.empty:
        print(f"Warning: no 5m data for {ticker}")
        return

    df = df.dropna(how="all")
    if df.empty:
        return

    df.index = df.index.tz_convert("Asia/Kolkata")
    symbol = ticker.replace(".NS", "")
    out_path = os.path.join(OUTPUT_DIR, f"{symbol}.csv")
    df.to_csv(out_path)
    print(f"{symbol}: {len(df)} 5m candles saved -> {out_path}")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    tickers = load_symbols()
    print(f"Using symbol list: {SYMBOLS_FILE}")
    print(f"Fetching {INTERVAL} candles for {len(tickers)} symbols (period={PERIOD})")

    for i in range(0, len(tickers), BATCH_SIZE):
        batch = tickers[i : i + BATCH_SIZE]
        print(f"\nBatch {i // BATCH_SIZE + 1}: {batch}")
        data = fetch_batch(batch)
        if data is None:
            continue

        if len(batch) == 1:
            save_symbol(batch[0], data)
        else:
            for ticker in batch:
                if ticker in data.columns.get_level_values(0):
                    save_symbol(ticker, data[ticker])

        time.sleep(SLEEP_BETWEEN_BATCHES)


if __name__ == "__main__":
    main()
