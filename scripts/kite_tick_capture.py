"""
Capture live NSE500 ticks (FULL mode: LTP, volume, OHLC, 5-level depth) via
KiteTicker and append them to daily Parquet files under data/ticks/.

Kite does not sell historical tick data — this script is how you build your
own tick history going forward. Run it during market hours (9:15-15:30 IST);
each day's ticks land in data/ticks/YYYY-MM-DD.parquet.

Usage:
  python scripts/kite_tick_capture.py
"""
import os
import threading
import time
from datetime import datetime

import pandas as pd
from dotenv import load_dotenv
from kiteconnect import KiteTicker

from kite_auth import get_credentials, load_access_token

load_dotenv()

INSTRUMENTS_FILE = "data/nse500_instruments.csv"
OUTPUT_DIR = "data/ticks"
FLUSH_INTERVAL_SEC = 30

_buffer = []
_buffer_lock = threading.Lock()


def load_instrument_tokens() -> list[int]:
    df = pd.read_csv(INSTRUMENTS_FILE)
    return df["instrument_token"].astype(int).tolist()


def on_ticks(ws, ticks):
    with _buffer_lock:
        for t in ticks:
            row = {
                "instrument_token": t.get("instrument_token"),
                "timestamp": t.get("last_trade_time") or t.get("exchange_timestamp"),
                "last_price": t.get("last_price"),
                "last_traded_quantity": t.get("last_traded_quantity"),
                "volume_traded": t.get("volume_traded"),
                "total_buy_quantity": t.get("total_buy_quantity"),
                "total_sell_quantity": t.get("total_sell_quantity"),
                "ohlc_open": t.get("ohlc", {}).get("open"),
                "ohlc_high": t.get("ohlc", {}).get("high"),
                "ohlc_low": t.get("ohlc", {}).get("low"),
                "ohlc_close": t.get("ohlc", {}).get("close"),
            }
            depth = t.get("depth", {})
            buy = depth.get("buy", [{}])
            sell = depth.get("sell", [{}])
            row["best_bid_price"] = buy[0].get("price") if buy else None
            row["best_bid_qty"] = buy[0].get("quantity") if buy else None
            row["best_ask_price"] = sell[0].get("price") if sell else None
            row["best_ask_qty"] = sell[0].get("quantity") if sell else None
            _buffer.append(row)


def on_connect(ws, response):
    tokens = load_instrument_tokens()
    print(f"Connected. Subscribing to {len(tokens)} instruments in FULL mode.")
    ws.subscribe(tokens)
    ws.set_mode(ws.MODE_FULL, tokens)


def on_close(ws, code, reason):
    print(f"Connection closed: {code} {reason}")


def flush_loop():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    while True:
        time.sleep(FLUSH_INTERVAL_SEC)
        with _buffer_lock:
            if not _buffer:
                continue
            df = pd.DataFrame(_buffer)
            _buffer.clear()
        today = datetime.now().strftime("%Y-%m-%d")
        path = os.path.join(OUTPUT_DIR, f"{today}.parquet")
        if os.path.exists(path):
            existing = pd.read_parquet(path)
            df = pd.concat([existing, df], ignore_index=True)
        df.to_parquet(path, index=False)
        print(f"Flushed {len(df)} rows to {path}")


def main():
    api_key, _ = get_credentials()
    access_token = load_access_token()

    kws = KiteTicker(api_key, access_token)
    kws.on_ticks = on_ticks
    kws.on_connect = on_connect
    kws.on_close = on_close

    threading.Thread(target=flush_loop, daemon=True).start()
    kws.connect()


if __name__ == "__main__":
    main()
