"""
Build the NSE500 -> Kite instrument_token mapping needed to subscribe on
the WebSocket ticker. Run locally after scripts/kite_auth.py has created
.kite_session.json.

Output: data/nse500_instruments.csv (tradingsymbol, instrument_token, name)
"""
import os

import pandas as pd
import requests
from dotenv import load_dotenv
from kiteconnect import KiteConnect

from kite_auth import get_credentials, load_access_token

load_dotenv()

NSE500_LIST_URL = "https://nsearchives.nseindia.com/content/indices/ind_nifty500list.csv"
OUTPUT_FILE = "data/nse500_instruments.csv"


def fetch_nse500_symbols() -> list[str]:
    headers = {"User-Agent": "Mozilla/5.0"}
    resp = requests.get(NSE500_LIST_URL, headers=headers, timeout=30)
    resp.raise_for_status()
    with open("data/.nse500_raw.csv", "wb") as f:
        f.write(resp.content)
    df = pd.read_csv("data/.nse500_raw.csv")
    os.remove("data/.nse500_raw.csv")
    return df["Symbol"].str.strip().tolist()


def build_mapping():
    api_key, _ = get_credentials()
    kite = KiteConnect(api_key=api_key)
    kite.set_access_token(load_access_token())

    symbols = set(fetch_nse500_symbols())
    print(f"NSE500 list: {len(symbols)} symbols")

    instruments = pd.DataFrame(kite.instruments("NSE"))
    nse500 = instruments[
        (instruments["tradingsymbol"].isin(symbols)) & (instruments["segment"] == "NSE")
    ][["tradingsymbol", "instrument_token", "name"]]

    missing = symbols - set(nse500["tradingsymbol"])
    if missing:
        print(f"Warning: {len(missing)} symbols not matched: {sorted(missing)}")

    os.makedirs("data", exist_ok=True)
    nse500.to_csv(OUTPUT_FILE, index=False)
    print(f"Saved {len(nse500)} instruments to {OUTPUT_FILE}")


if __name__ == "__main__":
    build_mapping()
