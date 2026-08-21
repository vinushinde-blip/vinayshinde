"""Fetch the current NSE500 constituent list.

NSE frequently blocks scripted/datacenter requests to archives.nseindia.com,
so this is best-effort: it warms up a session against the NSE homepage first
(NSE requires the cookies set there before the archive endpoint responds),
then downloads the list. If NSE blocks the request, fall back to manually
downloading the list from niftyindices.com and saving it as
data/nse500_symbols.csv with 'Symbol' and 'Company' columns -
scripts/fetch_intraday_5m.py falls back to data/nse500_symbols_sample.csv
in the meantime.
"""

import io
import os
import sys

import pandas as pd
import requests

NSE_LIST_URL = "https://archives.nseindia.com/content/indices/ind_nifty500list.csv"
OUTPUT_FILE = "data/nse500_symbols.csv"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    ),
    "Accept": "text/csv,application/vnd.ms-excel,*/*",
    "Referer": "https://www.nseindia.com/",
}


def fetch_nse500():
    session = requests.Session()
    session.headers.update(HEADERS)
    session.get("https://www.nseindia.com", timeout=10)
    resp = session.get(NSE_LIST_URL, timeout=10)
    resp.raise_for_status()

    df = pd.read_csv(io.StringIO(resp.text))
    df = df.rename(columns={"Symbol": "Symbol", "Company Name": "Company"})
    return df[["Symbol", "Company"]]


def main():
    try:
        df = fetch_nse500()
    except Exception as exc:
        print(
            f"Could not fetch the NSE500 list automatically ({exc}).\n"
            "NSE often blocks datacenter/CI IPs from archives.nseindia.com.\n"
            "Work around it by downloading the list manually from "
            "https://www.niftyindices.com/indices/equity/broad-based-indices/nifty500 "
            f"and saving it as {OUTPUT_FILE} with 'Symbol' and 'Company' columns."
        )
        sys.exit(1)

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"Saved {len(df)} NSE500 symbols -> {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
