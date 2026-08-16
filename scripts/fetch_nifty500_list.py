"""
Fetch the current Nifty 500 constituent list from NSE and save it to
data/nifty500_list.csv for use by scripts/pattern_scanner.py.

NSE's archives require a "browser-like" session (a plain GET is often
rejected), so we first hit the homepage to pick up cookies, then request
the CSV. Run this periodically (e.g. via the scan-nifty500 workflow) to
keep the constituent list current, since Nifty 500 membership changes a
few times a year.
"""
import os
import sys

import pandas as pd
import requests

NSE_HOME_URL = "https://www.nseindia.com"
NIFTY500_CSV_URL = "https://archives.nseindia.com/content/indices/ind_nifty500list.csv"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Accept": "text/csv,application/csv,text/plain,*/*",
}

OUTPUT_FILE = "data/nifty500_list.csv"


def fetch_nifty500() -> pd.DataFrame:
    session = requests.Session()
    session.headers.update(HEADERS)

    # Warm up the session so NSE hands out the cookies it expects.
    session.get(NSE_HOME_URL, timeout=15)

    resp = session.get(NIFTY500_CSV_URL, timeout=15)
    resp.raise_for_status()

    from io import StringIO

    df = pd.read_csv(StringIO(resp.text))
    df = df.rename(columns={c: c.strip() for c in df.columns})
    df = df.rename(columns={"Company Name": "Company Name", "Symbol": "Symbol"})

    if "Symbol" not in df.columns:
        raise ValueError(f"Unexpected CSV format, columns were: {list(df.columns)}")

    df["YF_Symbol"] = df["Symbol"].str.strip() + ".NS"
    return df


def main() -> None:
    os.makedirs("data", exist_ok=True)
    try:
        df = fetch_nifty500()
    except Exception as exc:  # noqa: BLE001 - report and exit non-zero for CI visibility
        print(f"ERROR: could not fetch Nifty 500 list from NSE: {exc}", file=sys.stderr)
        print(
            "If NSE is blocking automated requests from this network, "
            "download the list manually from https://www.niftyindices.com "
            f"and save it as {OUTPUT_FILE} with at least a 'Symbol' column.",
            file=sys.stderr,
        )
        sys.exit(1)

    df.to_csv(OUTPUT_FILE, index=False)
    print(f"Saved {len(df)} Nifty 500 constituents to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
