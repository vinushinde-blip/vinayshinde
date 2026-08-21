"""
Refresh just the daily OHLC bars for the NSE500 - the fast, cheap piece
of fetch_nse500_multiframe_ohlc.py's job, meant to run once per day
after market close so scanner3 always sees today's close.

One API call per symbol (day interval's chunk limit covers the whole
2-year window in one request), so ~500 calls total - a few minutes at
Kite's historical-data rate limit, unlike the full multi-timeframe
backfill which takes ~40-50 minutes.

Setup / Run: same env vars as the other scripts (KITE_API_KEY,
KITE_ACCESS_TOKEN, optionally KITE_WATCHLIST).

  python scripts/update_daily_bars.py
"""

import logging
import os
import sys

from kiteconnect import KiteConnect

sys.path.insert(0, os.path.dirname(__file__))
from fetch_nse500_multiframe_ohlc import fetch_interval, save_csv  # noqa: E402
from kite_scanner2 import fetch_nse500_symbols  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("update_daily_bars")


def main():
    api_key = os.environ.get("KITE_API_KEY")
    access_token = os.environ.get("KITE_ACCESS_TOKEN")
    if not api_key or not access_token:
        log.error("Set KITE_API_KEY and KITE_ACCESS_TOKEN environment variables.")
        sys.exit(1)

    kite = KiteConnect(api_key=api_key)
    kite.set_access_token(access_token)

    watchlist_raw = os.environ.get("KITE_WATCHLIST", "NSE500").strip()
    watchlist = fetch_nse500_symbols() if watchlist_raw.upper() == "NSE500" else \
        [s.strip().upper() for s in watchlist_raw.split(",") if s.strip()]

    instruments = kite.instruments("NSE")
    by_symbol = {i["tradingsymbol"]: i["instrument_token"] for i in instruments}

    updated = 0
    for i, symbol in enumerate(watchlist, 1):
        token = by_symbol.get(symbol)
        if token is None:
            continue
        df = fetch_interval(kite, token, "day", 2000)
        save_csv(df, "day", symbol)
        updated += 1
        if i % 50 == 0:
            log.info("[%d/%d] updated", i, len(watchlist))

    log.info("Done. %d/%d symbols' daily bars refreshed.", updated, len(watchlist))


if __name__ == "__main__":
    main()
