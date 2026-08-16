"""
Historical daily OHLCV data provider backed by Zerodha's Kite Connect API -
an alternative to the free yfinance/Yahoo Finance source pattern_scanner.py
uses by default.

Requires:
  - KITE_API_KEY       (from developers.kite.trade)
  - KITE_ACCESS_TOKEN  (generated daily - see scripts/kite_auth.py)

Kite identifies instruments by numeric instrument_token rather than trading
symbol, so this module downloads and locally caches NSE's instrument dump
the first time it's needed. It also rate-limits requests to stay under
Kite's ~3 requests/second API limit, since the kiteconnect client itself
does not throttle.
"""
import os
import threading
import time
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd

INSTRUMENTS_CACHE = "data/kite_instruments_NSE.csv"
INSTRUMENTS_CACHE_MAX_AGE_HOURS = 24
MAX_CHUNK_DAYS = 350  # keeps each historical_data() call comfortably within Kite's per-request range limit


class RateLimiter:
    """Thread-safe pacing so concurrent scanner workers still respect Kite's rate limit."""

    def __init__(self, calls_per_second: float = 2.5):
        self._min_interval = 1.0 / calls_per_second
        self._lock = threading.Lock()
        self._last_call = 0.0

    def wait(self) -> None:
        with self._lock:
            now = time.monotonic()
            elapsed = now - self._last_call
            if elapsed < self._min_interval:
                time.sleep(self._min_interval - elapsed)
            self._last_call = time.monotonic()


_rate_limiter = RateLimiter()
_kite_client = None
_client_lock = threading.Lock()
_instrument_map: Optional[dict] = None
_instrument_lock = threading.Lock()


def get_client():
    global _kite_client
    with _client_lock:
        if _kite_client is None:
            from kiteconnect import KiteConnect

            api_key = os.environ.get("KITE_API_KEY")
            access_token = os.environ.get("KITE_ACCESS_TOKEN")
            if not api_key or not access_token:
                raise RuntimeError(
                    "KITE_API_KEY and KITE_ACCESS_TOKEN must be set. "
                    "Run scripts/kite_auth.py to generate a fresh access token."
                )
            kite = KiteConnect(api_key=api_key)
            kite.set_access_token(access_token)
            _kite_client = kite
        return _kite_client


def _load_instrument_map() -> dict:
    global _instrument_map
    with _instrument_lock:
        if _instrument_map is not None:
            return _instrument_map

        need_refresh = True
        if os.path.exists(INSTRUMENTS_CACHE):
            age_hours = (time.time() - os.path.getmtime(INSTRUMENTS_CACHE)) / 3600
            need_refresh = age_hours > INSTRUMENTS_CACHE_MAX_AGE_HOURS

        if need_refresh:
            _rate_limiter.wait()
            kite = get_client()
            instruments = kite.instruments("NSE")
            df = pd.DataFrame(instruments)
            os.makedirs(os.path.dirname(INSTRUMENTS_CACHE), exist_ok=True)
            df.to_csv(INSTRUMENTS_CACHE, index=False)
        else:
            df = pd.read_csv(INSTRUMENTS_CACHE)

        # Plain equities only - excludes options/futures/rights entries that
        # can share a trading symbol root with the underlying stock.
        eq = df[(df["segment"] == "NSE") & (df["instrument_type"] == "EQ")]
        _instrument_map = dict(zip(eq["tradingsymbol"], eq["instrument_token"]))
        return _instrument_map


def get_instrument_token(nse_symbol: str) -> Optional[int]:
    return _load_instrument_map().get(nse_symbol)


def download_history(nse_symbol: str, period: str = "2y") -> Optional[pd.DataFrame]:
    """Fetch daily OHLCV for a plain NSE trading symbol, e.g. "RELIANCE" (no ".NS" suffix)."""
    token = get_instrument_token(nse_symbol)
    if token is None:
        return None

    years = int(period[:-1]) if period.endswith("y") else 2
    to_date = datetime.today()
    from_date = to_date - timedelta(days=365 * years + 30)

    kite = get_client()
    candles = []
    chunk_start = from_date
    while chunk_start < to_date:
        chunk_end = min(chunk_start + timedelta(days=MAX_CHUNK_DAYS), to_date)
        _rate_limiter.wait()
        candles.extend(kite.historical_data(token, chunk_start, chunk_end, interval="day"))
        chunk_start = chunk_end + timedelta(days=1)

    if not candles:
        return None

    df = pd.DataFrame(candles).drop_duplicates(subset="date").sort_values("date")
    df = df.rename(columns={
        "date": "Date", "open": "Open", "high": "High",
        "low": "Low", "close": "Close", "volume": "Volume",
    })
    df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None)
    df = df.set_index("Date")
    return df[["Open", "High", "Low", "Close", "Volume"]]
