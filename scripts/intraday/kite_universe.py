"""
Builds the intraday-tradable universe from NSE's official Nifty 500 index
constituent list, matched against Kite's live instrument master to get each
stock's instrument_token and to drop anything not currently tradable
intraday.

Why the official list instead of computing our own liquidity ranking:
Nifty 500 membership is itself turnover/market-cap based and reconstituted
twice a year by NSE Indices, so it's already "the ~500 most liquid NSE
stocks" — no need to reinvent that ranking from raw Kite volume data.

Why we still filter through Kite's instrument master: a stock can be a
Nifty 500 constituent but currently trading under a suffixed series (e.g.
"-BE", trade-to-trade) due to a temporary surveillance action — those
can't be shorted or even day-traded at all (compulsory delivery
settlement), so they're excluded here rather than left for the backtest
to silently assume they're tradable.

Usage:
    python3 kite_universe.py
"""

import os
import time
from datetime import datetime, timedelta

import pandas as pd

from kite_auth import get_kite

NIFTY500_URL = "https://archives.nseindia.com/content/indices/ind_nifty500list.csv"
OUTPUT_FILE = os.path.join(os.path.dirname(__file__), "..", "..", "data", "intraday", "universe_top500.csv")
REQUEST_DELAY_SEC = 0.35
MAX_RETRIES = 4


def fetch_nifty500_symbols() -> pd.DataFrame:
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
    df = pd.read_csv(NIFTY500_URL, storage_options=headers)
    df = df.rename(columns={"Symbol": "tradingsymbol", "Company Name": "name", "Industry": "industry"})
    return df[["tradingsymbol", "name", "industry"]]


def fetch_tradable_nse_equities(kite) -> pd.DataFrame:
    """Plain (no series-suffix) tradingsymbols only — real mainboard EQ-series
    stocks tradable intraday. Suffixed variants (-BE/-BZ trade-to-trade,
    -SM/-ST SME board, -SG/-GS/-GB/-N0..N9 bonds/SDLs/SGBs/NCDs) are excluded;
    none of them support intraday MIS orders either way.
    """
    instruments = pd.DataFrame(kite.instruments("NSE"))
    equities = instruments[
        (instruments["instrument_type"] == "EQ") &
        (instruments["segment"] == "NSE") &
        (~instruments["tradingsymbol"].str.contains("-"))
    ].copy()
    return equities[["instrument_token", "tradingsymbol"]]


def _fetch_with_retry(kite, token, from_date, to_date):
    for attempt in range(MAX_RETRIES):
        try:
            return kite.historical_data(token, from_date, to_date, interval="day")
        except Exception as e:
            wait = 2 ** attempt
            print(f"  retry {attempt + 1}/{MAX_RETRIES} for token {token} after error: {e} (sleep {wait}s)")
            time.sleep(wait)
    return []


def rank_by_liquidity(kite, universe: pd.DataFrame, lookback_days: int = 60) -> pd.DataFrame:
    """The Nifty 500 CSV lists constituents alphabetically, not by liquidity,
    but the backtest's slippage model needs a relative liquidity ordering
    within the 500 (most liquid = tightest assumed slippage). Rank these 500
    by recent average daily turnover from Kite.
    """
    to_date = datetime.today()
    from_date = to_date - timedelta(days=int(lookback_days * 1.6))

    turnovers = []
    for i, r in enumerate(universe.itertuples(), start=1):
        candles = _fetch_with_retry(kite, r.instrument_token, from_date, to_date)
        avg_turnover = 0.0
        if candles:
            df = pd.DataFrame(candles).tail(lookback_days)
            avg_turnover = (df["close"] * df["volume"]).mean()
        turnovers.append(avg_turnover)
        time.sleep(REQUEST_DELAY_SEC)
        if i % 100 == 0:
            print(f"  liquidity ranking progress: {i}/{len(universe)}")

    universe = universe.copy()
    universe["avg_turnover"] = turnovers
    universe = universe.sort_values("avg_turnover", ascending=False).reset_index(drop=True)
    universe["liquidity_rank"] = universe.index + 1
    return universe


def main():
    nifty500 = fetch_nifty500_symbols()
    print(f"Nifty 500 official constituents: {len(nifty500)}")

    kite = get_kite()
    tradable = fetch_tradable_nse_equities(kite)
    print(f"Currently tradable (plain-series) NSE equities in Kite: {len(tradable)}")

    merged = nifty500.merge(tradable, on="tradingsymbol", how="inner")
    dropped = set(nifty500["tradingsymbol"]) - set(merged["tradingsymbol"])
    if dropped:
        print(f"Dropped {len(dropped)} Nifty 500 symbols not currently intraday-tradable "
              f"(renamed/suspended/under a restricted series): {sorted(dropped)}")

    print(f"Ranking {len(merged)} stocks by recent liquidity (turnover)...")
    ranked = rank_by_liquidity(kite, merged)

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    ranked.to_csv(OUTPUT_FILE, index=False)
    print(f"Saved {len(ranked)} stocks to {OUTPUT_FILE}")
    print(ranked[["liquidity_rank", "tradingsymbol", "name", "avg_turnover"]].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
