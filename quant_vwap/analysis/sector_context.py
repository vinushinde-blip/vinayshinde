"""
Sector and broad-market co-movement filter (idea #3), built on Kite's NSE
sectoral indices. These indices carry no traded volume in Kite's feed (same
issue as Nifty 50), so a true VWAP isn't available for them. Substitute: a
session TWAP (time-weighted, not volume-weighted average price) as the
reference line, then a rolling-volatility z-score of the index's deviation
from that TWAP — the same z-score mechanism used for stocks in features_v2,
applied consistently to the index instead of inventing a different metric.

Regime bucket, at every bar:
  - "stable"  : |z| <= 0.5   (index roughly at its own TWAP)
  - "bullish" : z > 0.5      (index running hot above its TWAP)
  - "bearish" : z < -0.5     (index running hot below its TWAP)
"""
import numpy as np
import pandas as pd

ZSCORE_WINDOW_BARS = {"1minute": 60, "5minute": 12}

# Maps each NSE500 "Industry" category (from the constituent list) to the
# closest available Kite sectoral index. Categories with no reasonable match
# fall back to the broad NIFTY 500 index (handled by the caller).
INDUSTRY_TO_INDEX = {
    "Financial Services": "NIFTY_FIN_SERVICE",
    "Healthcare": "NIFTY_HEALTHCARE",
    "Automobile and Auto Components": "NIFTY_AUTO",
    "Fast Moving Consumer Goods": "NIFTY_FMCG",
    "Information Technology": "NIFTY_IT",
    "Chemicals": "NIFTY_CHEMICALS",
    "Metals & Mining": "NIFTY_METAL",
    "Power": "NIFTY_ENERGY",
    "Oil Gas & Consumable Fuels": "NIFTY_OIL_AND_GAS",
    "Consumer Durables": "NIFTY_CONSR_DURBL",
    "Consumer Services": "NIFTY_SERV_SECTOR",
    "Services": "NIFTY_SERV_SECTOR",
    "Realty": "NIFTY_REALTY",
    "Media Entertainment & Publication": "NIFTY_MEDIA",
    "Construction": "NIFTY_INFRA",
    "Construction Materials": "NIFTY_INFRA",
    # Capital Goods, Telecommunication, Textiles, Diversified -> fallback to NIFTY50
}


def build_index_regime(df: pd.DataFrame, timeframe: str, index_name: str) -> pd.DataFrame:
    df = df.sort_values("timestamp").reset_index(drop=True)
    session = df["timestamp"].dt.date
    typical = (df["high"] + df["low"] + df["close"]) / 3.0

    # session TWAP: equal-weight cumulative mean (no volume available)
    twap = typical.groupby(session).transform(lambda s: s.expanding().mean())
    dev_pct = (df["close"] - twap) / twap * 100

    window = ZSCORE_WINDOW_BARS[timeframe]
    roll_mean = dev_pct.groupby(session).transform(lambda s: s.rolling(window, min_periods=window).mean())
    roll_std = dev_pct.groupby(session).transform(lambda s: s.rolling(window, min_periods=window).std(ddof=0))
    z = (dev_pct - roll_mean) / roll_std.replace(0, np.nan)

    regime = np.where(z > 0.5, "bullish", np.where(z < -0.5, "bearish", "stable"))

    return pd.DataFrame({
        "timestamp": df["timestamp"],
        f"{index_name}_dev_pct": dev_pct,
        f"{index_name}_zscore": z,
        f"{index_name}_regime": regime,
    })


def load_all_sector_regimes(raw_dir: str, timeframe_subdir: str) -> dict:
    """Returns {index_name: regime_df} for every fetched sector index + NIFTY50."""
    import os
    names = ["NIFTY50"] + list(set(INDUSTRY_TO_INDEX.values()))
    out = {}
    for name in names:
        path = f"{raw_dir}/{timeframe_subdir}/{name}.parquet"
        if not os.path.exists(path):
            continue
        df = pd.read_parquet(path)
        tf_key = "1minute" if timeframe_subdir == "1minute" else "5minute"
        out[name] = build_index_regime(df, tf_key, name)
    return out


def stock_sector_index(industry: str) -> str:
    return INDUSTRY_TO_INDEX.get(industry, "NIFTY50")
