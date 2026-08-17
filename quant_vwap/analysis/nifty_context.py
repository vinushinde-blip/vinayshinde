"""
Nifty 50 index market-regime classification, used as a contextual filter on the
per-stock VWAP mean-reversion strategies: the index has no traded volume (Kite
reports volume=0 for index candles), so it gets no VWAP/delta% of its own — its
regime is derived purely from price/momentum structure.

Regime at each bar:
  - "strong_up":   close > EMA50, ADX(14) > 25, MACD histogram > 0
  - "strong_down": close < EMA50, ADX(14) > 25, MACD histogram < 0
  - "neutral":     anything else (choppy / weak trend)

Mean reversion against a strongly trending index is the higher-risk case (e.g.
buying an oversold stock while the whole market is in a strong downtrend), so the
filtered strategy variant skips:
  - LONG (price below its own VWAP, betting on reversion up) entries when the
    index regime is "strong_down"
  - SHORT (price above its own VWAP, betting on reversion down) entries when the
    index regime is "strong_up"
"""
import numpy as np
import pandas as pd

from features import macd, adx


def build_nifty_regime(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("timestamp").reset_index(drop=True)
    ema50 = df["close"].ewm(span=50, min_periods=50).mean()
    _, _, hist = macd(df["close"])
    adx14 = adx(df["high"], df["low"], df["close"])

    trend = np.where(
        (df["close"] > ema50) & (adx14 > 25) & (hist > 0), "strong_up",
        np.where((df["close"] < ema50) & (adx14 > 25) & (hist < 0), "strong_down", "neutral"),
    )

    return pd.DataFrame({
        "timestamp": df["timestamp"],
        "nifty_close": df["close"],
        "nifty_ema50": ema50,
        "nifty_adx14": adx14,
        "nifty_macd_hist": hist,
        "nifty_regime": trend,
    })


def load_nifty_regime(raw_dir: str, timeframe_subdir: str) -> pd.DataFrame:
    path = f"{raw_dir}/{timeframe_subdir}/NIFTY50.parquet"
    df = pd.read_parquet(path)
    return build_nifty_regime(df)
