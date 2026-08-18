"""
Volatility-normalized / structural filters, built only from what Kite's OHLCV
candles actually contain (no tick-level trade-direction data, no bid-ask
history — see the feasibility note in the report for what was deliberately
left out).

1. Rolling z-score of delta% (per-stock, per-session, volatility-normalized
   distance from VWAP) — replaces the fixed 1.5/2/3/4% thresholds used in the
   base study.
2. VWAP slope (first derivative of the VWAP line) — flags trending vs.
   range-bound regimes.
3. Velocity of dislocation (rate of change of the z-score) with a
   deceleration flag — panic-spike-then-fading vs. still-accelerating.
4. Volume-climax proxy — an oversized candle at a z-score extreme followed by
   a failure to extend, as a proxy for absorption. NOT a substitute for real
   order-flow / block-deal data, which this dataset doesn't contain.
5. Volume-weighted MACD (VWMA-based) with a simple new-high/new-low
   divergence flag — a heuristic proxy for the divergence idea, not a
   rigorous swing-point detector.
"""
import numpy as np
import pandas as pd

# ~1 hour trailing window, expressed in bars per timeframe
ZSCORE_WINDOW_BARS = {"1minute": 60, "5minute": 12, "15minute": 4}
VWAP_SLOPE_WINDOW = {"1minute": 5, "5minute": 5, "15minute": 3}
VELOCITY_LOOKBACK = 4
VOLUME_CLIMAX_MULT = 3.0
DIVERGENCE_LOOKBACK = 20


def add_zscore(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    window = ZSCORE_WINDOW_BARS[timeframe]
    grp = df.groupby("session")["delta_pct"]
    roll_mean = grp.transform(lambda s: s.rolling(window, min_periods=window).mean())
    roll_std = grp.transform(lambda s: s.rolling(window, min_periods=window).std(ddof=0))
    df["delta_zscore"] = (df["delta_pct"] - roll_mean) / roll_std.replace(0, np.nan)
    return df


def add_vwap_slope(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    window = VWAP_SLOPE_WINDOW[timeframe]
    grp = df.groupby("session")["vwap"]
    vwap_lag = grp.transform(lambda s: s.shift(window))
    df["vwap_slope_pct"] = (df["vwap"] - vwap_lag) / vwap_lag.replace(0, np.nan) * 100
    return df


def add_velocity(df: pd.DataFrame) -> pd.DataFrame:
    grp = df.groupby("session")["delta_zscore"]
    df["zscore_velocity"] = grp.transform(lambda s: s.diff(VELOCITY_LOOKBACK) / VELOCITY_LOOKBACK)
    # "was accelerating, now decelerating": |velocity| below its own recent local max
    vel_abs = df["zscore_velocity"].abs()
    roll_max = df.groupby("session")["zscore_velocity"].transform(
        lambda s: s.abs().rolling(VELOCITY_LOOKBACK, min_periods=1).max()
    )
    df["velocity_decelerating"] = vel_abs < roll_max
    return df


def add_volume_climax(df: pd.DataFrame) -> pd.DataFrame:
    is_climax = df["vol_ratio"] >= VOLUME_CLIMAX_MULT
    next_high = df["high"].shift(-1)
    next_low = df["low"].shift(-1)
    same_session_next = df["session"] == df["session"].shift(-1)
    failed_to_extend_up = (next_high <= df["high"]) & same_session_next
    failed_to_extend_down = (next_low >= df["low"]) & same_session_next
    df["volume_climax_up_absorbed"] = is_climax & failed_to_extend_up
    df["volume_climax_down_absorbed"] = is_climax & failed_to_extend_down
    return df


def add_vwma_macd(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    """Volume-weighted MACD: VWMA(fast) - VWMA(slow), using a rolling
    price*volume / volume ratio instead of EMA. Session-bounded."""
    pv = df["close"] * df["volume"]
    grp_pv = df.groupby("session")
    vwma_fast = pv.groupby(df["session"]).transform(lambda s: s.rolling(fast, min_periods=fast).sum()) / \
        df["volume"].groupby(df["session"]).transform(lambda s: s.rolling(fast, min_periods=fast).sum())
    vwma_slow = pv.groupby(df["session"]).transform(lambda s: s.rolling(slow, min_periods=slow).sum()) / \
        df["volume"].groupby(df["session"]).transform(lambda s: s.rolling(slow, min_periods=slow).sum())
    df["vmacd"] = vwma_fast - vwma_slow
    df["vmacd_signal"] = df.groupby("session")["vmacd"].transform(
        lambda s: s.ewm(span=signal, min_periods=signal, adjust=False).mean()
    )
    df["vmacd_hist"] = df["vmacd"] - df["vmacd_signal"]
    return df


def add_vmacd_divergence(df: pd.DataFrame) -> pd.DataFrame:
    """Bullish: price makes a fresh N-bar low but vmacd does not.
    Bearish: price makes a fresh N-bar high but vmacd does not."""
    grp = df.groupby("session")
    price_roll_min = grp["close"].transform(lambda s: s.rolling(DIVERGENCE_LOOKBACK, min_periods=DIVERGENCE_LOOKBACK).min())
    price_roll_max = grp["close"].transform(lambda s: s.rolling(DIVERGENCE_LOOKBACK, min_periods=DIVERGENCE_LOOKBACK).max())
    vmacd_roll_min = grp["vmacd"].transform(lambda s: s.rolling(DIVERGENCE_LOOKBACK, min_periods=DIVERGENCE_LOOKBACK).min())
    vmacd_roll_max = grp["vmacd"].transform(lambda s: s.rolling(DIVERGENCE_LOOKBACK, min_periods=DIVERGENCE_LOOKBACK).max())

    fresh_price_low = df["close"] <= price_roll_min
    vmacd_not_new_low = df["vmacd"] > vmacd_roll_min
    df["bullish_divergence"] = fresh_price_low & vmacd_not_new_low

    fresh_price_high = df["close"] >= price_roll_max
    vmacd_not_new_high = df["vmacd"] < vmacd_roll_max
    df["bearish_divergence"] = fresh_price_high & vmacd_not_new_high
    return df


def build_v2_features(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """Assumes df already has session/vwap/delta_pct/vol_ratio from build_feature_frame."""
    df = add_zscore(df, timeframe)
    df = add_vwap_slope(df, timeframe)
    df = add_velocity(df)
    df = add_volume_climax(df)
    df = add_vwma_macd(df)
    df = add_vmacd_divergence(df)
    return df
