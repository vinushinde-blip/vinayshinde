"""
Per-bar feature engineering: session VWAP, VWAP delta%, and a standard technical
indicator set (RSI, MACD, Bollinger Bands, ADX, ATR, volume ratio).

All indicators are computed with plain pandas/numpy (no TA-Lib dependency) so the
pipeline has no external build requirements.
"""
import numpy as np
import pandas as pd

DELTA_THRESHOLDS = [1.5, 2.0, 3.0, 4.0]  # percent


def add_session_vwap(df: pd.DataFrame) -> pd.DataFrame:
    """Adds session-reset VWAP (typical-price based) and delta% = (close-vwap)/vwap*100."""
    df = df.sort_values("timestamp").reset_index(drop=True)
    session = df["timestamp"].dt.date
    typical = (df["high"] + df["low"] + df["close"]) / 3.0
    pv = typical * df["volume"]

    cum_pv = pv.groupby(session).cumsum()
    cum_vol = df["volume"].groupby(session).cumsum()

    vwap = cum_pv / cum_vol.replace(0, np.nan)
    vwap = vwap.ffill()

    df["session"] = session.values
    df["vwap"] = vwap.values
    df["delta_pct"] = (df["close"] - df["vwap"]) / df["vwap"] * 100.0
    return df


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    out = 100 - (100 / (1 + rs))
    return out.where(avg_loss != 0, 100.0)


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def bollinger_bands(close: pd.Series, period: int = 20, num_std: float = 2.0):
    mid = close.rolling(period).mean()
    std = close.rolling(period).std(ddof=0)
    upper = mid + num_std * std
    lower = mid - num_std * std
    pct_b = (close - lower) / (upper - lower).replace(0, np.nan)
    return mid, upper, lower, pct_b


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    return tr.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()


def adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    tr = pd.concat(
        [high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()], axis=1
    ).max(axis=1)
    atr_ = tr.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()

    plus_di = 100 * pd.Series(plus_dm, index=high.index).ewm(alpha=1 / period, min_periods=period, adjust=False).mean() / atr_.replace(0, np.nan)
    minus_di = 100 * pd.Series(minus_dm, index=high.index).ewm(alpha=1 / period, min_periods=period, adjust=False).mean() / atr_.replace(0, np.nan)

    dx = (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan) * 100
    return dx.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df["rsi_14"] = rsi(df["close"], 14)
    macd_line, signal_line, hist = macd(df["close"])
    df["macd"] = macd_line
    df["macd_signal"] = signal_line
    df["macd_hist"] = hist
    bb_mid, bb_upper, bb_lower, pct_b = bollinger_bands(df["close"])
    df["bb_mid"] = bb_mid
    df["bb_upper"] = bb_upper
    df["bb_lower"] = bb_lower
    df["bb_pct_b"] = pct_b
    df["atr_14"] = atr(df["high"], df["low"], df["close"])
    df["adx_14"] = adx(df["high"], df["low"], df["close"])
    df["vol_sma_20"] = df["volume"].rolling(20).mean()
    df["vol_ratio"] = df["volume"] / df["vol_sma_20"].replace(0, np.nan)
    return df


def build_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    df = add_session_vwap(df)
    df = add_indicators(df)
    return df
