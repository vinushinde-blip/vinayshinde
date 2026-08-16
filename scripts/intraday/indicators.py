"""
Reusable technical indicators computed on a single symbol's intraday bar
DataFrame (open, high, low, close, volume), per trading day where the
indicator is inherently intraday (VWAP-relative, opening range), and
rolling across the whole series otherwise (RSI, MACD, etc. — computed
bar-to-bar without resetting at day boundaries, same as most charting
platforms do intraday).

Every function returns a pandas Series aligned to df.index. These are
building blocks for strategy_search.py, not strategies in themselves.
"""

import numpy as np
import pandas as pd


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def bollinger_bands(close: pd.Series, period: int = 20, n_std: float = 2.0):
    mid = close.rolling(period).mean()
    std = close.rolling(period).std()
    upper = mid + n_std * std
    lower = mid - n_std * std
    return upper, mid, lower


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()


def adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    tr_atr = atr(df, period).replace(0, np.nan)
    plus_di = 100 * pd.Series(plus_dm, index=df.index).ewm(alpha=1 / period, min_periods=period, adjust=False).mean() / tr_atr
    minus_di = 100 * pd.Series(minus_dm, index=df.index).ewm(alpha=1 / period, min_periods=period, adjust=False).mean() / tr_atr

    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()


def stochastic(df: pd.DataFrame, k_period: int = 14, d_period: int = 3):
    low_min = df["low"].rolling(k_period).min()
    high_max = df["high"].rolling(k_period).max()
    k = 100 * (df["close"] - low_min) / (high_max - low_min).replace(0, np.nan)
    d = k.rolling(d_period).mean()
    return k, d


def sma(close: pd.Series, period: int) -> pd.Series:
    return close.rolling(period).mean()


def ema(close: pd.Series, period: int) -> pd.Series:
    return close.ewm(span=period, adjust=False).mean()


def volume_zscore(volume: pd.Series, period: int = 20) -> pd.Series:
    mean = volume.rolling(period).mean()
    std = volume.rolling(period).std()
    return (volume - mean) / std.replace(0, np.nan)


def roc(close: pd.Series, period: int = 10) -> pd.Series:
    """Rate of change / momentum over `period` bars, as a fraction."""
    return close.pct_change(period)


def intraday_vwap(df: pd.DataFrame) -> pd.Series:
    """Running VWAP that resets at each day boundary."""
    typical = (df["high"] + df["low"] + df["close"]) / 3
    out = pd.Series(index=df.index, dtype=float)
    for _, day_df in df.groupby(df.index.date):
        tp = typical.loc[day_df.index]
        vol = day_df["volume"]
        cum_vol = vol.cumsum().replace(0, np.nan)
        out.loc[day_df.index] = (tp * vol).cumsum() / cum_vol
    return out.ffill()


def anchored_vwap(df: pd.DataFrame, anchor_date) -> pd.Series:
    """VWAP accumulated from `anchor_date` onward using DAILY typical price
    (H+L+C)/3 weighted by daily volume — a daily-bar approximation of
    intraday VWAP, since true intraday VWAP needs intraday data this
    multi-year daily backtest doesn't have. Used as a swing-trade support/
    trailing-stop reference (a well-regarded technique — Brian Shannon's
    anchored VWAP approach — normally computed intraday-anchored, here
    daily-anchored to fit the data available). Only rows at/after
    anchor_date get a value; earlier rows are NaN.
    """
    typical = (df["high"] + df["low"] + df["close"]) / 3
    mask = df.index >= anchor_date
    vol = df["volume"].where(mask, 0.0)
    tp_vol = (typical * df["volume"]).where(mask, 0.0)
    cum_vol = vol.cumsum().replace(0, np.nan)
    result = tp_vol.cumsum() / cum_vol
    return result.where(mask)
