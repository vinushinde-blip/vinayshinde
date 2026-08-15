"""
Trigger and filter building blocks for strategy_search.py.

A trigger produces a raw directional signal per bar: +1 (long), -1 (short),
0 (nothing) — computed only from information available at that bar's close.

A filter produces (long_ok, short_ok) boolean Series — additional
conditions a trigger's signal must also satisfy to be taken. Filters don't
generate signals themselves, only gate a trigger's.

Keeping these as a fixed, named, finite set (not "every indicator with
every parameter") is deliberate — it's what keeps the search space bounded
and reportable rather than an unbounded overfitting machine.
"""

import numpy as np
import pandas as pd

import indicators as ind


def _crossed_up(series: pd.Series, level) -> pd.Series:
    prev = series.shift(1)
    lvl = level.shift(1) if isinstance(level, pd.Series) else level
    lvl_now = level if isinstance(level, pd.Series) else level
    return (prev <= lvl) & (series > lvl_now)


def _crossed_down(series: pd.Series, level) -> pd.Series:
    prev = series.shift(1)
    lvl = level.shift(1) if isinstance(level, pd.Series) else level
    lvl_now = level if isinstance(level, pd.Series) else level
    return (prev >= lvl) & (series < lvl_now)


# ---------- triggers ----------

def trig_rsi_reversal(df, period=14, oversold=30, overbought=70):
    r = ind.rsi(df["close"], period)
    sig = pd.Series(0, index=df.index)
    sig[_crossed_up(r, oversold)] = 1
    sig[_crossed_down(r, overbought)] = -1
    return sig


def trig_macd_cross(df, fast=12, slow=26, signal=9):
    macd_line, signal_line, _ = ind.macd(df["close"], fast, slow, signal)
    sig = pd.Series(0, index=df.index)
    sig[_crossed_up(macd_line, signal_line)] = 1
    sig[_crossed_down(macd_line, signal_line)] = -1
    return sig


def trig_bb_breakout(df, period=20, n_std=2.0):
    upper, _, lower = ind.bollinger_bands(df["close"], period, n_std)
    sig = pd.Series(0, index=df.index)
    sig[_crossed_up(df["close"], upper)] = 1
    sig[_crossed_down(df["close"], lower)] = -1
    return sig


def trig_bb_reversion(df, period=20, n_std=2.0):
    upper, _, lower = ind.bollinger_bands(df["close"], period, n_std)
    sig = pd.Series(0, index=df.index)
    sig[_crossed_up(df["close"], lower)] = 1     # bounced back up through lower band
    sig[_crossed_down(df["close"], upper)] = -1  # bounced back down through upper band
    return sig


def trig_ma_cross(df, fast=9, slow=21):
    f = ind.ema(df["close"], fast)
    s = ind.ema(df["close"], slow)
    sig = pd.Series(0, index=df.index)
    sig[_crossed_up(f, s)] = 1
    sig[_crossed_down(f, s)] = -1
    return sig


def trig_vwap_cross(df):
    vwap = ind.intraday_vwap(df)
    sig = pd.Series(0, index=df.index)
    sig[_crossed_up(df["close"], vwap)] = 1
    sig[_crossed_down(df["close"], vwap)] = -1
    return sig


def trig_stochastic_cross(df, k_period=14, d_period=3, oversold=20, overbought=80):
    k, d = ind.stochastic(df, k_period, d_period)
    sig = pd.Series(0, index=df.index)
    bullish = _crossed_up(k, d) & (k < oversold + 10)
    bearish = _crossed_down(k, d) & (k > overbought - 10)
    sig[bullish] = 1
    sig[bearish] = -1
    return sig


TRIGGERS = {
    "rsi_reversal": trig_rsi_reversal,
    "macd_cross": trig_macd_cross,
    "bb_breakout": trig_bb_breakout,
    "bb_reversion": trig_bb_reversion,
    "ma_cross": trig_ma_cross,
    "vwap_cross": trig_vwap_cross,
    "stochastic_cross": trig_stochastic_cross,
}


# ---------- filters ----------

def filt_trend(df, threshold=20):
    a = ind.adx(df, 14)
    ok = a > threshold
    return ok, ok.copy()


def filt_volume(df, z_threshold=0.5):
    z = ind.volume_zscore(df["volume"], 20)
    ok = z > z_threshold
    return ok, ok.copy()


def filt_rsi_bias(df, period=14):
    r = ind.rsi(df["close"], period)
    return (r > 50), (r < 50)


def filt_vwap_alignment(df):
    vwap = ind.intraday_vwap(df)
    return (df["close"] > vwap), (df["close"] < vwap)


FILTERS = {
    "trend": filt_trend,
    "volume": filt_volume,
    "rsi_bias": filt_rsi_bias,
    "vwap_alignment": filt_vwap_alignment,
}
