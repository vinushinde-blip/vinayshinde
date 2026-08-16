"""
Objective NIFTY-based market regime classifier, shared by all the regime-
conditioned analysis in this project. Two independent axes:

  trend_regime (SMA200 level + slope, not level alone — a flat SMA200
  the price is oscillating around is a fundamentally different, choppier
  environment than one that's cleanly rising or falling, even though
  "close > SMA200" alone can't tell them apart):
    "bull"       - close > SMA200 AND SMA200 rising over slope_days
    "bear"       - close < SMA200 AND SMA200 falling over slope_days
    "transition" - everything else (e.g. above a still-falling average
                    during an early recovery, or below a still-rising one
                    during a rollover) - deliberately its own bucket
                    rather than being forced into bull or bear, since
                    that's exactly the kind of whipsaw-prone stretch
                    earlier regime-confirmation testing struggled with.

  vol_regime (trailing 20-day realized volatility, percentile-ranked
  against its own trailing 2-year history so "high vol" adapts over
  time rather than using a fixed absolute threshold):
    "high_vol" - top third percentile
    "low_vol"  - bottom third
    "mid_vol"  - middle third

Both are computed strictly from information available up to and including
each day (no look-ahead) so they're valid to use as a live filter, not
just for descriptive analysis.
"""

import pandas as pd


def classify_regime(nifty_daily: pd.DataFrame, slope_days: int = 20, vol_lookback_days: int = 20,
                     vol_percentile_window: int = 504) -> pd.DataFrame:
    close = nifty_daily["close"]
    sma200 = close.rolling(200).mean()
    sma200_rising = sma200 > sma200.shift(slope_days)
    above = close > sma200

    trend_regime = pd.Series("transition", index=nifty_daily.index)
    trend_regime[above & sma200_rising] = "bull"
    trend_regime[~above & ~sma200_rising] = "bear"

    realized_vol = close.pct_change().rolling(vol_lookback_days).std()
    vol_pct_rank = realized_vol.rolling(vol_percentile_window, min_periods=100).rank(pct=True)
    vol_regime = pd.Series("mid_vol", index=nifty_daily.index)
    vol_regime[vol_pct_rank >= 0.67] = "high_vol"
    vol_regime[vol_pct_rank <= 0.33] = "low_vol"

    out = pd.DataFrame({
        "trend_regime": trend_regime,
        "vol_regime": vol_regime,
        "sma200": sma200,
        "realized_vol_20d": realized_vol,
    }, index=nifty_daily.index)
    # rows before SMA200/vol have enough history are meaningless - mark clearly
    out.loc[sma200.isna(), "trend_regime"] = "warmup"
    out.loc[realized_vol.isna(), "vol_regime"] = "warmup"
    return out
