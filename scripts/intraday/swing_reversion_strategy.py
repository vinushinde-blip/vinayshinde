"""
Oversold-reversion swing strategy — built directly from swing_move_study.py's
data-driven finding: across 500 stocks and 10 years, the price action that
actually PRECEDED a clean >=15% move (train-set effect sizes, confirmed
again on the held-out test set) was NOT strength — it was RSI well below
50, price below its 50-day SMA, well off the 52-week high, and elevated
volatility. That's the opposite of the breakout/momentum framework tested
everywhere else in this project (RS-vs-Nifty leadership, above a rising
SMA). This module turns that DESCRIPTIVE finding into an actual forward
entry rule and stop/target/time-stop framework, so it can be honestly
backtested — a feature correlating with successful moves after the fact
is not the same as a profitable rule to trade going forward (most oversold
stocks are oversold because they keep falling, not because they're about
to bounce), which is exactly why this needs a real walk-forward test
before being trusted.

Entry, all on signal day D (backward-looking only, no forward information
used in the rule itself — only the ORIGINAL study used forward returns,
and only to decide which features to build a rule from):
  1. RSI14 <= rsi_max (oversold).
  2. Close at or below the pullback_sma_days-day SMA (real pullback, not
     just noise near the average).
  3. Well off the 52-week high: dist_from_252d_high <= -min_pullback_pct.
  4. Elevated volatility context: ATR% >= the stock's own trailing
     atr_lookback_days-day median ATR% (a flat, low-vol stock oversold on
     RSI alone isn't the same setup as a genuinely volatile stretch).
  5. Optional uptrend_context: close > SMA(trend_days) — "buying a dip in
     a stock that's still structurally in an uptrend" instead of a pure,
     unguarded mean-reversion bet — tested as a separate config, not
     baked in by default, since the study did NOT show launches requiring
     this (pct_above_sma200 was actually negative for launches too).
  6. Optional market_regime: same Nifty-above-its-own-SMA filter used
     elsewhere in this project, tested as a separate guard against
     buying dips during a broad bear market.

Entry: next day's open (max_entry_gap_pct guards against chasing a gap).
Stop: the lowest low over the trailing stop_lookback_days (the recent
floor the stock is bouncing from). Target: entry + target_r_multiple * R.
Time-stop: max_holding_days (set to match the 15-day window the study
measured "clean move" over).

Returns the same trade schema as the other swing strategies in this
project so it plugs into swing_backtest.py's cost/cap/metrics machinery
unchanged, PLUS a "score" column — how extreme this stock's own setup is
relative to its own trailing history (self-relative z-score composite,
not a true cross-sectional rank across the universe, since that would
need a per-day-across-symbols pass rather than this per-symbol one; a
time-series z-score is a reasonable proxy for "how unusual is this for
THIS stock" and is what swing_reversion_ranked_backtest.py uses to pick
only the most extreme few candidates each day instead of trading every
setup that crosses the loose threshold gate).
"""

import numpy as np
import pandas as pd

from indicators import rsi, atr, sma


def find_reversion_trades(daily: pd.DataFrame, nifty_daily: pd.DataFrame, rsi_max: float = 35.0,
                           pullback_sma_days: int = 50, min_pullback_pct: float = 0.20,
                           atr_lookback_days: int = 120, stop_lookback_days: int = 10,
                           target_r_multiple: float = 3.0, max_holding_days: int = 15,
                           max_entry_gap_pct: float = 3.0, uptrend_context: bool = False,
                           trend_days: int = 200, use_market_regime: bool = False,
                           regime_sma_days: int = 200) -> pd.DataFrame:
    min_history = max(pullback_sma_days, atr_lookback_days, 252, trend_days if uptrend_context else 0) + max_holding_days
    if len(daily) <= min_history:
        return pd.DataFrame()

    close, high, low = daily["close"], daily["high"], daily["low"]

    rsi14 = rsi(close, 14)
    sma_pb = sma(close, pullback_sma_days)
    dist_from_high = close / high.rolling(252).max() - 1
    atr_pct = atr(daily, 14) / close
    atr_pct_median = atr_pct.rolling(atr_lookback_days).median()
    stop_ref = low.rolling(stop_lookback_days).min()
    higher_low_count = (low > low.shift(1)).rolling(10).sum()

    def zscore(s: pd.Series, window: int = 252) -> pd.Series:
        mean = s.rolling(window, min_periods=60).mean()
        std = s.rolling(window, min_periods=60).std()
        return (s - mean) / std.replace(0, np.nan)

    # composite: how extreme is THIS setup for THIS stock, vs its own history
    # (sign-flipped so higher score = more extreme reversion setup)
    score = (-zscore(rsi14) - zscore(close / sma_pb - 1) - zscore(dist_from_high)
             + zscore(atr_pct) - zscore(higher_low_count)) / 5.0

    is_signal = (rsi14 <= rsi_max) & (close <= sma_pb) & \
                (dist_from_high <= -min_pullback_pct) & (atr_pct >= atr_pct_median)

    if uptrend_context:
        sma_trend = sma(close, trend_days)
        is_signal = is_signal & (close > sma_trend)

    if use_market_regime:
        nifty_sma = nifty_daily["close"].rolling(regime_sma_days).mean()
        regime_ok = (nifty_daily["close"] > nifty_sma).reindex(daily.index)
        is_signal = is_signal & regime_ok.fillna(False)

    is_signal = is_signal.fillna(False)

    trades = []
    bars = list(daily.itertuples())
    n = len(bars)
    i = 0
    while i < n:
        if not is_signal.iloc[i]:
            i += 1
            continue

        signal_idx = i
        signal_date = bars[signal_idx].Index
        signal_close = bars[signal_idx].close
        support_low = stop_ref.iloc[signal_idx]
        r = signal_close - support_low
        if pd.isna(r) or r <= 0 or signal_idx + 1 >= n:
            i += 1
            continue

        entry_idx = signal_idx + 1
        entry_price = bars[entry_idx].open
        if entry_price > signal_close * (1 + max_entry_gap_pct / 100):
            i = signal_idx + 1
            continue
        entry_date = bars[entry_idx].Index
        stop_price = support_low
        target_price = entry_price + target_r_multiple * r

        exit_price, exit_date, exit_reason = None, None, None
        window_end = min(entry_idx + max_holding_days, n - 1)
        for j in range(entry_idx, window_end + 1):
            bar = bars[j]
            hit_stop = bar.low <= stop_price
            hit_target = bar.high >= target_price
            if hit_stop and bar.open < stop_price:
                exit_price, exit_reason = bar.open, "stop_gap"
            elif hit_stop:
                exit_price, exit_reason = stop_price, "stop"
            elif hit_target:
                exit_price, exit_reason = target_price, "target"
            if exit_price is not None:
                exit_date = bar.Index
                break
        if exit_price is None:
            last_bar = bars[window_end]
            exit_price, exit_date, exit_reason = last_bar.close, last_bar.Index, "time_stop"

        signal_score = score.iloc[signal_idx]
        trades.append({
            "breakout_date": signal_date, "entry_date": entry_date,
            "entry_price": round(entry_price, 2), "stop_price": round(stop_price, 2),
            "target_price": round(target_price, 2), "exit_date": exit_date,
            "exit_price": round(exit_price, 2), "exit_reason": exit_reason,
            "gross_return": (exit_price - entry_price) / entry_price,
            "score": float(signal_score) if pd.notna(signal_score) else 0.0,
        })
        i = daily.index.get_loc(exit_date) + 1

    return pd.DataFrame(trades)
