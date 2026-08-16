"""
The formal swing breakout strategy — entry/exit rules on daily bars,
long-only. Builds on scanner_swing_breakout.py's detector (breakout
strength, volume vs median, rising 50-day trend, RSI momentum floor),
plus relative strength vs Nifty and a market regime filter added after a
5-year backtest showed severe regime dependency (strong in 2023/2024,
-36% in 2022, -22% in 2025).

Every condition below is an independently toggleable parameter — this
file does NOT bundle them into one fixed "best" strategy. The point is to
test each addition's actual effect on the 5-year backtest in isolation
(see swing_indicator_tests.py) rather than assume more indicators = better;
the intraday side of this project already found that a broad, untested
combination search produces a train-set winner that collapses out of
sample. Every condition here is individually reasoned and tested, not
picked by a grid search.

Conditions:
  - Breakout: close > prior lookback_days-day high by min_breakout_pct,
    on volume >= vol_mult x lookback_days-day MEDIAN volume.
  - Trend: close above a rising trend_days-day SMA.
  - RSI momentum floor: RSI(14) > rsi_min.
  - Relative strength (use_relative_strength): stock's trailing
    rs_lookback_days return > Nifty's return over the same window —
    only trade actual index-beating leadership, not stocks merely
    drifting with the market.
  - Market regime (use_market_regime): Nifty above its own
    regime_sma_days SMA, required to hold for regime_confirm_days
    CONSECUTIVE days before re-enabling entries — the consecutive-day
    requirement specifically targets the whipsaw problem found in 2022,
    where a single-day SMA crossover let entries back in during brief
    rallies inside a choppy, not-cleanly-trending decline.
  - ADX trend strength (use_adx_filter): the STOCK's own ADX(14) above
    adx_threshold at breakout — a second, independent measure of
    directional conviction beyond the SMA-slope trend check.
  - MACD confirmation (use_macd_filter): MACD line above its signal line
    at breakout — standard momentum confirmation.
  - Volatility ceiling (use_volatility_filter): ATR(14) as a %% of price
    below max_atr_pct — excludes stocks erratic enough to be prone to
    whipsawing through both the stop and target on noise alone.
  - Anchored VWAP exit (use_vwap_exit): once vwap_grace_days have passed,
    exit if close falls below the VWAP accumulated from the breakout day
    forward — a dynamic, trend-following stop tighter than waiting for
    the fixed stop-loss or the time-stop, on the logic that a genuine
    breakout should hold above the volume-weighted average price paid
    since it started.

Entry: next day's open after a valid breakout day (no look-ahead — every
condition uses only data at/before the breakout day).

Risk unit R = breakout day's close - breakout day's low.

Exit, checked in priority order each subsequent day:
  1. Stop-loss: breakout day's low (gap-through handled: exits at that
     day's open instead if it opens below the stop).
  2. Anchored VWAP break (if use_vwap_exit and past the grace period).
  3. Target: entry_price + target_r_multiple * R.
  4. Time-stop: max_holding_days trading days with neither triggered.

No costs applied here — that's swing_backtest.py's job.
"""

import pandas as pd

import indicators as ind


def find_swing_trades(daily: pd.DataFrame, nifty_daily: pd.DataFrame, lookback_days: int = 20,
                       vol_mult: float = 1.25, min_breakout_pct: float = 0.5, trend_days: int = 50,
                       trend_rising_days: int = 5, rsi_min: float = 50.0, target_r_multiple: float = 3.0,
                       max_holding_days: int = 10, max_entry_gap_pct: float = 3.0,
                       rs_lookback_days: int = 50, use_relative_strength: bool = True,
                       use_market_regime: bool = True, regime_sma_days: int = 200,
                       regime_confirm_days: int = 1, use_adx_filter: bool = False, adx_threshold: float = 20.0,
                       use_macd_filter: bool = False, use_volatility_filter: bool = False,
                       max_atr_pct: float = 5.0, use_vwap_exit: bool = False,
                       vwap_grace_days: int = 3) -> pd.DataFrame:
    """max_entry_gap_pct guards against a real risk-management gap found by
    testing: R is measured at the breakout bar (close - low), but entry
    happens at the NEXT day's open, which can gap up well beyond the
    breakout close overnight. The stop (breakout low) doesn't move, so a
    big gap silently balloons the realized risk past the intended R —
    confirmed on WOCKPHARMA (2026-06-01): entry 2377 vs stop 1872 was a
    21% risk on a trade sized for ~a few percent. Skip the trade instead
    of chasing an entry that's gapped too far past the signal.

    nifty_daily must span well before `daily`'s own start (regime_sma_days
    + regime_confirm_days of warmup), or the regime condition is
    undefined ("don't trade") for early dates.
    """
    min_history = max(lookback_days, trend_days, rs_lookback_days) + max_holding_days
    if len(daily) <= min_history:
        return pd.DataFrame()

    prior_high = daily["high"].rolling(lookback_days).max().shift(1)
    median_vol = daily["volume"].rolling(lookback_days).median().shift(1)
    price_condition = daily["close"] > prior_high * (1 + min_breakout_pct / 100)
    volume_condition = daily["volume"] >= vol_mult * median_vol

    sma = daily["close"].rolling(trend_days).mean()
    trend_condition = (daily["close"] > sma) & (sma > sma.shift(trend_rising_days))

    rsi = ind.rsi(daily["close"], 14)
    rsi_condition = rsi > rsi_min

    is_signal = price_condition & volume_condition & trend_condition & rsi_condition

    if use_relative_strength:
        stock_return_n = daily["close"] / daily["close"].shift(rs_lookback_days) - 1
        nifty_return_n = (nifty_daily["close"] / nifty_daily["close"].shift(rs_lookback_days) - 1).reindex(daily.index)
        rs_condition = stock_return_n > nifty_return_n
        is_signal = is_signal & rs_condition

    if use_market_regime:
        nifty_sma = nifty_daily["close"].rolling(regime_sma_days).mean()
        regime_ok = (nifty_daily["close"] > nifty_sma).reindex(daily.index).fillna(False)
        if regime_confirm_days > 1:
            regime_ok = regime_ok.rolling(regime_confirm_days).min().astype(bool)
        is_signal = is_signal & regime_ok

    if use_adx_filter:
        adx_series = ind.adx(daily, 14)
        is_signal = is_signal & (adx_series > adx_threshold)

    if use_macd_filter:
        macd_line, signal_line, _ = ind.macd(daily["close"])
        is_signal = is_signal & (macd_line > signal_line)

    if use_volatility_filter:
        atr_pct = ind.atr(daily, 14) / daily["close"] * 100
        is_signal = is_signal & (atr_pct < max_atr_pct)

    is_signal = is_signal.fillna(False)

    trades = []
    bars = list(daily.itertuples())
    n = len(bars)
    i = 0
    while i < n:
        if not is_signal.iloc[i]:
            i += 1
            continue

        breakout_idx = i
        breakout_close = bars[breakout_idx].close
        breakout_low = bars[breakout_idx].low
        breakout_date = bars[breakout_idx].Index
        r = breakout_close - breakout_low
        if r <= 0 or breakout_idx + 1 >= n:
            i += 1
            continue

        entry_idx = breakout_idx + 1
        entry_price = bars[entry_idx].open
        if entry_price > breakout_close * (1 + max_entry_gap_pct / 100):
            i = breakout_idx + 1
            continue
        entry_date = bars[entry_idx].Index
        stop_price = breakout_low
        target_price = entry_price + target_r_multiple * r

        vwap_series = None
        if use_vwap_exit:
            vwap_series = ind.anchored_vwap(daily, breakout_date)

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
            elif use_vwap_exit and (j - entry_idx) >= vwap_grace_days and bar.close < vwap_series.loc[bar.Index]:
                exit_price, exit_reason = bar.close, "vwap_break"
            elif hit_target:
                exit_price, exit_reason = target_price, "target"
            if exit_price is not None:
                exit_date = bar.Index
                break
        if exit_price is None:
            last_bar = bars[window_end]
            exit_price, exit_date, exit_reason = last_bar.close, last_bar.Index, "time_stop"

        trades.append({
            "breakout_date": breakout_date, "entry_date": entry_date,
            "entry_price": round(entry_price, 2), "stop_price": round(stop_price, 2),
            "target_price": round(target_price, 2), "exit_date": exit_date,
            "exit_price": round(exit_price, 2), "exit_reason": exit_reason,
            "gross_return": (exit_price - entry_price) / entry_price,
        })
        # next signal search starts after this trade's exit — a symbol can't
        # be in two of its own swing positions at once
        i = daily.index.get_loc(exit_date) + 1

    return pd.DataFrame(trades)
