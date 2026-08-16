"""
The formal swing breakout strategy — entry/exit rules on daily bars,
long-only. Builds on scanner_swing_breakout.py's detector (breakout
strength, volume vs median, rising 50-day trend, RSI momentum floor) but
adds the pieces a scanner doesn't need: an actual entry price, a hard
stop, a profit target, a time-stop, AND two market-relative conditions
added after a 5-year backtest showed a severe regime dependency (strong
in 2023/2024, -36% in 2022, -22% in 2025 — the strategy was buying
breakouts in stocks that were only moving because the whole market was,
not because they had real leadership, and got caught in every broad
selloff):

  - Relative strength vs Nifty: the stock's own trailing return over
    `rs_lookback_days` must EXCEED Nifty's return over the same window.
    Only trade stocks actually outperforming the index, not ones merely
    drifting up (or down less) with it — the classic CANSLIM/Minervini
    "relative strength leadership" filter.
  - Market regime: Nifty itself must be above its own `regime_sma_days`
    SMA. Sits out entirely when the broader market is in a confirmed
    downtrend, rather than fighting it stock by stock. This is the
    direct, targeted fix for 2022/2025 specifically.

Entry: next day's open after a valid breakout day (no look-ahead — the
breakout is only confirmed once day D's own bar is fully known, and both
new conditions use only data at/before that day too).

Risk unit R = breakout day's close - breakout day's low (the same
distance the scanner already used to define "failure").

Exit, checked in priority order each subsequent day:
  1. Stop-loss: breakout day's low. If the day's low breaches it, exit at
     the stop price — unless the day opens below the stop (a gap through),
     in which case exit at that day's open instead (can't fill better than
     the market opened).
  2. Target: entry_price + target_r_multiple * R. If the day's high
     reaches it, exit at the target price.
  3. Time-stop: if neither triggers within max_holding_days trading days,
     exit at that day's close.

No costs applied here — that's swing_backtest.py's job (swing_costs.py +
the concurrency/daily caps from scanner_swing_breakout.py).
"""

import pandas as pd

import indicators as ind


def find_swing_trades(daily: pd.DataFrame, nifty_daily: pd.DataFrame, lookback_days: int = 20,
                       vol_mult: float = 1.25, min_breakout_pct: float = 0.5, trend_days: int = 50,
                       trend_rising_days: int = 5, rsi_min: float = 50.0, target_r_multiple: float = 3.0,
                       max_holding_days: int = 10, max_entry_gap_pct: float = 3.0,
                       rs_lookback_days: int = 50, use_relative_strength: bool = True,
                       use_market_regime: bool = True, regime_sma_days: int = 200) -> pd.DataFrame:
    """max_entry_gap_pct guards against a real risk-management gap found by
    testing: R is measured at the breakout bar (close - low), but entry
    happens at the NEXT day's open, which can gap up well beyond the
    breakout close overnight. The stop (breakout low) doesn't move, so a
    big gap silently balloons the realized risk past the intended R —
    confirmed on WOCKPHARMA (2026-06-01): entry 2377 vs stop 1872 was a
    21% risk on a trade sized for ~a few percent. Skip the trade instead
    of chasing an entry that's gapped too far past the signal.

    nifty_daily must span at least as far back as `daily` minus
    regime_sma_days of warmup, or the regime condition is undefined
    (treated as "don't trade") for early dates — fetch Nifty history well
    before the stock backtest's own start date to avoid losing real
    coverage to warmup.
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
        regime_condition = (nifty_daily["close"] > nifty_sma).reindex(daily.index)
        is_signal = is_signal & regime_condition

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

        trades.append({
            "breakout_date": bars[breakout_idx].Index, "entry_date": entry_date,
            "entry_price": round(entry_price, 2), "stop_price": round(stop_price, 2),
            "target_price": round(target_price, 2), "exit_date": exit_date,
            "exit_price": round(exit_price, 2), "exit_reason": exit_reason,
            "gross_return": (exit_price - entry_price) / entry_price,
        })
        # next signal search starts after this trade's exit — a symbol can't
        # be in two of its own swing positions at once
        i = daily.index.get_loc(exit_date) + 1

    return pd.DataFrame(trades)
