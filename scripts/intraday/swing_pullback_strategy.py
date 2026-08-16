"""
Pullback-entry swing strategy — the alternative hypothesis to
swing_strategy.py's breakout entry. Same universe, same delivery costs,
same portfolio caps, same relative-strength/market-regime filters, same
R-multiple stop/target/time-stop risk framework — the ONLY thing that
changes is the entry trigger. That's deliberate: it isolates whether
"buy strength on a breakout" or "buy a pullback within confirmed
strength" is the better trigger, holding everything else constant, rather
than changing several things at once and not knowing which one mattered.

Why try this at all: the breakout version chases an already-extended
move, and testing showed it's prone to whipsaws (stopped out then the
stock resumes) — a known weakness of pure breakout entries. Buying a
pullback to the 20-day average within an already-confirmed uptrend buys
at a discount to the recent high with a tighter, more logical stop (just
below the support being tested), instead of paying up for a fresh high.

Entry, all on signal day D:
  1. Trend: close > a RISING trend_days-day SMA (same as breakout version).
  2. Recent leadership: made a new lookback_days-day high within the last
     recent_high_within_days days — confirms this is a real pullback from
     genuine strength, not a stock just drifting sideways near its
     average with no leadership behind it.
  3. Pullback + reclaim: yesterday's close was AT OR BELOW the
     pullback_sma_days-day SMA, today's close is back ABOVE it — buying
     the bounce off the average, not the dip itself (no look-ahead: today's
     own close, not tomorrow's, confirms the reclaim).
  4. Relative strength vs Nifty, market regime — same as breakout version.

Entry: next day's open. Stop: the lowest low since price was last above
the pullback SMA (the actual support level being defended). Target:
entry + target_r_multiple * R. Time-stop: max_holding_days.

Returns the same trade schema as swing_strategy.find_swing_trades so it
plugs into swing_backtest.py's cost/cap/metrics plumbing unchanged.
"""

import pandas as pd


def find_pullback_trades(daily: pd.DataFrame, nifty_daily: pd.DataFrame, lookback_days: int = 20,
                          recent_high_within_days: int = 10, trend_days: int = 50, trend_rising_days: int = 5,
                          pullback_sma_days: int = 20, target_r_multiple: float = 3.0,
                          max_holding_days: int = 10, max_entry_gap_pct: float = 3.0,
                          rs_lookback_days: int = 50, use_relative_strength: bool = True,
                          use_market_regime: bool = True, regime_sma_days: int = 200) -> pd.DataFrame:
    min_history = max(lookback_days, trend_days, rs_lookback_days, regime_sma_days if use_market_regime else 0) + max_holding_days
    if len(daily) <= min_history:
        return pd.DataFrame()

    sma_trend = daily["close"].rolling(trend_days).mean()
    trend_condition = (daily["close"] > sma_trend) & (sma_trend > sma_trend.shift(trend_rising_days))

    rolling_high = daily["high"].rolling(lookback_days).max()
    made_recent_high = (daily["high"] >= rolling_high).rolling(recent_high_within_days).max().astype(bool)

    sma_pb = daily["close"].rolling(pullback_sma_days).mean()
    reclaim_condition = (daily["close"].shift(1) <= sma_pb.shift(1)) & (daily["close"] > sma_pb)

    is_signal = trend_condition & made_recent_high & reclaim_condition

    if use_relative_strength:
        stock_return_n = daily["close"] / daily["close"].shift(rs_lookback_days) - 1
        nifty_return_n = (nifty_daily["close"] / nifty_daily["close"].shift(rs_lookback_days) - 1).reindex(daily.index)
        is_signal = is_signal & (stock_return_n > nifty_return_n)

    if use_market_regime:
        nifty_sma = nifty_daily["close"].rolling(regime_sma_days).mean()
        regime_ok = (nifty_daily["close"] > nifty_sma).reindex(daily.index)
        is_signal = is_signal & regime_ok

    is_signal = is_signal.fillna(False)

    # support level: lowest low over the JUST-COMPLETED below-SMA streak.
    # shift(1) matters here — without it, the reclaim day (the first day of
    # a NEW above-SMA streak) would read its own low as "support" instead
    # of the actual pullback low from the streak that was just defended.
    # Verified with a synthetic dip-and-reclaim sequence before trusting
    # this on real data.
    above_sma = daily["close"] > sma_pb
    below_streak_id = (~above_sma).ne((~above_sma).shift()).cumsum()
    pullback_support = daily["low"].groupby(below_streak_id).cummin().shift(1)

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
        support_low = pullback_support.iloc[signal_idx]
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

        trades.append({
            "breakout_date": signal_date, "entry_date": entry_date,
            "entry_price": round(entry_price, 2), "stop_price": round(stop_price, 2),
            "target_price": round(target_price, 2), "exit_date": exit_date,
            "exit_price": round(exit_price, 2), "exit_reason": exit_reason,
            "gross_return": (exit_price - entry_price) / entry_price,
        })
        i = daily.index.get_loc(exit_date) + 1

    return pd.DataFrame(trades)
