"""
Caps how many trades a strategy is allowed to take per day — both overall
(portfolio-wide) and per symbol ("script" in NSE trading parlance) — before
those trades get aggregated into a portfolio return.

Why this matters: strategies.py has no such limit, so e.g. vwap_mean_reversion
fires ~2,000 signals/day across the 500-stock universe. No retail trader —
and no automated system without serious infrastructure — can actually take
that many positions in a day; the backtest numbers without a cap describe a
strategy nobody could actually run. Capping makes the backtest describe
something executable.

Selection rule: strictly CHRONOLOGICAL, by entry_time, within each day.
This is deliberate, not incidental — a live trader/system encounters
signals in time order throughout the day and has no way to know in advance
which of today's signals will turn out best; picking "the best N signals
of the day" after the fact would be look-ahead bias. First-N-in-time is
the only selection rule a live system could actually implement.

Per-symbol cap is applied first (within a day), then the portfolio-wide
cap is applied to what's left, both still in chronological order.
"""

import pandas as pd


def risk_based_daily_cap(capital: float, risk_per_trade_pct: float = 0.5,
                          daily_risk_budget_pct: float = 3.0) -> int:
    """Derives a max-trades-per-day figure from actual risk management
    rather than picking an arbitrary round number.

    risk_per_trade_pct: how much of capital you're willing to lose on a
    single trade if its stop is hit (0.5% is a conservative, standard
    retail guideline — most professional risk frameworks cap single-trade
    risk at 0.5-1%).

    daily_risk_budget_pct: the point at which a real trader/system should
    stop for the day even if more signals are firing (3% is a common
    "daily loss limit" — many prop desks use 2-3%). Trading past this on a
    bad day is exactly how a string of small losses becomes a large one.

    max_trades/day = daily_risk_budget / risk_per_trade — e.g. 3% / 0.5% = 6.
    That's deliberately a small, conservative number: it assumes every
    trade taken that day could be a loser (worst case), which is the
    correct assumption for a RISK LIMIT (as opposed to an average-case
    expectation). Capital itself doesn't change the trade *count* here —
    it only changes how many rupees 0.5% is — but is accepted as a
    parameter so the reasoning is explicit and auditable.
    """
    if risk_per_trade_pct <= 0:
        raise ValueError("risk_per_trade_pct must be positive")
    return max(1, int(daily_risk_budget_pct / risk_per_trade_pct))


def apply_daily_caps(all_trades: dict, max_trades_per_day: int = None,
                      max_trades_per_symbol_per_day: int = None) -> dict:
    """all_trades: {symbol: trades_df}. Returns a same-shaped dict with
    trades dropped per the caps. None for either cap means uncapped."""
    if max_trades_per_day is None and max_trades_per_symbol_per_day is None:
        return all_trades

    frames = []
    for symbol, trades in all_trades.items():
        if trades is None or trades.empty:
            continue
        t = trades.copy()
        t["symbol"] = symbol
        frames.append(t)
    if not frames:
        return all_trades

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.sort_values(["date", "entry_time"]).reset_index(drop=True)

    kept_parts = []
    for day, day_df in combined.groupby("date"):
        day_df = day_df.sort_values("entry_time")
        if max_trades_per_symbol_per_day is not None:
            day_df = day_df.groupby("symbol", group_keys=False).head(max_trades_per_symbol_per_day)
            day_df = day_df.sort_values("entry_time")
        if max_trades_per_day is not None:
            day_df = day_df.head(max_trades_per_day)
        kept_parts.append(day_df)

    kept = pd.concat(kept_parts, ignore_index=True) if kept_parts else combined.iloc[0:0]

    result = {}
    for symbol, group in kept.groupby("symbol"):
        result[symbol] = group.drop(columns="symbol").reset_index(drop=True)
    return result
