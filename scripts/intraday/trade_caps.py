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

Selection rule when more signals fire than the day's budget allows: two
modes, both non-lookahead —

  "chronological": strictly by entry_time. A live trader/system encounters
  signals in time order and has no way to know in advance which of today's
  signals will turn out best — picking "the best N of the day" by REALIZED
  RETURN would be look-ahead bias no live system could implement.

  "liquidity": prioritize signals from more liquid stocks first (still
  chronological within the same liquidity tier). This is NOT look-ahead —
  liquidity_rank is computed from trailing turnover before the trade, the
  same way a real trader would keep a pre-ranked watchlist. More liquid
  names mean lower realized slippage (see costs.slippage_bps), so this is
  a legitimate, implementable-live way to bias toward better expected
  outcomes without touching the trade's own future return.

Explicitly NOT implemented: filtering by which trades actually won. That
would make the backtest describe a strategy nobody could run live.

Per-symbol cap is applied first (within a day), then the portfolio-wide
cap is applied to what's left.
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
                      max_trades_per_symbol_per_day: int = None,
                      priority: str = "chronological", rank: dict = None) -> dict:
    """all_trades: {symbol: trades_df}. Returns a same-shaped dict with
    trades dropped per the caps. None for either cap means uncapped.

    priority: "chronological" (default) or "liquidity" — see module
    docstring. "liquidity" requires `rank` ({symbol: liquidity_rank},
    1 = most liquid).
    """
    if max_trades_per_day is None and max_trades_per_symbol_per_day is None:
        return all_trades
    if priority not in ("chronological", "liquidity"):
        raise ValueError(f"unknown priority: {priority!r}")
    if priority == "liquidity" and not rank:
        raise ValueError("priority='liquidity' requires a rank dict")

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
    if priority == "liquidity":
        combined["_liquidity_rank"] = combined["symbol"].map(rank).fillna(len(rank) + 1)
        sort_cols = ["date", "_liquidity_rank", "entry_time"]
    else:
        sort_cols = ["date", "entry_time"]
    combined = combined.sort_values(sort_cols).reset_index(drop=True)

    kept_parts = []
    for day, day_df in combined.groupby("date"):
        day_df = day_df.sort_values(sort_cols[1:])
        if max_trades_per_symbol_per_day is not None:
            day_df = day_df.groupby("symbol", group_keys=False).head(max_trades_per_symbol_per_day)
            day_df = day_df.sort_values(sort_cols[1:])
        if max_trades_per_day is not None:
            day_df = day_df.head(max_trades_per_day)
        kept_parts.append(day_df)

    kept = pd.concat(kept_parts, ignore_index=True) if kept_parts else combined.iloc[0:0]
    kept = kept.drop(columns="_liquidity_rank", errors="ignore")

    result = {}
    for symbol, group in kept.groupby("symbol"):
        result[symbol] = group.drop(columns="symbol").reset_index(drop=True)
    return result
