"""
The final selected strategy — single source of truth for what actually
gets traded, distinct from strategies.py (which holds every CANDIDATE
tested) and strategy_search.py (which searches for one).

Selection basis, from the full backtest campaign in this directory:
  - Started with 4 hand-designed strategies (long+short) plus a 154-
    combination technical-indicator search, all under identical realistic
    constraints: Zerodha's actual intraday cost stack, liquidity-scaled
    slippage, real ~15:12 IST square-off timing, a 4-concurrent-position
    capital cap, 1 trade/stock/day, liquidity-priority when signals
    compete for a capped slot, and a top-200-liquid-stock universe.
  - Under those constraints, momentum_volume_breakout (LONG side only) was
    the sole candidate positive across train (+19.7% CAGR, Sharpe 1.20),
    full-period (+13.9%, 0.94), AND out-of-sample test (+5.7%, 0.46).
  - The 154-combo search's best validation performer (macd_cross +
    vwap_alignment, validation Sharpe 1.91) collapsed on the untouched
    holdout (Sharpe -8.50) — a clean overfitting signature. Nothing in
    that search beat momentum_volume_breakout's own out-of-sample
    consistency, so it was not adopted.

Known weakness, stated plainly: this strategy's CAGR drops to -16.2% if
transaction costs run 2x higher than modeled (costs.py) — the edge is real
in this backtest but not large relative to execution-cost uncertainty.
Treat this as the best-supported candidate from this research, not a
guaranteed live edge. Recommended next step before real capital: paper-
trade it forward for a period and confirm the edge persists out beyond
this backtest's holdout window too.
"""

import pandas as pd

from strategies import momentum_volume_breakout
from engine import apply_costs
from trade_caps import apply_concurrency_cap, restrict_to_liquid_universe, risk_based_daily_cap

STRATEGY_FN = momentum_volume_breakout
DIRECTION = 1  # long only

UNIVERSE_TOP_N = 200
MAX_CONCURRENT_POSITIONS = 4
MAX_TRADES_PER_SYMBOL_PER_DAY = 1
MAX_TRADES_PER_DAY = risk_based_daily_cap(capital=1.0, risk_per_trade_pct=0.5, daily_risk_budget_pct=3.0)


def generate_signals(bars: dict, rank: dict, as_of: pd.Timestamp = None) -> pd.DataFrame:
    """bars: {symbol: OHLCV DataFrame}. rank: {symbol: liquidity_rank}.
    as_of: if given, each symbol's bars are truncated to <= as_of first —
    this is what makes the function safe to call intraday, mid-session, on
    live-updating data: it only ever sees bars up to "now", so it can never
    look ahead. Returns the trade/signal list actually to be taken, after
    every real-world constraint (universe, concurrency, per-symbol/day
    caps, liquidity priority, entry-cutoff/square-off).
    """
    universe_size = len(rank) or len(bars)
    all_trades = {}

    for symbol, df in bars.items():
        if as_of is not None:
            df = df[df.index <= as_of]
        if df.empty:
            continue

        trades = STRATEGY_FN(df)
        if trades.empty:
            continue
        trades = trades[trades["direction"] == DIRECTION]
        if trades.empty:
            continue

        r = rank.get(symbol, universe_size)
        trades = apply_costs(trades, liquidity_rank=r, universe_size=universe_size)
        all_trades[symbol] = trades

    all_trades = restrict_to_liquid_universe(all_trades, rank, UNIVERSE_TOP_N)
    all_trades = apply_concurrency_cap(all_trades, MAX_CONCURRENT_POSITIONS,
                                        MAX_TRADES_PER_DAY, MAX_TRADES_PER_SYMBOL_PER_DAY, rank=rank)

    frames = []
    for symbol, trades in all_trades.items():
        t = trades.copy()
        t.insert(0, "symbol", symbol)
        frames.append(t)

    if not frames:
        return pd.DataFrame(columns=["symbol", "date", "direction", "entry_time", "entry_price",
                                      "exit_time", "exit_price", "exit_reason", "gross_return", "net_return"])

    result = pd.concat(frames, ignore_index=True)
    return result.sort_values("entry_time").reset_index(drop=True)
