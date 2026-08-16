"""
Backtests the oversold-reversion strategy (swing_reversion_strategy.py)
across the full 500-stock, 10-year universe, in three bounded configs —
same "change one thing from a baseline" discipline as
swing_indicator_tests.py, not a combinatorial search:

  1. pure reversion   - exactly what the price-action study found (no
                         trend or regime guard at all)
  2. + uptrend context - only buy the dip if the stock is still above its
                          own 200-day SMA (guard against falling knives)
  3. + market regime   - (2) plus the Nifty-above-its-own-200SMA filter
                          used in the momentum strategy, to see if it
                          helps a mean-reversion entry the way it helped
                          the breakout entry, or hurts it (a reversion
                          strategy's whole point is often to buy exactly
                          when the broad market looks weakest)

Reuses all of swing_backtest.py's cost/cap/MTM/walk-forward plumbing
unchanged, so results are directly comparable to the breakout and
pullback strategies already tested.

Usage:
    python3 swing_reversion_tests.py
"""

import pandas as pd

from swing_reversion_strategy import find_reversion_trades
from swing_backtest import (load_bars_and_rank, load_nifty_daily, apply_portfolio_caps,
                             apply_costs_to_trades, build_daily_mtm_returns, performance_metrics,
                             walk_forward_split, MAX_CONCURRENT_POSITIONS, MAX_NEW_SIGNALS_PER_DAY)

CONFIGS = {
    "pure reversion (RSI+pullback+off-highs+vol, no trend guard)": {},
    "+ uptrend context (above 200SMA)": {"uptrend_context": True},
    "+ uptrend context + market regime": {"uptrend_context": True, "use_market_regime": True},
}


def generate_all_reversion_trades(daily_by_symbol, rank, nifty_daily, **kwargs):
    frames = []
    for symbol, daily in daily_by_symbol.items():
        trades = find_reversion_trades(daily, nifty_daily, **kwargs)
        if trades.empty:
            continue
        trades.insert(0, "symbol", symbol)
        trades.insert(1, "liquidity_rank", rank.get(symbol, len(rank) + 1))
        frames.append(trades)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def run_config(name, kwargs, daily_by_symbol, rank, nifty_daily, universe_size):
    raw = generate_all_reversion_trades(daily_by_symbol, rank, nifty_daily, **kwargs)
    if raw.empty:
        return {"config": name, "trades": 0}
    capped = apply_portfolio_caps(raw, MAX_CONCURRENT_POSITIONS, MAX_NEW_SIGNALS_PER_DAY)
    costed = apply_costs_to_trades(capped, universe_size)
    daily_returns = build_daily_mtm_returns(costed, daily_by_symbol)
    full = performance_metrics(daily_returns)
    train, test = walk_forward_split(daily_returns)
    train_m = performance_metrics(train)
    test_m = performance_metrics(test)
    win_rate = (costed["net_return"] > 0).mean() * 100

    cost_pct_orig = costed["gross_return"] - costed["net_return"]
    costed_2x = capped.copy()
    costed_2x["gross_return"] = costed["gross_return"]
    costed_2x["net_return"] = costed["gross_return"] - cost_pct_orig * 2
    daily_2x = build_daily_mtm_returns(costed_2x, daily_by_symbol)
    m2x = performance_metrics(daily_2x)

    return {
        "config": name, "trades": len(costed), "win_rate": f"{win_rate:.1f}%",
        "full_CAGR": full["CAGR"], "full_Sharpe": full["Sharpe"], "full_MaxDD": full["MaxDrawdown"],
        "train_CAGR": train_m["CAGR"], "test_CAGR": test_m["CAGR"], "test_Sharpe": test_m["Sharpe"],
        "2x_cost_CAGR": m2x["CAGR"], "2x_cost_Sharpe": m2x["Sharpe"],
    }


def main():
    print("Loading data...")
    daily_by_symbol, rank = load_bars_and_rank("daily")
    nifty_daily = load_nifty_daily()
    universe_size = len(rank)
    print(f"Loaded {len(daily_by_symbol)} symbols.\n")

    rows = []
    for name, kwargs in CONFIGS.items():
        print(f"Running: {name} ...")
        row = run_config(name, kwargs, daily_by_symbol, rank, nifty_daily, universe_size)
        rows.append(row)
        print(f"  {row}\n")

    table = pd.DataFrame(rows).set_index("config")
    pd.set_option("display.width", 200)
    print("=== Reversion strategy comparison ===")
    print(table.to_string())
    table.to_csv("../../data/daily/swing_reversion_comparison.csv")
    print("\nSaved: ../../data/daily/swing_reversion_comparison.csv")


if __name__ == "__main__":
    main()
