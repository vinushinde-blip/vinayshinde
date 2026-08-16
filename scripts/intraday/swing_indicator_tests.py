"""
Tests each new indicator/condition individually against the RS+regime
baseline, on the full 500-stock 5-year backtest — not a combinatorial
grid search. Each row changes exactly ONE thing from the baseline, so the
effect is attributable and interpretable, and isolated results avoid the
"searched a lot of combos, best one is noise" trap the intraday side of
this project already ran into once.

Usage:
    python3 swing_indicator_tests.py
"""

import pandas as pd

from swing_backtest import (load_bars_and_rank, load_nifty_daily, generate_all_trades,
                             apply_portfolio_caps, apply_costs_to_trades, build_daily_mtm_returns,
                             performance_metrics, walk_forward_split,
                             MAX_CONCURRENT_POSITIONS, MAX_NEW_SIGNALS_PER_DAY)

CONFIGS = {
    "baseline (RS + regime)": {},
    "+ regime_confirm_days=5 (anti-whipsaw)": {"regime_confirm_days": 5},
    "+ regime_confirm_days=10": {"regime_confirm_days": 10},
    "+ ADX trend-strength filter": {"use_adx_filter": True},
    "+ MACD confirmation": {"use_macd_filter": True},
    "+ volatility ceiling (ATR<5%)": {"use_volatility_filter": True},
    "+ anchored VWAP exit": {"use_vwap_exit": True},
    "no RS, no regime (for reference)": {"use_relative_strength": False, "use_market_regime": False},
}


def run_config(name, kwargs, daily_by_symbol, rank, nifty_daily, universe_size):
    raw = generate_all_trades(daily_by_symbol, rank, nifty_daily, **kwargs)
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

    return {
        "config": name, "trades": len(costed), "win_rate": f"{win_rate:.1f}%",
        "full_CAGR": full["CAGR"], "full_Sharpe": full["Sharpe"], "full_MaxDD": full["MaxDrawdown"],
        "train_CAGR": train_m["CAGR"], "test_CAGR": test_m["CAGR"], "test_Sharpe": test_m["Sharpe"],
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
    print("=== Comparison: each row changes ONE thing from the baseline ===")
    print(table.to_string())
    table.to_csv("../../data/daily/swing_indicator_comparison.csv")
    print("\nSaved: ../../data/daily/swing_indicator_comparison.csv")


if __name__ == "__main__":
    main()
