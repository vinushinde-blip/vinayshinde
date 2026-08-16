"""
Full backtest of the pullback-entry strategy (swing_pullback_strategy.py)
across the NSE 500 universe — reuses every piece of plumbing from
swing_backtest.py (portfolio caps, delivery costs, mark-to-market daily
returns, walk-forward metrics, rupee capital simulation) unchanged, so
this is a clean, apples-to-apples comparison against the breakout
strategy: only the entry trigger differs.

Usage:
    python3 swing_pullback_backtest.py --capital 1000000
"""

import argparse
import os

import pandas as pd

from swing_pullback_strategy import find_pullback_trades
from swing_backtest import (load_bars_and_rank, load_nifty_daily, apply_portfolio_caps,
                             apply_costs_to_trades, build_daily_mtm_returns, performance_metrics,
                             walk_forward_split, simulate_capital, MAX_CONCURRENT_POSITIONS,
                             MAX_NEW_SIGNALS_PER_DAY)

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "intraday")


def generate_all_pullback_trades(daily_by_symbol: dict, rank: dict, nifty_daily: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for symbol, daily in daily_by_symbol.items():
        trades = find_pullback_trades(daily, nifty_daily)
        if trades.empty:
            continue
        trades.insert(0, "symbol", symbol)
        trades.insert(1, "liquidity_rank", rank.get(symbol, len(rank) + 1))
        frames.append(trades)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--capital", type=float, default=1_000_000)
    args = parser.parse_args()

    daily_by_symbol, rank = load_bars_and_rank("daily")
    if not daily_by_symbol:
        raise SystemExit("No daily bar data found.")
    universe_size = len(rank)
    span = [d for daily in daily_by_symbol.values() for d in daily.index[[0, -1]]]
    print(f"Loaded {len(daily_by_symbol)} symbols, {min(span).date()} to {max(span).date()}.")
    nifty_daily = load_nifty_daily()

    print("Generating pullback trades across the universe...")
    raw_trades = generate_all_pullback_trades(daily_by_symbol, rank, nifty_daily)
    print(f"Raw trades generated (pre-cap): {len(raw_trades)}")

    capped = apply_portfolio_caps(raw_trades, MAX_CONCURRENT_POSITIONS, MAX_NEW_SIGNALS_PER_DAY)
    print(f"After portfolio caps (max {MAX_CONCURRENT_POSITIONS} concurrent, "
          f"{MAX_NEW_SIGNALS_PER_DAY}/day): {len(capped)}")

    costed = apply_costs_to_trades(capped, universe_size)

    print("\n=== Trade-level stats ===")
    win_rate = (costed["net_return"] > 0).mean() * 100
    print(f"Total trades: {len(costed)}")
    print(f"Win rate: {win_rate:.1f}%")
    print(f"Avg net return/trade: {costed['net_return'].mean()*100:.2f}%")
    print(f"Exit reason breakdown:\n{costed['exit_reason'].value_counts().to_string()}")

    print("\n=== Percentage-return backtest (walk-forward, mark-to-market) ===")
    daily_returns = build_daily_mtm_returns(costed, daily_by_symbol)
    full_m = performance_metrics(daily_returns)
    train, test = walk_forward_split(daily_returns)
    train_m = performance_metrics(train)
    test_m = performance_metrics(test)
    print(f"Full period:  {full_m}")
    print(f"Train (70%):  {train_m}")
    print(f"Test (30%, out-of-sample): {test_m}")

    cost_pct_orig = costed["gross_return"] - costed["net_return"]
    costed_2x = capped.copy()
    costed_2x["gross_return"] = costed["gross_return"]
    costed_2x["net_return"] = costed["gross_return"] - cost_pct_orig * 2
    daily_2x = build_daily_mtm_returns(costed_2x, daily_by_symbol)
    print(f"At 2x costs:  {performance_metrics(daily_2x)}")

    print(f"\n=== Rupee capital simulation (Rs {args.capital:,.0f} starting) ===")
    equity, trade_log = simulate_capital(daily_returns, costed, args.capital)
    final_capital = equity["capital"].iloc[-1]
    n_years = max((pd.Timestamp(equity["date"].iloc[-1]) - pd.Timestamp(equity["date"].iloc[0])).days / 365.25, 1/365.25)
    cagr = (final_capital / args.capital) ** (1 / n_years) - 1
    running_max = equity["capital"].cummax()
    dd = (equity["capital"] - running_max) / running_max
    print(f"Starting capital: Rs {args.capital:,.0f}")
    print(f"Final capital: Rs {final_capital:,.0f}")
    print(f"Total return: {(final_capital/args.capital - 1)*100:+.2f}%")
    print(f"CAGR: {cagr*100:+.2f}%")
    print(f"Max drawdown: {dd.min()*100:.2f}% (Rs {(equity['capital']-running_max).min():,.0f})")

    os.makedirs(OUT_DIR, exist_ok=True)
    costed.to_csv(os.path.join(OUT_DIR, "swing_pullback_trades.csv"), index=False)
    equity.to_csv(os.path.join(OUT_DIR, "swing_pullback_equity_curve.csv"), index=False)
    print(f"\nSaved: {OUT_DIR}/swing_pullback_trades.csv, swing_pullback_equity_curve.csv")


if __name__ == "__main__":
    main()
