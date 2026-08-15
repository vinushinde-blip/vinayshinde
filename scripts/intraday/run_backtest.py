"""
Runs every strategy in strategies.STRATEGIES across the full fetched
universe (data/intraday/bars/*.parquet), long+short, with realistic costs
applied per-symbol based on that symbol's liquidity rank. Reports full-period
and walk-forward (train/test) metrics for each, plus a cost-sensitivity
check (results at 1x, 2x, 3x the assumed cost model) so we can see how much
of any edge is real vs. a costs assumption artifact.

Usage:
    python3 run_backtest.py
    python3 run_backtest.py --strategy opening_range_breakout
"""

import argparse
import glob
import os

import pandas as pd

from strategies import STRATEGIES
from engine import apply_costs, portfolio_daily_returns, equity_curve
from metrics import performance_metrics, walk_forward_split

BARS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "intraday", "bars")
UNIVERSE_FILE = os.path.join(os.path.dirname(__file__), "..", "..", "data", "intraday", "universe_top500.csv")
RESULTS_FILE = os.path.join(os.path.dirname(__file__), "..", "..", "data", "intraday", "backtest_results.csv")


def load_universe_rank() -> dict:
    if not os.path.exists(UNIVERSE_FILE):
        return {}
    df = pd.read_csv(UNIVERSE_FILE)
    return dict(zip(df["tradingsymbol"], df["liquidity_rank"]))


def load_bars() -> dict:
    rank = load_universe_rank()
    bars = {}
    for path in sorted(glob.glob(os.path.join(BARS_DIR, "*.parquet"))):
        symbol = os.path.splitext(os.path.basename(path))[0]
        bars[symbol] = pd.read_parquet(path)
    return bars, rank


def run_one_strategy(strat_name, strat_fn, bars: dict, rank: dict, universe_size: int,
                      cost_multiplier: float = 1.0):
    all_trades = {}
    for symbol, df in bars.items():
        trades = strat_fn(df)
        if trades.empty:
            continue
        r = rank.get(symbol, universe_size)
        trades = apply_costs(trades, liquidity_rank=r, universe_size=universe_size)
        if cost_multiplier != 1.0:
            gross = trades["gross_return"]
            base_cost = gross - trades["net_return"]
            trades["net_return"] = gross - base_cost * cost_multiplier
        all_trades[symbol] = trades

    daily = portfolio_daily_returns(all_trades)
    total_trades = sum(len(t) for t in all_trades.values())
    long_trades = sum((t["direction"] == 1).sum() for t in all_trades.values())
    short_trades = sum((t["direction"] == -1).sum() for t in all_trades.values())

    return daily, total_trades, long_trades, short_trades


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", choices=list(STRATEGIES), default=None)
    args = parser.parse_args()

    bars, rank = load_bars()
    if not bars:
        raise SystemExit(f"No bar data found in {BARS_DIR}. Run kite_fetch_intraday.py first.")

    universe_size = max(len(bars), len(rank) or 1)
    strategies_to_run = {args.strategy: STRATEGIES[args.strategy]} if args.strategy else STRATEGIES

    rows = []
    for strat_name, strat_fn in strategies_to_run.items():
        daily, n_trades, n_long, n_short = run_one_strategy(strat_name, strat_fn, bars, rank, universe_size)
        if daily.empty:
            print(f"{strat_name}: no trades generated, skipping")
            continue

        full = performance_metrics(daily)
        train, test = walk_forward_split(daily)
        train_m = performance_metrics(train)
        test_m = performance_metrics(test)

        # cost sensitivity: how fragile is this to costs being underestimated?
        daily_2x, _, _, _ = run_one_strategy(strat_name, strat_fn, bars, rank, universe_size, cost_multiplier=2.0)
        cagr_2x = performance_metrics(daily_2x).get("CAGR")

        rows.append({
            "strategy": strat_name,
            "trades": n_trades, "long": n_long, "short": n_short,
            "full_CAGR": full["CAGR"], "full_Sharpe": full["Sharpe"], "full_MaxDD": full["MaxDrawdown"],
            "train_CAGR": train_m["CAGR"], "train_Sharpe": train_m["Sharpe"],
            "test_CAGR (out-of-sample, what matters most)": test_m["CAGR"],
            "test_Sharpe": test_m["Sharpe"], "test_MaxDD": test_m["MaxDrawdown"],
            "CAGR_at_2x_costs": cagr_2x,
        })

    if not rows:
        raise SystemExit("No strategy produced any trades on this data.")

    table = pd.DataFrame(rows).set_index("strategy")
    print(table.to_string())
    os.makedirs(os.path.dirname(RESULTS_FILE), exist_ok=True)
    table.to_csv(RESULTS_FILE)
    print(f"\nSaved: {RESULTS_FILE}")
    print("\nJudge strategies on the out-of-sample (test) columns, not the full-period "
          "or train columns — those can look good purely from fitting to that data.")


if __name__ == "__main__":
    main()
