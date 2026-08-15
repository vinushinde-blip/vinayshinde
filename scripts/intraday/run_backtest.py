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
from trade_caps import apply_daily_caps, apply_concurrency_cap, restrict_to_liquid_universe, risk_based_daily_cap

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
                      cost_multiplier: float = 1.0, direction: int = None,
                      max_trades_per_day: int = None, max_trades_per_symbol_per_day: int = None,
                      cap_priority: str = "liquidity", max_concurrent_positions: int = None,
                      universe_top_n: int = None):
    """direction: None runs both legs, +1 long-only, -1 short-only (filters
    each symbol's trades before aggregating, so costs/aggregation logic is
    identical to the combined run — only which trades count differs).

    max_trades_per_day / max_trades_per_symbol_per_day: see trade_caps.py —
    without these, "all trades" includes far more signals per day than
    anyone could actually execute (e.g. vwap_mean_reversion fires ~2,000
    signals/day across the 500-stock universe).

    max_concurrent_positions: if set, uses the concurrency-simulating cap
    (apply_concurrency_cap) instead of the flat daily cap — the more
    realistic constraint, since capital frees up when a position closes,
    not just once/day. universe_top_n restricts to that many of the most
    liquid stocks before any other cap runs."""
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
        if direction is not None:
            trades = trades[trades["direction"] == direction]
            if trades.empty:
                continue
        all_trades[symbol] = trades

    if universe_top_n is not None:
        all_trades = restrict_to_liquid_universe(all_trades, rank, universe_top_n)

    if max_concurrent_positions is not None:
        all_trades = apply_concurrency_cap(all_trades, max_concurrent_positions,
                                            max_trades_per_day, max_trades_per_symbol_per_day, rank=rank)
    else:
        all_trades = apply_daily_caps(all_trades, max_trades_per_day, max_trades_per_symbol_per_day,
                                       priority=cap_priority, rank=rank)
    daily = portfolio_daily_returns(all_trades)
    total_trades = sum(len(t) for t in all_trades.values())
    long_trades = sum((t["direction"] == 1).sum() for t in all_trades.values())
    short_trades = sum((t["direction"] == -1).sum() for t in all_trades.values())

    return daily, total_trades, long_trades, short_trades


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", choices=list(STRATEGIES), default=None)
    parser.add_argument("--max-trades-per-day", type=int, default=None,
                         help="cap total trades/day directly (overrides the risk-based calc below if both given)")
    parser.add_argument("--max-trades-per-symbol-per-day", type=int, default=1,
                         help="cap trades/day for any one stock (default: 1 — no re-entering the same name same day)")
    parser.add_argument("--risk-per-trade-pct", type=float, default=0.5,
                         help="max %% of capital risked on a single trade if its stop is hit (default: 0.5%%)")
    parser.add_argument("--daily-risk-budget-pct", type=float, default=3.0,
                         help="max %% of capital at risk in a day before stopping, worst-case (default: 3%%)")
    parser.add_argument("--cap-priority", choices=["liquidity", "chronological"], default="liquidity",
                         help="when more signals fire than the cap allows, which to keep (default: liquidity — "
                              "see trade_caps.py; never realized-return-based, that would be look-ahead bias)")
    parser.add_argument("--max-concurrent-positions", type=int, default=4,
                         help="max positions open at once — the realistic version of the daily cap, since "
                              "capital frees up when a position closes rather than once/day (default: 4; "
                              "pass 0 to fall back to the flat daily-count cap instead)")
    parser.add_argument("--universe-top-n", type=int, default=200,
                         help="restrict to this many of the most liquid stocks before any other cap runs "
                              "(default: 200 of 500 — trading the illiquid tail isn't worth the execution risk; "
                              "pass 0 for no restriction)")
    args = parser.parse_args()
    max_concurrent_positions = args.max_concurrent_positions or None
    universe_top_n = args.universe_top_n or None

    max_trades_per_day = args.max_trades_per_day
    if max_trades_per_day is None:
        max_trades_per_day = risk_based_daily_cap(
            capital=1.0,  # capital cancels out of the ratio; only the two %% args matter for the count
            risk_per_trade_pct=args.risk_per_trade_pct,
            daily_risk_budget_pct=args.daily_risk_budget_pct,
        )
        print(f"No --max-trades-per-day given: using risk-based cap of {max_trades_per_day} trades/day "
              f"({args.daily_risk_budget_pct}% daily risk budget / {args.risk_per_trade_pct}% risk per trade).")

    bars, rank = load_bars()
    if not bars:
        raise SystemExit(f"No bar data found in {BARS_DIR}. Run kite_fetch_intraday.py first.")

    universe_size = max(len(bars), len(rank) or 1)
    strategies_to_run = {args.strategy: STRATEGIES[args.strategy]} if args.strategy else STRATEGIES

    def build_row(label, strat_name, strat_fn, direction):
        daily, n_trades, n_long, n_short = run_one_strategy(
            strat_name, strat_fn, bars, rank, universe_size, direction=direction,
            max_trades_per_day=max_trades_per_day,
            max_trades_per_symbol_per_day=args.max_trades_per_symbol_per_day, cap_priority=args.cap_priority, max_concurrent_positions=max_concurrent_positions, universe_top_n=universe_top_n)
        if daily.empty:
            print(f"{label}: no trades generated, skipping")
            return None

        full = performance_metrics(daily)
        train, test = walk_forward_split(daily)
        train_m = performance_metrics(train)
        test_m = performance_metrics(test)

        daily_2x, *_ = run_one_strategy(
            strat_name, strat_fn, bars, rank, universe_size, cost_multiplier=2.0, direction=direction,
            max_trades_per_day=max_trades_per_day,
            max_trades_per_symbol_per_day=args.max_trades_per_symbol_per_day, cap_priority=args.cap_priority, max_concurrent_positions=max_concurrent_positions, universe_top_n=universe_top_n)
        cagr_2x = performance_metrics(daily_2x).get("CAGR")

        return {
            "strategy": label,
            "trades": n_trades, "long": n_long, "short": n_short,
            "full_CAGR": full["CAGR"], "full_Sharpe": full["Sharpe"], "full_MaxDD": full["MaxDrawdown"],
            "train_CAGR": train_m["CAGR"], "train_Sharpe": train_m["Sharpe"],
            "test_CAGR (out-of-sample, what matters most)": test_m["CAGR"],
            "test_Sharpe": test_m["Sharpe"], "test_MaxDD": test_m["MaxDrawdown"],
            "CAGR_at_2x_costs": cagr_2x,
        }

    rows = []
    for strat_name, strat_fn in strategies_to_run.items():
        row = build_row(strat_name, strat_name, strat_fn, direction=None)
        if row:
            rows.append(row)

        # also break out long-only / short-only legs for strategies that
        # trade both directions, so we can see whether one side alone beats
        # the combined signal (e.g. "is shorting this actually the edge?")
        is_bidirectional = row is not None and row["long"] > 0 and row["short"] > 0
        if is_bidirectional:
            for label, direction in [(f"{strat_name} (long only)", 1), (f"{strat_name} (short only)", -1)]:
                sub_row = build_row(label, strat_name, strat_fn, direction=direction)
                if sub_row:
                    rows.append(sub_row)

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
