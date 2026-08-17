"""
Out-of-sample validation for the final recommended strategy (1-minute bars,
+/-4% delta from VWAP both directions, 1.5%:0.5% target:stop): splits the
2-year signal set into year 1 (discovery period) and year 2 (held out) and
backtests each independently against Rs 10,00,000, at both cost scenarios.

Also produces the per-symbol P&L leaderboard and the Nifty-filter effect
pivot (baseline vs nifty_filtered avg P&L for every threshold/direction/
timeframe/target cell) used in the detailed report.
"""
import json
import sys

import pandas as pd

sys.path.insert(0, __file__.rsplit("/", 1)[0])
import portfolio_backtest as pb

REPO_ROOT = "/home/user/vinayshinde/quant_vwap"
SPLIT_DATE = pd.Timestamp("2025-08-17", tz="Asia/Kolkata")


def load_final_strategy_trades():
    trades = pd.read_parquet(f"{REPO_ROOT}/results/backtest_trades.parquet")
    sel = trades[(trades.strategy_variant == "baseline") & (trades.timeframe == "1minute") &
                 (trades.threshold == 4.0) & (trades.target_pct == 1.5) & (trades.stop_pct == 0.5)]
    return sel.sort_values("entry_time").reset_index(drop=True)


def run_oos_split():
    sel = load_final_strategy_trades()
    train = sel[sel.entry_time < SPLIT_DATE]
    test = sel[sel.entry_time >= SPLIT_DATE]

    results = {}
    for label, df in [("train_yr1", train), ("test_yr2", test)]:
        for cost_label, cost in [("optimistic", 0.05), ("realistic", 0.08)]:
            pb.ROUND_TRIP_COST_PCT = cost
            final_eq, trade_log, eq_curve, skipped = pb.simulate(df)
            m = pb.compute_metrics(final_eq, trade_log, eq_curve)
            results[f"{label}_{cost_label}"] = m

    with open(f"{REPO_ROOT}/results/summary/final_strategy_oos_split.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    return results


def run_symbol_leaderboard():
    sel = load_final_strategy_trades()
    by_sym = sel.groupby("symbol").agg(
        n=("pnl_pct", "size"), total_pnl=("pnl_pct", "sum"), avg_pnl=("pnl_pct", "mean"),
        win_rate=("pnl_pct", lambda x: (x > 0).mean() * 100),
    ).reset_index().sort_values("total_pnl", ascending=False)
    by_sym.to_csv(f"{REPO_ROOT}/results/summary/final_strategy_symbol_leaderboard.csv", index=False)
    return by_sym


def run_nifty_filter_pivot():
    full = pd.read_csv(f"{REPO_ROOT}/results/summary/backtest_full_comparison.csv")
    base = full[full.strategy_variant == "baseline"][
        ["timeframe", "threshold", "direction", "target_pct", "stop_pct", "n_trades", "avg_pnl_pct"]]
    filt = full[full.strategy_variant == "nifty_filtered"][
        ["timeframe", "threshold", "direction", "target_pct", "stop_pct", "n_trades", "avg_pnl_pct"]]
    merged = base.merge(filt, on=["timeframe", "threshold", "direction", "target_pct", "stop_pct"],
                         suffixes=("_baseline", "_filtered"))
    merged["delta"] = merged["avg_pnl_pct_filtered"] - merged["avg_pnl_pct_baseline"]
    merged["filter_helps"] = merged["delta"] > 0
    merged = merged.sort_values(["timeframe", "threshold", "direction", "target_pct"])
    merged.to_csv(f"{REPO_ROOT}/results/summary/nifty_filter_effect_full.csv", index=False)
    print(f"Nifty filter helps in {merged['filter_helps'].sum()} of {len(merged)} cells")
    return merged


if __name__ == "__main__":
    oos = run_oos_split()
    print(json.dumps(oos, indent=2, default=str))
    run_symbol_leaderboard()
    run_nifty_filter_pivot()
