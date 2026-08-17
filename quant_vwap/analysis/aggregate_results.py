"""
Aggregates the per-symbol crossing/behavior/backtest results produced by
run_pipeline.py into universe-level summary tables for the report:
  - crossing counts by timeframe/threshold/direction (and per-symbol)
  - behavioral summaries sliced by each indicator bucket and by Nifty regime
  - backtest comparison across strategy_variant x timeframe x threshold x
    direction x target:stop
Writes CSVs under results/summary/.
"""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from behavior import summarize_events, HORIZONS
from backtest import summarize_trades

REPO_ROOT = "/home/user/vinayshinde/quant_vwap"
RESULTS_DIR = f"{REPO_ROOT}/results"
OUT_DIR = f"{RESULTS_DIR}/summary"
os.makedirs(OUT_DIR, exist_ok=True)


def main():
    cc = pd.read_parquet(f"{RESULTS_DIR}/crossing_counts_by_symbol.parquet")
    events = pd.read_parquet(f"{RESULTS_DIR}/behavior_events.parquet")
    trades = pd.read_parquet(f"{RESULTS_DIR}/backtest_trades.parquet")

    print(f"crossing rows: {len(cc)}, events: {len(events)}, trades: {len(trades)}")
    n_symbols = trades["symbol"].nunique()
    print(f"symbols covered: {n_symbols}")

    # --- 1. Crossing counts ---
    universe_counts = (
        cc.melt(id_vars=["threshold", "symbol", "timeframe"], value_vars=["above", "below"],
                var_name="direction", value_name="count")
        .groupby(["timeframe", "threshold", "direction"])["count"].sum()
        .reset_index()
        .sort_values(["timeframe", "threshold", "direction"])
    )
    universe_counts.to_csv(f"{OUT_DIR}/crossing_counts_universe.csv", index=False)

    per_symbol_totals = (
        cc.melt(id_vars=["threshold", "symbol", "timeframe"], value_vars=["above", "below"],
                var_name="direction", value_name="count")
        .groupby(["symbol", "timeframe"])["count"].sum()
        .reset_index()
        .sort_values("count", ascending=False)
    )
    per_symbol_totals.to_csv(f"{OUT_DIR}/crossing_counts_top_symbols.csv", index=False)

    # --- 2. Behavioral summaries ---
    summarize_events(events, ["timeframe", "threshold", "direction"]).to_csv(
        f"{OUT_DIR}/behavior_by_threshold.csv", index=False)
    summarize_events(events, ["timeframe", "threshold", "direction", "rsi_bucket"]).to_csv(
        f"{OUT_DIR}/behavior_by_rsi.csv", index=False)
    summarize_events(events, ["timeframe", "threshold", "direction", "adx_bucket"]).to_csv(
        f"{OUT_DIR}/behavior_by_adx.csv", index=False)
    summarize_events(events, ["timeframe", "threshold", "direction", "vol_bucket"]).to_csv(
        f"{OUT_DIR}/behavior_by_volume.csv", index=False)
    summarize_events(events, ["timeframe", "threshold", "direction", "bb_bucket"]).to_csv(
        f"{OUT_DIR}/behavior_by_bollinger.csv", index=False)
    summarize_events(events, ["timeframe", "threshold", "direction", "macd_sign"]).to_csv(
        f"{OUT_DIR}/behavior_by_macd.csv", index=False)
    summarize_events(events, ["timeframe", "threshold", "direction", "nifty_regime"]).to_csv(
        f"{OUT_DIR}/behavior_by_nifty_regime.csv", index=False)

    # --- 3. Backtest comparison ---
    full = summarize_trades(
        trades, ["strategy_variant", "timeframe", "threshold", "direction", "target_pct", "stop_pct"]
    )
    full.to_csv(f"{OUT_DIR}/backtest_full_comparison.csv", index=False)

    by_variant_tf = summarize_trades(trades, ["strategy_variant", "timeframe", "target_pct", "stop_pct"])
    by_variant_tf.to_csv(f"{OUT_DIR}/backtest_by_variant_timeframe.csv", index=False)

    top_level = summarize_trades(trades, ["strategy_variant", "timeframe"])
    top_level.to_csv(f"{OUT_DIR}/backtest_top_level.csv", index=False)

    by_direction = summarize_trades(trades, ["strategy_variant", "timeframe", "direction", "target_pct", "stop_pct"])
    by_direction.to_csv(f"{OUT_DIR}/backtest_by_direction.csv", index=False)

    by_nifty_regime = summarize_trades(
        trades, ["strategy_variant", "timeframe", "nifty_regime_at_signal", "target_pct", "stop_pct"]
    )
    by_nifty_regime.to_csv(f"{OUT_DIR}/backtest_by_nifty_regime.csv", index=False)

    # Per-symbol leaderboard for the best overall variant (useful sanity check)
    per_symbol = summarize_trades(trades, ["strategy_variant", "symbol", "timeframe"])
    per_symbol.to_csv(f"{OUT_DIR}/backtest_per_symbol.csv", index=False)

    print("Wrote summary CSVs to", OUT_DIR)
    print("\n=== TOP LEVEL (strategy_variant x timeframe) ===")
    print(top_level.to_string())


if __name__ == "__main__":
    main()
