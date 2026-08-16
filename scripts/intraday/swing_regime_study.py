"""
Regime-conditioned extension of swing_move_study.py. The pooled study
(all 10 years together) mainly ended up describing "what preceded a
clean move during the recent bull recovery," since that period dominates
the launch-day count. This script asks the sharper question: does the
same technical picture actually precede clean moves in EVERY regime, or
does the useful signal change (or vanish) depending on whether NIFTY
itself is trending up, trending down, or chopping?

Method:
  1. Classify every day into a NIFTY trend_regime (bull/bear/transition)
     and vol_regime (high/mid/low) using market_regime.py — objective,
     no look-ahead.
  2. Reuse swing_move_study's forward-looking launch-day labels and
     backward-looking features (same definitions, same TRAIN/TEST date
     split) - only the aggregation changes: effect sizes are computed
     SEPARATELY within each trend_regime bucket, on TRAIN data only.
  3. Also reports the base launch RATE per regime bucket - how much of
     the earlier "regime filter helps" finding is simply "clean moves are
     far more common in a bull regime," independent of any stock-level
     technical setup at all.

Usage:
    python3 swing_regime_study.py
"""

import glob
import os

import numpy as np
import pandas as pd

from market_regime import classify_regime
from swing_move_study import label_launch_days, compute_features, effect_size_report

BARS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "daily", "bars")
NIFTY_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "data", "daily", "NIFTY50.parquet")
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "daily")


def build_dataset_with_regime(daily_by_symbol: dict, nifty_daily: pd.DataFrame, regime: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for symbol, daily in daily_by_symbol.items():
        if len(daily) < 300:
            continue
        label = label_launch_days(daily)
        feats = compute_features(daily, nifty_daily)
        feats["is_launch"] = label
        feats["symbol"] = symbol
        feats["date"] = daily.index
        feats["trend_regime"] = regime["trend_regime"].reindex(daily.index).values
        feats["vol_regime"] = regime["vol_regime"].reindex(daily.index).values
        frames.append(feats)
    return pd.concat(frames, ignore_index=True)


def main():
    print("Loading data...")
    nifty_daily = pd.read_parquet(NIFTY_PATH)
    regime = classify_regime(nifty_daily)
    daily_by_symbol = {}
    for path in sorted(glob.glob(os.path.join(BARS_DIR, "*.parquet"))):
        symbol = os.path.splitext(os.path.basename(path))[0]
        daily_by_symbol[symbol] = pd.read_parquet(path)
    print(f"Loaded {len(daily_by_symbol)} symbols.")
    span = [d for daily in daily_by_symbol.values() for d in daily.index[[0, -1]]]

    print("Labeling launch days, computing features + regime tags across the universe...")
    dataset = build_dataset_with_regime(daily_by_symbol, nifty_daily, regime)
    dataset = dataset[dataset["trend_regime"] != "warmup"].copy()
    print(f"Rows: {len(dataset)}, launch days: {int(dataset['is_launch'].sum())}")

    cutoff = min(span) + (max(span) - min(span)) * 0.7
    train = dataset[dataset["date"] < cutoff].copy()
    test = dataset[dataset["date"] >= cutoff].copy()
    print(f"Train up to {cutoff.date()}, test from {cutoff.date()}\n")

    print("=== Launch RATE by regime (TRAIN period) — how much is just 'being in the right regime' ===")
    rate_table = train.groupby("trend_regime").agg(
        days=("is_launch", "size"), launches=("is_launch", "sum")
    )
    rate_table["launch_rate_pct"] = (rate_table["launches"] / rate_table["days"] * 100).round(3)
    print(rate_table.to_string())
    print()
    vol_rate_table = train.groupby("vol_regime").agg(
        days=("is_launch", "size"), launches=("is_launch", "sum")
    )
    vol_rate_table["launch_rate_pct"] = (vol_rate_table["launches"] / vol_rate_table["days"] * 100).round(3)
    print(vol_rate_table.to_string())

    feature_cols = [c for c in dataset.columns
                    if c not in ("is_launch", "symbol", "date", "trend_regime", "vol_regime")]

    all_reports = []
    for regime_name in ["bull", "bear", "transition"]:
        sub_train = train[train["trend_regime"] == regime_name]
        sub_test = test[test["trend_regime"] == regime_name]
        print(f"\n=== TRAIN-set feature comparison WITHIN regime = {regime_name} "
              f"({len(sub_train)} rows, {int(sub_train['is_launch'].sum())} launches) ===")
        if sub_train["is_launch"].sum() < 30:
            print("  (too few launches in this regime/period to report reliably)")
            continue
        report = effect_size_report(sub_train, feature_cols)
        report.insert(0, "regime", regime_name)
        print(report.to_string(index=False))
        all_reports.append(report)

        test_report = effect_size_report(sub_test, feature_cols) if sub_test["is_launch"].sum() >= 30 else pd.DataFrame()
        if not test_report.empty:
            test_report.insert(0, "regime", regime_name)
            print(f"  -- test-set check for {regime_name} ({len(sub_test)} rows, "
                  f"{int(sub_test['is_launch'].sum())} launches) --")
            print(test_report.head(6).to_string(index=False))

    os.makedirs(OUT_DIR, exist_ok=True)
    if all_reports:
        pd.concat(all_reports, ignore_index=True).to_csv(
            os.path.join(OUT_DIR, "swing_regime_effect_sizes.csv"), index=False)
    dataset.to_parquet(os.path.join(OUT_DIR, "swing_regime_study_dataset.parquet"))
    print(f"\nSaved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
