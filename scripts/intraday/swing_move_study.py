"""
Data-driven price-action study: instead of hand-picking indicators to test
one at a time, let the historical data itself say what actually precedes a
genuinely successful swing move, across the whole 500-stock universe and
now up to 10 years of history (2016-2026 — spans the 2018 correction, the
2020 COVID crash/recovery, the 2021 bull run, the 2022 bear market, and
the 2023-25 bull/2025 correction, far more regime diversity than the
earlier 5-year window).

Method (deliberately descriptive first, predictive second — same
look-ahead discipline used throughout this project):

1. LABEL "launch days" using FORWARD information (fwd_days ahead) — this
   is allowed because it's used only to describe history, never as a
   trading rule. A launch day is one where price goes on to gain at least
   min_gain within fwd_days trading days, WITHOUT dipping more than
   max_drawdown below the launch close at any point in that window (a
   clean, sustained move — not a spike that round-trips down again).
   Overlapping launch days from the same underlying move are de-duplicated
   (keep only the first day of each cluster).

2. COMPUTE FEATURES using only information available AT THE CLOSE of the
   launch day (backward-looking only — legitimate for real trading use):
   momentum, volatility contraction, volume behavior, distance from
   52-week high, trend position, RS vs Nifty, ADX, RSI, narrow-range/
   higher-low counts.

3. SPLIT BY DATE (not randomly) into train (first 70% of the calendar
   span) and test (last 30%) — same walk-forward discipline as the rest
   of this project.

4. On TRAIN ONLY, compare each feature's distribution on launch days vs.
   all other (control) days, ranked by standardized effect size (Cohen's
   d). This is pure description — "what did successful moves actually
   look like beforehand" — not yet a trading rule, so there is no
   look-ahead risk in this step even though it uses the forward-looking
   label, because nothing here is applied prospectively.

5. Only the top features (by train-set effect size) get turned into an
   actual composite entry filter and re-tested as real trades entirely
   within the TEST period via swing_backtest.py's existing cost/cap/MTM
   machinery — the genuine out-of-sample check.

Usage:
    python3 swing_move_study.py
"""

import glob
import os

import numpy as np
import pandas as pd

from indicators import rsi, atr, adx, sma

BARS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "daily", "bars")
NIFTY_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "data", "daily", "NIFTY50.parquet")
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "daily")

FWD_DAYS = 15
MIN_GAIN = 0.15
MAX_DRAWDOWN = 0.08


def label_launch_days(daily: pd.DataFrame) -> pd.Series:
    close = daily["close"]
    n = len(close)
    fwd_max_gain = pd.Series(np.nan, index=daily.index)
    fwd_worst_dip = pd.Series(np.nan, index=daily.index)
    close_vals = close.values
    for i in range(n - 1):
        window = close_vals[i + 1: i + 1 + FWD_DAYS]
        if len(window) == 0:
            continue
        fwd_max_gain.iloc[i] = window.max() / close_vals[i] - 1
        fwd_worst_dip.iloc[i] = window.min() / close_vals[i] - 1

    is_launch = (fwd_max_gain >= MIN_GAIN) & (fwd_worst_dip >= -MAX_DRAWDOWN)
    is_launch = is_launch.fillna(False)

    # de-duplicate: drop a launch day if one was already flagged within the
    # preceding FWD_DAYS (same underlying move, not a second independent one)
    already_recent = is_launch.shift(1).rolling(FWD_DAYS, min_periods=1).max().fillna(0).astype(bool)
    return is_launch & ~already_recent


def compute_features(daily: pd.DataFrame, nifty_daily: pd.DataFrame) -> pd.DataFrame:
    close, high, low, vol = daily["close"], daily["high"], daily["low"], daily["volume"]

    sma50 = sma(close, 50)
    sma200 = sma(close, 200)
    atr14 = atr(daily, 14)
    range20_high = high.rolling(20).max()
    range20_low = low.rolling(20).min()
    daily_range = high - low
    nr_median20 = daily_range.rolling(20).median()

    nifty_close = nifty_daily["close"].reindex(daily.index).ffill()

    f = pd.DataFrame(index=daily.index)
    f["ret_5"] = close.pct_change(5)
    f["ret_10"] = close.pct_change(10)
    f["ret_20"] = close.pct_change(20)
    f["atr_pct"] = atr14 / close
    f["range_compression_20"] = (range20_high - range20_low) / close
    f["vol_ratio_5_20"] = vol.rolling(5).mean() / vol.rolling(20).mean()
    f["dist_from_252d_high"] = close / high.rolling(252).max() - 1
    f["pct_above_sma50"] = close / sma50 - 1
    f["pct_above_sma200"] = close / sma200 - 1
    f["sma50_rising"] = (sma50 > sma50.shift(10)).astype(float)
    f["rsi14"] = rsi(close, 14)
    f["adx14"] = adx(daily, 14)
    f["rs_20"] = close.pct_change(20) - nifty_close.pct_change(20)
    f["rs_50"] = close.pct_change(50) - nifty_close.pct_change(50)
    f["narrow_range_count_10"] = (daily_range < nr_median20).rolling(10).sum()
    f["higher_low_count_10"] = (low > low.shift(1)).rolling(10).sum()
    return f


def build_dataset(daily_by_symbol: dict, nifty_daily: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for symbol, daily in daily_by_symbol.items():
        if len(daily) < 300:
            continue
        label = label_launch_days(daily)
        feats = compute_features(daily, nifty_daily)
        feats["is_launch"] = label
        feats["symbol"] = symbol
        feats["date"] = daily.index
        frames.append(feats)
    return pd.concat(frames, ignore_index=True)


def effect_size_report(df: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
    launch = df[df["is_launch"]]
    control = df[~df["is_launch"]]
    rows = []
    for col in feature_cols:
        l, c = launch[col].dropna(), control[col].dropna()
        if len(l) < 30 or len(c) < 30:
            continue
        pooled_std = np.sqrt((l.var() + c.var()) / 2)
        d = (l.mean() - c.mean()) / pooled_std if pooled_std > 0 else np.nan
        rows.append({
            "feature": col, "launch_mean": l.mean(), "control_mean": c.mean(),
            "launch_median": l.median(), "control_median": c.median(),
            "cohens_d": d, "n_launch": len(l), "n_control": len(c),
        })
    report = pd.DataFrame(rows).sort_values("cohens_d", key=lambda s: s.abs(), ascending=False)
    return report.reset_index(drop=True)


def main():
    print("Loading data...")
    nifty_daily = pd.read_parquet(NIFTY_PATH)
    daily_by_symbol = {}
    for path in sorted(glob.glob(os.path.join(BARS_DIR, "*.parquet"))):
        symbol = os.path.splitext(os.path.basename(path))[0]
        daily_by_symbol[symbol] = pd.read_parquet(path)
    print(f"Loaded {len(daily_by_symbol)} symbols.")
    span = [d for daily in daily_by_symbol.values() for d in daily.index[[0, -1]]]
    print(f"Universe span: {min(span).date()} to {max(span).date()}")

    print("Labeling launch days and computing features (this loops over the full universe)...")
    dataset = build_dataset(daily_by_symbol, nifty_daily)
    print(f"Rows: {len(dataset)}, launch days: {int(dataset['is_launch'].sum())} "
          f"({dataset['is_launch'].mean()*100:.2f}% of rows)")

    cutoff = min(span) + (max(span) - min(span)) * 0.7
    train = dataset[dataset["date"] < cutoff].copy()
    test = dataset[dataset["date"] >= cutoff].copy()
    print(f"Train: {len(train)} rows ({int(train['is_launch'].sum())} launches) up to {cutoff.date()}")
    print(f"Test:  {len(test)} rows ({int(test['is_launch'].sum())} launches) from {cutoff.date()}")

    feature_cols = [c for c in dataset.columns if c not in ("is_launch", "symbol", "date")]

    print("\n=== TRAIN-set feature comparison: launch days vs. control days ===")
    print("(ranked by |Cohen's d| — standardized effect size; >0.2 small, >0.5 medium, >0.8 large)")
    train_report = effect_size_report(train, feature_cols)
    pd.set_option("display.width", 160)
    pd.set_option("display.max_columns", 10)
    print(train_report.to_string(index=False))

    print("\n=== TEST-set feature comparison (for reference — NOT used to pick features) ===")
    test_report = effect_size_report(test, feature_cols)
    print(test_report.to_string(index=False))

    os.makedirs(OUT_DIR, exist_ok=True)
    dataset.to_parquet(os.path.join(OUT_DIR, "swing_move_study_dataset.parquet"))
    train_report.to_csv(os.path.join(OUT_DIR, "swing_move_study_train_effect_sizes.csv"), index=False)
    test_report.to_csv(os.path.join(OUT_DIR, "swing_move_study_test_effect_sizes.csv"), index=False)
    print(f"\nSaved dataset and effect-size reports to {OUT_DIR}/")


if __name__ == "__main__":
    main()
