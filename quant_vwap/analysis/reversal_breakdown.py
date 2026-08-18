"""
Session-unbounded reversal breakdown for delta% crossings beyond +/-2% of VWAP,
on 1-minute and 5-minute bars, across all NSE500 stocks over the 2-year window.

Unlike the fixed-horizon (5/10/20/40-bar) behavioral analysis elsewhere in this
project, this looks forward from each crossing all the way to the END OF THAT
SESSION (VWAP resets daily, so "reversal" can only be evaluated within the same
day) and classifies each crossing event into exactly one of four buckets:

  A = total crossings beyond +/-2%   (every event; B+C1+C05+D partitions it)
  B = full reversion: delta touches/crosses 0 (the VWAP line) before session end
  C1 = partial reversion of >=1.0 percentage point (not full)
  C05 = partial reversion of >=0.5 but <1.0 percentage point (not full)
  D = no meaningful reversal: pulled back <0.5 percentage point all session
"""
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from features import build_feature_frame

REPO_ROOT = "/home/user/vinayshinde/quant_vwap"
RAW_DIR = f"{REPO_ROOT}/data_cache/raw"
MAP_PATH = f"{REPO_ROOT}/data_cache/nse500_instrument_map.csv"
THRESHOLD = 2.0
TIMEFRAMES = {"1minute": "1minute", "5minute": "5minute"}


def session_last_idx(session: np.ndarray) -> np.ndarray:
    """For each row, the positional index of the last row in the same session."""
    n = len(session)
    is_last = np.empty(n, dtype=bool)
    is_last[:-1] = session[1:] != session[:-1]
    is_last[-1] = True
    last_positions = np.where(is_last)[0]
    # for each i, find the first last_position >= i
    insert_at = np.searchsorted(last_positions, np.arange(n), side="left")
    return last_positions[insert_at]


def classify_symbol(df: pd.DataFrame) -> pd.DataFrame:
    feat = build_feature_frame(df)
    n = len(feat)
    delta = feat["delta_pct"].to_numpy()
    session = feat["session"].to_numpy()
    sess_last = session_last_idx(session)

    prev_delta = np.roll(delta, 1)
    prev_session = np.roll(session, 1)
    first_of_session = session != prev_session

    rows = []
    for direction, cond in (
        ("above", (prev_delta < THRESHOLD) & (delta >= THRESHOLD) & (~first_of_session)),
        ("below", (prev_delta > -THRESHOLD) & (delta <= -THRESHOLD) & (~first_of_session)),
    ):
        idx_arr = np.where(cond)[0]
        for idx in idx_arr:
            entry_delta = delta[idx]
            entry_abs = abs(entry_delta)
            end_idx = sess_last[idx]
            if idx + 1 > end_idx:
                rows.append((direction, entry_abs, "D"))  # crossed on last bar of session
                continue

            window = delta[idx + 1: end_idx + 1]
            if direction == "above":
                full_reversion = (window <= 0).any()
            else:
                full_reversion = (window >= 0).any()

            if full_reversion:
                rows.append((direction, entry_abs, "B"))
                continue

            min_abs_after = np.abs(window).min()
            retrace = entry_abs - min_abs_after
            if retrace >= 1.0:
                cat = "C1"
            elif retrace >= 0.5:
                cat = "C05"
            else:
                cat = "D"
            rows.append((direction, entry_abs, cat))

    return pd.DataFrame(rows, columns=["direction", "entry_abs_delta", "category"])


def main():
    inst = pd.read_csv(MAP_PATH)
    symbols = inst["Symbol"].tolist()

    all_results = []
    for tf_label, subdir in TIMEFRAMES.items():
        for i, symbol in enumerate(symbols):
            path = f"{RAW_DIR}/{subdir}/{symbol}.parquet"
            try:
                df = pd.read_parquet(path)
            except FileNotFoundError:
                continue
            if len(df) < 50:
                continue
            res = classify_symbol(df)
            if res.empty:
                continue
            res["symbol"] = symbol
            res["timeframe"] = tf_label
            all_results.append(res)
            if (i + 1) % 100 == 0:
                print(f"{tf_label}: {i+1}/{len(symbols)} symbols processed")

    full = pd.concat(all_results, ignore_index=True)
    full.to_parquet(f"{REPO_ROOT}/results/reversal_breakdown_events.parquet", index=False)

    summary = full.groupby(["timeframe", "direction", "category"]).size().unstack(fill_value=0)
    summary["A_total"] = summary.sum(axis=1)
    summary = summary.reindex(columns=["A_total", "B", "C1", "C05", "D"], fill_value=0)
    summary.to_csv(f"{REPO_ROOT}/results/summary/reversal_breakdown_2pct.csv")
    print("\n=== Reversal breakdown, delta > 2% crossings ===")
    print(summary)

    combined = full.groupby(["timeframe", "category"]).size().unstack(fill_value=0)
    combined["A_total"] = combined.sum(axis=1)
    combined = combined.reindex(columns=["A_total", "B", "C1", "C05", "D"], fill_value=0)
    combined.to_csv(f"{REPO_ROOT}/results/summary/reversal_breakdown_2pct_combined.csv")
    print("\n=== Combined (both directions) ===")
    print(combined)


if __name__ == "__main__":
    main()
