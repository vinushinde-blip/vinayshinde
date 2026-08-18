"""
Z-score-based entry (idea #1/#5 combined), with two optional filter layers
stacked on top: VWAP-slope-flat (idea #7) and sector/index co-movement
(idea #3). Each layer is backtested as an independent signal set (not a
post-hoc filter of the same trade list) so the no-overlapping-position
cursor logic is correct for each layer, same approach as the nifty_filtered
comparison in the main study.

Layers:
  L1 "zscore_only"      : |delta_zscore| crosses >= Z_THRESHOLD
  L2 "+slope_flat"       : L1 AND |vwap_slope_pct| <= its timeframe's cutoff
  L3 "+slope+sector"     : L2 AND sector index not opposing the trade direction
"""
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from features import build_feature_frame
from features_v2 import build_v2_features
from backtest import _simulate, _session_last_idx, summarize_trades
from sector_context import load_all_sector_regimes, stock_sector_index

REPO_ROOT = "/home/user/vinayshinde/quant_vwap"
RAW_DIR = f"{REPO_ROOT}/data_cache/raw"
MAP_PATH = f"{REPO_ROOT}/data_cache/nse500_instrument_map.csv"

TIMEFRAMES = {"1minute": "1minute", "5minute": "5minute"}
Z_THRESHOLD = 2.5
SLOPE_CUTOFF = {"1minute": 0.02, "5minute": 0.09}  # ~75th pct of |vwap_slope_pct|, from a 20-stock sample
TARGET_STOP_VARIANTS = [(1.0, 0.5), (1.5, 0.5)]


def zscore_entry_events(z: np.ndarray, session: np.ndarray):
    prev_z = np.roll(z, 1)
    prev_session = np.roll(session, 1)
    first_of_session = session != prev_session
    valid = ~np.isnan(z) & ~np.isnan(prev_z)

    up = np.where((prev_z < Z_THRESHOLD) & (z >= Z_THRESHOLD) & (~first_of_session) & valid)[0]
    down = np.where((prev_z > -Z_THRESHOLD) & (z <= -Z_THRESHOLD) & (~first_of_session) & valid)[0]
    return [(Z_THRESHOLD, "above", up), (Z_THRESHOLD, "below", down)]


def run_symbol(symbol: str, timeframe: str, sector_regimes: dict, industry: str) -> pd.DataFrame:
    path = f"{RAW_DIR}/{timeframe}/{symbol}.parquet"
    df = pd.read_parquet(path)
    if len(df) < 200:
        return pd.DataFrame()

    feat = build_feature_frame(df)
    feat = build_v2_features(feat, timeframe)
    n = len(feat)

    open_ = feat["open"].to_numpy()
    high = feat["high"].to_numpy()
    low = feat["low"].to_numpy()
    close = feat["close"].to_numpy()
    z = feat["delta_zscore"].to_numpy()
    slope = feat["vwap_slope_pct"].to_numpy()
    session = feat["session"].to_numpy()
    ts = feat["timestamp"].to_numpy()
    sess_last = _session_last_idx(pd.factorize(session)[0])

    slope_ok = np.abs(slope) <= SLOPE_CUTOFF[timeframe]

    index_name = stock_sector_index(industry)
    sector_ok_above = np.ones(n, dtype=bool)
    sector_ok_below = np.ones(n, dtype=bool)
    if index_name in sector_regimes:
        regime_df = sector_regimes[index_name]
        merged = pd.merge(feat[["timestamp"]], regime_df[["timestamp", f"{index_name}_regime"]],
                           on="timestamp", how="left")
        regime_arr = merged[f"{index_name}_regime"].fillna("stable").to_numpy()
        sector_ok_above = regime_arr != "bullish"   # don't short into a euphoric sector
        sector_ok_below = regime_arr != "bearish"   # don't buy the dip into a crashing sector

    rows = []
    for thr, direction, idx_arr in zscore_entry_events(z, session):
        sector_ok = sector_ok_above if direction == "above" else sector_ok_below

        layers = {
            "zscore_only": idx_arr,
            "+slope_flat": idx_arr[slope_ok[idx_arr]],
            "+slope+sector": idx_arr[slope_ok[idx_arr] & sector_ok[idx_arr]],
        }
        for layer_name, layer_idx in layers.items():
            for target, stop in TARGET_STOP_VARIANTS:
                trades = _simulate(layer_idx, thr, direction, target, stop, open_, high, low, close,
                                    sess_last, n, ts, regime_arr=None, filtered=False)
                for row in trades:
                    rows.append((symbol, timeframe, layer_name) + row)

    cols = [
        "symbol", "timeframe", "filter_layer", "threshold", "direction", "target_pct", "stop_pct",
        "signal_time", "entry_time", "exit_time", "entry_price", "exit_price",
        "pnl_pct", "exit_reason", "bars_held", "nifty_regime_at_signal",
    ]
    return pd.DataFrame(rows, columns=cols)


def main():
    inst = pd.read_csv(MAP_PATH)

    all_trades = []
    for tf_label, subdir in TIMEFRAMES.items():
        print(f"Loading sector regimes for {tf_label}...")
        sector_regimes = load_all_sector_regimes(RAW_DIR, subdir)
        print(f"  loaded {len(sector_regimes)} index regimes")

        for i, row in inst.iterrows():
            symbol = row["Symbol"]
            industry = row["industry"] if "industry" in inst.columns else row.get("Industry", "")
            trades = run_symbol(symbol, tf_label, sector_regimes, industry)
            if not trades.empty:
                all_trades.append(trades)
            if (i + 1) % 100 == 0:
                print(f"{tf_label}: {i+1}/{len(inst)} symbols processed")

    trades = pd.concat(all_trades, ignore_index=True)
    trades.to_parquet(f"{REPO_ROOT}/results/v2_filtered_trades.parquet", index=False)

    summary = summarize_trades(trades, ["timeframe", "filter_layer", "direction", "target_pct", "stop_pct"])
    summary.to_csv(f"{REPO_ROOT}/results/summary/v2_filtered_summary.csv", index=False)

    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 20)
    print("\n=== v2 filtered strategy summary ===")
    print(summary[["timeframe", "filter_layer", "direction", "target_pct", "stop_pct", "n_trades",
                    "win_rate_pct", "avg_pnl_pct", "profit_factor", "sharpe_like"]]
          .sort_values(["timeframe", "target_pct", "direction", "filter_layer"]).to_string())


if __name__ == "__main__":
    main()
