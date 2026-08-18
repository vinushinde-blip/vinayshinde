"""
Tests a "capture the high-probability partial retracement" strategy, motivated
by the reversal_breakdown.py finding: at a 2% VWAP threshold, full reversion to
VWAP happens only 27-35% of the time, but SOME retracement (>=0.5 percentage
points) happens 78-86% of the time. The 1.0%/1.5% price targets used in the
main study compete against noise long enough for the 0.5% stop to often win
first, even when a retracement eventually happens. This tries much tighter
targets (0.5%, matched or tighter stops) to see if catching the retracement
early improves the win rate / profit factor enough to matter.

Reuses the exact entry/exit mechanics from backtest.py (next-bar-open entry,
target/stop/EOD-square-off exit, stop-first same-bar tie-break) — only the
target:stop pair changes.
"""
import sys

import pandas as pd

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from features import build_feature_frame
from backtest import _entry_events, _simulate, _session_last_idx, summarize_trades

REPO_ROOT = "/home/user/vinayshinde/quant_vwap"
RAW_DIR = f"{REPO_ROOT}/data_cache/raw"
MAP_PATH = f"{REPO_ROOT}/data_cache/nse500_instrument_map.csv"

TIMEFRAMES = {"1minute": "1minute"}
THRESHOLDS = [3.0, 4.0]
VARIANTS = [(0.5, 0.3), (0.7, 0.4), (1.0, 0.5)]  # (target_pct, stop_pct)


def backtest_symbol_custom(df: pd.DataFrame, symbol: str, timeframe: str) -> pd.DataFrame:
    df = df.reset_index(drop=True)
    feat = build_feature_frame(df)
    n = len(feat)
    open_ = feat["open"].to_numpy()
    high = feat["high"].to_numpy()
    low = feat["low"].to_numpy()
    close = feat["close"].to_numpy()
    delta = feat["delta_pct"].to_numpy()
    session = feat["session"].to_numpy()
    ts = feat["timestamp"].to_numpy()
    sess_last = _session_last_idx(pd.factorize(session)[0])

    rows = []
    for thr, direction, idx_arr in _entry_events(delta, session):
        if thr not in THRESHOLDS:
            continue
        for target, stop in VARIANTS:
            trades = _simulate(idx_arr, thr, direction, target, stop, open_, high, low, close,
                                sess_last, n, ts, regime_arr=None, filtered=False)
            for row in trades:
                rows.append((symbol, timeframe) + row)

    cols = [
        "symbol", "timeframe", "threshold", "direction", "target_pct", "stop_pct",
        "signal_time", "entry_time", "exit_time", "entry_price", "exit_price",
        "pnl_pct", "exit_reason", "bars_held", "nifty_regime_at_signal",
    ]
    return pd.DataFrame(rows, columns=cols)


def main():
    inst = pd.read_csv(MAP_PATH)
    symbols = inst["Symbol"].tolist()

    all_trades = []
    for tf_label, subdir in TIMEFRAMES.items():
        for i, symbol in enumerate(symbols):
            path = f"{RAW_DIR}/{subdir}/{symbol}.parquet"
            try:
                df = pd.read_parquet(path)
            except FileNotFoundError:
                continue
            if len(df) < 50:
                continue
            trades = backtest_symbol_custom(df, symbol, tf_label)
            if not trades.empty:
                all_trades.append(trades)
            if (i + 1) % 100 == 0:
                print(f"{tf_label}: {i+1}/{len(symbols)} symbols processed")

    trades = pd.concat(all_trades, ignore_index=True)
    trades.to_parquet(f"{REPO_ROOT}/results/tight_target_trades_v2_thr3-4.parquet", index=False)

    summary = summarize_trades(trades, ["timeframe", "threshold", "direction", "target_pct", "stop_pct"])
    summary.to_csv(f"{REPO_ROOT}/results/summary/tight_target_summary_v2_thr3-4.csv", index=False)
    print("\n=== Tight-target strategy summary ===")
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 20)
    print(summary[["timeframe", "threshold", "direction", "target_pct", "stop_pct", "n_trades",
                    "win_rate_pct", "avg_pnl_pct", "profit_factor", "sharpe_like", "avg_bars_held"]]
          .sort_values("profit_factor", ascending=False).to_string())


if __name__ == "__main__":
    main()
