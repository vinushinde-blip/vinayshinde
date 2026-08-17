"""
Orchestrates the full analysis over every fetched (symbol, timeframe) parquet file:
features -> crossing counts -> behavioral event table -> mean-reversion backtest.
Resumable: skips (symbol, timeframe) pairs already recorded in the checkpoint.
Writes accumulated results as parquet/CSV under RESULTS_DIR.
"""
import json
import logging
import os
import sys
import time

import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from features import build_feature_frame
from crossings import detect_crossings, crossing_counts
from behavior import build_event_table, summarize_events
from backtest import backtest_symbol, summarize_trades

REPO_ROOT = "/home/user/vinayshinde/quant_vwap"
MAP_PATH = f"{REPO_ROOT}/data_cache/nse500_instrument_map.csv"
RAW_DIR = f"{REPO_ROOT}/data_cache/raw"
RESULTS_DIR = f"{REPO_ROOT}/results"
CHECKPOINT_PATH = f"{RESULTS_DIR}/pipeline_checkpoint.json"
LOG_PATH = f"{RESULTS_DIR}/pipeline_log.txt"

TIMEFRAME_DIRS = {"1minute": "1minute", "5minute": "5minute", "15minute": "15minute"}

os.makedirs(RESULTS_DIR, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.FileHandler(LOG_PATH), logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("pipeline")


def load_checkpoint():
    if os.path.exists(CHECKPOINT_PATH):
        with open(CHECKPOINT_PATH) as f:
            return set(tuple(x) for x in json.load(f))
    return set()


def save_checkpoint(done):
    with open(CHECKPOINT_PATH, "w") as f:
        json.dump(sorted(list(done)), f)


def append_parquet(df: pd.DataFrame, path: str):
    if df.empty:
        return
    if os.path.exists(path):
        existing = pd.read_parquet(path)
        df = pd.concat([existing, df], ignore_index=True)
    df.to_parquet(path, index=False)


def main():
    inst = pd.read_csv(MAP_PATH)
    symbols = inst["Symbol"].tolist()
    done = load_checkpoint()

    cross_counts_path = f"{RESULTS_DIR}/crossing_counts_by_symbol.parquet"
    events_path = f"{RESULTS_DIR}/behavior_events.parquet"
    trades_path = f"{RESULTS_DIR}/backtest_trades.parquet"

    total = len(symbols) * len(TIMEFRAME_DIRS)
    n_done = len(done)
    log.info(f"Starting pipeline: {total} symbol-timeframe pairs, {n_done} already done")

    cc_buffer, ev_buffer, tr_buffer = [], [], []
    BUFFER_FLUSH = 40  # flush to disk every N symbol-timeframe pairs processed

    processed_since_flush = 0
    t_start = time.time()

    for symbol in symbols:
        for timeframe, subdir in TIMEFRAME_DIRS.items():
            key = (symbol, timeframe)
            if list(key) in [list(x) for x in done]:
                continue
            path = f"{RAW_DIR}/{subdir}/{symbol}.parquet"
            if not os.path.exists(path):
                continue  # not fetched yet; pipeline can be re-run later to pick it up

            try:
                df = pd.read_parquet(path)
                if len(df) < 50:
                    log.warning(f"SKIP {symbol}/{timeframe}: too few rows ({len(df)})")
                    done.add(key)
                    continue

                feat = build_feature_frame(df)

                cross_ev = detect_crossings(feat)
                cc = crossing_counts(cross_ev).reset_index()
                cc["symbol"] = symbol
                cc["timeframe"] = timeframe
                cc_buffer.append(cc)

                ev = build_event_table(feat, symbol, timeframe)
                if not ev.empty:
                    ev_buffer.append(ev)

                trades = backtest_symbol(feat, symbol, timeframe)
                if not trades.empty:
                    tr_buffer.append(trades)

                done.add(key)
            except Exception as e:
                log.error(f"FAILED {symbol}/{timeframe}: {e!r}")
                continue

            processed_since_flush += 1
            if processed_since_flush >= BUFFER_FLUSH:
                if cc_buffer:
                    append_parquet(pd.concat(cc_buffer, ignore_index=True), cross_counts_path)
                    cc_buffer = []
                if ev_buffer:
                    append_parquet(pd.concat(ev_buffer, ignore_index=True), events_path)
                    ev_buffer = []
                if tr_buffer:
                    append_parquet(pd.concat(tr_buffer, ignore_index=True), trades_path)
                    tr_buffer = []
                save_checkpoint(done)
                elapsed = time.time() - t_start
                rate = len(done) / elapsed if elapsed > 0 else 0
                log.info(f"PROGRESS: {len(done)}/{total} done, {rate:.2f}/s, elapsed {elapsed:.0f}s")
                processed_since_flush = 0

    if cc_buffer:
        append_parquet(pd.concat(cc_buffer, ignore_index=True), cross_counts_path)
    if ev_buffer:
        append_parquet(pd.concat(ev_buffer, ignore_index=True), events_path)
    if tr_buffer:
        append_parquet(pd.concat(tr_buffer, ignore_index=True), trades_path)
    save_checkpoint(done)

    log.info(f"DONE. {len(done)}/{total} symbol-timeframe pairs processed")


if __name__ == "__main__":
    main()
