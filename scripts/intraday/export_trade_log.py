"""
Exports the full historical trade/alert log (every signal each strategy
would have fired, with entry/exit prices and net-of-cost return) across
all 500 stocks, for all four strategies in strategies.py. Unlike
run_backtest.py this keeps the per-trade rows rather than aggregating to a
daily portfolio return — meant for browsing/auditing individual alerts,
not for performance metrics (use run_backtest.py for those).

Usage:
    python3 export_trade_log.py
"""

import glob
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd

from strategies import STRATEGIES
from engine import apply_costs

BARS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "intraday", "bars")
UNIVERSE_FILE = os.path.join(os.path.dirname(__file__), "..", "..", "data", "intraday", "universe_top500.csv")
OUT_FILE = os.path.join(os.path.dirname(__file__), "..", "..", "data", "intraday", "trade_log.csv")

_STATE = {}


def _init_worker(bars, rank, universe_size):
    _STATE.update(bars=bars, rank=rank, universe_size=universe_size)


def _run_strategy(strat_name):
    strat_fn = STRATEGIES[strat_name]
    bars, rank, universe_size = _STATE["bars"], _STATE["rank"], _STATE["universe_size"]
    rows = []
    for symbol, df in bars.items():
        trades = strat_fn(df)
        if trades.empty:
            continue
        r = rank.get(symbol, universe_size)
        trades = apply_costs(trades, liquidity_rank=r, universe_size=universe_size)
        trades = trades.copy()
        trades.insert(0, "symbol", symbol)
        trades.insert(1, "strategy", strat_name)
        rows.append(trades)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def main():
    universe = pd.read_csv(UNIVERSE_FILE)
    rank = dict(zip(universe["tradingsymbol"], universe["liquidity_rank"]))
    universe_size = len(rank)

    bars = {}
    for path in sorted(glob.glob(os.path.join(BARS_DIR, "*.parquet"))):
        symbol = os.path.splitext(os.path.basename(path))[0]
        bars[symbol] = pd.read_parquet(path)
    print(f"Loaded bars for {len(bars)} symbols.")

    all_frames = []
    # 3 workers, leaving headroom for run_backtest.py already running
    with ProcessPoolExecutor(max_workers=3, initializer=_init_worker,
                              initargs=(bars, rank, universe_size)) as pool:
        futures = {pool.submit(_run_strategy, name): name for name in STRATEGIES}
        for fut in as_completed(futures):
            name = futures[fut]
            df = fut.result()
            print(f"{name}: {len(df)} alerts")
            if not df.empty:
                all_frames.append(df)

    combined = pd.concat(all_frames, ignore_index=True)
    combined = combined.sort_values(["date", "entry_time"]).reset_index(drop=True)
    os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)
    combined.to_csv(OUT_FILE, index=False)
    print(f"\nTotal alerts across all strategies: {len(combined)}")
    print(f"Saved: {OUT_FILE}")
    print("\nPer strategy x direction counts:")
    print(combined.groupby(["strategy", "direction"]).size())


if __name__ == "__main__":
    main()
