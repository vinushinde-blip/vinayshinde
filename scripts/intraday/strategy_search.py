"""
Searches over (trigger x filter-combo x stop/target preset) combinations —
a bounded, named grammar (see triggers_filters.py), not a literal brute
force over "every indicator and every parameter" — and reports whichever
survives a proper three-way chronological split honestly.

Split (by calendar date, same cutoffs across all symbols — no leakage):
    search-train (first 50%)      -> not used for selection, just visible
    search-validation (next 30%)  -> combo SELECTION happens here
    final-holdout (last 20%)      -> touched exactly once, only for the
                                      single winning combo, at the very end

Why validation (not train) picks the winner: picking on train performance
would just reward whichever combo overfits train hardest. Selecting on a
separate validation slice, then confirming once on a never-touched holdout,
is the standard three-way guard against the "tried N things, reported the
best one" trap.

Runs on a liquidity-ranked SUBSET of the full 500 by default (--top-n),
since evaluating ~150 combos x 500 stocks x years of 5-min bars in a
Python per-bar loop is not fast — the subset should still be diverse
enough (different sectors/liquidity tiers) to mean something, but treat
this as a first-pass screen, not a final answer on the full universe.

Usage:
    python3 strategy_search.py --top-n 80
"""

import argparse
import glob
import itertools
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd

import triggers_filters as tf
from combo_engine import generate_trades
from engine import apply_costs, portfolio_daily_returns
from metrics import performance_metrics

BARS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "intraday", "bars")
UNIVERSE_FILE = os.path.join(os.path.dirname(__file__), "..", "..", "data", "intraday", "universe_top500.csv")
RESULTS_FILE = os.path.join(os.path.dirname(__file__), "..", "..", "data", "intraday", "strategy_search_results.csv")

STOP_TARGET_PRESETS = [(1.5, 3.0), (1.0, 2.0)]
MIN_TRADES_FOR_SELECTION = 100  # ignore combos with too few validation trades to mean anything


def filter_combos():
    names = list(tf.FILTERS.keys())
    combos = [()]
    combos += [(n,) for n in names]
    combos += list(itertools.combinations(names, 2))
    return combos


def apply_filters(df, names):
    if not names:
        idx = df.index
        return pd.Series(True, index=idx), pd.Series(True, index=idx)
    long_ok = pd.Series(True, index=df.index)
    short_ok = pd.Series(True, index=df.index)
    for name in names:
        lo, so = tf.FILTERS[name](df)
        long_ok &= lo.fillna(False)
        short_ok &= so.fillna(False)
    return long_ok, short_ok


def load_bars(top_n: int):
    universe = pd.read_csv(UNIVERSE_FILE)
    rank = dict(zip(universe["tradingsymbol"], universe["liquidity_rank"]))
    subset = set(universe[universe["liquidity_rank"] <= top_n]["tradingsymbol"])

    bars = {}
    for path in sorted(glob.glob(os.path.join(BARS_DIR, "*.parquet"))):
        symbol = os.path.splitext(os.path.basename(path))[0]
        if symbol in subset:
            bars[symbol] = pd.read_parquet(path)
    return bars, rank


def split_dates(bars: dict, train_frac=0.5, val_frac=0.3):
    all_dates = sorted({d for df in bars.values() for d in df.index.date})
    n = len(all_dates)
    train_end = all_dates[int(n * train_frac) - 1]
    val_end = all_dates[int(n * (train_frac + val_frac)) - 1]
    return train_end, val_end, all_dates[0], all_dates[-1]


def evaluate_combo(trigger_name, filter_names, stop_mult, target_mult, bars, rank, universe_size,
                    date_lo=None, date_hi=None):
    trigger_fn = tf.TRIGGERS[trigger_name]
    all_trades = {}
    for symbol, df in bars.items():
        signal = trigger_fn(df)
        long_ok, short_ok = apply_filters(df, filter_names)
        trades = generate_trades(df, signal, long_ok, short_ok, stop_mult, target_mult)
        if trades.empty:
            continue
        if date_lo is not None:
            mask = (trades["date"] >= date_lo) & (trades["date"] <= date_hi)
            trades = trades[mask]
        if trades.empty:
            continue
        r = rank.get(symbol, universe_size)
        trades = apply_costs(trades, liquidity_rank=r, universe_size=universe_size)
        all_trades[symbol] = trades

    daily = portfolio_daily_returns(all_trades)
    n_trades = sum(len(t) for t in all_trades.values())
    return daily, n_trades


# ---------- multiprocess worker (globals set per-process via initializer to
# avoid re-pickling the full bar dataset for every single combo task) ----------

_WORKER_STATE = {}


def _init_worker(bars, rank, universe_size, train_end, val_end):
    _WORKER_STATE.update(bars=bars, rank=rank, universe_size=universe_size,
                          train_end=train_end, val_end=val_end)


def _worker_evaluate(combo):
    trig, filt, stop, target = combo
    s = _WORKER_STATE
    daily_val, n_val = evaluate_combo(trig, filt, stop, target, s["bars"], s["rank"], s["universe_size"],
                                       date_lo=s["train_end"], date_hi=s["val_end"])
    m_val = performance_metrics(daily_val) if n_val >= MIN_TRADES_FOR_SELECTION else None
    return {
        "trigger": trig, "filters": "+".join(filt) if filt else "none",
        "stop_atr": stop, "target_atr": target,
        "val_trades": n_val,
        "val_CAGR": m_val["CAGR"] if m_val else None,
        "val_Sharpe": m_val["Sharpe"] if m_val else None,
        "val_MaxDD": m_val["MaxDrawdown"] if m_val else None,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--top-n", type=int, default=80,
                         help="search on the N most liquid stocks (full 500 is too slow for a search loop)")
    parser.add_argument("--workers", type=int, default=4, help="parallel processes for the combo search")
    args = parser.parse_args()

    bars, rank = load_bars(args.top_n)
    if not bars:
        raise SystemExit(f"No bar data found for top {args.top_n} in {BARS_DIR}. Run kite_fetch_intraday.py first.")
    universe_size = len(rank)

    train_end, val_end, date_start, date_end = split_dates(bars)
    print(f"Data span: {date_start} to {date_end}")
    print(f"Search-train: {date_start} to {train_end}")
    print(f"Search-validation (selection happens here): {train_end} to {val_end}")
    print(f"Final holdout (touched once, winner only): {val_end} to {date_end}")

    combos = [
        (trig, filt, stop, target)
        for trig in tf.TRIGGERS
        for filt in filter_combos()
        for stop, target in STOP_TARGET_PRESETS
    ]
    print(f"\nTotal combinations to evaluate: {len(combos)} "
          f"({len(tf.TRIGGERS)} triggers x {len(filter_combos())} filter-sets x {len(STOP_TARGET_PRESETS)} stop/target presets)")
    print(f"Searching on top {len(bars)} liquid stocks, {args.workers} parallel workers.\n")

    results = []
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_init_worker,
                              initargs=(bars, rank, universe_size, train_end, val_end)) as pool:
        futures = [pool.submit(_worker_evaluate, combo) for combo in combos]
        for i, fut in enumerate(as_completed(futures), start=1):
            results.append(fut.result())
            if i % 20 == 0:
                print(f"  evaluated {i}/{len(combos)} combos")

    table = pd.DataFrame(results)
    ranked = table.dropna(subset=["val_Sharpe"]).sort_values("val_Sharpe", ascending=False)

    os.makedirs(os.path.dirname(RESULTS_FILE), exist_ok=True)
    table.to_csv(RESULTS_FILE, index=False)
    print(f"\nAll {len(combos)} combo results saved to {RESULTS_FILE}")
    print(f"\nTop 10 by validation Sharpe (of {len(ranked)} combos with >= {MIN_TRADES_FOR_SELECTION} validation trades):")
    print(ranked.head(10).to_string(index=False))

    if ranked.empty:
        print("\nNo combo cleared the minimum trade count on validation — no winner to confirm on holdout.")
        return

    winner = ranked.iloc[0]
    print(f"\n=== Confirming winner on final holdout (touched once): "
          f"{winner['trigger']} + [{winner['filters']}] stop={winner['stop_atr']}x target={winner['target_atr']}x ===")
    daily_hold, n_hold = evaluate_combo(
        winner["trigger"], [] if winner["filters"] == "none" else winner["filters"].split("+"),
        winner["stop_atr"], winner["target_atr"], bars, rank, universe_size,
        date_lo=val_end, date_hi=date_end)
    if n_hold < MIN_TRADES_FOR_SELECTION:
        print(f"Only {n_hold} holdout trades — too few to trust either way. "
              f"Treat this combo as unconfirmed, not as a validated winner.")
    else:
        m_hold = performance_metrics(daily_hold)
        print(f"Holdout trades: {n_hold}")
        print(f"Holdout performance: {m_hold}")
        print(f"\nCompare this to the validation numbers above — a real edge should look "
              f"similar on both, not much worse out here.")

    print(f"\nReminder: {len(combos)} combinations were tried before picking this one. "
          f"Even a genuinely random process finds an apparent winner among that many trials — "
          f"the holdout confirmation above is the only number that partly guards against that, "
          f"and even it is a single holdout period, not proof of a durable edge.")


if __name__ == "__main__":
    main()
