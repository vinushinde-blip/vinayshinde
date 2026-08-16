"""
Swing-scale failed-breakout scanner — the multi-day half of the pattern
from the PPLPHARMA chart (breakout, then give-back over the following
session(s)). Runs across the full universe, derived from the same cached
intraday bars (resampled to daily), no new data needed.

Detects, per stock:
  1. Breakout day: close breaks above the prior `lookback_days`-day high,
     on volume >= `vol_mult` times the `lookback_days`-day MEDIAN volume
     (median, not mean — see find_breakouts_and_failures docstring).
  2. Over the following `confirm_days` sessions, does price close back
     below the breakout day's LOW (i.e. give back the entire move,
     re-entering the pre-breakout range)? That's the failure.

Three outputs, all multi-stock:
  - Historical base rate: of every breakout detected in the cached
    history, what fraction failed within confirm_days? Context for how
    reliable this setup actually is before trusting it.
  - "Recently failed" — breakout + confirmed failure within the last few
    days. This is the PPLPHARMA-chart situation happening RIGHT NOW
    elsewhere in the universe: a short/avoid-long candidate.
  - "Pending" — a breakout in the last confirm_days with no failure
    confirmed yet (could still go either way) — a watchlist, not a signal.

No look-ahead: a breakout is only flagged once its own day's bar is known
(close, volume), and failure is only flagged once the failing day's close
is known — same causal discipline as the rest of this pipeline.

Usage:
    python3 scanner_swing_breakout.py
    python3 scanner_swing_breakout.py --symbol PPLPHARMA   # inspect one stock's history
"""

import argparse
import glob
import os

import pandas as pd

BARS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "intraday", "bars")
UNIVERSE_FILE = os.path.join(os.path.dirname(__file__), "..", "..", "data", "intraday", "universe_top500.csv")
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "intraday", "scans")


def to_daily(df: pd.DataFrame) -> pd.DataFrame:
    daily = df.groupby(df.index.date).agg(
        open=("open", "first"), high=("high", "max"),
        low=("low", "min"), close=("close", "last"), volume=("volume", "sum"),
    )
    daily.index = pd.to_datetime(daily.index)
    return daily.sort_index()


def find_breakouts_and_failures(daily: pd.DataFrame, lookback_days: int = 20,
                                 vol_mult: float = 1.25, confirm_days: int = 5) -> pd.DataFrame:
    """vol_mult is checked against the rolling MEDIAN volume, not the mean.
    A stock that already had a volume-spike day within the lookback window
    (e.g. an earlier breakout) skews the mean baseline up, understating how
    unusual a fresh spike is — median is robust to that. Confirmed on
    PPLPHARMA's actual 2026-08-10 breakout: volume was ~1.17x the 20-day
    MEAN (below a naive 1.5x threshold, so it went undetected) but ~1.28x
    the 20-day MEDIAN — genuinely elevated once the baseline isn't
    distorted by its own prior spike days.
    """
    if len(daily) <= lookback_days + confirm_days:
        return pd.DataFrame()

    prior_high = daily["high"].rolling(lookback_days).max().shift(1)
    median_vol = daily["volume"].rolling(lookback_days).median().shift(1)

    is_breakout = (daily["close"] > prior_high) & (daily["volume"] >= vol_mult * median_vol)

    rows = []
    breakout_dates = daily.index[is_breakout.fillna(False)]
    for bdate in breakout_dates:
        bidx = daily.index.get_loc(bdate)
        breakout_low = daily["low"].iloc[bidx]
        breakout_close = daily["close"].iloc[bidx]

        outcome, failure_date, days_to_fail = "pending", None, None
        window_end = min(bidx + 1 + confirm_days, len(daily))
        for j in range(bidx + 1, window_end):
            if daily["close"].iloc[j] < breakout_low:
                outcome = "failed"
                failure_date = daily.index[j]
                days_to_fail = j - bidx
                break
        else:
            if window_end - bidx > confirm_days:
                outcome = "held"  # full confirm_days window passed without failing

        rows.append({
            "breakout_date": bdate, "breakout_close": round(breakout_close, 2),
            "breakout_low": round(breakout_low, 2), "outcome": outcome,
            "failure_date": failure_date, "days_to_fail": days_to_fail,
            "days_since_breakout": len(daily) - 1 - bidx,
        })

    return pd.DataFrame(rows)


def load_bars_and_rank():
    universe = pd.read_csv(UNIVERSE_FILE)
    rank = dict(zip(universe["tradingsymbol"], universe["liquidity_rank"]))
    bars = {}
    for path in sorted(glob.glob(os.path.join(BARS_DIR, "*.parquet"))):
        symbol = os.path.splitext(os.path.basename(path))[0]
        bars[symbol] = pd.read_parquet(path)
    return bars, rank


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lookback-days", type=int, default=20)
    parser.add_argument("--vol-mult", type=float, default=1.25)
    parser.add_argument("--confirm-days", type=int, default=5)
    parser.add_argument("--recent-window", type=int, default=5,
                         help="only surface breakouts/failures from the last N trading days as actionable")
    parser.add_argument("--symbol", default=None, help="inspect one stock's full breakout history instead of scanning")
    args = parser.parse_args()

    bars, rank = load_bars_and_rank()

    if args.symbol:
        if args.symbol not in bars:
            raise SystemExit(f"{args.symbol} not in cached universe.")
        daily = to_daily(bars[args.symbol])
        result = find_breakouts_and_failures(daily, args.lookback_days, args.vol_mult, args.confirm_days)
        print(f"{args.symbol}: {len(result)} breakout(s) detected in {len(daily)} cached daily bars\n")
        print(result.to_string(index=False) if not result.empty else "(none)")
        return

    all_results = []
    for symbol, df in bars.items():
        daily = to_daily(df)
        result = find_breakouts_and_failures(daily, args.lookback_days, args.vol_mult, args.confirm_days)
        if result.empty:
            continue
        result.insert(0, "symbol", symbol)
        result.insert(1, "liquidity_rank", rank.get(symbol))
        all_results.append(result)

    if not all_results:
        raise SystemExit("No breakouts detected across the universe with these parameters.")
    combined = pd.concat(all_results, ignore_index=True)

    resolved = combined[combined["outcome"].isin(["failed", "held"])]
    base_rate = (resolved["outcome"] == "failed").mean() * 100 if not resolved.empty else float("nan")
    print(f"=== Historical base rate across the universe ===")
    print(f"{len(combined)} breakouts detected total; {len(resolved)} resolved (failed or held {args.confirm_days} days).")
    print(f"Failure rate: {base_rate:.1f}% (this is the setup's actual track record here, not a guess)\n")

    recently_failed = combined[
        (combined["outcome"] == "failed") &
        (combined["days_since_breakout"] - combined["days_to_fail"] <= args.recent_window)
    ].sort_values("liquidity_rank")
    print(f"=== Recently failed (within last {args.recent_window} trading days) — {len(recently_failed)} ===")
    print(recently_failed.drop(columns="days_since_breakout").to_string(index=False) if not recently_failed.empty else "(none)")

    pending = combined[
        (combined["outcome"] == "pending") & (combined["days_since_breakout"] <= args.recent_window)
    ].sort_values("liquidity_rank")
    print(f"\n=== Pending — broke out in last {args.recent_window} days, not yet confirmed either way — {len(pending)} "
          f"(showing top 30 by liquidity) ===")
    if not pending.empty:
        print(pending.drop(columns=["failure_date", "days_to_fail"]).head(30).to_string(index=False))
    else:
        print("(none)")

    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, "swing_breakout_all_signals.csv")
    combined.sort_values(["symbol", "breakout_date"]).to_csv(out_path, index=False)
    print(f"\nSaved: {out_path} ({len(combined)} breakout events, full history, all outcomes)")


if __name__ == "__main__":
    main()
