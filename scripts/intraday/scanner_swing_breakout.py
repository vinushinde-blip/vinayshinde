"""
Swing-scale failed-breakout scanner — the multi-day half of the pattern
from the PPLPHARMA chart (breakout, then give-back over the following
session(s)). Runs across the full universe, derived from the same cached
intraday bars (resampled to daily), no new data needed.

Detection requires FOUR conditions at once, not just a new high on mild
volume — each one cuts the false-positive rate, at the cost of fewer
signals (that's the point: restricted, not exhaustive):
  1. Price: close breaks above the prior `lookback_days`-day high by at
     least `min_breakout_pct` — a bare poke through resistance doesn't
     count, only a decisive break.
  2. Volume: >= `vol_mult` times the `lookback_days`-day MEDIAN volume
     (median, not mean — see find_breakouts_and_failures docstring).
  3. Trend: close above a RISING 50-day SMA — only takes breakouts that
     already have an intermediate-term uptrend behind them, not a break
     against the larger trend.
  4. Some momentum behind it: RSI(14) above `rsi_min` at breakout — not a
     ceiling. An overbought RSI is exactly the profile of the breakouts
     most likely to fail (which is a big part of the point of this
     scanner — it tracks failures, it doesn't just want clean ones), so
     excluding high-RSI breakouts would filter out the best short
     candidates. `rsi_min` only screens out breakouts with essentially no
     momentum behind them at all.

A fifth thing — SQUEEZE (was the day before the breakout unusually quiet
vs. its own trailing volatility) — is computed and labeled on every
signal but NOT used as a hard filter. Reason, found by testing against
the real PPLPHARMA breakout that motivated this scanner: its daily ATR
was in the TOP 10% of its own 60-day range at breakout, not contracted —
the tightness visible in the chart was purely an intraday phenomenon (a
few quiet 5-minute candles), not a multi-day daily-bar contraction. A
stock can break out either from genuine quiet consolidation OR from an
already-trending, elevated-volatility state (grinding to a new extension)
— both are real patterns, and gating on "quiet only" would have excluded
the exact example this scanner exists to catch. Pass --require-squeeze to
gate on it anyway if you specifically want the quiet-consolidation subtype.

Then a CONCURRENCY CAP on top of the detector itself: a swing position
ties up capital for up to confirm_days (not minutes, like the intraday
strategies), so even a well-filtered detector can still generate more
simultaneous candidates than anyone could realistically hold. Simulates
`max_concurrent_swing_positions` open positions at once, in true
chronological order, with liquidity-priority when multiple breakouts
compete for a slot on the same day (same non-lookahead justification as
the intraday capping: liquidity_rank predates the trade).

Three outputs, all multi-stock, all post-filter/post-cap:
  - Historical base rate: of every RESTRICTED breakout, what fraction
    failed within confirm_days?
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

import indicators as ind

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
                                 vol_mult: float = 1.25, confirm_days: int = 5,
                                 min_breakout_pct: float = 0.5, squeeze_lookback: int = 60,
                                 squeeze_pctile: float = 0.4, require_squeeze: bool = False,
                                 trend_days: int = 50, trend_rising_days: int = 5,
                                 rsi_min: float = 50.0) -> pd.DataFrame:
    """vol_mult is checked against the rolling MEDIAN volume, not the mean.
    A stock that already had a volume-spike day within the lookback window
    (e.g. an earlier breakout) skews the mean baseline up, understating how
    unusual a fresh spike is — median is robust to that. Confirmed on
    PPLPHARMA's actual 2026-08-10 breakout: volume was ~1.17x the 20-day
    MEAN (below a naive 1.5x threshold, so it went undetected) but ~1.28x
    the 20-day MEDIAN — genuinely elevated once the baseline isn't
    distorted by its own prior spike days.

    See module docstring for the other conditions (breakout strength,
    trend, RSI floor, and the squeeze label/optional gate) — all computed
    only from data at/before the breakout bar, no look-ahead.
    """
    min_history = max(lookback_days, squeeze_lookback, trend_days) + confirm_days
    if len(daily) <= min_history:
        return pd.DataFrame()

    prior_high = daily["high"].rolling(lookback_days).max().shift(1)
    median_vol = daily["volume"].rolling(lookback_days).median().shift(1)
    price_condition = daily["close"] > prior_high * (1 + min_breakout_pct / 100)
    volume_condition = daily["volume"] >= vol_mult * median_vol

    atr = ind.atr(daily, 14)
    atr_pre_breakout = atr.shift(1)  # the day BEFORE the breakout — pre-move volatility only
    atr_threshold = atr.rolling(squeeze_lookback).quantile(squeeze_pctile).shift(1)
    squeeze_condition = atr_pre_breakout <= atr_threshold  # labeled on every row; gated only if require_squeeze

    sma = daily["close"].rolling(trend_days).mean()
    trend_condition = (daily["close"] > sma) & (sma > sma.shift(trend_rising_days))

    rsi = ind.rsi(daily["close"], 14)
    rsi_condition = rsi > rsi_min

    is_breakout = price_condition & volume_condition & trend_condition & rsi_condition
    if require_squeeze:
        is_breakout = is_breakout & squeeze_condition

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
            "was_squeeze": bool(squeeze_condition.loc[bdate]),
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


def apply_swing_concurrency_cap(combined: pd.DataFrame, rank: dict, max_concurrent: int,
                                 confirm_days: int) -> pd.DataFrame:
    """A swing position ties up capital from breakout_date until it
    resolves — failure_date if failed, else breakout_date + confirm_days
    for held/pending (the monitoring window's own boundary, since those
    have no single resolved exit date yet). Walks all detected breakouts
    across the WHOLE universe in true chronological order, only admitting
    a new one when a concurrency slot is free; when several breakouts
    share the same breakout_date and compete for the same freed slot, the
    more liquid one wins (liquidity_rank predates the trade, not
    look-ahead — see module docstring).
    """
    df = combined.copy()
    df["eff_close"] = df.apply(
        lambda r: r["failure_date"] if r["outcome"] == "failed"
        else r["breakout_date"] + pd.Timedelta(days=confirm_days * 2),  # calendar-day pad for weekends
        axis=1)
    df["liq_rank_tmp"] = df["symbol"].map(rank).fillna(len(rank) + 1)
    df = df.sort_values(["breakout_date", "liq_rank_tmp"])

    open_positions = []  # list of eff_close dates currently held
    kept_idx = []
    for bdate, batch in df.groupby("breakout_date", sort=True):
        open_positions = [d for d in open_positions if d > bdate]
        for row in batch.itertuples():
            if len(open_positions) >= max_concurrent:
                continue
            kept_idx.append(row.Index)
            open_positions.append(row.eff_close)

    return df.loc[kept_idx].drop(columns=["eff_close", "liq_rank_tmp"]).reset_index(drop=True)


def apply_daily_signal_cap(combined: pd.DataFrame, rank: dict, max_new_per_day: int) -> pd.DataFrame:
    """Separate from the concurrency cap on purpose: concurrency answers
    "how many positions can I realistically hold at once" (a portfolio-
    capacity question), this answers "how many NEW ideas should I even
    look at today" (a review-bandwidth question) — conflating the two
    into a single cap either starves the signal set down to nothing (a
    tight concurrency cap blocks almost everything before you even see
    it) or lets busy days dump dozens of new names on you at once (a
    loose one). Trims each day's admitted signals to the top
    `max_new_per_day` by liquidity.
    """
    df = combined.copy()
    df["liq_rank_tmp"] = df["symbol"].map(rank).fillna(len(rank) + 1)
    df = df.sort_values(["breakout_date", "liq_rank_tmp"])
    trimmed = df.groupby("breakout_date", group_keys=False).head(max_new_per_day)
    return trimmed.drop(columns="liq_rank_tmp").reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lookback-days", type=int, default=20)
    parser.add_argument("--vol-mult", type=float, default=1.25)
    parser.add_argument("--confirm-days", type=int, default=5)
    parser.add_argument("--min-breakout-pct", type=float, default=0.5,
                         help="close must exceed the prior high by at least this %% (default 0.5)")
    parser.add_argument("--squeeze-lookback", type=int, default=60)
    parser.add_argument("--squeeze-pctile", type=float, default=0.4,
                         help="pre-breakout ATR percentile threshold, used only if --require-squeeze "
                              "(default 0.4 — tighter 40%% band); otherwise just labeled on output")
    parser.add_argument("--require-squeeze", action="store_true",
                         help="gate on the squeeze condition (off by default — see module docstring; "
                              "this would have excluded the real PPLPHARMA example)")
    parser.add_argument("--trend-days", type=int, default=50)
    parser.add_argument("--trend-rising-days", type=int, default=5)
    parser.add_argument("--rsi-min", type=float, default=50.0,
                         help="skip breakouts with essentially no momentum behind them; NOT an overbought "
                              "ceiling, see module docstring for why (default 50)")
    parser.add_argument("--max-concurrent-swing-positions", type=int, default=10,
                         help="realistic total portfolio capacity — positions tie up capital for days, "
                              "not minutes, so this can be higher than the intraday cap (default 10; "
                              "pass 0 to disable)")
    parser.add_argument("--max-new-signals-per-day", type=int, default=3,
                         help="direct cap on how many NEW signals surface on any given day, separate from "
                              "portfolio capacity — this is the one that actually answers 'too many signals "
                              "in a day' (default 3; pass 0 to disable)")
    parser.add_argument("--recent-window", type=int, default=5,
                         help="only surface breakouts/failures from the last N trading days as actionable")
    parser.add_argument("--symbol", default=None, help="inspect one stock's full breakout history instead of scanning")
    args = parser.parse_args()

    detector_kwargs = dict(
        lookback_days=args.lookback_days, vol_mult=args.vol_mult, confirm_days=args.confirm_days,
        min_breakout_pct=args.min_breakout_pct, squeeze_lookback=args.squeeze_lookback,
        squeeze_pctile=args.squeeze_pctile, require_squeeze=args.require_squeeze,
        trend_days=args.trend_days, trend_rising_days=args.trend_rising_days, rsi_min=args.rsi_min,
    )

    bars, rank = load_bars_and_rank()

    if args.symbol:
        if args.symbol not in bars:
            raise SystemExit(f"{args.symbol} not in cached universe.")
        daily = to_daily(bars[args.symbol])
        result = find_breakouts_and_failures(daily, **detector_kwargs)
        print(f"{args.symbol}: {len(result)} breakout(s) detected in {len(daily)} cached daily bars\n")
        print(result.to_string(index=False) if not result.empty else "(none)")
        return

    all_results = []
    for symbol, df in bars.items():
        daily = to_daily(df)
        result = find_breakouts_and_failures(daily, **detector_kwargs)
        if result.empty:
            continue
        result.insert(0, "symbol", symbol)
        result.insert(1, "liquidity_rank", rank.get(symbol))
        all_results.append(result)

    if not all_results:
        raise SystemExit("No breakouts detected across the universe with these parameters.")
    combined = pd.concat(all_results, ignore_index=True)
    n_before_cap = len(combined)

    if args.max_concurrent_swing_positions > 0:
        combined = apply_swing_concurrency_cap(combined, rank, args.max_concurrent_swing_positions, args.confirm_days)
    n_after_concurrency = len(combined)

    if args.max_new_signals_per_day > 0:
        combined = apply_daily_signal_cap(combined, rank, args.max_new_signals_per_day)

    print(f"Detector found {n_before_cap} breakouts.")
    print(f"Portfolio-capacity cap (max {args.max_concurrent_swing_positions} concurrent) left {n_after_concurrency}.")
    print(f"Daily signal cap (max {args.max_new_signals_per_day}/day) left {len(combined)} "
          f"across {combined['breakout_date'].nunique() if not combined.empty else 0} days "
          f"({len(combined) / max(combined['breakout_date'].nunique(), 1):.1f}/day average).\n")

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
