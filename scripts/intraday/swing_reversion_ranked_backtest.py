"""
Second-generation reversion strategy, built to directly fix the two
concrete failure modes the first version showed in the full 10-year
backtest:

  1. LOW PRECISION: the raw threshold gate (RSI<=35, below 50-SMA, well
     off highs, elevated ATR%) only produces a "clean 15% move" 3-6.5% of
     the time (vs a 1-2% base rate) - a real ~3x lift, but a hard
     threshold trades every stock that merely crosses the line, most of
     which don't bounce. FIX: rank all of a day's qualifying candidates
     by "score" (how extreme the setup is relative to the STOCK'S OWN
     history, from swing_reversion_strategy.find_reversion_trades) and
     only take the top few - the same logic as "the lift is real but
     weak, so be much more selective," not a new hypothesis.

  2. CORRELATED CRASH DRAWDOWN: the regime study showed launches are
     actually MORE common (by rate) in bear/transition regimes than bull
     - but that's also when the equal-weight, up-to-10-concurrent-
     position sizing blew up (-77% to -81% max drawdown), because real
     selloffs put many stocks into this setup AT THE SAME TIME and they
     keep falling together. FIX: scale the max-concurrent-positions cap
     DOWN specifically in bear/transition regimes instead of either (a)
     excluding those regimes entirely, which throws away the regime with
     the best raw hit rate, or (b) sizing them the same as bull, which is
     what actually caused the drawdown.

Usage:
    python3 swing_reversion_ranked_backtest.py --capital 1000000
"""

import argparse
import os

import pandas as pd

from market_regime import classify_regime
from swing_reversion_strategy import find_reversion_trades
from swing_backtest import (load_bars_and_rank, load_nifty_daily, apply_costs_to_trades,
                             build_daily_mtm_returns, performance_metrics, walk_forward_split,
                             simulate_capital, MAX_NEW_SIGNALS_PER_DAY)

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "intraday")

# max concurrent open positions, by the NIFTY regime on the day a position is ENTERED
MAX_CONCURRENT_BY_REGIME = {"bull": 10, "transition": 5, "bear": 3, "warmup": 3}
TOP_N_PER_DAY = MAX_NEW_SIGNALS_PER_DAY  # keep the same "review bandwidth" cap as other strategies


def generate_all_reversion_trades(daily_by_symbol, rank, nifty_daily, **kwargs):
    frames = []
    for symbol, daily in daily_by_symbol.items():
        trades = find_reversion_trades(daily, nifty_daily, **kwargs)
        if trades.empty:
            continue
        trades.insert(0, "symbol", symbol)
        trades.insert(1, "liquidity_rank", rank.get(symbol, len(rank) + 1))
        frames.append(trades)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def apply_ranked_regime_caps(trades: pd.DataFrame, regime: pd.DataFrame, top_n_per_day: int) -> pd.DataFrame:
    """Two caps, same two-stage shape as apply_portfolio_caps in
    swing_backtest.py, but (a) the daily cap picks the highest-SCORE
    candidates instead of the most-liquid, and (b) the concurrency cap
    varies by the entry day's market regime instead of a single fixed
    number.
    """
    df = trades.sort_values(["entry_date", "score"], ascending=[True, False]).copy()
    df = df.groupby("entry_date", group_keys=False).head(top_n_per_day)
    df = df.sort_values(["entry_date", "score"], ascending=[True, False])

    entry_regime = regime["trend_regime"].reindex(df["entry_date"]).values
    df["entry_regime"] = entry_regime

    open_positions = []  # list of exit_date for currently held positions
    kept_idx = []
    for entry_date, batch in df.groupby("entry_date", sort=True):
        open_positions = [d for d in open_positions if d > entry_date]
        cap = MAX_CONCURRENT_BY_REGIME.get(batch["entry_regime"].iloc[0], 5)
        for row in batch.itertuples():
            if len(open_positions) >= cap:
                continue
            kept_idx.append(row.Index)
            open_positions.append(row.exit_date)

    return df.loc[kept_idx].drop(columns=["entry_regime"]).reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--capital", type=float, default=1_000_000)
    args = parser.parse_args()

    print("Loading data...")
    daily_by_symbol, rank = load_bars_and_rank("daily")
    nifty_daily = load_nifty_daily()
    universe_size = len(rank)
    regime = classify_regime(nifty_daily)
    span = [d for daily in daily_by_symbol.values() for d in daily.index[[0, -1]]]
    print(f"Loaded {len(daily_by_symbol)} symbols, {min(span).date()} to {max(span).date()}.")

    print("Generating reversion candidates (uptrend-context gated) across the universe...")
    raw_trades = generate_all_reversion_trades(daily_by_symbol, rank, nifty_daily, uptrend_context=True)
    print(f"Raw trades generated (pre-cap): {len(raw_trades)}")

    capped = apply_ranked_regime_caps(raw_trades, regime, TOP_N_PER_DAY)
    print(f"After rank+regime caps (top {TOP_N_PER_DAY}/day by score, "
          f"max concurrent {MAX_CONCURRENT_BY_REGIME}): {len(capped)}")

    costed = apply_costs_to_trades(capped, universe_size)

    print("\n=== Trade-level stats ===")
    win_rate = (costed["net_return"] > 0).mean() * 100
    print(f"Total trades: {len(costed)}")
    print(f"Win rate: {win_rate:.1f}%")
    print(f"Avg net return/trade: {costed['net_return'].mean()*100:.2f}%")
    print(f"Exit reason breakdown:\n{costed['exit_reason'].value_counts().to_string()}")

    print("\n=== Percentage-return backtest (walk-forward, mark-to-market) ===")
    daily_returns = build_daily_mtm_returns(costed, daily_by_symbol)
    full_m = performance_metrics(daily_returns)
    train, test = walk_forward_split(daily_returns)
    train_m = performance_metrics(train)
    test_m = performance_metrics(test)
    print(f"Full period:  {full_m}")
    print(f"Train (70%):  {train_m}")
    print(f"Test (30%, out-of-sample): {test_m}")

    cost_pct_orig = costed["gross_return"] - costed["net_return"]
    costed_2x = capped.copy()
    costed_2x["gross_return"] = costed["gross_return"]
    costed_2x["net_return"] = costed["gross_return"] - cost_pct_orig * 2
    daily_2x = build_daily_mtm_returns(costed_2x, daily_by_symbol)
    print(f"At 2x costs:  {performance_metrics(daily_2x)}")

    print(f"\n=== Rupee capital simulation (Rs {args.capital:,.0f} starting) ===")
    equity, trade_log = simulate_capital(daily_returns, costed, args.capital)
    final_capital = equity["capital"].iloc[-1]
    n_years = max((pd.Timestamp(equity["date"].iloc[-1]) - pd.Timestamp(equity["date"].iloc[0])).days / 365.25, 1/365.25)
    cagr = (final_capital / args.capital) ** (1 / n_years) - 1
    running_max = equity["capital"].cummax()
    dd = (equity["capital"] - running_max) / running_max
    print(f"Starting capital: Rs {args.capital:,.0f}")
    print(f"Final capital: Rs {final_capital:,.0f}")
    print(f"Total return: {(final_capital/args.capital - 1)*100:+.2f}%")
    print(f"CAGR: {cagr*100:+.2f}%")
    print(f"Max drawdown: {dd.min()*100:.2f}% (Rs {(equity['capital']-running_max).min():,.0f})")

    os.makedirs(OUT_DIR, exist_ok=True)
    costed.to_csv(os.path.join(OUT_DIR, "swing_reversion_ranked_trades.csv"), index=False)
    equity.to_csv(os.path.join(OUT_DIR, "swing_reversion_ranked_equity_curve.csv"), index=False)
    print(f"\nSaved: {OUT_DIR}/swing_reversion_ranked_*.csv")


if __name__ == "__main__":
    main()
