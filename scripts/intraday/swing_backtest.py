"""
Full backtest of the swing breakout strategy (swing_strategy.py) across
the NSE 500 universe: entry/exit rules -> portfolio caps -> delivery
costs -> two complete results: a percentage-return research backtest
(walk-forward CAGR/Sharpe/drawdown, cost sensitivity) and a real rupee
capital simulation.

Unlike the intraday backtests (each trade opens and closes the same day),
swing trades span multiple days, so both results use proper day-by-day
MARK-TO-MARKET, not just booking each trade's return on its exit day —
lumping a multi-day move into one day would understate the actual
volatility and drawdown a trader lived through while holding it.

Position sizing for both results: EQUAL-WEIGHT across that day's
concurrently open positions (capital split evenly, re-split as positions
open/close). This is deliberate, not a simplification of convenience —
an earlier capital backtest on the intraday strategy found that naive
risk-based (stop-distance) sizing systematically OVERSIZES the trades
most likely to lose (a tight stop implies both a bigger size from
risk_budget/stop_distance AND a higher chance of getting hit by ordinary
noise). Equal-weight sidesteps that failure mode entirely and is also
the same methodology the walk-forward research numbers already assume,
so the two results stay directly comparable.

Usage:
    python3 swing_backtest.py --capital 1000000
"""

import argparse
import glob
import os

import numpy as np
import pandas as pd

from scanner_swing_breakout import to_daily
from swing_strategy import find_swing_trades
from swing_costs import round_trip_cost_pct, slippage_bps

INTRADAY_BARS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "intraday", "bars")
DAILY_BARS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "daily", "bars")
UNIVERSE_FILE = os.path.join(os.path.dirname(__file__), "..", "..", "data", "intraday", "universe_top500.csv")
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "intraday")

MAX_CONCURRENT_POSITIONS = 10
MAX_NEW_SIGNALS_PER_DAY = 3
TRADING_DAYS = 252


def load_bars_and_rank(source: str = "daily"):
    """source="daily": genuine daily bars from kite_fetch_daily.py (years of
    history, one row per trading day). source="intraday": daily bars
    DERIVED from the 5-minute intraday cache (only as many days as that
    cache covers — a few months at most) — kept for comparison against the
    earlier short-window backtest, not for real multi-year testing.
    """
    universe = pd.read_csv(UNIVERSE_FILE)
    rank = dict(zip(universe["tradingsymbol"], universe["liquidity_rank"]))
    bars_dir = DAILY_BARS_DIR if source == "daily" else INTRADAY_BARS_DIR
    bars = {}
    for path in sorted(glob.glob(os.path.join(bars_dir, "*.parquet"))):
        symbol = os.path.splitext(os.path.basename(path))[0]
        df = pd.read_parquet(path)
        bars[symbol] = df if source == "daily" else to_daily(df)
    return bars, rank


def load_nifty_daily():
    path = os.path.join(os.path.dirname(__file__), "..", "..", "data", "daily", "NIFTY50.parquet")
    if not os.path.exists(path):
        raise SystemExit(f"NIFTY50 benchmark data not found at {path} — fetch it first.")
    return pd.read_parquet(path)


def generate_all_trades(daily_by_symbol: dict, rank: dict, nifty_daily: pd.DataFrame, **strategy_kwargs) -> pd.DataFrame:
    frames = []
    for symbol, daily in daily_by_symbol.items():
        trades = find_swing_trades(daily, nifty_daily, **strategy_kwargs)
        if trades.empty:
            continue
        trades.insert(0, "symbol", symbol)
        trades.insert(1, "liquidity_rank", rank.get(symbol, len(rank) + 1))
        frames.append(trades)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def apply_portfolio_caps(trades: pd.DataFrame, max_concurrent: int, max_new_per_day: int) -> pd.DataFrame:
    """Two caps, same reasoning as scanner_swing_breakout.py: portfolio
    capacity (concurrency, using the trade's REAL entry/exit dates now —
    more accurate than the scanner's approximate placeholder) and daily
    review bandwidth (new entries/day), both liquidity-priority tie-break.
    """
    df = trades.sort_values(["entry_date", "liquidity_rank"]).copy()

    # daily signal cap first (bounds how many even compete for concurrency slots)
    if max_new_per_day > 0:
        df = df.groupby("entry_date", group_keys=False).head(max_new_per_day)
        df = df.sort_values(["entry_date", "liquidity_rank"])

    if max_concurrent <= 0:
        return df.reset_index(drop=True)

    open_positions = []  # list of exit_date for currently held positions
    kept_idx = []
    for entry_date, batch in df.groupby("entry_date", sort=True):
        open_positions = [d for d in open_positions if d > entry_date]
        for row in batch.itertuples():
            if len(open_positions) >= max_concurrent:
                continue
            kept_idx.append(row.Index)
            open_positions.append(row.exit_date)

    return df.loc[kept_idx].reset_index(drop=True)


def apply_costs_to_trades(trades: pd.DataFrame, universe_size: int, notional_per_trade: float = 100_000) -> pd.DataFrame:
    df = trades.copy()
    cost_pct = df.apply(
        lambda r: round_trip_cost_pct(r["entry_price"], r["exit_price"],
                                       max(1, int(notional_per_trade / r["entry_price"]))),
        axis=1)
    slip_pct = df["liquidity_rank"].apply(lambda r: slippage_bps(r, universe_size) / 10000)
    df["net_return"] = df["gross_return"] - cost_pct - slip_pct
    return df


def build_daily_mtm_returns(trades: pd.DataFrame, daily_by_symbol: dict) -> pd.Series:
    """Proper day-by-day mark-to-market, equal-weighted across that day's
    concurrently open positions. Cost/slippage drag (already baked into
    each trade's net_return vs gross_return) is applied entirely on the
    exit day — a one-time realized drag, not smeared across the holding
    period.
    """
    day_contributions = {}  # date -> list of that day's per-position returns

    for row in trades.itertuples():
        sym_daily = daily_by_symbol[row.symbol]
        cost_drag = row.gross_return - row.net_return
        closes = sym_daily["close"]

        if row.entry_date == row.exit_date:
            ret = row.gross_return - cost_drag
            day_contributions.setdefault(row.entry_date, []).append(ret)
            continue

        span = closes.loc[row.entry_date:row.exit_date]
        span_dates = span.index
        for i, day in enumerate(span_dates):
            if day == row.entry_date:
                ret = (span.iloc[i] - row.entry_price) / row.entry_price
            elif day == row.exit_date:
                prev_close = span.iloc[i - 1]
                ret = (row.exit_price - prev_close) / prev_close - cost_drag
            else:
                ret = (span.iloc[i] - span.iloc[i - 1]) / span.iloc[i - 1]
            day_contributions.setdefault(day, []).append(ret)

    daily = {day: float(np.mean(rets)) for day, rets in day_contributions.items()}
    return pd.Series(daily).sort_index()


def performance_metrics(daily_returns: pd.Series) -> dict:
    if daily_returns.empty or len(daily_returns) < 2:
        return {"CAGR": None, "Sharpe": None, "MaxDrawdown": None, "Days": 0}
    equity = (1 + daily_returns.fillna(0)).cumprod()
    n_years = max((daily_returns.index[-1] - daily_returns.index[0]).days / 365.25, 1 / 365.25)
    cagr = equity.iloc[-1] ** (1 / n_years) - 1
    vol = daily_returns.std() * np.sqrt(TRADING_DAYS)
    sharpe = (daily_returns.mean() * TRADING_DAYS) / vol if vol > 0 else np.nan
    dd = equity / equity.cummax() - 1
    return {
        "CAGR": f"{cagr:.2%}", "Sharpe": f"{sharpe:.2f}",
        "MaxDrawdown": f"{dd.min():.2%}", "Days": len(daily_returns),
    }


def walk_forward_split(daily_returns: pd.Series, train_frac: float = 0.7):
    n = len(daily_returns)
    cut = int(n * train_frac)
    return daily_returns.iloc[:cut], daily_returns.iloc[cut:]


def simulate_capital(daily_returns: pd.Series, trades: pd.DataFrame, capital0: float) -> tuple:
    """Rupee capital curve, derived DIRECTLY from the already-computed
    equal-weighted daily_returns series — not a second, independent
    position-tracking simulation. Two separate simulations that are
    supposed to use the same methodology (equal-weight, mark-to-market)
    are exactly the kind of thing that quietly drifts apart into two
    different, hard-to-reconcile answers if implemented twice; deriving
    the capital curve from the one already-built return series guarantees
    they can't disagree.

    Also produces a per-trade rupee P&L approximation for a trade log:
    position size = capital at that trade's entry date / max concurrent
    positions (the same equal-weight assumption), applied to the trade's
    own net_return. This is an approximation (real equal-weight sizing
    depends on exactly how many positions are open that specific day, not
    a flat max-concurrent denominator) — good enough for a per-trade
    reference log, not re-derived for the headline capital numbers above.
    """
    equity = (capital0 * (1 + daily_returns.fillna(0)).cumprod()).rename("capital")
    equity_df = equity.reset_index()
    equity_df.columns = ["date", "capital"]
    equity_df["day_pnl"] = equity_df["capital"].diff().fillna(equity_df["capital"].iloc[0] - capital0)

    capital_at_entry = equity.reindex(trades["entry_date"]).fillna(capital0).values
    approx_position_size = capital_at_entry / MAX_CONCURRENT_POSITIONS
    trade_log = trades[["symbol", "entry_date", "exit_date", "entry_price", "exit_price",
                         "exit_reason", "net_return"]].copy()
    trade_log["approx_position_size_rs"] = approx_position_size.round(0)
    trade_log["approx_net_pnl_rs"] = (approx_position_size * trades["net_return"].values).round(0)

    return equity_df, trade_log


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--capital", type=float, default=1_000_000)
    parser.add_argument("--source", choices=["daily", "intraday"], default="daily",
                         help="'daily': genuine multi-year daily bars from kite_fetch_daily.py (default). "
                              "'intraday': daily bars derived from the 5-minute cache, only a few months.")
    args = parser.parse_args()

    daily_by_symbol, rank = load_bars_and_rank(args.source)
    if not daily_by_symbol:
        raise SystemExit(f"No {args.source} bar data found. Run kite_fetch_{args.source}.py first.")
    universe_size = len(rank)
    span = [d for daily in daily_by_symbol.values() for d in daily.index[[0, -1]]]
    print(f"Loaded {len(daily_by_symbol)} symbols, {args.source} source, "
          f"{min(span).date()} to {max(span).date()}.")
    nifty_daily = load_nifty_daily()
    print(f"NIFTY50 benchmark: {nifty_daily.index.min().date()} to {nifty_daily.index.max().date()}.")
    print("Generating swing trades across the universe...")
    raw_trades = generate_all_trades(daily_by_symbol, rank, nifty_daily)
    print(f"Raw trades generated (pre-cap): {len(raw_trades)}")

    capped = apply_portfolio_caps(raw_trades, MAX_CONCURRENT_POSITIONS, MAX_NEW_SIGNALS_PER_DAY)
    print(f"After portfolio caps (max {MAX_CONCURRENT_POSITIONS} concurrent, "
          f"{MAX_NEW_SIGNALS_PER_DAY}/day): {len(capped)}")

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

    # cost sensitivity: 2x costs
    costed_2x = capped.copy()
    cost_pct_orig = costed["gross_return"] - costed["net_return"]
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
    suffix = f"_{args.source}"
    costed.to_csv(os.path.join(OUT_DIR, f"swing_strategy_trades{suffix}.csv"), index=False)
    equity.to_csv(os.path.join(OUT_DIR, f"swing_capital_equity_curve{suffix}.csv"), index=False)
    trade_log.to_csv(os.path.join(OUT_DIR, f"swing_capital_trade_log{suffix}.csv"), index=False)
    print(f"\nSaved: {OUT_DIR}/swing_*{suffix}.csv")


if __name__ == "__main__":
    main()
