"""
Turns a strategy's raw trade list (entry/exit prices, no costs) into net
returns per trade, then aggregates across symbols into a daily portfolio
equity curve: each day, capital is split equally across every trade
triggered that day (assumes enough capital to take every signal — flag this
when position counts get large, since real capital/margin would cap it).
"""

import numpy as np
import pandas as pd

from costs import round_trip_cost_pct, slippage_bps


def apply_costs(trades: pd.DataFrame, liquidity_rank: int = 1, universe_size: int = 500,
                 quantity: int = 10) -> pd.DataFrame:
    if trades.empty:
        return trades.assign(gross_return=[], net_return=[])

    trades = trades.copy()
    gross = trades["direction"] * (trades["exit_price"] - trades["entry_price"]) / trades["entry_price"]

    cost_pct = trades.apply(
        lambda t: round_trip_cost_pct(t["entry_price"], t["exit_price"], quantity), axis=1
    )
    slip_pct = slippage_bps(liquidity_rank, universe_size) / 10000

    trades["gross_return"] = gross
    trades["net_return"] = gross - cost_pct - slip_pct
    return trades


def portfolio_daily_returns(all_trades: dict) -> pd.Series:
    """all_trades: {symbol: trades_df_with_net_return}. Returns a daily
    return series equal-weighting every trade triggered that day, across
    all symbols and both directions.
    """
    frames = []
    for symbol, trades in all_trades.items():
        if trades is None or trades.empty:
            continue
        t = trades[["date", "net_return"]].copy()
        t["symbol"] = symbol
        frames.append(t)

    if not frames:
        return pd.Series(dtype=float)

    combined = pd.concat(frames, ignore_index=True)
    daily = combined.groupby("date")["net_return"].mean()
    daily.index = pd.to_datetime(daily.index)
    daily = daily.sort_index()
    return daily


def equity_curve(daily_returns: pd.Series, start_value: float = 1.0) -> pd.Series:
    return start_value * (1 + daily_returns.fillna(0)).cumprod()
