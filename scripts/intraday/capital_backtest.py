"""
Turns the selected strategy's percentage-return trades into an actual
rupee P&L simulation against a starting capital, day by day.

Position sizing: risk-based, not fixed-notional. Each trade risks
risk_per_trade_pct (default 0.5%) of the CURRENT day-start capital, sized
from the trade's own planned stop distance:

    quantity = (capital * risk_per_trade_pct) / abs(entry_price - stop_price)

This means position size — and therefore risk in rupees — grows and
shrinks with the account, same as real risk management. A leverage cap
(MIS_LEVERAGE, default 5x — typical Zerodha intraday margin for liquid
large-caps; verify against your actual broker/segment before trusting this)
stops the risk-based formula from sizing an unrealistically large position
when a trade's stop happens to be very tight.

Trades within a day are all sized off that day's STARTING capital (not
compounded intraday) — a simplification; real intraday compounding would
require knowing the exact sequencing of fills within the day, which this
backtest's daily-granularity capital tracking doesn't attempt.

Usage:
    python3 capital_backtest.py --capital 1000000
"""

import argparse
import os

import numpy as np
import pandas as pd

from run_backtest import load_bars
from selected_strategy import generate_signals

MIS_LEVERAGE = 5.0
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "intraday")


def simulate(signals: pd.DataFrame, capital0: float, risk_per_trade_pct: float,
             max_concurrent_positions: int, min_stop_pct: float = 0.5, sizing: str = "fixed"):
    """sizing: "fixed" (default) allocates equal capital per trade —
    day_start_capital / max_concurrent_positions, leverage-capped — the
    same implicit assumption the percentage-return research backtest
    made (every trade counted equally in the daily average), so results
    stay explainable against those earlier numbers. "risk" sizes by
    risk_budget/stop_distance instead — sounds more sophisticated, but
    empirically this strategy's stop distance (opening-range width)
    negatively predicts outcome: trades with a naturally tight stop are
    both sized BIGGER by this formula and more likely to get stopped out
    by ordinary noise. Confirmed here: under "risk" sizing, stop-loss
    exits averaged LARGER notional than winners (Rs 4.5L vs Rs 3.3L),
    flipping an equal-weighted +CAGR to a risk-weighted -CAGR. min_stop_pct
    (risk mode only) floors the stop distance used for sizing at this %
    of entry price to blunt that effect, but doesn't fully fix it unless
    set close to the strategy's typical stop width (~1-2%).
    """
    capital = capital0
    equity_rows = []
    trade_rows = []

    for day, day_trades in signals.groupby("date"):
        day_start_capital = capital
        max_notional_per_trade = (day_start_capital / max_concurrent_positions) * MIS_LEVERAGE
        n_today = len(day_trades)
        day_pnl = 0.0

        for row in day_trades.itertuples():
            if sizing == "equal_weight":
                # Replicates the research backtest's own methodology exactly:
                # that day's return = mean of that day's trade returns, i.e.
                # capital split evenly across however many trades actually
                # fired that day (not a fixed per-slot amount) — this is the
                # apples-to-apples comparison against the earlier % numbers.
                quantity = int((day_start_capital / n_today) / row.entry_price)
            elif sizing == "fixed":
                quantity = int(max_notional_per_trade / row.entry_price)
            else:
                raw_risk_per_share = abs(row.entry_price - row.stop_price)
                if raw_risk_per_share <= 0:
                    continue
                risk_per_share = max(raw_risk_per_share, row.entry_price * min_stop_pct / 100)
                risk_amount = day_start_capital * risk_per_trade_pct / 100
                qty_by_risk = risk_amount / risk_per_share
                qty_by_leverage = max_notional_per_trade / row.entry_price
                quantity = int(min(qty_by_risk, qty_by_leverage))
            if quantity <= 0:
                continue

            notional = quantity * row.entry_price
            pnl = notional * row.net_return
            day_pnl += pnl

            trade_rows.append({
                "date": day, "symbol": row.symbol, "direction": row.direction,
                "entry_time": row.entry_time, "entry_price": row.entry_price,
                "exit_time": row.exit_time, "exit_price": row.exit_price,
                "exit_reason": row.exit_reason, "quantity": quantity,
                "notional": round(notional, 2), "net_return_pct": round(row.net_return * 100, 3),
                "pnl": round(pnl, 2), "capital_before": round(day_start_capital, 2),
            })

        capital = day_start_capital + day_pnl
        equity_rows.append({
            "date": day, "capital": round(capital, 2), "day_pnl": round(day_pnl, 2),
            "trades": len(day_trades),
        })

    return pd.DataFrame(equity_rows), pd.DataFrame(trade_rows)


def summarize(equity: pd.DataFrame, trades: pd.DataFrame, capital0: float):
    if equity.empty:
        return {"error": "no trades generated — nothing to simulate"}

    final_capital = equity["capital"].iloc[-1]
    total_return_pct = (final_capital / capital0 - 1) * 100

    dates = pd.to_datetime(equity["date"])
    n_years = max((dates.iloc[-1] - dates.iloc[0]).days / 365.25, 1 / 365.25)
    cagr = ((final_capital / capital0) ** (1 / n_years) - 1) * 100

    cum = equity["capital"]
    running_max = cum.cummax()
    drawdown_pct = (cum - running_max) / running_max * 100
    max_dd_pct = drawdown_pct.min()
    max_dd_rupees = (cum - running_max).min()

    win_rate = (trades["pnl"] > 0).mean() * 100 if not trades.empty else 0
    avg_win = trades.loc[trades["pnl"] > 0, "pnl"].mean() if (trades["pnl"] > 0).any() else 0
    avg_loss = trades.loc[trades["pnl"] <= 0, "pnl"].mean() if (trades["pnl"] <= 0).any() else 0

    return {
        "Starting capital": f"Rs {capital0:,.0f}",
        "Final capital": f"Rs {final_capital:,.0f}",
        "Total return": f"{total_return_pct:+.2f}%",
        "CAGR": f"{cagr:+.2f}%",
        "Max drawdown": f"{max_dd_pct:.2f}% (Rs {max_dd_rupees:,.0f})",
        "Total trades": len(trades),
        "Trading days": len(equity),
        "Win rate": f"{win_rate:.1f}%",
        "Avg win": f"Rs {avg_win:,.0f}",
        "Avg loss": f"Rs {avg_loss:,.0f}",
        "Best day": f"Rs {equity['day_pnl'].max():,.0f}",
        "Worst day": f"Rs {equity['day_pnl'].min():,.0f}",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--capital", type=float, default=1_000_000, help="starting capital in Rs (default: 10 lacs)")
    parser.add_argument("--risk-per-trade-pct", type=float, default=0.5)
    parser.add_argument("--min-stop-pct", type=float, default=0.5,
                         help="floor the stop distance used for position sizing at this %% of entry price "
                              "(default 0.5%%) — prevents an unusually tight technical stop from oversizing a trade")
    parser.add_argument("--max-concurrent-positions", type=int, default=4)
    parser.add_argument("--sizing", choices=["fixed", "risk", "equal_weight"], default="fixed",
                         help="position sizing method — see simulate() docstring (default: fixed). "
                              "'equal_weight' reproduces the earlier %% backtest's own methodology exactly, "
                              "for direct comparison")
    args = parser.parse_args()

    bars, rank = load_bars()
    if not bars:
        raise SystemExit("No bar data found. Run kite_fetch_intraday.py first.")

    print(f"Generating signals from the selected strategy across the full backtest period...")
    signals = generate_signals(bars, rank)
    if signals.empty:
        raise SystemExit("No signals generated — nothing to simulate.")
    print(f"{len(signals)} signals across {signals['date'].nunique()} trading days.\n")

    equity, trades = simulate(signals, args.capital, args.risk_per_trade_pct,
                               args.max_concurrent_positions, args.min_stop_pct, args.sizing)
    summary = summarize(equity, trades, args.capital)

    print("=== Capital backtest summary ===")
    for k, v in summary.items():
        print(f"{k}: {v}")

    equity_path = os.path.join(OUT_DIR, "capital_equity_curve.csv")
    trades_path = os.path.join(OUT_DIR, "capital_trade_log.csv")
    equity.to_csv(equity_path, index=False)
    trades.to_csv(trades_path, index=False)
    print(f"\nSaved: {equity_path}")
    print(f"Saved: {trades_path}")


if __name__ == "__main__":
    main()
