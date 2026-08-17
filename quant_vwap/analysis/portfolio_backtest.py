"""
Portfolio-level backtest of the single recommended strategy, with real capital,
position sizing, a concurrency cap, and transaction costs — on top of the
per-trade signals already computed in results/backtest_trades.parquet.

Strategy (final, both directions combined): 1-minute bars, delta% crosses
>=4% away from session VWAP in EITHER direction -> trade the reversion
(short if above, long if below), target 1.5%, stop 0.5%, EOD square-off.
Chosen over the narrower/tighter-threshold cells because it has the best
gross risk-adjusted profile (Sharpe, drawdown) of everything tested, more
trades than either single direction alone (more statistically reliable),
and is the simplest rule to state: any 4%+ deviation from VWAP is faded.

Position sizing: 10% of current realized equity per trade, capped at 10
concurrent open positions (so up to 100% of capital deployed at once). No
leverage/margin modeled — each position is backed 1:1 by capital.

Costs: a single round-trip cost assumption (brokerage + STT + exchange +
GST + stamp duty combined), applied as a flat % of trade notional. Run at
two scenarios: 0.05% (optimistic/low-cost execution) and 0.08% (realistic
all-in retail cost for NSE intraday equity), because this strategy's gross
edge is thin enough that the cost assumption changes the verdict.
"""
import json

import numpy as np
import pandas as pd

REPO_ROOT = "/home/user/vinayshinde/quant_vwap"
INITIAL_CAPITAL = 1_000_000.0
MAX_CONCURRENT = 10
SLOT_FRACTION = 0.10          # 10% of current equity per position
ROUND_TRIP_COST_PCT = 0.05    # overridden per run in main()

STRATEGY = dict(strategy_variant="baseline", timeframe="1minute", threshold=4.0,
                 target_pct=1.5, stop_pct=0.5)  # both directions (above + below)


def load_trades():
    trades = pd.read_parquet(f"{REPO_ROOT}/results/backtest_trades.parquet")
    sel = trades
    for k, v in STRATEGY.items():
        sel = sel[sel[k] == v]
    return sel.sort_values("entry_time").reset_index(drop=True)


def simulate(sel: pd.DataFrame):
    equity = INITIAL_CAPITAL
    open_positions = []  # list of dicts: {exit_time, reserved, net_pnl}
    equity_curve = []    # (timestamp, equity) at every close event
    trade_log = []
    skipped = 0

    for _, r in sel.iterrows():
        # release any positions that closed before this new signal's entry
        still_open = []
        for pos in open_positions:
            if pos["exit_time"] <= r.entry_time:
                equity += pos["net_pnl"]
                equity_curve.append((pos["exit_time"], equity))
            else:
                still_open.append(pos)
        open_positions = still_open

        if len(open_positions) >= MAX_CONCURRENT:
            skipped += 1
            continue

        reserved = equity * SLOT_FRACTION
        qty = int(reserved // r.entry_price)
        if qty <= 0:
            skipped += 1
            continue

        notional = qty * r.entry_price
        gross_pnl = notional * (r.pnl_pct / 100.0)
        cost = notional * (ROUND_TRIP_COST_PCT / 100.0)
        net_pnl = gross_pnl - cost

        open_positions.append({"exit_time": r.exit_time, "net_pnl": net_pnl})
        trade_log.append({
            "symbol": r.symbol, "entry_time": r.entry_time, "exit_time": r.exit_time,
            "entry_price": r.entry_price, "exit_price": r.exit_price, "qty": qty,
            "notional": notional, "gross_pnl": gross_pnl, "cost": cost, "net_pnl": net_pnl,
            "exit_reason": r.exit_reason,
        })

    # flush remaining open positions
    for pos in sorted(open_positions, key=lambda p: p["exit_time"]):
        equity += pos["net_pnl"]
        equity_curve.append((pos["exit_time"], equity))

    return equity, pd.DataFrame(trade_log), pd.DataFrame(equity_curve, columns=["timestamp", "equity"]), skipped


def compute_metrics(final_equity, trade_log, equity_curve):
    n_trades = len(trade_log)
    n_days = (trade_log.exit_time.max() - trade_log.entry_time.min()).days
    years = n_days / 365.25
    total_return_pct = (final_equity / INITIAL_CAPITAL - 1) * 100
    cagr_pct = ((final_equity / INITIAL_CAPITAL) ** (1 / years) - 1) * 100

    win_rate = (trade_log.net_pnl > 0).mean() * 100
    avg_win = trade_log.loc[trade_log.net_pnl > 0, "net_pnl"].mean()
    avg_loss = trade_log.loc[trade_log.net_pnl <= 0, "net_pnl"].mean()
    profit_factor = trade_log.loc[trade_log.net_pnl > 0, "net_pnl"].sum() / -trade_log.loc[trade_log.net_pnl <= 0, "net_pnl"].sum()
    total_costs = trade_log.cost.sum()

    eq = equity_curve.equity.to_numpy()
    running_max = np.maximum.accumulate(eq)
    drawdown_pct = (eq - running_max) / running_max * 100
    max_dd_pct = drawdown_pct.min()

    daily_eq = equity_curve.set_index("timestamp")["equity"].resample("1D").last().ffill()
    daily_ret = daily_eq.pct_change().dropna()
    sharpe = (daily_ret.mean() / daily_ret.std()) * np.sqrt(252) if daily_ret.std() > 0 else np.nan

    return dict(
        initial_capital=INITIAL_CAPITAL, final_equity=round(final_equity, 2),
        total_return_pct=round(total_return_pct, 2), cagr_pct=round(cagr_pct, 2),
        years=round(years, 2), n_trades=n_trades,
        win_rate_pct=round(win_rate, 2), avg_win_rupees=round(avg_win, 2),
        avg_loss_rupees=round(avg_loss, 2), profit_factor=round(profit_factor, 3),
        max_drawdown_pct=round(max_dd_pct, 2), sharpe_annualized=round(sharpe, 3),
        total_costs_rupees=round(total_costs, 2),
    )


def run_scenario(sel, cost_pct, label):
    global ROUND_TRIP_COST_PCT
    ROUND_TRIP_COST_PCT = cost_pct

    final_equity, trade_log, equity_curve, skipped = simulate(sel)
    metrics = compute_metrics(final_equity, trade_log, equity_curve)
    metrics["signals_total"] = len(sel)
    metrics["signals_skipped_concurrency_cap"] = skipped
    metrics["max_concurrent"] = MAX_CONCURRENT
    metrics["slot_fraction"] = SLOT_FRACTION
    metrics["round_trip_cost_pct"] = cost_pct
    metrics["scenario"] = label

    print(f"\n=== {label} (cost={cost_pct}%) ===")
    print(json.dumps(metrics, indent=2, default=str))

    trade_log.to_csv(f"{REPO_ROOT}/results/summary/final_strategy_trade_log_{label}.csv", index=False)

    daily = equity_curve.set_index("timestamp")["equity"].resample("1D").last().ffill().reset_index()
    daily.to_csv(f"{REPO_ROOT}/results/summary/final_strategy_equity_curve_{label}.csv", index=False)

    with open(f"{REPO_ROOT}/results/summary/final_strategy_metrics_{label}.json", "w") as f:
        json.dump(metrics, f, indent=2, default=str)

    monthly_eq = equity_curve.set_index("timestamp")["equity"].resample("ME").last().ffill()
    monthly_ret = (monthly_eq.pct_change().dropna() * 100).reset_index()
    monthly_ret.columns = ["month", "return_pct"]
    monthly_ret.to_csv(f"{REPO_ROOT}/results/summary/final_strategy_monthly_returns_{label}.csv", index=False)

    return metrics


def main():
    sel = load_trades()
    print(f"Combined (both directions) 4% threshold, 1min, 1.5:0.5 signals: {len(sel)}")

    m_opt = run_scenario(sel, 0.05, "optimistic")
    m_real = run_scenario(sel, 0.08, "realistic")

    summary = pd.DataFrame([m_opt, m_real])
    summary.to_csv(f"{REPO_ROOT}/results/summary/final_strategy_scenario_comparison.csv", index=False)
    print("\n=== Scenario comparison ===")
    print(summary[["scenario","round_trip_cost_pct","n_trades","total_return_pct","cagr_pct",
                    "max_drawdown_pct","win_rate_pct","profit_factor","sharpe_annualized"]].to_string())


if __name__ == "__main__":
    main()
