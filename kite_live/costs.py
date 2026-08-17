"""
Realistic NSE equity intraday (MIS) transaction cost model, using Zerodha's
published charge structure (this account's actual broker) plus an explicit
slippage assumption, so backtest P&L can be judged net of real costs, not
just the gross price move.

Rates (as of writing, intraday equity, both legs on the same day):
  - Brokerage: 0.03% or Rs 20 per executed order, whichever is LOWER, each leg
  - STT: 0.025% on the SELL leg only
  - Exchange transaction charges (NSE): ~0.00297%, both legs
  - SEBI charges: 0.0001%, both legs
  - Stamp duty: 0.003% on the BUY leg only
  - GST: 18% on (brokerage + exchange txn charges + SEBI charges)
  - Slippage: not a statutory charge -- an assumed fill-quality cost, applied
    both legs, since a real market/limit order won't fill at the exact
    modeled price
"""

BROKERAGE_RATE = 0.0003
BROKERAGE_CAP = 20.0
STT_RATE = 0.00025
EXCH_TXN_RATE = 0.0000297
SEBI_RATE = 0.000001
STAMP_DUTY_RATE = 0.00003
GST_RATE = 0.18
SLIPPAGE_RATE = 0.0003  # 0.03% per fill -- adjust if a stock's real spread differs


def round_trip_cost(buy_value, sell_value):
    brokerage = min(BROKERAGE_CAP, BROKERAGE_RATE * buy_value) + min(BROKERAGE_CAP, BROKERAGE_RATE * sell_value)
    stt = STT_RATE * sell_value
    exch_txn = EXCH_TXN_RATE * (buy_value + sell_value)
    sebi = SEBI_RATE * (buy_value + sell_value)
    stamp = STAMP_DUTY_RATE * buy_value
    gst = GST_RATE * (brokerage + exch_txn + sebi)
    slippage = SLIPPAGE_RATE * (buy_value + sell_value)
    return {
        "brokerage": brokerage, "stt": stt, "exch_txn": exch_txn, "sebi": sebi,
        "stamp": stamp, "gst": gst, "slippage": slippage,
        "total": brokerage + stt + exch_txn + sebi + stamp + gst + slippage,
    }


def position_size(entry_price, risk_per_trade=1000.0, stop_pct=0.5):
    """
    Risk-based sizing: quantity such that hitting the stop loses exactly
    `risk_per_trade` rupees. With a fixed stop_pct this reduces to a fixed
    capital-per-trade of risk_per_trade / (stop_pct/100) -- e.g. Rs 1,000 risk
    at a 0.5% stop implies a ~Rs 2,00,000 position, regardless of the stock's
    own price, which is the point of risk-based sizing.
    """
    risk_per_share = entry_price * (stop_pct / 100.0)
    if risk_per_share <= 0:
        return 1
    return max(int(risk_per_trade / risk_per_share), 1)


def net_trade(trade, quantity):
    """Adds gross/cost/net rupee and % P&L (net of realistic costs) to a trade dict."""
    is_long = trade["direction"] == "Upside"
    buy_price = trade["entry_price"] if is_long else trade["exit_price"]
    sell_price = trade["exit_price"] if is_long else trade["entry_price"]
    buy_value = quantity * buy_price
    sell_value = quantity * sell_price

    gross_pnl = sell_value - buy_value
    cost = round_trip_cost(buy_value, sell_value)["total"]
    net_pnl = gross_pnl - cost
    position_value = quantity * trade["entry_price"]

    out = dict(trade)
    out.update({
        "quantity": quantity,
        "position_value": round(position_value, 2),
        "gross_pnl_rupees": round(gross_pnl, 2),
        "cost_rupees": round(cost, 2),
        "net_pnl_rupees": round(net_pnl, 2),
        "net_pnl_pct": round(net_pnl / position_value * 100, 4) if position_value else 0.0,
        "net_win": net_pnl > 0,
    })
    return out


def aggregate_net(net_trades):
    if not net_trades:
        return {"count": 0}
    wins = [t for t in net_trades if t["net_win"]]
    losses = [t for t in net_trades if not t["net_win"]]
    gross_win_rs = sum(t["net_pnl_rupees"] for t in wins)
    gross_loss_rs = -sum(t["net_pnl_rupees"] for t in losses)
    total_cost_rs = sum(t["cost_rupees"] for t in net_trades)
    return {
        "count": len(net_trades),
        "net_win_rate": round(len(wins) / len(net_trades) * 100, 1),
        "avg_net_pnl_pct": round(sum(t["net_pnl_pct"] for t in net_trades) / len(net_trades), 4),
        "total_gross_pnl_rupees": round(sum(t["gross_pnl_rupees"] for t in net_trades), 2),
        "total_cost_rupees": round(total_cost_rs, 2),
        "total_net_pnl_rupees": round(sum(t["net_pnl_rupees"] for t in net_trades), 2),
        "net_profit_factor": round(gross_win_rs / gross_loss_rs, 2) if gross_loss_rs > 0 else None,
    }
