"""
Realistic per-trade cost model for NSE intraday equity via a discount broker
(Zerodha-style charges as of 2026; verify against the live brokerage
calculator before trusting these for real capital — rates change).

All charges are round-trip (one buy leg + one sell leg) unless noted.
Returned as a fraction of trade turnover (buy_value + sell_value), so it can
be subtracted directly from a trade's percentage return.
"""

BROKERAGE_PER_ORDER_FLAT = 20.0          # Rs, or 0.03% of order value, whichever is lower
BROKERAGE_PCT = 0.0003
STT_SELL_PCT = 0.00025                   # STT on intraday equity, sell side only
EXCHANGE_TXN_PCT = 0.0000297             # NSE transaction charges, both sides
SEBI_PCT = 0.000001                      # SEBI turnover fee: Rs 10/crore = 0.0001%, both sides
STAMP_DUTY_BUY_PCT = 0.00003             # stamp duty, buy side only
GST_PCT = 0.18                           # on (brokerage + exchange txn charges)
# DP charges do NOT apply — intraday square-off never touches the demat account.


def round_trip_cost_pct(buy_price: float, sell_price: float, quantity: int) -> float:
    """Total cost of one round-trip (buy+sell) trade, as a fraction of turnover.

    turnover = buy_value + sell_value (both legs).
    """
    buy_value = buy_price * quantity
    sell_value = sell_price * quantity
    turnover = buy_value + sell_value
    if turnover <= 0:
        return 0.0

    brokerage = 2 * min(BROKERAGE_PER_ORDER_FLAT, BROKERAGE_PCT * buy_value)
    stt = STT_SELL_PCT * sell_value
    exch_txn = EXCHANGE_TXN_PCT * turnover
    sebi = SEBI_PCT * turnover
    stamp = STAMP_DUTY_BUY_PCT * buy_value
    gst = GST_PCT * (brokerage + exch_txn)

    total = brokerage + stt + exch_txn + sebi + stamp + gst
    return total / turnover


def round_trip_cost_bps_estimate(price: float = 1000.0, quantity: int = 10) -> float:
    """Convenience: approximate round-trip cost in bps for a typical trade size,
    used when we want a flat bps assumption instead of per-trade exact calc."""
    return round_trip_cost_pct(price, price, quantity) * 10000


def slippage_bps(liquidity_rank: int, universe_size: int = 500, base_bps: float = 3.0,
                  worst_bps: float = 25.0) -> float:
    """Model market-impact slippage as increasing with illiquidity.

    liquidity_rank: 1 = most liquid stock in the universe, universe_size = least liquid.
    Linear interpolation between base_bps (most liquid) and worst_bps (least liquid).
    This is a coarse approximation, not a calibrated impact model — treat results
    for the bottom of the universe as more uncertain than the top.
    """
    rank = max(1, min(liquidity_rank, universe_size))
    frac = (rank - 1) / max(1, universe_size - 1)
    return base_bps + frac * (worst_bps - base_bps)
