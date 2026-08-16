"""
Cost model for NSE delivery (CNC) equity trades via Zerodha — DIFFERENT
from costs.py, which models intraday (MIS). Swing positions get delivered
to demat and sold out of it, not squared off same-day, so the charge
structure genuinely differs:

  - Brokerage: Rs 0 on equity delivery (Zerodha-specific; other discount
    brokers may charge — verify for your actual broker before trusting
    this for real capital)
  - STT: 0.1% on BOTH buy and sell (intraday is 0.025%, sell-side only —
    4x the rate, on twice the legs)
  - Stamp duty: 0.015% buy side (intraday is 0.003% — 5x)
  - DP (depository participant) charge: a FLAT fee per scrip per sell day
    (~Rs 15-20 + GST, varies by broker/depository), charged only on
    delivery sells since that's when shares actually move out of demat.
    This is the one intraday NEVER has at all. For a small position this
    can dominate the percentage cost — it does NOT scale with trade size.
  - Exchange transaction charges, SEBI fee: same as intraday, both sides.

Net effect: delivery trades cost more per rupee of turnover than intraday
at typical swing position sizes, particularly for smaller positions where
the flat DP charge is a larger fraction of the trade.
"""

BROKERAGE = 0.0                          # Zerodha: zero brokerage on equity delivery
STT_PCT = 0.001                          # 0.1%, BOTH buy and sell (not sell-only like intraday)
EXCHANGE_TXN_PCT = 0.0000297             # NSE transaction charges, both sides — same as intraday
SEBI_PCT = 0.000001                      # Rs 10/crore, both sides — same as intraday
STAMP_DUTY_BUY_PCT = 0.00015             # 0.015% buy side (vs 0.003% intraday)
DP_CHARGE_FLAT = 15.0                    # Rs per scrip per sell day, varies by broker/depository
GST_PCT = 0.18                           # on (brokerage + exchange txn + DP charge)


def round_trip_cost_pct(buy_price: float, sell_price: float, quantity: int) -> float:
    """Total cost of one round-trip delivery trade (buy + sell), as a
    fraction of turnover (buy_value + sell_value).
    """
    buy_value = buy_price * quantity
    sell_value = sell_price * quantity
    turnover = buy_value + sell_value
    if turnover <= 0:
        return 0.0

    brokerage = 2 * BROKERAGE
    stt = STT_PCT * turnover  # both legs
    exch_txn = EXCHANGE_TXN_PCT * turnover
    sebi = SEBI_PCT * turnover
    stamp = STAMP_DUTY_BUY_PCT * buy_value
    dp = DP_CHARGE_FLAT
    gst = GST_PCT * (brokerage + exch_txn + dp)

    total = brokerage + stt + exch_txn + sebi + stamp + dp + gst
    return total / turnover


def slippage_bps(liquidity_rank: int, universe_size: int = 500, base_bps: float = 3.0,
                  worst_bps: float = 20.0) -> float:
    """Same shape as costs.slippage_bps, slightly lower worst-case — swing
    entries aren't as time-pressured as intraday (you're not racing a
    5-minute bar close), so execution can be a little more patient even on
    less liquid names. Still a coarse approximation.
    """
    rank = max(1, min(liquidity_rank, universe_size))
    frac = (rank - 1) / max(1, universe_size - 1)
    return base_bps + frac * (worst_bps - base_bps)
