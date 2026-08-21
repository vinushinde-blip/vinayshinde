"""
Scanner2 — Price-Contraction / Sell-Absorption scanner on Kite live ticks.

Idea
----
Watch for stocks where sellers are clearly in control (order-book imbalance
is negative and getting MORE negative minute over minute) but the price
still refuses to fall and its range is contracting. That combination reads
as supply being absorbed quietly. History says that once the imbalance
flips positive from there, price tends to move fast — so this scanner's
job is to surface the "still contracting, not falling" stocks *before*
that flip, so you can watch them and be ready to buy the confirmation.

It does three things per instrument, per completed 1-minute candle:
  1. Build 1-min OHLC candles from live LTP ticks.
  2. Track order-book imbalance per tick from top-5 market depth
     (buy_qty - sell_qty) / (buy_qty + sell_qty), averaged per candle.
  3. Every time a candle closes, evaluate the last WINDOW candles for:
       a. Range contraction  - candle ranges are shrinking / below average.
       b. Imbalance          - negative and trending more negative (rising
                                sell pressure).
       c. Price resilience   - close is NOT making a new low across the
                                window (i.e. price isn't actually falling).
     All three together -> "WATCH" alert.
     A separate, louder "BUY TRIGGER" alert fires the moment imbalance
     flips from negative to positive while price is at/above the window's
     high — the "explosion" confirmation.

Setup
-----
  pip install kiteconnect

  export KITE_API_KEY=...
  export KITE_ACCESS_TOKEN=...      # from the daily login flow
  export KITE_WATCHLIST=RELIANCE,TCS,INFY,HDFCBANK   # NSE tradingsymbols

Run
---
  python scripts/kite_scanner2.py

Notes
-----
- Requires tick subscription in FULL mode (KiteTicker.MODE_FULL) since
  market depth is only present in full-mode ticks.
- All thresholds below are starting points, not calibrated constants —
  tune CONTRACTION_LOOKBACK / IMBALANCE_THRESHOLD / WINDOW to taste once
  you see how it behaves against real tick flow.
"""

import logging
import os
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime

from kiteconnect import KiteConnect, KiteTicker

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("scanner2")

# ---- tunables -----------------------------------------------------------

WINDOW = 5              # number of closed 1-min candles evaluated per scan
IMBALANCE_THRESHOLD = -0.15   # imbalance must be at least this negative to count as "sell pressure"
IMBALANCE_RISE_MARGIN = -0.02  # each successive candle's imbalance must be <= previous - margin
CONTRACTION_RATIO = 0.85       # latest candle range must be <= this * avg range of prior candles in window

# ---------------------------------------------------------------------


@dataclass
class MinuteBar:
    minute: datetime
    open: float
    high: float
    low: float
    close: float
    imbalance_sum: float = 0.0
    imbalance_ticks: int = 0

    @property
    def range(self) -> float:
        return self.high - self.low

    @property
    def avg_imbalance(self) -> float:
        if self.imbalance_ticks == 0:
            return 0.0
        return self.imbalance_sum / self.imbalance_ticks


@dataclass
class InstrumentState:
    symbol: str
    token: int
    bars: deque = field(default_factory=lambda: deque(maxlen=WINDOW + 1))
    _current: MinuteBar = None
    last_alert_minute: datetime = None
    last_trigger_minute: datetime = None

    def on_tick(self, ltp: float, depth: dict) -> None:
        now_minute = datetime.now().replace(second=0, microsecond=0)
        imbalance = _depth_imbalance(depth)

        if self._current is None or self._current.minute != now_minute:
            if self._current is not None:
                self.bars.append(self._current)
            self._current = MinuteBar(now_minute, ltp, ltp, ltp, ltp)

        bar = self._current
        bar.high = max(bar.high, ltp)
        bar.low = min(bar.low, ltp)
        bar.close = ltp
        if imbalance is not None:
            bar.imbalance_sum += imbalance
            bar.imbalance_ticks += 1

    def closed_bars(self):
        """Bars fully closed, most recent last. Excludes the in-progress bar."""
        return list(self.bars)


def _depth_imbalance(depth: dict) -> float:
    """(buy_qty - sell_qty) / (buy_qty + sell_qty) from top-5 market depth."""
    if not depth:
        return None
    buy_qty = sum(level["quantity"] for level in depth.get("buy", []))
    sell_qty = sum(level["quantity"] for level in depth.get("sell", []))
    total = buy_qty + sell_qty
    if total == 0:
        return None
    return (buy_qty - sell_qty) / total


def _is_contracting(bars: list) -> bool:
    ranges = [b.range for b in bars]
    latest = ranges[-1]
    prior = ranges[:-1]
    if not prior or latest == 0:
        return False
    avg_prior = sum(prior) / len(prior)
    return avg_prior > 0 and latest <= avg_prior * CONTRACTION_RATIO


def _is_rising_negative_imbalance(bars: list) -> bool:
    imbalances = [b.avg_imbalance for b in bars]
    if imbalances[-1] > IMBALANCE_THRESHOLD:
        return False
    for prev, curr in zip(imbalances, imbalances[1:]):
        if curr > prev + IMBALANCE_RISE_MARGIN:
            return False
    return True


def _is_price_resilient(bars: list) -> bool:
    closes = [b.close for b in bars]
    return closes[-1] >= min(closes)


def _flips_positive(bars: list) -> bool:
    if len(bars) < 2:
        return False
    prev, curr = bars[-2], bars[-1]
    return prev.avg_imbalance <= IMBALANCE_THRESHOLD and curr.avg_imbalance > 0 and curr.close >= max(b.high for b in bars)


def evaluate(state: InstrumentState) -> None:
    bars = state.closed_bars()
    if len(bars) < WINDOW:
        return

    window = bars[-WINDOW:]
    latest_minute = window[-1].minute

    if _flips_positive(window) and state.last_trigger_minute != latest_minute:
        state.last_trigger_minute = latest_minute
        log.info(
            "\033[92m[BUY TRIGGER] %-12s imbalance flipped positive at close=%.2f (window %s -> %s)\033[0m",
            state.symbol, window[-1].close, window[0].minute.strftime("%H:%M"), latest_minute.strftime("%H:%M"),
        )
        return

    if (
        _is_contracting(window)
        and _is_rising_negative_imbalance(window)
        and _is_price_resilient(window)
        and state.last_alert_minute != latest_minute
    ):
        state.last_alert_minute = latest_minute
        log.info(
            "[WATCH] %-12s range contracting, imbalance %.2f -> %.2f (selling), price holding (closes %s)",
            state.symbol,
            window[0].avg_imbalance,
            window[-1].avg_imbalance,
            [round(b.close, 2) for b in window],
        )


class Scanner2:
    def __init__(self, api_key: str, access_token: str, watchlist: list):
        self.kite = KiteConnect(api_key=api_key)
        self.kite.set_access_token(access_token)
        self.ticker = KiteTicker(api_key, access_token)
        self.states: dict[int, InstrumentState] = {}
        self._resolve_tokens(watchlist)

        self.ticker.on_ticks = self._on_ticks
        self.ticker.on_connect = self._on_connect
        self.ticker.on_close = self._on_close

    def _resolve_tokens(self, watchlist: list) -> None:
        instruments = self.kite.instruments("NSE")
        by_symbol = {i["tradingsymbol"]: i["instrument_token"] for i in instruments}
        for symbol in watchlist:
            token = by_symbol.get(symbol)
            if token is None:
                log.warning("Symbol not found on NSE, skipping: %s", symbol)
                continue
            self.states[token] = InstrumentState(symbol=symbol, token=token)
        if not self.states:
            raise SystemExit("No valid symbols resolved from watchlist; nothing to scan.")

    def _on_connect(self, ws, response):
        tokens = list(self.states.keys())
        ws.subscribe(tokens)
        ws.set_mode(ws.MODE_FULL, tokens)
        log.info("Subscribed to %d instruments in FULL mode.", len(tokens))

    def _on_close(self, ws, code, reason):
        log.warning("Ticker connection closed: %s %s", code, reason)

    def _on_ticks(self, ws, ticks):
        for tick in ticks:
            state = self.states.get(tick["instrument_token"])
            if state is None:
                continue
            state.on_tick(tick["last_price"], tick.get("depth"))

    def _scan_loop(self):
        while True:
            time.sleep(1)
            for state in self.states.values():
                evaluate(state)

    def run(self):
        threading.Thread(target=self._scan_loop, daemon=True).start()
        self.ticker.connect(threaded=False)


def main():
    api_key = os.environ.get("KITE_API_KEY")
    access_token = os.environ.get("KITE_ACCESS_TOKEN")
    watchlist_raw = os.environ.get("KITE_WATCHLIST", "")

    if not api_key or not access_token:
        log.error("Set KITE_API_KEY and KITE_ACCESS_TOKEN environment variables.")
        sys.exit(1)

    watchlist = [s.strip().upper() for s in watchlist_raw.split(",") if s.strip()]
    if not watchlist:
        log.error("Set KITE_WATCHLIST as a comma-separated list of NSE tradingsymbols.")
        sys.exit(1)

    scanner = Scanner2(api_key, access_token, watchlist)
    log.info("Scanner2 starting for: %s", ", ".join(watchlist))
    scanner.run()


if __name__ == "__main__":
    main()
