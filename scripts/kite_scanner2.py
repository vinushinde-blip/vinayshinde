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
  export KITE_WATCHLIST=NSE500      # default; or a comma list e.g. RELIANCE,TCS,INFY

Run
---
  python scripts/kite_scanner2.py

Then open http://localhost:8765 (or $KITE_SCANNER_PORT) — the page polls
/api/alerts and redraws itself every 30 seconds with whatever WATCH /
BUY TRIGGER alerts have fired so far. This has to be served locally by
this process (rather than as a hosted static page) because it needs a
live, authenticated Kite tick stream behind it.

Notes
-----
- Requires tick subscription in FULL mode (KiteTicker.MODE_FULL) since
  market depth is only present in full-mode ticks.
- NSE500 constituents are fetched live from NSE's archive CSV, falling
  back to data/nse500_constituents.csv (a point-in-time snapshot) if that
  fetch fails. Index membership drifts over time, so refresh that file
  periodically if you rely on the fallback.
- All thresholds below are starting points, not calibrated constants —
  tune IMBALANCE_THRESHOLD / IMBALANCE_RISE_MARGIN / CONTRACTION_RATIO /
  WINDOW to taste once you see how it behaves against real tick flow.
"""

import csv
import io
import json
import logging
import os
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import requests
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

NSE500_URL = "https://archives.nseindia.com/content/indices/ind_nifty500list.csv"
NSE500_FALLBACK_FILE = os.path.join(os.path.dirname(__file__), "..", "data", "nse500_constituents.csv")

DASHBOARD_PORT = int(os.environ.get("KITE_SCANNER_PORT", "8765"))
DASHBOARD_REFRESH_SECONDS = 30

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
    _discarded_partial_first_bar: bool = False
    last_alert_minute: datetime = None
    last_trigger_minute: datetime = None

    def on_tick(self, ltp: float, depth: dict) -> None:
        now_minute = datetime.now().replace(second=0, microsecond=0)
        imbalance = _depth_imbalance(depth)

        if self._current is None or self._current.minute != now_minute:
            if self._current is not None:
                # The very first bar built after process start can span less
                # than 60s (we start mid-minute), so its avg_imbalance is
                # noisy from too few ticks - drop it rather than let it seed
                # a spurious flip/contraction read on the first evaluation.
                if self._discarded_partial_first_bar:
                    self.bars.append(self._current)
                else:
                    self._discarded_partial_first_bar = True
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


class AlertStore:
    """Thread-safe recent-alerts buffer the dashboard HTTP handler reads from."""

    def __init__(self, max_items: int = 200):
        self._lock = threading.Lock()
        self.triggers = deque(maxlen=max_items)
        self.watches = deque(maxlen=max_items)
        self.started_at = datetime.now()
        self.tracked_count = 0
        self.last_tick_at = None

    def add_trigger(self, entry: dict) -> None:
        with self._lock:
            self.triggers.appendleft(entry)

    def add_watch(self, entry: dict) -> None:
        with self._lock:
            self.watches.appendleft(entry)

    def note_tick(self) -> None:
        self.last_tick_at = datetime.now()

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "generated_at": datetime.now().isoformat(timespec="seconds"),
                "started_at": self.started_at.isoformat(timespec="seconds"),
                "tracked_count": self.tracked_count,
                "last_tick_at": self.last_tick_at.isoformat(timespec="seconds") if self.last_tick_at else None,
                "triggers": list(self.triggers)[:50],
                "watches": list(self.watches)[:50],
            }


def evaluate(state: InstrumentState, store: AlertStore) -> None:
    bars = state.closed_bars()
    if len(bars) < WINDOW:
        return

    window = bars[-WINDOW:]
    latest_minute = window[-1].minute

    if _flips_positive(window) and state.last_trigger_minute != latest_minute:
        state.last_trigger_minute = latest_minute
        msg = (
            f"imbalance flipped positive at close={window[-1].close:.2f} "
            f"(window {window[0].minute.strftime('%H:%M')} -> {latest_minute.strftime('%H:%M')})"
        )
        log.info("\033[92m[BUY TRIGGER] %-12s %s\033[0m", state.symbol, msg)
        store.add_trigger({
            "symbol": state.symbol,
            "time": latest_minute.strftime("%H:%M"),
            "close": round(window[-1].close, 2),
            "message": msg,
        })
        return

    if (
        _is_contracting(window)
        and _is_rising_negative_imbalance(window)
        and _is_price_resilient(window)
        and state.last_alert_minute != latest_minute
    ):
        state.last_alert_minute = latest_minute
        msg = (
            f"range contracting, imbalance {window[0].avg_imbalance:.2f} -> {window[-1].avg_imbalance:.2f} "
            f"(selling), price holding"
        )
        log.info(
            "[WATCH] %-12s %s (closes %s)",
            state.symbol, msg, [round(b.close, 2) for b in window],
        )
        store.add_watch({
            "symbol": state.symbol,
            "time": latest_minute.strftime("%H:%M"),
            "closes": [round(b.close, 2) for b in window],
            "imbalance_from": round(window[0].avg_imbalance, 2),
            "imbalance_to": round(window[-1].avg_imbalance, 2),
            "message": msg,
        })


def fetch_nse500_symbols() -> list:
    """NSE-listed tradingsymbols for the current Nifty 500 constituents."""
    try:
        resp = requests.get(NSE500_URL, headers={"User-Agent": "Mozilla/5.0"}, timeout=20)
        resp.raise_for_status()
        reader = csv.DictReader(io.StringIO(resp.text))
        symbols = [row["Symbol"].strip() for row in reader if row.get("Symbol")]
        if symbols:
            log.info("Fetched %d NSE500 constituents live from NSE archives.", len(symbols))
            return symbols
    except Exception as e:
        log.warning("Live NSE500 fetch failed (%s); falling back to bundled snapshot.", e)

    with open(NSE500_FALLBACK_FILE, newline="") as f:
        reader = csv.DictReader(f)
        symbols = [row["Symbol"].strip() for row in reader if row.get("Symbol")]
    log.info("Loaded %d NSE500 constituents from bundled snapshot.", len(symbols))
    return symbols


DASHBOARD_HTML = """<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Scanner2 — NSE500 Contraction / Absorption</title>
<style>
  body { font-family: -apple-system, Segoe UI, Roboto, sans-serif; background: #0b0f14; color: #e6edf3; margin: 0; padding: 24px; }
  h1 { font-size: 18px; font-weight: 600; margin: 0 0 4px; }
  .meta { color: #8b949e; font-size: 13px; margin-bottom: 20px; }
  .section { margin-bottom: 28px; }
  .section h2 { font-size: 14px; text-transform: uppercase; letter-spacing: .04em; color: #8b949e; margin: 0 0 8px; }
  table { width: 100%; border-collapse: collapse; font-size: 13px; }
  th, td { text-align: left; padding: 6px 10px; border-bottom: 1px solid #21262d; }
  th { color: #8b949e; font-weight: 500; }
  tr.trigger td { color: #3fb950; }
  tr.watch td { color: #e6edf3; }
  .symbol { font-weight: 600; }
  .empty { color: #6e7681; font-style: italic; padding: 8px 10px; }
</style>
</head>
<body>
<h1>Scanner2 — NSE500 Contraction / Sell-Absorption</h1>
<div class="meta" id="meta">loading…</div>

<div class="section">
  <h2>Buy Triggers (imbalance flipped positive)</h2>
  <table id="triggers"><thead><tr><th>Time</th><th>Symbol</th><th>Close</th><th>Detail</th></tr></thead>
  <tbody></tbody></table>
</div>

<div class="section">
  <h2>Watch (contracting + rising sell imbalance + price holding)</h2>
  <table id="watches"><thead><tr><th>Time</th><th>Symbol</th><th>Closes</th><th>Imbalance</th></tr></thead>
  <tbody></tbody></table>
</div>

<script>
async function refresh() {
  try {
    const res = await fetch('/api/alerts', { cache: 'no-store' });
    const data = await res.json();

    document.getElementById('meta').textContent =
      `Tracking ${data.tracked_count} symbols · started ${data.started_at.replace('T', ' ')} · ` +
      `last tick ${data.last_tick_at ? data.last_tick_at.replace('T', ' ') : 'none yet'} · ` +
      `updated ${data.generated_at.replace('T', ' ')}`;

    const tBody = document.querySelector('#triggers tbody');
    tBody.innerHTML = data.triggers.length ? data.triggers.map(t =>
      `<tr class="trigger"><td>${t.time}</td><td class="symbol">${t.symbol}</td><td>${t.close}</td><td>${t.message}</td></tr>`
    ).join('') : '<tr><td class="empty" colspan="4">No triggers yet</td></tr>';

    const wBody = document.querySelector('#watches tbody');
    wBody.innerHTML = data.watches.length ? data.watches.map(w =>
      `<tr class="watch"><td>${w.time}</td><td class="symbol">${w.symbol}</td><td>${w.closes.join(' → ')}</td><td>${w.imbalance_from} → ${w.imbalance_to}</td></tr>`
    ).join('') : '<tr><td class="empty" colspan="4">No watch alerts yet</td></tr>';
  } catch (e) {
    document.getElementById('meta').textContent = 'Error loading alerts: ' + e;
  }
}
refresh();
setInterval(refresh, __REFRESH_MS__);
</script>
</body>
</html>
""".replace("__REFRESH_MS__", str(DASHBOARD_REFRESH_SECONDS * 1000))


def make_dashboard_handler(store: AlertStore):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            pass  # quiet; alerts are already logged separately

        def do_GET(self):
            if self.path == "/" or self.path == "/index.html":
                body = DASHBOARD_HTML.encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif self.path == "/api/alerts":
                body = json.dumps(store.snapshot()).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            else:
                self.send_response(404)
                self.end_headers()

    return Handler


class Scanner2:
    def __init__(self, api_key: str, access_token: str, watchlist: list):
        self.kite = KiteConnect(api_key=api_key)
        self.kite.set_access_token(access_token)
        self.ticker = KiteTicker(api_key, access_token)
        self.states: dict[int, InstrumentState] = {}
        self.store = AlertStore()
        self._resolve_tokens(watchlist)
        self.store.tracked_count = len(self.states)

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
        # Kite allows up to 3000 instruments per connection; well within that here.
        for i in range(0, len(tokens), 500):
            batch = tokens[i:i + 500]
            ws.subscribe(batch)
            ws.set_mode(ws.MODE_FULL, batch)
        log.info("Subscribed to %d instruments in FULL mode.", len(tokens))

    def _on_close(self, ws, code, reason):
        log.warning("Ticker connection closed: %s %s", code, reason)

    def _on_ticks(self, ws, ticks):
        self.store.note_tick()
        for tick in ticks:
            state = self.states.get(tick["instrument_token"])
            if state is None:
                continue
            state.on_tick(tick["last_price"], tick.get("depth"))

    def _scan_loop(self):
        while True:
            time.sleep(1)
            for state in self.states.values():
                evaluate(state, self.store)

    def _serve_dashboard(self):
        server = ThreadingHTTPServer(("0.0.0.0", DASHBOARD_PORT), make_dashboard_handler(self.store))
        log.info("Dashboard live at http://localhost:%d (refreshes every %ds)", DASHBOARD_PORT, DASHBOARD_REFRESH_SECONDS)
        server.serve_forever()

    def run(self):
        threading.Thread(target=self._scan_loop, daemon=True).start()
        threading.Thread(target=self._serve_dashboard, daemon=True).start()
        self.ticker.connect(threaded=False)


def main():
    api_key = os.environ.get("KITE_API_KEY")
    access_token = os.environ.get("KITE_ACCESS_TOKEN")
    watchlist_raw = os.environ.get("KITE_WATCHLIST", "NSE500").strip()

    if not api_key or not access_token:
        log.error("Set KITE_API_KEY and KITE_ACCESS_TOKEN environment variables.")
        sys.exit(1)

    if watchlist_raw.upper() == "NSE500":
        watchlist = fetch_nse500_symbols()
    else:
        watchlist = [s.strip().upper() for s in watchlist_raw.split(",") if s.strip()]

    if not watchlist:
        log.error("No symbols resolved; set KITE_WATCHLIST to NSE500 or a comma-separated symbol list.")
        sys.exit(1)

    scanner = Scanner2(api_key, access_token, watchlist)
    log.info("Scanner2 starting for %d symbols.", len(watchlist))
    scanner.run()


if __name__ == "__main__":
    main()
