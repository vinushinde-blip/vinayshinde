import csv
import statistics
import time
from collections import defaultdict
from datetime import datetime, timedelta

import config


def load_nse500_symbols(csv_path=config.NSE500_CSV):
    symbols = []
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            sym = row.get("Symbol", "").strip()
            if sym:
                symbols.append(sym)
    return symbols


def resolve_instruments(kite, symbols):
    """Map NSE500 tradingsymbols -> instrument_token using Kite's full NSE dump."""
    wanted = set(symbols)
    resolved = {}
    for inst in kite.instruments("NSE"):
        sym = inst["tradingsymbol"]
        if sym in wanted and inst.get("segment") == "NSE" and inst.get("instrument_type") == "EQ":
            resolved[sym] = inst["instrument_token"]
    return resolved


def _day_vwap_series(candles):
    """Given one trading day's 5-min candles (chronological), return per-candle distance% list."""
    cum_pv = 0.0
    cum_vol = 0.0
    dist_series = []
    for c in candles:
        typical = (c["high"] + c["low"] + c["close"]) / 3.0
        vol = c["volume"] or 0
        cum_pv += typical * vol
        cum_vol += vol
        if cum_vol <= 0:
            continue
        vwap = cum_pv / cum_vol
        if vwap:
            dist_series.append((c["close"] - vwap) / vwap * 100.0)
    return dist_series


def bootstrap_symbol_stats(kite, token, days=config.HISTORY_DAYS):
    """
    Pulls the last ~`days` trading days of 5-min candles for one instrument, computes:
      - level1 / level2: Average<->Medium and Medium<->Extreme zone boundaries
        (1x / 2x the historical std-dev of distance%, same logic as the Pine script)
      - ten_day_high: the highest |distance%| reached on any of those days
    """
    to_date = datetime.now()
    from_date = to_date - timedelta(days=int(days * 1.8) + 5)  # buffer for weekends/holidays

    candles = kite.historical_data(token, from_date, to_date, config.CANDLE_INTERVAL)
    by_day = defaultdict(list)
    for c in candles:
        day = c["date"].date()
        by_day[day].append(c)

    recent_days = sorted(by_day.keys())[-days:]

    all_dist = []
    day_highs = []
    for day in recent_days:
        day_dist = _day_vwap_series(by_day[day])
        if day_dist:
            all_dist.extend(day_dist)
            day_highs.append(max(abs(d) for d in day_dist))

    if len(all_dist) < 2:
        return None

    stdev = statistics.stdev(all_dist)
    level1 = stdev * 1.0
    level2 = stdev * 2.0
    ten_day_high = max(day_highs) if day_highs else 0.0
    return {"level1": level1, "level2": level2, "ten_day_high": ten_day_high}


def bootstrap_all(kite, instruments_map, progress_cb=None):
    """Sequentially bootstraps every symbol, respecting Kite's historical-data rate limit."""
    stats = {}
    total = len(instruments_map)
    for i, (symbol, token) in enumerate(instruments_map.items(), start=1):
        try:
            s = bootstrap_symbol_stats(kite, token)
            if s:
                stats[symbol] = s
        except Exception as e:
            if progress_cb:
                progress_cb(f"  [{i}/{total}] {symbol}: failed ({e})")
        if progress_cb and i % 25 == 0:
            progress_cb(f"  bootstrapped {i}/{total} symbols...")
        time.sleep(config.HISTORICAL_CALL_DELAY)
    return stats


def classify_zone(abs_dist, level1, level2):
    if abs_dist <= level1:
        return 1, "Average"
    if abs_dist <= level2:
        return 2, "Medium"
    return 3, "Extreme"


def fetch_live_quotes(kite, instruments_map):
    """Batched kite.quote() calls; returns {symbol: {'last_price', 'average_price'}}."""
    symbols = list(instruments_map.keys())
    out = {}
    for i in range(0, len(symbols), config.QUOTE_BATCH_SIZE):
        batch = symbols[i:i + config.QUOTE_BATCH_SIZE]
        keys = [f"NSE:{s}" for s in batch]
        quotes = kite.quote(keys)
        for sym in batch:
            q = quotes.get(f"NSE:{sym}")
            if not q or not q.get("average_price"):
                continue
            out[sym] = {
                "last_price": q["last_price"],
                "average_price": q["average_price"],  # Kite's own day VWAP
            }
    return out


def build_signals(kite, instruments_map, stats):
    """Combines live quotes + bootstrapped stats into the rows shown on the page."""
    quotes = fetch_live_quotes(kite, instruments_map)
    rows = []
    for symbol, s in stats.items():
        q = quotes.get(symbol)
        if not q:
            continue
        vwap = q["average_price"]
        ltp = q["last_price"]
        dist_pct = (ltp - vwap) / vwap * 100.0
        abs_dist = abs(dist_pct)
        zone_code, zone_label = classify_zone(abs_dist, s["level1"], s["level2"])
        crossing = abs_dist > s["ten_day_high"]
        rows.append({
            "symbol": symbol,
            "ltp": ltp,
            "vwap": vwap,
            "dist_pct": dist_pct,
            "zone_code": zone_code,
            "zone_label": zone_label,
            "ten_day_high": s["ten_day_high"],
            "crossing": crossing,
        })
    # Crossing signals first, then by how extreme the distance is.
    rows.sort(key=lambda r: (not r["crossing"], -abs(r["dist_pct"])))
    return rows
