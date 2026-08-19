"""
Backtests the VWAP-distance crossing scanner over a historical window, so
candidate entry filters and exit rules can be judged by measured win rate /
expectancy instead of guesswork. Nothing here places orders.

Core idea, day by day, symbol by symbol:
  - threshold for day D = the highest |distance%| seen over the 10 trading
    days strictly BEFORE D (rolling, never leaks D's own data into itself)
  - walk D's 5-min candles, tracking cumulative intraday VWAP
  - a signal fires on each *rising edge*: the first candle where |distance%|
    breaks back out past that threshold after having been inside it
  - a simulated trade enters at the NEXT candle's open (not the signal
    candle's own close -- avoids lookahead bias) and exits per whichever
    exit rule is being tested
"""

import statistics
from collections import defaultdict
from datetime import datetime, time as dtime, timedelta

import config

BASELINE_DAYS = 10
MAX_CALL_SPAN_DAYS = 90   # stay under Kite's ~100-calendar-day cap per historical_data call
ENTRY_CUTOFF = "15:10"     # no new entries this close to the session end
NO_ENTRY_BEFORE = "09:20"  # skip the noisy open


def fetch_chunked_history(kite, token, start_date, end_date):
    """historical_data() has a per-call span cap; stitch multiple calls together."""
    candles = []
    cursor = start_date
    while cursor < end_date:
        chunk_end = min(cursor + timedelta(days=MAX_CALL_SPAN_DAYS), end_date)
        candles.extend(kite.historical_data(token, cursor, chunk_end, config.CANDLE_INTERVAL))
        cursor = chunk_end
    return candles


def _group_by_day(candles, exclude_date=None):
    by_day = defaultdict(list)
    for c in candles:
        d = c["date"].date()
        if exclude_date and d == exclude_date:
            continue
        by_day[d].append(c)
    for d in by_day:
        by_day[d].sort(key=lambda c: c["date"])
    return by_day


def _intraday_series(day_candles):
    """Per-candle (timestamp, close, volume, vwap, dist_pct) for one day, cumulative VWAP reset at open."""
    cum_pv = 0.0
    cum_vol = 0.0
    out = []
    for c in day_candles:
        typical = (c["high"] + c["low"] + c["close"]) / 3.0
        vol = c["volume"] or 0
        cum_pv += typical * vol
        cum_vol += vol
        if cum_vol <= 0:
            continue
        vwap = cum_pv / cum_vol
        if not vwap:
            continue
        dist_pct = (c["close"] - vwap) / vwap * 100.0
        out.append({"candle": c, "vwap": vwap, "dist_pct": dist_pct, "volume": vol})
    return out


def generate_rolling_signals(by_day):
    """
    Walks every trading day (once at least BASELINE_DAYS of history precede it),
    using a rolling prior-10-day threshold, and returns raw signal events with
    enough context (the day's full intraday series, index into it) for both
    filtering and trade simulation.
    """
    sorted_days = sorted(by_day.keys())
    signals = []
    for i, day in enumerate(sorted_days):
        if i < BASELINE_DAYS:
            continue
        prior_days = sorted_days[i - BASELINE_DAYS:i]
        day_highs = []
        for pd in prior_days:
            series = _intraday_series(by_day[pd])
            if series:
                day_highs.append(max(abs(s["dist_pct"]) for s in series))
        if not day_highs:
            continue
        threshold = max(day_highs)

        series = _intraday_series(by_day[day])
        was_crossing = False
        for idx, s in enumerate(series):
            is_crossing = abs(s["dist_pct"]) > threshold
            if is_crossing and not was_crossing:
                signals.append({
                    "day": day,
                    "idx": idx,
                    "series": series,
                    "threshold": threshold,
                    "time": s["candle"]["date"].strftime("%H:%M"),
                    "dist_pct": s["dist_pct"],
                    "direction": "Upside" if s["dist_pct"] >= 0 else "Downside",
                    "volume": s["volume"],
                })
            was_crossing = is_crossing
    return signals


def today_raw_signals(symbol, today_candles, threshold):
    """
    Same rising-edge detection as generate_rolling_signals, but for a single day
    against an already-known threshold (used to refine today's live scan without
    a full historical re-fetch). Signal dicts carry the same shape FILTERS expect.
    """
    series = _intraday_series(today_candles)
    was_crossing = False
    signals = []
    for idx, s in enumerate(series):
        is_crossing = abs(s["dist_pct"]) > threshold
        if is_crossing and not was_crossing:
            signals.append({
                "symbol": symbol,
                "idx": idx,
                "series": series,
                "threshold": threshold,
                "time": s["candle"]["date"].strftime("%H:%M"),
                "dist_pct": s["dist_pct"],
                "direction": "Upside" if s["dist_pct"] >= 0 else "Downside",
                "volume": s["volume"],
            })
        was_crossing = is_crossing
    return signals


# ───────────────────────────── Entry filters ─────────────────────────────

def filter_volume_confirmation(sig):
    """Signal candle's volume is at least 1.5x the day's average candle volume so far."""
    vols = [s["volume"] for s in sig["series"][:sig["idx"] + 1]]
    avg = sum(vols) / len(vols)
    return avg > 0 and sig["volume"] >= 1.5 * avg


def filter_vwap_trend_aligned(sig):
    """VWAP itself must be sloping the same way as the breakout over the last 3 candles."""
    idx = sig["idx"]
    if idx < 3:
        return False
    vwap_now = sig["series"][idx]["vwap"]
    vwap_before = sig["series"][idx - 3]["vwap"]
    slope = vwap_now - vwap_before
    return (slope > 0) if sig["direction"] == "Upside" else (slope < 0)


def filter_time_of_day(sig):
    # Strictly before the cutoff -- the signal candle's own time passing the check
    # isn't enough on its own, since simulated entry happens on the *next* candle.
    return NO_ENTRY_BEFORE <= sig["time"] < ENTRY_CUTOFF


def filter_min_threshold(sig, min_pct=0.5):
    """Skip symbols whose 10-day range is so tight the threshold itself is noise-sized."""
    return sig["threshold"] >= min_pct


FILTERS = {
    "volume": filter_volume_confirmation,
    "trend": filter_vwap_trend_aligned,
    "time_of_day": filter_time_of_day,
    "min_threshold": filter_min_threshold,
}


def passes(sig, filter_names):
    return all(FILTERS[name](sig) for name in filter_names)


# ───────────────────────────── Trade simulation ─────────────────────────────

def simulate_fixed_target_stop(sig, target_pct=1.0, stop_pct=0.5):
    """Long/short from next candle's open; exit on target, stop, or the hard time square-off."""
    series = sig["series"]
    idx = sig["idx"]
    if idx + 1 >= len(series):
        return None  # signal was the last candle of the day, no room to enter

    entry_candle = series[idx + 1]["candle"]
    entry_price = entry_candle["open"]
    is_long = sig["direction"] == "Upside"
    target_price = entry_price * (1 + target_pct / 100.0) if is_long else entry_price * (1 - target_pct / 100.0)
    stop_price = entry_price * (1 - stop_pct / 100.0) if is_long else entry_price * (1 + stop_pct / 100.0)

    for s in series[idx + 1:]:
        c = s["candle"]
        if c["date"].strftime("%H:%M") >= ENTRY_CUTOFF:
            exit_price = c["close"]
            return _trade_result(sig.get("symbol"), sig["day"], sig["time"], sig["direction"], entry_candle["date"], entry_price, c["date"], exit_price, "time_exit")
        hit_stop = (c["low"] <= stop_price) if is_long else (c["high"] >= stop_price)
        hit_target = (c["high"] >= target_price) if is_long else (c["low"] <= target_price)
        if hit_stop:
            # Ambiguous intrabar ordering with only OHLC (no tick data) -- assume the
            # adverse outcome hits first, the conservative assumption for a backtest.
            return _trade_result(sig.get("symbol"), sig["day"], sig["time"], sig["direction"], entry_candle["date"], entry_price, c["date"], stop_price, "stop")
        if hit_target:
            return _trade_result(sig.get("symbol"), sig["day"], sig["time"], sig["direction"], entry_candle["date"], entry_price, c["date"], target_price, "target")

    last = series[-1]["candle"]
    return _trade_result(sig.get("symbol"), sig["day"], sig["time"], sig["direction"], entry_candle["date"], entry_price, last["date"], last["close"], "day_end")


def simulate_trail_vwap(sig):
    """Long/short from next candle's open; exit the first close back through VWAP, or time square-off."""
    series = sig["series"]
    idx = sig["idx"]
    if idx + 1 >= len(series):
        return None

    entry_candle = series[idx + 1]["candle"]
    entry_price = entry_candle["open"]
    is_long = sig["direction"] == "Upside"

    for s in series[idx + 1:]:
        c = s["candle"]
        if c["date"].strftime("%H:%M") >= ENTRY_CUTOFF:
            return _trade_result(sig.get("symbol"), sig["day"], sig["time"], sig["direction"], entry_candle["date"], entry_price, c["date"], c["close"], "time_exit")
        reverted = (c["close"] <= s["vwap"]) if is_long else (c["close"] >= s["vwap"])
        if reverted:
            return _trade_result(sig.get("symbol"), sig["day"], sig["time"], sig["direction"], entry_candle["date"], entry_price, c["date"], c["close"], "vwap_revert")

    last = series[-1]["candle"]
    return _trade_result(sig.get("symbol"), sig["day"], sig["time"], sig["direction"], entry_candle["date"], entry_price, last["date"], last["close"], "day_end")


# ─────────────────── Precise 1-minute entry (finer than the 5-min scan) ───────────────────

def fetch_day_1min(kite, token, day):
    """One trading day's 1-minute candles (Kite's minute-interval cap is 60 calendar
    days per call, so a single day is always well within range)."""
    start = datetime.combine(day, dtime(9, 15))
    end = datetime.combine(day, dtime(15, 30))
    return kite.historical_data(token, start, end, "minute")


def find_precise_entry_1min(day_candles_1m, threshold, direction, start_time="09:15"):
    """
    Re-derives the exact crossing moment using 1-minute candles (same cumulative-VWAP
    methodology as the 5-min scan, just finer-grained), instead of waiting up to ~5
    minutes for the next 5-min bar. Returns (series, idx) for the first candle at or
    after `start_time` where |distance%| crosses `threshold` in the signal's direction,
    or (series, None) if it never does.

    `start_time` should be the corresponding 5-min signal's own bar-start time -- when
    a symbol fires multiple 5-min-level signals in a day, searching from day-start every
    time would collapse them all onto the same (first) crossing.
    """
    series = _intraday_series(day_candles_1m)
    for idx, s in enumerate(series):
        if s["candle"]["date"].strftime("%H:%M") < start_time:
            continue
        dist = s["dist_pct"]
        hit = (dist > threshold) if direction == "Upside" else (dist < -threshold)
        if hit:
            return series, idx
    return series, None


def simulate_trade_1min(series, idx, day, direction, exit_rule="fixed", target_pct=1.0, stop_pct=0.5, symbol=None):
    """
    Enters AT the 1-minute candle where the crossing was detected (its close --
    trading the candle that hits the signal, not the next one), then continues
    monitoring subsequent 1-minute candles -- finer-grained than the 5-min
    scan -- for the target/stop or VWAP-revert exit.
    """
    entry_candle = series[idx]["candle"]
    entry_price = entry_candle["close"]
    signal_time = entry_candle["date"].strftime("%H:%M")
    is_long = direction == "Upside"

    if exit_rule == "fixed":
        target_price = entry_price * (1 + target_pct / 100.0) if is_long else entry_price * (1 - target_pct / 100.0)
        stop_price = entry_price * (1 - stop_pct / 100.0) if is_long else entry_price * (1 + stop_pct / 100.0)

    for s in series[idx + 1:]:
        c = s["candle"]
        if c["date"].strftime("%H:%M") >= ENTRY_CUTOFF:
            return _trade_result(symbol, day, signal_time, direction, entry_candle["date"], entry_price, c["date"], c["close"], "time_exit")
        if exit_rule == "fixed":
            hit_stop = (c["low"] <= stop_price) if is_long else (c["high"] >= stop_price)
            hit_target = (c["high"] >= target_price) if is_long else (c["low"] <= target_price)
            if hit_stop:
                return _trade_result(symbol, day, signal_time, direction, entry_candle["date"], entry_price, c["date"], stop_price, "stop")
            if hit_target:
                return _trade_result(symbol, day, signal_time, direction, entry_candle["date"], entry_price, c["date"], target_price, "target")
        else:  # trail_vwap
            reverted = (c["close"] <= s["vwap"]) if is_long else (c["close"] >= s["vwap"])
            if reverted:
                return _trade_result(symbol, day, signal_time, direction, entry_candle["date"], entry_price, c["date"], c["close"], "vwap_revert")

    last = series[-1]["candle"]
    return _trade_result(symbol, day, signal_time, direction, entry_candle["date"], entry_price, last["date"], last["close"], "day_end")


def _trade_result(symbol, day, signal_time, direction, entry_time, entry_price, exit_time, exit_price, reason):
    is_long = direction == "Upside"
    pnl_pct = (exit_price - entry_price) / entry_price * 100.0
    if not is_long:
        pnl_pct = -pnl_pct
    return {
        "symbol": symbol,
        "day": str(day),
        "signal_time": signal_time,
        "direction": direction,
        "entry_time": entry_time.strftime("%H:%M"),
        "entry_price": round(entry_price, 2),
        "exit_time": exit_time.strftime("%H:%M"),
        "exit_price": round(exit_price, 2),
        "exit_reason": reason,
        "pnl_pct": round(pnl_pct, 3),
        "win": pnl_pct > 0,
    }


def aggregate(trades):
    if not trades:
        return {"count": 0}
    wins = [t for t in trades if t["win"]]
    losses = [t for t in trades if not t["win"]]
    gross_win = sum(t["pnl_pct"] for t in wins)
    gross_loss = -sum(t["pnl_pct"] for t in losses)
    return {
        "count": len(trades),
        "win_rate": round(len(wins) / len(trades) * 100, 1),
        "avg_pnl_pct": round(statistics.mean(t["pnl_pct"] for t in trades), 3),
        "avg_win_pct": round(statistics.mean(t["pnl_pct"] for t in wins), 3) if wins else 0.0,
        "avg_loss_pct": round(statistics.mean(t["pnl_pct"] for t in losses), 3) if losses else 0.0,
        "profit_factor": round(gross_win / gross_loss, 2) if gross_loss > 0 else None,
    }
