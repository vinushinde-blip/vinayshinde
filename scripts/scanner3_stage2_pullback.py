"""
Scanner3 - Stage-2 uptrend contraction screener, run against the daily
OHLC files produced by fetch_nse500_multiframe_ohlc.py.

Detects five contraction-pattern archetypes, all gated by the same
underlying idea: a stock in a real uptrend, contracting in an orderly
way with volume drying up, rather than falling apart.

  1. single_pullback  - one clean 12-20% pullback off the high (the
     original 8-condition Stage-2 pullback-to-EMA setup).
  2. sequential_vcp   - 2-6 progressive pullback legs, each shallower
     and lower-volume than the last (classic VCP).
  3. flat_base        - sideways, narrow-range (8-16%) base for weeks
     with volume drying up.
  4. cup_handle       - a rounded (U-shaped) decline/recovery, followed
     by a shallow, low-volume handle near the highs.
  5. falling_wedge    - a coiling range: swing highs and lows both
     converging, volume shrinking as the apex approaches.

All five require the stage-2 trend template as a prerequisite (a real
uptrend has to exist before you can call anything a "contraction" of
it), and all report a volume dry-up flag and a breakout flag using the
same 20-day-vs-50-day volume comparison the "Key Stages" framework uses.

Interpretation notes (these are discretionary chart patterns, being
approximated computationally - concrete rules used, all tunable below):
  - Stage-2 trend template (simplified Minervini): EMA50 > SMA150 >
    SMA200 with SMA200 rising over the last month (the MAs stay stacked
    in uptrend order even while price dips into a base), close no more
    than 5% below EMA50, within 25% of the 52-week high, and >=30% above
    the 52-week low.
  - Swing pivots: a 5-bar fractal (a bar whose high/low is the extreme
    within +/-5 bars).
  - "Orderly" = no single close-to-close decline worse than -7%, and
    ATR(14) has contracted vs. the pattern's starting point.
  - "Mother candle" = the widest-range (high-low) candle in the last 10
    days of the base - entry is a stop order above its high.
  - Stoploss = the previous day's low.
  - Pivot, R1/R2, S1/S2 = classic floor-trader pivot points computed from
    the previous day's high/low/close (pivot = avg(H,L,C); R1/S1 =
    2*pivot -/+ low/high; R2/S2 = pivot +/- (high-low)).
  - already_triggered = close has already crossed above entry_trigger
    (price alone, no volume check) - a heads-up that the setup may be
    past the ideal entry, not a buy signal itself. breakout_now is the
    stricter version: crossed AND today's volume expanded 1.5x+ over the
    50-day average, the actual confirmation this framework calls for.

Run
---
  python scripts/scanner3_stage2_pullback.py

Reads data/ohlc/day/<SYMBOL>.csv (from the bulk downloader) - works with
whatever's been downloaded so far, coverage grows as that job progresses.
Writes data/scanner3_candidates.csv and prints a ranked table.
"""

import glob
import os

import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "ohlc", "day")
OUT_FILE = os.path.join(os.path.dirname(__file__), "..", "data", "scanner3_candidates.csv")

# --- shared trend / quality gates ---
TREND_TEMPLATE_HIGH_PROXIMITY = 0.25   # close must be within 25% of 52w high
TREND_TEMPLATE_LOW_MARGIN = 0.30       # close must be >=30% above 52w low
SMA200_RISING_LOOKBACK = 25
STAGE2_EMA50_TOLERANCE = 0.05          # close can be up to 5% below EMA50 (common mid-base)
MAX_SINGLE_DAY_DROP_PCT = -7.0
EMA20_UNDERCUT_TOLERANCE = 0.05
VOLUME_DRYUP_RATIO = 0.85              # vol20 must be below this x vol50
BREAKOUT_VOLUME_EXPANSION = 1.5        # today's volume vs vol50 to call a breakout

# --- single_pullback ---
SWING_LOOKBACK = 90
MIN_PULLBACK_DAYS = 5
MAX_PULLBACK_DAYS = 40
PULLBACK_MIN_PCT = 12.0
PULLBACK_MAX_PCT = 20.0

# --- pivots / sequential_vcp ---
PIVOT_WINDOW = 5
VCP_LOOKBACK_DAYS = 130
VCP_MIN_LEGS = 2
VCP_MAX_LEGS = 6
VCP_LEG_SHRINK_RATIO = 0.75            # each leg's depth must be <= this x previous leg

# --- flat_base ---
FLAT_BASE_WINDOW = 25
FLAT_BASE_MIN_PCT = 8.0
FLAT_BASE_MAX_PCT = 16.0

# --- cup_handle ---
CUP_WINDOW = 110
CUP_MIN_WINDOW = 60
HANDLE_WINDOW = 12
HANDLE_MAX_PCT = 15.0
CUP_RIM_TOLERANCE = 0.15               # start/end of cup must be within this of each other

# --- falling_wedge ---
WEDGE_MIN_PIVOT_PAIRS = 3
WEDGE_RANGE_SHRINK_RATIO = 0.6         # latest pivot-pair range vs earliest


def load_symbol(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)
    df["ema10"] = df["close"].ewm(span=10, adjust=False).mean()
    df["ema20"] = df["close"].ewm(span=20, adjust=False).mean()
    df["ema50"] = df["close"].ewm(span=50, adjust=False).mean()
    df["sma150"] = df["close"].rolling(150).mean()
    df["sma200"] = df["close"].rolling(200).mean()
    df["vol20"] = df["volume"].rolling(20).mean()
    df["vol50"] = df["volume"].rolling(50).mean()
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - df["close"].shift()).abs(),
        (df["low"] - df["close"].shift()).abs(),
    ], axis=1).max(axis=1)
    df["atr14"] = tr.rolling(14).mean()
    return df


def is_stage2(row, df, i) -> bool:
    if pd.isna(row["sma200"]) or i < SMA200_RISING_LOOKBACK:
        return False
    high_52w = df["high"].iloc[max(0, i - 252):i + 1].max()
    low_52w = df["low"].iloc[max(0, i - 252):i + 1].min()
    sma200_prior = df["sma200"].iloc[i - SMA200_RISING_LOOKBACK]

    return (
        row["close"] >= row["ema50"] * (1 - STAGE2_EMA50_TOLERANCE)
        and row["ema50"] > row["sma150"] > row["sma200"]
        and row["sma200"] > sma200_prior
        and row["close"] >= high_52w * (1 - TREND_TEMPLATE_HIGH_PROXIMITY)
        and row["close"] >= low_52w * (1 + TREND_TEMPLATE_LOW_MARGIN)
    )


def is_orderly(df: pd.DataFrame, start_idx: int, end_idx: int) -> bool:
    window = df.iloc[start_idx:end_idx + 1]
    daily_ret = window["close"].pct_change() * 100
    return daily_ret.min() is None or daily_ret.min() >= MAX_SINGLE_DAY_DROP_PCT


def find_pivots(df: pd.DataFrame, start_idx: int, end_idx: int, window: int = PIVOT_WINDOW):
    """5-bar fractal pivots: (index, 'high'/'low', price) in chronological order."""
    pivots = []
    for j in range(max(start_idx, window), min(end_idx, len(df) - 1 - window) + 1):
        seg_high = df["high"].iloc[j - window:j + window + 1]
        seg_low = df["low"].iloc[j - window:j + window + 1]
        if df["high"].iloc[j] == seg_high.max():
            pivots.append((j, "high", df["high"].iloc[j]))
        elif df["low"].iloc[j] == seg_low.min():
            pivots.append((j, "low", df["low"].iloc[j]))
    return pivots


def entry_and_stop(df: pd.DataFrame, base_start_idx: int, i: int):
    recent_base = df.iloc[max(base_start_idx, i - 10):i + 1]
    mother_candle = recent_base.loc[(recent_base["high"] - recent_base["low"]).idxmax()]
    entry_trigger = mother_candle["high"]
    stoploss = df.iloc[i - 1]["low"] if i > 0 else df.iloc[i]["low"]
    return entry_trigger, stoploss


def volume_dryup(row) -> bool:
    if pd.isna(row["vol20"]) or pd.isna(row["vol50"]) or row["vol50"] == 0:
        return False
    return row["vol20"] < row["vol50"] * VOLUME_DRYUP_RATIO


def is_breakout(df: pd.DataFrame, i: int, resistance: float) -> bool:
    row = df.iloc[i]
    if pd.isna(row["vol50"]) or row["vol50"] == 0:
        return False
    return row["close"] > resistance and row["volume"] > row["vol50"] * BREAKOUT_VOLUME_EXPANSION


def pivot_levels(df: pd.DataFrame, i: int):
    """Classic floor-trader pivots from the previous day's H/L/C."""
    if i == 0:
        return None
    prev = df.iloc[i - 1]
    high, low, close = prev["high"], prev["low"], prev["close"]
    pivot = (high + low + close) / 3
    r1, s1 = 2 * pivot - low, 2 * pivot - high
    r2, s2 = pivot + (high - low), pivot - (high - low)
    return {"pivot": round(pivot, 2), "r1": round(r1, 2), "r2": round(r2, 2), "s1": round(s1, 2), "s2": round(s2, 2)}


# ---- pattern detectors: each returns a dict of pattern-specific fields, or None ----

def detect_single_pullback(df, i):
    window = df.iloc[max(0, i - SWING_LOOKBACK):i + 1]
    swing_idx = window["high"].idxmax()
    swing_high = df.loc[swing_idx, "high"]
    pullback_days = i - swing_idx
    if not (MIN_PULLBACK_DAYS <= pullback_days <= MAX_PULLBACK_DAYS):
        return None

    close = df.iloc[i]["close"]
    pullback_pct = (swing_high - close) / swing_high * 100
    if not (PULLBACK_MIN_PCT <= pullback_pct <= PULLBACK_MAX_PCT):
        return None
    if not is_orderly(df, swing_idx, i):
        return None

    atr_at_swing, atr_now = df.loc[swing_idx, "atr14"], df.iloc[i]["atr14"]
    if pd.isna(atr_at_swing) or pd.isna(atr_now) or atr_now >= atr_at_swing:
        return None

    runup = df.iloc[max(0, swing_idx - pullback_days):swing_idx]
    if runup.empty or runup["volume"].mean() == 0:
        return None
    if df.iloc[swing_idx:i + 1]["volume"].mean() >= runup["volume"].mean() * VOLUME_DRYUP_RATIO:
        return None

    ema20_now = df.iloc[i]["ema20"]
    if df.iloc[swing_idx:i + 1]["low"].min() < ema20_now * (1 - EMA20_UNDERCUT_TOLERANCE):
        return None
    if close < ema20_now * (1 - EMA20_UNDERCUT_TOLERANCE):
        return None

    return {"base_start": swing_idx, "pullback_pct": round(pullback_pct, 1), "pullback_days": pullback_days}


def detect_sequential_vcp(df, i):
    start = max(0, i - VCP_LOOKBACK_DAYS)
    pivots = find_pivots(df, start, i)
    if len(pivots) < 4:
        return None

    legs = []
    for a, b in zip(pivots, pivots[1:]):
        if a[1] == "high" and b[1] == "low":
            depth_pct = (a[2] - b[2]) / a[2] * 100
            if depth_pct > 1:
                vol = df.iloc[a[0]:b[0] + 1]["volume"].mean()
                legs.append({"depth_pct": depth_pct, "avg_vol": vol})
    if not (VCP_MIN_LEGS <= len(legs) <= VCP_MAX_LEGS):
        return None

    for prev, curr in zip(legs, legs[1:]):
        if curr["depth_pct"] > prev["depth_pct"] * VCP_LEG_SHRINK_RATIO:
            return None
        if curr["avg_vol"] > prev["avg_vol"]:
            return None

    base_start = next(p[0] for p in pivots if p[1] == "high")
    return {"base_start": base_start, "num_legs": len(legs), "last_leg_pct": round(legs[-1]["depth_pct"], 1)}


def detect_flat_base(df, i):
    window = df.iloc[max(0, i - FLAT_BASE_WINDOW):i + 1]
    lo = window["low"].min()
    hi = window["high"].max()
    if lo == 0:
        return None
    range_pct = (hi - lo) / lo * 100
    if not (FLAT_BASE_MIN_PCT <= range_pct <= FLAT_BASE_MAX_PCT):
        return None
    if not volume_dryup(df.iloc[i]):
        return None
    if not is_orderly(df, window.index[0], i):
        return None
    return {"base_start": window.index[0], "range_pct": round(range_pct, 1)}


def detect_cup_handle(df, i):
    for cup_len in range(CUP_WINDOW, CUP_MIN_WINDOW - 1, -10):
        start = i - cup_len - HANDLE_WINDOW
        if start < 0:
            continue
        cup = df.iloc[start:i - HANDLE_WINDOW + 1]
        if cup.empty:
            continue
        min_idx_local = cup["close"].idxmin()
        pos = (min_idx_local - cup.index[0]) / max(1, len(cup) - 1)
        if not (0.2 <= pos <= 0.8):
            continue

        rim_left = cup["close"].iloc[0]
        rim_right = cup["close"].iloc[-1]
        if rim_left == 0 or abs(rim_right - rim_left) / rim_left > CUP_RIM_TOLERANCE:
            continue

        handle = df.iloc[i - HANDLE_WINDOW:i + 1]
        handle_high = handle["high"].max()
        handle_low = handle["low"].min()
        if handle_high == 0:
            continue
        handle_pct = (handle_high - handle_low) / handle_high * 100
        if handle_pct > HANDLE_MAX_PCT:
            continue
        if handle["volume"].mean() >= cup["volume"].mean() * VOLUME_DRYUP_RATIO:
            continue
        close = df.iloc[i]["close"]
        if close < rim_right * (1 - CUP_RIM_TOLERANCE):
            continue

        return {"base_start": start, "cup_days": cup_len, "handle_pct": round(handle_pct, 1)}
    return None


def detect_falling_wedge(df, i):
    start = max(0, i - VCP_LOOKBACK_DAYS)
    pivots = find_pivots(df, start, i)
    highs = [p for p in pivots if p[1] == "high"]
    lows = [p for p in pivots if p[1] == "low"]
    if len(highs) < WEDGE_MIN_PIVOT_PAIRS or len(lows) < WEDGE_MIN_PIVOT_PAIRS:
        return None

    pairs = []
    for h in highs:
        later_lows = [l for l in lows if l[0] > h[0]]
        if later_lows:
            pairs.append((h, later_lows[0]))
    if len(pairs) < WEDGE_MIN_PIVOT_PAIRS:
        return None

    ranges = [(h[2] - l[2]) for h, l in pairs]
    if ranges[0] <= 0:
        return None
    if ranges[-1] > ranges[0] * WEDGE_RANGE_SHRINK_RATIO:
        return None
    if not all(x >= y * 0.9 for x, y in zip(ranges, ranges[1:])):  # roughly monotonic shrink
        return None

    base_start = pairs[0][0][0]
    return {"base_start": base_start, "pivot_pairs": len(pairs), "range_shrink_pct": round((1 - ranges[-1] / ranges[0]) * 100, 1)}


DETECTORS = {
    "single_pullback": detect_single_pullback,
    "sequential_vcp": detect_sequential_vcp,
    "flat_base": detect_flat_base,
    "cup_handle": detect_cup_handle,
    "falling_wedge": detect_falling_wedge,
}


def evaluate_symbol(symbol: str, df: pd.DataFrame):
    if len(df) < 220:
        return None
    i = len(df) - 1
    row = df.iloc[i]

    if not is_stage2(row, df, i):
        return None

    matched = {}
    earliest_base_start = i
    for name, fn in DETECTORS.items():
        try:
            result = fn(df, i)
        except Exception:
            result = None
        if result:
            matched[name] = result
            earliest_base_start = min(earliest_base_start, result["base_start"])

    if not matched:
        return None

    entry_trigger, stoploss = entry_and_stop(df, earliest_base_start, i)
    resistance = df.iloc[earliest_base_start:i]["high"].max()
    pivots = pivot_levels(df, i) or {}

    out = {
        "symbol": symbol,
        "pattern_types": ", ".join(matched.keys()),
        "close": round(row["close"], 2),
        "volume_dryup": volume_dryup(row),
        "already_triggered": row["close"] > entry_trigger,
        "breakout_now": is_breakout(df, i, resistance),
        "entry_trigger": round(entry_trigger, 2),
        "stoploss": round(stoploss, 2),
        "distance_to_trigger_pct": round((entry_trigger - row["close"]) / row["close"] * 100, 1),
        "pivot": pivots.get("pivot"),
        "r1": pivots.get("r1"),
        "r2": pivots.get("r2"),
        "s1": pivots.get("s1"),
        "s2": pivots.get("s2"),
    }
    for name, result in matched.items():
        for k, v in result.items():
            if k != "base_start":
                out[f"{name}.{k}"] = v
    return out


def main():
    files = sorted(glob.glob(os.path.join(DATA_DIR, "*.csv")))
    if not files:
        print(f"No daily OHLC files found in {DATA_DIR} yet.")
        return

    results = []
    for path in files:
        symbol = os.path.splitext(os.path.basename(path))[0]
        try:
            df = load_symbol(path)
            match = evaluate_symbol(symbol, df)
            if match:
                results.append(match)
        except Exception as e:
            print(f"Skipping {symbol}: {e}")

    print(f"Scanned {len(files)} symbols, {len(results)} match at least one contraction pattern.\n")

    if results:
        out = pd.DataFrame(results).sort_values("distance_to_trigger_pct")
        out.to_csv(OUT_FILE, index=False)
        with pd.option_context("display.max_rows", None, "display.width", 200):
            print(out.to_string(index=False))
        print(f"\nSaved to {OUT_FILE}")


if __name__ == "__main__":
    main()
