"""
Scanner3 - Stage-2 uptrend pullback screener, run against the daily OHLC
files produced by fetch_nse500_multiframe_ohlc.py.

Implements the 8-condition setup:
  1. Strong prior uptrend required.
  2. Stock should be in a stage-2 uptrend.
  3. Pullback should be orderly (no crash-days) - a real, "hard" pullback,
     not a shallow wiggle.
  4. Pullback should be 12-20% off the high.
  5. Strong stocks respect the 10/20 EMA - price should be basing near it.
  6. Volume should contract during the pullback (selling being absorbed).
  7. Entry: above the "mother candle" high (the widest-range candle in the
     recent base).
  8. Stoploss: the previous day's low.

Interpretation notes (this is a discretionary charting pattern, being
approximated computationally - these are the concrete rules used):
  - Stage-2 trend template (simplified Minervini): close > EMA50 > SMA150
    > SMA200, SMA200 rising over the last month, close within 25% of the
    52-week high and >=30% above the 52-week low.
  - The pullback window is measured from the most recent swing high
    (highest High in the last 90 trading days) to today.
  - "Orderly" = no single close-to-close decline in the pullback window
    worse than -7%, and ATR(14) has contracted vs. the swing-high point.
  - "Respects 10/20 EMA" = the pullback's low didn't undercut EMA20 by
    more than 5%, and price is still within 5% of EMA20 now.
  - "Mother candle" = the widest-range (high-low) candle in the last 10
    days of the base - entry is a stop order above its high.

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

SWING_LOOKBACK = 90          # trading days to look back for the pre-pullback high
MIN_PULLBACK_DAYS = 5
MAX_PULLBACK_DAYS = 40
PULLBACK_MIN_PCT = 12.0
PULLBACK_MAX_PCT = 20.0
MAX_SINGLE_DAY_DROP_PCT = -7.0
EMA20_UNDERCUT_TOLERANCE = 0.05
TREND_TEMPLATE_HIGH_PROXIMITY = 0.25   # close must be within 25% of 52w high
TREND_TEMPLATE_LOW_MARGIN = 0.30       # close must be >=30% above 52w low
SMA200_RISING_LOOKBACK = 25


def load_symbol(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)
    df["ema10"] = df["close"].ewm(span=10, adjust=False).mean()
    df["ema20"] = df["close"].ewm(span=20, adjust=False).mean()
    df["ema50"] = df["close"].ewm(span=50, adjust=False).mean()
    df["sma150"] = df["close"].rolling(150).mean()
    df["sma200"] = df["close"].rolling(200).mean()
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
        row["close"] > row["ema50"] > row["sma150"] > row["sma200"]
        and row["sma200"] > sma200_prior
        and row["close"] >= high_52w * (1 - TREND_TEMPLATE_HIGH_PROXIMITY)
        and row["close"] >= low_52w * (1 + TREND_TEMPLATE_LOW_MARGIN)
    )


def find_pullback(df: pd.DataFrame, i: int):
    window = df.iloc[max(0, i - SWING_LOOKBACK):i + 1]
    swing_idx_local = window["high"].idxmax()
    swing_high = df.loc[swing_idx_local, "high"]
    swing_date = df.loc[swing_idx_local, "date"]
    pullback_days = i - swing_idx_local
    if not (MIN_PULLBACK_DAYS <= pullback_days <= MAX_PULLBACK_DAYS):
        return None
    return swing_idx_local, swing_high, swing_date, pullback_days


def evaluate_symbol(symbol: str, df: pd.DataFrame):
    if len(df) < 220:
        return None
    i = len(df) - 1
    row = df.iloc[i]

    if not is_stage2(row, df, i):
        return None

    pullback = find_pullback(df, i)
    if pullback is None:
        return None
    swing_idx, swing_high, swing_date, pullback_days = pullback

    close = row["close"]
    pullback_pct = (swing_high - close) / swing_high * 100
    if not (PULLBACK_MIN_PCT <= pullback_pct <= PULLBACK_MAX_PCT):
        return None

    pullback_window = df.iloc[swing_idx:i + 1]
    daily_ret = pullback_window["close"].pct_change() * 100
    if daily_ret.min() is not None and daily_ret.min() < MAX_SINGLE_DAY_DROP_PCT:
        return None

    atr_at_swing = df.loc[swing_idx, "atr14"]
    atr_now = row["atr14"]
    if pd.isna(atr_at_swing) or pd.isna(atr_now) or atr_now >= atr_at_swing:
        return None

    runup_window = df.iloc[max(0, swing_idx - pullback_days):swing_idx]
    if runup_window.empty:
        return None
    avg_vol_runup = runup_window["volume"].mean()
    avg_vol_pullback = pullback_window["volume"].mean()
    if avg_vol_runup == 0 or avg_vol_pullback >= avg_vol_runup * 0.85:
        return None

    ema20_now = row["ema20"]
    min_low_pullback = pullback_window["low"].min()
    if min_low_pullback < ema20_now * (1 - EMA20_UNDERCUT_TOLERANCE):
        return None
    if close < ema20_now * (1 - EMA20_UNDERCUT_TOLERANCE):
        return None

    recent_base = df.iloc[max(swing_idx, i - 10):i + 1]
    mother_candle = recent_base.loc[(recent_base["high"] - recent_base["low"]).idxmax()]
    entry_trigger = mother_candle["high"]
    stoploss = df.iloc[i - 1]["low"] if i > 0 else row["low"]

    return {
        "symbol": symbol,
        "close": round(close, 2),
        "swing_high": round(swing_high, 2),
        "swing_date": swing_date.date(),
        "pullback_days": pullback_days,
        "pullback_pct": round(pullback_pct, 1),
        "vol_contraction": round(avg_vol_pullback / avg_vol_runup, 2),
        "ema20": round(ema20_now, 2),
        "entry_trigger": round(entry_trigger, 2),
        "stoploss": round(stoploss, 2),
        "distance_to_trigger_pct": round((entry_trigger - close) / close * 100, 1),
    }


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

    print(f"Scanned {len(files)} symbols, {len(results)} match the Stage-2 pullback setup.\n")

    if results:
        out = pd.DataFrame(results).sort_values("distance_to_trigger_pct")
        out.to_csv(OUT_FILE, index=False)
        with pd.option_context("display.max_rows", None, "display.width", 160):
            print(out.to_string(index=False))
        print(f"\nSaved to {OUT_FILE}")


if __name__ == "__main__":
    main()
