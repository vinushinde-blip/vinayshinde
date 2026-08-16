"""
Nifty 500 "Stage-2 Pullback to 10/20 EMA" Pattern Scanner
==========================================================

Codifies the setup into 8 checkable rules:

  1. Strong prior uptrend required.
  2. Stock should be in a Stage-2 uptrend.
  3. Pullback should be in an orderly manner (a hard, sharp pullback
     followed by a tight, contained base - not continued erratic selling).
  4. Pullback should be within 12-20% of the swing high.
  5. Strong stocks respect the 10/20 EMA - price hugs the 10/20 EMA in
     the base.
  6. Volume should contract during the pullback/base.
  7. Entry - a buy-stop above the high of the most recent ("mother") candle.
  8. Stoploss - the low of the day before the mother candle.

This is a discretionary chart pattern. Every threshold below is a tunable
constant, not an NSE-official definition - treat matches as candidates to
review on a chart, not as auto-trade signals.

Usage:
    python scripts/pattern_scanner.py
    python scripts/pattern_scanner.py --symbols-file data/nifty500_list.csv \
        --output data/pattern_scan_results.csv --charts --charts-dir data/charts
    python scripts/pattern_scanner.py --symbols RELIANCE,TCS,INFY --limit 50
"""
from __future__ import annotations

import argparse
import dataclasses
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class ScanConfig:
    # --- data window ---
    history_period: str = "2y"
    trading_days_52w: int = 252

    # --- Stage-2 trend template ---
    sma200_trend_lookback: int = 20      # bars back to confirm SMA200 is rising
    min_pct_of_52w_high: float = 0.75    # close within 25% of the 52w high
    min_pct_above_52w_low: float = 1.30  # close at least 30% above the 52w low

    # --- prior uptrend (rule 1) ---
    prior_uptrend_lookback: int = 40     # bars examined before the peak
    prior_uptrend_min_pct: float = 25.0  # minimum % gain into the peak

    # --- base search (rules 3 & 4) ---
    peak_search_window: int = 130        # how far back to look for the swing high
    min_base_len: int = 12               # ~2.5 weeks
    max_base_len: int = 90               # ~4.5 months
    peak_local_window: int = 5           # bars each side for local-max detection
    peak_must_be_in_first_frac: float = 0.30  # the top must sit near the start of the base
    breakout_tolerance: float = 0.02     # base is "not yet broken out" within this tolerance
    min_pullback_pct: float = 12.0
    max_pullback_pct: float = 22.0       # a couple points of slack over the nominal 20%
    orderly_decline_max_frac: float = 0.55  # sharp drop should resolve within this frac of base
    post_trough_max_vol_pct: float = 8.0    # tightness of the basing range after the low

    # --- EMA respect (rule 5) ---
    ema_band_pct: float = 4.0
    ema_respect_min_pct: float = 45.0

    # --- volume contraction (rule 6) ---
    volume_contraction_ratio: float = 0.85  # base volume must be <= 85% of up-leg volume
    recent_volume_window: int = 10

    # --- entry / stoploss (rules 7 & 8) ---
    entry_buffer_pct: float = 0.3

    # --- liquidity filters ---
    min_price: float = 30.0
    min_avg_turnover: float = 1.0e7  # ~1 crore INR/day, 20-day average


@dataclass
class PatternMatch:
    symbol: str
    company: str
    last_close: float
    peak_date: str
    peak_price: float
    trough_date: str
    trough_price: float
    pullback_pct: float
    base_days: int
    ema_respect_pct: float
    volume_contraction_ratio: float
    mother_candle_date: str
    entry_price: float
    stop_loss: float
    risk_pct: float
    score: float


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["EMA10"] = df["Close"].ewm(span=10, adjust=False).mean()
    df["EMA20"] = df["Close"].ewm(span=20, adjust=False).mean()
    df["SMA50"] = df["Close"].rolling(50).mean()
    df["SMA150"] = df["Close"].rolling(150).mean()
    df["SMA200"] = df["Close"].rolling(200).mean()
    df["Turnover"] = df["Close"] * df["Volume"]
    return df


def is_stage2_uptrend(df: pd.DataFrame, cfg: ScanConfig) -> bool:
    last = df.iloc[-1]
    if pd.isna(last[["SMA50", "SMA150", "SMA200"]]).any():
        return False
    if len(df) <= cfg.sma200_trend_lookback:
        return False

    window52 = df.tail(cfg.trading_days_52w)
    high52 = window52["High"].max()
    low52 = window52["Low"].min()
    sma200_now = last["SMA200"]
    sma200_then = df["SMA200"].iloc[-cfg.sma200_trend_lookback]
    if pd.isna(sma200_then):
        return False

    return bool(
        last["Close"] > last["SMA50"]
        and last["SMA50"] > last["SMA150"]
        and last["SMA150"] > last["SMA200"]
        and sma200_now > sma200_then
        and last["Close"] >= cfg.min_pct_of_52w_high * high52
        and last["Close"] >= cfg.min_pct_above_52w_low * low52
    )


def _find_local_peaks(df: pd.DataFrame, cfg: ScanConfig, search_from: int) -> list[int]:
    w = cfg.peak_local_window
    highs = df["High"].values
    n = len(df)
    peaks = []
    for i in range(max(search_from, w), n - 1):  # leave room for a base after the peak
        segment = highs[max(0, i - w): min(n, i + w + 1)]
        if highs[i] == segment.max():
            peaks.append(i)
    return peaks


def find_base(df: pd.DataFrame, cfg: ScanConfig) -> Optional[dict]:
    n = len(df)
    search_from = max(0, n - cfg.peak_search_window)
    candidates = []

    for peak_pos in _find_local_peaks(df, cfg, search_from):
        base_len = n - peak_pos
        if not (cfg.min_base_len <= base_len <= cfg.max_base_len):
            continue

        base = df.iloc[peak_pos:]
        peak_price = base["High"].max()
        peak_offset = int(base["High"].values.argmax())
        if peak_offset > base_len * cfg.peak_must_be_in_first_frac:
            continue  # the highest point isn't near the start - not a top-down base

        # base must not have broken out meaningfully beyond the peak yet
        if base["High"].max() > peak_price * (1 + cfg.breakout_tolerance):
            continue

        post_peak = base.iloc[peak_offset:]
        trough_offset = int(post_peak["Low"].values.argmin())
        trough_price = post_peak["Low"].iloc[trough_offset]
        pullback_pct = (peak_price - trough_price) / peak_price * 100
        if not (cfg.min_pullback_pct <= pullback_pct <= cfg.max_pullback_pct):
            continue

        # rule 3: the decline into the trough should be quick (a "hard" pullback)
        if trough_offset > len(post_peak) * cfg.orderly_decline_max_frac:
            continue

        post_trough = post_peak.iloc[trough_offset:]
        if len(post_trough) < 3:
            continue
        vol_pct = post_trough["Close"].std() / post_trough["Close"].mean() * 100
        if vol_pct > cfg.post_trough_max_vol_pct:
            continue  # still too choppy/erratic to call it an orderly base

        # current price should still be inside the base, not broken down or
        # already run away above the top
        last_close = df["Close"].iloc[-1]
        if not (trough_price * 0.97 <= last_close <= peak_price * (1 + cfg.breakout_tolerance)):
            continue

        # rule 1: strong prior uptrend leading into the peak
        pre_start = max(0, peak_pos - cfg.prior_uptrend_lookback)
        if pre_start == peak_pos:
            continue
        pre_low = df["Low"].iloc[pre_start:peak_pos + 1].min()
        prior_gain_pct = (peak_price - pre_low) / pre_low * 100
        if prior_gain_pct < cfg.prior_uptrend_min_pct:
            continue

        candidates.append({
            "peak_pos": peak_pos + peak_offset,
            "trough_pos": peak_pos + peak_offset + trough_offset,
            "base_start_pos": peak_pos,
            "peak_price": float(peak_price),
            "trough_price": float(trough_price),
            "pullback_pct": float(pullback_pct),
            "base_len": base_len,
            "prior_gain_pct": float(prior_gain_pct),
            "post_trough_vol_pct": float(vol_pct),
        })

    if not candidates:
        return None

    # prefer the widest valid base (earliest start) - it best matches "the box"
    # a trader would draw around the whole consolidation
    candidates.sort(key=lambda c: c["base_start_pos"])
    return candidates[0]


def check_ema_respect(df: pd.DataFrame, base: dict, cfg: ScanConfig) -> float:
    post_trough = df.iloc[base["trough_pos"]:]
    if post_trough.empty:
        return 0.0
    close = post_trough["Close"]
    ema10 = post_trough["EMA10"]
    ema20 = post_trough["EMA20"]
    near_ema10 = (close - ema10).abs() / close <= cfg.ema_band_pct / 100
    near_ema20 = (close - ema20).abs() / close <= cfg.ema_band_pct / 100
    near_either = near_ema10 | near_ema20
    return float(near_either.mean() * 100)


def check_volume_contraction(df: pd.DataFrame, base: dict, cfg: ScanConfig) -> Optional[float]:
    up_leg_start = max(0, base["peak_pos"] - cfg.prior_uptrend_lookback)
    up_leg_vol = df["Volume"].iloc[up_leg_start:base["peak_pos"]].mean()
    if not up_leg_vol or pd.isna(up_leg_vol):
        return None
    recent_vol = df["Volume"].iloc[-cfg.recent_volume_window:].mean()
    ratio = recent_vol / up_leg_vol
    return float(ratio)


def build_score(base: dict, ema_pct: float, vol_ratio: float, cfg: ScanConfig) -> float:
    # closeness of pullback to the 12-20% sweet spot (16% center)
    pullback_score = max(0.0, 1 - abs(base["pullback_pct"] - 16) / 10)
    ema_score = min(1.0, ema_pct / 100)
    vol_score = max(0.0, min(1.0, (cfg.volume_contraction_ratio - vol_ratio) / cfg.volume_contraction_ratio + 0.5))
    tightness_score = max(0.0, 1 - base["post_trough_vol_pct"] / cfg.post_trough_max_vol_pct)
    prior_trend_score = min(1.0, base["prior_gain_pct"] / 60)
    weights = {
        "pullback": 0.25,
        "ema": 0.25,
        "vol": 0.20,
        "tightness": 0.15,
        "trend": 0.15,
    }
    total = (
        weights["pullback"] * pullback_score
        + weights["ema"] * ema_score
        + weights["vol"] * vol_score
        + weights["tightness"] * tightness_score
        + weights["trend"] * prior_trend_score
    )
    return round(total * 100, 1)


def detect_pattern(symbol: str, company: str, df: pd.DataFrame, cfg: ScanConfig) -> Optional[PatternMatch]:
    if df is None or len(df) < 210:
        return None

    df = df.dropna(subset=["Open", "High", "Low", "Close", "Volume"])
    if len(df) < 210:
        return None

    last_close = float(df["Close"].iloc[-1])
    avg_turnover = df["Turnover"].tail(20).mean()
    if last_close < cfg.min_price or avg_turnover < cfg.min_avg_turnover:
        return None

    if not is_stage2_uptrend(df, cfg):  # rule 2
        return None

    base = find_base(df, cfg)  # rules 1, 3, 4
    if base is None:
        return None

    ema_pct = check_ema_respect(df, base, cfg)  # rule 5
    if ema_pct < cfg.ema_respect_min_pct:
        return None

    vol_ratio = check_volume_contraction(df, base, cfg)  # rule 6
    if vol_ratio is None or vol_ratio > cfg.volume_contraction_ratio:
        return None

    # rules 7 & 8: mother candle = most recent bar, stop = the bar before it
    mother = df.iloc[-1]
    prev_day = df.iloc[-2]
    entry_price = float(mother["High"] * (1 + cfg.entry_buffer_pct / 100))
    stop_loss = float(prev_day["Low"])
    if stop_loss >= entry_price:
        return None
    risk_pct = (entry_price - stop_loss) / entry_price * 100

    score = build_score(base, ema_pct, vol_ratio, cfg)

    return PatternMatch(
        symbol=symbol,
        company=company,
        last_close=round(last_close, 2),
        peak_date=str(df.index[base["peak_pos"]].date()),
        peak_price=round(base["peak_price"], 2),
        trough_date=str(df.index[base["trough_pos"]].date()),
        trough_price=round(base["trough_price"], 2),
        pullback_pct=round(base["pullback_pct"], 2),
        base_days=base["base_len"],
        ema_respect_pct=round(ema_pct, 1),
        volume_contraction_ratio=round(vol_ratio, 2),
        mother_candle_date=str(df.index[-1].date()),
        entry_price=round(entry_price, 2),
        stop_loss=round(stop_loss, 2),
        risk_pct=round(risk_pct, 2),
        score=score,
    )


def load_symbols(path: str, limit: Optional[int] = None) -> list[tuple[str, str]]:
    """Returns (nse_symbol, company) pairs, nse_symbol WITHOUT any ".NS" suffix -
    each data source below maps that to whatever format it needs."""
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    if "YF_Symbol" in df.columns:
        nse_syms = df["YF_Symbol"].str.replace(".NS", "", regex=False)
        symbols = list(zip(nse_syms, df.get("Company Name", nse_syms)))
    elif "Symbol" in df.columns:
        nse_syms = df["Symbol"].str.strip().str.replace(".NS", "", regex=False)
        symbols = list(zip(nse_syms, df.get("Company Name", nse_syms)))
    else:
        # single unlabeled column of tickers
        col = df.columns[0]
        symbols = [(str(s).replace(".NS", ""), s) for s in df[col]]
    if limit:
        symbols = symbols[:limit]
    return symbols


def _download_history_yfinance(nse_symbol: str, period: str) -> Optional[pd.DataFrame]:
    import yfinance as yf

    df = yf.download(f"{nse_symbol}.NS", period=period, interval="1d", auto_adjust=False, progress=False)
    if df is None or df.empty:
        return None
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df[["Open", "High", "Low", "Close", "Volume"]]


def download_history(nse_symbol: str, period: str, source: str = "yfinance") -> Optional[pd.DataFrame]:
    if source == "yfinance":
        return _download_history_yfinance(nse_symbol, period)
    if source == "kite":
        import kite_data_provider

        return kite_data_provider.download_history(nse_symbol, period)
    raise ValueError(f"unknown data source: {source}")


def scan_symbol(nse_symbol: str, company: str, cfg: ScanConfig, source: str) -> Optional[PatternMatch]:
    display_symbol = f"{nse_symbol}.NS"
    try:
        raw = download_history(nse_symbol, cfg.history_period, source)
        if raw is None:
            return None
        df = compute_indicators(raw)
        return detect_pattern(display_symbol, company, df, cfg)
    except Exception as exc:  # noqa: BLE001 - keep scanning the rest of the universe
        print(f"  [warn] {display_symbol}: {exc}", file=sys.stderr)
        return None


def save_chart(symbol: str, df: pd.DataFrame, match: PatternMatch, out_dir: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.patches as patches
    import matplotlib.pyplot as plt

    os.makedirs(out_dir, exist_ok=True)
    plot_df = df.tail(180)

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(plot_df.index, plot_df["Close"], color="black", linewidth=1, label="Close")
    ax.plot(plot_df.index, plot_df["EMA10"], color="tab:blue", linewidth=1, label="EMA10")
    ax.plot(plot_df.index, plot_df["EMA20"], color="tab:orange", linewidth=1, label="EMA20")

    peak_date = pd.Timestamp(match.peak_date)
    last_date = plot_df.index[-1]
    if peak_date >= plot_df.index[0]:
        x0 = mdates.date2num(peak_date)
        x1 = mdates.date2num(last_date)
        rect = patches.Rectangle(
            (x0, match.trough_price),
            (x1 - x0) or 1,
            match.peak_price - match.trough_price,
            linewidth=1.5, edgecolor="red", facecolor="none",
        )
        ax.add_patch(rect)
    ax.axhline(match.entry_price, color="green", linestyle="--", linewidth=1, label="Entry")
    ax.axhline(match.stop_loss, color="red", linestyle="--", linewidth=1, label="Stoploss")

    ax.set_title(f"{symbol}  score={match.score}  pullback={match.pullback_pct}%")
    ax.legend(loc="upper left", fontsize=8)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"{symbol.replace('.', '_')}.png"), dpi=120)
    plt.close(fig)


def run_scan(args: argparse.Namespace) -> pd.DataFrame:
    cfg = ScanConfig()
    source = args.data_source

    if args.symbols:
        symbols = [(s.strip(), s.strip()) for s in args.symbols.split(",") if s.strip()]
    else:
        symbols = load_symbols(args.symbols_file, args.limit)

    print(f"Scanning {len(symbols)} symbols via {source}...")
    matches: list[PatternMatch] = []

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {}
        for symbol, company in symbols:
            futures[pool.submit(scan_symbol, symbol, company, cfg, source)] = (symbol, company)

        done = 0
        for fut in as_completed(futures):
            symbol, company = futures[fut]
            done += 1
            if done % 25 == 0:
                print(f"  ...{done}/{len(symbols)} scanned")
            result = fut.result()
            if result is not None:
                matches.append(result)
                print(f"  MATCH: {result.symbol}  score={result.score}  pullback={result.pullback_pct}%")

    if not matches:
        print("No matches found.")
        return pd.DataFrame()

    result_df = pd.DataFrame([dataclasses.asdict(m) for m in matches])
    result_df = result_df.sort_values("score", ascending=False).reset_index(drop=True)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    result_df.to_csv(args.output, index=False)
    print(f"\nSaved {len(result_df)} matches to {args.output}")

    if args.charts:
        print(f"Rendering charts to {args.charts_dir} ...")
        for m in matches:
            try:
                nse_symbol = m.symbol.replace(".NS", "")
                raw = download_history(nse_symbol, cfg.history_period, source)
                df = compute_indicators(raw)
                save_chart(m.symbol, df, m, args.charts_dir)
            except Exception as exc:  # noqa: BLE001
                print(f"  [warn] chart failed for {m.symbol}: {exc}", file=sys.stderr)

    return result_df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--symbols-file", default="data/nifty500_list.csv",
                         help="CSV with a 'Symbol' or 'YF_Symbol' column (see fetch_nifty500_list.py)")
    parser.add_argument("--symbols", default=None,
                         help="Comma-separated NSE symbols to scan instead of the symbols file, e.g. RELIANCE,TCS")
    parser.add_argument("--limit", type=int, default=None, help="Only scan the first N symbols (useful for testing)")
    parser.add_argument("--output", default="data/pattern_scan_results.csv")
    parser.add_argument("--charts", action="store_true", help="Save an annotated PNG chart per match")
    parser.add_argument("--charts-dir", default="data/charts")
    parser.add_argument("--workers", type=int, default=8,
                         help="Concurrent downloads. With --data-source kite, a shared rate limiter keeps "
                              "requests under Kite's API limit regardless of this value.")
    parser.add_argument("--data-source", choices=["yfinance", "kite"], default="yfinance",
                         help="yfinance (default, free, no setup) or kite (Zerodha Kite Connect - requires "
                              "KITE_API_KEY/KITE_ACCESS_TOKEN, see scripts/kite_auth.py)")
    args = parser.parse_args()

    run_scan(args)


if __name__ == "__main__":
    main()
