"""
Nifty 500 "Tight Base Breakout" Pattern Scanner
=================================================

A second, distinct setup from pattern_scanner.py's stage-2 pullback. This
one does NOT require a big prior uptrend to already be in place - it just
looks for a stock sitting in (or having just broken out of) a tight,
low-volatility, low-volume consolidation box: a classic Darvas-box / narrow-
range base, the kind of quiet 1-3 week flat spell that often precedes a
sharp directional move.

Rules (my own codification of the pattern, not an NSE-official definition):
  1. A recent N-day box where most days are individually "narrow range"
     ((High-Low)/Close under a threshold).
  2. The box as a whole stays tight (top-to-bottom spread is small, not
     just each day in isolation).
  3. Volume contracts inside the box versus the weeks before it.
  4. The stock isn't in an outright downtrend (a loose trend filter -
     deliberately looser than pattern_scanner.py's full Stage-2 template,
     since this setup can occur earlier in a base-building phase).
  5. Current price is inside the box, or has broken above it within the
     last few days.
  6. Entry = box high + buffer; Stoploss = box low (the whole base is the
     risk zone, the standard Darvas-box convention).

Usage:
    python scripts/tight_base_scanner.py
    python scripts/tight_base_scanner.py --data-source kite --charts
    python scripts/tight_base_scanner.py --symbols INOXWIND,IDEA --data-source kite
"""
from __future__ import annotations

import argparse
import dataclasses
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Optional

import pandas as pd

from pattern_scanner import compute_indicators, download_history, load_symbols


@dataclass
class TightBaseConfig:
    history_period: str = "1y"

    # --- box search ---
    box_min_len: int = 8
    box_max_len: int = 20
    box_search_window: int = 6        # how many recent bar-positions may serve as the box's last day

    # --- tightness (rule 1 & 2) ---
    narrow_day_max_range_pct: float = 4.0   # a single day's (High-Low)/Close to count as "narrow"
    narrow_day_min_frac: float = 0.70       # >= this fraction of box days must be narrow
    box_max_spread_pct: float = 9.0         # overall (box_high-box_low)/box_high cap

    # --- volume contraction (rule 3) ---
    pre_box_lookback: int = 20
    volume_contraction_ratio: float = 0.75  # box avg volume <= this fraction of the pre-box average

    # --- loose trend filter (rule 4) ---
    trend_filter_sma: int = 150
    trend_filter_min_ratio: float = 0.92    # close >= this fraction of SMA150 - not in a steep downtrend

    # --- position vs. the box (rule 5) ---
    breakdown_tolerance_pct: float = 3.0    # last close can't be more than this far below the box low
    breakout_max_extension_pct: float = 15.0  # last close can't be more than this far above the box high

    # --- entry / stoploss (rule 6) ---
    entry_buffer_pct: float = 0.3

    # --- liquidity filters ---
    min_price: float = 20.0
    min_avg_turnover: float = 1.0e7


@dataclass
class TightBaseMatch:
    symbol: str
    company: str
    status: str  # "BASE" (still inside the box) or "BREAKOUT" (closed above it recently)
    last_close: float
    box_start_date: str
    box_end_date: str
    box_len: int
    box_high: float
    box_low: float
    box_spread_pct: float
    narrow_day_frac: float
    volume_contraction_ratio: float
    entry_price: float
    stop_loss: float
    risk_pct: float
    score: float


def passes_trend_filter(df: pd.DataFrame, cfg: TightBaseConfig) -> bool:
    if len(df) < cfg.trend_filter_sma:
        return False
    sma = df["Close"].rolling(cfg.trend_filter_sma).mean().iloc[-1]
    if pd.isna(sma):
        return False
    return bool(df["Close"].iloc[-1] >= cfg.trend_filter_min_ratio * sma)


def find_tight_box(df: pd.DataFrame, cfg: TightBaseConfig) -> Optional[dict]:
    n = len(df)
    earliest_end = max(cfg.box_min_len, n - cfg.box_search_window)
    candidates = []

    for box_end in range(earliest_end, n):
        for box_len in range(cfg.box_min_len, cfg.box_max_len + 1):
            box_start = box_end - box_len + 1
            if box_start < cfg.pre_box_lookback:
                continue

            box = df.iloc[box_start:box_end + 1]
            day_range_pct = (box["High"] - box["Low"]) / box["Close"] * 100
            narrow_frac = float((day_range_pct <= cfg.narrow_day_max_range_pct).mean())
            if narrow_frac < cfg.narrow_day_min_frac:
                continue

            box_high = float(box["High"].max())
            box_low = float(box["Low"].min())
            box_spread_pct = (box_high - box_low) / box_high * 100
            if box_spread_pct > cfg.box_max_spread_pct:
                continue

            pre_box_vol = df["Volume"].iloc[box_start - cfg.pre_box_lookback:box_start].mean()
            if not pre_box_vol or pd.isna(pre_box_vol):
                continue
            box_vol = box["Volume"].mean()
            vol_ratio = float(box_vol / pre_box_vol)
            if vol_ratio > cfg.volume_contraction_ratio:
                continue

            candidates.append({
                "box_start": box_start,
                "box_end": box_end,
                "box_len": box_len,
                "box_high": box_high,
                "box_low": box_low,
                "box_spread_pct": box_spread_pct,
                "narrow_day_frac": narrow_frac,
                "vol_ratio": vol_ratio,
            })

    if not candidates:
        return None

    # prefer the widest (longest) box, most recently completed
    candidates.sort(key=lambda c: (-c["box_len"], -c["box_end"]))
    return candidates[0]


def build_score(box: dict, cfg: TightBaseConfig, status: str) -> float:
    tightness_score = max(0.0, 1 - box["box_spread_pct"] / cfg.box_max_spread_pct)
    narrow_score = box["narrow_day_frac"]
    vol_score = max(0.0, min(1.0, (cfg.volume_contraction_ratio - box["vol_ratio"]) / cfg.volume_contraction_ratio + 0.5))
    length_score = max(0.0, min(1.0, 1 - abs(box["box_len"] - 12) / 12))
    weights = {"tightness": 0.30, "narrow": 0.25, "vol": 0.25, "length": 0.20}
    total = (
        weights["tightness"] * tightness_score
        + weights["narrow"] * narrow_score
        + weights["vol"] * vol_score
        + weights["length"] * length_score
    )
    if status == "BREAKOUT":
        total = min(1.0, total + 0.10)  # already confirming, slightly favour it over a still-forming base
    return round(total * 100, 1)


def detect_tight_base(symbol: str, company: str, df: pd.DataFrame, cfg: TightBaseConfig) -> Optional[TightBaseMatch]:
    if df is None or len(df) < cfg.trend_filter_sma + cfg.pre_box_lookback:
        return None

    df = df.dropna(subset=["Open", "High", "Low", "Close", "Volume"])
    if len(df) < cfg.trend_filter_sma + cfg.pre_box_lookback:
        return None

    last_close = float(df["Close"].iloc[-1])
    avg_turnover = (df["Close"] * df["Volume"]).tail(20).mean()
    if last_close < cfg.min_price or avg_turnover < cfg.min_avg_turnover:
        return None

    if not passes_trend_filter(df, cfg):  # rule 4
        return None

    box = find_tight_box(df, cfg)  # rules 1, 2, 3
    if box is None:
        return None

    box_high, box_low = box["box_high"], box["box_low"]
    if last_close < box_low * (1 - cfg.breakdown_tolerance_pct / 100):
        return None  # base already failed to the downside
    if last_close > box_high * (1 + cfg.breakout_max_extension_pct / 100):
        return None  # already run too far past the box to use it as a reference

    status = "BREAKOUT" if last_close > box_high else "BASE"
    entry_price = float(box_high * (1 + cfg.entry_buffer_pct / 100))
    stop_loss = box_low
    if stop_loss >= entry_price:
        return None
    risk_pct = (entry_price - stop_loss) / entry_price * 100

    score = build_score(box, cfg, status)

    return TightBaseMatch(
        symbol=symbol,
        company=company,
        status=status,
        last_close=round(last_close, 2),
        box_start_date=str(df.index[box["box_start"]].date()),
        box_end_date=str(df.index[box["box_end"]].date()),
        box_len=box["box_len"],
        box_high=round(box_high, 2),
        box_low=round(box_low, 2),
        box_spread_pct=round(box["box_spread_pct"], 2),
        narrow_day_frac=round(box["narrow_day_frac"] * 100, 1),
        volume_contraction_ratio=round(box["vol_ratio"], 2),
        entry_price=round(entry_price, 2),
        stop_loss=round(stop_loss, 2),
        risk_pct=round(risk_pct, 2),
        score=score,
    )


def scan_symbol(nse_symbol: str, company: str, cfg: TightBaseConfig, source: str) -> Optional[TightBaseMatch]:
    display_symbol = f"{nse_symbol}.NS"
    try:
        raw = download_history(nse_symbol, cfg.history_period, source)
        if raw is None:
            return None
        df = compute_indicators(raw)
        return detect_tight_base(display_symbol, company, df, cfg)
    except Exception as exc:  # noqa: BLE001 - keep scanning the rest of the universe
        print(f"  [warn] {display_symbol}: {exc}", file=sys.stderr)
        return None


def save_chart(symbol: str, df: pd.DataFrame, match: TightBaseMatch, out_dir: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.patches as patches
    import matplotlib.pyplot as plt

    os.makedirs(out_dir, exist_ok=True)
    plot_df = df.tail(120)

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(plot_df.index, plot_df["Close"], color="black", linewidth=1, label="Close")

    box_start = pd.Timestamp(match.box_start_date)
    box_end = pd.Timestamp(match.box_end_date)
    if box_start >= plot_df.index[0]:
        x0 = mdates.date2num(box_start)
        x1 = mdates.date2num(box_end)
        rect = patches.Rectangle(
            (x0, match.box_low),
            (x1 - x0) or 1,
            match.box_high - match.box_low,
            linewidth=1.5, edgecolor="red", facecolor="none",
        )
        ax.add_patch(rect)
    ax.axhline(match.entry_price, color="green", linestyle="--", linewidth=1, label="Entry")
    ax.axhline(match.stop_loss, color="red", linestyle="--", linewidth=1, label="Stoploss")

    ax.set_title(f"{symbol}  status={match.status}  score={match.score}  box_spread={match.box_spread_pct}%")
    ax.legend(loc="upper left", fontsize=8)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"{symbol.replace('.', '_')}.png"), dpi=120)
    plt.close(fig)


def run_scan(args: argparse.Namespace) -> pd.DataFrame:
    cfg = TightBaseConfig()
    source = args.data_source

    if args.symbols:
        symbols = [(s.strip(), s.strip()) for s in args.symbols.split(",") if s.strip()]
    else:
        symbols = load_symbols(args.symbols_file, args.limit)

    print(f"Scanning {len(symbols)} symbols for tight-base setups via {source}...")
    matches: list[TightBaseMatch] = []

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(scan_symbol, symbol, company, cfg, source): (symbol, company) for symbol, company in symbols}

        done = 0
        for fut in as_completed(futures):
            done += 1
            if done % 25 == 0:
                print(f"  ...{done}/{len(symbols)} scanned")
            result = fut.result()
            if result is not None:
                matches.append(result)
                print(f"  MATCH: {result.symbol}  status={result.status}  score={result.score}  "
                      f"box_spread={result.box_spread_pct}%")

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
    parser.add_argument("--symbols-file", default="data/nifty500_list.csv")
    parser.add_argument("--symbols", default=None,
                         help="Comma-separated NSE symbols to scan instead of the symbols file")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output", default="data/tight_base_scan_results.csv")
    parser.add_argument("--charts", action="store_true")
    parser.add_argument("--charts-dir", default="data/tight_base_charts")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--data-source", choices=["yfinance", "kite"], default="yfinance")
    args = parser.parse_args()

    run_scan(args)


if __name__ == "__main__":
    main()
