"""
Lightweight sanity checks for pattern_scanner.py using synthetic OHLCV data,
since real market data isn't needed to validate the detection logic itself.

Run with: python scripts/test_pattern_scanner.py
"""
import numpy as np
import pandas as pd

from pattern_scanner import ScanConfig, compute_indicators, detect_pattern


def _make_frame(closes: np.ndarray, volumes: np.ndarray) -> pd.DataFrame:
    dates = pd.bdate_range("2024-01-01", periods=len(closes))
    high = closes * 1.01
    low = closes * 0.99
    open_ = closes * 1.0
    df = pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": closes, "Volume": volumes},
        index=dates,
    )
    return df


def build_matching_series(seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    slow_len = 150
    rally_len = 60
    uptrend_len = slow_len + rally_len
    decline_len = 12
    base_len = 68

    # a gentle base uptrend, followed by a strong accelerating rally into the
    # peak - real Stage-2 setups usually show a sharp final leg, not a
    # constant slope, which is what makes rule 1 ("strong prior uptrend")
    # meaningful.
    slow_leg = 100 + np.linspace(0, 100, slow_len) + rng.normal(0, 1.5, slow_len)
    rally_leg = 200 + np.linspace(0, 100, rally_len) + rng.normal(0, 1.5, rally_len)
    uptrend = np.concatenate([slow_leg, rally_leg])
    peak_price = 300.0
    uptrend[-1] = peak_price

    decline = np.linspace(peak_price, peak_price * 0.85, decline_len) + rng.normal(0, 1.0, decline_len)

    base = np.linspace(peak_price * 0.85, peak_price * 0.95, base_len) + rng.normal(0, 3.0, base_len)
    base = np.clip(base, peak_price * 0.83, peak_price * 0.99)

    closes = np.concatenate([uptrend, decline, base])

    up_vol = rng.normal(1_000_000, 50_000, uptrend_len - 40)
    pre_peak_vol = rng.normal(1_400_000, 80_000, 40)  # elevated volume into the peak
    decline_vol = rng.normal(1_100_000, 80_000, decline_len)
    base_vol = rng.normal(450_000, 40_000, base_len)  # contracted volume in the base
    volumes = np.concatenate([up_vol, pre_peak_vol, decline_vol, base_vol])
    volumes = np.clip(volumes, 100_000, None)

    return _make_frame(closes, volumes)


def build_too_deep_pullback_series(seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    uptrend_len = 210
    decline_len = 12
    base_len = 68

    uptrend = 100 + np.linspace(0, 200, uptrend_len) + rng.normal(0, 1.5, uptrend_len)
    peak_price = 300.0
    uptrend[-1] = peak_price

    # 40% pullback - well outside the 12-20% rule
    decline = np.linspace(peak_price, peak_price * 0.60, decline_len) + rng.normal(0, 1.0, decline_len)
    base = np.linspace(peak_price * 0.60, peak_price * 0.65, base_len) + rng.normal(0, 3.0, base_len)

    closes = np.concatenate([uptrend, decline, base])
    volumes = rng.normal(800_000, 50_000, len(closes))
    volumes = np.clip(volumes, 100_000, None)
    return _make_frame(closes, volumes)


def build_no_trend_series(seed: int = 2) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = 290
    closes = 150 + rng.normal(0, 5, n).cumsum() * 0.05  # flat/choppy, no real trend
    closes = np.clip(closes, 100, 200)
    volumes = rng.normal(500_000, 50_000, n)
    volumes = np.clip(volumes, 100_000, None)
    return _make_frame(closes, volumes)


def run() -> None:
    cfg = ScanConfig()

    positive_df = compute_indicators(build_matching_series())
    match = detect_pattern("TEST.NS", "Test Co", positive_df, cfg)
    assert match is not None, "expected the synthetic uptrend+pullback+base series to match"
    assert cfg.min_pullback_pct <= match.pullback_pct <= cfg.max_pullback_pct
    assert match.entry_price > match.stop_loss
    assert 0 <= match.score <= 100
    print(f"PASS positive case -> pullback={match.pullback_pct}% score={match.score} "
          f"entry={match.entry_price} stop={match.stop_loss}")

    deep_df = compute_indicators(build_too_deep_pullback_series())
    no_match = detect_pattern("DEEP.NS", "Deep Co", deep_df, cfg)
    assert no_match is None, "a 40% pullback should NOT match the 12-20% rule"
    print("PASS negative case (pullback too deep correctly rejected)")

    flat_df = compute_indicators(build_no_trend_series())
    no_match2 = detect_pattern("FLAT.NS", "Flat Co", flat_df, cfg)
    assert no_match2 is None, "a stock with no stage-2 uptrend should NOT match"
    print("PASS negative case (no stage-2 uptrend correctly rejected)")

    print("\nAll pattern_scanner sanity checks passed.")


if __name__ == "__main__":
    run()
