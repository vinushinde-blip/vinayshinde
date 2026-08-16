"""
Lightweight sanity checks for tight_base_scanner.py using synthetic OHLCV
data, mirroring the approach in test_pattern_scanner.py.

Run with: python scripts/test_tight_base_scanner.py
"""
import numpy as np
import pandas as pd

from pattern_scanner import compute_indicators
from tight_base_scanner import TightBaseConfig, detect_tight_base


def _make_frame(closes: np.ndarray, volumes: np.ndarray, day_range_pct: np.ndarray) -> pd.DataFrame:
    dates = pd.bdate_range("2024-01-01", periods=len(closes))
    half_range = closes * (day_range_pct / 100) / 2
    high = closes + half_range
    low = closes - half_range
    df = pd.DataFrame(
        {"Open": closes, "High": high, "Low": low, "Close": closes, "Volume": volumes},
        index=dates,
    )
    return df


def build_matching_series(seed: int = 0) -> pd.DataFrame:
    """A long flat/gentle history (to seed SMA150), then a tight ~15-day box
    with contracted volume - mirrors INOXWIND's pre-breakout base."""
    rng = np.random.default_rng(seed)

    history_len = 150
    pre_box_len = 20
    box_len = 15

    history = 55 + rng.normal(0, 1.0, history_len).cumsum() * 0.03
    history = np.clip(history, 45, 65)

    pre_box = 62 + rng.normal(0, 1.5, pre_box_len)

    box = 65 + rng.normal(0, 1.0, box_len)
    box = np.clip(box, 63, 68)  # keeps the box spread under ~8%

    closes = np.concatenate([history, pre_box, box])

    history_vol = rng.normal(500_000, 40_000, history_len)
    pre_box_vol = rng.normal(700_000, 60_000, pre_box_len)  # elevated volume right before the box
    box_vol = rng.normal(200_000, 20_000, box_len)          # contracted volume inside the box
    volumes = np.concatenate([history_vol, pre_box_vol, box_vol])
    volumes = np.clip(volumes, 50_000, None)

    history_range = rng.uniform(2.0, 5.0, history_len)
    pre_box_range = rng.uniform(2.0, 5.0, pre_box_len)
    box_range = rng.uniform(1.0, 3.0, box_len)  # narrow daily ranges inside the box
    day_range_pct = np.concatenate([history_range, pre_box_range, box_range])

    return _make_frame(closes, volumes, day_range_pct)


def build_choppy_no_box_series(seed: int = 1) -> pd.DataFrame:
    """Same length/trend baseline, but no tight box ever forms - wide daily
    ranges throughout, so this should NOT match."""
    rng = np.random.default_rng(seed)
    n = 185
    closes = 55 + rng.normal(0, 1.0, n).cumsum() * 0.03
    closes = np.clip(closes, 45, 70)
    volumes = rng.normal(500_000, 100_000, n)
    volumes = np.clip(volumes, 50_000, None)
    day_range_pct = rng.uniform(5.0, 12.0, n)  # consistently wide, never tight
    return _make_frame(closes, volumes, day_range_pct)


def build_downtrend_series(seed: int = 2) -> pd.DataFrame:
    """A tight box, but the stock is in a clear downtrend - should be
    rejected by the loose trend filter regardless of box quality."""
    rng = np.random.default_rng(seed)
    history_len = 150
    pre_box_len = 20
    box_len = 15

    history = 100 - np.linspace(0, 40, history_len) + rng.normal(0, 1.0, history_len)
    pre_box = 60 + rng.normal(0, 1.5, pre_box_len)
    box = 58 + rng.normal(0, 1.0, box_len)
    box = np.clip(box, 56, 60)

    closes = np.concatenate([history, pre_box, box])
    volumes = np.concatenate([
        rng.normal(500_000, 40_000, history_len),
        rng.normal(700_000, 60_000, pre_box_len),
        rng.normal(200_000, 20_000, box_len),
    ])
    volumes = np.clip(volumes, 50_000, None)
    day_range_pct = np.concatenate([
        rng.uniform(2.0, 5.0, history_len),
        rng.uniform(2.0, 5.0, pre_box_len),
        rng.uniform(1.0, 3.0, box_len),
    ])
    return _make_frame(closes, volumes, day_range_pct)


def run() -> None:
    cfg = TightBaseConfig()

    positive_df = compute_indicators(build_matching_series())
    match = detect_tight_base("TEST.NS", "Test Co", positive_df, cfg)
    assert match is not None, "expected the synthetic tight-box series to match"
    assert match.status in ("BASE", "BREAKOUT")
    assert match.entry_price > match.stop_loss
    assert 0 <= match.score <= 100
    print(f"PASS positive case -> status={match.status} box_len={match.box_len} "
          f"spread={match.box_spread_pct}% vol_ratio={match.volume_contraction_ratio} score={match.score} "
          f"entry={match.entry_price} stop={match.stop_loss}")

    choppy_df = compute_indicators(build_choppy_no_box_series())
    no_match = detect_tight_base("CHOP.NS", "Choppy Co", choppy_df, cfg)
    assert no_match is None, "a stock that never forms a tight box should NOT match"
    print("PASS negative case (no tight box correctly rejected)")

    downtrend_df = compute_indicators(build_downtrend_series())
    no_match2 = detect_tight_base("DOWN.NS", "Downtrend Co", downtrend_df, cfg)
    assert no_match2 is None, "a tight box during a clear downtrend should NOT match"
    print("PASS negative case (downtrend correctly rejected)")

    print("\nAll tight_base_scanner sanity checks passed.")


if __name__ == "__main__":
    run()
