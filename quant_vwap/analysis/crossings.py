"""
Detects and counts delta% threshold crossings: a bar where the price moves from
inside a VWAP-proximity band to outside it, crossing further away from VWAP through
one of the defined thresholds (1.5%, 2%, 3%, 4%), tracked separately for price-above-
VWAP and price-below-VWAP.
"""
import numpy as np
import pandas as pd

from features import DELTA_THRESHOLDS


def detect_crossings(df: pd.DataFrame) -> pd.DataFrame:
    """Returns one row per crossing event: timestamp, threshold, direction ('above'/'below')."""
    d = df["delta_pct"].to_numpy()
    session = df["session"].to_numpy()
    prev_d = np.roll(d, 1)
    prev_session = np.roll(session, 1)
    first_of_session = session != prev_session

    events = []
    for thr in DELTA_THRESHOLDS:
        # price above VWAP, crossing further above through +thr
        crossed_up = (prev_d < thr) & (d >= thr) & (~first_of_session)
        # price below VWAP, crossing further below through -thr
        crossed_down = (prev_d > -thr) & (d <= -thr) & (~first_of_session)

        for idx in np.where(crossed_up)[0]:
            events.append((df["timestamp"].iloc[idx], thr, "above", d[idx]))
        for idx in np.where(crossed_down)[0]:
            events.append((df["timestamp"].iloc[idx], thr, "below", d[idx]))

    if not events:
        return pd.DataFrame(columns=["timestamp", "threshold", "direction", "delta_pct"])

    out = pd.DataFrame(events, columns=["timestamp", "threshold", "direction", "delta_pct"])
    return out.sort_values("timestamp").reset_index(drop=True)


def crossing_counts(events: pd.DataFrame) -> pd.DataFrame:
    """Summary count table: rows=threshold, cols=direction."""
    if events.empty:
        idx = pd.Index(DELTA_THRESHOLDS, name="threshold")
        return pd.DataFrame({"above": 0, "below": 0}, index=idx)
    return events.groupby(["threshold", "direction"]).size().unstack(fill_value=0).reindex(
        DELTA_THRESHOLDS
    ).fillna(0).astype(int)
