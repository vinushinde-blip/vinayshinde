"""
Fully vectorized (no per-event Python loops) behavioral analysis of price action
after delta% crosses into a VWAP-proximity band: reversion probability at several
forward horizons, directional forward return, and max adverse excursion — all
conditioned on indicator state at the moment of entry. Designed to scale to
hundreds of thousands of crossing events per stock/timeframe without being the
bottleneck of the pipeline.
"""
import numpy as np
import pandas as pd

from features import DELTA_THRESHOLDS

HORIZONS = [5, 10, 20, 40]


def _fwd_rolling(series: pd.Series, window: int, how: str) -> pd.Series:
    rev = series[::-1]
    if how == "min":
        out = rev.rolling(window, min_periods=1).min()[::-1]
    elif how == "max":
        out = rev.rolling(window, min_periods=1).max()[::-1]
    else:
        raise ValueError(how)
    return out


def _rsi_bucket(rsi: pd.Series) -> pd.Series:
    return pd.cut(rsi, bins=[-1, 30, 70, 101], labels=["oversold(<30)", "neutral(30-70)", "overbought(>70)"])


def _adx_bucket(adx: pd.Series) -> pd.Series:
    return pd.cut(adx, bins=[-1, 20, 40, 1000], labels=["weak(<20)", "moderate(20-40)", "strong(>40)"])


def _vol_bucket(vr: pd.Series) -> pd.Series:
    return pd.cut(vr, bins=[-1, 0.8, 1.5, 1000], labels=["low(<0.8x)", "normal(0.8-1.5x)", "high(>1.5x)"])


def _bb_bucket(pct_b: pd.Series) -> pd.Series:
    return pd.cut(pct_b, bins=[-1000, 0, 1, 1000], labels=["below_lower", "inside_bands", "above_upper"])


def build_event_table(df: pd.DataFrame, symbol: str, timeframe: str, nifty_regime: pd.DataFrame = None) -> pd.DataFrame:
    """One row per crossing event with forward outcomes + indicator state at entry."""
    n = len(df)

    nifty_regime_arr = None
    if nifty_regime is not None:
        merged = pd.merge(
            df[["timestamp"]], nifty_regime[["timestamp", "nifty_regime"]],
            on="timestamp", how="left",
        )
        nifty_regime_arr = merged["nifty_regime"].fillna("neutral").to_numpy()
    delta = df["delta_pct"].to_numpy()
    abs_delta = np.abs(delta)
    session = df["session"].to_numpy()
    close = df["close"].to_numpy()

    prev_delta = np.roll(delta, 1)
    prev_session = np.roll(session, 1)
    first_of_session = session != prev_session

    fwd_min_delta = {h: _fwd_rolling(df["delta_pct"], h + 1, "min").to_numpy() for h in HORIZONS}
    fwd_max_delta = {h: _fwd_rolling(df["delta_pct"], h + 1, "max").to_numpy() for h in HORIZONS}
    fwd_min_abs_delta = {h: _fwd_rolling(df["delta_pct"].abs(), h + 1, "min").to_numpy() for h in HORIZONS}

    close_s = pd.Series(close)
    fwd_close = {h: close_s.shift(-h).to_numpy() for h in HORIZONS}

    session_at_h = {h: pd.Series(session).shift(-h).to_numpy() for h in HORIZONS}

    rsi_b = _rsi_bucket(df["rsi_14"]).to_numpy()
    adx_b = _adx_bucket(df["adx_14"]).to_numpy()
    vol_b = _vol_bucket(df["vol_ratio"]).to_numpy()
    bb_b = _bb_bucket(df["bb_pct_b"]).to_numpy()
    macd_sign = np.where(df["macd_hist"].to_numpy() >= 0, "positive", "negative")

    rows = []
    for thr in DELTA_THRESHOLDS:
        for direction, cond in (
            ("above", (prev_delta < thr) & (delta >= thr) & (~first_of_session)),
            ("below", (prev_delta > -thr) & (delta <= -thr) & (~first_of_session)),
        ):
            idx = np.where(cond)[0]
            if len(idx) == 0:
                continue
            sign = 1.0 if direction == "above" else -1.0

            rec = {
                "symbol": symbol,
                "timeframe": timeframe,
                "threshold": thr,
                "direction": direction,
                "timestamp": df["timestamp"].to_numpy()[idx],
                "entry_delta_pct": delta[idx],
                "entry_close": close[idx],
                "rsi_bucket": rsi_b[idx],
                "adx_bucket": adx_b[idx],
                "vol_bucket": vol_b[idx],
                "bb_bucket": bb_b[idx],
                "macd_sign": macd_sign[idx],
                "nifty_regime": nifty_regime_arr[idx] if nifty_regime_arr is not None else "n/a",
            }

            for h in HORIZONS:
                same_session = session_at_h[h][idx] == session[idx]
                raw_ret = (fwd_close[h][idx] - close[idx]) / close[idx] * 100.0
                directional_ret = raw_ret * (-sign)
                rec[f"ret_{h}"] = np.where(same_session, directional_ret, np.nan)

                band_exit = fwd_min_abs_delta[h][idx] < thr
                rec[f"band_exit_{h}"] = np.where(same_session, band_exit, np.nan)

                if direction == "above":
                    full_reversion = fwd_min_delta[h][idx] <= 0
                else:
                    full_reversion = fwd_max_delta[h][idx] >= 0
                rec[f"full_reversion_{h}"] = np.where(same_session, full_reversion, np.nan)

                if direction == "above":
                    adverse = fwd_max_delta[h][idx] - delta[idx]
                else:
                    adverse = delta[idx] - fwd_min_delta[h][idx]
                rec[f"adverse_excursion_{h}"] = np.where(same_session, adverse, np.nan)

            rows.append(pd.DataFrame(rec))

    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def summarize_events(events: pd.DataFrame, group_cols) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame()
    agg = {"symbol": "count"}
    for h in HORIZONS:
        agg[f"ret_{h}"] = "mean"
        agg[f"band_exit_{h}"] = "mean"
        agg[f"full_reversion_{h}"] = "mean"
        agg[f"adverse_excursion_{h}"] = "mean"
    out = events.groupby(group_cols, observed=True).agg(agg).rename(columns={"symbol": "n_events"})
    return out.reset_index()
