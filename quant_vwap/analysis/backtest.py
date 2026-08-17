"""
Mean-reversion backtest engine.

Strategy family: when delta% (price vs session VWAP) crosses further away from VWAP
through a threshold (1.5/2/3/4%), take a position betting on reversion toward VWAP:
  - price ABOVE VWAP by >= threshold  -> SHORT (expect reversion down)
  - price BELOW VWAP by >= threshold  -> LONG  (expect reversion up)

Entry executes at the NEXT bar's open (no lookahead). Exit is the first of:
  - target hit (take-profit)
  - stop-loss hit
  - end of session (forced intraday square-off; VWAP resets daily so positions
    are never held overnight)
On a same-bar ambiguity (both TP and SL touched within one bar), the stop-loss is
assumed to fire first (conservative).

Two target:stop-loss variants are backtested: 1.0%:0.5% and 1.5%:0.5%.
"""
import numpy as np
import pandas as pd

from features import DELTA_THRESHOLDS

TP_SL_VARIANTS = [(1.0, 0.5), (1.5, 0.5)]


def _session_last_idx(session: np.ndarray) -> np.ndarray:
    s = pd.Series(session)
    return s.groupby(s).transform(lambda g: g.index.max()).to_numpy()
    # NOTE: groups by value, but session values can repeat non-contiguously only
    # if data spans >1 fetch chunk with duplicates removed; timestamps are sorted
    # so within a stock's frame this is safe.


def _entry_events(delta: np.ndarray, session: np.ndarray):
    prev_delta = np.roll(delta, 1)
    prev_session = np.roll(session, 1)
    first_of_session = session != prev_session
    events = []
    for thr in DELTA_THRESHOLDS:
        up = np.where((prev_delta < thr) & (delta >= thr) & (~first_of_session))[0]
        down = np.where((prev_delta > -thr) & (delta <= -thr) & (~first_of_session))[0]
        events.append((thr, "above", up))
        events.append((thr, "below", down))
    return events


def backtest_symbol(df: pd.DataFrame, symbol: str, timeframe: str) -> pd.DataFrame:
    df = df.reset_index(drop=True)
    n = len(df)
    open_ = df["open"].to_numpy()
    high = df["high"].to_numpy()
    low = df["low"].to_numpy()
    close = df["close"].to_numpy()
    delta = df["delta_pct"].to_numpy()
    session = df["session"].to_numpy()
    ts = df["timestamp"].to_numpy()

    sess_last = _session_last_idx(pd.factorize(session)[0])

    trades = []
    for thr, direction, idx_arr in _entry_events(delta, session):
        for target, stop in TP_SL_VARIANTS:
            cursor = -1
            for sig_idx in idx_arr:
                entry_idx = sig_idx + 1
                if entry_idx <= cursor:
                    continue  # still in a prior open trade for this variant
                end_idx = sess_last[entry_idx] if entry_idx < n else -1
                if entry_idx >= n or end_idx < entry_idx:
                    continue  # no bars left in session to enter/hold

                entry_price = open_[entry_idx]
                is_long = direction == "below"

                if is_long:
                    tp_price = entry_price * (1 + target / 100)
                    sl_price = entry_price * (1 - stop / 100)
                else:
                    tp_price = entry_price * (1 - target / 100)
                    sl_price = entry_price * (1 + stop / 100)

                w_high = high[entry_idx:end_idx + 1]
                w_low = low[entry_idx:end_idx + 1]
                w_close = close[entry_idx:end_idx + 1]

                if is_long:
                    tp_hit = w_high >= tp_price
                    sl_hit = w_low <= sl_price
                else:
                    tp_hit = w_low <= tp_price
                    sl_hit = w_high >= sl_price

                tp_pos = np.argmax(tp_hit) if tp_hit.any() else None
                sl_pos = np.argmax(sl_hit) if sl_hit.any() else None

                if sl_pos is not None and (tp_pos is None or sl_pos <= tp_pos):
                    exit_offset, exit_price, reason = sl_pos, sl_price, "SL"
                elif tp_pos is not None:
                    exit_offset, exit_price, reason = tp_pos, tp_price, "TP"
                else:
                    exit_offset, exit_price, reason = len(w_close) - 1, w_close[-1], "EOD"

                exit_idx = entry_idx + exit_offset
                pnl_pct = (exit_price - entry_price) / entry_price * 100 * (1 if is_long else -1)

                trades.append((
                    symbol, timeframe, thr, direction, target, stop,
                    ts[sig_idx], ts[entry_idx], ts[exit_idx],
                    entry_price, exit_price, pnl_pct, reason, exit_offset + 1,
                ))
                cursor = exit_idx

    cols = [
        "symbol", "timeframe", "threshold", "direction", "target_pct", "stop_pct",
        "signal_time", "entry_time", "exit_time", "entry_price", "exit_price",
        "pnl_pct", "exit_reason", "bars_held",
    ]
    return pd.DataFrame(trades, columns=cols)


def summarize_trades(trades: pd.DataFrame, group_cols) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()

    def _agg(g: pd.DataFrame) -> pd.Series:
        wins = g.loc[g.pnl_pct > 0, "pnl_pct"]
        losses = g.loc[g.pnl_pct <= 0, "pnl_pct"]
        gross_win = wins.sum()
        gross_loss = -losses.sum()
        n = len(g)
        return pd.Series({
            "n_trades": n,
            "win_rate_pct": 100 * (g.pnl_pct > 0).mean(),
            "avg_pnl_pct": g.pnl_pct.mean(),
            "avg_win_pct": wins.mean() if len(wins) else 0.0,
            "avg_loss_pct": losses.mean() if len(losses) else 0.0,
            "total_pnl_pct": g.pnl_pct.sum(),
            "profit_factor": (gross_win / gross_loss) if gross_loss > 0 else np.inf,
            "expectancy_pct": g.pnl_pct.mean(),
            "sharpe_like": (g.pnl_pct.mean() / g.pnl_pct.std()) if g.pnl_pct.std() > 1e-6 else np.nan,
            "max_drawdown_pct": _max_drawdown(g.sort_values("exit_time")["pnl_pct"].to_numpy()),
            "avg_bars_held": g.bars_held.mean(),
            "tp_rate_pct": 100 * (g.exit_reason == "TP").mean(),
            "sl_rate_pct": 100 * (g.exit_reason == "SL").mean(),
            "eod_rate_pct": 100 * (g.exit_reason == "EOD").mean(),
        })

    return trades.groupby(group_cols, observed=True).apply(_agg, include_groups=False).reset_index()


def _max_drawdown(pnl_sequence: np.ndarray) -> float:
    if len(pnl_sequence) == 0:
        return 0.0
    equity = np.cumsum(pnl_sequence)
    running_max = np.maximum.accumulate(equity)
    drawdown = equity - running_max
    return drawdown.min()
