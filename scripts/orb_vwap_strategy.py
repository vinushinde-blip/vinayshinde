"""NSE500 intraday strategy: Opening-Range Breakout + VWAP + EMA trend filter.

Runs on 5-minute candles saved by scripts/fetch_intraday_5m.py
(data/intraday_5m/<SYMBOL>.csv).

Entry (long, mirrored for short):
  - 5m close breaks above the opening range high (first 15 minutes, 09:15-09:30)
  - close is above VWAP
  - EMA9 > EMA21 (trend confirmation)
  - candle volume > 1.5x the rolling 20-bar average (breakout conviction)
  - no new entries after 15:00 IST

Exit: ATR(14)-based stop, 2R target, or forced square-off at 15:15 IST.
Sizing: fixed-fractional risk (0.5% of equity per trade).

Usage: python scripts/orb_vwap_strategy.py
"""

import glob
import os

import numpy as np
import pandas as pd

DATA_DIR = "data/intraday_5m"
TRADES_OUT = "data/backtest_trades.csv"

MARKET_OPEN = "09:15"
MARKET_CLOSE = "15:30"
OR_END = "09:30"  # opening range = first 15 minutes (3 x 5-min candles)
NO_NEW_ENTRY_AFTER = "15:00"
SQUARE_OFF = "15:15"

RISK_PCT = 0.005  # risk 0.5% of equity per trade
REWARD_RISK_RATIO = 2.0  # target = 2R
VOL_SURGE_MULT = 1.5  # entry candle volume must exceed 1.5x rolling avg volume
ATR_LEN = 14
EMA_FAST, EMA_SLOW = 9, 21
STARTING_CAPITAL = 100_000


def load_symbol(path):
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df = df.rename(columns=str.title)
    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    df.index = df.index.tz_convert("Asia/Kolkata")
    return df


def add_indicators(df):
    df = df.copy()
    df["EmaFast"] = df["Close"].ewm(span=EMA_FAST, adjust=False).mean()
    df["EmaSlow"] = df["Close"].ewm(span=EMA_SLOW, adjust=False).mean()

    prev_close = df["Close"].shift(1)
    true_range = pd.concat(
        [
            df["High"] - df["Low"],
            (df["High"] - prev_close).abs(),
            (df["Low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    df["Atr"] = true_range.ewm(alpha=1 / ATR_LEN, adjust=False).mean()

    df["AvgVolume20"] = df["Volume"].rolling(20, min_periods=5).mean()

    date = df.index.date
    typical_price = (df["High"] + df["Low"] + df["Close"]) / 3
    cum_tp_vol = (typical_price * df["Volume"]).groupby(date).cumsum()
    cum_vol = df["Volume"].groupby(date).cumsum()
    df["Vwap"] = cum_tp_vol / cum_vol.replace(0, np.nan)

    return df


def backtest_symbol(symbol, df):
    df = add_indicators(df)
    trades = []

    for day, day_df in df.groupby(df.index.date):
        day_df = day_df.between_time(MARKET_OPEN, MARKET_CLOSE)
        opening = day_df.between_time(MARKET_OPEN, OR_END)
        if len(opening) < 2:
            continue
        or_high = opening["High"].max()
        or_low = opening["Low"].min()

        rest = day_df.between_time(OR_END, MARKET_CLOSE)
        in_position = False
        position = None

        for ts, row in rest.iterrows():
            if pd.isna(row["Atr"]) or pd.isna(row["AvgVolume20"]) or pd.isna(row["Vwap"]):
                continue

            if not in_position:
                if ts.strftime("%H:%M") > NO_NEW_ENTRY_AFTER:
                    break

                vol_ok = row["Volume"] > VOL_SURGE_MULT * row["AvgVolume20"]
                long_signal = (
                    row["Close"] > or_high
                    and row["Close"] > row["Vwap"]
                    and row["EmaFast"] > row["EmaSlow"]
                    and vol_ok
                )
                short_signal = (
                    row["Close"] < or_low
                    and row["Close"] < row["Vwap"]
                    and row["EmaFast"] < row["EmaSlow"]
                    and vol_ok
                )
                if not (long_signal or short_signal):
                    continue

                direction = 1 if long_signal else -1
                entry_price = row["Close"]
                stop_dist = max(row["Atr"], entry_price * 0.001)
                position = {
                    "symbol": symbol,
                    "date": day,
                    "direction": "LONG" if direction == 1 else "SHORT",
                    "entry_time": ts,
                    "entry_price": entry_price,
                    "stop_price": entry_price - direction * stop_dist,
                    "target_price": entry_price + direction * stop_dist * REWARD_RISK_RATIO,
                    "risk_per_share": stop_dist,
                }
                in_position = True
                continue

            direction = 1 if position["direction"] == "LONG" else -1
            hit_stop = (
                row["Low"] <= position["stop_price"]
                if direction == 1
                else row["High"] >= position["stop_price"]
            )
            hit_target = (
                row["High"] >= position["target_price"]
                if direction == 1
                else row["Low"] <= position["target_price"]
            )
            time_exit = ts.strftime("%H:%M") >= SQUARE_OFF

            if not (hit_stop or hit_target or time_exit):
                continue

            if hit_stop:
                exit_price, reason = position["stop_price"], "STOP"
            elif hit_target:
                exit_price, reason = position["target_price"], "TARGET"
            else:
                exit_price, reason = row["Close"], "TIME_EXIT"

            pnl_per_share = (exit_price - position["entry_price"]) * direction
            r_multiple = pnl_per_share / position["risk_per_share"]

            trades.append(
                {
                    **position,
                    "exit_time": ts,
                    "exit_price": exit_price,
                    "exit_reason": reason,
                    "pnl_per_share": pnl_per_share,
                    "r_multiple": r_multiple,
                }
            )
            in_position = False
            position = None

    return trades


def summarize(trades_df):
    if trades_df.empty:
        print("No trades generated.")
        return

    trades_df = trades_df.sort_values("exit_time").reset_index(drop=True)
    equity = STARTING_CAPITAL
    equity_curve = []
    for r in trades_df["r_multiple"]:
        equity += equity * RISK_PCT * r
        equity_curve.append(equity)
    trades_df["equity"] = equity_curve

    wins = trades_df[trades_df["r_multiple"] > 0]
    losses = trades_df[trades_df["r_multiple"] <= 0]
    win_rate = len(wins) / len(trades_df)
    avg_r = trades_df["r_multiple"].mean()
    profit_factor = (
        wins["r_multiple"].sum() / abs(losses["r_multiple"].sum()) if len(losses) else np.inf
    )

    running_max = trades_df["equity"].cummax()
    max_dd = ((trades_df["equity"] - running_max) / running_max).min()

    print(f"Trades:        {len(trades_df)}")
    print(f"Win rate:      {win_rate:.1%}")
    print(f"Avg R:         {avg_r:.2f}")
    print(f"Profit factor: {profit_factor:.2f}")
    print(f"Final equity:  {equity:,.0f} (start {STARTING_CAPITAL:,.0f})")
    print(f"Max drawdown:  {max_dd:.1%}")


def main():
    paths = sorted(glob.glob(os.path.join(DATA_DIR, "*.csv")))
    if not paths:
        print(f"No candle data found in {DATA_DIR}. Run scripts/fetch_intraday_5m.py first.")
        return

    all_trades = []
    for path in paths:
        symbol = os.path.splitext(os.path.basename(path))[0]
        df = load_symbol(path)
        if df.empty:
            continue
        trades = backtest_symbol(symbol, df)
        all_trades.extend(trades)
        print(f"{symbol}: {len(trades)} trades")

    trades_df = pd.DataFrame(all_trades)
    os.makedirs(os.path.dirname(TRADES_OUT), exist_ok=True)
    trades_df.to_csv(TRADES_OUT, index=False)
    print(f"\nSaved {len(trades_df)} trades -> {TRADES_OUT}")
    summarize(trades_df)


if __name__ == "__main__":
    main()
