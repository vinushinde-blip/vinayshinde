"""
Backtest known positional/swing strategies on GOLDBEES + NIFTYBEES daily OHLC data.

Strategies tested (all long-only, no leverage, no shorting):
  1. Buy & Hold NIFTYBEES
  2. Buy & Hold GOLDBEES
  3. Static 50/50, rebalanced monthly
  4. 200-day SMA trend-following per asset (in when Close > SMA200, else cash)
  5. Dual momentum rotation: monthly, hold whichever of {NIFTYBEES, GOLDBEES}
     had the higher trailing N-month return; go to cash if both are negative
  6. Golden/Death Cross rotation: hold NIFTYBEES while its 50-SMA > 200-SMA,
     switch to GOLDBEES while 50-SMA < 200-SMA (classic equity/gold rotation)

Costs: 0.1% per switch (buy or sell) to approximate brokerage + slippage.
Cash is assumed to earn 0% (conservative; real cash/liquid-fund yield would help
the trend-following and rotation strategies further).
"""

import numpy as np
import pandas as pd

TRADING_DAYS = 252
COST = 0.001  # 0.1% per trade (entry or exit)


def _clean_price_glitches(df, window=11, threshold=0.4):
    """Drop rows whose Close deviates wildly from the local rolling median.

    The Yahoo Finance feed for these NSE ETFs has occasional single/double-day
    glitches (e.g. GOLDBEES and NIFTYBEES both show a ~90% price collapse and
    round-trip on 2019-12-19/20 that isn't a real corporate action). Left in,
    a handful of these rows dominate volatility/drawdown stats. We flag any
    Close that is <threshold or >1/threshold times the centered rolling
    median and drop those rows entirely (OHLCV all suspect, not just Close).
    """
    med = df["Close"].rolling(window, center=True, min_periods=3).median()
    ratio = df["Close"] / med
    bad = (ratio < threshold) | (ratio > 1 / threshold)
    return df.loc[~bad.fillna(False)]


def load_data(path="data/goldbees_niftybees_ohlc_10y.xlsx"):
    xl = pd.ExcelFile(path)
    data = {}
    for sheet in xl.sheet_names:
        df = xl.parse(sheet)
        df = df.dropna(subset=["Close"]).copy()
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.set_index("Date").sort_index()
        df = _clean_price_glitches(df)
        data[sheet] = df
    return data


def metrics(equity: pd.Series, trades: int = None):
    ret = equity.pct_change().dropna()
    n_years = (equity.index[-1] - equity.index[0]).days / 365.25
    cagr = (equity.iloc[-1] / equity.iloc[0]) ** (1 / n_years) - 1
    vol = ret.std() * np.sqrt(TRADING_DAYS)
    sharpe = (ret.mean() * TRADING_DAYS) / vol if vol > 0 else np.nan
    dd = equity / equity.cummax() - 1
    max_dd = dd.min()
    monthly = equity.resample("ME").last().pct_change().dropna()
    win_rate = (monthly > 0).mean()
    out = {
        "CAGR": f"{cagr:.2%}",
        "Volatility": f"{vol:.2%}",
        "Sharpe": f"{sharpe:.2f}",
        "MaxDrawdown": f"{max_dd:.2%}",
        "MonthlyWinRate": f"{win_rate:.2%}",
        "FinalMultiple": f"{equity.iloc[-1] / equity.iloc[0]:.2f}x",
    }
    if trades is not None:
        out["Trades"] = trades
    return out


def buy_hold(df):
    close = df["Close"]
    equity = close / close.iloc[0]
    equity.iloc[0] *= (1 - COST)  # one entry cost
    return equity


def static_5050(dfs, rebalance="ME"):
    close = pd.DataFrame({k: v["Close"] for k, v in dfs.items()}).dropna()
    rets = close.pct_change().fillna(0)
    weights = pd.Series(0.5, index=close.columns)
    equity = [1.0]
    dates = close.index
    rebal_dates = set(close.resample(rebalance).last().index)
    cur_weights = weights.copy()
    for i in range(1, len(dates)):
        day_ret = (cur_weights * rets.iloc[i]).sum()
        new_val = equity[-1] * (1 + day_ret)
        # drift weights with price moves
        cur_weights = cur_weights * (1 + rets.iloc[i])
        cur_weights /= cur_weights.sum()
        if dates[i] in rebal_dates:
            turnover = (cur_weights - weights).abs().sum() / 2
            new_val *= (1 - turnover * COST)
            cur_weights = weights.copy()
        equity.append(new_val)
    return pd.Series(equity, index=dates) * (1 - COST)


def sma_trend(df, window=200):
    close = df["Close"]
    sma = close.rolling(window).mean()
    signal = (close > sma).astype(int)
    signal = signal.shift(1).fillna(0)  # trade next day on prior day's signal
    rets = close.pct_change().fillna(0)
    strat_ret = signal * rets
    switch_mask = (signal.diff().abs().fillna(0) == 1)
    trades = int(switch_mask.sum())
    strat_ret = strat_ret.where(~switch_mask, strat_ret - COST)
    equity = (1 + strat_ret).cumprod()
    return equity, trades


def dual_momentum_rotation(dfs, lookback_months=3):
    close = pd.DataFrame({k: v["Close"] for k, v in dfs.items()}).dropna()
    monthly = close.resample("ME").last()
    mom = monthly.pct_change(lookback_months)
    rets = close.pct_change().fillna(0)

    holding = None
    equity = [1.0]
    dates = close.index
    trades = 0
    decision_dates = monthly.index

    current_pick = None
    for i in range(1, len(dates)):
        d = dates[i]
        # rebalance decision made at each month-end, applied next trading day
        prev_month_ends = decision_dates[decision_dates < d]
        if len(prev_month_ends) == 0 or len(mom.dropna()) == 0:
            pick = None
        else:
            last_me = prev_month_ends[-1]
            if last_me in mom.index and not mom.loc[last_me].isna().all():
                row = mom.loc[last_me]
                best = row.idxmax()
                pick = best if row[best] > 0 else "CASH"
            else:
                pick = None

        if pick != current_pick:
            trades += 1
            cost = COST
        else:
            cost = 0.0

        day_ret = rets.iloc[i][pick] if pick not in (None, "CASH") else 0.0
        equity.append(equity[-1] * (1 + day_ret) * (1 - cost))
        current_pick = pick

    return pd.Series(equity, index=dates), trades


def golden_cross_rotation(dfs, fast=50, slow=200):
    n_close = dfs["NIFTYBEES"]["Close"]
    g_close = dfs["GOLDBEES"]["Close"]
    common = n_close.index.intersection(g_close.index)
    n_close, g_close = n_close.loc[common], g_close.loc[common]

    n_fast, n_slow = n_close.rolling(fast).mean(), n_close.rolling(slow).mean()
    regime = (n_fast > n_slow).astype(int)  # 1 = risk-on (hold Nifty), 0 = hold Gold
    regime = regime.shift(1).fillna(0)

    n_ret = n_close.pct_change().fillna(0)
    g_ret = g_close.pct_change().fillna(0)
    strat_ret = regime * n_ret + (1 - regime) * g_ret
    switch_mask = (regime.diff().abs().fillna(0) == 1)
    trades = int(switch_mask.sum())
    strat_ret = strat_ret.where(~switch_mask, strat_ret - COST)
    equity = (1 + strat_ret).cumprod()
    return equity, trades


def main():
    dfs = load_data()
    results = {}

    results["Buy&Hold NIFTYBEES"] = metrics(buy_hold(dfs["NIFTYBEES"]))
    results["Buy&Hold GOLDBEES"] = metrics(buy_hold(dfs["GOLDBEES"]))
    results["Static 50/50 (monthly rebal)"] = metrics(static_5050(dfs))

    eq, tr = sma_trend(dfs["NIFTYBEES"], 200)
    results["NIFTYBEES 200-SMA Trend"] = metrics(eq, tr)

    eq, tr = sma_trend(dfs["GOLDBEES"], 200)
    results["GOLDBEES 200-SMA Trend"] = metrics(eq, tr)

    eq, tr = dual_momentum_rotation(dfs, lookback_months=3)
    results["Dual Momentum Rotation (3M)"] = metrics(eq, tr)

    eq, tr = dual_momentum_rotation(dfs, lookback_months=6)
    results["Dual Momentum Rotation (6M)"] = metrics(eq, tr)

    eq, tr = golden_cross_rotation(dfs)
    results["Golden/Death Cross Rotation (50/200)"] = metrics(eq, tr)

    table = pd.DataFrame(results).T
    print(table.to_string())
    table.to_csv("data/backtest_results.csv")
    print("\nSaved: data/backtest_results.csv")


if __name__ == "__main__":
    main()
