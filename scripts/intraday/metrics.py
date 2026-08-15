import numpy as np
import pandas as pd

TRADING_DAYS = 252


def performance_metrics(daily_returns: pd.Series) -> dict:
    if daily_returns.empty or len(daily_returns) < 2:
        return {"CAGR": None, "Volatility": None, "Sharpe": None,
                "MaxDrawdown": None, "DailyWinRate": None, "Trades/Day": None}

    equity = (1 + daily_returns.fillna(0)).cumprod()
    n_years = max((daily_returns.index[-1] - daily_returns.index[0]).days / 365.25, 1 / 252)
    cagr = equity.iloc[-1] ** (1 / n_years) - 1
    vol = daily_returns.std() * np.sqrt(TRADING_DAYS)
    sharpe = (daily_returns.mean() * TRADING_DAYS) / vol if vol > 0 else np.nan
    dd = equity / equity.cummax() - 1
    max_dd = dd.min()
    win_rate = (daily_returns > 0).mean()

    return {
        "CAGR": f"{cagr:.2%}",
        "Volatility": f"{vol:.2%}",
        "Sharpe": f"{sharpe:.2f}",
        "MaxDrawdown": f"{max_dd:.2%}",
        "DailyWinRate": f"{win_rate:.2%}",
        "TradingDays": len(daily_returns),
    }


def walk_forward_split(daily_returns: pd.Series, train_frac: float = 0.6):
    """Split a return series chronologically into an in-sample (train) and
    out-of-sample (test) period. Parameters should be chosen only on train;
    the numbers that matter are the ones from test.
    """
    n = len(daily_returns)
    cut = int(n * train_frac)
    return daily_returns.iloc[:cut], daily_returns.iloc[cut:]
