"""
Sanity-checks the strategy/engine/metrics pipeline end-to-end on synthetic
15-minute bar data (no Kite connection needed). This does NOT tell us
whether any strategy has real edge — it only proves the mechanics (signal
generation, cost application, aggregation, metrics) run without errors and
produce internally consistent numbers. Real validation happens once actual
Kite intraday data is available.
"""

import numpy as np
import pandas as pd

from strategies import STRATEGIES
from engine import apply_costs, portfolio_daily_returns, equity_curve
from metrics import performance_metrics, walk_forward_split


def make_synthetic_symbol(seed: int, n_days: int = 120, bars_per_day: int = 25,
                           base_price: float = 500.0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    price = base_price
    start = pd.Timestamp("2024-01-01 09:15")
    for d in range(n_days):
        day_date = start + pd.Timedelta(days=d)
        if day_date.weekday() >= 5:
            continue
        day_open = price * (1 + rng.normal(0, 0.004))
        p = day_open
        for b in range(bars_per_day):
            drift = rng.normal(0, 0.0015)
            p = max(p * (1 + drift), 0.5)
            o = p
            h = p * (1 + abs(rng.normal(0, 0.001)))
            l = p * (1 - abs(rng.normal(0, 0.001)))
            c = p * (1 + rng.normal(0, 0.0008))
            v = max(int(rng.normal(10000, 4000)), 100)
            ts = day_date + pd.Timedelta(minutes=15 * b)
            rows.append({"datetime": ts, "open": o, "high": max(h, o, c),
                         "low": min(l, o, c), "close": c, "volume": v})
            p = c
        price = p
    df = pd.DataFrame(rows).set_index("datetime")
    return df


def main():
    symbols = {f"SYNTH{i}": make_synthetic_symbol(seed=i) for i in range(5)}

    for strat_name, strat_fn in STRATEGIES.items():
        all_trades = {}
        for rank, (sym, df) in enumerate(symbols.items(), start=1):
            trades = strat_fn(df)
            trades = apply_costs(trades, liquidity_rank=rank, universe_size=len(symbols))
            all_trades[sym] = trades

        daily = portfolio_daily_returns(all_trades)
        eq = equity_curve(daily)
        m = performance_metrics(daily)

        train, test = walk_forward_split(daily)
        m_train = performance_metrics(train)
        m_test = performance_metrics(test)

        total_trades = sum(len(t) for t in all_trades.values())
        print(f"\n=== {strat_name} ===")
        print(f"Total trades: {total_trades}, trading days with activity: {len(daily)}")
        print(f"Full period:  {m}")
        print(f"Train (60%):  {m_train}")
        print(f"Test (40%):   {m_test}")

        assert eq.iloc[0] > 0, "equity curve should start positive"
        assert not daily.isna().all(), "daily returns should not be all-NaN"

    print("\nSelf-test passed: pipeline runs end-to-end without errors.")


if __name__ == "__main__":
    main()
