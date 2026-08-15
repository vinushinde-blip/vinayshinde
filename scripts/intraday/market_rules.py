"""
Real broker-imposed intraday (MIS) constraints, shared by every strategy
and the combo-search engine so the backtest can't hold a position past
what would actually happen live.

Zerodha auto-square-offs MIS equity positions starting around 15:12 IST
(exact time can vary by symbol/segment and has moved over the years —
confirm against Zerodha's current published square-off schedule before
trusting this for real capital). Our bar data runs to the actual market
close (~15:25-15:30 bars), well past that — without this, strategies would
hold positions ~15-20 minutes longer than physically possible and price
"EOD" exits off bars that would never happen in a live account.
"""

import datetime as dt

import pandas as pd

SQUAREOFF_TIME = dt.time(15, 12)

# No new entries within ~12 minutes of the hard cutoff — not enough time
# left for a trade thesis to develop before a forced exit.
LAST_ENTRY_TIME = dt.time(15, 0)


def is_entry_allowed(ts) -> bool:
    return ts.time() < LAST_ENTRY_TIME


def infer_bar_interval(df: pd.DataFrame) -> pd.Timedelta:
    """Infers bar spacing from the data itself rather than assuming a fixed
    interval — this pipeline has used both 5-minute and 15-minute fetches,
    and a hardcoded assumption silently goes wrong if that changes. Uses
    the modal gap between consecutive timestamps within a single day (skips
    the overnight gap between days).
    """
    first_day = df.index[0].date()
    day_ts = df.index[df.index.date == first_day]
    if len(day_ts) < 2:
        return pd.Timedelta(minutes=5)  # not enough same-day bars to infer; conservative fallback
    gaps = day_ts.to_series().diff().dropna()
    return gaps.mode().iloc[0]


def squareoff_trigger_time(bar_interval: pd.Timedelta) -> dt.time:
    """The bar START time at/after which a still-open position must be
    force-exited. A bar spanning e.g. 15:00-15:15 (15-min bars) is the one
    actually open AT the real 15:12 cutoff, so the trigger is
    (SQUAREOFF_TIME - one bar interval), not SQUAREOFF_TIME itself —
    otherwise the exit lands one full bar (5-15 min) late.
    """
    dummy_date = dt.date(2000, 1, 1)
    trigger_dt = dt.datetime.combine(dummy_date, SQUAREOFF_TIME) - bar_interval
    return trigger_dt.time()
