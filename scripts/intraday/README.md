# Intraday strategy backtest (NSE top-500 liquid stocks)

Status: **code complete, not yet run on live data.** This environment's
network policy currently blocks `api.kite.trade`; once that's opened
(environment network settings, not something changeable from inside a
session) and Kite Connect credentials are provided as environment
variables, run the pipeline below.

## Credentials (never commit these)

Set as environment variables, not pasted into chat:

```
KITE_API_KEY=...
KITE_API_SECRET=...
```

## Run order

```bash
cd scripts/intraday

# 1. Log in (access tokens expire daily)
python3 kite_auth.py --login-url
# visit the URL, log in, copy the request_token from the redirect URL
python3 kite_auth.py --request-token <token>
python3 kite_auth.py --verify

# 2. Build the top-500-liquid universe from Kite's own volume data
#    (resumable — safe to re-run if it gets interrupted)
python3 kite_universe.py --lookback-days 60 --top-n 500

# 3. Fetch intraday bars for the universe (default 15-minute; see the
#    file's docstring for why — Kite's history limits make 1-minute bars
#    impractical at this scale, only 60 days retained)
python3 kite_fetch_intraday.py --interval 15minute --lookback-days 200

# 4. Backtest every strategy, long+short, with realistic costs and a
#    train/test (walk-forward) split so results aren't just overfit noise
python3 run_backtest.py
```

## What `run_backtest.py` reports

For each strategy in `strategies.py` (opening range breakout, VWAP mean
reversion, momentum+volume breakout):

- Full-period CAGR/Sharpe/max drawdown
- **Train (first 60%) vs test (last 40%) — judge the strategy on test,
  not train or full-period.** A strategy that only looks good on train is
  overfit, not real.
- CAGR at 2x the assumed cost model, to show how sensitive the result is to
  the cost/slippage assumptions being right.

## Design notes

- **Universe is liquidity-ranked from Kite's own data**, not NSE's Nifty
  500 index list — avoids needing `nseindia.com` access, and liquidity
  (traded turnover) is a more direct proxy for "can I actually get in/out
  of this without moving the price" than index membership.
- **Costs** (`costs.py`) model Zerodha's actual intraday equity charge
  stack (brokerage, STT, exchange fees, stamp duty, GST) plus a slippage
  estimate that scales with a stock's liquidity rank — the bottom of the
  500 is modeled as costing more to trade than the top, which is realistic
  but still an approximation.
- **No look-ahead**: every strategy enters at the *next* bar's open after
  the signal bar closes, never at the signal bar's own close.
- `selftest_synthetic.py` validates the engine mechanics on synthetic
  random-walk data with no dependency on Kite — it should always show
  negative returns after costs (there's no real signal in noise); if it
  ever shows suspiciously good returns, that's a look-ahead bug, not edge.

## Caveats

- Even a "rigorous" backtest here only covers whatever window of intraday
  history Kite retains (likely well under a year at 15-minute granularity)
  — a single, fairly narrow slice of market conditions.
- Intraday alpha at the top-500-liquid-stock level is heavily contested by
  institutional/HFT flow already; a naive strategy showing an edge in this
  backtest is more likely to be a costs/slippage modeling gap than real,
  persistent alpha. Treat the 2x-cost-sensitivity column as the first
  thing to check before believing any positive result.
- This is research tooling, not a signal to trade real capital on directly.
