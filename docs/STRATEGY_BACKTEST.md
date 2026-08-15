# NIFTYBEES / GOLDBEES Positional & Swing Strategy Backtest

Backtest of several well-known, publicly documented positional/swing strategies
using the 10-year daily OHLC data in `data/goldbees_niftybees_ohlc_10y.xlsx`
(Aug 2016 – Aug 2026). Code: `scripts/backtest_strategies.py`.
Raw results: `data/backtest_results.csv`. Equity curves: `data/backtest_equity_curves.png`.

Assumptions: long-only, no leverage/shorting, 0.1% cost per trade (entry or
exit) to approximate brokerage + slippage, cash pays 0% (conservative — real
liquid-fund yield would flatter trend/rotation strategies further).

**Data note:** the raw Yahoo Finance feed had a two-day glitch on
2019-12-19/20 where both symbols show a spurious ~90% price collapse and
round-trip (not a real corporate action — price reverts immediately). The
backtest script auto-detects and drops such rows via a rolling-median filter
before computing anything; without that filter, volatility/drawdown numbers
are meaningless (I hit this and want you to know it's handled).

## Key fact these two funds share

Daily-return correlation between GOLDBEES and NIFTYBEES over the full period
is **~0.02 — effectively zero**. Gold and Indian equity have moved
independently of each other historically (e.g. 2021: gold -5%, Nifty +26%;
2025: gold +72%, Nifty +12%). That's *why* any strategy that combines or
rotates between them tends to beat holding either alone — this isn't a
fitted result, it's the standard rationale for equity/gold tactical
allocation.

## Results (2016–2026)

| Strategy | CAGR | Volatility | Sharpe | Max Drawdown | Monthly Win Rate | Trades |
|---|---|---|---|---|---|---|
| Buy & Hold NIFTYBEES | 12.3% | 14.7% | 0.88 | -36.3% | 60.0% | – |
| Buy & Hold GOLDBEES | 16.0% | 15.9% | 1.03 | -24.4% | 60.8% | – |
| **Static 50/50 (rebalanced monthly)** | 15.0% | **11.0%** | **1.34** | **-20.9%** | **66.7%** | – |
| NIFTYBEES 200-SMA trend-following | 5.6% | 10.1% | 0.60 | -18.5% | 45.0% | 75 |
| GOLDBEES 200-SMA trend-following | 13.4% | 15.2% | 0.93 | -25.0% | 49.2% | 65 |
| **Dual Momentum Rotation (3-month lookback)** | **19.5%** | 15.5% | 1.25 | -24.4% | 60.0% | 30 |
| Dual Momentum Rotation (6-month lookback) | 15.5% | 16.3% | 0.98 | -24.4% | 58.3% | 23 |
| Golden/Death Cross Rotation (50/200 SMA) | 8.1% | 14.6% | 0.62 | -33.9% | 57.5% | 14 |

## What actually works here, in order of practicality

1. **Static 50/50, rebalanced monthly or quarterly.** Best risk-adjusted
   result (Sharpe 1.34) and shallowest drawdown (-21%) of everything tested,
   for near-zero effort: once a month/quarter, sell whichever fund has grown
   to more than 50% of the pot and buy the other back to 50/50. This is the
   standard "permanent portfolio"-style equity/gold split, and it works here
   because of the ~0 correlation above, not because of any market timing.

2. **Dual momentum rotation (3-month lookback), monthly.** Highest CAGR
   (19.5%) and second-best Sharpe. Rule: at each month-end, compute each
   fund's trailing 3-month return; hold 100% in whichever is higher, and go
   to cash (or a liquid fund) if *both* trailing returns are negative. This
   is Gary Antonacci-style dual momentum applied to a 2-asset gold/equity
   universe — a well-documented, non-fitted approach. It only requires ~1
   check and at most 1 trade a month, which fits "positional/swing," not
   day trading.

3. **200-day SMA trend-following on each fund individually underperformed**
   buy & hold here (NIFTYBEES 5.6% CAGR vs 12.3% B&H) — whipsaws in a
   choppy market ate the edge (75 and 65 trades respectively over 10 years).
   This is a known weakness of pure trend-following on a single asset in a
   market that isn't strongly trending; it's more useful as a risk-off
   filter layered on top of momentum/rotation than as a standalone signal.

4. **Golden/Death Cross rotation (park in gold when Nifty's 50-SMA <
   200-SMA)** was the weakest of the rotation approaches (8.1% CAGR, -34%
   drawdown) — the 50/200 cross is slow, so it stayed in gold too long
   after the 2020 COVID crash and missed part of the recovery.

## Caveats (read before using real money)

- 10 years, one country, one asset pair — this is not a statistically large
  sample. The 2016–2026 window includes exactly one big equity crash
  (COVID 2020); results would look different across a longer or different
  history.
- Past performance backtested on historical prices does not guarantee
  future returns. Momentum and correlation regimes can and do break down.
- Costs assumed are minimal (0.1%/trade, 0% on cash); real STT, exit loads,
  and cash drag will lower all numbers, especially the more-active
  strategies (rotation, trend-following).
- This is a research summary, not investment advice — treat it as a
  starting point for your own due diligence, not a signal to size a real
  position off of.

## Reproducing / extending

```
python3 scripts/backtest_strategies.py
```

Edit `scripts/backtest_strategies.py` to try other lookback windows,
rebalance frequencies, or add a 3rd asset (e.g. a liquid/debt fund) as an
explicit cash substitute that earns a return instead of 0%.
