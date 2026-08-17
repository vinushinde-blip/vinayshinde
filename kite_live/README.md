# NSE 500 VWAP Distance Signals (Kite Connect)

Live signal page for NSE 500 stocks: for each stock, compares the current
distance from today's VWAP (in %) against that stock's own last-10-trading-day
distribution, and flags stocks currently **crossing their own 10-day highest
VWAP distance %**. Signals only — this app never places orders.

Zones (per stock, from that stock's own 10-day 5-min VWAP-distance history):
- **Average** (green) — within 1 std-dev of its historical distance%
- **Medium** (orange) — 1–2 std-dev
- **Extreme** (red) — beyond 2 std-dev

## Setup

```bash
cd kite_live
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Set your Kite Connect credentials as environment variables — never hardcode
them in any file:

```bash
export KITE_API_KEY=your_api_key
export KITE_API_SECRET=your_api_secret
```

Your Kite Connect app's **Redirect URL** (in the Kite developer console) must
be set to `http://127.0.0.1:5000/callback`.

> If you pasted your API secret anywhere insecure (chat, a shared doc, etc.),
> regenerate it at developers.kite.trade before using it here.

## Run

```bash
python app.py
```

Open `http://127.0.0.1:5000`, click **Login with Kite**, and complete the
normal Zerodha login (user ID/password + 2FA) in your browser. Kite redirects
back to the app, which then:

1. Resolves NSE 500 tradingsymbols to instrument tokens.
2. Bootstraps each stock's last 10 trading days of 5-min candles to compute
   its zone thresholds and 10-day highest distance% (~3–5 minutes for 500
   stocks, due to Kite's historical-data rate limit).
3. Every 60 seconds, fetches live quotes for all 500 stocks in batched calls
   and recomputes the table. The page auto-refreshes every 60 seconds.

Kite access tokens expire daily — you'll need to log in again each trading
day (visit `/login` again).

## Files

- `app.py` — Flask app: login flow, background polling loop, page rendering.
- `vwap_signals.py` — VWAP/zone/crossing calculation logic.
- `config.py` — reads credentials from environment variables.
- `nse500_list.csv` — NSE 500 constituent list (from NSE's official archive).
- `templates/index.html` — the live table page.
