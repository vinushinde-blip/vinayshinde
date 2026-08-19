# Kite Connect tick capture — setup

Run all of this locally. Never paste `api_secret` or `access_token` into a
chat, issue, commit, or the `.env` file's git history — `.env` and
`.kite_session.json` are already gitignored.

1. `pip install -r requirements.txt`
2. Create `.env` in the repo root:
   ```
   KITE_API_KEY=your_api_key
   KITE_API_SECRET=your_api_secret
   ```
3. In your Kite Connect app (developers.kite.trade), set Redirect URL to
   a page you control (e.g. `http://127.0.0.1:5000/`).
4. Daily, before market open (access_token expires ~6am next day):
   ```
   python scripts/kite_auth.py --login-url
   # open the URL, log in, copy request_token from the redirect
   python scripts/kite_auth.py --generate-session <request_token>
   ```
5. One-time (re-run if NSE500 constituents change):
   ```
   python scripts/kite_instruments.py
   ```
6. During market hours (9:15-15:30 IST):
   ```
   python scripts/kite_tick_capture.py
   ```
   Ticks land in `data/ticks/YYYY-MM-DD.parquet`.

Kite does not sell historical tick/depth data, so this is how a tick
history gets built going forward. See main chat thread for options on
sourcing pre-existing historical tick data if you don't want to wait
weeks/months to accumulate a usable sample.
