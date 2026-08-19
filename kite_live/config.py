import os

API_KEY = os.environ.get("KITE_API_KEY")
API_SECRET = os.environ.get("KITE_API_SECRET")
REDIRECT_URL = os.environ.get("KITE_REDIRECT_URL", "http://127.0.0.1:5000/callback")

if not API_KEY or not API_SECRET:
    raise RuntimeError(
        "Set KITE_API_KEY and KITE_API_SECRET as environment variables before "
        "starting the app. Never hardcode them in source files."
    )

CANDLE_INTERVAL = "5minute"
HISTORY_DAYS = 10          # trading days of history used to build the zone thresholds / 10D high
LIVE_POLL_SECONDS = 60     # how often the live loop refreshes quotes
QUOTE_BATCH_SIZE = 200     # instruments per kite.quote() call (Kite's hard cap is 500 per call)
HISTORICAL_CALL_DELAY = 0.35  # seconds between historical_data calls during bootstrap (rate-limit safety)
NSE500_CSV = os.path.join(os.path.dirname(__file__), "nse500_list.csv")
