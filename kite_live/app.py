import threading
from datetime import datetime

from flask import Flask, redirect, render_template, request
from kiteconnect import KiteConnect

import config
import vwap_signals as vs

app = Flask(__name__)
kite = KiteConnect(api_key=config.API_KEY)

lock = threading.Lock()
state = {
    "status": "not_logged_in",   # not_logged_in -> bootstrapping -> ready | error
    "message": "",
    "signals": [],
    "last_updated": None,
}


def log(msg):
    print(msg, flush=True)
    with lock:
        state["message"] = msg


def background_loop():
    with lock:
        state["status"] = "bootstrapping"
    try:
        log("Loading NSE 500 list...")
        symbols = vs.load_nse500_symbols()

        log("Resolving instrument tokens...")
        instruments_map = vs.resolve_instruments(kite, symbols)
        log(f"Resolved {len(instruments_map)}/{len(symbols)} symbols.")

        log("Bootstrapping last-10-day VWAP stats (this takes a few minutes)...")
        stats = vs.bootstrap_all(kite, instruments_map, progress_cb=log)
        log(f"Bootstrap complete for {len(stats)} symbols.")

        with lock:
            state["status"] = "ready"

        while True:
            try:
                rows = vs.build_signals(kite, instruments_map, stats)
                with lock:
                    state["signals"] = rows
                    state["last_updated"] = datetime.now().strftime("%H:%M:%S")
                    state["message"] = ""
            except Exception as e:
                log(f"Live update failed: {e}")
            threading.Event().wait(config.LIVE_POLL_SECONDS)
    except Exception as e:
        log(f"Fatal error: {e}")
        with lock:
            state["status"] = "error"


@app.route("/")
def index():
    with lock:
        snapshot = dict(state)
    return render_template("index.html", **snapshot)


@app.route("/login")
def login():
    return redirect(kite.login_url())


@app.route("/callback")
def callback():
    request_token = request.args.get("request_token")
    if not request_token:
        return "No request_token received from Kite. Login failed or was cancelled.", 400
    try:
        session_data = kite.generate_session(request_token, api_secret=config.API_SECRET)
        kite.set_access_token(session_data["access_token"])
    except Exception as e:
        return f"Login failed: {e}", 400

    threading.Thread(target=background_loop, daemon=True).start()
    return redirect("/")


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
