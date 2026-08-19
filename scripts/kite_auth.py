"""
Kite Connect login helper. Run this LOCALLY on your own machine — never
paste api_secret or access_token into a chat, issue, or commit.

Setup (one time):
  1. Create a `.env` file in the repo root (already gitignored) with:
       KITE_API_KEY=your_api_key
       KITE_API_SECRET=your_api_secret
  2. In your Kite Connect app settings (developers.kite.trade), set the
     "Redirect URL" to something you control, e.g. http://127.0.0.1:5000/

Daily flow (access_token expires ~6am next trading day):
  1. python scripts/kite_auth.py --login-url
     Open the printed URL, log in, approve. Kite redirects to your
     Redirect URL with a `request_token` query param — copy that value.
  2. python scripts/kite_auth.py --generate-session <request_token>
     Saves the access_token to .kite_session.json (gitignored).
"""
import argparse
import json
import os

from dotenv import load_dotenv
from kiteconnect import KiteConnect

load_dotenv()

SESSION_FILE = ".kite_session.json"


def get_credentials():
    api_key = os.environ.get("KITE_API_KEY")
    api_secret = os.environ.get("KITE_API_SECRET")
    if not api_key or not api_secret:
        raise SystemExit(
            "Set KITE_API_KEY and KITE_API_SECRET in a local .env file (see script docstring)."
        )
    return api_key, api_secret


def print_login_url():
    api_key, _ = get_credentials()
    kite = KiteConnect(api_key=api_key)
    print(kite.login_url())


def generate_session(request_token: str):
    api_key, api_secret = get_credentials()
    kite = KiteConnect(api_key=api_key)
    data = kite.generate_session(request_token, api_secret=api_secret)
    with open(SESSION_FILE, "w") as f:
        json.dump({"access_token": data["access_token"], "login_time": data["login_time"]}, f)
    print(f"Saved access_token to {SESSION_FILE} (expires end of trading day).")


def load_access_token() -> str:
    if not os.path.exists(SESSION_FILE):
        raise SystemExit(f"{SESSION_FILE} not found — run --generate-session first.")
    with open(SESSION_FILE) as f:
        return json.load(f)["access_token"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--login-url", action="store_true", help="Print the Kite login URL")
    group.add_argument("--generate-session", metavar="REQUEST_TOKEN")
    args = parser.parse_args()

    if args.login_url:
        print_login_url()
    else:
        generate_session(args.generate_session)
