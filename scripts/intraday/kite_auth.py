"""
Kite Connect login helper.

Credentials are read from environment variables, never hardcoded or logged:
    KITE_API_KEY, KITE_API_SECRET      (from your Kite Connect app)
    KITE_ACCESS_TOKEN                  (optional: skip login if already set)

Kite access tokens expire daily. Flow:
    1. python3 kite_auth.py --login-url
       -> visit the printed URL, log in, get redirected to your app's
          redirect URL with a `request_token` query param.
    2. python3 kite_auth.py --request-token <token>
       -> exchanges it for an access_token, prints it, and writes it to
          .kite_session.json (gitignored) for other scripts in this
          directory to reuse for the rest of the day.
"""

import argparse
import json
import os
import sys

from kiteconnect import KiteConnect

SESSION_FILE = os.path.join(os.path.dirname(__file__), ".kite_session.json")


def get_api_key() -> str:
    key = os.environ.get("KITE_API_KEY")
    if not key:
        sys.exit("KITE_API_KEY environment variable is not set.")
    return key


def get_api_secret() -> str:
    secret = os.environ.get("KITE_API_SECRET")
    if not secret:
        sys.exit("KITE_API_SECRET environment variable is not set.")
    return secret


def load_access_token() -> str:
    token = os.environ.get("KITE_ACCESS_TOKEN")
    if token:
        return token
    if os.path.exists(SESSION_FILE):
        with open(SESSION_FILE) as f:
            return json.load(f)["access_token"]
    sys.exit(
        "No access token available. Run:\n"
        "  python3 kite_auth.py --login-url\n"
        "then after logging in:\n"
        "  python3 kite_auth.py --request-token <token from redirect URL>"
    )


def get_kite() -> KiteConnect:
    kite = KiteConnect(api_key=get_api_key())
    kite.set_access_token(load_access_token())
    return kite


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--login-url", action="store_true")
    parser.add_argument("--request-token")
    parser.add_argument("--verify", action="store_true", help="confirm the current token works")
    args = parser.parse_args()

    kite = KiteConnect(api_key=get_api_key())

    if args.login_url:
        print(kite.login_url())
        return

    if args.request_token:
        session = kite.generate_session(args.request_token, api_secret=get_api_secret())
        access_token = session["access_token"]
        with open(SESSION_FILE, "w") as f:
            json.dump({"access_token": access_token}, f)
        os.chmod(SESSION_FILE, 0o600)
        print(f"Access token saved to {SESSION_FILE} (valid until ~end of trading day).")
        return

    if args.verify:
        kite.set_access_token(load_access_token())
        profile = kite.profile()
        print(f"Authenticated as: {profile['user_name']} ({profile['user_id']})")
        return

    parser.print_help()


if __name__ == "__main__":
    main()
