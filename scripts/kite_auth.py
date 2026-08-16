"""
Generate a Zerodha Kite Connect access token for local, manual use.

Kite access tokens are valid for a single trading day, so run this once
each morning before scanning with `--data-source kite`. This performs the
standard interactive login (you log in via a browser yourself) rather than
storing your Zerodha password/TOTP anywhere - nothing sensitive beyond the
day's access token touches disk.

Requires KITE_API_KEY and KITE_API_SECRET as environment variables (never
hardcode credentials in a script or commit them to git).

Usage:
    export KITE_API_KEY=xxxx
    export KITE_API_SECRET=xxxx
    python scripts/kite_auth.py

    # then, for the rest of the day:
    export KITE_ACCESS_TOKEN=<token printed below>
    python scripts/pattern_scanner.py --data-source kite
"""
import json
import os
import sys

SESSION_FILE = ".kite_session.json"


def main() -> None:
    api_key = os.environ.get("KITE_API_KEY")
    api_secret = os.environ.get("KITE_API_SECRET")
    if not api_key or not api_secret:
        print("ERROR: set KITE_API_KEY and KITE_API_SECRET environment variables first.", file=sys.stderr)
        sys.exit(1)

    from kiteconnect import KiteConnect

    kite = KiteConnect(api_key=api_key)
    print("1. Open this URL in a browser and log in to Zerodha:\n")
    print(f"   {kite.login_url()}\n")
    print("2. After login you'll be redirected to your app's redirect URL, which")
    print("   contains a 'request_token=...' query parameter. Copy just that value.\n")

    request_token = input("Paste request_token here: ").strip()
    session = kite.generate_session(request_token, api_secret=api_secret)
    access_token = session["access_token"]

    with open(SESSION_FILE, "w") as f:
        json.dump({"access_token": access_token}, f)
    os.chmod(SESSION_FILE, 0o600)

    print(f"\nSaved access token to {SESSION_FILE} (valid until end of trading day, gitignored).")
    print("Or export it directly for this shell session:")
    print(f"   export KITE_ACCESS_TOKEN={access_token}")


if __name__ == "__main__":
    main()
