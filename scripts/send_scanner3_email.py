"""
Emails the latest scanner3_candidates.csv as an Excel attachment via
Gmail SMTP.

Setup
-----
  pip install pandas openpyxl

  # A Gmail App Password, not your real Gmail password - Gmail blocks
  # plain-password SMTP login. Generate one at:
  #   https://myaccount.google.com/apppasswords
  # (requires 2-Step Verification to be turned on for the account first)
  export GMAIL_SENDER_EMAIL=you@gmail.com
  export GMAIL_APP_PASSWORD=xxxxxxxxxxxxxxxx      # 16 chars, no spaces
  export GMAIL_RECIPIENT_EMAIL=you@gmail.com       # can be the same address

Run
---
  python scripts/send_scanner3_email.py

Reads data/scanner3_candidates.csv (written by scanner3_stage2_pullback.py -
run that first) and does nothing if the file doesn't exist or is empty,
rather than sending a blank/error email.
"""

import os
import smtplib
import sys
from datetime import datetime
from email.mime.application import MIMEApplication
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

import pandas as pd

CSV_FILE = os.path.join(os.path.dirname(__file__), "..", "data", "scanner3_candidates.csv")
XLSX_FILE = os.path.join(os.path.dirname(__file__), "..", "data", "scanner3_candidates.xlsx")


def build_excel() -> bool:
    if not os.path.exists(CSV_FILE):
        print(f"No {CSV_FILE} yet - run scanner3_stage2_pullback.py first.")
        return False
    df = pd.read_csv(CSV_FILE)
    if df.empty:
        print("scanner3_candidates.csv is empty - no matches today, skipping email.")
        return False
    with pd.ExcelWriter(XLSX_FILE, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="Scanner3", index=False)
        ws = writer.sheets["Scanner3"]
        for col_cells in ws.columns:
            width = max(len(str(c.value)) if c.value is not None else 0 for c in col_cells) + 2
            ws.column_dimensions[col_cells[0].column_letter].width = min(width, 40)
    return True


def send_email():
    sender = os.environ.get("GMAIL_SENDER_EMAIL")
    password = os.environ.get("GMAIL_APP_PASSWORD")
    recipient = os.environ.get("GMAIL_RECIPIENT_EMAIL")

    if not sender or not password or not recipient:
        print("Set GMAIL_SENDER_EMAIL, GMAIL_APP_PASSWORD, GMAIL_RECIPIENT_EMAIL.")
        sys.exit(1)

    df = pd.read_csv(CSV_FILE)
    today = datetime.now().strftime("%d-%b-%Y")

    msg = MIMEMultipart()
    msg["From"] = sender
    msg["To"] = recipient
    msg["Subject"] = f"Scanner3 candidates - {today} ({len(df)} matches)"

    top = df.sort_values("distance_to_trigger_pct").head(10)
    lines = [f"Scanner3 - {len(df)} stocks matched a contraction pattern today.\n", "Closest to trigger:\n"]
    for _, r in top.iterrows():
        lines.append(f"  {r['symbol']:<12} {r['pattern_types']:<40} close={r['close']}  trigger={r['entry_trigger']}  ({r['distance_to_trigger_pct']}% away)")
    lines.append("\nFull list attached as Excel.")
    msg.attach(MIMEText("\n".join(lines), "plain"))

    with open(XLSX_FILE, "rb") as f:
        part = MIMEApplication(f.read(), Name=os.path.basename(XLSX_FILE))
    part["Content-Disposition"] = f'attachment; filename="{os.path.basename(XLSX_FILE)}"'
    msg.attach(part)

    with smtplib.SMTP("smtp.gmail.com", 587) as server:
        server.starttls()
        server.login(sender, password)
        server.send_message(msg)

    print(f"Emailed {len(df)} candidates to {recipient}.")


def main():
    if build_excel():
        send_email()


if __name__ == "__main__":
    main()
