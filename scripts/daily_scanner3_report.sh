#!/bin/bash
# Post-market daily job: refresh today's daily bars, re-run scanner3,
# email the results. Meant to be triggered by a systemd timer (see
# scanner3-report.timer / scanner3-report.service) after market close.
set -e
cd "$(dirname "$0")/.."

source venv/bin/activate
set -a
source .env.scanner2
source .env.email
set +a

python3 scripts/update_daily_bars.py
python3 scripts/scanner3_stage2_pullback.py
python3 scripts/send_scanner3_email.py

deactivate
