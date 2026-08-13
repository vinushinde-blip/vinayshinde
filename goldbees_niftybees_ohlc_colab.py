# GOLDBEES & NIFTYBEES - 10 Year Daily OHLC Data Export
# ---------------------------------------------------------
# Run this in Google Colab. Paste each numbered block into its own cell
# (or paste the whole file into one cell and run it).

# --- Cell 1: install dependencies ---
!pip install -q yfinance openpyxl

# --- Cell 2: fetch and export ---
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

# NSE tickers on Yahoo Finance use the ".NS" suffix
TICKERS = {
    "GOLDBEES": "GOLDBEES.NS",
    "NIFTYBEES": "NIFTYBEES.NS",
}

end_date = datetime.today()
start_date = end_date - timedelta(days=365 * 10)

output_file = "goldbees_niftybees_ohlc_10y.xlsx"

with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
    for name, ticker in TICKERS.items():
        df = yf.download(
            ticker,
            start=start_date.strftime("%Y-%m-%d"),
            end=end_date.strftime("%Y-%m-%d"),
            interval="1d",
            auto_adjust=False,
            progress=False,
        )

        if df.empty:
            print(f"Warning: no data returned for {ticker}")
            continue

        # yfinance can return MultiIndex columns for a single ticker; flatten if needed
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df = df.reset_index()  # bring Date out of the index into a column
        df = df[["Date", "Open", "High", "Low", "Close", "Adj Close", "Volume"]]

        df.to_excel(writer, sheet_name=name, index=False)
        print(f"{name}: {len(df)} rows written ({df['Date'].min().date()} to {df['Date'].max().date()})")

print(f"\nSaved: {output_file}")

# --- Cell 3: download the file to your computer ---
from google.colab import files
files.download(output_file)
