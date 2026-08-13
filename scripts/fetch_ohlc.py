import os
from datetime import datetime, timedelta

import pandas as pd
import yfinance as yf

TICKERS = {
    "GOLDBEES": "GOLDBEES.NS",
    "NIFTYBEES": "NIFTYBEES.NS",
}

end_date = datetime.today()
start_date = end_date - timedelta(days=365 * 10)

os.makedirs("data", exist_ok=True)
output_file = "data/goldbees_niftybees_ohlc_10y.xlsx"

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

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df = df.reset_index()
        df = df[["Date", "Open", "High", "Low", "Close", "Adj Close", "Volume"]]

        df.to_excel(writer, sheet_name=name, index=False)
        print(f"{name}: {len(df)} rows written ({df['Date'].min().date()} to {df['Date'].max().date()})")

print(f"Saved: {output_file}")
