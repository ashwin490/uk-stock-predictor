import yfinance as yf
import pandas as pd
import os
import time

DATA_DIR = "./data/raw_ohlcv"
os.makedirs(DATA_DIR, exist_ok=True)

# Active LSE Small-Cap / AIM tickers on Yahoo Finance
SAMPLE_TICKERS = [
    "ITM.L",   # ITM Power
    "GGP.L",   # Greatland Gold
    "PREM.L",  # Premier African Minerals
    "HE1.L",   # Helium One Global
    "KOD.L",   # Kodal Minerals
    "BZT.L",   # Bezant Resources
    "HUR.L",   # Hurricane Energy
    "CWR.L",   # Ceres Power
]

def fetch_and_save_ticker(ticker: str, start_date="2000-01-01"):
    print(f"Fetching data for {ticker}...")
    try:
        t = yf.Ticker(ticker)
        # Using .history() is more reliable than yf.download() on recent Python versions
        data = t.history(start=start_date, auto_adjust=False)
        
        if data is None or data.empty:
            print(f"  [-] Warning: No data or delisted for {ticker}. Skipping.")
            return

        # Flatten MultiIndex columns if Yahoo returns them
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = [col[0] for col in data.columns]
            
        data.reset_index(inplace=True)
        
        # Standardize date column naming
        if "Date" in data.columns:
            data["Date"] = pd.to_datetime(data["Date"]).dt.tz_localize(None)
            
        data['Ticker'] = ticker
        
        # Save to Parquet
        file_path = os.path.join(DATA_DIR, f"{ticker.replace('.', '_')}.parquet")
        data.to_parquet(file_path, engine="pyarrow", index=False)
        print(f"  [+] Saved {len(data)} rows to {file_path}")
        
    except Exception as e:
        print(f"  [-] Failed to fetch {ticker}: {e}")
    
    # Polite delay to prevent rate-limiting from Yahoo
    time.sleep(1)

if __name__ == "__main__":
    for ticker in SAMPLE_TICKERS:
        fetch_and_save_ticker(ticker)
    print("\nData collection complete. Files stored in ./data/raw_ohlcv/")