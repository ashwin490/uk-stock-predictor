import os
import joblib
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
from feature_engine import engineer_features

MODEL_PATH = "./models/lse_ranker.joblib"

# Active LSE Small-Cap / AIM universe
WATCHLIST = [
    "ITM.L",   # ITM Power
    "GGP.L",   # Greatland Gold
    "PREM.L",  # Premier African Minerals
    "HE1.L",   # Helium One Global
    "KOD.L",   # Kodal Minerals
    "BZT.L",   # Bezant Resources
    "HUR.L",   # Hurricane Energy
    "CWR.L",   # Ceres Power
]

FEATURE_COLS = [
    'ret_1d', 
    'ret_5d', 
    'ret_20d',
    'turnover_gbp', 
    'turnover_ma20',
    'atr_14', 
    'atr_pct',
    'ratio_sma10_50',
    'vol_surge_5_20',
    'est_spread_pct'
]

def generate_daily_picks(top_n=5):
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("Model file not found. Please run train_model.py first.")

    print(f"Loading trained quantitative ranker from {MODEL_PATH}...")
    model = joblib.load(MODEL_PATH)

    candidates = []

    print(f"Fetching latest market data for {len(WATCHLIST)} LSE tickers...")
    for ticker in WATCHLIST:
        try:
            t = yf.Ticker(ticker)
            # Pull 120 days to ensure 50-day SMA and rolling windows calculate correctly
            df = t.history(period="120d", auto_adjust=False)
            
            if df is None or len(df) < 55:
                continue

            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [col[0] for col in df.columns]

            df.reset_index(inplace=True)
            if "Date" in df.columns:
                df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None)

            # Compute latest features
            feats = engineer_features(df)
            if feats.empty:
                continue

            # Take the latest closed trading day
            latest = feats.iloc[-1:].copy()
            latest_features = latest[FEATURE_COLS]

            # Predict probability of breakout
            prob = model.predict_proba(latest_features)[:, 1][0]
            
            close_price = latest['Close'].values[0]
            turnover_gbp = latest['turnover_gbp'].values[0]
            spread_est = latest['est_spread_pct'].values[0]
            last_date = latest['Date'].values[0]

            candidates.append({
                'Ticker': ticker,
                'Date': pd.to_datetime(last_date).strftime('%Y-%m-%d'),
                'Close (p)': round(close_price, 2),
                'Est Spread': f"{spread_est * 100:.1f}%",
                'Turnover (£)': f"£{turnover_gbp:,.0f}",
                'Breakout Score': prob
            })

        except Exception as e:
            print(f"[-] Error processing {ticker}: {e}")

    if not candidates:
        print("No candidates met data requirements.")
        return

    results_df = pd.DataFrame(candidates)
    
    # Sort candidates by the highest model confidence score
    ranked_df = results_df.sort_values(by="Breakout Score", ascending=False).reset_index(drop=True)
    
    # Top N Picks
    top_picks = ranked_df.head(top_n)

    print("\n" + "="*70)
    print(f" LSE AIM QUANT PREDICTOR: TOP {top_n} ASYMMETRIC PICKS ")
    print("="*70)
    
    # Format breakout score as percentage
    top_picks_display = top_picks.copy()
    top_picks_display['Breakout Score'] = top_picks_display['Breakout Score'].apply(lambda x: f"{x * 100:.1f}%")
    
    print(top_picks_display.to_string(index=False))
    print("="*70)
    print("Signal Note: 'Breakout Score' represents the model's assessed probability")
    print("that forward return will beat the estimated bid-ask spread hurdle over 5 days.")
    print("="*70 + "\n")

    # Save to CSV for reporting / app dashboard consumption
    os.makedirs("./output", exist_ok=True)
    today_str = datetime.now().strftime("%Y-%m-%d")
    out_csv = f"./output/lse_top5_{today_str}.csv"
    top_picks.to_csv(out_csv, index=False)
    print(f"[+] Daily signals written to {out_csv}")

if __name__ == "__main__":
    generate_daily_picks(top_n=5)