import os
import glob
import numpy as np
import pandas as pd

RAW_DATA_DIR = "./data/raw_ohlcv"
PROCESSED_DATA_DIR = "./data/features"
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

def compute_roll_spread(close: pd.Series) -> pd.Series:
    """Roll (1984) Effective Spread proxy based on serial covariance of price changes."""
    pct_changes = close.pct_change().fillna(0)
    cov = pct_changes.rolling(20).apply(lambda x: np.cov(x[:-1], x[1:])[0, 1] if len(x) > 2 else 0, raw=False)
    spread = np.where(cov < 0, 2.0 * np.sqrt(-cov), 0.0)
    return pd.Series(spread, index=close.index).clip(0.0, 0.25)

def compute_corwin_schultz_spread(df: pd.DataFrame) -> pd.Series:
    """Corwin & Schultz (2012) High-Low Spread Estimator."""
    high = np.where(df['High'].values <= 0, 1e-4, df['High'].values)
    low = np.where(df['Low'].values <= 0, 1e-4, df['Low'].values)

    log_hl = np.log(high / low) ** 2
    beta = log_hl[:-1] + log_hl[1:]

    h2 = np.maximum(high[:-1], high[1:])
    l2 = np.minimum(low[:-1], low[1:])
    gamma = np.log(h2 / l2) ** 2

    denom = 3.0 - 2.0 * np.sqrt(2.0)
    alpha = (np.sqrt(2.0 * beta) - np.sqrt(beta)) / denom - np.sqrt(gamma / denom)
    alpha = np.maximum(0, alpha)

    spread = 2.0 * (np.exp(alpha) - 1.0) / (1.0 + np.exp(alpha))
    spread_series = np.concatenate([[np.nan], spread])
    return pd.Series(spread_series, index=df.index).clip(0.0, 0.40)

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("Date").reset_index(drop=True).copy()

    # 1. Microstructure Return Horizons
    df['ret_1d'] = df['Close'].pct_change(1)
    df['ret_3d'] = df['Close'].pct_change(3)
    df['ret_5d'] = df['Close'].pct_change(5)
    df['ret_20d'] = df['Close'].pct_change(20)

    # 2. Pound Turnover (Pence to GBP)
    df['turnover_gbp'] = (df['Close'] * df['Volume']) / 100.0
    df['turnover_ma20'] = df['turnover_gbp'].rolling(20).mean()

    # 3. Amihud Illiquidity Ratio (|Return| / Volume in £10k units)
    df['amihud_illiq'] = (df['ret_1d'].abs() / ((df['turnover_gbp'] / 10000.0) + 1e-5)).rolling(10).mean()

    # 4. Volatility & Average True Range (ATR)
    hl = df['High'] - df['Low']
    hc = (df['High'] - df['Close'].shift(1)).abs()
    lc = (df['Low'] - df['Close'].shift(1)).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    df['atr_14'] = tr.rolling(14).mean()
    df['atr_pct'] = df['atr_14'] / df['Close']

    # 5. Trend & Momentum Dispersion
    df['sma_10'] = df['Close'].rolling(10).mean()
    df['sma_50'] = df['Close'].rolling(50).mean()
    df['ratio_sma10_50'] = df['sma_10'] / df['sma_50']
    df['vol_surge_5_20'] = df['Volume'].rolling(5).mean() / (df['Volume'].rolling(20).mean() + 1e-6)

    # 6. Dealer Spreads
    df['roll_spread'] = compute_roll_spread(df['Close'])
    df['est_spread_pct'] = compute_corwin_schultz_spread(df)

    # 7. Asymmetric Alpha Target: Forward 5-day return must exceed spread + 5% alpha hurdle
    fwd_ret_5d = df['Close'].shift(-5) / df['Close'] - 1.0
    hurdle = df['est_spread_pct'] + 0.05
    df['target_breakout_5d'] = (fwd_ret_5d > hurdle).astype(int)

    return df.dropna().reset_index(drop=True)

def process_all_files():
    files = glob.glob(os.path.join(RAW_DATA_DIR, "*.parquet"))
    for file in files:
        ticker = os.path.basename(file).replace(".parquet", "")
        df = pd.read_parquet(file)
        if len(df) >= 60:
            feat_df = engineer_features(df)
            feat_df.to_parquet(os.path.join(PROCESSED_DATA_DIR, f"{ticker}_features.parquet"), index=False)
    print(f"[+] Complete microstructure features generated in {PROCESSED_DATA_DIR}/")

if __name__ == "__main__":
    process_all_files()