"""ALPHA-LSE feature engineering (UK/LSE prices expressed in pence).

Input OHLC prices must already be verified and normalised to GBp by app.py.
The same feature names and definitions are retained for the existing ML model.

The target_breakout_5d column is a *training label*, NOT a live feature:
its last five rows are NaN because their future outcomes are not known yet.
Do not include it in model inputs; when training, filter rows with known labels.
"""

import glob
import os
from pathlib import Path

import numpy as np
import pandas as pd

RAW_DATA_DIR = "./data/raw_ohlcv"
PROCESSED_DATA_DIR = "./data/features"

# The output columns are deliberately stable: saved models depend on their names.
FEATURE_COLUMNS = (
    "ret_1d", "ret_3d", "ret_5d", "ret_20d",
    "turnover_gbp", "turnover_ma20", "amihud_illiq",
    "atr_14", "atr_pct", "sma_10", "sma_50",
    "ratio_sma10_50", "vol_surge_5_20",
    "roll_spread", "est_spread_pct",
)


def compute_roll_spread(close: pd.Series) -> pd.Series:
    """Roll (1984) spread proxy using adjacent returns in 20-day windows.

    IMPORTANT: np.where(condition, sqrt(-cov), 0) is unsafe because NumPy
    evaluates the sqrt for *all* covariance values, including positive ones.
    Here sqrt sees only non-negative inputs.
    """
    close = pd.to_numeric(close, errors="coerce")
    returns = close.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    def adjacent_covariance(window):
        if not np.isfinite(window).all():
            return np.nan
        return float(np.cov(window[:-1], window[1:], ddof=1)[0, 1])

    covariance = returns.rolling(20, min_periods=20).apply(adjacent_covariance, raw=True)
    safe_magnitude = (-covariance).clip(lower=0.0).fillna(0.0)
    spread = 2.0 * np.sqrt(safe_magnitude)
    return pd.Series(spread, index=close.index).clip(lower=0.0, upper=0.25)


def compute_corwin_schultz_spread(df: pd.DataFrame) -> pd.Series:
    """Corwin & Schultz (2012) high-low spread estimator, numerically safe.

    Invalid prices are represented by NaN rather than manufactured positive
    prices. The caller rejects invalid OHLC bars before creating features.
    """
    high = pd.to_numeric(df["High"], errors="coerce").to_numpy(dtype=float)
    low = pd.to_numeric(df["Low"], errors="coerce").to_numpy(dtype=float)
    n = len(df)
    out = np.full(n, np.nan, dtype=float)
    if n < 2:
        return pd.Series(out, index=df.index)

    valid = np.isfinite(high) & np.isfinite(low) & (low > 0) & (high >= low)
    pair_ok = valid[:-1] & valid[1:]
    pair_idxs = np.flatnonzero(pair_ok)
    if pair_idxs.size == 0:
        return pd.Series(out, index=df.index)

    # Do the logarithms only on verified, positive high/low pairs.
    h0, h1 = high[pair_idxs], high[pair_idxs + 1]
    l0, l1 = low[pair_idxs], low[pair_idxs + 1]
    beta = np.square(np.log(h0 / l0)) + np.square(np.log(h1 / l1))
    gamma = np.square(np.log(np.maximum(h0, h1) / np.minimum(l0, l1)))
    denominator = 3.0 - 2.0 * np.sqrt(2.0)
    alpha = ((np.sqrt(np.maximum(0.0, 2.0 * beta)) - np.sqrt(np.maximum(0.0, beta))) /
             denominator - np.sqrt(np.maximum(0.0, gamma / denominator)))
    alpha = np.maximum(alpha, 0.0)

    # 2*tanh(alpha/2) is equal to 2*(exp(alpha)-1)/(exp(alpha)+1),
    # but cannot overflow for unusually large alpha values.
    out[pair_idxs + 1] = 2.0 * np.tanh(alpha / 2.0)
    return pd.Series(out, index=df.index).clip(lower=0.0, upper=0.40)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build ML features from *verified pence-denominated* daily OHLCV bars.

    Latest five feature rows remain available for live predictions even though
    their forward 5-day training labels are intentionally unknown (NaN).
    """
    required = ("Date", "Open", "High", "Low", "Close", "Volume")
    if "Date" not in df.columns and df.index.name in ("Date", "Datetime"):
        df = df.reset_index().rename(columns={"Datetime": "Date"})
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing OHLCV columns: {', '.join(missing)}")

    df = df.sort_values("Date").drop_duplicates(subset=["Date"], keep="last").reset_index(drop=True).copy()
    for col in ("Open", "High", "Low", "Close", "Volume"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    prices = df[["Open", "High", "Low", "Close"]]
    valid = prices.notna().all(axis=1) & (prices.min(axis=1) > 0)
    valid &= df["High"] >= prices[["Open", "Low", "Close"]].max(axis=1)
    valid &= df["Low"] <= prices[["Open", "High", "Close"]].min(axis=1)
    valid &= df["Volume"].notna() & (df["Volume"] >= 0)
    df = df.loc[valid].reset_index(drop=True)
    if len(df) < 50:
        return pd.DataFrame(columns=list(df.columns) + list(FEATURE_COLUMNS) + ["target_breakout_5d"])

    # 1. Return horizons (same feature names and conventions as the original).
    for days in (1, 3, 5, 20):
        df[f"ret_{days}d"] = df["Close"].pct_change(periods=days, fill_method=None)

    # 2. Turnover: app.py normalises verified LSE prices to pence (GBp).
    df["turnover_gbp"] = df["Close"] * df["Volume"] / 100.0
    df["turnover_ma20"] = df["turnover_gbp"].rolling(20).mean()

    # 3. Amihud liquidity proxy.
    df["amihud_illiq"] = (
        df["ret_1d"].abs() / (df["turnover_gbp"] / 10000.0 + 1e-5)
    ).rolling(10).mean()

    # 4. Average True Range.
    hl = df["High"] - df["Low"]
    hc = (df["High"] - df["Close"].shift(1)).abs()
    lc = (df["Low"] - df["Close"].shift(1)).abs()
    true_range = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    df["atr_14"] = true_range.rolling(14).mean()
    df["atr_pct"] = df["atr_14"] / df["Close"]

    # 5. Moving average / volume ratios.
    df["sma_10"] = df["Close"].rolling(10).mean()
    df["sma_50"] = df["Close"].rolling(50).mean()
    df["ratio_sma10_50"] = df["sma_10"] / df["sma_50"]
    df["vol_surge_5_20"] = df["Volume"].rolling(5).mean() / (df["Volume"].rolling(20).mean() + 1e-6)

    # 6. Effective spread proxies.
    df["roll_spread"] = compute_roll_spread(df["Close"])
    df["est_spread_pct"] = compute_corwin_schultz_spread(df)

    # 7. Training label. Do not invent a negative outcome when the future
    # five trading days have not occurred; this is label leakage/noise.
    future_close = df["Close"].shift(-5)
    future_return = future_close / df["Close"] - 1.0
    above_hurdle = (future_return > df["est_spread_pct"] + 0.05).astype(float)
    df["target_breakout_5d"] = above_hurdle.where(future_close.notna(), np.nan)

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    # CRITICAL: do not drop rows based on the future label. The app needs the
    # latest completed bar for inference, even though its future label is NaN.
    return df.dropna(subset=list(FEATURE_COLUMNS)).reset_index(drop=True)


def process_all_files():
    Path(PROCESSED_DATA_DIR).mkdir(parents=True, exist_ok=True)
    written, skipped = 0, 0
    for file_path in sorted(glob.glob(os.path.join(RAW_DATA_DIR, "*.parquet"))):
        try:
            raw_df = pd.read_parquet(file_path)
            if len(raw_df) < 60:
                skipped += 1
                continue
            features = engineer_features(raw_df)
            if features.empty:
                skipped += 1
                continue
            ticker = Path(file_path).stem
            features.to_parquet(os.path.join(PROCESSED_DATA_DIR, f"{ticker}_features.parquet"), index=False)
            written += 1
        except Exception as exc:
            skipped += 1
            print(f"[WARN] Could not process {Path(file_path).name}: {exc}")
    print(f"[+] Finished: {written} feature files written; {skipped} skipped.")


if __name__ == "__main__":
    process_all_files()
