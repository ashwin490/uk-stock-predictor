import os
import glob
import joblib
import pandas as pd
import numpy as np
import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score

FEATURES_DIR = "./data/features"
MODELS_DIR = "./models"
os.makedirs(MODELS_DIR, exist_ok=True)

FEATURE_COLS = [
    'ret_1d', 'ret_3d', 'ret_5d', 'ret_20d',
    'turnover_gbp', 'turnover_ma20',
    'amihud_illiq', 'atr_14', 'atr_pct',
    'ratio_sma10_50', 'vol_surge_5_20',
    'roll_spread', 'est_spread_pct'
]
TARGET_COL = 'target_breakout_5d'

def train_ensemble_engine():
    files = glob.glob(os.path.join(FEATURES_DIR, "*_features.parquet"))
    if not files:
        raise FileNotFoundError("Run feature_engine.py first.")

    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    df = df[(df['turnover_gbp'] > 1000) & (df['Volume'] > 0)].sort_values("Date").reset_index(drop=True)

    # Time-based out-of-sample walk-forward split (80% train / 20% test)
    split_date = df['Date'].drop_duplicates().sort_values().iloc[int(len(df['Date'].unique()) * 0.80)]
    train_df = df[df['Date'] < split_date]
    test_df = df[df['Date'] >= split_date]

    X_train, y_train = train_df[FEATURE_COLS], train_df[TARGET_COL]
    X_test, y_test = test_df[FEATURE_COLS], test_df[TARGET_COL]

    # Model 1: LightGBM Gradient Booster
    pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    lgb_model = lgb.LGBMClassifier(
        n_estimators=300, learning_rate=0.03, max_depth=5,
        scale_pos_weight=pos_weight, random_state=42, verbose=-1
    )
    lgb_model.fit(X_train, y_train)

    # Model 2: CatBoost Classifier
    cb_model = CatBoostClassifier(
        iterations=300, learning_rate=0.03, depth=5,
        scale_pos_weight=pos_weight, random_seed=42, verbose=False
    )
    cb_model.fit(X_train, y_train)

    # Ensemble blending (50/50 weighting)
    lgb_probs = lgb_model.predict_proba(X_test)[:, 1]
    cb_probs = cb_model.predict_proba(X_test)[:, 1]
    ensemble_probs = 0.5 * lgb_probs + 0.5 * cb_probs

    auc = roc_auc_score(y_test, ensemble_probs)
    print(f"\n[+] Dual-Ensemble Out-of-Sample ROC-AUC: {auc:.4f}")

    # Export unified model artifact bundle
    bundle = {
        'lgb': lgb_model,
        'catboost': cb_model,
        'feature_cols': FEATURE_COLS
    }
    joblib.dump(bundle, os.path.join(MODELS_DIR, "ensemble_ranker.joblib"))
    print("[+] Model bundle exported to ./models/ensemble_ranker.joblib")

if __name__ == "__main__":
    train_ensemble_engine()