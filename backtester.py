import os
import glob
import json
import joblib
import pandas as pd
import numpy as np

MODELS_DIR = "./models"
FEATURES_DIR = "./data/features"
OUTPUT_DIR = "./output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def run_backtest_simulation(
    initial_capital=500.0,
    max_positions=5,
    max_alloc_per_trade=0.18,
    min_model_score=0.52,
    hold_horizon_days=5
):
    model_path = os.path.join(MODELS_DIR, "ensemble_ranker.joblib")
    if not os.path.exists(model_path):
        raise FileNotFoundError("Run train_model.py first to generate ensemble_ranker.joblib.")

    bundle = joblib.load(model_path)
    lgb_model = bundle['lgb']
    cb_model = bundle['catboost']
    feature_cols = bundle['feature_cols']

    feature_files = glob.glob(os.path.join(FEATURES_DIR, "*_features.parquet"))
    if not feature_files:
        raise FileNotFoundError("No engineered feature files found in ./data/features/")

    dfs = []
    for f in feature_files:
        ticker = os.path.basename(f).replace("_features.parquet", "").replace("_", ".")
        df = pd.read_parquet(f)
        df['Ticker'] = ticker
        dfs.append(df)

    all_data = pd.concat(dfs, ignore_index=True)
    all_data = all_data.sort_values("Date").reset_index(drop=True)

    # Use up to 500 past market trading days
    unique_dates = all_data['Date'].drop_duplicates().sort_values().values
    sim_dates = unique_dates[-500:] if len(unique_dates) > 500 else unique_dates

    # Multi-model blend
    p1 = lgb_model.predict_proba(all_data[feature_cols])[:, 1]
    p2 = cb_model.predict_proba(all_data[feature_cols])[:, 1]
    all_data['model_score'] = 0.5 * p1 + 0.5 * p2

    cash = float(initial_capital)
    portfolio_history = []
    trades = []
    active_positions = []

    for current_date in sim_dates:
        day_slice = all_data[all_data['Date'] == current_date]

        # 1. Manage active trades
        surviving_positions = []
        for pos in active_positions:
            tkr = pos['ticker']
            match = day_slice[day_slice['Ticker'] == tkr]

            if match.empty:
                pos['days_held'] += 1
                if pos['days_held'] >= hold_horizon_days:
                    cash += pos['shares'] * (pos['current_price'] / 100.0)
                    trades.append({
                        'Ticker': tkr, 'Date': current_date, 'Return': 0.0,
                        'Outcome': 'Timeout', 'PnL_GBP': 0.0
                    })
                else:
                    surviving_positions.append(pos)
                continue

            current_p = float(match['Close'].values[0])
            high_p = float(match['High'].values[0])
            low_p = float(match['Low'].values[0])
            pos['days_held'] += 1
            pos['current_price'] = current_p

            # Target reached
            if high_p >= pos['target_price']:
                realized_gbp = pos['shares'] * (pos['target_price'] / 100.0)
                pnl = realized_gbp - pos['cost_gbp']
                ret_pct = (pos['target_price'] - pos['entry_price']) / pos['entry_price']
                cash += realized_gbp
                trades.append({
                    'Ticker': tkr, 'Date': current_date, 'Return': ret_pct,
                    'Outcome': 'Win', 'PnL_GBP': pnl
                })
            # Stop loss reached
            elif low_p <= pos['stop_loss']:
                realized_gbp = pos['shares'] * (pos['stop_loss'] / 100.0)
                pnl = realized_gbp - pos['cost_gbp']
                ret_pct = (pos['stop_loss'] - pos['entry_price']) / pos['entry_price']
                cash += realized_gbp
                trades.append({
                    'Ticker': tkr, 'Date': current_date, 'Return': ret_pct,
                    'Outcome': 'Loss', 'PnL_GBP': pnl
                })
            # Time limit reached
            elif pos['days_held'] >= hold_horizon_days:
                realized_gbp = pos['shares'] * (current_p / 100.0)
                pnl = realized_gbp - pos['cost_gbp']
                ret_pct = (current_p - pos['entry_price']) / pos['entry_price']
                cash += realized_gbp
                trades.append({
                    'Ticker': tkr, 'Date': current_date, 'Return': ret_pct,
                    'Outcome': 'Win' if pnl > 0 else 'Loss', 'PnL_GBP': pnl
                })
            else:
                surviving_positions.append(pos)

        active_positions = surviving_positions

        # 2. Open new candidate positions
        available_slots = max_positions - len(active_positions)
        if available_slots > 0 and cash >= 30.0:
            qualifying = day_slice[
                (day_slice['turnover_gbp'] > 5000) & 
                (day_slice['est_spread_pct'] < 0.08) &
                (day_slice['model_score'] >= min_model_score)
            ].sort_values('model_score', ascending=False)

            current_active_tickers = [p['ticker'] for p in active_positions]
            for _, row in qualifying.iterrows():
                if available_slots <= 0:
                    break
                tkr = row['Ticker']
                if tkr in current_active_tickers:
                    continue

                entry_p = float(row['Close'])
                atr = float(row['atr_14'])
                spread = float(row['est_spread_pct'])

                stop_buf = max(0.04, min(0.08, (atr / entry_p) * 1.2))
                stop_p = entry_p * (1.0 - stop_buf)
                target_p = entry_p * (1.0 + max(0.06, (atr / entry_p) * 1.8 + spread))

                alloc_gbp = min(cash * max_alloc_per_trade, initial_capital * 0.20)
                if alloc_gbp < 25.0:
                    break

                shares = int((alloc_gbp * 100.0) / entry_p)
                if shares <= 0:
                    continue

                actual_cost = (shares * entry_p) / 100.0
                cash -= actual_cost

                active_positions.append({
                    'ticker': tkr,
                    'entry_price': entry_p,
                    'current_price': entry_p,
                    'target_price': target_p,
                    'stop_loss': stop_p,
                    'shares': shares,
                    'cost_gbp': actual_cost,
                    'days_held': 0
                })
                current_active_tickers.append(tkr)
                available_slots -= 1

        holdings_val = sum((p['shares'] * p['current_price'] / 100.0) for p in active_positions)
        total_equity = cash + holdings_val

        portfolio_history.append({
            'Date': pd.to_datetime(current_date).strftime('%Y-%m-%d'),
            'Total Equity (£)': round(total_equity, 2),
            'Cash (£)': round(cash, 2),
            'Invested (£)': round(holdings_val, 2),
            'Active Count': len(active_positions)
        })

    curve_df = pd.DataFrame(portfolio_history)
    trades_df = pd.DataFrame(trades)

    curve_df.to_csv(os.path.join(OUTPUT_DIR, "equity_curve.csv"), index=False)
    trades_df.to_csv(os.path.join(OUTPUT_DIR, "backtest_trades.csv"), index=False)

    total_trades = len(trades_df)
    wins = len(trades_df[trades_df['Outcome'] == 'Win'])
    win_rate = (wins / total_trades * 100) if total_trades > 0 else 0.0

    gross_profit = trades_df[trades_df['PnL_GBP'] > 0]['PnL_GBP'].sum()
    gross_loss = abs(trades_df[trades_df['PnL_GBP'] < 0]['PnL_GBP'].sum())
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else gross_profit

    final_equity = curve_df['Total Equity (£)'].iloc[-1]
    net_return = ((final_equity - initial_capital) / initial_capital) * 100

    metrics = {
        'initial_capital': initial_capital,
        'final_equity': round(final_equity, 2),
        'net_return_pct': round(net_return, 1),
        'win_rate_pct': round(win_rate, 1),
        'total_trades': total_trades,
        'profit_factor': round(profit_factor, 2)
    }

    with open(os.path.join(OUTPUT_DIR, "backtest_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    return metrics, curve_df, trades_df

if __name__ == "__main__":
    run_backtest_simulation()