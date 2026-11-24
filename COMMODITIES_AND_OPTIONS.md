> **Historical note (23 November 2026):** This document describes the original rule-only ETC release. The 24 November update adds independent commodity ML filtering and guarded model upgrades. Follow **COMMODITY_ML_SETUP.md** for current deployment and behaviour. The five original trend rules and safety checks below still apply.

# 23 November 2026 — Commodity predictions and US options retirement

## What you will see

The old **US Options** tab has been replaced with **🪙 Commodities: Buy / Sell / Stop**.
It shows the five verified sterling-quoted London-listed ETCs:

| Commodity exposure | LSE ticker | Strategy |
| --- | --- | --- |
| Gold | `SGLN.L` | Rule-based candidate/wait |
| Silver | `SSLN.L` | Rule-based candidate/wait |
| Brent crude oil | `BRNG.L` | Rule-based candidate/wait |
| WTI crude oil | `CRUP.L` | Rule-based candidate/wait |
| Natural gas | `NGSP.L` | Rule-based candidate/wait |

Prices displayed are **pence per ETC unit**: **Buy at**, **Sell target**, and **Stop-loss**.
Quantity and paper amount are shown when a candidate passes all rules and has available virtual cash.
The program shows *Wait* and an explanation when data, market hours, volatility, costs, trend or risk prevent an entry.

The decision rules inspect **completed daily OHLCV bars** and five observable trend conditions (score 4/5 required); this is deliberately NOT the stock ML model and is NOT a validated win probability. A 14-day ATR sets an indicative target at `entry + 2.4 × ATR` and initial stop at `entry − 1.5 × ATR`. Estimated spread and commission must still leave at least the normal minimum net-return hurdle. Each instrument must have at least 65 complete daily bars and sufficient trading turnover. The app refreshes the price before automatically recording a simulated order.

### Automatic simulated buys and sells

- **Settings → Automatically paper-buy qualifying commodity ETCs** is ON by default while the local or cloud Streamlit app is active. You can switch it OFF. This setting **does not connect to a broker**.
- During the UK market entry window, a qualifying ETC can be written to the regular paper trade journal with `asset_type = 'ETC'`, fully respecting existing cash, initial stop risk, sector exposure and duplicate-order rules. The scanner rechecks each ticker on the app's normal refresh loop. Manual Refresh in the commodities tab only checks signals; it does not submit a buy.
- Existing positions are monitored by the same daily + available one-minute paper audit as shares. When validated post-entry prices touch the target or stop, it records the result. If no reliable minute history is available, the app may miss a brief touch. **A target touched in historical OHLC data is not proof of an executable fill.** Stop gaps and illiquidity can lead to larger realised losses.
- The commodities performance ledger is **separate from share AI learning**. A rule-based ETC signal is not evidence of ML accuracy, and old US option results are not included in the new scoreboard.
- No automatic scan runs while the Streamlit browser session is closed, sleeping or disconnected; no background worker is included.

### Remove *previous* US options predictions

No **new** US call options are proposed, simulated, or audited in this version. The old option journal is retained only until the signed-in user deliberately deletes it.

1. Open **Settings → Remove old US options history**.
2. Click **Prepare options-only archive**, then **Download options archive ZIP**. Save it somewhere safe.
3. Type `DELETE OPTIONS`, then click **Permanently clear option journal**.
4. The code deletes only the local `daily_options_journal` and, when connected to Supabase, the remote `options_journal`. It does **not** delete `trade_journal` or the Supabase `predictions` table.
5. If connected to Supabase before, use the *same* Supabase credentials; without them, old online options may remain. On a remote failure, keep the downloaded backup and check the results before retrying.

Deletion is intentionally **not automatic on application startup**. That makes it harder to erase historic financial-research data accidentally or delete it before taking a backup.

### Install / upgrade

Back up the existing database, secrets (privately) and model files **before** replacing code.
Copy **together** `app.py`, `commodity_signals.py`, `option_retirement.py`, `commodity_etps.py`, `intraday_exit.py`, `feature_engine.py`, `requirements.txt`, and `tests/`. Do **not** delete `rns_scraper.py`, the model directory, your DuckDB journal or your Supabase Secrets. Run `pip install -r requirements.txt` in the deployment environment, then restart the app and verify that past share trades are still visible.

For a cloud deployment, keep `ALPHA_STORAGE_MODE = "cloud"` and valid Supabase credentials in **Streamlit Secrets**. For a laptop deployment, use the existing `start_local.bat`. Do not put `.streamlit/secrets.toml`, API keys or any database backup in your public GitHub repository.

**Important limitations:** offline tests do not validate profitability. Vendor feeds may be delayed and may change; ETCs are investment securities, not direct physical commodity purchases. Natural gas and oil ETC prices can differ materially from spot commodity movements because of futures-contract roll and currency exposure. The application requires ongoing dependency, market-data and security maintenance.

### Offline tests and live-deployment status

The updated program currently passes **65 offline regression tests**, including a target touched minutes after purchase, separate ETC scoring, safe order qualification and archive/deletion safeguards. Full integration testing with the installed Yahoo Finance feed, Streamlit Cloud instance, actual saved model and Supabase schema **has not been performed** in this environment. The real observed market outcomes will need monitoring before its predictions can be called accurate.
