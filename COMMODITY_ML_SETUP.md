# ALPHA-LSE — Final commodity ML integration (24 November 2026)

This version introduces a **separate historical machine-learning system** for the five existing London-listed ETCs, without changing the UK-share model. All orders are simulated.

## Critical: one-time setup for Streamlit Community Cloud

Streamlit Community Cloud's local filesystem does **not** reliably persist changed model files after restarts. For automatic model versions to survive restarts, create private Supabase Storage for the commodity model:

1. In **your existing Supabase project**, open **Storage → New bucket**.
2. Name the bucket `alpha-lse-models`. Keep **Public bucket OFF**. Do not make it publicly readable or writable.
3. In Supabase **Project Settings → API Keys**, find the server-side `service_role` key for your project. It has extensive privileges; never put it in code, GitHub, screenshots, or chat.
4. Open **Streamlit Community Cloud → your app → Settings → Secrets** and **add** these two entries to your existing secrets, preserving your current authentication and journal secrets:

   ```toml
   ALPHA_COMMODITY_ML_BUCKET = "alpha-lse-models"
   SUPABASE_MODEL_SERVICE_KEY = "YOUR_SUPABASE_SERVICE_ROLE_KEY"
   ```

   The existing entries `SUPABASE_URL`, `SUPABASE_KEY`, `ALPHA_STORAGE_MODE = "cloud"`, and your authentication credentials must remain present. Keep the `service_role` key only in server-side Streamlit Secrets. It bypasses row-level security, so restrict who can edit/read your app's secrets.
5. Deploy **all project code files together**, reboot the app, and open it in your browser. The historical ETC training check runs automatically when the app is opened, then at most once per day per active session. Its model state is stored as `commodity_ml_champion_v1.json` in that **private** bucket. If you do not configure the bucket/key, the original rule-based paper trading remains available and the app clearly displays that persistent ML is not configured.

> **Do not create a public bucket or add anonymous write policies.** This code uses the dedicated server-side key solely for the private model file. No broker account or real-money order integration exists.

## What the learner actually predicts

- Data: up to **five years of verified GBp daily OHLCV** for `SGLN.L`, `SSLN.L`, `BRNG.L`, `CRUP.L`, `NGSP.L` (as available from the existing market-data provider).
- Label: whether the ETC close is **more than 1.5% higher five trading sessions later, before costs**. This is **not** the chance a target order fills, a stop is avoided, or the trade makes money after broker costs.
- Features: only information available through the previous **completed** daily bars — momentum, moving averages, realised volatility, ATR, RSI, volume, turnover and ETC identity.
- Validation: chronological held-out later dates with a five-session embargo, compared with a constant-rate baseline. A challenger that fails the checks is **not** promoted.
- Initial model: requires at least three usable products, 450 training records and 85 genuinely held-out test records across at least 22 dates, with both possible outcomes present. Some data providers/products may lack sufficient history.
- Recheck: when at least 22 **new** completed labelled trading dates are available, it can test a challenger against its incumbent using later, out-of-sample observations. A model is only promoted if it improves both Brier error and log loss by material margins. The next champion is saved **and read back** before use.
- Later performance: if a champion fails its recent independent baseline comparison or becomes unverified for 60 days, new ML-filtered buys are paused. Existing positions continue to be audited.
- Actual finished ETC paper trades **also** inform the separate conservative per-product position-size feedback already present in the app. They are not falsely treated as independent training examples for the price-movement classifier.

**ML has no guaranteed edge.** If historical data are unavailable, or the challenger is no better than a constant-rate forecast, there may be **no promoted ML model**. The app then states **rule-only fallback** rather than inventing confidence estimates.

## How ML interacts with buying

1. Five existing trend conditions, verified GBp quote and today's eligible entry window must still pass.
2. Existing stop, ATR, turnover, commission, estimated net return, position sizing, virtual cash and portfolio risk checks must pass.
3. If a validated commodity ML champion exists and the Settings ML filter is ON (default), its estimated probability of the specified five-day event must meet a guarded threshold. The ML model can **reject** candidates; it can never bypass the rules.
4. When no champion has passed, the pre-existing rule-based simulator can continue making qualifying paper trades, clearly labelled as **rule-only fallback**. To avoid those fallback trades, turn OFF "Automatically paper-buy qualifying commodity ETCs" until the learner is ready.
5. The original buy price and target remain immutable after recording. Prices in WAIT remain provisional; actual trade results and model version are preserved in the journal.

The home page continues to display **open commodities, open P/L, realised P/L, wins/losses**. The commodity cards add the **latest available price** and any defensible ML forecast. The trade-history table adds **Strategy, Model version, and recorded five-day ML estimate**, including old and new trades.

## Deployment

1. Back up your existing local project and check that your cloud journal is intact.
2. Extract the complete ZIP into your local repo (`lse_penny_quant`), overwriting matching **code** files. **Do not overwrite/delete** existing model files, local DuckDB data, backups, or `.streamlit/secrets.toml`.
3. In PowerShell (from your project folder), run:

   ```powershell
   python -m pip install -r requirements.txt
   python -m pytest -q tests
   git status
   git add app.py commodity_ml.py commodity_dashboard.py commodity_signals.py commodity_etps.py auto_refresh_schedule.py tests README.md COMMODITY_ML_SETUP.md .github/workflows/regression.yml
   git diff --cached --stat
   git commit -m "Add separately validated and persistent commodity ML"
   git push origin main
   ```

   If `pytest` is not installed, run `python -m pip install pytest` first. Never commit `.streamlit/secrets.toml`, `.env`, database backups, or your private model JSON.
4. Add the two secrets above, then **reboot** on Streamlit Cloud. Verify in **Commodities → Commodity ML learning** that the storage check ran and historical evidence appears. An initial model is **not guaranteed to be promoted**; that is an intentional safety requirement.
5. Verify normal `Auto-refresh ON` during market hours, visible commodity cards, and your existing trade journal. Refresh and learning require the browser/Streamlit session to stay active: this does not create an always-on cloud scheduler.

## Troubleshooting

- **"Private ML bucket or dedicated server-side model key not configured"**: add the two Streamlit Secrets and create the private Storage bucket as above.
- **"Remote model persistence failed"**: ensure the bucket name is correct, it is in the same Supabase project as the server key, and upload/download is permitted. The app deliberately does **not** activate a failed upgrade.
- **"Insufficient historical data"**: Yahoo may not have enough verified GBp daily bars; there is no fabricated replacement or model.
- **"Challenger rejected"**: its later-data performance did not beat the benchmark. This is a valid outcome, not a crash.
- **"Model validation older than 60 days"**: the app has not verified its edge recently. It pauses new ML-filtered buys rather than advertising unjustified predictive accuracy.
- **Stock AI remains the original champion**: this new learner is **separate** and cannot modify the share ML model.

## Validation performed on the delivered code

- Python syntax compilation of all Python files.
- 105+ offline regression tests including synthetic-market causal-label, model-fit, chronological split/embargo, fallback, card escaping, version persistence, cloud failure and original app safety tests.
- No live yfinance/Supabase/Streamlit Cloud integration could be exercised in the offline build environment. **No real-data-trained model is bundled or claimed.** After deployment the app trains against genuine provider history if available and only promotes if evidence passes the gates.

**Automatic model upgrades are not automatic software/code upgrades.** New strategy definitions, dependency upgrades, changed regulatory requirements, broken providers and monitoring still need human review.
