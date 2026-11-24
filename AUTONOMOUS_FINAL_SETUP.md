# ALPHA-LSE — autonomous commodity research: FINAL INSTALL GUIDE

**Release:** 24 November 2026. This system is **paper-only**. There are no broker transactions or trade-order API calls in the scheduled worker.

## What this update actually automates

- GitHub Actions runs an independent hourly UK-market research scan. It no longer needs the Streamlit dashboard open to scan five commodity ETC price feeds.
- After the official London exchange close, the worker trains/tests the commodity model **only on complete historical sessions**, then writes a dated, permanent five-session forecast for each ETC when data is available.
- Five later **complete trading sessions** later, the worker records the actual price result next to that *previously saved* prediction and updates proper Brier-score comparisons against the baseline.
- Compares pooled versus five specialised commodity models using a separate time-purged selection window. No architecture is promoted merely for generating more positive predictions.
- Rejects underperforming models, can restore a previously validated champion **only after new independent evidence**, and pauses ML-authorised new entries if subsequent validated-model probability scores deteriorate.
- Keeps immutable forecasts, separate actual outcomes, model-trial archives, model snapshots, daily health records, optional alerts, and weekly compressed backups of the existing Supabase `predictions` trade journal.
- The Streamlit dashboard can read and show autonomous-worker health, forecasts and genuine forward results.

**Important boundary:** The background worker deliberately **does not make paper purchases or close paper positions**. Your existing app's simulated purchases and intraday audit still need the Streamlit session running. This avoids running a second, incompatible order journal that could create duplicate positions and corrupt the existing £50,000 virtual portfolio. If you want unattended paper orders too, that requires a separate transaction-safe database migration and verified test execution. Never assume this ZIP solves unattended paper orders.

## One-time installation — your existing `ashwin490/uk-stock-predictor`

1. Back up your local project directory and keep your existing **Streamlit Secrets**, Supabase bucket and **all** existing paper journal records. Do **not** delete `commodity_ml_champion_v1.json`.
2. Extract the new ZIP. Copy its files into the root of your existing local `lse_penny_quant` repository, including the `.github/workflows/commodity_autonomous.yml` file. Keep your own non-included trained stock model (`models/ensemble_ranker.joblib`), local DuckDB databases, `.streamlit/secrets.toml`, `.env`, and historical backups unchanged.
3. In GitHub, open **your repository → Settings → Secrets and variables → Actions → Secrets → New repository secret**. Create these two repository secrets (values are copied **privately** from the matching working Streamlit Secrets):
   - `SUPABASE_URL`: your existing project URL.
   - `SUPABASE_MODEL_SERVICE_KEY`: your existing Supabase private `service_role` key. **Never use `SUPABASE_KEY` if it is the public anon/publishable key.**
4. In the same GitHub Actions **Variables** section, create `ALPHA_COMMODITY_ML_BUCKET` with value `alpha-lse-models`, or let the workflow use that built-in default.
5. Optional: GitHub Actions secret `ALPHA_ALERT_WEBHOOK_URL` may point to a trusted Slack incoming-webhook URL or Discord webhook. Only short error summaries are sent. Without it, monitor GitHub Actions failed-run notifications manually.
6. In **Streamlit Community Cloud → Your app → Settings → Secrets**, KEEP every original line and add only:
   ```toml
   ALPHA_BACKGROUND_WORKER_ENABLED = true
   ```
   You already configured the private model bucket and `SUPABASE_MODEL_SERVICE_KEY` in Streamlit Secrets. Leave those intact. This new flag tells Streamlit to **read** model state and gives GitHub Actions sole responsibility for retraining. Only set it after GitHub Actions has passed a connection test.
7. In the project's VS Code PowerShell terminal:
   ```powershell
   git status --short
   git add app.py commodity_ml.py commodity_worker.py requirements-worker.txt .github/workflows/commodity_autonomous.yml .github/workflows/regression.yml tests AUTONOMOUS_FINAL_SETUP.md README.md
   git diff --cached --stat
   git diff --cached --name-only
   git commit -m "Add autonomous, validated commodity research and alerts"
   git push origin main
   ```
   You can commit any other intended accompanying files from the ZIP. **Do not stage** `.env`, `.streamlit/secrets.toml`, cloud keys, `.joblib` models or database files.
8. Open **GitHub repository → Actions → ALPHA-LSE autonomous commodity research → Run workflow**, select `check`, then **Run workflow**. A green run tests Supabase connectivity and durable write/readback without depending on stock market opening times. In Supabase `alpha-lse-models` you should see `research/connection_check_v1.json` and `research/worker_health_v1.json`.
9. After the **London exchange has closed** (normally after 17:05, excluding holidays) run the same workflow manually with `daily`, or leave it until the scheduled evening job. This writes `research/latest_forecasts_v1.json`, `research/predictions/2026.json`, and, after enough new price sessions, `research/outcomes/2026.json` and a scorecard. If fewer than three verified same-day ETC bars are available, it **fails safely**, alerts and retries at the later scheduled time.
10. Reboot Streamlit Cloud. On the **Commodities** tab, open **Commodity ML learning → Independent background research**. Check the last daily run, saved forecasts and accuracy scorecard. **Before counting these as live proof, wait five trading sessions for the first outcomes to mature.**

### GitHub Actions recurring schedule (all hours UTC)

- **Hourly at minute 17, weekdays 08:00–16:59 UTC:** independent price/trend scan (honours UK exchange sessions, no buys).
- **Weekdays 18:43 and 20:13 UTC:** daily model validation, new stored forecasts and mature-outcome audit. Second time is a retry and idempotent.
- **Weekdays 11:29 UTC:** check worker freshness and alert on a stale heartbeat.
- **Sundays 20:23 UTC:** immutable compressed backup of the Supabase `predictions` paper-trade table. Verify a snapshot after the first Sunday.

### What counts as reliable accuracy?

- A forecast must be saved with its **date and model version before its future result occurs**. A historical test is not enough.
- Compare matured predictions with the five subsequent exchange sessions, not intraday paper trade wins. This app predicts a >1.5% close-to-close change **before fees**.
- The dashboard shows ML versus simple baseline Brier scores and, where enough predictions exist, by-product precision. Lower Brier is better, but an apparent edge may still be statistical noise.
- Reject/challenge on future data only. **No guarantee of accuracy or profitability and no fixed timetable for an approved model.**

## 10-year reality and maintenance

This is a **maintainable architecture**, not a promise of ten years unattended. You must keep an eye on:

- **GitHub Actions limitations:** scheduled jobs can be delayed or dropped; in *public* repos GitHub automatically disables scheduled workflows after 60 days without repository activity. This prevents literal zero-touch operation for ten years. Use a maintained/private repository (subject to billing/limits) or a managed always-on scheduler for sustained long-term use.
- **Yahoo Finance and ETC instrument changes:** unofficial market feeds, tickers, corporate actions, product closures and currency conventions can change. The worker **fails closed** rather than fabricating prices.
- **Streamlit/Supabase/service-role keys:** keys can expire or be rotated; free quotas/storage may be reduced, provider prices and libraries change, and cloud data may be unavailable.
- **Backups:** a private Supabase Storage backup in the **same project** is not an offsite disaster-recovery plan. Periodically export the encrypted archive somewhere separate and verify restoring it.
- **Alerts:** an internal GitHub Actions health job cannot alert if GitHub itself stops dispatching jobs. A truly unattended production system requires an **independent external heartbeat monitor** and a paid/managed scheduler with retries and billing. Optionally configure an external uptime service to check a public proxy heartbeat without exposing private Supabase buckets or secrets.
- **Live orders:** not supported. This is simulated research and paper trading only.

## If it fails

- **GitHub Actions → autonomous workflow → failed run:** read the stage summary. Never paste your service key, API headers or secrets into support chats.
- **No forecast file:** run the `check` workflow first; then run `daily` after London market close on a trading day. A missing/invalid price feed deliberately blocks forecasts.
- **Dashboard says stale worker:** check GitHub scheduled workflows, secrets and Supabase permissions; approved ML filters are paused while the worker is stale.
- **Challenger rejected:** correct behaviour if it fails future-data testing. Research forecasts continue; don't lower thresholds merely to force trades.
- **Backup missing:** inspect the Sunday workflow and the `backups/paper_journal/YYYY-MM-DD.json.gz` path in your **private** Supabase bucket.
- **Secrets exposure suspected:** rotate credentials in Supabase and update both GitHub Actions and Streamlit Secrets.

## Local offline tests

```powershell
python -m pip install -r requirements.txt
python -m pip install pytest
python -m pytest -q tests
python -m py_compile app.py commodity_ml.py commodity_worker.py
```

**Live provider training, GitHub scheduling, Supabase writes, restoration, and ten-year stability were NOT tested from this offline environment.** Do not treat offline tests as those assurances.
