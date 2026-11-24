# ALPHA-LSE — final UK-share autonomous intelligence release

**24 November 2026 · UK shares + commodity ETCs · virtual/paper portfolio only**

## What has changed

- Existing original UK share buy-model, portfolio sizing, trade history, and commodity ML remain in place.
- New **independent UK share intelligence** runs through `.github/workflows/share_autonomous.yml` on a schedule, not in a browser session.
- Historical UK share data trains a separate **five-LSE-session, >1.5% adjusted-close price-event ML**. It **does not** substitute for the original model's target-hit/trade-win prediction.
- Historical training uses dated features, outcome-complete labels, a five-session outcome embargo, separate tuning and final holdout dates, and comparison with a base-rate forecast. A lower Brier score is better. The model cannot self-certify future accuracy.
- Unaccepted candidates are kept as **Preview** forecasts, without changing simulated share buying. A first research champion may be accepted after out-of-time validation. Later challengers are compared on genuinely forward-recorded **paired shadow forecasts**, and cannot be promoted using another version's results.
- The existing share buy engine gains a **negative-only** external safety filter, but only when that exact champion has ≥100 mature forward observations, beats its own baseline by a defined margin, has a fresh heartbeat, and a matching dated ticker forecast. No research prediction can originate a purchase. Where evidence is absent, the original share controls remain unchanged.
- Month-sharded, first-write-wins forecast ledgers in the **existing private** Supabase bucket (`research/shares/predictions/YYYY-MM.json`); separate outcome ledgers (`research/shares/outcomes/YYYY-MM.json`) and persistent JSON model versions. Automatic daily forecast recording; outcome scoring after five complete exchange sessions.
- A non-mutating independent **paper-ledger accounting audit** verifies saved entry/latest price, allocated cash, P/L and recorded commission/spread/tax assumptions; it marks missing old data as **unverifiable** rather than wrongly declaring zero fees. Duplicate rows and discrepancies are reported but never silently rewritten.
- A dedicated **Share Intelligence** panel on **🧠 Learning progress** shows dated model status, researched symbols, forward scores and accounting reconciliation. Commodity cards and the app header now have shorter, cleaner copy. The essential **PAPER ACCOUNT** and **Preview / Validated** distinctions remain visible.
- Existing Sunday Supabase paper-journal snapshots and commodity research workflows remain unmodified. The new share worker reuses the existing private bucket and GitHub Secrets.

## One-time deployment (you already have the two GitHub secrets)

1. **Back up** your current local project and verify the existing Supabase paper history. Retain any `models/ensemble_ranker.joblib`, previous models, local DuckDB, `.streamlit/secrets.toml`, and private `rns_scraper.py` separately. **Do not replace them** with ZIP content. The ZIP deliberately excludes your private trained model and credentials.
2. Unzip into your VS Code repository folder, replacing matching project files. Confirm these *new* files are present:
   - `share_research.py`
   - `share_worker.py`
   - `.github/workflows/share_autonomous.yml`
   - `tests/test_share_autonomous.py`
3. In VS Code PowerShell:

   ```powershell
   git status --short
   git add app.py commodity_dashboard.py share_research.py share_worker.py .github/workflows/share_autonomous.yml .github/workflows/regression.yml tests README.md FINAL_SHARE_RELEASE_GUIDE.md
   git diff --cached --name-only
   git diff --cached --check
   git commit -m "Add autonomous UK-share forecasting, audit and professional dashboard"
   git push origin main
   ```

   **Check the staged filenames** carefully before the commit. Do not commit `.streamlit/secrets.toml`, any `.env`, private keys, `.joblib` files, database files or backups. If an intended file is unchanged, Git may say nothing to commit; verify `git status` first.
4. Open [your GitHub Actions page](https://github.com/ashwin490/uk-stock-predictor/actions) → **ALPHA-LSE autonomous UK share intelligence** → **Run workflow** → `main` + `check`. A green tick tests private bucket write/readback using your existing secrets. `check` is **not** the daily forecast operation.
5. **After the UK market closes** (17:05 UK on ordinary trading days), manually select `daily` for the first run. It fetches up to **120** FTSE-listed UK shares by default, validates sterling currency and price bars, tests models, saves forecasts and verifies Supabase readback. The live FTSE rosters are attempted on scheduled runs with a bundled fallback. **If insufficient verified feeds are available it fails safely**, instead of fabricating predictions. Follow the job logs if red.
6. Run the same new workflow once with `audit`, to inspect your historical paper equity rows. The audit writes the report only; it **does not** change trade history. Old records with missing fee assumptions will show as unverifiable, not as mismatches.
7. Reboot Streamlit Cloud. Open **🧠 Learning progress → Share intelligence**. See the latest daily run, forecasts, evaluation and accounting audit. All the existing Streamlit Secrets and `ALPHA_BACKGROUND_WORKER_ENABLED = true` remain unchanged. **No new GitHub secret is required** if your previously working commodity `check` workflow already used `SUPABASE_URL` and `SUPABASE_MODEL_SERVICE_KEY`.

## Expected private Storage objects

```
alpha-lse-models/
  research/shares/
    connection_check_v1.json
    worker_health_v1.json
    model_state_v1.json
    model_trials/YYYY-MM-DD.json
    latest_forecasts_v1.json
    predictions/YYYY-MM.json
    outcomes/YYYY-MM.json
    scorecard_v1.json
    accounting_audit_v1.json
```

Model and forecast files only appear after a successful `daily` run with sufficient real data. Outcomes appear after five complete later exchange sessions; the earliest independent scores are not available immediately. Supabase paths here are private bucket object names, **not local Git files**.

## How long it runs without you

The GitHub workflow schedules research on weekdays after market close (19:07 and 20:47 UTC), a weekday heartbeat check, and a Sunday audit. You can also run `daily`, `check`, `monitor`, or `audit` manually. **Only the new price research and audit are browser-independent. Actual simulated purchases and closing of paper trades still run through the existing Streamlit session.** They will not run without an active app session.

This is a **low-touch** architecture, not a technically honest ten-year guarantee. GitHub Actions may delay/drop runs, disable schedules on inactive public repos, or run out of quota. Yahoo/tickers/UK calendars, Supabase billing/credentials and Python dependencies may change. An optional `ALPHA_ALERT_WEBHOOK_URL` GitHub secret can send safe failure summaries to a trusted webhook. For alerts even when GitHub stops scheduling, set up an external heartbeat monitor. Someone must respond to failures and maintain backups periodically. **Neither model's future accuracy can be promised.**

## Tests and limits

Use `python -m pip install -r requirements.txt`, followed by `python -m pip install pytest`, `python -m pytest -q tests`, and `python -m py_compile app.py share_research.py share_worker.py`. Included offline tests check feature leakage, purged train/tune/test dates, first-write-wins forecasts, version-specific forward scoring, idempotent daily runs, safe failure on bad data, and independent P/L auditing. **Actual live Yahoo Finance accuracy, this Supabase account, real GitHub scheduled execution and Streamlit Cloud rendering cannot be verified in the offline package.** Run `check`, `daily`, and `audit` once before treating the deployment as operational.
