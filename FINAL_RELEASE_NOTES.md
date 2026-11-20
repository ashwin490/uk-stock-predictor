# ALPHA-LSE final reviewed package — 20 November 2026

**This is the latest complete `app.py` replacement, including previous changes.** It remains a **paper-trading research simulator**: it does not submit actual broker orders. The system estimates outcomes from public, potentially delayed market data; it is not proven to be profitable, and no one can guarantee ten years without maintenance.

## What remains from earlier versions

- £50,000 starting **virtual account**; you may choose a higher/lower positive value; no fixed pound ceiling or fixed number-of-shares limit. Cash, portfolio exposure and initial stop-risk checks determine quantities.
- No fixed daily paper-trade count. Market-fall warning remains; fully qualifying ideas may be recorded under the default setting.
- FTSE 350 plus selected AIM tickers and configurable additional UK-listed shares, subject to quote, liquidity, volatility and sector checks. **Not every global market/security.**
- Original trained model plus history-informed learning and evidence-gated challenger validation. A trained model only replaces the active model if it improves defined held-out scores; this is NOT a guarantee of true predictive improvement.
- Existing DuckDB trading-journal schema and Supabase table names preserved; local-first backup/restore migration; plain-language, graph-free front end.
- Historical trade auditor safeguards for gap prices, stop ratchets, missing candles, stale data and splits. Missing reliable pricing leaves a trade awaiting review rather than fabricating an outcome.

## New in this final pass

1. **Safer large backups and restores**: archive, integrity-check and extraction stream data in chunks; if restoration fails midway, attempt to restore the previous files.
2. **Stronger authentication option**: new local installations get an Argon2id-hashed login with a user-chosen 16+ character password. Earlier cloud/local `AUTH_USER` plus `AUTH_PASS_HASH` (SHA-256) credentials continue working. For a future stronger cloud login, configure `AUTH_PASS_ARGON2ID` and keep `AUTH_USER`; never push secrets to GitHub. If an Argon2id secret is configured, login will not fall back to SHA-256 when it fails.
3. **Cloud-write acknowledgement**: the app does not report a successful paper trade when Supabase fails to return confirmation for a new trade or update.
4. **No unused chart engine**: the default UI retains simple written explanations and result tables.
5. **Offline system check**: on Windows, run `check_health.bat` after setup (ideally with the app stopped). This checks that the source, private model, login file, local journal and backups are where they should be, without showing secrets.
6. **Monthly automated detection**: `.github/workflows/regression.yml` runs offline tests monthly and on pushes; `.github/dependabot.yml` proposes dependency updates. **These features need GitHub Actions/Dependabot enabled and do not automatically fix problems.**

## Deploy safely — laptop

1. **Back up first**: copy your previous `app.py`, `lse_market_data.duckdb`, entire `models/` directory, and current local secrets to a *private* location. Also independently back up any older data stored in Supabase. Never wipe the old database to make an empty dashboard look healthy.
2. Replace **your existing** `app.py` with the supplied `app.py`. Copy `alpha_backup.py`, `health_check.py`, test files, `.github/`, requirements and launcher files from this package. **Keep your real** `feature_engine.py`, `rns_scraper.py`, `models/ensemble_ranker.joblib`, local DB, JSON data and `.streamlit/secrets.toml`. Those private/custom files were **not provided** and are not included in this ZIP.
3. If you already have a `.streamlit/secrets.toml` with a valid `AUTH_USER` and `AUTH_PASS_HASH`, **keep it unchanged**. Your existing password will still work. For a brand-new laptop-only installation, run `setup_windows.bat` and choose a new long password; it now generates Argon2id credentials.
4. Run `setup_windows.bat` to install updated requirements, then `start_local.bat`. Visit `http://127.0.0.1:8501`.
5. Verify the journal contains a known earlier trade, the right original model is detected, and the latest active positions are present. If records are missing, stop and fix the data source instead of generating new positions.
6. Run `check_health.bat` with Streamlit stopped to inspect backup integrity. Keep a recent backup on an **encrypted external drive**, not just on the same laptop.

## If you still run it on Streamlit Community Cloud

Configure these secrets in **app Settings → Secrets**: `ALPHA_STORAGE_MODE = "cloud"`, `AUTH_USER`, the existing `AUTH_PASS_HASH` or new `AUTH_PASS_ARGON2ID`, `SUPABASE_URL`, and `SUPABASE_KEY`. Do not set `ALPHA_STORAGE_MODE = "local"` on temporary Community Cloud disks when historical data must survive restarts. Back up Supabase history and check trade counts **before** recording new paper trades. Auto-retraining needs *persistent model storage*; Community Cloud's temporary disk is not sufficient.

## Future laptop migration and cloud upgrade

- Use **Settings → Save and move your AI's learning → Create portable learning backup**, or stop the local app and run `backup_local.bat`. Verify the ZIP with `check_health.bat` or `python alpha_backup.py verify <zip-path>`.
- On a new Windows laptop or a cloud VPS with a **persistent attached disk**, clone the project, copy the trusted feature modules, stop any running app, restore the verified data archive with `restore_local.bat`, configure login secrets privately, and check history/model counts. Don't delete the old machine's data before confirming the migration.
- This archive does **not** contain secrets, nor does it replace independent cloud/Supabase backups. Model files are executable Python deserialization objects: never load models from unknown senders.

## What must still be maintained

- **Do not leave it completely unattended for ten years.** Trading rules, exchange holidays, UK tax and duty rules, broker fees, Python/Windows releases, yfinance/Yahoo formats, Streamlit, Supabase APIs and model quality can all change.
- The current scanner still requires an **active Streamlit browser session** to trigger periodic refreshes. Closing the page or allowing the laptop to sleep stops background scanning; the app does not have a separate headless scheduler.
- An automatic monthly CI test can detect some issues, not fix them. Apply security updates, verify market data and backups, and review actual paper accuracy regularly.
- **Full end-to-end validation remains outstanding.** The private model, live market downloads, Streamlit UI, Supabase account and database engine are not available in this test environment. Offline tests validate selected code paths, not trading profitability or production correctness.

## Verification performed in this environment

- `python -m py_compile app.py alpha_backup.py setup_auth.py health_check.py` succeeded.
- `python -m pytest -q tests` — **43 passed**.
- DuckDB, Streamlit, Yahoo Finance data and your personal model/Supabase were **not installed or connected in this runtime**, so full integration tests must happen on your laptop.

## GitHub commit and push

After copying the project files into your existing repository and checking `git status`:

```bash
git add app.py alpha_backup.py health_check.py check_health.bat setup_auth.py setup_windows.bat start_local.bat backup_local.bat restore_local.bat requirements.txt tests .github README.md QUICK_START.md CHANGELOG.md FINAL_RELEASE_NOTES.md
git commit -m "Finalize ALPHA-LSE paper-trading reliability and backups"
git push origin HEAD
```

Never add `.streamlit/secrets.toml`, the `.duckdb` journal, private model binaries or backup ZIPs to GitHub.
