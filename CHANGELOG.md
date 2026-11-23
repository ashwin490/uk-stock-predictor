# Final reliability pass — 20 November 2026

- All fixes from the prior audit release are included.
- Added **Argon2id** login hashing for new local installations. Existing SHA-256 `AUTH_PASS_HASH` secrets still work unchanged; no default password or silent fallback is introduced.
- Streamed ZIP backups, restoration and checksum verification in 1 MiB chunks to avoid reading large databases/models entirely into memory.
- Restoration now attempts to roll back all replaced files if a later file operation fails. The operator must still stop the app before restoring.
- Supabase update and insert logic requires a positive provider acknowledgement before claiming a paper trade was saved.
- Removed unused plotting functions and optional Plotly imports; historical calibration summaries remain in plain English.
- Added `health_check.py` / `check_health.bat` for an offline, nonsecret setup and backup report.
- Added monthly GitHub Actions regression runs and Dependabot PRs. They **detect** potential incompatibilities; they do not self-update or repair the project.
- **43 offline regression tests passed**, plus syntax compilation. Full production, model and live provider integrations could not be exercised in this environment.

---

# ALPHA-LSE reliability update — 20 November 2026

## Changes made to the supplied `app.py`

1. **Corrected next-day auditing:** if a trade was checked in the middle of yesterday's trading session, yesterday's subsequently completed bar is inspected when the app resumes. Previously it could be skipped.
2. **Stopped retroactive break-even exits:** a newly raised protective stop is not applied to earlier lows from the same unfinished daily candle. Stops already in effect are still honoured.
3. **Protected same-day entries:** do not use an entire day's high/low to judge an entry made halfway through that day. On the next day, use only the entry-day close as a limited reconciliation fallback.
4. **Avoided inventing exits after laptop downtime:** the price-history lookback grows with the age of an open trade. If the provider still cannot show the relevant history, the position stays OPEN with a visible warning instead of receiving a fabricated outcome. Time exits are not recorded on weekends or without today's quote.
5. **Validated price candles:** bad OHLC records are discarded before modelling or trade audits.
6. **Tightened self-retraining evaluation:** the model challenger is evaluated on later entry dates with a seven-day separation from training dates, and enough independent observations must be available. This reduces look-ahead leakage but does not guarantee better future predictions.
7. **Simplified the market-summary interface:** replaced the graphical screening funnel and trend gauge with written counts and plain-language sector summaries; showed the latest AI validation sample where available.
8. Preserved the £50,000 starting **paper** balance, selectable unlimited upper account balance, dynamic quantities, no arbitrary share/daily-trade count cap, qualified-pick allowance in selloffs, backup formats, existing DuckDB schema, Supabase table names and fail-closed login.

## Validation

- 36 offline tests passed (`python -m pytest -q tests`).
- `python -m py_compile app.py alpha_backup.py setup_auth.py` passed.
- **Not tested:** a full Streamlit browser session, the user's real model/features, real Yahoo Finance candles, live market prices, or the user's Supabase credentials/schema. Those integrations need to be tested in the actual deployed environment.

## What the update does NOT promise

- It does not make a trading strategy profitable, make probability estimates scientifically validated, or ensure uninterrupted operation for ten years.
- A Streamlit browser session is still required for the built-in periodic refresh. Closing the browser stops those refreshes. For unattended scanning add a separate worker/scheduler.
- Streamlit Community Cloud local disk is not persistent. Use `ALPHA_STORAGE_MODE = "cloud"` with valid Supabase secrets there. Use the Windows `start_local.bat` to run in durable laptop-local mode.
- This is a **paper-trading simulator**. It does not execute real broker purchases.
- Password authentication still uses the existing SHA-256-hash format so the update does not lock existing users out. For public production exposure, migrate to a managed auth provider or Argon2id with a planned credentials migration.

## Safe upgrade

1. **Make a backup** of your existing project, DuckDB file, model files and Supabase historical data before replacing code.
2. Replace your current `app.py` with the `app.py` in this project. If your repository doesn't already have them, also include `alpha_backup.py` and the Windows utility files.
3. Keep your own `feature_engine.py`, `rns_scraper.py`, `models/ensemble_ranker.joblib`, and secrets; those original custom files were not part of the uploaded source and are not supplied in this package.
4. If running on Streamlit Cloud, keep `ALPHA_STORAGE_MODE = "cloud"`, `AUTH_USER`, `AUTH_PASS_HASH`, `SUPABASE_URL`, and `SUPABASE_KEY` configured in the app's **Secrets** settings. Never commit the secrets file.
5. Check **System health**, trade journal counts, original model loading and at least one known historic trade. If the journal is unexpectedly empty, **stop** and check the Supabase connection before recording new trades.
6. Run the app in paper mode and verify new signals and settlement before considering it reliable. Never delete historical files to resolve display errors.

## GitHub push (after reviewing staged files)

```bash
git status
git add app.py alpha_backup.py tests README.md QUICK_START.md CHANGELOG.md
git commit -m "Fix equity audit accuracy and improve model validation"
git push origin HEAD
```

The GitHub commands intentionally stage only source files and documents, not `secrets.toml`, model binaries or your trade database.

## 2026-11-23 — Commodity dashboard and market-hours scheduler
- Replaced the commodity signals table with five responsive visual cards showing buy, target, stop, decision and reason.
- Added separate commodity open count, realised and unrealised P/L, and win/loss summaries to the front page and commodity tab.
- Added full historical ETC purchase and outcome table with CSV export.
- Added conservative, trade-history-based commodity *paper stake reduction only* after sufficient evidence; no self-claimed AI forecasts.
- Added market-open/entry-window state transition auto-refresh and lightweight browser-session heartbeat.
- Added offline tests for history totals, visual card formatting, feedback and auto-refresh state transitions.
