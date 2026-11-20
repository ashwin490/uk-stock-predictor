# ALPHA-LSE — local-first simple investing lab (20 Nov 2026)

**Paper trading only:** nothing in this project connects to a real broker or buys live shares. It uses unofficial Yahoo Finance data, your previously trained ML model and trade-history evidence to produce estimates—not reliable predictions or a guarantee of profit. The full app has **not** been live-integrated with your existing `feature_engine.py`, model file, market-data feed, or Supabase account in this review.

## Best setup for your situation

Run from your **Windows laptop first**, saving the paper trade database, learning state and models on your local drive. You do **not** need Streamlit Community Cloud or Supabase while using this local mode. Later, use the same application on a VPS with a persistent disk; no database rewrite is required for that route. Cloud server providers still charge for server/disk storage. A public Streamlit Community Cloud instance is **not** suitable for durable local-only learning state.

## Before upgrading an existing installation

1. Make copies of your **existing** main Python file, `lse_market_data.duckdb` (if any), your entire `models/` folder, `feature_engine.py`, `rns_scraper.py`, and any `*.json` ticker lists.
2. This package includes the complete revised main `app.py` but **does not contain your own trained `ensemble_ranker.joblib`, `feature_engine.py`, or `rns_scraper.py`**, because those files were not uploaded. Copy them from your working project (do not download arbitrary `.joblib` from the internet).
3. If your old historical trades are in Supabase, they are **not** present in an empty local DuckDB database. You can import them once using the local app's **Settings → Save and move your AI's learning → Import existing Supabase paper history** button. You'll need `SUPABASE_URL` and `SUPABASE_KEY` in your local secrets. Confirm counts and closure statuses after import; do not discard the old Supabase history until you've verified the copy.
4. If you were previously using an app filename other than `app.py`, keep your original code as a backup and run `app.py` from this package rather than overwriting it without checking.

## Install on Windows

1. Install **Python 3.11** from python.org. The supplied setup script looks specifically for Python 3.11.
2. Extract this ZIP into a local folder, e.g. `C:\AlphaLSE`. Do not put the active DuckDB file in a OneDrive/Dropbox synced folder (simultaneous syncing or conflicts can corrupt data).
3. Copy `feature_engine.py`, `rns_scraper.py` and `models/ensemble_ranker.joblib` from your existing project into this folder. The `models` directory may need to be created.
4. Double-click `setup_windows.bat`. It creates the virtual environment, installs dependencies and prompts for **your own login credentials** if `.streamlit/secrets.toml` does not exist. The password is not preset; it requires a unique password of at least 16 characters. You can also reuse your old Streamlit secrets securely by copying them to the local `.streamlit/secrets.toml` (never upload to GitHub).
5. Double-click `start_local.bat`. Visit `http://127.0.0.1:8501`. Log in using the username and password you configured. Keep the terminal running.
6. Under **Settings → Virtual account balance**, use any positive amount of pretend money you want; the starting value is £50,000 and **there is no hard GBP upper limit**. Click **Save virtual account balance** so it persists.

Running locally is also possible manually:

```powershell
$env:ALPHA_STORAGE_MODE = "local"
.\.venv\Scripts\python.exe -m streamlit run app.py --server.address 127.0.0.1 --server.port 8501
```

## What was changed

- **Local-first persistence**: without Supabase, new paper trades and updates are stored in durable **local DuckDB**; cloud storage is not required.
- **£50,000 starting virtual account (not a maximum)**: set the size of the **virtual account**; quantities are whole shares, allocated according to stop distance, virtual cash and diversification, with no hard share-count or five-trade-per-day cap.
- **Risk controls** that grow with the virtual account: at most 1% account equity estimated to the initial stop on one new trade; max 5% combined estimated risk; max 25% equity in any one position and 35% in one sector. These are simulation rules, **not** a worst-case-loss guarantee. Stops can gap.
- **Closed trade P/L changes available paper cash**. The account does not create unlimited fictitious money.
- **Prevents duplicate paper trades**, validates the share price (pence/pounds), validates quantities against capital and rechecks sector/portfolio budget inside the journal lock.
- **Simpler interface**: plain-language summaries instead of charts, clearer pages and a local backup/import section.
- **Local model updates enabled**: after 150 labelled winning/losing trades, a challenger is trained on earlier trades and must beat the existing model on later held-out trades to replace the active model. The original `models/ensemble_ranker.joblib` is kept intact. Small sets of wins do not guarantee genuine improvement, and this still needs independent walk-forward evaluation to establish accuracy.
- Model promotion writes a temporary file, validates it, then swaps it into place. The `models/retrain_state.json` and promoted model are included in portable backups.
- A daily backup is attempted **when the Streamlit application is opened and running**; the last seven locally stored ZIP files are kept. They remain on the **same laptop**, so copy one to an encrypted external USB/other device regularly. Settings → Save and move your AI's learning also provides a manual backup download.
- A Windows backup-and-restore utility is included. The archive contains DuckDB, trained models, retraining history and nonsecret settings. **It intentionally excludes passwords and API keys.**

### Reliable operation caveats

**Important: keeping the Streamlit server running is not enough for unattended market scanning.** This existing architecture triggers scans through Streamlit sessions and browser auto-refresh. If the browser is closed, the app stops receiving those periodic reruns; background tabs may also be throttled. The laptop must be awake, connected to the internet and the app session active to keep monitoring. For true 24/7 unattended processing, a separate service/worker and job scheduler are still required; this package does not pretend that exists.

The app relies on changing upstream prices, market calendar, ticker rosters, trading taxes, dependencies and browser/Streamlit APIs. Security patches and monitoring are still necessary over ten years. Do not connect this directly to live brokerage execution.

## Back up and change laptops

While the app is running, use **Settings → Save and move your AI's learning → Create portable learning backup → Download learning backup ZIP**. Alternatively, stop Streamlit and run `backup_local.bat`. Archive files are saved to `backups/`. Check the ZIP with:

```powershell
.\.venv\Scripts\python.exe alpha_backup.py verify backups\alpha_backup_YYYYMMDD_HHMMSS.zip
```

On the new laptop:

1. Install the same Python version, copy the project code and your original modules (`feature_engine.py`, `rns_scraper.py`) and run setup. If using Git, clone from your own private repository.
2. **Stop** Streamlit, then drag the verified ZIP onto `restore_local.bat` to restore the DuckDB journal, champion and live models, retrain state and account settings.
3. Configure a *new* local login password (secrets are not in backups), launch and verify trade counts, the loaded model version and sample historical results. Keep the original laptop copy until checked.

**Security:** `.joblib` files can execute code on loading. Only restore backups you created or trust. Never place `.streamlit/secrets.toml` or the ZIPs in a public GitHub repository.

## Access the laptop remotely from your phone or next laptop

Use [Tailscale Serve](https://tailscale.com/docs/features/tailscale-serve), **not** public router port forwarding:

1. Install Tailscale on the hosting laptop and on your phone/other laptop, signed into your private tailnet.
2. With the Streamlit app running on port 8501, open an **Administrator PowerShell** window on Windows and run:

```powershell
tailscale serve --bg 8501
```

3. Run `tailscale serve status` to obtain the private HTTPS address. Visit it from your other device while connected to Tailscale. The application still has its own password login.

Do **not** use Tailscale Funnel unless you intentionally want to expose the service publicly. Tailscale Serve keeps it private to authorised tailnet devices. The laptop still needs power, network access and an active Streamlit browser session for periodic scanning.

## Moving to an always-on cloud server later

**Simplest route: VPS + persistent disk, not Streamlit Community Cloud.** Upload the same application project and restore your ZIP onto a mounted data directory. The included `Dockerfile` and `docker-compose.yml` use `ALPHA_DATA_DIR=/app/data` and bind `./alpha_data` on the host to that directory. Start the app through Docker Compose and access it privately with Tailscale / a secured reverse proxy. Put your own `feature_engine.py`, `rns_scraper.py` and trusted original model into the deployment before launching. Protect your login secrets and backups.

You can keep `ALPHA_STORAGE_MODE=local` **on the cloud VPS**: the word `local` means **the server's persistent disk**, not your original laptop. This avoids a Supabase storage quota. Verify that the VPS disk is actually persistent through redeploys. A VPS with an ephemeral filesystem would lose data, regardless of software settings.

The alternative `ALPHA_STORAGE_MODE=cloud` is for Supabase-backed persistence and requires correctly configured Supabase tables, credentials and a separate durable store for the live model. A portable ZIP alone cannot convert the database to Supabase; that is a separate migration, and must be tested first. The Docker route avoids that database conversion entirely.

## Tests

Offline tests: `python -m pytest -q tests/test_local_first.py`. The tests validate risk-based sizing, no hard monetary ceiling, break-even stop handling, realised losses, six local trades in a day without Supabase, duplicate checks, quote units, safe backups/restore, and corrupt archives. There is also `python -m py_compile app.py alpha_backup.py`.

**Not tested here:** real Streamlit UI interaction, actual Yahoo data, DuckDB file-level checkpoint/backup compatibility, your separate trained model and feature engine, real Supabase import, or predictive accuracy. These require acceptance checks on your actual laptop. Do not assume passing mocked tests means the app can safely trade real money.

## November 2026 revision: UK selloffs, £50,000 and older data

- The default **starting virtual account is £50,000**, not a £50k purchase for each company. Saved custom account balances remain unchanged. An untouched legacy £10k default is upgraded when reading settings.
- The UK market selloff is a **warning, not a universal pause**. Fully qualified paper-trade predictions may still be logged during a selloff after the same price, market-hours, cash, stop-risk, diversification, and news checks. Experimental learning picks are not allowed during selloffs. The sidebar setting **During a sharp UK market fall** can restore the full pause if you want.
- **Missing earlier results:** When local storage has zero entries and Supabase credentials are configured, the app attempts a read-only copy of existing trade history into its local database. It does not delete or overwrite the old cloud history. If no records load, an on-screen warning explains what to check. The manual import button remains in Settings → Save and move your AI's learning.
- **If you are running the app on Streamlit Community Cloud**, put `ALPHA_STORAGE_MODE = "cloud"` into that app's Streamlit Secrets alongside `AUTH_USER`, `AUTH_PASS_HASH`, `SUPABASE_URL`, and `SUPABASE_KEY`. Cloud mode uses the previous Supabase records and records new paper trades there, instead of relying on an ephemeral local database. Your laptop's `start_local.bat` explicitly selects local mode. Do **not** commit Streamlit Secrets to GitHub. Verify the old records exist before starting new simulated trades. Cloud mode also requires durable model storage for ongoing retraining.
- The £50k is **virtual cash for the entire account**, not a guaranteed investment amount per idea. With 25% single-position allocation, the initial cap for one trade is usually £12,500 before risk and stop-distance adjustments. This keeps the account from investing every pound in one idea.
- **Data safety:** Back up both your old cloud database and the current project before replacing the deployed code. Cloud transactions and historical restoration have not been tested against your personal Supabase schema or market-data feed in this environment.
