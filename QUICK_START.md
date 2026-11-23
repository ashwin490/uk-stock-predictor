> **LATEST: 23 NOVEMBER commodities upgrade** — You must also copy `commodity_signals.py` and `option_retirement.py` (not only `app.py`). The US options tab is retired; commodity price/target/stop predictions replace it. See [COMMODITIES_AND_OPTIONS.md](COMMODITIES_AND_OPTIONS.md).

> **IMPORTANT: 23 NOVEMBER 2026 RELEASE** — The complete replacement includes `app.py`, `feature_engine.py`, `intraday_exit.py`, `commodity_etps.py` and `requirements.txt`. **Copy the complete project, including the two additional commodity modules.** Back up your trade history, model and secrets first. Existing `rns_scraper.py` and model files remain untouched. No new learning picks. US call-option engine now removed. See [NOV23_UPDATE.md](NOV23_UPDATE.md).

> **Final reviewed package:** See [FINAL_RELEASE_NOTES.md](FINAL_RELEASE_NOTES.md) for the replacement steps, password compatibility, and validation limits.

> **New reliability build:** replace `app.py` only after backing up your trade database and model. Review [CHANGELOG.md](CHANGELOG.md); See COMMODITIES_AND_OPTIONS.md for current test results.

# ALPHA-LSE — November 2026 update

## What is new?

- Default virtual portfolio: **£50,000** (not £50,000 for every stock).
- **No hard maximum number of shares or daily paper trades**; quantity is still bounded by virtual cash, stop-loss distance and diversification.
- When the FTSE falls sharply, *fully qualified* predictions may still trigger paper trades. Automatic unqualified experimental learning picks have been removed entirely; their historical records remain visible.
- The market fall remains visible as a warning. In the sidebar choose **During a sharp UK market fall → Pause all new buys** to restore the old setting.
- The app alerts you when the chosen database is empty. If local history is empty and valid legacy Supabase secrets exist, it attempts a read-only import.
- Your private original ML model is **not** included; keep it. A corrected `feature_engine.py` is included and must replace the old version.

## If you use Streamlit Community Cloud (GitHub website app)

1. **Back up** the existing Supabase data and your current repository before replacing anything.
2. Replace your repository's Streamlit entrypoint with the supplied `app.py` (or update the Streamlit deployment to point to `app.py`). Replace `feature_engine.py` with the supplied corrected one; keep your own `rns_scraper.py` and model files.
3. In Streamlit Cloud → your app → **Settings → Secrets**, ensure the following are present (preserve actual existing values; never commit them):

   ```toml
   ALPHA_STORAGE_MODE = "cloud"
   AUTH_USER = "your_user_name"
   AUTH_PASS_HASH = "your_existing_64_character_sha256_hash"
   SUPABASE_URL = "your_existing_project_url"
   SUPABASE_KEY = "your_existing_secret_key"
   ```

4. Save and restart. **Verify the historical journal and previous win/loss counts before making new paper trades.** Incorrect Supabase credentials or table permissions can still result in no history. If the remote history is still empty, do not delete the old tables; diagnose the issue first.
5. In cloud mode the app stores transactions in Supabase. Re-trained model persistence is a separate setup: model retraining remains disabled unless a persistent model volume is configured.

## If you run on your laptop

1. Back up the current local `.duckdb` database and `models/` folder if they already exist.
2. Copy the project files to a normal folder, e.g. `C:\AlphaLSE`.
3. Use the corrected supplied `feature_engine.py`; keep your old `rns_scraper.py` and trusted trained model in the indicated locations.
4. Run `setup_windows.bat` once, then `start_local.bat`.
5. The default virtual account is £50,000. Change it in the sidebar if desired.
6. To restore old **Supabase** history onto the laptop, add `SUPABASE_URL` and `SUPABASE_KEY` to local `.streamlit/secrets.toml` (alongside login credentials), then restart. If the local journal was empty, import is attempted automatically; otherwise choose **Settings → Save and move your AI's learning → Import existing Supabase paper history**.

## Testing and safety

- `python -m pytest -q tests` — 55 offline tests passed in the supplied build environment.
- `python -m py_compile app.py alpha_backup.py setup_auth.py` — syntax passed.
- I did **not** validate live quotes, your actual Supabase project, your trained model, or end-to-end Streamlit deployment. Trading forecasts are not guaranteed to be accurate.
- This application records **simulated paper trades**, not actual brokerage orders.

### New Commodity dashboard

Copy **both** `commodity_dashboard.py` and `auto_refresh_schedule.py` with the updated `app.py`. On the home page look for **Commodities at a glance**. On the Commodities tab check the five visual cards, completed trade outcomes and CSV export. Auto-refresh automatically starts on the next UK market opening while the app browser tab is active; it can still be manually paused for the session.
