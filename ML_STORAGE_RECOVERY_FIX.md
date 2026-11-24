# ALPHA-LSE — Commodity ML storage recovery (24 November 2026)

## Symptom

The commodity dashboard displays an older historical evaluation but reports:

> ML storage or training unavailable (RuntimeError).

This is **not** evidence that you need another 22 trading days before seeing historical research predictions. The previous test is already saved and a one-time research backfill should add its historical forecasts. An error while updating the *existing* Supabase object is one plausible failure point. The old app hid the failing stage and, after an error, blocked another ML attempt for the rest of the day in that session.

## Changes

- Keeps model file `commodity_ml_champion_v1.json` and existing history in place.
- Uses Supabase Storage's normal upsert to save, with a specific `update()` fallback when the server reports **object already exists**. Never deletes the file or treats a permissions/connection error like a conflict.
- Verifies that the uploaded model JSON can be read back exactly before any candidate is activated.
- Reports *which operation failed* (reading, historical backfill, saving, training) with a sanitized exception type and HTTP status where available. Never displays the API secret or unfiltered exception text.
- After a failed operation, automatically retries after 15 minutes **while the Streamlit app session is active**, rather than never retrying that day.
- Retains previously saved model history and keeps rejected experimental ML estimates from authorising paper purchases.

## Install

1. Back up your existing source project. **Do not delete** the Supabase bucket, model JSON file or paper-trade tables.
2. Extract this ZIP, and copy `app.py` and `commodity_ml.py` over your existing project files. Include the tests if you use GitHub test checks.
3. Do not touch existing `.streamlit/secrets.toml`, local database, trained share model or private Supabase Storage settings.
4. In PowerShell from your Git repository:

   ```powershell
   git add app.py commodity_ml.py tests/test_commodity_ml_storage_recovery.py ML_STORAGE_RECOVERY_FIX.md
   git diff --cached --stat
   git commit -m "Fix commodity ML cloud state updates and retries"
   git push origin main
   ```

5. Restart Streamlit Cloud. Open **Commodities → Commodity ML learning**. Historical research examples and experimental forecasts should appear if the earlier state and verified completed-bar history can be processed and saved.
6. If it still fails, copy only the **Commodity ML update failed while ...** message and the corresponding warning from Streamlit Cloud logs. **Never send any Supabase service-role key or Streamlit Secrets.** The stage and HTTP status should now identify what needs fixing next.

## What is *not* guaranteed

No cloud network integration could be exercised here without your private credentials. This package passes offline tests but cannot establish that your specific Supabase permissions, network or price data are fully healthy. A rejected ML model is still not allowed to buy commodities through the ML filter. It does not guarantee accurate forecasts or profitable trades.
