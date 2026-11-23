# ALPHA-LSE: Commodity price cards and automatic refresh correction

## 23 November 2026 update

- **Auto-refresh is ON by default** when you open the app during an active UK trading session, even if an older toggle value is False.
- The tab checks periodically outside market hours to switch ON at the next opening. A deliberate manual OFF selection is respected during the current market session; the next opening resets it.
- Refresh only functions with the `streamlit-autorefresh` package installed and the app running in an active browser session. It is **not** a server-side background worker.
- Each commodity card shows the **latest available market-data quote**, which may be delayed, separately from any suggested or saved paper prices.
- **WAIT**: entry, target and stop-loss are *provisional* and may change as new market data arrives.
- **BUY CANDIDATE**: levels are *suggested*, not an executed purchase.
- **PAPER OPEN**: entry and target are read from the saved trade journal and remain fixed; the stop may change only under an existing risk rule. Latest quote and paper P/L remain visible.
- **CLOSED**: original purchase and final sale prices stay in the historical trade table; a new WAIT idea will not overwrite them.
- No existing trades or models are deleted by this code change, and there are no database schema changes.

## Deploy

1. Make a backup of your existing project and trading data.
2. Copy the package contents into your existing project, replacing existing files with matching names. You need **app.py**, **auto_refresh_schedule.py** and **commodity_dashboard.py** together.
3. Keep your existing `models/`, DuckDB files, `rns_scraper.py` and private `.streamlit/secrets.toml` unchanged.
4. Commit and push the code. Do **not** commit your secrets, models, personal trade backups or database files.
5. On Streamlit Community Cloud, check that app installation includes `streamlit-autorefresh`, then restart and verify the toggle is ON when the market is open.

## Verification

Run `python -m pytest -q tests` in a compatible Python environment. Tests cover fresh-session refresh, manual pauses, next-market opening, WAIT prices, open trade entries/targets, and historical snapshots.

Offline tests cannot verify your live exchange feed, Cloud deployment or your Supabase project. Keep paper trading until those integrations have been checked.
