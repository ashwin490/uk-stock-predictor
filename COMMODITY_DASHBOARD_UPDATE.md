# Commodity dashboard upgrade — 23 November 2026

## What's new

- Five visual commodity cards (gold, silver, Brent, WTI, natural gas) replace the wide signal table. Every card shows reference buy price, sell target, stop-loss, current paper units and spend, trend checks, decision and reason.
- Commodity performance boxes appear **on both the Home screen and Commodities tab**: open ETC count; estimated unrealised open P/L; realised P/L from finished ETC trades; won versus lost counts (with protected and expired totals). They use the existing trade journal and do not mix results with the stock AI.
- A full commodity paper-trade history table includes **buy time**, instrument, units, buy price, invested £, sell target, stop-loss, latest/exit price, exit time, realised or estimated P/L in £ and %, and the outcome. Downloadable as CSV. Old and new ETC trades remain in the same journal.
- **Cautious, separate commodity feedback**: after at least **20 finished trades for a particular commodity**, if its most recent 10 finished trades average **-1% or worse after estimated costs** and at most 3 were profitable, its *future paper stake* is halved. Otherwise position sizing stays unchanged. The mechanism never increases risk and does not affect the share AI, historical records or stop-loss calculations. It is *not* proven model retraining or a validated win probability.
- Auto-refresh now starts on opening transitions and runs a lightweight browser-session wake-up check while disabled, so a page opened before the market opens can enable scanning without manually flicking the toggle. Manual pause is still available; it resets at the next trading session/window. After-hours heartbeat reruns the page but does not run full market scans when auto-refresh is off.

## Installing (existing GitHub/Streamlit Cloud project)

1. Back up the existing project, Supabase database, trained models and private Streamlit Secrets.
2. Copy files from the ZIP into your existing repository, replacing the project files. **Do not delete** `feature_engine.py`, `rns_scraper.py`, your `models` directory, database or `.streamlit/secrets.toml`.
3. **Required new files** are `commodity_dashboard.py` and `auto_refresh_schedule.py` alongside `app.py`. Do not deploy only `app.py`.
4. Keep `ALPHA_STORAGE_MODE = "cloud"`, `SUPABASE_URL` and `SUPABASE_KEY` in Streamlit Cloud Secrets if that's where you run the app. Authentication settings `AUTH_USER` and `AUTH_PASS_HASH` must also remain configured.
5. Commit and push the changes:
   ```bash
   git add app.py commodity_dashboard.py auto_refresh_schedule.py commodity_signals.py commodity_etps.py tests COMMODITY_DASHBOARD_UPDATE.md
   git commit -m "Add commodity cards, trade history and market-hours auto refresh"
   git push origin HEAD
   ```
6. In Streamlit Cloud verify **Home → Commodities at a glance** and **Commodities → five cards + full history**. Check Supabase sync and the actual market-data service before relying on results.

## Important limitations

- Streamlit auto-refresh works only while the app **has an active browser session**; browser throttling, sleep and delayed Yahoo data can prevent a check at the exact opening bell or intraday target touch. It is **not** a background trading service.
- This remains **paper trading only**, with estimated spreads, costs and fills. Buy/target/stop prices are for sterling-quoted London ETC units in **pence**, not global spot commodity prices.
- A completed paper record is not proof that a real broker could have executed its exit at the same price.
- Offline tests and syntax compilation do not establish live Supabase connectivity or profitability. A complete on-deployment validation is still required.
