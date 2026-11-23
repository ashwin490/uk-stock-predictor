> **Historical change record; superseded by [COMMODITIES_AND_OPTIONS.md](COMMODITIES_AND_OPTIONS.md).** The new version retires the US options tab and enables separate rule-based commodity paper-trading signals.

# ALPHA-LSE — 23 November 2026 (paper-trading research upgrade)

## Scope

This package is a complete update built on the previous November 20 app, including **55 passing offline regression tests**. It is a **paper-trading simulator**, not a brokerage connection and **not evidence of profitable trading**. Results before and after this update should be compared carefully because audit and options-accounting rules have changed.

### Improvements

1. **No new learning-pick trades:** The scanner automatically records only ideas that pass the regular qualification and risk rules. Previous learning picks remain labelled in historical performance to avoid silently rewriting trade outcomes. This means the program may place no simulated trades for an extended period; that is intentional.
2. **Brief target touch detection:** A new `intraday_exit.py` module can inspect timezone-aware, 1-minute Yahoo OHLC bars **strictly after the saved buy time**, using the first target/stop hit. If target and stop are reached in the same minute, it assumes STOP was reached first. It ignores the minute containing the purchase because that minute could include highs from before the purchase. Entry-date fallback checks do not use an unverified later raised stop. A cloud record without a trustworthy original entry timestamp will not receive a fabricated intraday win.
3. **Delayed/limited data warning:** Yahoo intraday prices can be delayed, missing, stale or unavailable. The virtual exit may be recorded when the next refresh eventually observes it, not instantly after the five-minute target touch. A **high print is not a guaranteed fill** at a real exchange or broker. If exact fills or unattended guarantees are required, a licensed live market feed, persistent minute-bar/event log, and separate background worker are required. No free public price feed can guarantee every sub-minute crossing or order execution.
4. **Commodity ETC watchlist (not stock AI predictions):** Added UK-listed sterling-traded **SGLN.L** (gold), **SSLN.L** (silver), **BRNG.L** (Brent oil), **CRUP.L** (WTI oil) and **NGSP.L** (natural gas). Products must pass quote currency, liquidity and volatility gates. Users may manually authorise a **paper** ETC order, with fresh quote and risk-size checks. These products are **not** auto-bought by a stock-trained model; energy ETCs can diverge from spot due to futures rollover, expenses and leverage exposure depending on product. Other international futures (**GC=F**, **SI=F**, **BZ=F**, **CL=F**, **NG=F**) are price references only, NOT a futures simulator. Product tickers/currencies must be reviewed as markets evolve.
5. **Honest options evidence:** The main options profit/loss now includes only contracts created with observed US option bid/ask quotes. Older made-up Black–Scholes practice trades remain visible separately but **cannot make the options score look profitable**. The independent options learner also only uses real-quote results. **New US call ideas are OFF by default** after poor early real-quote outcomes; existing options still receive audits. A manual advanced switch can enable new paper calls for experiments. Real options also suffer from delayed quotes and missing intraday peaks, so 5-minute perfect target capture is not claimed for options.
6. **Feature/dependency repairs:** Includes corrected `feature_engine.py` to prevent `sqrt` warnings and preserve valid feature columns, plus `requirements.txt` pinning scikit-learn to **1.7.2** for the saved encoder version reported in your logs. Review actual model compatibility before redeploying. Data vendor versions still need monitoring.
7. **No change to your private history or model files:** Database schema and stored history are retained. Do not delete old synthetic call rows or legacy learning picks; they are needed for auditability. The provided project **does not contain** your trained `.joblib` model, `rns_scraper.py`, passwords, Supabase database or trading history.

## Cloud deployment (your current website)

1. Back up your **old `app.py`**, Supabase database and trained model first. Confirm old historical results and sample trade IDs.
2. Replace these files **as a set** in your GitHub project: `app.py`, `feature_engine.py`, `requirements.txt`. Add new `intraday_exit.py`, `commodity_etps.py` and the updated `tests/`, `.github/workflows/regression.yml`. Keep your existing `rns_scraper.py` and private trained model. Keep passwords and Supabase keys only in Streamlit Secrets.
3. Streamlit Community Cloud: keep `ALPHA_STORAGE_MODE = "cloud"`, valid `SUPABASE_URL`, `SUPABASE_KEY`, `AUTH_USER`, and whichever of `AUTH_PASS_HASH` or `AUTH_PASS_ARGON2ID` you use. Code pushes do **not** update Streamlit Secrets.
4. After redeploy, inspect login, the historical 109+ finished share trades, the 23 past option trades, your open positions, one-minute history availability, and cloud sync status. Historic totals can differ depending on whether pending records have settled; **do not reset your Supabase tables** to force them to match.
5. Keep it paper-only. Live integration tests against your market-data source, actual model and Supabase have **not** been run here.

## Laptop deployment

Same project files. Run `setup_windows.bat` on Python 3.11 and `start_local.bat`. Store models, DuckDB and backups outside Git. See [README.md](README.md). For accurate minute exits, keep the browser session/auto-refresh active; no separate headless scheduler is provided.

## Smoke tests

```bash
python -m py_compile app.py feature_engine.py intraday_exit.py commodity_etps.py alpha_backup.py
python -m pytest -q tests
```

As of this build the result is **55 passing offline tests**, not an independently validated accuracy score. At least one test simulates a target touched minutes after the recorded entry followed by a drop below target; the recorded result is **target hit**, with stop-first treatment of minute ambiguity.

## GitHub

```bash
git add app.py feature_engine.py requirements.txt intraday_exit.py commodity_etps.py tests .github README.md QUICK_START.md NOV23_UPDATE.md FINAL_RELEASE_NOTES.md
git commit -m "Improve ALPHA-LSE intraday exits, remove learning picks, add commodity research"
git push origin HEAD
```

**Never add `.streamlit/secrets.toml`, private database files, model binaries or learning backups to a public repository.**
