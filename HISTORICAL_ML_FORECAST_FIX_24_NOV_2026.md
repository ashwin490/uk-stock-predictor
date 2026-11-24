# ALPHA-LSE — Historical Commodity ML Predictions (24 Nov 2026)

This update solves a visibility gap: the old system DID test a commodity ML candidate on historical dates, but showed only a rejected score and hid all individual historical and current forecasts until a model passed promotion. It is reasonable to expect to see what the historical model actually predicted, even when its accuracy is inadequate for live paper-order filtering.

## What changed

- **Five cards:** show a dated, per-ETC, research-only estimate of the probability of the ETC closing more than 1.5% higher after five further trading sessions, based on completed historical daily price bars. These are NOT verified win probabilities, recommendations or approval to buy. The date and optional experimental cutoff are clearly shown. A missing historical feed will still show no estimate.
- **Historical predictions table:** each ETC now shows how many held-out historical examples were tested, how many rises the old candidate predicted at its experimental threshold, how many actually happened, and a per-ETC Brier score. An expandable table shows up to 3 past held-out predictions per ETC with known future outcomes. This is not a trading-strategy backtest.
- **Research model:** if a candidate fails validation, it is kept as a *research-only* JSON model in the private Supabase state so probabilities can be displayed safely without being used for paper buys.
- **One-time backfill:** existing Supabase ML state already evaluated on 24 November is updated on first run to include research-only predictions without needing to wait for a new trading day. Backfill does not change the training watermark, model promotion, champion status or automatic trading permissions. It only saves additions after Supabase readback validation.
- **Historical evidence retained:** when there are not yet 22 new independent test sessions, the last dated research evidence remains visible rather than vanishing on the next day's evaluation.
- **Misleading setup reminder corrected:** when state exists in cloud mode, the lack of an approved champion is not misreported as proof that Supabase settings are missing.

## What did NOT change

- No training on intraday incomplete bars or future-labelled test dates.
- No weakening of candidate validation rules or automated paper-trading safeguards.
- An unvalidated *research_model* is NEVER treated as the *champion* and can NEVER authorise ML-filtered paper trades. Existing rule-based fallback trading, when enabled and all its safeguards pass, remains as before.
- The UK share-trained AI, paper trade journal, Supabase secrets, and existing bucket contents are not erased.
- Accepted ML upgrades still require fresh independent evidence (at least 22 later labelled test trading dates), a functioning Streamlit session and persistent private cloud storage. No guaranteed approval or profit.

## Installation

1. Back up your existing GitHub/local project folder.
2. Extract the included ZIP archive.
3. Copy **app.py**, **commodity_ml.py**, and **commodity_dashboard.py** into the root of your existing project. Replace those three files as a matching set. The full ZIP also includes tests and other unchanged supporting files.
4. Optionally replace the **tests** folder (this is recommended for future regression checks). Do not copy old .duckdb files, local secrets, or trainable model files over your existing data. Keep your Streamlit Secrets untouched.
5. In VS Code PowerShell:

   ```powershell
   git add app.py commodity_ml.py commodity_dashboard.py tests
   git diff --cached --stat
   git commit -m "Show research-only historical commodity ML predictions"
   git push origin main
   ```

6. Reboot your Streamlit Cloud app. Open **Commodities: Buy / Sell / Stop**. The cards should show **EXPERIMENTAL ML — NOT VALIDATED** with dated probability estimates (if the historic GBp feed is valid). Under **Commodity ML learning**, inspect historical per-ETC test summary and the expandable dated examples.
7. Verify the storage object **alpha-lse-models/commodity_ml_champion_v1.json** has been updated without sharing its contents or credentials publicly. If no research forecast appears, check the app's cloud logs for `Commodity ML unavailable` or `Research preview unavailable` and send only the non-sensitive error text.

## Validation

The code has offline regression tests plus a synthetic-data training/persistence smoke test, but the deployment has **not** been tested against live Supabase/Streamlit/Yahoo feeds. The trained model is not expected to be good merely because it generates a probability. A rejected candidate's predictions are *research output*, not validated instructions to trade.
