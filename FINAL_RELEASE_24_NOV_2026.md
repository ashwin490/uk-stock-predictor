# ALPHA-LSE — Final Commodity ML Release (24 November 2026)

This ZIP is a **full replacement code release based on the 23 November final commodity-card and auto-refresh build**, not a standalone one-file patch. It keeps the previously corrected cards, dynamic WAIT/locked OPEN price levels, automatic UK-hours refresh, GBP-only ETC checks, paper trading, Supabase historical journals, share ML, and existing fallback/risk safeguards.

**Added:** `commodity_ml.py`; one-time `COMMODITY_ML_SETUP.md`; new historical ML tests; user-facing ML validation, probability-event labels, versioned per-trade history, guarded promotion and rollback-aware champion retention, private Supabase JSON persistence.

**Deployment action required:** See `COMMODITY_ML_SETUP.md` to create a **private** Supabase Storage bucket and add a separate server-only model key to Streamlit Secrets. Your existing `SUPABASE_KEY` for journal records can remain unchanged.

**Important boundaries:** All trading is paper-only. ML forecasts a specified five-session gross price movement, NOT realised profit, target execution, or guaranteed return. Only validated models can filter entries. No historical production market data could be fetched during offline packaging, so initial real-data training occurs on Streamlit startup after secure storage setup. Automated re-evaluation needs an active Streamlit browser session and adequate provider history.
