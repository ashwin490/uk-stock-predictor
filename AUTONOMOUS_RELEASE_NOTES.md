# ALPHA-LSE background research release — 24 November 2026

- New single-writer GitHub Actions scheduled worker, without importing Streamlit.
- Historical candidate search: pooled linear vs individual ETC logistic models, selected on time-purged inner data; separate chronological test required to promote.
- JSON-only versioned model history; automatically retain stronger approved model; cautious rollback only with new independent evidence.
- Immutable dated forecasts; outcome labels after five actual complete London market sessions; year-based cloud ledgers; cumulative and per-product forward Brier scores.
- Hourly GBP/GBp currency-verified ETC research scans, alerts, health signals, weekly Supabase `predictions` backup.
- New Streamlit health/prediction views. When activated, app no longer writes the shared ML state; cloud worker does.
- Existing stock AI, shares and commodity paper journal, old ML model versions and cloud Secrets are not deleted.
- **No independent paper orders**: the current journal and portfolio risk engine remain browser-session-dependent. This avoids duplicate positions and unsafe cross-worker fills.
- GitHub scheduled runs are best-effort, sometimes delayed/dropped, and public scheduled repositories can stop after 60 days of no activity.
- Offline tests do not demonstrate profitable predictions, reliable live data, credential permissions or long-term uptime.
