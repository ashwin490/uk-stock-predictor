# ALPHA-LSE final UI / trading-window fix — 23 November 2026

## What was fixed

- Commodity cards now use `st.html` instead of `st.markdown(..., unsafe_allow_html=True)`. This avoids nested `<div class="cm-why">` and `<div class="cm-learning">` appearing as literal text.
- CSS is also rendered using `st.html`, supported by the project's Streamlit >=1.45 requirement. All dynamic commodity text is HTML-escaped.
- A UK market that is open **after the entry cutoff** is clearly labelled: no new simulated purchase after 16:00 UK time on normal trading days, while open positions continue to be monitored until 16:30. Half-day buying ends 30 minutes before the early close.
- WAIT/closed/stale-data reasons are more explicit. A WAIT signal never becomes a recorded trade without passing the existing checks.
- Rejected LSE price bars and unexpected-currency warnings include the ticker for diagnosis. Incorrect USD/EUR unit conversion remains prohibited.
- Previous entries, targets, stop records, portfolio accounting, virtual money, model files and database layout are unchanged.
- Added offline regression tests to catch UI regressions and mismatched helper function imports.

## Update your Streamlit Cloud app

1. Back up your project and historical data. **Do not delete your local model files or Supabase tables.**
2. Extract this ZIP. Copy the project files into the same GitHub repository folder (including `app.py`, `commodity_dashboard.py` and `auto_refresh_schedule.py`). Do **not** copy private secrets or databases.
3. Run `git status`, `git add app.py commodity_dashboard.py auto_refresh_schedule.py tests`, `git commit -m "Fix commodity cards and trading-window status"`, then `git push origin HEAD`.
4. Reboot Streamlit Cloud if needed. Confirm that the app displays five formatted cards without raw `<div>` tags, and after 16:00 displays the monitoring-only notice.
5. Verify old trades still appear and that no simulated commodity trade is recorded unless it qualifies.

## Important limits

The tests are offline. A live end-to-end test with your actual Streamlit Cloud, Yahoo Finance, Supabase and trained model is still required. No long-term stock model or strategy can be guaranteed bug-free, profitable or maintenance-free.
