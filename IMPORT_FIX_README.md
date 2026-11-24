# ALPHA-LSE import error fix — 23 November 2026

The latest `app.py` imports `decorate_commodity_card` from `commodity_dashboard.py`. An older version of `commodity_dashboard.py` does not contain that function.

## Fix

1. Copy the included `commodity_dashboard.py` into your **GitHub project folder, beside `app.py`**, replacing the older file.
2. In VS Code Terminal, from the project folder, run:

```powershell
python -c "from commodity_dashboard import decorate_commodity_card; print('Import OK')"
git add commodity_dashboard.py
git commit -m "Fix commodity dashboard import mismatch"
git push origin HEAD
```

3. Confirm the files on GitHub are both in the same folder, then use Streamlit Cloud **Manage app → Reboot app**, if it does not redeploy automatically.

The ZIP also contains the matching `app.py` and other components, in case you need to replace the full project rather than one file. Do not delete your models, database, previous trading records, or Streamlit Secrets.

If another error appears after this fix, check the first and last lines of the new Streamlit traceback; do not share passwords or API keys.
