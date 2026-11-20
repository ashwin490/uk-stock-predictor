"""Offline, read-only health report for an ALPHA-LSE laptop installation.

Run while Streamlit is STOPPED when possible:
    python health_check.py
    python health_check.py --verify-backup

Never prints credentials, changes the trading state or downloads market prices.
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
import tomllib

from alpha_backup import verify_backup


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description='Check ALPHA-LSE installation safely')
    parser.add_argument('--verify-backup', action='store_true', help='Verify latest ZIP member checksums (can take a while)')
    parser.add_argument('--strict', action='store_true', help='Exit nonzero on critical setup problems')
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parent
    data = Path(os.getenv('ALPHA_DATA_DIR', str(root))).expanduser().resolve()
    issues = 0

    def show(ok, label, *, serious=False):
        nonlocal issues
        print(('OK   ' if ok else 'CHECK'), label)
        if serious and not ok:
            issues += 1

    print('ALPHA-LSE health report — no account secrets displayed')
    print('Data directory:', data)
    show((root / 'app.py').is_file(), 'Main Streamlit application', serious=True)
    show((root / 'feature_engine.py').is_file(), 'Original feature_engine.py is available', serious=True)
    show((root / 'rns_scraper.py').is_file(), 'Optional RNS news integration is available')
    show((data / 'models' / 'ensemble_ranker.joblib').is_file() or
         (data / 'models' / 'ensemble_live.joblib').is_file(), 'Original or promoted ML model is available', serious=True)

    secrets_file = root / '.streamlit' / 'secrets.toml'
    if secrets_file.is_file():
        try:
            with secrets_file.open('rb') as fh:
                secrets = tomllib.load(fh)
            has_user = bool(secrets.get('AUTH_USER'))
            has_hash = bool(secrets.get('AUTH_PASS_ARGON2ID') or secrets.get('AUTH_PASS_HASH'))
            show(has_user and has_hash, 'Local login settings present (values not shown)', serious=True)
        except (OSError, ValueError):
            show(False, 'Local secrets file is unreadable or malformed', serious=True)
    else:
        show(False, 'Local Streamlit secrets file (needed for laptop login)', serious=True)

    db = data / 'lse_market_data.duckdb'
    if not db.is_file():
        show(False, 'No local trade history file; first run or Supabase/cloud records may be elsewhere')
    else:
        try:
            import duckdb
            con = duckdb.connect(str(db), read_only=True)
            try:
                shares = con.execute('SELECT COUNT(*) FROM trade_journal').fetchone()[0]
                options = con.execute('SELECT COUNT(*) FROM daily_options_journal').fetchone()[0]
            finally:
                con.close()
            show(True, f'Local journal readable: {shares} equity trades, {options} option ideas')
        except Exception as exc:
            # DuckDB can deny access if the app is running. Do not call that corruption.
            show(False, f'Local journal cannot be inspected ({type(exc).__name__}); stop the app and retry')

    archives = sorted((data / 'backups').glob('alpha_backup_*.zip')) if (data / 'backups').exists() else []
    if archives:
        newest = max(archives, key=lambda p: p.stat().st_mtime)
        age_days = (datetime.now(timezone.utc).timestamp() - newest.stat().st_mtime) / 86400
        show(age_days <= 7, f'{len(archives)} local backup(s); newest is {age_days:.1f} days old')
        if args.verify_backup:
            try:
                m = verify_backup(newest)
                show(True, f'Latest backup checksums valid ({len(m["files"])} files)')
            except Exception as exc:
                show(False, f'Backup verification failed ({type(exc).__name__})', serious=True)
    else:
        show(False, 'No local ZIP backup yet; create one before changing laptops')

    print('Reminder: an independent encrypted external backup is essential; copies on the same laptop are not enough.')
    print('Reminder: this does not test live prices, model quality, or cloud connectivity.')
    return 1 if args.strict and issues else 0


if __name__ == '__main__':
    sys.exit(main())
