"""ALPHA-LSE background *research* worker. No broker orders and no paper fills.

Launched by GitHub Actions (or a trusted scheduler); never import app.py here.
All state stays in private Supabase Storage. Does not delete prior objects.
Fails closed when prices, permissions, calendar, or remote persistence are suspect.
"""
from __future__ import annotations

import argparse
from datetime import date, datetime, timedelta, time, timezone
import hashlib
import io
import json
import logging
import math
import os
import time as sleep_time
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from commodity_etps import COMMODITY_ETPS
from commodity_ml import (CommodityMLStore, PRODUCTS, HORIZON, GAIN_THRESHOLD,
                          clean_completed_bars, latest_forecast, train_or_update)
from commodity_signals import evaluate_commodity_signal

LOG = logging.getLogger('alpha_lse.worker')
LONDON = ZoneInfo('Europe/London')
STATE_KEY = 'commodity_ml_champion_v1.json'
LEDGER_PREFIX = 'research/predictions'
HEALTH_KEY = 'research/worker_health_v1.json'
LATEST_KEY = 'research/latest_forecasts_v1.json'
MARKET_KEY = 'research/market_snapshot_v1.json'
MAX_BYTES = 4_000_000


def london_now():
    return datetime.now(LONDON)


def calendar_session(day: date) -> bool:
    """Prefer actual XLON exchange sessions; never trade on estimated weekdays."""
    import exchange_calendars as xcals
    cal = xcals.get_calendar('XLON')
    return bool(cal.is_session(pd.Timestamp(day.isoformat())))


def is_after_close(now: datetime) -> bool:
    import exchange_calendars as xcals
    cal = xcals.get_calendar('XLON')
    day = now.astimezone(LONDON).date()
    if not calendar_session(day):
        return False
    close = cal.session_close(pd.Timestamp(day.isoformat())).to_pydatetime()
    # 35-min buffer for delayed Yahoo daily close; worker still checks latest bar date.
    return now.astimezone(timezone.utc) >= close + timedelta(minutes=35)


def normalize_lse_bars(bars: pd.DataFrame, currency: str, ticker: str):
    if currency in ('GBp', 'GBX'):
        factor = 1.0
    elif currency == 'GBP':
        factor = 100.0
    else:
        raise ValueError(f'{ticker}: quote currency not verified as sterling')
    if bars is None or bars.empty or not {'Open','High','Low','Close','Volume'}.issubset(bars.columns):
        raise ValueError(f'{ticker}: missing verified OHLCV')
    x = bars.copy()
    if not isinstance(x.index, pd.DatetimeIndex) or x.index.tz is None:
        raise ValueError(f'{ticker}: price dates lack timezone')
    x.index = x.index.tz_convert(LONDON).tz_localize(None).normalize()
    x = x[~x.index.duplicated(keep='last')].sort_index()
    for k in ('Open','High','Low','Close'):
        x[k] = pd.to_numeric(x[k], errors='coerce') * factor
    x['Volume'] = pd.to_numeric(x['Volume'], errors='coerce')
    arr = x[['Open','High','Low','Close','Volume']]
    ok = (np.isfinite(arr).all(axis=1) & (arr[['Open','High','Low','Close']] > 0).all(axis=1)
          & (arr['Volume'] >= 0)
          & (arr['High'] >= arr[['Open','Close','Low']].max(axis=1))
          & (arr['Low'] <= arr[['Open','Close','High']].min(axis=1)))
    if len(x) and float(ok.mean()) < .97:
        raise ValueError(f'{ticker}: too many rejected price bars')
    x = x.loc[ok].copy()
    if len(x) < 120:
        raise ValueError(f'{ticker}: insufficient good daily bars')
    x.attrs['price_currency'] = 'GBp'
    x.attrs['quote_currency_original'] = currency
    return x


def fetch_histories(period='5y'):
    import yfinance as yf
    output = {}
    failures = {}
    for ticker in PRODUCTS:
        try:
            stock = yf.Ticker(ticker)
            bars = stock.history(period=period, auto_adjust=False, repair=False, timeout=16)
            meta = getattr(stock, 'history_metadata', None)
            cur = meta.get('currency') if isinstance(meta, dict) else None
            if not cur:
                cur = stock.fast_info.get('currency')
            output[ticker] = normalize_lse_bars(bars, cur, ticker)
        except Exception as e:
            failures[ticker] = type(e).__name__
            LOG.warning('Data invalid for %s (%s)', ticker, type(e).__name__)
    return output, failures


def json_bytes(obj) -> bytes:
    data = json.dumps(obj, allow_nan=False, separators=(',', ':'), sort_keys=True).encode('utf-8')
    if len(data) > MAX_BYTES:
        raise ValueError('JSON state exceeds permitted size')
    return data


def not_found(exc) -> bool:
    status = str(getattr(exc, 'status_code', '') or getattr(exc, 'statusCode', ''))
    code = str(getattr(exc, 'code', '')).lower()
    s = str(exc).lower()
    return status == '404' or code in ('404','not_found','no_such_key') or 'not found' in s or 'does not exist' in s


def read_json(bucket, key, default=None):
    try:
        raw = bytes(bucket.download(key))
    except Exception as exc:
        if not_found(exc):
            return default
        raise RuntimeError('Cloud read failed: ' + key) from exc
    if len(raw) > MAX_BYTES:
        raise RuntimeError('Oversized cloud record')
    return json.loads(raw.decode('utf-8'))


def save_json(bucket, key, value, *, immutable=False):
    payload = json_bytes(value)
    old = read_json(bucket, key)
    if immutable and old is not None:
        if json_bytes(old) != payload:
            raise RuntimeError('Immutable forecast cannot be overwritten: ' + key)
        return False
    if old is not None and json_bytes(old) == payload:
        return False
    options = {'content-type': 'application/json'}
    try:
        if old is None:
            bucket.upload(key, payload, file_options=options)
        else:
            bucket.update(key, payload, file_options=options)
    except Exception as exc:
        # Upsert race: only accept exact content readback, never assume success.
        check = read_json(bucket, key)
        if check is None or json_bytes(check) != payload:
            raise RuntimeError('Cloud write not verified: ' + key) from exc
    if json_bytes(read_json(bucket, key)) != payload:
        raise RuntimeError('Cloud readback mismatch: ' + key)
    return True


def connection_from_env():
    from supabase import create_client
    url = os.environ.get('SUPABASE_URL', '').strip()
    key = os.environ.get('SUPABASE_MODEL_SERVICE_KEY', '').strip()
    name = os.environ.get('ALPHA_COMMODITY_ML_BUCKET', 'alpha-lse-models').strip()
    if not (url.startswith('https://') and key and name):
        raise RuntimeError('Missing server-side Supabase secrets')
    url = url.split('/rest/v1')[0].rstrip('/')
    client = create_client(url, key)
    return client, client.storage.from_(name)


def future_label(bars: pd.DataFrame, day: str):
    """Record outcome only after five genuinely completed sessions after the forecast date."""
    try:
        day = pd.Timestamp(day)
        closes = bars['Close'].loc[bars.index >= day]
        if not len(closes) or closes.index[0] != day or len(closes) <= HORIZON:
            return None
        # Require exactly the next five *exchange* sessions, not the fifth
        # available vendor row after an outage (which would shift the label).
        import exchange_calendars as xcals
        cal = xcals.get_calendar('XLON')
        expected = cal.sessions_in_range(day, closes.index[HORIZON])
        if (len(expected) != HORIZON + 1
                or [pd.Timestamp(v).date() for v in expected]
                   != [v.date() for v in closes.index[:HORIZON+1]]):
            return None
        future = (float(closes.iloc[HORIZON]) / float(closes.iloc[0])) - 1
        return {'actual_five_session_pct': round(100 * future, 4),
                'event_happened': bool(future > GAIN_THRESHOLD),
                'outcome_date': closes.index[HORIZON].date().isoformat()}
    except (KeyError, ValueError, TypeError, IndexError):
        return None


def ledger_key(day: str):
    return f'{LEDGER_PREFIX}/{str(day)[:4]}.json'


def append_forecasts(bucket, forecasts):
    """Append-only dated forecasts, separate from mutable outcomes."""
    grouped = {}
    for f in forecasts:
        grouped.setdefault(f['date'][:4], []).append(f)
    added = 0
    for yr, items in grouped.items():
        key = f'{LEDGER_PREFIX}/{yr}.json'
        doc = read_json(bucket, key, default={'schema': 1, 'forecasts': []})
        if doc.get('schema') != 1 or not isinstance(doc.get('forecasts'), list):
            raise RuntimeError('Invalid forecast ledger schema')
        rows = {str(item['date']) + ':' + str(item['ticker']): item for item in doc['forecasts']}
        for item in items:
            id_ = item['date']+':'+item['ticker']
            if id_ in rows:
                if rows[id_] != item:
                    raise RuntimeError('Existing forecast immutable; refusing to revise it')
                continue
            rows[id_] = item
            added += 1
        doc['forecasts'] = [rows[k] for k in sorted(rows)]
        save_json(bucket, key, doc)
    return added


def collect_outcomes(bucket, histories, now_date: date):
    changed = 0
    checked = 0
    for yr in (str(now_date.year - 1), str(now_date.year)):
        data = read_json(bucket, f'{LEDGER_PREFIX}/{yr}.json')
        if not data:
            continue
        outcome_key = f'research/outcomes/{yr}.json'
        out = read_json(bucket, outcome_key, default={'schema':1,'outcomes':[]})
        if out.get('schema') != 1 or not isinstance(out.get('outcomes'), list):
            raise RuntimeError('Invalid outcome ledger')
        existing = {r['date']+':'+r['ticker']:r for r in out['outcomes']}
        before_count = len(existing)
        for row in data.get('forecasts', []):
            k = row['date']+':'+row['ticker']
            if k in existing or row['ticker'] not in histories:
                continue
            # At the after-close scheduled run, today's OHLC is completed.
            result = future_label(histories[row['ticker']], row['date'])
            if result:
                existing[k] = {'date':row['date'], 'ticker':row['ticker'],
                               'probability':row['probability'], 'baseline_probability':row['baseline_probability'],
                               'validated':row['validated'], **result}
                changed += 1
        out['outcomes'] = [existing[k] for k in sorted(existing)]
        checked += len(out['outcomes'])
        if len(out['outcomes']) > before_count:
            save_json(bucket, outcome_key, out)
    return changed, checked


def historical_scorecard(bucket, today: date):
    data = []
    for yr in (str(y) for y in range(2026, today.year + 1)):
        chunk = read_json(bucket, f'research/outcomes/{yr}.json')
        if chunk:
            data.extend(chunk.get('outcomes', []))
    if not data:
        return {'matured': 0, 'message':'Waiting for five completed sessions'}
    y = np.array([int(x['event_happened']) for x in data], dtype=float)
    p = np.array([float(x['probability']) for x in data], dtype=float)
    b = np.array([float(x['baseline_probability']) for x in data], dtype=float)
    details = []
    for ticker in PRODUCTS:
        one = [r for r in data if r['ticker'] == ticker]
        if not one:
            continue
        yt = np.array([int(r['event_happened']) for r in one])
        pt = np.array([float(r['probability']) for r in one])
        bt = np.array([float(r['baseline_probability']) for r in one])
        # Count flagged moves only if the saved probability crosses the
        # immutable model-at-forecast threshold; never invent a 70% cutoff.
        threshold_by_date = {}
        for yr in sorted({r['date'][:4] for r in one}):
            ledger = read_json(bucket, f'{LEDGER_PREFIX}/{yr}.json') or {}
            threshold_by_date.update({r['date']+':'+r['ticker']:float(r.get('threshold',1.01))
                                      for r in ledger.get('forecasts', [])})
        flag = np.array([float(r['probability']) >= threshold_by_date.get(r['date']+':'+r['ticker'],1.01)
                         for r in one])
        hits = int(np.sum(flag & (yt==1)))
        details.append({'ticker':ticker,'matured':len(one),'actual_rises':int(yt.sum()),
                        'predicted_rises':int(flag.sum()),'correct_predicted_rises':hits,
                        'signal_precision_pct':round(100*hits/int(flag.sum()),1) if flag.sum() else None,
                        'brier':round(float(np.mean((yt-pt)**2)),4),
                        'baseline_brier':round(float(np.mean((yt-bt)**2)),4)})
    return {'matured':len(data), 'brier':round(float(np.mean((y-p)**2)),5),
            'baseline_brier':round(float(np.mean((y-b)**2)),5),
            'last_outcome_date':max(x['outcome_date'] for x in data),
            'by_product': details,
            'research_only':True}


def alert(message):
    """Optional Slack/Discord-compatible webhook; never send API keys or exception bodies."""
    url = os.environ.get('ALPHA_ALERT_WEBHOOK_URL','').strip()
    if not url:
        return
    import requests
    try:
        payload = {'content': 'ALPHA-LSE worker: ' + message[:300]} if 'discord.com/api/webhooks/' in url else {'text': 'ALPHA-LSE worker: ' + message[:300]}
        resp = requests.post(url, json=payload, timeout=8)
        resp.raise_for_status()
    except Exception:
        LOG.error('Alert webhook failed (details suppressed)')


def save_health(bucket, mode, success, today, *, detail=None):
    previous = read_json(bucket, HEALTH_KEY, default={'schema': 1})
    out = dict(previous)
    out['schema'] = 1
    out['last_'+mode+'_utc'] = datetime.now(timezone.utc).isoformat()
    out['last_'+mode+'_status'] = 'ok' if success else 'failed'
    if detail:
        out['last_'+mode+'_detail'] = detail
    save_json(bucket, HEALTH_KEY, out)


def daily_worker(client, bucket, now):
    local = now.astimezone(LONDON)
    if not is_after_close(now):
        LOG.info('Not an after-close XLON trading session; daily ML skipped')
        return {'status':'not_after_close'}
    histories, failures = fetch_histories('5y')
    if len(histories) < 3:
        raise RuntimeError('Fewer than three verified ETC price histories')
    # Fail closed on a stale Yahoo daily bar. Never label an incomplete session.
    required_date = local.date().isoformat()
    fresh = {t:b for t,b in histories.items() if b.index[-1].date().isoformat() == required_date}
    if len(fresh) < 3:
        raise RuntimeError('Fewer than three ETCs have a completed bar dated today')
    histories = fresh
    store = CommodityMLStore('_unused.json', cloud_mode=True, bucket=bucket)
    old = store.load()
    # After official close today's last completed bar is usable: pass tomorrow
    # to the trainer, which excludes bars >=today argument.
    effective_day = local.date() + timedelta(days=1)
    proposed, report, should_save = train_or_update(histories, old, effective_day)
    if should_save:
        store.save(proposed)  # Must persist before any model promotion is visible.
    else:
        proposed = old
    model = proposed.get('champion') or proposed.get('research_model')
    forecasts = []
    if model:
        validated = bool(proposed.get('champion') and not proposed.get('champion_paused'))
        version = str(model.get('version', 'research'))
        for t, bars in histories.items():
            try:
                f = latest_forecast(model, t, bars, effective_day)
                if f['bar_date'] != required_date:
                    continue
                forecasts.append({'date':f['bar_date'], 'ticker':t, 'probability':f['probability'],
                                  'threshold':f['threshold'], 'predicted_rise':f['qualified'],
                                  'baseline_probability':round(float(model['event_base_rate']),4),
                                  'validated':validated,'model_version':version,
                                  'event':f['event']})
            except (ValueError, TypeError, KeyError):
                LOG.warning('No ML forecast for %s', t)
    added = append_forecasts(bucket, forecasts) if forecasts else 0
    save_json(bucket, LATEST_KEY, {'schema':1, 'as_of':required_date,'forecasts':forecasts,
                                    'status':str(report.get('status','Model unchanged'))})
    matured, total = collect_outcomes(bucket, histories, local.date())
    score = historical_scorecard(bucket, local.date())
    save_json(bucket, 'research/scorecard_v1.json', {'schema':1,'as_of':required_date, **score})
    # Forward-dated research scores are not a buy strategy backtest. If a
    # validated champion develops a strong record of poor probability estimates,
    # stop NEW ML-filtered orders until the next valid evaluation can recover.
    # This is a negative-only safety brake; a good score never promotes a model.
    if (proposed.get('champion') and score.get('matured', 0) >= 150
            and score.get('brier', 1) > score.get('baseline_brier', 0) + .025):
        if not proposed.get('champion_paused'):
            proposed['champion_paused'] = True
            proposed['forward_score_pause_day'] = required_date
            store.save(proposed)
            alert('Validated commodity model paused: forward Brier score deteriorated')
    # Archive ALL actual independent evaluation decisions (including rejected
    # challengers) so a ten-year research history isn't silently overwritten.
    previous_test = (old.get('last_evaluation') or old.get('report') or {}).get('test_through')
    fresh_test = (proposed.get('last_evaluation') or {}).get('test_through')
    if fresh_test and fresh_test != previous_test:
        save_json(bucket, f'backups/model_trials/{required_date}.json', {
            'schema':1, 'as_of':required_date, 'report':proposed['last_evaluation'],
            'champion_version':(proposed.get('champion') or {}).get('version'),
            'champion_paused':bool(proposed.get('champion_paused'))}, immutable=True)
    if proposed.get('champion'):
        # Immutable day-stamped safe-JSON snapshot; no executable pickle.
        save_json(bucket, f'backups/model_states/{required_date}.json', proposed, immutable=True)
    detail = {'data_tickers':len(histories),'missing_tickers':list(failures),
              'forecasts_saved':added,'outcomes_saved':matured, 'model_status':str(report.get('status',''))[:120]}
    save_health(bucket, 'daily', True, local.date(), detail=detail)
    LOG.info('Daily research completed: %s', detail)
    return detail


def monitor_worker(client, bucket, now):
    local = now.astimezone(LONDON)
    if not calendar_session(local.date()):
        return {'status':'exchange_closed'}
    if not time(8,0) <= local.time() <= time(16,45):
        return {'status':'outside_market_hours'}
    histories, failures = fetch_histories('6mo')
    if len(histories) < 3:
        raise RuntimeError('Background scanner cannot verify three current ETC feeds')
    cards = []
    for ticker in PRODUCTS:
        b = histories.get(ticker)
        if b is None:
            cards.append({'ticker':ticker, 'status':'unavailable', 'reason': failures.get(ticker,'Unknown')})
            continue
        today = local.date()
        complete = b[b.index.date < today]
        q = b.tail(1)
        if q.empty or q.index[-1].date() != today or len(complete) < 65:
            cards.append({'ticker':ticker, 'status':'wait', 'reason':'No verified current-session bar'})
            continue
        sig = evaluate_commodity_signal(complete, float(q['Close'].iloc[-1]), min_turnover_gbp=250000)
        cards.append({'ticker':ticker,'status':sig.action, 'trend_score':sig.score,
                      'reason':sig.reason,'reference_price_pence':round(float(q['Close'].iloc[-1]),4),
                      'indicative_target_pence':sig.target_pence, 'indicative_stop_pence':sig.stop_pence,
                      'quote_day':today.isoformat(), 'research_only':True})
    snapshot = {'schema':1,'as_of_utc':now.astimezone(timezone.utc).isoformat(),
                'date':local.date().isoformat(),'data_errors':failures,'signals':cards,
                'note':'Informational background scan only. No paper orders are placed by this worker.'}
    save_json(bucket, MARKET_KEY, snapshot)
    save_health(bucket, 'monitor', True, local.date(), detail={'verified':sum(x.get('status') not in ('unavailable','wait') for x in cards)})
    return snapshot


def health_worker(client, bucket, now):
    obj = read_json(bucket, HEALTH_KEY)
    if not obj or not obj.get('last_daily_utc'):
        raise RuntimeError('Worker has no successful daily heartbeat yet')
    last = datetime.fromisoformat(obj['last_daily_utc'])
    delta = now.astimezone(timezone.utc) - last.astimezone(timezone.utc)
    # Calendar holidays/weekends need not generate pointless page alerts.
    if delta > timedelta(days=5):
        raise RuntimeError('Worker daily heartbeat is over five days old')
    return {'last_daily_utc':last.isoformat(),'age_hours':round(delta.total_seconds()/3600,1)}


def backup_worker(client, bucket, now):
    """Durable, dated, compressed Supabase trade-journal snapshot, read-only source."""
    import gzip
    rows, offset = [], 0
    while True:
        batch = client.table('predictions').select('*').range(offset,offset+999).execute().data
        if batch is None:
            raise RuntimeError('Cloud trade history read returned no acknowledgement')
        rows.extend(batch)
        if len(batch) < 1000:
            break
        offset += 1000
        if offset > 100_000:
            raise RuntimeError('Paper-journal backup exceeds safety limit')
    raw = json.dumps({'schema':1,'source':'Supabase predictions','rows':rows,
                      'backed_up_utc':now.astimezone(timezone.utc).isoformat()},
                     allow_nan=False, separators=(',', ':'), default=str).encode('utf-8')
    if len(raw) > 100_000_000:
        raise RuntimeError('Backup too large; needs archival migration')
    compressed = gzip.compress(raw)
    name = f'backups/paper_journal/{now.astimezone(LONDON).date().isoformat()}.json.gz'
    try:
        old = bytes(bucket.download(name))
    except Exception as e:
        if not_found(e):
            old = None
        else:
            raise RuntimeError('Backup read failed') from e
    if old is None:
        bucket.upload(name, compressed, file_options={'content-type':'application/gzip'})
        check = compressed
    else:
        # A retry never overwrites the first durable snapshot of the day.
        check = old
    if bytes(bucket.download(name)) != check:
        raise RuntimeError('Journal backup verification failed')
    if old is not None:
        # Report what was actually saved, not a new unsaved hypothetical copy.
        raw = gzip.decompress(old)
        old_doc = json.loads(raw.decode('utf-8'))
        rows = old_doc['rows']
    out = {'date':now.astimezone(LONDON).date().isoformat(),'rows':len(rows),
           'sha256':hashlib.sha256(raw).hexdigest(),'compressed_bytes':len(check)}
    save_health(bucket, 'backup', True, now.date(),detail=out)
    return out


def check_worker(client, bucket, now):
    """Safe one-click connection test; can run at any time, market open or shut."""
    stamp = now.astimezone(timezone.utc).isoformat()
    rec = {'schema':1,'checked_utc':stamp,'status':'connected',
           'note':'Private bucket authenticated. Does not verify future schedules.'}
    save_json(bucket,'research/connection_check_v1.json', rec)
    save_health(bucket, 'check', True, now.astimezone(LONDON).date())
    return {'status':'connected','readback_verified':True}


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['check','daily','monitor','health','backup'], required=True)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    now = datetime.now(timezone.utc)
    client, bucket = connection_from_env()
    try:
        op = {'check':check_worker,'daily':daily_worker,'monitor':monitor_worker,
              'health':health_worker,'backup':backup_worker}[args.mode]
        result = op(client, bucket, now)
        LOG.info('Worker result: %s', str(result)[:600])
        return 0
    except Exception as exc:
        safe = args.mode + ' failed: ' + type(exc).__name__
        LOG.error('Worker failed in %s (%s); inspect provider status and server configuration', args.mode, type(exc).__name__)
        alert(safe)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
