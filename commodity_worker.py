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
import secrets
import time as sleep_time
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from commodity_etps import COMMODITY_ETPS
from commodity_ml import (MODEL_SCHEMA, PRODUCTS, HORIZON, GAIN_THRESHOLD,
                          clean_completed_bars, latest_forecast, train_or_update, validate_model)
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


def latest_verified_closed_session(now: datetime) -> date | None:
    """Most recent XLON session whose *official* close + feed buffer has passed.

    A GitHub schedule can be delayed past midnight. The research date must be
    derived from the exchange calendar, NOT from the runner's local date.
    Vendor OHLCV dates are still verified separately before any forecast save.
    """
    import exchange_calendars as xcals
    if now.tzinfo is None:
        raise ValueError('Scheduler timestamp must be timezone-aware')
    cal = xcals.get_calendar('XLON')
    local_day = now.astimezone(LONDON).date()
    sessions = cal.sessions_in_range(
        pd.Timestamp(local_day - timedelta(days=14)), pd.Timestamp(local_day))
    cutoff = now.astimezone(timezone.utc) - timedelta(minutes=35)
    for session in reversed(sessions):
        close = cal.session_close(session).to_pydatetime()
        if close <= cutoff:
            return session.date()
    return None


def completed_history_through(bars: pd.DataFrame, session: date) -> pd.DataFrame:
    """Exclude every later bar, including an unfinished current-session bar.

    Crucial to prevent a delayed scheduler from accidentally training on a
    newer partial day's quotes when issuing an older as-of close forecast.
    """
    if not isinstance(bars.index, pd.DatetimeIndex):
        raise ValueError('Historical price index is not dated')
    return bars.loc[bars.index.normalize() <= pd.Timestamp(session)].copy()


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
        raise ValueError(f'{ticker}: too many rejected price bars ({int((~ok).sum())} of {len(x)} rows)')
    x = x.loc[ok].copy()
    if len(x) < 120:
        raise ValueError(f'{ticker}: insufficient good daily bars')
    x.attrs['price_currency'] = 'GBp'
    x.attrs['quote_currency_original'] = currency
    return x


# Only these locally generated validation errors are safe to show in logs.
# Never print raw Yahoo error bodies, credentials, or HTTP request URLs.
SAFE_PRICE_FAILURES = (
    'quote currency not verified as sterling',
    'missing verified OHLCV',
    'price dates lack timezone',
    'too many rejected price bars',
    'insufficient good daily bars',
)


def price_failure_reason(exc: Exception) -> str:
    if isinstance(exc, ValueError):
        message = str(exc)
        for reason in SAFE_PRICE_FAILURES:
            if reason in message:
                # The diagnostic is constructed by our own normaliser.
                if reason == 'too many rejected price bars':
                    import re
                    match = re.search(r'\((\d+) of (\d+) rows\)', message)
                    return f'{reason} ({match.group(1)}/{match.group(2)})' if match else reason
                return reason
    return type(exc).__name__


def rejected_bar_diagnostics(bars: pd.DataFrame) -> str:
    """Report *which* vendor OHLCV checks failed, never repair values here.

    Counts are diagnostic only; the strict normalizer still decides which
    historical feeds are acceptable. Do not log prices or untrusted error bodies.
    """
    keys = ('Open', 'High', 'Low', 'Close', 'Volume')
    if not isinstance(bars, pd.DataFrame) or not set(keys).issubset(bars.columns):
        return 'OHLCV columns missing'
    x = bars[list(keys)].apply(pd.to_numeric, errors='coerce')
    finite = pd.Series(np.isfinite(x).all(axis=1), index=x.index)
    positive = (x[['Open', 'High', 'Low', 'Close']] > 0).all(axis=1)
    volume = x['Volume'] >= 0
    high = x['High'] >= x[['Open', 'Close', 'Low']].max(axis=1)
    low = x['Low'] <= x[['Open', 'Close', 'High']].min(axis=1)
    bad = ~(finite & positive & volume & high & low)
    return (f'invalid={int(bad.sum())}/{len(x)}; '
            f'nonfinite={int((~finite).sum())}; '
            f'nonpositive_price={int((~positive).sum())}; '
            f'negative_volume={int((~volume).sum())}; '
            f'high_below_price={int((~high).sum())}; '
            f'low_above_price={int((~low).sum())}')


def can_retry_short_history(exc: Exception) -> bool:
    # A shorter, independently validated history can bypass older corrupted bars.
    # DO NOT retry a non-sterling or unverifiable currency with a guessed unit.
    return isinstance(exc, ValueError) and any(term in str(exc) for term in (
        'too many rejected price bars', 'insufficient good daily bars',
        'missing verified OHLCV', 'price dates lack timezone'))


def fetch_histories(period='5y'):
    """Prefer independently verified raw OHLCV; try yfinance repair if all fail.

    Yahoo's repair algorithm is NOT trusted as ground truth: repaired OHLCV
    must pass the identical strict GBP/pence, price, volume and history checks.
    Unknown currencies or provider errors are never guessed or bypassed.
    """
    import yfinance as yf
    output, failures = {}, {}
    for ticker in PRODUCTS:
        periods = (period, '3y', '2y', '1y') if period == '5y' else (period,)
        try:
            stock = yf.Ticker(ticker)
        except Exception as exc:
            failures[ticker] = type(exc).__name__
            LOG.warning('ETC %s: Yahoo ticker setup failed (%s)', ticker, failures[ticker])
            continue
        found = False
        stop = False
        for repaired in (False, True):
            for i, candidate_period in enumerate(periods):
                bars = None
                try:
                    bars = stock.history(period=candidate_period, auto_adjust=False,
                                         repair=repaired, timeout=16)
                    meta = getattr(stock, 'history_metadata', None)
                    cur = meta.get('currency') if isinstance(meta, dict) else None
                    if not cur:
                        cur = stock.fast_info.get('currency')
                    cleaned = normalize_lse_bars(bars, cur, ticker)
                    output[ticker] = cleaned
                    failures.pop(ticker, None)
                    found = True
                    if repaired:
                        LOG.info('ETC %s: Yahoo-repaired %s OHLCV passed independent strict checks',
                                 ticker, candidate_period)
                    elif i:
                        LOG.info('ETC %s: verified %s raw history after older bars failed checks',
                                 ticker, candidate_period)
                    break
                except Exception as exc:
                    reason = price_failure_reason(exc)
                    failures[ticker] = reason
                    diagnostic = ''
                    if isinstance(exc, ValueError) and 'too many rejected price bars' in str(exc):
                        try:
                            diagnostic = '; ' + rejected_bar_diagnostics(bars)
                        except (TypeError, ValueError, KeyError):
                            pass
                    retryable = can_retry_short_history(exc)
                    LOG.warning('ETC %s (%s, repair=%s): %s%s',
                                ticker, candidate_period, repaired, reason, diagnostic)
                    if not retryable:
                        stop = True
                        break
            if found or stop:
                break
            if not repaired:
                LOG.info('ETC %s: no raw history validated; testing Yahoo price repair with strict revalidation', ticker)
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


def _fresh_private_json(key):
    """Uncached, authenticated private Storage GET for worker read/write checks.

    Supabase Storage may temporarily serve old bytes after an overwrite.  The
    nonce makes each read a distinct cache key without exposing service keys in
    a URL.  This is only used in the trusted server-side worker environment.
    """
    from urllib.parse import quote
    import requests

    root = os.environ['SUPABASE_URL'].strip().split('/rest/v1')[0].rstrip('/')
    token = os.environ['SUPABASE_MODEL_SERVICE_KEY'].strip()
    bucket = os.environ.get('ALPHA_COMMODITY_ML_BUCKET', 'alpha-lse-models').strip()
    if not (root.startswith('https://') and token and bucket):
        raise RuntimeError('Fresh cloud read configuration is invalid')
    path = f'{quote(bucket, safe="")}/{quote(key, safe="/")}'
    try:
        r = requests.get(
            root + '/storage/v1/object/authenticated/' + path,
            headers={'apikey': token, 'Authorization': 'Bearer ' + token,
                     'Cache-Control': 'no-cache'},
            params={'cacheNonce': secrets.token_hex(12)}, timeout=20)
    except requests.RequestException as exc:
        raise RuntimeError('Fresh cloud read failed for ' + key) from exc
    if r.status_code == 404:
        return None
    if r.status_code == 400 and ('not found' in r.text.lower() or 'does not exist' in r.text.lower()):
        return None
    if r.status_code != 200:
        # No response body or HTTP headers: avoid accidentally logging secrets.
        raise RuntimeError(f'Fresh cloud read failed for {key} (HTTP {r.status_code})')
    if len(r.content) > MAX_BYTES:
        raise RuntimeError('Oversized cloud record: ' + key)
    try:
        return json.loads(r.content.decode('utf-8'))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError('Invalid JSON in cloud record: ' + key) from exc


def read_json(bucket, key, default=None, *, fresh=False):
    if (fresh and os.environ.get('SUPABASE_URL', '').startswith('https://')
            and os.environ.get('SUPABASE_MODEL_SERVICE_KEY', '').strip()):
        result = _fresh_private_json(key)
        return default if result is None else result
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
    # Read mutable object state from origin rather than a cached older version.
    old = read_json(bucket, key, fresh=True)
    if immutable and old is not None:
        if json_bytes(old) != payload:
            raise RuntimeError('Immutable forecast cannot be overwritten: ' + key)
        return False
    if old is not None and json_bytes(old) == payload:
        return False
    options = {'content-type': 'application/json', 'cache-control': '0'}
    write_exc = None
    try:
        if old is None:
            bucket.upload(key, payload, file_options=options)
        else:
            bucket.update(key, payload, file_options=options)
    except Exception as exc:
        # Upsert races must still be checked, not silently treated as success.
        write_exc = exc
    # CDN invalidation may lag successful PUT, so use uncached requests and
    # bounded retries.  Never report success if the expected bytes don't appear.
    for attempt in range(6):
        try:
            check = read_json(bucket, key, fresh=True)
        except RuntimeError:
            if attempt == 5:
                raise
            check = None
        if check is not None and json_bytes(check) == payload:
            return True
        if attempt < 5:
            sleep_time.sleep(min(0.5 * (2 ** attempt), 4.0))
    if write_exc:
        raise RuntimeError('Cloud write not verified: ' + key) from write_exc
    raise RuntimeError('Cloud readback mismatch after retries: ' + key)



def read_worker_model_state(bucket) -> dict:
    """Load the durable model from an authenticated uncached storage read.

    The generic model store uses SDK downloads, which may be CDN-stale directly
    after an overwrite. A worker must not train against stale champion state.
    """
    state = read_json(bucket, STATE_KEY, default={}, fresh=True)
    if not isinstance(state, dict):
        raise ValueError('Cloud commodity model state is not a JSON object')
    if not state:
        return {}
    if state.get('schema') != MODEL_SCHEMA:
        raise ValueError('Incompatible cloud commodity model state schema')
    if len(json_bytes(state)) > 250_000:
        raise ValueError('Oversized commodity model state')
    for role in ('champion', 'research_model'):
        if state.get(role):
            validate_model(state[role])
    return state


def save_worker_model_state(bucket, state: dict) -> None:
    """Persist an ML state with validation and bounded uncached readback.

    Unlike CommodityMLStore.save's immediate SDK download, save_json retries
    non-cached, authenticated reads. A failed verification remains a hard error:
    the caller must NEVER use an unconfirmed new champion for paper decisions.
    """
    if not isinstance(state, dict) or state.get('schema') != MODEL_SCHEMA:
        raise ValueError('Invalid commodity ML state')
    if len(json_bytes(state)) > 250_000:
        raise ValueError('Oversized commodity model state')
    for role in ('champion', 'research_model'):
        if state.get(role):
            validate_model(state[role])
    save_json(bucket, STATE_KEY, state)


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
    previous = read_json(bucket, HEALTH_KEY, default={'schema': 1}, fresh=True)
    out = dict(previous)
    out['schema'] = 1
    out['last_'+mode+'_utc'] = datetime.now(timezone.utc).isoformat()
    out['last_'+mode+'_status'] = 'ok' if success else 'failed'
    if detail:
        out['last_'+mode+'_detail'] = detail
    save_json(bucket, HEALTH_KEY, out)


def daily_worker(client, bucket, now):
    local = now.astimezone(LONDON)
    research_day = latest_verified_closed_session(now)
    if research_day is None:
        LOG.info('No completed XLON session eligible for daily ML yet')
        return {'status':'no_verified_completed_session'}
    LOG.info('Commodity daily target session: %s (worker ran %s)',
             research_day.isoformat(), now.astimezone(timezone.utc).isoformat())
    histories, failures = fetch_histories('5y')
    if len(histories) < 3:
        raise RuntimeError('Fewer than three verified ETC price histories; accepted=%s, rejected=%s' % (sorted(histories), failures))
    # Fail closed on a stale Yahoo daily bar. Never label an incomplete session.
    required_date = research_day.isoformat()
    histories = {t: completed_history_through(b, research_day) for t, b in histories.items()}
    fresh = {t:b for t,b in histories.items() if len(b) and b.index[-1].date() == research_day}
    if len(fresh) < 3:
        raise RuntimeError('Fewer than three ETCs have a completed bar for target XLON session ' + required_date + '; latest accepted bars: ' + str({t: b.index[-1].date().isoformat() if len(b) else 'none' for t,b in histories.items()}))
    histories = fresh
    old = read_worker_model_state(bucket)
    # After the target session's official close, its completed bar is usable.
    # Pass the following date so the trainer excludes all bars >= that date.
    effective_day = research_day + timedelta(days=1)
    proposed, report, should_save = train_or_update(histories, old, effective_day)
    if should_save:
        save_worker_model_state(bucket, proposed)  # Persist before any model promotion is visible.
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
    if not forecasts:
        raise RuntimeError('No commodity forecasts generated for completed session ' + required_date)
    added = append_forecasts(bucket, forecasts)
    save_json(bucket, LATEST_KEY, {'schema':1, 'as_of':required_date,
                                    'processed_utc':now.astimezone(timezone.utc).isoformat(),
                                    'forecasts':forecasts,
                                    'status':str(report.get('status','Model unchanged'))})
    matured, total = collect_outcomes(bucket, histories, research_day)
    score = historical_scorecard(bucket, research_day)
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
            save_worker_model_state(bucket, proposed)
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
    detail['target_session'] = required_date
    detail['worker_ran_utc'] = now.astimezone(timezone.utc).isoformat()
    save_health(bucket, 'daily', True, research_day, detail=detail)
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
        # Safe detail only for errors constructed locally from validated ticker/date
        # information. Never log arbitrary upstream exception bodies or tokens.
        message = str(exc)
        if message.startswith(('Fewer than three verified ETC price histories',
                               'Fewer than three ETCs have a completed bar',
                               'No commodity forecasts generated for completed session')):
            LOG.error('Worker failed in %s: %s', args.mode, message[:550])
        else:
            LOG.error('Worker failed in %s (%s); inspect provider status and server configuration',
                      args.mode, type(exc).__name__)
        alert(safe)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
