"""Isolated commodity ETC ML: historical daily bars -> safe, time-tested JSON models.

Forecast event: adjusted? No. Raw London ETC close in GBp rises >1.5% over
five subsequent trading sessions, *before* costs. Not a forecast of the app's
stop/target trade win, and never a promise of execution or profitability.

This module never places orders or touches the share ML. Only previously
completed bars are accepted; labels need five *future* completed sessions.
Model records are pure JSON (not untrusted pickle), safe for Supabase Storage.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
import json
import math
import os
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

HORIZON = 5
GAIN_THRESHOLD = 0.015
MODEL_SCHEMA = 1
STRATEGY_NAME_ML = "COMMODITY_ETC_ML_V1"
PRODUCTS = ("SGLN.L", "SSLN.L", "BRNG.L", "CRUP.L", "NGSP.L")
FEATURE_NAMES = (
    "ret_1", "ret_5", "ret_10", "ret_20", "sma_10", "sma_20", "sma_50",
    "sma20_50", "vol_10", "vol_20", "atr_14", "rsi_14", "range_5",
    "volume_rel20", "drawdown20", "turnover20",
) + tuple("product_" + t for t in PRODUCTS)
MIN_TRAIN = 450
MIN_TEST = 85
MIN_TEST_DAYS = 22
MIN_NEW_TEST_DAYS = 22


def _safe_float(v):
    try:
        x = float(v)
        return x if math.isfinite(x) else None
    except (ValueError, TypeError, OverflowError):
        return None


def clean_completed_bars(bars: pd.DataFrame, today: date) -> pd.DataFrame:
    """Reject unverified currencies, current-day bars, splits and bad OHLCV.

    Prices must have passed the app's issuer/currency validation; a rescaled
    USD/GBP series cannot enter this trainer without an explicit GBp tag.
    """
    if not isinstance(bars, pd.DataFrame) or bars.empty or bars.attrs.get('price_currency') != 'GBp':
        return pd.DataFrame()
    keys = ["Open", "High", "Low", "Close", "Volume"]
    if not isinstance(bars.index, pd.DatetimeIndex) or not set(keys).issubset(bars.columns):
        return pd.DataFrame()
    x = bars.copy().sort_index()
    x = x[~x.index.duplicated(keep='last')]
    dates = pd.DatetimeIndex(x.index).date
    x = x.loc[np.asarray(dates) < today].copy()  # Never train on an unfinished current day.
    if len(x) < 90:
        return pd.DataFrame()
    vals = x[keys].apply(pd.to_numeric, errors="coerce")
    valid = (np.isfinite(vals).all(axis=1) & (vals[keys[:4]] > 0).all(axis=1)
             & (vals['Volume'] >= 0)
             & (vals['High'] >= vals[['Open', 'Close', 'Low']].max(axis=1))
             & (vals['Low'] <= vals[['Open', 'Close', 'High']].min(axis=1)))
    # Large invalid segments indicate feed problems; skip the instrument rather
    # than creating fake data by forward filling.
    if valid.mean() < 0.97:
        return pd.DataFrame()
    if 'Stock Splits' in x.columns:
        splits = pd.to_numeric(x['Stock Splits'], errors='coerce').fillna(0)
        # A split changes unit prices. Require sufficient post-split history.
        split_rows = np.flatnonzero(splits.to_numpy() != 0)
        if len(split_rows):
            x = x.iloc[int(split_rows[-1]) + 1:]
            valid = valid.loc[x.index]
    return vals.loc[x.index].loc[valid].copy() if len(x) >= 90 else pd.DataFrame()


def _features(bars: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Vectorized causal features: at date t, uses OHLCV through t only."""
    c = bars['Close'].astype(float)
    h, l, v = (bars[k].astype(float) for k in ('High', 'Low', 'Volume'))
    r = c.pct_change(fill_method=None)
    sma10 = c.rolling(10).mean()
    sma20 = c.rolling(20).mean()
    sma50 = c.rolling(50).mean()
    prev = c.shift(1)
    tr = pd.concat([h-l, (h-prev).abs(), (l-prev).abs()], axis=1).max(axis=1)
    d = c.diff()
    gains = d.clip(lower=0).rolling(14).mean()
    losses = (-d.clip(upper=0)).rolling(14).mean()
    rs = gains / losses.replace(0, np.nan)
    rsi = (100 - 100 / (1 + rs)).fillna(50)
    vol_ma = v.rolling(20).mean().replace(0, np.nan)
    f = pd.DataFrame(index=bars.index)
    f['ret_1'] = r
    for n in (5, 10, 20):
        f[f'ret_{n}'] = c.pct_change(n, fill_method=None)
    for n, ma in ((10,sma10),(20,sma20),(50,sma50)):
        f[f'sma_{n}'] = (c / ma) - 1
    f['sma20_50'] = (sma20 / sma50) - 1
    f['vol_10'] = r.rolling(10).std()
    f['vol_20'] = r.rolling(20).std()
    f['atr_14'] = tr.rolling(14).mean() / c
    f['rsi_14'] = rsi / 100
    f['range_5'] = ((h-l) / c).rolling(5).mean()
    f['volume_rel20'] = (v / vol_ma).clip(0, 10)
    f['drawdown20'] = (c / c.rolling(20).max()) - 1
    f['turnover20'] = np.log1p((c * v / 100).rolling(20).mean().clip(lower=0))
    for name in PRODUCTS:
        f['product_' + name] = float(ticker == name)
    f.replace([np.inf, -np.inf], np.nan, inplace=True)
    return f[list(FEATURE_NAMES)].dropna()


def make_dataset(histories: Mapping[str, pd.DataFrame], today: date) -> tuple[pd.DataFrame, dict]:
    """Only matured labels: t+5 is present and completed; never fill missing future."""
    frames = []
    usable = {}
    for ticker in PRODUCTS:
        b = clean_completed_bars(histories.get(ticker), today)
        if len(b) < 120:
            continue
        features = _features(b, ticker)
        # The label uses a future close, but features are strictly causal.
        outcome = b['Close'].shift(-HORIZON) / b['Close'] - 1
        frame = features.copy()
        frame['label'] = (outcome > GAIN_THRESHOLD).astype(int)
        frame['future_return'] = outcome
        frame = frame.loc[outcome.notna()]
        frame = frame.replace([np.inf, -np.inf], np.nan).dropna()
        frame['ticker'] = ticker
        frame['feature_date'] = [pd.Timestamp(t).date().isoformat() for t in frame.index]
        if len(frame) < 70:
            continue
        frames.append(frame.reset_index(drop=True))
        usable[ticker] = len(frame)
    if not frames:
        return pd.DataFrame(), usable
    all_rows = pd.concat(frames, ignore_index=True)
    all_rows.sort_values(['feature_date', 'ticker'], kind='stable', inplace=True)
    return all_rows.reset_index(drop=True), usable


def _fit_model(rows: pd.DataFrame) -> dict:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    X = rows[list(FEATURE_NAMES)].to_numpy(dtype=float)
    y = rows['label'].to_numpy(dtype=int)
    if len(set(y.tolist())) != 2:
        raise ValueError("Need positive and negative historical examples")
    scaler = StandardScaler().fit(X)
    Z = np.clip(scaler.transform(X), -8, 8)
    clf = LogisticRegression(C=0.12, max_iter=500, solver='lbfgs', random_state=41)
    clf.fit(Z, y)
    # Store simple numerical weights, not executable pickle files.
    model = dict(schema=MODEL_SCHEMA, features=list(FEATURE_NAMES),
                 mean=scaler.mean_.tolist(), scale=scaler.scale_.tolist(),
                 coef=clf.coef_[0].tolist(), intercept=float(clf.intercept_[0]),
                 event_base_rate=float(y.mean()), rows=int(len(rows)),
                 trained_through=str(rows['feature_date'].max()))
    validate_model(model)
    return model


def validate_model(model: dict):
    if not isinstance(model, dict) or model.get('schema') != MODEL_SCHEMA or model.get('features') != list(FEATURE_NAMES):
        raise ValueError('Unsupported commodity model format')
    for key in ('mean', 'scale', 'coef'):
        v = model.get(key)
        if not isinstance(v, list) or len(v) != len(FEATURE_NAMES) or any(_safe_float(x) is None for x in v):
            raise ValueError('Invalid commodity ML parameters')
    if any(float(x) <= 0 for x in model['scale']):
        raise ValueError('Invalid commodity ML scaling')
    if _safe_float(model.get('intercept')) is None:
        raise ValueError('Invalid commodity ML intercept')
    p = _safe_float(model.get('event_base_rate'))
    if p is None or not 0 < p < 1:
        raise ValueError('Invalid commodity ML baseline')


def predict_proba(model: dict, frame: pd.DataFrame) -> np.ndarray:
    validate_model(model)
    x = frame[list(FEATURE_NAMES)].to_numpy(dtype=float)
    z = np.clip((x - np.asarray(model['mean'])) / np.asarray(model['scale']), -8, 8)
    logits = z @ np.asarray(model['coef']) + float(model['intercept'])
    logits = np.clip(logits, -35, 35)
    return 1.0 / (1.0 + np.exp(-logits))


def latest_forecast(model: dict, ticker: str, completed: pd.DataFrame, today: date) -> dict:
    if ticker not in PRODUCTS:
        raise ValueError('Unlisted commodity')
    bars = clean_completed_bars(completed, today)
    if bars.empty or len(bars) < 70:
        raise ValueError('Not enough verified completed bars for a forecast')
    feats = _features(bars, ticker)
    if feats.empty or feats.index[-1] != bars.index[-1]:
        raise ValueError('Latest completed bar lacks valid features')
    p = float(predict_proba(model, feats.tail(1))[0])
    base = float(model['event_base_rate'])
    # Threshold is for a historically defined event, not the probability of
    # achieving the separately calculated sell target or an actual profit.
    threshold = min(0.80, max(0.57, base + 0.05))
    return {'probability': round(p, 4), 'threshold': round(threshold, 4),
            'qualified': p >= threshold, 'bar_date': str(feats.index[-1].date()),
            'event': 'ETC 5-session close-to-close rise >1.5%, before costs'}


def _metrics(y, p):
    y = np.asarray(y, dtype=float)
    p = np.clip(np.asarray(p, dtype=float), 0.0001, 0.9999)
    return {'brier': float(np.mean((y-p)**2)),
            'logloss': float(np.mean(-y*np.log(p)-(1-y)*np.log1p(-p)))}


def _split_by_date(rows: pd.DataFrame, earliest_test_after: str | None):
    days = sorted(rows['feature_date'].unique().tolist())
    if len(days) < 150:
        return None
    if earliest_test_after:
        test_days = [d for d in days if d > earliest_test_after]
        if len(test_days) < MIN_NEW_TEST_DAYS:
            return None
        test_start = test_days[0]
    else:
        start_idx = max(0, int(len(days)*0.78))
        test_days = days[start_idx:]
        test_start = test_days[0]
    idx = days.index(test_start)
    if idx < HORIZON + 75:
        return None
    # Exclude all feature dates whose future labels can overlap testing.
    safe_end = days[idx-HORIZON-1]
    train = rows.loc[rows.feature_date <= safe_end]
    test = rows.loc[rows.feature_date >= test_start]
    if (len(train) < MIN_TRAIN or len(test) < MIN_TEST
            or len(test_days) < MIN_TEST_DAYS or train['label'].nunique() < 2
            or test['label'].nunique() < 2):
        return None
    return train, test, test_start


def _historical_research_evidence(test: pd.DataFrame, probabilities: np.ndarray, prior: float) -> dict:
    """Compact truly held-out forecasts and outcomes; never a trading strategy P/L.

    Each forecast was made by a candidate fit on earlier feature dates only.
    Only a small recent sample is saved so cloud state stays within its limit.
    """
    threshold = min(0.80, max(0.57, float(prior) + 0.05))
    observed = test[['ticker', 'feature_date', 'label', 'future_return']].copy()
    observed['probability'] = np.asarray(probabilities, dtype=float)
    observed['flagged'] = observed['probability'] >= threshold
    per_product = []
    for ticker in PRODUCTS:
        subset = observed.loc[observed['ticker'] == ticker]
        if subset.empty:
            continue
        picked = subset.loc[subset['flagged']]
        per_product.append({
            'ticker': ticker,
            'tested': int(len(subset)),
            'predicted_rises': int(len(picked)),
            'correct_rises': int(picked['label'].sum()),
            'actual_rises': int(subset['label'].sum()),
            'brier': round(float(np.mean((subset['probability'].to_numpy() - subset['label'].to_numpy()) ** 2)), 4),
        })
    # Last three known outcomes from each ETC, chronological test data only.
    last = observed.groupby('ticker', sort=False).tail(3).sort_values(['feature_date', 'ticker'])
    examples = [{
        'date': str(r.feature_date), 'ticker': str(r.ticker),
        'predicted_chance_pct': round(float(r.probability) * 100, 1),
        'predicted_rise': bool(r.flagged),
        'actual_five_session_pct': round(float(r.future_return) * 100, 2),
        'event_happened': bool(r.label),
    } for r in last.itertuples(index=False)]
    return {'threshold_pct': round(threshold * 100, 1),
            'by_product': per_product, 'recent_examples': examples}


def _is_better(candidate: dict, comparator: dict, *, first: bool) -> bool:
    # Reject modest differences likely to be noise; Brier and logloss must both
    # improve. With first model baseline is a constant trained prior.
    brier_delta = 0.004 if first else 0.006
    loss_delta = 0.006 if first else 0.008
    return (candidate['brier'] <= comparator['brier'] - brier_delta
            and candidate['logloss'] <= comparator['logloss'] - loss_delta)


def train_or_update(histories: Mapping[str, pd.DataFrame], old_state: dict | None, today: date):
    """Candidate/chronological test; never overwrite champion on a failed test.

    Return (proposed_state, report, should_persist). Caller MUST persist before
    activating the new champion in cloud mode. Periodic checks require new
    completed trading dates and test only dates not seen by old champion.
    """
    state = dict(old_state or {})
    if state.get('schema', MODEL_SCHEMA) != MODEL_SCHEMA:
        raise ValueError('Incompatible commodity ML state')
    state['schema'] = MODEL_SCHEMA
    if state.get('last_attempt_day') == today.isoformat():
        return state, dict(state.get('report') or {'status':'Already checked today'}), False
    dataset, usable = make_dataset(histories, today)
    state['last_attempt_day'] = today.isoformat()
    report = {'status':'Research only; insufficient historical data', 'sample_count':len(dataset),
              'products':usable, 'forecast_event':'5-session ETC rise >1.5% before costs'}
    if len(usable) < 3 or len(dataset) < MIN_TRAIN+MIN_TEST:
        state['report'] = report
        return state, report, True
    incumbent = state.get('champion')
    if incumbent:
        validate_model(incumbent)
    next_dates = (state.get('evaluated_through') or incumbent.get('market_data_through')
                  if incumbent else state.get('evaluated_through'))
    pair = _split_by_date(dataset, next_dates)
    if pair is None:
        report['status'] = 'Waiting for enough independent, newly completed test sessions' if next_dates else 'Not enough independent validation history'
        state['report'] = report
        return state, report, True
    train, test, test_start = pair
    model = _fit_model(train)
    pred = predict_proba(model, test)
    base_rate = float(train['label'].mean())
    baseline = _metrics(test['label'], np.full(len(test), base_rate))
    candidate_metrics = _metrics(test['label'], pred)
    comparator_metrics = _metrics(test['label'], predict_proba(incumbent, test)) if incumbent else baseline
    pass_gate = (_is_better(candidate_metrics, baseline, first=True)
                 and _is_better(candidate_metrics, comparator_metrics, first=incumbent is None))
    research_evidence = _historical_research_evidence(test, pred, base_rate)
    report.update({'research_evidence': research_evidence,
                   'trained_count':len(train), 'test_count':len(test),
                   'test_days':test['feature_date'].nunique(), 'test_from':test_start,
                   'test_through':test['feature_date'].max(), 'base_rate':round(base_rate,4),
                   'baseline':baseline, 'challenger':candidate_metrics,
                   'champion_score':comparator_metrics if incumbent else None,
                   'promoted':bool(pass_gate)})
    # Do not repeatedly tune on already inspected days: another comparison only
    # after a fresh block of new matured observations has accrued.
    state['evaluated_through'] = str(dataset.feature_date.max())
    if pass_gate:
        promoted = _fit_model(dataset)
        version = int(state.get('version', 0)) + 1
        promoted['version'] = version
        promoted['promoted_day'] = today.isoformat()
        promoted['market_data_through'] = max(str(pd.Timestamp(b.index[-1]).date())
                                              for t,bars in histories.items()
                                              if t in usable and not (b := clean_completed_bars(bars,today)).empty)
        promoted['validation'] = {'baseline':baseline, 'challenger':candidate_metrics,
                                  'incumbent':comparator_metrics if incumbent else None,
                                  'n_test':len(test), 'start':test_start}
        # The champion was trained on labels observed through the latest RAW
        # completed bar. Future comparisons must start *after* that bar, not
        # merely after the latest labelled feature date (five days earlier).
        state['evaluated_through'] = promoted['market_data_through']
        state['previous_champion'] = incumbent
        # A validated champion supersedes the unvalidated research preview.
        state.pop('research_model', None)
        state['champion'] = promoted
        state['champion_last_confirmed_day'] = today.isoformat()
        state['champion_paused'] = False
        state['version'] = version
        report['status'] = f'Version {version} promoted after independent historical evaluation'
    else:
        # Show the user's historical-data research estimates immediately, even
        # though this rejected model NEVER authorises a paper purchase.
        # Fit only on fully matured history; today's unfinished bar is excluded.
        state['research_model'] = _fit_model(dataset)
        state['research_model']['research_only'] = True
        state['research_model']['created_day'] = today.isoformat()
        report['status'] = 'Challenger rejected: did not beat baseline and current champion'
        if incumbent:
            # An incumbent must continue to earn trust on genuinely later data.
            # If it fails a constant-prior benchmark, pause NEW ML entries.
            still_beats_prior = _is_better(comparator_metrics, baseline, first=True)
            state['champion_paused'] = not still_beats_prior
            if still_beats_prior:
                state['champion_last_confirmed_day'] = today.isoformat()
            else:
                report['status'] += '; incumbent also failed baseline, new ML buys paused'
    # Keep the last evaluated predictions visible even while awaiting 22 NEW
    # test days; do not silently erase evidence on tomorrow's rerun.
    state['last_evaluation'] = report
    state['report'] = report
    return state, report, True


def backfill_research_preview(histories: Mapping[str, pd.DataFrame],
                              old_state: dict, today: date) -> tuple[dict, bool]:
    """Expose a pre-existing rejected test without re-running model promotion.

    Older release states stored test metrics but not test-date predictions or a
    rejected model. This migration ONLY adds research evidence; neither the
    promotion gate, validation watermark, nor live trading status changes.
    """
    state = dict(old_state)
    previous = state.get('last_evaluation') or state.get('report') or {}
    if state.get('champion') or state.get('research_model') or not previous.get('test_count'):
        return state, False
    dataset, usable = make_dataset(histories, today)
    if len(usable) < 3 or len(dataset) < MIN_TRAIN + MIN_TEST:
        return state, False
    pair = _split_by_date(dataset, None)
    if pair is None:
        return state, False
    train, test, start = pair
    candidate = _fit_model(train)
    pred = predict_proba(candidate, test)
    prior = float(train['label'].mean())
    historical = _historical_research_evidence(test, pred, prior)
    # Only attach predictions to old scores when the holdout dates match.
    # Otherwise mark this explicitly as a new display-only evaluation.
    same_test = (str(previous.get('test_from')) == str(start)
                 and int(previous.get('test_count') or 0) == len(test))
    report = dict(previous) if same_test else {
        'status': 'Display-only historical research test; NOT model validation or promotion',
        'test_from': start, 'test_count': len(test),
        'test_days': int(test['feature_date'].nunique()),
        'test_through': str(test['feature_date'].max()),
        'baseline': _metrics(test['label'], np.full(len(test), prior)),
        'challenger': _metrics(test['label'], pred),
        'promoted': False,
    }
    report['research_evidence'] = historical
    report['research_backfilled'] = True
    state['last_evaluation'] = report
    state['research_model'] = _fit_model(dataset)
    state['research_model']['research_only'] = True
    state['research_model']['created_day'] = today.isoformat()
    return state, True


class CommodityMLStore:
    """Local atomic JSON or private Supabase Storage persistence.

    Cloud mode requires a configured private bucket and write-capable server
    credentials. NEVER treat ephemeral container files as a durable champion.
    """
    def __init__(self, path, *, cloud_mode=False, bucket=None):
        self.path = Path(path)
        self.cloud_mode = bool(cloud_mode)
        self.bucket = bucket

    def load(self) -> dict:
        if self.cloud_mode:
            if self.bucket is None:
                raise RuntimeError('Private Supabase ML bucket is not configured')
            try:
                payload = self.bucket.download('commodity_ml_champion_v1.json')
            except Exception as exc:
                msg = str(exc).lower()
                if 'not found' in msg or '404' in msg or 'object not found' in msg:
                    return {}
                raise RuntimeError('Remote commodity model read unavailable') from exc
            data = bytes(payload)
        else:
            if not self.path.exists():
                return {}
            data = self.path.read_bytes()
        if len(data) > 250000:
            raise ValueError('Oversized commodity model state')
        obj = json.loads(data.decode('utf-8'))
        if not isinstance(obj, dict) or obj.get('schema') != MODEL_SCHEMA:
            raise ValueError('Invalid commodity model state')
        if obj.get('champion'):
            validate_model(obj['champion'])
        if obj.get('research_model'):
            validate_model(obj['research_model'])
        return obj

    def save(self, state: dict) -> None:
        if not isinstance(state, dict) or state.get('schema') != MODEL_SCHEMA:
            raise ValueError('Invalid commodity model state')
        if state.get('champion'):
            validate_model(state['champion'])
        if state.get('research_model'):
            validate_model(state['research_model'])
        payload = json.dumps(state, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf-8')
        if len(payload) > 250000:
            raise ValueError('Oversized commodity model state')
        if self.cloud_mode:
            if self.bucket is None:
                raise RuntimeError('Private Supabase ML bucket is not configured')
            try:
                self.bucket.upload('commodity_ml_champion_v1.json', payload,
                                   file_options={'content-type':'application/json','upsert':'true'})
                check = bytes(self.bucket.download('commodity_ml_champion_v1.json'))
                if check != payload:
                    raise ValueError('Cloud model readback differs from saved model')
            except Exception as exc:
                raise RuntimeError('Remote model persistence failed; model not activated') from exc
        else:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_name(self.path.name + '.tmp')
            try:
                tmp.write_bytes(payload)
                os.replace(tmp, self.path)
            finally:
                if tmp.exists():
                    tmp.unlink()
