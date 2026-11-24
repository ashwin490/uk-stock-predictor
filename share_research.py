"""Independent end-of-day UK-share price-event model.

This does not pretend to predict whether a target/stop paper order wins.
It records the chance of a >1.5% adjusted-close gain after five XLON sessions.
Model serialization is JSON data, never executable pickle from the network.
"""
from __future__ import annotations
import math
from datetime import timedelta
import numpy as np
import pandas as pd

HORIZON = 5
GAIN = .015
FEATURES = ('r1','r5','r10','r20','r60','vol10','vol20','vrel20','range14','atr14')


def _safe_numeric(x):
    return pd.to_numeric(x, errors='coerce').astype(float)


def create_feature_frame(bars: pd.DataFrame) -> pd.DataFrame:
    """Predictors at day t require <= t bars only. Sorted, adjusted prices."""
    if bars is None or len(bars) < 100:
        return pd.DataFrame()
    d = bars.copy().sort_index()
    if not isinstance(d.index, pd.DatetimeIndex) or d.index.has_duplicates:
        raise ValueError('Unordered or duplicate dates')
    if not {'Open','High','Low','Close','Volume'}.issubset(d.columns):
        raise ValueError('OHLCV missing')
    cl = _safe_numeric(d['Close'])
    hi, lo = _safe_numeric(d['High']), _safe_numeric(d['Low'])
    op, volume = _safe_numeric(d['Open']), _safe_numeric(d['Volume'])
    if ((cl <= 0) | (hi < lo) | (volume < 0)).any():
        raise ValueError('Invalid market bars')
    ret = cl.pct_change(fill_method=None)
    previous = cl.shift(1)
    tr = pd.concat([hi-lo, (hi-previous).abs(), (lo-previous).abs()],axis=1).max(axis=1)
    x = pd.DataFrame(index=d.index)
    for period in (1,5,10,20,60):
        x[f'r{period}'] = cl.pct_change(period, fill_method=None).clip(-.95, 5)
    x['vol10'] = ret.rolling(10).std().clip(0,2)
    x['vol20'] = ret.rolling(20).std().clip(0,2)
    x['vrel20'] = (volume/(volume.rolling(20).mean()+1)).clip(0,50)
    x['range14'] = ((hi-lo)/cl).rolling(14).mean().clip(0,2)
    x['atr14'] = (tr.rolling(14).mean()/cl).clip(0,2)
    x = x.replace([np.inf,-np.inf],np.nan)
    x['close'] = cl
    return x


def training_rows(ticker: str, bars: pd.DataFrame, complete_sessions: pd.DatetimeIndex) -> pd.DataFrame:
    x = create_feature_frame(bars)
    if x.empty:
        return pd.DataFrame()
    sessions = pd.DatetimeIndex(complete_sessions).tz_localize(None).normalize()
    ix = pd.DatetimeIndex(x.index).tz_localize(None).normalize()
    # Index into an authoritative exchange calendar, and reject a missing
    # session rather than wrongly labelling five vendor rows as five days.
    positions = sessions.get_indexer(ix)
    calendar_ok = positions >= 0
    pos_map = {int(p): i for i,p in enumerate(positions) if p >= 0}
    labels = np.full(len(x), np.nan)
    end_days = [''] * len(x)
    close = x['close'].to_numpy(dtype=float)
    for i,p in enumerate(positions):
        if not calendar_ok[i]:
            continue
        j = pos_map.get(int(p)+HORIZON)
        if j is None or j <= i:
            continue
        # All intervening exchange sessions must be present.
        if any((int(p)+q) not in pos_map for q in range(1,HORIZON+1)):
            continue
        labels[i] = float((close[j]/close[i]-1) > GAIN)
        end_days[i] = ix[j].date().isoformat()
    x = x.assign(label=labels,label_end=end_days,ticker=ticker,day=ix.strftime('%Y-%m-%d'))
    return x.dropna(subset=list(FEATURES)+['label']).reset_index(drop=True)


def selection_windows(df: pd.DataFrame, min_train=300, min_tune=100, min_test=100):
    """Train/tune/holdout by DATE with a five-session outcome embargo."""
    days = sorted(df['day'].unique())
    if len(days) < 110:
        return None
    cut1, cut2 = days[int(len(days)*.60)],days[int(len(days)*.80)]
    train = df.loc[(df['day']<cut1) & (df['label_end']<cut1)]
    tune = df.loc[(df['day']>=cut1)&(df['day']<cut2)&(df['label_end']<cut2)]
    test = df.loc[df['day']>=cut2]
    if len(train)<min_train or len(tune)<min_tune or len(test)<min_test:
        return None
    if any(len(set(part['label'].tolist())) < 2 for part in (train,tune,test)):
        return None
    return train,tune,test


def _fit(df, c):
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    scaler = StandardScaler()
    xx = scaler.fit_transform(df[list(FEATURES)].to_numpy(dtype=float))
    mod = LogisticRegression(C=c,random_state=41,max_iter=250,class_weight=None)
    mod.fit(xx,df['label'].to_numpy(dtype=int))
    return {'type':'logistic_v1','features':list(FEATURES),'mu':scaler.mean_.tolist(),
            'sigma':scaler.scale_.tolist(),'coef':mod.coef_[0].tolist(),
            'intercept':float(mod.intercept_[0]),'regularization_C':c,
            'train_count':int(len(df)), 'train_last_day':max(df['day']),
            'base_rate':float(df['label'].mean())}


def predict(model, data: pd.DataFrame) -> np.ndarray:
    if not model or model.get('type') != 'logistic_v1' or model.get('features') != list(FEATURES):
        raise ValueError('Unrecognised share model')
    sigma = np.asarray(model['sigma'],dtype=float)
    if np.any(sigma<=0) or not np.isfinite(sigma).all():
        raise ValueError('Invalid share model scale')
    x = (data[list(FEATURES)].to_numpy(dtype=float)-np.asarray(model['mu']))/sigma
    z = np.clip(x@np.asarray(model['coef'],dtype=float)+float(model['intercept']),-35,35)
    return 1/(1+np.exp(-z))


def _score(y, probs):
    from sklearn.metrics import roc_auc_score
    pp = np.clip(np.asarray(probs,dtype=float),1e-6,1-1e-6)
    yy=np.asarray(y,dtype=int)
    return {'n':int(len(yy)), 'brier':round(float(np.mean((pp-yy)**2)),6),
            'auc':round(float(roc_auc_score(yy,pp)),5),
            'events':int(yy.sum()), 'rise_signals':int(np.sum(pp>=.57))}


def train_candidate(df: pd.DataFrame):
    windows = selection_windows(df)
    if windows is None:
        return None, {'status':'insufficient independent labelled history'}
    tr,tu,te = windows
    from sklearn.metrics import brier_score_loss
    params = []
    for c in (.05,.5,2.0):
        m=_fit(tr,c)
        params.append((brier_score_loss(tu['label'],predict(m,tu)),c))
    c = sorted(params)[0][1]
    # The final holdout remains untouched by hyperparameter selection.
    m = _fit(pd.concat([tr,tu],ignore_index=True),c)
    te_p = predict(m,te)
    scores = _score(te['label'],te_p)
    base=float(pd.concat([tr,tu])['label'].mean())
    baseline=_score(te['label'],np.full(len(te),base))
    approved = (scores['brier']+.008 < baseline['brier'] and scores['auc']>=.54
                and scores['rise_signals']>=3)
    # Only a candidate that beats the baseline on independent holdout qualifies.
    final = _fit(df,c)
    return {'candidate':final,'approved':bool(approved),
            'report':{'test_from':min(te['day']), 'test_through':max(te['day']),
                      'train_count':int(len(tr)),'tune_count':int(len(tu)),
                      'test_count':int(len(te)), 'candidate':scores,'baseline':baseline,
                      'event':'>1.5% adjusted-close increase in 5 XLON sessions'}}, {'status':'evaluated'}


def latest_predictions(model, histories:dict, date_str:str):
    out=[]
    for ticker,bars in histories.items():
        try:
            df=create_feature_frame(bars).dropna(subset=list(FEATURES))
            if df.empty or df.index[-1].date().isoformat()!=date_str:
                continue
            p=float(predict(model,df.iloc[-1:])[0])
            out.append({'date':date_str,'ticker':ticker,'probability':round(p,6),
                        'predicted_rise':bool(p>=.57),'threshold':.57,
                        'baseline_probability':round(float(model['base_rate']),6),
                        'model_version':str(model.get('version','')),
                        'event':'adjusted close >1.5% after five XLON sessions',
                        'basis':'end-of-day adjusted prices'})
        except (KeyError,ValueError,TypeError,IndexError,OverflowError):
            continue
    return out


def reconcile_paper_equity_rows(rows: list[dict], tolerance_gbp:float=0.15) -> dict:
    """Non-mutating accounting audit. Missing cost basis is UNKNOWN, not a zero-fee pass."""
    mismatch=[]; unknown=[]; audited=0; duplicates=[]
    seen=set()
    for r in rows:
        ticker=str(r.get('ticker','')).upper()
        day=str(r.get('predicted_date',''))[:10]
        key=(ticker,day)
        if not ticker or not day:
            unknown.append({'ticker':ticker,'date':day,'issue':'missing identity'});continue
        if key in seen:
            duplicates.append({'ticker':ticker,'date':day});continue
        seen.add(key)
        f=r.get('features_json') or {}
        if isinstance(f,str):
            try: import json; f=json.loads(f)
            except ValueError: f={}
        if not isinstance(f,dict): f={}
        if f.get('asset_class') == 'COMMODITY_ETC' or ticker in ('SGLN.L','SSLN.L','BRNG.L','CRUP.L','NGSP.L'):
            continue
        try:
            entry=float(r.get('entry_price'))
            latest=float(r.get('latest_price'))
            cap=float(r.get('position_gbp'))
            reported=float(r.get('pnl_pct'))
            # Can reconstruct only if both round-trip spread/SDRT and commission
            # were saved. Avoid guessing tax treatment for historical rows.
            if not ('friction_pct' in f and 'commission_gbp' in f):
                raise ValueError('missing recorded fee basis')
            friction=float(f['friction_pct']) + 200*float(f['commission_gbp'])/cap
            if not all(math.isfinite(x) for x in (entry,latest,cap,reported,friction)) or min(entry,latest,cap)<=0:
                raise ValueError('non-finite amounts')
            calculated=(latest/entry-1)*100-friction
            audited+=1
            if abs((calculated-reported)*cap/100)>tolerance_gbp:
                mismatch.append({'ticker':ticker,'date':day,
                                 'reported_pnl_gbp':round(reported*cap/100,2),
                                 'recomputed_pnl_gbp':round(calculated*cap/100,2)})
        except (ValueError,TypeError,ZeroDivisionError):
            unknown.append({'ticker':ticker,'date':day,'issue':'unverifiable fee or price fields'})
    return {'records_seen':len(rows),'audited':audited,'mismatch_count':len(mismatch),
            'unknown_count':len(unknown),'duplicate_count':len(duplicates),
            'mismatches':mismatch[:25], 'unknown_examples':unknown[:12],
            'duplicate_examples':duplicates[:12],
            'status':'review' if mismatch or duplicates else ('incomplete' if unknown else 'ok')}
