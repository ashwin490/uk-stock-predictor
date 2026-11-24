"""Offline regression tests for causal historical commodity ML and safe persistence."""
from datetime import date
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from commodity_ml import (PRODUCTS, MODEL_SCHEMA, FEATURE_NAMES, CommodityMLStore,
                          clean_completed_bars, make_dataset, _features, _fit_model,
                          predict_proba, latest_forecast, _split_by_date,
                          train_or_update, validate_model,
                          backfill_research_preview)


def bars(n=700, seed=1):
    rng=np.random.default_rng(seed)
    ret=rng.normal(0.0004,0.016,size=n) + .002*np.sin(np.arange(n)/16+seed)
    c=130*np.exp(np.cumsum(ret))
    ix=pd.date_range('2022-01-03',periods=n,freq='B',tz='Europe/London')
    h=pd.DataFrame({'Open':c,'High':c*1.02,'Low':c*.98,'Close':c,
                    'Volume':np.full(n,5_000_000)},index=ix)
    h.attrs['price_currency']='GBp'
    return h


def histories():
    return {t:bars(seed=i+1) for i,t in enumerate(PRODUCTS)}


def test_reject_unverified_currency_even_if_numbers_look_good():
    b=bars(); b.attrs['price_currency']='USD'
    assert clean_completed_bars(b,date(2026,11,24)).empty
    b.attrs.clear()
    assert clean_completed_bars(b,date(2026,11,24)).empty


def test_excludes_unfinished_current_day_and_prevents_future_lookahead():
    b=bars()
    day=b.index[-1].date()
    filtered=clean_completed_bars(b,day)
    assert filtered.index[-1]==b.index[-2]
    dataset,_=make_dataset({'SGLN.L':b},day)
    # The last 5 rows cannot have matured outcomes; a current day is removed.
    assert dataset.feature_date.max() == b.index[-7].date().isoformat()
    assert dataset.feature_date.max() < day.isoformat()


def test_rejects_crossed_ohlc_and_larger_invalid_regions():
    b=bars()
    b.iloc[-80:, b.columns.get_loc('High')]=1
    assert clean_completed_bars(b,date(2026,11,24)).empty


def test_product_one_hot_and_purely_causal_indicators():
    b=bars()
    f=_features(clean_completed_bars(b,date(2026,11,24)), 'SGLN.L')
    assert not f.empty and np.isfinite(f.to_numpy()).all()
    assert set(f.columns)==set(FEATURE_NAMES)
    assert f['product_SGLN.L'].eq(1).all() and f['product_NGSP.L'].eq(0).all()
    c=b.copy()
    c.iloc[-1,c.columns.get_loc('Close')]*=1.005
    f2=_features(clean_completed_bars(c,date(2026,11,24)),'SGLN.L')
    # Changing last bar never changes earlier feature values.
    assert np.allclose(f.iloc[:-1].to_numpy(),f2.iloc[:-1].to_numpy())


def test_fit_and_predict_portable_json_model():
    data,_=make_dataset(histories(),date(2026,11,24))
    model=_fit_model(data)
    validate_model(model)
    encoded=json.dumps(model)
    restored=json.loads(encoded)
    p1=predict_proba(model,data.head(100))
    p2=predict_proba(restored,data.head(100))
    assert np.allclose(p1,p2)
    assert ((p1>0)&(p1<1)).all()
    fc=latest_forecast(restored,'SGLN.L',bars(),date(2026,11,24))
    assert 0<=fc['probability']<=1 and .5<fc['threshold']<1
    assert 'before costs' in fc['event']


def test_train_test_five_session_embargo_and_later_dates():
    data,_=make_dataset(histories(),date(2026,11,24))
    train,test,start=_split_by_date(data,None)
    dates=sorted(data.feature_date.unique())
    assert train.feature_date.max()<=dates[dates.index(start)-6]
    assert test.feature_date.min()>train.feature_date.max()
    assert len(train)>=450 and test.feature_date.nunique()>=22


def test_train_then_retrain_once_daily_and_dont_reuse_same_holdout():
    today=date(2026,11,24)
    state,report,save=train_or_update(histories(),{},today)
    assert save and report['test_count']>0
    assert state['last_attempt_day']==today.isoformat()
    same,_,again=train_or_update(histories(),state,today)
    assert not again and same==state
    # With no additional labelled days, no new challenger can be assessed.
    next_state,next_report,_=train_or_update(histories(),state,date(2026,11,25))
    assert 'Waiting for enough' in next_report['status']
    assert next_state.get('version')==state.get('version')
    if state.get('champion'):
        assert state['evaluated_through']==state['champion']['market_data_through']


def test_fail_closed_on_sparse_samples_or_wrong_units():
    s,r,save=train_or_update({}, {}, date(2026,11,24))
    assert save and s.get('champion') is None and 'insufficient' in r['status']
    bad=histories(); bad['SGLN.L'].attrs['price_currency']='USD'
    d,meta=make_dataset(bad,date(2026,11,24))
    assert 'SGLN.L' not in meta and set(d.ticker.unique())==set(PRODUCTS)-{'SGLN.L'}


def test_model_version_validates_exact_schema_and_finite_numbers():
    data,_=make_dataset(histories(),date(2026,11,24)); m=_fit_model(data)
    m['coef'][0]=float('nan')
    with pytest.raises(ValueError):validate_model(m)
    m['coef'][0]=0
    m['features'][0]='look_ahead_return'
    with pytest.raises(ValueError):validate_model(m)


def test_local_store_atomic_roundtrip(tmp_path):
    state,_,_=train_or_update(histories(),{},date(2026,11,24))
    st=CommodityMLStore(tmp_path/'models'/'commodity.json')
    st.save(state)
    assert st.load()==state
    assert not (tmp_path/'models'/'commodity.json.tmp').exists()
    assert st.load()['schema']==MODEL_SCHEMA


def test_local_store_rejects_bad_pickle_like_or_corrupt_content(tmp_path):
    p=tmp_path/'model.json';p.write_bytes(b'not a json file')
    with pytest.raises(ValueError):CommodityMLStore(p).load()


class FakeBucket:
    def __init__(self): self.saved={}
    def download(self,path):
        if path not in self.saved:raise RuntimeError('Object not found (404)')
        return self.saved[path]
    def upload(self,path,payload,file_options=None):
        assert file_options.get('upsert')=='true'
        self.saved[path]=payload


def test_cloud_storage_persists_exact_bytes_and_loads(tmp_path):
    b=FakeBucket()
    store=CommodityMLStore(tmp_path/'ephemeral.json',cloud_mode=True,bucket=b)
    assert store.load()=={}
    state,_,_=train_or_update(histories(),{},date(2026,11,24))
    store.save(state)
    assert store.load()==state
    assert not (tmp_path/'ephemeral.json').exists()


def test_missing_cloud_bucket_fails_closed(tmp_path):
    store=CommodityMLStore(tmp_path/'ephemeral.json',cloud_mode=True)
    with pytest.raises(RuntimeError,match='not configured'):store.load()
    with pytest.raises(RuntimeError,match='not configured'):store.save({'schema':MODEL_SCHEMA})


def test_failed_cloud_readback_must_not_be_assumed_persisted(tmp_path):
    class BadBucket(FakeBucket):
        def download(self,path):return b'{}'
    store=CommodityMLStore(tmp_path/'file',cloud_mode=True,bucket=BadBucket())
    with pytest.raises(RuntimeError,match='not activated'):
        store.save({'schema':MODEL_SCHEMA,'last_attempt_day':'2026-11-24'})


def test_existing_champion_is_not_overwritten_without_new_test_days():
    state,_,_=train_or_update(histories(),{},date(2026,11,24))
    original=json.dumps(state.get('champion'),sort_keys=True)
    next_state,next_report,_=train_or_update(histories(),state,date(2026,11,25))
    assert json.dumps(next_state.get('champion'),sort_keys=True)==original
    assert 'Waiting for enough' in next_report['status']


def test_card_output_shows_explicit_five_day_forecast_not_trade_win():
    from commodity_dashboard import commodity_signal_card_html
    html=commodity_signal_card_html({'Commodity':'Gold','Ticker':'SGLN.L','Decision':'WAIT',
                                    'ML forecast (%)':63.2, 'ML threshold (%)':58.0,
                                    'ML status':'historically tested', 'Current (p)':4500})
    assert '63.2%' in html and '58.0%' in html
    assert 'before costs' in html and '5-session ETC gain' in html


def test_source_integration_never_reuses_stock_ml_or_overrides_securities():
    app=(Path(__file__).resolve().parents[1]/'app.py').read_text()
    assert 'from commodity_ml import' in app
    assert "STORAGE_MODE != 'cloud'" in app
    assert "_get_commodity_ml_store" in app
    assert "store.save(proposed)  # MUST persist before in-session activation." in app
    assert 'CommodityMLQualified' in app
    assert 'CommodityRuleQualified' in app
    assert 'is_entry_window_open()' in app


def test_rejected_candidate_has_research_predictions_but_no_champion():
    today=date(2026,11,24)
    state, report, _ = train_or_update(histories(), {}, today)
    assert not report['promoted'], 'Synthetic regression fixture expects failed challenger'
    assert state.get('champion') is None
    assert state['research_model']['research_only'] is True
    assert state['last_evaluation']['research_evidence']['by_product']
    assert len(state['last_evaluation']['research_evidence']['recent_examples']) == 15
    assert sum(p['tested'] for p in report['research_evidence']['by_product']) == report['test_count']
    # Rejected research-only forecasts are explicitly separate from a trading champion.
    assert 'champion' not in state


def test_research_evidence_remains_when_waiting_for_new_independent_dates():
    today=date(2026,11,24)
    state, report, _ = train_or_update(histories(), {}, today)
    old=state['last_evaluation']['research_evidence']
    new_state, waiting, _ = train_or_update(histories(), state, date(2026,11,25))
    assert 'Waiting for enough' in waiting['status']
    assert new_state['last_evaluation']['research_evidence'] == old
    assert new_state['research_model'] == state['research_model']


def test_backfill_old_rejected_cloud_state_keeps_promotion_watermark_and_state():
    today=date(2026,11,24)
    generated, report, _ = train_or_update(histories(), {}, today)
    old = {k:v for k,v in generated.items() if k not in ('research_model','last_evaluation')}
    before = old.get('evaluated_through')
    backfilled, changed = backfill_research_preview(histories(),old,today)
    assert changed
    assert backfilled['research_model']['research_only']
    assert backfilled['last_evaluation']['research_evidence']['recent_examples']
    assert backfilled.get('champion') == old.get('champion')
    assert backfilled.get('evaluated_through') == before
    assert backfilled['last_attempt_day'] == old['last_attempt_day']
    again, changed_again = backfill_research_preview(histories(),backfilled,today)
    assert not changed_again and again == backfilled


def test_research_card_is_explicitly_unvalidated_and_not_trade_win():
    from commodity_dashboard import commodity_signal_card_html
    html=commodity_signal_card_html({'Commodity':'Gold','Ticker':'SGLN.L','Decision':'WAIT',
                                    'ML research forecast (%)':47.8,
                                    'ML research threshold (%)':57.0,
                                    'ML research bar date':'2026-11-23','Current (p)':4500})
    assert '47.8%' in html and '57.0%' in html and '2026-11-23' in html
    assert 'Commodity ML · Preview' in html and 'Model: unvalidated' in html
    assert 'Does not authorise entries' in html and 'five sessions' in html


def test_research_only_model_never_assigned_to_trade_filter():
    app=(Path(__file__).resolve().parents[1]/'app.py').read_text()
    assert "research_model = (st.session_state.get('commodity_ml_state') or {}).get('research_model')" in app
    assert "if not champion and research_model:" in app
    assert "ml_qualified = bool(champion and ml_enabled" in app
    assert "store.save(_new_state)" in app
    assert "backfill_research_preview" in app
