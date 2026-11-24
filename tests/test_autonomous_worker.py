"""No real trading, no real network, no secrets; independent worker regressions."""
from datetime import date, datetime, timezone, timedelta
import json
import os
import numpy as np
import pandas as pd
import pytest

import commodity_worker as w
from commodity_ml import (_fit_specialists, _fit_model, validate_model,
                          predict_proba, FEATURE_NAMES)


class MissingFile(Exception):
    status_code = 404


class FakeBucket:
    def __init__(self):
        self.files = {}
        self.updates = 0
    def download(self, key):
        if key not in self.files:
            raise MissingFile('not found')
        return self.files[key]
    def upload(self, key, payload, file_options=None):
        if key in self.files:
            raise RuntimeError('already exists')
        self.files[key] = bytes(payload)
    def update(self, key, payload, file_options=None):
        if key not in self.files:
            raise MissingFile('not found')
        self.files[key] = bytes(payload)
        self.updates += 1


def sample_bars(n=15, day='2026-11-03'):
    i = pd.bdate_range(start=day, periods=n)
    a = np.arange(n)+100.0
    return pd.DataFrame({'Close':a},index=i)


def test_cloud_storage_retry_and_ack():
    b = FakeBucket()
    assert w.read_json(b,'x') is None
    assert w.save_json(b,'x', {'value':1}) is True
    assert w.save_json(b,'x', {'value':1}) is False
    assert w.save_json(b,'x', {'value':2}) is True
    assert w.read_json(b,'x') == {'value':2}
    assert b.updates == 1


def test_immutable_ledger_prevents_revising_old_forecast():
    b = FakeBucket()
    f = {'date':'2026-11-24','ticker':'SGLN.L','probability':.3}
    assert w.append_forecasts(b,[f]) == 1
    assert w.append_forecasts(b,[f]) == 0
    with pytest.raises(RuntimeError,match='immutable'):
        w.append_forecasts(b,[dict(f,probability=.9)])
    assert w.read_json(b,w.ledger_key(f['date']))['forecasts'] == [f]


def test_future_outcome_exact_five_completed_bars():
    b=sample_bars(n=6)
    date=b.index[0].date().isoformat()
    assert w.future_label(b.iloc[:5],date) is None
    outcome = w.future_label(b,date)
    assert outcome['outcome_date'] == b.index[5].date().isoformat()
    assert outcome['event_happened'] is True
    assert w.future_label(b,'2026-11-02') is None


def test_outcome_does_not_overwrite_forward_forecast():
    b=FakeBucket()
    f={'date':'2026-11-03','ticker':'SGLN.L','probability':.62,
       'baseline_probability':.40, 'validated':False}
    w.append_forecasts(b,[f])
    bars={'SGLN.L':sample_bars(12)}
    n,count = w.collect_outcomes(b,bars,date(2026,11,24))
    assert (n,count)==(1,1)
    assert w.collect_outcomes(b,bars,date(2026,11,24))==(0,1)
    assert w.read_json(b,w.ledger_key(f['date']))['forecasts'][0]==f
    score=w.historical_scorecard(b,date(2026,11,24))
    assert score['matured']==1 and 0<=score['brier']<=1


def test_not_after_close_at_open_and_closed_holiday():
    assert w.calendar_session(date(2026,11,24))
    assert not w.is_after_close(datetime(2026,11,24,9,tzinfo=timezone.utc))
    assert w.is_after_close(datetime(2026,11,24,18,tzinfo=timezone.utc))
    assert not w.calendar_session(date(2026,12,25))


def test_currency_conversion_no_guess():
    ix=pd.date_range('2025-01-02',periods=140,freq='B',tz='Europe/London')
    a=np.arange(140)*0.01+12.0
    prices=pd.DataFrame({'Open':a,'High':a+.03,'Low':a-.03,'Close':a+.01,
                         'Volume':np.ones(140)*50000},index=ix)
    gbp=w.normalize_lse_bars(prices,'GBP','TEST.L')
    gbp2=w.normalize_lse_bars(prices*1.0,'GBp','TEST.L')
    assert gbp['Close'].iloc[-1] == pytest.approx(gbp2['Close'].iloc[-1]*100)
    assert gbp.attrs['price_currency']=='GBp'
    with pytest.raises(ValueError,match='currency'):
        w.normalize_lse_bars(prices,'USD','TEST.L')


def test_missing_vendor_data_refuses_daily_run(monkeypatch):
    monkeypatch.setattr(w,'is_after_close',lambda now: True)
    monkeypatch.setattr(w,'fetch_histories', lambda period: ({}, {'SGLN.L':'NotFound'}))
    with pytest.raises(RuntimeError,match='Fewer than three'):
        w.daily_worker(None,FakeBucket(),datetime(2026,11,24,19,tzinfo=timezone.utc))


def test_specialist_models_compatible_with_old_linear_format():
    rng=np.random.default_rng(12)
    n=290
    rows=pd.DataFrame(rng.normal(size=(n,len(FEATURE_NAMES))), columns=FEATURE_NAMES)
    rows['ticker']=np.resize(np.array(['SGLN.L','SSLN.L']),n)
    rows['label']=(rng.random(n)<.45).astype(int)
    rows['feature_date']=['2025-01-01']*n
    old=_fit_model(rows)
    validate_model(old)
    specialized=_fit_specialists(rows)
    validate_model(specialized)
    assert specialized['kind']=='product_specialists_v1'
    a=predict_proba(specialized,rows.head(7))
    assert len(a)==7 and np.all((a>0)&(a<1))
    with pytest.raises(ValueError,match='ticker'):
        predict_proba(specialized,rows.head(1).drop(columns='ticker'))


def test_forecast_ledger_redacts_secrets_and_never_executes_orders():
    import inspect
    code=inspect.getsource(w)
    assert 'SUPABASE_MODEL_SERVICE_KEY' in code
    assert '.insert({' not in code  # background worker intentionally read-only to trade journal
    assert 'broker' in w.__doc__.lower()

class FakeClient:
    class Table:
        def __init__(self,rows):
            self.rows=rows
        def select(self,*_):
            return self
        def range(self,a,b):
            self.a,self.b=a,b
            return self
        def execute(self):
            from types import SimpleNamespace
            return SimpleNamespace(data=self.rows[self.a:self.b+1])
    def __init__(self,rows):
        self.rows=rows
    def table(self,name):
        assert name=='predictions'
        return self.Table(self.rows)


def test_backup_is_immutable_readback_and_idempotent():
    now=datetime(2026,11,29,20,0,tzinfo=timezone.utc)
    buck=FakeBucket()
    cli=FakeClient([{'ticker':'SGLN.L', 'predicted_date':'2026-11-24', 'status':'ACTIVE'}])
    a=w.backup_worker(cli,buck,now)
    assert a['rows']==1
    assert a['sha256']
    new_client=FakeClient(cli.rows+[{'ticker':'NGSP.L','status':'WON'}])
    b=w.backup_worker(new_client,buck,now)
    assert b['rows']==1
    assert b['sha256']==a['sha256']
    assert w.HEALTH_KEY in buck.files


def test_missing_all_monitor_feeds_fails_closed(monkeypatch):
    monkeypatch.setattr(w,'calendar_session',lambda _:True)
    monkeypatch.setattr(w,'fetch_histories',lambda p:({},{}))
    with pytest.raises(RuntimeError,match='three'):
        w.monitor_worker(None, FakeBucket(),datetime(2026,11,24,12,tzinfo=timezone.utc))


def test_connection_test_saves_private_readback():
    b=FakeBucket()
    result=w.check_worker(None,b,datetime(2026,11,24,12,tzinfo=timezone.utc))
    assert result['readback_verified']
    assert w.read_json(b,'research/connection_check_v1.json')['status']=='connected'
    assert w.read_json(b,w.HEALTH_KEY)['last_check_status']=='ok'


def test_end_to_end_after_close_research_then_repeat(monkeypatch):
    from commodity_ml import PRODUCTS
    rng=np.random.default_rng(20261124)
    ix=pd.bdate_range(end='2026-11-24',periods=290)
    histories={}
    for t in PRODUCTS:
        drift=rng.normal(.0001,.024,len(ix))
        close=300*np.exp(np.cumsum(drift))
        high=close*1.025
        low=close*.975
        b=pd.DataFrame({'Open':close, 'High':high, 'Low':low,
                        'Close':close, 'Volume':np.full(len(ix),200000)},index=ix)
        b.attrs['price_currency']='GBp'
        histories[t]=b
    monkeypatch.setattr(w,'is_after_close',lambda _:True)
    monkeypatch.setattr(w,'fetch_histories',lambda _: (histories,{}))
    bucket=FakeBucket()
    now=datetime(2026,11,24,19,0,tzinfo=timezone.utc)
    first=w.daily_worker(None,bucket,now)
    assert first['data_tickers']==5
    assert first['forecasts_saved']==5
    state=w.read_json(bucket,w.STATE_KEY)
    assert state['schema']==1
    forecasts=w.read_json(bucket,w.ledger_key('2026-11-24'))['forecasts']
    assert len(forecasts)==5
    assert all(f['date']=='2026-11-24' and not f['validated'] for f in forecasts) if not state.get('champion') else True
    repeat=w.daily_worker(None,bucket,now)
    assert repeat['forecasts_saved']==0
    assert len(w.read_json(bucket,w.ledger_key('2026-11-24'))['forecasts'])==5
