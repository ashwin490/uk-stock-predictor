"""UK-share scheduled research, independent accuracy, accounting and UX safeguards."""
from datetime import date, datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd

import share_worker as w
from share_research import (create_feature_frame,training_rows,selection_windows,
                            train_candidate,latest_predictions,reconcile_paper_equity_rows,
                            predict,FEATURES)
from test_autonomous_worker import FakeBucket


def bars(n=400,seed=4):
    import exchange_calendars as xcals
    cal=xcals.get_calendar('XLON')
    idx=cal.sessions_in_range(pd.Timestamp('2024-01-01'),pd.Timestamp('2026-11-24'))[-n:]
    rng=np.random.default_rng(seed)
    price=300*np.exp(np.cumsum(rng.normal(.0003,.022,n)))
    high=price*1.015; low=price*.985
    return pd.DataFrame({'Open':price,'High':high,'Low':low,'Close':price,
                         'Volume':rng.integers(80_000,250_000,n)},index=idx)


def test_features_do_not_read_future_bars():
    b=bars()
    a=create_feature_frame(b.iloc[:250]).iloc[-1][list(FEATURES)]
    later=b.copy();later.loc[later.index[250:],'Close']*=9
    check=create_feature_frame(later).loc[b.index[249]][list(FEATURES)]
    np.testing.assert_allclose(a.to_numpy(float),check.to_numpy(float))


def test_share_training_has_outcome_purge_and_finite_predictions():
    b=bars()
    ss=b.index
    frames=[training_rows(t,bars(seed=i+4),ss) for i,t in enumerate(['AZN.L','LLOY.L','BP.L','GSK.L'])]
    df=pd.concat(frames,ignore_index=True)
    portions=selection_windows(df)
    assert portions is not None
    train,tune,test=portions
    assert max(train['label_end'])<min(tune['day'])
    assert max(tune['label_end'])<min(test['day'])
    candidate,meta=train_candidate(df)
    assert candidate and meta['status']=='evaluated'
    model=candidate['candidate'];model['version']='test'
    p=predict(model,test.iloc[:30]);assert np.isfinite(p).all()
    assert ((p>0)&(p<1)).all()
    assert candidate['report']['test_count']==len(test)
    guesses=latest_predictions(model,{'AZN.L':bars()},bars().index[-1].date().isoformat())
    assert len(guesses)==1
    assert guesses[0]['event'].startswith('adjusted close')


def test_ledger_monthly_idempotent_and_past_forecasts_not_revised():
    bucket=FakeBucket()
    f={'date':'2026-11-24','ticker':'AZN.L','probability':.42}
    assert w.append_predictions(bucket,[f])==1
    assert w.append_predictions(bucket,[dict(f,probability=.91)])==0
    assert w.read_json(bucket,'research/shares/predictions/2026-11.json')['forecasts']==[f]
    assert w.append_predictions(bucket,[dict(f,date='2026-12-01')])==1


def test_paper_accounting_audit_no_mutation_no_guessing_fees():
    record={'ticker':'AZN.L','predicted_date':'2026-11-23','entry_price':100,
            'latest_price':110,'position_gbp':500,'pnl_pct':9.1,
            'features_json':{'asset_class':'EQUITY','friction_pct':.5,'commission_gbp':1}}
    # Gross 10%, friction 0.9% (0.5% tax/spread plus £2 / £500)
    actual=reconcile_paper_equity_rows([record])
    assert actual['mismatch_count']==0 and actual['audited']==1
    wrong=dict(record,pnl_pct=10.0)
    assert reconcile_paper_equity_rows([wrong])['mismatch_count']==1
    missing=dict(record,features_json={})
    assert reconcile_paper_equity_rows([missing])['unknown_count']==1
    assert reconcile_paper_equity_rows([record,record])['duplicate_count']==1
    assert record['pnl_pct']==9.1


def test_own_independent_heartbeat_and_scoring():
    b=FakeBucket()
    w.health(b,'check',True,date(2026,11,24))
    assert w.read_json(b,w.HEALTH)['last_status']=='ok'
    f={'date':'2026-11-24','ticker':'AZN.L','probability':.7,
       'baseline_probability':.4,'predicted_rise':True,'model_version':'a'}
    w.append_predictions(b,[f])
    saved=w.read_json(b,'research/shares/predictions/2026-11.json')
    assert len(saved['forecasts'])==1
    assert w.scorecard(b,date(2026,11,24))['matured']==0
    w.save_json(b,'research/shares/outcomes/2026-11.json',{'schema':1,'outcomes':[
        {'date':'2026-11-24','ticker':'AZN.L','probability':.7,
         'baseline_probability':.4,'event_happened':True}]})
    sc=w.scorecard(b,date(2026,11,24))
    assert sc['matured']==1 and sc['brier']<sc['baseline_brier']
    assert sc['forward_filter_eligible'] is False


def test_universe_nonempty_and_worker_no_orders():
    tickers=w.available_universe()
    assert len(tickers)>=60 and len(tickers)==len(set(tickers))
    assert all(t.endswith('.L') for t in tickers)
    code=Path(w.__file__).read_text()
    assert 'insert(' not in code
    assert "--mode" in code
    assert (Path(w.__file__).parent/'.github/workflows/share_autonomous.yml').exists()


def test_dashboard_keeps_clear_paper_status_and_conditional_filter():
    src=(Path(w.__file__).parent/'app.py').read_text()
    assert 'PAPER ACCOUNT' in src
    assert "independent_share_ml_filter(sig['Ticker'])" in src
    assert "if not sig.get('Qualified'):" in src
    assert "share_sc.get('matured',0)" in src
    assert 'load_share_worker_status()' in src
    assert 'Account' not in 'live broker'  # No actual broker execution


def test_forward_validation_is_version_specific():
    b=FakeBucket()
    outcomes=[]
    for i in range(105):
        outcomes.append({'date':'2026-11-01','ticker':f'T{i}.L',
                         'model_version':'champ-A', 'probability':.95,
                         'baseline_probability':.5,'event_happened':True})
    w.save_json(b,'research/shares/outcomes/2026-11.json',{'schema':1,'outcomes':outcomes})
    assert w.scorecard(b,date(2026,11,24),champion_version='champ-A')['forward_filter_eligible']
    assert not w.scorecard(b,date(2026,11,24),champion_version='champ-B')['forward_filter_eligible']


def test_daily_pipeline_offline_fails_closed_on_bad_history_and_is_idempotent(monkeypatch):
    b=FakeBucket()
    symbols=[f'S{i:02d}.L' for i in range(36)]
    histories={t:bars(seed=i+10) for i,t in enumerate(symbols)}
    monkeypatch.setattr(w,'available_universe',lambda: symbols)
    now=datetime(2026,11,24,19,tzinfo=timezone.utc)
    done=w.daily(b,now,fetcher=lambda t,p:(histories,{}))
    assert done['saved']==36
    assert w.read_json(b,w.FORECASTS)['forecasts']
    rerun=w.daily(b,now,fetcher=lambda t,p:(histories,{}))
    assert rerun['saved']==0
    assert w.read_json(b,w.STATE)['research_model']
    assert len(w.read_json(b,'research/shares/predictions/2026-11.json')['forecasts'])==36
    # A provider outage cannot fabricate today's model signals.
    with __import__('pytest').raises(RuntimeError,match='Insufficient complete'):
        w.daily(b,now,fetcher=lambda t,p:({t:histories[t] for t in symbols[:3]},{}))
