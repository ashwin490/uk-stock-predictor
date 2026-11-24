"""Autonomous end-of-day UK-share research and independent paper-journal audit.

GitHub Actions runner, not a broker and NOT the Streamlit original share-buy model.
Uses private Supabase Storage; the original model bundle is never overwritten.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone, timedelta
import hashlib
import json
import logging
import os
import re
from pathlib import Path
import math
import pandas as pd

from commodity_worker import (connection_from_env, read_json, save_json, calendar_session,
                              is_after_close, future_label, alert)
from share_research import (training_rows, train_candidate, latest_predictions,
                            predict, reconcile_paper_equity_rows, HORIZON, GAIN, FEATURES)

LOG = logging.getLogger('alpha_lse.shares')
BASE = 'research/shares'
STATE = BASE+'/model_state_v1.json'
FORECASTS = BASE+'/latest_forecasts_v1.json'
HEALTH = BASE+'/worker_health_v1.json'
AUDIT = BASE+'/accounting_audit_v1.json'
SCORE = BASE+'/scorecard_v1.json'


def public_constituents(page_html: str) -> list[str]:
    """Public FTSE table reader. Silently falls back to bundled rosters if changed."""
    from html import unescape
    tables=re.findall(r'<table[^>]+(?:id=["\']constituents["\']|class=["\'][^"\']*wikitable)[^>]*>.*?</table>',page_html,re.I|re.S)
    for table in tables:
        raw_rows=re.findall(r'<tr[^>]*>(.*?)</tr>',table,re.I|re.S)
        rows=[]
        for raw in raw_rows:
            cols=re.findall(r'<t[hd][^>]*>(.*?)</t[hd]>',raw,re.I|re.S)
            rows.append([unescape(re.sub(r'<[^>]+>','',col)).strip() for col in cols])
        if not rows:continue
        head=[x.lower() for x in rows[0]]
        ticker_col=next((i for i,k in enumerate(head) if 'ticker' in k or 'epic' in k),None)
        if ticker_col is None:continue
        result=[]
        for row in rows[1:]:
            if len(row)<=ticker_col:continue
            ticker=re.sub(r'\s+','',row[ticker_col]).upper().replace('.','-')
            if re.fullmatch(r'[A-Z0-9-]{1,12}',ticker):
                result.append(ticker+'.L')
        if len(result)>=70:
            return list(dict.fromkeys(result))
    return []


def current_public_ftse_tickers() -> list[str]:
    import requests
    out=[]
    for url,minimum in (("https://en.wikipedia.org/wiki/FTSE_100_Index",70),
                        ("https://en.wikipedia.org/wiki/FTSE_250_Index",140)):
        try:
            r=requests.get(url,timeout=8,headers={'User-Agent':'ALPHA-LSE/1.0 (research ticker roster)'})
            r.raise_for_status()
            parsed=public_constituents(r.text)
            if len(parsed)>=minimum:
                out.extend(parsed)
        except Exception:
            continue
    return list(dict.fromkeys(out))


def available_universe(app_file: str | Path | None = None) -> list[str]:
    """Use existing bundled FTSE lists without importing and starting Streamlit."""
    if app_file is None:
        app_file=Path(__file__).with_name('app.py')
    src=Path(app_file).read_text(encoding='utf-8')
    tickers=[]
    for label in ('FTSE100_BUILTIN','FTSE250_BUILTIN'):
        m=re.search(r'\b'+label+r'\s*=\s*"""(.*?)"""',src,re.S)
        if not m:
            raise ValueError('Missing existing UK universe '+label)
        for line in m.group(1).splitlines():
            key=line.split('|',1)[0].strip().upper().replace('.','-')
            if re.fullmatch(r'[0-9A-Z-]{1,12}',key):
                tickers.append(key+'.L')
    if os.getenv('ALPHA_REFRESH_UNIVERSE','0').lower() in ('1','true','yes'):
        fresh=current_public_ftse_tickers()
        # Live symbols first, bundled roster only for provider outages. Bad
        # symbols are rejected by verified Yahoo history/currency checks.
        if len(fresh)>=70:
            tickers=fresh + tickers
    # Price/currency verification per symbol is intentionally bounded to
    # remain within hosted scheduler and data-provider rate limits.
    maximum=max(20,min(350,int(os.getenv('ALPHA_SHARE_MAX_SYMBOLS','120'))))
    return list(dict.fromkeys(tickers))[:maximum]


def fetch_one(ticker: str, period='3y'):
    import yfinance as yf
    from commodity_worker import normalize_lse_bars
    stock=yf.Ticker(ticker)
    bars=stock.history(period=period,auto_adjust=True,repair=False,timeout=18)
    metadata=getattr(stock,'history_metadata',{}) or {}
    currency=metadata.get('currency') if isinstance(metadata,dict) else None
    if not currency:
        currency=stock.fast_info.get('currency')
    cleaned=normalize_lse_bars(bars,currency,ticker)
    # Corporate actions: adjusted history cannot be equated with an old fill.
    # Use only for forward percentage changes, never repricing old paper trades.
    return cleaned


def fetch_universe(tickers: list[str],period='3y'):
    data={}; bad={}
    with ThreadPoolExecutor(max_workers=10) as exe:
        futures={exe.submit(fetch_one,t,period):t for t in tickers}
        for f in as_completed(futures):
            t=futures[f]
            try:
                data[t]=f.result()
            except Exception as err:
                bad[t]=type(err).__name__
    return data,bad


def append_predictions(bucket, forecasts):
    groups={}
    for f in forecasts:
        groups.setdefault(f['date'][:7],[]).append(f)
    added=0
    for month,items in groups.items():
        key=f'{BASE}/predictions/{month}.json'
        ledger=read_json(bucket,key,{'schema':1,'forecasts':[]})
        if ledger.get('schema')!=1 or not isinstance(ledger.get('forecasts'),list):
            raise RuntimeError('Unrecognised share prediction ledger')
        old={r['date']+':'+r['ticker']:r for r in ledger['forecasts']}
        for row in items:
            idx=row['date']+':'+row['ticker']
            # First stored prediction is authoritative; no hindsight revisions.
            if idx not in old:
                old[idx]=row; added+=1
        ledger['forecasts']=[old[k] for k in sorted(old)]
        save_json(bucket,key,ledger)
    return added


def mature_outcomes(bucket,histories,today):
    # Monthly sharding avoids the 4MB state limit as history accumulates.
    months=pd.period_range(pd.Timestamp(today)-pd.DateOffset(months=13), pd.Timestamp(today),freq='M').strftime('%Y-%m')
    new=0;total=0
    for month in months:
        preds=read_json(bucket,f'{BASE}/predictions/{month}.json')
        if not preds:
            continue
        key=f'{BASE}/outcomes/{month}.json'
        doc=read_json(bucket,key,{'schema':1,'outcomes':[]})
        if doc.get('schema')!=1:
            raise RuntimeError('Invalid share outcome ledger')
        existing={r['date']+':'+r['ticker']:r for r in doc['outcomes']}
        for f in preds['forecasts']:
            id_=f['date']+':'+f['ticker']
            if id_ in existing or f['ticker'] not in histories:
                continue
            outcome=future_label(histories[f['ticker']],f['date'])
            if outcome is not None:
                existing[id_]={'date':f['date'],'ticker':f['ticker'],'probability':f['probability'],
                               'baseline_probability':f['baseline_probability'],
                               'model_version':f['model_version'],
                               'shadow_probability':f.get('shadow_probability'),
                               'shadow_version':f.get('shadow_version'),**outcome}
                new+=1
        total+=len(existing)
        doc['outcomes']=[existing[k] for k in sorted(existing)]
        save_json(bucket,key,doc)
    return new,total


def scorecard(bucket,today,champion_version=None):
    outcomes=[]
    months=pd.period_range(pd.Timestamp(today)-pd.DateOffset(months=13),pd.Timestamp(today),freq='M').strftime('%Y-%m')
    for month in months:
        doc=read_json(bucket,f'{BASE}/outcomes/{month}.json') or {}
        outcomes.extend(doc.get('outcomes',[]))
    if not outcomes:
        return {'matured':0}
    vals=[]
    for f in outcomes:
        p=float(f['probability']); b=float(f['baseline_probability']); y=int(f['event_happened'])
        if min(p,b)<0 or max(p,b)>1:
            continue
        vals.append((p,b,y,f['ticker']))
    if not vals: return {'matured':0}
    n=len(vals)
    groups={}
    version_groups={}
    shadow_groups={}
    for f in outcomes:
        try:
            row=(float(f['probability']),float(f['baseline_probability']),int(f['event_happened']))
            t=f['ticker']
            v=str(f.get('model_version') or 'unknown')
            groups.setdefault(t,[]).append(row)
            version_groups.setdefault(v,[]).append(row)
            if f.get('shadow_probability') is not None and f.get('shadow_version'):
                shadow_groups.setdefault(str(f['shadow_version']),[]).append(
                    (row[0],float(f['shadow_probability']),row[2]))
        except (ValueError,TypeError,KeyError):
            continue
    def calc(rows):
        rows=[r[:3] for r in rows]
        return {'matured':len(rows),
                'brier':round(sum((p-y)**2 for p,b,y in rows)/len(rows),5),
                'baseline_brier':round(sum((b-y)**2 for p,b,y in rows)/len(rows),5),
                'predicted_rises':sum(p>=.57 for p,b,y in rows),
                'correct_rises':sum((p>=.57) and y for p,b,y in rows)}
    result=calc(vals)
    result['by_version']={version:calc(rows) for version,rows in version_groups.items()}
    shadow_reports={}
    for version,rows in shadow_groups.items():
        if rows:
            shadow_reports[version]={
                'n':len(rows),
                'shadow_brier':round(sum((sh-y)**2 for p,sh,y in rows)/len(rows),5),
                'champion_brier':round(sum((p-y)**2 for p,sh,y in rows)/len(rows),5)}
    result['shadow_by_version']=shadow_reports
    result['by_symbol']=[{'ticker':t,**calc(rows)} for t,rows in sorted(groups.items())]
    current=result['by_version'].get(str(champion_version),{})
    result['forward_filter_eligible']=(
        current.get('matured',0)>=100 and
        current.get('brier',1)+.01<current.get('baseline_brier',0))
    return result


def health(bucket,mode:str,ok:bool,today,details=None):
    old=read_json(bucket,HEALTH,{},fresh=True)
    old.update({'schema':1,'last_mode':mode,'last_run_utc':datetime.now(timezone.utc).isoformat(),
                'last_status':'ok' if ok else 'failed','last_detail':details or {}})
    if mode=='daily' and ok:
        old.update({'last_daily_utc':old['last_run_utc'],'last_daily_status':'ok',
                    'last_daily_date':today.isoformat()})
    save_json(bucket,HEALTH,old)


def daily(bucket,now,*,fetcher=fetch_universe):
    if not is_after_close(now):
        return {'status':'not_after_verified_exchange_close'}
    from zoneinfo import ZoneInfo
    local=now.astimezone(ZoneInfo('Europe/London'))
    day=local.date().isoformat()
    tickers=available_universe()
    histories,failures=fetcher(tickers,'3y')
    today_data={t:b for t,b in histories.items() if len(b) and b.index[-1].date().isoformat()==day}
    if len(today_data)<max(12,min(30,len(tickers)//4)):
        raise RuntimeError('Insufficient complete same-day UK share price feeds')
    # Training data must be complete and adjusted, never today's partial bars.
    import exchange_calendars as xcals
    cal=xcals.get_calendar('XLON')
    start=min(b.index[0].date() for b in histories.values())
    sessions=cal.sessions_in_range(pd.Timestamp(start),pd.Timestamp(local.date()))
    prepared=[training_rows(t,b,sessions) for t,b in histories.items()]
    parts=[p for p in prepared if not p.empty]
    if not parts:
        raise RuntimeError('No dated price labels for share research')
    df=pd.concat(parts,ignore_index=True)
    state=read_json(bucket,STATE,{'schema':1,'champion':None,'research_model':None})
    last=state.get('last_training_date')
    due=not last or (local.date()-pd.Timestamp(last).date()).days >= 30
    decision='retained previous model'
    if due:
        candidate,meta=train_candidate(df)
        if candidate:
            report=candidate['report']
            model=candidate['candidate']
            version=hashlib.sha256(json.dumps(model,sort_keys=True).encode('utf-8')).hexdigest()[:12]
            model['version']=version
            # A permanent versioned snapshot lets future runs recover or
            # audit each challenger without relying on the mutable state file.
            save_json(bucket,f'{BASE}/models/{day}-{version}.json',model,immutable=True)
            if candidate['approved']:
                if state.get('champion') is None:
                    state['champion']=model
                    state['champion_paused']=False
                    decision='first share research champion promoted after holdout test'
                else:
                    # Do not compare a newly retrained model on dates that the
                    # existing champion has already trained on. Track parallel
                    # shadow forecasts with future outcomes instead.
                    state['challenger']=model
                    decision='challenger accepted for forward shadow evaluation'
            else:
                decision='challenger rejected; lower quality than baseline'
            state.update({'research_model':model,'last_training_date':day,
                          'last_evaluation':report,'last_decision':decision})
            # Audit model promotion before persistent storage mutation.
            save_json(bucket,STATE,state)
            save_json(bucket,f'{BASE}/model_trials/{day}.json',
                      {'schema':1,'report':report,'decision':decision},immutable=True)
        else:
            decision=meta['status']
            state['last_decision']=decision
            # No attempt to invent a model from thin data.
    model=state.get('champion') or state.get('research_model')
    new_predictions=latest_predictions(model,today_data,day) if model else []
    challenger=state.get('challenger')
    if state.get('champion') and challenger and new_predictions:
        shadow={r['ticker']:r for r in latest_predictions(challenger,today_data,day)}
        for p in new_predictions:
            if p['ticker'] in shadow:
                p['shadow_probability']=shadow[p['ticker']]['probability']
                p['shadow_version']=challenger.get('version')
    for p in new_predictions:
        p['validated']=bool(state.get('champion') and not state.get('champion_paused'))
        p['strategy']='share_price_event_v1'
    added=append_predictions(bucket,new_predictions)
    save_json(bucket,FORECASTS,{'schema':1,'as_of':day,'forecasts':new_predictions,
                                'universe_count':len(tickers), 'verified_count':len(today_data),
                                'decision':decision,'failed_feed_count':len(failures)})
    matured,count=mature_outcomes(bucket,histories,local.date())
    sc=scorecard(bucket,local.date(),
                 champion_version=(state.get('champion') or {}).get('version'))
    selected_shadow=(sc.get('shadow_by_version') or {}).get(
        (state.get('challenger') or {}).get('version'),{})
    if (state.get('champion') and state.get('challenger')
            and selected_shadow.get('n',0)>=100
            and selected_shadow.get('shadow_brier',1)+.01 < selected_shadow.get('champion_brier',0)
            and selected_shadow.get('shadow_brier',1)+.008 < sc.get('baseline_brier',0)):
        state['previous_champion']=state['champion']
        state['champion']=state.pop('challenger')
        state['champion_paused']=False
        state['last_forward_promotion_date']=day
        sc['current_champion_version']=state['champion']['version']
        sc['forward_filter_eligible']=False # New version requires its own future evidence.
        save_json(bucket,STATE,state)
        alert('Validated UK share research challenger was promoted after forward comparison')
    if (sc.get('matured',0)>=100 and
            sc['brier']>sc['baseline_brier']+.03 and state.get('champion')):
        if not state.get('champion_paused'):
            state['champion_paused']=True
            save_json(bucket,STATE,state)
            alert('UK-share forward forecast quality deteriorated; ML advisory paused')
    save_json(bucket,SCORE,{'schema':1,'as_of':day,**sc})
    details={'coverage':len(today_data),'universe':len(tickers),
             'saved':added,'matured_added':matured,'total_resolved':count,
             'decision':decision}
    health(bucket,'daily',True,local.date(),details)
    return details


def accounting(bucket,client,now):
    rows=[];start=0
    while True:
        batch=client.table('predictions').select('*').range(start,start+999).execute().data
        if batch is None: raise RuntimeError('Paper journal read failed')
        rows.extend(batch)
        if len(batch)<1000: break
        start+=1000
        if start>100000:raise RuntimeError('Paper journal exceeds audit bound')
    report=reconcile_paper_equity_rows(rows)
    report.update({'schema':1,'audited_utc':now.isoformat(),
                   'interpretation':'Independent paper journal estimate consistency; not broker fill verification'})
    save_json(bucket,AUDIT,report)
    if report['mismatch_count'] or report['duplicate_count']:
        alert(f'UK share paper journal needs review: {report["mismatch_count"]} mismatches; '
              f'{report["duplicate_count"]} duplicate rows')
    health(bucket,'audit',True,now.date(),{'status':report['status'],'audited':report['audited']})
    return report


def check(bucket,now):
    name=BASE+'/connection_check_v1.json'
    save_json(bucket,name,{'schema':1,'checked_utc':now.isoformat(),'connected':True})
    health(bucket,'check',True,now.date(),{'status':'connected'})
    return {'status':'connected'}


def monitor(bucket,now):
    h=read_json(bucket,HEALTH) or {}
    last=h.get('last_daily_utc')
    if not last: raise RuntimeError('No successful share research run recorded')
    import exchange_calendars as xcals
    from zoneinfo import ZoneInfo
    last_date=datetime.fromisoformat(last).astimezone(ZoneInfo('Europe/London')).date()
    current=now.astimezone(ZoneInfo('Europe/London')).date()
    if current<last_date:
        raise RuntimeError('Share research heartbeat is future dated')
    if current>last_date:
        schedule=xcals.get_calendar('XLON').sessions_in_range(
            pd.Timestamp(last_date+timedelta(days=1)),pd.Timestamp(current))
        if len(schedule)>5:
            raise RuntimeError('Share research heartbeat is older than five UK exchange sessions')
    return {'status':'healthy','last_daily_utc':last}


def main(argv=None):
    p=argparse.ArgumentParser()
    p.add_argument('--mode',choices=['check','daily','audit','monitor'],required=True)
    args=p.parse_args(argv)
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s')
    now=datetime.now(timezone.utc)
    client,bucket=connection_from_env()
    try:
        if args.mode=='check':result=check(bucket,now)
        elif args.mode=='daily':result=daily(bucket,now)
        elif args.mode=='audit':result=accounting(bucket,client,now)
        else: result=monitor(bucket,now)
        LOG.info('Worker outcome: %s',str(result)[:800]); return 0
    except Exception as exc:
        LOG.exception('UK share worker stage %s failed (%s)',args.mode,type(exc).__name__)
        alert('UK share background task '+args.mode+' failed: '+type(exc).__name__)
        return 1

if __name__=='__main__':
    raise SystemExit(main())
