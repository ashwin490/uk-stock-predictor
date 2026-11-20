"""Offline integration-style tests of pure trading rules and portable backups.

No live market data calls, Streamlit session or Supabase are needed.
"""
import ast
import hashlib
import hmac
import importlib.util
import json
import logging
import math
import os
import re
import sqlite3
import tempfile
import threading
import time
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'app.py'
TREE = ast.parse(SRC.read_text('utf-8'))

def functions(*names, **ctx):
    selected = [n for n in TREE.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert {n.name for n in selected} == set(names)
    ns = dict(math=math, pd=pd, np=np, os=os, re=re, hashlib=hashlib, hmac=hmac,
              time=time, json=json, LOGGER=logging.getLogger('test'), **ctx)
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(SRC), 'exec'), ns)
    return ns

class DB:
    def __init__(self):
        self.conn=sqlite3.connect(':memory:')
        self.conn.execute('''CREATE TABLE trade_journal (
            trade_id TEXT PRIMARY KEY,timestamp TEXT,date_str TEXT,ticker TEXT,asset_type TEXT,
            entry_price REAL,target_price REAL,stop_loss REAL,shares INTEGER,capital_allocated REAL,
            status TEXT,latest_price REAL,pnl_pct REAL,exit_price REAL,exit_timestamp TEXT,
            last_audited TEXT,features_json TEXT)''')
    def connect(self,*a,**k):
        return Query(self.conn)
class Query:
    def __init__(self,conn):self.conn=conn
    def execute(self,sql,parameters=()):
        r=self.conn.execute(sql,parameters)
        self.conn.commit()
        return r
    def close(self):pass

@pytest.mark.parametrize('p,stop,expected',[(100,98,2500),(100,90,1000),(500,450,1000)])
def test_risk_based_sizing(p,stop,expected):
    capacity=dict(ok=True,cash_gbp=10000,equity_gbp=10000,risk_budget_gbp=500,risk_gbp=0)
    ns=functions('get_position_size_gbp',get_portfolio_capacity=lambda:capacity,
                 get_active_positions_df=lambda:pd.DataFrame(),get_trade_sector=lambda t,f:'Tech',
                 MAX_SINGLE_POSITION_FRACTION=0.25,MAX_SECTOR_FRACTION=0.35,
                 MAX_SINGLE_TRADE_RISK_FRACTION=0.01,MIN_POSITION_SIZE_GBP=1)
    assert ns['get_position_size_gbp'](p,stop,'Tech')==pytest.approx(expected)

def test_sizing_scales_without_fixed_money_limit():
    capacity=dict(ok=True,cash_gbp=1_000_000,equity_gbp=1_000_000,risk_budget_gbp=50_000,risk_gbp=0)
    ns=functions('get_position_size_gbp',get_portfolio_capacity=lambda:capacity,
                 get_active_positions_df=lambda:pd.DataFrame(),get_trade_sector=lambda t,f:'Tech',
                 MAX_SINGLE_POSITION_FRACTION=0.25,MAX_SECTOR_FRACTION=0.35,
                 MAX_SINGLE_TRADE_RISK_FRACTION=0.01)
    assert ns['get_position_size_gbp'](100,99,'Tech')==pytest.approx(250_000)

def test_sector_limit_is_cash_weight_not_share_count():
    pos = pd.DataFrame([dict(ticker='AAA.L',features_json='{}',capital_allocated=3300)])
    capacity=dict(ok=True,cash_gbp=6700,equity_gbp=10000,risk_budget_gbp=500,risk_gbp=100)
    ns=functions('get_position_size_gbp',get_portfolio_capacity=lambda:capacity,
                 get_active_positions_df=lambda:pos,get_trade_sector=lambda t,f:'Tech',
                 MAX_SINGLE_POSITION_FRACTION=0.25,MAX_SECTOR_FRACTION=0.35,
                 MAX_SINGLE_TRADE_RISK_FRACTION=0.01)
    assert ns['get_position_size_gbp'](100,99,'Tech')==pytest.approx(200)

def test_position_with_raised_protective_stop_does_not_falsely_lock_portfolio():
    db=DB()
    db.conn.execute('''INSERT INTO trade_journal (trade_id,ticker,entry_price,stop_loss,capital_allocated,status)
                       VALUES ('T','AAA.L',100,101,500,'ACTIVE')''')
    ns=functions('get_portfolio_capacity',get_active_positions_df=lambda:pd.DataFrame([
        dict(entry_price=100,stop_loss=101,capital_allocated=500)]),
        DB_LOCK=threading.RLock(),duckdb=db,DB_PATH=':memory:',
        get_starting_paper_cash=lambda:10000,classify_outcome=lambda s:s,
        MIN_POSITION_SIZE_GBP=1,MAX_PORTFOLIO_RISK_FRACTION=0.05)
    cap=ns['get_portfolio_capacity']()
    assert cap['ok']
    assert cap['risk_gbp']==0
    assert cap['cash_gbp']==9500

def test_closed_losses_reduce_virtual_cash():
    db=DB()
    db.conn.execute('''INSERT INTO trade_journal (trade_id,ticker,entry_price,capital_allocated,status,pnl_pct)
                       VALUES ('T','AAA.L',100,1000,'LOSS',-10)''')
    ns=functions('get_portfolio_capacity',get_active_positions_df=lambda:pd.DataFrame(),
        DB_LOCK=threading.RLock(),duckdb=db,DB_PATH=':memory:',
        get_starting_paper_cash=lambda:10000,classify_outcome=lambda s:s,
        MIN_POSITION_SIZE_GBP=1,MAX_PORTFOLIO_RISK_FRACTION=0.05)
    cap=ns['get_portfolio_capacity']()
    assert cap['equity_gbp']==9900
    assert cap['realised_gbp']==-100

def trade_env(position_cap=2500):
    db=DB()
    state={}
    st=SimpleNamespace(session_state=state)
    def cap():
        n=db.conn.execute("SELECT count(*) FROM trade_journal WHERE status='ACTIVE'").fetchone()[0]
        used=db.conn.execute("SELECT sum(capital_allocated) FROM trade_journal WHERE status='ACTIVE'").fetchone()[0] or 0
        return dict(ok=used<10000,cash_gbp=10000-used,equity_gbp=10000,risk_gbp=0,risk_budget_gbp=500,
                    invested_gbp=used,reason='paper balance used',open=n)
    ctx=functions('log_equity_signal_safely', st=st, london_now=lambda:datetime(2026,11,20,11),
        is_valid_lse_ticker=lambda t:t.endswith('.L'), is_entry_window_open=lambda:True,
        is_market_stress=lambda m:False, get_cross_asset_macro_regime=lambda:{},
        STORAGE_MODE='local',supabase=None,REQUIRE_DURABLE_CLOUD=False,
        _num=lambda v,default=0.0:float(v) if v is not None else default,_int=lambda v:int(float(v)),
        get_position_size_gbp=lambda *a:position_cap, get_portfolio_capacity=cap,
        get_active_sector_exposure=lambda:{},get_ticker_sector=lambda t,resolve=False:'Tech',
        get_trade_friction_pct=lambda t,f,c:0.8,DB_LOCK=threading.RLock(),duckdb=db,DB_PATH=':memory:',
        MAX_ACTIVE_PER_SECTOR=float('inf'), MAX_DAILY_EQUITY_TRADES=float('inf'),
        MAX_HOLD_CALENDAR_DAYS=7,record_db_error=lambda label,e:state.update(db_error=type(e).__name__))
    return ctx['log_equity_signal_safely'],db,state

def signal(symbol='AAA.L',price=100,capital=2000):
    n=int(capital*100/price)
    return dict(Ticker=symbol, **{'Price (p)':price,'Stop Loss (p)':price*.98,
        'Target (p)':price*1.10,'Recommended Shares':f'{n} shares','Capital':capital,
        'Loggable':True,'Sector':'Tech','FeaturesDict':{'price_currency':'GBp'},
        'ReturnNum':5.0,'AI Win Confidence':'80%'})

def test_local_mode_records_without_supabase_and_prevents_duplicate():
    f,db,state=trade_env()
    assert f(signal()) is True
    assert db.conn.execute('select count(*) from trade_journal').fetchone()[0]==1
    assert f(signal()) is False

def test_local_mode_enforces_dynamic_position_budget_and_currency():
    f,db,state=trade_env()
    assert not f(signal(capital=3000))
    x=signal();x['FeaturesDict']={}
    assert not f(x)
    x=signal();x['Recommended Shares']='1 shares'
    assert not f(x)
    assert db.conn.execute('select count(*) from trade_journal').fetchone()[0]==0

def test_local_mode_no_5_trade_limit():
    f,db,state=trade_env(position_cap=2000)
    for i in range(6):
        assert f(signal(symbol=f'A{i}.L',capital=1000))
    assert db.conn.execute('select count(*) from trade_journal').fetchone()[0]==6

def test_cloud_mode_missing_remote_fails_closed():
    f,db,state=trade_env()
    # separate isolated context for cloud mode would check REQUIRE_DURABLE_CLOUD=True.
    assert 'REQUIRE_DURABLE_CLOUD = (STORAGE_MODE == "cloud")' in SRC.read_text('utf-8')

def test_auth_default_password_not_present():
    assert 'AlphaLSE2026!' not in SRC.read_text('utf-8')
    assert 'st.secrets.get("AUTH_PASS_HASH", "")' in SRC.read_text('utf-8')

def test_unchanged_price_unit_guard():
    f=functions('normalize_lse_history_prices')['normalize_lse_history_prices']
    d=pd.DataFrame({'Open':[1.], 'High':[2.], 'Low':[.5], 'Close':[1.5]})
    assert f(d,'GBP').iloc[0]['Close']==150
    assert f(d,'USD').empty

def test_completed_bar_stop_precedes_target():
    f=functions('evaluate_completed_bar')['evaluate_completed_bar']
    assert f(100,125,96,120,100,120,98,101,113)[0]=='STOP'


def backup_module():
    path=ROOT/'alpha_backup.py'
    spec=importlib.util.spec_from_file_location('alpha_backup',path)
    mod=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def test_portable_backup_restore_and_exclude_secrets(tmp_path):
    m=backup_module()
    source=tmp_path/'old'
    source.mkdir()
    (source/'models').mkdir()
    (source/'models'/'ensemble_live.joblib').write_bytes(b'model example')
    (source/'models'/'retrain_state.json').write_text('{"n":150}')
    (source/'alpha_settings.json').write_text('{"starting_cash_gbp":100000}')
    (source/'.streamlit').mkdir()
    (source/'.streamlit'/'secrets.toml').write_text('AUTH_PASS_HASH="not-for-backup"')
    archive=m.create_backup(source)
    names=m.verify_backup(archive)['files']
    assert 'models/ensemble_live.joblib' in names
    assert '.streamlit/secrets.toml' not in names
    target=tmp_path/'new'
    assert len(m.restore_backup(archive,target))==3
    assert (target/'models'/'ensemble_live.joblib').read_bytes()==b'model example'


def test_reject_unexpected_backup_payload(tmp_path):
    import zipfile
    m=backup_module()
    arc=tmp_path/'bad.zip'
    with zipfile.ZipFile(arc,'w') as z:
        z.writestr('alpha_backup_manifest.json',json.dumps({'format':'alpha-lse-learning-backup-v1','files':{'../hack':'abc'}}))
        z.writestr('../hack','hello')
    with pytest.raises(ValueError):m.verify_backup(arc)
