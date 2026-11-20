"""New regression tests for a £50k paper account, market selloffs and legacy history."""
import ast
import math
import json
import os
import threading
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from test_local_first import functions, signal, trade_env


def test_50k_default_and_old_default_migration(tmp_path):
    sfile=tmp_path/'alpha_settings.json'
    ns=functions('load_paper_settings','get_starting_paper_cash',
        SETTINGS_PATH=sfile,DEFAULT_PAPER_STARTING_GBP=50_000.0,
        LEGACY_DEFAULT_PAPER_GBP=10_000.0,MIN_POSITION_SIZE_GBP=1)
    assert ns['get_starting_paper_cash']()==50_000.0
    sfile.write_text(json.dumps({'starting_cash_gbp':10_000}),encoding='utf-8')
    assert ns['get_starting_paper_cash']()==50_000.0
    sfile.write_text(json.dumps({'starting_cash_gbp':10_000,'starting_cash_customised':True}),encoding='utf-8')
    assert ns['get_starting_paper_cash']()==10_000.0
    sfile.write_text(json.dumps({'starting_cash_gbp':100_000}),encoding='utf-8')
    assert ns['get_starting_paper_cash']()==100_000.0


def test_sharp_market_fall_is_warning_by_default():
    f=functions('market_stress_blocks_entries',is_market_stress=lambda m: m.get('ftse_5d',0)<=-3,
                get_market_stress_policy=lambda:'qualified')['market_stress_blocks_entries']
    assert not f({'ftse_5d':-6})
    g=functions('market_stress_blocks_entries',is_market_stress=lambda m:m.get('ftse_5d',0)<=-3,
                get_market_stress_policy=lambda:'pause')['market_stress_blocks_entries']
    assert g({'ftse_5d':-6})
    assert not g({'ftse_5d':1})


def test_qualified_paper_trade_is_allowed_in_selloff_but_weak_prediction_is_not():
    f,db,state=trade_env()
    f.__globals__['is_market_stress']=lambda macro:True
    f.__globals__['market_stress_blocks_entries']=lambda macro:False
    good=signal(symbol='AAA.L',capital=2000)
    assert f(good)
    weak=signal(symbol='BBB.L',capital=2000)
    weak['Qualified']=False
    assert not f(weak)
    assert db.conn.execute('select count(*) from trade_journal').fetchone()[0]==1
    assert 'passed all checks' in state['last_log_reason']
    experimental=signal(symbol='CCC.L',capital=2000)
    experimental['Qualified']=False
    experimental['FeaturesDict']['learning_pick']=True
    assert not f(experimental), 'even explicit learning candidates are blocked during selloffs'


def test_selloff_pause_mode_blocks_even_qualified_paper_ideas():
    f,db,state=trade_env()
    f.__globals__['is_market_stress']=lambda macro:True
    f.__globals__['market_stress_blocks_entries']=lambda macro:True
    assert not f(signal())
    assert db.conn.execute('select count(*) from trade_journal').fetchone()[0]==0


def test_weak_manual_prediction_is_blocked_even_in_normal_conditions():
    f,db,state=trade_env()
    s=signal();s['Qualified']=False
    assert not f(s)
    assert not db.conn.execute('select * from trade_journal').fetchall()


def test_history_auto_import_when_local_empty_and_cloud_backup_exists():
    state={}
    counts={'total':0,'opt_total':0}
    remote=object()
    def hydrate():
        counts.update(total=18,opt_total=7)
        state['hydrate_info']={'remote':18,'bad_rows':0}
    ctx=functions('restore_legacy_history_if_empty',STORAGE_MODE='local',
        st=SimpleNamespace(session_state=state),journal_counts=lambda:dict(counts),
        get_supabase_client=lambda:remote,hydrate_duckdb_from_supabase=hydrate,
        record_db_error=lambda *a:None,supabase=None)
    ctx['restore_legacy_history_if_empty']()
    assert counts=={'total':18,'opt_total':7}
    assert 'Recovered 18' in state['history_notice']
    assert ctx['supabase'] is None, 'restoration may not permanently switch local writes to cloud'
    ctx['restore_legacy_history_if_empty']()
    assert counts['total']==18


def test_never_import_old_cloud_records_over_existing_local_journal():
    state={}
    calls=[]
    ctx=functions('restore_legacy_history_if_empty',STORAGE_MODE='local',
        st=SimpleNamespace(session_state=state),journal_counts=lambda:{'total':3,'opt_total':0},
        get_supabase_client=lambda:calls.append('connect'),
        hydrate_duckdb_from_supabase=lambda:calls.append('import'),
        supabase=None)
    ctx['restore_legacy_history_if_empty']()
    assert calls==[]


def test_local_empty_journal_without_cloud_secrets_explains_zeros():
    state={}
    ctx=functions('restore_legacy_history_if_empty',STORAGE_MODE='local',
        st=SimpleNamespace(session_state=state),journal_counts=lambda:{'total':0,'opt_total':0},
        get_supabase_client=lambda:None,supabase=None)
    ctx['restore_legacy_history_if_empty']()
    assert 'SUPABASE_URL' in state['history_notice']


def test_storage_mode_allows_streamlit_secrets_and_env_override(monkeypatch):
    ns=functions('_configured_storage_mode', st=SimpleNamespace(secrets={'ALPHA_STORAGE_MODE':'cloud'}))
    monkeypatch.delenv('ALPHA_STORAGE_MODE',raising=False)
    assert ns['_configured_storage_mode']()=='cloud'
    monkeypatch.setenv('ALPHA_STORAGE_MODE','local')
    assert ns['_configured_storage_mode']()=='local'
