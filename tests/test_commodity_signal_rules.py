"""Checks for no-lookahead ETC price predictions and safe paper journal usage."""
import ast
import math
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
import pytest
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from commodity_signals import evaluate_commodity_signal, STRATEGY_NAME
from test_local_first import trade_env, signal, functions

ROOT = Path(__file__).resolve().parents[1]


def synthetic_bars(n=100, descending=False):
    ix=pd.date_range('2026-06-01', periods=n, freq='B', tz='Europe/London')
    close=np.linspace(100, 145, n) if not descending else np.linspace(145, 100, n)
    close+=np.sin(np.arange(n)*1.85)*0.7
    return pd.DataFrame({'Open':close,'High':close+1,'Low':close-1,
                         'Close':close, 'Volume':np.full(n,2_000_000)}, index=ix)


def test_commodity_candidate_has_explicit_buy_target_stop_and_not_probability():
    df=synthetic_bars()
    s=evaluate_commodity_signal(df, entry_price_pence=float(df['Close'].iloc[-1]))
    assert s.action=='BUY_CANDIDATE', s.reason
    assert 0 < s.stop_pence < float(df['Close'].iloc[-1]) < s.target_pence
    assert 0 <= s.score <= 5
    assert 'rule-based, not ML' in s.reason


def test_no_signal_on_high_volatility_broken_ohlc_or_thin_turnover():
    df=synthetic_bars()
    df_bad=df.copy(); df_bad.iloc[-1,df_bad.columns.get_loc('High')]=1
    assert evaluate_commodity_signal(df_bad,145).action=='WAIT'
    df_thin=df.copy(); df_thin['Volume']=1
    assert evaluate_commodity_signal(df_thin,145).action=='WAIT'
    assert evaluate_commodity_signal(df,0).action=='WAIT'
    assert evaluate_commodity_signal(df.iloc[:15],145).action=='WAIT'


def test_declining_market_signal_waits():
    df=synthetic_bars(descending=True)
    s=evaluate_commodity_signal(df,100)
    assert s.action=='WAIT'


def test_auto_commodity_order_needs_rule_proof_not_share_ai():
    fn,db,state=trade_env()
    fn.__globals__['is_commodity_etc']=lambda t: t=='SGLN.L'
    fn.__globals__['STRATEGY_NAME']=STRATEGY_NAME
    fake=signal(symbol='SGLN.L',capital=1000)
    fake['CommodityRuleQualified']=True
    fake['FeaturesDict']['model_validated']=False
    fake['FeaturesDict']['strategy_id']=STRATEGY_NAME
    assert fn(fake) is True
    assert db.conn.execute('SELECT asset_type FROM trade_journal').fetchone()[0]=='ETC'


def test_auto_commodity_order_rejects_missing_proof():
    fn,db,state=trade_env()
    fn.__globals__['is_commodity_etc']=lambda t: t=='SGLN.L'
    fn.__globals__['STRATEGY_NAME']=STRATEGY_NAME
    fake=signal(symbol='SGLN.L',capital=1000)
    fake['CommodityRuleQualified']=True
    fake['FeaturesDict']['model_validated']=True
    fake['FeaturesDict']['strategy_id']=STRATEGY_NAME
    assert not fn(fake)
    assert db.conn.execute('SELECT COUNT(*) FROM trade_journal').fetchone()[0]==0


def test_options_archive_requires_exact_confirmation():
    import importlib
    import sys
    import types
    sys.modules.setdefault("duckdb", types.SimpleNamespace(connect=None))
    from option_retirement import retire_options
    with pytest.raises(ValueError,match='DELETE OPTIONS'):
        retire_options('missing.db',b'',confirmation='no')
