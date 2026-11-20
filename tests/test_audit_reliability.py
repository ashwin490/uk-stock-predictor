"""Regression checks for the 2026-11-20 targeted reliability release."""
from datetime import datetime, date, timedelta
import logging
import math
from types import SimpleNamespace
import pandas as pd
import pytest

from test_local_first import functions


def _intraday():
    ns = functions('evaluate_live_equity_bar', 'classify_stop_exit', 'calc_net_equity_pnl_pct', BE_TOLERANCE_PCT=0.5)
    return ns['evaluate_live_equity_bar']


def test_partial_prior_day_must_be_replayed_at_next_visit():
    f = functions('needs_completed_bar_replay', datetime=datetime, dtime=datetime.min.time().__class__)['needs_completed_bar_replay']
    assert f(date(2026, 11, 19), date(2026, 11, 17), '2026-11-19 11:12:00', date(2026, 11, 20))
    assert not f(date(2026, 11, 19), date(2026, 11, 17), '2026-11-19 17:00:00', date(2026, 11, 20))
    assert f(date(2026, 11, 18), date(2026, 11, 17), '2026-11-17 13:00:00', date(2026, 11, 20))
    assert not f(date(2026, 11, 17), date(2026, 11, 17), '2026-11-17 13:00:00', date(2026, 11, 20))
    assert not f(date(2026, 11, 20), date(2026, 11, 17), '2026-11-19 13:00:00', date(2026, 11, 20))


def test_half_day_audit_after_close_is_not_replayed():
    f = functions('needs_completed_bar_replay', datetime=datetime, dtime=datetime.min.time().__class__)['needs_completed_bar_replay']
    assert not f(date(2026,12,24), date(2026,12,21), '2026-12-24 13:00:00', date(2026,12,29))
    assert f(date(2026,12,24), date(2026,12,21), '2026-12-24 11:45:00', date(2026,12,29))


def test_entry_day_ignores_high_low_before_purchase():
    f = _intraday()
    # Day high touched target BEFORE the simulated 15:00 entry, current price did not.
    state, fill, stop = f(100, 110, 95, 100, 115, 91, 101, 106, 0.8, True, False)
    assert state == 'ACTIVE' and fill == 0 and stop == 95


def test_incomplete_day_does_not_ratchet_stop():
    f = _intraday()
    state, fill, new_stop = f(100, 115, 95, 108, 110, 99, 101, 107, 0.8, False, False)
    assert state == 'ACTIVE' and new_stop == 95
    state, fill, new_stop = f(100, 115, 95, 108, 110, 99, 101, 107, 0.8, False, True)
    assert state == 'ACTIVE' and new_stop == 101


def test_old_protective_stop_must_consider_day_low():
    f = _intraday()
    # The 101p stop was in force from the previous day; low=99p triggers it.
    state, fill, stop = f(100, 120, 101, 112, 115, 99, 102, 114, 0.8, False, False)
    assert 'LOSS' in state or 'BREAK-EVEN' in state
    assert fill == 101


def test_stop_first_and_gap_down_execution():
    f = _intraday()
    state, fill, stop = f(100, 110, 96, 98, 115, 88, 101, 107, 0.8,
                          False, False, opening=91)
    assert 'LOSS' in state
    assert fill <= 91


def test_invalid_ohlc_rows_excluded():
    f = functions('normalize_lse_history_prices')['normalize_lse_history_prices']
    import numpy as np
    ns = f.__globals__
    ns['np']=np
    ns['LOGGER']=logging.getLogger('test')
    df = pd.DataFrame({'Open':[1.0,3.0,2.0], 'High':[2.0,2.5,2.5],
                       'Low':[0.5,1.0,0.0], 'Close':[1.5,2.0,2.2]})
    valid = f(df,'GBP')
    assert len(valid) == 1
    assert valid.iloc[0]['Close'] == 150.0


def test_retraining_has_date_embargo_and_shared_day_groups():
    f = functions('make_time_purged_train_test', timedelta=timedelta, datetime=datetime)['make_time_purged_train_test']
    # Multiple shares with identical entry dates: never mix those between sets.
    start = date(2026,1,1)
    recs = [(str(start+timedelta(days=d)),[float(x)],int(x%2)) for d in range(80) for x in range(3)]
    train,test,split_day = f(recs)
    assert {r[0] for r in train}.isdisjoint({r[0] for r in test})
    assert max(datetime.strptime(r[0],'%Y-%m-%d').date() for r in train) < datetime.strptime(split_day,'%Y-%m-%d').date()-timedelta(days=7)
    assert all(r[0] >= split_day for r in test)


def test_retraining_refuses_tiny_validation_sample():
    f = functions('make_time_purged_train_test', timedelta=timedelta, datetime=datetime)['make_time_purged_train_test']
    d = date(2026,1,1)
    few = [(str(d+timedelta(days=i)),[1.0],i%2) for i in range(25)]
    assert f(few) is None


def test_market_data_lookback_expands_for_long_shutdown():
    f = functions('history_period_for_open_trade')['history_period_for_open_trade']
    today = date(2026, 11, 20)
    assert f(today-timedelta(days=5),today) == '1mo'
    assert f(today-timedelta(days=50),today) == '6mo'
    assert f(today-timedelta(days=300),today) == '2y'
    assert f(today-timedelta(days=900),today) == 'max'


def test_missing_historical_coverage_refuses_to_invent_trade_exit():
    f = functions('sufficient_audit_history', _to_date=lambda s: datetime.strptime(s[:10],'%Y-%m-%d').date())['sufficient_audit_history']
    assert not f(date(2026,1,1), '2026-01-10 12:00:00', date(2026, 10, 1))
    assert f(date(2026,1,1), '2026-11-19 12:00:00', date(2026,10,1))


def test_entry_day_is_not_rechecked_after_later_stop_ratchets():
    f = functions('entry_day_close_needs_replay', _to_date=lambda s: datetime.strptime(s[:10], '%Y-%m-%d').date() if s else None)['entry_day_close_needs_replay']
    entry = date(2026,11,16)
    assert f(entry, '2026-11-16 10:00:00', date(2026,11,17))
    # Later audits may have raised the stop; cannot apply that raised stop to entry-day prices.
    assert not f(entry, '2026-11-18 13:00:00', date(2026,11,20))
    assert not f(entry, '2026-11-19 17:00:00', date(2026,11,20))
