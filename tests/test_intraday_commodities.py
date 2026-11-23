"""New Nov-23 checks: post-entry minute exits, ETC safety, options evidence."""
import ast
from datetime import date, datetime
from pathlib import Path
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT))
from intraday_exit import first_intraday_exit
from commodity_etps import COMMODITY_ETPS, is_commodity_etc, FUTURES_REFERENCES


def minute_bars(rows):
    return pd.DataFrame(rows, index=pd.DatetimeIndex([
        '2026-11-23 10:52:00', '2026-11-23 10:53:00',
        '2026-11-23 10:54:00', '2026-11-23 10:55:00',
        '2026-11-23 10:56:00', '2026-11-23 10:57:00'
    ], tz='Europe/London'))


def test_5_minute_target_hit_even_if_later_price_drops_below_target():
    h=minute_bars(dict(Open=[100,101,101,102,103,101],High=[125,102,103,111,104,102],
                       Low=[99,100,100,101,100,99],Close=[101,101,102,103,101,100]))
    r=first_intraday_exit(h,'2026-11-23 10:52:20',target=110,stop=95,through_day=date(2026,11,23))
    assert r is not None and r.outcome=='WIN'
    assert r.fill_price==110
    assert r.observed_at=='2026-11-23 10:55:00'


def test_pre_entry_high_does_not_fake_win():
    h=minute_bars(dict(Open=[100]*6,High=[160,105,105,105,105,105],
                       Low=[99]*6,Close=[100]*6))
    assert first_intraday_exit(h,'2026-11-23 10:52:20',target=150,stop=95) is None


def test_same_minute_target_and_stop_is_pessimistic():
    h=minute_bars(dict(Open=[100]*6,High=[100,120,105,105,105,105],
                       Low=[99,90,98,98,98,98],Close=[100,100,100,100,100,100]))
    r=first_intraday_exit(h,'2026-11-23 10:52:01',target=110,stop=95)
    assert r.outcome=='STOP' and r.ambiguous and r.fill_price==95


def test_gap_down_below_stop_uses_worse_open_price():
    h=minute_bars(dict(Open=[100,90,100,100,100,100],High=[101,93,100,100,100,100],
                       Low=[99,88,98,98,98,98],Close=[100,89,100,100,100,100]))
    r=first_intraday_exit(h,'2026-11-23 10:52:01',110,95)
    assert r.outcome=='STOP' and r.fill_price==90


def test_naive_provider_minute_timestamps_cannot_be_guessed():
    h=minute_bars(dict(Open=[100]*6,High=[100,112,100,100,100,100],
                       Low=[99]*6,Close=[100]*6))
    h.index = h.index.tz_localize(None)
    assert first_intraday_exit(h,'2026-11-23 10:52:20',110,95) is None


def test_commodity_tickers_are_sterling_etcs_and_not_futures():
    for code in ('SGLN.L','SSLN.L','BRNG.L','CRUP.L','NGSP.L'):
        assert is_commodity_etc(code)
        assert COMMODITY_ETPS[code]['sector'] in {'Precious Metals','Oil & Gas ETCs'}
    for ticker in FUTURES_REFERENCES:
        assert not is_commodity_etc(ticker)


def test_new_automatic_learning_fallback_is_removed():
    s=(ROOT/'app.py').read_text(encoding='utf-8')
    assert 'LEARNING_PICKS_PER_DAY' not in s
    tree=ast.parse(s)
    fn=next(x for x in tree.body if isinstance(x, ast.FunctionDef) and x.name=='run_predictions')
    assert 'learning_pick=True' not in ast.unparse(fn)
    assert 'if is_commodity_etc(ticker):' in ast.unparse(fn)


def test_old_cloud_rows_not_given_fabricated_purchase_timestamp():
    s=(ROOT/'app.py').read_text(encoding='utf-8')
    assert 'original_entry_ts = str(f_meta.get("entry_timestamp_london") or "")' in s
    assert '"entry_timestamp_london=now_str"' not in s  # timestamp stored as actual metadata key below
    assert 'entry_timestamp_london=now_str' in s


def test_no_auto_new_calls_as_default():
    s=(ROOT/'app.py').read_text(encoding='utf-8')
    assert 'key="allow_new_call_ideas"' in s
    assert 'if not st.session_state.get("allow_new_call_ideas", False):' in s
    assert 'recent = real_d[' in s


def test_paper_etc_requires_explicit_authorisation():
    # Inject the pure log function from the existing test harness; no UI/network.
    from test_local_first import trade_env,signal
    f, db, state=trade_env()
    f.__globals__['is_commodity_etc']=lambda t: t=='SGLN.L'
    order=signal(symbol='SGLN.L',capital=1000)
    order['Qualified']=False
    order['ManualETC']=True
    assert f(order) is False
    assert f(order, manual_etc_authorised=True) is True
    assert db.conn.execute("select asset_type from trade_journal").fetchone()[0]=='ETC'


def test_main_option_summary_does_not_count_old_synthetic_gains():
    source=(ROOT/'app.py').read_text(encoding='utf-8')
    assert 'if perf_real_opt["n"]:' in source
    assert "perf_real_opt['net_gbp']" in source
    assert "if perf_opt[\"n\"]:" not in source
    assert 'Older practice-price ideas' in source


def test_entry_day_fallback_uses_original_stop_not_later_raised_stop():
    source=(ROOT/'app.py').read_text(encoding='utf-8')
    assert 'entry_close_stop = entry_stop if (original_ts and 0 < entry_stop < entry) else None' in source
    assert 'classify_stop_exit(entry, entry_close_stop, c, friction)' in source
