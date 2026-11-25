"""GitHub cron can run late; catch up strictly using completed XLON bars."""
from datetime import date, datetime, timezone

import numpy as np
import pandas as pd
import pytest

import commodity_worker as cw
import share_worker as sw
from commodity_ml import PRODUCTS
from test_autonomous_worker import FakeBucket
from test_share_autonomous import bars as share_bars


def test_last_verified_session_handles_overnight_and_market_close():
    assert cw.latest_verified_closed_session(datetime(2026, 11, 25, 3, 46, tzinfo=timezone.utc)) == date(2026, 11, 24)
    assert cw.latest_verified_closed_session(datetime(2026, 11, 25, 16, 59, tzinfo=timezone.utc)) == date(2026, 11, 24)
    assert cw.latest_verified_closed_session(datetime(2026, 11, 25, 17, 7, tzinfo=timezone.utc)) == date(2026, 11, 25)
    assert cw.latest_verified_closed_session(datetime(2026, 11, 29, 10, tzinfo=timezone.utc)) == date(2026, 11, 27)
    assert cw.latest_verified_closed_session(datetime(2026, 12, 25, 10, tzinfo=timezone.utc)) == date(2026, 12, 24)


def test_completed_history_excludes_unfinished_next_day_bars():
    b = pd.DataFrame({'Close': [100, 102, 10000]}, index=pd.to_datetime(['2026-11-23', '2026-11-24', '2026-11-25']))
    got = cw.completed_history_through(b, date(2026, 11, 24))
    assert got.index.max() == pd.Timestamp('2026-11-24')
    assert got['Close'].max() == 102


def test_delayed_share_worker_produces_prev_day_forecasts(monkeypatch):
    symbols = [f'S{i:02d}.L' for i in range(36)]
    saved = {t: share_bars(seed=i+10) for i, t in enumerate(symbols)}
    # Inject an unfinished next-day bar: it must not enter features or labels.
    for t, b in saved.items():
        previous = b.iloc[-1:].copy()
        previous.index = pd.DatetimeIndex([pd.Timestamp('2026-11-25')])
        previous['Close'] = 1_000_000.0
        previous['High'] = 1_000_001.0
        saved[t] = pd.concat([b,previous])
    monkeypatch.setattr(sw, 'available_universe', lambda: symbols)
    bkt = FakeBucket()
    run = datetime(2026, 11, 25, 3, 46, tzinfo=timezone.utc)
    result = sw.daily(bkt, run, fetcher=lambda t, p: (saved, {}))
    assert result['target_session'] == '2026-11-24'
    assert result['saved'] == len(symbols)
    stored = sw.read_json(bkt, sw.FORECASTS)
    assert stored['as_of'] == '2026-11-24'
    assert all(row['date'] == '2026-11-24' for row in stored['forecasts'])
    rerun = sw.daily(bkt, run, fetcher=lambda t, p: (saved, {}))
    assert rerun['saved'] == 0


def test_delayed_commodity_worker_produces_prev_day_forecasts(monkeypatch):
    rng = np.random.default_rng(77)
    ix = pd.bdate_range(end='2026-11-24', periods=290)
    histories = {}
    for ticker in PRODUCTS:
        close = 300 * np.exp(np.cumsum(rng.normal(.0001,.025,len(ix))))
        b = pd.DataFrame({'Open':close,'High':close*1.025,'Low':close*.975,
                          'Close':close,'Volume':np.full(len(ix),200_000)},index=ix)
        next_bar = b.iloc[-1:].copy()
        next_bar.index = pd.DatetimeIndex([pd.Timestamp('2026-11-25')])
        next_bar['Close'] = next_bar['High'] * 50
        histories[ticker] = pd.concat([b,next_bar])
        histories[ticker].attrs['price_currency'] = 'GBp'
    monkeypatch.setattr(cw,'fetch_histories',lambda period: (histories,{}))
    bucket = FakeBucket()
    run = datetime(2026,11,25,3,31,tzinfo=timezone.utc)
    result = cw.daily_worker(None,bucket,run)
    assert result['target_session'] == '2026-11-24'
    assert result['forecasts_saved'] == len(PRODUCTS)
    current = cw.read_json(bucket,cw.LATEST_KEY)
    assert current['as_of'] == '2026-11-24'
    assert all(x['date'] == '2026-11-24' for x in current['forecasts'])
    repeat = cw.daily_worker(None,bucket,run)
    assert repeat['forecasts_saved'] == 0


def test_missing_target_bars_raises_instead_of_green_success(monkeypatch):
    monkeypatch.setattr(cw,'fetch_histories',lambda period: ({}, {}))
    with pytest.raises(RuntimeError,match='three'):
        cw.daily_worker(None,FakeBucket(),datetime(2026,11,25,3,31,tzinfo=timezone.utc))
