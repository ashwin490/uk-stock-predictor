import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from auto_refresh_schedule import choose_auto_refresh


def policy(**kw):
    return choose_auto_refresh(**kw)


def test_open_during_trading_window_defaults_on_even_if_old_toggle_false():
    assert policy(market_open=True, in_entry_window=True,
                  previous_market_open=None, previous_entry_window=None,
                  current_setting=False, session_initialized=False)


def test_open_before_entry_window_defaults_on_for_monitoring():
    assert policy(market_open=True, in_entry_window=False,
                  previous_market_open=None, previous_entry_window=None,
                  current_setting=False, session_initialized=False)


def test_auto_starts_on_market_open_or_entry_window_boundary():
    assert policy(market_open=True, in_entry_window=False,
                  previous_market_open=False, previous_entry_window=False,
                  current_setting=False, manually_paused=True, session_initialized=True)
    assert policy(market_open=True, in_entry_window=True,
                  previous_market_open=True, previous_entry_window=False,
                  current_setting=False, manually_paused=True, session_initialized=True)


def test_manual_pause_stays_off_on_regular_reruns_within_session():
    assert not policy(market_open=True, in_entry_window=True,
                      previous_market_open=True, previous_entry_window=True,
                      current_setting=False, manually_paused=True, session_initialized=True)


def test_toggle_back_on_resumes_scanning():
    assert policy(market_open=True, in_entry_window=True,
                  previous_market_open=True, previous_entry_window=True,
                  current_setting=True, manually_paused=False, session_initialized=True)


def test_market_closed_is_always_off():
    assert not policy(market_open=False, in_entry_window=False,
                      previous_market_open=True, previous_entry_window=True,
                      current_setting=True, session_initialized=True)


def test_no_false_carry_over_when_not_paused():
    assert policy(market_open=True, in_entry_window=True,
                  previous_market_open=True, previous_entry_window=True,
                  current_setting=False, manually_paused=False, session_initialized=True)
