import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from auto_refresh_schedule import choose_auto_refresh


def policy(**kw):
    return choose_auto_refresh(**kw)


def test_auto_starts_when_market_opens_and_when_entry_begins():
    assert policy(market_open=True,in_entry_window=False,previous_market_open=False,previous_entry_window=False,current_setting=False)
    assert policy(market_open=True,in_entry_window=True,previous_market_open=True,previous_entry_window=False,current_setting=False)
    assert policy(market_open=True,in_entry_window=True,previous_market_open=None,previous_entry_window=None,current_setting=None)


def test_manually_paused_session_stays_paused_until_next_session():
    assert not policy(market_open=True,in_entry_window=True,previous_market_open=True,previous_entry_window=True,current_setting=False)
    assert not policy(market_open=False,in_entry_window=False,previous_market_open=True,previous_entry_window=True,current_setting=True)


def test_normal_open_session_does_not_override_manual_selection():
    assert policy(market_open=True,in_entry_window=True,previous_market_open=True,previous_entry_window=True,current_setting=True)
