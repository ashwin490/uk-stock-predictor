"""A simulated purchase must not be rewritten by a refreshed market signal."""
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from commodity_dashboard import decorate_commodity_card, commodity_signal_card_html


def _signal():
    return {'Commodity': 'Gold', 'Ticker': 'SGLN.L', 'Decision': 'WAIT — trend',
            'Current (p)': 5490.0, 'Quote date': '2026-11-23',
            'Buy (p)': 5490.0, 'Sell target (p)': 5680.0,
            'Stop-loss (p)': 5330.0, 'Units': 0, 'Paper amount (£)': 0.0,
            'Explanation': 'Insufficient trend strength'}


def test_wait_is_provisional_and_quote_is_separate():
    view = decorate_commodity_card(_signal(), pd.DataFrame())
    assert view['Display mode'] == 'WAIT'
    rendered = commodity_signal_card_html(view)
    assert 'Latest available price' in rendered
    assert 'Buy at (provisional)' in rendered
    assert 'Sell target (indicative)' in rendered
    assert '5,490.00p' in rendered


def test_open_trade_uses_journal_levels_not_changing_signal():
    journal = pd.DataFrame([{
        'ticker':'SGLN.L','status':'ACTIVE','entry_price':5400.,'target_price':5600.,
        'stop_loss':5300.,'shares':3,'capital_allocated':162.,'latest_price':5480.,
        'pnl_pct':1.4,'timestamp':'2026-11-23 10:00','last_audited':'2026-11-23 12:00'
    }])
    view=decorate_commodity_card(_signal(),journal)
    assert view['Display mode'] == 'OPEN'
    assert view['Current (p)'] == 5490.  # new quote still displayed
    assert view['Buy (p)'] == 5400.    # immutable saved entry
    assert view['Sell target (p)'] == 5600.
    assert view['Stop-loss (p)'] == 5300.
    assert view['Paper P/L (£)'] == 2.27
    rendered=commodity_signal_card_html(view)
    assert 'Buy at (recorded)' in rendered
    assert 'Sell target (locked)' in rendered
    assert 'Latest available price' in rendered


def test_closed_trade_does_not_change_a_new_wait_signal():
    journal=pd.DataFrame([{
        'ticker':'SGLN.L','status':'🎯 WIN (TARGET HIT)','entry_price':5400.,
        'exit_price':5600.,'pnl_pct':3.2,'exit_timestamp':'2026-11-22 14:00'
    }])
    view=decorate_commodity_card(_signal(),journal)
    assert view['Display mode'] == 'WAIT'
    assert view['Buy (p)'] == 5490.
    assert 'Won: bought 5,400.00p' in view['Last finished trade']
