"""Fast, offline tests of the new commodity cards, history and risk feedback."""
import sys
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from commodity_dashboard import (commodity_signal_card_html, commodity_journal_view,
                                 summarise_commodity_history, assess_commodity_history)


def make_row(ticker, status, pct, when='2026-11-20 10:35:00', amount=100):
    return dict(ticker=ticker, status=status, pnl_pct=pct, date_str='2026-11-20',
                timestamp=when, exit_timestamp=when, last_audited=when, features_json='{}',
                capital_allocated=amount, entry_price=150, target_price=170,
                stop_loss=140, shares=66, latest_price=170, exit_price=170)


def test_totals_are_split_realised_unrealised_and_never_include_shares():
    df=pd.DataFrame([make_row('SGLN.L','ACTIVE',-10),make_row('SGLN.L','🎯 WIN',10),
                     make_row('SSLN.L','LOSS',-5),make_row('SSLN.L','EXPIRED',1)])
    s=summarise_commodity_history(df)
    assert (s['open'],s['closed'],s['won'],s['lost'],s['expired'])==(1,3,1,1,1)
    assert s['realised_gbp']==6
    assert s['unrealised_gbp']==-10
    assert s['win_rate']==50


def test_history_contains_original_purchase_and_exit_columns():
    df=pd.DataFrame([make_row('SGLN.L','ACTIVE',-2)])
    out=commodity_journal_view(df)
    assert out.iloc[0]['Bought on']=='2026-11-20 10:35:00'
    assert out.iloc[0]['Result']=='Open'
    assert out.iloc[0]['Quantity']==66
    assert out.iloc[0]['Paper P/L (£)']==-2
    assert out.iloc[0]['Sold on']==''


def test_cards_show_price_levels_and_escape_untrusted_text():
    out=commodity_signal_card_html(dict(Commodity='<script>hello</script>', Ticker='SGLN.L',
             Decision='WAIT',Explanation='<b>bad</b>',**{'Buy (p)':5447.9,'Sell target (p)':5721.11,
                                                        'Stop-loss (p)':5277.15,'Trend checks':'0/5',
                                                        'Units':0,'Paper amount (£)':0.0}))
    assert '5,447.90p' in out and '5,721.11p' in out and '5,277.15p' in out
    assert '<script>' not in out and '&lt;script&gt;' in out
    assert '<b>bad</b>' not in out


def test_feedback_waits_for_20_trades_and_only_reduces_risk():
    losing = pd.DataFrame([make_row('SGLN.L','LOSS',-3,when=f'2026-11-{i+1:02d} 11:00:00') for i in range(20)])
    s=assess_commodity_history(losing, 'SGLN.L')
    assert s.count==20 and s.stake_fraction==0.5
    assert assess_commodity_history(losing.head(10),'SGLN.L').stake_fraction==1
    winners=losing.copy(); winners['pnl_pct']=2
    assert assess_commodity_history(winners,'SGLN.L').stake_fraction==1
    assert assess_commodity_history(pd.DataFrame(),'SGLN.L').stake_fraction==1
