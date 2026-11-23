"""Check the full Streamlit app includes the new UX without importing live dependencies."""
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def test_home_and_commodity_summary_use_same_commodity_history():
    src=(ROOT/'app.py').read_text(encoding='utf-8')
    assert 'Commodities at a glance' in src
    assert 'commodity_signal_card_html(it)' in src
    assert "_etc_summary = summarise_commodity_history(etc_df)" in src
    assert "_comm_home = summarise_commodity_history(etc_df)" in src
    assert "commodity_journal_view(etc_df)" in src
    assert "file_name='alpha_lse_commodity_trades.csv'" in src
    assert 'share_df = df_eq[' in src and "etc_df = df_eq[df_eq['asset_type'] == 'ETC']" in src


def test_auto_refresh_has_heartbeat_during_off_hours():
    src=(ROOT/'app.py').read_text(encoding='utf-8')
    assert 'choose_auto_refresh(' in src
    assert 'wake_interval_sec = refresh_interval_sec if auto_mode else 300' in src
    assert 'if st_autorefresh:' in src
    assert 'bool(auto_mode) and loop_tick != st.session_state.get("last_loop_tick")' in src


def test_new_modules_are_included_with_main_app():
    for f in ['app.py','commodity_dashboard.py','auto_refresh_schedule.py','commodity_signals.py']:
        assert (ROOT/f).is_file()
