"""Offline regression guards for commodity card rendering and entry-cutoff copy."""
import ast
from pathlib import Path
import pandas as pd
from commodity_dashboard import commodity_signal_card_html, decorate_commodity_card

APP = Path(__file__).resolve().parents[1] / "app.py"


def test_streamlit_html_not_markdown_for_commodity_cards():
    source = APP.read_text(encoding="utf-8")
    tree = ast.parse(source)
    html_calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
                  and isinstance(n.func, ast.Attribute)
                  and isinstance(n.func.value, ast.Name)
                  and n.func.value.id == "st" and n.func.attr == "html"]
    assert any(n.args and isinstance(n.args[0], ast.Call)
               and isinstance(n.args[0].func, ast.Name)
               and n.args[0].func.id == "commodity_signal_card_html"
               for n in html_calls)
    assert "st.markdown(commodity_signal_card_html(it)" not in source


def test_explicit_entry_cutoff_and_monitoring_message():
    source = APP.read_text(encoding="utf-8")
    assert "new paper purchases stop at " in source
    assert "Existing trades continue to be monitored" in source
    assert "UK market is open for monitoring" in source
    assert "entry_window_end().strftime" in source


def test_price_warning_includes_ticker_but_never_converts_usd_eur():
    source = APP.read_text(encoding="utf-8")
    assert 'ticker: str = "unknown"' in source
    assert 'unexpected currency %s", ticker, currency' in source
    assert 'invalid LSE price bars for %s", rejected, ticker' in source
    assert 'return normalize_lse_history_prices(df, cur, ticker=ticker)' in source
    assert 'return normalize_lse_history_prices(bars, currency, ticker=ticker)' in source


def test_card_html_is_escaped_for_user_data():
    obj=decorate_commodity_card({
        "Commodity": "<script>alert(1)</script>", "Ticker": "SGLN.L",
        "Decision": "WAIT", "Explanation": "<img src=x onerror=alert(1)>",
        "Current (p)": 5490.0,"Buy (p)": 5500.0,
        "Sell target (p)": 5800.0, "Stop-loss (p)": 5300.0,
    },pd.DataFrame())
    output=commodity_signal_card_html(obj)
    assert "<script>" not in output and "&lt;script&gt;" in output
    assert "<img " not in output and "&lt;img" in output
    assert "Latest available price" in output
    assert "Buy at (provisional)" in output
