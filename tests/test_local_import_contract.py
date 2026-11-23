"""Check that local imports in app.py exist in the files deployed with it."""
import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HELPERS = {'alpha_backup', 'intraday_exit', 'commodity_etps', 'commodity_signals',
           'commodity_dashboard', 'auto_refresh_schedule', 'option_retirement'}


def test_every_local_import_declared_in_app_exists():
    parsed = ast.parse((ROOT / 'app.py').read_text(encoding='utf-8'))
    for node in parsed.body:
        if isinstance(node, ast.ImportFrom) and node.module in HELPERS:
            file = ROOT / f'{node.module}.py'
            assert file.is_file(), f'Missing {file.name}'
            module_ast = ast.parse(file.read_text(encoding='utf-8'))
            symbols = set()
            for entry in module_ast.body:
                if isinstance(entry, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    symbols.add(entry.name)
                elif isinstance(entry, (ast.Import, ast.ImportFrom)):
                    symbols.update(x.asname or x.name.split('.')[0] for x in entry.names)
                elif isinstance(entry, (ast.Assign, ast.AnnAssign)):
                    targets = entry.targets if isinstance(entry, ast.Assign) else [entry.target]
                    symbols.update(x.id for x in targets if isinstance(x, ast.Name))
            for alias in node.names:
                assert alias.name in symbols, f'{node.module}.py missing {alias.name} – deploy matching helper with app.py'


def test_commodity_dashboard_imports_and_card_render():
    from commodity_dashboard import (commodity_signal_card_html, commodity_journal_view,
        summarise_commodity_history, assess_commodity_history, decorate_commodity_card)
    import pandas as pd
    item = decorate_commodity_card({'Commodity':'Gold', 'Ticker':'SGLN.L',
        'Decision':'WAIT', 'Buy (p)':5000.0}, pd.DataFrame())
    assert item['Display mode'] == 'WAIT'
    assert 'Latest available price' in commodity_signal_card_html(item)
