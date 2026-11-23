"""Pure, chart-free commodity journal views and cautious paper-risk feedback.

No Streamlit state, network or database operations occur here. The journal is
provided by the app, so it works equally with cloud-hydrated or local history.
Neither feedback nor the displayed trend checks are validated probabilities.
"""
from __future__ import annotations

from dataclasses import dataclass
import html
import json
import math

import pandas as pd


MIN_FEEDBACK_TRADES = 20
FEEDBACK_RECENT_TRADES = 10


def _number(value, default=0.0):
    try:
        result = float(value)
        return result if math.isfinite(result) else default
    except (ValueError, TypeError):
        return default


def _kind(raw):
    val = str(raw or "").upper()
    if "BREAK-EVEN" in val:
        return "SAVED"
    if "WIN" in val:
        return "WON"
    if "LOSS" in val:
        return "LOST"
    if "EXPIRED" in val:
        return "EXPIRED"
    if val == "ACTIVE" or "ACTIVE" in val:
        return "OPEN"
    return "OTHER"


def summarise_commodity_history(df: pd.DataFrame) -> dict:
    """Totals from all journal rows; no market data or assumptions about fills."""
    out = dict(open=0, closed=0, won=0, lost=0, saved=0, expired=0,
               realised_gbp=0.0, unrealised_gbp=0.0, decided=0, win_rate=None)
    if df is None or df.empty:
        return out
    for item in df.to_dict('records'):
        typ = _kind(item.get('status'))
        amount = _number(item.get('capital_allocated')) * _number(item.get('pnl_pct')) / 100.0
        if typ == 'OPEN':
            out['open'] += 1
            out['unrealised_gbp'] += amount
        elif typ in ('WON', 'LOST', 'SAVED', 'EXPIRED'):
            out['closed'] += 1
            out['realised_gbp'] += amount
            out[{'WON':'won','LOST':'lost','SAVED':'saved','EXPIRED':'expired'}[typ]] += 1
    out['decided'] = out['won'] + out['lost']
    if out['decided']:
        out['win_rate'] = 100 * out['won'] / out['decided']
    return out


def commodity_journal_view(df: pd.DataFrame) -> pd.DataFrame:
    """User-friendly full history including old and new trades; never overwrite rows."""
    columns = ['Bought on', 'Commodity', 'Symbol', 'Quantity', 'Buy (p)',
               'Invested (£)', 'Sell target (p)', 'Stop-loss (p)',
               'Latest / exit (p)', 'Sold on', 'Paper P/L (£)', 'Paper P/L (%)', 'Result']
    if df is None or df.empty:
        return pd.DataFrame(columns=columns)
    rows = []
    for rec in df.to_dict('records'):
        try:
            meta = rec.get('features_json') or '{}'
            meta = meta if isinstance(meta, dict) else json.loads(meta)
            if not isinstance(meta, dict):
                meta = {}
        except (ValueError, TypeError):
            meta = {}
        ticker = str(rec.get('ticker') or '')
        kind = _kind(rec.get('status'))
        spent = _number(rec.get('capital_allocated'))
        pct = _number(rec.get('pnl_pct'))
        price = _number(rec.get('exit_price')) if kind != 'OPEN' else 0.0
        if price <= 0:
            price = _number(rec.get('latest_price'))
        label = rec.get('commodity_name') or ticker.replace('.L', '')
        bought = meta.get('entry_timestamp_london') or rec.get('timestamp') or rec.get('date_str') or 'Unknown'
        sold = rec.get('exit_timestamp') if kind != 'OPEN' else ''
        rows.append({'Bought on': str(bought), 'Commodity': str(label), 'Symbol': ticker,
                     'Quantity': int(_number(rec.get('shares'))), 'Buy (p)': _number(rec.get('entry_price')),
                     'Invested (£)': round(spent,2), 'Sell target (p)': _number(rec.get('target_price')),
                     'Stop-loss (p)': _number(rec.get('stop_loss')), 'Latest / exit (p)': price,
                     'Sold on': str(sold or ''), 'Paper P/L (£)': round(spent * pct/100.0,2),
                     'Paper P/L (%)': round(pct,2),
                     'Result': {'WON':'Won — target','LOST':'Lost — stop',
                                'SAVED':'Protected exit','EXPIRED':'Expired',
                                'OPEN':'Open','OTHER':'Review needed'}[kind]})
    return pd.DataFrame(rows, columns=columns).sort_values('Bought on', ascending=False, kind='stable').reset_index(drop=True)


@dataclass(frozen=True)
class CommodityFeedback:
    count: int
    stake_fraction: float
    note: str


def assess_commodity_history(df: pd.DataFrame, ticker: str) -> CommodityFeedback:
    """Only *reduce*, never increase exposure after consistent weak realised results.

    More evidence is required than the 5-trend-condition signal. This is a
    transparent conservative risk response, not a self-training forecast.
    """
    if df is None or df.empty or 'ticker' not in df or 'status' not in df:
        return CommodityFeedback(0, 1.0, 'No finished paper trades yet; rules are unchanged')
    rows = df[df['ticker'].astype(str) == str(ticker)].copy()
    rows = rows[rows['status'].map(lambda s: _kind(s) in ('WON','LOST','SAVED','EXPIRED'))]
    if rows.empty:
        return CommodityFeedback(0, 1.0, 'No finished paper trades yet; rules are unchanged')
    rows['_finish'] = rows['exit_timestamp'].fillna('') if 'exit_timestamp' in rows else ''
    if 'last_audited' in rows:
        rows['_finish'] = rows['_finish'].where(rows['_finish'].astype(str).str.len() > 0, rows['last_audited'].fillna(''))
    rows = rows.sort_values('_finish', kind='stable')
    n = len(rows)
    if n < MIN_FEEDBACK_TRADES:
        return CommodityFeedback(n, 1.0, f'{n}/{MIN_FEEDBACK_TRADES} completed trades; waiting for sufficient evidence')
    latest = rows.tail(FEEDBACK_RECENT_TRADES)['pnl_pct'].apply(_number)
    avg = float(latest.mean())
    profitable = int((latest > 0).sum())
    if avg <= -1.0 and profitable <= 3:
        return CommodityFeedback(n, 0.5, f'Recent {FEEDBACK_RECENT_TRADES} trades averaged {avg:+.1f}% ({profitable} profitable); future paper stake halved')
    return CommodityFeedback(n, 1.0, f'{n} completed trades; normal paper risk limits retained (recent average {avg:+.1f}%)')


def commodity_signal_card_html(item: dict) -> str:
    """Escaped accessible text cards; display only, no buttons or network calls."""
    e = lambda val: html.escape(str(val), quote=True)
    def fmt_price(v):
        x = _number(v, float('nan'))
        return f'{x:,.4f}p' if math.isfinite(x) and 0 < abs(x) < 1 else (f'{x:,.2f}p' if math.isfinite(x) else '—')
    decision = str(item.get('Decision') or 'WAIT')
    tone = 'ok' if 'BOUGHT' in decision or 'CANDIDATE' in decision else 'hold'
    explanation = str(item.get('Explanation') or 'No explanation available')
    return f'''<div class="commodity-card" role="group" aria-label="{e(item.get('Commodity','Commodity'))}">
      <div class="cm-top"><div><div class="cm-name">{e(item.get('Commodity','Commodity'))}</div><div class="cm-symbol">{e(item.get('Ticker',''))} · sterling-listed ETC</div></div><div class="cm-status {tone}">{e(decision)}</div></div>
      <div class="cm-price-grid"><div><small>Buy at</small><b>{e(fmt_price(item.get('Buy (p)')))}</b></div>
      <div><small>Sell target</small><b>{e(fmt_price(item.get('Sell target (p)')))}</b></div>
      <div><small>Stop-loss</small><b>{e(fmt_price(item.get('Stop-loss (p)')))}</b></div></div>
      <div class="cm-details"><span>Trend: <b>{e(item.get('Trend checks','—'))}</b></span><span>Quantity: <b>{int(_number(item.get('Units'))):,}</b></span><span>Paper spend: <b>£{_number(item.get('Paper amount (£)')):,.2f}</b></span></div>
      <div class="cm-why"><b>Why:</b> {e(explanation)}</div>
      <div class="cm-learning">History: {e(item.get('History note') or 'No completed commodity trades yet')}</div>
      </div>'''
