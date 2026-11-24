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
               'Latest / exit (p)', 'Sold on', 'Paper P/L (£)', 'Paper P/L (%)',
               'Strategy', 'ML model version', 'ML 5d event estimate (%)', 'Result']
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
        strategy_code = str(meta.get('strategy_id') or meta.get('model_source') or '')
        strategy = ('ML + trend rules' if strategy_code == 'COMMODITY_ETC_ML_V1'
                    else 'Trend rules' if strategy_code == 'COMMODITY_ETC_TREND_V1'
                    else 'Manual ETC' if 'MANUAL_ETC' in strategy_code
                    else 'Legacy (unknown)' if not strategy_code else strategy_code)
        ml_version = meta.get('commodity_ml_version')
        ml_estimate = _number(meta.get('commodity_ml_probability'),float('nan'))
        rows.append({'Bought on': str(bought), 'Commodity': str(label), 'Symbol': ticker,
                     'Quantity': int(_number(rec.get('shares'))), 'Buy (p)': _number(rec.get('entry_price')),
                     'Invested (£)': round(spent,2), 'Sell target (p)': _number(rec.get('target_price')),
                     'Stop-loss (p)': _number(rec.get('stop_loss')), 'Latest / exit (p)': price,
                     'Sold on': str(sold or ''), 'Paper P/L (£)': round(spent * pct/100.0,2),
                     'Paper P/L (%)': round(pct,2), 'Strategy': strategy,
                     'ML model version': ml_version if ml_version is not None else '—',
                     'ML 5d event estimate (%)': round(100*ml_estimate,1) if math.isfinite(ml_estimate) else None,
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


def decorate_commodity_card(signal: dict, journal: pd.DataFrame) -> dict:
    """Combine a fresh signal with *read-only* persisted trade details.

    The live/last available quote always stays separate from historical buys.
    In particular, WAIT signals are never treated as executed orders. A saved
    open paper trade overrides recalculated levels, even if a current quote is
    unavailable. Closed trades remain unchanged in the journal/history table.
    """
    item = dict(signal)
    item.setdefault('Current (p)', item.get('Buy (p)'))
    item['Display mode'] = 'WAIT'
    item['Quote source'] = 'Latest available LSE market-data daily bar; may be delayed'
    decision = str(item.get('Decision') or '').upper()
    if 'BUY CANDIDATE' in decision or 'PAPER BOUGHT' in decision:
        item['Display mode'] = 'CANDIDATE'
    if journal is None or journal.empty or not {'ticker','status'}.issubset(journal.columns):
        return item
    matches = journal[journal['ticker'].astype(str) == str(item.get('Ticker') or '')]
    active = matches[matches['status'].map(lambda v: _kind(v) == 'OPEN')]
    if not active.empty:
        # Prefer the latest open entry; under normal safeguards there is one.
        if 'timestamp' in active.columns:
            active = active.sort_values('timestamp', ascending=False, kind='stable')
        rec = active.iloc[0].to_dict()
        item['Display mode'] = 'OPEN'
        item['Decision'] = 'PAPER OPEN — monitoring'
        item['Buy (p)'] = _number(rec.get('entry_price'), float('nan'))
        item['Sell target (p)'] = _number(rec.get('target_price'), float('nan'))
        item['Stop-loss (p)'] = _number(rec.get('stop_loss'), float('nan'))
        item['Units'] = int(_number(rec.get('shares')))
        item['Paper amount (£)'] = _number(rec.get('capital_allocated'))
        item['Recorded price (p)'] = _number(rec.get('latest_price'), float('nan'))
        item['Paper P/L (%)'] = _number(rec.get('pnl_pct'))
        item['Paper P/L (£)'] = round(item['Paper amount (£)'] * item['Paper P/L (%)']/100.0, 2)
        item['Paper last audited'] = str(rec.get('last_audited') or 'Unknown')
        item['Paper bought on'] = str(rec.get('timestamp') or rec.get('date_str') or 'Unknown')
        item['Explanation'] = 'Simulated purchase already recorded. Original entry and target stay fixed; active stop only changes through the configured protection rule.'
        return item
    # Historical closes do not freeze a new WAIT signal: instead preserve the
    # last deal summary separately, while all original levels remain in history.
    closed = matches[matches['status'].map(lambda v: _kind(v) in ('WON','LOST','SAVED','EXPIRED'))]
    if not closed.empty:
        if 'exit_timestamp' in closed.columns:
            closed = closed.sort_values('exit_timestamp', ascending=False, kind='stable')
        rec = closed.iloc[0].to_dict()
        label = {'WON':'Won','LOST':'Lost','SAVED':'Protected','EXPIRED':'Expired'}[_kind(rec.get('status'))]
        item['Last finished trade'] = (f"{label}: bought {_number(rec.get('entry_price')):,.2f}p, "
                                      f"exited {_number(rec.get('exit_price')):,.2f}p, "
                                      f"paper P/L {_number(rec.get('pnl_pct')):+.2f}%")
    return item


def commodity_signal_card_html(item: dict) -> str:
    """Accessible, escaped display: latest quote ≠ suggested entry ≠ actual purchase."""
    e = lambda val: html.escape(str(val), quote=True)
    def fmt_price(v):
        x = _number(v, float('nan'))
        return (f'{x:,.4f}p' if math.isfinite(x) and 0 < abs(x) < 1
                else f'{x:,.2f}p' if math.isfinite(x) and x > 0 else '—')
    def fmt_pl(v):
        x = _number(v, float('nan'))
        return f'{x:+,.2f}' if math.isfinite(x) else '—'
    decision = str(item.get('Decision') or 'WAIT')
    mode = str(item.get('Display mode') or 'WAIT').upper()
    tone = 'ok' if mode == 'OPEN' or mode == 'CANDIDATE' else 'hold'
    explanation = str(item.get('Explanation') or 'No explanation available')
    if mode == 'OPEN':
        entry_label, target_label, stop_label = 'Buy at (recorded)', 'Sell target (locked)', 'Stop-loss (active)'
    elif mode == 'CANDIDATE':
        entry_label, target_label, stop_label = 'Buy at (suggested)', 'Sell target (proposed)', 'Stop-loss (proposed)'
    else:
        entry_label, target_label, stop_label = 'Buy at (provisional)', 'Sell target (indicative)', 'Stop-loss (indicative)'
    quote = item.get('Current (p)', item.get('Buy (p)'))
    quote_date = str(item.get('Quote date') or 'date unavailable')
    quote_note = f'Latest available market quote · bar dated {quote_date} · may be delayed'
    ml_status = str(item.get('ML status') or 'Historical ML model not yet validated')
    forecast = _number(item.get('ML forecast (%)'), float('nan'))
    threshold = _number(item.get('ML threshold (%)'), float('nan'))
    if math.isfinite(forecast) and math.isfinite(threshold):
        ml_details = (f'<div class="cm-learning"><b>Commodity ML:</b> '
                      f'{e(f"{forecast:.1f}%")} estimated chance of a 5-session ETC gain '
                      f'over 1.5% <b>before costs</b> · entry filter {e(f"{threshold:.1f}%")} · '
                      f'{e(ml_status)}. Not a target-hit / trade-win probability.</div>')
    else:
        research_probability = _number(item.get('ML research forecast (%)'), float('nan'))
        research_threshold = _number(item.get('ML research threshold (%)'), float('nan'))
        research_date = str(item.get('ML research bar date') or 'date unavailable')
        if math.isfinite(research_probability):
            cutoff_text = (f' · experimental threshold {e(f"{research_threshold:.1f}%")}'
                           if math.isfinite(research_threshold) else '')
            ml_details = (f'<div class="cm-learning"><b>EXPERIMENTAL ML — NOT VALIDATED:</b> '
                          f'{e(f"{research_probability:.1f}%")} estimated chance of an ETC gain over '
                          f'1.5% across five sessions <b>before costs</b>{cutoff_text}, based on completed bars '
                          f'through {e(research_date)}. <b>Candidate failed historical validation;</b> '
                          f'this estimate cannot approve a paper buy. Not a trade-win probability.</div>')
        else:
            ml_details = f'<div class="cm-learning"><b>Commodity ML:</b> {e(ml_status)}</div>'
    trade_details = ''
    if mode == 'OPEN':
        trade_details = (f'<div class="cm-why"><b>Recorded paper trade:</b> '
                         f'bought {e(item.get("Paper bought on") or "Unknown")} · '
                         f'last audited {e(item.get("Paper last audited") or "Unknown")} · '
                         f'paper P/L <b>{e(fmt_pl(item.get("Paper P/L (%)")))}%</b> '
                         f'(£{e(fmt_pl(item.get("Paper P/L (£)")))}) · '
                         f'journal mark {e(fmt_price(item.get("Recorded price (p)")))}</div>')
    elif item.get('Last finished trade'):
        trade_details = f'<div class="cm-learning">Last closed trade: {e(item["Last finished trade"])}. Full original details remain in the history table.</div>'
    return f'''<div class="commodity-card" role="group" aria-label="{e(item.get('Commodity','Commodity'))}">
      <div class="cm-top"><div><div class="cm-name">{e(item.get('Commodity','Commodity'))}</div><div class="cm-symbol">{e(item.get('Ticker',''))} · UK-listed ETC</div></div><div class="cm-status {tone}">{e(decision)}</div></div>
      <div class="cm-price-grid">
      <div><small>Latest available price</small><b>{e(fmt_price(quote))}</b></div>
      <div><small>{e(entry_label)}</small><b>{e(fmt_price(item.get('Buy (p)')))}</b></div>
      <div><small>{e(target_label)}</small><b>{e(fmt_price(item.get('Sell target (p)')))}</b></div>
      <div><small>{e(stop_label)}</small><b>{e(fmt_price(item.get('Stop-loss (p)')))}</b></div></div>
      <div class="cm-learning">{e(quote_note)}</div>
      {ml_details}
      <div class="cm-details"><span>Trend: <b>{e(item.get('Trend checks','—'))}</b></span><span>Quantity: <b>{int(_number(item.get('Units'))):,}</b></span><span>Paper spend: <b>£{_number(item.get('Paper amount (£)')):,.2f}</b></span></div>
      {trade_details}
      <div class="cm-why"><b>Why:</b> {e(explanation)}</div>
      <div class="cm-learning">History/risk feedback: {e(item.get('History note') or 'No completed commodity trades yet')}</div>
      </div>'''
