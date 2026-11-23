"""Conservative, explainable trend signals for GBP/GBp London-listed commodity ETCs.

This is NOT the share ML model, not a trained predictor, and not an estimate
of win probability. Scoring uses only completed daily OHLCV bars. The newest
partial session may be supplied separately as a validated current paper entry.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional
import numpy as np
import pandas as pd

STRATEGY_NAME = "COMMODITY_ETC_TREND_V1"

@dataclass(frozen=True)
class CommoditySignal:
    action: str  # BUY_CANDIDATE or WAIT
    score: int   # number of 5 observable conditions satisfied; NOT a probability
    reason: str
    atr_pence: Optional[float] = None
    momentum_5d_pct: Optional[float] = None
    momentum_20d_pct: Optional[float] = None
    rsi: Optional[float] = None
    target_pence: Optional[float] = None
    stop_pence: Optional[float] = None


def evaluate_commodity_signal(completed: pd.DataFrame, entry_price_pence: float,
                              *, min_turnover_gbp: float = 1_000_000.0,
                              max_atr_pct: float = 8.0) -> CommoditySignal:
    """Score an ETC without observing future returns or today's unfinished bar.

    Rejects negative, crossed, missing or implausible OHLCV bars. The caller
    must independently verify quote currency, freshness and trading hours.
    All prices are in *pence*, volume is in whole ETC units.
    """
    waiting = lambda why: CommoditySignal("WAIT", 0, why)
    try:
        price = float(entry_price_pence)
    except (TypeError, ValueError):
        return waiting("No verified current GBp quote")
    if not math.isfinite(price) or price <= 0:
        return waiting("No verified current GBp quote")
    if completed is None or len(completed) < 65:
        return waiting("At least 65 completed trading days are needed")
    needed = ("Open", "High", "Low", "Close", "Volume")
    if not all(c in completed for c in needed):
        return waiting("Historical OHLCV columns are missing")
    bars = completed[list(needed)].tail(65).apply(pd.to_numeric, errors="coerce")
    vals = bars.to_numpy(dtype=float)
    if (not np.isfinite(vals).all() or (bars[list(needed[:-1])] <= 0).any().any()
            or (bars['Volume'] < 0).any()
            or (bars['High'] < bars[['Open', 'Close', 'Low']].max(axis=1)).any()
            or (bars['Low'] > bars[['Open', 'Close', 'High']].min(axis=1)).any()):
        return waiting("Invalid or inconsistent daily prices")
    c, h, l, v = (bars[k].astype(float) for k in ('Close', 'High', 'Low', 'Volume'))
    avg_turnover = float((c.tail(20) * v.tail(20) / 100.0).mean())
    if not math.isfinite(avg_turnover) or avg_turnover < min_turnover_gbp:
        return waiting("Below the minimum £ turnover or untradeable volume")
    if int((v.tail(20) == 0).sum()) >= 2:
        return waiting("Multiple recent zero-volume days")
    prev = c.shift(1)
    tr = pd.concat([h - l, (h - prev).abs(), (l - prev).abs()], axis=1).max(axis=1)
    atr = float(tr.tail(14).mean())
    atr_pct = atr / float(c.iloc[-1]) * 100.0
    if not math.isfinite(atr) or atr <= 0 or atr_pct > max_atr_pct:
        return waiting("Volatility too high or unmeasurable")
    sma20 = float(c.tail(20).mean())
    sma50 = float(c.tail(50).mean())
    m5 = (float(c.iloc[-1]) / float(c.iloc[-6]) - 1.0) * 100
    m20 = (float(c.iloc[-1]) / float(c.iloc[-21]) - 1.0) * 100
    delta = c.diff()
    gain = float(delta.clip(lower=0).tail(14).mean())
    loss = float((-delta.clip(upper=0)).tail(14).mean())
    rsi = (100.0 if gain > 0 and loss == 0 else
           50.0 if gain == 0 and loss == 0 else
           100.0 - 100.0 / (1 + gain / loss) if loss > 0 else 0.0)

    conditions = (
        ("medium-term upward trend", sma20 > sma50),
        ("above 20-day average", float(c.iloc[-1]) > sma20),
        ("rising in five days", m5 >= 0.5),
        ("rising in twenty days", m20 >= 2.0),
        ("momentum not overheated", 48 <= rsi <= 72),
    )
    score = sum(ok for _, ok in conditions)
    target = round(price + 2.4 * atr, 4)
    stop = round(price - 1.5 * atr, 4)
    if stop <= 0 or not stop < price < target:
        return waiting("Invalid stop/goal at this price")
    notes = ", ".join(label for label, ok in conditions if not ok)
    qualified = score >= 4 and m20 > 0 and float(c.iloc[-1]) >= 0.95 * price
    reason = (f"{score}/5 trend conditions passed (rule-based, not ML)."
              if qualified else
              f"{score}/5 trend conditions; not eligible" + (f" — failed: {notes}" if notes else ""))
    return CommoditySignal("BUY_CANDIDATE" if qualified else "WAIT", score, reason,
                           round(atr, 4), round(m5, 2), round(m20, 2),
                           round(rsi, 1), target, stop)
