"""Conservative one-minute OHLC target/stop verification for PAPER trades.

This module does not execute broker orders and cannot prove an exchange fill.
Bar timestamps mark the START of the minute. The entry minute is deliberately
excluded to avoid using any high/low observed before the virtual entry.
"""
from dataclasses import dataclass
from datetime import datetime
import math
import pandas as pd

LONDON_TZ = "Europe/London"

@dataclass(frozen=True)
class IntradayExit:
    outcome: str          # "WIN" or "STOP"
    fill_price: float     # simulated pence price
    observed_at: str      # Europe/London local time
    ambiguous: bool = False


def as_london_timestamp(value):
    if value is None or str(value).strip().lower() in {"", "none", "nan", "nat"}:
        raise ValueError("Missing entry timestamp")
    ts = pd.Timestamp(value)
    if pd.isna(ts):
        raise ValueError("Invalid entry timestamp")
    if ts.tzinfo is None:
        # The existing journal writes its timestamps in Europe/London local time.
        ts = ts.tz_localize(LONDON_TZ, ambiguous="raise", nonexistent="raise")
    else:
        ts = ts.tz_convert(LONDON_TZ)
    return ts


def first_intraday_exit(bars: pd.DataFrame, entry_time, target: float, stop: float,
                        from_time=None, through_day=None):
    """Return first *observed* post-entry exit from ordered 1-minute bars, or None.

    If both thresholds are crossed within the same minute, choose STOP first.
    A gap below the stop uses the minute's observed opening price (worse fill).
    Bars that start at or before the entry instant are excluded; partial entry
    minutes must not manufacture a pre-entry win.
    """
    if bars is None or bars.empty:
        return None
    if any(x not in bars.columns for x in ("Open", "High", "Low", "Close")):
        return None
    entry = as_london_timestamp(entry_time)
    lower = max(entry, as_london_timestamp(from_time)) if from_time is not None else entry
    if not (math.isfinite(target) and math.isfinite(stop) and 0 < stop < target):
        raise ValueError("Invalid target or stop")
    frame = bars.sort_index()
    if not isinstance(frame.index, pd.DatetimeIndex):
        return None
    idx = frame.index
    if idx.tz is None:
        # yfinance should return exchange-local timezone-aware intraday bars;
        # never guess whether naive values are UTC or London.
        return None
    local_idx = idx.tz_convert(LONDON_TZ)
    open_col, high_col, low_col, close_col = (frame[c].to_numpy() for c in ("Open", "High", "Low", "Close"))
    for t, o, h, l, c in zip(local_idx, open_col, high_col, low_col, close_col):
        if t <= lower:
            continue
        if through_day is not None and t.date() != through_day:
            continue
        vals = (o, h, l, c)
        if not all(math.isfinite(float(v)) for v in vals):
            continue
        o, h, l, c = (float(v) for v in vals)
        if not (0 < l <= min(o, c) and h >= max(o, c, l)):
            continue
        hit_stop = l <= stop
        hit_target = h >= target
        when = t.strftime("%Y-%m-%d %H:%M:%S")
        if hit_stop:
            return IntradayExit("STOP", min(stop, o), when, ambiguous=hit_target)
        if hit_target:
            return IntradayExit("WIN", target, when)
    return None
