import os
import re
import sys
import math
import json
import time
import hmac
import html
import hashlib
import threading
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from datetime import datetime, time as dtime
import random
import requests

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import yfinance as yf
import pytz
import duckdb

try:
    import plotly.graph_objects as go
    PLOTLY_OK = True
except ImportError:
    go = None
    PLOTLY_OK = False

try:
    from streamlit_autorefresh import st_autorefresh
except ImportError:
    st_autorefresh = None

warnings.filterwarnings("ignore")

try:
    from feature_engine import engineer_features
except ImportError:
    engineer_features = None

try:
    from rns_scraper import fetch_direct_rns_for_ticker
except ImportError:
    def fetch_direct_rns_for_ticker(ticker: str):
        return {"headline": "Standard Market Flow", "status": "📰 Flow Verified", "delta": 0.0, "timestamp": datetime.now().strftime("%Y-%m-%d")}

# set_page_config must be the first Streamlit command that renders anything
st.set_page_config(page_title="ALPHA-LSE Quant Terminal", page_icon="⚡", layout="wide", initial_sidebar_state="collapsed")

ROOT_DIR = Path(__file__).resolve().parent
DB_PATH = os.path.join(ROOT_DIR, "lse_market_data.duckdb")
MODEL_PATH = os.path.join(ROOT_DIR, "models", "ensemble_ranker.joblib")          # original champion: NEVER overwritten
LIVE_MODEL_PATH = os.path.join(ROOT_DIR, "models", "ensemble_live.joblib")        # promoted challenger (only if it beats champion)
RETRAIN_STATE_PATH = os.path.join(ROOT_DIR, "models", "retrain_state.json")
AIM_OVERRIDE_PATH = os.path.join(ROOT_DIR, "aim_tickers.json")                    # optional: JSON list of AIM tickers

# ---- Portfolio / risk guardrails ----
MAX_DAILY_EQUITY_TRADES = 5
MAX_HOLD_CALENDAR_DAYS = 7        # 5 LSE trading days
BREAKEVEN_TRIGGER_RATIO = 0.65    # Ratchet stop to friction-adjusted break-even at >= 65% of target distance
MAX_ACTIVE_PER_SECTOR = 2         # max 2 active positions per sector
MAX_OPEN_POSITIONS = 10           # total open-position cap (market-wide correlation guard)
MAX_OPEN_RISK_GBP = 250.0         # total open risk (entry->stop distance x size) cap
MARKET_STRESS_FTSE_5D = -3.0      # pause new entries if FTSE 100 5-day change <= this
COOLDOWN_CALENDAR_DAYS = 3        # anti-churn: no re-entry within 3 days of actual exit
HALF_LIFE_DAYS = 30.0             # exponential decay half-life for historical penalties
POSITION_SIZE_GBP = 500.0

# ---- Entry realism / liquidity ----
ENTRY_WINDOW_START = dtime(8, 30)   # skip the opening auction / first-30-min noise
ENTRY_WINDOW_END = dtime(16, 0)
MIN_PRICE_P = 20.0                  # skip penny stocks (pence)
MIN_AVG_TURNOVER_GBP = 1_000_000    # min 20d average daily traded value (GBP)
MIN_QUALIFY_CONFIDENCE = 65
MIN_NET_RETURN_PCT = 2.0
BE_TOLERANCE_PCT = 0.5              # a protected stop only counts as "break-even" if net loss is within this

# ---- Costs ----
SDRT_PCT = 0.50                     # UK stamp duty reserve tax on purchases of Main Market shares
DEFAULT_COMMISSION_GBP = 3.0        # per trade (each side); adjustable in the sidebar
OPTION_HALF_SPREAD_PCT = 4.0        # simulated option bid/ask half-spread

# ---- ML governance ----
MIN_SETTLED_TO_RETRAIN = 150        # labelled (WIN/LOSS) live trades required before a challenger is even trained
RETRAIN_STEP_INTERVAL = 50          # then only after N additional labelled trades
MIN_CALIBRATION_SAMPLES = 80        # samples required to switch from shrinkage to isotonic calibration
CONF_SHRINK = 0.5                   # shrink raw model probability toward the realised base rate

# Strict LSE Ticker Regex (Integrity Gate against untrusted web scraping)
VALID_LSE_TICKER_RE = re.compile(r"^[A-Z0-9]{1,6}(-[A-Z0-9]{1,2})?\.L$")

# ==============================================================================
# 0. UK TCA, AIM REGISTRY, SECTOR MAP & NLP LEXICON
# ==============================================================================
# AIM-listed equities are exempt from the 0.50% SDRT. VERIFY this list against LSE data (or drop a JSON list
# named aim_tickers.json next to this file to override it). VIC.L (Victrex) removed: it is Main Market.
_DEFAULT_AIM = {
    "AOM.L", "SEE.L", "SRC.L", "SAV.L", "YOU.L", "PTAL.L", "JET2.L", "CER.L",
    "RWS.L", "RKH.L", "KGH.L", "NFG.L", "KP2.L", "BIG.L", "JHD.L", "LTHM.L",
    "CMCL.L", "CAML.L", "CHRT.L", "CNC.L", "CRW.L", "DOTD.L", "FEVR.L", "ITM.L",
    "BUR.L", "BRCK.L", "MIDW.L"
}

def _load_aim_set() -> set:
    try:
        if os.path.exists(AIM_OVERRIDE_PATH):
            with open(AIM_OVERRIDE_PATH, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            if isinstance(data, list) and data:
                return {str(x).strip().upper() for x in data}
    except Exception:
        pass
    return set(_DEFAULT_AIM)

AIM_EXEMPT_TICKERS = _load_aim_set()

SECTOR_MAP = {
    # Energy
    "SHEL.L": "Energy", "BP.L": "Energy", "PTAL.L": "Energy", "RKH.L": "Energy", "ITM.L": "Energy",
    # Basic Materials
    "RIO.L": "Basic Materials", "GLEN.L": "Basic Materials", "LTHM.L": "Basic Materials",
    "CMCL.L": "Basic Materials", "CAML.L": "Basic Materials", "SAV.L": "Basic Materials", "KP2.L": "Basic Materials",
    # Financials
    "HSBA.L": "Financials", "BARC.L": "Financials", "LSEG.L": "Financials", "BUR.L": "Financials",
    "LLOY.L": "Financials", "NWG.L": "Financials", "PRU.L": "Financials", "LGEN.L": "Financials", "AV.L": "Financials",
    # Healthcare
    "AZN.L": "Healthcare", "GSK.L": "Healthcare", "HLN.L": "Healthcare", "SN.L": "Healthcare", "HIK.L": "Healthcare",
    # Consumer
    "ULVR.L": "Consumer", "BATS.L": "Consumer", "FEVR.L": "Consumer", "JET2.L": "Consumer", "NFG.L": "Consumer",
    "TSCO.L": "Consumer", "SBRY.L": "Consumer", "MKS.L": "Consumer", "NXT.L": "Consumer", "DGE.L": "Consumer", "RKT.L": "Consumer",
    # Telecom & Utilities
    "NG.L": "Telecom & Utilities", "VOD.L": "Telecom & Utilities", "BT-A.L": "Telecom & Utilities", "SSE.L": "Telecom & Utilities", "CNA.L": "Telecom & Utilities",
    # Technology
    "DOTD.L": "Technology", "BIG.L": "Technology", "SEE.L": "Technology", "AOM.L": "Technology",
    "YOU.L": "Technology", "CER.L": "Technology", "RWS.L": "Technology", "KGH.L": "Technology", "AUTO.L": "Technology", "SGE.L": "Technology",
    # Defense & Aero
    "CHRT.L": "Defense & Aero", "CNC.L": "Defense & Aero", "BA.L": "Defense & Aero", "RR.L": "Defense & Aero", "QQ.L": "Defense & Aero",
    # Industrials
    "CRW.L": "Industrials", "BRCK.L": "Industrials", "MIDW.L": "Industrials", "VIC.L": "Industrials",
    "SRC.L": "Industrials", "JHD.L": "Industrials", "REL.L": "Industrials", "EXPN.L": "Industrials", "AHT.L": "Industrials"
}

YF_SECTOR_BUCKETS = {
    "Energy": "Energy", "Basic Materials": "Basic Materials", "Financial Services": "Financials",
    "Healthcare": "Healthcare", "Consumer Cyclical": "Consumer", "Consumer Defensive": "Consumer",
    "Communication Services": "Telecom & Utilities", "Utilities": "Telecom & Utilities",
    "Technology": "Technology", "Industrials": "Industrials", "Real Estate": "Real Estate",
}

FTSE_EXPORTERS = {"AZN.L", "GSK.L", "SHEL.L", "BP.L", "ULVR.L", "BATS.L", "RIO.L", "GLEN.L", "DGE.L", "REL.L", "CRW.L"}

# Indicative dividend yields used ONLY by the simulated option pricer (assumptions, not live data)
OPTION_DIV_YIELD = {"SHEL": 0.04, "BP": 0.05, "HSBA": 0.06, "BARC": 0.03, "GLEN": 0.02}

RNS_BULLISH_LEXICON = {
    "ahead of expectations": 0.08, "exceeds expectations": 0.08, "materially ahead": 0.09,
    "contract win": 0.06, "new contract": 0.05, "share buyback": 0.05,
    "director dealing": 0.04, "pdmr": 0.03, "dividend increase": 0.04,
    "record revenue": 0.06, "upgraded guidance": 0.07, "recommended cash offer": 0.10
}

RNS_BEARISH_LEXICON = {
    "placing": -0.25, "subscription": -0.20, "accelerated bookbuild": -0.25,
    "dilution": -0.25, "profit warning": -0.25, "below expectations": -0.18,
    "materially below": -0.22, "going concern": -0.30, "suspension": -0.30,
    "covenant": -0.15, "winding up": -0.35
}

@st.cache_resource
def get_db_lock():
    return threading.Lock()

DB_LOCK = get_db_lock()

@st.cache_resource
def get_sector_cache():
    return {}

@st.cache_resource
def get_hist_store():
    return {"lock": threading.Lock(), "data": {}}

# ---------- small shared helpers ----------
def london_now() -> datetime:
    return datetime.now(pytz.timezone('Europe/London'))

def esc(x) -> str:
    return html.escape(str(x))

def _collapse(s: str) -> str:
    return re.sub(r"\s*\n\s*", "", s.strip())

def _parse_features(f) -> dict:
    if isinstance(f, dict):
        return f
    try:
        obj = json.loads(f) if f else {}
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}

def _to_date(s):
    try:
        return datetime.strptime(str(s)[:10], "%Y-%m-%d").date()
    except Exception:
        return None

def _first_valid(*vals):
    for v in vals:
        if v is None:
            continue
        if isinstance(v, float) and math.isnan(v):
            continue
        sv = str(v).strip()
        if sv and sv.lower() not in ("nan", "none", "nat"):
            return sv
    return ""

def _bar_dates(h: pd.DataFrame) -> list:
    return [pd.Timestamp(x).date() for x in h.index]

def is_valid_lse_ticker(ticker: str) -> bool:
    return bool(isinstance(ticker, str) and VALID_LSE_TICKER_RE.match(ticker.strip()))

def is_aim_exempt(ticker: str) -> bool:
    return str(ticker).strip().upper() in AIM_EXEMPT_TICKERS

def classify_outcome(status) -> str:
    s = str(status).upper()
    if "WIN" in s:
        return "WIN"
    if "BREAK-EVEN" in s:
        return "BE"
    if "LOSS" in s:
        return "LOSS"
    if "EXPIRED" in s:
        return "EXPIRED"
    if "ACTIVE" in s:
        return "ACTIVE"
    return "OTHER"

def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple:
    if n <= 0:
        return 0.0, 0.0
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)

# ---------- sector resolution (no more random md5 sectors) ----------
def get_ticker_sector(ticker: str, resolve: bool = True) -> str:
    t = str(ticker).strip().upper()
    if t in SECTOR_MAP:
        return SECTOR_MAP[t]
    cache = get_sector_cache()
    hit = cache.get(t)
    if hit and (hit[0] != "Unclassified" or (time.time() - hit[1]) < 3600):
        return hit[0]
    if not resolve or not is_valid_lse_ticker(t):
        return hit[0] if hit else "Unclassified"
    sector = "Unclassified"
    try:
        info = yf.Ticker(t).info or {}
        raw = info.get("sector")
        if raw:
            sector = YF_SECTOR_BUCKETS.get(raw, str(raw))
    except Exception:
        pass
    cache[t] = (sector, time.time())
    return sector

def peek_sector(ticker: str):
    """Sector if already known (static map or cache); None otherwise. Never hits the network."""
    t = str(ticker).strip().upper()
    if t in SECTOR_MAP:
        return SECTOR_MAP[t]
    hit = get_sector_cache().get(t)
    if hit and hit[0] != "Unclassified":
        return hit[0]
    return None

def get_trade_sector(ticker: str, f_json) -> str:
    t = str(ticker).strip().upper()
    if t in SECTOR_MAP:
        return SECTOR_MAP[t]
    f = _parse_features(f_json)
    if f.get("sector"):
        return str(f["sector"])
    return get_ticker_sector(t, resolve=False)

# ---------- UK trading costs ----------
def get_commission_gbp() -> float:
    try:
        return float(st.session_state.get("commission_gbp", DEFAULT_COMMISSION_GBP))
    except Exception:
        return DEFAULT_COMMISSION_GBP

def commission_round_trip_pct(commission_gbp: float) -> float:
    return round(2.0 * float(commission_gbp) / POSITION_SIZE_GBP * 100.0, 3)

def get_sdrt_pct(ticker: str) -> float:
    return 0.0 if is_aim_exempt(ticker) else SDRT_PCT

def default_spread_pct(ticker: str) -> float:
    return 0.25 if is_aim_exempt(ticker) else 0.10

def estimate_spread_pct(ticker: str, avg_turnover_gbp) -> float:
    """Liquidity-tiered round-trip spread estimate."""
    try:
        v = float(avg_turnover_gbp)
        if not math.isfinite(v):
            raise ValueError
        if v >= 20_000_000:
            base = 0.10
        elif v >= 5_000_000:
            base = 0.20
        elif v >= 1_000_000:
            base = 0.45
        else:
            base = 0.90
    except Exception:
        base = default_spread_pct(ticker)
    return max(base, 0.25) if is_aim_exempt(ticker) else base

def compute_friction_pct(ticker: str, spread_pct=None, commission_gbp=None) -> float:
    spread = default_spread_pct(ticker) if spread_pct is None else float(spread_pct)
    comm = get_commission_gbp() if commission_gbp is None else float(commission_gbp)
    return round(get_sdrt_pct(ticker) + spread + commission_round_trip_pct(comm), 3)

def get_trade_friction_pct(ticker: str, f_json) -> float:
    """Friction for a stored trade. Old rows (no commission/spread fields) keep their original assumptions."""
    f = _parse_features(f_json)
    sdrt = get_sdrt_pct(ticker)
    if "spread_pct" in f:
        try:
            spread = float(f["spread_pct"])
        except Exception:
            spread = default_spread_pct(ticker)
    elif "friction_pct" in f:
        try:
            spread = max(0.0, float(f["friction_pct"]) - sdrt)
        except Exception:
            spread = default_spread_pct(ticker)
    else:
        spread = default_spread_pct(ticker)
    try:
        commission = float(f.get("commission_gbp", 0.0))
    except Exception:
        commission = 0.0
    return round(sdrt + spread + commission_round_trip_pct(commission), 3)

def get_breakeven_exit_price(entry_price: float, friction_pct: float) -> float:
    return round(float(entry_price) * (1.0 + (float(friction_pct) / 100.0)), 2)

def calc_net_equity_pnl_pct(entry_price: float, current_price: float, friction_pct: float) -> float:
    if entry_price <= 0:
        return 0.0
    gross_pct = ((float(current_price) - float(entry_price)) / float(entry_price)) * 100.0
    return round(gross_pct - float(friction_pct), 2)

def is_stop_breakeven_protected(entry_price: float, stop_loss: float, ticker: str = "") -> bool:
    return float(stop_loss) >= float(entry_price) - 1e-4

def classify_stop_exit(entry: float, stop_at_hit: float, fill: float, friction_pct: float) -> str:
    """BE label only when the stop was ratcheted AND the actual fill is close to break-even net of costs."""
    net = calc_net_equity_pnl_pct(entry, fill, friction_pct)
    if stop_at_hit >= entry - 1e-4 and net >= -BE_TOLERANCE_PCT:
        return "🛡️ BREAK-EVEN (PROTECTED EXIT)"
    return "🛑 LOSS (STOPPED OUT)"

def evaluate_completed_bar(o, hi, lo, c, entry, target, stop, be_floor, be_trigger):
    """Walk one COMPLETED daily bar. Gap-aware, stop-first on ambiguity, ratchet only affects the NEXT bar.
    Returns (outcome, fill_price, new_stop) with outcome in {None, 'WIN', 'STOP'}."""
    if o >= target:
        return "WIN", o, stop
    if o <= stop:
        return "STOP", o, stop
    if lo <= stop:                      # includes bars that touched both -> conservative: stop first
        return "STOP", stop, stop
    if hi >= target:
        return "WIN", target, stop
    new_stop = stop
    if target > entry and stop < be_floor and hi >= be_trigger and c > be_floor:
        new_stop = be_floor
    return None, c, new_stop

def evaluate_rns_nlp_sentiment(ticker: str) -> dict:
    raw_rns = fetch_direct_rns_for_ticker(ticker)
    headline = str(raw_rns.get("headline", "")).lower()
    status = str(raw_rns.get("status", "📰 Flow Verified"))
    base_delta = float(raw_rns.get("delta", 0.0))

    lex_delta = 0.0
    matched_tags = []
    for phrase, weight in RNS_BEARISH_LEXICON.items():
        if phrase in headline:
            lex_delta += weight
            matched_tags.append(f"⚠️ {phrase.title()}")
    for phrase, weight in RNS_BULLISH_LEXICON.items():
        if phrase in headline:
            lex_delta += weight
            matched_tags.append(f"🚀 {phrase.title()}")

    total_delta = max(-0.30, min(0.15, base_delta + lex_delta))
    if total_delta <= -0.15 or "Dilution" in status:
        status = "🚨 Dilution / Adverse RNS"
    elif total_delta >= 0.04:
        status = "🟢 Bullish RNS Catalyst"

    return {
        "headline": raw_rns.get("headline", "Standard Market Flow"),
        "status": status,
        "delta": round(total_delta, 4),
        "nlp_tags": ", ".join(matched_tags) if matched_tags else "Neutral Regulatory Flow"
    }

# ==============================================================================
# 1. HYBRID DATABASE (THREAD-SAFE DUCKDB + SUPABASE CLOUD)
# ==============================================================================
def init_duckdb_storage():
    with DB_LOCK:
        con = duckdb.connect(DB_PATH, read_only=False)
        try:
            con.execute("""
                CREATE TABLE IF NOT EXISTS trade_journal (
                    trade_id VARCHAR PRIMARY KEY,
                    timestamp VARCHAR,
                    date_str VARCHAR,
                    ticker VARCHAR,
                    asset_type VARCHAR DEFAULT 'EQUITY',
                    entry_price DOUBLE,
                    target_price DOUBLE,
                    stop_loss DOUBLE,
                    shares INTEGER,
                    capital_allocated DOUBLE,
                    status VARCHAR DEFAULT 'ACTIVE',
                    latest_price DOUBLE,
                    pnl_pct DOUBLE DEFAULT 0.0,
                    exit_price DOUBLE DEFAULT 0.0,
                    exit_timestamp VARCHAR,
                    last_audited VARCHAR,
                    features_json VARCHAR DEFAULT '{}'
                )
            """)
            try:
                con.execute("ALTER TABLE trade_journal ADD COLUMN features_json VARCHAR DEFAULT '{}'")
            except Exception:
                pass

            con.execute("""
                CREATE TABLE IF NOT EXISTS daily_options_journal (
                    date_key VARCHAR PRIMARY KEY,
                    timestamp VARCHAR,
                    share_name VARCHAR,
                    option_contract VARCHAR,
                    strike_price DOUBLE,
                    expiry_days INTEGER,
                    underlying_spot DOUBLE,
                    lot_size INTEGER,
                    entry_premium DOUBLE,
                    current_option_price DOUBLE,
                    target_premium DOUBLE,
                    stop_loss_premium DOUBLE,
                    total_capital DOUBLE,
                    ai_confidence DOUBLE,
                    implied_vol DOUBLE,
                    status VARCHAR DEFAULT 'ACTIVE',
                    pnl_pct DOUBLE DEFAULT 0.0,
                    last_audited VARCHAR
                )
            """)
        finally:
            con.close()

init_duckdb_storage()

try:
    from supabase import create_client
except ImportError:
    create_client = None

@st.cache_resource
def get_supabase_client():
    if create_client is None:
        return None
    url = st.secrets.get("SUPABASE_URL") if hasattr(st, "secrets") else None
    key = st.secrets.get("SUPABASE_KEY") if hasattr(st, "secrets") else None
    if not url or not key:
        return None
    try:
        clean_url = url.strip().split("/rest/v1")[0].rstrip("/")
        return create_client(clean_url, key.strip())
    except Exception:
        st.session_state["db_error"] = "Supabase connection failed (Check secrets configuration)."
        return None

supabase = get_supabase_client()

def record_db_error(context: str, err: Exception):
    raw_msg = str(err)
    sanitized = re.sub(r"https?://[^\s'\"]+", "[REDACTED_URL]", raw_msg)
    st.session_state["db_error"] = f"[{context}] {sanitized[:140]}"

def cloud_update(table: str, match: dict, payload: dict, label: str) -> bool:
    """Supabase update with a retry queue so a failed sync is re-pushed instead of being reverted by hydrate."""
    if not supabase:
        return True
    try:
        q = supabase.table(table).update(payload)
        for k, v in match.items():
            q = q.eq(k, v)
        q.execute()
        return True
    except Exception as e:
        record_db_error(label, e)
        if "pending_cloud_sync" not in st.session_state:
            st.session_state["pending_cloud_sync"] = {}
        key = (table, tuple(sorted((str(k), str(v)) for k, v in match.items())))
        st.session_state["pending_cloud_sync"][key] = (table, dict(match), dict(payload))
        return False

def flush_pending_cloud_sync():
    pend = st.session_state.get("pending_cloud_sync", {})
    if not supabase or not pend:
        return
    for key, (table, match, payload) in list(pend.items()):
        try:
            q = supabase.table(table).update(payload)
            for k, v in match.items():
                q = q.eq(k, v)
            q.execute()
            del pend[key]
        except Exception as e:
            record_db_error("Retry Cloud Sync", e)
            break

def fetch_all_supabase_rows(table_name: str) -> list:
    """Paginated retrieval beyond the PostgREST 1,000-row ceiling."""
    if not supabase:
        return []
    all_rows = []
    chunk_size = 1000
    start = 0
    while True:
        try:
            res = supabase.table(table_name).select("*").range(start, start + chunk_size - 1).execute()
            if not res.data:
                break
            all_rows.extend(res.data)
            if len(res.data) < chunk_size:
                break
            start += chunk_size
        except Exception as e:
            record_db_error(f"Paginate {table_name}", e)
            break
    return all_rows

def hydrate_duckdb_from_supabase():
    """Restores trades from Supabase WITHOUT clobbering newer local state (a closed trade never reverts to ACTIVE)."""
    if not supabase:
        return
    eq_data = fetch_all_supabase_rows("predictions")      # network I/O outside the DB lock
    opt_data = fetch_all_supabase_rows("options_journal")

    with DB_LOCK:
        con = duckdb.connect(DB_PATH, read_only=False)
        try:
            local_eq = {}
            local_opt = {}
            try:
                for r in con.execute("SELECT trade_id, status, last_audited, ticker, date_str, stop_loss, latest_price, pnl_pct FROM trade_journal").fetchall():
                    local_eq[r[0]] = r
                for r in con.execute("SELECT date_key, status, last_audited, current_option_price, pnl_pct FROM daily_options_journal").fetchall():
                    local_opt[r[0]] = r
            except Exception:
                pass

            try:
                for r in (eq_data or []):
                    tkr = str(r.get('ticker', '')).strip()
                    if not is_valid_lse_ticker(tkr):
                        continue
                    pred_date = str(r.get('predicted_date', london_now().strftime('%Y-%m-%d')))
                    trade_id = f"{tkr}_{pred_date}"
                    remote_status = str(r.get('status', 'ACTIVE')).upper()
                    remote_ts = str(r.get('last_checked') or "")
                    loc = local_eq.get(trade_id)
                    if loc:
                        l_status, l_ts = str(loc[1]), str(loc[2] or "")
                        local_closed_remote_active = (l_status != "ACTIVE" and remote_status == "ACTIVE")
                        local_newer = bool(l_ts) and (not remote_ts or l_ts >= remote_ts)
                        if local_closed_remote_active or local_newer:
                            if local_closed_remote_active:
                                cloud_update("predictions", {"ticker": loc[3], "predicted_date": loc[4]},
                                             {"status": l_status, "stop_loss": float(loc[5] or 0.0), "latest_price": float(loc[6] or 0.0),
                                              "pnl_pct": float(loc[7] or 0.0), "last_checked": l_ts}, "Re-push Closed Trade")
                            continue
                    last_check = remote_ts or london_now().strftime('%Y-%m-%d %H:%M:%S')
                    f_json = json.dumps(r.get('features_json') or {})
                    exit_ts_val = last_check if remote_status != "ACTIVE" else None
                    con.execute("""
                        INSERT OR REPLACE INTO trade_journal 
                        (trade_id, timestamp, date_str, ticker, asset_type, entry_price, target_price,
                         stop_loss, shares, capital_allocated, status, latest_price, pnl_pct, exit_price,
                         exit_timestamp, last_audited, features_json)
                        VALUES (?, ?, ?, ?, 'EQUITY', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, [
                        trade_id, last_check, pred_date, tkr,
                        float(r.get('entry_price', 0.0)), float(r.get('target_price', 0.0)),
                        float(r.get('stop_loss', 0.0)), int(r.get('shares_qty', 1)),
                        float(r.get('position_gbp', 0.0)), remote_status,
                        float(r.get('latest_price', 0.0)), float(r.get('pnl_pct', 0.0)),
                        float(r.get('latest_price', 0.0)) if remote_status != "ACTIVE" else 0.0,
                        exit_ts_val, last_check, f_json
                    ])
                if "db_error" in st.session_state and "Hydrate" in st.session_state["db_error"]:
                    del st.session_state["db_error"]
            except Exception as e:
                record_db_error("Hydrate Equities", e)

            try:
                for o in (opt_data or []):
                    date_k = str(o.get('date_key', london_now().strftime('%Y-%m-%d')))
                    remote_status = str(o.get('status', 'ACTIVE')).upper()
                    remote_ts = str(o.get('last_audited') or "")
                    loc = local_opt.get(date_k)
                    if loc:
                        l_status, l_ts = str(loc[1]), str(loc[2] or "")
                        if (l_status != "ACTIVE" and remote_status == "ACTIVE") or (l_ts and (not remote_ts or l_ts >= remote_ts)):
                            if l_status != "ACTIVE" and remote_status == "ACTIVE":
                                cloud_update("options_journal", {"date_key": date_k},
                                             {"status": l_status, "current_option_price": float(loc[3] or 0.0),
                                              "pnl_pct": float(loc[4] or 0.0), "last_audited": l_ts}, "Re-push Closed Option")
                            continue
                    con.execute("""
                        INSERT OR REPLACE INTO daily_options_journal 
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, [
                        date_k, str(o.get('timestamp') or ""), str(o.get('share_name')),
                        str(o.get('option_contract')), float(o.get('strike_price', 0.0)),
                        int(o.get('expiry_days', 21)), float(o.get('underlying_spot', 0.0)),
                        int(o.get('lot_size', 5000)), float(o.get('entry_premium', 0.0)),
                        float(o.get('current_option_price', 0.0)), float(o.get('target_premium', 0.0)),
                        float(o.get('stop_loss_premium', 0.0)), float(o.get('total_capital', 0.0)),
                        float(o.get('ai_confidence', 80.0)), float(o.get('implied_vol', 20.0)),
                        remote_status, float(o.get('pnl_pct', 0.0)), remote_ts
                    ])
            except Exception as e:
                record_db_error("Hydrate Options", e)
        finally:
            con.close()

# ==============================================================================
# 2. TIMEZONE, MARKET CLOCK, MACRO FEED & PARALLEL DATA CACHING
# ==============================================================================
LOT_SIZES = {"SHEL": 1000, "AZN": 500, "HSBA": 2000, "ULVR": 500, "BP": 3000, "BARC": 5000, "RIO": 250, "GLEN": 4000}

def is_lse_market_open() -> bool:
    now_lon = london_now()
    return dtime(8, 0) <= now_lon.time() <= dtime(16, 30) and now_lon.weekday() <= 4

def is_entry_window_open() -> bool:
    now_lon = london_now()
    return ENTRY_WINDOW_START <= now_lon.time() <= ENTRY_WINDOW_END and now_lon.weekday() <= 4

def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

def calculate_black_scholes_call(spot: float, strike: float, days_to_exp: float, r: float, sigma: float, q: float = 0.0) -> float:
    """Black-Scholes call with continuous dividend yield q (mid price)."""
    T = max(days_to_exp, 1.0) / 365.0
    if spot <= 0 or strike <= 0 or sigma <= 0:
        return max(0.0, spot - strike)
    d1 = (math.log(spot / strike) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    call_price = spot * math.exp(-q * T) * norm_cdf(d1) - strike * math.exp(-r * T) * norm_cdf(d2)
    return max(round(call_price, 2), 0.50)

def option_ask_price(spot, strike, days, sigma, q):
    return round(calculate_black_scholes_call(spot, strike, days, 0.05, sigma, q) * (1.0 + OPTION_HALF_SPREAD_PCT / 100.0), 2)

def option_bid_price(spot, strike, days, sigma, q):
    return max(round(calculate_black_scholes_call(spot, strike, days, 0.05, sigma, q) * (1.0 - OPTION_HALF_SPREAD_PCT / 100.0), 2), 0.01)

@st.cache_data(ttl=300, show_spinner=False)
def fetch_cached_history(ticker: str, period: str = "120d") -> pd.DataFrame:
    """Fault-tolerant history fetcher with retry and timeout guards."""
    if not is_valid_lse_ticker(ticker) and not str(ticker).startswith("^"):
        return pd.DataFrame()
    for attempt in range(2):
        try:
            df = yf.Ticker(ticker).history(period=period, auto_adjust=False, timeout=6)
            if df is not None and not df.empty:
                return df
        except Exception:
            if attempt == 0:
                time.sleep(0.5)
    return pd.DataFrame()

def _download_history_raw(ticker: str, period: str) -> pd.DataFrame:
    for attempt in range(2):
        try:
            df = yf.Ticker(ticker).history(period=period, auto_adjust=False, timeout=6)
            if df is not None and not df.empty:
                return df
        except Exception:
            time.sleep(0.4)
    return pd.DataFrame()

def prefetch_histories(tickers: list, period: str = "120d", ttl: int = 300):
    """Parallel download for the scan (no Streamlit calls inside worker threads)."""
    store = get_hist_store()
    now = time.time()
    with store["lock"]:
        for k in [k for k, v in store["data"].items() if now - v[0] > 3600]:
            del store["data"][k]
        todo = [t for t in tickers if is_valid_lse_ticker(t) and ((t, period) not in store["data"] or now - store["data"][(t, period)][0] > ttl)]
    if not todo:
        return
    with ThreadPoolExecutor(max_workers=8) as ex:
        results = list(ex.map(lambda t: (t, _download_history_raw(t, period)), todo))
    with store["lock"]:
        for t, df in results:
            store["data"][(t, period)] = (now, df)

def get_history_fast(ticker: str, period: str = "120d") -> pd.DataFrame:
    store = get_hist_store()
    with store["lock"]:
        item = store["data"].get((ticker, period))
    if item is not None and item[1] is not None and not item[1].empty:
        return item[1].copy()
    return fetch_cached_history(ticker, period)

@st.cache_data(ttl=1800, show_spinner=False)
def get_cross_asset_macro_regime() -> dict:
    macro = {"brent_5d": 0.0, "copper_5d": 0.0, "gbpusd_5d": 0.0, "ftse_5d": 0.0}
    symbols = {"brent_5d": "BZ=F", "copper_5d": "HG=F", "gbpusd_5d": "GBPUSD=X", "ftse_5d": "^FTSE"}
    for key, sym in symbols.items():
        try:
            h = yf.Ticker(sym).history(period="10d", timeout=5)
            if h is not None and len(h) >= 5:
                c_now = float(h["Close"].iloc[-1])
                c_5d = float(h["Close"].iloc[-5])
                if c_5d > 0:
                    macro[key] = round(((c_now - c_5d) / c_5d) * 100.0, 2)
        except Exception:
            pass
    return macro

@st.cache_data(ttl=1800, show_spinner=False)
def get_ftse_close_series() -> pd.Series:
    try:
        h = yf.Ticker("^FTSE").history(period="6mo", timeout=6)
        if h is None or h.empty:
            return pd.Series(dtype=float)
        s = h["Close"].copy()
        s.index = pd.to_datetime([pd.Timestamp(x).date() for x in s.index])
        return s.sort_index()
    except Exception:
        return pd.Series(dtype=float)

def ftse_return_between(series: pd.Series, d0, d1):
    try:
        if series is None or series.empty or d0 is None or d1 is None:
            return None
        a = series.asof(pd.Timestamp(d0))
        b = series.asof(pd.Timestamp(d1))
        if pd.isna(a) or pd.isna(b) or float(a) <= 0:
            return None
        return (float(b) / float(a) - 1.0) * 100.0
    except Exception:
        return None

def is_market_stress(macro: dict) -> bool:
    return float(macro.get("ftse_5d", 0.0)) <= MARKET_STRESS_FTSE_5D

def compute_macro_lead_lag_adjustment(ticker: str, sector: str, macro: dict) -> tuple:
    delta = 0
    notes = []
    brent = macro.get("brent_5d", 0.0)
    copper = macro.get("copper_5d", 0.0)
    gbp = macro.get("gbpusd_5d", 0.0)
    ftse = macro.get("ftse_5d", 0.0)

    if sector == "Energy":
        if brent >= 2.0:
            delta += 3
            notes.append(f"+3% (Brent Crude +{brent:.1f}%)")
        elif brent <= -2.5:
            delta -= 3
            notes.append(f"-3% (Brent Crude {brent:.1f}%)")
    elif sector == "Basic Materials":
        if copper >= 1.5:
            delta += 3
            notes.append(f"+3% (Copper +{copper:.1f}%)")
        elif copper <= -2.0:
            delta -= 3
            notes.append(f"-3% (Copper {copper:.1f}%)")

    if ticker in FTSE_EXPORTERS and gbp <= -0.8:
        delta += 2
        notes.append(f"+2% (FX Exporter Tailwind GBP {gbp:.1f}%)")

    if ftse <= -2.0:
        delta -= 2
        notes.append(f"-2% (FTSE 100 Risk-Off {ftse:.1f}%)")
    elif ftse >= 1.5 and delta == 0:
        delta += 1
        notes.append(f"+1% (FTSE 100 Tailwind +{ftse:.1f}%)")

    delta = max(-4, min(4, delta))
    return delta, (" | ".join(notes) if notes else "Neutral Macro Regime")

# ==============================================================================
# 3. HARDENED ZERO-TRUST AUTHENTICATION (UNCHANGED)
# ==============================================================================
def check_password() -> bool:
    if st.session_state.get("password_correct", False):
        return True

    now_ts = time.time()
    lockout_until = st.session_state.get("auth_lockout_until", 0.0)
    if now_ts < lockout_until:
        rem_min = int(math.ceil((lockout_until - now_ts) / 60.0))
        st.error(f"🔒 Terminal Locked due to repeated failed attempts. Try again in {rem_min} minute(s).")
        return False

    expected_user = st.secrets.get("AUTH_USER", "admin") if hasattr(st, "secrets") else "admin"
    default_hash = hashlib.sha256("AlphaLSE2026!".encode("utf-8")).hexdigest()
    expected_hash = st.secrets.get("AUTH_PASS_HASH", default_hash) if hasattr(st, "secrets") else default_hash

    st.subheader("🔐 ALPHA-LSE Quant Terminal — Restricted Institutional Access")
    with st.form("cia_secure_login_form", clear_on_submit=True):
        user_in = st.text_input("Operator Username")
        pass_in = st.text_input("Cryptographic Passkey", type="password")
        submitted = st.form_submit_button("Authenticate Session", width="stretch", type="primary")

        if submitted:
            input_hash = hashlib.sha256(pass_in.encode("utf-8")).hexdigest()
            user_ok = hmac.compare_digest(user_in.strip(), str(expected_user).strip())
            pass_ok = hmac.compare_digest(input_hash, str(expected_hash).strip().lower())

            if user_ok and pass_ok:
                st.session_state["password_correct"] = True
                st.session_state["failed_auth_attempts"] = 0
                st.rerun()
            else:
                fails = st.session_state.get("failed_auth_attempts", 0) + 1
                st.session_state["failed_auth_attempts"] = fails
                if fails >= 5:
                    st.session_state["auth_lockout_until"] = time.time() + 900
                    st.error("🔒 Maximum authentication attempts exceeded. Terminal locked for 15 minutes.")
                else:
                    st.error(f"😕 Invalid credentials ({5 - fails} attempt(s) remaining before lockout).")
    return False

if not check_password():
    st.stop()

if "hydrated_once" not in st.session_state:
    hydrate_duckdb_from_supabase()
    st.session_state["hydrated_once"] = True

# ==============================================================================
# 4. AUDIT & RECONCILIATION ENGINE (BAR-WALKING, GAP-AWARE, STOP-FIRST)
# ==============================================================================
def audit_and_reconcile_all_trades():
    now_lon = london_now()
    now_str = now_lon.strftime('%Y-%m-%d %H:%M:%S')
    today_date = now_lon.date()

    flush_pending_cloud_sync()

    with DB_LOCK:
        con = duckdb.connect(DB_PATH, read_only=False)
        try:
            active_trades = con.execute("SELECT * FROM trade_journal WHERE status = 'ACTIVE'").df()
            active_opts = con.execute("SELECT * FROM daily_options_journal WHERE status = 'ACTIVE'").df()
        finally:
            con.close()

    # ---------------- Equities ----------------
    if not active_trades.empty:
        for _, tr in active_trades.iterrows():
            tkr = str(tr["ticker"]).strip()
            if not is_valid_lse_ticker(tkr):
                continue
            try:
                h = fetch_cached_history(tkr, period="15d")
                if h.empty:
                    continue
                dates = _bar_dates(h)

                entry = float(tr["entry_price"])
                target = float(tr["target_price"])
                stop = float(tr["stop_loss"])
                friction = get_trade_friction_pct(tkr, tr.get("features_json"))
                be_floor = get_breakeven_exit_price(entry, friction)
                target_dist = target - entry
                be_trigger = entry + BREAKEVEN_TRIGGER_RATIO * target_dist

                entry_dt = _to_date(tr["date_str"])
                if entry_dt is None:
                    continue
                last_aud_dt = _to_date(tr.get("last_audited")) or entry_dt
                start_after = max(entry_dt, last_aud_dt)

                new_status = "ACTIVE"
                exit_price = 0.0
                exit_ts = None
                curr = float(h["Close"].iloc[-1])

                # 1) COMPLETED bars since the last audit (catches days the app was asleep)
                for i, d in enumerate(dates):
                    if d <= start_after or d >= today_date:
                        continue
                    row = h.iloc[i]
                    o, hi, lo, c = float(row["Open"]), float(row["High"]), float(row["Low"]), float(row["Close"])
                    if not all(math.isfinite(v) for v in (o, hi, lo, c)):
                        continue
                    outcome, fill, new_stop = evaluate_completed_bar(o, hi, lo, c, entry, target, stop, be_floor, be_trigger)
                    if outcome == "WIN":
                        new_status, exit_price = "🎯 WIN (TARGET HIT)", fill
                    elif outcome == "STOP":
                        new_status = classify_stop_exit(entry, stop, fill, friction)
                        exit_price = fill
                    else:
                        stop = new_stop
                        curr = c
                    if outcome:
                        exit_ts = f"{d} 16:30:00"
                        break

                # 2) LIVE bar (today) — original entry-day guard preserved
                if new_status == "ACTIVE":
                    have_live = bool(dates) and dates[-1] == today_date
                    if have_live:
                        last = h.iloc[-1]
                        curr = float(last["Close"])
                        day_high = float(last["High"]) if "High" in h.columns else curr
                        day_low = float(last["Low"]) if "Low" in h.columns else curr
                        same_day = (entry_dt == today_date)

                        if target_dist > 0 and stop < be_floor:
                            ratchet_hit = (curr >= be_trigger) if same_day else (curr >= be_trigger or day_high >= be_trigger)
                            if ratchet_hit and curr > be_floor:
                                stop = be_floor
                        is_be = is_stop_breakeven_protected(entry, stop)

                        hit_target = (curr >= target) if same_day else (curr >= target or day_high >= target)
                        hit_stop = (curr <= stop) if (same_day or is_be) else (curr <= stop or day_low <= stop)

                        if hit_stop:      # stop wins ties (conservative)
                            fill = curr if curr <= stop else stop
                            new_status = classify_stop_exit(entry, stop, fill, friction)
                            exit_price = fill
                        elif hit_target:
                            new_status, exit_price = "🎯 WIN (TARGET HIT)", target

                # 3) Time exit
                if new_status == "ACTIVE" and (today_date - entry_dt).days >= MAX_HOLD_CALENDAR_DAYS:
                    new_status = "⏱️ EXPIRED (TIME EXIT)"
                    exit_price = curr

                mark = exit_price if new_status != "ACTIVE" else curr
                pnl = calc_net_equity_pnl_pct(entry, mark, friction)
                ts_for_exit = exit_ts or now_str

                with DB_LOCK:
                    con = duckdb.connect(DB_PATH, read_only=False)
                    try:
                        con.execute("""
                            UPDATE trade_journal
                            SET latest_price = ?, stop_loss = ?, pnl_pct = ?, status = ?, exit_price = ?,
                                exit_timestamp = CASE WHEN ? != 'ACTIVE' THEN ? ELSE exit_timestamp END,
                                last_audited = ?
                            WHERE trade_id = ?
                        """, [mark, stop, pnl, new_status, exit_price, new_status, ts_for_exit, now_str, tr["trade_id"]])
                    finally:
                        con.close()

                cloud_update("predictions", {"ticker": tkr, "predicted_date": tr["date_str"]},
                             {"status": new_status, "stop_loss": stop, "latest_price": mark, "pnl_pct": pnl, "last_checked": now_str},
                             "Audit Equity Sync")
            except Exception:
                continue

    # ---------------- Options (SIMULATED) ----------------
    if not active_opts.empty:
        for _, opt in active_opts.iterrows():
            try:
                share = str(opt['share_name'])
                h = fetch_cached_history(f"{share}.L", period="15d")
                if h.empty:
                    continue
                dates = _bar_dates(h)
                strike = float(opt["strike_price"])
                entry_prem = float(opt["entry_premium"])
                target_prem = float(opt["target_premium"])
                stop_prem = float(opt["stop_loss_premium"])
                sigma = float(opt["implied_vol"]) / 100.0
                q = OPTION_DIV_YIELD.get(share, 0.03)
                expiry_days = int(opt["expiry_days"])
                opt_dt = _to_date(opt["date_key"]) or today_date
                last_aud_dt = _to_date(opt.get("last_audited")) or opt_dt
                start_after = max(opt_dt, last_aud_dt)

                def prem_for(spot, d):
                    rem = max(1, expiry_days - max(0, (d - opt_dt).days))
                    return option_bid_price(spot, strike, rem, sigma, q)

                opt_status = "ACTIVE"
                live_prem = None

                for i, d in enumerate(dates):
                    if d <= start_after or d >= today_date:
                        continue
                    row = h.iloc[i]
                    lo_p = prem_for(float(row["Low"]), d)
                    hi_p = prem_for(float(row["High"]), d)
                    if lo_p <= stop_prem:
                        opt_status, live_prem = "🛑 LOSS (STOPPED OUT)", stop_prem
                        break
                    if hi_p >= target_prem:
                        opt_status, live_prem = "🎯 WIN (TARGET HIT)", target_prem
                        break

                current_spot = float(h["Close"].iloc[-1])
                elapsed_days = max(0, (today_date - opt_dt).days)
                if opt_status == "ACTIVE":
                    live_prem = prem_for(current_spot, today_date)
                    if live_prem >= target_prem:
                        opt_status, live_prem = "🎯 WIN (TARGET HIT)", target_prem
                    elif live_prem <= stop_prem:
                        opt_status, live_prem = "🛑 LOSS (STOPPED OUT)", stop_prem
                    elif elapsed_days >= MAX_HOLD_CALENDAR_DAYS:
                        opt_status = "⏱️ EXPIRED (TIME EXIT)"

                pnl_pct = round(((live_prem - entry_prem) / entry_prem) * 100.0, 2) if entry_prem > 0 else 0.0

                with DB_LOCK:
                    con = duckdb.connect(DB_PATH, read_only=False)
                    try:
                        con.execute("""
                            UPDATE daily_options_journal
                            SET current_option_price = ?, pnl_pct = ?, status = ?, last_audited = ?
                            WHERE date_key = ?
                        """, [live_prem, pnl_pct, opt_status, now_str, opt["date_key"]])
                    finally:
                        con.close()

                cloud_update("options_journal", {"date_key": opt["date_key"]},
                             {"status": opt_status, "current_option_price": live_prem, "pnl_pct": pnl_pct, "last_audited": now_str},
                             "Audit Option Sync")
            except Exception:
                continue

# ==============================================================================
# 5. CONTINUOUS LEARNING: DECAYED MEMORY, CALIBRATION & GOVERNED RETRAINING
# ==============================================================================
def load_closed_trades_df() -> pd.DataFrame:
    with DB_LOCK:
        con = duckdb.connect(DB_PATH, read_only=True)
        try:
            return con.execute("""
                SELECT trade_id, ticker, date_str, exit_timestamp, last_audited, status, pnl_pct,
                       entry_price, capital_allocated, features_json
                FROM trade_journal WHERE status != 'ACTIVE'
            """).df()
        except Exception:
            return pd.DataFrame()
        finally:
            con.close()

def get_self_learning_adjustment(ticker: str, current_atr_pct: float, hist=None) -> dict:
    """Exponential half-life time decay based on actual exit date."""
    delta = 0.0
    reasons = []
    today_dt = london_now().date()

    if hist is None:
        hist = load_closed_trades_df()
    if hist is None or hist.empty:
        return {"delta": 0, "reason": "Neutral (Building memory)"}

    try:
        t_hist = hist[hist["ticker"] == ticker]
        if not t_hist.empty:
            decayed_loss = 0.0
            decayed_win = 0.0
            for _, r in t_hist.iterrows():
                try:
                    raw_exit = _first_valid(r.get("exit_timestamp"), r.get("last_audited"), r.get("date_str"))[:10]
                    dt = datetime.strptime(raw_exit, "%Y-%m-%d").date()
                    days_ago = max(0, (today_dt - dt).days)
                except Exception:
                    days_ago = 15
                weight = math.pow(0.5, days_ago / HALF_LIFE_DAYS)
                st_val = str(r["status"]).upper()
                if "LOSS" in st_val:
                    decayed_loss += 15.0 * weight
                elif "WIN" in st_val:
                    decayed_win += 5.0 * weight

            if decayed_loss >= 1.0:
                pen_i = int(round(decayed_loss))
                delta -= pen_i
                reasons.append(f"-{pen_i}% (Decayed stop-out memory)")
            if decayed_win >= 1.0:
                bst_i = int(round(decayed_win))
                delta += bst_i
                reasons.append(f"+{bst_i}% (Decayed target hit memory)")

        all_losses = hist[hist["status"].astype(str).str.contains("LOSS", na=False)]
        if not all_losses.empty:
            loss_atrs = []
            for f_str in all_losses["features_json"].dropna():
                f_obj = _parse_features(f_str)
                if "atr_pct" in f_obj:
                    try:
                        loss_atrs.append(float(f_obj["atr_pct"]))
                    except Exception:
                        pass
            if loss_atrs:
                p75_loss_atr = max(3.0, float(np.percentile(loss_atrs, 75)))
                if current_atr_pct >= p75_loss_atr:
                    delta -= 10
                    reasons.append(f"-10% (High-ATR regime >= {p75_loss_atr:.1f}%)")
    except Exception:
        pass

    return {"delta": int(round(delta)), "reason": " | ".join(reasons) if reasons else "No adverse historical match"}

def build_calibrator(closed_df: pd.DataFrame):
    """Maps raw model probability -> honest probability using realised outcomes.
    <MIN_CALIBRATION_SAMPLES: shrink toward realised base rate. Otherwise blend isotonic + shrinkage."""
    wins = losses = 0
    xs, ys = [], []
    if closed_df is not None and not closed_df.empty:
        for _, r in closed_df.iterrows():
            s = str(r["status"]).upper()
            if "WIN" in s:
                y = 1
            elif "LOSS" in s:
                y = 0
            else:
                continue
            wins += y
            losses += (1 - y)
            rp = _parse_features(r["features_json"]).get("raw_prob")
            if rp is not None:
                try:
                    xs.append(float(rp))
                    ys.append(y)
                except Exception:
                    pass
    decided = wins + losses
    prior = min(0.75, max(0.35, wins / decided)) if decided >= 10 else 0.5

    def shrink(p):
        return prior + CONF_SHRINK * (float(p) - 0.5)

    iso = None
    mode = f"Shrinkage to base rate {prior:.0%} (n={decided})"
    if len(xs) >= MIN_CALIBRATION_SAMPLES and len(set(ys)) == 2:
        try:
            from sklearn.isotonic import IsotonicRegression
            iso = IsotonicRegression(out_of_bounds="clip").fit(xs, ys)
            mode = f"Isotonic + shrinkage blend (n={len(xs)})"
        except Exception:
            iso = None

    def calibrate(p):
        base = shrink(p)
        if iso is not None:
            try:
                base = 0.5 * float(iso.predict([float(p)])[0]) + 0.5 * base
            except Exception:
                pass
        return min(0.95, max(0.05, base))

    return calibrate, {"mode": mode, "prior": prior, "decided": decided, "iso_samples": len(xs)}

def _load_retrain_state() -> dict:
    try:
        if os.path.exists(RETRAIN_STATE_PATH):
            with open(RETRAIN_STATE_PATH, "r", encoding="utf-8") as fh:
                return json.load(fh)
    except Exception:
        pass
    return {}

def _save_retrain_state(state: dict):
    try:
        os.makedirs(os.path.dirname(RETRAIN_STATE_PATH), exist_ok=True)
        with open(RETRAIN_STATE_PATH, "w", encoding="utf-8") as fh:
            json.dump(state, fh)
    except Exception:
        pass

def check_and_auto_retrain_model(feature_cols: list, closed_df: pd.DataFrame):
    """Champion/challenger: a challenger is trained on live trades, validated walk-forward on the most recent
    30% of trades, and only PROMOTED (to a separate file) if it beats the current champion. The original
    ensemble_ranker.joblib is never overwritten."""
    if closed_df is None or closed_df.empty:
        return
    records = []
    for _, r in closed_df.iterrows():
        s = str(r["status"]).upper()
        if "WIN" in s:
            y = 1
        elif "LOSS" in s:
            y = 0
        else:
            continue
        raw = _parse_features(r["features_json"]).get("raw_features", {})
        if not isinstance(raw, dict) or not all(c in raw for c in feature_cols):
            continue
        try:
            vec = [float(raw[c]) for c in feature_cols]
        except Exception:
            continue
        records.append((str(r.get("date_str") or ""), vec, y))

    n = len(records)
    if n < MIN_SETTLED_TO_RETRAIN:
        return
    state = _load_retrain_state()
    today_s = london_now().strftime("%Y-%m-%d")
    if state.get("last_attempt_date") == today_s:
        return
    if n - int(state.get("last_labeled_count", 0)) < RETRAIN_STEP_INTERVAL:
        return

    records.sort(key=lambda x: x[0])
    X_all = pd.DataFrame([r[1] for r in records], columns=feature_cols)
    y_all = np.array([r[2] for r in records])
    split = int(n * 0.7)
    X_tr, X_te, y_tr, y_te = X_all.iloc[:split], X_all.iloc[split:], y_all[:split], y_all[split:]
    state.update({"last_attempt_date": today_s, "last_labeled_count": n})

    if len(np.unique(y_tr)) < 2 or len(np.unique(y_te)) < 2:
        _save_retrain_state(state)
        return

    try:
        from lightgbm import LGBMClassifier
        from catboost import CatBoostClassifier
        from sklearn.metrics import log_loss, roc_auc_score

        def fit_pair(X, y):
            m1 = LGBMClassifier(n_estimators=250, learning_rate=0.03, max_depth=5, min_child_samples=10, random_state=42, verbose=-1)
            m1.fit(X, y)
            m2 = CatBoostClassifier(iterations=300, learning_rate=0.03, depth=5, verbose=0, random_seed=42)
            m2.fit(X, y)
            return m1, m2

        ch_l, ch_c = fit_pair(X_tr, y_tr)
        p_ch = np.clip(0.5 * ch_l.predict_proba(X_te)[:, 1] + 0.5 * ch_c.predict_proba(X_te)[:, 1], 1e-4, 1 - 1e-4)

        champ = get_ml_model()
        if champ is None:
            _save_retrain_state(state)
            return
        p_cm = np.clip(0.5 * champ["lgb"].predict_proba(X_te)[:, 1] + 0.5 * champ["catboost"].predict_proba(X_te)[:, 1], 1e-4, 1 - 1e-4)

        ll_ch, ll_cm = float(log_loss(y_te, p_ch)), float(log_loss(y_te, p_cm))
        auc_ch, auc_cm = float(roc_auc_score(y_te, p_ch)), float(roc_auc_score(y_te, p_cm))
        promote = (ll_ch < ll_cm * 0.98) and (auc_ch >= auc_cm)
        state.update({"last_ll_challenger": ll_ch, "last_ll_champion": ll_cm, "last_auc_challenger": auc_ch, "last_auc_champion": auc_cm})

        if promote:
            f_l, f_c = fit_pair(X_all, y_all)
            new_bundle = {"lgb": f_l, "catboost": f_c, "feature_cols": feature_cols,
                          "meta": {"trained_on": n, "promoted_at": london_now().strftime("%Y-%m-%d %H:%M:%S"),
                                   "holdout_logloss": ll_ch, "champion_logloss": ll_cm}}
            os.makedirs(os.path.dirname(LIVE_MODEL_PATH), exist_ok=True)
            joblib.dump(new_bundle, LIVE_MODEL_PATH)
            st.session_state["ml_model_bundle"] = new_bundle
            state["last_promoted"] = london_now().strftime("%Y-%m-%d %H:%M:%S")
            st.session_state["retrain_notice"] = "🧠 The AI updated itself: the new version tested better than the old one."
        else:
            st.session_state["retrain_notice"] = "🧪 The AI tested a new version but kept the old one (the new one wasn't better)."
        _save_retrain_state(state)
    except Exception as e:
        _save_retrain_state(state)
        record_db_error("Retrain ML", e)

# ==============================================================================
# 6. OPTIONS ENGINE (SIMULATED — BLUE-CHIP SELECTION UNDER £500 CAP)
# ==============================================================================
def generate_daily_options_alpha() -> dict:
    now_lon = london_now()
    today_str = now_lon.strftime('%Y-%m-%d')
    now_str = now_lon.strftime('%Y-%m-%d %H:%M:%S')

    with DB_LOCK:
        con = duckdb.connect(DB_PATH, read_only=True)
        try:
            df = con.execute("SELECT * FROM daily_options_journal WHERE date_key = ?", [today_str]).df()
            if not df.empty:
                return df.iloc[0].to_dict()
            active_underlyings = set(con.execute("SELECT DISTINCT share_name FROM daily_options_journal WHERE status = 'ACTIVE'").df()["share_name"].tolist())
        except Exception:
            active_underlyings = set()
        finally:
            con.close()

    candidates = ["BARC", "BP", "HSBA", "GLEN", "SHEL"]
    best_sig = None
    best_score = -999.0

    for sym in candidates:
        if sym in active_underlyings:        # avoid stacking correlated calls on the same underlying
            continue
        try:
            h = fetch_cached_history(f"{sym}.L", period="30d")
            if h.empty or len(h) < 15:
                continue
            spot = float(h["Close"].iloc[-1])
            ret_5d = float((spot - h["Close"].iloc[-5]) / h["Close"].iloc[-5])
            returns = np.log(h["Close"] / h["Close"].shift(1)).dropna()
            sigma = max(0.15, min(0.45, float(returns.std() * np.sqrt(252))))   # 30d REALISED vol

            days_to_expiry = 21
            lot_size = LOT_SIZES.get(sym, 1000)
            strike = round(spot * 1.02)
            q = OPTION_DIV_YIELD.get(sym, 0.03)
            entry_prem = option_ask_price(spot, strike, days_to_expiry, sigma, q)
            total_cap = round((entry_prem * lot_size) / 100.0, 2)

            if total_cap <= 500.0 and ret_5d > best_score:
                best_score = ret_5d
                score = round(min(92.0, max(68.0, 78.0 + (ret_5d * 150.0))), 1)   # heuristic momentum score, NOT a probability
                best_sig = {
                    "date_key": today_str, "timestamp": now_str, "share_name": sym,
                    "option_contract": f"{sym} {int(strike)}p Call", "strike_price": float(strike),
                    "expiry_days": days_to_expiry, "underlying_spot": float(spot), "lot_size": lot_size,
                    "entry_premium": float(entry_prem), "current_option_price": option_bid_price(spot, strike, days_to_expiry, sigma, q),
                    "target_premium": round(entry_prem * 1.60, 2), "stop_loss_premium": round(entry_prem * 0.50, 2),
                    "total_capital": float(total_cap), "ai_confidence": score, "implied_vol": round(sigma * 100, 1),
                    "status": "ACTIVE", "pnl_pct": 0.0, "last_audited": now_str
                }
        except Exception:
            continue

    if best_sig is None:
        return {}
    best_sig["pnl_pct"] = round(((best_sig["current_option_price"] - best_sig["entry_premium"]) / best_sig["entry_premium"]) * 100.0, 2)

    with DB_LOCK:
        con = duckdb.connect(DB_PATH, read_only=False)
        try:
            con.execute("""
                INSERT OR REPLACE INTO daily_options_journal 
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'ACTIVE', ?, ?)
            """, [best_sig[k] for k in ["date_key", "timestamp", "share_name", "option_contract", "strike_price", "expiry_days", "underlying_spot", "lot_size", "entry_premium", "current_option_price", "target_premium", "stop_loss_premium", "total_capital", "ai_confidence", "implied_vol", "pnl_pct", "last_audited"]])
        finally:
            con.close()

    if supabase:
        try:
            c_check = supabase.table("options_journal").select("date_key").eq("date_key", today_str).execute()
            if not c_check.data:
                supabase.table("options_journal").insert(best_sig).execute()
        except Exception as e:
            record_db_error("Insert Option", e)

    return best_sig

# ==============================================================================
# 7. LOGGING, DAILY QUOTA, SECTOR & PORTFOLIO-RISK GUARDRAILS
# ==============================================================================
def get_currently_active_tickers() -> set:
    with DB_LOCK:
        con = duckdb.connect(DB_PATH, read_only=True)
        try:
            df = con.execute("SELECT DISTINCT ticker FROM trade_journal WHERE status = 'ACTIVE'").df()
            return set(df["ticker"].tolist()) if not df.empty else set()
        except Exception:
            return set()
        finally:
            con.close()

def get_active_positions_df() -> pd.DataFrame:
    with DB_LOCK:
        con = duckdb.connect(DB_PATH, read_only=True)
        try:
            return con.execute("SELECT ticker, entry_price, stop_loss, capital_allocated, features_json FROM trade_journal WHERE status = 'ACTIVE'").df()
        except Exception:
            return pd.DataFrame()
        finally:
            con.close()

def get_active_sector_exposure() -> dict:
    counts = {}
    df = get_active_positions_df()
    if df.empty:
        return counts
    for _, r in df.iterrows():
        sec = get_trade_sector(r["ticker"], r["features_json"])
        counts[sec] = counts.get(sec, 0) + 1
    return counts

def get_portfolio_capacity() -> dict:
    df = get_active_positions_df()
    n_open = len(df)
    risk = 0.0
    for _, r in df.iterrows():
        try:
            e, s, cap = float(r["entry_price"]), float(r["stop_loss"]), float(r["capital_allocated"])
            if e > 0 and s < e:
                risk += cap * (e - s) / e
        except Exception:
            pass
    ok = (n_open < MAX_OPEN_POSITIONS) and (risk < MAX_OPEN_RISK_GBP)
    reason = ""
    if n_open >= MAX_OPEN_POSITIONS:
        reason = f"You already have the maximum number of open trades ({n_open}/{MAX_OPEN_POSITIONS})"
    elif risk >= MAX_OPEN_RISK_GBP:
        reason = f"Too much money is at risk already (£{risk:.0f} of £{MAX_OPEN_RISK_GBP:.0f})"
    return {"open": n_open, "risk_gbp": round(risk, 2), "ok": ok, "reason": reason}

def get_todays_logged_equities() -> pd.DataFrame:
    today_str = london_now().strftime('%Y-%m-%d')
    with DB_LOCK:
        con = duckdb.connect(DB_PATH, read_only=True)
        try:
            return con.execute("SELECT * FROM trade_journal WHERE date_str = ? ORDER BY timestamp ASC, ticker ASC", [today_str]).df()
        except Exception:
            return pd.DataFrame()
        finally:
            con.close()

def get_recent_cooldown_tickers() -> set:
    """Anti-churn: excludes tickers closed within COOLDOWN_CALENDAR_DAYS based on actual exit date."""
    today_dt = london_now().date()
    cooldown = set()
    with DB_LOCK:
        con = duckdb.connect(DB_PATH, read_only=True)
        try:
            recent_closed = con.execute(
                "SELECT ticker, exit_timestamp, last_audited, date_str FROM trade_journal WHERE status != 'ACTIVE'"
            ).df()
            for _, r in recent_closed.iterrows():
                c_dt = _to_date(_first_valid(r.get("exit_timestamp"), r.get("last_audited"), r.get("date_str")))
                if c_dt and (today_dt - c_dt).days <= COOLDOWN_CALENDAR_DAYS:
                    cooldown.add(str(r["ticker"]).strip())
        except Exception:
            pass
        finally:
            con.close()
    return cooldown

def log_equity_signal_safely(sig: dict, enforce_sector_cap: bool = True) -> bool:
    now = london_now()
    today_str = now.strftime('%Y-%m-%d')
    now_str = now.strftime('%Y-%m-%d %H:%M:%S')
    ticker = str(sig.get('Ticker', '')).strip()

    def fail(msg):
        st.session_state["last_log_reason"] = msg
        return False

    if not is_valid_lse_ticker(ticker):
        return fail("Invalid ticker")
    if not is_entry_window_open():
        return fail("It is outside buying hours (08:30–16:00 London, Mon–Fri)")
    if not sig.get("Loggable", True):
        return fail("No live price from today yet")
    if is_market_stress(get_cross_asset_macro_regime()):
        return fail("Buying is paused because the UK market fell sharply this week")
    cap_state = get_portfolio_capacity()
    if not cap_state["ok"]:
        return fail(cap_state["reason"])

    sector = str(sig.get('Sector') or get_ticker_sector(ticker))
    if enforce_sector_cap:
        sec_counts = get_active_sector_exposure()
        if sec_counts.get(sector, 0) >= MAX_ACTIVE_PER_SECTOR:
            return fail(f"You already hold {MAX_ACTIVE_PER_SECTOR} trades in {sector}")

    f_dict = sig.get('FeaturesDict', {})
    f_json_str = json.dumps(f_dict)
    friction = get_trade_friction_pct(ticker, f_json_str)
    init_net_pnl = -friction
    logged_local = False

    with DB_LOCK:
        con = duckdb.connect(DB_PATH, read_only=False)
        try:
            today_count_df = con.execute("SELECT COUNT(*) AS cnt FROM trade_journal WHERE date_str = ?", [today_str]).df()
            if not today_count_df.empty and int(today_count_df["cnt"].iloc[0]) >= MAX_DAILY_EQUITY_TRADES:
                return fail("All of today's picks are already used")

            existing = con.execute("""
                SELECT trade_id FROM trade_journal 
                WHERE ticker = ? AND (status = 'ACTIVE' OR date_str = ?)
            """, [ticker, today_str]).df()

            if existing.empty:
                trade_id = f"{ticker}_{today_str}"
                shares_num = int(str(sig['Recommended Shares']).split()[0])
                con.execute("""
                    INSERT INTO trade_journal 
                    (trade_id, timestamp, date_str, ticker, asset_type, entry_price, target_price,
                     stop_loss, shares, capital_allocated, status, latest_price, pnl_pct, exit_price,
                     exit_timestamp, last_audited, features_json)
                    VALUES (?, ?, ?, ?, 'EQUITY', ?, ?, ?, ?, ?, 'ACTIVE', ?, ?, 0.0, NULL, ?, ?)
                """, [
                    trade_id, now_str, today_str, ticker, float(sig['Price (p)']),
                    float(sig['Target (p)']), float(sig['Stop Loss (p)']),
                    shares_num, float(sig['Capital']), float(sig['Price (p)']), init_net_pnl, now_str, f_json_str
                ])
                logged_local = True
            else:
                return fail("You already hold this stock")
        finally:
            con.close()

    if supabase and logged_local:
        try:
            today_cloud = supabase.table("predictions").select("id").eq("predicted_date", today_str).execute()
            if today_cloud.data and len(today_cloud.data) >= MAX_DAILY_EQUITY_TRADES:
                return logged_local

            c_check = supabase.table("predictions").select("id, status, predicted_date").eq("ticker", ticker).execute()
            already_open_or_today = False
            if c_check.data:
                for row in c_check.data:
                    if str(row.get("status", "")).upper() == "ACTIVE" or str(row.get("predicted_date", "")) == today_str:
                        already_open_or_today = True
                        break

            if not already_open_or_today:
                shares_num = int(str(sig['Recommended Shares']).split()[0])
                supabase.table("predictions").insert({
                    "predicted_date": today_str,
                    "ticker": ticker,
                    "company_name": f"{ticker} ({sector})",
                    "entry_price": float(sig['Price (p)']),
                    "target_price": float(sig['Target (p)']),
                    "stop_loss": float(sig['Stop Loss (p)']),
                    "position_gbp": float(sig['Capital']),
                    "shares_qty": shares_num,
                    "profit_goal": float(sig['ReturnNum']),
                    "confidence": int(float(str(sig['AI Win Confidence']).replace("%", ""))),
                    "hold_days": 5,
                    "status": "ACTIVE",
                    "latest_price": float(sig['Price (p)']),
                    "pnl_pct": init_net_pnl,
                    "news_status": sig.get('RNS', 'Clean'),
                    "rns_headline": str(sig.get('RNSHeadline', 'Active LSE Quant Signal'))[:120],
                    "features_json": f_dict,
                    "last_checked": now_str
                }).execute()
        except Exception as e:
            record_db_error("Insert Equity", e)

    return logged_local

# ==============================================================================
# 8. UNIVERSE & DUAL-ENSEMBLE ML ENGINE
# ==============================================================================
def get_ml_model():
    if "ml_model_bundle" in st.session_state:
        return st.session_state["ml_model_bundle"]
    for path in (LIVE_MODEL_PATH, MODEL_PATH):
        if os.path.exists(path):
            try:
                bundle = joblib.load(path)
                st.session_state["ml_model_bundle"] = bundle
                st.session_state["ml_model_source"] = "Promoted live model" if path == LIVE_MODEL_PATH else "Original champion model"
                return bundle
            except Exception:
                continue
    return None

@st.cache_data(ttl=3600, show_spinner=False)
def get_live_lse_universe() -> list:
    raw_tickers = []
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
    for url in ("https://en.wikipedia.org/wiki/FTSE_100_Index", "https://en.wikipedia.org/wiki/FTSE_250_Index"):
        try:
            resp = requests.get(url, headers=headers, timeout=4)
            if resp.status_code == 200:
                df_idx = pd.read_html(resp.text, attrs={'id': 'constituents'})[0]
                raw_tickers.extend([f"{str(t).strip().replace('.', '-')}.L" for t in df_idx['Ticker'].dropna()])
        except Exception:
            pass

    fallback_pool = [
        "SHEL.L", "BP.L", "PTAL.L", "RKH.L", "ITM.L",
        "RIO.L", "GLEN.L", "LTHM.L", "CMCL.L", "CAML.L", "SAV.L", "KP2.L",
        "HSBA.L", "BARC.L", "LSEG.L", "BUR.L", "LLOY.L", "NWG.L", "AV.L", "PRU.L", "LGEN.L",
        "AZN.L", "GSK.L", "HLN.L", "SN.L", "HIK.L",
        "ULVR.L", "BATS.L", "FEVR.L", "JET2.L", "NFG.L", "TSCO.L", "SBRY.L", "MKS.L", "NXT.L", "DGE.L", "RKT.L",
        "NG.L", "SSE.L", "CNA.L", "VOD.L", "BT-A.L",
        "DOTD.L", "BIG.L", "SEE.L", "AOM.L", "YOU.L", "CER.L", "RWS.L", "KGH.L", "AUTO.L", "SGE.L",
        "CHRT.L", "CNC.L", "BA.L", "RR.L", "QQ.L",
        "CRW.L", "BRCK.L", "MIDW.L", "VIC.L", "SRC.L", "JHD.L", "REL.L", "EXPN.L", "AHT.L"
    ]
    combined = set(raw_tickers + fallback_pool)
    return sorted([t for t in combined if is_valid_lse_ticker(t)])

def run_predictions():
    ml_bundle = get_ml_model()
    if ml_bundle is None:
        st.session_state["scan_stats"] = {"error": "ML model file not found"}
        return pd.DataFrame(), False

    feature_cols = ml_bundle['feature_cols']
    closed_df = load_closed_trades_df()
    check_and_auto_retrain_model(feature_cols, closed_df)
    ml_bundle = get_ml_model() or ml_bundle
    feature_cols = ml_bundle['feature_cols']
    calibrate, cal_info = build_calibrator(closed_df)
    st.session_state["cal_info"] = cal_info

    now_lon = london_now()
    today_date = now_lon.date()
    market_open = is_lse_market_open()
    window_open = is_entry_window_open()

    active_held = get_currently_active_tickers()
    active_sectors = get_active_sector_exposure()
    saturated_sectors = {sec for sec, cnt in active_sectors.items() if cnt >= MAX_ACTIVE_PER_SECTOR}
    cooldown_tickers = get_recent_cooldown_tickers()
    macro_regime = get_cross_asset_macro_regime()
    entries_paused = is_market_stress(macro_regime)
    capacity = get_portfolio_capacity()

    stats = {"scanned": 0, "no_data": 0, "illiquid": 0, "rns_blocked": 0, "sector_blocked": 0, "model_errors": 0,
             "paused": entries_paused, "capacity": capacity, "window_open": window_open, "logged": 0}

    full_universe = [
        t for t in get_live_lse_universe()
        if t not in active_held and t not in cooldown_tickers and peek_sector(t) not in saturated_sectors
    ]
    if not full_universe:
        st.session_state["scan_stats"] = stats
        return pd.DataFrame(), False

    scan_chunk = random.sample(full_universe, min(35, len(full_universe)))

    results = []
    lgb_model = ml_bundle['lgb']
    cb_model = ml_bundle['catboost']

    prog = st.progress(0, text=f"Downloading price history for {len(scan_chunk)} UK equities in parallel...")
    prefetch_histories(scan_chunk, "120d")

    for i, ticker in enumerate(scan_chunk):
        try:
            df = get_history_fast(ticker, "120d")
            if df.empty or engineer_features is None:
                stats["no_data"] += 1
                continue

            dates = _bar_dates(df)
            live_bar_today = (dates[-1] == today_date)
            partial_bar = live_bar_today and market_open
            live_price = float(df["Close"].iloc[-1])
            df_model = df.iloc[:-1] if partial_bar else df          # model sees COMPLETED bars only (train/inference parity)
            if len(df_model) < 55:
                stats["no_data"] += 1
                continue

            recent = df_model.tail(20)
            avg_turnover = float((recent["Close"] * recent["Volume"]).mean() / 100.0)
            if (not math.isfinite(avg_turnover)) or live_price < MIN_PRICE_P or avg_turnover < MIN_AVG_TURNOVER_GBP:
                stats["illiquid"] += 1
                continue

            sector = get_ticker_sector(ticker)
            if sector in saturated_sectors:
                stats["sector_blocked"] += 1
                continue

            d2 = df_model.copy()
            d2.reset_index(inplace=True)
            feats = engineer_features(d2)
            if feats.empty:
                stats["no_data"] += 1
                continue
            latest = feats.iloc[-1:].copy()

            rns_data = evaluate_rns_nlp_sentiment(ticker)
            if "Dilution" in rns_data['status']:
                stats["rns_blocked"] += 1
                continue

            p1 = float(lgb_model.predict_proba(latest[feature_cols])[:, 1][0])
            p2 = float(cb_model.predict_proba(latest[feature_cols])[:, 1][0])
            blended = max(0.01, min(0.99, (0.5 * p1 + 0.5 * p2) + rns_data['delta']))

            atr = float(latest['atr_14'].values[0])
            atr_pct = round((atr / live_price) * 100.0, 2) if live_price > 0 else 0.0

            target = live_price + (1.8 * atr)
            stop = live_price - (1.2 * atr)
            gross_return_pct = ((target - live_price) / live_price) * 100.0 if live_price > 0 else 0.0

            spread_pct = estimate_spread_pct(ticker, avg_turnover)
            commission = get_commission_gbp()
            friction_pct = compute_friction_pct(ticker, spread_pct, commission)
            net_return_pct = round(gross_return_pct - friction_pct, 1)
            tax_regime = "AIM (0% SDRT)" if is_aim_exempt(ticker) else "Main (0.5% SDRT)"

            base_confidence = int(min(95, max(25, round(calibrate(blended) * 100.0))))
            learner = get_self_learning_adjustment(ticker, atr_pct, closed_df)
            macro_delta, macro_note = compute_macro_lead_lag_adjustment(ticker, sector, macro_regime)

            total_delta = learner["delta"] + macro_delta
            final_confidence = int(min(98, max(25, base_confidence + total_delta)))

            shares = max(1, int((POSITION_SIZE_GBP * 100) / live_price)) if live_price > 0 else 1
            is_qualified = (final_confidence >= MIN_QUALIFY_CONFIDENCE) and (net_return_pct >= MIN_NET_RETURN_PCT)

            raw_feature_map = {c: float(latest[c].values[0]) for c in feature_cols if c in latest.columns}
            feature_snapshot = {
                "raw_features": raw_feature_map,
                "atr_pct": atr_pct,
                "raw_prob": round(blended, 4),
                "base_ml_conf": base_confidence,
                "learner_delta": learner["delta"],
                "macro_delta": macro_delta,
                "sector": sector,
                "spread_pct": spread_pct,
                "commission_gbp": commission,
                "friction_pct": round(get_sdrt_pct(ticker) + spread_pct, 3),   # SDRT + spread (commission stored separately)
                "avg_turnover_gbp": round(avg_turnover, 0),
                "tax_regime": tax_regime,
                "nlp_tags": rns_data["nlp_tags"],
                "entry_bar": str(dates[-1]),
                "cal_mode": cal_info["mode"],
            }

            stats["scanned"] += 1
            results.append({
                "Ticker": ticker, "Company": ticker, "Sector": sector, "Tax Regime": tax_regime,
                "Friction": f"-{friction_pct:.2f}%", "Price (p)": round(live_price, 2),
                "Expected Return": f"+{net_return_pct}% Net", "ReturnNum": net_return_pct,
                "Est. Time to Target": "3-7 Days", "Recommended Shares": f"{shares} shares",
                "Total Cost (£)": "£500", "Capital": POSITION_SIZE_GBP,
                "Target (p)": round(target, 2), "Stop Loss (p)": round(stop, 2),
                "Base ML": f"{base_confidence}%", "Learner Delta": f"{total_delta:+d}%",
                "Learner Note": f"{learner['reason']} | Macro: {macro_note}",
                "AI Win Confidence": f"{final_confidence}%", "Adjusted Score": final_confidence,
                "RNS": rns_data['status'], "RNSHeadline": rns_data['headline'],
                "Liquidity (£m/day)": round(avg_turnover / 1e6, 1),
                "Loggable": bool(live_bar_today and window_open),
                "FeaturesDict": feature_snapshot, "Qualified": is_qualified
            })
        except Exception:
            stats["model_errors"] += 1
        finally:
            prog.progress((i + 1) / len(scan_chunk))

    prog.empty()
    if not results:
        st.session_state["scan_stats"] = stats
        return pd.DataFrame(), False

    df_out = pd.DataFrame(results).sort_values(by=["Qualified", "Adjusted Score", "ReturnNum"], ascending=[False, False, False]).reset_index(drop=True)
    qualified_only = df_out[df_out["Qualified"] == True].copy().reset_index(drop=True)

    if not qualified_only.empty and not entries_paused:
        for _, sig in qualified_only.iterrows():
            if not bool(sig.get("Loggable", False)):
                continue
            if len(get_todays_logged_equities()) >= MAX_DAILY_EQUITY_TRADES:
                break
            if log_equity_signal_safely(sig.to_dict(), enforce_sector_cap=True):
                stats["logged"] += 1

    st.session_state["scan_stats"] = stats
    return qualified_only, not qualified_only.empty

# ==============================================================================
# 9. PERFORMANCE ANALYTICS (HONEST NUMBERS)
# ==============================================================================
def prep_perf_frame(df: pd.DataFrame, cap_col: str, is_option: bool = False) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    d = df[df["status"].astype(str).str.upper() != "ACTIVE"].copy()
    if d.empty:
        return d
    d["cap"] = pd.to_numeric(d[cap_col], errors="coerce").fillna(POSITION_SIZE_GBP)
    d["pnl_pct"] = pd.to_numeric(d["pnl_pct"], errors="coerce").fillna(0.0)
    d["pnl_gbp"] = d["cap"] * d["pnl_pct"] / 100.0
    d["outcome"] = d["status"].apply(classify_outcome)
    if is_option:
        d["entry_date"] = d["date_key"].apply(_to_date)
        d["exit_date"] = d["last_audited"].apply(_to_date)
    else:
        d["entry_date"] = d["date_str"].apply(_to_date)
        d["exit_date"] = d.apply(lambda r: _to_date(_first_valid(r.get("exit_timestamp"), r.get("last_audited"), r.get("date_str"))), axis=1)
    d["exit_date"] = d.apply(lambda r: r["exit_date"] or r["entry_date"], axis=1)
    return d.sort_values(["exit_date", "entry_date"], kind="stable").reset_index(drop=True)

def compute_performance(d: pd.DataFrame) -> dict:
    n = len(d)
    if n == 0:
        return {"n": 0}
    wins = int((d["outcome"] == "WIN").sum())
    losses = int((d["outcome"] == "LOSS").sum())
    be = int((d["outcome"] == "BE").sum())
    exp = int((d["outcome"] == "EXPIRED").sum())
    pos = d.loc[d["pnl_gbp"] > 0, "pnl_gbp"]
    neg = d.loc[d["pnl_gbp"] < 0, "pnl_gbp"]
    profitable = int((d["pnl_gbp"] > 0).sum())
    cum = d["pnl_gbp"].cumsum().tolist()
    curve = np.array([0.0] + cum)
    dd = curve - np.maximum.accumulate(curve)
    lo, hi = wilson_interval(wins, n)
    return {
        "n": n, "wins": wins, "losses": losses, "be": be, "expired": exp,
        "target_hit_rate": wins / n * 100.0,
        "ci_low": lo * 100.0, "ci_high": hi * 100.0,
        "profitable_rate": profitable / n * 100.0,
        "avg_win_gbp": float(pos.mean()) if len(pos) else 0.0,
        "avg_loss_gbp": float(neg.mean()) if len(neg) else 0.0,
        "expectancy_pct": float(d["pnl_pct"].mean()),
        "expectancy_gbp": float(d["pnl_gbp"].mean()),
        "profit_factor": (float(pos.sum()) / abs(float(neg.sum()))) if len(neg) and neg.sum() != 0 else None,
        "max_dd_gbp": float(dd.min()),
        "net_gbp": float(d["pnl_gbp"].sum()),
        "cum": cum,
    }

# ==============================================================================
# 10. UI THEME & COMPONENTS (PLAIN ENGLISH)
# ==============================================================================
CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
:root{--a1:#6366f1;--a2:#06b6d4;--good:#16a34a;--bad:#dc2626;--warn:#d97706;--line:rgba(128,128,128,.28);--soft:rgba(128,128,128,.08);}
html,body,.stApp{font-family:'Inter',system-ui,-apple-system,sans-serif;}
.block-container{padding-top:1.1rem;max-width:1400px;}
footer{visibility:hidden;}
@keyframes drift{0%{background-position:0% 50%}50%{background-position:100% 50%}100%{background-position:0% 50%}}
.hero{border:1px solid var(--line);border-radius:22px;padding:24px 28px;background:linear-gradient(120deg,rgba(99,102,241,.28),rgba(6,182,212,.18),rgba(99,102,241,.20));background-size:200% 200%;animation:drift 14s ease infinite;margin-bottom:14px;}
.hero h1{margin:0;font-size:2rem;font-weight:800;letter-spacing:-.03em;background:linear-gradient(90deg,#6366f1,#06b6d4);-webkit-background-clip:text;-webkit-text-fill-color:transparent;}
.hero .sub{opacity:.85;font-size:.95rem;margin-top:6px;}
.chips{display:flex;flex-wrap:wrap;gap:8px;margin-top:14px;}
.chip{border:1px solid var(--line);background:var(--soft);padding:4px 12px;border-radius:999px;font-size:.78rem;font-weight:600;white-space:nowrap;}
.chip.up{color:var(--good);border-color:rgba(22,163,74,.5);}
.chip.down{color:var(--bad);border-color:rgba(220,38,38,.5);}
.sec{display:flex;align-items:center;gap:10px;margin:24px 0 10px;}
.sec .bar{width:5px;height:28px;border-radius:4px;background:linear-gradient(var(--a1),var(--a2));}
.sec h3{margin:0;font-size:1.3rem;font-weight:800;letter-spacing:-.01em;}
.sec small{opacity:.65;margin-left:8px;font-size:.82rem;font-weight:500;}
.score{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:14px;margin:6px 0 14px;}
.sc{position:relative;border:1px solid var(--line);border-radius:18px;padding:18px 20px;background:var(--soft);overflow:hidden;transition:transform .18s ease,box-shadow .18s ease;}
.sc:hover{transform:translateY(-3px);box-shadow:0 12px 30px rgba(0,0,0,.16);}
.sc:before{content:"";position:absolute;left:0;top:0;right:0;height:4px;background:linear-gradient(90deg,var(--a1),var(--a2));}
.sc.good:before{background:linear-gradient(90deg,#16a34a,#4ade80);}
.sc.bad:before{background:linear-gradient(90deg,#dc2626,#f87171);}
.sc.warn:before{background:linear-gradient(90deg,#d97706,#fbbf24);}
.sc .lbl{font-size:.72rem;font-weight:700;letter-spacing:.07em;text-transform:uppercase;opacity:.65;}
.sc .val{font-size:2.3rem;font-weight:800;letter-spacing:-.03em;line-height:1.15;margin-top:6px;}
.sc .val.pos{color:var(--good);}
.sc .val.neg{color:var(--bad);}
.sc .sub{font-size:.8rem;opacity:.72;margin-top:4px;}
.callout{border-radius:16px;padding:14px 18px;margin:8px 0 14px;border:1px solid var(--line);font-size:.95rem;display:flex;gap:12px;align-items:flex-start;}
.callout .ic{font-size:1.4rem;line-height:1.25;}
.callout.good{background:rgba(22,163,74,.12);border-color:rgba(22,163,74,.45);}
.callout.bad{background:rgba(220,38,38,.12);border-color:rgba(220,38,38,.45);}
.callout.warn{background:rgba(217,119,6,.13);border-color:rgba(217,119,6,.45);}
.callout.info{background:rgba(99,102,241,.12);border-color:rgba(99,102,241,.45);}
.tcard,.mini,.stat,.lesson{border:1px solid var(--line);border-radius:16px;padding:16px 18px;background:var(--soft);margin-bottom:12px;}
.tcard,.mini{transition:transform .15s ease,box-shadow .15s ease;}
.tcard:hover,.mini:hover{transform:translateY(-3px);box-shadow:0 10px 28px rgba(0,0,0,.18);}
.mini{padding:12px 14px;}
.tcard-top{display:flex;justify-content:flex-start;gap:6px;flex-wrap:wrap;margin-bottom:8px;}
.tag{font-size:.68rem;font-weight:700;letter-spacing:.04em;text-transform:uppercase;padding:3px 9px;border-radius:999px;border:1px solid var(--line);}
.tag.good{color:var(--good);border-color:rgba(22,163,74,.5);}
.tag.bad{color:var(--bad);border-color:rgba(220,38,38,.5);}
.tag.warn{color:var(--warn);border-color:rgba(217,119,6,.5);}
.tag.info{color:var(--a1);border-color:rgba(99,102,241,.5);}
.row{display:flex;justify-content:space-between;align-items:center;gap:10px;}
.tk{font-size:1.5rem;font-weight:800;letter-spacing:-.01em;}
.mtk{font-size:1.05rem;font-weight:800;}
.pnl{font-size:1.8rem;font-weight:800;line-height:1.1;text-align:right;}
.mp{font-size:1.1rem;font-weight:800;}
.pos{color:var(--good);}
.neg{color:var(--bad);}
.sub2{font-size:.78rem;opacity:.72;}
.rbar{position:relative;height:10px;border-radius:999px;background:linear-gradient(90deg,rgba(220,38,38,.55),rgba(217,119,6,.45),rgba(22,163,74,.6));margin:18px 0 4px;}
.rbar.c{margin:12px 0 4px;height:8px;}
.rdot{position:absolute;top:-4px;width:18px;height:18px;border-radius:50%;background:#fff;border:3px solid var(--a1);transform:translateX(-50%);box-shadow:0 2px 8px rgba(0,0,0,.35);}
.rbar.c .rdot{top:-4px;width:16px;height:16px;}
.rtick{position:absolute;top:-3px;width:2px;height:16px;background:rgba(128,128,128,.9);transform:translateX(-50%);}
.rlab{display:flex;justify-content:space-between;font-size:.74rem;opacity:.75;gap:6px;}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:6px 14px;margin-top:10px;font-size:.82rem;}
.grid2 b{opacity:.6;font-weight:600;margin-right:4px;}
.ringwrap{position:relative;width:64px;height:64px;flex:none;}
.ring{width:64px;height:64px;border-radius:50%;-webkit-mask:radial-gradient(farthest-side,transparent 70%,#000 72%);mask:radial-gradient(farthest-side,transparent 70%,#000 72%);}
.ringnum{position:absolute;inset:0;display:grid;place-items:center;font-weight:800;font-size:.95rem;}
.note{font-size:.75rem;opacity:.75;margin-top:8px;}
.stat .l{font-size:.72rem;font-weight:700;letter-spacing:.06em;text-transform:uppercase;opacity:.6;}
.stat .v{font-size:1.25rem;font-weight:800;margin-top:4px;}
.hbars{display:flex;flex-direction:column;gap:10px;margin:6px 0 10px;}
.hb{display:grid;grid-template-columns:150px 1fr 110px;gap:10px;align-items:center;font-size:.86rem;}
.hb .t{height:12px;border-radius:999px;background:rgba(128,128,128,.18);overflow:hidden;}
.hb .f{height:100%;border-radius:999px;}
.hb .f.pos{background:linear-gradient(90deg,#16a34a,#4ade80);}
.hb .f.neg{background:linear-gradient(90deg,#dc2626,#f87171);}
.meter{height:16px;border-radius:999px;background:rgba(128,128,128,.2);overflow:hidden;margin:8px 0 4px;}
.meter>div{height:100%;border-radius:999px;background:linear-gradient(90deg,var(--a1),var(--a2));}
.lesson{padding:12px 16px;margin-bottom:8px;font-size:.9rem;}
.disc{font-size:.76rem;opacity:.6;text-align:center;margin:30px 0 6px;}
[data-testid="stMetric"]{border:1px solid var(--line);border-radius:14px;padding:12px 14px;background:var(--soft);}
button[data-baseweb="tab"]{font-weight:700;font-size:1rem;}
.stButton>button{border-radius:12px;font-weight:600;}
@media(max-width:700px){.hb{grid-template-columns:96px 1fr 80px;}.sc .val{font-size:1.8rem;}.hero h1{font-size:1.5rem;}}
"""

def inject_css():
    st.markdown(f"<style>{_collapse(CSS)}</style>", unsafe_allow_html=True)

def friendly_status(status, be_locked: bool = False) -> str:
    o = classify_outcome(status)
    if o == "ACTIVE":
        return "🟢 Open (profit protected)" if be_locked else "🟢 Open"
    return {"WIN": "✅ Won — hit the goal", "LOSS": "❌ Lost — safety exit hit",
            "BE": "🛡️ Saved — sold with no loss", "EXPIRED": "⏱️ Closed after 7 days"}.get(o, str(status))

def outcome_style(status) -> tuple:
    return {"WIN": ("Won", "good"), "LOSS": ("Lost", "bad"), "BE": ("Saved — no loss", "warn"),
            "EXPIRED": ("Time's up", "warn"), "ACTIVE": ("Open", "info")}.get(classify_outcome(status), (str(status)[:14], "info"))

def sec_title(icon: str, title: str, sub: str = ""):
    small = f"<small>{esc(sub)}</small>" if sub else ""
    st.markdown(_collapse(f'<div class="sec"><div class="bar"></div><h3>{esc(icon)} {esc(title)}{small}</h3></div>'), unsafe_allow_html=True)

def callout(kind: str, icon: str, text_html: str):
    """text_html must already be escaped by the caller where it contains dynamic text."""
    st.markdown(_collapse(f'<div class="callout {kind}"><div class="ic">{icon}</div><div>{text_html}</div></div>'), unsafe_allow_html=True)

def score_cards(items: list):
    cards = []
    for label, value, sub, kind, vcls in items:
        cards.append(f'<div class="sc {kind}"><div class="lbl">{esc(label)}</div><div class="val {vcls}">{esc(value)}</div><div class="sub">{esc(sub)}</div></div>')
    st.markdown(_collapse('<div class="score">' + "".join(cards) + '</div>'), unsafe_allow_html=True)

def stat_card(label: str, value: str, sub: str = ""):
    st.markdown(_collapse(f'<div class="stat"><div class="l">{esc(label)}</div><div class="v">{esc(value)}</div><div class="sub2">{esc(sub)}</div></div>'), unsafe_allow_html=True)

def hbars_html(rows: list) -> str:
    """rows: (label, numeric value, text shown on the right)."""
    if not rows:
        return ""
    m = max(abs(v) for _, v, _ in rows) or 1.0
    parts = []
    for label, v, txt in rows:
        w = max(4.0, abs(v) / m * 100.0)
        cls = "pos" if v >= 0 else "neg"
        parts.append(f'<div class="hb"><div>{esc(label)}</div><div class="t"><div class="f {cls}" style="width:{w:.0f}%"></div></div><div><b>{esc(txt)}</b></div></div>')
    return _collapse('<div class="hbars">' + "".join(parts) + '</div>')

def price_bar_html(entry: float, target: float, stop: float, live: float, compact: bool = False) -> str:
    span = target - stop
    if span <= 0:
        return ""
    def pos(v):
        return max(0.0, min(100.0, (v - stop) / span * 100.0))
    bar = f'<div class="rbar{" c" if compact else ""}"><div class="rtick" style="left:{pos(entry):.1f}%"></div><div class="rdot" style="left:{pos(live):.1f}%"></div></div>'
    if compact:
        return _collapse(bar)
    return _collapse(bar + f'<div class="rlab"><span>🛑 Exit if it falls to {stop:.2f}p</span><span>Bought at {entry:.2f}p</span><span>🎯 Goal {target:.2f}p</span></div>')

def trade_card_html(slot, ticker, sector, status, entry, target, stop, live, pnl_pct, shares, capital,
                    conf, base_c, adj, tax_tag, friction, audited, be_locked) -> str:
    s_text, s_kind = outcome_style(status)
    cls = "pos" if pnl_pct >= 0 else "neg"
    lock_tag = '<span class="tag warn">🛡️ Profit protected</span>' if be_locked and classify_outcome(status) == "ACTIVE" else ""
    return _collapse(f"""
    <div class="tcard">
      <div class="tcard-top"><span class="tag info">Pick {esc(slot)}</span><span class="tag">{esc(sector)}</span><span class="tag {s_kind}">{esc(s_text)}</span>{lock_tag}</div>
      <div class="row"><div><div class="tk">{esc(ticker)}</div><div class="sub2">AI was {esc(conf)}% confident when buying</div></div>
      <div><div class="pnl {cls}">{pnl_pct:+.2f}%</div><div class="sub2" style="text-align:right">after fees &amp; tax</div></div></div>
      {price_bar_html(entry, target, stop, live)}
      <div class="grid2">
        <div><b>Price now</b>{live:.2f}p</div><div><b>Money in</b>£{capital:.0f} ({int(shares)} shares)</div>
        <div><b>UK stamp duty</b>{esc(tax_tag)}</div><div><b>Total costs</b>-{friction:.2f}%</div>
      </div>
      <div class="note">🕒 Last checked {esc(str(audited)[:16])} · prices can be ~15 min behind</div>
    </div>
    """)

def mini_trade_html(ticker, sector, entry, target, stop, live, pnl_pct, locked) -> str:
    cls = "pos" if pnl_pct >= 0 else "neg"
    lock = " 🛡️" if locked else ""
    return _collapse(f"""
    <div class="mini">
      <div class="row"><span class="mtk">{esc(ticker)}{lock}</span><span class="mp {cls}">{pnl_pct:+.2f}%</span></div>
      {price_bar_html(entry, target, stop, live, compact=True)}
      <div class="sub2">{esc(sector)} · now {live:.2f}p</div>
    </div>
    """)

def signal_card_html(rank, row, loggable) -> str:
    conf = int(row["Adjusted Score"])
    colour = "#16a34a" if conf >= 75 else ("#6366f1" if conf >= 65 else "#d97706")
    log_tag = '<span class="tag good">Ready to buy</span>' if loggable else '<span class="tag warn">Watch only (market closed)</span>'
    return _collapse(f"""
    <div class="tcard">
      <div class="tcard-top"><span class="tag info">🔥 Pick {rank}</span><span class="tag">{esc(row['Sector'])}</span>{log_tag}</div>
      <div class="row"><div><div class="tk">{esc(row['Ticker'])}</div><div class="sub2">AI confidence</div></div>
      <div class="ringwrap"><div class="ring" style="background:conic-gradient({colour} {conf}%,rgba(128,128,128,.25) 0)"></div><div class="ringnum">{conf}%</div></div></div>
      <div class="pnl pos" style="text-align:left;margin-top:8px">{esc(str(row['Expected Return']).replace(' Net', ''))}</div><div class="sub2">possible profit if it reaches the goal (after fees &amp; tax)</div>
      {price_bar_html(float(row['Price (p)']), float(row['Target (p)']), float(row['Stop Loss (p)']), float(row['Price (p)']))}
      <div class="grid2">
        <div><b>Shares</b>{esc(row['Recommended Shares'])}</div><div><b>Money in</b>£500</div>
        <div><b>Easy to trade</b>£{row.get('Liquidity (£m/day)', 0)}m a day</div><div><b>Costs</b>{esc(row['Friction'])}</div>
      </div>
      <div class="note">🧠 {esc(row['Learner Note'])}</div>
    </div>
    """)

# ---------- charts ----------
def show_fig(fig):
    if not PLOTLY_OK or fig is None:
        return
    try:
        st.plotly_chart(fig, width="stretch")
    except TypeError:
        st.plotly_chart(fig, use_container_width=True)

def _base_layout(fig, height=320, title=None):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=44 if title else 16, b=10), title=title,
                      legend=dict(orientation="h", y=1.14), hovermode="x unified")
    return fig

def chart_cumulative_pnl(perf_df: pd.DataFrame, title: str):
    if perf_df.empty:
        return
    x = [str(d) for d in perf_df["exit_date"]]
    cum = perf_df["pnl_gbp"].cumsum()
    if not PLOTLY_OK:
        st.line_chart(pd.DataFrame({"Running profit/loss (£)": cum.values}, index=x))
        return
    colour = "#16a34a" if cum.iloc[-1] >= 0 else "#dc2626"
    fig = go.Figure(go.Scatter(x=x, y=cum, mode="lines+markers", line=dict(color=colour, width=3, shape="hv"),
                               fill="tozeroy", name="Running profit/loss (£)",
                               hovertemplate="%{x}<br>£%{y:,.2f}<extra></extra>"))
    show_fig(_base_layout(fig, 320, title))

def chart_outcome_donut(perf: dict):
    labels = ["Hit the goal", "Safety exit", "Sold with no loss", "Time ran out"]
    vals = [perf["wins"], perf["losses"], perf["be"], perf["expired"]]
    if not PLOTLY_OK:
        st.bar_chart(pd.DataFrame({"trades": vals}, index=labels))
        return
    fig = go.Figure(go.Pie(labels=labels, values=vals, hole=0.62, sort=False,
                           marker=dict(colors=["#16a34a", "#dc2626", "#d97706", "#64748b"]), textinfo="label+percent"))
    fig.update_layout(height=320, margin=dict(l=10, r=10, t=44, b=10), title="How your trades ended", showlegend=False)
    show_fig(fig)

def chart_pnl_hist(perf_df: pd.DataFrame):
    if perf_df.empty:
        return
    if not PLOTLY_OK:
        st.bar_chart(perf_df["pnl_pct"].round(1).value_counts().sort_index())
        return
    fig = go.Figure(go.Histogram(x=perf_df["pnl_pct"], nbinsx=24, marker=dict(color="#6366f1"), hovertemplate="%{x:.1f}% : %{y} trades<extra></extra>"))
    fig.add_vline(x=0, line_dash="dash", line_color="gray")
    fig.update_layout(height=320, margin=dict(l=10, r=10, t=44, b=10), title="How much each trade made or lost (%) — right of the dashed line = profit", bargap=0.05)
    show_fig(fig)

def chart_sector_exposure(exp: dict):
    if not exp:
        return
    items = sorted(exp.items(), key=lambda x: x[1])
    if not PLOTLY_OK:
        st.bar_chart(pd.DataFrame({"open trades": [v for _, v in items]}, index=[k for k, _ in items]))
        return
    fig = go.Figure(go.Bar(x=[v for _, v in items], y=[k for k, _ in items], orientation="h",
                           marker=dict(color=["#dc2626" if v >= MAX_ACTIVE_PER_SECTOR else "#6366f1" for _, v in items]),
                           hovertemplate="%{y}: %{x} open<extra></extra>"))
    fig.add_vline(x=MAX_ACTIVE_PER_SECTOR, line_dash="dot", line_color="#d97706", annotation_text=f"max {MAX_ACTIVE_PER_SECTOR}")
    fig.update_layout(height=max(220, 38 * len(items) + 60), margin=dict(l=10, r=10, t=40, b=10), title="Open trades per sector (red = full)", xaxis=dict(dtick=1))
    show_fig(fig)

def calibration_rows(closed_df: pd.DataFrame) -> list:
    rows = []
    if closed_df is None or closed_df.empty:
        return rows
    for _, r in closed_df.iterrows():
        o = classify_outcome(r["status"])
        if o not in ("WIN", "LOSS"):
            continue
        f = _parse_features(r["features_json"])
        if f.get("base_ml_conf") is None:
            continue
        try:
            conf = float(f.get("base_ml_conf", 0)) + float(f.get("learner_delta", 0)) + float(f.get("macro_delta", 0))
        except Exception:
            continue
        rows.append((conf, 1 if o == "WIN" else 0))
    return rows

def chart_calibration(rows: list):
    d = pd.DataFrame(rows, columns=["conf", "hit"])
    d["bucket"] = pd.cut(d["conf"], bins=[0, 60, 70, 80, 101], labels=["Under 60%", "60–70%", "70–80%", "80%+"])
    g = d.groupby("bucket", observed=True).agg(stated=("conf", "mean"), realised=("hit", lambda s: s.mean() * 100.0), n=("hit", "size")).reset_index()
    if not PLOTLY_OK:
        st.dataframe(g.rename(columns={"bucket": "AI confidence", "stated": "AI said (%)", "realised": "Actually won (%)", "n": "Trades"}), hide_index=True)
        return
    fig = go.Figure()
    fig.add_bar(x=g["bucket"].astype(str), y=g["stated"], name="What the AI said", marker_color="#6366f1")
    fig.add_bar(x=g["bucket"].astype(str), y=g["realised"], name="How often it actually won", marker_color="#16a34a",
                customdata=g["n"], hovertemplate="%{y:.0f}% won (from %{customdata} trades)<extra></extra>")
    fig.update_layout(barmode="group", height=320, margin=dict(l=10, r=10, t=44, b=10), title="When the AI says “X% sure”, how often is it right?",
                      yaxis=dict(range=[0, 100], title="%"), legend=dict(orientation="h", y=1.14))
    show_fig(fig)

def chart_rolling(perf_df: pd.DataFrame, window: int = 10):
    n = len(perf_df)
    if n < window + 5:
        st.info(f"This chart appears once {window + 5} trades have finished (so far: {n}).")
        return
    wins = (perf_df["outcome"] == "WIN").astype(float)
    roll = (wins.rolling(window).mean() * 100.0).tolist()
    overall = float(wins.mean() * 100.0)
    x = list(range(1, n + 1))
    if not PLOTLY_OK:
        st.line_chart(pd.DataFrame({f"Win rate of last {window} trades (%)": roll}, index=x))
        return
    fig = go.Figure(go.Scatter(x=x, y=roll, mode="lines", line=dict(color="#6366f1", width=3), name=f"Last {window} trades",
                               hovertemplate="After trade %{x}: %{y:.0f}% won<extra></extra>"))
    fig.add_hline(y=overall, line_dash="dash", line_color="gray", annotation_text=f"overall {overall:.0f}%", annotation_position="top left")
    fig.update_layout(height=300, margin=dict(l=10, r=10, t=30, b=10), yaxis=dict(range=[0, 100], title="% won"), xaxis_title="Finished trade number", showlegend=False)
    show_fig(fig)

def chart_position(ticker: str, entry: float, target: float, stop: float):
    """Simple price line (last 3 months) with three labelled lines: Bought at / Goal / Safety exit."""
    h = fetch_cached_history(ticker, period="3mo")
    if h.empty:
        st.info("No price history available right now.")
        return
    xs = [str(d) for d in _bar_dates(h)]
    close = [float(v) for v in h["Close"]]
    if not PLOTLY_OK:
        st.line_chart(pd.DataFrame({"Price": close, "Goal": target, "Bought at": entry, "Safety exit": stop}, index=xs))
        return
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xs, y=close, mode="lines", name="Share price", line=dict(color="#6366f1", width=3),
                             hovertemplate="%{x}<br>%{y:.2f}p<extra></extra>"))
    fig.add_trace(go.Scatter(x=[xs[0], xs[-1]], y=[stop, target], mode="markers", marker=dict(opacity=0), showlegend=False, hoverinfo="skip"))
    for label, val, col in [("🎯 Goal", target, "#16a34a"), ("Bought at", entry, "#64748b"), ("🛑 Safety exit", stop, "#dc2626")]:
        fig.add_hline(y=val, line_dash="dash", line_color=col, annotation_text=f"{label} {val:.2f}p", annotation_position="top left")
    fig.update_layout(height=380, margin=dict(l=10, r=10, t=20, b=10), showlegend=False, hovermode="x unified", yaxis_title="Price (pence)")
    show_fig(fig)

def style_ledger(df: pd.DataFrame):
    """Colourised, sortable table; falls back to the plain table if styling is unavailable."""
    try:
        def pnl_color(v):
            try:
                return "color:#16a34a;font-weight:700" if float(v) > 0 else ("color:#dc2626;font-weight:700" if float(v) < 0 else "")
            except Exception:
                return ""
        sty = df.style.format({"Bought at (p)": "{:.2f}", "Goal (p)": "{:.2f}", "Safety exit (p)": "{:.2f}",
                               "Price now (p)": "{:.2f}", "Profit/Loss (%)": "{:+.2f}%"}, na_rep="-")
        mapper = sty.map if hasattr(sty, "map") else sty.applymap
        return mapper(pnl_color, subset=["Profit/Loss (%)"])
    except Exception:
        return df

# ---------- "what is the AI learning" helpers ----------
def compute_memory(closed_df: pd.DataFrame) -> tuple:
    """Same decay maths the AI really uses: returns (careful, liked) lists of (ticker, points, days_until_it_fades)."""
    careful, liked = [], []
    if closed_df is None or closed_df.empty:
        return careful, liked
    today_dt = london_now().date()
    acc = {}
    for _, r in closed_df.iterrows():
        o = classify_outcome(r["status"])
        if o not in ("WIN", "LOSS"):
            continue
        d = _to_date(_first_valid(r.get("exit_timestamp"), r.get("last_audited"), r.get("date_str")))
        days_ago = max(0, (today_dt - d).days) if d else 15
        w = math.pow(0.5, days_ago / HALF_LIFE_DAYS)
        a = acc.setdefault(str(r["ticker"]), {"loss": 0.0, "win": 0.0})
        if o == "LOSS":
            a["loss"] += 15.0 * w
        else:
            a["win"] += 5.0 * w
    for t, a in acc.items():
        if a["loss"] >= 1.0:
            careful.append((t, int(round(a["loss"])), int(round(HALF_LIFE_DAYS * math.log2(max(a["loss"], 1.0))))))
        if a["win"] >= 1.0:
            liked.append((t, int(round(a["win"])), int(round(HALF_LIFE_DAYS * math.log2(max(a["win"], 1.0))))))
    careful.sort(key=lambda x: -x[1])
    liked.sort(key=lambda x: -x[1])
    return careful, liked

def sector_stats(perf_df: pd.DataFrame) -> pd.DataFrame:
    if perf_df.empty:
        return pd.DataFrame()
    d = perf_df.copy()
    d["Sector"] = d.apply(lambda x: get_trade_sector(x["ticker"], x["features_json"]), axis=1)
    g = d.groupby("Sector").agg(n=("pnl_pct", "size"), avg=("pnl_pct", "mean"), wins=("outcome", lambda s: (s == "WIN").sum())).reset_index()
    g["win_rate"] = g["wins"] / g["n"] * 100.0
    return g

def build_insights(perf_df: pd.DataFrame) -> list:
    """Plain-English observations; only states what the data can support."""
    out = []
    if perf_df.empty:
        return out
    sg = sector_stats(perf_df)
    sg3 = sg[sg["n"] >= 3] if not sg.empty else sg
    if not sg3.empty:
        best = sg3.sort_values("avg", ascending=False).iloc[0]
        worst = sg3.sort_values("avg", ascending=True).iloc[0]
        out.append(f"Best sector so far: **{best['Sector']}** (average {best['avg']:+.1f}% per trade over {int(best['n'])} trades).")
        if worst["Sector"] != best["Sector"]:
            out.append(f"Toughest sector so far: **{worst['Sector']}** (average {worst['avg']:+.1f}% per trade over {int(worst['n'])} trades).")
    try:
        frs = [get_trade_friction_pct(r["ticker"], r["features_json"]) for _, r in perf_df.iterrows()]
        if frs:
            out.append(f"Fees, spread and stamp duty take about **{np.mean(frs):.1f}%** off every trade — a trade has to beat that before it earns anything.")
    except Exception:
        pass
    wins = perf_df[perf_df["outcome"] == "WIN"]
    losses = perf_df[perf_df["outcome"] == "LOSS"]
    try:
        def hold(df_):
            v = [(b - a).days for a, b in zip(df_["entry_date"], df_["exit_date"]) if a is not None and b is not None]
            return float(np.mean(v)) if v else None
        hw, hl = hold(wins), hold(losses)
        if hw is not None and hl is not None and len(wins) >= 3 and len(losses) >= 3:
            out.append(f"Winners reached their goal in about **{hw:.1f} days**; losing trades hit their safety exit after about **{hl:.1f} days**.")
    except Exception:
        pass
    try:
        def atr_mean(df_):
            v = []
            for _, r in df_.iterrows():
                a = _parse_features(r["features_json"]).get("atr_pct")
                if a is not None:
                    v.append(float(a))
            return (float(np.mean(v)), len(v)) if v else (None, 0)
        aw, nw = atr_mean(wins)
        al, nl = atr_mean(losses)
        if aw is not None and al is not None and nw >= 5 and nl >= 5:
            if al > aw + 0.3:
                out.append(f"Losing trades were on shares that move about **{al:.1f}% a day**, versus **{aw:.1f}%** for winners — very jumpy shares have been riskier, so the AI is wary of them.")
            else:
                out.append(f"Winners and losers were on shares with similar day-to-day movement (about {aw:.1f}% vs {al:.1f}%), so jumpiness hasn't been the deciding factor.")
    except Exception:
        pass
    return out

def current_streak(perf_df: pd.DataFrame) -> tuple:
    if perf_df.empty:
        return 0, True
    signs = (perf_df["pnl_gbp"] > 0).tolist()
    last = signs[-1]
    k = 0
    for s in reversed(signs):
        if s == last:
            k += 1
        else:
            break
    return k, last

# ==============================================================================
# 11. SIDEBAR & AUTONOMOUS LOOP CONTROLLER
# ==============================================================================
inject_css()

market_is_open = is_lse_market_open()

st.sidebar.header("⚙️ Settings")
st.sidebar.number_input("Broker fee per trade (£)", min_value=0.0, max_value=25.0, value=DEFAULT_COMMISSION_GBP, step=0.5, key="commission_gbp",
                        help="Charged when you buy and again when you sell. Included in every profit/loss number you see.")
broker_mode = st.sidebar.selectbox("Trading mode", ["Paper Trading (Simulated)", "Interactive Brokers", "IG Group API"],
                                   help="Paper trading uses pretend money. Nothing is bought for real.")
auto_mode = st.sidebar.toggle("Auto-refresh", value=market_is_open, help="Keeps checking your trades while the market is open.")
refresh_interval_sec = st.sidebar.selectbox("Check every", [300, 600, 3600], format_func=lambda x: f"{x//60} minutes")
show_sim_options = st.sidebar.toggle("Show practice options in My Trades", value=True)

with st.sidebar.expander("🛡️ Built-in safety rules"):
    st.markdown(
        f"- Max **{MAX_DAILY_EQUITY_TRADES} new trades** per day\n"
        f"- Max **{MAX_ACTIVE_PER_SECTOR} open trades** in any one sector\n"
        f"- Max **{MAX_OPEN_POSITIONS} open trades** in total (and about £{MAX_OPEN_RISK_GBP:.0f} at risk)\n"
        f"- Once a trade is about **{int(BREAKEVEN_TRIGGER_RATIO * 100)}%** of the way to its goal, its safety exit moves up so it can't lose money\n"
        f"- A stock isn't bought again for **{COOLDOWN_CALENDAR_DAYS} days** after you sell it\n"
        f"- Buying only **{ENTRY_WINDOW_START.strftime('%H:%M')}–{ENTRY_WINDOW_END.strftime('%H:%M')}** London time, using real prices\n"
        f"- Skips tiny or hard-to-trade shares\n"
        f"- Buying pauses if the UK market falls sharply in a week\n"
        f"- Every result includes UK stamp duty, spread and your broker fee"
    )

loop_tick = 0
if auto_mode and st_autorefresh:
    loop_tick = st_autorefresh(interval=refresh_interval_sec * 1000, key="unified_autonomous_loop")

is_new_loop_tick = ("last_loop_tick" not in st.session_state) or (loop_tick != st.session_state["last_loop_tick"])

if is_new_loop_tick:
    audit_and_reconcile_all_trades()
    hydrate_duckdb_from_supabase()
    st.session_state["last_loop_tick"] = loop_tick
    st.session_state["last_audit_str"] = london_now().strftime("%d %b %H:%M")

with st.sidebar.expander("🩺 System health"):
    try:
        prices_ok = not fetch_cached_history("SHEL.L", period="5d").empty
    except Exception:
        prices_ok = False
    st.write(f"{'🟢' if prices_ok else '🔴'} Share prices {'arriving' if prices_ok else 'not arriving (Yahoo may be down)'}")
    st.write(f"{'🟢' if supabase else '🔴'} Online backup {'connected' if supabase else 'not connected'}")
    st.write(f"{'🟢' if get_ml_model() is not None else '🔴'} AI model {'loaded' if get_ml_model() is not None else 'missing'}")
    st.write(f"🕒 Last check: {st.session_state.get('last_audit_str', '-')}")

if "retrain_notice" in st.session_state:
    st.sidebar.success(st.session_state["retrain_notice"])
if "ml_model_source" in st.session_state:
    st.sidebar.caption(f"🤖 AI version in use: **{st.session_state['ml_model_source']}**")

if "db_error" in st.session_state:
    st.sidebar.error(f"⚠️ Online backup problem: {st.session_state['db_error']}")
if st.session_state.get("pending_cloud_sync"):
    st.sidebar.warning(f"⏳ {len(st.session_state['pending_cloud_sync'])} update(s) waiting to be saved online.")

if st.sidebar.button("🚪 Log Out", width="stretch"):
    st.session_state["password_correct"] = False
    st.rerun()

market_status = "OPEN" if market_is_open else "CLOSED"
macro_bar = get_cross_asset_macro_regime()

def macro_chip(label, v):
    cls = "up" if v >= 0 else "down"
    arrow = "▲" if v >= 0 else "▼"
    return f'<span class="chip {cls}">{esc(label)} {arrow} {v:+.1f}%</span>'

stress_chip = '<span class="chip down">⛔ Buying paused — UK market fell sharply this week</span>' if is_market_stress(macro_bar) else ""
st.markdown(_collapse(f"""
<div class="hero">
  <h1>⚡ ALPHA-LSE Quant Terminal</h1>
  <div class="sub">An AI that picks UK shares and practises trading them with pretend money — so you can see if it really works.</div>
  <div class="chips">
    <span class="chip {'up' if market_is_open else 'down'}">UK market {market_status}</span>
    <span class="chip {'up' if supabase else 'down'}">{'☁️ Saved online' if supabase else '☁️ Not saved online'}</span>
    {macro_chip('UK market this week', macro_bar['ftse_5d'])}{macro_chip('Oil', macro_bar['brent_5d'])}{macro_chip('Copper', macro_bar['copper_5d'])}{macro_chip('£ vs $', macro_bar['gbpusd_5d'])}
    {stress_chip}
    <span class="chip">Last checked {esc(st.session_state.get('last_audit_str', '-'))}</span>
  </div>
</div>
"""), unsafe_allow_html=True)

with st.expander("💡 New here? 30-second guide"):
    st.markdown(
        "- This app **pretends to buy** UK shares, about **£500 each**. It's practice — no real money is used.\n"
        "- Every pick has a **Goal** (the price to sell at for a profit) and a **Safety exit** (the price to sell at to limit the loss).\n"
        "- When a trade gets about two-thirds of the way to its goal, the safety exit moves up so the trade **can't lose money** any more.\n"
        "- **AI confidence** is how sure the AI is that the goal is reached before the safety exit. It's a rough guide, not a promise.\n"
        "- Every profit/loss number is **after fees and UK stamp duty**.\n"
        "- If neither the goal nor the safety exit is reached, the trade closes after **7 days**.\n"
        "- The **Overview** tab tells you honestly whether it's making money. The **What the AI Learned** tab shows how it's improving."
    )

tab_home, tab_scanner, tab_journal, tab_learn, tab_options = st.tabs([
    "🏠 Overview",
    "🎯 Today's Picks",
    "📖 My Trades",
    "🧠 What the AI Learned",
    "🎲 Options (Practice)"
])

# ==============================================================================
# 12. TODAY'S PICKS
# ==============================================================================
if "dispatched_orders" not in st.session_state:
    st.session_state["dispatched_orders"] = set()

def render_cohort(todays_df: pd.DataFrame):
    cols = st.columns(min(len(todays_df), 3))
    for pos_i, (_, t_row) in enumerate(todays_df.iterrows()):
        tkr_sym = str(t_row["ticker"])
        f_meta = _parse_features(t_row.get("features_json"))
        base_c = int(f_meta.get("base_ml_conf", 75))
        adj = int(f_meta.get("learner_delta", 0)) + int(f_meta.get("macro_delta", 0))
        sec_name = get_trade_sector(tkr_sym, f_meta)
        tax_tag = f_meta.get("tax_regime") or ("AIM (0% SDRT)" if is_aim_exempt(tkr_sym) else "Main (0.5% SDRT)")
        be_active = is_stop_breakeven_protected(float(t_row["entry_price"]), float(t_row["stop_loss"]), tkr_sym)
        with cols[pos_i % 3]:
            st.markdown(trade_card_html(
                pos_i + 1, tkr_sym, sec_name, t_row["status"], float(t_row["entry_price"]), float(t_row["target_price"]),
                float(t_row["stop_loss"]), float(t_row["latest_price"]), float(t_row["pnl_pct"]), int(t_row["shares"]),
                float(t_row["capital_allocated"]), base_c + adj, base_c, adj, tax_tag,
                get_trade_friction_pct(tkr_sym, f_meta), t_row["last_audited"], be_active), unsafe_allow_html=True)

with tab_scanner:
    if "flash" in st.session_state:
        st.info(st.session_state.pop("flash"))

    todays_logged_df = get_todays_logged_equities()
    quota_filled = len(todays_logged_df) >= MAX_DAILY_EQUITY_TRADES

    c1, c2, c3, c4 = st.columns([1, 1, 1, 1.2])
    cap_state = get_portfolio_capacity()
    c1.metric("Picks today", f"{len(todays_logged_df)} of {MAX_DAILY_EQUITY_TRADES}")
    c2.metric("Trades open", f"{cap_state['open']} of {MAX_OPEN_POSITIONS}")
    c3.metric("Money at risk", f"£{cap_state['risk_gbp']:.0f}", help="Roughly what you'd lose if every open trade hit its safety exit.")
    with c4:
        st.write("")
        re_scan = st.button("🔄 Find new picks now", width="stretch", type="primary")

    if is_market_stress(macro_bar):
        st.error("⛔ The UK market has fallen sharply this week, so buying is paused to protect you. Picks are shown for information only.")
    elif not cap_state["ok"]:
        st.warning(f"🧯 {cap_state['reason']}. New buying resumes when some trades close.")
    elif not is_entry_window_open():
        st.info("🕗 The market isn't in its buying window (08:30–16:00 London, Mon–Fri). Picks below are a watch-list only, so every purchase is made at a real price.")

    if quota_filled and not re_scan:
        st.info(f"🔒 You've used all {MAX_DAILY_EQUITY_TRADES} picks for today. This stops over-trading.")
        sec_title("📌", "Today's trades")
        render_cohort(todays_logged_df)
    else:
        if re_scan or is_new_loop_tick or "scan_results" not in st.session_state:
            with st.spinner("Looking through UK shares and checking your open trades..."):
                if re_scan:
                    audit_and_reconcile_all_trades()
                    hydrate_duckdb_from_supabase()
                    st.session_state["last_audit_str"] = london_now().strftime("%d %b %H:%M")
                res_df, has_cleared = run_predictions()
                st.session_state["scan_results"] = res_df
                st.session_state["has_cleared"] = has_cleared
                st.session_state["scan_ts"] = london_now().strftime("%H:%M:%S")
            todays_logged_df = get_todays_logged_equities()

        if not todays_logged_df.empty:
            with st.expander(f"📌 Already bought today ({len(todays_logged_df)})", expanded=False):
                render_cohort(todays_logged_df)

        df_res = st.session_state.get("scan_results", pd.DataFrame())
        stats = st.session_state.get("scan_stats", {})
        if stats and "error" not in stats:
            st.caption(
                f"Last check {st.session_state.get('scan_ts', '-')}: looked at {stats.get('scanned', 0)} shares · "
                f"skipped {stats.get('illiquid', 0)} (too small or hard to trade) · {stats.get('rns_blocked', 0)} blocked by bad news · "
                f"{stats.get('sector_blocked', 0)} in a full sector · bought {stats.get('logged', 0)} automatically")
        elif stats.get("error"):
            st.error(stats["error"])

        if st.session_state.get("has_cleared", False) and not df_res.empty:
            st.success(f"🟢 {len(df_res)} share(s) passed every check")
            sec_title("🔥", "Best picks right now")
            cols = st.columns(min(len(df_res), 3))
            today_key = london_now().strftime('%Y-%m-%d')
            for idx, row in df_res.head(3).iterrows():
                with cols[idx % 3]:
                    st.markdown(signal_card_html(idx + 1, row, bool(row.get("Loggable", False))), unsafe_allow_html=True)
                    idem_token = hashlib.sha256(f"{row['Ticker']}_{today_key}_{broker_mode}".encode()).hexdigest()[:12]
                    already_sent = idem_token in st.session_state["dispatched_orders"]
                    btn_label = "✅ Already bought" if already_sent else f"🚀 Buy now ({broker_mode})"
                    if st.button(btn_label, key=f"exec_{idem_token}", width="stretch", disabled=already_sent):
                        ok = log_equity_signal_safely(row.to_dict(), enforce_sector_cap=True)
                        if ok:
                            st.session_state["dispatched_orders"].add(idem_token)
                            st.session_state["flash"] = f"✅ Bought {row['Ticker']} ({broker_mode})."
                        else:
                            st.session_state["flash"] = f"⚠️ Couldn't buy {row['Ticker']}: {st.session_state.get('last_log_reason', 'a safety rule blocked it')}."
                        st.rerun()

            sec_title("📋", f"All {len(df_res)} picks", "best first")
            table = df_res.rename(columns={
                "Ticker": "Stock", "Price (p)": "Buy at (p)", "Target (p)": "Goal (p)", "Stop Loss (p)": "Safety exit (p)",
                "Expected Return": "Possible profit", "AI Win Confidence": "AI confidence", "Recommended Shares": "Shares",
                "RNS": "News check", "Loggable": "Ready to buy"})
            show_cols = ["Stock", "Sector", "Buy at (p)", "Goal (p)", "Safety exit (p)", "Possible profit", "AI confidence", "Shares", "News check", "Ready to buy"]
            st.dataframe(table[[c for c in show_cols if c in table.columns]], width="stretch", hide_index=True)
        else:
            st.warning("🛡️ **Nothing to buy right now.** No share passes all the checks (trend, trading volume, news, sector room, AI confidence). Doing nothing is a valid choice.")

# ==============================================================================
# 13. OPTIONS (PRACTICE)
# ==============================================================================
with tab_options:
    sec_title("🎲", "Options idea of the day", "practice only")
    st.caption("⚠️ These are pretend prices worked out with a maths formula (including a typical buy/sell gap). They are not real market prices "
               "and are **not** counted in your main results. Real UK options on single shares are thin, so real results would differ.")
    opt_signal = generate_daily_options_alpha()
    if opt_signal:
        with st.container(border=True):
            o1, o2, o3 = st.columns(3)
            with o1:
                st.metric("Idea", opt_signal["option_contract"], help="A 'Call' option gains value if the share price goes up.")
                st.markdown(f"Share price when picked: **{opt_signal['underlying_spot']:.2f}p**")
            with o2:
                st.metric("Value now", f"{opt_signal['current_option_price']:.2f}p", f"{opt_signal['pnl_pct']:+.2f}% vs cost ({opt_signal['entry_premium']:.2f}p)")
                st.markdown(f"Recent share wobble: **{opt_signal['implied_vol']}%** a year")
            with o3:
                st.metric("Trend strength", f"{opt_signal['ai_confidence']}", help="Based on the share's last 5 days. It is NOT a chance of winning.")
                st.markdown(f"Cost: **£{opt_signal['total_capital']:,.2f}** ({int(opt_signal['lot_size'])} units)")
            m1, m2, m3 = st.columns(3)
            m1.info(f"🎯 Goal: {opt_signal['target_premium']:.2f}p (+60%)")
            m2.warning(f"🛑 Safety exit: {opt_signal['stop_loss_premium']:.2f}p (-50%)")
            m3.success(f"Status: **{friendly_status(opt_signal['status'])}**")
    else:
        st.info("No practice option fits today (budget limit, or each candidate share already has one open).")
    with st.expander("How the practice options work"):
        st.markdown(
            f"- Prices come from a standard maths formula (Black-Scholes) using recent share wobble and typical dividends, with a ±{OPTION_HALF_SPREAD_PCT:.0f}% buy/sell gap.\n"
            "- The strike is 2% above the current share price. Only one practice option per share is open at a time.\n"
            "- Goal is +60% and safety exit is -50%. If both could have happened on the same day, the safety exit is assumed to come first.\n"
            "- These are **not real quotes**, so don't treat them as proof of anything.")

# ==============================================================================
# DATA LOAD (after scanner/options so anything just bought is included)
# ==============================================================================
df_eq = pd.DataFrame()
df_opt = pd.DataFrame()
with DB_LOCK:
    _con = duckdb.connect(DB_PATH, read_only=True)
    try:
        df_eq = _con.execute("SELECT * FROM trade_journal ORDER BY date_str DESC, ticker ASC").df()
        df_opt = _con.execute("SELECT * FROM daily_options_journal ORDER BY date_key DESC").df()
    except Exception:
        pass
    finally:
        _con.close()

perf_eq_df = prep_perf_frame(df_eq, "capital_allocated")
perf_opt_df = prep_perf_frame(df_opt, "total_capital", is_option=True)
perf_eq = compute_performance(perf_eq_df)

active_eq = df_eq[df_eq["status"].astype(str).str.upper() == "ACTIVE"].copy() if not df_eq.empty else pd.DataFrame()
active_cap = float(active_eq["capital_allocated"].sum()) if not active_eq.empty else 0.0
open_unreal = float((active_eq["capital_allocated"] * active_eq["pnl_pct"] / 100.0).sum()) if not active_eq.empty else 0.0
be_prot = int(sum(is_stop_breakeven_protected(float(r["entry_price"]), float(r["stop_loss"])) for _, r in active_eq.iterrows())) if not active_eq.empty else 0
open_ret = (open_unreal / active_cap * 100.0) if active_cap > 0 else 0.0
closed_all = load_closed_trades_df()

# ==============================================================================
# 14. OVERVIEW (SCOREBOARD)
# ==============================================================================
with tab_home:
    n_ = perf_eq["n"]
    if n_ == 0:
        sec_title("🏠", "Scoreboard")
        score_cards([
            ("Total profit/loss", "£0.00", "no finished trades yet", "", ""),
            ("Win rate", "–", "needs finished trades", "", ""),
            ("Finished trades", "0", "come back soon", "", ""),
            ("Open right now", str(len(active_eq)), f"£{active_cap:,.0f} invested (pretend)", "", ""),
        ])
        callout("info", "⏳", "<b>Nothing has finished yet.</b> Once the first trades reach their goal or safety exit, your results and the AI's lessons will appear here.")
    else:
        exp_ = perf_eq["expectancy_pct"]
        net = perf_eq["net_gbp"]
        sec_title("🏠", "Scoreboard", "finished shares trades, after fees & tax")
        score_cards([
            ("Total profit/loss", f"£{net:+,.2f}", f"£{perf_eq['expectancy_gbp']:+.2f} per trade on average", "good" if net > 0 else ("bad" if net < 0 else ""), "pos" if net > 0 else ("neg" if net < 0 else "")),
            ("Win rate", f"{perf_eq['target_hit_rate']:.0f}%", f"{perf_eq['wins']} won · {perf_eq['losses']} lost · {perf_eq['be']} saved · {perf_eq['expired']} timed out", "", ""),
            ("Finished trades", str(n_), f"likely win rate {perf_eq['ci_low']:.0f}–{perf_eq['ci_high']:.0f}%", "", ""),
            ("Open right now", str(len(active_eq)), f"£{open_unreal:+,.2f} so far · 🛡️ {be_prot} can't lose", "good" if open_unreal > 0 else ("bad" if open_unreal < 0 else ""), ""),
        ])
        if n_ < 30:
            callout("warn", "⏳", f"<b>Too early to judge.</b> Only {n_} trades have finished. Results this early can be pure luck.")
        elif exp_ > 0.3:
            callout("good", "✅", f"<b>Looking positive.</b> On average each trade made <b>{exp_:+.2f}%</b> after costs ({n_} trades).")
        elif exp_ > -0.3:
            callout("info", "➖", f"<b>Roughly break-even.</b> On average <b>{exp_:+.2f}%</b> per trade after costs ({n_} trades). Not clearly making or losing money yet.")
        else:
            callout("bad", "⚠️", f"<b>Losing money after costs.</b> On average <b>{exp_:+.2f}%</b> per trade ({n_} trades). Don't use real money.")
        if n_ < 100 and n_ >= 30:
            st.caption("Still a fairly small sample — be cautious about drawing firm conclusions.")

        ch1, ch2 = st.columns([3, 2])
        with ch1:
            chart_cumulative_pnl(perf_eq_df, "Your running profit/loss (£) — going up is good")
        with ch2:
            chart_outcome_donut(perf_eq)

        sec_title("✨", "Highlights")
        best_i = perf_eq_df["pnl_pct"].idxmax()
        worst_i = perf_eq_df["pnl_pct"].idxmin()
        streak_n, streak_win = current_streak(perf_eq_df)
        dh = [(b - a).days for a, b in zip(perf_eq_df["entry_date"], perf_eq_df["exit_date"]) if a is not None and b is not None]
        h1, h2, h3, h4 = st.columns(4)
        with h1:
            stat_card("🏆 Best trade", f"{perf_eq_df.loc[best_i, 'ticker']} {perf_eq_df.loc[best_i, 'pnl_pct']:+.1f}%", "after costs")
        with h2:
            stat_card("📉 Worst trade", f"{perf_eq_df.loc[worst_i, 'ticker']} {perf_eq_df.loc[worst_i, 'pnl_pct']:+.1f}%", "after costs")
        with h3:
            stat_card("🔥 Current streak", f"{streak_n} {'profitable' if streak_win else 'unprofitable'}" if streak_n else "–", "trades in a row")
        with h4:
            stat_card("⏱️ Typical hold", f"{np.mean(dh):.1f} days" if dh else "–", "from buy to sell")

        ftse_s = get_ftse_close_series()
        first_d = perf_eq_df["entry_date"].dropna().min()
        ftse_since = ftse_return_between(ftse_s, first_d, london_now().date()) if first_d else None
        excess = []
        for _, rr in perf_eq_df.iterrows():
            fr_ = ftse_return_between(ftse_s, rr["entry_date"], rr["exit_date"])
            if fr_ is not None:
                excess.append(float(rr["pnl_pct"]) - fr_)
        if ftse_since is not None:
            beat = (f" On the same days, your trades beat the market by <b>{np.mean(excess):+.2f}%</b> per trade on average." if excess else "")
            callout("info", "🇬🇧", f"<b>Compared with the UK market:</b> the FTSE 100 moved <b>{ftse_since:+.2f}%</b> since your first trade.{beat}")

    if not active_eq.empty:
        sec_title("🟢", "Open right now", "dot = current price · green side = nearer the goal")
        act = active_eq.sort_values("pnl_pct", ascending=False).head(12)
        recs = list(act.iterrows())
        for i in range(0, len(recs), 4):
            cols = st.columns(4)
            for j, (_, r) in enumerate(recs[i:i + 4]):
                with cols[j]:
                    st.markdown(mini_trade_html(
                        r["ticker"], get_trade_sector(r["ticker"], r["features_json"]), float(r["entry_price"]), float(r["target_price"]),
                        float(r["stop_loss"]), float(r["latest_price"]), float(r["pnl_pct"]),
                        is_stop_breakeven_protected(float(r["entry_price"]), float(r["stop_loss"]))), unsafe_allow_html=True)
        if len(active_eq) > 12:
            st.caption(f"Showing 12 of {len(active_eq)} open trades. See all of them in My Trades.")

    if n_ > 0:
        with st.expander("🔍 More detail"):
            b1, b2, b3, b4 = st.columns(4)
            b1.metric("Average winning trade", f"£{perf_eq['avg_win_gbp']:+.2f}")
            b2.metric("Average losing trade", f"£{perf_eq['avg_loss_gbp']:+.2f}")
            b3.metric("Money won ÷ money lost", f"{perf_eq['profit_factor']:.2f}" if perf_eq["profit_factor"] is not None else "no losses yet",
                      help="Above 1 means more money won than lost.")
            b4.metric("Biggest drop from a peak", f"£{perf_eq['max_dd_gbp']:+.2f}", help="The worst losing streak, measured from the highest point your running total reached.")
            chart_pnl_hist(perf_eq_df)
            sg = sector_stats(perf_eq_df)
            if not sg.empty:
                tbl = sg.rename(columns={"n": "Trades", "avg": "Average %", "win_rate": "Win rate %"})[["Sector", "Trades", "Average %", "Win rate %"]].round(1)
                st.markdown("**Results by sector**")
                st.dataframe(tbl.sort_values("Average %", ascending=False), width="stretch", hide_index=True)
        with st.expander("🧪 Practice options (pretend prices — not counted above)"):
            perf_opt = compute_performance(perf_opt_df)
            if perf_opt["n"] == 0:
                st.write("No finished practice options yet.")
            else:
                o1, o2, o3 = st.columns(3)
                o1.metric("Finished", perf_opt["n"])
                o2.metric("Win rate", f"{perf_opt['target_hit_rate']:.0f}%")
                o3.metric("Total profit/loss", f"£{perf_opt['net_gbp']:+,.2f}")
                chart_cumulative_pnl(perf_opt_df, "Practice options — running profit/loss (£)")

# ==============================================================================
# 15. MY TRADES
# ==============================================================================
with tab_journal:
    sec_title("📖", "My Trades")
    j1, j2 = st.columns([4, 1])
    with j1:
        st.caption("All profit/loss figures are after fees and UK stamp duty. Share prices from Yahoo can be about 15 minutes behind.")
    with j2:
        if st.button("🔄 Update now", width="stretch"):
            audit_and_reconcile_all_trades()
            hydrate_duckdb_from_supabase()
            st.session_state["last_audit_str"] = london_now().strftime("%d %b %H:%M")
            st.toast("Trades updated with the latest prices.", icon="✅")
            st.rerun()

    k1, k2, k3, k4 = st.columns(4)
    if perf_eq["n"] > 0:
        k1.metric("Win rate", f"{perf_eq['target_hit_rate']:.0f}%",
                  f"{perf_eq['wins']} won · {perf_eq['losses']} lost · {perf_eq['be']} saved · {perf_eq['expired']} timed out", delta_color="off",
                  help="Share of ALL finished trades that reached their goal.")
        k4.metric("Profit/loss on finished trades", f"£{perf_eq['net_gbp']:+,.2f}", f"{perf_eq['n']} trades", delta_color="off")
    else:
        k1.metric("Win rate", "–", "no finished trades yet", delta_color="off")
        k4.metric("Profit/loss on finished trades", "£+0.00", "0 trades", delta_color="off")
    k2.metric("Money in open trades", f"£{active_cap:,.0f}", f"🛡️ {be_prot} can no longer lose", delta_color="off")
    k3.metric("Profit/loss on open trades", f"£{open_unreal:+,.2f}", f"{open_ret:+.2f}%")

    sec_exp = get_active_sector_exposure()
    if sec_exp:
        with st.expander("📊 Open trades by sector", expanded=False):
            chart_sector_exposure(sec_exp)

    if not active_eq.empty:
        sec_title("🔬", "Look closer at an open trade")
        sel = st.selectbox("Pick a trade", list(active_eq["ticker"]), key="inspect_ticker")
        r = active_eq[active_eq["ticker"] == sel].iloc[0]
        ic1, ic2, ic3, ic4 = st.columns(4)
        ic1.metric("Bought at", f"{r['entry_price']:.2f}p")
        ic2.metric("Price now", f"{r['latest_price']:.2f}p", f"{r['pnl_pct']:+.2f}% after costs")
        ic3.metric("Goal", f"{r['target_price']:.2f}p")
        locked = is_stop_breakeven_protected(float(r["entry_price"]), float(r["stop_loss"]))
        ic4.metric("Safety exit", f"{r['stop_loss']:.2f}p", "🛡️ moved up — can't lose" if locked else "original level", delta_color="off")
        st.caption("The purple line is the share price over the last 3 months. Green dashed = your goal. Red dashed = your safety exit.")
        chart_position(sel, float(r["entry_price"]), float(r["target_price"]), float(r["stop_loss"]))

    master_list = []
    if not df_eq.empty:
        d = df_eq.copy()
        d["Sector"] = d.apply(lambda x: get_trade_sector(x["ticker"], x["features_json"]), axis=1)
        d["Type"] = "Share"
        d["_o"] = d["status"].apply(classify_outcome)
        d["Result"] = d.apply(lambda x: friendly_status(x["status"], is_stop_breakeven_protected(float(x["entry_price"]), float(x["stop_loss"]))), axis=1)
        master_list.append(d.rename(columns={"date_str": "Date", "ticker": "Stock", "entry_price": "Bought at (p)", "target_price": "Goal (p)",
                                             "stop_loss": "Safety exit (p)", "latest_price": "Price now (p)", "pnl_pct": "Profit/Loss (%)",
                                             "last_audited": "Last checked"}))
    if show_sim_options and not df_opt.empty:
        d = df_opt.copy()
        d["Type"] = "Option (practice)"
        d["Sector"] = d["share_name"].apply(lambda s: get_ticker_sector(f"{s}.L", resolve=False))
        d["_o"] = d["status"].apply(classify_outcome)
        d["Result"] = d["status"].apply(friendly_status)
        master_list.append(d.rename(columns={"date_key": "Date", "option_contract": "Stock", "entry_premium": "Bought at (p)",
                                             "current_option_price": "Price now (p)", "target_premium": "Goal (p)",
                                             "stop_loss_premium": "Safety exit (p)", "pnl_pct": "Profit/Loss (%)", "last_audited": "Last checked"}))

    if master_list:
        cols_keep = ["Date", "Stock", "Sector", "Type", "Bought at (p)", "Goal (p)", "Safety exit (p)", "Price now (p)", "Profit/Loss (%)", "Result", "Last checked", "_o"]
        master_df = pd.concat([m[[c for c in cols_keep if c in m.columns]] for m in master_list], ignore_index=True)
        master_df = master_df.sort_values("Date", ascending=False, kind="stable").reset_index(drop=True)

        sec_title("🔎", "All trades")
        f1, f2, f3, f4 = st.columns(4)
        label_map = {"Open": "ACTIVE", "Won": "WIN", "Lost": "LOSS", "Saved (no loss)": "BE", "Closed after 7 days": "EXPIRED"}
        chosen_status = f1.multiselect("Result", list(label_map.keys()), default=list(label_map.keys()))
        type_opts = sorted(master_df["Type"].unique())
        chosen_types = f2.multiselect("Type", type_opts, default=type_opts)
        sector_opts = sorted(master_df["Sector"].astype(str).unique())
        chosen_sectors = f3.multiselect("Sector", sector_opts, default=sector_opts)
        search = f4.text_input("Search a stock", "")

        wanted = {label_map[s] for s in chosen_status}
        view = master_df[master_df["_o"].isin(wanted) & master_df["Type"].isin(chosen_types) & master_df["Sector"].astype(str).isin(chosen_sectors)]
        if search.strip():
            view = view[view["Stock"].astype(str).str.contains(search.strip(), case=False, na=False)]
        view = view.drop(columns=["_o"])

        st.caption(f"Showing **{len(view)}** of {len(master_df)} trades. Click a column heading to sort.")
        st.dataframe(style_ledger(view), width="stretch", hide_index=True)
        st.download_button("⬇️ Download as spreadsheet (CSV)", view.to_csv(index=False).encode("utf-8"), "my_trades.csv", "text/csv")
    else:
        st.info("No trades yet. They appear here automatically once the AI buys something.")

# ==============================================================================
# 16. WHAT THE AI HAS LEARNED
# ==============================================================================
with tab_learn:
    n_ = perf_eq["n"]
    decided_n = int((perf_eq_df["outcome"].isin(["WIN", "LOSS"])).sum()) if not perf_eq_df.empty else 0
    level = "🌱 Rookie" if n_ < 30 else ("📚 Learning" if n_ < 100 else "🎓 Experienced")
    sec_title("🧠", "What the AI has learned", "in plain English")
    meter_pct = min(100.0, decided_n / MIN_SETTLED_TO_RETRAIN * 100.0)
    st.markdown(_collapse(
        f'<div class="stat"><div class="l">Experience level</div><div class="v">{level} — {n_} finished trades</div>'
        f'<div class="meter"><div style="width:{meter_pct:.0f}%"></div></div>'
        f'<div class="sub2">{decided_n} of {MIN_SETTLED_TO_RETRAIN} clear win/loss trades needed before the AI is allowed to consider upgrading itself.</div></div>'), unsafe_allow_html=True)

    if n_ == 0:
        callout("info", "🌱", "<b>The AI hasn't finished any trades yet</b>, so it has nothing to learn from. After a few trades close you'll see its lessons, strengths and weak spots here.")
    else:
        insights = build_insights(perf_eq_df)
        sec_title("💬", "What I've noticed")
        if insights:
            for line in insights:
                st.markdown(f"- {line}")
        else:
            st.write("Not enough trades yet to spot patterns.")

        sec_title("📰", "Latest lessons", "most recent finished trades")
        for _, r in perf_eq_df.tail(5).iloc[::-1].iterrows():
            o_ = classify_outcome(r["status"])
            lesson = {"WIN": "The AI gets a small confidence boost for this stock for a while.",
                      "LOSS": "The AI becomes more cautious about this stock for a while.",
                      "BE": "The safety exit protected the money. No penalty, no bonus.",
                      "EXPIRED": "Timed out — nothing learned from this one."}.get(o_, "")
            st.markdown(_collapse(
                f'<div class="lesson"><b>{esc(r["ticker"])}</b> · {esc(friendly_status(r["status"]))} · '
                f'<b class="{"pos" if r["pnl_pct"] >= 0 else "neg"}">{float(r["pnl_pct"]):+.1f}%</b><br><span class="sub2">{esc(lesson)}</span></div>'), unsafe_allow_html=True)

        sec_title("🏭", "Where it's working and where it isn't", "by sector")
        sg = sector_stats(perf_eq_df)
        sg3 = sg[sg["n"] >= 3] if not sg.empty else sg
        lc, rc = st.columns(2)
        with lc:
            st.markdown("**✅ Working well**")
            good = sg3[sg3["avg"] > 0].sort_values("avg", ascending=False).head(5) if not sg3.empty else sg3
            if good is not None and len(good):
                st.markdown(hbars_html([(r["Sector"], float(r["avg"]), f"{r['avg']:+.1f}% · {int(r['n'])} trades") for _, r in good.iterrows()]), unsafe_allow_html=True)
            else:
                st.caption("No sector has a clear profit yet (needs 3+ trades each).")
        with rc:
            st.markdown("**⚠️ Struggling**")
            bad = sg3[sg3["avg"] < 0].sort_values("avg").head(5) if not sg3.empty else sg3
            if bad is not None and len(bad):
                st.markdown(hbars_html([(r["Sector"], float(r["avg"]), f"{r['avg']:+.1f}% · {int(r['n'])} trades") for _, r in bad.iterrows()]), unsafe_allow_html=True)
            else:
                st.caption("No sector is clearly losing (or not enough trades yet).")

        sec_title("📈", "Is it getting better?", "win rate of the last 10 trades")
        chart_rolling(perf_eq_df, 10)
        st.caption("A line trending up means the AI is improving. Flat or falling means it isn't (yet).")

        sec_title("🧷", "Stocks it remembers", "lessons fade by half every 30 days")
        careful, liked = compute_memory(closed_all)
        mc1, mc2 = st.columns(2)
        with mc1:
            st.markdown("**🚧 Being extra careful with**")
            if careful:
                st.markdown(hbars_html([(t, -float(p), f"-{p} pts · ~{d}d left") for t, p, d in careful[:6]]), unsafe_allow_html=True)
            else:
                st.caption("None right now.")
        with mc2:
            st.markdown("**👍 Slightly more confident about**")
            if liked:
                st.markdown(hbars_html([(t, float(p), f"+{p} pts · ~{d}d left") for t, p, d in liked[:6]]), unsafe_allow_html=True)
            else:
                st.caption("None right now.")
        st.caption("“pts” are points added to or taken from the AI's confidence for that stock when it next looks at it.")

    sec_title("🎚️", "Can you trust its confidence number?")
    cal_rows = calibration_rows(closed_all)
    if len(cal_rows) >= 10:
        avg_said = float(np.mean([c for c, _ in cal_rows]))
        actual = float(np.mean([h for _, h in cal_rows]) * 100.0)
        if actual < avg_said - 8:
            callout("warn", "🎚️", f"On average the AI said it was <b>{avg_said:.0f}% sure</b> but was right <b>{actual:.0f}%</b> of the time — it has been overconfident. The app already scales its confidence down using your real results.")
        elif actual > avg_said + 8:
            callout("good", "🎚️", f"On average the AI said it was <b>{avg_said:.0f}% sure</b> and was right <b>{actual:.0f}%</b> of the time — it has been a little modest.")
        else:
            callout("good", "🎚️", f"On average the AI said it was <b>{avg_said:.0f}% sure</b> and was right <b>{actual:.0f}%</b> of the time — its confidence is a fair guide.")
        chart_calibration(cal_rows)
        st.caption("If the green bars are about as tall as the purple ones, the confidence number is trustworthy.")
    else:
        st.info("This appears once at least 10 trades have clearly won or lost.")

    sec_title("🔧", "The AI itself")
    state = _load_retrain_state()
    cal = st.session_state.get("cal_info")
    st.markdown(
        f"- **Version in use:** {st.session_state.get('ml_model_source', 'not loaded yet')}\n"
        f"- **Last upgrade:** {state.get('last_promoted', 'never — still the original')}\n"
        f"- **Upgrade rule:** only after {MIN_SETTLED_TO_RETRAIN}+ clear win/loss trades, and only if the new version tests better than the old one. Your original AI file is never overwritten."
        + (f"\n- **Confidence tuning:** based on {cal['decided']} trades that clearly won or lost." if cal else ""))

    with st.expander("How the AI picks a share"):
        st.markdown(
            "1. Looks at around 35 UK shares and skips tiny or hard-to-trade ones.\n"
            "2. Scores each one on how likely it is to reach its goal before its safety exit.\n"
            "3. Adjusts that score using lessons from past trades and the oil, copper and £ moves this week.\n"
            "4. Blocks any share with bad news (share sales, profit warnings and similar).\n"
            f"5. Only keeps shares with **{MIN_QUALIFY_CONFIDENCE}%+ confidence** and at least **{MIN_NET_RETURN_PCT:.0f}% possible profit after costs**.")

    if not closed_all.empty:
        with st.expander("Every finished trade and what was learned"):
            ca = closed_all.copy()
            ca["_k"] = ca.apply(lambda x: _first_valid(x.get("exit_timestamp"), x.get("last_audited")), axis=1)
            ca = ca.sort_values("_k", ascending=False)
            show_all = st.toggle("Show every finished trade (default: latest 25)", value=False)
            for _, c_row in (ca if show_all else ca.head(25)).iterrows():
                o_ = classify_outcome(c_row['status'])
                lesson = {"LOSS": "More cautious about this stock for ~30 days.", "WIN": "Slightly more confident about this stock for ~30 days.",
                          "BE": "Safety exit protected the money — no change.", "EXPIRED": "Timed out — nothing learned."}.get(o_, "")
                st.markdown(f"**{c_row['ticker']}** ({c_row['date_str']}) · {friendly_status(c_row['status'])} · {float(c_row['pnl_pct']):+.2f}% — _{lesson}_")

st.markdown('<div class="disc">Practice tool using pretend money. Not financial advice. Past results don’t predict future results.</div>', unsafe_allow_html=True)