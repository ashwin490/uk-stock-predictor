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
            st.session_state["retrain_notice"] = f"🧠 Challenger PROMOTED (holdout logloss {ll_ch:.3f} vs {ll_cm:.3f}, AUC {auc_ch:.2f} vs {auc_cm:.2f}, n={n})."
        else:
            st.session_state["retrain_notice"] = f"🧪 Challenger rejected — champion kept (logloss {ll_ch:.3f} vs {ll_cm:.3f}, AUC {auc_ch:.2f} vs {auc_cm:.2f}, n={n})."
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
        reason = f"Open-position cap reached ({n_open}/{MAX_OPEN_POSITIONS})"
    elif risk >= MAX_OPEN_RISK_GBP:
        reason = f"Open-risk cap reached (£{risk:.0f}/£{MAX_OPEN_RISK_GBP:.0f})"
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
        return fail("Entry window closed (08:30–16:00 London, Mon–Fri)")
    if not sig.get("Loggable", True):
        return fail("No live intraday price for today — not loggable")
    if is_market_stress(get_cross_asset_macro_regime()):
        return fail("Market-stress brake active (FTSE 5D drawdown)")
    cap_state = get_portfolio_capacity()
    if not cap_state["ok"]:
        return fail(cap_state["reason"])

    sector = str(sig.get('Sector') or get_ticker_sector(ticker))
    if enforce_sector_cap:
        sec_counts = get_active_sector_exposure()
        if sec_counts.get(sector, 0) >= MAX_ACTIVE_PER_SECTOR:
            return fail(f"Sector cap reached for {sector}")

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
                return fail("Daily quota filled")

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
                return fail("Already held or already logged today")
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
# 10. UI THEME, COMPONENTS & CHARTS
# ==============================================================================
CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
:root{--a1:#6366f1;--a2:#06b6d4;--good:#16a34a;--bad:#dc2626;--warn:#d97706;--line:rgba(128,128,128,.28);--soft:rgba(128,128,128,.08);}
html,body,.stApp{font-family:'Inter',system-ui,-apple-system,sans-serif;}
.block-container{padding-top:1.2rem;max-width:1500px;}
footer{visibility:hidden;}
.hero{border:1px solid var(--line);border-radius:18px;padding:20px 24px;background:linear-gradient(135deg,rgba(99,102,241,.20),rgba(6,182,212,.12));margin-bottom:14px;}
.hero h1{margin:0;font-size:1.9rem;font-weight:800;letter-spacing:-.02em;background:linear-gradient(90deg,#6366f1,#06b6d4);-webkit-background-clip:text;-webkit-text-fill-color:transparent;}
.hero .sub{opacity:.75;font-size:.85rem;margin-top:4px;}
.chips{display:flex;flex-wrap:wrap;gap:8px;margin-top:12px;}
.chip{border:1px solid var(--line);background:var(--soft);padding:4px 11px;border-radius:999px;font-size:.78rem;font-weight:600;white-space:nowrap;}
.chip.up{color:var(--good);border-color:rgba(22,163,74,.5);}
.chip.down{color:var(--bad);border-color:rgba(220,38,38,.5);}
.chip.warn{color:var(--warn);border-color:rgba(217,119,6,.5);}
.tcard{border:1px solid var(--line);border-radius:16px;padding:16px 18px;background:var(--soft);transition:transform .15s ease,box-shadow .15s ease;margin-bottom:12px;}
.tcard:hover{transform:translateY(-3px);box-shadow:0 10px 28px rgba(0,0,0,.18);}
.tcard-top{display:flex;justify-content:space-between;gap:6px;flex-wrap:wrap;margin-bottom:6px;}
.tag{font-size:.66rem;font-weight:700;letter-spacing:.05em;text-transform:uppercase;padding:3px 9px;border-radius:999px;border:1px solid var(--line);}
.tag.good{color:var(--good);border-color:rgba(22,163,74,.5);}
.tag.bad{color:var(--bad);border-color:rgba(220,38,38,.5);}
.tag.warn{color:var(--warn);border-color:rgba(217,119,6,.5);}
.tag.info{color:var(--a1);border-color:rgba(99,102,241,.5);}
.row{display:flex;justify-content:space-between;align-items:center;gap:10px;}
.tk{font-size:1.5rem;font-weight:800;letter-spacing:-.01em;}
.pnl{font-size:1.8rem;font-weight:800;line-height:1.1;text-align:right;}
.pnl.pos{color:var(--good);}
.pnl.neg{color:var(--bad);}
.sub2{font-size:.76rem;opacity:.7;}
.rbar{position:relative;height:10px;border-radius:999px;background:linear-gradient(90deg,rgba(220,38,38,.55),rgba(217,119,6,.45),rgba(22,163,74,.6));margin:16px 0 4px;}
.rdot{position:absolute;top:-4px;width:18px;height:18px;border-radius:50%;background:#fff;border:3px solid var(--a1);transform:translateX(-50%);box-shadow:0 2px 8px rgba(0,0,0,.35);}
.rtick{position:absolute;top:-3px;width:2px;height:16px;background:rgba(128,128,128,.9);transform:translateX(-50%);}
.rlab{display:flex;justify-content:space-between;font-size:.72rem;opacity:.7;}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:6px 14px;margin-top:10px;font-size:.8rem;}
.grid2 b{opacity:.6;font-weight:600;margin-right:4px;}
.ringwrap{position:relative;width:64px;height:64px;flex:none;}
.ring{width:64px;height:64px;border-radius:50%;-webkit-mask:radial-gradient(farthest-side,transparent 70%,#000 72%);mask:radial-gradient(farthest-side,transparent 70%,#000 72%);}
.ringnum{position:absolute;inset:0;display:grid;place-items:center;font-weight:800;font-size:.95rem;}
.note{font-size:.74rem;opacity:.75;margin-top:8px;}
[data-testid="stMetric"]{border:1px solid var(--line);border-radius:14px;padding:12px 14px;background:var(--soft);}
button[data-baseweb="tab"]{font-weight:600;}
.stButton>button{border-radius:12px;font-weight:600;}
"""

def inject_css():
    st.markdown(f"<style>{_collapse(CSS)}</style>", unsafe_allow_html=True)

def outcome_style(status) -> tuple:
    o = classify_outcome(status)
    return {"WIN": ("TARGET HIT", "good"), "LOSS": ("STOPPED OUT", "bad"), "BE": ("BREAK-EVEN EXIT", "warn"),
            "EXPIRED": ("TIME EXIT", "warn"), "ACTIVE": ("LIVE", "info")}.get(o, (str(status)[:14], "info"))

def price_bar_html(entry: float, target: float, stop: float, live: float) -> str:
    span = target - stop
    if span <= 0:
        return ""
    def pos(v):
        return max(0.0, min(100.0, (v - stop) / span * 100.0))
    return _collapse(f"""
        <div class="rbar"><div class="rtick" style="left:{pos(entry):.1f}%"></div><div class="rdot" style="left:{pos(live):.1f}%"></div></div>
        <div class="rlab"><span>🛑 {stop:.2f}p</span><span>Entry {entry:.2f}p</span><span>🎯 {target:.2f}p</span></div>
    """)

def trade_card_html(slot, ticker, sector, status, entry, target, stop, live, pnl_pct, shares, capital,
                    conf, base_c, adj, tax_tag, friction, audited, be_locked) -> str:
    s_text, s_kind = outcome_style(status)
    cls = "pos" if pnl_pct >= 0 else "neg"
    be_tag = '<span class="tag warn">🛡️ BE+TAX LOCK</span>' if be_locked and classify_outcome(status) == "ACTIVE" else ""
    return _collapse(f"""
    <div class="tcard">
      <div class="tcard-top"><span class="tag info">Slot #{esc(slot)}</span><span class="tag">{esc(sector)}</span><span class="tag {s_kind}">{esc(s_text)}</span>{be_tag}</div>
      <div class="row"><div><div class="tk">{esc(ticker)}</div><div class="sub2">Entry AI score {esc(conf)}% (base {esc(base_c)}% · adj {adj:+d}%)</div></div>
      <div><div class="pnl {cls}">{pnl_pct:+.2f}%</div><div class="sub2" style="text-align:right">net of costs</div></div></div>
      {price_bar_html(entry, target, stop, live)}
      <div class="grid2">
        <div><b>Last</b>{live:.2f}p <span class="sub2">(~15m delayed)</span></div><div><b>Size</b>{int(shares)} sh · £{capital:.0f}</div>
        <div><b>Tax</b>{esc(tax_tag)}</div><div><b>Friction</b>-{friction:.2f}%</div>
      </div>
      <div class="note">🕒 Audited {esc(str(audited)[:19])}</div>
    </div>
    """)

def signal_card_html(rank, row, loggable) -> str:
    conf = int(row["Adjusted Score"])
    colour = "#16a34a" if conf >= 75 else ("#6366f1" if conf >= 65 else "#d97706")
    log_tag = '<span class="tag good">Entry window open</span>' if loggable else '<span class="tag warn">Watchlist only</span>'
    return _collapse(f"""
    <div class="tcard">
      <div class="tcard-top"><span class="tag info">🔥 Pick #{rank}</span><span class="tag">{esc(row['Sector'])}</span>{log_tag}</div>
      <div class="row"><div><div class="tk">{esc(row['Ticker'])}</div><div class="sub2">{esc(row['Tax Regime'])} · {esc(row['Friction'])} friction</div></div>
      <div class="ringwrap"><div class="ring" style="background:conic-gradient({colour} {conf}%,rgba(128,128,128,.25) 0)"></div><div class="ringnum">{conf}%</div></div></div>
      <div class="pnl pos" style="text-align:left;margin-top:8px">{esc(row['Expected Return'])}</div><div class="sub2">target gain after tax, spread &amp; commission</div>
      {price_bar_html(float(row['Price (p)']), float(row['Target (p)']), float(row['Stop Loss (p)']), float(row['Price (p)']))}
      <div class="grid2">
        <div><b>Calibrated base</b>{esc(row['Base ML'])}</div><div><b>Adj</b>{esc(row['Learner Delta'])}</div>
        <div><b>Size</b>{esc(row['Recommended Shares'])}</div><div><b>Liquidity</b>£{row.get('Liquidity (£m/day)', 0)}m/day</div>
      </div>
      <div class="note">🧠 {esc(row['Learner Note'])}</div>
    </div>
    """)

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
        st.line_chart(pd.DataFrame({"Cumulative £": cum.values}, index=x))
        return
    colour = "#16a34a" if cum.iloc[-1] >= 0 else "#dc2626"
    fig = go.Figure(go.Scatter(x=x, y=cum, mode="lines+markers", line=dict(color=colour, width=3, shape="hv"),
                               fill="tozeroy", name="Cumulative net P&L (£)",
                               hovertemplate="%{x}<br>£%{y:,.2f}<extra></extra>"))
    show_fig(_base_layout(fig, 320, title))

def chart_outcome_donut(perf: dict):
    labels = ["Target hit", "Stopped out", "Break-even", "Time exit"]
    vals = [perf["wins"], perf["losses"], perf["be"], perf["expired"]]
    if not PLOTLY_OK:
        st.bar_chart(pd.DataFrame({"count": vals}, index=labels))
        return
    fig = go.Figure(go.Pie(labels=labels, values=vals, hole=0.62, sort=False,
                           marker=dict(colors=["#16a34a", "#dc2626", "#d97706", "#64748b"]), textinfo="label+percent"))
    fig.update_layout(height=320, margin=dict(l=10, r=10, t=44, b=10), title="Outcome mix (all closed trades)", showlegend=False)
    show_fig(fig)

def chart_pnl_hist(perf_df: pd.DataFrame):
    if perf_df.empty:
        return
    if not PLOTLY_OK:
        st.bar_chart(perf_df["pnl_pct"].round(1).value_counts().sort_index())
        return
    fig = go.Figure(go.Histogram(x=perf_df["pnl_pct"], nbinsx=24, marker=dict(color="#6366f1"), hovertemplate="%{x:.1f}% : %{y} trades<extra></extra>"))
    fig.add_vline(x=0, line_dash="dash", line_color="gray")
    fig.update_layout(height=320, margin=dict(l=10, r=10, t=44, b=10), title="Net P&L per trade (%)", bargap=0.05)
    show_fig(fig)

def chart_sector_exposure(exp: dict):
    if not exp:
        return
    items = sorted(exp.items(), key=lambda x: x[1])
    if not PLOTLY_OK:
        st.bar_chart(pd.DataFrame({"active": [v for _, v in items]}, index=[k for k, _ in items]))
        return
    fig = go.Figure(go.Bar(x=[v for _, v in items], y=[k for k, _ in items], orientation="h",
                           marker=dict(color=["#dc2626" if v >= MAX_ACTIVE_PER_SECTOR else "#6366f1" for _, v in items]),
                           hovertemplate="%{y}: %{x} active<extra></extra>"))
    fig.add_vline(x=MAX_ACTIVE_PER_SECTOR, line_dash="dot", line_color="#d97706", annotation_text=f"cap {MAX_ACTIVE_PER_SECTOR}")
    fig.update_layout(height=max(220, 38 * len(items) + 60), margin=dict(l=10, r=10, t=40, b=10), title="Active sector exposure", xaxis=dict(dtick=1))
    show_fig(fig)

def chart_calibration(closed_df: pd.DataFrame):
    rows = []
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
    if len(rows) < 10:
        st.info("Calibration chart appears once ≥10 decided (win/loss) trades have a recorded entry score.")
        return
    d = pd.DataFrame(rows, columns=["conf", "hit"])
    d["bucket"] = pd.cut(d["conf"], bins=[0, 60, 70, 80, 101], labels=["<60", "60–70", "70–80", "80+"])
    g = d.groupby("bucket", observed=True).agg(stated=("conf", "mean"), realised=("hit", lambda s: s.mean() * 100.0), n=("hit", "size")).reset_index()
    if not PLOTLY_OK:
        st.dataframe(g, hide_index=True)
        return
    fig = go.Figure()
    fig.add_bar(x=g["bucket"].astype(str), y=g["stated"], name="Stated confidence", marker_color="#6366f1")
    fig.add_bar(x=g["bucket"].astype(str), y=g["realised"], name="Realised target-hit rate", marker_color="#16a34a",
                customdata=g["n"], hovertemplate="%{y:.0f}% realised (n=%{customdata})<extra></extra>")
    fig.update_layout(barmode="group", height=320, margin=dict(l=10, r=10, t=44, b=10), title="Confidence calibration: stated vs realised",
                      yaxis=dict(range=[0, 100], title="%"), legend=dict(orientation="h", y=1.14))
    show_fig(fig)

def chart_position(ticker: str, entry: float, target: float, stop: float, be_floor: float):
    h = fetch_cached_history(ticker, period="3mo")
    if h.empty:
        st.info("No price history available.")
        return
    if not PLOTLY_OK:
        st.line_chart(h["Close"])
        return
    xs = [str(d) for d in _bar_dates(h)]
    fig = go.Figure(go.Candlestick(x=xs, open=h["Open"], high=h["High"], low=h["Low"], close=h["Close"], name=ticker,
                                   increasing_line_color="#16a34a", decreasing_line_color="#dc2626"))
    for label, val, col in [("Target", target, "#16a34a"), ("Entry", entry, "#6366f1"), ("Break-even", be_floor, "#d97706"), ("Stop", stop, "#dc2626")]:
        fig.add_hline(y=val, line_dash="dash", line_color=col, annotation_text=f"{label} {val:.2f}p", annotation_position="right")
    fig.update_layout(height=420, margin=dict(l=10, r=70, t=30, b=10), xaxis_rangeslider_visible=False, hovermode="x unified")
    show_fig(fig)

def style_ledger(df: pd.DataFrame):
    """Colourised, numerically-sortable ledger; falls back to the plain frame if Styler is unavailable."""
    try:
        def pnl_color(v):
            try:
                return "color:#16a34a;font-weight:700" if float(v) > 0 else ("color:#dc2626;font-weight:700" if float(v) < 0 else "")
            except Exception:
                return ""
        sty = df.style.format({"Entry (p)": "{:.2f}p", "Target (p)": "{:.2f}p", "Stop (p)": "{:.2f}p", "Live Price (p)": "{:.2f}p", "Net P&L (%)": "{:+.2f}%"}, na_rep="-")
        mapper = sty.map if hasattr(sty, "map") else sty.applymap
        return mapper(pnl_color, subset=["Net P&L (%)"])
    except Exception:
        return df

# ==============================================================================
# 11. SIDEBAR & UNIFIED AUTONOMOUS LOOP CONTROLLER
# ==============================================================================
inject_css()

st.sidebar.header("⚙️ Institutional Guardrails")
selected_universe = st.sidebar.selectbox("Universe Mode", ["Rotating Active Market Basket (FTSE + AIM)"])
st.sidebar.number_input("Broker commission per trade (£)", min_value=0.0, max_value=25.0, value=DEFAULT_COMMISSION_GBP, step=0.5, key="commission_gbp",
                        help="Applied to both entry and exit. Rounds into every new signal's friction and net P&L.")
st.sidebar.caption(f"Daily Auto-Log Cap: **Top {MAX_DAILY_EQUITY_TRADES} Picks/Day**")
st.sidebar.caption(f"Sector Cap: **Max {MAX_ACTIVE_PER_SECTOR} Active/Sector** · Open cap: **{MAX_OPEN_POSITIONS} positions / £{MAX_OPEN_RISK_GBP:.0f} risk**")
st.sidebar.caption(f"Break-Even Ratchet: **≥ {int(BREAKEVEN_TRIGGER_RATIO * 100)}% of Target (Net of Costs)**")
st.sidebar.caption(f"Post-Exit Cooldown: **{COOLDOWN_CALENDAR_DAYS} Days** · Decay Half-Life: **{int(HALF_LIFE_DAYS)} Days**")
st.sidebar.caption("UK TCA: **0.50% SDRT (Main) / 0% (AIM) + liquidity-tiered spread + commission**")
st.sidebar.caption(f"Entry window: **{ENTRY_WINDOW_START.strftime('%H:%M')}–{ENTRY_WINDOW_END.strftime('%H:%M')} London** · Min liquidity: **£{MIN_AVG_TURNOVER_GBP/1e6:.0f}m/day**")

st.sidebar.markdown("---")
st.sidebar.header("🔌 Broker Execution Bridge")
broker_mode = st.sidebar.selectbox("Gateway", ["Paper Trading (Simulated)", "Interactive Brokers", "IG Group API"])

st.sidebar.markdown("---")
st.sidebar.header("🔄 Autonomous Loop")
market_is_open = is_lse_market_open()
auto_mode = st.sidebar.toggle("Continuous Background Mode", value=market_is_open)
refresh_interval_sec = st.sidebar.selectbox("Refresh Interval", [300, 600, 3600], format_func=lambda x: f"{x//60} Minutes")
show_sim_options = st.sidebar.toggle("Show simulated options in ledger", value=True)

loop_tick = 0
if auto_mode and st_autorefresh:
    loop_tick = st_autorefresh(interval=refresh_interval_sec * 1000, key="unified_autonomous_loop")

is_new_loop_tick = ("last_loop_tick" not in st.session_state) or (loop_tick != st.session_state["last_loop_tick"])

if is_new_loop_tick:
    audit_and_reconcile_all_trades()
    hydrate_duckdb_from_supabase()
    st.session_state["last_loop_tick"] = loop_tick

if "retrain_notice" in st.session_state:
    st.sidebar.success(st.session_state["retrain_notice"])
if "ml_model_source" in st.session_state:
    st.sidebar.caption(f"🤖 Model in use: **{st.session_state['ml_model_source']}**")

if "db_error" in st.session_state:
    st.sidebar.error(f"⚠️ Cloud Sync Warning: {st.session_state['db_error']}")
if st.session_state.get("pending_cloud_sync"):
    st.sidebar.warning(f"⏳ {len(st.session_state['pending_cloud_sync'])} cloud update(s) queued for retry.")

if st.sidebar.button("🚪 Log Out", width="stretch"):
    st.session_state["password_correct"] = False
    st.rerun()

market_status = "OPEN" if market_is_open else "CLOSED"
macro_bar = get_cross_asset_macro_regime()

def macro_chip(label, v):
    cls = "up" if v >= 0 else "down"
    arrow = "▲" if v >= 0 else "▼"
    return f'<span class="chip {cls}">{esc(label)} {arrow} {v:+.1f}%</span>'

stress_chip = '<span class="chip down">⛔ Market-stress brake ON</span>' if is_market_stress(macro_bar) else ""
st.markdown(_collapse(f"""
<div class="hero">
  <h1>⚡ ALPHA-LSE Quant Terminal</h1>
  <div class="sub">Autonomous UK equity screener · calibrated ML · UK tax-aware · paper-traded &amp; audited bar-by-bar</div>
  <div class="chips">
    <span class="chip {'up' if market_is_open else 'down'}">LSE {market_status}</span>
    <span class="chip {'up' if supabase else 'down'}">{'☁️ Supabase online' if supabase else '☁️ Supabase offline'}</span>
    {macro_chip('FTSE 5D', macro_bar['ftse_5d'])}{macro_chip('Brent', macro_bar['brent_5d'])}{macro_chip('Copper', macro_bar['copper_5d'])}{macro_chip('GBP/USD', macro_bar['gbpusd_5d'])}
    {stress_chip}
    <span class="chip">Cycle #{loop_tick} · every {refresh_interval_sec // 60}m</span>
  </div>
</div>
"""), unsafe_allow_html=True)

tab_scanner, tab_options, tab_journal, tab_perf, tab_reasoning = st.tabs([
    "🎯 Equity Signals",
    "📊 Leveraged Alpha (Simulated)",
    "📖 Trade Journal & Inspector",
    "📈 Performance Analytics",
    "🧠 AI Reasoning & Learning"
])

# ==============================================================================
# 12. TAB 1: EQUITY SCANNER
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
    c1.metric("Today's slots", f"{len(todays_logged_df)}/{MAX_DAILY_EQUITY_TRADES}")
    c2.metric("Open positions", f"{cap_state['open']}/{MAX_OPEN_POSITIONS}")
    c3.metric("Open risk", f"£{cap_state['risk_gbp']:.0f}/£{MAX_OPEN_RISK_GBP:.0f}")
    with c4:
        st.write("")
        re_scan = st.button("🔄 Run Live Scan Now", width="stretch", type="primary")

    if is_market_stress(macro_bar):
        st.error("⛔ Market-stress brake: FTSE 100 is down heavily over 5 days — new entries are paused (signals shown as watchlist).")
    elif not cap_state["ok"]:
        st.warning(f"🧯 {cap_state['reason']} — new entries paused until positions close.")
    elif not is_entry_window_open():
        st.info("🕗 Outside the entry window (08:30–16:00 London, Mon–Fri). Signals are shown as a watchlist and are NOT auto-logged, so every entry uses a real intraday price.")

    if quota_filled and not re_scan:
        st.info(f"🔒 **Daily quota filled ({len(todays_logged_df)}/{MAX_DAILY_EQUITY_TRADES})** — scanner locked onto today's cohort to prevent over-trading.")
        st.markdown("### 📌 Today's Executed Cohort (net of UK friction)")
        render_cohort(todays_logged_df)
    else:
        if re_scan or is_new_loop_tick or "scan_results" not in st.session_state:
            with st.spinner("Running quant screen, macro overlay & reconciling live trades..."):
                if re_scan:
                    audit_and_reconcile_all_trades()
                    hydrate_duckdb_from_supabase()
                res_df, has_cleared = run_predictions()
                st.session_state["scan_results"] = res_df
                st.session_state["has_cleared"] = has_cleared
                st.session_state["scan_ts"] = london_now().strftime("%H:%M:%S")

        if not todays_logged_df.empty:
            with st.expander(f"📌 Already logged today ({len(todays_logged_df)})", expanded=False):
                render_cohort(todays_logged_df)

        df_res = st.session_state.get("scan_results", pd.DataFrame())
        stats = st.session_state.get("scan_stats", {})
        if stats and "error" not in stats:
            st.caption(
                f"Last scan {st.session_state.get('scan_ts', '-')}: {stats.get('scanned', 0)} scored · "
                f"{stats.get('illiquid', 0)} filtered (price/liquidity) · {stats.get('rns_blocked', 0)} RNS-blocked · "
                f"{stats.get('sector_blocked', 0)} sector-capped · {stats.get('no_data', 0)} no data · auto-logged {stats.get('logged', 0)}")
        elif stats.get("error"):
            st.error(stats["error"])

        if st.session_state.get("has_cleared", False) and not df_res.empty:
            st.success(f"🟢 {len(df_res)} setup(s) cleared all gates (net of tax, spread, commission, caps)")
            st.markdown("### 🔥 Top Conviction Spotlights")
            cols = st.columns(min(len(df_res), 3))
            today_key = london_now().strftime('%Y-%m-%d')
            for idx, row in df_res.head(3).iterrows():
                with cols[idx % 3]:
                    st.markdown(signal_card_html(idx + 1, row, bool(row.get("Loggable", False))), unsafe_allow_html=True)
                    idem_token = hashlib.sha256(f"{row['Ticker']}_{today_key}_{broker_mode}".encode()).hexdigest()[:12]
                    already_sent = idem_token in st.session_state["dispatched_orders"]
                    btn_label = "✅ Order Dispatched (Idempotent Lock)" if already_sent else f"🚀 Execute Buy ({broker_mode})"
                    if st.button(btn_label, key=f"exec_{idem_token}", width="stretch", disabled=already_sent):
                        ok = log_equity_signal_safely(row.to_dict(), enforce_sector_cap=True)
                        if ok:
                            st.session_state["dispatched_orders"].add(idem_token)
                            st.session_state["flash"] = f"✅ Order [{idem_token}] for {row['Ticker']} dispatched to {broker_mode}."
                        else:
                            st.session_state["flash"] = f"⚠️ {row['Ticker']} not logged: {st.session_state.get('last_log_reason', 'blocked by a guardrail')}."
                        st.rerun()

            st.markdown(f"### 📋 All {len(df_res)} qualified setups (ranked by final AI score)")
            display_cols = ["Ticker", "Sector", "Tax Regime", "Price (p)", "Target (p)", "Stop Loss (p)", "Expected Return",
                            "Base ML", "Learner Delta", "AI Win Confidence", "Liquidity (£m/day)", "Recommended Shares", "RNS", "Loggable"]
            st.dataframe(df_res[[c for c in display_cols if c in df_res.columns]], width="stretch", hide_index=True)
        else:
            st.warning("🛡️ **Capital Protection Active:** no equities currently pass all liquidity, trend, sector, calibration and ML filters.")

# ==============================================================================
# 13. TAB 2: OPTIONS (SIMULATED)
# ==============================================================================
with tab_options:
    st.subheader("📊 FTSE Blue-Chip Leveraged Alpha — SIMULATED")
    st.caption("⚠️ Paper simulation only: Black-Scholes prices using 30-day realised volatility, an indicative dividend yield and a "
               f"±{OPTION_HALF_SPREAD_PCT:.0f}% bid/ask haircut. These are NOT real quotes and are excluded from headline P&L.")
    opt_signal = generate_daily_options_alpha()
    if opt_signal:
        with st.container(border=True):
            o1, o2, o3 = st.columns(3)
            with o1:
                st.metric("Simulated Contract", opt_signal["option_contract"])
                st.markdown(f"Entry spot: **{opt_signal['underlying_spot']:.2f}p**")
            with o2:
                st.metric("Simulated Bid Price", f"{opt_signal['current_option_price']:.2f}p", f"{opt_signal['pnl_pct']:+.2f}% vs ask entry ({opt_signal['entry_premium']:.2f}p)")
                st.markdown(f"30d realised vol: **{opt_signal['implied_vol']}%**")
            with o3:
                st.metric("Momentum Score (heuristic)", f"{opt_signal['ai_confidence']}", help="Derived from 5-day momentum. This is NOT a probability.")
                st.markdown(f"Capital required: **£{opt_signal['total_capital']:,.2f}** ({int(opt_signal['lot_size'])} units)")
            m1, m2, m3 = st.columns(3)
            m1.info(f"🎯 Target premium: {opt_signal['target_premium']:.2f}p (+60%)")
            m2.warning(f"🛑 Stop premium: {opt_signal['stop_loss_premium']:.2f}p (-50%)")
            m3.success(f"Status: **{opt_signal['status']}** (audited {str(opt_signal['last_audited'])[:16]})")
    else:
        st.info("No simulated option qualifies today (budget cap, or all candidate underlyings already have an active contract).")

# ==============================================================================
# 14. TAB 3: JOURNAL, FILTERS, POSITION INSPECTOR
# ==============================================================================
with tab_journal:
    st.subheader("📖 Master Ledger")

    j1, j2 = st.columns([4, 1])
    with j1:
        st.caption("Equity P&L is net of SDRT, spread and commission. Break-even ratchet at "
                   f"≥ {int(BREAKEVEN_TRIGGER_RATIO * 100)}% of target. Prices from Yahoo are typically ~15 min delayed.")
    with j2:
        if st.button("🔄 Force Manual Sync", width="stretch"):
            audit_and_reconcile_all_trades()
            hydrate_duckdb_from_supabase()
            st.toast("Ledger reconciled with live LSE data and Supabase.", icon="✅")
            st.rerun()

    df_eq = pd.DataFrame()
    df_opt = pd.DataFrame()
    with DB_LOCK:
        con = duckdb.connect(DB_PATH, read_only=True)
        try:
            df_eq = con.execute("SELECT * FROM trade_journal ORDER BY date_str DESC, ticker ASC").df()
            df_opt = con.execute("SELECT * FROM daily_options_journal ORDER BY date_key DESC").df()
        except Exception:
            pass
        finally:
            con.close()

    perf_eq_df = prep_perf_frame(df_eq, "capital_allocated")
    perf_opt_df = prep_perf_frame(df_opt, "total_capital", is_option=True)
    perf_eq = compute_performance(perf_eq_df)

    active_eq = df_eq[df_eq["status"].astype(str).str.upper() == "ACTIVE"] if not df_eq.empty else pd.DataFrame()
    active_cap = float(active_eq["capital_allocated"].sum()) if not active_eq.empty else 0.0
    open_unreal = float((active_eq["capital_allocated"] * active_eq["pnl_pct"] / 100.0).sum()) if not active_eq.empty else 0.0
    be_prot = int(sum(is_stop_breakeven_protected(float(r["entry_price"]), float(r["stop_loss"])) for _, r in active_eq.iterrows())) if not active_eq.empty else 0
    open_ret = (open_unreal / active_cap * 100.0) if active_cap > 0 else 0.0

    k1, k2, k3, k4 = st.columns(4)
    if perf_eq["n"] > 0:
        k1.metric("Target-hit rate (all closed equities)", f"{perf_eq['target_hit_rate']:.1f}%",
                  f"{perf_eq['wins']}W · {perf_eq['losses']}L · {perf_eq['be']}BE · {perf_eq['expired']}EXP", delta_color="off")
        k4.metric("Closed net realised (equities)", f"£{perf_eq['net_gbp']:+,.2f}", f"{perf_eq['n']} settled trades", delta_color="off")
    else:
        k1.metric("Target-hit rate", "–", "no closed equities yet", delta_color="off")
        k4.metric("Closed net realised (equities)", "£+0.00", "0 settled trades", delta_color="off")
    k2.metric("Active equity capital", f"£{active_cap:,.0f}", f"🛡️ {be_prot} break-even protected", delta_color="off")
    k3.metric("Open net unrealised (equities)", f"£{open_unreal:+,.2f}", f"{open_ret:+.2f}% on active capital")

    sec_exp = get_active_sector_exposure()
    if sec_exp:
        with st.expander("📊 Active sector exposure", expanded=False):
            chart_sector_exposure(sec_exp)

    # ---- Position inspector ----
    if not active_eq.empty:
        st.markdown("#### 🔬 Position Inspector")
        sel = st.selectbox("Select an active position", list(active_eq["ticker"]), key="inspect_ticker")
        r = active_eq[active_eq["ticker"] == sel].iloc[0]
        fr = get_trade_friction_pct(sel, r["features_json"])
        ic1, ic2, ic3, ic4 = st.columns(4)
        ic1.metric("Entry", f"{r['entry_price']:.2f}p")
        ic2.metric("Live", f"{r['latest_price']:.2f}p", f"{r['pnl_pct']:+.2f}% net")
        ic3.metric("Target", f"{r['target_price']:.2f}p")
        ic4.metric("Stop", f"{r['stop_loss']:.2f}p", "🛡️ BE-locked" if is_stop_breakeven_protected(float(r["entry_price"]), float(r["stop_loss"])) else "initial ATR stop", delta_color="off")
        chart_position(sel, float(r["entry_price"]), float(r["target_price"]), float(r["stop_loss"]), get_breakeven_exit_price(float(r["entry_price"]), fr))

    # ---- Ledger with filters ----
    master_list = []
    if not df_eq.empty:
        d = df_eq.copy()
        d["Sector"] = d.apply(lambda x: get_trade_sector(x["ticker"], x["features_json"]), axis=1)
        d["Asset"] = "EQUITY"
        d["Status"] = d.apply(lambda x: "🟢 ACTIVE (🛡️ BE STOP)" if str(x["status"]) == "ACTIVE" and is_stop_breakeven_protected(float(x["entry_price"]), float(x["stop_loss"])) else str(x["status"]), axis=1)
        master_list.append(d.rename(columns={"date_str": "Date", "ticker": "Symbol", "entry_price": "Entry (p)", "target_price": "Target (p)",
                                             "stop_loss": "Stop (p)", "latest_price": "Live Price (p)", "pnl_pct": "Net P&L (%)", "last_audited": "Last Checked"}))
    if show_sim_options and not df_opt.empty:
        d = df_opt.copy()
        d["Asset"] = "OPTIONS (SIM)"
        d["Sector"] = d["share_name"].apply(lambda s: get_ticker_sector(f"{s}.L", resolve=False))
        d["Status"] = d["status"]
        master_list.append(d.rename(columns={"date_key": "Date", "option_contract": "Symbol", "entry_premium": "Entry (p)",
                                             "current_option_price": "Live Price (p)", "target_premium": "Target (p)",
                                             "stop_loss_premium": "Stop (p)", "pnl_pct": "Net P&L (%)", "last_audited": "Last Checked"}))

    if master_list:
        cols_keep = ["Date", "Symbol", "Sector", "Asset", "Entry (p)", "Target (p)", "Stop (p)", "Live Price (p)", "Net P&L (%)", "Status", "Last Checked"]
        master_df = pd.concat([m[[c for c in cols_keep if c in m.columns]] for m in master_list], ignore_index=True)
        master_df = master_df.sort_values("Date", ascending=False, kind="stable").reset_index(drop=True)

        st.markdown("#### 🔎 Filter the ledger")
        f1, f2, f3, f4 = st.columns([2, 2, 2, 2])
        status_opts = ["Active", "Win", "Loss", "Break-even", "Time exit"]
        chosen_status = f1.multiselect("Outcome", status_opts, default=status_opts)
        asset_opts = sorted(master_df["Asset"].unique())
        chosen_assets = f2.multiselect("Asset", asset_opts, default=asset_opts)
        sector_opts = sorted(master_df["Sector"].astype(str).unique())
        chosen_sectors = f3.multiselect("Sector", sector_opts, default=sector_opts)
        search = f4.text_input("Search symbol", "")

        label_map = {"Active": "ACTIVE", "Win": "WIN", "Loss": "LOSS", "Break-even": "BE", "Time exit": "EXPIRED"}
        wanted = {label_map[s] for s in chosen_status}
        view = master_df[master_df["Status"].apply(classify_outcome).isin(wanted) & master_df["Asset"].isin(chosen_assets) & master_df["Sector"].astype(str).isin(chosen_sectors)]
        if search.strip():
            view = view[view["Symbol"].astype(str).str.contains(search.strip(), case=False, na=False)]

        st.caption(f"Showing **{len(view)}** of {len(master_df)} rows")
        st.dataframe(style_ledger(view), width="stretch", hide_index=True)
        st.download_button("⬇️ Download filtered ledger (CSV)", view.to_csv(index=False).encode("utf-8"), "alpha_lse_ledger.csv", "text/csv")
    else:
        st.info("No trades logged yet. Positions appear here as the engine confirms signals inside the entry window.")

# ==============================================================================
# 15. TAB 4: PERFORMANCE ANALYTICS
# ==============================================================================
with tab_perf:
    st.subheader("📈 Performance Analytics — equities only (options shown separately, simulated)")
    if perf_eq["n"] == 0:
        st.info("No closed equity trades yet.")
    else:
        if perf_eq["n"] < 100:
            st.warning(f"Small sample: only {perf_eq['n']} closed trades. Treat every number below as provisional — "
                       "a few months of data are needed before drawing conclusions.")
        a1, a2, a3, a4 = st.columns(4)
        a1.metric("Closed trades", perf_eq["n"])
        a2.metric("Target-hit rate", f"{perf_eq['target_hit_rate']:.1f}%", f"95% CI {perf_eq['ci_low']:.0f}–{perf_eq['ci_high']:.0f}%", delta_color="off")
        a3.metric("Profitable-trade rate", f"{perf_eq['profitable_rate']:.1f}%", help="Share of ALL closed trades with positive net P&L (time exits included).")
        a4.metric("Expectancy / trade", f"{perf_eq['expectancy_pct']:+.2f}%", f"£{perf_eq['expectancy_gbp']:+.2f}")
        b1, b2, b3, b4 = st.columns(4)
        b1.metric("Avg win", f"£{perf_eq['avg_win_gbp']:+.2f}")
        b2.metric("Avg loss", f"£{perf_eq['avg_loss_gbp']:+.2f}")
        b3.metric("Profit factor", f"{perf_eq['profit_factor']:.2f}" if perf_eq["profit_factor"] is not None else "∞")
        b4.metric("Max drawdown", f"£{perf_eq['max_dd_gbp']:+.2f}")

        ftse_s = get_ftse_close_series()
        first_d = perf_eq_df["entry_date"].dropna().min() if not perf_eq_df.empty else None
        ftse_since = ftse_return_between(ftse_s, first_d, london_now().date()) if first_d else None
        excess = []
        for _, rr in perf_eq_df.iterrows():
            fr_ = ftse_return_between(ftse_s, rr["entry_date"], rr["exit_date"])
            if fr_ is not None:
                excess.append(float(rr["pnl_pct"]) - fr_)
        c1, c2 = st.columns(2)
        c1.metric("FTSE 100 since first trade", f"{ftse_since:+.2f}%" if ftse_since is not None else "n/a")
        c2.metric("Avg excess vs FTSE over same holds", f"{np.mean(excess):+.2f}% / trade" if excess else "n/a",
                  help="Each trade's net return minus the FTSE 100 return over the same holding window — a quick beta check.")

        ch1, ch2 = st.columns([3, 2])
        with ch1:
            chart_cumulative_pnl(perf_eq_df, "Cumulative net realised P&L — equities (£)")
        with ch2:
            chart_outcome_donut(perf_eq)
        ch3, ch4 = st.columns(2)
        with ch3:
            chart_pnl_hist(perf_eq_df)
        with ch4:
            sec_tbl = perf_eq_df.copy()
            sec_tbl["Sector"] = sec_tbl.apply(lambda x: get_trade_sector(x["ticker"], x["features_json"]), axis=1)
            grp = sec_tbl.groupby("Sector").agg(Trades=("pnl_gbp", "size"), Net_GBP=("pnl_gbp", "sum"), Avg_pct=("pnl_pct", "mean"),
                                                Hit_rate=("outcome", lambda s: (s == "WIN").mean() * 100.0)).round(2).reset_index()
            st.markdown("**By sector**")
            st.dataframe(grp.sort_values("Net_GBP", ascending=False), width="stretch", hide_index=True)

    with st.expander("🧪 Simulated options (not real fills — excluded from headline numbers)"):
        perf_opt = compute_performance(perf_opt_df)
        if perf_opt["n"] == 0:
            st.write("No closed simulated options yet.")
        else:
            o1, o2, o3 = st.columns(3)
            o1.metric("Closed (sim)", perf_opt["n"])
            o2.metric("Target-hit rate (sim)", f"{perf_opt['target_hit_rate']:.1f}%")
            o3.metric("Net P&L (sim)", f"£{perf_opt['net_gbp']:+,.2f}")
            chart_cumulative_pnl(perf_opt_df, "Cumulative simulated option P&L (£)")

# ==============================================================================
# 16. TAB 5: AI REASONING & SELF-LEARNING
# ==============================================================================
with tab_reasoning:
    st.subheader("🧠 Explainable AI, UK TCA & Governed Self-Learning")
    cal = st.session_state.get("cal_info")
    if cal:
        st.caption(f"Confidence calibration mode: **{cal['mode']}** · realised base rate (decided trades): **{cal['prior']:.0%}**")
    st.caption(f"Retraining policy: challenger trained only after ≥{MIN_SETTLED_TO_RETRAIN} decided trades, validated walk-forward, promoted only if it beats the champion. The original model file is never overwritten.")

    df_scan = st.session_state.get("scan_results", pd.DataFrame())
    todays_eq = get_todays_logged_equities()

    st.markdown("### 🔍 Live Equity Decision Matrix")
    if not df_scan.empty:
        r_cols = st.columns(min(len(df_scan), 3))
        for idx, row in df_scan.head(3).iterrows():
            with r_cols[idx % 3]:
                with st.container(border=True):
                    st.markdown(f"#### #{idx+1} {row['Ticker']} ({row['Sector']})")
                    st.write(f"**Final AI Confidence:** `{row['AI Win Confidence']}` *(Calibrated base: {row['Base ML']}, adj: {row['Learner Delta']})*")
                    st.progress(min(float(str(row['AI Win Confidence']).replace('%', '')) / 100.0, 1.0))
                    st.markdown(f"""
* 🤖 **Calibrated ML ensemble:** LightGBM + CatBoost (`{row['Base ML']}`)
* 🧠 **Memory & macro:** `{row['Learner Note']}`
* 🏛️ **UK TCA:** `{row['Tax Regime']}` (`{row['Friction']}` incl. commission)
* 📰 **RNS NLP gate:** `{row['RNS']}`
* 💧 **Liquidity:** £{row.get('Liquidity (£m/day)', 0)}m/day
* ⚖️ **Capital:** £500 (`{row['Recommended Shares']}`)
""")
    elif not todays_eq.empty:
        r_cols = st.columns(min(len(todays_eq), 3))
        for pos_i, (_, t_row) in enumerate(todays_eq.head(3).iterrows()):
            with r_cols[pos_i % 3]:
                with st.container(border=True):
                    tkr_sym = str(t_row["ticker"])
                    f_meta = _parse_features(t_row.get("features_json"))
                    base_c = int(f_meta.get("base_ml_conf", 75))
                    del_c = int(f_meta.get("learner_delta", 0)) + int(f_meta.get("macro_delta", 0))
                    final_c = base_c + del_c
                    sec_name = get_trade_sector(tkr_sym, f_meta)
                    tax_tag = f_meta.get("tax_regime") or ("AIM (0% SDRT)" if is_aim_exempt(tkr_sym) else "Main (0.5% SDRT)")
                    be_locked = is_stop_breakeven_protected(float(t_row["entry_price"]), float(t_row["stop_loss"]), tkr_sym)
                    st.markdown(f"#### #{pos_i+1} {tkr_sym} ({sec_name})")
                    st.write(f"**Entry AI confidence:** `{final_c}%` *(base {base_c}%, adj {del_c:+d}%)*")
                    st.progress(min(max(final_c / 100.0, 0.0), 1.0))
                    st.markdown(f"""
* 🤖 **Calibrated ML ensemble:** LightGBM + CatBoost (`{base_c}%`)
* 🏛️ **UK TCA:** `{tax_tag}` (`-{get_trade_friction_pct(tkr_sym, f_meta):.2f}%` total friction)
* 🧠 **Recorded ATR regime:** `{f_meta.get('atr_pct', 'N/A')}%`
* 🛡️ **Stop protection:** `{'Break-even + tax locked' if be_locked else 'Initial ATR stop'}`
* ⚖️ **Capital:** £{t_row['capital_allocated']:.0f} (`{int(t_row['shares'])} shares`)
""")
    else:
        st.info("No active equity signals to analyse. Run the live scan on the Equity tab first.")

    st.markdown("---")
    st.markdown("### 🎚️ Is the confidence score honest?")
    closed_all = load_closed_trades_df()
    if not closed_all.empty:
        chart_calibration(closed_all)
    else:
        st.info("Calibration chart appears after the first closed trades.")

    st.markdown("---")
    st.markdown("### 📊 Simulated Option Reasoning")
    opt_sig = generate_daily_options_alpha()
    if opt_sig:
        with st.container(border=True):
            d1, d2 = st.columns([1, 2])
            with d1:
                st.metric("Contract evaluated", opt_sig["option_contract"])
                st.write(f"**30d realised vol:** `{opt_sig['implied_vol']}%`")
                st.write(f"**Entry spot:** `{opt_sig['underlying_spot']:.2f}p`")
                st.write(f"**Strike (OTM):** `{opt_sig['strike_price']:.0f}p`")
            with d2:
                st.markdown(f"""
**Simulated pricing model**
* 📐 Black-Scholes with continuous dividend yield, r = 5% and a ±{OPTION_HALF_SPREAD_PCT:.0f}% bid/ask haircut (ask on entry, bid on exit).
* 🎯 Strike +2% out-of-the-money; one active contract per underlying to avoid stacking a single correlated bet.
* 🛡️ Budget under £500; stop at -50% premium; tie-break assumes the stop is hit first.
* ⚠️ Not real market quotes — retail UK single-stock options are thin, so live execution would differ.
""")

    st.markdown("---")
    st.markdown("### 🛑 Post-Trade Autopsies (Active Memory Bank)")
    if not closed_all.empty:
        closed_all = closed_all.copy()
        closed_all["_k"] = closed_all.apply(lambda x: _first_valid(x.get("exit_timestamp"), x.get("last_audited")), axis=1)
        closed_all = closed_all.sort_values("_k", ascending=False)
        show_all = st.toggle("Show all closed trades", value=False)
        for _, c_row in (closed_all if show_all else closed_all.head(25)).iterrows():
            with st.expander(f"{c_row['status']} | {c_row['ticker']} ({c_row['date_str']}) — Net P&L: {float(c_row['pnl_pct']):+.2f}%"):
                st.write(f"**Entry:** `{float(c_row['entry_price']):.2f}p` | **Sector:** `{get_trade_sector(c_row['ticker'], c_row['features_json'])}`")
                st.code(f"Recorded Feature Vector: {c_row['features_json']}", language="json")
                if "LOSS" in str(c_row['status']):
                    st.error("Active rule: future setups on this ticker receive a 30-day exponentially decayed penalty (-15% initial), and ATR feeds the 75th-percentile volatility floor.")
                elif "BREAK-EVEN" in str(c_row['status']):
                    st.info("Active rule: capital and UK stamp duty preserved via the break-even ratchet. Neutral memory weight.")
                elif "WIN" in str(c_row['status']):
                    st.success("Active rule: future setups on this ticker receive a 30-day exponentially decayed boost (+5% initial).")
                else:
                    st.info("Time exit: no memory adjustment applied.")
    else:
        st.success("🏆 **No closed trades in memory yet.**\n\nAs trades close, their feature vectors are stored and used for calibration, memory adjustments and governed retraining.")