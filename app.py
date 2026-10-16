import os
import re
import math
import json
import time
import hmac
import hashlib
import threading
import warnings
import shutil
from pathlib import Path
from datetime import datetime, time as dtime, timedelta
import random
import requests
from collections import defaultdict

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import yfinance as yf
import pytz
import duckdb

try:
    from streamlit_autorefresh import st_autorefresh
except ImportError:
    st_autorefresh = None

try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

warnings.filterwarnings("ignore")

try:
    from feature_engine import engineer_features
except ImportError:
    engineer_features = None

try:
    from rns_scraper import fetch_direct_rns_for_ticker
except ImportError:
    def fetch_direct_rns_for_ticker(ticker: str):
        return {"headline": "Standard Market Flow", "status": "📰 Flow Verified",
                "delta": 0.0, "timestamp": datetime.now().strftime("%Y-%m-%d")}

# ==============================================================================
# PATHS & CONSTANTS
# ==============================================================================
ROOT_DIR = Path(__file__).resolve().parent
DB_PATH = str(ROOT_DIR / "lse_market_data.duckdb")
MODEL_DIR = ROOT_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH = str(MODEL_DIR / "ensemble_ranker.joblib")
MODEL_BACKUP_DIR = MODEL_DIR / "backups"
MODEL_BACKUP_DIR.mkdir(parents=True, exist_ok=True)

MAX_DAILY_EQUITY_TRADES = 5
MAX_HOLD_CALENDAR_DAYS = 7
BREAKEVEN_TRIGGER_RATIO = 0.65
MAX_ACTIVE_PER_SECTOR = 2
COOLDOWN_CALENDAR_DAYS = 3
HALF_LIFE_DAYS = 30.0
MIN_SETTLED_TO_RETRAIN = 80          # raised for safety
RETRAIN_STEP_INTERVAL = 30
MAX_MODEL_BACKUPS = 40
MIN_PRICE_PENCE = 15.0               # reject micro-price names
MIN_AVG_VOLUME = 50000               # basic liquidity gate (shares)

VALID_LSE_TICKER_RE = re.compile(r"^[A-Z0-9]{1,6}(-[A-Z0-9]{1,2})?\.L$")

# ==============================================================================
# STATIC MAPS (verify AIM list periodically)
# ==============================================================================
AIM_EXEMPT_TICKERS = {
    "AOM.L", "SEE.L", "SRC.L", "SAV.L", "YOU.L", "PTAL.L", "JET2.L", "CER.L",
    "RWS.L", "RKH.L", "KGH.L", "NFG.L", "KP2.L", "BIG.L", "JHD.L", "LTHM.L",
    "CMCL.L", "CAML.L", "CHRT.L", "CNC.L", "CRW.L", "DOTD.L", "FEVR.L", "ITM.L",
    "BUR.L", "BRCK.L", "MIDW.L"
    # VIC.L removed – Victrex is Main Market
}

SECTOR_MAP = {
    "SHEL.L": "Energy", "BP.L": "Energy", "PTAL.L": "Energy", "RKH.L": "Energy", "ITM.L": "Energy",
    "RIO.L": "Basic Materials", "GLEN.L": "Basic Materials", "LTHM.L": "Basic Materials",
    "CMCL.L": "Basic Materials", "CAML.L": "Basic Materials", "SAV.L": "Basic Materials", "KP2.L": "Basic Materials",
    "HSBA.L": "Financials", "BARC.L": "Financials", "LSEG.L": "Financials", "BUR.L": "Financials",
    "LLOY.L": "Financials", "NWG.L": "Financials", "PRU.L": "Financials", "LGEN.L": "Financials", "AV.L": "Financials",
    "AZN.L": "Healthcare", "GSK.L": "Healthcare", "HLN.L": "Healthcare", "SN.L": "Healthcare", "HIK.L": "Healthcare",
    "ULVR.L": "Consumer", "BATS.L": "Consumer", "FEVR.L": "Consumer", "JET2.L": "Consumer", "NFG.L": "Consumer",
    "TSCO.L": "Consumer", "SBRY.L": "Consumer", "MKS.L": "Consumer", "NXT.L": "Consumer", "DGE.L": "Consumer", "RKT.L": "Consumer",
    "NG.L": "Telecom & Utilities", "VOD.L": "Telecom & Utilities", "BT-A.L": "Telecom & Utilities",
    "SSE.L": "Telecom & Utilities", "CNA.L": "Telecom & Utilities",
    "DOTD.L": "Technology", "BIG.L": "Technology", "SEE.L": "Technology", "AOM.L": "Technology",
    "YOU.L": "Technology", "CER.L": "Technology", "RWS.L": "Technology", "KGH.L": "Technology",
    "AUTO.L": "Technology", "SGE.L": "Technology",
    "CHRT.L": "Defense & Aero", "CNC.L": "Defense & Aero", "BA.L": "Defense & Aero",
    "RR.L": "Defense & Aero", "QQ.L": "Defense & Aero",
    "CRW.L": "Industrials", "BRCK.L": "Industrials", "MIDW.L": "Industrials", "VIC.L": "Industrials",
    "SRC.L": "Industrials", "JHD.L": "Industrials", "REL.L": "Industrials", "EXPN.L": "Industrials", "AHT.L": "Industrials"
}

FTSE_EXPORTERS = {"AZN.L", "GSK.L", "SHEL.L", "BP.L", "ULVR.L", "BATS.L", "RIO.L", "GLEN.L", "DGE.L", "REL.L", "CRW.L"}

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

# ==============================================================================
# DATABASE – LONG-LIVED, INDEXED, MIGRATION-SAFE
# ==============================================================================
@st.cache_resource(show_spinner=False)
def get_db_lock():
    return threading.RLock()

DB_LOCK = get_db_lock()

@st.cache_resource(show_spinner=False)
def get_duckdb_connection():
    con = duckdb.connect(DB_PATH, read_only=False)
    try:
        con.execute("PRAGMA threads=4")
        con.execute("PRAGMA enable_object_cache=true")
    except Exception:
        pass
    return con

def init_duckdb_storage():
    with DB_LOCK:
        con = get_duckdb_connection()
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
        # Persistent system state (retrain counter, etc.)
        con.execute("""
            CREATE TABLE IF NOT EXISTS system_state (
                key VARCHAR PRIMARY KEY,
                value VARCHAR,
                updated_at VARCHAR
            )
        """)
        for col, typ, default in [("features_json", "VARCHAR", "'{}'")]:
            try:
                con.execute(f"ALTER TABLE trade_journal ADD COLUMN {col} {typ} DEFAULT {default}")
            except Exception:
                pass
        for idx in [
            "CREATE INDEX IF NOT EXISTS idx_tj_status ON trade_journal(status)",
            "CREATE INDEX IF NOT EXISTS idx_tj_date ON trade_journal(date_str)",
            "CREATE INDEX IF NOT EXISTS idx_tj_ticker ON trade_journal(ticker)",
            "CREATE INDEX IF NOT EXISTS idx_opt_status ON daily_options_journal(status)",
        ]:
            try:
                con.execute(idx)
            except Exception:
                pass

init_duckdb_storage()

def get_system_state(key: str, default: str = "0") -> str:
    with DB_LOCK:
        con = get_duckdb_connection()
        try:
            row = con.execute("SELECT value FROM system_state WHERE key = ?", [key]).fetchone()
            return row[0] if row else default
        except Exception:
            return default

def set_system_state(key: str, value: str):
    now = datetime.now(pytz.timezone("Europe/London")).strftime("%Y-%m-%d %H:%M:%S")
    with DB_LOCK:
        con = get_duckdb_connection()
        con.execute("""
            INSERT OR REPLACE INTO system_state (key, value, updated_at) VALUES (?, ?, ?)
        """, [key, str(value), now])

# ==============================================================================
# HELPERS
# ==============================================================================
def is_valid_lse_ticker(ticker: str) -> bool:
    return bool(isinstance(ticker, str) and VALID_LSE_TICKER_RE.match(ticker.strip()))

def get_ticker_sector(ticker: str) -> str:
    t = str(ticker).strip().upper()
    return SECTOR_MAP.get(t, "Unknown")

def is_aim_exempt(ticker: str) -> bool:
    return str(ticker).strip().upper() in AIM_EXEMPT_TICKERS

def get_uk_friction_pct(ticker: str) -> float:
    return 0.25 if is_aim_exempt(ticker) else 0.60

def get_breakeven_exit_price(entry_price: float, ticker: str) -> float:
    return round(float(entry_price) * (1.0 + get_uk_friction_pct(ticker) / 100.0), 2)

def calc_net_equity_pnl_pct(entry_price: float, current_price: float, ticker: str) -> float:
    if entry_price <= 0:
        return 0.0
    gross = ((float(current_price) - float(entry_price)) / float(entry_price)) * 100.0
    return round(gross - get_uk_friction_pct(ticker), 2)

def evaluate_rns_nlp_sentiment(ticker: str) -> dict:
    raw = fetch_direct_rns_for_ticker(ticker)
    headline = str(raw.get("headline", "")).lower()
    status = str(raw.get("status", "📰 Flow Verified"))
    base_delta = float(raw.get("delta", 0.0))
    lex_delta = 0.0
    tags = []
    for p, w in RNS_BEARISH_LEXICON.items():
        if p in headline:
            lex_delta += w
            tags.append(f"⚠️ {p.title()}")
    for p, w in RNS_BULLISH_LEXICON.items():
        if p in headline:
            lex_delta += w
            tags.append(f"🚀 {p.title()}")
    total = max(-0.30, min(0.15, base_delta + lex_delta))
    if total <= -0.15 or "Dilution" in status:
        status = "🚨 Dilution / Adverse RNS"
    elif total >= 0.04:
        status = "🟢 Bullish RNS Catalyst"
    return {"headline": raw.get("headline", "Standard Market Flow"), "status": status,
            "delta": round(total, 4), "nlp_tags": ", ".join(tags) if tags else "Neutral"}

# ==============================================================================
# SUPABASE
# ==============================================================================
try:
    from supabase import create_client
except ImportError:
    create_client = None

@st.cache_resource(show_spinner=False)
def get_supabase_client():
    if create_client is None:
        return None
    url = st.secrets.get("SUPABASE_URL") if hasattr(st, "secrets") else None
    key = st.secrets.get("SUPABASE_KEY") if hasattr(st, "secrets") else None
    if not url or not key:
        return None
    try:
        return create_client(url.strip().split("/rest/v1")[0].rstrip("/"), key.strip())
    except Exception:
        return None

supabase = get_supabase_client()

def record_db_error(ctx: str, err: Exception):
    msg = re.sub(r"https?://[^\s'\"]+", "[REDACTED]", str(err))
    st.session_state["db_error"] = f"[{ctx}] {msg[:140]}"

def fetch_all_supabase_rows(table: str) -> list:
    if not supabase:
        return []
    rows, start, size = [], 0, 1000
    while True:
        try:
            res = supabase.table(table).select("*").range(start, start + size - 1).execute()
            if not res.data:
                break
            rows.extend(res.data)
            if len(res.data) < size:
                break
            start += size
        except Exception as e:
            record_db_error(f"Paginate {table}", e)
            break
    return rows

def hydrate_duckdb_from_supabase():
    """Hydrate ONLY missing or newer rows. Never resurrect a locally closed trade to ACTIVE."""
    if not supabase:
        return
    with DB_LOCK:
        con = get_duckdb_connection()
        # Equities
        try:
            for r in fetch_all_supabase_rows("predictions"):
                tkr = str(r.get("ticker", "")).strip()
                if not is_valid_lse_ticker(tkr):
                    continue
                pred_date = str(r.get("predicted_date", ""))
                trade_id = f"{tkr}_{pred_date}"
                cloud_status = str(r.get("status", "ACTIVE")).upper()
                # Protection: if local row exists and is already closed, do not overwrite with ACTIVE
                local = con.execute("SELECT status FROM trade_journal WHERE trade_id = ?", [trade_id]).fetchone()
                if local and local[0] != "ACTIVE" and cloud_status == "ACTIVE":
                    continue  # never resurrect
                last_check = str(r.get("last_checked") or datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
                f_json = json.dumps(r.get("features_json") or {})
                exit_ts = last_check if cloud_status != "ACTIVE" else None
                con.execute("""
                    INSERT OR REPLACE INTO trade_journal
                    (trade_id, timestamp, date_str, ticker, asset_type, entry_price, target_price,
                     stop_loss, shares, capital_allocated, status, latest_price, pnl_pct, exit_price,
                     exit_timestamp, last_audited, features_json)
                    VALUES (?, ?, ?, ?, 'EQUITY', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, [
                    trade_id, last_check, pred_date, tkr,
                    float(r.get("entry_price", 0)), float(r.get("target_price", 0)),
                    float(r.get("stop_loss", 0)), int(r.get("shares_qty", 1)),
                    float(r.get("position_gbp", 0)), cloud_status,
                    float(r.get("latest_price", 0)), float(r.get("pnl_pct", 0)),
                    float(r.get("latest_price", 0)) if cloud_status != "ACTIVE" else 0.0,
                    exit_ts, last_check, f_json
                ])
        except Exception as e:
            record_db_error("Hydrate Equities", e)
        # Options
        try:
            for o in fetch_all_supabase_rows("options_journal"):
                date_k = str(o.get("date_key", ""))
                local = con.execute("SELECT status FROM daily_options_journal WHERE date_key = ?", [date_k]).fetchone()
                cloud_status = str(o.get("status", "ACTIVE")).upper()
                if local and local[0] != "ACTIVE" and cloud_status == "ACTIVE":
                    continue
                con.execute("""
                    INSERT OR REPLACE INTO daily_options_journal VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """, [
                    date_k, str(o.get("timestamp")), str(o.get("share_name")),
                    str(o.get("option_contract")), float(o.get("strike_price", 0)),
                    int(o.get("expiry_days", 21)), float(o.get("underlying_spot", 0)),
                    int(o.get("lot_size", 1000)), float(o.get("entry_premium", 0)),
                    float(o.get("current_option_price", 0)), float(o.get("target_premium", 0)),
                    float(o.get("stop_loss_premium", 0)), float(o.get("total_capital", 0)),
                    float(o.get("ai_confidence", 80)), float(o.get("implied_vol", 20)),
                    cloud_status, float(o.get("pnl_pct", 0)), str(o.get("last_audited"))
                ])
        except Exception as e:
            record_db_error("Hydrate Options", e)

if "hydrated_once" not in st.session_state:
    hydrate_duckdb_from_supabase()
    st.session_state["hydrated_once"] = True

# ==============================================================================
# PAGE CONFIG + DARK THEME
# ==============================================================================
st.set_page_config(page_title="ALPHA-LSE Quant Terminal", page_icon="⚡", layout="wide",
                   initial_sidebar_state="expanded")
st.markdown("""
<style>
    .stApp { background-color: #0e1117; color: #fafafa; }
    section[data-testid="stSidebar"] { background-color: #161b22; }
    .stMetric { background-color: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 10px; }
    .stTabs [data-baseweb="tab"] { background-color: #21262d; border-radius: 6px; }
    .stTabs [aria-selected="true"] { background-color: #238636; color: white; }
</style>
""", unsafe_allow_html=True)

LOT_SIZES = {"SHEL": 1000, "AZN": 500, "HSBA": 2000, "ULVR": 500, "BP": 3000, "BARC": 5000, "RIO": 250, "GLEN": 4000}

def is_lse_market_open() -> bool:
    now = datetime.now(pytz.timezone("Europe/London"))
    return dtime(8, 0) <= now.time() <= dtime(16, 30) and now.weekday() <= 4

def norm_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

def calculate_black_scholes_call(spot, strike, days, r, sigma):
    T = max(days, 1.0) / 365.0
    if spot <= 0 or strike <= 0 or sigma <= 0:
        return max(0.0, spot - strike)
    d1 = (math.log(spot / strike) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return max(round(spot * norm_cdf(d1) - strike * math.exp(-r * T) * norm_cdf(d2), 2), 0.50)

@st.cache_data(ttl=300, show_spinner=False)
def fetch_cached_history(ticker: str, period: str = "120d") -> pd.DataFrame:
    if not is_valid_lse_ticker(ticker):
        return pd.DataFrame()
    for _ in range(2):
        try:
            df = yf.Ticker(ticker).history(period=period, auto_adjust=False, timeout=6)
            if df is not None and not df.empty:
                return df
        except Exception:
            time.sleep(0.3)
    return pd.DataFrame()

@st.cache_data(ttl=1800, show_spinner=False)
def get_cross_asset_macro_regime() -> dict:
    macro = {"brent_5d": 0.0, "copper_5d": 0.0, "gbpusd_5d": 0.0, "ftse_5d": 0.0}
    for key, sym in {"brent_5d": "BZ=F", "copper_5d": "HG=F", "gbpusd_5d": "GBPUSD=X", "ftse_5d": "^FTSE"}.items():
        try:
            h = yf.Ticker(sym).history(period="10d", timeout=5)
            if h is not None and len(h) >= 5:
                c0, c5 = float(h["Close"].iloc[-1]), float(h["Close"].iloc[-5])
                if c5 > 0:
                    macro[key] = round(((c0 - c5) / c5) * 100.0, 2)
        except Exception:
            pass
    return macro

def compute_macro_lead_lag_adjustment(ticker, sector, macro):
    delta, notes = 0, []
    brent, copper, gbp, ftse = macro.get("brent_5d", 0), macro.get("copper_5d", 0), macro.get("gbpusd_5d", 0), macro.get("ftse_5d", 0)
    if sector == "Energy":
        if brent >= 2: delta += 3; notes.append(f"+3% Brent")
        elif brent <= -2.5: delta -= 3; notes.append(f"-3% Brent")
    elif sector == "Basic Materials":
        if copper >= 1.5: delta += 3; notes.append(f"+3% Copper")
        elif copper <= -2: delta -= 3; notes.append(f"-3% Copper")
    if ticker in FTSE_EXPORTERS and gbp <= -0.8:
        delta += 2; notes.append("+2% FX")
    if ftse <= -2: delta -= 2; notes.append("-2% FTSE Risk-Off")
    elif ftse >= 1.5 and delta == 0: delta += 1; notes.append("+1% FTSE")
    return max(-4, min(4, delta)), (" | ".join(notes) if notes else "Neutral Macro")

# ==============================================================================
# AUTH – FAIL CLOSED
# ==============================================================================
def check_password() -> bool:
    if st.session_state.get("password_correct", False):
        return True
    if not hasattr(st, "secrets") or not st.secrets.get("AUTH_PASS_HASH") or not st.secrets.get("AUTH_USER"):
        st.error("🔒 AUTH_USER / AUTH_PASS_HASH missing from secrets. Refusing to start (fail-closed).")
        st.stop()
        return False
    now_ts = time.time()
    if now_ts < st.session_state.get("auth_lockout_until", 0):
        st.error("🔒 Locked out. Try later.")
        return False
    expected_user = str(st.secrets["AUTH_USER"]).strip()
    expected_hash = str(st.secrets["AUTH_PASS_HASH"]).strip().lower()
    st.subheader("🔐 ALPHA-LSE Quant Terminal — Restricted Access")
    with st.form("login", clear_on_submit=True):
        u = st.text_input("Username")
        p = st.text_input("Passkey", type="password")
        if st.form_submit_button("Authenticate", type="primary", use_container_width=True):
            ok_u = hmac.compare_digest(u.strip(), expected_user)
            ok_p = hmac.compare_digest(hashlib.sha256(p.encode()).hexdigest(), expected_hash)
            if ok_u and ok_p:
                st.session_state["password_correct"] = True
                st.session_state["failed_auth_attempts"] = 0
                st.rerun()
            fails = st.session_state.get("failed_auth_attempts", 0) + 1
            st.session_state["failed_auth_attempts"] = fails
            if fails >= 5:
                st.session_state["auth_lockout_until"] = time.time() + 900
            st.error("Invalid credentials.")
    return False

if not check_password():
    st.stop()

# ==============================================================================
# AUDIT – CONSERVATIVE, MULTI-BAR
# ==============================================================================
def is_stop_breakeven_protected(entry, stop, ticker):
    return float(stop) >= float(entry) - 1e-4

def audit_and_reconcile_all_trades():
    lon = pytz.timezone("Europe/London")
    now = datetime.now(lon)
    now_str = now.strftime("%Y-%m-%d %H:%M:%S")
    today = now.date()

    with DB_LOCK:
        con = get_duckdb_connection()
        active = con.execute("SELECT * FROM trade_journal WHERE status = 'ACTIVE'").df()
        active_opts = con.execute("SELECT * FROM daily_options_journal WHERE status = 'ACTIVE'").df()

    for _, tr in active.iterrows():
        tkr = str(tr["ticker"]).strip()
        if not is_valid_lse_ticker(tkr):
            continue
        try:
            # Fetch enough history to cover the whole holding period
            h = fetch_cached_history(tkr, period="30d")
            if h.empty:
                continue
            entry = float(tr["entry_price"])
            target = float(tr["target_price"])
            stop = float(tr["stop_loss"])
            be_floor = get_breakeven_exit_price(entry, tkr)
            entry_date = datetime.strptime(str(tr["date_str"]), "%Y-%m-%d").date()

            # Restrict to bars on/after entry date
            h = h.copy()
            h["date"] = pd.to_datetime(h.index).date
            bars = h[h["date"] >= entry_date].sort_index()
            if bars.empty:
                continue

            curr = float(bars["Close"].iloc[-1])
            new_status = "ACTIVE"
            exit_price = 0.0
            final_stop = stop

            # Walk bars chronologically for conservatism
            for i, (idx, bar) in enumerate(bars.iterrows()):
                bar_high = float(bar["High"])
                bar_low = float(bar["Low"])
                bar_close = float(bar["Close"])
                is_entry_bar = (i == 0)

                # Ratchet check
                target_dist = target - entry
                if target_dist > 0 and final_stop < be_floor:
                    trigger = entry + BREAKEVEN_TRIGGER_RATIO * target_dist
                    hit = bar_close >= trigger if is_entry_bar else (bar_close >= trigger or bar_high >= trigger)
                    if hit and bar_close > be_floor:
                        final_stop = be_floor

                be_prot = is_stop_breakeven_protected(entry, final_stop, tkr)

                # Conservative: on any bar, if both target and stop touched → assume STOP first
                touched_stop = bar_low <= final_stop if not is_entry_bar else bar_close <= final_stop
                touched_target = bar_high >= target if not is_entry_bar else bar_close >= target

                if touched_stop and touched_target:
                    # Stop first (conservative)
                    if be_prot:
                        new_status = "🛡️ BREAK-EVEN (PROTECTED EXIT)"
                        exit_price = min(bar_close, be_floor)  # cannot assume better than reality
                        curr = exit_price
                    else:
                        new_status = "🛑 LOSS (STOPPED OUT)"
                        exit_price = min(bar_close, final_stop)
                        curr = exit_price
                    break
                elif touched_stop:
                    if be_prot:
                        new_status = "🛡️ BREAK-EVEN (PROTECTED EXIT)"
                        exit_price = min(bar_close, be_floor)
                        curr = exit_price
                    else:
                        new_status = "🛑 LOSS (STOPPED OUT)"
                        exit_price = min(bar_close, final_stop)
                        curr = exit_price
                    break
                elif touched_target:
                    new_status = "🎯 WIN (TARGET HIT)"
                    exit_price = max(bar_close, target)
                    curr = exit_price
                    break

            if new_status == "ACTIVE":
                if (today - entry_date).days >= MAX_HOLD_CALENDAR_DAYS:
                    new_status = "⏱️ EXPIRED (TIME EXIT)"
                    exit_price = curr

            pnl = 0.0 if "BREAK-EVEN" in new_status else calc_net_equity_pnl_pct(entry, curr, tkr)

            with DB_LOCK:
                con = get_duckdb_connection()
                con.execute("""
                    UPDATE trade_journal SET
                        latest_price=?, stop_loss=?, pnl_pct=?, status=?, exit_price=?,
                        exit_timestamp=CASE WHEN ? != 'ACTIVE' THEN ? ELSE exit_timestamp END,
                        last_audited=?
                    WHERE trade_id=?
                """, [curr, final_stop, pnl, new_status, exit_price, new_status, now_str, now_str, tr["trade_id"]])

            if supabase and new_status != "ACTIVE":
                try:
                    supabase.table("predictions").update({
                        "status": new_status, "stop_loss": final_stop, "latest_price": curr,
                        "pnl_pct": pnl, "last_checked": now_str
                    }).eq("ticker", tkr).eq("predicted_date", tr["date_str"]).execute()
                except Exception as e:
                    record_db_error("Audit Equity Sync", e)
        except Exception:
            continue

    # Options audit (still simulated – labelled later)
    for _, opt in active_opts.iterrows():
        try:
            sym = f"{opt['share_name']}.L"
            h = fetch_cached_history(sym, period="15d")
            if h.empty:
                continue
            spot = float(h["Close"].iloc[-1])
            strike = float(opt["strike_price"])
            entry_p = float(opt["entry_premium"])
            target_p = float(opt["target_premium"])
            stop_p = float(opt["stop_loss_premium"])
            sigma = float(opt["implied_vol"]) / 100.0
            elapsed = 0
            try:
                elapsed = max(0, (today - datetime.strptime(str(opt["date_key"]), "%Y-%m-%d").date()).days)
            except Exception:
                pass
            rem = max(1, int(opt["expiry_days"]) - elapsed)
            live = calculate_black_scholes_call(spot, strike, rem, 0.05, sigma)
            pnl = round(((live - entry_p) / entry_p) * 100, 2) if entry_p > 0 else 0.0
            status = "ACTIVE"
            if live >= target_p or pnl >= 60:
                status = "🎯 WIN (TARGET HIT)"
            elif live <= stop_p or pnl <= -50:
                status = "🛑 LOSS (STOPPED OUT)"
            elif elapsed >= MAX_HOLD_CALENDAR_DAYS:
                status = "⏱️ EXPIRED (TIME EXIT)"
            with DB_LOCK:
                con = get_duckdb_connection()
                con.execute("UPDATE daily_options_journal SET current_option_price=?, pnl_pct=?, status=?, last_audited=? WHERE date_key=?",
                            [live, pnl, status, now_str, opt["date_key"]])
        except Exception:
            continue

# ==============================================================================
# SELF-LEARNING + CHAMPION-CHALLENGER RETRAIN
# ==============================================================================
def get_self_learning_adjustment(ticker: str, atr_pct: float) -> dict:
    delta, reasons = 0.0, []
    today = datetime.now(pytz.timezone("Europe/London")).date()
    with DB_LOCK:
        con = get_duckdb_connection()
        try:
            hist = con.execute("SELECT ticker, exit_timestamp, last_audited, date_str, status, features_json FROM trade_journal WHERE status != 'ACTIVE'").df()
        except Exception:
            hist = pd.DataFrame()
    if hist.empty:
        return {"delta": 0, "reason": "Building memory"}
    t_hist = hist[hist["ticker"] == ticker]
    decayed_loss = decayed_win = 0.0
    for _, r in t_hist.iterrows():
        try:
            raw = str(r.get("exit_timestamp") or r.get("last_audited") or r.get("date_str") or "")[:10]
            days = max(0, (today - datetime.strptime(raw, "%Y-%m-%d").date()).days)
        except Exception:
            days = 15
        w = math.pow(0.5, days / HALF_LIFE_DAYS)
        stv = str(r["status"]).upper()
        if "LOSS" in stv:
            decayed_loss += 15 * w
        elif "WIN" in stv:
            decayed_win += 5 * w
    if decayed_loss >= 1:
        d = int(round(decayed_loss)); delta -= d; reasons.append(f"-{d}% decayed loss")
    if decayed_win >= 1:
        d = int(round(decayed_win)); delta += d; reasons.append(f"+{d}% decayed win")
    # High ATR penalty from historical losses
    loss_atrs = []
    for fs in hist[hist["status"].str.contains("LOSS", na=False)]["features_json"].dropna():
        try:
            obj = json.loads(fs)
            if "atr_pct" in obj:
                loss_atrs.append(float(obj["atr_pct"]))
        except Exception:
            pass
    if loss_atrs and atr_pct >= max(3.0, float(np.percentile(loss_atrs, 75))):
        delta -= 10
        reasons.append("-10% high-ATR regime")
    return {"delta": int(round(delta)), "reason": " | ".join(reasons) if reasons else "No adverse match"}

def _backup_model():
    if not os.path.exists(MODEL_PATH):
        return
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    shutil.copy2(MODEL_PATH, MODEL_BACKUP_DIR / f"ensemble_ranker_{ts}.joblib")
    backups = sorted(MODEL_BACKUP_DIR.glob("ensemble_ranker_*.joblib"), key=os.path.getmtime)
    while len(backups) > MAX_MODEL_BACKUPS:
        backups.pop(0).unlink(missing_ok=True)

def check_and_auto_retrain_model(feature_cols: list):
    """Champion-challenger only. Counter persisted in DB."""
    with DB_LOCK:
        con = get_duckdb_connection()
        closed = con.execute("SELECT status, features_json FROM trade_journal WHERE status != 'ACTIVE'").df()
    total = len(closed)
    if total < MIN_SETTLED_TO_RETRAIN:
        return
    last = int(get_system_state("last_retrained_count", "0"))
    if (total - last) < RETRAIN_STEP_INTERVAL:
        return

    rows, labels, weights = [], [], []
    for _, r in closed.iterrows():
        try:
            raw = json.loads(r["features_json"]).get("raw_features", {})
            if not all(c in raw for c in feature_cols):
                continue
            stv = str(r["status"]).upper()
            if "WIN" in stv:
                labels.append(1); weights.append(3.0)
            elif "LOSS" in stv:
                labels.append(0); weights.append(3.5)
            elif "BREAK-EVEN" in stv:
                labels.append(1); weights.append(1.5)
            else:
                continue
            rows.append([float(raw[c]) for c in feature_cols])
        except Exception:
            continue
    if len(rows) < MIN_SETTLED_TO_RETRAIN or len(set(labels)) < 2:
        return

    try:
        from lightgbm import LGBMClassifier
        from catboost import CatBoostClassifier
        from sklearn.model_selection import cross_val_score
        X = pd.DataFrame(rows, columns=feature_cols)
        y = np.array(labels)
        w = np.array(weights)

        # Challenger
        lgb = LGBMClassifier(n_estimators=200, learning_rate=0.03, max_depth=5, random_state=42, verbosity=-1)
        cb = CatBoostClassifier(iterations=250, learning_rate=0.03, depth=5, verbose=0, random_seed=42)
        lgb.fit(X, y, sample_weight=w)
        cb.fit(X, y, sample_weight=w)

        # Simple CV score as promotion gate
        try:
            scores = cross_val_score(lgb, X, y, cv=min(5, len(set(y))), scoring="roc_auc", fit_params={"sample_weight": w})
            challenger_score = float(np.mean(scores))
        except Exception:
            challenger_score = 0.55

        champion_score = float(get_system_state("champion_cv_score", "0.50"))
        if challenger_score < champion_score + 0.01:  # must be meaningfully better
            set_system_state("last_retrained_count", str(total))  # still advance counter to avoid spin
            return

        _backup_model()
        bundle = {"lgb": lgb, "catboost": cb, "feature_cols": feature_cols,
                  "trained_on_n": len(rows), "cv_score": challenger_score,
                  "trained_at": datetime.now().isoformat()}
        tmp = MODEL_PATH + ".tmp"
        joblib.dump(bundle, tmp)
        os.replace(tmp, MODEL_PATH)
        st.session_state["ml_model_bundle"] = bundle
        set_system_state("last_retrained_count", str(total))
        set_system_state("champion_cv_score", f"{challenger_score:.4f}")
        st.session_state["retrain_notice"] = f"🧠 New champion promoted (CV {challenger_score:.3f} > {champion_score:.3f}) on {len(rows)} trades."
    except Exception as e:
        record_db_error("Retrain", e)

# ==============================================================================
# OPTIONS (SIMULATED)
# ==============================================================================
def generate_daily_options_alpha() -> dict:
    lon = pytz.timezone("Europe/London")
    today = datetime.now(lon).strftime("%Y-%m-%d")
    now_str = datetime.now(lon).strftime("%Y-%m-%d %H:%M:%S")
    with DB_LOCK:
        con = get_duckdb_connection()
        df = con.execute("SELECT * FROM daily_options_journal WHERE date_key = ?", [today]).df()
        if not df.empty:
            return df.iloc[0].to_dict()
    best, best_score = None, -999.0
    for sym in ["BARC", "BP", "HSBA", "GLEN", "SHEL"]:
        try:
            h = fetch_cached_history(f"{sym}.L", period="30d")
            if h.empty or len(h) < 15:
                continue
            spot = float(h["Close"].iloc[-1])
            ret = (spot - float(h["Close"].iloc[-5])) / float(h["Close"].iloc[-5])
            sig = max(0.15, min(0.45, float(np.log(h["Close"]/h["Close"].shift(1)).dropna().std() * np.sqrt(252))))
            strike = round(spot * 1.02)
            prem = calculate_black_scholes_call(spot, strike, 21, 0.05, sig)
            cap = round(prem * LOT_SIZES.get(sym, 1000) / 100, 2)
            if cap <= 500 and ret > best_score:
                best_score = ret
                conf = round(min(88, max(65, 75 + ret * 120)), 1)
                best = {
                    "date_key": today, "timestamp": now_str, "share_name": sym,
                    "option_contract": f"{sym} {int(strike)}p CE (SIM)", "strike_price": float(strike),
                    "expiry_days": 21, "underlying_spot": spot, "lot_size": LOT_SIZES.get(sym, 1000),
                    "entry_premium": prem, "current_option_price": prem,
                    "target_premium": round(prem * 1.6, 2), "stop_loss_premium": round(prem * 0.5, 2),
                    "total_capital": cap, "ai_confidence": conf, "implied_vol": round(sig * 100, 1),
                    "status": "ACTIVE", "pnl_pct": 0.0, "last_audited": now_str
                }
        except Exception:
            continue
    if not best:
        return {}
    with DB_LOCK:
        con = get_duckdb_connection()
        con.execute("INSERT OR REPLACE INTO daily_options_journal VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    [best[k] for k in ["date_key","timestamp","share_name","option_contract","strike_price","expiry_days",
                                       "underlying_spot","lot_size","entry_premium","current_option_price","target_premium",
                                       "stop_loss_premium","total_capital","ai_confidence","implied_vol","status","pnl_pct","last_audited"]])
    return best

# ==============================================================================
# LOGGING / GUARDRAILS
# ==============================================================================
def get_currently_active_tickers() -> set:
    with DB_LOCK:
        con = get_duckdb_connection()
        df = con.execute("SELECT DISTINCT ticker FROM trade_journal WHERE status='ACTIVE'").df()
        return set(df["ticker"].tolist()) if not df.empty else set()

def get_active_sector_exposure() -> dict:
    c = defaultdict(int)
    for t in get_currently_active_tickers():
        c[get_ticker_sector(t)] += 1
    return dict(c)

def get_todays_logged_equities() -> pd.DataFrame:
    today = datetime.now(pytz.timezone("Europe/London")).strftime("%Y-%m-%d")
    with DB_LOCK:
        con = get_duckdb_connection()
        return con.execute("SELECT * FROM trade_journal WHERE date_str=? ORDER BY timestamp ASC", [today]).df()

def get_recent_cooldown_tickers() -> set:
    today = datetime.now(pytz.timezone("Europe/London")).date()
    out = set()
    with DB_LOCK:
        con = get_duckdb_connection()
        df = con.execute("SELECT ticker, exit_timestamp, last_audited, date_str FROM trade_journal WHERE status!='ACTIVE'").df()
        for _, r in df.iterrows():
            try:
                raw = str(r.get("exit_timestamp") or r.get("last_audited") or r.get("date_str") or "")[:10]
                if (today - datetime.strptime(raw, "%Y-%m-%d").date()).days <= COOLDOWN_CALENDAR_DAYS:
                    out.add(str(r["ticker"]))
            except Exception:
                pass
    return out

def log_equity_signal_safely(sig: dict, enforce_sector_cap=True) -> bool:
    lon = pytz.timezone("Europe/London")
    now = datetime.now(lon)
    today = now.strftime("%Y-%m-%d")
    now_str = now.strftime("%Y-%m-%d %H:%M:%S")
    ticker = str(sig.get("Ticker", "")).strip()
    if not is_valid_lse_ticker(ticker):
        return False
    sector = get_ticker_sector(ticker)
    if enforce_sector_cap and get_active_sector_exposure().get(sector, 0) >= MAX_ACTIVE_PER_SECTOR:
        return False
    f_json = json.dumps(sig.get("FeaturesDict", {}))
    init_pnl = -get_uk_friction_pct(ticker)
    logged = False
    with DB_LOCK:
        con = get_duckdb_connection()
        cnt = con.execute("SELECT COUNT(*) FROM trade_journal WHERE date_str=?", [today]).fetchone()[0]
        if cnt >= MAX_DAILY_EQUITY_TRADES:
            return False
        exists = con.execute("SELECT 1 FROM trade_journal WHERE ticker=? AND (status='ACTIVE' OR date_str=?)", [ticker, today]).fetchone()
        if not exists:
            shares = int(str(sig["Recommended Shares"]).split()[0])
            con.execute("""
                INSERT INTO trade_journal
                (trade_id, timestamp, date_str, ticker, asset_type, entry_price, target_price, stop_loss,
                 shares, capital_allocated, status, latest_price, pnl_pct, exit_price, exit_timestamp, last_audited, features_json)
                VALUES (?,?,?,?,'EQUITY',?,?,?,?,?,'ACTIVE',?,?,0.0,NULL,?,?)
            """, [f"{ticker}_{today}", now_str, today, ticker, float(sig["Price (p)"]), float(sig["Target (p)"]),
                  float(sig["Stop Loss (p)"]), shares, float(sig["Capital"]), float(sig["Price (p)"]), init_pnl, now_str, f_json])
            logged = True
    if logged and supabase:
        try:
            supabase.table("predictions").insert({
                "predicted_date": today, "ticker": ticker, "company_name": f"{ticker} ({sector})",
                "entry_price": float(sig["Price (p)"]), "target_price": float(sig["Target (p)"]),
                "stop_loss": float(sig["Stop Loss (p)"]), "position_gbp": float(sig["Capital"]),
                "shares_qty": int(str(sig["Recommended Shares"]).split()[0]),
                "profit_goal": float(sig["ReturnNum"]), "confidence": int(float(str(sig["AI Win Confidence"]).replace("%",""))),
                "hold_days": 5, "status": "ACTIVE", "latest_price": float(sig["Price (p)"]),
                "pnl_pct": init_pnl, "news_status": sig.get("RNS","Clean"),
                "rns_headline": str(sig.get("RNSHeadline",""))[:120], "features_json": sig.get("FeaturesDict",{}),
                "last_checked": now_str
            }).execute()
        except Exception as e:
            record_db_error("Insert Equity", e)
    return logged

# ==============================================================================
# ML + SCANNER
# ==============================================================================
def get_ml_model():
    if "ml_model_bundle" in st.session_state:
        return st.session_state["ml_model_bundle"]
    if os.path.exists(MODEL_PATH):
        try:
            b = joblib.load(MODEL_PATH)
            st.session_state["ml_model_bundle"] = b
            return b
        except Exception:
            return None
    return None

@st.cache_data(ttl=3600, show_spinner=False)
def get_live_lse_universe() -> list:
    raw = []
    headers = {"User-Agent": "Mozilla/5.0"}
    for url in ["https://en.wikipedia.org/wiki/FTSE_100_Index", "https://en.wikipedia.org/wiki/FTSE_250_Index"]:
        try:
            r = requests.get(url, headers=headers, timeout=4)
            if r.status_code == 200:
                dfs = pd.read_html(r.text, attrs={"id": "constituents"})
                if dfs:
                    raw.extend([f"{str(t).strip().replace('.','-')}.L" for t in dfs[0]["Ticker"].dropna()])
        except Exception:
            pass
    fallback = list(SECTOR_MAP.keys())
    return sorted({t for t in raw + fallback if is_valid_lse_ticker(t)})

def run_predictions():
    bundle = get_ml_model()
    if bundle is None:
        return pd.DataFrame(), False
    feature_cols = bundle["feature_cols"]
    check_and_auto_retrain_model(feature_cols)
    held = get_currently_active_tickers()
    saturated = {s for s, c in get_active_sector_exposure().items() if c >= MAX_ACTIVE_PER_SECTOR}
    cooldown = get_recent_cooldown_tickers()
    universe = [t for t in get_live_lse_universe()
                if t not in held and t not in cooldown and get_ticker_sector(t) not in saturated
                and get_ticker_sector(t) != "Unknown"]
    if not universe:
        return pd.DataFrame(), False
    chunk = random.sample(universe, min(35, len(universe)))
    macro = get_cross_asset_macro_regime()
    results = []
    lgb, cb = bundle["lgb"], bundle["catboost"]
    prog = st.progress(0, text=f"Scanning {len(chunk)} names...")
    for i, ticker in enumerate(chunk):
        try:
            df = fetch_cached_history(ticker, period="120d")
            if df.empty or len(df) < 55 or engineer_features is None:
                prog.progress((i+1)/len(chunk)); continue
            # Liquidity / price gate
            close_now = float(df["Close"].iloc[-1])
            if close_now < MIN_PRICE_PENCE:
                prog.progress((i+1)/len(chunk)); continue
            avg_vol = float(df["Volume"].tail(20).mean()) if "Volume" in df.columns else 0
            if avg_vol < MIN_AVG_VOLUME:
                prog.progress((i+1)/len(chunk)); continue

            df = df.reset_index()
            feats = engineer_features(df)
            if feats.empty:
                prog.progress((i+1)/len(chunk)); continue
            latest = feats.iloc[-1:]
            rns = evaluate_rns_nlp_sentiment(ticker)
            if "Dilution" in rns["status"]:
                prog.progress((i+1)/len(chunk)); continue
            p1 = float(lgb.predict_proba(latest[feature_cols])[:,1][0])
            p2 = float(cb.predict_proba(latest[feature_cols])[:,1][0])
            blended = max(0.01, min(0.99, 0.5*p1 + 0.5*p2 + rns["delta"]))
            close = float(latest["Close"].values[0])
            atr = float(latest["atr_14"].values[0])
            atr_pct = round((atr/close)*100, 2) if close > 0 else 0
            target = close + 1.8*atr
            stop = close - 1.2*atr
            net_ret = round(((target-close)/close)*100 - get_uk_friction_pct(ticker), 1)
            sector = get_ticker_sector(ticker)
            tax = "AIM (0% SDRT)" if is_aim_exempt(ticker) else "Main (0.5% SDRT)"
            base_conf = int(min(96, max(45, round(45 + (blended**0.85)*52))))
            learner = get_self_learning_adjustment(ticker, atr_pct)
            m_delta, m_note = compute_macro_lead_lag_adjustment(ticker, sector, macro)
            final_conf = int(min(98, max(25, base_conf + learner["delta"] + m_delta)))
            shares = int(50000 / close) if close > 0 else 1
            qual = final_conf >= 65 and net_ret >= 2.0
            raw_map = {c: float(latest[c].values[0]) for c in feature_cols if c in latest.columns}
            results.append({
                "Ticker": ticker, "Sector": sector, "Tax Regime": tax,
                "Friction": f"-{get_uk_friction_pct(ticker):.2f}%", "Price (p)": round(close,2),
                "Expected Return": f"+{net_ret}% Net", "ReturnNum": net_ret,
                "Recommended Shares": f"{shares} shares", "Capital": 500.0,
                "Target (p)": round(target,2), "Stop Loss (p)": round(stop,2),
                "Base ML": f"{base_conf}%", "Learner Delta": f"{learner['delta']+m_delta:+d}%",
                "Learner Note": f"{learner['reason']} | {m_note}",
                "AI Win Confidence": f"{final_conf}%", "Adjusted Score": final_conf,
                "RNS": rns["status"], "RNSHeadline": rns["headline"],
                "FeaturesDict": {"raw_features": raw_map, "atr_pct": atr_pct, "base_ml_conf": base_conf,
                                 "learner_delta": learner["delta"], "macro_delta": m_delta,
                                 "sector": sector, "tax_regime": tax},
                "Qualified": qual
            })
        except Exception:
            pass
        prog.progress((i+1)/len(chunk))
    prog.empty()
    if not results:
        return pd.DataFrame(), False
    df = pd.DataFrame(results).sort_values(["Qualified","Adjusted Score","ReturnNum"], ascending=[False,False,False])
    qual = df[df["Qualified"]]
    for _, s in qual.iterrows():
        if len(get_todays_logged_equities()) >= MAX_DAILY_EQUITY_TRADES:
            break
        log_equity_signal_safely(s.to_dict())
    return qual, not qual.empty

# ==============================================================================
# SIDEBAR + LOOP
# ==============================================================================
st.sidebar.header("⚙️ Guardrails")
st.sidebar.caption(f"Daily Cap: {MAX_DAILY_EQUITY_TRADES} | Sector Cap: {MAX_ACTIVE_PER_SECTOR}")
st.sidebar.caption(f"BE Ratchet ≥ {int(BREAKEVEN_TRIGGER_RATIO*100)}% | Cooldown {COOLDOWN_CALENDAR_DAYS}d")
st.sidebar.caption(f"Min Price {MIN_PRICE_PENCE}p | Min Vol {MIN_AVG_VOLUME:,.0f}")
st.sidebar.markdown("---")
broker_mode = st.sidebar.selectbox("Broker", ["Paper Trading (Simulated)", "Interactive Brokers", "IG Group API"])
st.sidebar.markdown("---")
auto_mode = st.sidebar.toggle("Continuous Mode", value=is_lse_market_open())
refresh_sec = st.sidebar.selectbox("Refresh", [300, 600, 3600], format_func=lambda x: f"{x//60} min")
loop_tick = st_autorefresh(interval=refresh_sec*1000, key="loop") if auto_mode and st_autorefresh else 0
if ("last_loop_tick" not in st.session_state) or (loop_tick != st.session_state.get("last_loop_tick")):
    audit_and_reconcile_all_trades()
    hydrate_duckdb_from_supabase()
    st.session_state["last_loop_tick"] = loop_tick
if "retrain_notice" in st.session_state:
    st.sidebar.success(st.session_state["retrain_notice"])
if "db_error" in st.session_state:
    st.sidebar.error(st.session_state["db_error"])
if st.sidebar.button("Log Out", use_container_width=True):
    st.session_state["password_correct"] = False
    st.rerun()

macro = get_cross_asset_macro_regime()
st.title("⚡ ALPHA-LSE Quant Terminal")
st.caption(f"DB: {'🟢 Supabase' if supabase else '🟡 Local'} | Market: {'🟢 OPEN' if is_lse_market_open() else '🔴 CLOSED'} | "
           f"FTSE {macro['ftse_5d']:+.1f}% | Brent {macro['brent_5d']:+.1f}% | Copper {macro['copper_5d']:+.1f}% | GBP {macro['gbpusd_5d']:+.1f}%")

tab_dash, tab_scan, tab_opt, tab_journal, tab_ai = st.tabs([
    "🏠 Dashboard", "🎯 Equity Signals", "📊 Options (Simulated)", "📖 Journal & Honest P&L", "🧠 AI Memory"
])

# Load journals once
with DB_LOCK:
    con = get_duckdb_connection()
    df_eq = con.execute("SELECT * FROM trade_journal ORDER BY date_str DESC").df()
    df_opt = con.execute("SELECT * FROM daily_options_journal ORDER BY date_key DESC").df()

# ==============================================================================
# HONEST PERFORMANCE CALCULATOR
# ==============================================================================
def compute_honest_stats(df_eq, df_opt):
    eq_closed = df_eq[df_eq["status"] != "ACTIVE"] if not df_eq.empty else pd.DataFrame()
    opt_closed = df_opt[df_opt["status"] != "ACTIVE"] if not df_opt.empty else pd.DataFrame()
    eq_active = df_eq[df_eq["status"] == "ACTIVE"] if not df_eq.empty else pd.DataFrame()

    def sleeve_stats(df, capital_col="capital_allocated", pnl_col="pnl_pct"):
        if df.empty:
            return {"n": 0, "wins": 0, "losses": 0, "be": 0, "expired": 0,
                    "win_rate_all": 0.0, "expectancy": 0.0, "profit_factor": 0.0,
                    "avg_win": 0.0, "avg_loss": 0.0, "realized_gbp": 0.0}
        wins = df[df["status"].str.contains("WIN", na=False)]
        losses = df[df["status"].str.contains("LOSS", na=False)]
        be = df[df["status"].str.contains("BREAK-EVEN", na=False)]
        exp = df[df["status"].str.contains("EXPIRED", na=False)]
        n = len(df)
        realized = (df[capital_col] * df[pnl_col] / 100).sum()
        avg_win = wins[pnl_col].mean() if len(wins) else 0
        avg_loss = losses[pnl_col].mean() if len(losses) else 0
        gross_profit = (wins[capital_col] * wins[pnl_col] / 100).sum() if len(wins) else 0
        gross_loss = abs((losses[capital_col] * losses[pnl_col] / 100).sum()) if len(losses) else 0
        pf = gross_profit / gross_loss if gross_loss > 0 else float("inf")
        return {
            "n": n, "wins": len(wins), "losses": len(losses), "be": len(be), "expired": len(exp),
            "win_rate_all": (len(wins) / n * 100) if n else 0,
            "expectancy": (df[pnl_col].mean() if n else 0),
            "profit_factor": pf, "avg_win": avg_win, "avg_loss": avg_loss,
            "realized_gbp": realized
        }

    eq_s = sleeve_stats(eq_closed)
    opt_s = sleeve_stats(opt_closed, "total_capital", "pnl_pct")
    active_cap = eq_active["capital_allocated"].sum() if not eq_active.empty else 0
    open_pnl = (eq_active["capital_allocated"] * eq_active["pnl_pct"] / 100).sum() if not eq_active.empty else 0
    return eq_s, opt_s, active_cap, open_pnl

eq_stats, opt_stats, active_cap, open_pnl = compute_honest_stats(df_eq, df_opt)

# ==============================================================================
# DASHBOARD
# ==============================================================================
with tab_dash:
    st.markdown("### Honest Performance (All Closed Trades)")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Equity Win Rate (All Closed)", f"{eq_stats['win_rate_all']:.1f}%",
              f"{eq_stats['wins']}W / {eq_stats['losses']}L / {eq_stats['be']}BE / {eq_stats['expired']}Exp")
    c2.metric("Equity Expectancy", f"{eq_stats['expectancy']:+.2f}%",
              f"PF {eq_stats['profit_factor']:.2f}" if eq_stats['profit_factor'] != float('inf') else "PF ∞")
    c3.metric("Equity Realized P&L", f"£{eq_stats['realized_gbp']:+,.0f}",
              f"Avg W {eq_stats['avg_win']:+.1f}% / L {eq_stats['avg_loss']:.1f}%")
    c4.metric("Active Equity Capital", f"£{active_cap:,.0f}", f"Open £{open_pnl:+,.0f}")

    st.caption("⚠️ Options are **simulated** (Black-Scholes on realized vol) and excluded from headline equity stats.")
    o1, o2 = st.columns(2)
    o1.metric("Options (Sim) Win Rate", f"{opt_stats['win_rate_all']:.1f}%", f"{opt_stats['n']} closed")
    o2.metric("Options (Sim) Realized", f"£{opt_stats['realized_gbp']:+,.0f}")

    st.markdown("### Active Equity Positions")
    active = df_eq[df_eq["status"]=="ACTIVE"] if not df_eq.empty else pd.DataFrame()
    if not active.empty:
        cols = st.columns(min(3, len(active)))
        for i, (_, r) in enumerate(active.iterrows()):
            with cols[i % 3]:
                with st.container(border=True):
                    be = is_stop_breakeven_protected(r["entry_price"], r["stop_loss"], r["ticker"])
                    st.markdown(f"**{r['ticker']}** · {get_ticker_sector(r['ticker'])}")
                    st.metric("Net P&L", f"{r['pnl_pct']:+.2f}%")
                    st.caption(f"🎯 {r['target_price']:.1f}p | 🛑 {r['stop_loss']:.1f}p{' 🛡️' if be else ''}")
    else:
        st.info("No active equity positions.")

# ==============================================================================
# SCANNER
# ==============================================================================
with tab_scan:
    today_df = get_todays_logged_equities()
    quota = len(today_df) >= MAX_DAILY_EQUITY_TRADES
    if st.button("🔄 Run Live Scan", type="primary") or ("scan_results" not in st.session_state and not quota):
        with st.spinner("Scanning..."):
            res, ok = run_predictions()
            st.session_state["scan_results"] = res
            st.session_state["has_cleared"] = ok
    if quota:
        st.info(f"Daily quota filled ({len(today_df)}/{MAX_DAILY_EQUITY_TRADES}). Showing today's entries:")
        for i, (_, r) in enumerate(today_df.iterrows(), 1):
            status_icon = "🟢" if r["status"]=="ACTIVE" else ("🎯" if "WIN" in str(r["status"]) else "•")
            st.write(f"{status_icon} **#{i} {r['ticker']}** · {r['status']} · Net {r['pnl_pct']:+.2f}%")
    else:
        res = st.session_state.get("scan_results", pd.DataFrame())
        if not res.empty:
            st.success(f"{len(res)} qualified setups")
            st.dataframe(res[["Ticker","Sector","Price (p)","Target (p)","Stop Loss (p)",
                              "Expected Return","AI Win Confidence","RNS"]], use_container_width=True, hide_index=True)
        else:
            st.warning("No setups currently pass all filters (confidence, return, liquidity, sector, RNS).")

# ==============================================================================
# OPTIONS (SIMULATED)
# ==============================================================================
with tab_opt:
    st.warning("⚠️ **SIMULATED ONLY** — Black-Scholes on realized volatility. No real bid/ask, dividends, or fills. Excluded from headline P&L.")
    sig = generate_daily_options_alpha()
    if sig:
        c1, c2, c3 = st.columns(3)
        c1.metric("Contract (Sim)", sig["option_contract"])
        c2.metric("Theo Premium", f"{sig['current_option_price']:.2f}p", f"{sig['pnl_pct']:+.1f}%")
        c3.metric("Theo Confidence", f"{sig['ai_confidence']}%")
        st.caption(f"Strike {sig['strike_price']:.0f}p · Cap £{sig['total_capital']} · Vol {sig['implied_vol']}% (realized)")

# ==============================================================================
# JOURNAL
# ==============================================================================
with tab_journal:
    st.markdown("### Honest Ledger")
    st.caption("Equity stats include ALL closed outcomes (wins, losses, break-evens, time-expired). Options shown separately and labelled simulated.")
    if st.button("Force Sync"):
        audit_and_reconcile_all_trades()
        hydrate_duckdb_from_supabase()
        st.rerun()
    # Show combined table
    if not df_eq.empty:
        show = df_eq[["date_str","ticker","status","entry_price","target_price","stop_loss",
                      "latest_price","pnl_pct","capital_allocated"]].copy()
        show.columns = ["Date","Ticker","Status","Entry","Target","Stop","Latest","Net%","Capital"]
        st.dataframe(show, use_container_width=True, hide_index=True)

# ==============================================================================
# AI MEMORY
# ==============================================================================
with tab_ai:
    st.markdown("### Permanent Trade Memory (Never Deleted)")
    st.caption("Feature vectors + outcomes drive the 30-day exponential self-learning. Model promotion is champion-challenger only.")
    closed = df_eq[df_eq["status"] != "ACTIVE"].head(30) if not df_eq.empty else pd.DataFrame()
    for _, r in closed.iterrows():
        with st.expander(f"{r['status']} | {r['ticker']} | {r['pnl_pct']:+.2f}%"):
            st.json(json.loads(r["features_json"]) if r["features_json"] else {})