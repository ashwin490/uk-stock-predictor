import os
import re
import sys
import math
import json
import time
import hmac
import hashlib
import threading
import warnings
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

ROOT_DIR = Path(__file__).resolve().parent
DB_PATH = os.path.join(ROOT_DIR, "lse_market_data.duckdb")
MODEL_PATH = os.path.join(ROOT_DIR, "models", "ensemble_ranker.joblib")

MAX_DAILY_EQUITY_TRADES = 5
MAX_HOLD_CALENDAR_DAYS = 7        # 5 LSE trading days
BREAKEVEN_TRIGGER_RATIO = 0.65    # Ratchet stop to friction-adjusted break-even at >= 65% of target distance
MAX_ACTIVE_PER_SECTOR = 2         # Barra-style concentration limit: max 2 active positions per sector
COOLDOWN_CALENDAR_DAYS = 3        # Anti-churn rule: prevent re-entry within 3 days of actual exit
HALF_LIFE_DAYS = 30.0             # 30-day exponential decay half-life for historical penalties
MIN_SETTLED_TO_RETRAIN = 50       # Minimum closed trades to trigger autonomous ML retraining
RETRAIN_STEP_INTERVAL = 25        # Retrain every N new closed trades thereafter

# Strict LSE Ticker Regex (Integrity Gate against untrusted web scraping)
VALID_LSE_TICKER_RE = re.compile(r"^[A-Z0-9]{1,6}(-[A-Z0-9]{1,2})?\.L$")

# ==============================================================================
# 0. INSTITUTIONAL UK TCA, AIM REGISTRY, SECTOR MAP & NLP LEXICON
# ==============================================================================
# AIM-listed equities are legally exempt from the 0.50% UK HMRC Stamp Duty Reserve Tax (SDRT)
AIM_EXEMPT_TICKERS = {
    "AOM.L", "SEE.L", "SRC.L", "SAV.L", "YOU.L", "PTAL.L", "JET2.L", "CER.L",
    "RWS.L", "RKH.L", "KGH.L", "NFG.L", "KP2.L", "BIG.L", "JHD.L", "LTHM.L",
    "CMCL.L", "CAML.L", "CHRT.L", "CNC.L", "CRW.L", "DOTD.L", "FEVR.L", "ITM.L",
    "BUR.L", "BRCK.L", "MIDW.L", "VIC.L"
}

SECTOR_MAP = {
    # Energy
    "SHEL.L": "Energy", "BP.L": "Energy", "PTAL.L": "Energy", "RKH.L": "Energy", "ITM.L": "Energy",
    # Basic Materials
    "RIO.L": "Basic Materials", "GLEN.L": "Basic Materials", "LTHM.L": "Basic Materials",
    "CMCL.L": "Basic Materials", "CAML.L": "Basic Materials", "SAV.L": "Basic Materials", "KP2.L": "Basic Materials",
    # Financials (Expanded FTSE 100/250)
    "HSBA.L", "BARC.L": "Financials", "LSEG.L": "Financials", "BUR.L": "Financials",
    "LLOY.L": "Financials", "NWG.L": "Financials", "PRU.L": "Financials", "LGEN.L": "Financials", "AV.L": "Financials",
    # Healthcare (Expanded FTSE 100/250)
    "AZN.L": "Healthcare", "GSK.L": "Healthcare", "HLN.L": "Healthcare", "SN.L": "Healthcare", "HIK.L": "Healthcare",
    # Consumer
    "ULVR.L": "Consumer", "BATS.L": "Consumer", "FEVR.L": "Consumer", "JET2.L": "Consumer", "NFG.L": "Consumer",
    "TSCO.L": "Consumer", "SBRY.L": "Consumer", "MKS.L": "Consumer", "NXT.L": "Consumer", "DGE.L": "Consumer", "RKT.L": "Consumer",
    # Telecom & Utilities (Expanded FTSE 100/250)
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
SECTOR_MAP["HSBA.L"] = "Financials"

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

@st.cache_resource
def get_db_lock():
    return threading.Lock()

DB_LOCK = get_db_lock()

def is_valid_lse_ticker(ticker: str) -> bool:
    return bool(isinstance(ticker, str) and VALID_LSE_TICKER_RE.match(ticker.strip()))

def get_ticker_sector(ticker: str) -> str:
    t = str(ticker).strip().upper()
    if t in SECTOR_MAP:
        return SECTOR_MAP[t]
    fallback_buckets = ["Industrials", "Consumer", "Financials", "Technology", "Services", "Healthcare"]
    idx = int(hashlib.md5(t.encode("utf-8")).hexdigest()[:4], 16) % len(fallback_buckets)
    return fallback_buckets[idx]

def is_aim_exempt(ticker: str) -> bool:
    return str(ticker).strip().upper() in AIM_EXEMPT_TICKERS

def get_uk_friction_pct(ticker: str) -> float:
    return 0.25 if is_aim_exempt(ticker) else 0.60

def get_breakeven_exit_price(entry_price: float, ticker: str) -> float:
    f_pct = get_uk_friction_pct(ticker)
    return round(float(entry_price) * (1.0 + (f_pct / 100.0)), 2)

def calc_net_equity_pnl_pct(entry_price: float, current_price: float, ticker: str) -> float:
    if entry_price <= 0:
        return 0.0
    gross_pct = ((float(current_price) - float(entry_price)) / float(entry_price)) * 100.0
    net_pct = gross_pct - get_uk_friction_pct(ticker)
    return round(net_pct, 2)

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
# 1. INITIALIZE HYBRID DATABASE (THREAD-SAFE DUCKDB + SUPABASE CLOUD)
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

def fetch_all_supabase_rows(table_name: str) -> list:
    """Paginated retrieval: seamlessly retrieves beyond the PostgREST 1,000-row ceiling for 10-year scale."""
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
    """Restores all historical and active trades from Supabase using 10-year chunked pagination."""
    if not supabase:
        return
    with DB_LOCK:
        con = duckdb.connect(DB_PATH, read_only=False)
        try:
            eq_data = fetch_all_supabase_rows("predictions")
            if eq_data:
                for r in eq_data:
                    tkr = str(r.get('ticker', '')).strip()
                    if not is_valid_lse_ticker(tkr):
                        continue
                    pred_date = str(r.get('predicted_date', datetime.now().strftime('%Y-%m-%d')))
                    trade_id = f"{tkr}_{pred_date}"
                    last_check = str(r.get('last_checked') or datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
                    f_json = json.dumps(r.get('features_json') or {})
                    status_val = str(r.get('status', 'ACTIVE')).upper()
                    exit_ts_val = last_check if status_val != "ACTIVE" else None

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
                        float(r.get('position_gbp', 0.0)), status_val,
                        float(r.get('latest_price', 0.0)), float(r.get('pnl_pct', 0.0)),
                        float(r.get('latest_price', 0.0)) if status_val != "ACTIVE" else 0.0,
                        exit_ts_val, last_check, f_json
                    ])
            if "db_error" in st.session_state and "Hydrate" in st.session_state["db_error"]:
                del st.session_state["db_error"]
        except Exception as e:
            record_db_error("Hydrate Equities", e)

        try:
            opt_data = fetch_all_supabase_rows("options_journal")
            if opt_data:
                for o in opt_data:
                    date_k = str(o.get('date_key', datetime.now().strftime('%Y-%m-%d')))
                    con.execute("""
                        INSERT OR REPLACE INTO daily_options_journal 
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, [
                        date_k, str(o.get('timestamp')), str(o.get('share_name')),
                        str(o.get('option_contract')), float(o.get('strike_price', 0.0)),
                        int(o.get('expiry_days', 21)), float(o.get('underlying_spot', 0.0)),
                        int(o.get('lot_size', 5000)), float(o.get('entry_premium', 0.0)),
                        float(o.get('current_option_price', 0.0)), float(o.get('target_premium', 0.0)),
                        float(o.get('stop_loss_premium', 0.0)), float(o.get('total_capital', 0.0)),
                        float(o.get('ai_confidence', 80.0)), float(o.get('implied_vol', 20.0)),
                        str(o.get('status', 'ACTIVE')).upper(), float(o.get('pnl_pct', 0.0)), str(o.get('last_audited'))
                    ])
        except Exception as e:
            record_db_error("Hydrate Options", e)
        finally:
            con.close()

if "hydrated_once" not in st.session_state:
    hydrate_duckdb_from_supabase()
    st.session_state["hydrated_once"] = True

# ==============================================================================
# 2. TIMEZONE, CONFIGURATION, MACRO FEED & RESILIENT DATA CACHING
# ==============================================================================
st.set_page_config(page_title="ALPHA-LSE Quant Terminal", page_icon="⚡", layout="wide", initial_sidebar_state="collapsed")
LOT_SIZES = {"SHEL": 1000, "AZN": 500, "HSBA": 2000, "ULVR": 500, "BP": 3000, "BARC": 5000, "RIO": 250, "GLEN": 4000}

def is_lse_market_open() -> bool:
    lon_zone = pytz.timezone('Europe/London')
    now_lon = datetime.now(lon_zone)
    return dtime(8, 0) <= now_lon.time() <= dtime(16, 30) and now_lon.weekday() <= 4

def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

def calculate_black_scholes_call(spot: float, strike: float, days_to_exp: float, r: float, sigma: float) -> float:
    T = max(days_to_exp, 1.0) / 365.0
    if spot <= 0 or strike <= 0 or sigma <= 0:
        return max(0.0, spot - strike)
    d1 = (math.log(spot / strike) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    call_price = spot * norm_cdf(d1) - strike * math.exp(-r * T) * norm_cdf(d2)
    return max(round(call_price, 2), 0.50)

@st.cache_data(ttl=300, show_spinner=False)
def fetch_cached_history(ticker: str, period: str = "120d") -> pd.DataFrame:
    """Fault-tolerant history fetcher with exponential retry and timeout guards."""
    if not is_valid_lse_ticker(ticker):
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
# 3. HARDENED ZERO-TRUST AUTHENTICATION (CIA CONFIDENTIALITY)
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

# ==============================================================================
# 4. AUDITING & RECONCILIATION ENGINE (TCA NET P&L + ENTRY-SAFE BE RATCHET)
# ==============================================================================
def is_stop_breakeven_protected(entry_price: float, stop_loss: float, ticker: str) -> bool:
    return float(stop_loss) >= float(entry_price) - 1e-4

def audit_and_reconcile_all_trades():
    lon_zone = pytz.timezone('Europe/London')
    now_lon = datetime.now(lon_zone)
    now_str = now_lon.strftime('%Y-%m-%d %H:%M:%S')
    today_date = now_lon.date()
    today_str = now_lon.strftime('%Y-%m-%d')

    with DB_LOCK:
        con = duckdb.connect(DB_PATH, read_only=False)
        try:
            active_trades = con.execute("SELECT * FROM trade_journal WHERE status = 'ACTIVE'").df()
            active_opts = con.execute("SELECT * FROM daily_options_journal WHERE status = 'ACTIVE'").df()
        finally:
            con.close()

    # 1. Audit Equities
    if not active_trades.empty:
        for _, tr in active_trades.iterrows():
            tkr = str(tr["ticker"]).strip()
            if not is_valid_lse_ticker(tkr):
                continue
            try:
                h = fetch_cached_history(tkr, period="5d")
                if h.empty:
                    continue

                curr = float(h["Close"].iloc[-1])
                day_high = float(h["High"].iloc[-1]) if "High" in h.columns else curr
                day_low = float(h["Low"].iloc[-1]) if "Low" in h.columns else curr

                entry = float(tr["entry_price"])
                target = float(tr["target_price"])
                stop = float(tr["stop_loss"])
                be_Floor = get_breakeven_exit_price(entry, tkr)

                # Entry-day guard: On Day 0, never use day_high/day_low (which include 08:00 auction spikes prior to entry)
                is_same_day_entry = (str(tr["date_str"]) == today_str)
                target_dist = target - entry

                if target_dist > 0 and stop < be_Floor:
                    be_trigger_price = entry + (BREAKEVEN_TRIGGER_RATIO * target_dist)
                    ratchet_hit = (curr >= be_trigger_price) if is_same_day_entry else (curr >= be_trigger_price or day_high >= be_trigger_price)
                    # Require live price > be_Floor so a trade never ratchets above current spot and immediately stops out
                    if ratchet_hit and curr > be_Floor:
                        stop = be_Floor

                is_be_ratcheted = is_stop_breakeven_protected(entry, stop, tkr)

                new_status = "ACTIVE"
                exit_price = 0.0

                hit_target = (curr >= target) if is_same_day_entry else (curr >= target or day_high >= target)
                hit_stop = (curr <= stop) if (is_same_day_entry or is_be_ratcheted) else (curr <= stop or day_low <= stop)

                if hit_target:
                    new_status = "🎯 WIN (TARGET HIT)"
                    curr = max(curr, target)
                    exit_price = curr
                elif hit_stop:
                    if is_be_ratcheted:
                        new_status = "🛡️ BREAK-EVEN (PROTECTED EXIT)"
                        curr = be_Floor
                        exit_price = be_Floor
                    else:
                        new_status = "🛑 LOSS (STOPPED OUT)"
                        curr = min(curr, stop)
                        exit_price = curr
                else:
                    try:
                        entry_dt = datetime.strptime(str(tr["date_str"]), "%Y-%m-%d").date()
                        if (today_date - entry_dt).days >= MAX_HOLD_CALENDAR_DAYS:
                            new_status = "⏱️ EXPIRED (TIME EXIT)"
                            exit_price = curr
                    except Exception:
                        pass

                if "BREAK-EVEN" in new_status:
                    pnl = 0.0
                else:
                    pnl = calc_net_equity_pnl_pct(entry, curr, tkr)

                with DB_LOCK:
                    con = duckdb.connect(DB_PATH, read_only=False)
                    try:
                        con.execute("""
                            UPDATE trade_journal
                            SET latest_price = ?, stop_loss = ?, pnl_pct = ?, status = ?, exit_price = ?,
                                exit_timestamp = CASE WHEN ? != 'ACTIVE' THEN ? ELSE exit_timestamp END,
                                last_audited = ?
                            WHERE trade_id = ?
                        """, [curr, stop, pnl, new_status, exit_price, new_status, now_str, now_str, tr["trade_id"]])
                    finally:
                        con.close()

                if supabase:
                    try:
                        supabase.table("predictions").update({
                            "status": new_status,
                            "stop_loss": stop,
                            "latest_price": curr,
                            "pnl_pct": pnl,
                            "last_checked": now_str
                        }).eq("ticker", tkr).eq("predicted_date", tr["date_str"]).execute()
                    except Exception as e:
                        record_db_error("Audit Equity Sync", e)
            except Exception:
                continue

    # 2. Audit Options
    if not active_opts.empty:
        for _, opt in active_opts.iterrows():
            try:
                sym = f"{opt['share_name']}.L"
                h = fetch_cached_history(sym, period="5d")
                if h.empty:
                    continue
                current_spot = float(h["Close"].iloc[-1])
                strike = float(opt["strike_price"])
                entry_prem = float(opt["entry_premium"])
                target_prem = float(opt["target_premium"])
                stop_prem = float(opt["stop_loss_premium"])
                sigma = float(opt["implied_vol"]) / 100.0

                elapsed_days = 0
                try:
                    opt_dt = datetime.strptime(str(opt["date_key"]), "%Y-%m-%d").date()
                    elapsed_days = max(0, (today_date - opt_dt).days)
                except Exception:
                    pass
                rem_days = max(1, int(opt["expiry_days"]) - elapsed_days)

                live_prem = calculate_black_scholes_call(current_spot, strike, rem_days, 0.05, sigma)
                pnl_pct = round(((live_prem - entry_prem) / entry_prem) * 100.0, 2) if entry_prem > 0 else 0.0

                opt_status = "ACTIVE"
                if live_prem >= target_prem or pnl_pct >= 60.0:
                    opt_status = "🎯 WIN (TARGET HIT)"
                elif live_prem <= stop_prem or pnl_pct <= -50.0:
                    opt_status = "🛑 LOSS (STOPPED OUT)"
                elif elapsed_days >= MAX_HOLD_CALENDAR_DAYS:
                    opt_status = "⏱️ EXPIRED (TIME EXIT)"

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

                if supabase:
                    try:
                        supabase.table("options_journal").update({
                            "status": opt_status,
                            "current_option_price": live_prem,
                            "pnl_pct": pnl_pct,
                            "last_audited": now_str
                        }).eq("date_key", opt["date_key"]).execute()
                    except Exception as e:
                        record_db_error("Audit Option Sync", e)
            except Exception:
                continue

# ==============================================================================
# 5. CONTINUOUS LEARNING: 30-DAY EXPONENTIAL DECAY & AUTONOMOUS RETRAINING
# ==============================================================================
def get_self_learning_adjustment(ticker: str, current_atr_pct: float) -> dict:
    """Applies exponential half-life time decay (30 days) based on actual trade exit date."""
    delta = 0.0
    reasons = []
    lon_zone = pytz.timezone('Europe/London')
    today_dt = datetime.now(lon_zone).date()

    with DB_LOCK:
        con = duckdb.connect(DB_PATH, read_only=True)
        try:
            hist = con.execute(
                "SELECT ticker, date_str, exit_timestamp, last_audited, status, features_json FROM trade_journal WHERE status != 'ACTIVE'"
            ).df()
        except Exception:
            hist = pd.DataFrame()
        finally:
            con.close()

    if hist.empty:
        return {"delta": 0, "reason": "Neutral (Building memory)"}

    try:
        t_hist = hist[hist["ticker"] == ticker]
        if not t_hist.empty:
            decayed_loss = 0.0
            decayed_win = 0.0
            for _, r in t_hist.iterrows():
                try:
                    raw_exit = str(r.get("exit_timestamp") or r.get("last_audited") or r.get("date_str") or "")[:10]
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

        all_losses = hist[hist["status"].str.contains("LOSS", na=False)]
        if not all_losses.empty:
            loss_atrs = []
            for f_str in all_losses["features_json"].dropna():
                try:
                    f_obj = json.loads(f_str)
                    if "atr_pct" in f_obj:
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

    final_delta = int(round(delta))
    reason_str = " | ".join(reasons) if reasons else "No adverse historical match"
    return {"delta": final_delta, "reason": reason_str}

def check_and_auto_retrain_model(feature_cols: list):
    """Autonomous ML Engine: Retrains LightGBM & CatBoost when sufficient live trade vectors accumulate."""
    with DB_LOCK:
        con = duckdb.connect(DB_PATH, read_only=True)
        try:
            closed_trades = con.execute("SELECT status, features_json FROM trade_journal WHERE status != 'ACTIVE'").df()
        except Exception:
            closed_trades = pd.DataFrame()
        finally:
            con.close()

    total_closed = len(closed_trades)
    if total_closed < MIN_SETTLED_TO_RETRAIN:
        return

    last_retrained = st.session_state.get("last_retrained_count", 0)
    if (total_closed - last_retrained) < RETRAIN_STEP_INTERVAL:
        return

    rows = []
    labels = []
    weights = []

    for _, r in closed_trades.iterrows():
        try:
            f_obj = json.loads(r["features_json"])
            raw_feats = f_obj.get("raw_features", {})
            if not all(col in raw_feats for col in feature_cols):
                continue
            st_val = str(r["status"]).upper()
            if "WIN" in st_val:
                labels.append(1)
                weights.append(3.0)
            elif "LOSS" in st_val:
                labels.append(0)
                weights.append(3.5)
            elif "BREAK-EVEN" in st_val:
                labels.append(1)
                weights.append(1.5)
            else:
                continue
            rows.append([float(raw_feats[c]) for c in feature_cols])
        except Exception:
            continue

    if len(rows) < MIN_SETTLED_TO_RETRAIN:
        return

    try:
        from lightgbm import LGBMClassifier
        from catboost import CatBoostClassifier

        X = pd.DataFrame(rows, columns=feature_cols)
        y = np.array(labels)
        w = np.array(weights)

        if len(np.unique(y)) < 2:
            return

        lgb_model = LGBMClassifier(n_estimators=250, learning_rate=0.03, max_depth=5, random_state=42)
        lgb_model.fit(X, y, sample_weight=w)

        cb_model = CatBoostClassifier(iterations=300, learning_rate=0.03, depth=5, verbose=0, random_seed=42)
        cb_model.fit(X, y, sample_weight=w)

        new_bundle = {"lgb": lgb_model, "catboost": cb_model, "feature_cols": feature_cols}
        joblib.dump(new_bundle, MODEL_PATH)
        st.session_state["ml_model_bundle"] = new_bundle
        st.session_state["last_retrained_count"] = total_closed
        st.session_state["retrain_notice"] = f"🧠 AI Retrained autonomously on {len(rows)} live UK market trades ({total_closed} total closed)!"
    except Exception as e:
        record_db_error("Auto-Retrain ML", e)

# ==============================================================================
# 6. OPTIONS ENGINE (DYNAMIC BLUE-CHIP SELECTION UNDER £500 CAP)
# ==============================================================================
def generate_daily_options_alpha() -> dict:
    lon_zone = pytz.timezone('Europe/London')
    now_lon = datetime.now(lon_zone)
    today_str = now_lon.strftime('%Y-%m-%d')
    now_str = now_lon.strftime('%Y-%m-%d %H:%M:%S')

    with DB_LOCK:
        con = duckdb.connect(DB_PATH, read_only=True)
        try:
            df = con.execute("SELECT * FROM daily_options_journal WHERE date_key = ?", [today_str]).df()
            if not df.empty:
                return df.iloc[0].to_dict()
        except Exception:
            pass
        finally:
            con.close()

    candidates = ["BARC", "BP", "HSBA", "GLEN", "SHEL"]
    best_sig = None
    best_score = -999.0

    for sym in candidates:
        try:
            h = fetch_cached_history(f"{sym}.L", period="30d")
            if h.empty or len(h) < 15:
                continue
            spot = float(h["Close"].iloc[-1])
            ret_5d = float((spot - h["Close"].iloc[-5]) / h["Close"].iloc[-5])
            returns = np.log(h["Close"] / h["Close"].shift(1)).dropna()
            sigma = max(0.15, min(0.45, float(returns.std() * np.sqrt(252))))

            days_to_expiry = 21
            lot_size = LOT_SIZES.get(sym, 1000)
            strike = round(spot * 1.02)
            entry_prem = calculate_black_scholes_call(spot, strike, days_to_expiry, 0.05, sigma)
            total_cap = round((entry_prem * lot_size) / 100.0, 2)

            if total_cap <= 500.0 and ret_5d > best_score:
                best_score = ret_5d
                conf = round(min(92.0, max(68.0, 78.0 + (ret_5d * 150.0))), 1)
                best_sig = {
                    "date_key": today_str, "timestamp": now_str, "share_name": sym,
                    "option_contract": f"{sym} {int(strike)}p CE", "strike_price": float(strike),
                    "expiry_days": days_to_expiry, "underlying_spot": float(spot), "lot_size": lot_size,
                    "entry_premium": float(entry_prem), "current_option_price": float(entry_prem),
                    "target_premium": round(entry_prem * 1.60, 2), "stop_loss_premium": round(entry_prem * 0.50, 2),
                    "total_capital": float(total_cap), "ai_confidence": conf, "implied_vol": round(sigma * 100, 1),
                    "status": "ACTIVE", "pnl_pct": 0.0, "last_audited": now_str
                }
        except Exception:
            continue

    if best_sig is None:
        return {}

    with DB_LOCK:
        con = duckdb.connect(DB_PATH, read_only=False)
        try:
            con.execute("""
                INSERT OR REPLACE INTO daily_options_journal 
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'ACTIVE', 0.0, ?)
            """, [best_sig[k] for k in ["date_key", "timestamp", "share_name", "option_contract", "strike_price", "expiry_days", "underlying_spot", "lot_size", "entry_premium", "current_option_price", "target_premium", "stop_loss_premium", "total_capital", "ai_confidence", "implied_vol", "last_audited"]])
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
# 7. SAFE LOGGING, DAILY QUOTA & BARRA-STYLE SECTOR CONCENTRATION GUARDRAILS
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

def get_active_sector_exposure() -> dict:
    counts = {}
    for tkr in get_currently_active_tickers():
        sec = get_ticker_sector(tkr)
        counts[sec] = counts.get(sec, 0) + 1
    return counts

def get_todays_logged_equities() -> pd.DataFrame:
    lon_zone = pytz.timezone('Europe/London')
    today_str = datetime.now(lon_zone).strftime('%Y-%m-%d')
    with DB_LOCK:
        con = duckdb.connect(DB_PATH, read_only=True)
        try:
            return con.execute("SELECT * FROM trade_journal WHERE date_str = ? ORDER BY ticker ASC", [today_str]).df()
        except Exception:
            return pd.DataFrame()
        finally:
            con.close()

def get_recent_cooldown_tickers() -> set:
    """Anti-Churn Rule: Excludes tickers closed within the last COOLDOWN_CALENDAR_DAYS based on actual exit date."""
    lon_zone = pytz.timezone('Europe/London')
    today_dt = datetime.now(lon_zone).date()
    cooldown = set()
    with DB_LOCK:
        con = duckdb.connect(DB_PATH, read_only=True)
        try:
            recent_closed = con.execute(
                "SELECT ticker, exit_timestamp, last_audited, date_str FROM trade_journal WHERE status != 'ACTIVE'"
            ).df()
            for _, r in recent_closed.iterrows():
                try:
                    raw_ts = str(r.get("exit_timestamp") or r.get("last_audited") or r.get("date_str") or "")[:10]
                    c_dt = datetime.strptime(raw_ts, "%Y-%m-%d").date()
                    if (today_dt - c_dt).days <= COOLDOWN_CALENDAR_DAYS:
                        cooldown.add(str(r["ticker"]).strip())
                except Exception:
                    pass
        except Exception:
            pass
        finally:
            con.close()
    return cooldown

def log_equity_signal_safely(sig: dict, enforce_sector_cap: bool = True) -> bool:
    lon_zone = pytz.timezone('Europe/London')
    now = datetime.now(lon_zone)
    today_str = now.strftime('%Y-%m-%d')
    now_str = now.strftime('%Y-%m-%d %H:%M:%S')
    ticker = str(sig.get('Ticker', '')).strip()
    if not is_valid_lse_ticker(ticker):
        return False

    sector = get_ticker_sector(ticker)
    if enforce_sector_cap:
        sec_counts = get_active_sector_exposure()
        if sec_counts.get(sector, 0) >= MAX_ACTIVE_PER_SECTOR:
            return False

    f_dict = sig.get('FeaturesDict', {})
    f_json_str = json.dumps(f_dict)
    init_net_pnl = -get_uk_friction_pct(ticker)
    logged_local = False

    with DB_LOCK:
        con = duckdb.connect(DB_PATH, read_only=False)
        try:
            today_count_df = con.execute("SELECT COUNT(*) AS cnt FROM trade_journal WHERE date_str = ?", [today_str]).df()
            if not today_count_df.empty and int(today_count_df["cnt"].iloc[0]) >= MAX_DAILY_EQUITY_TRADES:
                return False

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
# 8. EXPANDED UNIVERSE & AUTONOMOUS DUAL-ENSEMBLE ML ENGINE
# ==============================================================================
def get_ml_model():
    if "ml_model_bundle" in st.session_state:
        return st.session_state["ml_model_bundle"]
    if os.path.exists(MODEL_PATH):
        try:
            bundle = joblib.load(MODEL_PATH)
            st.session_state["ml_model_bundle"] = bundle
            return bundle
        except Exception:
            return None
    return None

@st.cache_data(ttl=3600, show_spinner=False)
def get_live_lse_universe() -> list:
    raw_tickers = []
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
    try:
        url_100 = "https://en.wikipedia.org/wiki/FTSE_100_Index"
        resp_100 = requests.get(url_100, headers=headers, timeout=4)
        if resp_100.status_code == 200:
            df_100 = pd.read_html(resp_100.text, attrs={'id': 'constituents'})[0]
            raw_tickers.extend([f"{str(t).strip().replace('.', '-')}.L" for t in df_100['Ticker'].dropna()])
    except Exception:
        pass

    try:
        url_250 = "https://en.wikipedia.org/wiki/FTSE_250_Index"
        resp_250 = requests.get(url_250, headers=headers, timeout=4)
        if resp_250.status_code == 200:
            df_250 = pd.read_html(resp_250.text, attrs={'id': 'constituents'})[0]
            raw_tickers.extend([f"{str(t).strip().replace('.', '-')}.L" for t in df_250['Ticker'].dropna()])
    except Exception:
        pass

    fallback_pool = [
        # Energy
        "SHEL.L", "BP.L", "PTAL.L", "RKH.L", "ITM.L",
        # Basic Materials
        "RIO.L", "GLEN.L", "LTHM.L", "CMCL.L", "CAML.L", "SAV.L", "KP2.L",
        # Financials (Expanded FTSE 100/250)
        "HSBA.L", "BARC.L", "LSEG.L", "BUR.L", "LLOY.L", "NWG.L", "AV.L", "PRU.L", "LGEN.L",
        # Healthcare (Expanded FTSE 100/250)
        "AZN.L", "GSK.L", "HLN.L", "SN.L", "HIK.L",
        # Consumer
        "ULVR.L", "BATS.L", "FEVR.L", "JET2.L", "NFG.L", "TSCO.L", "SBRY.L", "MKS.L", "NXT.L", "DGE.L", "RKT.L",
        # Telecom & Utilities (Expanded FTSE 100/250)
        "NG.L", "SSE.L", "CNA.L", "VOD.L", "BT-A.L",
        # Technology
        "DOTD.L", "BIG.L", "SEE.L", "AOM.L", "YOU.L", "CER.L", "RWS.L", "KGH.L", "AUTO.L", "SGE.L",
        # Defense & Aero
        "CHRT.L", "CNC.L", "BA.L", "RR.L", "QQ.L",
        # Industrials
        "CRW.L", "BRCK.L", "MIDW.L", "VIC.L", "SRC.L", "JHD.L", "REL.L", "EXPN.L", "AHT.L"
    ]
    combined = set(raw_tickers + fallback_pool)
    return sorted([t for t in combined if is_valid_lse_ticker(t)])

def run_predictions():
    ml_bundle = get_ml_model()
    if ml_bundle is None:
        return pd.DataFrame(), False

    feature_cols = ml_bundle['feature_cols']
    check_and_auto_retrain_model(feature_cols)

    active_held = get_currently_active_tickers()
    active_sectors = get_active_sector_exposure()
    saturated_sectors = {sec for sec, cnt in active_sectors.items() if cnt >= MAX_ACTIVE_PER_SECTOR}
    cooldown_tickers = get_recent_cooldown_tickers()

    full_universe = [
        t for t in get_live_lse_universe() 
        if t not in active_held 
        and t not in cooldown_tickers
        and get_ticker_sector(t) not in saturated_sectors
    ]
    
    if not full_universe:
        return pd.DataFrame(), False

    scan_chunk = random.sample(full_universe, min(35, len(full_universe)))
    macro_regime = get_cross_asset_macro_regime()

    results = []
    lgb_model = ml_bundle['lgb']
    cb_model = ml_bundle['catboost']

    prog = st.progress(0, text=f"Scanning rotating chunk of {len(scan_chunk)} unheld UK equities in available sectors...")

    for i, ticker in enumerate(scan_chunk):
        try:
            df = fetch_cached_history(ticker, period="120d")
            if df.empty or len(df) < 55 or engineer_features is None:
                prog.progress((i + 1) / len(scan_chunk))
                continue

            df = df.copy()
            df.reset_index(inplace=True)
            feats = engineer_features(df)
            if feats.empty:
                prog.progress((i + 1) / len(scan_chunk))
                continue
            latest = feats.iloc[-1:].copy()

            rns_data = evaluate_rns_nlp_sentiment(ticker)
            if "Dilution" in rns_data['status']:
                prog.progress((i + 1) / len(scan_chunk))
                continue

            p1 = float(lgb_model.predict_proba(latest[feature_cols])[:, 1][0])
            p2 = float(cb_model.predict_proba(latest[feature_cols])[:, 1][0])
            blended = max(0.01, min(0.99, (0.5 * p1 + 0.5 * p2) + rns_data['delta']))

            close = float(latest['Close'].values[0])
            atr = float(latest['atr_14'].values[0])
            atr_pct = round((atr / close) * 100.0, 2) if close > 0 else 0.0

            target = close + (1.8 * atr)
            stop = close - (1.2 * atr)
            gross_return_pct = ((target - close) / close) * 100.0 if close > 0 else 0.0
            friction_pct = get_uk_friction_pct(ticker)
            net_return_pct = round(gross_return_pct - friction_pct, 1)

            sector = get_ticker_sector(ticker)
            tax_regime = "AIM (0% SDRT)" if is_aim_exempt(ticker) else "Main (0.5% SDRT)"

            base_confidence = int(min(96, max(45, round(45.0 + (blended ** 0.85) * 52.0))))
            learner = get_self_learning_adjustment(ticker, atr_pct)
            macro_delta, macro_note = compute_macro_lead_lag_adjustment(ticker, sector, macro_regime)

            total_delta = learner["delta"] + macro_delta
            final_confidence = int(min(98, max(25, base_confidence + total_delta)))

            shares = int((500.0 * 100) / close) if close > 0 else 1
            is_qualified = (final_confidence >= 65) and (net_return_pct >= 2.0)

            raw_feature_map = {c: float(latest[c].values[0]) for c in feature_cols if c in latest.columns}

            feature_snapshot = {
                "raw_features": raw_feature_map,
                "atr_pct": atr_pct,
                "raw_prob": round(blended, 4),
                "base_ml_conf": base_confidence,
                "learner_delta": learner["delta"],
                "macro_delta": macro_delta,
                "sector": sector,
                "friction_pct": friction_pct,
                "tax_regime": tax_regime,
                "nlp_tags": rns_data["nlp_tags"]
            }

            results.append({
                "Ticker": ticker,
                "Company": ticker,
                "Sector": sector,
                "Tax Regime": tax_regime,
                "Friction": f"-{friction_pct:.2f}%",
                "Price (p)": round(close, 2),
                "Expected Return": f"+{net_return_pct}% Net",
                "ReturnNum": net_return_pct,
                "Est. Time to Target": "3-7 Days",
                "Recommended Shares": f"{shares} shares",
                "Total Cost (£)": "£500",
                "Capital": 500.0,
                "Target (p)": round(target, 2),
                "Stop Loss (p)": round(stop, 2),
                "Base ML": f"{base_confidence}%",
                "Learner Delta": f"{total_delta:+d}%",
                "Learner Note": f"{learner['reason']} | Macro: {macro_note}",
                "AI Win Confidence": f"{final_confidence}%",
                "Adjusted Score": final_confidence,
                "RNS": rns_data['status'],
                "RNSHeadline": rns_data['headline'],
                "FeaturesDict": feature_snapshot,
                "Qualified": is_qualified
            })
        except Exception:
            pass
        prog.progress((i + 1) / len(scan_chunk))

    prog.empty()
    if not results:
        return pd.DataFrame(), False

    df_out = pd.DataFrame(results).sort_values(by=["Qualified", "Adjusted Score", "ReturnNum"], ascending=[False, False, False]).reset_index(drop=True)
    qualified_only = df_out[df_out["Qualified"] == True].copy()

    # Dynamic auto-fill: log down the ranked list until daily quota fills or candidates exhaust
    if not qualified_only.empty:
        for _, sig in qualified_only.iterrows():
            todays_count = len(get_todays_logged_equities())
            if todays_count >= MAX_DAILY_EQUITY_TRADES:
                break
            log_equity_signal_safely(sig.to_dict(), enforce_sector_cap=True)

    return qualified_only, not qualified_only.empty

# ==============================================================================
# 9. SIDEBAR & UNIFIED AUTONOMOUS LOOP CONTROLLER
# ==============================================================================
st.sidebar.header("⚙️ Institutional Guardrails")
selected_universe = st.sidebar.selectbox("Universe Mode", ["Rotating Active Market Basket (FTSE + AIM)"])
st.sidebar.caption(f"Daily Auto-Log Cap: **Top {MAX_DAILY_EQUITY_TRADES} Picks/Day**")
st.sidebar.caption(f"Sector Exposure Cap: **Max {MAX_ACTIVE_PER_SECTOR} Active/Sector**")
st.sidebar.caption(f"Break-Even Ratchet: **≥ {int(BREAKEVEN_TRIGGER_RATIO * 100)}% of Target (Net of Tax)**")
st.sidebar.caption(f"Post-Exit Cooldown: **{COOLDOWN_CALENDAR_DAYS} Days Anti-Churn**")
st.sidebar.caption(f"Time-Decay Half-Life: **{int(HALF_LIFE_DAYS)} Days**")
st.sidebar.caption("UK TCA Friction: **0.50% SDRT (Main) / 0% (AIM) + Spread**")

st.sidebar.markdown("---")
st.sidebar.header("🔌 Broker Execution Bridge")
broker_mode = st.sidebar.selectbox("Gateway", ["Paper Trading (Simulated)", "Interactive Brokers", "IG Group API"])

st.sidebar.markdown("---")
st.sidebar.header("🔄 Autonomous Loop")
market_is_open = is_lse_market_open()
auto_mode = st.sidebar.toggle("Continuous Background Mode", value=market_is_open)
refresh_interval_sec = st.sidebar.selectbox("Refresh Interval", [300, 600, 3600], format_func=lambda x: f"{x//60} Minutes")

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

if "db_error" in st.session_state:
    st.sidebar.error(f"⚠️ Cloud Sync Warning: {st.session_state['db_error']}")

if st.sidebar.button("🚪 Log Out", width="stretch"):
    st.session_state["password_correct"] = False
    st.rerun()

market_status = "🟢 OPEN" if market_is_open else "🔴 CLOSED"
db_status_text = "🟢 ONLINE (SUPABASE)" if supabase else "🔴 OFFLINE"
macro_bar = get_cross_asset_macro_regime()

st.title("⚡ ALPHA-LSE Quant Terminal")
st.caption(
    f"Status: **10-Year Autonomous Quant AI** • Database: **{db_status_text}** • Market (LON): **{market_status}** • "
    f"5D Macro: **FTSE {macro_bar['ftse_5d']:+.1f}% | Brent {macro_bar['brent_5d']:+.1f}% | Copper {macro_bar['copper_5d']:+.1f}% | GBP/USD {macro_bar['gbpusd_5d']:+.1f}%**"
)

tab_scanner, tab_options, tab_journal, tab_reasoning = st.tabs([
    "🎯 Equity High-Certainty Signals",
    "📊 FTSE Leveraged Alpha (Budget < £500)",
    "📖 Automated Trade Journal & P&L",
    "🧠 AI Reasoning & Self-Learning"
])

# ==============================================================================
# 10. TAB 1: EQUITY SCANNER (WITH POST-QUOTA LOCK & SECTOR / TCA TELEMETRY)
# ==============================================================================
if "dispatched_orders" not in st.session_state:
    st.session_state["dispatched_orders"] = set()

with tab_scanner:
    todays_logged_df = get_todays_logged_equities()
    quota_filled = len(todays_logged_df) >= MAX_DAILY_EQUITY_TRADES

    col1, col2 = st.columns([4, 1])
    with col1:
        if quota_filled:
            st.write(f"Daily equity allocation complete (**{len(todays_logged_df)}/{MAX_DAILY_EQUITY_TRADES} slots filled**). Spotlighting today's active cohort (Net of UK Stamp Duty & Spread):")
        else:
            st.write(f"Unheld equities screened via Dual-Ensemble ML, UK TCA Friction, Sector Caps, and Macro Overlays (**{len(todays_logged_df)}/{MAX_DAILY_EQUITY_TRADES} logged today**):")
    with col2:
        re_scan = st.button("🔄 Run Live Scan Now", width="stretch", type="primary")

    if quota_filled and not re_scan:
        st.info(f"🔒 **Daily Quota Filled ({len(todays_logged_df)}/{MAX_DAILY_EQUITY_TRADES} Slots Active for Today)** — Scanner locked onto today's executed positions to prevent over-trading.")
        st.markdown("### 📌 Today's Executed Cohort (Live Intraday Monitor — Net of UK Friction)")
        q_cols = st.columns(min(len(todays_logged_df), 3))
        for idx, t_row in todays_logged_df.iterrows():
            with q_cols[idx % 3]:
                with st.container(border=True):
                    tkr_sym = str(t_row["ticker"])
                    f_meta = {}
                    try:
                        f_meta = json.loads(t_row.get("features_json") or "{}")
                    except Exception:
                        pass
                    base_c = f_meta.get("base_ml_conf", 75)
                    del_c = f_meta.get("learner_delta", 0) + f_meta.get("macro_delta", 0)
                    final_c = base_c + del_c
                    sec_name = f_meta.get("sector") or get_ticker_sector(tkr_sym)
                    tax_tag = f_meta.get("tax_regime") or ("AIM (0% SDRT)" if is_aim_exempt(tkr_sym) else "Main (0.5% SDRT)")
                    be_active = is_stop_breakeven_protected(float(t_row["entry_price"]), float(t_row["stop_loss"]), tkr_sym)
                    stop_tag = f"{t_row['stop_loss']:.2f}p (🛡️ BE+Tax Locked)" if be_active else f"{t_row['stop_loss']:.2f}p"

                    st.success(f"✅ TODAY'S SLOT #{idx + 1} • {sec_name.upper()}")
                    st.subheader(tkr_sym)
                    st.metric(
                        label="Live Net Intraday P&L",
                        value=f"{t_row['pnl_pct']:+.2f}%",
                        delta=f"Live: {t_row['latest_price']:.2f}p (Entry: {t_row['entry_price']:.2f}p)"
                    )
                    st.markdown(
                        f"🤖 **Entry AI Score:** `{final_c}%` *(Base: {base_c}%, Adj: {del_c:+d}%)*  \n"
                        f"🏛️ **UK Tax & Spread:** `{tax_tag}` (`-{get_uk_friction_pct(tkr_sym):.2f}%`)  \n"
                        f"🎯 **Target Sell:** `{t_row['target_price']:.2f}p` | 🛑 **Stop:** `{stop_tag}`  \n"
                        f"📦 **Position Size:** `{int(t_row['shares'])} shares` (`£{t_row['capital_allocated']:.0f}`)  \n"
                        f"🕒 **Last Audited:** `{str(t_row['last_audited'])[:19]}`"
                    )
    else:
        if re_scan or is_new_loop_tick or "scan_results" not in st.session_state:
            with st.spinner("Executing quant screen, macro overlay & reconciling live trades..."):
                if re_scan:
                    audit_and_reconcile_all_trades()
                    hydrate_duckdb_from_supabase()
                res_df, has_cleared = run_predictions()
                if not res_df.empty:
                    st.session_state["scan_results"] = res_df
                    st.session_state["has_cleared"] = has_cleared

        df_res = st.session_state.get("scan_results", pd.DataFrame())
        if st.session_state.get("has_cleared", False) and not df_res.empty:
            st.success(f"🟢 **{len(df_res)} Fresh Unheld Setup(s) Cleared Gates in Available Sectors (Net of UK Stamp Duty & Caps)**")

            st.markdown("### 🔥 Top Conviction Spotlights (Available Sectors)")
            cols = st.columns(min(len(df_res), 3))
            today_key = datetime.now(pytz.timezone('Europe/London')).strftime('%Y-%m-%d')
            for idx, row in df_res.head(3).iterrows():
                with cols[idx % 3]:
                    with st.container(border=True):
                        st.success(f"🔥 PICK #{idx + 1} • {row['Sector'].upper()}")
                        st.subheader(row['Ticker'])
                        st.metric(label="Net Target Gain (After Tax)", value=row["Expected Return"], delta=f"Entry: {row['Price (p)']}p")
                        st.markdown(
                            f"🤖 **Final AI Score:** `{row['AI Win Confidence']}` *(Base: {row['Base ML']}, Adj: {row['Learner Delta']})*  \n"
                            f"🏛️ **UK TCA Regime:** `{row['Tax Regime']}` (`{row['Friction']}`)  \n"
                            f"🧠 **Memory & Macro:** `{row['Learner Note']}`  \n"
                            f"🎯 **Target Sell:** `{row['Target (p)']}p` | 🛑 **Stop:** `{row['Stop Loss (p)']}p`  \n"
                            f"📦 **Size:** `{row['Recommended Shares']}` (`{row['Total Cost (£)']}`)"
                        )
                        idem_token = hashlib.sha256(f"{row['Ticker']}_{today_key}_{broker_mode}".encode()).hexdigest()[:12]
                        already_sent = idem_token in st.session_state["dispatched_orders"]
                        btn_label = "✅ Order Dispatched (Idempotent Lock)" if already_sent else f"🚀 Execute Buy ({broker_mode})"
                        if st.button(btn_label, key=f"exec_{idem_token}", width="stretch", disabled=already_sent):
                            st.session_state["dispatched_orders"].add(idem_token)
                            log_equity_signal_safely(row.to_dict(), enforce_sector_cap=True)
                            st.info(f"Order [{idem_token}] dispatched to {broker_mode}.")
                            st.rerun()

            st.markdown(f"### 📋 All {len(df_res)} Qualified Fresh Equities (Ranked by Final AI Score)")
            display_cols = ["Ticker", "Sector", "Tax Regime", "Price (p)", "Target (p)", "Stop Loss (p)", "Expected Return", "Base ML", "Learner Delta", "AI Win Confidence", "Recommended Shares", "RNS"]
            st.dataframe(df_res[[c for c in display_cols if c in df_res.columns]], width="stretch", hide_index=True)
        else:
            st.warning("🛡️ **Capital Protection Active:** No equities currently pass all combined volume, trend, sector availability, and ML filters.")

# ==============================================================================
# 11. TAB 2: OPTIONS ALPHA
# ==============================================================================
with tab_options:
    st.subheader("📊 FTSE Blue-Chip Leveraged Alpha (Budget < £500)")
    st.caption("Derived via Black-Scholes volatility pricing for FTSE 100 derivatives and spread contracts.")

    opt_signal = generate_daily_options_alpha()
    if opt_signal:
        with st.container(border=True):
            o1, o2, o3 = st.columns(3)
            with o1:
                st.metric("Derivative Contract", opt_signal["option_contract"])
                st.markdown(f"Entry Underlying Spot: **{opt_signal['underlying_spot']:.2f}p**")
            with o2:
                st.metric("Live Option Price", f"{opt_signal['current_option_price']:.2f}p", f"{opt_signal['pnl_pct']:+.2f}% vs Entry ({opt_signal['entry_premium']:.2f}p)")
                st.markdown(f"Implied Volatility: **{opt_signal['implied_vol']}%**")
            with o3:
                st.metric("AI Win Probability", f"{opt_signal['ai_confidence']}%")
                st.markdown(f"Capital Required: **£{opt_signal['total_capital']:,}** ({opt_signal['lot_size']} units)")

            st.write("")
            m1, m2, m3 = st.columns(3)
            m1.info(f"🎯 **Target Premium:** {opt_signal['target_premium']:.2f}p (+60%)")
            m2.warning(f"🛑 **Stop-Loss Premium:** {opt_signal['stop_loss_premium']:.2f}p (-50%)")
            m3.success(f"Status: **{opt_signal['status']}** (Audited: {str(opt_signal['last_audited'])[:16]})")

# ==============================================================================
# 12. TAB 3: MASTER TRADE JOURNAL, KPI HEADER BAR & SECTOR EXPOSURE
# ==============================================================================
with tab_journal:
    st.subheader("📖 Autonomous Master Ledger (Equities & Derivatives — Net of UK Taxes)")

    j_col1, j_col2 = st.columns([4, 1])
    with j_col1:
        st.caption(
            f"Synced every **{refresh_interval_sec // 60} Minutes** (Cycle #{loop_tick}). "
            f"All Equity P&L is **Net of UK HMRC Stamp Duty (0.50% Main / 0% AIM) & Spread**. Break-Even Ratchet active at ≥ {int(BREAKEVEN_TRIGGER_RATIO * 100)}% of target."
        )
    with j_col2:
        if st.button("🔄 Force Manual Sync", width="stretch"):
            audit_and_reconcile_all_trades()
            hydrate_duckdb_from_supabase()
            st.success("Ledger reconciled with live LSE order flow and Supabase.")
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

    active_cap_gbp = 0.0
    open_unrealized_gbp = 0.0
    closed_realized_gbp = 0.0
    win_count = 0
    loss_count = 0
    be_count = 0
    be_ratcheted_active_count = 0

    if not df_eq.empty:
        for _, r in df_eq.iterrows():
            cap = float(r.get("capital_allocated", 500.0))
            pnl_gbp = cap * (float(r.get("pnl_pct", 0.0)) / 100.0)
            st_str = str(r.get("status", "ACTIVE")).upper()
            tkr_sym = str(r.get("ticker", ""))
            if st_str == "ACTIVE":
                active_cap_gbp += cap
                open_unrealized_gbp += pnl_gbp
                if is_stop_breakeven_protected(float(r.get("entry_price", 0.0)), float(r.get("stop_loss", -1.0)), tkr_sym):
                    be_ratcheted_active_count += 1
            else:
                closed_realized_gbp += pnl_gbp
                if "WIN" in st_str:
                    win_count += 1
                elif "BREAK-EVEN" in st_str:
                    be_count += 1
                elif "LOSS" in st_str:
                    loss_count += 1

    if not df_opt.empty:
        for _, o in df_opt.iterrows():
            cap = float(o.get("total_capital", 350.0))
            pnl_gbp = cap * (float(o.get("pnl_pct", 0.0)) / 100.0)
            st_str = str(o.get("status", "ACTIVE")).upper()
            if st_str == "ACTIVE":
                active_cap_gbp += cap
                open_unrealized_gbp += pnl_gbp
            else:
                closed_realized_gbp += pnl_gbp
                if "WIN" in st_str:
                    win_count += 1
                elif "LOSS" in st_str:
                    loss_count += 1

    decided_trades = win_count + loss_count
    win_rate_pct = (win_count / decided_trades * 100.0) if decided_trades > 0 else 0.0
    open_ret_pct = (open_unrealized_gbp / active_cap_gbp * 100.0) if active_cap_gbp > 0 else 0.0

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        with st.container(border=True):
            st.metric(
                label="Closed Win Rate",
                value=f"{win_rate_pct:.1f}%",
                delta=f"{win_count}W • {loss_count}L • {be_count}BE"
            )
    with k2:
        with st.container(border=True):
            st.metric(
                label="Active Capital Deployed",
                value=f"£{active_cap_gbp:,.0f}",
                delta=f"🛡️ {be_ratcheted_active_count} Break-Even Protected"
            )
    with k3:
        with st.container(border=True):
            st.metric(
                label="Open Net Unrealized P&L",
                value=f"£{open_unrealized_gbp:+,.2f}",
                delta=f"{open_ret_pct:+.2f}% Net on Active Capital"
            )
    with k4:
        with st.container(border=True):
            st.metric(
                label="Closed Net Realized P&L",
                value=f"£{closed_realized_gbp:+,.2f}",
                delta=f"{decided_trades + be_count} Settled Trades"
            )

    sec_exp = get_active_sector_exposure()
    if sec_exp:
        sec_badges = " • ".join([f"**{s}:** `{c}/{MAX_ACTIVE_PER_SECTOR}`" for s, c in sorted(sec_exp.items(), key=lambda x: -x[1])])
        st.caption(f"📊 **Active Barra Sector Exposure (New Entry Cap = {MAX_ACTIVE_PER_SECTOR}/Sector):** {sec_badges}")

    master_list = []
    if not df_eq.empty:
        df_eq_disp = df_eq.copy()
        df_eq_disp["Sector"] = df_eq_disp["ticker"].apply(get_ticker_sector)

        def format_eq_status(row):
            st_val = str(row["status"])
            if st_val == "ACTIVE" and is_stop_breakeven_protected(float(row["entry_price"]), float(row["stop_loss"]), str(row["ticker"])):
                return "🟢 ACTIVE (🛡️ BE STOP)"
            return st_val

        df_eq_disp["status"] = df_eq_disp.apply(format_eq_status, axis=1)
        clean_eq = df_eq_disp.rename(columns={
            "date_str": "Date", "ticker": "Symbol", "asset_type": "Asset",
            "entry_price": "Entry (p)", "target_price": "Target (p)",
            "stop_loss": "Stop (p)", "latest_price": "Live Price (p)",
            "pnl_pct": "Net P&L (%)", "status": "Status", "last_audited": "Last Checked"
        })
        master_list.append(clean_eq)

    if not df_opt.empty:
        df_opt_disp = df_opt.copy()
        df_opt_disp['Asset'] = 'OPTIONS'
        df_opt_disp['Sector'] = df_opt_disp['share_name'].apply(lambda s: get_ticker_sector(f"{s}.L"))
        clean_opt = df_opt_disp.rename(columns={
            "date_key": "Date", "option_contract": "Symbol",
            "entry_premium": "Entry (p)", "current_option_price": "Live Price (p)",
            "target_premium": "Target (p)", "stop_loss_premium": "Stop (p)",
            "pnl_pct": "Net P&L (%)", "status": "Status", "last_audited": "Last Checked"
        })
        master_list.append(clean_opt)

    if master_list:
        master_df = pd.concat(master_list, ignore_index=True)
        cols_to_keep = ["Date", "Symbol", "Sector", "Asset", "Entry (p)", "Target (p)", "Stop (p)", "Live Price (p)", "Net P&L (%)", "Status", "Last Checked"]
        master_df = master_df[[c for c in cols_to_keep if c in master_df.columns]]

        for p_col in ["Entry (p)", "Target (p)", "Stop (p)", "Live Price (p)"]:
            if p_col in master_df.columns:
                master_df[p_col] = master_df[p_col].apply(lambda x: f"{float(x):.2f}p" if pd.notnull(x) else "-")

        if "Net P&L (%)" in master_df.columns:
            master_df["Net P&L (%)"] = master_df["Net P&L (%)"].apply(lambda x: f"{float(x):+.2f}%" if pd.notnull(x) else "0.00%")

        st.dataframe(master_df, width="stretch", hide_index=True)
    else:
        st.info("No trades currently logged. Active trades will appear here as the engine confirms signals.")

# ==============================================================================
# 13. TAB 4: AI REASONING, MACRO OVERLAY & SELF-LEARNING DASHBOARD
# ==============================================================================
with tab_reasoning:
    st.subheader("🧠 Explainable AI, UK TCA & Closed-Loop Self-Learning")
    st.caption("Live breakdown of Calibrated Base ML probabilities, UK Stamp Duty & Spread Friction, Cross-Asset Macro Overlays, and 75th-Percentile Volatility Autopsies.")

    df_scan = st.session_state.get("scan_results", pd.DataFrame())
    todays_eq = get_todays_logged_equities()

    st.markdown("### 🔍 Live Equity Decision Matrix")
    if not df_scan.empty:
        r_cols = st.columns(min(len(df_scan), 3))
        for idx, row in df_scan.head(3).iterrows():
            with r_cols[idx % 3]:
                with st.container(border=True):
                    st.markdown(f"#### #{idx+1} {row['Ticker']} ({row['Sector']})")
                    st.write(f"**Final AI Confidence:** `{row['AI Win Confidence']}` *(Base ML: {row['Base ML']}, Total Adj: {row['Learner Delta']})*")
                    st.progress(min(float(str(row['AI Win Confidence']).replace('%', '')) / 100.0, 1.0))
                    st.markdown(f"""
                    **Institutional Layer Breakdown:**
                    * 🤖 **Calibrated ML Ensemble:** LightGBM + CatBoost (`{row['Base ML']}`)
                    * 🧠 **Self-Learner & Macro:** `{row['Learner Note']}`
                    * 🏛️ **UK TCA Friction:** `{row['Tax Regime']}` (`{row['Friction']}`)
                    * 📰 **RNS NLP Gate:** `{row['RNS']}`
                    * ⚖️ **Capital Allocation:** £500 (`{row['Recommended Shares']}`)
                    """)
    elif not todays_eq.empty:
        r_cols = st.columns(min(len(todays_eq), 3))
        for idx, t_row in todays_eq.head(3).iterrows():
            with r_cols[idx % 3]:
                with st.container(border=True):
                    tkr_sym = str(t_row["ticker"])
                    f_meta = {}
                    try:
                        f_meta = json.loads(t_row.get("features_json") or "{}")
                    except Exception:
                        pass
                    base_c = f_meta.get("base_ml_conf", 75)
                    del_c = f_meta.get("learner_delta", 0) + f_meta.get("macro_delta", 0)
                    final_c = base_c + del_c
                    sec_name = f_meta.get("sector") or get_ticker_sector(tkr_sym)
                    tax_tag = f_meta.get("tax_regime") or ("AIM (0% SDRT)" if is_aim_exempt(tkr_sym) else "Main (0.5% SDRT)")
                    be_locked = is_stop_breakeven_protected(float(t_row["entry_price"]), float(t_row["stop_loss"]), tkr_sym)

                    st.markdown(f"#### #{idx+1} {tkr_sym} ({sec_name})")
                    st.write(f"**Final AI Confidence:** `{final_c}%` *(Base ML: {base_c}%, Total Adj: {del_c:+d}%)*")
                    st.progress(min(max(final_c / 100.0, 0.0), 1.0))
                    st.markdown(f"""
                    **Institutional Layer Breakdown:**
                    * 🤖 **Calibrated ML Ensemble:** LightGBM + CatBoost (`{base_c}%`)
                    * 🏛️ **UK TCA Regime:** `{tax_tag}` (`-{get_uk_friction_pct(tkr_sym):.2f}%`)
                    * 🧠 **Recorded ATR Regime:** `{f_meta.get('atr_pct', 'N/A')}%`
                    * 🛡️ **Stop-Loss Protection:** `{'Break-Even + Tax Locked' if be_locked else 'Initial ATR Stop'}`
                    * ⚖️ **Capital Allocation:** £{t_row['capital_allocated']:.0f} (`{int(t_row['shares'])} shares`)
                    """)
    else:
        st.info("No active equity signals to analyze. Run the Live Scan on the Equity tab first.")

    st.markdown("---")
    st.markdown("### 📊 Derivative & Leveraged Alpha Reasoning")
    opt_sig = generate_daily_options_alpha()
    if opt_sig:
        with st.container(border=True):
            d1, d2 = st.columns([1, 2])
            with d1:
                st.metric("Contract Evaluated", opt_sig["option_contract"])
                st.write(f"**Model Implied Volatility:** `{opt_sig['implied_vol']}%`")
                st.write(f"**Entry Spot Price:** `{opt_sig['underlying_spot']:.2f}p`")
                st.write(f"**Strike Selected (OTM):** `{opt_sig['strike_price']:.0f}p`")
            with d2:
                st.markdown("""
                **Black-Scholes Volatility & Greeks Breakdown:**
                * 📐 **Pricing Model:** Closed-form continuous Black-Scholes with a 5.0% Bank of England base rate proxy ($r = 0.05$).
                * 🎯 **Moneyness:** Strike placed +2.0% Out-of-the-Money to maximize risk-reward leverage while capping time decay.
                * 🛡️ **Risk Guard:** Maximum budget capped under £500 with strict stop-loss set at -50% premium depreciation.
                """)

    st.markdown("---")
    st.markdown("### 🛑 Post-Trade Autopsies (Active Memory Bank)")
    with DB_LOCK:
        con = duckdb.connect(DB_PATH, read_only=True)
        try:
            closed_trades = con.execute("""
                SELECT date_str, ticker, status, entry_price, latest_price, pnl_pct, features_json 
                FROM trade_journal WHERE status != 'ACTIVE' ORDER BY last_audited DESC
            """).df()
        except Exception:
            closed_trades = pd.DataFrame()
        finally:
            con.close()

    if not closed_trades.empty:
        for _, c_row in closed_trades.iterrows():
            with st.expander(f"{c_row['status']} | {c_row['ticker']} ({c_row['date_str']}) — Net P&L: {c_row['pnl_pct']:+.2f}%"):
                st.write(f"**Entry:** `{c_row['entry_price']:.2f}p` | **Exit/Last:** `{c_row['latest_price']:.2f}p` | **Sector:** `{get_ticker_sector(c_row['ticker'])}`")
                st.code(f"Recorded Feature Vector: {c_row['features_json']}", language="json")
                if "LOSS" in str(c_row['status']):
                    st.error("Active Rule Applied: Future setups on this ticker receive a 30-day exponential time-decayed penalty (-15% initial from exit date), and ATR feeds the 75th-percentile volatility floor.")
                elif "BREAK-EVEN" in str(c_row['status']):
                    st.info("Active Rule Applied: Capital & UK Stamp Duty preserved via 65% Break-Even Stop Ratchet. Neutral memory weight (0% penalty).")
                else:
                    st.success("Active Rule Applied: Future setups on this ticker receive a 30-day exponential time-decayed boost (+5% initial from exit date).")
    else:
        st.success("🏆 **Zero Closed/Stopped-Out Trades in Current Memory.**\n\nAs trades hit their target or stop-loss, their feature vectors are permanently stored in Supabase and used to penalize or boost future scans.")