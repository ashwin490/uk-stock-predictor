import os
import re
import sys
import math
import json
import time
import hmac
import html
import hashlib
import calendar
import threading
import warnings
import io
import urllib.parse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from intraday_exit import scan_intraday_minutes
from nse_option_chain import (expiry_labels, supported_expiries, matching_rows, nearby_strikes, option_payload_shape, parse_expiry_date,
    exact_contract, bid_ask, quote_is_fresh, underlying_value, exchange_timestamp)
from reconciliation_policy import preserve_local_row, closed_record_conflict
from nse_ai_dashboard import worker_status, valid_forecasts
from nse_ai_private_reader import private_research_snapshot, PrivateResearchReadError
from datetime import datetime, time as dtime, timedelta
import random
import shutil
import requests

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pytz
import duckdb
from quant_guardrails import (CAPITAL_LIMIT_INR, EQUITY_TARGET_INR, OPTION_MAX_SINGLE_INR,
    OPTION_TOTAL_LIMIT_INR, allowable_equity_shares, can_allocate_option,
    review_long_option, remaining_capital)

try:
    from streamlit_autorefresh import st_autorefresh
except ImportError:
    st_autorefresh = None

warnings.filterwarnings("ignore")

# set_page_config must be the first Streamlit command that renders anything
st.set_page_config(
    page_title="NSE AI • Dark Research Terminal",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="expanded"
)

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# ==============================================================================
# 0. SETTINGS — everything you might ever want to change lives in this block
# ==============================================================================
DB_PATH = os.path.join(ROOT_DIR, "market_data.duckdb")
MODEL_PATH = os.path.join(ROOT_DIR, "models", "lgbm_stock_ranker.pkl")          # your original model: NEVER overwritten
LIVE_MODEL_PATH = os.path.join(ROOT_DIR, "models", "lgbm_stock_ranker_live.pkl")  # upgraded model (only if it tests better)

# ---- Portfolio & risk rules ----
POSITION_SIZE_INR = EQUITY_TARGET_INR
MAX_DAILY_EQUITY_TRADES = 10
MAX_HOLD_CALENDAR_DAYS = 7
BREAKEVEN_TRIGGER_RATIO = 0.65
MAX_ACTIVE_PER_SECTOR = 2
MAX_ACTIVE_OTHER_SECTOR = 4            # shares we couldn't classify share one bigger bucket
MAX_OPEN_POSITIONS = 10
MAX_OPEN_RISK_INR = 30000.0
COOLDOWN_CALENDAR_DAYS = 3
HALF_LIFE_DAYS = 30.0
# Latest user's historical baseline: 106 total share rows and 21 option rows.
# These are floor checks, NOT reconstructed data. Explicitly set the matching
# Streamlit Secrets setting to 0 only after knowingly accepting a fresh start.
HISTORY_EXPECTED_MIN_SHARES = 106
HISTORY_EXPECTED_MIN_OPTIONS = 21
# ---- Whole-market scan (every NSE share in the EQ series) ----
SCAN_WORKERS = 12                      # parallel price downloads
SCAN_TIME_BUDGET_SEC = 240             # stop downloading after this long; the rest is checked next time
SWEEP_TTL_SEC = 900                    # re-download a tradable share's prices at most every 15 min
QUIET_TTL_SEC = 21600                  # thinly traded shares are re-checked every 6 hours
FRESH_PRICE_TTL_SEC = 60               # finalists get a fresh price (≤1 min old) before any buy
FRESH_PRICE_MAX_AGE_SEC = 180          # ...and are skipped if that fresh download failed (never buy on an old price)
MAX_DEEP_CHECKS = 40                   # finalists that get the full news / memory / market check
DEEP_CHECK_MARGIN = 12.0               # finalists must be within this many points of the confidence bar
MAX_ATR_PCT = 7.0                      # skip shares that swing more than this % on a normal day
MIN_LISTED_FOR_VALID_LIST = 500        # a downloaded share list smaller than this is treated as broken
PULSE_REFRESH_SEC = 1800               # after today's picks are used up, still refresh the market pulse every 30 min
STRESS_NIFTY_5D = -3.0                 # pause buying if Nifty falls this much in 5 days ...
STRESS_VIX = 25.0                      # ... or India VIX is at/above this

# ---- Entry realism & quality filters ----
ENTRY_WINDOW_START = dtime(9, 30)
ENTRY_WINDOW_END = dtime(15, 0)
MIN_CALIBRATED_CONFIDENCE = 60.0       # (kept for older saved trades; picks are now chosen on expected value — see MIN_EV_PCT)
MIN_EV_PCT = 0.2                       # a pick must be expected to make at least this much after ALL costs (in % per trade)
LEARNING_PICKS_PER_DAY = 0  # do not paper-buy rejected ideas just to train             # while still learning, at most this many clearly-marked 🧪 learning picks a day
CAL_PRIOR_TRADES = 20                  # the confidence correction starts near 50% and moves to your real win rate as results arrive
MIN_RVOL_THRESHOLD = 0.45
MIN_NET_RETURN_PCT = 1.10
MIN_PRICE_INR = 0.01  # eligible at any price with liquidity safeguards
MIN_AVG_TURNOVER_INR = 20_000_000      # ₹2 crore traded per day
BE_TOLERANCE_PCT = 0.5

# ---- Costs (approximate NSE delivery-trade charges; edit if rates change) ----
STT_PCT_PER_SIDE = 0.100
STAMP_DUTY_PCT_BUY = 0.015
EXCHANGE_TXN_PCT_PER_SIDE = 0.00297
SEBI_PCT_PER_SIDE = 0.0001
GST_PCT = 18.0
DEFAULT_BROKERAGE_PER_ORDER_INR = 0.0
DEFAULT_DP_CHARGE_INR = 15.9           # depository charge per sale (varies by broker)
DEFAULT_FRICTION_PCT = 0.28            # used for old trades that didn't store their own

# ---- Options (practice) ----
OPTION_BUDGET_INR = OPTION_MAX_SINGLE_INR
RISK_FREE_RATE = 0.06
OPTION_MODEL_HALF_SPREAD = 0.03
OPTION_MAX_SPREAD_PCT = 15.0
OPTION_MIN_OPEN_INTEREST = 50
OPTION_TARGET_MULT = 1.65
OPTION_STOP_MULT = 0.50
# NSE revises lot sizes now and then — VERIFY these against NSE's current contract list.
FNO_STOCKS = ["RELIANCE.NS", "HDFCBANK.NS", "ICICIBANK.NS", "INFY.NS", "TCS.NS", "SBIN.NS", "BHARTIARTL.NS", "ITC.NS", "LT.NS"]
LOT_SIZES = {"RELIANCE": 250, "HDFCBANK": 400, "ICICIBANK": 700, "INFY": 400, "TCS": 175, "SBIN": 750, "BHARTIARTL": 950, "ITC": 1600, "LT": 300}
# ---- Options learner (separate from the share AI; it only ever changes the options idea) ----
OPT_LEARN_MIN_TRADES = 10              # finished option ideas (in the window below) before it becomes pickier
OPT_LEARN_WINDOW_DAYS = 60             # only the last 60 days count, so a strict rule lifts by itself and gets re-tested
OPT_REST_THRESHOLD = 0.75              # a share 'rests' from options after a recent loss (one loss ≈ 12 days)
OPT_LIVE_ONLY_MIN_MODEL = 8            # estimated-price ideas needed before it may decide to use real NSE quotes only
OPT_MAX_TREND_BAR = 82.0               # the pickiest it can get (trend strength needed)
STRIKE_STEPS = {"RELIANCE": 20, "HDFCBANK": 10, "ICICIBANK": 10, "INFY": 20, "TCS": 50, "SBIN": 10, "BHARTIARTL": 20, "ITC": 10, "LT": 50}

# ---- AI governance ----
MIN_SETTLED_TO_RETRAIN = 100
RETRAIN_STEP_INTERVAL = 25
MIN_CALIBRATION_SAMPLES = 80
CONF_SHRINK = 0.5

USER_AGENTS = [
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
]

yf_session = requests.Session()
yf_session.headers.update({
    "User-Agent": USER_AGENTS[0],
    "Accept": "application/json,text/plain,*/*",
    "Accept-Language": "en-US,en;q=0.9",
    "Origin": "https://finance.yahoo.com",
    "Referer": "https://finance.yahoo.com/",
    "Connection": "keep-alive"
})
if hasattr(yf_session, "mount"):
    try:
        from requests.adapters import HTTPAdapter
        yf_session.mount("https://", HTTPAdapter(pool_connections=4, pool_maxsize=SCAN_WORKERS + 8, max_retries=0))
    except Exception:
        pass

@st.cache_resource
def get_db_lock():
    return threading.Lock()

DB_LOCK = get_db_lock()

@st.cache_resource
def get_hist_store():
    return {"lock": threading.Lock(), "data": {}}

@st.cache_resource
def get_nse_http_box():
    return {"session": None, "ts": 0.0, "lock": threading.Lock()}

# ---- Emergency backup list: used ONLY if NSE's full share list can't be downloaded and no saved copy exists ----
FALLBACK_SHARES = [
    "SBIN.NS", "ITC.NS", "TATAPOWER.NS", "BEL.NS", "NTPC.NS",
    "POWERGRID.NS", "ONGC.NS", "COALINDIA.NS", "PFC.NS", "RECLTD.NS",
    "BHEL.NS", "TATASTEEL.NS", "VEDL.NS", "HINDALCO.NS", "WIPRO.NS",
    "TATAMOTORS.NS", "BANKBARODA.NS", "PNB.NS", "CANBK.NS", "UNIONBANK.NS",
    "IDFCFIRSTB.NS", "FEDERALBNK.NS", "ASHOKLEY.NS", "MOTHERSON.NS", "EXIDEIND.NS",
    "GAIL.NS", "IOC.NS", "BPCL.NS", "PETRONET.NS", "IGL.NS",
    "BIOCON.NS", "LAURUSLABS.NS", "GRANULES.NS", "NATIONALUM.NS", "NMDC.NS",
    "SAIL.NS", "JINDALSTEL.NS", "IRCTC.NS", "CONCOR.NS", "IRFC.NS",
    "RVNL.NS", "NHPC.NS", "SJVN.NS", "CESC.NS", "DABUR.NS",
    "MARICO.NS", "GODREJCP.NS", "BERGEPAINT.NS", "ZEEL.NS", "MANAPPURAM.NS"
]

NSE_SECTOR_MAP = {
    "RELIANCE": "Energy & Conglomerate", "ONGC": "Energy & Conglomerate", "GAIL": "Energy & Conglomerate",
    "IOC": "Energy & Conglomerate", "BPCL": "Energy & Conglomerate", "PETRONET": "Energy & Conglomerate",
    "IGL": "Energy & Conglomerate", "NTPC": "Utilities & Power", "POWERGRID": "Utilities & Power",
    "TATAPOWER": "Utilities & Power", "NHPC": "Utilities & Power", "SJVN": "Utilities & Power",
    "CESC": "Utilities & Power", "COALINDIA": "Basic Materials", "HDFCBANK": "Financials",
    "ICICIBANK": "Financials", "SBIN": "Financials", "KOTAKBANK": "Financials", "AXISBANK": "Financials",
    "BAJFINANCE": "Financials", "BAJAJFINSV": "Financials", "PFC": "Financials", "RECLTD": "Financials",
    "BANKBARODA": "Financials", "PNB": "Financials", "CANBK": "Financials", "UNIONBANK": "Financials",
    "IDFCFIRSTB": "Financials", "FEDERALBNK": "Financials", "IRFC": "Financials", "MANAPPURAM": "Financials",
    "INFY": "IT & Technology", "TCS": "IT & Technology", "HCLTECH": "IT & Technology", "WIPRO": "IT & Technology",
    "TECHM": "IT & Technology", "BHARTIARTL": "Telecom", "ZEEL": "Telecom", "ITC": "Consumer & FMCG",
    "HINDUNILVR": "Consumer & FMCG", "TATACONSUM": "Consumer & FMCG", "NESTLEIND": "Consumer & FMCG",
    "DABUR": "Consumer & FMCG", "MARICO": "Consumer & FMCG", "GODREJCP": "Consumer & FMCG",
    "BERGEPAINT": "Consumer & FMCG", "IRCTC": "Consumer & FMCG", "LT": "Industrials & Infra",
    "BEL": "Defense & Aero", "HAL": "Defense & Aero", "BHEL": "Industrials & Infra",
    "CONCOR": "Industrials & Infra", "RVNL": "Industrials & Infra", "TATAMOTORS": "Auto",
    "M&M": "Auto", "MARUTI": "Auto", "BAJAJ-AUTO": "Auto", "EICHERMOT": "Auto", "ASHOKLEY": "Auto",
    "MOTHERSON": "Auto", "EXIDEIND": "Auto", "SUNPHARMA": "Healthcare", "DRREDDY": "Healthcare",
    "CIPLA": "Healthcare", "BIOCON": "Healthcare", "LAURUSLABS": "Healthcare", "GRANULES": "Healthcare",
    "TATASTEEL": "Metals & Mining", "JSWSTEEL": "Metals & Mining", "HINDALCO": "Metals & Mining",
    "VEDL": "Metals & Mining", "NATIONALUM": "Metals & Mining", "NMDC": "Metals & Mining",
    "SAIL": "Metals & Mining", "JINDALSTEL": "Metals & Mining"
}

INR_IT_PHARMA_EXPORTERS = {"INFY", "TCS", "HCLTECH", "WIPRO", "TECHM", "SUNPHARMA", "DRREDDY", "CIPLA", "BIOCON", "LAURUSLABS", "GRANULES"}
CRUDE_SENSITIVE_USERS = {"ASIANPAINT", "BERGEPAINT", "INDIGO", "BPCL", "HPCL", "IOC"}

NSE_BULLISH_LEXICON = {
    "order win": 6.0, "letter of award": 7.0, "loa": 5.0, "l1 bidder": 7.0,
    "new contract": 5.0, "share buyback": 6.0, "bonus issue": 5.0, "stock split": 4.0,
    "usfda approval": 8.0, "tentative approval": 5.0, "eir": 6.0, "pli scheme": 5.0,
    "promoter buying": 6.0, "rating upgrade": 5.0, "record revenue": 6.0,
    "joint venture": 4.0, "capacity expansion": 4.0, "debt reduction": 5.0
}

NSE_BEARISH_LEXICON = {
    "sebi notice": -25.0, "show cause": -25.0, "sebi order": -25.0,
    "ed raid": -30.0, "enforcement directorate": -30.0, "income tax search": -25.0,
    "form 483": -22.0, "import alert": -28.0, "oai": -25.0,
    "promoter pledge": -18.0, "invocation of pledge": -28.0,
    "qip floor": -15.0, "offer for sale": -15.0, "ofs": -15.0, "dilution": -20.0,
    "auditor resignation": -30.0, "default": -30.0, "nclt": -30.0, "insolvency": -35.0,
    "rating downgrade": -20.0, "fraud": -35.0, "penalty": -15.0, "plant shutdown": -20.0
}
# Whole-word matching (so "loa" no longer fires inside "loan", or "eir" inside "their")
_BULL_PATTERNS = [(re.compile(r"\b" + re.escape(p) + r"\b"), p, w) for p, w in NSE_BULLISH_LEXICON.items()]
_BEAR_PATTERNS = [(re.compile(r"\b" + re.escape(p) + r"\b"), p, w) for p, w in NSE_BEARISH_LEXICON.items()]

# ==============================================================================
# 0b. SMALL HELPERS
# ==============================================================================
def ist_now() -> datetime:
    return datetime.now(pytz.timezone('Asia/Kolkata'))

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

def clean_sym_name(sym: str) -> str:
    raw = str(sym).strip().lstrip("$")
    base_token = raw.split()[0] if raw else ""
    return base_token.replace(".NS", "").replace(".BO", "")

def normalize_ts_str(val, fallback_str: str) -> str:
    if val is None or pd.isna(val):
        return fallback_str
    s = str(val).strip().replace("T", " ").replace(" IST", "")
    if "+" in s:
        s = s.split("+")[0].strip()
    if "." in s:
        s = s.split(".")[0].strip()
    return s[:19] if len(s) >= 10 else fallback_str

def get_ticker_sector(ticker: str) -> str:
    t = clean_sym_name(ticker).upper()
    if t in NSE_SECTOR_MAP:
        return NSE_SECTOR_MAP[t]
    return get_nse_industry_map().get(t, "Other")

def sector_cap(sector: str) -> int:
    return MAX_ACTIVE_OTHER_SECTOR if sector == "Other" else MAX_ACTIVE_PER_SECTOR

def get_trade_sector(ticker: str, f_json=None) -> str:
    t = clean_sym_name(ticker).upper()
    if t in NSE_SECTOR_MAP:
        return NSE_SECTOR_MAP[t]
    f = _parse_features(f_json)
    stored = str(f.get("sector") or "")
    if stored and stored != "Other":
        return stored
    return get_nse_industry_map().get(t, "Other")

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

def is_nse_market_open() -> bool:
    now_ist = ist_now()
    return get_market_state()["trading_day"] and dtime(9, 15) <= now_ist.time() <= dtime(15, 30)

def is_entry_window_open() -> bool:
    now_ist = ist_now()
    return get_market_state()["trading_day"] and ENTRY_WINDOW_START <= now_ist.time() <= ENTRY_WINDOW_END

def normalize_ticker_for_yf(ticker_str: str) -> str:
    clean = str(ticker_str).split()[0].strip()
    if clean.endswith(".NS") or clean.endswith(".BO") or clean.startswith("^") or "=" in clean:
        return clean
    return f"{clean}.NS"

# ---------- costs ----------
def get_brokerage_inr() -> float:
    try:
        return float(st.session_state.get("brokerage_inr", DEFAULT_BROKERAGE_PER_ORDER_INR))
    except Exception:
        return DEFAULT_BROKERAGE_PER_ORDER_INR

def get_dp_charge_inr() -> float:
    try:
        return float(st.session_state.get("dp_charge_inr", DEFAULT_DP_CHARGE_INR))
    except Exception:
        return DEFAULT_DP_CHARGE_INR

def slippage_pct_per_side(avg_turnover_inr) -> float:
    try:
        v = float(avg_turnover_inr)
        if not math.isfinite(v):
            raise ValueError
        if v >= 500_000_000:
            return 0.03
        if v >= 100_000_000:
            return 0.05
        if v >= 20_000_000:
            return 0.10
        return 0.20
    except Exception:
        return 0.10

def compute_costs_pct(position_value: float, slip_per_side_pct: float, brokerage=None, dp=None) -> float:
    """Round-trip cost of a delivery trade (buy then sell) as a % of the position."""
    v = max(float(position_value), 1.0)
    b = get_brokerage_inr() if brokerage is None else float(brokerage)
    d = get_dp_charge_inr() if dp is None else float(dp)
    stt = STT_PCT_PER_SIDE / 100.0 * v * 2
    stamp = STAMP_DUTY_PCT_BUY / 100.0 * v
    exch = EXCHANGE_TXN_PCT_PER_SIDE / 100.0 * v * 2
    sebi = SEBI_PCT_PER_SIDE / 100.0 * v * 2
    gst = GST_PCT / 100.0 * (2 * b + exch + sebi)
    slip = slip_per_side_pct / 100.0 * v * 2
    total = stt + stamp + exch + sebi + gst + 2 * b + d + slip
    return round(total / v * 100.0, 3)

def get_trade_friction_pct(f_json) -> float:
    f = _parse_features(f_json)
    try:
        return float(f["friction_pct"]) if "friction_pct" in f else DEFAULT_FRICTION_PCT
    except Exception:
        return DEFAULT_FRICTION_PCT

def get_breakeven_exit_price(entry_price: float, friction_pct: float) -> float:
    return round(float(entry_price) * (1.0 + (float(friction_pct) / 100.0)), 2)

def calc_net_equity_pnl_pct(entry_price: float, current_price: float, friction_pct: float) -> float:
    if entry_price <= 0:
        return 0.0
    gross_pct = ((float(current_price) - float(entry_price)) / float(entry_price)) * 100.0
    return round(gross_pct - float(friction_pct), 2)

def is_stop_breakeven_protected(entry_price: float, stop_loss: float) -> bool:
    return float(stop_loss) >= float(entry_price) - 1e-4

def classify_stop_exit(entry: float, stop_at_hit: float, fill: float, friction_pct: float) -> str:
    net = calc_net_equity_pnl_pct(entry, fill, friction_pct)
    if stop_at_hit >= entry - 1e-4 and net >= -BE_TOLERANCE_PCT:
        return "🛡️ BREAK-EVEN (PROTECTED EXIT)"
    return "🛑 LOSS (STOPPED OUT)"

def evaluate_completed_bar(o, hi, lo, c, entry, target, stop, be_floor, be_trigger):
    """One COMPLETED daily bar: gap-aware, stop-first on ambiguity, break-even ratchet only affects the NEXT bar."""
    if o >= target:
        return "WIN", o, stop
    if o <= stop:
        return "STOP", o, stop
    if lo <= stop:
        return "STOP", stop, stop
    if hi >= target:
        return "WIN", target, stop
    new_stop = stop
    if target > entry and stop < be_floor and hi >= be_trigger and c > be_floor:
        new_stop = be_floor
    return None, c, new_stop

# ==============================================================================
# 0c. THE WHOLE NSE MARKET: SHARE LIST + INDUSTRY LABELS (downloaded, saved, with fallbacks)
# ==============================================================================
NSE_EQUITY_LIST_URLS = [
    "https://nsearchives.nseindia.com/content/equities/EQUITY_L.csv",
    "https://archives.nseindia.com/content/equities/EQUITY_L.csv",
]
NSE_INDUSTRY_LIST_URLS = [
    "https://nsearchives.nseindia.com/content/indices/ind_niftytotalmarket_list.csv",
    "https://www.niftyindices.com/IndexConstituent/ind_niftytotalmarket_list.csv",
    "https://nsearchives.nseindia.com/content/indices/ind_nifty500list.csv",
    "https://www.niftyindices.com/IndexConstituent/ind_nifty500list.csv",
]
UNIVERSE_SAVE_PATH = os.path.join(ROOT_DIR, "nse_all_shares.json")
INDUSTRY_SAVE_PATH = os.path.join(ROOT_DIR, "nse_industries.json")
_SYM_OK = re.compile(r"^[A-Z0-9][A-Z0-9&\-]{0,19}$")
# NSE industry names -> the app's sector names (anything unmatched keeps NSE's own name)
_INDUSTRY_TO_SECTOR = [
    ("financial", "Financials"), ("information technology", "IT & Technology"), ("oil gas", "Energy & Conglomerate"),
    ("fast moving consumer", "Consumer & FMCG"), ("automobile", "Auto"), ("capital goods", "Industrials & Infra"),
    ("construction materials", "Basic Materials"), ("construction", "Industrials & Infra"), ("healthcare", "Healthcare"),
    ("metals", "Metals & Mining"), ("power", "Utilities & Power"), ("utilities", "Utilities & Power"),
    ("telecom", "Telecom"), ("chemical", "Chemicals"), ("consumer durables", "Consumer Durables"),
    ("consumer services", "Consumer Services"), ("realty", "Realty"), ("media", "Media"), ("textile", "Textiles"),
    ("forest", "Basic Materials"), ("diversified", "Energy & Conglomerate"), ("services", "Services"),
]

def _industry_to_sector(industry: str) -> str:
    low = str(industry).strip().lower()
    if not low or low in ("nan", "none"):
        return "Other"
    for key, label in _INDUSTRY_TO_SECTOR:
        if key in low:
            return label
    return str(industry).strip().title()[:30]

def _http_text(url: str, timeout: int = 10) -> str:
    s = requests.Session()
    s.headers.update({"User-Agent": USER_AGENTS[0], "Accept": "text/csv,text/plain,*/*", "Accept-Language": "en-US,en;q=0.9",
                      "Referer": "https://www.nseindia.com/"})
    try:
        r = s.get(url, timeout=timeout)
        if r.status_code == 200:
            return str(getattr(r, "text", "") or "")
    except Exception:
        pass
    return ""

def _read_csv_text(text: str) -> pd.DataFrame:
    if not text or "," not in text:
        return pd.DataFrame()
    try:
        df = pd.read_csv(io.StringIO(text), dtype=str)
    except Exception:
        return pd.DataFrame()
    df.columns = [str(c).strip().upper() for c in df.columns]
    return df

def parse_equity_list(text: str) -> list:
    """NSE's EQUITY_L.csv -> sorted list of symbols in the normal 'EQ' series (excludes trade-to-trade/surveillance series)."""
    df = _read_csv_text(text)
    if df.empty or "SYMBOL" not in df.columns:
        return []
    if "SERIES" in df.columns:
        df = df[df["SERIES"].astype(str).str.strip().str.upper() == "EQ"]
    syms = {str(x).strip().upper() for x in df["SYMBOL"].tolist()}
    return sorted(x for x in syms if _SYM_OK.match(x))

def _save_json(path: str, obj):
    try:
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(obj, fh)
        os.replace(tmp, path)
    except Exception:
        pass

def _load_json(path: str):
    if not os.path.exists(path):
        return None
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return None

def get_all_nse_symbols() -> tuple:
    """(symbols, where_from). Today's NSE list if reachable; otherwise the last saved copy; otherwise a small backup list.
    Cached for 12 hours (2 hours if NSE didn't answer)."""
    box = get_nse_http_box()
    now = time.time()
    cached = box.get("universe")
    if cached and now - cached[0] < (43200 if cached[3] else 7200):
        return cached[1], cached[2]
    syms, source, fresh = [], "", False
    for url in NSE_EQUITY_LIST_URLS:
        syms = parse_equity_list(_http_text(url, timeout=8))
        if len(syms) >= MIN_LISTED_FOR_VALID_LIST:
            source, fresh = "NSE's full share list (today)", True
            _save_json(UNIVERSE_SAVE_PATH, {"saved": ist_now().strftime("%Y-%m-%d %H:%M"), "symbols": syms})
            break
        syms = []
    if not syms:
        saved = _load_json(UNIVERSE_SAVE_PATH)
        if isinstance(saved, dict) and isinstance(saved.get("symbols"), list) and len(saved["symbols"]) >= MIN_LISTED_FOR_VALID_LIST:
            syms = sorted({str(x).strip().upper() for x in saved["symbols"] if _SYM_OK.match(str(x).strip().upper())})
            source = f"saved NSE share list from {saved.get('saved', 'earlier')} (NSE not reachable right now)"
    if not syms:
        syms = sorted({clean_sym_name(x).upper() for x in (FALLBACK_SHARES + DEFAULT_NIFTY_BASKET)})
        source = "small backup list — NSE's full list couldn't be downloaded"
    box["universe"] = (now, syms, source, fresh)
    return syms, source

def get_nse_industry_map() -> dict:
    """{'SYMBOL': 'Sector'} from NSE's index constituent lists (covers the ~750 most traded shares). Saved to disk; cached."""
    box = get_nse_http_box()
    now = time.time()
    cached = box.get("industries")
    if cached and now - cached[0] < (43200 if cached[2] else 7200):
        return cached[1]
    box["industries"] = (now, (cached[1] if cached else {}), False)      # stops re-entry while downloading
    out = {}
    for url in NSE_INDUSTRY_LIST_URLS:
        df = _read_csv_text(_http_text(url, timeout=6))
        if not df.empty and "SYMBOL" in df.columns and "INDUSTRY" in df.columns:
            for sym, ind in zip(df["SYMBOL"].tolist(), df["INDUSTRY"].tolist()):
                k = str(sym).strip().upper()
                if _SYM_OK.match(k) and k not in out:
                    out[k] = _industry_to_sector(ind)
        if len(out) >= 700:
            break
    fresh = len(out) >= 300
    if fresh:
        _save_json(INDUSTRY_SAVE_PATH, out)
    else:
        saved = _load_json(INDUSTRY_SAVE_PATH)
        out = saved if isinstance(saved, dict) and saved else out
    box["industries"] = (now, out, fresh)
    return out

# ==============================================================================
# 1. DATABASE (THREAD-SAFE DUCKDB)
# ==============================================================================
from schema_migrations import ensure_schema, SCHEMA_VERSION

def init_duckdb_storage():
    """Safely migrate in-place; never silently ignore a failed schema upgrade."""
    with DB_LOCK:
        return ensure_schema(DB_PATH, ROOT_DIR / "backups")

try:
    DB_SCHEMA_STATUS = init_duckdb_storage()
except Exception:
    import logging
    logging.exception("DuckDB schema upgrade failed")
    st.error("The saved trading database could not be upgraded safely. No trading history was intentionally deleted. Check Streamlit Cloud logs and restore the backed-up database if necessary.")
    st.stop()

def journal_counts() -> dict:
    """Plain counts straight from the journal (used for the health check)."""
    with DB_LOCK:
        con = duckdb.connect(DB_PATH, read_only=True)
        try:
            df = con.execute("SELECT status FROM trade_journal WHERE ticker NOT LIKE '%.L'").df()
            n_open = int((df["status"].apply(classify_outcome) == "ACTIVE").sum()) if not df.empty else 0
            return {"total": len(df), "open": n_open, "closed": len(df) - n_open}
        except Exception:
            return {"total": 0, "open": 0, "closed": 0}
        finally:
            con.close()

def get_history_integrity() -> dict:
    """Block new paper allocations when *either* previous journal looks incomplete.

    This is a count floor based on the user's known displayed history, not proof
    of data correctness. It never deletes, reconstructs, or rewrites old rows.
    """
    try:
        local_shares = journal_counts()["total"]
    except Exception:
        local_shares = 0
    try:
        with DB_LOCK:
            con = duckdb.connect(DB_PATH, read_only=True)
            try:
                local_options = int(con.execute("SELECT COUNT(*) FROM daily_options_journal").fetchone()[0])
            finally:
                con.close()
    except Exception:
        local_options = 0
    cloud_info = st.session_state.get("hydrate_info") or {}
    shares = history_state(
        local_shares, HISTORY_EXPECTED_MIN_SHARES,
        cloud_configured=bool(supabase),
        cloud_shares=cloud_info.get("remote") if supabase and cloud_info else None,
        cloud_failed=bool(cloud_info.get("fetch_failed", False)),
    )
    options = history_state(
        local_options, HISTORY_EXPECTED_MIN_OPTIONS,
        cloud_configured=bool(supabase),
        cloud_shares=cloud_info.get("remote_options") if supabase and cloud_info else None,
        cloud_failed=bool(cloud_info.get("fetch_failed", False)),
        record_label="option",
    )
    return {
        **shares,
        "verified_count": shares["verified_count"] and options["verified_count"],
        "allow_new_paper_entries": shares["allow_new_paper_entries"] and options["allow_new_paper_entries"],
        "reason": " ".join(x["reason"] for x in (shares, options) if not x["verified_count"]),
        "local_options": options["local"],
        "minimum_options": options["minimum"],
        "missing_options": options["missing"],
    }

def get_meta(key: str, default=None):
    with DB_LOCK:
        con = duckdb.connect(DB_PATH, read_only=True)
        try:
            df = con.execute("SELECT meta_val FROM system_metadata WHERE meta_key = ?", [key]).df()
            return df["meta_val"].iloc[0] if not df.empty else default
        except Exception:
            return default
        finally:
            con.close()

def set_meta(key: str, val: str):
    with DB_LOCK:
        con = duckdb.connect(DB_PATH, read_only=False)
        try:
            con.execute("INSERT OR REPLACE INTO system_metadata (meta_key, meta_val) VALUES (?, ?)", [key, str(val)])
        except Exception:
            pass
        finally:
            con.close()

# ==============================================================================
# 2. SAFE IMPORTS OF YOUR OTHER MODULES (WITH FALLBACKS)
# ==============================================================================
DEFAULT_NIFTY_BASKET = [
    "RELIANCE.NS", "HDFCBANK.NS", "ICICIBANK.NS", "INFY.NS", "TCS.NS",
    "SBIN.NS", "BHARTIARTL.NS", "ITC.NS", "LT.NS", "AXISBANK.NS",
    "KOTAKBANK.NS", "BAJFINANCE.NS", "MARUTI.NS", "SUNPHARMA.NS", "NTPC.NS",
    "TATAMOTORS.NS", "POWERGRID.NS", "ONGC.NS", "COALINDIA.NS", "BEL.NS",
    "TATAPOWER.NS", "PFC.NS", "RECLTD.NS", "BHEL.NS", "TATASTEEL.NS",
    "JSWSTEEL.NS", "HINDALCO.NS", "VEDL.NS", "WIPRO.NS", "HCLTECH.NS"
]

ANNOUNCEMENTS_AVAILABLE = True
try:
    from core.announcements import check_corporate_announcements
except Exception:
    ANNOUNCEMENTS_AVAILABLE = False
    def check_corporate_announcements(sym: str):
        return {"risk_penalty": 1.0, "headline": "Standard NSE Exchange Flow"}

try:
    from supabase import create_client, Client
except ImportError:
    create_client, Client = None, None

FEATURE_COLS = [
    "dist_ema20_pct", "trend_spread_pct", "atr_pct", "rvol", "rsi_14", "deliv_shock"
]

# ==============================================================================
# 3. HARDENED ZERO-TRUST AUTHENTICATION  (kept exactly as you had it)
# ==============================================================================
def _secret(name: str, default="") -> str:
    try:
        return str(st.secrets.get(name, os.environ.get(name, default)))
    except Exception:
        return str(os.environ.get(name, default))

from history_integrity import history_state

# Streamlit Cloud can override the baseline without committing any credentials.
try:
    HISTORY_EXPECTED_MIN_SHARES = max(0, int(_secret("HISTORY_EXPECTED_MIN_SHARES", str(HISTORY_EXPECTED_MIN_SHARES))))
except ValueError:
    HISTORY_EXPECTED_MIN_SHARES = 106
try:
    HISTORY_EXPECTED_MIN_OPTIONS = max(0, int(_secret("HISTORY_EXPECTED_MIN_OPTIONS", str(HISTORY_EXPECTED_MIN_OPTIONS))))
except ValueError:
    HISTORY_EXPECTED_MIN_OPTIONS = 21

def check_password() -> bool:
    if st.session_state.get("password_correct"):
        return True
    expected_user = _secret("AUTH_USER")
    expected_hash = _secret("AUTH_PASS_HASH")
    if not expected_user or not expected_hash:
        st.error("Please set AUTH_USER and AUTH_PASS_HASH in Streamlit secrets before starting.")
        return False
    if time.time() < st.session_state.get("auth_lockout_until", 0):
        st.error("Login temporarily locked after repeated attempts.")
        return False
    st.title("🔐 NSE Quant Research")
    with st.form("login"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submit = st.form_submit_button("Sign in")
    if submit:
        ok = hmac.compare_digest(username.strip(), expected_user.strip()) and hmac.compare_digest(
            hashlib.sha256(password.encode()).hexdigest(), expected_hash.strip().lower())
        if ok:
            st.session_state["password_correct"] = True
            st.session_state["failed_auth_attempts"] = 0
            st.rerun()
        n = st.session_state.get("failed_auth_attempts", 0) + 1
        st.session_state["failed_auth_attempts"] = n
        if n >= 5:
            st.session_state["auth_lockout_until"] = time.time() + 900
        st.error("Incorrect credentials")
    return False

if not check_password():
    st.stop()

# ==============================================================================
# 4. CLOUD BACKUP (SUPABASE) — NEVER LETS OLD CLOUD DATA OVERWRITE NEWER LOCAL DATA
# ==============================================================================
@st.cache_resource
def get_supabase_client():
    if create_client is None:
        return None
    url, key = _secret("SUPABASE_URL"), _secret("SUPABASE_KEY")
    if not url or not key:
        return None
    try:
        clean_url = url.strip().split("/rest/v1")[0].rstrip("/")
        return create_client(clean_url, key.strip())
    except Exception:
        st.session_state["db_error"] = "Supabase connection failed (Check secrets)."
        return None

supabase = get_supabase_client()

def record_db_error(context: str, err: Exception):
    raw_msg = str(err)
    sanitized = re.sub(r"https?://[^\s'\"]+", "[REDACTED_URL]", raw_msg)
    st.session_state["db_error"] = f"[{context}] {sanitized[:140]}"

def cloud_update(table: str, match: dict, payload: dict, label: str) -> bool:
    """Update the cloud copy; if it fails, queue it so it is retried (and hydrate can't undo it)."""
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
    if not supabase:
        return []
    all_rows = []
    st.session_state.setdefault("supabase_fetch_errors", {})
    st.session_state["supabase_fetch_errors"].pop(table_name, None)
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
            st.session_state.setdefault("supabase_fetch_errors", {})[table_name] = type(e).__name__
            break
    return all_rows

def _num(x, default=0.0):
    if x is None or (isinstance(x, str) and not x.strip()):
        return default
    try:
        v = float(x)
        return v if math.isfinite(v) else default
    except Exception:
        return default

def _int(x, default=0):
    if x is None or (isinstance(x, str) and not x.strip()):
        return default
    try:
        return int(float(x))
    except Exception:
        return default

def _norm_status(raw) -> str:
    if raw is None or str(raw).strip() == "":
        return "ACTIVE"
    return "ACTIVE" if classify_outcome(raw) == "ACTIVE" else str(raw).strip().upper()

def hydrate_duckdb_from_supabase():
    """Restore trades from the online backup. Each row is handled on its own, so ONE unreadable old row can never
    stop the newer rows (usually your open trades) from loading. Newer local data is never overwritten."""
    info = {"remote": 0, "remote_options": 0, "remote_active": 0, "restored": 0, "kept_local": 0, "closed_conflicts": 0, "bad_rows": 0, "first_error": ""}
    st.session_state["hydrate_info"] = info
    if not supabase:
        return
    ist_now_str = ist_now().strftime('%Y-%m-%d %H:%M:%S')
    eq_data = fetch_all_supabase_rows("predictions")
    opt_data = fetch_all_supabase_rows("options_journal")
    info["fetch_failed"] = bool(st.session_state.get("supabase_fetch_errors", {}).get("predictions"))
    info["remote"] = len(eq_data or [])
    info["remote_options"] = len(opt_data or [])
    info["remote_active"] = sum(1 for r in (eq_data or []) if _norm_status(r.get("status")) == "ACTIVE" and clean_sym_name(r.get("ticker", "")) and not str(r.get("ticker", "")).endswith(".L"))

    def bad(e):
        info["bad_rows"] += 1
        if not info["first_error"]:
            info["first_error"] = f"{type(e).__name__}: {str(e)[:80]}"

    with DB_LOCK:
        con = duckdb.connect(DB_PATH, read_only=False)
        try:
            local_eq, local_opt = {}, {}
            try:
                for r in con.execute("SELECT trade_id, status, last_audited, ticker, date_str, stop_loss, latest_price, pnl_pct FROM trade_journal").fetchall():
                    local_eq[r[0]] = r
                for r in con.execute("SELECT date_key, status, last_audited, current_option_price, pnl_pct FROM daily_options_journal").fetchall():
                    local_opt[r[0]] = r
            except Exception:
                pass

            for r in (eq_data or []):
                try:
                    ticker_val = clean_sym_name(r.get('ticker', ''))
                    if not ticker_val or ticker_val.endswith('.L'):
                        continue
                    pred_date = str(r.get('predicted_date') or ist_now_str[:10])[:10]
                    trade_id = f"{ticker_val}_{pred_date}"
                    remote_status = _norm_status(r.get('status'))
                    remote_ts = str(r.get('last_checked') or "")
                    loc = local_eq.get(trade_id)
                    if loc:
                        l_status, l_ts = str(loc[1]), str(loc[2] or "")
                        # Finalised local trades are immutable: cloud `last_checked`
                        # is NOT an exact execution timestamp. Never rewrite a
                        # closed trade or its exit time because the cloud has a
                        # newer audit/checkpoint time. Preserve local evidence.
                        closed_vs_active = (l_status != "ACTIVE" and remote_status == "ACTIVE")
                        if preserve_local_row(l_status, remote_status, l_ts, remote_ts):
                            info["kept_local"] += 1
                            if closed_vs_active:
                                cloud_update("predictions", {"ticker": loc[3], "predicted_date": loc[4]},
                                             {"status": l_status, "stop_loss": _num(loc[5]), "latest_price": _num(loc[6]),
                                              "pnl_pct": _num(loc[7]), "last_checked": l_ts}, "Re-push Closed Trade")
                            elif closed_record_conflict(l_status, remote_status, l_ts, remote_ts):
                                info["closed_conflicts"] += 1
                            continue
                    last_check = normalize_ts_str(r.get('last_checked'), f"{pred_date} 15:30:00")
                    # Supabase's historic `last_checked` reflects the last
                    # audit, not necessarily the time a target/stop was hit.
                    # Preserve that provenance for any imported closed trade.
                    imported_features = _parse_features(r.get('features_json'))
                    if remote_status != "ACTIVE" and not imported_features.get('exit_timestamp_source'):
                        imported_features['exit_timestamp_source'] = 'cloud_last_checked_estimate'
                    entry = _num(r.get('entry_price'))
                    tgt, stp = _num(r.get('target_price')), _num(r.get('stop_loss'))
                    if remote_status == "ACTIVE" and (entry <= 0 or tgt <= 0 or stp <= 0 or tgt <= stp):
                        raise ValueError(f"{ticker_val}: missing or impossible prices (bought {r.get('entry_price')}, goal {r.get('target_price')}, exit {r.get('stop_loss')})")
                    shares_n = max(1, _int(r.get('shares_qty'), 1))
                    cap = _num(r.get('position_gbp'))
                    if cap <= 0:
                        cap = round(shares_n * entry, 2) if entry > 0 else POSITION_SIZE_INR
                    latest = _num(r.get('latest_price'), entry)
                    exit_p = latest if remote_status != "ACTIVE" else 0.0
                    con.execute("""
                        INSERT OR REPLACE INTO trade_journal 
                        (trade_id, timestamp, date_str, ticker, asset_type, entry_price, target_price,
                         stop_loss, shares, capital_allocated, status, latest_price, pnl_pct, exit_price,
                         exit_timestamp, last_audited, features_json)
                        VALUES (?, ?, ?, ?, 'EQUITY', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, [trade_id, last_check, pred_date, ticker_val, entry, tgt, stp,
                          shares_n, cap, remote_status, latest, _num(r.get('pnl_pct')), exit_p,
                          (last_check if remote_status != "ACTIVE" else None), last_check, json.dumps(imported_features)])
                    info["restored"] += 1
                except Exception as e:
                    bad(e)

            for o in (opt_data or []):
                try:
                    date_k = str(o.get('date_key') or ist_now_str[:10])[:10]
                    remote_status = _norm_status(o.get('status'))
                    remote_ts = str(o.get('last_audited') or "")
                    loc = local_opt.get(date_k)
                    if loc:
                        l_status, l_ts = str(loc[1]), str(loc[2] or "")
                        if preserve_local_row(l_status, remote_status, l_ts, remote_ts):
                            if l_status != "ACTIVE" and remote_status == "ACTIVE":
                                cloud_update("options_journal", {"date_key": date_k},
                                             {"status": l_status, "current_option_price": _num(loc[3]), "pnl_pct": _num(loc[4]), "last_audited": l_ts}, "Re-push Closed Option")
                            elif closed_record_conflict(l_status, remote_status, l_ts, remote_ts):
                                info["closed_conflicts"] += 1
                            continue
                    ts_str = normalize_ts_str(o.get('timestamp'), f"{date_k} 15:30:00")
                    aud_str = normalize_ts_str(o.get('last_audited') or o.get('timestamp'), ts_str)
                    cur_p = _num(o.get('current_option_price'))
                    if _num(o.get('entry_premium'), cur_p) <= 0 or not str(o.get('option_contract') or '').strip():
                        raise ValueError(f"option {date_k}: missing contract or price")
                    con.execute("""
                        INSERT OR REPLACE INTO daily_options_journal
                        (date_key, timestamp, share_name, option_contract, strike_price, expiry_days,
                         underlying_spot, lot_size, entry_premium, current_option_price, target_premium,
                         stop_loss_premium, total_capital, ai_confidence, implied_vol, status, pnl_pct, last_audited)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, [date_k, ts_str, str(o.get('share_name') or 'ICICIBANK'), str(o.get('option_contract') or ''), _num(o.get('strike_price')),
                          _int(o.get('expiry_days'), 28), _num(o.get('underlying_spot')), _int(o.get('lot_size'), 700),
                          _num(o.get('entry_premium'), cur_p), cur_p, _num(o.get('target_premium')), _num(o.get('stop_loss_premium')),
                          _num(o.get('total_capital')), _num(o.get('ai_confidence'), 70.0), _num(o.get('implied_vol'), 20.0), remote_status,
                          _num(o.get('pnl_pct')), aud_str])
                except Exception as e:
                    bad(e)
        finally:
            con.close()
    if info["bad_rows"]:
        record_db_error("Restore from cloud", Exception(f"{info['bad_rows']} backup row(s) were unreadable and skipped ({info['first_error']})"))
    if info["closed_conflicts"]:
        st.session_state["history_close_conflicts"] = info["closed_conflicts"]

if "hydrated_once" not in st.session_state:
    hydrate_duckdb_from_supabase()
    st.session_state["hydrated_once"] = True


def sync_local_history_to_supabase(batch_size: int = 200, offset: int = 0) -> dict:
    """Upload missing old records without overwriting existing cloud rows.

    Uses a resumable offset for large histories. This is best-effort only; verify
    cloud totals before trusting the remote database as your durable backup.
    """
    if not supabase:
        raise RuntimeError("Add SUPABASE_URL and SUPABASE_KEY to Streamlit Secrets first")
    with DB_LOCK:
        con = duckdb.connect(DB_PATH, read_only=True)
        try:
            eq_rows = con.execute("SELECT * FROM trade_journal ORDER BY date_str DESC NULLS LAST").df().to_dict("records")
            opt_rows = con.execute("SELECT * FROM daily_options_journal ORDER BY date_key DESC NULLS LAST").df().to_dict("records")
        finally:
            con.close()
    records = [("share", r) for r in eq_rows] + [("option", r) for r in opt_rows]
    start = max(0, min(int(offset), len(records)))
    end = min(len(records), start + max(1, int(batch_size)))
    stats = {"shares_inserted": 0, "options_inserted": 0, "already_remote": 0,
             "failed": 0, "errors": [], "more_to_check": False, "next_offset": start,
             "total_rows": len(records)}
    first_failed = None
    for i in range(start, end):
        kind, row = records[i]
        try:
            if kind == "share":
                ticker = str(row.get("ticker") or "").strip()
                dt = str(row.get("date_str") or "")[:10]
                if not ticker or len(dt) != 10:
                    stats["next_offset"] = i + 1
                    continue
                prior = supabase.table("predictions").select("id").eq("ticker",ticker).eq("predicted_date",dt).limit(1).execute()
                if prior.data:
                    stats["already_remote"] += 1
                else:
                    entry = _num(row.get("entry_price"))
                    payload = {
                        "predicted_date": dt, "ticker": ticker,
                        "company_name": f"{ticker} (restored paper history)",
                        "entry_price": entry, "target_price": _num(row.get("target_price")),
                        "stop_loss": _num(row.get("stop_loss")),
                        "position_gbp": _num(row.get("capital_allocated")),
                        "shares_qty": _int(row.get("shares")),
                        "status": str(row.get("status") or "UNKNOWN"),
                        "latest_price": _num(row.get("latest_price"),entry),
                        "pnl_pct": _num(row.get("pnl_pct")),
                        "features_json": _parse_features(row.get("features_json")),
                        "last_checked": str(row.get("last_audited") or row.get("timestamp") or f"{dt} 15:30:00")[:19],
                    }
                    supabase.table("predictions").insert(payload).execute()
                    stats["shares_inserted"] += 1
            else:
                key = str(row.get("date_key") or "")[:10]
                if not key or len(key) != 10:
                    stats["next_offset"] = i + 1
                    continue
                prior = supabase.table("options_journal").select("date_key").eq("date_key",key).limit(1).execute()
                if prior.data:
                    stats["already_remote"] += 1
                else:
                    payload = {}
                    for c in ("date_key", "timestamp", "share_name", "option_contract", "strike_price",
                              "expiry_days", "underlying_spot", "lot_size", "entry_premium",
                              "current_option_price", "target_premium", "stop_loss_premium",
                              "total_capital", "ai_confidence", "implied_vol", "status", "pnl_pct", "last_audited"):
                        v = row.get(c)
                        if v is not None and not pd.isna(v):
                            payload[c] = v.item() if hasattr(v, "item") else v
                    supabase.table("options_journal").insert(payload).execute()
                    stats["options_inserted"] += 1
        except Exception as exc:
            stats["failed"] += 1
            if first_failed is None:
                first_failed = i
            stats["errors"].append(f"{kind}: {type(exc).__name__}: {str(exc)[:120]}")
            if stats["failed"] >= 3:
                break
        stats["next_offset"] = i + 1
    if first_failed is not None:
        stats["next_offset"] = first_failed  # allow retry; never silently skip a failed cloud record
    stats["more_to_check"] = stats["next_offset"] < len(records)
    return stats

def persist_daily_option_to_supabase(signal: dict):
    """Best-effort save for a newly logged option idea, without remote overwrite."""
    if not supabase or not isinstance(signal, dict):
        return
    date_key = str(signal.get("date_key") or "")[:10]
    if not date_key or st.session_state.get("last_option_cloud_verified") == date_key:
        return
    try:
        prior = supabase.table("options_journal").select("date_key").eq("date_key",date_key).limit(1).execute()
        if not prior.data:
            payload = {}
            for col in ("date_key", "timestamp", "share_name", "option_contract", "strike_price",
                        "expiry_days", "underlying_spot", "lot_size", "entry_premium",
                        "current_option_price", "target_premium", "stop_loss_premium", "total_capital",
                        "ai_confidence", "implied_vol", "status", "pnl_pct", "last_audited"):
                value = signal.get(col)
                if value is not None and not pd.isna(value):
                    payload[col] = value.item() if hasattr(value, "item") else value
            supabase.table("options_journal").insert(payload).execute()
        st.session_state["last_option_cloud_verified"] = date_key
    except Exception as exc:
        record_db_error("Save daily option history", exc)

# ==============================================================================
# 5. PRICE FEED (YAHOO V8 CHART) WITH PARALLEL PREFETCH
# ==============================================================================
def _fetch_history_raw(yf_sym: str, period: str = "6mo", interval: str = "1d") -> pd.DataFrame:
    encoded_sym = urllib.parse.quote(str(yf_sym).strip(), safe="")
    hosts = ["https://query1.finance.yahoo.com", "https://query2.finance.yahoo.com"]
    for idx, host in enumerate(hosts):
        try:
            url = f"{host}/v8/finance/chart/{encoded_sym}"
            params = {"range": period, "interval": interval, "includePrePost": "false"}
            headers = {
                "User-Agent": USER_AGENTS[idx % len(USER_AGENTS)],
                "Accept": "application/json,text/plain,*/*",
                "Accept-Language": "en-US,en;q=0.9",
                "Origin": "https://finance.yahoo.com",
                "Referer": "https://finance.yahoo.com/"
            }
            resp = yf_session.get(url, params=params, headers=headers, timeout=4)
            if resp.status_code == 200:
                payload = resp.json()
                result_list = (payload.get("chart") or {}).get("result")
                if result_list:
                    res_obj = result_list[0]
                    timestamps = res_obj.get("timestamp") or []
                    quote_list = ((res_obj.get("indicators") or {}).get("quote") or [{}])
                    quote = quote_list[0] if quote_list else {}
                    if timestamps and "close" in quote:
                        idx_ist = pd.to_datetime(timestamps, unit="s", utc=True).tz_convert("Asia/Kolkata").tz_localize(None)
                        df = pd.DataFrame({
                            "Open": quote.get("open", [None] * len(timestamps)),
                            "High": quote.get("high", [None] * len(timestamps)),
                            "Low": quote.get("low", [None] * len(timestamps)),
                            "Close": quote.get("close", [None] * len(timestamps)),
                            "Volume": quote.get("volume", [0] * len(timestamps)),
                        }, index=idx_ist)
                        df = df.dropna(subset=["Close"]).copy()
                        if not df.empty:
                            df["Open"] = df["Open"].fillna(df["Close"]).astype(float)
                            df["High"] = df["High"].fillna(df["Close"]).astype(float)
                            df["Low"] = df["Low"].fillna(df["Close"]).astype(float)
                            df["Close"] = df["Close"].astype(float)
                            df["Volume"] = df["Volume"].fillna(0).astype(float)
                            return df
        except Exception:
            pass
    return pd.DataFrame()

@st.cache_data(ttl=120, show_spinner=False)
def fetch_cached_history(yf_sym: str, period: str = "6mo", interval: str = "1d") -> pd.DataFrame:
    return _fetch_history_raw(yf_sym, period, interval)

def prefetch_histories(syms: list, period: str = "6mo", ttl=120, workers: int = 6, progress=None, time_budget: float = None) -> bool:
    """Download price histories in parallel into a shared in-memory store.
    ttl can be a number or a function(symbol)->seconds. Returns False if the time budget ran out first."""
    store = get_hist_store()
    now = time.time()
    ttl_of = ttl if callable(ttl) else (lambda _s: ttl)
    with store["lock"]:
        for k in [k for k, v in store["data"].items() if now - v[0] > QUIET_TTL_SEC + 600]:
            del store["data"][k]
        todo = [s for s in dict.fromkeys(syms) if (s, period) not in store["data"] or now - store["data"][(s, period)][0] > ttl_of(s)]
    if not todo:
        if progress:
            progress(1, 1)
        return True
    started, done, complete = time.time(), 0, True
    batch_n = max(workers * 6, 24)
    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        for i in range(0, len(todo), batch_n):
            if time_budget is not None and time.time() - started > time_budget:
                complete = False
                break
            part = todo[i:i + batch_n]
            res = list(ex.map(lambda s: (s, _fetch_history_raw(s, period)), part))
            stamp = time.time()
            with store["lock"]:
                for s, df in res:
                    old = store["data"].get((s, period))
                    if df.empty and old is not None and not old[1].empty:
                        continue                                    # never replace good data with a failed download
                    store["data"][(s, period)] = (stamp, df)
            done += len(part)
            if progress:
                progress(done, len(todo))
            if len(part) >= 12 and sum(1 for _, d in res if d.empty) > len(part) * 0.6:
                time.sleep(1.5)                                     # the price service looks busy: back off politely
    return complete

def peek_history(yf_sym: str, period: str = "6mo", max_age: float = None) -> pd.DataFrame:
    """Prices already downloaded by prefetch_histories (no network). With max_age, older copies count as missing."""
    store = get_hist_store()
    with store["lock"]:
        item = store["data"].get((yf_sym, period))
    if item is None or item[1] is None or item[1].empty:
        return pd.DataFrame()
    if max_age is not None and time.time() - item[0] > max_age:
        return pd.DataFrame()
    return item[1].copy()

def get_history_fast(yf_sym: str, period: str = "6mo") -> pd.DataFrame:
    df = peek_history(yf_sym, period, max_age=SWEEP_TTL_SEC)
    if not df.empty:
        return df
    return fetch_cached_history(yf_sym, period, "1d")

def _bar_dates(h: pd.DataFrame) -> list:
    return [pd.Timestamp(x).date() for x in h.index]

# ==============================================================================
# 6. NEWS / FILINGS CHECK (WHOLE-WORD MATCHING) & MACRO
# ==============================================================================
@st.cache_data(ttl=900, show_spinner=False)
def evaluate_nse_filing_nlp(ticker: str) -> dict:
    clean_sym = clean_sym_name(ticker)
    headlines = []
    base_penalty = 1.0
    try:
        ann = check_corporate_announcements(clean_sym)
        if isinstance(ann, dict):
            base_penalty = float(ann.get("risk_penalty", 1.0))
            for k in ["headline", "title", "subject", "summary", "reason"]:
                if ann.get(k):
                    headlines.append(str(ann[k]))
    except Exception:
        pass

    combined_text = " | ".join(headlines).lower()
    lex_delta = 0.0
    matched_tags = []
    for pat, phrase, weight in _BEAR_PATTERNS:
        if pat.search(combined_text):
            lex_delta += weight
            matched_tags.append(f"🚨 {phrase.title()} ({weight:+.0f}%)")
    for pat, phrase, weight in _BULL_PATTERNS:
        if pat.search(combined_text):
            lex_delta += weight
            matched_tags.append(f"🚀 {phrase.title()} ({weight:+.0f}%)")

    if base_penalty < 0.85 and lex_delta == 0.0:
        lex_delta -= 15.0
        matched_tags.append("🚨 Adverse Corporate Event (-15%)")

    total_delta = round(max(-30.0, min(12.0, lex_delta)), 1)
    is_hard_blocked = (total_delta <= -15.0) or (base_penalty <= 0.75)

    if not ANNOUNCEMENTS_AVAILABLE:
        status = "⚪ News check unavailable"
    elif is_hard_blocked:
        status = "🚨 Bad news — blocked"
    elif total_delta >= 4.0:
        status = "🟢 Good news"
    else:
        status = "📰 No red flags"

    primary_headline = headlines[0][:120] if headlines else "No recent headline found"
    return {
        "status": status, "delta": total_delta,
        "nlp_tags": ", ".join(matched_tags) if matched_tags else "Nothing unusual",
        "headline": primary_headline, "is_blocked": is_hard_blocked, "checked": ANNOUNCEMENTS_AVAILABLE
    }

@st.cache_data(ttl=1800, show_spinner=False)
def get_indian_cross_asset_macro() -> dict:
    macro = {"nifty_5d": 0.0, "usdinr_5d": 0.0, "brent_5d": 0.0, "vix_level": 14.0}
    symbols = {"nifty_5d": "^NSEI", "usdinr_5d": "INR=X", "brent_5d": "BZ=F"}
    for key, sym in symbols.items():
        try:
            h = fetch_cached_history(sym, period="1mo", interval="1d")
            if not h.empty and len(h) >= 5:
                c_now = float(h["Close"].iloc[-1])
                c_5d = float(h["Close"].iloc[-5])
                if c_5d > 0:
                    macro[key] = round(((c_now - c_5d) / c_5d) * 100.0, 2)
        except Exception:
            pass
    try:
        h_vix = fetch_cached_history("^INDIAVIX", period="5d", interval="1d")
        if not h_vix.empty:
            macro["vix_level"] = round(float(h_vix["Close"].iloc[-1]), 1)
    except Exception:
        pass
    return macro

def is_market_stress(macro: dict) -> bool:
    return float(macro.get("nifty_5d", 0.0)) <= STRESS_NIFTY_5D or float(macro.get("vix_level", 14.0)) >= STRESS_VIX

def compute_nse_macro_lead_lag(ticker: str, macro: dict) -> tuple:
    t = clean_sym_name(ticker).upper()
    delta = 0.0
    notes = []
    brent = macro.get("brent_5d", 0.0)
    usdinr = macro.get("usdinr_5d", 0.0)
    nifty = macro.get("nifty_5d", 0.0)

    if t in INR_IT_PHARMA_EXPORTERS and usdinr >= 0.4:
        delta += 2.5
        notes.append(f"+2.5% (weaker rupee helps exporters, +{usdinr:.1f}%)")
    if t in CRUDE_SENSITIVE_USERS and brent >= 3.0:
        delta -= 3.0
        notes.append(f"-3.0% (oil up +{brent:.1f}% hurts costs)")
    elif t in CRUDE_SENSITIVE_USERS and brent <= -3.0:
        delta += 2.5
        notes.append(f"+2.5% (oil down {brent:.1f}% helps costs)")
    if nifty <= -2.0:
        delta -= 2.0
        notes.append(f"-2.0% (Nifty down {nifty:.1f}% this week)")
    elif nifty >= 1.5 and delta == 0:
        delta += 1.5
        notes.append(f"+1.5% (Nifty up +{nifty:.1f}% this week)")
    return round(max(-5.0, min(5.0, delta)), 1), (" | ".join(notes) if notes else "Nothing unusual in the markets")

# ==============================================================================
# 7. AI MODEL, CALIBRATION, MEMORY & GOVERNED RETRAINING
# ==============================================================================
def get_ml_model():
    if "ml_model_bundle" in st.session_state:
        return st.session_state["ml_model_bundle"]
    for path in (LIVE_MODEL_PATH, MODEL_PATH):
        if os.path.exists(path):
            try:
                bundle = joblib.load(path)
                st.session_state["ml_model_bundle"] = bundle
                st.session_state["ml_model_source"] = "Upgraded model" if path == LIVE_MODEL_PATH else "Original model"
                return bundle
            except Exception:
                continue
    return None

def ensemble_proba(bundle, X: pd.DataFrame):
    """Average win probability from whatever model format is stored. Returns an array (one value per row)."""
    parts = []
    if isinstance(bundle, dict):
        for k in ("lgb", "xgb"):
            if k in bundle:
                parts.append(np.asarray(bundle[k].predict_proba(X))[:, 1])
        if not parts and "model" in bundle:
            parts.append(np.asarray(bundle["model"].predict_proba(X))[:, 1])
    else:
        parts.append(np.asarray(bundle.predict_proba(X))[:, 1])
    if not parts:
        raise ValueError("unrecognised model format")
    return np.mean(parts, axis=0)

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

def get_closed_loop_self_learning(ticker: str, current_atr_pct: float, hist=None) -> dict:
    """Exponentially decayed memory of past wins/losses on this stock (+ a caution for very jumpy shares)."""
    delta = 0.0
    reasons = []
    today_dt = ist_now().date()
    if hist is None:
        hist = load_closed_trades_df()
    if hist is None or hist.empty:
        return {"delta": 0.0, "reason": "Nothing learned yet"}
    try:
        t_clean = clean_sym_name(ticker)
        t_hist = hist[hist["ticker"].apply(clean_sym_name) == t_clean]
        decayed_loss = decayed_win = 0.0
        for _, r in t_hist.iterrows():
            d = _to_date(_first_valid(r.get("exit_timestamp"), r.get("last_audited"), r.get("date_str")))
            days_ago = max(0, (today_dt - d).days) if d else 15
            weight = math.pow(0.5, days_ago / HALF_LIFE_DAYS)
            o = classify_outcome(r["status"])
            if o == "LOSS":
                decayed_loss += 15.0 * weight
            elif o == "WIN":
                decayed_win += 5.0 * weight
        if decayed_loss >= 1.0:
            delta -= round(decayed_loss, 1)
            reasons.append(f"-{decayed_loss:.1f}% (lost on this stock recently)")
        if decayed_win >= 1.0:
            delta += round(decayed_win, 1)
            reasons.append(f"+{decayed_win:.1f}% (won on this stock recently)")
        losses = hist[hist["status"].astype(str).str.contains("LOSS", na=False)]
        loss_atrs = []
        for f_str in losses["features_json"].dropna():
            f_obj = _parse_features(f_str)
            a = f_obj.get("atr_pct", (f_obj.get("raw_features") or {}).get("atr_pct"))
            if a is not None:
                try:
                    loss_atrs.append(float(a))
                except Exception:
                    pass
        if len(loss_atrs) >= 5:
            p75 = max(3.0, float(np.percentile(loss_atrs, 75)))
            if current_atr_pct >= p75:
                delta -= 8.0
                reasons.append(f"-8% (very jumpy — losing trades were this jumpy, {p75:.1f}%+ a day)")
    except Exception:
        pass
    return {"delta": round(delta, 1), "reason": " | ".join(reasons) if reasons else "No past lesson applies"}

def build_calibrator(closed_df: pd.DataFrame):
    """Turns the model's raw probability into an honest one using real results."""
    wins = losses = 0
    xs, ys = [], []
    if closed_df is not None and not closed_df.empty:
        for _, r in closed_df.iterrows():
            o = classify_outcome(r["status"])
            if o == "WIN":
                y = 1
            elif o == "LOSS":
                y = 0
            else:
                continue
            wins += y
            losses += (1 - y)
            f = _parse_features(r["features_json"])
            rp = f.get("raw_prob")
            if rp is None and f.get("raw_ml_prob") is not None:
                rp = float(f["raw_ml_prob"]) / 100.0
            if rp is not None:
                try:
                    xs.append(float(rp))
                    ys.append(y)
                except Exception:
                    pass
    decided = wins + losses
    prior = min(0.90, max(0.10, (wins + 0.5 * CAL_PRIOR_TRADES) / (decided + CAL_PRIOR_TRADES)))   # your real win rate, gently pulled to 50% while results are few
    iso = None
    mode = f"Pulled toward your real win rate ({prior:.0%}, from {decided} clear results)"
    if len(xs) >= MIN_CALIBRATION_SAMPLES and len(set(ys)) == 2:
        try:
            from sklearn.isotonic import IsotonicRegression
            iso = IsotonicRegression(out_of_bounds="clip").fit(xs, ys)
            mode = f"Tuned on {len(xs)} real results"
        except Exception:
            iso = None

    ref = float(np.mean(xs)) if len(xs) >= 10 else 0.5      # how the AI usually scored its past picks

    def calibrate(p):
        # A share scoring like the AI's past picks gets your real win rate; only a clearly higher score earns more.
        base = prior + CONF_SHRINK * (float(p) - ref)
        if iso is not None:
            try:
                base = 0.5 * float(iso.predict([float(p)])[0]) + 0.5 * base
            except Exception:
                pass
        return min(0.95, max(0.05, base))

    return calibrate, {"mode": mode, "prior": prior, "decided": decided, "ref_score": round(ref, 3)}

def check_and_auto_retrain_model(feature_cols: list, closed_df: pd.DataFrame):
    """Champion/challenger: a new model is trained on live trades, tested on the most recent 30%, and only
    replaces the old one (saved to a SEPARATE file) if it is clearly better. Your original file is never touched."""
    if closed_df is None or closed_df.empty:
        return
    records = []
    for _, r in closed_df.iterrows():
        o = classify_outcome(r["status"])
        if o not in ("WIN", "LOSS"):
            continue
        raw = _parse_features(r["features_json"]).get("raw_features", {})
        if not isinstance(raw, dict) or not all(c in raw for c in feature_cols):
            continue
        try:
            vec = [float(raw[c]) for c in feature_cols]
        except Exception:
            continue
        records.append((str(r.get("date_str") or ""), vec, 1 if o == "WIN" else 0))
    n = len(records)
    if n < MIN_SETTLED_TO_RETRAIN:
        return
    try:
        state = json.loads(get_meta("retrain_state", "{}") or "{}")
    except Exception:
        state = {}
    today_s = ist_now().strftime("%Y-%m-%d")
    if state.get("last_attempt_date") == today_s or n - int(state.get("last_labeled_count", 0)) < RETRAIN_STEP_INTERVAL:
        return
    records.sort(key=lambda x: x[0])
    X_all = pd.DataFrame([r[1] for r in records], columns=feature_cols)
    y_all = np.array([r[2] for r in records])
    split = int(n * 0.7)
    X_tr, X_te, y_tr, y_te = X_all.iloc[:split], X_all.iloc[split:], y_all[:split], y_all[split:]
    state.update({"last_attempt_date": today_s, "last_labeled_count": n})
    if len(np.unique(y_tr)) < 2 or len(np.unique(y_te)) < 2:
        set_meta("retrain_state", json.dumps(state))
        return
    try:
        from lightgbm import LGBMClassifier
        from xgboost import XGBClassifier
        from sklearn.metrics import log_loss, roc_auc_score

        def fit_pair(X, y):
            m1 = LGBMClassifier(n_estimators=250, learning_rate=0.03, max_depth=5, min_child_samples=10, random_state=42, verbose=-1)
            m1.fit(X, y)
            m2 = XGBClassifier(n_estimators=300, learning_rate=0.03, max_depth=5, random_state=42, eval_metric="logloss")
            m2.fit(X, y)
            return m1, m2

        c_l, c_x = fit_pair(X_tr, y_tr)
        p_ch = np.clip(0.5 * c_l.predict_proba(X_te)[:, 1] + 0.5 * c_x.predict_proba(X_te)[:, 1], 1e-4, 1 - 1e-4)
        champ = get_ml_model()
        if champ is None:
            set_meta("retrain_state", json.dumps(state))
            return
        p_cm = np.clip(ensemble_proba(champ, X_te), 1e-4, 1 - 1e-4)
        ll_ch, ll_cm = float(log_loss(y_te, p_ch)), float(log_loss(y_te, p_cm))
        auc_ch, auc_cm = float(roc_auc_score(y_te, p_ch)), float(roc_auc_score(y_te, p_cm))
        if (ll_ch < ll_cm * 0.98) and (auc_ch >= auc_cm):
            f_l, f_x = fit_pair(X_all, y_all)
            new_bundle = {"lgb": f_l, "xgb": f_x, "type": "ensemble", "feature_cols": feature_cols}
            os.makedirs(os.path.dirname(LIVE_MODEL_PATH), exist_ok=True)
            # Save a rollback copy of the previous champion before promotion.
            old = Path(LIVE_MODEL_PATH)
            if old.exists():
                archive = ROOT_DIR / "models" / "archive"
                archive.mkdir(parents=True, exist_ok=True)
                stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                shutil.copy2(old, archive / f"previous_champion_{stamp}.pkl")
            temp = LIVE_MODEL_PATH + ".tmp"
            joblib.dump(new_bundle, temp)
            os.replace(temp, LIVE_MODEL_PATH)
            st.session_state["ml_model_bundle"] = new_bundle
            st.session_state["ml_model_source"] = "Upgraded model"
            state["last_promoted"] = ist_now().strftime("%Y-%m-%d %H:%M")
            st.session_state["retrain_notice"] = "🧠 The AI upgraded itself: the new version tested better than the old one."
        else:
            st.session_state["retrain_notice"] = "🧪 The AI tested a new version but kept the old one (the new one wasn't better)."
        set_meta("retrain_state", json.dumps(state))
    except Exception as e:
        set_meta("retrain_state", json.dumps(state))
        record_db_error("AI upgrade check", e)

# ==============================================================================
# 8. OPTIONS ENGINE (LIVE NSE QUOTE WHEN AVAILABLE, OTHERWISE A CLEARLY-LABELLED MODEL PRICE)
# ==============================================================================
def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

def calculate_black_scholes_call(spot: float, strike: float, days_to_exp: float, r: float, sigma: float) -> float:
    T = max(days_to_exp, 1.0) / 365.0
    if spot <= 0 or strike <= 0 or sigma <= 0:
        return max(0.0, spot - strike)
    d1 = (math.log(spot / strike) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    call_price = spot * norm_cdf(d1) - strike * math.exp(-r * T) * norm_cdf(d2)
    return max(round(call_price, 2), 0.05)

def model_option_price(spot, strike, days, sigma, side: str) -> float:
    mid = calculate_black_scholes_call(spot, strike, days, RISK_FREE_RATE, sigma)
    if side == "ask":
        return round(mid * (1.0 + OPTION_MODEL_HALF_SPREAD), 2)
    return max(round(mid * (1.0 - OPTION_MODEL_HALF_SPREAD), 2), 0.05)

def get_nse_monthly_expiry(current_date: datetime) -> tuple:
    """NSE stock-option monthly expiry = LAST TUESDAY of the month (changed from Thursday in Sept 2025).
    Holidays aren't known here; if the exchange moves the date earlier, the contract is simply marked at expiry."""
    def last_tuesday_of(y, m):
        cal = calendar.monthcalendar(y, m)
        tuesdays = [week[1] for week in cal if week[1] != 0]
        return datetime(y, m, tuesdays[-1], 15, 30)
    year, month = current_date.year, current_date.month
    expiry_dt = last_tuesday_of(year, month)
    days_left = (expiry_dt.date() - current_date.date()).days
    if days_left < 5:
        month2 = month + 1 if month < 12 else 1
        year2 = year if month < 12 else year + 1
        expiry_dt = last_tuesday_of(year2, month2)
        days_left = (expiry_dt.date() - current_date.date()).days
    return expiry_dt, max(1, days_left)

def _nse_get_json(url: str):
    box = get_nse_http_box()
    with box["lock"]:
        now = time.time()
        if box["session"] is None or now - box["ts"] > 600:
            s = requests.Session()
            s.headers.update({"User-Agent": USER_AGENTS[0], "Accept": "application/json,text/plain,*/*",
                              "Accept-Language": "en-US,en;q=0.9", "Referer": "https://www.nseindia.com/option-chain"})
            try:
                s.get("https://www.nseindia.com", timeout=5)      # NSE needs this handshake to hand out cookies
            except Exception:
                pass
            box["session"], box["ts"] = s, now
        s = box["session"]
    is_option_chain = "/api/option-chain" in url
    try:
        r = s.get(url, timeout=6)
        if is_option_chain:
            st.session_state["_nse_option_http_status"] = str(r.status_code)
        if r.status_code == 200:
            try:
                return r.json()
            except (ValueError, TypeError):
                if is_option_chain:
                    st.session_state["_nse_option_http_status"] = "invalid-json"
    except requests.Timeout:
        if is_option_chain:
            st.session_state["_nse_option_http_status"] = "timeout"
    except requests.RequestException:
        if is_option_chain:
            st.session_state["_nse_option_http_status"] = "network-error"
    except Exception:
        if is_option_chain:
            st.session_state["_nse_option_http_status"] = "unexpected-error"
    return None

# ---------- public-holiday awareness ----------
def get_nse_holidays() -> dict:
    """{'YYYY-MM-DD': 'Holiday name'} from NSE's own trading-holiday list. Cached; retried every 30 min if NSE didn't answer."""
    box = get_nse_http_box()
    now = time.time()
    cached = box.get("holidays")
    if cached and (now - cached[0] < (43200 if cached[1] else 1800)):
        return cached[1]
    out = {}
    data = _nse_get_json("https://www.nseindia.com/api/holiday-master?type=trading")
    try:
        for seg in ("CM", "FO"):
            for item in (data or {}).get(seg, []) or []:
                d = datetime.strptime(str(item.get("tradingDate")), "%d-%b-%Y").date()
                out[d.isoformat()] = str(item.get("description", "Market holiday"))[:60]
            if out:
                break
    except Exception:
        out = {}
    box["holidays"] = (now, out)
    return out

def get_market_state() -> dict:
    """Is today a trading day? Uses weekends, NSE's holiday list, and (if the list is unavailable) a live check that the
    Nifty has a price bar dated today. If every check fails we assume a normal trading day."""
    now = ist_now()
    today = now.date()
    if now.weekday() > 4:
        return {"trading_day": False, "reason": "Weekend"}
    hol = get_nse_holidays()
    if today.isoformat() in hol:
        return {"trading_day": False, "reason": f"Market holiday — {hol[today.isoformat()]}"}
    if not hol and dtime(9, 45) <= now.time() <= dtime(15, 30):
        try:
            h = fetch_cached_history("^NSEI", period="5d", interval="1d")
            if not h.empty and _bar_dates(h)[-1] != today:
                return {"trading_day": False, "reason": "No trading data today — probably a market holiday"}
        except Exception:
            pass
    return {"trading_day": True, "reason": ""}

def next_open_text() -> str:
    now = ist_now()
    hol = get_nse_holidays()
    for add in range(0, 12):
        d = now.date() + timedelta(days=add)
        if d.weekday() > 4 or d.isoformat() in hol:
            continue
        if add == 0 and now.time() >= dtime(15, 30):
            continue
        if add == 0 and now.time() >= dtime(9, 15):
            return "now"
        when = "today" if add == 0 else ("tomorrow" if add == 1 else d.strftime("%a %d %b"))
        return f"{when} 09:15 IST"
    return "soon"

def upcoming_holidays(n: int = 4) -> list:
    today = ist_now().date().isoformat()
    return [(d, name) for d, name in sorted(get_nse_holidays().items()) if d >= today][:n]

def _track_option_request(health: dict | None, endpoint: str) -> None:
    if health is None:
        return
    status = str(st.session_state.get("_nse_option_http_status", "not-requested"))
    key = f"{endpoint} {status}"
    counts = health["http_status_counts"]
    counts[key] = counts.get(key, 0) + 1


def _option_payload_diagnostic(health, endpoint: str, symbol: str,
                               expiry: str, payload) -> None:
    """Count sanitized response shapes, not just HTTP statuses.

    HTTP 200 can be an empty list, an error envelope, an unrelated expiry or
    valid actual contract rows. Never store raw bid/ask data in session logs.
    """
    if health is None:
        return
    shape, reason = option_payload_shape(payload, expiry)
    counter = {
        'empty': 'empty_payloads',
        'other-expiry': 'different_expiry_payloads',
        'undated': 'undated_payloads',
        'unknown-shape': 'unknown_shape_payloads',
        'not-json-object': 'unknown_shape_payloads',
    }.get(shape)
    if counter:
        health[counter] = health.get(counter, 0) + 1
        examples = health.setdefault('payload_examples', [])
        if len(examples) < 4:
            examples.append(f'{symbol} {expiry} {endpoint}: {reason}')


def _exchange_equity_option_chain(symbol: str, expiry_label: str | None = None,
                                  lots_for_symbol: dict | None = None,
                                  health: dict | None = None):
    """Require explicit expiry rows, real exchange timestamps and verified lots.

    Tries a second *exchange-listed* supported expiry if the first returns
    no valid chain. An HTTP 200 or contract-info list cannot become a quote.
    No more than two expiry-scoped attempts per underlying.
    """
    name = clean_sym_name(symbol)
    quoted = urllib.parse.quote(name, safe='')
    today = ist_now().date()
    old_url = 'https://www.nseindia.com/api/option-chain-equities?symbol=' + quoted
    legacy = None
    if expiry_label:
        dt = parse_expiry_date(expiry_label)
        if dt is None:
            return None
        candidates = [(dt.strftime('%d-%b-%Y'), (dt - today).days, 0)]
    else:
        info = _nse_get_json('https://www.nseindia.com/api/option-chain-contract-info?symbol=' + quoted)
        _track_option_request(health, 'contract-info')
        if health is not None and expiry_labels(info):
            health['contract_info_received'] += 1
        candidates = supported_expiries(info, lots_for_symbol or {}, today)
        if not candidates:
            legacy = _nse_get_json(old_url)
            _track_option_request(health, 'legacy')
            candidates = supported_expiries(legacy, lots_for_symbol or {}, today)
        if not candidates:
            if health is not None:
                health['no_supported_expiry_lot'] += 1
            return None
        # Prefer soonest expiry, but never invent an alternative or its lot.
        candidates = candidates[:2]

    stale_candidate = None
    for index, (selected, days_left, lot) in enumerate(candidates):
        if health is not None:
            health['expiry_matches'] += 1
            health['expiry_attempts'] = health.get('expiry_attempts', 0) + 1
            if index:
                health['alternate_expiry_attempts'] = health.get('alternate_expiry_attempts', 0) + 1
        url = ('https://www.nseindia.com/api/option-chain-v3?type=Equity&symbol=' + quoted +
               '&expiry=' + urllib.parse.quote(selected, safe=''))
        v3 = _nse_get_json(url)
        _track_option_request(health, 'v3')
        _option_payload_diagnostic(health, 'v3', name, selected, v3)
        v3_rows = matching_rows(v3, selected)
        if v3_rows:
            if health is not None:
                health['v3_rows_received'] += 1
            if quote_is_fresh(v3, ist_now()):
                return v3, selected, days_left, lot
            if stale_candidate is None:
                stale_candidate = (v3, selected, days_left, lot)

        # If v3 failed or is stale, check legacy once and reuse it for another
        # expiry; the legacy endpoint is not expiry-scoped.
        if legacy is None:
            legacy = _nse_get_json(old_url)
            _track_option_request(health, 'legacy')
        _option_payload_diagnostic(health, 'legacy', name, selected, legacy)
        legacy_rows = matching_rows(legacy, selected)
        if legacy_rows:
            if health is not None:
                health['legacy_rows_received'] += 1
            if quote_is_fresh(legacy, ist_now()):
                return legacy, selected, days_left, lot
            if stale_candidate is None:
                stale_candidate = (legacy, selected, days_left, lot)
        # Honour NSE rate limiting; do not multiply requests after 429.
        if health is not None and any(str(key).endswith(' 429')
                                      for key in health.get('http_status_counts', {})):
            break

    if stale_candidate is not None:
        return stale_candidate  # caller will refuse stale quote; counter stays honest
    if health is not None:
        health['empty_expiry_rows'] = health.get('empty_expiry_rows', 0) + 1
    return None

def get_nse_option_quote(symbol: str, expiry_dt: datetime, strike: float, side: str = "CE"):
    """Refresh the EXACT exchange CE/PE contract; no model-price substitutions."""
    if side not in ("CE", "PE"):
        return None
    chain_result = _exchange_equity_option_chain(symbol, expiry_dt.strftime("%d-%b-%Y"))
    if not chain_result:
        return None
    chain, expiry_str, _, _ = chain_result
    if not _quote_age_is_valid(chain):
        return None
    leg = exact_contract(matching_rows(chain, expiry_str), side, strike)
    if not leg:
        return None
    bid, ask = bid_ask(leg)
    if bid <= 0 or ask <= 0:
        return None
    return {"bid":bid, "ask":ask, "oi":_num(leg.get("openInterest")),
            "iv":_num(leg.get("impliedVolatility")),
            "spread_pct":round(200*(ask-bid)/(ask+bid),2)}


def _quote_age_is_valid(chain: dict) -> bool:
    return quote_is_fresh(chain, ist_now())


def parse_nse_lot_file(csv_content: str) -> dict:
    """Parse current NSE lot file by exact ticker SYMBOL column, not company name."""
    from nse_lot_parser import parse_nse_lot_file as parse_official_nse_lots
    return parse_official_nse_lots(csv_content)


def _current_nse_lots() -> dict:
    """Current NSE month-specific F&O lot sizes (no guessed values)."""
    box = get_nse_http_box()
    now = time.time()
    cached = box.get("fo_lots")
    if cached and now-cached[0] < (21600 if cached[1] else 900):
        return cached[1]
    content = _http_text("https://nsearchives.nseindia.com/content/fo/fo_mktlots.csv", timeout=10)
    lots = parse_nse_lot_file(content)
    box["fo_lots"] = (now, lots)
    return lots

def option_paper_feedback() -> dict:
    """Directional outcome evidence from old *live-quoted* paper options.

    Synthetic model-priced outcomes remain in the database but must not train buy ratings.
    The result is deliberately a small caution gate, NOT a calibrated forecast.
    """
    d = load_options_journal_df()
    if d.empty:
        return {}
    d = d[(d["status"].apply(classify_outcome) != "ACTIVE") &
          d["option_contract"].apply(is_live_option)].copy()
    if d.empty:
        return {}
    d["side"] = d["option_contract"].apply(lambda x: "PE" if re.search(r"\bPE\b",str(x)) else "CE")
    # Outcomes marked win/loss only: do not treat unclassified or still-open as training evidence.
    d = d[d["status"].apply(classify_outcome).isin(("WIN", "LOSS", "EXPIRED"))]
    d["pnl_pct"] = pd.to_numeric(d["pnl_pct"],errors="coerce")
    d = d[np.isfinite(d["pnl_pct"]) & (d["pnl_pct"] > -101) & (d["pnl_pct"] < 1000)]
    return {side:{"n":int(len(g)),"mean":float(g["pnl_pct"].mean()),
                  "profitable":float((g["pnl_pct"] > 0).mean())}
            for side,g in d.groupby("side")}


def scan_option_candidates(max_underlyings: int = 35) -> pd.DataFrame:
    """Review nearby exchange-listed CE AND PE strikes with exact live NSE quotes.

    The most suitable validated candidate of each type is shown per underlying.
    Expanded diagnostics distinguish API access from real bid/ask contract reviews.
    Quoted contract rows never depend on theoretical option premiums.
    """
    from scan_health import new_option_scan_health, option_scan_summary
    started = time.monotonic()
    now = ist_now()
    # Keep the last MARKET-HOURS diagnostics when a browser refresh occurs after
    # 15:00. Never mistake an outside-window refresh for a successful option scan.
    if not is_entry_window_open():
        st.session_state.setdefault("option_scan_health", new_option_scan_health())
        st.session_state["option_scan_note"] = (
            "NSE is outside the verified entry window. New CE/PE ratings are paused until "
            "a fresh market-hours option chain is available. Your historical options remain below.")
        return pd.DataFrame()
    st.session_state["option_scan_health"] = new_option_scan_health()
    st.session_state.pop("opt_note", None)
    lots = _current_nse_lots()
    if not lots:
        st.session_state["option_scan_note"] = "Cannot verify NSE lot sizes; option trade recommendations paused."
        return pd.DataFrame()
    all_eq, _ = get_all_nse_symbols()
    from option_universe_health import validate_fno_universe
    coverage = validate_fno_universe(lots, all_eq)
    st.session_state["fno_coverage"] = {k:v for k,v in coverage.items() if k != "matched_symbols"}
    if not coverage["plausible"]:
        st.session_state["option_scan_note"] = (
            "⚠️ Incomplete F&O data: " + coverage["problem"] +
            " No complete-market option scan or buyable signal is claimed.")
        return pd.DataFrame()
    syms = coverage["matched_symbols"]
    batch = max(1, min(max_underlyings, len(syms)))
    # Cursor advances by the ACTUAL number attempted. If a request time limit
    # interrupts the batch, the unvisited names are considered next refresh.
    cursor = int(st.session_state.get("option_scan_cursor", now.date().toordinal()*17)) % len(syms)
    selected = [syms[(cursor+i) % len(syms)] for i in range(batch)]
    health = new_option_scan_health(len(selected), len(syms))
    prefetch_histories([t+".NS" for t in selected], "3mo", workers=6, ttl=SWEEP_TTL_SEC, time_budget=70)
    eq_used, opt_used = get_open_exposures()
    feedback = option_paper_feedback()
    reviews = []
    for name in selected:
        # On an NSE rate limit, stop requesting more data rather than hammering
        # public endpoints. Retain the same cursor position for unvisited names.
        if any(str(key).endswith(' 429') for key in health['http_status_counts']):
            health['rate_limited'] = 1
            break
        # Avoid a long series of NSE timeouts blocking an entire Streamlit session.
        if time.monotonic() - started > 150:
            health["time_budget_reached"] = 1
            break
        health["underlyings_attempted"] += 1
        h = get_history_fast(name+".NS", "3mo")
        if h.empty or len(h)<35:
            continue
        health["price_histories"] += 1
        c = h["Close"].astype(float)
        historical_spot = float(c.iloc[-1])
        if not math.isfinite(historical_spot) or historical_spot <= 0:
            continue
        change5 = (historical_spot/float(c.iloc[-6])-1)*100
        log_returns = np.log(c/c.shift(1)).dropna().tail(20)
        rv = float(log_returns.std()*np.sqrt(252)*100) if len(log_returns)>=15 else 0
        diff = c.diff().tail(14)
        up = float(diff.clip(lower=0).mean())
        down = float((-diff.clip(upper=0)).mean())
        rsi = 100-100/(1+up/max(1e-6, down))
        if abs(change5)<0.6:
            continue
        health["directional"] += 1
        fetched = _exchange_equity_option_chain(name, lots_for_symbol=lots[name], health=health)
        if not fetched:
            continue
        chain, expiry_label, days_left, lot = fetched
        health["chains_received"] += 1
        quote_fresh = is_entry_window_open() and _quote_age_is_valid(chain)
        if not quote_fresh:
            health["stale_or_untimed"] += 1
            continue
        health["fresh_chains"] += 1
        series = matching_rows(chain, expiry_label)
        if not series:
            health["empty_expiry_rows"] += 1
            continue
        health["chain_rows"] += len(series)
        spot = underlying_value(chain)
        if not spot:
            health["missing_underlying_spot"] += 1
            continue  # a Yahoo daily close is not a verified live NSE underlying quote
        if abs(spot-historical_spot)/historical_spot > 0.10:
            health["spot_disagreements"] += 1
            continue  # history and exchange cannot be reconciled confidently
        blocked = evaluate_nse_filing_nlp(name).get("is_blocked", False)
        for side in ("CE", "PE"):
            close_strikes = nearby_strikes(series, side, spot)
            if not close_strikes:
                health["missing_nearby_strikes"] += 1
                continue
            choices = []
            for contract in close_strikes:
                health["strikes_examined"] += 1
                strike = float(contract["strikePrice"])
                data = contract[side]
                bid, ask = bid_ask(data)
                if not (0 < bid <= ask):
                    health["missing_bid_ask"] += 1
                    continue
                health["valid_bid_ask"] += 1
                oi = _num(data.get("openInterest"))
                iv = _num(data.get("impliedVolatility"))
                result = review_long_option(kind=side, spot=spot, strike=strike, bid=bid, ask=ask,
                    lot_size=lot, days_left=days_left, open_interest=oi, iv_pct=iv,
                    realized_vol_pct=rv, ret_5d_pct=change5, rsi=rsi,
                    recent_filing_risk=blocked, equity_exposure=eq_used,
                    option_exposure=opt_used, quote_fresh=quote_fresh)
                health["contract_reviews"] += 1
                if result.decision == 'AVOID':
                    health["guardrail_rejections"] += 1
                # Prefer tradable risk-checked near-ATM strikes over a single unquoted OTM strike.
                rank = {"CONSIDER":2,"WATCH":1,"AVOID":0}[result.decision]
                choices.append((rank, -result.spread_pct, -abs(strike-spot)/spot, oi,
                                strike, data, bid, ask, oi, iv, result))
            if not choices:
                continue
            best = max(choices, key=lambda a:a[:4])
            _,_,_,_,strike,data,bid,ask,oi,iv,result = best
            decision, why = result.decision, result.reason
            history = feedback.get(side, {})
            if (decision == "CONSIDER" and history.get("n",0) >= 12 and
                    history["mean"] < -10 and history["profitable"] < 0.45):
                decision = "WATCH"
                why = (f"Historical live-quote {side} paper ideas have averaged {history['mean']:+.1f}% "
                       f"over {history['n']} finished samples; require better validation")
            health[decision.lower()] = health.get(decision.lower(), 0) + 1
            score = (100 if decision=="CONSIDER" else 50 if decision=="WATCH" else 0) + abs(change5)-result.spread_pct/3
            expiry_dt = datetime.strptime(expiry_label, '%d-%b-%Y')
            reviews.append({"Stock":name,"Type":side,"Direction":"Up" if side=="CE" else "Down",
                "Decision":decision,"Why":why,"Expiry":expiry_dt.strftime('%d %b %Y'),
                "Days left":days_left,"Strike (₹)":strike,"Share price (₹)":round(spot,2),
                "Buy premium (₹)":ask,"Sell premium (₹)":bid,"Lot":lot,"Cost (₹)":result.cost_inr,
                "Maximum premium loss (₹)":result.max_loss_inr,
                "Breakeven at expiry (₹)":result.breakeven_spot,
                "Spread (%)":result.spread_pct,"Open interest":int(oi),"IV (%)":iv,
                "Historical volatility (%)":round(rv,1),"5-day share move (%)":round(change5,2),
                "RSI":round(rsi,1),
                "Score":round(score,2),"Fresh quote":quote_fresh,
                "Exchange quote at":exchange_timestamp(chain).strftime('%Y-%m-%d %H:%M:%S'),
                "Checked at":now.strftime('%Y-%m-%d %H:%M:%S')})
    df = pd.DataFrame(reviews)
    if not df.empty:
        df = df.sort_values(["Score","Open interest"],ascending=[False,False]).reset_index(drop=True)
        with DB_LOCK:
            con=duckdb.connect(DB_PATH,read_only=False)
            try:
                for _,r in df.iterrows():
                    con.execute("""INSERT OR REPLACE INTO option_watchlist
                       VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",[
                        now.date().isoformat(),r["Stock"],r["Type"],r["Expiry"],r["Strike (₹)"],
                        r["Share price (₹)"],r["Sell premium (₹)"],r["Buy premium (₹)"],
                        r["Lot"],r["Cost (₹)"],r["Decision"],r["Why"],r["Breakeven at expiry (₹)"],
                        r["Spread (%)"],r["IV (%)"],r["Historical volatility (%)"],
                        r["5-day share move (%)"],r["Checked at"]])
            finally:
                con.close()
    st.session_state["option_scan_cursor"] = (cursor + health["underlyings_attempted"]) % len(syms)
    st.session_state["option_scan_health"] = health
    st.session_state["option_scan_note"] = option_scan_summary(health)
    return df


def generate_daily_options_alpha() -> dict:
    """One highest-scored *paper* CE or PE per market day; no broker integration."""
    date_key = ist_now().strftime('%Y-%m-%d')
    with DB_LOCK:
        con=duckdb.connect(DB_PATH,read_only=True)
        try:
            old=con.execute("SELECT * FROM daily_options_journal WHERE date_key=?",[date_key]).df()
            if not old.empty:
                saved_option = old.iloc[0].to_dict()
                # Database fetch is completed before accessing the cloud below.
            else:
                saved_option = None
        finally:
            con.close()
    if saved_option is not None:
        persist_daily_option_to_supabase(saved_option)
        return saved_option
    if not get_history_integrity()["allow_new_paper_entries"]:
        st.session_state["opt_note"] = "New paper option ideas are paused until the missing share history is restored."
        return {}
    reviews = st.session_state.get("option_candidates")
    if not isinstance(reviews,pd.DataFrame) or reviews.empty or not is_entry_window_open():
        return {}
    choices = reviews[(reviews["Decision"]=="CONSIDER") & (reviews["Fresh quote"]==True)]
    if choices.empty:
        return {}
    best=choices.iloc[0]
    if any(field not in best.index for field in ("RSI", "Checked at", "Historical volatility (%)", "5-day share move (%)")):
        st.session_state["opt_note"] = "Pre-upgrade CE/PE shortlist lacks entry features; rescan before new paper entry."
        return {}
    now=ist_now()
    try:
        last_checked = datetime.strptime(str(best["Checked at"]), '%Y-%m-%d %H:%M:%S')
        seconds_old = (now.replace(tzinfo=None)-last_checked).total_seconds()
    except (ValueError, TypeError, KeyError):
        seconds_old = 99999
    if not -60 <= seconds_old <= 300:
        st.session_state["opt_note"] = "An old CE/PE shortlist cannot create a new paper position; rescan quotes."
        return {}
    expiry = datetime.strptime(best["Expiry"], '%d %b %Y')
    # A previous 15-minute scan is NEVER an executable entry quote. Re-fetch the
    # exact expiry, side and strike and re-apply conservative risk rules.
    quote = get_nse_option_quote(str(best["Stock"]), expiry, float(best["Strike (₹)"]), str(best["Type"]))
    if not quote:
        st.session_state["opt_note"] = "Fresh quote for shortlisted CE/PE contract unavailable; paper entry skipped."
        return {}
    old_ask = float(best["Buy premium (₹)"])
    if old_ask <= 0 or abs(quote["ask"]-old_ask)/old_ask > 0.15:
        st.session_state["opt_note"] = "Option premium changed more than 15% since the scan; rescan before paper entry."
        return {}
    eq_used,opt_used=get_open_exposures()
    refreshed_review = review_long_option(
        kind=str(best["Type"]), spot=float(best["Share price (₹)"]),
        strike=float(best["Strike (₹)"]), bid=quote["bid"], ask=quote["ask"],
        lot_size=int(best["Lot"]), days_left=(expiry.date()-now.date()).days,
        open_interest=quote["oi"], iv_pct=quote["iv"],
        realized_vol_pct=float(best["Historical volatility (%)"]),
        ret_5d_pct=float(best["5-day share move (%)"]), rsi=float(best["RSI"]),
        recent_filing_risk=evaluate_nse_filing_nlp(str(best["Stock"])).get("is_blocked", False),
        equity_exposure=eq_used, option_exposure=opt_used, quote_fresh=True)
    if refreshed_review.decision != "CONSIDER" or not can_allocate_option(refreshed_review.cost_inr,eq_used,opt_used):
        st.session_state["opt_note"] = "Fresh option quote failed risk/liquidity review; paper entry skipped."
        return {}
    bid, ask = quote["bid"], quote["ask"]
    sig={"date_key":date_key,"timestamp":now.strftime('%Y-%m-%d %H:%M:%S'),
        "share_name":best["Stock"],"option_contract":f"{best['Stock']} {expiry.strftime('%d %b %Y').upper()} {best['Strike (₹)']:g} {best['Type']} (live quote)",
        "strike_price":float(best["Strike (₹)"]),"expiry_days":(expiry.date()-now.date()).days,
        "underlying_spot":float(best["Share price (₹)"]),"lot_size":int(best["Lot"]),
        "entry_premium":ask,"current_option_price":bid,
        "target_premium":round(ask*OPTION_TARGET_MULT,2),
        "stop_loss_premium":round(ask*OPTION_STOP_MULT,2),
        "total_capital":refreshed_review.cost_inr,"ai_confidence":round(min(100.0,float(best["Score"])),1),
        "implied_vol":quote["iv"],"status":"ACTIVE",
        "pnl_pct":round(100*(bid/ask-1),2),
        "last_audited":now.strftime('%Y-%m-%d %H:%M:%S')}
    with DB_LOCK:
        con=duckdb.connect(DB_PATH,read_only=False)
        try:
            # avoid a concurrent duplicate paper entry; database enforces date_key uniqueness
            con.execute("""INSERT INTO daily_options_journal
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,'ACTIVE',?,?) ON CONFLICT(date_key) DO NOTHING""",
                [sig[k] for k in ("date_key","timestamp","share_name","option_contract","strike_price",
                 "expiry_days","underlying_spot","lot_size","entry_premium","current_option_price",
                 "target_premium","stop_loss_premium","total_capital","ai_confidence","implied_vol",
                 "pnl_pct","last_audited")])
        finally:
            con.close()
    persist_daily_option_to_supabase(sig)
    return sig

def is_live_option(contract) -> bool:
    return "(live quote)" in str(contract)

def audit_option(opt, now_str: str, today_date):
    """Live-quote options are re-quoted on NSE; model-price options are re-priced with the SAME volatility they started with."""
    name = str(opt["share_name"])
    strike = float(opt["strike_price"])
    entry_prem = float(opt["entry_premium"])
    target_prem = float(opt["target_premium"])
    stop_prem = float(opt["stop_loss_premium"])
    sigma = float(opt["implied_vol"]) / 100.0
    opt_dt = _to_date(opt["date_key"]) or today_date
    expiry_dt = datetime.combine(opt_dt, dtime(15, 30)) + timedelta(days=int(opt["expiry_days"]))
    elapsed = max(0, (today_date - opt_dt).days)
    last_price = float(opt["current_option_price"])
    live_src = is_live_option(opt["option_contract"])
    kind = "PE" if re.search(r"\bPE\b", str(opt["option_contract"])) else "CE"

    h = fetch_cached_history(f"{name}.NS", period="1mo", interval="1d")
    if h.empty:
        st.session_state["price_feed_problem"] = True
        return
    dates = _bar_dates(h)
    start_after = max(opt_dt, _to_date(opt.get("last_audited")) or opt_dt)
    status, live = "ACTIVE", last_price

    def prem_at(spot, d, side="bid"):
        rem = max(1, (expiry_dt.date() - d).days)
        return model_option_price(spot, strike, rem, sigma, side)

    if live_src:
        if is_nse_market_open():
            q = get_nse_option_quote(name, expiry_dt, strike, kind)
            if q:
                live = q["bid"]
                if live >= target_prem:
                    status = "🎯 WIN (TARGET HIT)"
                elif live <= stop_prem:
                    status = "🛑 LOSS (STOPPED OUT)"
    else:
        for i, d in enumerate(dates):
            if d <= start_after or d >= today_date:
                continue
            row = h.iloc[i]
            if prem_at(float(row["Low"]), d) <= stop_prem:
                status, live = "🛑 LOSS (STOPPED OUT)", stop_prem
                break
            if prem_at(float(row["High"]), d) >= target_prem:
                status, live = "🎯 WIN (TARGET HIT)", target_prem
                break
        if status == "ACTIVE":
            spot = float(h["Close"].iloc[-1])
            live = prem_at(spot, today_date)
            if live >= target_prem:
                status, live = "🎯 WIN (TARGET HIT)", target_prem
            elif live <= stop_prem:
                status, live = "🛑 LOSS (STOPPED OUT)", stop_prem

    if status == "ACTIVE" and today_date >= expiry_dt.date():
        status = "⏱️ EXPIRED (TIME EXIT)"
        live = max(0.0, (float(h["Close"].iloc[-1]) - strike) if kind == "CE"
                   else strike - float(h["Close"].iloc[-1]))  # expiry intrinsic, not a broker fill
    elif status == "ACTIVE" and elapsed >= MAX_HOLD_CALENDAR_DAYS:
        status = "⏱️ EXPIRED (TIME EXIT)"

    pnl_pct = round(((live - entry_prem) / entry_prem) * 100.0, 2) if entry_prem > 0 else 0.0
    spot_now = float(h["Close"].iloc[-1])
    with DB_LOCK:
        con = duckdb.connect(DB_PATH, read_only=False)
        try:
            con.execute("""
                UPDATE daily_options_journal
                SET current_option_price = ?, underlying_spot = ?, pnl_pct = ?, status = ?, last_audited = ?
                WHERE date_key = ?
            """, [live, spot_now, pnl_pct, status, now_str, opt["date_key"]])
        finally:
            con.close()
    cloud_update("options_journal", {"date_key": opt["date_key"]},
                 {"current_option_price": live, "underlying_spot": spot_now, "pnl_pct": pnl_pct, "status": status, "last_audited": now_str},
                 "Audit Option")


# ---------- the OPTIONS learner: simple, visible rules that adapt to its own option results ----------
def load_options_journal_df() -> pd.DataFrame:
    with DB_LOCK:
        con = duckdb.connect(DB_PATH, read_only=True)
        try:
            return con.execute("SELECT * FROM daily_options_journal ORDER BY date_key ASC").df()
        except Exception:
            return pd.DataFrame()
        finally:
            con.close()

def _opt_group(g: pd.DataFrame):
    if g is None or g.empty:
        return None
    return {"n": int(len(g)), "avg": round(float(g["pnl_pct"].mean()), 1),
            "win_rate": round(float((g["outcome"] == "WIN").mean() * 100.0), 0),
            "wavg": round(float((g["w"] * g["pnl_pct"]).sum() / max(1e-9, g["w"].sum())), 1)}

def options_learning(df_all: pd.DataFrame = None) -> dict:
    """Learns ONLY from finished option ideas and ONLY changes the options idea:
       1) a share whose option lost recently takes a break from options (fades like the share memory: half every 30 days),
       2) after enough results, if recent ideas lose on average, it only takes shares with a stronger 5-day climb,
       3) if estimated-price ideas keep losing, it only takes ideas with a real NSE quote.
       Rules 2 and 3 look at the last 60 days only, so if a rule stops all ideas it lifts by itself and gets re-tested."""
    out = {"n": 0, "recent_n": 0, "ready": False, "bar": 60.0, "rest": {}, "live_only": False, "wavg": None,
           "live": None, "model": None, "by_share": {}}
    if df_all is None:
        df_all = load_options_journal_df()
    d = prep_perf_frame(df_all, "total_capital", is_option=True)
    if d is None or d.empty:
        return out
    today = ist_now().date()
    d = d.copy()
    d["age"] = d["exit_date"].apply(lambda x: max(0, (today - x).days) if x else 0)
    d["w"] = 0.5 ** (d["age"] / HALF_LIFE_DAYS)
    d["live_src"] = d["option_contract"].apply(is_live_option)
    out["n"] = int(len(d))
    out["live"], out["model"] = _opt_group(d[d["live_src"]]), _opt_group(d[~d["live_src"]])
    for share, g in d.groupby("share_name"):
        out["by_share"][str(share)] = _opt_group(g)
        sc = 0.0
        for _, r in g.iterrows():
            o = r["outcome"]
            v = 1.0 if o == "WIN" else (-1.0 if o == "LOSS" else (-0.5 if float(r["pnl_pct"]) < 0 else 0.0))
            sc += float(r["w"]) * v
        if sc <= -OPT_REST_THRESHOLD:
            out["rest"][str(share)] = max(1, int(math.ceil(HALF_LIFE_DAYS * math.log2(abs(sc) / OPT_REST_THRESHOLD))))
    recent = d[d["age"] <= OPT_LEARN_WINDOW_DAYS]
    out["recent_n"] = int(len(recent))
    if len(recent) >= OPT_LEARN_MIN_TRADES:
        out["ready"] = True
        wavg = float((recent["w"] * recent["pnl_pct"]).sum() / max(1e-9, recent["w"].sum()))
        out["wavg"] = round(wavg, 1)
        if wavg < 0:
            out["bar"] = round(min(OPT_MAX_TREND_BAR, 60.0 + abs(wavg) * 0.5), 1)
    rm, rl = _opt_group(recent[~recent["live_src"]]), _opt_group(recent[recent["live_src"]])
    if rm and rm["n"] >= OPT_LIVE_ONLY_MIN_MODEL and rm["wavg"] < 0 and (rl is None or rl["n"] < 5 or rl["wavg"] > rm["wavg"]):
        out["live_only"] = True
    return out

def trend_bar_to_rise(bar: float) -> float:
    """Trend strength = 70 + 5-day rise(%) × 1.5  ->  the 5-day rise (%) a share needs to reach a given bar."""
    return (float(bar) - 70.0) / 1.5

NO_DATA_GRACE_DAYS = 3

# Missing market prices must not generate a simulated sale at the last known
# quote: this would release paper capital without observing an actual exit.
# Keep the trade ACTIVE and show a data-quality warning until reliable data
# arrives or a separate reconciliation process can verify its outcome.

# ==============================================================================
# 9. AUDIT ENGINE (BAR-BY-BAR, GAP-AWARE, STOP-FIRST, HONEST FILLS)
# ==============================================================================
def audit_and_reconcile_all_trades():
    now_ist = ist_now()
    now_str = now_ist.strftime('%Y-%m-%d %H:%M:%S')
    today_date = now_ist.date()
    st.session_state["price_feed_problem"] = False
    st.session_state["intraday_feed_problem"] = False
    st.session_state["intraday_feed_missing_tickers"] = []
    flush_pending_cloud_sync()

    with DB_LOCK:
        con = duckdb.connect(DB_PATH, read_only=False)
        try:
            active_trades = con.execute("SELECT * FROM trade_journal WHERE status = 'ACTIVE'").df()
            active_opts = con.execute("SELECT * FROM daily_options_journal WHERE status = 'ACTIVE'").df()
        except Exception:
            active_trades, active_opts = pd.DataFrame(), pd.DataFrame()
        finally:
            con.close()

    got_any_price = False
    for _, tr in (active_trades.iterrows() if not active_trades.empty else []):
        tkr = str(tr["ticker"]).strip()
        try:
            h = fetch_cached_history(normalize_ticker_for_yf(tkr), period="1mo", interval="1d")
            if h.empty:
                st.session_state["price_feed_problem"] = True
                st.session_state["intraday_feed_missing_tickers"].append(tkr)
                continue  # never auto-close a position using an old price
            got_any_price = True
            dates = _bar_dates(h)
            entry, target, stop = float(tr["entry_price"]), float(tr["target_price"]), float(tr["stop_loss"])
            friction = get_trade_friction_pct(tr.get("features_json"))
            be_floor = get_breakeven_exit_price(entry, friction)
            target_dist = target - entry
            be_trigger = entry + BREAKEVEN_TRIGGER_RATIO * target_dist
            entry_dt = _to_date(tr["date_str"])
            if entry_dt is None:
                continue
            start_after = max(entry_dt, _to_date(tr.get("last_audited")) or entry_dt)

            new_status, exit_price, exit_ts = "ACTIVE", 0.0, None
            features = _parse_features(tr.get('features_json'))
            curr = float(h["Close"].iloc[-1])

            # 1) completed days we haven't looked at yet (covers days the app was asleep)
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
                    new_status, exit_price = classify_stop_exit(entry, stop, fill, friction), fill
                else:
                    stop, curr = new_stop, c
                if outcome:
                    exit_ts = f"{d} 15:30:00"
                    break

            # 2) today's intraday minute path: a brief target or stop touch must
            # not be lost just because the market returned inside the range by
            # the next 15-minute dashboard refresh. The *first observed* event
            # wins; same-minute double touches use the conservative stop-first
            # rule. Missing minute candles => do NOT claim a verified exit.
            intraday_uncertain = False
            if new_status == "ACTIVE" and dates and dates[-1] == today_date:
                row_now = h.iloc[-1]
                curr = float(row_now["Close"])
                try:
                    entry_ts = pd.Timestamp(str(tr.get("timestamp") or "").replace("T", " "))
                    if pd.isna(entry_ts):
                        raise ValueError('missing entry timestamp')
                    audit_ts = pd.Timestamp(str(tr.get("last_audited") or "").replace("T", " "))
                    if pd.isna(audit_ts):
                        audit_ts = entry_ts
                    start_today = pd.Timestamp(f"{today_date} 09:15:00")
                    features = _parse_features(tr.get("features_json"))
                    base_stop = features.get('initial_stop_price')
                    if entry_dt == today_date:
                        # For NEW trades replay from exact entry time using the
                        # original stop (previous audits may have ratcheted it).
                        # For restored/legacy trades with unknown original stop,
                        # only inspect since last audit to prevent false exits.
                        if base_stop is not None and _num(base_stop) > 0:
                            from_ts = entry_ts
                            minute_stop = float(base_stop)
                        else:
                            from_ts = max(entry_ts, audit_ts)
                            minute_stop = stop
                    else:
                        from_ts = max(start_today, audit_ts) if audit_ts.date() == today_date else start_today
                        minute_stop = stop
                    minute_data = fetch_cached_history(normalize_ticker_for_yf(tkr), period='1d', interval='1m')
                    check = scan_intraday_minutes(minute_data, entry=entry, target=target,
                        stop=minute_stop, from_time=from_ts, as_of=now_ist,
                        be_floor=be_floor, be_trigger=be_trigger)
                    if check.covered:
                        stop = check.new_stop
                        if check.outcome == 'WIN':
                            new_status, exit_price, exit_ts = '🎯 WIN (TARGET HIT)', check.fill, check.touched_at
                        elif check.outcome == 'STOP':
                            new_status, exit_price, exit_ts = classify_stop_exit(entry,stop,check.fill,friction), check.fill, check.touched_at
                        if check.outcome:
                            features['intraday_exit_source'] = 'Yahoo 1-minute OHLC (paper estimate)'
                            features['intraday_exit_note'] = check.note
                            features['intraday_same_minute_ambiguous'] = bool(check.ambiguous)
                    else:
                        intraday_uncertain = True
                        st.session_state["intraday_feed_problem"] = True
                        st.session_state["intraday_feed_missing_tickers"].append(tkr)
                except Exception as exc:
                    intraday_uncertain = True
                    st.session_state["intraday_feed_problem"] = True
                    st.session_state["intraday_feed_missing_tickers"].append(tkr)
                    logging.warning('Intraday paper reconciliation unavailable for %s: %s',tkr,type(exc).__name__)

            # The partial daily candle's high/low includes pre-entry values and
            # cannot establish which barrier came first. Never turn that into a
            # guaranteed intraday exit if the minute path was unavailable.

            # 3) time exit
            # Never invent an exit at a stale closing price merely because
            # the calendar timer expired; wait for a real session price.
            if (new_status == "ACTIVE" and (today_date - entry_dt).days >= MAX_HOLD_CALENDAR_DAYS
                    and dates and dates[-1] == today_date and not intraday_uncertain):
                new_status, exit_price = "⏱️ EXPIRED (TIME EXIT)", curr

            mark = exit_price if new_status != "ACTIVE" else curr
            pnl = calc_net_equity_pnl_pct(entry, mark, friction)
            # Keep the old intraday checkpoint if today's feed failed, so the
            # next successful check can still recover a temporary price touch.
            audit_watermark = (str(tr.get('last_audited') or now_str)
                               if intraday_uncertain and new_status == 'ACTIVE' else now_str)
            updated_features = json.dumps(features)
            with DB_LOCK:
                con = duckdb.connect(DB_PATH, read_only=False)
                try:
                    con.execute("""
                        UPDATE trade_journal
                        SET latest_price = ?, stop_loss = ?, pnl_pct = ?, status = ?, exit_price = ?,
                            exit_timestamp = CASE WHEN ? != 'ACTIVE' THEN ? ELSE exit_timestamp END,
                            last_audited = ?, features_json = ?
                        WHERE trade_id = ?
                    """, [mark, stop, pnl, new_status, exit_price, new_status, exit_ts or now_str,
                           audit_watermark, updated_features, tr["trade_id"]])
                finally:
                    con.close()
            cloud_update("predictions", {"ticker": tkr, "predicted_date": tr["date_str"]},
                         {"status": new_status, "stop_loss": stop, "latest_price": mark, "pnl_pct": pnl,
                          "last_checked": audit_watermark, "features_json": json.loads(updated_features)}, "Audit Equity")
        except Exception:
            continue

    for _, opt in (active_opts.iterrows() if not active_opts.empty else []):
        try:
            audit_option(opt, now_str, today_date)
            got_any_price = True
        except Exception:
            continue

    if got_any_price or (active_trades.empty and active_opts.empty):
        st.session_state["last_price_check"] = now_ist.strftime("%d %b %H:%M")

# ==============================================================================
# 10. LOGGING & GUARDRAILS
# ==============================================================================
def get_active_positions_df() -> pd.DataFrame:
    with DB_LOCK:
        con = duckdb.connect(DB_PATH, read_only=True)
        try:
            return con.execute("SELECT ticker, entry_price, latest_price, target_price, stop_loss, capital_allocated, pnl_pct, status, features_json FROM trade_journal WHERE status = 'ACTIVE'").df()
        except Exception:
            return pd.DataFrame()
        finally:
            con.close()

def get_currently_active_tickers() -> set:
    df = get_active_positions_df()
    return set(clean_sym_name(t) for t in df["ticker"].tolist()) if not df.empty else set()

def get_active_sector_exposure() -> dict:
    counts = {}
    df = get_active_positions_df()
    for _, r in (df.iterrows() if not df.empty else []):
        sec = get_trade_sector(r["ticker"], r["features_json"])
        counts[sec] = counts.get(sec, 0) + 1
    return counts

def get_open_exposures() -> tuple:
    """Paper capital allocated to active stock and option ideas (combined INR cap)."""
    with DB_LOCK:
        con = duckdb.connect(DB_PATH, read_only=True)
        try:
            eq = _num(con.execute("SELECT COALESCE(SUM(capital_allocated),0) FROM trade_journal WHERE status='ACTIVE'").fetchone()[0])
            opt = _num(con.execute("SELECT COALESCE(SUM(total_capital),0) FROM daily_options_journal WHERE status='ACTIVE'").fetchone()[0])
            return eq, opt
        finally:
            con.close()

def get_portfolio_capacity() -> dict:
    df = get_active_positions_df()
    risk = 0.0
    for _, r in (df.iterrows() if not df.empty else []):
        e, stop, cap = _num(r["entry_price"]), _num(r["stop_loss"]), _num(r["capital_allocated"])
        if e > stop > 0:
            risk += cap * (e - stop) / e
    eq, opt = get_open_exposures()
    remaining = remaining_capital(eq, opt)
    open_count = len(df)
    legacy_open_excess = max(0, open_count - MAX_OPEN_POSITIONS)
    fresh_slot_limit_reached = open_count >= MAX_OPEN_POSITIONS
    fresh_slots_left = max(0, MAX_OPEN_POSITIONS - min(open_count, MAX_OPEN_POSITIONS))
    with DB_LOCK:
        con = duckdb.connect(DB_PATH, read_only=True)
        try:
            unpriced = int(con.execute("""SELECT COUNT(*) FROM trade_journal WHERE status='ACTIVE'
                AND (capital_allocated IS NULL OR capital_allocated<=0 OR entry_price IS NULL OR entry_price<=0)""").fetchone()[0])
            unpriced += int(con.execute("""SELECT COUNT(*) FROM daily_options_journal WHERE status='ACTIVE'
                AND (total_capital IS NULL OR total_capital<=0)""").fetchone()[0])
        finally:
            con.close()
    if unpriced:
        reason = f"{unpriced} open paper position(s) still have missing allocations; review History before adding more"
    elif fresh_slot_limit_reached:
        if legacy_open_excess > 0:
            reason = (f"{open_count} active share records are loaded, {legacy_open_excess} above the current fresh-entry limit. "
                      f"Fresh research ideas stay capped at {MAX_OPEN_POSITIONS} until some active positions are reconciled or closed.")
        else:
            reason = f"Fresh share ideas are capped at {MAX_OPEN_POSITIONS} active positions"
    elif risk >= MAX_OPEN_RISK_INR:
        reason = f"Open equity stop-risk limit {inr(MAX_OPEN_RISK_INR)} reached"
    elif remaining <= 0:
        reason = "Combined ₹10 lakh allocation already used"
    else:
        reason = ""
    return {
        "open": open_count,
        "risk_inr": round(risk,2),
        "equity_inr": round(eq,2),
        "option_inr":round(opt,2),
        "remaining_inr":round(remaining,2),
        "ok": not reason,
        "reason": reason,
        "fresh_slots_left": fresh_slots_left,
        "fresh_slot_limit_reached": fresh_slot_limit_reached,
        "legacy_open_excess": legacy_open_excess,
        "effective_limit": MAX_OPEN_POSITIONS,
    }

def get_todays_logged_equities() -> pd.DataFrame:
    today_str = ist_now().strftime('%Y-%m-%d')
    with DB_LOCK:
        con = duckdb.connect(DB_PATH, read_only=True)
        try:
            return con.execute("SELECT * FROM trade_journal WHERE date_str = ? ORDER BY timestamp ASC, ticker ASC", [today_str]).df()
        except Exception:
            return pd.DataFrame()
        finally:
            con.close()

def get_recent_cooldown_tickers() -> set:
    today_dt = ist_now().date()
    cooldown = set()
    with DB_LOCK:
        con = duckdb.connect(DB_PATH, read_only=True)
        try:
            eq_closed = con.execute("SELECT ticker, COALESCE(exit_timestamp, last_audited, date_str) AS close_dt FROM trade_journal WHERE status != 'ACTIVE'").df()
            opt_closed = con.execute("SELECT share_name AS ticker, COALESCE(last_audited, date_key) AS close_dt FROM daily_options_journal WHERE status != 'ACTIVE'").df()
            for _, r in pd.concat([eq_closed, opt_closed], ignore_index=True).iterrows():
                c_dt = _to_date(r["close_dt"])
                if c_dt and 0 <= (today_dt - c_dt).days <= COOLDOWN_CALENDAR_DAYS:
                    cooldown.add(clean_sym_name(r["ticker"]))
        except Exception:
            pass
        finally:
            con.close()
    return cooldown

def log_equity_signal_safely(sig: dict, enforce_sector_cap: bool = True) -> bool:
    now = ist_now()
    today_str = now.strftime('%Y-%m-%d')
    now_str = now.strftime('%Y-%m-%d %H:%M:%S')
    ticker = clean_sym_name(sig.get('Ticker', ''))

    def fail(msg):
        st.session_state["last_log_reason"] = msg
        return False

    if not ticker:
        return fail("Invalid stock")
    if not get_history_integrity()["allow_new_paper_entries"]:
        return fail("Historical share records have not been reconciled. New paper entries are paused to avoid exceeding the true portfolio allocation.")
    if not bool(sig.get("Qualified", False)):
        return fail("Unqualified signal cannot be recorded as a research buy")
    if not is_entry_window_open():
        return fail("It is outside buying hours (09:30–15:00 IST, Mon–Fri)")
    if not sig.get("Loggable", True):
        return fail("No live price from today yet (holiday or data delay)")
    if is_market_stress(get_indian_cross_asset_macro()):
        return fail("Buying is paused because the market is under stress (Nifty falling or India VIX high)")
    cap_state = get_portfolio_capacity()
    if not cap_state["ok"]:
        return fail(cap_state["reason"])
    sector = str(sig.get('Sector') or get_ticker_sector(ticker))
    if enforce_sector_cap and get_active_sector_exposure().get(sector, 0) >= sector_cap(sector):
        return fail(f"You already hold the maximum in {sector}")

    f_dict = dict(sig.get('FeaturesDict') or {})
    f_dict.setdefault('initial_stop_price', float(sig['Stop Loss (₹)']))
    f_json_str = json.dumps(f_dict)
    init_net_pnl = -get_trade_friction_pct(f_json_str)
    logged_local = False
    with DB_LOCK:
        con = duckdb.connect(DB_PATH, read_only=False)
        try:
            cnt = con.execute("SELECT COUNT(*) AS cnt FROM trade_journal WHERE date_str = ?", [today_str]).df()
            if not cnt.empty and int(cnt["cnt"].iloc[0]) >= MAX_DAILY_EQUITY_TRADES:
                return fail("All of today's picks are already used")
            existing = con.execute("SELECT trade_id FROM trade_journal WHERE ticker = ? AND (status = 'ACTIVE' OR date_str = ?)", [ticker, today_str]).df()
            if not existing.empty:
                return fail("You already hold this stock")
            shares_num = int(str(sig['Recommended Shares']).split()[0])
            entry, stop = float(sig['Price (₹)']), float(sig['Stop Loss (₹)'])
            equity_used = _num(con.execute("SELECT COALESCE(SUM(capital_allocated),0) FROM trade_journal WHERE status='ACTIVE'").fetchone()[0])
            options_used = _num(con.execute("SELECT COALESCE(SUM(total_capital),0) FROM daily_options_journal WHERE status='ACTIVE'").fetchone()[0])
            risk_used = _num(con.execute("""SELECT COALESCE(SUM(capital_allocated*(entry_price-stop_loss)/entry_price),0)
                FROM trade_journal WHERE status='ACTIVE' AND entry_price > stop_loss AND entry_price > 0""").fetchone()[0])
            allowed = allowable_equity_shares(entry, stop, equity_used, options_used, risk_used)
            if shares_num <= 0 or shares_num > allowed:
                return fail("Combined capital, single position or risk allowance prevents this paper trade")
            con.execute("""
                INSERT INTO trade_journal 
                (trade_id, timestamp, date_str, ticker, asset_type, entry_price, target_price,
                 stop_loss, shares, capital_allocated, status, latest_price, pnl_pct, exit_price,
                 exit_timestamp, last_audited, features_json)
                VALUES (?, ?, ?, ?, 'EQUITY', ?, ?, ?, ?, ?, 'ACTIVE', ?, ?, 0.0, NULL, ?, ?)
            """, [f"{ticker}_{today_str}", now_str, today_str, ticker, float(sig['Price (₹)']), float(sig['Target (₹)']), float(sig['Stop Loss (₹)']),
                  shares_num, round(shares_num * entry, 2), entry, init_net_pnl, now_str, f_json_str])
            logged_local = True
        except Exception as e:
            record_db_error("Log trade", e)
        finally:
            con.close()

    if supabase and logged_local:
        try:
            cloud_exist = supabase.table("predictions").select("id").eq("ticker", ticker).eq("predicted_date", today_str).execute()
            if not cloud_exist.data:
                supabase.table("predictions").insert({
                    "predicted_date": today_str, "ticker": ticker, "company_name": f"{ticker} ({sector})",
                    "entry_price": float(sig['Price (₹)']), "target_price": float(sig['Target (₹)']), "stop_loss": float(sig['Stop Loss (₹)']),
                    "position_gbp": float(sig['RawCapital']), "shares_qty": int(str(sig['Recommended Shares']).split()[0]),
                    "profit_goal": float(sig['ReturnNum']), "confidence": int(float(str(sig['Adjusted Score']))), "hold_days": 5,
                    "status": "ACTIVE", "latest_price": float(sig['Price (₹)']), "pnl_pct": init_net_pnl,
                    "news_status": sig.get("FilingStatus", "Clean"), "rns_headline": str(sig.get("FilingHeadline", "Active AI Quant Signal"))[:120],
                    "features_json": f_dict, "last_checked": now_str
                }).execute()
        except Exception as e:
            record_db_error("Insert Equity", e)
    return logged_local

# ==============================================================================
# 11. FEATURES & THE SCAN
# ==============================================================================
def calculate_technical_features(df_hist: pd.DataFrame, deliv_shock: float = 1.0) -> pd.DataFrame:
    """Built from COMPLETED daily bars only."""
    if df_hist.empty or len(df_hist) < 20:
        return pd.DataFrame()
    df = pd.DataFrame()
    df['close'] = df_hist['Close'].astype(float).replace(0.0, np.nan)
    df['volume'] = df_hist['Volume'].astype(float)
    df = df.dropna(subset=['close'])
    if len(df) < 20:
        return pd.DataFrame()
    df['ema20'] = df['close'].ewm(span=20, adjust=False).mean().replace(0.0, np.nan)
    df['ema50'] = df['close'].ewm(span=50, adjust=False).mean().replace(0.0, np.nan)
    high = df_hist['High'].astype(float)
    low = df_hist['Low'].astype(float)
    prev_close = df['close'].shift()
    tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    df['atr_14'] = tr.rolling(14).mean()
    df['dist_ema20_pct'] = ((df['close'] - df['ema20']) / df['ema20']) * 100.0
    df['trend_spread_pct'] = ((df['ema20'] - df['ema50']) / df['ema50']) * 100.0
    df['atr_pct'] = (df['atr_14'] / df['close']) * 100.0
    vol_20 = df['volume'].rolling(20).mean().replace(0.0, np.nan)
    df['rvol'] = (df['volume'] / vol_20).fillna(1.0).clip(lower=0.5, upper=5.0)
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0.0).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(window=14).mean()
    rs = gain / loss.replace(0.0, np.nan)
    df['rsi_14'] = 100.0 - (100.0 / (1.0 + rs.fillna(1.0)))
    df['deliv_shock'] = float(deliv_shock)
    return df.replace([np.inf, -np.inf], np.nan).dropna()

def analyse_share(clean_sym: str, df_hist: pd.DataFrame, today_date, market_open: bool):
    """Cheap checks every share gets. Returns (code, info); code is one of no_data / illiquid / wild / ok."""
    if df_hist is None or df_hist.empty:
        return "no_data", None
    dates = _bar_dates(df_hist)
    live_bar_today = (dates[-1] == today_date)              # also catches NSE holidays
    partial_bar = live_bar_today and market_open
    live_price = float(df_hist["Close"].iloc[-1])
    df_model = df_hist.iloc[:-1] if partial_bar else df_hist
    if len(df_model) < 30 or not np.isfinite(live_price) or live_price <= 0:
        return "no_data", None
    recent = df_model.tail(20)
    avg_turnover = float((recent["Close"] * recent["Volume"]).mean())
    if (not math.isfinite(avg_turnover)) or live_price < MIN_PRICE_INR or avg_turnover < MIN_AVG_TURNOVER_INR:
        return "illiquid", None
    last10 = df_model.tail(10)
    frozen_days = int(((last10["High"] - last10["Low"]).abs() <= last10["Close"].abs() * 0.0005).sum())
    if frozen_days >= 3:
        return "wild", None                                 # stuck at its price limit (circuit) on several days
    df_feat = calculate_technical_features(df_model, deliv_shock=1.15)
    if df_feat.empty or len(df_feat) < 20:
        return "no_data", None
    latest = df_feat.iloc[-1]
    model_close = float(latest["close"])
    ema20 = float(latest["ema20"])
    atr = max(float(latest.get("atr_14", model_close * 0.02)), model_close * 0.012)
    atr_pct = round((atr / live_price) * 100.0, 2)
    if float(latest["atr_pct"]) > MAX_ATR_PCT:
        return "wild", None
    feat = {
        "dist_ema20_pct": round(float(latest["dist_ema20_pct"]), 2),
        "trend_spread_pct": round(float(latest["trend_spread_pct"]), 2),
        "atr_pct": round(float(latest["atr_pct"]), 2),
        "rvol": round(float(latest["rvol"]), 2),
        "rsi_14": round(float(latest["rsi_14"]), 1),
        "deliv_shock": round(max(1.0, min(2.5, float(latest["rvol"]) * 1.1)), 2)   # proxy: no delivery data feed
    }
    closes = df_model["Close"]
    ret20 = float(closes.iloc[-1] / closes.iloc[-21] - 1.0) * 100.0 if len(closes) > 21 else 0.0
    return "ok", {
        "dates": dates, "live_bar_today": live_bar_today, "live_price": live_price, "avg_turnover": avg_turnover,
        "latest": latest, "model_close": model_close, "ema20": ema20, "atr": atr, "atr_pct": atr_pct, "feat": feat,
        "ret20": round(ret20, 2),
        "is_above_trend": (model_close >= ema20) and (feat["rsi_14"] >= 48.0) and (feat["trend_spread_pct"] >= -1.0),
        "is_exhausted": (feat["rsi_14"] > 78.0) or (feat["dist_ema20_pct"] > 8.5),
        "has_volume": feat["rvol"] >= MIN_RVOL_THRESHOLD,
    }

def _score_share(ml_bundle, feature_cols, calibrate, info):
    raw_feature_map = {c: float(info["feat"].get(c, info["latest"].get(c, 0.0))) for c in feature_cols}
    raw_prob = float(ensemble_proba(ml_bundle, pd.DataFrame([raw_feature_map], columns=feature_cols))[0])
    if not math.isfinite(raw_prob):
        raise ValueError("model returned a non-number")
    raw_prob = max(0.01, min(0.99, raw_prob))
    base_conf = round(min(95.0, max(5.0, calibrate(raw_prob) * 100.0)), 1)
    return raw_feature_map, raw_prob, base_conf

def _nifty_20d_return() -> float:
    try:
        h = fetch_cached_history("^NSEI", period="3mo", interval="1d")
        c = h["Close"]
        return float(c.iloc[-1] / c.iloc[-21] - 1.0) * 100.0 if len(c) > 21 else 0.0
    except Exception:
        return 0.0

def expected_value_pct(p_win: float, entry: float, target: float, stop: float, friction_pct: float) -> float:
    """What a pick is worth on average, in %, after ALL costs: chance × net gain − (1 − chance) × net loss."""
    if entry <= 0:
        return 0.0
    gain = (target - entry) / entry * 100.0 - friction_pct
    loss = (entry - stop) / entry * 100.0 + friction_pct
    return round(p_win * gain - (1.0 - p_win) * loss, 2)

def breakeven_win_pct(entry: float, target: float, stop: float, friction_pct: float) -> float:
    """How often a trade like this must reach its goal (before its safety exit) just to break even after costs."""
    if entry <= 0:
        return 100.0
    gain = (target - entry) / entry * 100.0 - friction_pct
    loss = (entry - stop) / entry * 100.0 + friction_pct
    return round(loss / (gain + loss) * 100.0, 1) if gain > 0 and (gain + loss) > 0 else 100.0

def todays_learning_picks() -> int:
    t = get_todays_logged_equities()
    return int(sum(bool(_parse_features(f).get("learning_pick")) for f in t["features_json"])) if not t.empty and "features_json" in t.columns else 0

def is_learning_trade(f_json) -> bool:
    return bool(_parse_features(f_json).get("learning_pick"))

def run_predictions():
    """Looks at EVERY share in NSE's normal (EQ) series:
       1) download prices for all of them (in parallel; quiet shares are re-checked less often),
       2) cheap filters for everyone: tradable, not stuck at price limits, uptrend, not overbought, normal volume,
       3) AI score for the survivors, 4) the best finalists get a fresh price + the full news/memory/market check."""
    stats = {"listed": 0, "priced": 0, "no_data": 0, "illiquid": 0, "wild": 0, "scanned": 0, "uptrend": 0,
             "not_uptrend": 0, "overbought": 0, "quiet": 0, "held": 0, "low_score": 0, "deep": 0, "qualified": 0,
             "errors": 0, "stale": 0, "logged": 0, "paused": False, "window_open": is_entry_window_open(), "error": "",
             "source": "", "partial": False, "sector_breadth": {}, "leaders": [], "seconds": 0.0}
    t_start = time.time()
    ml_bundle = get_ml_model()
    if ml_bundle is None:
        stats["error"] = "The AI model file wasn't found, so picks are paused (nothing is guessed)."
        st.session_state["scan_stats"] = stats
        return pd.DataFrame(), False
    feature_cols = ml_bundle.get("feature_cols", FEATURE_COLS) if isinstance(ml_bundle, dict) else FEATURE_COLS

    closed_df = load_closed_trades_df()
    check_and_auto_retrain_model(feature_cols, closed_df)
    ml_bundle = get_ml_model() or ml_bundle
    feature_cols = ml_bundle.get("feature_cols", FEATURE_COLS) if isinstance(ml_bundle, dict) else FEATURE_COLS
    calibrate, cal_info = build_calibrator(closed_df)
    st.session_state["cal_info"] = cal_info

    today_date = ist_now().date()
    market_open = is_nse_market_open()
    window_open = is_entry_window_open()
    macro = get_indian_cross_asset_macro()
    paused = is_market_stress(macro)
    stats["paused"] = paused

    symbols, source = get_all_nse_symbols()
    stats["listed"], stats["source"] = len(symbols), source
    get_nse_industry_map()                                   # warm the sector labels once
    active_held = get_currently_active_tickers()
    sector_counts = get_active_sector_exposure()
    cooldown = get_recent_cooldown_tickers()

    box = get_nse_http_box()
    quiet_seen = box.setdefault("quiet_seen", {})          # share -> when it was last seen thinly traded
    ttl_for = lambda ys: QUIET_TTL_SEC if (time.time() - quiet_seen.get(ys, 0.0)) < QUIET_TTL_SEC else SWEEP_TTL_SEC
    yf_syms = [f"{c}.NS" for c in symbols]
    prog = st.progress(0, text=f"Downloading prices for {len(yf_syms):,} NSE shares...")
    def _tick(done, total):
        try:
            prog.progress(min(1.0, done / max(1, total)), text=f"Downloading prices: {done:,} of {total:,} still needed")
        except Exception:
            pass
    complete = prefetch_histories(yf_syms, "6mo", ttl=ttl_for, workers=SCAN_WORKERS, progress=_tick, time_budget=SCAN_TIME_BUDGET_SEC)
    stats["partial"] = not complete

    nifty20 = _nifty_20d_return()
    breadth, candidates, leaders = {}, [], []
    for clean_sym in symbols:
        ys = f"{clean_sym}.NS"
        df_hist = peek_history(ys, "6mo")
        if df_hist.empty:
            stats["no_data"] += 1
            continue
        stats["priced"] += 1
        try:
            code, info = analyse_share(clean_sym, df_hist, today_date, market_open)
        except Exception:
            stats["errors"] += 1
            continue
        if code != "ok":
            stats[code] += 1
            if code == "illiquid":
                quiet_seen[ys] = time.time()
            continue
        quiet_seen.pop(ys, None)
        stats["scanned"] += 1
        sec = get_ticker_sector(clean_sym)
        b = breadth.setdefault(sec, [0, 0])
        b[1] += 1
        if info["is_above_trend"]:
            stats["uptrend"] += 1
            b[0] += 1
            leaders.append((round(info["ret20"] - nifty20, 2), clean_sym, sec))
        if not info["is_above_trend"]:
            stats["not_uptrend"] += 1
            continue
        if info["is_exhausted"]:
            stats["overbought"] += 1
            continue
        if not info["has_volume"]:
            stats["quiet"] += 1
            continue
        if clean_sym in active_held or clean_sym in cooldown or sector_counts.get(sec, 0) >= sector_cap(sec):
            stats["held"] += 1
            continue
        try:
            _, _, base_conf = _score_share(ml_bundle, feature_cols, calibrate, info)
        except Exception:
            stats["errors"] += 1                            # no guessing when the model can't score a share
            continue
        candidates.append((base_conf, clean_sym))
    stats["sector_breadth"] = {k: (v[0], v[1]) for k, v in breadth.items()}
    stats["leaders"] = [(t, sec, rs) for rs, t, sec in sorted(leaders, reverse=True)[:8]]
    stats["nifty_20d"] = round(nifty20, 2)

    candidates.sort(key=lambda x: (-x[0], x[1]))
    stats["candidates"] = len(candidates)
    finalists = [c for _, c in candidates[:MAX_DEEP_CHECKS]]
    stats["low_score"] += max(0, len(candidates) - len(finalists))
    if finalists:
        prog.progress(1.0, text=f"Double-checking the best {len(finalists)} shares with fresh prices...")
        prefetch_histories([f"{c}.NS" for c in finalists], "6mo", ttl=FRESH_PRICE_TTL_SEC, workers=6)

    results = []
    for clean_sym in finalists:
        try:
            df_hist = peek_history(f"{clean_sym}.NS", "6mo", max_age=FRESH_PRICE_MAX_AGE_SEC)
            if df_hist.empty:
                stats["stale"] += 1                         # no fresh price -> never buy on an old one
                continue
            code, info = analyse_share(clean_sym, df_hist, today_date, market_open)
            if code != "ok":
                stats[code] += 1
                continue
            try:
                raw_feature_map, raw_prob, base_conf = _score_share(ml_bundle, feature_cols, calibrate, info)
            except Exception:
                stats["errors"] += 1
                continue
            live_price, feat, atr, atr_pct, dates = info["live_price"], info["feat"], info["atr"], info["atr_pct"], info["dates"]
            avg_turnover = info["avg_turnover"]
            is_above_trend, is_exhausted, has_volume = info["is_above_trend"], info["is_exhausted"], info["has_volume"]

            nlp = evaluate_nse_filing_nlp(clean_sym)
            learner = get_closed_loop_self_learning(clean_sym, atr_pct, closed_df)
            macro_delta, macro_note = compute_nse_macro_lead_lag(clean_sym, macro)
            trend_penalty = 0.0 if is_above_trend else -10.0
            total_adj = round(learner["delta"] + macro_delta + nlp["delta"] + trend_penalty, 1)
            final_score = round(min(98.0, max(5.0, base_conf + total_adj)), 1)

            target = round(live_price + 1.6 * atr, 2)
            stop = round(live_price - 1.1 * atr, 2)
            live_price = round(live_price, 2)                      # the price the trade is recorded at
            cap_state = get_portfolio_capacity()
            shares = allowable_equity_shares(live_price, stop, cap_state["equity_inr"],
                        cap_state["option_inr"], cap_state["risk_inr"])
            capital = round(shares * live_price, 2)                    # so shares x price always equals money in
            slip = slippage_pct_per_side(avg_turnover)
            friction = compute_costs_pct(capital, slip)
            net_return = round(((target - live_price) / live_price) * 100.0 - friction, 2)
            sector = get_ticker_sector(clean_sym)
            rel_strength = round(info["ret20"] - nifty20, 2)

            ev_pct = expected_value_pct(final_score / 100.0, live_price, target, stop, friction)
            be_pct = breakeven_win_pct(live_price, target, stop, friction)
            safe = (is_above_trend and not is_exhausted and has_volume and not nlp["is_blocked"] and net_return >= MIN_NET_RETURN_PCT)
            qualified = safe and ev_pct > MIN_EV_PCT and shares > 0
            if shares <= 0:
                reason = "Insufficient available capital or stop-risk budget"
            elif nlp["is_blocked"]:
                reason = f"Bad news flagged: {nlp['nlp_tags']}"
            elif not is_above_trend:
                reason = "Not in an uptrend any more (fresh price)"
            elif is_exhausted:
                reason = "Already run up too far (overbought)"
            elif not has_volume:
                reason = f"Too little trading today (below {MIN_RVOL_THRESHOLD}× normal volume)"
            elif net_return < MIN_NET_RETURN_PCT:
                reason = f"Possible profit after costs is only {net_return:+.2f}%"
            elif ev_pct <= MIN_EV_PCT:
                reason = f"Not worth it after costs: needs a {be_pct:.0f}% chance of reaching the goal, the AI estimates {final_score:.0f}% (worth {ev_pct:+.2f}% on average)"
            else:
                reason = (f"Available price/model checks passed; filings NOT verified "
                          f"(needs {be_pct:.0f}%, model estimate {final_score:.0f}%)" if not ANNOUNCEMENTS_AVAILABLE
                          else f"Passed configured checks (needs {be_pct:.0f}%, model estimates {final_score:.0f}%)")

            # Save an honest, entry-time target-vs-stop timeline with NEW paper trades.
            # Never infer historical forecasts for legacy trades after their exits.
            try:
                from forecast_timeline import estimate_target_timeline, completed_bars_only
                study_bars = completed_bars_only(
                    get_history_fast(f"{clean_sym}.NS", period="6mo"), ist_now().date(), market_open)
                timeline_study = estimate_target_timeline(
                    study_bars, live_price, target, stop, max_sessions=5)
                saved_timeline = {
                    "label": timeline_study.label,
                    "horizon_sessions": timeline_study.horizon_sessions,
                    "trials": timeline_study.trials,
                    "target_hits": timeline_study.target_hits,
                    "stop_hits": timeline_study.stop_hits,
                    "typical_sessions": timeline_study.typical_sessions,
                    "historical_target_rate_pct": timeline_study.target_rate_pct,
                    "credible": timeline_study.credible,
                    "recorded_ist": ist_now().strftime('%Y-%m-%d %H:%M:%S'),
                    "method": "completed_ohlc_stop_first_v1",
                }
            except Exception:
                # A missing timeline must never create a made-up estimate or
                # interfere with the existing share scan and risk decisions.
                saved_timeline = {"label": "Insufficient history", "credible": False,
                                  "horizon_sessions": 5, "method": "unavailable"}

            snapshot = {
                "timeline_at_entry": saved_timeline,
                "raw_features": raw_feature_map, "atr_pct": atr_pct, "raw_prob": round(raw_prob, 4), "base_ml_conf": base_conf,
                "learner_delta": learner["delta"], "macro_delta": macro_delta, "nlp_delta": nlp["delta"], "sector": sector,
                "friction_pct": friction, "avg_turnover_inr": round(avg_turnover, 0), "entry_bar": str(dates[-1]), "final_score": final_score,
                "rel_strength_20d": rel_strength, "ev_pct": ev_pct, "breakeven_pct": be_pct
            }
            results.append({
                "Ticker": clean_sym, "Sector": sector, "Price (₹)": round(live_price, 2),
                "Expected Return": f"+{net_return}% after costs", "ReturnNum": net_return,
                "Recommended Shares": f"{shares} shares", "RawCapital": capital, "Total Cost (₹)": f"₹{capital:,.0f}",
                "Target (₹)": target, "Stop Loss (₹)": stop, "Costs %": friction,
                "AI Win Confidence": f"{base_conf}%", "Base": base_conf, "Adjustment": total_adj,
                "Why": f"{learner['reason']} | {macro_note}", "FilingStatus": nlp["status"], "FilingTags": nlp["nlp_tags"],
                "FilingHeadline": nlp["headline"], "Adjusted Score": final_score, "Qualified": qualified, "Rejection Reason": reason,
                "Easy to trade (₹cr/day)": round(avg_turnover / 1e7, 1), "Loggable": bool(info["live_bar_today"] and window_open),
                "Vs Nifty (20d)": rel_strength, "Expected value": ev_pct, "Needs to win (%)": be_pct, "Safe": safe, "Learning": False,
                "FeaturesDict": snapshot, "FullSymbol": f"{clean_sym}.NS"
            })
        except Exception:
            stats["errors"] += 1
    prog.empty()
    stats["deep"] = len(results)
    stats["seconds"] = round(time.time() - t_start, 1)

    if not results:
        st.session_state["scan_stats"] = stats
        return pd.DataFrame(), False
    df_sorted = pd.DataFrame(results).sort_values(by=["Qualified", "Expected value", "Adjusted Score"], ascending=[False, False, False]).reset_index(drop=True)
    qualified_only = df_sorted[df_sorted["Qualified"] == True]
    stats["qualified"] = int(len(qualified_only))
    if not qualified_only.empty and not paused:
        for _, sig in qualified_only.iterrows():
            if not bool(sig.get("Loggable", False)):
                continue
            if len(get_todays_logged_equities()) >= MAX_DAILY_EQUITY_TRADES:
                break
            if log_equity_signal_safely(sig.to_dict(), enforce_sector_cap=True):
                stats["logged"] += 1
    decided_n = int((cal_info or {}).get("decided", 0))
    stats["learning_mode"] = decided_n < MIN_CALIBRATION_SAMPLES
    stats["decided"] = decided_n
    stats["learning_logged"] = 0
    loggable_confident = not qualified_only.empty and bool(qualified_only["Loggable"].any())
    if stats["learning_mode"] and not paused and not loggable_confident:
        room = LEARNING_PICKS_PER_DAY - todays_learning_picks()
        pool = df_sorted[(df_sorted["Safe"] == True) & (df_sorted["Qualified"] == False) & (df_sorted["Loggable"] == True)]
        pool = pool.sort_values(by=["Adjusted Score", "Expected value"], ascending=[False, False])
        for idx, sig in pool.iterrows():
            if room <= 0 or todays_learning_picks() >= LEARNING_PICKS_PER_DAY or len(get_todays_logged_equities()) >= MAX_DAILY_EQUITY_TRADES:
                break                                       # re-checked right before buying, in case another open tab just bought one
            d = sig.to_dict()
            d["FeaturesDict"] = dict(d["FeaturesDict"], learning_pick=True)
            if log_equity_signal_safely(d, enforce_sector_cap=True):
                df_sorted.at[idx, "Learning"] = True
                df_sorted.at[idx, "Rejection Reason"] = "🧪 Bought as today's learning pick (paper money, so the AI keeps learning)"
                stats["learning_logged"] += 1
                stats["logged"] += 1
                room -= 1
    st.session_state["scan_stats"] = stats
    return df_sorted, not qualified_only.empty

# ==============================================================================
# 12. PERFORMANCE MATHS
# ==============================================================================
def prep_perf_frame(df: pd.DataFrame, cap_col: str, is_option: bool = False) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    d = df[df["status"].apply(classify_outcome) != "ACTIVE"].copy()
    if d.empty:
        return d
    d["cap"] = pd.to_numeric(d[cap_col], errors="coerce").fillna(POSITION_SIZE_INR)
    d["pnl_pct"] = pd.to_numeric(d["pnl_pct"], errors="coerce").fillna(0.0)
    d["pnl_inr"] = d["cap"] * d["pnl_pct"] / 100.0
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
    wins, losses = int((d["outcome"] == "WIN").sum()), int((d["outcome"] == "LOSS").sum())
    be, exp = int((d["outcome"] == "BE").sum()), int((d["outcome"] == "EXPIRED").sum())
    pos, neg = d.loc[d["pnl_inr"] > 0, "pnl_inr"], d.loc[d["pnl_inr"] < 0, "pnl_inr"]
    curve = np.array([0.0] + d["pnl_inr"].cumsum().tolist())
    lo, hi = wilson_interval(wins, n)
    return {
        "n": n, "wins": wins, "losses": losses, "be": be, "expired": exp,
        "target_hit_rate": wins / n * 100.0, "ci_low": lo * 100.0, "ci_high": hi * 100.0,
        "avg_win": float(pos.mean()) if len(pos) else 0.0, "avg_loss": float(neg.mean()) if len(neg) else 0.0,
        "expectancy_pct": float(d["pnl_pct"].mean()), "expectancy_inr": float(d["pnl_inr"].mean()),
        "profit_factor": (float(pos.sum()) / abs(float(neg.sum()))) if len(neg) and neg.sum() != 0 else None,
        "max_dd": float((curve - np.maximum.accumulate(curve)).min()), "net_inr": float(d["pnl_inr"].sum()),
    }

@st.cache_data(ttl=1800, show_spinner=False)
def get_nifty_close_series() -> pd.Series:
    try:
        h = fetch_cached_history("^NSEI", period="6mo", interval="1d")
        if h is None or h.empty:
            return pd.Series(dtype=float)
        s = h["Close"].copy()
        s.index = pd.to_datetime([pd.Timestamp(x).date() for x in s.index])
        return s.sort_index()
    except Exception:
        return pd.Series(dtype=float)

def nifty_return_between(series: pd.Series, d0, d1):
    try:
        if series is None or series.empty or d0 is None or d1 is None:
            return None
        a, b = series.asof(pd.Timestamp(d0)), series.asof(pd.Timestamp(d1))
        if pd.isna(a) or pd.isna(b) or float(a) <= 0:
            return None
        return (float(b) / float(a) - 1.0) * 100.0
    except Exception:
        return None

# ==============================================================================
# 13. LOOK & FEEL
# ==============================================================================
CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
:root{--a1:#f97316;--a2:#16a34a;--good:#16a34a;--bad:#dc2626;--warn:#d97706;--info:#2563eb;--line:rgba(128,128,128,.28);--soft:rgba(128,128,128,.08);}
html,body,.stApp{font-family:'Inter',system-ui,-apple-system,sans-serif;}
.block-container{padding-top:1.1rem;max-width:1450px;}
footer{visibility:hidden;}
@keyframes drift{0%{background-position:0% 50%}50%{background-position:100% 50%}100%{background-position:0% 50%}}
.hero{border:1px solid var(--line);border-radius:22px;padding:22px 26px;background:linear-gradient(120deg,rgba(249,115,22,.26),rgba(250,204,21,.10),rgba(22,163,74,.22));background-size:200% 200%;animation:drift 16s ease infinite;margin-bottom:12px;}
.hero h1{margin:0;font-size:1.9rem;font-weight:800;letter-spacing:-.03em;background:linear-gradient(90deg,#f97316,#16a34a);-webkit-background-clip:text;-webkit-text-fill-color:transparent;}
.hero .sub{opacity:.85;font-size:.95rem;margin-top:6px;}
.chips{display:flex;flex-wrap:wrap;gap:8px;margin-top:14px;}
.chip{border:1px solid var(--line);background:var(--soft);padding:4px 12px;border-radius:999px;font-size:.78rem;font-weight:600;white-space:nowrap;}
.chip.up{color:var(--good);border-color:rgba(22,163,74,.5);}
.chip.down{color:var(--bad);border-color:rgba(220,38,38,.5);}
.chip.warn{color:var(--warn);border-color:rgba(217,119,6,.55);}
.sec{display:flex;align-items:center;gap:10px;margin:20px 0 10px;}
.sec .bar{width:5px;height:28px;border-radius:4px;background:linear-gradient(var(--a1),var(--a2));}
.sec h3{margin:0;font-size:1.25rem;font-weight:800;letter-spacing:-.01em;}
.sec small{opacity:.65;margin-left:8px;font-size:.82rem;font-weight:500;}
.score{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:14px;margin:6px 0 14px;}
.sc{position:relative;border:1px solid var(--line);border-radius:18px;padding:18px 20px;background:var(--soft);overflow:hidden;transition:transform .18s ease,box-shadow .18s ease;}
.sc:hover{transform:translateY(-3px);box-shadow:0 12px 30px rgba(0,0,0,.16);}
.sc:before{content:"";position:absolute;left:0;top:0;right:0;height:4px;background:linear-gradient(90deg,var(--a1),#facc15);}
.sc.good:before{background:linear-gradient(90deg,#16a34a,#4ade80);}
.sc.bad:before{background:linear-gradient(90deg,#dc2626,#f87171);}
.sc .lbl{font-size:.72rem;font-weight:700;letter-spacing:.07em;text-transform:uppercase;opacity:.65;}
.sc .val{font-size:2.2rem;font-weight:800;letter-spacing:-.03em;line-height:1.15;margin-top:6px;}
.sc .sub{font-size:.8rem;opacity:.72;margin-top:4px;}
.pos{color:var(--good);}
.neg{color:var(--bad);}
.callout{border-radius:16px;padding:14px 18px;margin:8px 0 14px;border:1px solid var(--line);font-size:.95rem;display:flex;gap:12px;align-items:flex-start;}
.callout .ic{font-size:1.4rem;line-height:1.25;}
.callout.good{background:rgba(22,163,74,.12);border-color:rgba(22,163,74,.45);}
.callout.bad{background:rgba(220,38,38,.12);border-color:rgba(220,38,38,.45);}
.callout.warn{background:rgba(217,119,6,.13);border-color:rgba(217,119,6,.45);}
.callout.info{background:rgba(37,99,235,.10);border-color:rgba(37,99,235,.40);}
.colh{display:flex;justify-content:space-between;align-items:center;border-radius:14px;padding:10px 14px;margin-bottom:10px;font-weight:800;border:1px solid var(--line);}
.colh.buy{background:linear-gradient(90deg,rgba(249,115,22,.22),rgba(249,115,22,.05));}
.colh.run{background:linear-gradient(90deg,rgba(22,163,74,.22),rgba(22,163,74,.05));}
.colh.done{background:linear-gradient(90deg,rgba(100,116,139,.25),rgba(100,116,139,.05));}
.colh span.n{font-size:.8rem;opacity:.8;font-weight:700;}
.tcard,.mini,.stat,.lesson,.fin,.empty{border:1px solid var(--line);border-radius:16px;padding:14px 16px;background:var(--soft);margin-bottom:10px;}
.tcard,.mini{transition:transform .15s ease,box-shadow .15s ease;}
.tcard:hover,.mini:hover{transform:translateY(-3px);box-shadow:0 10px 28px rgba(0,0,0,.18);}
.mini{padding:12px 14px;}
.mini.opt{border-left:4px solid var(--info);}
.fin{padding:10px 14px;display:flex;justify-content:space-between;align-items:center;gap:8px;margin-bottom:8px;}
.empty{text-align:center;opacity:.75;font-size:.9rem;padding:22px 16px;border-style:dashed;}
.tag{font-size:.66rem;font-weight:700;letter-spacing:.04em;text-transform:uppercase;padding:3px 9px;border-radius:999px;border:1px solid var(--line);}
.tag.good{color:var(--good);border-color:rgba(22,163,74,.5);}
.tag.bad{color:var(--bad);border-color:rgba(220,38,38,.5);}
.tag.warn{color:var(--warn);border-color:rgba(217,119,6,.5);}
.tag.info{color:var(--info);border-color:rgba(37,99,235,.5);}
.tags{display:flex;gap:6px;flex-wrap:wrap;margin-bottom:8px;}
.row{display:flex;justify-content:space-between;align-items:center;gap:10px;}
.tk{font-size:1.4rem;font-weight:800;letter-spacing:-.01em;}
.mtk{font-size:1.05rem;font-weight:800;}
.big{font-size:1.6rem;font-weight:800;line-height:1.1;}
.mp{font-size:1.05rem;font-weight:800;}
.sub2{font-size:.78rem;opacity:.72;}
.rbar{position:relative;height:10px;border-radius:999px;background:linear-gradient(90deg,rgba(220,38,38,.55),rgba(217,119,6,.45),rgba(22,163,74,.6));margin:16px 0 4px;}
.rbar.c{margin:12px 0 4px;height:8px;}
.rdot{position:absolute;top:-4px;width:18px;height:18px;border-radius:50%;background:#fff;border:3px solid var(--a1);transform:translateX(-50%);box-shadow:0 2px 8px rgba(0,0,0,.35);}
.rbar.c .rdot{width:16px;height:16px;}
.rtick{position:absolute;top:-3px;width:2px;height:16px;background:rgba(128,128,128,.9);transform:translateX(-50%);}
.rlab{display:flex;justify-content:space-between;font-size:.72rem;opacity:.75;gap:6px;}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:6px 12px;margin-top:10px;font-size:.8rem;}
.grid2 b{opacity:.6;font-weight:600;margin-right:4px;}
.ringwrap{position:relative;width:60px;height:60px;flex:none;}
.ring{width:60px;height:60px;border-radius:50%;-webkit-mask:radial-gradient(farthest-side,transparent 70%,#000 72%);mask:radial-gradient(farthest-side,transparent 70%,#000 72%);}
.ringnum{position:absolute;inset:0;display:grid;place-items:center;font-weight:800;font-size:.9rem;}
.note{font-size:.74rem;opacity:.75;margin-top:8px;}
.stat .l{font-size:.72rem;font-weight:700;letter-spacing:.06em;text-transform:uppercase;opacity:.6;}
.stat .v{font-size:1.2rem;font-weight:800;margin-top:4px;}
.hbars{display:flex;flex-direction:column;gap:10px;margin:6px 0 10px;}
.hb{display:grid;grid-template-columns:150px 1fr 120px;gap:10px;align-items:center;font-size:.86rem;}
.hb .t{height:12px;border-radius:999px;background:rgba(128,128,128,.18);overflow:hidden;}
.hb .f{height:100%;border-radius:999px;}
.hb .f.pos{background:linear-gradient(90deg,#16a34a,#4ade80);}
.hb .f.neg{background:linear-gradient(90deg,#dc2626,#f87171);}
.meter{height:16px;border-radius:999px;background:rgba(128,128,128,.2);overflow:hidden;margin:8px 0 4px;}
.meter>div{height:100%;border-radius:999px;background:linear-gradient(90deg,var(--a1),var(--a2));}
.lesson{padding:12px 16px;margin-bottom:8px;font-size:.9rem;}
.funnel{display:flex;flex-direction:column;gap:7px;margin:6px 0 12px;}
.fn{display:grid;grid-template-columns:210px 1fr 80px;gap:10px;align-items:center;font-size:.86rem;}
.fn .t{height:22px;border-radius:8px;background:rgba(128,128,128,.12);overflow:hidden;}
.fn .f{height:100%;border-radius:8px;background:linear-gradient(90deg,var(--a1),var(--a2));opacity:.9;}
.fn .f.last{background:linear-gradient(90deg,#16a34a,#4ade80);}
.fn .c{text-align:right;font-weight:800;font-variant-numeric:tabular-nums;}
.gauge{height:14px;border-radius:999px;background:linear-gradient(90deg,rgba(220,38,38,.6),rgba(217,119,6,.5) 50%,rgba(22,163,74,.65));position:relative;margin:12px 0 6px;}
.gauge .needle{position:absolute;top:-5px;width:6px;height:24px;border-radius:3px;background:#fff;border:2px solid #111;transform:translateX(-50%);box-shadow:0 2px 8px rgba(0,0,0,.35);}
.leaders{display:flex;flex-wrap:wrap;gap:8px;margin:6px 0 4px;}
.ld{border:1px solid var(--line);border-radius:12px;padding:7px 11px;background:var(--soft);font-size:.84rem;}
.ld b{font-weight:800;margin-right:6px;}
@media(max-width:700px){.fn{grid-template-columns:120px 1fr 56px;font-size:.78rem;}}
.disc{font-size:.76rem;opacity:.6;text-align:center;margin:30px 0 6px;}
[data-testid="stMetric"]{border:1px solid var(--line);border-radius:14px;padding:12px 14px;background:var(--soft);}
[data-baseweb="tab-list"]{gap:6px;flex-wrap:wrap;}
button[data-baseweb="tab"],button[data-testid="stTab"],.stTabs button[role="tab"]{height:auto !important;padding:12px 18px !important;border-radius:12px 12px 0 0;}
button[data-baseweb="tab"],button[data-baseweb="tab"] p,button[data-baseweb="tab"] div,button[data-baseweb="tab"] span,
button[data-testid="stTab"] p,.stTabs button[role="tab"] p{font-size:1.15rem !important;font-weight:700 !important;line-height:1.3 !important;}
button[data-baseweb="tab"][aria-selected="true"],button[data-testid="stTab"][aria-selected="true"]{background:var(--soft);}
.stButton>button{border-radius:12px;font-weight:600;}
@media(max-width:700px){button[data-baseweb="tab"],button[data-testid="stTab"]{padding:9px 11px !important;}button[data-baseweb="tab"],button[data-baseweb="tab"] p,button[data-baseweb="tab"] div,button[data-baseweb="tab"] span,button[data-testid="stTab"] p,.stTabs button[role="tab"] p{font-size:1rem !important;}.hb{grid-template-columns:96px 1fr 80px;}.sc .val{font-size:1.7rem;}.hero h1{font-size:1.45rem;}}
"""

def inject_css():
    st.markdown(f"<style>{_collapse(CSS)}</style>", unsafe_allow_html=True)

def inr(x, d=0) -> str:
    try:
        v = float(x)
        if not math.isfinite(v):
            return "₹–"
        txt = f"₹{abs(v):,.{d}f}"
        return ("-" + txt) if round(v, d) < 0 else txt
    except Exception:
        return "₹–"

def friendly_status(status, be_locked: bool = False) -> str:
    o = classify_outcome(status)
    if o == "ACTIVE":
        return "🟢 Open (profit protected)" if be_locked else "🟢 Open"
    if "NO PRICE" in str(status).upper():
        return "⏱️ Closed — the share stopped trading (last known price)"
    return {"WIN": "✅ Won — hit the goal", "LOSS": "❌ Lost — safety exit hit",
            "BE": "🛡️ Saved — sold with no loss", "EXPIRED": "⏱️ Closed after 7 days"}.get(o, str(status))

def outcome_style(status) -> tuple:
    return {"WIN": ("Won", "good"), "LOSS": ("Lost", "bad"), "BE": ("Saved", "warn"),
            "EXPIRED": ("Time's up", "warn"), "ACTIVE": ("Open", "info")}.get(classify_outcome(status), (str(status)[:12], "info"))

def sec_title(icon: str, title: str, sub: str = ""):
    small = f"<small>{esc(sub)}</small>" if sub else ""
    st.markdown(_collapse(f'<div class="sec"><div class="bar"></div><h3>{esc(icon)} {esc(title)}{small}</h3></div>'), unsafe_allow_html=True)

def callout(kind: str, icon: str, text_html: str):
    st.markdown(_collapse(f'<div class="callout {kind}"><div class="ic">{icon}</div><div>{text_html}</div></div>'), unsafe_allow_html=True)

def score_cards(items: list):
    cards = [f'<div class="sc {kind}"><div class="lbl">{esc(label)}</div><div class="val {vcls}">{esc(value)}</div><div class="sub">{esc(sub)}</div></div>'
             for label, value, sub, kind, vcls in items]
    st.markdown(_collapse('<div class="score">' + "".join(cards) + '</div>'), unsafe_allow_html=True)

def stat_card(label: str, value: str, sub: str = ""):
    st.markdown(_collapse(f'<div class="stat"><div class="l">{esc(label)}</div><div class="v">{esc(value)}</div><div class="sub2">{esc(sub)}</div></div>'), unsafe_allow_html=True)

def col_header(kind: str, title: str, n: int):
    st.markdown(_collapse(f'<div class="colh {kind}"><div>{esc(title)}</div><span class="n">{n}</span></div>'), unsafe_allow_html=True)

def empty_box(text: str):
    st.markdown(_collapse(f'<div class="empty">{esc(text)}</div>'), unsafe_allow_html=True)

def hbars_html(rows: list) -> str:
    if not rows:
        return ""
    m = max(abs(v) for _, v, _ in rows) or 1.0
    parts = []
    for label, v, txt in rows:
        w = max(4.0, abs(v) / m * 100.0)
        parts.append(f'<div class="hb"><div>{esc(label)}</div><div class="t"><div class="f {"pos" if v >= 0 else "neg"}" style="width:{w:.0f}%"></div></div><div><b>{esc(txt)}</b></div></div>')
    return _collapse('<div class="hbars">' + "".join(parts) + '</div>')

def funnel_html(rows: list) -> str:
    """rows: [(label, count)] from widest to narrowest."""
    rows = [(lbl, int(max(0, n))) for lbl, n in rows]
    top = max([n for _, n in rows] + [1])
    parts = []
    for i, (lbl, n) in enumerate(rows):
        w = 0.0 if n == 0 else max(1.5, n / top * 100.0)
        last = " last" if i == len(rows) - 1 else ""
        parts.append(f'<div class="fn"><div>{esc(lbl)}</div><div class="t"><div class="f{last}" style="width:{w:.1f}%"></div></div><div class="c">{n:,}</div></div>')
    return _collapse('<div class="funnel">' + "".join(parts) + '</div>')

def breadth_word(pct: float) -> str:
    pct = round(float(pct))
    if pct >= 60:
        return "Rising broadly"
    if pct >= 45:
        return "Mixed"
    if pct >= 30:
        return "Mostly weak"
    return "Falling broadly"

def gauge_html(pct: float) -> str:
    p = max(0.0, min(100.0, float(pct)))
    return _collapse(f'<div class="gauge"><div class="needle" style="left:{p:.1f}%"></div></div>'
                     '<div class="rlab" style="margin-bottom:14px"><span>Most shares falling</span><span>Half and half</span><span>Most shares rising</span></div>')

def sector_heat_html(breadth: dict, min_n: int = 5, top: int = 8) -> str:
    rows = [(sec, up / n * 100.0, up, n) for sec, (up, n) in breadth.items() if n >= min_n and sec != "Other"]
    rows.sort(key=lambda x: (-x[1], -x[3]))
    if not rows:
        return ""
    parts = []
    for sec, pct, up, n in rows[:top]:
        cls = "pos" if pct >= 50 else "neg"
        parts.append(f'<div class="hb"><div>{esc(sec)}</div><div class="t"><div class="f {cls}" style="width:{max(3.0, pct):.0f}%"></div></div>'
                     f'<div><b>{pct:.0f}% rising</b> <span class="sub2">of {n}</span></div></div>')
    return _collapse('<div class="hbars">' + "".join(parts) + '</div>')

def leaders_html(leaders: list) -> str:
    if not leaders:
        return ""
    chips = [f'<span class="ld"><b>{esc(t)}</b><span class="{"pos" if rs >= 0 else "neg"}">{rs:+.1f}%</span> <span class="sub2">{esc(sec)}</span></span>'
             for t, sec, rs in leaders]
    return _collapse('<div class="leaders">' + "".join(chips) + '</div>')

def price_bar_html(entry: float, target: float, stop: float, live: float, compact: bool = False) -> str:
    span = target - stop
    if span <= 0:
        return ""
    pos = lambda v: max(0.0, min(100.0, (v - stop) / span * 100.0))
    bar = f'<div class="rbar{" c" if compact else ""}"><div class="rtick" style="left:{pos(entry):.1f}%"></div><div class="rdot" style="left:{pos(live):.1f}%"></div></div>'
    if compact:
        return _collapse(bar)
    return _collapse(bar + f'<div class="rlab"><span>🛑 Exit if it falls to ₹{stop:,.2f}</span><span>Bought ₹{entry:,.2f}</span><span>🎯 Goal ₹{target:,.2f}</span></div>')

def _fmt_rs(v) -> str:
    try:
        v = float(v)
        return f"{v:+.1f}%" if math.isfinite(v) else "–"
    except Exception:
        return "–"

def signal_card_html(row) -> str:
    conf = int(round(float(row["Adjusted Score"])))
    colour = "#16a34a" if conf >= 75 else ("#f97316" if conf >= 65 else "#d97706")
    ready = '<span class="tag good">Ready to buy</span>' if bool(row.get("Loggable", False)) else '<span class="tag warn">Watch only</span>'
    return _collapse(f"""
    <div class="tcard">
      <div class="tags"><span class="tag">{esc(row['Sector'])}</span>{ready}</div>
      <div class="row"><div><div class="tk">{esc(row['Ticker'])}</div><div class="sub2">AI confidence</div></div>
      <div class="ringwrap"><div class="ring" style="background:conic-gradient({colour} {conf}%,rgba(128,128,128,.25) 0)"></div><div class="ringnum">{conf}%</div></div></div>
      <div class="big pos" style="margin-top:8px">{esc(str(row['Expected Return']))}</div><div class="sub2">possible profit if it reaches the goal</div>
      {price_bar_html(float(row['Price (₹)']), float(row['Target (₹)']), float(row['Stop Loss (₹)']), float(row['Price (₹)']))}
      <div class="grid2"><div><b>Shares</b>{esc(row['Recommended Shares'])}</div><div><b>Money in</b>{inr(row['RawCapital'])}</div>
      <div><b>Costs</b>{float(row['Costs %']):.2f}%</div><div><b>Easy to trade</b>₹{row.get('Easy to trade (₹cr/day)', 0)}cr/day</div>
      <div><b>Vs Nifty</b>{_fmt_rs(row.get('Vs Nifty (20d)'))} (20 days)</div><div><b>Sell by</b>{MAX_HOLD_CALENDAR_DAYS} days at most</div></div>
      <div class="note">🧠 {esc(row['Why'])}</div>
      <div class="note">📰 {esc(row['FilingStatus'])}</div>
    </div>
    """)

def mini_trade_html(ticker, sector, entry, target, stop, live, pnl_pct, locked, shares, learning: bool = False) -> str:
    cls = "pos" if pnl_pct >= 0 else "neg"
    return _collapse(f"""
    <div class="mini">
      <div class="row"><span class="mtk">{esc(ticker)}{' 🛡️' if locked else ''}{' 🧪' if learning else ''}</span><span class="mp {cls}">{pnl_pct:+.2f}%</span></div>
      {price_bar_html(entry, target, stop, live, compact=True)}
      <div class="sub2">{esc(sector)} · now ₹{live:,.2f} · {int(shares)} shares{' · learning pick' if learning else ''}</div>
    </div>
    """)

def option_short_name(contract) -> str:
    return str(contract).split(" (")[0]

def option_mini_html(o) -> str:
    entry, target, stop = float(o["entry_premium"]), float(o["target_premium"]), float(o["stop_loss_premium"])
    live, pnl = float(o["current_option_price"]), float(o["pnl_pct"])
    src = '<span class="tag good">Live NSE quote</span>' if is_live_option(o["option_contract"]) else '<span class="tag warn">Estimated price</span>'
    cls = "pos" if pnl >= 0 else "neg"
    return _collapse(f"""
    <div class="mini opt">
      <div class="tags"><span class="tag info">Option</span>{src}</div>
      <div class="row"><span class="mtk">{esc(option_short_name(o['option_contract']))}</span><span class="mp {cls}">{pnl:+.2f}%</span></div>
      {price_bar_html(entry, target, stop, live, compact=True)}
      <div class="sub2">bought ₹{entry:,.2f} · now ₹{live:,.2f} · 1 lot of {int(float(o['lot_size']))} · since {esc(str(o['date_key']))}</div>
    </div>
    """)

def finished_row_html(ticker, status, pnl_pct, when) -> str:
    s_text, s_kind = outcome_style(status)
    cls = "pos" if pnl_pct >= 0 else "neg"
    return _collapse(f'<div class="fin"><div><b>{esc(ticker)}</b> <span class="tag {s_kind}">{esc(s_text)}</span><div class="sub2">{esc(when)}</div></div><div class="mp {cls}">{pnl_pct:+.2f}%</div></div>')

# No charts are created or displayed: all results use words and tables.

def style_ledger(df: pd.DataFrame):
    try:
        def pnl_color(v):
            try:
                return "color:#16a34a;font-weight:700" if float(v) > 0 else ("color:#dc2626;font-weight:700" if float(v) < 0 else "")
            except Exception:
                return ""
        sty = df.style.format({"Bought at (₹)": "{:,.2f}", "Goal (₹)": "{:,.2f}", "Safety exit (₹)": "{:,.2f}", "Price now (₹)": "{:,.2f}", "Profit/Loss (%)": "{:+.2f}%"}, na_rep="-")
        mapper = sty.map if hasattr(sty, "map") else sty.applymap
        return mapper(pnl_color, subset=["Profit/Loss (%)"])
    except Exception:
        return df

# ---------- learning helpers ----------
def compute_memory(closed_df: pd.DataFrame) -> tuple:
    careful, liked = [], []
    if closed_df is None or closed_df.empty:
        return careful, liked
    today_dt = ist_now().date()
    acc = {}
    for _, r in closed_df.iterrows():
        o = classify_outcome(r["status"])
        if o not in ("WIN", "LOSS"):
            continue
        d = _to_date(_first_valid(r.get("exit_timestamp"), r.get("last_audited"), r.get("date_str")))
        w = math.pow(0.5, (max(0, (today_dt - d).days) if d else 15) / HALF_LIFE_DAYS)
        a = acc.setdefault(clean_sym_name(r["ticker"]), {"loss": 0.0, "win": 0.0})
        a["loss" if o == "LOSS" else "win"] += (15.0 if o == "LOSS" else 5.0) * w
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
    out = []
    if perf_df.empty:
        return out
    sg = sector_stats(perf_df)
    sg3 = sg[sg["n"] >= 3] if not sg.empty else sg
    if not sg3.empty:
        best, worst = sg3.sort_values("avg", ascending=False).iloc[0], sg3.sort_values("avg").iloc[0]
        out.append(f"Best sector so far: **{best['Sector']}** (average {best['avg']:+.1f}% per trade over {int(best['n'])} trades).")
        if worst["Sector"] != best["Sector"]:
            out.append(f"Toughest sector so far: **{worst['Sector']}** (average {worst['avg']:+.1f}% per trade over {int(worst['n'])} trades).")
    try:
        frs = [get_trade_friction_pct(r["features_json"]) for _, r in perf_df.iterrows()]
        out.append(f"Government taxes, broker and exchange charges take about **{np.mean(frs):.2f}%** off every trade — a trade has to beat that before it earns anything.")
    except Exception:
        pass
    wins, losses = perf_df[perf_df["outcome"] == "WIN"], perf_df[perf_df["outcome"] == "LOSS"]
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
            v = [float(_parse_features(r["features_json"]).get("atr_pct")) for _, r in df_.iterrows() if _parse_features(r["features_json"]).get("atr_pct") is not None]
            return (float(np.mean(v)), len(v)) if v else (None, 0)
        (aw, nw), (al, nl) = atr_mean(wins), atr_mean(losses)
        if aw is not None and al is not None and nw >= 5 and nl >= 5:
            if al > aw + 0.3:
                out.append(f"Losing trades were on shares that move about **{al:.1f}% a day**, versus **{aw:.1f}%** for winners — very jumpy shares have been riskier, so the AI is wary of them.")
            else:
                out.append(f"Winners and losers moved a similar amount day to day ({aw:.1f}% vs {al:.1f}%), so jumpiness hasn't been the deciding factor.")
    except Exception:
        pass
    return out

def current_streak(perf_df: pd.DataFrame) -> tuple:
    if perf_df.empty:
        return 0, True
    signs = (perf_df["pnl_inr"] > 0).tolist()
    k = 0
    for s in reversed(signs):
        if s == signs[-1]:
            k += 1
        else:
            break
    return k, signs[-1]

def _safe_num_series(s):
    return pd.to_numeric(s, errors="coerce") if s is not None else pd.Series(dtype=float)


def summarise_learning_progress() -> dict:
    """One consistent read for the learning counter and recent finished trades."""
    from dashboard_insights import summarize_share_journal

    err = None
    with DB_LOCK:
        con = duckdb.connect(DB_PATH, read_only=True)
        try:
            all_shares = con.execute("""SELECT ticker, status, pnl_pct, date_str,
                                               exit_timestamp, last_audited
                                        FROM trade_journal""").df()
        except Exception as e:
            logging.exception("Failed loading share history for dashboard")
            err = f"Could not read share history: {type(e).__name__}. See Manage app → Logs."
            all_shares = pd.DataFrame()
        finally:
            con.close()
    summary = summarize_share_journal(all_shares, recent_limit=4)
    last_state_raw = get_meta("retrain_state", "{}")
    try:
        train_state = json.loads(last_state_raw or "{}")
    except (ValueError, TypeError):
        train_state = {}
    return {
        "completed": summary["settled"],
        "win_rate": summary["win_rate"],
        "decided": summary["decided"],
        "recent_finished": summary["recent"],
        "read_error": err,
        "latest_retrain": train_state.get("last_attempt_date", "Not eligible yet"),
        "last_promotion": train_state.get("last_promoted", "None recorded"),
    }


def select_top_share_ideas(df: pd.DataFrame, limit: int = 3) -> pd.DataFrame:
    if not isinstance(df, pd.DataFrame) or df.empty:
        return pd.DataFrame()
    work = df.copy()
    work["Decision"] = work.apply(lambda r: "CONSIDER" if r.get("Qualified", False)
        else "AVOID" if "bad news" in str(r.get("Rejection Reason", "")).lower() or "risk" in str(r.get("Rejection Reason", "")).lower()
        else "WATCH", axis=1)
    pref = {"CONSIDER": 0, "WATCH": 1, "AVOID": 2}
    work["_rank_decision"] = work["Decision"].map(pref).fillna(9)
    work["_rank_score"] = pd.to_numeric(work.get("Adjusted Score"), errors="coerce").fillna(-9999)
    work["_rank_ev"] = pd.to_numeric(work.get("Expected value"), errors="coerce").fillna(-9999)
    work = work.sort_values(["_rank_decision", "_rank_score", "_rank_ev"], ascending=[True, False, False])
    return work.head(limit)


def select_top_option_ideas(df: pd.DataFrame, limit_each: int = 2) -> pd.DataFrame:
    if not isinstance(df, pd.DataFrame) or df.empty:
        return pd.DataFrame()
    work = df.copy()
    pref = {"CONSIDER": 0, "WATCH": 1, "AVOID": 2}
    work["_rank_decision"] = work["Decision"].map(pref).fillna(9)
    work["_rank_oi"] = pd.to_numeric(work.get("Open interest"), errors="coerce").fillna(0)
    work["_rank_spread"] = pd.to_numeric(work.get("Spread (%)"), errors="coerce").fillna(9999)
    work["_rank_days"] = pd.to_numeric(work.get("Days left"), errors="coerce").fillna(9999)
    out = []
    for side in ["CE", "PE"]:
        side_df = work[work.get("Type").astype(str) == side].sort_values(
            ["_rank_decision", "_rank_spread", "_rank_oi", "_rank_days"],
            ascending=[True, True, False, True]
        )
        if not side_df.empty:
            out.append(side_df.head(limit_each))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()



# ==============================================================================
# 14. PREMIUM DARK RESEARCH WORKSPACE — THREE VIEWS, NO CHARTS
# ==============================================================================
import logging
import tempfile
from premium_dark import (
    THEMES as DARK_THEMES, theme_css as dark_theme_css, hero as dark_hero,
    stats as dark_stats, header as dark_header, note as dark_note,
    signal_card as dark_card, learning_tile as dark_learning_tile,
)
from forecast_timeline import estimate_target_timeline, completed_bars_only
from learning_insights import (
    describe_trade, stock_evidence, stock_history_summary, is_training_ready, forecast_review,
)
from history_recovery import merge_history_database, snapshot_database, count_history
from scan_health import option_http_note
from portfolio_overview import portfolio_open_snapshot
from learning_tiles import build_learning_tile
from learning_feed import latest_completed_by_stock


def read_full_paper_ledger() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Read one coherent local snapshot of history; errors must never look like zero trades."""
    with DB_LOCK:
        con = duckdb.connect(DB_PATH, read_only=True)
        try:
            a = con.execute("SELECT * FROM trade_journal ORDER BY date_str DESC NULLS LAST, timestamp DESC NULLS LAST").df()
            b = con.execute("SELECT * FROM daily_options_journal ORDER BY date_key DESC NULLS LAST").df()
            c = con.execute("SELECT * FROM option_watchlist ORDER BY checked_at DESC NULLS LAST").df()
            return a, b, c
        finally:
            con.close()


def stock_timeline(symbol: str, entry: float, target: float, stop: float, sessions: int):
    """Only completed bars, cached historical data; not a trained time-to-target AI."""
    sym = normalize_ticker_for_yf(symbol)
    history = get_history_fast(sym, '6mo')
    return estimate_target_timeline(
        completed_bars_only(history, ist_now().date(), is_nse_market_open()),
        entry, target, stop, max_sessions=sessions,
    )


def labelled_decision(row) -> str:
    if bool(row.get('Qualified', False)):
        return 'CONSIDER'
    reason = str(row.get('Rejection Reason', '')).lower()
    if any(w in reason for w in ('bad news', 'risk', 'filing', 'not verified', 'blocked')):
        return 'AVOID'
    return 'WATCH'


def ranked_shares(rows: pd.DataFrame) -> pd.DataFrame:
    if rows is None or rows.empty:
        return pd.DataFrame()
    df = rows.copy()
    df['Decision'] = df.apply(labelled_decision, axis=1)
    df['_order'] = df['Decision'].map({'CONSIDER':0,'WATCH':1,'AVOID':2}).fillna(3)
    df['_ev'] = pd.to_numeric(df['Expected value'],errors='coerce').fillna(-1e6)
    df['_score'] = pd.to_numeric(df['Adjusted Score'],errors='coerce').fillna(-1e6)
    return df.sort_values(['_order','_ev','_score'], ascending=[True,False,False])


def option_sort(rows: pd.DataFrame) -> pd.DataFrame:
    if rows is None or rows.empty:
        return pd.DataFrame()
    df = rows.copy()
    df['_order'] = df['Decision'].map({'CONSIDER':0,'WATCH':1,'AVOID':2}).fillna(3)
    df['_liquidity'] = pd.to_numeric(df['Open interest'],errors='coerce').fillna(0)
    df['_spread'] = pd.to_numeric(df['Spread (%)'],errors='coerce').fillna(1e6)
    return df.sort_values(['_order','_spread','_liquidity'], ascending=[True,True,False])


def short_money(n, d=2) -> str:
    try:
        v = float(n)
        return f'₹{v:,.{d}f}' if math.isfinite(v) else '—'
    except (TypeError,ValueError):
        return '—'


def render_history_recovery() -> None:
    """Move the original history repair workflow into the Ledger without data loss."""
    with st.expander('🛠️ Backup and recover history', expanded=False):
        st.caption('Keep the original Supabase project and your local DuckDB backups. An upload MERGES IDs; it never deliberately resets the journal.')
        if supabase:
            if st.button('☁️ Retry restoring from Supabase', key='dark_cloud_restore', width="stretch"):
                with st.spinner('Comparing cloud and local paper history...'):
                    hydrate_duckdb_from_supabase()
                st.rerun()
            hi = st.session_state.get('hydrate_info',{})
            st.caption(f"Last cloud check: {hi.get('remote',0)} share rows, {hi.get('remote_options',0)} option rows, "
                       f"{hi.get('restored',0)} share rows restored, {hi.get('bad_rows',0)} skipped")
            if st.button('📤 Sync up to 200 missing records to Supabase', key='dark_cloud_sync', width="stretch"):
                with st.spinner('Uploading only rows absent from cloud...'):
                    result = sync_local_history_to_supabase(200, st.session_state.get('history_sync_offset',0))
                st.session_state['history_sync_offset'] = result['next_offset'] if result['more_to_check'] else 0
                if result['failed']:
                    st.error(f"Sync incomplete: {result['failed']} failed. Check app logs. No existing remote rows were deliberately overwritten.")
                else:
                    st.success(f"Added {result['shares_inserted']} share and {result['options_inserted']} option records; "
                               f"{result['already_remote']} were already in cloud.")
                if result['more_to_check']:
                    st.info('More local records remain to check. Click Sync again to continue.')
        else:
            st.warning('Supabase is not configured on this deployment. Restore using the ORIGINAL SUPABASE_URL and SUPABASE_KEY secrets, or upload a local backup.')
        previous = st.file_uploader('Merge an older DuckDB backup', type=['duckdb'], key='dark_old_db')
        if previous is not None:
            st.caption(f'{previous.name} · {previous.size/1048576:.1f} MB')
            if previous.size > 50*1024*1024:
                st.error('50 MB maximum for direct uploads. Restore larger backups using a managed process.')
            elif st.button('Merge my previous history',key='dark_merge_db',type='primary'):
                try:
                    with tempfile.TemporaryDirectory(prefix='paper_restore_') as tmpdir:
                        old_path = Path(tmpdir)/'older_history.duckdb'
                        old_path.write_bytes(previous.getvalue())
                        with DB_LOCK:
                            report = merge_history_database(old_path, DB_PATH, ROOT_DIR/'backups')
                    st.session_state['history_import_result'] = report
                    st.rerun()
                except Exception as error:
                    logging.exception('History merge failed')
                    st.error(f'Import safely stopped: {type(error).__name__}. See Streamlit logs.')
        report = st.session_state.get('history_import_result')
        if report:
            st.success('Added (not replaced): '
                       f"{report['added'].get('trade_journal',0)} stocks, "
                       f"{report['added'].get('daily_options_journal',0)} options, "
                       f"{report['added'].get('option_watchlist',0)} reviews.")
            st.warning('Uploaded records remain local until backed up/synced to persistent storage.')
        if st.button('💾 Prepare full DuckDB backup', key='dark_prepare_backup'):
            with DB_LOCK:
                outfile = snapshot_database(DB_PATH, ROOT_DIR/'backups', 'manual_history')
            st.session_state['dark_backup_bytes'] = outfile.read_bytes()
            st.session_state['dark_backup_stamp'] = ist_now().strftime('%Y%m%d_%H%M')
        if st.session_state.get('dark_backup_bytes'):
            st.download_button('⬇️ Download full history (.duckdb)',
                data=st.session_state['dark_backup_bytes'],
                file_name=f"NSE_paper_history_{st.session_state['dark_backup_stamp']}.duckdb",
                mime='application/octet-stream', key='dark_download_history')
        st.caption('Streamlit Cloud local files are not permanent. Store a backup externally and verify cloud sync before relying on unattended paper research.')


# -----------------------------------------------------------------------------
# Minimal dark workspace · presentation only (all values escaped)
# No changes to trading rules, database operations, scan schedules or ML models.
# -----------------------------------------------------------------------------
def minimal_dark_css() -> str:
    return r"""
    :root {--ml-bg:#0b1019;--ml-panel:#171e2a;--ml-panel2:#1c2532;--ml-border:#2d3949;
           --ml-copy:#edf3f9;--ml-muted:#99aabc;--ml-green:#72e6af;--ml-red:#ff969e;}
    html,body,.stApp,[data-testid='stAppViewContainer'] {font-family:'DM Sans',system-ui,sans-serif;}
    .stApp{background:radial-gradient(ellipse 50% 34% at 96% 0%,rgba(107,89,181,.11),transparent 80%),
      radial-gradient(ellipse 38% 33% at 2% 93%,rgba(31,143,122,.06),transparent 85%),var(--ml-bg);}
    .block-container {max-width:1600px;padding:1.15rem clamp(1.0rem,2.6vw,2.7rem) 3rem;}
    [data-testid='stSidebar']{background:#0c121d;border-right:1px solid #263143;}
    [data-testid='stSidebar'] [data-testid='stVerticalBlock'] {gap:.7rem;}
    .ml-brand{padding:12px 4px 16px;border-bottom:1px solid #263345;margin-bottom:8px;}
    .ml-brand-name{display:flex;align-items:center;gap:11px;font:750 19px 'Outfit',sans-serif;color:#f4f8ff;line-height:1.18;}
    .ml-brand-icon{display:grid;place-items:center;width:39px;height:39px;background:linear-gradient(145deg,#194b3e,#17342e);
        color:#8af5c5;border-radius:11px;font-size:24px;}
    .ml-brand-caption{color:#8399b4;font-size:11px;margin:12px 0 0 1px;letter-spacing:.06em;}
    .ml-hero{display:flex;align-items:center;justify-content:space-between;gap:16px;flex-wrap:wrap;margin:5px 0 17px;}
    .ml-hero h1{font:750 clamp(26px,2.7vw,38px)/1.16 'Outfit','DM Sans',sans-serif;color:#f7f9ff;letter-spacing:-.035em;margin:0;}
    .ml-sub{font-size:13px;color:#a1b2c7;margin:4px 0 0;}
    .ml-meta{display:flex;align-items:center;flex-wrap:wrap;justify-content:flex-end;gap:10px;color:#b6c6d8;font-size:12px;}
    .ml-market{color:#7aeeb4;font-weight:750;display:inline-flex;align-items:center;gap:6px;}
    .ml-market.closed{color:#d3b6ef;}.ml-dot{width:7px;height:7px;border-radius:99px;background:currentColor;display:inline-block;}
    .ml-divider{height:15px;width:1px;background:#344051;}
    .ml-paper{background:#172233;border:1px solid #344258;color:#aec0d6;border-radius:99px;padding:5px 9px;font-size:11px;}
    .ml-stats{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px;margin:6px 0 19px;}
    .ml-stat{border:1px solid var(--ml-border);border-radius:16px;background:linear-gradient(135deg,#1a2230,#161d29);
        padding:16px 18px;min-width:0;min-height:110px;box-shadow:0 8px 28px rgba(0,0,0,.11);}
    .ml-stat-heading{display:flex;align-items:center;gap:9px;color:#b5c5d7;font-size:12px;font-weight:650;}
    .ml-stat-icon{width:27px;height:27px;display:grid;place-items:center;background:#273547;border-radius:9px;color:#9fbde2;font-size:15px;}
    .ml-stat:nth-child(2) .ml-stat-icon{color:var(--ml-green);background:#183c33;}
    .ml-stat-value{color:#f4f8ff;font:750 clamp(21px,2.25vw,28px) 'Outfit',sans-serif;line-height:1.25;
        margin:8px 0 5px;font-variant-numeric:tabular-nums;overflow-wrap:anywhere;}
    .ml-stat-value.gain{color:var(--ml-green);}.ml-stat-value.loss{color:var(--ml-red);}
    .ml-stat-sub{font-size:11px;color:#92a6bc;line-height:1.4;}
    .stTabs [data-baseweb='tab-list']{background:transparent;border:none;border-bottom:1px solid #344051;border-radius:0;
        padding:0;gap:15px;min-height:48px;margin-top:2px;}
    .stTabs button[role='tab'],.stTabs [data-baseweb='tab']{background:transparent !important;box-shadow:none !important;
        color:#a4b6cb !important;border:0 !important;border-bottom:2px solid transparent !important;
        border-radius:0 !important;padding:10px 12px !important;min-height:45px;}
    .stTabs button[role='tab'] p{font-size:13px !important;font-weight:650 !important;}
    .stTabs button[role='tab'][aria-selected='true']{color:#e5fff2 !important;border-bottom:2px solid #78e7b2 !important;}
    .ml-panel-title{margin:14px 0 12px;display:flex;flex-wrap:wrap;justify-content:space-between;align-items:baseline;gap:10px;}
    .ml-panel-title h2{font:750 23px 'Outfit',sans-serif;letter-spacing:-.025em;margin:0;color:#f7f9ff;}
    .ml-panel-title p{font-size:12px;color:#96aac0;margin:2px 0 0;}
    .ml-count{font-size:11px;color:#acbfd2;background:#17263a;border:1px solid #344057;padding:6px 10px;border-radius:99px;}
    .ml-row{border:1px solid #313e50;border-radius:13px;background:linear-gradient(120deg,#1c2531,#171f2b);
        padding:14px 14px;display:grid;grid-template-columns:minmax(122px,1.48fr) minmax(73px,.75fr)
        repeat(3,minmax(75px,.86fr)) minmax(93px,1.1fr);gap:10px;align-items:center;
        margin:8px 0;transition:border-color .18s,transform .18s,box-shadow .18s;}
    .ml-row:hover{border-color:#52627b;transform:translateY(-1px);box-shadow:0 10px 25px rgba(0,0,0,.15);}
    .ml-ticker{font:750 15px 'Outfit',sans-serif;color:#f4f8ff;overflow-wrap:anywhere;}
    .ml-ticker-meta{font-size:10px;color:#93a8be;margin-top:3px;overflow-wrap:anywhere;line-height:1.4;}
    .ml-symbol{display:flex;align-items:center;gap:11px;min-width:0;}
    .ml-monogram{flex:0 0 34px;height:34px;width:34px;background:#213649;border-radius:50%;border:1px solid #36506a;
        color:#8edccf;display:grid;place-items:center;font-size:14px;font-weight:800;}
    .ml-col small{font-size:10px;color:#8ea1b9;display:block;margin-bottom:4px;}
    .ml-col strong{font-size:13px;font-weight:750;color:#e5edf8;word-wrap:anywhere;font-variant-numeric:tabular-nums;}
    .ml-col strong.target{color:#7cecb8;}.ml-col strong.stop{color:#ffa0a7;}
    .ml-pill{display:inline-block;font-size:9px;letter-spacing:.035em;font-weight:800;padding:5px 9px;
        border-radius:99px;background:#1a4838;color:#90f3c2;white-space:nowrap;}
    .ml-pill.watch{background:#273c58;color:#afd0fa;}.ml-pill.avoid{background:#522e38;color:#ffb5bd;}
    .ml-side-panel{border:1px solid #314054;border-radius:16px;background:linear-gradient(138deg,#1b2631,#141c29);
        padding:19px 18px;margin:14px 0;box-shadow:0 10px 30px rgba(0,0,0,.10);}
    .ml-side-panel h3{font:750 18px 'Outfit',sans-serif;color:#eff8ff;margin:0 0 3px;}
    .ml-side-panel .ml-side-sub{font-size:11px;color:#98abc0;margin:0 0 12px;}
    .ml-insight-item{display:flex;gap:9px;padding:12px 0;border-top:1px solid #2d3a4c;color:#d0deeb;
        font-size:12px;line-height:1.5;}
    .ml-insight-dot{margin-top:6px;background:#74e8b0;height:7px;width:7px;border-radius:50%;flex:0 0 7px;}
    .ml-health-row{display:flex;justify-content:space-between;align-items:center;gap:10px;padding:10px 0;border-top:1px solid #2d3a4c;
        font-size:11px;color:#afc1d6;}
    .ml-health-pill{font-size:10px;white-space:nowrap;color:#e6d39a;background:#443923;padding:5px 8px;border-radius:99px;}
    .ml-health-pill.good{color:#92f2bf;background:#194537;}.ml-health-pill.bad{color:#ffb4b9;background:#522a36;}
    .ml-health-foot{font-size:10px;color:#899bb2;margin-top:12px;line-height:1.5;}
    .dash-header{margin-top:17px;}.dash-header h3{font-size:20px;}
    .learning-tiles{grid-template-columns:repeat(auto-fit,minmax(min(100%,360px),1fr));gap:12px;}
    .learning-tile{background:linear-gradient(135deg,#192433,#121c29);border-color:#344155;}
    [data-testid='stExpander'],[data-testid='stDataFrame']{border-color:#354357;}
    [data-testid='stAlert']{padding:9px 12px;border-radius:12px;}
    .stButton>button[kind='primary']{background:#1e573f;color:#e7fff1;border:1px solid #347b58;box-shadow:none;}
    .stButton>button[kind='primary']:hover{background:#246a4d;}
    @media (max-width:1170px){.ml-stats{grid-template-columns:repeat(2,minmax(0,1fr));}
        .ml-row{grid-template-columns:minmax(135px,1.4fr) repeat(3,minmax(78px,1fr));}
        .ml-symbol{grid-column:1 / -1;}.ml-meta{justify-content:flex-start;}}
    @media (max-width:730px){.ml-stats{gap:8px;}.ml-stat{min-height:102px;padding:12px;}
        .ml-row{grid-template-columns:repeat(2,minmax(0,1fr));gap:11px;padding:14px;}
        .ml-symbol{grid-column:1/-1;}.ml-hero h1{font-size:26px;}.ml-meta{font-size:10px;}
        .ml-panel-title h2{font-size:21px;}.block-container{padding:1rem .85rem 2.5rem;}}
    @media(max-width:430px){.ml-stat-value{font-size:20px;}.ml-market{white-space:nowrap;}}
    """


def minimal_hero(stamp: str, is_open: bool) -> str:
    market = 'NSE Open' if is_open else 'NSE Closed'
    css = '' if is_open else ' closed'
    return (
        '<div class="ml-hero"><div><h1>NSE AI Research Terminal</h1>'
        '<div class="ml-sub">Sharper signals. Less noise. · Stocks, CE &amp; PE research</div></div>'
        '<div class="ml-meta">'
        f'<span class="ml-market{css}"><i class="ml-dot"></i>{html.escape(market)}</span>'
        '<i class="ml-divider"></i>'
        f'<span>{html.escape(str(stamp))} IST</span>'
        '<span class="ml-paper">Paper trading only</span></div></div>'
    )


def minimal_stats(items: list[tuple[str,str,str]]) -> str:
    icons = ('▣','↗','▱','◫')
    parts=[]
    for i, (label, value, description) in enumerate(items):
        pnl_tone = (' gain' if i==1 and str(value).startswith('+') else
                    ' loss' if i==1 and (str(value).startswith('-') or str(value).startswith('−') or '₹-' in str(value)) else '')
        parts.append(
            f'<div class="ml-stat"><div class="ml-stat-heading">'
            f'<span class="ml-stat-icon">{icons[i%len(icons)]}</span>{html.escape(str(label))}</div>'
            f'<div class="ml-stat-value{pnl_tone}">{html.escape(str(value))}</div>'
            f'<div class="ml-stat-sub">{html.escape(str(description))}</div></div>'
        )
    return '<div class="ml-stats">'+''.join(parts)+'</div>'


def minimal_prediction_row(name: str, decision: str, description: str,
                           entry: str, target: str, stop: str, timeline: str) -> str:
    tone = str(decision).lower() if decision in ('WATCH','AVOID') else 'consider'
    # Never interpolate unescaped ticker names, NSE-derived fields or exchange values.
    esc_ui = lambda v: html.escape(str(v),quote=True)
    symbol_icon = (name.split(' · ')[0][:1] or '•').upper()
    cells=[('Signal',f'<span class="ml-pill {tone}">{esc_ui(decision)}</span>'),
           ('Entry',f'<strong>{esc_ui(entry)}</strong>'),
           ('Target',f'<strong class="target">{esc_ui(target)}</strong>'),
           ('Stop-loss',f'<strong class="stop">{esc_ui(stop)}</strong>'),
           ('Timeline',f'<strong>{esc_ui(timeline)}</strong>')]
    return (
        f'<article class="ml-row"><div class="ml-symbol">'
        f'<span class="ml-monogram">{esc_ui(symbol_icon)}</span><div>'
        f'<div class="ml-ticker">{esc_ui(name)}</div><div class="ml-ticker-meta">{esc_ui(description)}</div></div></div>'
        +''.join(f'<div class="ml-col"><small>{esc_ui(label)}</small>{value}</div>' for label,value in cells)
        +'</article>'
    )


def minimal_sidebar_panels(idea_lines: list[str], health: list[tuple[str,str,str]],
                            footnote: str) -> str:
    bullet = ''.join('<div class="ml-insight-item"><span class="ml-insight-dot"></span><span>'
                     +html.escape(str(x))+'</span></div>' for x in idea_lines[:4])
    states={'good','warn','bad'}
    rows=''.join(f'<div class="ml-health-row"><span>{html.escape(str(label))}</span>'
                 f'<span class="ml-health-pill {tone if tone in states else "warn"}">{html.escape(str(value))}</span></div>'
                 for label,value,tone in health)
    return ('<div class="ml-side-panel"><h3>✧ AI Insights</h3>'
            '<p class="ml-side-sub">What the current research actually shows</p>'
            f'{bullet}</div><div class="ml-side-panel"><h3>⌁ System health</h3>'
            '<p class="ml-side-sub">Data checks and autonomous learning</p>'
            f'{rows}<div class="ml-health-foot">{html.escape(str(footnote))}</div></div>')

# Streamlit styles: always use a vetted dark palette, and native base="dark".
with st.sidebar:
    st.markdown('<div class="ml-brand"><div class="ml-brand-name"><span class="ml-brand-icon">↗</span>'
                '<span>NSE AI<br>Research Terminal</span></div>'
                '<div class="ml-brand-caption">PREDICT · LEARN · REVIEW</div></div>', unsafe_allow_html=True)
    palette_choice = st.selectbox('Appearance', DARK_THEMES, index=0, key='dark_theme_selector')
st.markdown(f'<style>{dark_theme_css(palette_choice)}\n{minimal_dark_css()}</style>', unsafe_allow_html=True)

now_ist = ist_now()
market_open = is_nse_market_open()
with st.sidebar:
    st.caption('SESSION CONTROLS')
    auto_refresh = st.toggle('Auto-scan during NSE hours', value=True, key='dark_auto_scan')
    refresh_mins = st.selectbox('Refresh interval', [15,30,60], index=0,
        format_func=lambda v:f'{v} minutes', key='dark_refresh_minutes')
    st.caption('Scans run while this Streamlit session is active. They are not an unattended background service.')
    with st.expander('⚙️ Trading assumptions'):
        st.write('Shared research budget: **₹10,00,000**')
        st.write(f'Share slots: **{MAX_OPEN_POSITIONS}** · Stock time-exit: **{MAX_HOLD_CALENDAR_DAYS} calendar days**')
        st.number_input('Brokerage per order (₹)', min_value=0.0, max_value=100.0,
                        value=DEFAULT_BROKERAGE_PER_ORDER_INR, key='brokerage_inr')
        st.number_input('Depository fee per sale (₹)', min_value=0.0, max_value=100.0,
                        value=DEFAULT_DP_CHARGE_INR, key='dp_charge_inr')
    with st.expander('🩺 Source and backup health'):
        st.caption(f"Database schema v{DB_SCHEMA_STATUS.get('version','?')}")
        st.caption(f"Minimum history expected: {HISTORY_EXPECTED_MIN_SHARES} shares / {HISTORY_EXPECTED_MIN_OPTIONS} options")
        st.caption(f"Supabase: {'connected' if supabase else 'not configured'}")
        st.caption(f"Model: {st.session_state.get('ml_model_source','Not yet checked')}")
        if st.session_state.get('db_error'):
            st.warning(str(st.session_state['db_error'])[:180])
    if st.button('Sign out', key='dark_signout', width="stretch"):
        st.session_state['password_correct'] = False
        st.rerun()

if auto_refresh and st_autorefresh:
    st_autorefresh(interval=int(refresh_mins)*60*1000, key='dark_session_refresh')

st.markdown(minimal_hero(now_ist.strftime('%d %b %Y · %H:%M'), market_open),unsafe_allow_html=True)

# Render three views only; never create additional top-level tabs with verbose tables.
# Expensive scans run at most once each refresh interval per browser session.
scan_interval = int(refresh_mins)*60
force_all = st.session_state.pop('dark_force_all', False)
force_shares = st.session_state.pop('dark_force_shares', False) or force_all
force_options = st.session_state.pop('dark_force_options', False) or force_all
force_audit = st.session_state.pop('dark_force_audit', False)
refresh_due = market_open and auto_refresh
if force_audit or (refresh_due and time.time()-st.session_state.get('dark_audit_epoch',0)>scan_interval):
    with st.spinner('Reviewing open paper positions...'):
        audit_and_reconcile_all_trades()
    st.session_state['dark_audit_epoch'] = time.time()

if force_shares or (refresh_due and time.time()-st.session_state.get('simple_share_epoch',0)>scan_interval):
    with st.spinner('Analysing NSE shares and updating qualifying paper research records...'):
        try:
            ideas, _ = run_predictions()
            st.session_state['simple_share_results'] = ideas
            st.session_state['simple_share_epoch'] = time.time()
            st.session_state['last_stock_scan'] = ist_now().strftime('%d %b · %H:%M IST')
        except Exception as error:
            logging.exception('Stock scan failed')
            st.session_state['dark_stock_error'] = type(error).__name__

if force_options or (refresh_due and time.time()-st.session_state.get('options_epoch',0)>scan_interval):
    with st.spinner('Reviewing call and put contracts with NSE quote checks...'):
        try:
            st.session_state['option_candidates'] = scan_option_candidates()
            st.session_state['options_epoch'] = time.time()
            st.session_state['last_options_scan'] = ist_now().strftime('%d %b · %H:%M IST')
        except Exception as error:
            logging.exception('Options scan failed')
            st.session_state['dark_options_error'] = type(error).__name__

# Automatic paper option recording is separate from displaying a CONSIDER card.
# A real fresh NSE quote, risk checks and budget are mandatory for a new record.
try:
    latest_option_idea = generate_daily_options_alpha()
except Exception as error:
    logging.exception('Option paper entry failed')
    latest_option_idea = {}
    st.session_state['dark_options_error'] = type(error).__name__

cap = get_portfolio_capacity()
integrity = get_history_integrity()
share_results = ranked_shares(st.session_state.get('simple_share_results',pd.DataFrame()))
option_results = option_sort(st.session_state.get('option_candidates',pd.DataFrame()))
try:
    journal_eq, journal_opt, journal_reviews = read_full_paper_ledger()
except Exception as error:
    logging.exception('Journal snapshot failed')
    journal_eq = journal_opt = journal_reviews = pd.DataFrame()
    st.error(f'Could not load complete paper history ({type(error).__name__}). Check Manage app → Logs; no missing rows are assumed closed.')

open_summary = portfolio_open_snapshot(journal_eq, journal_opt)
used = cap['equity_inr'] + cap['option_inr']
option_details = (f"{open_summary['shares']} shares · {open_summary['calls']} CE · "
                  f"{open_summary['puts']} PE" +
                  (f" · {open_summary['other_options']} unclassified" if open_summary['other_options'] else ''))
if open_summary['missing_count']:
    pnl_value = f"{short_money(open_summary['paper_pnl_inr'], 0)} partial"
    pnl_note = (f"Only {open_summary['valued_count']}/{open_summary['open_total']} open positions priced; "
                "not a verified total")
elif open_summary['open_total'] == 0:
    pnl_value = short_money(0, 0)
    pnl_note = 'No open positions · excludes closed trades'
else:
    pnl_value = ('+' if open_summary['paper_pnl_inr'] > 0 else '-' if open_summary['paper_pnl_inr'] < 0 else '') + short_money(abs(open_summary['paper_pnl_inr']), 0)
    pnl_note = 'Last recorded paper marks · unrealised, not live fills'
st.markdown(minimal_stats([
    ('Open positions',str(open_summary['open_total']),option_details),
    ('Open paper P/L',pnl_value,pnl_note),
    ('Allocated',short_money(used,0),'Across all active shares, CE and PE'),
    ('Unallocated',short_money(cap['remaining_inr'],0) if integrity['verified_count'] else 'Unverified',
     'Not necessarily deployable · ₹10 lakh combined cap'),
]), unsafe_allow_html=True)
if open_summary['missing_count']:
    st.warning(f"Paper P/L is incomplete: {open_summary['missing_count']} open records have no reliable latest price, "
               "allocation or marked return. The displayed total excludes them.")

if not integrity['verified_count']:
    st.markdown(dark_note('History integrity is not verified. New paper entries are paused until older records are restored. The figures may be incomplete.',True),unsafe_allow_html=True)
elif cap['fresh_slot_limit_reached']:
    st.markdown(dark_note(f"New share paper entries paused: {cap['open']} open positions versus {MAX_OPEN_POSITIONS} available slots. Existing history is preserved; research still runs."),unsafe_allow_html=True)
if st.session_state.get('price_feed_problem'):
    st.warning('⚠️ Price-feed outage: one or more paper positions could not be reconciled. Their capital stays reserved; no assumed sale was recorded.')
if st.session_state.get('intraday_feed_problem'):
    affected = ', '.join(st.session_state.get('intraday_feed_missing_tickers',[])[:5])
    st.warning(f'⚠️ Minute-price coverage is incomplete for open paper positions ({affected or "some shares"}). '
               'A temporary stop/target touch may not be detectable; affected positions remain unverified until quotes recover.')
if not ANNOUNCEMENTS_AVAILABLE:
    st.caption('⚠️ Corporate filing/news checker unavailable. Technical recommendations are incomplete without those checks.')

# Independent background daily learner (read-only). It never updates existing
# historical journals or replaces the old live model. If setup is incomplete,
# the original dashboard and paper monitoring still work as before.
@st.cache_data(ttl=240, show_spinner=False)
def read_independent_nse_research() -> dict:
    # Prefer isolated server-side credentials for the private autonomous tables.
    # Keep the ordinary ledger client on SUPABASE_KEY unchanged.
    private_key = _secret('SUPABASE_AI_READ_KEY')
    if private_key:
        try:
            snapshot = private_research_snapshot(_secret('SUPABASE_URL'), private_key)
            all_rows = snapshot['forecasts']
            last_date = str(all_rows[0].get('forecast_for','')) if all_rows else None
            fresh = [r for r in all_rows if str(r.get('forecast_for')) == last_date]
            champion = snapshot['champion']
            return {"configured": True, "champion": champion,
                    "run": snapshot['run'],
                    "forecasts": valid_forecasts(fresh, champion.get('model_id'), max_cards=5)}
        except PrivateResearchReadError:
            return {"configured": False, "forecasts": [], "champion": {}, "run": {}}
    if not supabase:
        return {"configured": False, "forecasts": [], "champion": {}, "run": {}}
    try:
        settings = supabase.table('nse_ai_settings').select('value').eq('key','champion').limit(1).execute()
        champion = settings.data[0]['value'] if settings.data else {}
        latest_run = (supabase.table('nse_ai_runs').select('run_id,status,finished_at,started_at,kind')
                      .eq('kind','daily').order('started_at',desc=True).limit(1).execute())
        rows = (supabase.table('nse_ai_forecasts')
                .select('forecast_for,ticker,model_id,entry_price,target_price,stop_loss,target_first_probability,conservative_ev_pct,time_estimate,horizon_sessions,decision')
                .order('forecast_for',desc=True).limit(80).execute())
        all_rows = rows.data or []
        last_date = str(all_rows[0].get('forecast_for','')) if all_rows else None
        fresh = [r for r in all_rows if str(r.get('forecast_for')) == last_date]
        return {"configured": True, "champion": champion or {},
                "run": (latest_run.data or [None])[0],
                "forecasts": valid_forecasts(fresh, (champion or {}).get('model_id'), max_cards=5)}
    except Exception:
        # Old Supabase schema / missing table / anon-key RLS: no UI crash and
        # no inferred ML prediction. Deployment instructions explain migration.
        return {"configured": False, "forecasts": [], "champion": {}, "run": {}}

independent_research = read_independent_nse_research()
if independent_research['configured']:
    background_health = worker_status(independent_research['run'])
    st.caption('🧠 Independent daily learning: ' + background_health['label'] +
               ' · Research-only; never a broker execution')
    if background_health['state'] in ('STALE','FAILED'):
        st.warning('Scheduled daily learning is stale or failed. Check GitHub Actions and Supabase worker-run history; do not assume forecasts have refreshed.')
else:
    st.caption('🧠 Independent background learning: unavailable. Add SUPABASE_AI_READ_KEY to Streamlit Secrets for private-table access, or check the NSE project/key and worker logs. Existing research remains unchanged.')

page_preds, page_learning, page_ledger = st.tabs(['✦ Predictions','🧠 AI Learning','📒 Trade Ledger'])

with page_preds:
    if not market_open:
        st.markdown(dark_note('NSE is closed: reference prices may be stale; new paper entries are paused.'),unsafe_allow_html=True)
    if st.session_state.get('dark_stock_error'):
        st.warning('Last share scan failed: '+str(st.session_state['dark_stock_error']))
    if st.session_state.get('dark_options_error'):
        st.warning('Last options scan failed: '+str(st.session_state['dark_options_error']))
    p1,p2,p3 = st.columns([1.55, 1.0, 1.45],gap="small")
    with p1:
        product_filter = st.segmented_control('Market', ['All','Stocks','Calls (CE)','Puts (PE)'], default='All', key='dark_product_filter')
    with p2:
        decision_filter = st.selectbox('Signal', ['CONSIDER','ALL','WATCH','AVOID'], key='dark_decision_filter')
    with p3:
        keyword = st.text_input('Search',placeholder='Symbol or sector…', key='dark_prediction_search')
    with st.expander('⚙️ Scan controls and timing', expanded=False):
        a,b,c = st.columns([1,1,1])
        with a:
            if st.button('🔍 Scan stocks',key='dark_scan_stocks',width="stretch"):
                st.session_state['dark_force_shares'] = True
                st.rerun()
        with b:
            if st.button('🎯 Scan CE / PE',key='dark_scan_options',width="stretch"):
                st.session_state['dark_force_options'] = True
                st.rerun()
        with c:
            if st.button('↻ Refresh everything',type='primary',key='dark_scan_all',width="stretch"):
                st.session_state['dark_force_all'] = True
                st.session_state['dark_force_audit'] = True
                st.rerun()
        horizon = st.selectbox('Stock timing study', [3,5,7], index=1,
                               format_func=lambda x:f'{x} trading sessions', key='dark_horizon')

    stock_view = (share_results.copy() if not share_results.empty else pd.DataFrame())
    if not stock_view.empty and product_filter not in ('All','Stocks'):
        stock_view = stock_view.iloc[0:0]
    if not stock_view.empty and decision_filter!='ALL':
        stock_view = stock_view[stock_view['Decision']==decision_filter]
    if not stock_view.empty and keyword.strip():
        term = re.escape(keyword.strip())
        stock_view = stock_view[stock_view['Ticker'].astype(str).str.contains(term,case=False,na=False) |
                                stock_view['Sector'].astype(str).str.contains(term,case=False,na=False)]

    opt_view = option_results.copy() if not option_results.empty else pd.DataFrame()
    if not opt_view.empty and product_filter=='Stocks':
        opt_view = opt_view.iloc[0:0]
    if not opt_view.empty and product_filter=='Calls (CE)':
        opt_view = opt_view[opt_view['Type']=='CE']
    if not opt_view.empty and product_filter=='Puts (PE)':
        opt_view = opt_view[opt_view['Type']=='PE']
    if not opt_view.empty and decision_filter!='ALL':
        opt_view = opt_view[opt_view['Decision']==decision_filter]
    if not opt_view.empty and keyword.strip():
        opt_view = opt_view[opt_view['Stock'].astype(str).str.contains(re.escape(keyword.strip()),case=False,na=False)]

    st.caption(f"{len(stock_view)} shares · {len(opt_view)} options matching filters · "
               f"Stocks {st.session_state.get('last_stock_scan','not scanned')} · "
               f"Options {st.session_state.get('last_options_scan','not scanned')}")
    main_predictions, right_insights = st.columns([3.15, 1.2], gap="large")
    with main_predictions:
        st.markdown('<div class="ml-panel-title"><div><h2>AI Predictions</h2>'
                    '<p>Stock, call and put ideas · Entry, target, stop-loss and estimated time</p></div>'
                    f'<span class="ml-count">{len(stock_view)} stocks · {len(opt_view)} options</span></div>',
                    unsafe_allow_html=True)
        shown_stocks = stock_view.head(5) if not stock_view.empty else pd.DataFrame()
        for _, r in shown_stocks.iterrows():
            ticker = str(r['Ticker'])
            entry,target,stop = float(r['Price (₹)']),float(r['Target (₹)']),float(r['Stop Loss (₹)'])
            estimate = stock_timeline(ticker, entry,target,stop,horizon)
            subtitle = (f"{estimate.target_hits}/{estimate.trials} historical windows hit target first"
                        if estimate.trials else 'Not enough validated historical windows')
            action = ('Qualifies for research, but new paper entries are paused by portfolio constraints.'
                      if r['Decision']=='CONSIDER' and not cap['ok'] else str(r.get('Rejection Reason') or 'Research filters passed'))
            st.markdown(minimal_prediction_row(ticker,str(r['Decision']),str(r['Sector']),
                short_money(entry),short_money(target),short_money(stop),estimate.label),unsafe_allow_html=True)
            with st.expander(f'Details · {ticker}',expanded=False):
                st.caption(action)
                st.caption(subtitle)
                st.write(f"Current study: **{horizon} trading sessions**. {estimate.note}")
                st.write(f"Historical target-first frequency: **{f'{estimate.target_rate_pct:.1f}%' if estimate.target_rate_pct is not None else 'Unavailable'}**. "
                         'This is not a validated AI win probability.')
                st.write('Model score:',f"{_num(r.get('Adjusted Score')):.1f}",
                         '· Expected value after modelled costs:',f"{_num(r.get('Expected value')):+.2f}%")
                st.caption('Entry and stop prices may gap past their levels; slippage and fees affect realised outcomes. The app currently exits stock ideas after 7 CALENDAR days, which may be shorter than a 7-session study.')
        if product_filter in ('All','Stocks') and stock_view.empty:
            st.info('No stocks match this filter. Run the stock scan to refresh research; a missing idea is not a buy signal.')
        if len(stock_view)>len(shown_stocks):
            st.caption(f'Showing the first {len(shown_stocks)} ranked shares of {len(stock_view)}. Narrow your search or open the table below.')

        shown_options = opt_view.head(4) if not opt_view.empty else pd.DataFrame()
        for _, r in shown_options.iterrows():
            kind = str(r['Type'])
            premium = float(r['Buy premium (₹)'])
            stop = premium * OPTION_STOP_MULT
            target = premium * OPTION_TARGET_MULT
            st.markdown(minimal_prediction_row(f"{r['Stock']} · {kind}",str(r['Decision']),
                f"Strike {short_money(r['Strike (₹)'])} · expiry {r['Expiry']}",
                short_money(premium),short_money(target),short_money(stop),
                f"{r['Days left']} days to expiry"),unsafe_allow_html=True)
            with st.expander(f'Contract details · {r["Stock"]} {kind}',expanded=False):
                st.caption(str(r["Why"]))
                st.caption("No validated day-to-target forecast for this option premium.")
                st.write(f"Bid {short_money(r['Sell premium (₹)'])} · Ask {short_money(premium)} · "
                         f"Lot {int(r['Lot'])} · Spread {_num(r['Spread (%)']):.1f}% · "
                         f"Open interest {int(r['Open interest']):,}")
                st.write(f"Spot break-even at expiry: **{short_money(r['Breakeven at expiry (₹)'])}**. "
                         f"Maximum long-option premium risk: **{short_money(r['Maximum premium loss (₹)'])}**.")
                st.caption('Premium target/stop are research exit guidelines, not guaranteed executions. Long stock options may have physical settlement obligations; review broker and expiry rules.')
        if product_filter in ('All','Calls (CE)','Puts (PE)') and opt_view.empty:
            if not option_results.empty and decision_filter == 'CONSIDER':
                held = option_results if product_filter == 'All' else option_results[option_results['Type']==('CE' if product_filter=='Calls (CE)' else 'PE')]
                st.info(f'No CE/PE contracts passed the CONSIDER rules. {len(held)} quoted contracts have ' 
                        'WATCH/AVOID outcomes; select Research signal → ALL to inspect the reasons. ' 
                        'No theoretical option premium is shown as an exchange quote.')
            else:
                st.info('No verified CE/PE contracts for the current filters. The app will not present a theoretical option premium as a tradable quote.')
        if st.session_state.get('opt_note'):
            st.caption('CE/PE paper entry: ' + str(st.session_state['opt_note']))

        if product_filter in ('All','Stocks') and independent_research['forecasts']:
            with st.expander('🧠 Background AI daily share forecasts · time-tested research', expanded=False):
                st.caption('These are separate EOD research forecasts from a time-tested model, NOT live execution quotes. '
                           'Their probabilities are model estimates, not guaranteed or proven live calibration. '
                           'They do not replace the original stock ranker or automatically open paper positions.')
                for auto in independent_research['forecasts']:
                    entry, target, stop = float(auto['entry_price']),float(auto['target_price']),float(auto['stop_loss'])
                    model_score = float(auto['target_first_probability'])*100
                    st.markdown(dark_card(str(auto['ticker']), str(auto['decision']),
                        f"Daily ML as of {auto['forecast_for']} · model {str(auto['model_id'])[:20]}", [
                        ('EOD entry reference', short_money(entry)),
                        ('Target · historical model', short_money(target)),
                        ('Stop-loss · risk', short_money(stop)),
                        ('Target-first estimate', f'{model_score:.1f}%')],
                        f"Conservative modelled value {float(auto['conservative_ev_pct']):+.2f}% after assumed costs",
                        f"Horizon {auto['horizon_sessions']} sessions · target history: {auto['time_estimate']}",
                        'The latest close can gap before next entry. Missing CE/PE quotes are unaffected.'),
                        unsafe_allow_html=True)

    with right_insights:
        stock_consider = int((share_results['Decision']=='CONSIDER').sum()) if not share_results.empty else 0
        option_consider = int((option_results['Decision']=='CONSIDER').sum()) if not option_results.empty else 0
        checked = st.session_state.get('option_scan_health') or {}
        background = worker_status(independent_research.get('run')) if independent_research['configured'] else {'state':'NOT_CONFIGURED','label':'Not configured'}
        insights = [
            f'{stock_consider} stock research ideas meet current CONSIDER checks; filings may remain unverified.',
            (f'{option_consider} CE/PE contracts meet the current research checks.' if option_consider else
             'No validated CE/PE CONSIDER quote is available; no premium prediction is invented.'),
            (f'{cap["open"]} of {MAX_OPEN_POSITIONS} share slots occupied; new share entries are paused.' if cap['fresh_slot_limit_reached'] else
             f'{cap["open"]} of {MAX_OPEN_POSITIONS} share slots occupied; risk limits still apply.'),
            ('Stock timing windows come from historical paths, not guaranteed target dates.'),
        ]
        option_valid_quotes = int(checked.get('valid_quotes',0) or checked.get('valid_bid_ask',0) or 0)
        health = [
            ('Daily learner',str(background.get('label') or background['state']),
                'good' if background['state']=='HEALTHY' else 'bad' if background['state'] in ('FAILED','STALE') else 'warn'),
            ('Saved history','Verified' if integrity['verified_count'] else 'Recovery needed',
                'good' if integrity['verified_count'] else 'bad'),
            ('CE/PE quote checks','Quotes reviewed' if option_valid_quotes or len(option_results)>0 else 'Unverified',
                'good' if option_valid_quotes or len(option_results)>0 else 'warn'),
            ('Minute candles','Incomplete' if st.session_state.get('intraday_feed_problem') else 'No gaps flagged',
                'warn' if st.session_state.get('intraday_feed_problem') else 'good'),
            ('Corporate filings','Unavailable' if not ANNOUNCEMENTS_AVAILABLE else 'Module available',
                'warn' if not ANNOUNCEMENTS_AVAILABLE else 'good'),
        ]
        st.markdown(minimal_sidebar_panels(insights,health,
            'These labels report app checks, not verified prediction accuracy, an exchange service guarantee, or brokerage execution.'),
            unsafe_allow_html=True)

    with st.expander('🩺 Scan quality / full candidate list'):
        if not share_results.empty:
            st.caption(f"Share universe: {st.session_state.get('scan_stats',{}).get('listed','?')} NSE symbols. "
                       'Only some have usable data and pass the shortlist; a scan may be partial.')
            cols = ['Ticker','Sector','Decision','Price (₹)','Target (₹)','Stop Loss (₹)','Expected value','Rejection Reason']
            st.dataframe(share_results[[x for x in cols if x in share_results.columns]],hide_index=True,width="stretch")
        else:
            st.caption('No share candidate table saved in this session.')
        health = st.session_state.get('option_scan_health') or {}
        # Report both the scan outcome and HTTP statuses. A 200 response alone
        # does not imply fresh, usable, exchange-validated option contracts.
        st.caption(st.session_state.get('option_scan_note','No option scan diagnostics yet.'))
        http_note = option_http_note(health)
        if http_note:
            st.caption(http_note)
        if not option_results.empty:
            cols = ['Stock','Type','Decision','Expiry','Strike (₹)','Buy premium (₹)','Cost (₹)','Why']
            st.dataframe(option_results[[x for x in cols if x in option_results.columns]],hide_index=True,width="stretch")
        if not ANNOUNCEMENTS_AVAILABLE:
            st.warning('NSE corporate filings module is unavailable. News checks are incomplete; treat technical CONSIDER scores cautiously.')

with page_learning:
    champion_live = independent_research.get('champion') or {}
    if champion_live:
        stats = champion_live.get('metrics') or {}
        st.caption(f"Independent daily market AI · champion {str(champion_live.get('model_id','?'))[:24]} · "
                   f"trained through {champion_live.get('trained_until','?')} · "
                   f"held-out Brier {stats.get('brier','?')} (lower is better). "
                   "This uses newly labelled daily market history, separately from your trade journal.")
    else:
        st.caption('Independent daily market learner: no promoted challenger yet.' if independent_research['configured'] else 'Independent daily market learner is not configured or cannot be read.')
    st.markdown(dark_header('AI Learning — one trade at a time',
        'Individual results, observations and tests the model could learn from · no graphs'),
        unsafe_allow_html=True)
    if journal_eq.empty:
        st.info('No stock training history is available locally. Check the Ledger recovery tools.')
    else:
        totals = stock_history_summary(journal_eq)
        state_raw = get_meta('retrain_state','{}')
        try:
            train_state = json.loads(state_raw or '{}')
        except (ValueError,TypeError):
            train_state = {}
        st.caption(f"{totals['finished']} completed recorded outcomes · "
                   f"{totals['potential_training']} possible entry-feature examples (not all validated) · "
                   f"Last model promotion: {train_state.get('last_promoted','None recorded')}")
        st.caption('Each tile is based on one journal record. It explains a possible future test, not a proven reason for a win/loss or an automatic model update.')
        # One automatic feed covering different shares: no symbol picker, filters
        # or card-count dropdown. A card describes a genuine finished journal
        # outcome; an open trade cannot be described as a learned win/loss.
        completed_feed = latest_completed_by_stock(journal_eq)
        if not completed_feed:
            st.info('No completed stock outcomes yet. Current open trades will appear here after their recorded outcomes are confirmed.')
        else:
            if 'learning_feed_visible' not in st.session_state:
                st.session_state['learning_feed_visible'] = 15
            visible_count = min(max(15, int(st.session_state['learning_feed_visible'])), len(completed_feed))
            records = completed_feed[:visible_count]
            st.markdown(dark_header('Latest stock learning observations',
                f'{visible_count} of {len(completed_feed)} distinct stocks · latest completed result per stock · newest first'),
                unsafe_allow_html=True)
            st.caption('Each card shows one actual closed paper trade for a different stock. These are observations and proposed future tests, not proof the AI has already learned the pattern.')
            cards = [dark_learning_tile(build_learning_tile(rec)) for rec in records]
            st.markdown('<div class="learning-tiles">' + ''.join(cards) + '</div>', unsafe_allow_html=True)
            if visible_count < len(completed_feed):
                if st.button('↓ Show 15 more stocks', key='show_more_learning_feed', width='stretch'):
                    st.session_state['learning_feed_visible'] = min(visible_count + 15, len(completed_feed))
                    st.rerun()
            with st.expander('Exact trade prices and saved forecasts for the cards above'):
                for n, rec in enumerate(records):
                    obs = describe_trade(rec)
                    stock_name = str(rec.get('ticker') or 'Unknown stock')
                    st.markdown(f"**{esc(stock_name)} · {str(rec.get('date_str') or '')[:10]} · {obs['outcome']}**")
                    st.caption(f"Entry {short_money(rec.get('entry_price'))} · "
                               f"Target {short_money(rec.get('target_price'))} · "
                               f"Stop {short_money(rec.get('stop_loss'))} · "
                               f"Held {obs['days_held'] if obs['days_held'] is not None else 'unknown'} calendar days")
                    st.caption('Original forecast: ' + forecast_review(rec))
                    if n < len(records)-1:
                        st.divider()
        with st.expander('🧠 Model training status and validation'):
            st.write(f'Minimum: **{MIN_SETTLED_TO_RETRAIN}** suitable, validated share outcomes. '
                     f'Next retraining attempts after roughly **{RETRAIN_STEP_INTERVAL}** additional eligible cases.')
            st.write('Last retraining attempt:', train_state.get('last_attempt_date','Not eligible yet'))
            st.write('Last challenger promotion:', train_state.get('last_promoted','None recorded'))
            st.caption('One paper trade does not prove which factor caused its result. Missing original features or forecasts are not recreated. '
                       'A replacement model is promoted only after time-ordered validation against the existing model.')

with page_ledger:
    st.markdown(dark_header('Trade ledger','All paper stocks, calls and puts · open and closed'),unsafe_allow_html=True)
    if journal_eq.empty and journal_opt.empty:
        st.warning('No trading history available from this local database. Use recovery below. Do not reset your existing database.')
    st.markdown(dark_stats([
        ('Share records',str(len(journal_eq)),'Includes old and current entries'),
        ('Active shares',str(cap['open']),f"Current entry limit {MAX_OPEN_POSITIONS}"),
        ('Option records',str(len(journal_opt)),'Live-quoted and older estimated-price ideas'),
        ('Reviews saved',str(len(journal_reviews)),'Newer CE/PE quote checks'),
    ]),unsafe_allow_html=True)
    ledger_kind = st.segmented_control('Ledger type', ['Stocks','Calls & Puts','Option research reviews'],default='Stocks', key='dark_ledger_type')
    l1,l2,l3 = st.columns([1.1,1.5,1])
    with l1:
        ledger_status = st.selectbox('Status', ['ALL','OPEN','CLOSED','WINS','LOSSES'],key='dark_ledger_status')
    with l2:
        ledger_search = st.text_input('Search your ledger',placeholder='Symbol or contract…',key='dark_ledger_search')
    with l3:
        visible_rows = st.selectbox('Rows shown',[15,30,75,150],key='dark_rows_shown')
    if ledger_kind=='Stocks':
        view = journal_eq.copy()
        filter_col = 'ticker'
    elif ledger_kind=='Calls & Puts':
        view = journal_opt.copy()
        filter_col = 'option_contract'
    else:
        view = journal_reviews.copy()
        filter_col = 'symbol'
    if not view.empty:
        if 'status' in view.columns:
            outcomes = view['status'].apply(classify_outcome)
            if ledger_status=='OPEN':
                view=view[outcomes=='ACTIVE']
            elif ledger_status=='CLOSED':
                view=view[outcomes!='ACTIVE']
            elif ledger_status=='WINS':
                view=view[outcomes=='WIN']
            elif ledger_status=='LOSSES':
                view=view[outcomes=='LOSS']
        elif ledger_status != 'ALL':
            st.caption('This research-review table has no open/closed trade statuses.')
        if ledger_search.strip() and filter_col in view.columns:
            view = view[view[filter_col].astype(str).str.contains(re.escape(ledger_search.strip()),case=False,na=False)]
    st.caption(f"{len(view):,} matching records. Showing up to {visible_rows}; CSV export includes every match. No history is removed by filters.")
    if view.empty:
        st.info('No matching ledger rows. Clear filters or check recovery below.')
    else:
        if ledger_kind=='Stocks':
            friendly_cols = ['date_str','ticker','status','entry_price','target_price','stop_loss','latest_price','shares','capital_allocated','pnl_pct','exit_timestamp']
            display_names = {'date_str':'Date','ticker':'Share','status':'Status','entry_price':'Entry ₹','target_price':'Target ₹','stop_loss':'Stop ₹','latest_price':'Latest ₹',
                             'shares':'Qty','capital_allocated':'Allocated ₹','pnl_pct':'Paper P/L %','exit_timestamp':'Closed at'}
        elif ledger_kind=='Calls & Puts':
            friendly_cols = ['date_key','option_contract','status','entry_premium','target_premium','stop_loss_premium','current_option_price','lot_size','total_capital','pnl_pct','last_audited']
            display_names = {'date_key':'Date','option_contract':'CE / PE Contract','status':'Status','entry_premium':'Entry premium ₹','target_premium':'Target ₹','stop_loss_premium':'Stop ₹',
                             'current_option_price':'Last premium ₹','lot_size':'Lot','total_capital':'Allocated ₹','pnl_pct':'Paper P/L %','last_audited':'Checked at'}
        else:
            friendly_cols = ['checked_at','symbol','side','decision','reason']
            display_names = {'checked_at':'Checked at','symbol':'Stock','side':'CE / PE','decision':'Decision','reason':'Reason'}
        st.dataframe(view[[c for c in friendly_cols if c in view.columns]].head(visible_rows).rename(columns=display_names),
                     hide_index=True,width="stretch",height=min(550,90+36*min(visible_rows,len(view))))
        st.download_button('↓ Export all matching records (CSV)',data=view.to_csv(index=False).encode('utf-8'),
                           file_name='nse_'+ledger_kind.lower().replace(' ','_').replace('&','and')+'_ledger.csv',
                           mime='text/csv',key='dark_ledger_export')
    if st.button('↻ Reconcile open paper positions', key='dark_reconcile', width="content"):
        st.session_state['dark_force_audit'] = True
        st.rerun()
    # Cloud sync check is informational: the restore counter may be older
    # than newly written local records. Never delete local history to match it.
    cloud_info = st.session_state.get('hydrate_info') or {}
    if supabase and 'remote_options' in cloud_info and len(journal_opt) > _int(cloud_info.get('remote_options')):
        st.warning(
            f"The local ledger contains {len(journal_opt):,} option ideas, while the last Supabase "
            f"check saw {_int(cloud_info.get('remote_options')):,}. This may reflect an unsynced "
            "idea or an older cloud snapshot. Export a backup and verify cloud sync before a restart."
        )
    if st.session_state.get('history_close_conflicts'):
        st.info(f"Cloud and local timestamps/statuses differ on "
                f"{st.session_state['history_close_conflicts']} closed record(s). "
                "Local completed outcomes were preserved; verify the original execution times from the ledger or source backups.")
    render_history_recovery()
    st.caption('No broker orders. Stop-loss fills and estimated model targets are uncertain. Streamlit session auto-refresh is NOT an unattended 10-year service.')

st.caption('NSE market research · Paper capital only · Uncertain forecasts · Never treat a CONSIDER label as a guaranteed trade or return')
