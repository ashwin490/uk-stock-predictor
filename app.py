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
from datetime import datetime, time as dtime, timedelta
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

# ---- Whole-market scan (every FTSE 350 share + liquid AIM shares) ----
SCAN_WORKERS = 10                   # parallel price downloads
SCAN_TIME_BUDGET_SEC = 240          # stop downloading after this long; the rest is checked next time
SWEEP_TTL_SEC = 900                 # re-download a tradable share's prices at most every 15 min
QUIET_TTL_SEC = 21600               # thinly traded shares are re-checked every 6 hours
FRESH_PRICE_TTL_SEC = 60            # finalists get a fresh price (≤1 min old) before any buy
FRESH_PRICE_MAX_AGE_SEC = 180       # ...and are skipped if that fresh download failed (never buy on an old price)
MAX_DEEP_CHECKS = 30                # finalists that get the full news / memory / market check
DEEP_CHECK_MARGIN = 12              # finalists must be within this many points of the confidence bar
MAX_ATR_PCT = 8.0                   # skip shares that swing more than this % on a normal day
PULSE_REFRESH_SEC = 1800            # after today's picks are used up, still refresh the market pulse every 30 min
MIN_EV_PCT = 0.0                    # a pick's expected value after ALL costs must be above this (in %)

# ---- Evidence learner: learns which market conditions really win, from YOUR finished trades ----
EVIDENCE_MIN_DECIDED = 40           # clear win/loss trades needed before it changes any score
EVIDENCE_MIN_BUCKET = 6             # trades needed in a group before that group counts
EVIDENCE_PRIOR_TRADES = 10          # shrinks small groups toward the overall win rate
EVIDENCE_HALF_LIFE_DAYS = 90        # older trades count less (markets change)
EVIDENCE_POINTS_PER_PCT = 0.6       # 10 percentage points better win rate -> +6 confidence points
EVIDENCE_FACTOR_CAP = 5             # max points any one factor can add or take away
EVIDENCE_TOTAL_CAP = 10             # max points all factors together can add or take away
EVIDENCE_BACKFILL_PER_RUN = 40      # past trades whose context is rebuilt from price history per check

# ---- Options learner (separate from the share AI; it only ever changes the options idea) ----
OPT_LEARN_MIN_TRADES = 10
OPT_LEARN_WINDOW_DAYS = 60
OPT_REST_THRESHOLD = 0.75
OPT_BASE_TREND_BAR = 68.0           # the options idea's normal minimum trend strength
OPT_MAX_TREND_BAR = 88.0

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

FTSE100_BUILTIN = """III|Financial services
ABDN|Financial Services
ADM|Insurance
AAF|Telecommunications services
ALW|Investment Trusts
AAL|Mining
ANTO|Mining
ABF|Food & tobacco
AZN|Pharmaceuticals & biotechnology
AUTO|Media
AV|Life insurance
BAB|Aerospace & defence
BA|Aerospace & defence
BBY|Construction & Materials
BARC|Banks
BTRW|Household goods & home construction
BP|Oil & gas producers
BATS|Tobacco
BLND|Real estate
BT-A|Telecommunications services
BNZL|Support services
BRBY|Personal goods
CNA|Multiline utilities
CCEP|Beverages
CCH|Beverages
CPG|Support services
CCC|Software & Computer Services
CTEC|Health care equipment & supplies
CRDA|Chemicals
DCC|Support services
DGE|Beverages
DPLM|Industrial Support services
EZJ|Travel and Leisure
EDV|Mining
EXPN|Support services
FCIT|Collective investments
FRES|Mining
GAW|Leisure Goods
GLEN|Mining
GSK|Pharmaceuticals & biotechnology
HLN|Pharmaceuticals & biotechnology
HLMA|Electronic equipment & parts
HSX|Non-life Insurance
HWDN|Homebuilding & construction supplies
HSBA|Banks
ICG|Financial services
IGG|Financial services
IHG|Travel & leisure
IMI|Industrial engineering
IMB|Tobacco
INF|Media
IAG|Travel & leisure
ITRK|Support services
INVP|Financial Services
ITH|Oil & Gas Producers
JD|General retailers
BGEO|Banking Services
KGF|Retailers
LAND|Real estate investment trusts
LGEN|Life insurance
LLOY|Banks
LMP|Real Estate Investment Trusts
LSEG|Financial services
MNG|Financial services
MKS|Food & drug retailing
MRO|Aerospace & defence
MTLN|Multiline utilities
NG|Multiline utilities
NWG|Banks
NXT|General retailers
PSON|Media
PSH|Financial services
PCT|Investment trusts
PRU|Life insurance
RKT|Household goods & home construction
REL|Media
RTO|Support services
RIO|Mining
RR|Aerospace & defence
SGE|Software & computer services
SBRY|Food & drug retailing
SMT|Collective investments
SGRO|Real estate investment trusts
SVT|Multiline utilities
SHEL|Oil & gas producers
SMIN|General industrials
SN|Health care equipment & supplies
SPX|Industrial engineering
SSE|Electrical utilities & independent power producers
STAN|Banks
SDLF|Life insurance
STJ|Financial services
TSCO|Food & drug retailing
BBOX|Real Estate Investment Trusts
ULVR|Personal goods
UU|Multiline utilities
VOD|Mobile telecommunications
WEIR|Industrial goods and services
WTB|Retail hospitality
WPP|Media"""

FTSE250_BUILTIN = """3IN|Financial Services
FOUR|Media
AAS|Investment Trusts
ASL|Investment Trusts
AEP|Food Producers
ALFA|Software and Computer Services
ATT|Investment Trusts
AMRQ|Industrial Metals and Mining
AO|Retail
APN|Food Producers
ASHM|Financial Services
AIE|Investment Trusts
ASC|Retailers
ATYM|Industrial Metals and Mining
ATG|Software and Computer Services
AGT|Investment Trusts
AVON|Industrial Goods and Services
BME|Retailers
BGFD|Investment Trusts
USA|Investment Trusts
BCG|Consumer Digital Services
BNKR|Investment Trusts
BAG|Beverages
AJB|Financial Services
BWY|Home Construction
BKG|Household goods & home construction
BHMG|Hedge Funds
BYG|Real Estate Investment Trusts
BPCR|Investment Trusts
BRGE|Investment Trusts
BRSC|Investment Trusts
BRWM|Investment Trusts
BMY|Media
BOY|Industrial Engineering
BREE|Construction & Materials
BPT|Financial Services
BUT|Investment Trusts
BYIT|Software and Computer Services
CLDN|Investment Trusts
CGT|Investment Trusts
CWR|Alternative Energy
CHG|Aerospace & Defence
CSN|Life Insurance
CTY|Investment Trusts
CKN|Industrial Transportation
CBG|Financial Services
CMCX|Financial Services
COA|Consumer Staples
CORD|Investment Trusts
COST|Engineering and Construction
CWK|Food Producers
CURY|Retailers
CVSG|Health Care
DLN|Real Estate Investment Trusts
DSCV|Support Services
DOM|Travel & Leisure
DRX|Electricity
DOCS|Retailers
DNLM|Retailers
EDIN|Investment Trusts
EWI|Investment Trusts
ELM|Chemicals
ENOG|Oil & Gas Producers
ENT|Travel & leisure
ESCT|Investment Trusts
EWG|Financial Services
FCSS|Investment Trusts
FEML|Investment Trusts
FEV|Investment Trusts
FSV|Investment Trusts
FGT|Equity Investments
FGP|Travel & Leisure
FGEN|Investment Trusts
FSG|Investment Banking and Brokerage Services
FRAS|Retailers
FCH|Financial Services
GFRD|Construction
GAMA|Telecommunications
GCP|General Financial
GEN|Construction & Materials
GNS|Pharmaceuticals & Biotechnology
DATA|Industrial Goods and Services
GSCT|Investment Trusts
GDWN|General Industrials
GFTU|Support Services
GRI|Real Estate Investment & Services
GPE|Real Estate Investment Trusts
UKW|Gas, Water & Multiutilities
GNC|Food, Beverage and Tobacco
GRG|Food & Drug Retailers
HFD|Retailers
HMSO|Real Estate Investment Trusts
HANA|Investment Trusts
HBR|Oil & Gas Producers
HVPE|Equity Investments
HWG|Real Estate
HAS|Support Services
HTWS|Telecomms
HFEL|Investment Trusts
HSL|Investment Trusts
HRI|Investment Trusts
HGT|Investment Trusts
HICL|Investment Trusts
HIK|Pharmaceuticals and Biotechnology
HILS|Industrial Engineering
HFG|Food Producers
HOC|Basic Resources
BOWL|Travel and leisure
HTG|Oil, Gas and Coal
ICGT|Investment Trusts
INCH|Retailers
IHP|Financial Services
INPP|Financial Services
IWG|Support Services
IAD|Investment Trusts
IPO|Financial Services
ITV|Media
JMAT|Chemicals
JSG|Industrial Support Services
JAM|Investment Trusts
JCH|Investment Trusts
JEMI|Investment Trusts
JMGI|Investment Trusts
JEDT|Investment Trusts
JEGI|Investment Trusts
JGGI|Investment Trusts
JFJ|Investment Trusts
JUP|Financial Services
KNOS|Software & Computer Services
KLR|Construction & Materials
KIE|Construction & Materials
LRE|Nonlife Insurance
LWDB|Investment Trusts
EMG|Financial Services
MEGP|Leisure Goods
MRC|Collective Investments
MRCH|Investment Trusts
MNO|Industrial Metals and Mining
MTRO|Banks
PAGE|Support Services
MAB|Travel & Leisure
MTO|Support Services
GROW|Financial Services
MNDI|Containers & packaging
MNKS|Investment Trusts
MONY|Media
MOON|Retailers
MGAM|Electronic & Electrical Equipment
MGNS|Construction & Materials
MUT|Investment Trusts
MYI|Investment Trusts
NCC|Software and Computer Services
NBPE|Collective Investments
N91|Financial Services
NAIT|Investment Trusts
NAS|Investment Trusts
OCI|Investment Trusts
OCDO|Food & drug retailers
OSB|Banks
OXB|Pharmaceuticals and Biotechnology
OXIG|Electronic & Electrical Equipment
ONT|Health Care
PHI|Investment Trusts
PAF|Precious Metals and Mining
PINT|Investment Trusts
PIN|Equity Investments
PAG|Financial Services
PEY|Investment Trusts
PPET|Investment Trusts
PNN|Gas, Water & Multiutilities
PSN|Household goods & home construction
PNL|Investment Trusts
PETS|Retailers
PTEC|Software & Computer Services
PLUS|Financial Services
PCGH|Investment Trusts
POLN|Financial Services
PPH|Travel and Leisure
PFD|Food Producers
PHP|Real Estate Investment Trusts
PRN|Food Producers
QQ|Aerospace & Defence
QLT|Financial Services
RNK|Travel and Leisure
RPI|Technology
RAT|Financial Services
RSW|Electronic & Electrical Equipment
RHIM|Support Services
RMV|Media
RCP|Investment Trusts
ROSE|Industrial Goods and Services
ROR|Industrial Engineering
RS1|Industrials
RTW|Investment Trusts
RICA|Investment Trusts
SAFE|Real Estate Investment Trusts
SAGA|Travel and Leisure
SVS|Real Estate Investment & Services
MNTN|Investment Trusts
ATR|Investment Trusts
SDP|Investment Trusts
SOI|Investment Trusts
SAIN|Investment Trusts
SEIT|Investment Trusts
SEQI|Investment Trusts
SSIT|Investment Trusts
SRP|Support Services
SHC|Real Estate Investment Trusts
SHAW|Banks
SRE|Real Estate
SCT|Software & Computer Services
SPI|Health Care Equipment & Services
SSPG|Travel and Leisure
SUPR|Real Estate Investment Trusts
SYNC|Pharmaceuticals & Biotechnology
THRL|Real Estate Investment Trusts
TATE|Food Producers
TW|Household goods & home construction
TBCG|Banks
TEP|Telecomms
TMPL|Collective Investments
TEM|Investment Trusts
TRIG|Investment Trusts
THG|Personal Products
TCAP|Financial Services
TRN|Industrial Transportation
TPK|Support Services
TRY|Investment Trusts
TRST|Technology
TFIF|Investment Trusts
UTG|Real estate investment trusts
UEM|Investment Trusts
VSVS|General Industrials
VCT|Chemicals
VEIL|Investment Trusts
VOF|Investment Trusts
VTY|Home Construction
VLX|Industrial Goods and Services
FAN|Construction & Materials
WOSG|Personal Goods
JDW|Travel & Leisure
SMWH|Retailers
WIX|Retailers
WIZZ|Travel & Leisure
WKP|Real Estate Investment Trusts
WWH|Investment Trusts
XPP|Industrial Goods and Services
XPS|Financial Services
YNGA|Travel and Leisure
ZIG|Support Services"""

def wiki_sector_to_bucket(raw) -> str:
    """Turns an FTSE/ICB sector name into one of the app's sector groups."""
    s = str(raw or "").strip().lower()
    if not s or s in ("nan", "none"):
        return "Unclassified"
    rules = [
        (("real estate", "reit"), "Real Estate"),
        (("investment trust", "collective invest", "equity invest", "hedge fund"), "Investment Trusts"),
        (("utilit", "electricity", "water", "multiutil"), "Telecom & Utilities"),
        (("telecom",), "Telecom & Utilities"),
        (("oil", "gas producer", "alternative energy", "energy", "coal"), "Energy"),
        (("pharma", "health", "biotech"), "Healthcare"),
        (("aerospace", "defence", "defense"), "Defense & Aero"),
        (("software", "technology", "electronic", "computer"), "Technology"),
        (("mining", "metal", "chemical", "container", "basic resource", "forest"), "Basic Materials"),
        (("bank", "financ", "insurance", "brokerage"), "Financials"),
        (("retail", "travel", "leisure", "food", "beverage", "tobacco", "personal", "household", "home construction",
          "homebuilding", "media", "consumer", "hospitality"), "Consumer"),
        (("industrial", "support service", "engineering", "construction", "transport", "general industrials"), "Industrials"),
    ]
    for keys, label in rules:
        if any(k in s for k in keys):
            return label
    return str(raw).strip().title()[:30]

def _parse_builtin(block: str) -> dict:
    out = {}
    for line in block.strip().splitlines():
        if "|" in line:
            t, sec = line.split("|", 1)
            out[f"{t.strip().upper()}.L"] = wiki_sector_to_bucket(sec)
    return out

BUILTIN_UNIVERSE = {**_parse_builtin(FTSE250_BUILTIN), **_parse_builtin(FTSE100_BUILTIN)}

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
    if s is None or str(s).strip().lower() in ("", "none", "nan", "nat"):
        return None
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

def gbp(x, d=2, sign=False) -> str:
    """£1,234.50 · -£12.00 · +£3.10 (sign=True). Never '£-12'."""
    try:
        v = float(x)
        if not math.isfinite(v):
            return "£–"
    except Exception:
        return "£–"
    txt = f"£{abs(v):,.{d}f}"
    if round(v, d) < 0:
        return "-" + txt
    return ("+" + txt) if (sign and round(v, d) > 0) else txt

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

def _norm_status(raw) -> str:
    """Any spelling of an open trade ('Active', '🟢 ACTIVE', blank) becomes exactly 'ACTIVE'."""
    if raw is None or str(raw).strip() == "" or str(raw).strip().lower() in ("nan", "none"):
        return "ACTIVE"
    return "ACTIVE" if classify_outcome(raw) == "ACTIVE" else str(raw).strip().upper()

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
    known = get_universe_sectors().get(t)
    if known and known != "Unclassified":
        return known
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
    """Sector if already known (static map, FTSE lists or cache); None otherwise. Never hits the network."""
    t = str(ticker).strip().upper()
    if t in SECTOR_MAP:
        return SECTOR_MAP[t]
    known = get_universe_sectors().get(t)
    if known and known != "Unclassified":
        return known
    hit = get_sector_cache().get(t)
    if hit and hit[0] != "Unclassified":
        return hit[0]
    return None

def get_trade_sector(ticker: str, f_json) -> str:
    t = str(ticker).strip().upper()
    if t in SECTOR_MAP:
        return SECTOR_MAP[t]
    f = _parse_features(f_json)
    if f.get("sector") and str(f["sector"]) != "Unclassified":
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

_RNS_BEAR_RE = [(re.compile(r"\b" + re.escape(p) + r"\b"), p, w) for p, w in RNS_BEARISH_LEXICON.items()]
_RNS_BULL_RE = [(re.compile(r"\b" + re.escape(p) + r"\b"), p, w) for p, w in RNS_BULLISH_LEXICON.items()]

def evaluate_rns_nlp_sentiment(ticker: str) -> dict:
    try:
        raw_rns = fetch_direct_rns_for_ticker(ticker) or {}
    except Exception:
        raw_rns = {}
    if not isinstance(raw_rns, dict):
        raw_rns = {}
    headline = str(raw_rns.get("headline", "")).lower()
    status = str(raw_rns.get("status", "📰 Flow Verified"))
    try:
        base_delta = float(raw_rns.get("delta", 0.0) or 0.0)
        base_delta = base_delta if math.isfinite(base_delta) else 0.0
    except Exception:
        base_delta = 0.0

    lex_delta = 0.0
    matched_tags = []
    for pat, phrase, weight in _RNS_BEAR_RE:                 # whole words only: "replacing" is not a "placing"
        if pat.search(headline):
            lex_delta += weight
            matched_tags.append(f"⚠️ {phrase.title()}")
    for pat, phrase, weight in _RNS_BULL_RE:
        if pat.search(headline):
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
# 0b. THE MARKET: WHICH SHARES (FTSE 350 + LIQUID AIM) AND WHICH DAYS (UK BANK HOLIDAYS, HALF DAYS)
# ==============================================================================
UNIVERSE_SAVE_PATH = os.path.join(ROOT_DIR, "uk_universe.json")
FTSE_LIST_PAGES = [("https://en.wikipedia.org/wiki/FTSE_100_Index", 80), ("https://en.wikipedia.org/wiki/FTSE_250_Index", 200)]
GOV_UK_HOLIDAYS_URL = "https://www.gov.uk/bank-holidays.json"
EXTRA_AIM_POOL = ["PTAL.L", "RKH.L", "ITM.L", "LTHM.L", "CMCL.L", "CAML.L", "SAV.L", "KP2.L", "BUR.L", "FEVR.L", "JET2.L", "NFG.L",
                  "DOTD.L", "BIG.L", "SEE.L", "AOM.L", "YOU.L", "CER.L", "RWS.L", "KGH.L", "CHRT.L", "CNC.L", "CRW.L", "BRCK.L",
                  "MIDW.L", "SRC.L", "JHD.L"]

@st.cache_resource
def get_market_box():
    return {"lock": threading.Lock()}

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

def html_table_rows(page: str, table_id: str = "constituents") -> list:
    """Plain-Python table reader (no lxml/bs4 needed): returns rows of cell texts."""
    m = re.search(r"""<table[^>]*id=["']""" + re.escape(table_id) + r"""["'][^>]*>(.*?)</table>""", page or "", re.S | re.I)
    if not m:
        return []
    rows = []
    for r in re.findall(r"<tr[^>]*>(.*?)</tr>", m.group(1), re.S | re.I):
        cells = re.findall(r"<t[hd][^>]*>(.*?)</t[hd]>", r, re.S | re.I)
        rows.append([re.sub(r"\[\d+\]", "", html.unescape(re.sub(r"<[^>]+>", "", c))).strip() for c in cells])
    return rows

def parse_constituents(page: str) -> dict:
    """{'TICKER.L': 'Sector group'} from a Wikipedia FTSE constituents table."""
    rows = html_table_rows(page)
    if not rows:
        return {}
    head = [h.lower() for h in rows[0]]
    t_i = next((i for i, h in enumerate(head) if "ticker" in h or "epic" in h), None)
    s_i = next((i for i, h in enumerate(head) if "sector" in h or "industry" in h or "icb" in h), None)
    if t_i is None:
        return {}
    out = {}
    for r in rows[1:]:
        if len(r) <= t_i:
            continue
        raw = re.sub(r"\s+", "", r[t_i]).upper().replace(".", "-")
        tkr = f"{raw}.L"
        if is_valid_lse_ticker(tkr):
            out[tkr] = wiki_sector_to_bucket(r[s_i]) if s_i is not None and len(r) > s_i else "Unclassified"
    return out

def _aim_pool() -> dict:
    pool = {}
    for t in sorted(set(EXTRA_AIM_POOL) | set(AIM_EXEMPT_TICKERS) | set(SECTOR_MAP)):
        if is_valid_lse_ticker(t):
            pool[t] = SECTOR_MAP.get(t, "Unclassified")
    return pool

def get_uk_universe() -> tuple:
    """(tickers, {ticker: sector}, where_from). Today's FTSE 100 + FTSE 250 lists when Wikipedia answers, otherwise the
    last saved copy, otherwise the built-in copy (Oct 2026). Liquid AIM shares are always added. Cached 12 h (2 h on failure)."""
    box = get_market_box()
    now = time.time()
    cached = box.get("universe")
    if cached and now - cached[0] < (43200 if cached[4] else 7200):
        return cached[1], cached[2], cached[3]
    live, fresh_pages = {}, 0
    for url, min_rows in FTSE_LIST_PAGES:
        try:
            resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}, timeout=6)
            if resp.status_code == 200:
                got = parse_constituents(resp.text)
                if len(got) >= min_rows:
                    live.update(got)
                    fresh_pages += 1
        except Exception:
            pass
    fresh = fresh_pages == len(FTSE_LIST_PAGES)
    if fresh:
        sectors = dict(live)
        source = "today's FTSE 100 + FTSE 250 lists"
        _save_json(UNIVERSE_SAVE_PATH, {"saved": london_now().strftime("%Y-%m-%d %H:%M"), "sectors": sectors})
    else:
        saved = _load_json(UNIVERSE_SAVE_PATH)
        if isinstance(saved, dict) and isinstance(saved.get("sectors"), dict) and len(saved["sectors"]) >= 250:
            sectors = {k: str(v) for k, v in saved["sectors"].items() if is_valid_lse_ticker(k)}
            source = f"FTSE 100 + FTSE 250 lists saved {saved.get('saved', 'earlier')} (list pages not reachable right now)"
        else:
            sectors = dict(BUILTIN_UNIVERSE)
            source = "built-in FTSE 100 + FTSE 250 lists (Oct 2026) — list pages not reachable right now"
        sectors.update({k: v for k, v in live.items()})
    for t, sec in _aim_pool().items():
        sectors.setdefault(t, sec)
    for t, sec in SECTOR_MAP.items():
        if t in sectors:
            sectors[t] = sec
    tickers = sorted(t for t in sectors if is_valid_lse_ticker(t))
    source = f"{source} + {sum(1 for t in tickers if is_aim_exempt(t))} AIM shares"
    box["universe"] = (now, tickers, sectors, source, fresh)
    return tickers, sectors, source

def get_universe_sectors() -> dict:
    box = get_market_box()
    cached = box.get("universe")
    if cached:
        return cached[2]
    return BUILTIN_UNIVERSE

# ---------- UK bank holidays (the LSE is shut) and half days (12:30 close) ----------
def _easter_sunday(y: int):
    a, b, c = y % 19, y // 100, y % 100
    d, e = b // 4, b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = c // 4, c % 4
    l_ = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l_) // 451
    month = (h + l_ - 7 * m + 114) // 31
    day = ((h + l_ - 7 * m + 114) % 31) + 1
    return datetime(y, month, day).date()

def uk_bank_holidays_calc(y: int) -> dict:
    """England & Wales bank holidays by rule (used when gov.uk can't be reached). One-off extra holidays are only
    known from gov.uk."""
    out = {}
    d = datetime(y, 1, 1).date()
    out[(d + timedelta(days={5: 2, 6: 1}.get(d.weekday(), 0))).isoformat()] = "New Year's Day"
    es = _easter_sunday(y)
    out[(es - timedelta(days=2)).isoformat()] = "Good Friday"
    out[(es + timedelta(days=1)).isoformat()] = "Easter Monday"
    may1 = datetime(y, 5, 1).date()
    out[(may1 + timedelta(days=(7 - may1.weekday()) % 7)).isoformat()] = "Early May bank holiday"
    may31 = datetime(y, 5, 31).date()
    out[(may31 - timedelta(days=may31.weekday())).isoformat()] = "Spring bank holiday"
    aug31 = datetime(y, 8, 31).date()
    out[(aug31 - timedelta(days=aug31.weekday())).isoformat()] = "Summer bank holiday"
    xmas, box_ = datetime(y, 12, 25).date(), datetime(y, 12, 26).date()
    if xmas.weekday() == 5:                                  # Sat: Mon 27 + Tue 28
        out[(xmas + timedelta(days=2)).isoformat()] = "Christmas Day (substitute)"
        out[(xmas + timedelta(days=3)).isoformat()] = "Boxing Day (substitute)"
    elif xmas.weekday() == 6:                                # Sun: Mon 26 Boxing + Tue 27 substitute
        out[box_.isoformat()] = "Boxing Day"
        out[(xmas + timedelta(days=2)).isoformat()] = "Christmas Day (substitute)"
    else:
        out[xmas.isoformat()] = "Christmas Day"
        if box_.weekday() == 5:
            out[(box_ + timedelta(days=2)).isoformat()] = "Boxing Day (substitute)"
        else:
            out[box_.isoformat()] = "Boxing Day"
    return out

def get_uk_holidays() -> dict:
    """{'YYYY-MM-DD': name}. gov.uk's official list when reachable (includes one-off holidays), plus the rules above for
    any year gov.uk doesn't cover. Cached 12 h (2 h if gov.uk didn't answer)."""
    box = get_market_box()
    now = time.time()
    cached = box.get("holidays")
    if cached and now - cached[0] < (43200 if cached[2] else 7200):
        return cached[1]
    official = {}
    try:
        resp = requests.get(GOV_UK_HOLIDAYS_URL, headers={"User-Agent": "Mozilla/5.0"}, timeout=5)
        if resp.status_code == 200:
            for ev in (resp.json().get("england-and-wales", {}) or {}).get("events", []) or []:
                d = _to_date(ev.get("date"))
                if d:
                    official[d.isoformat()] = str(ev.get("title", "Bank holiday"))[:60]
    except Exception:
        official = {}
    per_year = {}
    for k in official:
        per_year[k[:4]] = per_year.get(k[:4], 0) + 1
    years_official = {y for y, n in per_year.items() if n >= 6}         # only trust a year gov.uk lists in full
    y0 = london_now().year
    out = dict(official)
    for y in (y0 - 1, y0, y0 + 1):
        if str(y) not in years_official:
            out.update(uk_bank_holidays_calc(y))
    box["holidays"] = (now, out, bool(official))
    return out

def get_market_state() -> dict:
    """Is the LSE trading today, and when does it close? Weekends, bank holidays and the 12:30 half days on
    Christmas Eve and New Year's Eve are all handled."""
    now = london_now()
    today = now.date()
    if now.weekday() > 4:
        return {"trading_day": False, "reason": "Weekend", "close": dtime(16, 30), "half_day": False}
    hol = get_uk_holidays()
    if today.isoformat() in hol:
        return {"trading_day": False, "reason": f"Bank holiday — {hol[today.isoformat()]}", "close": dtime(16, 30), "half_day": False}
    if (today.month, today.day) in ((12, 24), (12, 31)):
        return {"trading_day": True, "reason": "Half day — the market closes at 12:30", "close": dtime(12, 30), "half_day": True}
    return {"trading_day": True, "reason": "", "close": dtime(16, 30), "half_day": False}

def next_open_text() -> str:
    now = london_now()
    hol = get_uk_holidays()
    for add in range(0, 12):
        d = now.date() + timedelta(days=add)
        if d.weekday() > 4 or d.isoformat() in hol:
            continue
        close_t = dtime(12, 30) if (d.month, d.day) in ((12, 24), (12, 31)) else dtime(16, 30)
        if add == 0 and now.time() >= close_t:
            continue
        if add == 0 and now.time() >= dtime(8, 0):
            return "now"
        when = "today" if add == 0 else ("tomorrow" if add == 1 else d.strftime("%a %d %b"))
        return f"{when} 08:00"
    return "soon"

def upcoming_holidays(n: int = 5) -> list:
    today = london_now().date().isoformat()
    return [(d, name) for d, name in sorted(get_uk_holidays().items()) if d >= today][:n]

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
            for tbl in ("trade_journal", "daily_options_journal"):
                try:
                    con.execute(f"UPDATE {tbl} SET status = 'ACTIVE' WHERE status IS NULL OR TRIM(status) = '' OR (UPPER(status) LIKE '%ACTIVE%' AND status != 'ACTIVE')")
                except Exception:
                    pass
        finally:
            con.close()

init_duckdb_storage()

def journal_counts() -> dict:
    out = {"total": 0, "open": 0, "closed": 0, "opt_total": 0, "opt_open": 0}
    with DB_LOCK:
        con = duckdb.connect(DB_PATH, read_only=True)
        try:
            for r in con.execute("SELECT status FROM trade_journal").fetchall():
                out["total"] += 1
                out["open" if classify_outcome(r[0]) == "ACTIVE" else "closed"] += 1
            for r in con.execute("SELECT status FROM daily_options_journal").fetchall():
                out["opt_total"] += 1
                out["opt_open"] += int(classify_outcome(r[0]) == "ACTIVE")
        except Exception:
            pass
        finally:
            con.close()
    return out

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
    """Restore trades from the online backup. Each row is handled on its own, so ONE unreadable old row can never stop
    the newer rows (usually your open trades) from loading. Newer local data is never overwritten, and a trade that has
    closed here never re-opens."""
    info = {"remote": 0, "remote_active": 0, "restored": 0, "kept_local": 0, "bad_rows": 0, "first_error": ""}
    st.session_state["hydrate_info"] = info
    if not supabase:
        return
    eq_data = fetch_all_supabase_rows("predictions")      # network I/O outside the DB lock
    opt_data = fetch_all_supabase_rows("options_journal")
    info["remote"] = len(eq_data or [])
    info["remote_active"] = sum(1 for r in (eq_data or []) if _norm_status(r.get("status")) == "ACTIVE" and is_valid_lse_ticker(str(r.get("ticker", "")).strip().upper()))

    def bad(e):
        info["bad_rows"] += 1
        if not info["first_error"]:
            info["first_error"] = f"{type(e).__name__}: {str(e)[:90]}"

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
                    tkr = str(r.get('ticker', '')).strip().upper()
                    if not is_valid_lse_ticker(tkr):
                        continue
                    pred_date = str(r.get('predicted_date') or london_now().strftime('%Y-%m-%d'))[:10]
                    trade_id = f"{tkr}_{pred_date}"
                    remote_status = _norm_status(r.get('status'))
                    remote_ts = str(r.get('last_checked') or "")
                    loc = local_eq.get(trade_id)
                    if loc:
                        l_status, l_ts = str(loc[1]), str(loc[2] or "")
                        closed_vs_active = (classify_outcome(l_status) != "ACTIVE" and remote_status == "ACTIVE")
                        local_newer = bool(l_ts) and (not remote_ts or l_ts >= remote_ts)
                        if closed_vs_active or local_newer:
                            info["kept_local"] += 1
                            if closed_vs_active:
                                cloud_update("predictions", {"ticker": loc[3], "predicted_date": loc[4]},
                                             {"status": l_status, "stop_loss": _num(loc[5]), "latest_price": _num(loc[6]),
                                              "pnl_pct": _num(loc[7]), "last_checked": l_ts}, "Re-push Closed Trade")
                            continue
                    entry, tgt, stp = _num(r.get('entry_price')), _num(r.get('target_price')), _num(r.get('stop_loss'))
                    if remote_status == "ACTIVE" and (entry <= 0 or tgt <= 0 or stp <= 0 or tgt <= stp):
                        raise ValueError(f"{tkr}: missing or impossible prices (bought {r.get('entry_price')}, goal {r.get('target_price')}, exit {r.get('stop_loss')})")
                    last_check = remote_ts or london_now().strftime('%Y-%m-%d %H:%M:%S')
                    f_obj = r.get('features_json')
                    f_json = json.dumps(_parse_features(f_obj) if not isinstance(f_obj, dict) else f_obj)
                    shares_n = max(1, _int(r.get('shares_qty'), 1))
                    cap = _num(r.get('position_gbp'))
                    if cap <= 0:
                        cap = round(shares_n * entry / 100.0, 2) if entry > 0 else POSITION_SIZE_GBP
                    latest = _num(r.get('latest_price'), entry)
                    con.execute("""
                        INSERT OR REPLACE INTO trade_journal 
                        (trade_id, timestamp, date_str, ticker, asset_type, entry_price, target_price,
                         stop_loss, shares, capital_allocated, status, latest_price, pnl_pct, exit_price,
                         exit_timestamp, last_audited, features_json)
                        VALUES (?, ?, ?, ?, 'EQUITY', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, [trade_id, last_check, pred_date, tkr, entry, tgt, stp, shares_n, cap, remote_status,
                          latest, _num(r.get('pnl_pct')), latest if remote_status != "ACTIVE" else 0.0,
                          last_check if remote_status != "ACTIVE" else None, last_check, f_json])
                    info["restored"] += 1
                except Exception as e:
                    bad(e)

            for o in (opt_data or []):
                try:
                    date_k = str(o.get('date_key') or london_now().strftime('%Y-%m-%d'))[:10]
                    remote_status = _norm_status(o.get('status'))
                    remote_ts = str(o.get('last_audited') or "")
                    loc = local_opt.get(date_k)
                    if loc:
                        l_status, l_ts = str(loc[1]), str(loc[2] or "")
                        if (classify_outcome(l_status) != "ACTIVE" and remote_status == "ACTIVE") or (l_ts and (not remote_ts or l_ts >= remote_ts)):
                            if classify_outcome(l_status) != "ACTIVE" and remote_status == "ACTIVE":
                                cloud_update("options_journal", {"date_key": date_k},
                                             {"status": l_status, "current_option_price": _num(loc[3]),
                                              "pnl_pct": _num(loc[4]), "last_audited": l_ts}, "Re-push Closed Option")
                            continue
                    cur_p = _num(o.get('current_option_price'))
                    if _num(o.get('entry_premium'), cur_p) <= 0 or not str(o.get('option_contract') or '').strip():
                        raise ValueError(f"option {date_k}: missing contract or price")
                    con.execute("""
                        INSERT OR REPLACE INTO daily_options_journal 
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, [date_k, str(o.get('timestamp') or ""), str(o.get('share_name') or ""), str(o.get('option_contract')),
                          _num(o.get('strike_price')), _int(o.get('expiry_days'), 21), _num(o.get('underlying_spot')),
                          _int(o.get('lot_size'), 100), _num(o.get('entry_premium'), cur_p), cur_p, _num(o.get('target_premium')),
                          _num(o.get('stop_loss_premium')), _num(o.get('total_capital')), _num(o.get('ai_confidence'), 80.0),
                          _num(o.get('implied_vol'), 20.0), remote_status, _num(o.get('pnl_pct')), remote_ts])
                except Exception as e:
                    bad(e)
        finally:
            con.close()
    if info["bad_rows"]:
        record_db_error("Restore from cloud", Exception(f"{info['bad_rows']} backup row(s) were unreadable and skipped ({info['first_error']})"))
    elif "db_error" in st.session_state and "Restore" in str(st.session_state.get("db_error")):
        del st.session_state["db_error"]

# ==============================================================================
# 2. TIMEZONE, MARKET CLOCK, MACRO FEED & PARALLEL DATA CACHING
# ==============================================================================
LOT_SIZES = {"SHEL": 1000, "AZN": 500, "HSBA": 2000, "ULVR": 500, "BP": 3000, "BARC": 5000, "RIO": 250, "GLEN": 4000}

def is_lse_market_open() -> bool:
    """Open 08:00–16:30 London on trading days (12:30 close on half days); shut at weekends and on bank holidays."""
    state = get_market_state()
    return state["trading_day"] and dtime(8, 0) <= london_now().time() <= state["close"]

def entry_window_end() -> dtime:
    close = get_market_state()["close"]
    half_hour_before = (datetime.combine(datetime(2000, 1, 1), close) - timedelta(minutes=30)).time()
    return min(ENTRY_WINDOW_END, half_hour_before)

def is_entry_window_open() -> bool:
    state = get_market_state()
    return state["trading_day"] and ENTRY_WINDOW_START <= london_now().time() <= entry_window_end()

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

def prefetch_histories(tickers: list, period: str = "120d", ttl=300, workers: int = 8, progress=None, time_budget: float = None) -> bool:
    """Parallel download into a shared in-memory store (no Streamlit calls inside worker threads).
    ttl can be a number or a function(ticker)->seconds. Returns False if the time budget ran out first."""
    store = get_hist_store()
    now = time.time()
    ttl_of = ttl if callable(ttl) else (lambda _t: ttl)
    with store["lock"]:
        for k in [k for k, v in store["data"].items() if now - v[0] > QUIET_TTL_SEC + 600]:
            del store["data"][k]
        todo = [t for t in dict.fromkeys(tickers) if (is_valid_lse_ticker(t) or str(t).startswith("^")) and
                ((t, period) not in store["data"] or now - store["data"][(t, period)][0] > ttl_of(t))]
    if not todo:
        if progress:
            progress(1, 1)
        return True
    started, done, complete = time.time(), 0, True
    batch_n = max(workers * 5, 20)
    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        for i in range(0, len(todo), batch_n):
            if time_budget is not None and time.time() - started > time_budget:
                complete = False
                break
            part = todo[i:i + batch_n]
            res = list(ex.map(lambda t: (t, _download_history_raw(t, period)), part))
            stamp = time.time()
            with store["lock"]:
                for t, df in res:
                    old = store["data"].get((t, period))
                    if (df is None or df.empty) and old is not None and old[1] is not None and not old[1].empty:
                        continue                                    # never replace good prices with a failed download
                    store["data"][(t, period)] = (stamp, df if df is not None else pd.DataFrame())
            done += len(part)
            if progress:
                progress(done, len(todo))
            if len(part) >= 10 and sum(1 for _, d in res if d is None or d.empty) > len(part) * 0.6:
                time.sleep(1.5)                                     # the price service looks busy: back off politely
    return complete

def peek_history(ticker: str, period: str = "120d", max_age: float = None) -> pd.DataFrame:
    """Prices already downloaded by prefetch_histories (no network). With max_age, older copies count as missing."""
    store = get_hist_store()
    with store["lock"]:
        item = store["data"].get((ticker, period))
    if item is None or item[1] is None or item[1].empty:
        return pd.DataFrame()
    if max_age is not None and time.time() - item[0] > max_age:
        return pd.DataFrame()
    return item[1].copy()

def get_history_fast(ticker: str, period: str = "120d") -> pd.DataFrame:
    df = peek_history(ticker, period, max_age=SWEEP_TTL_SEC)
    if not df.empty:
        return df
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

@st.cache_data(ttl=1800, show_spinner=False)
def get_ftse_close_series_long() -> pd.Series:
    """About a year of FTSE 100 closes — used to rebuild the market context of older trades."""
    try:
        h = yf.Ticker("^FTSE").history(period="1y", timeout=6)
        if h is None or h.empty:
            return pd.Series(dtype=float)
        s_ = h["Close"].copy()
        s_.index = pd.to_datetime([pd.Timestamp(x).date() for x in s_.index])
        return s_.sort_index()
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
    st.session_state["price_feed_problem"] = False

    with DB_LOCK:
        con = duckdb.connect(DB_PATH, read_only=False)
        try:
            active_trades = con.execute("SELECT * FROM trade_journal WHERE status = 'ACTIVE'").df()
            active_opts = con.execute("SELECT * FROM daily_options_journal WHERE status = 'ACTIVE'").df()
        except Exception:
            active_trades, active_opts = pd.DataFrame(), pd.DataFrame()
        finally:
            con.close()

    # ---------------- Equities ----------------
    if not active_trades.empty:
        for _, tr in active_trades.iterrows():
            tkr = str(tr["ticker"]).strip()
            if not is_valid_lse_ticker(tkr):
                continue
            try:
                h = fetch_cached_history(tkr, period="1mo")
                if h.empty:
                    st.session_state["price_feed_problem"] = True
                    continue                                       # do NOT pretend we checked
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

    # ---------------- Options (real quotes for new ones; old practice rows keep their formula) ----------------
    if not active_opts.empty:
        for _, opt in active_opts.iterrows():
            try:
                if is_real_option(opt["option_contract"]):
                    audit_real_option(opt, now_str, today_date)
                    continue
                share = str(opt['share_name'])
                h = fetch_cached_history(f"{share}.L", period="1mo")
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


# ---------- market context of a trade (same maths at buying time and when rebuilding old trades) ----------
def compute_context(df_model: pd.DataFrame, ftse_series: pd.Series = None, breadth=None) -> dict:
    """Plain facts about a share on its last COMPLETED bar: strength vs the FTSE 100, momentum, distance above its
    20-day average, volume vs normal, daily jumpiness, and how much of the market was rising."""
    c = pd.to_numeric(df_model["Close"], errors="coerce").astype(float)
    v = pd.to_numeric(df_model["Volume"], errors="coerce").fillna(0.0).astype(float)
    hi = pd.to_numeric(df_model["High"], errors="coerce").astype(float)
    lo = pd.to_numeric(df_model["Low"], errors="coerce").astype(float)
    last = float(c.iloc[-1])
    ema20 = c.ewm(span=20, adjust=False).mean()
    delta = c.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    g_, l_ = float(gain.iloc[-1]), float(loss.iloc[-1])
    rsi = 100.0 if l_ == 0 and g_ > 0 else (50.0 if l_ == 0 else 100.0 - 100.0 / (1.0 + g_ / l_))
    prev_c = c.shift(1)
    tr = pd.concat([hi - lo, (hi - prev_c).abs(), (lo - prev_c).abs()], axis=1).max(axis=1)
    atr = float(tr.rolling(14).mean().iloc[-1])
    vol_norm = float(v.iloc[-21:-1].mean()) if len(v) > 21 else float(v.mean())
    ret20 = (last / float(c.iloc[-21]) - 1.0) * 100.0 if len(c) > 21 and float(c.iloc[-21]) > 0 else 0.0
    ctx = {
        "rsi14": round(rsi, 1),
        "dist20": round((last / float(ema20.iloc[-1]) - 1.0) * 100.0, 2) if float(ema20.iloc[-1]) > 0 else 0.0,
        "rvol": round(float(v.iloc[-1]) / vol_norm, 2) if vol_norm > 0 else 1.0,
        "atr_pct": round(atr / last * 100.0, 2) if last > 0 and math.isfinite(atr) else 0.0,
        "mom5": round((last / float(c.iloc[-6]) - 1.0) * 100.0, 2) if len(c) > 6 and float(c.iloc[-6]) > 0 else 0.0,
        "ret20": round(ret20, 2),
        "uptrend": bool(last >= float(ema20.iloc[-1]) and ret20 > 0),
    }
    if ftse_series is not None and not ftse_series.empty:
        try:
            end = pd.Timestamp(pd.Timestamp(df_model.index[-1]).date())
            fs = ftse_series[ftse_series.index <= end]
            if len(fs) > 50:
                f20 = (float(fs.iloc[-1]) / float(fs.iloc[-21]) - 1.0) * 100.0
                ctx["rs20"] = round(ret20 - f20, 2)
                ctx["ftse_up"] = int(float(fs.iloc[-1]) > float(fs.iloc[-50:].mean()))
            elif len(fs) > 21:
                ctx["rs20"] = round(ret20 - (float(fs.iloc[-1]) / float(fs.iloc[-21]) - 1.0) * 100.0, 2)
        except Exception:
            pass
    if breadth is not None:
        ctx["breadth"] = round(float(breadth), 1)
    for k, val in list(ctx.items()):
        if isinstance(val, float) and not math.isfinite(val):
            ctx.pop(k)
    return ctx

EVIDENCE_FACTORS = [
    ("rs20", "Strength vs the FTSE 100 (last 20 days)", [(-1e9, -3, "Lagging the FTSE by 3%+"), (-3, 0, "Lagging it a little"), (0, 3, "Beating it a little"), (3, 1e9, "Beating it by 3%+")]),
    ("rsi14", "Momentum (RSI)", [(-1e9, 45, "Weak (under 45)"), (45, 55, "Neutral (45–55)"), (55, 65, "Firm (55–65)"), (65, 75, "Strong (65–75)"), (75, 1e9, "Very strong (75+)")]),
    ("dist20", "Distance above its 20-day average", [(-1e9, 0, "Below its average"), (0, 2, "Just above (0–2%)"), (2, 5, "Well above (2–5%)"), (5, 1e9, "Stretched (5%+)")]),
    ("rvol", "Trading volume vs normal", [(-1e9, 0.8, "Quieter than usual"), (0.8, 1.3, "Normal"), (1.3, 2.0, "Busy (1.3–2×)"), (2.0, 1e9, "Very busy (2×+)")]),
    ("atr_pct", "How jumpy the share is", [(-1e9, 1.5, "Calm (under 1.5% a day)"), (1.5, 2.5, "Normal (1.5–2.5%)"), (2.5, 4.0, "Lively (2.5–4%)"), (4.0, 1e9, "Very jumpy (4%+)")]),
    ("breadth", "How much of the market was rising", [(-1e9, 35, "Most shares falling"), (35, 55, "Mixed market"), (55, 1e9, "Most shares rising")]),
    ("ftse_up", "FTSE 100 above its 50-day average", [(-0.5, 0.5, "No — market in a dip"), (0.5, 1.5, "Yes — market trending up")]),
]

def ctx_bucket(bins: list, value):
    if value is None:
        return None
    try:
        v = float(value)
    except Exception:
        return None
    if not math.isfinite(v):
        return None
    for lo_, hi_, label in bins:
        if lo_ <= v < hi_:
            return label
    return None

def build_evidence_model(closed_df: pd.DataFrame) -> dict:
    """Which conditions have really won for YOU. Win rate per group, shrunk toward your overall win rate so small groups
    can't fool it, with older trades counting less. Only used once there are enough clear wins/losses."""
    model = {"ready": False, "n": 0, "with_ctx": 0, "p0": None, "factors": {}}
    if closed_df is None or closed_df.empty:
        return model
    today = london_now().date()
    rows = []
    for _, r in closed_df.iterrows():
        o = classify_outcome(r["status"])
        if o not in ("WIN", "LOSS"):
            continue
        ctx = _parse_features(r["features_json"]).get("ctx")
        if not isinstance(ctx, dict) or ctx.get("unavailable"):
            continue
        d = _to_date(_first_valid(r.get("exit_timestamp"), r.get("last_audited"), r.get("date_str")))
        w = math.pow(0.5, max(0, (today - d).days) / EVIDENCE_HALF_LIFE_DAYS) if d else 0.5
        rows.append((ctx, 1 if o == "WIN" else 0, w))
    model["with_ctx"] = len(rows)
    model["n"] = len(rows)
    if not rows:
        return model
    w_tot = sum(w for _, _, w in rows)
    p0 = sum(w * y for _, y, w in rows) / w_tot if w_tot > 0 else 0.5
    model["p0"] = p0
    model["ready"] = len(rows) >= EVIDENCE_MIN_DECIDED
    for key, title, bins in EVIDENCE_FACTORS:
        groups = {}
        for ctx, y, w in rows:
            b = ctx_bucket(bins, ctx.get(key))
            if b is None:
                continue
            g = groups.setdefault(b, {"n": 0, "wins": 0, "w": 0.0, "wy": 0.0})
            g["n"] += 1
            g["wins"] += y
            g["w"] += w
            g["wy"] += w * y
        out = {}
        for _, _, label in bins:
            g = groups.get(label)
            if not g:
                continue
            shrunk = (g["wy"] + EVIDENCE_PRIOR_TRADES * p0) / (g["w"] + EVIDENCE_PRIOR_TRADES)
            diff = (shrunk - p0) * 100.0
            active = g["n"] >= EVIDENCE_MIN_BUCKET
            pts = max(-EVIDENCE_FACTOR_CAP, min(EVIDENCE_FACTOR_CAP, diff * EVIDENCE_POINTS_PER_PCT)) if active else 0.0
            out[label] = {"n": g["n"], "raw_rate": g["wins"] / g["n"] * 100.0, "shrunk": shrunk * 100.0, "diff": diff, "points": round(pts, 1), "active": active}
        model["factors"][key] = {"title": title, "groups": out}
    return model

def evidence_adjustment(model: dict, ctx: dict) -> tuple:
    """(points, [plain-English reasons]) — 0 until the learner has enough evidence."""
    if not model or not model.get("ready") or not isinstance(ctx, dict):
        return 0, []
    total, notes = 0.0, []
    for key, title, bins in EVIDENCE_FACTORS:
        b = ctx_bucket(bins, ctx.get(key))
        g = (model["factors"].get(key, {}).get("groups", {}) or {}).get(b) if b else None
        if not g or not g["active"] or abs(g["points"]) < 0.5:
            continue
        total += g["points"]
        notes.append(f"{g['points']:+.0f} ({b}: {g['raw_rate']:.0f}% won of {g['n']})")
    total = max(-EVIDENCE_TOTAL_CAP, min(EVIDENCE_TOTAL_CAP, total))
    return int(round(total)), notes

def backfill_trade_context(limit: int = EVIDENCE_BACKFILL_PER_RUN) -> int:
    """Rebuilds the market context of finished trades bought before context was recorded, from their price history,
    so the evidence learner can learn from your past trades straight away. A few per check; saved once done."""
    closed = load_closed_trades_df()
    if closed is None or closed.empty:
        return 0
    today_s = london_now().strftime("%Y-%m-%d")
    todo = []
    for _, r in closed.iterrows():
        if classify_outcome(r["status"]) not in ("WIN", "LOSS"):
            continue
        f = _parse_features(r["features_json"])
        c = f.get("ctx")
        if isinstance(c, dict) and (not c.get("unavailable") or c.get("tried") == today_s):
            continue
        if not is_valid_lse_ticker(str(r["ticker"])):
            continue
        todo.append((r["trade_id"], str(r["ticker"]), str(r["date_str"])[:10], f))
        if len(todo) >= limit:
            break
    if not todo:
        return 0
    prefetch_histories([t for _, t, _, _ in todo], "1y", ttl=QUIET_TTL_SEC, workers=SCAN_WORKERS)
    ftse_long = get_ftse_close_series_long()
    done = 0
    for trade_id, tkr, d_str, f in todo:
        try:
            h = peek_history(tkr, "1y")
            entry_bar = _to_date(f.get("entry_bar"))
            buy_day = _to_date(d_str)
            if h.empty or buy_day is None:
                ctx = {"unavailable": True, "tried": today_s}
            else:
                dates = _bar_dates(h)
                cutoff = entry_bar or (buy_day - timedelta(days=1))
                keep = [i for i, d in enumerate(dates) if d <= cutoff]
                if len(keep) < 30:
                    ctx = {"unavailable": True, "tried": today_s}
                else:
                    ctx = compute_context(h.iloc[: keep[-1] + 1], ftse_long)
                    ctx["source"] = "rebuilt from price history"
            f["ctx"] = ctx
            with DB_LOCK:
                con = duckdb.connect(DB_PATH, read_only=False)
                try:
                    con.execute("UPDATE trade_journal SET features_json = ? WHERE trade_id = ?", [json.dumps(f), trade_id])
                finally:
                    con.close()
            done += int(not ctx.get("unavailable"))
        except Exception:
            continue
    return done

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
# 6. OPTIONS ENGINE (REAL US-LISTED OPTION QUOTES ON UK COMPANIES — PAPER TRADED)
# ==============================================================================
# Several UK companies also trade in New York (as ADRs) and those have real, liquid options. Quotes come from Yahoo
# (~15 min delayed, US dollars, 100 shares per contract). Nothing is bought for real — trades are tracked with pretend money.
US_OPTION_UNDERLYINGS = {"SHEL": "Shell", "BP": "BP", "HSBC": "HSBC", "AZN": "AstraZeneca", "GSK": "GSK",
                         "RIO": "Rio Tinto", "BCS": "Barclays", "BTI": "British American Tobacco", "UL": "Unilever", "VOD": "Vodafone"}
ADR_TO_LSE = {"SHEL": "SHEL.L", "BP": "BP.L", "HSBC": "HSBA.L", "AZN": "AZN.L", "GSK": "GSK.L",
              "RIO": "RIO.L", "BCS": "BARC.L", "BTI": "BATS.L", "UL": "ULVR.L", "VOD": "VOD.L"}
OPTION_MULTIPLIER = 100             # one US option contract = 100 shares
OPTION_BUDGET_GBP = 500.0
OPTION_MIN_DAYS = 14
OPTION_MAX_DAYS = 45
OPTION_MIN_OPEN_INTEREST = 50       # must be genuinely traded
OPTION_MAX_SPREAD_PCT = 15.0        # buy/sell gap must be tight

def is_real_option(contract) -> bool:
    return "$" in str(contract)

def us_options_open() -> bool:
    ny = london_now().astimezone(pytz.timezone('America/New_York'))
    return ny.weekday() <= 4 and dtime(9, 35) <= ny.time() <= dtime(15, 55)

@st.cache_data(ttl=600, show_spinner=False)
def get_gbpusd() -> float:
    try:
        h = yf.Ticker("GBPUSD=X").history(period="5d", timeout=5)
        v = float(h["Close"].iloc[-1])
        if 0.8 < v < 2.0:
            return v
    except Exception:
        pass
    return 1.30

def _quote_from_row(r):
    """Valid real quote (bid & ask both live) or None."""
    try:
        bid = float(r.get("bid", 0) or 0)
        ask = float(r.get("ask", 0) or 0)
    except Exception:
        return None
    if not (math.isfinite(bid) and math.isfinite(ask) and bid > 0 and ask > 0 and ask >= bid):
        return None
    try:
        oi = float(r.get("openInterest", 0) or 0)
        oi = oi if math.isfinite(oi) else 0.0
    except Exception:
        oi = 0.0
    try:
        iv = float(r.get("impliedVolatility", 0) or 0)
        iv = iv if math.isfinite(iv) else 0.0
    except Exception:
        iv = 0.0
    return {"bid": bid, "ask": ask, "oi": oi, "iv": iv * 100.0, "spread_pct": (ask - bid) / ((ask + bid) / 2.0) * 100.0}

def get_option_quote(sym: str, expiry_str: str, strike: float):
    try:
        ch = yf.Ticker(sym).option_chain(expiry_str).calls
        row = ch[(ch["strike"] - float(strike)).abs() < 1e-6]
        if row.empty:
            return None
        return _quote_from_row(row.iloc[0])
    except Exception:
        return None

def find_real_option_idea(sym: str, gbpusd: float):
    """One candidate per underlying: ~2% out-of-the-money call, ~4 weeks out, only if the REAL quote is good enough."""
    try:
        h = _download_history_raw(sym, "30d")
        if h.empty or len(h) < 15:
            return None
        spot = float(h["Close"].iloc[-1])
        ret_5d = spot / float(h["Close"].iloc[-5]) - 1.0
        tk = yf.Ticker(sym)
        today = london_now().date()
        best_exp, best_gap = None, 999
        for e in list(tk.options or []):
            d = _to_date(e)
            if d is None:
                continue
            days = (d - today).days
            if OPTION_MIN_DAYS <= days <= OPTION_MAX_DAYS and abs(days - 28) < best_gap:
                best_exp, best_gap = e, abs(days - 28)
        if best_exp is None:
            return None
        calls = tk.option_chain(best_exp).calls
        calls = calls[calls["strike"] >= spot * 1.02].sort_values("strike")
        if calls.empty:
            return None
        r = calls.iloc[0]
        q = _quote_from_row(r)
        if q is None or q["oi"] < OPTION_MIN_OPEN_INTEREST or q["spread_pct"] > OPTION_MAX_SPREAD_PCT:
            return None
        cost_gbp = q["ask"] * OPTION_MULTIPLIER / gbpusd
        if cost_gbp > OPTION_BUDGET_GBP:
            return None
        return {"sym": sym, "spot": spot, "ret_5d": ret_5d, "strike": float(r["strike"]), "expiry": str(best_exp),
                "bid": q["bid"], "ask": q["ask"], "iv": q["iv"], "oi": q["oi"], "cost_gbp": round(cost_gbp, 2)}
    except Exception:
        return None

@st.cache_data(ttl=600, show_spinner=False)
def scan_real_option_ideas(gbpusd_key: float, day_key: str, skip: tuple) -> list:
    syms = [s for s in US_OPTION_UNDERLYINGS if s not in skip]
    if not syms:
        return []
    with ThreadPoolExecutor(max_workers=6) as ex:
        res = list(ex.map(lambda s: find_real_option_idea(s, gbpusd_key), syms))
    return [r for r in res if r]


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
       1) a share whose option lost recently takes a break from options (fades by half every 30 days),
       2) after enough results, if recent ideas lose on average, it only takes shares with a stronger 5-day climb.
       Rule 2 looks at the last 60 days only, so if it ever stops all ideas it lifts by itself and gets re-tested."""
    out = {"n": 0, "recent_n": 0, "ready": False, "bar": OPT_BASE_TREND_BAR, "rest": {}, "wavg": None, "real": None, "old": None, "by_share": {}}
    if df_all is None:
        df_all = load_options_journal_df()
    d = prep_perf_frame(df_all, "total_capital", is_option=True)
    if d is None or d.empty:
        return out
    today = london_now().date()
    d = d.copy()
    d["age"] = d["exit_date"].apply(lambda x: max(0, (today - x).days) if x else 0)
    d["w"] = 0.5 ** (d["age"] / HALF_LIFE_DAYS)
    d["real_src"] = d["option_contract"].apply(is_real_option)
    out["n"] = int(len(d))
    out["real"], out["old"] = _opt_group(d[d["real_src"]]), _opt_group(d[~d["real_src"]])
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
            out["bar"] = round(min(OPT_MAX_TREND_BAR, OPT_BASE_TREND_BAR + abs(wavg) * 0.5), 1)
    return out

def trend_bar_to_rise(bar: float) -> float:
    """Options trend strength = 78 + 5-day rise(%) × 1.5  ->  the 5-day rise (%) needed for a given bar."""
    return (float(bar) - 78.0) / 1.5

def generate_daily_options_alpha() -> dict:
    now_lon = london_now()
    today_str = now_lon.strftime('%Y-%m-%d')
    now_str = now_lon.strftime('%Y-%m-%d %H:%M:%S')
    st.session_state["opt_note"] = ""

    with DB_LOCK:
        con = duckdb.connect(DB_PATH, read_only=True)
        try:
            df = con.execute("SELECT * FROM daily_options_journal WHERE date_key = ?", [today_str]).df()
            if not df.empty:
                return df.iloc[0].to_dict()
            active_underlyings = set(con.execute(
                "SELECT DISTINCT share_name FROM daily_options_journal WHERE status = 'ACTIVE' AND option_contract LIKE '%$%'").df()["share_name"].tolist())
        except Exception:
            active_underlyings = set()
        finally:
            con.close()

    if not us_options_open():
        st.session_state["opt_note"] = "The US options market is closed right now. Real quotes are available roughly 14:30–21:00 UK time, Monday to Friday — the idea of the day appears then."
        return {}

    learn = options_learning()
    resting = set(learn["rest"])
    ideas = scan_real_option_ideas(round(get_gbpusd(), 3), today_str, tuple(sorted(active_underlyings | resting)))
    if not ideas:
        why = f" The options learner is also resting {', '.join(sorted(resting))} after recent losses." if resting else ""
        st.session_state["opt_note"] = ("No option passed the quality checks right now (a live quote, enough real trading, a tight buy/sell gap and a cost under £"
                                        f"{OPTION_BUDGET_GBP:.0f}). That's normal — it will try again when the page refreshes." + why)
        return {}
    def _trend_score(i):
        return round(min(92.0, max(68.0, 78.0 + i["ret_5d"] * 150.0)), 1)          # trend-strength score, NOT a probability
    strong = [i for i in ideas if _trend_score(i) >= learn["bar"]]
    if not strong:
        st.session_state["opt_note"] = (f"No option idea today — the options learner is being picky: recent ideas lost money on average, so it now needs "
                                        f"trend strength {learn['bar']:.0f}+ (a 5-day rise of about {max(0.0, trend_bar_to_rise(learn['bar'])):.1f}%+), and none of today's "
                                        f"{len(ideas)} candidate(s) had that. Doing nothing is a valid choice.")
        return {}

    best = max(strong, key=lambda i: i["ret_5d"])
    exp_d = _to_date(best["expiry"])
    days_to_expiry = max(1, (exp_d - now_lon.date()).days)
    ask, bid = float(best["ask"]), float(best["bid"])
    score = _trend_score(best)
    sig = {
        "date_key": today_str, "timestamp": now_str, "share_name": best["sym"],
        "option_contract": f"{best['sym']} ${best['strike']:g} Call · exp {exp_d.strftime('%d %b %Y')}",
        "strike_price": float(best["strike"]), "expiry_days": int(days_to_expiry), "underlying_spot": float(best["spot"]),
        "lot_size": OPTION_MULTIPLIER, "entry_premium": ask, "current_option_price": bid,
        "target_premium": round(ask * 1.60, 2), "stop_loss_premium": round(ask * 0.50, 2),
        "total_capital": float(best["cost_gbp"]), "ai_confidence": score, "implied_vol": round(float(best["iv"]), 1),
        "status": "ACTIVE", "pnl_pct": round((bid - ask) / ask * 100.0, 2), "last_audited": now_str
    }

    with DB_LOCK:
        con = duckdb.connect(DB_PATH, read_only=False)
        try:
            con.execute("""
                INSERT OR REPLACE INTO daily_options_journal 
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'ACTIVE', ?, ?)
            """, [sig[k] for k in ["date_key", "timestamp", "share_name", "option_contract", "strike_price", "expiry_days", "underlying_spot", "lot_size", "entry_premium", "current_option_price", "target_premium", "stop_loss_premium", "total_capital", "ai_confidence", "implied_vol", "pnl_pct", "last_audited"]])
        finally:
            con.close()

    if supabase:
        try:
            c_check = supabase.table("options_journal").select("date_key").eq("date_key", today_str).execute()
            if not c_check.data:
                supabase.table("options_journal").insert(sig).execute()
        except Exception as e:
            record_db_error("Insert Option", e)

    return sig

def audit_real_option(opt, now_str: str, today_date):
    """Re-quote a real option contract. Checks happen when the app updates during US market hours, so intraday
    spikes between updates can be missed. If no live quote is available the last known price is kept."""
    share = str(opt["share_name"])
    strike = float(opt["strike_price"])
    entry_prem = float(opt["entry_premium"])
    target_prem = float(opt["target_premium"])
    stop_prem = float(opt["stop_loss_premium"])
    opt_dt = _to_date(opt["date_key"]) or today_date
    expiry_dt = opt_dt + timedelta(days=int(opt["expiry_days"]))
    last_price = float(opt["current_option_price"])
    elapsed = max(0, (today_date - opt_dt).days)

    bid = None
    if us_options_open():
        q = get_option_quote(share, expiry_dt.strftime("%Y-%m-%d"), strike)
        if q:
            bid = q["bid"]
    live = bid if bid is not None else last_price

    status = "ACTIVE"
    if bid is not None:
        if bid >= target_prem:
            status = "🎯 WIN (TARGET HIT)"
        elif bid <= stop_prem:
            status = "🛑 LOSS (STOPPED OUT)"
    if status == "ACTIVE" and today_date >= expiry_dt:
        status = "⏱️ EXPIRED (TIME EXIT)"
        try:
            hh = _download_history_raw(share, "5d")
            if not hh.empty:
                live = max(0.0, float(hh["Close"].iloc[-1]) - strike)     # value at expiry = intrinsic value
        except Exception:
            pass
    elif status == "ACTIVE" and elapsed >= MAX_HOLD_CALENDAR_DAYS:
        status = "⏱️ EXPIRED (TIME EXIT)"

    pnl_pct = round(((live - entry_prem) / entry_prem) * 100.0, 2) if entry_prem > 0 else 0.0
    with DB_LOCK:
        con = duckdb.connect(DB_PATH, read_only=False)
        try:
            con.execute("""
                UPDATE daily_options_journal
                SET current_option_price = ?, pnl_pct = ?, status = ?, last_audited = ?
                WHERE date_key = ?
            """, [live, pnl_pct, status, now_str, opt["date_key"]])
        finally:
            con.close()
    cloud_update("options_journal", {"date_key": opt["date_key"]},
                 {"status": status, "current_option_price": live, "pnl_pct": pnl_pct, "last_audited": now_str}, "Audit Option Sync")

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
        return fail("It is outside buying hours (08:30–16:00 London on trading days; shut at weekends and on bank holidays)")
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

def analyse_share(ticker: str, df: pd.DataFrame, today_date, market_open: bool, ftse_series: pd.Series = None):
    """Cheap checks every share gets. Returns (code, info); code is one of no_data / illiquid / wild / ok."""
    if df is None or df.empty:
        return "no_data", None
    dates = _bar_dates(df)
    live_bar_today = (dates[-1] == today_date)
    partial_bar = live_bar_today and market_open
    live_price = float(df["Close"].iloc[-1])
    df_model = df.iloc[:-1] if partial_bar else df          # model sees COMPLETED bars only (train/inference parity)
    if len(df_model) < 55 or not math.isfinite(live_price) or live_price <= 0:
        return "no_data", None
    recent = df_model.tail(20)
    avg_turnover = float((recent["Close"] * recent["Volume"]).mean() / 100.0)      # pence -> £
    if (not math.isfinite(avg_turnover)) or live_price < MIN_PRICE_P or avg_turnover < MIN_AVG_TURNOVER_GBP:
        return "illiquid", None
    last10 = df_model.tail(10)
    frozen = int(((last10["High"] - last10["Low"]).abs() <= last10["Close"].abs() * 0.0005).sum()) + int((last10["Volume"] <= 0).sum())
    if frozen >= 3:
        return "wild", None                                  # suspended / not really trading
    ctx = compute_context(df_model, ftse_series)
    if float(ctx.get("atr_pct", 0.0)) > MAX_ATR_PCT:
        return "wild", None
    return "ok", {"dates": dates, "live_bar_today": live_bar_today, "live_price": live_price, "df_model": df_model,
                  "avg_turnover": avg_turnover, "ctx": ctx}

def _model_prob(ml_bundle, feature_cols, df_model):
    """Raw blended model probability (before news and calibration) + the feature row, from completed bars only."""
    d2 = df_model.copy()
    d2.reset_index(inplace=True)
    feats = engineer_features(d2)
    if feats is None or feats.empty:
        raise ValueError("no features")
    latest = feats.iloc[-1:].copy()
    p1 = float(ml_bundle['lgb'].predict_proba(latest[feature_cols])[:, 1][0])
    p2 = float(ml_bundle['catboost'].predict_proba(latest[feature_cols])[:, 1][0])
    p = 0.5 * p1 + 0.5 * p2
    if not math.isfinite(p):
        raise ValueError("model returned a non-number")
    return p, latest

def expected_value_pct(p_win: float, entry: float, target: float, stop: float, friction_pct: float) -> float:
    """What a pick is worth on average, in %, after ALL costs: chance × net gain − (1 − chance) × net loss."""
    if entry <= 0:
        return 0.0
    gain = (target - entry) / entry * 100.0 - friction_pct
    loss = (entry - stop) / entry * 100.0 + friction_pct
    return round(p_win * gain - (1.0 - p_win) * loss, 2)

def run_predictions():
    """Looks at EVERY FTSE 350 share plus liquid AIM shares:
       1) download prices for all of them (in parallel; quiet shares are re-checked less often),
       2) cheap checks for everyone (easy to trade, not suspended, not wildly jumpy) + an AI score,
       3) the best finalists get a fresh price, the news check, past lessons, the market, and the evidence learner,
       4) only picks expected to make money AFTER all costs are kept."""
    stats = {"listed": 0, "priced": 0, "no_data": 0, "illiquid": 0, "wild": 0, "scanned": 0, "uptrend": 0, "held": 0,
             "model_errors": 0, "low_score": 0, "candidates": 0, "deep": 0, "rns_blocked": 0, "sector_blocked": 0,
             "qualified": 0, "stale": 0, "logged": 0, "paused": False, "capacity": {}, "window_open": False, "source": "",
             "partial": False, "sector_breadth": {}, "leaders": [], "ftse_20d": 0.0, "breadth_pct": None, "seconds": 0.0,
             "backfilled": 0, "evidence_ready": False}
    t_start = time.time()
    ml_bundle = get_ml_model()
    if ml_bundle is None:
        stats["error"] = "The AI model file wasn't found, so picks are paused (nothing is guessed)."
        st.session_state["scan_stats"] = stats
        return pd.DataFrame(), False
    if engineer_features is None:
        stats["error"] = "The feature_engine module wasn't found, so the AI can't score shares. Picks are paused (nothing is guessed)."
        st.session_state["scan_stats"] = stats
        return pd.DataFrame(), False

    feature_cols = ml_bundle['feature_cols']
    try:
        stats["backfilled"] = backfill_trade_context()
    except Exception:
        stats["backfilled"] = 0
    closed_df = load_closed_trades_df()
    check_and_auto_retrain_model(feature_cols, closed_df)
    ml_bundle = get_ml_model() or ml_bundle
    feature_cols = ml_bundle['feature_cols']
    calibrate, cal_info = build_calibrator(closed_df)
    st.session_state["cal_info"] = cal_info
    evidence = build_evidence_model(closed_df)
    st.session_state["evidence_model"] = evidence
    stats["evidence_ready"] = bool(evidence.get("ready"))

    now_lon = london_now()
    today_date = now_lon.date()
    market_open = is_lse_market_open()
    window_open = is_entry_window_open()
    stats["window_open"] = window_open

    active_held = get_currently_active_tickers()
    active_sectors = get_active_sector_exposure()
    saturated_sectors = {sec for sec, cnt in active_sectors.items() if cnt >= MAX_ACTIVE_PER_SECTOR}
    cooldown_tickers = get_recent_cooldown_tickers()
    macro_regime = get_cross_asset_macro_regime()
    entries_paused = is_market_stress(macro_regime)
    stats["paused"] = entries_paused
    stats["capacity"] = get_portfolio_capacity()

    universe, uni_sectors, source = get_uk_universe()
    stats["listed"], stats["source"] = len(universe), source
    box = get_market_box()
    quiet_seen = box.setdefault("quiet_seen", {})
    ttl_for = lambda t: QUIET_TTL_SEC if (time.time() - quiet_seen.get(t, 0.0)) < QUIET_TTL_SEC else SWEEP_TTL_SEC
    prog = st.progress(0, text=f"Downloading prices for {len(universe):,} UK shares...")
    def _tick(done, total):
        try:
            prog.progress(min(1.0, done / max(1, total)), text=f"Downloading prices: {done:,} of {total:,} still needed")
        except Exception:
            pass
    complete = prefetch_histories(universe, "120d", ttl=ttl_for, workers=SCAN_WORKERS, progress=_tick, time_budget=SCAN_TIME_BUDGET_SEC)
    stats["partial"] = not complete

    ftse_series = get_ftse_close_series()
    try:
        stats["ftse_20d"] = round((float(ftse_series.iloc[-1]) / float(ftse_series.iloc[-21]) - 1.0) * 100.0, 2) if len(ftse_series) > 21 else 0.0
    except Exception:
        stats["ftse_20d"] = 0.0

    breadth, leaders, candidates = {}, [], []
    for ticker in universe:
        df = peek_history(ticker, "120d")
        if df.empty:
            stats["no_data"] += 1
            continue
        stats["priced"] += 1
        try:
            code, info = analyse_share(ticker, df, today_date, market_open, ftse_series)
        except Exception:
            stats["model_errors"] += 1
            continue
        if code != "ok":
            stats[code] += 1
            if code == "illiquid":
                quiet_seen[ticker] = time.time()
            continue
        quiet_seen.pop(ticker, None)
        stats["scanned"] += 1
        sector = SECTOR_MAP.get(ticker) or uni_sectors.get(ticker) or peek_sector(ticker) or "Unclassified"
        b = breadth.setdefault(sector, [0, 0])
        b[1] += 1
        if info["ctx"].get("uptrend"):
            stats["uptrend"] += 1
            b[0] += 1
        if "rs20" in info["ctx"]:
            leaders.append((float(info["ctx"]["rs20"]), ticker, sector))
        if ticker in active_held or ticker in cooldown_tickers or sector in saturated_sectors:
            stats["held"] += 1
            continue
        try:
            p_raw, _ = _model_prob(ml_bundle, feature_cols, info["df_model"])
        except Exception:
            stats["model_errors"] += 1
            continue
        base_conf = int(min(95, max(25, round(calibrate(max(0.01, min(0.99, p_raw))) * 100.0))))
        if base_conf < MIN_QUALIFY_CONFIDENCE - DEEP_CHECK_MARGIN:
            stats["low_score"] += 1
            continue
        candidates.append((base_conf, ticker))
    breadth_pct = round(stats["uptrend"] / stats["scanned"] * 100.0, 1) if stats["scanned"] else None
    stats["breadth_pct"] = breadth_pct
    stats["sector_breadth"] = {k: (v[0], v[1]) for k, v in breadth.items()}
    stats["leaders"] = [(t, sec, round(rs, 1)) for rs, t, sec in sorted(leaders, reverse=True)[:8]]

    candidates.sort(key=lambda x: (-x[0], x[1]))
    stats["candidates"] = len(candidates)
    finalists = [t for _, t in candidates[:MAX_DEEP_CHECKS]]
    stats["low_score"] += max(0, len(candidates) - len(finalists))
    if finalists:
        prog.progress(1.0, text=f"Double-checking the best {len(finalists)} shares with fresh prices...")
        prefetch_histories(finalists, "120d", ttl=FRESH_PRICE_TTL_SEC, workers=6)

    results = []
    for ticker in finalists:
        try:
            df = peek_history(ticker, "120d", max_age=FRESH_PRICE_MAX_AGE_SEC)
            if df.empty:
                stats["stale"] += 1                          # no fresh price -> never buy on an old one
                continue
            code, info = analyse_share(ticker, df, today_date, market_open, ftse_series)
            if code != "ok":
                stats[code] += 1
                continue
            live_price, df_model, dates = info["live_price"], info["df_model"], info["dates"]
            avg_turnover = info["avg_turnover"]
            ctx = dict(info["ctx"])
            if breadth_pct is not None:
                ctx["breadth"] = breadth_pct
            sector = SECTOR_MAP.get(ticker) or uni_sectors.get(ticker) or peek_sector(ticker) or "Unclassified"
            if sector == "Unclassified":
                sector = get_ticker_sector(ticker, resolve=True)
            if sector in saturated_sectors:
                stats["sector_blocked"] += 1
                continue
            p_raw, latest = _model_prob(ml_bundle, feature_cols, df_model)
            rns_data = evaluate_rns_nlp_sentiment(ticker)
            blocked = "Dilution" in rns_data["status"]
            if blocked:
                stats["rns_blocked"] += 1
            blended = max(0.01, min(0.99, p_raw + rns_data['delta']))

            atr = float(latest['atr_14'].values[0])
            if not math.isfinite(atr) or atr <= 0:
                atr = live_price * float(ctx.get("atr_pct", 2.0)) / 100.0
            atr_pct = round((atr / live_price) * 100.0, 2)
            target = round(live_price + (1.8 * atr), 2)
            stop = round(live_price - (1.2 * atr), 2)
            gross_return_pct = ((target - live_price) / live_price) * 100.0

            spread_pct = estimate_spread_pct(ticker, avg_turnover)
            commission = get_commission_gbp()
            friction_pct = compute_friction_pct(ticker, spread_pct, commission)
            net_return_pct = round(gross_return_pct - friction_pct, 1)
            tax_regime = "AIM (0% SDRT)" if is_aim_exempt(ticker) else "Main (0.5% SDRT)"

            base_confidence = int(min(95, max(25, round(calibrate(blended) * 100.0))))
            learner = get_self_learning_adjustment(ticker, atr_pct, closed_df)
            macro_delta, macro_note = compute_macro_lead_lag_adjustment(ticker, sector, macro_regime)
            ev_delta, ev_notes = evidence_adjustment(evidence, ctx)
            total_delta = int(learner["delta"] + macro_delta + ev_delta)
            final_confidence = int(min(98, max(25, base_confidence + total_delta)))
            ev_pct = expected_value_pct(final_confidence / 100.0, live_price, target, stop, friction_pct)

            shares = max(1, int((POSITION_SIZE_GBP * 100) / live_price))
            capital = round(shares * live_price / 100.0, 2)            # what those shares really cost (£)
            is_qualified = (not blocked and final_confidence >= MIN_QUALIFY_CONFIDENCE and net_return_pct >= MIN_NET_RETURN_PCT and ev_pct > MIN_EV_PCT)
            if blocked:
                reason = f"Bad news: {rns_data['nlp_tags']}"
            elif final_confidence < MIN_QUALIFY_CONFIDENCE:
                reason = f"AI confidence {final_confidence}% is below the {MIN_QUALIFY_CONFIDENCE}% needed"
            elif net_return_pct < MIN_NET_RETURN_PCT:
                reason = f"Possible profit after costs is only {net_return_pct:+.1f}% (needs {MIN_NET_RETURN_PCT:.0f}%+)"
            elif ev_pct <= MIN_EV_PCT:
                reason = f"Expected to lose {ev_pct:+.2f}% on average once costs are counted"
            else:
                reason = "Passed every check"

            raw_feature_map = {c: float(latest[c].values[0]) for c in feature_cols if c in latest.columns}
            feature_snapshot = {
                "raw_features": raw_feature_map, "atr_pct": atr_pct, "raw_prob": round(blended, 4),
                "base_ml_conf": base_confidence, "learner_delta": learner["delta"], "macro_delta": macro_delta,
                "evidence_delta": ev_delta, "ev_pct": ev_pct, "ctx": ctx, "sector": sector, "spread_pct": spread_pct,
                "commission_gbp": commission, "friction_pct": round(get_sdrt_pct(ticker) + spread_pct, 3),
                "avg_turnover_gbp": round(avg_turnover, 0), "tax_regime": tax_regime, "nlp_tags": rns_data["nlp_tags"],
                "entry_bar": str(df_model.index[-1].date()) if hasattr(df_model.index[-1], "date") else str(dates[-1]),
                "cal_mode": cal_info["mode"],
            }
            note_parts = [learner["reason"], f"Macro: {macro_note}"]
            if ev_notes:
                note_parts.append("Evidence: " + "; ".join(ev_notes))
            results.append({
                "Ticker": ticker, "Company": ticker, "Sector": sector, "Tax Regime": tax_regime,
                "Friction": f"-{friction_pct:.2f}%", "Price (p)": round(live_price, 2),
                "Expected Return": f"+{net_return_pct}% Net", "ReturnNum": net_return_pct,
                "Expected value": ev_pct, "Vs FTSE (20d)": ctx.get("rs20"),
                "Est. Time to Target": "3-7 Days", "Recommended Shares": f"{shares} shares",
                "Total Cost (£)": gbp(capital), "Capital": capital,
                "Target (p)": target, "Stop Loss (p)": stop,
                "Base ML": f"{base_confidence}%", "Learner Delta": f"{total_delta:+d}%",
                "Learner Note": " | ".join(note_parts),
                "AI Win Confidence": f"{final_confidence}%", "Adjusted Score": final_confidence,
                "RNS": rns_data['status'], "RNSHeadline": rns_data['headline'],
                "Liquidity (£m/day)": round(avg_turnover / 1e6, 1),
                "Loggable": bool(info["live_bar_today"] and window_open),
                "FeaturesDict": feature_snapshot, "Qualified": is_qualified, "Rejection Reason": reason,
            })
        except Exception:
            stats["model_errors"] += 1
    prog.empty()
    stats["deep"] = len(results)
    stats["seconds"] = round(time.time() - t_start, 1)
    if not results:
        st.session_state["scan_stats"] = stats
        return pd.DataFrame(), False

    df_out = pd.DataFrame(results).sort_values(by=["Qualified", "Expected value", "Adjusted Score"], ascending=[False, False, False]).reset_index(drop=True)
    qualified_only = df_out[df_out["Qualified"] == True].copy().reset_index(drop=True)
    stats["qualified"] = int(len(qualified_only))

    if not qualified_only.empty and not entries_paused:
        for _, sig in qualified_only.iterrows():
            if not bool(sig.get("Loggable", False)):
                continue
            if len(get_todays_logged_equities()) >= MAX_DAILY_EQUITY_TRADES:
                break
            if log_equity_signal_safely(sig.to_dict(), enforce_sector_cap=True):
                stats["logged"] += 1

    st.session_state["scan_stats"] = stats
    return df_out, not qualified_only.empty

# ==============================================================================
# 9. PERFORMANCE ANALYTICS (HONEST NUMBERS)
# ==============================================================================
def prep_perf_frame(df: pd.DataFrame, cap_col: str, is_option: bool = False) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    d = df[df["status"].apply(classify_outcome) != "ACTIVE"].copy()
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
.rbar.c{margin:12px 0 12px;height:8px;}
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
[data-baseweb="tab-list"]{gap:6px;flex-wrap:wrap;}
button[data-baseweb="tab"],button[data-testid="stTab"],.stTabs button[role="tab"]{height:auto !important;padding:12px 18px !important;border-radius:12px 12px 0 0;}
button[data-baseweb="tab"],button[data-baseweb="tab"] p,button[data-baseweb="tab"] div,button[data-baseweb="tab"] span,
button[data-testid="stTab"] p,.stTabs button[role="tab"] p{font-size:1.15rem !important;font-weight:700 !important;line-height:1.3 !important;}
button[data-baseweb="tab"][aria-selected="true"],button[data-testid="stTab"][aria-selected="true"]{background:var(--soft);}
.mini.opt{border-left:4px solid var(--a1);}
.brief{border:1px solid var(--line);border-radius:20px;padding:18px 22px;margin:4px 0 14px;background:linear-gradient(120deg,rgba(99,102,241,.10),rgba(6,182,212,.08));}
.brief .bh{font-size:.78rem;font-weight:800;letter-spacing:.06em;text-transform:uppercase;opacity:.7;margin-bottom:6px;}
.brief .bt{font-size:1.02rem;line-height:1.65;}
.tags{display:flex;gap:6px;flex-wrap:wrap;margin-bottom:8px;}
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
.stButton>button{border-radius:12px;font-weight:600;}
@media(max-width:700px){button[data-baseweb="tab"],button[data-testid="stTab"]{padding:9px 11px !important;}button[data-baseweb="tab"],button[data-baseweb="tab"] p,button[data-baseweb="tab"] div,button[data-baseweb="tab"] span,button[data-testid="stTab"] p,.stTabs button[role="tab"] p{font-size:1rem !important;}.fn{grid-template-columns:120px 1fr 56px;font-size:.78rem;}.hb{grid-template-columns:96px 1fr 80px;}.sc .val{font-size:1.8rem;}.hero h1{font-size:1.5rem;}}
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

def funnel_html(rows: list) -> str:
    rows = [(lbl, int(max(0, n or 0))) for lbl, n in rows]
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
    rows = [(sec, up / n * 100.0, up, n) for sec, (up, n) in breadth.items() if n >= min_n and sec != "Unclassified"]
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
    chips = [f'<span class="ld"><b>{esc(str(t).replace(".L", ""))}</b><span class="{"pos" if rs >= 0 else "neg"}">{rs:+.1f}%</span> <span class="sub2">{esc(sec)}</span></span>'
             for t, sec, rs in leaders]
    return _collapse('<div class="leaders">' + "".join(chips) + '</div>')

def option_short_name(contract) -> str:
    return str(contract).split(" · ")[0]

def option_mini_html(o) -> str:
    entry, target, stop = _num(o["entry_premium"]), _num(o["target_premium"]), _num(o["stop_loss_premium"])
    live, pnl = _num(o["current_option_price"]), _num(o["pnl_pct"])
    real = is_real_option(o["option_contract"])
    unit = "$" if real else ""
    src = '<span class="tag good">Real US quote</span>' if real else '<span class="tag warn">Old practice price</span>'
    cls = "pos" if pnl >= 0 else "neg"
    return _collapse(f"""
    <div class="mini opt">
      <div class="tags"><span class="tag info">Option</span>{src}</div>
      <div class="row"><span class="mtk">{esc(option_short_name(o['option_contract']))}</span><span class="mp {cls}">{pnl:+.2f}%</span></div>
      {price_bar_html(entry, target, stop, live, compact=True)}
      <div class="sub2">bought {unit}{entry:,.2f} · now {unit}{live:,.2f} · cost {gbp(o['total_capital'])} · since {esc(str(o['date_key']))}</div>
    </div>
    """)

def _fmt_pct(v) -> str:
    try:
        v = float(v)
        return f"{v:+.1f}%" if math.isfinite(v) else "–"
    except Exception:
        return "–"

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
        <div><b>Shares</b>{esc(row['Recommended Shares'])}</div><div><b>Money in</b>{gbp(row.get('Capital', POSITION_SIZE_GBP))}</div>
        <div><b>Easy to trade</b>£{row.get('Liquidity (£m/day)', 0)}m a day</div><div><b>Costs</b>{esc(row['Friction'])}</div>
        <div><b>Worth on average</b>{_fmt_pct(row.get('Expected value'))} after costs</div><div><b>Vs FTSE 100</b>{_fmt_pct(row.get('Vs FTSE (20d)'))} (20 days)</div>
      </div>
      <div class="note">🧠 {esc(row['Learner Note'])}</div>
    </div>
    """)

# ---------- charts ----------
def show_fig(fig, key=None):
    if not PLOTLY_OK or fig is None:
        return
    try:
        st.plotly_chart(fig, width="stretch", key=key)
    except TypeError:
        st.plotly_chart(fig, use_container_width=True, key=key)

def _base_layout(fig, height=320, title=None):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=44 if title else 16, b=10), title=title,
                      legend=dict(orientation="h", y=1.14), hovermode="x unified")
    return fig

def chart_cumulative_pnl(perf_df: pd.DataFrame, title: str, key: str = None):
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
    show_fig(_base_layout(fig, 320, title), key=key)

def chart_outcome_donut(perf: dict, title: str = "How your trades ended", key: str = None):
    labels = ["Hit the goal", "Safety exit", "Sold with no loss", "Time ran out"]
    vals = [perf["wins"], perf["losses"], perf["be"], perf["expired"]]
    if not PLOTLY_OK:
        st.bar_chart(pd.DataFrame({title: vals}, index=labels))
        return
    fig = go.Figure(go.Pie(labels=labels, values=vals, hole=0.62, sort=False,
                           marker=dict(colors=["#16a34a", "#dc2626", "#d97706", "#64748b"]), textinfo="label+percent"))
    fig.update_layout(height=320, margin=dict(l=10, r=10, t=44, b=10), title=title, showlegend=False)
    show_fig(fig, key=key)

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

mkt_state = get_market_state()
market_is_open = is_lse_market_open()

st.sidebar.header("⚙️ Settings")
st.sidebar.caption("🔭 Looks at **every FTSE 350 share plus liquid AIM shares** each time.")
st.sidebar.number_input("Broker fee per trade (£)", min_value=0.0, max_value=25.0, value=DEFAULT_COMMISSION_GBP, step=0.5, key="commission_gbp",
                        help="Charged when you buy and again when you sell. Included in every profit/loss number you see. Many UK apps charge £0 — "
                             "set yours, because on a £500 trade a £3 fee each way costs 1.2%, which decides which picks are worth it.")
broker_mode = "paper trade"      # everything here is pretend money
if "auto_loop_toggle" not in st.session_state or st.session_state.get("last_market_open_state") != market_is_open:
    st.session_state["auto_loop_toggle"] = market_is_open
    st.session_state["last_market_open_state"] = market_is_open
auto_mode = st.sidebar.toggle("Auto-refresh", key="auto_loop_toggle", help="Turns on by itself when the London market opens and off when it closes (including bank holidays).")
refresh_interval_sec = st.sidebar.selectbox("Check every", [300, 900, 3600], format_func=lambda x: f"{x//60} minutes",
                                            help="Open trades are checked every time. Prices for the whole market are re-downloaded at most every 15 minutes; finalists always get a fresh price.")

with st.sidebar.expander("🛡️ Built-in safety rules"):
    st.markdown(
        f"- Max **{MAX_DAILY_EQUITY_TRADES} new trades** per day\n"
        f"- Max **{MAX_ACTIVE_PER_SECTOR} open trades** in any one sector\n"
        f"- Max **{MAX_OPEN_POSITIONS} open trades** in total (and about £{MAX_OPEN_RISK_GBP:.0f} at risk)\n"
        f"- Once a trade is about **{int(BREAKEVEN_TRIGGER_RATIO * 100)}%** of the way to its goal, its safety exit moves up so it can't lose money\n"
        f"- A stock isn't bought again for **{COOLDOWN_CALENDAR_DAYS} days** after you sell it\n"
        f"- Buying only **{ENTRY_WINDOW_START.strftime('%H:%M')}–{ENTRY_WINDOW_END.strftime('%H:%M')}** London time, using real prices\n"
        f"- Skips shares under {MIN_PRICE_P:.0f}p, trading under £{MIN_AVG_TURNOVER_GBP / 1e6:.0f}m a day, suspended, or swinging more than {MAX_ATR_PCT:.0f}% on a normal day\n"
        f"- Only buys when a pick is **expected to make money after all costs**\n"
        f"- Buying pauses if the UK market falls sharply in a week\n"
        f"- Shut at weekends and on UK bank holidays; closes at 12:30 on Christmas Eve and New Year's Eve\n"
        f"- Every result includes UK stamp duty, spread and your broker fee"
    )

with st.sidebar.expander("🏖️ Market holidays"):
    for d_, name_ in upcoming_holidays(5):
        st.write(f"**{datetime.strptime(d_, '%Y-%m-%d').strftime('%a %d %b %Y')}** — {name_}")
    _hb = get_market_box().get("holidays")
    st.caption("From gov.uk's official list." if _hb and _hb[2] else "gov.uk isn't reachable right now, so these come from the standard bank-holiday rules.")

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
    _ub = get_market_box().get("universe")
    if _ub:
        st.write(f"{'🟢' if _ub[4] else '🟠'} Share list: {len(_ub[1]):,} shares — {_ub[3]}")
    jc, hi_ = journal_counts(), st.session_state.get("hydrate_info", {})
    st.write(f"📒 Journal: {jc['total']} share trades — {jc['open']} open, {jc['closed']} finished · {jc['opt_total']} option ideas ({jc['opt_open']} open)")
    if supabase:
        st.write(f"☁️ Backup: {hi_.get('remote', 0)} trades online ({hi_.get('remote_active', 0)} open) · last sync added {hi_.get('restored', 0)}, "
                 f"{hi_.get('kept_local', 0)} already up to date here, {hi_.get('bad_rows', 0)} unreadable")
    if not st_autorefresh:
        st.caption("Auto-refresh library not installed — use “Find new picks now” to update.")

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

macro_bar = get_cross_asset_macro_regime()

def macro_chip(label, v):
    cls = "up" if v >= 0 else "down"
    arrow = "▲" if v >= 0 else "▼"
    return f'<span class="chip {cls}">{esc(label)} {arrow} {v:+.1f}%</span>'

stress_chip = '<span class="chip down">⛔ Buying paused — UK market fell sharply this week</span>' if is_market_stress(macro_bar) else ""
if not mkt_state["trading_day"]:
    mkt_chip = f'<span class="chip down">🏖️ {esc(mkt_state["reason"])} · opens {esc(next_open_text())}</span>'
elif market_is_open:
    mkt_chip = '<span class="chip up">UK market OPEN</span>' + (f'<span class="chip">⏰ {esc(mkt_state["reason"])}</span>' if mkt_state.get("half_day") else "")
else:
    mkt_chip = f'<span class="chip down">UK market CLOSED · opens {esc(next_open_text())}</span>'
feed_chip = '<span class="chip down">⚠️ Price feed problem</span>' if st.session_state.get("price_feed_problem") else ""
_ps = st.session_state.get("scan_stats", {}) or {}
pulse_chip = (f'<span class="chip {"up" if (_ps.get("breadth_pct") or 0) >= 50 else "down"}">🔭 {_ps["scanned"]:,} shares checked · {(_ps.get("breadth_pct") or 0):.0f}% rising</span>'
              if _ps.get("scanned") else "")
st.markdown(_collapse(f"""
<div class="hero">
  <h1>⚡ ALPHA-LSE Quant Terminal</h1>
  <div class="sub">An AI that picks UK shares and practises trading them with pretend money — so you can see if it really works.</div>
  <div class="chips">
    {mkt_chip}
    <span class="chip {'up' if supabase else 'down'}">{'☁️ Saved online' if supabase else '☁️ Not saved online'}</span>
    {macro_chip('UK market this week', macro_bar['ftse_5d'])}{macro_chip('Oil', macro_bar['brent_5d'])}{macro_chip('Copper', macro_bar['copper_5d'])}{macro_chip('£ vs $', macro_bar['gbpusd_5d'])}
    {stress_chip}{feed_chip}{pulse_chip}
    <span class="chip">Last checked {esc(st.session_state.get('last_audit_str', '-'))}</span>
  </div>
</div>
"""), unsafe_allow_html=True)

with st.expander("💡 New here? 30-second guide"):
    st.markdown(
        "- This app **pretends to buy** UK shares, about **£500 each**. It's practice — no real money is used.\n"
        "- Every time it checks, it looks at **every FTSE 350 share plus liquid AIM shares**, throws out the ones that are hard to trade or risky, and only buys picks that are expected to make money after all costs.\n"
        "- Every pick has a **Goal** (the price to sell at for a profit) and a **Safety exit** (the price to sell at to limit the loss).\n"
        "- When a trade gets about two-thirds of the way to its goal, the safety exit moves up so the trade **can't lose money** any more.\n"
        "- **AI confidence** is how sure the AI is that the goal is reached before the safety exit. It's a rough guide, not a promise.\n"
        "- Every profit/loss number is **after fees and UK stamp duty**.\n"
        "- If neither the goal nor the safety exit is reached, the trade closes after **7 days**.\n"
        "- **Today** is your daily briefing: what's open, what it might buy, and how the market looks. **Performance** tells you honestly whether it's making money. **What the AI Learned** shows how it's improving."
    )

tab_home, tab_perf, tab_learn, tab_options = st.tabs([
    "☀️ Today",
    "📈 Performance",
    "🧠 What the AI Learned",
    "🎲 Options"
])

# ==============================================================================
# 12. TODAY'S PICKS
# ==============================================================================
if "dispatched_orders" not in st.session_state:
    st.session_state["dispatched_orders"] = set()

def render_cohort(todays_df: pd.DataFrame):
    cols = st.columns(min(len(todays_df), 2))
    for pos_i, (_, t_row) in enumerate(todays_df.iterrows()):
        tkr_sym = str(t_row["ticker"])
        f_meta = _parse_features(t_row.get("features_json"))
        base_c = int(f_meta.get("base_ml_conf", 75))
        adj = int(f_meta.get("learner_delta", 0)) + int(f_meta.get("macro_delta", 0))
        sec_name = get_trade_sector(tkr_sym, f_meta)
        tax_tag = f_meta.get("tax_regime") or ("AIM (0% SDRT)" if is_aim_exempt(tkr_sym) else "Main (0.5% SDRT)")
        be_active = is_stop_breakeven_protected(float(t_row["entry_price"]), float(t_row["stop_loss"]), tkr_sym)
        with cols[pos_i % len(cols)]:
            st.markdown(trade_card_html(
                pos_i + 1, tkr_sym, sec_name, t_row["status"], float(t_row["entry_price"]), float(t_row["target_price"]),
                float(t_row["stop_loss"]), float(t_row["latest_price"]), float(t_row["pnl_pct"]), int(t_row["shares"]),
                float(t_row["capital_allocated"]), base_c + adj, base_c, adj, tax_tag,
                get_trade_friction_pct(tkr_sym, f_meta), t_row["last_audited"], be_active), unsafe_allow_html=True)

with tab_home:
    brief_box = st.container()
    if "flash" in st.session_state:
        st.info(st.session_state.pop("flash"))
    col_pos, col_picks = st.columns([1.1, 1], gap="large")
    pos_box = col_pos.container()
    with col_picks:

        todays_logged_df = get_todays_logged_equities()
        quota_filled = len(todays_logged_df) >= MAX_DAILY_EQUITY_TRADES

        cap_state = get_portfolio_capacity()
        sec_title("🎯", "Today's picks", f"{len(todays_logged_df)} of {MAX_DAILY_EQUITY_TRADES} used")
        re_scan = st.button("🔄 Find new picks now", width="stretch", type="primary")

        if not mkt_state["trading_day"]:
            st.warning(f"🏖️ **The London market is shut today ({mkt_state['reason']}).** Nothing will be bought, and open trades keep the last trading day's prices. Opens {next_open_text()}.")
        elif is_market_stress(macro_bar):
            st.error("⛔ The UK market has fallen sharply this week, so buying is paused to protect you. Picks are shown for information only.")
        elif not cap_state["ok"]:
            st.warning(f"🧯 {cap_state['reason']}. New buying resumes when some trades close.")
        elif not is_entry_window_open():
            st.info(f"🕗 The market isn't in its buying window (08:30–{entry_window_end().strftime('%H:%M')} London on trading days). Picks below are a watch-list only, so every purchase is made at a real price.")

        pulse_stale = (time.time() - float(st.session_state.get("scan_epoch", 0.0))) > PULSE_REFRESH_SEC
        if quota_filled and not re_scan and (is_new_loop_tick and pulse_stale):
            with st.spinner("Refreshing the market pulse..."):
                res_df, has_cleared = run_predictions()
                st.session_state["scan_results"] = res_df
                st.session_state["has_cleared"] = has_cleared
                st.session_state["scan_ts"] = london_now().strftime("%H:%M:%S")
                st.session_state["scan_epoch"] = time.time()
        if quota_filled and not re_scan:
            st.info(f"🔒 You've used all {MAX_DAILY_EQUITY_TRADES} picks for today. This stops over-trading. They're in Your positions.")
            with st.expander(f"📌 Today's {len(todays_logged_df)} trades in detail (costs, confidence, tax)"):
                render_cohort(todays_logged_df)
        else:
            if re_scan or is_new_loop_tick or "scan_results" not in st.session_state:
                with st.spinner("Checking every FTSE 350 + liquid AIM share and your open trades (the first check of the day takes a minute or two)..."):
                    if re_scan:
                        audit_and_reconcile_all_trades()
                        hydrate_duckdb_from_supabase()
                        st.session_state["last_audit_str"] = london_now().strftime("%d %b %H:%M")
                    res_df, has_cleared = run_predictions()
                    st.session_state["scan_results"] = res_df
                    st.session_state["has_cleared"] = has_cleared
                    st.session_state["scan_ts"] = london_now().strftime("%H:%M:%S")
                    st.session_state["scan_epoch"] = time.time()
                todays_logged_df = get_todays_logged_equities()

            if not todays_logged_df.empty:
                with st.expander(f"📌 Already bought today ({len(todays_logged_df)})", expanded=False):
                    render_cohort(todays_logged_df)

            df_all_res = st.session_state.get("scan_results", pd.DataFrame())
            held_now = get_currently_active_tickers() | set(todays_logged_df["ticker"].tolist() if not todays_logged_df.empty else [])
            df_res = (df_all_res[(df_all_res["Qualified"] == True) & (~df_all_res["Ticker"].isin(held_now))].reset_index(drop=True)
                      if (not df_all_res.empty and "Qualified" in df_all_res.columns) else pd.DataFrame())
            quota_now = len(todays_logged_df) >= MAX_DAILY_EQUITY_TRADES
            stats = st.session_state.get("scan_stats", {})
            if stats and not stats.get("error"):
                st.caption(
                    f"Last check {st.session_state.get('scan_ts', '-')} (took {stats.get('seconds', 0):.0f}s): {stats.get('scanned', 0):,} tradable shares checked out of "
                    f"{stats.get('listed', 0):,} · {stats.get('deep', 0)} finalists · {stats.get('rns_blocked', 0)} blocked by bad news · "
                    f"bought {stats.get('logged', 0)} automatically · {len(todays_logged_df)} of {MAX_DAILY_EQUITY_TRADES} picks used today")
                if stats.get("listed", 0) >= 50 and stats.get("priced", 0) < stats["listed"] * 0.5:
                    st.warning(f"📡 Prices arrived for only {stats.get('priced', 0):,} of {stats['listed']:,} shares. The price service may be busy. "
                               "Nothing is guessed: shares without prices are simply skipped, and the next check tries again.")
                elif stats.get("partial"):
                    st.info("⏳ Some shares' prices were still downloading when the time limit was reached. They'll be included in the next check.")
                if stats.get("source") and "today's" not in str(stats.get("source")):
                    st.caption(f"📃 Share list: {stats['source']}.")
            elif stats.get("error"):
                st.error(stats["error"])

            if quota_now:
                st.info(f"🔒 All {MAX_DAILY_EQUITY_TRADES} picks for today are used (they're in Your positions). This stops over-trading."
                        + (f" {len(df_res)} more share(s) passed every check — see the finalists list below." if not df_res.empty else ""))
            elif st.session_state.get("has_cleared", False) and not df_res.empty:
                st.success(f"🟢 {len(df_res)} share(s) passed every check")

                today_key = london_now().strftime('%Y-%m-%d')
                for idx, row in df_res.head(3).iterrows():
                    with st.container():
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

                with st.expander(f"📋 All {len(df_res)} picks · best first"):
                    table = df_res.rename(columns={
                        "Ticker": "Stock", "Price (p)": "Buy at (p)", "Target (p)": "Goal (p)", "Stop Loss (p)": "Safety exit (p)",
                        "Expected Return": "Possible profit", "AI Win Confidence": "AI confidence", "Recommended Shares": "Shares",
                        "RNS": "News check", "Loggable": "Ready to buy"})
                    show_cols = ["Stock", "Sector", "Buy at (p)", "Goal (p)", "Safety exit (p)", "Possible profit", "Expected value", "AI confidence", "Shares", "News check", "Ready to buy"]
                    st.dataframe(table[[c for c in show_cols if c in table.columns]], width="stretch", hide_index=True)
            else:
                st.warning("🛡️ **Nothing to buy right now.** No share passes every check (easy to trade, no bad news, room in its sector, enough AI confidence, "
                           "and expected to make money after all costs). Doing nothing is a valid choice.")

    # ---------------- market pulse (the whole UK market, from the latest check) ----------------
    stats_p = st.session_state.get("scan_stats", {}) or {}
    if stats_p.get("scanned"):
        sec_title("🌡️", "Market pulse", "the whole UK market, from this check")
        n_sc = int(stats_p.get("scanned", 0))
        up_pct = float(stats_p.get("breadth_pct") or 0.0)
        f20 = float(stats_p.get("ftse_20d", 0.0) or 0.0)
        score_cards([
            ("Shares looked at", f"{stats_p.get('listed', 0):,}", "FTSE 350 + liquid AIM", "", ""),
            ("Easy to trade", f"{n_sc:,}", f"{stats_p.get('illiquid', 0):,} too thin · {stats_p.get('wild', 0):,} too risky", "", ""),
            ("Market mood", f"{up_pct:.0f}% rising", breadth_word(up_pct), "good" if up_pct >= 55 else ("bad" if up_pct < 40 else ""), "pos" if up_pct >= 50 else "neg"),
            ("FTSE 100, last 20 days", f"{f20:+.1f}%", "the market as a whole", "good" if f20 > 0 else ("bad" if f20 < 0 else ""), "pos" if f20 >= 0 else "neg"),
        ])
        p1, p2 = st.columns([1.1, 1])
        with p1:
            st.markdown("**How today's picks were found**")
            st.markdown(funnel_html([
                ("Shares looked at", stats_p.get("listed", 0)),
                ("Prices available", stats_p.get("priced", 0)),
                ("Easy & safe to trade", n_sc),
                ("Room to buy (not held, sector free)", max(0, n_sc - int(stats_p.get("held", 0)))),
                ("Strong AI score", stats_p.get("candidates", 0)),
                ("Finalists, fully checked", stats_p.get("deep", 0)),
                ("Passed every check", stats_p.get("qualified", 0)),
            ]), unsafe_allow_html=True)
            st.caption(f"{stats_p.get('held', 0)} shares were left out because you already own them, sold them in the last {COOLDOWN_CALENDAR_DAYS} days, or their sector is full.")
        with p2:
            st.markdown("**How much of the market is rising**")
            st.markdown(gauge_html(up_pct), unsafe_allow_html=True)
            st.caption("Share of easy-to-trade shares above their 20-day average and up over 20 days.")
            heat = sector_heat_html(stats_p.get("sector_breadth", {}) or {})
            if heat:
                st.markdown("**Strongest sectors right now**")
                st.markdown(heat, unsafe_allow_html=True)
        if stats_p.get("leaders"):
            st.markdown("**Beating the FTSE 100 most over 20 days** · for information, not buy tips")
            st.markdown(leaders_html(stats_p["leaders"]), unsafe_allow_html=True)
    df_fin = st.session_state.get("scan_results", pd.DataFrame())
    if isinstance(df_fin, pd.DataFrame) and not df_fin.empty and "Rejection Reason" in df_fin.columns:
        with st.expander(f"🔍 The {len(df_fin)} finalists — and why each did or didn't pass"):
            diag = df_fin.rename(columns={"Ticker": "Stock", "Price (p)": "Price (p)", "AI Win Confidence": "AI confidence", "Rejection Reason": "Result"})
            cols_ = ["Stock", "Sector", "Price (p)", "Expected Return", "Expected value", "AI confidence", "Vs FTSE (20d)", "Liquidity (£m/day)", "Qualified", "Result"]
            st.dataframe(diag[[c for c in cols_ if c in diag.columns]], width="stretch", hide_index=True)
            st.caption("“Expected value” = what a pick is worth on average after stamp duty, spread and fees, using the AI's chance of reaching the goal first.")

# ==============================================================================
# 13. OPTIONS (PRACTICE)
# ==============================================================================
with tab_options:
    sec_title("🎲", "Options idea of the day", "real quotes · paper trading")
    st.caption("Real prices from US-listed options on UK companies (about 15 minutes delayed, in US dollars). Nothing is bought for real — it is tracked with "
               "pretend money, and it is **not** counted in your main share results.")
    opt_signal = generate_daily_options_alpha()
    if opt_signal:
        real_opt = is_real_option(opt_signal["option_contract"])
        with st.container(border=True):
            o1, o2, o3 = st.columns(3)
            if real_opt:
                with o1:
                    st.metric("Idea", opt_signal["option_contract"], help="A 'Call' option gains value if the share price goes up. Strike = the price it needs to beat.")
                    st.markdown(f"Share price when picked: **${opt_signal['underlying_spot']:.2f}**")
                with o2:
                    st.metric("Sell price now", f"${opt_signal['current_option_price']:.2f}", f"{opt_signal['pnl_pct']:+.2f}% vs cost (${opt_signal['entry_premium']:.2f})",
                              help="What you could sell it for right now (the 'bid'). You buy at the higher 'ask' price, so a new trade starts slightly down.")
                    st.markdown(f"Market's expected wobble: **{opt_signal['implied_vol']}%** a year")
                with o3:
                    st.metric("Cost to buy", f"£{opt_signal['total_capital']:,.0f}", f"${opt_signal['entry_premium'] * int(opt_signal['lot_size']):,.0f} for {int(opt_signal['lot_size'])} shares' worth", delta_color="off")
                    st.markdown(f"Trend strength: **{opt_signal['ai_confidence']}** (not a chance of winning)")
                m1, m2, m3 = st.columns(3)
                m1.info(f"🎯 Goal: ${opt_signal['target_premium']:.2f} (+60%)")
                m2.warning(f"🛑 Safety exit: ${opt_signal['stop_loss_premium']:.2f} (-50%)")
                m3.success(f"Status: **{friendly_status(opt_signal['status'])}**")
                st.caption(f"Last checked {str(opt_signal['last_audited'])[:16]} · prices are only refreshed while the US market is open.")
            else:
                st.warning("This is an older pretend-price option from before real quotes were added. It will finish normally.")
                with o1:
                    st.metric("Idea", opt_signal["option_contract"])
                with o2:
                    st.metric("Value now", f"{opt_signal['current_option_price']:.2f}p", f"{opt_signal['pnl_pct']:+.2f}% vs cost ({opt_signal['entry_premium']:.2f}p)")
                with o3:
                    st.metric("Cost", f"£{opt_signal['total_capital']:,.2f}")
                st.info(f"Status: **{friendly_status(opt_signal['status'])}**")
    else:
        st.info(st.session_state.get("opt_note") or "No option idea today.")
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
perf_opt = compute_performance(perf_opt_df)
active_opt = df_opt[df_opt["status"].apply(classify_outcome) == "ACTIVE"].copy() if not df_opt.empty else pd.DataFrame()
opt_open_unreal = float((pd.to_numeric(active_opt["total_capital"], errors="coerce").fillna(0) * pd.to_numeric(active_opt["pnl_pct"], errors="coerce").fillna(0) / 100.0).sum()) if not active_opt.empty else 0.0
opt_learn = options_learning(df_opt)

active_eq = df_eq[df_eq["status"].apply(classify_outcome) == "ACTIVE"].copy() if not df_eq.empty else pd.DataFrame()
active_cap = float(active_eq["capital_allocated"].sum()) if not active_eq.empty else 0.0
open_unreal = float((active_eq["capital_allocated"] * active_eq["pnl_pct"] / 100.0).sum()) if not active_eq.empty else 0.0
be_prot = int(sum(is_stop_breakeven_protected(float(r["entry_price"]), float(r["stop_loss"])) for _, r in active_eq.iterrows())) if not active_eq.empty else 0
open_ret = (open_unreal / active_cap * 100.0) if active_cap > 0 else 0.0
open_unreal_all = open_unreal + opt_open_unreal
n_open_all = len(active_eq) + len(active_opt)

def open_split_text() -> str:
    """'3 shares · 1 option' — the same wording wherever open trades are counted."""
    parts = [f"{len(active_eq)} share{'s' if len(active_eq) != 1 else ''}"]
    if len(active_opt):
        parts.append(f"{len(active_opt)} option{'s' if len(active_opt) != 1 else ''}")
    return " · ".join(parts)

closed_all = load_closed_trades_df()


# ==============================================================================
# 13b. OPTIONS — SCOREBOARD, OPEN IDEAS, LEARNER, HISTORY (needs the data loaded above)
# ==============================================================================
with tab_options:
    sec_title("📊", "Options scoreboard", "finished option ideas, kept apart from your share results")
    if perf_opt["n"] == 0:
        score_cards([("Options profit/loss", "£0.00", "no finished ideas yet", "", ""), ("Win rate", "–", "needs finished ideas", "", ""),
                     ("Finished ideas", "0", "one idea a day at most", "", ""), ("Open right now", str(len(active_opt)), f"{gbp(opt_open_unreal, sign=True)} so far", "", "")])
        callout("info", "⏳", "<b>No option idea has finished yet.</b> Each one closes at its goal, its safety exit, or after 7 days.")
    else:
        net_o = perf_opt["net_gbp"]
        score_cards([
            ("Options profit/loss", gbp(net_o, sign=True), f"{gbp(perf_opt['expectancy_gbp'], sign=True)} per idea on average", "good" if net_o > 0 else ("bad" if net_o < 0 else ""), "pos" if net_o > 0 else ("neg" if net_o < 0 else "")),
            ("Win rate", f"{perf_opt['target_hit_rate']:.0f}%", f"{perf_opt['wins']} won · {perf_opt['losses']} lost · {perf_opt['expired']} timed out", "", ""),
            ("Finished ideas", str(perf_opt["n"]), f"likely win rate {perf_opt['ci_low']:.0f}–{perf_opt['ci_high']:.0f}%", "", ""),
            ("Open right now", str(len(active_opt)), f"{gbp(opt_open_unreal, sign=True)} so far", "good" if opt_open_unreal > 0 else ("bad" if opt_open_unreal < 0 else ""), ""),
        ])
        if perf_opt["n"] < 20:
            callout("warn", "⏳", f"<b>Too early to judge.</b> Only {perf_opt['n']} option idea{'s have' if perf_opt['n'] != 1 else ' has'} finished. Options swing a lot, so early results are mostly luck.")
        oc1, oc2 = st.columns([3, 2])
        with oc1:
            chart_cumulative_pnl(perf_opt_df, "Options running profit/loss (£)", key="cum_options")
        with oc2:
            chart_outcome_donut(perf_opt, title="How your option ideas ended", key="outcomes_options")
        if opt_learn.get("old"):
            rl, ol = opt_learn.get("real"), opt_learn.get("old")
            q1, q2 = st.columns(2)
            with q1:
                stat_card("📡 Ideas with real US quotes", f"{rl['avg']:+.1f}% average" if rl else "none yet", f"{rl['n']} finished · {rl['win_rate']:.0f}% won" if rl else "")
            with q2:
                stat_card("🧮 Older practice-price ideas", f"{ol['avg']:+.1f}% average", f"{ol['n']} finished · estimates, not real quotes")

    sec_title("🟢", "Open option ideas", f"{len(active_opt)} running")
    if active_opt.empty:
        st.caption("No option ideas are open right now.")
    else:
        oc = st.columns(min(3, len(active_opt)))
        for i_, (_, o) in enumerate(active_opt.iterrows()):
            with oc[i_ % len(oc)]:
                st.markdown(option_mini_html(o), unsafe_allow_html=True)

    sec_title("🧠", "What the options learner has learned", "separate from the share AI")
    rn = int(opt_learn.get("recent_n", 0))
    st.markdown(_collapse(
        f'<div class="stat"><div class="l">Options experience</div><div class="v">{opt_learn["n"]} finished idea{"s" if opt_learn["n"] != 1 else ""}</div>'
        f'<div class="meter"><div style="width:{min(100.0, rn / OPT_LEARN_MIN_TRADES * 100.0):.0f}%"></div></div>'
        f'<div class="sub2">{min(rn, OPT_LEARN_MIN_TRADES)} of {OPT_LEARN_MIN_TRADES} recent ideas (last {OPT_LEARN_WINDOW_DAYS} days) needed before it may become pickier. '
        f'Until then only the “rest a share after a loss” rule is used.</div></div>'), unsafe_allow_html=True)
    rules = []
    if opt_learn["rest"]:
        rules.append("🚧 <b>Taking a break from:</b> " + ", ".join(f"{esc(k)} (~{v} day{'s' if v != 1 else ''} left)" for k, v in sorted(opt_learn["rest"].items(), key=lambda x: -x[1]))
                     + " — their recent options lost money. The break fades by itself.")
    if opt_learn["bar"] > OPT_BASE_TREND_BAR:
        rise = trend_bar_to_rise(opt_learn["bar"])
        rules.append(f"📈 <b>Pickier:</b> recent ideas lost <b>{opt_learn['wavg']:+.1f}%</b> on average, so it now needs trend strength {opt_learn['bar']:.0f}+"
                     + (f" (a 5-day rise of about <b>{rise:+.1f}%</b>)" if rise > 0 else "") + " before it picks a share.")
    elif opt_learn["ready"]:
        rules.append(f"✅ <b>Normal pickiness:</b> recent ideas made <b>{opt_learn['wavg']:+.1f}%</b> on average, so it isn't adding extra filters.")
    if rules:
        for r_ in rules:
            callout("info", "🧠", r_)
    else:
        callout("info", "🌱", "<b>No rules in force yet.</b> It hasn't seen enough finished option ideas to change anything.")
    if opt_learn.get("by_share"):
        bs = sorted(((k, v) for k, v in opt_learn["by_share"].items() if v), key=lambda x: -x[1]["avg"])
        st.markdown(hbars_html([(k, float(v["avg"]), f"{v['avg']:+.1f}% · {v['n']} idea{'s' if v['n'] != 1 else ''}") for k, v in bs[:10]]), unsafe_allow_html=True)
        st.caption("Average result of finished option ideas, per company.")

    if not df_opt.empty:
        sec_title("📜", "Every option idea")
        od = df_opt.copy()
        od["Price source"] = od["option_contract"].apply(lambda c: "Real US quote ($)" if is_real_option(c) else "Old practice price")
        od["Contract"] = od["option_contract"].apply(option_short_name)
        od["Result"] = od["status"].apply(friendly_status)
        od = od.rename(columns={"date_key": "Date", "entry_premium": "Bought at", "current_option_price": "Price now", "target_premium": "Goal",
                                "stop_loss_premium": "Safety exit", "pnl_pct": "Profit/Loss (%)", "ai_confidence": "Trend strength", "last_audited": "Last checked",
                                "total_capital": "Cost (£)"})
        ocols = ["Date", "Contract", "Price source", "Cost (£)", "Bought at", "Goal", "Safety exit", "Price now", "Profit/Loss (%)", "Trend strength", "Result", "Last checked"]
        st.dataframe(style_ledger(od[[c for c in ocols if c in od.columns]]), width="stretch", hide_index=True)
        st.caption("Option prices are per share, in US dollars for real quotes (older practice rows are in pence). Costs are in pounds for one contract.")

with tab_options:
    with st.expander("How the options idea works"):
        st.markdown(
            "- Shell, BP, HSBC, AstraZeneca, GSK, Rio Tinto, Barclays, BAT, Unilever and Vodafone also trade in New York, where their options are real and well traded.\n"
            "- The idea is a **call option** about 2% above the share price, expiring in roughly 4 weeks. It's only offered if the live quote is genuine: enough real trading, a tight buy/sell gap, and a cost under £500.\n"
            "- You \"buy\" at the asking price and are valued at the selling price, so every trade starts slightly down — that's the real cost of trading.\n"
            "- Goal is +60% and safety exit is -50%. They're checked each time the page updates while the US market is open (about 14:30–21:00 UK time), so quick spikes in between can be missed.\n"
            "- After 7 days it closes anyway.\n"
            "- **The options learner** only learns from finished option ideas and only changes the options idea — never the share AI. It rests a share after a recent option loss, "
            f"and gets pickier about the 5-day climb if recent ideas lose on average. That stricter rule only looks at the last {OPT_LEARN_WINDOW_DAYS} days, so it lifts by itself and gets re-tested.\n"
            "- ⚠️ Many UK brokers don't let UK retail investors trade US-listed options. Check yours before ever thinking about real money.")

# ==============================================================================
# 14. OVERVIEW (SCOREBOARD)
# ==============================================================================
with tab_perf:
    n_ = perf_eq["n"]
    if n_ == 0:
        sec_title("📈", "Scoreboard")
        score_cards([
            ("Share profit/loss", "£0.00", "no finished share trades yet", "", ""),
            ("Win rate", "–", "needs finished share trades", "", ""),
            ("Finished share trades", "0", (f"+ {len(perf_opt_df)} option idea{'s' if len(perf_opt_df) != 1 else ''} → 🎲 Options tab" if len(perf_opt_df) else "come back soon"), "", ""),
            ("Open right now", str(n_open_all), f"{open_split_text()} · {gbp(open_unreal_all, sign=True)} so far", "good" if open_unreal_all > 0 else ("bad" if open_unreal_all < 0 else ""), ""),
        ])
        callout("info", "⏳", "<b>No share trade has finished yet.</b> Once the first ones reach their goal or safety exit, your results and the AI's lessons will appear here."
                + (" Option results are on the 🎲 Options tab." if len(perf_opt_df) or len(active_opt) else ""))
    else:
        exp_ = perf_eq["expectancy_pct"]
        net = perf_eq["net_gbp"]
        sec_title("📈", "Scoreboard", "finished share trades, after fees & tax")
        score_cards([
            ("Share profit/loss", gbp(net, sign=True), f"{gbp(perf_eq['expectancy_gbp'], sign=True)} per trade on average", "good" if net > 0 else ("bad" if net < 0 else ""), "pos" if net > 0 else ("neg" if net < 0 else "")),
            ("Win rate", f"{perf_eq['target_hit_rate']:.0f}%", f"{perf_eq['wins']} won · {perf_eq['losses']} lost · {perf_eq['be']} saved · {perf_eq['expired']} timed out", "", ""),
            ("Finished share trades", str(n_), f"likely win rate {perf_eq['ci_low']:.0f}–{perf_eq['ci_high']:.0f}%" + (f" · + {len(perf_opt_df)} option idea{'s' if len(perf_opt_df) != 1 else ''} → 🎲 Options" if len(perf_opt_df) else ""), "", ""),
            ("Open right now", str(n_open_all), f"{open_split_text()} · {gbp(open_unreal_all, sign=True)} so far · 🛡️ {be_prot} can't lose", "good" if open_unreal_all > 0 else ("bad" if open_unreal_all < 0 else ""), ""),
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
            chart_cumulative_pnl(perf_eq_df, "Your running profit/loss (£) — going up is good", key="cum_shares")
        with ch2:
            chart_outcome_donut(perf_eq, key="outcomes_shares")

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
            ex_ = float(np.mean(excess)) if excess else None
            beat = ("" if ex_ is None else (f" On the same days, your trades <b>beat the market by {ex_:.2f}%</b> per trade on average." if ex_ >= 0
                                            else f" On the same days, your trades <b>trailed the market by {abs(ex_):.2f}%</b> per trade on average."))
            callout("info", "🇬🇧", f"<b>Compared with the UK market:</b> the FTSE 100 moved <b>{ftse_since:+.2f}%</b> since your first trade.{beat}")

    if n_ > 0:
        with st.expander("🔍 More detail"):
            b1, b2, b3, b4 = st.columns(4)
            b1.metric("Average winning trade", gbp(perf_eq['avg_win_gbp'], sign=True))
            b2.metric("Average losing trade", gbp(perf_eq['avg_loss_gbp'], sign=True))
            b3.metric("Money won ÷ money lost", f"{perf_eq['profit_factor']:.2f}" if perf_eq["profit_factor"] is not None else "no losses yet",
                      help="Above 1 means more money won than lost.")
            b4.metric("Biggest drop from a peak", gbp(perf_eq['max_dd_gbp'], sign=True), help="The worst losing streak, measured from the highest point your running total reached.")
            chart_pnl_hist(perf_eq_df)
            sg = sector_stats(perf_eq_df)
            if not sg.empty:
                tbl = sg.rename(columns={"n": "Trades", "avg": "Average %", "win_rate": "Win rate %"})[["Sector", "Trades", "Average %", "Win rate %"]].round(1)
                st.markdown("**Results by sector**")
                st.dataframe(tbl.sort_values("Average %", ascending=False), width="stretch", hide_index=True)
        if perf_opt["n"]:
            st.caption(f"🎲 Options (kept separate): {perf_opt['n']} finished, {gbp(perf_opt['net_gbp'], sign=True)} — full scoreboard on the Options tab.")


# ==============================================================================
# 14b. TODAY — YOUR POSITIONS (left column) AND THE DAILY BRIEFING (top)
# ==============================================================================
with pos_box:
    sec_title("🟢", "Your positions", open_split_text() if n_open_all else "nothing open")
    hi_ = st.session_state.get("hydrate_info", {})
    if supabase and (hi_.get("remote_active", 0) > len(active_eq) or hi_.get("bad_rows", 0)):
        callout("warn", "☁️", f"<b>The online backup lists {hi_.get('remote_active', 0)} open share trade(s), but {len(active_eq)} show here.</b> "
                              f"{hi_.get('bad_rows', 0)} backup row(s) were unreadable ({esc(hi_.get('first_error', '') or 'none')}). Press “Update now” on the Performance tab to retry. "
                              "(Trades that already closed here legitimately differ from an older backup.)")
    if n_open_all:
        cards = [("eq", r) for _, r in active_eq.sort_values("pnl_pct", ascending=False).head(12).iterrows()] if not active_eq.empty else []
        cards += [("opt", o) for _, o in active_opt.iterrows()] if not active_opt.empty else []
        for i in range(0, len(cards), 2):
            pc = st.columns(2)
            for j, (kind, r) in enumerate(cards[i:i + 2]):
                with pc[j]:
                    if kind == "opt":
                        st.markdown(option_mini_html(r), unsafe_allow_html=True)
                    else:
                        st.markdown(mini_trade_html(
                            r["ticker"], get_trade_sector(r["ticker"], r["features_json"]), float(r["entry_price"]), float(r["target_price"]),
                            float(r["stop_loss"]), float(r["latest_price"]), float(r["pnl_pct"]),
                            is_stop_breakeven_protected(float(r["entry_price"]), float(r["stop_loss"]))), unsafe_allow_html=True)
        st.caption("Dot = price now · tick = where it was bought · green side = nearer the goal · 🛡️ = can no longer lose")
        if len(active_eq) > 12:
            st.caption(f"Showing 12 of {len(active_eq)} open share trades. See all of them on the Performance tab.")
        if len(active_opt):
            st.caption("Options are paper-traded separately and are not counted in the share results — see the 🎲 Options tab.")
        sec_exp = get_active_sector_exposure()
        if sec_exp:
            with st.expander("📊 Open trades by sector"):
                chart_sector_exposure(sec_exp)
    else:
        callout("info", "🌙", "<b>Nothing open right now.</b> When a share passes every check during buying hours, it's bought (with pretend money) and appears here.")

def build_briefing() -> str:
    """A few plain-English sentences: the market, the mood, what the AI did today, your positions, your results."""
    stt = st.session_state.get("scan_stats", {}) or {}
    parts = []
    if not mkt_state["trading_day"]:
        parts.append(f"🏖️ The London market is <b>shut today</b> ({esc(mkt_state['reason'])}); it opens {esc(next_open_text())}.")
    elif market_is_open:
        parts.append(f"🟢 The London market is <b>open</b> until {mkt_state['close'].strftime('%H:%M')}" + (" — <b>a half day</b>." if mkt_state.get("half_day") else "."))
    else:
        parts.append(f"🌙 The London market is <b>closed</b>; it opens {esc(next_open_text())}.")
    if stt.get("scanned"):
        bp = float(stt.get("breadth_pct") or 0.0)
        parts.append(f"Across <b>{stt['scanned']:,}</b> easy-to-trade UK shares, <b>{bp:.0f}%</b> are rising ({breadth_word(bp).lower()}), and the FTSE 100 is "
                     f"<b>{float(stt.get('ftse_20d', 0.0)):+.1f}%</b> over 20 days.")
    if is_market_stress(macro_bar):
        parts.append("⛔ The market fell sharply this week, so <b>buying is paused</b> to protect you.")
    t_df = get_todays_logged_equities()
    names = ", ".join(str(t).replace(".L", "") for t in (t_df["ticker"].tolist() if not t_df.empty else []))
    q_n = int(stt.get("qualified", 0) or 0)
    parts.append(f"Today it has used <b>{len(t_df)} of {MAX_DAILY_EQUITY_TRADES}</b> picks" + (f" ({esc(names)})" if names else "")
                 + (f", and <b>{q_n}</b> share{'s' if q_n != 1 else ''} currently pass every check." if stt else "."))
    if n_open_all:
        parts.append(f"You have <b>{n_open_all} trade{'s' if n_open_all != 1 else ''} open</b> ({esc(open_split_text())}), "
                     f"<b class=\"{'pos' if open_unreal_all >= 0 else 'neg'}\">{gbp(open_unreal_all, sign=True)}</b> so far" + (f"; {be_prot} can no longer lose." if be_prot else "."))
    if perf_eq["n"]:
        parts.append(f"So far <b>{perf_eq['n']}</b> share trades have finished: {perf_eq['wins']} hit their goal, "
                     f"<b class=\"{'pos' if perf_eq['net_gbp'] >= 0 else 'neg'}\">{gbp(perf_eq['net_gbp'], sign=True)}</b> after all costs"
                     + (" — still too early to judge." if perf_eq["n"] < 30 else "."))
    else:
        parts.append("No share trade has finished yet, so there are no results to judge.")
    return " ".join(parts)

with brief_box:
    st.markdown(_collapse(f'<div class="brief"><div class="bh">☀️ Your daily briefing · {esc(london_now().strftime("%A %d %B, %H:%M"))}</div>'
                          f'<div class="bt">{build_briefing()}</div></div>'), unsafe_allow_html=True)
    cap_now = get_portfolio_capacity()
    todays_n = len(get_todays_logged_equities())
    score_cards([
        ("Open right now", str(n_open_all), f"{open_split_text()} · {gbp(open_unreal_all, sign=True)}", "good" if open_unreal_all > 0 else ("bad" if open_unreal_all < 0 else ""), ""),
        ("Picks used today", f"{todays_n} of {MAX_DAILY_EQUITY_TRADES}", "stops over-trading", "warn" if todays_n >= MAX_DAILY_EQUITY_TRADES else "", ""),
        ("Money at risk", gbp(cap_now["risk_gbp"], 0), f"if every open trade hit its safety exit (cap {gbp(MAX_OPEN_RISK_GBP, 0)})", "warn" if not cap_now["ok"] else "", ""),
        ("Share profit/loss", gbp(perf_eq.get("net_gbp", 0.0), sign=True) if perf_eq["n"] else "£0.00", f"{perf_eq['n']} finished · after all costs" if perf_eq["n"] else "nothing finished yet",
         ("good" if perf_eq.get("net_gbp", 0) > 0 else ("bad" if perf_eq.get("net_gbp", 0) < 0 else "")) if perf_eq["n"] else "", ("pos" if perf_eq.get("net_gbp", 0) > 0 else ("neg" if perf_eq.get("net_gbp", 0) < 0 else "")) if perf_eq["n"] else ""),
    ])

# ==============================================================================
# 15. ALL TRADES (bottom of the Performance tab)
# ==============================================================================
with tab_perf:
    sec_title("📖", "All trades", "shares and options")
    j1, j2 = st.columns([4, 1])
    with j1:
        st.caption("Share profit/loss figures are after fees and UK stamp duty. Share prices are in pence; option prices are in US dollars. Prices from Yahoo can be about 15 minutes behind.")
    with j2:
        if st.button("🔄 Update now", width="stretch"):
            audit_and_reconcile_all_trades()
            hydrate_duckdb_from_supabase()
            st.session_state["last_audit_str"] = london_now().strftime("%d %b %H:%M")
            st.toast("Trades updated with the latest prices.", icon="✅")
            st.rerun()


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
    if not df_opt.empty:
        d = df_opt.copy()
        d["Type"] = d["option_contract"].apply(lambda c: "Option (real quote, $)" if is_real_option(c) else "Option (old practice)")
        d["Sector"] = d["share_name"].apply(lambda s: get_ticker_sector(ADR_TO_LSE.get(str(s), f"{s}.L"), resolve=False))
        d["_o"] = d["status"].apply(classify_outcome)
        d["Result"] = d["status"].apply(friendly_status)
        master_list.append(d.rename(columns={"date_key": "Date", "option_contract": "Stock", "entry_premium": "Bought at (p)",
                                             "current_option_price": "Price now (p)", "target_premium": "Goal (p)",
                                             "stop_loss_premium": "Safety exit (p)", "pnl_pct": "Profit/Loss (%)", "last_audited": "Last checked"}))

    if master_list:
        cols_keep = ["Date", "Stock", "Sector", "Type", "Bought at (p)", "Goal (p)", "Safety exit (p)", "Price now (p)", "Profit/Loss (%)", "Result", "Last checked", "_o"]
        master_df = pd.concat([m[[c for c in cols_keep if c in m.columns]] for m in master_list], ignore_index=True)
        master_df = master_df.sort_values("Date", ascending=False, kind="stable").reset_index(drop=True)

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

    sec_title("🔬", "What's been working", "the evidence learner — from your own finished trades")
    ev_model = build_evidence_model(closed_all)
    n_ctx = int(ev_model.get("n", 0))
    st.markdown(_collapse(
        f'<div class="stat"><div class="l">Evidence collected</div><div class="v">{n_ctx} clear win/loss trades with their market conditions</div>'
        f'<div class="meter"><div style="width:{min(100.0, n_ctx / EVIDENCE_MIN_DECIDED * 100.0):.0f}%"></div></div>'
        f'<div class="sub2">{min(n_ctx, EVIDENCE_MIN_DECIDED)} of {EVIDENCE_MIN_DECIDED} needed before it may change the AI\'s confidence. '
        f'Past trades get their conditions rebuilt from price history automatically.</div></div>'), unsafe_allow_html=True)
    if ev_model.get("p0") is not None and n_ctx >= 10:
        p0 = ev_model["p0"] * 100.0
        st.caption(f"Overall, {p0:.0f}% of these trades reached their goal. Bars show how each condition compares (green = better than usual).")
        shown = 0
        for key, title, bins in EVIDENCE_FACTORS:
            groups = (ev_model.get("factors", {}).get(key, {}) or {}).get("groups", {})
            rows_ = [(lbl, g) for lbl, g in groups.items() if g["n"] >= 3]
            if len(rows_) < 2:
                continue
            shown += 1
            st.markdown(f"**{title}**")
            st.markdown(hbars_html([(lbl, float(g["raw_rate"] - p0), f"{g['raw_rate']:.0f}% won · {g['n']} trades" + ("" if g["active"] else " (too few to use)"))
                                    for lbl, g in rows_]), unsafe_allow_html=True)
        if not shown:
            st.caption("Not enough trades in each group yet to compare conditions.")
        callout("good" if ev_model.get("ready") else "info", "🔬",
                ("<b>In use:</b> the AI's confidence is now nudged up or down (at most ±"
                 f"{EVIDENCE_TOTAL_CAP} points) by what has really worked for you. Small groups are pulled toward your overall win rate so a lucky streak can't fool it."
                 if ev_model.get("ready") else
                 f"<b>Watching, not acting yet.</b> Once {EVIDENCE_MIN_DECIDED} trades have clearly won or lost, these patterns start nudging the AI's confidence."))
    else:
        st.info("This fills in as trades finish. It compares conditions like strength vs the FTSE 100, momentum, volume and how much of the market was rising.")

    sec_title("🔧", "The AI itself")
    state = _load_retrain_state()
    cal = st.session_state.get("cal_info")
    st.markdown(
        f"- **Version in use:** {st.session_state.get('ml_model_source', 'not loaded yet')}\n"
        f"- **Last upgrade:** {state.get('last_promoted', 'never — still the original')}\n"
        f"- **Upgrade rule:** only after {MIN_SETTLED_TO_RETRAIN}+ clear win/loss trades, and only if the new version tests better than the old one. Your original AI file is never overwritten."
        + (f"\n- **Confidence tuning:** based on {cal['decided']} trades that clearly won or lost." if cal else "")
        + f"\n- **Evidence learner:** {'in use' if ev_model.get('ready') else 'collecting evidence'} (see above)."
        + "\n- **Options:** this AI learns from share trades only. Options have their own separate learner — see the 🎲 Options tab.")

    with st.expander("How the AI picks a share"):
        st.markdown(
            "1. Downloads prices for **every FTSE 350 share plus liquid AIM shares**, skipping ones that are tiny, thinly traded, suspended or extremely jumpy.\n"
            "2. Scores each one on how likely it is to reach its goal before its safety exit (completed days only, no peeking at today's unfinished bar).\n"
            f"3. The best {MAX_DEEP_CHECKS} become finalists and get a **fresh price**, the news check, lessons from past trades, this week's oil/copper/£ moves, and the evidence learner.\n"
            "4. Blocks any share with bad news (share sales, profit warnings and similar).\n"
            f"5. Only keeps shares with **{MIN_QUALIFY_CONFIDENCE}%+ confidence**, at least **{MIN_NET_RETURN_PCT:.0f}% possible profit after costs**, and a **positive expected value after all costs** "
            "(chance of winning × what it would make, minus chance of losing × what it would lose).")

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