"""Verified London-listed, sterling-traded commodity-linked securities.

Only GBP / GBp/GBX share classes; no futures margin or FX conversion in the
existing equity simulator. These are ETCs, NOT physical bullion or oil/gas
futures. Their risk and model distribution differ from equity securities.
Issuer listing data checked 23 November 2026; symbols may later change.
"""
COMMODITY_ETPS = {
    "SGLN.L": {"name": "iShares Physical Gold ETC", "sector": "Precious Metals",
               "underlying": "Gold", "issuer": "iShares", "risk": "Physical gold ETC"},
    "SSLN.L": {"name": "iShares Physical Silver ETC", "sector": "Precious Metals",
               "underlying": "Silver", "issuer": "iShares", "risk": "Physical silver ETC"},
    "BRNG.L": {"name": "WisdomTree Brent Crude Oil ETC", "sector": "Oil & Gas ETCs",
               "underlying": "Brent crude", "issuer": "WisdomTree", "risk": "Futures-linked; roll yield"},
    "CRUP.L": {"name": "WisdomTree WTI Crude Oil ETC", "sector": "Oil & Gas ETCs",
               "underlying": "WTI crude", "issuer": "WisdomTree", "risk": "Futures-linked; roll yield"},
    "NGSP.L": {"name": "WisdomTree Natural Gas ETC", "sector": "Oil & Gas ETCs",
               "underlying": "Natural gas", "issuer": "WisdomTree", "risk": "Futures-linked; high volatility"},
}

# Track as market context only. These are USD-denominated futures references;
# they must never be recorded as UK share/ETC purchases in pence.
FUTURES_REFERENCES = {
    "GC=F": "Gold futures", "SI=F": "Silver futures",
    "BZ=F": "Brent oil futures", "CL=F": "WTI oil futures",
    "NG=F": "Natural gas futures",
}

def is_commodity_etc(ticker: str) -> bool:
    return str(ticker).strip().upper() in COMMODITY_ETPS
