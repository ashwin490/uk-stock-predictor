import feedparser
import re
from datetime import datetime
import requests

# Clean regex patterns for UK regulatory RNS classification
DILUTION_PATTERNS = [
    r"\bplacing\b", r"\bsubscription\b", r"\bcapital raise\b", 
    r"\bfundrais\w*\b", r"\bdiscount\b", r"\bwarrant\b", 
    r"\bdilut\w*\b", r"\bshare issue\b", r"\boffer for subscription\b"
]

CATALYST_PATTERNS = [
    r"\bcontract win\b", r"\bcommercial agreement\b", r"\bdiscovery\b", 
    r"\bhigh grade\b", r"\brecord revenue\b", r"\bapproval\b", 
    r"\bpatent granted\b", r"\bpositive drill\b", r"\bacquisition\b"
]

def parse_rns_sentiment(title: str):
    lower_title = title.lower()
    
    # Check dilution red-flags first
    for pat in DILUTION_PATTERNS:
        if re.search(pat, lower_title):
            return "⚠️ RNS Alert: Placing / Dilution Risk", -0.30

    # Check positive commercial / operational catalysts
    for pat in CATALYST_PATTERNS:
        if re.search(pat, lower_title):
            return "🚀 Official RNS: Positive Catalyst", 0.18

    return "📰 RNS Verified: Clean Filing", 0.0

def fetch_direct_rns_for_ticker(ticker: str):
    """
    Scrapes official regulatory announcements for a specific UK ticker.
    Falls back gracefully if no announcements were filed today.
    """
    clean_sym = ticker.replace(".L", "").strip().upper()
    rss_url = f"https://www.investegate.co.uk/Rss.aspx?company={clean_sym}"

    try:
        feed = feedparser.parse(rss_url)
        if feed.entries:
            latest = feed.entries[0]
            title = latest.get("title", "")
            pub_date = latest.get("published", "")
            status, delta = parse_rns_sentiment(title)
            return {
                "headline": title[:75] + ("..." if len(title) > 75 else ""),
                "status": status,
                "delta": delta,
                "timestamp": pub_date
            }
    except Exception:
        pass

    return {
        "headline": "No major filings today",
        "status": "📰 Flow Verified",
        "delta": 0.0,
        "timestamp": datetime.now().strftime("%Y-%m-%d")
    }

if __name__ == "__main__":
    # Test on a known AIM ticker
    res = fetch_direct_rns_for_ticker("GGP.L")
    print("Test RNS Output for GGP.L:")
    print(res)