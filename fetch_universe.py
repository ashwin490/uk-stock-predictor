import os
import json

UNIVERSE_FILE = "./data/universe.json"
os.makedirs("./data", exist_ok=True)

# Comprehensive active UK AIM & Micro-Cap Universe
INSTITUTIONAL_AIM_UNIVERSE = [
    # Energy, CleanTech & Hydrogen
    "ITM.L", "CWR.L", "PHE.L", "EQT.L", "AFC.L", "SERE.L", "ATOME.L", "AMER.L",
    # Mining, Critical Metals & Lithium
    "GGP.L", "PREM.L", "KOD.L", "BZT.L", "HE1.L", "ALL.L", "EMH.L", "EUA.L", 
    "SAV.L", "EEE.L", "POW.L", "KIBO.L", "MKA.L", "THR.L", "JAG.L", "PAL.L",
    # Biotech, Life Sciences & Diagnostics
    "AVCT.L", "POLX.L", "SYN.L", "SNG.L", "ANGLE.L", "GENI.L", "TRX.L", "ONC.L", 
    "DVRG.L", "RENX.L", "VAL.L", "BMSO.L",
    # Technology, AI & Software
    "BANGO.L", "IQE.L", "PEBB.L", "TPX.L", "SFOR.L", "BEG.L", "BIRD.L", "FAN.L", 
    "GMR.L", "AUG.L", "ERIS.L", "SND.L", "CRW.L",
    # Oil, Gas & Natural Resources
    "HUR.L", "CHAR.L", "UJO.L", "RBD.L", "SAVE.L", "PRD.L", "ZEN.L", "ADV.L", 
    "COPL.L", "ECO.L", "DELT.L",
    # Consumer, Retail & Financial Services
    "BOO.L", "ASOS.L", "CARD.L", "FDP.L", "WAND.L", "FPO.L", "GLE.L", "VLX.L"
]

def update_universe():
    clean_tickers = sorted(list(set([t.strip().upper() for t in INSTITUTIONAL_AIM_UNIVERSE])))
    with open(UNIVERSE_FILE, "w") as f:
        json.dump(clean_tickers, f, indent=2)
    print(f"[+] Successfully registered {len(clean_tickers)} UK AIM tickers into {UNIVERSE_FILE}")

if __name__ == "__main__":
    update_universe()