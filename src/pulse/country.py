"""ISO-2 country of the corresponding author.

Evidence order: email country-code TLD (.edu/.gov -> US), ROR code for the
corresponding-institution string, JATS first-affiliation country, then name
patterns for China and Japan. Hong Kong, Macau and Taiwan stay separate (HK/MO/TW):
their researchers are not in the NSFC general programmes.
"""

from __future__ import annotations

import re

NAME2ISO = {
    "united states": "US",
    "usa": "US",
    "u.s.a.": "US",
    "united states of america": "US",
    "us": "US",
    "china": "CN",
    "p.r. china": "CN",
    "pr china": "CN",
    "p. r. china": "CN",
    "people's republic of china": "CN",
    "peoples republic of china": "CN",
    "hong kong": "HK",
    "hong kong sar": "HK",
    "macau": "MO",
    "macao": "MO",
    "taiwan": "TW",
    "japan": "JP",
    "korea": "KR",
    "south korea": "KR",
    "republic of korea": "KR",
    "germany": "DE",
    "france": "FR",
    "united kingdom": "GB",
    "uk": "GB",
    "england": "GB",
    "scotland": "GB",
    "wales": "GB",
    "canada": "CA",
    "australia": "AU",
    "india": "IN",
    "italy": "IT",
    "spain": "ES",
    "netherlands": "NL",
    "the netherlands": "NL",
    "switzerland": "CH",
    "sweden": "SE",
    "brazil": "BR",
    "israel": "IL",
    "singapore": "SG",
    "denmark": "DK",
    "belgium": "BE",
    "austria": "AT",
    "norway": "NO",
    "finland": "FI",
}
CN_TEXT = re.compile(
    r"\bChina\b|Chinese Academy|Beijing|Shanghai|Wuhan|Guangzhou|Shenzhen|Hangzhou|Nanjing|Chengdu|"
    r"Zhejiang|Sun Yat-sen|Fudan|Tsinghua|Peking|Sichuan|Shandong|Hunan|Hubei|Jiangsu|Xi.an|Tianjin|"
    r"Chongqing|Wenzhou|Soochow|Guangdong|Henan|Anhui|Fujian|Jilin|Harbin|Shenyang|Xiamen|Lanzhou|Kunming"
)
HK_TEXT = re.compile(r"Hong Kong|Macau|Macao|Taiwan|Taipei")
JP_TEXT = re.compile(r"\bJapan\b|Tokyo|Kyoto|Osaka|RIKEN|Nagoya|Tohoku|Kyushu|Hokkaido|Keio|Waseda|Tsukuba")


def _email(email) -> str | None:
    if not isinstance(email, str) or "@" not in email:
        return None
    tld = email.rsplit(".", 1)[-1].lower().strip(">) ")
    if tld in {"edu", "gov", "mil", "us"}:
        return "US"
    if len(tld) == 2:
        return {"uk": "GB"}.get(tld, tld.upper())
    return None


def iso2(email=None, institution=None, first_aff_country=None, ror_code=None) -> str | None:
    e = _email(email)
    if e:
        return e
    if isinstance(ror_code, str) and len(ror_code) == 2:
        return ror_code.upper()
    if isinstance(first_aff_country, str):
        k = first_aff_country.strip().lower().rstrip(".")
        if k in NAME2ISO:
            return NAME2ISO[k]
    if isinstance(institution, str):
        if HK_TEXT.search(institution):
            return "HK" if "Hong Kong" in institution else "TW" if "Tai" in institution else "MO"
        if CN_TEXT.search(institution):
            return "CN"
        if JP_TEXT.search(institution):
            return "JP"
    return None
