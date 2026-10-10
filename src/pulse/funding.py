"""Classify a funding statement: NIH-funded? which activity codes?

UK "NIHR" (National Institute for Health Research) is not NIH; the word
boundary in `\\bNIH\\b` keeps it out, and the full name is excluded by
matching "Institutes of Health" rather than "Institute for Health".
"""

from __future__ import annotations

import re

# NIH institute and center names / acronyms
_IC_NAMES = [
    r"National Institutes? of Health",
    r"\bNIH\b",
    r"National Cancer Institute",
    r"\bNCI\b",
    r"National Heart,? Lung,? and Blood Institute",
    r"National Institute (?:of|on) [A-Z][\w ,&-]{3,80}",
    r"National Center for Advancing Translational Sciences",
    r"National Library of Medicine",
    r"National Human Genome Research Institute",
    r"National Eye Institute",
    r"Fogarty International Center",
    r"National Center for Complementary and Integrative Health",
    r"\bN(?:IAID|HLBI|IMH|IDA|IA|IDDK|ICHD|INDS|IEHS|IGMS|HGRI|IAAA|IAMS|IBIB|IDCD|IDCR|EI|IMHD|INR|CATS|CCIH|LM)\b",
]
NIH_NAME = re.compile("|".join(_IC_NAMES))

# two-letter institute codes used in NIH grant numbers
_IC = "AI|HL|MH|CA|DA|AG|DK|HD|NS|ES|GM|HG|AA|AR|EB|DC|DE|EY|MD|NR|TR|AT|LM|TW|OD|RR|NR"
_ACT = r"(?:R\d{2}|K\d{2}|F\d{2}|U\d{2}|P\d{2}|T\d{2}|DP\d|UL1|UM1|UG3|UH3|S10|R00|ZIA)"
GRANT = re.compile(rf"\b\d?\s?({_ACT})\s?-?\s?(?:{_IC})\s?-?\s?\d{{5,6}}")
ACT_ONLY = re.compile(rf"\b({_ACT})\b")

NONE_STMT = re.compile(
    r"\b(?:no (?:specific |external |financial )?funding|did not receive (?:any )?(?:specific )?funding|"
    r"received no (?:specific |external )?funding|not funded|unfunded)\b",
    re.I,
)


def classify(text: str | None) -> dict:
    text = text if isinstance(text, str) else ""
    grants = GRANT.findall(text)
    nih_name = bool(NIH_NAME.search(text))
    acts = sorted(set(grants) | (set(ACT_ONLY.findall(text)) if nih_name else set()))
    return {
        "nih": nih_name or bool(grants),
        "nih_grant_number": bool(grants),
        "nih_activity_codes": acts,
        "has_K": any(a.startswith("K") for a in acts),
        "has_F": any(a.startswith("F") for a in acts),
        "has_R": any(a.startswith("R") for a in acts),
        "no_funding": bool(NONE_STMT.search(text)) and not (nih_name or grants),
    }
