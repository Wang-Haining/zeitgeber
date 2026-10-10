"""Regex extraction of ratio estimates (OR/HR/RR/IRR/PR) with 95% CI,
P values, sample size, and hype-adjective rate from abstract or results text.

Primary per-paper effect: the first ratio reported in the abstract
(the headline), as |log(ratio)|; z = log(est) / SE with
SE = (log(ub) - log(lb)) / 3.92.
"""

from __future__ import annotations

import math
import re
from functools import lru_cache
from pathlib import Path

_NUM = r"(\d{1,4}(?:\.\d+)?|\.\d+)"
_SEP = r"\s*(?:-|–|—|‒|−|to|,|;)\s*"
_LABEL = (
    r"(?:adjusted |crude |unadjusted |pooled |multivariable |subdistribution )?"
    r"(?:odds ratio|hazard ratio|risk ratio|relative risk|rate ratio|incidence rate ratio|"
    r"prevalence ratio|a?OR|a?HR|a?RR|a?IRR|a?PR|sHR|SHR|aOR|aHR)"
)
RATIO = re.compile(
    rf"(?P<label>\b{_LABEL})\b"
    rf"[^0-9]{{0,40}}?"  # '[OR] =', 's of', ': '
    rf"{_NUM.replace('(', '(?P<est>', 1)}"
    rf"[^0-9]{{0,30}}?"  # ' (95% CI ' / ', 95% confidence interval: '
    rf"(?:95\s?%\s?(?:CI|confidence interval|CrI|UI)[^0-9]{{0,12}}?)?"
    rf"{_NUM.replace('(', '(?P<lb>', 1)}{_SEP}{_NUM.replace('(', '(?P<ub>', 1)}",
    re.I,
)

PVAL = re.compile(
    r"\b[pP]\s?(?:-?\s?values?)?\s?(?:(?P<cmp>[<>=≤≥]|<=|>=)|of|was|:)\s?(?P<p>0?\.\d+|\d(?:\.\d+)?)(?:\s?[x×]\s?10\s?[-−–](?P<exp>\d+))?"
)

_NOUNS = (
    r"(?:patients|participants|individuals|subjects|adults|children|women|men|people|persons|"
    r"cases|respondents|infants|pregnancies|births|veterans|residents|users|students|admissions|"
    r"encounters|hospitali[sz]ations|visits|samples|healthcare workers|households)"
)
SAMPLE = re.compile(
    rf"\b[nN]\s?=\s?(\d{{1,3}}(?:,\d{{3}})+|\d+)|(?<![\d.,])(\d{{1,3}}(?:,\d{{3}})+|\d{{2,9}})\s+(?:\w+\s){{0,3}}?{_NOUNS}\b"
)


@lru_cache(1)
def hype_words() -> frozenset[str]:
    p = Path(__file__).parent / "resources" / "millar_hype_139.txt"
    return frozenset(w.strip() for w in p.read_text().splitlines() if w.strip())


def ratios(text: str) -> list[dict]:
    out = []
    for m in RATIO.finditer(text or ""):
        try:
            est, lb, ub = float(m["est"]), float(m["lb"]), float(m["ub"])
        except (TypeError, ValueError):
            continue
        if not (0 < lb < ub and lb <= est <= ub) or est == 95 or ub > 1000:
            continue
        se = (math.log(ub) - math.log(lb)) / 3.92
        if se <= 0:
            continue
        out.append(
            {
                "label": m["label"],
                "est": est,
                "lb": lb,
                "ub": ub,
                "abs_log": abs(math.log(est)),
                "z": math.log(est) / se,
                "se": se,
                "pos": m.start(),
            }
        )
    return out


def pvalues(text: str) -> list[dict]:
    out = []
    for m in PVAL.finditer(text or ""):
        p = float(m["p"])
        if m["exp"]:
            p *= 10 ** -int(m["exp"])
        if 0 <= p <= 1:
            out.append({"cmp": m["cmp"] or "=", "p": p})
    return out


def sample_size(text: str) -> int | None:
    vals = []
    for m in SAMPLE.finditer(text or ""):
        s = m.group(1) or m.group(2)
        v = int(s.replace(",", ""))
        if 10 <= v <= 500_000_000 and not (1900 <= v <= 2030 and "," not in s):  # skip years
            vals.append(v)
    return max(vals) if vals else None


def hype_rate(text: str, drop: tuple[str, ...] = ()) -> tuple[int, int]:
    """(hype adjective count, word count)."""
    words = re.findall(r"[a-z][a-z\-]*", (text or "").lower())
    lex = hype_words() - set(drop)
    return sum(w in lex for w in words), len(words)


def summarize(text: str) -> dict:
    r = ratios(text)
    p = pvalues(text)
    h, n_words = hype_rate(text)
    h_ns, _ = hype_rate(text, drop=("significant",))
    exact = [x["p"] for x in p if x["cmp"] == "="]
    head = r[0] if r else {}
    return {
        "n_ratios": len(r),
        "head_label": head.get("label"),
        "head_est": head.get("est"),
        "head_abs_log": head.get("abs_log"),
        "head_z": head.get("z"),
        "head_se": head.get("se"),
        "max_abs_log": max((x["abs_log"] for x in r), default=None),
        "median_abs_z": sorted(abs(x["z"]) for x in r)[len(r) // 2] if r else None,
        "n_p": len(p),
        "n_p_exact": len(exact),
        "n_p_caliper_below": sum(0.04 <= v < 0.05 for v in exact),
        "n_p_caliper_above": sum(0.05 < v <= 0.06 for v in exact),
        "share_p_sig": (sum((x["p"] < 0.05) or (x["cmp"] == "<" and x["p"] <= 0.05) for x in p) / len(p))
        if p
        else None,
        "n_sample": sample_size(text),
        "hype": h,
        "hype_nosig": h_ns,
        "n_words": n_words,
    }
