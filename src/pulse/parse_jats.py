"""Parse medRxiv/bioRxiv JATS into one row per preprint.

uv run python -m pulse.parse_jats medrxiv   ->  data/interim/{server}_jats.parquet
"""

from __future__ import annotations

import gzip
import re
import sys
from pathlib import Path

import pandas as pd
from lxml import etree

from pulse.fetch_jats import JATS

INTERIM = Path(__file__).resolve().parents[2] / "data" / "interim"
_parser = etree.XMLParser(recover=True, huge_tree=True)


def _text(el) -> str:
    return re.sub(r"\s+", " ", "".join(el.itertext())).strip() if el is not None else ""


def parse(xml: bytes) -> dict:
    root = etree.fromstring(xml, _parser)
    if root is None:
        return {}
    # strip namespaces from tags so plain xpath works
    for el in root.iter():
        if isinstance(el.tag, str) and "}" in el.tag:
            el.tag = el.tag.split("}", 1)[1]

    funding = " ".join(_text(n) for n in root.xpath("//notes[@notes-type='financial-disclosure']"))
    funding += " " + " ".join(_text(n) for n in root.xpath("//funding-group"))

    countries = [_text(c) for c in root.xpath("//aff/country")]
    corresp = " ".join(_text(c) for c in root.xpath("//corresp"))
    emails = [_text(e) for e in root.xpath("//corresp//email")]

    abstract = _text(root.find(".//abstract"))
    body = root.find(".//body")
    results = []
    if body is not None:
        for sec in body.xpath("./sec"):
            title = _text(sec.find("title"))
            if re.search(r"result|finding", title, re.I):
                results.append(_text(sec))
    body_text = _text(body)

    return {
        "funding": funding.strip(),
        "aff_countries": countries,
        "first_aff_country": countries[0] if countries else None,
        "corresp": corresp,
        "corresp_email": emails[0] if emails else None,
        "n_affs": len(root.xpath("//aff")),
        "n_authors": len(root.xpath("//contrib[@contrib-type='author']")),
        "abstract_jats": abstract,
        "results_text": " ".join(results),
        "body_words": len(body_text.split()),
        "has_trial_stmt": bool(root.xpath("//notes[@notes-type='clinical-trial-statement']")),
    }


def _one(f: Path) -> dict:
    try:
        rec = parse(gzip.decompress(f.read_bytes()))
    except Exception as e:  # malformed file: keep a stub so it is visible
        rec = {"parse_error": str(e)[:200]}
    # medRxiv/bioRxiv moved from the 10.1101 to the 10.64898 prefix on 2025-12-02;
    # files are named by DOI suffix, so build.py joins on the suffix
    rec["doi_suffix"] = f.name.removesuffix(".xml.gz")
    return rec


def main(server: str) -> None:
    import os
    from multiprocessing import Pool

    files = sorted((JATS / server).glob("*.xml.gz"))
    with Pool(int(os.environ.get("SLURM_CPUS_PER_TASK", "1"))) as pool:
        rows = []
        for i, rec in enumerate(pool.imap(_one, files, chunksize=200)):
            rows.append(rec)
            if (i + 1) % 20000 == 0:
                print(f"{i + 1}/{len(files)}", flush=True)
    INTERIM.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(INTERIM / f"{server}_jats.parquet", index=False)
    print(f"parsed {len(rows)}")


if __name__ == "__main__":
    main(sys.argv[1])
