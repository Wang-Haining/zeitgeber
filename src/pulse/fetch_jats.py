"""Download JATS XML for version-1 medRxiv preprints (gzipped, one file each).

    python -m pulse.fetch_jats medrxiv --workers 1 --dois list.txt [--delay 7.5] [--max-seconds 6300]

Reads data/raw/{server}/*.parquet, keeps version 1 records, and saves
data/jats/{server}/{doi_suffix}.xml.gz. Existing files are skipped.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
JATS = ROOT / "data" / "jats"
PAUSE = 7.5  # per worker; medrxiv.org robots.txt sets Crawl-delay: 7
