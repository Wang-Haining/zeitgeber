"""Download bioRxiv / medRxiv metadata (all versions) from api.biorxiv.org.

One parquet file per server-month under data/raw/{server}/YYYY-MM.parquet,
so an interrupted run resumes by skipping months already on disk.
The current month is always re-fetched because it is still filling up.

    uv run python -m pulse.fetch_rxiv medrxiv 2019-06 2026-10
"""

from __future__ import annotations

import sys
import time
from datetime import date
from pathlib import Path

import httpx
import pandas as pd

API = "https://api.biorxiv.org/details/{server}/{start}/{end}/{cursor}/json"
RAW = Path(__file__).resolve().parents[2] / "data" / "raw"
PAUSE = 1.0  # seconds between requests


def month_bounds(ym: str) -> tuple[str, str]:
    start = pd.Period(ym, freq="M")
    return str(start.start_time.date()), str(start.end_time.date())


def get(client: httpx.Client, url: str, tries: int = 6) -> dict:
    """JSON page. The API sometimes answers 404 transiently, so 404 is retried too."""
    for k in range(tries):
        try:
            r = client.get(url, timeout=60)
            if r.status_code == 429:
                time.sleep(int(r.headers.get("retry-after", "300")) + 5)
                continue
            r.raise_for_status()
            return r.json()
        except (httpx.HTTPError, ValueError):
            time.sleep(min(60, 2 ** (k + 1)))
    raise RuntimeError(f"failed after {tries} tries: {url}")


def fetch_month(client: httpx.Client, server: str, ym: str) -> pd.DataFrame:
    start, end = month_bounds(ym)
    rows, cursor, total = [], 0, None
    while total is None or cursor < total:
        js = get(client, API.format(server=server, start=start, end=end, cursor=cursor))
        msg = js["messages"][0]
        if total is None:
            total = int(msg.get("total", msg.get("count", 0)) or 0)
            if total == 0:
                break
        batch = js.get("collection", [])
        if not batch:
            break
        rows.extend(batch)
        cursor += len(batch)
        time.sleep(PAUSE)
    df = pd.DataFrame(rows)
    if total and len(df) < total:
        raise RuntimeError(f"{server} {ym}: got {len(df)} of {total}")
    return df


def main(server: str, first: str, last: str) -> None:
    out = RAW / server
    out.mkdir(parents=True, exist_ok=True)
    this_month = pd.Period(date.today(), freq="M")
    with httpx.Client(headers={"User-Agent": "pulse-research/0.1"}) as client:
        for p in pd.period_range(first, last, freq="M"):
            f = out / f"{p}.parquet"
            if f.exists() and p < this_month:
                continue
            t0 = time.time()
            try:
                df = fetch_month(client, server, str(p))
            except RuntimeError as e:  # incomplete month: leave it unsaved so a rerun retries it
                print(f"skip {p}: {e}", flush=True)
                continue
            df.to_parquet(f, index=False)
            print(f"{server} {p}: {len(df):>6} records ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:4])
