"""Resolve corresponding-institution strings to a country with the ROR
affiliation-matching API, for strings the text heuristics in geo.py leave open.

    uv run python -m pulse.ror medrxiv biorxiv [--shard i --nshards n]

Cache: data/interim/ror_cache*.json  {institution string: country code or ""};
each shard writes its own file so shards never clobber each other.
"""

from __future__ import annotations

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import httpx
import pandas as pd

from pulse import geo
from pulse.fetch_rxiv import RAW
from pulse.parse_jats import INTERIM

API = "https://api.ror.org/v2/organizations"
PAUSE = 0.7  # per worker; stays well under ROR's ~2000 requests / 5 min
WORKERS = 2


def load_cache() -> dict[str, str]:
    out: dict[str, str] = {}
    for f in sorted(INTERIM.glob("ror_cache*.json")):
        out.update(json.loads(f.read_text()))
    return out


def lookup(client: httpx.Client, s: str) -> str:
    for k in range(4):
        try:
            r = client.get(API, params={"affiliation": s[:300]}, timeout=30)
            if r.status_code == 429:
                time.sleep(30)
                continue
            r.raise_for_status()
            items = r.json().get("items", [])
            best = next((i for i in items if i.get("chosen")), None)
            if best is None and items and items[0].get("score", 0) >= 0.9:
                best = items[0]
            if best is None:
                return ""
            return best["organization"]["locations"][0]["geonames_details"]["country_code"]
        except (httpx.HTTPError, KeyError, IndexError, ValueError):
            time.sleep(2**k)
    return ""


def main(servers: list[str], shard: int = 0, nshards: int = 1) -> None:
    import zlib

    known = load_cache()
    out_file = INTERIM / (f"ror_cache_{shard:02d}.json" if nshards > 1 else "ror_cache.json")
    cache = json.loads(out_file.read_text()) if out_file.exists() else {}
    strings: set[str] = set()
    for s in servers:
        for f in (RAW / s).glob("*.parquet"):
            inst = pd.read_parquet(f, columns=["author_corresponding_institution"]).iloc[:, 0]
            strings.update(x for x in inst.dropna().unique() if geo._from_text(x) is None)
    todo = sorted(s for s in strings if s not in known and zlib.crc32(s.encode()) % nshards == shard)
    print(f"{len(strings)} unresolved strings, {len(todo)} not cached", flush=True)
    lock = threading.Lock()
    done = [0]

    def work(s: str) -> None:
        code = lookup(client, s)
        time.sleep(PAUSE)
        with lock:
            cache[s] = code
            done[0] += 1
            if done[0] % 1000 == 0:
                out_file.write_text(json.dumps(cache))
                print(f"{done[0]}/{len(todo)}", flush=True)

    with httpx.Client(headers={"User-Agent": "pulse-research/0.1"}) as client:
        with ThreadPoolExecutor(WORKERS) as ex:
            list(ex.map(work, todo))
    out_file.write_text(json.dumps(cache))


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("servers", nargs="+")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    a = ap.parse_args()
    main(a.servers, a.shard, a.nshards)
