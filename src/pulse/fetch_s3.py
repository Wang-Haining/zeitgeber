"""Pull JATS XML for medRxiv/bioRxiv from the official requester-pays S3 TDM bucket.

Each object is a .meca (zip) holding PDF, figures and the JATS XML. Only the
zip central directory and the XML member are read with ranged GETs, so the
egress is ~1-2% of the full archive.

    python -m pulse.fetch_s3 list medrxiv            -> data/interim/s3_keys_medrxiv.txt
    python -m pulse.fetch_s3 get medrxiv --shard i --nshards n

Output matches fetch_jats: data/jats/{server}/{doi_suffix}.xml.gz
Requires AWS credentials (the requester pays the transfer).
"""

from __future__ import annotations

import argparse
import gzip
import io
import os
import re
import time
import zipfile
from pathlib import Path

import boto3
from botocore.config import Config

from pulse.fetch_jats import JATS
from pulse.parse_jats import INTERIM

BUCKET = {"medrxiv": "medrxiv-src-monthly", "biorxiv": "biorxiv-src-monthly"}
DOI_RE = re.compile(rb'<article-id pub-id-type="doi"[^>]*>\s*(10\.(?:1101|64898)/[^<\s]+)\s*<')


def client():
    return boto3.client("s3", region_name="us-east-1", config=Config(retries={"max_attempts": 10, "mode": "adaptive"}))


class RangedObject(io.RawIOBase):
    """Seekable read-only view of an S3 object backed by ranged GETs."""

    def __init__(self, s3, bucket: str, key: str, size: int):
        self.s3, self.bucket, self.key, self.size, self.pos = s3, bucket, key, size, 0
        self.bytes_read = 0
        self.gets = 0

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=io.SEEK_SET):
        self.pos = {io.SEEK_SET: 0, io.SEEK_CUR: self.pos, io.SEEK_END: self.size}[whence] + offset
        return self.pos

    def read(self, n=-1):
        if self.pos >= self.size:
            return b""
        self.gets += 1
        end = self.size - 1 if n is None or n < 0 else min(self.size - 1, self.pos + n - 1)
        body = self.s3.get_object(
            Bucket=self.bucket, Key=self.key, Range=f"bytes={self.pos}-{end}", RequestPayer="requester"
        )["Body"].read()
        self.pos += len(body)
        self.bytes_read += len(body)
        return body

    def readinto(self, b):
        data = self.read(len(b))
        b[: len(data)] = data
        return len(data)


def list_keys(server: str) -> Path:
    s3 = client()
    out = INTERIM / f"s3_keys_{server}.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with out.open("w") as fh:
        for page in s3.get_paginator("list_objects_v2").paginate(Bucket=BUCKET[server], RequestPayer="requester"):
            for o in page.get("Contents", []):
                if o["Key"].endswith(".meca"):
                    fh.write(f"{o['Key']}\t{o['Size']}\n")
                    n += 1
    print(f"{n} .meca objects -> {out}")
    return out


GETS = [0]


def get_xml(s3, server: str, key: str, size: int) -> tuple[bytes | None, int]:
    f = RangedObject(s3, BUCKET[server], key, size)
    with zipfile.ZipFile(io.BufferedReader(f, buffer_size=int(os.environ.get("S3_BUF_KB", "64")) * 1024)) as z:
        names = [n for n in z.namelist() if n.startswith("content/") and n.endswith(".xml") and "manifest" not in n]
        if not names:
            return None, f.bytes_read
        xml = z.read(min(names, key=len))
    GETS[0] += f.gets
    return xml, f.bytes_read


def fetch(server: str, shard: int, nshards: int, max_seconds: float) -> None:
    s3 = client()
    keys = [line.split("\t") for line in (INTERIM / f"s3_keys_{server}.txt").read_text().splitlines()]
    keys = [(k, int(sz)) for i, (k, sz) in enumerate(keys) if i % nshards == shard]
    done_file = INTERIM / f"s3_done_{server}_{shard:03d}.txt"
    done = set(done_file.read_text().splitlines()) if done_file.exists() else set()
    out_dir = JATS / server
    out_dir.mkdir(parents=True, exist_ok=True)
    t0, egress, ok, miss = time.time(), 0, 0, 0
    with done_file.open("a") as log:
        for key, size in keys:
            if key in done:
                continue
            if time.time() - t0 > max_seconds:
                print("time budget reached; rerun this shard to continue")
                break
            try:
                xml, nbytes = get_xml(s3, server, key, size)
            except (zipfile.BadZipFile, KeyError) as e:
                print(f"bad {key}: {e}")
                log.write(key + "\n")
                continue
            egress += nbytes
            m = DOI_RE.search(xml or b"")
            if m:
                suffix = m.group(1).decode().split("/")[-1]
                dest = out_dir / f"{suffix}.xml.gz"
                if not dest.exists():  # several versions share a suffix; keep the first (and any v1 from fetch_jats)
                    dest.write_bytes(gzip.compress(xml))
                ok += 1
            else:
                miss += 1
            log.write(key + "\n")
            if (ok + miss) % 1000 == 0:
                log.flush()
                print(f"{ok + miss} done, ok={ok} no-doi={miss}, egress {egress / 1e9:.2f} GB", flush=True)
    print(
        f"shard {shard}: ok={ok} no-doi={miss} egress {egress / 1e9:.3f} GB, GETs {GETS[0]} in {time.time() - t0:.0f}s"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["list", "get"])
    ap.add_argument("server", choices=list(BUCKET))
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--max-seconds", type=float, default=6600)
    a = ap.parse_args()
    if a.cmd == "list":
        list_keys(a.server)
    else:
        fetch(a.server, a.shard, a.nshards, a.max_seconds)


if __name__ == "__main__":
    main()
