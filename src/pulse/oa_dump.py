"""Extracts from a local OpenAlex snapshot (26 June 2026, parquet) with DuckDB.

    OPENALEX_SNAPSHOT=/path/to/openalex/data/parquet python src/pulse/oa_dump.py <step> [--part i --nparts n]

Steps: match, firstyear, nihawards, authornames, nihacks, pubinfo, pubabs, then reduce. Large steps split the works
table by file into parts. Inputs and outputs live under $PULSE_DIR/data/interim (default: the repository root).
"""

from __future__ import annotations

import argparse
import glob
import os
import time
from pathlib import Path

import duckdb

SNAP = os.environ["OPENALEX_SNAPSHOT"]  # .../openalex/2026-06-26/data/parquet
ROOT = Path(os.environ.get("PULSE_DIR", Path(__file__).resolve().parents[2]))
INTERIM = ROOT / "data" / "interim"
PARTS = INTERIM / "oa_parts"


def work_files(part: int, nparts: int) -> list[str]:
    files = sorted(glob.glob(f"{SNAP}/works/updated_date=*/*.parquet"))
    return files[part::nparts]


def con() -> duckdb.DuckDBPyConnection:
    c = duckdb.connect()
    c.execute(f"SET threads = {int(os.environ.get('SLURM_CPUS_PER_TASK', '8'))}")
    c.execute("SET memory_limit = '48GB'")
    c.execute(f"SET temp_directory = '{ROOT / 'data' / 'duckdb_tmp'}'")
    return c


def match(part: int, nparts: int) -> None:
    PARTS.mkdir(parents=True, exist_ok=True)
    out = PARTS / f"match_{part:03d}.parquet"
    c = con()
    c.execute(
        "CREATE TABLE dois AS SELECT lower(doi) AS doi, server FROM read_csv(?, header=true)",
        [str(INTERIM / "rxiv_dois.csv")],
    )
    c.execute(
        f"""
        COPY (
          SELECT d.doi, d.server, w.id AS work_id, w.publication_year,
                 len(w.authorships) AS n_authors,
                 w.authorships[1].author.id AS first_id,
                 w.authorships[len(w.authorships)].author.id AS last_id,
                 array_to_string(w.authorships[1].countries, ',') AS first_countries,
                 array_to_string(w.authorships[len(w.authorships)].countries, ',') AS last_countries,
                 w.authorships[1].institutions[1].type AS first_inst_type,
                 w.authorships[len(w.authorships)].institutions[1].type AS last_inst_type,
                 w.authorships[len(w.authorships)].institutions[1].id AS last_inst_id,
                 w.authorships[len(w.authorships)].institutions[1].display_name AS last_inst_name,
                 w.authorships[len(w.authorships)].institutions[1].country_code AS last_inst_country,
                 len(w.funders) AS n_funders,
                 array_to_string(list_transform(w.funders, x -> x.display_name), ' | ') AS funder_names,
                 array_to_string(list_transform(w.awards, x -> coalesce(x.funder_display_name, '') || ': ' ||
                                                coalesce(x.funder_award_id, '')), ' | ') AS awards
          FROM read_parquet(?) w
          JOIN dois d ON lower(replace(w.doi, 'https://doi.org/', '')) = d.doi
          WHERE (w.doi LIKE 'https://doi.org/10.1101/%' OR w.doi LIKE 'https://doi.org/10.64898/%')
            AND len(w.authorships) > 0
        ) TO '{out}' (FORMAT parquet)
    """,
        [work_files(part, nparts)],
    )
    n = c.execute(f"SELECT count(*) FROM read_parquet('{out}')").fetchone()[0]
    print(f"match part {part}: {n} rows")


def firstyear(part: int, nparts: int) -> None:
    PARTS.mkdir(parents=True, exist_ok=True)
    out = PARTS / f"firstyear_{part:03d}.parquet"
    c = con()
    c.execute(f"""
        CREATE TABLE targets AS
        SELECT DISTINCT id FROM (
          SELECT first_id AS id FROM read_parquet('{INTERIM}/oa_match.parquet')
          UNION SELECT last_id FROM read_parquet('{INTERIM}/oa_match.parquet')) WHERE id IS NOT NULL
    """)
    c.execute(
        f"""
        COPY (
          SELECT s.aid AS id, min(s.publication_year) AS first_year,
                 count(*) AS n_works, list(s.publication_year) AS years
          FROM (SELECT publication_year, unnest(authorships).author.id AS aid
                FROM read_parquet(?) WHERE publication_year BETWEEN 1950 AND 2026) s
          JOIN targets ON targets.id = s.aid
          GROUP BY 1
        ) TO '{out}' (FORMAT parquet)
    """,
        [work_files(part, nparts)],
    )
    print(f"firstyear part {part}: done")


NIH_FUNDER = (
    r"National Institutes of Health|National Cancer Institute|National Heart, Lung|National Human Genome|"
    r"National Eye Institute|National Library of Medicine|National Center for Advancing Translational|"
    r"National Institute (of|on) |Fogarty"
)
AWARD_RE = r"^([0-9])([a-z][0-9a-z]{2})([a-z]{2})([0-9]{6})-([0-9]{2})([a-z][0-9])?"


def nih_awards() -> None:
    """NIH award records with the project number split into type, activity, IC, serial, year, suffix."""
    c = con()
    files = glob.glob(f"{SNAP}/awards/updated_date=*/*.parquet")
    c.execute(
        f"""
        COPY (
          SELECT lower(funder_award_id) AS award_id, funder.display_name AS funder,
                 start_date, end_date, funding_type, provenance,
                 lead_investigator.given_name AS pi_given, lead_investigator.family_name AS pi_family,
                 lead_investigator.affiliation.name AS pi_aff,
                 regexp_extract(lower(funder_award_id), '{AWARD_RE}', 1) AS app_type,
                 regexp_extract(lower(funder_award_id), '{AWARD_RE}', 2) AS activity,
                 regexp_extract(lower(funder_award_id), '{AWARD_RE}', 3) AS ic,
                 regexp_extract(lower(funder_award_id), '{AWARD_RE}', 4) AS serial,
                 regexp_extract(lower(funder_award_id), '{AWARD_RE}', 5) AS support_year,
                 regexp_extract(lower(funder_award_id), '{AWARD_RE}', 6) AS suffix
          FROM read_parquet(?) WHERE regexp_matches(funder.display_name, '{NIH_FUNDER}')
            AND end_date >= DATE '2014-01-01'
        ) TO '{INTERIM}/nih_awards.parquet' (FORMAT parquet)
    """,
        [files],
    )
    print(
        c.execute(
            f"SELECT count(*), count(NULLIF(app_type,'')), count(NULLIF(suffix,'')), "
            f"count(pi_family) FROM read_parquet('{INTERIM}/nih_awards.parquet')"
        ).fetchall()
    )
    print(
        c.execute(
            f"SELECT app_type, activity, count(*) n FROM read_parquet('{INTERIM}/nih_awards.parquet') "
            f"GROUP BY 1,2 ORDER BY n DESC LIMIT 12"
        ).fetchall()
    )


def nih_acks(part: int, nparts: int) -> None:
    """Works (2012+) that acknowledge an NIH award: award ids and authors (id, position, name).
    Used to anchor RePORTER PIs to OpenAlex author ids through their own awards."""
    PARTS.mkdir(parents=True, exist_ok=True)
    out = PARTS / f"nihack_{part:03d}.parquet"
    c = con()
    c.execute(
        f"""
        COPY (
          SELECT id AS work_id, publication_year,
                 list_transform(list_filter(awards, x -> regexp_matches(coalesce(x.funder_display_name, ''),
                                '{NIH_FUNDER}')), x -> x.funder_award_id) AS award_ids,
                 list_transform(authorships, x -> {{'id': x.author.id, 'pos': x.author_position,
                                                   'name': x.author.display_name}}) AS authors
          FROM read_parquet(?)
          WHERE publication_year BETWEEN 2012 AND 2026 AND len(awards) > 0
            AND len(list_filter(awards, x -> regexp_matches(coalesce(x.funder_display_name, ''), '{NIH_FUNDER}'))) > 0
        ) TO '{out}' (FORMAT parquet)
    """,
        [work_files(part, nparts)],
    )
    n = c.execute(f"SELECT count(*) FROM read_parquet('{out}')").fetchone()[0]
    print(f"nihack part {part}: {n} works")


def pubinfo(part: int, nparts: int) -> None:
    """Publication date, citations and yearly citation counts for preprint and published DOIs."""
    PARTS.mkdir(parents=True, exist_ok=True)
    out = PARTS / f"pubinfo_{part:03d}.parquet"
    c = con()
    c.execute(
        "CREATE TABLE d AS SELECT lower(doi) AS doi, kind FROM read_parquet(?)", [str(INTERIM / "pub_dois.parquet")]
    )
    c.execute(
        f"""
        COPY (
          SELECT d.doi, d.kind, w.id AS work_id, w.publication_date, w.cited_by_count,
                 list_transform(w.counts_by_year, x -> {{'year': x.year, 'n': x.cited_by_count}}) AS cites_by_year
          FROM read_parquet(?) w
          JOIN d ON lower(replace(w.doi, 'https://doi.org/', '')) = d.doi
        ) TO '{out}' (FORMAT parquet)
    """,
        [work_files(part, nparts)],
    )
    print(f"pubinfo part {part}: done")


def pub_abstracts(part: int, nparts: int) -> None:
    """Abstracts (inverted-index JSON) of the journal versions of rxiv preprints."""
    PARTS.mkdir(parents=True, exist_ok=True)
    out = PARTS / f"pubabs_{part:03d}.parquet"
    c = con()
    c.execute(
        "CREATE TABLE d AS SELECT lower(doi) AS doi FROM read_parquet(?) WHERE kind = 'published'",
        [str(INTERIM / "pub_dois.parquet")],
    )
    c.execute(
        f"""
        COPY (
          SELECT d.doi, w.title, w.abstract_inverted_index AS aii
          FROM read_parquet(?) w
          JOIN d ON lower(replace(w.doi, 'https://doi.org/', '')) = d.doi
          WHERE w.abstract_inverted_index IS NOT NULL
        ) TO '{out}' (FORMAT parquet)
    """,
        [work_files(part, nparts)],
    )
    print(f"pubabs part {part}: done")


def author_names() -> None:
    """Display names for last authors in oa_match (for PI linkage)."""
    c = con()
    files = glob.glob(f"{SNAP}/authors/updated_date=*/*.parquet")
    c.execute(
        f"""
        COPY (
          SELECT a.id, a.display_name
          FROM read_parquet(?) a
          JOIN (SELECT DISTINCT last_id FROM read_parquet('{INTERIM}/oa_match.parquet')) m ON m.last_id = a.id
        ) TO '{INTERIM}/oa_last_author_names.parquet' (FORMAT parquet)
    """,
        [files],
    )
    print(c.execute(f"SELECT count(*) FROM read_parquet('{INTERIM}/oa_last_author_names.parquet')").fetchall())


def reduce(kind: str) -> None:
    c = con()
    if kind == "pubabs":
        c.execute(
            f"COPY (SELECT * FROM read_parquet('{PARTS}/pubabs_*.parquet')) "
            f"TO '{INTERIM}/oa_pub_abstracts.parquet' (FORMAT parquet)"
        )
        print(c.execute(f"SELECT count(*) FROM read_parquet('{INTERIM}/oa_pub_abstracts.parquet')").fetchall())
        return
    if kind == "pubinfo":
        c.execute(
            f"COPY (SELECT * FROM read_parquet('{PARTS}/pubinfo_*.parquet')) "
            f"TO '{INTERIM}/oa_pubinfo.parquet' (FORMAT parquet)"
        )
        print(
            c.execute(f"SELECT kind, count(*) FROM read_parquet('{INTERIM}/oa_pubinfo.parquet') GROUP BY 1").fetchall()
        )
        return
    if kind == "nihack":
        c.execute(
            f"COPY (SELECT * FROM read_parquet('{PARTS}/nihack_*.parquet')) "
            f"TO '{INTERIM}/oa_nih_acks.parquet' (FORMAT parquet)"
        )
        print(c.execute(f"SELECT count(*) FROM read_parquet('{INTERIM}/oa_nih_acks.parquet')").fetchall())
        return
    if kind == "match":
        c.execute(
            f"COPY (SELECT * FROM read_parquet('{PARTS}/match_*.parquet')) "
            f"TO '{INTERIM}/oa_match.parquet' (FORMAT parquet)"
        )
        print(
            c.execute(f"SELECT server, count(*) FROM read_parquet('{INTERIM}/oa_match.parquet') GROUP BY 1").fetchall()
        )
    else:
        # per author: earliest year, total works, and the earliest year with >= 2 works to date
        # (guards against a single misattributed early paper)
        c.execute(f"""
            COPY (
              WITH y AS (SELECT id, unnest(years) AS yr FROM read_parquet('{PARTS}/firstyear_*.parquet')),
              byyr AS (SELECT id, yr, count(*) AS n FROM y GROUP BY 1, 2),
              cum AS (SELECT id, yr, sum(n) OVER (PARTITION BY id ORDER BY yr) AS cum FROM byyr)
              SELECT id, min(yr) AS first_year, min(yr) FILTER (WHERE cum >= 2) AS first_year_2,
                     max(cum) AS n_works
              FROM cum GROUP BY id
            ) TO '{INTERIM}/oa_author_firstyear.parquet' (FORMAT parquet)
        """)
        print(
            c.execute(
                f"SELECT count(*), median(first_year), median(n_works) "
                f"FROM read_parquet('{INTERIM}/oa_author_firstyear.parquet')"
            ).fetchall()
        )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "cmd",
        choices=[
            "match",
            "firstyear",
            "nihawards",
            "authornames",
            "nihacks",
            "pubinfo",
            "pubabs",
            "reduce",
        ],
    )
    ap.add_argument("--part", type=int, default=0)
    ap.add_argument("--nparts", type=int, default=1)
    ap.add_argument("--kind", choices=["match", "firstyear", "nihack", "pubinfo", "pubabs"], default="match")
    a = ap.parse_args()
    t0 = time.time()
    {
        "match": lambda: match(a.part, a.nparts),
        "firstyear": lambda: firstyear(a.part, a.nparts),
        "nihawards": nih_awards,
        "authornames": author_names,
        "nihacks": lambda: nih_acks(a.part, a.nparts),
        "pubinfo": lambda: pubinfo(a.part, a.nparts),
        "pubabs": lambda: pub_abstracts(a.part, a.nparts),
        "reduce": lambda: reduce(a.kind),
    }[a.cmd]()
    print(f"{a.cmd} took {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
