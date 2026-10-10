"""Fetch new (type 1) NIH awards from the RePORTER v2 API.

R-type research grants and K career awards, fiscal years 2008-2026 for R01/R37 (to date each
PI's first R01) and 2014-2026 for the rest. Partitioned by fiscal year x activity code so
each query stays under the 15,000-offset limit; <= 1 request per second.

    python -m pulse.reporter   ->  data/interim/reporter_type1.parquet
"""

from __future__ import annotations

import time

import httpx
import pandas as pd

from pulse.parse_jats import INTERIM

API = "https://api.reporter.nih.gov/v2/projects/search"
R_CODES = ["R01", "R37", "R21", "R03", "R15", "R34", "R35", "DP2", "R61", "UG3"]
K_CODES = ["K01", "K08", "K23", "K25", "K99", "K22", "K07"]
FIELDS = [
    "ApplId",
    "ProjectNum",
    "CoreProjectNum",
    "ActivityCode",
    "AwardType",
    "FiscalYear",
    "AwardNoticeDate",
    "ProjectStartDate",
    "ProjectEndDate",
    "BudgetStart",
    "Organization",
    "OrganizationType",
    "PrincipalInvestigators",
    "FullStudySection",
    "ProjectTitle",
    "AbstractText",
    "OpportunityNumber",
    "SpendingCategoriesDesc",
]


def fetch(client: httpx.Client, fy: int, code: str, award_type: str = "1") -> list[dict]:
    rows, offset, total = [], 0, None
    while total is None or offset < min(total, 14_999):
        body = {
            "criteria": {"fiscal_years": [fy], "activity_codes": [code], "award_types": [award_type]},
            "include_fields": FIELDS,
            "offset": offset,
            "limit": 500,
        }
        for k in range(6):
            try:
                r = client.post(API, json=body, timeout=120)
                if r.status_code == 429:
                    time.sleep(60)
                    continue
                r.raise_for_status()
                break
            except httpx.HTTPError:
                time.sleep(5 * 2**k)
        js = r.json()
        total = js["meta"]["total"]
        res = js.get("results", [])
        if not res:
            break
        for p in res:
            pis = p.get("principal_investigators") or []
            contact = next((x for x in pis if x.get("is_contact_pi")), pis[0] if pis else {})
            org = p.get("organization") or {}
            rows.append(
                {
                    "appl_id": p.get("appl_id"),
                    "project_num": p.get("project_num"),
                    "core": p.get("core_project_num"),
                    "activity": p.get("activity_code"),
                    "award_type": p.get("award_type"),
                    "fy": p.get("fiscal_year"),
                    "notice": p.get("award_notice_date"),
                    "start": p.get("project_start_date"),
                    "end": p.get("project_end_date"),
                    "budget_start": p.get("budget_start"),
                    "org": org.get("org_name"),
                    "org_state": org.get("org_state"),
                    "org_type": (p.get("organization_type") or {}).get("name"),
                    "pi_id": contact.get("profile_id"),
                    "pi_first": contact.get("first_name"),
                    "pi_middle": contact.get("middle_name"),
                    "pi_last": contact.get("last_name"),
                    "n_pis": len(pis),
                    "srg": (p.get("full_study_section") or {}).get("srg_code"),
                    "all_pi_ids": [x.get("profile_id") for x in pis],
                    "all_pi_names": [f"{x.get('first_name') or ''}|{x.get('last_name') or ''}" for x in pis],
                    "foa": p.get("opportunity_number"),
                    "spending": p.get("spending_categories_desc"),
                    "title": p.get("project_title"),
                    "abstract": p.get("abstract_text"),
                }
            )
        offset += len(res)
        time.sleep(1.0)
    return rows


def main() -> None:
    jobs = [(fy, c, t) for t in ("1", "2") for c in ("R01", "R37") for fy in range(2008, 2027)]
    jobs += [
        (fy, c, t) for t in ("1", "2") for c in R_CODES + K_CODES if c not in ("R01", "R37") for fy in range(2014, 2027)
    ]
    rows = []
    with httpx.Client(headers={"User-Agent": "pulse-research/0.1"}) as client:
        for i, (fy, c, t) in enumerate(jobs):
            got = fetch(client, fy, c, t)
            rows += got
            print(f"{i + 1}/{len(jobs)} FY{fy} {c} type {t}: {len(got)}", flush=True)
    d = pd.DataFrame(rows).drop_duplicates("appl_id")
    d.to_parquet(INTERIM / "reporter_awards.parquet", index=False)
    print("total", len(d))


if __name__ == "__main__":
    main()
