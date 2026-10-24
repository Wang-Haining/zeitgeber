# zeitgeber

Code for *Funding deadlines act as a zeitgeber for science* (Haining Wang and Jing Su). It rebuilds every number, figure and
table in the paper and its Supplementary Materials, including Data S1, from public sources:

- bioRxiv and medRxiv: the public API (metadata, all versions) and the requester-pays text-and-data-mining bucket
  (JATS full text)
- NIH RePORTER v2 API
- OpenAlex snapshot of 26 June 2026
- Research Organization Registry (ROR) affiliation matching

## Setup

```bash
uv sync
```

Python 3.12, with versions pinned in `uv.lock`. Run commands from the repository root with `uv run`, or in the
activated environment. Inputs go to `data/`, estimates to `results/`, and tables, figures and Data S1 to
`manuscript/`. Steps that read the full corpus use up to 160 GB of memory, and the download and extraction steps can
be split into parts (`--shard`, `--part`).

## 1. Data

| Step | Command | Output (`data/interim/` unless noted) |
|---|---|---|
| Preprint metadata, all versions | `python -m pulse.fetch_rxiv biorxiv 2017-01 2026-10`; `python -m pulse.fetch_rxiv medrxiv 2019-06 2026-10` | `data/raw/` |
| JATS full text (AWS credentials required) | `python -m pulse.fetch_s3 list <server>`; `python -m pulse.fetch_s3 get <server> --shard i --nshards n` | `data/jats/` |
| Parsed JATS | `python -m pulse.parse_jats <server>` | `<server>_jats.parquet` |
| Institution countries | `python -m pulse.ror biorxiv medrxiv [--shard i --nshards n]` | `ror_cache*.json` |
| Analysis table, one row per first version | `python -m pulse.build <server>` | `<server>_v1.parquet` |
| DOI lists for OpenAlex | `python -m pulse.build dois` | `rxiv_dois.csv`, `pub_dois.parquet` |
| OpenAlex extracts | `OPENALEX_SNAPSHOT=<path> python src/pulse/oa_dump.py <match\|firstyear\|nihawards\|authornames\|nihacks\|pubinfo\|pubabs> [--part i --nparts n]`, then `python src/pulse/oa_dump.py reduce --kind <kind>` | `oa_*.parquet`, `nih_awards.parquet` |
| NIH awards | `python -m pulse.reporter` | `reporter_awards.parquet` |
| Investigator links | `python -m pulse.pi_design link` | `pi_links.parquet` |
| bioRxiv 2013-2016 | `python -m pulse.pre2017 fetch`; `python -m pulse.pre2017 ror <i> <n>` | `data/raw_pre2017/` |

## 2. Analyses

Each command writes CSV files to `results/`.

| Result | Command |
|---|---|
| R01-window share and window lengths | `python -m pulse.robustness`; `python -m pulse.daily_step` |
| Group rates, years, funders (Fig. 1A) | `python -m pulse.group_rates` |
| Drop at the due date (Fig. 1B) and at shifted dates (Fig. 1C) | `python -m pulse.daily_step`; `python -m pulse.shift_scan` |
| Calendar-month fixed effects (Fig. 1D) | `python -m pulse.season_checks` |
| Discontinuity and business-day due dates | `python -m pulse.rd_phase` |
| Lunar New Year and Golden Week | `python -m pulse.holiday_check` |
| Phase scans (Fig. 3A; fig. S1) | `python -m pulse.phase_scan` |
| Cycle-by-cycle estimates, HIV contrast | `python -m pulse.design_checks` |
| HIV, intramural and linked-investigator counts | `python -m pulse.counts` |
| Within-investigator designs and placebo | `python -m pulse.pi_design within <main\|failed_a0\|renewal\|itt> 200` |
| Two-week profile (Fig. 2B), pooled event study | `python -m pulse.pi_design phase_profile 200`; `python -m pulse.pi_design events` |
| First submissions and resubmissions (Fig. 2C), resubmission dates nationally, kernel amplitudes | `python -m pulse.resub_checks <investigator\|aggregate\|kernel>` |
| Links without preprints, R01 vs R21/R03, cycle-family fixed effects | `python -m pulse.pi_design <norxiv\|activity\|family> 100` |
| Further investigator-level checks | `python -m pulse.pi_design <main\|sim\|bymonth\|totals\|decomp\|bycycle\|adjacent\|duedist\|moderators>` |
| Share of the national rise applicants can supply (text S4) | `python -m pulse.pi_design composition` (after `pi_design events` and `pulse.group_rates`) |
| Kernel model (Fig. 3A) | `python -m pulse.conv_model` |
| Kernel moved off the due dates (Fig. 3B), field profiles (fig. S3A) | `python -m pulse.post_checks` |
| Field dependence (Fig. 3C), robustness | `python -m pulse.field_split`; `python -m pulse.robustness` |
| Similarity to the forthcoming award (Fig. 4A) | `python -m pulse.sim_check` |
| Content (Fig. 4B; fig. S3B) | `python -m pulse.pulse_content` |
| Citation percentile, publication, promotional adjectives (Fig. 4B) | `python -m pulse.citations` |
| Preprint-to-journal shrinkage | `python -m pulse.shrink` |
| Institutional dependence | `python -m pulse.inst_dose` |
| Other countries and funders | `python -m pulse.country_calendar`; `python -m pulse.funder_pulse [nonus]` |
| Before and after 2017 (fig. S2) | `python -m pulse.pre2017 <analyze\|byfield>` |
| Scale, crowding, award acknowledgment | `python -m pulse.consequences <scale\|congestion\|ackaward>` |

## 3. Tables, figures and Data S1

These read `results/` only.

```bash
python -m pulse.si_tables   # manuscript/si_tables.tex
python -m pulse.figs_main   # manuscript/figures/fig1-fig4, figS1-figS3 (PDF and PNG)
python -m pulse.data_s1     # manuscript/data_S1.xlsx
```

`si_tables` numbers tables by first citation when `manuscript/main.tex` and `manuscript/supplement_body.tex` are
present, and keeps generator order otherwise.

## License

MIT
