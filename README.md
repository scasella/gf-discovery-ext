# gf-discovery-ext

A label-free, preregistered search for the variables `gemma-3-1b-pt` tracks as it reads, with Gemma Scope 2's sparse
autoencoders (SAEs) as the reference. Claude Opus 5.5 agents designed it, ran it and kept this record, in October 2026,
on one M4 Pro MacBook Pro (24 GB, Apple MPS). The write-up is at
<https://casella.dev/blog_gemma_blind_search.html>.

**What it found, in three lines** (details and every source in `SUMMARY.md`; read its
[correction of 2026-10-24](SUMMARY.md) at the top first):
- The v2 instrument recovers parenthesis and quotation state from units extracted without labels: |AUROC| 0.779 (L7)
  and 0.736 (L12) for paren, 0.747 (L12) for quote, against a token-matched null p99.9 of about 0.52
  (`results/e63_occurrence_v2.json`). The settings were tuned with parenthesis labels on calibration block P2, so
  quotation is the untuned control.
- One L12 direction (logcosh #13 / skew #16) passed a preregistered steering test on untouched text: Δ 1.49, against a
  random-direction max of 0.75 and a shuffled-signature max of 0.85 (`results/e66_confirm.json`).
- Gemma Scope 2 already covers that direction as a blend of subordinator and preposition latents (`results/e67_*`). No
  coverage test tried here could certify "uncovered", and no unnamed variable is claimed.

## Start here

| File | What it is |
|---|---|
| `SUMMARY.md` | The distilled result, with a source key mapping every number to a REPORT section and result file. |
| `REPORT.md` | The full record, stage by stage, including failures and post hoc readings. Verdicts are never edited. |
| `PREREG_*.md`, `NOTE_*.md` | Preregistrations and rubrics, each hashed before the step it governs. |
| `docs/HASHES.txt` | The sha256 ledger: 155 entries over 142 paths, in the order they were written. |
| `PROCESS.md` | A proposal for a repeatable version of this process. Not adopted; it authorises nothing. |
| `BUDGET.md` | Disk, memory and time budgets, written before any compute. |

## Layout

| Path | Contents |
|---|---|
| `scripts/` | Every experiment script. `e01`–`e73` are this project's; the numbered `02`–`51` scripts are copied from the earlier gf-discovery project, as is `src/`. |
| `orig_scripts/` | The earlier project's scripts as copied, unmodified. |
| `src/gfdisc/` | The shared package: model and SAE loading, caching, fits, nulls. |
| `tools/` | `launch.py` and `watchdog.py` (detached runs with a swap guard and chained verdicts), `verify_hashes.py` (added for this release). |
| `docs/` | The ledger, block and seed records, addenda, smoke tests and post hoc check scripts. |
| `results/` | Every result JSON, and NPZ files under 5 MB. |
| `cache/occ/` | Token metadata for the discovery, replication and shuffled blocks, needed to rebuild signatures. |
| `anchors/` | The two files holding the 5,152 row-to-token anchors that verify the corpus (from the earlier project). |
| `logs/` | Run logs, including the corpus identity checks. |

## Checking the ledger

```sh
python3 tools/verify_hashes.py
```

It reports each ledger path as OK, APPENDED (`e63_occurrence_v2.json`, appended to after its freeze hash; the ledger
records its final sha), NOT RELEASED (too large, listed in `RELEASE_EXCLUDED.txt`) or ABSENT (deleted upstream after
use). On this release: 120 OK, 1 APPENDED, 15 NOT RELEASED, 6 ABSENT, 0 mismatches.

## What is not here, and how to get it

- **Model weights.** `google/gemma-3-1b-pt` is gated on Hugging Face: accept its terms, then export `HF_TOKEN`. The
  scripts load it from a local snapshot under `hf_home/`, so set `HF_HOME=hf_home`.
- **SAE weights.** Gemma Scope 2 (`google/gemma-scope-2-1b-pt`) is downloaded on first use by `src/gfdisc/saes.py`.
- **Corpus tokens** (`results/*.i32`). `scripts/e01_pass.py` rebuilds them from the FineWeb `sample-10BT` stream and
  aborts if any of the 5,152 anchors fails, which would mean the dataset order changed. It reads the anchors from
  `~/Downloads/gf-discovery/results/` (its `OLD` constant): copy `anchors/` there, or point `OLD` at `anchors/`.
- **Activation caches, occurrence sets and raw-row arrays.** These are several GB and all regenerable by the scripts.
  `RELEASE_EXCLUDED.txt` lists each file with its size and whether the ledger records it.
- **Python environments.** `requirements.txt` (Python 3.13: torch 2.14.1, transformers 5.19.0) and
  `requirements-parse.txt` (Python 3.12: spaCy 3.8 for the grammar rules) list the exact versions used. The spaCy models
  `en_core_web_sm` and `en_core_web_trf` 3.8.0 install with `python -m spacy download`.

## Caveats

- Hashed scripts are byte-identical to the ledger, so a few still hard-code the original path
  `/Users/scasella/Downloads/gf-discovery-ext`. Run from a checkout at that path, or edit the constant locally (the
  ledger will then flag that file).
- bf16 results drift slightly across torch versions; REPORT.md's Stage 0 records the size of that drift.
- Every run here was local on Apple MPS. Timings in REPORT.md are for that machine.

## License

Code and documents in this repository: MIT (see `LICENSE`). The model, the SAEs and FineWeb are not redistributed and
keep their own terms. Short text excerpts in `results/` and `cache/occ/` come from FineWeb (ODC-By 1.0).
