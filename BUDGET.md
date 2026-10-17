# BUDGET.md — Stage 0: disk, layers, corpus (written 2026-10-14, before any compute)

Binding constraint: disk. Volume at 92–93% (33–39 GB free, fluctuating with other sessions).
Rule for this project: **peak footprint of `~/Downloads/gf-discovery-ext/` ≤ 20 GB**, and abort if
system free space falls below 8 GB at any checkpoint.

## 1. SAE inventory (Gemma Scope 2, `google/gemma-scope-2-1b-pt`, ungated, CC-BY-4.0)

Checked against the HF file listing and each SAE's own `config.json` on 2026-10-14.

| site | layers available | widths | L0 variants |
|---|---|---|---|
| `resid_post_all` | 0–25 (every layer) | 16k, 262k | small, big |
| `resid_post` | 7, 13, 17, 22 only | 16k, 65k, 262k, 1m | small, medium, big (+ 262k medium_seed_1) |

Chosen suite: **`resid_post_all`, width 16,384, `l0_big`**, the same suite the L0–L10 work used.
`params.safetensors` = 151,131,000 B each.

| layer | 5 | 6 | 7 | 8 | 9 | 12 | 14 | 16 | 18 | 20 | 22 | 24 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| nominal L0 (`l0_big`, 16k) | 94 | 101 | 108 | 115 | 120 | 120 | 120 | 120 | 120 | 120 | 120 | 120 |
| nominal L0 (`l0_small`, 16k) | 15 | 16 | 18 | 19 | 20 | 20 | 20 | 20 | 20 | 20 | 20 | 20 |
| nominal L0 (`l0_big`, 262k) | – | – | – | – | – | 120 | 120 | 120 | 120 | 120 | 120 | 120 |

`resid_post` (non-"all") 16k L0s: L7 18/54/132, L13/17/22 20/60/150 (small/medium/big). Not used.
The 262k SAEs are 2.42 GB each, and 262k > 65,535 breaks the cache's uint16 index format.
Not used.

Deep layers chosen: **12, 14, 16, 18, 20, 22, 24** (the suggested set). By `results/closure_depth.json`
(templated raw residual, supervised), the months ring closes at 16 and 18–20, so 16, 18 and 20
double as supervised positive-control layers. L5–L9 SAEs are also needed: Stage 2 blind-runs
there, and their type means were never persisted.

## 2. Corpus: identical to the L0–L10 cache, not smaller

The FineWeb `sample-10BT` stream is reproduced exactly: first qualifying documents (≥ 256
tokens after truncation), seq_len 256, BOS dropped, giving **15,696 sequences × 255 = 4,002,480
rows**. 06's `--n-tokens 4000000 --bs 16` ended at 981 batches.

Why not cut the corpus to save disk: the pool rule (count ≥ 100) depends on corpus size. With
the identical corpus, the pool is the identical 3,795 types. That makes deep-layer results
comparable row-for-row with L0–L10, and keeps every stored artifact valid:
- the L6/L7 lag deltas;
- `token_map.json`;
- the member hashes, which are md5 over pool positions.

Identity is **checked, not assumed** (see §6).

Held-out reserve: the stream continues past the corpus and saves **tokens only** for the next
8,000 qualifying sequences (about 8 MB). This is the fresh-context pool for Stage 3 and for
every replication tier. Plane and tangent fits use corpus type means; steering and replication
are scored on reserve contexts. This follows the cross-project note's warning that in-sample
fitting inflates steering wins.

## 3. Per-row cost

CSR as written by `gfdisc.cache.SparseWriter` (06), per layer:
- indices: uint16, 2 B per nnz;
- values: float16, 2 B per nnz;
- indptr: int64, 8 B per row;
- tokens.i32: 4 B per row.

nnz/row in the old `l0_big` cache ran 1.00–1.02× nominal (L9 121.4, L10 122.4). The deep-layer
`l0_small` SAEs ran 1.16–1.24× nominal in §0.8 of results.md, so the deep `l0_big` overshoot is
unknown. Two rows are budgeted:

| scenario | nnz/row | B/row/layer | GB/layer (4,002,480 rows) |
|---|---|---|---|
| planned (1.02×) | 122.4 | 4·122.4 + 12 = 501.6 | **2.008** |
| sensitivity (1.20×) | 144 | 4·144 + 12 = 588 | **2.353** |

All 7 deep layers at once would take 14.06 GB (planned) or 16.47 GB (sensitivity) of cache. On
top of the fixed 4.7 GB below, that is 18.7–21.2 GB, which breaks the 20 GB cap in the
sensitivity case. **Rejected.**

## 4. Plan: two passes, deleting the cache between them

Fixed costs, measured:
- venv: 1.0 GB;
- `hf_home`: 3.6 GB (gemma-3-1b-pt 2.0 GB plus 12 SAEs 1.81 GB);
- copied input results: 0.05 GB.

Total fixed: **4.65 GB**.

**Pass 1** (one forward over the corpus, capturing L4–L9 and all 7 deep layers):
- Sparse CSR cache written for **L12, L14, L16, L18**.
- Per-type accumulators in fp64 over pool types, split by sequence parity (R9 halves):
  - raw residual for L4–L9 plus all deep layers (13 layers);
  - SAE-decoded `fp16(code) @ W_dec` for L5–L9 plus all deep layers (12 layers).

  Script 22 builds its own code-mean matrix from the cache, so these are not a substitute for
  it. They are the split-half means for Stage 2. The L4–L9 raw means rebuild the deleted
  `phase3_token_type_mean_resid_fulln.npz` in 35b's format, which 42, 47, 47b and 49 need.
- Tokens for the 8,000-sequence reserve.

**Stage 1, group A**:
- script 22, unchanged, on L12–L18;
- flag list → 42-deep (lag-delta streams from the cache, saved npz);
- then **delete the L12–L18 cache**, keeping `tokens.i32`.

**Pass 2**: forward over the same corpus; sparse cache for **L20, L22, L24** only.

**Stage 1, group B**: 22 → 42-deep, then **delete the cache**.

**Stages 2–4**: no sparse cache.
- Stage 2 runs on the saved type means.
- Stages 3 and 4 run their own MPS forwards on demand. Stage-4 per-occurrence matrices are
  capped at 2.5 GB in total (float16).

| item | planned GB | sensitivity GB |
|---|---|---|
| fixed (venv + hf_home + inputs) | 4.65 | 4.65 |
| pass-1 cache, 4 layers | 8.03 | 9.41 |
| type-mean npz (A/B/full, raw 13 layers + SAE 12 layers, fp32) | 1.31 | 1.31 |
| rebuilt fulln npz (35b format) | 0.23 | 0.23 |
| lag-delta npz, 7 layers | 0.12 | 0.12 |
| reserve tokens, logs, result JSON | 0.20 | 0.20 |
| **peak (pass 1 / group A)** | **14.5** | **15.9** |
| peak (pass 2 / group B, after group-A deletion) | 12.5 | 13.6 |
| peak (stage 4, per-occurrence ≤ 2.5) | 9.0 | 9.0 |

**Peak ≤ 15.9 GB against the 20 GB cap**, leaving at least 4.1 GB of headroom.

Guards in code:
- The pass script aborts **loudly**, with no silent truncation, if any of these happen:
  - cache bytes exceed 10.5 GB (pass 1) or 8.0 GB (pass 2);
  - `du` of the project folder exceeds 19.5 GB;
  - system free space drops below 8 GB.
- nnz/row is measured on the first 50 batches and appended below before the pass continues.
- Every captured residual batch gets an `isfinite` check. The old project once lost 196 rows to
  NaNs under MPS memory pressure.

## 5. Time (estimates, replaced by measurements below)

The old 11-layer pass ran at 515 tok/s, which is 2.2 h for 4M tokens. Per-batch cost is
dominated by CSR conversion of the cached layers: a 267 MB dense code block per layer per
batch. Accumulator layers only move 19 MB of decoded vectors.

| step | estimate |
|---|---|
| pass 1 | 1.5–2.5 h |
| pass 2 | 1–1.5 h |
| script 22 | about 20 min/layer (the old run took 1,116 s for 11 layers plus caching I/O) |
| 42-deep streams | 3–5 min/layer |

This is all MPS or CPU on the M4 Pro. No cloud.

## 6. Stop rules declared before any compute

1. **Corpus identity (early abort).** These two stored files hold exact row → token anchors:
   - `phase3_per_occurrence_coords.npz`: `month_row_index`/`month_token_id`, 2,930 rows;
   - `phase3_context_ladder_natural_means.npz`: `occ_rows_<tokid>`.

   Every anchor row below the rebuilt prefix is asserted equal after about 1,000 sequences, and
   again at the end. The final pool must equal the stored `pool_ids` (3,795 types). If either
   fails, the pass stops, and no new pool is declared without reporting first.
2. **Numeric reproduction gate (the Stage-1 stop gate).** The rebuild must reproduce:
   - strict-12 raw type-mean gap_ratio within ±0.01 of the phase-4 prereg §0 values:
     L4 1.585, L5 1.475, L6 1.521, L7 1.611, L8 2.012, L9 1.766;
   - script-22 SAE-channel pre-drop gap_ratio within ±1e-3 of 42's `ANCHOR_GAP_RATIO`:
     13-set at L6 1.6588, 14-set at L7 1.6202.

   This is the supervised positive control for the instrument. If it fails, stop and report.
3. **Deep-layer months/weekdays recovery is REPORTED, NOT GATING.** The deep closure evidence
   (L16, L18–20) comes from templated, supervised raw residuals. Script 22 on corpus type means
   recovered the months at only L6/L7 out of the 11 closing layers in L0–L10. There is no prior
   evidence that corpus type means close at depth. So:
   - a deep-layer miss is reported per layer as a measurement;
   - it is not a pipeline failure and does not stop the run;
   - a deep-layer hit is not a validation of anything beyond that layer.
4. **Disk.** Any guard in §4 trips → stop and report.

## 7. Model provenance

`google/gemma-3-1b-pt` revision `fcf18a2a879aab110ca39f8bffbccd5d49d8eb29` was downloaded once
into `hf_home/`, using a token the user supplied in chat. The token was passed as an environment
variable for that single download and is not stored in any file here. Everything afterwards
runs with `HF_HUB_OFFLINE=1`.

`model.safetensors` sha256 is `ee5250f6eb1aa7cfb729dfd4dc8d9964fd772324776c6d00bf2bc674c069cb27`.
That matches the hub's LFS record, and also the ungated `unsloth/gemma-3-1b-pt` mirror.
`~/Downloads/gf-discovery/hf_token.rtf` was never read.

## 8. Measured (appended during the run)

(filled in below as measurements arrive)

### 8.1 Smoke/profile run (640 sequences, full pass-1 layer set), 2026-10-14 23:5x

- Corpus identity: 148 of 148 early anchors match at sequence 320, and 242 of 242 at sequence 640.
- **nnz/row, measured**: L12 119.1, L14 121.8, L16 120.9, L18 121.7, i.e. 0.99–1.015× nominal.
  That is **≈ 2.00 GB/layer**, so the pass-1 cache is ≈ 8.0 GB, matching the planned row.
- Throughput: 1,102 tok/s with 4 cached layers, 12 SAE accumulators and 13 raw accumulators.
  Breakdown per 16-sequence batch:

  | data | forward | CSR (4 layers) | SAE accumulators (12) | raw accumulators (13) |
  |---|---|---|---|---|
  | 0.4 s | 1.55 s | 0.86 s | 0.75 s | 0.12 s |

  That puts **pass 1 at ≈ 62 min** and pass 2 at ≈ 45 min.
- System free space fell from 39 to 29 GB during setup because of other sessions' writes; this
  project accounts for 4.94 GB of that. The 8 GB free-space abort guard stays.
- Fixed footprint measured: **4.94 GB** (venv 1.0, hf_home 3.6, copied scripts/results 0.3).

### 8.2 Pass-1 attempt 1 aborted by the free-space guard (2026-10-15 00:5x)

- The project held 9.56 GB (cache 4.70 GB) at the abort. System free space fell from 28.7 GB
  to 4.8 GB in about 10 minutes.
- The cause was not the cache. It was macOS swap: 25 GB of swap files on the shared APFS
  container. My own concurrent smoke test of e46 had a 300 × 3795 × 1152 float64 distance
  tensor in `plant()`, giving a 20 GB RSS.
- The guard aborted loudly, as designed. The test was killed, the bug fixed (distances via
  dot products), and the partial cache deleted.
- **New rule:** nothing memory-heavy runs alongside a pass, and swap is checked before every
  pass.
- Pass 1 was restarted from sequence 0.

### 8.3 Pass 1 (attempt 2) actuals — 2026-10-15 01:45 → 02:35

- **nnz/row**: L12 119.06, L14 121.80, L16 121.21, L18 122.11.
- **Corpus loop**: 2,967 s at 1,347 tok/s. The reserve slice followed.
- **Project folder at the end**: 14.45 GB, against 14.5 planned. System free space was 22 GB
  afterwards.
- **Identity**: all 5,152 anchors pass, and the pool is reproduced exactly.
- **Gate**: see `docs/addendum_A1_gate_tolerance_2026-10-15.md`.

### 8.4 Stage-4 storage, revised before any extraction

- §4 allotted ≤ 2.5 GB to per-occurrence matrices. The preregistered design needs four sets:
  P, D, S and R.
- Each set is 1,000 sequences × 255 rows × 1,152 × 2 bytes per layer, about 0.59 GB.
- At 3 layers that is **about 7.1 GB**.
- The pass-2 cache (about 6 GB) is deleted before extraction. That gives a peak of roughly
  6.3 (fixed + type means) + 7.1 + 0.3 ≈ **13.7 GB**, inside the 20 GB cap.
- P is deleted after the pilot. S is deleted after the freeze.

### 8.5 Stage 4 and actual overall peak (2026-10-15)

- **Overall project peak: 18.3 GB** (apparent size; 17.9 GB on disk). It occurred with the
  Stage-4 D/S/R occurrence sets (4.9 GB) coexisting with the L20–24 sparse cache (5.7 GB).
- **This deviates from §8.4**, which said to delete the pass-2 cache before extraction. The cache
  was kept so the optional, user-deprioritised e42d on L20–24 could still run.
- The peak stayed under the 20 GB cap. System free space was 35–37 GB at the time.
- Pass 2 measured nnz/row of 124.2 / 124.5 / 125.6 at L20 / 22 / 24, in 2,126 s, with the project
  at 12.8 GB at the end of the pass.

### 8.6 Final footprint (2026-10-15, end of run)

- Deleted as regenerable intermediates:
  - the sparse cache (L20–24; L12–18 had been deleted earlier);
  - the Stage-4 occurrence sets (P, D, S, R).
- Kept: `results/` (type means, fulln, lag deltas, corpus/reserve tokens, all JSON), `hf_home`
  (model plus 12 SAEs), the venv, scripts and docs.
- **Final project size: 6.5G**.
