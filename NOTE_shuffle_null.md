# NOTE — Selection-matched shuffle null for the e72 raw-residual lead (e72b; rubric fixed 2026-10-16 before any number it governs)

**What this is.** e72 found one lead on raw L12 h (skew #49). It beat 10,000 same-subspace random directions on SAE
error share (ES) and passed every screen. Random directions are not a selection-matched null, though: ICA picks
non-Gaussian directions. On e72's one shuffle (S2), 10 ICA units beat S2's random max, but none of them were
screened. On D2, 4 of the units above the max were screened. That is n = 1 shuffle. This NOTE asks whether
**ICA on shuffled text also produces SCREENED units whose ES beats that shuffle's own random max**, over 5 new
shuffles.
- **Data.** Burned D2 only (corpus 1,000–1,999), re-ordered within each sequence. No fresh corpus text (2,000+).
  One MPS forward pass of burned text per shuffle (L12 residual only; no SAE encode, no decoded memmaps).
  **The capture runs only after the user confirms it.**
- **No verdict is amended.** e64–e72 stand. This is a sharpening of e72's GO/STOP reading, not a new decision tier.
- **What was seen before this note.**
  - REPORT.md and SUMMARY.md through e72.
  - `results/e72_coverage.json`: D2 counts, the S2 block (10 above max, 0 screened; skew #49 at S2 p51).
  - `results/e72_verify.json` (V1–V3).
  - Before hashing, `capture` was smoke-tested with `--test-dir` and no model. It built and checked the 5 shuffle
    metas and wrote synthetic rows; the only numbers were kept-row counts, a property of token positions only.
  - Before hashing, the `analyze` code was dry-run once with `--test-dir`. That run used real S2 rows (shuffle 0),
    whose primary numbers are already in e72_coverage.json, plus synthetic rows for shuffles 1–5. Its S2 numbers
    are the reproduction check (§5). The new S2 descriptives it computed (secondary screens, D2 filters on S2)
    play no part in the decision.
  - **No number on any new shuffle has been computed.**

## 1. Shuffles (new seeds; S2 is not reused)

- Shuffles **k = 1…5**, tags `occS2b1` … `occS2b5`. Sequence i (corpus row 1,000 + i) is permuted by
  `np.random.default_rng(700_000 + 10_000·k + 1_000 + i).permutation(255)`. That is e60's recipe with a new offset.
  S2's seeds are 601,000–601,999; these are 711,000–751,999, so none coincide.
- Meta files: `cache/occ/<tag>_meta.npz` with e60's exact fields (tokens, seq, pos, shuffled = True,
  source = "corpus", layers = [12]).
- **Asserted:**
  - every sequence is a permutation of D2's;
  - no shuffle equals D2, S2 or another new shuffle, overall or in any single sequence.
- Rows: e63's kept-row rule on each shuffle's own meta, via `e64.rows_only(tag)`. Counts are recorded at capture,
  and the analysis asserts it sees the same counts.

## 2. Capture (MPS; after confirmation)

- `residual()` is copied verbatim from e69 phase p1 (it is nested there, so it cannot be imported). It prepends
  BOS, uses batches of 16 sequences, takes the layer-12 output `[:, 1:, :]`, casts to fp16 and asserts finiteness.
- **Drift check first.** The first 32 D2 sequences are re-captured and compared with `e67_raw_L12_rows.npy` (e69's
  bar: ≥ 99% identical, min coordinate corr ≥ 0.999). If it fails, nothing is captured.
- Output per shuffle: `results/e72b_S2b{k}_raw_L12_rows.npy` (fp16 kept rows, about 470 MB). Each meta and each
  rows file is hashed as soon as it is written.
- `results/e72b_capture.json` (seeds, row counts, drift check) is hashed at the end.

## 3. Analysis (CPU): exactly e72's S2 branch, per shuffle

This is `s2_descriptive` in `scripts/e72_raw_residual_pilot.py` (sha `338c10b2…`, asserted), mirrored line for
line. Only the tag, rows path and random seed are parametrised; the e72 module's own functions are called.
- D2's raw PCA and maps from `results/e72_units_raw_L12.npz` are applied to the shuffle's demean3'd raw rows
  (`project(…, own_mean=False)`, `e64.apply_maps`).
- Independent `e63.ica_units` fits: logcosh and skew, 64 each, seeds 0/1/2, split-half by sequence parity.
- The error e = h − SAE(h) uses L12 `resid_post_all` 16k l0_big (`e72.error_W`), pushed through the same maps.
- **ES** = var(b·W_e)/var(b·W_h) for each unit's exact readout b, against **that shuffle's own 10,000 random
  W-space directions**. Seeds: 7201 for S2 (as e72), 7211–7215 for k = 1…5. None equals 7200 or 7300.
- **Screened** = `e63.ica_pass(u, 0.8, False)`: η²(token) ≤ 0.3, η²(token) of y² ≤ 0.3, split-half repro ≥ 0.8.
- **Shuffle 0 = S2** (`results/e69_p1_S2_raw_L12_rows.npy`, seed 7201) runs through the same code. It must
  reproduce e72's S2 block (§5).

## 4. Statistics

**Primary, per new shuffle k:**
- **s_k** = the number of SCREENED units (of 128) with ES above shuffle k's random max.
- Also reported: n_k, all units above the max, per kind.

**The D2 reference is fixed: s_D2 = 4** (logcosh #37, #49; skew #49, #52; e72_coverage.json). D2's lead count
under the full lead rule is 1 (skew #49).

**Carried over, per shuffle.** Each D2 filter (b from D2's readout) evaluated on the shuffle: ES, its percentile
among that shuffle's random directions, whether it beats the max, and its presence (best |cos| of the D2 W-space
pattern with the shuffle's mixing columns, `e64.best_match`; ≥ 0.8 = context-generic).
- **skew #49: primary report.**
- logcosh #37, skew #52, logcosh #49: descriptive.

**Secondary, descriptive only (declared now, no role in the decision).** ℓ_k = the number of shuffle units that
pass e72's **full** lead rule (NOTE_raw_residual_pilot §6), with the screens recomputed on the shuffle:
- ES above the max, on all rows and on each sequence-parity half (each half against its own random max, same
  directions);
- SCREENED and not identity-suspect (`e64.identity_diag`: tail TDI ≤ 3, largest tail type share ≤ 0.30);
- |Spearman| ≤ 0.3 with the demean3'd log‖h‖, log‖e‖ and L0;
- |Spearman| < 0.3 with line_pos, sent_pos (e61 battery on the shuffled text) and tokens since punctuation.
- (Shuffles have no control units.)

The identity screen cut D2 from 4 to 1, so ℓ_k is the closest match to how skew #49 was selected. It is still
descriptive, because the brief fixes the primary count.

**Also descriptive:**
- s_k as a share of the shuffle's screened units, next to D2's 4 of 60. S2 had only 35 screened units.
- FVU, the random-ES quantiles, and the per-kind ES quantiles.
- The pooled count over S2 plus the 5 new shuffles.

## 5. Checks (asserted; any failure stops the analysis)

- `sha256(scripts/e72_raw_residual_pilot.py) == 338c10b2…`. `e72.integrity()` holds: the e64 frozen version and
  the choice file are unchanged.
- The capture JSON, metas and rows are hashed, with row counts as recorded.
- **S2 reproduction** (shuffle 0) against the hashed `e72_coverage.json` S2 block:
  - n rows 202,866;
  - per-kind n_above_max 5 / 5, n_screened_above_max 0 / 0, n_screened 15 / 20;
  - random-ES max and p50 within 1e-9;
  - skew #49's ES within 1e-6, with percentile 0.5088.

## 6. Decision rule (fixed here, before any number)

This is computed on the **5 new shuffles only**. S2's count (0) was seen before this rule was written, so it is
reported but not used.

| reading | condition |
|---|---|
| **SELECTION EFFECT** | any s_k ≥ 4, **or** median(s_1…s_5) ≥ 2 |
| **SURVIVES THIS NULL** | s_k = 0 for all five |
| **INDETERMINATE** | otherwise: the null does produce screened above-max units, but fewer than D2 |

- **SELECTION EFFECT:** D2's "4 screened above max" is within what shuffled text produces, so the e72 lead is read
  as a selection effect of ICA plus screening.
- **SURVIVES THIS NULL:** this null alone does not explain D2's count.
- **This is a count comparison, not a test.** With K = 5 the smallest attainable empirical p is 1/6. So
  "survives" never supports a discovery claim. It only removes one objection. e72's other caveats still stand
  (single-filter route, 76% SAE share, the same unit as decoded skew #31).
- **The null is procedure-matched, not distribution-matched (fixed in advance).** The SAE does worse on shuffled
  text: S2's FVU was 0.43 against D2's 0.19, and its random ES spans 0.31–0.80 against 0.27–0.36. D2's maps are
  applied unchanged, so W on a shuffle is not white. And the screened count differs (S2: 35 against D2's 60).
  Each shuffle's units are compared with that shuffle's own random max, as e72 did. Any difference between D2 and
  the shuffles therefore mixes "context structure" with "different SAE regime". The reading must say so whichever
  way it falls.

## 7. Single-filter route (cited, no new computation)

From `results/e72_verify.json` (V3, LS pullback filter in the 1,151 raw coordinates; random max 0.246, LS R² min
0.784):
- skew #49: 0.231, below the max (p99.6), LS R² 0.765;
- logcosh #37: 0.359, above the max, R² 0.696;
- skew #52: 0.355, above the max, R² 0.693.

## 8. Phases, outputs, constraints

| phase | output (each hashed with `e64.append_hash` as soon as written) |
|---|---|
| capture (MPS, after the user confirms) | `cache/occ/occS2b{k}_meta.npz`, `results/e72b_S2b{k}_raw_L12_rows.npy`, `results/e72b_capture.json` |
| analyze (CPU) | `results/e72b_null_arrays.npz`, `results/e72b_null.json` |

- `scripts/e72b_shuffle_null.py` asserts that this NOTE and itself are in docs/HASHES.txt.
- Each phase is launched once, detached under `caffeinate`, with a 12 GB swap watchdog. A crashed capture may be
  resumed: hashed files are skipped, as in e69 p1.
- Write-up: a REPORT.md section "e72b" after e72's, a top addendum line, and a one-line SUMMARY.md pointer.
  e64–e72 verdicts are not amended. Nothing is raised upstream.
