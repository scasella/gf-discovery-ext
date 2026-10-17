# NOTE — Raw-residual pilot: the Stage 4 v2 occurrence instrument on raw L12 h, and whether any unit lies measurably off the SAE span (e72; rubric fixed 2026-10-16 before any number it governs)

**What this is.** Every occurrence unit so far (e63–e71) was found in the SAE-decoded residual,
fp16(codes) @ W_dec (e60). Each such unit is a linear readout of SAE codes by construction, so
"uncovered" could not be measured (e68, e71). Raw h has never been searched. This pilot runs the
frozen v2 instrument on the **raw** L12 residual and only afterwards measures how much of each unit
lies in the SAE error.
- **Not option 2 resumed.** e69/e70 searched the SAE error alone. Here the search space is the full
  residual; the error enters only in the coverage stat.
- **Data.** Burned D2 only (corpus 1,000–1,999): `results/e67_raw_L12_rows.npy` (D2 kept rows, raw L12,
  fp16). S2 (`results/e69_p1_S2_raw_L12_rows.npy`, D2 shuffled within sequences) is a descriptive
  comparison. No fresh corpus text, no model forward pass, no extraction. CPU only.
- **No verdict is amended.** e64–e71 stand.
- **What was seen before this note.** REPORT.md and SUMMARY.md, including e70's review numbers (along
  e69's error-space filters raw h is about 70% SAE span, error share 0.26–0.31). The shapes of the input
  files. **No number on any raw-h unit, map or error share has been computed.**

## 1. Rows and input

- Rows: e63's kept-row rule on D2's own meta (positions ≥ 4, token types with ≥ 8 rows), via
  `e64.rows_only("occD2")`. Asserted equal to `occ_units_v2_L12.npz["row"]` (203,328 rows) and to the
  length of `e67_raw_L12_rows.npy`, as e69 does.
- h is the stored fp16 row cast to fp32. Coordinate 1038 (the near-overflow coordinate; also
  `e61.drop_for(12)`) is dropped before identity removal (1,151 coordinates).

## 2. Instrument (frozen v2, reused unchanged; nothing tuned)

| step | code | rule |
|---|---|---|
| identity removal | `e63.demean` | type, then document, then position means, cycled 3 times, exactly as `e63.represent_base` |
| PCA | `e63.pca_white` | PCA-64 fitted on every third row, whitened |
| equalisation | `e64.fit_maps` (= `e63.type_whiten`, asserted < 1e-4) | per-type covariance whitening, N_MIN 32, n0 16; rarer types use the pooled within-type covariance; then a global re-whitening |
| units | `e63.ica_units` | full-rank FastICA-64, logcosh and skew, seeds 0/1/2, split-half by sequence parity |
| screens | `e63.ica_pass(u, 0.8, False)` | η²(token) ≤ 0.3, η²(token) of y² ≤ 0.3, split-half repro ≥ 0.8, no bimodality bar (the frozen `e63_pilot_choice.json`) |

- Scope as e64: ICA units only, no clusters.
- The units (scores, mixing, linear readout from W, maps, patterns) are written and hashed **before any
  label is computed**.

## 3. Gate (positive control on raw h)

- Labels: `paren_inside`, `quote_inside` from `e61.feature_battery`.
- Null: `e63.null_engine`, 1,000 token-matched label permutations (seed 99).
- `e63.evaluate` over the SCREENED units of each kind, floor **0.70** (the frozen L12 `auc_floor`).
  A label is **recovered** iff, for logcosh or skew, the max |AUROC| over screened units is ≥ 0.70
  **and** above the null p99.9 of that max.
- **PASS iff both labels are recovered.** This is stricter than S4v2's any-of rule (reported
  alongside), because the task names both controls and both passed on decoded L12 (0.736 / 0.747).
- Reported for context: the unscreened bounds and the supervised probe AUROC.
- **Control units** := for each label and kind that recovers, the best unit (e64's `control_units`).
- **If the gate fails: STOP.** The instrument is not validated on raw h. No coverage number is computed.

## 4. Coverage stat

**Error.** e = h − (JumpReLU_fp32(h) @ W_dec + b_dec), `resid_post_all/layer_12_width_16k_l0_big`,
computed on all 1,152 coordinates of the fp16-stored row, 1038 dropped afterwards (`e69.error_rows`,
HF_HOME = `<ext>/hf_home`). SAE(h) = h − e.

**Filter.** Per-type equalisation makes a unit's score piecewise linear in the residual, so a unit has
no single raw-space filter. Every step after the type-aware first moment is linear given the row's
type. So e and SAE(h) are pushed through **the same fixed maps** fitted on h:
- demean3 (same groups) → PCA projection with h's U and sd (each part centred on its own mean) →
  `e64.apply_maps` (each part's own type means and global mean subtracted; h's per-type and global
  matrices) → W_e, W_s.
- Asserted: `apply_maps(h)` reproduces the fitted W_h (< 1e-4), and W_h = W_e + W_s (< 1e-4).
- A unit u is y_u = b_u · W + c_u, with b_u its exact linear readout from W (least squares on
  [W, 1]; R² ≥ 0.999 asserted). For a row of type t this is a filter f_{u,t} on the residual;
  var(e · f) / var(h · f) is evaluated as var(b_u · W_e) / var(b_u · W_h).

**Statistic (primary).** ES_u = var(b_u · W_e) / var(b_u · W_h), over all D2 kept rows.
- Reported next to it: ES_cov_u = cov(b_u · W_e, b_u · W_h) / var(b_u · W_h). This is e70's
  convention; ES_cov plus the SAE share equals 1. The two differ when e and SAE(h) correlate along b_u.

**Random baseline (same subspace).** 10,000 directions g, iid uniform on the unit sphere of the
64-d equalised W space (seed 7200), each scored by the same ES.
- This follows e64's `N_NULL_DIRS` convention. The task's floor is ≥ 100, but 100 draws cannot give a
  p99.9.
- Reported: p50, p99, p99.9 and max, for ES and ES_cov; the count of the 128 units above each.
- **There is no absolute bar.** The random ES is low when the SAE's FVU is low. The discriminating
  stat is whether a unit beats the random max (and p99.9).
- Expected false positives if units were exchangeable with random directions: 128 / 10,001 ≈ 0.013
  above the max.
- **Ceiling (descriptive).** The generalised eigenvalues of (cov W_e, cov W_h): the largest ES any
  direction in this space can reach.

**Controls.** The gate's control units: ES, ES_cov and their percentiles among the random directions.

**Match to e64's decoded-span units.**
- Each raw unit's pattern is Cov(Z, y) pulled back to raw coordinates, P_U · (sd ⊙ Cov(Z, y)) — e64's
  `reconstruct` recipe with the raw PCA. It is compared by |cos| in the 1,151 kept coordinates with
  e64's 128 `dir_*` (`results/e64_maps_L12.npz`). **|cos| ≥ 0.8 = same unit.**
- A null: the same pattern for the 10,000 random directions, max |cos| against e64's 128 (p99.9, max).
- Sanity check: |corr| of the scores with e64's decoded-span scores (`occ_units_v2_L12.npz`) on the
  same 203,328 rows.

## 5. Screens (sources cited; thresholds unchanged from their source)

| screen | source | pass iff |
|---|---|---|
| identity | e63 `ica_pass` + e64 `identity_diag` | SCREENED (§2) **and** not identity-suspect: 1% heavy-tail TDI ≤ 3 and largest tail type share ≤ 0.30 |
| norm, L0 | e69 SAE-state screen | \|Spearman\| ≤ 0.3 with each of identity-removed (demean3) log‖h‖, log‖e‖ and L0 (16k l0_big) |
| position | e70 position screen | \|Spearman\| < 0.3 with each of line_pos, sent_pos (e61 battery) and tokens since the last punctuation piece (e70's definition) |

## 6. Lead rule (fixed here, before any number)

A unit (any of the 64 logcosh and 64 skew) is a **LEAD** iff all hold:
1. **Off span:** ES_u > the random max (§4), on all D2 kept rows.
2. **Identity, norm, L0, position:** it passes every screen in §5.
3. **Split-half reproducible:** split-half repro ≥ 0.8 (part of SCREENED), **and** ES_u, computed
   separately on the even-sequence rows and on the odd-sequence rows (the same full-fit b_u), beats
   the random max computed on that same half (the same 10,000 directions).
4. **Not a control:** it is not one of the gate's control units. Those are named by construction;
   their ES is reported as a control.

**A lead only licenses writing a fresh-text prereg. It never supports a discovery claim.**

## 7. Independent same-subspace verification (required before any GO is reported)

- `docs/e72_verify.py`. Drafted and smoke-tested on synthetic rows before any real run, and hashed
  together with this NOTE and the script, before any number. If it has to change after the coverage
  numbers are seen, the new version is labelled post hoc and hashed as a superseding entry before its
  run. Burned D2 only.
- **V1, map route re-implemented.** Fresh code that does not call `e64.apply_maps` or
  `e69.error_rows`: numpy SAE error, its own per-type loop. Agreement: each lead's ES within 1e-3 of
  the coverage value.
- **V2, fresh baseline.** 10,000 new random W-space directions (seed 7300). Agreement: each lead
  still beats the V2 max.
- **V3, single-filter route (descriptive).** A least-squares pullback filter in raw coordinates
  (y_u regressed on demean3'd h), with its R² reported; var(e·f)/var(h·f) for the leads and the
  random directions on this route.
- A lead is **verified** iff V1 and V2 agree. If no lead exists, the verification still checks the
  controls and the top-ES unit (V1 to 1e-3; V2 rank) so the STOP rests on checked numbers.

## 8. Decision

- **STOP-GATE:** the gate fails (§3). Report that the instrument is not validated on raw h.
- **GO:** at least one verified lead. GO licenses writing a fresh-text prereg only.
- **STOP:** otherwise.

**Fixed reading in advance.** ICA selects non-Gaussian directions, and JumpReLU codes are sparse.
So ICA units in raw h may land *inside* the SAE span, with ES below the random median. If so, that
is the headline: searching raw h re-finds SAE-span structure, and "off span" has to be looked for
another way. It is not an instrument failure.

## 9. Descriptive only (no role in the decision)

- FVU (raw and identity-removed), the per-PC error share, ES_cov, the ceiling.
- Naming of each lead and of the 5 highest-ES units: e64 battery associations (`e64.associations`
  on `e64.extended_battery`), the best e64c rule |AUROC|, e64's 12 + 12 heavy-tail contexts.
- **S2 (shuffle) comparison.** D2's raw PCA and maps applied to S2's raw rows (e64 `represent_other`
  recipe); independent ICA fits (`e63.ica_units`); S2's error through the same maps.
  - The count of S2 units with ES above S2's own random max (10,000, seed 7201).
  - For each lead: its presence in S2 (best |cos| of its W-space pattern with S2's mixing columns,
    ≥ 0.8 = context-generic, e64's rule), and its filter's ES on S2.

## 10. Phases, outputs, constraints

| phase | output (each hashed with `e64.append_hash` as soon as written) |
|---|---|
| gate | `results/e72_units_raw_L12.npz` (before any label), then `results/e72_gate.json` |
| coverage (only after a hashed PASS) | `results/e72_coverage.json`, `results/e72_coverage_arrays.npz` |
| verify | `docs/e72_verify.py` (hashed with this NOTE), `results/e72_verify.json` |

- `scripts/e72_raw_residual_pilot.py` asserts that this NOTE and itself are in docs/HASHES.txt.
- Smoke-tested before hashing on **synthetic** rows only, in a test directory.
- Local CPU only (torch CPU for the SAE matmul; no MPS, no cloud). Each phase is launched once,
  detached (`start_new_session`) under `caffeinate`, with a 12 GB swap watchdog. No retries.
- Write-up: a new final REPORT.md section and an optional one-line SUMMARY.md pointer. e64–e71
  verdicts are not amended.
