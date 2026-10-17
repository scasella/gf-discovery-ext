# NOTE — Coverage census: are any e64 candidate families NOT covered by a Gemma Scope 2 feature? (rubric, fixed 2026-10-15 before any census latent number)

**What this is.** A read-only census of every e64 discovery candidate at L7 and L12. It asks whether
an existing Gemma Scope 2 SAE latent already covers each candidate.
- **Authority.** The user asked for it ("option 1").
- **Data.** D2 only (corpus 1,000–1,999, already used by e64–e67). It processes no new data, steers
  nothing, and amends no earlier verdict (e64–e67).
- **Purpose.** It **decides whether a new confirmatory round has any target.** Only an uncovered,
  replicated, unnamed family is worth one. A false "not covered" creates a spurious target, which
  is the costly error here, so the rules below lean toward COVERED wherever they must choose.

**What existed before freezing.** Three adversarial reviewers (statistics, target selection,
leakage) read a first draft. Three code reviewers (conformance, runtime, logic) then read the
implementation. During review they computed, read-only on D2, **each unit's own
signature shares and base rates**: what the units' own top-200 rows score against their own
signatures. **No latent-level number for any non-family unit existed.** The only coverage numbers
were e67's, for L12 logcosh #13 / skew #16. The match thresholds below were set *after* the
reviewers reported those unit-level shares; this note says so plainly.

The code reviewers also computed, from existing e64c files, **each NAMED_BY_CONTEXT unit's best
e64c |AUROC|**. They found that the two L7 "after and/or/but" units cannot be tested with a
next-token signature: their own share is 0.085 against a base of 0.068–0.072. That finding is why
the **tail-overlap match** below exists, and why the calibration set is now derived by rule.

## Units

**Candidates:** every e64 discovery candidate (battery-UNNAMED and not NAMED_BY_CONTEXT):
- 34 at L7;
- 31 at L12.

**Families.** Families are taken from `results/e64_verdict.json["layers"][L]["families"]`,
restricted to candidate members. That gives 22 families at L7 and 21 at L12, of which 12 and 11
are R2-replicated. Each family also carries its **twins among all screened units**, from the
`twins` in `e64_labels.json`. These include battery-NAMED, context-named and control twins.

**Calibration units** (never targets):
- the 26 NAMED_BY_CONTEXT units;
- the 6 control units.

**Joint-null units:** 40 random directions per layer, built below. They go through the whole
rubric exactly as the units do.

**Inputs per unit:**
- d_u = `dir_{kind}[id]` from `e64_maps_L{L}.npz`, the residual-space pattern;
- y_u = the frozen D2 score `Y_{kind}[:, id]`, oriented by `sign_{kind}[id]` so that the heavy
  tail is the top;
- T_u = the 20-token blind signature from `e65_frozen.json`, built from the heavy tail.

## Disclosure: the source dictionary

The units were found in the **SAE-decoded** residual. e60 stores fp16(codes) @ W_dec of
`resid_post_all/layer_{L}_width_16k_l0_big`. So within each token type, y_u is a linear readout of
**that** dictionary's codes, and part of any coverage by it is built in.
- Every verdict is reported **with and without** the source dictionary.
- "Covered only via the source dictionary" is a reported class.
- **Target selection uses the verdict with the source included** (lenient toward COVERED).

## Dictionaries (Gemma Scope 2, gemma-3-1b-pt; hook `model.layers.N.output` for both sites)

| set | L7 units | L12 units |
|---|---|---|
| **Primary** | `resid_post_all` L5–L9 16k l0_big + l0_small (10); `resid_post` L7 16k/65k/262k l0_medium (3) = 13 | `resid_post_all` L10–L14 16k l0_big + l0_small (10); `resid_post` L13 16k/65k l0_medium, 65k l0_big, 262k l0_medium (4) = 14 |
| **Extended** (applied only to provisional targets, in a second pass; it can only remove targets) | `resid_post_all` L6–L8 262k l0_big + l0_small; `resid_post` L7 16k/65k/262k l0_big + l0_small | `resid_post_all` L11–L13 262k l0_big + l0_small; `resid_post` L13 16k/262k l0_big + l0_small, 65k l0_small |

Every extended file not already on disk is downloaded, used and deleted one at a time, to stay
under the 20 GB cap. Before each download the script asserts that the folder plus the file stays
≤ 20 GB, and stops otherwise. The primary set's two 262k l0_medium files are deleted before the
extended pass (they are not part of it).

**Excluded, with reasons:**
- the **1M width** (about 10 GB each), which exceeds the disk cap;
- **transcoders, attn_out and mlp_out SAEs, CLT and crosscoders**. These are not residual-stream
  dictionaries, and the question is coverage of a residual-stream direction.

**Masking.** Every dictionary's decoder rows are compared with the **unit's layer mask**
(keep = all coordinates except 1038; `e61.drop_for(7)` = `drop_for(12)` = [1038]) and then
renormalised. Unmasked cosines are reported too. SAE encoding always uses the unmasked float32
residual.

## Pre-computation steps (run and hashed before any latent number)

1. **Alignment gate.**
   - Capture D2 residuals with the census's own forward pass.
   - Rebuild the units' scores through the frozen chain: source-SAE encode, fp16(code) @ W_dec,
     e63 first-moment removal, `e61.pca_apply` (maps P), `e64.apply_maps`, then the e64 readout.
   - Require **R² ≥ 0.999 for the four calibration units** (L7 logcosh #12 and #37; L12 logcosh
     #13 and #33). Otherwise STOP.
2. **Opposite-tail signatures.** For every unit, build T_u⁻ with the frozen `e64.signature` code on
   the bottom 1% of the oriented score.
3. **Joint-null units.**
   - Draw 40 random unit vectors g in the rebuilt 64-d W space (seed 6830 + L).
   - Each one's score is y = W·g, its pattern d is pulled back exactly as e64 did, and its
     signatures T and T⁻ come from the frozen signature code.
4. **Self-match and signature testability**, per unit and joint-null unit:
   - s_u = the share of the unit's own 200 strongest non-final kept rows (by oriented score) whose
     next token is in T_u;
   - base_u = the same share over all non-final kept rows.
   - **The signature route is TESTABLE iff T_u is non-empty and s_u ≥ 2·base_u.** The same rule is
     applied separately to the opposite side (T_u⁻ with its own s and base), which is used when a
     latent's r is negative.
   - When a side is not testable, latents on that side are matched by tail overlap only
     (Measurement 4b).
   - **Every unit is testable by tail overlap**, so there is no INDETERMINATE class.
5. **Joint-null scores.** The scores of the 40 joint-null directions are saved
   (`results/e68_prep_G{L}_nullscores.npy`) and hashed with the prep file, so that an interrupted
   run can resume from the hashed prep.

**Hash-before-use.** These outputs are hashed before the first latent is encoded.

## Measurements per (unit, dictionary)

1. **Decoder cosine.** Signed cos(d_u, row) for every row. Report the top 5 by |cos|, and every row
   with |cos| ≥ 0.7.
   - **Subspace null:** max |cos| over the dictionary's rows, for 1,000 pullback-matched random
     directions: `e65.rand_bank(z['P_U'], z['P_sd'], keep, 1000, 6820 + L)`, one bank per layer.
     Report its p50 / p99 / p99.9 (`np.quantile`, linear).
2. **Activation correlation.** Pearson r between every latent's activation and the oriented y_u,
   over the 203,328 kept rows. Report the top 5 by |r| and every latent with |r| ≥ 0.7. The
   within-type r of the top latents (activation residualised on current token type) is reported,
   descriptive only.
3. **Sparse reconstruction.** Greedy OMP over unit-normalised masked rows: signed, and
   non-negative on +d_u.
   - Report the atoms needed to reach cos 0.8 and 0.9, with a cap of 48 atoms (32 for 262k).
   - **Null:** the same procedure, **per OMP mode** (signed and non-negative), for 100
     pullback-matched directions per dictionary (30 for 262k), with seed 6840 + L. Report the
     null's p1 / p5 / p50 for cos 0.8 and 0.9.
   - Each route's k08 is compared with **its own mode's** null p1.
   - The atoms of every OMP route are always tested for a match.
4. **Behavioural match.** A latent matches u if it passes **either** 4a (signature, when testable)
   **or** 4b (tail overlap). This leans toward COVERED.

   **4a. Signature match** (unit-relative; sign-aware; only on a testable side).
   - For latent ℓ, take its 200 strongest-activating D2 kept rows, excluding sequence-final rows.
   - **Signature:** use T_u if r(ℓ, oriented y_u) ≥ 0, and T_u⁻ (with its own s and base) if
     r < 0.
   - Let s_ℓ = the share of those rows whose next token is in that signature.
   - **ℓ matches u iff all of:**
     - s_ℓ ≥ base_u + 0.5·(s_u − base_u);
     - ℓ has ≥ 50 such rows, from ≥ 30 distinct sequences;
     - s_ℓ > the **p99 of a random-latent null.** That null is 500 latents drawn at random
       (seed 6850 + L) from all of the dictionary's latents with ≥ 50 active non-final rows,
       scored with the same signature. If fewer than 500 latents are eligible, all of them are
       used and the shortfall is recorded.
   - **Latents tested** for a given (u, D): the top 5 by |cos|, the top 5 by |r|, every latent
     with |cos| ≥ 0.7 or |r| ≥ 0.7, and the atoms used in each OMP route.
   - **Reported, descriptive only,** for every tested latent: the type-conditional ratio Σobs/Σexp,
     where exp comes from D2's current-type next-token table.

   **4b. Tail-overlap match** (signature-free: does ℓ fire where u fires?).
   - **Tail:** u's tail on the side given by the sign of r(ℓ, oriented y_u). That is the top 1% of
     the oriented score when r ≥ 0 (2,034 rows) and the bottom 1% when r < 0.
   - Let o_ℓ = the share of ℓ's 200 strongest non-final rows that fall in that tail. Its base is the
     tail's share of non-final rows, about 0.01; the unit's own top-200 have o = 1.
   - **ℓ matches by overlap iff all of:**
     - o_ℓ ≥ base + 0.5·(1 − base);
     - the same ≥ 50 rows / ≥ 30 sequences rule;
     - o_ℓ > the p99 of the same 500-random-latent null, scored for overlap.
   - **Within-type r** of each unit's top-5 |r| latents (both activation and score residualised on
     the current token type) is reported, descriptive only.

## Rubric

**Per unit u:**

| reading | requires |
|---|---|
| **COVERED** | Some tested latent, in any primary dictionary, has (decoder \|cos\| ≥ 0.7 **or** activation \|r\| ≥ 0.7) **and** matches u. This is e67's route, kept for comparability. |
| **PARTIALLY COVERED** | Not COVERED, but in some primary dictionary the atoms needed to reach **cos 0.8** (signed or non-negative OMP) are **≤ that dictionary's null p1**, **and** a majority of those atoms match u. |
| **NOT COVERED** | Neither of the above. |

- **"Aligned beyond chance"** (descriptive only): in **some** dictionary, the unit's max |cos|
  exceeds that dictionary's subspace-null p99.9. It is also reported without the source
  dictionary.
  - This replaces the draft's strict reading.
  - **It is expected to call L12 #13/#16 not aligned beyond chance**: e67's 0.735 sat below the
    L12 16k big p99.9 of 0.80.
- **Source dictionary.** Each reading is reported with and without the source dictionary. This
  covers unit verdicts, family not-covered, the calibration gate and the joint-null rates.
  Targets always use the with-source reading.

**Per family:**
- **NOT COVERED** iff every candidate member is NOT COVERED. A member that is COVERED or PARTIAL
  makes the family **not** NOT COVERED.
- The record lists **all** twins of every member, with each twin's role.
- **Targets** must satisfy all of:
  1. the family is NOT COVERED;
  2. it is R2-replicated: some member is "replicated geometric" in e64;
  3. no member is e64c-nameable: no e64c rule (transformer parse) reaches |AUROC| ≥ 0.65 on a
     member's frozen D2 score;
  4. no twin of any member is battery-NAMED, NAMED_BY_CONTEXT or a control;
  5. no member is identity-suspect or context-generic, which matches e65's qualification;
  6. it is not |cos| ≥ 0.8 with any named, control or COVERED unit at the same layer;
  7. it survives the **extended-dictionary second pass**, under the same rule.
- **Tiers among targets:**
  - **L12 tier A:** a member passed e65 B3, and did not fail B4 where B4 was run.
  - **L12 tier B:** e65 B3 failed.
  - **L7:** "no validated behavioural test at this layer", since the e65 anchor failed there.
  - **Only tier A is "worth a new round" directly.** Tier B and L7 need, respectively, an argument
    against their recorded failure and a new rig.

**Calibration.**
- **Gate set, derived by rule:** every control unit, plus every NAMED_BY_CONTEXT unit whose
  **best** e64c rule (transformer parse, max over the 16 rules, |AUROC| on kept rows) reaches
  ≥ 0.65. The script asserts that the derived set equals this list:
  - **L7:** logcosh #37, skew #37, logcosh #50, skew #38, logcosh #55, skew #49, plus controls
    logcosh #12 and skew #62;
  - **L12:** logcosh #60, skew #28, logcosh #24, skew #24, logcosh #44, skew #39, logcosh #10,
    skew #15, logcosh #9, skew #52, plus controls logcosh #11, skew #23, logcosh #33, skew #33.
- **Gate:** if more than 25% of a layer's gate set comes out NOT COVERED, **that layer nominates no
  target**, and the rule would need revising under a new hash.
- **Also reported** (not gates):
  - e67's family under this rule;
  - the **joint-null rate**: the share of the 40 random directions per layer that come out
    COVERED, or COVERED-or-PARTIAL. This is the baseline against which the units' coverage rate is
    read.
- **The rule is never swapped after the data are seen.** A calibration failure is reported and
  stops nomination. It does not change any verdict.

## Verification of targets (after the census)

- **Skeptics.** Three independent read-only skeptic agents per target, each trying to show the
  family is covered, nameable or artefactual. A majority is 2 of 3.
- **Contested targets.** A target that a majority shows to be covered or nameable is reported as
  **contested**. Contested targets do **not** go into a confirmatory round without the user's
  explicit approval.
- **Feasibility.** A round on any target needs fresh text: corpus sequences 2,000–12,999 (unused by
  occurrence-level work), or new FineWeb. An L7 target also needs a new rig. Feasibility is
  reported separately from the census verdict.

## Resources

- **Separate forwards.** L7 (layers 5–9) and L12 (layers 10–14) run as separate forwards. Each
  holds only the kept rows in float32, about 0.94 GB per layer.
- **No residual array is written to disk.** The single exception is the alignment gate's temporary
  fp16 decoded-residual cache (tag `occD2c{L}`, one per group), which is deleted once the gate
  passes.
- **The model is freed** after the capture.
- **Swap** is checked before every dictionary, in both the decoder and the acts steps.
- **Calibration set first.** The derived-equals-listed calibration assert also runs at the start of
  each group's run, before the model is loaded.
- **Resume.** If a group's run stops after its prep file is hashed, a re-run loads the hashed prep
  and the saved joint-null scores rather than recomputing them. It skips any dictionary already
  done, and appends the decoder and acts hashes only once every dictionary is present.
- **Folder size** is recorded after each dictionary, measured with `du`.

## Outputs

| item | file |
|---|---|
| script | `scripts/e68_coverage_census.py` |
| outputs (each hashed with the e64 helper when written) | `results/e68_prep_G{7,12}.json` (with `_nullscores.npy`), `results/e68_decoder_G{7,12}.json`, `results/e68_acts_G{7,12}.json`, `results/e68_census.json`, `results/e68_extended.json` |
| report | a REPORT.md section |
