# PROCESS.md: a repeatable process for discovering structure in Gemma models

**What this is.** A proposal, written 2026-10-17 from the e64–e72f record by a multi-agent design review (three competing designs, two judges, a synthesis and a replay critic). Nothing in it is adopted, hashed or run, and it amends no verdict.

**What it adds to current practice:**
1. A claim ladder (§2) with the controls each rung needs, and the meaning of "discovery" fixed in advance and approved by the user.
2. A buildable positive control for "uncovered by the dictionary": knock out the dictionary's own paren and quote latents (§3).
3. Pilots that already carry the same-subspace random baseline and a real-forward power estimate before any prereg (§4.2), plus a per-layer steering-rig gate (§4.4) and download-after-gate (§4.5).
4. A block ledger with a protected confirmation reserve (§5).
5. Templates, launches with status files and chained verdicts, and tooling in build order (§6–§8).

**To start the next cycle,** go to §9. §0 is the fact base the rest relies on. §11 replays every recorded failure through the process.

Status: process design, 2026-10-17. It authorises no run and amends no verdict. Numbers come from REPORT.md, SUMMARY.md, the hashed results and the logs. *(inferred)* marks an estimate or a threshold proposed here.

These files are proposed here and do not exist yet: `DISCOVERY.md`, `INSTRUMENTS.md`, `RESULTS.tsv`, `docs/RIG.json`, `docs/BLOCKS.jsonl`, `docs/SEEDS.jsonl`, `lib/`, `tools/`, `templates/`.

## 0. Record facts this document relies on (checked 2026-10-17)

- **e72b is finished and reported.** It reads INDETERMINATE, s = [2, 2, 1, 1, 0] against D2's 4 (`results/e72b_null.json`, HASHES line 140, 22:18:30; REPORT lines 2871–2951; SUMMARY pointer). The full e72 lead rule fires once on 4 of 5 shuffles, as it does on D2.
- **e72f is hashed but unreported.** It reads INCONCLUSIVE (`results/e72f_verdict.json`, HASHES line 127, 22:06:31). REPORT.md and SUMMARY.md have no section for it. REPORT line 2877 only says the e72b capture ran after the e72f steer exited. A draft card is in §6.4.
- **Downloads.**
  - e67 put about 8 GB of dictionaries into `hf_home` (REPORT line 1497). Its three 262k files (7.3 GB) were removed later.
  - e68 downloaded 7 primary files, 2 of them 262k l0_medium (119 s and 122 s, `logs/e68_download.log`). The folder peaked at 17.7 GB. The extended pass downloaded nothing (REPORT lines 1545–1546).
  - e72f read 262k rows with HTTP range requests and kept no file. It skipped the 262k arms because only 64 of 99 share-matched rows existed (`logs/e72f_freeze.log`).
- **The 20-latent knockout (KO) list is frozen only indirectly.** It is in `docs/sae_gap_calibration/design_panel_result.json` (sha `63da457b…`). That file is listed in `docs/sae_gap_calibration_MANIFEST.sha256`, and the manifest is HASHES line 68. The file's own sha is not in HASHES, so `e64.require_hashed` would refuse it. `PREREG_sae_gap.md` was never frozen and is not hashed.
- **Ledger helpers.** `e64.append_hash` appends without a lock. `e64.require_hashed` and `check_integrity` only test whether a sha occurs somewhere in HASHES, so a superseded sha still passes.
- **Dictionaries on disk** (Gemma Scope 2 snapshot `b738dc06…`):
  - `resid_post_all` 16k l0_big and l0_small at L5–L14, and 16k l0_big at L16, 18, 20, 22 and 24 (151,131,000 B each);
  - `resid_post` L13: 16k_l0_medium, 65k_l0_medium and 65k_l0_big (each 65k file is 604,509,056 B);
  - `resid_post` L7: 16k and 65k l0_medium;
  - the `resid_post` 262k_l0_medium folders at L7 and L13 hold only `config.json`. No 262k parameters are on disk.
- **Encoding gotcha.** X's FVU is 0.032 with coordinate 1038 included and 0.19 without it (design_panel_result.json; REPORT line 2778 gives 0.190 on kept coordinates). Zeroing 1038 before encoding raises L0 from 110 to 4,594. Always encode the full row and drop 1038 afterwards. `results/e67_raw_L12_rows.npy` is 203,328 × 1152 fp16 (468 MB) with 1038 stored. The L12 peak of 1038 is 63,232, under the fp16 maximum of 65,504.
- **The launcher and watchdog exist only in a session scratchpad:** `/private/tmp/claude-501/-Users-scasella-Downloads/f61c0116-241a-4f9f-b41a-d43e379b84ba/scratchpad/steer/` (`launch.py`, `watchdog.py`, `pids.txt`).
  - `launch.py` hard-codes the e72f steer phase.
  - `watchdog.py` reads machine-wide swap, so another project's growth can trigger the kill. It records no exit code.
- **Disk.** The folder is at 15 GB of its 20 GB cap.
  - `hf_home` is 7.4 GB.
  - `results` is 5.6 GB. It includes seven raw L12 fp16 row files of 467–468 MB each, about 3.3 GB in all: `e67_raw_L12_rows.npy`, `e69_p1_S2_raw_L12_rows.npy` and `e72b_S2b{1..5}_raw_L12_rows.npy`. It also includes `typemeans_pass1.npz` (1.31 GB).
  - `.venv` is 1.1 GB and `.venv-parse` 1.2 GB.
  - The volume has 35 GB free and is 92% used.
- **Seed collisions already exist** (§5.1).
  - e72f's RAND seed (7210 + 12 = 7222) equals its own R_big bank seed (7222).
  - e72f's site seed (7200 + 12 = 7212) equals e72b's shuffle-2 RAND seed (7210 + 2 = 7212).
  - The draws differ in kind, so no result changes, but nothing checks for collisions today.

## 1. Purpose and scope

A repeatable discovery process takes a candidate structure from a blind search to a fixed label on a fixed claim ladder. It uses admitted instruments, same-subspace baselines, burned-data power checks and one block ledger. Any session gets the same verdict from the same inputs.

Each cycle produces:
1. a hashed verdict JSON in the §6.2 schema;
2. a result card in REPORT.md and SUMMARY.md (§6.3);
3. ledger updates (HASHES, BLOCKS, SEEDS, INSTRUMENTS, RIG);
4. one label: DISCOVERY, or the highest rung reached with the control that stopped it;
5. the measured cost against the estimate.

Negative results are products. They use the same card.

**Scope.** gemma-3-1b-pt residual streams, later gemma-3-4b-pt. Gemma Scope 2 is the reference dictionary X. Everything runs on the local M4 Pro (24 GB, MPS, shared with other projects). Claims that the model "uses" a direction are out of scope.

## 2. The claim ladder

### 2.1 Rungs, in the order they run

The cheap CPU falsifiers, R5 and R4, run on the discovery block's rows before any behaviour block is claimed. A lead that reads COVERED never gets a behaviour block.

"Baseline space" means random directions drawn the same way as the target:
- **W-space** for raw-h targets. This is e72's G: 10,000 unit vectors in the 64-d equalised W space (seed 7200), scored through 64 × 64 second moments and never as n × N projections.
- **Pullback-matched** for decoded-span units. This is `e65.rand_bank`: 99 directions in the units' own 64-d subspace.

A pullback bank is never used for a raw-h target.

| Rung | Claim | Output words | Positive control | Negative control | Baseline (space) | Power / feasibility on burned data | Data |
|---|---|---|---|---|---|---|---|
| R0 Admitted and screened | the instrument works at (model, layer, representation); the unit passes the screens | QUALIFIED / NOT QUALIFIED; n screened | paren (tuned) and quote (untuned) reach the floor and beat the permutation p99.9 (about 0.52) | planted-source unit test (ICA dimension = input dimension; TDI excludes the row itself); shuffled text | token-matched permutation, 1,000 draws | floor at or below the pilot's supervised probe ceiling; planted recall at κ ∈ {1, 2, 4} | P-type burned block |
| R1 Reproducible | an independent refit finds the same unit | REPLICATED / NOT REPLICATED | controls replicate (e64: cos 0.94–0.99) | random pairs | "same unit" bar above the random-pair p99.9 *in that representation* (raw h: p99.9 0.824, max 0.875; e72's 0.8 bar sat below it) | split-half on the pilot block | second claimed block |
| R2 Context-specific | not identity, not generic | SPECIFIC / GENERIC / SELECTION EFFECT / INDETERMINATE | controls lose their signal on within-sequence shuffles | a planted type-only direction reads GENERIC | selection-matched shuffle null: the whole pipeline and the full lead rule re-run on streamed shuffles (§4.2) | shuffle count set by §4.2 | D block plus streamed shuffles |
| R5 Adds to X | X's thresholded codes miss the unit | ADDS-TO-X / COVERED / UNRESOLVED / GRADED PARTIAL COVER / NO VERDICT | KO ladder (§3) | controls against full X; firing-rate-matched random 10-latent ablations; the dummy cosine test must fail | W-space randoms; raw-h units only | §3.2 rules 1–8 | CPU on the D block's rows |
| R4 Unnamed | no frozen label fits | NAMED / UNNAMED-v`<battery sha>` / NAMEABLE POST HOC | paren and quote units come out NAMED | the UNNAMED rate among random directions, reported alongside | random-direction naming rate | none | CPU, same rows |
| R3a Potent, exploratory | steering moves next-token predictions | EXPLORATORY PASS / HELD-OUT PASS / FAIL / NO VERDICT | same-layer anchor first on each block; rig VALID (§4.4) | shuffled signature (SIG); sign consistency | 99 RAND at natural dose c·σ_d (c* = 2), in the target's space; a null holds null draws only | anchor margin and every precondition, with real forwards | two claimed B blocks |
| R6 Not a blend of X | X's best signed sparse reconstruction does not reproduce the behaviour | NOT A BLEND / COVERED-BLEND / UNRESOLVED / GEOMETRIC ONLY | T-BEH knockout-quote calibration on the L12 rig (§3.3) | constructed blends read COVERED-BLEND (sanity check only) | RAND max for d̂ and for d − d̂ | defined only where d's own Δ beats the RAND max | burned sites at a rig-VALID layer |
| R3b Confirmed | one primary test on a protected block | CONFIRMED / NOT CONFIRMED / NO VERDICT | anchor first on that block | SIG | 99 RAND; per-test false-pass bound ≤ 1/77 (e66) | power table from R3a | one protected block |

**Wording limits.**
- **R3 CONFIRMED** means that steering along d at natural dose moves next-token predictions more than random directions in its subspace do. It does not mean the model uses d. Report the per-context Δ distribution and the dormant-pathway split descriptively.
- **UNNAMED** always carries the battery version's sha.
- **ADDS-TO-X from a test admitted only at the graded level** (§3.2) is worded "adds to X's thresholded codes relative to a dedicated-latent knockout". "Uncovered" is never written without that qualifier.
- **"Not a blend"** is claimed only at R6.

### 2.2 What counts as a discovery (frozen before any further data)

`DISCOVERY.md` holds the definition below. The user approves it, and it is hashed through `e64.append_hash` before any new block is read. Any change, including to its dictionary list, needs a new hash and the user's approval.

> A **DISCOVERY** is a raw-h direction d at (model, layer) that:
> 1. comes from an instrument ADMITTED for that model, layer and representation;
> 2. is R1 REPLICATED on an independent block and R2 SPECIFIC under a selection-matched null;
> 3. is R5 ADDS-TO-X for every dictionary on the frozen list, under a coverage test ADMITTED (not ADMITTED-GRADED) by the §3 calibration;
> 4. is R4 UNNAMED under the hashed battery version;
> 5. is R3a HELD-OUT PASS on two exploratory blocks, and R6 NOT A BLEND against the source dictionary, at a rig-VALID layer;
> 6. is R3b CONFIRMED on one protected block, with one primary test and the anchor run first.

**Frozen dictionary list for 1B L12.** All four are cached: `resid_post_all` L12 16k l0_big, L12 16k l0_small, L11 16k l0_big and L13 16k l0_big. A 262k dictionary joins only through the download-after-gate rule (§4.5). DISCOVERY does not depend on a retrained SAE.

**Decoded-span units are ineligible for R5 and for DISCOVERY.** These are units extracted from SAE-decoded h, which includes every e63/e64 ICA unit. Each is a readout of the source dictionary's codes (F7).

**Labels for anything short of a discovery:**

| Label | Requires |
|---|---|
| ADDS-TO-X (GRADED) | items 1–6, but the coverage test is only ADMITTED-GRADED |
| SAE GAP (NAMED) | items 1–3, 5 and 6, but the unit is NAMED |
| CONFIRMED UNIFICATION | R3b CONFIRMED and R6 COVERED-BLEND |
| CONFIRMED, COVERED | R3b CONFIRMED and R5 COVERED |
| CONFIRMED, NOT ELIGIBLE | R3b CONFIRMED on a decoded-span unit; R5 is undefined |
| LEAD | ADDS-TO-X under an admitted test without R3, or HELD-OUT PASS without R3b |
| NULL at rung *n* | the highest rung reached, with the failing control named |

**L12 logcosh #13 / skew #16 under these rules** is CONFIRMED, NOT ELIGIBLE. e66 measured Δ 1.49 against a RAND max of 0.75 and a SIG max of 0.85, under e65's post hoc tier.
- e67's "COVERED as a blend" came from an unadmitted rubric whose cosine route sat inside the same-subspace null.
- It is not relabelled, and it is not ground truth for any gate.
- It may illustrate "blend-like" only under e71's sparsity readout, the one direction in which that readout calibrates.

## 3. The positive control for "uncovered", and the negative controls

### 3.1 Construction: the knockout (KO) ladder

**No retraining is needed.** `src/gfdisc/saes.py` encodes each latent independently (`pre = x.float() @ w_enc + b_enc; z = pre * (pre > threshold)`). Removing a set K therefore leaves every other code unchanged. The ablated dictionary X∖K has error

> e_K = e + Σ_{k∈K} z_k · W_dec[k]

and needs no refit. This is the C1 knockout object of the e69/e70 design panel (`docs/sae_gap_calibration/err_ko.py`).

**Frozen latent sets.** They are in L12 `resid_post_all` 16k l0_big, ranked on D2 by firing-rate enrichment (design_panel_result.json). K carries 4.6% of the error's variance.
- paren: 2682, 7780, 4001, 4493, 1559, 8951, 3503, 1442, 507, 1056
- quote: 5254, 1229, 1994, 1534, 564, 1882, 7302, 10306, 398, 1550

**The ladder:**
- k ∈ {0, 1, 3, 10};
- then 10 + near-duplicates: rows whose decoder |cos| to a K member, or whose activation correlation with the label, exceeds the p99.9 of a random-row null (never a fixed 0.7, F4);
- then m ∈ {30, 100, 256}, extending the ranking by the same enrichment rule computed on even D2 sequences and scored on odd ones.

**Each further dictionary on the list gets its own ladder,** built from its own top-enriched latents with the same frozen rule.

**Targets.** The targets are raw-h control units from e72, never decoded-span units:
- paren skew #45 (|AUROC| 0.749);
- quote logcosh #29 (0.762) and quote skew #29 (0.755).

**Coverage classes** (bars frozen in the NOTE, values *(inferred)*), scored against a held-out W-space bank:
- ADDS: above the bank's max;
- COVERED: at or below the bank's median;
- UNRESOLVED: in between.

**Tests to calibrate:**
- **T-ES:** the error share of d, from `e72.error_W`, against 10,000 W-space randoms through 64 × 64 second moments. At k = 0 this is exactly e72's ES.
- **T-MG:** marginal gain, in the sense of Leask et al. 2025 stitching. Append d as one JumpReLU latent, fit its threshold on even sequences, and measure the drop in held-out error variance on odd ones. Compare with at least 200 W-space randoms built the same way, in batches, never materialised as n × N arrays.
- **T-COS (dummy):** max |cos(d, removed decoder rows)|. It must fail admission (§3.2).

**Burned evidence that the mechanism moves information.**
- On D2, the controls' T-ES at k = 0 is:
  - quote logcosh #29: 0.274 (p0.4);
  - quote skew #29: 0.276 (p0.6);
  - paren skew #45: 0.295 (p13.5).
  
  The random median is 0.309 and the random max 0.359.
- In e_K at α = 1, the design panel's pipeline recovers paren at 0.781 and quote at 0.747. Probes on the unablated error top out at 0.57/0.58.

### 3.2 Validation rules (frozen in the calibration NOTE before any number)

**The control order is frozen:** paren first, then quote, and both must pass. There is no fallback to another control. If either fails, the test is NOT ADMITTED.

1. **Covered at k = 0.** Both controls read COVERED (at or below the bank median). If a test cannot call the controls covered, the test fails, not the control. T-ES passes this on existing numbers (p0.4, p0.6, p13.5).
2. **Monotone.** The score never decreases as k grows. The control reads ADDS at 10 + near-duplicates, or at a recorded step m on the frozen ladder.
3. **Cross-specific.** Removing the quote set does not uncover paren, and the reverse holds too.
4. **Matched random removal.** Over 20 random 10-latent ablations matched on firing rate, paren reads ADDS at most once.
5. **Random-direction stability.** Under each ablated dictionary, at most 2 of 40 directions from a second, held-out W-space bank change class (e71's cap). This bank is not the one that set the class bars.
6. **Sub-threshold leak, absolute bar.** After removal, fit a held-out probe of paren state on X∖K's activations plus clip(pre, 0, θ) of the removed latents and of the top-32 remaining latents.
   - At the k where the test reads ADDS, the probe's AUROC must lie inside the shuffled-label band (at or below its p99).
   - If it lies above the band, the result at that k is GRADED PARTIAL COVER.
   - Today the pre-activations of #2682 and #4001 still separate paren among inactive rows (AUROC 0.67 and 0.72), and 70–90% of paren rows have the top latent inactive.
7. **Planted dense variable (MDE).** Plant C2′, a two-state Markov variable (rate 0.06, mean run 16; e70's constants), in raw h along a random direction u at κ ∈ {1, 2, 4}·σ. Re-encode with full X.
   - The smallest κ at which the test reads ADDS is its minimum detectable effect (MDE).
   - The test must detect C2′ at κ ≤ 4 *(inferred bar)*.
   - Any target's margin must exceed the MDE.
8. **Lost-information ground truth**, independent of every test statistic.
   - A held-out linear probe for paren on X∖K's identity-removed reconstruction must fall inside the shuffled-label band, while the raw-h probe keeps its level.
   - m\* is the smallest ladder step where this holds. The test must read ADDS at m\* and COVERED wherever the reconstruction probe sits above the band.
   - If no m ≤ 256 reaches the band, rule 8 reports "ground truth not reached". Given rule 6's numbers, this outcome is likely *(inferred)*.

**Admission levels:**
- **ADMITTED:** rules 1–8 all pass.
- **ADMITTED-GRADED:** rules 1–7 pass and rule 8 reports "ground truth not reached". Every ADDS such a test emits carries the limited wording of §2.1, and it cannot satisfy DISCOVERY item 3.
- **NOT ADMITTED:** anything else.

**Fallback, stated in advance.**
- If rule 8 is not reached, the path continues as ADMITTED-GRADED with limited wording. The best possible label is then ADDS-TO-X (GRADED).
- If rules 1–7 fail on paren or quote, R5 and R6 close for that dictionary family, and a methods card is written. The 4B scale-up stays deferred.
- No retrained SAE is substituted.

**The gate cannot be passed on geometry alone.**
- T-COS passes rules 2 and 3 by construction.
- It fails rule 7: the planted u is random, so its cosine to K is near zero, T-COS never reads it ADDS, and the MDE is undefined.
- It also disagrees with rule 8 wherever siblings absorb paren.

The calibration runs T-COS through every rule and records it NOT ADMITTED. If T-COS is admitted, the rule set is broken, and no test is admitted.

### 3.3 Other controls and their limits

**Width ladder at L13 `resid_post`.** It compares 16k_l0_medium with 65k medium and big. All three are cached, so no download is needed.
- Novel 65k latents, in the stitching sense, must read ADDS against the 16k. Reconstruction latents must read COVERED.
- For T-MG this is an agreement check only, because stitching uses the same gain idea. For T-ES it is a partly independent positive *(inferred)*.
- Its positives are sparse latents, which are easier than an ICA unit.
- It needs one raw L13 capture of D2 (103–198 s of MPS at e60's rate), stored in fp32 or with a coordinate-1038 sidecar. L13's 1038 peak is unmeasured. It is 63,232 at L12 and 67,072 at L14.

**T-BEH for R6** (L12 only).
- Steer with X's signed sparse reconstruction of quote logcosh #29, then with X∖K_quote's.
- Under X the reconstruction must recover the anchor's Δ. Under X∖K_quote it must not.
- Raw-h steering must first pass §4.4 with a raw-h anchor.
- Cost: about 200 passes, about 27 min of MPS on burned sites *(inferred, at about 8 s per 200-prompt pass, PREREG_fragmentation §10)*.

**Constructed blends** are sanity checks only. They are signed sums of 2–3 decoder rows that co-fire on burned data. They must read COVERED-BLEND, but they never set thresholds, because OMP recovers them exactly.

**A data-ablated SAE retrain is an optional hard positive.**
- No SAE training code exists in `scripts/` or `src/`.
- Before use, its training cost on about 1M burned rows must be measured under the watchdog.
- The retrained X′ is not Gemma Scope X. X's FVU (0.19 on kept coordinates) and its thresholds do not transfer.
- DISCOVERY does not depend on it.

### 3.4 Cost and stop rule

**Cost of the calibration.** It is CPU only.
- Inputs on disk: `results/e67_raw_L12_rows.npy` and the two L12 16k files (151 MB each).
- A base encode of D2 takes about 6 s per dictionary (PREREG_fragmentation §10).
- e72's gate plus coverage took 175 s (109 + 66 s) with swap at 2.2–2.5 GB.
- The whole ladder takes 30–60 min of CPU, under 4 GB RSS, and writes under 0.1 GB *(inferred)*. That covers T-ES, T-MG, T-COS, the leak probes, the ground-truth probes and every null.
- No MPS, no download, no fresh text.

**L11 and L13 dictionaries** need raw D2 rows at those layers. One capture covers both: about 3 min of MPS, about 1.4 GB with L13 in fp32 *(inferred)*. Run it only after L12 admits a test (§9, step 3).

**Stop rule.** If no test is admitted, R5 and R6 close at 1B for that dictionary family, and a methods card is written. The 4B scale-up stays deferred. This calibration, together with T-BEH, is the written "not a blend" criterion that SUMMARY §7 asks for.

## 4. Stages, gates and stop rules

### 4.1 Stage table

| Stage | Data | Gate to leave | Stop rule |
|---|---|---|---|
| 0 Preflight | none | every §7.1 item passes | any failure: no launch |
| A Admission | burned only | INSTRUMENTS.md card ADMITTED for the claim type; RIG.json VALID for any layer to be steered; coverage card per dictionary | a missing card means no claim of that type at that layer; at most 2 redesigns on the pilot block, then retire |
| B Pilot (NOTE hashed before any pilot number) | burned only | every §4.2 field present; multiplicity, bar-design and power checks pass | void bar or power < 0.8: no prereg; methods card |
| C Freeze | burned dry run with the **real** forward | `tools/freeze.py` accepts; blocks claimed; imports pinned | a precondition fails in the dry run: no block claimed |
| D Discovery run (R0–R2) | claimed D and R blocks; streamed shuffles | anchor first; R0–R2 | anchor fails: no candidate number is read |
| E Cheap falsifiers (R5, R4) | CPU on D-block rows | R5 ADDS-TO-X for every listed X; R4 recorded | COVERED by any listed X: stop and report the rung |
| F Exploratory behaviour (R3a, R6) | two claimed B blocks; burned sites for R6 | rig VALID; passes both blocks; R6 NOT A BLEND | fails either block: no protected block |
| G Confirmation (R3b) | one protected block | one primary test; anchor first | anchor fails: NO VERDICT, and the block is still spent |
| H Report | none | result card written; `report_sync` clean | none |

Protected blocks are claimed only for DISCOVERY-eligible leads that passed stages E and F. A decoded-span unit may be confirmed for methods purposes only on an exploratory block, and only with the user's approval.

### 4.2 Pilot before prereg

The pilot produces the baseline and the power estimate. They are not left to a later review. `tools/freeze.py` refuses a prereg whose pilot JSON lacks any of these fields:
1. candidate values;
2. for every bar, the random pass rate and the random max in the correct space (at least 100 draws; e72 used 10,000);
3. the multiplicity record below;
4. a power table:
   - planted recall at κ ∈ {1, 2, 4};
   - for behavioural tests, the anchor, the RAND bank, every comparison bank and every precondition, run with real forwards on burned sites. For an e72f-type test this is about 200 passes, about 27 min *(inferred)*.
5. feasibility: every floor at or below the pilot's supervised ceiling;
6. measured wall clock and peak memory from a real-shape dry run under the watchdog.

**Multiplicity rule.**
- **Per-unit bar.** Use the max of N same-subspace randoms, with N ≥ 20 × n_screened. Under exchangeability the expected number of false passes is at most n_screened / (N + 1); e72 reports 0.013.
- **Selection breaks exchangeability.** e72 had 7 units above the random max against 0.013 expected, and e72b's shuffles still produced 2, 2, 1, 1, 0 screened units above their own max.
- **Count under the full selection rule.** The pipeline's GO statistic is the number of units that pass the *whole* rule that produces a lead: screened, not identity-suspect, norm/L0 and position screens, both parity halves, and above the max. A partial count is not enough. e72b's partial count s (4 against 2, 2, 1, 1, 0) looked favourable only because D2's excess was identity-suspect units. Under the full rule ℓ, D2's 1 sat against 1, 1, 1, 1, 0.
- **GO requires the real count to exceed every one of k = 19 selection-matched shuffle counts** (rank p ≤ 0.05).
- **Sequential stop.** Run shuffles one at a time. Stop with STOP at the first shuffle whose count reaches the real count. The full 19 are spent only while the real count is winning.
- **A bar is void** if n_screened × (random or selection-matched pass rate) > 0.5 *(inferred)*.
  - e69's shuffle bar, which 71% of random filters pass, is void at any n.
  - e72's lead rule fires on 4 of 5 shuffles (rate 0.8), so it is void as a GO rule.
- **Single primary tests** use the RAND max of 99, with a bound of at most 1/77.

**Bar-design rule.** Compare like with like. A precondition never compares a bank median with a max-of-99 null. e72f's RAND S had a median of −0.02, a p95 of 0.94 and a max of 2.21, with two of 99 draws above 2.2. On the review's post hoc reading of `results/e72f_steer.json` (unhashed analysis, not a verdict), its bank-median precondition (1.29 against 2.21) therefore failed on a tail property of RAND, not on block F. Use matched quantiles, or check power for the max on burned data first.

**Shuffle streaming.** Each shuffle is captured, analysed and deleted before the next.
- Per raw L12 shuffle: about 95 s of MPS, about 40 s of CPU, and 0.47 GB of transient disk. e72b took 472 s to capture 5 shuffles and 221 s to analyse 6 sets, S2 included.
- Twenty stored shuffles would need about 9.4 GB against about 5 GB of headroom, so storing them is forbidden.

**Power gate.** P(anchor and every precondition pass) must be at least 0.8, by split-block bootstrap on burned data *(inferred threshold)*. The freeze dry run must use the real forward for any effect-size precondition. e72f's dry run used a fake forward (PREREG_fragmentation.md line 137).

### 4.3 Current instrument status (1B)

| Instrument | Status | Reason |
|---|---|---|
| occurrence v2 ICA-64 on decoded h (e63) | ADMITTED R0–R2 at L7 and L12; REJECTED at L24 | paren 0.779 / 0.736 and quote 0.747 against p99.9 about 0.52; decoded span, so R5-ineligible |
| raw-h v2 plus error share (e72) | PROBATION | controls pass on raw h (paren skew #45 0.749, quote 0.762 / 0.755) |
| e72 lead rule as a GO rule | VOID | fires on 4 of 5 selection-matched shuffles (e72b) |
| AMUSE on SAE error (e69) | REJECTED as a screen | 71% of random filters pass its shuffle bar |
| sibling-max steering null (e64) | REJECTED | siblings are real effects (subject-NP Δ 1.52) |
| steering rig (e65/e66) | VALID at L12 only | §4.4 |
| single-latent coverage (e67/e68) | REJECTED | 3 of 4 L12 controls NOT COVERED; 6 of 40 random L7 directions COVERED |
| sparsity readout (e71) | descriptive, "blend-like" only | paren controls read chance-like |
| fragmentation banks (e72f) | PROBATION | bank-median precondition failed (bar design and power) |
| type-mean rings (S1), rulers and lattices (e46) | REJECTED at depth / not qualified | strict-12 fails at all 7 deep layers; kinship lattice invisible; number words confounded with frequency (ρ −1.00) |

### 4.4 Per-layer rig gate

**`validate_layer(L)`.** Steer the layer's anchor at natural dose c·σ_d (c* = 2) against 99 RAND and SIG on two disjoint burned site sets of 200 sites each. Both sets must pass e65's rule:
- Δ above the RAND max;
- Δ above the SIG max;
- sign consistency ≥ 0.8 at some c ≤ 2;
- plus Δ / RAND max ≥ 1.5 *(inferred margin)*.

The cost is about 104 passes, about 14 min per site set (PREREG_fragmentation §10). Each result is one line in `docs/RIG.json`: layer, anchor, block, Δ, RAND max, SIG max, sign consistency, dose ratio and script sha. Re-validate whenever the forward function, dose rule, prompt builder or block source changes.

**L12 is VALID.** Quote logcosh #33 passed on four blocks:

| block | Δ | RAND max | ratio |
|---|---|---|---|
| B3 | 0.856 | 0.417 | 2.05 |
| B4 | 0.873 | 0.560 | 1.56 |
| e66 C | 0.84 | 0.48 | 1.75 |
| e72f F (SIG max 0.541) | 0.856 | 0.461 | 1.86 |

**L7 is INVALID.**
- e64 (ε dose): Δ 0.012 against a full-space random max of 0.022 and a sibling max of 0.150.
- e65 B3 (natural dose): Δ 0.017 against RAND 0.185 and SIG 0.085. L7 skew #62 also failed (0.052 against 0.239).
- The "enclosure/aside" direction (REPORT line 839) *is* this failed anchor, logcosh #12.
- No other L7 variable was recovered blind: L7 quote reached 0.621 / 0.654, below the 0.70 floor.

**Bounded L7 search.** The search needs a new, user-approved anchor candidate, for example a supervised L7 quote probe direction (held-out probe 0.723) *(inferred choice)*. It gets at most 2 candidates, at about 28 min each (two site sets) on burned sites. Otherwise L7 is GEOMETRIC ONLY now: candidates there stop at R2, and R5 is reported as geometric only.

**Raw-h directions** need their own anchor. Validate raw quote logcosh #29 (|cos| 0.975 to decoded #33) with a W-space RAND bank mapped through the raw pattern pullback *(inferred design)* before steering any raw-h direction.

### 4.5 Download after gate

A download phase runs only when all of these hold:
1. the gate verdict it serves is hashed and reads PASS;
2. folder size plus the download stays at or below 17 GB *(inferred launch limit)*;
3. the user has approved it.

The gate always runs alone first, on cached 16k dictionaries. NO VERDICT means nothing is downloaded. Prefer HTTP range reads of the needed rows, as e72f did (about 30 MB, nothing cached). 262k files (2.42 GB each) are deleted after use, and the pinned revision `b738dc06…` allows re-fetching.

### 4.6 Standing stop rules

- **A frozen test phase on a claimed block is never re-run.** INCONCLUSIVE, NO VERDICT, aborted and watchdog-killed runs all spend the block.
- **A capture of already-burned text is resumable.** It may resume after a crash or kill. It skips hashed outputs and runs a drift check before reuse; e72b's re-capture of 32 D2 sequences matched bit for bit. e69 P1's resume after the app crash is the precedent.
- A failed lead never gets a second protected block.
- At least 2 protected blocks always stay unspent. Going below that floor needs the user.
- The user approves every MPS phase on fresh text, every fresh block, every download, every HASHES append outside a frozen phase, any use of the word "discovery", and the 4B scale-up.

## 5. The data and block ledger

### 5.1 Format

`docs/BLOCKS.jsonl` is append-only, with one record per state change. It is written only through `lib/blocks.claim()` under the ledger lock. The file's sha goes to HASHES after each change.

```
{"block":"corpus:9000-9999","state":"PROTECTED","rung":"R3b","doc_sha":null,"exposure":["pass1 type means (whole corpus)","unigram counts","original gf-discovery per-position cache rows"],"per_occurrence_reads":[],"seeds":null,"session":null,"time":"2026-10-17T00:00:00"}
{"block":"corpus:2000-2999","state":"CLAIMED","rung":"R0-R2 discovery D3","doc_sha":"<prereg sha>","seeds":[7430,7449],"session":"<id>","time":"..."}
{"block":"corpus:2000-2999","state":"SPENT","rung":"R0-R2 discovery D3","verdict_sha":"<verdict sha>","time":"..."}
```

**States:** UNTOUCHED → EARMARKED(rung) → CLAIMED(doc sha) → SPENT(verdict sha). Protected blocks start as PROTECTED. Any other read sends a block to BURNED.

**`lib/blocks.read(block, doc_sha, purpose)`** returns tokens in only two cases:
- the latest record is CLAIMED under that doc sha;
- the block is BURNED or SPENT, and the purpose is pilot, power, rig, description, admission or shuffle.

The duplicate screen (first 32 tokens) runs against every used set. Add corpus 8,000–8,999 to it now: e72f's `USED_CORPUS` lists only (0, 2000) and (13000, 15000).

**`docs/SEEDS.jsonl`** registers realised seeds (base + L where the code adds the layer) and refuses overlaps. Back-fill it with:

| experiment | seeds |
|---|---|
| e60 shuffles | 600,000 + seq_start + i (S2: 601,000–601,999) |
| e64 | 6400 + L, 6462 + L |
| e65 | sites 6500 + L (B3), 6600 + L (B4); RAND 6510 + L, 6610 + L; SIG order uses 6400 + L |
| e66 | sites 6700 + L (C), 6800 + L (D); RAND 6710 + L |
| e67 | 6720 |
| e68 | 6820, 6830, 6840, 6850 (each + L) |
| e69 | 6900 |
| e70 (never run) | 7020, 7030, 7031, 7040, 9000 + k |
| e72 | 7200 (RAND), 7201 (S2) |
| e72_verify | 7300 |
| e72f | sites 7200 + 12 = 7212; RAND 7210 + 12 = 7222; banks 7220–7225; 262k pool 7230; references 7240 + j; filler 7250 |
| e72b | shuffles 700,000 + 10,000·k + 1,000 + i (711,000–751,999); RAND 7210 + k (7211–7215), and 7201 for S2 as shuffle 0 |

The next experiment starts at 7400 *(inferred; no ext script uses 74xx)*.

### 5.2 Current state and earmarks

| Block | State | Use |
|---|---|---|
| corpus 0–999 (v1 D and S) | BURNED | admission, pilots |
| corpus 1,000–1,999 (D2; S2, S2b1–5, occD2x) | BURNED, the main pilot block | calibration, pilots, power, description |
| corpus 8,000–8,999 (F) | SPENT (e72f INCONCLUSIVE) | burned uses only |
| corpus 13,000–13,999 (P2), 14,000–14,999 (P) | BURNED | admission, rig validation |
| reserve 0–7,999 | BURNED (see note) | rig validation, power, description |
| corpus 2,000–2,999 / 3,000–3,999 | EARMARKED: D3 discovery / R1 replication | stage D |
| corpus 4,000–4,999 / 5,000–5,999 | EARMARKED: B5 / B6 | stage F |
| corpus 6,000–7,999 | spare: one new line of inquiry | user decides |
| **corpus 9,000–12,999** | **PROTECTED, C3–C6** | R3b only |
| corpus 15,000–15,695 (696 sequences) | reserved | blind planted round (§10) |

**Notes:**
- **Reserve 0–1,999 is unresolved.** PREREG_fragmentation §3 counts it as used. PREREG_occurrence_v2_discovery §0 (line 57) and PREREG_occurrence_v2_steering §0.5 (lines 186–187) say 0–999 was extracted and deleted, never analysed, and that 1,000–1,999 was never used. The ledger marks both BURNED, the conservative choice.
- **e70's earmark of corpus 2,000–6,999 is void.** e70 never froze and read no text.
- **Exposure caveat.** Several earlier reads touched every block:
  - the whole-corpus type means and unigram counts;
  - Stage 3's 8-token slices of all 8,000 reserve sequences;
  - the original gf-discovery scripts that took per-position cache rows across all 15,696 sequences (37b, 38b, 39, 41b, 47, 49; PREREG_fragmentation §3).
  
  So "untouched" means untouched by per-occurrence work only, and every prereg must say so.

### 5.3 Spending rules

1. Claim before reading. Run the duplicate screen against every used set.
2. Description always runs on burned blocks. e66 spent reserve 7,000–7,999 on description that D2 could have served *(inferred)*.
3. A protected block gets one primary test, and only for a DISCOVERY-eligible lead that passed stages E and F.
4. Keep at least 2 protected blocks.
5. Blocks spent at 1B count as BURNED at 4B, because the tokenizer and text are shared. Protected blocks stay protected for every model.
6. More fresh text means tokenising further FineWeb sample-10BT sequences with the S0 rule. That is a download and needs the user's approval *(inferred option)*.

## 6. Templates and the standard output

### 6.1 PREREG skeleton

★ marks a section that was missing somewhere in the record where it later mattered. `tools/freeze.py` refuses a prereg with any section missing or empty.

0. Banner: the question and rung; FROZEN; "authorises no run"; what the script asserts; "amends nothing"; the DISCOVERY.md sha.
1. What was seen before freezing, including earlier use of each block.
2. Claimed rung(s) and their output words (§2.1).
3. ★ Identification assumptions, and what each conclusion becomes if one fails.
4. Frozen inputs as a sha table, ★ including imported modules (e70 review).
5. ★ Block and seed claims (BLOCKS and SEEDS line references), the duplicate screen and the exposure statement.
6. ★ The instrument admission card sha(s), the rig card for each steered layer, and the coverage card sha.
7. The positive control or anchor, run first. ★ A negative control, or the sentence "none exists, so the claim stops at rung *n*".
8. Nulls:
   - ★ a composition statement (null draws only; no siblings, no controls);
   - the space (W or pullback);
   - ★ the pilot's random pass rate per bar;
   - ★ the selection-matched null under the full lead rule, with its sequential stop.
9. ★ Power and feasibility table (pilot sha), from real forwards, showing floor ≤ ceiling and like-for-like precondition quantiles.
10. Decision rule: classes, a first-match verdict table, multiplicity, the number of looks.
11. Outcome wording fixed in advance; the descriptive-only list.
12. ★ Stop rules.
13. ★ Ops plan per phase: memory, disk, wall clock, launch command, status file, the chained verdict, resumable or not, and which hash gates any download.
14. Open choices with defaults.
15. ★ Review log, kept in the file.
16. The result card, to be filled in.

**NOTE skeleton** (burned-data pilots, rubrics, admissions): sections 0, 1, 4–9, 11, 12 and 13, plus a GO/STOP rule.

**Admission card** (one entry in INSTRUMENTS.md, which is hashed) lists:
- instrument, code sha, model, layer, representation;
- the claim types it may emit;
- the plant, with the MDE at κ;
- the anchor result;
- the null: space, n, pass rate;
- power;
- cost;
- status: ADMITTED / ADMITTED-GRADED / PROBATION / REJECTED;
- the admission NOTE's sha.

A change of code sha, layer, model, dictionary or representation triggers re-admission. `tools/admit.py` regenerates the card from the NOTE and its outputs, so re-admission stays cheap.

### 6.2 Verdict JSON schema

```
{"experiment":"e73","rung":"R5","class":"ADMITTED-GRADED","label":"...","headline":"...","wording":"...",
 "controls":{"pos":[{"name":"paren skew #45","value":..,"null":..,"pass":true}],"neg":[...]},
 "baseline":{"space":"W","n":10000,"random_pass_rate":..,"random_max":..},
 "selection":{"rule":"full lead rule","k_run":..,"stopped_early":..,"count_real":..,"counts_shuffles":[..]},
 "power":{"pilot_sha":"..","preconditions":{"name":{"value":..,"margin":..}}},
 "blocks":[{"block":"corpus:1000-1999","before":"BURNED","after":"BURNED"}],"seeds":[..],
 "cards":{"instrument":"..","rig":"..","coverage":".."},
 "cost":{"wall_s":..,"mps_s":..,"cpu_s":..,"peak_swap_gb":..,"disk_delta_gb":..,"exit_code":0},
 "amends":"nothing","discovery_md_sha":".."}
```

### 6.3 Result card

Each card is appended to REPORT.md and SUMMARY.md in the session that produced the verdict, and indexed in `RESULTS.tsv`. It contains:
- the headline, "**`<id>`, `<rung>`: `<CLASS>`.**";
- a controls table, each control's number against its null;
- the primary number against its same-subspace null, with the random pass rate;
- what the result rules out, and what it does not show;
- blocks spent;
- estimated against actual cost;
- the next decision for the user.

A STOP uses the same card.

### 6.4 The missing e72f card (draft for the commissioning session to paste)

**e72f, fragmentation test of the e66 direction (R6 side branch): INCONCLUSIVE.**
- Block F is corpus 8,000–8,999.
- Prereg `PREREG_fragmentation.md` rev 3 (`ca16541a…`); script `825ac723…`.
- Outputs: `results/e72f_steer.json` (`df290b6d…`) and `results/e72f_verdict.json` (`ce849745…`).
- Amends nothing.

| Item | Result |
|---|---|
| Quote anchor (e65 rule) | PASS: Δ 0.856, RAND max 0.461, SIG max 0.541 |
| D, the e66 direction itself (e65 rule) | PASS on block F, its fourth behavioural block |
| T1 (d moves both classes) | PASS |
| Planted tilt control | PC PASS: both arms pass at f = 1.0 and f = 0.5; at f = 0.25 the md arm fails and the wu arm passes |
| L_big (the gated row) | INCONCLUSIVE: the bank median is not potent in both classes (S 1.29 against RAND max 2.21, corrected readout; 1.27 against 2.60, raw) |
| x320 (agreement check) | INCONCLUSIVE; agrees with L_big |
| R_big | TWO-SIDED on the corrected readout, INCONCLUSIVE on the raw readout, so INCONCLUSIVE (the readouts must agree) |
| 262k | skipped at freeze: only 64 of 99 share-matched rows |
| D's own S shift | 2.79, which is 1.26× the S RAND max |

- **Why it failed (post hoc reading, not part of the frozen verdict).** The RAND S distribution is heavy-tailed: median −0.02, p95 0.94, max 2.21, with two of 99 draws above 2.2. The precondition compared a bank median with a max-of-99.
- **Rules out:** nothing about fragmentation. The κ-matched bank lacked power under this bar on block F.
- **Block spent:** corpus 8,000–8,999.
- **Cost:**
  - steer 3,672 s of MPS (the prereg planned 548 passes, about 73 min);
  - freeze 81 s of CPU;
  - peak machine swap 7.36 GB;
  - about 5 MB written.
- **The verdict waited 3 h.** The steer exited at 19:04 and the verdict ran at 22:06.
- **Process lessons:** F8 (power on burned data with real forwards), the §4.2 bar-design rule, and the chained verdict (§7.3).

## 7. Ops runbook

### 7.1 Preflight checklist (`tools/preflight.py`, run by the launcher)

- **Environment.** `lib/config.py` sets HF_HOME=hf_home and HF_HUB_OFFLINE=1; the launcher's shell does not. The model and SAE snapshot paths and shas are pinned, replacing the pasted `gm.MODEL_NAME` lines.
- **Inputs.** Every input is hashed (`require_hashed`, with active-sha semantics), and the script's sha appears in the prereg or NOTE.
- **Outputs.** All outputs are absent, unless the phase is declared resumable (§4.6).
- **Swap.** Swap is under 8 GB at launch *(inferred)*.
- **Disk.** Free disk is at least 8 GB, and folder size plus declared output is at most 17 GB *(inferred)*.
- **Lock.** The machine-wide MPS lock is free. It lives at a fixed path outside the project, for example `~/Downloads/.mps.lock` *(proposal)*.
- **Process scan with `ps`.** Refuse if:
  - another e6x/e7x job is running;
  - any python process has more than 4 GB RSS;
  - any other MPS job is running.
  
  The refusal message names other projects' jobs for the user. A lock binds only sessions that use the launcher.
- **Download phases.** The gate's PASS sha must be in HASHES, and the user's approval must be recorded.

### 7.2 Budgets (measured unless marked)

| Step | MPS | CPU | Peak swap seen | Disk |
|---|---|---|---|---|
| e60 capture per 1,000-sequence set | 103–198 s | none | not logged | 0.59 GB per layer (decoded fp16); 0.47 GB per raw L12 kept-row set |
| R0 v2 control | none | about 90 s per layer (4.5 min for 3); pilot 9 min | not logged | about 97 MB of units per layer |
| raw-h pilot (e72) | none | gate 109 s, coverage 66 s, verify 101 s | 2.2–2.5 GB | small |
| R2 shuffles (e72b, 5) | 472 s | 221 s (6 sets) | 0.31 GB | 0.47 GB per shuffle; stream from now on |
| rig validation per site set | about 14 min (104 passes) | none | at most 4.9 GB (e66) | about 0 |
| R3 exploratory / confirm | 1,562 s (e65 B3 L12) / 987 s (e66) | none | at most 4.9 GB | about 0 |
| R5 KO calibration | none | 30–60 min *(inferred)* | under 4 GB *(inferred)* | under 0.1 GB |
| R6 fragmentation (e72f) | 3,672 s; power run about 27 min *(inferred)* | freeze 81 s | 7.36 GB | about 5 MB |
| dictionaries | none | none | none | 16k 151 MB; 65k 605 MB; 262k 2.42 GB |

Each MPS phase is capped at 75 min per launch *(inferred)*. Longer work is split into resumable phases.

### 7.3 Launch, status and completion

**`tools/launch.py <phase>`** runs preflight and takes the lock. It then starts the job under `caffeinate -i` with `start_new_session`, and starts the watchdog.

**`logs/<run>.status.json`** records:
- run, phase, owning session, pid, pgid;
- script sha and prereg sha;
- cmd and env;
- start and expected_s;
- a heartbeat every 60 s;
- end, **exit_code** and peak_swap_gb;
- the output shas it hashed.

A heartbeat older than 3 min means the run is dead. Today's `logs/e72f_steer_watchdog.log` records only "run exited; peak swap 7.36 GB", with no exit code.

**Completion.** An idle session does not poll, so a status file alone wakes nobody.
- **Chain the verdict.** Every frozen CPU verdict phase runs inside the same launched job as its MPS phase, so no verdict waits on a parent session.
- **If a parent must act afterwards,** it schedules the check. Use Monitor with an until-loop on the status file's `end` field, or CronCreate at a 20–30 min cadence with a timeout of 2 × expected_s. Never rely on a bare background waiter.
- **A dead pid with no exit record is CRASHED.** Report it to the user. A frozen phase is not re-run; a capture resumes under §4.6.

### 7.4 Swap, memory and disk rules

- **Swap.** The watchdog kills the process group at 12 GB of machine-wide swap. Guards also run inside every loop: every 20 forwards, every dictionary, every encode batch. e67's start-only check let swap reach 18.7 GB.
- **Memory.**
  - Never materialise full-vocabulary logits (262,144 columns); gather union columns only. The Claude app crashed at about 14 GB of swap during e69 P1's full-vocabulary pass.
  - Never build n × N_rand projections; use second moments. A 203k × 10,000 projection was killed at 12.5 GB.
- **Disk.**
  - Delete 262k files and shuffle captures after use.
  - Keep the six S2-type row files (`e69_p1_S2_*`, `e72b_S2b{1..5}_*`, about 2.8 GB) until §9 step 4 is done; they are its inputs. Then, with the user's approval, delete them.
  - Their HASHES entries stay. Regeneration is deterministic and is verified against those shas.
  - Keep `e67_raw_L12_rows.npy`.

### 7.5 Dtype sidecars

- Assert per-coordinate max |x| < 60,000 before any fp16 cast. The fp16 maximum is 65,504. Coordinate 1038 reaches 63,232 at L12 and 67,072 at L14.
- Coordinate 1038 (and 1125 at L ≤ 2) always goes to an fp32 sidecar, restored on load. L ≥ 13 is stored in fp32.
- Encode with 1038 present, and drop it only after encoding (§0).
- e60's unchecked fp16 cast of the decoded output (`scripts/e60_extract.py` line 70) gets the same assert.
- No float64 on MPS (the e67 crash).

### 7.6 Multi-session etiquette

- One MPS job runs at a time, machine-wide. CPU phases may run concurrently when their declared RSS fits.
- One session owns each phase, and the status file names it. Other sessions read the status file; they do not relaunch.
- Ledger appends (HASHES, BLOCKS, SEEDS, RIG) happen under a file lock. e65 was launched by two sessions at once. That duplicated HASHES lines 42–47 and garbled the start of the B3 log.
- A session that finds another project's job running tells the user and waits.

## 8. Tooling, in build order

The first science job does not wait for the full stack. Items 0 and 1 run first, and the rest is built in parallel. All estimates are *(inferred)*.

| # | Tool | Extracted from | Targets | Effort |
|---|---|---|---|---|
| 0 | Copy `launch.py` and `watchdog.py` into `tools/`, generalise the phase argument, add exit_code and heartbeat, and hash them; append `design_panel_result.json`'s sha to HASHES; write the e72f card; back-fill SEEDS and BLOCKS | scratchpad `steer/` | F9; e72f gap | 1–2 h |
| 1 | KO calibration script under a hashed NOTE (§9, steps 1–2) | `e72.error_W`, e72's G (seed 7200), `saes.py`, `docs/sae_gap_calibration/err_ko.py`, `e64.append_hash` | F5, F7 | 3–4 h build; 30–60 min run |
| 2 | `lib/config.py`, `tools/preflight.py`, `tools/launch.py`, `tools/status.py` (heartbeat, exit_code, chained verdict) | e72f `guard`, e68 `folder_gb`, e64 `check_swap`, item 0 | F9 | 0.5 day |
| 3 | `lib/ledger.py`: lock, an active/superseded field instead of substring matching, a `ledger_root` argument, import pinning. Also `BLOCKS.jsonl`, `SEEDS.jsonl` and `blocks.read` | e64 `sha`/`append_hash`/`require_hashed`/`check_integrity`; e72f `USED_CORPUS`/`dup_screen` | F9; reserve exhaustion | 0.5 day |
| 4 | `lib/capture.py`: raw, decoded and error captures; fp32 sidecar; overflow assert; drift check; streamed shuffles; resumable | `e68.capture_rows`, the e69/e72b `residual`, e72b drift check, e70 sidecar | F9 | 0.5 day |
| 5 | `lib/nulls.py`: W-space and pullback banks; selection-matched runner with sequential stop; every screen returns its pass rate | `e65.rand_bank`, e72 G/V2, `e72f.cos_bank`/`eligible`, e68 joint nulls | F2, F4, F6 | 0.5 day |
| 6 | `lib/power.py`: planted κ ladder; split-block bootstrap; real forwards; floor ≤ ceiling check; like-quantile check | e72f `planted_dirs`/`graded_plants`, e70 `PLANT_KAPPA` | F1, F8 | 0.5 day |
| 7 | `lib/rig.py` with `validate_layer`; `docs/RIG.json` | `e66.make_forward`, e65 `natural_sigma`/`readout` and gates, `e72f.steer_engine` | F3 | 0.5 day |
| 8 | `lib/coverage.py`, generalising item 1 (T-ES, T-MG, T-COS, T-BEH, leak and ground-truth probes); `lib/instrument.py` and `tools/admit.py` | item 1; e67 `omp` | F5, F7 | 0.5 day |
| 9 | `templates/`, `tools/freeze.py`, `tools/report_sync.py` (flags hashed verdicts missing from REPORT/SUMMARY), `tools/budget.py`, `tools/smoke.py` (shared `--test-dir`, fake SAE, tiny model, real-shape memory dry run; smokes hashed into `docs/`) | PREREG_fragmentation, NOTE_*, e72f smoke | all | 1 day |

The total is about 4.5 days. Items 2 and 3 must exist before any MPS phase on fresh text.

## 9. One cycle end to end: the next cycle for this project

**Goal.** Admit or reject a coverage test at L12 on burned data, and settle what the existing raw-h candidates are worth under it. The work is CPU first, with at most one short MPS capture. No fresh block is spent unless step 6 says GO.

**Step 0. Housekeeping (no compute).** Each item writes to the record, so each needs the user's go-ahead.
- Write the e72f card (§6.4) into REPORT.md and SUMMARY.md.
- Copy, generalise and hash the launcher.
- Append `design_panel_result.json`'s sha.
- Back-fill SEEDS and BLOCKS, and add corpus 8,000–8,999 to the duplicate screen.
- Draft DISCOVERY.md (§2.2) for approval.
- **Decide whether to retire skew #49 on existing numbers (the review's reading; the user decides).** Under the full lead rule its count is 1 on D2, against 0 on S2 and 1, 1, 1, 1, 0 on S2b1–5. The review argues that no number of further shuffles can give GO, and that the rule's firing rate (0.8) voids it. No compute. The e72 and e72b verdicts stand as recorded.

**Step 1. `NOTE_coverage_admission_L12.md`, hashed before any number.** It fixes:
- the tests: T-ES, T-MG and T-COS (dummy);
- the targets: paren skew #45, quote logcosh #29 and quote skew #29 (raw h, e72);
- the KO sets (via sha `63da457b…`) and the extension rule;
- the class bars and rules 1–8 with their bars;
- the control order: paren, then quote, stop on the first failure;
- the 40-direction held-out bank (seed 7400), the 20 matched ablations (seeds 7401–7420) and C2′ at κ ∈ {1, 2, 4} (seeds 7421–7423) *(inferred seeds)*;
- the outcome wording: ADMITTED / ADMITTED-GRADED / NOT ADMITTED.

**Step 2. Run the calibration** on `results/e67_raw_L12_rows.npy` with L12 16k l0_big, then with l0_small (its own enrichment-ranked latents).
- Launch through the launcher with the verdict chained.
- CPU only: 30–60 min, under 4 GB *(inferred)*.
- Output: `results/e73_coverage_admission.json`, hashed, with its card in INSTRUMENTS.md.
- If no test is admitted at L12, R5 and R6 close at L12 for the 16k family. Write the methods card and stop.

**Step 3. Only if a test is ADMITTED or ADMITTED-GRADED at L12:** capture raw D2 at L11 and L13.
- This is one MPS phase on burned text and needs the user's approval: about 3 min and about 1.4 GB, with L13 in fp32 *(inferred)*.
- Run the step 2 ladder on L11 and L13 16k l0_big.
- Run the L13 width ladder (16k_l0_medium against 65k medium and big, all cached) as the second positive.
- If any list dictionary's test is NOT ADMITTED, DISCOVERY is unreachable at L12 for now. Write the card and stop.

**Step 4. Re-screen pilot (CPU, burned data).** Apply each admitted test to the 60 screened raw-h L12 units on D2. Apply it unchanged to the units of the six shuffle sets already on disk (S2 and S2b1–5).
- For T-ES this re-reads e72 and e72b, because T-ES at k = 0 is e72's ES. Only a changed class bar can change the answer. T-MG is new computation.
- Count the units that pass the full selection: screened, not identity-suspect, norm/L0 and position screens, both parity halves, and ADDS-TO-X.
- If D2's count does not exceed all six existing shuffle counts: STOP, with no MPS.
- Otherwise stream further shuffles one at a time, stopping at the first whose count reaches D2's (STOP) or at k = 19 (GO). That is up to 13 more shuffles, at most about 21 min of MPS *(inferred)*.

**Step 5. Cheap falsifiers on the survivors.**
- R5 with the admitted test against all four list dictionaries.
- R4 with the hashed battery, with the random-direction UNNAMED rate alongside.
- Both run on CPU in minutes.

**Step 6. Decision.**
- **GO for a fresh prereg** only if all of these hold:
  - step 4 gives GO;
  - at least one unit reads ADDS-TO-X for all four dictionaries, with a margin above the test's MDE;
  - split-block bootstrap power for the D3 run is at least 0.8.
- **Expected outcome: STOP** *(inferred)*. The T-ES answer is already in e72b, and e72's only lead has about 76% SAE share.
- **Output either way:** an admitted or rejected coverage test, its MDE, and a card.

**Step 7. Only on GO:**
- a discovery prereg on D3 (corpus 2,000–2,999), with R1 on corpus 3,000–3,999;
- stage E on D3's rows;
- raw-h rig validation (§4.4), then R3a on B5 and B6 (corpus 4,000–5,999) and R6 at L12;
- one R3b on corpus 9,000–9,999.

Each step ends with a card, and the next step needs the user's go-ahead.

## 10. Scaling

**New layer.**
- Run admission (R0 controls, planted recall), rig validation, and the KO ladder for each dictionary at that layer.
- If no anchor passes after the bounded search, the layer holds R0–R2 and geometric-only R5.

**New dictionary.**
- Run the KO ladder on its own top-enriched latents, then add it to DISCOVERY.md (new hash and the user's approval).
- Transcoders with an affine skip term keep the skip path after ablation, so their knockout must remove or account for it *(inferred)*.
- Whether Gemma Scope 2 ships transcoders or crosscoders for 1B is *unverified locally*: no project file records it.

**gemma-3-4b-pt** (REPORT Stage 5, lines 533–596):
- d_model 2560, 34 layers;
- Gemma Scope 2 `resid_post_all` 16k and 262k, small and big, at all 34 layers. The 16k files are 335,686,016 B. l0_big is 91 at L6 and 120 at L12/L20/L28.
- the model's safetensors total 8.60 GB, including the vision tower;
- 14–16 GB of unified memory;
- about 9–13 h for a 6-layer geometry scan with one behavioural adjudication.

REPORT's "Stage 4 not recommended at 4B as it stands" predates v2, because it cites the v1 failure. Its disk point still stands: four occurrence sets × 3 layers is about 15.7 GB of fp16.

**4B entry conditions:**
1. **A disk plan, approved before anything else.** The 8.6 GB model plus the current 15 GB exceeds the 20 GB cap. Either delete regenerable files (about 3.3 GB of fp16 row files plus the 1.31 GB type means, about 4.6 GB in all) or give 4B its own folder and cap.
2. An ADMITTED or ADMITTED-GRADED coverage test at 1B.
3. At least 2 protected blocks remaining.
4. Re-admission at each 4B layer:
   - an outlier-coordinate scan on a burned block, to find 4B's equivalent of 1038;
   - R0 with paren and quote;
   - a rig card;
   - a KO ladder on the 4B dictionary.

Captures at d = 2560 are about 1.04 GB per layer per block *(inferred)*, so they stream into CPU accumulators. A 4B run needs the machine to itself. The lock and the `ps` refusal enforce that among sessions that use the launcher.

**Other structure types** need the same five interface methods (plant, anchor, null, power, cost) before reuse:
- **Rings:** months as the anchor, at the layers where the supervised ring closes in the representation searched, with a class-permutation negative.
- **Rulers:** planted rulers (detected about 90% of the time), with a frequency-matched null (number words came out at ρ −1.00).
- **Lattices:** stay NOT QUALIFIED until a planted lattice is recovered; kinship was invisible.

**Blind round.** Before a new instrument family's claims are accepted, one session plants a direction in corpus 15,000–15,695. A second session runs the frozen pipeline without knowing where it is.

## 11. Replay appendix

The table walks the record through this process. Savings are *(inferred)* from timestamps and logs. "Confidence" is about whether the step would have fired, not about the size of the saving.

| Item | Step that catches it | When | Saving | Confidence |
|---|---|---|---|---|
| S1: blind deep ring scan and two-channel adjudication ran although the supervised month ring was open in the searched representation at every deep layer (strict-12 failed at all 7) | R0 admission per layer: supervised control first; "anchor fails: no candidate number is read" | before the deep script-22 and e42d runs | deep e42d adjudication (12–28 min per layer) and the deep script-22 runs | high |
| S3: 8-token slices sampled from all 8,000 reserve sequences | `blocks.read` refuses unclaimed blocks; Stage 3 claims one block | before the retest | a narrower exposure statement for every later block | medium |
| F1 v1: ICA-32 on whitened input; TDI self-matches; floor 0.85 above the probe ceiling 0.83 | planted-source unit test (ICA dimension = input dimension; self-match excluded) and §4.2 item 5. Both numbers are in the frozen PREREG_occurrence.md (lines 85, 105–106; file sha matches HASHES line 3) | at the v1 pilot, before the v1 freeze (about 08:49, 10-15) | the v1 D/S/R extraction and labelling (about 08:50–09:06); reserve 0–999 never extracted. The v2 diagnosis starts from a failing unit test, not a failed control; that time saving is not quantified | high |
| F2 e64: sibling-max null made of real effects (subject-NP Δ 1.52; anchor 0.246) | §4.2 composition statement (null draws only), plus a burned dry run of the anchor against the bank, plus §4.4 | e64 freeze (12:24) | e64's void behaviour test (about 34 min), the post hoc status of e65/e66, and part of the e65 redesign cycle. Reserve 3,000–3,999 is still spent, but on a valid test | high |
| F3 L7 anchor fails twice | §4.4 `validate_layer(7)` on burned sites; bounded L7 search | before e64 steers L7 | net MPS about zero: validation (about 28 min) costs about what e64 L7 (741 s) and e65 L7 B3 (846 s) spent. The real saving is the L7 share of reserve 3,000–4,999, no L7 candidates carried forward, and L7 labelled GEOMETRIC ONLY a day earlier | high |
| F4 e67: cosines 0.71–0.74 inside the subspace null (p99.9 0.74–0.85) | §4.2 item 2 (thresholds set against the same-subspace null; e65's pullback bank existed at 15:36, the NOTE froze at 17:27). Also §2.2: #13/#16 is decoded-span, so no R5 verdict runs on it, and §4.5 blocks the 262k downloads without an admitted gate | NOTE_prior_feature_check freeze (17:27) | e67's cosine verdict, e68 inheriting the 0.7 route, e67's about 8 GB of downloads and its 18.7 GB swap episode. A descriptive prior-feature reading, if the user wanted one, could still run on cached 16k | high for the cosine route; medium for the downloads |
| F5 e68: single-latent census; gate failed; no "uncovered" control | Stage A coverage admission on cached 16k: rule 1 fails (3 of 4 L12 controls NOT COVERED). The C1 KO object already existed (`err_ko.py` 20:42; manifest hashed 21:13) before the e68 freeze (21:21). §2.2 also leaves the decoded-span census with no eligible target | before the e68 freeze | e68's G7/G12 runs (565 + 709 s), its 7 primary downloads (2 of them 262k at about 120 s each; folder peak 17.7 GB), and two rubric reviews plus two code reviews | high |
| F6 e69/e70: 71% of random error filters pass the shuffle bar | §4.2 item 2 and the void rule | e69 pilot (NOTE frozen 21:29; GO at 22:07) | PREREG_sae_gap drafting and the three-reviewer review (to 22:46). Not saved: the P1 crash at 21:54 happened inside the pilot; that is F9's full-vocabulary rule | high |
| F7 e71: readout tautological for decoded-span units; sparsity calibrates one way | §2.2 decoded-span ineligibility, and §3.2 rule 1 (paren controls with dedicated latents read chance-like, so the readout cannot call controls covered) | e68 or, at the latest, the e71 NOTE (09:35, 10-16) | the e71 rubric and review cycle (about 1–2 h). The raw-h search (e72) starts right after the e63 admission instead of on 10-16 | medium-high |
| F8 e72f: bank median 1.29 below RAND max 2.21 on fresh block F | §4.2 item 4: power with real forwards on burned D2 sites (about 200 passes, about 27 min); the real-forward freeze dry run; the bar-design rule (RAND S p95 0.94 against max 2.21) | before the e72f freeze (12:38, 10-16) | block F and the 61-min steer (3,672 s). Not saved: the three pre-freeze review rounds | high |
| F9a HF_HOME unset (e68 attempt 1) | `lib/config.py` plus preflight | launch | one relaunch | high |
| F9b app crash at about 14 GB swap in e69 P1's 262k-vocab logits pass | §7.4 union columns plus the watchdog | design time | the crash and the resume | high |
| F9c fp16 overflow of 1038 at L14 (e67) | §7.5 assert and sidecar | first capture | the void L14 cells and the e67b supplement | high |
| F9d verdict waited about 3 h (steer exit 19:04, verdict 22:06) | §7.3 chained verdict | launch | about 3 h. A second stall in another session is saved only if it also used the launcher | medium |
| F9e untouched reserve exhausted by e66 | §2.2 and §4.1: #13/#16 is decoded-span, so it never claims a protected block; §5.3 rule 2: description on D2 | e66 prereg | reserve 7,000–7,999 kept. Reserve 6,000–6,999 is kept unless the user approves a methods confirmation on it as an exploratory block | medium |
| F9f e65 double launch; e67 18.7 GB swap; other projects' jobs | ledger lock and launcher lock; in-loop guards; `ps` scan | launch | the garbled B3 log and the duplicate HASHES lines; the swap excursion. Other projects are only reported, not stopped | medium (lock), high (guards), low (other projects) |
| e72: GO by rule on one thin lead | §4.2 full-selection count with sequential stop: S2 gives 0, then the first new shuffle (S2b1) already fires 1, so STOP. The void rule (1 × 0.8 > 0.5) agrees | e72 pilot (10-16) | four of e72b's five captures (about 6 min of MPS, 1.9 GB) and the GO/STOP ambiguity | medium-high |

**What this process still does not catch:**
1. **The top rung may stay out of reach.** If rule 8 is never reached, the best label is ADDS-TO-X (GRADED).
2. **Swap is machine-wide.** The watchdog can kill a frozen phase because of another project's job, and that spends the block.
   - The v2 extractions (swap rising from about 6.3 to about 13 GB, REPORT Stage 4 v2) would have been killed at 12 GB.
   - Captures must therefore be resumable.
3. **Locks bind only sessions that use the launcher.**
4. **Power estimated on D2 may not transfer to a fresh block.** The 0.8 bar reduces INCONCLUSIVE outcomes but does not remove them.
5. **No positive control is known not to be a blend in the steering sense.** T-BEH uses a constructed knockout, which certifies sparse removal at the SAE's own amplitude only.
