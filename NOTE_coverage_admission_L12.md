# NOTE: coverage-test admission at L12 by a knockout positive control (e73)

**Status: FROZEN when its sha is appended to `docs/HASHES.txt`, before any number it governs.** It authorises one CPU-only calibration run (PROCESS.md §9 step 2) on burned D2 rows. No MPS, no download, no fresh text. It amends nothing: every e64–e72f verdict stands as recorded. `DISCOVERY.md` does not exist yet, so no DISCOVERY sha applies. Steps 3+ of PROCESS §9 need a new go-ahead from the user whatever this run says.

**Question (rung R5, Stage A admission).** Can a coverage test tell "covered by the dictionary X" from "not covered", when the not-covered case is built on purpose? The positive control knocks out X's own paren and quote latents, which makes those variables uncovered by construction (PROCESS §3). A test is admitted only if it detects that, stays quiet where it should, and detects a planted dense variable.

**Script:** `scripts/e73_coverage_admission.py`, phases `calibrate_l0_big` and `calibrate_l0_small` (all numbers for one dictionary each, in separate processes) and `verdict` (applies §10 to the hashed numbers). **Smoke test:** `docs/e73_smoke_synthetic.py`. Both are hashed with this NOTE. The script asserts its own sha is recorded in HASHES and that this NOTE's sha is there.

## 1. What was seen before freezing

- **Burned numbers already in the record** (D2, corpus 1,000–1,999):
  - e72's error share (ES) at full X = L12 16k l0_big: quote logcosh #29 0.274 (p0.4), quote skew #29 0.276 (p0.6), paren skew #45 0.295 (p13.5). Seed-7200 random bank: median 0.309, p99.9 0.350, max 0.359. FVU 0.190 raw, 0.315 identity-removed.
  - e72's W-space logistic probes: paren 0.778, quote 0.780 (`e72_gate.json`).
  - The design panel's knockout pipeline (α = 1) recovers paren 0.781 and quote 0.747; probes on the unablated error top out at 0.57/0.58; the knockout set carries 4.6% of the error variance.
  - Gating leak (design panel, `err_split.py`): the pre-activations of #2682 and #4001 still separate paren among their inactive rows (AUROC 0.67, 0.72); 70–90% of paren rows have the top latent inactive.
  - `err_split.py` names l0_small top-5 lists (paren 4493, 1351, 2682, 1559, 645; quote 1229, 2899, 5082, 1472, 10306) from a ranking whose code is not on disk.
- **Not seen:** any statistic of T-ES, T-MG or T-COS under any knockout, under l0_small, or on any plant; any W-space number under l0_small; any ranking computed by the §4.3 rule; any probe on a knocked-out reconstruction.
- **The design panel's enrichment-ranking code is not on disk** (searched `docs/sae_gap_calibration/`, `scripts/`, the session scratchpads). The frozen 10 + 10 come from `design_panel_result.json` by sha. §4.3 defines the rule used beyond them.
- The smoke test ran on synthetic activations only (real D2 metadata and labels, as e72's test mode did). The adversarial reviewer's rule-5 simulation used e72's published bank quantiles and synthetic covariances, no new real number.

## 2. Claimed rung and output words

R5 admission, per (test, dictionary): **ADMITTED / ADMITTED-GRADED / NOT ADMITTED**, with the test's minimum detectable effect (MDE, κ units) and the first failing rule when not admitted. A dictionary with no hashed calibration output reads **NO VERDICT**. Tests: T-ES, T-MG and T-COS (the dummy). Dictionaries: L12 `resid_post_all` 16k l0_big, then 16k l0_small.

## 3. Identification assumptions, and what fails with each

1. **Knockout is exact.** `saes.py` gates each latent on its own pre-activation, so removing a set K leaves every other code unchanged and the ablated error is e_K = e + Σ_{k∈K} z_k·W_dec[k] (the C1 object of `err_ko.py`). Asserted numerically on 4,096 rows per dictionary (§4.2).
2. **The e72 maps are linear.** demean3, the PCA projection and `e64.apply_maps` are linear, so W(e_K) = W(e) + W(Σ z_k W_dec[k]). Asserted (fast path against full path, §4.2). If either assert fails the run stops with no number.
3. **Knockout removes codes, not information.** Other latents and sub-threshold drive may still carry paren and quote. Rules 6 and 8 measure this. If it holds strongly, rule 8 reports "ground truth not reached" and the best possible level is ADMITTED-GRADED.
4. **The plant target is an oracle.** Rule 7 scores the plant through its least-squares W-space filter, not through a blind instrument. The MDE is therefore optimistic: a lower bound on what a blind screen would need.
5. **The positive control is an SAE-amplitude, SAE-sparsity variable.** Admission certifies detection of dictionary-shaped losses only (PROCESS §11, item 5).

## 4. Frozen inputs and definitions

### 4.1 Inputs (sha256; the script asserts each)

| input | sha256 | in HASHES before this NOTE |
|---|---|---|
| `results/e67_raw_L12_rows.npy` (203,328 × 1152 fp16, 1038 stored) | `14d80c9661c71fcff0a87f89358d64957cdf68d94e63a6ca10200645152fa0b9` | yes |
| `results/e72_units_raw_L12.npz` (maps, W, Z, readouts, dirs) | `3a4da05a324400fbbb92f3a13a3fd061a2a29d5d4cdd630ccd0c35771dec68dc` | yes |
| `results/e72_gate.json` (control units) | `2f761d23d520079e34097797d7d9f67567d3ed8e39048b7a6462928cd74cb3bd` | yes |
| `results/e72_coverage.json` (k = 0 reproduction target) | `a6b239b36e9f2104ac7a017becca052059cd761e7ca5d6ed316bfb161f727474` | yes |
| `scripts/e72_raw_residual_pilot.py` (`error_W`, `rows`, `covs`, `es`, `qs`, `demean3`, `project`) | `338c10b252b816eb40072d4b6492f81c072850edf00a1fbf1995d1cfeb27d394` | yes |
| `scripts/e64_occurrence_v2_discovery.py` (`append_hash`, `require_hashed`, `apply_maps`, `check_integrity`) | `1c97e42b4fc2fbf64410e8dda39fb8a9d6ae938e3fa960e6bf36d1a009fcf1f4` | yes |
| `scripts/e69_sae_gap_pilot.py` (`sae_params`, `error_rows`) | `ef1f04dd47dc92437040885ef52bde25d260c0134b5a93fe9303f88caff69bc5` | yes |
| `scripts/e70_sae_gap_geometric.py` (`markov_plant` constants: rate 0.06, mean run 16) | `3fd0fbe2950d28624a6358e184f071cff3d9f941b571a2c6f37f6bb0aab03246` | no (asserted by sha) |
| `src/gfdisc/saes.py` (`JumpReLUSAE.encode`) | `b904ec290cf7076a29e64c8dc5e52e5fcf0a081500dd133c80ed6b18de5ba75f` | no (asserted by sha) |
| `docs/sae_gap_calibration/err_ko.py` (knockout formula; reference) | `b61360bacfca9915a38c1011810b3bd814a5bc6187be4e041f5c9cf68708db94` | no (asserted by sha) |
| `docs/sae_gap_calibration/design_panel_result.json` (frozen 20-latent list) | `63da457bf5647c8bd1c359c9da6559bbb2272073837fada1c838cf4adc0e041b` | yes (manifest line 68; own line appended 2026-10-17) |
| `cache/occ/occD2_meta.npz` | `bc910456116d5b875438c741c42379e24715576a2bea29e33b0dd9c1a92bc5a7` | yes |
| L12 16k l0_big `params.safetensors` (snapshot `b738dc06…`) | `e936d310077f7bf8b09c0f2602e2b62087b69b8c8f82ba1263cf9ed3bd4471ff` | no (asserted by sha) |
| L12 16k l0_small `params.safetensors` (snapshot `b738dc06…`) | `6e1ae92a5a4d0956f1e88bd2e79e89e80f5998c76e5bab33a81a35a98488c96e` | no (asserted by sha) |

`e64.check_integrity()` also runs (the frozen e61/e63/e60 files).

### 4.2 Data, encoding and the fixed maps

- **Rows:** e72's D2 kept rows (`e72.rows("occD2")`, 203,328), asserted equal to `e72_units_raw_L12.npz["row"]`. Labels: `paren_inside`, `quote_inside` from `e61.feature_battery`, as in e72's gate. Even/odd = D2 sequence id parity.
- **Encoding:** full 1152-d rows in fp32 (`saes.JumpReLUSAE.encode`: pre = x·W_enc + b_enc; z = pre·[pre > θ]). Coordinate 1038 is dropped only after encoding and decoding. Codes are kept once per dictionary as a sparse matrix.
- **Error:** e = h − (z·W_dec + b_dec) on kept coordinates, through `e72.error_W` unchanged. Its W-image W_e and the fixed W_h (`e72_units_raw_L12.npz["W"]`) are exactly e72's.
- **Knockout term:** for a set K, W_K = L(Σ_{k∈K} z_k·W_dec[k, kept]), where L is e72's linear map (demean3 → PCA projection with own mean → `e64.apply_maps`). It is computed on the fast path L = apply_maps(demean3(X·P_U / P_sd) − mean), which uses linearity. **Asserted once per dictionary:** fast path against the full 1151-d path for K = that dictionary's paren top-10, max |Δ| ≤ 1e-3 × the RMS of the full-path result. Then W_e(K) = W_e + W_K.
- **Knockout identity (asserted once per dictionary):** on the first 4,096 rows, the error of X with θ_k = +∞ for k ∈ K equals e + Σ_{k∈K} z_k·W_dec[k] on kept coordinates (max |Δ| ≤ 1e-3 × max |e|), for K = l0_big's frozen paren 10, or for l0_small the 10 latents with the largest even |r(pre, paren)|.
- **Reconstruction image:** W_s(K) = W_h − W_e(K), X∖K's identity-removed reconstruction in W space.
- **k = 0 reproduction (l0_big only):** T-ES of the three targets and the seed-7200 bank quantiles must equal `e72_coverage.json` to 1e-9. Otherwise the run stops.
- **Swap guard:** `e64.swap_used_gb()` is checked before every condition; above 10 GB the phase stops (the watchdog kills at 12 GB). Every progress line carries current and peak RSS.

### 4.3 The knockout ladder, per dictionary and per control ℓ ∈ {paren, quote}

- **Top-10 set K10_ℓ.** l0_big: the frozen lists, parsed from the hashed `design_panel_result.json` in its stated order — paren 2682, 7780, 4001, 4493, 1559, 8951, 3503, 1442, 507, 1056; quote 5254, 1229, 1994, 1534, 564, 1882, 7302, 10306, 398, 1550. l0_small: the top 10 of rule E below.
- **Enrichment rule E (frozen here).** On even sequences: f1_j = share of label rows with z_j > 0, f0_j = share of non-label rows with z_j > 0; E_j = log((f1_j + 1e-4) / (f0_j + 1e-4)). Eligible: z_j > 0 on at least 20 even label rows. Rank by E_j descending, ties by f1_j. Reported descriptively: the overlap of E's top 10 with the frozen 10 (l0_big) and with `err_split.py`'s top-5 (l0_small), and the share of each step's added latents whose odd-sequence E is above 0.
- **Near-duplicates ND_ℓ** (latents not in K10_ℓ), either of:
  - **decoder:** max_{k∈K10_ℓ} |cos(W_dec[j], W_dec[k])| on kept coordinates above τ_dec, the p99.9 of the same statistic for 200 random 10-row sets drawn from rows outside K10_paren ∪ K10_quote (seed 7424; the statistic is taken for every row outside the drawn set);
  - **activation:** the within-sequence correlation |r(z_j, y_ℓ)| on even sequences (each sequence's mean removed from both z_j and y_ℓ, by second moments; codes stay sparse) above τ_act,ℓ, the p99.9 of the same statistic against 5 sequence-swapped copies of y_ℓ, pooled over every latent outside K10_ℓ (seed 7425; §4.6 swap rule). *Change after review:* the draft used raw correlations, which admit every document-genre latent (paren and quote rates carry 18.5% and 32.7% of their variance between sequences).
- **Steps, nested by construction:** s0 = ∅; s1 = first latent of K10; s3 = first 3; s10 = K10; sND = K10 ∪ ND; s30, s100, s256 = sND ∪ (top 30 / 100 / 256 of rule E). The ladder order is s0, s1, s3, s10, sND, s30, s100, s256. |K| and the knockout term's share of the W-space error variance (trace) are reported per step.
- **Matched random ablations (rule 4a):** 20 sets of 10 per control (paren seeds 7401–7420, quote seeds 7431–7450, one `default_rng(seed)` per set). For each k in K10_ℓ, in list order, draw uniformly among latents whose D2 firing-rate rank (all rows) is within ±50 of k's, excluding K(s256)_paren ∪ K(s256)_quote and latents already drawn for that set. If the ±50 window holds no eligible latent, it doubles until one does (recorded).
- **Size-matched random ablations (rule 4b; added after review):** for each control and each step s ∈ {sND, s30, s100, s256} whose set is larger than the previous step's, 20 sets drawn by the same per-latent firing-rate rule for every latent of K(s) (seed 7500 + 100·c + 20·i_s + i, c = 0 paren / 1 quote, i_s = 0…3 in that step order, i = 0…19). A step whose set equals the previous one reuses its draws.

### 4.4 Targets and banks

- **Targets** (asserted to be exactly `e72_gate.json["control_units"]`): paren = {skew #45}; quote = {logcosh #29, skew #29}. W-space filter b = `readout_<kind>[:64, j]`. Raw direction r(b) = P_U·(P_sd ⊙ Cov(Z, W)·b), normalised: e72's `cos_null` map; asserted |cos| ≥ 0.9999 with `dir_<kind>[j]` on kept coordinates.
- **Bar bank:** e72's seed-7200 bank G (64 × 10,000 unit columns; reused, not a new draw). T-ES and T-COS use all 10,000; T-MG uses the first 200 columns.
- **Stability bank S40:** 40 unit vectors in W space, `default_rng(7400).standard_normal((64, 40))`, normalised. It never sets a bar.
- **Class of a statistic** under a condition c (a removal set or a plant), against the bar bank scored under the same c: **ADDS** if above the bank max; **COVERED** if at or below the bank median; **UNRESOLVED** otherwise.

### 4.5 The three tests

- **T-ES(b; c)** = bᵀC_ee(c)b / bᵀC_hh(c)b, with C from `e72.covs` over all rows of W_h(c) and W_e(c) (`e72.es`). At c = full X, l0_big, this is e72's ES exactly. Second moments only.
- **T-MG(b; c), marginal gain in W space.** A one-latent JumpReLU appended to X∖K reads h through the fixed maps: s = W_h(c)·b, standardised by its even-row mean and SD. For sign σ ∈ {+1, −1} and threshold θ ∈ {0, 0.5, 1, 1.5, 2, 2.5, 3}: a = σs·[σs > θ]. With ã = a − mean_even(a) and ε̃ = W_e(c) − mean_even(W_e(c)), the decoder is v = Σ_even ã·ε̃ / Σ_even ã² (64-d). (σ, θ) is the pair with the largest even gain ‖Σ_even ã ε̃‖² / Σ_even ã². **T-MG = 1 − Σ_odd ‖ε̃ − ã·v‖² / Σ_odd ‖ε̃‖²**, the held-out drop in W-space error variance. Targets and bank columns are fitted the same way, in batches of at most 50 directions (never n × N_bank).
- **T-COS(b; c)** (dummy) = max_{k∈K(c)} |cos(r(b), W_dec[k, kept])|; 0 when K(c) = ∅ (full X and every plant). Bank columns use the same raw map.

### 4.6 Probes and their band (rules 6 and 8)

- **Probe:** ridge regression on standardised features (standardised on even rows; float32 features, float64 moments), λ = 1e-3 × trace(XᵀX)/p, fitted on even sequences; statistic = AUROC of the odd-row scores for the label.
- **Band (sequence-swap null):** the same probe refitted on 200 label copies (seed 7423). In each copy every D2 sequence takes the kept-row label path of another sequence, drawn uniformly with replacement among the other sequences, circularly offset by a uniform start and wrapped to its own kept-row count. Run lengths and base rates are kept; alignment with the sequence's own features is not. The band's top is the p99 of the 200 null AUROCs. The same 200 copies serve every probe in both dictionaries. The activation null of §4.3 uses the same swap rule with seed 7425.
- *Design change before freezing (synthetic data only):* the first draft used within-sequence circular shifts. On the synthetic smoke its band rose with the signal the probe carries (band top 0.73 against a probe of 0.90), because long paren and quote runs stay partly aligned after a shift. It was replaced by the swap null above.
- **Rule 8 probe (lost-information ground truth):** features = W_s(K(s)), the 64-d reconstruction image. The raw-h reference is the same ridge probe on W_h (unchanged by knockout; reported).
- **Rule 6 probe (sub-threshold leak):** features = W_s(K(s)), plus demean3'd sub-threshold drive pre_k·[0 < pre_k < θ_k] for the removed latents (at most 128: those with the largest even |r(pre_k, y_ℓ)|), plus demean3'd [z_j, clip(pre_j, 0, θ_j)] for the 32 remaining latents with the largest even |r(pre_j, y_ℓ)| (e69 `subthreshold` precedent). *Change after review:* clip(pre, 0, θ) of a removed latent equals θ whenever it fires, which hands the probe the removed firing indicator; only the sub-threshold part is kept. W_s is included on purpose (the leak is everything X∖K still exposes), so the rule-6 AUROC is at least the rule-8 AUROC up to fitting noise.

### 4.7 The plant C2′ (rule 7)

- e70's two-state Markov path (rate 0.06, mean run 16; `markov_plant` logic copied, per-sequence seed 74,220,000 + sequence id) on the full 255-token D2 sequences, restricted to kept rows. **One path and one direction for all κ.**
- **Gating direction (W span):** u = P_U·g with g = `default_rng(7421).standard_normal(64)`, normalised: a random direction inside the 64-d PCA span that every test reads. 0 at 1038. σ = SD of demean3(h·u) over the kept rows.
- **Descriptive direction (full rank):** `default_rng(7426)`, 1151-d standard normal on kept coordinates, normalised. A full-rank random u lies mostly outside the 64-d W span, so its plant measures the representation's blind spot (the design panel's C2 caveat), not the test. Its MDE is reported and never gates.
- h′ = h + κσ·y·u for κ ∈ {1, 2, 4}, fp32, full rows; re-encoded with full X (no removal); e′ through `e72.error_W`; W_h′ = W_h + L(κσ·y·u).
- Target: the oracle filter b = argmin Σ_even (y − ȳ − W_h′·b)², scored like any target against the bar bank under the planted condition.
- **MDE** = the smallest κ at which the test reads ADDS on the W-span plant; "undefined" if none.

## 5. Blocks, seeds and exposure

- **Block:** D2, corpus 1,000–1,999, BURNED (the main pilot block). No block changes state. No fresh text, so no duplicate screen applies.
- **Seeds (new, registered in `docs/SEEDS.jsonl` with this NOTE):** 7400 (S40), 7401–7420 (paren-matched ablations), 7421 (plant u), 74,220,000 + sequence id (plant path), 7423 (probe-band sequence swaps), 7424 (decoder near-duplicate null), 7425 (activation near-duplicate null), 7426 (descriptive full-rank plant direction), 7431–7450 (quote-matched ablations), 7500–7579 and 7600–7679 (paren and quote size-matched ablations). Both dictionaries use the same seeds for the same draw kinds. 7200 is e72's bank, reused. 7422 itself is not drawn.
- **Exposure:** D2 has been read by every pilot since e63; this run adds nothing to any other block.

## 6. Cards

No admission card, rig card or coverage card exists yet (`INSTRUMENTS.md` and `docs/RIG.json` are not built). No steering happens, so no rig card is needed. This NOTE is the coverage-admission rubric; its result card goes in REPORT.md.

## 7. Positive and negative controls

- **Positive:** the knockout ladder of §4.3 for paren, then quote, and the plant of §4.7.
- **Negative:** matched random ablations, 10-latent and size-matched (rule 4); the stability bank (rule 5); cross-label removal (rule 3); and **T-COS**, a geometry-only dummy that must come out NOT ADMITTED. If T-COS comes out ADMITTED or ADMITTED-GRADED for any dictionary, the rule set is broken and **no test is admitted for any dictionary**.
- **What the guard does and does not show (stated after review).** T-COS is excluded structurally: every plant has K = ∅, so its statistic is 0, its MDE is undefined and rule 7 fails. On the synthetic smoke T-COS passed rules 1–4 (it reads ADDS from s1, while the rule-8 probe had not moved), so **rules 1–4 alone cannot tell geometry from information**. T-COS's rule 1–5 outcomes are reported descriptively. T-ES and T-MG cannot pass on geometry alone in the same way (the knockout term Σ z_k W_dec[k] is label-correlated by how K was chosen), but on the graded path nothing checks whether what X∖K still reconstructs makes an ADDS redundant; the graded wording concedes exactly this. As a descriptive check, the verdict reports at s*_ℓ the rule-8 AUROC drop from s0 against that step's null spread (band p99 − band median).

## 8. Nulls

- **Composition:** null draws only. The bar bank and S40 are random W-space unit vectors; no unit, control or sibling sits in any bank.
- **Space:** W space throughout (raw-h targets). No pullback bank is used.
- **Per condition:** every class is read against the bar bank rescored under the same condition (same dictionary, same knockout or plant).
- **Expected false ADDS per target and condition** under exchangeability: 1/10,001 for T-ES and T-COS, 1/201 for T-MG (reported).
- **No selection-matched null** is needed: targets and controls are fixed in advance; nothing is selected by this run.

## 9. Power and feasibility

- **Burned evidence that the mechanism moves information:** knockout recovers paren 0.781 and quote 0.747 in the design panel's pipeline (§1).
- **MDE** (rule 7) is this NOTE's power statement, in κ units, per test and dictionary.
- **Rule 1 feasibility for T-ES at l0_big** is already in the record (p0.4, p0.6, p13.5). Nothing is known for T-MG, T-COS or l0_small.
- **Expected outcome (inferred):** rule 8 does not reach ground truth by m = 256 (§1's leak numbers), so the best level is ADMITTED-GRADED; NOT ADMITTED is as likely.

## 10. Decision rule (first match, per test and dictionary)

Each control is judged on all its targets: a quote rule passes only if it passes for both quote targets. s*_ℓ is the first ladder step (s1 onwards) at which every target of ℓ reads ADDS; "none" if no step does.

1. **Covered at k = 0:** every target of ℓ reads COVERED at s0.
2. **Monotone and uncovers:** each target's class never moves down the order COVERED < UNRESOLVED < ADDS along s0 → s256, and s*_ℓ exists. The raw statistic's largest single-step decrease is reported (descriptive). *Design change before freezing (synthetic data only):* the first draft applied "never decreases" to the raw statistic. On the synthetic smoke a constructed control whose class goes COVERED → ADDS exactly where its carriers are removed failed that rule on ES dips of 0.008–0.024 caused by removing unrelated latents (error and reconstruction are not orthogonal for a JumpReLU SAE), so the rule now reads the class.
3. **Cross-specific:** no target of ℓ reads ADDS under the other control's removal at any step up to and including that control's s* (all steps if it has none).
4. **Matched random removal:** (a) the targets of ℓ all read ADDS under at most 1 of ℓ's 20 matched 10-latent ablations; and (b) if s*_ℓ is sND, s30, s100 or s256, under at most 1 of the 20 size-matched ablations of that step.
5. **Random-direction stability:** under every ablated condition (each step s1…s256 of both ladders, all 40 matched 10-latent ablations and all size-matched ablations), at most 2 of the 40 S40 directions move into or out of ADDS relative to s0. *Change after review:* PROCESS §3.2 says "change class"; counting COVERED ↔ UNRESOLVED moves around the bank median fails any test for reasons unrelated to coverage (the reviewer's simulation with e72's bank quantiles: a 4.6% low-rank bump to the error covariance moves 6–10 of 40 across the median, none into or out of ADDS). Moves into or out of ADDS match e71's cap ("≤ 2 of 40 nulls satisfy the rule"). All class moves are still reported.
6. **Sub-threshold leak (grading rule):** **CLEAN** if at some step where every target of ℓ reads ADDS, ℓ's rule-6 probe is at or below its band top; otherwise **GRADED PARTIAL COVER**. It never fails a test by itself. "At some step" is deliberately more lenient than PROCESS's "at the k where the test reads ADDS": a later ADDS step with no leak still shows the test detecting a fully removed variable.
7. **Planted dense variable:** the W-span plant's MDE is defined and at most 4. PROCESS's clause "any target's margin must exceed the MDE" applies to targets screened later (§9 steps 4–6), not to this calibration.
8. **Lost-information ground truth:** m*_ℓ = the first step s1…s256 at which ℓ's rule-8 probe is at or below its band top.
   - If m*_ℓ exists: **PASS** iff every target of ℓ reads ADDS at m*_ℓ and reads COVERED at every step before m*_ℓ where the rule-8 probe is above its band top; otherwise **FAIL**. (Steps after m*_ℓ are not held to COVERED: a probe excursion above the p99 band there is a 1-in-100 event per step, not information returning.)
   - If no step reaches the band: **NOT REACHED** ("ground truth not reached").

**Order:** paren rules 1 → 5, then quote rules 1 → 5, then rule 7. The first failure is the reported reason. Every number is computed in the one run either way; quote numbers after a paren failure are descriptive.

**Admission levels:**
- **ADMITTED:** rules 1–5 and 7 pass for both controls; rule 6 CLEAN for both; rule 8 PASS for both.
- **ADMITTED-GRADED:** rules 1–5 and 7 pass for both controls; rule 8 is PASS or NOT REACHED for each (never FAIL); and the ADMITTED conditions are not all met. *Resolution recorded here:* PROCESS §3.2 says rules 1–7 "pass"; rule 6 is a grading rule, so GRADED PARTIAL COVER lands a test here, not in NOT ADMITTED. The same holds when rule 8 is PASS for one control and NOT REACHED for the other.
- **NOT ADMITTED:** anything else.
- **Rule-set guard:** T-COS ADMITTED or ADMITTED-GRADED for any dictionary → every test for every dictionary reads NOT ADMITTED (rule set broken).
- **Missing data:** a dictionary whose calibration output is absent or unhashed reads NO VERDICT for every test.
- **Both tests admitted:** both levels are reported; the pre-specified pointer for later steps is the higher level, then T-ES before T-MG.

## 11. Outcome wording (fixed now)

- **ADMITTED:** "T-x is admitted at L12 16k <l0>: it reads the knocked-out paren and quote variables as ADDS where a held-out probe of the 64-d W-space image of X∖K's reconstruction says they are gone, stays COVERED where it says they are not, and detects a planted Markov variable at κ = MDE (oracle target)."
- **ADMITTED-GRADED:** "T-x is admitted at the graded level at L12 16k <l0>. Its ADDS means 'adds to X's thresholded codes relative to a dedicated-latent knockout'; 'uncovered' is never written. MDE κ = …"
- **NOT ADMITTED:** "T-x is not admitted at L12 16k <l0>; first failure: <control>, rule <n> (<number>)."
- **Descriptive only:** T-COS's rule 1–5 outcomes, the rule-8 drop at s*, all S40 class moves, the raw-statistic dips, the full-rank plant's MDE, the enrichment overlap checks, odd-sequence enrichment, |K| and variance shares, largest monotonicity dips, the probe AUROCs below rule 6/8's decisions, the raw-h probe level, the quote numbers after a paren failure, and the T-MG (σ, θ) choices.

## 12. Stop rules and fallback (stated in advance)

- **One launch, no retries.** A crash or watchdog kill leaves the missing dictionary NO VERDICT; nothing is relaunched without the user.
- **NO VERDICT for l0_big means no decision:** the path neither closes nor continues; the user decides.
- **If no test is ADMITTED or ADMITTED-GRADED for l0_big (with a verdict):** R5 and R6 close at L12 for the 16k family, a methods card is written, the 4B scale-up stays deferred, and no retrained SAE is substituted.
- **If a test is only ADMITTED-GRADED:** the path may continue with the limited wording; its best possible label is ADDS-TO-X (GRADED), and it cannot satisfy DISCOVERY item 3.
- **If l0_small disagrees with l0_big,** both levels are reported; DISCOVERY (PROCESS §2.2) needs every dictionary on the list, and whether step 3 proceeds is the user's decision.
- **Steps 3+ (the L11/L13 MPS capture, the re-screen) need the user's go-ahead** whatever this run says.

## 13. Ops plan

- **Launch:** `tools/launch.py --run e73 --script scripts/e73_coverage_admission.py --phase calibrate_l0_big --phase calibrate_l0_small --phase verdict --expected-s 3600 --declared-out-gb 0.1 --require-hashed NOTE_coverage_admission_L12.md scripts/e73_coverage_admission.py docs/e73_smoke_synthetic.py`. Detached, `caffeinate -i`, 12 GB machine-swap watchdog, the verdict chained in the same job, status in `logs/e73.status.json` (60 s heartbeat, exit codes).
- **Order:** l0_big, then l0_small, then the verdict, as three chained phases of one launched job; each dictionary runs in a fresh process (on the real-shape smoke, a single process carried 5.3 GB of l0_big's freed memory into l0_small and peaked at 9.0 GB) and its output is hashed as soon as its phase completes. The runner runs every chained phase whatever the previous exit code, so a crash in one dictionary leaves the other and the verdict (NO VERDICT for the missing one).
- **Budget (measured on the synthetic smoke):** about 22–30 min CPU (1,302 s for both dictionaries at d_sae 2,048, rule 4b's 160 size-matched conditions included); peak RSS at most 7 GB per phase, a GO condition checked on the real-shape smoke (a single-dictionary process peaked at 6.3 GB there). PROCESS's "under 4 GB" was an inferred figure: the reused `e72.error_W` alone adds about 1.7–2.5 GB transient (an n × 1151 fp32 error matrix and its demeaning) on a baseline of about 3 GB. Under 0.1 GB written.
- **Outputs:** `results/e73_calib_l0_big.json` + `.npz`, `results/e73_calib_l0_small.json` + `.npz`, `results/e73_coverage_admission.json`, each hashed via `e64.append_hash` by the phase that writes it.
- **Not resumable.** Preflight refuses if swap ≥ 8 GB, another e6x/e7x job is running, or a python process exceeds 4 GB RSS.
- **GO/STOP rule for this NOTE:** the run proceeds only if the real-shape smoke (d_sae 16,384) passes assertions A–G on the exact files hashed with this NOTE (A–F include T-COS NOT ADMITTED, the geometry trap and a positive end-to-end admission of T-ES; G is every phase's peak RSS ≤ 7 GB), and the adversarial review raises no unresolved blocking item. The d_sae 2,048 smoke passed A–F on the version just before three memory and process edits (uint8 label matrices, a float32 raw bank, per-dictionary phases); it was not rerun on the final bytes.

## 14. Review log

- 2026-10-17: design check (advisor) before drafting. Adopted: per-condition bars, an oracle plant target labelled optimistic, a nested ladder, near-duplicate nulls with fixed draw counts, a per-dictionary output with NO VERDICT on absence, the T-COS guard coded, and a smoke geometry trap. Kept against the advice: T-MG in W space with the latent reading h (not e) through the fixed maps, because the targets and banks live there; probes in the 64-d W space the tests read, not PCA-256.
- 2026-10-17: first synthetic smoke (d_sae 2048, then 16,384). Three design changes on synthetic data only: the probe band moved from within-sequence shifts to the sequence-swap null (§4.6); rule 2 reads the class, not the raw statistic (§10); the probe moved to float32 with float64 moments. It also showed a 9 GB peak, mostly test-mode map building and a synthetic L0 of 770; the smoke now builds e72-format units itself (so the script runs its real path) at L0 about 110.
- 2026-10-17: adversarial review (independent agent, read-only, no real activation) of the NOTE, script and smoke. Dispositions:
  - BLOCKING, stale smoke: rerun on the final files at both shapes (GO rule, §13). Adopted.
  - BLOCKING, rule 5 fails for reasons unrelated to coverage: count only moves into or out of ADDS (§10.5). Adopted.
  - BLOCKING, activation near-duplicates admit document-genre latents: within-sequence correlation (§4.3). Adopted.
  - BLOCKING, the guard cannot tell geometry from information beyond rule 7: stated in §7; T-COS rules 1–5 descriptive; the rule-8 drop at s* reported descriptively. The optional gating "rule-8-lite" was not adopted: it would add an admission rule PROCESS §3.2 does not have, and the graded wording already concedes the gap.
  - SHOULD-FIX, adopted: rule-6 removed-latent features keep only the sub-threshold drive; size-matched ablations at s* (rule 4b); a numeric knockout-identity assert; memory (no extra float32 copy, 4,096-row encode batches, RSS on every progress line, real-shape RSS as a GO condition); NO VERDICT means no decision (§12); a positive end-to-end smoke assertion.
  - NOTE-ONLY, adopted: rule 8 holds COVERED only before m*; the err_split overlap is computed; "shift" wording removed; PROCESS's MDE-margin clause deferred to steps 4–6; ADMITTED wording names the 64-d W-space image; tie-break and l0_small disagreement stated; rule 6's "at some step" marked deliberate.
  - Checked and fine by the reviewer: fast-path linearity, T-MG algebra, the swap null never picks the sequence itself, judge's order and levels, k = 0 determinism across thread counts, PlantedH through error_W, markov_plant identical to e70.

## 15. Pre-hash review outcome (2026-10-17, synthetic data only)

- **Real-shape smoke on the exact bytes hashed with this NOTE (d_sae 16,384, L0 ≈ 113): A–G PASS.**
  - Levels on synthetic data: T-ES and T-MG ADMITTED-GRADED for both synthetic dictionaries (rule 8 NOT REACHED: the synthetic paren and quote labels are correlated, so each reconstruction keeps the other's signal); T-COS NOT ADMITTED, first failure rule 7 (MDE undefined), as PROCESS §3.2 predicts.
  - Geometry trap reproduced: T-COS reads ADDS from s1 while sibling latents still reconstruct paren; T-ES and T-MG read ADDS only at sND, once the siblings are removed.
  - W-span plant MDE κ = 1 for T-ES and T-MG (oracle target, synthetic).
  - Cost: calibrate_l0_big 695 s, calibrate_l0_small 707 s, verdict 1 s. Peak RSS 6.52 GB (l0_big, at the encode) and 6.91 GB (l0_small, in the plant phase). Machine swap rose from 1.7 to 3.9 GB over the run, with other projects active.
- **Logic smoke (d_sae 2,048) on the version just before the last three memory/process edits:** A–F PASS with the same levels, 1,302 s, 5.74 GB in one process.
- **Superseded smoke runs (kept for the record):** 9.0 GB with both dictionaries in one process; 8.6 GB per phase before `e72.error_W` was moved ahead of the encode and the codes were built as per-batch CSR blocks.
- **Synthetic near-duplicate counts** (decoder 8, activation 133–144 at d_sae 16,384) say nothing about real data: the synthetic rows have no document structure.
- **Final design check (advisor):** the target-direction assert was verified on e72's hashed npz before freezing; |cos(r(b), dir)| = 1.0000 for all three targets (a reproduction of e72's own artefact, not a calibration number).
- **Adversarial review:** no unresolved blocking item (dispositions in §14).
- **GO.**

## 16. Result card

(filled in REPORT.md after the run)
