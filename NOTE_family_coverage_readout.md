# NOTE — Per-family coverage readout, sparsity relative to the null, and a small-K readout pre-check (e71; rubric fixed 2026-10-16 before any per-family number)

**What this is.** A descriptive follow-up to e68. It covers the 7 would-be-target families listed in
REPORT.md "Coverage census (e68) → Descriptive only", plus the e66/e67 family L12 logcosh #13 / skew #16
as the reference.
- **Data.** Already-hashed outputs, plus already-used D2 (corpus 1,000–1,999) only. No fresh corpus
  text (2,000–12,999) is read.
- **No verdict is amended.** e64–e70 stand. In particular, e68's COVERED / NOT COVERED calls and its
  failed calibration gate are not revised.
- **Not a verdict either.** Nothing here is a coverage verdict or a target nomination. It cannot
  license fresh text.
- **What was seen before this note.**
  - REPORT.md, including the e68 family table (best |cos|, max |r|);
  - e67's numbers for #13/#16;
  - the key structure of the e68 JSONs.
  - **No per-family OMP, latent-level match or context number for the 7 families has been read.**

**Units.**

| role | layer | units |
|---|---|---|
| targets (descriptive) | L7 | logcosh #18 / skew #36; logcosh #33; logcosh #49 / skew #8; logcosh #52; logcosh #57 / skew #33 |
| targets (descriptive) | L12 | logcosh #45 / skew #5; logcosh #54 / skew #54 |
| reference | L12 | logcosh #13 / skew #16 |
| **anchors** (calibration only) | L12 | logcosh #13, skew #16 (e67: COVERED as a blend); paren/quote controls logcosh #11, skew #23, logcosh #33, skew #33 |
| joint nulls (calibration only) | L12 | the 40 joint-null random directions of `results/e68_prep_G12.json` (scores in `_nullscores.npy`) |

- Family membership and twins are taken as recorded in `results/e68_census.json`.
- Scores are e68's oriented scores: sign_{kind}[id]·Y_{kind}[:, id].
- **L7 is template (a) only.** Its subspace null is degenerate (decoder p99.9 0.83–0.90; recon null
  p1 = 1–2 atoms), so (b) and (c) are confined to L12.

## (a) Per-family descriptive template (one page per family; no new compute except as stated)

1. **Identity.**
   - Members with role, e64 tier and R2-replicated flag. Every twin, with its role.
   - The best e64c rule |AUROC|. e65 B3/B4 outcomes (L12).
   - The positive signature T_u, decoded. All of these come from `e68_census.json` and `e68_prep_G{L}.json`.
2. **Unit contexts.**
   - The 12 strongest D2 rows of each member's oriented score, in e67's format: 15 tokens before,
     `[[token]]`, 3 after. This needs only D2 tokens and the tokenizer; no model or SAE.
   - e64's 12 random heavy-tail contexts (`e64_labels.json`) are cited alongside.
3. **Decoder.**
   - The pooled top-5 |cos| over the layer's primary dictionaries: dictionary, latent, signed cos.
   - Per dictionary, max |cos| against the subspace null p50 / p99.9, summarised as "above p99.9 in
     n of N dictionaries", with and without the source dictionary.
4. **Activation.** The pooled top-5 |r| (dictionary, latent, r, within-type r), from `e68_acts_G{L}.json`.
5. **Each tested latent's match**, for every latent in (3) and (4), as stored in `e68_acts`:
   - side, rows and sequences;
   - signature share against its threshold and the 500-random-latent p99 (when testable);
   - tail overlap o_ℓ against its threshold and p99;
   - the match flag.
   - Also: how many of all the unit's tested latents, in any primary dictionary, match.
6. **OMP.** Per dictionary and per mode (signed, non-negative): k08 and k09, plus cos_at_kmax when k08
   is unreached, against that dictionary's per-mode null p1 / p5 / p50 (`null_recon`).
7. **Context sharing.** The test is mechanical: **a latent "shares the unit's contexts" iff e68 records
   it as a match (signature or tail overlap).**
   - The page also gives the largest o_ℓ over all tested latents, with its threshold.
   - **Text** is pulled only for latents of `resid_post_all` L12 16k l0_small. That is the one
     dictionary encoded here; raw rows are stored only at L12 (`results/e67_raw_L12_rows.npy`). These
     latents are, per member, the top 3 by |cos| and the top 3 by |r| in that dictionary.
     - Each gets its 12 strongest D2 contexts (e67 format) and the share of those 12 rows inside the
       member's 1% heavy tail.
   - #13/#16's latent contexts are cited from `e67_acts.json`; nothing is recomputed.
   - For every other dictionary, and for all of L7, the latent side is the stored overlap and
     signature numbers only.
8. **Reading.** The analyst's reading: which construction the unit's contexts show, and whether the
   nearest latents' contexts show the same one. It is labelled as a reading, never as a verdict.

L12 pages add their (b) class and (c) K\*.

## (b) Sparsity relative to the null (L12 only; 14 primary dictionaries, and 13 without the source L12 16k big)

**Cells and bands.**
- A cell is (dictionary D, mode m ∈ {signed, nonneg}).
- k = the unit's OMP atoms to cos 0.8 (`omp[m].k08`). **Unreached (None) counts as k_max + 1** (49, or
  33 for 262k), exactly as the null did; cos_at_kmax is reported.
- q1, q5, q50 = `null_recon[m].k08.{p1, p5, p50}` of the same D.
- **Bands:** ≤p1 (k ≤ q1); ≤p5 (k ≤ q5, which includes ≤p1); mid (q5 < k ≤ q50); >p50 (k > q50).

**Per-unit statistics.** Per mode: S1_m and S5_m (the number of dictionaries at ≤p1 and ≤p5) and
H_m (the number at >p50). Every per-mode count is reported. The rule uses:
- S_p = max over modes of S_p,m. This leans toward "sparser", that is, toward blend-consistent.
- H = min over modes of H_m. This leans away from "denser".

**Cut for "sparser than chance".** The candidate rules are "S_p ≥ c" for p ∈ {p1, p5} and
c ∈ {1, …, N}.
- A rule is **admissible** iff at most 2 of the 40 L12 joint-null directions satisfy it (≤ 5%).
- **Choose** the admissible rule that makes the most of the 6 anchors "sparser".
- **Ties,** in order: fewer joint nulls passing; then p1 before p5; then the larger c.

**Calibration.**
- **Informative** iff the chosen rule makes **≥ 4 of the 6 anchors** sparser, and these include at
  least one of logcosh #13 / skew #16.
- **Otherwise** the sparsity reading **fails calibration and is reported as uninformative**:
  - targets' bands are listed descriptively, with no class;
  - nothing is revised.

**"Denser than chance."**
- The rule is H ≥ c_d, where c_d is the smallest c in 1..N such that ≤ 2 of the 40 joint nulls have
  H ≥ c. If no such c exists, the class is empty.
- An anchor that reads "denser" is flagged.
- **"Chance-like"** means neither sparser nor denser. A unit that meets both rules is read as
  "sparser" (lenient toward blend).

**Readings.**
- **Without the source dictionary,** the same procedure is re-derived on the 13 dictionaries, with its
  own cut and its own calibration.
- A family's class is its sparsest member's class.
- If the with- and without-source classes disagree, the family reading is the more blend-consistent
  one: sparser over chance-like over denser.

**Meaning (fixed now).**

| class | reading |
|---|---|
| **sparser than chance** | Consistent with a blend of a few existing latents, like the anchors. No "not a blend" support. |
| **chance-like** | Reconstructs like a random direction in the same subspace. It is not a sparse blend at this cut. This is **not** evidence of an uncovered variable, since random directions are chance-like by construction. |
| **denser than chance** | Descriptive only. |

None of these changes an e68 verdict.

**Order (enforced in code).** Stage `calib` reads only the anchors' and the joint nulls' cells. It
writes `results/e71_calib.json` and hashes it before stage `readout` reads any target cell.

## (c) Small-K readout pre-check (L12 only)

**Inputs.**
- **Dictionary:** `resid_post_all` L12 16k l0_small (non-source, same layer). It is encoded once on
  `results/e67_raw_L12_rows.npy`. Those are D2's 203,328 kept rows in fp16, in
  `occ_units_v2_L12.npz["row"]` order, with HF_HOME set to `hf_home` and HF_HUB_OFFLINE=1. The fp16
  rounding of the residual is disclosed.
- **Scores:** all 55 L12 screened units in the census (31 candidates, 20 context-named, 4 controls),
  plus the 40 joint nulls.
- **Halves:** even sequences (row // 255 even) for selection and fitting, odd sequences for testing.

**Ranking (e69's).** Latents are ranked by |Pearson r| between the raw pre-activation and the score on
the even half. Top-K means the first K of that score's list.

**Features.** For each top-K latent:
- a = pre·1[pre > θ] and s = min(max(pre, 0), θ);
- each column is type-centred on the current token type, with type means over all kept rows
  (label-free, as e68's within-type r did);
- then each is z-scored on the even half.

**Fit.** e69's `heldout_r2(X, y, even, "ridge")`, verbatim: RidgeCV with alphas
logspace(−2, 4, 7), y z-scored on the even half, R² on the odd half. It is computed for
K ∈ {4, 8, 16, 32, 64}. K\*_u is the smallest K with R² ≥ 0.5, or ">64".

**Calibration (in stage `calib`, on the anchors and the joint nulls only).**
- K_anc = the smallest K at which ≥ 4 of the 6 anchors reach R² ≥ 0.5.
- **Informative** iff K_anc exists **and** ≤ 2 of the 40 joint nulls reach R² ≥ 0.5 at K_anc.
- **Otherwise** (the nulls cross 0.5 at the anchors' K, or the anchors never reach it), linear
  predictability is **uninformative and reported as such**, with no per-target reading.

**Reported regardless.** Per-group R² at each K and the distribution of K\* (anchors, candidates,
context-named, targets, joint nulls).
- If the check is informative, a target with K\* ≤ K_anc is "linearly readable like the anchors".
- A target with K\* > K_anc is "not readable at the anchors' K". This is not evidence of novelty; it
  is an absence of the anchors' signature.

## Resources and outputs

- **Script:** `scripts/e71_family_readout.py`. It is hashed before its first run on real data.
  - Stages: `calib` (b-cuts, the one 16k-small encode for the rankings, and the (c) anchor/null fits);
    `readout` (bands, the (c) fits for every other score, family pages data, and the 16k-small latent
    contexts).
  - Both stages run under `caffeinate`, detached, with a 12 GB swap watchdog. The encode is streamed
    in batches, and no code matrix is materialised.
- **Outputs,** each hashed with `e64.append_hash` when written: `results/e71_calib.json`,
  `results/e71_rank_G12.npz`, `results/e71_readout.json`.
- **Write-up:** a REPORT.md section, then `SUMMARY.md`.
