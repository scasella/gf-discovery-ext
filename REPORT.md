# gf-discovery-ext — REPORT

Extending gf-discovery's label-free structure search on `gemma-3-1b-pt` with Gemma Scope 2 SAEs:
deeper layers, lines/grids, the discovery-verbs retest, and an identity-removed per-occurrence
search. Everything ran locally (M4 Pro, 24 GB, MPS), with no cloud spend.
`~/Downloads/gf-discovery` was not edited; its scripts and package are copied into this folder.

*Report appended stage by stage; complete as of 2026-10-15. All stages are done. e42d on L20–24, which the user marked lowest priority, also ran.*

*Addendum (2026-10-15): **Stage 4 v2** (last section) supersedes Stage 4's instrument verdict. With identity also removed from the second moment, and with ICA at full rank, the preregistered positive control **passes at L7 and L12**. It also corrects two measurement defects in v1, described in that section. The Stage 4 rows in the candidate table and bottom line below describe v1.*

*Addendum (2026-10-15): the **Stage 4 v2 discovery tier** (final section) ran under its own frozen prereg. It finds **no discovery**: both behavioural anchors fail the sibling-maximum criterion. 23 unnamed direction families replicate geometrically on R2; a post-verdict mechanical check of the context reading finds about 4 of them nameable, so about 19 remain. Many battery-unnamed L12 units turn out to be syntactic-position variables.*

*Addendum (2026-10-15): a **redesigned, post hoc steering tier (e65)** (final section) validates the L12 quote anchor on two held-out context blocks. Under it, one L12 candidate family (logcosh #13 / skew #16; its signature is temporal and causal subordinators) passes both blocks: a lead for a blind confirmatory test, not a discovery. The L7 anchor fails again.*

*Addendum (2026-10-15): the **e66 confirmation** (final section) ran under its own frozen prereg (`PREREG_occurrence_v2_confirm.md`, sha `59f1ae5b…`, registered before any reserve 6,000–7,999 token was used). The L12 logcosh #13 / skew #16 family **passes on untouched contexts**: Δ 1.49 against a random-direction maximum of 0.75 and a shuffled-signature maximum of 0.85, with the quote anchor passing too. It is the first unit here to clear all three tiers out of sample: geometric, steering and replication. Caveats: the steering tier was redesigned after its first version failed, and "unnamed" means unnamed by this project's battery and rules. Descriptively it reads as "clause core complete, adjunct slot open". **The bottom line below predates e66; whether to call this a discovery is the user's call.***

*Addendum (2026-10-15): the **preregistered confirmation (e66)** on untouched reserve 6,000–6,999 **CONFIRMS** the L12 logcosh #13 / skew #16 family (final section). The anchor passes; #13's Δ is 1.49, against a random max of 0.75 and a SIG max of 0.85. Described after the fact, it is a "clause complete, adjunct slot open" direction: pushing it promotes because/before/during/with/through and demotes line and sentence ends. e64's and e65's verdicts are not amended.*

*Addendum (2026-10-15): a **prior-feature check (e67)** finds the confirmed direction **COVERED** by existing Gemma Scope 2 latents under its pre-registered rubric. Examples: L13 16k-medium #183 and L12 16k-big #11562, with decoder |cos| 0.71–0.74 and matching contexts and logits. Read as a blend of existing adjunct-prediction latents, it is not a new concept (final section).*

*Addendum (2026-10-15): the **coverage census (e68)** checked every e64 candidate against 27 Gemma Scope 2 dictionaries. Its calibration gate **failed at both layers**: half the known-named and control units came out NOT COVERED, and at L7 6 of 40 random directions came out COVERED. So **no target is nominated**. A single-latent rule cannot certify that an identity-removed unit is uncovered. The e66/e67 family is COVERED again (second-to-last section).*

*Addendum (2026-10-15): the **SAE-gap pilot (e69)**, temporal blind source separation on the L12 SAE reconstruction error, gives **GO** by its frozen rule. 2 of 3 slow error units survive the shuffled-text, predictability and bag falsifiers, but the shuffle test is passed narrowly (46% of lag-1 autocorrelation kept, against a 50% bar). GO licenses writing a fresh-text prereg only (second-to-last section).*

*Addendum (2026-10-15): the **fresh-text prereg (e70) was drafted and reviewed but NOT frozen**. The pre-freeze review, checked independently on burned data, shows the pilot's shuffle test does not discriminate: 71% of random error-space directions pass it, and the survivors sit at the random median. The survivors' directions are also about 70% SAE-reconstructed in raw h (error share 0.26–0.31, against a 0.5 bar). So D2 already decides the round; no fresh text was spent (final section).*

*Addendum (2026-10-16): a **per-family coverage readout (e71)**, descriptive only, covers the 7 would-be-target families plus #13/#16, under a rubric hashed before any per-family number (final section). Sparsity relative to the null is informative, but both L12 target families read "chance-like", the class that also holds the paren controls. A small-K sibling-dictionary readout is uninformative: no anchor reaches R² 0.5. e68's context match does not separate controls from random same-subspace directions at L12. No target is nominated and no verdict is amended. A distilled summary is in `SUMMARY.md`.*

*Addendum (2026-10-16): a **raw-residual pilot (e72)** ran the v2 instrument on the raw L12 residual instead of the SAE-decoded one (final section; burned D2, rubric hashed before any number). The positive control **passes** on raw h. One unit, L12 skew #49, beats 10,000 same-subspace random directions on SAE error share and passes every screen, so the frozen rule gives **GO** (a fresh-text prereg only). The lead is thin: it is the same unit as decoded-span skew #31, about 76% SAE share, not off span on a single residual-space filter, and at the random median on shuffled text. Not recommended for fresh text on its own.*

*Addendum (2026-10-16): a **selection-matched shuffle null (e72b)** (final section) re-ran e72's S2 branch on 5 new within-sequence shuffles of burned D2 (rubric hashed before any shuffle number). Screened above-max units: 2, 2, 1, 1, 0 against D2's 4, which is **INDETERMINATE** by the frozen rule. Descriptively, e72's full lead rule fires once on 4 of 5 shuffles, as it did once on D2, so the e72 lead carries no evidence on its own. D2's excess is in identity-suspect units. Recommendation firmer: no fresh text on skew #49. e72's GO is not amended.*

*Addendum (2026-10-17, for a run of 2026-10-16): the **fragmentation test (e72f)** (final section) steered the e66 direction's best 16k-big decoder row against a share-matched, cosine-matched bank on fresh block F (corpus 8,000–8,999), under a prereg frozen before any block-F token was read. Headline **INCONCLUSIVE**: the quote anchor, the e66 direction itself (its fourth behavioural block), the two-class premise T1 and the planted positive control all pass, but the bank median is not potent in both classes against the random maximum, so the row cannot be classed. Post hoc, its tilt sits inside its bank's band, so no one-sided behaviour was seen. The fragmentation line is closed at 1B by the user's decision. e66 and e67 are not amended.*

*Addendum (2026-10-17): a **coverage-test admission at L12 (e73)** (final section) tried to build the missing positive control for "uncovered" by knocking out the dictionary's own paren and quote latents, on burned D2 rows only (CPU, rubric hashed before any number). **No coverage test is admitted** for L12 16k l0_big or l0_small. Error share (T-ES) and marginal gain (T-MG) both fail on the paren control, and the geometry-only dummy (T-COS) fails too, so the rule set held. Even with 1,015 near-duplicate latents removed (l0_big), the knockout never makes paren uncovered. By the frozen stop rule, and with the user's acceptance, the "adds to the dictionary" and "not a blend" rungs (R5 and R6) close at L12 for the 16k family. 4B stays deferred. Nothing is amended.*

*Correction (2026-10-24), to two readings and one figure in this report. (1) **The e66 false-pass bound of ≤ 1/77 is a rank bound, not the probability that the confirmed family is a false pass.** It is the preregistered bound of `PREREG_occurrence_v2_steering.md` §9, restated in `PREREG_occurrence_v2_confirm.md` §6, and it holds only under the exchangeability nulls stated there: absent a specific effect, the direction's push moves its signature like any of the 99 random pushes, or its signature responds like any of the 76 other signatures. Neither the direction nor its signature was assigned at random, so randomisation does not establish either null. The family was also selected from L12's 31 candidates by its B3 and B4 passes on the same rig, and under the prereg's stable-rank null a unit that has passed twice is more likely to pass again. The evidence is the margin over the controls on block C: Δ 1.49 against a random-direction maximum of 0.75 (median about −0.04) and a shuffled-signature maximum of 0.85. (2) **"Isolates known in-context variables blindly"** (Stage 4 v2, Reading) means the units are extracted without labels. The instrument's settings were chosen with parenthesis labels on calibration block P2 (corpus 13,000–13,999), as that section's design and caveats record, so parenthesis state is a tuned target and quotation, never used in any choice, is the untuned control. (3) **The e66 block-C table gave logcosh #13's Δ at c = 4 as 2.89.** `results/e66_confirm.json` gives 2.8846, so the cell now reads 2.88; this is the only figure changed in place. The write-up at <https://casella.dev/blog_gemma_blind_search.html> gives the same account. No verdict is amended.*

---

## Stage 0 — budget, SAEs, corpus

Details: `BUDGET.md` (written before any compute) and `docs/addendum_A1_gate_tolerance_2026-10-15.md`.

**Layers and suite.**
- L12, 14, 16, 18, 20, 22, 24 for the deep scan.
- L5–L9 type means rebuilt for Stage 2.
- Suite: `resid_post_all`, 16k, `l0_big`, the same suite as the L0–L10 work.

**SAE inventory.** `resid_post_all` has every layer 0–25 at 16k and 262k, each with small/big L0.
`resid_post` has only L7, 13, 17 and 22, at 16k/65k/262k/1m. Nominal L0 values:

| layer | 5 | 6 | 7 | 8 | 9 | 12–24 |
|---|---|---|---|---|---|---|
| L0, l0_big 16k | 94 | 101 | 108 | 115 | 120 | 120 |
| L0, l0_small 16k | 15 | 16 | 18 | 19 | 20 | 20 |

The 262k SAEs are 2.4 GB each and overflow the cache's uint16 indices, so they are not used.

**Corpus.** The identical FineWeb `sample-10BT` stream as the old cache: 15,696 sequences ×
255 rows = 4,002,480 rows. It was checked, not assumed:
- all 5,152 stored row→token anchors match;
- the pool reproduces exactly (3,795 types; 83,310 distinct; 3,197,155 rows covered).

So the pool, the member hashes and the stored L6/L7 lag deltas all remain valid. A held-out
**reserve** of the next 8,000 sequences (tokens only) supplies fresh contexts for Stages 3–4.

**Disk.**
- All 7 deep layers at once would need 18.7–21.2 GB, so they were split into two passes:
  - pass 1: L12–18 cached, plus split-half type-mean accumulators for L4–L9 and every deep
    layer;
  - pass 2: L20–24 cached.
  - The L12–18 cache was deleted after its Stage-1 analysis. The L20–24 cache was deleted after
    e42d on L20–24 (see below).
- Measured nnz/row: 119.1–122.1 at L12–18 and 124.2–125.6 at L20–24, i.e. 0.99–1.05 ×
  nominal.
- Measured project size by step:
  - pass 1: 14.45 GB (planned 14.5);
  - pass 2: 12.8 GB;
  - **overall peak: 18.3 GB (apparent size; 17.9 GB on disk)**, reached during Stage 4.
- **The peak deviated from BUDGET §8.4.** §8.4 promised to delete the L20–24 cache (5.7 GB)
  before extracting the Stage-4 occurrence sets (4.9 GB). The cache was kept so the optional
  e42d on L20–24 could still run. The peak stayed under the 20 GB cap. System free space never
  fell below 8 GB except in the swap incident below.
- Pass-1 attempt 1 was **aborted by the free-space guard**. The cause was macOS swap growth from
  my own concurrent smoke test, which had a 20 GB distance tensor, not the cache. The bug was
  fixed and the pass rerun from scratch. Logged in BUDGET §8.2.

**Weights.** Google's gated repo was downloaded once with the user's token, via an environment
variable only, never written to disk. sha256 `ee5250f6…cb27` equals the hub record.
`hf_token.rtf` was never read.

**Reproduction gate — a declared-tolerance miss, reported in full.**
- Raw strict-12 gap ratios reproduce within ±0.01 at L4–L9 (largest Δ 0.0073).
- The SAE-channel month-flag anchors miss the declared ±1e-3: L6 Δ 0.0023, L7 Δ 0.0042.

Diagnosis:
- The decode path equals script 22's cache path to 7e-6 relative.
- The deviation grows with depth on the raw channel too, which is the signature of bf16 drift
  between the August stack and torch 2.14.1 / transformers 5.19.0.
- Every order statistic reproduces exactly: strict-12 L6 gap 1.5424 and rho 0.993; candidate
  (d) exact; the verbs set's rho is 1.0 (gap 2.0795 → 2.0806).

The tolerance was borrowed from a same-cache re-read check. It was amended to ±0.01 in a dated
addendum before any deep-layer analysis ran.

---

## Stage 1 — deeper layers (script 22 unchanged; script 42 two-channel adjudication)

(L12–24 complete: script 22 at all 7 layers; two-channel adjudication on L12–18, and on L20–24 last, at the user's lowest priority.)

**Supervised control: do the known sets close in corpus type means?** (`results/e02_gate.json`)
Closure here means p_closed < 0.05 and p_bg ≤ 0.05 against 2,000 random same-size pool sets.

| layer | months, raw | months, SAE (22's representation) | weekdays, raw | numbers 1–10 (ordinal control) |
|---|---|---|---|---|
| 4 | closes (1.58) | — | closes (1.32) | open |
| 5–7 | closes (1.48–1.61) | closes (1.45–1.64) | open (1.69–1.97) | open |
| 8 | closes (2.01) | open (2.10) | open | open |
| 9 | closes (1.76) | closes (1.61) | open | open |
| 12 | closes (1.98) | open (2.10, p 0.070) | open | open |
| 14 | open (2.16, p 0.089) | open (2.04, p 0.051) | open | open |
| 16 | closes (1.86) | open (2.11, p 0.072) | open | open |
| 18 | closes (1.97) | open (2.27, p 0.135) | open | open |
| 20–24 | open (4.2–5.4) | open (5.9–6.3) | open | open |

The ordinal control never closes, so the instrument is not manufacturing rings.

The month ring survives in the raw residual through L18, closing at L12, L16 and L18. This
partly agrees with the templated closure sweep, which closed at L16 and L18–20: type means do not
close at L20. In the SAE type-mean representation that script 22 uses, it is
open at every deep layer. **Before any blind search, then, the representation script 22 reads
offers no closed month ring at depth.**

**Blind search (script 22), L12–L18** (`results/token_type_geometry_deepA.json`):
- **Months: not recovered at any layer.** The label-blind clusters do co-locate all 12 months
  in one cluster at every layer (sizes 12–20). Those clusters do not close (cluster p_closed
  0.94–1.0). The 12-month raw-arm cluster at L12 has rho 0.993 and p_closed 0.070.
- **Weekdays: not recovered.** At L12 and L14 they form exact 7-member clusters with order
  rho 1.0, but the clusters are open (p_closed 0.42 and 0.47).
- **27 distinct ring flags out of 1,260 screened clusters.** The original run used 0.05 ×
  screened (63 here) as its uncorrected reference; the flag rule is a conjunction, so its true
  null rate is lower.
  - None survives BH.
  - All are morphology or semantic-class families: buy/purchase…, what/how…,
    probably/likely…, image/photo…, child/children/woman…, summer/spring/winter (3 seasons
    plus capitalised variants), kinship nouns (k = 8, L18), and others.
  - Only 2 have k ≥ 9, which is the minimum for two-channel adjudication.

**Two-channel adjudication (script 42 functions, amendments K/L/M), L12–18**
(`results/e42d_deepA.json`):

| layer | drop rule → coordinates | strict-12 positive control | ordinal | freq-pseudo passes | class-perm passes | clean-plant joint recall at ρ = 0.15 / 0.20 |
|---|---|---|---|---|---|---|
| 12 | {1038} (share 0.88) | FAIL, order (rho 0.54) | FAIL | 0/107 | 0/200 | 0.97 / 0.74 |
| 14 | {1038} (0.77) | FAIL, LOO closure (gap 2.38, rho 1.0) | FAIL | 0/105 | 0/200 | 0.86 / 0.34 |
| 16 | {1038} (0.56) | FAIL, order (rho 0.79) | FAIL | 0/97 | 0/200 | 0.85 / 0.33 |
| 18 | {1038} (0.44) | FAIL, closure (gap 2.70) | FAIL | 0/103 | 0/200 | 0.99 / 0.69 |

- **The two-channel tier is not validated at any of L12–18.** The months' type-mean order and
  successor-delta order stop agreeing at depth, so a FAIL there carries no information.
- The 2 adjudicable flags fail anyway:
  - k = 9 occupational nouns: closure fails post-drop (3.52);
  - k = 11 child/woman/girl…: order fails, rho 0.982 against p99 0.991.
- Passers: 0. Expected false passers: about 0.01.

**Blind search (script 22), L20–L24** (`results/token_type_geometry_deepB.json`):
- **Months: not recovered.** All 12 months still co-locate in a 20-member cluster at every
  layer, unflagged. Supervised, the month ring is gone at these layers (gap 4.2–6.3).
- **Weekdays: not recovered.** They co-locate in 29–44-member clusters.
- **16 distinct ring flags out of 1,021 screened** (0.05 × screened reference: 51). None
  survives BH. All are semantic or morphological classes:
  - events/activities/sessions…;
  - intensifier adverbs (really, completely, fully, …);
  - problem/issue/situation…;
  - object and reflexive pronouns (us, me, them, him, …, ourselves, herself);
  - provided/presented/published/released…;
  - evaluative adjectives (good, best, great, …, k = 26);
  - materials (wood, glass, metal, steel, plastic, stone);
  - seasons plus capitalised variants;
  - and similar families.

**Positive-control recovery per deep layer** (blind = script 22; supervised = e02 on the same
type means):

| layer | months: blind co-location | months: blind recovered | months: supervised closure, raw / SAE (gap) | weekdays: blind co-location | weekdays: recovered | weekdays: supervised raw (gap) |
|---|---|---|---|---|---|---|
| 12 | 12/12 in a 15-cluster, unflagged | no | **closes** (1.98) / open (2.10) | 7/7 in a 7-cluster | no | open (2.19) |
| 14 | 12/12 in 19, unflagged | no | open (2.16) / open (2.04) | 7/7 in 7 | no | open (2.33) |
| 16 | 12/12 in 20, unflagged | no | **closes** (1.86) / open (2.11) | 7/7 in 34 | no | open (2.62) |
| 18 | 12/12 in 20, unflagged | no | **closes** (1.97) / open (2.27) | 7/7 in 40 | no | open (2.65) |
| 20 | 12/12 in 20, unflagged | no | open (4.42) / open (5.87) | 7/7 in 29 | no | open (2.60) |
| 22 | 12/12 in 20, unflagged | no | open (5.39) / open (6.33) | 7/7 in 30 | no | open (2.49) |
| 24 | 12/12 in 20, unflagged | no | open (4.24) / open (6.10) | 7/7 in 44 | no | open (2.30) |

The label-blind clustering always co-locates both sets. What fails is closure in script 22's
representation.

**Two-channel adjudication, L20–24** (`results/e42d_deepB.json`; run last, at the user's lowest
priority):

| layer | drop rule → coordinates | strict-12 | ordinal | freq-pseudo | class-perm | clean-plant recall at ρ = 0.15 / 0.20 |
|---|---|---|---|---|---|---|
| 20 | {1038} (share 0.31) | FAIL, closure (gap 5.71) | FAIL | 0/115 | 0/200 | 0.81 / 0.42 |
| 22 | {} | FAIL, closure (6.33) | FAIL | 0/119 | 0/200 | 0.77 / 0.52 |
| 24 | {} | FAIL, closure (6.10) | FAIL | 0/109 | 0/200 | 0.73 / 0.43 |

Of the 7 flags with k ≥ 9:
- 6 FAIL on order:
  - intensifier adverbs, L20 (rho 0.53);
  - problem/issue, L22 (0.69);
  - object pronouns, L24, k = 12 (0.57) and k = 11 (0.52);
  - evaluative adjectives, k = 26 (0.55);
  - group/team, k = 9 (0.50).
- 1 is NOT ADJUDICABLE: provided/presented/published…, L24, k = 10, rho 1.0, but in a
  saturated null cell (Addendum L1).
- **Passers: 0** (expected false: 0.07).

Note on drop rules: at L20 the lag-delta drop rule removes {1038}, whereas the type-mean fallback
used by Stage 2's L20 calibration and scan removed nothing. At L22 and L24 the two rules agree
(nothing dropped), including the L24 set Stage 4 used.

**Stage 1 summary, all seven deep layers.**
- 43 distinct closure flags out of 2,281 screened clusters. That is below the original run's
  uncorrected 0.05 × screened reference (114). The joint p_closed ∧ p_bg null is lower than that
  reference, so this is not a deficit. Nothing survives BH.
- **Months and weekdays are recovered blind at no deep layer** (0/7 each). The month ring does
  survive supervised in the raw residual at L12, L16 and L18. In the SAE type-mean
  representation that script 22 reads, it is open at every deep layer.
- The two-channel tier has **no valid positive control at depth**: strict-12 fails at all 7 deep layers. Of 9 adjudicable flags, 8 FAIL and 1 is not adjudicable; 0 pass.
- The deep scan found **no geometric candidate**. Every flag is a namable lexical or semantic
  class, and none passes an adjudicable tier.

---

## Stage 2 — lines and grids (script 46, never written before; written here as `e46`)

(calibration, templated controls and blind scan complete; the behavioural tier on the top 5 ruler candidates is below)

**Spec and amendments.** The spec is the Phase-4 plan §4 ruler/lattice, recovered verbatim from
the August transcript (`docs/stage2_ruler_lattice_spec.md`). Calibration forced dated addenda,
all written before any blind run (`docs/stage2_addendum_C_calibration_2026-10-15.md`,
`docs/stage2_addendum_C5_2026-10-15.md`):
- C1: plants anchored to real family geometry;
- C2: nuisance threshold;
- C3: graded spacing added, because the number line is log-compressed;
- C4: a shape-matched parametric null **withdrawn** in C5.

C4 failed its own null pool: on a Gaussian pool with no rulers, 275 sets passed BH. The causes
are a winner's curse on the selected statistic, and kNN balls being more regular than Gaussian
surrogates. Stage 2 is **frozen at C5**:
- a geometric candidate = the population prefilter (order, spacing even-or-graded at p ≤ 0.01,
  nuisance) ∧ cross-layer persistence;
- selection is handled behaviourally against sibling candidates.

**Calibration, L5–L9** (`results/e46_calibrate_L5-9.json`):

| layer | number words: graded spacing p_bg | number words: ρ(order, log count) | prefilter T1∧T2∧T3 | ruler plant recall, jitter 0 / 0.1 / 0.2 / 0.3 | lattice plant recall, cleanest cell (jitter 0 / 0.1) | Gaussian-null prefilter count (3 pools) | data prefilter count | data share below null p01 (even / graded, k = 6–12) | class-perm FPR |
|---|---|---|---|---|---|---|---|---|---|
| 5 | 0.00027 | −1.00 | **FAIL** (nuisance) | 0.93 / 0.92 / 0.73 / 0.48 | 0.97 / 0.60 | 275 / 277 / 277 | 241 | 0.5–0.7% / 0.6–0.9% | 0 |
| 6 | 0.00027 | −1.00 | **FAIL** | 0.94 / 0.93 / 0.81 / 0.43 | 0.87 / 0.70 | 261 / 274 / 273 | 257 | 0.3–0.5% / 0.4–0.7% | 0 |
| 7 | 0.00027 | −1.00 | **FAIL** | 0.90 / 0.94 / 0.77 / 0.40 | 0.90 / 0.70 | 274 / 281 / 274 | 235 | 0.4–0.5% / 0.3–0.9% | 0 |
| 8 | 0.00133 | −0.99 | **FAIL** | 0.96 / 0.99 / 0.76 / 0.46 | 0.73 / 0.57 | 274 / 269 / 278 | 238 | 0.1–0.6% / 0.3–0.6% | 0 |
| 9 | non-monotone | −0.96 | **FAIL** | 0.76 / 0.70 / 0.58 / 0.22 | 0.87 / 0.40 | 279 / 262 / 276 | 219 | 0.3–0.6% / 0.4–0.9% | 0 |

The plant columns average the offset multipliers 0.5, 1 and 1.5, at off-line scatter = 1 × the
number words'.

What calibration establishes:
- **The supervised positive control fails, at every early layer, for an identifiability reason
  rather than a tuning one.**
  - At L5–L8 the number words lie on a smooth, log-compressed line. At L6 their PC1 positions are
    −279, −184, −102, −43, 16, 62, 105, 128, 140, 157. That is graded spacing at p ≈ 0.0003,
    and parametrically 0 of 20,000 shape-matched surrogates are as smooth.
  - But their order is **exactly** their frequency rank (Zipf: ρ = −1.00), and nearly their
    norm rank (0.88–0.92).
  - The nuisance tier exists to reject "a direction that sorts by log count". In this pool the
    one known ruler is indistinguishable from such a direction. No threshold separates them
    without letting frequency sorts through.
  - Blind proposal does reach the family: its best set has Jaccard 0.82 (`two`…`ten` + `fourth`,
    without `one`). That set is not smooth (p_bg 0.13).
  - Military ranks (3 of 8 in the pool) and kinship (6 of 13) cannot be tested at pool level.
    The templated versions are below.
- **Planted power is real.** An 8-level ruler at the number words' own geometry is recovered
  about 90% of the time up to along-axis jitter 0.1 × spacing, and about 75% at 0.2.
- **The duplicate guard blinds the instrument to tight families.** At off-line scatter 0.5× and
  offset ≥ 1×, planted rulers are never proposed: members exceed cosine 0.90 and R15 merges them.
- **Lattice power is weak** except for clean, tight plants.
- **The false-positive floor is the design's own 1–2%.** About 270 Gaussian-null prefilter
  passes per layer against 219–257 in the data.
- **No excess regularity.** The data has *fewer* proposals beyond the Gaussian null's 1st
  percentile (0.1–0.9%) than the null itself (1%).
- **Class-permuted FPR is 0**, so the kill rule does not fire.

**Calibration, L12–L24** (`results/e46_calibrate_deep.json`).

Outlier-drop sets at depth:
- L12–18 use the lag-delta Addendum-H rule from e42d: {1038}.
- L20–24 used the declared fallback, the same rule on the pass-1 type-mean halves, because their
  lag deltas did not exist when e46 ran. It drops nothing at L20–24 (largest share 0.13–0.19).

| layer | number words: graded p_bg | ρ(order, true order) | ruler plant recall, jitter 0 / 0.1 / 0.2 / 0.3 | Gaussian-null prefilter | data prefilter | max share below null p01 | kill |
|---|---|---|---|---|---|---|---|
| 12 | 0.23 | 0.99 | 0.58 / 0.66 / 0.57 / 0.34 | 258–263 | 222 | 0.66% | no |
| 14 | non-monotone | 0.96 | 0.49 / 0.49 / 0.39 / 0.28 | 262–277 | 243 | 0.72% | no |
| 16 | non-monotone | 0.42 | 0.62 / 0.60 / 0.50 / 0.43 | 266–272 | 226 | 0.91% | no |
| 18 | non-monotone | 0.96 | 0.62 / 0.64 / 0.57 / 0.28 | 268–280 | 226 | 0.85% | no |
| 20 | non-monotone | 0.95 | 0.56 / 0.64 / 0.56 / 0.29 | 260–283 | 221 | 0.74% | no |
| 22 | non-monotone | 0.94 | 0.59 / 0.58 / 0.60 / 0.29 | 274–278 | 221 | 0.85% | no |
| 24 | non-monotone | 0.79 | 0.62 / 0.58 / 0.60 / 0.40 | 279–287 | 210 | 0.65% | no |

At depth the number words keep their PC1 order (ρ 0.79–0.99, except L16), but they are no
longer monotone on the held-out half, so even the graded tier cannot see them. Plant recall drops
to about 0.6. There is again no excess over the Gaussian null at any layer.


**Templated supervised controls** (`e46s` + `e46 --mode templated`, `results/e46_templated.json`).
Each word is read at its last token in 16 neutral templates, split 8/8 into halves. The null is
300 random pool words through the same templates.
- **Military ranks have no order at all** along their own PC1: ρ with true rank is −0.14 to
  +0.12 at every layer from L5 to L24. They cannot serve as a ruler control in this model.
  Phase 2 found the same with its own templates and called it polysemy.
- **Number words are ordered** (ρ 0.88–0.99 at most layers) but rarely smoothly spaced on the
  held-out templates. Graded p_bg is 0.017 (L5), 0.023 (L8) and 0.009 (L20); every other layer
  is non-monotone.
- **Kinship has real factorial structure, which the lattice tier cannot see.**
  - The supervised additive model (generation × gender) explains R² = 0.26–0.40 of the top-5-PC
    variance, permutation p = 0.001–0.034 at every layer.
  - The blind parallelogram statistic is grid_B ≈ 0 (p_bg 0.31–1.0).
  - **The lattice tier fails its supervised control.**

**Reading before the blind run.** Stage 2 has power against planted rulers but **no validated
positive control** in this model and pool. Under the user's freeze, no further addendum is made.
The blind scan therefore runs as a **descriptive** measurement, and nothing it flags can rise
above "geometric candidate of an unvalidated instrument" without the behavioural tier, whose own
anchor is the number words.

**Blind scan, descriptive only (frozen at C5)** (`results/e46_blind_L5-9.json`,
`results/e46_blind_deep.json`):

| layers | ruler prefilter per layer | Gaussian null prefilter per layer | ruler geometric candidates (prefilter ∧ persistence) | lattice geometric candidates |
|---|---|---|---|---|
| L5–L9 | 219–257 | about 270 | 549 | 141 |
| L12–L24 | 210–243 | about 270 | 334 | 23 |

- The candidate count is the instrument's own 1–2% tail, which the data matches or undershoots.
- The top-ranked "rulers" are mostly unordered semantic families: pronouns; citizens/residents/
  victims; tech nouns; dash and ellipsis punctuation; office/agency/department. There is one
  plausible graded scale: younger, young, age, older, old, ancient (L7, k = 9).
- The top lattice is a **named** structure: compass directions × capitalisation (South, North,
  West, East with lower-case forms; L7, k = 8), which is the case-difference vector crossed with
  direction. Next come suffix lattices (es/ans/ens/ons…).
- The withdrawn C4 "BH" column passes nearly every prefiltered set in the data, just as it did on
  the null pool. It is reported only to show it is uninformative.

**Behavioural tier for ruler candidates** (`scripts/e47r_ruler_behaviour.py`,
`results/e47r_ruler_behaviour.json`). Its specific rules live in the script's docstring, written
before it ran, not in a dated addendum:
- interior members only;
- reserve contexts 2000–7999;
- 50 random k-sets and 10 random directions;
- siblings drawn from the same layer and k;
- the number-word anchor first.
- Rig: 51's rig on a line. Interior members are steered along the ruler axis, and the readout
  is logit(next level) − logit(previous level).
- Contexts: reserve sequences 2000–7999.
- Every arm at 40 prompts.
- **The anchor, number words at L6, FAILS**: mean Δ@0.02 = +0.042, sign consistency
  0.63/0.75/0.75 over the 8 interior members, random-set p99 +0.154, random-direction maximum
  +0.017.
- Under the C5 rule **no behavioural verdict is issued on any ruler candidate**.

**Stage 2 verdict: the instrument is not validated.**
- Geometrically, its only available ruler control is frequency-confounded.
- Its lattice control (kinship) is invisible to it.
- Behaviourally, its anchor fails.
- It does have measured power against planted rulers at realistic geometry (about 90% at small
  jitter).
- The blind scan shows no excess over the Gaussian null.
- **No candidate from Stage 2 rises above "geometric candidate of an unvalidated instrument".**

---

## Stage 3 — discovery-verbs confirmatory retest: **FAIL, retired**

Preregistered in `docs/PREREG_verbs_retest.md` (sha256 `6671d099…`, asserted at start and end).
Run by `scripts/e53_verbs_retest.py`, which imports script 51's rig and enumeration. Output:
`results/e53_verbs_retest.json`.

**Set.** L7, k = 10: find, showed, discover, identified, identify, revealed, confirmed,
recognized, realize, noticed.

**What changed from Phase 5.**
- Every context is an 8-token slice of the **8,000 held-out reserve sequences**, which no type
  mean or earlier test used.
- **Every arm runs at 40 prompts per member and the full ε grid**: candidate, anchor, 10 random
  orthogonal directions, ±90° rotations, 100 random k-sets, and **all 30 siblings**.
- The siblings are the candidate's own proposer route (channel-2 k = 16 assembly → R15 0.92 →
  P1 guard → closers). It was rebuilt with 51's code and matches 51's stored hashes 30/30.

**Geometry recheck.** On the rebuilt means the verbs set has gap 2.081 (stored 2.079) and
two-channel rho 1.0. strict-12 at L6: 1.542 / 0.993, identical to stored.

| set | mean Δ at ε = 0.005 / 0.01 / 0.02 | sign consistency | random-set p99 | max random direction | sibling p99 | verdict |
|---|---|---|---|---|---|---|
| strict-12 months, L6 (anchor gate) | +0.082 / +0.170 / **+0.343** | 12/12 at every ε | +0.191 (50 sets) | +0.035 | — | **PASS** (gate open) |
| discovery verbs, L7 | +0.065 / +0.129 / **+0.262** | 10/10 at every ε | +0.237 (100 sets) | +0.032 | **+0.286** (30) | **FAIL** |

**Verdict: FAIL (not confirmed), and the candidate is retired.**
- The verbs beat random sets and random directions.
- Their sign consistency and monotone ε response replicate on fresh contexts.
- But they sit at **empirical rank p = 0.065 among their 30 proposer-matched siblings**. The
  best sibling is a speech-verb set (report, talk, told, article, meeting, interview, announced,
  speak, chat, spoke) at +0.286, and 1 of the 30 siblings passes the same rule on its own.

The behavioural effect is what the proposer's family produces in general, not something specific
to this set. It also still fails the geometric LOO audit (Phase 5: rho_min 0.517). No rescue
analyses were run.

---

## Stage 4 — identity-removed per-occurrence search: **positive control FAILED; Stage 4 stopped as preregistered**

`PREREG_occurrence.md` was frozen before any D, S or R data existed (sha256 `58772029…` in
`docs/HASHES.txt`, asserted by `e61` at every phase). Addendum S1
(`docs/stage4_addendum_S1_screen_sweep_2026-10-15.md`) was written before the sweep it governs.

Scripts:
- `e60_extract.py`: per-occurrence SAE-decoded residuals;
- `e61_occurrence_screen.py`: representation, units, screens, arrangements, positive control,
  labels;
- `e61s_screen_sweep.py`: Addendum S1;
- `e62_occurrence_behaviour.py`: the behavioural tier, written but never run, because the gate
  failed.

### Design (as frozen)

**Representation.** Per occurrence: the SAE-decoded residual (l0_big 16k) minus its own token
type's mean in the same set (identity removed), with outlier coordinates dropped. Then document
centring, which removes topic, and position deflation, which removes the BOS-distance code.
Finally PCA to 64 dimensions, whitened.

**Units (blind).** k-means with K = 128, and FastICA with 32 components.

**Screens:**
- cluster size, type diversity and document spread;
- split-half reproducibility;
- token-driven index (TDI): how often a same-type random occurrence falls in the same unit,
  relative to the base rate;
- ICA bimodality, η² on token type, and reproducibility;
- presence in the context-shuffled set S, reported only.

**Arrangements.** Ring and ruler screens among screened centroids, with false flags measured on S.

**Measured false-flag rate (S, context-shuffled, all of S's sizeable centroids, about 120 per
layer).**

| k | L7 ring / ruler flags (out of ~118–123 proposals) | L12 ring / ruler flags | L24 ring / ruler flags |
|---|---|---|---|
| 5 | 14 / 6 | 11 / 0 | 5 / 2 |
| 6 | 7 / 3 | 6 / 0 | 4 / 0 |
| 7 | 5 / 2 | 7 / 2 | 2 / 0 |
| 8 | 6 / 0 | 9 / 1 | 1 / 1 |

That is 1–12% per k. On D the arrangement screen never ran, because fewer than 8 centroids
passed the unit screens.

**Positive control.** Blind recovery of `paren_inside` or `quote_inside`. It requires F1 ≥ 0.5
or AUROC ≥ 0.85, above the p99.9 of a 1,000-permutation token-matched null.

**Layers.** L7 (the anchor) plus L24 and L12 by the Stage-1 flag-count rule.

**Data.** Four disjoint sets of 1,000 sequences: P (pilot), D (discovery), S (D shuffled
within sequences) and R (reserve, held out).

### Results

**The representation is not the failure point** (P only). A supervised linear probe on the final
representation decodes `paren_inside` with held-out AUROC **0.825** (0.832 in the whitened PCA-64
space), and `quote_inside` with **0.826**. Both in-context variables survive identity removal,
document centring and position deflation. No single direction carries them: the best single
whitened PC reaches 0.68 and 0.59.

**The unit extraction is the failure point.**

| set / step | screened clusters | screened ICA | best paren F1 / AUROC (any unit) | recovered |
|---|---|---|---|---|
| P pilot, 12 settings (K × ICA × PCA) | 0–1 | **0 at every setting** | — | 0/12 |
| P, Addendum S1 sweep, 216 combinations (TDI cap to ∞, reproducibility to 0, ICA screens off) | 0–136 | 0–32 | **0.247 / 0.69** (screens off) | **0/216** |
| D, L7 | 1/128 | 0/32 | 0.023 / — | no |
| D, L12 | 0/128 | 1/32 | — / 0.513 (null 0.516) | no |
| D, L24 | 1/128 | 0/32 | 0.022 / — | no |

Floors: F1 ≥ 0.50 or AUROC ≥ 0.85.

The D runs used K = 128, 32 ICA components and PCA 64, chosen because nothing recovered on P.
`quote_inside` fails identically.

Per-screen pass counts on D (L7 / L12 / L24):

| screen | L7 | L12 | L24 | distribution |
|---|---|---|---|---|
| cluster TDI ≤ 3 | 2 | 2 | 1 | median TDI 8.1 / 8.5 / 7.2, q90 19–34 |
| cluster split-half reproducibility ≥ 0.8 | 70 | 70 | 73 | median 0.84–0.86 |
| ICA bimodality > 0.555 | 0 | 1 | 0 | q90 0.25–0.27 |
| ICA reproducibility ≥ 0.8 | 0 | 1 | 1 | median 0.37–0.39 |
| context-shuffled presence | — | — | — | median cosine 0.36–0.50 |

The last row means most D clusters do *not* appear in shuffled text, so they do depend on natural
context. My advance note expected this screen to pass everything. It did not, and it was
informative.

**Reading.**
1. **Subtracting the type mean removes token identity from the first moment, not the second.**
   Identity-removed occurrences still cluster by token type: a random same-type occurrence lands
   in the same k-means cluster at a median of 7–8.5× the base rate. The *shape* of each type's
   contextual scatter is type-specific, so unsupervised partitions recover "which token family
   this is" before any in-context variable.
2. **Known in-context variables are linearly present but distributed.** The paren and quote
   states are decodable by a supervised probe (AUROC 0.83), yet no k-means cell or ICA source
   aligns with them: the best is F1 0.25 or AUROC 0.69 even with every screen removed. ICA
   components are also unstable across halves (|cos| ≈ 0.4).
3. Under the frozen rule, **the instrument is not validated**. No unnamed unit, arrangement or
   behavioural claim is made. The arrangement screens never ran on D: fewer than 8 screened
   centroids existed at any layer. R was extracted but never touched. e62 was not run.

**What would have to change** (not run; a design note for a future preregistration):
- Second-moment identity removal: whiten each token type's residual by its own covariance, or
  restrict to occurrences of high-count types with type-specific whitening.
- A supervised-then-blind design: probe-guided subspaces such as sparse dictionary learning over
  many probes, rather than k-means or ICA on the raw identity-removed cloud.

Neither is in this run.

---

## Stage 5 — scoping `gemma-3-4b-pt` (estimate only; nothing was run)

**Which instrument.** Only one chain in this campaign both recovered a known structure blind and
had its behavioural anchor validated on fresh contexts: the script-22 type-mean ring screen, then
the two-channel tier (script 42), then the 47-style behavioural rig.
- The months were recovered blind at L6/L7 in the original campaign.
- In Stage 3, the 47 anchor passed on reserve contexts at a matched budget (+0.343 against a
  random-set p99 of +0.191).

Stage 2 (rulers/lattices) and Stage 4 (occurrences) each failed their own positive controls. They
are not worth porting as they stand.

**Facts checked against the hub (2026-10-15).**
- `google/gemma-scope-2-4b-pt` (ungated) has `resid_post_all` SAEs for **all 34 layers**, at 16k
  and 262k, small and big.
- 16k files are 335,686,016 B each, which implies d_model = 2560.
- `l0_big` is 91 at L6 and 120 at L12/L20/L28.
- `gemma-3-4b-pt` safetensors total 8.60 GB, including the vision tower. The gated repo would
  need the token once.
- Gemma 3 shares one tokenizer, so **the identical 4M-token corpus and 3,795-type pool carry
  over**, giving a direct cross-scale comparison.

**Measured at 1B on this machine (M4 Pro, 24 GB, MPS).**
- Forward of 16 × 256 tokens: 1.45–1.5 s.
- CSR write: 0.2 s per cached layer per batch.
- Pass 1: 2,967 s (4 cached layers plus 25 accumulators).
- Pass 2: 2,126 s (3 cached layers).
- Script 22: about 150 s per layer.
- e42d: 12–28 min per layer.
- Stage-3 retest: 2,074 s, about 320k short forwards.

**Scaling.** Non-embedding parameters go from about 0.7B (1B model) to about 3.2B (4B model):
about 4.6× FLOPs per token, so roughly 7 s per 16 × 256 batch, or about 600 tok/s.

**Recommended design at 4B: cache-free.**
- The sparse cache exists only so that script 22 and the lag deltas can re-read codes.
- At 4B both can be accumulated in the forward pass instead:
  - pass A: split-half raw and SAE-decoded type means;
  - pass B: lag+1 successor sums against pass A's means.
- That removes the 2 GB-per-layer cache, which would no longer fit beside an 8.6 GB model under
  a 20 GB cap.

Six layers at the same relative depth as 1B L5–L9 (4B L6–L12):

| item | estimate |
|---|---|
| **Disk peak** | model 8.6 + 6 SAEs 2.0 + venv 1.0 + type means and deltas 1.0 + results ≈ **13 GB**. With the sparse cache it would be 3 cached layers per pass at about 18 GB. |
| **Unified memory** | bf16 weights about 8.6 GB + 6 fp32 SAEs 2.0 GB + activations and codes about 1 GB on MPS + CPU accumulators about 2–4 GB ≈ **14–16 GB**. That is feasible only with the machine otherwise idle; an unrelated 8.5 GB job was resident at times in this run. |
| **Time: passes** | 2 × about 2.1 h ≈ 4.2 h |
| **Time: script 22** | about 20 min |
| **Time: two-channel nulls** | d = 2559, so about 2× per draw: 25–50 min per layer, 2.5–5 h serial, about 1.5 h on 3 processes |
| **Time: behavioural** | anchor gate about 1 h; each candidate at full null budgets about 2.5 h |
| **Total** | **≈ 9–13 h wall-clock for a 6-layer scan with one behavioural adjudication. No cloud needed.** |

**Not recommended at 4B as they stand.**
- **Stage 4.** Four sets × 3 layers × 1,000 sequences at d = 2560 is about 15.7 GB of fp16
  occurrences, and its instrument failed its positive control at 1B. Fix the
  second-moment identity removal first.
- **Stage 2.** No validated positive control.
- **The deep two-channel tier.** The months' two-channel agreement fails at 1B depth, so there
  is no anchor.

---

## Every candidate and the tier it reached

Tiers: **geometric candidate** = passes its own geometric nulls; **behavioural pass** = 47/47b-style
steering beats a sibling null at a matched budget; **replicated** = holds on fresh contexts.
"Discovery" requires all three.

| stage | candidates | geometric | behavioural | replicated | named / new |
|---|---|---|---|---|---|
| 1 (deep L12–24) | 43 distinct closure flags (2,281 screened; reference 0.05 × screened = 114) | **0** pass an adjudicable tier. 9 have k ≥ 9: 8 FAIL; 1 (provided/presented…, L24) is not adjudicable (saturated cell). The two-channel tier has no valid anchor at depth. | — | — | all namable lexical/semantic classes |
| 2 (lines/grids) | 883 ruler and 164 lattice sets pass prefilter ∧ persistence, at or below the null rate | "geometric candidate of an **unvalidated** instrument" only | **no verdict**: the anchor (number words) failed | — | the top lattice is compass × capitalisation (named); the top rulers are unordered families; one plausible age scale (young…old…ancient), untested behaviourally because the anchor failed |
| 3 (discovery verbs, L7) | 1 | closure, two-channel rho 1.0 and frequency arm pass; **LOO fails** (Phase 5) | **FAIL**: beats random sets and directions, rank p = 0.065 among its 30 siblings | — (fresh contexts used for the behavioural test itself) | **retired** |
| 4 (identity-removed occurrences) | 0 | instrument **not validated**: the positive control failed at L7, L12 and L24 | not run (gate) | R never touched | — |

## Bottom line

**No.** At 1B scale, none of the four extensions finds a structure nobody has named. More usefully,
each one fails at a measured, specific point rather than for lack of trying.

1. **Deeper layers.**
   - The month ring survives supervised in the raw residual through L18.
   - But the SAE type-mean representation that script 22 reads opens at every deep layer, and the
     two-channel tier loses its anchor at depth.
   - The blind scan's 43 flags, below the original 0.05 × screened reference of 114, are all
     lexical classes, and none survives BH.
   - Depth adds no new geometry to the type-mean instrument, and it removes the one known ring
     the instrument could see.
2. **Lines and grids.**
   - The instrument has real power against planted rulers (about 90% at small jitter).
   - In this pool, though, the only real ruler (number words, a smooth log-compressed line at
     L5–L8) is ordered *exactly* by frequency (ρ = −1.00), so no nuisance rule can certify it.
   - The real kinship lattice (supervised R² up to 0.40, p ≤ 0.03) is invisible to the
     parallelogram statistic.
   - The behavioural anchor fails, and there is no excess regularity over a Gaussian null.
   - A shape-matched parametric null I designed (C4) failed its own null-pool check and was
     withdrawn before the blind run.
3. **The discovery-verbs near-miss** is retired by the preregistered retest. Its steering
   effect is real against random sets but typical of its proposer family.
4. **Identity-removed occurrences.**
   - Known in-context variables are linearly present after identity removal (paren/quote probe
     AUROC 0.83), so the representation is not the bottleneck.
   - Unsupervised units (k-means/ICA) do not isolate them.
   - Mean subtraction leaves token identity in the second moment: identity-removed clusters
     re-collect token families at 7–8.5× chance.
   - The instrument fails its positive control, so no blind claim can be made.

**What this changes relative to the August answer** ("in principle yes, in practice it has only
re-found named ones"):
- the deep, line/grid and per-occurrence directions are now measured, not merely unexplored;
- each has a quantified blind spot;
- the one open lead is closed.

**What would change the answer** (design notes, not run):
- per-type covariance whitening, or a probe-guided dictionary, for occurrence-level search;
- a frequency-matched ruler test, comparing within frequency strata;
- the 4B scan in Stage 5 (about 9–13 h local, no cloud).

## Open items

- e42d on L20–24 ran last and is reported in Stage 1: 0 passers.
- Caches deleted at the end; the final footprint is in BUDGET §8.6. Kept:
  - type means (`typemeans_pass1.npz`) and the rebuilt fulln artifact;
  - lag deltas L12–24;
  - corpus and reserve tokens;
  - every result JSON.

  Regenerable and deleted:
  - the sparse cache (`e01_pass.py`, about 35 min per pass);
  - the occurrence sets (`e60_extract.py`, about 3 min per set).
- The Stage-2 lattice branch and Stage-4 redesigns need new preregistrations before any use.
- `~/Downloads/gf-discovery` was not modified.

---

## Stage 4 v2 — identity removed from both moments, full-rank ICA: **positive control PASSED at L7 and L12**; stopped at the gate as instructed

**Chain of documents, each hashed in `docs/HASHES.txt` before the step it governs:**
1. `docs/stage4v2_calibration_plan_2026-10-15.md`, before any v2 statistic;
2. addendum A1 (ICA dimension), after an L7 smoke run;
3. addendum A2, after the full P2 pilot and the P2 diagnostics;
4. **`PREREG_occurrence_v2.md`** (sha `edf75750…`), frozen at 11:28 *before* D2 was extracted
   (11:28–11:31);
5. the D2 freeze hashes, recorded before any D2 label was read;
6. the control.

**Scripts:**
- `e63_occurrence_v2.py` (sha `6f822a43…`, asserted by the prereg);
- `e63d_pilot_diagnostics.py` (P2 only);
- `e63r_d2_identity_descriptives.py` (post-verdict, label-free).

**Data** (1,000 sequences each; all disjoint from v1's sets):

| set | content | use |
|---|---|---|
| P2 | corpus 13,000–13,999 | calibration only |
| D2 | corpus 1,000–1,999 | units, screens, control |
| R2 | reserve 2,000–2,999 | not extracted |

### First, a correction to v1

v1's account of its own failure is **incomplete, not wrong**. Two measurement defects sat beside
the second-moment explanation.

1. **v1's ICA searched a subspace chosen by noise.**
   - `FastICA(n_components = 32)` first projects its input onto its top 32 principal components.
   - v1 passed it input that had already been whitened (PCA-64), where every eigenvalue is about
     1. The 32-dimensional search space was therefore set by sampling noise, not by the data.
   - This is why v1's ICA units were unstable, with split-half reproducibility of 0.37–0.39.
   - At full rank (64 components) the same quantity is 0.74–0.91.
2. **v1's TDI counted self-matches.**
   - The same-type "replacement" could be the row itself.
   - On P2, v1's representation gives a median TDI of 7.7 / 7.1 / 7.4 at L7 / L12 / L24 counted
     v1's way, and **5.8 / 4.7 / 5.4 with the row excluded**. On D2 the figures are
     8.7 / 7.3 / 7.4 against 6.9 / 5.2 / 5.8.
   - Identity really was left in the second moment, but by about 5–7×, not 7–8.5×.
3. **v1's floors could not be met.**
   - At K = 128 a cluster's F1 is capped at about 0.41 by size alone.
   - The ICA AUROC floor (0.85) sat above v1's own supervised probe ceiling (0.83).
   - v1's D is **not** re-run as an erratum. Its floor was above its own probe ceiling, so its
     verdict cannot change.

**P2 ablation: which change mattered.** Max screened paren |AUROC|, logcosh ICA-64:

| layer | B1: iterated first moments, no per-type step | v2: per-type whitening |
|---|---|---|
| L7 | 0.774 | 0.774 |
| L12 | 0.604 | 0.718 |
| L24 | 0.573 | 0.554 |

- **At L7 the full-rank ICA fix alone recovers paren.**
- Per-type whitening is what lifts L12 paren (0.604 → 0.718).
- Per-type whitening is also what takes identity out of the geometry. On P2, from B1 to v2:
  - K = 128 TDI 6.8 / 5.0 / 5.2 → 1.5 / 1.5 / 1.7;
  - KIR 39 / 30 / 26 → 14 / 9 / 9;
  - ICA η²(y²) about 0.10 → 0.035.

### Design (as frozen)

**Representation (D2's own rows only):**
1. Type, document and position means are removed, and the cycle is repeated three times.
2. PCA-64, whitened.
3. Per-type covariance equalisation:
   - types with ≥ 32 rows use Σ̂_t = (1−λ)S_t + λS_pool, with λ = 16/(n_t + 16);
   - rarer types use S_pool.
4. Global re-whitening.

N_MIN = 32 and n0 = 16 were chosen independently at every layer, as the lowest TDI under a
paren-probe guard.

**Units:**
- k-means, K = 16;
- FastICA-64, logcosh contrast;
- FastICA-64, skewness contrast.

**Screens:**

| unit kind | screens |
|---|---|
| clusters | v1's size, type, share and document screens; split-half reproducibility ≥ 0.8; self-excluded TDI ≤ 3 |
| ICA | η²(token) of y ≤ 0.3; **η²(token) of y² ≤ 0.3** (the second-moment identity screen; the first-moment one is vacuous by construction); split-half \|cos\| ≥ 0.8; bimodality off |

**Floors:**
- Clusters: F1 ≥ 0.50.
- ICA: AUROC ≥ clip(0.5 + 0.75·(probe_P2 − 0.5), 0.70, 0.85), which is **0.70 at every layer**.
- Each floor also requires exceeding the token-matched permutation null at p99.9 (1,000
  permutations).
- 18 looks, so the family-wise false recovery is ≤ 1.8%.

**PASS** = either label recovered at one or more layers.

- paren_inside was tuned on P2. quote_inside was never used in any choice.
- The frozen prereg states that L24 cannot pass: its 0.70 floor is above the P2 probe ceiling
  of 0.653.
- The prereg does **not** separately flag L12's thin pilot margin (P2 0.718 against 0.70); I
  note it here. On D2 the margin is 0.036.

### Results on D2

**Positive control.** "✓" marks a recovered unit.

| | L7 | L12 | L24 |
|---|---|---|---|
| **TDI at K = 128**, v1 representation (self-excluded / v1 self-inclusive) → v2 | 6.85 / 8.71 → **1.76** | 5.23 / 7.30 → **1.57** | 5.85 / 7.40 → **1.71** |
| **KIR**, v1 → v2 | 40.0 → 14.7 | 31.7 → 9.3 | 28.6 → 9.6 |
| TDI of the frozen K = 16 clusters (median) | 1.01 | 1.01 | 1.08 |
| held-out probe on v2, paren / quote (context only) | 0.787 / 0.723 | 0.761 / 0.765 | 0.636 / 0.691 |
| **paren**, logcosh \|AUROC\| (null p99.9) | **0.779** (0.519) ✓ | **0.736** (0.520) ✓ | 0.558 (0.516) |
| **paren**, skew \|AUROC\| (null p99.9) | **0.777** (0.518) ✓ | **0.732** (0.517) ✓ | 0.552 (0.516) |
| **quote**, logcosh \|AUROC\| (null p99.9) | 0.621 (0.512) | **0.747** (0.515) ✓ | 0.587 (0.512) |
| **quote**, skew \|AUROC\| (null p99.9) | 0.654 (0.512) | **0.745** (0.513) ✓ | 0.568 (0.512) |
| clusters, best screened F1, paren / quote (floor 0.50) | no screened cluster | 0.034 / 0.413 (null 0.041 / 0.061) | no screened cluster |
| **layer verdict** | **PASS** | **PASS** | FAIL (expected: floor above the probe ceiling) |

**Overall: PASS.** The instrument is validated at **L7**, for paren with both ICA kinds. It is
validated at **L12** for paren and for the untuned quote control, with both ICA kinds. It is not
validated at L24, and the cluster kind is validated nowhere. With every screen off, the ICA
maxima equal the screened ones in every recovering cell. The largest rise is L7 quote, logcosh
0.621 → 0.680, still below 0.70. Unscreened cluster F1 is ≤ 0.11, except quote at L12 (0.413).

**Per-screen pass counts (D2; L7 / L12 / L24).**

| unit kind (number) | screen | passing |
|---|---|---|
| clusters (16) | size, types, share, documents | 16 / 16 / 16 |
| | TDI ≤ 3 | 16 / 16 / 16 |
| | split-half reproducibility ≥ 0.8 | **0 / 2 / 0** |
| | **all screens** | **0 / 2 / 0** |
| logcosh ICA (64) | η²(y) ≤ 0.3 | 64 / 64 / 64 |
| | η²(y²) ≤ 0.3 | 64 / 64 / 64 (max 0.16 / 0.14 / 0.08) |
| | reproducibility ≥ 0.8 | **35 / 36 / 35** (median 0.86 / 0.87 / 0.84) |
| | **all screens** | **35 / 36 / 35** |
| skew ICA (64) | η²(y) ≤ 0.3 | 64 / 64 / 64 |
| | η²(y²) ≤ 0.3 | 64 / 64 / 64 (max 0.09 / 0.09 / 0.08) |
| | reproducibility ≥ 0.8 | **30 / 42 / 41** (median 0.78 / 0.91 / 0.85) |
| | **all screens** | **30 / 42 / 41** |

Bimodality is off. Under the v1 threshold it would pass 0 units at every layer.

**Specificity of every recovering unit (D2).** The full association tables (every battery feature,
with base rates) are in `results/e63_occurrence_v2.json` → `positive_control.<label>.<kind>.specificity`.

| unit | target \|AUROC\| | top battery association | top-1% composition (base rate) |
|---|---|---|---|
| L7 logcosh #12 (paren) | 0.779 | paren_inside | **68.9% paren** (3.1%), 15.7% quote (6.0%), 1.8% bracket (0.24%) |
| L7 skew #62 (paren) | 0.777 | paren_inside | **63.9% paren**, 23.6% quote, 1.7% bracket |
| L12 logcosh #11 (paren) | 0.736 | paren_inside | **50.7% paren**, 6.3% quote, 1.1% bracket |
| L12 skew #23 (paren) | 0.732 | paren_inside | **49.2% paren**, 5.6% quote, 0.9% bracket |
| L12 logcosh #33 (quote) | 0.747 | quote_inside | **91.2% quote** (6.0%), 0.8% paren |
| L12 skew #33 (quote) | 0.745 | quote_inside | **89.8% quote**, 1.0% paren |

- The next associations are all rare or weak:
  - in_url |AUROC| 0.67–0.68 at L7, on a 0.07% feature whose top-1% rate is about 0 (those rows
    sit in the opposite tail);
  - after_newline 0.55–0.63;
  - every other feature ≤ 0.58.
- **Sampled contexts:**
  - the L7 paren units fire inside parenthetical asides ("(Fun fact: …", "(I use …)",
    "(he's a Pats fan - boo!)");
  - the L12 paren unit also fires inside comma-delimited asides ("recovering, yet still
    fragile,");
  - the L12 quote unit fires inside quoted speech.
- **On P2** the best L7 paren unit's top association was bracket_inside (0.84, on a 0.10%
  feature), and its top 1% was 61% paren and 22% quote. The L7 direction is best described as
  **enclosure/aside state, dominated by parentheses**, not as a paren-specific detector. The
  battery does not count curly single quotes, which appear in its top contexts.

### Reading

1. **The v2 instrument isolates known in-context variables blindly.** (Corrected 2026-10-24, see
   the top of this report: the units are extracted without labels, but the settings were tuned
   with parenthesis labels on P2; quotation is the untuned control.)
   - The representation removes token identity from the first moment and largely from the
     second. Full-rank ICA then returns reproducible, identity-screened directions.
   - Among those directions, two at L7 and two at L12 (one per ICA contrast) track
     parenthesis state, beyond a token-matched null with a large margin (about 0.78 against
     0.52).
   - At L12 another tracks quotation state; this is the untuned control.
   - A single direction carries **90–97% of the held-out supervised probe's excess over
     chance** on the same representation: 0.779 against 0.787 at L7, 0.736 against 0.761 for
     paren at L12, and 0.747 against 0.765 for quote at L12.
2. **How strong the claim is.**
   - **AUROC 0.73–0.78 means a unit isolates most of the linearly decodable signal. It is not a
     clean detector:** top-1% precision is 50–69% for paren and about 90% for quote.
   - paren was tuned on P2. The independent evidence is quote at L12 and the replication of the
     P2 numbers on disjoint D2: 0.774 → 0.779 at L7, 0.718 → 0.736 at L12.
   - The cluster kind contributes nothing. After per-type whitening the cloud is close to
     isotropic, and k-means partitions do not reproduce.
3. **Identity is reduced, not gone.**
   - The TDI of the frozen K = 16 clusters is about 1.0, and K = 128 TDI is 1.6–1.8.
   - But kNN identity (KIR) is still 9–15× chance in 64 dimensions.
   - Plausible sources:
     - types below 32 rows (about 20% of rows), which keep the pooled covariance;
     - type-specific higher moments.
   - Any discovery tier must keep the η²(y²) and TDI screens and report KIR.
4. **L24 stays dark.** The probe ceiling for paren is 0.64–0.65, and no unit passes there.

**Not run (stopped at the gate, as instructed):**
- S2 (context-shuffled);
- arrangements;
- unit labelling beyond the control units;
- the behavioural tier;
- R2.

**The discovery tier needs its own preregistration first.** It should extend the battery with
curly single quotes and a generic enclosure/aside feature *before* any discovery data exists, so
that units like the L7 one are named correctly.

**Resources.**
- Two extractions (P2, D2) of about 3 min each on MPS.
- The CPU work totalled under 30 minutes:
  - pilot 9 min;
  - diagnostics 2.5 min;
  - freeze 1.5 min;
  - control 4.5 min;
  - the D2 descriptives and the dry runs, a few minutes each.
- Swap grew from about 6.3 to about 13 GB across the MPS extractions; disk stayed ≥ 33 GB free.
- The folder peaked at about 10 GB, under the 20 GB cap.
- The P2 and D2 occurrence caches were deleted at the end. Units and results are kept.
- **Final folder size: 6.7 GB** (results 2.1 GB, of which the v2 unit files are 0.29 GB).

---

## Stage 4 v2, discovery tier: **no discovery**; both behavioural anchors failed

**Chain of documents, each hashed in `docs/HASHES.txt` before the step it governs:**
1. `PREREG_occurrence_v2_discovery.md` (sha `1e24fcdf…`), frozen before any discovery-tier data
   existed;
2. addendum D1 (`docs/stage4v2_discovery_addendum_D1_2026-10-15.md`, sha `6082dc9b…`): swap guard
   raised from 8 to 16 GB at the user's request; no analysis effect;
3. `scripts/e64_occurrence_v2_discovery.py` (D1 version, sha `1c97e42b…`);
4. then each phase's output, in order: reconstruct, labels, context reading, S2, behaviour, R2,
   verdict.

**Scope.** L7 and L12; ICA-64 logcosh and skew; the 143 frozen screened v2 units (65 at L7, 78 at
L12). No clusters, no L24, no arrangements.

**Data:**
- D2 was re-extracted.
- S2 is D2 shuffled within each sequence.
- R2 is reserve 2,000–2,999.
- The steering sites came from reserve 3,000–3,999.

### Reconstruction

**The frozen units were reproduced exactly.**
- D2 was re-extracted, and its representation was recomputed with the frozen code.
- Every frozen unit's scores are a linear readout of the recomputed W, with **R² = 1 − 4·10⁻¹⁴**.
- The kept maps reproduce `e63.type_whiten` to 0.0.

### Labelling: battery, then the context reading

| | L7 | L12 |
|---|---|---|
| screened units | 65 | 78 |
| battery-UNNAMED (extended battery, §6.1) | 40 | 51 |
| NAMED_BY_CONTEXT (analyst reading, §6.2) | 6 | 20 |
| **discovery candidates** | **34** (22 families) | **31** (21 families) |

The six control units all came out NAMED, as the rule required.

**The reading standard.** It was recorded in `results/e64_context_reading.json` and sealed
before any later phase.
- A unit is context-named when ≥ 8 of the 12 contexts in one tail share one specific
  construction.
- Twins are read as a family.
- Broad categories do not count.

**What the reading found.** Battery-unnamed units that are really **syntactic-position
variables**, which the battery lacks.

| layer | constructions |
|---|---|
| L7 | the token right after a coordinating conjunction; right after "in"/"at"; inside a temporal clause (when/before/after) |
| L12 | inside a negation's scope; inside the subject NP before its verb; inside a fronted adverbial; inside a copular predicate; inside a partitive/superlative NP; inside the second conjunct of "X and Y"; the onset of the next item in a series; inside an enumeration; inside a temporal expression; the token right after "the"; the first clause of a line |

### S2: context-shuffled presence and yield

**Presence.**
- Only **4 of 65** (L7) and **3 of 78** (L12) D2 units reappear in S2 at |cos| ≥ 0.8. The chance
  p99.9 is about 0.50.
- **The control units are not present in S2** (cos 0.27–0.60). Their paren and quote directions
  need natural word order.

**Shuffled-text yield** (screened and battery-UNNAMED on S2; S2 had no context reading, so the
like-for-like D2 figure is the battery-UNNAMED count):
- **L7: 34 on S2, against 40 on D2**, a slight excess.
- **L12: 27 on S2, against 51 on D2**, a clear excess.
- After the context reading, D2 drops to 34 and 31 candidates. No post-reading comparison
  exists, because S2 was never read.

**KIR on S2** (in D2's coordinates): 25.9 at L7, 17.8 at L12.

### Behavioural tier: both anchors fail, so no tier-2 verdict at either layer

| unit | Δ at ε 0.005 / 0.01 / 0.02 | sign consistency | random max | sibling max (n) | PASS |
|---|---|---|---|---|---|
| **L7 anchor** logcosh #12 (paren) | 0.003 / 0.007 / 0.012 | 0.3–0.4 | 0.022 | 0.150 (63) | **no** |
| L7 skew #62 (paren) | −0.002 / 0.002 / 0.007 | 0.3–0.55 | 0.030 | 0.150 | no |
| **L12 anchor** logcosh #33 (quote) | 0.059 / 0.124 / **0.246** | 0.90–0.95 | 0.086 | **1.524** (76) | **no** |
| L12 skew #33 (quote) | 0.047 / 0.099 / 0.212 | 0.9 | 0.035 | 1.524 | no |
| L12 logcosh #11 (paren) | 0.032 / 0.062 / 0.118 | 0.65–0.8 | 0.085 | 1.524 | no |
| L12 skew #23 (paren) | 0.015 / 0.038 / 0.083 | 0.5–0.7 | 0.062 | 1.524 | no |

**The blind signatures are the right ones:**
- paren: `)`, `),`, `).`;
- quote: `,"`, `,”`, `."`, `.”`.

**Why the anchors fail.**
- **L7 paren**: steering barely moves the signature at these ε.
- **L12 quote**: the effect is clean and dose-dependent, and beats every random direction. It
  loses only to the **sibling maximum**.

**What sets the sibling maximum.**
- The largest steering effect of any unit is the **L12 subject-NP family**: logcosh #10, Δ 1.52,
  and skew #15, Δ 1.42. That is about 12× the sibling median of 0.125. This family was
  context-named, and it is identity-suspect (tail-TDI 3.4–3.5).
- Several units named by `next_punct` follow, at 0.46–0.66.
- The L12 quote anchor is outranked by roughly ten siblings, so its failure does not hinge on one
  outlier.

**Consequence.** Per prereg §8.1, no candidate was steered, and **no discovery can be claimed**.

### R2 replication

**Controls (the replication anchors) all replicate:**

| layer | |cos| | R2 |AUROC| |
|---|---|---|
| L7, paren | 0.97 | 0.82 |
| L12, paren | 0.94–0.96 | 0.76 |
| L12, quote | 0.98–0.99 | 0.75 |

Replication is therefore informative at both layers.

| | L7 | L12 |
|---|---|---|
| candidates replicated (cos ≥ 0.8, screens hold, still unnamed on R2) | **17 / 34** (12 families) | **17 / 31** (11 families) |
| median best |cos| (all battery-UNNAMED units) / chance p99.9 | 0.91 / 0.50 | 0.95 / 0.51 |
| failure modes (all battery-UNNAMED units) | 9 not screened, 7 cos < 0.8, 1 named on R2 | 10 not screened, 7 cos < 0.8, 1 named on R2 |
| context-named units replicated | 6 / 6 | 16 / 20 |
| KIR on R2 (D2 coordinates) | 16.6 | 11.8 |

### Verdict

**DISCOVERIES: none.** In the prereg's wording: *no unnamed occurrence variable was found at L7 or
L12 by this instrument.* That is not a claim that none exists.

| class (not discoveries) | L7 | L12 |
|---|---|---|
| replicated geometric (tiers 1 + 3) | 17 units / 12 families | 17 units / 11 families |
| geometric candidate only | 17 | 14 |

No unit has a tier-2 status.

### Reading

1. **The bottleneck is the behavioural rig, not the geometry.**
   - The anchors are known-variable directions, validated by AUROC on D2 and replicated on R2.
   - The prereg's tier-2 rule (v1 §9.2 and e62: ε ≤ 0.02·‖h‖; PASS needs Δ above the
     **maximum** over siblings) cannot pass them.
   - The sibling null treats other units' effects as noise. On v2 units they are real effects,
     some large.
   - A future tier 2 needs both of these:
     - **a null that is not the maximum of real effects**, for example a percentile of the
       siblings, random directions matched in pattern norm, or a shuffled-signature control;
     - **ε set on each unit's natural scale**, for example the SD of the residual along d.
   - These are design notes only. Nothing was run with them.
2. **What does stand.** About half of the unnamed candidates are reproducible directions on fresh
   data: 23 families across the two layers replicate on R2, with screens holding and still
   unnamed. They have no behavioural status. The post-verdict rule check below finds that 4 of the 23 families (|AUROC| ≥ 0.65) are
   nameable by the reading's own constructions.
3. **What the context reading found.** The instrument recovers syntactic state: negation scope,
   subject NP, fronted adverbials, copular predicates, coordination. None of these is in the
   battery.
   - The strongest causal direction in the run, the subject NP at L12, is one of them.
   - It is NAMED_BY_CONTEXT under the prereg, so it is not a discovery. It is still a positive
     descriptive result about what the v2 directions contain.
4. **The excess over shuffled text is slight at L7 and clear at L12.** Battery-UNNAMED counts
   are 40 against 34 at L7 and 51 against 27 at L12. At L7, most of the unnamed residue has a
   counterpart in what the instrument finds in word salad.
5. **Identity is unchanged.** KIR on R2 is 11.8–16.6, close to D2's 9.3–14.7. S2 is higher, at
   17.8–25.9.

**Caveats:**
- The context reading is one analyst's judgment. Its standard and per-unit counts are recorded
  and hashed.
- R2 (reserve 2,000–2,999) and the steering block (reserve 3,000–3,999) are now used.
- paren was tuned on P2; quote is the untuned control.

**Resources:**

| step | cost |
|---|---|
| extractions | three, 103–198 s each |
| behaviour | about 34 min on MPS |
| CPU phases | under 5 min each |
| swap | observed up to about 8.3 GB (sampled), under D1's 16 GB guard |

The D2, S2 and R2 activation caches were deleted; their small meta files were kept.

**Final folder size: 6.8 GB.** The discovery-tier outputs are `results/e64_*`, 39 MB.

### Post-verdict check of the context reading: the 14 constructions as mechanical rules (descriptive; e64c)

**Why.** The context reading (§6.2) was one analyst's judgment. Here each of its 14 constructions
becomes a fixed rule over D2's tokens, scored against the frozen D2 unit scores.

**Script.** `scripts/e64c_context_rules.py`, sha `a6eafd2d…`.
- The rules were written, tested on hand-written sentences only, and hashed before any D2 parse
  or score.
- Outputs: `results/e64c_*`, hashed.
- **No e64 tier and no part of the verdict changes.**

**Rules:**
- **Lexical**, on the previous D2 token:
  - after and/or/but;
  - after "in"/"at";
  - after "the";
  - within 5 tokens of a newline.
- **Parse-based**, on a spaCy 3.8 dependency parse of each D2 sequence: everything else.
  - Primary parser: `en_core_web_trf` 3.8.0. Robustness parser: `en_core_web_sm` 3.8.0, in a
    separate `.venv-parse`.
- The exact definitions are in the script's docstring.

**Scoring.** Each D2 token takes its spaCy word's value. Scores are oriented to the tail the
reading named. "Rank" is the unit's place among all screened units at its layer by |AUROC| for
that rule.

**Parser agreement.** The small parser is within 0.03 of the transformer on all 26 units. Every
count below holds under both parsers except one: four L7 misses at ≥ 0.65 instead of five.

**Precision: does a unit's own rule fit it?** Transformer parse; the two twins of a family are
given together.

| construction (units) | own-rule AUROC, oriented | read-tail rate vs base | rank | best rule | verdict |
|---|---|---|---|---|---|
| after and/or/but (L7 logcosh 37, skew 37) | 0.955 / 0.956 | 0.73 vs 0.025 (29×) | 1–2 / 65 | itself | **strong** |
| temporal clause (L7 skew 38, logcosh 50) | 0.68 / 0.67 | 0.36 vs 0.029 (12×) | 1–2 | itself | supported |
| after a preposition (L7 skew 49, logcosh 55) | 0.63 / 0.60 | 0.43–0.51 vs 0.086 (5–6×) | 1–2 | **after "in"/"at": 0.85 / 0.83** | supported; better as "after in/at" |
| partitive/superlative NP (L12 skew 2, logcosh 2) | 0.64 / 0.63 | 0.38–0.40 vs 0.040 (10×) | 1–2 | itself | supported |
| second conjunct of "X and Y" (L12 skew 52, logcosh 9) | 0.62 / 0.64 | 0.44–0.49 vs 0.090 (5×) | 1–2 | **after and/or/but: 0.78 / 0.82** | supported; better as "right after a conjunction" |
| subject NP before its verb (L12 logcosh 10, skew 15) | 0.68 / 0.68 | 0.86–0.88 vs 0.16 (5.5×) | 1–2 | multi-token subjects: 0.70 | supported |
| fronted adverbial (L12 logcosh 24, skew 24) | 0.76 / 0.75 | 0.57–0.59 vs 0.053 (11×) | 1–2 | itself | **strong** |
| onset of the next item in a series (L12 skew 37, logcosh 37) | 0.53 / 0.54 | 0.16–0.17 vs 0.063 (2.6×) | 20 and 10 / 78 | after and/or/but: 0.58–0.61 | **not supported** |
| first clause of a line (L12 logcosh 40) | 0.63 | 0.40 vs 0.124 (3.2×) | 1 | itself | supported (moderate) |
| negation scope (L12 skew 39, logcosh 44) | 0.71 / 0.71 | 0.71 vs 0.050 (14×) | 1–2 | itself | **strong** |
| copular predicate (L12 skew 34, logcosh 46) | 0.61 / 0.59 | 0.51–0.56 vs 0.10 (5×) | 1–2 | itself | supported (moderate) |
| enumeration or range (L12 logcosh 48) | 0.55 | 0.34 vs 0.09 (3.8×) | 5 | itself | weak |
| temporal expression (L12 logcosh 56, skew 12) | 0.56 / 0.56 | 0.26 vs 0.069 (3.8×) | 4–6 | temporal clause, 0.56 | weak |
| after "the" (L12 logcosh 60, skew 28) | 0.84 / 0.82 | 0.41–0.50 vs 0.032 (13–16×) | 1–2 | itself | **strong** |

**Precision, summarised.**
- 24 of 26 units meet the §6.1 naming thresholds for their own construction in the tail that was
  read. The exception is the series-onset pair.
- For 11 of the 14 constructions, the named unit or its twin is the best-matching unit in its
  layer.
- Two readings fit a simpler previous-token rule better:
  - the "second conjunct" pair is better described as "right after a conjunction";
  - the "after a preposition" pair is better described as "after in/at".

**Recall: did the reading miss units its own constructions name?** It did. These are candidates
read as "none" that reach |AUROC| ≥ 0.65 on one of the rules:

| unit | rule, |AUROC| | replicated on R2 |
|---|---|---|
| L7 logcosh #31 / skew #35 (twins) | after "the", 0.87 / 0.85 | yes |
| L7 logcosh #41 / skew #41 (twins) | negation scope, 0.68 / 0.67 | yes |
| L7 skew #59 | multi-token subject NP, 0.66 | no |
| L12 skew #8 | after "in"/"at", 0.79 | yes |
| L12 skew #0 / logcosh #0 (twins) | after and/or/but, 0.71 / 0.70 | yes |

**How many candidates the rules would name:**

| threshold | candidates (of 65) | replicated families (of 23) |
|---|---|---|
| §6.1 thresholds | 17 (13 at L7, 4 at L12) | 7 |
| |AUROC| ≥ 0.65 | 8 | 4 |
| |AUROC| ≥ 0.75 | 3 | — |

Many of the threshold hits at L7 come through the enrichment clause, at |AUROC| around 0.55, and
are weak.

**Reading.**
1. **The reading is accurate where it named something.** 11 of 14 constructions are confirmed,
   and the strongest structural ones are clear:
   - negation scope, 14× in the read tail;
   - fronted adverbials, 0.76;
   - the subject NP, 88% of the tail;
   - temporal clauses, 12×.

   The coordinated-series reading is not supported; enumeration and temporal expressions are weak.
2. **The reading under-called.** Its 8-of-12 standard missed four strong previous-token or
   negation directions among the candidates. So **"23 replicated unnamed families" should be
   read as about 19 after this check**, or 16 if every §6.1-threshold hit counts.
3. **Several of the cleanest directions are previous-token variables** (after "the",
   after "in"/"at", after a conjunction). Those are local context rather than nested state.
4. **The verdict does not change.** No unit could reach tier 2 under the frozen rule. No e64
   class or count was edited; the numbers above are descriptive only.

**Caveats:**
- The rules parse whole sequences, so they see future tokens, while the units see only the
  prefix.
- Parse errors push AUROC toward 0.5.
- The partitive, enumeration and temporal-expression rules are coarse.
- The §6.1 thresholds are reused here descriptively; they were not preregistered for this check.

---

## Stage 4 v2, redesigned steering tier (e65): **L7 anchor fails again; at L12 the rig validates, and one candidate family passes B3 and B4 (post hoc)**

**The redesign follows a seen failure.** `PREREG_occurrence_v2_steering.md` (revision 2, sha
`d0b7f5ea…`) and `scripts/e65_steering_redesign.py` (`5e37cab2…`) were written after the e64
anchors failed and after e64c was read. The prereg says so in its banner.

**What changed from e64's tier 2:**

| | e64 | e65 |
|---|---|---|
| dose | ε·‖h‖ | c·σ_d, where σ_d is the unsteered SD along the direction; gate at c\* = 2 |
| random null | full-space random directions | 99 random directions drawn in the units' own 64-d subspace (pullback-matched) |
| sibling null | the maximum over siblings' own effects | replaced by a shuffled-signature control (SIG): the unit's own push scored with every other unit's signature |
| contexts | reserve 3,000–3,999 | B3 = reserve 4,000–4,999; held-out B4 = 5,000–5,999 |

**Run on 2026-10-15, local MPS:**
- **Integrity.** Before anything ran, every prereg §2 hash was checked independently. All 14
  inputs, the prereg and the script match. Swap was 0.4 GB at the start and ≤ 0.7 GB throughout.
- **Freeze.** The rebuilt blind signatures of all 6 control units equal e64's.
- **Timings.** B3: 15 min at L7 and 28 min at L12. B4: 12 min, at L12 only. Every output is in
  HASHES.

### Anchors

| block | unit | Δ at c\* | random max | SIG max | sign consistency | dose vs e64 push | PASS |
|---|---|---|---|---|---|---|---|
| B3 | **L7 logcosh #12 (paren)** | 0.017 | 0.185 | 0.085 | ≤ 0.6 | 1.46× | **no**: fails all three gates |
| B3 | L7 skew #62 (paren) | 0.052 | 0.239 | 0.305 | ≤ 0.65 | 4.2× | no |
| B3 | **L12 logcosh #33 (quote)** | 0.856 | 0.417 | 0.570 | 0.85 | 3.7× | **yes** |
| B4 | **L12 logcosh #33 (quote)** | 0.873 | 0.560 | 0.546 | 0.8–0.9 | 3.7× | **yes** |
| B3 / B4 | L12 skew #33 (quote) | 0.70 / 0.80 | 0.30 / 0.46 | 0.67 / 0.62 | 0.75–0.9 | 3.9× | yes / yes |
| B3 / B4 | L12 logcosh #11, skew #23 (paren) | ≤ 0.16 | ≥ 0.56 | — | — | 0.75–1.1× | no / no |

**Reading the anchors.**
- **L7** gets no behavioural verdict, again. The paren direction does not move its own
  `)`-signature more than random directions in the same subspace do.
- **At L12 the quote direction is a validated steering anchor** on two independent context blocks.
  Its natural-scale dose is 3.7× e64's push, which is part of why e64's tier 2 failed it.
- **The paren units fail** at both layers.

### L12 candidates and context-named units

**B3.** 4 of 31 candidates pass, against an expected-false bound of Σ 1/(n′+1) = 0.40, so the
excess rule is met.
- Gate failures among candidates: 26 RAND, 15 SIG, 9 sign consistency.
- Four context-named units pass as well.

**B4** (fresh sites, σ and random bank):

| unit | e64 status | B3 | B4 | e65 class |
|---|---|---|---|---|
| **logcosh #13** | replicated geometric | Δ 1.64 (random 1.46, SIG 0.88) | Δ 1.63 (1.22, 0.96) | **candidate discovery under the redesigned tier 2 (post hoc)** |
| **skew #16** (twin, \|cos\| 0.95) | replicated geometric | Δ 1.41 (0.97, 1.03) | Δ 1.48 (0.81, 1.09) | **the same** |
| logcosh #54 | geometric candidate (not replicated) | Δ 0.46 | Δ 0.52 | held-out pass, not replicated geometrically |
| skew #54 (twin) | replicated geometric | Δ 0.35 | Δ 0.32 (random 0.32) | exploratory (B3 only) |
| logcosh #10 / skew #15 (subject NP) | context-named | Δ 1.52 / 1.51; SIG margin 8–10× | Δ 1.57 / 1.54 | held-out pass, context-named |
| logcosh #44 (negation scope) | context-named | Δ 0.76 | Δ 0.86 | held-out pass, context-named |
| logcosh #60 (after "the") | context-named | Δ 0.35 | Δ 0.41 (random 0.44) | exploratory (B3 only) |

**The one top-class family: L12 logcosh #13 / skew #16.**
- It replicated on R2 and is not context-generic.
- It is not identity-suspect: tail-TDI 1.8–2.0, largest tail type share 4%.
- No e64c rule names it, and the context reading left it "none". My partial note on #13 was
  "clause-final content word", in 6 of 12 contexts.
- **Its blind signature:** " before", " times", " when", " couple", " again", "?", " since",
  " without", " because", " as", " few", " today", " while", " next", " during".
- **What that suggests (descriptive):** the direction marks a completed clause that is open to a
  temporal or causal adjunct.
- Its strongest SIG rivals are the copular-predicate family (logcosh #46 / skew #34).
- Its random-direction margins are modest: 1.12× and 1.33× for #13; 1.45× and 1.83× for #16.

**Wording, per the prereg.**
- This is a **lead under a post hoc rig**, not a discovery in e64's sense. `e64_verdict.json` ("no
  discovery") is not amended.
- Per prereg §9, the conservative bound for a null unit passing both blocks is the single-block
  bound, about 0.013 per unit, or Σ ≈ 0.40 over L12's 31 candidates. Two families (three units)
  passed both blocks. One of them is the top-class family.
- A confirmatory test would need a new prereg, written before reserve 6,000–7,999 is touched.

**What is used now.** Reserve 4,000–4,999 (both layers) and 5,000–5,999 (L12). Reserve
6,000–7,999 is untouched.

**Provenance (a duplicate launch).** Two sessions started e65 at the same time.

- **Freeze.** Both freezes ran and wrote byte-identical outputs. That is why `docs/HASHES.txt`
  lines 42–44 and 45–47 repeat the same three shas.
- **B3.** Both B3 runs started 18 s apart (PIDs 18799 and 18896). The other session stopped 18799
  after about 8 minutes, during L7's baseline passes. It wrote no B3 output.
- **What is hashed.** `results/e65_steer_B3.json` and its single HASHES entry (16:19:49) come from
  18896. B4 and the verdict also come from 18896's session.
- **The log.** The overlap garbled the start of `logs/e65_steer_B3.log`; the L7 baseline line was
  overwritten.
- **Effect on results.** The overlap only slowed L7. Results come from the frozen inputs and seeds.

---

## Stage 4 v2, confirmation (e66): **CONFIRMED.** The L12 logcosh #13 / skew #16 family passes on untouched contexts

**Approval and documents.**
- The user approved this step; the parent session relayed it.
- `PREREG_occurrence_v2_confirm.md` (sha `59f1ae5b…`) and `scripts/e66_confirm_l12_family.py`
  (`21d47ae3…`) were frozen and registered in HASHES before any token of reserve 6,000–7,999 was
  processed.
- e66 imports e65 as a frozen module. **The tier-2 rule is e65's, unchanged:**
  - dose c·σ_d, with the gate at c\* = 2;
  - 99 random directions in the units' own subspace (RAND);
  - the shuffled-signature control (SIG);
  - sign consistency ≥ 0.8.
- **Single primary test: logcosh #13**, on block C. The rule fixed it in advance: the logcosh
  member first, which is also the member with the thinner earlier margins.
- **Blocks:** C = reserve 6,000–6,999 for the test; D = 7,000–7,999 for the description only.

**Run on 2026-10-15, local MPS.**
- Every pinned hash matched, including e65's and e64's chains.
- No other e64/e65/e66 process was running, and swap stayed ≤ 4.9 GB.
- Each phase was launched once.
- confirm took 16.5 min; describe took 2.5 min.
- All outputs (`results/e66_*`) are hashed.

### Block C: the verdict

| unit | Δ at c\* (c = 0.5 / 1 / 2 / 4) | random max | SIG max (n = 76) | sign consistency | dose vs e64 push | PASS |
|---|---|---|---|---|---|---|
| anchor: logcosh #33 (quote) | 0.19 / 0.41 / **0.84** / 1.64 | 0.48 (1.76×) | 0.51 (1.65×) | 0.8–0.9 | 3.65× | **yes** |
| **primary: logcosh #13** | 0.28 / 0.63 / **1.49** / 2.88 | 0.75 (**1.99×**) | 0.85 (**1.76×**) | 0.95–1.0 | 3.07× | **yes** |
| secondary: skew #16 | 0.28 / 0.64 / **1.38** / 2.17 | 0.72 (1.90×) | 1.03 (1.34×) | 0.95–1.0 | 3.50× | **yes** |

**Verdict: CONFIRMED**, and the family as a whole is confirmed: both members pass.
- **The bound.** The per-test false-pass bound is ≤ 1/77 ≈ 0.013. It is conservative, given the
  earlier B3/B4 passes. (Corrected 2026-10-24, see the top of this report: this is a rank bound
  under exchangeability nulls, and the family was selected by those B3/B4 passes, so it is not
  the probability that the family is a false pass.)
- **What the family now has.** It is the first unit in this project to clear all three of e64's
  tiers with tier 2 confirmed out of sample:
  - battery-unnamed;
  - replicated on R2;
  - not named by any e64c rule;
  - a tier-2 pass on three disjoint context blocks (B3, B4, C).
- **What is not amended.** e64's verdict ("no discovery" under e64's tier 2) and e65's classes are
  not amended.
- **Whether to call this a discovery is the user's decision.**

### What the direction does (descriptive; block D and D2)

**Steering on block D** (logcosh #13, push at the site, σ_d = 299).

At c = +2:
- **Promoted:** words that open an **adjunct**, by 2–2.6 nats of log-probability.
  - Subordinators: *during, because, while, before, when, until, without, since, after, unless,
    if, as*.
  - Prepositions: *with, through, from, by, for, within, into, along, in, on, to, at*.
- **Demoted:** paragraph and line breaks, sentence-final ".", sentence-starting pronouns (*It*,
  *My*, *We*) and bare nouns.
- **Probability mass:** prepositions gain +0.016 and subordinators +0.009, out of a base of 0.013.

At c = +4 the effect doubles: *because* and *before* gain about 5 nats, and prepositions gain +0.073
of mass. **c = −2 reverses it**: newlines and nouns up, adjunct openers down.

It is **not** about digits (digit mass ±0.0002), punctuation as such, or position.

**Continuations** (8 block-D prompts, the push held from the site on):
- **+2σ:** clauses are extended with adjuncts and subordinate clauses. "Rats … are used in a
  variety of research projects, including to study…"; "… for the first time, to make sure that
  you are getting the right attention, by the right…".
- **−4σ:** text closes off into terse, list-like, line-broken output. "are used to study
  cancer.⏎|Mouse Cancer Models⏎"; "March 28⏎– Thursday, March 29⏎…".
- **+4σ:** starts to degrade ("I's because I've").

**Where it is naturally active** (D2, frozen scores):
- **The token that completes a clause's core:**
  - a stranded preposition: *get rid **of** because…*, *exposed **to** during the day*, *in the
    direction **of** when…*;
  - a closing quote: *'procedures**'** without asking*;
  - an object's head noun: *absorb the **material** in manageable segments*, *come out of her
    **shell** in college*.
- **What follows those tokens:**

  | next token | after a heavy-tail position | at random |
  |---|---|---|
  | preposition | 28% | 8% |
  | subordinator | 6.0% | 1.5% |
  | temporal word | 3.5% | 1.3% |
  | punctuation | 22% | 10% |
  | digit | 1.4% | 6.1% |
  | capitalised word | 3% | 11% |

- **Position:** sentence and line position profiles are flat (mean z between −0.1 and +0.1 in
  every bin).
- **Labels:** no battery feature exceeds |AUROC| 0.58 (the best is next_newline, 0.576), and all
  16 e64c rules sit at 0.44–0.51 under both parses. Skew #16 is the same on every measure.

**Reading.**
- The confirmed direction encodes something like **"the clause's core is complete; an adjunct (a
  prepositional phrase or subordinate clause) can attach here"**. Pushing it makes the model
  elaborate the clause; pulling it makes the model close the clause or line.
- This is a **name found after the fact**. Neither the battery nor the e64c construction rules
  capture it, and my context reading had only a partial note ("clause-final content word", 6 of
  12). So the family is **nameable after the fact**, as the prereg requires me to say.
- Its strongest SIG rivals are the copular-predicate family (logcosh #46 / skew #34) and the
  temporal-clause units, which are syntactically related.

**What is used now.** Reserve 6,000–7,999. **No untouched reserve remains** for Stage-4 work.

---

## Prior-feature check (e67): is the confirmed L12 direction already a Gemma Scope 2 feature? **COVERED, by the pre-registered rubric**

**What this is.** A read-only check, approved by the user (relayed by the parent session). It used
D2 only, and it amends no e64, e65 or e66 verdict.

**Rubric.** `NOTE_prior_feature_check.md` was hashed before any comparison (`22bfd530…`). Addendum A
(`cdbb31de…`) was also hashed before any comparison: it added the three Neuronpedia-hosted L13
`resid_post` l0_medium dictionaries.

| verdict | requires |
|---|---|
| **COVERED** | One latent with decoder \|cos\| ≥ 0.7 (or activation \|r\| ≥ 0.7 with the unit's D2 score) that also passes a behavioural match. The match: ≥ 30% of its 200 strongest D2 rows are followed by a preposition, subordinator or temporal word. The unit itself is 37.5%; the base rate is 10.7%. |
| **PARTIALLY COVERED** | ≤ 5 decoder rows reach cos 0.9. |
| **NOT COVERED** | Otherwise. |

**Script.** `scripts/e67_prior_feature_check.py` (`b121979b…`). Outputs: `results/e67_*`, all hashed.

**Dictionaries.** All 16 are Gemma Scope 2 for gemma-3-1b-pt:
- L12: 16k and 262k, each in l0_big and l0_small;
- L10, L11, L13, L14: 16k, l0_big and l0_small;
- L13 `resid_post`: 65k l0_big, and the Neuronpedia-hosted 16k/65k/262k l0_medium.

### Results

| dictionary | max \|cos\| #13 / #16 (latent) | isotropic null p99.9 | subspace null p50 / p99.9 | atoms for cos 0.8 / 0.9 (#13, signed OMP) | null atoms for 0.9 (median) | non-neg OMP, 0.9 | best \|r\| with unit score #13 / #16 |
|---|---|---|---|---|---|---|---|
| L12 16k big | **0.735** (#11562) / **0.722** | 0.16 | 0.44 / 0.80 | 3 / 16 | 48 | 20 | 0.24 / 0.27 |
| L12 16k small | 0.607 / 0.693 (#2764) | 0.15 | 0.46 / 0.83 | 5 / 18 | 44 | 17 | 0.18 / 0.18 |
| L12 262k big | 0.513 / 0.541 | 0.17 | 0.40 / 0.75 | 9 / 25 | 51 | 31 | 0.24 / 0.25 |
| L12 262k small | 0.685 / **0.706** (#2084) | 0.17 | 0.45 / 0.85 | 4 / 16 | 39 | 18 | 0.19 / 0.19 |
| L11 16k big / small | 0.510–0.630 | 0.15 | 0.42–0.45 / 0.80–0.82 | 12–20 / 48–62 | 75 | 55–65 | ≤ 0.22 |
| L13 16k big / small | 0.621–0.688 | 0.15–0.16 | 0.42–0.44 / 0.78–0.81 | 6 / 27–30 | 64–68 | 32–33 | ≤ 0.31 |
| L13 rp 65k big | 0.635 / 0.626 | 0.16 | 0.38 / 0.72 | 8 / 34 | 76 | 42 | 0.33 / 0.35 |
| **L13 rp 16k medium** | 0.680 / **0.709** (#183) | 0.16 | 0.41 / 0.74 | 6 / 30 | 71 | 32 | **0.38 / 0.40 (#183)** |
| L13 rp 65k medium | 0.612 / 0.636 | 0.16 | 0.41 / 0.81 | 5 / 27 | 65 | 31 | 0.38 / 0.38 (#877) |
| L13 rp 262k medium | 0.560 / 0.597 | 0.17 | 0.42 / 0.85 | 7 / 27 | 60 | 29 | 0.31 / 0.33 (#7055) |
| L10 16k big / small | 0.509–0.654 | 0.15–0.16 | 0.43–0.45 / 0.81–0.84 | 10–16 / 49–54 | 83–84 | 61–67 | ≤ 0.19 |
| L14 16k big / small | 0.607–0.679 | 0.15–0.16 | 0.40–0.41 / 0.78–0.81 | 11–12 / 44–50 | 80–83 | 50–56 | 0.23 / 0.27 (#282, big); 0.24 / 0.24 (#156, small) ‡ |

"rp" = `resid_post`. The subspace null uses 1,000 random directions built like the unit's
(e65's pullback); the isotropic null uses 1,000 random unit vectors.

‡ From the float32 supplement `e67b` (see Off-plan). Neither L14 dictionary has a covering latent.

**Which rule triggered COVERED.** Four (latent, member) pairs reach |cos| ≥ 0.7 and pass the
behavioural match:

| latent | \|cos\| | behavioural share | its strongest D2 contexts |
|---|---|---|---|
| L12 16k big **#11562** | 0.735 (the decoder points *opposite* to d) | 0.485 | the object head of a transitive verb, followed by a PP: "Ship the **product** in a box", "wrap the **board** in a sheet", "absorb the **material** in manageable segments" |
| L12 262k small **#2084** | 0.706 (#16) | 0.40 | "open the **file** in Adobe", "make use of this **facility** by switching" |
| L13 16k medium **#183** | 0.709 (#16) | 0.41 | "smoke **them** because…", "rode that **bike** every single day", "ran the **business** until…" |

**#183** also has the highest activation correlation with the unit's score: r 0.38 / 0.40.

**Neuronpedia (L13 l0_medium only; no L12 dictionary is hosted):**
- #183 has no text explanation. Its top promoted logits are " sooner", " everytime", " without",
  " before", " anytime".
- Related latents split the behaviour by type of adjunct:
  - 65k #877 and 262k #7055 ("when during before") take temporal subordinators;
  - 16k #559 ("describes actions or states") and 262k #2795 ("creating or by means of") take
    prepositions;
  - 16k #136 and 65k #263 take *during, through, throughout*.

**Supervised baseline: the probe for "the next token is a preposition or subordinator".**
- In the unit's own representation, its held-out AUROC is 0.69. The unit itself scores only 0.57
  as a classifier of that label.
- **The probe's pattern has cos 0.67 with #13 (0.69 with #16).** That is just above the random
  pullback-direction p99.9 of 0.63 (max 0.68 of 2,000).
- In the whitened space, the pattern-to-pattern cos is 0.41, against a random p99.9 of 0.40.
- On the raw residual, the probe's pattern has cos 0.63 / 0.70 and its filter 0.10. The
  class-mean difference has cos 0.65 / 0.72.
- **Rubric reading: "distinct from" the probe direction** (< 0.7), and only modestly related.

### Reading

1. **Verdict: COVERED.** Existing Gemma Scope 2 latents already represent the construction: a
   clause or verb phrase completed by its object or complement head, with an adjunct coming next.
   - Their contexts match the unit's.
   - Their promoted logits are the adjunct openers the steering found.
2. **The qualifications are stated as computed.**
   - The covering cosines (0.71–0.74) exceed the isotropic null by far. But they sit *inside* the
     subspace null (p99.9 0.74–0.85; median 0.40–0.46). Decoder alignment alone would not prove
     coverage; the behavioural match does the work.
   - No single latent's activation tracks the unit closely: the best is r 0.40.
   - It takes 11–62 decoder rows to reconstruct the direction at cos 0.9 (3–20 for cos 0.8).
     Random directions of the same kind need 39–84.
   - The direction is best described as **a blend of several existing adjunct-prediction
     latents**: the temporal-subordinator latents and the preposition latents, which the SAEs
     split.
3. **What the confirmed direction adds is modest, and methodological rather than conceptual.**
   - It is a single, identity-removed direction that unifies the preposition and subordinator
     halves of this behaviour across token types.
   - It has a causally confirmed effect on three disjoint context blocks.
   - It is not a new concept, and the next-token-syntax probe does not explain it away either.
4. **The e66 verdict (CONFIRMED) stands**, as does the user's prerogative over the word
   "discovery". This check answers "is it new?" with "largely no".

### Off-plan

- **A re-run.** The acts stage crashed once, on an MPS float64 cast, before computing any
  correlation. It was fixed and re-run once.
- **L14 activation correlations, now filled in.** e67's acts stage stored raw residuals in fp16,
  and coordinate 1038 overflows at L14 (67,072, against an fp16 maximum of 65,504), so e67's L14
  cells were void. At L12 the same coordinate peaks at 63,232.
  - `scripts/e67b_l14_fp32.py` (sha `cee829d3…`) recomputed them with the residual held in float32
    throughout. Output: `results/e67b_acts_L14_fp32.json` (`fb2e5d75…`).
  - All r values are finite. The best |r| with the unit's score is 0.23 / 0.27 (16k big #282) and
    0.24 / 0.24 (16k small #156). No L14 latent covers the direction.
  - The verdict is unchanged. e67's own hashed files were not modified.
- **Swap.** During the acts stage swap rose to about 18.7 GB, above D1's 16 GB guard; the guard is
  checked only at stage start. The run completed, and swap fell back to about 8 GB.
- **Downloads.** The new dictionaries (about 8 GB) went into `hf_home`. The occD2x re-extraction
  cache was deleted after the probe.
  - The three 262k dictionaries (L12 big, L12 small, L13 rp medium; 7.3 GB) were later removed.
    That happened at about 18:14, outside this session, before the user's request to delete them
    reached it. The removal was clean: no dangling cache links remain, and every other dictionary
    is intact, including those the frozen pipeline uses.
  - The folder is now 11 GB.
  - The 262k results above stand; the files can be re-downloaded.
- **HASHES audit (after e67b).** All 65 entries in `docs/HASHES.txt` were checked against the
  files on disk:
  - **60 verify directly.**
  - **4 are superseded** by a later entry for the same file that does verify: e64's pre-D1
    version, e65's revision-1 prereg and script, and this rubric note before addendum A.
  - **1 is the freeze-time hash of `e63_occurrence_v2.json`.** That file was appended to after the
    freeze, as HASHES note line 13 records. Its final sha in that note matches the current file,
    and all three frozen blocks re-verify by `freeze_md5` and their units' npz sha256.
- **Neuronpedia.** Explanations were fetched read-only via curl. The system Python's SSL
  certificates failed for urllib.

---

## Coverage census (e68): **the calibration gate failed at both layers, so no target is nominated**

**Question (the user's "option 1").** Is any replicated, unnamed e64 candidate family **not**
covered by an existing Gemma Scope 2 latent? Only such a family would be worth fresh text and a
new prereg.

**Setup.**
- **Rubric:** `NOTE_coverage_census.md` (sha `75c0bf48…`).
- **Script:** `scripts/e68_coverage_census.py` (sha `6c1c806d…`).
- **Smoke test:** `docs/e68_smoke_synthetic.py` (sha `d72c9ae7…`), on fake dictionaries and
  residuals.
- All three were frozen after two rubric reviews and two code reviews, and before any census
  latent number existed. The pre-freeze reviews fixed two crash bugs and one disk-cap bug.
- **Data:** D2 only (corpus 1,000–1,999, already used).
- **Dictionaries:** 27 primary ones.
  - L7: `resid_post_all` L5–L9 16k big/small, plus `resid_post` L7 16k/65k/262k medium.
  - L12: `resid_post_all` L10–L14 16k big/small, plus `resid_post` L13 16k medium, 65k medium,
    65k big and 262k medium.
- **Units per layer:** every candidate (34 at L7, 31 at L12), the NAMED_BY_CONTEXT units, the
  controls, and 40 joint-null random directions.

**Run.**
- The first group-7 launch stopped before computing anything. `HF_HOME` was unset, so the SAE
  lookup failed; no output was written. It was relaunched with `HF_HOME=hf_home`, an environment
  setting only, with the script unchanged.
- The alignment gate passed at both layers (R² = 1.0000 for L7 logcosh #12/#37 and L12
  logcosh #13/#33). The census therefore rebuilds the frozen unit scores exactly.
- Swap peaked at 7.5 GB. The folder peaked at 17.7 GB.
- The extended pass had no target, so it downloaded nothing.
- The two primary 262k l0_medium files were deleted afterwards. The folder is now 12 GB.

### Result

| layer | role | COVERED | PARTIAL | NOT COVERED |
|---|---|---|---|---|
| L7 | candidates (34) | 16 | 0 | 18 |
| L7 | context-named (6) | 3 | 0 | 3 |
| L7 | controls (2) | 1 | 0 | 1 |
| L7 | **joint-null random directions (40)** | **6** | 0 | 34 |
| L12 | candidates (31) | 9 | 1 | 21 |
| L12 | context-named (20) | 5 | 3 | 12 |
| L12 | controls (4) | 0 | 1 | 3 |
| L12 | joint-null random directions (40) | 0 | 0 | 40 |

**Calibration gate: FAIL at both layers.** The gate set is the controls plus the context-named
units that an e64c rule names at |AUROC| ≥ 0.65. The rule derives this set, and it matched the
listed set exactly.

| layer | gate units NOT COVERED |
|---|---|
| L7 | 4 of 8 (logcosh #50, skew #38, skew #49, control skew #62) |
| L12 | 7 of 14, including 3 of the 4 paren/quote controls (logcosh #11, skew #23, skew #33), plus logcosh #24/#60 and skew #24/#28 |

The bar is 25%. The result is the same without the source dictionary. Under the frozen rule,
**neither layer nominates a target**, and the rule is not revised after the fact.

**Sanity check.** The e66/e67 family (L12 logcosh #13 / skew #16) comes out **COVERED** under
this rubric too, consistent with e67.

### Why the gate failed: single-latent coverage is the wrong yardstick for these units

- **No unit at either layer is covered through activation correlation.** No latent reaches
  |r| ≥ 0.7 with any unit's score.
  - Even the paren/quote controls peak at |r| 0.34–0.44.
  - Median max |r| is about 0.25 for candidates and 0.13 for random directions.
  - The units are identity-removed, whitened directions, so no single sparse code tracks them.
- **Every COVERED call comes through decoder |cos| ≥ 0.7** (plus a behavioural match). That
  route does not discriminate at L7:
  - **6 of 40 random pullback-matched directions are COVERED.** Their median best |cos| is 0.64,
    against 0.73 for candidates.
  - The subspace null's p99.9 is 0.83–0.90 per dictionary.
  - The reconstruction null reaches cos 0.8 with 1–2 atoms at p1, so the PARTIAL route is
    effectively closed at L7.
  - L12 is less degenerate: p99.9 is 0.71–0.84, p1 is 4–12 atoms, and no random direction is
    covered.
- **Variables the SAEs certainly represent fail the rule.** Paren-inside and quote-inside have
  dedicated latents in these dictionaries (for example the 16k-big paren and quote latents used
  in the option-2 design's knockout control). Yet 3 of the 4 L12 paren/quote control units come
  out NOT COVERED.

**So, for v2's units, "NOT COVERED" under a single-latent rule does not mean "no SAE feature".**
The census can say what is covered, but it cannot certify that anything is uncovered. It
therefore cannot hand option 1 a target. A distributed coverage class (code-level R² or a
sub-threshold class, calibrated so the controls come out covered) would be needed. Writing one
now would be designed after seeing these data, so it is left to the user.

### Descriptive only (not targets)

Seven families would pass every other target condition if the gate is ignored: NOT COVERED,
replicated, not e64c-nameable, no named twin, not identity-suspect or context-generic, and not
near a named or covered unit.

**L7 (no validated behavioural test at this layer):**

| family | best \|cos\| | max \|r\| |
|---|---|---|
| logcosh #18 / skew #36 | 0.70 | 0.24 |
| logcosh #33 | 0.51 | 0.22 |
| logcosh #49 / skew #8 | 0.61 | 0.32 |
| logcosh #52 | 0.72 | 0.21 |
| logcosh #57 / skew #33 | 0.52 | 0.24 |

**L12:**

| family | tier | best \|cos\| | max \|r\| |
|---|---|---|---|
| logcosh #45 / skew #5 | B (B3 failed) | 0.54 | 0.29 |
| **logcosh #54 / skew #54** | A by the census rule | 0.49 | 0.22 |

Family #54 has the strongest record, but it is a weaker lead than #13/#16 was:
- logcosh #54 passed e65's B3 and B4, but it is not R2-replicated.
- Its replicated twin, skew #54, passed B3 and tied the random maximum on B4 (Δ 0.32 against
  0.32).
- Its best decoder |cos| (0.49) equals the L12 random-direction median (0.49).

A confirmation would need fresh text (corpus 2,000+) and a new prereg.

**Outputs.** All are hashed in `docs/HASHES.txt`:
- `results/e68_prep_G{7,12}.json` (plus `_nullscores.npy`);
- `results/e68_decoder_G{7,12}.json`;
- `results/e68_acts_G{7,12}.json`;
- `results/e68_census.json`;
- `results/e68_extended.json`.

Logs are in `logs/e68_*.log`.

---

## SAE-gap pilot (e69, option 2): **GO by the frozen rule. Two of three slow error units survive, but only just**

**Question (the user's "option 2", "look where SAEs fail").** Is there persistent,
context-specific structure in the part of the L12 residual that the SAE fails to reconstruct?

**Design.**
- A judge panel picked it. Four independent designs were scored by three independent judges, and
  all three chose **sae_error**: multi-lag AMUSE (second-order temporal blind source separation)
  on the identity-removed reconstruction error e = h − SAE(h) of `resid_post_all` L12 16k
  l0_big.
- The record is `docs/sae_gap_calibration/` (manifest sha `6ed34758…`).
- The designers' read-only D2 calibration showed that e is Gaussian in every marginal, so ICA
  finds almost nothing. Its only reproducible structure is persistence across tokens.

**Pilot.**
- **Plan:** `NOTE_sae_gap_pilot.md` (sha `4a1bd995…`).
- **Script:** `scripts/e69_sae_gap_pilot.py` (sha `ef1f04dd…`).
- **Smoke test:** `docs/e69_smoke_synthetic.py`.
- All three were frozen after a three-lens review, which fixed two GO-leaning calibration bugs.
- **Data:** burned only. D2, and S2 (D2's within-sequence shuffle, e60's `occS2`). No fresh text
  was touched.

**Run.**

| phase | what happened |
|---|---|
| P0 | **Reproduced the design's calibration exactly:** 3 screened slow units (ids 0–2), split-half 0.94 / 0.84 / 0.83, lag-1 acf about 0.38, SAE-state \|ρ\| ≤ 0.04. |
| P1 | The D2 re-capture matched e67's stored rows exactly (100% identical values). The first P1 attempt died in the Claude app crash at about 21:54, after its S2 rows had been saved and hashed. The resume skipped them and redid only the D2 logits pass. Peak swap 6.5 GB. |
| P2 | CPU only; peak swap 4.0 GB. |

### Result (`results/e69_pilot.json`, sha `05dcc30f…`)

| unit | GENERIC-a (S2 refit \|cos\|) | GENERIC-b: acf₁ S2/D2 | acov₁ ratio | predictability max \|ρ\| | bag max R² | verdict |
|---|---|---|---|---|---|---|
| 0 | 0.23 | **0.51** | 0.34 | 0.04 | 0.04 | removed (generic-b) |
| 1 | 0.16 | 0.46 | 0.26 | 0.04 | 0.01 | **survivor** |
| 2 | 0.16 | 0.46 | 0.28 | 0.02 | 0.02 | **survivor** |

The bars are |cos| ≥ 0.8, a ratio ≥ 0.5, |ρ| ≥ 0.3 and R² ≥ 0.3. **Decision: GO** (survivors 1
and 2).

**How to read it.**
- **The survivors pass the shuffle test by a small margin.** Their D2 filters keep 46% of their
  lag-1 autocorrelation on shuffled text, against a 50% bar, and 26–28% of their lag-1
  autocovariance.
  - This is not error inflation. The filters' variance falls on S2 (ratio 0.56–0.60), and S2's
    mean log‖e‖ is only slightly higher (6.28 against 6.20).
  - About half of these units' persistence is context-specific; the rest survives shuffling.
- **Shuffled text has its own, stronger slow structure in the error.**
  - S2's top lagged eigenvalues are 0.49 / 0.45 / 0.42, against D2's 0.23 / 0.21 / 0.21.
  - S2 has 6 screened units, against 3.
  - These are different directions: no S2 unit matches a D2 unit above |cos| 0.23.
  - So the SAE error carries a large, generic persistent component that out-of-distribution text
    amplifies. The D2 units are not its dominant directions.
- **Nothing names them.**
  - next-token entropy and surprisal, raw or identity-removed, and their running means: |ρ| ≤ 0.04;
  - a bag of the previous 8–64 tokens, ridge or MLP: held-out R² ≤ 0.04;
  - the 17 known labels: |AUROC| ≤ 0.56.
- **The SAEs barely see them (descriptive).**
  - The max pre-activation |r| with any latent is 0.09–0.17 (16k big and small). That is above
    the RAND-E maximum (0.07–0.08) but tiny.
  - Ridge on 128 latents' codes plus sub-threshold drive: R² ≤ 0.10.

**What GO means.** By the plan, GO only licenses *writing* the full prereg on fresh text:
- D3 = corpus 2,000–2,999, plus S3, R3, the steering blocks and a confirmation block;
- the judges' grafts: knockout and planted controls (planted in h and re-encoded), an error-share
  decomposition, and a calibrated sub-threshold coverage class.

It is not evidence of a discovery. The units are Gaussian, slow (lag-1 acf about 0.37, about 0.05
by lag 16) and diffuse. The design itself flagged its behavioural tier as weak for such units.

**Updated odds.** Before the pilot, the design's chain put a confirmed, uncovered and unnamed
finding at about 0.02–0.03, with S3 as the main risk (P about 0.35). The pilot passed S3 on
burned data, but by a small margin. **The overall chance is now about 0.03–0.05.** The large
remaining risks:
- the margin may not reproduce on fresh D3;
- steering a diffuse slow unit;
- coverage by wider or neighbouring-layer codes.

---

## SAE-gap round prereg (e70): **drafted and reviewed, NOT frozen**. The review shows D2 already decides it

`PREREG_sae_gap.md` and `scripts/e70_sae_gap_geometric.py` were drafted. They cover the e69
survivors as fixed D2 filters on fresh text, with the judges' grafts. The script passed two
synthetic end-to-end smoke tests (`docs/e70_smoke_*.py`).

Three independent read-only reviewers then examined both: design validity, code conformance and
preregistration discipline. The code and discipline issues were fixable, and most fixes are
applied in the draft script. **The design review found that the round is decided on burned data
before any fresh text is read.** An independent check (`docs/e70_review_verification.py`, output
`results/e70_review_verification.json`, both hashed) reproduces this:

| filters on D2 | lag-1 acf | acf ratio S2/D2 | acov ratio | error share (16k big) |
|---|---|---|---|---|
| survivor unit 1 | 0.38 | 0.46 | 0.26 | 0.31 |
| survivor unit 2 | 0.36 | 0.46 | 0.28 | 0.26 |
| 100 random error-space filters (p5 / median / p95) | 0.24 / 0.25 / 0.26 | 0.42 / 0.47 / 0.55 | 0.26 / 0.31 / 0.38 | 0.19 / 0.21 / 0.24 |

- **The pilot's shuffle test does not discriminate.**
  - 71% of random error-space directions pass GENERIC-b (both ratios < 0.5).
  - The survivors sit at the random median, and unit 0's removal at 0.507 is within noise.
  - The pilot had no random-direction baseline, so **its GO carries no information about units 1
    and 2**. It shows only that the L12 error's slow subspace as a whole is roughly half
    context-dependent.
- **Persistence is not distinctive either.** Every random direction clears G1 (acf₁ ≥ 0.20). The
  survivors are more persistent than random (0.37 against 0.25), but AMUSE selects them for exactly
  that.
- **"Where SAEs fail" fails on D2.** Along the survivors' filters, the raw residual is about 70%
  SAE-reconstructed: error share 0.26–0.31 under the source SAE, against the prereg's 0.5 bar, and
  no error-space direction tested reaches 0.5. Pushing these directions would push mostly
  SAE-span signal.

**The prereg was therefore not frozen.** Freezing it would commit fresh text (corpus 2,000–3,999)
to an outcome that D2 already fixes (SAE-SPAN). The reviewers' other findings:
- the coverage gates do not calibrate the error-space regime;
- the SIG pool is ambiguous;
- the sealed context reading was dropped;
- the single-coordinate screen was missing;
- the hash chain did not pin imported modules.

Those are recorded in the draft and moot unless the round is redesigned.

**Status of option 2.** By its own pilot, the SAE-gap search found the expected generic slow
structure in the L12 16k-big error. It found nothing unit-specific that is context-dependent
beyond what random error-space directions show, and the directions are mostly SAE span in raw h.
No fresh text was spent.

---

## Per-family coverage readout (e71): **descriptive only. Neither L12 would-be target separates from the paren controls or from random same-subspace directions on any of the three measures. The small-K readout check is uninformative: no anchor reaches R² 0.5.**

**What this is.** The e67-style readout requested after e68–e70. It covers:
- the 7 would-be-target families in "Coverage census (e68) → Descriptive only";
- the e66/e67 family L12 logcosh #13 / skew #16, as the reference;
- a sparsity-relative-to-null reading;
- a small-K linear-readout pre-check.

**Limits.**
- **Data:** already-hashed outputs and already-used D2 (corpus 1,000–1,999) only. No fresh corpus text was read, nothing was steered, and no model forward pass was run.
- **No verdict is amended** (e64–e70), and **no target is nominated.**

**Documents.** Each was hashed with `e64.append_hash` before the step it governs.
- **Rubric:** `NOTE_family_coverage_readout.md` (sha `6d1a479b…`). It was fixed before any per-family number was read: the template, the sparsity cut procedure and the pre-check rule.
- **Script:** `scripts/e71_family_readout.py` (`e70c45fa…`), hashed before its first real run, after a synthetic smoke test only.
- **Stage `calib`** read only the 6 anchors and the 40 joint nulls. It wrote `results/e71_rank_G12.npz` (`86afb1db…`) and `results/e71_calib.json` (`e94e9e58…`), both hashed before stage `readout` read any target cell.
- **Stage `readout`** wrote `results/e71_readout.json` (`e6bf66e0…`).
- **Post hoc, descriptive, not in the rubric:**
  - `docs/e71_match_null_baseline.py` → `results/e71_match_null.json` (revision 2: `4b4787a5…` / `d4a97432…`);
  - `docs/e71_diag_logcosh11_k64.py` (`9924fd76…`);
  - `docs/e71_null_band_check.py` → `results/e71_null_band_check.json` (`2873f54f…` / `0f3a1600…`);
  - the page formatter `docs/e71_format_pages.py`. Every table below is rendered from the JSONs by it.

**Run.** CPU only. `calib` took 49 s and `readout` 51 s. Both ran detached under `caffeinate` with a 12 GB swap watchdog; swap stayed at 2.2–2.3 GB. There was one SAE encode: `resid_post_all` L12 16k l0_small on e67's stored D2 L12 rows (`e67_raw_L12_rows.npy`, fp16, which is disclosed). L12 latent text is therefore available only for that dictionary.

### (b) Sparsity relative to the null: calibration is informative, and **the paren controls read "chance-like"**

**Statistics.**
- Per (dictionary, OMP mode), each unit's atoms to cos 0.8 are banded against that dictionary's per-mode reconstruction null (`null_recon` p1 / p5 / p50).
- S1 and S5 count the dictionaries at ≤ p1 and ≤ p5; this is the max over modes.
- H counts the dictionaries at > p50; this is the min over modes.

**Calibration.**
- The rule was chosen mechanically on the 6 anchors, under a cap of ≤ 2 of the 40 L12 joint nulls (`results/e71_calib.json`).
- **With source (14 dictionaries): "sparser" = S1 ≥ 7**, i.e. at least 7 dictionaries at or below the recon-null *p1*. The procedure preferred S1 over S5. 4 of 6 anchors pass, including both #13 and #16, and 2 of 40 nulls pass. By the rubric, the reading is informative.
- **Without source (13 dictionaries): S1 ≥ 6.** The same 4 anchors and 2 nulls pass.

| anchor | S1 / S5 / H (signed; nonneg) | class |
|---|---|---|
| logcosh #13 (e67: COVERED as a blend) | 6 / 8 / 0; 7 / 9 / 0 | sparser |
| skew #16 | 11 / 12 / 0; 11 / 12 / 0 | sparser |
| logcosh #33 (quote control) | 11 / 13 / 0; 10 / 14 / 0 | sparser |
| skew #33 (quote control) | 11 / 14 / 0; 12 / 14 / 0 | sparser |
| **logcosh #11 (paren control)** | 0 / 0 / 2; 0 / 0 / 2 | **chance-like** |
| **skew #23 (paren control)** | 0 / 0 / 5; 0 / 0 / 3 | **chance-like** |

**What the null shows.**
- **Joint nulls:** S1 is 0 for 37 of 40. The other three have 1, 7 and 14 (null/14 and null/27 pass the rule).
- **The "denser" class is empty.** No c ≤ 14 leaves ≤ 2 nulls with H ≥ c, because the joint nulls' H is bimodal: 18 of 40 have H = 0 and 11 have H = 14.
  - The cause is that a random direction's band is strongly correlated across the 14 dictionaries. It is *not* that the two random constructions differ.
  - Cell by cell, the 40 joint nulls (built exactly like the units) sit above e65's recon-null median in 251 and 249 of 560 cells (signed / non-negative; 45%). They sit at or below p1 in 19 and 22 cells, and at or below p5 in 33 and 40 (post hoc, `docs/e71_null_band_check.py` → `results/e71_null_band_check.json`).
  - So the 14 dictionary counts are not independent tests. That is why the cut is calibrated on the joint nulls rather than read as a binomial.

**Across all 55 L12 units** (with and without source alike):
- candidates: 10 sparser and 21 chance-like;
- context-named: 4 sparser and 16 chance-like;
- controls: 2 and 2.

**"Sparser" largely restates e68's decoder route.**
- 12 of the 14 e68-COVERED L12 units read sparser. The two exceptions are logcosh #44 and skew #51.
- Only 2 of the 36 NOT COVERED units do: skew #6 and the quote control skew #33.
- Both L12 would-be targets, #45/#5 and #54/#54, read **chance-like** in every member: S1 = S5 = 0, with and without source.

**Reading, as the rubric fixed it.**
- A chance-like unit "reconstructs like a random direction in the same subspace". That is not evidence of an uncovered variable.
- **And here the chance-like class demonstrably contains known-represented variables**: the two L12 paren units, for which these dictionaries have dedicated paren latents (e68's "Why the gate failed").
- So a chance-like target cannot be told apart from a represented, non-sparse variable like paren. Sparsity can confirm "blend-like" (#13/#16, quote); it cannot certify "not a blend".

### (c) Small-K readout pre-check: **uninformative. The anchors never reach R² 0.5, so K_anc does not exist.**

**Method.** Held-out R² (even → odd sequences) of each frozen D2 score, from the identity-removed {a, clip(pre, 0, θ)} of its top-K L12 16k-small latents. The latents were chosen on the even half by e69's |r(pre)|, and the fit is e69's `heldout_r2`, verbatim.

**Result.** **No anchor reaches 0.5 at any K ≤ 64.** The anchor maximum is 0.307 (skew #16 at K = 32). Under the rubric that is the "anchors never reach it" branch, so linear predictability is uninformative, and there is no per-target reading. No unit and no null reaches 0.5 either. The per-group table is reported because the rubric requires it:

| group (n) | R² at K = 32: median (min–max) | R² at K = 64: median (min–max) | reaching 0.5 at any K ≤ 64 |
|---|---|---|---|
| anchors (6) | 0.253 (0.228–0.307) | 0.276 (-0.016–0.298) | 0 |
| targets (#45/#5, #54/#54) (4) | 0.180 (0.162–0.200) | 0.216 (0.208–0.233) | 0 |
| all candidates (31) | 0.198 (0.082–0.332) | 0.237 (-0.176–0.368) | 0 |
| context-named (20) | 0.247 (0.172–0.375) | 0.294 (0.188–0.393) | 0 |
| controls (4) | 0.253 (0.240–0.268) | 0.269 (-0.016–0.287) | 0 |
| joint nulls (40) | 0.109 (0.081–0.151) | 0.157 (-0.060–0.203) | 0 |

**Why the K = 64 column is not read.**
- K = 64 is not robust. e69's verbatim fit z-scores each column on the even half, and a column that is almost never active there explodes on the odd half.
- Example: for anchor logcosh #11 (0.268 at K = 32 → −0.016 at K = 64), one code column has even-half SD 0.003 and |z| ≈ 43,800 on the odd half. Without the 5 such columns, K = 64 gives 0.294 (`docs/e71_diag_logcosh11_k64.py`, `logs/e71_diag_logcosh11_k64.log`).
- The same failure can only lower R², and it also touches a candidate (−0.176) and a null (−0.060). So the K = 64 ordering is descriptive at most.

**At K = 32 (descriptive).**
- The anchors (0.23–0.31) and the four target units (0.16–0.20) all sit above the null maximum (0.151).
- So at small K the sibling readout is not tautological in the sense expected after e70, where random same-subspace directions would be predicted as well as units.
- But neither the anchors nor anything else reaches the frozen bar, and the rule is not revised.

### Context sharing: the rubric's criterion, then a random baseline it fails at L12

**The rubric's mechanical criterion: "a latent shares the unit's contexts iff e68 records it as a match".**
- By that criterion, every family shares its contexts with existing latents.
- Per member there are 19–99 matching tested latents, against 60 per member for the #13/#16 reference.
- The largest tail overlap (the share of a latent's strongest D2 rows, up to 200, that fall in the unit's 1% tail) is 0.97–1.00 for 7 of the 8 L7 members (logcosh #52: 0.69). For the L12 targets it is 0.53–1.00.
- No matching latent reaches |cos| 0.7 or |r| 0.7, and no OMP route meets the PARTIAL bar. That is why e68 called these families NOT COVERED.

**Post hoc qualification** (`results/e71_match_null.json`; not in the rubric). The 40 joint-null random directions went through the identical e68 match procedure.
- **L12:**
  - the nulls have a median of 37 matching latents (max 96);
  - **19 of 40 have a latent with tail overlap ≥ 0.95**;
  - on a fixed-size set (per dictionary, top-5 |cos| ∪ top-5 |r|), their match rate is median 0.14, max 0.46;
  - **no L12 control exceeds the null maximum on that fixed-set rate.**
  - **The mechanical context-sharing criterion therefore does not discriminate at L12.** Random same-subspace directions also have latents that fire exactly where they fire.
  - That is expected for decoded-span directions: any direction's heavy tail consists of rows where some latents are strongly active.
- **The full-set rate is confounded.** The full tested set includes the OMP atoms, so it grows with k08. The full-set match rate therefore falls with sparsity, which made it look as if it tracked the (b) classes. The fixed set removes that confound, and the apparent agreement disappears.
- **L7:**
  - null median 6.5 matches (max 46); 12 of 40 have a latent with overlap ≥ 0.95; fixed-set rate median 0.02, max 0.29;
  - **every L7 target except logcosh #52, and both L7 controls, exceed all 40 nulls on both match measures.** At L7 the targets look like the controls. L7 has no validated behavioural test, and its decoder null is degenerate.

### Summary of the eight families

| family | e68 (with source) | best \|cos\| / max \|r\| | e68 matches: all (heavy side) | fixed-set match rate (nulls ≥, of 40) | (b) sparsity | (c) R² K = 32 / 64 |
|---|---|---|---|---|---|---|
| L7 logcosh #18 / skew #36 | NOT COVERED; NOT COVERED | 0.70 / 0.24; 0.63 / 0.23 | 76 (0); 70 (70) | 0.57 (0); 0.50 (0) | — (L7); — (L7) | — (L7); — (L7) |
| L7 logcosh #33 | NOT COVERED | 0.51 / 0.22 | 50 (45) | 0.38 (0) | — (L7) | — (L7) |
| L7 logcosh #49 / skew #8 | NOT COVERED; NOT COVERED | 0.60 / 0.32; 0.57 / 0.31 | 57 (55); 51 (51) | 0.56 (0); 0.46 (0) | — (L7); — (L7) | — (L7); — (L7) |
| L7 logcosh #52 | NOT COVERED | 0.72 / 0.21 | 19 (19) | 0.15 (4) | — (L7) | — (L7) |
| L7 logcosh #57 / skew #33 | NOT COVERED; NOT COVERED | 0.52 / 0.21; 0.52 / 0.24 | 62 (62); 62 (62) | 0.46 (0); 0.40 (0) | — (L7); — (L7) | — (L7); — (L7) |
| L12 logcosh #45 / skew #5 | NOT COVERED; NOT COVERED | 0.54 / 0.29; 0.55 / 0.28 | 49 (49); 32 (32) | 0.32 (5); 0.17 (17) | chance-like; chance-like | 0.20 / 0.23; 0.20 / 0.22 |
| L12 logcosh #54 / skew #54 | NOT COVERED; NOT COVERED | 0.49 / 0.22; 0.47 / 0.18 | 99 (88); 89 (89) | 0.45 (1); 0.47 (0) | chance-like; chance-like | 0.16 / 0.21; 0.16 / 0.21 |
| L12 logcosh #13 / skew #16 | COVERED; COVERED | 0.74 / 0.38; 0.72 / 0.40 | 60 (57); 60 (60) | 0.41 (2); 0.42 (2) | sparser; sparser | 0.23 / 0.27; 0.31 / 0.30 |
| **joint nulls** L7 / L12, median (max) | — | — | 6.5 (46) / 37 (96) | 0.02 (0.29) / 0.14 (0.46) | L12: 38 of 40 chance-like | L12 K = 64: 0.16 (0.20) |

**Controls, for comparison (L12).**
- paren logcosh #11 / skew #23: chance-like; fixed-set rate 0.37 / 0.40 (4 and 2 of 40 nulls ≥).
- quote logcosh #33 / skew #33: sparser; 0.28 / 0.29 (8 and 5 nulls ≥).
- R² at K = 32 is 0.24–0.27 for all four.

### Family pages

Rendered by `docs/e71_format_pages.py` from `results/e71_readout.json`. Each page ends with the analyst's **reading**. The readings are descriptive: they were written after the numbers and contexts were seen, and they are not verdicts.

**Legend.**
- "sig" is the share of the latent's 200 strongest D2 rows whose next token is in the unit's signature (the opposite-tail signature T⁻ when r < 0), with e68's threshold and its 500-random-latent p99.
- "ovl" is the share of those rows inside the unit's 1% tail on that side.
- "**match**" means either test passes.
- Bands: [≤p1], [≤p5], [mid] and [>p50] against the dictionary's per-mode recon null.
- At L7 no latent text exists: raw rows are stored only at L12, and nothing was re-captured.
- **Read the per-page match tallies with the baseline above.** They are the rubric's context-sharing criterion. As "Context sharing" shows, that criterion does not discriminate at L12, and a near-1.0 tail overlap is common among random directions (12 of 40 joint nulls at L7, 19 of 40 at L12). So "fires only there" in a reading describes the latent; it is not evidence beyond chance.

#### L7 logcosh #18 / skew #36

| member | e68 verdict (with / without source) | best \|cos\| (dictionary) | max \|r\| | e64c best rule, \|AUROC\| | e65 B3 / B4 | signature testable (s_self vs base) |
|---|---|---|---|---|---|---|
| logcosh #18 | NOT COVERED / NOT COVERED | 0.70 (L7 16k small) | 0.24 | C, 0.54 | not run (L7) | False (0.125 vs 0.079) |
| skew #36 | NOT COVERED / NOT COVERED | 0.63 (rp L7 262k medium) | 0.23 | C, 0.54 | not run (L7) | True (0.655 vs 0.047) |

Census record: R2-replicated **True**; e64c-nameable False; named twin False; identity/generic False; near a named/covered unit False; census tier L7 (no validated behavioural test).
Signature T of logcosh #18: `May` `April` `July` `th` `$` `' '` `/` `with` `for` `at` `"` `it` `what` `do` `)` `3` `year` `(` `,` `more`
Signature T of skew #36: `S` `m` `formats` `or` `g` `9` `D` `me` `com` `“` `what` `not` `if` `0` `there` `when` `from` `we` `that` `I`

**logcosh #18: nearest latents (pooled over the 13 primary dictionaries).** max \|cos\| above the subspace-null p99.9 in 0 of 13 dictionaries (0 without source). Tested latents 363; **76 match** (0 on the heavy side, 76 on the opposite side; 68 outside the source dictionary). Largest tail overlap: L6 16k small #962, 1.000 (threshold 0.505, null p99 0.092).

| | top \|cos\| latent | cos | match test | top \|r\| latent | r (within type) | match test |
|---|---|---|---|---|---|---|
| 1 | L7 16k small #61 | -0.702 | neg; sig 0.02 (≥0.37, p99 0.26); ovl 0.04 (≥0.51, p99 0.07); no match | rp L7 16k medium #556 | -0.240 (-0.24) | neg; sig 0.48 (≥0.37, p99 0.23); ovl 0.84 (≥0.51, p99 0.08); **match** |
| 2 | L7 16k big #61 | -0.641 | neg; sig 0.32 (≥0.37, p99 0.27); ovl 0.09 (≥0.51, p99 0.13); no match | L9 16k small #164 | -0.238 (-0.24) | neg; sig 0.58 (≥0.37, p99 0.26); ovl 0.83 (≥0.51, p99 0.08); **match** |
| 3 | L7 16k big #8446 | 0.634 | neg; sig 0.84 (≥0.37, p99 0.27); ovl 0.57 (≥0.51, p99 0.13); **match** | L8 16k small #549 | -0.237 (-0.26) | neg; sig 0.78 (≥0.37, p99 0.33); ovl 0.67 (≥0.51, p99 0.08); **match** |
| 4 | rp L7 262k medium #8025 | -0.624 | neg; sig 0.03 (≥0.37, p99 0.21); ovl 0.04 (≥0.51, p99 0.07); no match | L7 16k big #556 | -0.236 (-0.24) | neg; sig 0.59 (≥0.37, p99 0.27); ovl 0.91 (≥0.51, p99 0.13); **match** |
| 5 | rp L7 16k medium #556 | -0.608 | neg; sig 0.48 (≥0.37, p99 0.23); ovl 0.84 (≥0.51, p99 0.08); **match** | L7 16k big #539 | -0.232 (-0.24) | neg; sig 0.48 (≥0.37, p99 0.27); ovl 0.80 (≥0.51, p99 0.13); **match** |

**skew #36: nearest latents (pooled over the 13 primary dictionaries).** max \|cos\| above the subspace-null p99.9 in 0 of 13 dictionaries (0 without source). Tested latents 344; **70 match** (70 on the heavy side, 0 on the opposite side; 61 outside the source dictionary). Largest tail overlap: L8 16k small #6235, 1.000 (threshold 0.505, null p99 0.074).

| | top \|cos\| latent | cos | match test | top \|r\| latent | r (within type) | match test |
|---|---|---|---|---|---|---|
| 1 | rp L7 262k medium #8149 | 0.631 | pos; sig 0.70 (≥0.35, p99 0.26); ovl 0.98 (≥0.51, p99 0.07); **match** | rp L7 16k medium #556 | 0.231 (0.24) | pos; sig 0.47 (≥0.35, p99 0.25); ovl 0.79 (≥0.51, p99 0.08); **match** |
| 2 | L5 16k big #51 | 0.571 | pos; sig 0.60 (≥0.35, p99 0.23); ovl 0.18 (≥0.51, p99 0.06); **match** | L8 16k small #549 | 0.229 (0.25) | pos; sig 0.77 (≥0.35, p99 0.36); ovl 0.62 (≥0.51, p99 0.07); **match** |
| 3 | rp L7 16k medium #118 | 0.564 | pos; sig 0.05 (≥0.35, p99 0.25); ovl 0.00 (≥0.51, p99 0.08); no match | L7 16k big #11050 | 0.227 (0.23) | pos; sig 0.86 (≥0.35, p99 0.27); ovl 0.90 (≥0.51, p99 0.10); **match** |
| 4 | L7 16k small #10164 | -0.540 | neg; sig 0.00 (≥0.11, p99 0.32); ovl 0.04 (≥0.51, p99 0.10); no match | L9 16k small #164 | 0.227 (0.23) | pos; sig 0.57 (≥0.35, p99 0.27); ovl 0.74 (≥0.51, p99 0.09); **match** |
| 5 | rp L7 262k medium #8025 | 0.529 | pos; sig 0.03 (≥0.35, p99 0.26); ovl 0.04 (≥0.51, p99 0.07); no match | rp L7 262k medium #8149 | 0.226 (0.23) | pos; sig 0.70 (≥0.35, p99 0.26); ovl 0.98 (≥0.51, p99 0.07); **match** |

**OMP atoms to cos 0.8** (band vs that dictionary's per-mode null; `>` = unreached by k_max, banded as k_max + 1):

| dictionary | logcosh #18 signed / nonneg k08 | skew #36 signed / nonneg k08 | null k08 p1 / p5 / p50, signed | nonneg |
|---|---|---|---|---|
| L5 16k big | 8 [mid] / 14 [>p50] | 10 [>p50] / 14 [>p50] | 1 / 2 / 8 | 2 / 2 / 9 |
| L5 16k small | 11 [>p50] / 19 [>p50] | 15 [>p50] / 16 [>p50] | 1 / 1.95 / 8.5 | 1 / 2 / 10 |
| L6 16k big | 8 [mid] / 15 [>p50] | 12 [>p50] / 11 [>p50] | 1 / 2 / 8 | 1 / 2 / 10 |
| L6 16k small | 6 [mid] / 13 [>p50] | 10 [>p50] / 14 [>p50] | 1 / 1 / 8 | 1 / 2 / 10 |
| L7 16k big | 4 [mid] / 7 [mid] | 7 [mid] / 8 [mid] | 1 / 2 / 8 | 1 / 2 / 9 |
| L7 16k small | 3 [mid] / 11 [>p50] | 7 [mid] / 8 [mid] | 1 / 1 / 8 | 1 / 1.95 / 8 |
| L8 16k big | 12 [>p50] / 14 [>p50] | 12 [>p50] / 12 [>p50] | 1 / 1 / 10 | 1 / 2 / 11 |
| L8 16k small | 10 [>p50] / 14 [>p50] | 13 [>p50] / 14 [>p50] | 1 / 1 / 9 | 1 / 2 / 11 |
| L9 16k big | 18 [>p50] / 23 [>p50] | 21 [>p50] / 24 [>p50] | 1.99 / 2 / 13 | 1.99 / 2 / 14.5 |
| L9 16k small | 15 [>p50] / 21 [>p50] | 22 [>p50] / 23 [>p50] | 1 / 2 / 11.5 | 1 / 2 / 14 |
| rp L7 16k medium | 7 [mid] / 13 [>p50] | 7 [mid] / 7 [mid] | 1 / 1 / 8.5 | 1 / 2 / 9.5 |
| rp L7 65k medium | 5 [mid] / 13 [>p50] | 7 [mid] / 7 [mid] | 1 / 1 / 8 | 1 / 1 / 9 |
| rp L7 262k medium | 5 [mid] / 10 [>p50] | 7 [mid] / 7 [mid] | 1 / 1 / 7 | 1 / 1 / 8 |

<details><summary>logcosh #18: 12 strongest D2 contexts (oriented score)</summary>

- ` to the Mopar Nationals is like taking refuge in a fortress, a safe[[ haven]], or comfort`
- `2012, at 19:20:54[[⏎]]In reply to`
- ` bags, because we are transforming the Taproom you know and love into a[[ haven]] for flamingos`
- ` dog bite injuries in two rabies endemic areas of south Bhutan.⏎Three hundred[[ and]] twenty four dog`
- ` source tells Y!— Marc J. Spears (@SpearsNBAYahoo[[)]] July 1`
- `ve learned a lot about what you actually need and what is unnecessary. Live[[ and]] learn, right`
- ` I don't rightly know what to do here. Am I stayin[[']] or goin'`
- `!⏎Feeling duped and sceptical⏎She decided what to do.⏎[[']]If I disguise`
- ` while many expected La La Land to do well, few expected it to tie[[ for]] the record of`
- ` it spat-out,⏎Saying, with wonder, to herself⏎[[']]How did it`
- ` that it is not restricted by patents, however that does not have an effect[[ on]] you as a`
- ` product label). The new 150 size is available only as Version[[ I]] with the flat`

</details>

<details><summary>skew #36: 12 strongest D2 contexts (oriented score)</summary>

- ` learning methods with model-based knowledge to develop a robust, end-to[[-]]end learnable`
- ` mannequins to large department stores. The company relocated its production facility to Shaft[[s]]bury in `
- ` was speaking at the event as part of a wider series of meetings with U[[.]]K. government`
- `935 (entitled “Programmable Channel Search Reader”) and U[[.]]S. Patent`
- ` Suite. Avatar now incorporates the benefits of cloud-based, end-to[[-]]end communications and`
- `:30 a.m., noon, 1:30 p[[.]]m. and`
- ` high-rise four-lane span started around 5:30 p[[.]]m. Friday`
- ` CancerCare’s oncology social workers are available to help you face-to[[-]]face, online`
- ` topics:⏎● storm impacts on property⏎● disruptions of day-to[[-]]day life⏎`
- ` member of Social Media Club of Long Island.⏎Through a peer-to[[-]]peer lending site`
- `2.168.0.1-192.1[[6]]8.0`
- ` Administration Building, is open Monday-Thursday, 7:30 a[[.]]m. until`

</details>

**Reading.** The twins are anti-aligned: their direction cos is −0.72.
- **skew #36's heavy tail** is the token inside a punctuation-joined unit:
  - the hyphen of *end-to-end*, *face-to-face*, *day-to-day*, *peer-to-peer*;
  - the period inside *U.K.*, *U.S.*, *p.m.*, *a.m.*
- Its signature (`S`, `m`, `formats`, `g`, `com`) continues such units.
- **logcosh #18:** all 76 of its matches lie on its opposite tail, which is skew #36's heavy tail. Its own heavy tail is mixed: *haven*, *and*, an opening quote, *for*, *on*.
- **So the family is one construction, seen through skew #36.**
  - 70 latents match it. L8 16k small #6235 fires only there (overlap 1.000).
  - It reads as an orthographic-unit continuation, carried by many partial detectors.
  - No e64c rule covers positions inside punctuation; the best is 0.54.

#### L7 logcosh #33

| member | e68 verdict (with / without source) | best \|cos\| (dictionary) | max \|r\| | e64c best rule, \|AUROC\| | e65 B3 / B4 | signature testable (s_self vs base) |
|---|---|---|---|---|---|---|
| logcosh #33 | NOT COVERED / NOT COVERED | 0.51 (rp L7 262k medium) | 0.22 | D, 0.55 | not run (L7) | True (0.305 vs 0.044) |

Census record: R2-replicated **True**; e64c-nameable False; named twin False; identity/generic False; near a named/covered unit False; census tier L7 (no validated behavioural test).
Signature T of logcosh #33: `point` `end` `moment` `time` `:` `same` `meeting` `hard` `help` `way` `back` `well` `very` `to` `get` `7` `4` `an` `not` `from`

**logcosh #33: nearest latents (pooled over the 13 primary dictionaries).** max \|cos\| above the subspace-null p99.9 in 0 of 13 dictionaries (0 without source). Tested latents 321; **50 match** (45 on the heavy side, 5 on the opposite side; 44 outside the source dictionary). Largest tail overlap: L9 16k big #1895, 1.000 (threshold 0.505, null p99 0.065).

| | top \|cos\| latent | cos | match test | top \|r\| latent | r (within type) | match test |
|---|---|---|---|---|---|---|
| 1 | rp L7 262k medium #4061 | 0.514 | pos; sig 0.23 (≥0.17, p99 0.31); ovl 0.82 (≥0.50, p99 0.09); **match** | L7 16k big #5945 | 0.222 (0.23) | pos; sig 0.29 (≥0.17, p99 0.20); ovl 0.97 (≥0.50, p99 0.08); **match** |
| 2 | L8 16k small #4616 | 0.503 | pos; sig 0.06 (≥0.17, p99 0.24); ovl 0.96 (≥0.50, p99 0.10); **match** | L9 16k big #1895 | 0.217 (0.22) | pos; sig 0.32 (≥0.17, p99 0.21); ovl 1.00 (≥0.50, p99 0.07); **match** |
| 3 | L8 16k small #5945 | 0.493 | pos; sig 0.26 (≥0.17, p99 0.24); ovl 0.99 (≥0.50, p99 0.10); **match** | L8 16k big #1461 | 0.216 (0.22) | pos; sig 0.36 (≥0.17, p99 0.25); ovl 0.99 (≥0.50, p99 0.08); **match** |
| 4 | L5 16k small #6063 | 0.488 | pos; sig 0.33 (≥0.17, p99 0.32); ovl 0.59 (≥0.50, p99 0.10); **match** | L5 16k big #1895 | 0.203 (0.21) | pos; sig 0.33 (≥0.17, p99 0.26); ovl 0.99 (≥0.50, p99 0.10); **match** |
| 5 | rp L7 65k medium #1887 | 0.481 | pos; sig 0.01 (≥0.17, p99 0.23); ovl 0.01 (≥0.50, p99 0.08); no match | L6 16k big #1895 | 0.200 (0.20) | pos; sig 0.31 (≥0.17, p99 0.18); ovl 0.99 (≥0.50, p99 0.07); **match** |

**OMP atoms to cos 0.8** (band vs that dictionary's per-mode null; `>` = unreached by k_max, banded as k_max + 1):

| dictionary | logcosh #33 signed / nonneg k08 | null k08 p1 / p5 / p50, signed | nonneg |
|---|---|---|---|
| L5 16k big | 10 [>p50] / 11 [>p50] | 1 / 2 / 8 | 2 / 2 / 9 |
| L5 16k small | 9 [>p50] / 11 [>p50] | 1 / 1.95 / 8.5 | 1 / 2 / 10 |
| L6 16k big | 12 [>p50] / 12 [>p50] | 1 / 2 / 8 | 1 / 2 / 10 |
| L6 16k small | 10 [>p50] / 11 [>p50] | 1 / 1 / 8 | 1 / 2 / 10 |
| L7 16k big | 12 [>p50] / 14 [>p50] | 1 / 2 / 8 | 1 / 2 / 9 |
| L7 16k small | 9 [>p50] / 9 [>p50] | 1 / 1 / 8 | 1 / 1.95 / 8 |
| L8 16k big | 10 [mid] / 11 [mid] | 1 / 1 / 10 | 1 / 2 / 11 |
| L8 16k small | 10 [>p50] / 11 [mid] | 1 / 1 / 9 | 1 / 2 / 11 |
| L9 16k big | 12 [mid] / 15 [>p50] | 1.99 / 2 / 13 | 1.99 / 2 / 14.5 |
| L9 16k small | 14 [>p50] / 15 [>p50] | 1 / 2 / 11.5 | 1 / 2 / 14 |
| rp L7 16k medium | 8 [mid] / 10 [>p50] | 1 / 1 / 8.5 | 1 / 2 / 9.5 |
| rp L7 65k medium | 9 [>p50] / 11 [>p50] | 1 / 1 / 8 | 1 / 1 / 9 |
| rp L7 262k medium | 8 [>p50] / 9 [>p50] | 1 / 1 / 7 | 1 / 1 / 8 |

<details><summary>logcosh #33: 12 strongest D2 contexts (oriented score)</summary>

- `62 per share, according to IPO documents unsealed on Thursday. At[[ that]] price, Williams`
- `You Can Now Get Dib[[s]] On Pat McGrath`
- `- The old city⏎- The Obridorio square⏎TRIP AT[[ A]] GLANCE⏎`
- ` canal. It can be worn 24/7 for several months at[[ a]] time without the`
- `Restoring Indoor Health, One House at[[ a]] Time⏎January`
- ` several times before he and Jane became involved. We weren't intimate at[[ that]] time, and`
- ` world. I felt there was a real lack of support from the colleges at[[ that]] time and I`
- ` the City of Kingston website (Kingston Live) after each meeting. At[[ this]] time, council`
- ` has been "a great life for me in the NFL."⏎"At[[ my]] age, I`
- `27-year-old suffered a serious ankle ligament injury during the Hammers[[’]] clash with Aston`
- ` help one or more of the divorcees recover. That's because at[[ an]] advanced age,`
- ` the Affordable Care Act in ways that would benefit health insurers.⏎But at[[ these]] levels, the`

</details>

**Reading.** The determiner or possessive right after *at*, in fixed phrases: *at that price*, *at a time*, *at that time*, *At this time*, *At my age*, *at an advanced age*, *at these levels*.
- It is a previous-token variable, narrower than e64c's *after in/at* rule, which mixes in the far more frequent *in*.
- No e64c rule names it. The best is the partitive rule, at 0.55.
- Latents that fire almost only in its tail:
  - #1895 in L5, L6 and L9 16k big (overlap 0.99–1.00);
  - L8 16k big #1461 (0.995).

#### L7 logcosh #49 / skew #8

| member | e68 verdict (with / without source) | best \|cos\| (dictionary) | max \|r\| | e64c best rule, \|AUROC\| | e65 B3 / B4 | signature testable (s_self vs base) |
|---|---|---|---|---|---|---|
| logcosh #49 | NOT COVERED / NOT COVERED | 0.60 (rp L7 262k medium) | 0.32 | D, 0.56 | not run (L7) | True (0.130 vs 0.036) |
| skew #8 | NOT COVERED / NOT COVERED | 0.57 (rp L7 262k medium) | 0.31 | D, 0.55 | not run (L7) | True (0.125 vs 0.036) |

Census record: R2-replicated **True**; e64c-nameable False; named twin False; identity/generic False; near a named/covered unit False; census tier L7 (no validated behavioural test).
Signature T of logcosh #49: `doing` `do` `wanted` `say` `did` `best` `?` `these` `about` `most` `how` `want` `what` `going` `you` `right` `of` `like` `us` `think`
Signature T of skew #8: `do` `doing` `about` `wanted` `how` `what` `did` `most` `say` `?` `going` `want` `right` `these` `of` `best` `like` `really` `you` `when`

**logcosh #49: nearest latents (pooled over the 13 primary dictionaries).** max \|cos\| above the subspace-null p99.9 in 0 of 13 dictionaries (0 without source). Tested latents 325; **57 match** (55 on the heavy side, 2 on the opposite side; 49 outside the source dictionary). Largest tail overlap: L9 16k big #3487, 1.000 (threshold 0.505, null p99 0.055).

| | top \|cos\| latent | cos | match test | top \|r\| latent | r (within type) | match test |
|---|---|---|---|---|---|---|
| 1 | rp L7 262k medium #5021 | 0.605 | pos; sig 0.23 (≥0.08, p99 0.30); ovl 0.99 (≥0.51, p99 0.11); **match** | L7 16k big #3487 | 0.317 (0.32) | pos; sig 0.20 (≥0.08, p99 0.18); ovl 0.94 (≥0.51, p99 0.08); **match** |
| 2 | L8 16k small #3487 | 0.580 | pos; sig 0.23 (≥0.08, p99 0.30); ovl 0.97 (≥0.51, p99 0.11); **match** | rp L7 262k medium #5021 | 0.309 (0.32) | pos; sig 0.23 (≥0.08, p99 0.30); ovl 0.99 (≥0.51, p99 0.11); **match** |
| 3 | rp L7 65k medium #1375 | 0.577 | pos; sig 0.10 (≥0.08, p99 0.39); ovl 0.01 (≥0.51, p99 0.08); no match | rp L7 16k medium #3487 | 0.297 (0.30) | pos; sig 0.17 (≥0.08, p99 0.23); ovl 0.93 (≥0.51, p99 0.10); **match** |
| 4 | L7 16k small #3487 | 0.554 | pos; sig 0.17 (≥0.08, p99 0.28); ovl 0.97 (≥0.51, p99 0.12); **match** | L6 16k big #4993 | 0.281 (0.29) | pos; sig 0.16 (≥0.08, p99 0.24); ovl 0.93 (≥0.51, p99 0.10); **match** |
| 5 | L6 16k small #68 | 0.524 | pos; sig 0.10 (≥0.08, p99 0.19); ovl 0.01 (≥0.51, p99 0.09); no match | L7 16k small #3487 | 0.278 (0.28) | pos; sig 0.17 (≥0.08, p99 0.28); ovl 0.97 (≥0.51, p99 0.12); **match** |

**skew #8: nearest latents (pooled over the 13 primary dictionaries).** max \|cos\| above the subspace-null p99.9 in 0 of 13 dictionaries (0 without source). Tested latents 323; **51 match** (51 on the heavy side, 0 on the opposite side; 46 outside the source dictionary). Largest tail overlap: L9 16k big #3487, 0.995 (threshold 0.505, null p99 0.060).

| | top \|cos\| latent | cos | match test | top \|r\| latent | r (within type) | match test |
|---|---|---|---|---|---|---|
| 1 | rp L7 262k medium #5021 | 0.566 | pos; sig 0.23 (≥0.08, p99 0.30); ovl 0.99 (≥0.50, p99 0.10); **match** | L7 16k big #3487 | 0.312 (0.32) | pos; sig 0.20 (≥0.08, p99 0.18); ovl 0.95 (≥0.50, p99 0.08); **match** |
| 2 | rp L7 65k medium #1375 | 0.554 | pos; sig 0.11 (≥0.08, p99 0.39); ovl 0.01 (≥0.50, p99 0.08); no match | rp L7 262k medium #5021 | 0.305 (0.31) | pos; sig 0.23 (≥0.08, p99 0.30); ovl 0.99 (≥0.50, p99 0.10); **match** |
| 3 | L8 16k small #3487 | 0.546 | pos; sig 0.23 (≥0.08, p99 0.30); ovl 0.98 (≥0.50, p99 0.10); **match** | rp L7 16k medium #3487 | 0.292 (0.30) | pos; sig 0.18 (≥0.08, p99 0.23); ovl 0.94 (≥0.50, p99 0.09); **match** |
| 4 | L7 16k small #3487 | 0.527 | pos; sig 0.18 (≥0.08, p99 0.28); ovl 0.98 (≥0.50, p99 0.12); **match** | L6 16k big #4993 | 0.278 (0.28) | pos; sig 0.17 (≥0.08, p99 0.24); ovl 0.93 (≥0.50, p99 0.09); **match** |
| 5 | L6 16k small #68 | 0.497 | pos; sig 0.11 (≥0.08, p99 0.20); ovl 0.01 (≥0.50, p99 0.09); no match | L7 16k small #3487 | 0.275 (0.28) | pos; sig 0.18 (≥0.08, p99 0.28); ovl 0.98 (≥0.50, p99 0.12); **match** |

**OMP atoms to cos 0.8** (band vs that dictionary's per-mode null; `>` = unreached by k_max, banded as k_max + 1):

| dictionary | logcosh #49 signed / nonneg k08 | skew #8 signed / nonneg k08 | null k08 p1 / p5 / p50, signed | nonneg |
|---|---|---|---|---|
| L5 16k big | 12 [>p50] / 14 [>p50] | 12 [>p50] / 14 [>p50] | 1 / 2 / 8 | 2 / 2 / 9 |
| L5 16k small | 13 [>p50] / 15 [>p50] | 12 [>p50] / 13 [>p50] | 1 / 1.95 / 8.5 | 1 / 2 / 10 |
| L6 16k big | 12 [>p50] / 13 [>p50] | 12 [>p50] / 13 [>p50] | 1 / 2 / 8 | 1 / 2 / 10 |
| L6 16k small | 11 [>p50] / 13 [>p50] | 11 [>p50] / 12 [>p50] | 1 / 1 / 8 | 1 / 2 / 10 |
| L7 16k big | 10 [>p50] / 14 [>p50] | 12 [>p50] / 13 [>p50] | 1 / 2 / 8 | 1 / 2 / 9 |
| L7 16k small | 10 [>p50] / 11 [>p50] | 10 [>p50] / 10 [>p50] | 1 / 1 / 8 | 1 / 1.95 / 8 |
| L8 16k big | 15 [>p50] / 18 [>p50] | 16 [>p50] / 16 [>p50] | 1 / 1 / 10 | 1 / 2 / 11 |
| L8 16k small | 12 [>p50] / 13 [>p50] | 12 [>p50] / 15 [>p50] | 1 / 1 / 9 | 1 / 2 / 11 |
| L9 16k big | 18 [>p50] / 21 [>p50] | 19 [>p50] / 19 [>p50] | 1.99 / 2 / 13 | 1.99 / 2 / 14.5 |
| L9 16k small | 16 [>p50] / 18 [>p50] | 16 [>p50] / 19 [>p50] | 1 / 2 / 11.5 | 1 / 2 / 14 |
| rp L7 16k medium | 10 [>p50] / 11 [>p50] | 10 [>p50] / 12 [>p50] | 1 / 1 / 8.5 | 1 / 2 / 9.5 |
| rp L7 65k medium | 8 [mid] / 10 [>p50] | 9 [>p50] / 11 [>p50] | 1 / 1 / 8 | 1 / 1 / 9 |
| rp L7 262k medium | 9 [>p50] / 10 [>p50] | 10 [>p50] / 10 [>p50] | 1 / 1 / 7 | 1 / 1 / 8 |

<details><summary>logcosh #49: 12 strongest D2 contexts (oriented score)</summary>

- ` in my head hitting a disgusting high of triple digits.⏎"So what[[ have]] you been do`
- ` means for your group and your area on our Volunteer FAQ page.⏎What[[ will]] we aim to`
- ` the scene.⏎The resulting images left no one in any doubt about what[[ had]] occurred; they`
- ` to think of and execute new ideas to help grow the business.⏎What[[ Will]] You Be Doing`
- `3helpmecom 27 jan 2018. What[[ were]] the causes of`
- ` inequality is a sure sign our economic model is broken.⏎So what exactly[[ has]] gone wrong?`
- ` the knot sooner⏎TRUMP TAX STORY NO BOMBSHELL⏎What[[ will]] $75`
- ` So here are a few Frequently Asked Questions (and answers.)⏎“What[[ can]] I do?`
- `, before leaving for Malawi the question frequently posed to me was, “what[[ will]] you be doing`
- ` my stretches (or find a regular massage) and I'm wondering what[[ the]] best way to`
- `Blog Rocket Review – What[[ Are]] Click Magnets`
- ` the event that you are obstructed when visiting a site, this is the thing[[ that]] will help you`

</details>

<details><summary>skew #8: 12 strongest D2 contexts (oriented score)</summary>

- ` in my head hitting a disgusting high of triple digits.⏎"So what[[ have]] you been do`
- ` the scene.⏎The resulting images left no one in any doubt about what[[ had]] occurred; they`
- ` means for your group and your area on our Volunteer FAQ page.⏎What[[ will]] we aim to`
- ` the knot sooner⏎TRUMP TAX STORY NO BOMBSHELL⏎What[[ will]] $75`
- ` to think of and execute new ideas to help grow the business.⏎What[[ Will]] You Be Doing`
- ` inequality is a sure sign our economic model is broken.⏎So what exactly[[ has]] gone wrong?`
- ` media center, this is the video for you.⏎Shannon Hager: What[[ are]] the Longhorn`
- `3helpmecom 27 jan 2018. What[[ were]] the causes of`
- ` wedding attire for men. So let's get into it.⏎What[[ the]] groom and grooms`
- ` the event that you are obstructed when visiting a site, this is the thing[[ that]] will help you`
- `, before leaving for Malawi the question frequently posed to me was, “what[[ will]] you be doing`
- `, which surveyed 538 UK adults, asked respondents what they define[[ as]] good and bad`

</details>

**Reading.** The token right after *what*, mostly an auxiliary in a wh-question or wh-clause: *what have*, *What will*, *what had*, *What were*, *What can*, *what exactly has*.
- It is a previous-token variable, the kind e64c found among the cleanest directions (after *the*, after *in/at*, after a conjunction). But its trigger is *what*, and no e64c rule has that trigger; the best is 0.56.
- Latent #3487 of the L7, L8 and L9 16k dictionaries, and of rp L7 16k medium, fires almost only in its tail (overlap 0.92–1.00).

#### L7 logcosh #52

| member | e68 verdict (with / without source) | best \|cos\| (dictionary) | max \|r\| | e64c best rule, \|AUROC\| | e65 B3 / B4 | signature testable (s_self vs base) |
|---|---|---|---|---|---|---|
| logcosh #52 | NOT COVERED / NOT COVERED | 0.72 (L7 16k small) | 0.21 | B, 0.57 | not run (L7) | False (0.155 vs 0.092) |

Census record: R2-replicated **True**; e64c-nameable False; named twin False; identity/generic False; near a named/covered unit False; census tier L7 (no validated behavioural test).
Signature T of logcosh #52: `announce` `opportunity` `—` `being` `feel` `But` `but` `because` `that` `!` `when` `-` `them` `it` `I` `,` `.` `this` `6` `how`

**logcosh #52: nearest latents (pooled over the 13 primary dictionaries).** max \|cos\| above the subspace-null p99.9 in 0 of 13 dictionaries (0 without source). Tested latents 280; **19 match** (19 on the heavy side, 0 on the opposite side; 17 outside the source dictionary). Largest tail overlap: L5 16k small #7457, 0.691 (threshold 0.505, null p99 0.118).

| | top \|cos\| latent | cos | match test | top \|r\| latent | r (within type) | match test |
|---|---|---|---|---|---|---|
| 1 | L7 16k small #222 | 0.718 | pos; sig 0.26 (≥0.12, p99 0.40); ovl 0.13 (≥0.50, p99 0.12); no match | rp L7 262k medium #3973 | 0.209 (0.23) | pos; sig 0.24 (≥0.12, p99 0.37); ovl 0.66 (≥0.50, p99 0.16); **match** |
| 2 | rp L7 16k medium #222 | 0.630 | pos; sig 0.24 (≥0.12, p99 0.36); ovl 0.10 (≥0.50, p99 0.15); no match | L7 16k big #3314 | 0.201 (0.21) | pos; sig 0.15 (≥0.12, p99 0.41); ovl 0.67 (≥0.50, p99 0.08); **match** |
| 3 | L7 16k big #222 | 0.625 | pos; sig 0.24 (≥0.12, p99 0.41); ovl 0.12 (≥0.50, p99 0.08); no match | L5 16k big #7457 | 0.188 (0.19) | pos; sig 0.16 (≥0.12, p99 0.37); ovl 0.69 (≥0.50, p99 0.08); **match** |
| 4 | L8 16k small #770 | 0.599 | pos; sig 0.17 (≥0.12, p99 0.46); ovl 0.33 (≥0.50, p99 0.16); no match | L7 16k big #11621 | 0.187 (0.19) | pos; sig 0.20 (≥0.12, p99 0.41); ovl 0.64 (≥0.50, p99 0.08); **match** |
| 5 | L6 16k small #742 | 0.579 | pos; sig 0.18 (≥0.12, p99 0.36); ovl 0.23 (≥0.50, p99 0.14); no match | L9 16k big #11621 | 0.176 (0.18) | pos; sig 0.17 (≥0.12, p99 0.37); ovl 0.66 (≥0.50, p99 0.08); **match** |

**OMP atoms to cos 0.8** (band vs that dictionary's per-mode null; `>` = unreached by k_max, banded as k_max + 1):

| dictionary | logcosh #52 signed / nonneg k08 | null k08 p1 / p5 / p50, signed | nonneg |
|---|---|---|---|
| L5 16k big | 11 [>p50] / 14 [>p50] | 1 / 2 / 8 | 2 / 2 / 9 |
| L5 16k small | 10 [>p50] / 10 [mid] | 1 / 1.95 / 8.5 | 1 / 2 / 10 |
| L6 16k big | 10 [>p50] / 11 [>p50] | 1 / 2 / 8 | 1 / 2 / 10 |
| L6 16k small | 10 [>p50] / 11 [>p50] | 1 / 1 / 8 | 1 / 2 / 10 |
| L7 16k big | 8 [mid] / 9 [mid] | 1 / 2 / 8 | 1 / 2 / 9 |
| L7 16k small | 4 [mid] / 4 [mid] | 1 / 1 / 8 | 1 / 1.95 / 8 |
| L8 16k big | 11 [>p50] / 11 [mid] | 1 / 1 / 10 | 1 / 2 / 11 |
| L8 16k small | 10 [>p50] / 10 [mid] | 1 / 1 / 9 | 1 / 2 / 11 |
| L9 16k big | 16 [>p50] / 16 [>p50] | 1.99 / 2 / 13 | 1.99 / 2 / 14.5 |
| L9 16k small | 14 [>p50] / 18 [>p50] | 1 / 2 / 11.5 | 1 / 2 / 14 |
| rp L7 16k medium | 7 [mid] / 8 [mid] | 1 / 1 / 8.5 | 1 / 2 / 9.5 |
| rp L7 65k medium | 8 [mid] / 8 [mid] | 1 / 1 / 8 | 1 / 1 / 9 |
| rp L7 262k medium | 7 [mid] / 7 [mid] | 1 / 1 / 7 | 1 / 1 / 8 |

<details><summary>logcosh #52: 12 strongest D2 contexts (oriented score)</summary>

- ` from the Petals & Paisleys Specialty Designer Series Paper Pack. I love[[ the]] rich colors of`
- ` within the community. Well, I personally want to say that "We love[[ it]]".⏎"`
- ` surgery, wound care and reconstruction, and mass removal.⏎We take pride[[ in]] our ability to`
- `⏎The event brought a lively discussion, almost immediately.⏎"I enjoyed[[ the]] panel because it`
- ` in a news release recapping the wide-ranging probe that she is proud[[ of]] the case and`
- `attly). CM has just celebrated its first birthday, and the best part[[ is]] that this lecture`
- ` had better all pray that He is merciful.⏎Whew, powerful words[[.]]⏎Gobry`
- ` always. His character isn't Newton, but himself. But I like[[ that]] because he is`
- `um, Frankie’s performances are always the real deal. “I love[[ the]] honesty in her`
- ` little things, like arranging my office furniture . . .⏎|I like[[ the]] simple “door`
- ` unconstructive responses to criticism before (here and here). I also like[[ the]] study design which`
- ` the center of the small room which backed onto the open grill. We liked[[ the]] fact that you`

</details>

**Reading.** The token right after an affective or evaluative verb or adjective: *love the*, *love it*, *pride in*, *enjoyed the*, *proud of*, *like that*, *liked the*.
- It is the weakest family on every latent measure:
  - 19 matches, all by overlap; its largest overlap is 0.69;
  - fixed-set match rate 0.15, with 4 of 40 nulls at or above it;
  - its signature is not testable.
- Its best |cos| (0.72, L7 16k small #222) belongs to a latent that fails the match.

#### L7 logcosh #57 / skew #33

| member | e68 verdict (with / without source) | best \|cos\| (dictionary) | max \|r\| | e64c best rule, \|AUROC\| | e65 B3 / B4 | signature testable (s_self vs base) |
|---|---|---|---|---|---|---|
| logcosh #57 | NOT COVERED / NOT COVERED | 0.52 (L6 16k small) | 0.21 | C_inat, 0.56 | not run (L7) | True (0.175 vs 0.036) |
| skew #33 | NOT COVERED / NOT COVERED | 0.52 (L9 16k big) | 0.24 | D, 0.56 | not run (L7) | True (0.175 vs 0.034) |

Census record: R2-replicated **True**; e64c-nameable False; named twin False; identity/generic False; near a named/covered unit False; census tier L7 (no validated behavioural test).
Signature T of logcosh #57: `things` `com` `people` `other` `information` `already` `time` `may` `work` `-` `few` `different` `year` `good` `of` `best` `need` `were` `at` `on`
Signature T of skew #33: `com` `things` `people` `other` `information` `time` `different` `good` `being` `-` `need` `made` `best` `work` `of` `were` `more` `at` `get` `most`

**logcosh #57: nearest latents (pooled over the 13 primary dictionaries).** max \|cos\| above the subspace-null p99.9 in 0 of 13 dictionaries (0 without source). Tested latents 318; **62 match** (62 on the heavy side, 0 on the opposite side; 60 outside the source dictionary). Largest tail overlap: L7 16k small #9554, 0.966 (threshold 0.505, null p99 0.115).

| | top \|cos\| latent | cos | match test | top \|r\| latent | r (within type) | match test |
|---|---|---|---|---|---|---|
| 1 | L6 16k small #728 | 0.517 | pos; sig 0.46 (≥0.11, p99 0.22); ovl 0.09 (≥0.51, p99 0.14); **match** | rp L7 65k medium #4192 | 0.212 (0.22) | pos; sig 0.07 (≥0.11, p99 0.40); ovl 0.52 (≥0.51, p99 0.11); **match** |
| 2 | rp L7 16k medium #625 | 0.475 | pos; sig 0.04 (≥0.11, p99 0.22); ovl 0.18 (≥0.51, p99 0.14); no match | L7 16k big #4792 | 0.200 (0.21) | pos; sig 0.06 (≥0.11, p99 0.22); ovl 0.43 (≥0.51, p99 0.08); no match |
| 3 | rp L7 16k medium #868 | 0.474 | pos; sig 0.41 (≥0.11, p99 0.22); ovl 0.29 (≥0.51, p99 0.14); **match** | rp L7 262k medium #11808 | 0.194 (0.20) | pos; sig 0.07 (≥0.11, p99 0.30); ovl 0.47 (≥0.51, p99 0.08); no match |
| 4 | L6 16k big #728 | 0.460 | pos; sig 0.23 (≥0.11, p99 0.25); ovl 0.04 (≥0.51, p99 0.08); no match | L6 16k big #1494 | 0.176 (0.19) | pos; sig 0.17 (≥0.11, p99 0.25); ovl 0.77 (≥0.51, p99 0.08); **match** |
| 5 | rp L7 262k medium #9332 | 0.454 | pos; sig 0.79 (≥0.11, p99 0.30); ovl 0.10 (≥0.51, p99 0.08); **match** | rp L7 262k medium #1137 | 0.169 (0.17) | pos; sig 0.03 (≥0.11, p99 0.30); ovl 0.26 (≥0.51, p99 0.08); no match |

**skew #33: nearest latents (pooled over the 13 primary dictionaries).** max \|cos\| above the subspace-null p99.9 in 0 of 13 dictionaries (0 without source). Tested latents 299; **62 match** (62 on the heavy side, 0 on the opposite side; 59 outside the source dictionary). Largest tail overlap: L6 16k small #3398, 0.980 (threshold 0.505, null p99 0.151).

| | top \|cos\| latent | cos | match test | top \|r\| latent | r (within type) | match test |
|---|---|---|---|---|---|---|
| 1 | L9 16k big #14247 | 0.515 | pos; sig 0.07 (≥0.10, p99 0.12); ovl 0.57 (≥0.51, p99 0.07); **match** | L7 16k big #4792 | 0.239 (0.25) | pos; sig 0.07 (≥0.10, p99 0.21); ovl 0.58 (≥0.51, p99 0.09); **match** |
| 2 | L6 16k small #728 | 0.490 | pos; sig 0.48 (≥0.10, p99 0.21); ovl 0.09 (≥0.51, p99 0.15); **match** | rp L7 65k medium #4192 | 0.228 (0.23) | pos; sig 0.09 (≥0.10, p99 0.39); ovl 0.55 (≥0.51, p99 0.09); **match** |
| 3 | L7 16k small #566 | 0.466 | pos; sig 0.06 (≥0.10, p99 0.30); ovl 0.12 (≥0.51, p99 0.10); no match | L6 16k big #1494 | 0.203 (0.22) | pos; sig 0.19 (≥0.10, p99 0.26); ovl 0.78 (≥0.51, p99 0.09); **match** |
| 4 | rp L7 262k medium #9332 | 0.463 | pos; sig 0.79 (≥0.10, p99 0.30); ovl 0.09 (≥0.51, p99 0.08); **match** | L8 16k big #13215 | 0.203 (0.21) | pos; sig 0.15 (≥0.10, p99 0.17); ovl 0.84 (≥0.51, p99 0.07); **match** |
| 5 | rp L7 16k medium #625 | 0.463 | pos; sig 0.04 (≥0.10, p99 0.22); ovl 0.18 (≥0.51, p99 0.11); no match | rp L7 16k medium #1301 | 0.192 (0.20) | pos; sig 0.20 (≥0.10, p99 0.22); ovl 0.77 (≥0.51, p99 0.11); **match** |

**OMP atoms to cos 0.8** (band vs that dictionary's per-mode null; `>` = unreached by k_max, banded as k_max + 1):

| dictionary | logcosh #57 signed / nonneg k08 | skew #33 signed / nonneg k08 | null k08 p1 / p5 / p50, signed | nonneg |
|---|---|---|---|---|
| L5 16k big | 14 [>p50] / 16 [>p50] | 11 [>p50] / 13 [>p50] | 1 / 2 / 8 | 2 / 2 / 9 |
| L5 16k small | 16 [>p50] / 18 [>p50] | 13 [>p50] / 13 [>p50] | 1 / 1.95 / 8.5 | 1 / 2 / 10 |
| L6 16k big | 13 [>p50] / 15 [>p50] | 12 [>p50] / 12 [>p50] | 1 / 2 / 8 | 1 / 2 / 10 |
| L6 16k small | 12 [>p50] / 12 [>p50] | 9 [>p50] / 10 [mid] | 1 / 1 / 8 | 1 / 2 / 10 |
| L7 16k big | 11 [>p50] / 11 [>p50] | 11 [>p50] / 11 [>p50] | 1 / 2 / 8 | 1 / 2 / 9 |
| L7 16k small | 10 [>p50] / 11 [>p50] | 8 [mid] / 8 [mid] | 1 / 1 / 8 | 1 / 1.95 / 8 |
| L8 16k big | 13 [>p50] / 15 [>p50] | 13 [>p50] / 13 [>p50] | 1 / 1 / 10 | 1 / 2 / 11 |
| L8 16k small | 10 [>p50] / 12 [>p50] | 8 [mid] / 9 [mid] | 1 / 1 / 9 | 1 / 2 / 11 |
| L9 16k big | 16 [>p50] / 18 [>p50] | 15 [>p50] / 14 [mid] | 1.99 / 2 / 13 | 1.99 / 2 / 14.5 |
| L9 16k small | 14 [>p50] / 16 [>p50] | 12 [>p50] / 14 [mid] | 1 / 2 / 11.5 | 1 / 2 / 14 |
| rp L7 16k medium | 10 [>p50] / 10 [>p50] | 9 [>p50] / 9 [mid] | 1 / 1 / 8.5 | 1 / 2 / 9.5 |
| rp L7 65k medium | 10 [>p50] / 12 [>p50] | 10 [>p50] / 10 [>p50] | 1 / 1 / 8 | 1 / 1 / 9 |
| rp L7 262k medium | 10 [>p50] / 10 [>p50] | 8 [>p50] / 8 [mid] | 1 / 1 / 7 | 1 / 1 / 8 |

<details><summary>logcosh #57: 12 strongest D2 contexts (oriented score)</summary>

- ` for his life.⏎But once God gave a response to all of Moses[[’]] weak excuses,`
- ` allegations of abuse among its supplier network in China. Many, if not most[[,]] of the workers`
- ` out that sultry stare!⏎This beautiful black dress from ModCloth hits all[[ the]] right jazz notes`
- ` distinction of being recognized for its general irreverence and overall humorousness. All[[ this]] does, frankly`
- ` Lightbox Popup and a Custom Opt-in form in my sidebar. All[[ this]] was done with`
- ` Kristofferson⏎Frankie Hart, vocalist and guitarist, has shared many[[ a]] stage with Ray`
- ` buy them from other travelers.⏎Flight tickets are transferable. Lots of airlines[[,]] mostly in Europe`
- ` Board Readiness:⏎All women on the list should have at least five years[[’]] experience in one`
- ` southern Missouri.⏎Information from: The Kansas City Star, http://www[[.]]kcstar.`
- `This website (http://www[[.]]crownwatchblog`
- ` connected to the Internet is patently absurd these days.⏎But all of[[ this]] rapid progress is`
- ` pitched.⏎Through it all, the team pushed on. Much of that[[ was]] due to the`

</details>

<details><summary>skew #33: 12 strongest D2 contexts (oriented score)</summary>

- ` for his life.⏎But once God gave a response to all of Moses[[’]] weak excuses,`
- ` out that sultry stare!⏎This beautiful black dress from ModCloth hits all[[ the]] right jazz notes`
- ` Lightbox Popup and a Custom Opt-in form in my sidebar. All[[ this]] was done with`
- ` Kristofferson⏎Frankie Hart, vocalist and guitarist, has shared many[[ a]] stage with Ray`
- ` distinction of being recognized for its general irreverence and overall humorousness. All[[ this]] does, frankly`
- ` that I can hardly drag myself out of bed, get dressed (and all[[ that]] goes with that`
- ` allegations of abuse among its supplier network in China. Many, if not most[[,]] of the workers`
- ` connected to the Internet is patently absurd these days.⏎But all of[[ this]] rapid progress is`
- ` Ranas (Littlefield of Frogs).⏎Or you can skip all[[ that]] and read about`
- `⏎That's what it looked like once I had packaged almost all of[[ it]] up. I`
- ` Very honest and priced well.”⏎“Our experience with PSG has been nothing[[ but]] positive since the`
- ` am in Orlando, Florida for the IBM Edge conference. Thursday evening after all[[ the]] other sessions,`

</details>

**Reading.** The token right after a quantifier: *all the*, *All this*, *all of this*, *many a*, *most,*, *Much of that was*.
- It sits next to e64c's partitive rule, which requires *of*. Its best e64c rules (after *in/at* for logcosh #57, partitive for skew #33) reach only 0.56.
- Both members match 62 latents each. The largest overlaps:
  - L7 16k small #9554 (0.97–0.98);
  - #13215 in rp L7 16k medium and L8 16k small (0.90–0.95).

#### L12 logcosh #45 / skew #5

| member | e68 verdict (with / without source) | best \|cos\| (dictionary) | max \|r\| | e64c best rule, \|AUROC\| | e65 B3 / B4 | signature testable (s_self vs base) |
|---|---|---|---|---|---|---|
| logcosh #45 | NOT COVERED / NOT COVERED | 0.54 (L12 16k small) | 0.29 | C_inat, 0.63 | False / None | True (0.160 vs 0.026) |
| skew #5 | NOT COVERED / NOT COVERED | 0.55 (L11 16k small) | 0.28 | C_inat, 0.55 | False / None | True (0.100 vs 0.035) |

Census record: R2-replicated **True**; e64c-nameable False; named twin False; identity/generic False; near a named/covered unit False; census tier B.
Signature T of logcosh #45: `share` `article` `few` `post` `below` `look` `:` `here` `we` `today` `!` `find` `going` `my` `some` `how` `first` `from` `take` `I`
Signature T of skew #5: `share` `look` `today` `post` `below` `few` `here` `part` `I` `we` `first` `I` `how` `my` `this` `:` `!` `with` `some` `are`

**logcosh #45: nearest latents (pooled over the 14 primary dictionaries).** max \|cos\| above the subspace-null p99.9 in 0 of 14 dictionaries (0 without source). Tested latents 651; **49 match** (49 on the heavy side, 0 on the opposite side; 43 outside the source dictionary). Largest tail overlap: L10 16k big #3474, 0.610 (threshold 0.505, null p99 0.050).

| | top \|cos\| latent | cos | match test | top \|r\| latent | r (within type) | match test |
|---|---|---|---|---|---|---|
| 1 | L12 16k small #1276 | 0.544 | pos; sig 0.21 (≥0.09, p99 0.13); ovl 0.58 (≥0.51, p99 0.08); **match** | L12 16k big #1423 | 0.289 (0.30) | pos; sig 0.08 (≥0.09, p99 0.12); ovl 0.55 (≥0.51, p99 0.08); **match** |
| 2 | L11 16k small #1018 | 0.524 | pos; sig 0.09 (≥0.09, p99 0.14); ovl 0.50 (≥0.51, p99 0.08); no match | rp L13 65k big #1459 | 0.278 (0.30) | pos; sig 0.14 (≥0.09, p99 0.15); ovl 0.50 (≥0.51, p99 0.06); no match |
| 3 | L12 16k big #1423 | 0.485 | pos; sig 0.08 (≥0.09, p99 0.12); ovl 0.55 (≥0.51, p99 0.08); **match** | L10 16k big #3474 | 0.257 (0.26) | pos; sig 0.10 (≥0.09, p99 0.10); ovl 0.61 (≥0.51, p99 0.05); **match** |
| 4 | L12 16k small #12830 | 0.437 | pos; sig 0.06 (≥0.09, p99 0.13); ovl 0.36 (≥0.51, p99 0.08); no match | L13 16k big #1276 | 0.233 (0.25) | pos; sig 0.18 (≥0.09, p99 0.10); ovl 0.58 (≥0.51, p99 0.11); **match** |
| 5 | L12 16k big #9505 | -0.437 | pos; sig 0.11 (≥0.09, p99 0.12); ovl 0.12 (≥0.51, p99 0.08); no match | L12 16k small #1276 | 0.222 (0.23) | pos; sig 0.21 (≥0.09, p99 0.13); ovl 0.58 (≥0.51, p99 0.08); **match** |

**skew #5: nearest latents (pooled over the 14 primary dictionaries).** max \|cos\| above the subspace-null p99.9 in 0 of 14 dictionaries (0 without source). Tested latents 557; **32 match** (32 on the heavy side, 0 on the opposite side; 28 outside the source dictionary). Largest tail overlap: L11 16k small #1018, 0.525 (threshold 0.505, null p99 0.084).

| | top \|cos\| latent | cos | match test | top \|r\| latent | r (within type) | match test |
|---|---|---|---|---|---|---|
| 1 | L11 16k small #1018 | 0.551 | pos; sig 0.09 (≥0.07, p99 0.16); ovl 0.53 (≥0.50, p99 0.08); **match** | L12 16k big #1423 | 0.278 (0.29) | pos; sig 0.11 (≥0.07, p99 0.13); ovl 0.50 (≥0.50, p99 0.08); no match |
| 2 | L12 16k small #1276 | 0.523 | pos; sig 0.20 (≥0.07, p99 0.15); ovl 0.49 (≥0.50, p99 0.07); **match** | rp L13 65k big #1459 | 0.265 (0.29) | pos; sig 0.12 (≥0.07, p99 0.18); ovl 0.43 (≥0.50, p99 0.08); no match |
| 3 | L10 16k small #2575 | 0.473 | pos; sig 0.06 (≥0.07, p99 0.19); ovl 0.20 (≥0.50, p99 0.10); no match | L10 16k big #3474 | 0.225 (0.23) | pos; sig 0.10 (≥0.07, p99 0.16); ovl 0.47 (≥0.50, p99 0.07); no match |
| 4 | L12 16k big #1423 | 0.461 | pos; sig 0.11 (≥0.07, p99 0.13); ovl 0.50 (≥0.50, p99 0.08); no match | rp L13 65k medium #1459 | 0.220 (0.23) | pos; sig 0.08 (≥0.07, p99 0.17); ovl 0.26 (≥0.50, p99 0.11); no match |
| 5 | L11 16k small #1541 | 0.457 | pos; sig 0.07 (≥0.07, p99 0.16); ovl 0.21 (≥0.50, p99 0.08); no match | L11 16k small #1018 | 0.215 (0.22) | pos; sig 0.09 (≥0.07, p99 0.16); ovl 0.53 (≥0.50, p99 0.08); **match** |

**OMP atoms to cos 0.8** (band vs that dictionary's per-mode null; `>` = unreached by k_max, banded as k_max + 1):

| dictionary | logcosh #45 signed / nonneg k08 | skew #5 signed / nonneg k08 | null k08 p1 / p5 / p50, signed | nonneg |
|---|---|---|---|---|
| L10 16k big | 32 [>p50] / 37 [>p50] | 28 [mid] / 32 [mid] | 5.99 / 8 / 31 | 6.99 / 10.85 / 36 |
| L10 16k small | 30 [>p50] / 34 [mid] | 24 [mid] / 26 [mid] | 6.99 / 7.95 / 29 | 6.99 / 9 / 36.5 |
| L11 16k big | 20 [mid] / 25 [mid] | 20 [mid] / 24 [mid] | 5 / 8.95 / 28 | 5.99 / 10 / 31.5 |
| L11 16k small | 17 [mid] / 18 [mid] | 14 [mid] / 14 [mid] | 5 / 8 / 26 | 6 / 9.95 / 32 |
| L12 16k big | 11 [mid] / 14 [mid] | 12 [mid] / 13 [mid] | 4.99 / 6 / 18 | 6 / 7 / 21.5 |
| L12 16k small | 10 [mid] / 11 [mid] | 9 [mid] / 9 [mid] | 4 / 6 / 17 | 5.98 / 8 / 22 |
| L13 16k big | 23 [mid] / 27 [mid] | 20 [mid] / 26 [mid] | 7 / 8 / 25 | 7.99 / 10.95 / 30 |
| L13 16k small | 21 [mid] / 25 [mid] | 19 [mid] / 21 [mid] | 5.99 / 7.95 / 23 | 7.99 / 8.95 / 30 |
| L14 16k big | 39 [>p50] / 48 [>p50] | 35 [>p50] / 37 [mid] | 8.98 / 10.95 / 33.5 | 10.98 / 13 / 39.5 |
| L14 16k small | 32 [>p50] / 36 [mid] | 27 [mid] / 31 [mid] | 7.99 / 10 / 31 | 8.99 / 12 / 41.5 |
| rp L13 16k medium | 26 [mid] / 33 [mid] | 23 [mid] / 28 [mid] | 6.99 / 10.95 / 28 | 9 / 11 / 36 |
| rp L13 65k medium | 30 [>p50] / 33 [mid] | 28 [>p50] / 29 [mid] | 7.98 / 9 / 27 | 9 / 11.9 / 33 |
| rp L13 65k big | 30 [mid] / 34 [mid] | 29 [mid] / 34 [mid] | 9.99 / 11 / 30 | 11.99 / 14 / 38 |
| rp L13 262k medium | 29 [>p50] / 30 [mid] | 24 [mid] / 27 [mid] | 9.29 / 10.45 / 26 | 11.29 / 12 / 31.5 |

**(b) sparsity and (c) pre-check.**

| member | (b) class, with source (S1 / S5 / H; per mode s, nn) | without source | (c) held-out R² at K = 4 / 8 / 16 / 32 / 64 | K\* |
|---|---|---|---|---|
| logcosh #45 | **chance-like** (0 / 0 / 2; s 0/0/6, nn 0/0/2) | chance-like (0 / 0 / 2) | 0.121 / 0.143 / 0.157 / 0.198 / 0.233 | >64 |
| skew #5 | **chance-like** (0 / 0 / 0; s 0/0/2, nn 0/0/0) | chance-like (0 / 0 / 0) | 0.111 / 0.140 / 0.175 / 0.200 / 0.219 | >64 |

**logcosh #45: L12 16k-small latents** (top 3 by \|cos\| and by \|r\|; their 12 strongest D2 contexts; share of those 12 rows in the member's 1% heavy tail):

| latent | cos | r | e68 match | top-12 in tail | strongest contexts (first 3) |
|---|---|---|---|---|---|
| #1276 | 0.544 | 0.222 | True | 8/12 | ` investors. After that, the project was renamed to Free TON. In this[[ article]], we will` ⏐ ` friends but you were not sure whether to use it or not. In this[[ article]], we will` ⏐ ` installed – whether in windows, frames, or any other format? In this[[ article]], we’` |
| #12830 | 0.437 | 0.091 | False | 8/12 | `Steve Kolbe is joined[[ by]] Kevin Thomas &` ⏐ `Steve Kolbe is[[ joined]] by Kevin Thomas` ⏐ `Steve Kolbe is joined by Kevin Thomas & Andy Snakovsky[[ to]] bring you the` |
| #5296 | 0.374 | 0.094 | True | 1/12 | `paca that David & I are buying together.⏎On the knitting front[[,]] yesterday I seemed` ⏐ `Obviously, i survived Xmas in Chicago with the parents[[.]] Actually, it` ⏐ ` chores, make dinner, and collapse. This has to get better!⏎[[The]] bridal shower invitations` |
| #3065 | — | 0.149 | True | 11/12 | ` are 7 lesser known facts about fragrances that you probably didn’t know[[.]]⏎1.` ⏐ `, yet unsustainably fished, favorites with their more ocean-friendly replacements[[.]]⏎ATLANTIC` ⏐ ` could fit with what the Patriots like to do on either side of the ball[[.]]⏎Think of` |
| #812 | — | 0.119 | True | 3/12 | ` connectivity. There are⏎no proxies operating on the server.⏎Here'[[s]] how the server` ⏐ ` choosing the “Process > Stitch” command will invoke the video Stitch dialog as[[ below]]:⏎Snap` ⏐ `09, the theme is 'software (r)evolutions'. Below[[ is]] a blurb` |

**skew #5: L12 16k-small latents** (top 3 by \|cos\| and by \|r\|; their 12 strongest D2 contexts; share of those 12 rows in the member's 1% heavy tail):

| latent | cos | r | e68 match | top-12 in tail | strongest contexts (first 3) |
|---|---|---|---|---|---|
| #1276 | 0.523 | 0.202 | True | 7/12 | ` investors. After that, the project was renamed to Free TON. In this[[ article]], we will` ⏐ ` friends but you were not sure whether to use it or not. In this[[ article]], we will` ⏐ ` installed – whether in windows, frames, or any other format? In this[[ article]], we’` |
| #8882 | 0.428 | 0.086 | False | 1/12 | `I decide to embed a weekly[[ post]] on my blog` ⏐ `The importance of appearing confident⏎I've been neglecting this[[ blog]] for some time` ⏐ ` it’s been almost a whole year since I’ve written anything on[[ here]]. Oops!` |
| #2072 | 0.427 | 0.094 | True | 5/12 | `012⏎This is still a work in progress. I've[[ just]] ordered some Tim` ⏐ ` you for reading and for your support!⏎I’m updating this post[[ since]] finding out some` ⏐ `I almost forgot[[ to]] write this race` |
| #3065 | — | 0.123 | False | 5/12 | ` are 7 lesser known facts about fragrances that you probably didn’t know[[.]]⏎1.` ⏐ `, yet unsustainably fished, favorites with their more ocean-friendly replacements[[.]]⏎ATLANTIC` ⏐ ` could fit with what the Patriots like to do on either side of the ball[[.]]⏎Think of` |
| #5296 | 0.422 | 0.113 | True | 3/12 | `paca that David & I are buying together.⏎On the knitting front[[,]] yesterday I seemed` ⏐ `Obviously, i survived Xmas in Chicago with the parents[[.]] Actually, it` ⏐ ` chores, make dinner, and collapse. This has to get better!⏎[[The]] bridal shower invitations` |

<details><summary>logcosh #45: 12 strongest D2 contexts (oriented score)</summary>

- ` all! I’ve picked a few of my favourite items out to share[[,]] ones I will`
- ` May 2018⏎Just some snaps from the week gone by[[,]] despite the rain`
- ` last session, as people fly out Friday morning. Here are the questions and[[ the]] panel responses during`
- ` price distortions and offer special trading opportunities.⏎The below is the third instalment[[ of]] this site’`
- ` head of Britains’s biggest transport union, the RMT. Below[[,]] Crow explains why`
- `Today I’m going[[ to]] show you how`
- `, 23 May 2018⏎Just some snaps from[[ the]] week gone by`
- ` Clare kindly wrote a piece for us to share to mark Severe ME Awareness Day[[,]] 8th`
- `0p will be available from either Verizon Wireless or Sprint, and we here[[ at]] Digital Lifestyle Magazine`
- ` Jenkin, CEO of Nimbla, joins Robin on Podcast 18[[8]] to discuss why`
- ` and try to live my life through, a lens of social justice. This[[ piece]] has no judgment`
- ` notification can read all the information related to UP NHM CHO Recruitment in the[[ table]] below.⏎`

</details>

<details><summary>skew #5: 12 strongest D2 contexts (oriented score)</summary>

- ` May 2018⏎Just some snaps from the week gone by[[,]] despite the rain`
- ` price distortions and offer special trading opportunities.⏎The below is the third instalment[[ of]] this site’`
- ` all! I’ve picked a few of my favourite items out to share[[,]] ones I will`
- ` Clare kindly wrote a piece for us to share to mark Severe ME Awareness Day[[,]] 8th`
- `, 23 May 2018⏎Just some snaps from[[ the]] week gone by`
- ` last session, as people fly out Friday morning. Here are the questions and[[ the]] panel responses during`
- `0p will be available from either Verizon Wireless or Sprint, and we here[[ at]] Digital Lifestyle Magazine`
- `3 May 2018⏎Just some snaps from the week gone[[ by]], despite the`
- `, I hope that various artists will find the time to contribute their perspectives to[[ the]] Nasher Blog`
- ` Jenkin, CEO of Nimbla, joins Robin on Podcast 18[[8]] to discuss why`
- ` Evening” spanned decades, and generations, of Kiwi Motorsport design. This is[[ a]] CAD blog and`
- ` that explaining out of the way, buckle down, this will be a long[[ one]]. With the`

</details>

**Reading.** The writer's framing of their own post or article: *to share,*, *the week gone by,*, *The below is the third instalment of*, *Below,*, *Today I'm going to show you*, *we here at* ⟨site⟩, *This piece*, *the table below*.
- Its signature: *share, article, few, post, below, look, here, today*.
- **In 16k small, latent #1276 is the same construction:** *In this **article**, we will*; *going **to** show you*.
  - cos 0.54 and r 0.22, an e68 match, and 8 of its 12 strongest rows in logcosh #45's tail.
- Latents #8882 (*weekly **post***, *this **blog***) and #12830 (*joined **by***, a podcast intro) carry pieces of it.
- e64c's *after in/at* rule reaches 0.63 on logcosh #45, just under the 0.65 nameability bar.
- **Existing latents carry its contexts, but not sparsely.**
  - (b) reads chance-like.
  - Its fixed-set match rate is inside the null spread (5 and 17 of 40 nulls ≥).

#### L12 logcosh #54 / skew #54

| member | e68 verdict (with / without source) | best \|cos\| (dictionary) | max \|r\| | e64c best rule, \|AUROC\| | e65 B3 / B4 | signature testable (s_self vs base) |
|---|---|---|---|---|---|---|
| logcosh #54 | NOT COVERED / NOT COVERED | 0.49 (L14 16k small) | 0.22 | B, 0.56 | True / True | True (0.115 vs 0.011) |
| skew #54 | NOT COVERED / NOT COVERED | 0.47 (L12 16k big) | 0.18 | B, 0.55 | True / False | True (0.080 vs 0.014) |

Census record: R2-replicated **True**; e64c-nameable False; named twin False; identity/generic False; near a named/covered unit False; census tier A.
Signature T of logcosh #54: `Make` `Add` `AM` `making` `start` `look` `go` `make` `take` `help` `use` `want` `getting` `see` `get` `going` `said` `being` `You` `then`
Signature T of skew #54: `take` `making` `Add` `Make` `asked` `get` `look` `help` `“` `said` `getting` `being` `make` `go` `You` `then` `had` `use` `has` `where`

**logcosh #54: nearest latents (pooled over the 14 primary dictionaries).** max \|cos\| above the subspace-null p99.9 in 0 of 14 dictionaries (0 without source). Tested latents 788; **99 match** (88 on the heavy side, 11 on the opposite side; 94 outside the source dictionary). Largest tail overlap: L14 16k big #14344, 0.780 (threshold 0.505, null p99 0.080).

| | top \|cos\| latent | cos | match test | top \|r\| latent | r (within type) | match test |
|---|---|---|---|---|---|---|
| 1 | L14 16k small #320 | 0.490 | pos; sig 0.10 (≥0.06, p99 0.08); ovl 0.50 (≥0.50, p99 0.07); **match** | L12 16k big #1940 | 0.219 (0.23) | pos; sig 0.08 (≥0.06, p99 0.07); ovl 0.69 (≥0.50, p99 0.07); **match** |
| 2 | L12 16k big #181 | 0.462 | pos; sig 0.06 (≥0.06, p99 0.07); ovl 0.04 (≥0.50, p99 0.07); no match | L14 16k small #320 | 0.185 (0.20) | pos; sig 0.10 (≥0.06, p99 0.08); ovl 0.50 (≥0.50, p99 0.07); **match** |
| 3 | L12 16k small #145 | 0.458 | pos; sig 0.19 (≥0.06, p99 0.09); ovl 0.01 (≥0.50, p99 0.11); **match** | L13 16k big #11985 | 0.183 (0.19) | pos; sig 0.10 (≥0.06, p99 0.07); ovl 0.66 (≥0.50, p99 0.06); **match** |
| 4 | L13 16k big #228 | 0.453 | pos; sig 0.17 (≥0.06, p99 0.07); ovl 0.03 (≥0.50, p99 0.06); **match** | L14 16k big #320 | 0.178 (0.19) | pos; sig 0.10 (≥0.06, p99 0.06); ovl 0.46 (≥0.50, p99 0.08); **match** |
| 5 | L14 16k big #320 | 0.445 | pos; sig 0.10 (≥0.06, p99 0.06); ovl 0.46 (≥0.50, p99 0.08); **match** | L14 16k big #14344 | 0.163 (0.16) | pos; sig 0.10 (≥0.06, p99 0.06); ovl 0.78 (≥0.50, p99 0.08); **match** |

**skew #54: nearest latents (pooled over the 14 primary dictionaries).** max \|cos\| above the subspace-null p99.9 in 0 of 14 dictionaries (0 without source). Tested latents 732; **89 match** (89 on the heavy side, 0 on the opposite side; 88 outside the source dictionary). Largest tail overlap: L10 16k small #13424, 1.000 (threshold 0.505, null p99 0.058).

| | top \|cos\| latent | cos | match test | top \|r\| latent | r (within type) | match test |
|---|---|---|---|---|---|---|
| 1 | L12 16k big #181 | 0.474 | pos; sig 0.06 (≥0.05, p99 0.09); ovl 0.04 (≥0.51, p99 0.09); no match | L12 16k big #1940 | 0.182 (0.19) | pos; sig 0.09 (≥0.05, p99 0.09); ovl 0.49 (≥0.51, p99 0.09); no match |
| 2 | L14 16k small #320 | 0.459 | pos; sig 0.07 (≥0.05, p99 0.08); ovl 0.35 (≥0.51, p99 0.08); no match | L14 16k small #320 | 0.178 (0.19) | pos; sig 0.07 (≥0.05, p99 0.08); ovl 0.35 (≥0.51, p99 0.08); no match |
| 3 | L12 16k small #1202 | 0.458 | pos; sig 0.04 (≥0.05, p99 0.09); ovl 0.18 (≥0.51, p99 0.10); no match | L12 16k big #181 | 0.176 (0.24) | pos; sig 0.06 (≥0.05, p99 0.09); ovl 0.04 (≥0.51, p99 0.09); no match |
| 4 | L14 16k big #320 | 0.429 | pos; sig 0.07 (≥0.05, p99 0.07); ovl 0.34 (≥0.51, p99 0.07); no match | L14 16k big #320 | 0.173 (0.19) | pos; sig 0.07 (≥0.05, p99 0.07); ovl 0.34 (≥0.51, p99 0.07); no match |
| 5 | L11 16k small #75 | 0.428 | pos; sig 0.11 (≥0.05, p99 0.11); ovl 0.03 (≥0.51, p99 0.08); **match** | rp L13 65k big #1563 | 0.167 (0.19) | pos; sig 0.09 (≥0.05, p99 0.11); ovl 0.30 (≥0.51, p99 0.08); no match |

**OMP atoms to cos 0.8** (band vs that dictionary's per-mode null; `>` = unreached by k_max, banded as k_max + 1):

| dictionary | logcosh #54 signed / nonneg k08 | skew #54 signed / nonneg k08 | null k08 p1 / p5 / p50, signed | nonneg |
|---|---|---|---|---|
| L10 16k big | 39 [>p50] / 48 [>p50] | 39 [>p50] / 42 [>p50] | 5.99 / 8 / 31 | 6.99 / 10.85 / 36 |
| L10 16k small | 33 [>p50] / 43 [>p50] | 37 [>p50] / 38 [>p50] | 6.99 / 7.95 / 29 | 6.99 / 9 / 36.5 |
| L11 16k big | 35 [>p50] / 43 [>p50] | 31 [>p50] / 40 [>p50] | 5 / 8.95 / 28 | 5.99 / 10 / 31.5 |
| L11 16k small | 31 [>p50] / 35 [>p50] | 27 [>p50] / 33 [>p50] | 5 / 8 / 26 | 6 / 9.95 / 32 |
| L12 16k big | 17 [mid] / 19 [mid] | 16 [mid] / 18 [mid] | 4.99 / 6 / 18 | 6 / 7 / 21.5 |
| L12 16k small | 15 [mid] / 18 [mid] | 16 [mid] / 16 [mid] | 4 / 6 / 17 | 5.98 / 8 / 22 |
| L13 16k big | 25 [mid] / 31 [>p50] | 26 [>p50] / 28 [mid] | 7 / 8 / 25 | 7.99 / 10.95 / 30 |
| L13 16k small | 19 [mid] / 28 [mid] | 22 [mid] / 25 [mid] | 5.99 / 7.95 / 23 | 7.99 / 8.95 / 30 |
| L14 16k big | 39 [>p50] / 47 [>p50] | 40 [>p50] / 48 [>p50] | 8.98 / 10.95 / 33.5 | 10.98 / 13 / 39.5 |
| L14 16k small | 33 [>p50] / 40 [mid] | 36 [>p50] / 38 [mid] | 7.99 / 10 / 31 | 8.99 / 12 / 41.5 |
| rp L13 16k medium | 27 [mid] / 33 [mid] | 28 [mid] / 32 [mid] | 6.99 / 10.95 / 28 | 9 / 11 / 36 |
| rp L13 65k medium | 25 [mid] / 29 [mid] | 29 [>p50] / 31 [mid] | 7.98 / 9 / 27 | 9 / 11.9 / 33 |
| rp L13 65k big | 29 [mid] / 37 [mid] | 34 [>p50] / 34 [mid] | 9.99 / 11 / 30 | 11.99 / 14 / 38 |
| rp L13 262k medium | 22 [mid] / 28 [mid] | 25 [mid] / 25 [mid] | 9.29 / 10.45 / 26 | 11.29 / 12 / 31.5 |

**(b) sparsity and (c) pre-check.**

| member | (b) class, with source (S1 / S5 / H; per mode s, nn) | without source | (c) held-out R² at K = 4 / 8 / 16 / 32 / 64 | K\* |
|---|---|---|---|---|
| logcosh #54 | **chance-like** (0 / 0 / 6; s 0/0/6, nn 0/0/6) | chance-like (0 / 0 / 6) | 0.040 / 0.080 / 0.127 / 0.162 / 0.213 | >64 |
| skew #54 | **chance-like** (0 / 0 / 5; s 0/0/9, nn 0/0/5) | chance-like (0 / 0 / 5) | 0.048 / 0.090 / 0.116 / 0.163 / 0.208 | >64 |

**logcosh #54: L12 16k-small latents** (top 3 by \|cos\| and by \|r\|; their 12 strongest D2 contexts; share of those 12 rows in the member's 1% heavy tail):

| latent | cos | r | e68 match | top-12 in tail | strongest contexts (first 3) |
|---|---|---|---|---|---|
| #145 | 0.458 | 0.106 | True | 0/12 | `, like, the first three or four outings, so it's cool[[ to]] get that start` ⏐ `m not the only person who still remembers Second Noah. It was very sad[[ to]] see it go` ⏐ ` receive, you are a consumer of services, and you should not be afraid[[ to]] ask questions about` |
| #1202 | 0.421 | 0.066 | False | 4/12 | `.⏎The get-togethers in Friend Me!⏎gradually help[[ women]] move from getting` ⏐ `⏎Our talented team has a history of providing custom software development services which help[[ clients]] solve their most` ⏐ ` to take away the insecurities and awkwardness of joining a new group and help[[ women]] make connections in` |
| #267 | 0.352 | 0.100 | False | 5/12 | `ibtek develops custom software for forward-thinking businesses, helping them accelerate growth[[ and]] become leaders in` ⏐ ` concepts and principles that IT development has been using for years to help minimize risk[[ and]] work more efficiently` ⏐ ` including: reducing elk-vehicle collisions, reducing crop depredation conflicts with farmers[[,]] increasing the appreciation` |
| #9333 | — | 0.138 | True | 9/12 | ` time you pull out of the driveway? Do you eat? Comb your hair[[?]] Put on makeup` ⏐ ` many homeowners who are ready to install their own HVAC, pour a foundation,[[ or]] build an addition` ⏐ ` its Seattle headquarters where employees can sit by a creek, walk on suspension bridges[[ and]] brainstorm in the` |

**skew #54: L12 16k-small latents** (top 3 by \|cos\| and by \|r\|; their 12 strongest D2 contexts; share of those 12 rows in the member's 1% heavy tail):

| latent | cos | r | e68 match | top-12 in tail | strongest contexts (first 3) |
|---|---|---|---|---|---|
| #1202 | 0.458 | 0.077 | False | 6/12 | `.⏎The get-togethers in Friend Me!⏎gradually help[[ women]] move from getting` ⏐ `⏎Our talented team has a history of providing custom software development services which help[[ clients]] solve their most` ⏐ ` to take away the insecurities and awkwardness of joining a new group and help[[ women]] make connections in` |
| #145 | 0.420 | 0.113 | True | 0/12 | `, like, the first three or four outings, so it's cool[[ to]] get that start` ⏐ `m not the only person who still remembers Second Noah. It was very sad[[ to]] see it go` ⏐ ` receive, you are a consumer of services, and you should not be afraid[[ to]] ask questions about` |
| #323 | 0.392 | 0.111 | True | 6/12 | ` for the gameplay, it was already mentioned that it is very simple. The[[ developers]] of the game` ⏐ ` dog may have been neglected in his early puppyhood.⏎“The adoption[[ agency]] brought him over` ⏐ ` their income eligibility and their potential level of energy weatherization assistance.The intake[[ staff]] member also explained` |
| #9333 | — | 0.125 | False | 7/12 | ` time you pull out of the driveway? Do you eat? Comb your hair[[?]] Put on makeup` ⏐ ` many homeowners who are ready to install their own HVAC, pour a foundation,[[ or]] build an addition` ⏐ ` its Seattle headquarters where employees can sit by a creek, walk on suspension bridges[[ and]] brainstorm in the` |
| #73 | 0.357 | 0.114 | True | 6/12 | ` cents and the timer gets reset to 30 seconds. Once the bidding[[ time]] has expired after` ⏐ ` said Ms. Rinehart could become the richest person in the world if commodity[[ prices]] and Ms.` ⏐ ` Fox’s⏎request to order Dish to stop skipping commercials while the copyright[[ case]] was⏎being` |

<details><summary>logcosh #54: 12 strongest D2 contexts (oriented score)</summary>

- `Hotspot Shield Premium APK is a VPN that assists you[[ with]] concealing your IP`
- ` game app that allows for you to do several things:⏎Build a team[[⏎]]Plan a strategy`
- ` they started living together, it was nothing but the unseen force 'necessity[[']] and 'urg`
- ` permission with:⏎$ chmod +x /usr/bin/vinfo[[⏎]]Then from your`
- ` (IMat), one of Georgia Tech’s Interdisciplinary Research Institutes charged[[ with]] fostering an innovation`
- ` for you to do several things:⏎Build a team⏎Plan a strategy[[⏎]]Lead a team`
- ` the village in the New York Times.⏎Get in your car in Madrid[[,]] head north on`
- ` you touch could have drops of fluids from a person that has been infected.[[⏎]]Wearing a mask`
- ` the country.⏎The closures would limit human access to the caves in hopes[[ of]] stemming the spread`
- ` the new rim -- it's all covered here.⏎So let'[[s]] get to it`
- ` Apr-08 Jul-08 Oct-08 Jan-0[[9]] Apr-0`
- ` silk. There's no right or wrong, do what you like.[[⏎]]Squirt,`

</details>

<details><summary>skew #54: 12 strongest D2 contexts (oriented score)</summary>

- ` (IMat), one of Georgia Tech’s Interdisciplinary Research Institutes charged[[ with]] fostering an innovation`
- `Hotspot Shield Premium APK is a VPN that assists you[[ with]] concealing your IP`
- ` game app that allows for you to do several things:⏎Build a team[[⏎]]Plan a strategy`
- `- Find Help⏎- Our Partners⏎- Contact Us⏎Give. Advocate[[.]] Volunteer. LIVE`
- ` for you to do several things:⏎Build a team⏎Plan a strategy[[⏎]]Lead a team`
- ` they started living together, it was nothing but the unseen force 'necessity[[']] and 'urg`
- `; consult insurance companies about the development and introduction of insurance products and assist in[[ the]] implementation of marketing`
- ` Tim Wells says troopers shut down the bridge until the D.O.T[[.]] could treat the`
- ` silk. There's no right or wrong, do what you like.[[⏎]]Squirt,`
- `60), he convinced the Chinese that Russia was a friendly power and succeeded[[ in]] negotiating the Treaty`
- ` permission with:⏎$ chmod +x /usr/bin/vinfo[[⏎]]Then from your`
- ` the Senate Agriculture Appropriations subcommittee, where she led a successful effort to prevent USDA[[ from]] limiting certain healthy`

</details>

**Reading.** *A verb comes next*: a base-form or gerund verb follows.
- Examples: *assists you **with** concealing*, *charged **with** fostering*, *in hopes **of** stemming*, *succeeded **in** negotiating*, *prevent USDA **from** limiting*. Also the next item of an imperative list: *Build a team⏎Plan a strategy⏎Lead a team*.
- Its signature is verbs: *Make, Add, making, start, look, go, take, help, use*.
- **In 16k small the slot is split across latents by construction:**

  | latent | construction | rows of its 12 strongest in the tail |
  |---|---|---|
  | #145 | adjective + *to*-infinitive (*cool **to** get*, *sad **to** see*); matches by signature only | 0 of 12 |
  | #1202 | *help* + object + bare infinitive (*help **women** move*) | 4–6 of 12 |
  | #9333 | coordinated verb lists (*Comb your hair**?** Put on makeup*) | 7–9 of 12 |
  | #323 / #73 | the head noun of a subject NP before its verb | 6 of 12 |

- **This is the shape e67 found for #13/#16:** a "next-token class" direction spread over latents that split it. Here the class is a verb rather than an adjunct opener.
- **The differences from #13/#16:**
  - it is chance-like on (b);
  - its fixed-set match rates (0.45 / 0.47) are the highest of the three L12 families, with 1 and 0 of 40 nulls at or above them;
  - it is e65's only tier-A leftover. logcosh #54 passed B3 and B4 but is not R2-replicated; skew #54 passed B3 and tied the random maximum on B4.

#### L12 logcosh #13 / skew #16

| member | e68 verdict (with / without source) | best \|cos\| (dictionary) | max \|r\| | e64c best rule, \|AUROC\| | e65 B3 / B4 | signature testable (s_self vs base) |
|---|---|---|---|---|---|---|
| logcosh #13 | COVERED / PARTIALLY COVERED | 0.74 (L12 16k big) | 0.38 | C_inat, 0.56 | True / True | True (0.265 vs 0.028) |
| skew #16 | COVERED / COVERED | 0.72 (L12 16k big) | 0.40 | F_multi, 0.55 | True / True | True (0.285 vs 0.032) |

Census record: R2-replicated **True**; e64c-nameable False; named twin False; identity/generic False; near a named/covered unit False; census tier A.
Signature T of logcosh #13: `before` `times` `when` `couple` `again` `?` `since` `without` `because` `as` `few` `today` `while` `next` `in` `during` `by` `than` `if` `at`
Signature T of skew #16: `before` `times` `couple` `when` `again` `?` `without` `because` `after` `last` `few` `next` `since` `today` `as` `from` `in` `other` `on` `by`

**logcosh #13: nearest latents (pooled over the 14 primary dictionaries).** max \|cos\| above the subspace-null p99.9 in 0 of 14 dictionaries (0 without source). Tested latents 257; **60 match** (57 on the heavy side, 3 on the opposite side; 55 outside the source dictionary). Largest tail overlap: rp L13 65k big #44032, 1.000 (threshold 0.505, null p99 0.090).

| | top \|cos\| latent | cos | match test | top \|r\| latent | r (within type) | match test |
|---|---|---|---|---|---|---|
| 1 | L12 16k big #11562 | -0.735 | pos; sig 0.20 (≥0.15, p99 0.14); ovl 0.68 (≥0.51, p99 0.12); **match** | rp L13 65k medium #877 | 0.384 (0.44) | pos; sig 0.24 (≥0.15, p99 0.22); ovl 0.72 (≥0.51, p99 0.08); **match** |
| 2 | rp L13 16k medium #183 | 0.680 | pos; sig 0.18 (≥0.15, p99 0.13); ovl 0.52 (≥0.51, p99 0.10); **match** | rp L13 16k medium #183 | 0.375 (0.44) | pos; sig 0.18 (≥0.15, p99 0.13); ovl 0.52 (≥0.51, p99 0.10); **match** |
| 3 | L13 16k small #183 | 0.668 | pos; sig 0.19 (≥0.15, p99 0.18); ovl 0.54 (≥0.51, p99 0.11); **match** | rp L13 65k big #1384 | 0.327 (0.37) | pos; sig 0.28 (≥0.15, p99 0.15); ovl 0.41 (≥0.51, p99 0.09); **match** |
| 4 | L14 16k big #14104 | -0.644 | pos; sig 0.22 (≥0.15, p99 0.12); ovl 0.55 (≥0.51, p99 0.11); **match** | rp L13 262k medium #7055 | 0.314 (0.35) | pos; sig 0.32 (≥0.15, p99 0.14); ovl 0.59 (≥0.51, p99 0.10); **match** |
| 5 | L12 16k big #320 | 0.644 | pos; sig 0.19 (≥0.15, p99 0.14); ovl 0.61 (≥0.51, p99 0.12); **match** | rp L13 65k big #1546 | 0.291 (0.34) | pos; sig 0.19 (≥0.15, p99 0.15); ovl 0.43 (≥0.51, p99 0.09); **match** |

**skew #16: nearest latents (pooled over the 14 primary dictionaries).** max \|cos\| above the subspace-null p99.9 in 0 of 14 dictionaries (0 without source). Tested latents 186; **60 match** (60 on the heavy side, 0 on the opposite side; 55 outside the source dictionary). Largest tail overlap: rp L13 65k medium #877, 0.650 (threshold 0.505, null p99 0.070).

| | top \|cos\| latent | cos | match test | top \|r\| latent | r (within type) | match test |
|---|---|---|---|---|---|---|
| 1 | L12 16k big #11562 | -0.722 | pos; sig 0.27 (≥0.16, p99 0.16); ovl 0.64 (≥0.50, p99 0.10); **match** | rp L13 16k medium #183 | 0.395 (0.46) | pos; sig 0.24 (≥0.16, p99 0.17); ovl 0.54 (≥0.50, p99 0.12); **match** |
| 2 | rp L13 16k medium #183 | 0.709 | pos; sig 0.24 (≥0.16, p99 0.17); ovl 0.54 (≥0.50, p99 0.12); **match** | rp L13 65k medium #877 | 0.383 (0.44) | pos; sig 0.29 (≥0.16, p99 0.23); ovl 0.65 (≥0.50, p99 0.07); **match** |
| 3 | L12 16k small #2764 | -0.693 | pos; sig 0.23 (≥0.16, p99 0.18); ovl 0.12 (≥0.50, p99 0.13); **match** | rp L13 65k big #1384 | 0.352 (0.40) | pos; sig 0.31 (≥0.16, p99 0.17); ovl 0.42 (≥0.50, p99 0.09); **match** |
| 4 | L13 16k small #183 | 0.688 | pos; sig 0.24 (≥0.16, p99 0.16); ovl 0.54 (≥0.50, p99 0.10); **match** | rp L13 262k medium #7055 | 0.328 (0.36) | pos; sig 0.39 (≥0.16, p99 0.16); ovl 0.52 (≥0.50, p99 0.11); **match** |
| 5 | L14 16k big #14104 | -0.679 | pos; sig 0.23 (≥0.16, p99 0.15); ovl 0.52 (≥0.50, p99 0.11); **match** | L13 16k big #336 | 0.306 (0.34) | pos; sig 0.28 (≥0.16, p99 0.18); ovl 0.35 (≥0.50, p99 0.10); **match** |

**OMP atoms to cos 0.8** (band vs that dictionary's per-mode null; `>` = unreached by k_max, banded as k_max + 1):

| dictionary | logcosh #13 signed / nonneg k08 | skew #16 signed / nonneg k08 | null k08 p1 / p5 / p50, signed | nonneg |
|---|---|---|---|---|
| L10 16k big | 16 [mid] / 18 [mid] | 12 [mid] / 11 [mid] | 5.99 / 8 / 31 | 6.99 / 10.85 / 36 |
| L10 16k small | 10 [mid] / 11 [mid] | 6 [<=p1] / 5 [<=p1] | 6.99 / 7.95 / 29 | 6.99 / 9 / 36.5 |
| L11 16k big | 20 [mid] / 19 [mid] | 12 [mid] / 13 [mid] | 5 / 8.95 / 28 | 5.99 / 10 / 31.5 |
| L11 16k small | 12 [mid] / 13 [mid] | 7 [<=p5] / 7 [<=p5] | 5 / 8 / 26 | 6 / 9.95 / 32 |
| L12 16k big | 3 [<=p1] / 5 [<=p1] | 2 [<=p1] / 4 [<=p1] | 4.99 / 6 / 18 | 6 / 7 / 21.5 |
| L12 16k small | 5 [<=p5] / 4 [<=p1] | 3 [<=p1] / 3 [<=p1] | 4 / 6 / 17 | 5.98 / 8 / 22 |
| L13 16k big | 6 [<=p1] / 8 [<=p5] | 4 [<=p1] / 5 [<=p1] | 7 / 8 / 25 | 7.99 / 10.95 / 30 |
| L13 16k small | 6 [<=p5] / 6 [<=p1] | 3 [<=p1] / 4 [<=p1] | 5.99 / 7.95 / 23 | 7.99 / 8.95 / 30 |
| L14 16k big | 12 [mid] / 15 [mid] | 5 [<=p1] / 8 [<=p1] | 8.98 / 10.95 / 33.5 | 10.98 / 13 / 39.5 |
| L14 16k small | 11 [mid] / 11 [<=p5] | 5 [<=p1] / 5 [<=p1] | 7.99 / 10 / 31 | 8.99 / 12 / 41.5 |
| rp L13 16k medium | 6 [<=p1] / 6 [<=p1] | 4 [<=p1] / 4 [<=p1] | 6.99 / 10.95 / 28 | 9 / 11 / 36 |
| rp L13 65k medium | 5 [<=p1] / 5 [<=p1] | 4 [<=p1] / 4 [<=p1] | 7.98 / 9 / 27 | 9 / 11.9 / 33 |
| rp L13 65k big | 8 [<=p1] / 10 [<=p1] | 5 [<=p1] / 6 [<=p1] | 9.99 / 11 / 30 | 11.99 / 14 / 38 |
| rp L13 262k medium | 7 [<=p1] / 7 [<=p1] | 5 [<=p1] / 4 [<=p1] | 9.29 / 10.45 / 26 | 11.29 / 12 / 31.5 |

**(b) sparsity and (c) pre-check.**

| member | (b) class, with source (S1 / S5 / H; per mode s, nn) | without source | (c) held-out R² at K = 4 / 8 / 16 / 32 / 64 | K\* |
|---|---|---|---|---|
| logcosh #13 | **sparser** (7 / 9 / 0; s 6/8/0, nn 7/9/0) | sparser (6 / 8 / 0) | 0.140 / 0.171 / 0.230 / 0.228 / 0.267 | >64 |
| skew #16 | **sparser** (11 / 12 / 0; s 11/12/0, nn 11/12/0) | sparser (10 / 11 / 0) | 0.149 / 0.185 / 0.247 / 0.307 / 0.298 | >64 |

**logcosh #13: L12 16k-small latents** (top 3 by \|cos\| and by \|r\|; their 12 strongest D2 contexts; share of those 12 rows in the member's 1% heavy tail):

| latent | cos | r | e68 match | top-12 in tail | strongest contexts (first 3) |
|---|---|---|---|---|---|
| #2764 | -0.607 | 0.054 | True | 1/12 | `-ins to Twitter and Facebook.⏎And you can see how you stack[[ up]] against the world` ⏐ ` some things ironed out."⏎Q: Was the writing was on the[[ wall]] going into the` ⏐ `, it was a moment to reflect.⏎“Gaming has come a long[[ way]],” he said` |
| #136 | 0.599 | 0.173 | False | 6/12 | ` to do bad things to a computer. Some are spyware, trying to steal[[ information]] from the computer` ⏐ ` hard drive. Some others send users to web sites that offer to sell them[[ things]]. Some are` ⏐ `, includes a built-in transfer client to allow users to upload and download[[ files]] from their Web` |
| #211 | 0.577 | 0.114 | True | 3/12 | ` never much, of course, and almost always, it’s completely wiped[[ out]] after a single` ⏐ ` burns over 55 percent of his body. His⏎legs were amput[[ated]] and he lost` ⏐ ` growing determines their appetite. As a result, the appetite can go up and[[ down]].⏎F` |
| #576 | 0.562 | 0.184 | False | 9/12 | ` of you watched it yet?⏎I'm thinking of buying the whole[[ series]] of 4` ⏐ ` product and provide the researcher with a full refund.⏎Researchers may use this[[ product]] for research use` ⏐ ` bags, wrap the board in a sheet of aluminum foil.⏎Ship the[[ product]] in a box` |
| #673 | — | 0.173 | False | 7/12 | ` gets her message out by way of a rich, warm voice. Color her[[ sound]] blue and green` ⏐ ` made all those photos can become a real chore. GeoTagr tags your[[ photos]] with the location` ⏐ ` board. If you don’t have anti-static bags, wrap the[[ board]] in a sheet` |

**skew #16: L12 16k-small latents** (top 3 by \|cos\| and by \|r\|; their 12 strongest D2 contexts; share of those 12 rows in the member's 1% heavy tail):

| latent | cos | r | e68 match | top-12 in tail | strongest contexts (first 3) |
|---|---|---|---|---|---|
| #2764 | -0.693 | 0.065 | True | 2/12 | `-ins to Twitter and Facebook.⏎And you can see how you stack[[ up]] against the world` ⏐ ` some things ironed out."⏎Q: Was the writing was on the[[ wall]] going into the` ⏐ `, it was a moment to reflect.⏎“Gaming has come a long[[ way]],” he said` |
| #136 | 0.645 | 0.183 | True | 6/12 | ` to do bad things to a computer. Some are spyware, trying to steal[[ information]] from the computer` ⏐ ` hard drive. Some others send users to web sites that offer to sell them[[ things]]. Some are` ⏐ `, includes a built-in transfer client to allow users to upload and download[[ files]] from their Web` |
| #211 | 0.623 | 0.131 | True | 3/12 | ` never much, of course, and almost always, it’s completely wiped[[ out]] after a single` ⏐ ` burns over 55 percent of his body. His⏎legs were amput[[ated]] and he lost` ⏐ ` growing determines their appetite. As a result, the appetite can go up and[[ down]].⏎F` |
| #576 | — | 0.177 | True | 6/12 | ` of you watched it yet?⏎I'm thinking of buying the whole[[ series]] of 4` ⏐ ` product and provide the researcher with a full refund.⏎Researchers may use this[[ product]] for research use` ⏐ ` bags, wrap the board in a sheet of aluminum foil.⏎Ship the[[ product]] in a box` |
| #673 | — | 0.170 | False | 7/12 | ` gets her message out by way of a rich, warm voice. Color her[[ sound]] blue and green` ⏐ ` made all those photos can become a real chore. GeoTagr tags your[[ photos]] with the location` ⏐ ` board. If you don’t have anti-static bags, wrap the[[ board]] in a sheet` |

**logcosh #13: e67's stored contexts for its pooled top latents** (cited from `results/e67_acts.json`; not recomputed):

| latent | strongest contexts (first 3) |
|---|---|
| L12 16k big #11562 | ` made all those photos can become a real chore. GeoTagr tags your[[ photos]] with the location` ⏐ ` board. If you don’t have anti-static bags, wrap the[[ board]] in a sheet` ⏐ ` this season, JATE and HUDDY.⏎I used to watch my[[ shows]] like Dawson'` |
| rp L13 16k medium #183 | ` this season, JATE and HUDDY.⏎I used to watch my[[ shows]] like Dawson'` ⏐ ` saves travel expenses, preserves your work day, and allows you to absorb the[[ material]] in manageable segments` ⏐ ` cigarette smokers don’t necessarily like the taste of cigarettes, but they smoke[[ them]] because they are` |
| L13 16k small #183 | ` this season, JATE and HUDDY.⏎I used to watch my[[ shows]] like Dawson'` ⏐ ` it's default:⏎German. But you can start JPhotoT[[agger]] in any supported` ⏐ `, Experienced and Reliable Administrators.⏎*** Great Team Support⏎I joined this[[ program]] when it launched` |
| L14 16k big #14104 | ` lakes of frozen seepage from the melt”⏎(Extract from Philip Dadson[[’]]s diary written` ⏐ ` that allowed just a single, unearned run to score before punching out Murray[[']]s Ramsey Scott` ⏐ ` guess one of the objections I have to Arc is this: the way they[[']]ve implemented the` |
| L12 16k big #320 | ` board. If you don’t have anti-static bags, wrap the[[ board]] in a sheet` ⏐ ` made all those photos can become a real chore. GeoTagr tags your[[ photos]] with the location` ⏐ ` padded bag; this does not provide enough protection.⏎Cushion the[[ product]] inside the box` |
| rp L13 65k medium #877 | ` approach in this article and in a few more to come. I opened my[[ practice]] cold using this` ⏐ ` Something about having a live sax player is just awesome. I made this mash[[up]] as part of` ⏐ `, Experienced and Reliable Administrators.⏎*** Great Team Support⏎I joined this[[ program]] when it launched` |
| rp L13 65k big #1384 | ` This is a great example that we really can do this and we can do[[ it]] for things that` ⏐ ` I need to tune a PDF to the Kindle display. I plan on doing[[ this]] when mine finally` ⏐ ` totally said YES! I also loved the idea that Abigail and I could do[[ this]] together. She` |
| rp L13 262k medium #7055 | ` the club’s leaders, Justin Van Ee, has been with the[[ club]] for three years` ⏐ ` in 2006 — although he worked in the Dolphins' front[[ office]] for several seasons` ⏐ ` to tell you a little bit about my tour. Although I have travelled to[[ Egypt]] several times before` |
| rp L13 65k big #1546 | ` made all those photos can become a real chore. GeoTagr tags your[[ photos]] with the location` ⏐ ` instructions to help assist you with your printables.⏎Can you print my[[ order]]?⏎Please` ⏐ ` padded bag; this does not provide enough protection.⏎Cushion the[[ product]] inside the box` |

**skew #16: e67's stored contexts for its pooled top latents** (cited from `results/e67_acts.json`; not recomputed):

| latent | strongest contexts (first 3) |
|---|---|
| L12 16k big #11562 | ` made all those photos can become a real chore. GeoTagr tags your[[ photos]] with the location` ⏐ ` board. If you don’t have anti-static bags, wrap the[[ board]] in a sheet` ⏐ ` this season, JATE and HUDDY.⏎I used to watch my[[ shows]] like Dawson'` |
| rp L13 16k medium #183 | ` this season, JATE and HUDDY.⏎I used to watch my[[ shows]] like Dawson'` ⏐ ` saves travel expenses, preserves your work day, and allows you to absorb the[[ material]] in manageable segments` ⏐ ` cigarette smokers don’t necessarily like the taste of cigarettes, but they smoke[[ them]] because they are` |
| L12 16k small #2764 | `-ins to Twitter and Facebook.⏎And you can see how you stack[[ up]] against the world` ⏐ ` some things ironed out."⏎Q: Was the writing was on the[[ wall]] going into the` ⏐ `, it was a moment to reflect.⏎“Gaming has come a long[[ way]],” he said` |
| L13 16k small #183 | ` this season, JATE and HUDDY.⏎I used to watch my[[ shows]] like Dawson'` ⏐ ` it's default:⏎German. But you can start JPhotoT[[agger]] in any supported` ⏐ `, Experienced and Reliable Administrators.⏎*** Great Team Support⏎I joined this[[ program]] when it launched` |
| L14 16k big #14104 | ` lakes of frozen seepage from the melt”⏎(Extract from Philip Dadson[[’]]s diary written` ⏐ ` that allowed just a single, unearned run to score before punching out Murray[[']]s Ramsey Scott` ⏐ ` guess one of the objections I have to Arc is this: the way they[[']]ve implemented the` |
| rp L13 65k medium #877 | ` approach in this article and in a few more to come. I opened my[[ practice]] cold using this` ⏐ ` Something about having a live sax player is just awesome. I made this mash[[up]] as part of` ⏐ `, Experienced and Reliable Administrators.⏎*** Great Team Support⏎I joined this[[ program]] when it launched` |
| rp L13 65k big #1384 | ` This is a great example that we really can do this and we can do[[ it]] for things that` ⏐ ` I need to tune a PDF to the Kindle display. I plan on doing[[ this]] when mine finally` ⏐ ` totally said YES! I also loved the idea that Abigail and I could do[[ this]] together. She` |
| rp L13 262k medium #7055 | ` the club’s leaders, Justin Van Ee, has been with the[[ club]] for three years` ⏐ ` in 2006 — although he worked in the Dolphins' front[[ office]] for several seasons` ⏐ ` to tell you a little bit about my tour. Although I have travelled to[[ Egypt]] several times before` |
| L13 16k big #336 | ` supernova is destined for great things. While she has only been on the music[[ scene]] for three years` ⏐ ` speed of upto 5 Mbps⏎Not so Good Things⏎- Have stayed[[ here]] almost 4` ⏐ ` town was only 10 years old. It has been in its current[[ location]] since 1` |

<details><summary>logcosh #13: 12 strongest D2 contexts (oriented score)</summary>

- ` life, all for an even lower price!⏎Can I purchase the XT[[2]]?⏎US`
- ` Canadian seasons is a revitalizing experience for us, which should be taken advantage[[ of]] to refresh our`
- ` to offer 3 laddoos to the Gods, follow all 'procedures[[']] without asking any`
- ` Johnson. The first-time dog owner didn’t realize how sick Tucker[[ was]] she adopted him`
- `: my drawer full of adorable swaddles that I refuse to get rid[[ of]] because they are`
- ` process. Each demonstrated a playfulness and willingness not to take the 'shoot[[']] too seriously.`
- ` of your kid/dog/car probably isn’t generating new ‘Likes[[’]] on Facebook anymore`
- ` think.⏎As to the production, I can make the 'black box[[']] schematic use standard`
- ` kids and dogs… they all plan on being in front of their TV’[[s]] on Sunday.`
- ` Intel's success, i.e. how could Intel have become so[[ successful]] by behaving so`
- ` favourite items out to share, ones I will be pointing people in the direction[[ of]] when they start`
- ` fifteen minutes and the happily ever after.⏎It figured. She clicked the[[ remote]]. The picture`

</details>

<details><summary>skew #16: 12 strongest D2 contexts (oriented score)</summary>

- ` Canadian seasons is a revitalizing experience for us, which should be taken advantage[[ of]] to refresh our`
- ` life, all for an even lower price!⏎Can I purchase the XT[[2]]?⏎US`
- `: my drawer full of adorable swaddles that I refuse to get rid[[ of]] because they are`
- ` Johnson. The first-time dog owner didn’t realize how sick Tucker[[ was]] she adopted him`
- ` to offer 3 laddoos to the Gods, follow all 'procedures[[']] without asking any`
- ` of your kid/dog/car probably isn’t generating new ‘Likes[[’]] on Facebook anymore`
- ` favourite items out to share, ones I will be pointing people in the direction[[ of]] when they start`
- ` is worn like a beeper and tracks what radio stations the user is exposed[[ to]] during the day`
- `’ve added anyone I know with FaceTime (who I’d actually talk[[ to]]) to Favorites`
- ` Intel's success, i.e. how could Intel have become so[[ successful]] by behaving so`
- ` kids and dogs… they all plan on being in front of their TV’[[s]] on Sunday.`
- ` process. Each demonstrated a playfulness and willingness not to take the 'shoot[[']] too seriously.`

</details>

**Reading (reference).** As in e67: the object head or stranded preposition that completes a clause's core, with an adjunct next.
- 16k-small #136 (*steal **information** from*, *absorb the **material** in*) and #576 (*Ship the **product** in a box*) are object-head latents of e67's kind.
- #2764 (cos −0.61 / −0.69; *stack **up***, *come a long **way***) points the opposite way.
- **Of the three measures, #13/#16 stands out only on sparsity.**
  - (b): sparser, like the quote controls.
  - Fixed-set match rate: 0.41 / 0.42, at the null's upper edge rather than beyond it (2 of 40 nulls ≥).
  - So the one confirmed-then-COVERED family is distinguished from random directions by sparsity, not by context sharing.

### Reading

1. **No would-be target separates from the anchors' failure modes on any of the three measures.** Each measure either fails to discriminate or puts the L12 targets in the same class as a known-represented control:
   - (b) puts #45/#5 and #54/#54 with the paren controls (chance-like);
   - (c) cannot be read, because no anchor reaches R² 0.5;
   - context sharing does not separate controls from random same-subspace directions at L12.
2. **What the readings add.** Read with their contexts, each family names a construction, and existing latents carry pieces of it.
   - **L7:** mostly previous-token variables (after *what*, after *at*, after a quantifier, after an affective verb), plus an orthographic-unit continuation.
     - No L7 latent text could be pulled.
     - What separates the L7 targets from random directions is the match *count and rate*: all except #52's exceed every joint null, as the L7 controls' do.
     - A latent with near-1.0 tail overlap is not distinctive by itself: 12 of 40 L7 joint nulls have one.
   - **L12 #45/#5:** blog meta-discourse. 16k-small #1276 has 8 of its 12 strongest rows in the logcosh #45 tail.
   - **L12 #54/#54:** "a verb comes next", split over at least four latents by construction. That is the same shape as #13/#16's "an adjunct comes next".
   - These names are post hoc. No e64c rule names any of these families; the best |AUROC| is 0.54–0.63.
3. **For a "not a blend" criterion** (the stated condition for the deferred 4B step), these three results bound what a criterion could rest on:
   - **Sparsity relative to the null** can certify *blend-like*: it agrees with e68's decoder route for 12 of the 14 COVERED units. But its complement contains known-represented paren units.
   - **A small-K linear readout** from a sibling dictionary does not reach R² 0.5 even for the anchors.
   - **Tail-overlap context sharing** is satisfied by random same-subspace directions at L12.
   - **Per-dictionary counts are not independent.** A random direction's sparsity band is strongly correlated across dictionaries, so any count rule must be calibrated on random directions built like the units, not read as a binomial.
   - **A usable criterion would need a positive control known *not* to be a blend.** This project has none; that is the open methods problem, not a result about these families.
4. **Nothing is amended or nominated.** e68's NOT COVERED calls for the seven families stand as recorded, with e68's own caveat: NOT COVERED under a single-latent rule does not mean "no SAE feature".

## Raw-residual pilot (e72): **gate PASSES on raw h. GO by the frozen rule, with one verified lead (L12 skew #49). It is thin: the same unit as a decoded-span unit, about 76% SAE share, and not off span on the single-filter route. Not recommended for fresh text on its own.**

**What this is.** Every occurrence unit in e63–e71 was found in the SAE-decoded residual, fp16(codes) @ W_dec. So each was a readout of SAE codes by construction, and "uncovered" could not be measured (e68, e71). This pilot runs the frozen v2 instrument on the **raw** L12 residual and only afterwards measures how much of each unit lies in the SAE error. It is not option 2 resumed: e69/e70 searched the error alone.

**Limits.**
- **Data:** burned D2 only (corpus 1,000–1,999): `results/e67_raw_L12_rows.npy` (D2 kept rows, fp16, 203,328 rows, aligned with `occ_units_v2_L12.npz["row"]`; coordinate 1038 dropped). S2 (`results/e69_p1_S2_raw_L12_rows.npy`) for the shuffle comparison.
- No fresh text, no model forward pass, no extraction. CPU only. No verdict (e64–e71) is amended.

**Documents.** Each was hashed with `e64.append_hash` before the numbers it governs.
- **Rubric:** `NOTE_raw_residual_pilot.md` (sha `1aa4dd91…`). It fixes the instrument, the gate (PASS iff *both* paren and quote are recovered), the error-share stat, the 10,000-direction same-subspace baseline, the screens, the lead rule and the verification.
- **Script:** `scripts/e72_raw_residual_pilot.py` (`338c10b2…`).
- **Verification:** `docs/e72_verify.py` (`4860cf38…`), hashed with the NOTE before any real run.
- All three were smoke-tested on synthetic rows only.
  - The smoke test caught one memory bug in the verifier, a 203k × 10,000 projection that the swap watchdog killed at 12.5 GB. It was fixed before hashing.
- **Outputs:**
  - `results/e72_units_raw_L12.npz` (`3a4da05a…`, hashed before any label);
  - `results/e72_gate.json` (`2f761d23…`);
  - `results/e72_coverage.json` (`a6b239b3…`) and `results/e72_coverage_arrays.npz` (`24c14148…`);
  - `results/e72_verify.json` (`e97c11e1…`).
- **Post hoc, descriptive, not in the NOTE:** `docs/e72_posthoc_readout.py` → `results/e72_posthoc.json` (`17a2e12c…` / `24b58f55…`). It reads the hashed JSONs only.

**Run.** Each phase was launched once, detached under `caffeinate` with a 12 GB swap watchdog: gate 109 s, coverage 66 s, verify 101 s. Swap stayed at 2.2–2.5 GB.

### Instrument and gate: the v2 instrument is validated on raw h

- The frozen steps were reused unchanged: demean3, PCA-64 whitened, per-type equalisation (N_MIN 32, n0 16), re-whitening, full-rank FastICA-64 (logcosh and skew), and the e63 screens.
  - `e64.fit_maps` reproduces `e63.type_whiten` exactly (max |Δ| 0.0), and every unit is an exact linear readout of W (R² 1.000000).
  - PCA-64 keeps 38.3% of the identity-removed variance.
- **Screened:** 31 logcosh and 29 skew units (decoded L12: 36 and 42). KIR 7.85.

| label | logcosh (screened max \| unscreened) | skew (screened max \| unscreened) | null p99.9 | recovered | probe |
|---|---|---|---|---|---|
| paren_inside | 0.596 (#43) \| 0.736 (#47, repro < 0.8) | **0.749 (#45)** | 0.516–0.520 | yes, via skew only | 0.778 |
| quote_inside | **0.762 (#29)** | **0.755 (#29)** | 0.513–0.514 | yes | 0.780 |

- **Gate PASS.** The NOTE's both-labels rule and S4v2's any-of rule agree. Decoded L12 recovered both labels on both kinds (0.736 / 0.747).
- Control units: paren skew #45; quote logcosh #29 / skew #29. Each matches its decoded counterpart above the cosine null: logcosh #29 ↔ decoded logcosh #33, |cos| 0.975; skew #29 ↔ skew #33, 0.951; skew #45 ↔ skew #23, 0.877.

### Coverage: error share along each unit vs 10,000 random same-subspace directions

**Method.** The error is e = h − SAE(h), with L12 `resid_post_all` 16k l0_big (e69's `error_rows`). e and SAE(h) are pushed through the same fixed maps fitted on h. maps(h) = maps(e) + maps(SAE(h)) to 1.4e-6.
- **ES** = var(b·W_e) / var(b·W_h), with b a unit's exact readout from W.
- **ES_cov** = cov(b·W_e, b·W_h) / var(b·W_h). ES_cov plus the SAE share equals 1.
- **Random directions:** 10,000 uniform in the 64-d equalised W space (seed 7200).

| quantity | value |
|---|---|
| FVU, raw (kept coordinates) | 0.190 |
| FVU, identity-removed | 0.315 |
| per-PC error share, PC 1 → PC 64 | 0.05 → 0.22 (rising) |
| random ES: p50 / p99.9 / max | 0.309 / 0.350 / 0.359 |
| random ES_cov: p50 / max | 0.166 / 0.210 |
| ceiling (largest ES of any direction in W; generalised eigenvalue) | 0.513 (next 0.445) |
| units above the random max / p99.9 / below p50 | 7 / 11 / 74 of 128 (screened: 4 / – / 45 of 60) |
| expected above the max if units were exchangeable with random | 0.013 |

**Reading the table.**
- The random median (0.309) is the identity-removed FVU (0.315). Random directions carry the average error.
- **The controls sit inside the SAE span**, as the NOTE's fixed reading anticipated:
  - quote logcosh #29: ES 0.274 (p0.4), ES_cov 0.099 (p0.0);
  - quote skew #29: 0.276 (p0.6), ES_cov 0.104 (p0.0);
  - paren skew #45: 0.295 (p13.5), ES_cov 0.160 (p30).

**The 7 units above the random max.** All 5 screens are listed; the "split-half" column bundles split-half repro ≥ 0.8 **and** beating the half-specific random max on both the even and odd sequences (NOTE §6.3).

| unit | ES (ES_cov) | even / odd | screened (repro) | tail TDI | identity | split-half | decoded-span match (\|cos\|, \|r\|) | lead |
|---|---|---|---|---|---|---|---|---|
| logcosh #37 | 0.481 (0.266) | 0.475 / 0.487 | yes (0.98) | 3.83 | **fail** | pass | logcosh #12 (0.929, 0.70) | no |
| skew #52 | 0.478 (0.263) | 0.471 / 0.485 | yes (0.97) | 3.98 | **fail** | pass | logcosh #12 (0.909, 0.70) | no |
| logcosh #49 | 0.398 (0.247) | 0.398 / 0.397 | yes (0.97) | 3.05 | **fail** | pass | logcosh #1 (0.918, 0.72) | no |
| **skew #49** | **0.384 (0.239)** | 0.382 / 0.385 | yes (0.96) | 2.70 | pass | pass | skew #31 (0.911, 0.72) | **LEAD** |
| skew #32 | 0.363 (0.168) | 0.374 / 0.352 | no (0.67) | 1.33 | fail | fail | skew #62 (0.567, 0.57) | no |
| logcosh #5 | 0.362 (0.219) | 0.362 / 0.362 | no (0.43) | 1.33 | fail | fail | logcosh #39 (0.670, 0.58) | no |
| logcosh #63 | 0.362 (0.191) | 0.360 / 0.363 | no (0.45) | 1.47 | fail | fail | skew #49 (0.427, 0.42) | no |

- **Norm and L0 pass for every unit** (|ρ| ≤ 0.17 with the identity-removed log‖h‖, log‖e‖ and L0). **Position fails for only one unit** (127 of 128 pass).
- So the lead rule bites through identity (54 of 128 pass), screening, and the half-specific maximum.
- The controls read split-half "fail" only because they do not beat the half maximum; their repro is 0.92–0.99.

**The near-ceiling pair: logcosh #37 / skew #52** (ES 0.48 against a ceiling of 0.513; ex-kurtosis 44).
- It is the one place in this space where the SAE error concentrates. It beats the random max on the single-filter route too (below).
- It is identity-suspect: tail TDI 3.8–4.0, with tails on ’ / ' / the / ,.
- It is the same unit as decoded logcosh #12 (|cos| 0.93), which e64's battery names next_continuation. Not a lead by the rule.

### The lead: L12 skew #49

- **Error share.** ES 0.384 against a random max of 0.359 (p100), ES_cov 0.239 against 0.210. The SAE share along it is therefore about 76%.
- **Halves.** Even 0.382 against an even max of 0.359; odd 0.385 against 0.361.
- **Screens.** Repro 0.96, η² ≤ 0.3, tail TDI 2.70 (its twin logcosh #49 fails at 3.05), largest tail type share 0.08, |ρ| ≤ 0.11 with norm and L0, |ρ| ≤ 0.02 with position.
- **Unnamed.**
  - e64 battery best |AUROC| 0.59 (after_colon); best e64c rule 0.54.
  - Heavy-tail token types: comma, period, digits, *the*.
  - Contexts are mixed: digit continuations (*12,[[0]]00*, *([[2]]017*), *a[[ Time]]*, *shut[[-]]ins*.
- **It is the same unit as decoded-span e64 skew #31.** |cos| 0.911 is above the cosine null max of 0.875; score |r| 0.72.
  - e64 screened skew #31, and the battery left it unnamed.
  - e64's identity diagnostic recorded tail TDI 3.05 (suspect), and its context reading was 'none'.
  - So the direction was already found in the decoded search. The raw search adds only that its raw-h version carries somewhat more error than a random direction.

**Independent verification** (`docs/e72_verify.py`; fresh numpy SAE error, own demean3, own per-type loop):
- **V1:** every checked unit reproduces its coverage ES to < 1e-9 (W from its own maps: max |Δ| 2.4e-6).
- **V2:** a fresh 10,000-direction baseline (seed 7300) gives max 0.367. skew #49 beats it (p100). **Verified lead: skew #49.**
- **V3 (descriptive), the single-filter route.** This is a least-squares pullback filter f in the 1,151 raw coordinates, with var(e·f)/var(h·f).
  - **skew #49 does not beat the random max on this route:** 0.231 against 0.246 (p99.6).
  - Its LS R² is 0.765, below every random direction's (min 0.784, median 0.85). Its readout leans on the per-type equalisation more than any random direction's does.
  - The near-ceiling pair beats the max on V3 (0.359 / 0.355). skew #32 (unscreened) also beats it on V3 (0.261) though not on the map route. Route dependence cuts both ways.

**S2 (shuffled D2; descriptive).** D2's PCA and maps were applied, and independent ICA fits run on S2.
- On S2 the SAE does worse: FVU 0.43; random ES p50 0.573, max 0.798.
- **10 S2 units beat S2's own random max (5 logcosh, 5 skew), none of them screened.** On D2 the figures are 7 and 4.
- So ICA directions beat uniform random directions generically, even in shuffled text. The informative count is "screened and above the max": 4 vs 0, from one shuffle.
- skew #49's filter sits at S2's random median (p51). It is not present in S2 (best |cos| 0.49 against S2's units).

**Post hoc readout** (`results/e72_posthoc.json`):
- **ES does not track non-Gaussianity in general.** Spearman ES vs |ex-kurtosis| is −0.29 over all 128 units and 0.01 over the 60 screened. The kurtosis-44 pair is the exception, not the rule.
- **The raw search mostly re-finds the decoded-span units.**
  - **60 of 128** raw units match an e64 decoded unit above the cosine null max (0.875); most sit at 0.95–0.99.
  - The NOTE's 0.8 bar ("same unit": 72 of 128) is **below this null's p99.9 (0.824)**, so matches between 0.8 and 0.875 are not readable. Only matches above the null max are.

### GO / STOP

**By the frozen rule: GO.** There is one lead, L12 skew #49, verified by V1 and V2. GO licenses writing a fresh-text prereg only; it never supports a discovery claim.

**Recommendation: a raw-residual round is not yet worth fresh text on this lead alone.**
- The question was whether any raw-h unit lies measurably off the SAE span. For skew #49 the answer is "barely, and only through the instrument's per-type filter":
  - its error share exceeds the random maximum by 0.025;
  - about 76% of it is still SAE span;
  - on a single residual-space filter it does not beat the random maximum;
  - its filter is unremarkable on shuffled text;
  - it is the same direction e64 had already found in the decoded span (skew #31, there identity-suspect).
- **What the pilot does establish:**
  - the v2 instrument is validated on raw h;
  - known variables (paren, quote) sit inside the SAE span (quote at p0 on ES_cov);
  - raw-h ICA largely re-finds the decoded-span units (60 of 128 above the cosine null max);
  - the one error-concentrated direction (logcosh #37 / skew #52, ES 0.48 against a ceiling of 0.513) is identity-suspect.
- **What would change the recommendation:**
  - a unit beating the random maximum on the single-filter route as well as the map route;
  - a selection-matched null (ICA refit on shuffled text) yielding no screened above-max units on more than one shuffle;
  - both holding on a fresh capture.
- **Whether to write that prereg is the user's call.**

## Selection-matched shuffle null for the e72 lead (e72b): **INDETERMINATE by the frozen rule (s = 2, 2, 1, 1, 0 against D2's 4). Descriptively, the full e72 lead rule fires once on 4 of 5 shuffles, as it did once on D2. The e72 lead does not survive as evidence on its own.**

**What this is.** e72's lead (L12 skew #49) beat 10,000 random same-subspace directions on SAE error share. But ICA selects non-Gaussian directions, so random directions are not a selection-matched null. This round re-runs e72's S2 branch on 5 new within-sequence shuffles of D2. The question: does ICA on context-destroyed text also produce SCREENED units whose error share beats that shuffle's own random max?

**Limits.**
- **Data:** burned D2 only (corpus 1,000–1,999), re-ordered within each sequence. No fresh corpus text.
- **Forward pass:** one MPS pass of burned text, L12 residual only, approved by the user and run after the e72f steer run had exited.
- **Verdicts:** none amended (e64–e72 stand). e72's GO is unchanged; this sharpens its reading.

**Documents.** Each was hashed with `e64.append_hash` before the numbers it governs.
- **Rubric:** `NOTE_shuffle_null.md` (sha `1e6a5903…`). It fixes the shuffles, the mirrored e72 S2 branch, the primary count, the secondary full-lead-rule count (descriptive), the S2 reproduction check and the three-way reading.
- **Script:** `scripts/e72b_shuffle_null.py` (`55c4c3fa…`). It asserts the e72 script is the frozen `338c10b2…` and calls its functions.
- **Smoke tests before hashing:**
  - a no-model capture (metas built, synthetic rows);
  - analyze on real S2 plus synthetic shuffles. S2 reproduced e72's S2 block exactly. The new S2 descriptives it computed were seen before hashing and play no part in the decision (NOTE, "What was seen").
- **Outputs:**
  - `results/e72b_capture.json` (`d818c995…`);
  - per shuffle k = 1…5, `cache/occ/occS2b{k}_meta.npz` and `results/e72b_S2b{k}_raw_L12_rows.npy` (fp16, about 467 MB each);
  - `results/e72b_null.json` (`c67f234c…`) and `results/e72b_null_arrays.npz` (`ae0dd186…`).

**Run.** Each phase was launched once, detached under `caffeinate` with a 12 GB swap watchdog.
- capture: 472 s, peak swap 0.31 GB;
- analyze: 221 s, peak swap 0.30 GB.

### Method

- **Shuffles.** Sequence i is permuted with seed 700,000 + 10,000·k + 1,000 + i, so k = 1…5 uses seeds 711,000–751,999 (S2 used 601,000–601,999).
  - Asserted: every sequence is a permutation of D2's, and no sequence equals D2, S2 or another shuffle.
  - Kept rows follow e63's rule on each shuffle's own meta: 202,815–202,870 rows (D2 203,328).
- **Capture.** e69 p1's `residual()`, copied verbatim. The drift check re-captured the first 32 D2 sequences (6,555 rows) and matched `e67_raw_L12_rows.npy` bit for bit (max |Δ| 0.0).
- **Analysis: e72's S2 branch, line for line.**
  - D2's raw PCA and maps from `e72_units_raw_L12.npz` are applied, then independent FastICA-64 fits (logcosh, skew).
  - The SAE error (16k l0_big) goes through the same maps.
  - ES is compared with each shuffle's own 10,000 random W-space directions (seeds 7211–7215).
  - Screened = `e63.ica_pass` (η² ≤ 0.3 for y and y², split-half repro ≥ 0.8).
- **S2 as shuffle 0 reproduces e72 exactly.** Counts 5 / 5 above max, 0 / 0 screened; random max 0.798; skew #49 at p50.88; unit-ES quantiles to 0.0.
- **Primary statistic.** s_k = screened units above shuffle k's random max. D2's reference is **4**: logcosh #37, #49; skew #49, #52.
- **Secondary (descriptive, declared in the NOTE).** ℓ_k = units passing e72's *full* lead rule, recomputed on the shuffle: above the max on all rows and on both parity halves; screened and not identity-suspect; norm/L0 and position screens. D2's ℓ = 1 (skew #49).

### Results

| set | FVU | random ES p50 / max | above max (all) | screened | **s** (screened above max) | **ℓ** (full lead rule) | skew #49 filter: pct (best \|cos\|) |
|---|---|---|---|---|---|---|---|
| D2 (e72) | 0.190 | 0.309 / 0.359 | 7 | 60 | **4** | **1** | p100 (itself) |
| S2 (e72's shuffle) | 0.430 | 0.573 / 0.798 | 10 | 35 | 0 | 0 | p50.9 (0.49) |
| S2b1 | 0.431 | 0.572 / 0.798 | 12 | 42 | **2** | **1** | p53.0 (0.44) |
| S2b2 | 0.429 | 0.568 / 0.772 | 16 | 42 | **2** | **1** | p49.1 (0.52) |
| S2b3 | 0.431 | 0.572 / 0.771 | 24 | 33 | **1** | **1** | p58.0 (0.60) |
| S2b4 | 0.431 | 0.573 / 0.786 | 12 | 38 | **1** | **1** | p49.9 (0.46) |
| S2b5 | 0.429 | 0.571 / 0.800 | 6 | 35 | **0** | **0** | p50.3 (0.53) |

- **Decision (5 new shuffles): s = [2, 2, 1, 1, 0], median 1, max 2 → INDETERMINATE.** The null produces screened above-max units, but fewer than D2.
  - Pooled: 6 of 190 screened units (3.2%) against D2's 4 of 60 (6.7%).
  - With K = 5 this is a count comparison, not a test.
- **All 6 screened above-max shuffle units pass the identity screen** (tail TDI ≤ 2.6). D2's 4 do not: only skew #49 does.
  - Counting screened, not-identity-suspect units above the max (a descriptive recount from the declared screens) gives
    D2 **1** against shuffles **2, 2, 1, 1, 0**.
- **The full lead rule fires once on 4 of 5 shuffles:** S2b1 logcosh #59, S2b2 skew #9, S2b3 logcosh #54, S2b4 skew #26.
  - Each is screened (repro 0.84–0.98), not identity-suspect (tail TDI 1.5–2.6, largest type share ≤ 0.10), passes norm/L0 and position, and beats both half maxima.
  - Their heavy tails sit mostly on *the*, newline, comma and *(*. skew #49's sit on comma, period, digits and *the*.
- **skew #49's own filter is unremarkable on every shuffle** (p49–p58), and it is not present in any (best |cos| 0.44–0.60 < 0.8).
- **The near-ceiling pair logcosh #37 / skew #52 is present in every shuffle** (best |cos| 0.82–0.91 ≥ 0.8, context-generic by e64's rule). Its filters sit at p30–p44 there. logcosh #49 is not present (0.43–0.57).
- **Post hoc (not in the NOTE): margins.** Beating the max is measured in units of each set's random spread, (ES − max)/(max − p50).
  - The shuffle leads clear their maxima by 0.01–0.21. skew #49 clears D2's by 0.48, and D2's identity-suspect units by 0.77–2.4.
  - The SAE regimes differ (FVU 0.43 vs 0.19), so this comparison is not calibrated.
- **Single-filter route (cited from `e72_verify.json`, V3; random max 0.246).**
  - skew #49: 0.231, does not beat the max; LS R² 0.765 < random min 0.784.
  - logcosh #37: 0.359; skew #52: 0.355. Both beat it.

### Reading

The frozen rule says **INDETERMINATE**. Shuffled text yields screened above-max units on 4 of 5 shuffles (1–2 each), fewer than D2's 4.
- **But D2's excess sits entirely in identity-suspect units.** logcosh #37, #49 and skew #52 fail e64's tail-TDI screen, and the #37 / #52 pair is present in every shuffle.
- **Apply the selection that actually produced skew #49**, the full lead rule, and D2's single lead is what the instrument yields on context-destroyed text most of the time: 1 on 4 of 5 shuffles, against 1 on D2.
- So "one verified lead" carries no evidence on its own. The e72 lead is consistent with a selection effect of ICA plus screening, even though the frozen primary count does not call it one.
- **Two things still favour skew #49, both weak:**
  - it is absent from every shuffle (it needs context);
  - it beats its max by a wider relative margin than any shuffle lead (post hoc, regime-confounded).
- **With e72's own caveats** (76% SAE share, no off-span signal on the single-filter route, the same unit as decoded skew #31), **the recommendation stands and is firmer: do not spend fresh text on skew #49.**
- **What would change this:** a raw-h unit whose off-span margin exceeds the shuffle leads' distribution on a pre-registered, regime-matched scale and that also beats the single-filter max. Such a rule would have to be written before any fresh capture.
- **Whether to pursue that is the user's call.**

## Fragmentation test of the e66 direction (e72f): **INCONCLUSIVE.** The anchor, the e66 direction (fourth block), T1 and the planted control pass; the bank-median precondition fails, so the best SAE row cannot be classed

**What this is.** e67 found the confirmed L12 direction d (logcosh #13) COVERED, reading it as a blend of subordinator and preposition latents. This test asks the behavioural version: steered alone, does d's best 16k-big decoder row move only one of the two word classes (subordinators S, prepositions P), while d moves both? It was written up here on 2026-10-17, after the run of 2026-10-16.

**Limits.**
- **Data:** fresh block F = corpus 8,000–8,999 (200 sites). Its tokens were first read by the freeze phase, after the prereg was hashed.
- **Verdicts:** none amended. e66 CONFIRMED and e67 COVERED stand.

**Documents.** Each was hashed with `e64.append_hash` before the numbers it governs.
- **Prereg:** `PREREG_fragmentation.md` rev 3 (sha `ca16541a…`), frozen with the user's approval after three rounds of adversarial review.
- **Script:** `scripts/e72f_fragmentation.py` (`825ac723…`); smoke test `docs/e72f_smoke_synthetic.py` (`7c7a297a…`).
- **Freeze outputs, before any forward pass:** `results/e72f_dirs.npz` (`71aaafc1…`) and `results/e72f_frozen.json` (`82d3ea75…`).
- **Run outputs:** `results/e72f_steer.json` (`df290b6d…`) and `results/e72f_verdict.json` (`ce849745…`).

**Run.** Each phase was launched once.
- freeze: 81 s of CPU. The 262k arms were skipped by the pre-set rule: only 64 share-matched 262k rows exist, and the bank needs 99.
- steer: 3,672 s of MPS, detached under `caffeinate` with the 12 GB swap watchdog. Peak machine swap 7.36 GB.
- verdict: CPU, run at 22:06. It waited about 3 h after the steer exited at 19:04, because the background waiter never fired.

### Results (frozen rules, §8–§9 of the prereg)

| item | result |
|---|---|
| quote anchor (e65 rule) | **PASS:** Δ 0.856, RAND max 0.461, SIG max 0.541 (n 76), sign consistency 0.95 |
| d itself (e65 rule) | **PASS**, its fourth behavioural block: Δ 1.78, RAND max 1.14, SIG max 0.81, sign consistency 1.0 |
| T1, premise (d moves both classes beyond the RAND max, both readouts) | **PASS:** S 2.79 vs 2.21 and P 1.66 vs 1.21 (corrected); S 2.69 vs 2.60 and P 1.84 vs 1.69 (raw) |
| planted positive control | **PC PASS:** both planted arms pass at f = 1.0 and f = 0.5; at f = 0.25 the mean-difference arm fails and the unembedding arm passes |
| L_big (the gated row, #11562 masked and sign-flipped) | **INCONCLUSIVE:** precondition failed. Bank median S 1.29 vs RAND max 2.21 (corrected); 1.27 vs 2.60 (raw). Covariates inside the bank, not VOID |
| x320 (agreement check) | INCONCLUSIVE (precondition); agrees with L_big |
| R_big (5-row reconstruction, T3) | INCONCLUSIVE: TWO-SIDED on the corrected readout, INCONCLUSIVE on the raw one, and the readouts must agree |
| near-d bank (qualifier) | INCONCLUSIVE |
| 262k small | not tested (skipped at freeze) |
| **headline** | **INCONCLUSIVE** |

### Reading

- **What it rules out:** nothing about fragmentation by the frozen rules. The test could not class the row on block F.
- **What did hold:** the rig worked on a fourth block (anchor PASS), d again moved its own signature beyond every random direction, and d moved both word classes, so the question was well posed.
- **Why the precondition failed (post hoc reading, not part of the frozen verdict).**
  - The RAND S distribution is heavy-tailed: median −0.02, p95 0.94, max 2.21, with 2 of 99 draws above 2.2. Random natural-scale pushes move a 9-word class's log-probability with a heavy tail.
  - The precondition compared a bank *median* with a RAND *maximum*, so it failed on a tail property of RAND rather than on anything about block F.
  - Power for that comparison was not estimated on burned data with real forward passes before the freeze (the freeze dry run used a fake forward).
  - Even a p95 gate would still fail the raw readout narrowly (bank median 1.27 against RAND p95 1.33), so no re-scoring was done.
- **No tilt seen (post hoc reading, not a verdict).**
  - L_big's tilt statistic τ lies inside its bank's band under both readouts: 0.252 in [−0.292, 0.436] corrected, and 0.188 in [−0.239, 0.297] raw.
  - The same band caught both half-strength planted tilts.
  - So no one-sided behaviour was seen, and e67's "blend" reading is not supported in behaviour. The frozen rule still cannot call the row NOT FRAGMENTED, because its precondition failed.
- **Descriptive extras (no null):**
  - #11562 steered in its native sign (+row) demotes both classes (S −0.95, P −0.49).
  - The quote anchor's best single latent, 16k-small #5117, moves the anchor's signature the wrong way (ratio −0.28 of the anchor's Δ).
  - Of R_big's five atoms, only #33 leans (toward S). R_big's NNLS reconstruction of d has cos 0.807.
  - The duplicate screen dropped 5 of block F's 1,000 sequences; the 200 sites were drawn from the 995 kept.
- **Block spent:** corpus 8,000–8,999.
- **Methods lesson** (also in `PROCESS.md` §4.2 and §7.3):
  - A per-class "beyond the max of 99 random directions" potency gate fails for rare token classes. Future preregs should fix rank- or z-based gates, and estimate power on burned data with real forwards. Never compare a median with a max-of-99.
  - Chain the verdict phase into the launched job.
- **Reusable assets:**
  - the coordinate-1038-share-matched, d-oriented decoder-row bank. Plain random-row banks are not exchangeable with a near-d row, because the 1038 share rises with |cos| to d;
  - graded planted tilt controls;
  - the conjunction of raw and frequency-corrected readouts. Frequency-matched references for *of* and *in* can only be function words.
- **Gotcha:** #11562 and #320 are an antipodal co-firing pair (masked cos −0.93), and 55–61% of their squared norm sits on coordinate 1038.
- **The fragmentation line is CLOSED at 1B by the user's decision** (2026-10-17).


## Coverage-test admission at L12 by a knockout positive control (e73, R5 admission): **NOT ADMITTED.** No test is admitted for L12 16k l0_big or l0_small, and the geometry-only dummy T-COS is NOT ADMITTED too, so the rule set held. By the frozen stop rule, R5 and R6 close at L12 for the 16k family

**What this is.** PROCESS.md §3 and §9 steps 1–2. It is the first positive control for "uncovered by the dictionary". Knocking out the dictionary's own paren and quote latents makes those variables uncovered by construction. A coverage test is admitted only if it does all of these:
- reads the knockout as ADDS;
- stays quiet under matched random knockouts and on random directions;
- detects a planted Markov variable.

Three tests were calibrated:
- **T-ES:** e72's error share.
- **T-MG:** marginal gain of a one-latent JumpReLU in W space.
- **T-COS:** max |cos| to the removed decoder rows. It is a dummy that must fail.

**Limits.**
- **Data:** burned D2 (corpus 1,000–1,999), raw L12 rows only. CPU only: no MPS, no download, no fresh text. No block changes state.
- **Verdicts:** none amended. e64–e72f stand.

**Documents.** Each was hashed with `e64.append_hash` before the numbers it governs.
- **Rubric:** `NOTE_coverage_admission_L12.md` (`2f74f277…`).
- **Script:** `scripts/e73_coverage_admission.py` (`236e5c13…`).
- **Smoke test:** `docs/e73_smoke_synthetic.py` (`2823bafe…`). Assertions A–G pass at real shape on these bytes. On synthetic data T-ES and T-MG reach ADMITTED-GRADED and T-COS is NOT ADMITTED by rule 7.
- **Review:** one adversarial review before hashing (NOTE §14). Changes adopted, all on synthetic data only:
  - a sequence-swap probe band;
  - class-order monotonicity;
  - rule 5 counting only moves into or out of ADDS;
  - within-sequence near-duplicate correlations;
  - size-matched random ablations (rule 4b);
  - sub-threshold-only leak features.
- **Outputs:**
  - `results/e73_calib_l0_big.json` (`cf0e570f…`) and `.npz` (`d576fde4…`);
  - `results/e73_calib_l0_small.json` (`cf481603…`) and `.npz` (`0823a5af…`);
  - `results/e73_coverage_admission.json` (`91c5a880…`).

**Housekeeping in the same cycle** (PROCESS §8 item 0; all hashed):
- `design_panel_result.json`'s sha (`63da457b…`) appended to HASHES, so the frozen 20-latent knockout list passes `require_hashed`.
- `tools/launch.py` and `tools/watchdog.py`, generalised from the e72f scratchpad. Phases are arguments; each phase records its exit code; a status file gets a 60 s heartbeat; the verdict is chained into the same job. Smoke test: `docs/launch_smoke.py`. The launcher was re-hashed after one fix: its `ps` scan had refused on its own launching shell.
- `docs/SEEDS.jsonl` and `docs/BLOCKS.jsonl` back-filled from PROCESS §5. Corpus 8,000–8,999 is recorded as SPENT and added to the duplicate-screen used sets.

**Run.** One launch, 11:17:14–11:39:17, detached under `caffeinate` with the 12 GB swap watchdog.
- **Phases:** calibrate_l0_big 658 s, calibrate_l0_small 625 s, verdict 1 s. All exit 0.
- **Memory:** peak RSS 5.8 GB. Machine swap was 3.8 GB at launch and peaked at 6.4 GB.
- **Integrity checks, all passed:**
  - input shas;
  - the knockout identity on 4,096 rows (max |Δ| 6e-4 for l0_big, 1e-4 for l0_small);
  - fast path against full path (2.5e-6, 4.2e-7);
  - the k = 0 T-ES of the three targets and the bank quantiles reproduce `e72_coverage.json` to 1e-9.

### Results (frozen rules, NOTE §10)

| dictionary | test | level | first failure (paren first, then quote, then rule 7) | MDE κ, W-span plant (oracle target) |
|---|---|---|---|---|
| l0_big | T-ES | **NOT ADMITTED** | paren, rule 2: never ADDS (UNRESOLVED from s10 on) | 2 |
| l0_big | T-MG | **NOT ADMITTED** | paren, rule 1: UNRESOLVED at k = 0 | 1 |
| l0_big | T-COS | **NOT ADMITTED** | paren, rule 2: ADDS at s1–s3, then back to UNRESOLVED | undefined |
| l0_small | T-ES | **NOT ADMITTED** | paren, rule 1: UNRESOLVED at k = 0 (0.507 against median 0.506) | 2 |
| l0_small | T-MG | **NOT ADMITTED** | paren, rule 1: UNRESOLVED at k = 0 | 1 |
| l0_small | T-COS | **NOT ADMITTED** | paren, rule 2: not monotone | undefined |

**T-ES of the controls along the ladder.** Each cell gives the value, its class, and the same-condition bank's (10,000 W-space directions) median / max.

| step | l0_big paren skew #45 | l0_big quote logcosh #29 | l0_small paren skew #45 | l0_small quote logcosh #29 |
|---|---|---|---|---|
| s0 (full X) | 0.295 C (0.309 / 0.359) | 0.274 C (0.309 / 0.359) | 0.507 U (0.506 / 0.584) | 0.463 C (0.506 / 0.584) |
| s10 | 0.336 U (0.326 / 0.675) | 0.374 U (0.318 / 0.471) | 0.535 U (0.507 / 0.584) | 0.518 U (0.508 / 0.588) |
| sND | 0.392 U (0.377 / 0.987) | 0.794 U (0.370 / 0.810) | 0.659 U (0.541 / 0.739) | 0.741 **A** (0.528 / 0.612) |
| s256 | 0.410 U (0.380 / 1.022) | 0.794 U (0.370 / 0.811) | 0.733 U (0.575 / 1.020) | 0.866 **A** (0.548 / 0.859) |

**Removed-set sizes.**
- **s10 sets:** l0_big uses the design panel's frozen 10 + 10. l0_small uses rule E's top 10.
- **sND sizes:** 1,015 and 736 latents (l0_big paren, quote); 315 and 253 (l0_small).
- **s256 sizes:** 1,026 and 787; 448 and 396.

**Other rules.** Where an earlier rule already failed, these are descriptive.
- **Rule 3 (cross-specific):** passes for every test and control.
- **Rule 4:** no target reads ADDS under any of the 40 matched 10-latent ablations or the 120 size-matched ablations (up to 1,026 latents), under any test.
- **Rule 5:** at most 1 of the 40 S40 directions moves into or out of ADDS under any ablated condition. 11–26 of 40 move between COVERED and UNRESOLVED, which the draft rule would have counted.
- **Rule 6:** GRADED PARTIAL COVER for every test and control. The leak probe (reconstruction image plus sub-threshold drive) is above its band at every step (AUROC 0.70–0.78), so no ADDS step could have been CLEAN; for controls with no ADDS step the grade is by default.
- **Rule 8: NOT REACHED anywhere.** The band top (sequence-swap null) is 0.51–0.52. The W-space reconstruction probe falls:
  - l0_big: paren 0.756 → 0.663, quote 0.756 → 0.641;
  - l0_small: paren 0.735 → 0.594, quote 0.729 → 0.613.

  The raw-h ridge probe is 0.777 (paren) and 0.775 (quote).
- **T-MG:** passes rules 1–5 for quote at l0_big (ADDS from s10). paren's gain at k = 0 already sits between the 200-direction bank's median and max at both dictionaries.
- **Plants:**
  - W-span plant: T-ES reads ADDS from κ = 2 (l0_big 0.385 against a max of 0.361) and T-MG from κ = 1.
  - Full-rank plant (descriptive): T-ES never reads ADDS at κ ≤ 4. T-MG reads ADDS only at κ = 4 (l0_big) and κ = 2 (l0_small).
- **Ladder shape:**
  - The near-duplicate rule admitted 996 (paren) and 726 (quote) latents by within-sequence activation correlation at l0_big, but only 11 and 6 by decoder cosine. At l0_small: 298 and 243 by activation, 11 and 5 by cosine.
  - So the ladder jumps from 10 latents to about 1,000 at l0_big, where the knockout term carries 36% of the W-space error variance, and s30/s100 add almost nothing.
  - Rule E's top 10 shares 3 (paren) and 0 (quote) latents with the frozen design-panel lists. The frozen lists were used for l0_big, as the NOTE requires.
- **l0_small's regime:** L0 18.8, FVU 0.278 raw and 0.448 identity-removed. l0_big: L0 110, FVU 0.190 and 0.315.

### Reading

- **Frozen verdict: NOT ADMITTED for every test at both dictionaries. The rule set held: T-COS is NOT ADMITTED and the guard did not fire.**
  - By the NOTE's stop rule, R5 and R6 close at L12 for the 16k family, and this section is the methods card.
  - The 4B scale-up stays deferred. No retrained SAE is substituted.
  - PROCESS §9 step 3 (the L11/L13 capture) is not triggered.
- **Why (post hoc reading, not part of the verdict).** Two failure modes.
  1. **paren sits on the COVERED/UNRESOLVED boundary at k = 0.** T-ES at l0_small reads 0.507 against a median of 0.506, and T-MG sits there at both dictionaries. So rule 1 is knife-edge for paren.
  2. **Under knockout, the bar moves faster than the target.** At l0_big the bank's max climbs from 0.36 to about 1.0, while its median moves only from 0.31 to 0.38.
     - That is consistent with a knockout term that concentrates its variance in a few W directions.
     - paren's ES rises (0.295 → 0.410). quote's rises much more (0.274 → 0.794, just under the max of 0.810).
     - Neither clears a max-of-10,000 bar that the knockout itself inflates.
- **The statistics do respond to the constructed loss.**
  - Every control's T-ES rises along the ladder.
  - No random or size-matched knockout reads ADDS.
  - The W-span plant is detected at κ = 1–2.

  What fails is the admission bar applied to this ladder, not the direction of the statistic.
- **T-COS behaved as designed.** Geometry alone reads ADDS at s1, where the reconstruction probe has barely moved (a drop of 0.013 against a null spread of 0.022). The class-order rule then catches the drop back to UNRESOLVED.
- **What it rules out:**
  - T-ES and T-MG as admitted coverage tests at L12 16k under PROCESS §3.2's bars;
  - therefore any R5 ADDS-TO-X claim at L12 for the 16k family, graded or not;
  - therefore DISCOVERY item 3 at 1B L12, for now.
- **What it does not show:** that no coverage test can be admitted.
  - A different class bar (a quantile below the max, or bars fixed at k = 0) or a gentler near-duplicate rule might behave differently.
  - Any such variant would be chosen after seeing these numbers. It would need a new NOTE on burned data and would be a pilot, not a confirmation.
- **Blocks spent:** none. D2 is BURNED before and after.
- **Cost, estimated against actual (NOTE §13):**
  - wall clock: estimated 22–30 min CPU; actual 22 min (1,283 s of calibration and 1 s of verdict);
  - memory: at most 7 GB per phase estimated; actual peak RSS 5.8 GB, machine swap peak 6.4 GB;
  - disk: about 28 MB written.
- **Next decision (the user's).** Either:
  - accept the closure, and decide whether to add a SUMMARY.md pointer and an INSTRUMENTS.md card (T-ES and T-MG NOT ADMITTED at L12 16k); or
  - commission a new burned-data NOTE for a redesigned bar, labelled post hoc.

  Steps 3+ need a go-ahead either way. Still open from PROCESS §9 step 0 and outside this cycle's brief: the DISCOVERY.md draft and the skew #49 retirement decision.
- **User decision (2026-10-17): closure accepted.** R5 and R6 stay closed at L12 for the 16k family; no redesigned bar is commissioned. The SUMMARY.md pointer and the top addendum are left to the parent session.

### Post hoc: alternative class bars (descriptive, user-requested, 2026-10-17; amends nothing)

**What this is.** A re-read of the hashed e73 outputs, to decide whether a frozen bar-sensitivity run on an unused block is worth spending one. e73's own `judge()` runs unchanged; only the class bars differ.
- **Grid**, fixed in the script before any variant was computed: 16 bar combinations.
  - ADDS above the bank's max (e73), p99.9, p99 or p95;
  - bars rescored per condition (e73), or fixed at s0 for every ablated condition (plants always use their own bank);
  - COVERED at or below the median (e73) or the p75.
- **Files:** `docs/e73_posthoc_bars.py` (`3d6deb76…`) → `results/e73_posthoc_bars.json` (`63fd3288…`).
- **Check:** the e73 combination reproduces the frozen levels and first failures exactly.
- **Correction:** the HASHES note for the output says "1 of 32 cells"; the grid has 64 T-ES/T-MG cells.

**Result.**
- **1 of the 64 T-ES/T-MG cells admits anything:** T-MG at l0_big under (p99, per condition, p75), ADMITTED-GRADED. It is isolated.
  - At p99.9, paren drops from ADDS back to UNRESOLVED at s10 (rule 2).
  - At p95, 5 of the 40 random directions move into ADDS (rule 5).
- **T-COS stays NOT ADMITTED everywhere,** and the guard never fires.
- **The two bar families fail for opposite reasons:**
  - **Bars that move with the condition:** the targets never clear the knockout-inflated tails. Under T-ES at l0_big, even the loosest of them (p95 ADDS, p75 COVERED) leaves paren COVERED at every step, because random directions' error share rises more than paren's.
  - **Bars fixed at s0:** specificity collapses. Under T-ES at l0_big, 6 of the 20 size-matched random knockouts read paren as ADDS, and up to 37 of the 40 random directions move into ADDS. Large knockouts make almost anything read uncovered.

**Reading (post hoc).**
- The obstacle is the ladder, not the choice of bar. From sND on, the knockout term inflates error variance across W space generally, and not specifically along the controls' directions. Quote under T-MG is the partial exception.
- A single graded cell out of 64, between two failing neighbours, does not justify spending an unused block on a frozen confirmation. **None is recommended.**
- A redesign would have to change the near-duplicate rule, which here admitted about 1,000 latents. That would be a new post hoc NOTE.
