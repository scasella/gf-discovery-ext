# gf-discovery-ext — summary

*Distilled from `REPORT.md`, as of 2026-10-16. Model: `gemma-3-1b-pt`; reference dictionaries: Gemma Scope 2.
All runs were local (M4 Pro, 24 GB, MPS). Every number below is copied from REPORT.md or a hashed result file. The
bracketed tag after it names the source, and the key at the end maps each tag to its REPORT section and files.
Nothing here amends a verdict.*

*Pointer (2026-10-16, after this summary): the e72 raw-residual pilot (REPORT.md, final section) passes the positive control on raw L12 h and gives GO by its frozen rule on one thin lead (skew #49, the same unit as decoded skew #31, ~76% SAE share); it is not recommended for fresh text.*

*Pointer (2026-10-16): the e72b selection-matched shuffle null (REPORT.md, final section) is INDETERMINATE by its frozen rule (screened above-max units 2, 2, 1, 1, 0 on 5 new shuffles vs D2's 4). The full e72 lead rule fires once on 4 of 5 shuffles, as on D2, so skew #49 carries no evidence on its own. No fresh text on it.*

*Pointer (2026-10-17): the e72f fragmentation test (REPORT.md, final section) is INCONCLUSIVE on fresh block F (corpus 8,000–8,999). The quote anchor, the e66 direction itself (its fourth behavioural block, Δ 1.78 vs RAND max 1.14), the two-class premise and the planted control pass. But the bank-median precondition fails (S 1.29 vs RAND max 2.21), so the e66 direction's best 16k-big SAE row cannot be classed as one-sided or two-sided. Post hoc readings: the precondition compared a bank median with a heavy-tailed max-of-99 (RAND S p95 0.94); and the row's tilt sits inside its bank's band on both readouts while that band caught half-strength planted tilts, so no one-sided behaviour was seen (not a verdict). The fragmentation line is closed at 1B by the user's decision. Nothing is amended.*

*Pointer (2026-10-17): the e73 coverage-test admission (REPORT.md, final section) tried to manufacture a positive control for "uncovered" by knocking out L12 16k latents for paren and quote, on burned D2 rows (CPU only). No test is admitted for l0_big or l0_small: error share and marginal gain fail on the paren control (first failures: rule 2 for error share on l0_big, rule 1 otherwise), and the geometry-only dummy fails as required. On l0_big, a probe still recovers paren from the reconstruction after the largest knockout (0.664, against a chance band topping out near 0.52), so the knockout never made paren uncovered. With the user's acceptance, R5/R6 are closed at L12 for the 16k family. This answers SUMMARY §7's open item for now: no usable "not a blend" criterion exists at 1B with these dictionaries. Nothing is amended.*

*Correction (2026-10-24), to two readings in this summary and one figure in REPORT.md. (1) The e66 false-pass bound of ≤ 1/77 (§3) is the preregistered rank bound (`PREREG_occurrence_v2_steering.md` §9, restated in `PREREG_occurrence_v2_confirm.md` §6). It holds only under the exchangeability nulls stated there: absent a specific effect, the direction's push moves its signature like any of the 99 random pushes, or its signature responds like any of the 76 other signatures. Neither the direction nor its signature was assigned at random, so randomisation does not establish either null. The family was also selected from the 31 L12 candidates by its B3 and B4 passes on the same rig, and under the prereg's stable-rank null a unit that has passed twice is more likely to pass again. So 1/77 is not the probability that this family is a false pass. The evidence is the margin over the controls: Δ 1.49 against a random-direction maximum of 0.75 (median about −0.04) and a shuffled-signature maximum of 0.85. (2) "Recovers known in-context state blindly" (headline 1) means the units are extracted without labels. The instrument's settings were chosen with parenthesis labels on calibration block P2 (corpus 13,000–13,999), so parenthesis state is a tuned target and quotation, never used in any choice, is the untuned control. (3) REPORT.md's e66 table gave logcosh #13's Δ at c = 4 as 2.89; `results/e66_confirm.json` gives 2.8846, so the table now reads 2.88. The write-up at <https://casella.dev/blog_gemma_blind_search.html> gives the same account. No verdict is amended.*

## Headline: a methods result, not a discovery

1. **A validated, identity-removed per-occurrence instrument.**
   - Per-type covariance whitening followed by full-rank ICA-64 recovers known in-context state blindly at L7 and L12. (Units are extracted without labels, but the settings were tuned with parenthesis labels on P2; see the 2026-10-24 correction above.)
   - Parenthesis state reaches |AUROC| 0.779 (L7) and 0.736 (L12), and the untuned quote control 0.747 (L12). The token-matched null p99.9 is about 0.52 and the preregistered floor 0.70. [S4v2]
2. **One direction was confirmed out of sample, and then found COVERED as a blend.**
   - The L12 family logcosh #13 / skew #16 passed a preregistered steering test on untouched contexts: Δ 1.49, against a random-direction maximum of 0.75 and a shuffled-signature maximum of 0.85. [e66]
   - Existing Gemma Scope 2 latents already encode its construction: a clause core completed, with an adjunct next. It is best described as a blend of subordinator and preposition latents. [e67]
3. **For these decoded-span units, "unnamed" ≠ "uncovered", and no coverage test tried here can certify "uncovered".**
   - **The single-latent census** fails calibration: it marks 3 of the 4 L12 paren/quote controls NOT COVERED, and 6 of 40 random L7 directions COVERED. [e68]
   - **Sparsity relative to a random null** does calibrate, but only one way. It flags #13/#16 and the quote controls as sparser than chance, yet its "chance-like" class also holds the paren controls, which have dedicated SAE latents. [e71]
   - **A small-K linear readout** from a sibling dictionary does not reach R² 0.5 even for the known anchors. [e71]
   - **e68's behavioural match** ("a latent fires where the unit fires") is met just as well by random same-subspace directions at L12. [e71]
4. **Every screen on a derived direction needs a random baseline drawn in the same subspace.**
   - Without one, the SAE-gap pilot's GO was uninformative: 71% of random error-space directions pass the same shuffle bar, and the survivors sit at the random median. [e70]
   - The same lesson recurs:
     - in e64's sibling null, made of real effects;
     - in e67's covering cosines, which sat inside the subspace null;
     - in e68's L7 random-direction COVERED rate;
     - in e71's context match.

No unnamed structure is claimed at 1B scale.

## 1. Setup and discipline

- **Model and dictionaries.**
  - `gemma-3-1b-pt`.
  - The searched representation is the SAE-decoded residual of `resid_post_all` 16k l0_big at the layer: fp16(codes) @ W_dec.
  - Coverage checks use 27 primary Gemma Scope 2 dictionaries in all: 13 for L7 (L5–L9, plus `resid_post` L7) and 14 for L12 (L10–L14, plus `resid_post` L13). [e68]
- **Corpus.**
  - The FineWeb `sample-10BT` stream: 15,696 sequences × 255 rows, verified against all 5,152 stored row→token anchors. [S0]
  - Stage-4 data blocks are disjoint: P2 (13,000–13,999) for calibration, D2 (1,000–1,999) for units, and reserve blocks for replication, steering and confirmation. **No untouched reserve remains for Stage-4 work** after e66. Corpus 2,000–12,999 is still unused for occurrence work. [S4v2, e66]
- **Discipline.**
  - Every prereg, rubric and script was hashed in `docs/HASHES.txt` before the step it governs.
  - Earlier verdicts were never amended. Post hoc redesigns are labelled post hoc.

## 2. The instrument: why v1 failed and v2 works

**v1 failed its positive control. The stated cause, identity left in the second moment, was incomplete.** [S4v2]

| v1 defect | evidence |
|---|---|
| ICA searched a noise-chosen subspace: `FastICA(n_components=32)` on input already whitened | split-half reproducibility 0.37–0.39; 0.74–0.91 at full rank |
| TDI counted self-matches | on D2, 8.7 / 7.3 / 7.4 as counted, 6.9 / 5.2 / 5.8 with the row excluded |
| floors could not be met | cluster F1 capped at about 0.41 at K = 128; the ICA floor of 0.85 sat above the supervised probe ceiling of 0.83 |

**v2** works as follows:
1. type, document and position means are removed, iterated three times;
2. PCA-64, whitened;
3. per-type covariance equalisation (types with ≥ 32 rows, shrinkage n0 = 16);
4. global re-whitening;
5. full-rank ICA-64, logcosh and skew.

**Results on D2** [S4v2]:

| | L7 | L12 | L24 |
|---|---|---|---|
| TDI at K = 128 (v1 → v2) | 6.85 → 1.76 | 5.23 → 1.57 | 5.85 → 1.71 |
| paren, logcosh \|AUROC\| (null p99.9) | **0.779** (0.519) | **0.736** (0.520) | 0.558 (0.516) |
| quote, logcosh \|AUROC\| (null p99.9) | 0.621 (0.512) | **0.747** (0.515) | 0.587 (0.512) |
| verdict | PASS | PASS | FAIL (floor above the probe ceiling) |

- A single direction carries 90–97% of the held-out supervised probe's excess over chance. [S4v2]
- **It is not a clean detector:** top-1% precision is 50–69% for paren and about 90% for quote.
- Identity is reduced, not gone: kNN identity (KIR) is still 9–15× chance. [S4v2]

## 3. The search, and the one direction that was confirmed

**Discovery tier: no discovery.** [e64]
- 143 screened units (65 at L7, 78 at L12) gave 34 and 31 discovery candidates (22 and 21 families).
- Both behavioural anchors failed the frozen tier-2 rule. Under that rule a unit must beat the *maximum* over its sibling units, and siblings carry real effects: the L12 subject-NP family reaches Δ 1.52, against the quote anchor's 0.246.
- 23 unnamed families replicate geometrically on R2. A post-verdict mechanical check (e64c) finds about 4 of them nameable, so about 19 remain.
- The context reading found that many battery-unnamed L12 units are syntactic-position variables: negation scope, subject NP, fronted adverbials, copular predicates.

**Redesigned steering, e65 (post hoc).** [e65]
- Changes: natural-scale dose (c·σ_d), 99 random directions in the units' own subspace, and a shuffled-signature control.
- The L12 quote anchor validated on two blocks: Δ 0.856 (random max 0.417, SIG max 0.570) and Δ 0.873 (0.560, 0.546).
- One candidate family, logcosh #13 / skew #16, passed both blocks.
- The L7 anchor failed again, so L7 has no validated behavioural test.

**Confirmation, e66, preregistered on untouched reserve 6,000–6,999.** [e66]

| unit | Δ at c\* | random max | SIG max | PASS |
|---|---|---|---|---|
| anchor logcosh #33 (quote) | 0.84 | 0.48 | 0.51 | yes |
| **primary logcosh #13** | **1.49** | 0.75 | 0.85 | **yes** |
| skew #16 | 1.38 | 0.72 | 1.03 | yes |

- The per-test false-pass bound is ≤ 1/77 ≈ 0.013. [e66] (A rank bound under exchangeability nulls, not the probability that this family is a false pass; see the 2026-10-24 correction above.)
- **Described after the fact:** pushing the direction promotes adjunct openers (*during, because, while, before*; *with, through, from*) by 2–2.6 nats. Pulling it closes the clause or line. [e66]

**Prior-feature check, e67: COVERED.** [e67]
- Three existing latents match the direction's behaviour and contexts: L12 16k-big #11562, L12 262k-small #2084 and L13 16k-medium #183.
- Their decoder |cos| (0.71–0.74) sits inside the same-subspace null (p99.9 0.74–0.85). The best activation r is 0.40.
- Reconstruction to cos 0.9 takes 11–62 decoder rows, against 39–84 for random directions of the same kind.
- **Reading:** the direction unifies the preposition and subordinator halves of one behaviour that the SAEs split. That is methodologically useful, but it is not a new concept.

## 4. Why "unnamed" ≠ "uncovered" for these units

**The units live in the source SAE's decoded span.** Within a token type, a unit's score is a linear readout of the source dictionary's codes, so some coverage by that dictionary is built in. [e68]

**e68, the single-latent census.** The calibration gate failed at both layers, so no target was nominated. [e68]
- **Gate:** 4 of 8 gate units are NOT COVERED at L7, and 7 of 14 at L12. The L12 failures include 3 of the 4 paren/quote controls. The bar was 25%.
- **No unit at either layer reaches activation |r| ≥ 0.7.** The controls peak at 0.34–0.44.
- **The decoder route does not discriminate at L7:** 6 of 40 random directions come out COVERED, and the subspace null p99.9 is 0.83–0.90.

**e71, a per-family readout of the 7 would-be targets plus #13/#16.**
- It ran under a rubric hashed before any per-family number (`NOTE_family_coverage_readout.md`).
- It is descriptive only: nothing is nominated and nothing is amended. [e71]

| family | e68 | best \|cos\| / max \|r\| | e68 matches (heavy side) | fixed-set match rate (nulls ≥, of 40) | sparsity vs null | readout R², K = 32 | post hoc reading |
|---|---|---|---|---|---|---|---|
| L7 logcosh #18 / skew #36 | NOT COVERED | 0.70 / 0.24; 0.63 / 0.23 | 76 (0); 70 (70) | 0.57 (0); 0.50 (0) | — | — | inside a punctuation-joined unit (*end-to-end*, *U.K.*, *p.m.*); anti-aligned twins |
| L7 logcosh #33 | NOT COVERED | 0.51 / 0.22 | 50 (45) | 0.38 (0) | — | — | the determiner after *at* (*at that time*, *at my age*) |
| L7 logcosh #49 / skew #8 | NOT COVERED | 0.60 / 0.32; 0.57 / 0.31 | 57 (55); 51 (51) | 0.56 (0); 0.46 (0) | — | — | the token after *what* (*what have*, *What will*) |
| L7 logcosh #52 | NOT COVERED | 0.72 / 0.21 | 19 (19) | 0.15 (4) | — | — | after an affective verb (*love the*, *proud of*) |
| L7 logcosh #57 / skew #33 | NOT COVERED | 0.52 / 0.21; 0.52 / 0.24 | 62 (62); 62 (62) | 0.46 (0); 0.40 (0) | — | — | after a quantifier (*all the*, *many a*) |
| L12 logcosh #45 / skew #5 | NOT COVERED | 0.54 / 0.29; 0.55 / 0.28 | 49 (49); 32 (32) | 0.32 (5); 0.17 (17) | chance-like | 0.20; 0.20 | blog meta-discourse (*In this article*, *Below,*); 16k-small #1276 shares 8 of its 12 strongest rows |
| L12 logcosh #54 / skew #54 | NOT COVERED | 0.49 / 0.22; 0.47 / 0.18 | 99 (88); 89 (89) | 0.45 (1); 0.47 (0) | chance-like | 0.16; 0.16 | "a verb comes next", split over ≥ 4 latents (*in hopes **of** stemming*) |
| L12 logcosh #13 / skew #16 (reference) | COVERED | 0.74 / 0.38; 0.72 / 0.40 | 60 (57); 60 (60) | 0.41 (2); 0.42 (2) | sparser | 0.23; 0.31 | e67: clause core complete, adjunct next |
| joint nulls, L7 / L12: median (max) | — | — | 6.5 (46) / 37 (96) | 0.02 (0.29) / 0.14 (0.46) | L12: 38 of 40 chance-like | L12: 0.11 (0.15) | — |

What the three e71 measures show [e71]:

- **Sparsity relative to the null is informative, but only one way.**
  - The rule was calibrated on 6 anchors under a cap of ≤ 2 of 40 random directions. "Sparser" = at least 7 of 14 dictionaries at or below the reconstruction-null p1.
  - It calls #13/#16 and both quote controls sparser, and 2 of 40 nulls.
  - It agrees with e68's decoder route for 12 of the 14 COVERED L12 units.
  - **But both paren controls read chance-like, and those variables have dedicated SAE latents.** Both L12 would-be targets read chance-like too. So sparsity can certify "blend-like", but it cannot certify "not a blend".
- **The small-K readout is uninformative.**
  - No anchor reaches held-out R² 0.5 at any K ≤ 64; the anchor maximum is 0.307.
  - At K = 32, anchors (0.23–0.31) and targets (0.16–0.20) sit above the null maximum (0.151). That is descriptive only: the frozen bar is not met.
- **Context sharing is universal, so it is not evidence.**
  - Every family has 19–99 latents per member that e68 records as firing where the unit fires.
  - But at L12, 19 of 40 random same-subspace directions have a latent with tail overlap ≥ 0.95, and no control exceeds the null maximum on the fixed-set match rate.
  - At L7, targets other than #52 exceed every null, as the L7 controls do.
  - Any direction in the decoded span has a heavy tail made of rows where some latents fire strongly.
- **Per-dictionary counts are not independent.**
  - A random direction's sparsity band is strongly correlated across the 14 dictionaries: 11 of 40 joint nulls sit above the reconstruction-null median in every dictionary, and 18 in none.
  - Cell by cell they sit above it 45% of the time.
  - So count rules must be calibrated on random directions built like the units. That is also why the "denser" class came out empty.

## 5. Same-subspace random baselines

**The SAE-gap pilot (e69)** ran temporal blind source separation on the identity-removed L12 SAE error. By its frozen rule it gave **GO**: 2 of 3 slow error units kept 46% of their lag-1 autocorrelation on shuffled text, against a 50% bar. [e69]

**The pre-freeze review of its fresh-text prereg (e70)** reran the bar on 100 random error-space filters, on burned D2 only. [e70]
- **71% of them pass the same shuffle test.** The survivors sit at the random median, so the GO carried no information about the units.
- Along the survivors' filters the raw residual is about 70% SAE-reconstructed: error share 0.26–0.31, against a 0.5 bar.
- The prereg was not frozen, and no fresh text was spent.

| where | bar without a same-subspace baseline | what the baseline showed |
|---|---|---|
| e64 tier 2 | Δ above the sibling maximum | siblings are real effects (Δ up to 1.52); the anchors cannot pass [e64] |
| e67 | decoder \|cos\| 0.71–0.74 above the isotropic null (p99.9 0.15–0.17) | inside the subspace null (p99.9 0.74–0.85) [e67] |
| e68 | decoder \|cos\| ≥ 0.7 + behavioural match | 6 of 40 random L7 directions COVERED [e68] |
| e69 | shuffle persistence ratio < 0.5 | 71% of random error-space filters pass [e70] |
| e71 | latent fires where the unit fires | 19 of 40 random L12 directions have a latent with tail overlap ≥ 0.95 [e71] |

## 6. Earlier stages, briefly

- **Deep layers L12–24.**
  - Months and weekdays are not recovered blind. The SAE type-mean ring is open at depth, and the two-channel tier has no deep anchor.
  - The blind scan's 43 flags (the reference was 114) are all lexical classes. [S1]
- **Lines and grids.**
  - The instrument detects planted rulers about 90% of the time.
  - The only real ruler, number words, is exactly frequency-sorted (ρ = −1.00).
  - The kinship lattice (supervised R² up to 0.40) is invisible to the parallelogram statistic. Not validated. [S2]
- **Discovery-verbs retest.** FAIL, rank p = 0.065 among its 30 siblings. Retired. [S3]

## 7. What remains open

- **A "not a blend" criterion.**
  - The 4B scale-up (an estimated 9–13 h locally) stays deferred until a written criterion exists. [S5]
  - e71 shows what such a criterion cannot rest on:
    - single-latent cosines or correlations;
    - small-K linear readout;
    - tail-overlap context sharing;
    - sparsity's complement.
  - It needs a positive control known *not* to be a blend, and this project has none.
- **Fresh text.** Corpus 2,000–12,999 and 15,000–15,695 are unused for occurrence work. Any confirmatory round on a new target needs a new prereg written before that text is read.
- **The e65 steering rig is validated at L12 only.** The L7 anchor fails twice.

## Source key

| tag | REPORT.md section | main hashed outputs |
|---|---|---|
| S0 | Stage 0 — budget, SAEs, corpus | `BUDGET.md` |
| S1 | Stage 1 — deeper layers | `results/e42d_deep*.json`, `token_type_geometry_deep*.json` |
| S2 | Stage 2 — lines and grids | `results/e46_*.json` |
| S3 | Stage 3 — discovery-verbs retest | `results/e53_verbs_retest.json` |
| S5 | Stage 5 — scoping gemma-3-4b-pt | (estimate only) |
| S4v2 | Stage 4 v2 (positive control) | `results/e63_occurrence_v2.json`; prereg `PREREG_occurrence_v2.md` (`edf75750…`) |
| e64 | Stage 4 v2, discovery tier (+ e64c post-verdict check) | `results/e64_verdict.json`, `e64_behaviour.json`, `e64_replicate.json`, `e64c_context_rules.json` |
| e65 | Stage 4 v2, redesigned steering tier | `results/e65_steer_B3.json`, `e65_steer_B4.json`, `e65_verdict.json` |
| e66 | Stage 4 v2, confirmation | `results/e66_confirm.json` (`e8ca659d…`), `e66_describe.json` |
| e67 | Prior-feature check | `results/e67_decoder.json`, `e67_acts.json`, `e67_verdict.json` |
| e68 | Coverage census | `results/e68_census.json` (`b599ec0f…`), `e68_decoder_G{7,12}.json`, `e68_acts_G{7,12}.json` |
| e69 | SAE-gap pilot | `results/e69_pilot.json` (`05dcc30f…`) |
| e70 | SAE-gap round prereg (not frozen) | `results/e70_review_verification.json` (`10da0ace…`) |
| e71 | Per-family coverage readout | `results/e71_calib.json` (`e94e9e58…`), `e71_readout.json` (`e6bf66e0…`); post hoc `e71_match_null.json` (`d4a97432…`), `e71_null_band_check.json` (`0f3a1600…`); rubric `NOTE_family_coverage_readout.md` (`6d1a479b…`) |
