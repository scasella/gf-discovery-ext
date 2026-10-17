# PREREG — Stage 4 v2, redesigned behavioural tier: natural-scale steering of the frozen L7/L12 units (FROZEN 2026-10-15, revision 2)

**Revision 2. Supersedes revision 1** (sha `d3560312…`, with script `6cc1e966…`). Both were frozen
at 15:17 and **superseded before any e65 phase ran**. Revision 2 changes four things:
1. It adds §0.6: the post-verdict e64c check of the context reading, which arrived after
   revision 1 was frozen.
2. It bars from the top verdict class any family that an e64c rule names at |AUROC| ≥ 0.65
   (§11). This is the only rule change, and it is implemented in the script.
3. It corrects a disclosure in §0.2.
4. It adds crash recovery (§13).

Nothing else changed.

> **THIS REDESIGN FOLLOWS A SEEN FAILURE.**
>
> The e64 behavioural tier ran on 2026-10-15 and **both anchors failed**:
> - the L7 paren anchor barely moved its signature;
> - the L12 quote anchor had a clean, dose-dependent effect, but lost to the sibling maximum, which
>   was set by the L12 subject-NP family.
>
> Everything in this file was written **after** those results, the full sibling bank and every
> other e64 output had been read. Three consequences follow:
> - **The new null (§7) and the new dose (§6) are chosen post hoc.** They answer a known failure
>   mode. They were not fixed in advance of it.
> - **The anchors are no longer blind.** Their old effect sizes, sign consistencies and signatures
>   are known (§0.2). A pass by an anchor here is weaker evidence that the rig works than a blind
>   pass would have been.
> - **Any pass on the test block B3 is EXPLORATORY.** It counts as more than exploratory only if it
>   passes again, under the same rule, on the further held-out context block B4 (§10), with the
>   layer's anchor passing on B4 too. Even then it carries the label *post hoc* (§11).
> - **The post-verdict e64c rule check has also been read** (§0.6). It finds that 9 of the 65
>   "unnamed" candidates are nameable after the fact by a simple rule. Those families can never
>   reach the top class here.
>
> **The e64 verdict ("no discovery") is not amended by anything here.**

When this file's sha256 was appended to `docs/HASHES.txt`:
- no e65 phase had run;
- no reserve token in 4,000–7,999 had been processed by any Stage-4 code;
- the model had not been loaded.

`scripts/e65_steering_redesign.py` asserts at every data phase:
- that this file's sha256 is in `docs/HASHES.txt`;
- that its own sha256 appears in this text (§12);
- that every input in §2 matches its sha256;
- that e64's own integrity check passes.

**This file authorises no run.** Each phase needs the user's go-ahead. Changing any rule here
means writing an addendum and hashing it before the first phase that the rule governs.

## 0. What was seen before freezing, and what that means

### 0.1 The failure this redesign answers

e64's tier-2 rule (v1 §9.2 and e62) was:
- push h ← h + ε‖h‖·d_u, with ε ∈ {0.005, 0.01, 0.02};
- PASS needs sign consistency ≥ 0.8, Δ above the maximum of 10 random full-space directions, and
  Δ above the **sibling p99**. With fewer than 100 siblings, that is the **sibling maximum**.

Two defects explain the failure (REPORT, "Stage 4 v2, discovery tier", Reading 1):
1. **The sibling null is a maximum of real effects, not of null draws.** Every sibling is a real
   v2 direction scored with its own blind signature. Some have large effects (L12 logcosh #10,
   Δ 1.52), so the maximum measures the strongest real unit at the layer, not chance.
2. **ε·‖h‖ ignores each direction's own scale.** The same fraction of ‖h‖ can be a negligible or a
   large push, depending on how much the residual naturally varies along d.

### 0.2 What was read while drafting (concrete)

**Documents:**
- REPORT.md, final section;
- `PREREG_occurrence_v2_discovery.md`, all of it, and §8 in detail;
- addendum D1;
- `scripts/e64_occurrence_v2_discovery.py`, the behaviour, verdict and reconstruct code;
- the hand-over note `gf-discovery-ext-stage4v2-discovery-prereg.md`;
- the headers and reserve-slicing lines of `e47r_ruler_behaviour.py` and `e53_verbs_retest.py`.

**From `results/e64_behaviour.json`, the anchors and the secondary anchors in full:**

| unit | Δ at ε 0.005 / 0.01 / 0.02 | sign consistency | old random max | sibling max (n) |
|---|---|---|---|---|
| L7 logcosh #12 (paren, anchor) | 0.003 / 0.007 / 0.012 | 0.3 / 0.4 / 0.4 | 0.022 | 0.150 (63) |
| L7 skew #62 (paren) | −0.002 / 0.002 / 0.007 | 0.55 / 0.55 / 0.3 | 0.030 | 0.150 |
| L12 logcosh #33 (quote, anchor) | 0.059 / 0.124 / 0.246 | 0.9 / 0.95 / 0.9 | 0.086 | 1.524 (76) |
| L12 skew #33 (quote) | 0.047 / 0.099 / 0.212 | 0.9 / 0.9 / 0.9 | 0.035 | 1.524 |
| L12 logcosh #11 (paren) | 0.032 / 0.062 / 0.118 | 0.65 / 0.8 / 0.8 | 0.085 | 1.524 |
| L12 skew #23 (paren) | 0.015 / 0.038 / 0.083 | 0.5 / 0.6 / 0.7 | 0.062 | 1.524 |

The anchors' signatures and random sets were read too:
- L7 paren: `)`, `),`, `).`, …;
- L12 quote: `,"`, `,”`, `."`, `.”`, ….

**The full sibling bank: every screened unit's own-signature Δ at ε = 0.02·‖h‖.**

| quantile | min | p10 | p25 | p50 | p75 | p90 | p95 | p99 | max |
|---|---|---|---|---|---|---|---|---|---|
| L7 (65 units) | −0.012 | −0.001 | 0.007 | 0.018 | 0.032 | 0.053 | 0.058 | 0.118 | 0.150 |
| L12 (78 units) | −0.037 | 0.037 | 0.072 | 0.125 | 0.208 | 0.326 | 0.517 | 1.446 | 1.524 |

**Which units these were.** The top 12 and bottom 6 units by that Δ were printed with their
identities:
- **L7 top:** skew #32 0.150, skew #21 0.101, logcosh #54 0.067, skew #39 0.059, skew #29 0.057,
  logcosh #11 0.056, logcosh #55 0.053, skew #35 0.052, skew #59 0.051, logcosh #31 0.049,
  skew #27 0.040, logcosh #49 0.038.
- **L7 bottom:** logcosh #34, #28, #38, skew #33, logcosh #15, skew #18 (−0.012 to −0.003).
- **L12 top:** logcosh #10 1.524, skew #15 1.423, logcosh #16 0.658, skew #61 0.645,
  skew #38 0.495, logcosh #23 0.459, logcosh #13 0.397, logcosh #21 0.338, skew #16 0.321,
  skew #52 0.301, logcosh #9 0.295, logcosh #24 0.255.
- **L12 bottom:** logcosh #29, #0, skew #11, #63, #31, logcosh #32 (−0.037 to 0.027).

**What was already known about these units' labels** (correction in revision 2):
- REPORT names the L12 subject-NP family (logcosh #10 / skew #15) as context-named.
- REPORT also says that several L12 units at Δ 0.46–0.66 are named by `next_punct`. Those are
  battery-NAMED, outside the candidate population.
- §0.6 adds what e64c says: three of L7's top-12 old movers (logcosh #31, skew #35, skew #59) and
  one of L12's bottom movers (logcosh #0, −0.021) are candidates that an e64c rule names.

Beyond that, the top movers were **not cross-referenced against the labels** while drafting.

**From the other e64 outputs, structure only:**
- per-layer counts:
  - L7: 65 screened, 40 battery-UNNAMED, 6 context-named, 34 candidates;
  - L12: 78 screened, 51 battery-UNNAMED, 20 context-named, 31 candidates;
- the control units and their |AUROC|s;
- the first three twin pairs at each layer;
- the first few context-reading entries (L7 logcosh #0, #3, #5, #6, #8: "none");
- the first candidate row of `e64_replicate.json` at each layer;
- REPORT's R2 and S2 summaries.

**From the maps:** `e64_maps_L7.npz` and `_L12.npz` were opened to list their keys. At L7, P_U was
checked to be orthonormal, and its first and last PCA scales were printed (139 … 16). No
direction, pattern or readout was inspected.

### 0.3 What is therefore not blind

- **The anchors.** Under e64's old rule, dropping the sibling gate alone would have let the L12
  quote anchor pass: sign consistency 0.9, Δ 0.246 against a random max of 0.086. The new gates
  are not that. The RAND null (§7.1) draws directions in the space the unit directions live in,
  not in the full 1,152-d space, and the SIG control (§7.2) is new, so **the new rule is not known
  to pass either anchor**.
- **The dose against the L7 anchor.** The L7 paren anchor is expected to fail again unless 2σ_d
  (§6) is a larger push than 0.02·‖h‖ was. Whether it is cannot be known before the run, and
  **c\* = 2 is fixed regardless**. The run reports the ratio for every unit.
- **The candidates' old effect sizes.** Their old effects at ε = 0.02·‖h‖ are in the sibling bank.
  The design was not tuned on them: no gate threshold here references a bank value, and every
  battery-UNNAMED unit is tested, as in e64 §8.1.
- **One analyst throughout.** The same analyst (Claude) did the e64 context reading, read the e64
  outputs and wrote this file.

### 0.4 Code run while drafting (no data)

**Selftest.** `e65 --phase selftest` passes all 12 checks, on synthetic arrays only:
- the parser for the e64c output (§0.6);
- the natural-scale σ and the within-type σ;
- the RAND draw and pullback;
- site draws;
- the lse readout;
- SIG set construction;
- the gates;
- the false-pass bounds, by Monte Carlo:
  - exchangeable null: P(RAND) = 0.0104 against 1/100;
  - exchangeable null: P(SIG) = 0.0158 against 1/64;
  - the conjunction, 0.0064, is ≤ the smaller of the two;
  - a generically potent direction with no pairing effect passes both at 0.0160 against 1/64.

**A mock run on a tiny fake model.** It used synthetic tokens and synthetic directions, in the
session scratchpad, on CPU. It exercised `steer` B3, `steer` B4 and `verdict` end to end:
- planted effects passed;
- unplanted ones failed;
- the anchor gate, the B4 plan and the verdict classes behaved as specified;
- in revision 2, a family flagged by a synthetic e64c file lost the top class, and an unflagged
  family kept it.

The mock read no reserve, corpus or e64 data and loaded no model.

**The e64c parser on the real file.** The script's e64c parser was run on the real
`results/e64c_context_rules.json`, which had already been read for §0.6. It returns exactly the
9 units listed there, plus 13 + 4 threshold hits.

**sha256 only.** `occD2_meta.npz`, `corpus_tokens_pass1.i32`, `reserve_tokens.i32` and
`lag_deltas_L12.npz` were hashed (bytes only) so that §2 can pin them.

### 0.5 Earlier use of the reserve (8,000 sequences after the 4M-token corpus)

| reserve | earlier use |
|---|---|
| 0–999 | v1's R: extracted and deleted, never analysed |
| 1,000–1,999 | v1's behavioural block: never used |
| 2,000–2,999 | e64's R2 (replication): **used** |
| 3,000–3,999 | e64's steering sites: **used** |
| **4,000–7,999** | **Not used by any Stage-4 code.** Touched only as 8-token slices by: **e47r** (Stage 2 ruler behaviour; `reshape(-1, 256)[2000:8000, 1:]`) and **e53** (Stage 3 verbs retest; all 8,000 sequences). |

**What e47r and e53 did.** Both steered type-mean objects:
- ruler PC1 axes (e47r);
- verb planes (e53).

Both substituted a member token at the site. Neither computed an occurrence representation, and
neither selected or informed any Stage-4 unit. No analyst read those contexts as text during
Stage 4.

**The overlap.** The B3 and B4 prompts (up to 64 preceding tokens) can contain the same token
spans as those 8-token slices. There is no selection link between those runs and these units.

**This prereg uses 4,000–5,999 only. 6,000–7,999 stays untouched** and is available for any later
confirmatory test.

### 0.6 The post-verdict e64c check of the context reading (seen; read for revision 2)

**What e64c is.** Another session turned the 14 context-reading constructions into fixed rules:
- lexical previous-token rules;
- spaCy-parse rules.

It scored them against the frozen D2 unit scores. It wrote `scripts/e64c_context_rules.py`
(sha `a6eafd2d…`) and `results/e64c_*`, all hashed at 15:09, and a new final subsection of
REPORT. **It changes no e64 tier and no e64 file**; `e64.check_integrity()` still passes.

**Read for this revision:**
- that REPORT subsection;
- the parent session's summary;
- in `results/e64c_context_rules.json`: each rule's `top5` and `would_name` lists, for both
  parsers.

**What it found:**
- **Precision.** 11 of the 14 constructions are confirmed. "Onset of the next item in a series"
  (the L12 #37 pair) is not supported. Enumeration and temporal expressions are weak.
- **Recall.** The reading under-called: several candidates read as "none" are matched well by a
  rule. Transformer parse, |AUROC| ≥ 0.65, oriented as e64c reports it:

| layer | candidate | rule, \|AUROC\| | replicated on R2 (REPORT) | old own-signature Δ (§0.2) |
|---|---|---|---|---|
| L7 | logcosh #31 / skew #35 (twins) | after "the": 0.867 / 0.851 | yes | 0.049 / 0.052 (L7 top-12) |
| L7 | logcosh #41 / skew #41 (twins) | negation scope: 0.682 / 0.674 | yes | not listed |
| L7 | skew #59 | multi-token subject NP: 0.659 | no | 0.051 (L7 top-12) |
| L7 | **logcosh #53** | **line-initial: 0.660** | no (`e64_replicate.json`: R2 match at cos 0.97 is not screened on R2) | not listed |
| L12 | skew #8 | after "in"/"at": 0.794 | yes | not listed |
| L12 | skew #0 / logcosh #0 (twins) | after and/or/but: 0.712 / 0.703 | yes | logcosh #0: −0.021 (L12 bottom) |

**A discrepancy with REPORT.** REPORT's recall table lists 8 units. **L7 logcosh #53**
(line_initial, 0.660) is in the JSON's top5 for that rule but not in REPORT's table. Why is not
known; it may have been omitted deliberately because line-initial is close to the battery's
line/newline features. **This prereg includes it**, the conservative direction. In practice it
changes nothing: #53 did not replicate on R2, so it could not reach the top class anyway.

**A coverage limit.** The JSON keeps only each rule's top five units and its §6.1 hits. A
candidate between 0.65 and a rule's fifth-best value would not be visible in it. REPORT, computed
from the full scores, lists no such unit beyond the ones above.

**Under e64 §6.1 thresholds.** These rules would name **17 candidates** (13 at L7, 4 at L12), many
of them weakly, through the enrichment clause at |AUROC| ≈ 0.55.

**What this revision does with it:**
- **The steered set and the candidate count do not change.** The brief keeps e64's labels and
  context reading frozen. The e64c rules see whole-sequence parses, including future tokens, and
  were not preregistered as naming rules. All 34 + 31 candidates are steered, and all count in
  §9's Σ.
- **A family with any member in the table above can never reach the top class** (§11). It is
  reported as "held-out behavioural pass, nameable post hoc by an e64c rule (not a discovery
  class)".
- The script parses this set from the pinned JSON and **asserts that it equals the 9 units
  listed** (`E64C_STRONG`).
- The 17 threshold hits are reported per unit, not barred (§14, item 14).

## 1. Question and scope

**Question.** The question e64 §8 asked, with a rig whose null and dose fit v2 units: does a frozen
v2 direction, pushed on its own natural scale, specifically raise the probability of its own blind
next-token signature?

**Scope:**
- the 143 frozen screened ICA-64 units at L7 and L12 (logcosh and skew);
- roles taken from e64's frozen labels and sealed context reading:

| role | L7 | L12 | steered at |
|---|---|---|---|
| anchor | logcosh #12 (paren) | logcosh #33 (quote) | every dose, first |
| secondary anchors (other control units) | skew #62 | logcosh #11, skew #23, skew #33 | every dose, reported |
| discovery candidates (battery-UNNAMED, not context-named) | 34 | 31 | every dose, if the anchor passes |
| NAMED_BY_CONTEXT | 6 | 20 | every dose, if the anchor passes; reported separately, never a discovery class |
| battery-NAMED, not a control | 23 | 23 | c\* only, if the anchor passes; for the sibling rank only (§7.3) |

**Not in scope:**
- S2, R2, the labels and the context reading. They are not recomputed; their e64 outputs are used
  as frozen.
- L24 and clusters, as in e64.

## 2. Frozen inputs (unchanged; asserted by sha256 at every data phase)

| file | sha256 | role |
|---|---|---|
| `scripts/e64_occurrence_v2_discovery.py` (D1 version) | 1c97e42b4fc2fbf64410e8dda39fb8a9d6ae938e3fa960e6bf36d1a009fcf1f4 | imported as a module: `signature`, `screened_ids`, `control_units`, `candidates`, `load_maps`, `ANCHOR`, hashing, swap guard; `e64.check_integrity()` is also run (its prereg, D1 and the frozen v2 inputs) |
| `results/e64_maps_L7.npz` | 1b0c38ab872bb55f8c1af5127267ef3123dc4d743bd7d6c51db23db49e0b4faf | maps, unit directions d_u (`dir_*`), heavy-tail signs, P_U and P_sd |
| `results/e64_maps_L12.npz` | ac1b55b2eee03594b2d9f512632db2e9980e2446d051560d950ade8726200237 | the same |
| `results/e64_labels.json` | c86d651922a8929f2c9b8d17b1258e3082d14c6528a5b5a24ceb073e5bdf1336 | NAMED / UNNAMED, twins |
| `results/e64_context_reading.json` | 4847f400019b720c3e38702b52d98a80613460ecc409b49d2455c06d491f3a52 | NAMED_BY_CONTEXT |
| `results/e64_shuffled.json` | 342268918d4255dfda9adbe170e8e501c41c54526f0cd9f929cce6db8dd55b7a | S2 labels (carried into the verdict) |
| `results/e64_behaviour.json` | c5a213e49b87830e02fa01eced2c1ca2d93a5afe4a34a03fa92a5aea4b6c6bb8 | the control units' signatures and random sets: the rebuild acceptance check (§5) |
| `results/e64_replicate.json` | b781ea197bb10752133ad570b04b974b0300bf068d1387137fde339ab31b423f | tier 3 |
| `results/e64_verdict.json` | bacdc3ee6f4845ab87875283e1b8f4b1c3fafd1730b747f15ef4fac247804ea9 | tiers 1 and 3 per unit; **not amended** |
| `results/e64c_context_rules.json` | 36f52a478cc76b7e0751e721f0c68436f3d7ae328a9c7e87bf7b20a1143f018d | the e64c rule matches: the top-class bar and the per-unit flags (§0.6, §11) |
| `cache/occ/occD2_meta.npz` | bc910456116d5b875438c741c42379e24715576a2bea29e33b0dd9c1a92bc5a7 | D2 tokens for the signature rebuild (newly pinned here) |
| `results/corpus_tokens_pass1.i32` | 8fbf98c3f29ebf9785885c1ac2df8ebeee7d0978c4663cd25517c5a925ba517e | corpus counts for R_u (newly pinned here) |
| `results/reserve_tokens.i32` | 3da028b79acf01f7b67e70eb602ebed37a8883b074f9dac9f08f6ccc2b79b5b5 | B3 and B4 contexts (newly pinned here) |
| `results/lag_deltas_L12.npz` | 65b546b3ebb6f7d0a75807e68d712916697b14408008c176eb3af3075afa21ba | L12's dropped coordinate, via `e61.drop_for` (newly pinned here) |

**Unchanged objects:**
- the units;
- the maps;
- the **pattern pullback direction**: d_u = sign_u · normalise(U_pca · diag(sd) · Cov(Z, ỹ_u)),
  read from the maps (e64 §8.2);
- the **blind signatures** T_u and **random sets** R_u, rebuilt by e64's code (§5);
- the labels;
- the context reading;
- the intervention site: the output of decoder layer L, at the site position only;
- the readout: Δ = [lse(logits[T_u]) − lse(logits[R_u])] steered, minus the same unsteered;
- the sign-consistency definition (§8);
- the model and its dtype: gemma-3-1b-pt, bf16, MPS, eager attention, as in e64;
- the prompt: BOS, then up to 64 preceding reserve tokens, then the site token.

## 3. What changes from e64 §8, and why

| item | e64 | here | why |
|---|---|---|---|
| dose | ε·‖h‖, ε ∈ {0.005, 0.01, 0.02} | c·σ_d, c ∈ {0.5, 1, 2, 4}, gate at c\* = 2 (§6) | a fraction of ‖h‖ is not a unit's natural scale |
| random null | 10 full-space random directions orthogonal to d_u, max | 99 **pullback-matched** random directions per layer and block, each dosed by its own σ, max (§7.1) | full-space directions mostly leave the 64-d subspace the units live in, which makes them a weak null |
| sibling null | the **maximum of real effects** (each sibling with its own signature) | replaced by the **shuffled-signature control**: u's own forward pass scored with every other unit's signature, max (§7.2); the sibling rank is reported only (§7.3) | a maximum of real effects measures the strongest unit, not chance; SIG asks whether u's effect is specific to *its* signature |
| sites | 200 from reserve 3,000–3,999, with replacement | 200 **distinct** sequences from B3 = reserve 4,000–4,999; confirmation on B4 = 5,000–5,999 (§4) | fresh contexts, a held-out block |
| status of a pass | tier 2 | B3 pass = exploratory; B3 + B4 pass = held-out, still post hoc (§11) | the design follows a seen failure |
| what runs if the anchor fails | the bank, then the anchor | the RAND bank and the anchor (+ secondaries); **nothing else at that layer** | no candidate is steered under a rig that failed its anchor |

## 4. Contexts: blocks B3 and B4

| block | reserve | role | site seed | RAND seed |
|---|---|---|---|---|
| **B3** | 4,000–4,999 | test block: anchors first, then every candidate | 6500 + L | 6510 + L |
| **B4** | 5,000–5,999 | held-out confirmation: the anchor, the secondary anchors and every B3 pass at the layer | 6600 + L | 6610 + L |

**Site rule** (`draw_sites`):
- 200 distinct sequences per block and layer, drawn without replacement;
- one position each, uniform in [5, 255] of the BOS-included 256-token row;
- prompt = BOS + `rv[s, max(1, p−64) : p+1]`, as e64 did.

The sites are drawn, written and hashed by the `freeze` phase **before any forward pass**.

**B4 runs at a layer only if both hold:**
- the B3 anchor PASSed there;
- at least one candidate or context-named unit PASSed B3 there.

Otherwise reserve 5,000–5,999 stays unused at that layer, and B4 writes a not-run record.

## 5. Signatures: rebuilt, not re-chosen (`--phase freeze`, CPU)

**The rebuild.** e64 stored the decoded T_u and R_u only for the six control units. For every
screened unit, the freeze phase rebuilds them with e64's own code, in e64's order:
- T_u = `e64.signature(...)` on u's heavy-tail 1% of D2 rows, with the next-token table built
  from `occD2_meta.npz` exactly as `e64.behaviour` built it;
- R_u = frequency-matched tokens (corpus count within ±10%, excluding T_u), drawn from a fresh
  `default_rng(6400 + L)` consumed over `[(kind, id) for kind in (logcosh, skew) for id in
  screened]`. This is e64's RNG and e64's order: in e64 the random sets were drawn before any
  other use of that RNG.

**Acceptance.** For every control unit, the decoded T_u and R_u must equal the strings in
`e64_behaviour.json`. That is 2 units at L7 and 4 at L12. If any differ: **STOP**. The output is
hashed first and then the assertion runs, as in e64's reconstruct.

**Also written and hashed by the freeze phase:**
- the B3 and B4 sites;
- the RAND banks (`results/e65_rand_L{7,12}.npz`).

The phase checks that the kept-coordinate mask from `e61.drop_for(L)` has zeros exactly where
every e64 unit direction is zero.

**Units with an empty signature** (fewer than one token with count ≥ 5) are recorded and never
PASS.

## 6. Dose on each direction's natural scale

**σ_d**, computed per layer and per block, for every direction d (unit or RAND) from the
**unsteered** forward pass:
- the residual at the output of decoder layer L;
- at every non-padding prompt position with index ≥ 4 (BOS = index 0), up to about 12,000
  positions over the block's 200 prompts;
- projected on d.

**σ_d** = the SD (ddof 1) of that projection. This is the dose scale. It is computed from the
inputs only, never from any Δ.

**Intervention:** h ← h + c·σ_d·d at the site position.
- **c ∈ {0.5, 1, 2, 4}.**
- The gates use **c\* = 2**.
- Sign consistency counts at any c ≤ c\*, that is {0.5, 1, 2}.
- c = 4 is descriptive: the shape of the dose response.

**Reported per unit:**
- σ_d;
- the **type-centred σ**: the pooled within-type SD over token types with ≥ 8 positions in the
  block;
- **c\*·σ_d / mean‖h_site‖**, the equivalent ε;
- its ratio to e64's 0.02 push;
- the **realised dose**: ‖h_steered − h‖ / (c·σ_d) after the cast back to bf16, averaged over
  sites. bf16 rounding can eat a small push, and this measures it.

**Why the raw SD, not the type-centred one.** The raw SD is computable from the block alone,
without any extraction. It is at least the within-type SD, so c\* = 2 is at least two within-type
SDs.

## 7. Nulls

### 7.1 RAND: pullback-matched random directions (a gate)

**Construction.** Per layer and per block:
1. Draw 99 vectors g ~ N(0, I₆₄) in D2's whitened PCA-64 coordinates.
2. Pull them back with the same map that e64 applied to Cov(Z, ỹ_u): U_pca·diag(sd)·g.
3. Embed them at the kept coordinates and normalise.

**How each is used:**
- each is dosed by **its own** σ_r at c\* = 2, so norm matching is implicit in the dose rule;
- the bank is shared by all units at the layer;
- each random forward pass is scored with **every** unit's own (T_u, R_u);
- the draws are **not** orthogonalised to d_u (the max |cos| to any unit direction is reported).

**Gate.** Δ_uu(c\*) > max over the 99 of Δ_ru(c\*), where Δ_ru is random direction r scored with
u's signature.

### 7.2 SIG: the shuffled-signature control (a gate)

**Construction.** u's own steered forward pass at c\* is scored with every other screened unit v's
signature at the layer, both kinds, under three rules:
- **u's twin is excluded**;
- **T_u's tokens are removed** from T_v and from R_v. Otherwise a T_u token inside R_v would lower
  Δ_uv, and a shared token would credit u's own effect to v;
- v is used only if both T_v and R_v keep ≥ 5 tokens.

**Gate.** Δ_uu(c\*) > max over the eligible v (n′_u of them) of Δ_uv(c\*).

**What SIG asks.** Is u's effect specific to its own signature, or does u's direction move any
signature built the same way?

**Also reported:**
- the three largest Δ_uv;
- the dropped v's;
- a scale-free version of the comparison, in which each Δ is divided by the across-site SD of
  that signature's unsteered readout, with u's rank among them. This is descriptive only:
  signatures differ in size and base rate, so Δ is not perfectly comparable across signatures.

### 7.3 Sibling rank (reported, not a gate)

When the anchor passes, every screened unit is steered at c\* with its own signature. For each
anchor, secondary anchor, candidate and context-named unit, the share of its siblings (twin
excluded) with a smaller own-signature Δ is reported.

**Why it is not a gate.** The siblings' effects are real effects. A percentile of them ranks u
among real units; it is not a null.

## 8. PASS rule (identical on B3 and B4)

PASS needs all four:
1. **Δ_uu(c\*) > 0**;
2. **sign consistency ≥ 0.8 at some c ∈ {0.5, 1, 2}**. This is e64's definition, unchanged: the
   share of the 20 commonest site token types whose mean Δ is > 0;
3. **RAND**: Δ_uu(c\*) > the maximum of the 99 RAND values for u's signature;
4. **SIG**: Δ_uu(c\*) > the maximum of the n′_u SIG values.

**The anchor gate is kept.** The anchor runs first at each layer, under this rule. If it fails,
**no behavioural verdict is issued at that layer**. The secondary anchors are still evaluated and
reported, and no other unit is steered there.

## 9. Expected false passes

**Per unit.** Both gates compare Δ_uu with a maximum of null draws, not of real effects.

| gate | null assumption | per-unit false-pass bound |
|---|---|---|
| RAND, under H0_R | d_u's effect on T_u is exchangeable with the 99 RAND directions' effects | P(Δ_uu > max) ≤ 1/(99 + 1) = 0.010 |
| SIG, under H0_S | T_u is exchangeable with the n′_u other signatures as a readout of d_u | P(Δ_uu > max) ≤ 1/(n′_u + 1) |

n′_u is at most 63–64 at L7 and 76–77 at L12, and smaller if signatures are dropped. So the SIG
bound is ≥ 1/64 = 0.0156 at L7 and ≥ 1/77 = 0.0130 at L12.

**The conjunction.** The two gates share Δ_uu, so the conjunction is **not** the product of the
two rates. Under the global null (H0_R and H0_S both true), P(PASS) ≤ min(0.010, 1/(n′_u + 1))
= 0.010.

**The case that matters, and the bound used.** A unit can be generically "potent": its direction
moves next-token predictions more than random directions do, with no specific pairing to its
signature (H0_R false, H0_S true). Then RAND passes easily and only SIG protects:
P(PASS) ≤ 1/(n′_u + 1). **The bound used is therefore max(0.010, 1/(n′_u + 1)) = 1/(n′_u + 1)**.
That bound is valid if either null holds.

**Expected false passes on B3, among discovery candidates:**

| layer | candidates | Σ 1/(n′+1) (used) | under the global null |
|---|---|---|---|
| L7 | 34 | ≈ 0.53 | 0.34 |
| L12 | 31 | ≈ 0.40 | 0.31 |
| both | 65 | ≈ 0.93 | 0.65 |

The script computes the exact Σ from each unit's n′_u. Context-named units are evaluated but are
not counted.

**Layer excess rule (B3).** At a layer, a candidate's B3 pass counts only if the layer's PASS count
among candidates exceeds that layer's Σ. With Σ < 1, one pass suffices.

**Why B3 alone is exploratory.** Across both layers the expected number of false B3 passes is
close to 1.

**B4.** B4 tests the anchor and each B3 pass on fresh sites, with a fresh RAND bank and a freshly
computed σ. The signatures and directions are the same. How much protection that adds depends on
the null:
- **If the null is exchangeable** (a null unit's rank is a stable property of that unit), the B3
  and B4 outcomes are positively dependent. The honest bound for a null unit passing **both**
  blocks is then the single-block bound, about 0.016 per unit. B4 then protects only against
  site-sample noise and the B3 winner's curse.
- **If the null is "u's expected effect is no larger than the strongest null draw"**, a pass needs
  favourable noise on each block, and the blocks' noise is independent. The two-block rate is
  then roughly the product, about 2.5·10⁻⁴ per unit, or about 0.016 expected across the 65
  candidates.

**This prereg claims only the first, conservative figure for a B3 + B4 pass.**

## 10. Order, with the anchors first

At each layer, L7 then L12.

**`steer --block B3`:**
1. The unsteered forward pass: logits, plus σ for every unit and RAND direction, plus ‖h‖ at the
   sites.
2. The 99 RAND forward passes at c\*.
3. **The anchor**, at c ∈ {0.5, 1, 2, 4}, then its gates.
4. The secondary anchors, at every dose (reported).
5. **Only if the anchor PASSed:**
   - every candidate and context-named unit, at every dose;
   - then every battery-NAMED non-control unit at c\* only (for §7.3);
   - then the layer's PASS count, its Σ bound and the excess flag.

**`steer --block B4`.** At each qualifying layer (§4), with B4's sites, σ and RAND bank:
1. the anchor first;
2. the secondary anchors;
3. then, **only if the B4 anchor PASSed**, every B3 pass.

If the B4 anchor fails, no B4 verdict is issued at that layer.

## 11. Outcomes and wording

**Per layer:**
- **Anchor fails on B3.** "No behavioural verdict at L (redesigned rig failed its anchor)."
- **The anchor passes on B3 but not on B4.** The layer's B3 passes stay **exploratory**.

**Per unit** (`results/e65_verdict.json`). Each unit carries e64's tier, labels, twin, identity
diagnostics, S2 label and R2 status, plus one class:

| class | requires |
|---|---|
| no behavioural verdict | the B3 anchor failed at the layer |
| no behavioural pass | — |
| **exploratory behavioural pass (B3 only)** | a B3 PASS with the B3 anchor passing (and, for a candidate, the layer excess), but no B4 pass |
| held-out behavioural pass, NAMED_BY_CONTEXT | B3 + B4 for a context-named unit; never a discovery class |
| held-out behavioural pass, not replicated geometrically | B3 + B4 for a candidate that failed e64's R2 replication |
| held-out behavioural pass, nameable post hoc by an e64c rule (not a discovery class) | B3 + B4 for a "replicated geometric" candidate that is among the 9 units in §0.6 |
| **candidate discovery under the redesigned tier 2 (post hoc)** | B3 + B4 for a candidate that is e64 "replicated geometric" (tiers 1 + 3) **and not among the 9 units in §0.6** |

**Counting.** Families are counted as in e64 §5.
- A family qualifies if any member reaches the top class **and no member is among the 9 units in
  §0.6**.
- Every unit row also carries its e64c matches: the |AUROC| ≥ 0.65 rules, and its §6.1-threshold
  hits.

**Relation to e64.** `e64_verdict.json` is not amended, and its wording stands: *no unnamed
occurrence variable was found at L7 or L12 by this instrument.* A "candidate discovery under the
redesigned tier 2 (post hoc)" is a lead for a fully blind confirmatory test (for example on
reserve 6,000–7,999, under a new prereg written before that block is touched). It is not a
discovery in e64's sense. **Whether it may ever be called one is the user's decision**
(§14, item 11).

**Every claim is reported with:**
- its per-unit bound;
- the layer's Σ;
- its realised dose;
- its σ ratio to e64's push;
- its identity-suspect flag.

## 12. Integrity

| file | sha256 |
|---|---|
| `scripts/e65_steering_redesign.py` (revision 2) | 5e37cab244e820337c7edd02f3f43ce0f30afa3fe4186e8444154a530aba5e86 |

**What e65 asserts at every data phase:**
- this file's sha256 is in `docs/HASHES.txt`;
- its own sha256 is in this text;
- every §2 input matches;
- `e64.check_integrity()` passes;
- swap used is ≤ 16 GB (the guard set by D1).

**Before each phase:** its predecessors' outputs are in `docs/HASHES.txt`.

**What each phase writes and appends to `docs/HASHES.txt`.** No phase writes any `e64_*` file.

| phase | writes and appends |
|---|---|
| freeze | `results/e65_rand_L7.npz`, `results/e65_rand_L12.npz`, `results/e65_frozen.json` |
| steer B3 | `results/e65_steer_B3.json` |
| steer B4 | `results/e65_steer_B4.json` (a not-run record if no layer qualifies) |
| verdict | `results/e65_verdict.json` |

## 13. Commands and resources (each step needs the user's go-ahead)

All commands run from `~/Downloads/gf-discovery-ext` with `HF_HOME=hf_home`, offline. Check
`sysctl vm.swapusage` first. At freeze time, swap was 0 GB used and 51 GB of disk was free.

```
.venv/bin/python scripts/e65_steering_redesign.py --phase freeze
.venv/bin/python scripts/e65_steering_redesign.py --phase steer --block B3
.venv/bin/python scripts/e65_steering_redesign.py --phase steer --block B4
.venv/bin/python scripts/e65_steering_redesign.py --phase verdict
```

**No extraction is needed.** The rig reads the kept D2 meta, the maps, the reserve tokens and the
model.

**Forward passes** (200 prompts each), with e64's measured cost of about 9.2 s per pass on MPS (857
s for 92 passes at L7):

| step | passes per layer | time |
|---|---|---|
| B3, anchor passes | L7: 1 + 99 + 4 × 42 + 23 = 291; L12: 1 + 99 + 4 × 55 + 23 = 343 | **about 1.6 h for both layers** |
| B3, anchor fails | L7: 108; L12: 116 | about 17–18 min per layer |
| B4, per qualifying layer | about 108–116, + 4 per B3 pass | about 17–19 min |
| freeze, verdict | — | CPU, under a few minutes each |

**Disk.** e65's outputs are small (under 50 MB). The folder is 8.1 GB at revision 2, under the
20 GB cap. The rise from 6.8 GB is the e64c session's `.venv-parse` (1.2 GB), not e65.

**Crash recovery.**
- **steer.** `steer` refuses to start if its block's output file already exists, and it dumps
  after each layer. If an MPS run is interrupted:
  1. A partial `results/e65_steer_B*.json` is left on disk, **unhashed**.
  2. Delete it and re-run the same command.

  HASHES is untouched by a partial run. The sites, the σ rule and the RAND banks were frozen by
  `freeze`, so the re-run is the same test. Report that a re-run happened.
- **freeze.** If `freeze` STOPs after hashing (a control-signature mismatch), its files stay listed
  in HASHES but are **void**. Report the mismatch; do not re-run it under this prereg.

## 14. Open choices (each has the default stated above)

1. **Dose grid and c\*.** {0.5, 1, 2, 4} with the gate at 2; sign consistency over {0.5, 1, 2}.
   Alternatives: gate at 1 or 4; drop c = 4.
2. **σ_d.** The raw SD over prompt positions ≥ 4 of the block. Alternatives:
   - the type-centred SD (reported anyway);
   - the SD of the unit's own score mapped to residual space, ‖Cov(h, ỹ_u)‖. This needs a D2
     re-extraction: the maps store only the normalised direction.
3. **RAND bank.** 99 draws, isotropic in whitened Z, pulled back, not orthogonalised, shared
   across units. Alternatives:
   - per-unit draws orthogonal to d_u (about 99 × N forward passes);
   - full-space random directions (e64's, a weaker null);
   - M = 199 (bound 0.005; about 15 min more per layer and block).
4. **SIG threshold.** The maximum (bound 1/(n′+1) ≈ 0.013–0.016). Alternative: the p95, which has
   more power if signatures share related tokens, but a bound of about 0.06 per unit, about
   2 expected false passes per layer on B3.
5. **SIG overlap handling.** Remove T_u's tokens from the others' T and R, keep v only if ≥ 5
   tokens remain, and exclude the twin. Alternative: exclude any v sharing ≥ 5 tokens with T_u.
6. **Sibling percentile.** Reported only. Alternative: add "above the sibling median" as a gate.
   Note that this ranks u among real effects; it is not a null.
7. **Which units are steered.** Every battery-UNNAMED unit (e64 §8.1). Alternative: only the
   17 + 17 candidates that replicated on R2. Only those can reach the top class, and this halves
   B3's multiplicity (Σ ≈ 0.27 / 0.22).
8. **B4 trigger.** The B3 anchor passes and there is ≥ 1 B3 pass. Alternative: always re-test the
   anchors on B4, to learn whether the rig replicates, at about 17 min per layer of reserve
   5,000–5,999.
9. **Sites.** 200 distinct sequences per block. More sites add power at linear cost.
10. **Blocks.** B3 = 4,000–4,999, B4 = 5,000–5,999; 6,000–7,999 untouched. Alternative: larger
    blocks. This doesn't matter at 200 sites.
11. **Wording.** Whether a B3 + B4 pass of a tier-1 + 3 family may ever be called a "discovery".
    Default: "candidate discovery under the redesigned tier 2 (post hoc)", with e64's verdict
    standing.
12. **dtype.** bf16, as in e64. Alternative: float32, which removes the bf16 rounding of small
    pushes. The realised dose is reported either way.
13. **Named non-control units at c\*.** Steered, for the sibling rank. Alternative: skip them and
    save about 3.5 min per layer.
14. **How e64c is used (§0.6).** Default: steer every candidate, count every candidate in Σ, and
    bar only the 9 units at |AUROC| ≥ 0.65 (counted by family) from the top class. Alternatives:
    - also bar the 17 §6.1-threshold hits;
    - drop the 9 from steering altogether. Σ then becomes 28/64 ≈ 0.44 at L7 and 28/77 ≈ 0.36
      at L12. This changes the frozen candidate set, which the brief kept;
    - follow REPORT's 8 and leave L7 logcosh #53 eligible;
    - ignore e64c.
