# PREREG — Is the confirmed L12 direction split across SAE latents in behaviour? (FROZEN rev 3, 2026-10-16)

**Status: frozen, revision 3** (after three rounds of adversarial review; §17).
- The user approved the freeze on 2026-10-16. This file, `scripts/e72f_fragmentation.py` and
  `docs/e72f_smoke_synthetic.py` are hashed in `docs/HASHES.txt` in that order.
- No block-F token had been read when this file was frozen.
- **The freeze authorises no run.** Block F, the 262k row fetch and the MPS run each still need the
  user's approval (§15).

**Freezing** means: write the script's sha256 into §14, then append this file's sha256 and the
script's to `docs/HASHES.txt` through `e64.append_hash`.

At every data phase, `scripts/e72f_fragmentation.py` asserts:
- that this file's sha256 is in `docs/HASHES.txt`, and that its own sha256 appears in this text
  (§14);
- that every §2 input matches its sha256;
- that e66's integrity check passes. That check covers e66's prereg, e65's prereg and inputs, and
  e64's chain.

**Nothing here amends any earlier verdict:**
- e64 "no discovery";
- e65 "candidate discovery under the redesigned tier 2 (post hoc)";
- e66 CONFIRMED;
- e67 COVERED;
- e68 gate FAILED;
- e69/e70;
- e71 descriptive.

SUMMARY.md and REPORT.md are not edited.

## 0. What was seen before freezing

**Read in full, everything up to and including e71:** REPORT, SUMMARY, every e64–e71 prereg, note
and result, and the sibling hand-over notes.

**For this direction specifically** (all on burned data):
- **e66 confirm (block C = reserve 6,000–6,999):** logcosh #13 Δ 1.49 (RAND max 0.75, SIG max
  0.85); skew #16 1.38 (0.72, 1.03); quote anchor logcosh #33 0.84 (0.48, 0.51).
- **e66 describe (block D = reserve 7,000–7,999), push +2σ_d.**
  - Top promoted tokens: *during* 2.62, *because* 2.60, *while* 2.56, *before* 2.53, *when* 2.37,
    *until* 2.29, *without* 2.29, *with* 2.24, *through* 2.17, *since* 2.16, *from* 2.11 nats, down
    to *in* 1.83 and *on* 1.79.
  - Category mass: preposition +0.016 (base 0.107), subordinator +0.009 (base 0.0135).
  - There is generic flattening too: "." +1.04 mean Δlog p while its Δp is −0.0046.
  - So T1 (§8.2) was expected to pass on the raw readout. **T1 is now judged on the corrected
    readout (§5), which nobody has computed, and it could fail where block D's raw numbers
    passed.**
- **e67 (D2).** COVERED by its decoder-cosine rubric:

  | latent | cos with #13 / #16 (masked) | activation r with the unit's D2 score | behavioural match share |
  |---|---|---|---|
  | `resid_post_all` L12 16k l0_big #11562 | **−0.735** / −0.722 | +0.19 | 0.485 |
  | `resid_post_all` L12 262k l0_small #2084 | +0.685 / +0.706 | +0.16 | 0.40 |
  | `resid_post` L13 16k l0_medium #183 | +0.680 / +0.709 | +0.375 | 0.41 |

  - **For #13, only #11562 clears 0.7.** #2084 and #183 clear it only for the twin #16.
  - #11562's |cos| of 0.735 lies **inside** e67's same-subspace null for 16k big (p99 0.756,
    p99.9 0.801).
  - e67's rubric also required a next-token context match, so the covering latents were not chosen
    blind to behaviour.
- **Contexts read.** I read the 12 strongest D2 contexts of #11562, #2084, #183, #320, #219, #226
  (16k big) and #2764 (16k small) in `e67_acts.json`.
  - #11562 and #320 fire on object heads before a preposition (*wrap the **board** in*, *tag your
    **photos** with*). The expected tilt from contexts is therefore toward P. No rule uses this.
  - #226 fires on particles and adverbs (*show **up** at*).
  - #183 fires on objects and pronouns (*smoke **them** because*).
  - #2084 fires on demonstrative-object heads (*open the **file** in*).
- **Decoder geometry computed while drafting** (CPU; decoder rows, frozen d and burned D2 only).
  - **The antipodal pair.** #11562 has negative cosine with d yet fires in d's heavy tail
    (r +0.19). Its masked row has cos **−0.928** with #320 (unmasked −0.968), and the two share
    their top contexts. #320 is the best positive-cosine 16k-big row (+0.644).
  - **Coordinate 1038** (the dropped massive coordinate) holds 55% of #11562's unmasked squared
    norm, 61% of #320's, 58% of #226's, 72% of #270's, 44% of #33's and 20% of #6428's. Across the
    dictionary, this share rises with |cos| to d. Its median is 0.027 at |cos| < 0.1 and 0.53 at
    0.4–0.5.
  - **R_big.** Non-negative OMP of +d over masked 16k-big rows, with e67's atoms
    [320, 226, 6428, 270, 33], reproduces e67's cos 0.807 at K = 5. The NNLS coefficients are
    0.387, 0.278, 0.198, 0.181 and 0.209.
  - **The quote anchor** has no single latent at |cos| ≥ 0.7. e68 calls it PARTIALLY COVERED, with
    best 0.634; at L12, 16k small #5117 reaches 0.614.
  - Every class word in §5 is a single token `" word"`.
- **Bank diagnostics on all D2 rows.** These use the frozen bank rule of §7 (computed in a scratch
  dry run that wrote nothing to the project). Each arm's percentile within its bank:

  | arm | corr(h·d, h·u) | SD(h·u)/SD(h·d) | shared-component projection | 1038 share |
  |---|---|---|---|---|
  | L_big | 0.30 | 0.19 | 0.20 | 0.59 |
  | #320 | 0.14 | 0.08 | 0.13 | 0.63 |
  | R_big | 0.97 | 0.85 | 0.84 | 0.74 |

  All are inside the bank's range. The pools hold 1,172 (L_big), 971 (#320) and 591 (R_big) rows.
- **Other D2 facts.**
  - The D2 next-token class counts are 714 S and 10,158 P rows.
  - Every S and P word gets a frequency-matched reference within ±25%.
- **Reviewer estimates (D2, linear d-push only).**
  - The d-component push at c\* is 0.59 of D's for L_big and 0.44 for #320; R_big ≈ 0.9.
  - PC_quote carries about 0.76 of the anchor's push.
- **Proximity (second review, D2).** On the S−P mean-difference axis u_md, the d-oriented off-d parts
  of near-d rows (|cos| 0.3–0.65) have mean projection +0.222, against +0.070 for the share-matched
  |cos| < 0.3 pool. So rows near d lean along the class axis more than far rows do.
  - The spread is wider than the shift: the pool's median |projection| is 0.106 and its p99 0.445.
  - L_big's |projection| sits at the 62nd percentile of the pool, #320's at the 53rd.
  - This is why §7 adds a near-d secondary bank (193 share-matched rows).
- **Planted-control geometry (second review, D2).** u_md separates S-next from P-next rows by 306,
  which is 1.08 SD of h·u_md. At L_big's α the full plant pushes 330 off d. u_md is a loud direction
  (SD ratio 0.93, against a bank median near 0.18), so the f = 1 plant is a near-maximal tilt. Hence
  the graded plants in §8.5.
- **e71:** #13/#16 and the quote controls read "sparser than chance" (blend-like); the paren
  controls read chance-like.

**Code run before freezing** (no block-F text, no model forward pass, no SAE activation outside D2):

| run | checks | what it covers |
|---|---|---|
| `e72f --phase selftest` | 25, all pass | the pure helpers, including the two-readout combination, the #320 agreement check and graded plants |
| `docs/e72f_smoke_synthetic.py` | 29, all pass | below |

The smoke test covers:
- the duplicate screen, sites and prompts on a fake corpus;
- `make_forward` on a tiny CPU torch model, which equals a direct computation;
- the HTTP-range row and block readers against the **local** 16k file via `file://`;
- the real steer engine and verdict on a fake linear model in ten constructed worlds, each returning
  its known headline: FRAGMENTED (with the 262k branch, and once with the near-d qualifier firing),
  NOT FRAGMENTED, TILTED, SPLIT NOT REASSEMBLED, NO VERDICT, UNINFORMATIVE, and INCONCLUSIVE for low
  power, a void band and disagreement;
- a bound check under both readouts: 3 of 300 exchangeable arms are flagged ONE-SIDED or TILTED,
  against a bound of 0.02. This is a code check only; it draws the arm from the bank's own pool, so
  it cannot show non-exchangeability. The §7 covariates address that on real rows.

**Full freeze dry run on a burned block** (scratch script `freeze_dry.py`). `freeze()` was run end
to end with block F replaced by **P2 (corpus 13,000–13,999, already used)**, the duplicate screen's
used list adjusted so P2 is not screened out, outputs written to the session scratchpad and
`append_hash` stubbed. `docs/HASHES.txt` was asserted unchanged.
- P2 kept 998 sequences after the screen (2 dropped).
- All four e67 cosine asserts reproduce.
- No covariate VOID; bank pools 1,172 / 971 / 591 / 193.
- The real frozen record then ran through the plan, the engine (fake forward) and the verdict: 548
  passes.

Block F was not read; its sites and screen are computed only at the real freeze.

**Never computed or seen by anyone:**
- any steering effect of any SAE decoder row, reconstruction, planted arm or bank member;
- any per-class (S vs P) split of any direction's effect;
- any statistic of corpus 8,000–8,999.

**What is not blind.**
- The direction was selected by e65 and confirmed by e66.
- Its raw two-class profile was seen on block D.
- The arms are fixed by e67's outputs and decoder geometry, not by steering.

## 1. Question and hypotheses

**Question.** d is the residual direction of ICA unit logcosh #13, causally confirmed by e66; it
promotes both kinds of adjunct opener. The question is whether the Gemma Scope 2 dictionaries'
**best single latent rows**, used as steering directions, carry only one of the two classes, while a
sparse reconstruction carries both.

| hypothesis | prediction on block F |
|---|---|
| **H_frag** | d moves S and P. The gated 16k-big row (L_big), oriented to d, moves one class and keeps less than half of what κ-matched perturbations keep of the other, beyond its cos- and share-matched bank under both readouts; #320 agrees; the 5-row reconstruction moves both. |
| **H_tilt** | the row leans toward one class beyond its bank but keeps at least half of what κ-matched perturbations keep of the other. |
| **H_single** | the rows move both classes like κ-matched perturbations of d. |
| **H_weak** | the rows are not potent at natural scale; nothing is decided. |

**Scope.** This tests decoder rows as **steering directions**, at the validated layer, masked at
coordinate 1038. It does not test encoder directions, what latents detect, or whether the SAE
"should" merge them. A finer SAE is not a "wrong" SAE.

## 2. Frozen inputs (asserted by sha256)

| file | sha256 |
|---|---|
| `scripts/e66_confirm_l12_family.py` (imported; it imports e65 and e64 with their own sha checks) | 21d47ae38b05c701c63f604ccc435ec5b60ac9f2b3c22f6ebd31b3034f2b3174 |
| `PREREG_occurrence_v2_confirm.md` | 59f1ae5b30d6f9c99ca5ca8473e58b5e445ae051e2782380e167e1cf851c87f4 |
| `results/e64_maps_L12.npz` | ac1b55b2eee03594b2d9f512632db2e9980e2446d051560d950ade8726200237 |
| `results/e65_frozen.json` | bd856844dae5db63d19518063d65eedf25ff8aaeee66565b09743f2da2fb2be9 |
| `results/e66_confirm.json` | e8ca659da4789ccaa54679544b7ba64997638bdb25c504c7f5786e3e4cfd3b7b |
| `results/e66_describe.json` (reference exclusions) | 7c760cceebd9f42badaefa68ee163635127e4250277c5f48fec6d6e50effcbe3 |
| `results/e67_decoder.json` | a5b76d05d32c991976e244d83b1de3fc631b3b565c0c489c535f632203ad2640 |
| `results/e67_acts.json` | d58c5b45138e1f2fbbf1a512cfe900181b133d1726deb6760724beac3f914dd1 |
| `results/e67_verdict.json` | dcebb7473d832a40751f7967e5787001345a7099c28680384977264b9a6013a2 |
| `results/e67_raw_L12_rows.npy` (D2 raw L12 rows) | 14d80c9661c71fcff0a87f89358d64957cdf68d94e63a6ca10200645152fa0b9 |
| `results/e68_decoder_G12.json` | fbc4a08d223fccff90d64c9f10d8f4b6432387253dd8e1e7f09fd0c20cfa369e |
| `results/e68_census.json` | b599ec0f5a4f6b84c2994f4bc9870ece100145eedabc186771ddbf273f1f39c8 |
| `results/occ_units_v2_L12.npz` (D2 kept rows) | b84a7b13d52bd54acdc665175c7972b4e9418659a3e6b56d47bdc3bb895b5d5b |
| `cache/occ/occD2_meta.npz` (D2 tokens) | bc910456116d5b875438c741c42379e24715576a2bea29e33b0dd9c1a92bc5a7 |
| `results/corpus_tokens_pass1.i32` | 8fbf98c3f29ebf9785885c1ac2df8ebeee7d0978c4663cd25517c5a925ba517e |
| `results/reserve_tokens.i32` (duplicate screen) | 3da028b79acf01f7b67e70eb602ebed37a8883b074f9dac9f08f6ccc2b79b5b5 |
| gemma-3-1b-pt `model.safetensors` (snapshot `fcf18a2a…`; planted unembedding arm) | ee5250f6eb1aa7cfb729dfd4dc8d9964fd772324776c6d00bf2bc674c069cb27 |
| Gemma Scope 2 `…/layer_12_width_16k_l0_big/params.safetensors` (snapshot `b738dc06…`) | e936d310077f7bf8b09c0f2602e2b62087b69b8c8f82ba1263cf9ed3bd4471ff |
| `…/layer_12_width_16k_l0_small/params.safetensors` | 6e1ae92a5a4d0956f1e88bd2e79e89e80f5998c76e5bab33a81a35a98488c96e |
| `…/resid_post/layer_13_width_16k_l0_medium/params.safetensors` | 2b57a8bfb01d73b5cb7d26b48d6d8f1ec5c2b1d8554488e5a81796a4b710829d |

**Unchanged from e65/e66:**
- the model and dtype: gemma-3-1b-pt, bf16, MPS;
- the hook: the output of decoder layer 12, at the last prompt position only, with h ← h + α·v
  computed in float32 and cast back;
- batch 32; `use_cache=False`;
- e65's σ rule, RAND bank and PASS rule.

## 3. Contexts: block F = main corpus 8,000–8,999

**Why the main corpus.** No untouched reserve is left. Reserve 0–7,999 is used:
- e61 v1 R and e62;
- R2 2,000–2,999;
- e64 3,000–3,999;
- e65 4,000–5,999;
- e66 6,000–7,999;
- plus e47r/e53 8-token slices.

**Verification that corpus 8,000–8,999 is unused for any test:**

| source | corpus blocks named |
|---|---|
| `PREREG_occurrence.md` (v1) | D and S 0–999; P 14,000–14,999 |
| `PREREG_occurrence_v2.md`, the stage-4v2 calibration plan, REPORT lines 686–688 | D2 1,000–1,999; P2 13,000–13,999 |
| e63, e67, e68, e69, e71 docstrings; e70 `USED_CORPUS` | D2 only; e70 records [(0, 2000), (13000, 15000)] as used |
| `scripts/e70_sae_gap_geometric.py` `SETS` (prereg **not frozen**, no text read) | earmarks 2,000–6,999 |
| grep of every ext and original-project script for 8000 / 8999 / 9000 | no corpus-row use |

Block F avoids e70's earmarked range.

**Whole-corpus exposure (disclosed). None of it concerns d, SAE latents or the S/P classes.**
- The Stage-0 passes (`e01_pass`) ran type-mean accumulators at L4–9 and L12–24, with caches since
  deleted.
- The corpus unigram counts that e62/e64/e65 use span every sequence, block F included.
- The original gf-discovery month, weekday and ring scripts took **per-position** cache rows or
  contexts across all 15,696 sequences: scripts 37b, 38b ("300 cached documents"), 39 ("all cache
  occurrences"), 41b, 47 and 49.
- Nothing in block F has been decoded, printed or scored by any Stage-4 or later code.
- The grep above cannot detect whole-corpus sampling; this list is from the reviewer's reading.

**Duplicate screen (e70's rule).**
- A block-F sequence is dropped if its first 32 tokens equal those of any sequence in corpus
  0–1,999 or 13,000–14,999, of any reserve sequence (BOS column removed), or of an earlier block-F
  sequence.
- The reviewer found 11 duplicate keys in 11,000 burned rows, so a few drops are expected.

**Layout.** `corpus_tokens_pass1.i32` is 15,696 × 255 with **BOS dropped**. Corpus and reserve come
from one document-start stream: truncated to 256, short documents skipped.
- **Sites:** 200 distinct screened sequences, one position each, p uniform in [4, 254] of the
  BOS-dropped row (= e65's [5, 255]). Seed `default_rng(7200 + 12)`.
- **Prompt:** BOS, then `row[max(0, p − 64) : p + 1]`. This is token-for-token e66's rule; the
  selftest checks it.
- **Freeze reads block F only as 32-token keys** (for the screen) and draws site indices. The
  prompts are built in `steer`.

## 4. Rig (e65/e66; contexts and readouts new)

| element | rule |
|---|---|
| dose | c·σ_v. σ_v = SD of the unsteered layer-12 residual projected on v over the block's prompt positions ≥ 4 (e65 `natural_sigma`). c ∈ {0.5, 1, 2, 4}; every gate at **c\* = 2**. **Exceptions (§7):** bank members and planted arms are dosed at their reference arm's α = c\*·σ_arm, not their own σ. |
| directions | unit norm, coordinate 1038 set to 0 (the e64/e67 convention: d and RAND are 0 there) |
| RAND | 99 directions from e65's `rand_bank`, seed 7210 + 12, each at its own σ at c\* |
| anchor gate | L12 logcosh #33 under e65's PASS rule on block F: Δ > 0; sign consistency ≥ 0.8 at some c ≤ 2; > RAND max; > SIG max. It is re-implemented from e65's primitives exactly as e66 did; e66's evaluate is a closure and cannot be imported. |

## 5. Readouts

**Class token sets** (from e66's word lists; each word is the single token `" word"`; the freeze
stores the ids):

| class | words | readout |
|---|---|---|
| **S, subordinators** (clause-introducing only) | when, while, because, if, though, although, whenever, unless, whereas | corrected (gated) |
| **P, prepositions** (NP-introducing only) | in, on, at, by, with, of, from, without, into, over, under, through, about, around, between, within, across, against, toward, towards, per | corrected (gated) |
| AMB (dual-use) | before, after, since, until, till, as, once, to, for | raw, descriptive |
| TMP | e66 `TEMPORAL` single-token words not in S, P or AMB (33 words, including during) | raw, descriptive |

**Frequency-matched references** (e65's rule, M5 of §17):
- Each S and P token gets one reference token. The reference's count is within ±10% of the class
  token's, widened to ±25% and then ±50% only if no candidate exists; it is drawn without
  replacement. Seeds 7240 (S) and 7241 (P).
- Counts are taken over the corpus **excluding block F**.
- **Excluded from references:** every class, AMB and TMP token; every token in e66 describe's
  top-30 promoted or demoted lists (by Δlog p and by Δp) at c = −2 and +2 (141 tokens); BOS, EOS
  and pad.
- The chosen references and the tolerance used are stored in the frozen JSON.
- **Pre-committed references.** These follow deterministically from counts excluding block F, the
  exclusions and the seeds; they were drawn while drafting:
  - **S:** when→" new", while→" New", because→" need", if→" her", though→" N", although→" dream",
    whenever→" reduced", unless→" COVID", whereas→" relatives".
  - **P:** in→"-" (±25%), on→" you", at→" (", by→" have", with→"’", of→" a" (±25%), from→" or",
    without→" number", into→" -", over→" make", under→" something", through→" many",
    about→" more", around→"S", between→" information", within→" four", across→" feel",
    against→" says", toward→" facility", towards→" regular", per→"or".
  - Punctuation, word-continuation pieces ("-", "’", "S", "or") and function words (" a", " or",
    " you") remain among the P references. That is the concrete reason every gated call also needs
    the raw readout.

**Readout.**
- **Per-site corrected class shift:**
  Δ_c = [lse(LP[c]) − lse(LP[R_c])]_steered − [same]_unsteered, where LP is the last-position
  log-softmax. The primary statistic is its mean over the 200 sites, at c\* unless stated.
- **Also reported:** the raw shift of log P_c, the reference's own shift (Δ_R_c), and the share of
  sites with Δ_c > 0.

**Profile statistic.**
- θ_D = atan2(Δ_P(D), Δ_S(D)), measured on block F.
- For any direction v, **τ(v) = −Δ_S(v)·sin θ_D + Δ_P(v)·cos θ_D**: the perpendicular displacement,
  in nats, of v's (Δ_S, Δ_P) from D's profile direction. τ > 0 leans toward P; τ < 0 toward S.
- τ replaces the earlier angle. A bank member that moves neither class no longer voids the band.

**Both readouts (second review, N2).** For the top prepositions (*of*, *in*, *on*, *with*) the only
frequency-matched references are function words such as ' a', ' you' and ' or'. No open-class word
lies within ±50%, and a few of those function words sit in the same decision.

So every gated call uses **both** readouts: the corrected one above, and the raw shift of log P_c,
which has its own bias (generic flattening favours the rare class S). The two are combined
conservatively (§8.3). A call that must hold under both cannot ride on either readout's own bias, and
every bound still holds. T1 also needs both.

**Also reported:**
- per-word Δ log p for every S, P, AMB and TMP word, and for every reference token;
- D's e65 readout on its own signature with e65's gates (a replication on block F, not a gate).

## 6. Directions (arms)

All SAE rows are masked at 1038 and renormalised. **"d-oriented"** = multiplied by
sign(cos(row, d)). Nothing is chosen by behaviour on block F.

**Gated, 16k big** (`resid_post_all` L12 16k l0_big, local):

| arm | definition | cos with d | role |
|---|---|---|---|
| **D** | d = logcosh #13 | 1 | T1; profile direction |
| **L_big** | #11562, d-oriented (= −row; no non-negative code can write it) | 0.735 | **the gated row** (e67's covering latent) |
| **x320** | #320, d-oriented (= +row; writable) | 0.644 | **agreement check** (best positive row; first non-negative OMP atom) |
| **R_big** | NNLS reconstruction of +d from e67's atoms [320, 226, 6428, 270, 33], unit norm | 0.807 (asserted ±0.001) | reassembly check |

**L_big and x320 are nearly one direction** (cos 0.928, an antipodal co-firing pair; Sun et al.
2025). x320 is an agreement check, not a second instance.
- It is not a co-primary, because its d-push is only 0.44 of D's: its bank-median precondition may
  fail and would sink a clean L_big result (second review, N3).
- Disagreement makes the verdict INCONCLUSIVE; an x320 precondition failure is only reported.

**Planted positive controls** (§8.5), graded. Each is pushed at **L_big's α** and judged against
**L_big's bank**. With κ = 0.735 and s = √(1 − κ²):

  Π_{u,f}± = κ·d ± s·w_f,  w_f = unit(√f·u + √(1 − f)·u_fill⊥),  f ∈ {1, 0.5, 0.25}.

- **u = u_md**: the off-d unit part of the D2 residual mean difference between rows whose next token
  is in S and rows whose next token is in P (all kept D2 rows).
- **u = u_wu**: the off-d unit part of mean_S − mean_P of (1 + final-norm weight) ⊙ embedding row
  (tied unembedding). This separates "the band is too wide" from "the mean difference is a poor
  steering vector".
- **u_fill**: the off-d part of one unused, d-oriented row of L_big's own pool (seed 7250), made
  orthogonal to u and d.
- In every case + leans toward S. That gives 12 arms.

**Conditional, 262k small** (secondary replication; only if the row fetch is approved and succeeds;
§11):

| arm | definition | cos with d |
|---|---|---|
| **L_262** | #2084, d-oriented (= +row) | 0.685 (asserted ±1e-4 against e67, for #13 and #16) |
| **R_262** | NNLS from e67's atoms [2084, 7872, 5346, 3714, 77508] | 0.829 (asserted ±0.001 against e67 `cos_at[5]`) |

**Descriptive only** (never gated; doses in brackets):

| arm | why |
|---|---|
| D16 = skew #16 [all] | the twin |
| L_big native = +row #11562 [all] | what the SAE can write. **Predicted: demotes both classes** (cos −0.735). |
| PC_quote = 16k small #5117, oriented to #33 [all], on the anchor's signature | the briefed positive control. Demoted to descriptive: its pass is expected on geometry (≈ 0.76 of the anchor's push), and it shares no dictionary, readout or unit with L_big. |
| R_big's other atoms 226, 6428, 270, 33 [c\*] | do the blend's components divide by class? |
| L13 16k l0_medium #183, d-oriented, **pushed at the L12 hook** [c\*] | e67's third covering latent (for #16), at an unvalidated layer; exploratory |
| 16k small #2764, d-oriented [c\*] | best 16k-small row for d (cos −0.607) |
| R_big at c\*·σ_D [c\*] | the reconstruction at d's own dose |
| signed OMP K = 3 of 16k big, [11562, 219, 226] [c\*] | e67's signed reconstruction (asserted against e67's cos 0.8076) |
| R_262's other atoms [c\*], if fetched | as for R_big |

**Recorded at freeze.**
- For every single-row SAE arm: the raw decoder cosine with d, the orientation sign, the unmasked
  1038 share and the masked row norm. For PC_quote these are taken against the anchor.
- For 16k and L13 rows, the raw encoder cosine too. The 262k encoder is never fetched.
- For L12 16k latents, the D2 activation frequency, and the mean and p90 activation when active.
  L13 has no stored D2 rows.
- For reconstructions, the share of Σ cᵢ·rawᵢ, with cᵢ the NNLS weights on the masked unit rows.

## 7. Nulls

**RAND (per class).** "Potent in class c" = Δ_c(v) > the max over the 99 RAND passes of Δ_c.

**Banks: one per banked arm** (L_big, x320, R_big; L_262, R_262 if fetched). M = 99 directions

  b_j = κ·d + √(1 − κ²)·offd(r_j),  offd(r) = unit(r − (r·d)·d), κ = cos(arm, d),

where r_j is a **d-oriented**, masked, renormalised decoder row of the arm's own dictionary.

**The pool** (positions drawn without replacement, seeds 7220–7224) is every row that is:
- alive;
- |cos(r, d)| < 0.3;
- **|share_1038(r) − share_1038(arm)| ≤ 0.15**;
- not an arm, an atom, #219 or a descriptive latent.

"Alive" is defined per dictionary:
- 16k big: active on ≥ 1 D2 kept row. This removes about 110 rows, so it is nearly vacuous.
- 262k small: shipped `feature_frequencies` ≥ 1e-6.

**Every b_j is dosed at the arm's α = c\*·σ_arm** (an exception to §4's own-σ rule). Each b_j has cos
κ with d exactly.

**Near-d secondary bank for L_big (second review, N1).**
- The same construction, with the pool restricted to rows with |cos(r, d)| ∈ [0.3, 0.65], share
  matched and alive: 193 rows, M = 99, seed 7225, dosed at L_big's α.
- It does not change the headline category.
- The headline carries the qualifier "the tilt is shared by d's dictionary neighbourhood" **only if
  τ(L_big) lies inside the near-d band under both readouts**.
- If the near-d check is otherwise INCONCLUSIVE, "near-d check inconclusive" is reported beside the
  headline (round 3, MAJOR 1).
- The headline's null is therefore stated as "share-matched latents far from d" (|cos| < 0.3).

**Why these choices.** The first draft's bank (any 1038 share, random signs, each member at its own
σ) was not comparable with the arm. It differed on 1038 share, on correlation with d and on dose
(§17, B1). Matching the share, orienting rows to d and matching the dose put each L arm inside its
bank on every recorded covariate (§0).

**Covariate diagnostics** (freeze, all D2 kept rows). For each gated arm and each bank member, on
the off-d unit vector u:
- corr(h·d, h·u);
- SD(h·u)/SD(h·d);
- the projection on the dictionary's **shared component**: the off-d unit mean of the off-d unit
  parts of the d-oriented rows with share ≥ 0.5 and |cos| < 0.3. For 262k it uses the fetched pool
  rows only;
- the 1038 share.

**Void rule.**
- The **gated row** (L_big; L_262) is VOID, and its band is not used, if it lies outside its bank's
  [min, max] on any of these.
- If x320 lies outside, its agreement check is "not assessable" and does not block.
- For R arms (R_big sits at the 97th percentile on corr) and the near-d bank, these numbers are a
  caveat only: a single-row bank is an approximate null for a 5-row blend.
- Under true exchangeability a [min, max] test on 4 covariates would give about 8% spurious VOIDs.
  The dry run (§0) already shows L_big inside, so the rule only guards what §0 discloses.

**Calibration on real data.** Each bank member is classified against the other 98 by §8.3. The rates
of ONE-SIDED, TILTED, TWO-SIDED and INCONCLUSIVE are reported.

**Bound.** Under exchangeability of the arm with its bank, P(τ below the band) and P(τ above it)
are each ≤ 1/(M + 1) = 0.01.

## 8. Gates, classes and checks

**8.1 Anchor gate.** The anchor PASSes e65's rule on block F. If not: NO VERDICT, and nothing else is
steered.

**8.2 T1, premise:** Δ_S(D) > RAND max of Δ_S **and** Δ_P(D) > RAND max of Δ_P, under **both**
readouts.

**8.3 Classes of a gated arm A**, at c\*, against its bank. Notation:
- band = [min_j τ(b_j), max_j τ(b_j)];
- med_c = the bank's median Δ_c;
- floor_c = 0.5·med_c.

**Precondition:** if med_S or med_P is not itself > the RAND max for that class, A is INCONCLUSIVE.
The κ-matched perturbations would then not carry the effect.

| class | requires (first match wins) |
|---|---|
| **S-ONE-SIDED** | τ(A) < band min **and** S potent **and** Δ_P(A) < floor_P |
| **S-TILTED** | τ(A) < band min **and** S potent **and** Δ_P(A) ≥ floor_P |
| **P-ONE-SIDED** / **P-TILTED** | the mirror image (τ > band max) |
| **TWO-SIDED** | τ inside the band **and** both classes potent **and** Δ_S ≥ floor_S **and** Δ_P ≥ floor_P |
| **INCONCLUSIVE** | anything else. The failing conditions are listed mechanically. |
| **VOID** | an L arm failing the §7 covariate rule |

The floors are relative to the bank. A row is judged against what a κ- and dose-matched
perturbation of d keeps, not against d itself (§17, M2).

**Two readouts combined (conservative).** The table is applied separately to the corrected and the
raw readout, each with its own θ_D, band, medians and RAND max.
- The same class under both stands.
- ONE-SIDED under one and TILTED under the other, on the same side, gives TILTED.
- Anything else is INCONCLUSIVE ("readouts disagree").

So ONE-SIDED and TWO-SIDED need both readouts, and TILTED needs both on one side.

**x320 agreement** (with L_big's combined class):
- for a one-sided or tilted L_big: x320 is not one-sided or tilted to the opposite side, **and**
  τ(x320) lies on L_big's side of x320's own bank-median τ under both readouts;
- for a TWO-SIDED L_big: x320 is not one-sided or tilted;
- an INCONCLUSIVE x320 (e.g. its precondition) does not by itself disagree.

**8.4 Reassembly check (T3):** R_g is TWO-SIDED against its own bank.
- It is **a consistency check expected to pass on geometry**: R_big's d-component push is about
  0.9 of D's.
- It is a condition for FRAGMENTED. If it fails, the "blend" reading is not supported in behaviour.
- If R fails only through its bank-median precondition, the dictionary verdict is INCONCLUSIVE, not
  SPLIT.
- It is not evidence of splitting.
- Descriptively, τ(R) is reported beside τ(L_big) and τ(x320).

**8.5 Planted positive control.**
- A planted pair (u, f) **passes** if its + member lands below L_big's band and its − member above
  it, under both readouts.
- **PC PASS** = at least one of u_md and u_wu passes at **f = 0.5**. This shows the band detects a
  half-strength class tilt at κ = 0.735 and L_big's dose. Detection at f = 1 and f = 0.25 is
  reported.
- The same planted result gates the 262k NOT FRAGMENTED, although it was measured on L_big's band
  and dose.
- **Caveat:** at f = 0.5 each plant contains one random filler row. That row's own tilt adds to both
  members, so PASS at half strength partly depends on which filler was drawn. It does not affect
  validity.
- **PC gates only NOT FRAGMENTED.** Without power, "inside the band" means little, so a TWO-SIDED
  gated row with PC FAIL gives INCONCLUSIVE.
- **PC does not gate FRAGMENTED or TILTED.** Their bound holds whatever the power.

## 9. Verdict (first match wins; stated before any block-F data)

**Per dictionary.** The gated row is L_big for 16k big and L_262 for 262k small. x320 is an
agreement check for 16k big only.

| dictionary verdict | requires |
|---|---|
| **INCONCLUSIVE** | the gated row VOID, or x320 disagrees |
| **NOT FRAGMENTED** | the gated row TWO-SIDED **and** PC PASS (with PC FAIL: INCONCLUSIVE) |
| **FRAGMENTED** | the gated row ONE-SIDED **and** T3 |
| **SPLIT, NOT REASSEMBLED** | the gated row ONE-SIDED, T3 failed, and not only through R's precondition (that case: INCONCLUSIVE) |
| **TILTED** | the gated row TILTED |
| **INCONCLUSIVE** | otherwise |

**Headline:**

| outcome | requires | wording |
|---|---|---|
| **NO VERDICT** | anchor fails on F | "The rig failed its anchor on block F; no verdict." |
| **UNINFORMATIVE** | anchor passes; T1 fails | "d did not move both classes on block F (corrected readout), so the question does not arise there." |
| **FRAGMENTED** | 16k-big FRAGMENTED | "On fresh contexts, d promotes both subordinators and prepositions. Its best 16k-big decoder row, used as a steering direction (#11562, masked at coordinate 1038 and sign-flipped; its writable antipodal partner #320 agrees), promotes mainly [S/P]. Under both readouts it keeps less than half of what 99 share- and dose-matched perturbations of d at the same cosine keep of the other class, against latents far from d (|cos| < 0.3). A 5-row reconstruction from the same dictionary moves both classes, as geometry predicts. The label names this pattern, one row carrying one class and several rows carrying both, not a mechanism; it is not a new concept (§12)." Then, descriptively and with no null: "the reconstruction's atoms [do / do not] point both ways". This holds if, under both readouts, at least one atom is potent and S-leaning by τ sign and one is potent and P-leaning. It is never a claim of a split along S/P. The near-d qualifier is added if it applies. |
| **SPLIT, NOT REASSEMBLED** | 16k-big SPLIT, NOT REASSEMBLED | "The best rows are one-sided, but the reconstruction does not restore the two-sided effect. 'Blend' is not supported in behaviour; reconstructions can lose steering effect for unrelated reasons (Kharlapenko et al. 2024)." |
| **TILTED** | 16k-big TILTED | "The best row leans toward [S/P] beyond chance but keeps at least half of what κ-matched perturbations of d keep of the other class. A tilt, not a split." The near-d qualifier is added if it applies. |
| **NOT FRAGMENTED** | 16k-big NOT FRAGMENTED | "The best 16k-big row moves both classes as κ-matched perturbations of d do. It shows no tilt as large as the half-strength planted control's, so e67's 'blend' reading is not supported in behaviour at that sensitivity. A hedged latent (Chanin et al. 2025) would also give this." |
| **INCONCLUSIVE** | otherwise | the failing conditions, the VOID flags and the PC result |

**Reported beside the headline, never changing it:**
- the 262k-small verdict (or "not tested: reason") and whether it agrees;
- L_big's class under the near-d bank;
- x320's class and agreement;
- the leave-one-out bank rates, under each readout;
- every planted pair's τ values against the band, at each f;
- PC_quote's ratio to the anchor;
- R_big's atoms (and x320), classed S- or P-leaning by τ sign plus potency under both readouts.
  This is descriptive with no null; five atoms will often point both ways by chance;
- L_big native against its prediction;
- every arm's raw and corrected shifts, Δ_R_c, per-word table, σ, realised dose, covariates, 1038
  share, encoder and decoder cosines, and D2 activation stats.

**False-pass bound.** Under the null that L_big is exchangeable with its (far, share-matched) bank,
FRAGMENTED or TILTED requires τ(L_big) outside the band under both readouts: ≤ 0.02. x320 and the
second readout can only remove calls. The headline uses one dictionary and one row, so there is no
multiplicity.

**Not amended:** e64–e71. Whether any result is written up as a finding is the user's decision.

## 10. Order of work and cost

| phase | where | does |
|---|---|---|
| `selftest` | CPU, no data | 25 synthetic checks |
| `freeze` | CPU, plus HTTP range only with `--with-262k` | **262k first** (fetched and asserted in memory; any failure → recorded "skipped: reason", freeze continues); class ids; references; duplicate screen (block-F 32-token keys only) and sites; 16k rows and asserts; D2 encodes (16k big and small, about 6 s each); RAND; banks (including the near-d bank); graded planted directions; then the 262k arms, banks and shared component, all-or-nothing; covariates. Writes and hashes `e72f_rows_262k_small.npz` (if any), `e72f_dirs.npz` and `e72f_frozen.json`, in that order, **before any forward**. |
| `steer` | MPS | unsteered pass (all σ) → RAND → anchor (stop if it fails) → D, D16 → L_big + bank → x320 + bank → R_big + bank → [L_262 + bank → R_262 + bank] → near-d bank → 12 planted → PC_quote → descriptive. Dumps after each group; swap guard every 20 passes; hashes `e72f_steer.json` at the end. |
| `verdict` | CPU | §8–§9 mechanically; hashes `e72f_verdict.json` |

**Cost** (e66's rate, about 8 s per 200-prompt pass):
- 16k path: 1 + 99 RAND + 4 anchor + 8 D/D16 + 3 × (4 + 99) + 99 near-d + 12 planted + 4
  PC_quote + 12 descriptive ≈ **548 passes ≈ 73 min**. If the anchor fails, about 104 passes ≈
  14 min.
- 262k path: + 210 passes ≈ 28 min.

## 11. Dictionary rows (the deleted 262k dictionaries)

Only `resid_post_all/layer_12_width_262k_l0_small` is needed, read at the **pinned revision
`b738dc06…`** (the local Gemma Scope 2 snapshot).

| item | size |
|---|---|
| the safetensors headers of `params` and `examples` | small |
| `feature_frequencies` (262,144 floats) | ~1 MB |
| the w_dec rows of #2084, 7872, 5346, 3714 and 77508 | 5 requests |
| **96 random non-overlapping blocks of 64 consecutive rows** (6,144 rows; block starts from `default_rng(7230)`) | 96 requests, ~28 MB |

**Bank selection.**
- The pool is the alive, non-atom rows in the blocks (about 6,100).
- Per bank, the share-matched, |cos| < 0.3 rows are eligible, and 99 are drawn by seed 7223 or 7224.
  The two banks may share rows.
- At the second review's eligible rates (4–8%), 6,144 rows give about 250–500 eligible.
- Fewer than 99 eligible → 262k skipped, with the reason recorded.

**Default (recommended): HTTP range reads, no file download.**
- Script 27's wall-clock-bounded curl reader, coalesced into blocks, with a byte-count check on
  every range. One 900 s deadline bounds the whole fetch. Total ≈ 30 MB; nothing is cached by
  `huggingface_hub`.
- Rows go to `results/e72f_rows_262k_small.npz` (hashed at freeze).
- **Integrity:** #2084's masked cos with #13 and #16 must equal e67's (±1e-4), and R_262's cos must
  equal e67's `cos_at[5]` (±0.001).
- **Any failure skips the 262k arms and nothing else.** This covers the fetch, the asserts, the arms,
  the banks and the shared component, which are built locally and merged only if all succeed. The
  16k headline needs no network.

**Alternative: full download** (2.42 GB) with `HF_HOME=<ext>/hf_home`.
- Folder 13 → 15.4 GB (cap 20 GB); free disk 12–16 GB → about 10–13 GB, against BUDGET's 8 GB
  abort guard. The blob would be deleted after extraction.
- Not recommended: same rows, about 80× the bytes.

## 12. Prior work and honest novelty

Checked 2026-10-16 by a web search (notes with URLs: `lit_fragmentation.md` in the session notes
folder). † = found and opened by the literature agent, not re-checked by me.

**The concept is not new.**
- **Feature splitting:** Bricken et al. 2023, "Towards Monosemanticity"; Templeton et al. 2024,
  "Scaling Monosemanticity". Split siblings are compared by weights, not steered against a parent
  direction.
- **Absorption:** Chanin et al. 2024, "A is for Absorption" (arXiv:2409.14507). Encoder and decoder
  behave asymmetrically. It predicts token-specific covers, not a class-level split.
- **Non-atomic latents, no canonical width:** Bussmann et al. 2024, meta-SAEs (LessWrong/AF); Leask
  et al. 2025 (arXiv:2502.04878, ICLR 2025).
- **"Fragmentation":** Nabeshima 2024, "Matryoshka Sparse Autoencoders" (AF). Its "words in quotes"
  example warns that the quote control may itself fragment. Bussmann et al. 2025
  (arXiv:2503.17547) show Matryoshka training reduces absorption. The Gemma Scope 2 technical
  paper† describes a Matryoshka loss; the local `config.json` records only `jump_relu`, so this is
  stated with that caveat.
- **Hedging:** Chanin et al. 2025 (arXiv:2505.11756). Narrow SAEs merge correlated features. A
  hedged latent would read TWO-SIDED here.
- **Antipodal pairs:** Sun et al. 2025 (arXiv:2506.15679) report antipodal latent pairs that encode
  one residual direction. #320/#11562 look like a sparse instance.

**Steering with SAE latents.**
- Arad, Mueller & Belinkov 2025 (arXiv:2505.20063): mid-depth latents are weak input and output
  features, so H_weak is real.
- Wu et al. 2025, AxBench (arXiv:2501.17148): dense baselines beat SAE steering.
- Mayne et al. 2024 (arXiv:2411.08790): steering vectors carry negative projections that a
  non-negative code cannot express. Hence L_big is −row, x320 is +row, and the signed
  reconstruction is reported.
- Kharlapenko et al. 2024 (AF): SAE reconstructions can lose steering effect.

**Closest prior work.**
- Kissane et al. 2024 (LessWrong/AF) reconstruct a refusal direction from decoder rows and compare
  single-latent and direction ablations, on one behaviour.
- Hiramatsu et al. 2026† (arXiv:2609.07037) split a steering vector into components with distinct
  effects, using a purpose-trained SAE.
- Bondielli et al. 2026† (arXiv:2609.29362) find subordinators and prepositions sharing SAE latents,
  correlationally and via shared words. This is the lexical confound that §5's disjoint sets remove.
- No prior work found shows "each best covering row moves one output class; the reconstruction moves
  both".

**Honest novelty.**
- A FRAGMENTED result would be one behaviourally confirmed instance, at 1B, in a released Gemma
  Scope 2 suite: the best decoder rows for one causally confirmed residual direction carry only one
  of its two output classes as steering vectors, and a sparse reconstruction carries both.
- It is a data point on an open question, not a new concept.
- It is claimed for these dictionaries, masked at coordinate 1038, and not for the model. Latents
  vary across seeds and widths.
- Must cite: Chanin et al. 2024, Leask et al. 2025, Kissane et al. 2024, Nabeshima 2024, Sun et al.
  2025.

## 13. Constraints

**Before any MPS phase:**
- `ps` shows no other e6x/e7x process (checked by hand);
- swap ≤ **12 GB**: the script's guard at phase start and every 20 passes, stricter than e64's 16;
- free disk ≥ 8 GB (the verdict phase checks swap only).

**Launch:** detached under `caffeinate -i`, in its own session, logging to `logs/e72f_<phase>.log`.
The `caffeinate` and the detached launch are done by the operator, not the script.

**One launch per phase, no retries:**
- If `steer` aborts (guard, crash), its partial JSON stays unhashed and is reported. It is not re-run
  without the user's go-ahead. The sites, banks and directions are frozen, so a re-run would be the
  same test.
- `freeze` asserts everything before writing anything. A failed 16k assert writes nothing; report
  it, and do not edit to make it pass. A 262k failure is recorded and skipped.

**HASHES** is written only by the phases' own `append_hash` calls, plus the registration of this file
and the script.

## 14. Integrity

| file | sha256 |
|---|---|
| `scripts/e72f_fragmentation.py` | 825ac723e33c724b7731a62518f06dbea3e366b32802d44d4190bb4193de6014 |
| `docs/e72f_smoke_synthetic.py` | 7c7a297a9c060901925b2bbf41fe3df8881d471658cf91deb2e0bb164a5b1a87 (hashed with the script, as for e68–e70) |

**Freeze procedure** (order matters: `check_integrity` requires the script's sha to appear in this
file):
1. Finalise the script and the smoke test, and run both: `--phase selftest` and the smoke test.
2. Write both sha256 values into the table above.
3. Hash, in this order, through `e64.append_hash`: this file, the script, the smoke test.
4. Only then run `--phase freeze`.

**Outputs**, each appended to `docs/HASHES.txt` by its phase:
- `results/e72f_rows_262k_small.npz` (if fetched);
- `results/e72f_dirs.npz`;
- `results/e72f_frozen.json`;
- `results/e72f_steer.json`;
- `results/e72f_verdict.json`.

**Commands** (from `~/Downloads/gf-discovery-ext`, `HF_HOME=hf_home`; offline except the range
reads):

```
.venv/bin/python scripts/e72f_fragmentation.py --phase selftest
.venv/bin/python docs/e72f_smoke_synthetic.py
.venv/bin/python scripts/e72f_fragmentation.py --phase freeze [--with-262k]
caffeinate -i .venv/bin/python scripts/e72f_fragmentation.py --phase steer   # detached, see §13
.venv/bin/python scripts/e72f_fragmentation.py --phase verdict
```

## 15. Open choices (defaults in bold)

1. **Block F:** **corpus 8,000–8,999**. Any screened 1,000-sequence block in 7,000–12,999 works.
2. **262k rows:** **range reads**, full download, or skip.
3. **Bank size M:** **99**.
4. **Bank pool:** **far (|cos| < 0.3), share ±0.15, d-oriented, dose at the arm's α**, with the
   near-d bank (|cos| 0.3–0.65, 193 rows) as a secondary that only qualifies the wording. The
   alternative is to require both banks.
5. **Classes:** **S and P without dual-use words**.
6. **Floors:** **0.5 × bank median**.
7. **Gated row:** **#11562 (e67's covering latent), with #320 as an agreement check**; or #320
   gated with its d-push matched to L_big's.
8. **Planted controls:** **mean-difference and unembedding plants at f ∈ {1, 0.5, 0.25}; PC PASS if
   either passes at f = 0.5**.
9. **Readouts:** **both required, combined conservatively**; or corrected only, with stricter
   references.
10. **Sites:** **200**.
11. **L13 #183:** **exploratory at the L12 hook**.

**Changing any default** means editing the script and this text, both before freezing.

## 16. What a result would and would not show

- **FRAGMENTED** shows that, for one confirmed direction, the best 16k-big decoder row (and its
  antipodal partner) is a one-class steering vector, beyond far share-matched latents, and a few rows
  together are a two-class one. Whether the atoms divide by class is reported descriptively; no
  split along S/P is claimed. It does not show that the
  SAE is wrong, that the model has separate S and P directions, or anything about other units.
- **NOT FRAGMENTED** shows no tilt as large as the half-strength planted control's. It does not
  distinguish a faithful latent from a hedged one.
- **TILTED** shows a class preference without a split.
- **SPLIT, NOT REASSEMBLED** has a mundane reading (Kharlapenko et al. 2024).
- No outcome bears on whether any e64 unit is "uncovered" (SUMMARY §4). None licenses the 4B
  scale-up.
- Every outcome is claimed for these Gemma Scope 2 dictionaries, as masked steering directions,
  only.

## 17. Review log (2026-10-16: two adversarial reviewers, two rounds)

### Round 1: reviews of revision 1

**Design review.**

| finding | fix |
|---|---|
| **B1.** The bank was not exchangeable with the arm: different 1038 share, a shared 1038 component, correlation with d, dose. | §7 share-matched, d-oriented pool, dose at the arm's α, covariate void rule. Verified on D2 (§0). |
| **B2.** ONE-SIDED did not mean "mainly one class". | §8.3 adds the other-class floor and a separate TILTED class, which cannot yield FRAGMENTED. |
| M1. The antipodal pair. | x320 gets its own bank. In round 1 it was a co-primary; round 2 demoted it to an agreement check (N3). |
| M2. The 0.5 × D floor was set by geometry. | Floors are now relative to the bank median. |
| M3. The PC was disconnected and likely to pass. | Planted-tilt controls gate NOT FRAGMENTED; PC_quote is descriptive. |
| M4. T3 was near-automatic. | Relabelled as a consistency check, and the headline wording is fixed. |
| M5. Generic flattening. | Frequency-matched references, with exclusions from e66 describe. |
| M6. The angle band was fragile. | τ replaces it, with leave-one-out calibration. |
| M7. Overclaiming. | §1, §9, §12 and §16 reworded (an ICA direction; masked and sign-flipped; the subspace null; hedging). |
| Minors. | §3 per-position disclosure; duplicate screen; §0 coverage wording; §11 matches the code; Matryoshka hedged. |

**Code review.** It found no blocker on the 16k path. All freeze asserts were reproduced by hand.

| finding | fix |
|---|---|
| M1. A 262k failure could kill the 16k test. | Fetch and assert in memory first; skip on failure; nothing is written before the 16k asserts pass. |
| M2. Encoder-cosine sign. | Raw decoder and encoder cosines and the orientation are stored; PC is measured against a33. |
| M3. The §11 pool did not match the code. | Rewritten. |
| Minors. | `share_1038`, masked norms and encoder cosines for every SAE arm; byte-count and shape checks; BF16 support; a pinned revision; the signed-K3 assert; RAND and bank σ and realised dose recorded; swap guard every 20 passes; the verdict lists failing conditions; the anchor gate described as re-implemented; the smoke test now covers the range reader, verdict and all classes. |

### Round 2: re-reviews of revision 2

**Design re-review.** Rev 2 closes B1 on the measured covariates and B2 in the rules. No blocker.

| finding | fix |
|---|---|
| N1. Proximity: near-d rows lean along the class axis more than far rows, and the covariates do not measure that. | Near-d secondary bank (§7); the null is stated as "far" latents; a qualifier if the side differs; u_md numbers in §0. |
| N2. The P references are entangled function tokens (' a', ' or', ' you'). | Every gated call and T1 must hold under both readouts, combined conservatively (§5, §8.3); each reference's own shift is reported. Cleaner references are infeasible: no open-class word lies within ±50% of *of* or *in*. |
| N3. The x320 precondition would likely sink a clean L_big result. | x320 is demoted to an agreement check (§6, §8.3). |
| N4. The planted control showed power only at near-maximal tilt. | Graded plants f ∈ {1, 0.5, 0.25}; NOT FRAGMENTED is gated at f = 0.5 and reworded. |
| N5. Wording. | §9 and §16 rewritten: "keeps less than half of what κ-matched perturbations keep"; "its best row (and that row's antipodal partner)"; the S/P atom clause was made conditional, and round 3 made it descriptive only. |
| N6. Minors. | SPLIT → INCONCLUSIVE if R fails only its precondition; the spurious-void rate is disclosed; R's covariate caveat added. |

**Code re-review.** No blocker and no major. Confirmed:
- the 262k skip path writes nothing;
- the frozen record serialises;
- `verdict_of` reads only keys the engine writes;
- the planted unembedding construction matches the installed Gemma3RMSNorm (1 + weight, tied
  embeddings);
- the D2 row alignment.

| minor | fix |
|---|---|
| The R_262 bank could come up short. | The pool is now 96 blocks × 64 rows. |
| 2,009 separate requests and no overall deadline. | Block reads and a 900 s deadline. |
| Some post-fetch 262k failures were uncaught. | The arms, banks and shared component are built locally and merged all-or-nothing. |
| §6/§7 metadata text overstated. | Narrowed. |
| The R share formula was unstated. | Stated in §6. |
| The freeze read the whole corpus. | It now uses a memmap. |
| PC reuse for 262k. | Acknowledged in §8.5. |
| Shared-component definition. | Stated in §7. |
| Stale 262k asserts after a skip. | Removed and labelled. |

### Round 3: blocker-only check of the round-2 fixes

No blocker. It confirmed that N2, N3, N4 and N6 are implemented as stated and that the conjunction
keeps the bound. Two majors were fixed:

| finding | fix |
|---|---|
| The near-d qualifier fired on a merely inconclusive near-d check. | It now fires only if τ(L_big) lies inside the near-d band under both readouts (§7). A smoke world checks the positive case. |
| "Fragmentation along S/P" rested on an atom check with no null and one readout. | The atom check now needs both readouts, is labelled descriptive with no null, and no S/P split is claimed (§9, §16). |

The single-filler caveat (untagged) is recorded in §8.5.

