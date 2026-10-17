# PREREG — Stage 4 v2: confirmation of the L12 logcosh #13 / skew #16 candidate on untouched contexts (FROZEN 2026-10-15)

Frozen before any token of reserve 6,000–7,999 was processed by any Stage-4 code. When this
file's sha256 was appended to `docs/HASHES.txt`, no e66 phase had run.

`scripts/e66_confirm_l12_family.py` asserts at every data phase:
- that this file's sha256 is in `docs/HASHES.txt`;
- that its own sha256 appears in this text (§9);
- that every §2 input matches its sha256;
- that e65's integrity check passes. That check covers e65's prereg, its inputs and e64's chain.

**Authorisation.** The user approved a fresh, preregistered confirmation of this family on
reserve 6,000–7,999, local MPS only, plus a description of what the direction does when
steered. The approval was relayed by the parent session.

**Nothing here amends any earlier verdict:**
- `e64_verdict.json`: "no discovery" under e64's tier 2;
- `e65_verdict.json`: this family is a "candidate discovery under the redesigned tier 2 (post
  hoc)".

## 0. What was seen before freezing

**Everything up to and including e65**, all of it read:
- e64's units, labels, context reading, S2, R2 and behaviour;
- e64c's rule check;
- e65's B3 and B4 outputs and verdict;
- REPORT through its e65 section.

**For this family specifically:**
- **B3** (reserve 4,000–4,999):

  | unit | Δ at c\* | random max | SIG max |
  |---|---|---|---|
  | logcosh #13 | 1.643 | 1.464 (1.12×) | 0.883 |
  | skew #16 | 1.409 | 0.971 (1.45×) | 1.028 (1.37×) |

- **B4** (reserve 5,000–5,999):

  | unit | Δ at c\* | random max | SIG max |
  |---|---|---|---|
  | logcosh #13 | 1.632 | 1.222 (1.33×) | 0.959 |
  | skew #16 | 1.483 | 0.809 | 1.085 |

- **Signatures:** " before", " times", " when", " couple", " again", "?", " since", " without",
  " because", " as", " few", " today", " while", " next", " during".
- **D2 context reading:** "none". My partial note on #13 was "clause-final content word", in 6 of
  12 contexts.
- **Other labels:** no e64c rule at |AUROC| ≥ 0.65; replicated on R2 (twin |cos| 0.95); not
  identity-suspect; not context-generic.

**Code run before freezing (no reserve data):**
- `e66 --phase selftest`: 4 synthetic checks, all pass.
- A smoke test of e66's model code paths, run with the real model on 40 D2 corpus prompts (data
  already used by e64):
  - σ for the anchor, #13 and #16 came out 358 / 307 / 350, against e65's B3 values of
    365 / 306 / 350;
  - the SIG set has 76 comparisons;
  - full-vocabulary accumulation, generation and the D2 profile all ran.
  The D2 profile's content was not printed or read. Only its keys were checked.

**What is not blind:**
- The family was selected by e65's B3 and B4 passes.
- The primary unit (§1) is chosen by rule, not by margin (§1).

This is a **confirmation of a post hoc lead on fresh contexts**, not a blind discovery test.

## 1. The family and the single primary test

**The family:** L12 ICA-64 logcosh #13 and its twin, skew #16 (|cos| 0.95), as frozen in
`results/occ_units_v2_L12.npz`.

**The primary test: logcosh #13** under e65's PASS rule (§4) on block C, given that the anchor
passes first on block C.

**Why logcosh #13.** It follows the project's standing convention: the logcosh member first, as
in e64's and e65's anchor choice. It was fixed before any block-C data existed. It is also the
member with the *thinner* RAND margin on B3 and B4, so the choice does not favour an easy pass.

**Secondary (reported, not part of the verdict):** skew #16, under the same rule, on the same
block.

## 2. Frozen inputs (asserted by sha256)

| file | sha256 |
|---|---|
| `scripts/e65_steering_redesign.py` (revision 2; imported for every tier-2 helper) | 5e37cab244e820337c7edd02f3f43ce0f30afa3fe4186e8444154a530aba5e86 |
| `PREREG_occurrence_v2_steering.md` (revision 2) | d0b7f5eae5ff32394de0a00925f47daade424599377d39e17c08ca4400d1d1d1 |
| `results/e65_frozen.json` (the blind signatures T_u and R_u of every screened L12 unit) | bd856844dae5db63d19518063d65eedf25ff8aaeee66565b09743f2da2fb2be9 |
| `results/e65_steer_B3.json` | a77c0893425ceab2f085cffc78f32c2fce3df7bd7b2789d68ecbba39dc02e1ee |
| `results/e65_steer_B4.json` | 15b04685a3fad65f549c9d02d8a485f70820b921e540d8f4967293fb0e3e1c97 |
| `results/e65_verdict.json` | a9a2628b3706dc57a69a116c33655a3055cc409b98431c2921e34efe45ed2aa1 |
| `results/occ_units_v2_L12.npz` | b84a7b13d52bd54acdc665175c7972b4e9418659a3e6b56d47bdc3bb895b5d5b |
| `results/e64c_rules_en_core_web_trf.npz` | 0dc1b07a4965455655ccb5bf117a664d1c40d7056f4e43f542cb5925e686be4d |
| `results/e64c_rules_en_core_web_sm.npz` | 5c1a228e8abbbbe0a725f81ff04e38aab62b64efadcb427df5d34b047cb41270 |

**Also covered, through `e65.check_integrity()`:** e64's maps, labels, context reading,
`reserve_tokens.i32`, corpus counts, `occD2_meta.npz` and e64's own chain.

**Unchanged from e65:**
- the unit directions (e64 maps);
- the signatures;
- the model and dtype (gemma-3-1b-pt, bf16, MPS);
- the prompt: BOS, then up to 64 preceding tokens, then the site token;
- the intervention point: the output of decoder layer 12, at the site position only.

## 3. Contexts (the last untouched reserve)

| block | reserve | role | site seed | RAND seed |
|---|---|---|---|---|
| **C** | 6,000–6,999 | **confirmatory steering**: the anchor, then the primary and the secondary | 6700 + 12 | 6710 + 12 |
| **D** | 7,000–7,999 | **description only** (§7), run after the confirm output is hashed | 6800 + 12 | — |

**Site rule** (as in e65):
- 200 distinct sequences per block;
- one position each, uniform in [5, 255] of the BOS-included row.

The `freeze` phase draws and hashes the sites and the block-C RAND bank before any forward
pass.

**Earlier exposure.** These sequences were touched only as 8-token slices, by e47r and e53
(Stage 2/3 type-mean steering). No Stage-4 code has read them.

## 4. Tier-2 rule: e65's, unchanged

The code is imported from e65.

| element | rule |
|---|---|
| **dose** | c·σ_d, c ∈ {0.5, 1, 2, 4}. σ_d is the SD of the unsteered layer-12 residual projected on d, over the block's prompt positions ≥ 4. Gate at **c\* = 2**. |
| **RAND** | 99 random directions drawn isotropically in D2's whitened PCA-64 coordinates, pulled back with U·diag(sd), and each dosed by its own σ at c\*. Gate: Δ_uu(c\*) > the maximum of the 99, scored with u's signature. |
| **SIG** | u's own steered pass at c\*, scored with every other screened L12 unit's signature. The twin is excluded, T_u's tokens are removed from the others' T and R, and a set needs ≥ 5 tokens left. Gate: Δ_uu(c\*) > their maximum. |
| **PASS** | all of: Δ_uu(c\*) > 0; sign consistency ≥ 0.8 at some c ∈ {0.5, 1, 2}; RAND; SIG. |
| **readout** | Δ = [lse(logits[T_u]) − lse(logits[R_u])] steered, minus the same unsteered |

## 5. Order (`--phase confirm`, block C, L12 only)

1. The unsteered pass: logits, σ for the anchor, primary, secondary and all 99 RAND directions,
   and ‖h‖ at the sites.
2. The 99 RAND passes at c\*.
3. **The anchor, logcosh #33 (quote), at every dose, and its gates.**
4. **Only if the anchor PASSes:** the primary (logcosh #13), then the secondary (skew #16), each
   at every dose, with their gates.

If the anchor fails, nothing else is steered.

## 6. Outcomes and wording (stated before any block-C data)

| outcome | requires | wording |
|---|---|---|
| **NO VERDICT** | the anchor fails on C | "The rig failed its anchor on block C; no confirmation verdict." |
| **CONFIRMED** | the anchor and the primary (logcosh #13) PASS on C | "The L12 logcosh #13 / skew #16 family passes a preregistered confirmatory steering test on untouched contexts. It is battery-unnamed (e64 §6.1), replicated on R2 (e64 tier 3), not named by any e64c rule at \|AUROC\| ≥ 0.65, and passes the e65 tier-2 rule on three disjoint context blocks (B3, B4, C)." The secondary's result is reported beside it. If both pass, the family as a whole is confirmed. |
| **NOT CONFIRMED** | the anchor PASSes on C and the primary fails | "The post hoc lead does not confirm on fresh contexts." The failing gate(s) are reported. |

**False-pass bound for the single primary test.** Under the null, the bound is
max(1/100, 1/(n′ + 1)), which is about 0.013 with n′ = 76 SIG comparisons. The anchor is a gate,
not a test. This bound is conservative here: B3 and B4 are already passed, and under an
exchangeable null the pass events are positively dependent (e65 §9).

**Not amended.** e64 and e65's verdicts and classes stand as written.
- A CONFIRMED family is the first unit in this project to clear all three of e64's tiers with
  tier 2 confirmed out of sample.
- **Whether to call it a "discovery" is the user's decision.**
- If the description (§7) finds a simple name for what the direction does, that is reported
  plainly. It means the family is "nameable after the fact", and the report says so. It does not
  change the confirmation verdict.

## 7. Description (`--phase describe`; descriptive only, never part of the verdict)

**When it runs.** After the confirm output is hashed, whatever the verdict.

**On block D** (the primary, logcosh #13; σ_d recomputed on D by the same rule):
1. **Output shift**, at c ∈ {−2, +2, +4}, with the push at the site only, as in the test. For
   every vocabulary token:
   - the mean change in log-probability, and the mean change in probability, over the 200 sites;
   - the top 30 promoted and demoted tokens by mean Δ log p, among tokens with mean unsteered
     p ≥ 1e-4;
   - the top 30 promoted and demoted by mean Δp;
   - the change in total probability per coarse token category. The categories are digit,
     punctuation, whitespace, subordinator, temporal word, preposition, function word,
     capitalised, word and other; the word lists are fixed in the script.
2. **Continuations.** For the first 8 block-D prompts: greedy 24 tokens, unsteered and with the
   push at c ∈ {+2, +4, −4}·σ_d. The push is added at the site position and at every generated
   position.

**On D2** (frozen scores; data already used), for both family members:
1. the 30 strongest heavy-tail contexts;
2. the category mix of the current and the next token in the heavy 1% tail, against a random
   20,000-row baseline;
3. the mean score and the tail rate by sentence position and by line position;
4. the top 10 battery associations, from `e64_labels.json`;
5. the oriented AUROC and tail rate of all 16 e64c rules, under both parses.

The REPORT write-up classifies the effect as digits/numbers, punctuation, position, or
something else, from items 1 (block D) and 2–3 (D2).

## 8. Constraints and order of work

- **Before any MPS phase:**
  - `ps` shows no other e64/e65/e66 process;
  - swap ≤ 16 GB, as e64's guard asserts;
  - disk is checked.
- **One launch per phase, no retries.**
  - If a phase crashes, its partial output is left unhashed and reported. It is not re-run without
    the user's go-ahead.
- **HASHES** is written only by the phases' own `append_hash` calls, plus the registration of this
  file and the script through the same helper.

## 9. Integrity

| file | sha256 |
|---|---|
| `scripts/e66_confirm_l12_family.py` | 21d47ae38b05c701c63f604ccc435ec5b60ac9f2b3c22f6ebd31b3034f2b3174 |

**Outputs**, each appended to `docs/HASHES.txt` by its phase:

| phase | writes |
|---|---|
| freeze | `results/e66_rand_L12.npz`, `results/e66_frozen.json` |
| confirm | `results/e66_confirm.json` |
| describe | `results/e66_describe.json` |

**Commands** (from `~/Downloads/gf-discovery-ext`, `HF_HOME=hf_home`, offline):

```
.venv/bin/python scripts/e66_confirm_l12_family.py --phase freeze
.venv/bin/python scripts/e66_confirm_l12_family.py --phase confirm
.venv/bin/python scripts/e66_confirm_l12_family.py --phase describe
```

**Expected cost:**
- confirm: about 112 forward passes, about 10–15 min on MPS;
- describe: about 4 passes plus 768 short generation steps, about 5 min, plus a few minutes of
  CPU for D2.
