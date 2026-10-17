# NOTE — Does an existing Gemma Scope 2 feature already cover the confirmed L12 direction? (rubric fixed 2026-10-15, before any comparison)

**What this is.** A read-only check of whether the e66-confirmed L12 logcosh #13 / skew #16
direction is already a Gemma Scope 2 feature.
- **Authority.** The user approved it; the parent session relayed it.
- **Data.** Local only, on D2 (corpus 1,000–1,999), which earlier stages already used. No reserve
  data is used; none is left.
- **Verdicts.** e64, e65 and e66 are not amended.

**When this note was hashed:**
- no decoder cosine, latent correlation, reconstruction or probe for this direction had been
  computed;
- no Neuronpedia page had been opened.

## Objects

**The direction.** `d` = `dir_logcosh[13]` and `dir_skew[16]` from `results/e64_maps_L12.npz`:
e64's residual-space pattern pullback, unit norm, signed toward the heavy tail. It is 0 at the
dropped coordinate 1038.

**Comparison convention.**
- Decoder rows are compared with coordinate 1038 masked, then renormalised. This is the space the
  direction lives in.
- Unmasked values are reported too.

**SAE dictionaries.** All are Gemma Scope 2 for gemma-3-1b-pt, from `google/gemma-scope-2-1b-pt`.

| layer | site | width / L0 |
|---|---|---|
| L12 | `resid_post_all` | 16k l0_big (local); 16k l0_small; 262k l0_big; 262k l0_small |
| L11, L13 | `resid_post_all` | 16k l0_big; 16k l0_small |
| L10, L14 | `resid_post_all` | 16k l0_big; 16k l0_small |
| L13 | `resid_post` | 65k l0_big (cheap to add) |

Not used:
- the 262k widths at L10, L11, L13 and L14 (disk budget);
- the 1M width (about 10 GB).

## Measurements (`scripts/e67_prior_feature_check.py`)

1. **Decoder cosine.** |cos(d, decoder row)| for every row of every dictionary; the top 10 per
   dictionary, for both members.
   - **Isotropic null:** the p99.9 of the max |cos| over the dictionary's rows, for 1,000
     isotropic random unit vectors in 1,152 dimensions (masked).
   - **Subspace null:** the same for 1,000 pullback-matched random directions (e65's RAND
     construction, seed 6720). These live in the subspace the unit directions live in.
2. **Activation correlation (D2).**
   - Run the model on D2 once.
   - Encode the layer's residual with each dictionary (fp32 JumpReLU, as in the project).
   - Over D2's 203,328 kept rows, compute the Pearson r of every latent's activation with the
     unit's frozen D2 score (Y from `occ_units_v2_L12.npz`).
   - Report the top 10 by |r| per dictionary. Also report r with the raw projection h·d, at the same
     rows.
3. **Sparse reconstruction** (per dictionary, at L12 and its neighbours):
   - greedy OMP over unit-normalised (masked) decoder rows, signed;
   - greedy non-negative OMP, run on d and on −d;
   - the number of atoms needed to reach cos 0.8 and 0.9, capped at 256.
   - **Null:** the same for 10 pullback-matched random directions (median atom count).
4. **Supervised baseline: the "ordinary explanation" probe.** On D2, the label is "the next token
   is a preposition or subordinator". The categories are the e66 word lists, decided by
   `e66.category` on the decoded next token.
   - **Primary.** A logistic probe (C = 1) on the unit's own representation W: identity-removed
     and per-type whitened, 64-d, rebuilt by e64's frozen code from a re-extraction of D2 by e60.
     Train on even sequences, test on odd. Its pattern, Cov(Z, standardised probe score), is
     pulled back exactly as d was. Report cos(probe pattern, d).
   - **Also reported:** a logistic probe on the raw L12 residual (its filter and its pattern), and
     the class-mean difference of the raw residual. Each is reported by its cosine with d.
5. **Explanations.**
   - Neuronpedia's explanation for each top match, if Neuronpedia hosts these Gemma Scope 2
     dictionaries for gemma-3-1b. Read-only GET, reported verbatim.
   - In all cases, each top latent's 12 strongest D2 contexts.

## Rubric (fixed now)

**Behavioural match (mechanical).** A latent "matches the observed behaviour" iff, among its
200 strongest-activating D2 kept rows (sequence-final rows excluded), the share whose **next
token** is a preposition, subordinator or temporal word (`e66.category`) is **≥ 0.30**.

| | share of that next-token category |
|---|---|
| the unit's own heavy 1% tail | 0.375 |
| D2 base rate | 0.111 |

A Neuronpedia explanation is reported alongside and is a qualitative check only.

**Verdict:**

| verdict | requires |
|---|---|
| **COVERED** | Some single latent, in any listed dictionary, has **decoder \|cos\| ≥ 0.7** with d (either member), **or activation \|r\| ≥ 0.7** with the unit's D2 score. **And** that latent passes the behavioural match. |
| **PARTIALLY COVERED** | Not covered, but **≤ 5 decoder rows of one dictionary** reconstruct d at **cos ≥ 0.9** (signed or non-negative OMP), **and** a majority of those rows pass the behavioural match. |
| **NOT COVERED** | Otherwise. |

**Relation to the ordinary explanation** (cos of the W-space probe pattern with d, for logcosh
#13):

| \|cos\| | reading |
|---|---|
| ≥ 0.9 | essentially the next-token-syntax probe direction |
| 0.7–0.9 | closely related |
| < 0.7 | distinct from it |

**Not decided by this note:** whether the confirmed direction is called a "discovery". That
stays the user's decision.

## Addendum A (2026-10-15, before any comparison was computed)

**What Neuronpedia actually hosts.** It hosts Gemma Scope 2 for gemma-3-1b only at the
`resid_post` layers it serves, 7 and 13. The source record of `13-gemmascope-2-res-65k` names
`resid_post/layer_13_width_65k_l0_medium`. It hosts none of the L12 `resid_post_all`
dictionaries listed above.

**Change.** To make explanations available, three dictionaries are **added** to the comparison
set, with every measurement and the rubric unchanged:
- `resid_post/layer_13_width_16k_l0_medium`;
- `resid_post/layer_13_width_65k_l0_medium`;
- `resid_post/layer_13_width_262k_l0_medium`.

Adding dictionaries can only make **COVERED** or **PARTIALLY COVERED** easier to reach.

**Explanations.** Neuronpedia's explanation is reported for top matches in these three
dictionaries. For every other dictionary, the top latents' D2 contexts serve as the explanation.
