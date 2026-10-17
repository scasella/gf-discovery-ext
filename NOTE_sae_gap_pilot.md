# NOTE — SAE-gap pilot: do the persistent components of the L12 SAE error survive the cheap falsifiers? (plan fixed 2026-10-15, before any pilot number)

**What this is.** The go/no-go pilot for option 2 ("look where SAEs fail").
- **Authority.** The user asked for it ("option 2").
- **Design.** A judge panel picked the design: 4 independent designs, 3 independent judges, all
  3 choosing **sae_error**. The record is `docs/sae_gap_calibration/design_panel_result.json`,
  hashed via `docs/sae_gap_calibration_MANIFEST.sha256`.
- **Data.** Burned data only: D2 (corpus 1,000–1,999) and S2 (D2's within-sequence shuffle, e60's
  `occS2`). No fresh corpus text is read or extracted.
- **Verdicts.** Nothing earlier (e64–e68) is amended.

**Why a pilot.** The design's own chain puts its main risk early:
- S3, "context-generic persistence", fails about 0.65 of the time.
- Naming by a predictability or bag-of-tokens rule is the next risk.

Both can be tested on burned data in about an hour. This pilot decides whether a full prereg
(fresh D3/S3/R3/B5/B6) is written at all.

**What existed before this note was hashed.** The designers' read-only D2 calibration
(`docs/sae_gap_calibration/err_*.py`, `design_sae_error_draft.txt`):
- 3 screened slow units, with split-half 0.94 / 0.84 / 0.83 and lag-1 acf about 0.38;
- type-and-position labels ≤ 0.568;
- linear bag-of-previous-token-types R² ≤ 0.04.

No S2 error rows, no next-token entropy or log p, no nonlinear bag model and no sub-threshold
number had been computed for these units.

## Instrument (frozen from the design; reproduced here, not tuned)

| step | rule |
|---|---|
| error | e = h − (JumpReLU_fp32(h) @ W_dec + b_dec), with `resid_post_all/layer_12_width_16k_l0_big`. h is the raw L12 residual (output of decoder layer 12), cast to fp16 and back, exactly as D2's stored `results/e67_raw_L12_rows.npy`. Coordinate 1038 is dropped after the error is formed (1,151 coordinates). |
| rows | e63's kept-row rule on the set's own meta: positions ≥ 4, token types with ≥ 8 rows in the set. For D2 these are `occ_units_v2_L12.npz["row"]` (asserted). |
| identity removal | type, then document, then position means subtracted, cycled 3 times (`demean3`). |
| whitening | PCA-256 fitted on every third row, whitened. |
| units | multi-lag AMUSE: M = mean over τ ∈ {1, 2, 4, 8, 16} of sym(C_τ), where C_τ = mean of z_t z_{t−τ}ᵀ over within-sequence pairs with both rows kept; top m = 32 eigenvectors w_u; score y_u = Z w_u. |
| filter / pattern | filter f_u = U diag(1/sd) w_u; pattern p_u = U diag(sd) w_u (residual space, 1,151-d), normalised. |
| split-half | AMUSE refitted on even- and odd-sequence halves; rep_u = min over halves of max \|cos\| of w_u against the half's top 64. |

**Screens** (a unit is SCREENED iff all hold):
- rep_u ≥ 0.8;
- η²(token) of y_u ≤ 0.3;
- η²(token) of y_u² ≤ 0.3;
- \|Spearman\| ≤ 0.3 with each of the identity-removed log‖h‖, log‖e‖ and L0;
- lag-1 acf ≥ 0.2.

## Phases (`scripts/e69_sae_gap_pilot.py`)

**P0, reproduce (CPU, stored D2 rows).** Run the instrument on D2.
- Report n_screened, the screened ids, rep, acf at lags 1/16, η², the SAE-state ρ, and the top
  eigenvalues.
- The design's D2 reading was 3 screened units.
- **STOP-P0:** fewer than 2 screened units ("no reproducible persistent structure even on
  calibration"). A count other than 3 is reported, but is not a stop.
- The P0 output (units, filters, patterns, PCA maps) is **hashed before P1 runs**.

**P1, model passes (MPS, one process).** Two forwards; only the outputs listed under Outputs are
written.
- **S2:** the raw L12 residual of `occS2_meta.npz` tokens (asserted shuffled, from the corpus, and
  a per-sequence permutation of D2's tokens), at S2's kept rows. Cast fp16→fp32 as D2.
- **D2:** next-token entropy H_t and log p(actual next token) at every non-final position, from
  the model's logits (fp32 softmax). The pass also re-captures D2's raw L12 rows (fp16) and
  asserts that they match `e67_raw_L12_rows.npy`: ≥ 99% of values are identical and the
  per-coordinate correlation is ≥ 0.999. The max \|Δ\| is reported. This checks that S2 is
  captured by the same path as D2.
- Check swap before the job; the model is freed after it.

**P2, tests (CPU).** For every P0-screened unit u:

1. **S3-generic (a), refit match.** The full instrument is refitted on S2 (its own error, kept
   rows, identity removal, PCA-256 and AMUSE). u is GENERIC-a if some S2 unit among the top 64 has
   pattern \|cos\| ≥ 0.8 with p_u.
2. **S3-generic (b), transfer.** Apply u's D2 filter f_u to S2's identity-removed error. u is
   GENERIC-b if acf₁(S2) ≥ 0.5 · acf₁(D2) **or** the lag-1 autocovariance under the same filter
   satisfies acov₁(S2) ≥ 0.5 · acov₁(D2).
   - The autocovariance arm guards against shuffled text inflating the SAE error, which would
     dilute the autocorrelation while the persistent component survives.
   - var(y_S2)/var(y_D2) is reported, as are the mean log‖e‖, log‖h‖ and L0 on S2 against D2.
   - S2's set-level eigenvalues and screened count are reported against D2's.
3. **Predictability naming.** u is NAMED-P if \|Spearman\| ≥ 0.3 with any of:
   - H_t;
   - −log p_t;
   - their causal running means over the current and previous 3 or 15 positions of the same
     sequence (window 4, window 16).
   - Rows are D2's non-final kept rows.
   - Each feature is tested twice: raw, and identity-removed by `demean3` on the non-final kept
     rows (as the unit scores and the bag features are). Either one reaching \|ρ\| ≥ 0.3 names the
     unit.
4. **Bag naming.** u is NAMED-B if any held-out R² ≥ 0.3. The input is the causal mean, over the
   previous W ∈ {8, 16, 32, 64} tokens of the same sequence (the current token excluded, as in the
   design), of a 128-d type embedding. The embedding is the PCA of D2's per-type mean raw L12
   residual, as in `err_bag.py`. Two predictors are used:
   - ridge (RidgeCV α ∈ 10^[−2..4]);
   - an MLP (sklearn `MLPRegressor`, hidden (256,), early stopping, seed 0).
   - Both are trained on even sequences and tested on odd. The features are identity-removed by
     `demean3` first.
5. **Sub-threshold coverage, descriptive (calibration for the full prereg; no gate).**
   - Report the max \|r\| of y_u with every L12 16k big and small latent's **pre-activation**.
   - Report the held-out R² (even → odd, RidgeCV) of y_u from the 128 latents with the highest
     even-half \|r\|, using the features {a_j, clip(pre_j, 0, θ_j)}.
   - Both are also computed for 32 RAND-E directions (g ~ N(0, I_256) in D2's whitened PCA-256
     coordinates, seed 6900), and u's values are reported as percentiles of those.
   - The full prereg will set its SUB-THRESHOLD rule from these numbers.
6. **Known labels, descriptive.** The best \|AUROC\| over the 7 battery-style and 10 e64c labels,
   as in `err_sobi5.py`.

**P3, decision (CPU, from the P2 file).** A unit is a **PILOT SURVIVOR** iff it is screened, not
GENERIC-a, not GENERIC-b, not NAMED-P and not NAMED-B.

| outcome | consequence |
|---|---|
| ≥ 1 survivor | **GO:** write the full prereg on fresh text (D3 = corpus 2,000–2,999 and onward, as in the design plus the judges' grafts), with its own adversarial review and freeze. |
| 0 survivors | **STOP:** no fresh text is spent. Reported as an informative negative, saying which falsifier removed each unit (shuffled-context persistence, predictability or bag). |

**Lean of the rules.** Every rule above leans toward STOP:
- a lenient top-64 match for GENERIC-a;
- a 50% transfer bar for GENERIC-b;
- several predictability and bag features at 0.3.

A false GO wastes fresh text; a false STOP only costs a lead that the pilot's own evidence did not
support.

## Not in this pilot

These are left to the full prereg (all from the design and the judges' grafts):
- the knockout and planted controls (C1, C2, C2' with plant-in-h and re-encode);
- R3 replication, steering and coverage;
- the error-share (ES) decomposition.

The pilot can only stop a round or license writing one. It cannot produce a discovery claim.

## Resources

- **Memory.** D2 and S2 each hold about 0.94 GB of fp32 error rows; the demeaning works in
  float64 per call.
- **Disk.** P1 stores S2's kept-row raw L12 residual in fp16 (about 0.47 GB, the same format as
  `e67_raw_L12_rows.npy`). Everything else is JSON or small npz files. The pilot runs after the
  census's two 262k medium dictionaries are deleted.
- **Order.** It runs only after the e68 census MPS stages finish, never concurrently with another
  MPS job.

## Outputs

| item | file |
|---|---|
| P0 | `results/e69_p0_units.npz` (hashed), `results/e69_p0.json` (hashed) |
| P1 | `results/e69_p1_S2_raw_L12_rows.npy` (fp16, S2 kept rows; hashed as soon as it is written, so a crash in the logits pass can resume) and `results/e69_p1_d2_pred.npz` (D2 entropy and log p at every position); both hashed before P2 |
| P2 / P3 | `results/e69_pilot.json` (hashed) |
