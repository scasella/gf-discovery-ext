# Stage 4 v2, calibration addendum A1: ICA dimension (2026-10-15)

Written after a reduced-grid smoke run of `e63 --phase pilot` on P2 at L7, and before the full
pilot. The smoke run used 5 permutations and the grid N_MIN = 64, n0 = 64, K ∈ {16, 32},
N_ICA = 16; its output is in the scratchpad, not in `results/`. D2 does not exist.

## What the smoke run showed (P2, L7; disclosed in full)

**Step A.**

| setting | self-excluded TDI | self-inclusive TDI (v1 statistic) | KIR | paren probe |
|---|---|---|---|---|
| B0 (the v1 representation) | 5.8 | 7.7 | 39 | 0.755 |
| B1 (iterated first moments) | 6.8 | — | 39 | 0.765 |
| per-type whitening (N_MIN 64, n0 64) | 1.9 | — | 15 | 0.757 |

**Step B, unscreened.**
- paren F1: K = 16 gives 0.35 against a 5-permutation null of 0.06; K = 32 gives 0.30.
- ICA AUROC with 16 components: 0.66 (logcosh) and 0.67 (skew).
- quote: F1 0.16 / 0.21, and ICA AUROC 0.64.

**Moments.** The paren-probe direction has skewness 0.49 and excess kurtosis 3.1. The most skewed
component of either 16-component ICA fit has skewness 0.34.

## The defect

- `FastICA(n_components = N, whiten = "unit-variance")` first projects its input onto the top N
  principal components.
- e61 (v1) and e63 both pass it an input that is already whitened:
  - v1 used PCA-64 whitened data;
  - v2 is globally re-whitened after the per-type step, on the very rows ICA is fitted to.
- Every eigenvalue is therefore about 1, and in v2 exactly 1. For N < 64, the N-dimensional
  subspace ICA searches is fixed by sampling noise and floating-point order, not by the data.
- This affects:
  - v1's 32-component ICA, a likely contributor to v1's ICA split-half reproducibility of
    0.37–0.39 median at L7/L12/L24;
  - this plan's N_ICA ∈ {16, 32}.
- The smoke run's moment check is consistent with this. The paren direction is more skewed than
  anything the skew-contrast ICA returned, so the search space missed it.

This is a structural fact about the estimator, independent of any label. It is not a choice
made from the paren numbers above.

## Change

- **N_ICA = 64 for both contrasts**, the full whitened space, so that no noise-chosen projection
  is involved.
- Replaces the plan's N_ICA ∈ {16, 32}; the plan's defaults become N_logcosh = N_skew = 64.
- Nothing else in the plan changes: Step A grid and rule, K grid, screens grid, floors, null,
  choice rule.
