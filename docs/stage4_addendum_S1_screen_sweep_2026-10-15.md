# Stage 4, Addendum S1: declared screen-threshold sweep on the pilot set P (2026-10-15)

Authorised by the user's instruction (3): "if the probe decodes it but no screened unit does, you
may add a declared screen-threshold sweep to the P-only pilot before freezing for D, written up
as an addendum". Written before the sweep runs and before any D data exists.

## What triggered it (P only, L7)

- **Supervised linear probe** on the final representation (identity-removed,
  document-centred, position-deflated; held-out on odd sequences):

  | variable | base rate | AUROC, 1151-d | AUROC, PCA-64 whitened | best single whitened PC |
  |---|---|---|---|---|
  | paren_inside | 2.5% | **0.825** | 0.832 | 0.68 |
  | quote_inside | 5.8% | 0.826 | 0.760 | 0.59 |

  Both variables **survive** the representation. The representation is not the failure point.
- **Pilot grid** (12 settings): **0 screened ICA units at every setting**, and 0–1 screened
  clusters. Per criterion at K = 128:
  - token-driven index ≤ 3: passed by 3 of 128 clusters (median TDI 8.2);
  - split-half reproducibility ≥ 0.8: passed by 68 of 128;
  - ICA bimodality > 0.555: passed by 0 of 32 (q90 0.25);
  - ICA split-half reproducibility ≥ 0.8: passed by 0 of 32 (max ≈ 0.52).
- **Upper bound with every screen removed**: best cluster F1 for paren is 0.09 (K = 128), 0.105
  (K = 64) and 0.247 (K = 256); best ICA AUROC is 0.69. The §6 floors are F1 ≥ 0.50 and
  AUROC ≥ 0.85.

## Sweep (declared now)

- Settings: K ∈ {64, 128, 256}, N_ICA = 32, PCA = 64.
- Threshold grid:
  - cluster TDI ∈ {3, 5, 10, ∞};
  - cluster split-half reproducibility ∈ {0.8, 0.6, 0};
  - ICA bimodality ∈ {0.555, 0};
  - ICA reproducibility ∈ {0.8, 0.5, 0};
  - the size, type-diversity, type-share and document screens are unchanged.
- Each combination reports the number of screened clusters and ICA units, and the
  paren/quote max F1 and AUROC against the token-matched null p99.9 (1,000 permutations, as
  preregistered).
- **Choice rule:** the combination that recovers paren_inside on P with the strictest
  thresholds. Ties go to the smaller TDI cap, then the higher reproducibility.
- **If no combination recovers** (expected, since the unscreened upper bound is already below
  the floors):
  - the preregistered defaults are frozen unchanged;
  - D, S and R are still run under PREREG_occurrence.md, so the positive control is evaluated on
    D at all three layers as registered;
  - if it fails there, Stage 4 stops with **"instrument not validated: the variable is linearly
    present (probe AUROC 0.83) but no unsupervised unit isolates it"**.

The recovery floors (F1 ≥ 0.5, AUROC ≥ 0.85) and the null are **not** swept.
