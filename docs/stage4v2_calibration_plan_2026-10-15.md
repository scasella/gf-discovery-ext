# Stage 4 v2: pilot calibration plan (written 2026-10-15, before any v2 pilot statistic is computed)

This plan governs the P2-only calibration. Its outcome is frozen into `PREREG_occurrence_v2.md`
before any D2 data exist.

The only v2 data in hand when this was written is the P2 extraction (corpus sequences
13,000–13,999, layers 7/12/24, tag `occP2`). Nothing has been computed from it except the
type-count table below.

## Sets (disjoint from v1's P/D/S/R)

| set | content | role |
|---|---|---|
| P2 | corpus 13,000–13,999 | calibration only |
| D2 | corpus 1,000–1,999 | positive control (this run) |
| R2 | reserve 2,000–2,999 | replication; untouched in this run |

v1 used corpus 14,000–14,999 (P), corpus 0–999 (D and its shuffle S), and reserve 0–999 (R).
The v1 behavioural tier reserved reserve 1,000–1,999.

**P2 type counts** (positions ≥ 4, types with ≥ 8 rows): 202,854 rows over 3,553 types.

| types with at least | number of types | share of rows |
|---|---|---|
| 32 rows | 752 | 80.5% |
| 64 rows | 335 | 71.7% |
| 128 rows | 174 | 65.0% |

## Why v1 could not pass, before any v2 data

1. **Mean subtraction leaves identity in the second moment.** Cluster TDI was 7–8.5×, and that
   is the target of v2.
2. **The cluster F1 floor (0.50) was nearly unreachable at K = 128.**
   - A cluster of size s can reach at most F1 = 2·min(s, n₁)/(s + n₁), even at perfect
     precision.
   - On v1 D the median cluster held 1,472 rows against about 5,700 paren-inside rows, so the
     cap was about 0.41.
3. **The ICA AUROC floor (0.85) sat above the supervised linear probe's ceiling on that
   representation** (0.83 on P).
4. **v1's TDI sampled the same-type replacement from all occurrences of the type, the row
   itself included.**
   - This adds about 128·E[1/n_t] ≈ 2.2 to a cluster's TDI on P2 if types are spread evenly,
     and more for clusters rich in rare types.
   - v2 samples among the *other* occurrences. Both versions are reported on P2 for the
     baseline.

## v2 representation (per set, using only that set's rows)

1. **Rows:** positions ≥ 4, types with ≥ 8 rows in the set, outlier coordinates dropped as in
   v1 §4.
2. **Iterated first-moment removal:** subtract type, document and position means in turn, and
   repeat the cycle 3 times. v1 ran a single pass, and the three groupings are not orthogonal.
3. **Global PCA to 64 dimensions** (fitted on the set itself, every third row, as in v1), then
   whitening. This is space Z.
4. **Per-type second-moment equalisation in Z:**
   - Each type t with n_t ≥ N_MIN uses
     Σ̂_t = (1−λ_t)·S_t + λ_t·S_pool, with λ_t = n0/(n_t + n0).
   - S_t is the type's own covariance, and S_pool is the pooled within-type covariance.
   - Types with n_t < N_MIN use S_pool (λ = 1).
   - Each row becomes (z − μ_t)·Σ̂_t^(−1/2). The result is then globally re-whitened (rotation
     and scale only, still 64 dimensions).

## Step A: whitening setting, chosen per layer by a label-free criterion plus a probe guard

**Grid per layer:**
- N_MIN ∈ {32, 64, 128};
- n0 ∈ {16, 64, 256};
- two baselines:
  - **B0**, the v1 representation (single pass, no per-type step);
  - **B1**, iterated first moments without the per-type step.

**Measured per setting:**
- **TDI**, self-excluded: k-means with K = 128 (MiniBatchKMeans, as v1), median over clusters.
  B0 also reports the self-inclusive v1 statistic.
- **KIR**, the kNN identity ratio, reported only:
  - 4,000 random query rows, 10 nearest neighbours each;
  - the observed same-type neighbour rate divided by the expected rate (n_t − 1)/(n − 1),
    averaged over queries.
- **paren_inside probe**, the guard: a held-out linear probe (logistic regression, C = 1).
  - It trains on even sequences and tests on odd ones, in the 64-dimensional representation.
  - quote_inside is **not computed in Step A**, so it stays untuned.

**Choice per layer:** the setting with the lowest median TDI among those whose paren probe AUROC
is ≥ B1's probe AUROC − 0.02. Ties go to the larger n0 (more shrinkage), then the larger N_MIN.
If no per-type setting satisfies the guard, B1 is chosen and this is reported.

## Step B: units and screens, paren-tuned (on the Step-A representation for each layer)

**Unit kinds:**
- k-means with K ∈ {16, 32, 64, 128}. The F1 caps at K = 16/32/64 are above 0.5.
- FastICA, logcosh contrast (v1's), with N ∈ {16, 32}.
- FastICA, **skewness contrast** (G(u) = u³/3), with N ∈ {16, 32}.
  - Declared now, because a sparse binary context variable (base rate 3–6%) is a skewed
    source rather than a bimodal or heavy-tailed one.
  - Pilot moment check, reported: skewness and excess kurtosis of the P2 paren-probe
    projection against those of the fitted ICA components.

**Screens:** v1 §5, except for the declared grid below.

| unit kind | screen grid |
|---|---|
| cluster | TDI (self-excluded) cap ∈ {3, 5}; split-half reproducibility ∈ {0.8, 0.6} |
| ICA | η²(token) ≤ 0.30 (fixed); split-half \|cos\| ∈ {0.8, 0.5}; bimodality coefficient ∈ {> 0.555, off} |

- Bimodality is "off" by default in v2. A weakly separated 3% subpopulation is unimodal, so the
  v1 screen excludes the target class by construction.
- The cluster size (≥ 300), type (≥ 30), type-share (≤ 0.30) and document (≥ 100) screens are
  unchanged.

**Recovery floors (frozen rule, fixed now):**
- Cluster: F1 ≥ 0.50 (unchanged).
- Directional (either ICA kind), at layer L:
  AUROC ≥ clip(0.5 + 0.75·(probe_P2,L − 0.5), 0.70, 0.85).
  - probe_P2,L is the Step-A paren probe AUROC at the chosen setting.
  - So a unit must carry at least three quarters of the linearly decodable excess.
- Both floors also require exceeding the token-matched label-permutation null at p99.9 (1,000
  permutations, max over screened units, as v1 §6).

**Choice:** one unit configuration (K, N_ICA for each contrast) and one screen setting for all
three layers, chosen in this order:
1. the number of layers at which paren_inside is recovered;
2. then the strictest screens (TDI cap 3 before 5, higher reproducibility first, bimodality on
   before off);
3. then the mean margin over the null.

**If nothing recovers on P2:**
- freeze the defaults: K = 32, N_ICA = 32 for both contrasts, v1 cluster screens with the
  self-excluded TDI, and ICA screens with η² ≤ 0.30, reproducibility ≥ 0.8 and bimodality off;
- still run D2 as registered.

**quote_inside** is computed in Step B and reported, but never used in any choice.
