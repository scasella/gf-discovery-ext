# Stage 2, Addendum C5: the C4 parametric null fails its own calibration and is withdrawn (2026-10-15)

Exposure: L5 (complete) and L6 (supervised controls only) under C4. No blind run had started.

## What was measured

On the **Gaussian matched null pool** at L5 (no rulers by construction):
- **275 prefiltered sets pass BH(q = 0.05)** on the C4 per-set parametric null;
- **115 lattice sets** do the same;
- min p_param is at the floor (1e-4).

Under a calibrated null, BH would pass about 0.

## Why

1. **Winner's curse.** The prefilter keeps the 1% most regular proposals. The parametric null
   then asks the same set whether it is more regular than surrogates, so the p-values of selected
   sets are near the floor by construction.
2. **Shape mismatch.** A seed's k nearest neighbours are a truncated ball, not a Gaussian draw.
   Their projections onto their own PC1 are more evenly spread than Gaussian surrogates with the
   same covariance.

So C4 is anti-conservative and **cannot carry a claim**. It is reported descriptively only.

## Replacement (no tier is claimed that the null pool has not been run through)

| tier | definition |
|---|---|
| Geometric candidate | prefilter (T1 ∧ T2 ∧ T3, population-relative) ∧ T4 persistence at every adjacent scanned layer |
| Expected-false rate | the same conjunction on Gaussian matched null pools. T4 needs adjacent layers, so the null pool's single-layer prefilter rate is reported, with T4 applied to data only. |
| Excess | the fraction of data proposals whose even or graded regularity falls below the Gaussian null pool's 1st percentile, against 1%. Descriptive: the Gaussian pool lacks the data's family structure. |
| Selection | handled behaviourally, as in Phase 5 (Addendum R / script 51). Ruler candidates go to the 47-style test against a **sibling null of other geometric candidates from the same layer and k**, which were selected the same way, at matched prompt budgets on reserve contexts. Expected false behavioural passers = n tested × 0.01. |

Calibration therefore reports:
- supervised controls (population p_bg, even and graded);
- plant recall through prefilter and T1–T3;
- the Gaussian-null prefilter rate;
- the class-permuted FPR (kill if > 5%).

C4 plant recall from the partial run is kept in
`results/e46_calibrate_C4_withdrawn_partial.json` for the record.
