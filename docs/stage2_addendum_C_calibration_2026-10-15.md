# Stage 2, Addendum C: changes forced by calibration, written before any blind run (2026-10-15)

Exposure before this file was written:
- the L6 supervised pool controls (number words, kinship, ranks);
- L6 plant probes;
- one Gaussian null pool at L6 (the first spec version);
- the L6 cell thresholds.

Nothing else was seen: no blind candidate list, no other layer's statistics, and no blind
membership.

## C1. Plant scale (the original plant had zero power for a measured, fixable reason)

The first plant used the random-8 offset (132) as its unit and the nearest-neighbour distance
(227) as its spacing. That put families inside the bulk with spacing 4–5× wider than real ones,
and proposal recall was 0.

Plants are now anchored to the measured geometry of the one supervised 1-D family in the pool,
the number words, at each layer. At L6 that is:
- offset |mean| = 314;
- PC1 spacing = 48;
- off-line RMS = 189.

Sweep: offset ×{0.5, 1, 1.5}, off-line scatter ×{0.5, 1}, along-axis jitter {0, 0.1, 0.2, 0.3}
× spacing.

**Measured limit, kept, not tuned away.** At offset ≥ 1× with off-line ×0.5, adjacent planted
members reach cosine > 0.90. The R15 duplicate guard then collapses them, and the plant is never
proposed. Very tight families are outside the detectable class. The guard exists because
capitalisation and morphology variants sit in that regime.

## C2. Nuisance tier threshold: |ρ| < 0.30 → |ρ| < 0.80

For k = 8–12, an independent nuisance variable exceeds |ρ| = 0.3 about 40% of the time per
variable. With 3 or more variables, the 0.30 rule rejected true plants about 80% of the time.
The tier's stated purpose is "not a direction that sorts by log-count or norm (ρ ≈ 1)".
|ρ| ≥ 0.80 corresponds to p ≈ 0.01 for an independent variable at k ≈ 10. Nuisance directions
are also linearly deflated from the coordinates beforehand (spec §1).

## C3. Graded spacing (T2g) added, alongside even spacing (T2even)

- The supervised number words lie on a **log-compressed** line at L6: PC1 positions −279, −184,
  −102, −43, 16, 62, 105, 128, 140, 157, with spacings shrinking from 95 to 12.
- Even spacing gives them p_bg = 0.041, which fails. The representation's magnitude scale is not
  evenly spaced, so a T2even-only tier is blind to it.
- T2g is defined only for sets that are monotone on B. It is the RMS residual of a quadratic-in-
  rank fit to the B positions, divided by the mean spacing.
- Numbers: graded 0.070, p_bg 0.00027.
- T2 := T2even ∨ T2g. Each is at cell p ≤ 0.01, so the per-proposal T2 rate is at most 2% by
  construction.

## C4. The tier that carries the claim: a per-set shape-matched parametric null, then BH

Population-percentile tiers (T1/T2) pass about 1–2% of proposals by construction (56–57
single-layer passers per (L, k) at L6). They cannot separate a ruler from the tail of a
continuum. So T1 ∧ T2 ∧ T3 is demoted to a **prefilter**.

For every prefiltered set S:
- Draw 4,000 surrogate k-sets from N(μ_S, Σ_S), the set's own mean and covariance within its
  own span.
- B surrogate = A surrogate + S's own B−A noise rows, permuted.
- Compute even and graded regularity on each surrogate.
- p_param = 2 × min(p_even, p_graded), Bonferroni over the two statistics.

This asks whether S is more evenly or smoothly spaced than a Gaussian cloud of the same shape.

**RULER PASS** = prefilter ∧ BH(q = 0.05) over all prefiltered sets in the layer ∧ T4
persistence.

**LATTICE** gets the same parametric null on grid_B (2,000 surrogates), with the same BH and
persistence.

**Calibration reports through this full conjunction:**
- supervised number words;
- plant recall;
- Gaussian matched null pools, giving expected false passers.

Class-permuted FPR > 5% on the prefilter still kills the instrument.
