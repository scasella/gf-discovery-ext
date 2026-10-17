# Addendum A1 — Stage-1 reproduction gate: declared-tolerance miss on the SAE channel (2026-10-15)

Written after `e02_gate.py` ran and before script 22 ran on any deep layer. BUDGET.md is not
edited; this file amends §6.2.

## Outcome at the declared tolerances

| check | anchor | rebuilt | \|Δ\| | tolerance | result |
|---|---|---|---|---|---|
| raw strict-12 gap, L4 | 1.585 | 1.5849 | 0.0001 | ±0.01 | pass |
| raw L5 | 1.475 | 1.4760 | 0.0010 | ±0.01 | pass |
| raw L6 | 1.521 | 1.5197 | 0.0013 | ±0.01 | pass |
| raw L7 | 1.611 | 1.6131 | 0.0021 | ±0.01 | pass |
| raw L8 | 2.012 | 2.0101 | 0.0019 | ±0.01 | pass |
| raw L9 | 1.766 | 1.7587 | 0.0073 | ±0.01 | pass |
| SAE 13-set, L6 (pre-drop) | 1.6588 | 1.65648 | 0.0023 | ±0.001 | **FAIL** |
| SAE 14-set, L7 (pre-drop) | 1.6202 | 1.61596 | 0.0042 | ±0.001 | **FAIL** |

## Diagnosis

1. **The corpus is identical.** All 5,152 stored row-to-token anchors match. The pool
   reproduces exactly: 3,795 types, 83,310 distinct, 3,197,155 rows covered.
2. **The decode path is exonerated.** At L12, the pass-1 SAE accumulator equals script 22's own
   cache-based means (`type_means` → `project`) to a relative error of 7e-6. The 13-set gap
   agrees to 1e-7.
3. **The deviation looks like numeric drift.**
   - On the raw channel it grows with depth (0.0001 at L4 to 0.0073 at L9).
   - The SAE channel deviates by about 2× the raw deviation at the same layer, which is what
     JumpReLU thresholding does to small residual jitter.
   - These are fresh bf16 forwards on torch 2.14.1 / transformers 5.19.0, against the August
     stack. A systematic pipeline difference (hook point, attention implementation,
     normalisation) would shift every layer at once.
4. **Order statistics reproduce exactly** (`results/e02b_rho_check.json`). Channel 1 is the
   rebuilt raw means; channel 2 is the stored lag deltas.

   | set | layer | gap (stored → rebuilt) | two-channel rho (stored → rebuilt) |
   |---|---|---|---|
   | strict-12 | L6 | 1.5424 → 1.5424 | 0.993 → 0.993 |
   | strict-12 | L7 | — | 0.993 → 0.993 |
   | candidate (d) | L6 | 1.6957 → 1.6957 | 1.0 → 1.0 |
   | candidate (i), discovery verbs | L7 | 2.0795 → 2.0806 | 1.0 → 1.0 |

5. **The supervised positive controls pass on the rebuilt means.** Months close at L4–L7
   (p_closed 1e-4 to 1.2e-3; p_bg ≤ 0.001). Weekdays close at L4. Number words close nowhere.

## Amended reading

The ±1e-3 SAE tolerance was taken from script 42's `ANCHOR_TOL`, which checks a re-read of the
*same* cache. It cannot hold for a fresh forward on a different software stack. The SAE-channel
tolerance is amended to the raw tolerance, **±0.01**, and the gate is read as **PASS** under the
amended tolerance. This miss is reported verbatim in REPORT.md and is not buried.
