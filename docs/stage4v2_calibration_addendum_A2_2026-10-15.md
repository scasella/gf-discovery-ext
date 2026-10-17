# Stage 4 v2, calibration addendum A2: changes after the full pilot (2026-10-15)

Written after:
- the full P2 pilot (`results/e63_pilot.json`, `results/e63_pilot_choice.json`);
- the P2-only diagnostics (`scripts/e63d_pilot_diagnostics.py` → `results/e63_pilot_diagnostics.json`);
- one P2 context inspection of the L7 paren unit, run in the scratchpad.

It was also written before `PREREG_occurrence_v2.md` was frozen and before any D2 data existed.

**None of these changes alters the pilot choice.** It is checked below that the new screen
excludes no P2 unit at the chosen setting.

## 1. ICA fit retry rule

**Seen.**
- In e63d, the skew-contrast FastICA diverged to NaN on the B1 input (no per-type step) at L24.
- It never diverged on the chosen representation in the pilot.

**Rule.**
- Each ICA fit tries seeds s, s+100 and s+200, and keeps the first fit that is finite.
- If the main fit fails on all three seeds, that unit kind has no units at that layer, and the
  failure is reported.
- If a split-half fit fails, the reproducibility of every unit of that kind is 0.

## 2. A second-moment identity screen for directional units: η²(token) of y² ≤ 0.30

**Seen.** The registered η²(token) of the projection y is vacuous by construction. Type means are
removed exactly, so every linear projection has η² ≈ 0:
- below 1e-4 on B1;
- about 1e-15 after the per-type step;
- in v1 it was likewise near 0.

The second-moment analogue is η²(token) on y², the share of the variance of y² that token type
explains.

| P2 ICA units (logcosh and skew) | max η²(y²) | median η²(y²) |
|---|---|---|
| B1, no per-type step | 0.47 | 0.08–0.11 |
| chosen setting | 0.148 (L7 logcosh); every other layer and kind ≤ 0.125 | 0.032–0.037 |

**Rule.**
- The directional screen becomes: η²(y) ≤ 0.30, η²(y²) ≤ 0.30, split-half |cos| ≥ 0.8, and
  bimodality off.
- On P2 at the chosen setting the new criterion excludes **no** unit, so the pilot choice is
  unchanged.
- It is not tuned to any label. It gives ICA units a working identity screen, in the role TDI
  plays for clusters.

## 3. Tie-break for K

**Seen.** Every K ∈ {16, 32, 64, 128} tied on:
- layers recovered (all recoveries come from ICA units);
- strictness;
- mean margin.

**Rule.** `max()` returned the first grid entry, **K = 16**, and that is frozen.

**Measured.** On P2, **0 of 16 clusters pass split-half reproducibility ≥ 0.8 at any layer**. After
the per-type step the cloud is close to isotropic and k-means partitions are not reproducible. The
cluster kind is therefore expected to contribute no screened unit. This is reported, not changed.

## 4. Specificity: reported for every recovering unit on D2, never a gate

**Seen (P2, L7).**
- The best paren unit (logcosh #15, AUROC 0.774) has as its **top** battery association
  bracket_inside (0.841). That is a feature with a 0.10% base rate.
- Its top 2,000 rows are 61% paren-inside (base 3.0%), 22% quote-inside (base 5.5%) and 2.6%
  bracket-inside.
- Sampled contexts are asides inside parentheses or quotation marks, including curly single
  quotes, which the battery's quote label does not count.
- It is an **enclosure-state** direction, dominated by paren.

**Seen (P2, L12).**
- Best paren unit (logcosh #10): bracket 0.733, paren 0.718.
- Best quote unit (#28): specific, with quote_inside its top association at 0.767.

**Rule.** No specificity gate. A rule written now, after seeing this table, would be built to pass
or to fail this unit. For every recovering unit on D2, the report gives:
- the full battery association table, with base rates;
- the label composition of its top 1% of rows.

## 5. Ablation (P2, reported only)

Max screened |AUROC| for paren / quote, ICA-64 logcosh:

| layer | B1 (no per-type step) | chosen per-type whitening |
|---|---|---|
| L7 | 0.774 / 0.659 | 0.774 / 0.662 |
| L12 | 0.604 / 0.764 | 0.718 / 0.767 |
| L24 | 0.573 / 0.569 | 0.554 / 0.567 |

- **At L7 the recovery comes from the ICA-dimension fix (addendum A1) alone.**
- Per-type whitening is what moves the screened paren unit at L12 from 0.604 to 0.718.
- It is also what removes identity from the geometry. The table below is P2, median, B1 → chosen.

| layer | TDI | KIR | ICA η²(y²) |
|---|---|---|---|
| L7 | 6.8 → 1.5 | 38.6 → 14.3 | about 0.10 → 0.035 |
| L12 | 5.0 → 1.5 | 29.5 → 9.0 | — |
| L24 | 5.2 → 1.7 | 25.6 → 9.1 | — |

## 6. L24 floor

- The plan's rule, clip(0.5 + 0.75·(probe − 0.5), 0.70, 0.85), gives 0.70 at every layer.
- At L24 the P2 probe ceiling is 0.653, so under that floor L24 is not expected to recover paren.
- The rule was declared before any P2 number. It is kept, and PASS needs only one layer.

## 7. Code changes with no analysis effect

- `--test-dir`, a dry-run redirect for the prereg, hashes and outputs, used once on P2 L24.
  It reproduced the pilot's L24 numbers exactly.
- The permutation count is wired explicitly to `N_PERM` (1,000).
- Empty-kind guards were added.
- An η²(y²) pass count was added to the per-screen summary.
