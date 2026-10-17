# PREREG (DRAFT) — Stage 4: structure among identity-removed occurrences

DRAFT. The layer choice is filled in from Stage 1 by the rule in §2. The frozen copy goes to
`PREREG_occurrence.md` at the project root. Its sha256 is recorded in `docs/HASHES.txt` before
any blind run, and e61 asserts that hash at every phase.

## 1. Question

Is there structure among single occurrences, once token identity is removed, that no one has
named? That is where in-context variables would live: nesting, list position, counting,
quotation state. Prior per-occurrence work (scripts 39 and 49) never removed identity.

## 2. Layers

- Three layers. L7 is the anchor, where the 47 rig and the two-channel tier are validated.
- The other two are deep layers, chosen by a rule applied to the Stage-1 output before any
  occurrence data exists:
  - the deep layer with the most Stage-1 closure flags, and
  - the deep layer furthest from it in the deep list.
  - Ties go to the shallower layer.

## 3. Data (disjoint, 1,000 sequences × 255 tokens each)

| set | content |
|---|---|
| P (pilot) | corpus sequences 14,000–14,999; calibration on the positive control only |
| D (discovery) | corpus sequences 0–999 |
| S (context-shuffled null) | D's sequences with tokens permuted within each sequence |
| R (replication) | reserve sequences 0–999 |

## 4. Representation (per set, using only that set's rows)

- v = SAE-decoded residual (l0_big 16k; fp16 codes @ W_dec), minus the set's own type mean.
  Only types with at least 8 rows are kept, and positions < 4 are excluded.
- Outlier coordinates are dropped (R10 / the Addendum-H set).
- Document centring, then position deflation, which removes the BOS-distance code.
- PCA to 64 dimensions fitted on D, whitened, and applied to S and R.

## 5. Units and blind screens

- MiniBatchKMeans, K = 128.
- FastICA, 32 components.

| unit | screen |
|---|---|
| cluster | size ≥ 300; ≥ 30 types; largest type share ≤ 0.30; ≥ 100 sequences; split-half reproducibility ≥ 0.80; token-driven index (TDI) ≤ 3 |
| ICA component | bimodality coefficient > 0.555; η²(token) ≤ 0.30; split-half \|cos\| ≥ 0.80 |

- Context-shuffled presence (cosine to S's units) is reported for every unit. A unit with
  presence ≥ 0.8 is labelled "context-generic", which is not a disqualification.
- **Arrangements** among screened centroids: rings (closure p < 0.05 and p_bg ≤ 0.05) and
  rulers (spacing p_bg ≤ 0.01 and order_B ≥ 0.9). The false-flag count comes from S through
  identical code.
- **Freeze**: units and screens are hashed before any label is computed.

## 6. Positive control

Labels:
- paren_inside: the running count of "(" minus ")" is at least 1;
- quote_inside: an odd number of double quotes so far.

Statistics: max F1 over screened clusters, and max AUROC over screened ICA components.

Null: token-matched label permutation. Labels are permuted among occurrences of the same token
type, 1,000 permutations, and the threshold is the **p99.9** of the max-over-units statistic.

There are 12 looks: 2 labels × 2 unit kinds × 3 layers. At p99.9 each, the family-wise false
recovery rate is about 1.2%.

**Recovered** iff, for either label at some layer:
- F1 ≥ 0.50 and above the null p99.9, or
- AUROC ≥ 0.85 and above the null p99.9.

If neither is recovered, Stage 4 stops: instrument not validated, and no unnamed unit is
claimed.

## 7. Pilot calibration (P only, at L7 only)

- The pilot runs at the anchor layer **L7 only**. The chosen setting is frozen for all three
  layers.
- Reason: per the user's ordering, the pilot runs before the deep layers are fixed by §2's rule.
  §2 needs script 22 on L20–24.
- Grid: K ∈ {64, 128, 256}, N_ICA ∈ {16, 32}, PCA dimensions ∈ {32, 64}.
- The setting with the best paren_inside recovery on P is frozen. The choice is first by the
  number of layers recovered, then by the mean margin above the null.
- If no setting recovers, the §5 defaults are used, and that is reported.
- **paren_inside is therefore tuned on P.** On D it is an out-of-sample check under settings
  chosen for it, not a blind validation.
- **quote_inside is the untuned secondary control** and is never used in the choice.

## 8. Labelling

For every screened unit:
- the named-feature battery:
  - paren, bracket and quote inside;
  - line position; sentence position;
  - after newline; list item; after colon;
  - in URL; after digit; caps run;
  - previous token punctuation; previous token capitalised;
  - word continuation (no leading space);
  - next token punctuation, newline or continuation;
  - code/markup within 8 tokens;
- its commonest token types;
- 12 sampled contexts.

A unit is NAMED if:
- a battery feature reaches enrichment ≥ 3×, AUROC ≥ 0.75, or \|ρ\| ≥ 0.5; or
- the contexts show one consistent named construction.

Otherwise it is UNNAMED.

## 9. Tiers (unnamed units)

1. **Geometric candidate**: passes every §5 screen. For an arrangement, it is flagged above S's
   count.
2. **Behavioural pass**, run with contexts from **reserve sequences 1000–1999**. R (reserve
   0–999) stays untouched until replication. Stage 3 samples 8-token slices from all 8,000
   reserve sequences, but it tests a type-mean object and selects no Stage-4 unit.
   - Steering: h += ε‖h‖·d̂ at occurrences outside the unit.
   - Readout: Δ on the unit's 20 signature next tokens against a frequency-matched random set.
   - Nulls, all at the same 400 occurrences: 10 random orthogonal directions; the sibling
     units.
   - PASS: sign consistency ≥ 0.8, and Δ above the sibling p99 and the random maximum.
   - The positive-control unit is the behavioural anchor and is run first.
3. **Replicated**: an independent fit on R finds a match (cosine ≥ 0.8), and the screens hold.

Discovery = all three tiers.

## 10. Known weak screens (stated before the run)

Split-half reproducibility and context-shuffled presence will pass almost every unit:
independent k-means fits on a dense cloud always have a nearby centroid. The screens that do
real work are the token-driven index and type diversity. A high pass rate on the other two is
not evidence of anything.

## 11. Multiplicity

- Expected false behavioural passes = number of units tested × 0.01.
- Expected false arrangement flags = S's count.
