# PREREG — Stage 4 v2: identity removed from the first and second moments (FROZEN 2026-10-15)

Frozen before any D2 data existed: `cache/occ/occD2_*` had not been extracted when this file's
sha256 was appended to `docs/HASHES.txt`. `scripts/e63_occurrence_v2.py` asserts, at every phase:
- that this file's sha256 is in `docs/HASHES.txt`;
- that the pilot choice file's sha256 (§8) appears in this text.

## 0. What was seen before freezing

**v1, all of it:**
- the P, D and S statistics;
- the R extraction (never analysed);
- `PREREG_occurrence.md`, Addendum S1 and REPORT's Stage 4.

**P2 only, for v2:**
1. `docs/stage4v2_calibration_plan_2026-10-15.md`, hashed before any v2 statistic.
2. A reduced-grid smoke run at L7, then addendum A1 (ICA dimension), hashed before the full
   pilot.
3. The full pilot (`results/e63_pilot.json`).
4. The P2 diagnostics (`results/e63_pilot_diagnostics.json`: ablation, η²(y²), specificity), one
   context inspection of the L7 paren unit, then addendum A2, hashed before this file.
5. Two dry runs of the freeze and control code on P2 (L24, then L7) with 20 permutations. Both
   reproduced the pilot's numbers.

No D2 or R2 statistic exists.

## 1. Question and what v2 changes

The question is v1 §1: is there structure among single occurrences once token identity is
removed? This run asks only the gate question: **does the v2 instrument blindly recover a known
in-context variable on D2?**

**Changes at v1's measured failure points:**

| v1 failure point | v2 change |
|---|---|
| Identity left in the second moment (cluster TDI 7–8.5×) | Per-type covariance equalisation with shrinkage, after iterated first-moment removal. |
| The TDI statistic counted self-matches | It now samples among the *other* same-type occurrences. On P2, v1's representation gives a self-excluded TDI of 5.8 / 4.7 / 5.4 at L7 / L12 / L24, against the self-inclusive 7.7 / 7.1 / 7.4. |
| ICA on a noise-chosen subspace: FastICA with n_components < dim on whitened input (addendum A1); v1 split-half reproducibility 0.37–0.39 | Full-rank ICA (64 components) with two contrasts: logcosh, and skewness for sparse binary sources. |
| Floors that could not be met: cluster F1 capped at about 0.41 by size at K = 128; ICA AUROC floor 0.85 above the probe ceiling of 0.83 | Smaller K (16); a probe-referenced AUROC floor (§6). |
| A vacuous ICA identity screen: η²(token) of y is about 0 by construction | η²(token) of y² ≤ 0.30 (addendum A2). |

**P2 ablation (A2 §5).** It is stated so the D2 verdict is not over-read.
- At L7, paren recovery on P2 comes from **the ICA-dimension fix alone**: without the per-type
  step the score is also 0.774.
- Per-type whitening moves screened paren at L12 from 0.604 to 0.718.
- It also removes identity from the geometry. P2 medians, from B1 (iterated first moments only)
  to the chosen per-type whitening:
  - TDI 6.8/5.0/5.2 → 1.5/1.5/1.7;
  - KIR 38.6/29.5/25.6 → 14.3/9.0/9.1;
  - ICA η²(y²) about 0.10 → 0.035.

**P2 specificity (A2 §4).** The L7 paren unit is an **enclosure-state** direction.
- Its top 1% is 61% paren-inside and 22% quote-inside.
- Its top battery association is bracket_inside, a feature with a 0.1% base rate.

## 2. Layers

**L7, L12, L24.** This is v1 §2's rule, unchanged.

## 3. Data (1,000 sequences × 255 tokens each, disjoint from v1's P/D/S/R)

| set | content | role |
|---|---|---|
| P2 | corpus 13,000–13,999 | calibration (done) |
| D2 | corpus 1,000–1,999 | units, screens and positive control |
| R2 | reserve 2,000–2,999 | replication; **not extracted or used in this run** |

D2 is extracted by `scripts/e60_extract.py --source corpus --seq-start 1000 --n-seq 1000
--layers 7,12,24 --tag occD2`.

## 4. Representation (D2's own rows only; frozen)

1. **Rows:** positions ≥ 4, types with ≥ 8 rows in D2. Outlier coordinates are dropped by the v1
   rule (`e61.drop_for`): L7 {1038}, L12 {1038}, L24 none.
2. **First moment:** type, document and position means are subtracted in turn, and the cycle is
   repeated 3 times.
3. **PCA-64,** fitted on D2 (every third row), then whitened.
4. **Per-type equalisation:**
   - types with n_t ≥ **N_MIN = 32** use
     Σ̂_t = (1−λ_t)·S_t + λ_t·S_pool, with λ_t = **16**/(n_t + 16);
   - rarer types use S_pool;
   - each row becomes (z − μ_t)·Σ̂_t^(−1/2), followed by a global re-whitening to 64 dimensions.
   - The same setting (N_MIN 32, n0 16) was chosen independently at all three layers. It is the
     lowest TDI under the probe guard (plan, Step A).

## 5. Units and screens (frozen; K and the tie-break per A2 §3)

| unit kind | number | screen |
|---|---|---|
| k-means (MiniBatchKMeans) | K = **16** | size ≥ 300; ≥ 30 types; largest type share ≤ 0.30; ≥ 100 documents; split-half reproducibility ≥ **0.8**; self-excluded TDI ≤ **3** |
| FastICA, logcosh contrast | **64** | η²(token) of y ≤ 0.30; **η²(token) of y² ≤ 0.30**; split-half \|cos\| ≥ **0.8**; bimodality **off** |
| FastICA, skewness contrast (G(u) = u³/3) | **64** | same as logcosh |

- **ICA fitting:** each ICA kind is fitted on every second row.
- **Split halves:** units are refitted on the even- and odd-sequence halves, with seeds 1 and 2.
  Reproducibility is the smaller of the two matches.
- **Failed fits (A2 §1):** each fit tries seeds s, s+100 and s+200.
  - If the main fit fails on all three, that kind has no units.
  - If a half fit fails, every unit of that kind has reproducibility 0.

**Freeze.** `--phase freeze` writes `results/occ_units_v2_L{L}.npz` and `results/e63_occurrence_v2.json`
before any label is computed. The JSON records the npz sha256 and an md5 of the frozen block.
`--phase control` asserts the npz sha256.

## 6. Positive control (`--phase control`)

**Labels:** paren_inside and quote_inside, the v1 battery definitions (`e61.feature_battery`).

**Statistics, max over screened units:**
- F1 for clusters;
- |AUROC| for each ICA kind.

**Null:** token-matched label permutation (labels permuted among occurrences of the same type),
**1,000 permutations**, p99.9 of the max-over-screened-units statistic. The same permutations are
used for both labels.

**Floors (frozen from P2 by the plan's rule):**
- Cluster: F1 ≥ 0.50.
- ICA: AUROC ≥ clip(0.5 + 0.75·(probe_P2 − 0.5), 0.70, 0.85).

| layer | P2 paren probe (held-out) | ICA AUROC floor |
|---|---|---|
| L7 | 0.753 | **0.70** |
| L12 | 0.734 | **0.70** |
| L24 | 0.653 | **0.70** (above the L24 probe ceiling; L24 is not expected to recover paren — A2 §6) |

**Looks:** 2 labels × 3 unit kinds × 3 layers = 18 looks at p99.9, so the family-wise false
recovery is ≤ 1.8%.

**Recovered.** A label is recovered at layer L iff some unit kind's statistic is ≥ its floor
**and** above its null p99.9.

**PASS** iff either label is recovered at one or more layers. The instrument counts as validated
only at the layers (and for the unit kinds) that recover.

**FAIL** iff no label is recovered at any layer. Stage 4 v2 then stops with "instrument not
validated", and no unit is labelled or claimed.

**paren_inside was tuned on P2. quote_inside was never used in any choice.** On P2 under this
configuration:

| layer | paren | quote |
|---|---|---|
| L7 | recovered | not recovered |
| L12 | recovered | recovered |
| L24 | neither | neither |

**Reported, not gating:**
- the held-out D2 probe AUROC for each label;
- the unscreened maxima for each kind;
- TDI (self-excluded, and v1's self-inclusive) and KIR;
- per-screen pass counts;
- for every recovering unit: the full battery association table with base rates, the top-1%
  label composition, and 12 sampled contexts.

## 7. After a PASS (not run in this session; needs its own go-ahead)

- S2 = D2's sequences shuffled within each sequence (`e60 --shuffle`).
- Arrangements, labelling, behavioural tier and replication follow v1 §5 and §8–§9 on the v2
  representation, at the validated layers only, with R2 in place of R.
- Behavioural contexts come from reserve 3,000–3,999.

## 8. Integrity (sha256 at freeze)

| file | sha256 |
|---|---|
| `scripts/e63_occurrence_v2.py` | 6f822a439025e01b176ae446d96413dc736078d5dbe740d03f0f05052f6e4543 |
| `scripts/e61_occurrence_screen.py` (imported helpers) | 43a445016e82735c70ca6f2c623917fa864fce6920317832c9f2da9035bf92e1 |
| `scripts/e60_extract.py` | a6d99f92f1e21d613e80cceb594724dee9faed719131e5fc372d7e4f113cdd20 |
| `results/e63_pilot_choice.json` | 4c71cd1b25166465b7a776a66e828923026c78c303b31dd996b521faf589b5fa |
| `results/e63_pilot.json` | ad5d1a3ce1d43aad38fb410c9902da5a1faf3e5743f1cd6d8ad5f2b99bfc6cf7 |
| `results/e63_pilot_diagnostics.json` | f3e529bef1b91b8346a527ec6d3c67b8dbef8cd2867485d66ea2bf6766d46cb4 |
| `docs/stage4v2_calibration_plan_2026-10-15.md` | 05e19ea3b518f34fff48050256dfe1622337ba399167fa2a16661470911aee09 |
| addendum A1 | d9984318dfd0e4f764b7e67ae997eaf6173d4599d919166f6370665011d1b78f |
| addendum A2 | 13df0c7d284e75f52d639f8f21ae6f9409e96520b5aeb476d0e504b50aeda847 |

**Commands:**
- `e63_occurrence_v2.py --phase freeze --layers 7,12,24`
- then `--phase control --layers 7,12,24`

Both run with HF_HOME=hf_home, offline.
