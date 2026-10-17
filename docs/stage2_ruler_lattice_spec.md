# Stage 2 — script 46 (ruler / lattice): specification, frozen before any blind run

Source spec. `phase4_prereg_2026-08-22.md` §2.5 points to the Phase-4 plan's §4. That plan file
no longer exists. Its text was recovered verbatim from the 2026-08-22 session transcript and is
quoted in §0 below. Where the original is silent, every choice below is fixed **before** the
blind run and **after** only these exposures:
- the supervised controls in §4;
- the planted curves in §5;
- the null runs in §6.

The blind run then executes this file unchanged. Its sha256 is recorded in `REPORT.md`.

## 0. The original spec (Phase-4 plan §4, recovered)

> Object: direction u and population S (discovered as 1-D KDE modes of the projection) whose
> levels are reproducibly ordered **and more evenly spaced than chance**
> (max-spacing/mean-spacing — the linear analogue of max-gap). Not "a direction that sorts":
> log-count and norm sort everything at rho ≈ 1.
> Tiers (conjunction required): B order rho > 99th percentile of matched sets ∧ partial rho vs
> log-count / norm / mean-position / token-class < 0.3 ∧ spacing regularity below 1st
> percentile ∧ same level assignment L5→L9. Whole path on class-permuted data is mandatory.
> Controls: months recovered by the same code path with periodicity free; SAE arm L8/L9 fail.
> Plant: k-level lattice over amplitude × jitter. First experiment: spacing regularity on top-d
> deflated directions at L6/L7 with the class-permuted FPR alongside; FPR > 5% ⇒ kill.
> Pre-registrable: *A direction u_A carrying ≥ 6 discovered levels with half-B spacing
> regularity below the 1st percentile, nuisance partial rho < 0.3, and successor log-probs rising
> in level order under tangent steering in ≥ k−1 of k steps.*

## 1. Representation

- **Raw residual token-type means** over the identical 4,002,480-row corpus (35b construction,
  rebuilt in pass 1), split into halves:
  - **A** = even sequence ids;
  - **B** = odd sequence ids (R9).
- Layers: blind run at L5–L9 and at L12, 14, 16, 18, 20, 22, 24.
- **Outlier-coordinate drop** (R10/Addendum H):
  - L3–L10: drop {1038};
  - deep layers: the set computed by the same split-half-share rule in `e42d`.
- **Nuisance deflation**:
  - Regress every coordinate of the half-A means on Z = [1, log10 count, ‖full mean‖,
    mean document position, lexical-class one-hots (leading-space × capitalisation, 4 blocks)].
  - Fit on A; subtract the same fitted map from A and from B.
  - Centre on A's pool mean.

## 2. Proposer: label-free, the line analogue of the ring assemblies

For each layer, each k ∈ {6, 8, 10, 12} and each of the 3,795 seeds:
- S = seed plus its nearest neighbours by cosine on the deflated A coordinates.
- Duplicate guard R15 (0.90): a candidate with cosine > 0.90 to a current member replaces that
  member only if it has the higher count. Otherwise it is skipped. Fill continues in
  neighbour order until |S| = k.
- Unique sets only.

The original spec's "1-D KDE modes of the projection" is a direction-first route. It has
measured zero power for local structure: a planted local ruler cannot move a global direction.
It is replaced by this set-first route, the same move Phase 5 made for rings. This is a declared
deviation.

## 3. Statistics per set S (k members)

| statistic | definition |
|---|---|
| axis | u_A = PC1 of S's centred A means. pA = (A_S − mean)·u_A, pB = (B_S − mean)·u_A. |
| order_B | Spearman(pA, pB): the ORDER tier. Known to be weak, because split-half type means are stable. |
| spacing_reg_B | Order members by pA; take consecutive differences d of pB in that order. If all d > 0: max(d)/mean(d), where 1.0 is perfectly even. Otherwise +∞. This is the tier that carries the claim, the linear analogue of max-gap. |
| line_frac_B | PC1 variance fraction of S's B means. Reported only. |
| grid_B (lattice branch) | In S's top-2 PCA plane fitted on A: count disjoint pair-pairs whose difference vectors coincide within τ = 0.25 × the plane's median nearest-neighbour distance (parallelograms), on B, divided by the number of disjoint pair-pairs. |
| grid_repro | Fraction of A-parallelograms that persist on B. |

## 4. Tiers and PASS

Backgrounds are taken per (layer, k): the cell is the whole population of unique proposals at
that cell (cluster-like matched sets). Random k-sets are reported alongside.

| tier | rule |
|---|---|
| T1 order | order_B ≥ cell p99. Ties at 1.0 count as passing; saturation is reported. |
| T2 spacing | p_bg(spacing_reg_B) ≤ 0.01, the cell's 1st percentile. Infinite (non-monotone) values never pass. |
| T3 nuisance | \|Spearman(pA, v)\| < 0.3 for v ∈ {log10 count, ‖mean‖, mean position}. Also, for every lexical block holding 1…k−1 members, \|Spearman(pA, indicator)\| < 0.3. |
| T4 persistence | At every scanned adjacent layer (L5–L9: L±1; deep: ±2 within the deep list), the A-order of S on that layer's own axis agrees with layer L's (\|Spearman\| ≥ 0.9), and spacing_reg_B there has p_bg ≤ 0.05 in that layer's cell. |

**RULER PASS = T1 ∧ T2 ∧ T3 ∧ T4.**

**LATTICE PASS** (k ∈ {8, 9, 12} only) =
- grid_B ≥ cell p99, and
- grid_repro ≥ 0.8, and
- T3, and
- grid_B ≥ cell p95 at every adjacent layer.

Leave-one-out is **not** a tier for rulers. Deleting an interior member of an evenly spaced
ruler doubles one gap by construction, which is the same defect the ring LOO had (Phase 4,
script 42 preflight).

## 5. Calibration (all before the blind run)

1. **Supervised positive controls.** The set is chosen by name and statistics are computed on
   the set itself.
   - Pool-level: number words ` one`…` ten` (k = 10). Kinship and rank terms present in the
     pool, reported with how many exist.
   - Templated-level (`e46s`): ranks (8), number words (10) and kinship (13), each read at the
     word's last token. There are 16 neutral templates, split 8/8 into A and B. The null is 300
     random pool words through the same templates.
   - Ruler controls must give spacing p_bg ≤ 0.01 *if the representation holds evenly spaced
     ordinal structure*. A failure is reported as a limit of the tier, never tuned away.
   - Kinship is the lattice control (generation × lineality/gender).
2. **Planted power.**
   - An 8-level ruler is planted into 8 random carriers, replacing their A means, at offset
     f × the median offset of random 8-sets (f ∈ {1, 2, 3}).
   - Spacing s equals the median nearest-neighbour distance. Off-line scatter has norm 1 × s,
     and along-axis jitter has sd j × s (j ∈ {0, 0.05, 0.1, 0.2, 0.3}).
   - B = plant + the carriers' own real B−A noise.
   - 50 plants per cell.
   - Recall = some proposal holding ≥ 7 of the 8 carriers passes T1 ∧ T2 ∧ T3 (single layer).
   - The lattice gets the same treatment with a 3 × 3 grid.
3. **Null false-positive rate.**
   - (a) **Gaussian matched null pool**: same covariance as the deflated A means, and B = null +
     real B−A noise permuted within count deciles. The full proposer and T1–T3 run with the
     null pool's own cell thresholds. The result is the expected false passers per layer.
   - (b) **Class-permuted** (mandatory per the original spec): B rows are permuted within
     (lexical block × count decile). T1 ∧ T2 is adjudicated against the real cell thresholds.
     **Per-proposal FPR > 5% ⇒ kill the instrument.**

## 6. After the blind run

- Data passers are compared with the expected false passers.
- Candidates with a member count of at least 6 go, ranked by p_bg:
  - to **fresh-context geometric replication**: type means re-estimated on the 8,000 held-out
    reserve sequences, where T1 and T2 must hold against the reserve pool's own cell;
  - and then to the **behavioural tier**. This is 47's rig on reserve contexts, with Δ_i =
    [logit(next level) − logit(previous level)] steered minus unsteered, for interior members.
    Every null is at the same 40-prompt budget:
    - (i) 10 random orthogonal directions;
    - (iii) 50 random k-sets, each with its own axis;
    - (iv) sibling sets, i.e. other T1 ∧ T2 proposals of the same cell.

    PASS = sign consistency ≥ (n−1)/n over the n interior members, and mean Δ@0.02 above the
    (iii) p99, the (iv) p99 and the (i) maximum.
- Nothing unnamed is called a discovery without geometric nulls, behavioural pass and fresh
  replication.
