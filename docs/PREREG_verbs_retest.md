# PREREG — Stage 3: confirmatory re-test of the "discovery verbs" near-miss

Frozen 2026-10-15, before any forward pass of the re-test. Run by `scripts/e53_verbs_retest.py`,
which writes this file's sha256 into its output stub and asserts it unchanged at the end.

## Object

Candidate (i) of Phase 5, layer 7, k = 10, member_hash `2eff8657fbd3`. Members: `find`, `showed`,
`discover`, `identified`, `identify`, `revealed`, `confirmed`, `recognized`, `realize`,
`noticed` (all with a leading space). It is the follow-up Addendum R defined and did not run:
"fresh contexts, matched prompt budgets, full-budget sibling null, pre-registered".

## What changes relative to scripts 47/47b/51

| | 47b / 51 (Aug) | this re-test |
|---|---|---|
| contexts | 8-token slices of the 4M corpus | 8-token slices of the **8,000 reserve sequences** (stream positions 15,697–23,696; never used by any type mean, proposal or test) |
| candidate prompts | 40 | 40 |
| null prompts | 12 | **40, every arm** |
| null (iii) random k-sets | 50 | **100** (candidate), 50 (anchor) |
| null (iv) siblings | 30 at 12 prompts, 1 eps | **every sibling in the route family at 40 prompts and the full eps grid** |
| seeds | 5100–5104 | 5300–5303 |

The geometry is unchanged. Plane, circle, angles and tangents come from the raw L7 type means
(R10 drop of 1038), using 51's `plane_of`, on the rebuilt fulln artifact. The rebuilt
artifact's gap_ratio and two-channel rho are re-checked against the stored values and
reported.

The sibling family is rebuilt with 51's own code along the candidate's own route:
- channel-2 lag+1 delta cloud at L7, seed plus 15 nearest, giving k = 16;
- R15 dedup at 0.92, then the P1 guard;
- keep sets of exactly 10 members that close (gap_ratio < 2.5);
- remove the candidate itself.

Overlap with 51's 30 stored sibling hashes is reported.

## Rules (fixed now)

- Δ_i = [logit(succ) − logit(pred)] steered minus the same unsteered. Steering is h ← h + ε‖h‖
  t̂_i at resid_post L7, at the member's own position. ε grid (0.005, 0.01, 0.02); null ε =
  0.02.
- **Anchor gate.** strict-12 months at L6 must PASS:
  - sign consistency ≥ 11/12 at some ε;
  - mean Δ@0.02 > the p99 of null (iii) (`np.percentile(..., 99, method="higher")`);
  - mean Δ@0.02 > the maximum of null (i) (10 random directions orthogonal to each tangent).

  If the anchor fails, the run stops and reports.
- **Candidate PASS** requires all of:
  - sign consistency ≥ 9/10 at some ε;
  - mean Δ@0.02 > p99 of null (iii);
  - mean Δ@0.02 > max of null (i);
  - mean Δ@0.02 > p99 of null (iv), the full sibling family.

  With 100 and 30 draws the "higher" p99 is the maximum of each null.
- Reported, not gating:
  - the candidate's empirical rank p against the siblings;
  - realised false rates, i.e. the fraction of random and sibling sets that would pass the
    sign-consistency and (iii) conjuncts themselves;
  - monotonicity in ε;
  - the (ii) rotations.

## Readings fixed in advance

- **PASS**: the behavioural tier is confirmed on fresh contexts at a matched budget. The
  candidate still fails the geometric leave-one-out audit (Phase 5: LOO rho_min 0.517). By the
  campaign's naming clause, and by this extension's discovery criteria (geometric nulls,
  behavioural pass and fresh replication all required), it remains **not a discovery**. It is
  recorded as "behavioural pass, geometric LOO failure".
- **FAIL**: retired. The Phase-5 behavioural pass is read as a context/budget artifact.
- **Anchor fails**: no verdict on the candidate.
