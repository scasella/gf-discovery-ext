"""Circle and helix fits, plus the cyclic-ordering statistic for the Phase-1 gate.

Two things are being measured and they must not be conflated:

  1. *Is the point cloud low-dimensional and circular?*  -- unsupervised, no labels.
     Reported as the top-2 PC variance fraction plus the normalised residual from
     a fitted circle.
  2. *Does the recovered angle order agree with the calendar?* -- supervised, and
     only ever evaluated after (1). A circle has no privileged origin or handedness,
     so the honest statistic maximises agreement over the 2k rotation/reflection
     symmetries of a k-cycle. That maximisation is a selection effect, so the
     p-value is calibrated by a permutation null that applies the same maximisation.
"""
import numpy as np


def degenerate(points, rel_tol=1e-6):
    """True if the point set carries no usable geometry.

    Guards the case above: points that are identical (or effectively so) have no
    meaningful angular order, so any ordering statistic computed on them is noise
    at best and an argsort artifact at worst.
    """
    X = np.asarray(points, dtype=np.float64)
    c = X - X.mean(0, keepdims=True)
    scale = np.linalg.norm(X, axis=1).mean()
    spread = np.linalg.norm(c, axis=1)
    if scale <= 0 or spread.max() <= rel_tol * max(scale, 1e-12):
        return True
    # also reject when a single point supplies essentially all the spread
    return bool((spread > rel_tol * scale).sum() < 3)


def pca_2d(points):
    X = np.asarray(points, dtype=np.float64)
    X = X - X.mean(0, keepdims=True)
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    var = S ** 2
    ratios = var / max(var.sum(), 1e-12)
    return X @ Vt[:2].T, ratios, Vt


def fit_circle(xy):
    """Algebraic (Kasa) circle fit. Returns (cx, cy, r, normalised RMS residual).

    The residual is divided by r so it is comparable across clusters of different scale.
    """
    P = np.asarray(xy, dtype=np.float64)
    x, y = P[:, 0], P[:, 1]
    A = np.column_stack([2 * x, 2 * y, np.ones(len(x))])
    b = x ** 2 + y ** 2
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    cx, cy = sol[0], sol[1]
    r = np.sqrt(max(sol[2] + cx ** 2 + cy ** 2, 1e-12))
    d = np.hypot(x - cx, y - cy)
    return cx, cy, r, float(np.sqrt(np.mean((d - r) ** 2)) / max(r, 1e-12))


def circularity(xy, cx, cy, r):
    """Does this point set actually close into a ring, or is it a line/arc?

    The normalised circle residual alone cannot tell: the Kasa fit is unbounded in
    radius, so a collinear set is fitted by an enormous circle and resid/r -> 0.
    A regular heptagon and a near-collinear set both score 0.0000. Likewise the
    cyclic-order statistic maximises over rotations including the identity, so a
    monotone 1-D arrangement scores rho = 1.000 -- it detects ORDER, not closure.

    Two quantities separate them:
      radius_ratio -- fitted radius over the points' own spread. ~1 for a ring,
                      huge for a line (the fit escapes to infinity).
      gap_ratio    -- largest angular gap over the 2*pi/k expected of k evenly
                      spaced points. ~1 for a closed cycle, >>1 for an open arc.
    """
    P = np.asarray(xy, dtype=np.float64)
    spread = float(np.linalg.norm(P - P.mean(0), axis=1).mean())
    a = np.sort(np.arctan2(P[:, 1] - cy, P[:, 0] - cx))
    gaps = np.diff(np.concatenate([a, [a[0] + 2 * np.pi]]))
    k = len(P)
    return dict(radius_ratio=float(r / max(spread, 1e-12)),
                gap_ratio=float(gaps.max() / (2 * np.pi / k)),
                max_gap_deg=float(np.degrees(gaps.max())),
                spread=spread, radius=float(r))


def is_ring(circ, max_radius_ratio=3.0, max_gap_ratio=2.5):
    """A conservative closure test. Both must hold for the word 'circle' to be used."""
    return bool(circ["radius_ratio"] <= max_radius_ratio and circ["gap_ratio"] <= max_gap_ratio)


def angles(xy, cx=0.0, cy=0.0):
    P = np.asarray(xy, dtype=np.float64)
    return np.arctan2(P[:, 1] - cy, P[:, 0] - cx)


def _spearman_perm(a, b, n):
    """Spearman for two tie-free rank permutations: 1 - 6*sum(d^2)/(n(n^2-1)).

    The gate evaluates this ~28k times per cluster (2k symmetries x n_perm), and
    both inputs are always permutations of 0..n-1, so the closed form applies and
    is orders of magnitude cheaper than a general correlation routine.
    """
    d = a - b
    return 1.0 - 6.0 * float(d @ d) / (n * (n * n - 1))


def _rank_of(values):
    """Ranks 0..n-1 of `values`, ties broken by index (values here are distinct)."""
    order = np.argsort(np.asarray(values), kind="stable")
    r = np.empty(len(order), dtype=np.float64)
    r[order] = np.arange(len(order), dtype=np.float64)
    return r


def cyclic_order_score(recovered_angles, label_index, k, n_perm=2000, seed=0, full_k=None):
    """Agreement between recovered angular order and a known cycle.

    Returns dict with `rho` (best over the 2*full_k symmetries), a permutation
    p-value computed under the same maximisation, and the winning symmetry.

    `full_k` is the size of the cycle the labels live on, which is not always the
    number of labels being scored: when a month has too little support to be
    scored, the survivors still sit at their true calendar positions. Passing
    `label_index` as those true positions together with `full_k` (12 or 7) makes
    the expected angular position of label i exactly 2*pi*label_index[i]/full_k,
    gaps included, instead of pretending the survivors are contiguous.

    Recorded because it is not obvious and was checked rather than assumed: doing
    this changes NO number this statistic reports. Spearman sees only ranks, and
    dropping labels leaves the survivors' cyclic ORDER untouched, so rotating the
    true positions through full_k shifts yields the same k distinct reference rank
    vectors as rotating a contiguous cycle through k. Verified identical in rho,
    p and symmetry over 300 randomised gapped cases
    (`scripts/tests/test_gate_mods.py`). The parameter is kept because it states
    the reference the gate intends and would matter to any spacing-sensitive
    statistic; the gap structure does bite, but on closure (`circularity`), not
    here.
    """
    ang = np.asarray(recovered_angles, dtype=float)
    lab = np.asarray(label_index, dtype=int)
    K = int(k if full_k is None else full_k)
    # np.argsort is STABLE, so tied angles come back in index order -- which is the
    # label order. A cluster that never fires on the labelled tokens yields identical
    # (all-zero) means, every angle ties, and the "recovered" ordering is then the
    # calendar order by construction: rho = 1.000 from pure degeneracy. Shuffling the
    # labels does NOT catch this, because the artifact is exactly
    # "recovered order == label index order". Break ties randomly instead.
    jitter = np.random.default_rng(seed).permutation(len(ang))
    order = np.lexsort((jitter, ang))
    rank_of = np.empty_like(order); rank_of[order] = np.arange(len(ang))

    rank_of = rank_of.astype(np.float64)

    # The reference ordering under each of the 2*K rotation/reflection symmetries,
    # as ranks rather than raw cycle positions: with gaps the rotated positions are
    # no longer 0..n-1, and only their ORDER is meaningful to a rank correlation.
    # Computed once on the identity labelling, because permuting the labels only
    # reindexes the ranks -- which keeps the n_perm null as cheap as it was.
    n = len(lab)
    syms = [((flip, shift), _rank_of((flip * lab + shift) % K))
            for flip in (1, -1) for shift in range(K)]

    def best(sigma):
        out, arg = -2.0, None
        for s, base in syms:
            r = _spearman_perm(rank_of, base[sigma], n)
            if r > out:
                out, arg = r, s
        return out, arg

    rho, sym = best(np.arange(n))
    rng = np.random.default_rng(seed)
    null = np.array([best(rng.permutation(n))[0] for _ in range(n_perm)])
    # add-one (Phipson & Smyth): (null >= rho).mean() can return literal 0.0, which is
    # not a valid probability and was being reported as "p = 0.00000".
    return dict(rho=rho, p=float((1 + (null >= rho).sum()) / (1 + len(null))), symmetry=sym,
                null_mean=float(null.mean()), null_p95=float(np.percentile(null, 95)))


def fit_helix(points, values, periods=(2, 5, 10, 100)):
    """Least-squares fit of a base-10 helix to numeric-token activations.

    Follows the Kantamneni & Tegmark parameterisation: a linear ramp in the value
    plus cos/sin pairs at fixed periods. Reported as R^2 against a linear-only
    baseline, so a high score means the periodic structure is doing real work.
    """
    X = np.asarray(points, dtype=np.float64)
    X = X - X.mean(0, keepdims=True)
    v = np.asarray(values, dtype=np.float64)

    def design(use_periodic):
        cols = [np.ones_like(v), v]
        if use_periodic:
            for T in periods:
                cols += [np.cos(2 * np.pi * v / T), np.sin(2 * np.pi * v / T)]
        return np.column_stack(cols)

    def r2(B):
        coef, *_ = np.linalg.lstsq(B, X, rcond=None)
        resid = X - B @ coef
        return 1 - (resid ** 2).sum() / max((X ** 2).sum(), 1e-12)

    return dict(r2_helix=float(r2(design(True))),
                r2_linear=float(r2(design(False))),
                periods=list(periods))


# ---------------------------------------------------------------------------
# Label-free ringness, on decoder directions rather than the activation cloud.
# ---------------------------------------------------------------------------
# Three earlier rankers failed, each for a different reason, so the reasoning is
# recorded here rather than in a commit message:
#
#   top-2 PCA variance fraction  -- trivially ~1.0 for a 3-latent cluster, so it
#       ranked by size and selected the smallest clusters.
#   circle residual / fitted radius -- the Kasa fit is unbounded in radius, so a
#       collinear set gets an enormous circle and the ratio tends to 0. A regular
#       heptagon and a near-collinear set both score 0.0000.
#   angular gap on the activation cloud -- the normaliser used the number of
#       POINTS, but a k-spoke fan has only k distinct angles, so gap_ratio = n/k
#       exactly (measured 1428.6 at n=1e4, k=7) while a structureless Gaussian
#       blob scores 9.4. Ranking ascending preferred noise by ~100x. Mean-centring
#       made it worse: with non-negative codes the centroid sits inside the cone,
#       so three unrelated directions score 13.7 centred against 2876.8 anchored.
#
# The cloud was never the right object. Only ~1.13 latents of a 22-latent cluster
# fire at any one position (L0=20 over 16384 latents), so a per-position
# reconstruction is a single scaled decoder direction. The cloud's angles ARE the
# decoder directions, which are fixed, unit-norm, k of them, and need no centring
# of a cone. Working on them directly removes the size dependence and the
# manufactured coverage at once.


def direction_ring_score(w_cluster, n_perm=4000, seed=0):
    """Are a cluster's decoder directions spread evenly around their own plane?

    Returns the angular gap ratio (1.0 = perfectly even, large = an arc or a
    clump) and a p-value against the only sensible null: k random unit vectors in
    high dimensions, projected into their own top-2 principal plane.
    """
    W = np.asarray(w_cluster, dtype=np.float64)
    k = len(W)
    W = W - W.mean(0)
    U, S, Vt = np.linalg.svd(W, full_matrices=False)
    Y = W @ Vt[:2].T
    R = Y / np.maximum(np.linalg.norm(Y, axis=1, keepdims=True), 1e-12)
    th = np.sort(np.arctan2(R[:, 1], R[:, 0]))
    gaps = np.diff(np.concatenate([th, [th[0] + 2 * np.pi]]))
    gr = float(gaps.max() / (2 * np.pi / k))

    rng = np.random.default_rng(seed)
    worse = 0
    for _ in range(n_perm):
        V = rng.normal(size=(k, 8))
        V = V - V.mean(0)
        _, _, Vt2 = np.linalg.svd(V, full_matrices=False)
        Z = V @ Vt2[:2].T
        Z = Z / np.maximum(np.linalg.norm(Z, axis=1, keepdims=True), 1e-12)
        a = np.sort(np.arctan2(Z[:, 1], Z[:, 0]))
        g = np.diff(np.concatenate([a, [a[0] + 2 * np.pi]]))
        worse += (g.max() / (2 * np.pi / k)) <= gr
    return dict(k=k, gap_ratio=gr, p_closed=float((1 + worse) / (1 + n_perm)),
                top2_var=float((S[:2] ** 2).sum() / max((S ** 2).sum(), 1e-12)))


# Measured false-acceptance on purely random directions: k=3 gives 40/40, k=4
# 5/40, k=5 6/40, k=6 1/40. Three random unit vectors in high dimensions are
# always near-equilateral in their own plane, so small clusters are excluded.
MIN_RING_K = 6
