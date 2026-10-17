"""Null models and controls.

Gemma Scope 2 is Matryoshka-trained (GS2 technical report §3.3 and Appendix A),
which means latent *index* is not exchangeable: low-index latents are
systematically higher-frequency and more reconstruction-important. Measured on
these SAEs, Spearman(index, firing frequency) = -0.48 (p ~ 0) with median
frequency falling ~20x from the first index octile to the last.

That breaks the obvious null. A globally shuffled control would compare a real
cluster against random latents drawn from a *different* frequency regime, so
almost any coherent group would look significant, and a blind run could
rediscover the Matryoshka ordering and report it as geometry. Every null here is
therefore stratified: resampled latents are matched to the observed cluster on
index band and on firing frequency.
"""
import numpy as np


def index_strata(n_features, n_strata=8):
    """Contiguous index bands -- the axis the Matryoshka loss orders latents along."""
    edges = np.linspace(0, n_features, n_strata + 1).astype(int)
    return [(edges[i], edges[i + 1]) for i in range(n_strata)]


def stratified_resample(cluster, n_features, rng, n_strata=8):
    """A random latent set matching the cluster's index-band composition."""
    strata = index_strata(n_features, n_strata)
    out = []
    for lo, hi in strata:
        k = int(((np.asarray(cluster) >= lo) & (np.asarray(cluster) < hi)).sum())
        if k:
            out.extend(rng.choice(np.arange(lo, hi), size=k, replace=False).tolist())
    return np.array(sorted(out))


def frequency_matched_resample(cluster, freqs, rng, tol=0.25):
    """A random latent set matching each cluster member's firing frequency (+/- tol in log space)."""
    lf = np.log10(np.maximum(freqs, 1e-12))
    out, used = [], set(int(c) for c in cluster)
    for c in cluster:
        lo, hi = lf[c] - tol, lf[c] + tol
        pool = np.flatnonzero((lf >= lo) & (lf <= hi))
        pool = np.array([p for p in pool if int(p) not in used])
        if len(pool) == 0:
            continue
        pick = int(rng.choice(pool))
        used.add(pick)
        out.append(pick)
    return np.array(sorted(out))


def index_concentration(cluster, n_features, rng, n_null=500):
    """Is this cluster tighter in index than a random cluster of the same size?

    Returns (iqr, null_5th_percentile, p). A small p means the cluster is
    plausibly an artifact of Matryoshka index ordering rather than a concept.
    """
    a = np.asarray(cluster)
    iqr = float(np.diff(np.percentile(a, [25, 75]))[0])
    null = np.array([
        np.diff(np.percentile(rng.choice(n_features, len(a), replace=False), [25, 75]))[0]
        for _ in range(n_null)
    ])
    return iqr, float(np.percentile(null, 5)), float((null <= iqr).mean())


def shuffled_labels(labels, rng):
    """Permute post-hoc labels; the ordering statistic must collapse under this."""
    p = np.array(labels, copy=True)
    rng.shuffle(p)
    return p


def benjamini_hochberg(pvals, alpha=0.05):
    """FDR control across the clusters x layers scan."""
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    order = np.argsort(p)
    thresh = alpha * (np.arange(1, n + 1) / n)
    passed = p[order] <= thresh
    k = np.flatnonzero(passed).max() + 1 if passed.any() else 0
    keep = np.zeros(n, dtype=bool)
    keep[order[:k]] = True
    return keep
