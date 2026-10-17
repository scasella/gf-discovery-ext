"""Feature clustering over SAE latents.

Bhalla et al. (arXiv 2604.28119) report that decoder cosine similarity *fails* to
recover manifold block structure, because manifold membership is a functional
property (which atoms co-activate) rather than a geometric one (where atoms
point); they recommend Ising couplings or conditional co-activation instead. That
comparison is small, though -- three manifolds, top-3 features each, one SAE -- so
we run cosine and the functional metrics co-primary and report their agreement
rather than dropping either.

Gemma Scope 2 decoder rows are exactly unit norm (verified: std 5.8e-8), so
``w_dec @ w_dec.T`` is already the cosine matrix, and decoder norm carries no
information for filtering dead latents -- use firing frequency for that.
"""
import numpy as np


def cosine_matrix(w_dec):
    """[F, F] decoder cosine similarity. Rows are unit-norm, so this is a gram matrix."""
    W = np.asarray(w_dec, dtype=np.float32)
    W = W / np.linalg.norm(W, axis=1, keepdims=True)
    return W @ W.T


def binary_matrix(reader, features=None):
    """[n_tokens, n_features] CSR indicator of which latents fired where."""
    import scipy.sparse as sp
    indptr = reader.indptr
    idx = np.asarray(reader.indices, dtype=np.int64)
    B = sp.csr_matrix((np.ones(len(idx), dtype=np.float32), idx, indptr),
                      shape=(len(reader), reader.meta["d_sae"]))
    return B[:, features] if features is not None else B


def coactivation_matrix(reader, features=None, min_count=1):
    """P(j fires | i fires), computed as a sparse Gram matrix.

    Conditional rather than symmetric: a rare latent that always co-fires with a
    common one is strong evidence of shared structure, and symmetrising loses that.
    The naive per-token double loop is ~1e9 operations at our corpus size, so this
    goes through a sparse matmul instead.
    """
    B = binary_matrix(reader, features)
    joint = np.asarray((B.T @ B).todense(), dtype=np.float32)   # float32 halves 1.6 GB -> 0.8 GB
    counts = np.diag(joint).astype(np.float64).copy()
    joint /= np.maximum(counts[:, None], 1.0).astype(np.float32)  # in place: no second copy
    joint[counts < min_count, :] = 0.0
    np.fill_diagonal(joint, 0.0)
    return np.nan_to_num(joint, copy=False), counts


def correlation_matrix(reader, features):
    """Pearson correlation of activation magnitudes over the cached corpus."""
    import scipy.sparse as sp
    feats = np.asarray(features)
    pos = -np.ones(reader.meta["d_sae"], dtype=np.int64)
    pos[feats] = np.arange(len(feats))
    rows, cols, vals = [], [], []
    idx = np.asarray(reader.indices, dtype=np.int64)
    val = np.asarray(reader.values, dtype=np.float32)
    keep = pos[idx] >= 0
    rowid = np.searchsorted(reader.indptr, np.flatnonzero(keep), side="right") - 1
    M = sp.csr_matrix((val[keep], (rowid, pos[idx[keep]])),
                      shape=(len(reader), len(feats))).toarray()
    return np.corrcoef(M, rowvar=False), M


def ising_couplings(M, l1=0.05, max_features=1500):
    """Inverse Ising couplings by L1-regularised logistic pseudolikelihood.

    Goodfire report this as the metric that separates manifold blocks most cleanly.
    It is O(F) logistic regressions each with F-1 predictors, so it is only
    tractable on a restricted candidate set -- pass the surviving features, not
    all 16384.
    """
    from sklearn.linear_model import LogisticRegression
    X = (np.asarray(M) > 0).astype(np.float64)
    F = X.shape[1]
    if F > max_features:
        raise ValueError(f"{F} features exceeds max_features={max_features}; restrict first")
    J = np.zeros((F, F))
    for i in range(F):
        y = X[:, i]
        if y.std() == 0:
            continue
        Z = np.delete(X, i, axis=1)
        # sklearn >=1.8 deprecates penalty="l1" in favour of l1_ratio=1
        lr = LogisticRegression(l1_ratio=1.0, solver="liblinear", C=1.0 / max(l1, 1e-9))
        lr.fit(Z, y)
        w = lr.coef_[0]
        J[i, np.arange(F) != i] = w
    return (J + J.T) / 2.0          # symmetrise; sign and magnitude both matter


def graph_clusters(sim, threshold, min_size=3, method="leiden"):
    """Community-detect a thresholded similarity graph.

    Louvain in the paper; Leiden is its strictly-better successor (it guarantees
    well-connected communities) and is what igraph exposes, so we use Leiden and
    say so.
    """
    import igraph as ig
    import leidenalg

    # Memory matters here: at ~14k alive latents a float64 copy is 1.6 GB and
    # np.triu would double that again. Threshold straight to edge lists instead,
    # keeping only the boolean mask and the surviving indices.
    n = sim.shape[0]
    src, dst = np.nonzero(sim >= threshold)
    keep = src < dst                                  # upper triangle, drops the diagonal
    src, dst = src[keep], dst[keep]
    if len(src) == 0:
        return []
    w = np.asarray(sim[src, dst], dtype=np.float64)
    g = ig.Graph(n=n, edges=np.column_stack([src, dst]).tolist())
    g.es["weight"] = w.tolist()
    part = leidenalg.find_partition(
        g, leidenalg.ModularityVertexPartition, weights="weight", seed=0)
    return [sorted(c) for c in part if len(c) >= min_size]


def pca_spectrum(points):
    """Explained-variance ratios; the paper validates a cluster by a sharp spectral gap."""
    X = np.asarray(points, dtype=np.float64)
    X = X - X.mean(0, keepdims=True)
    if X.shape[0] < 2:
        return np.array([]), None
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    var = S ** 2
    return var / max(var.sum(), 1e-12), Vt


def spectral_gap(ratios, k=2):
    """Ratio of the k-th to (k+1)-th explained variance. Large => k-dimensional."""
    if len(ratios) <= k:
        return float("inf")
    return float(ratios[k - 1] / max(ratios[k], 1e-12))
