"""45 -- masked difference-family census (Phase 4 section 2.4; Addendum M5 authorises the run).

THE OBJECT. Not a shape inside one population but a RELATION between two: a set of ordered
token-type pairs {(a_m, b_m)} whose difference vectors b - a concentrate on a single
direction, i.e. a shared transformation T with b ~ T(a). Tier 1 (this script) is T =
translation. The label-free discriminator is that the map GENERALISES: fit t-hat on half of a
family's pairs, retrieve the held-out targets by nearest neighbour over all 3,795 types.

WHY THE MONTHS ARE THE NEGATIVE CONTROL HERE. Pre-freeze exposure (prereg section 0): month
successor differences have mean pairwise cosine -0.09 (adjacent -0.33, i.e. the -0.5 shared-
middle-term prediction for near-orthogonal type means), and a successor translation fitted on
6 pairs retrieves the held-out successor 0/6 WITHIN THE TWELVE at L6 and 0/6 at L8. The ring
is a ring; it is not a translation-carrying relation. Capitalisation is: 482 pool pairs, mean
pairwise difference cosine 0.531, split-half retrieval top-1 0.842 over 3,795 types. So the
INSTRUMENT's positive control is capitalisation with the mask off, and the months are the
negative control. This inversion is the deviation recorded in the plan of record and in
prereg section 0; it is not a post-hoc substitution.

R9 NOTE (declared). "Halves are by sequence id" does not apply to this script. Nothing here
reads a cache row: the unit of analysis is an ordered PAIR of token types, and the split-half
is over pairs of a family, not over occurrences. The type means themselves come from the
frozen full-n artifact (all 15,696 sequences). Stated so the omission is not read as a lapse.

R10. Coordinate 1038 is dropped from R_L6/R_L7 before anything else (1151 kept dims).
R7. Every threshold below is a null percentile or a declared structural constant (family size,
    neighbour count, centroid count), never a months-anchored value.
R12. CPU only. No MPS, no cache streaming, no forward pass. The only model artifact read is
    the tokenizer (token strings, for the morphology mask and for post-hoc naming).

WHAT IS SUBSAMPLED, AND WHY (stated up front, per the coordinator's instruction). The
assembly-level false-proposal rate asks for the identical pipeline on 100 frequency-matched
pseudo-pair populations and 100 class-block-permuted populations. One arm-1 assembly is
minutes; 200 of them is a day. So the false-proposal rate is measured at two levels:
  (a) TIER level, full count: 100 frequency-matched and 100 class-block-permuted pseudo-pair
      families per real family, through the identical retrieval statistic. This is the
      threshold that adjudicates (prereg 2.4: "top-10 above the 99th percentile of
      frequency-matched pseudo-pair families").
  (b) ASSEMBLY level, subsampled to N_ASSEMBLY_NULL replicates of each kind at L6 arm 1: the
      whole pipeline (pair set -> mask -> blocking -> Leiden -> family filter -> tier) on a
      population whose targets have been replaced. Reported as families-per-replicate and as
      tier-passers-per-replicate, with the subsample size stated in the output.
"""
import argparse, gc, json, os, sys, time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HOME", str(REPO / "cache/hf"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
sys.path.insert(0, str(REPO / "src"))

# ---------------------------------------------------------------------------
# frozen constants
# ---------------------------------------------------------------------------
FULLN_RAW = "results/phase3_token_type_mean_resid_fulln.npz"
MODEL_NAME = "google/gemma-3-1b-pt"
D_MODEL = 1152
R10_DROP = {L: (1125 if L <= 2 else 1038) for L in range(0, 11)}
LAYERS = (6, 7)

KNN_K = 50                      # arm 1: ordered pairs per source type
N_FAR_SAMPLE = 500_000          # arm 2: sampled ordered pairs for the centroid fit
N_CENTROIDS = 2_000             # arm 2 (and arm-1 blocking)
KMEANS_ITERS = 12
ARM2_KEEP_QUANTILE = 99.9       # percentile of the SAMPLE's best-centroid cosine (R7)
FAMILY_MIN_PAIRS = 6
FAMILY_MIN_SOURCES = 6
LEIDEN_RESOLUTION = 1.0         # scripts/22 convention
LEIDEN_SEED = 0
EDGE_BUDGET = 2_000_000         # scripts/22 convention
TAU_PERCENTILE = 99.9           # graph edge threshold = this pct of random difference cosines
N_TAU_SAMPLE = 200_000
BLOCK_NEIGHBOUR_CENTROIDS = 4   # arm-1 graph: own centroid + this many nearest centroids

N_TIER_NULL = 100               # (a) tier-level nulls, per family, per kind
FREQ_TOL_DEX = 0.25             # log10-count tolerance, nulls.frequency_matched_resample
N_ASSEMBLY_NULL = 5             # (b) assembly-level nulls per kind, L6 arm 1 (SUBSAMPLED)

PLANT_M = (6, 10, 20, 50)
PLANT_SIGMA = (0.1, 0.3, 0.5, 0.7, 0.9)
PLANT_SIGMA_EXT = (1.2, 1.5, 2.0)   # co-reported extension: the declared grid saturates
PLANT_REPS = 30                 # SUBSAMPLED from 50 (wall-clock); stated in the output
PLANT_BACKGROUND = 8_000        # background pairs per plant replicate (SUBSAMPLED, stated)
PLANT_CENTROIDS = 200           # blocking centroids scaled to the smaller background

SUFFIXES = ("s", "es", "ed", "ing", "er", "ly", "'d", "'t")
LEV_MAX = 2
SEED = 0

MONTH_IDS_CAL_ORDER = [6027, 7250, 5261, 5365, 3353, 5142, 5505, 5729, 5857, 6198, 6243, 6384]


# ---------------------------------------------------------------------------
def declared_rules(av):
    return dict(
        prereg="docs/phase4_prereg_2026-08-22.md section 2.4; rules R7-R12 of section 1",
        addenda_in_force=["docs/phase4_addendum_K_2026-08-22.md",
                          "docs/phase4_addendum_L_2026-08-22.md",
                          "docs/phase4_addendum_M_2026-08-22.md (M5 authorises this run)"],
        plan_of_record="~/.claude/plans/synthetic-percolating-raven.md (approved)",
        object=("a RELATION: a set of ordered token-type pairs whose difference vectors share "
                "one direction. Tier 1 map class = translation."),
        inputs=dict(type_means=FULLN_RAW, layers=list(LAYERS),
                    r10_drop={str(L): R10_DROP[L] for L in LAYERS},
                    centring="pool mean over the 3,795 types, per layer, after the R10 drop",
                    no_cache_rows_read=True, no_forward_pass=True, device="cpu"),
        r9_not_applicable=(
            "R9 (halves by sequence id) does not apply: the unit of analysis is an ordered "
            "PAIR of token types and the split-half is over a family's pairs. No cache row is "
            "read; the type means are the frozen full-n artifact over all 15,696 sequences."),
        assembly=dict(
            arm1=f"ordered pairs (i, j) with j among i's top-{KNN_K} cosine neighbours among "
                 f"the 3,795 centred type means; both directions are distinct rows",
            arm2=f"{N_FAR_SAMPLE} uniformly sampled ordered pairs -> spherical k-means to "
                 f"{N_CENTROIDS} centroids on the normalised differences -> ALL 3795*3794 "
                 f"ordered pairs stream-assigned to their best centroid via the bilinear "
                 f"identity cos(d_ij, c) = (X[j].c - X[i].c)/||X[j]-X[i]||, so the "
                 f"14.4M x 1151 pair matrix is never materialised",
            mask=("PAIR-level morphology mask, applied before clustering; types stay in the "
                  "pool so every null stays composition-matched. s = decoded token string, "
                  "stripped and lowercased. Pair masked iff s_i == s_j, OR the two agree "
                  f"after removing one suffix from {list(SUFFIXES)} from either or both, OR "
                  f"Levenshtein(s_i, s_j) <= {LEV_MAX}."),
            clustering=(f"Leiden (RBConfiguration, resolution {LEIDEN_RESOLUTION}, seed "
                        f"{LEIDEN_SEED}) on the normalised difference vectors. The candidate "
                        f"edge set is blocked by the k-means partition (own centroid + "
                        f"{BLOCK_NEIGHBOUR_CENTROIDS} nearest centroids); every retained edge "
                        f"weight is the EXACT full-1151-d cosine, and an edge is kept iff that "
                        f"cosine >= tau."),
            tau=(f"tau = the {TAU_PERCENTILE}th percentile of the cosine between "
                 f"{N_TAU_SAMPLE} random pairs of normalised differences from the SAME masked "
                 f"pair set (a null percentile, R7). If the graph exceeds {EDGE_BUDGET} edges, "
                 f"tau is raised on a declared grid until it does not, and the raise is logged."),
            family=f"a cluster with >= {FAMILY_MIN_PAIRS} pairs AND >= {FAMILY_MIN_SOURCES} "
                   f"distinct source types, after masking",
            arm2_family=(f"a centroid's assigned pairs, restricted to pairs whose "
                         f"best-centroid cosine is above the {ARM2_KEEP_QUANTILE}th percentile "
                         f"of that statistic on the {N_FAR_SAMPLE}-pair sample (R7), then "
                         f"masked and filtered by the same family rule")),
        tiers=dict(
            retrieval=("split-half TRANSLATION retrieval. t_hat = mean(X[b] - X[a]) over a "
                       "random half of the family's pairs; for each held-out pair, rank the "
                       "true target b by cosine to X[a] + t_hat over all 3,795 types with the "
                       "source excluded. Reported: top-1, top-10, median rank."),
            retrieval_repeat_asymmetry=(
                "the real family's top-1/top-10 average 5 random half-splits; each null "
                "family uses 1. The null percentile is therefore computed on a HIGHER-"
                "variance statistic, which makes the PASS threshold harder to clear. "
                "Declared as conservative, not corrected."),
            retrieval_threshold=(f"PASS iff the family's top-10 rate exceeds the 99th "
                                 f"percentile of {N_TIER_NULL} frequency-matched pseudo-pair "
                                 f"families (same sources, each target replaced by a type "
                                 f"within {FREQ_TOL_DEX} dex of its log10 count) AND the 99th "
                                 f"percentile of {N_TIER_NULL} class-block-permuted families "
                                 f"(targets permuted within token-class blocks)."),
            persistence=("L6<->L7: for each PASSing family at one layer, the maximum Jaccard "
                         "over pairs against every family at the other layer, with the null = "
                         "the same maximum against size-matched random pair sets.")),
        controls=dict(
            positive=("mask OFF at L6 arm 1: the capitalisation family must be recovered "
                      "blind (largest cluster by count of capitalisation pairs) with "
                      "split-half top-1 >= 0.80. If it is not, the instrument is NOT "
                      "VALIDATED and no blind family is reported as discovery."),
            negative=("the 12 month successor pairs (calendar order, prereg section 0): they "
                      "must NOT form a translation family -- expected retrieval 0/6 within "
                      "the twelve. A census that proposes them is broken."),
            leakage=("any family with >= 50% of member pairs sharing a 4-character prefix "
                     "after masking is reported as MASK LEAKAGE, not discovery.")),
        planted=dict(m=list(PLANT_M), sigma_over_t=list(PLANT_SIGMA),
                     sigma_over_t_extension_co_reported=list(PLANT_SIGMA_EXT),
                     reps=PLANT_REPS,
                     norm="||t|| = the mean capitalisation-pair difference norm at that layer",
                     background=f"{PLANT_BACKGROUND} masked pairs sampled from the layer's own "
                                f"arm-1 set (SUBSAMPLED from 190k for wall-clock; stated)",
                     layer="L6 only (declared; wall-clock)",
                     recall="a replicate is a HIT iff one cluster holds >= 6 planted pairs"),
        false_proposal_rate=dict(
            tier_level=f"{N_TIER_NULL} frequency-matched + {N_TIER_NULL} class-permuted "
                       f"pseudo-pair families per real family (full count)",
            assembly_level=f"{N_ASSEMBLY_NULL} replicates of each kind at L6 arm 1 "
                           f"(SUBSAMPLED from the 100 named in 2.4; wall-clock. Stated in the "
                           f"output as n_assembly_null.)"),
        labels=("R3. Token strings enter (i) the morphology MASK, which is a declared "
                "label-free string rule applied identically to every population and null, "
                "(ii) the capitalisation positive control and the month negative control, "
                "both label-known by design, and (iii) post-hoc naming of blind families. No "
                "label enters any blind decision statistic."),
        exposure_before_this_run=(
            "SMOKE RUN, L6 only, 10 tier nulls, 3 plant reps, 1 FA replicate (killed during "
            "the FA stage). What was seen: COUNTS only -- arm 1 189,750 pairs, mask 16.6%, "
            "1,349,721 edges at tau 0.541, 17,299 clusters, 532 families; arm 2 kept 12,415 "
            "of 14,398,230 pairs above theta 0.921, 83 families; 615 families scored, 221 "
            "'PASS' against a 10-draw null (a max, not a p99); planted recall 3/3 in every "
            "declared cell. NO family membership, token string, or per-family statistic was "
            "inspected. Two defects were found and fixed before this run: (a) the "
            "capitalisation control set mixed capitalise-in-place with de-space pairs (717 "
            "pairs, retrieval top-1 0.599); it is now restricted to same-leading-space pairs, "
            "which is the construction behind prereg section 0's 0.842; (b) tau_from_null's "
            "percentile was a bound default, so an attempted percentile sweep was a no-op -- "
            "the sweep was ABANDONED, not repaired: TAU_PERCENTILE and LEIDEN_RESOLUTION keep "
            "their declared values and were never tuned on any control. A control-only grid "
            "at resolution 0.3 was started and killed after one cell (giant-blob assembly, "
            "cap purity 0.054); its output is discarded and its setting is not used."),
        seeds=dict(base=SEED, leiden=LEIDEN_SEED),
        args={k: (str(v) if isinstance(v, Path) else v) for k, v in av.items()},
    )


# ---------------------------------------------------------------------------
# morphology mask
# ---------------------------------------------------------------------------
def norm_string(s):
    return s.strip().lower()


def stem_variants(n):
    out = {n}
    for suf in SUFFIXES:
        if n.endswith(suf) and len(n) > len(suf):
            out.add(n[:-len(suf)])
    return out


def lev_le(a, b, k=LEV_MAX):
    """Bounded Levenshtein: True iff distance <= k. Early exit on length gap."""
    if abs(len(a) - len(b)) > k:
        return False
    if a == b:
        return True
    la, lb = len(a), len(b)
    prev = list(range(lb + 1))
    for i in range(1, la + 1):
        cur = [i]
        ca = a[i - 1]
        lo = max(1, i - k)
        hi = min(lb, i + k)
        for j in range(1, lb + 1):
            if j < lo or j > hi:
                cur.append(k + 1)
                continue
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb_at(b, j))))
        if min(cur) > k:
            return False
        prev = cur
    return prev[lb] <= k


def cb_at(b, j):
    return b[j - 1]


class Masker:
    """Pair-level morphology mask with memoisation over normalised string pairs."""

    def __init__(self, strings):
        self.n = [norm_string(s) for s in strings]
        self.v = [stem_variants(x) for x in self.n]
        self.cache = {}
        self.calls = 0

    def masked(self, i, j):
        key = (i, j) if i < j else (j, i)
        hit = self.cache.get(key)
        if hit is not None:
            return hit
        if len(self.cache) > 3_000_000:      # bound: FA replicates create fresh pair sets
            self.cache.clear()
        self.calls += 1
        a, b = self.n[key[0]], self.n[key[1]]
        r = bool(a == b or (self.v[key[0]] & self.v[key[1]]) or lev_le(a, b))
        self.cache[key] = r
        return r

    def mask_pairs(self, pairs):
        """Boolean array: True where the pair survives the mask."""
        keep = np.ones(len(pairs), dtype=bool)
        for r in range(len(pairs)):
            if self.masked(int(pairs[r, 0]), int(pairs[r, 1])):
                keep[r] = False
        return keep


def token_class(s):
    t = s.lstrip()
    lead = s[:1] == " "
    if not t:
        return ("space", lead)
    if any(c.isdigit() for c in t):
        cat = "digit"
    elif not any(c.isalnum() for c in t):
        cat = "punct"
    elif t[0].isupper():
        cat = "cap"
    elif t[0].islower():
        cat = "lower"
    else:
        cat = "other"
    return (cat, lead)


# ---------------------------------------------------------------------------
# difference machinery (never materialises the full pair matrix)
# ---------------------------------------------------------------------------
class Diffs:
    """Normalised difference vectors of an ordered pair list, generated on demand."""

    def __init__(self, X, pairs):
        self.X = X                       # (n_types, d) float32, centred
        self.pairs = np.asarray(pairs, dtype=np.int32)

    def __len__(self):
        return len(self.pairs)

    def rows(self, idx):
        p = self.pairs[idx]
        D = self.X[p[:, 1]] - self.X[p[:, 0]]
        nrm = np.linalg.norm(D, axis=1, keepdims=True)
        np.maximum(nrm, 1e-12, out=nrm)
        D /= nrm
        return D

    def chunk(self, lo, hi):
        return self.rows(np.arange(lo, min(hi, len(self.pairs))))


def spherical_kmeans(diffs, K, iters, seed, chunk=25_000, log=print):
    import scipy.sparse as sp
    n = len(diffs)
    rng = np.random.default_rng(seed)
    C = diffs.rows(rng.choice(n, size=min(K, n), replace=False)).astype(np.float32)
    K = len(C)
    assign = np.zeros(n, dtype=np.int32)
    for it in range(iters):
        moved = 0
        acc = np.zeros((K, C.shape[1]), dtype=np.float32)
        cnt = np.zeros(K, dtype=np.int64)
        for lo in range(0, n, chunk):
            hi = min(lo + chunk, n)
            D = diffs.chunk(lo, hi)
            a = (D @ C.T).argmax(1).astype(np.int32)
            moved += int((a != assign[lo:hi]).sum())
            assign[lo:hi] = a
            M = sp.csr_matrix((np.ones(hi - lo, dtype=np.float32),
                               (a, np.arange(hi - lo))), shape=(K, hi - lo))
            acc += (M @ D).astype(np.float32)
            cnt += np.bincount(a, minlength=K)
        empty = np.flatnonzero(cnt == 0)
        if len(empty):
            acc[empty] = diffs.rows(rng.choice(n, size=len(empty), replace=False))
        nrm = np.linalg.norm(acc, axis=1, keepdims=True)
        np.maximum(nrm, 1e-12, out=nrm)
        C = acc / nrm
        if moved == 0:
            log(f"    kmeans converged at iter {it}")
            break
    return C, assign


def leiden_edges(n_nodes, ei, ej, w, resolution=LEIDEN_RESOLUTION, seed=LEIDEN_SEED):
    import igraph as ig, leidenalg as la
    g = ig.Graph(n=int(n_nodes), edges=list(zip(ei.tolist(), ej.tolist())))
    g.es["weight"] = np.asarray(w, dtype=float).tolist()
    p = la.find_partition(g, la.RBConfigurationVertexPartition, weights="weight",
                          resolution_parameter=resolution, seed=seed, n_iterations=-1)
    return [np.asarray(sorted(int(x) for x in c), dtype=np.int64) for c in p]


def tau_from_null(diffs, seed, n_sample=N_TAU_SAMPLE, pct=None):
    pct = TAU_PERCENTILE if pct is None else pct
    """R7 threshold: a percentile of cosines between RANDOM difference pairs of this set."""
    rng = np.random.default_rng(seed + 17)
    n = len(diffs)
    m = min(n_sample, max(1000, n * 4))
    a = rng.integers(0, n, size=m)
    b = rng.integers(0, n, size=m)
    ok = a != b
    a, b = a[ok], b[ok]
    vals = np.empty(len(a), dtype=np.float32)
    step = 50_000
    for lo in range(0, len(a), step):
        hi = min(lo + step, len(a))
        A = diffs.rows(a[lo:hi])
        B = diffs.rows(b[lo:hi])
        vals[lo:hi] = np.einsum("ij,ij->i", A, B)
    return float(np.percentile(vals, pct)), vals


def build_graph_and_cluster(diffs, K, seed, log=print, resolution=LEIDEN_RESOLUTION):
    """k-means blocking -> exact-cosine edges above tau -> Leiden. Returns (clusters, info)."""
    n = len(diffs)
    if n < FAMILY_MIN_PAIRS:
        return [], dict(n_pairs=int(n), skipped="too few pairs")
    K = int(max(2, min(K, n // 4)))
    C, assign = spherical_kmeans(diffs, K, KMEANS_ITERS, seed, log=log)
    K = len(C)
    tau, tau_vals = tau_from_null(diffs, seed, pct=TAU_PERCENTILE)
    order = np.argsort(assign, kind="stable")
    starts = np.searchsorted(assign[order], np.arange(K + 1))
    members = [order[starts[c]:starts[c + 1]] for c in range(K)]
    CC = C @ C.T
    np.fill_diagonal(CC, -2.0)
    nb = np.argsort(-CC, axis=1)[:, :BLOCK_NEIGHBOUR_CENTROIDS]

    grid = [tau, tau + 0.05, tau + 0.10, tau + 0.15, tau + 0.20, tau + 0.30]
    for tau_used in grid:
        ei, ej, ew = [], [], []
        n_edges = 0
        overflow = False
        for c in range(K):
            src = members[c]
            if len(src) == 0:
                continue
            cand = np.concatenate([members[c]] + [members[x] for x in nb[c]])
            if len(cand) == 0:
                continue
            A = diffs.rows(src)
            B = diffs.rows(cand)
            S = A @ B.T
            r, s = np.nonzero(S >= tau_used)
            if len(r) == 0:
                continue
            u = src[r]
            v = cand[s]
            keep = u < v
            u, v, val = u[keep], v[keep], S[r, s][keep]
            if len(u):
                ei.append(u); ej.append(v); ew.append(val)
                n_edges += len(u)
            if n_edges > EDGE_BUDGET:
                overflow = True
                break
        if not overflow:
            break
    if not ei:
        return [], dict(n_pairs=int(n), n_edges=0, tau=float(tau_used),
                        tau_null_pct=TAU_PERCENTILE, note="no edge above tau")
    ei = np.concatenate(ei); ej = np.concatenate(ej); ew = np.concatenate(ew)
    # dedupe (a pair can appear from both endpoints' blocks)
    key = ei.astype(np.int64) * n + ej.astype(np.int64)
    uq, first = np.unique(key, return_index=True)
    ei, ej, ew = ei[first], ej[first], ew[first]
    clusters = leiden_edges(n, ei, ej, ew, resolution=resolution)
    return clusters, dict(n_pairs=int(n), n_edges=int(len(ei)), n_centroids=int(K),
                          tau=float(tau_used), tau_base=float(tau),
                          tau_raised=bool(tau_used > tau),
                          tau_null_pct=TAU_PERCENTILE,
                          random_diff_cos_mean=float(tau_vals.mean()))


def families_from_clusters(clusters, pairs):
    out = []
    for cl in clusters:
        if len(cl) < FAMILY_MIN_PAIRS:
            continue
        p = pairs[cl]
        if len(np.unique(p[:, 0])) < FAMILY_MIN_SOURCES:
            continue
        out.append(cl)
    return out


# ---------------------------------------------------------------------------
# tiers
# ---------------------------------------------------------------------------
def retrieval(X, Xn, pairs, seed, n_repeats=5):
    """Split-half translation retrieval. Returns top-1, top-10, median rank, n_held_out."""
    p = np.asarray(pairs)
    k = len(p)
    if k < 4:
        return None
    rng = np.random.default_rng(seed)
    t1, t10, med, nte = [], [], [], 0
    for _ in range(n_repeats):
        perm = rng.permutation(k)
        h = k // 2
        tr, te = perm[:h], perm[h:]
        t_hat = (X[p[tr, 1]] - X[p[tr, 0]]).mean(0)
        P = X[p[te, 0]] + t_hat
        P = P / np.maximum(np.linalg.norm(P, axis=1, keepdims=True), 1e-12)
        S = P @ Xn.T
        S[np.arange(len(te)), p[te, 0]] = -2.0
        true = S[np.arange(len(te)), p[te, 1]]
        rank = (S > true[:, None]).sum(1)
        t1.append(float((rank == 0).mean()))
        t10.append(float((rank < 10).mean()))
        med.append(float(np.median(rank)))
        nte = len(te)
    return dict(top1=float(np.mean(t1)), top10=float(np.mean(t10)),
                median_rank=float(np.mean(med)), n_held_out=int(nte),
                n_pairs=int(k), n_repeats=int(n_repeats))


def freq_matched_targets(pairs, logc, rng, tol=FREQ_TOL_DEX, order=None):
    """Replace each target by a type within tol dex of its log10 count."""
    if order is None:
        order = np.argsort(logc)
    lo = np.searchsorted(logc[order], logc[pairs[:, 1]] - tol, side="left")
    hi = np.searchsorted(logc[order], logc[pairs[:, 1]] + tol, side="right")
    out = pairs.copy()
    span = np.maximum(hi - lo, 1)
    pick = lo + (rng.random(len(pairs)) * span).astype(np.int64)
    pick = np.minimum(pick, len(order) - 1)
    out[:, 1] = order[pick]
    bad = out[:, 1] == out[:, 0]
    out[bad, 1] = pairs[bad, 1]
    return out


def class_permuted_targets(pairs, class_id, rng):
    """Permute targets within token-class blocks."""
    out = pairs.copy()
    tgt = pairs[:, 1]
    for c in np.unique(class_id[tgt]):
        m = np.flatnonzero(class_id[tgt] == c)
        out[m, 1] = tgt[m][rng.permutation(len(m))]
    bad = out[:, 1] == out[:, 0]
    out[bad, 1] = pairs[bad, 1]
    return out


def tier_nulls(X, Xn, pairs, logc, class_id, seed, n=N_TIER_NULL, order=None):
    rng = np.random.default_rng(seed + 7717)
    f10, c10, f1, c1 = [], [], [], []
    for _ in range(n):
        r = retrieval(X, Xn, freq_matched_targets(pairs, logc, rng, order=order), seed, n_repeats=1)
        if r:
            f10.append(r["top10"]); f1.append(r["top1"])
        r = retrieval(X, Xn, class_permuted_targets(pairs, class_id, rng), seed, n_repeats=1)
        if r:
            c10.append(r["top10"]); c1.append(r["top1"])
    q = lambda v, p: float(np.percentile(v, p)) if len(v) else None
    return dict(n=int(n),
                freq_top10_p99=q(f10, 99), freq_top10_mean=q(f10, 50),
                class_top10_p99=q(c10, 99), class_top10_mean=q(c10, 50),
                freq_top1_p99=q(f1, 99), class_top1_p99=q(c1, 99))


def jaccard(a, b):
    A = set(map(tuple, np.asarray(a).tolist()))
    B = set(map(tuple, np.asarray(b).tolist()))
    return len(A & B) / max(len(A | B), 1)


# ---------------------------------------------------------------------------
def log_open(path):
    f = open(path, "a", buffering=1)

    def log(*a):
        msg = " ".join(str(x) for x in a)
        print(msg, flush=True)
        f.write(msg + "\n")
    return log, f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/phase4_difference_families.json")
    ap.add_argument("--log", default="logs/45_difference_families.log")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    t0 = time.time()
    Path(REPO / "logs").mkdir(exist_ok=True)
    log, logf = log_open(REPO / a.log)

    # ---- R1: declaration stub to disk BEFORE any read ----
    av = dict(vars(a))
    stub = dict(status="DECLARATION STUB -- rules frozen, no results yet",
                script="scripts/45_difference_family_census.py",
                item="Phase 4 section 2.4 -- masked difference-family census",
                declared_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
                declared_rules=declared_rules(av))
    Path(REPO / a.out).write_text(json.dumps(stub, indent=2))
    log("=== 45 DECLARATION STUB WRITTEN ===", a.out, stub["declared_at"])

    smoke = a.smoke
    n_tier_null = 10 if smoke else N_TIER_NULL
    n_asm_null = 1 if smoke else N_ASSEMBLY_NULL
    plant_reps = 3 if smoke else PLANT_REPS
    layers = (6,) if smoke else LAYERS

    # ---- pool, strings, classes ----
    z = np.load(REPO / FULLN_RAW)
    pool_ids = z["pool_ids"].astype(np.int64)
    counts = z["counts"].astype(np.int64)
    n_types = len(pool_ids)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    strings = [tok.decode([int(t)]) for t in pool_ids]
    masker = Masker(strings)
    classes = [token_class(s) for s in strings]
    cls_names = sorted(set(classes))
    class_id = np.array([cls_names.index(c) for c in classes], dtype=np.int32)
    logc = np.log10(np.maximum(counts, 1)).astype(np.float64)
    logc_order = np.argsort(logc)
    pos = {int(t): i for i, t in enumerate(pool_ids)}
    log(f"pool {n_types} types; classes {cls_names}")

    # capitalisation and plural reference pairs (label-known: controls only)
    from collections import defaultdict
    by_lower = defaultdict(list)
    for i, s in enumerate(strings):
        by_lower[s.strip().lower()].append(i)
    # The capitalisation control pairs types that differ ONLY by case. Pairs that also
    # change the leading space (' the' -> 'The') mix TWO transformations -- capitalise-in-
    # place and de-space -- and are excluded: prereg section 0's 0.842 is the same-leading-
    # space construction, and the mixed 717-pair set measures 0.599 (recorded as exposure).
    cap_pairs = []
    for _, v in by_lower.items():
        for x in v:
            for y in v:
                if x == y or strings[x] == strings[y]:
                    continue
                if (strings[x][:1] == " ") != (strings[y][:1] == " "):
                    continue
                if strings[x].strip()[:1].islower() and strings[y].strip()[:1].isupper():
                    cap_pairs.append((x, y))
    cap_pairs = np.asarray(cap_pairs, dtype=np.int32)
    month_rows = [pos[t] for t in MONTH_IDS_CAL_ORDER]
    month_pairs = np.asarray([[month_rows[i], month_rows[(i + 1) % 12]] for i in range(12)],
                             dtype=np.int32)
    log(f"capitalisation reference pairs {len(cap_pairs)}; month successor pairs 12")

    out = dict(stub)
    out["status"] = "RUNNING"
    out["pool"] = dict(n_types=int(n_types), n_cap_pairs=int(len(cap_pairs)),
                       classes=[list(c) for c in cls_names])
    out["layers"] = {}
    out["subsampling"] = dict(n_tier_null=int(n_tier_null), n_assembly_null=int(n_asm_null),
                              plant_reps=int(plant_reps),
                              plant_background=int(PLANT_BACKGROUND),
                              note=("assembly-level FA is subsampled from the 100+100 named "
                                    "in prereg 2.4 for wall-clock; the tier-level FA (which "
                                    "is the adjudicating threshold) runs at full count"))

    for L in layers:
        tL = time.time()
        log(f"\n===== LAYER {L} =====")
        A = z[f"R_L{L}"].astype(np.float64)
        assert np.isfinite(A).all(), f"non-finite raw means L{L}"
        keep = np.ones(D_MODEL, dtype=bool); keep[R10_DROP[L]] = False
        X = np.ascontiguousarray(A[:, keep], dtype=np.float32)
        del A
        X -= X.mean(0, keepdims=True)
        Xn = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
        rec = dict(layer=int(L), d_kept=int(X.shape[1]), dropped_coord=int(R10_DROP[L]))

        # ---------- arm 1: kNN-50 ordered pairs ----------
        S = (Xn @ Xn.T).astype(np.float32)
        np.fill_diagonal(S, -2.0)
        nn = np.argpartition(-S, KNN_K, axis=1)[:, :KNN_K]
        src = np.repeat(np.arange(n_types, dtype=np.int32), KNN_K)
        pairs1 = np.column_stack([src, nn.reshape(-1).astype(np.int32)])
        rec["arm1_pairs_raw"] = int(len(pairs1))
        rec["knn_mean_kth_cosine"] = float(np.sort(S, axis=1)[:, -KNN_K].mean())
        del S
        gc.collect()

        tm = time.time()
        keep1 = masker.mask_pairs(pairs1)
        pairs1m = pairs1[keep1]
        rec["arm1_masked_fraction"] = float(1.0 - keep1.mean())
        rec["arm1_pairs_after_mask"] = int(len(pairs1m))
        log(f"  arm1: {len(pairs1)} pairs, mask removes {100*(1-keep1.mean()):.1f}% -> "
            f"{len(pairs1m)} ({time.time()-tm:.0f}s)")

        d1 = Diffs(X, pairs1m)
        cl1, info1 = build_graph_and_cluster(d1, N_CENTROIDS, SEED + L, log=log)
        fam1 = families_from_clusters(cl1, pairs1m)
        rec["arm1_assembly"] = info1
        rec["arm1_n_clusters"] = len(cl1)
        rec["arm1_n_families"] = len(fam1)
        log(f"  arm1: {info1.get('n_edges')} edges, tau {info1.get('tau'):.3f}, "
            f"{len(cl1)} clusters, {len(fam1)} families ({time.time()-tL:.0f}s)")

        # ---------- arm 2: far pairs, centroids, all-pair assignment ----------
        rng = np.random.default_rng(SEED + 1000 + L)
        fi = rng.integers(0, n_types, size=int(N_FAR_SAMPLE * 1.05))
        fj = rng.integers(0, n_types, size=int(N_FAR_SAMPLE * 1.05))
        ok = fi != fj
        far = np.column_stack([fi[ok][:N_FAR_SAMPLE], fj[ok][:N_FAR_SAMPLE]]).astype(np.int32)
        d2s = Diffs(X, far)
        C2, _ = spherical_kmeans(d2s, N_CENTROIDS, KMEANS_ITERS, SEED + 2000 + L, log=log)
        log(f"  arm2: centroids {C2.shape} fitted ({time.time()-tL:.0f}s)")

        # threshold from the SAMPLE's best-centroid cosine (R7)
        best_sample = []
        for lo in range(0, len(far), 25_000):
            D = d2s.chunk(lo, lo + 25_000)
            best_sample.append((D @ C2.T).max(1))
        best_sample = np.concatenate(best_sample)
        theta = float(np.percentile(best_sample, ARM2_KEEP_QUANTILE))
        rec["arm2_theta"] = theta
        rec["arm2_theta_pct"] = ARM2_KEEP_QUANTILE
        del d2s, far, best_sample
        gc.collect()

        # stream-assign ALL ordered pairs via the bilinear identity
        P = (X @ C2.T).astype(np.float32)                       # (n_types, K)
        g = np.einsum("ij,ij->i", X, X).astype(np.float32)
        kept_i, kept_j, kept_c, kept_v = [], [], [], []
        step = 8
        for lo in range(0, n_types, step):
            hi = min(lo + step, n_types)
            N2 = g[None, :] + g[lo:hi, None] - 2.0 * (X[lo:hi] @ X.T)
            np.maximum(N2, 1e-12, out=N2)
            Nrm = np.sqrt(N2, dtype=np.float32)
            Sc = P[None, :, :] - P[lo:hi][:, None, :]
            Sc /= Nrm[:, :, None]
            for r in range(hi - lo):
                Sc[r, lo + r, :] = -2.0
            bc = Sc.argmax(2)
            bv = np.take_along_axis(Sc, bc[:, :, None], axis=2)[:, :, 0]
            m = bv >= theta
            if m.any():
                rr, jj = np.nonzero(m)
                kept_i.append((lo + rr).astype(np.int32))
                kept_j.append(jj.astype(np.int32))
                kept_c.append(bc[rr, jj].astype(np.int32))
                kept_v.append(bv[rr, jj].astype(np.float32))
            del Sc, N2, Nrm
        kept_i = np.concatenate(kept_i) if kept_i else np.zeros(0, np.int32)
        kept_j = np.concatenate(kept_j) if len(kept_j) else np.zeros(0, np.int32)
        kept_c = np.concatenate(kept_c) if len(kept_c) else np.zeros(0, np.int32)
        kept_v = np.concatenate(kept_v) if len(kept_v) else np.zeros(0, np.float32)
        pairs2 = np.column_stack([kept_i, kept_j])
        rec["arm2_pairs_kept"] = int(len(pairs2))
        rec["arm2_pairs_total_scanned"] = int(n_types * (n_types - 1))
        keep2 = masker.mask_pairs(pairs2) if len(pairs2) else np.zeros(0, bool)
        rec["arm2_masked_fraction"] = float(1.0 - keep2.mean()) if len(pairs2) else None
        pairs2m, cent2 = pairs2[keep2], kept_c[keep2]
        fam2 = []
        for c in np.unique(cent2):
            idx = np.flatnonzero(cent2 == c)
            if len(idx) < FAMILY_MIN_PAIRS:
                continue
            if len(np.unique(pairs2m[idx, 0])) < FAMILY_MIN_SOURCES:
                continue
            fam2.append((int(c), idx))
        rec["arm2_n_families"] = len(fam2)
        log(f"  arm2: kept {len(pairs2)} of {n_types*(n_types-1)} pairs above theta "
            f"{theta:.3f}; {len(fam2)} families ({time.time()-tL:.0f}s)")
        del P, kept_i, kept_j, kept_c, kept_v
        gc.collect()

        # ---------- tiers ----------
        fam_records = []
        for arm, fams in (("arm1", [(i, pairs1m[c]) for i, c in enumerate(fam1)]),
                          ("arm2", [(c, pairs2m[idx]) for c, idx in fam2])):
            for fid, fp in fams:
                r = retrieval(X, Xn, fp, SEED + int(fid))
                if r is None:
                    continue
                nl = tier_nulls(X, Xn, fp, logc, class_id, SEED + int(fid),
                                n=n_tier_null, order=logc_order)
                pref = [strings[int(i)].strip().lower()[:4] for i in fp[:, 0]]
                from collections import Counter
                topfrac = Counter(pref).most_common(1)[0][1] / len(pref) if pref else 0.0
                passed = bool(r["top10"] > (nl["freq_top10_p99"] or 1.0)
                              and r["top10"] > (nl["class_top10_p99"] or 1.0))
                fam_records.append(dict(
                    arm=arm, family_id=int(fid), n_pairs=int(len(fp)),
                    n_sources=int(len(np.unique(fp[:, 0]))),
                    retrieval=r, nulls=nl, passed=passed,
                    shared_prefix_frac=float(topfrac),
                    mask_leakage=bool(topfrac >= 0.5),
                    mean_pairwise_cos=float(_mean_pairwise_cos(X, fp)),
                    pairs=fp.tolist()))
        rec["families"] = fam_records
        rec["n_pass"] = int(sum(1 for f in fam_records if f["passed"]
                                and not f["mask_leakage"]))
        log(f"  tiers: {len(fam_records)} families scored, {rec['n_pass']} PASS "
            f"(non-leakage) ({time.time()-tL:.0f}s)")

        # ---------- controls ----------
        ctrl = {}
        # negative: months
        ctrl["months_negative"] = dict(
            retrieval_all_types=retrieval(X, Xn, month_pairs, SEED),
            within_12=_within_set_retrieval(X, month_pairs, np.asarray(month_rows), SEED),
            mean_pairwise_diff_cos=float(_mean_pairwise_cos(X, month_pairs)),
            in_any_family=bool(any(
                len(set(map(tuple, np.asarray(f["pairs"]).tolist()))
                    & set(map(tuple, month_pairs.tolist()))) >= 3
                for f in fam_records)))
        # positive: capitalisation, mask OFF, arm 1
        if L == 6 or smoke:
            d1u = Diffs(X, pairs1)
            cl1u, info1u = build_graph_and_cluster(d1u, N_CENTROIDS, SEED + 99 + L, log=log)
            capset = set(map(tuple, cap_pairs.tolist()))
            best, bestn = None, 0
            dom = []
            for cl in cl1u:
                p = pairs1[cl]
                nc = sum(1 for r in p.tolist() if tuple(r) in capset)
                if nc > bestn:
                    best, bestn = cl, nc
                if len(cl) >= FAMILY_MIN_PAIRS and nc / len(cl) >= 0.5:
                    dom.append(cl)
            cp = dict(assembly=info1u,
                      reference_retrieval=retrieval(X, Xn, cap_pairs, SEED),
                      n_reference_pairs=int(len(cap_pairs)),
                      reading_note=(
                          "PRIMARY reading = the UNION of capitalisation-dominant clusters "
                          "(>= 50% capitalisation pairs, >= 6 pairs). That is the like-for-"
                          "like comparison with prereg section 0's 0.842, which was measured "
                          "on the whole family; a fragmented but PURE recovery is still a "
                          "recovery. The single largest-overlap cluster is co-reported. "
                          "Neither the Leiden resolution nor tau was tuned on this control."))
            if best is not None:
                pu = pairs1[best]
                ru = retrieval(X, Xn, pu, SEED)
                cp.update(best_cluster=dict(size=int(len(pu)), n_cap_pairs=int(bestn),
                                            purity=float(bestn / len(pu)), retrieval=ru))
            if dom:
                un = np.concatenate(dom)
                pun = pairs1[un]
                nc = sum(1 for r in pun.tolist() if tuple(r) in capset)
                run_ = retrieval(X, Xn, pun, SEED)
                cp.update(union=dict(n_clusters=len(dom), size=int(len(pun)),
                                     n_cap_pairs=int(nc),
                                     purity=float(nc / len(pun)),
                                     recovered_fraction=float(nc / max(len(cap_pairs), 1)),
                                     retrieval=run_),
                          recovered=True, passes=bool(run_["top1"] >= 0.80))
            else:
                cp.update(union=None, recovered=False, passes=False)
            ctrl["capitalisation_positive"] = cp
            del d1u
            gc.collect()
        rec["controls"] = ctrl
        log("  controls: " + json.dumps({k: (v.get("passes") if isinstance(v, dict) else None)
                                         for k, v in ctrl.items()}))

        # ---------- planted power ----------
        cap_norm = float(np.linalg.norm(X[cap_pairs[:, 1]] - X[cap_pairs[:, 0]],
                                        axis=1).mean()) if len(cap_pairs) else 1.0
        rec["plant_norm"] = cap_norm
        if L == 6:   # DECLARED: the planted power curve is run at L6 only (wall-clock)
            rec["planted"] = _planted(X, pairs1m, cap_norm, plant_reps, log, SEED + 300 + L)

        # ---------- assembly-level FA (subsampled) ----------
        if L == 6:
            rec["assembly_false_proposal"] = _assembly_fa(
                X, Xn, pairs1m, logc, logc_order, class_id, masker, n_asm_null,
                n_tier_null, log, SEED + 500)

        rec["seconds"] = round(time.time() - tL, 1)
        out["layers"][str(L)] = rec
        Path(REPO / a.out).write_text(json.dumps(out, indent=2, default=float))
        log(f"  layer {L} done in {rec['seconds']}s")
        del X, Xn
        gc.collect()

    # ---------- persistence ----------
    if len(out["layers"]) == 2:
        out["persistence"] = _persistence(out, SEED)
    out["status"] = "complete"
    out["runtime_seconds"] = round(time.time() - t0, 1)
    out["summary"] = _summary(out)
    Path(REPO / a.out).write_text(json.dumps(out, indent=2, default=float))
    log("\n=== SUMMARY ===")
    log(json.dumps(out["summary"], indent=2, default=float))
    log(f"total {out['runtime_seconds']}s")
    logf.close()


def _mean_pairwise_cos(X, pairs):
    D = X[pairs[:, 1]] - X[pairs[:, 0]]
    D = D / np.maximum(np.linalg.norm(D, axis=1, keepdims=True), 1e-12)
    if len(D) > 800:
        D = D[np.random.default_rng(0).choice(len(D), 800, replace=False)]
    C = D @ D.T
    iu = np.triu_indices(len(D), 1)
    return float(C[iu].mean()) if len(iu[0]) else 0.0


def _within_set_retrieval(X, pairs, member_rows, seed):
    """Retrieval restricted to the population itself -- the sharp form of the month control."""
    p = np.asarray(pairs)
    k = len(p)
    tr = np.arange(0, k, 2)
    te = np.arange(1, k, 2)
    t_hat = (X[p[tr, 1]] - X[p[tr, 0]]).mean(0)
    M = X[member_rows]
    Mn = M / np.maximum(np.linalg.norm(M, axis=1, keepdims=True), 1e-12)
    P = X[p[te, 0]] + t_hat
    P = P / np.maximum(np.linalg.norm(P, axis=1, keepdims=True), 1e-12)
    S = P @ Mn.T
    pos = {int(r): i for i, r in enumerate(member_rows)}
    ranks = []
    for r in range(len(te)):
        true = S[r, pos[int(p[te[r], 1])]]
        ranks.append(int((S[r] > true).sum()))
    return dict(ranks=ranks, top1=int(sum(1 for x in ranks if x == 0)), n=len(ranks),
                chance_top1_rate=1.0 / len(member_rows))


def _planted(X, pairs, tnorm, reps, log, seed):
    """Recall of a planted parallel-difference family against a subsampled background."""
    rng = np.random.default_rng(seed)
    d = X.shape[1]
    n_types = X.shape[0]
    res = []
    t_start = time.time()
    for m in PLANT_M:
        for sig in tuple(PLANT_SIGMA) + tuple(PLANT_SIGMA_EXT):
            hits, purity = 0, []
            for rep in range(reps):
                bg = pairs[rng.choice(len(pairs), size=min(PLANT_BACKGROUND, len(pairs)),
                                      replace=False)]
                srcs = rng.choice(n_types, size=m, replace=False)
                t = rng.normal(size=d).astype(np.float32)
                t *= tnorm / np.linalg.norm(t)
                E = rng.normal(size=(m, d)).astype(np.float32)
                E *= (sig * tnorm) / np.maximum(np.linalg.norm(E, axis=1, keepdims=True), 1e-12)
                Dp = t[None, :] + E
                Dp /= np.maximum(np.linalg.norm(Dp, axis=1, keepdims=True), 1e-12)
                Dbg = X[bg[:, 1]] - X[bg[:, 0]]
                Dbg /= np.maximum(np.linalg.norm(Dbg, axis=1, keepdims=True), 1e-12)
                Dall = np.vstack([Dbg, Dp])
                planted_idx = np.arange(len(Dbg), len(Dall))
                dd = _RawDiffs(Dall)
                cl, _ = build_graph_and_cluster(dd, PLANT_CENTROIDS, seed + rep, log=lambda *a: None)
                best = 0
                bestp = 0.0
                for c in cl:
                    npl = int(np.isin(c, planted_idx).sum())
                    if npl > best:
                        best, bestp = npl, npl / len(c)
                if best >= FAMILY_MIN_PAIRS:
                    hits += 1
                    purity.append(bestp)
            res.append(dict(m=int(m), sigma_over_t=float(sig), reps=int(reps),
                            declared_grid=bool(sig in PLANT_SIGMA),
                            recall=float(hits / reps),
                            mean_purity=float(np.mean(purity)) if purity else None))
            log(f"    plant m={m} sigma={sig}: recall {hits}/{reps} "
                f"({time.time()-t_start:.0f}s)")
    return res


class _RawDiffs:
    """Diffs interface over an already-materialised unit-norm matrix (plant backgrounds)."""

    def __init__(self, D):
        self.D = D

    def __len__(self):
        return len(self.D)

    def rows(self, idx):
        return self.D[idx]

    def chunk(self, lo, hi):
        return self.D[lo:min(hi, len(self.D))]


def _assembly_fa(X, Xn, pairs, logc, logc_order, class_id, masker, n_rep, n_tier_null,
                 log, seed):
    """(b) assembly-level FA: the identical pipeline on pseudo-pair populations."""
    rng = np.random.default_rng(seed)
    out = dict(n_assembly_null=int(n_rep), kinds={})
    for kind in ("frequency_matched", "class_permuted"):
        fams, passers = [], []
        for rep in range(n_rep):
            pp = (freq_matched_targets(pairs, logc, rng, order=logc_order)
                  if kind == "frequency_matched"
                  else class_permuted_targets(pairs, class_id, rng))
            keep = masker.mask_pairs(pp)
            ppm = pp[keep]
            cl, _ = build_graph_and_cluster(Diffs(X, ppm), N_CENTROIDS, seed + rep,
                                            log=lambda *a: None)
            f = families_from_clusters(cl, ppm)
            npass = 0
            for c in f:
                fp = ppm[c]
                r = retrieval(X, Xn, fp, seed + rep)
                if r is None:
                    continue
                nl = tier_nulls(X, Xn, fp, logc, class_id, seed + rep,
                                n=max(20, n_tier_null // 5), order=logc_order)
                if r["top10"] > (nl["freq_top10_p99"] or 1.0) and \
                   r["top10"] > (nl["class_top10_p99"] or 1.0):
                    npass += 1
            fams.append(len(f)); passers.append(npass)
            log(f"    FA[{kind}] rep {rep}: {len(f)} families, {npass} tier-passers")
        out["kinds"][kind] = dict(families_per_replicate=fams,
                                  tier_passers_per_replicate=passers,
                                  mean_families=float(np.mean(fams)),
                                  mean_passers=float(np.mean(passers)))
    return out


def _persistence(out, seed):
    L6 = out["layers"].get("6", {}).get("families", [])
    L7 = out["layers"].get("7", {}).get("families", [])
    rows = []
    rng = np.random.default_rng(seed + 99)
    all7 = [np.asarray(f["pairs"]) for f in L7]
    for f in L6:
        if not (f["passed"] and not f["mask_leakage"]):
            continue
        p6 = np.asarray(f["pairs"])
        js = [jaccard(p6, p7) for p7 in all7]
        best = float(max(js)) if js else 0.0
        null = []
        if all7:
            allpairs7 = np.vstack(all7)
            for _ in range(200):
                idx = rng.choice(len(allpairs7), size=min(len(p6), len(allpairs7)),
                                 replace=False)
                null.append(jaccard(p6, allpairs7[idx]))
        rows.append(dict(arm=f["arm"], family_id=f["family_id"], n_pairs=f["n_pairs"],
                         max_jaccard_L7=best,
                         null_p99=float(np.percentile(null, 99)) if null else None,
                         persists=bool(null and best > np.percentile(null, 99))))
    return dict(direction="L6 PASS families vs all L7 families", rows=rows,
                n_checked=len(rows))


def _summary(out):
    s = dict(layers={})
    for L, rec in out["layers"].items():
        fams = rec.get("families", [])
        pas = [f for f in fams if f["passed"] and not f["mask_leakage"]]
        s["layers"][L] = dict(
            arm1_pairs_after_mask=rec.get("arm1_pairs_after_mask"),
            arm1_masked_fraction=rec.get("arm1_masked_fraction"),
            n_families=len(fams), n_pass=len(pas),
            family_sizes=sorted((f["n_pairs"] for f in fams), reverse=True)[:20],
            pass_ids=[[f["arm"], f["family_id"], f["n_pairs"],
                       round(f["retrieval"]["top1"], 3), round(f["retrieval"]["top10"], 3)]
                      for f in pas],
            capitalisation_control=(rec.get("controls", {})
                                    .get("capitalisation_positive", {}).get("passes")),
            months_within12_top1=(rec.get("controls", {}).get("months_negative", {})
                                  .get("within_12", {}) or {}).get("top1"),
            seconds=rec.get("seconds"))
    s["persistence"] = out.get("persistence", {}).get("n_checked")
    return s


if __name__ == "__main__":
    main()
