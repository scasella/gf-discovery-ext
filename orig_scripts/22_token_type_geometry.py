"""A1 -- token-type geometry discovery: does a label-blind pipeline find rings over TOKEN TYPES?

DECLARED EXPLORATORY. Not pre-registered; nothing here can change a gate verdict.

Why token types at all. Every supervised closure result in this project is a
statement about *month-token means*: script 11's control averages the full
16,384-latent reconstruction over the positions where a month token sits, and
`results/hand_cluster_gate.json` shows that the full readout closes at layer 9
(p_closed 0.0071) while the identified month latents do not. So the geometry
that survives the SAE lives over token types, not over latent subsets -- and
latent-subset discovery is separately exhausted (`results/month_cofire.json`,
`results/window_passer_audit.json`). This script asks the only remaining
discovery question: with ZERO label input, does clustering over per-token-type
mean reconstructions hand back a ring whose members turn out to be the months?

Pipeline, in the order it runs (labels touch nothing until stage 4):

  1. per-token-type mean SAE-code vector from the existing cache (no forward
     passes), projected through W_dec to 1152-d;
  2. LABEL-BLIND partitions: cosine graph + Leiden over a declared threshold
     grid with a declared size-profile selection rule, plus a declared
     recursive-refinement arm and an HDBSCAN arm;
  3. LABEL-FREE ring screening of every cluster of size 5-60 against a matched
     empirical background of random same-size token-type subsets;
  4. ONLY THEN labels: content of every flagged cluster, ordering against the
     known cycles, shuffled-label controls, both SVD sign conventions.

Everything in `DECLARED` below is frozen before the run and is written to
`results/token_type_geometry.json` as a stub BEFORE any clustering happens, so
the declaration has an mtime that precedes the results.

Disclosure of the one calibration that preceded the freeze: the cosine
thresholds grids and the edge budget were set from a LABEL-BLIND density probe
of layer 0 (percentiles of the centred and raw cosine distributions, edge counts
per threshold, and the Leiden size profile). No label set and no cluster content
was inspected. It is recorded in `declared_rules.calibration_disclosure`.

Memory: the accumulator is [n_types_extended, 16384] float64 (~1.4 GB), built in
one streaming pass over the CSR memmaps per layer and freed before the next.
Peak RSS measured ~1.7 GB.
"""
import argparse, gc, hashlib, json, os, sys, time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HUB_OFFLINE", "1")
sys.path.insert(0, str(REPO / "src"))
from gfdisc.saes import load_sae, get_decoder
from gfdisc.fits import (pca_2d, fit_circle, circularity, angles,
                         cyclic_order_score, degenerate)
from gfdisc.nulls import benjamini_hochberg
from gfdisc.tokens import DAYS, MONTHS

# ===========================================================================
# DECLARED RULES -- frozen before the run, written to disk before any results
# ===========================================================================
COUNT_FLOOR = 100            # occurrences required for a token type to enter the pool
COUNT_FLOOR_EXT = 30         # extended pool, used ONLY by the supervised colour sensitivity arm
D_MODEL = 1152
SCREEN_MIN, SCREEN_MAX = 5, 60      # cluster sizes that get ring-screened
N_PERM_CLUSTER = 20000       # uniform-angle null for screened/scored clusters
N_PERM_BACKGROUND = 2000     # uniform-angle null inside the broad background sweep
N_BACKGROUND_DRAWS = 500     # random same-size token-type subsets per represented size
N_PERM_ORDER = 20000         # permutation null inside cyclic_order_score
N_SHUFFLE_LABELS = 1000      # explicit shuffled-labels control
GRID_CENTRED = [round(0.10 + 0.05 * i, 2) for i in range(17)]   # 0.10 .. 0.90
GRID_RAW = [round(0.50 + 0.05 * i, 2) for i in range(10)]       # 0.50 .. 0.95
EDGE_BUDGET = 2_000_000      # a threshold whose graph exceeds this is skipped, not clustered
LEIDEN_RESOLUTION = 1.0
LEIDEN_SEED = 0
REFINE_RESOLUTIONS = [2, 4, 8, 16, 32, 64]
HDBSCAN_MIN_CLUSTER_SIZE = 5
HDBSCAN_MIN_SAMPLES = 5
SUPPORT_FRACTION = 2 / 3     # the project's support rule: ceil(2/3 * full_k) labels present
FLAG_ALPHA = 0.05

# Colour terms. Declared here, before any result is seen. Hue order is the
# standard colour-circle order; the ring closes because purple -> red is a real
# perceptual adjacency, which is what makes colours a genuine cyclic ontology
# rather than an ordinal one.
COLORS_HUE6 = ["red", "orange", "yellow", "green", "blue", "purple"]
COLORS_HUE7 = ["red", "orange", "yellow", "green", "blue", "purple", "pink"]   # declared variant
COLORS_HUE6_VIOLET = ["red", "orange", "yellow", "green", "blue", "violet"]    # declared variant
COLOR_FORMS = ("{w}", " {w}", "{W}", " {W}")   # single-token surface forms only

# Ordinal controls. Number words carry an order but no wrap-around and MUST fail
# closure; a run in which they close is invalid rather than interesting.
NUMS = ["one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"]
DIGITS = list("0123456789")


def declared_rules():
    return dict(
        representation=(
            "per token type t: mean over ALL cache rows whose tokens.i32 == t of the "
            "sparse SAE code vector, then projected mean_code @ W_dec -> 1152-d. b_dec "
            "is NOT added (the task text says 'W_dec projection'); it is a constant "
            "offset that PCA removes exactly and that cancels exactly in the centred "
            "cosine arm."),
        count_floor=COUNT_FLOOR,
        count_floor_extended=COUNT_FLOOR_EXT,
        extended_pool_use=("floor-30 pool is used ONLY for the supervised colour "
                           "sensitivity arm and its matched background; it never enters "
                           "discovery, clustering or the main background"),
        accumulator_dtype="float64, divided by count, cast to float32 before the W_dec matmul",
        clustering_arms=dict(
            centred_leiden=("cosine on pool-mean-centred means; edges where cos >= tau, "
                            "weights = cos; Leiden RBConfigurationVertexPartition, "
                            f"resolution {LEIDEN_RESOLUTION}, seed {LEIDEN_SEED}, "
                            "n_iterations=-1. PRIMARY ARM (declared primary a priori: the "
                            "pool-mean direction is common-mode and carries no "
                            "type-distinguishing information)"),
            raw_leiden="identical but cosine on the uncentred means. SECONDARY ARM",
            refined=("declared refinement of each Leiden arm's selected partition: any "
                     f"cluster larger than {SCREEN_MAX} is re-partitioned by Leiden on its "
                     f"induced subgraph at resolutions {REFINE_RESOLUTIONS} in turn, taking "
                     f"the first partition whose max sub-cluster size <= {SCREEN_MAX}; if "
                     "none achieves it, the last resolution's partition is used and only "
                     "sub-clusters in the screening band are kept. Clusters already in the "
                     "band pass through unchanged. Fully label-blind"),
            hdbscan=(f"HDBSCAN on L2-normalised pool-mean-centred means, metric euclidean, "
                     f"min_cluster_size {HDBSCAN_MIN_CLUSTER_SIZE}, min_samples "
                     f"{HDBSCAN_MIN_SAMPLES}, cluster_selection_method eom. SECONDARY ARM")),
        threshold_grids=dict(centred=GRID_CENTRED, raw=GRID_RAW, edge_budget=EDGE_BUDGET),
        threshold_selection_rule=(
            "sweep the whole declared grid and record it. A threshold is VALID iff its "
            "graph is within the edge budget AND the number of clusters of size >= 5 lies "
            "in [10, 300] AND the largest cluster holds <= 10% of the pool. Select the "
            "SMALLEST valid threshold. Fallback 1 if none is valid: the smallest threshold "
            "whose largest cluster holds <= 10% of the pool. Fallback 2: the largest "
            "threshold in the grid."),
        screening=(
            f"every cluster with {SCREEN_MIN} <= size <= {SCREEN_MAX}: PCA top-2 of the "
            "member means (fits.pca_2d), Kasa circle fit, fits.circularity -> gap_ratio, "
            "max_gap_deg, radius_ratio; p_closed = uniform-angle gap null at "
            f"{N_PERM_CLUSTER} permutations (identical to scripts/13_rerun_gate.gap_pvalue, "
            "with the null draws cached per (k, n_perm, seed) -- the same numbers, computed "
            "once)"),
        background=(
            f"for every cluster size represented, {N_BACKGROUND_DRAWS} random same-size "
            "token-type subsets drawn WITHOUT replacement from the identical >= "
            f"{COUNT_FLOOR}-count pool, through the identical screening at "
            f"{N_PERM_BACKGROUND} permutations. p_bg for an observed cluster = "
            "(1 + #{background draws with max_gap_deg <= observed}) / (1 + n_draws), "
            "computed on max_gap_deg rather than p_closed so the two permutation counts "
            "never have to be reconciled (at fixed k the gap p-value is monotone in the "
            "gap). A secondary occurrence-count-decile-matched background is drawn for "
            "flagged and label-bearing clusters only."),
        flag_rule=(f"FLAGGED iff p_closed < {FLAG_ALPHA} AND p_bg <= {FLAG_ALPHA}. "
                   "BH-FDR over p_closed within (layer, arm) and globally is reported "
                   "alongside but does not change the flag."),
        label_stage=(
            "runs only after every partition and every screening number is frozen. A label "
            "counts as PRESENT in a cluster if any of its qualifying token-type variants is "
            "a member; when several variants are present the highest-cache-count variant is "
            "the one scored. A cluster is label-scored if it is flagged, or if it holds >= "
            "ceil(2/3 * full_k) labels of a set -- the latter at ANY size, including "
            "clusters far larger than the screening band, which is how the negative space "
            "gets reported."),
        ordering=(
            f"fits.cyclic_order_score at {N_PERM_ORDER} permutations, max over the 2*full_k "
            "rotation/reflection symmetries, with the survivors' TRUE cycle positions. "
            "Reported from the cluster-level PCA angles (primary) and from a PCA of the "
            "label members alone (secondary). Both SVD sign conventions are reported: the "
            "max-over-symmetries rho is reflection-invariant by construction, so the "
            "rotation-only rho is reported per sign convention as the quantity that is not. "
            f"A shuffled-labels control of {N_SHUFFLE_LABELS} draws is reported separately "
            "from the internal permutation null."),
        ordering_is_not_evidence=(
            "ESTABLISHED IN THIS PROJECT AND CARRIED FORWARD: the ordering tier accepts "
            "random subsets at ~0.16 and an ordinal control out-orders months at every "
            "layer. Ordering alone is never evidence here. Closure against the matched "
            "background is the discriminating tier."),
        label_sets=dict(
            months="12, cyclic, ids from results/token_map.json",
            weekdays="7, cyclic, ids from results/token_map.json",
            colors_hue6=f"{COLORS_HUE6}, cyclic, single-token forms {COLOR_FORMS}",
            colors_hue7_pink=f"{COLORS_HUE7}, cyclic, declared variant",
            colors_hue6_violet=f"{COLORS_HUE6_VIOLET}, cyclic, declared variant",
            numbers_one_to_ten=(f"{NUMS}, ORDINAL CONTROL -- must not close. Truncated at "
                                "ten because ' eleven' (28) and ' twelve' (44) fall below "
                                "the count floor; the truncation is declared, not chosen "
                                "after seeing a result"),
            digits_0_9=("0-9, scored on the mod-10 cycle. CONFOUND DECLARED IN ADVANCE: at "
                        "token-type level the token identity IS the value, so a digit ring "
                        "is a statement about token-type geometry and can never be read as "
                        "a base-10 helix or a depth claim")),
        seed_rule="int(md5(f'{layer}|{arm}|{cluster_id}').hexdigest()[:6], 16), deterministic",
        calibration_disclosure=(
            "Before the freeze, a LABEL-BLIND probe at layer 0 measured the centred and raw "
            "cosine percentiles, the edge count per candidate threshold, and the Leiden "
            "size profile, to set the two grids and the edge budget so that a valid "
            "threshold exists. No label set, no token string and no cluster content was "
            "inspected. Nothing else was calibrated."),
        layers_0_4_note=(
            "Layers 0-4 are embedding-adjacent: a hit there validates the pipeline and says "
            "the geometry is present near the input, NOT that it is computed at depth."),
        context_averaging_note=(
            "A per-token-type mean over all corpus occurrences is a context-AVERAGED object, "
            "closer to a static embedding than to a computed feature. A ring here is a "
            "statement about the model's average representation of those token types, and "
            "is weaker than a claim about a ring the model uses in context."),
    )


# ===========================================================================
# stage 1 -- per-token-type mean reconstructions
# ===========================================================================
def build_pool(root, layer0=0):
    """Token types over the count floors. tokens.i32 is byte-identical across layers."""
    tok = np.fromfile(Path(root) / f"layer_{layer0}" / "tokens.i32", dtype=np.int32)
    cnt = np.bincount(tok)
    ext = np.flatnonzero(cnt >= COUNT_FLOOR_EXT)
    main = np.flatnonzero(cnt >= COUNT_FLOOR)
    return tok, cnt, main, ext


def type_means(root, layer, tok, cnt, ext_ids, chunk=200_000):
    """Streaming groupby of the CSR cache by token id -> [n_ext, 16384] mean codes."""
    d = Path(root) / f"layer_{layer}"
    meta = json.loads((d / "meta.json").read_text())
    d_sae = int(meta["d_sae"])
    indptr = np.load(d / "indptr.npy")
    indices = np.memmap(d / "indices.u16", dtype=np.uint16, mode="r")
    values = np.memmap(d / "values.f16", dtype=np.float16, mode="r")
    lut = np.full(int(tok.max()) + 1, -1, dtype=np.int64)
    lut[ext_ids] = np.arange(len(ext_ids))
    tidx = lut[tok]
    acc = np.zeros((len(ext_ids), d_sae), dtype=np.float64)
    n_rows_used = 0
    for lo in range(0, len(tok), chunk):
        hi = min(lo + chunk, len(tok))
        a, b = int(indptr[lo]), int(indptr[hi])
        cols = np.asarray(indices[a:b], dtype=np.int64)
        vals = np.asarray(values[a:b], dtype=np.float32)
        ct = indptr[lo + 1:hi + 1] - indptr[lo:hi]
        rid = np.repeat(tidx[lo:hi], ct)
        m = rid >= 0
        np.add.at(acc, (rid[m], cols[m]), vals[m])
        n_rows_used += int((tidx[lo:hi] >= 0).sum())
    acc /= cnt[ext_ids][:, None]
    return acc, d_sae, n_rows_used, meta


def project(acc, W, chunk=2000):
    out = np.empty((acc.shape[0], W.shape[1]), dtype=np.float32)
    for lo in range(0, acc.shape[0], chunk):
        hi = min(lo + chunk, acc.shape[0])
        out[lo:hi] = acc[lo:hi].astype(np.float32) @ W
    return out


# ===========================================================================
# stage 2 -- label-blind partitions
# ===========================================================================
def cosine_matrix(R, centre):
    X = R - R.mean(0, keepdims=True) if centre else R.astype(np.float64)
    X = np.asarray(X, dtype=np.float32)
    X = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
    S = X @ X.T
    np.fill_diagonal(S, -2.0)
    return S


def leiden(n, ei, ej, w, resolution=LEIDEN_RESOLUTION, seed=LEIDEN_SEED):
    import igraph as ig, leidenalg as la
    g = ig.Graph(n=n, edges=list(zip(ei.tolist(), ej.tolist())))
    if len(w):
        g.es["weight"] = np.asarray(w, dtype=float).tolist()
        p = la.find_partition(g, la.RBConfigurationVertexPartition, weights="weight",
                              resolution_parameter=resolution, seed=seed, n_iterations=-1)
    else:
        p = la.find_partition(g, la.RBConfigurationVertexPartition,
                              resolution_parameter=resolution, seed=seed, n_iterations=-1)
    return [sorted(int(x) for x in c) for c in p]


def sweep_and_select(S, grid, n_types):
    """The declared threshold sweep + selection rule. Returns (rows, tau, clusters)."""
    iu = np.triu_indices(len(S), 1)
    sv = S[iu]
    rows, partitions = [], {}
    for tau in grid:
        m = sv >= tau
        ne = int(m.sum())
        row = dict(tau=tau, n_edges=ne)
        if ne > EDGE_BUDGET:
            row.update(skipped="over declared edge budget", valid=False)
            rows.append(row); continue
        t0 = time.time()
        cl = leiden(len(S), iu[0][m], iu[1][m], sv[m])
        sizes = sorted((len(c) for c in cl), reverse=True)
        n5 = int(sum(1 for s in sizes if s >= 5))
        row.update(seconds=round(time.time() - t0, 1),
                   n_clusters_total=len(cl), n_clusters_ge5=n5,
                   n_clusters_in_screen_band=int(sum(1 for s in sizes
                                                     if SCREEN_MIN <= s <= SCREEN_MAX)),
                   max_size=int(sizes[0]) if sizes else 0,
                   coverage_ge5=float(sum(s for s in sizes if s >= 5) / n_types),
                   top_sizes=[int(s) for s in sizes[:8]])
        row["valid"] = bool(10 <= n5 <= 300 and row["max_size"] <= 0.10 * n_types)
        rows.append(row)
        partitions[tau] = cl
    valid = [r for r in rows if r.get("valid")]
    if valid:
        tau, rule = valid[0]["tau"], "primary: smallest valid threshold"
    else:
        fb = [r for r in rows if "skipped" not in r and r.get("max_size", 1e9) <= 0.10 * n_types]
        if fb:
            tau, rule = fb[0]["tau"], "fallback 1: smallest threshold with max cluster <= 10% of pool"
        else:
            ok = [r for r in rows if "skipped" not in r]
            tau, rule = (ok[-1]["tau"], "fallback 2: largest clusterable threshold in grid") \
                if ok else (None, "no threshold was clusterable within the edge budget")
    return rows, tau, rule, (partitions.get(tau) or [])


def refine(clusters, S, tau):
    """Declared refinement: split oversized clusters on their induced subgraph."""
    out, log = [], []
    for c in clusters:
        if len(c) <= SCREEN_MAX:
            out.append(c); continue
        idx = np.asarray(c)
        sub = S[np.ix_(idx, idx)]
        iu = np.triu_indices(len(idx), 1)
        sv = sub[iu]
        m = sv >= tau
        chosen, used = None, None
        for res in REFINE_RESOLUTIONS:
            parts = leiden(len(idx), iu[0][m], iu[1][m], sv[m], resolution=res)
            chosen, used = parts, res
            if max((len(p) for p in parts), default=0) <= SCREEN_MAX:
                break
        subs = [sorted(int(idx[i]) for i in p) for p in (chosen or [])]
        kept = [p for p in subs if len(p) <= SCREEN_MAX]
        out.extend(kept)
        log.append(dict(parent_size=len(c), resolution_used=used,
                        n_subclusters=len(subs), n_kept=len(kept),
                        kept_sizes=sorted((len(p) for p in kept), reverse=True)[:12]))
    return out, log


def run_hdbscan(R):
    import hdbscan as _h
    X = R - R.mean(0, keepdims=True)
    X = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
    lab = _h.HDBSCAN(min_cluster_size=HDBSCAN_MIN_CLUSTER_SIZE,
                     min_samples=HDBSCAN_MIN_SAMPLES,
                     metric="euclidean",
                     cluster_selection_method="eom").fit_predict(X.astype(np.float64))
    cl = [sorted(np.flatnonzero(lab == g).tolist()) for g in sorted(set(lab.tolist()) - {-1})]
    return cl, int((lab == -1).sum())


# ===========================================================================
# stage 3 -- label-free ring screening + matched background
# ===========================================================================
_NULL = {}


def gap_null(k, n_perm, seed=0):
    """The uniform-angle max-gap null of scripts/13_rerun_gate.gap_pvalue, cached per k."""
    key = (int(k), int(n_perm), int(seed))
    if key not in _NULL:
        rng = np.random.default_rng(seed)
        a = np.sort(rng.uniform(0, 2 * np.pi, size=(n_perm, k)), axis=1)
        _NULL[key] = np.diff(np.concatenate([a, a[:, :1] + 2 * np.pi], axis=1), axis=1).max(1)
    return _NULL[key]


def gap_p(k, max_gap_rad, n_perm, seed=0):
    g = gap_null(k, n_perm, seed)
    return float((1 + (g <= max_gap_rad).sum()) / (1 + n_perm))


def screen(M, n_perm, seed=0):
    """Label-free ring screening of a point set. None if the set carries no geometry."""
    M = np.asarray(M, dtype=np.float64)
    if len(M) < 3 or degenerate(M):
        return None
    xy, ratios, Vt = pca_2d(M)
    cx, cy, r, resid = fit_circle(xy)
    c = circularity(xy, cx, cy, r)
    return dict(k=int(len(M)), gap_ratio=c["gap_ratio"], max_gap_deg=c["max_gap_deg"],
                radius_ratio=c["radius_ratio"], circle_resid_over_r=float(resid),
                top2_var=float(ratios[:2].sum()), pc1_var=float(ratios[0]),
                p_closed=gap_p(len(M), np.radians(c["max_gap_deg"]), n_perm, seed))


def background_for_size(R, pool_n, k, seed, n_draws=N_BACKGROUND_DRAWS, weights=None):
    """Random same-size token-type subsets through the identical screening."""
    rng = np.random.default_rng(seed)
    gaps, ps, rr, t2 = [], [], [], []
    for _ in range(n_draws):
        if weights is None:
            idx = rng.choice(pool_n, size=k, replace=False)
        else:
            idx = weights(rng, k)
        s = screen(R[idx], N_PERM_BACKGROUND, seed=0)
        if s is None:
            continue
        gaps.append(s["max_gap_deg"]); ps.append(s["p_closed"])
        rr.append(s["radius_ratio"]); t2.append(s["top2_var"])
    gaps = np.asarray(gaps)
    return dict(k=int(k), n_draws=int(n_draws), n_scorable=int(len(gaps)),
                rate_p_closed_lt_05=float(np.mean(np.asarray(ps) < 0.05)) if len(ps) else None,
                max_gap_deg_pct=[float(x) for x in np.percentile(gaps, [1, 5, 25, 50])]
                if len(gaps) else None,
                mean_max_gap_deg=float(gaps.mean()) if len(gaps) else None,
                mean_radius_ratio=float(np.mean(rr)) if rr else None,
                mean_top2_var=float(np.mean(t2)) if t2 else None,
                _gaps=gaps)


def p_bg_of(bg, max_gap_deg):
    g = bg["_gaps"]
    return float((1 + int((g <= max_gap_deg).sum())) / (1 + len(g))) if len(g) else None


def within_cluster_background(xy_all, n_sub, seed, n_draws=N_BACKGROUND_DRAWS):
    """The exactly-matched null for a label subset read in its CLUSTER's PCA plane.

    `background_for_size` draws random token types and gives each draw its OWN
    top-2 plane, which is not the comparison being made when a label subset is
    read off the angles of the cluster that contains it. This draws random
    same-size subsets OF THAT CLUSTER and measures their gap in the SAME plane,
    so the only thing that varies is which members were chosen.
    """
    P_all = np.asarray(xy_all, dtype=np.float64)
    if len(P_all) <= n_sub or n_sub < 3:
        return None
    rng = np.random.default_rng(seed)
    gaps = []
    for _ in range(n_draws):
        P = P_all[rng.choice(len(P_all), size=n_sub, replace=False)]
        cx, cy, r, _ = fit_circle(P)
        gaps.append(circularity(P, cx, cy, r)["max_gap_deg"])
    g = np.asarray(gaps)
    return dict(n_draws=int(n_draws), n_cluster_members=int(len(P_all)), n_sub=int(n_sub),
                mean_max_gap_deg=float(g.mean()),
                max_gap_deg_pct=[float(x) for x in np.percentile(g, [1, 5, 25, 50])], _g=g)


# ===========================================================================
# stage 4 -- labels (nothing above this line may see a label)
# ===========================================================================
def resolve_label_sets(cnt, pool_ids, ext_ids, tokenizer, token_map):
    """Token-type ids per label, per declared label set. Pool membership decided by count."""
    pool = set(int(x) for x in pool_ids)
    ext = set(int(x) for x in ext_ids)

    def enc_forms(w):
        ids = []
        for f in COLOR_FORMS:
            s = f.format(w=w, W=w.capitalize())
            e = tokenizer.encode(s, add_special_tokens=False)
            if len(e) == 1:
                ids.append(int(e[0]))
        return sorted(set(ids))

    def pack(names, id_map, cyclic, note=""):
        labels = []
        for i, n in enumerate(names):
            ids = [int(x) for x in id_map[n]]
            in_pool = sorted((x for x in ids if x in pool),
                             key=lambda x: -int(cnt[x]))
            in_ext = sorted((x for x in ids if x in ext), key=lambda x: -int(cnt[x]))
            labels.append(dict(name=n, index=i, all_ids=ids,
                               ids_in_pool=in_pool, ids_in_ext=in_ext,
                               primary_id=in_pool[0] if in_pool else None,
                               primary_id_ext=in_ext[0] if in_ext else None,
                               counts={str(x): int(cnt[x]) for x in ids if x < len(cnt)}))
        return dict(names=names, full_k=len(names), cyclic=cyclic, note=note, labels=labels)

    sets = {}
    sets["months"] = pack(MONTHS, token_map["months"], True)
    sets["weekdays"] = pack(DAYS, token_map["days"], True)
    for key, names in (("colors_hue6", COLORS_HUE6), ("colors_hue7_pink", COLORS_HUE7),
                       ("colors_hue6_violet", COLORS_HUE6_VIOLET)):
        sets[key] = pack(names, {w: enc_forms(w) for w in names}, True,
                         "hue-circle order, declared before the run")
    sets["numbers_one_to_ten"] = pack(
        NUMS, {w: enc_forms(w) for w in NUMS}, False,
        "ORDINAL CONTROL: must not close. Truncated at ten by the declared count floor")
    sets["digits_0_9"] = pack(
        DIGITS, {w: enc_forms(w) for w in DIGITS}, True,
        "mod-10 reading. CONFOUND: at token-type level the token identity IS the value, "
        "so a ring here is token-type geometry and never a helix or a depth claim")
    return sets


def present_labels(lset, member_set, field="ids_in_pool"):
    """(label index, chosen token-type id) for every label with a variant in the cluster."""
    out = []
    for lab in lset["labels"]:
        hits = [x for x in lab[field] if x in member_set]
        if hits:
            out.append((lab["index"], int(hits[0]), lab["name"]))   # ids already count-sorted
    return out


def score_ordering(xy_all, pos_in_cluster, label_index, full_k, seed):
    """Ordering + closure for the label members, from an already-computed 2-D embedding."""
    P = np.asarray(xy_all)[np.asarray(pos_in_cluster)]
    cx, cy, r, resid = fit_circle(P)
    c = circularity(P, cx, cy, r)
    ang = angles(P, cx, cy)
    k = len(P)
    s = cyclic_order_score(ang, np.asarray(label_index), k, n_perm=N_PERM_ORDER,
                           seed=seed, full_k=full_k)
    # rotation-only rho, per SVD sign convention: the max-over-symmetries rho is
    # reflection-invariant by construction, so this is the number that is not.
    rot = {}
    for sign, a in (("as_returned", ang), ("pc2_negated", -ang)):
        rank = np.empty(k, dtype=np.float64)
        jit = np.random.default_rng(seed).permutation(k)
        order = np.lexsort((jit, a)); rank[order] = np.arange(k)
        best = -2.0
        lab = np.asarray(label_index)
        for shift in range(full_k):
            ref = np.argsort(np.argsort((lab + shift) % full_k)).astype(np.float64)
            d = rank - ref
            best = max(best, 1.0 - 6.0 * float(d @ d) / (k * (k * k - 1)))
        rot[sign] = float(best)
    return dict(n_labels=k, label_index=[int(x) for x in label_index],
                rho=s["rho"], p_order=s["p"], symmetry=[int(x) for x in s["symmetry"]],
                null_mean=s["null_mean"], null_p95=s["null_p95"],
                rho_rotation_only=rot,
                gap_ratio=c["gap_ratio"], max_gap_deg=c["max_gap_deg"],
                radius_ratio=c["radius_ratio"],
                p_closed=gap_p(k, np.radians(c["max_gap_deg"]), N_PERM_CLUSTER, seed=0),
                top2_note="angles taken in the 2-D embedding named by the caller")


def shuffled_label_control(xy_all, pos_in_cluster, label_index, full_k, seed, observed_rho):
    """Explicit shuffled-labels control, independent of cyclic_order_score's internal null."""
    P = np.asarray(xy_all)[np.asarray(pos_in_cluster)]
    cx, cy, r, _ = fit_circle(P)
    ang = angles(P, cx, cy)
    k = len(P)
    rng = np.random.default_rng(seed + 991)
    lab = np.asarray(label_index)
    null = []
    for _ in range(N_SHUFFLE_LABELS):
        s = cyclic_order_score(ang, lab[rng.permutation(k)], k, n_perm=1, seed=seed, full_k=full_k)
        null.append(s["rho"])
    null = np.asarray(null)
    return dict(n_shuffles=N_SHUFFLE_LABELS, mean_rho=float(null.mean()),
                p95_rho=float(np.percentile(null, 95)), max_rho=float(null.max()),
                p_shuffled=float((1 + (null >= observed_rho).sum()) / (1 + len(null))))


# ===========================================================================
def cid_seed(layer, arm, cid):
    return int(hashlib.md5(f"{layer}|{arm}|{cid}".encode()).hexdigest()[:6], 16)


def member_hash(members):
    return hashlib.md5(",".join(str(int(x)) for x in sorted(members)).encode()).hexdigest()[:12]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="cache/sparse_big")
    ap.add_argument("--layers", default="0,1,2,3,4,5,6,7,8,9,10")
    ap.add_argument("--l0", default="big")
    ap.add_argument("--out", default="results/token_type_geometry.json")
    ap.add_argument("--partial-dir", default=None)
    a = ap.parse_args()
    t_start = time.time()
    layers = [int(x) for x in a.layers.split(",")]
    partial = Path(a.partial_dir) if a.partial_dir else None
    if partial:
        partial.mkdir(parents=True, exist_ok=True)

    # ---- write the declaration to disk BEFORE anything is computed ----
    stub = dict(status="DECLARATION STUB -- rules frozen, no results yet",
                script="scripts/22_token_type_geometry.py",
                declared_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
                declared_rules=declared_rules(), layers=layers)
    Path(a.out).write_text(json.dumps(stub, indent=2))
    print("=== DECLARED RULES (frozen, written to disk before any result) ===", flush=True)
    print(json.dumps(declared_rules(), indent=2)[:4000], flush=True)

    tok, cnt, pool_ids, ext_ids = build_pool(a.root)
    n_types = len(pool_ids)
    pool_pos = {int(t): i for i, t in enumerate(pool_ids)}
    ext_pos = {int(t): i for i, t in enumerate(ext_ids)}
    print(f"\npool: {n_types} token types with >= {COUNT_FLOOR} occurrences "
          f"({int(cnt[pool_ids].sum())} of {len(tok)} cache rows); extended pool "
          f"{len(ext_ids)} types at floor {COUNT_FLOOR_EXT}", flush=True)

    # tokens.i32 identical across layers -- verified, not assumed
    same = {}
    for L in layers:
        t2 = np.fromfile(Path(a.root) / f"layer_{L}" / "tokens.i32", dtype=np.int32)
        same[str(L)] = bool(len(t2) == len(tok) and np.array_equal(t2, tok))
        del t2
    assert all(same.values()), f"tokens.i32 differs across layers: {same}"

    from transformers import AutoTokenizer
    from gfdisc.env import MODEL_NAME
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=os.environ.get("HF_TOKEN"))
    token_map = json.loads((REPO / "results/token_map.json").read_text())
    LSETS = resolve_label_sets(cnt, pool_ids, ext_ids, tokenizer, token_map)
    decoded_pool = [tokenizer.decode([int(t)]) for t in pool_ids]

    label_availability = {}
    for key, ls in LSETS.items():
        have = [l["name"] for l in ls["labels"] if l["primary_id"] is not None]
        have_ext = [l["name"] for l in ls["labels"] if l["primary_id_ext"] is not None]
        label_availability[key] = dict(
            full_k=ls["full_k"], n_in_pool=len(have), names_in_pool=have,
            missing_from_pool=[l["name"] for l in ls["labels"] if l["primary_id"] is None],
            n_in_extended_pool=len(have_ext),
            missing_from_extended_pool=[l["name"] for l in ls["labels"]
                                        if l["primary_id_ext"] is None],
            min_labels_needed=int(np.ceil(SUPPORT_FRACTION * ls["full_k"])))
    print("\nlabel availability (post-freeze bookkeeping only):", flush=True)
    for k, v in label_availability.items():
        print(f"  {k:22s} {v['n_in_pool']}/{v['full_k']} in pool; missing "
              f"{v['missing_from_pool']}", flush=True)

    layer_out = []
    for L in layers:
        tL = time.time()
        print(f"\n================ layer {L} ================", flush=True)
        acc, d_sae, n_rows_used, meta = type_means(a.root, L, tok, cnt, ext_ids)
        sae = load_sae(L, a.l0, token=os.environ.get("HF_TOKEN"))
        W = get_decoder(sae); del sae
        R_ext = project(acc, W)
        del acc; gc.collect()
        R = R_ext[np.asarray([ext_pos[int(t)] for t in pool_ids])]
        print(f"  means built: {R.shape} from {n_rows_used} cache rows "
              f"({round(time.time()-tL,1)}s)", flush=True)

        arms, screened_all = {}, []
        S_centred = cosine_matrix(R, True)
        S_raw = cosine_matrix(R, False)
        for arm, S, grid in (("centred_leiden", S_centred, GRID_CENTRED),
                             ("raw_leiden", S_raw, GRID_RAW)):
            rows, tau, rule, clusters = sweep_and_select(S, grid, n_types)
            ref, rlog = refine(clusters, S, tau) if tau is not None else ([], [])
            arms[arm] = dict(sweep=rows, selected_tau=tau, selection_rule=rule,
                             clusters=clusters)
            arms[arm.replace("_leiden", "_refined")] = dict(
                derived_from=arm, selected_tau=tau, refinement_log=rlog, clusters=ref)
            print(f"  [{arm}] tau={tau} ({rule}) -> {len(clusters)} clusters, "
                  f"refined -> {len(ref)}", flush=True)
        hcl, n_noise = run_hdbscan(R)
        arms["hdbscan"] = dict(clusters=hcl, n_noise=int(n_noise))
        print(f"  [hdbscan] {len(hcl)} clusters, {n_noise} noise points", flush=True)

        # ---------- stage 3: screening + matched background (still label-free) ----------
        sizes_needed = sorted({len(c) for arm in arms.values() for c in arm["clusters"]
                               if SCREEN_MIN <= len(c) <= SCREEN_MAX})
        bg = {}
        tb = time.time()
        for k in sizes_needed:
            bg[k] = background_for_size(R, n_types, k, seed=10_000 * L + k)
        print(f"  background: {len(sizes_needed)} sizes x {N_BACKGROUND_DRAWS} draws "
              f"({round(time.time()-tb,1)}s)", flush=True)

        for arm, d in arms.items():
            rowsA = []
            for ci, c in enumerate(d["clusters"]):
                if not (SCREEN_MIN <= len(c) <= SCREEN_MAX):
                    continue
                s = screen(R[np.asarray(c)], N_PERM_CLUSTER, seed=0)
                if s is None:
                    continue
                s.update(layer=L, arm=arm, cluster_id=ci,
                         member_hash=member_hash(c),
                         p_bg=p_bg_of(bg[len(c)], s["max_gap_deg"]),
                         background_rate_p_closed_lt_05=bg[len(c)]["rate_p_closed_lt_05"],
                         background_mean_max_gap_deg=bg[len(c)]["mean_max_gap_deg"])
                s["flagged"] = bool(s["p_closed"] < FLAG_ALPHA and s["p_bg"] is not None
                                    and s["p_bg"] <= FLAG_ALPHA)
                rowsA.append(s)
            if rowsA:
                keep = benjamini_hochberg([r["p_closed"] for r in rowsA])
                for r, kk in zip(rowsA, keep):
                    r["bh_pass_within_layer_arm"] = bool(kk)
            d["screened"] = rowsA
            screened_all.extend(rowsA)
            nf = sum(r["flagged"] for r in rowsA)
            print(f"  [{arm}] screened {len(rowsA)} clusters in band, {nf} FLAGGED", flush=True)

        # ---------- stage 4: labels ----------
        need = {k: int(np.ceil(SUPPORT_FRACTION * v["full_k"])) for k, v in LSETS.items()}
        label_rows, placement = [], []
        for arm, d in arms.items():
            scr_by_id = {r["cluster_id"]: r for r in d["screened"]}
            for ci, c in enumerate(d["clusters"]):
                ms = set(int(x) for x in np.asarray(pool_ids)[np.asarray(c)])
                srow = scr_by_id.get(ci)
                flagged = bool(srow and srow["flagged"])
                hits = {k: present_labels(LSETS[k], ms) for k in LSETS}
                interesting = flagged or any(len(h) >= need[k] for k, h in hits.items())
                if not interesting:
                    continue
                idx = np.asarray(c)
                xy_all, ratios, _ = pca_2d(R[idx]) if len(idx) >= 3 else (None, None, None)
                strings = [decoded_pool[i] for i in c]
                base = dict(layer=L, arm=arm, cluster_id=ci, size=len(c),
                            member_hash=member_hash(c), flagged=flagged,
                            screened=srow is not None,
                            screening=None if srow is None else
                            {k: srow[k] for k in ("k", "gap_ratio", "max_gap_deg",
                                                  "radius_ratio", "p_closed", "p_bg",
                                                  "top2_var",
                                                  "background_rate_p_closed_lt_05")},
                            in_screen_band=bool(SCREEN_MIN <= len(c) <= SCREEN_MAX),
                            token_strings=strings if len(c) <= 80 else strings[:80],
                            n_strings_truncated=max(0, len(c) - 80),
                            members_token_ids=[int(x) for x in np.asarray(pool_ids)[idx]]
                            if len(c) <= 80 else None,
                            cluster_top2_var=None if ratios is None else float(ratios[:2].sum()))
                scored = {}
                for key, h in hits.items():
                    if len(h) < need[key] or xy_all is None:
                        if h:
                            scored[key] = dict(n_labels_present=len(h),
                                               names=[x[2] for x in h],
                                               scored=False,
                                               reason=f"fewer than {need[key]} of "
                                                      f"{LSETS[key]['full_k']} labels present")
                        continue
                    pos = [c.index(pool_pos[tid]) for _, tid, _ in h]
                    li = [i for i, _, _ in h]
                    sd = cid_seed(L, arm, ci)
                    prim = score_ordering(xy_all, pos, li, LSETS[key]["full_k"], sd)
                    xy_lab, rlab, _ = pca_2d(R[np.asarray([pool_pos[t] for _, t, _ in h])])
                    sec = score_ordering(xy_lab, list(range(len(h))), li,
                                         LSETS[key]["full_k"], sd)
                    sec["top2_var_label_only"] = float(rlab[:2].sum())
                    ctrl = shuffled_label_control(xy_all, pos, li, LSETS[key]["full_k"],
                                                  sd, prim["rho"])
                    bgl = background_for_size(R, n_types, len(h), seed=77_000 + 100 * L + len(h))
                    wcb = within_cluster_background(xy_all, len(h), sd)
                    scored[key] = dict(
                        n_labels_present=len(h), names=[x[2] for x in h], scored=True,
                        cyclic=LSETS[key]["cyclic"], full_k=LSETS[key]["full_k"],
                        from_cluster_pca=prim, from_label_only_pca=sec,
                        shuffled_labels_control=ctrl,
                        label_subset_p_bg_cluster_pca=p_bg_of(bgl, prim["max_gap_deg"]),
                        label_subset_p_bg_label_pca=p_bg_of(bgl, sec["max_gap_deg"]),
                        label_subset_p_within_cluster=None if wcb is None else
                        float((1 + int((wcb["_g"] <= prim["max_gap_deg"]).sum()))
                              / (1 + len(wcb["_g"]))),
                        within_cluster_background=None if wcb is None else
                        {k2: v2 for k2, v2 in wcb.items() if k2 != "_g"},
                        background_rate_p_closed_lt_05_at_this_k=bgl["rate_p_closed_lt_05"],
                        supervision_note=(
                            "from_cluster_pca reads the label members' angles off the "
                            "label-blind cluster's own top-2 plane. from_label_only_pca "
                            "re-runs PCA on the label token types alone and is SUPERVISED -- "
                            "it selects those types by name, so it is a control/anchor, "
                            "never a discovery result."),
                        note=LSETS[key]["note"])
                base["label_scores"] = scored
                label_rows.append(base)

            # negative space: where each label set actually lands in this arm
            for key, ls in LSETS.items():
                ids_pool = {int(x) for l in ls["labels"] for x in l["ids_in_pool"]}
                best = None
                for ci, c in enumerate(d["clusters"]):
                    ms = set(int(x) for x in np.asarray(pool_ids)[np.asarray(c)])
                    n_hit = len(present_labels(ls, ms))
                    if best is None or n_hit > best[1]:
                        best = (ci, n_hit, len(c))
                if best is None:
                    continue
                ci, n_hit, sz = best
                srow = {r["cluster_id"]: r for r in d["screened"]}.get(ci)
                placement.append(dict(
                    layer=L, arm=arm, label_set=key, full_k=ls["full_k"],
                    n_labels_in_pool=sum(1 for l in ls["labels"] if l["primary_id"]),
                    best_cluster_id=ci, best_cluster_size=sz,
                    n_labels_in_best_cluster=n_hit,
                    fraction_of_set_together=round(n_hit / max(ls["full_k"], 1), 3),
                    best_cluster_in_screen_band=bool(SCREEN_MIN <= sz <= SCREEN_MAX),
                    best_cluster_screening=None if srow is None else
                    {k: srow[k] for k in ("gap_ratio", "max_gap_deg", "radius_ratio",
                                          "p_closed", "p_bg", "top2_var", "flagged")},
                    reason_unscreened=None if srow is not None else
                    (f"cluster size {sz} outside the declared screening band "
                     f"[{SCREEN_MIN},{SCREEN_MAX}]" if not (SCREEN_MIN <= sz <= SCREEN_MAX)
                     else "screening returned None (degenerate point set)")))

        # ---------- declared supervised colour sensitivity arm (floor-30 pool) ----------
        colour_sens = []
        for key in ("colors_hue6", "colors_hue7_pink", "colors_hue6_violet"):
            ls = LSETS[key]
            ids = [l["primary_id_ext"] for l in ls["labels"]]
            if any(i is None for i in ids):
                colour_sens.append(dict(label_set=key, scored=False,
                                        reason="a hue term has no single-token variant even "
                                               f"at the floor-{COUNT_FLOOR_EXT} pool"))
                continue
            P = R_ext[np.asarray([ext_pos[int(i)] for i in ids])]
            xy, ratios, _ = pca_2d(P)
            sd = cid_seed(L, "colour_sensitivity", 0)
            prim = score_ordering(xy, list(range(len(ids))), list(range(len(ids))),
                                  ls["full_k"], sd)
            n_ext = len(ext_ids)
            bgc = background_for_size(R_ext, n_ext, len(ids), seed=555_000 + L)
            colour_sens.append(dict(
                label_set=key, scored=True, status="SUPERVISED SENSITIVITY, not discovery",
                token_ids=[int(i) for i in ids],
                token_strings=[tokenizer.decode([int(i)]) for i in ids],
                counts=[int(cnt[int(i)]) for i in ids],
                top2_var=float(ratios[:2].sum()), **prim,
                p_bg_vs_floor30_pool=p_bg_of(bgc, prim["max_gap_deg"]),
                background_rate_p_closed_lt_05=bgc["rate_p_closed_lt_05"],
                shuffled_labels_control=shuffled_label_control(
                    xy, list(range(len(ids))), list(range(len(ids))), ls["full_k"],
                    sd, prim["rho"])))

        lr = dict(
            layer=L, nominal_l0=meta.get("nominal_l0"), d_sae=d_sae,
            n_cache_rows_used=int(n_rows_used),
            embedding_adjacent=bool(L <= 4),
            depth_note=("layers 0-4 are embedding-adjacent: a hit here is a PIPELINE "
                        "VALIDATION and a statement about near-input geometry, not a depth "
                        "claim" if L <= 4 else
                        "layer >= 5: not embedding-adjacent, but the representation is still "
                        "a context-averaged token-type mean"),
            arms={k: dict(selected_tau=v.get("selected_tau"),
                          selection_rule=v.get("selection_rule"),
                          derived_from=v.get("derived_from"),
                          n_noise=v.get("n_noise"),
                          n_clusters=len(v["clusters"]),
                          size_histogram_top=sorted((len(c) for c in v["clusters"]),
                                                    reverse=True)[:10],
                          n_screened=len(v.get("screened", [])),
                          n_flagged=int(sum(r["flagged"] for r in v.get("screened", []))),
                          sweep=v.get("sweep"), refinement_log=v.get("refinement_log"),
                          screened=v.get("screened"))
                  for k, v in arms.items()},
            background_by_size={str(k): {kk: vv for kk, vv in b.items() if kk != "_gaps"}
                                for k, b in bg.items()},
            label_scored_clusters=label_rows,
            label_placement=placement,
            colour_supervised_sensitivity=colour_sens,
            seconds=round(time.time() - tL, 1))
        layer_out.append(lr)
        if partial:
            (partial / f"layer_{L}.json").write_text(json.dumps(lr, indent=2))
        nf = sum(v["n_flagged"] for v in lr["arms"].values())
        print(f"  layer {L} done in {lr['seconds']}s: {nf} flagged clusters, "
              f"{len(label_rows)} label-scored clusters", flush=True)
        del R, R_ext, S_centred, S_raw, arms
        gc.collect()

    # ---------------------------- verdict ----------------------------
    def flagged_rows():
        for lr in layer_out:
            for arm, v in lr["arms"].items():
                for r in (v["screened"] or []):
                    if r["flagged"]:
                        yield lr["layer"], arm, r

    flags = list(flagged_rows())
    seen, other = set(), []
    label_by_hash = {}
    for lr in layer_out:
        for row in lr["label_scored_clusters"]:
            label_by_hash[(row["layer"], row["member_hash"])] = row

    def set_recovered(key):
        """A label set counts as RECOVERED iff a label-blind cluster holds >= 2/3 of it,
        that cluster is ring-flagged, and the ordering against the known cycle holds."""
        hits = []
        for lr in layer_out:
            for row in lr["label_scored_clusters"]:
                sc = row["label_scores"].get(key)
                if not sc or not sc.get("scored") or not row["flagged"]:
                    continue
                prim = sc["from_cluster_pca"]
                hits.append(dict(layer=row["layer"], arm=row["arm"], size=row["size"],
                                 n_labels=sc["n_labels_present"], rho=prim["rho"],
                                 p_order=prim["p_order"], p_closed=row["screening"]["p_closed"],
                                 p_bg=row["screening"]["p_bg"],
                                 p_shuffled=sc["shuffled_labels_control"]["p_shuffled"],
                                 member_hash=row["member_hash"]))
        strong = [h for h in hits if h["p_order"] <= 0.05 and h["p_shuffled"] <= 0.05]
        return bool(strong), hits, strong

    def label_set_summary(key):
        """Per label set, the three separable questions, kept separate on purpose:
        (a) does label-blind clustering CO-LOCATE the set,
        (b) does the cluster that holds it pass the ring screen (discovery), and
        (c) does the label subset itself close (supervised anchor)."""
        colo, cluster_level, subset = [], [], []
        for lr in layer_out:
            for p in lr["label_placement"]:
                if p["label_set"] == key:
                    colo.append(p)
            for row in lr["label_scored_clusters"]:
                sc = row["label_scores"].get(key)
                if not sc or not sc.get("scored"):
                    continue
                prim, sec = sc["from_cluster_pca"], sc["from_label_only_pca"]
                cluster_level.append(dict(
                    layer=row["layer"], arm=row["arm"], cluster_size=row["size"],
                    cluster_flagged=row["flagged"], n_labels=sc["n_labels_present"],
                    rho=prim["rho"], p_order=prim["p_order"],
                    subset_gap_ratio=prim["gap_ratio"], subset_p_closed=prim["p_closed"],
                    p_within_cluster=sc["label_subset_p_within_cluster"],
                    p_shuffled=sc["shuffled_labels_control"]["p_shuffled"]))
                subset.append(dict(
                    layer=row["layer"], arm=row["arm"], n_labels=sc["n_labels_present"],
                    rho=sec["rho"], p_order=sec["p_order"], gap_ratio=sec["gap_ratio"],
                    max_gap_deg=sec["max_gap_deg"], p_closed=sec["p_closed"],
                    p_bg=sc["label_subset_p_bg_label_pca"],
                    top2_var=sec.get("top2_var_label_only")))
        best_colo = max(colo, key=lambda p: p["fraction_of_set_together"]) if colo else None
        best_sub = min(subset, key=lambda s: (s["p_closed"], -s["n_labels"])) if subset else None
        best_cl = min(cluster_level, key=lambda s: (s["subset_p_closed"], -s["n_labels"])) \
            if cluster_level else None
        by_layer = {}
        for p in colo:
            k2 = str(p["layer"])
            if k2 not in by_layer or p["fraction_of_set_together"] > by_layer[k2]["frac"]:
                by_layer[k2] = dict(frac=p["fraction_of_set_together"], arm=p["arm"],
                                    cluster_size=p["best_cluster_size"],
                                    n_labels=p["n_labels_in_best_cluster"],
                                    cluster_flagged=(p["best_cluster_screening"] or {}).get("flagged"),
                                    cluster_p_closed=(p["best_cluster_screening"] or {}).get("p_closed"))
        return dict(
            colocation_best=best_colo, colocation_by_layer=by_layer,
            best_cluster_level_reading=best_cl,
            best_supervised_label_subset_closure=best_sub,
            supervised_note=("best_supervised_label_subset_closure re-runs PCA on the label "
                             "token types alone. It selects them by name, so it is the "
                             "supervised anchor, not a discovery result. Read it together "
                             "with colocation_best, which IS label-blind."),
            all_cluster_level_readings=cluster_level,
            all_supervised_subset_readings=subset)

    verdict = {}
    for key in ("months", "weekdays", "colors_hue6"):
        ok, hits, strong = set_recovered(key)
        verdict[key] = dict(recovered=ok, n_flagged_clusters_containing_set=len(hits),
                            hits=hits, ordering_and_shuffle_pass=strong,
                            summary=label_set_summary(key))
    for key in ("colors_hue7_pink", "numbers_one_to_ten", "digits_0_9"):
        ok, hits, strong = set_recovered(key)
        verdict[key] = dict(recovered=ok, n_flagged_clusters_containing_set=len(hits),
                            hits=hits, summary=label_set_summary(key),
                            role=("ORDINAL CONTROL -- must not close"
                                  if key == "numbers_one_to_ten" else
                                  "declared additional set"))
    months_ok = verdict["months"]["recovered"]
    weekdays_ok = verdict["weekdays"]["recovered"]

    col_avail = label_availability["colors_hue6"]
    colors_verdict = (None if col_avail["n_in_pool"] < col_avail["full_k"]
                      else verdict["colors_hue6"]["recovered"])

    for L, arm, r in flags:
        h = (L, r["member_hash"])
        if h in seen:
            continue
        seen.add(h)
        row = label_by_hash.get(h)
        other.append(dict(layer=L, arm=arm, k=r["k"], gap_ratio=r["gap_ratio"],
                          max_gap_deg=r["max_gap_deg"], p_closed=r["p_closed"],
                          p_bg=r["p_bg"], top2_var=r["top2_var"],
                          background_rate_p_closed_lt_05=r["background_rate_p_closed_lt_05"],
                          bh_pass=r.get("bh_pass_within_layer_arm"),
                          token_strings=None if row is None else row["token_strings"],
                          label_sets_present=None if row is None else
                          {k: v.get("n_labels_present") for k, v in row["label_scores"].items()}))

    n_screened_total = sum(len(v["screened"] or []) for lr in layer_out
                           for v in lr["arms"].values())
    months_negative = [p for lr in layer_out for p in lr["label_placement"]
                       if p["label_set"] == "months"]

    out = dict(
        status="exploratory, not pre-registered; declared-exploratory run",
        item="A1 -- token-type geometry discovery (unsupervised ring recovery over token types)",
        script="scripts/22_token_type_geometry.py",
        generated=time.strftime("%Y-%m-%dT%H:%M:%S"),
        declared_at=stub["declared_at"],
        declared_rules=declared_rules(),
        config=dict(root=a.root, l0=a.l0, layers=layers,
                    tokens_identical_across_layers=same),
        pool=dict(n_types_over_floor=int(n_types), count_floor=COUNT_FLOOR,
                  n_cache_rows=int(len(tok)),
                  n_cache_rows_covered=int(cnt[pool_ids].sum()),
                  fraction_rows_covered=float(cnt[pool_ids].sum() / len(tok)),
                  n_distinct_types_in_cache=int((cnt > 0).sum()),
                  n_types_extended_pool=int(len(ext_ids)), count_floor_extended=COUNT_FLOOR_EXT),
        label_availability=label_availability,
        layers=layer_out,
        verdict=dict(
            months_recovered=months_ok,
            weekdays_recovered=weekdays_ok,
            colors_recovered=colors_verdict,
            colors_reason=(None if colors_verdict is not None else
                           "UNTESTABLE under the frozen pool rule: "
                           f"{col_avail['missing_from_pool']} have no single-token variant "
                           f"with >= {COUNT_FLOOR} cache occurrences, so the 6-term hue ring "
                           "cannot be assembled from the discovery pool at all. See "
                           "colour_supervised_sensitivity for the floor-30 supervised arm."),
            other_rings=other,
            n_flagged_total=len(other), n_screened_total=int(n_screened_total),
            expected_false_flags_at_alpha_05=round(0.05 * n_screened_total, 1),
            detail=verdict,
            months_negative_space=months_negative),
        caveats=[
            "DECLARED EXPLORATORY. Not pre-registered. Nothing here can change a gate verdict.",
            "The representation is a CONTEXT-AVERAGED token-type mean over every corpus "
            "occurrence. It is closer to a static embedding than to a computed feature, so a "
            "ring here is a claim about the model's average representation of those token "
            "types, not about a structure the model manipulates in context.",
            "Layers 0-4 are embedding-adjacent. A hit at those layers validates the pipeline "
            "and locates geometry near the input; it is not a depth claim.",
            "ORDERING IS NEVER EVIDENCE HERE. This project measured the ordering tier "
            "accepting random subsets at ~0.16 with an ordinal control out-ordering months at "
            "every layer. Closure against the matched same-size background is the tier that "
            "discriminates, and the per-size background acceptance rate is reported next to "
            "every flag so the reader can see the false-acceptance floor at that k.",
            "The flag rule is uncorrected across the whole scan by design (it is the declared "
            "rule); bh_pass_within_layer_arm and expected_false_flags_at_alpha_05 are reported "
            "so a flag count can be read against its own multiplicity.",
            "The colour arm is not a discovery result: at the frozen count floor two of the six "
            "hue terms do not exist in the pool, so colours are UNTESTABLE by discovery here, "
            "and the floor-30 arm that does score them is supervised (the six token types are "
            "chosen by name) and matched only against a floor-30 random background.",
            "digits_0_9 carries the confound declared before the run: at token-type level the "
            "token identity IS the value, so any digit structure is token-type geometry and "
            "must not be read as a base-10 helix or as evidence about depth.",
            "numbers_one_to_ten is an ordinal control and is truncated at ten by the count "
            "floor; it must fail closure, and a run in which it closes is invalid rather than "
            "interesting.",
            "The refinement arm splits oversized clusters label-blind, but it is an extra "
            "selection surface: more candidate clusters means more chances to flag one, which "
            "is why the matched background and the expected-false-flag count are reported per "
            "arm rather than pooled.",
            "The cosine grids and the edge budget were calibrated on a label-blind layer-0 "
            "density probe before the freeze; this is disclosed rather than hidden, and no "
            "label or cluster content was inspected in that probe.",
        ],
        runtime_seconds=round(time.time() - t_start, 1))
    Path(a.out).write_text(json.dumps(out, indent=2))
    print(f"\nmonths_recovered={months_ok} weekdays_recovered={weekdays_ok} "
          f"colors_recovered={colors_verdict}")
    print(f"{len(other)} distinct flagged rings out of {n_screened_total} screened clusters")
    print(f"wrote {a.out}  ({out['runtime_seconds']}s)")


if __name__ == "__main__":
    main()
