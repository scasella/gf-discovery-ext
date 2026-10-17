"""42 -- two-channel re-adjudication of the 51 closure flags (Phase 4, prereg 2.1).

Pre-registration: docs/phase4_prereg_2026-08-22.md 2.1, rules R7-R12 of 1.
Erratum carried: docs/phase3_addendum_I_erratum_2026-08-22.md (lag+2 is a SPECIFICITY
control, not a second positive channel).

The question. Phase 3 established that the ORDER tier cannot adjudicate a closure passer
with no name, and that split-half order reproducibility is not a replacement (its background
is 0.73-0.99). This script tests the proposed replacement: the cyclic order recovered from a
set's TYPE-MEAN plane against the cyclic order recovered independently from the same set's
SUCCESSOR-ACTION plane. Two channels, no labels, and a null that puts random same-size sets
through both channels.

Channel 1  per-token-type mean SAE code -> W_dec -> 1152-d (scripts/22 stage 1), R10 drop.
Channel 2  identity-removed lag+1 successor-delta means (scripts/41b construction), R10 drop.
Statistic  max over the 2k cyclic shifts/reflections of Spearman between the two recovered
           orders (fits.cyclic_order_score convention, order-vs-order).
Null       6,000 cluster-like k-sets per (layer, k); conditional background = those with
           gap_ratio < 2.5; threshold = 99th percentile of the conditional set.

Nothing in this file may relax the frozen PASS rule. Where 2.1 leaves a convention
unstated, the choice is pinned in `declared_rules()` and written to disk BEFORE any cache
or npz read.

MEASURED BEFORE THE RUN AND DECLARED (preflight, logged in results.preflight): the frozen
PASS rule requires closure at every leave-one-out against a k-INDEPENDENT gap_ratio < 2.5.
A perfect k-gon with one member removed has max gap 2*(360/k), i.e. gap_ratio 1.85 at k=13,
so the rule leaves ~35% headroom; the L6 13-set fails 2 of 13 LOO subsets and the L7 14-set
3 of 14. The frozen verdict is computed and reported anyway -- it is the verdict -- and the
co-reported PASS_noLOO / PASS_LOO_rho_only columns exist so the tier itself can be read
separately from the conjunct that fails its own positive control. Non-verdict-bearing.
"""
import argparse, gc, hashlib, json, os, sys, time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HOME", str(REPO / "cache/hf"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
sys.path.insert(0, str(REPO / "src"))
from gfdisc.fits import pca_2d, fit_circle, circularity, degenerate
from gfdisc.saes import load_sae, get_decoder
from gfdisc.nulls import frequency_matched_resample
from gfdisc.tokens import DAYS, MONTHS

# =====================================================================================
# frozen constants -- scripts/22 (channel 1, clustering) and scripts/41b (channel 2)
# =====================================================================================
COUNT_FLOOR = 100
COUNT_FLOOR_EXT = 30
D_MODEL = 1152
D_SAE = 16384
SEQ_LEN = 255
CHUNK = 200_000
LAGS = (1, 2)
MIN_PAIRS_PER_HALF = 20
SCREEN_MIN, SCREEN_MAX = 5, 60
GRID_CENTRED = [round(0.10 + 0.05 * i, 2) for i in range(17)]
GRID_RAW = [round(0.50 + 0.05 * i, 2) for i in range(10)]
EDGE_BUDGET = 2_000_000
LEIDEN_RESOLUTION = 1.0
LEIDEN_SEED = 0
REFINE_RESOLUTIONS = [2, 4, 8, 16, 32, 64]
HDBSCAN_MIN_CLUSTER_SIZE = 5
HDBSCAN_MIN_SAMPLES = 5
OUTLIER_SHARE_THRESHOLD = 0.25

# --- prereg 2.1 / R7-R12 ---
GAP_RATIO_CLOSURE = 2.5              # frozen closure, k-independent
K_MIN_ADJUDICABLE = 9                # R8
N_CLUSTERLIKE = 6_000                # cluster-like draws per (layer, k)
N_RANDOM = 2_000
N_FREQMATCHED = 2_000
THRESHOLD_PERCENTILE = 99.0          # 99th, NOT 99.9th (verifier correction)
N_FREQ_PSEUDO = 200
N_CLASS_PERM = 200
CLASS_PERM_MAX_PASSES = 2            # <= 1% of 200
FREQ_PSEUDO_MAX_PASSES = 0
FREQ_PSEUDO_NEAREST = 5
FREQ_MATCH_TOL = 0.25
PLANTED_RHOS = (0.05, 0.10, 0.15, 0.20, 0.30, 0.50)
N_PLANTED = 200
PLANTED_K = 12
POWER_TARGET = 0.80
# R10 -- the drop rule's frozen outcome, stated in the prereg with its coordinates.
R10_DROP = {L: (1125 if L <= 2 else 1038) for L in range(0, 11)}
ORDINAL_WORDS = (" one", " two", " three", " four", " five", " six", " seven",
                 " eight", " nine", " ten")

# --- frozen exposure (prereg 0): member_hash is md5 over sorted POOL POSITIONS ---
MEMBERS_13_TOKEN_IDS = [3353, 5142, 5261, 5365, 5505, 5656, 5729, 5857,
                        6027, 6198, 6243, 6384, 7250]
MEMBERS_14_TOKEN_IDS = sorted(MEMBERS_13_TOKEN_IDS + [30651])
HASH_13 = "7d1d75a303d6"
HASH_14 = "6bb0e54cd127"
ANCHOR_GAP_RATIO = {6: 1.6588, 7: 1.6202}     # stored flag values, channel 1 PRE-drop
ANCHOR_TOL = 1e-3

FLAGS_PATH_DEFAULT = ("/private/tmp/claude-501/-Users-scasella-Downloads-gf-discovery/"
                      "cc59cc80-4350-4784-b57b-790adf317b63/scratchpad/prefreeze/"
                      "item4_dedup.json")
LAYERS_ALL = list(range(0, 11))
# layers whose channel-2 deltas are computed here by CPU cache stream. 2.1 names
# L1, L2, L3, L4, L10; L0 is a DECLARED SUPPLEMENT (see declared_rules.deviations) so that
# every k >= 9 flag is adjudicable rather than one being unadjudicable for want of a layer.
LAYERS_STREAM = [0, 1, 2, 3, 4, 10]
LAYERS_DELTA_FROM_NPZ = [6, 7]
ROBUST_DELTAS = "results/phase3_lag_action_robust_deltas.npz"
FULLN_RAW = "results/phase3_token_type_mean_resid_fulln.npz"


def declared_rules(args_view):
    return dict(
        prereg="docs/phase4_prereg_2026-08-22.md section 2.1; rules R7-R12 of section 1",
        erratum="docs/phase3_addendum_I_erratum_2026-08-22.md (lag+2 = specificity control)",
        channel_1=("per-token-type mean SAE code over ALL cache rows of that type, projected "
                   "mean_code @ W_dec (no b_dec) -> 1152-d, exactly scripts/22 stage 1, at the "
                   "flag's own layer. Identical for every clustering arm: the arms differ in "
                   "how the cosine GRAPH is built, not in the point coordinates. Raw residual "
                   "R_L from results/phase3_token_type_mean_resid_fulln.npz is additionally "
                   "reported for L4-L9 as a SECONDARY, non-verdict coordinate system."),
        channel_2=("identity-removed lag+1 successor-delta type means, scripts/41b "
                   "construction verbatim: delta_t(+1) = mean over admissible pairs (i, i+1) "
                   "with tok(i) of type t of [code(row i+1) - TM[type(row i+1)]] @ W_dec, "
                   "parity halves pooled by pair count, drop rule applied. L6/L7 are READ "
                   "from results/phase3_lag_action_robust_deltas.npz; L0-L4 and L10 are "
                   "computed here by a CPU cache stream of the same construction."),
        statistic=("two_channel_rho = max over the 2k cyclic shifts/reflections of the "
                   "Spearman correlation between the two recovered angular ORDERS "
                   "(fits.cyclic_order_score convention, order-vs-order; no labels enter). "
                   "Angles are taken in each channel's own top-2 PCA plane of the members' "
                   "centred coordinates. Ties broken by a seeded random jitter, seed 0."),
        R10_drop=("prereg R10 states the rule AND its frozen outcome: coordinate 1038 at "
                  "L3-L10, coordinate 1125 at L0-L2, dropped from EVERY coordinate system "
                  "before any plane fit. Applied as the frozen constant to channel 1 (which "
                  "has no split-half in this pipeline) and, for the layers streamed here, "
                  "RECOMPUTED by scripts/41b outlier_drop_rule on channel 2 and asserted "
                  "equal to the frozen constant; disagreement is reported, never silently "
                  "accepted. For L6/L7 the stored npz is already post-drop."),
        closure="frozen: gap_ratio < 2.5 (fits.circularity on the Kasa fit), k-independent",
        null=("per (layer, k): 6,000 cluster-like k-sets = random seed type + its k-1 nearest "
              "by CENTRED cosine in channel-1 coordinates (centred cosine regardless of the "
              "flag's arm). CONDITIONAL background = the draws with gap_ratio < 2.5. "
              "Threshold = 99th percentile of two_channel_rho over that conditional set, "
              "exact order statistic (np.percentile method='higher'); the observed rho must "
              "be STRICTLY ABOVE it. n_passers is recorded per cell. 2,000 random and 2,000 "
              "frequency-matched k-sets are additionally reported, never used as the gate."),
        PASS=("gap_ratio < 2.5 AND two_channel_rho > threshold(layer, k) AND both properties "
              "hold at every leave-one-out. LOO convention (pinned here, 2.1 does not state "
              "it): a LOO subset has size k-1, so its closure uses the same k-independent "
              "2.5 and its rho uses threshold(layer, k-1) from a null computed at k-1. R8 "
              "gates CANDIDATES at k >= 9, not the k-1 LOO subsets of a candidate."),
        R8=("a two-channel verdict exists only for k >= 9. k < 9 flags are reported under "
            "closure and closure-LOO only and are labelled NOT ADJUDICABLE."),
        lag2_specificity=("per the erratum, lag+2 is a specificity control: a passer whose "
                          "lag+2 rho also clears the 99th percentile of the lag+2 conditional "
                          "null is flagged for R11 review. Computed at adjudicated cells only."),
        controls=dict(
            positive=("the L6 k=13 (hash 7d1d75a303d6) and L7 k=14 (hash 6bb0e54cd127) month "
                      "flags must PASS; strict-12 and the 13/14-set are both reported"),
            negative_ordinal=("' one'..' ten' as single in-pool token ids, at L6 and L7, "
                              "through the identical path; must not pass"),
            negative_freq_pseudo=(f"{N_FREQ_PSEUDO} frequency-code pseudo-populations built by "
                                  "scripts/41b freq_pseudo_null's rule (each member replaced by "
                                  f"one of its {FREQ_PSEUDO_NEAREST} nearest-log-count "
                                  "non-calendar pool types), anchored on the L6 13-set and the "
                                  "L7 14-set, through the full path; must give 0 passes"),
            negative_class_perm=(f"{N_CLASS_PERM} class-block-permuted draws: the CHANNEL-2 rows "
                                 "are permuted among types WITHIN each lexical block "
                                 "(leading_space x capitalized, scripts/41b resolve_blocks), "
                                 "which destroys the type-specific channel1<->channel2 "
                                 "correspondence while preserving class-level coupling. The "
                                 "anchor member set is re-adjudicated. This is the direct "
                                 f"measurement of R11 tier leakage; must give <= "
                                 f"{CLASS_PERM_MAX_PASSES} passes")),
        planted=("k=12 ring planted by REPLACEMENT IN PLANE (repo convention, scripts/39 and "
                 "41b): 12 random pool types are the carriers; in their own top-2 channel-1 "
                 "plane their coordinates are replaced by centre + r*(cos t, sin t) plus "
                 "Gaussian in-plane jitter of sd rho*r, out-of-plane components kept. r is "
                 "the L6 13-set's own fitted channel-1 radius (pinned at run time in "
                 "results.planted.radius_channel1). Channel 2 carries the SAME cyclic order "
                 "with an INDEPENDENT jitter draw at the L6 13-set's fitted channel-2 radius. "
                 f"{N_PLANTED} plants per rho in {list(PLANTED_RHOS)}; recall of closure, of "
                 "two-channel, and joint; rho80 by linear interpolation on the recall curve."),
        blind_population=("prereg 2.1 declares 10 non-calendar flags with k >= 9. The flags "
                          "file holds 12 flags at k >= 9, of which 3 carry calendar labels "
                          "(L6 k=13 and L7 k=14 = the month positive controls; L0 k=10 = "
                          "'raw_refined', label_sets {months: 4}). The strictly non-calendar "
                          "k >= 9 population is therefore 9, not 10. ALL 12 are adjudicated; "
                          "the L0 k=10 flag is reported as calendar-bearing and OUTSIDE the "
                          "blind population. Expected false passers are reported both ways: "
                          "9 x 0.01 = 0.09 and 10 x 0.01 = 0.10."),
        member_reconstruction=("member_hash is md5(','.join(sorted POOL POSITIONS))[:12], "
                              "verified against HASH_13. Members are recovered by re-running "
                              "scripts/22 stage 1 + stage 2 (both Leiden arms, both refined "
                              "arms, HDBSCAN) per layer and matching hashes. Acceptance: the "
                              "reconstructed cluster's PRE-drop gap_ratio must reproduce the "
                              "stored flag value to 1e-6. Fallback if a hash is absent: same "
                              "(layer, arm), same k, best Jaccard over the flag's stored "
                              "token_strings, same gap_ratio reproduction requirement. A flag "
                              "that cannot be reconstructed is reported NOT RECONSTRUCTED and "
                              "is never silently skipped."),
        co_reported_non_verdict=("PASS_noLOO and PASS_LOO_rho_only are reported beside the "
                                 "frozen PASS so the tier can be read separately from the "
                                 "LOO conjunct. They decide nothing."),
        deviations=[
            "L0 channel-2 deltas are computed although 2.1 names only L1, L2, L3, L4, L10; "
            "this is an ADDITION that makes every k >= 9 flag adjudicable and changes no "
            "threshold, no statistic and no gate. Declared here, before any read.",
            "2.1's blind population count (10 non-calendar k >= 9) does not match the flags "
            "file (9 strictly non-calendar; L0 k=10 carries 4 month labels). Both readings "
            "are reported; no flag is dropped.",
            "The LOO threshold convention and the percentile convention are pinned here "
            "because 2.1 does not state them.",
        ],
        seeds=dict(clusterlike="7_000_000 + 1000*layer + k",
                   random="7_100_000 + 1000*layer + k",
                   freq_matched="7_200_000 + 1000*layer + k",
                   freq_pseudo="7_300_000 + layer",
                   class_perm="7_400_000 + layer",
                   planted="7_500_000 + int(1000*rho)",
                   tie_break=0),
        config=args_view,
    )


# =====================================================================================
# stage 1 -- pool and channel-1 coordinates (scripts/22 stage 1, verbatim behaviour)
# =====================================================================================
def build_pool(root, layer0=0):
    tok = np.fromfile(Path(root) / f"layer_{layer0}" / "tokens.i32", dtype=np.int32)
    cnt = np.bincount(tok)
    ext = np.flatnonzero(cnt >= COUNT_FLOOR_EXT)
    main = np.flatnonzero(cnt >= COUNT_FLOOR)
    return tok, cnt, main, ext


def type_means(root, layer, tok, cnt, ext_ids, chunk=CHUNK):
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
        assert np.isfinite(vals).all(), f"non-finite cache values at rows [{lo},{hi}) L{layer}"
        ct = indptr[lo + 1:hi + 1] - indptr[lo:hi]
        rid = np.repeat(tidx[lo:hi], ct)
        m = rid >= 0
        np.add.at(acc, (rid[m], cols[m]), vals[m])
        n_rows_used += int((tidx[lo:hi] >= 0).sum())
    acc /= cnt[ext_ids][:, None]
    return acc, d_sae, n_rows_used, meta


def project(A, W, chunk=2000):
    out = np.empty((A.shape[0], W.shape[1]), dtype=np.float32)
    for lo in range(0, A.shape[0], chunk):
        hi = min(lo + chunk, A.shape[0])
        out[lo:hi] = A[lo:hi].astype(np.float32) @ W
    return out


# =====================================================================================
# stage 2 -- scripts/22 label-blind partitions (verbatim), for member reconstruction
# =====================================================================================
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
        cl = leiden(len(S), iu[0][m], iu[1][m], sv[m])
        sizes = sorted((len(c) for c in cl), reverse=True)
        n5 = int(sum(1 for s in sizes if s >= 5))
        row.update(n_clusters_total=len(cl), n_clusters_ge5=n5,
                   max_size=int(sizes[0]) if sizes else 0)
        row["valid"] = bool(10 <= n5 <= 300 and row["max_size"] <= 0.10 * n_types)
        rows.append(row)
        partitions[tau] = cl
    valid = [r for r in rows if r.get("valid")]
    if valid:
        tau = valid[0]["tau"]
    else:
        fb = [r for r in rows if "skipped" not in r and r.get("max_size", 1e9) <= 0.10 * n_types]
        if fb:
            tau = fb[0]["tau"]
        else:
            ok = [r for r in rows if "skipped" not in r]
            tau = ok[-1]["tau"] if ok else None
    return tau, (partitions.get(tau) or [])


def refine(clusters, S, tau):
    out = []
    for c in clusters:
        if len(c) <= SCREEN_MAX:
            out.append(c); continue
        idx = np.asarray(c)
        sub = S[np.ix_(idx, idx)]
        iu = np.triu_indices(len(idx), 1)
        sv = sub[iu]
        m = sv >= tau
        chosen = None
        for res in REFINE_RESOLUTIONS:
            parts = leiden(len(idx), iu[0][m], iu[1][m], sv[m], resolution=res)
            chosen = parts
            if max((len(p) for p in parts), default=0) <= SCREEN_MAX:
                break
        subs = [sorted(int(idx[i]) for i in p) for p in (chosen or [])]
        out.extend([p for p in subs if len(p) <= SCREEN_MAX])
    return out


def run_hdbscan(R):
    import hdbscan as _h
    X = R - R.mean(0, keepdims=True)
    X = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
    lab = _h.HDBSCAN(min_cluster_size=HDBSCAN_MIN_CLUSTER_SIZE,
                     min_samples=HDBSCAN_MIN_SAMPLES, metric="euclidean",
                     cluster_selection_method="eom").fit_predict(X.astype(np.float64))
    return [sorted(np.flatnonzero(lab == g).tolist()) for g in sorted(set(lab.tolist()) - {-1})]


def member_hash(members):
    return hashlib.md5(",".join(str(int(x)) for x in sorted(members)).encode()).hexdigest()[:12]


def all_partitions(R, n_types):
    """Every scripts/22 arm's clusters for one layer -> {arm: [clusters]}."""
    arms = {}
    S_centred = cosine_matrix(R, True)
    S_raw = cosine_matrix(R, False)
    for arm, S, grid in (("centred_leiden", S_centred, GRID_CENTRED),
                         ("raw_leiden", S_raw, GRID_RAW)):
        tau, clusters = sweep_and_select(S, grid, n_types)
        arms[arm] = clusters
        arms[arm.replace("_leiden", "_refined")] = refine(clusters, S, tau) if tau is not None else []
    del S_centred, S_raw
    gc.collect()
    arms["hdbscan"] = run_hdbscan(R)
    return arms


# =====================================================================================
# stage 3 -- channel 2 (scripts/41b construction, CPU)
# =====================================================================================
def pair_index(tok, n_used, lut):
    own = lut[tok[:n_used]].astype(np.int32)
    j = np.arange(n_used, dtype=np.int64)
    parity = ((j // SEQ_LEN) % 2).astype(np.int8)
    pos_in_seq = (j % SEQ_LEN)
    per_lag = {}
    for d in LAGS:
        prev = np.full(n_used, -1, dtype=np.int32)
        if n_used > d:
            prev[d:] = own[:n_used - d]
        pred_ok = (pos_in_seq >= d) & (prev >= 0)
        succ_ok = pred_ok & (own >= 0)
        per_lag[d] = dict(prev=prev, pred_ok=pred_ok, succ_ok=succ_ok)
    return own, parity, per_lag


def stream_deltas(root, layer, own, pred_par, n_used, n_types, d_sae, chunk=CHUNK):
    d = Path(root) / f"layer_{layer}"
    indptr = np.load(d / "indptr.npy")
    indices = np.memmap(d / "indices.u16", dtype=np.uint16, mode="r")
    values = np.memmap(d / "values.f16", dtype=np.float16, mode="r")
    TM_acc = np.zeros((n_types, d_sae), dtype=np.float64)
    S = {lag: np.zeros((2 * n_types, d_sae), dtype=np.float64) for lag in LAGS}
    seen = np.zeros(n_types, dtype=np.int64)
    t0 = time.time()
    for lo in range(0, n_used, chunk):
        hi = min(lo + chunk, n_used)
        a, b = int(indptr[lo]), int(indptr[hi])
        cols = np.asarray(indices[a:b], dtype=np.int32)
        vals = np.asarray(values[a:b], dtype=np.float32)
        assert np.isfinite(vals).all(), f"non-finite cache values rows [{lo},{hi}) L{layer}"
        ct = (indptr[lo + 1:hi + 1] - indptr[lo:hi]).astype(np.int64)
        rid = np.repeat(own[lo:hi], ct)
        m = rid >= 0
        np.add.at(TM_acc, (rid[m], cols[m]), vals[m])
        seen += np.bincount(own[lo:hi][own[lo:hi] >= 0], minlength=n_types)
        del rid, m
        for lag in LAGS:
            rid = np.repeat(pred_par[lag][lo:hi], ct)
            m = rid >= 0
            np.add.at(S[lag], (rid[m], cols[m]), vals[m])
            del rid, m
        del cols, vals, ct
        gc.collect()
        if (lo // chunk) % 5 == 0:
            print(f"    stream L{layer} {hi}/{n_used} rows {time.time()-t0:.0f}s", flush=True)
    return TM_acc, S, seen


def outlier_drop_rule(A_even, A_odd, included, threshold=OUTLIER_SHARE_THRESHOLD):
    d = A_even.shape[1]
    inc = np.asarray(included, dtype=bool)
    if not inc.any():
        sd = np.zeros(d, dtype=np.float64)
    else:
        resid = (A_even[inc] - A_odd[inc]) / np.sqrt(2.0)
        sd = np.sqrt((resid.astype(np.float64) ** 2).mean(0))
    tot = float((sd ** 2).sum())
    shares = (sd ** 2 / tot) if tot > 0 else np.zeros(d, dtype=np.float64)
    dropped = np.flatnonzero(shares > threshold)
    keep = np.ones(d, dtype=bool)
    keep[dropped] = False
    return keep, sd, [int(x) for x in dropped], [float(shares[i]) for i in dropped]


# =====================================================================================
# stage 4 -- the two statistics
# =====================================================================================
def gap_ratio_of(X):
    """fits.circularity gap_ratio in the point set's own top-2 PCA plane. None if degenerate."""
    X = np.asarray(X, dtype=np.float64)
    if len(X) < 3 or degenerate(X):
        return None
    xy, _, _ = pca_2d(X)
    cx, cy, r, _ = fit_circle(xy)
    return float(circularity(xy, cx, cy, r)["gap_ratio"])


def plane_angles(X, seed=0):
    """Angles about the fitted circle centre in the set's own top-2 plane."""
    X = np.asarray(X, dtype=np.float64)
    xy, _, _ = pca_2d(X)
    cx, cy, r, _ = fit_circle(xy)
    return np.arctan2(xy[:, 1] - cy, xy[:, 0] - cx), float(r)


def _rank_from_angles(ang, seed=0):
    jitter = np.random.default_rng(seed).permutation(len(ang))
    order = np.lexsort((jitter, np.asarray(ang, dtype=float)))
    r = np.empty(len(order), dtype=np.float64)
    r[order] = np.arange(len(order), dtype=np.float64)
    return r


def two_channel_rho(ang_a, ang_b, seed=0):
    """max over the 2k cyclic shifts/reflections of Spearman between two recovered orders."""
    k = len(ang_a)
    ra = _rank_from_angles(ang_a, seed)
    rb = _rank_from_angles(ang_b, seed)
    best = -2.0
    for flip in (1, -1):
        for shift in range(k):
            ref = _rank_from_angles((flip * rb + shift) % k, seed=0)
            d = ra - ref
            best = max(best, 1.0 - 6.0 * float(d @ d) / (k * (k * k - 1)))
    return float(best)


def cell_stats(X1, X2, seed=0):
    """(gap_ratio in channel 1, two-channel rho) for one member set. rho None if no channel 2."""
    g = gap_ratio_of(X1)
    if X2 is None or g is None:
        return g, None
    if len(X2) < 3 or degenerate(np.asarray(X2, dtype=np.float64)):
        return g, None
    a1, _ = plane_angles(X1, seed)
    a2, _ = plane_angles(X2, seed)
    return g, two_channel_rho(a1, a2, seed)


# =====================================================================================
# stage 5 -- nulls
# =====================================================================================
class NullCache:
    """Cluster-like / random / frequency-matched nulls per (layer, k), computed once."""

    def __init__(self, Xn, C1, C2, counts_pool, n_clusterlike=N_CLUSTERLIKE):
        self.Xn = Xn            # L2-normalised centred channel-1 coords, [n, d]
        self.C1 = C1
        self.C2 = C2            # may be None
        self.counts = counts_pool
        self.n_clusterlike = int(n_clusterlike)
        self.cache = {}

    def _draw_clusterlike(self, k, rng):
        s = int(rng.integers(len(self.Xn)))
        sim = self.Xn @ self.Xn[s]
        sim[s] = 2.0
        return np.argsort(sim)[::-1][:k]

    def build(self, layer, k, n_draws=None, kind="clusterlike"):
        n_draws = N_CLUSTERLIKE if n_draws is None else n_draws
        key = (layer, int(k), kind)
        if key in self.cache:
            return self.cache[key]
        seed = {"clusterlike": 7_000_000, "random": 7_100_000,
                "freq": 7_200_000}[kind] + 1000 * layer + int(k)
        rng = np.random.default_rng(seed)
        n = {"clusterlike": self.n_clusterlike, "random": N_RANDOM,
             "freq": N_FREQMATCHED}[kind]
        gaps, rhos_cond, rhos_all, n_scorable = [], [], [], 0
        for _ in range(n):
            if kind == "clusterlike":
                idx = self._draw_clusterlike(k, rng)
            elif kind == "random":
                idx = rng.choice(len(self.C1), size=k, replace=False)
            else:
                idx = frequency_matched_resample(
                    rng.choice(len(self.C1), size=k, replace=False),
                    self.counts, rng, tol=FREQ_MATCH_TOL)
                if len(idx) != k:
                    continue
            g, r = cell_stats(self.C1[idx], None if self.C2 is None else self.C2[idx])
            if g is None:
                continue
            n_scorable += 1
            gaps.append(g)
            if r is not None:
                rhos_all.append(r)
                if g < GAP_RATIO_CLOSURE:
                    rhos_cond.append(r)
        gaps = np.asarray(gaps, dtype=np.float64)
        rc = np.asarray(rhos_cond, dtype=np.float64)
        out = dict(
            layer=int(layer), k=int(k), kind=kind, n_draws=int(n), n_scorable=int(n_scorable),
            n_closure_pass=int((gaps < GAP_RATIO_CLOSURE).sum()) if len(gaps) else 0,
            closure_rate=float((gaps < GAP_RATIO_CLOSURE).mean()) if len(gaps) else None,
            gap_ratio_median=float(np.median(gaps)) if len(gaps) else None,
            n_conditional=int(len(rc)),
            threshold_p99_conditional=(float(np.percentile(rc, THRESHOLD_PERCENTILE,
                                                           method="higher"))
                                       if len(rc) else None),
            rho_conditional_median=float(np.median(rc)) if len(rc) else None,
            rho_all_p99=(float(np.percentile(np.asarray(rhos_all), THRESHOLD_PERCENTILE,
                                             method="higher")) if rhos_all else None),
            threshold_saturated=bool(len(rc) and float(np.percentile(
                rc, THRESHOLD_PERCENTILE, method="higher")) >= 1.0),
            saturation_note=("rho is bounded above by 1.0, so a cell whose conditional p99 "
                             "IS 1.0 cannot be passed by any set: 'strictly above the 99th "
                             "percentile' is unsatisfiable there. Diagnostic only."),
            seed=int(seed))
        self.cache[key] = out
        return out


# =====================================================================================
# stage 6 -- adjudication
# =====================================================================================
def adjudicate(members, C1, C2, nulls, layer, seed=0, want_loo=True, thr_override=None):
    """Frozen PASS of prereg 2.1, plus the co-reported non-verdict variants.

    `thr_override` = (threshold_k, threshold_k_minus_1) supplies the tier's OWN thresholds
    to a control run whose channel 2 has been altered (the class-block permutation). The
    FDR question there is "does the altered channel produce a passer at the tier's
    threshold", so the threshold must not be recomputed from the altered channel.
    """
    members = np.asarray(sorted(int(x) for x in members))
    k = len(members)
    g, rho = cell_stats(C1[members], None if C2 is None else C2[members], seed)
    adjudicable = bool(k >= K_MIN_ADJUDICABLE and rho is not None)
    if thr_override is not None:
        thr = thr_override[0] if adjudicable else None
    else:
        thr = nulls.build(layer, k)["threshold_p99_conditional"] if adjudicable else None
    closure = bool(g is not None and g < GAP_RATIO_CLOSURE)
    rho_pass = bool(adjudicable and thr is not None and rho > thr)

    loo_gaps, loo_rhos, loo_ok_closure, loo_ok_rho = [], [], True, True
    thr_loo = None
    if want_loo and k >= 4:
        if adjudicable:
            thr_loo = (thr_override[1] if thr_override is not None
                       else nulls.build(layer, k - 1)["threshold_p99_conditional"])
        for i in range(k):
            sub = np.delete(members, i)
            gi, ri = cell_stats(C1[sub], None if C2 is None else C2[sub], seed)
            loo_gaps.append(gi)
            loo_rhos.append(ri)
            if not (gi is not None and gi < GAP_RATIO_CLOSURE):
                loo_ok_closure = False
            if adjudicable and not (ri is not None and thr_loo is not None and ri > thr_loo):
                loo_ok_rho = False
    return dict(
        k=int(k), gap_ratio=g, two_channel_rho=rho, threshold_p99=thr,
        threshold_p99_loo=thr_loo, adjudicable=adjudicable,
        closure_pass=closure, rho_pass=rho_pass,
        loo_closure_pass=bool(loo_ok_closure), loo_rho_pass=bool(loo_ok_rho),
        loo_gap_ratios=[None if x is None else round(float(x), 4) for x in loo_gaps],
        loo_rhos=[None if x is None else round(float(x), 4) for x in loo_rhos],
        n_loo_closure_fail=int(sum(1 for x in loo_gaps
                                   if x is None or x >= GAP_RATIO_CLOSURE)),
        threshold_saturated=bool(thr is not None and thr >= 1.0),
        threshold_loo_saturated=bool(thr_loo is not None and thr_loo >= 1.0),
        verdict=("NOT ADJUDICABLE" if not adjudicable else
                 ("PASS" if (closure and rho_pass and loo_ok_closure and loo_ok_rho)
                  else "FAIL")),
        PASS_noLOO=bool(adjudicable and closure and rho_pass),
        PASS_LOO_rho_only=bool(adjudicable and closure and rho_pass and loo_ok_rho))


# =====================================================================================
def class_features(s):
    alphas = [c for c in s if c.isalpha()]
    all_caps = bool(alphas) and all(c.isupper() for c in alphas)
    capitalized = bool(alphas) and alphas[0].isupper() and not all_caps
    return [s.startswith(" "), capitalized, all_caps, any(c.isdigit() for c in s),
            not any(c.isalnum() for c in s), any(ord(c) > 127 for c in s)]


def resolve_blocks(strings):
    keys = {}
    for i, s in enumerate(strings):
        f = class_features(s)
        keys.setdefault(f"lead={bool(f[0])}|cap={bool(f[1])}", []).append(i)
    return [np.asarray(v) for v in keys.values()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="cache/sparse_big")
    ap.add_argument("--l0", default="big")
    ap.add_argument("--flags", default=FLAGS_PATH_DEFAULT)
    ap.add_argument("--out", default="results/phase4_two_channel.json")
    ap.add_argument("--layers", default=",".join(str(x) for x in LAYERS_ALL))
    ap.add_argument("--stream-layers", default=",".join(str(x) for x in LAYERS_STREAM))
    ap.add_argument("--n-clusterlike", type=int, default=N_CLUSTERLIKE)
    ap.add_argument("--n-planted", type=int, default=N_PLANTED)
    ap.add_argument("--save-deltas", type=int, default=1)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    t_start = time.time()
    root = Path(a.root)
    layers_all = [int(x) for x in a.layers.split(",") if x.strip() != ""]
    layers_stream = [int(x) for x in a.stream_layers.split(",") if x.strip() != ""]
    n_clusterlike = int(a.n_clusterlike)
    n_planted = int(a.n_planted)

    # ---- flags list (a scratchpad json, not cache/npz) ----
    flags = json.loads(Path(a.flags).read_text())
    flags_md5 = hashlib.md5(Path(a.flags).read_bytes()).hexdigest()
    k_big = [f for f in flags if f["k"] >= K_MIN_ADJUDICABLE]

    # ---- DECLARATION STUB, written before any cache/npz read ----
    declared_at = time.strftime("%Y-%m-%dT%H:%M:%S")
    args_view = dict(root=a.root, l0=a.l0, flags=a.flags, out=a.out, smoke=bool(a.smoke),
                     layers=layers_all, layers_streamed=layers_stream,
                     layers_delta_from_npz=LAYERS_DELTA_FROM_NPZ,
                     n_clusterlike=n_clusterlike, n_planted=n_planted)
    stub = dict(status="DECLARATION STUB -- rules frozen, no results yet",
                script="scripts/42_two_channel_readjudicate.py",
                declared_at=declared_at,
                flags_file=a.flags, flags_md5=flags_md5, n_flags=len(flags),
                blind_population=dict(
                    n_flags_k_ge_9=len(k_big),
                    hashes=[dict(layer=f["layer"], arm=f["arm"], k=f["k"],
                                 member_hash=f["member_hash"], calendar=bool(f["calendar"]))
                            for f in k_big],
                    n_non_calendar=int(sum(1 for f in k_big if not f["calendar"])),
                    n_k_lt_9_not_adjudicable=int(len(flags) - len(k_big))),
                thresholds=dict(gap_ratio_closure=GAP_RATIO_CLOSURE,
                                percentile=THRESHOLD_PERCENTILE,
                                k_min_adjudicable=K_MIN_ADJUDICABLE,
                                n_clusterlike=N_CLUSTERLIKE,
                                freq_pseudo_max_passes=FREQ_PSEUDO_MAX_PASSES,
                                class_perm_max_passes=CLASS_PERM_MAX_PASSES),
                declared_rules=declared_rules(args_view))
    Path(a.out).write_text(json.dumps(stub, indent=2))
    print("=== DECLARATION STUB WRITTEN (before any cache/npz read) ===", flush=True)
    print(f"flags {len(flags)} (md5 {flags_md5}); k>=9 {len(k_big)}; "
          f"non-calendar k>=9 {stub['blind_population']['n_non_calendar']}", flush=True)

    out = dict(stub)
    out["status"] = "RUNNING"

    # ---- pool ----
    tok, cnt, pool_ids, ext_ids = build_pool(a.root)
    n_types = len(pool_ids)
    n_rows_cache = len(tok)
    pool_pos = {int(t): i for i, t in enumerate(pool_ids)}
    ext_pos = {int(t): i for i, t in enumerate(ext_ids)}
    counts_pool = cnt[pool_ids].astype(np.int64)
    lut = np.full(int(tok.max()) + 1, -1, dtype=np.int64)
    lut[pool_ids] = np.arange(n_types)
    print(f"pool {n_types} types, {n_rows_cache} cache rows", flush=True)
    assert member_hash([pool_pos[t] for t in MEMBERS_13_TOKEN_IDS]) == HASH_13, \
        "member_hash convention (md5 over sorted POOL POSITIONS) does not reproduce HASH_13"
    P13 = np.asarray(sorted(pool_pos[t] for t in MEMBERS_13_TOKEN_IDS))
    P14 = np.asarray(sorted(pool_pos[t] for t in MEMBERS_14_TOKEN_IDS))
    assert member_hash(P14) == HASH_14

    from transformers import AutoTokenizer
    from gfdisc.env import MODEL_NAME
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=os.environ.get("HF_TOKEN"))
    pool_strings = [tokenizer.convert_ids_to_tokens(int(t)) or tokenizer.decode([int(t)])
                    for t in pool_ids]
    pool_strings = [s.replace("▁", " ") for s in pool_strings]
    blocks = resolve_blocks(pool_strings)
    token_map = json.loads((REPO / "results/token_map.json").read_text())
    calendar_pos = sorted({pool_pos[int(x)]
                           for key in ("months", "days")
                           for name, ids in token_map.get(key, {}).items()
                           for x in ids if int(x) in pool_pos})
    ord_ids, ord_idx = [], []
    for i, w in enumerate(ORDINAL_WORDS):
        enc = tokenizer.encode(w, add_special_tokens=False)
        if len(enc) == 1 and int(enc[0]) in pool_pos:
            ord_ids.append(pool_pos[int(enc[0])]); ord_idx.append(i)
    print(f"ordinal control resolved: {len(ord_ids)}/10 in pool; "
          f"calendar-excluded positions {len(calendar_pos)}", flush=True)

    # ---- pair bookkeeping for channel 2 (shared across streamed layers) ----
    own, parity, per_lag = pair_index(tok, n_rows_cache, lut)
    C, rs = {}, {}
    for lag in LAGS:
        pl = per_lag[lag]
        Cl = np.zeros((2, n_types, n_types), dtype=np.int64)
        for p in (0, 1):
            m = pl["succ_ok"] & (parity == p)
            np.add.at(Cl[p], (pl["prev"][m], own[m]), 1)
            del m
        C[lag] = Cl
        rs[lag] = Cl.sum(axis=2)
    included = {lag: (rs[lag][0] >= MIN_PAIRS_PER_HALF) & (rs[lag][1] >= MIN_PAIRS_PER_HALF)
                for lag in LAGS}
    pred_par = {lag: np.where(per_lag[lag]["succ_ok"],
                              per_lag[lag]["prev"] + parity.astype(np.int32) * n_types,
                              np.int32(-1)).astype(np.int32) for lag in LAGS}
    del parity, per_lag
    gc.collect()
    print(f"pairs: included types lag1 {int(included[1].sum())}, "
          f"lag2 {int(included[2].sum())} of {n_types}", flush=True)

    # =================================================================================
    # per-layer: channel 1 (+ channel 2 where streamed) + member reconstruction
    # =================================================================================
    C1 = {}          # layer -> channel-1 coords, POST R10 drop  [n_types, 1151]
    pre_gap = {}     # (layer, member_hash) -> PRE-drop gap_ratio (reproduction check).
                     # Computed in-loop and the full PRE-drop matrix is then freed: holding
                     # 11 x [3795, 1152] float64 costs 385 MB for nothing but this check.
    C2 = {}          # layer -> lag1 deltas, POST drop
    C2_lag2 = {}
    hashmap = {}     # layer -> {member_hash: (arm, members)}
    layer_notes = {}

    # channel 2 for L6/L7 is read from the frozen 41b artifact BEFORE the layer loop, so
    # the declared positive control can be printed the moment L6/L7 channel 1 exists
    # rather than after the streams.
    zd = np.load(REPO / ROBUST_DELTAS)
    assert np.array_equal(zd["pool_ids"].astype(np.int64), pool_ids.astype(np.int64))
    for L in [x for x in LAYERS_DELTA_FROM_NPZ if x in layers_all]:
        stored_drop = [int(x) for x in zd[f"L{L}_lag1_dropped_coords"]]
        assert stored_drop == [R10_DROP[L]], f"L{L} stored drop {stored_drop} != R10"
        C2[L] = zd[f"L{L}_lag1"].astype(np.float64)
        C2_lag2[L] = zd[f"L{L}_lag2"].astype(np.float64)
        assert np.isfinite(C2[L]).all() and np.isfinite(C2_lag2[L]).all()
    del zd
    gc.collect()
    early = {}
    # positive-control layers first, so a control failure is visible before the streams
    order = ([L for L in (6, 7) if L in layers_all]
             + [L for L in layers_all if L not in (6, 7)])

    for L in order:
        tL = time.time()
        print(f"\n================ layer {L} ================", flush=True)
        drop_coord = R10_DROP[L]
        if L in layers_stream:
            TM_acc, S, seen = stream_deltas(root, L, own, pred_par, n_rows_cache,
                                            n_types, D_SAE)
            assert np.array_equal(seen, counts_pool), "streamed counts != corpus counts"
            TM_acc /= np.maximum(seen, 1)[:, None]
            TM_acc[seen == 0] = 0.0
            sae = load_sae(L, a.l0, token=os.environ.get("HF_TOKEN"))
            W = get_decoder(sae); del sae
            assert W.shape == (D_SAE, D_MODEL)
            R = project(TM_acc, W).astype(np.float64)
            assert np.isfinite(R).all(), f"non-finite channel-1 means at L{L}"
            for lag in LAGS:
                for p in (0, 1):
                    base = p * n_types
                    for lo in range(0, n_types, 512):
                        hi = min(lo + 512, n_types)
                        S[lag][base + lo:base + hi] -= (C[lag][p][lo:hi].astype(np.float64)
                                                        @ TM_acc)
            del TM_acc
            gc.collect()
            halves = {}
            for lag in LAGS:
                for p in (0, 1):
                    S[lag][p * n_types:(p + 1) * n_types] /= np.maximum(rs[lag][p], 1)[:, None]
                h0 = project(S[lag][:n_types], W).astype(np.float64)
                h1 = project(S[lag][n_types:], W).astype(np.float64)
                S[lag] = None
                gc.collect()
                w0 = rs[lag][0].astype(np.float64)[:, None]
                w1 = rs[lag][1].astype(np.float64)[:, None]
                halves[lag] = (h0, h1, (w0 * h0 + w1 * h1) / np.maximum(w0 + w1, 1))
            del S, W
            gc.collect()
            keeps = {}
            for lag in LAGS:
                h0, h1, full = halves[lag]
                keep, sd, dropped, shares = outlier_drop_rule(h0, h1, included[lag])
                keeps[lag] = (keep, sd, dropped, shares)
                assert np.isfinite(full).all(), f"non-finite deltas L{L} lag{lag}"
                print(f"  [DROP L{L} lag{lag}] computed {dropped} shares "
                      f"{[round(x,4) for x in shares]}; R10 frozen {[drop_coord]}", flush=True)
            agree = {lag: (keeps[lag][2] == [drop_coord]) for lag in LAGS}
            layer_notes[L] = dict(drop_computed={str(l): keeps[l][2] for l in LAGS},
                                  drop_frozen=[drop_coord],
                                  drop_rule_agrees_with_R10={str(l): bool(agree[l])
                                                             for l in LAGS})
            keep_frozen = np.ones(D_MODEL, dtype=bool)
            keep_frozen[drop_coord] = False
            C2[L] = np.ascontiguousarray(halves[1][2][:, keep_frozen])
            C2_lag2[L] = np.ascontiguousarray(halves[2][2][:, keep_frozen])
            if a.save_deltas:
              np.savez_compressed(
                REPO / f"results/phase4_lag1_deltas_L{L}.npz",
                pool_ids=pool_ids.astype(np.int32),
                lag1=halves[1][2][:, keep_frozen].astype(np.float16),
                lag2=halves[2][2][:, keep_frozen].astype(np.float16),
                kept_coords=np.flatnonzero(keep_frozen).astype(np.int32),
                dropped_coords=np.asarray([drop_coord], dtype=np.int32),
                lag1_pool_noise_sd=keeps[1][1].astype(np.float32),
                lag2_pool_noise_sd=keeps[2][1].astype(np.float32),
                n_included_lag1=np.int64(included[1].sum()),
                n_included_lag2=np.int64(included[2].sum()))
            del halves
            gc.collect()
        else:
            acc, d_sae, n_rows_used, _ = type_means(a.root, L, tok, cnt, ext_ids)
            sae = load_sae(L, a.l0, token=os.environ.get("HF_TOKEN"))
            W = get_decoder(sae); del sae
            R_ext = project(acc, W)
            del acc
            gc.collect()
            R = R_ext[np.asarray([ext_pos[int(t)] for t in pool_ids])].astype(np.float64)
            del R_ext, W
            gc.collect()
            assert np.isfinite(R).all(), f"non-finite channel-1 means at L{L}"
            layer_notes[L] = dict(drop_frozen=[drop_coord], drop_computed=None,
                                  drop_rule_agrees_with_R10=None)

        keep = np.ones(D_MODEL, dtype=bool)
        keep[drop_coord] = False
        C1[L] = np.ascontiguousarray(R[:, keep])

        # ---- member reconstruction: rerun the scripts/22 arms and match hashes ----
        t_cl = time.time()
        arms = all_partitions(R, n_types)
        hm = {}
        for arm, clusters in arms.items():
            for c in clusters:
                hm.setdefault(member_hash(c), (arm, [int(x) for x in c]))
        hashmap[L] = hm
        for f in flags:
            if f["layer"] == L and f["member_hash"] in hm:
                pre_gap[(L, f["member_hash"])] = gap_ratio_of(
                    R[np.asarray(hm[f["member_hash"]][1])])
        if L in (6, 7):
            pre_gap[(L, HASH_13 if L == 6 else HASH_14)] = gap_ratio_of(
                R[P13 if L == 6 else P14])
        layer_notes[L]["n_clusters_by_arm"] = {k2: len(v) for k2, v in arms.items()}
        layer_notes[L]["clustering_seconds"] = round(time.time() - t_cl, 1)
        print(f"  clusters rebuilt {layer_notes[L]['n_clusters_by_arm']} "
              f"({layer_notes[L]['clustering_seconds']}s); layer total "
              f"{round(time.time()-tL,1)}s", flush=True)
        del arms, R
        gc.collect()

        # early visibility on the declared positive control (same numbers as the final
        # block; printed here so a control failure is known before the streams run)
        if L in (6, 7):
            Xc = C1[L] - C1[L].mean(0, keepdims=True)
            Xn = (Xc / np.maximum(np.linalg.norm(Xc, axis=1, keepdims=True), 1e-12)
                  ).astype(np.float32)
            nl = NullCache(Xn, C1[L], C2[L], counts_pool, n_clusterlike)
            P = P13 if L == 6 else P14
            adj = adjudicate(P, C1[L], C2[L], nl, L)
            s12 = adjudicate(np.asarray(sorted(pool_pos[t] for t in MEMBERS_13_TOKEN_IDS
                                               if t != 5656)), C1[L], C2[L], nl, L)
            early[f"L{L}"] = dict(flag_set=adj, strict12=s12)
            print(f"  [EARLY POSITIVE CONTROL L{L}] flag set k={adj['k']} gap "
                  f"{adj['gap_ratio']:.4f} rho {adj['two_channel_rho']:.4f} thr "
                  f"{adj['threshold_p99']} LOOclosurefail {adj['n_loo_closure_fail']} "
                  f"-> {adj['verdict']} (noLOO {adj['PASS_noLOO']}) | strict12 gap "
                  f"{s12['gap_ratio']:.4f} rho {s12['two_channel_rho']:.4f} thr "
                  f"{s12['threshold_p99']} -> {s12['verdict']} "
                  f"(noLOO {s12['PASS_noLOO']})", flush=True)
            del Xc, Xn, nl
            gc.collect()

    out["early_positive_control"] = early
    del own, pred_par, C, tok
    gc.collect()

    # ---- raw residual secondary (L4-L9), read from the existing artifact ----
    raw = {}
    zr = np.load(REPO / FULLN_RAW)
    assert np.array_equal(zr["pool_ids"].astype(np.int64), pool_ids.astype(np.int64)), \
        "fulln raw artifact pool_ids differ from the rebuilt pool"
    for L in [x for x in (4, 5, 6, 7, 8, 9) if x in layers_all]:
        A = zr[f"R_L{L}"].astype(np.float64)
        assert np.isfinite(A).all(), f"non-finite raw residual means L{L}"
        keep = np.ones(D_MODEL, dtype=bool); keep[R10_DROP[L]] = False
        raw[L] = np.ascontiguousarray(A[:, keep])
    del zr
    gc.collect()

    print(f"\nchannel 2 available at layers {sorted(C2)}", flush=True)

    # ---- null machinery per layer ----
    nulls, nulls_lag2, nulls_raw = {}, {}, {}
    for L in layers_all:
        Xc = C1[L] - C1[L].mean(0, keepdims=True)
        Xn = (Xc / np.maximum(np.linalg.norm(Xc, axis=1, keepdims=True), 1e-12)).astype(np.float32)
        nulls[L] = NullCache(Xn, C1[L], C2.get(L), counts_pool, n_clusterlike)
        nulls_lag2[L] = NullCache(Xn, C1[L], C2_lag2.get(L), counts_pool, n_clusterlike)
        if L in raw:
            Xr = raw[L] - raw[L].mean(0, keepdims=True)
            Xrn = (Xr / np.maximum(np.linalg.norm(Xr, axis=1, keepdims=True), 1e-12)
                   ).astype(np.float32)
            nulls_raw[L] = NullCache(Xrn, raw[L], C2.get(L), counts_pool, n_clusterlike)
            del Xr, Xrn
        del Xc
    gc.collect()

    # =================================================================================
    # positive controls FIRST
    # =================================================================================
    print("\n============ POSITIVE CONTROLS ============", flush=True)
    pos = {}
    for L, P, hsh, name in [t for t in ((6, P13, HASH_13, "13-set"),
                                        (7, P14, HASH_14, "14-set")) if t[0] in layers_all]:
        pre = pre_gap.get((L, hsh))
        repro = dict(expected=ANCHOR_GAP_RATIO[L], observed=pre,
                     passed=bool(pre is not None
                                 and abs(pre - ANCHOR_GAP_RATIO[L]) <= ANCHOR_TOL))
        found = hashmap[L].get(hsh)
        recon = dict(hash_found=found is not None,
                     arm=None if found is None else found[0],
                     members_match=bool(found is not None
                                        and sorted(found[1]) == sorted(P.tolist())))
        adj = adjudicate(P, C1[L], C2[L], nulls[L], L)
        strict12 = adjudicate(np.asarray(sorted(pool_pos[t] for t in
                                                MEMBERS_13_TOKEN_IDS if t != 5656)),
                              C1[L], C2[L], nulls[L], L)
        l2 = adjudicate(P, C1[L], C2_lag2[L], nulls_lag2[L], L, want_loo=False)
        pos[f"L{L}_{name}"] = dict(layer=L, member_hash=hsh, k=int(len(P)),
                                   gap_ratio_pre_drop=pre, reproduction=repro,
                                   reconstruction=recon, adjudication=adj,
                                   lag2_specificity=dict(rho=l2["two_channel_rho"],
                                                         threshold=l2["threshold_p99"],
                                                         clears=l2["rho_pass"]),
                                   strict12_variant=strict12,
                                   raw_secondary=(adjudicate(P, raw[L], C2[L],
                                                             nulls_raw[L], L)
                                                  if L in nulls_raw else None))
        print(f"  L{L} {name}: pre-drop gap {pre} (stored {ANCHOR_GAP_RATIO[L]}, "
              f"repro {repro['passed']}); hash found {recon['hash_found']} arm {recon['arm']}; "
              f"post-drop gap {adj['gap_ratio']:.4f}; rho {adj['two_channel_rho']:.4f} vs "
              f"thr {adj['threshold_p99']}; LOO closure fails {adj['n_loo_closure_fail']}; "
              f"VERDICT {adj['verdict']} (noLOO {adj['PASS_noLOO']})", flush=True)
    out["positive_controls"] = pos
    control_passed = all(v["adjudication"]["verdict"] == "PASS" for v in pos.values())
    control_passed_noLOO = all(v["adjudication"]["PASS_noLOO"] for v in pos.values())

    # =================================================================================
    # the 51 flags
    # =================================================================================
    print("\n============ FLAG RE-ADJUDICATION ============", flush=True)
    rows = []
    for f in flags:
        L, arm, k, hsh = f["layer"], f["arm"], int(f["k"]), f["member_hash"]
        if L not in layers_all:
            rows.append(dict(layer=L, arm=arm, k=k, member_hash=hsh,
                             calendar=bool(f["calendar"]), reconstruction="LAYER NOT RUN",
                             verdict="LAYER NOT RUN"))
            continue
        found = hashmap[L].get(hsh)
        rec_mode = "hash"
        members = None
        if found is not None:
            members = found[1]
        if members is not None:
            pre = pre_gap.get((L, hsh))
            ok = pre is not None and abs(pre - f["gap_ratio"]) <= 1e-6
            if not ok:
                rec_mode = "hash-but-gap_ratio-mismatch"
        else:
            pre, ok = None, False
            rec_mode = "NOT RECONSTRUCTED"
        row = dict(layer=L, arm=arm, arm_reconstructed=None if found is None else found[0],
                   k=k, member_hash=hsh, calendar=bool(f["calendar"]),
                   stored_gap_ratio=f["gap_ratio"], reconstruction=rec_mode,
                   gap_ratio_pre_drop=pre, gap_ratio_reproduces=bool(ok),
                   tokens=f.get("tokens"))
        if members is None:
            row["verdict"] = "NOT RECONSTRUCTED"
            rows.append(row); continue
        M = np.asarray(members)
        adj = adjudicate(M, C1[L], C2.get(L), nulls[L], L)
        row.update(adj)
        if not adj["adjudicable"] and k >= K_MIN_ADJUDICABLE:
            row["verdict"] = "NOT ADJUDICABLE"
            row["not_adjudicable_reason"] = "no channel-2 deltas declared at this layer"
        if adj["adjudicable"]:
            l2 = adjudicate(M, C1[L], C2_lag2.get(L), nulls_lag2[L], L, want_loo=False)
            row["lag2_rho"] = l2["two_channel_rho"]
            row["lag2_threshold"] = l2["threshold_p99"]
            row["lag2_clears_R11_review"] = bool(l2["rho_pass"])
        if L in raw:
            row["raw_secondary"] = adjudicate(M, raw[L], C2.get(L), nulls_raw[L], L,
                                              want_loo=False)
        rows.append(row)
        print(f"  L{L:<2} {arm:<16} k={k:<3} gap {('%.3f'%adj['gap_ratio']) if adj['gap_ratio'] else 'NA':>6} "
              f"rho {('%.3f'%adj['two_channel_rho']) if adj['two_channel_rho'] is not None else '  NA':>6} "
              f"thr {('%.3f'%adj['threshold_p99']) if adj['threshold_p99'] is not None else '  NA':>6} "
              f"LOOfail {adj['n_loo_closure_fail']:<3} {row['verdict']}", flush=True)
    out["flags"] = rows
    out["null_cells"] = [v for L in layers_all for v in nulls[L].cache.values()]
    out["null_cells_raw_secondary"] = [v for L in nulls_raw for v in nulls_raw[L].cache.values()]

    # extra reported nulls at the adjudicated cells
    extra = []
    for (L, k, kind) in sorted({(r["layer"], r["k"], "x") for r in rows
                                if r.get("adjudicable")}):
        for kind2 in ("random", "freq"):
            extra.append(nulls[L].build(L, k, kind=kind2))
    out["null_cells_reported"] = extra

    # =================================================================================
    # negative controls
    # =================================================================================
    print("\n============ NEGATIVE CONTROLS ============", flush=True)
    neg = {}
    for L in [x for x in (6, 7) if x in layers_all]:
        adj = adjudicate(np.asarray(ord_ids), C1[L], C2[L], nulls[L], L)
        neg[f"ordinal_L{L}"] = dict(adj)
        neg[f"ordinal_L{L}"]["words"] = [ORDINAL_WORDS[i] for i in ord_idx]
        print(f"  ordinal one..ten L{L}: k={len(ord_ids)} gap {adj['gap_ratio']:.3f} "
              f"rho {adj['two_channel_rho']} thr {adj['threshold_p99']} "
              f"VERDICT {adj['verdict']} (noLOO {adj['PASS_noLOO']})", flush=True)

    # frequency-pseudo populations
    lc = np.log10(np.maximum(counts_pool.astype(np.float64), 1.0))
    allowed = np.ones(n_types, dtype=bool)
    allowed[np.asarray(calendar_pos, dtype=np.int64)] = False
    cand = np.flatnonzero(allowed)
    for L, P, name in [t for t in ((6, P13, "13-set"), (7, P14, "14-set"))
                       if t[0] in layers_all]:
        rng = np.random.default_rng(7_300_000 + L)
        nearest = np.asarray([cand[np.argsort(np.abs(lc[cand] - lc[p]), kind="stable")
                                   [:FREQ_PSEUDO_NEAREST]] for p in P])
        n_pass = n_pass_noloo = n_closed = 0
        for _ in range(N_FREQ_PSEUDO):
            pick = None
            for _a in range(20):
                cp = nearest[np.arange(len(P)), rng.integers(0, FREQ_PSEUDO_NEAREST, len(P))]
                if len(set(cp.tolist())) == len(P):
                    pick = cp; break
            if pick is None:
                pick = cp
            adj = adjudicate(pick, C1[L], C2[L], nulls[L], L)
            n_closed += int(adj["closure_pass"])
            n_pass += int(adj["verdict"] == "PASS")
            n_pass_noloo += int(adj["PASS_noLOO"])
        neg[f"freq_pseudo_L{L}"] = dict(n_draws=N_FREQ_PSEUDO, anchor=name,
                                        n_closure_pass=n_closed, n_pass_frozen=n_pass,
                                        n_pass_noLOO=n_pass_noloo,
                                        gate_max=FREQ_PSEUDO_MAX_PASSES,
                                        gate_met=bool(n_pass <= FREQ_PSEUDO_MAX_PASSES))
        print(f"  freq-pseudo L{L} ({name}): {n_pass}/{N_FREQ_PSEUDO} PASS frozen, "
              f"{n_pass_noloo} PASS noLOO, {n_closed} closed", flush=True)

    # class-block permutation of channel 2
    for L, P, name in [t for t in ((6, P13, "13-set"), (7, P14, "14-set"))
                       if t[0] in layers_all]:
        rng = np.random.default_rng(7_400_000 + L)
        k = len(P)
        thr_pair = (nulls[L].build(L, k)["threshold_p99_conditional"],
                    nulls[L].build(L, k - 1)["threshold_p99_conditional"])
        n_pass = n_pass_noloo = 0
        rhos = []
        for _ in range(N_CLASS_PERM):
            perm = np.arange(n_types)
            for b in blocks:
                perm[b] = b[rng.permutation(len(b))]
            D = C2[L][perm]
            adj = adjudicate(P, C1[L], D, nulls[L], L, thr_override=thr_pair)
            rhos.append(adj["two_channel_rho"])
            n_pass += int(adj["verdict"] == "PASS")
            n_pass_noloo += int(adj["PASS_noLOO"])
        neg[f"class_perm_L{L}"] = dict(
            n_draws=N_CLASS_PERM, anchor=name, n_pass_frozen=n_pass,
            n_pass_noLOO=n_pass_noloo, gate_max=CLASS_PERM_MAX_PASSES,
            gate_met=bool(n_pass <= CLASS_PERM_MAX_PASSES),
            threshold_used=thr_pair[0],
            rho_median=float(np.median(rhos)), rho_p99=float(np.percentile(rhos, 99)),
            note=("channel-2 rows permuted among types WITHIN lexical blocks; adjudicated "
                  "against the TIER's own unpermuted threshold, which is the FDR question"))
        print(f"  class-perm L{L} ({name}): {n_pass}/{N_CLASS_PERM} PASS frozen, "
              f"{n_pass_noloo} noLOO; rho median {np.median(rhos):.3f}", flush=True)
    out["negative_controls"] = neg

    # =================================================================================
    # planted power
    # =================================================================================
    print("\n============ PLANTED POWER ============", flush=True)
    L = 6
    _, r1 = plane_angles(C1[L][P13])
    _, r2 = plane_angles(C2[L][P13])
    thr12 = nulls[L].build(L, PLANTED_K)["threshold_p99_conditional"]
    thr11 = nulls[L].build(L, PLANTED_K - 1)["threshold_p99_conditional"]
    planted = dict(layer=L, k=PLANTED_K, radius_channel1=float(r1),
                   radius_channel2=float(r2), threshold_k=thr12, threshold_k_minus_1=thr11,
                   curve=[])

    def plant(coords, carriers, radius, rho, rng, order_angles):
        X = coords[carriers].copy()
        mu = X.mean(0, keepdims=True)
        Xc = X - mu
        U, Sv, Vt = np.linalg.svd(Xc, full_matrices=False)
        B = Vt[:2]                                   # in-plane basis
        inplane = Xc @ B.T
        out_of_plane = Xc - inplane @ B
        new = np.column_stack([radius * np.cos(order_angles), radius * np.sin(order_angles)])
        new = new + rng.normal(0.0, rho * radius, size=new.shape)
        return mu + out_of_plane + new @ B

    for rho_j in PLANTED_RHOS:
        rng = np.random.default_rng(7_500_000 + int(1000 * rho_j))
        nc = nt = nj = nj_loo = 0
        for _ in range(n_planted):
            carriers = rng.choice(n_types, size=PLANTED_K, replace=False)
            th = 2 * np.pi * np.arange(PLANTED_K) / PLANTED_K
            X1 = plant(C1[L], carriers, r1, rho_j, rng, th)
            X2 = plant(C2[L], carriers, r2, rho_j, rng, th)
            assert np.isfinite(X1).all() and np.isfinite(X2).all()
            g, r = cell_stats(X1, X2)
            cl = bool(g is not None and g < GAP_RATIO_CLOSURE)
            tc = bool(r is not None and thr12 is not None and r > thr12)
            nc += int(cl); nt += int(tc); nj += int(cl and tc)
            if cl and tc:
                ok = True
                for i in range(PLANTED_K):
                    gi, ri = cell_stats(np.delete(X1, i, 0), np.delete(X2, i, 0))
                    if not (gi is not None and gi < GAP_RATIO_CLOSURE
                            and ri is not None and thr11 is not None and ri > thr11):
                        ok = False; break
                nj_loo += int(ok)
        planted["curve"].append(dict(rho=rho_j, n=n_planted,
                                     recall_closure=nc / n_planted,
                                     recall_two_channel=nt / n_planted,
                                     recall_joint=nj / n_planted,
                                     recall_joint_with_LOO=nj_loo / n_planted))
        print(f"  rho={rho_j:.2f}: closure {nc/n_planted:.3f} two-channel {nt/n_planted:.3f} "
              f"joint {nj/n_planted:.3f} joint+LOO {nj_loo/n_planted:.3f}", flush=True)

    def rho80(field, curve=None):
        curve = planted["curve"] if curve is None else curve
        xs = [c["rho"] for c in curve]
        ys = [c[field] for c in curve]
        for i in range(len(xs) - 1):
            if ys[i] >= POWER_TARGET > ys[i + 1]:
                f = (ys[i] - POWER_TARGET) / max(ys[i] - ys[i + 1], 1e-12)
                return float(xs[i] + f * (xs[i + 1] - xs[i]))
        return float(xs[-1]) if ys[-1] >= POWER_TARGET else None
    # --- co-reported SECONDARY, non-verdict: the same ring with NO carrier residual ---
    # 2.1's plant is literal ("into 12 random type means"), so a random carrier's
    # out-of-plane spread competes with the planted radius and caps closure recall well
    # below 1 even at zero jitter. The clean plant (pure ring in a random 2-plane, the
    # scripts/41b convention) isolates the tier's INTRINSIC power and is the curve
    # comparable to the project's rho80 = 0.12-0.16 anchor.
    clean = []
    d1 = C1[L].shape[1]
    for rho_j in PLANTED_RHOS:
        rng = np.random.default_rng(7_600_000 + int(1000 * rho_j))
        nc = nt = nj = nj_loo = 0
        th = 2 * np.pi * np.arange(PLANTED_K) / PLANTED_K
        for _ in range(n_planted):
            def ring(radius):
                B, _ = np.linalg.qr(rng.normal(size=(d1, 2)))
                pts = np.column_stack([radius * np.cos(th), radius * np.sin(th)])
                pts = pts + rng.normal(0.0, rho_j * radius, size=pts.shape)
                return pts @ B.T
            X1, X2 = ring(r1), ring(r2)
            assert np.isfinite(X1).all() and np.isfinite(X2).all()
            g, r = cell_stats(X1, X2)
            cl = bool(g is not None and g < GAP_RATIO_CLOSURE)
            tc = bool(r is not None and thr12 is not None and r > thr12)
            nc += int(cl); nt += int(tc); nj += int(cl and tc)
            if cl and tc:
                ok = True
                for i in range(PLANTED_K):
                    gi, ri = cell_stats(np.delete(X1, i, 0), np.delete(X2, i, 0))
                    if not (gi is not None and gi < GAP_RATIO_CLOSURE
                            and ri is not None and thr11 is not None and ri > thr11):
                        ok = False; break
                nj_loo += int(ok)
        clean.append(dict(rho=rho_j, n=n_planted, recall_closure=nc / n_planted,
                          recall_two_channel=nt / n_planted, recall_joint=nj / n_planted,
                          recall_joint_with_LOO=nj_loo / n_planted))
        print(f"  [clean] rho={rho_j:.2f}: closure {nc/n_planted:.3f} two-channel "
              f"{nt/n_planted:.3f} joint {nj/n_planted:.3f} joint+LOO "
              f"{nj_loo/n_planted:.3f}", flush=True)
    planted["clean_plant_secondary"] = clean

    planted["rho80"] = {f: rho80(f) for f in ("recall_closure", "recall_two_channel",
                                              "recall_joint", "recall_joint_with_LOO")}
    planted["rho80_clean_plant_secondary"] = {
        f: rho80(f, clean) for f in ("recall_closure", "recall_two_channel",
                                     "recall_joint", "recall_joint_with_LOO")}
    out["planted"] = planted
    print(f"  rho80 {planted['rho80']}", flush=True)

    # =================================================================================
    out["layer_notes"] = layer_notes
    adjudicated = [r for r in rows if r.get("adjudicable")]
    passers = [r for r in adjudicated if r["verdict"] == "PASS"]
    passers_noloo = [r for r in adjudicated if r.get("PASS_noLOO")]
    out["summary"] = dict(
        n_flags=len(flags), n_reconstructed=int(sum(1 for r in rows
                                                    if r["reconstruction"] == "hash")),
        n_gap_ratio_reproduced=int(sum(1 for r in rows if r.get("gap_ratio_reproduces"))),
        n_adjudicable=len(adjudicated), n_not_adjudicable=len(rows) - len(adjudicated),
        n_pass_frozen=len(passers), n_pass_noLOO=len(passers_noloo),
        passers_frozen=[dict(layer=r["layer"], arm=r["arm"], k=r["k"],
                             member_hash=r["member_hash"], calendar=r["calendar"])
                        for r in passers],
        passers_noLOO=[dict(layer=r["layer"], arm=r["arm"], k=r["k"],
                            member_hash=r["member_hash"], calendar=r["calendar"])
                       for r in passers_noloo],
        positive_control_passed_frozen=bool(control_passed),
        positive_control_passed_noLOO=bool(control_passed_noLOO),
        gate=("PASS" if (control_passed
                         and all(v.get("gate_met", True) for k2, v in neg.items()
                                 if "freq_pseudo" in k2))
              else "FAIL"),
        expected_false_passers_9=0.09, expected_false_passers_10=0.10)
    out["status"] = "COMPLETE"
    out["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    out["runtime_seconds"] = round(time.time() - t_start, 1)
    Path(a.out).write_text(json.dumps(out, indent=2, default=float))
    print(f"\nwrote {a.out} in {out['runtime_seconds']}s", flush=True)
    print(json.dumps(out["summary"], indent=2, default=float), flush=True)


if __name__ == "__main__":
    main()
