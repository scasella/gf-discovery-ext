"""49 -- occurrence-plane screen (Phase 5, prereg 2.2, proposer C).

Pre-registration: docs/phase5_prereg_2026-08-22.md section 2.2, rules R13-R18.
Tier imported verbatim from scripts/42_two_channel_readjudicate.py (closure first,
L1 saturation, L2 seed band, LOO, frequency-matched arm).

The proposer. Phase 4 established that every proposer so far assembles from type-mean
proximity and inherits token-class clumping. This one does not touch type means at all:
it clusters TOKEN TYPES by the geometry of their per-occurrence residual clouds --
specifically by the second principal angle between the clouds' own top-2 planes. That
plane is a WITHIN-type object (context variance) and is near-orthogonal to the
between-type ring plane, which is why the extra conjunct of 2.2 (> 60 deg from the
cluster's own type-mean ring plane) is checkable rather than vacuous.

Nothing here may relax the frozen tier. Conventions 2.2 leaves unstated are pinned in
declared_rules() and written to disk BEFORE any cache or npz read.
"""
import argparse, gc, json, os, sys, time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HOME", str(REPO / "cache/hf"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

import importlib
s42 = importlib.import_module("42_two_channel_readjudicate")
from gfdisc.cache import SparseReader
from gfdisc.saes import load_sae, get_decoder
from gfdisc.fits import pca_2d

import scipy.sparse as sp
import scipy.cluster.hierarchy as sch
from scipy.spatial.distance import squareform

D_MODEL, D_SAE, SEQ_LEN = 1152, 16384, 255
LAYERS = (5, 6, 7)
TAUS = (30.0, 40.0, 45.0)
ROWS_PER_PARITY = 100
K_MIN = 9                      # R8
DEDUP_LADDER = (0.88, 0.90, 0.92)
DEDUP_PRIMARY = 0.90           # R15
RING_ANGLE_MIN = 60.0          # 2.2 extra conjunct, MINIMUM principal angle
N_CLUSTERLIKE = 6_000
N_SEEDS_BAND = 3               # Addendum L2
N_PERM_SEARCH = 20             # R13
N_PLANT_CARRIERS = 400
PLANT_DELTAS = (0, 5, 10, 15, 20, 25, 30, 40, 50, 60)
N_PLANTS = 50
PLANT_K = 12
CH3_KMAX = 10                  # R16
N_CH3_RANDOM = 500
MONTH_TOKEN_IDS = [3353, 5142, 5261, 5365, 5505, 5729, 5857, 6027, 6198, 6243, 6384, 7250]
MODEL_ST = ("cache/hf/hub/models--google--gemma-3-1b-pt/snapshots/"
            "fcf18a2a879aab110ca39f8bffbccd5d49d8eb29/model.safetensors")
ROBUST_DELTAS = "results/phase3_lag_action_robust_deltas.npz"
FULLN_RAW = "results/phase3_token_type_mean_resid_fulln.npz"

# families seen in the 619-pool exposure (prereg section 0) -- any passer overlapping
# these must be flagged as exposed, per the coordinator's instruction.
EXPOSED_619 = set("""
 topics  experts  tasks  methods  communities  buildings  guidelines  vehicles  protection
 requires  artists  consumer  transition  users  Please  cultural  hundreds  brands
 involving  communication  throughout  directly  difficult  real  independent  big
 primarily  small  except  mainly  working  System  system  Hospital  football  Systems
 Education  colour  patients  objects  messages  images  pink  meetings  Special  Business
 Federal  Children  Small  album  boys  app  song  hotel  essential  typical  amazing
 important  critical  Saturday  Sunday  N  eight  K  Q
""".split("  ")) | {"In", "Last", "Once", "There", "Most", "For", "While", "Who",
                    "Another", "Before", "These", "Other", "Have",
                    "ac", "om", "aw", "us", "ier", "il"}


def declared_rules(av):
    return dict(
        prereg="docs/phase5_prereg_2026-08-22.md section 2.2 (proposer C); rules R13-R18",
        object_class=("an unordered set of token types whose per-occurrence residual clouds "
                      "share a top-2 principal plane. WITHIN-type context variance, not "
                      "between-type mean geometry."),
        rows=("SAE-decoded residual rows: CSR code row @ W_dec, NO b_dec (prereg 4 "
              "convention, identical to scripts/22/39)."),
        row_sampling_rule=(
            "DECLARED CHOICE: random-seeded, not first-200-by-sequence-order. Per type, at "
            "most %d even-sequence rows and at most %d odd-sequence rows, drawn without "
            "replacement by np.random.default_rng(token_id) when that parity holds more than "
            "%d rows, all rows of the parity otherwise. Rationale: first-200-by-sequence "
            "order confounds the plane with document identity (the cache is document-ordered), "
            "which is exactly the nuisance the proposer would then cluster on. n is equalised "
            "at 200 for every type with >= %d rows in each parity; residual n-dependence is "
            "controlled by the R13 null, whose strata include the log10-count decile."
            % (ROWS_PER_PARITY, ROWS_PER_PARITY, ROWS_PER_PARITY, ROWS_PER_PARITY)),
        plane=("top-2 principal directions of the cloud centred at its OWN mean, post-R10 "
               "drop, computed from the eigendecomposition of the n x n Gram. Declared "
               "because np.linalg.svd of the n x d matrix fails to converge on some clouds "
               "(measured during the pre-freeze probe); the Gram route is exact and stable."),
        angle=("second principal angle theta2 between two 2-planes, closed form from the 2x2 "
               "singular values of Q_i^T Q_j. theta2 (not theta1) because two planes sharing "
               "only one direction are not the same plane."),
        clustering="scipy average linkage on the theta2 matrix, cut at tau in %s deg" % (TAUS,),
        tau_provenance=(
            "R7: the ladder is a null percentile, not a months-anchored constant. Calibrated "
            "on the pre-freeze 619-type probe (prereg section 0): 30 deg < count-matched "
            "random-pair p0.1 = 35.1 deg; 45 deg ~ random-pair p1 = 48.3 deg; 40 deg is the "
            "midpoint rung. The full-pool percentiles are RECOMPUTED here and reported; if "
            "they move, the ladder is reported against the moved percentiles, not re-cut."),
        proposals="clusters of size >= %d after R15 dedup at %.2f" % (K_MIN, DEDUP_PRIMARY),
        dedup="R15: drop the lower-count member of any pair with channel-1 centred cosine "
              "> tau_dedup; greedy by descending count; ladder %s reported" % (DEDUP_LADDER,),
        channel_1=("raw per-type mean residual R_L from %s (SAE-free, full n), post-R10 drop. "
                   "Coordinator-specified; 42 used the SAE mean-code decode as channel 1 and "
                   "raw as a secondary. Both are reported; RAW is verdict-bearing here."
                   % FULLN_RAW),
        channel_2=("identity-removed lag+1 successor-delta means, post-R10 drop. L6/L7 read "
                   "from %s (41b artifact); L5 computed here by the 42 delta stream "
                   "(s42.stream_deltas) with the identical construction, float32 cast."
                   % ROBUST_DELTAS),
        tier=("imported verbatim from scripts/42 (cell_stats, NullCache, adjudicate): "
              "closure < 2.5 evaluated FIRST (Addendum M1), L1 saturation rule, L2 three-seed "
              "band, LOO on both conjuncts, frequency-matched arm."),
        extra_conjunct=("prereg 2.2: the cluster's shared plane must be > %.0f deg from the "
                        "cluster's own type-mean ring plane. DECLARED READING: the MINIMUM "
                        "principal angle theta1 must exceed the threshold (the strict reading: "
                        "no direction of one plane lies within %.0f deg of the other). Shared "
                        "plane := top-2 eigenvectors of the sum of the members' plane "
                        "projectors." % (RING_ANGLE_MIN, RING_ANGLE_MIN)),
        null_R13=("%d permutations of PLANE LABELS within (token class x log10-count decile) "
                  "strata, each re-clustered at every tau and re-tiered; PLUS the channel-2 "
                  "row permutation within the same strata applied to the real proposals "
                  "(43c-style member specificity). Permuted-search tiers use a single-seed "
                  "%d-draw null (declared reduction; the real proposals use the 3-seed band)."
                  % (N_PERM_SEARCH, N_CLUSTERLIKE)),
        channel_3=("R16: mean over the layer's 4 query heads of the fraction of the "
                   "candidate's shared plane retained by the column space of that head's "
                   "o_proj slice (gemma-3-1b has 1 KV head, so the heads differ only in the "
                   "o slice). Conjunct for k <= %d, reported for all k." % CH3_KMAX),
        plant=("prereg 2.2 REPLACE-not-add: for %d count-matched carrier types, the cloud's "
               "top-2 plane is replaced by a common plane rotated per type by delta, "
               "A_new = (A - T V^T) + T P2^T, which preserves the top-2 variance fraction "
               "exactly. Both delta80 are reported: RAW (kNN-10 within-set purity >= 0.8) and "
               "NULL-CALIBRATED (median within-set theta2 < the count-matched pair p0.1). The "
               "raw curve is known to be leakage-inflated -- replaced planes leave the natural "
               "plane ensemble, so plants become mutual kNNs by being alien; the measured "
               "signature is within-set theta2 = arccos(cos^2 delta). The NULL-CALIBRATED "
               "value is the reportable one." % PLANT_K),
        positive_control=("R18: strict-12 months must fall out BLIND (no seeding, no carrier) "
                          "at L6 and PASS at L6; every (L, tau) cell reports the largest "
                          "all-month cluster and its rank by tightness."),
        negative_control="weekdays (7 types, counts 385-562) must not cluster.",
        exposure=("the 619-type pre-freeze probe (prereg section 0) saw non-calendar cluster "
                  "memberships at 40-50 deg. Every passer is checked against that list and "
                  "flagged `exposed_in_619` if any member appears in it."),
        deviations=[
            "frequency-matched arm uses 2,000 draws rather than R14's 200 (finer threshold "
            "estimate; reported, not relaxed).",
            "permuted-search tiers use a single-seed null (see null_R13).",
            "no MPS anywhere (R17); no forward pass of any kind.",
        ],
        args=av)


# ---------------------------------------------------------------------------
def top2_plane(M):
    """(2, d) orthonormal basis of the cloud's top-2 plane + eigenvalues. Gram route."""
    A = M - M.mean(0)
    G = A @ A.T
    w, U = np.linalg.eigh(G)
    o = np.argsort(w)[::-1]
    w = w[o]
    V = (A.T @ U[:, o[:2]]) / np.sqrt(np.maximum(w[:2], 1e-30))
    V /= np.maximum(np.linalg.norm(V, axis=0, keepdims=True), 1e-30)
    return np.linalg.qr(V)[0].T, w


def theta2_matrix(Q, block=1024):
    """Second principal angle (deg) between every pair of 2-planes. Closed form."""
    n = len(Q)
    A = np.ascontiguousarray(Q[:, 0, :], dtype=np.float32)
    B = np.ascontiguousarray(Q[:, 1, :], dtype=np.float32)
    out = np.empty((n, n), dtype=np.float64)
    for lo in range(0, n, block):
        hi = min(lo + block, n)
        aa = (A[lo:hi] @ A.T).astype(np.float64)
        ab = (A[lo:hi] @ B.T).astype(np.float64)
        ba = (B[lo:hi] @ A.T).astype(np.float64)
        bb = (B[lo:hi] @ B.T).astype(np.float64)
        T = aa * aa + ab * ab + ba * ba + bb * bb
        Dt = (aa * bb - ab * ba) ** 2
        s2sq = (T - np.sqrt(np.maximum(T * T - 4.0 * Dt, 0.0))) / 2.0
        out[lo:hi] = np.degrees(np.arccos(np.sqrt(np.clip(s2sq, 0.0, 1.0))))
        del aa, ab, ba, bb, T, Dt, s2sq
    out = (out + out.T) / 2.0
    np.fill_diagonal(out, 0.0)
    return out


def princ_angles(Ba, Bb):
    """Both principal angles (deg) between two (2, d) row-spanned planes."""
    Qa = np.linalg.qr(np.asarray(Ba).T)[0]
    Qb = np.linalg.qr(np.asarray(Bb).T)[0]
    s = np.clip(np.linalg.svd(Qa.T @ Qb, compute_uv=False), -1.0, 1.0)
    return np.degrees(np.arccos(s))


def shared_plane(Q, members):
    P = np.zeros((Q.shape[2], Q.shape[2]))
    for i in members:
        P += Q[i].T @ Q[i]
    w, V = np.linalg.eigh(P)
    o = np.argsort(w)[::-1][:2]
    return V[:, o].T, float(w[o].sum() / (2 * len(members)))


def dedup(members, C1c, counts, tau):
    """R15. members: array of pool positions. Returns kept positions."""
    m = sorted(members, key=lambda i: -counts[i])
    X = C1c[m]
    X = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-30)
    keep = []
    for i in range(len(m)):
        if all(float(X[i] @ X[j]) <= tau for j in keep):
            keep.append(i)
    return np.asarray(sorted(m[i] for i in keep))


def class_key(s):
    b = s.strip()
    if not b:
        return "ws"
    if b.isdigit():
        return "digit"
    if not b[0].isalnum():
        return "punct"
    if b[0].isupper():
        return "Cap"
    return "lower"


def strata(pool_strings, counts):
    dec = np.minimum((np.log10(np.maximum(counts, 1)) * 4).astype(int), 40)
    key = np.asarray([f"{class_key(s)}|{d}" for s, d in zip(pool_strings, dec)])
    return {k: np.flatnonzero(key == k) for k in np.unique(key)}


def stratified_perm(strat, n, rng):
    p = np.arange(n)
    for idx in strat.values():
        if len(idx) > 1:
            p[idx] = rng.permutation(idx)
    return p


# ---------------------------------------------------------------------------
def stream_planes(root, layer, tok, pool_ids, W, keep, keep_clouds=(), log_every=500):
    """One pass per type: gather its rows, decode, fit full/even/odd top-2 planes."""
    rd = SparseReader(root, layer)
    ip = rd.indptr
    order = np.argsort(tok, kind="stable")
    st = np.searchsorted(tok[order], pool_ids, side="left")
    en = np.searchsorted(tok[order], pool_ids, side="right")
    n = len(pool_ids)
    d = int(keep.sum())
    Q = np.zeros((n, 2, d)); Qa = np.zeros((n, 2, d)); Qb = np.zeros((n, 2, d))
    f2 = np.zeros(n); nrow = np.zeros(n, dtype=np.int64); sh = np.zeros(n)
    clouds = {}
    want = set(int(x) for x in keep_clouds)
    t0 = time.time()

    def decode(rows):
        s = ip[rows]
        ln = (ip[rows + 1] - s).astype(np.int64)
        cum = np.concatenate([[0], np.cumsum(ln)])
        flat = np.repeat(s - cum[:-1], ln) + np.arange(cum[-1])
        M = sp.csr_matrix((np.asarray(rd.values[flat]).astype(np.float32),
                           np.asarray(rd.indices[flat]).astype(np.int32),
                           cum.astype(np.int64)), shape=(len(rows), D_SAE))
        return np.asarray(M @ W, dtype=np.float32)[:, keep]

    for i in range(n):
        r = np.sort(order[st[i]:en[i]])
        g = np.random.default_rng(int(pool_ids[i]))
        seq = r // SEQ_LEN
        ev, od = r[seq % 2 == 0], r[seq % 2 == 1]
        if len(ev) > ROWS_PER_PARITY:
            ev = np.sort(g.choice(ev, ROWS_PER_PARITY, replace=False))
        if len(od) > ROWS_PER_PARITY:
            od = np.sort(g.choice(od, ROWS_PER_PARITY, replace=False))
        A = decode(ev).astype(np.float64) if len(ev) >= 3 else np.zeros((0, d))
        B = decode(od).astype(np.float64) if len(od) >= 3 else np.zeros((0, d))
        M = np.concatenate([A, B]) if len(A) + len(B) else np.zeros((0, d))
        nrow[i] = len(M)
        assert np.isfinite(M).all(), f"non-finite decoded rows, type {pool_ids[i]} L{layer}"
        if len(M) < 5:
            Q[i] = np.linalg.qr(np.random.default_rng(i).normal(size=(d, 2)))[0].T
            sh[i] = 90.0
            continue
        q, w = top2_plane(M)
        Q[i] = q
        f2[i] = float(w[:2].sum() / max(w.sum(), 1e-30))
        if len(A) >= 5 and len(B) >= 5:
            Qa[i] = top2_plane(A)[0]; Qb[i] = top2_plane(B)[0]
            sh[i] = float(princ_angles(Qa[i], Qb[i])[1])
        else:
            Qa[i] = Qb[i] = q; sh[i] = 90.0
        if i in want:
            clouds[i] = M.astype(np.float32)
        if log_every and (i + 1) % log_every == 0:
            print(f"    L{layer} planes {i+1}/{n}  {time.time()-t0:.0f}s", flush=True)
    return Q, Qa, Qb, f2, sh, nrow, clouds, float(time.time() - t0)


def l5_channel2(root, tok, pool_ids, counts_pool, drop_coord):
    """Channel 2 at L5 via 42's delta stream, lag+1 only."""
    s42.LAGS = (1,)
    n_types = len(pool_ids)
    lut = np.full(int(tok.max()) + 1, -1, dtype=np.int64)
    lut[pool_ids] = np.arange(n_types)
    own, parity, per_lag = s42.pair_index(tok, len(tok), lut)
    pl = per_lag[1]
    Cl = np.zeros((2, n_types, n_types), dtype=np.int64)
    for p in (0, 1):
        m = pl["succ_ok"] & (parity == p)
        np.add.at(Cl[p], (pl["prev"][m], own[m]), 1)
        del m
    rs = Cl.sum(axis=2)
    included = (rs[0] >= s42.MIN_PAIRS_PER_HALF) & (rs[1] >= s42.MIN_PAIRS_PER_HALF)
    pred_par = {1: np.where(pl["succ_ok"], pl["prev"] + parity.astype(np.int32) * n_types,
                            np.int32(-1)).astype(np.int32)}
    del parity, per_lag, pl
    gc.collect()
    TM, S, seen = s42.stream_deltas(root, 5, own, pred_par, len(tok), n_types, D_SAE)
    assert np.array_equal(seen, counts_pool), "streamed counts != corpus counts (L5)"
    TM /= np.maximum(seen, 1)[:, None]
    TM[seen == 0] = 0.0
    sae = load_sae(5, "big", token=os.environ.get("HF_TOKEN"))
    W = get_decoder(sae); del sae
    for p in (0, 1):
        base = p * n_types
        for lo in range(0, n_types, 512):
            hi = min(lo + 512, n_types)
            S[1][base + lo:base + hi] -= Cl[p][lo:hi].astype(np.float64) @ TM
    del TM, Cl
    gc.collect()
    for p in (0, 1):
        S[1][p * n_types:(p + 1) * n_types] /= np.maximum(rs[p], 1)[:, None]
    h0 = s42.project(S[1][:n_types], W).astype(np.float64)
    h1 = s42.project(S[1][n_types:], W).astype(np.float64)
    del S, W
    gc.collect()
    w0 = rs[0].astype(np.float64)[:, None]; w1 = rs[1].astype(np.float64)[:, None]
    full = (w0 * h0 + w1 * h1) / np.maximum(w0 + w1, 1)
    _, sd, dropped, shares = s42.outlier_drop_rule(h0, h1, included)
    kp = np.ones(D_MODEL, dtype=bool); kp[drop_coord] = False
    assert np.isfinite(full).all(), "non-finite L5 deltas"
    return (np.ascontiguousarray(full[:, kp].astype(np.float32)).astype(np.float64),
            dict(drop_computed=dropped, drop_shares=[round(float(x), 4) for x in shares],
                 drop_frozen=[int(drop_coord)],
                 agrees=bool(dropped == [int(drop_coord)]),
                 n_included=int(included.sum())))


def ov_images(layer):
    """(4, 1152, 256) orthonormal bases of each query head's o_proj column space."""
    from safetensors.torch import safe_open
    with safe_open(str(REPO / MODEL_ST), "pt") as f:
        o = f.get_tensor(f"model.layers.{layer}.self_attn.o_proj.weight").float().numpy()
    # o_proj.weight is [hidden_out, heads*head_dim]; image = column space per head slice
    return np.stack([np.linalg.qr(o[:, h * 256:(h + 1) * 256])[0] for h in range(4)])


def ch3_retention(B, imgs, keep):
    """Mean over heads of the fraction of plane B (2 x d_kept) retained by the OV image."""
    Bf = np.zeros((2, D_MODEL)); Bf[:, keep] = B
    Bf /= np.maximum(np.linalg.norm(Bf, axis=1, keepdims=True), 1e-30)
    return float(np.mean([float(((imgs[h].T @ Bf.T) ** 2).sum() / 2.0) for h in range(4)]))


# ---------------------------------------------------------------------------
# nulls -- cluster-like / random / frequency-matched cells, our own builder so the
# conditional rho array (needed for the L1 exact-1.0 rate) is available.
# ---------------------------------------------------------------------------
def build_cell(Xn, C1, C2, counts, k, seed, kind, n_draws):
    rng = np.random.default_rng(seed)
    gaps, rc, ones = [], [], 0
    n = len(C1)
    for _ in range(n_draws):
        if kind == "clusterlike":
            s = int(rng.integers(n))
            sim = Xn @ Xn[s]; sim[s] = 2.0
            idx = np.argpartition(-sim, k)[:k]
        elif kind == "random":
            idx = rng.choice(n, size=k, replace=False)
        else:
            idx = s42.frequency_matched_resample(rng.choice(n, size=k, replace=False),
                                                 counts, rng, tol=s42.FREQ_MATCH_TOL)
            if len(idx) != k:
                continue
        g, r = s42.cell_stats(C1[idx], C2[idx])
        if g is None:
            continue
        gaps.append(g)
        if r is not None and g < s42.GAP_RATIO_CLOSURE:
            rc.append(r)
            if r >= 1.0 - 1e-12:
                ones += 1
    rc = np.asarray(rc)
    return dict(kind=kind, k=int(k), seed=int(seed), n_draws=int(n_draws),
                n_conditional=int(len(rc)),
                closure_rate=float(np.mean(np.asarray(gaps) < s42.GAP_RATIO_CLOSURE))
                if gaps else None,
                p99=float(np.percentile(rc, 99.0, method="higher")) if len(rc) else None,
                exact_one_rate=float(ones / len(rc)) if len(rc) else None,
                rho_median=float(np.median(rc)) if len(rc) else None)


class Cells:
    def __init__(self, Xn, C1, C2, counts, n_cl, seeds):
        self.a = (Xn, C1, C2, counts); self.n_cl = n_cl; self.seeds = seeds; self.c = {}

    def band(self, layer, k):
        key = (layer, k)
        if key not in self.c:
            Xn, C1, C2, counts = self.a
            cl = [build_cell(Xn, C1, C2, counts, k, 7_000_000 + 1000 * layer + 97 * s + k,
                             "clusterlike", self.n_cl) for s in self.seeds]
            fq = build_cell(Xn, C1, C2, counts, k, 7_200_000 + 1000 * layer + k,
                            "freq", 2000)
            rnd = build_cell(Xn, C1, C2, counts, k, 7_100_000 + 1000 * layer + k,
                             "random", 2000)
            p = [c["p99"] for c in cl if c["p99"] is not None]
            self.c[key] = dict(k=int(k), clusterlike=cl, freq=fq, random=rnd,
                               p99_max=max(p) if p else None, p99_min=min(p) if p else None,
                               exact_one_rate=float(np.mean([c["exact_one_rate"] for c in cl
                                                             if c["exact_one_rate"] is not None]))
                               if any(c["exact_one_rate"] is not None for c in cl) else None,
                               freq_p99=fq["p99"], saturated=bool(p and max(p) >= 1.0 - 1e-12))
        return self.c[key]


def verdict_of(members, C1, C2, cells, layer, seed=0):
    """R14 with the L1 saturation rule and the L2 seed band. Closure FIRST (M1)."""
    members = np.asarray(sorted(int(x) for x in members))
    k = len(members)
    g, rho = s42.cell_stats(C1[members], C2[members], seed)
    closure = bool(g is not None and g < s42.GAP_RATIO_CLOSURE)
    row = dict(k=int(k), gap_ratio=None if g is None else round(float(g), 4),
               rho=None if rho is None else round(float(rho), 6), closure_pass=closure)
    if not closure:
        row["verdict"] = "FAIL"; row["reason"] = "closure"; return row
    if k < K_MIN or rho is None:
        row["verdict"] = "NOT ADJUDICABLE"; row["reason"] = "k<9 or degenerate ch2"; return row
    cb = cells.band(layer, k)
    cbl = cells.band(layer, k - 1)
    row.update(p99_max=cb["p99_max"], p99_min=cb["p99_min"], freq_p99=cb["freq_p99"],
               saturated=cb["saturated"], exact_one_rate=cb["exact_one_rate"],
               p99_max_loo=cbl["p99_max"], freq_p99_loo=cbl["freq_p99"],
               saturated_loo=cbl["saturated"])
    def order_ok(r, cell):
        if cell["saturated"]:
            if cell["exact_one_rate"] is not None and cell["exact_one_rate"] < 0.01:
                return (r >= 1.0 - 1e-12), False
            return False, True          # NOT ADJUDICABLE
        return (cell["p99_max"] is not None and r > cell["p99_max"]), False
    ok, na = order_ok(rho, cb)
    if na:
        row["verdict"] = "NOT ADJUDICABLE"; row["reason"] = "saturated cell, L1"; return row
    marginal = bool(not ok and cb["p99_min"] is not None and rho > cb["p99_min"])
    freq_ok = bool(cb["freq_p99"] is None or rho > cb["freq_p99"])
    loo_g, loo_r, loo_ok_g, loo_ok_r = [], [], True, True
    for i in range(k):
        sub = np.delete(members, i)
        gi, ri = s42.cell_stats(C1[sub], C2[sub], seed)
        loo_g.append(gi); loo_r.append(ri)
        if not (gi is not None and gi < s42.GAP_RATIO_CLOSURE):
            loo_ok_g = False
        oki, nai = (False, False) if ri is None else order_ok(ri, cbl)
        if not oki:
            loo_ok_r = False
    row.update(loo_closure_pass=bool(loo_ok_g), loo_rho_pass=bool(loo_ok_r),
               loo_gap_max=None if any(x is None for x in loo_g) else round(max(loo_g), 4),
               loo_rho_min=None if any(x is None for x in loo_r) else round(min(loo_r), 6),
               freq_arm_pass=freq_ok, order_pass=bool(ok), marginal_band=marginal)
    if ok and loo_ok_g and loo_ok_r and freq_ok:
        row["verdict"] = "PASS"
    elif (marginal or ok) and loo_ok_g:
        row["verdict"] = "MARGINAL"
    else:
        row["verdict"] = "FAIL"
    return row


def clusters_at(Z, tau, n, kmin):
    lab = sch.fcluster(Z, tau, "distance")
    out = []
    for c in np.unique(lab):
        m = np.flatnonzero(lab == c)
        if len(m) >= kmin:
            out.append(m)
    return out, lab


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="cache/sparse_big")
    ap.add_argument("--out", default="results/phase5_occurrence_plane_screen.json")
    ap.add_argument("--layers", default="6,7,5")
    ap.add_argument("--n-clusterlike", type=int, default=N_CLUSTERLIKE)
    ap.add_argument("--n-perm", type=int, default=N_PERM_SEARCH)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    t_start = time.time()
    layers = [int(x) for x in a.layers.split(",")]
    n_cl = 400 if a.smoke else int(a.n_clusterlike)
    n_perm = 2 if a.smoke else int(a.n_perm)

    # ---- DECLARATION STUB, before ANY read ----
    av = dict(root=a.root, out=a.out, layers=layers, taus=list(TAUS),
              rows_per_parity=ROWS_PER_PARITY, n_clusterlike=n_cl, n_perm=n_perm,
              smoke=bool(a.smoke), k_min=K_MIN, dedup=DEDUP_PRIMARY,
              ring_angle_min=RING_ANGLE_MIN)
    stub = dict(status="DECLARATION STUB -- rules frozen, no results yet",
                script="scripts/49_occurrence_plane_screen.py",
                declared_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
                declared_rules=declared_rules(av))
    Path(a.out).write_text(json.dumps(stub, indent=2))
    print("=== DECLARATION STUB WRITTEN (before any cache/npz read) ===", flush=True)
    out = dict(stub); out["status"] = "RUNNING"

    # ---- pool ----
    tok, cnt, pool_ids, _ = s42.build_pool(a.root)
    n_types = len(pool_ids)
    counts_pool = cnt[pool_ids].astype(np.int64)
    pos = {int(t): i for i, t in enumerate(pool_ids)}
    from transformers import AutoTokenizer
    from gfdisc.env import MODEL_NAME
    tkz = AutoTokenizer.from_pretrained(MODEL_NAME, token=os.environ.get("HF_TOKEN"))
    strings = [(tkz.convert_ids_to_tokens(int(t)) or tkz.decode([int(t)])).replace("▁", " ")
               for t in pool_ids]
    strat = strata(strings, counts_pool)
    months = np.asarray(sorted(pos[t] for t in MONTH_TOKEN_IDS))
    tmj = json.loads((REPO / "results/token_map.json").read_text())
    weekdays = np.asarray(sorted({pos[int(v[0])] for v in tmj["days"].values()
                                  if int(v[0]) in pos}))
    print(f"pool {n_types} types; months {len(months)}; weekdays {len(weekdays)}; "
          f"strata {len(strat)}", flush=True)
    out["pool"] = dict(n_types=n_types, n_cache_rows=int(len(tok)),
                       n_strata=len(strat), n_months=int(len(months)),
                       n_weekdays=int(len(weekdays)))

    zr = np.load(REPO / FULLN_RAW)
    assert np.array_equal(zr["pool_ids"].astype(np.int64), pool_ids.astype(np.int64))
    zd = np.load(REPO / ROBUST_DELTAS)
    assert np.array_equal(zd["pool_ids"].astype(np.int64), pool_ids.astype(np.int64))

    out["layers"] = {}
    for L in layers:
        tL = time.time()
        print(f"\n================ layer {L} ================", flush=True)
        drop = s42.R10_DROP[L]
        keep = np.ones(D_MODEL, dtype=bool); keep[drop] = False
        C1 = np.ascontiguousarray(zr[f"R_L{L}"][:, keep].astype(np.float64))
        note = {}
        if L in (6, 7):
            assert [int(x) for x in zd[f"L{L}_lag1_dropped_coords"]] == [drop]
            C2 = zd[f"L{L}_lag1"].astype(np.float64)
        else:
            C2, note = l5_channel2(a.root, tok, pool_ids, counts_pool, drop)
            print(f"  [L5 channel 2] {note}", flush=True)
        assert C1.shape == C2.shape == (n_types, D_MODEL - 1)

        sae = load_sae(L, "big", token=os.environ.get("HF_TOKEN"))
        W = get_decoder(sae).astype(np.float32); del sae
        rng0 = np.random.default_rng(490_000 + L)
        hi_cnt = np.flatnonzero(counts_pool >= 400)
        carriers_pool = (rng0.choice(np.setdiff1d(hi_cnt, np.concatenate([months, weekdays])),
                                     min(N_PLANT_CARRIERS, len(hi_cnt) - 20), replace=False)
                         if L == 6 else ())
        Q, Qa, Qb, f2, sh, nrow, clouds, tstream = stream_planes(
            a.root, L, tok, pool_ids, W, keep, carriers_pool)
        del W; gc.collect()
        print(f"  planes done {tstream:.0f}s; n rows median {int(np.median(nrow))}; "
              f"n=200 for {int((nrow==200).sum())}/{n_types}", flush=True)

        t0 = time.time()
        TH = theta2_matrix(Q)
        print(f"  theta2 matrix {time.time()-t0:.0f}s", flush=True)

        # --- R7: recompute the ladder's null percentiles on the full pool ---
        rr = TH[np.ix_(hi_cnt, hi_cnt)]
        iu = np.triu_indices(len(hi_cnt), 1)
        rrv = rr[iu]
        mm = TH[np.ix_(months, months)][np.triu_indices(len(months), 1)]
        ww = TH[np.ix_(weekdays, weekdays)][np.triu_indices(len(weekdays), 1)]
        pct = dict(count_matched_p001=float(np.percentile(rrv, 0.01)),
                   count_matched_p01=float(np.percentile(rrv, 0.1)),
                   count_matched_p1=float(np.percentile(rrv, 1.0)),
                   count_matched_p5=float(np.percentile(rrv, 5.0)),
                   count_matched_median=float(np.median(rrv)),
                   months_median=float(np.median(mm)), months_max=float(np.max(mm)),
                   weekdays_median=float(np.median(ww)),
                   floor_months=float(np.median(sh[months])),
                   floor_weekdays=float(np.median(sh[weekdays])),
                   floor_count_matched=float(np.median(sh[hi_cnt])),
                   f2_months=float(np.median(f2[months])),
                   f2_count_matched=float(np.median(f2[hi_cnt])))
        print("  ladder percentiles:", {k: round(v, 2) for k, v in pct.items()}, flush=True)

        Z = sch.linkage(squareform(TH, checks=False), method="average")
        C1c = C1 - C1.mean(0, keepdims=True)
        Xn = C1c / np.maximum(np.linalg.norm(C1c, axis=1, keepdims=True), 1e-30)
        cells = Cells(Xn, C1, C2, counts_pool, n_cl, (0, 1, 2))
        imgs = ov_images(L)

        lay = dict(stream_seconds=round(tstream, 1), percentiles=pct, l5_note=note,
                   n_rows_median=int(np.median(nrow)), n_rows_eq200=int((nrow == 200).sum()),
                   taus={})
        for tau in TAUS:
            cl, lab = clusters_at(Z, tau, n_types, K_MIN)
            all_cl, _ = clusters_at(Z, tau, n_types, 2)
            props = []
            for m in cl:
                dd = {f"{t:.2f}": [int(x) for x in dedup(m, C1c, counts_pool, t)]
                      for t in DEDUP_LADDER}
                kept = np.asarray(dd[f"{DEDUP_PRIMARY:.2f}"])
                if len(kept) < K_MIN:
                    props.append(dict(k_raw=int(len(m)), k_dedup=int(len(kept)),
                                      verdict="NOT ADJUDICABLE",
                                      reason="k<9 after R15 dedup",
                                      dedup_ladder={t: len(v) for t, v in dd.items()},
                                      tokens=[strings[i] for i in m]))
                    continue
                B, ret = shared_plane(Q, kept)
                ring = pca_2d(C1[kept])[2][:2]
                pa = princ_angles(B, ring)
                v = verdict_of(kept, C1, C2, cells, L)
                v.update(k_raw=int(len(m)), k_dedup=int(len(kept)),
                         dedup_ladder={t: len(vv) for t, vv in dd.items()},
                         tokens=[strings[i] for i in kept],
                         token_ids=[int(pool_ids[i]) for i in kept],
                         counts=[int(counts_pool[i]) for i in kept],
                         n_months=int(len(set(kept.tolist()) & set(months.tolist()))),
                         n_weekdays=int(len(set(kept.tolist()) & set(weekdays.tolist()))),
                         within_theta2_median=float(np.median(
                             TH[np.ix_(kept, kept)][np.triu_indices(len(kept), 1)])),
                         shared_plane_ring_angles=[round(float(x), 1) for x in pa],
                         ring_conjunct_pass=bool(pa[0] > RING_ANGLE_MIN),
                         shared_plane_coherence=round(ret, 3),
                         channel3_ov_retention=round(ch3_retention(B, imgs, keep), 4),
                         exposed_in_619=sorted(set(strings[i] for i in kept) & EXPOSED_619))
                if v["verdict"] == "PASS" and not v["ring_conjunct_pass"]:
                    v["verdict"] = "FAIL"; v["reason"] = "ring conjunct (2.2)"
                props.append(v)
            best_month = max((len(set(m.tolist()) & set(months.tolist())) for m in all_cl),
                             default=0)
            month_cl = [m for m in all_cl if set(months.tolist()) <= set(m.tolist())]
            tight = sorted(all_cl, key=lambda m: np.median(
                TH[np.ix_(m, m)][np.triu_indices(len(m), 1)]))
            m_rank = next((i + 1 for i, m in enumerate(tight)
                           if len(set(m.tolist()) & set(months.tolist())) >= 9), None)
            wd_cl = max((len(set(m.tolist()) & set(weekdays.tolist())) for m in all_cl),
                        default=0)
            lay["taus"][f"{tau:.0f}"] = dict(
                n_clusters_ge2=len(all_cl), n_clusters_ge9=len(cl),
                n_adjudicable=int(sum(1 for p in props
                                      if p["verdict"] not in ("NOT ADJUDICABLE",))),
                verdicts={v: int(sum(1 for p in props if p["verdict"] == v))
                          for v in ("PASS", "MARGINAL", "FAIL", "NOT ADJUDICABLE")},
                positive_control=dict(
                    max_months_in_any_cluster=int(best_month),
                    strict12_contained=bool(month_cl),
                    strict12_cluster_size=int(min((len(m) for m in month_cl), default=0)),
                    month_cluster_tightness_rank=m_rank,
                    n_clusters_total=len(all_cl)),
                negative_control=dict(max_weekdays_in_any_cluster=int(wd_cl)),
                proposals=props)
            print(f"  tau={tau:.0f}: clusters>=2 {len(all_cl)}  >=9 {len(cl)}  "
                  f"verdicts {lay['taus'][f'{tau:.0f}']['verdicts']}  "
                  f"months-in-a-cluster {best_month} strict12 {bool(month_cl)}", flush=True)

        # --- R18 blind positive control: strict-12 through the tier at this layer ---
        lay["positive_control_strict12"] = verdict_of(months, C1, C2, cells, L)
        lay["negative_control_weekdays"] = verdict_of(weekdays, C1, C2, cells, L)
        print(f"  strict-12 tier verdict: {lay['positive_control_strict12']['verdict']} "
              f"(gap {lay['positive_control_strict12'].get('gap_ratio')}, "
              f"rho {lay['positive_control_strict12'].get('rho')})", flush=True)

        # --- R13 permuted-search null ---
        t0 = time.time()
        perm_counts = []
        for s in range(n_perm):
            rng = np.random.default_rng(4900 + 13 * s + L)
            p = stratified_perm(strat, n_types, rng)
            THp = TH[np.ix_(p, p)]
            Zp = sch.linkage(squareform(THp, checks=False), method="average")
            npass = {}
            for tau in TAUS:
                cl, _ = clusters_at(Zp, tau, n_types, K_MIN)
                n_ok = 0
                for m in cl:
                    kept = dedup(m, C1c, counts_pool, DEDUP_PRIMARY)
                    if len(kept) < K_MIN:
                        continue
                    v = verdict_of(kept, C1, C2, cells, L)
                    if v["verdict"] == "PASS":
                        B, _ = shared_plane(Q, kept)
                        if princ_angles(B, pca_2d(C1[kept])[2][:2])[0] > RING_ANGLE_MIN:
                            n_ok += 1
                npass[f"{tau:.0f}"] = n_ok
            perm_counts.append(npass)
            print(f"    perm {s+1}/{n_perm} passers {npass} ({time.time()-t0:.0f}s)",
                  flush=True)
        lay["permuted_search"] = dict(
            n_permutations=n_perm, per_permutation=perm_counts,
            mean_passers={f"{t:.0f}": float(np.mean([c[f'{t:.0f}'] for c in perm_counts]))
                          for t in TAUS} if perm_counts else {},
            seconds=round(time.time() - t0, 1))

        # --- channel-2 row permutation (43c-style member specificity) ---
        ch2 = []
        for s in range(min(5, n_perm)):
            rng = np.random.default_rng(4950 + 7 * s + L)
            p = stratified_perm(strat, n_types, rng)
            C2p = C2[p]
            n_ok = 0
            for tau in TAUS:
                for pr in lay["taus"][f"{tau:.0f}"]["proposals"]:
                    if pr["verdict"] != "PASS":
                        continue
                    mem = np.asarray([pos[t] for t in pr["token_ids"]])
                    if verdict_of(mem, C1, C2p, cells, L)["verdict"] == "PASS":
                        n_ok += 1
            ch2.append(n_ok)
        lay["channel2_row_permutation"] = dict(n=len(ch2), passers=ch2)

        # --- plant (L6 only), prereg 2.2 ---
        if L == 6 and clouds:
            lay["plant"] = run_plant(Q, TH, clouds, np.asarray(sorted(clouds)),
                                     pct["count_matched_p01"])
            print(f"  plant: {lay['plant']['summary']}", flush=True)

        lay["layer_seconds"] = round(time.time() - tL, 1)
        out["layers"][str(L)] = lay
        del Q, Qa, Qb, TH, Z, C1, C2, C1c, Xn, clouds
        gc.collect()
        Path(a.out).write_text(json.dumps(out, indent=2, default=float))

    out["status"] = "COMPLETE"
    out["wall_clock_seconds"] = round(time.time() - t_start, 1)
    Path(a.out).write_text(json.dumps(out, indent=2, default=float))
    print(f"\nDONE in {out['wall_clock_seconds']}s -> {a.out}", flush=True)


def run_plant(Q, TH, clouds, cand, p01):
    """REPLACE-not-add plant; raw and null-calibrated delta80."""
    d = Q.shape[2]
    rows = []
    rng = np.random.default_rng(4977)
    A32 = np.ascontiguousarray(Q[:, 0, :], dtype=np.float32)
    B32 = np.ascontiguousarray(Q[:, 1, :], dtype=np.float32)
    for delta in PLANT_DELTAS:
        raw_hits, cal_hits, meds, f2keep = 0, 0, [], []
        for _ in range(N_PLANTS):
            sel = rng.choice(cand, PLANT_K, replace=False)
            Pc = np.linalg.qr(rng.normal(size=(d, 2)))[0]
            Qp = {}
            for s in sel:
                M = clouds[int(s)].astype(np.float64)
                mu = M.mean(0); A = M - mu
                V = Q[s].T
                T = A @ V
                U = rng.normal(size=(d, 2)); U -= Pc @ (Pc.T @ U)
                U = np.linalg.qr(U)[0]
                P2 = np.linalg.qr(np.cos(np.radians(delta)) * Pc
                                  + np.sin(np.radians(delta)) * U)[0]
                An = (A - T @ V.T) + T @ P2.T
                q, w = top2_plane(An + mu)
                Qp[int(s)] = q
                f2keep.append(float(w[:2].sum() / max(w.sum(), 1e-30)))
            # patched rows of the angle matrix (only the planted types change)
            Aq = A32.copy(); Bq = B32.copy()
            for s, q in Qp.items():
                Aq[s] = q[0]; Bq[s] = q[1]
            si = np.asarray(sorted(Qp))
            aa = Aq[si] @ Aq.T; ab = Aq[si] @ Bq.T
            ba = Bq[si] @ Aq.T; bb = Bq[si] @ Bq.T
            T2 = (aa.astype(np.float64) ** 2 + ab.astype(np.float64) ** 2
                  + ba.astype(np.float64) ** 2 + bb.astype(np.float64) ** 2)
            Dt = (aa.astype(np.float64) * bb.astype(np.float64)
                  - ab.astype(np.float64) * ba.astype(np.float64)) ** 2
            s2 = (T2 - np.sqrt(np.maximum(T2 * T2 - 4 * Dt, 0))) / 2
            row = np.degrees(np.arccos(np.sqrt(np.clip(s2, 0, 1))))
            for j, s in enumerate(si):
                row[j, s] = 1e9
            sub = row[:, si]
            iu = np.triu_indices(len(si), 1)
            med = float(np.median(sub[iu]))
            meds.append(med)
            nn = np.argsort(row, 1)[:, :10]
            pur = float(np.mean([[int(x) in set(si.tolist()) for x in nn[j]]
                                 for j in range(len(si))]))
            raw_hits += int(pur >= 0.8)
            cal_hits += int(med < p01)
        rows.append(dict(delta_deg=delta,
                         raw_recall=raw_hits / N_PLANTS,
                         calibrated_recall=cal_hits / N_PLANTS,
                         within_theta2_median=round(float(np.median(meds)), 2),
                         analytic_arccos_cos2=round(float(np.degrees(np.arccos(
                             np.cos(np.radians(delta)) ** 2))), 2),
                         top2_fraction=round(float(np.median(f2keep)), 3)))
        print(f"    plant delta={delta:2d} raw {rows[-1]['raw_recall']:.2f} "
              f"cal {rows[-1]['calibrated_recall']:.2f} th2 "
              f"{rows[-1]['within_theta2_median']:.1f}", flush=True)

    def d80(key):
        prev = None
        for r in rows:
            if r[key] < 0.80:
                if prev is None:
                    return None
                return prev["delta_deg"]
            prev = r
        return rows[-1]["delta_deg"]
    return dict(n_plants_per_delta=N_PLANTS, k=PLANT_K, carrier_pool=len(cand),
                null_p01_deg=round(float(p01), 2), rows=rows,
                delta80_raw=d80("raw_recall"), delta80_calibrated=d80("calibrated_recall"),
                summary=(f"delta80 raw {d80('raw_recall')} deg, "
                         f"null-calibrated {d80('calibrated_recall')} deg "
                         f"(null p0.1 {p01:.1f} deg)"))


if __name__ == "__main__":
    main()
