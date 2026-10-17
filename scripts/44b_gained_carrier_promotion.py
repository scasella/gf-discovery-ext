"""Phase 4 section 2.3 + Addendum J -- gained-metric carrier populations, PROMOTION stage.

Frozen design: docs/phase4_prereg_2026-08-22.md section 2.3 and docs/phase4_addendum_J_2026-08-22.md
(J1-J5). This is a NEW script; scripts/44_gained_carrier_populations.py (the kill test) is not
touched and nothing is imported from it -- the shared helpers below are COPIED, and any drift
between the two files is a bug.

Funding: the section 2.3 kill test PASSED (results/phase4_gained_carriers.json, declared
2026-08-22T14:59:13): 12/12 strict months in the top-24 of 63 of 1,130 planes at L6 and of 10
planes at L7, frequency-pseudo max-over-planes <= 2 (L6) / <= 4 (L7).

DECLARED BEFORE ANY READ (stub written to --out; declared_at is the audit anchor)
--------------------------------------------------------------------------------
D1  POPULATION. scripts/22 build_pool: token types with corpus count >= 100 in
    cache/sparse_big/layer_0/tokens.i32 -- 3,795 types. Asserted equal to the pool_ids of
    results/phase3_token_type_mean_resid_fulln.npz.

D2  HALVES (R9). By SEQUENCE id, never row parity: cache row r belongs to sequence r // 255;
    half A = even sequence index, half B = odd. 15,696 sequences -> 7,848 per half.

D3  ARM, and the ONE deviation from section 2.3's wording -- DEVIATION J6. Section 2.3
    adjudicates closure in RAW residual coordinates. The full-n raw type means
    (results/phase3_token_type_mean_resid_fulln.npz) exist only POOLED, and scripts/35b built
    them with a full-cache MPS forward pass (26,655.9 s). R12 forbids running that
    concurrently with the cache stream of scripts/42, and the orchestrator's instruction
    forbids launching MPS at all. Therefore BOTH the gained proposal and the closure
    adjudication run on the SAE mean-code arm: per-half per-type mean SAE codes streamed from
    the CSR cache and projected through W_dec (scripts/22 type_means -> project; b_dec is a
    constant per-type-independent shift and is removed by the intercept of D5's deflation).
    This is the arm in which the months' own Leiden cluster closes at L6 (gap_ratio 1.6588 /
    p_closed 0.00125 / p_bg 0.00399) and L7 (1.6204 / 0.00115 / 0.00200) and FAILS at L8
    (3.6356 / 0.746 / 0.351) -- i.e. the arm whose positive and negative controls are both
    already measured. Declared consequence: this run cannot speak about the raw residual.

D4  COORDINATES, fixed by J5 exactly as run in the kill test: R10 drops residual coordinate
    1038 BEFORE the RMS is taken (and its w_in entry with it); x_hat = RMSNorm(M_L) * (1 +
    w_in[L]) with eps = 1e-6 (scripts/40 rms_norm_x), M_L the half's decoded mean matrix;
    the DEFLATED norm is the energy-fraction denominator. The R[L-1] variant is not run here.

D5  DEFLATION, per half, on that half's own counts: one least-squares solve against
    [1, log10(count_half), 6 token-class indicators (scripts/34 class_features), ||x_hat||].

D6  PROPOSER (J1: every plane, blind). Top-36 SVD of the deflated half-A cloud; all
    C(36,2) = 630 coordinate planes plus 500 seeded random 2-planes inside the top-36
    subspace. 1,130 planes per layer. The half-B plane P_B is the SAME construction applied
    to half B's own top-36 basis (coordinate pair (i,j), or the same random Q), so the
    principal angle P_A vs P_B measures whether the proposer finds the same plane twice.

D7  SELECTION (J1). S_A = types whose energy fraction e_t = ||z_t||^2/||x_hat_t||^2 in P_A
    exceeds the 99th percentile of the plane's frequency-pseudo e-distribution, capped at the
    top-60 by e. DECLARED IDENTITY, fixed before the run: every pseudo-population in this
    campaign is a RELABELLING of the same pool rows (frequency-pseudo = nearest-count row
    substitution; class-block permutation = within-class identity permutation), and a
    relabelling leaves the multiset of e values in a fixed plane invariant. The plane's
    frequency-pseudo e-distribution is therefore identically the plane's own pool
    e-distribution, and the threshold is its 99th percentile -> |S_A| = 38 for every plane.
    S_B is the same rule in P_B at the size |S_A|.

D8  CONFIRMATION (J2). Jaccard(S_A, S_B) against (i) the two-independent-draw null (2,000
    draws of two random subsets of the same sizes) and (ii) the class-block-permutation null
    (200 within-token-class permutations of type identity applied to S_B) -- (ii) is the
    verdict-bearing one under J3, because it is the null that can fail a coherent cluster.
    Closure of S = S_A ∩ S_B on HALF B: scripts/22 screen() (top-2 PCA of the centred member
    block, Kasa circle, max-gap) with p_bg against 500 matched random same-size type sets
    drawn from the same half-B cloud, cached per size.

D9  CANDIDATE RULE (J4), frozen: p_bg < 0.002 AND class-permuted Jaccard p <= 0.01 (above the
    99th percentile) AND closure survives every leave-one-out. CALENDAR label (J1): a plane
    whose top-24 in P_A contains >= 6 strict-12 months or weekdays. Calendar planes are the
    positive control and are reported separately; they are never counted as discoveries.

D10 FALSE-PROMOTION RATE (J3/J4). (a) class-permuted: the identical promotion path with S_B
    relabelled, 50 permutations x every plane (subsampled from 200 for wall-clock, declared
    here), closure screened only where the Jaccard gate is cleared, giving the per-plane rate
    and expected false promotions = 1,130 x rate. (b) cluster-like: for every represented
    size, 200 size-matched cluster-like sets (random seed + k-1 nearest by centred cosine on
    half B, the section 2.1 construction) through the same closure test, giving the rate at
    which a COHERENT cluster of that size clears p_bg < 0.002. The frequency-pseudo null is
    reported and is not verdict-bearing (J3).

D11 SEEDS. Random planes 442000 + 100*L; background 443000 + 100*L + k; class permutations
    444000 + 100*L; cluster-like 445000 + 100*L + k; two-draw Jaccard null 446000 + 100*L + k.

D12 No promotion beyond this: candidates with |S| >= 9 are handed to the section 2.1
    two-channel tier when scripts/42 completes. No MPS, no forward pass, no edit of 44.
"""
import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

COUNT_FLOOR = 100
LAYERS = (6, 7)
D_SVD = 36
N_RANDOM_PLANES = 500
TOP_N = 24
CAP_S = 60                      # J1 cap (= scripts/22 SCREEN_MAX)
PCT_E = 99.0                    # J1 percentile
SCREEN_MIN = 5                  # scripts/22
N_BACKGROUND_DRAWS = 500        # scripts/22
N_PERM_BACKGROUND = 2000        # scripts/22
N_PERM_SCREEN = 20000
N_CLASSPERM = 200               # Jaccard null draws
N_CLASSPERM_CONJ = 50           # D10(a) subsample for the full conjunction
N_CLUSTERLIKE = 200             # D10(b)
N_TWODRAW = 2000
P_BG_MAX = 0.002                # J4
JACCARD_P_MAX = 0.01            # J4 (99th percentile)
CAL_MIN_IN_TOP24 = 6            # J1 CALENDAR label
RMS_EPS = 1e-6
DROP_COORDS = (1038,)
CHUNK = 200_000
SEQ_LEN = 255
SNAP = ("cache/hf/hub/models--google--gemma-3-1b-pt/snapshots/"
        "fcf18a2a879aab110ca39f8bffbccd5d49d8eb29/model.safetensors")
MONTH_NAMES = ["January", "February", "March", "April", "May", "June", "July",
               "August", "September", "October", "November", "December"]


# ---- copied helpers (scripts/34, scripts/40, scripts/22, gfdisc.fits) ------
def type_string(tokenizer, tid):
    s = tokenizer.convert_ids_to_tokens(int(tid))
    if s is None:
        s = tokenizer.decode([int(tid)])
    return s.replace("▁", " ")


def class_features(s):
    alphas = [c for c in s if c.isalpha()]
    all_caps = bool(alphas) and all(c.isupper() for c in alphas)
    capitalized = bool(alphas) and alphas[0].isupper() and not all_caps
    return [s.startswith(" "), capitalized, all_caps,
            any(c.isdigit() for c in s),
            not any(c.isalnum() for c in s),
            any(ord(c) > 127 for c in s)]


def deflate(M, D):
    M = np.asarray(M, dtype=np.float64)
    coef, *_ = np.linalg.lstsq(D, M, rcond=None)
    return M - D @ coef


def rms_norm_x(R, w_in, eps=RMS_EPS):
    R = np.asarray(R, dtype=np.float64)
    inv = 1.0 / np.sqrt((R ** 2).mean(1, keepdims=True) + eps)
    return (R * inv) * w_in[None, :]


def member_hash(members):
    return hashlib.md5(",".join(str(int(x)) for x in sorted(members)).encode()).hexdigest()[:12]


def pca_2d(points):                                     # gfdisc.fits VERBATIM
    X = np.asarray(points, dtype=np.float64)
    X = X - X.mean(0, keepdims=True)
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    var = S ** 2
    return X @ Vt[:2].T, var / max(var.sum(), 1e-12), Vt


def fit_circle(xy):                                     # gfdisc.fits VERBATIM
    P = np.asarray(xy, dtype=np.float64)
    x, y = P[:, 0], P[:, 1]
    A = np.column_stack([2 * x, 2 * y, np.ones(len(x))])
    b = x ** 2 + y ** 2
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    cx, cy = sol[0], sol[1]
    r = np.sqrt(max(sol[2] + cx ** 2 + cy ** 2, 1e-12))
    d = np.hypot(x - cx, y - cy)
    return cx, cy, r, float(np.sqrt(np.mean((d - r) ** 2)) / max(r, 1e-12))


def circularity(xy, cx, cy, r):                         # gfdisc.fits VERBATIM
    P = np.asarray(xy, dtype=np.float64)
    spread = float(np.linalg.norm(P - P.mean(0), axis=1).mean())
    a = np.sort(np.arctan2(P[:, 1] - cy, P[:, 0] - cx))
    gaps = np.diff(np.concatenate([a, [a[0] + 2 * np.pi]]))
    k = len(P)
    return dict(radius_ratio=float(r / max(spread, 1e-12)),
                gap_ratio=float(gaps.max() / (2 * np.pi / k)),
                max_gap_deg=float(np.degrees(gaps.max())), spread=spread, radius=float(r))


def degenerate(points, rel_tol=1e-6):                   # gfdisc.fits VERBATIM
    X = np.asarray(points, dtype=np.float64)
    c = X - X.mean(0, keepdims=True)
    scale = np.linalg.norm(X, axis=1).mean()
    spread = np.linalg.norm(c, axis=1)
    if scale <= 0 or spread.max() <= rel_tol * max(scale, 1e-12):
        return True
    return bool((spread > rel_tol * scale).sum() < 3)


_NULL = {}


def gap_null(k, n_perm, seed=0):                        # scripts/22 VERBATIM
    key = (int(k), int(n_perm), int(seed))
    if key not in _NULL:
        rng = np.random.default_rng(seed)
        a = np.sort(rng.uniform(0, 2 * np.pi, size=(n_perm, k)), axis=1)
        _NULL[key] = np.diff(np.concatenate([a, a[:, :1] + 2 * np.pi], axis=1), axis=1).max(1)
    return _NULL[key]


def gap_p(k, max_gap_rad, n_perm, seed=0):              # scripts/22 VERBATIM
    g = gap_null(k, n_perm, seed)
    return float((1 + (g <= max_gap_rad).sum()) / (1 + n_perm))


def screen(M, n_perm, seed=0):                          # scripts/22 VERBATIM
    M = np.asarray(M, dtype=np.float64)
    if len(M) < 3 or degenerate(M):
        return None
    xy, ratios, Vt = pca_2d(M)
    cx, cy, r, resid = fit_circle(xy)
    c = circularity(xy, cx, cy, r)
    return dict(k=int(len(M)), gap_ratio=c["gap_ratio"], max_gap_deg=c["max_gap_deg"],
                radius_ratio=c["radius_ratio"], circle_resid_over_r=float(resid),
                top2_var=float(ratios[:2].sum()),
                p_closed=gap_p(len(M), np.radians(c["max_gap_deg"]), n_perm, seed))


# ===========================================================================
def read_w_in(path, layers):
    from safetensors import safe_open
    out = {}
    with safe_open(str(path), framework="pt", device="cpu") as f:
        for L in layers:
            out[L] = 1.0 + f.get_tensor(
                f"model.layers.{L}.input_layernorm.weight").float().numpy().astype(np.float64)
    return out


def half_type_means(root, layer, tok, pool_ids, parity, W, chunk=CHUNK):
    """Per-half mean SAE code per pool type, projected through W_dec (scripts/22 D3/J6)."""
    d = Path(root) / f"layer_{layer}"
    meta = json.loads((d / "meta.json").read_text())
    d_sae = int(meta["d_sae"])
    indptr = np.load(d / "indptr.npy")
    indices = np.memmap(d / "indices.u16", dtype=np.uint16, mode="r")
    values = np.memmap(d / "values.f16", dtype=np.float16, mode="r")
    lut = np.full(int(tok.max()) + 1, -1, dtype=np.int64)
    lut[pool_ids] = np.arange(len(pool_ids))
    tidx = lut[tok]
    acc = np.zeros((len(pool_ids), d_sae), dtype=np.float64)
    cnt = np.zeros(len(pool_ids), dtype=np.int64)
    for lo in range(0, len(tok), chunk):
        hi = min(lo + chunk, len(tok))
        rows = np.arange(lo, hi)
        keep = (((rows // SEQ_LEN) % 2) == parity) & (tidx[lo:hi] >= 0)
        if not keep.any():
            continue
        a, b = int(indptr[lo]), int(indptr[hi])
        ct = (indptr[lo + 1:hi + 1] - indptr[lo:hi]).astype(np.int64)
        m = np.repeat(keep, ct)
        cols = np.asarray(indices[a:b], dtype=np.int64)[m]
        vals = np.asarray(values[a:b], dtype=np.float32)[m]
        rid = np.repeat(tidx[lo:hi], ct)[m]
        np.add.at(acc, (rid, cols), vals)
        cnt += np.bincount(tidx[lo:hi][keep], minlength=len(pool_ids))
    assert cnt.min() > 0, f"empty type in half parity={parity}: min count {cnt.min()}"
    acc /= cnt[:, None]
    out = np.empty((len(pool_ids), W.shape[1]), dtype=np.float64)
    for lo in range(0, len(pool_ids), 500):
        hi = min(lo + 500, len(pool_ids))
        out[lo:hi] = acc[lo:hi].astype(np.float32) @ W
    del acc
    return out, cnt, meta


def build_basis(X, d=D_SVD):
    Vt = np.linalg.svd(X, full_matrices=False)[2]
    return np.ascontiguousarray(Vt[:d].T)


def plane_specs(seed, d=D_SVD):
    specs = [("coord", (i, j)) for i in range(d) for j in range(i + 1, d)]
    rng = np.random.default_rng(seed)
    for r in range(N_RANDOM_PLANES):
        specs.append(("rand", np.linalg.qr(rng.normal(size=(d, 2)))[0]))
    return specs


def plane_coords(Y, spec):
    kind, p = spec
    return Y[:, list(p)] if kind == "coord" else Y @ p


def plane_basis(V, spec):
    kind, p = spec
    return V[:, list(p)] if kind == "coord" else V @ p


def principal_angles(B1, B2):
    s = np.linalg.svd(B1.T @ B2, compute_uv=False)
    return [float(np.degrees(np.arccos(np.clip(x, -1.0, 1.0)))) for x in s]


def jaccard(a, b):
    A, B = set(a.tolist()), set(b.tolist())
    return len(A & B) / max(len(A | B), 1)


# ===========================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/phase4_gained_carriers_promotion.json")
    ap.add_argument("--root", default="cache/sparse_big")
    ap.add_argument("--l0", default="big")
    a = ap.parse_args()
    t_start = time.time()
    out_path = REPO / a.out
    declared_at = datetime.now().isoformat(timespec="seconds")
    stub = dict(script="scripts/44b_gained_carrier_promotion.py",
                item="Phase 4 section 2.3 + Addendum J -- promotion stage (SAE mean-code arm, J6)",
                prereg=("docs/phase4_prereg_2026-08-22.md section 2.3 (FROZEN) + "
                        "docs/phase4_addendum_J_2026-08-22.md J1-J5 (FROZEN)"),
                funded_by="results/phase4_gained_carriers.json (kill test PASS)",
                status="DECLARATION STUB (written before any read)",
                declared_at=declared_at, declarations=__doc__,
                deviation_J6=("closure and proposal run on the SAE mean-code arm, not the raw "
                              "residual: raw half means require an MPS forward pass (scripts/35b, "
                              "26,655.9 s) which R12 and the orchestrator forbid here"),
                config=dict(layers=list(LAYERS), d_svd=D_SVD, n_random_planes=N_RANDOM_PLANES,
                            n_planes_per_layer=630 + N_RANDOM_PLANES, top_n=TOP_N, cap_S=CAP_S,
                            pct_e=PCT_E, p_bg_max=P_BG_MAX, jaccard_p_max=JACCARD_P_MAX,
                            n_classperm=N_CLASSPERM, n_classperm_conjunction=N_CLASSPERM_CONJ,
                            n_clusterlike=N_CLUSTERLIKE, n_background_draws=N_BACKGROUND_DRAWS,
                            rms_eps=RMS_EPS, drop_coords=list(DROP_COORDS), no_mps=True))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(stub, indent=1))
    print(f"stub -> {out_path} at {declared_at}", flush=True)

    # ---- pool ------------------------------------------------------------
    tok = np.fromfile(REPO / a.root / "layer_0" / "tokens.i32", dtype=np.int32)
    cnt_all = np.bincount(tok)
    pool_ids = np.flatnonzero(cnt_all >= COUNT_FLOOR).astype(np.int64)
    z = np.load(REPO / "results/phase3_token_type_mean_resid_fulln.npz")
    assert np.array_equal(pool_ids, np.asarray(z["pool_ids"], dtype=np.int64)), "pool mismatch"
    n_types = len(pool_ids)
    n_seq = len(tok) // SEQ_LEN
    print(f"pool {n_types}; rows {len(tok)}; sequences {n_seq}", flush=True)

    os.environ.setdefault("HF_HOME", str(REPO / "cache/hf"))
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from transformers import AutoTokenizer
    tk = AutoTokenizer.from_pretrained("google/gemma-3-1b-pt")
    strings = [type_string(tk, int(t)) for t in pool_ids]
    feats = np.asarray([class_features(s) for s in strings], dtype=bool)
    class_id = np.array([int("".join("1" if v else "0" for v in row), 2) for row in feats])

    token_map = json.loads((REPO / "results/token_map.json").read_text())
    row_of = {int(t): i for i, t in enumerate(pool_ids)}
    month_rows = []
    for m in MONTH_NAMES:
        ip = sorted((int(x) for x in token_map["months"][m] if int(x) in row_of),
                    key=lambda x: -int(cnt_all[x]))
        month_rows.append(row_of[ip[0]])
    month_rows = np.asarray(month_rows, dtype=np.int64)
    weekday_rows = np.asarray(sorted({row_of[int(x)] for v in token_map["days"].values()
                                      for x in v if int(x) in row_of}), dtype=np.int64)
    cal_mask = np.zeros(n_types, dtype=bool)
    cal_mask[month_rows] = True
    cal_mask[weekday_rows] = True

    w_in_full = read_w_in(REPO / SNAP, LAYERS)
    keep_c = np.setdiff1d(np.arange(1152), np.asarray(DROP_COORDS))
    from gfdisc.saes import load_sae, get_decoder

    out_layers = {}
    for L in LAYERS:
        tL = time.time()
        sae = load_sae(L, a.l0, token=os.environ.get("HF_TOKEN"))
        W = get_decoder(sae)
        del sae
        halves, counts_h = {}, {}
        for name, parity in (("A", 0), ("B", 1)):
            t0 = time.time()
            M, c, meta = half_type_means(REPO / a.root, L, tok, pool_ids, parity, W)
            halves[name], counts_h[name] = M, c
            print(f"[L{L}] half {name}: means {M.shape} in {time.time()-t0:.1f}s "
                  f"(count min {int(c.min())} median {int(np.median(c))})", flush=True)
        del W

        X, Y, V, norms = {}, {}, {}, {}
        for name in ("A", "B"):
            Xr = rms_norm_x(halves[name][:, keep_c], w_in_full[L][keep_c])
            D = np.column_stack([np.ones(n_types),
                                 np.log10(np.maximum(counts_h[name].astype(np.float64), 1.0)),
                                 feats.astype(np.float64),
                                 np.sqrt(np.maximum((Xr ** 2).sum(1), 1e-300))])
            Xd = deflate(Xr, D)
            X[name] = Xd
            norms[name] = np.maximum((Xd ** 2).sum(1), 1e-300)
            V[name] = build_basis(Xd)
            Y[name] = Xd @ V[name]
        # half-B raw-member cloud used for every closure screen (centred inside screen())
        MB = halves["B"][:, keep_c]
        specs = plane_specs(442000 + 100 * L)

        # ---- background + cluster-like caches, per size ------------------
        bg_cache, cl_cache = {}, {}
        cosB = None

        def background(k):
            if k not in bg_cache:
                rng = np.random.default_rng(443000 + 100 * L + k)
                gaps = []
                for _ in range(N_BACKGROUND_DRAWS):
                    s = screen(MB[rng.choice(n_types, size=k, replace=False)], N_PERM_BACKGROUND)
                    if s is not None:
                        gaps.append(s["max_gap_deg"])
                bg_cache[k] = np.asarray(gaps)
            return bg_cache[k]

        def p_bg(k, max_gap_deg):
            g = background(k)
            return float((1 + int((g <= max_gap_deg).sum())) / (1 + len(g)))

        def clusterlike_rate(k):
            nonlocal cosB
            if k not in cl_cache:
                if cosB is None:
                    Z = MB - MB.mean(0, keepdims=True)
                    Z = (Z / np.maximum(np.linalg.norm(Z, axis=1, keepdims=True), 1e-12)
                         ).astype(np.float32)
                    cosB = Z @ Z.T
                    np.fill_diagonal(cosB, -2.0)
                rng = np.random.default_rng(445000 + 100 * L + k)
                hits = 0
                for _ in range(N_CLUSTERLIKE):
                    seed_t = int(rng.integers(0, n_types))
                    nb = np.argpartition(-cosB[seed_t], k - 1)[:k - 1]
                    idx = np.concatenate([[seed_t], nb])
                    s = screen(MB[idx], N_PERM_BACKGROUND)
                    if s is not None and p_bg(k, s["max_gap_deg"]) < P_BG_MAX:
                        hits += 1
                cl_cache[k] = hits / N_CLUSTERLIKE
            return cl_cache[k]

        def twodraw_p(k_a, k_b, obs):
            rng = np.random.default_rng(446000 + 100 * L + k_a)
            vals = [jaccard(rng.choice(n_types, size=k_a, replace=False),
                            rng.choice(n_types, size=k_b, replace=False))
                    for _ in range(N_TWODRAW)]
            v = np.asarray(vals)
            return float((1 + int((v >= obs).sum())) / (1 + len(v)))

        # class-block permutations of type identity (D8/D10)
        rngp = np.random.default_rng(444000 + 100 * L)
        perms = []
        for _ in range(N_CLASSPERM):
            p = np.arange(n_types)
            for c in np.unique(class_id):
                idx = np.flatnonzero(class_id == c)
                p[idx] = rngp.permutation(idx)
            perms.append(p)
        perms = np.asarray(perms)

        # ---- per-plane promotion -----------------------------------------
        rows, n_cal, n_noncal, conj_hits, conj_cells = [], 0, 0, 0, 0
        best_month_top24 = 0
        for pi, spec in enumerate(specs):
            eA = (plane_coords(Y["A"], spec) ** 2).sum(1) / norms["A"]
            thr = np.percentile(eA, PCT_E)
            SA = np.flatnonzero(eA > thr)
            if len(SA) > CAP_S:
                SA = SA[np.argsort(-eA[SA])[:CAP_S]]
            eB = (plane_coords(Y["B"], spec) ** 2).sum(1) / norms["B"]
            SB = np.argpartition(-eB, len(SA))[:len(SA)]
            top24 = SA[np.argsort(-eA[SA])[:TOP_N]] if len(SA) >= TOP_N else SA
            n_cal_top = int(cal_mask[top24].sum())
            best_month_top24 = max(best_month_top24, int(np.isin(top24, month_rows).sum()))
            is_cal = n_cal_top >= CAL_MIN_IN_TOP24
            J = jaccard(SA, SB)
            maskA = np.zeros(n_types, dtype=bool); maskA[SA] = True
            jn = np.asarray([maskA[perms[t][SB]].sum() for t in range(N_CLASSPERM)],
                            dtype=np.float64)
            jn = jn / (2 * len(SA) - jn)
            p_j = float((1 + int((jn >= J).sum())) / (1 + len(jn)))
            S = np.intersect1d(SA, SB)
            rec = dict(plane=("coord_%d_%d" % spec[1] if spec[0] == "coord" else f"rand_{pi-630}"),
                       kind=spec[0], calendar=bool(is_cal), n_calendar_in_top24=n_cal_top,
                       n_months_in_top24=int(np.isin(top24, month_rows).sum()),
                       size_SA=int(len(SA)), size_SB=int(len(SB)), overlap=int(len(S)),
                       jaccard=float(J), jaccard_p_classperm=p_j,
                       principal_angles_deg=principal_angles(plane_basis(V["A"], spec),
                                                             plane_basis(V["B"], spec)))
            if len(S) >= SCREEN_MIN:
                s = screen(MB[S], N_PERM_SCREEN)
                if s is not None:
                    pb = p_bg(len(S), s["max_gap_deg"])
                    rec.update(gap_ratio=s["gap_ratio"], p_closed=s["p_closed"], p_bg=pb,
                               radius_ratio=s["radius_ratio"], top2_var=s["top2_var"])
                    if pb < P_BG_MAX and p_j <= JACCARD_P_MAX:
                        loo = []
                        for m in S:
                            sub = S[S != m]
                            s2 = screen(MB[sub], N_PERM_BACKGROUND)
                            loo.append(bool(s2 is not None
                                            and p_bg(len(sub), s2["max_gap_deg"]) < P_BG_MAX))
                        rec["loo_pass_fraction"] = float(np.mean(loo))
                        rec["loo_all"] = bool(all(loo))
                        rec["promoted"] = bool(all(loo))
                    else:
                        rec["promoted"] = False
            rec.setdefault("promoted", False)
            if rec["promoted"]:
                order = np.argsort(-eB[S])
                rec["members"] = [dict(row=int(S[o]), string=strings[S[o]],
                                       count=int(cnt_all[pool_ids[S[o]]])) for o in order]
                rec["member_hash"] = member_hash(S)
                n_cal += int(is_cal); n_noncal += int(not is_cal)
            # D10(a) class-permuted conjunction, subsampled permutations
            if len(SA) >= SCREEN_MIN:
                for t in range(N_CLASSPERM_CONJ):
                    conj_cells += 1
                    SBp = perms[t][SB]
                    inter = np.intersect1d(SA, SBp)
                    jp = len(inter) / (2 * len(SA) - len(inter))
                    p_jp = (1 + int((jn >= jp).sum())) / (1 + len(jn))
                    if p_jp > JACCARD_P_MAX or len(inter) < SCREEN_MIN:
                        continue
                    sp = screen(MB[inter], N_PERM_BACKGROUND)
                    if sp is not None and p_bg(len(inter), sp["max_gap_deg"]) < P_BG_MAX:
                        conj_hits += 1
            rows.append(rec)
            if (pi + 1) % 200 == 0:
                print(f"[L{L}] {pi+1}/{len(specs)} planes ({time.time()-tL:.0f}s)", flush=True)

        sizes = sorted({r["overlap"] for r in rows if r["overlap"] >= SCREEN_MIN})
        cl_rates = {str(k): clusterlike_rate(k) for k in sizes}
        promoted = [r for r in rows if r["promoted"]]
        rate_conj = conj_hits / max(conj_cells, 1)
        out_layers[str(L)] = dict(
            n_planes=len(specs), best_month_count_in_top24_halfA=best_month_top24,
            n_calendar_planes=int(sum(r["calendar"] for r in rows)),
            n_promoted=len(promoted), n_promoted_calendar=n_cal, n_promoted_noncalendar=n_noncal,
            promoted=promoted,
            jaccard_summary=dict(
                median=float(np.median([r["jaccard"] for r in rows])),
                p90=float(np.percentile([r["jaccard"] for r in rows], 90)),
                max=float(max(r["jaccard"] for r in rows)),
                median_calendar=float(np.median([r["jaccard"] for r in rows if r["calendar"]])
                                      if any(r["calendar"] for r in rows) else float("nan"))),
            twodraw_null_example=dict(
                k=int(rows[0]["size_SA"]),
                p_for_median_jaccard=twodraw_p(rows[0]["size_SA"], rows[0]["size_SA"],
                                               float(np.median([r["jaccard"] for r in rows])))),
            classperm_conjunction=dict(n_cells=conj_cells, n_hits=conj_hits, rate=rate_conj,
                                       n_permutations=N_CLASSPERM_CONJ,
                                       expected_false_promotions=rate_conj * len(specs)),
            clusterlike_closure_rate_by_size=cl_rates,
            clusterlike_expected_false_promotions_max=float(
                max(cl_rates.values()) * len(specs)) if cl_rates else None,
            background_sizes=sizes, seconds=round(time.time() - tL, 1))
        print(f"[L{L}] promoted {len(promoted)} ({n_cal} CALENDAR / {n_noncal} non-calendar); "
              f"class-perm conjunction rate {rate_conj:.5f} -> expected false "
              f"{rate_conj*len(specs):.2f}; cluster-like rates "
              f"{ {k: round(v,3) for k, v in cl_rates.items()} }; {time.time()-tL:.0f}s",
              flush=True)
        del halves, X, Y, V, MB

    stub.update(status="COMPLETE (promotion stage)", layers=out_layers,
                runtime_seconds=round(time.time() - t_start, 1))
    out_path.write_text(json.dumps(stub, indent=1))
    print(f"wrote {a.out} ({stub['runtime_seconds']}s)", flush=True)


if __name__ == "__main__":
    main()
