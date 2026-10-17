"""Phase 3, Stage 1, probe 1 -- nuisance-deflated, split-half-calibrated closing-plane search.

PRE-REGISTERED. Spec: docs/phase3_prereg_2026-08-21.md section 2.1 (FROZEN). Rationale:
docs/next_avenues_2026-08-21.md avenue 1. Nothing in `DECLARED` below was chosen after
seeing a Phase-3 number; the stub carrying it is written to --out before any cache row is
read (global rule R1).

What this is. The campaign has a calibrated ring *test* (screen() + matched same-size
background) and no calibrated *search*: 3,748 screens produced 51 flags at roughly the
chance rate. This replaces Leiden proximity assembly over the 3,795-type pool with
spectral assembly, deflates the nuisance axes that token-type means are known to carry
(corpus frequency, token class), and calibrates the search by SPLIT-HALF -- candidates are
assembled on the even-sequence half and adjudicated on the odd-sequence half with the
frozen, unchanged closure gate. The operating point (expected false confirmations) is then
n_candidates_k x the measured per-k background acceptance rate.

Pipeline, in the order it runs:
  1. declaration stub -> --out (before any cache read);
  2. one streaming pass over the CSR memmaps per layer, splitting rows by SEQUENCE parity
     ((row // 255) % 2) into two float64 [3795, 16384] accumulators, each divided by its own
     half-count and projected through w_dec (no b_dec) -> M_A, M_B in R^{3795 x 1152};
  3. deflation, identical on both halves: every one of the 1152 columns is residualised on
     [1, log10(count), leading_space, capitalized, all_caps, has_digit, is_punct, non_ascii];
  4. DISCOVERY on deflated M_A only: SVD -> top-32 right-singular vectors -> the 496
     coordinate planes -> per plane a k in [5,15] at the largest relative radius drop ->
     candidate iff the frozen circularity thresholds hold;
  5. CONFIRMATION on deflated M_B: the unchanged screen() at 20,000 permutations plus
     background_for_size at 500 draws;
  6. controls, survivor audits, and only then token strings (labels_after_freeze).

Provenance of the copied code. gap_null, gap_p, screen, background_for_size and p_bg_of are
COPIED VERBATIM (up to the constants they close over) from scripts/22_token_type_geometry.py
lines 364-420, and member_hash from line 561. They live in a script, not in src/gfdisc, so
they cannot be imported; copying rather than re-deriving is what keeps the confirmation gate
byte-identical to the one that produced results/token_type_geometry.json. Everything imported
from src/gfdisc (fits.pca_2d, fit_circle, circularity, angles, degenerate;
nulls.benjamini_hochberg) is the same code path the frozen results used.

Memory. One float64 accumulator per half is 3795 x 16384 x 8 = 497 MB; the two halves are
~1 GB and only one layer's pair is ever held. The projected 1152-d matrices are 35 MB each
and ARE retained across layers, because the fallback arm (if control (ii) fails) has to
re-enter every layer's deflated halves.
"""
import argparse, gc, hashlib, json, os, sys, time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HUB_OFFLINE", "1")
sys.path.insert(0, str(REPO / "src"))
from gfdisc.fits import pca_2d, fit_circle, circularity, angles, degenerate
from gfdisc.nulls import benjamini_hochberg
from gfdisc.saes import load_sae, get_decoder

# ===========================================================================
# DECLARED RULES -- every threshold below is frozen by prereg section 2.1
# ===========================================================================
COUNT_FLOOR = 100              # build_pool floor: the 3,795-type population
D_MODEL = 1152
SEQ_LEN = 255                  # cache rows per sequence (BOS dropped at cache build)
DEFAULT_LAYERS = [5, 6, 7, 8, 9]

N_SVD = 32                     # top-d right-singular vectors of deflated M_A
K_MIN, K_MAX = 5, 15           # candidate sizes
CAND_MAX_GAP_RATIO = 2.0       # circularity gate on the discovery plane
CAND_RADIUS_RATIO_LO = 0.5
CAND_RADIUS_RATIO_HI = 3.0

N_PERM_CONFIRM = 20000         # screen() permutations at confirmation
N_PERM_BACKGROUND = 2000       # screen() permutations inside each background draw
N_BACKGROUND_DRAWS = 500       # random same-size type subsets per represented k
N_PERM_LOO = 2000              # leave-one-out jackknife permutations (survivor audit)
N_PERM_ORDER = 20000           # within-lexical-block permutations for the frequency nuisance
ALPHA = 0.05                   # p_closed and p_bg threshold for CONFIRMED
BH_Q = 0.05                    # BH over confirmed p_closed, per layer

JACCARD_THRESHOLD = 0.7        # "is this candidate the known month object?"
TOP_N_CONFIRMED = 5            # control (ii) looks at the top-5 confirmed
CONTROL_GAP_RATIO_REF = 1.659  # recorded L6 value of the 13-member set
CONTROL_GAP_RATIO_TOL = 0.2
CONTROL_I_LAYERS = [6, 7]
CONTROL_II_LAYERS = [6, 7]
CONTROL_III_LAYER = 8

N_FALLBACK_PLANES = 500        # random orthonormal 2-planes inside span(V), if (ii) fails
FALLBACK_SEED = 0

# Prereg section 0 exposure record: the L6 centred_leiden cluster 14 member set,
# member_hash 7d1d75a303d6 (over POOL INDICES, as scripts/22 computes it).
MONTH13_TOKEN_IDS = [3353, 5142, 5261, 5365, 5505, 5656, 5729, 5857,
                     6027, 6198, 6243, 6384, 7250]
MONTH13_MEMBER_HASH = "7d1d75a303d6"

TOKEN_CLASS_NAMES = ["leading_space", "capitalized", "all_caps",
                     "has_digit", "is_punct", "non_ascii"]
DESIGN_COLUMNS = ["intercept", "log10_count"] + TOKEN_CLASS_NAMES


def declared_rules():
    return dict(
        spec="docs/phase3_prereg_2026-08-21.md section 2.1, FROZEN",
        rationale="docs/next_avenues_2026-08-21.md avenue 1",
        population=(f"the token-type pool at count >= {COUNT_FLOOR} (build_pool of "
                    "scripts/22_token_type_geometry.py), expected 3,795 types"),
        count_floor=COUNT_FLOOR,
        split=("SEQUENCE parity: cache row r belongs to half A iff ((r // 255) % 2) == 0, "
               "half B otherwise. Same-document rows never straddle the split. Each half's "
               "accumulator is divided by that half's own per-type row count"),
        representation=("per token type and per half: mean over that half's cache rows of the "
                        "sparse SAE code vector, then mean_code @ w_dec -> 1152-d. b_dec is "
                        "NOT added (identical to scripts/22; a constant offset that the "
                        "deflation intercept and PCA both remove exactly)"),
        accumulator_dtype="float64, divided by half-count, cast to float32 for the w_dec matmul",
        deflation=(f"identical on both halves: least squares of all {D_MODEL} columns on the "
                   f"single design [{', '.join(DESIGN_COLUMNS)}] (3795 x "
                   f"{len(DESIGN_COLUMNS)}), residual retained. log10_count uses the FULL-cache "
                   "pool count (not the half count), so the design is byte-identical for A "
                   "and B"),
        token_class_derivation=(
            "the token-class indicators are computed from tokenizer.convert_ids_to_tokens for "
            "EVERY one of the 3,795 pool types with the SentencePiece boundary marker U+2581 "
            "replaced by a space; non_ascii is evaluated AFTER that replacement so the marker "
            "itself does not set it. leading_space = string starts with ' '; all_caps = at "
            "least one alphabetic char and all alphabetic chars upper; capitalized = first "
            "alphabetic char upper and NOT all_caps; has_digit = any digit; is_punct = no "
            "alphanumeric char at all; non_ascii = any codepoint > 127. This is a blind "
            "structural nuisance basis applied uniformly to the whole pool -- it is not a "
            "label read: no per-candidate string enters any decision statistic, and candidate "
            "token strings appear only in the post-freeze labels_after_freeze block"),
        discovery=(
            f"deflated M_A ONLY. SVD -> top-{N_SVD} right-singular vectors V ({D_MODEL} x "
            f"{N_SVD}); the C({N_SVD},2) = 496 coordinate planes. Per plane (i<j): coords = "
            "M_A_defl @ V[:, [i,j]] (exactly the columns i,j of M_A_defl @ V, V orthonormal); "
            "per-type in-plane radius and angle; radii sorted descending; k chosen in "
            f"[{K_MIN},{K_MAX}] maximising r_k / r_(k+1); candidate = the top-k types by "
            f"radius iff fit_circle + circularity on their 2-D plane coords give gap_ratio <= "
            f"{CAND_MAX_GAP_RATIO} AND radius_ratio in [{CAND_RADIUS_RATIO_LO}, "
            f"{CAND_RADIUS_RATIO_HI}]. Candidates are deduplicated by frozenset of pool "
            "indices; the provenance plane recorded is the one with the SMALLEST discovery "
            "gap_ratio. n_candidates is reported per k, raw (per plane) and deduplicated"),
        confirmation=(
            f"deflated M_B, same deflation recipe, so both halves are treated identically. "
            "RECORDED DEVIATION FROM THE PREREG WORDING: section 2.1 says 'Confirmation (M_B, "
            "full 1152-d)'. That phrase distinguishes screening in the FULL 1152-d space from "
            "screening inside the discovery plane; it is honoured -- screen() runs its own "
            "pca_2d over all 1152 deflated coordinates. The half that is deflated is M_B, not "
            "raw M_B, so that the discovery and confirmation halves differ only by which cache "
            "rows they average. Both are reported for the positive control so the effect of "
            "deflation on the known object is visible. "
            f"screen(M_B_defl[members], n_perm={N_PERM_CONFIRM}) -> gap_ratio, max_gap_deg, "
            f"p_closed; p_bg from background_for_size(M_B_defl, pool_n, k, seed=layer, "
            f"n_draws={N_BACKGROUND_DRAWS}) via p_bg_of on max_gap_deg. CONFIRMED iff "
            f"p_closed < {ALPHA} AND p_bg < {ALPHA} (strict, per section 2.1). "
            f"Benjamini-Hochberg at q={BH_Q} over the confirmed candidates' p_closed within "
            "each layer, reported alongside; it does not change CONFIRMED"),
        operating_point=("expected false confirmations at size k = (deduplicated "
                         "n_candidates at k) x (that layer's background rate_p_closed_lt_05 "
                         "at k). Reported per k and summed per layer. It prices the closure "
                         "tier only, which is the tier the spec names"),
        controls=dict(
            i=(f"the 13-member set (token ids {MONTH13_TOKEN_IDS}, member_hash "
               f"{MONTH13_MEMBER_HASH} over pool indices) passes screen() on deflated M_A AND "
               f"on deflated M_B at layers {CONTROL_I_LAYERS} with |gap_ratio - "
               f"{CONTROL_GAP_RATIO_REF}| <= {CONTROL_GAP_RATIO_TOL} and p_closed < {ALPHA}. "
               "The same four screens are ALSO reported on UNdeflated M_A/M_B at every layer, "
               "as the visible measure of what deflation does to the known object"),
            ii=(f"a candidate with Jaccard >= {JACCARD_THRESHOLD} to the 13-member set is among "
                f"the top-{TOP_N_CONFIRMED} CONFIRMED candidates, ranked ascending by (p_bg, "
                f"max_gap_deg), at layers {CONTROL_II_LAYERS}. If this fails on the coordinate "
                f"planes, the fallback arm runs: {N_FALLBACK_PLANES} random orthonormal "
                f"2-planes inside span(V), seed {FALLBACK_SEED}, identical candidate rule, "
                "reported SEPARATELY. The fallback arm never rescues the section 2.1 verdict, "
                "which is computed on the coordinate-plane arm"),
            iii=(f"no candidate with Jaccard >= {JACCARD_THRESHOLD} to the 13-member set is "
                 f"CONFIRMED at layer {CONTROL_III_LAYER} (the byte-identical set is the "
                 "recorded L8 failure: gap_ratio 3.6356, p_closed 0.746)")),
        survivor_audit=(
            f"every CONFIRMED candidate with Jaccard < {JACCARD_THRESHOLD} to the 13-member "
            f"set: (a) leave-one-out jackknife -- all k LOO subsets must keep p_closed < "
            f"{ALPHA} at {N_PERM_LOO} permutations on deflated M_B; (b) frequency nuisance -- "
            "Spearman(angle rank in the M_B_defl top-2 plane of the members, log10 count) with "
            f"{N_PERM_ORDER} WITHIN-LEXICAL-BLOCK permutations of the counts, blocks = "
            "leading_space x capitalized, two-sided"),
        decision=("all three controls pass -> the blind CONFIRMED list is reportable. Any "
                  "control fails -> 'spectral branch closed; reason = which control'. If the "
                  "layers a control needs were not run, the control is NOT EVALUABLE and no "
                  "verdict is issued (global rule R2: no blind verdict without a validated "
                  "instrument)"),
        r1_note=("this dict is written to --out as a declaration stub BEFORE any cache row is "
                 "read; declared_at is carried into the final JSON"),
        copied_code=("gap_null, gap_p, screen, background_for_size, p_bg_of and member_hash "
                     "are copied verbatim from scripts/22_token_type_geometry.py (lines "
                     "364-420 and 561); they are not importable. pca_2d, fit_circle, "
                     "circularity, angles, degenerate and benjamini_hochberg are imported "
                     "from src/gfdisc, the same code path the frozen results used"),
        residual_limitation=("split-half removes selection-on-noise, not shared nuisance "
                             "geometry: any structure present in BOTH halves and not spanned "
                             "by the 8-column design survives confirmation by construction. "
                             "The angle used by the frequency nuisance check is circular, so "
                             "its linear rank depends on the arctan2 branch cut; the "
                             "permutation null is computed under the identical convention"),
    )


# ===========================================================================
# stage 1 -- pool and per-half type means
# ===========================================================================
def build_pool(root, layer0=0):
    """COPIED from scripts/22_token_type_geometry.py build_pool (L211), floor-100 arm only.

    tokens.i32 is byte-identical across layers. Counts are over the WHOLE cache, so
    --max-rows (smoke only) never changes the population or the deflation design.
    """
    tok = np.fromfile(Path(root) / f"layer_{layer0}" / "tokens.i32", dtype=np.int32)
    cnt = np.bincount(tok)
    main = np.flatnonzero(cnt >= COUNT_FLOOR)
    return tok, cnt, main


def half_means(root, layer, tok, pool_ids, max_rows=None, chunk=200_000):
    """Streaming groupby of the CSR cache by (token id, sequence parity).

    Same shape as scripts/22 type_means (L220) -- 200k-row chunks over the memmaps,
    np.add.at into a float64 accumulator -- but with TWO accumulators, one per half.
    """
    d = Path(root) / f"layer_{layer}"
    meta = json.loads((d / "meta.json").read_text())
    d_sae = int(meta["d_sae"])
    indptr = np.load(d / "indptr.npy")
    indices = np.memmap(d / "indices.u16", dtype=np.uint16, mode="r")
    values = np.memmap(d / "values.f16", dtype=np.float16, mode="r")

    n_rows = len(tok) if max_rows is None else min(int(max_rows), len(tok))
    lut = np.full(int(tok.max()) + 1, -1, dtype=np.int64)
    lut[pool_ids] = np.arange(len(pool_ids))
    tidx = lut[tok]
    parity = ((np.arange(len(tok), dtype=np.int64) // SEQ_LEN) % 2).astype(np.int8)

    acc = [np.zeros((len(pool_ids), d_sae), dtype=np.float64) for _ in range(2)]
    cnt_half = [np.zeros(len(pool_ids), dtype=np.int64) for _ in range(2)]
    n_used = [0, 0]
    for lo in range(0, n_rows, chunk):
        hi = min(lo + chunk, n_rows)
        a, b = int(indptr[lo]), int(indptr[hi])
        cols = np.asarray(indices[a:b], dtype=np.int64)
        vals = np.asarray(values[a:b], dtype=np.float32)
        ct = indptr[lo + 1:hi + 1] - indptr[lo:hi]
        rid = np.repeat(tidx[lo:hi], ct)
        rpar = np.repeat(parity[lo:hi], ct)
        ti, pi = tidx[lo:hi], parity[lo:hi]
        for h in (0, 1):
            m = (rid >= 0) & (rpar == h)
            np.add.at(acc[h], (rid[m], cols[m]), vals[m])
            sel = ti[(ti >= 0) & (pi == h)]
            cnt_half[h] += np.bincount(sel, minlength=len(pool_ids))
            n_used[h] += int(len(sel))
        del cols, vals, rid, rpar
    n_zero = []
    for h in (0, 1):
        n_zero.append(int((cnt_half[h] == 0).sum()))
        acc[h] /= np.maximum(cnt_half[h], 1)[:, None]     # guard: zero-row types stay at 0
    return acc, cnt_half, n_used, n_zero, n_rows, d_sae, meta


def project(acc, W, chunk=2000):
    """COPIED from scripts/22_token_type_geometry.py project (L247)."""
    out = np.empty((acc.shape[0], W.shape[1]), dtype=np.float32)
    for lo in range(0, acc.shape[0], chunk):
        hi = min(lo + chunk, acc.shape[0])
        out[lo:hi] = acc[lo:hi].astype(np.float32) @ W
    return out


# ===========================================================================
# stage 2 -- the frozen ring test, COPIED VERBATIM from scripts/22
# (gap_null L364, gap_p L374, screen L379, background_for_size L393, p_bg_of L418,
#  member_hash L561). They are defined inside a script and cannot be imported; copying
#  rather than re-deriving is what keeps this gate byte-identical to the frozen one.
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


def member_hash(members):
    return hashlib.md5(",".join(str(int(x)) for x in sorted(members)).encode()).hexdigest()[:12]


# ===========================================================================
# token-class nuisance basis (blind: computed for the whole pool, never per candidate)
# ===========================================================================
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


def build_design(counts, feats):
    """[1, log10(count), 6 token-class indicators] -- one 3795 x 8 design for both halves."""
    n = len(counts)
    return np.column_stack([np.ones(n), np.log10(np.maximum(counts.astype(np.float64), 1.0)),
                            feats.astype(np.float64)])


def deflate(M, D):
    """Residualise every column of M on D by one least-squares solve."""
    M = np.asarray(M, dtype=np.float64)
    coef, *_ = np.linalg.lstsq(D, M, rcond=None)
    return M - D @ coef


# ===========================================================================
# stage 3 -- discovery on deflated M_A
# ===========================================================================
def plane_candidate(coords):
    """The frozen per-plane rule. coords is [n_types, 2] in the plane."""
    rad = np.hypot(coords[:, 0], coords[:, 1])
    order = np.argsort(-rad)
    rs = rad[order]
    best_k, best_ratio = None, -np.inf
    for k in range(K_MIN, K_MAX + 1):
        den = float(rs[k])                       # r_{k+1}
        if den <= 0:
            continue
        ratio = float(rs[k - 1]) / den
        if ratio > best_ratio:
            best_ratio, best_k = ratio, k
    if best_k is None:
        return None
    sel = order[:best_k]
    P = coords[sel]
    if degenerate(P):
        return None
    cx, cy, r, resid = fit_circle(P)
    c = circularity(P, cx, cy, r)
    accepted = bool(c["gap_ratio"] <= CAND_MAX_GAP_RATIO
                    and CAND_RADIUS_RATIO_LO <= c["radius_ratio"] <= CAND_RADIUS_RATIO_HI)
    return dict(k=int(best_k), members=np.sort(sel), accepted=accepted,
                radius_drop_ratio=float(best_ratio),
                gap_ratio=float(c["gap_ratio"]), max_gap_deg=float(c["max_gap_deg"]),
                radius_ratio=float(c["radius_ratio"]), circle_resid_over_r=float(resid),
                min_member_radius=float(rs[best_k - 1]),
                next_radius=float(rs[best_k]))


def dedup(hits):
    """Collapse identical member sets; provenance = the plane with the smallest gap_ratio."""
    best = {}
    for h in hits:
        key = frozenset(int(x) for x in h["members"])
        cur = best.get(key)
        if cur is None:
            h = dict(h); h["n_planes_hit"] = 1
            best[key] = h
        else:
            cur["n_planes_hit"] += 1
            if h["gap_ratio"] < cur["gap_ratio"]:
                n = cur["n_planes_hit"]
                h = dict(h); h["n_planes_hit"] = n
                best[key] = h
    return list(best.values())


# ===========================================================================
# stage 4 -- Spearman with within-lexical-block permutations
# ===========================================================================
def _rankdata(x):
    x = np.asarray(x, dtype=np.float64)
    order = np.argsort(x, kind="stable")
    ranks = np.empty(len(x), dtype=np.float64)
    ranks[order] = np.arange(1, len(x) + 1, dtype=np.float64)
    sx = x[order]
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and sx[j + 1] == sx[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + j + 2) / 2.0
        i = j + 1
    return ranks


def _spearman(a, b):
    ra, rb = _rankdata(a), _rankdata(b)
    ra = ra - ra.mean(); rb = rb - rb.mean()
    den = float(np.sqrt((ra @ ra) * (rb @ rb)))
    return float(ra @ rb / den) if den > 0 else 0.0


def block_perm_spearman(ang, log_count, blocks, n_perm, seed):
    """Spearman(angle rank, log10 count), null = counts permuted WITHIN lexical blocks."""
    rho = _spearman(ang, log_count)
    groups = {}
    for i, b in enumerate(blocks):
        groups.setdefault(b, []).append(i)
    groups = {k: np.asarray(v) for k, v in groups.items()}
    rng = np.random.default_rng(seed)
    null = np.empty(n_perm, dtype=np.float64)
    lc = np.asarray(log_count, dtype=np.float64)
    for t in range(n_perm):
        x = lc.copy()
        for g in groups.values():
            if len(g) > 1:
                x[g] = lc[g][rng.permutation(len(g))]
        null[t] = _spearman(ang, x)
    return dict(rho=rho, n_perm=int(n_perm),
                p_two_sided=float((1 + int((np.abs(null) >= abs(rho)).sum())) / (1 + n_perm)),
                null_mean=float(null.mean()), null_abs_p95=float(np.percentile(np.abs(null), 95)),
                blocks={str(k): int(len(v)) for k, v in groups.items()},
                n_blocks=int(len(groups)),
                note=("blocks = leading_space x capitalized; a block of size 1 is fixed under "
                      "the null by construction. angle is circular, so its linear rank depends "
                      "on the arctan2 branch cut -- the null uses the identical convention"))


def jaccard(a, b):
    a, b = set(int(x) for x in a), set(int(x) for x in b)
    return float(len(a & b) / max(len(a | b), 1))


# ===========================================================================
def confirm_candidates(cands, MB, pool_n, layer, bg_cache, n_perm, n_bg, pool_ids,
                       month13_idx, arm):
    """Screen every candidate on deflated M_B, attach p_bg, CONFIRMED and Jaccard."""
    rows = []
    for c in cands:
        mem = np.asarray(sorted(int(x) for x in c["members"]))
        k = len(mem)
        if k not in bg_cache:
            bg_cache[k] = background_for_size(MB, pool_n, k, seed=layer, n_draws=n_bg)
        bg = bg_cache[k]
        s = screen(MB[mem], n_perm, seed=0)
        p_bg = None if s is None else p_bg_of(bg, s["max_gap_deg"])
        confirmed = bool(s is not None and s["p_closed"] < ALPHA
                         and p_bg is not None and p_bg < ALPHA)
        rows.append(dict(
            arm=arm, k=int(k),
            members_pool_idx=[int(x) for x in mem],
            members_token_ids=[int(pool_ids[i]) for i in mem],
            member_hash=member_hash(mem),
            jaccard_to_month13=jaccard(mem, month13_idx),
            provenance=dict(plane=c.get("plane"), n_planes_hit=int(c.get("n_planes_hit", 1)),
                            radius_drop_ratio=c["radius_drop_ratio"],
                            discovery_gap_ratio=c["gap_ratio"],
                            discovery_max_gap_deg=c["max_gap_deg"],
                            discovery_radius_ratio=c["radius_ratio"],
                            discovery_circle_resid_over_r=c["circle_resid_over_r"],
                            min_member_radius=c["min_member_radius"],
                            next_radius=c["next_radius"]),
            confirmation=None if s is None else dict(
                s, p_bg=p_bg,
                background_rate_p_closed_lt_05=bg["rate_p_closed_lt_05"],
                background_mean_max_gap_deg=bg["mean_max_gap_deg"]),
            confirmed=confirmed, bh_pass_within_layer=False))
    conf = [r for r in rows if r["confirmed"]]
    if conf:
        keep = benjamini_hochberg([r["confirmation"]["p_closed"] for r in conf], alpha=BH_Q)
        for r, kk in zip(conf, keep):
            r["bh_pass_within_layer"] = bool(kk)
    return rows


def rank_confirmed(rows):
    """Top-5 rule of control (ii): ascending by p_bg, then max_gap_deg."""
    conf = [r for r in rows if r["confirmed"]]
    return sorted(conf, key=lambda r: (r["confirmation"]["p_bg"],
                                       r["confirmation"]["max_gap_deg"]))


def control_ii_for_layer(rows):
    ranked = rank_confirmed(rows)
    top = ranked[:TOP_N_CONFIRMED]
    hit = [r for r in top if r["jaccard_to_month13"] >= JACCARD_THRESHOLD]
    any_conf = [r for r in ranked if r["jaccard_to_month13"] >= JACCARD_THRESHOLD]
    return dict(n_confirmed=len(ranked), passed=bool(hit),
                top_ranked=[dict(rank=i + 1, k=r["k"], member_hash=r["member_hash"],
                                 jaccard_to_month13=r["jaccard_to_month13"],
                                 p_bg=r["confirmation"]["p_bg"],
                                 p_closed=r["confirmation"]["p_closed"],
                                 max_gap_deg=r["confirmation"]["max_gap_deg"])
                            for i, r in enumerate(top)],
                best_month_like_rank=(None if not any_conf else
                                      int(ranked.index(any_conf[0]) + 1)),
                best_month_like_jaccard=(max((r["jaccard_to_month13"] for r in ranked),
                                             default=None)))


def survivor_audit(rows, MB, counts, feats, pool_ids, layer, n_perm_order):
    """LOO jackknife + frequency nuisance for CONFIRMED non-month candidates."""
    out = []
    for r in rows:
        if not r["confirmed"] or r["jaccard_to_month13"] >= JACCARD_THRESHOLD:
            continue
        mem = np.asarray(r["members_pool_idx"])
        loo = []
        for drop in range(len(mem)):
            sub = np.delete(mem, drop)
            s = screen(MB[sub], N_PERM_LOO, seed=0)
            loo.append(dict(dropped_pool_idx=int(mem[drop]),
                            dropped_token_id=int(pool_ids[mem[drop]]),
                            p_closed=None if s is None else s["p_closed"],
                            gap_ratio=None if s is None else s["gap_ratio"]))
        loo_pass = bool(loo) and all(x["p_closed"] is not None and x["p_closed"] < ALPHA
                                     for x in loo)
        xy, ratios, _ = pca_2d(MB[mem])
        cx, cy, rr, _ = fit_circle(xy)
        ang = angles(xy, cx, cy)
        blocks = [f"lead={bool(feats[i, 0])}|cap={bool(feats[i, 1])}" for i in mem]
        freq = block_perm_spearman(ang, np.log10(np.maximum(counts[mem], 1.0)), blocks,
                                   n_perm_order, seed=10_000 * layer + int(mem[0]))
        out.append(dict(member_hash=r["member_hash"], arm=r["arm"], k=r["k"],
                        members_pool_idx=r["members_pool_idx"],
                        members_token_ids=r["members_token_ids"],
                        loo_jackknife=dict(n_perm=N_PERM_LOO, all_pass=loo_pass, subsets=loo),
                        frequency_nuisance=freq,
                        angles_deg=[float(x) for x in np.degrees(ang) % 360],
                        counts=[int(counts[i]) for i in mem]))
    return out


def random_planes(C, n_planes, seed):
    """n orthonormal 2-planes inside span(V), expressed in the V coordinate frame.

    C = M_defl @ V, so plane coords = C @ Q for a 32x2 orthonormal Q.
    """
    rng = np.random.default_rng(seed)
    for p in range(n_planes):
        G = rng.normal(size=(C.shape[1], 2))
        Q, _ = np.linalg.qr(G)
        yield p, C @ Q


# ===========================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="cache/sparse_big")
    ap.add_argument("--layers", nargs="+", type=int, default=DEFAULT_LAYERS)
    ap.add_argument("--l0", default="big")
    ap.add_argument("--out", default="results/phase3_spectral_search.json")
    ap.add_argument("--max-rows", type=int, default=None,
                    help="SMOKE ONLY: truncate the cache pass. Recorded as smoke=true.")
    ap.add_argument("--n-perm", type=int, default=N_PERM_CONFIRM,
                    help=f"SMOKE ONLY: confirmation/order permutations (frozen {N_PERM_CONFIRM})")
    ap.add_argument("--n-bg", type=int, default=N_BACKGROUND_DRAWS,
                    help=f"SMOKE ONLY: background draws per k (frozen {N_BACKGROUND_DRAWS})")
    a = ap.parse_args()
    t_start = time.time()
    layers = [int(x) for x in a.layers]
    smoke = bool(a.max_rows is not None or a.n_perm != N_PERM_CONFIRM
                 or a.n_bg != N_BACKGROUND_DRAWS)
    n_perm, n_bg = int(a.n_perm), int(a.n_bg)

    # ---- R1: declaration to disk BEFORE any cache row is read ----
    stub = dict(status="DECLARATION STUB -- rules frozen, no results yet",
                script="scripts/34_spectral_search.py",
                declared_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
                declared_rules=declared_rules(), layers=layers)
    Path(a.out).write_text(json.dumps(stub, indent=2))
    print("=== DECLARED RULES (frozen by prereg 2.1, written to disk before any cache read) ===",
          flush=True)
    print(json.dumps(declared_rules(), indent=2)[:3000], flush=True)
    if smoke:
        print(f"\n*** SMOKE RUN: max_rows={a.max_rows} n_perm={n_perm} n_bg={n_bg} -- "
              "mechanics only, NOT a verdict ***", flush=True)

    # ---- population, design, controls' index set ----
    tok, cnt, pool_ids = build_pool(a.root)
    pool_n = len(pool_ids)
    pool_pos = {int(t): i for i, t in enumerate(pool_ids)}
    counts = cnt[pool_ids].astype(np.int64)
    print(f"\npool: {pool_n} token types with >= {COUNT_FLOOR} occurrences "
          f"({int(counts.sum())} of {len(tok)} cache rows)", flush=True)

    same = {}
    for L in layers:
        t2 = np.fromfile(Path(a.root) / f"layer_{L}" / "tokens.i32", dtype=np.int32)
        same[str(L)] = bool(len(t2) == len(tok) and np.array_equal(t2, tok))
        del t2
    assert all(same.values()), f"tokens.i32 differs across layers: {same}"

    from transformers import AutoTokenizer
    from gfdisc.env import MODEL_NAME
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=os.environ.get("HF_TOKEN"))
    pool_strings = [type_string(tokenizer, t) for t in pool_ids]
    feats = np.asarray([class_features(s) for s in pool_strings], dtype=np.float64)
    D = build_design(counts, feats)
    print(f"design: {D.shape} [{', '.join(DESIGN_COLUMNS)}]; column sums "
          f"{[int(x) for x in feats.sum(0)]}", flush=True)

    missing13 = [t for t in MONTH13_TOKEN_IDS if t not in pool_pos]
    month13_idx = np.asarray(sorted(pool_pos[t] for t in MONTH13_TOKEN_IDS
                                    if t in pool_pos))
    m13_hash = member_hash(month13_idx)
    print(f"13-member set: {len(month13_idx)}/13 in pool, member_hash {m13_hash} "
          f"(expected {MONTH13_MEMBER_HASH}, match={m13_hash == MONTH13_MEMBER_HASH})",
          flush=True)

    # ---- per layer ----
    layer_out, cache = [], {}
    for L in layers:
        tL = time.time()
        print(f"\n================ layer {L} ================", flush=True)
        t0 = time.time()
        acc, cnt_half, n_used, n_zero, n_rows_scanned, d_sae, meta = half_means(
            a.root, L, tok, pool_ids, max_rows=a.max_rows)
        t_cache = time.time() - t0
        print(f"  cache pass: {n_rows_scanned} rows, halves used {n_used}, "
              f"zero-count types per half {n_zero} ({round(t_cache, 1)}s)", flush=True)

        sae = load_sae(L, a.l0, token=os.environ.get("HF_TOKEN"))
        W = get_decoder(sae); del sae
        MA_raw = project(acc[0], W).astype(np.float64)
        MB_raw = project(acc[1], W).astype(np.float64)
        del acc, W; gc.collect()

        MA = deflate(MA_raw, D)
        MB = deflate(MB_raw, D)

        def _removed(raw, defl):
            """Fraction of CENTRED energy the 8-column design took out (the intercept
            column removes the mean, so the comparison must be against centred raw)."""
            num = float((np.asarray(defl) ** 2).sum())
            den = float(((np.asarray(raw) - np.asarray(raw).mean(0)) ** 2).sum())
            return float(1.0 - num / max(den, 1e-12))

        defl_frac = dict(half_A=_removed(MA_raw, MA), half_B=_removed(MB_raw, MB))

        # ---- control (i): the 13-member set, deflated AND undeflated, both halves ----
        c_i = {}
        for name, Mx in (("deflated_M_A", MA), ("deflated_M_B", MB),
                         ("undeflated_M_A", MA_raw), ("undeflated_M_B", MB_raw)):
            s = screen(Mx[month13_idx], n_perm, seed=0)
            if s is not None:
                s["gap_ratio_within_tol"] = bool(
                    abs(s["gap_ratio"] - CONTROL_GAP_RATIO_REF) <= CONTROL_GAP_RATIO_TOL)
                s["p_closed_lt_alpha"] = bool(s["p_closed"] < ALPHA)
                s["passes_rule_i"] = bool(s["gap_ratio_within_tol"] and s["p_closed_lt_alpha"])
            c_i[name] = s
        c_i["layer_passes_control_i"] = bool(
            c_i["deflated_M_A"] and c_i["deflated_M_B"]
            and c_i["deflated_M_A"]["passes_rule_i"] and c_i["deflated_M_B"]["passes_rule_i"])
        c_i["rule"] = (f"|gap_ratio - {CONTROL_GAP_RATIO_REF}| <= {CONTROL_GAP_RATIO_TOL} and "
                       f"p_closed < {ALPHA}, on deflated M_A AND deflated M_B; the undeflated "
                       "screens are reported so the effect of deflation on the known object "
                       "is visible, and are not part of the rule")
        print(f"  control (i) 13-set: defl_A gap_ratio="
              f"{None if not c_i['deflated_M_A'] else round(c_i['deflated_M_A']['gap_ratio'], 4)}"
              f" p={None if not c_i['deflated_M_A'] else c_i['deflated_M_A']['p_closed']}"
              f" | defl_B gap_ratio="
              f"{None if not c_i['deflated_M_B'] else round(c_i['deflated_M_B']['gap_ratio'], 4)}"
              f" p={None if not c_i['deflated_M_B'] else c_i['deflated_M_B']['p_closed']}"
              f" -> pass={c_i['layer_passes_control_i']}", flush=True)
        del MA_raw, MB_raw; gc.collect()

        # ---- discovery on deflated M_A ----
        U, S, Vt = np.linalg.svd(MA, full_matrices=False)
        V = Vt[:N_SVD].T                                   # [1152, 32], orthonormal
        C = MA @ V                                         # [3795, 32]
        raw_by_k, hits = {}, []
        for i in range(N_SVD):
            for j in range(i + 1, N_SVD):
                h = plane_candidate(C[:, [i, j]])
                if h is None or not h["accepted"]:
                    continue
                h["plane"] = [int(i), int(j)]
                raw_by_k[h["k"]] = raw_by_k.get(h["k"], 0) + 1
                hits.append(h)
        cands = dedup(hits)
        dedup_by_k = {}
        for c in cands:
            dedup_by_k[c["k"]] = dedup_by_k.get(c["k"], 0) + 1
        print(f"  discovery: {N_SVD * (N_SVD - 1) // 2} coordinate planes -> {len(hits)} "
              f"accepted, {len(cands)} unique member sets "
              f"({sorted(dedup_by_k.items())})", flush=True)

        # ---- confirmation on deflated M_B ----
        t0 = time.time()
        bg_cache = {}
        rows = confirm_candidates(cands, MB, pool_n, L, bg_cache, n_perm, n_bg,
                                  pool_ids, month13_idx, arm="coordinate_planes")
        n_conf = sum(r["confirmed"] for r in rows)
        print(f"  confirmation: {len(rows)} candidates, {n_conf} CONFIRMED "
              f"({round(time.time() - t0, 1)}s)", flush=True)

        exp_false = {}
        for k, nk in sorted(dedup_by_k.items()):
            rate = bg_cache.get(k, {}).get("rate_p_closed_lt_05")
            exp_false[str(k)] = dict(n_candidates=int(nk), background_rate_p_closed_lt_05=rate,
                                     expected_false_confirmations=(None if rate is None
                                                                   else round(nk * rate, 3)))
        exp_total = sum(v["expected_false_confirmations"] or 0.0 for v in exp_false.values())

        c_ii = control_ii_for_layer(rows)
        c_iii_hits = [r for r in rows
                      if r["confirmed"] and r["jaccard_to_month13"] >= JACCARD_THRESHOLD]
        audits = survivor_audit(rows, MB, counts, feats, pool_ids, L, n_perm)
        print(f"  control (ii) here: passed={c_ii['passed']} (n_confirmed="
              f"{c_ii['n_confirmed']}, best month-like Jaccard="
              f"{c_ii['best_month_like_jaccard']}); survivor audits: {len(audits)}", flush=True)

        layer_out.append(dict(
            layer=L, nominal_l0=meta.get("nominal_l0"), d_sae=int(d_sae),
            n_cache_rows_scanned=int(n_rows_scanned),
            n_rows_used_half_A=int(n_used[0]), n_rows_used_half_B=int(n_used[1]),
            n_zero_count_types_half_A=int(n_zero[0]), n_zero_count_types_half_B=int(n_zero[1]),
            half_count_summary=dict(
                A=dict(min=int(cnt_half[0].min()), median=float(np.median(cnt_half[0])),
                       max=int(cnt_half[0].max())),
                B=dict(min=int(cnt_half[1].min()), median=float(np.median(cnt_half[1])),
                       max=int(cnt_half[1].max()))),
            deflation=dict(design_columns=DESIGN_COLUMNS,
                           design_shape=[int(x) for x in D.shape],
                           fraction_centred_energy_removed=defl_frac),
            svd=dict(singular_values_top32=[float(x) for x in S[:N_SVD]],
                     variance_fraction_top32=float((S[:N_SVD] ** 2).sum()
                                                   / max((S ** 2).sum(), 1e-12))),
            controls=dict(control_i=c_i, control_ii_this_layer=c_ii,
                          control_iii_this_layer=dict(
                              applies=bool(L == CONTROL_III_LAYER),
                              n_month_like_confirmed=len(c_iii_hits),
                              member_hashes=[r["member_hash"] for r in c_iii_hits])),
            n_candidates=dict(n_planes=N_SVD * (N_SVD - 1) // 2,
                              n_planes_accepted=len(hits),
                              raw_by_k={str(k): int(v) for k, v in sorted(raw_by_k.items())},
                              dedup_by_k={str(k): int(v) for k, v in sorted(dedup_by_k.items())},
                              n_unique=len(cands)),
            candidates=rows,
            expected_false_confirmations=dict(by_k=exp_false, total=round(exp_total, 3)),
            background_by_size={str(k): {kk: vv for kk, vv in b.items() if kk != "_gaps"}
                               for k, b in sorted(bg_cache.items())},
            survivor_audits=audits,
            seconds=round(time.time() - tL, 1),
            cache_pass_seconds=round(t_cache, 1)))
        cache[L] = dict(MA=MA, MB=MB, C=C, bg=bg_cache)
        print(f"  layer {L} done in {round(time.time() - tL, 1)}s", flush=True)
        gc.collect()

    # ---- control (ii) across layers; fallback arm iff it fails on coordinate planes ----
    ii_needed = [L for L in CONTROL_II_LAYERS]
    ii_evaluable = all(L in layers for L in ii_needed)
    per_layer_ii = {str(lr["layer"]): lr["controls"]["control_ii_this_layer"]["passed"]
                    for lr in layer_out}
    c_ii_pass = bool(ii_evaluable and all(per_layer_ii.get(str(L), False) for L in ii_needed))

    fallback = None
    if ii_evaluable and not c_ii_pass:
        print("\n=== control (ii) FAILED on coordinate planes -> fallback arm "
              f"({N_FALLBACK_PLANES} random 2-planes in span(V), seed {FALLBACK_SEED}) ===",
              flush=True)
        fallback = dict(n_planes=N_FALLBACK_PLANES, seed=FALLBACK_SEED, layers={})
        for L in layers:
            t0 = time.time()
            st = cache[L]
            raw_by_k, hits = {}, []
            for p, coords in random_planes(st["C"], N_FALLBACK_PLANES, FALLBACK_SEED):
                h = plane_candidate(coords)
                if h is None or not h["accepted"]:
                    continue
                h["plane"] = ["random", int(p)]
                raw_by_k[h["k"]] = raw_by_k.get(h["k"], 0) + 1
                hits.append(h)
            fc = dedup(hits)
            fd_by_k = {}
            for c in fc:
                fd_by_k[c["k"]] = fd_by_k.get(c["k"], 0) + 1
            frows = confirm_candidates(fc, st["MB"], pool_n, L, st["bg"], n_perm, n_bg,
                                       pool_ids, month13_idx, arm="random_planes")
            fii = control_ii_for_layer(frows)
            fexp = {}
            for k, nk in sorted(fd_by_k.items()):
                rate = st["bg"].get(k, {}).get("rate_p_closed_lt_05")
                fexp[str(k)] = dict(n_candidates=int(nk),
                                    background_rate_p_closed_lt_05=rate,
                                    expected_false_confirmations=(None if rate is None
                                                                  else round(nk * rate, 3)))
            fallback["layers"][str(L)] = dict(
                n_planes_accepted=len(hits), n_unique=len(fc),
                raw_by_k={str(k): int(v) for k, v in sorted(raw_by_k.items())},
                dedup_by_k={str(k): int(v) for k, v in sorted(fd_by_k.items())},
                candidates=frows, control_ii_this_layer=fii,
                n_confirmed=int(sum(r["confirmed"] for r in frows)),
                expected_false_confirmations=dict(
                    by_k=fexp, total=round(sum(v["expected_false_confirmations"] or 0.0
                                               for v in fexp.values()), 3)),
                survivor_audits=survivor_audit(frows, st["MB"], counts, feats, pool_ids, L,
                                               n_perm),
                seconds=round(time.time() - t0, 1))
            print(f"  [fallback] layer {L}: {len(fc)} unique candidates, "
                  f"{fallback['layers'][str(L)]['n_confirmed']} CONFIRMED, control (ii) "
                  f"passed={fii['passed']} ({round(time.time() - t0, 1)}s)", flush=True)
        fallback["control_ii_pass"] = bool(
            all(fallback["layers"].get(str(L), {}).get("control_ii_this_layer", {}).get("passed",
                                                                                        False)
                for L in ii_needed))
        fallback["note"] = ("reported SEPARATELY. Per declared_rules.controls.ii the section "
                            "2.1 verdict is computed on the coordinate-plane arm; the fallback "
                            "arm does not rescue control (ii)")

    # ---- controls (i) and (iii) across layers ----
    i_evaluable = all(L in layers for L in CONTROL_I_LAYERS)
    per_layer_i = {str(lr["layer"]): lr["controls"]["control_i"]["layer_passes_control_i"]
                   for lr in layer_out}
    c_i_pass = bool(i_evaluable and all(per_layer_i.get(str(L), False) for L in CONTROL_I_LAYERS))

    iii_evaluable = CONTROL_III_LAYER in layers
    l8 = [lr for lr in layer_out if lr["layer"] == CONTROL_III_LAYER]
    n_l8_month_conf = (l8[0]["controls"]["control_iii_this_layer"]["n_month_like_confirmed"]
                       if l8 else None)
    c_iii_pass = bool(iii_evaluable and n_l8_month_conf == 0)

    all_evaluable = i_evaluable and ii_evaluable and iii_evaluable
    failed = [nm for nm, ok in (("i", c_i_pass), ("ii", c_ii_pass), ("iii", c_iii_pass))
              if not ok]
    if not all_evaluable:
        call = "NOT EVALUABLE -- the layers a control needs were not run"
        reportable = None
    elif not failed:
        call = "controls pass; the blind CONFIRMED list is reportable"
        reportable = True
    else:
        call = f"spectral branch closed; reason = control ({', '.join(failed)}) failed"
        reportable = False
    if smoke:
        # A reduced run can present every control layer and still be built on truncated
        # data or reduced permutation counts. It must never read as a verdict, whatever
        # the controls did; the per-control detail below stays intact as mechanics info.
        call = ("SMOKE -- mechanics only, NOT a verdict (run parameters differ from the "
                "frozen ones; see runtime_overrides). Control detail: " + call)
        reportable = None

    # ---- labels_after_freeze: the FIRST place any token string appears ----
    def strings_for(idx):
        return {str(int(pool_ids[i])): pool_strings[i] for i in idx}

    labels = dict(
        note=("written after every statistic above. No decision statistic in this file reads a "
              "token string; the token-class indicators are a blind structural covariate over "
              "the whole 3,795-type pool, declared in declared_rules.token_class_derivation"),
        month13=dict(token_ids=MONTH13_TOKEN_IDS, pool_idx=[int(x) for x in month13_idx],
                     member_hash=m13_hash, expected_member_hash=MONTH13_MEMBER_HASH,
                     hash_matches=bool(m13_hash == MONTH13_MEMBER_HASH),
                     missing_from_pool=missing13, strings=strings_for(month13_idx)),
        confirmed_candidates={}, survivor_candidates={})
    for lr in layer_out:
        for r in lr["candidates"]:
            if r["confirmed"]:
                labels["confirmed_candidates"][f"L{lr['layer']}:{r['member_hash']}"] = dict(
                    layer=lr["layer"], arm=r["arm"], k=r["k"],
                    jaccard_to_month13=r["jaccard_to_month13"],
                    strings=strings_for(r["members_pool_idx"]),
                    counts=[int(counts[i]) for i in r["members_pool_idx"]])
        for au in lr["survivor_audits"]:
            labels["survivor_candidates"][f"L{lr['layer']}:{au['member_hash']}"] = dict(
                layer=lr["layer"], strings=strings_for(au["members_pool_idx"]),
                counts=au["counts"], angles_deg=au["angles_deg"])
    if fallback:
        for Lk, fl in fallback["layers"].items():
            for r in fl["candidates"]:
                if r["confirmed"]:
                    labels["confirmed_candidates"][f"L{Lk}:fallback:{r['member_hash']}"] = dict(
                        layer=int(Lk), arm=r["arm"], k=r["k"],
                        jaccard_to_month13=r["jaccard_to_month13"],
                        strings=strings_for(r["members_pool_idx"]),
                        counts=[int(counts[i]) for i in r["members_pool_idx"]])

    out = dict(
        status=("SMOKE RUN -- mechanics only, no verdict" if smoke else
                "PRE-REGISTERED, prereg section 2.1 (frozen)"),
        item="Phase 3 Stage 1 probe 1 -- nuisance-deflated split-half closing-plane search",
        script="scripts/34_spectral_search.py",
        smoke=bool(smoke),
        generated=time.strftime("%Y-%m-%dT%H:%M:%S"),
        declared_at=stub["declared_at"],
        declared_rules=declared_rules(),
        runtime_overrides=dict(
            max_rows=a.max_rows, n_perm_confirm_used=n_perm, n_background_draws_used=n_bg,
            n_perm_order_used=n_perm, n_perm_loo_used=N_PERM_LOO,
            frozen_values=dict(n_perm_confirm=N_PERM_CONFIRM,
                               n_background_draws=N_BACKGROUND_DRAWS,
                               n_perm_order=N_PERM_ORDER, n_perm_loo=N_PERM_LOO),
            note=("empty overrides at the frozen values means the run is the pre-registered "
                  "one. --max-rows truncates ONLY the accumulation pass; the pool, the counts "
                  "and the deflation design always come from the whole cache")),
        config=dict(root=a.root, l0=a.l0, layers=layers,
                    tokens_identical_across_layers=same),
        pool=dict(n_types=int(pool_n), count_floor=COUNT_FLOOR, n_cache_rows=int(len(tok)),
                  n_cache_rows_covered=int(counts.sum()),
                  n_sequences=int(len(tok) // SEQ_LEN),
                  token_class_column_sums={n: int(v) for n, v in
                                           zip(TOKEN_CLASS_NAMES, feats.sum(0))}),
        layers=layer_out,
        fallback_arm=fallback,
        labels_after_freeze=labels,
        verdict=dict(
            call=call, reportable_blind_list=reportable, all_controls_evaluable=all_evaluable,
            control_i=dict(passed=c_i_pass, evaluable=i_evaluable,
                           layers_required=CONTROL_I_LAYERS, per_layer=per_layer_i),
            control_ii=dict(passed=c_ii_pass, evaluable=ii_evaluable,
                            layers_required=CONTROL_II_LAYERS, per_layer=per_layer_ii,
                            fallback_arm_run=bool(fallback is not None),
                            fallback_arm_passed=(None if fallback is None
                                                 else fallback["control_ii_pass"])),
            control_iii=dict(passed=c_iii_pass, evaluable=iii_evaluable,
                             layer_required=CONTROL_III_LAYER,
                             n_month_like_confirmed_at_L8=n_l8_month_conf),
            confirmed_by_layer={str(lr["layer"]): int(sum(r["confirmed"]
                                                          for r in lr["candidates"]))
                                for lr in layer_out},
            expected_false_confirmations_by_layer={
                str(lr["layer"]): lr["expected_false_confirmations"]["total"]
                for lr in layer_out}),
        caveats=[
            "Split-half removes selection-on-noise, not shared nuisance geometry. Structure "
            "present in both halves and outside the span of the 8-column design survives "
            "confirmation by construction; that is the declared residual limitation.",
            "The representation is a context-AVERAGED token-type mean over every corpus "
            "occurrence, carried forward from scripts/22. A ring here is a claim about the "
            "model's average representation of those token types.",
            "Expected false confirmations price the CLOSURE tier only (n_candidates_k x the "
            "per-k background rate_p_closed_lt_05). CONFIRMED additionally requires p_bg < "
            "0.05, so the number is an upper bound on the closure-only operating point, not a "
            "calibrated FDR for the conjunction.",
            "p_bg has a hard floor of 1/(n_background_draws + 1); at the frozen 500 draws that "
            "is 0.001996, so no candidate can score below it however good its gap is.",
            "The deflation is a linear projection out of an 8-dimensional design. Nonlinear "
            "frequency or token-class effects are not removed by it.",
            "The angle used by the frequency-nuisance Spearman is circular; its linear rank "
            "depends on the arctan2 branch cut. The block-permutation null uses the identical "
            "convention, so the p-value is valid, but the rho itself is convention-dependent.",
            "No cache-only statistic here is a claim about absolute sequence position: every "
            "cached row index is a document-relative index (prereg section 0).",
        ],
        runtime_seconds=round(time.time() - t_start, 1))
    Path(a.out).write_text(json.dumps(out, indent=2))
    print(f"\nVERDICT: {call}", flush=True)
    print(f"  control (i)={c_i_pass} (evaluable={i_evaluable})  "
          f"control (ii)={c_ii_pass} (evaluable={ii_evaluable})  "
          f"control (iii)={c_iii_pass} (evaluable={iii_evaluable})", flush=True)
    print(f"  confirmed by layer: {out['verdict']['confirmed_by_layer']}", flush=True)
    print(f"  expected false confirmations by layer: "
          f"{out['verdict']['expected_false_confirmations_by_layer']}", flush=True)
    print(f"wrote {a.out} ({out['runtime_seconds']}s)", flush=True)


if __name__ == "__main__":
    main()
