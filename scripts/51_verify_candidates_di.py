"""51 -- INDEPENDENT full-chain re-derivation of candidate findings (d) and (i).

Every statistic in this file is re-implemented from the DEFINITION, not imported. Scripts
42, 47, 47b, 48 and 48b were read to learn the constructions and are not imported; nothing
from gfdisc.fits is used either, because those are the statistics under audit. The only
imported machinery is the model/tokeniser loader and the cache-side context sampler
(gfdisc.model, gfdisc.rig), which are instrumentation, not statistics.

THE TWO CANDIDATES (results/phase5_readjudication_P.json, source48 PASS_P2 rows):
  d  L6, k = 9   ' environment',' security',' safety',' rights',' protection',
                 ' behavior',' efficiency',' identity',' privacy'
  i  L7, k = 10  ' find',' showed',' discover',' identified',' identify',' revealed',
                 ' confirmed',' recognized',' realize',' noticed'
ANCHOR: the strict-12 months at L6 (member_hash a36a4f4f3e3b).

THREE STAGES.

(1) PROPOSAL. Both candidates are claimed to come from proposer B: a channel-2
    "cluster-like assembly", S(s, k) = {s} + the k-1 nearest to s by pool-centred cosine in
    the layer's lag+1 successor-delta cloud. The stored `sources` fields are NOT the same
    shape for the two: d is B/L6/k9/{raw, dedup0.88, dedup0.90, dedup0.92}, so d should be a
    RAW k = 9 assembly; i is B/L7/k16/dedup0.92 ALONE, so i is a k = 16 assembly collapsed
    to 10 members by the R15 dedup ladder at 0.92. Testing only "is this set exactly
    {seed + (k-1) NN}" would therefore falsely fail i. This stage runs the whole chain --
    raw assembly at every k in 9..20 from every one of the 3,795 seeds, then the R15 dedup
    ladder (channel-1 pool-centred cosine > thr, drop the lower-count member), then the P1
    duplicate-referent guard -- and reports, for each candidate, every (k, rung) at which it
    is produced and from how many seeds.

(2) GEOMETRY TIER. Channel 1 = the layer's raw token-type mean residual with coordinate
    1038 dropped (R10). Channel 2 = the lag+1 robust delta rows (already 1151-dim; their
    stored dropped coordinate is asserted to be 1038). Closure = max angular gap over the
    2*pi/k expected of k evenly spaced points, in channel 1's own top-2 PCA plane about its
    Kasa-fitted circle centre (script 22's gap_ratio); PASS < 2.5. Order = two-channel rho,
    the maximum over the 2k cyclic shifts and reflections of the Spearman correlation
    between the two channels' recovered angular ranks. LOO drops each member in turn and
    re-scores both against the (layer, k-1) cell. The frequency-matched arm draws
    count-matched k-sets (+-0.25 log10 per member). The cluster-like conditional cell is the
    proposer-matched null: it is the EXACT enumeration of all 3,795 seed + (k-1) NN sets in
    CHANNEL-2 space, closure-conditioned -- exact rather than the task's 20,000 resamples
    because the draw is a deterministic function of the seed, so 3,795 distinct sets is the
    whole population that 20,000 draws with replacement would only estimate.

    MULTIPLICITY. The same enumeration answers the question the behavioural tier has to
    absorb: how many OTHER sets in that population are indistinguishable from the candidate
    on geometry alone (rho = 1.0, closure, LOO, frequency arm). Those are the "siblings".

(3) BEHAVIOURAL TIER, 47's rig re-implemented. At each member's own token position, add
    eps * ||h|| * t_hat_i to resid_post of layer L, where t_hat_i is the in-plane unit
    tangent at member i pointing toward INCREASING angle -- i.e. toward the member's angular
    successor, by construction from the same single PCA basis that defines the successor.
    Delta_i = [logit(succ) - logit(pred)]_steered - the same unsteered. Four nulls:
      (i)   10 random unit directions per member, orthogonalised against that member's tangent;
      (ii)  the tangent rotated +-90 degrees in plane (the radial directions);
      (iii) 50 random k-sets from the pool, each with its OWN plane, order and tangents;
      (iv)  50 SIBLING sets -- other seed + (k-1) NN assemblies from the same channel-2 cloud
            that also close. This is the proposer-matched behavioural null and it is new
            here; 47/47b did not have it. It is the null that says whether the behavioural
            tier can tell the candidate apart from the family it was drawn from.
    PASS (47's rule, unchanged): sign consistency >= (k-1)/k at SOME eps AND mean Delta at
    eps = 0.02 above null (iii)'s p99 AND above ALL of null (i)'s direction means.

DECLARED DEVIATION FROM THE TASK BRIEF, made before any forward. The brief's prose asks for
null (iii) "at the FULL 40-prompt budget"; its own arithmetic asks for "100 sets x k x
12 prompts x 2 ~ 60-80k forwards ~ 1.5-2 h". These are inconsistent: 40 prompts for
(iii)+(iv) is ~200k forwards, ~4.5 h at the measured 13 forwards/s, over the stated cap even
at the brief's 30-set fallback (which reduces sets, not prompts). The arithmetic is taken as
the operative spec -- nulls at 12 prompts, 50 sets each for (iii) and (iv), 10 directions for
(i) -- which is also exactly 47's rig and therefore keeps the anchor comparison meaningful.
Each candidate is scored BOTH ways: its 40-prompt mean Delta against the null p99 (47's
declared rule, and the number 47b's stored verdicts were built on) and its matched 12-prompt
value against the same p99.

THE ANCHOR IS A GATE. The strict-12 at L6 runs first through identical code. Context seeds
are new, so it is not expected to reproduce 47's numbers bit-for-bit; it is expected to PASS
by the rule. If it does not, the run stops and reports, because a rig that cannot recover the
established positive control cannot retire or confirm anything.
"""
import argparse
import contextlib
import hashlib
import json
import os
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HOME", str(REPO / "cache/hf"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
sys.path.insert(0, str(REPO / "src"))

from gfdisc import rig                            # noqa: E402
from gfdisc.env import DEVICE, MODEL_NAME         # noqa: E402
from gfdisc.model import capture, load_model      # noqa: E402

# =====================================================================================
# frozen constants -- all read off the campaign's own documents, none re-tuned here
# =====================================================================================
FULLN_RAW = "results/phase3_token_type_mean_resid_fulln.npz"
ROBUST_DELTAS = "results/phase3_lag_action_robust_deltas.npz"
CACHE_ROOT = "cache/sparse_big"

R10_DROP = 1038                 # prereg R10, layers >= 3
GAP = 2.5                       # frozen closure threshold
KMIN = 9                        # R8
PCT = 99.0
DEDUP_LADDER = (0.88, 0.90, 0.92)
PREFIX_MIN = 3                  # P1
K_RANGE = tuple(range(9, 21))
FREQ_TOL = 0.25
N_FREQ_TASK = 2000              # the brief's joint frequency arm
N_FREQ_48 = 200                 # 48's rule, for comparability with the stored pass_

EPS_GRID = (0.005, 0.01, 0.02)
EPS_NULL = 0.02
N_PROMPTS = 40
N_PROMPTS_NULL = 12
CONTEXT_LEN = 8
N_RAND_DIRS = 10
N_NULL_SETS = 50
BATCH = 16

SEED_CONTEXT = 5100             # new seeds: this is an independent re-derivation
SEED_DIRS = 5101
SEED_NULLSETS = 5102
SEED_SIBLINGS = 5103
SEED_FREQ = 5104

STRICT12_IDS = [3353, 5142, 5261, 5365, 5505, 5729, 5857, 6027, 6198, 6243, 6384, 7250]
STRICT12_HASH = "a36a4f4f3e3b"

# The two candidates, as stored token ids in results/phase5_readjudication_P.json.
CANDIDATES = [
    dict(name="L6_anchor_strict12", label="anchor", layer=6, k=12,
         member_hash=STRICT12_HASH, token_ids=STRICT12_IDS,
         stored_sources=None, stored_gap=1.542422222283976, stored_rho=0.993006993006993),
    dict(name="L6_d_k9", label="d", layer=6, k=9, member_hash="045a6dd71065",
         token_ids=[3453, 5052, 5646, 5726, 6733, 6868, 9286, 9834, 15603],
         stored_sources=["B/L6/k9/dedup0.88", "B/L6/k9/dedup0.9",
                         "B/L6/k9/dedup0.92", "B/L6/k9/raw"],
         stored_gap=1.6957314411958258, stored_rho=1.0),
    dict(name="L7_i_k10", label="i", layer=7, k=10, member_hash="2eff8657fbd3",
         token_ids=[1586, 6418, 6878, 8385, 8701, 9937, 10778, 12659, 13808, 14275],
         stored_sources=["B/L7/k16/dedup0.92"],
         stored_gap=2.079493930026796, stored_rho=1.0),
]


def mhash(members):
    """md5 over sorted POOL POSITIONS, first 12 hex -- script 42's frozen convention."""
    return hashlib.md5(",".join(str(int(x)) for x in sorted(members)).encode()).hexdigest()[:12]


# =====================================================================================
# STAGE 2a -- the geometry, re-implemented from the definition
# =====================================================================================
def pca2(X):
    """Top-2 principal plane coordinates of a mean-centred point set, plus the basis."""
    X = np.asarray(X, dtype=np.float64)
    Xc = X - X.mean(0, keepdims=True)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    var = S ** 2
    return Xc @ Vt[:2].T, var / max(var.sum(), 1e-12), Vt


def kasa_circle(xy):
    """Algebraic (Kasa) circle fit: minimise |2cx*x + 2cy*y + c0 - (x^2+y^2)|^2."""
    P = np.asarray(xy, dtype=np.float64)
    x, y = P[:, 0], P[:, 1]
    A = np.column_stack([2 * x, 2 * y, np.ones(len(x))])
    b = x ** 2 + y ** 2
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    cx, cy = float(sol[0]), float(sol[1])
    r = float(np.sqrt(max(sol[2] + cx ** 2 + cy ** 2, 1e-12)))
    d = np.hypot(x - cx, y - cy)
    return cx, cy, r, float(np.sqrt(np.mean((d - r) ** 2)) / max(r, 1e-12))


def is_degenerate(X, rel_tol=1e-6):
    """No usable geometry: coincident points, or fewer than 3 points carrying the spread."""
    X = np.asarray(X, dtype=np.float64)
    if len(X) < 3:
        return True
    c = X - X.mean(0, keepdims=True)
    scale = float(np.linalg.norm(X, axis=1).mean())
    spread = np.linalg.norm(c, axis=1)
    if scale <= 0 or spread.max() <= rel_tol * max(scale, 1e-12):
        return True
    return bool((spread > rel_tol * scale).sum() < 3)


def gap_ratio_of(X):
    """Script 22's max-gap closure: largest angular gap / (2*pi/k). None if degenerate."""
    X = np.asarray(X, dtype=np.float64)
    if len(X) < 3 or is_degenerate(X):
        return None
    xy, _, _ = pca2(X)
    cx, cy, _r, _ = kasa_circle(xy)
    a = np.sort(np.arctan2(xy[:, 1] - cy, xy[:, 0] - cx))
    gaps = np.diff(np.concatenate([a, [a[0] + 2 * np.pi]]))
    return float(gaps.max() / (2 * np.pi / len(X)))


def plane_angles(X):
    """Angles about the fitted circle centre, in the set's own top-2 plane."""
    xy, _, _ = pca2(X)
    cx, cy, _r, _ = kasa_circle(xy)
    return np.arctan2(xy[:, 1] - cy, xy[:, 0] - cx)


def rank_from_angles(ang, seed=0):
    """Ranks 0..k-1 of the angles, ties broken by a seed-fixed random permutation.

    The random tie-break is load bearing and is copied from the definition, not chosen: a
    stable argsort returns tied angles in INDEX order, which for a set whose channel never
    fires is the member order itself, manufacturing rho = 1.0 out of pure degeneracy.
    """
    jitter = np.random.default_rng(seed).permutation(len(ang))
    order = np.lexsort((jitter, np.asarray(ang, dtype=float)))
    r = np.empty(len(order), dtype=np.float64)
    r[order] = np.arange(len(order), dtype=np.float64)
    return r


def two_channel_rho(ang_a, ang_b, seed=0):
    """max over the 2k cyclic shifts/reflections of Spearman between the recovered orders.

    A circle has no privileged origin or handedness, so the honest agreement statistic
    maximises over the dihedral symmetries of a k-cycle. That maximisation is a selection
    effect, which is exactly why the cell null below applies the identical maximisation.
    """
    k = len(ang_a)
    ra = rank_from_angles(ang_a, seed)
    rb = rank_from_angles(ang_b, seed)
    best = -2.0
    for flip in (1, -1):
        for shift in range(k):
            ref = rank_from_angles((flip * rb + shift) % k, seed=0)
            d = ra - ref
            best = max(best, 1.0 - 6.0 * float(d @ d) / (k * (k * k - 1)))
    return float(best)


def cell_stats(X1, X2, seed=0):
    """(channel-1 gap_ratio, two-channel rho). rho None when either channel is unusable."""
    g = gap_ratio_of(X1)
    if g is None or X2 is None:
        return g, None
    X2 = np.asarray(X2, dtype=np.float64)
    if len(X2) < 3 or is_degenerate(X2):
        return g, None
    return g, two_channel_rho(plane_angles(X1), plane_angles(X2), seed)


def centred_unit(X):
    """Pool-centred, L2-normalised rows -- the space both proposers measure cosines in."""
    Z = X - X.mean(0)
    n = np.linalg.norm(Z, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return Z / n


# =====================================================================================
# STAGE 1 -- the proposer chain: assembly, R15 dedup ladder, P1 guard
# =====================================================================================
def assemblies(Z2, kmax):
    """Row s -> [s, then the kmax-1 nearest to s by pool-centred cosine in channel 2]."""
    S = (Z2 @ Z2.T).astype(np.float32)
    np.fill_diagonal(S, 2.0)
    idx = np.argpartition(-S, kmax, axis=1)[:, :kmax]
    r = np.arange(len(Z2))[:, None]
    out = idx[r, np.argsort(-S[r, idx], axis=1)]
    del S
    return out


def dedup_r15(members, Z1n, counts, thr):
    """R15: drop the lower-count member of the closest pair while its channel-1 pool-centred
    cosine exceeds thr."""
    S = [int(x) for x in members]
    while len(S) > 3:
        C = (Z1n[S] @ Z1n[S].T).astype(np.float64)
        np.fill_diagonal(C, -9.0)
        i, j = np.unravel_index(int(np.argmax(C)), C.shape)
        if C[i, j] <= thr:
            break
        drop = S[i] if counts[S[i]] < counts[S[j]] else S[j]
        S = [x for x in S if x != drop]
    return sorted(S)


def p1_normalise(s):
    return (s[1:] if s.startswith(" ") else s).lower()


def lev_le1(a, b):
    """Levenshtein(a, b) <= 1, by direct case analysis rather than a full DP table."""
    la, lb = len(a), len(b)
    if abs(la - lb) > 1:
        return False
    if la == lb:
        diff = 0
        for x, y in zip(a, b):
            if x != y:
                diff += 1
                if diff > 1:
                    return False
        return True
    if la > lb:
        a, b, la, lb = b, a, lb, la
    i = 0
    while i < la and a[i] == b[i]:
        i += 1
    return a[i:] == b[i + 1:]


def p1_is_dup(a, b):
    """P1: equal, or one a prefix of the other with the shorter of length >= 3, or Lev <= 1."""
    if a == b:
        return True
    if min(len(a), len(b)) >= PREFIX_MIN and (a.startswith(b) or b.startswith(a)):
        return True
    return lev_le1(a, b)


def build_dupmap(strings):
    """i -> {j : j is a P1 duplicate of i}. Candidate pairs come from equality buckets, a
    prefix dictionary and deletion-neighbourhood hashing (complete for Levenshtein <= 1);
    every candidate pair is then verified exactly, so the index is a filter, not the rule."""
    norm = [p1_normalise(s) for s in strings]
    cand = defaultdict(set)

    def add(i, j):
        if i != j:
            cand[i].add(j)
            cand[j].add(i)

    eq = defaultdict(list)
    for i, a in enumerate(norm):
        eq[a].append(i)
    for _a, idxs in eq.items():
        for x in range(len(idxs)):
            for y in range(x + 1, len(idxs)):
                add(idxs[x], idxs[y])
    for i, a in enumerate(norm):
        for L in range(PREFIX_MIN, len(a)):
            for j in eq.get(a[:L], ()):
                add(i, j)
    delmap = defaultdict(list)
    for i, a in enumerate(norm):
        keys = {a}
        for p in range(len(a)):
            keys.add(a[:p] + a[p + 1:])
        for kk in keys:
            delmap[kk].append(i)
    for _kk, idxs in delmap.items():
        if len(idxs) < 2:
            continue
        for x in range(len(idxs)):
            for y in range(x + 1, len(idxs)):
                add(idxs[x], idxs[y])

    dup = {}
    for i, js in cand.items():
        keep = {j for j in js if p1_is_dup(norm[i], norm[j])}
        if keep:
            dup[i] = keep
    return dup, norm


def p1_guard(members, dup, key):
    """P1. key[i] = (count, -pool index); repeatedly drop the globally smallest-key member
    that is the lower half of some still-live duplicate pair."""
    S = sorted(int(x) for x in members)
    if not any(i in dup for i in S):
        return S, []
    dropped = []
    while True:
        alive = set(S)
        losers = []
        for i in S:
            js = dup.get(i)
            if not js:
                continue
            if any(j in alive and j != i and key[i] < key[j] for j in js):
                losers.append(i)
        if not losers:
            break
        drop = min(losers, key=lambda i: key[i])
        S = [x for x in S if x != drop]
        dropped.append(int(drop))
        if len(S) < 2:
            break
    return sorted(S), dropped


def proposal_search(nn, Z1n, counts, dup, key, targets_by_hash, ks=K_RANGE):
    """Run every seed through the whole chain and record which targets come out where.

    Returns {target_hash: {"raw_k<k>": [seeds], "dedup<thr>_k<k>": [seeds], ...}} plus the
    per-rung population sizes needed to read those seed counts as rates.
    """
    hits = {h: defaultdict(list) for h in targets_by_hash}
    pop = defaultdict(int)
    n = nn.shape[0]
    for k in ks:
        for s in range(n):
            base = sorted(int(x) for x in nn[s, :k])
            rungs = [("raw", base)]
            for thr in DEDUP_LADDER:
                rungs.append((f"dedup{thr}", dedup_r15(base, Z1n, counts, thr)))
            for rung, mem in rungs:
                g, _ = p1_guard(mem, dup, key)
                pop[f"{rung}_k{k}"] += 1
                h = mhash(g)
                if h in hits:
                    hits[h][f"{rung}_k{k}"].append(int(s))
    return ({h: {kk: vv for kk, vv in sorted(v.items())} for h, v in hits.items()},
            dict(pop))


# =====================================================================================
# STAGE 2b -- cells, LOO, frequency arm, siblings
# =====================================================================================
def _enumerate_cell(nn, C1, C2, k, seed=0):
    """Score all 3,795 seed + (k-1) NN sets of one ordering. Returns (summary, gaps, rhos)."""
    gaps = np.full(len(nn), np.nan)
    rhos = np.full(len(nn), np.nan)
    for s in range(len(nn)):
        idx = nn[s, :k]
        g, r = cell_stats(C1[idx], C2[idx], seed)
        if g is not None:
            gaps[s] = g
        if r is not None:
            rhos[s] = r
    ok = np.isfinite(gaps)
    cond = ok & (gaps < GAP) & np.isfinite(rhos)
    rc = rhos[cond]
    p99 = float(np.percentile(rc, PCT, method="higher")) if len(rc) else None
    return dict(k=int(k), enumerated=True, n_seeds=int(len(nn)), n_scorable=int(ok.sum()),
                n_conditional=int(len(rc)),
                closure_rate=float((gaps[ok] < GAP).mean()) if ok.any() else None,
                p99=p99, exact1_rate=float((rc >= 1.0).mean()) if len(rc) else None,
                saturated=bool(p99 is not None and p99 >= 1.0)), gaps, rhos


def build_cells(nn1, nn2, C1, C2, k, seed=0):
    """The conditional cluster-like cell at (layer, k), in BOTH orderings.

    EXACT enumeration of all 3,795 seed + (k-1) NN sets, closure-conditioned on channel 1.
    Exact rather than sampled: the draw is a deterministic function of the seed, so 3,795
    distinct sets IS the population that 20,000 draws with replacement only estimate.

    Two orderings, because the brief and script 48 mean different things by "the cell".
    The brief asks for the PROPOSER-MATCHED cell -- assemblies grown in CHANNEL 2, which is
    where proposer B actually draws. Script 48's registered verdict used the "strict" view:
    the maximum of the channel-1-seeded and channel-2-seeded cells, which is a stricter bar
    and is what produced the stored `cell_saturated` and `p2_exact1_rate` fields. Both are
    reported so the stored verdicts and the proposer-matched ones can be read side by side.
    The per-seed arrays returned are CHANNEL 2's, because the sibling search is about the
    family the candidate was actually drawn from.
    """
    c1cell, _g1, _r1 = _enumerate_cell(nn1, C1, C2, k, seed)
    c2cell, g2, r2 = _enumerate_cell(nn2, C1, C2, k, seed)
    ps = [c["p99"] for c in (c1cell, c2cell) if c["p99"] is not None]
    e1 = [c["exact1_rate"] for c in (c1cell, c2cell) if c["exact1_rate"] is not None]
    strict = dict(k=int(k), p99=max(ps) if ps else None,
                  saturated=bool(c1cell["saturated"] or c2cell["saturated"]),
                  exact1_rate=max(e1) if e1 else None,
                  note="max over the channel-1-seeded and channel-2-seeded cells; this is "
                       "script 48's registered view")
    return dict(ch1=c1cell, ch2=c2cell, strict=strict), g2, r2


def loo_report(members, C1, C2, cell_km1, seed=0):
    """Drop each member in turn; closure and rho against the (layer, k-1) cell."""
    members = np.asarray(sorted(int(x) for x in members))
    gaps, rhos, states = [], [], []
    for i in range(len(members)):
        sub = np.delete(members, i)
        g, r = cell_stats(C1[sub], C2[sub], seed)
        gaps.append(None if g is None else float(g))
        rhos.append(None if r is None else float(r))
        if g is None or g >= GAP:
            states.append("FAIL")
        elif r is None or cell_km1["p99"] is None:
            states.append("NOT ADJUDICABLE")
        elif cell_km1["saturated"]:
            states.append("PASS" if (r >= 1.0 and (cell_km1["exact1_rate"] or 1.0) < 0.01)
                          else "NOT ADJUDICABLE")
        elif r > cell_km1["p99"]:
            states.append("PASS")
        else:
            states.append("FAIL")
    closure_ok = all(x is not None and x < GAP for x in gaps)
    if not closure_ok:
        overall = "closure fail"
    elif "FAIL" in states:
        overall = "FAIL"
    elif "NOT ADJUDICABLE" in states:
        overall = "NOT ADJUDICABLE"
    else:
        overall = "PASS"
    return dict(loo_closure_pass=bool(closure_ok), loo_order_state=overall,
                loo_states=states,
                loo_gap_max=max([x for x in gaps if x is not None], default=None),
                loo_rho_min=min([x for x in rhos if x is not None], default=None),
                loo_gaps=[None if x is None else round(x, 4) for x in gaps],
                loo_rhos=[None if x is None else round(x, 4) for x in rhos])


def frequency_arm(members, C1, C2, counts, rho, n, seed):
    """Count-matched k-sets: each drawn member is replaced by a pool type within +-0.25
    log10 of its own count. Reports BOTH statistics -- the brief's joint fraction (rho >=
    the candidate's AND closure) and 48's p99-of-rho rule -- because the stored `pass_`
    field was computed under the second."""
    rng = np.random.default_rng(seed)
    k = len(members)
    lc = np.log10(np.maximum(counts, 1.0))
    order = np.argsort(lc, kind="stable")
    lcs = lc[order]
    tgt = lc[np.asarray(sorted(int(x) for x in members))]
    lo = np.searchsorted(lcs, tgt - FREQ_TOL, side="left")
    hi = np.searchsorted(lcs, tgt + FREQ_TOL, side="right")
    rhos, gaps = [], []
    tries = 0
    while len(rhos) < n and tries < 20 * n:
        tries += 1
        idx, seen, ok = [], set(), True
        for a, b in zip(lo, hi):
            if b <= a:
                ok = False
                break
            for _t in range(30):
                cand = int(order[rng.integers(a, b)])
                if cand not in seen:
                    seen.add(cand)
                    idx.append(cand)
                    break
            else:
                ok = False
                break
        if not ok or len(idx) != k:
            continue
        g, r = cell_stats(C1[np.asarray(idx)], C2[np.asarray(idx)])
        if g is None or r is None:
            continue
        rhos.append(r)
        gaps.append(g)
    rhos = np.asarray(rhos)
    gaps = np.asarray(gaps)
    p99 = float(np.percentile(rhos, PCT, method="higher")) if len(rhos) else None
    joint = float(((rhos >= rho - 1e-12) & (gaps < GAP)).mean()) if len(rhos) else None
    return dict(n=int(len(rhos)), p99=p99,
                median=float(np.median(rhos)) if len(rhos) else None,
                closure_rate=float((gaps < GAP).mean()) if len(gaps) else None,
                joint_fraction_ge_rho_and_closed=joint,
                pass_48_rule=bool(p99 is not None and rho is not None and rho > p99))


# =====================================================================================
# STAGE 3 -- the behavioural rig
# =====================================================================================
@contextlib.contextmanager
def add_at(model, layer, idx_b, idx_p, delta):
    """Add delta[i] to layer `layer`'s residual stream at (idx_b[i], idx_p[i])."""
    def hook(_m, _i, out):
        t = out[0] if isinstance(out, tuple) else out
        new = t.float()
        new[idx_b, idx_p] = new[idx_b, idx_p] + delta
        new = new.to(t.dtype)
        return (new,) + out[1:] if isinstance(out, tuple) else new

    h = model.model.layers[layer].register_forward_hook(hook)
    try:
        yield
    finally:
        h.remove()


def plane_of(R_members):
    """Plane, circle, angles, cyclic successor/predecessor and unit tangents of one set.

    Every one of those comes from a SINGLE PCA basis, which is what makes the sign of Delta
    a prediction rather than a coin flip: theta_i is measured in (u1, u2), the successor is
    the next member counter-clockwise, and t_hat_i = -sin(theta_i) u1 + cos(theta_i) u2
    points toward increasing theta. A PCA sign flip flips the basis, the angles, the
    successor relation and the tangent together, so the verdict is invariant. Asserted.
    """
    X = np.asarray(R_members, dtype=np.float64)
    k = X.shape[0]
    xy, ratios, Vt = pca2(X)
    u1, u2 = Vt[0], Vt[1]
    cx, cy, r, resid = kasa_circle(xy)
    th = np.arctan2(xy[:, 1] - cy, xy[:, 0] - cx)
    order = np.argsort(th)
    succ = np.empty(k, dtype=np.int64)
    pred = np.empty(k, dtype=np.int64)
    for j, i in enumerate(order):
        succ[i] = order[(j + 1) % k]
        pred[i] = order[(j - 1) % k]
    tang = (-np.sin(th)[:, None] * u1[None, :]) + (np.cos(th)[:, None] * u2[None, :])
    tang /= np.maximum(np.linalg.norm(tang, axis=1, keepdims=True), 1e-12)
    rad = np.stack([np.cos(th), np.sin(th)], axis=1)
    rad90 = np.stack([-rad[:, 1], rad[:, 0]], axis=1)
    inplane = np.stack([tang @ u1, tang @ u2], axis=1)
    dots = np.sum(inplane * rad90, axis=1)
    assert np.all(dots > 0.999), f"tangent is not the radial rotated +90: min {dots.min():.6f}"
    spread = float(np.linalg.norm(xy - xy.mean(0), axis=1).mean())
    a = np.sort(th)
    gaps = np.diff(np.concatenate([a, [a[0] + 2 * np.pi]]))
    return dict(k=k, u1=u1, u2=u2, theta=th, order=order.tolist(), succ=succ, pred=pred,
                tangent=tang, pc_var_top2=float(ratios[:2].sum()),
                gap_ratio=float(gaps.max() / (2 * np.pi / k)),
                radius_ratio=float(r / max(spread, 1e-12)), circ_resid=float(resid))


def embed_1152(v, drop=R10_DROP, d=1152):
    """Re-insert the R10-dropped coordinate as 0.0. Norm preserving, so eps*||h|| still
    means what it says once the tangent is put back into the model's own space."""
    v = np.asarray(v, dtype=np.float64)
    if v.ndim == 1:
        out = np.zeros(d)
        out[:drop] = v[:drop]
        out[drop + 1:] = v[drop:]
        return out
    out = np.zeros((v.shape[0], d))
    out[:, :drop] = v[:, :drop]
    out[:, drop + 1:] = v[:, drop:]
    return out


def random_orth_dirs(t_hat, n, rng):
    """n unit directions in the full 1152-dim space, each orthogonalised against t_hat."""
    d = t_hat.shape[0]
    out = np.empty((n, d))
    for j in range(n):
        g = rng.standard_normal(d)
        g -= (g @ t_hat) * t_hat
        out[j] = g / max(np.linalg.norm(g), 1e-12)
    return out


class Runner:
    """One member's prompt block: build once, forward unsteered once, steer many times."""

    def __init__(self, model, tok, layer, device):
        self.model, self.tok, self.layer, self.device = model, tok, layer, device
        self.n_forward = 0

    def prepare(self, contexts, target_id):
        uniq, weights, _ = rig.dedupe(contexts)
        ids, attn, tpos = rig.build_batch(uniq, int(target_id),
                                          self.tok.bos_token_id, self.tok.pad_token_id)
        last_only = bool(torch.all(tpos == ids.shape[1] - 1))
        return dict(ids=ids.to(self.device), attn=attn.to(self.device),
                    tpos=tpos.to(self.device), last_only=last_only,
                    w=torch.tensor(weights, dtype=torch.float64))

    def _logits(self, blk, delta=None):
        ids, attn, tpos = blk["ids"], blk["attn"], blk["tpos"]
        keep = 1 if blk["last_only"] else 0
        out = []
        for lo in range(0, ids.shape[0], BATCH):
            hi = min(lo + BATCH, ids.shape[0])
            b_ids, b_attn, b_tpos = ids[lo:hi], attn[lo:hi], tpos[lo:hi]
            idx_b = torch.arange(hi - lo, device=self.device)
            with torch.no_grad():
                if delta is None:
                    lg = self.model(input_ids=b_ids, attention_mask=b_attn,
                                    use_cache=False, logits_to_keep=keep).logits
                else:
                    with add_at(self.model, self.layer, idx_b, b_tpos, delta[lo:hi]):
                        lg = self.model(input_ids=b_ids, attention_mask=b_attn,
                                        use_cache=False, logits_to_keep=keep).logits
            row = (lg[:, -1] if keep else lg[idx_b, b_tpos]).float().cpu()
            del lg
            assert torch.isfinite(row).all(), "non-finite logits"
            out.append(row)
            self.n_forward += hi - lo
        return torch.cat(out, 0)

    def baseline(self, blk):
        ids, attn, tpos = blk["ids"], blk["attn"], blk["tpos"]
        keep = 1 if blk["last_only"] else 0
        lg_all, hn_all = [], []
        for lo in range(0, ids.shape[0], BATCH):
            hi = min(lo + BATCH, ids.shape[0])
            b_ids, b_attn, b_tpos = ids[lo:hi], attn[lo:hi], tpos[lo:hi]
            idx_b = torch.arange(hi - lo, device=self.device)
            with torch.no_grad(), capture(self.model, [self.layer]) as store:
                lg = self.model(input_ids=b_ids, attention_mask=b_attn,
                                use_cache=False, logits_to_keep=keep).logits
            h = store[self.layer][idx_b, b_tpos].float()
            row = (lg[:, -1] if keep else lg[idx_b, b_tpos]).float().cpu()
            del lg
            store.clear()
            assert torch.isfinite(row).all() and torch.isfinite(h).all(), "non-finite baseline"
            lg_all.append(row)
            hn_all.append(h.norm(dim=-1).cpu())
            self.n_forward += hi - lo
        return torch.cat(lg_all, 0), torch.cat(hn_all, 0)

    def steered(self, blk, hnorm, direction, eps):
        d = torch.tensor(direction, dtype=torch.float32, device=self.device)
        delta = (eps * hnorm.to(self.device).float()).unsqueeze(1) * d.unsqueeze(0)
        return self._logits(blk, delta=delta)


def wmean(x, w):
    return float((x.double() * w).sum() / w.sum())


def cell_delta(lg_steer, lg_base, blk, succ_id, pred_id, member_ids):
    """Delta, raw and baseline-subtracted, plus the in-set softmax readout.

    The baseline subtraction is the declared reading of the prereg's statistic: without it
    the number is dominated by the standing logit asymmetry between two arbitrary vocabulary
    items, which is a fact about token frequency and not about steering.
    """
    w = blk["w"]
    d_st = lg_steer[:, succ_id] - lg_steer[:, pred_id]
    d_ba = lg_base[:, succ_id] - lg_base[:, pred_id]
    ids = torch.tensor(member_ids, dtype=torch.long)
    p_st = torch.softmax(lg_steer[:, ids].double(), dim=-1)
    j_s, j_p = member_ids.index(succ_id), member_ids.index(pred_id)
    lse_all = torch.logsumexp(lg_steer.double(), dim=-1)
    lse_set = torch.logsumexp(lg_steer[:, ids].double(), dim=-1)
    return dict(delta_raw=wmean(d_st, w), delta=wmean(d_st - d_ba, w),
                p_succ_inset=wmean(p_st[:, j_s], w), p_pred_inset=wmean(p_st[:, j_p], w),
                inset_mass=wmean(torch.exp(lse_set - lse_all), w))


def eval_set(runner, geo, member_pos, pool_ids, tokens2d, eligible, n_prompts,
             eps_list, rng, arm="natural", extra_dirs=None):
    """Score one member set. Every condition reuses the member's single unsteered forward,
    so Delta is exactly paired within a member."""
    k = geo["k"]
    member_ids = [int(pool_ids[p]) for p in member_pos]
    tang = embed_1152(geo["tangent"])
    rows = []
    for i in range(k):
        if arm == "natural":
            ctx, _ = rig.sample_contexts(tokens2d, n_prompts, CONTEXT_LEN, rng,
                                         eligible=eligible)
        else:
            ctx = [np.zeros(0, dtype=np.int64)]
        blk = runner.prepare(ctx, member_ids[i])
        lg_base, hnorm = runner.baseline(blk)
        s_id = member_ids[int(geo["succ"][i])]
        p_id = member_ids[int(geo["pred"][i])]
        cell = dict(member_index=i, k=k, n_rows=int(blk["ids"].shape[0]), conds={})
        for e in eps_list:
            lg = runner.steered(blk, hnorm, tang[i], e)
            cell["conds"][f"eps={e}"] = cell_delta(lg, lg_base, blk, s_id, p_id, member_ids)
        for name, D in (extra_dirs or {}).items():
            lg = runner.steered(blk, hnorm, D[i], EPS_NULL)
            cell["conds"][name] = cell_delta(lg, lg_base, blk, s_id, p_id, member_ids)
        rows.append(cell)
    return rows


def summarise(rows, cond):
    d = np.array([r["conds"][cond]["delta"] for r in rows], dtype=np.float64)
    return dict(mean_delta=float(d.mean()), sign_consistency=float((d > 0).mean()),
                n_pos=int((d > 0).sum()), k=len(d),
                mean_delta_raw=float(np.mean([r["conds"][cond]["delta_raw"] for r in rows])),
                p_succ_inset=float(np.mean([r["conds"][cond]["p_succ_inset"] for r in rows])),
                p_pred_inset=float(np.mean([r["conds"][cond]["p_pred_inset"] for r in rows])),
                inset_mass=float(np.mean([r["conds"][cond]["inset_mass"] for r in rows])))


# =====================================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/phase5_verify_candidates_di.json")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--skip-behavioural", action="store_true")
    a = ap.parse_args()
    out_path = REPO / a.out
    t0 = time.time()

    def log(msg):
        print(f"[{time.time()-t0:7.1f}s] {msg}", flush=True)

    n_prompts = 4 if a.smoke else N_PROMPTS
    n_prompts_null = 2 if a.smoke else N_PROMPTS_NULL
    n_rand = 2 if a.smoke else N_RAND_DIRS
    n_nullsets = 2 if a.smoke else N_NULL_SETS
    n_freq_task = 50 if a.smoke else N_FREQ_TASK
    ks = (9, 10, 12, 16) if a.smoke else K_RANGE
    KMAX = max(max(ks), max(c["k"] for c in CANDIDATES)) + 1
    if a.smoke and out_path.name == "phase5_verify_candidates_di.json":
        raise SystemExit("refusing to write a smoke run to the real --out")

    # ---------------- declaration stub, written BEFORE any artifact is read -------------
    declared_at = datetime.now().isoformat(timespec="seconds")
    res = dict(
        status="DECLARED", script="scripts/51_verify_candidates_di.py",
        declared_at=declared_at,
        stub_note="written before any read or forward; results overwrite this file "
                  "carrying declared_at",
        purpose="independent full-chain re-derivation of PASS_P2 candidates d (L6 k9) and "
                "i (L7 k10) -- proposal, geometry tier, multiplicity, behavioural tier",
        addenda_in_force=["docs/phase5_prereg_2026-08-22.md R13-R18",
                          "docs/phase5_addendum_P_2026-08-22.md P1-P4",
                          "docs/phase5_addendum_Q_2026-08-22.md Q1-Q2"],
        independence=("every statistic in scripts/51 is re-implemented from its definition. "
                      "scripts/42, 47, 47b, 48 and 48b were READ to learn the constructions "
                      "and are not imported; gfdisc.fits is not imported either. The only "
                      "imported machinery is gfdisc.model (loader, capture hook) and "
                      "gfdisc.rig (cache-side context sampling, batching, dedupe)."),
        rules=dict(
            channel_1="layer's raw token-type mean residual, coordinate 1038 dropped (R10)",
            channel_2="lag+1 robust successor-delta type means (1151-dim; the stored dropped "
                      "coordinate is asserted to be 1038), fp16 -> float32 -> float64 and "
                      "pool-centred / unit-normalised in float32, matching 48's casting so "
                      "that nearest-neighbour memberships are reproducible",
            closure="max angular gap / (2*pi/k) in channel 1's own top-2 PCA plane about the "
                    "Kasa circle centre (script 22's gap_ratio); PASS < 2.5, closure first",
            order="two_channel_rho = max over the 2k cyclic shifts and reflections of "
                  "Spearman between the two channels' recovered angular ranks, with the "
                  "seed-fixed random tie-break",
            cell="proposer-matched conditional cell = EXACT enumeration of all 3,795 "
                 "seed + (k-1) NN sets in CHANNEL-2 space, conditioned on channel-1 closure. "
                 "Exact rather than the brief's 20,000 resamples because the draw is a "
                 "deterministic function of the seed, so 3,795 sets is the whole population.",
            loo="every single-member drop, re-scored for closure and for rho against the "
                "(layer, k-1) cell",
            frequency_arm=f"{n_freq_task} count-matched k-sets (+-{FREQ_TOL} log10 per "
                          "member). Reported BOTH as the brief's joint fraction (rho >= the "
                          f"candidate's AND gap < {GAP}) and as 48's p99-of-rho rule, since "
                          "the stored pass_ field was computed under the second",
            multiplicity="siblings = other seed + (k-1) NN sets at the candidate's (layer, k) "
                         "that are indistinguishable from it on geometry: rho = 1.0, closure, "
                         "LOO and the frequency arm",
            behavioural=("h <- h + eps*||h||*t_hat_i at resid_post of layer L at the member's "
                         "own token position; Delta_i = [logit(succ) - logit(pred)]_steered "
                         "minus the same unsteered, averaged over that member's prompts"),
            behavioural_PASS=("sign consistency >= (k-1)/k at SOME eps in the grid (natural "
                              f"arm) AND mean Delta at eps={EPS_NULL} above null (iii)'s p99 "
                              "AND above ALL of null (i)'s direction means -- 47's rule, "
                              "unchanged"),
            nulls=dict(
                i=f"{n_rand} random unit directions per member, orthogonalised against that "
                  f"member's tangent, at eps={EPS_NULL}",
                ii="the tangent rotated +90 and -90 degrees in plane (the radial directions)",
                iii=f"{n_nullsets} random k-sets from the pool, each with its own plane, "
                    "circle, angular order and tangents",
                iv=f"{n_nullsets} SIBLING sets -- other seed + (k-1) NN assemblies from the "
                   "same channel-2 cloud that also close. The proposer-matched behavioural "
                   "null; new here, 47/47b did not have it"),
            anchor="strict-12 at L6 runs first through identical code and must PASS by the "
                   "rule. Context seeds are NEW, so a bit-match with 47 is not expected and "
                   "is not the gate; if the anchor does not PASS the run stops and reports.",
        ),
        deviations=[
            "NULL PROMPT BUDGET. The brief's prose asks for null (iii) at the full 40-prompt "
            "budget; the brief's own arithmetic asks for 12 prompts and 60-80k forwards in "
            "1.5-2 h. Those are inconsistent -- 40 prompts for nulls (iii)+(iv) is ~200k "
            "forwards, ~4.5 h at the measured 13 forwards/s, over the cap even at the "
            "brief's 30-set fallback (which reduces sets, not prompts). The arithmetic is "
            f"taken as the operative spec: candidates at {n_prompts} prompts, all nulls at "
            f"{n_prompts_null}, {n_nullsets} sets each for (iii) and (iv), {n_rand} "
            "directions for (i). This is also exactly 47's rig, which keeps the anchor "
            "comparison meaningful. Each candidate is scored BOTH ways: its 40-prompt mean "
            "Delta against the null p99 (47's declared rule and the basis of 47b's stored "
            "verdicts) and its matched 12-prompt value against the same p99.",
            "The cluster-like cell is enumerated exactly (3,795 sets) rather than sampled "
            "20,000 times; the enumeration is the population the sampling estimates.",
            "Null (iv) is run for the two candidates, not for the anchor: the anchor's "
            "verdict is not at issue and the brief asks for null (iv) on the candidates.",
        ] + (["SMOKE RUN -- not a result"] if a.smoke else []),
        config=dict(model=MODEL_NAME, device=DEVICE, dtype="bfloat16", attn="eager",
                    batch_size=BATCH, n_prompts=n_prompts, n_prompts_null=n_prompts_null,
                    context_len=CONTEXT_LEN, eps_grid=list(EPS_GRID), eps_null=EPS_NULL,
                    n_rand_dirs=n_rand, n_null_sets=n_nullsets, k_range=list(ks),
                    seeds=dict(context=SEED_CONTEXT, dirs=SEED_DIRS,
                               nullsets=SEED_NULLSETS, siblings=SEED_SIBLINGS,
                               freq=SEED_FREQ)),
        candidates=[{k: v for k, v in c.items()} for c in CANDIDATES],
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    def save(status="RUNNING"):
        res["status"] = status
        res["elapsed_seconds"] = round(time.time() - t0, 1)
        out_path.write_text(json.dumps(res, indent=1, default=str))

    out_path.write_text(json.dumps(res, indent=1, default=str))
    log(f"stub written to {out_path}")

    # ---------------- artifacts ----------------
    z = np.load(REPO / FULLN_RAW)
    pool_ids = z["pool_ids"].astype(np.int64)
    counts = z["counts"].astype(np.int64)
    layers = sorted({c["layer"] for c in CANDIDATES})
    d2 = np.load(REPO / ROBUST_DELTAS)
    assert np.array_equal(d2["pool_ids"].astype(np.int64), pool_ids), "pool mismatch"

    C1, C2, Z1n, Z2n = {}, {}, {}, {}
    for L in layers:
        assert list(d2[f"L{L}_lag1_dropped_coords"]) == [R10_DROP], \
            f"channel 2 at L{L} did not drop coordinate {R10_DROP}"
        C1[L] = np.delete(z[f"R_L{L}"].astype(np.float64), R10_DROP, axis=1)
        C2[L] = d2[f"L{L}_lag1"].astype(np.float32).astype(np.float64)
        Z1n[L] = centred_unit(C1[L]).astype(np.float32)
        Z2n[L] = centred_unit(C2[L]).astype(np.float32)
    log(f"channels built at layers {layers}; pool {len(pool_ids)} types")

    # ---------------- member recovery, hash-checked before anything else ---------------
    pos_of_id = {int(t): i for i, t in enumerate(pool_ids)}
    for c in CANDIDATES:
        mp = sorted(pos_of_id[int(t)] for t in c["token_ids"])
        assert len(mp) == c["k"], f"{c['name']}: {len(mp)} members, expected {c['k']}"
        got = mhash(mp)
        assert got == c["member_hash"], \
            f"{c['name']}: positions hash {got}, stored {c['member_hash']}"
        c["member_pos"] = mp
    res["member_recovery"] = {c["name"]: dict(member_pos=c["member_pos"],
                                              member_hash=c["member_hash"], reproduced=True)
                              for c in CANDIDATES}
    log("member recovery: all three sets reproduce their stored member_hash")
    save()

    # ================================================================================
    # STAGE 1 -- PROPOSAL
    # ================================================================================
    model, tok = load_model(token=os.environ.get("HF_TOKEN"))
    strings = [(tok.convert_ids_to_tokens(int(t)) or tok.decode([int(t)])).replace("▁", " ")
               for t in pool_ids]
    for c in CANDIDATES:
        got = [strings[p] for p in c["member_pos"]]
        log(f"{c['name']} tokens: {got}")
        c["tokens"] = got
    dup, _norm = build_dupmap(strings)
    key = {i: (int(counts[i]), -i) for i in range(len(pool_ids))}
    log(f"P1 dupmap: {len(dup)}/{len(pool_ids)} types are in at least one duplicate pair")

    targets_by_hash = {c["member_hash"]: c["name"] for c in CANDIDATES}
    proposal = {}
    for L in layers:
        nn = assemblies(Z2n[L], KMAX)
        hits, pop = proposal_search(nn, Z1n[L], counts, dup, key, targets_by_hash, ks)
        proposal[f"L{L}"] = dict(population_per_rung=pop, hits=hits)
        for h, name in targets_by_hash.items():
            if hits[h]:
                log(f"  L{L} proposal hits for {name}: "
                    f"{ {kk: len(vv) for kk, vv in hits[h].items()} }")
        del nn
    res["stage1_proposal"] = proposal
    # the literal question the brief asks: is the set exactly {seed + (k-1) NN}, raw?
    res["stage1_raw_knn_answer"] = {}
    for c in CANDIDATES:
        L, k, h = c["layer"], c["k"], c["member_hash"]
        seeds = proposal[f"L{L}"]["hits"][h].get(f"raw_k{k}", [])
        res["stage1_raw_knn_answer"][c["name"]] = dict(
            exactly_seed_plus_k1_nn_raw=bool(seeds), n_seeds=len(seeds),
            seeds=seeds, seed_tokens=[strings[s] for s in seeds],
            stored_sources=c["stored_sources"])

    # The route the sibling family is rebuilt along: the (k, rung) that produces this
    # candidate from the most seeds. Preferring the raw rung on a tie keeps the family as
    # close to the bare proposer as the candidate's own provenance allows.
    res["stage1_route"] = {}
    for c in CANDIDATES:
        L, h = c["layer"], c["member_hash"]
        hits = proposal[f"L{L}"]["hits"][h]
        if not hits:
            res["stage1_route"][c["name"]] = None
            log(f"  {c['name']}: NOT PRODUCED by any (k, rung) of the proposer chain")
            continue
        best = max(hits.items(),
                   key=lambda kv: (len(kv[1]), kv[0].startswith("raw"), -int(kv[0].split("_k")[1])))
        rung, kk = best[0].split("_k")
        res["stage1_route"][c["name"]] = dict(k=int(kk), rung=rung, n_seeds=len(best[1]),
                                              all_routes={a: len(b) for a, b in hits.items()})
        log(f"  {c['name']} route: k={kk} rung={rung} from {len(best[1])} seeds")
    save()
    log("stage 1 complete")

    # ================================================================================
    # STAGE 2 -- GEOMETRY TIER + MULTIPLICITY
    # ================================================================================
    geom = {}
    cell_cache = {}
    nn_cache = {}
    for c in CANDIDATES:
        L, k, mp = c["layer"], c["k"], np.asarray(c["member_pos"])
        if L not in nn_cache:
            nn_cache[L] = dict(ch1=assemblies(Z1n[L], KMAX),
                               ch2=assemblies(Z2n[L], KMAX))
        nn = nn_cache[L]["ch2"]
        g, rho = cell_stats(C1[L][mp], C2[L][mp])
        for kk in (k, k - 1):
            if (L, kk) not in cell_cache:
                cell_cache[(L, kk)] = build_cells(nn_cache[L]["ch1"], nn_cache[L]["ch2"],
                                                  C1[L], C2[L], kk)
                cc = cell_cache[(L, kk)][0]
                log(f"  cell L{L} k={kk}: ch2 p99={cc['ch2']['p99']} "
                    f"exact1={cc['ch2']['exact1_rate']} closure={cc['ch2']['closure_rate']} "
                    f"| strict p99={cc['strict']['p99']} "
                    f"exact1={cc['strict']['exact1_rate']} sat={cc['strict']['saturated']}")
        cell_k, gaps_k, rhos_k = cell_cache[(L, k)]
        cell_km1 = cell_cache[(L, k - 1)][0]
        loo = loo_report(mp, C1[L], C2[L], cell_km1["ch1"])
        loo_ch2 = loo_report(mp, C1[L], C2[L], cell_km1["ch2"])
        freq = frequency_arm(mp, C1[L], C2[L], counts, rho, n_freq_task,
                             SEED_FREQ + 100 * L + k)
        freq48 = frequency_arm(mp, C1[L], C2[L], counts, rho, N_FREQ_48,
                               SEED_FREQ + 100 * L + k)
        # cyclic orders in each channel, for the record
        o1 = np.argsort(plane_angles(C1[L][mp]))
        o2 = np.argsort(plane_angles(C2[L][mp]))
        geom[c["name"]] = dict(
            layer=L, k=k, member_hash=c["member_hash"], tokens=c["tokens"],
            gap_ratio=g, stored_gap_ratio=c["stored_gap"],
            gap_matches_stored=bool(g is not None
                                    and abs(g - c["stored_gap"]) < 1e-6),
            closure_pass=bool(g is not None and g < GAP),
            two_channel_rho=rho, stored_rho=c["stored_rho"],
            rho_matches_stored=bool(rho is not None
                                    and abs(rho - c["stored_rho"]) < 1e-9),
            channel1_cyclic_order=[c["tokens"][j] for j in o1],
            channel2_cyclic_order=[c["tokens"][j] for j in o2],
            cell_k=cell_k, cell_km1=cell_km1,
            loo_vs_ch1_cell=loo, loo_vs_ch2_cell=loo_ch2,
            order_state_vs_proposer_matched_ch2_cell=(
                "NOT ADJUDICABLE" if cell_k["ch2"]["p99"] is None else
                ("PASS" if (not cell_k["ch2"]["saturated"] and rho is not None
                            and rho > cell_k["ch2"]["p99"])
                 else ("PASS (saturated cell, rho == 1.0, exact1 < 0.01)"
                       if (cell_k["ch2"]["saturated"] and rho is not None and rho >= 1.0
                           and (cell_k["ch2"]["exact1_rate"] or 1.0) < 0.01)
                       else ("NOT ADJUDICABLE" if cell_k["ch2"]["saturated"] else "FAIL")))),
            frequency_arm_task=freq, frequency_arm_48rule=freq48)
        log(f"{c['name']}: gap {g:.4f} (stored {c['stored_gap']:.4f}) rho {rho:.6f} "
            f"(stored {c['stored_rho']:.6f}) LOO(ch1) {loo['loo_order_state']} "
            f"LOO(ch2) {loo_ch2['loo_order_state']} "
            f"freq48_pass={freq48['pass_48_rule']} "
            f"joint={freq['joint_fraction_ge_rho_and_closed']}")
        save()

    # ---- multiplicity: siblings at each candidate's (layer, k) ----
    sib = {}
    for c in CANDIDATES:
        if c["label"] == "anchor":
            continue
        L, k, mp = c["layer"], c["k"], sorted(c["member_pos"])
        nn = nn_cache[L]["ch2"]
        cell_k = cell_cache[(L, k)][0]
        cell_km1 = cell_cache[(L, k - 1)][0]["ch1"]
        cand_h = mhash(mp)

        # The sibling family must be PROPOSER-MATCHED, and the two candidates were not
        # proposed the same way: d is a raw k = 9 assembly, i is a k = 16 assembly collapsed
        # by the R15 ladder at 0.92. Taking raw k-NN sets at the candidate's own k would
        # compare i against a family it was never drawn from. So the family is rebuilt by
        # running every seed through the SAME (k, rung) chain that stage 1 found produces
        # this candidate, keeping the results that still have k members.
        route = res["stage1_route"].get(c["name"])
        k_prop, rung = (route["k"], route["rung"]) if route else (k, "raw")
        seen, closers, exact1 = {}, [], []
        for s in range(len(nn)):
            base = sorted(int(x) for x in nn[s, :k_prop])
            mem = base if rung == "raw" else dedup_r15(base, Z1n[L], counts,
                                                       float(rung[len("dedup"):]))
            mem, _ = p1_guard(mem, dup, key)
            if len(mem) != k:
                continue
            h = mhash(mem)
            if h in seen:
                continue
            seen[h] = mem
            gg, rr = cell_stats(C1[L][np.asarray(mem)], C2[L][np.asarray(mem)])
            if gg is None or gg >= GAP:
                continue
            closers.append((s, h, mem, gg))
            if rr is not None and rr >= 1.0:
                exact1.append((s, h, mem, gg))
        closers = [(s, h, i_) for s, h, i_, _g in closers]
        exact1_g = {h: g for _s, h, _i, g in exact1}
        exact1 = [(s, h, i_) for s, h, i_, _g in exact1]
        sibs_full = []
        for s, h, idx in exact1:
            if h == cand_h:
                continue
            lo = loo_report(idx, C1[L], C2[L], cell_km1)
            fq = frequency_arm(idx, C1[L], C2[L], counts, 1.0, N_FREQ_48,
                               SEED_FREQ + 100 * L + k)
            sibs_full.append(dict(seed=s, member_hash=h, member_pos=idx,
                                  tokens=[strings[j] for j in idx],
                                  gap_ratio=float(exact1_g[h]),
                                  loo_closure_pass=lo["loo_closure_pass"],
                                  loo_order_state=lo["loo_order_state"],
                                  freq_pass_48rule=fq["pass_48_rule"],
                                  freq_p99=fq["p99"]))
        n_cand_grade = sum(1 for x in sibs_full
                           if x["loo_closure_pass"] and x["freq_pass_48rule"])
        n_strict_loo = sum(1 for x in sibs_full
                           if x["loo_order_state"] == "PASS" and x["freq_pass_48rule"])
        sib[c["name"]] = dict(
            layer=L, k=k, proposal_route=dict(k=k_prop, rung=rung),
            n_distinct_assemblies=len(seen), n_closers=len(closers),
            n_rho_exactly_1_and_closed=len(exact1),
            route_exact1_rate=(float(len(exact1) / len(closers)) if closers else None),
            route_exact1_rate_note="exact-1 rate within the candidate's OWN proposal route "
                                   "(this (k, rung) chain), closure-conditioned -- the "
                                   "honest proposer-matched false rate for this candidate",
            candidate_in_that_population=bool(cand_h in {h for _s, h, _i in exact1}),
            n_siblings_candidate_grade=n_cand_grade,
            n_siblings_strict_LOO_pass=n_strict_loo,
            note=("candidate-grade = rho 1.0, closure, LOO CLOSURE and the frequency arm -- "
                  "the exact bar the candidate itself clears, since the candidate's own LOO "
                  "ORDER state is NOT ADJUDICABLE (the k-1 cell is saturated). "
                  "strict_LOO_pass additionally requires LOO order PASS."),
            siblings=sibs_full)
        log(f"{c['name']} multiplicity: {len(seen)} distinct assemblies, "
            f"{len(closers)} close, {len(exact1)} have rho=1.0, "
            f"{n_cand_grade} candidate-grade siblings, {n_strict_loo} strict-LOO")
        # the sibling pool the behavioural null (iv) draws from
        c["_sibling_pool"] = [idx for _s, h, idx in closers if h != cand_h]
    res["stage2_geometry"] = geom
    res["stage2_multiplicity"] = sib
    save()
    log("stage 2 complete")

    if a.skip_behavioural:
        save("COMPLETE_GEOMETRY_ONLY")
        return

    # ================================================================================
    # STAGE 3 -- BEHAVIOURAL TIER
    # ================================================================================
    tokens2d = rig.load_tokens2d(REPO / CACHE_ROOT)
    eligible = rig.eligible_docs(tokens2d, ())
    probe = Runner(model, tok, 6, DEVICE)
    blk0 = probe.prepare([np.zeros(0, dtype=np.int64)], int(pool_ids[0]))
    lg0, hn0 = probe.baseline(blk0)
    lg_z = probe.steered(blk0, hn0, np.zeros(1152), 0.0)
    hook_max = float((lg0 - lg_z).abs().max())
    log(f"hook identity: max |delta logit| = {hook_max:.3e}")
    assert hook_max == 0.0, "zero-delta hook is not an identity"
    res["hook_identity_max_logit_diff"] = hook_max

    rng_dirs = np.random.default_rng(SEED_DIRS)
    rng_null = np.random.default_rng(SEED_NULLSETS)
    rng_sib = np.random.default_rng(SEED_SIBLINGS)
    behav = {}
    for c in CANDIDATES:
        name, L, k = c["name"], c["layer"], c["k"]
        mp = np.asarray(sorted(c["member_pos"]))
        geo = plane_of(C1[L][mp])
        runner = Runner(model, tok, L, DEVICE)
        t_set = time.time()
        log(f"{name}: L{L} k={k} starting (gap {geo['gap_ratio']:.4f})")

        tang = embed_1152(geo["tangent"])
        dirs = {}
        rd = np.stack([random_orth_dirs(tang[i], n_rand, rng_dirs) for i in range(k)])
        for j in range(n_rand):
            dirs[f"rand{j}"] = rd[:, j, :]
        u1, u2, th = geo["u1"], geo["u2"], geo["theta"]
        dirs["rot_p90"] = embed_1152(-(np.cos(th)[:, None] * u1[None, :])
                                     - (np.sin(th)[:, None] * u2[None, :]))
        dirs["rot_m90"] = embed_1152((np.cos(th)[:, None] * u1[None, :])
                                     + (np.sin(th)[:, None] * u2[None, :]))

        rows_nat = eval_set(runner, geo, mp, pool_ids, tokens2d, eligible, n_prompts,
                            EPS_GRID, np.random.default_rng(SEED_CONTEXT + 100 * L + k))
        rows_dirs = eval_set(runner, geo, mp, pool_ids, tokens2d, eligible, n_prompts_null,
                             (), np.random.default_rng(SEED_CONTEXT + 50_000 + 100 * L + k),
                             extra_dirs=dirs)
        rows_matched = eval_set(runner, geo, mp, pool_ids, tokens2d, eligible,
                                n_prompts_null, (EPS_NULL,),
                                np.random.default_rng(SEED_CONTEXT + 50_000 + 100 * L + k))
        rows_bos = eval_set(runner, geo, mp, pool_ids, tokens2d, eligible, 1, EPS_GRID,
                            np.random.default_rng(SEED_CONTEXT + 100 * L + k), arm="bos")
        nat = {f"eps={e}": summarise(rows_nat, f"eps={e}") for e in EPS_GRID}
        bos = {f"eps={e}": summarise(rows_bos, f"eps={e}") for e in EPS_GRID}
        null_i = {f"rand{j}": summarise(rows_dirs, f"rand{j}") for j in range(n_rand)}
        null_ii = {nm: summarise(rows_dirs, nm) for nm in ("rot_p90", "rot_m90")}
        matched = summarise(rows_matched, f"eps={EPS_NULL}")
        log(f"  {name} candidate arms done "
            f"({runner.n_forward} forwards, {time.time()-t_set:.0f}s); "
            f"meanD@{EPS_NULL}={nat[f'eps={EPS_NULL}']['mean_delta']:+.4f}")

        def score_sets(picks, tag, seed_base):
            out = []
            for j, pick in enumerate(picks):
                pick = np.asarray(sorted(int(x) for x in pick))
                Rn = C1[L][pick]
                if is_degenerate(Rn):
                    continue
                gn = plane_of(Rn)
                rr = eval_set(runner, gn, pick, pool_ids, tokens2d, eligible,
                              n_prompts_null, (EPS_NULL,),
                              np.random.default_rng(seed_base + 1000 * j))
                sm = summarise(rr, f"eps={EPS_NULL}")
                sm["member_hash"] = mhash(pick)
                sm["gap_ratio"] = gn["gap_ratio"]
                out.append(sm)
                if (j + 1) % 10 == 0:
                    log(f"    {name} {tag} {len(out)}/{len(picks)} "
                        f"({time.time()-t0:.0f}s elapsed)")
            return out

        # null (iii): random k-sets
        picks_iii = []
        while len(picks_iii) < n_nullsets:
            pick = sorted(rng_null.choice(len(pool_ids), size=k, replace=False).tolist())
            if pick == sorted(mp.tolist()) or is_degenerate(C1[L][np.asarray(pick)]):
                continue
            picks_iii.append(pick)
        null_iii = score_sets(picks_iii, "null-iii", SEED_NULLSETS + 10_000 * L + k)

        # null (iv): sibling sets -- proposer-matched
        null_iv = []
        pool_sib = c.get("_sibling_pool") or []
        if pool_sib:
            sel = rng_sib.choice(len(pool_sib), size=min(n_nullsets, len(pool_sib)),
                                 replace=False)
            null_iv = score_sets([pool_sib[int(x)] for x in sel], "null-iv",
                                 SEED_SIBLINGS + 10_000 * L + k)

        m_iii = np.array([x["mean_delta"] for x in null_iii])
        p99_iii = float(np.percentile(m_iii, PCT)) if len(m_iii) else None
        m_iv = np.array([x["mean_delta"] for x in null_iv])
        p99_iv = float(np.percentile(m_iv, PCT)) if len(m_iv) else None
        rand_means = np.array([null_i[f"rand{j}"]["mean_delta"] for j in range(n_rand)])
        sc_ok = any(nat[f"eps={e}"]["sign_consistency"] >= (k - 1) / k for e in EPS_GRID)
        d_pin = nat[f"eps={EPS_NULL}"]["mean_delta"]
        above_iii = bool(p99_iii is not None and d_pin > p99_iii)
        above_i = bool(d_pin > rand_means.max())
        passed = bool(sc_ok and above_iii and above_i)

        # the realised false rate: how often the sibling null itself would PASS this rule.
        # Siblings are scored at one eps and 12 prompts, so their sign-consistency has one
        # chance where the candidate has three; the estimate is therefore a LOWER bound.
        sib_pass = [bool(x["sign_consistency"] >= (k - 1) / k
                         and p99_iii is not None and x["mean_delta"] > p99_iii)
                    for x in null_iv]
        behav[name] = dict(
            layer=L, k=k, member_hash=c["member_hash"], label=c["label"],
            geometry=dict(pc_var_top2=geo["pc_var_top2"], gap_ratio=geo["gap_ratio"],
                          radius_ratio=geo["radius_ratio"], circ_resid=geo["circ_resid"]),
            natural=nat, bos_only=bos, matched_budget_at_eps_null=matched,
            null_i_random_orth=null_i, null_ii_rotations=null_ii,
            null_iii_random_sets=null_iii, null_iv_sibling_sets=null_iv,
            null_iii_p99=p99_iii, null_iii_max=float(m_iii.max()) if len(m_iii) else None,
            null_iii_mean=float(m_iii.mean()) if len(m_iii) else None,
            null_iv_p99=p99_iv, null_iv_max=float(m_iv.max()) if len(m_iv) else None,
            null_iv_mean=float(m_iv.mean()) if len(m_iv) else None,
            n_null_iv=len(null_iv),
            null_i_max=float(rand_means.max()),
            above_null_iv_p99=bool(p99_iv is not None and d_pin > p99_iv),
            matched_above_null_iii_p99=bool(p99_iii is not None
                                            and matched["mean_delta"] > p99_iii),
            monotonic_in_eps=bool(nat["eps=0.005"]["mean_delta"]
                                  < nat["eps=0.01"]["mean_delta"]
                                  < nat["eps=0.02"]["mean_delta"]),
            PASS=passed,
            pass_detail=dict(sign_consistency_ok=bool(sc_ok), above_null_iii_p99=above_iii,
                             above_all_null_i=above_i, mean_delta_at_eps_null=d_pin),
            realised_false_rate_sibling_null=dict(
                n=len(null_iv), n_pass=int(sum(sib_pass)),
                rate=float(np.mean(sib_pass)) if sib_pass else None,
                note="fraction of proposer-matched sibling sets that would themselves clear "
                     "the behavioural PASS rule. Siblings are scored at one eps and "
                     f"{n_prompts_null} prompts where the candidate gets three eps and "
                     f"{n_prompts} prompts, so this is a LOWER bound on the false rate."),
            per_member=dict(natural=rows_nat, null_dirs=rows_dirs, bos=rows_bos),
            n_forward=runner.n_forward, seconds=round(time.time() - t_set, 1))
        log(f"{name}: PASS={passed} meanD={d_pin:+.4f} matched={matched['mean_delta']:+.4f} "
            f"sc={[round(nat[f'eps={e}']['sign_consistency'],3) for e in EPS_GRID]} "
            f"iii_p99={p99_iii} iv_p99={p99_iv} i_max={rand_means.max():+.4f} "
            f"({runner.n_forward} forwards, {time.time()-t_set:.0f}s)")
        res["stage3_behavioural"] = behav
        save()

        if c["label"] == "anchor" and not passed:
            res["anchor_gate"] = dict(
                passed=False,
                message="ANCHOR FAILED: the strict-12 at L6 did not PASS the behavioural "
                        "rule under this rig. Per the brief, the run stops here and "
                        "reports; no candidate verdict can be trusted from a rig that "
                        "cannot recover the established positive control.")
            save("STOPPED_ANCHOR_FAILED")
            log("ANCHOR FAILED -- stopping")
            return
        if c["label"] == "anchor":
            res["anchor_gate"] = dict(passed=True,
                                      message="strict-12 at L6 PASSES under this rig")

    res["total_forwards"] = int(sum(v["n_forward"] for v in behav.values()))
    res["finished_at"] = datetime.now().isoformat(timespec="seconds")
    save("COMPLETE")
    log(f"wrote {out_path}")


if __name__ == "__main__":
    main()
