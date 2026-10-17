"""48 -- Phase 5 §2.1: proposer A (mutual two-channel agreement graph) and proposer B
(channel-2 cluster-like assembly), adjudicated by the imported 42/43b tier.

Nothing about the tier changes. What changes is where the candidates come from:
  A  edge i-j iff j is in the top-m of i in BOTH channel 1 (type mean) and channel 2
     (lag+1 successor delta) AND i is in the top-m of j in both; proposals = connected
     components of the mutual graph with >= 9 members, plus the R15 dedup ladder.
  B  for every type s and every k in 9..20, S(s,k) = {s} + the k-1 nearest by centred
     cosine in the channel-2 delta cloud, plus the R15 dedup ladder.

Rules in force: R1-R12 (Phase 3/4), Addenda K-O, and Phase 5 R13-R18.
  R13  verdict-bearing null = proposer-matched permuted search: both proposers AND the tier
       re-run under within-(token class x log10-count decile) permutations of channel-2 rows.
  R14  PASS := closure < 2.5 (post-R10, evaluated FIRST) AND [rho >= the screen's maximum
       tie class OR rho > the max-seed cell p99] AND LOO on both AND the frequency-matched
       arm AND permuted-search mean < 0.5.
  R15  dedup: drop the lower-count member of any within-proposal pair with channel-1 centred
       cosine > 0.90; the 0.88 / 0.90 / 0.92 ladder is reported.
  R16  channel 3 = OV-image retention (weights only, no forwards); declared conjunct for
       k <= 10, reported for all k.
  R17  CPU only; no MPS in this phase.
  R18  strict-12 must fall out of each proposer BLIND at L6 and PASS at L6; L7 reported.

The stub is written to --out before any artifact is opened.
"""
import argparse, hashlib, importlib.util, json, os, sys, time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HOME", str(REPO / "cache/hf"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
sys.path.insert(0, str(REPO / "src"))

FULLN_RAW = "results/phase3_token_type_mean_resid_fulln.npz"
ROBUST_DELTAS = "results/phase3_lag_action_robust_deltas.npz"
WEIGHTS = ("cache/hf/hub/models--google--gemma-3-1b-pt/snapshots/"
           "fcf18a2a879aab110ca39f8bffbccd5d49d8eb29/model.safetensors")

LAYERS = (6, 7)
M_GRID = (8, 10, 12)
K_RANGE = tuple(range(9, 21))
DEDUP_LADDER = (0.88, 0.90, 0.92)
DEDUP_REGISTERED = 0.90
GAP = 2.5
KMIN = 9
PCT = 99.0
SEED_TRIPLE = (0, 1, 2)
N_CELL = 6_000
N_CELL_CAND = 20_000
N_FREQ = 200
FREQ_TOL = 0.25
N_OV_NULL = 500
PLANT_JITTERS = (0.05, 0.10, 0.15, 0.20, 0.30, 0.50)
PLANT_OFFSETS = (1.0, 2.0, 3.0)
PLANT_K = 12
STRICT12_IDS = [3353, 5142, 5261, 5365, 5505, 5729, 5857, 6027, 6198, 6243, 6384, 7250]
STRICT12_HASH = "a36a4f4f3e3b"


def _load(name, rel):
    sp = importlib.util.spec_from_file_location(name, str(REPO / rel))
    m = importlib.util.module_from_spec(sp)
    sp.loader.exec_module(m)
    return m


def mhash(members):
    return hashlib.md5(",".join(str(int(x)) for x in sorted(members)).encode()).hexdigest()[:12]


# =====================================================================================
# declaration
# =====================================================================================
def declared_rules(cfg):
    return dict(
        prereg="docs/phase5_prereg_2026-08-22.md (frozen 2026-08-22), section 2.1",
        addenda_in_force="Phase 3 R1-R6, Phase 4 R7-R12, Addenda K-O, Phase 5 R13-R18",
        channels=("channel 1 = per-token-type mean raw residual (results/phase3_token_type_"
                  "mean_resid_fulln.npz), post-R10 (coordinate 1038 dropped at L4-L9); "
                  "channel 2 = identity-removed lag+1 successor-delta means "
                  "(results/phase3_lag_action_robust_deltas.npz), same drop, cast float32 "
                  "then float64 before any fit (L5)."),
        proposer_A=("mutual two-channel k-nearest-neighbour agreement graph. For each type i "
                    "let N1_m(i) and N2_m(i) be its m nearest by POOL-CENTRED cosine in "
                    "channel 1 and channel 2. Directed edge i->j iff j in N1_m(i) & N2_m(i). "
                    "Graph = the mutual part (i->j and j->i). Proposals = connected "
                    "components with >= 9 members, for m in " + repr(M_GRID) + ", at each "
                    "layer, plus each component's R15 dedup ladder image. No labels, no "
                    "plane fit, no energy ranking enters the proposer."),
        proposer_B=("channel-2 cluster-like assembly. For every one of the 3,795 types s and "
                    "every k in 9..20, S(s,k) = {s} + the k-1 nearest to s by pool-centred "
                    "cosine in the channel-2 delta cloud. Deduplicated by member hash, plus "
                    "the R15 dedup ladder image."),
        R13_null=("proposer-matched permuted search: channel-2 ROWS are permuted within "
                  "(token class x log10-count decile) blocks and BOTH proposers and the tier "
                  "are re-run from scratch. Token class = 42.class_features -> "
                  "lead-space x capitalised (42.resolve_blocks), crossed with log10-count "
                  "deciles of the pool. Thresholds are NOT recomputed on the permuted "
                  "channel (42's thr_override convention): the question is whether the "
                  "permuted channel produces a passer at the REAL tier's threshold."),
        R14_PASS=("closure < 2.5 post-R10 evaluated FIRST (a closure failure is FAIL "
                  "regardless of cell saturation, M1); then the order conjunct = "
                  "[rho >= the screen's own maximum rho tie class for that (layer, "
                  "proposer)] OR [rho > the max-seed cell p99 (L2 band upper edge)]; then "
                  "LOO closure and LOO order at the (k-1) cell; then the frequency-matched "
                  "arm (200 count-matched k-sets, +-0.25 log10, candidate must exceed the "
                  "arm's 99th percentile); then permuted-search mean < 0.5."),
        L1_saturation=("cells with conditional p99 == 1.0: the order conjunct passes iff "
                       "rho == 1.0 AND the cell's conditional exact-1.0 rate < 0.01; else "
                       "NOT ADJUDICABLE in that cell."),
        L2_band="every cell's p99 under 3 RNG seeds; rho inside [min, max] is MARGINAL.",
        cells=("cluster-like conditional nulls per (layer, k) are built in BOTH coordinate "
               "systems -- seeded and grown in channel 1 (42's registered null) and seeded "
               "and grown in channel 2 (proposer-B-matched). The threshold used is the "
               "MAX of the two, i.e. the stricter; both are reported. "
               f"{N_CELL} draws per (layer, k, seed) for the screen; any candidate that "
               f"reaches PASS has its cells rebuilt at {N_CELL_CAND} (K2)."),
        R15_dedup=("within a proposal, repeatedly drop the lower-count member of the pair "
                   "with the largest channel-1 pool-centred cosine while that cosine "
                   "exceeds t; t in " + repr(DEDUP_LADDER) + f"; registered t = "
                   f"{DEDUP_REGISTERED}. The deduped image of every proposal is itself a "
                   "proposal and is counted in the multiplicity."),
        R16_channel3=("OV-image retention: for each of the layer's 4 attention heads, the "
                      "fraction of the candidate's channel-1 top-2 plane retained by the "
                      "column space of o_proj[:, 256h:256(h+1)] (256-d of 1152); channel 3 "
                      "= the mean over the 4 heads. Null = 500 random k-type planes at the "
                      "same layer. Declared conjunct only for k <= 10; reported for all k. "
                      "Weights read from the local snapshot; no forward pass."),
        R17="CPU only; numpy/safetensors; no MPS tensor is created anywhere in this script.",
        R18_positive_control=("strict-12 (pool positions of token ids "
                              f"{STRICT12_IDS}, hash {STRICT12_HASH}) must be proposed "
                              "BLIND -- as a component (A) or as a seed's k-1 nearest "
                              "neighbourhood (B), before or after R15 dedup -- at L6, and "
                              "must PASS at L6. L7 is reported; L3/N7 expect MARGINAL."),
        plants=("REPLACE-not-add with carrier offset (Phase 5 section 0 correction): a "
                "random unit direction u; the carrier mean is displaced to offset*u where "
                "offset is a multiple of the MEDIAN random-12 carrier offset in that "
                "channel; a ring of the months' own radius is placed in a random 2-plane "
                "orthogonal to u; the carrier's own residual is projected off {u, plane} "
                "and rescaled so the planted set's top-2 fraction matches the months'; the "
                "SAME cyclic order with INDEPENDENT jitter draws is used in both channels "
                "(the harder version of 'same jittered angles'). Recall is measured for A "
                "and B on the same plants. A dispersed-carrier plant (no offset) is "
                "reported and is expected to give 0."),
        deviations=[
            "D1 (cell choice for the LOO conjunct). The k cell uses the STRICTER of the "
            "ch1-drawn and ch2-drawn cluster-like nulls (proposer-matched, per the prereg "
            "cells clause). The (k-1) LOO cell uses the ch1-drawn null -- 42/43b's "
            "registered cluster-like null -- because LOO subsets are not proposals of "
            "either proposer, so the proposer-matched null does not apply to them. "
            "Measured reason: at L6, 6,000 draws, the ch2-drawn k = 11 cell is saturated "
            "(p99 = 1.0, conditional exact-1.0 rate 0.021) while the ch1-drawn k = 11 cell "
            "gives p99 = 0.9818; under the strict rule every k = 12 candidate, INCLUDING "
            "the strict-12 positive control, is NOT ADJUDICABLE at its LOO cell for want "
            "of a null, not for want of order. The strict-cell LOO states are co-reported "
            "for every candidate as loo_states_strict_cell.",
            "D2 (permutation budget). R13 asks for 20 permutations for both proposers. "
            "Proposer B re-screens ~73,000 unique proposals per run; 20 x B does not fit "
            "the phase's CPU budget alongside 43d and 45b. Budget as run is recorded in "
            "config.perms_A / config.perms_B; any shortfall against 20 is stated in the "
            "summary.",
            "D4 (exact cluster-like null). 42/43b sampled the cluster-like null with "
            "replacement (6,000 / K2's 20,000 draws). The draw is a deterministic function "
            "of its seed type, so the null has exactly 3,795 distinct members per "
            "(layer, k, coordinate system); this script ENUMERATES all of them. The p99 is "
            "therefore exact, L2's 3-seed band collapses to a point (reported as [p, p]), "
            "and K2's draw-count question does not arise. Measured motivation: at 6,000 "
            "sampled draws the L6 k = 11 channel-1 cell saturated in 1 of 3 seeds "
            "(band [0.9818, 1.0]), which made every k = 12 candidate NOT ADJUDICABLE at "
            "its LOO cell -- a Monte-Carlo artifact, not a property of the null.",
            "D5 (per-k tie class). R14's 'the screen's maximum tie class' is evaluated per "
            "(layer, proposer, k). The screen enumerates a different family at every k and "
            "a rho of 1.0 reachable at k = 9 carries no information about k = 12; a single "
            "screen-wide maximum would make the clause unsatisfiable at every k above the "
            "smallest. Per-k maxima and tie sizes are reported for every cell.",
            "D3 (plant budget). 100 plants per cell at offset 1x; "
            "config.plants_per_cell_extra_offsets at 2x and 3x.",
        ],
        exposure=("Phase 5 section 0. For A: component-size counts only (62 at L6 m=10; "
                  "40-67 at L7) and the months' own component. For B: at k=12, the counts "
                  "of tier-reaching unique sets (3 at L6, 2 at L7) and their OV scalars, no "
                  "membership. No non-calendar proposal's membership has been read."),
    )


# =====================================================================================
# geometry helpers (thin wrappers on the imported tier)
# =====================================================================================
def centred_unit(X):
    Z = X - X.mean(0)
    n = np.linalg.norm(Z, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return Z / n


def topm(Z, m):
    """Indices of the m nearest by centred cosine (self excluded), per row."""
    S = (Z @ Z.T).astype(np.float32)
    np.fill_diagonal(S, -9.0)
    idx = np.argpartition(-S, m, axis=1)[:, :m]
    rows = np.arange(len(Z))[:, None]
    order = np.argsort(-S[rows, idx], axis=1)
    return idx[rows, order], S


def _mutual(o1, o2, m, n):
    r = np.arange(n)[:, None]
    A = np.zeros((n, n), bool); A[r, o1[:, :m]] = True
    B = np.zeros((n, n), bool); B[r, o2[:, :m]] = True
    D = A & B
    return D & D.T


def agreement_components(Z1, Z2, m, min_size=KMIN):
    import scipy.sparse as sp
    n = len(Z1)
    o1, _ = topm(Z1, m)
    o2, _ = topm(Z2, m)
    Mut = _mutual(o1, o2, m, n)
    ncomp, lab = sp.csgraph.connected_components(sp.csr_matrix(Mut), directed=False)
    sizes = np.bincount(lab, minlength=ncomp)
    comps = [np.where(lab == c)[0] for c in np.where(sizes >= min_size)[0]]
    return comps, sizes, lab


def channel2_assembly(Z2, ks):
    """{k: list of index arrays} -- every type as seed, k-1 nearest in the delta cloud."""
    n = len(Z2)
    kmax = max(ks)
    S = (Z2 @ Z2.T).astype(np.float32)
    np.fill_diagonal(S, 2.0)
    idx = np.argpartition(-S, kmax, axis=1)[:, :kmax]
    rows = np.arange(n)[:, None]
    order = np.argsort(-S[rows, idx], axis=1)
    nn = idx[rows, order]                      # column 0 is the seed itself
    del S
    return {k: nn[:, :k] for k in ks}


def dedup(members, Z1n, counts, thr):
    """R15. Z1n = pool-centred, unit-normalised channel-1 rows."""
    S = list(int(x) for x in members)
    while len(S) > 3:
        C = (Z1n[S] @ Z1n[S].T).astype(np.float64)
        np.fill_diagonal(C, -9.0)
        i, j = np.unravel_index(int(np.argmax(C)), C.shape)
        if C[i, j] <= thr:
            break
        drop = S[i] if counts[S[i]] < counts[S[j]] else S[j]
        S = [x for x in S if x != drop]
    return sorted(S)


# =====================================================================================
# cells and verdicts (43b's L1/L2, both coordinate systems)
# =====================================================================================
def build_cell(S42, C1, C2, orders, layer, k, seed, n_draws):
    """EXACT enumeration of the cluster-like null.

    42's cluster-like draw is seed s -> {s} + the k-1 nearest to s, a DETERMINISTIC function
    of s, so the null has exactly n_types distinct members per (layer, k). Enumerating all
    3,795 of them is the exact distribution 42/43b were sampling with replacement (6,000 /
    20,000 draws); it removes both the Monte-Carlo seed variability that L2's 3-seed band
    was written for and K2's draw-count question.
    """
    gaps, rho_cond = [], []
    n = len(orders)
    for s in range(n):
        idx = orders[s, :k]
        g, r = S42.cell_stats(C1[idx], C2[idx])
        if g is None:
            continue
        gaps.append(g)
        if r is not None and g < GAP:
            rho_cond.append(r)
    rc = np.asarray(rho_cond)
    gaps = np.asarray(gaps)
    p99 = float(np.percentile(rc, PCT, method="higher")) if len(rc) else None
    return dict(k=int(k), seed=int(seed), n_draws=int(n), enumerated=True,
                n_conditional=int(len(rc)),
                closure_rate=float((gaps < GAP).mean()) if len(gaps) else None,
                p99=p99, exact1_rate=float((rc >= 1.0).mean()) if len(rc) else None,
                saturated=bool(p99 is not None and p99 >= 1.0))


class Cells:
    def __init__(self, S42, C1, C2, Z1, Z2, layer):
        self.S42, self.C1, self.C2, self.layer = S42, C1, C2, layer
        kmax = max(max(K_RANGE), max(M_GRID)) + 1
        self.orders = {"ch1": _order_with_self(Z1, kmax),
                       "ch2": _order_with_self(Z2, kmax)}
        self.cache = {}

    def get(self, k, n_draws=N_CELL):
        key = (k, n_draws)
        if key in self.cache:
            return self.cache[key]
        out = {}
        for kind in ("ch1", "ch2"):
            per = [build_cell(self.S42, self.C1, self.C2, self.orders[kind], self.layer, k,
                              0, n_draws)]
            ps = [c["p99"] for c in per if c["p99"] is not None]
            out[kind] = dict(per_seed=per, p99_min=min(ps) if ps else None,
                             p99_max=max(ps) if ps else None,
                             saturated=any(c["saturated"] for c in per),
                             exact1_rate=float(np.mean([c["exact1_rate"] for c in per
                                                        if c["exact1_rate"] is not None]))
                             if any(c["exact1_rate"] is not None for c in per) else None,
                             closure_rate=float(np.mean([c["closure_rate"] for c in per
                                                         if c["closure_rate"] is not None]))
                             if any(c["closure_rate"] is not None for c in per) else None)
        both = [out[x]["p99_max"] for x in ("ch1", "ch2") if out[x]["p99_max"] is not None]
        out["k"] = int(k)
        out["p99_max_strict"] = max(both) if both else None
        out["p99_min_strict"] = max([out[x]["p99_min"] for x in ("ch1", "ch2")
                                     if out[x]["p99_min"] is not None], default=None)
        out["saturated"] = out["ch1"]["saturated"] or out["ch2"]["saturated"]
        out["exact1_rate"] = max([out[x]["exact1_rate"] for x in ("ch1", "ch2")
                                  if out[x]["exact1_rate"] is not None], default=None)
        self.cache[key] = out
        return out


def _order_with_self(Z, kmax):
    """Row s -> [s, then the kmax-1 nearest by centred cosine]. 42's cluster-like draw."""
    S = (Z @ Z.T).astype(np.float32)
    np.fill_diagonal(S, 2.0)
    idx = np.argpartition(-S, kmax, axis=1)[:, :kmax]
    r = np.arange(len(Z))[:, None]
    return idx[r, np.argsort(-S[r, idx], axis=1)]


def cell_view(cell, kind):
    """'strict' = max over the ch1- and ch2-drawn cluster-like cells; 'ch1' = 42's
    registered null (cluster-like seeded and grown in channel 1)."""
    if cell is None:
        return None
    if kind == "strict":
        return cell
    c = cell[kind]
    return dict(p99_min_strict=c["p99_min"], p99_max_strict=c["p99_max"],
                saturated=c["saturated"], exact1_rate=c["exact1_rate"], k=cell["k"])


def order_state(rho, cell, tie_floor):
    """R14 order conjunct for one cell. (state, detail)."""
    if rho is None:
        return "NOT ADJUDICABLE", "no rho"
    if cell is None or cell["p99_max_strict"] is None:
        return "NOT ADJUDICABLE", "empty conditional null"
    if tie_floor is not None and rho >= tie_floor - 1e-12:
        return "PASS", f"rho {rho:.4f} in the screen's maximum tie class ({tie_floor:.4f})"
    if cell["saturated"]:
        if rho >= 1.0 and (cell["exact1_rate"] if cell["exact1_rate"] is not None
                           else 1.0) < 0.01:
            return "PASS", f"L1 saturated cell, rho == 1.0, exact1 {cell['exact1_rate']:.4f}"
        return "NOT ADJUDICABLE", (f"L1 saturated cell (p99 = 1.0, exact1 "
                                   f"{cell['exact1_rate']}), rho {rho:.4f}")
    hi, lo = cell["p99_max_strict"], cell["p99_min_strict"]
    if rho > hi:
        return "PASS", f"rho {rho:.4f} > max seed p99 {hi:.4f}"
    if rho >= lo:
        return "MARGINAL", f"rho {rho:.4f} inside seed p99 band [{lo:.4f}, {hi:.4f}]"
    return "FAIL", f"rho {rho:.4f} < min seed p99 {lo:.4f}"


def adjudicate(S42, members, C1, C2, cells, tie_floor, want_loo=True, n_draws=N_CELL):
    """Closure FIRST (M1), then the R14 order conjunct, then LOO on both."""
    members = np.asarray(sorted(int(x) for x in members))
    k = len(members)
    g = S42.gap_ratio_of(C1[members])
    rec = dict(k=int(k), gap_ratio=None if g is None else float(g), closure_pass=False,
               two_channel_rho=None, order_state=None, order_detail=None,
               loo_closure_pass=None, loo_order_state=None, loo_rho_min=None,
               loo_gap_max=None, verdict="FAIL", adjudicable=bool(k >= KMIN))
    if g is None or g >= GAP:
        rec["verdict"] = "FAIL"
        rec["order_detail"] = "closure first: gap_ratio >= 2.5 (or degenerate)"
        return rec
    rec["closure_pass"] = True
    if k < KMIN:
        rec["verdict"] = "NOT ADJUDICABLE"
        return rec
    _, rho = S42.cell_stats(C1[members], C2[members])
    rec["two_channel_rho"] = None if rho is None else float(rho)
    cell = cells.get(k, n_draws)
    st, det = order_state(rho, cell_view(cell, "strict"), tie_floor)
    rec["order_state"], rec["order_detail"] = st, det
    rec["cell_p99_band"] = [cell["p99_min_strict"], cell["p99_max_strict"]]
    rec["cell_saturated"] = cell["saturated"]
    if st in ("FAIL", "NOT ADJUDICABLE"):
        rec["verdict"] = st
        return rec
    if not want_loo:
        rec["verdict"] = st
        return rec
    cell_loo = cells.get(k - 1, n_draws)
    loo_states, loo_gaps, loo_rhos, loo_states_strict = [], [], [], []
    for i in range(k):
        sub = np.delete(members, i)
        gi = S42.gap_ratio_of(C1[sub])
        loo_gaps.append(None if gi is None else float(gi))
        if gi is None or gi >= GAP:
            loo_states.append("FAIL"); loo_states_strict.append("FAIL")
            loo_rhos.append(None); continue
        _, ri = S42.cell_stats(C1[sub], C2[sub])
        loo_rhos.append(None if ri is None else float(ri))
        loo_states.append(order_state(ri, cell_view(cell_loo, "ch1"), None)[0])
        loo_states_strict.append(order_state(ri, cell_view(cell_loo, "strict"), None)[0])
    rec["loo_gap_max"] = max([x for x in loo_gaps if x is not None], default=None)
    rec["loo_rho_min"] = min([x for x in loo_rhos if x is not None], default=None)
    rec["loo_closure_pass"] = all(x is not None and x < GAP for x in loo_gaps)
    rec["loo_states"] = loo_states
    rec["loo_states_strict_cell"] = loo_states_strict
    rec["loo_cell_p99_ch1"] = [cell_loo["ch1"]["p99_min"], cell_loo["ch1"]["p99_max"]]
    rec["loo_cell_p99_strict"] = [cell_loo["p99_min_strict"], cell_loo["p99_max_strict"]]
    rec["loo_gap_ratios"] = [None if x is None else round(x, 4) for x in loo_gaps]
    rec["loo_rhos"] = [None if x is None else round(x, 4) for x in loo_rhos]
    if not rec["loo_closure_pass"]:
        rec["verdict"] = "FAIL"; rec["loo_order_state"] = "closure fail"
        return rec
    if "FAIL" in loo_states:
        rec["loo_order_state"] = "FAIL"; rec["verdict"] = "FAIL"
    elif "NOT ADJUDICABLE" in loo_states:
        rec["loo_order_state"] = "NOT ADJUDICABLE"; rec["verdict"] = "NOT ADJUDICABLE"
    elif "MARGINAL" in loo_states:
        rec["loo_order_state"] = "MARGINAL"; rec["verdict"] = "MARGINAL"
    else:
        rec["loo_order_state"] = "PASS"
        rec["verdict"] = "PASS" if st == "PASS" else st
    return rec


def frequency_arm(S42, members, C1, C2, counts, rho, n=N_FREQ, seed=0):
    """200 count-matched k-sets (+-0.25 log10). Candidate must beat the arm's p99."""
    rng = np.random.default_rng(seed)
    k = len(members)
    lc = np.log10(np.maximum(counts, 1.0))
    order = np.argsort(lc, kind="stable")
    lcs = lc[order]
    tgt = lc[np.asarray(members)]
    lo = np.searchsorted(lcs, tgt - FREQ_TOL, side="left")
    hi = np.searchsorted(lcs, tgt + FREQ_TOL, side="right")
    rhos, closed = [], 0
    tries = 0
    while len(rhos) < n and tries < 20 * n:
        tries += 1
        idx, seen, ok = [], set(), True
        for a, b in zip(lo, hi):
            if b <= a:
                ok = False; break
            for _try in range(30):
                cand = int(order[rng.integers(a, b)])
                if cand not in seen:
                    seen.add(cand); idx.append(cand); break
            else:
                ok = False; break
        if not ok or len(idx) != k:
            continue
        idx = np.asarray(idx)
        g, r = S42.cell_stats(C1[idx], C2[idx])
        if g is None or r is None:
            continue
        rhos.append(r)
        if g < GAP:
            closed += 1
    rhos = np.asarray(rhos)
    p99 = float(np.percentile(rhos, PCT, method="higher")) if len(rhos) else None
    return dict(n=int(len(rhos)), p99=p99, median=float(np.median(rhos)) if len(rhos) else None,
                closure_rate=float(closed / max(len(rhos), 1)),
                pass_=bool(p99 is not None and rho is not None and rho > p99))


class Channel3:
    """OV-image retention, weights only (R16)."""

    def __init__(self, layers, keep):
        """keep[L] = the post-R10 coordinate index set; the OV image is restricted to the
        same coordinates as the channels, so the retention is measured in one space."""
        from safetensors import safe_open
        self.Q = {}
        with safe_open(str(REPO / WEIGHTS), "pt") as f:
            for L in layers:
                W = f.get_tensor(f"model.layers.{L}.self_attn.o_proj.weight").float().numpy()
                W = W[np.asarray(keep[L]), :]
                qs = []
                for h in range(4):
                    B = W[:, 256 * h:256 * (h + 1)].astype(np.float64)
                    q, _ = np.linalg.qr(B)
                    qs.append(q)
                self.Q[L] = qs

    def plane(self, X):
        Z = np.asarray(X, dtype=np.float64)
        Z = Z - Z.mean(0)
        _, _, Vt = np.linalg.svd(Z, full_matrices=False)
        return Vt[:2]

    def retention(self, L, X):
        P = self.plane(X)
        vals = [float(np.linalg.norm(q.T @ P.T, "fro") ** 2 / 2.0) for q in self.Q[L]]
        return float(np.mean(vals)), vals


def ov_null(c3, L, C1, k, n=N_OV_NULL, seed=0):
    rng = np.random.default_rng(seed + 100 * L + k)
    v = [c3.retention(L, C1[rng.choice(len(C1), k, replace=False)])[0] for _ in range(n)]
    v = np.asarray(v)
    return dict(n=int(n), p99=float(np.percentile(v, PCT, method="higher")),
                mean=float(v.mean()), max=float(v.max()))


# =====================================================================================
# screens
# =====================================================================================
def screen_A(S42, C1, C2, Z1, Z2, C1cent, counts, cells, layer, m_grid=M_GRID,
             collect=True):
    """Returns (proposals dict hash->members, per-m stats)."""
    props, stats = {}, {}
    for m in m_grid:
        comps, sizes, _ = agreement_components(Z1, Z2, m)
        n_ge9 = int(len(comps))
        added = 0
        for c in comps:
            for tag, mem in [("raw", sorted(int(x) for x in c))] + \
                    [(f"dedup{t}", dedup(c, C1cent, counts, t)) for t in DEDUP_LADDER]:
                if len(mem) < KMIN:
                    continue
                h = mhash(mem)
                if h not in props:
                    props[h] = dict(members=mem, source=[], k=len(mem))
                    added += 1
                tagstr = f"A/L{layer}/m{m}/{tag}"
                if tagstr not in props[h]["source"]:
                    props[h]["source"].append(tagstr)
        stats[f"m{m}"] = dict(n_components_ge9=n_ge9,
                              n_components_ge5=int((sizes >= 5).sum()),
                              max_component=int(sizes.max()),
                              n_new_unique_proposals=added)
    return props, stats


def screen_B(S42, C1, C2, Z2, C1cent, counts, layer, ks=K_RANGE):
    props, stats = {}, {}
    nn = channel2_assembly(Z2, ks)
    for k in ks:
        added = 0
        seen_k = set()
        for row in nn[k]:
            mem = sorted(int(x) for x in row)
            h = mhash(mem)
            seen_k.add(h)
            for tag, mm in [("raw", mem)] + \
                    [(f"dedup{t}", dedup(mem, C1cent, counts, t)) for t in DEDUP_LADDER]:
                if len(mm) < KMIN:
                    continue
                hh = mhash(mm)
                if hh not in props:
                    props[hh] = dict(members=mm, source=[], k=len(mm))
                    added += 1
                tagstr = f"B/L{layer}/k{k}/{tag}"
                if tagstr not in props[hh]["source"]:
                    props[hh]["source"].append(tagstr)
        stats[f"k{k}"] = dict(n_unique_raw=len(seen_k), n_new_unique_proposals=added)
    return props, stats


def adjudicate_screen(S42, props, C1, C2, cells, counts, label, log):
    """Two passes: closure+rho for everything, then the tie class, then LOO on survivors."""
    t0 = time.time()
    quick = {}
    for h, p in props.items():
        mem = np.asarray(p["members"])
        g = S42.gap_ratio_of(C1[mem])
        if g is None or g >= GAP:
            quick[h] = dict(gap=None if g is None else float(g), rho=None)
            continue
        _, r = S42.cell_stats(C1[mem], C2[mem])
        quick[h] = dict(gap=float(g), rho=None if r is None else float(r))
    closers = {h: q for h, q in quick.items() if q["rho"] is not None
               and props[h]["k"] >= KMIN}
    # R14's tie class is the screen's maximum AT THAT k: the screen enumerates a different
    # family at every k, and a rho of 1.0 reachable at k = 9 says nothing about k = 12.
    rho_max_k, tie_k = {}, {}
    for h, q in closers.items():
        k = props[h]["k"]
        if k not in rho_max_k or q["rho"] > rho_max_k[k]:
            rho_max_k[k] = q["rho"]
    for h, q in closers.items():
        k = props[h]["k"]
        if q["rho"] >= rho_max_k[k] - 1e-12:
            tie_k.setdefault(k, []).append(h)
    rho_max = max(rho_max_k.values(), default=None)
    log(f"  [{label}] {len(props)} unique proposals, {len(closers)} adjudicable closers, "
        f"per-k screen max rho { {k: round(v, 4) for k, v in sorted(rho_max_k.items())} }, "
        f"tie sizes { {k: len(v) for k, v in sorted(tie_k.items())} } "
        f"({time.time()-t0:.0f} s)")
    return quick, rho_max_k, tie_k


def permuted_blocks(strings, counts, n_dec=10):
    from collections import defaultdict
    lc = np.log10(np.maximum(counts, 1.0))
    edges = np.quantile(lc, np.linspace(1.0 / n_dec, 1 - 1.0 / n_dec, n_dec - 1))
    dec = np.digitize(lc, edges)
    blocks = defaultdict(list)
    for i, s in enumerate(strings):
        a = [c for c in s if c.isalpha()]
        allcap = bool(a) and all(c.isupper() for c in a)
        cap = bool(a) and a[0].isupper() and not allcap
        blocks[(s.startswith(" "), cap, int(dec[i]))].append(i)
    return [np.asarray(v) for v in blocks.values()]


# =====================================================================================
# plants (REPLACE-not-add with carrier offset)
# =====================================================================================
def top_orders(Z, mmax):
    S = (Z @ Z.T).astype(np.float32)
    np.fill_diagonal(S, -9.0)
    idx = np.argpartition(-S, mmax, axis=1)[:, :mmax]
    rows = np.arange(len(Z))[:, None]
    return idx[rows, np.argsort(-S[rows, idx], axis=1)]


def comps_from_orders(o1, o2, m, n, min_size=KMIN):
    import scipy.sparse as sp
    Mut = _mutual(o1, o2, m, n)
    ncomp, lab = sp.csgraph.connected_components(sp.csr_matrix(Mut), directed=False)
    return lab, np.bincount(lab, minlength=ncomp)


def plant_stats(X, idx):
    Z = X[idx] - X[idx].mean(0)
    s = np.linalg.svd(Z, compute_uv=False)
    return float((s[:2] ** 2).sum() / (s ** 2).sum())


def make_plant(X, carrier, order, jitter, offset_norm, radius, top2frac, rng):
    d = X.shape[1]
    mu_pool = X.mean(0)
    u = rng.normal(size=d); u /= np.linalg.norm(u)
    A = rng.normal(size=(2, d))
    A -= np.outer(A @ u, u)
    Q, _ = np.linalg.qr(A.T)
    a, b = Q[:, 0], Q[:, 1]
    k = len(carrier)
    th = 2 * np.pi * np.asarray(order) / k + rng.normal(scale=jitter, size=k)
    ring = radius * (np.cos(th)[:, None] * a + np.sin(th)[:, None] * b)
    resid = X[carrier] - X[carrier].mean(0)
    B = np.vstack([u, a, b])
    resid = resid - (resid @ B.T) @ B
    e_ring = float((ring ** 2).sum())
    e_res = float((resid ** 2).sum())
    want = e_ring * (1.0 - top2frac) / max(top2frac, 1e-9)
    resid = resid * (np.sqrt(want / e_res) if e_res > 0 else 0.0)
    Y = X.copy()
    Y[carrier] = mu_pool + offset_norm * u + ring + resid
    return Y


def run_plants(S42, C1, C2, counts, cells, layer, months, rng, n_per_cell,
               jitters=PLANT_JITTERS, offsets=PLANT_OFFSETS, dispersed=False, log=print,
               cell_draws=N_CELL):
    n = len(C1)
    mu1, mu2 = C1.mean(0), C2.mean(0)
    def ring_radius(X):
        Z = X[months] - X[months].mean(0)
        _, _, Vt = np.linalg.svd(Z, full_matrices=False)
        P = Z @ Vt[:2].T
        return float(np.linalg.norm(P, axis=1).mean())
    r1, r2 = ring_radius(C1), ring_radius(C2)
    f1, f2 = plant_stats(C1, months), plant_stats(C2, months)
    off_ref1 = float(np.median([np.linalg.norm(C1[rng.choice(n, PLANT_K, False)].mean(0) - mu1)
                                for _ in range(200)]))
    off_ref2 = float(np.median([np.linalg.norm(C2[rng.choice(n, PLANT_K, False)].mean(0) - mu2)
                                for _ in range(200)]))
    months_off1 = float(np.linalg.norm(C1[months].mean(0) - mu1))
    months_off2 = float(np.linalg.norm(C2[months].mean(0) - mu2))
    meta = dict(ring_radius=[r1, r2], top2_fraction=[f1, f2],
                random12_median_offset=[off_ref1, off_ref2],
                months_offset=[months_off1, months_off2],
                months_offset_in_units=[months_off1 / off_ref1, months_off2 / off_ref2])
    rows = []
    cell12 = cells.get(12, cell_draws)
    thr = cell12["p99_max_strict"]
    for off in (offsets if not dispersed else [0.0]):
        for jit in jitters:
            recA = recB = clo = joint = 0
            for _ in range(n_per_cell):
                carrier = rng.choice(n, PLANT_K, replace=False)
                order = rng.permutation(PLANT_K)
                Y1 = make_plant(C1, carrier, order, jit, off * off_ref1, r1, f1, rng)
                Y2 = make_plant(C2, carrier, order, jit, off * off_ref2, r2, f2, rng)
                Zy1, Zy2 = centred_unit(Y1), centred_unit(Y2)
                o1, o2 = top_orders(Zy1, max(M_GRID)), top_orders(Zy2, max(K_RANGE))
                hitA = False
                for m in M_GRID:
                    lab, sizes = comps_from_orders(o1, o2, m, n)
                    labs = lab[carrier]
                    vals, cnts = np.unique(labs, return_counts=True)
                    best = int(cnts.max())
                    if best >= 9 and sizes[vals[int(np.argmax(cnts))]] >= 9:
                        hitA = True; break
                hitB = any(len(set(o2[s, :12].tolist()) & set(carrier.tolist())) >= 9
                           for s in carrier)
                recA += hitA; recB += hitB
                if hitA or hitB:
                    g = S42.gap_ratio_of(Y1[carrier])
                    if g is not None and g < GAP:
                        clo += 1
                        _, rr = S42.cell_stats(Y1[carrier], Y2[carrier])
                        if rr is not None and thr is not None and rr > thr:
                            joint += 1
            prop = max(recA, recB)
            rows.append(dict(layer=int(layer), offset_mult=float(off), jitter=float(jit),
                             n=int(n_per_cell), recall_A=recA / n_per_cell,
                             recall_B=recB / n_per_cell,
                             recall_either=None,
                             closure_given_proposed=(clo / prop) if prop else None,
                             joint=joint / n_per_cell))
            log(f"    plant L{layer} off {off}x jit {jit}: A {recA/n_per_cell:.2f} "
                f"B {recB/n_per_cell:.2f} joint {joint/n_per_cell:.2f}")
    return meta, rows


# =====================================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/phase5_agreement_screen.json")
    ap.add_argument("--layers", default="6,7")
    ap.add_argument("--perms-a", type=int, default=20)
    ap.add_argument("--perms-b", type=int, default=10)
    ap.add_argument("--plants", type=int, default=100)
    ap.add_argument("--plants-extra", type=int, default=50)
    ap.add_argument("--plant-layers", default="")
    ap.add_argument("--cell-draws", type=int, default=N_CELL)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    layers = [int(x) for x in args.layers.split(",")]
    out = REPO / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    cfg = dict(layers=layers, m_grid=list(M_GRID), k_range=list(K_RANGE),
               dedup_ladder=list(DEDUP_LADDER), dedup_registered=DEDUP_REGISTERED,
               cell_draws=args.cell_draws, cell_draws_candidate=N_CELL_CAND,
               seeds=list(SEED_TRIPLE), n_freq=N_FREQ, n_ov_null=N_OV_NULL,
               perms_A=args.perms_a, perms_B=args.perms_b,
               plants_per_cell=args.plants, plants_per_cell_extra_offsets=args.plants_extra,
               smoke=bool(args.smoke))
    doc = dict(status="RUNNING -- declaration stub written before any artifact was opened",
               script="scripts/48_agreement_graph_screen.py",
               declared_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
               stub_note=("written before any npz, weight file or tokenizer is opened. The "
                          "blind population is DEFINED here; its counts are filled in by the "
                          "run. No proposal membership outside the calendar positive control "
                          "is inspected by the script's author before the tier has run."),
               config=cfg, declared_rules=declared_rules(cfg),
               declared_blind_population=dict(
                   A="connected components with >= 9 members of the mutual two-channel "
                     "agreement graph, per (layer, m in {8,10,12}), plus their R15 dedup "
                     "ladder images; counts filled by the run",
                   B="all unique k >= 9 channel-2 cluster-like assemblies, per layer, k in "
                     "9..20, plus their R15 dedup ladder images; counts filled by the run",
                   expected_false_passers="(adjudicable proposals) x 0.01, and the "
                                          "permuted-search mean (R13), both reported"))
    out.write_text(json.dumps(doc, indent=1))

    logf = open(REPO / "logs/48_agreement.log", "a")
    def log(msg):
        print(msg, flush=True)
        logf.write(msg + "\n"); logf.flush()
    log(f"=== 48 start {doc['declared_at']} cfg={cfg}")

    S43 = _load("s43", "scripts/43_neighbour_assembly_screen.py")
    S42 = S43.S42
    log("tier imported from scripts/42 via scripts/43 (closure, cell_stats, "
        "frequency_matched_resample, class_features)")

    d1 = np.load(REPO / FULLN_RAW)
    d2 = np.load(REPO / ROBUST_DELTAS)
    pool = d1["pool_ids"].astype(int)
    assert (pool == d2["pool_ids"].astype(int)).all()
    counts = d1["counts"].astype(np.int64)
    n_types = len(pool)
    pos = {int(t): i for i, t in enumerate(pool)}
    months = np.asarray(sorted(pos[t] for t in STRICT12_IDS))
    assert mhash(months) == STRICT12_HASH, (mhash(months), STRICT12_HASH)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("google/gemma-3-1b-pt")
    strings = [tok.decode([int(i)]) for i in pool]
    log(f"pool {n_types} types; strict-12 hash {mhash(months)} verified")

    C1, C2, Z1, Z2, C1c, keepc = {}, {}, {}, {}, {}, {}
    for L in layers:
        keep = d2[f"L{L}_lag1_kept_coords"]
        keepc[L] = np.asarray(keep)
        assert list(d2[f"L{L}_lag1_dropped_coords"]) == [S42.R10_DROP[L]]
        C1[L] = d1[f"R_L{L}"][:, keep].astype(np.float64)
        C2[L] = d2[f"L{L}_lag1"].astype(np.float32).astype(np.float64)   # L5
        Z1[L] = centred_unit(C1[L]).astype(np.float32)
        Z2[L] = centred_unit(C2[L]).astype(np.float32)
        C1c[L] = C1[L] - C1[L].mean(0)
    log("channels built (R10 applied via kept_coords; deltas cast float32 -> float64, L5)")

    c3 = Channel3(layers, keepc)
    log("channel 3 loaded: o_proj head bases for " + str(layers))

    blocks = permuted_blocks(strings, counts)
    log(f"R13 permutation blocks: {len(blocks)} (class x log10-count decile); "
        f"sizes {sorted([len(b) for b in blocks])[-5:]} max")

    results = dict(layers={}, positive_control={}, permuted_search={}, plants={},
                   channel3_nulls={})
    cellsL = {L: Cells(S42, C1[L], C2[L], Z1[L], Z2[L], L) for L in layers}

    # ---------------- screens ----------------
    all_props = {}
    for L in layers:
        t0 = time.time()
        pA, sA = screen_A(S42, C1[L], C2[L], Z1[L], Z2[L], Z1[L], counts, cellsL[L], L)
        pB, sB = screen_B(S42, C1[L], C2[L], Z2[L], Z1[L], counts, L)
        log(f"L{L} screens: A {len(pA)} unique proposals {sA}; B {len(pB)} unique "
            f"({time.time()-t0:.0f} s)")
        all_props[L] = dict(A=pA, B=pB)
        results["layers"][f"L{L}"] = dict(A=dict(proposer_stats=sA, n_unique=len(pA)),
                                          B=dict(proposer_stats=sB, n_unique=len(pB)))
        for name, props in (("A", pA), ("B", pB)):
            hit = STRICT12_HASH in props
            results["positive_control"][f"L{L}_{name}"] = dict(
                strict12_proposed_blind=bool(hit),
                sources=props[STRICT12_HASH]["source"] if hit else [])
            log(f"  R18 L{L} proposer {name}: strict-12 proposed blind = {hit} "
                f"{props[STRICT12_HASH]['source'] if hit else ''}")
        out.write_text(json.dumps({**doc, **results, "status": "RUNNING"}, indent=1,
                                  default=float))

    # ---------------- adjudication ----------------
    for L in layers:
        for name in ("A", "B"):
            props = all_props[L][name]
            quick, rho_max_k, tie_k = adjudicate_screen(S42, props, C1[L], C2[L], cellsL[L],
                                                       counts, f"L{L}/{name}", log)
            rho_max = max(rho_max_k.values(), default=None)
            gate = []
            for h, q in quick.items():
                if q["rho"] is None or props[h]["k"] < KMIN:
                    continue
                kk = props[h]["k"]
                cell = cell_view(cellsL[L].get(kk, args.cell_draws), "strict")
                lo = cell["p99_min_strict"]
                tf = rho_max_k.get(kk)
                if (tf is not None and q["rho"] >= tf - 1e-12) or \
                        (lo is not None and q["rho"] >= lo) or cell["saturated"]:
                    gate.append(h)
            log(f"  [L{L}/{name}] {len(gate)} proposals reach the order gate; adjudicating "
                f"with LOO")
            recs = {}
            n_small = sum(1 for h in props if props[h]["k"] < KMIN)
            n_open = sum(1 for h, q in quick.items()
                         if props[h]["k"] >= KMIN and (q["gap"] is None or q["gap"] >= GAP))
            tally = {"PASS": 0, "MARGINAL": 0,
                     "FAIL": n_open + (len(props) - n_small - n_open - len(gate)),
                     "NOT ADJUDICABLE": n_small}
            for h in gate:
                r = adjudicate(S42, props[h]["members"], C1[L], C2[L], cellsL[L],
                               rho_max_k.get(props[h]["k"]), n_draws=args.cell_draws)
                r["member_hash"] = h
                r["source"] = props[h]["source"]
                tally[r["verdict"]] = tally.get(r["verdict"], 0) + 1
                if r["verdict"] in ("PASS", "MARGINAL"):
                    recs[h] = r
            # candidate-grade cells (K2) + frequency arm + channel 3
            for h, r in recs.items():
                mem = np.asarray(props[h]["members"])
                k = len(mem)
                cell_hi = cellsL[L].get(k, N_CELL_CAND)
                r["cell_p99_band_20k"] = [cell_hi["p99_min_strict"],
                                          cell_hi["p99_max_strict"]]
                st, det = order_state(r["two_channel_rho"],
                                      cell_view(cell_hi, "strict"), rho_max_k.get(k))
                r["order_state_20k"] = st
                r["order_detail_20k"] = det
                r["frequency_arm"] = frequency_arm(S42, mem, C1[L], C2[L], counts,
                                                   r["two_channel_rho"])
                ret, per_head = c3.retention(L, C1[L][mem])
                key = f"L{L}_k{k}"
                if key not in results["channel3_nulls"]:
                    results["channel3_nulls"][key] = ov_null(c3, L, C1[L], k)
                nullv = results["channel3_nulls"][key]
                r["channel3"] = dict(ov_retention=ret, per_head=per_head,
                                     null_p99=nullv["p99"], null_mean=nullv["mean"],
                                     above_p99=bool(ret > nullv["p99"]),
                                     declared_conjunct=bool(k <= 10))
                r["tokens"] = [strings[i] for i in mem]
                r["token_ids"] = [int(pool[i]) for i in mem]
                r["counts"] = [int(counts[i]) for i in mem]
                r["is_strict12"] = bool(h == STRICT12_HASH)
            results["layers"][f"L{L}"][name].update(
                screen_max_rho=rho_max,
                screen_max_rho_by_k={str(k): v for k, v in sorted(rho_max_k.items())},
                tie_class_size_by_k={str(k): len(v) for k, v in sorted(tie_k.items())},
                n_adjudicable_closers=len([1 for q in quick.values()
                                           if q["rho"] is not None]),
                n_order_gate=len(gate), tally=tally,
                candidates=list(recs.values()),
                expected_false_analytic=0.01 * len(gate))
            log(f"  [L{L}/{name}] tally {tally}; {len(recs)} PASS/MARGINAL candidates")
            out.write_text(json.dumps({**doc, **results, "status": "RUNNING"}, indent=1,
                                      default=float))

    # ---------------- R13 permuted search ----------------
    for L in layers:
        tie_A = {int(k): v for k, v in
                 results["layers"][f"L{L}"]["A"]["screen_max_rho_by_k"].items()}
        tie_B = {int(k): v for k, v in
                 results["layers"][f"L{L}"]["B"]["screen_max_rho_by_k"].items()}
        for name, n_perm, tie_map in (("A", args.perms_a, tie_A), ("B", args.perms_b, tie_B)):
            runs = []
            for p in range(n_perm):
                rng = np.random.default_rng(48_000_000 + 1000 * L + 10 * p +
                                            (0 if name == "A" else 5))
                perm = np.arange(n_types)
                for b in blocks:
                    perm[b] = b[rng.permutation(len(b))]
                Z2p = Z2[L][perm]
                C2p = C2[L][perm]
                t0 = time.time()
                if name == "A":
                    props, _ = screen_A(S42, C1[L], C2p, Z1[L], Z2p, Z1[L], counts,
                                        cellsL[L], L)
                else:
                    props, _ = screen_B(S42, C1[L], C2p, Z2p, Z1[L], counts, L)
                npass = 0
                passers = []
                for h, pr in props.items():
                    mem = np.asarray(pr["members"])
                    if len(mem) < KMIN:
                        continue
                    g = S42.gap_ratio_of(C1[L][mem])
                    if g is None or g >= GAP:
                        continue
                    _, r = S42.cell_stats(C1[L][mem], C2p[mem])
                    if r is None:
                        continue
                    cell = cell_view(cellsL[L].get(len(mem), args.cell_draws), "strict")
                    st, _d = order_state(r, cell, tie_map.get(len(mem)))
                    if st != "PASS":
                        continue
                    rr = adjudicate(S42, mem, C1[L], C2p, cellsL[L], tie_map.get(len(mem)),
                                    n_draws=args.cell_draws)
                    if rr["verdict"] == "PASS":
                        npass += 1
                        passers.append(dict(member_hash=h, k=int(len(mem)),
                                            gap=rr["gap_ratio"], rho=rr["two_channel_rho"]))
                runs.append(dict(perm=p, n_proposals=len(props), n_pass=npass,
                                 passers=passers[:10], seconds=round(time.time() - t0, 1)))
                log(f"  [perm L{L}/{name}] run {p}: {len(props)} proposals, {npass} PASS "
                    f"({time.time()-t0:.0f} s)")
            mean_pass = float(np.mean([r["n_pass"] for r in runs])) if runs else None
            results["permuted_search"][f"L{L}_{name}"] = dict(
                n_permutations=len(runs), mean_pass=mean_pass,
                pass_counts=[r["n_pass"] for r in runs],
                proposal_counts=[r["n_proposals"] for r in runs], runs=runs)
            log(f"  [perm L{L}/{name}] mean passers {mean_pass}")
            out.write_text(json.dumps({**doc, **results, "status": "RUNNING"}, indent=1,
                                      default=float))

    # ---------------- R14 final conjunction ----------------
    for L in layers:
        for name in ("A", "B"):
            blk = results["layers"][f"L{L}"][name]
            mp = results["permuted_search"].get(f"L{L}_{name}", {}).get("mean_pass")
            for r in blk["candidates"]:
                ok = (r["verdict"] == "PASS"
                      and r.get("frequency_arm", {}).get("pass_", False)
                      and mp is not None and mp < 0.5
                      and (r["channel3"]["above_p99"] if r["k"] <= 10 else True))
                r["R14_PASS"] = bool(ok)
                r["permuted_search_mean"] = mp
            blk["n_R14_PASS"] = sum(1 for r in blk["candidates"] if r["R14_PASS"])

    # ---------------- plants ----------------
    plant_layers = ([int(x) for x in args.plant_layers.split(",") if x]
                    or layers)
    for L in plant_layers:
        rng = np.random.default_rng(4800 + L)
        meta, rows = run_plants(S42, C1[L], C2[L], counts, cellsL[L], L, months, rng,
                                args.plants, offsets=(1.0,), log=log,
                                cell_draws=args.cell_draws)
        meta2, rows2 = run_plants(S42, C1[L], C2[L], counts, cellsL[L], L, months, rng,
                                  args.plants_extra, offsets=(2.0, 3.0), log=log,
                                  cell_draws=args.cell_draws)
        _m, rows3 = run_plants(S42, C1[L], C2[L], counts, cellsL[L], L, months, rng,
                               max(args.plants_extra, 50), jitters=(0.10,), dispersed=True,
                               log=log, cell_draws=args.cell_draws)
        results["plants"][f"L{L}"] = dict(meta=meta, offset_1x=rows, offset_2x_3x=rows2,
                                          dispersed_carrier=rows3)
        out.write_text(json.dumps({**doc, **results, "status": "RUNNING"}, indent=1,
                                  default=float))

    # ---------------- summary ----------------
    summary = dict(
        n_layers=len(layers),
        per_layer={f"L{L}": {nm: dict(
            n_unique_proposals=results["layers"][f"L{L}"][nm]["n_unique"],
            n_adjudicable_closers=results["layers"][f"L{L}"][nm]["n_adjudicable_closers"],
            tally=results["layers"][f"L{L}"][nm]["tally"],
            screen_max_rho=results["layers"][f"L{L}"][nm]["screen_max_rho"],
            n_R14_PASS=results["layers"][f"L{L}"][nm]["n_R14_PASS"],
            permuted_mean=results["permuted_search"].get(f"L{L}_{nm}", {}).get("mean_pass"),
            expected_false_analytic=results["layers"][f"L{L}"][nm]["expected_false_analytic"])
            for nm in ("A", "B")} for L in layers},
        positive_control=results["positive_control"],
        R14_passers=[dict(layer=L, proposer=nm, member_hash=r["member_hash"], k=r["k"],
                          gap=r["gap_ratio"], rho=r["two_channel_rho"],
                          is_strict12=r["is_strict12"], tokens=r["tokens"])
                     for L in layers for nm in ("A", "B")
                     for r in results["layers"][f"L{L}"][nm]["candidates"] if r["R14_PASS"]],
        deviations=doc["declared_rules"]["deviations"])
    doc["status"] = "COMPLETE"
    doc["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    doc["runtime_seconds"] = round(time.time() - t_start, 1)
    out.write_text(json.dumps({**doc, **results, "summary": summary}, indent=1, default=float))
    log(f"=== 48 complete in {doc['runtime_seconds']} s -> {out}")
    logf.close()


if __name__ == "__main__":
    main()
