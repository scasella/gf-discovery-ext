"""43 -- ring-specific assembly re-screen at L6/L7 (Phase 4 prereg 2.2 under Addendum K).

Pre-registration: docs/phase4_prereg_2026-08-22.md 2.2; docs/phase4_addendum_K_2026-08-22.md
K1-K8. Tier functions are IMPORTED from scripts/42_two_channel_readjudicate.py; that file is
not edited and not re-run.

Why this exists. 42 showed the two-channel tier discriminates (0 of 10 blind k>=9 flags pass,
0/400 nulls, strict-12 months pass at L6) but that Leiden proximity assembly hands it almost
nothing ring-shaped: 51 flags, 12 adjudicable, 0 passers. 2.2 replaces the ASSEMBLY with two
ring-specific rules and asks whether a rule whose admissibility criterion is itself a ring
property proposes objects the tier can then adjudicate.

  A1 neighbour consistency -- S is admissible iff every member's two nearest IN-PLANE metric
     neighbours inside S are its two cyclic-angular neighbours. Clumps with uniform angles
     fail this; rings pass it.
  A2 plane-seeded greedy extension with LOO -- add the type that most reduces the max angular
     gap; accept only if closure survives removal of every current member.

Both run on the raw residual type means at L6/L7 (R10 applied), channel 2 is the lag+1
successor-delta means, and every k >= 9 proposal is adjudicated by the frozen 42 tier with
K2's 20,000-draw conditional null.
"""
import argparse, gc, hashlib, importlib.util, json, os, sys, time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HOME", str(REPO / "cache/hf"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
sys.path.insert(0, str(REPO / "src"))

_spec = importlib.util.spec_from_file_location(
    "s42", str(REPO / "scripts/42_two_channel_readjudicate.py"))
S42 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(S42)              # imports only; 42's main() is not called

GAP = S42.GAP_RATIO_CLOSURE                # 2.5, frozen
KMIN = S42.K_MIN_ADJUDICABLE               # 9, R8
PCT = S42.THRESHOLD_PERCENTILE             # 99.0
R10 = S42.R10_DROP
D_MODEL = S42.D_MODEL

# ---------------------------------------------------------------- declared rules ----
LAYERS = (6, 7)
N_SEED_NN = 10                 # seeds = each type with each of its 10 nearest neighbours
N_CAND_NN = 60                 # extension candidate pool = 60-NN of the two seed types
K_MAX = 30                     # 2.2's stop rule
K_RECORD_MIN = 5               # admissible sets are recorded from k=5; only k>=9
                               # are STORED as proposals (R8), the rest as tallies
N_CLUSTERLIKE = 20_000         # K2
N_CLASSPERM = 200
N_FREQPERM = 200
N_NULL_ADJUDICATE = 50         # proposals re-adjudicated per null draw (declared cap,
                               # set from the smoke run's measured cost to keep the
                               # whole screen inside the wall-clock budget)
FREQ_BAND_DEX = 0.25           # log10-count band width for the frequency-matched permutation
PLANTED_RHOS = (0.05, 0.10, 0.15, 0.20, 0.30)
N_PLANTED = 100
PLANTED_K = 12
RAW_NPZ = "results/phase3_token_type_mean_resid_fulln.npz"
DELTAS_NPZ = "results/phase3_lag_action_robust_deltas.npz"
MONTH_TOKEN_IDS_13 = S42.MEMBERS_13_TOKEN_IDS
STRICT12_HASH = "a36a4f4f3e3b"          # results/phase4_gained_carriers.json strict12_member_hash


def declared_rules(cfg, carriers):
    return dict(
        prereg="docs/phase4_prereg_2026-08-22.md 2.2 under docs/phase4_addendum_K_2026-08-22.md K1-K8",
        tier_source=("closure, two_channel_rho, cell_stats, adjudicate and NullCache are "
                     "IMPORTED from scripts/42_two_channel_readjudicate.py. 42 is neither "
                     "edited nor re-run; its verdicts stand."),
        channel_1=("raw residual per-token-type means R_L from "
                   f"{RAW_NPZ} at L6/L7, R10 drop (coordinate 1038) applied before any plane "
                   "fit. 2.2 specifies the raw arm; the SAE mean-code arm is not used here."),
        channel_2=f"lag+1 successor-delta type means from {DELTAS_NPZ} (already post-drop)",
        seed_rule=(
            f"DECLARED SEED RESTRICTION. 2.2 says 'each pair of types' = 3795*3794/2 = "
            f"7,199,115 seeds, which is not runnable in the budget. Seeds are instead each "
            f"type paired with each of its {N_SEED_NN} nearest neighbours by CENTRED COSINE "
            f"in channel-1 coordinates, deduplicated as unordered pairs (~25-30k seeds per "
            f"layer). Justification measured before the run and recorded in "
            f"results.precheck: at L6 and L7 every one of the 11 other strict-12 months lies "
            f"within rank 17 of each month's centred-cosine neighbour list, so the month ring "
            f"is reachable from month seeds under this restriction. The restriction is "
            f"applied IDENTICALLY to the nulls and the plants."),
        candidate_pool=(f"extension candidates for a seed (i,j) = the union of the {N_CAND_NN} "
                        "nearest centred-cosine neighbours of i and of j, fixed for that "
                        "seed's run (<=121 types). Declared, not adaptive."),
        A1=("neighbour consistency. From the seed pair, extend by the candidate that most "
            "reduces the max angular gap in the CURRENT S's top-2 channel-1 plane; the "
            "extended S is admissible iff for every member its two nearest in-plane Euclidean "
            "neighbours within S are exactly its two cyclic-angular neighbours. Extension "
            "stops at the first inadmissible step or at k=30."),
        A2=("plane-seeded greedy extension with LOO. Same extension rule; the extended S is "
            "admissible iff gap_ratio < 2.5 AND closure survives removal of every current "
            "member. ASSEMBLY-TIME LOO is evaluated in S's own plane WITHOUT refitting the "
            "plane per subset (a proposal heuristic; declared). The VERDICT-bearing LOO of "
            "the tier refits the plane per subset exactly as 42 does."),
        K3_duplicate_guard=(
            "a candidate c is rejected if any current member m has BOTH |dtheta_1(c,m)| and "
            "|dtheta_2(c,m)| below 360/(4*(k+1)) degrees, where theta_1 is the angle in S's "
            "channel-1 plane and theta_2 the angle in S's channel-2 plane (both planes fitted "
            "on the current S, candidates projected into them). Applied identically in the "
            "real run, the nulls and the plants."),
        proposal_recording=(
            f"every admissible S of size >= {K_RECORD_MIN} reached along a seed's extension is "
            "recorded, deduplicated across seeds by md5 over sorted pool positions. Sets of "
            "size < 9 are not recorded as proposals (R8 makes them unadjudicable) but the "
            "assembly still passes through them. Because a seed emits nested sets, proposal "
            "counts are reported BOTH as unique member sets and as distinct seeds, and K8's "
            "expected-false arithmetic is reported under both."),
        tier=("K1/K2/K7: PASS := gap_ratio < 2.5 AND two_channel_rho > the 99th percentile of "
              f"the closure-conditioned cluster-like null at the same (layer, k) with "
              f"{N_CLUSTERLIKE} draws AND both properties at every leave-one-out, the LOO "
              "threshold being the (layer, k-1) cell's p99. k < 9 NOT ADJUDICABLE, reported "
              "under closure and closure-LOO only. Null cells are computed only for the k "
              "values actually proposed (and their k-1), as permitted."),
        run_a="assembly over the full L6/L7 pool",
        run_b=("K6: the same assembly seeded INSIDE each discovered 44b carrier S with "
               "|S| >= 12. Seeds are EVERY pair inside the carrier (2.2's full pair "
               "seeding, affordable at |S| <= 35) and the extension candidate pool is "
               "the carrier itself, so every proposal is a subset of the carrier. "
               "Carriers are listed in carriers/carriers_resolved."),
        nulls=(
            "MEASURED INVARIANCE, declared: A1 and A2 read channel 1 only, so permuting rows "
            "among types leaves the point cloud -- and therefore the set of proposed SHAPES "
            "and the proposal COUNT -- unchanged; only the member identities move. A literal "
            "re-assembly on a permuted population would therefore reproduce the real proposal "
            "count 400 times and measure nothing. The nulls are implemented as the "
            "equivalent and informative form: the channel-1/channel-2 CORRESPONDENCE is "
            "broken and the tier is re-run on the same proposals, which is exactly the "
            "false-PASS rate K8 needs. (i) class-block permutation: channel-2 rows permuted "
            "among types within leading_space x capitalized blocks (scripts/41b "
            f"resolve_blocks), {N_CLASSPERM} draws. (ii) frequency-band permutation: "
            f"channel-2 rows permuted within {FREQ_BAND_DEX}-dex log10 corpus-count bands, "
            f"{N_FREQPERM} draws. Both adjudicate against the TIER's own unpermuted "
            f"thresholds. Capped at the first {N_NULL_ADJUDICATE} k>=9 proposals per "
            "(layer, rule) ordered by k descending then gap ascending; the cap is stated with "
            "the resulting rate."),
        planted=(
            f"K4 anchor. A clean k={PLANTED_K} ring at the strict-12 months' own channel-1 "
            "radius is planted by REPLACING the channel-1 and channel-2 rows of 12 random pool "
            "types with ring points in a random 2-plane plus Gaussian in-plane jitter of sd "
            "rho*radius (channel 2 carries the same cyclic order with an independent jitter "
            "draw, as in 42's clean plant). ASSEMBLY RECALL = the fraction of plants for which "
            "A1 or A2, seeded from the planted types under the identical seed rule, proposes a "
            "set containing >= 11 of the 12 planted types; JOINT RECALL additionally requires "
            f"that proposal to PASS the tier. {N_PLANTED} plants per rho in "
            f"{list(PLANTED_RHOS)}. 42's clean-plant anchors: joint rho80 0.231, "
            "joint+LOO 0.118."),
        positive_control=(
            "K1: the strict-12 months at L6 must be RECOVERED by A1 or A2 from at least one "
            "month seed (recovery := a proposal whose member set equals the strict-12 set; "
            "the best Jaccard over all proposals is reported either way) and must PASS the "
            "tier. The L6 k=13 / L7 k=14 month FLAGS are reported alongside as the "
            "duplicate-variant negative (42: rho 0.539 / 0.648)."),
        negative_control="ordinal ' one'..' ten' through the identical tier at L6 and L7",
        no_mps="R12: CPU only, no forward passes, no MPS. Nothing here touches the SAE cache.",
        config=cfg, carriers=carriers)


# ===================================================================================
# geometry helpers -- vectorised versions of the 42/fits statistics, for the inner loop
# ===================================================================================
def plane_of(P):
    """top-2 principal plane of P via the k x k Gram (k <= 30, d = 1151)."""
    mu = P.mean(0)
    C = P - mu
    G = C @ C.T
    w, V = np.linalg.eigh(G)
    B = V[:, -2:].T @ C
    n = np.linalg.norm(B, axis=1, keepdims=True)
    return mu, B / np.maximum(n, 1e-12)


def angles_in(P, mu, B):
    Y = (P - mu) @ B.T
    return np.arctan2(Y[:, 1], Y[:, 0]), Y


def gap_ratio_of_angles(th):
    a = np.sort(th)
    g = np.diff(np.concatenate([a, [a[0] + 2 * np.pi]]))
    return float(g.max() / (2 * np.pi / len(th)))


def insert_gap_ratios(a_sorted, cand_th):
    """max-gap ratio that results from inserting each candidate angle into a_sorted."""
    k = len(a_sorted)
    ext = np.concatenate([a_sorted, [a_sorted[0] + 2 * np.pi]])
    g = np.diff(ext)
    pre = np.maximum.accumulate(g)
    suf = np.maximum.accumulate(g[::-1])[::-1]
    idx = np.searchsorted(a_sorted, cand_th)
    slot = (idx - 1) % k
    left = np.mod(cand_th - a_sorted[slot], 2 * np.pi)
    right = g[slot] - left
    lo = np.where(slot > 0, pre[np.maximum(slot - 1, 0)], -1.0)
    hi = np.where(slot < k - 1, suf[np.minimum(slot + 1, k - 1)], -1.0)
    newmax = np.maximum(np.maximum(lo, hi), np.maximum(left, right))
    return newmax / (2 * np.pi / (k + 1))


def loo_gap_ratios_fixed_plane(th):
    """gap ratio of every leave-one-out subset, angles fixed (assembly-time criterion)."""
    k = len(th)
    order = np.argsort(th)
    a = th[order]
    ext = np.concatenate([a, [a[0] + 2 * np.pi]])
    g = np.diff(ext)
    out = np.empty(k)
    pre = np.maximum.accumulate(g)
    suf = np.maximum.accumulate(g[::-1])[::-1]
    for t in range(k):
        merged = g[t - 1] + g[t] if t > 0 else g[k - 1] + g[0]
        if t == 0:
            # removing a[0] merges g[k-1] and g[0]; the survivors are g[1..k-2]
            other = float(np.max(g[1:k - 1])) if k > 2 else -1.0
        else:
            lo = pre[t - 2] if t >= 2 else -1.0
            hi = suf[t + 1] if t + 1 <= k - 1 else -1.0
            other = max(lo, hi)
        out[order[t]] = max(merged, other) / (2 * np.pi / (k - 1))
    return out


def neighbour_consistent(Y, th):
    """A1: each member's two nearest in-plane neighbours in S are its cyclic-angular ones."""
    k = len(th)
    order = np.argsort(th)
    rank = np.empty(k, dtype=np.int64)
    rank[order] = np.arange(k)
    D = np.linalg.norm(Y[:, None, :] - Y[None, :, :], axis=2)
    np.fill_diagonal(D, np.inf)
    for m in range(k):
        two = set(np.argsort(D[m])[:2].tolist())
        cyc = {int(order[(rank[m] - 1) % k]), int(order[(rank[m] + 1) % k])}
        if two != cyc:
            return False
    return True


def member_hash(members):
    return hashlib.md5(",".join(str(int(x)) for x in sorted(members)).encode()).hexdigest()[:12]


# ===================================================================================
# the two assembly rules
# ===================================================================================
def assemble(X1, X2, seed_i, seed_j, cand, rule, k_max=K_MAX, k_rec=K_RECORD_MIN):
    """One seed's extension under A1 or A2. Returns [(members, gap_ratio)] for k >= k_rec."""
    S = [int(seed_i), int(seed_j)]
    out = []
    cand = np.asarray(cand)
    for _ in range(k_max - 2):
        k = len(S)
        P1 = X1[S]
        mu1, B1 = plane_of(P1)
        th1_S, _ = angles_in(P1, mu1, B1)
        free = cand[~np.isin(cand, S)]
        if len(free) == 0:
            break
        th1_c, _ = angles_in(X1[free], mu1, B1)
        a = np.sort(th1_S)
        newg = insert_gap_ratios(a, th1_c)
        # K3 duplicate guard, evaluated in the CURRENT S's two planes
        mu2, B2 = plane_of(X2[S])
        th2_S, _ = angles_in(X2[S], mu2, B2)
        th2_c, _ = angles_in(X2[free], mu2, B2)
        tol = 2 * np.pi / (4 * (k + 1))
        d1 = np.abs(np.mod(th1_c[:, None] - th1_S[None, :] + np.pi, 2 * np.pi) - np.pi)
        d2 = np.abs(np.mod(th2_c[:, None] - th2_S[None, :] + np.pi, 2 * np.pi) - np.pi)
        blocked = ((d1 < tol) & (d2 < tol)).any(1)
        newg = np.where(blocked, np.inf, newg)
        if not np.isfinite(newg).any():
            break
        # try candidates best-first until one is admissible
        added = False
        for ci in np.argsort(newg):
            if not np.isfinite(newg[ci]):
                break
            T = sorted(S + [int(free[ci])])
            PT = X1[T]
            muT, BT = plane_of(PT)
            thT, YT = angles_in(PT, muT, BT)
            if rule == "A1":
                ok = neighbour_consistent(YT, thT)
            else:
                gr = gap_ratio_of_angles(thT)
                ok = bool(gr < GAP and (loo_gap_ratios_fixed_plane(thT) < GAP).all())
            if ok:
                S = T
                if len(S) >= k_rec:
                    out.append((list(S), gap_ratio_of_angles(thT)))
                added = True
                break
        if not added:
            break
    return out


def run_assembly(X1, X2, NN, seeds, rule, log_every=5000, tag="", small_tally=True,
                 cand_restrict=None):
    """All seeds under one rule.

    Returns ({member_hash: dict} for k >= 9, bookkeeping). Admissible sets of size 5..8 are
    NOT stored as proposals (R8 makes them unadjudicable) but ARE counted, together with
    their closure and closure-LOO tallies, so the small-k band is reported.
    """
    props, seed_hits, k_hist = {}, 0, {}
    seen_small = set()
    small = dict(n_unique=0, n_closure=0, n_closure_and_loo=0)
    t0 = time.time()
    for n, (i, j) in enumerate(seeds):
        cand = np.union1d(NN[i], NN[j])
        cand = np.union1d(cand, np.asarray([i, j]))
        if cand_restrict is not None:
            # K6 run (b): the assembly must produce SUBSETS of the carrier, so the extension
            # pool is the carrier itself, not the pool-wide neighbourhood.
            cand = np.intersect1d(cand, cand_restrict)
            cand = np.union1d(cand, np.asarray([i, j]))
        got = assemble(X1, X2, i, j, cand, rule)
        if got:
            seed_hits += 1
        for members, gr in got:
            k = len(members)
            k_hist[k] = k_hist.get(k, 0) + 1
            h = member_hash(members)
            if k < KMIN:
                if small_tally and h not in seen_small:
                    seen_small.add(h)
                    small["n_unique"] += 1
                    if gr < GAP:
                        small["n_closure"] += 1
                        P = X1[members]
                        mu, B = plane_of(P)
                        th, _ = angles_in(P, mu, B)
                        if bool((loo_gap_ratios_fixed_plane(th) < GAP).all()):
                            small["n_closure_and_loo"] += 1
                continue
            if h not in props:
                props[h] = dict(members=[int(x) for x in members], k=k,
                                gap_ratio=float(gr), seed=[int(i), int(j)], rule=rule)
        if log_every and (n + 1) % log_every == 0:
            print(f"    [{tag}{rule}] {n+1}/{len(seeds)} seeds, {len(props)} unique k>=9 "
                  f"proposals, {time.time()-t0:.0f}s", flush=True)
    return props, dict(n_seeds=len(seeds), n_seeds_with_proposal=seed_hits,
                       n_unique_k_ge_9=len(props), k_histogram_nonunique=k_hist,
                       small_k_5_to_8=small, seconds=round(time.time() - t0, 1))


def build_seeds(NN, n_nn=N_SEED_NN, restrict=None):
    seen, seeds = set(), []
    rows = range(len(NN)) if restrict is None else [int(x) for x in restrict]
    allowed = None if restrict is None else set(int(x) for x in restrict)
    for i in rows:
        for j in NN[i][:n_nn]:
            j = int(j)
            if allowed is not None and j not in allowed:
                continue
            key = (min(i, j), max(i, j))
            if key in seen:
                continue
            seen.add(key)
            seeds.append(key)
    return seeds


# ===================================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/phase4_assembly_screen.json")
    ap.add_argument("--layers", default="6,7")
    ap.add_argument("--n-clusterlike", type=int, default=N_CLUSTERLIKE)
    ap.add_argument("--n-planted", type=int, default=N_PLANTED)
    ap.add_argument("--max-seeds", type=int, default=0, help="0 = all (smoke only)")
    a = ap.parse_args()
    t_start = time.time()
    layers = [int(x) for x in a.layers.split(",")]
    n_cl = int(a.n_clusterlike)

    # ---- carriers (K6), read from 44b artifacts BEFORE the stub so they can be declared ----
    # NOTE: 44b stores carrier members as TOKEN STRINGS, not pool positions. They are declared
    # here by plane/size/source and resolved to pool rows after the tokenizer is loaded; the
    # resolution is verified against the stored member_hash where 44b recorded one.
    carriers = []
    nc = json.loads((REPO / "results/phase4_gained_carriers_near_candidates.json").read_text())
    for L in ("6", "7"):
        for p in nc["layers"][L]["planes"]:
            if len(p.get("members") or []) >= 12:
                carriers.append(dict(layer=int(L), plane=p["plane"], source="near_candidates",
                                     n=len(p["members"]),
                                     member_strings=list(p["members"]),
                                     member_hash_stored=p.get("member_hash")))
    dg = json.loads((REPO / "results/phase4_gained_carriers_promotion_diagnostic.json").read_text())
    for L in ("6", "7"):
        for p in dg["layers"][L].get("top_calendar_planes", []):
            if len(p.get("members") or []) >= 12:
                carriers.append(dict(layer=int(L), plane=p["plane"],
                                     source="promotion_diagnostic.top_calendar_planes",
                                     n=len(p["members"]),
                                     member_strings=list(p["members"]),
                                     member_hash_stored=None))

    cfg = dict(layers=layers, n_seed_nn=N_SEED_NN, n_cand_nn=N_CAND_NN, k_max=K_MAX,
               k_record_min=K_RECORD_MIN, n_clusterlike=n_cl, n_classperm=N_CLASSPERM,
               n_freqperm=N_FREQPERM, n_null_adjudicate=N_NULL_ADJUDICATE,
               n_planted=int(a.n_planted), planted_rhos=list(PLANTED_RHOS),
               raw_npz=RAW_NPZ, deltas_npz=DELTAS_NPZ, max_seeds=int(a.max_seeds),
               no_mps=True, no_forward_passes=True)
    declared_at = time.strftime("%Y-%m-%dT%H:%M:%S")
    stub = dict(status="DECLARATION STUB -- rules frozen, no results yet",
                script="scripts/43_neighbour_assembly_screen.py", declared_at=declared_at,
                thresholds=dict(gap_ratio_closure=GAP, percentile=PCT, k_min_adjudicable=KMIN,
                                n_clusterlike=n_cl),
                carriers=[{k: v for k, v in c.items() if k != "member_strings"}
                          for c in carriers],
                n_carriers=len(carriers),
                precheck=dict(
                    statement=("measured before the freeze of the seed rule: at L6 and L7 "
                               "every one of the 11 other strict-12 months lies within rank "
                               "17 of each month's centred-cosine neighbour list, so the "
                               "10-NN seed rule and the 60-NN candidate pool can reach the "
                               "month ring"),
                    worst_rank_L6=17, worst_rank_L7=17),
                declared_rules=declared_rules(cfg, carriers))
    Path(a.out).write_text(json.dumps(stub, indent=2))
    print("=== DECLARATION STUB WRITTEN (before any coordinate read) ===", flush=True)
    print(f"carriers with |S|>=12: {len(carriers)} "
          f"{[(c['layer'], c['plane'], c['n']) for c in carriers]}", flush=True)
    out = dict(stub)
    out["status"] = "RUNNING"

    # ---------------------------------------------------------------- data ----
    zr = np.load(REPO / RAW_NPZ)
    pool = zr["pool_ids"].astype(np.int64)
    counts = zr["counts"].astype(np.int64)
    n_types = len(pool)
    pos = {int(t): i for i, t in enumerate(pool)}
    zd = np.load(REPO / DELTAS_NPZ)
    assert np.array_equal(zd["pool_ids"].astype(np.int64), pool)
    X1, X2, NNt = {}, {}, {}
    for L in layers:
        keep = np.ones(D_MODEL, dtype=bool)
        keep[R10[L]] = False
        A = zr[f"R_L{L}"].astype(np.float64)[:, keep]
        assert np.isfinite(A).all(), f"non-finite channel 1 at L{L}"
        X1[L] = np.ascontiguousarray(A)
        B = zd[f"L{L}_lag1"].astype(np.float64)
        assert np.isfinite(B).all(), f"non-finite channel 2 at L{L}"
        X2[L] = np.ascontiguousarray(B)
        Xc = X1[L] - X1[L].mean(0, keepdims=True)
        Xn = (Xc / np.maximum(np.linalg.norm(Xc, axis=1, keepdims=True), 1e-12)).astype(np.float32)
        Sm = Xn @ Xn.T
        np.fill_diagonal(Sm, -2.0)
        NNt[L] = np.argsort(-Sm, axis=1)[:, :N_CAND_NN].astype(np.int64)
        del Sm, Xc
        gc.collect()
    del zr, zd
    gc.collect()
    print(f"pool {n_types}; layers {layers}", flush=True)

    S12 = np.asarray(sorted(pos[t] for t in MONTH_TOKEN_IDS_13 if t != 5656))
    assert member_hash(S12) == STRICT12_HASH, "strict-12 rows do not reproduce 44's hash"
    S13 = np.asarray(sorted(pos[t] for t in MONTH_TOKEN_IDS_13))
    month_rows = set(int(x) for x in S12)

    from transformers import AutoTokenizer
    from gfdisc.env import MODEL_NAME
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=os.environ.get("HF_TOKEN"))
    strings = [(tokenizer.convert_ids_to_tokens(int(t)) or tokenizer.decode([int(t)])
                ).replace("▁", " ") for t in pool]
    blocks = S42.resolve_blocks(strings)
    smap = {}
    for i, st in enumerate(strings):
        smap.setdefault(st, []).append(i)
    seen_h, resolved = set(), []
    for c in carriers:
        rows, missing = [], []
        for st in c["member_strings"]:
            v = smap.get(st)
            if not v:
                missing.append(st); continue
            rows.append(max(v, key=lambda i: int(counts[i])))
        h = member_hash(rows)
        c["members"] = sorted(int(x) for x in rows)
        c["n_resolved"] = len(rows)
        c["n_missing"] = len(missing)
        c["member_hash_recomputed"] = h
        c["hash_reproduces"] = (None if c["member_hash_stored"] is None
                                else bool(h == c["member_hash_stored"]))
        if c["hash_reproduces"] is False:
            print(f"  WARNING carrier {c['plane']} hash {h} != stored "
                  f"{c['member_hash_stored']}", flush=True)
        if h in seen_h:
            continue
        seen_h.add(h)
        resolved.append(c)
    carriers = resolved
    out["carriers_resolved"] = [{k: v for k, v in c.items() if k != "member_strings"}
                                for c in carriers]
    print(f"  carriers resolved to pool rows: {len(carriers)} "
          f"{[(c['layer'], c['plane'], c['n_resolved'], c['hash_reproduces']) for c in carriers]}",
          flush=True)
    ord_rows = []
    for w in S42.ORDINAL_WORDS:
        e = tokenizer.encode(w, add_special_tokens=False)
        if len(e) == 1 and int(e[0]) in pos:
            ord_rows.append(pos[int(e[0])])
    ord_rows = np.asarray(sorted(ord_rows))

    # null caches, one per layer, K2 draw count
    nulls = {}
    for L in layers:
        Xc = X1[L] - X1[L].mean(0, keepdims=True)
        Xn = (Xc / np.maximum(np.linalg.norm(Xc, axis=1, keepdims=True), 1e-12)).astype(np.float32)
        nulls[L] = S42.NullCache(Xn, X1[L], X2[L], counts, n_clusterlike=n_cl)
        del Xc, Xn
    gc.collect()

    def adjudicate(members, L, X2m=None, thr_override=None, want_loo=True):
        return S42.adjudicate(np.asarray(members), X1[L],
                              X2[L] if X2m is None else X2m, nulls[L], L,
                              want_loo=want_loo, thr_override=thr_override)

    # ---------------------------------------------------------------- run (a) ----
    print("\n============ RUN (a): full-pool assembly ============", flush=True)
    props, asm_stats = {}, {}
    for L in layers:
        seeds = build_seeds(NNt[L])
        if a.max_seeds:
            seeds = seeds[:a.max_seeds]
        print(f"  L{L}: {len(seeds)} unordered seed pairs", flush=True)
        for rule in ("A1", "A2"):
            p, st = run_assembly(X1[L], X2[L], NNt[L], seeds, rule, tag=f"L{L} ")
            props[(L, rule)] = p
            asm_stats[f"L{L}_{rule}"] = st
            print(f"  [L{L} {rule}] {st['n_unique_k_ge_9']} unique k>=9 proposals from "
                  f"{st['n_seeds_with_proposal']}/{st['n_seeds']} seeds "
                  f"({st['seconds']}s)", flush=True)

    # ---------------------------------------------------------------- run (b) ----
    print("\n============ RUN (b): carrier-seeded assembly (K6) ============", flush=True)
    props_b, asm_stats_b = {}, {}
    for c in carriers:
        L = c["layer"]
        if L not in layers:
            continue
        rows = np.asarray(sorted(set(int(x) for x in c["members"])))
        # inside a carrier of <= 35 members every pair is affordable, so run (b) uses the
        # FULL pair seeding of 2.2 rather than the 10-NN restriction (declared).
        seeds = [(int(rows[i]), int(rows[j])) for i in range(len(rows))
                 for j in range(i + 1, len(rows))]
        for rule in ("A1", "A2"):
            p, st = run_assembly(X1[L], X2[L], NNt[L], seeds, rule, log_every=0,
                                 cand_restrict=rows)
            p2 = {}
            for h, v in p.items():
                if set(v["members"]) <= set(int(x) for x in rows):
                    v = dict(v); v["carrier"] = c["plane"]; p2[h] = v
            props_b[(L, c["plane"], rule)] = p2
            asm_stats_b[f"L{L}_{c['plane']}_{rule}"] = dict(
                n_seeds=len(seeds), n_unique_inside_carrier=len(p2), n_unique_any=len(p),
                carrier_n=len(rows), seconds=st["seconds"])
        print(f"  L{L} {c['plane']} (|S|={len(rows)}): "
              f"A1 {len(props_b[(L, c['plane'], 'A1')])} / "
              f"A2 {len(props_b[(L, c['plane'], 'A2')])} proposals inside the carrier",
              flush=True)

    # ---------------------------------------------------------------- tier ----
    print("\n============ TIER ============", flush=True)
    all_props = []
    for (L, rule), p in props.items():
        for h, v in p.items():
            all_props.append(dict(v, layer=L, run="a", member_hash=h))
    for (L, plane, rule), p in props_b.items():
        for h, v in p.items():
            all_props.append(dict(v, layer=L, run="b", member_hash=h))
    ks = sorted({v["k"] for v in all_props if v["k"] >= KMIN})
    print(f"  {len(all_props)} proposals; k>=9 k-values {ks}", flush=True)
    t0 = time.time()
    for L in layers:
        for k in ks:
            nulls[L].build(L, k)
            if k - 1 >= 3:
                nulls[L].build(L, k - 1)
    print(f"  null cells built ({time.time()-t0:.0f}s)", flush=True)

    def adjudicate_2stage(members, L):
        """Frozen tier, evaluated cheaply: the LOO conjunct can only matter when closure and
        rho already hold, so it is computed only then. Identical verdicts, ~6x less work."""
        a1 = adjudicate(members, L, want_loo=False)
        if not (a1["adjudicable"] and a1["closure_pass"] and a1["rho_pass"]):
            if a1["adjudicable"]:
                a1["verdict"] = "FAIL"
            a1["loo_evaluated"] = False
            return a1
        a2 = adjudicate(members, L)
        a2["loo_evaluated"] = True
        return a2

    rows_out = []
    for v in all_props:
        L = v["layer"]
        adj = adjudicate_2stage(v["members"], L)
        r = dict(layer=L, run=v["run"], rule=v["rule"], carrier=v.get("carrier"),
                 seed=v["seed"], member_hash=v["member_hash"], **adj)
        r["is_strict12"] = bool(sorted(v["members"]) == sorted(S12.tolist()))
        r["n_months"] = int(len(month_rows & set(v["members"])))
        if adj["verdict"] == "PASS":
            r["tokens"] = [strings[i] for i in sorted(v["members"])]
        rows_out.append(r)
    passers = [r for r in rows_out if r["verdict"] == "PASS"]
    adjud = [r for r in rows_out if r.get("adjudicable")]
    print(f"  adjudicable {len(adjud)}; PASS {len(passers)}", flush=True)

    # ---------------------------------------------------------------- controls ----
    print("\n============ CONTROLS ============", flush=True)
    ctrl = {}
    for L in layers:
        s12 = adjudicate(S12, L)
        s13 = adjudicate(S13, L)
        ordc = adjudicate(ord_rows, L)
        best_j, best_h = 0.0, None
        recovered = False
        for v in all_props:
            if v["layer"] != L:
                continue
            s = set(v["members"])
            j = len(s & month_rows) / len(s | month_rows)
            if j > best_j:
                best_j, best_h = j, v["member_hash"]
            if j == 1.0:
                recovered = True
        ctrl[f"L{L}"] = dict(
            strict12_tier=s12, thirteen_set_tier=s13, ordinal_tier=ordc,
            strict12_recovered_by_assembly=bool(recovered),
            best_jaccard_to_strict12=float(best_j), best_jaccard_member_hash=best_h,
            months_in_best=int(round(best_j * 12)) if best_j else 0)
        print(f"  L{L}: strict12 gap {s12['gap_ratio']:.4f} rho {s12['two_channel_rho']:.4f} "
              f"thr {s12['threshold_p99']:.4f} -> {s12['verdict']}; recovered by assembly "
              f"{recovered} (best Jaccard {best_j:.3f}); 13-set rho "
              f"{s13['two_channel_rho']:.4f} -> {s13['verdict']}; ordinal gap "
              f"{ordc['gap_ratio']:.3f} -> {ordc['verdict']}", flush=True)

    # ---------------------------------------------------------------- nulls ----
    print("\n============ FALSE-PASS NULLS ============", flush=True)
    null_out = {}
    lc = np.log10(np.maximum(counts.astype(np.float64), 1.0))
    band = np.floor(lc / FREQ_BAND_DEX).astype(np.int64)
    fblocks = [np.flatnonzero(band == b) for b in np.unique(band)]
    for L in layers:
        for rule in ("A1", "A2"):
            sel = sorted([r for r in rows_out
                          if r["layer"] == L and r["rule"] == rule and r.get("adjudicable")],
                         key=lambda r: (-r["k"], r["gap_ratio"]))[:N_NULL_ADJUDICATE]
            if not sel:
                continue
            thr = {r["member_hash"]: (r["threshold_p99"], r["threshold_p99_loo"]) for r in sel}
            mem = {r["member_hash"]: None for r in sel}
            for r in sel:
                mem[r["member_hash"]] = next(v["members"] for v in all_props
                                             if v["member_hash"] == r["member_hash"])
            for name, blks, n_draw, seed in (("class_perm", blocks, N_CLASSPERM, 43_100_000 + L),
                                             ("freq_perm", fblocks, N_FREQPERM, 43_200_000 + L)):
                rng = np.random.default_rng(seed + (0 if rule == "A1" else 7))
                n_pass = n_pass_noloo = 0
                for _ in range(n_draw):
                    perm = np.arange(n_types)
                    for b in blks:
                        if len(b) > 1:
                            perm[b] = b[rng.permutation(len(b))]
                    D = X2[L][perm]
                    for h, m in mem.items():
                        # same two-stage evaluation as the real run: the LOO conjunct can
                        # only matter once closure and rho hold
                        adj = S42.adjudicate(np.asarray(m), X1[L], D, nulls[L], L,
                                             want_loo=False, thr_override=thr[h])
                        if not (adj["closure_pass"] and adj["rho_pass"]):
                            continue
                        n_pass_noloo += 1
                        adj = S42.adjudicate(np.asarray(m), X1[L], D, nulls[L], L,
                                             thr_override=thr[h])
                        n_pass += int(adj["verdict"] == "PASS")
                tot = n_draw * len(sel)
                null_out[f"L{L}_{rule}_{name}"] = dict(
                    n_draws=n_draw, n_proposals_adjudicated=len(sel), n_cells=tot,
                    n_pass_frozen=n_pass, n_pass_noLOO=n_pass_noloo,
                    false_pass_rate=n_pass / max(tot, 1),
                    cap_note=f"first {N_NULL_ADJUDICATE} k>=9 proposals by (k desc, gap asc)")
                print(f"  L{L} {rule} {name}: {n_pass}/{tot} PASS "
                      f"({n_pass/max(tot,1):.5f}), noLOO {n_pass_noloo}", flush=True)

    # ---------------------------------------------------------------- planted ----
    print("\n============ PLANTED ASSEMBLY RECALL ============", flush=True)
    L = layers[0]
    mu1, B1 = plane_of(X1[L][S12])
    Y1 = (X1[L][S12] - mu1) @ B1.T
    r1 = float(np.median(np.linalg.norm(Y1 - Y1.mean(0), axis=1)))
    mu2, B2 = plane_of(X2[L][S12])
    Y2 = (X2[L][S12] - mu2) @ B2.T
    r2 = float(np.median(np.linalg.norm(Y2 - Y2.mean(0), axis=1)))
    d1 = X1[L].shape[1]
    planted = dict(layer=L, k=PLANTED_K, radius_channel1=r1, radius_channel2=r2, curve=[])
    th_ring = 2 * np.pi * np.arange(PLANTED_K) / PLANTED_K
    for rho_j in PLANTED_RHOS:
        rng = np.random.default_rng(43_500_000 + int(1000 * rho_j))
        n_prop = n_joint = 0
        for _ in range(int(a.n_planted)):
            rows = rng.choice(n_types, size=PLANTED_K, replace=False)
            A1m = X1[L].copy()
            A2m = X2[L].copy()
            for XX, rad in ((A1m, r1), (A2m, r2)):
                Bq, _ = np.linalg.qr(rng.normal(size=(XX.shape[1], 2)))
                pts = np.column_stack([rad * np.cos(th_ring), rad * np.sin(th_ring)])
                pts = pts + rng.normal(0.0, rho_j * rad, size=pts.shape)
                XX[rows] = pts @ Bq.T
            Xc = A1m - A1m.mean(0, keepdims=True)
            Xn = Xc / np.maximum(np.linalg.norm(Xc, axis=1, keepdims=True), 1e-12)
            sim = Xn[rows] @ Xn.T
            for t, rr in enumerate(rows):
                sim[t, rr] = -2.0
            NNp = np.argsort(-sim, axis=1)[:, :N_CAND_NN]
            NNfull = NNt[L].copy()
            NNfull[rows] = NNp
            seeds = []
            seen = set()
            for t, rr in enumerate(rows):
                for j in NNp[t][:N_SEED_NN]:
                    key = (min(int(rr), int(j)), max(int(rr), int(j)))
                    if key not in seen:
                        seen.add(key); seeds.append(key)
            best = 0
            best_members = None
            for rule in ("A1", "A2"):
                p, _ = run_assembly(A1m, A2m, NNfull, seeds, rule, log_every=0)
                for v in p.values():
                    hit = len(set(v["members"]) & set(int(x) for x in rows))
                    if hit > best:
                        best, best_members = hit, v["members"]
            proposed = bool(best >= PLANTED_K - 1)
            n_prop += int(proposed)
            if proposed:
                nl = S42.NullCache(Xn.astype(np.float32), A1m, A2m, counts, n_clusterlike=1)
                thr = (nulls[L].build(L, len(best_members))["threshold_p99_conditional"],
                       nulls[L].build(L, len(best_members) - 1)["threshold_p99_conditional"])
                adj = S42.adjudicate(np.asarray(best_members), A1m, A2m, nl, L,
                                     thr_override=thr)
                n_joint += int(adj["verdict"] == "PASS")
        planted["curve"].append(dict(rho=rho_j, n=int(a.n_planted),
                                     assembly_recall=n_prop / max(int(a.n_planted), 1),
                                     joint_recall=n_joint / max(int(a.n_planted), 1)))
        print(f"  rho={rho_j:.2f}: assembly recall {n_prop/max(int(a.n_planted),1):.3f}, "
              f"joint {n_joint/max(int(a.n_planted),1):.3f}", flush=True)

    # ---------------------------------------------------------------- summary ----
    n_k9 = {f"L{L}_{rule}": sum(1 for r in rows_out
                                if r["layer"] == L and r["rule"] == rule and r.get("adjudicable"))
            for L in layers for rule in ("A1", "A2")}
    n_seed_k9 = {f"L{L}_{rule}": len({tuple(r["seed"]) for r in rows_out
                                      if r["layer"] == L and r["rule"] == rule
                                      and r.get("adjudicable")})
                 for L in layers for rule in ("A1", "A2")}
    out["assembly_run_a"] = asm_stats
    out["assembly_run_b"] = asm_stats_b
    keep_rows, per_cell = [], {}
    for r in sorted(rows_out, key=lambda r: (-r["k"], r["gap_ratio"])):
        key = (r["layer"], r["rule"], r["run"])
        per_cell[key] = per_cell.get(key, 0) + 1
        if r["verdict"] == "PASS" or per_cell[key] <= 200:
            keep_rows.append(r)
    out["proposals"] = keep_rows
    out["proposals_storage_rule"] = (
        "every PASS is stored; otherwise the first 200 per (layer, rule, run) ordered by k "
        "descending then gap ascending. Aggregate counts below cover all proposals.")
    out["null_cells"] = [v for L in layers for v in nulls[L].cache.values()]
    out["controls"] = ctrl
    out["false_pass_nulls"] = null_out
    out["planted"] = planted
    out["summary"] = dict(
        n_proposals_total=len(rows_out), n_adjudicable=len(adjud),
        n_pass=len(passers),
        passers=[{k: v for k, v in r.items() if k not in ("loo_gap_ratios", "loo_rhos")}
                 for r in passers],
        n_k9_by_cell=n_k9, n_distinct_seeds_k9_by_cell=n_seed_k9,
        expected_false_passers_K8_by_proposal={k: round(v * 0.01, 3) for k, v in n_k9.items()},
        expected_false_passers_K8_by_seed={k: round(v * 0.01, 3) for k, v in n_seed_k9.items()},
        positive_control_recovered={k: v["strict12_recovered_by_assembly"]
                                    for k, v in ctrl.items()},
        positive_control_tier={k: v["strict12_tier"]["verdict"] for k, v in ctrl.items()},
        gate=("PASS" if (ctrl.get("L6", {}).get("strict12_recovered_by_assembly")
                         and ctrl.get("L6", {}).get("strict12_tier", {}).get("verdict") == "PASS")
              else "FAIL"))
    out["status"] = "COMPLETE"
    out["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    out["runtime_seconds"] = round(time.time() - t_start, 1)
    Path(a.out).write_text(json.dumps(out, indent=2, default=float))
    print(f"\nwrote {a.out} in {out['runtime_seconds']}s", flush=True)
    print(json.dumps(out["summary"], indent=2, default=float)[:3000], flush=True)


if __name__ == "__main__":
    main()
