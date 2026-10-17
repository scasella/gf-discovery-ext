"""45b -- masked difference-family census v2 (Addendum O, O1-O6).

45's verdict was INSTRUMENT NOT VALIDATED with five measured defects. This file is a NEW
script; scripts/45_difference_family_census.py is imported by path for the pieces that are
unchanged (pool build, morphology mask, kNN pair construction, difference generator, spherical
k-means, blocked exact-cosine graph, retrieval, frequency-matched targets) and is not edited.

WHAT O CHANGES, DEFECT BY DEFECT
  O1 (defect 4, the star artifact): a family needs >= 6 pairs, >= 6 distinct SOURCES AND
     >= 6 distinct TARGETS, and no source or target may repeat more than twice. 45's rule
     required distinct sources only, so a many-to-one hub qualified and maxed the retrieval
     statistic by construction (119 of the 124 families with top-1 >= 0.9 were hubs).
  O2 (defect 1, agglomeration): compactness -- a family's difference vectors must have mean
     pairwise cosine >= the 99th percentile of size-matched random pair-sets drawn from the
     same kNN graph. And the Leiden resolution is SWEPT, on the capitalisation control only;
     45's single frozen resolution produced a 5,494-pair blob that swallowed 390 of the 482
     control pairs at purity 0.071.
  O3 (defects 2 and 3, size inversion and saturation): the adjudicating statistic is split-
     half top-1 (top-10 co-reported, never verdict-bearing), against a SIZE-MATCHED
     frequency-matched pseudo-family null with >= 200 draws per family size, under Addendum
     L1's saturated-cell rule: where the null's p99 is 1.0, "strictly above" is unsatisfiable,
     so a family passes iff its top-1 is 1.0 AND the null's exact-1.0 rate is < 0.01,
     otherwise the cell is NOT ADJUDICABLE. 45's class-permuted conjunct saturated at 1.0 for
     68/68 small families, which is what inverted the tier.
  O4 (defect 5): the plant is injected into the REAL masked 158k-pair background, whose graph
     is built once and reused, so the power curve is measured against the agglomeration that
     actually defeated the control rather than against an 8,000-pair synthetic background.
  O5: the positive control must be recovered AS A FAMILY under O1+O2 with split-half top-1
     >= 0.80 -- not merely as a reference relation (45 measured the relation at 0.847 while
     the assembly failed to recover it).
  O6: expected false families = (number of blind families) x the measured per-size false-
     acceptance of the whole rule, estimated on an INDEPENDENT second null batch.

EXPOSURE CARRIED IN FROM 45 (prereg section 0 discipline; the full record is in 45's own
declared_rules.exposure_before_this_run and in results/phase4_difference_families*.json):
  - 45's blind families and their statistics have been seen: 615 (L6) / 678 (L7) families,
    219 / 213 passers under 45's now-retired tier, the size-stratified table, the star counts,
    and the 15 non-hub high-retrieval candidates including the L7 arm-2 word-to-initial set
    (' system'->' S', ' process'->' P', ' line'->' L', ' model'->' M'). Under O5 that set, if
    re-found, is reported as a KNOWN-CLASS relation, never as a discovery.
  - the capitalisation reference relation's split-half retrieval (top-1 0.847 / top-10 0.935)
    and the month negative control (0/6 within-12, difference cosine -0.091).
  - 45's L6 arm-1 graph cost 68 s (the layer's 3,081 s was plants 1,342 s plus the
    assembly-level FA ~1,270 s), so the O2 resolution sweep is affordable. No edge list was
    persisted by 45; the graph is rebuilt here.

R7 (thresholds are null percentiles), R9 (not applicable -- the unit is a pair, not a cache
row), R10 (coordinate 1038 dropped), R12 (CPU only, no MPS, no cache streaming) all carry.
Arm 2 (the far-pair centroid arm) is NOT re-run: O1-O6 are written for the kNN arm, and 45's
arm-2 positive control was never established. Its single candidate is carried as exposure.
"""
import argparse, gc, importlib.util, json, os, sys, time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HOME", str(REPO / "cache/hf"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
sys.path.insert(0, str(REPO / "src"))


def _load(name, rel):
    sp = importlib.util.spec_from_file_location(name, str(REPO / rel))
    m = importlib.util.module_from_spec(sp)
    sp.loader.exec_module(m)
    return m


S45 = _load("s45", "scripts/45_difference_family_census.py")

FULLN_RAW = S45.FULLN_RAW
MODEL_NAME = S45.MODEL_NAME
D_MODEL = S45.D_MODEL
R10_DROP = S45.R10_DROP
KNN_K = S45.KNN_K
N_CENTROIDS = S45.N_CENTROIDS
LAYERS = (6, 7)
SEED = 0

# ---- O1 ----
FAM_MIN_PAIRS = 6
FAM_MIN_SOURCES = 6
FAM_MIN_TARGETS = 6
FAM_MAX_REPEAT = 2
# ---- O2 ----
RESOLUTION_GRID = (1.0, 2.0, 4.0, 8.0, 16.0)
COMPACT_PCT = 99.0
N_COMPACT_DRAWS = 200
# ---- O3 ----
N_TIER_DRAWS = 200            # batch A: threshold
N_TIER_DRAWS_B = 200          # batch B: independent false-acceptance (O6)
TIER_PCT = 99.0
SATURATION_MAX_EXACT_RATE = 0.01
FREQ_TOL_DEX = S45.FREQ_TOL_DEX
RETRIEVAL_REPEATS = 5
# ---- O4 ----
PLANT_M = (6, 10, 20)           # SUBSAMPLED from 45's (6,10,20,50); stated
PLANT_SIGMA = (0.3, 0.6, 0.9)
PLANT_REPS = 3                  # SUBSAMPLED (each replicate re-clusters the real graph)
# ---- O5 ----
CONTROL_TOP1_GATE = 0.80
CONTROL_MIN_PURITY = 0.50

MONTH_IDS_CAL_ORDER = S45.MONTH_IDS_CAL_ORDER


def declared_rules(av):
    return dict(
        prereg="docs/phase4_prereg_2026-08-22.md section 2.4 as amended by Addendum O",
        addenda_in_force=[f"docs/phase4_addendum_{x}_2026-08-22.md" for x in
                          ("K", "L", "M", "O")] + ["docs/phase4_addendum_J_2026-08-22.md"],
        supersedes="results/phase4_difference_families.json (45, INSTRUMENT NOT VALIDATED)",
        imports="scripts/45_difference_family_census.py by path; that file is not edited",
        o1_family_rule=(f">= {FAM_MIN_PAIRS} pairs, >= {FAM_MIN_SOURCES} distinct sources AND "
                        f">= {FAM_MIN_TARGETS} distinct targets, no source or target repeated "
                        f"more than {FAM_MAX_REPEAT} times"),
        o2_compactness=(f"mean pairwise cosine of the family's difference vectors >= the "
                        f"{COMPACT_PCT}th percentile of {N_COMPACT_DRAWS} size-matched random "
                        f"pair-sets drawn from the SAME kNN pair set (R7 null percentile)"),
        o2_resolution_sweep=(f"Leiden resolution swept over {list(RESOLUTION_GRID)} on the "
                             f"CAPITALISATION CONTROL ONLY (unmasked pair set). Selection rule, "
                             f"frozen: the SMALLEST resolution at which the control is "
                             f"recovered per O5. The control's pairs are then EXCLUDED from "
                             f"the blind population (they are morphology-masked in any case; "
                             f"the exclusion is asserted, not assumed)."),
        o3_tier=(f"split-half top-1 retrieval over all 3,795 types ({RETRIEVAL_REPEATS} random "
                 f"half-splits), adjudicated against a SIZE-MATCHED frequency-matched pseudo-"
                 f"family null: {N_TIER_DRAWS} draws at each family size, each draw being k "
                 f"pairs sampled from the same masked kNN pair set with every target replaced "
                 f"by a type within {FREQ_TOL_DEX} dex of its log10 count. PASS iff top-1 > the "
                 f"null's {TIER_PCT}th percentile. ADDENDUM L1 SATURATION RULE: if that "
                 f"percentile is 1.0, PASS requires top-1 == 1.0 AND the null's exact-1.0 rate "
                 f"< {SATURATION_MAX_EXACT_RATE}; otherwise NOT ADJUDICABLE. top-10 is "
                 f"co-reported and is never verdict-bearing."),
        o4_plant=(f"parallel-difference families of m in {list(PLANT_M)} at the measured "
                  f"capitalisation difference norm, jitter sigma/||t|| in {list(PLANT_SIGMA)}, "
                  f"{PLANT_REPS} replicates per cell (m grid and reps SUBSAMPLED for wall-clock; each replicate re-clusters the whole real graph), injected into the REAL masked ~158k-pair "
                  f"background whose graph is built once and reused. A replicate is a HIT iff "
                  f"one Leiden cluster holds >= {FAM_MIN_PAIRS} planted pairs."),
        o5_positive_control=(f"the capitalisation family must be recovered AS A FAMILY under "
                             f"O1+O2 with split-half top-1 >= {CONTROL_TOP1_GATE} and purity "
                             f">= {CONTROL_MIN_PURITY}. If no resolution achieves it, the "
                             f"verdict is INSTRUMENT NOT VALIDATED and nothing blind is "
                             f"reported."),
        o6_expected_false=("expected false families = sum over blind families of the measured "
                           "false-acceptance of the WHOLE rule at that family's size, estimated "
                           f"on an INDEPENDENT second batch of {N_TIER_DRAWS_B} null draws "
                           "scored against batch A's threshold"),
        arm2_not_rerun=("the far-pair centroid arm is not re-run: O1-O6 are written for the "
                        "kNN arm and 45's arm-2 positive control was never established. 45's "
                        "arm-2 word-to-initial candidate is carried as exposure and, if "
                        "re-found here, is reported as a KNOWN-CLASS relation (O5)."),
        exposure_from_45=("45's blind families and statistics have been seen (615/678 families, "
                          "219/213 passers under the retired tier, the size-stratified table, "
                          "star counts, and the 15 non-hub candidates). O1-O6 were written by "
                          "the coordinator from that diagnosis; no threshold here is fitted to "
                          "any blind family, and the only tuned quantity is the Leiden "
                          "resolution, tuned on the label-known control and declared above."),
        r9_not_applicable=("the unit of analysis is an ordered PAIR of token types; no cache "
                           "row is read and the split-half is over a family's pairs"),
        inputs=dict(type_means=FULLN_RAW, layers=list(LAYERS),
                    r10_drop={str(L): R10_DROP[L] for L in LAYERS},
                    device="cpu", no_forward_pass=True),
        args=av,
    )


# ---------------------------------------------------------------------------
def o1_valid(pairs):
    """O1: >=6 pairs, >=6 distinct sources AND targets, no endpoint repeated > twice."""
    p = np.asarray(pairs)
    if len(p) < FAM_MIN_PAIRS:
        return False
    su, sc = np.unique(p[:, 0], return_counts=True)
    tu, tc = np.unique(p[:, 1], return_counts=True)
    if len(su) < FAM_MIN_SOURCES or len(tu) < FAM_MIN_TARGETS:
        return False
    return bool(sc.max() <= FAM_MAX_REPEAT and tc.max() <= FAM_MAX_REPEAT)


def mean_pairwise_cos(X, pairs, cap=600, seed=0):
    D = X[np.asarray(pairs)[:, 1]] - X[np.asarray(pairs)[:, 0]]
    D = D / np.maximum(np.linalg.norm(D, axis=1, keepdims=True), 1e-12)
    if len(D) > cap:
        D = D[np.random.default_rng(seed).choice(len(D), cap, replace=False)]
    C = D @ D.T
    iu = np.triu_indices(len(D), 1)
    return float(C[iu].mean()) if len(iu[0]) else 0.0


class CompactNull:
    """O2: p99 of mean pairwise cosine for size-matched random pair-sets, cached by size."""

    def __init__(self, X, pool_pairs, seed):
        self.X, self.P, self.seed, self.cache = X, np.asarray(pool_pairs), seed, {}

    def p99(self, k):
        key = int(k) if k <= 60 else int(round(k / 10.0) * 10)
        if key in self.cache:
            return self.cache[key]
        rng = np.random.default_rng(self.seed + key)
        v = [mean_pairwise_cos(self.X, self.P[rng.choice(len(self.P), key, replace=False)],
                               seed=i) for i in range(N_COMPACT_DRAWS)]
        self.cache[key] = (float(np.percentile(v, COMPACT_PCT)), float(np.mean(v)))
        return self.cache[key]


class TierNull:
    """O3/O6: size-matched frequency-matched pseudo-family top-1 null, two batches."""

    def __init__(self, X, Xn, pool_pairs, logc, logc_order, seed):
        self.X, self.Xn, self.P = X, Xn, np.asarray(pool_pairs)
        self.logc, self.order, self.seed, self.cache = logc, logc_order, seed, {}

    def _draws(self, k, n, rng):
        out = []
        for _ in range(n):
            pp = self.P[rng.choice(len(self.P), k, replace=False)]
            pp = S45.freq_matched_targets(pp, self.logc, rng, order=self.order)
            r = S45.retrieval(self.X, self.Xn, pp, 0, n_repeats=1)
            if r:
                out.append(r["top1"])
        return np.asarray(out)

    def get(self, k):
        key = int(k) if k <= 60 else int(round(k / 10.0) * 10)
        if key in self.cache:
            return self.cache[key]
        rng = np.random.default_rng(self.seed + 7 * key)
        A = self._draws(key, N_TIER_DRAWS, rng)
        B = self._draws(key, N_TIER_DRAWS_B, rng)
        thr = float(np.percentile(A, TIER_PCT))
        exact = float((A >= 1.0).mean())
        saturated = bool(thr >= 1.0)
        if saturated:
            fa = float((B >= 1.0).mean()) if exact < SATURATION_MAX_EXACT_RATE else None
        else:
            fa = float((B > thr).mean())
        self.cache[key] = dict(k_bin=key, threshold_p99=thr, exact_1_rate=exact,
                               saturated=saturated, mean=float(A.mean()),
                               false_acceptance_batchB=fa, n_A=int(len(A)), n_B=int(len(B)))
        return self.cache[key]


def adjudicate(top1, nl):
    """O3 + Addendum L1."""
    if nl["saturated"]:
        if nl["exact_1_rate"] >= SATURATION_MAX_EXACT_RATE:
            return "NOT_ADJUDICABLE"
        return "PASS" if top1 >= 1.0 else "FAIL"
    return "PASS" if top1 > nl["threshold_p99"] else "FAIL"


# ---------------------------------------------------------------------------
def build_edges(diffs, K, seed, log, tau_override=None):
    """45's blocked exact-cosine edge construction, returned instead of clustered."""
    n = len(diffs)
    K = int(max(2, min(K, n // 4)))
    C, assign = S45.spherical_kmeans(diffs, K, S45.KMEANS_ITERS, seed, log=log)
    K = len(C)
    tau0, _ = S45.tau_from_null(diffs, seed, pct=S45.TAU_PERCENTILE)
    if tau_override is not None:
        tau0 = float(tau_override)
    order = np.argsort(assign, kind="stable")
    starts = np.searchsorted(assign[order], np.arange(K + 1))
    members = [order[starts[c]:starts[c + 1]] for c in range(K)]
    CC = C @ C.T
    np.fill_diagonal(CC, -2.0)
    nb = np.argsort(-CC, axis=1)[:, :S45.BLOCK_NEIGHBOUR_CENTROIDS]
    # 45's declared EDGE_BUDGET guard, inherited unchanged: a graph over the budget has tau
    # raised on the declared grid until it is not. A compute guard, not a statistical rule --
    # under Addendum O the granularity is set by the RESOLUTION sweep, not by tau.
    grid = [tau0] if tau_override is not None else \
        [tau0 + d for d in (0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40)]
    for tau in grid:
        ei, ej, ew, n_e, over = [], [], [], 0, False
        for c in range(K):
            src = members[c]
            if not len(src):
                continue
            cand = np.concatenate([members[c]] + [members[x] for x in nb[c]])
            A = diffs.rows(src)
            B = diffs.rows(cand)
            Sm = A @ B.T
            r, s = np.nonzero(Sm >= tau)
            if not len(r):
                continue
            u, v = src[r], cand[s]
            keep = u < v
            if keep.any():
                ei.append(u[keep]); ej.append(v[keep]); ew.append(Sm[r, s][keep])
                n_e += int(keep.sum())
            if n_e > S45.EDGE_BUDGET:
                over = True
                break
        if not over:
            break
    ei = np.concatenate(ei); ej = np.concatenate(ej); ew = np.concatenate(ew)
    key = ei.astype(np.int64) * n + ej.astype(np.int64)
    _, first = np.unique(key, return_index=True)
    return ei[first], ej[first], ew[first], float(tau), int(K)


def make_graph(n_nodes, ei, ej, ew):
    import igraph as ig
    g = ig.Graph(n=int(n_nodes), edges=np.column_stack([ei, ej]).tolist())
    g.es["weight"] = np.asarray(ew, dtype=float).tolist()
    return g


def cluster(g, resolution, seed=S45.LEIDEN_SEED):
    import leidenalg as la
    p = la.find_partition(g, la.RBConfigurationVertexPartition, weights="weight",
                          resolution_parameter=float(resolution), seed=seed, n_iterations=-1)
    return [np.asarray(sorted(int(x) for x in c), dtype=np.int64) for c in p]


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/phase4_difference_families_v2.json")
    ap.add_argument("--log", default="logs/45b_families_v2.log")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    t0 = time.time()
    (REPO / "logs").mkdir(exist_ok=True)
    log, logf = S45.log_open(REPO / a.log)

    av = dict(vars(a))
    stub = dict(status="DECLARATION STUB -- rules frozen, no results yet",
                script="scripts/45b_difference_family_census_v2.py",
                item="Phase 4 Addendum O -- masked difference-family census v2",
                declared_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
                declared_rules=declared_rules(av))
    Path(REPO / a.out).write_text(json.dumps(stub, indent=2))
    log("=== 45b DECLARATION STUB WRITTEN ===", a.out, stub["declared_at"])

    smoke = a.smoke
    layers = (6,) if smoke else LAYERS
    plant_reps = 2 if smoke else PLANT_REPS
    res_grid = (1.0, 4.0) if smoke else RESOLUTION_GRID

    z = np.load(REPO / FULLN_RAW)
    pool_ids = z["pool_ids"].astype(np.int64)
    counts = z["counts"].astype(np.int64)
    n_types = len(pool_ids)
    from transformers import AutoTokenizer
    tk = AutoTokenizer.from_pretrained(MODEL_NAME)
    strings = [tk.decode([int(t)]) for t in pool_ids]
    masker = S45.Masker(strings)
    logc = np.log10(np.maximum(counts, 1)).astype(np.float64)
    logc_order = np.argsort(logc)
    pos = {int(t): i for i, t in enumerate(pool_ids)}

    from collections import defaultdict
    by = defaultdict(list)
    for i, s in enumerate(strings):
        by[s.strip().lower()].append(i)
    cap_pairs = []
    for _, v in by.items():
        for x in v:
            for y in v:
                if x == y or strings[x] == strings[y]:
                    continue
                if (strings[x][:1] == " ") != (strings[y][:1] == " "):
                    continue
                if strings[x].strip()[:1].islower() and strings[y].strip()[:1].isupper():
                    cap_pairs.append((x, y))
    cap_pairs = np.asarray(cap_pairs, dtype=np.int32)
    capset = set(map(tuple, cap_pairs.tolist()))
    month_rows = [pos[t] for t in MONTH_IDS_CAL_ORDER]
    month_pairs = np.asarray([[month_rows[i], month_rows[(i + 1) % 12]] for i in range(12)],
                             dtype=np.int32)
    log(f"pool {n_types}; capitalisation reference pairs {len(cap_pairs)}")

    out = dict(stub); out["status"] = "RUNNING"; out["layers"] = {}
    out["pool"] = dict(n_types=int(n_types), n_cap_reference_pairs=int(len(cap_pairs)))

    for L in layers:
        tL = time.time()
        log(f"\n===== LAYER {L} =====")
        A = z[f"R_L{L}"].astype(np.float64)
        keep = np.ones(D_MODEL, bool); keep[R10_DROP[L]] = False
        X = np.ascontiguousarray(A[:, keep], dtype=np.float32); del A
        X -= X.mean(0, keepdims=True)
        Xn = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
        rec = dict(layer=int(L), d_kept=int(X.shape[1]), dropped_coord=int(R10_DROP[L]))

        Sm = (Xn @ Xn.T).astype(np.float32); np.fill_diagonal(Sm, -2.0)
        nn = np.argpartition(-Sm, KNN_K, axis=1)[:, :KNN_K]
        del Sm; gc.collect()
        pairs_all = np.column_stack([np.repeat(np.arange(n_types, dtype=np.int32), KNN_K),
                                     nn.reshape(-1).astype(np.int32)])
        keepm = masker.mask_pairs(pairs_all)
        pairs_blind = pairs_all[keepm]
        n_cap_in_blind = sum(1 for r in pairs_blind.tolist() if tuple(r) in capset)
        assert n_cap_in_blind == 0, f"control pairs leaked into the blind population: {n_cap_in_blind}"
        rec.update(pairs_raw=int(len(pairs_all)), pairs_blind=int(len(pairs_blind)),
                   masked_fraction=float(1 - keepm.mean()),
                   control_pairs_excluded_from_blind=True)
        log(f"  pairs {len(pairs_all)} -> blind {len(pairs_blind)} "
            f"({100*(1-keepm.mean()):.1f}% masked); control exclusion asserted")

        # ---------- O2/O5: resolution sweep on the CONTROL ONLY ----------
        ei, ej, ew, tau_c, K_c = build_edges(S45.Diffs(X, pairs_all), N_CENTROIDS,
                                             SEED + 900 + L, log)
        g_ctrl = make_graph(len(pairs_all), ei, ej, ew)
        log(f"  control graph: {len(ei)} edges, tau {tau_c:.3f} ({time.time()-tL:.0f}s)")
        cn_ctrl = CompactNull(X, pairs_all, SEED + 11)
        sweep, chosen = [], None
        for res in res_grid:
            cl = cluster(g_ctrl, res)
            best = None
            for c in cl:
                p = pairs_all[c]
                nc = sum(1 for r in p.tolist() if tuple(r) in capset)
                if nc < FAM_MIN_PAIRS or not o1_valid(p):
                    continue
                pur = nc / len(p)
                if best is None or nc > best[1]:
                    best = (c, nc, pur)
            row = dict(resolution=float(res), n_clusters=len(cl))
            if best is None:
                row.update(recovered=False, reason="no O1-valid cluster with >=6 control pairs")
            else:
                c, nc, pur = best
                p = pairs_all[c]
                r = S45.retrieval(X, Xn, p, SEED, n_repeats=RETRIEVAL_REPEATS)
                cmp99, cmean = cn_ctrl.p99(len(p))
                cos = mean_pairwise_cos(X, p)
                row.update(recovered=True, size=int(len(p)), n_control_pairs=int(nc),
                           purity=float(pur), top1=r["top1"], top10=r["top10"],
                           mean_pairwise_cos=cos, compact_p99=cmp99, compact_null_mean=cmean,
                           compact_ok=bool(cos >= cmp99),
                           o5_pass=bool(r["top1"] >= CONTROL_TOP1_GATE
                                        and pur >= CONTROL_MIN_PURITY and cos >= cmp99))
            sweep.append(row)
            log(f"    res {res}: " + json.dumps({k: v for k, v in row.items()
                                                 if k != "resolution"}))
            if chosen is None and row.get("o5_pass"):
                chosen = float(res)
        rec["o2_resolution_sweep"] = sweep
        rec["o5_selected_resolution"] = chosen
        del g_ctrl, ei, ej, ew; gc.collect()

        if chosen is None:
            rec["verdict"] = "INSTRUMENT NOT VALIDATED (O5: control not recovered at any swept resolution)"
            rec["seconds"] = round(time.time() - tL, 1)
            out["layers"][str(L)] = rec
            Path(REPO / a.out).write_text(json.dumps(out, indent=2, default=float))
            log(f"  VERDICT: {rec['verdict']}")
            if L == 6:      # non-verdict diagnostic AFTER the verdict is on disk
                rec["tau_diagnostic_non_verdict"] = tau_diagnostic(
                    X, Xn, pairs_all, capset, log, SEED + 700)
                rec["seconds"] = round(time.time() - tL, 1)
                out["layers"][str(L)] = rec
                Path(REPO / a.out).write_text(json.dumps(out, indent=2, default=float))
            del X, Xn; gc.collect()
            continue
        log(f"  O5 PASSES at resolution {chosen}")

        # ---------- blind ----------
        bei, bej, bew, tau_b, K_b = build_edges(S45.Diffs(X, pairs_blind), N_CENTROIDS,
                                                SEED + L, log)
        g_blind = make_graph(len(pairs_blind), bei, bej, bew)
        rec.update(blind_edges=int(len(bei)), blind_tau=float(tau_b))
        cl = cluster(g_blind, chosen)
        rec["blind_n_clusters"] = len(cl)
        cn = CompactNull(X, pairs_blind, SEED + 21)
        tn = TierNull(X, Xn, pairs_blind, logc, logc_order, SEED + 31)
        fams, n_o1, n_compact = [], 0, 0
        for cid, c in enumerate(cl):
            p = pairs_blind[c]
            if not o1_valid(p):
                continue
            n_o1 += 1
            cos = mean_pairwise_cos(X, p)
            cp99, cmean = cn.p99(len(p))
            if cos < cp99:
                continue
            n_compact += 1
            r = S45.retrieval(X, Xn, p, SEED + cid, n_repeats=RETRIEVAL_REPEATS)
            nl = tn.get(len(p))
            verdict = adjudicate(r["top1"], nl)
            fams.append(dict(family_id=int(cid), n_pairs=int(len(p)),
                             n_sources=int(len(np.unique(p[:, 0]))),
                             n_targets=int(len(np.unique(p[:, 1]))),
                             mean_pairwise_cos=cos, compact_p99=cp99,
                             top1=r["top1"], top10=r["top10"],
                             median_rank=r["median_rank"], null=nl, verdict=verdict,
                             pairs=p.tolist()))
        rec.update(n_clusters_o1_valid=int(n_o1), n_families_after_compactness=int(n_compact),
                   families=fams,
                   n_pass=int(sum(1 for f in fams if f["verdict"] == "PASS")),
                   n_not_adjudicable=int(sum(1 for f in fams
                                             if f["verdict"] == "NOT_ADJUDICABLE")))
        # O6
        ef, missing = 0.0, 0
        for f in fams:
            fa = f["null"]["false_acceptance_batchB"]
            if fa is None:
                missing += 1
            else:
                ef += fa
        rec["o6_expected_false"] = dict(expected_false_families=float(ef),
                                        n_families=len(fams),
                                        n_sizes_without_fa_estimate=int(missing),
                                        observed_pass=rec["n_pass"])
        log(f"  blind: {len(cl)} clusters -> {n_o1} O1-valid -> {n_compact} compact; "
            f"{rec['n_pass']} PASS, {rec['n_not_adjudicable']} NOT ADJUDICABLE; "
            f"expected false {ef:.2f} ({time.time()-tL:.0f}s)")

        # ---------- month negative control ----------
        rec["months_negative"] = dict(
            within_12=S45._within_set_retrieval(X, month_pairs, np.asarray(month_rows), SEED),
            all_types=S45.retrieval(X, Xn, month_pairs, SEED),
            mean_pairwise_diff_cos=mean_pairwise_cos(X, month_pairs),
            o1_valid=bool(o1_valid(month_pairs)))

        # ---------- O4 plant, real background ----------
        if L == 6:
            cap_norm = float(np.linalg.norm(X[cap_pairs[:, 1]] - X[cap_pairs[:, 0]],
                                            axis=1).mean())
            rec["plant_norm"] = cap_norm
            rec["planted"] = plant_real(X, pairs_blind, bei, bej, bew, chosen, cap_norm,
                                        plant_reps, tau_b, log, SEED + 400)
        rec["seconds"] = round(time.time() - tL, 1)
        rec["verdict"] = "INSTRUMENT VALIDATED (O5 met); blind families adjudicated under O3"
        out["layers"][str(L)] = rec
        Path(REPO / a.out).write_text(json.dumps(out, indent=2, default=float))
        log(f"  layer {L} done in {rec['seconds']}s")
        del X, Xn, g_blind, bei, bej, bew; gc.collect()

    if len(out["layers"]) == 2:
        out["persistence"] = persistence(out)
    out["status"] = "complete"
    out["runtime_seconds"] = round(time.time() - t0, 1)
    Path(REPO / a.out).write_text(json.dumps(out, indent=2, default=float))
    log(f"\ntotal {out['runtime_seconds']}s")
    logf.close()


def plant_real(X, pairs_bg, ei, ej, ew, resolution, tnorm, reps, tau, log, seed):
    """O4: inject m parallel differences into the REAL background graph and re-cluster."""
    rng = np.random.default_rng(seed)
    n_bg, d = len(pairs_bg), X.shape[1]
    DBG = S45.Diffs(X, pairs_bg)
    res, t0 = [], time.time()
    for m in PLANT_M:
        for sig in PLANT_SIGMA:
            hits = []
            for rep in range(reps):
                srcs = rng.choice(len(X), size=m, replace=False)
                t = rng.normal(size=d).astype(np.float32)
                t *= tnorm / np.linalg.norm(t)
                E = rng.normal(size=(m, d)).astype(np.float32)
                E *= (sig * tnorm) / np.maximum(np.linalg.norm(E, axis=1, keepdims=True), 1e-12)
                Dp = t[None, :] + E
                Dp /= np.maximum(np.linalg.norm(Dp, axis=1, keepdims=True), 1e-12)
                # edges from planted nodes to the background and to each other
                pe_i, pe_j, pe_w = [], [], []
                PP = Dp @ Dp.T
                for x in range(m):
                    for y in range(x + 1, m):
                        if PP[x, y] >= tau:
                            pe_i.append(n_bg + x); pe_j.append(n_bg + y); pe_w.append(PP[x, y])
                step = 40_000
                for lo in range(0, n_bg, step):
                    hi = min(lo + step, n_bg)
                    B = DBG.chunk(lo, hi)
                    Sc = Dp @ B.T
                    r, s = np.nonzero(Sc >= tau)
                    if len(r):
                        pe_i.extend((n_bg + r).tolist())
                        pe_j.extend((lo + s).tolist())
                        pe_w.extend(Sc[r, s].tolist())
                    del B, Sc
                g = make_graph(n_bg + m, np.concatenate([ei, np.asarray(pe_i, dtype=np.int64)]),
                               np.concatenate([ej, np.asarray(pe_j, dtype=np.int64)]),
                               np.concatenate([ew, np.asarray(pe_w, dtype=np.float32)]))
                cl = cluster(g, resolution)
                planted = np.arange(n_bg, n_bg + m)
                best = max((int(np.isin(c, planted).sum()) for c in cl), default=0)
                hits.append(best >= FAM_MIN_PAIRS)
                del g, cl
                gc.collect()
            res.append(dict(m=int(m), sigma_over_t=float(sig), reps=int(reps),
                            recall=float(np.mean(hits))))
            log(f"    plant m={m} sigma={sig}: recall {sum(hits)}/{len(hits)} "
                f"({time.time()-t0:.0f}s)")
    return res


def tau_diagnostic(X, Xn, pairs_all, capset, log, seed):
    """CONTROL-ONLY, EXPLICITLY NON-VERDICT (Addendum O declares a RESOLUTION sweep, not a
    tau sweep). Measures whether the capitalisation family is recoverable at ANY graph
    threshold, so the next addendum knows which knob is the binding one rather than repeating
    a dead round. No blind family is clustered, scored, or reported from these settings."""
    d = S45.Diffs(X, pairs_all)
    fam_cos = mean_pairwise_cos(X, np.asarray([p for p in pairs_all.tolist()
                                               if tuple(p) in capset], dtype=np.int32))
    rows = [dict(note="mean pairwise cosine of the capitalisation family's own differences",
                 control_family_internal_cos=float(fam_cos))]
    log(f"    [non-verdict] control family internal cosine {fam_cos:.3f}")
    DIAG_EDGE_CAP = 6_000_000     # memory guard (<= 6 GB); a skipped tau is recorded
    for tau in (0.50, 0.40, 0.30):
        ei, ej, ew, tau_u, _ = build_edges(d, N_CENTROIDS, seed, lambda *a: None,
                                           tau_override=tau)
        if len(ei) > DIAG_EDGE_CAP:
            rows.append(dict(tau=float(tau), n_edges=int(len(ei)),
                             skipped=f"over the {DIAG_EDGE_CAP} diagnostic edge cap"))
            log(f"    [non-verdict] tau {tau}: {len(ei)} edges, SKIPPED (memory guard)")
            del ei, ej, ew
            gc.collect()
            continue
        g = make_graph(len(pairs_all), ei, ej, ew)
        for res in (1.0, 4.0):
            cl = cluster(g, res)
            best = None
            for c in cl:
                p = pairs_all[c]
                nc = sum(1 for r in p.tolist() if tuple(r) in capset)
                if nc < FAM_MIN_PAIRS or not o1_valid(p):
                    continue
                if best is None or nc > best[1]:
                    best = (p, nc)
            row = dict(tau=float(tau), resolution=float(res), n_edges=int(len(ei)),
                       n_clusters=len(cl))
            if best is None:
                row.update(recovered=False)
            else:
                p, nc = best
                r = S45.retrieval(X, Xn, p, 0, n_repeats=RETRIEVAL_REPEATS)
                row.update(recovered=True, size=int(len(p)), n_control_pairs=int(nc),
                           purity=float(nc / len(p)), top1=r["top1"], top10=r["top10"])
            rows.append(row)
            log("    [non-verdict] " + json.dumps(row))
        del g, ei, ej, ew
        gc.collect()
    return dict(status="NON-VERDICT DIAGNOSTIC; no blind family was clustered at these settings",
                rows=rows)


def persistence(out):
    L6 = [f for f in out["layers"].get("6", {}).get("families", []) if f["verdict"] == "PASS"]
    L7 = [np.asarray(f["pairs"]) for f in out["layers"].get("7", {}).get("families", [])]
    rows = []
    for f in L6:
        p6 = np.asarray(f["pairs"])
        js = [S45.jaccard(p6, p7) for p7 in L7]
        rows.append(dict(family_id=f["family_id"], n_pairs=f["n_pairs"],
                         max_jaccard_L7=float(max(js)) if js else 0.0))
    return dict(n_checked=len(rows), rows=rows,
                note="max Jaccard over pairs against every L7 family that cleared O1+O2")


if __name__ == "__main__":
    main()
