"""43c -- full-multiplicity null for the 43/43b assembly screen (Addendum M, M1-M3).

Addenda in force: K1-K8 (via 43), L1-L7 (via 43b), M1-M3 (this file).
42, 43 and 43b are IMPORTED by path and are neither edited nor re-run.

The question M1 asks. 43 proposed 358,693 sets and 43b's L-tier passed 10 of them (8 unique;
one is the strict-12 month positive control). K8's analytic 1%-per-proposal FDR does not apply
to a nested proposal family, and 43's 0/80,000 permutation null was capped at 50 proposals per
cell against the anchor threshold -- it bounds the anchor, not the search. The only honest
measurement of the search's false-pass rate is to run the WHOLE screen against a channel-2
null. Because the assembly reads channel 1 only, a permutation of channel-2 rows leaves the
proposal set identical, so the whole screen can be re-adjudicated without re-proposing.

  M1  M = 5 within-token-class permutations of the channel-2 rows (lexical blocks of
      42/43: leading_space x capitalized), 43b's cached cells, L1-L7 rules, and CLOSURE
      EVALUATED FIRST -- a closure failure is FAIL regardless of cell saturation. This
      reverses 43b's ordering and returns the ordinal control to FAIL.
  M2  the three fixed readings; which one obtains is decided by the permuted PASS counts.
  M3  frequency-partial: for the 7 non-calendar passers and strict-12, two-channel rho
      recomputed after the log-count rank is partialled out of BOTH channels' angle ranks,
      against the same cell's p99 recomputed on identically partialled cluster-like nulls.

Two exact shortcuts, both declared, neither an approximation of the reported quantity:
  S1  channel-1 quantities (gap_ratio, angle rank) do not depend on the permutation and are
      computed once for all 358,693 proposals, then reused by every permutation.
  S2  a candidate in a SATURATED cell can never PASS under L1: L1 requires rho == 1.0 AND the
      cell's conditional exact-1.0 rate < 0.01, and all six saturated cells measured by 43b
      have exact-1.0 rates 0.0144-0.0351. Cells are not recomputed under permutation, so
      those candidates are non-PASS in every permutation and their channel-2 work is skipped.
      This is exact for the PASS counts M1 asks for.
"""
import argparse, gc, hashlib, importlib.util, json, os, sys, time
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


S43B = _load("s43b", "scripts/43b_assembly_screen_addendumL.py")
S43 = S43B.S43
S42 = S43B.S42
GAP = S42.GAP_RATIO_CLOSURE
KMIN = S42.K_MIN_ADJUDICABLE
PCT = S42.THRESHOLD_PERCENTILE
R10 = S42.R10_DROP
D_MODEL = S42.D_MODEL

M_PERM = 5
N_PARTIAL_NULL = 20_000
EXACT1_MAX = 0.01                 # L1's saturated-cell escape condition
SOURCE_43 = "results/phase4_assembly_screen.json"
SOURCE_43B = "results/phase4_assembly_screen_addendumL.json"
PROP_CACHE = None                 # set from --prop-cache


def declared_rules(cfg, targets):
    return dict(
        addenda_in_force=["K1-K8 (via scripts/43)", "L1-L7 (via scripts/43b)",
                          "M1-M3 (docs/phase4_addendum_M_2026-08-22.md, this file)"],
        M1=("the identical 358,693-proposal set is re-adjudicated under M = 5 within-token-"
            "class permutations of the CHANNEL-2 rows. Blocks are the lexical blocks used by "
            "42/43 (scripts/41b resolve_blocks: leading_space x capitalized). The assembly "
            "reads channel 1 only, so the proposal set is permutation-invariant and is "
            "reconstructed, not re-proposed. Cells are 43b's cached (layer, k) cells -- they "
            "are NOT recomputed per permutation, exactly as M1 specifies."),
        closure_first=("CLOSURE IS EVALUATED FIRST. gap_ratio >= 2.5 -> FAIL, regardless of "
                       "whether the candidate's cell is saturated. This reverses 43b's "
                       "ordering (43b tested the L1 saturated branch before closure, which is "
                       "why its ordinal control read NOT ADJUDICABLE); under this ordering the "
                       "ordinal control returns to FAIL. Applied identically to the "
                       "unpermuted reference run and to every permutation."),
        order_conjunct=("L1 + L2 exactly as 43b: in a saturated cell the order conjunct passes "
                        "iff rho == 1.0 AND the cell's conditional exact-1.0 rate < 0.01, else "
                        "the cell is NOT ADJUDICABLE; otherwise rho > max-seed p99 is PASS, "
                        "rho inside the seed p99 band is MARGINAL, below the min is FAIL."),
        exact_shortcut_S1=("channel-1 gap_ratio and channel-1 angle ranks are permutation-"
                           "invariant and are computed once for all proposals and reused by "
                           "every permutation. Exact, not an approximation."),
        exact_shortcut_S2=("a candidate whose (layer, k) cell is SATURATED cannot PASS under "
                           "L1, because every saturated cell measured by 43b has a conditional "
                           f"exact-1.0 rate above {EXACT1_MAX} (0.0144-0.0351) and cells are "
                           "not recomputed under permutation. Channel-2 work is therefore "
                           "skipped for those candidates in the permuted runs. Exact for the "
                           "PASS counts M1 asks for; those candidates are counted and reported "
                           "separately as NOT ADJUDICABLE (closure-first still sends the ones "
                           "that fail closure to FAIL)."),
        loo_short_circuit=("43b's short-circuit, carried and stated: the LOO conjunct is "
                           "evaluated only where it can change a verdict. In the permuted runs "
                           "that means only candidates whose k-cell order state is PASS, which "
                           "is EXACT for the PASS counts (a candidate that is not PASS at k "
                           "cannot be PASS overall). MARGINAL counts in the permuted runs are "
                           "therefore reported as an order-state-only UPPER BOUND and labelled "
                           "as such. The unpermuted reference run evaluates the LOO for every "
                           "candidate whose order state is PASS or MARGINAL."),
        no_subsampling="every one of the 358,693 proposals is adjudicated in every permutation",
        M3_partial=("for each target set S: c = rank of log10(corpus count) within S; r1, r2 = "
                    "the channel-1 and channel-2 angle ranks (42's plane_angles convention, "
                    "angles about the fitted circle centre); each rank is residualised on c by "
                    "ordinary least squares within S; the residual is re-ranked and the "
                    "IDENTICAL max-over-2k-cyclic-shifts-and-reflections Spearman is applied to "
                    "the two residual orders. The survival threshold is that cell's p99 "
                    f"recomputed on {N_PARTIAL_NULL} cluster-like draws put through the same "
                    "partialling, conditioned on closure exactly as the unpartialled null."),
        M3_targets=targets,
        reporting=("the 7 non-calendar passers are identified by member_hash and by a class "
                   "description only; their token strings are not quoted."),
        proposal_reconstruction=("43's assembly re-run through the IMPORTED 43 functions, "
                                 "asserted against 43b's stored per-cell counts and the "
                                 "358,693 total before anything is adjudicated. A flat "
                                 "members/offsets cache may be written to and read from the "
                                 "scratchpad; it carries no verdict content and its contents "
                                 "are re-asserted against the same stored counts on load."),
        no_mps="CPU only; no forward passes; the SAE cache is not touched",
        config=cfg)


# ------------------------------------------------------------------ statistics ----
def rho_from_ranks(ra, rb):
    """42's two_channel_rho inner loop, with the two rank vectors supplied directly."""
    k = len(ra)
    best = -2.0
    for flip in (1, -1):
        for shift in range(k):
            ref = S42._rank_from_angles((flip * rb + shift) % k, seed=0)
            d = ra - ref
            best = max(best, 1.0 - 6.0 * float(d @ d) / (k * (k * k - 1)))
    return float(best)


def partial_rank(r, c):
    """Residualise rank vector r on rank vector c by OLS within the set, then re-rank."""
    r = np.asarray(r, dtype=np.float64)
    c = np.asarray(c, dtype=np.float64)
    cc = c - c.mean()
    rc = r - r.mean()
    den = float(cc @ cc)
    resid = rc - (float(cc @ rc) / den) * cc if den > 0 else rc
    out = np.empty(len(resid), dtype=np.float64)
    out[np.lexsort((np.arange(len(resid)), resid))] = np.arange(len(resid), dtype=np.float64)
    return out


def partialled_rho(X1m, X2m, logc):
    """M3's frequency-partialled two-channel rho for one member set."""
    if len(X1m) < 3 or S42.degenerate(np.asarray(X1m, float)) or S42.degenerate(np.asarray(X2m, float)):
        return None
    a1, _ = S42.plane_angles(X1m)
    a2, _ = S42.plane_angles(X2m)
    r1 = S42._rank_from_angles(a1, 0)
    r2 = S42._rank_from_angles(a2, 0)
    c = S42._rank_from_angles(logc, 0)
    return rho_from_ranks(partial_rank(r1, c), partial_rank(r2, c))


def order_state(rho, cell):
    """L1 + L2 order conjunct against a cached 43b cell."""
    if cell is None or cell.get("p99") is None:
        return "NOT ADJUDICABLE", "empty conditional null"
    if cell["saturated"]:
        if rho is not None and rho >= 1.0 and (cell["exact1_rate"] or 1.0) < EXACT1_MAX:
            return "PASS", f"L1 saturated, rho == 1.0, exact1 {cell['exact1_rate']:.4f}"
        return "NOT ADJUDICABLE", (f"L1 saturated cell (p99 = 1.0, exact1 "
                                   f"{cell['exact1_rate']:.4f})")
    lo, hi = cell["p99_min"], cell["p99_max"]
    if rho is None:
        return "NOT ADJUDICABLE", "rho undefined"
    if rho > hi:
        return "PASS", f"rho {rho:.4f} > max seed p99 {hi:.4f}"
    if rho >= lo:
        return "MARGINAL", f"rho {rho:.4f} in seed band [{lo:.4f}, {hi:.4f}]"
    return "FAIL", f"rho {rho:.4f} < min seed p99 {lo:.4f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/phase4_full_multiplicity_null.json")
    ap.add_argument("--layers", default="6,7")
    ap.add_argument("--m-perm", type=int, default=M_PERM)
    ap.add_argument("--n-partial-null", type=int, default=N_PARTIAL_NULL)
    ap.add_argument("--max-seeds", type=int, default=0)
    ap.add_argument("--prop-cache", default="")
    a = ap.parse_args()
    t0 = time.time()
    layers = [int(x) for x in a.layers.split(",")]

    # ---- source runs (results JSON, not coordinates): needed to declare the targets ----
    prev43 = json.loads((REPO / SOURCE_43).read_text())
    prevL = json.loads((REPO / SOURCE_43B).read_text())
    assert prevL["status"] == "COMPLETE" and prev43["status"] == "COMPLETE"
    cells = {}
    for c in prevL["null_cell_inventory_after_controls"]:
        cells[(int(c["layer"]), int(c["k"]))] = c
    sat_cells = {k: v for k, v in cells.items() if v["saturated"]}
    for k, v in sat_cells.items():
        assert v["exact1_rate"] >= EXACT1_MAX, (
            f"saturated cell {k} has exact1 {v['exact1_rate']} < {EXACT1_MAX}; shortcut S2 "
            f"is NOT valid and this run must be re-declared")
    seen, targets = set(), []
    for p in prevL["summary"]["passers"]:
        key = (int(p["layer"]), p["member_hash"])
        if key in seen:
            continue
        seen.add(key)
        targets.append(dict(layer=int(p["layer"]), rule=p["rule"], run=p["run"], k=int(p["k"]),
                            member_hash=p["member_hash"], is_strict12=bool(p["is_strict12"]),
                            n_months=int(p["n_months"])))
    strict12_hash = next(t["member_hash"] for t in targets if t["is_strict12"])
    seven = [t["member_hash"] for t in targets if not t["is_strict12"]]
    assert len(seven) == 7, f"expected 7 non-calendar passers, got {len(seven)}"

    declared_at = time.strftime("%Y-%m-%dT%H:%M:%S")
    cfg = dict(layers=layers, m_perm=int(a.m_perm), n_partial_null=int(a.n_partial_null),
               max_seeds=int(a.max_seeds), source_43=SOURCE_43, source_43b=SOURCE_43B,
               n_cells_cached=len(cells), n_saturated_cells=len(sat_cells))
    stub = dict(status="DECLARATION STUB -- rules frozen, no results yet",
                script="scripts/43c_full_multiplicity_null.py", declared_at=declared_at,
                stub_note=("written before any COORDINATE read. The two source results JSONs "
                           "are read first, and only to name the declared targets and to load "
                           "43b's cached cells; no npz and no cache file has been opened."),
                source_runs=dict(
                    s43=dict(file=SOURCE_43, declared_at=prev43["declared_at"],
                             n_proposals=prev43["summary"]["n_proposals_total"]),
                    s43b=dict(file=SOURCE_43B, declared_at=prevL["declared_at"],
                              tally=prevL["verdict_tally"])),
                targets=targets, seven_non_calendar=seven, strict12_hash=strict12_hash,
                saturated_cells=[dict(layer=k[0], k=k[1], exact1_rate=v["exact1_rate"])
                                 for k, v in sorted(sat_cells.items())],
                declared_rules=declared_rules(cfg, targets))
    Path(a.out).write_text(json.dumps(stub, indent=2))
    print("=== DECLARATION STUB WRITTEN (before any coordinate read) ===", flush=True)
    print(f"targets: {len(targets)} unique passer sets ({len(seven)} non-calendar + strict-12); "
          f"cells {len(cells)} of which {len(sat_cells)} saturated", flush=True)
    out = dict(stub)
    out["status"] = "RUNNING"

    # ---------------------------------------------------------------- data (L5) ----
    zr = np.load(REPO / S43.RAW_NPZ)
    pool = zr["pool_ids"].astype(np.int64)
    counts = zr["counts"].astype(np.int64)
    n_types = len(pool)
    pos = {int(t): i for i, t in enumerate(pool)}
    zd = np.load(REPO / S43.DELTAS_NPZ)
    X1, X2, NNt = {}, {}, {}
    for L in layers:
        keep = np.ones(D_MODEL, dtype=bool)
        keep[R10[L]] = False
        X1[L] = np.ascontiguousarray(zr[f"R_L{L}"].astype(np.float32)[:, keep].astype(np.float64))
        X2[L] = np.ascontiguousarray(zd[f"L{L}_lag1"].astype(np.float32).astype(np.float64))
        assert np.isfinite(X1[L]).all() and np.isfinite(X2[L]).all()
        Xc = X1[L] - X1[L].mean(0, keepdims=True)
        Xn = (Xc / np.maximum(np.linalg.norm(Xc, axis=1, keepdims=True), 1e-12)).astype(np.float32)
        Sm = Xn @ Xn.T
        np.fill_diagonal(Sm, -2.0)
        NNt[L] = np.argsort(-Sm, axis=1)[:, :S43.N_CAND_NN].astype(np.int64)
        del Sm, Xc, Xn
        gc.collect()
    del zr, zd
    gc.collect()
    logcount = np.log10(np.maximum(counts.astype(np.float64), 1.0))

    from transformers import AutoTokenizer
    from gfdisc.env import MODEL_NAME
    tk = AutoTokenizer.from_pretrained(MODEL_NAME, token=os.environ.get("HF_TOKEN"))
    strings = [(tk.convert_ids_to_tokens(int(t)) or tk.decode([int(t)])).replace("▁", " ")
               for t in pool]
    blocks = S42.resolve_blocks(strings)
    print(f"pool {n_types}; lexical blocks {len(blocks)} sizes {[len(b) for b in blocks]}",
          flush=True)
    ord_rows = []
    for w in S42.ORDINAL_WORDS:
        e = tk.encode(w, add_special_tokens=False)
        if len(e) == 1 and int(e[0]) in pos:
            ord_rows.append(pos[int(e[0])])
    ord_rows = np.asarray(sorted(ord_rows))
    S12 = np.asarray(sorted(pos[t] for t in S43.MONTH_TOKEN_IDS_13 if t != 5656))
    assert S43.member_hash(S12) == strict12_hash

    # ---------------------------------------- proposal reconstruction (43 exact) ----
    print("\n=== reconstructing the 358,693-proposal set ===", flush=True)
    cache = Path(a.prop_cache) if a.prop_cache else None
    flat = offs = meta_layer = meta_k = None
    recon = {}
    loaded = False
    if cache is not None and cache.exists():
        z = np.load(cache)
        flat, offs = z["flat"], z["offs"]
        meta_layer, meta_k = z["layer"], z["k"]
        recon = json.loads(str(z["recon"]))
        loaded = True
        print(f"  loaded proposal cache {cache} ({len(offs)-1} proposals)", flush=True)
    if not loaded:
        members_all, lay_all = [], []
        for L in layers:
            seeds = S43.build_seeds(NNt[L])
            if a.max_seeds:
                seeds = seeds[:a.max_seeds]
            for rule in ("A1", "A2"):
                p, st = S43.run_assembly(X1[L], X2[L], NNt[L], seeds, rule, tag=f"L{L} ")
                stored = (prevL.get("reconstruction", {}).get(f"L{L}_{rule}", {})
                          .get("stored_by_43"))
                recon[f"L{L}_{rule}"] = dict(n_unique_k_ge_9=st["n_unique_k_ge_9"],
                                             stored_by_43=stored,
                                             reproduces=bool(stored == st["n_unique_k_ge_9"]))
                print(f"  L{L} {rule}: {st['n_unique_k_ge_9']} (43 stored {stored}; reproduces "
                      f"{recon[f'L{L}_{rule}']['reproduces']}) {st['seconds']}s", flush=True)
                for v in p.values():
                    members_all.append(v["members"]); lay_all.append(L)
        carriers_src = prev43.get("carriers_resolved", [])
        assert carriers_src, ("scripts/43's results carry the resolved carriers; 43b's do not. "
                              "Reading them from the wrong source silently drops run (b).")
        for c in carriers_src:
            L = int(c["layer"])
            if L not in layers:
                continue
            rows = np.asarray(sorted(int(x) for x in c["members"]))
            cseeds = [(int(rows[i]), int(rows[j])) for i in range(len(rows))
                      for j in range(i + 1, len(rows))]
            for rule in ("A1", "A2"):
                p, _ = S43.run_assembly(X1[L], X2[L], NNt[L], cseeds, rule, log_every=0,
                                        cand_restrict=rows)
                for v in p.values():
                    if set(v["members"]) <= set(int(x) for x in rows):
                        members_all.append(v["members"]); lay_all.append(L)
        offs = np.zeros(len(members_all) + 1, dtype=np.int64)
        for i, m in enumerate(members_all):
            offs[i + 1] = offs[i] + len(m)
        flat = np.empty(offs[-1], dtype=np.int32)
        for i, m in enumerate(members_all):
            flat[offs[i]:offs[i + 1]] = np.asarray(m, dtype=np.int32)
        meta_layer = np.asarray(lay_all, dtype=np.int8)
        meta_k = (offs[1:] - offs[:-1]).astype(np.int16)
        del members_all
        gc.collect()
        if cache is not None:
            cache.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(cache, flat=flat, offs=offs, layer=meta_layer, k=meta_k,
                                recon=json.dumps(recon))
            print(f"  wrote proposal cache {cache}", flush=True)
    n_prop = len(offs) - 1
    if not a.max_seeds:
        for key, v in recon.items():
            assert v["reproduces"] or v["stored_by_43"] is None, \
                f"reconstruction mismatch {key}: got {v['n_unique_k_ge_9']}, 43 stored {v['stored_by_43']}"
    expected_total = prev43["summary"]["n_proposals_total"]
    print(f"  total proposals {n_prop} (43 reported {expected_total}); "
          f"match {n_prop == expected_total}", flush=True)
    if not a.max_seeds:
        assert n_prop == expected_total, "proposal count differs from 43"
    out["reconstruction"] = dict(per_cell=recon, n_proposals=int(n_prop),
                                 expected_total=int(expected_total),
                                 matches=bool(n_prop == expected_total),
                                 from_cache=bool(loaded))

    # ------------------------------- S1: permutation-invariant channel-1 quantities ----
    print("\n=== channel-1 invariants (computed once, reused by every permutation) ===",
          flush=True)
    t1 = time.time()
    gap1 = np.full(n_prop, np.nan)
    rank1 = [None] * n_prop
    sat = np.zeros(n_prop, dtype=bool)
    has_cell = np.zeros(n_prop, dtype=bool)
    for i in range(n_prop):
        L = int(meta_layer[i])
        m = flat[offs[i]:offs[i + 1]]
        P = X1[L][m]
        g = S42.gap_ratio_of(P)
        gap1[i] = np.nan if g is None else g
        cell = cells.get((L, int(meta_k[i])))
        has_cell[i] = cell is not None and cell.get("p99") is not None
        sat[i] = bool(cell is not None and cell["saturated"])
        if g is not None and g < GAP and has_cell[i] and not sat[i]:
            a1, _ = S42.plane_angles(P)
            rank1[i] = S42._rank_from_angles(a1, 0)
        if (i + 1) % 50_000 == 0:
            print(f"    channel-1 {i+1}/{n_prop} {time.time()-t1:.0f}s", flush=True)
    closes = np.isfinite(gap1) & (gap1 < GAP)
    need2 = closes & has_cell & (~sat)
    print(f"  closure passes {int(closes.sum())}/{n_prop}; saturated-cell {int(sat.sum())}; "
          f"channel-2 needed for {int(need2.sum())} ({time.time()-t1:.0f}s)", flush=True)
    out["channel1_invariants"] = dict(
        n_closure_pass=int(closes.sum()), n_closure_fail=int((~closes).sum()),
        n_in_saturated_cell=int(sat.sum()), n_needing_channel2=int(need2.sum()),
        seconds=round(time.time() - t1, 1))

    idx2 = np.flatnonzero(need2)
    hash_of = {}

    def adjudicate_one(i, D, want_loo, cell):
        """closure-first L1/L2 verdict for proposal i against channel-2 matrix D."""
        L = int(meta_layer[i])
        m = flat[offs[i]:offs[i + 1]]
        if not closes[i]:
            return "FAIL", None, None
        if sat[i] or not has_cell[i]:
            return "NOT ADJUDICABLE", None, None
        P2 = D[m]
        if len(P2) < 3 or S42.degenerate(P2):
            return "NOT ADJUDICABLE", None, None
        a2, _ = S42.plane_angles(P2)
        rho = rho_from_ranks(rank1[i], S42._rank_from_angles(a2, 0))
        st, _ = order_state(rho, cell)
        if st == "FAIL" or st == "NOT ADJUDICABLE":
            return st, rho, None
        if not want_loo:
            return st + " (order-state only)", rho, None
        cell_loo = cells.get((L, int(meta_k[i]) - 1))
        states, loo_close = [], True
        for j in range(len(m)):
            sub = np.delete(m, j)
            g, r = S42.cell_stats(X1[L][sub], D[sub])
            if not (g is not None and g < GAP):
                loo_close = False
                break
            states.append(order_state(r, cell_loo)[0])
        if not loo_close:
            return "FAIL", rho, None
        if "NOT ADJUDICABLE" in states:
            return "NOT ADJUDICABLE", rho, states
        if "FAIL" in states:
            return "FAIL", rho, states
        if "MARGINAL" in states or st == "MARGINAL":
            return "MARGINAL", rho, states
        return "PASS", rho, states

    D0 = {L: X2[L] for L in layers}

    def run_for(Dmap, label, full_loo):
        t = time.time()
        tally, passers, marg = {}, [], 0
        for n, i in enumerate(idx2):
            i = int(i)
            L = int(meta_layer[i])
            cell = cells[(L, int(meta_k[i]))]
            st, rho, _ = adjudicate_one(i, Dmap[L], full_loo, cell)
            base = st.split(" (")[0]
            tally[base] = tally.get(base, 0) + 1
            if base == "MARGINAL":
                marg += 1
            if base == "PASS":
                m = flat[offs[i]:offs[i + 1]]
                h = hash_of.get(i)
                if h is None:
                    h = S43.member_hash(m); hash_of[i] = h
                passers.append(dict(index=i, layer=L, k=int(meta_k[i]), member_hash=h,
                                    gap_ratio=float(gap1[i]), rho=float(rho)))
            if (n + 1) % 40_000 == 0:
                print(f"    [{label}] {n+1}/{len(idx2)}, {len(passers)} PASS, "
                      f"{time.time()-t:.0f}s", flush=True)
        tally["FAIL_closure_skipped"] = int((~closes).sum())
        tally["NOT_ADJUDICABLE_saturated_skipped"] = int((closes & sat).sum())
        uniq = sorted({p["member_hash"] for p in passers})
        return dict(label=label, tally=tally, n_pass=len(passers), n_unique_pass=len(uniq),
                    unique_hashes=uniq, passers=passers,
                    n_marginal_order_state=marg, full_loo=bool(full_loo),
                    seconds=round(time.time() - t, 1))

    reference = run_for(D0, "unpermuted", True)
    print(f"  unpermuted: PASS {reference['n_pass']} ({reference['n_unique_pass']} unique), "
          f"MARGINAL {reference['n_marginal_order_state']} ({reference['seconds']}s)",
          flush=True)
    out["unpermuted_reference"] = reference
    ref_unique = set(reference["unique_hashes"])
    print(f"  overlap with 43b's 8 unique passers: "
          f"{len(ref_unique & set(seven + [strict12_hash]))}/8", flush=True)

    # ---------------------------------------------------------- controls (M1) ----
    ctrl = {}
    for L in layers:
        for name, rows in (("strict12", S12), ("ordinal", ord_rows)):
            g = S42.gap_ratio_of(X1[L][rows])
            cell = cells.get((L, len(rows)))
            if g is None or g >= GAP:
                st, rho = "FAIL", None
            else:
                a1, _ = S42.plane_angles(X1[L][rows])
                a2, _ = S42.plane_angles(X2[L][rows])
                rho = rho_from_ranks(S42._rank_from_angles(a1, 0),
                                     S42._rank_from_angles(a2, 0))
                st = order_state(rho, cell)[0]
            ctrl[f"{name}_L{L}"] = dict(k=int(len(rows)), gap_ratio=g, rho=rho,
                                        order_state_or_verdict=st,
                                        closure_pass=bool(g is not None and g < GAP),
                                        cell_saturated=bool(cell and cell["saturated"]))
            print(f"  control {name} L{L}: gap {g:.4f} -> {st}"
                  + (f", rho {rho:.4f}" if rho is not None else ""), flush=True)
    out["controls_closure_first"] = ctrl
    for L in layers:
        assert ctrl[f"ordinal_L{L}"]["order_state_or_verdict"] == "FAIL", \
            f"ordinal control at L{L} must be FAIL under closure-first"

    # ------------------------------------------------------------ M1 permutations ----
    print(f"\n=== M1: {a.m_perm} within-token-class permutations of channel 2 ===", flush=True)
    perms = []
    for p_i in range(int(a.m_perm)):
        rng = np.random.default_rng(43_300_000 + p_i)
        Dp = {}
        for L in layers:
            perm = np.arange(n_types)
            for b in blocks:
                if len(b) > 1:
                    perm[b] = b[rng.permutation(len(b))]
            Dp[L] = X2[L][perm]
        r = run_for(Dp, f"perm{p_i}", True)
        r["seed"] = int(43_300_000 + p_i)
        u = set(r["unique_hashes"])
        r["overlap_with_seven"] = sorted(u & set(seven))
        r["n_overlap_with_seven"] = len(u & set(seven))
        r["strict12_passes"] = bool(strict12_hash in u)
        perms.append(r)
        print(f"  perm{p_i}: PASS {r['n_pass']} ({r['n_unique_pass']} unique); "
              f"overlap with the 7 = {r['n_overlap_with_seven']}; strict-12 passes "
              f"{r['strict12_passes']}; MARGINAL {r['n_marginal_order_state']} "
              f"({r['seconds']}s)", flush=True)
        del Dp
        gc.collect()
    out["permutations"] = perms
    mean_pass = float(np.mean([r["n_pass"] for r in perms]))
    mean_unique = float(np.mean([r["n_unique_pass"] for r in perms]))
    mean_overlap = float(np.mean([r["n_overlap_with_seven"] for r in perms]))
    n_strict12 = int(sum(r["strict12_passes"] for r in perms))
    if mean_unique >= 3:
        reading = "(i)"
        reading_text = ("permuted runs yield passers at a rate comparable to 7: the 7 are "
                        "search noise at this multiplicity. NO DISCOVERY; the measured "
                        "full-multiplicity false rate is the reportable number.")
    elif mean_overlap >= 1:
        reading = "(ii)"
        reading_text = ("the same sets pass under permutation: order agreement is "
                        "class-generic. Leakage confirmed, mechanism = class-level successor "
                        "dynamics. NO DISCOVERY.")
    elif mean_unique < 1:
        reading = "(iii)"
        reading_text = ("permuted runs yield ~0 passers and the 7 do not survive permutation: "
                        "candidate structure remains open; each of the 7 goes to the M3 "
                        "frequency-partial check and, if it survives, to the behavioural "
                        "tier (47) before any naming.")
    else:
        reading = "between (i) and (iii)"
        reading_text = ("mean unique permuted passers is in [1, 3) with no overlap with the 7: "
                        "neither M2 branch obtains cleanly; the measured rate is reported and "
                        "the coordinator adjudicates.")
    print(f"\n  mean PASS {mean_pass:.2f}; mean unique {mean_unique:.2f}; mean overlap with "
          f"the 7 {mean_overlap:.2f}; strict-12 passes in {n_strict12}/{len(perms)} perms",
          flush=True)
    print(f"  M2 reading: {reading} -- {reading_text}", flush=True)
    out["M2_reading"] = dict(reading=reading, text=reading_text, mean_pass=mean_pass,
                             mean_unique_pass=mean_unique,
                             mean_overlap_with_seven=mean_overlap,
                             n_perms_where_strict12_passes=n_strict12,
                             n_perms=len(perms))

    # ------------------------------------------------------------------ M3 ----
    print("\n=== M3: frequency-partialled two-channel rho ===", flush=True)
    idx_by_hash = {}
    for i in range(n_prop):
        if int(meta_k[i]) < KMIN:
            continue
        h = hash_of.get(i)
        if h is None:
            h = S43.member_hash(flat[offs[i]:offs[i + 1]])
            hash_of[i] = h
        if h in seven and h not in idx_by_hash:
            idx_by_hash[h] = i
    partial_cells = {}

    def partial_cell(L, k, n_draws, seed):
        key = (L, k)
        if key in partial_cells:
            return partial_cells[key]
        Xc = X1[L] - X1[L].mean(0, keepdims=True)
        Xn = (Xc / np.maximum(np.linalg.norm(Xc, axis=1, keepdims=True), 1e-12)).astype(np.float32)
        rng = np.random.default_rng(seed)
        vals = []
        for _ in range(n_draws):
            s = int(rng.integers(n_types))
            sim = Xn @ Xn[s]
            sim[s] = 2.0
            m = np.argsort(sim)[::-1][:k]
            g = S42.gap_ratio_of(X1[L][m])
            if g is None or g >= GAP:
                continue
            r = partialled_rho(X1[L][m], X2[L][m], logcount[m])
            if r is not None:
                vals.append(r)
        v = np.asarray(vals)
        res = dict(layer=L, k=k, n_draws=int(n_draws), n_conditional=int(len(v)),
                   p99=float(np.percentile(v, PCT, method="higher")) if len(v) else None,
                   exact1_rate=float((v >= 1.0).mean()) if len(v) else None,
                   saturated=bool(len(v) and float(np.percentile(v, PCT, method="higher")) >= 1.0),
                   seed=int(seed))
        partial_cells[key] = res
        del Xc, Xn
        gc.collect()
        print(f"    partial cell L{L} k={k}: n_cond {res['n_conditional']} p99 {res['p99']} "
              f"exact1 {res['exact1_rate']} saturated {res['saturated']}", flush=True)
        return res

    m3 = []
    for t in targets:
        L, k, h = t["layer"], t["k"], t["member_hash"]
        if t["is_strict12"]:
            rows_by_layer = {LL: S12 for LL in layers}
        else:
            i = idx_by_hash.get(h)
            if i is None:
                m3.append(dict(**t, error="member set not found in the reconstruction"))
                continue
            rows_by_layer = {L: flat[offs[i]:offs[i + 1]]}
        for LL, rows in rows_by_layer.items():
            g = S42.gap_ratio_of(X1[LL][rows])
            _, rho_plain = S42.cell_stats(X1[LL][rows], X2[LL][rows])
            rp = partialled_rho(X1[LL][rows], X2[LL][rows], logcount[rows])
            pc = partial_cell(LL, len(rows), int(a.n_partial_null), 43_400_000 + 100 * LL + len(rows))
            st, det = order_state(rp, dict(p99=pc["p99"], p99_min=pc["p99"], p99_max=pc["p99"],
                                           saturated=pc["saturated"],
                                           exact1_rate=pc["exact1_rate"]))
            unp = cells.get((LL, len(rows)))
            m3.append(dict(member_hash=h, layer=LL, k=int(len(rows)),
                           is_strict12=t["is_strict12"],
                           gap_ratio=g, rho_unpartialled=rho_plain,
                           rho_partialled=rp,
                           partial_cell_p99=pc["p99"],
                           partial_cell_saturated=pc["saturated"],
                           partial_cell_n_conditional=pc["n_conditional"],
                           unpartialled_cell_p99_band=None if unp is None else
                           [unp["p99_min"], unp["p99_max"]],
                           survives_partial=bool(st == "PASS"),
                           partial_order_state=st, detail=det))
            print(f"  M3 {h} L{LL} k={len(rows)}: rho {rho_plain:.4f} -> partialled "
                  f"{rp:.4f} vs partial p99 {pc['p99']} -> {st}", flush=True)
    out["M3_frequency_partial"] = m3
    out["M3_partial_cells"] = [v for v in partial_cells.values()]
    n_surv = sum(1 for r in m3 if r.get("survives_partial") and not r.get("is_strict12"))
    s12_surv = [r for r in m3 if r.get("is_strict12") and r.get("survives_partial")]

    out["summary"] = dict(
        n_proposals=int(n_prop),
        unpermuted_pass=reference["n_pass"], unpermuted_unique=reference["n_unique_pass"],
        unpermuted_reproduces_43b_passers=bool(
            set(reference["unique_hashes"]) == set(seven + [strict12_hash])),
        permuted_pass_counts=[r["n_pass"] for r in perms],
        permuted_unique_counts=[r["n_unique_pass"] for r in perms],
        permuted_overlap_with_seven=[r["n_overlap_with_seven"] for r in perms],
        mean_permuted_pass=mean_pass, mean_permuted_unique=mean_unique,
        mean_overlap_with_seven=mean_overlap,
        n_perms_strict12_passes=n_strict12,
        M2_reading=reading,
        ordinal_control_closure_first=[ctrl[f"ordinal_L{L}"]["order_state_or_verdict"]
                                       for L in layers],
        n_seven_surviving_frequency_partial=int(n_surv),
        strict12_survives_frequency_partial=[dict(layer=r["layer"], state=r["partial_order_state"])
                                             for r in s12_surv])
    out["status"] = "COMPLETE"
    out["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    out["runtime_seconds"] = round(time.time() - t0, 1)
    Path(a.out).write_text(json.dumps(out, indent=2, default=float))
    print(f"\nwrote {a.out} in {out['runtime_seconds']}s", flush=True)
    print(json.dumps(out["summary"], indent=2, default=float), flush=True)


if __name__ == "__main__":
    main()
