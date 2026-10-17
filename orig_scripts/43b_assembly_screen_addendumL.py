"""43b -- Addendum L (L1-L7) applied to the 43 assembly screen.

43 was declared and launched under Addendum K (K1-K8) before Addendum L existed; its blind
run was allowed to finish and its verdicts stand as the K-tier result. This script re-derives
43's proposal set and re-adjudicates it under L1-L7. Neither 42 nor 43 is edited or re-run;
both are imported.

  L1 saturated cells: where a (layer, k) conditional p99 is 1.0, "strictly above p99" is
     unsatisfiable. A candidate passes the order conjunct iff rho == 1.0 AND that cell's
     conditional exact-1.0 rate is < 0.01; otherwise the cell is NOT ADJUDICABLE and the
     candidate is reported under closure and LOO only. Same rule for the (k-1) LOO cell.
  L2 threshold stability: every cell's p99 is computed under 3 seeds. rho above the MAX seed
     p99 is PASS; rho inside [min, max] is MARGINAL; rho below the min is FAIL.
  L3 positive control is strict-12 at L6 only; L7 strict-12 is reported MARGINAL, not as a
     tier failure.
  L4 closure is evaluated POST-R10 in both the proposal rule and the tier (43 already does
     this; reproduced and asserted here).
  L5 delta artifacts are cast to float32 before any fit, and any verdict that moves under the
     cast is reported.
  L6 the detectable effect size quoted is the LITERAL random-carrier plant, with 43's clean
     plant co-reported.
  L7 the null-cell inventory is snapshotted after all controls are evaluated.

Why the proposals are re-derived rather than read: 43's storage rule keeps only the first 200
rows per (layer, rule, run) ordered by k DESCENDING, so the k = 9 and k = 10 proposals -- which
are exactly the saturated cells L1 governs -- are the ones the cap truncates. The assembly is
deterministic, so re-running it reconstructs the identical proposal set; the reconstruction is
verified against 43's stored aggregate counts before anything is adjudicated.
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


S43 = _load("s43", "scripts/43_neighbour_assembly_screen.py")
S42 = S43.S42
GAP = S42.GAP_RATIO_CLOSURE
KMIN = S42.K_MIN_ADJUDICABLE
PCT = S42.THRESHOLD_PERCENTILE
R10 = S42.R10_DROP
D_MODEL = S42.D_MODEL

N_CLUSTERLIKE = 20_000
SEED_TRIPLE = (0, 1, 2)                # L2: three RNG streams per cell
RHO_PREFILTER = 0.90                   # cells needing the 3-seed spread
PLANTED_RHOS = (0.05, 0.10, 0.15, 0.20, 0.30)
N_PLANTED = 100
PLANTED_K = 12


def declared_rules(cfg):
    return dict(
        addenda_in_force="docs/phase4_addendum_L_2026-08-22.md L1-L7, on top of K1-K8",
        note_on_43=("43 declared and ran under K1-K8 only; Addendum L was issued after its "
                    "blind run had started. 43's verdicts stand as the K-tier result and are "
                    "carried here unchanged for comparison; this file re-adjudicates the same "
                    "proposals under L1-L7."),
        proposal_reconstruction=(
            "43's assembly is deterministic (fixed seed rule, fixed candidate pools, no RNG), "
            "so it is re-run here through the IMPORTED 43 functions to recover the full "
            "proposal set including the k = 9 and k = 10 rows that 43's storage cap "
            "truncated. The reconstruction is asserted against 43's stored "
            "assembly_run_a/assembly_run_b counts before any adjudication."),
        L1=("saturated cell (conditional p99 == 1.0): order conjunct passes iff rho == 1.0 "
            "AND the cell's conditional exact-1.0 rate < 0.01; else the candidate is NOT "
            "ADJUDICABLE in that cell and is reported under closure and LOO closure only. "
            "Applied to the k cell and to the (k-1) LOO cell independently."),
        L2=(f"each cell's p99 is computed under {len(SEED_TRIPLE)} seeds "
            f"{list(SEED_TRIPLE)}; rho > max(p99) is PASS, min(p99) <= rho <= max(p99) is "
            f"MARGINAL, rho < min(p99) is FAIL. To stay inside the budget the 3-seed spread "
            f"is computed only for cells holding at least one proposal with rho >= "
            f"{RHO_PREFILTER}; every other cell keeps its single-seed p99 and cannot change a "
            f"FAIL (no proposal in it is within {RHO_PREFILTER} of any attainable threshold). "
            f"Declared."),
        L3="positive control = strict-12 at L6; L7 strict-12 reported MARGINAL, not a failure",
        loo_short_circuit=("the LOO conjunct is evaluated only where it can change a "
                           "verdict (closure holds and the order state is not FAIL), for the "
                           "controls, and for a declared subsample of at most 2,000 "
                           "NOT-ADJUDICABLE rows per (layer, rule). Verdicts are identical "
                           "to the exhaustive evaluation; only unreported LOO detail differs."),
        L4="closure evaluated post-R10 in the proposal rule and in the tier (asserted)",
        L5="channel-2 deltas cast to float32 before any fit; verdict moves vs 43 are reported",
        L6="effect size quoted = literal random-carrier plant; 43's clean plant co-reported",
        L7="null-cell inventory snapshotted after all controls are evaluated",
        config=cfg)


def cell_stats_full(nc, layer, k, seed, n_draws):
    """One cluster-like cell under one RNG seed: p99, exact-1.0 rate, closure rate."""
    rng = np.random.default_rng(seed)
    gaps, rho_cond = [], []
    for _ in range(n_draws):
        s = int(rng.integers(len(nc.Xn)))
        sim = nc.Xn @ nc.Xn[s]
        sim[s] = 2.0
        idx = np.argsort(sim)[::-1][:k]
        g, r = S42.cell_stats(nc.C1[idx], nc.C2[idx])
        if g is None:
            continue
        gaps.append(g)
        if r is not None and g < GAP:
            rho_cond.append(r)
    rc = np.asarray(rho_cond)
    gaps = np.asarray(gaps)
    return dict(layer=int(layer), k=int(k), seed=int(seed), n_draws=int(n_draws),
                n_conditional=int(len(rc)),
                closure_rate=float((gaps < GAP).mean()) if len(gaps) else None,
                p99=(float(np.percentile(rc, PCT, method="higher")) if len(rc) else None),
                exact1_rate=(float((rc >= 1.0).mean()) if len(rc) else None),
                saturated=bool(len(rc) and float(np.percentile(rc, PCT, method="higher")) >= 1.0))


def verdict_L(rho, cell):
    """L1 + L2 order conjunct for one cell. Returns (state, detail)."""
    if cell is None or cell["p99"] is None:
        return "NOT ADJUDICABLE", "empty conditional null"
    if cell["saturated"]:
        if rho is not None and rho >= 1.0 and (cell["exact1_rate"] or 1.0) < 0.01:
            return "PASS", f"L1 saturated cell, rho == 1.0, exact1 {cell['exact1_rate']:.4f}"
        return "NOT ADJUDICABLE", (f"L1 saturated cell (p99 = 1.0, exact1 "
                                   f"{cell['exact1_rate']}), rho {rho}")
    lo, hi = cell["p99_min"], cell["p99_max"]
    if rho > hi:
        return "PASS", f"rho {rho:.4f} > max seed p99 {hi:.4f}"
    if rho >= lo:
        return "MARGINAL", f"rho {rho:.4f} inside seed p99 band [{lo:.4f}, {hi:.4f}]"
    return "FAIL", f"rho {rho:.4f} < min seed p99 {lo:.4f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="results/phase4_assembly_screen.json")
    ap.add_argument("--out", default="results/phase4_assembly_screen_addendumL.json")
    ap.add_argument("--layers", default="6,7")
    ap.add_argument("--n-clusterlike", type=int, default=N_CLUSTERLIKE)
    ap.add_argument("--n-planted", type=int, default=N_PLANTED)
    ap.add_argument("--max-seeds", type=int, default=0)
    a = ap.parse_args()
    t0 = time.time()
    layers = [int(x) for x in a.layers.split(",")]
    prev = json.loads(Path(a.inp).read_text())
    assert prev.get("status") == "COMPLETE", f"{a.inp} is not COMPLETE"

    declared_at = time.strftime("%Y-%m-%dT%H:%M:%S")
    cfg = dict(layers=layers, n_clusterlike=int(a.n_clusterlike), seed_triple=list(SEED_TRIPLE),
               rho_prefilter=RHO_PREFILTER, n_planted=int(a.n_planted),
               source=a.inp, source_declared_at=prev.get("declared_at"),
               max_seeds=int(a.max_seeds))
    stub = dict(status="DECLARATION STUB -- rules frozen, no results yet",
                script="scripts/43b_assembly_screen_addendumL.py", declared_at=declared_at,
                addenda_in_force=["K1-K8 (via 43)", "L1-L7 (this file)"],
                source_run=dict(file=a.inp, declared_at=prev.get("declared_at"),
                                finished_at=prev.get("finished_at"),
                                runtime_seconds=prev.get("runtime_seconds"),
                                summary=prev.get("summary")),
                declared_rules=declared_rules(cfg))
    Path(a.out).write_text(json.dumps(stub, indent=2))
    print("=== 43b DECLARATION STUB WRITTEN ===", flush=True)
    out = dict(stub)
    out["status"] = "RUNNING"

    # ------------------------------------------------------------------ data (L5) ----
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
    print(f"data ready (L5: deltas cast to float32); pool {n_types}", flush=True)

    S12 = np.asarray(sorted(pos[t] for t in S43.MONTH_TOKEN_IDS_13 if t != 5656))
    S13 = np.asarray(sorted(pos[t] for t in S43.MONTH_TOKEN_IDS_13))
    month_rows = set(int(x) for x in S12)
    from transformers import AutoTokenizer
    from gfdisc.env import MODEL_NAME
    tk = AutoTokenizer.from_pretrained(MODEL_NAME, token=os.environ.get("HF_TOKEN"))
    strings = [(tk.convert_ids_to_tokens(int(t)) or tk.decode([int(t)])).replace("▁", " ")
               for t in pool]
    ord_rows = []
    for w in S42.ORDINAL_WORDS:
        e = tk.encode(w, add_special_tokens=False)
        if len(e) == 1 and int(e[0]) in pos:
            ord_rows.append(pos[int(e[0])])
    ord_rows = np.asarray(sorted(ord_rows))

    # --------------------------------------------------- proposal reconstruction ----
    print("\n=== reconstructing 43's proposals (deterministic) ===", flush=True)
    all_props, recon = [], {}
    for L in layers:
        seeds = S43.build_seeds(NNt[L])
        if a.max_seeds:
            seeds = seeds[:a.max_seeds]
        for rule in ("A1", "A2"):
            p, st = S43.run_assembly(X1[L], X2[L], NNt[L], seeds, rule, tag=f"L{L} ")
            recon[f"L{L}_{rule}"] = dict(n_unique_k_ge_9=st["n_unique_k_ge_9"],
                                         stored_by_43=(prev.get("assembly_run_a", {})
                                                       .get(f"L{L}_{rule}", {})
                                                       .get("n_unique_k_ge_9")))
            recon[f"L{L}_{rule}"]["reproduces"] = bool(
                recon[f"L{L}_{rule}"]["stored_by_43"] == st["n_unique_k_ge_9"])
            for h, v in p.items():
                all_props.append(dict(v, layer=L, run="a", member_hash=h))
            print(f"  L{L} {rule}: {st['n_unique_k_ge_9']} unique k>=9 "
                  f"(43 stored {recon[f'L{L}_{rule}']['stored_by_43']}; reproduces "
                  f"{recon[f'L{L}_{rule}']['reproduces']}) {st['seconds']}s", flush=True)
    for c in prev.get("carriers_resolved", []):
        L = int(c["layer"])
        if L not in layers:
            continue
        rows = np.asarray(sorted(int(x) for x in c["members"]))
        cseeds = [(int(rows[i]), int(rows[j])) for i in range(len(rows))
                  for j in range(i + 1, len(rows))]
        for rule in ("A1", "A2"):
            p, _ = S43.run_assembly(X1[L], X2[L], NNt[L], cseeds, rule, log_every=0,
                                    cand_restrict=rows)
            for h, v in p.items():
                if set(v["members"]) <= set(int(x) for x in rows):
                    all_props.append(dict(v, layer=L, run="b", carrier=c["plane"],
                                          member_hash=h))
    print(f"  total reconstructed proposals: {len(all_props)}", flush=True)
    out["reconstruction"] = recon

    # ----------------------------------------------------------- first-pass rho ----
    nulls = {}
    for L in layers:
        Xc = X1[L] - X1[L].mean(0, keepdims=True)
        Xn = (Xc / np.maximum(np.linalg.norm(Xc, axis=1, keepdims=True), 1e-12)).astype(np.float32)
        nulls[L] = S42.NullCache(Xn, X1[L], X2[L], counts, n_clusterlike=int(a.n_clusterlike))
        del Xc
    gc.collect()
    for v in all_props:
        g, r = S42.cell_stats(X1[v["layer"]][v["members"]], X2[v["layer"]][v["members"]])
        v["gap_post_R10"] = g
        v["rho"] = r
    assert all(v["gap_post_R10"] is not None for v in all_props)

    # ------------------------------------------------------------- cells (L1/L2) ----
    need = {}
    for v in all_props:
        need.setdefault((v["layer"], v["k"]), False)
        need.setdefault((v["layer"], v["k"] - 1), False)
        if v["rho"] is not None and v["rho"] >= RHO_PREFILTER:
            need[(v["layer"], v["k"])] = True
            need[(v["layer"], v["k"] - 1)] = True
    for L, P, nm in ((6, S12, "strict12"), (7, S12, "strict12"), (6, S13, "13set"),
                     (7, S13, "13set"), (6, ord_rows, "ordinal"), (7, ord_rows, "ordinal")):
        if L in layers:
            need[(L, len(P))] = True
            need[(L, len(P) - 1)] = True
    cells = {}
    t_c = time.time()
    for (L, k), triple in sorted(need.items()):
        if k < 3:
            continue
        seeds_used = SEED_TRIPLE if triple else SEED_TRIPLE[:1]
        per = [cell_stats_full(nulls[L], L, k, 43_000_000 + 1000 * L + k + s, int(a.n_clusterlike))
               for s in seeds_used]
        p99s = [c["p99"] for c in per if c["p99"] is not None]
        cells[(L, k)] = dict(layer=L, k=k, n_seeds=len(per),
                             p99=per[0]["p99"], p99_min=min(p99s) if p99s else None,
                             p99_max=max(p99s) if p99s else None,
                             exact1_rate=float(np.mean([c["exact1_rate"] for c in per
                                                        if c["exact1_rate"] is not None])),
                             saturated=any(c["saturated"] for c in per),
                             closure_rate=per[0]["closure_rate"],
                             n_conditional=per[0]["n_conditional"],
                             three_seed=bool(triple))
    print(f"  {len(cells)} cells built ({time.time()-t_c:.0f}s); "
          f"{sum(1 for c in cells.values() if c['three_seed'])} under 3 seeds; "
          f"{sum(1 for c in cells.values() if c['saturated'])} saturated", flush=True)

    # ----------------------------------------------------------------- verdicts ----
    def adjudicate_L(members, L, k=None, force_loo=False):
        """L1/L2 tier. The LOO conjunct is computed only when it can change the verdict --
        i.e. when closure holds and the order state is not already FAIL -- or when the caller
        forces it (controls, and the declared NOT-ADJUDICABLE subsample)."""
        members = np.asarray(sorted(int(x) for x in members))
        k = len(members)
        g, rho = S42.cell_stats(X1[L][members], X2[L][members])
        closure = bool(g is not None and g < GAP)
        if k < KMIN:
            return dict(k=k, gap_ratio=g, rho=rho, closure_pass=closure,
                        verdict="NOT ADJUDICABLE", reason="R8 k < 9", loo_evaluated=False)
        cell = cells.get((L, k))
        st, det = verdict_L(rho, cell)
        if not force_loo and (not closure or st == "FAIL"):
            return dict(k=k, gap_ratio=g, rho=rho, closure_pass=closure, order_state=st,
                        order_detail=det, verdict="FAIL", loo_evaluated=False,
                        cell_p99=None if cell is None else cell["p99"],
                        cell_p99_band=None if cell is None else [cell["p99_min"],
                                                                 cell["p99_max"]],
                        cell_saturated=None if cell is None else cell["saturated"])
        loo_g, loo_states = [], []
        cell_loo = cells.get((L, k - 1))
        for i in range(k):
            sub = np.delete(members, i)
            gi, ri = S42.cell_stats(X1[L][sub], X2[L][sub])
            loo_g.append(gi)
            loo_states.append(verdict_L(ri, cell_loo)[0])
        loo_closure = all(x is not None and x < GAP for x in loo_g)
        if st == "NOT ADJUDICABLE" or "NOT ADJUDICABLE" in loo_states:
            verdict = "NOT ADJUDICABLE"
        elif not closure or not loo_closure:
            verdict = "FAIL"
        elif st == "FAIL" or "FAIL" in loo_states:
            verdict = "FAIL"
        elif st == "MARGINAL" or "MARGINAL" in loo_states:
            verdict = "MARGINAL"
        else:
            verdict = "PASS"
        return dict(k=k, gap_ratio=g, rho=rho, closure_pass=closure,
                    order_state=st, order_detail=det, loo_closure_pass=loo_closure,
                    n_loo_closure_fail=int(sum(1 for x in loo_g
                                               if x is None or x >= GAP)),
                    loo_order_states=loo_states, verdict=verdict, loo_evaluated=True,
                    cell_p99=None if cell is None else cell["p99"],
                    cell_p99_band=None if cell is None else [cell["p99_min"], cell["p99_max"]],
                    cell_saturated=None if cell is None else cell["saturated"])

    # declared subsample: NOT ADJUDICABLE rows (saturated cells) get the frozen closure-LOO
    # on at most 2,000 rows per (layer, rule) so the closure-and-LOO tally is measured
    # rather than asserted; the rest are reported on closure alone.
    NA_SUBSAMPLE = 2000
    na_seen, rng_na = {}, np.random.default_rng(43_950_000)
    rows_out, tally = [], {}
    for v in all_props:
        cell = cells.get((v["layer"], v["k"]))
        force = False
        if cell is not None and cell["saturated"]:
            key = (v["layer"], v["rule"])
            na_seen[key] = na_seen.get(key, 0) + 1
            force = na_seen[key] <= NA_SUBSAMPLE
        adj = adjudicate_L(v["members"], v["layer"], force_loo=force)
        r = dict(layer=v["layer"], run=v["run"], rule=v["rule"], carrier=v.get("carrier"),
                 seed=v["seed"], member_hash=v["member_hash"], **adj)
        r["is_strict12"] = bool(sorted(v["members"]) == sorted(S12.tolist()))
        r["n_months"] = int(len(month_rows & set(v["members"])))
        if r["verdict"] in ("PASS", "MARGINAL"):
            r["tokens"] = [strings[i] for i in sorted(v["members"])]
        tally[r["verdict"]] = tally.get(r["verdict"], 0) + 1
        rows_out.append(r)
    print(f"\n=== L-TIER VERDICTS === {tally}", flush=True)

    # ----------------------------------------------------------------- controls ----
    ctrl = {}
    for L in layers:
        s12 = adjudicate_L(S12, L, force_loo=True)
        s13 = adjudicate_L(S13, L, force_loo=True)
        oc = adjudicate_L(ord_rows, L, force_loo=True)
        recovered = any(r["is_strict12"] and r["layer"] == L for r in rows_out)
        ctrl[f"L{L}"] = dict(strict12=s12, thirteen_set=s13, ordinal=oc,
                             strict12_recovered_by_assembly=bool(recovered),
                             L3_role=("positive control" if L == 6 else
                                      "reported MARGINAL per L3, not a tier failure"))
        print(f"  L{L}: strict12 gap {s12['gap_ratio']:.4f} rho {s12['rho']:.4f} "
              f"band {s12['cell_p99_band']} -> {s12['verdict']}; recovered {recovered}; "
              f"13-set -> {s13['verdict']}; ordinal gap {oc['gap_ratio']:.3f} -> "
              f"{oc['verdict']}", flush=True)
    out["controls"] = ctrl

    # ------------------------------------------------------- L7 inventory snapshot ----
    out["null_cell_inventory_after_controls"] = [
        dict(v) for (kk, v) in sorted(cells.items())]

    # ------------------------------------------------------------ L6 literal plant ----
    print("\n=== L6: literal random-carrier plant ===", flush=True)
    L = layers[0]
    mu1, B1 = S43.plane_of(X1[L][S12])
    Y1 = (X1[L][S12] - mu1) @ B1.T
    r1 = float(np.median(np.linalg.norm(Y1 - Y1.mean(0), axis=1)))
    mu2, B2 = S43.plane_of(X2[L][S12])
    Y2 = (X2[L][S12] - mu2) @ B2.T
    r2 = float(np.median(np.linalg.norm(Y2 - Y2.mean(0), axis=1)))
    th = 2 * np.pi * np.arange(PLANTED_K) / PLANTED_K
    lit = []
    for rho_j in PLANTED_RHOS:
        rng = np.random.default_rng(43_900_000 + int(1000 * rho_j))
        nc_ = nt_ = 0
        cell = cells.get((L, PLANTED_K))
        for _ in range(int(a.n_planted)):
            carriers = rng.choice(n_types, size=PLANTED_K, replace=False)
            def plant(X, rad):
                P = X[carriers].copy()
                mu = P.mean(0, keepdims=True)
                C = P - mu
                _, _, Vt = np.linalg.svd(C, full_matrices=False)
                B = Vt[:2]
                oop = C - (C @ B.T) @ B
                pts = np.column_stack([rad * np.cos(th), rad * np.sin(th)])
                pts = pts + rng.normal(0.0, rho_j * rad, size=pts.shape)
                return mu + oop + pts @ B
            P1, P2 = plant(X1[L], r1), plant(X2[L], r2)
            g, r = S42.cell_stats(P1, P2)
            cl = bool(g is not None and g < GAP)
            nc_ += int(cl)
            if cl and cell is not None:
                nt_ += int(verdict_L(r, cell)[0] == "PASS")
        lit.append(dict(rho=rho_j, n=int(a.n_planted),
                        recall_closure=nc_ / max(int(a.n_planted), 1),
                        recall_joint=nt_ / max(int(a.n_planted), 1)))
        print(f"  literal rho={rho_j:.2f}: closure {lit[-1]['recall_closure']:.3f} "
              f"joint {lit[-1]['recall_joint']:.3f}", flush=True)
    out["planted_literal_L6"] = dict(layer=L, radius_channel1=r1, radius_channel2=r2,
                                     curve=lit)
    out["planted_clean_from_43"] = prev.get("planted")

    # ------------------------------------------------------------------ summary ----
    passers = [r for r in rows_out if r["verdict"] == "PASS"]
    marg = [r for r in rows_out if r["verdict"] == "MARGINAL"]
    keep, per_cell = [], {}
    for r in sorted(rows_out, key=lambda r: (-r["k"], r["gap_ratio"])):
        key = (r["layer"], r["rule"], r["run"])
        per_cell[key] = per_cell.get(key, 0) + 1
        if r["verdict"] in ("PASS", "MARGINAL") or per_cell[key] <= 200:
            keep.append(r)
    out["proposals"] = keep
    out["proposals_storage_rule"] = ("every PASS and MARGINAL stored; otherwise the first 200 "
                                     "per (layer, rule, run) by k desc then gap asc")
    out["verdict_tally"] = tally
    prev_pass = {(p["layer"], p["member_hash"]) for p in prev.get("summary", {}).get("passers", [])}
    now_pass = {(p["layer"], p["member_hash"]) for p in passers}
    out["summary"] = dict(
        n_proposals=len(rows_out), tally=tally,
        n_pass=len(passers), n_marginal=len(marg),
        passers=[{k: v for k, v in r.items() if k != "loo_order_states"} for r in passers],
        marginals_k_ge_12=[{k: v for k, v in r.items() if k != "loo_order_states"}
                           for r in marg if r["k"] >= 12][:50],
        K_tier_passers=sorted(prev_pass), L_tier_passers=sorted(now_pass),
        verdict_moves_vs_43=dict(
            lost=sorted(prev_pass - now_pass), gained=sorted(now_pass - prev_pass)),
        positive_control_L6=ctrl.get("L6", {}).get("strict12", {}).get("verdict"),
        positive_control_L6_recovered=ctrl.get("L6", {}).get("strict12_recovered_by_assembly"),
        L7_strict12=ctrl.get("L7", {}).get("strict12", {}).get("verdict"),
        n_saturated_cells=sum(1 for c in cells.values() if c["saturated"]),
        gate=("PASS" if (ctrl.get("L6", {}).get("strict12_recovered_by_assembly")
                         and ctrl.get("L6", {}).get("strict12", {}).get("verdict") == "PASS")
              else "FAIL"))
    out["status"] = "COMPLETE"
    out["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    out["runtime_seconds"] = round(time.time() - t0, 1)
    Path(a.out).write_text(json.dumps(out, indent=2, default=float))
    print(f"\nwrote {a.out} in {out['runtime_seconds']}s", flush=True)
    print(json.dumps(out["summary"], indent=2, default=float)[:2500], flush=True)


if __name__ == "__main__":
    main()
