"""43d -- permuted-SEARCH null for the 43 assembly screen (Addendum N, N2/N4).

Addenda in force: K1-K8 (via 43), L1-L7 (via 43b), M1-M3 (via 43c), N1-N7 (this file).
42, 43, 43b and 43c are IMPORTED by path; none of them is edited or re-run.

Why this file exists. 43c permuted channel 2 at ADJUDICATION time only, on a fixed proposal
set, on the premise that the assembly reads channel 1 alone. verify43 showed that premise is
false (N1): the K3 duplicate guard fits a channel-2 plane at every extension step, so the
proposal set is itself a function of channel 2 (real-vs-permuted proposal Jaccard 0.013-0.028).
43c is therefore a member-specificity test of the order agreement, not a multiplicity null.

43d is the multiplicity null. The channel-2 rows are permuted within lexical class BEFORE the
assembly runs, so the K3 guard, the proposal set, and the tier all see one and the same
permuted world. Everything else -- seeds, candidate pools, A1/A2 rules, 43b's cached cells,
closure-first L1/L2 -- is identical to the real screen. The mean number of passers over the
permutations is the full-multiplicity false-pass count against which the real 7 are read.

Scope, per N2: run (a) only (full-pool L6/L7, A1 and A2), the same 28,000 / 28,082 unordered
seed pairs. Run (b) (carrier-seeded) is not repeated: its carriers were selected by 44b on
channel-1 geometry and contribute 1,546 of 358,693 proposals.
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


S43C = _load("s43c", "scripts/43c_full_multiplicity_null.py")
S43B, S43, S42 = S43C.S43B, S43C.S43, S43C.S42
GAP, KMIN, R10, D_MODEL = S42.GAP_RATIO_CLOSURE, S42.K_MIN_ADJUDICABLE, S42.R10_DROP, S42.D_MODEL
EXACT1_MAX = S43C.EXACT1_MAX
N_PERM = 3
SOURCE_43 = "results/phase4_assembly_screen.json"
SOURCE_43B = "results/phase4_assembly_screen_addendumL.json"


def declared_rules(cfg, seven, s12h):
    return dict(
        addenda_in_force=["K1-K8 (via 43)", "L1-L7 (via 43b)", "M1-M3 (via 43c)",
                          "N1-N7 (docs/phase4_addendum_N_2026-08-22.md, this file)"],
        N1_correction=("the K3 duplicate guard fits a channel-2 plane at every assembly step, "
                       "so the proposal set is NOT invariant to a channel-2 permutation. 43c "
                       "is a member-specificity test; THIS file is the multiplicity null."),
        design=("for each permutation: (1) permute the channel-2 rows within the lexical "
                "blocks of 42/43 (leading_space x capitalized), seeded; (2) re-run the FULL "
                "assembly A1 and A2 at L6 and L7 on the identical seed set with that permuted "
                "channel 2 supplied to scripts/43's assemble(), so the K3 guard sees it; "
                "(3) adjudicate every k >= 9 proposal with the L-tier against the SAME "
                "permuted channel 2. One coherent permuted world per permutation."),
        scope="run (a) only (full pool, L6/L7, A1/A2); run (b)'s 1,546 carrier proposals are "
              "not repeated because 44b selected those carriers on channel-1 geometry",
        cells="43b's cached (layer, k) cells, unchanged and NOT recomputed per permutation",
        closure_first=("closure is evaluated first: gap_ratio >= 2.5 -> FAIL regardless of "
                       "cell saturation (43c's ordering, which returns the ordinal control to "
                       "FAIL). gap_ratio for the tier is 42's fit_circle convention, "
                       "recomputed; the assembly's internal centroid-plane gap is not reused."),
        exact_shortcut=("a candidate in a SATURATED cell cannot PASS under L1 (every saturated "
                        f"cell has conditional exact-1.0 rate > {EXACT1_MAX}); channel-2 work "
                        "is skipped for those. Exact for PASS counts, as in 43c."),
        loo="43b/43c short-circuit: the LOO is evaluated only where the k-cell order state is "
            "PASS or MARGINAL; exact for PASS counts",
        N2_reading=("mean passer count over the permutations is the full-multiplicity "
                    "false-pass number: >= 3 -> the real 7 are search noise, NO DISCOVERY; "
                    "< 1 together with the 7 not surviving 43c -> open, proceed to M3 and 47."),
        N4_frequency_floor=("member min corpus count reported for each of the 7 and for "
                            "strict-12; members are recovered from 43b's stored passer token "
                            "lists and the recovery is asserted against the stored "
                            "member_hash."),
        incremental_write=("the results file is rewritten after every permutation, so a "
                           "truncated run still reports the permutations that finished"),
        reporting="the 7 are named by member_hash and class description only; no token strings",
        targets_seven=seven, strict12_hash=s12h,
        no_mps="CPU only; no forward passes; the SAE cache is not touched",
        config=cfg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/phase4_permuted_search_null.json")
    ap.add_argument("--layers", default="6,7")
    ap.add_argument("--n-perm", type=int, default=N_PERM)
    ap.add_argument("--max-seeds", type=int, default=0)
    a = ap.parse_args()
    t0 = time.time()
    layers = [int(x) for x in a.layers.split(",")]

    prev43 = json.loads((REPO / SOURCE_43).read_text())
    prevL = json.loads((REPO / SOURCE_43B).read_text())
    assert prev43["status"] == "COMPLETE" and prevL["status"] == "COMPLETE"
    cells = {(int(c["layer"]), int(c["k"])): c
             for c in prevL["null_cell_inventory_after_controls"]}
    for key, v in cells.items():
        if v["saturated"]:
            assert v["exact1_rate"] >= EXACT1_MAX, f"saturated cell {key} breaks the shortcut"
    seen, targets = set(), []
    for p in prevL["summary"]["passers"]:
        k = (int(p["layer"]), p["member_hash"])
        if k in seen:
            continue
        seen.add(k)
        targets.append(p)
    seven = [t["member_hash"] for t in targets if not t["is_strict12"]]
    s12h = next(t["member_hash"] for t in targets if t["is_strict12"])
    assert len(seven) == 7

    declared_at = time.strftime("%Y-%m-%dT%H:%M:%S")
    cfg = dict(layers=layers, n_perm=int(a.n_perm), max_seeds=int(a.max_seeds),
               source_43=SOURCE_43, source_43b=SOURCE_43B)
    stub = dict(status="DECLARATION STUB -- rules frozen, no results yet",
                script="scripts/43d_permuted_search_null.py", declared_at=declared_at,
                stub_note=("written before any COORDINATE read; the two source results JSONs "
                           "are read first only to name the targets and load 43b's cells"),
                real_screen=dict(n_proposals_43=prev43["summary"]["n_proposals_total"],
                                 tally_43b=prevL["verdict_tally"],
                                 n_unique_passers=len(targets)),
                declared_rules=declared_rules(cfg, seven, s12h))
    Path(a.out).write_text(json.dumps(stub, indent=2))
    print("=== DECLARATION STUB WRITTEN (before any coordinate read) ===", flush=True)
    out = dict(stub)
    out["status"] = "RUNNING"

    # ---------------------------------------------------------------- data ----
    zr = np.load(REPO / S43.RAW_NPZ)
    pool = zr["pool_ids"].astype(np.int64)
    counts = zr["counts"].astype(np.int64)
    n_types = len(pool)
    zd = np.load(REPO / S43.DELTAS_NPZ)
    X1, X2, NNt, seeds_of = {}, {}, {}, {}
    for L in layers:
        keep = np.ones(D_MODEL, dtype=bool)
        keep[R10[L]] = False
        X1[L] = np.ascontiguousarray(zr[f"R_L{L}"].astype(np.float32)[:, keep].astype(np.float64))
        X2[L] = np.ascontiguousarray(zd[f"L{L}_lag1"].astype(np.float32).astype(np.float64))
        Xc = X1[L] - X1[L].mean(0, keepdims=True)
        Xn = (Xc / np.maximum(np.linalg.norm(Xc, axis=1, keepdims=True), 1e-12)).astype(np.float32)
        Sm = Xn @ Xn.T
        np.fill_diagonal(Sm, -2.0)
        NNt[L] = np.argsort(-Sm, axis=1)[:, :S43.N_CAND_NN].astype(np.int64)
        s = S43.build_seeds(NNt[L])
        seeds_of[L] = s[:a.max_seeds] if a.max_seeds else s
        del Sm, Xc, Xn
        gc.collect()
    del zr, zd
    gc.collect()

    from transformers import AutoTokenizer
    from gfdisc.env import MODEL_NAME
    tk = AutoTokenizer.from_pretrained(MODEL_NAME, token=os.environ.get("HF_TOKEN"))
    strings = [(tk.convert_ids_to_tokens(int(t)) or tk.decode([int(t)])).replace("▁", " ")
               for t in pool]
    blocks = S42.resolve_blocks(strings)
    smap = {}
    for i, s in enumerate(strings):
        smap.setdefault(s, []).append(i)
    print(f"pool {n_types}; blocks {[len(b) for b in blocks]}; seeds "
          f"{ {L: len(seeds_of[L]) for L in layers} }", flush=True)

    # ------------------------------------------------------- N4 frequency floor ----
    floor = []
    for t in targets:
        rows = [smap[s][0] for s in t["tokens"] if len(smap.get(s, [])) == 1]
        ok = len(rows) == len(t["tokens"]) and S43.member_hash(rows) == t["member_hash"]
        assert ok, f"member recovery failed for {t['member_hash']}"
        c = counts[rows]
        floor.append(dict(member_hash=t["member_hash"], layer=int(t["layer"]), k=int(t["k"]),
                          is_strict12=bool(t["is_strict12"]),
                          min_count=int(c.min()), median_count=int(np.median(c)),
                          max_count=int(c.max()),
                          n_members_at_pool_floor=int((c < 120).sum())))
        print(f"  N4 {t['member_hash']} L{t['layer']} k={t['k']}: min count {int(c.min())}, "
              f"median {int(np.median(c))}, max {int(c.max())}"
              + ("  [strict-12]" if t["is_strict12"] else ""), flush=True)
    out["N4_frequency_floor"] = floor
    out["N4_note"] = ("the pool's count floor is 100 (scripts/22 COUNT_FLOOR); a member set "
                      "whose minimum member count is at 100-111 is drawn from the noisiest "
                      "type means in the pool")
    Path(a.out).write_text(json.dumps(out, indent=2, default=float))

    # ------------------------------------------------------------- permutations ----
    def adjudicate(members, L, D):
        g = S42.gap_ratio_of(X1[L][members])
        if g is None or g >= GAP:
            return "FAIL", g, None
        cell = cells.get((L, len(members)))
        if cell is None or cell.get("p99") is None or cell["saturated"]:
            return "NOT ADJUDICABLE", g, None
        P2 = D[members]
        if len(P2) < 3 or S42.degenerate(P2):
            return "NOT ADJUDICABLE", g, None
        a1, _ = S42.plane_angles(X1[L][members])
        a2, _ = S42.plane_angles(P2)
        rho = S43C.rho_from_ranks(S42._rank_from_angles(a1, 0), S42._rank_from_angles(a2, 0))
        st, _ = S43C.order_state(rho, cell)
        if st in ("FAIL", "NOT ADJUDICABLE"):
            return st, g, rho
        cl = cells.get((L, len(members) - 1))
        if cl is None or cl.get("p99") is None:
            return "NOT ADJUDICABLE", g, rho
        states = []
        for j in range(len(members)):
            sub = np.delete(members, j)
            gg, rr = S42.cell_stats(X1[L][sub], D[sub])
            if not (gg is not None and gg < GAP):
                return "FAIL", g, rho
            states.append(S43C.order_state(rr, cl)[0])
        if "NOT ADJUDICABLE" in states:
            return "NOT ADJUDICABLE", g, rho
        if "FAIL" in states:
            return "FAIL", g, rho
        if "MARGINAL" in states or st == "MARGINAL":
            return "MARGINAL", g, rho
        return "PASS", g, rho

    perms = []
    for p_i in range(int(a.n_perm)):
        tp = time.time()
        rng = np.random.default_rng(43_600_000 + p_i)
        Dp = {}
        for L in layers:
            perm = np.arange(n_types)
            for b in blocks:
                if len(b) > 1:
                    perm[b] = b[rng.permutation(len(b))]
            Dp[L] = X2[L][perm]
        print(f"\n=== permutation {p_i} (seed {43_600_000 + p_i}) ===", flush=True)
        props, counts_by_cell = [], {}
        for L in layers:
            for rule in ("A1", "A2"):
                p, st = S43.run_assembly(X1[L], Dp[L], NNt[L], seeds_of[L], rule,
                                         tag=f"p{p_i} L{L} ")
                counts_by_cell[f"L{L}_{rule}"] = dict(
                    n_unique_k_ge_9=st["n_unique_k_ge_9"],
                    n_seeds_with_proposal=st["n_seeds_with_proposal"],
                    real_n_unique_k_ge_9=(prevL.get("reconstruction", {})
                                          .get(f"L{L}_{rule}", {}).get("stored_by_43")),
                    seconds=st["seconds"])
                for v in p.values():
                    props.append((L, np.asarray(v["members"])))
                print(f"  [p{p_i}] L{L} {rule}: {st['n_unique_k_ge_9']} unique k>=9 "
                      f"(real screen {counts_by_cell[f'L{L}_{rule}']['real_n_unique_k_ge_9']}) "
                      f"{st['seconds']}s", flush=True)
        tally, passers = {}, []
        t_adj = time.time()
        for n, (L, m) in enumerate(props):
            stt, g, rho = adjudicate(m, L, Dp[L])
            tally[stt] = tally.get(stt, 0) + 1
            if stt == "PASS":
                h = S43.member_hash(m)
                passers.append(dict(layer=L, k=int(len(m)), member_hash=h,
                                    gap_ratio=float(g), rho=float(rho),
                                    min_member_count=int(counts[m].min())))
            if (n + 1) % 50_000 == 0:
                print(f"    [p{p_i}] adjudicated {n+1}/{len(props)}, {len(passers)} PASS, "
                      f"{time.time()-t_adj:.0f}s", flush=True)
        uniq = sorted({x["member_hash"] for x in passers})
        rec = dict(permutation=p_i, seed=int(43_600_000 + p_i),
                   proposal_counts=counts_by_cell, n_proposals=len(props),
                   tally=tally, n_pass=len(passers), n_unique_pass=len(uniq),
                   unique_hashes=uniq, k_of_passers=sorted({x["k"] for x in passers}),
                   passers=passers[:200],
                   overlap_with_real_seven=sorted(set(uniq) & set(seven)),
                   n_overlap_with_real_seven=len(set(uniq) & set(seven)),
                   strict12_passes=bool(s12h in uniq),
                   seconds=round(time.time() - tp, 1))
        perms.append(rec)
        print(f"  [p{p_i}] proposals {len(props)}; PASS {len(passers)} "
              f"({len(uniq)} unique); k of passers {rec['k_of_passers']}; overlap with the "
              f"real 7 = {rec['n_overlap_with_real_seven']} ({rec['seconds']}s)", flush=True)
        out["permutations"] = perms
        out["status"] = f"RUNNING ({p_i + 1}/{a.n_perm} permutations complete)"
        Path(a.out).write_text(json.dumps(out, indent=2, default=float))
        del Dp
        gc.collect()

    mean_pass = float(np.mean([r["n_pass"] for r in perms]))
    mean_uni = float(np.mean([r["n_unique_pass"] for r in perms]))
    mean_prop = float(np.mean([r["n_proposals"] for r in perms]))
    if mean_uni >= 3:
        reading, text = "N2(noise)", (
            "the permuted search yields passers at a rate comparable to the real 7: the 7 are "
            "search noise at this multiplicity. NO DISCOVERY; the measured full-multiplicity "
            "false-pass rate is the reportable number.")
    elif mean_uni < 1:
        reading, text = "N2(open)", (
            "the permuted search yields ~0 passers: the search itself does not manufacture "
            "passers at this multiplicity. Whether the 7 stand also depends on the "
            "member-specificity outcome recorded in results/phase4_full_multiplicity_null.json "
            "(43c) and on M3's frequency-partial test; the coordinator reconciles the two.")
    else:
        reading, text = "N2(intermediate)", (
            "mean unique permuted passers lies in [1, 3): neither N2 branch obtains cleanly; "
            "the measured rate is reported and the coordinator adjudicates.")
    out["summary"] = dict(
        n_perms=len(perms), mean_proposals=mean_prop,
        real_proposals_run_a=sum(v["stored_by_43"] for v in
                                 prevL.get("reconstruction", {}).values()
                                 if v.get("stored_by_43")),
        pass_counts=[r["n_pass"] for r in perms],
        unique_counts=[r["n_unique_pass"] for r in perms],
        overlap_with_real_seven=[r["n_overlap_with_real_seven"] for r in perms],
        mean_pass=mean_pass, mean_unique_pass=mean_uni,
        N2_reading=reading, N2_text=text,
        N4_min_counts={f["member_hash"]: f["min_count"] for f in floor})
    out["status"] = "COMPLETE"
    out["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    out["runtime_seconds"] = round(time.time() - t0, 1)
    Path(a.out).write_text(json.dumps(out, indent=2, default=float))
    print(f"\nwrote {a.out} in {out['runtime_seconds']}s", flush=True)
    print(json.dumps(out["summary"], indent=2, default=float), flush=True)


if __name__ == "__main__":
    main()
