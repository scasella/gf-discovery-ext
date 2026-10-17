"""48c -- the R13 permuted-search null FOR PASS_P2 (Addendum P).

48 stored only passers[:10] (empty) from its permuted searches, not the permuted proposal
memberships, so the P1 guard and the P2 adjudicator cannot be applied to stored permuted
rows. They CAN be applied to regenerated ones: 48's permuted searches are deterministic
functions of their seeds (rng = default_rng(48_000_000 + 1000*L + 10*p + {0 for A, 5 for B}))
over the frozen channels, exactly as its real screens are.

This script re-runs those permuted searches with the identical seeding and, on each permuted
population, applies the SAME pipeline 48b applied to the real population: P1 guard, 48's
tier, then P2 (channel 3 vs the 99th percentile of 500 random-k planes, then the
frequency-matched arm). The reported quantity is the mean number of PASS_P2 per permutation
-- the verdict-bearing null for PASS_P2, which 48b could only report analytically
(0.01 x n_adjudicated).

Conventions carried from 48 unchanged:
  - thresholds are NOT recomputed on the permuted channel (42's thr_override convention):
    the cluster-like cells are the REAL ones, and the per-k tie floors are the REAL GUARDED
    screen's (results/phase5_readjudication_P.json), because the question is whether a
    permuted channel produces a passer at the REAL tier's threshold;
  - channel 1 is untouched by the permutation, so channel 3 (a channel-1 + weights quantity)
    is computed exactly as in 48b;
  - the frequency-matched arm and rho use the PERMUTED channel 2, as in 48's permuted search.
R17 (CPU only) is in force.
"""
import argparse, importlib.util, json, os, sys, time
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HOME", str(REPO / "cache/hf"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
sys.path.insert(0, str(REPO / "src"))

READJ = "results/phase5_readjudication_P.json"


def _load(name, rel):
    sp = importlib.util.spec_from_file_location(name, str(REPO / rel))
    m = importlib.util.module_from_spec(sp)
    sp.loader.exec_module(m)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/phase5_permuted_P2_null.json")
    ap.add_argument("--layers", default="6,7")
    ap.add_argument("--perms-a", type=int, default=20)
    ap.add_argument("--perms-b", type=int, default=10)
    args = ap.parse_args()
    layers = [int(x) for x in args.layers.split(",")]

    out = REPO / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    doc = dict(status="RUNNING -- declaration stub written before any artifact was opened",
               script="scripts/48c_permuted_P2_null.py",
               declared_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
               config=dict(layers=layers, perms_A=args.perms_a, perms_B=args.perms_b),
               declared_rules=dict(
                   question=("the verdict-bearing null for PASS_P2: how many PASS_P2 rows "
                             "does the whole P1+P2 pipeline produce per within-(token class "
                             "x log10-count decile) permutation of the channel-2 rows?"),
                   why_regenerated=("48 stored passers[:10] (empty) from its permuted "
                                    "searches, not permuted proposal memberships, so the "
                                    "guard and P2 cannot be applied to stored rows. The "
                                    "permuted searches are deterministic functions of their "
                                    "seeds and are reproduced here with 48's exact seeding: "
                                    "48_000_000 + 1000*L + 10*p + (0 for A, 5 for B)."),
                   thresholds=("NOT recomputed on the permuted channel (42/48's "
                               "thr_override convention). Cluster-like cells are the REAL "
                               "ones; per-k tie floors are the REAL GUARDED screen's, read "
                               f"from {READJ}. Channel 3 uses channel 1 and the weights, "
                               "both untouched by the permutation. rho and the "
                               "frequency-matched arm use the PERMUTED channel 2."),
                   pipeline=("identical to 48b: P1 duplicate-referent guard on every "
                             "proposal, 48's tier (closure first, order, LOO), then for "
                             "NOT ADJUDICABLE rows with k <= 10, rho == 1.0, LOO closure "
                             "and a saturated cell, the P2 adjudicator (channel 3 > the "
                             "99th percentile of 500 random-k planes, then the "
                             "frequency-matched arm)."),
                   deviations=[
                       "F1 (proposer A). A produced 0 proposals in all 40 of 48's permuted "
                       "runs, so its PASS_P2 null is 0 by construction; it is re-run anyway "
                       "and the proposal counts are reported.",
                       "F2 (budget). perms_A / perms_B as run are in config; 48 ran 20 / 10 "
                       "(its D2 shortfall against R13's 20 for B carries over unchanged).",
                       "F3 (nulls unguarded). As in 48b (E2), the cluster-like cells, the "
                       "frequency-arm draws and the channel-3 random-k planes are not "
                       "guarded. The comparison between the real and permuted pipelines is "
                       "therefore like-for-like.",
                   ]))
    out.write_text(json.dumps(doc, indent=1))

    logf = open(REPO / "logs/48c_permP2.log", "a")

    def log(msg):
        print(msg, flush=True)
        logf.write(msg + "\n"); logf.flush()

    log(f"=== 48c start {doc['declared_at']} cfg={doc['config']}")
    res = dict(runs={}, summary={})

    def save(status="RUNNING"):
        out.write_text(json.dumps({**doc, **res, "status": status}, indent=1, default=float))

    b48 = _load("b48mod", "scripts/48b_readjudicate_P.py")
    s48 = _load("s48mod", "scripts/48_agreement_graph_screen.py")
    S42 = _load("s43mod", "scripts/43_neighbour_assembly_screen.py").S42
    readj = json.loads((REPO / READJ).read_text())
    log("tier + guard imported (48b over 48 over 42); real guarded tie floors loaded")

    d1 = np.load(REPO / s48.FULLN_RAW)
    d2 = np.load(REPO / s48.ROBUST_DELTAS)
    pool = d1["pool_ids"].astype(int)
    counts = d1["counts"].astype(np.int64)
    n_types = len(pool)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("google/gemma-3-1b-pt")
    strings = [tok.decode([int(i)]) for i in pool]

    C1, C2, Z1, Z2, keepc = {}, {}, {}, {}, {}
    for L in layers:
        keep = d2[f"L{L}_lag1_kept_coords"]
        keepc[L] = np.asarray(keep)
        C1[L] = d1[f"R_L{L}"][:, keep].astype(np.float64)
        C2[L] = d2[f"L{L}_lag1"].astype(np.float32).astype(np.float64)
        Z1[L] = s48.centred_unit(C1[L]).astype(np.float32)
        Z2[L] = s48.centred_unit(C2[L]).astype(np.float32)
    c3 = s48.Channel3(layers, keepc)
    cellsL = {L: s48.Cells(S42, C1[L], C2[L], Z1[L], Z2[L], L) for L in layers}
    dup, _ = b48.build_dupmap(strings)
    key = {i: (int(counts[i]), -i) for i in range(n_types)}
    blocks = s48.permuted_blocks(strings, counts)
    log(f"channels, cells, channel 3, guard map ({len(dup)} types) and "
        f"{len(blocks)} permutation blocks ready")

    ch3_nulls = {}

    def ch3_null(L, k):
        kk = f"L{L}_k{k}"
        if kk not in ch3_nulls:
            ch3_nulls[kk] = s48.ov_null(c3, L, C1[L], k)
        return ch3_nulls[kk]

    for L in layers:
        for name, n_perm, off in (("A", args.perms_a, 0), ("B", args.perms_b, 5)):
            tie_real = {int(k): v for k, v in
                        readj["source48"][f"L{L}_{name}"][
                            "guarded_screen_max_rho_by_k"].items()}
            runs = []
            for p in range(n_perm):
                t0 = time.time()
                rng = np.random.default_rng(48_000_000 + 1000 * L + 10 * p + off)
                perm = np.arange(n_types)
                for b in blocks:
                    perm[b] = b[rng.permutation(len(b))]
                Z2p, C2p = Z2[L][perm], C2[L][perm]
                if name == "A":
                    props, _st = s48.screen_A(S42, C1[L], C2p, Z1[L], Z2p, Z1[L], counts,
                                              cellsL[L], L)
                else:
                    props, _st = s48.screen_B(S42, C1[L], C2p, Z2p, Z1[L], counts, L)
                n_raw = len(props)
                gprops = {}
                for h, pr in props.items():
                    g, _dr = b48.guard(pr["members"], dup, key)
                    if len(g) < 2:
                        continue
                    gh = s48.mhash(g)
                    if gh not in gprops:
                        gprops[gh] = dict(members=g, k=len(g))
                del props
                n_pass = n_p2 = n_pass_p2 = 0
                p2_rows, pass_rows = [], []
                if gprops:
                    ga = _adjudicate(s48, S42, gprops, C1[L], C2p, cellsL[L], tie_real)
                    for gh, r in ga.items():
                        mem = gprops[gh]["members"]
                        if r["verdict"] == "PASS":
                            fa = s48.frequency_arm(S42, np.asarray(mem), C1[L], C2p, counts,
                                                   r["two_channel_rho"])
                            ret, _ph = c3.retention(L, C1[L][np.asarray(mem)])
                            nl = ch3_null(L, len(mem))
                            ok = fa["pass_"] and (ret > nl["p99"] if len(mem) <= 10 else True)
                            if ok:
                                n_pass += 1
                                pass_rows.append(dict(k=len(mem), rho=r["two_channel_rho"],
                                                      gap=r["gap_ratio"],
                                                      tokens=[strings[i] for i in mem]))
                            continue
                        ok, why, sat, ex1 = b48.p2_eligible(r, mem, cellsL[L], s48, S42,
                                                            C1[L])
                        if not ok:
                            continue
                        n_p2 += 1
                        ret, _ph = c3.retention(L, C1[L][np.asarray(mem)])
                        nl = ch3_null(L, len(mem))
                        if ret <= nl["p99"]:
                            continue
                        fa = s48.frequency_arm(S42, np.asarray(mem), C1[L], C2p, counts,
                                               r["two_channel_rho"])
                        if fa["pass_"]:
                            n_pass_p2 += 1
                            p2_rows.append(dict(k=len(mem), rho=r["two_channel_rho"],
                                                gap=r["gap_ratio"], ch3=ret,
                                                ch3_p99=nl["p99"], exact1=ex1,
                                                tokens=[strings[i] for i in mem]))
                runs.append(dict(perm=p, n_proposals_raw=n_raw, n_guarded_unique=len(gprops),
                                 n_R14_PASS=n_pass, n_P2_adjudicated=n_p2,
                                 n_PASS_P2=n_pass_p2, PASS_P2_rows=p2_rows,
                                 R14_PASS_rows=pass_rows,
                                 seconds=round(time.time() - t0, 1)))
                log(f"  [permP2 L{L}/{name}] run {p}: {n_raw} raw -> {len(gprops)} guarded, "
                    f"{n_pass} R14 PASS, {n_p2} P2-adjudicated, {n_pass_p2} PASS_P2 "
                    f"({time.time()-t0:.0f} s)")
                res["runs"][f"L{L}_{name}"] = dict(
                    n_permutations=len(runs),
                    mean_PASS_P2=float(np.mean([r["n_PASS_P2"] for r in runs])),
                    mean_P2_adjudicated=float(np.mean([r["n_P2_adjudicated"] for r in runs])),
                    mean_R14_PASS=float(np.mean([r["n_R14_PASS"] for r in runs])),
                    PASS_P2_counts=[r["n_PASS_P2"] for r in runs],
                    P2_adjudicated_counts=[r["n_P2_adjudicated"] for r in runs],
                    R14_PASS_counts=[r["n_R14_PASS"] for r in runs],
                    proposal_counts=[r["n_proposals_raw"] for r in runs], runs=runs)
                save()
            log(f"  [permP2 L{L}/{name}] mean PASS_P2 = "
                f"{res['runs'][f'L{L}_{name}']['mean_PASS_P2']}")

    real = {k: dict(n_PASS_P2=v["n_PASS_P2"], n_P2_adjudicated=v["n_P2_adjudicated"],
                    n_R14_PASS=v["n_R14_PASS"])
            for k, v in readj["source48"].items()}
    res["summary"] = dict(
        real_pipeline=real,
        permuted={k: dict(n_permutations=v["n_permutations"],
                          mean_PASS_P2=v["mean_PASS_P2"],
                          mean_P2_adjudicated=v["mean_P2_adjudicated"],
                          mean_R14_PASS=v["mean_R14_PASS"],
                          PASS_P2_counts=v["PASS_P2_counts"])
                  for k, v in res["runs"].items()},
        channel3_nulls=ch3_nulls,
        reading=("R14/R13 asks the permuted-search mean to be < 0.5. The same bar is "
                 "applied here to PASS_P2. Compare mean_PASS_P2 per permutation with the "
                 "real pipeline's n_PASS_P2 at the same (layer, proposer)."),
        deviations=doc["declared_rules"]["deviations"])
    doc["status"] = "COMPLETE"
    doc["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    doc["runtime_seconds"] = round(time.time() - t_start, 1)
    save("COMPLETE")
    log(f"=== 48c complete in {doc['runtime_seconds']} s -> {out}")
    logf.close()


def _adjudicate(s48, S42, props, C1, C2, cells, tie_real):
    """48's gate + adjudicate over a guarded permuted population, at the REAL thresholds."""
    GAP, KMIN = s48.GAP, s48.KMIN
    recs = {}
    for h, p in props.items():
        mem = np.asarray(p["members"])
        if len(mem) < KMIN:
            continue
        g = S42.gap_ratio_of(C1[mem])
        if g is None or g >= GAP:
            continue
        _, r = S42.cell_stats(C1[mem], C2[mem])
        if r is None:
            continue
        k = len(mem)
        cell = s48.cell_view(cells.get(k), "strict")
        lo = cell["p99_min_strict"]
        tf = tie_real.get(k)
        if not ((tf is not None and r >= tf - 1e-12) or (lo is not None and r >= lo)
                or cell["saturated"]):
            continue
        rec = s48.adjudicate(S42, mem, C1, C2, cells, tf)
        rec["member_hash"] = h
        recs[h] = rec
    return recs


if __name__ == "__main__":
    main()
