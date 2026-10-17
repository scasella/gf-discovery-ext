"""e53 -- confirmatory re-test of the 'discovery verbs' near-miss (phase5_addendum_R; Stage 3).

Candidate (i), L7, k = 10: ' find',' showed',' discover',' identified',' identify',' revealed',
' confirmed',' recognized',' realize',' noticed' (member_hash 2eff8657fbd3). Anchor gate:
strict-12 months at L6 (a36a4f4f3e3b).

What Addendum R defines and this script does:
  * FRESH CONTEXTS -- every context (8 preceding tokens + the member token) is drawn from the
    8,000 reserve sequences that follow the 4M-token corpus in the FineWeb stream and were
    never part of any type mean, proposal, tier or earlier behavioural run.
  * MATCHED PROMPT BUDGETS -- every arm (candidate, anchor, null i, null ii, null iii, null iv)
    is scored at the SAME 40 prompts per member and the SAME eps grid. In 47/47b/51 the
    candidate had 40 prompts and the nulls 12.
  * FULL-BUDGET SIBLING NULL -- null (iv) is the whole proposer-matched family: every set the
    candidate's own route (k = 16 assembly -> R15 dedup 0.92 -> P1 guard, channel-2 cloud,
    L7) produces with k = 10 members that closes, minus the candidate (51 found 30). Rebuilt
    here by 51's own code and checked against 51's stored hashes.
  * PRE-REGISTERED -- docs/PREREG_verbs_retest.md; its sha256 is written into the stub before
    any forward and asserted unchanged at the end.

Everything else (plane/tangent geometry, the Runner, Delta, the context sampler) is IMPORTED
from scripts/51_verify_candidates_di.py, which re-implemented 47's rig independently.
"""
import argparse, hashlib, importlib.util, json, os, sys, time
from pathlib import Path

import numpy as np

EXT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXT / "src"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")


def _imp(name, fn):
    spec = importlib.util.spec_from_file_location(name, EXT / "scripts" / fn)
    m = importlib.util.module_from_spec(spec); sys.modules[name] = m
    spec.loader.exec_module(m); return m


S51 = _imp("s51", "51_verify_candidates_di.py")
import gfdisc.model as gm                                          # noqa: E402
gm.MODEL_NAME = str(sorted((EXT / "hf_home/hub/models--google--gemma-3-1b-pt/snapshots").glob("*"))[0])
S51.load_model = gm.load_model
from gfdisc.env import DEVICE                                      # noqa: E402

PREREG = EXT / "docs/PREREG_verbs_retest.md"
N_PROMPTS = 40
EPS_GRID = (0.005, 0.01, 0.02)
EPS_NULL = 0.02
N_RAND = 10
N_NULL_III = dict(anchor=50, candidate=100)
SEED_CTX, SEED_DIRS, SEED_NULL, SEED_SIB = 5300, 5301, 5302, 5303
ROUTE = dict(k=16, thr=0.92)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(EXT / "results/e53_verbs_retest.json"))
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    t0 = time.time()
    prereg_sha = hashlib.sha256(PREREG.read_bytes()).hexdigest()
    res = dict(script="scripts/e53_verbs_retest.py", prereg=str(PREREG), prereg_sha256=prereg_sha,
               declared_at=time.strftime("%Y-%m-%dT%H:%M:%S"), smoke=a.smoke,
               config=dict(n_prompts_all_arms=N_PROMPTS, eps_grid=EPS_GRID, eps_null=EPS_NULL,
                           n_rand=N_RAND, n_null_iii=N_NULL_III, seeds=dict(
                               ctx=SEED_CTX, dirs=SEED_DIRS, null=SEED_NULL, sib=SEED_SIB),
                           contexts="reserve_tokens.i32 (fresh)"))

    def save(status):
        res["status"] = status; res["elapsed_s"] = round(time.time() - t0, 1)
        Path(a.out).write_text(json.dumps(res, indent=1, default=str))
    save("DECLARED")

    def log(m):
        print(f"[{time.time()-t0:7.1f}s] {m}", flush=True)

    z = np.load(EXT / "results/phase3_token_type_mean_resid_fulln.npz")
    pool_ids = z["pool_ids"].astype(np.int64)
    counts = z["counts"].astype(np.int64)
    d2 = np.load(EXT / "results/phase3_lag_action_robust_deltas.npz")
    assert np.array_equal(d2["pool_ids"].astype(np.int64), pool_ids)
    C1, C2, Z1n, Z2n = {}, {}, {}, {}
    for L in (6, 7):
        C1[L] = np.delete(z[f"R_L{L}"].astype(np.float64), S51.R10_DROP, axis=1)
        C2[L] = d2[f"L{L}_lag1"].astype(np.float32).astype(np.float64)
        Z1n[L] = S51.centred_unit(C1[L]).astype(np.float32)
        Z2n[L] = S51.centred_unit(C2[L]).astype(np.float32)
    pos = {int(t): i for i, t in enumerate(pool_ids)}
    cands = {c["name"]: c for c in S51.CANDIDATES}
    anchor, cand = cands["L6_anchor_strict12"], cands["L7_i_k10"]
    for c in (anchor, cand):
        c["member_pos"] = sorted(pos[int(t)] for t in c["token_ids"])
        assert S51.mhash(c["member_pos"]) == c["member_hash"], c["name"]

    model, tok = gm.load_model(token=None)
    strings = [(tok.convert_ids_to_tokens(int(t)) or tok.decode([int(t)])).replace("▁", " ")
               for t in pool_ids]
    res["candidate_tokens"] = [strings[p] for p in cand["member_pos"]]

    # ---- geometry re-check on the rebuilt fulln (closure + two-channel rho) ----
    geo_chk = {}
    for c in (anchor, cand):
        g, rho = S51.cell_stats(C1[c["layer"]][c["member_pos"]], C2[c["layer"]][c["member_pos"]])
        geo_chk[c["name"]] = dict(gap_ratio=g, rho=rho, stored_gap=c["stored_gap"],
                                  stored_rho=c["stored_rho"])
    res["geometry_recheck"] = geo_chk
    log(f"geometry recheck {geo_chk}")

    # ---- sibling family: the candidate's own route, rebuilt with 51's code ----
    dup, _ = S51.build_dupmap(strings)
    key = {i: (int(counts[i]), -i) for i in range(len(pool_ids))}
    nn = S51.assemblies(Z2n[7], ROUTE["k"] + 1)
    seen, closers = {}, []
    for s in range(len(nn)):
        base = sorted(int(x) for x in nn[s, :ROUTE["k"]])
        mem = S51.dedup_r15(base, Z1n[7], counts, ROUTE["thr"])
        mem, _ = S51.p1_guard(mem, dup, key)
        if len(mem) != 10:
            continue
        h = S51.mhash(mem)
        if h in seen:
            continue
        seen[h] = mem
        g, _r = S51.cell_stats(C1[7][np.asarray(mem)], C2[7][np.asarray(mem)])
        if g is not None and g < S51.GAP:
            closers.append((h, mem))
    sib = [(h, m) for h, m in closers if h != cand["member_hash"]]
    stored = json.loads((EXT / "results/phase5_verify_candidates_di.json").read_text())
    stored_h = {x["member_hash"] for x in
                stored["stage3_behavioural"]["L7_i_k10"]["null_iv_sibling_sets"]}
    res["sibling_family"] = dict(n_distinct=len(seen), n_closers=len(closers),
                                 candidate_in_family=bool(cand["member_hash"] in {h for h, _ in closers}),
                                 n_siblings=len(sib), n_stored=len(stored_h),
                                 n_overlap_with_stored=len({h for h, _ in sib} & stored_h),
                                 hashes=[h for h, _ in sib])
    log(f"sibling family: {res['sibling_family']['n_siblings']} rebuilt, "
        f"{res['sibling_family']['n_overlap_with_stored']}/{len(stored_h)} match 51's stored set")
    save("SIBLINGS_BUILT")

    # ---- fresh contexts ----
    rv = np.fromfile(EXT / "results/reserve_tokens.i32", dtype=np.int32).reshape(-1, 256)
    tokens2d = rv[:, 1:]                               # drop BOS -> (8000, 255), rig's layout
    eligible = np.arange(len(tokens2d), dtype=np.int64)

    probe = S51.Runner(model, tok, 6, DEVICE)
    blk0 = probe.prepare([np.zeros(0, dtype=np.int64)], int(pool_ids[0]))
    lg0, hn0 = probe.baseline(blk0)
    hook = float((lg0 - probe.steered(blk0, hn0, np.zeros(1152), 0.0)).abs().max())
    assert hook == 0.0
    res["hook_identity"] = hook

    n_prompts = 4 if a.smoke else N_PROMPTS
    rng_dirs = np.random.default_rng(SEED_DIRS)
    rng_null = np.random.default_rng(SEED_NULL)

    def score(runner, L, members, seed, extra=None):
        mp = np.asarray(sorted(members))
        geo = S51.plane_of(C1[L][mp])
        rows = S51.eval_set(runner, geo, mp, pool_ids, tokens2d, eligible, n_prompts, EPS_GRID,
                            np.random.default_rng(seed), extra_dirs=extra)
        sm = {f"eps={e}": S51.summarise(rows, f"eps={e}") for e in EPS_GRID}
        return geo, rows, sm

    def sc_ok(sm, k):
        return any(sm[f"eps={e}"]["sign_consistency"] >= (k - 1) / k for e in EPS_GRID)

    out_sets = {}
    for role, c in (("anchor", anchor), ("candidate", cand)):
        L, k = c["layer"], c["k"]
        runner = S51.Runner(model, tok, L, DEVICE)
        mp = np.asarray(sorted(c["member_pos"]))
        geo = S51.plane_of(C1[L][mp])
        tang = S51.embed_1152(geo["tangent"])
        rd = np.stack([S51.random_orth_dirs(tang[i], N_RAND, rng_dirs) for i in range(k)])
        dirs = {f"rand{j}": rd[:, j, :] for j in range(N_RAND)}
        u1, u2, th = geo["u1"], geo["u2"], geo["theta"]
        dirs["rot_p90"] = S51.embed_1152(-(np.cos(th)[:, None] * u1) - (np.sin(th)[:, None] * u2))
        dirs["rot_m90"] = S51.embed_1152((np.cos(th)[:, None] * u1) + (np.sin(th)[:, None] * u2))
        # the candidate's natural arm and the direction nulls share the SAME contexts
        _, rows, nat = score(runner, L, mp, SEED_CTX + 100 * L + k, extra=dirs)
        null_i = {f"rand{j}": S51.summarise(rows, f"rand{j}") for j in range(N_RAND)}
        null_ii = {nm: S51.summarise(rows, nm) for nm in ("rot_p90", "rot_m90")}
        log(f"{role}: meanD@{EPS_NULL} {nat[f'eps={EPS_NULL}']['mean_delta']:+.4f} "
            f"sc {[nat[f'eps={e}']['sign_consistency'] for e in EPS_GRID]}")
        n3 = 3 if a.smoke else N_NULL_III[role]
        null_iii = []
        while len(null_iii) < n3:
            pick = sorted(rng_null.choice(len(pool_ids), size=k, replace=False).tolist())
            if pick == sorted(mp.tolist()) or S51.is_degenerate(C1[L][np.asarray(pick)]):
                continue
            _, _, sm = score(runner, L, pick, SEED_NULL + 1000 * len(null_iii) + k)
            null_iii.append(dict(member_hash=S51.mhash(pick), sc_ok=sc_ok(sm, k),
                                 **{f"d@{e}": sm[f"eps={e}"]["mean_delta"] for e in EPS_GRID}))
            if len(null_iii) % 10 == 0:
                log(f"  {role} null-iii {len(null_iii)}/{n3}")
        null_iv = []
        if role == "candidate":
            for j, (h, m) in enumerate(sib[: (3 if a.smoke else len(sib))]):
                _, _, sm = score(runner, L, m, SEED_SIB + 1000 * j + k)
                null_iv.append(dict(member_hash=h, tokens=[strings[x] for x in m],
                                    sc_ok=sc_ok(sm, k),
                                    **{f"d@{e}": sm[f"eps={e}"]["mean_delta"] for e in EPS_GRID}))
                if (j + 1) % 10 == 0:
                    log(f"  sibling {j+1}/{len(sib)}")
        d = nat[f"eps={EPS_NULL}"]["mean_delta"]
        iii = np.asarray([x[f"d@{EPS_NULL}"] for x in null_iii])
        p99_iii = float(np.percentile(iii, 99, method="higher"))
        imax = max(v["mean_delta"] for v in null_i.values())
        rec = dict(layer=L, k=k, member_hash=c["member_hash"],
                   tokens=[strings[p] for p in mp], geometry=dict(gap_ratio=geo["gap_ratio"]),
                   natural=nat, null_i=null_i, null_ii=null_ii, null_iii=null_iii,
                   null_iii_p99=p99_iii, null_i_max=imax,
                   sign_consistency_ok=sc_ok(nat, k), above_iii=bool(d > p99_iii),
                   above_i=bool(d > imax), mean_delta_at_eps_null=d,
                   realised_false_rate_random=float(np.mean([x["sc_ok"] and x[f"d@{EPS_NULL}"] > p99_iii
                                                             for x in null_iii])))
        if role == "candidate":
            iv = np.asarray([x[f"d@{EPS_NULL}"] for x in null_iv])
            p99_iv = float(np.percentile(iv, 99, method="higher")) if len(iv) else None
            rec.update(null_iv=null_iv, null_iv_p99=p99_iv,
                       above_iv=bool(p99_iv is not None and d > p99_iv),
                       rank_p_vs_siblings=float((1 + (iv >= d).sum()) / (1 + len(iv))),
                       realised_false_rate_siblings=float(np.mean(
                           [x["sc_ok"] and x[f"d@{EPS_NULL}"] > p99_iii for x in null_iv]))
                       if null_iv else None)
            rec["PASS"] = bool(rec["sign_consistency_ok"] and rec["above_iii"] and rec["above_i"]
                               and rec["above_iv"])
        else:
            rec["PASS"] = bool(rec["sign_consistency_ok"] and rec["above_iii"] and rec["above_i"])
        rec["n_forward"] = runner.n_forward
        out_sets[role] = rec
        res["sets"] = out_sets
        save(f"{role.upper()}_DONE")
        log(f"{role}: PASS={rec['PASS']} d={d:+.4f} iii_p99={p99_iii:+.4f} i_max={imax:+.4f}"
            + (f" iv_p99={rec['null_iv_p99']} rank_p={rec['rank_p_vs_siblings']:.3f}"
               if role == "candidate" else ""))
        if role == "anchor" and not rec["PASS"] and not a.smoke:
            res["verdict"] = "ANCHOR GATE FAILED -- rig cannot recover the positive control; stop"
            save("STOPPED")
            return
    assert hashlib.sha256(PREREG.read_bytes()).hexdigest() == prereg_sha, "prereg changed mid-run"
    c = out_sets["candidate"]
    res["verdict"] = ("BEHAVIOURAL PASS (confirmatory, fresh contexts, matched budget)" if c["PASS"]
                      else "FAIL -- not confirmed")
    save("COMPLETE")
    log(res["verdict"])


if __name__ == "__main__":
    main()
