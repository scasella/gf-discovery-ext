"""e47r -- behavioural tier for Stage-2 ruler candidates (Addendum C5), 51's rig adapted to a line.

For a ruler S with level order m_1 < ... < m_k along its PC1 axis u (A-half raw type means, R10/H
drop, nuisance-deflated as in e46): at each INTERIOR member m_i (i = 2..k-1), steer
h <- h + eps*||h||*u_hat (u_hat re-embedded into 1152 dims, pointing toward increasing level) at
resid_post L at the member's own position, and read
  Delta_i = [logit(m_{i+1}) - logit(m_{i-1})]_steered - the same unsteered.
Contexts: 8-token slices from reserve sequences 2000-7999 (disjoint from Stage 4's slices).
Every arm at the SAME 40 prompts. Nulls: (i) 10 random directions orthogonal to u per member;
(iii) 50 random k-sets, each with its own PC1 axis and order; (iv) sibling geometric candidates
(other GEOM_CANDIDATE rulers of the same layer and k). PASS = sign consistency over interior
members >= (n-1)/n at some eps AND mean Delta@0.02 > (iii) p99 AND > max(i) AND > (iv) p99.
ANCHOR (label-known, run first): number words one..ten in their own PC1 order at the layer.
If the anchor fails, no verdict is issued on any candidate.
"""
import argparse, importlib.util, json, os, sys, time
from pathlib import Path

import numpy as np
import torch

EXT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXT / "src"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")


def _imp(name, fn):
    spec = importlib.util.spec_from_file_location(name, EXT / "scripts" / fn)
    m = importlib.util.module_from_spec(spec); sys.modules[name] = m
    spec.loader.exec_module(m); return m


S51 = _imp("s51", "51_verify_candidates_di.py")
E46 = _imp("e46", "e46_ruler_lattice.py")
import gfdisc.model as gm                                          # noqa: E402
gm.MODEL_NAME = str(sorted((EXT / "hf_home/hub/models--google--gemma-3-1b-pt/snapshots").glob("*"))[0])
from gfdisc import rig                                             # noqa: E402
from gfdisc.env import DEVICE                                      # noqa: E402

N_PROMPTS = 40
EPS_GRID = (0.005, 0.01, 0.02)
EPS_NULL = 0.02
N_RAND = 10
N_III = 50


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--blind", default=str(EXT / "results/e46_blind.json"))
    ap.add_argument("--max-candidates", type=int, default=5)
    ap.add_argument("--out", default=str(EXT / "results/e47r_ruler_behaviour.json"))
    a = ap.parse_args()
    t0 = time.time()
    blind = json.loads(Path(a.blind).read_text())
    cands = [c for c in blind["candidates"] if c.get("GEOM_CANDIDATE_RULER")]
    cands.sort(key=lambda c: (c["p_bg_graded"] if c["T2g"] else c["p_bg_spacing"]))
    P = E46.Pool(EXT / "results/typemeans_pass1.npz", EXT / "results/corpus_tokens_pass1.i32")
    model, tok = gm.load_model(token=None)
    rv = np.fromfile(EXT / "results/reserve_tokens.i32", dtype=np.int32).reshape(-1, 256)[2000:8000, 1:]
    eligible = np.arange(len(rv), dtype=np.int64)
    res = dict(script="scripts/e47r_ruler_behaviour.py", rules=__doc__, n_geom_candidates=len(cands))

    def axis_and_order(Ld, members):
        idx = np.asarray(members)
        r = E46.ruler_stats(Ld["A"][idx], Ld["B"][idx])
        u = r["u"]
        o = np.argsort(r["pA"])
        return u, [int(idx[i]) for i in o]

    def embed(u, L):
        drop = P.drop_for(L)
        keep = np.ones(1152, bool); keep[drop] = False
        v = np.zeros(1152); v[keep] = u
        return v / np.linalg.norm(v)

    def score(runner, L, ordered, u_hat, n_prompts, eps_list, seed, extra=None):
        ids = [int(P.pool_ids[i]) for i in ordered]
        rng = np.random.default_rng(seed)
        rows = []
        for i in range(1, len(ordered) - 1):
            ctx, _ = rig.sample_contexts(rv, n_prompts, 8, rng, eligible=eligible)
            blk = runner.prepare(ctx, ids[i])
            lg0, hn = runner.baseline(blk)
            cell = dict(i=i, conds={})
            for e in eps_list:
                lg = runner.steered(blk, hn, u_hat, e)
                cell["conds"][f"eps={e}"] = S51.cell_delta(lg, lg0, blk, ids[i + 1], ids[i - 1], ids)
            for nm, D in (extra or {}).items():
                lg = runner.steered(blk, hn, D[i], EPS_NULL)
                cell["conds"][nm] = S51.cell_delta(lg, lg0, blk, ids[i + 1], ids[i - 1], ids)
            rows.append(cell)
        return rows

    def summ(rows, cond):
        d = np.asarray([r["conds"][cond]["delta"] for r in rows])
        return dict(mean_delta=float(d.mean()), sign_consistency=float((d > 0).mean()), n=len(d))

    def run_set(L, members, label, sibs):
        Ld = P.layer(L)
        u, ordered = axis_and_order(Ld, members)
        u_hat = embed(u, L)
        runner = S51.Runner(model, tok, L, DEVICE)
        rngd = np.random.default_rng(4747 + L)
        dirs = {f"rand{j}": np.stack([S51.random_orth_dirs(u_hat, 1, rngd)[0] for _ in ordered])
                for j in range(N_RAND)}
        rows = score(runner, L, ordered, u_hat, N_PROMPTS, EPS_GRID, 4700 + L, extra=dirs)
        nat = {f"eps={e}": summ(rows, f"eps={e}") for e in EPS_GRID}
        nrand = [summ(rows, f"rand{j}")["mean_delta"] for j in range(N_RAND)]
        rng = np.random.default_rng(4800 + L)
        iii = []
        while len(iii) < N_III:
            pick = rng.choice(P.n, size=len(members), replace=False)
            u2, o2 = axis_and_order(Ld, pick)
            r2 = score(runner, L, o2, embed(u2, L), N_PROMPTS, (EPS_NULL,), 4900 + len(iii))
            iii.append(summ(r2, f"eps={EPS_NULL}")["mean_delta"])
        ivv = []
        for sm in sibs:
            u3, o3 = axis_and_order(Ld, sm)
            r3 = score(runner, L, o3, embed(u3, L), N_PROMPTS, (EPS_NULL,), 5000 + len(ivv))
            ivv.append(summ(r3, f"eps={EPS_NULL}")["mean_delta"])
        n_int = len(ordered) - 2
        d = nat[f"eps={EPS_NULL}"]["mean_delta"]
        sc_ok = any(nat[f"eps={e}"]["sign_consistency"] >= (n_int - 1) / n_int for e in EPS_GRID)
        p99_iii = float(np.percentile(iii, 99, method="higher"))
        p99_iv = float(np.percentile(ivv, 99, method="higher")) if ivv else None
        rec = dict(label=label, layer=L, k=len(members), tokens=[P.strings[i] for i in ordered],
                   natural=nat, null_i=nrand, null_iii_p99=p99_iii, null_iv=ivv, null_iv_p99=p99_iv,
                   PASS=bool(sc_ok and d > p99_iii and d > max(nrand)
                             and (p99_iv is None or d > p99_iv)),
                   n_forward=runner.n_forward)
        print(f"[{time.time()-t0:.0f}s] {label} L{L} k={len(members)} PASS={rec['PASS']} d={d:+.4f} "
              f"sc={[nat[f'eps={e}']['sign_consistency'] for e in EPS_GRID]} iii_p99={p99_iii:+.4f} "
              f"iv_p99={p99_iv} rand_max={max(nrand):+.4f}", flush=True)
        return rec

    out = []
    num_ids = [P.pos[int(P.tok.encode(w, add_special_tokens=False)[0])] for w in E46.NUM_WORDS]
    anchor_L = 6
    rec = run_set(anchor_L, num_ids, "anchor_numbers", [])
    out.append(rec); res["sets"] = out
    Path(a.out).write_text(json.dumps(res, indent=1, default=float))
    if not rec["PASS"]:
        res["verdict"] = "ANCHOR FAILED -- no behavioural verdict issued on any ruler candidate"
        Path(a.out).write_text(json.dumps(res, indent=1, default=float))
        print(res["verdict"], flush=True)
        return
    for c in cands[:a.max_candidates]:
        sibs = [s["members"] for s in cands if s["layer"] == c["layer"] and s["k"] == c["k"]
                and s["members"] != c["members"]][:30]
        out.append(run_set(c["layer"], c["members"], "candidate", sibs))
        res["sets"] = out
        Path(a.out).write_text(json.dumps(res, indent=1, default=float))
    res["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    Path(a.out).write_text(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main()
