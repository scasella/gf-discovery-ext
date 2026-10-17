"""e61s -- Addendum S1 screen-threshold sweep on the pilot set P (L7). Units are fitted once per
K; per-unit statistics and their 1,000 token-matched permutation values are computed once;
every threshold combination then takes the max over its screened subset."""
import importlib.util, itertools, json, sys, time
from pathlib import Path
import numpy as np
EXT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXT / "src"))
spec = importlib.util.spec_from_file_location("e61", EXT / "scripts/e61_occurrence_screen.py")
e61 = importlib.util.module_from_spec(spec); sys.modules["e61"] = e61; spec.loader.exec_module(e61)
from scipy.stats import rankdata

D = e61.represent("occP", 7)
tk = e61.token_strings(); full = np.load(EXT / "cache/occ/occP_meta.npz")["tokens"]
F = e61.feature_battery(D, tk, full, None)
n = len(D["V"])
_, inv = np.unique(D["tok"], return_inverse=True)
slots = np.argsort(inv, kind="stable")
rng = np.random.default_rng(99)
perms = []
for _ in range(e61.N_PERM_PC):
    src = np.lexsort((rng.random(n), inv)); perms.append(src)
out = dict(addendum="docs/stage4_addendum_S1_screen_sweep_2026-10-15.md", rows=[])
for K_ in (64, 128, 256):
    e61.CFG.update(K=K_, N_ICA=32, N_PCA=64)
    t0 = time.time()
    fr, fit = e61.screen_layer(7, D, None, {})
    lab, Y = fit["lab"], fit["Y"]
    cl = [u for u in fr["units"] if u["kind"] == "cluster"]
    ic = [u for u in fr["units"] if u["kind"] == "ica"]
    size = np.bincount(lab, minlength=K_).astype(float)
    R = np.stack([rankdata(Y[:, j]) for j in range(32)], 1)
    stat = {}
    for name in ("paren_inside", "quote_inside"):
        y0 = F[name] > 0.5
        def per_unit(y):
            n1 = y.sum(); n0 = n - n1
            f1 = 2 * np.bincount(lab[y], minlength=K_) / (size + n1)
            auc = (R[y].sum(0) - n1 * (n1 + 1) / 2) / (n1 * n0)
            return f1, np.maximum(auc, 1 - auc)
        obs = per_unit(y0)
        nf, na = [], []
        for src in perms:
            yp = np.empty_like(y0); yp[slots] = y0[src]
            a_, b_ = per_unit(yp); nf.append(a_); na.append(b_)
        stat[name] = (obs, np.asarray(nf), np.asarray(na))
    for tdi, crep, bim, irep in itertools.product((3, 5, 10, np.inf), (0.8, 0.6, 0.0), (0.555, 0.0), (0.8, 0.5, 0.0)):
        cs = np.asarray([u["id"] for u in cl if u["size"] >= 300 and u["n_types"] >= 30 and u["max_type_share"] <= 0.3
                         and u["n_docs"] >= 100 and u["repro"] >= crep and u["tdi"] <= tdi], int)
        isel = np.asarray([u["id"] for u in ic if u["bimodality"] > bim and u["eta2_token"] <= 0.3 and u["repro"] >= irep], int)
        row = dict(K=K_, tdi_cap=float(tdi), cluster_repro=crep, ica_bimod=bim, ica_repro=irep,
                   n_clusters_screened=int(len(cs)), n_ica_screened=int(len(isel)))
        for name, ((f1, au), nf, na) in stat.items():
            mf = float(f1[cs].max()) if len(cs) else 0.0
            ma = float(au[isel].max()) if len(isel) else 0.5
            nfm = np.asarray([x[cs].max() for x in nf]) if len(cs) else np.zeros(len(nf))
            nam = np.asarray([x[isel].max() for x in na]) if len(isel) else np.full(len(na), 0.5)
            p_f, p_a = float(np.percentile(nfm, 99.9)), float(np.percentile(nam, 99.9))
            row[name] = dict(max_f1=mf, null_p999_f1=p_f, max_auroc=ma, null_p999_auroc=p_a,
                             recovered=bool((mf >= 0.5 and mf > p_f) or (ma >= 0.85 and ma > p_a)))
        out["rows"].append(row)
    print(f"K={K_} done in {time.time()-t0:.0f}s", flush=True)
    (EXT / "results/e61s_screen_sweep.json").write_text(json.dumps(out, indent=1, default=float))
rec = [r for r in out["rows"] if r["paren_inside"]["recovered"]]
out["n_recovering_combinations"] = len(rec)
out["chosen"] = (sorted(rec, key=lambda r: (r["tdi_cap"], -r["cluster_repro"], -r["ica_repro"], -r["ica_bimod"]))[0]
                 if rec else "none recovers: preregistered defaults frozen unchanged")
best = max(out["rows"], key=lambda r: max(r["paren_inside"]["max_f1"], r["paren_inside"]["max_auroc"] - 0.35))
out["best_paren_row"] = best
(EXT / "results/e61s_screen_sweep.json").write_text(json.dumps(out, indent=1, default=float))
print("recovering combinations:", len(rec)); print("best paren row:", json.dumps(best)[:600])
