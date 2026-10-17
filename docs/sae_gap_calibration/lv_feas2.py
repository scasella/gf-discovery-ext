"""Read-only feasibility, part 2 (D2, L12 raw, block = PCs 65-256 from part 1): identity leakage (KIR,
eta2 of y^2) and whether one full-rank ICA unit recovers the controls. Scratchpad only."""
import json, sys, time
import numpy as np

SP = sys.argv[1]
t0 = time.time()
Z = np.load(f"{SP}/lv_feas_L12_Z65_256.npy")
aux = np.load(f"{SP}/lv_feas_L12_aux.npz")
tok, seq = aux["tok"], aux["seq"]
tinv = np.unique(tok, return_inverse=True)[1]
n_t = np.bincount(tinv)
res = {}


def kir(W, tinv, seed=0, n_q=4000, k=10, chunk=250):
    rng = np.random.default_rng(seed)
    q = rng.choice(len(W), n_q, replace=False)
    sq = (W.astype(np.float32) ** 2).sum(1)
    obs = np.zeros(n_q)
    for a in range(0, n_q, chunk):
        qq = q[a:a + chunk]
        d = sq[qq][:, None] - 2 * (W[qq] @ W.T) + sq[None, :]
        d[np.arange(len(qq)), qq] = np.inf
        nn = np.argpartition(d, k, axis=1)[:, :k]
        obs[a:a + chunk] = (tinv[nn] == tinv[qq][:, None]).mean(1)
    exp_ = (n_t[tinv[q]] - 1) / (len(W) - 1)
    return float(obs.mean() / exp_.mean())


def eta2(v):
    mt = np.bincount(tinv, weights=v) / n_t
    return float(((mt[tinv] - v.mean()) ** 2).sum() / max(((v - v.mean()) ** 2).sum(), 1e-12))


res["kir_block_no_typewhite"] = kir(Z, tinv)
res["n_types"] = int(len(n_t)); res["types_ge_256"] = int((n_t >= 256).sum())
res["row_share_types_ge_256"] = float(n_t[n_t >= 256].sum() / len(tinv))
print(res, round(time.time() - t0), "s", flush=True)

from sklearn.decomposition import FastICA
from sklearn.metrics import roc_auc_score
ica = FastICA(n_components=Z.shape[1], whiten="unit-variance", random_state=0, max_iter=400,
              fun="logcosh").fit(Z[::2])
res["ica_n_iter"] = int(ica.n_iter_)
Y = ica.transform(Z)
print("ica", ica.n_iter_, round(time.time() - t0), "s", flush=True)
labs = {k: aux[k] > 0.5 for k in ("paren_inside", "quote_inside", "after_colon", "in_caps_run", "bracket_inside", "code_markup")}
au = {}
for k, y in labs.items():
    a = np.array([roc_auc_score(y, Y[:, j]) for j in range(Y.shape[1])])
    a = np.maximum(a, 1 - a)
    o = np.argsort(-a)[:3]
    au[k] = [(int(j), float(a[j])) for j in o]
res["ica_top_auroc"] = au
e2 = np.array([eta2(Y[:, j] ** 2) for j in range(Y.shape[1])])
ku = ((Y - Y.mean(0)) ** 4).mean(0) / Y.var(0) ** 2 - 3
res["eta2_sq"] = dict(max=float(e2.max()), med=float(np.median(e2)), n_gt_03=int((e2 > 0.3).sum()))
res["ex_kurtosis"] = dict(med=float(np.median(ku)), q90=float(np.quantile(ku, 0.9)), max=float(ku.max()),
                          n_gt_1=int((ku > 1).sum()))
# split-half reproducibility, even/odd sequences, same basis (one half only, seed 1)
ev = seq % 2 == 0
h = FastICA(n_components=Z.shape[1], whiten="unit-variance", random_state=1, max_iter=400, fun="logcosh").fit(Z[ev][::2])
A = ica.mixing_.T / np.linalg.norm(ica.mixing_.T, axis=1, keepdims=True)
B = h.mixing_.T / np.linalg.norm(h.mixing_.T, axis=1, keepdims=True)
rep = np.abs(A @ B.T).max(1)
res["repro_even_half"] = dict(med=float(np.median(rep)), n_ge_08=int((rep >= 0.8).sum()), n_ge_07=int((rep >= 0.7).sum()),
                              control_units={k: float(rep[au[k][0][0]]) for k in au})
res["control_units_eta2sq"] = {k: float(e2[au[k][0][0]]) for k in au}
res["control_units_kurt"] = {k: float(ku[au[k][0][0]]) for k in au}
print(json.dumps(res, indent=1), flush=True)
json.dump(res, open(f"{SP}/lv_feas2_L12.json", "w"), indent=1)
print("done", round(time.time() - t0), "s", flush=True)
