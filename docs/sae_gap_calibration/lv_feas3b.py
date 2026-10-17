"""Read-only feasibility, part 3 (D2, L12 raw): how many block (PC 65-256) ICA units are shadows of
top-64 ICA units? Tail overlap and energy correlation. Scratchpad only."""
import json, sys, time
import numpy as np
import scipy.sparse as sp
from sklearn.decomposition import FastICA
from sklearn.metrics import roc_auc_score

EXT = "/Users/scasella/Downloads/gf-discovery-ext"
SP = sys.argv[1]
t0 = time.time()
H = np.load(f"{EXT}/results/e67_raw_L12_rows.npy", mmap_mode="r")
aux = np.load(f"{SP}/lv_feas_L12_aux.npz")
tok, seq, pos = aux["tok"], aux["seq"], aux["pos"]
keep = np.ones(1152, bool); keep[1038] = False
V = np.asarray(H, dtype=np.float32)[:, keep]


def demean(V, g):
    n, G = len(g), int(g.max()) + 1
    M = sp.csr_matrix((np.ones(n), (g, np.arange(n))), shape=(G, n))
    cnt = np.maximum(np.asarray(M.sum(1)).ravel(), 1)
    for j in range(0, V.shape[1], 128):
        b = V[:, j:j + 128].astype(np.float64)
        V[:, j:j + 128] = (b - (M @ b / cnt[:, None])[g]).astype(np.float32)


groups = [np.unique(x, return_inverse=True)[1] for x in (tok, seq, pos)]
for _ in range(3):
    for g in groups:
        demean(V, g)
mu = V.mean(0)
w, U = np.linalg.eigh(np.cov((V - mu)[::3].T)); o = np.argsort(w)[::-1]; w, U = w[o], U[:, o]
Z1 = ((V - mu) @ U[:, :64] / np.sqrt(w[:64])).astype(np.float32)
del V
Zb = np.load(f"{SP}/lv_feas_L12_Z65_256.npy")
print("rep", round(time.time() - t0), flush=True)
fit = lambda Z, n: FastICA(n_components=n, whiten="unit-variance", random_state=0, max_iter=400, fun="logcosh").fit(Z[::2])
Y1 = fit(Z1, 64).transform(Z1)
Yb = fit(Zb, 192).transform(Zb)
print("ica", round(time.time() - t0), flush=True)
n = len(Yb); k1 = int(np.ceil(0.01 * n))


def heavy_top(y):
    lo, hi = np.quantile(y, [0.01, 0.99])
    y = y if abs(hi) >= abs(lo) else -y
    return np.argsort(-y)[:k1]


def two_sided(y):
    o = np.argsort(y)
    s = np.zeros(n, bool); s[o[:k1]] = True; s[o[-k1:]] = True
    return s


T1 = np.stack([two_sided(Y1[:, j]) for j in range(64)], 1)
E1 = (Y1 ** 2 - (Y1 ** 2).mean(0)) / (Y1 ** 2).std(0)
Eb = (Yb ** 2 - (Yb ** 2).mean(0)) / (Yb ** 2).std(0)
ecorr = np.abs(Eb.T @ E1) / n                      # 192 x 64
ov = np.zeros((192, 64))
for u in range(192):
    ov[u] = T1[heavy_top(Yb[:, u])].mean(0)
res = dict(overlap_max=dict(q=[float(x) for x in np.quantile(ov.max(1), [0.1, 0.25, 0.5, 0.75, 0.9])],
                            n_ge_020=int((ov.max(1) >= 0.20).sum()), n_ge_010=int((ov.max(1) >= 0.10).sum())),
           ecorr_max=dict(q=[float(x) for x in np.quantile(ecorr.max(1), [0.1, 0.25, 0.5, 0.75, 0.9])],
                          n_ge_020=int((ecorr.max(1) >= 0.20).sum()), n_ge_010=int((ecorr.max(1) >= 0.10).sum())),
           shadow_either_020=int(((ov.max(1) >= 0.20) | (ecorr.max(1) >= 0.20)).sum()))
labs = {k: aux[k] > 0.5 for k in ("paren_inside", "quote_inside", "after_colon")}
ctrl = {}
for k, y in labs.items():
    a1 = np.array([roc_auc_score(y, Y1[:, j]) for j in range(64)]); a1 = np.maximum(a1, 1 - a1)
    ab = np.array([roc_auc_score(y, Yb[:, j]) for j in range(192)]); ab = np.maximum(ab, 1 - ab)
    u, v = int(ab.argmax()), int(a1.argmax())
    ctrl[k] = dict(block_unit=u, block_auroc=float(ab[u]), top64_unit=v, top64_auroc=float(a1[v]),
                   overlap_with_top64_ctrl=float(ov[u, v]), ecorr_with_top64_ctrl=float(ecorr[u, v]),
                   max_overlap=float(ov[u].max()), argmax_overlap=int(ov[u].argmax()),
                   max_ecorr=float(ecorr[u].max()), argmax_ecorr=int(ecorr[u].argmax()),
                   block_top1pct_rate=float(y[heavy_top(Yb[:, u])].mean()), base=float(y.mean()))
res["controls"] = ctrl
print(json.dumps(res, indent=1), flush=True)
json.dump(res, open(f"{SP}/lv_feas3_L12.json", "w"), indent=1)
np.save(f"{SP}/Y1.npy", Y1.astype(np.float32)); np.save(f"{SP}/Yb.npy", Yb.astype(np.float32))
print("done", round(time.time() - t0), flush=True)
