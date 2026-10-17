# read-only D2 calibration: (1) knockout and planted controls in e; (2) context autocorrelation of e
import sys, time, json, numpy as np, torch, gc
from pathlib import Path
from safetensors.numpy import load_file
import scipy.sparse as sp
from sklearn.decomposition import FastICA
from sklearn.metrics import roc_auc_score
EXT = Path("/Users/scasella/Downloads/gf-discovery-ext"); t0 = time.time()
fz = np.load(EXT / "results/occ_units_v2_L12.npz"); rows = fz["row"].astype(np.int64)
m = np.load(EXT / "cache/occ/occD2_meta.npz"); toks_full = m["tokens"].astype(np.int64)
tok, seq, pos = toks_full[rows], m["seq"].astype(np.int64)[rows], m["pos"].astype(np.int64)[rows]
groups = [np.unique(a, return_inverse=True)[1] for a in (tok, seq, pos)]
Ms = []
for g in groups:
    G = int(g.max()) + 1; M = sp.csr_matrix((np.ones(len(g)), (g, np.arange(len(g)))), shape=(G, len(g)))
    Ms.append((M, np.maximum(np.asarray(M.sum(1)).ravel(), 1), g))
def demean3(V):
    V = V.astype(np.float64)
    for _ in range(3):
        for M, c, g in Ms: V -= (M @ V / c[:, None])[g]
    return V
def inv_sqrt(S_):
    w, U = np.linalg.eigh(S_); w = np.maximum(w, 1e-6 * w.max()); return (U / np.sqrt(w)) @ U.T
tinv = groups[0]; cnt = np.bincount(tinv)
def type_whiten(Z, n_min=32, n0=16):
    n_t = np.bincount(tinv); order = np.argsort(tinv, kind="stable"); start = np.cumsum(n_t) - n_t
    Z = Z.astype(np.float64)
    mu = np.stack([np.bincount(tinv, weights=Z[:, j], minlength=len(n_t)) for j in range(Z.shape[1])], 1)
    Zc = Z - (mu / n_t[:, None])[tinv]; Sp = Zc.T @ Zc / len(Zc); o_ = Zc @ inv_sqrt(Sp)
    for t in np.flatnonzero(n_t >= n_min):
        ii = order[start[t]:start[t] + n_t[t]]; St = Zc[ii].T @ Zc[ii] / n_t[t]; lam = n0 / (n_t[t] + n0)
        o_[ii] = Zc[ii] @ inv_sqrt((1 - lam) * St + lam * Sp)
    o_ -= o_.mean(0); w, U = np.linalg.eigh(np.cov(o_[::2].T)); oo = np.argsort(w)[::-1]
    return (o_ @ U[:, oo] / np.sqrt(w[oo])).astype(np.float32)
def eta2(v):
    mt = np.bincount(tinv, weights=v) / cnt
    return float(((mt[tinv] - v.mean()) ** 2).sum() / max(((v - v.mean()) ** 2).sum(), 1e-12))
def skew_g(x): return x ** 2, (2 * x).mean(axis=-1)
def fit(Wx, n, contrast, s):
    for k in range(3):
        try:
            f = FastICA(n_components=n, whiten="unit-variance", random_state=s + 100 * k, max_iter=400,
                        fun="logcosh" if contrast == "logcosh" else skew_g).fit(Wx[::2])
            if np.isfinite(f.mixing_).all(): return f
        except Exception: pass
def dm(A, B):
    A = A.T / np.linalg.norm(A.T, axis=1, keepdims=True); B = B.T / np.linalg.norm(B.T, axis=1, keepdims=True); return np.abs(A @ B.T).max(1)
ev = seq % 2 == 0
from transformers import AutoTokenizer
tk = AutoTokenizer.from_pretrained(str(sorted((EXT / "hf_home/hub/models--google--gemma-3-1b-pt/snapshots").glob("*"))[0]))
S = {}; par = np.zeros(len(toks_full), bool); quo = np.zeros(len(toks_full), bool)
for s0 in range(0, len(toks_full), 255):
    d = q = 0
    for j in range(255):
        t = int(toks_full[s0 + j]); st = S.setdefault(t, tk.decode([t])); par[s0 + j] = d > 0; quo[s0 + j] = q % 2 == 1
        d = max(0, d + st.count("(") - st.count(")")); q += st.count('"') + st.count("“") + st.count("”")
ypar, yquo = par[rows], quo[rows]
H = np.load(EXT / "results/e67_raw_L12_rows.npy").astype(np.float32)
snap = sorted((EXT / "hf_home/hub/models--google--gemma-scope-2-1b-pt/snapshots").glob("*"))[0]
P = load_file(str(snap / "resid_post_all/layer_12_width_16k_l0_big/params.safetensors"))
We, be, th, Wd, bd = [torch.from_numpy(P[k]) for k in ("w_enc", "b_enc", "threshold", "w_dec", "b_dec")]
keep = np.ones(1152, bool); keep[1038] = False
KP = [2682, 7780, 4001, 4493, 1559, 8951, 3503, 1442, 507, 1056]; KQ = [5254, 1229, 1994, 1534, 564, 1882, 7302, 10306, 398, 1550]
K = KP + KQ
E = np.empty((len(H), 1151), np.float32); KO = np.empty((len(H), 1151), np.float32); D = np.empty((len(H), 1151), np.float32)
for a in range(0, len(H), 8192):
    x = torch.from_numpy(H[a:a + 8192]); pre = x @ We + be; z = pre * (pre > th)
    E[a:a + 8192] = (x - z @ Wd - bd).numpy()[:, keep]; KO[a:a + 8192] = (z[:, K] @ Wd[K]).numpy()[:, keep]; D[a:a + 8192] = (z @ Wd).numpy()[:, keep]
del H; gc.collect()
print("KO share of e variance (raw):", round(float(KO.var(0).sum() / E.var(0).sum()), 3), flush=True)
def pipeline(X, tag, contrasts=("logcosh", "skew"), k=64, halves=True):
    V = demean3(X); mu = V.mean(0); C = np.cov((V - mu)[::3].T); w, U = np.linalg.eigh(C); o = np.argsort(w)[::-1][:k]
    Z = ((V - mu) @ U[:, o]) / np.sqrt(w[o]); del V; Wt = type_whiten(Z)
    out = {}
    for c in contrasts:
        f = fit(Wt, k, c, 0); Y = f.transform(Wt)
        if halves:
            hs = [fit(Wt[ev], k, c, 1), fit(Wt[~ev], k, c, 2)]
            rep = np.min([dm(f.mixing_, h.mixing_) if h is not None else np.zeros(k) for h in hs], 0)
        else:
            rep = np.ones(k)
        e2 = np.array([eta2(Y[:, j] ** 2) for j in range(k)]); scr = (rep >= 0.8) & (e2 <= 0.3)
        r = dict(n_screened=int(scr.sum()))
        for lb, y in (("paren", ypar), ("quote", yquo)):
            a = np.array([roc_auc_score(y, Y[:, j]) for j in range(k)]); a = np.maximum(a, 1 - a)
            r[lb] = dict(screened_max=round(float(a[scr].max()), 3) if scr.any() else None, unscreened_max=round(float(a.max()), 3))
        out[c] = r
    print(tag, json.dumps(out), f"{time.time()-t0:.0f}s", flush=True)
for alpha in (1.0, 0.5, 0.25):
    pipeline(E + alpha * KO, f"KO alpha={alpha}")
# planted single random direction, natural units
rng = np.random.default_rng(12345); u = rng.standard_normal(1151); u /= np.linalg.norm(u)
Ve0 = demean3(E); su = float((Ve0 @ u).std()); del Ve0
for kap in (0.5, 1.0, 2.0):
    pipeline(E + (kap * su * ypar.astype(np.float32))[:, None] * u[None, :].astype(np.float32), f"planted paren kappa={kap}", contrasts=("skew",))
# (2) context autocorrelation of e: identity-removed e_t vs e_{t-1} and dec_{t-1}, pairs with both rows kept
Ve = demean3(E).astype(np.float32); Vd = demean3(D).astype(np.float32); del E, D, KO; gc.collect()
ridx = {int(r): i for i, r in enumerate(rows)}
cur = np.array([i for i, r in enumerate(rows) if (int(r) - 1) in ridx and (r % 255) != 0]); prv = np.array([ridx[int(rows[i]) - 1] for i in cur])
evp = seq[cur] % 2 == 0
def r2(Xs, Ys, lam_rel=1e-3):
    G = Xs[evp].T.astype(np.float64) @ Xs[evp]; lam = lam_rel * np.trace(G) / G.shape[0]
    B = np.linalg.solve(G + lam * np.eye(G.shape[0]), Xs[evp].T.astype(np.float64) @ Ys[evp])
    R = Ys[~evp] - Xs[~evp] @ B; return round(float(1 - (R ** 2).sum() / ((Ys[~evp] - Ys[~evp].mean(0)) ** 2).sum()), 4)
Yc = Ve[cur].astype(np.float64)
print("pairs", len(cur), "R2 e_t ~ dec_t", r2(Vd[cur], Yc), " e_t ~ e_{t-1}", r2(Ve[prv], Yc), " e_t ~ dec_{t-1}", r2(Vd[prv], Yc),
      " e_t ~ [dec_t, dec_{t-1}]", r2(np.hstack([Vd[cur], Vd[prv]]), Yc), f"{time.time()-t0:.0f}s", flush=True)
# same for dec as reference
Dc = Vd[cur].astype(np.float64)
print("reference: R2 dec_t ~ dec_{t-1}", r2(Vd[prv], Dc), flush=True)
