# read-only: per-PC kurtosis and split-half ICA reproducibility at 64/128/256 dims, both contrasts, in e (big, small) and dec (big)
import sys, time, json, numpy as np, torch, gc
from pathlib import Path
from safetensors.numpy import load_file
import scipy.sparse as sp
from sklearn.decomposition import FastICA
EXT = Path("/Users/scasella/Downloads/gf-discovery-ext"); t0 = time.time()
OUT = Path(sys.argv[1])
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
H = np.load(EXT / "results/e67_raw_L12_rows.npy").astype(np.float32)
snap = sorted((EXT / "hf_home/hub/models--google--gemma-scope-2-1b-pt/snapshots").glob("*"))[0]
keep = np.ones(1152, bool); keep[1038] = False
res = {}
jobs = [("err_big", "big", (64, 128, 256)), ("err_small", "small", (64, 128, 256)), ("dec_big", "big", (128, 256))]
for name, which, dims in jobs:
    P = load_file(str(snap / f"resid_post_all/layer_12_width_16k_l0_{which}/params.safetensors"))
    We, be, th, Wd, bd = [torch.from_numpy(P[k]) for k in ("w_enc", "b_enc", "threshold", "w_dec", "b_dec")]
    X = np.empty((len(H), 1151), np.float32)
    for a in range(0, len(H), 8192):
        x = torch.from_numpy(H[a:a + 8192]); pre = x @ We + be; z = pre * (pre > th)
        r = (x - z @ Wd - bd) if name.startswith("err") else (z @ Wd)
        X[a:a + 8192] = r.numpy()[:, keep]
    V = demean3(X); del X; gc.collect()
    mu = V.mean(0); C = np.cov((V - mu)[::3].T); w, U = np.linalg.eigh(C); o = np.argsort(w)[::-1]; w, U = w[o], U[:, o]
    S = ((V - mu) @ U[:, :512]).astype(np.float32); del V; gc.collect()
    ku = ((S - S.mean(0)) ** 4).mean(0) / S.var(0) ** 2 - 3
    rec = dict(var_frac={k: round(float(w[:k].sum() / w.sum()), 3) for k in (64, 128, 256, 512)},
               eig_ratio={f"{k}/{k+1}": round(float(w[k - 1] / w[k]), 3) for k in (64, 128, 256)},
               pc_exkurt=dict(med_1_64=round(float(np.median(ku[:64])), 2), med_65_256=round(float(np.median(ku[64:256])), 2), med_257_512=round(float(np.median(ku[256:512])), 2),
                              n_gt3=int((ku > 3).sum()), n_gt10=int((ku > 10).sum()), max=round(float(ku.max()), 1), argmax=int(ku.argmax())), ica={})
    print(name, json.dumps(rec), f"{time.time()-t0:.0f}s", flush=True)
    for k in dims:
        Z = S[:, :k] / np.sqrt(w[:k]); Wt = type_whiten(Z)
        for contrast in ("logcosh", "skew"):
            f = fit(Wt, k, contrast, 0); hs = [fit(Wt[ev], k, contrast, 1), fit(Wt[~ev], k, contrast, 2)]
            rep = np.min([dm(f.mixing_, h.mixing_) if h is not None else np.zeros(k) for h in hs], 0)
            Y = f.transform(Wt); kk = ((Y - Y.mean(0)) ** 4).mean(0) / Y.var(0) ** 2 - 3
            e2sq = np.array([eta2(Y[:, j] ** 2) for j in range(k)])
            scr = (rep >= 0.8) & (e2sq <= 0.3)
            rec["ica"][f"{contrast}{k}"] = dict(repro_med=round(float(np.median(rep)), 2), n_rep08=int((rep >= 0.8).sum()), n_rep06=int((rep >= 0.6).sum()),
                                               n_screened=int(scr.sum()), eta2sq_max=round(float(e2sq.max()), 3),
                                               exkurt_screened=sorted([round(float(x), 1) for x in kk[scr]], reverse=True)[:12])
            print(name, contrast, k, json.dumps(rec["ica"][f"{contrast}{k}"]), f"{time.time()-t0:.0f}s", flush=True)
            res[name] = rec; OUT.write_text(json.dumps(res, indent=1))
    del S; gc.collect()
print("done", f"{time.time()-t0:.0f}s", flush=True)
