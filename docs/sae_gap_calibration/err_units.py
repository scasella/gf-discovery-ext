# read-only: do v2-style ICA units exist in the L12 16k-big SAE error on D2 (burned), and what do they track?
import time, json, numpy as np, torch
from pathlib import Path
from safetensors.numpy import load_file
import scipy.sparse as sp
from sklearn.decomposition import FastICA
from scipy.stats import spearmanr
EXT = Path("/Users/scasella/Downloads/gf-discovery-ext"); t0 = time.time()
fz = np.load(EXT / "results/occ_units_v2_L12.npz"); rows = fz["row"].astype(np.int64)
m = np.load(EXT / "cache/occ/occD2_meta.npz"); toks_full = m["tokens"].astype(np.int64)
tok, seq, pos = toks_full[rows], m["seq"].astype(np.int64)[rows], m["pos"].astype(np.int64)[rows]
H = np.load(EXT / "results/e67_raw_L12_rows.npy").astype(np.float32)
snap = sorted((EXT / "hf_home/hub/models--google--gemma-scope-2-1b-pt/snapshots").glob("*"))[0]
P = load_file(str(snap / "resid_post_all/layer_12_width_16k_l0_big/params.safetensors"))
We, be, th, Wd, bd = [torch.from_numpy(P[k]) for k in ("w_enc", "b_enc", "threshold", "w_dec", "b_dec")]
E = np.empty_like(H); L0 = np.empty(len(H)); zmax = np.empty(len(H))
for a in range(0, len(H), 8192):
    x = torch.from_numpy(H[a:a + 8192]); pre = x @ We + be; z = pre * (pre > th)
    E[a:a + 8192] = (x - z @ Wd - bd).numpy(); L0[a:a + 8192] = (z > 0).sum(1).numpy(); zmax[a:a + 8192] = z.max(1).values.numpy()
keep = np.ones(1152, bool); keep[1038] = False
hn = np.linalg.norm(H[:, keep], axis=1); en = np.linalg.norm(E[:, keep], axis=1); h1038 = H[:, 1038]
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
V = demean3(E[:, keep]).astype(np.float32)
mu = V.mean(0); C = np.cov((V - mu)[::3].T); w, U = np.linalg.eigh(C); o = np.argsort(w)[::-1]; w, U = w[o], U[:, o]
print("err eigen spectrum: top1/top64/top256 frac", round(w[0] / w.sum(), 4), round(w[:64].sum() / w.sum(), 3), round(w[:256].sum() / w.sum(), 3),
      " eig ratio 1/64", round(w[0] / w[63], 2), " 64/65", round(w[63] / w[64], 3), flush=True)
Z = ((V - mu) @ U[:, :64]) / np.sqrt(w[:64]); Wt = type_whiten(Z)
def fit(Wx, s):
    for k in range(3):
        try:
            f = FastICA(n_components=64, whiten="unit-variance", random_state=s + 100 * k, max_iter=400).fit(Wx[::2])
            if np.isfinite(f.mixing_).all(): return f
        except Exception: pass
ica = fit(Wt, 0); Y = ica.transform(Wt)
ev = seq % 2 == 0
h1, h2 = fit(Wt[ev], 1), fit(Wt[~ev], 2)
def dm(A, B):
    A = A.T / np.linalg.norm(A.T, axis=1, keepdims=True); B = B.T / np.linalg.norm(B.T, axis=1, keepdims=True); return np.abs(A @ B.T).max(1)
rep = np.minimum(dm(ica.mixing_, h1.mixing_), dm(ica.mixing_, h2.mixing_))
ku = ((Y - Y.mean(0)) ** 4).mean(0) / Y.var(0) ** 2 - 3; sk = ((Y - Y.mean(0)) ** 3).mean(0) / Y.std(0) ** 3
e2sq = np.array([eta2(Y[:, j] ** 2) for j in range(64)])
scr = (e2sq <= 0.3) & (rep >= 0.8)
print(f"ICA-64 logcosh on err: repro median {np.median(rep):.2f}, >=0.8: {(rep>=0.8).sum()}, eta2(y^2)<=0.3: {(e2sq<=0.3).sum()} (max {e2sq.max():.2f}), screened {scr.sum()}", flush=True)
print("ex-kurtosis of screened units, sorted:", np.round(np.sort(ku[scr])[::-1], 1).tolist(), flush=True)
covs = dict(log_hnorm=np.log(hn), log_enorm=np.log(en), L0=L0, zmax=zmax, h1038=h1038, pos=pos.astype(float))
lp = np.zeros(len(toks_full)); 
from transformers import AutoTokenizer
tk = AutoTokenizer.from_pretrained(str(sorted((EXT / "hf_home/hub/models--google--gemma-3-1b-pt/snapshots").glob("*"))[0]))
S = {}
for s0 in range(0, len(toks_full), 255):
    c = 0
    for j in range(255):
        t = int(toks_full[s0 + j]); st = S.setdefault(t, tk.decode([t])); lp[s0 + j] = c; c = 0 if "\n" in st else c + 1
covs["line_pos"] = lp[rows]
tab = []
for j in np.flatnonzero(scr):
    r = {k: round(float(abs(spearmanr(Y[::3, j], v[::3]).statistic)), 2) for k, v in covs.items()}
    r.update(unit=int(j), kurt=round(float(ku[j]), 1), skew=round(float(sk[j]), 2), rep=round(float(rep[j]), 2), e2sq=round(float(e2sq[j]), 3))
    tab.append(r)
tab.sort(key=lambda r: -r["kurt"])
for r in tab[:25]: print(r)
mx = {k: max(r[k] for r in tab) for k in covs}
print("max |spearman| over screened units:", mx, flush=True)
# contexts for the 4 most kurtotic screened units
for r in tab[:4]:
    j = r["unit"]; y = Y[:, j] * np.sign(sk[j] if abs(sk[j]) > 0.05 else 1)
    top = np.argsort(-y)[:200]; print(f"unit {j} kurt {r['kurt']} top-200 distinct types {len(set(tok[top]))}, docs {len(set(seq[top]))}")
    for i in np.random.default_rng(0).choice(top, 6, replace=False):
        rr = rows[i]; s0 = (rr // 255) * 255
        print("   ", repr(tk.decode(toks_full[max(s0, rr - 10):rr].tolist()) + "[[" + tk.decode([int(toks_full[rr])]) + "]]" + tk.decode(toks_full[rr + 1:min(s0 + 255, rr + 3)].tolist())))
print(f"{time.time()-t0:.0f}s")
