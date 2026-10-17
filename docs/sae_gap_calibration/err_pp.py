# read-only D2: (1) full-rank whitening + deflationary projection pursuit in e, Gaussian surrogate, planted control; (2) lagged CCA of e
import time, json, numpy as np, torch, gc
from pathlib import Path
from safetensors.numpy import load_file
import scipy.sparse as sp
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
tinv = groups[0]; cnt = np.bincount(tinv)
def eta2(v):
    mt = np.bincount(tinv, weights=v) / cnt
    return float(((mt[tinv] - v.mean()) ** 2).sum() / max(((v - v.mean()) ** 2).sum(), 1e-12))
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
E = np.empty((len(H), 1151), np.float32)
for a in range(0, len(H), 8192):
    x = torch.from_numpy(H[a:a + 8192]); pre = x @ We + be; z = pre * (pre > th); E[a:a + 8192] = (x - z @ Wd - bd).numpy()[:, keep]
del H; gc.collect()
Ve = demean3(E).astype(np.float32); del E; gc.collect()
mu = Ve.mean(0); C = np.cov((Ve - mu)[::3].T); w, U = np.linalg.eigh(C); o = np.argsort(w)[::-1]; w, U = w[o], U[:, o]
WH = U / np.sqrt(w)                                    # full-rank whitening (1151)
ev = seq % 2 == 0
def pp(Z, m=24, contrast="pow3", seed=0, iters=200):
    """deflationary FastICA in the full whitened space; returns unit vectors (m x d)"""
    rng = np.random.default_rng(seed); d = Z.shape[1]; Wf = np.zeros((m, d)); Zt = torch.from_numpy(Z)
    for i in range(m):
        v = rng.standard_normal(d); v /= np.linalg.norm(v)
        for it in range(iters):
            y = Zt @ torch.from_numpy(v.astype(np.float32))
            if contrast == "pow3": g, gp = y ** 3, 3 * y ** 2
            else: g, gp = y ** 2, 2 * y
            vn = (Zt.T @ g).numpy().astype(np.float64) / len(Z) - float(gp.mean()) * v
            vn -= Wf[:i].T @ (Wf[:i] @ vn); vn /= np.linalg.norm(vn)
            conv = abs(abs(vn @ v) - 1) < 1e-6; v = vn
            if conv: break
        Wf[i] = v
    return Wf
def stats(Z, Wf):
    Y = Z @ Wf.T.astype(np.float32); Y = (Y - Y.mean(0)) / Y.std(0)
    return np.round(((Y ** 4).mean(0) - 3), 2), np.round((Y ** 3).mean(0), 2), Y
def dm(A, B): return np.abs(A @ B.T).max(1)
Z = ((Ve - mu) @ WH).astype(np.float32)
out = {}
for contrast in ("pow3", "skew"):
    Wf = pp(Z, contrast=contrast); ku, sk, Y = stats(Z, Wf)
    Wa = pp(Z[ev], contrast=contrast, seed=1); Wb = pp(Z[~ev], contrast=contrast, seed=2)
    rep = np.minimum(dm(Wf, Wa), dm(Wf, Wb))
    e2 = np.array([eta2(Y[:, j] ** 2) for j in range(len(Wf))])
    ap = [round(float(max(roc_auc_score(ypar, Y[:, j]), 1 - roc_auc_score(ypar, Y[:, j]))), 3) for j in range(len(Wf))]
    aq = [round(float(max(roc_auc_score(yquo, Y[:, j]), 1 - roc_auc_score(yquo, Y[:, j]))), 3) for j in range(len(Wf))]
    print("e", contrast, "exkurt", ku[:12].tolist(), "skew", sk[:12].tolist(), "repro", np.round(rep[:12], 2).tolist(),
          "eta2(y2)", np.round(e2[:12], 3).tolist(), "maxAUROC paren/quote", max(ap), max(aq), f"{time.time()-t0:.0f}s", flush=True)
# Gaussian surrogate, same n and d
G = np.random.default_rng(7).standard_normal(Z.shape).astype(np.float32)
for contrast in ("pow3", "skew"):
    Wf = pp(G, m=6, contrast=contrast); ku, sk, _ = stats(G, Wf)
    print("gauss", contrast, "exkurt", ku.tolist(), "skew", sk.tolist(), flush=True)
del G; gc.collect()
# planted paren along a random direction, natural units (kappa * sd of e along u), then full whitening + PP (skew contrast)
rng = np.random.default_rng(12345); u = rng.standard_normal(1151); u /= np.linalg.norm(u); su = float(((Ve - mu) @ u).std())
for kap in (1.0, 2.0):
    Vp = Ve + (kap * su * ypar.astype(np.float32))[:, None] * u[None, :].astype(np.float32)
    Vp = demean3(Vp).astype(np.float32); mp = Vp.mean(0); Cp = np.cov((Vp - mp)[::3].T); wp, Up = np.linalg.eigh(Cp)
    Zp = ((Vp - mp) @ (Up / np.sqrt(np.maximum(wp, 1e-9 * wp.max())))).astype(np.float32); del Vp
    Wf = pp(Zp, m=8, contrast="skew"); ku, sk, Y = stats(Zp, Wf)
    ap = [round(float(max(roc_auc_score(ypar, Y[:, j]), 1 - roc_auc_score(ypar, Y[:, j]))), 3) for j in range(len(Wf))]
    print(f"planted kappa={kap}: skew-PP top-8 |skew|", np.abs(sk).tolist(), "paren AUROC", ap, f"{time.time()-t0:.0f}s", flush=True)
    del Zp; gc.collect()
# lagged CCA in PCA-256 whitened e: canonical correlations at lag 1 and lag 5
Z256 = ((Ve - mu) @ (U[:, :256] / np.sqrt(w[:256]))).astype(np.float64)
ridx = {int(r): i for i, r in enumerate(rows)}
for lag in (1, 5, 20):
    cur = np.array([i for i, r in enumerate(rows) if (int(r) - lag) in ridx and (int(r) % 255) >= lag]); prv = np.array([ridx[int(rows[i]) - lag] for i in cur])
    A = Z256[cur] - Z256[cur].mean(0); B = Z256[prv] - Z256[prv].mean(0)
    def isq(S_): ww, UU = np.linalg.eigh(S_); return (UU / np.sqrt(ww)) @ UU.T
    Caa, Cbb, Cab = A.T @ A / len(A), B.T @ B / len(B), A.T @ B / len(A)
    s = np.linalg.svd(isq(Caa) @ Cab @ isq(Cbb), compute_uv=False)
    # null: shuffle pairing within the same set
    perm = np.random.default_rng(3).permutation(len(B)); Bn = B[perm]
    sn = np.linalg.svd(isq(Caa) @ (A.T @ Bn / len(A)) @ isq(Bn.T @ Bn / len(Bn)), compute_uv=False)
    print(f"lag {lag}: pairs {len(cur)} top canonical corr", np.round(s[:8], 3).tolist(), "null top", np.round(sn[:3], 3).tolist(), f"{time.time()-t0:.0f}s", flush=True)
