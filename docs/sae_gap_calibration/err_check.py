# read-only feasibility: does paren/quote state survive in the L12 SAE error on D2 (burned data)?
import sys, time, json, numpy as np, torch
from pathlib import Path
from safetensors.numpy import load_file
EXT = Path("/Users/scasella/Downloads/gf-discovery-ext")
t0 = time.time()
fz = np.load(EXT / "results/occ_units_v2_L12.npz"); rows = fz["row"].astype(np.int64)
m = np.load(EXT / "cache/occ/occD2_meta.npz"); toks_full = m["tokens"].astype(np.int64)
tok, seq, pos = toks_full[rows], m["seq"].astype(np.int64)[rows], m["pos"].astype(np.int64)[rows]
H = np.load(EXT / "results/e67_raw_L12_rows.npy").astype(np.float32)
print("H", H.shape, "max|1038|", np.abs(H[:, 1038]).max(), f"{time.time()-t0:.0f}s", flush=True)
snap = sorted((EXT / "hf_home/hub/models--google--gemma-scope-2-1b-pt/snapshots").glob("*"))[0]
which = sys.argv[1] if len(sys.argv) > 1 else "big"
P = load_file(str(snap / f"resid_post_all/layer_12_width_16k_l0_{which}/params.safetensors"))
We, be, th, Wd, bd = [torch.from_numpy(P[k]) for k in ("w_enc", "b_enc", "threshold", "w_dec", "b_dec")]
# labels (battery logic: state before the token)
from transformers import AutoTokenizer
tk = AutoTokenizer.from_pretrained(str(sorted((EXT / "hf_home/hub/models--google--gemma-3-1b-pt/snapshots").glob("*"))[0]))
S = {}
def s_of(t):
    if t not in S: S[t] = tk.decode([int(t)])
    return S[t]
par = np.zeros(len(toks_full), bool); quo = np.zeros(len(toks_full), bool)
for s0 in range(0, len(toks_full), 255):
    d = 0; q = 0
    for j in range(255):
        r = s0 + j; st = s_of(toks_full[r])
        par[r] = d > 0; quo[r] = q % 2 == 1
        d = max(0, d + st.count("(") - st.count(")")); q += st.count('"') + st.count("“") + st.count("”")
ypar, yquo = par[rows], quo[rows]
print("base rates paren/quote", ypar.mean().round(4), yquo.mean().round(4), flush=True)
# SAE error, latent association streams
E = np.empty_like(H); D = np.empty_like(H)
n = len(H); nl = We.shape[1]
sum_p = torch.zeros(nl, dtype=torch.float64); sum_n = torch.zeros(nl, dtype=torch.float64)
act_p = torch.zeros(nl, dtype=torch.float64); act_n = torch.zeros(nl, dtype=torch.float64)
sum_q = torch.zeros(nl, dtype=torch.float64); act_q = torch.zeros(nl, dtype=torch.float64)
l0 = 0.0
torch.set_num_threads(10)
for a in range(0, n, 8192):
    x = torch.from_numpy(H[a:a + 8192])
    pre = x @ We + be; z = pre * (pre > th)
    dec = z @ Wd + bd
    E[a:a + 8192] = (x - dec).numpy(); D[a:a + 8192] = (z @ Wd).numpy()
    yp = torch.from_numpy(ypar[a:a + 8192]); yq = torch.from_numpy(yquo[a:a + 8192])
    zd = z.double()
    sum_p += zd[yp].sum(0); sum_n += zd[~yp].sum(0); act_p += (zd[yp] > 0).sum(0); act_n += (zd[~yp] > 0).sum(0)
    sum_q += zd[yq].sum(0); act_q += (zd[yq] > 0).sum(0)
    l0 += float((z > 0).sum())
print(f"encode done {time.time()-t0:.0f}s  L0 {l0/n:.1f}", flush=True)
keep = np.ones(1152, bool); keep[1038] = False
def fvu(X, R, kc):
    Xc = X[:, kc] - X[:, kc].mean(0); Rc = R[:, kc] - R[:, kc].mean(0)
    return float((Rc ** 2).sum() / (Xc ** 2).sum())
out = dict(sae=which, L0=l0 / n, fvu_all=fvu(H, E, np.ones(1152, bool)), fvu_drop1038=fvu(H, E, keep))
ev = E - E.mean(0); out["err_coord_share_top5"] = sorted(((float(v), int(i)) for i, v in enumerate((ev ** 2).sum(0) / (ev ** 2).sum())), reverse=True)[:5]
del ev
# latents most associated with paren / quote
n_p, n_n = ypar.sum(), (~ypar).sum(); n_q = yquo.sum()
fr_p = (act_p / n_p).numpy(); fr_n = (act_n / n_n).numpy()
top_par = np.argsort(-(fr_p - fr_n))[:10]
fr_q = (act_q / n_q).numpy(); fr_all = ((act_p + act_n) / n).numpy()
top_quo = np.argsort(-(fr_q - fr_all))[:10]
out["paren_latents"] = [(int(i), round(float(fr_p[i]), 3), round(float(fr_n[i]), 4)) for i in top_par]
out["quote_latents"] = [(int(i), round(float(fr_q[i]), 3), round(float(fr_all[i]), 4)) for i in top_quo]
print(json.dumps(out), flush=True)
import scipy.sparse as sp
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
groups = [np.unique(a, return_inverse=True)[1] for a in (tok, seq, pos)]
Ms = []
for g in groups:
    G = int(g.max()) + 1; M = sp.csr_matrix((np.ones(len(g)), (g, np.arange(len(g)))), shape=(G, len(g)))
    Ms.append((M, np.maximum(np.asarray(M.sum(1)).ravel(), 1), g))
def demean3(V):
    V = V.astype(np.float64)
    for _ in range(3):
        for M, c, g in Ms:
            V -= (M @ V / c[:, None])[g]
    return V.astype(np.float32)
def pca(V, k):
    mu = V.mean(0); C = np.cov((V - mu)[::3].T); w, U = np.linalg.eigh(C); o = np.argsort(w)[::-1]
    return mu, U[:, o], np.sqrt(np.maximum(w[o], 1e-12)), w[o]
evn = seq % 2 == 0
def probe(W, y, C=1.0):
    clf = LogisticRegression(C=C, max_iter=3000).fit(W[evn], y[evn])
    return round(float(roc_auc_score(y[~evn], clf.decision_function(W[~evn]))), 3)
def inv_sqrt(S):
    w, U = np.linalg.eigh(S); w = np.maximum(w, 1e-6 * w.max()); return (U / np.sqrt(w)) @ U.T
tinv = groups[0]
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
def eta2_type(V):
    tot = ((V - V.mean(0)) ** 2).sum()
    M, c, g = Ms[0]; mt = M @ V.astype(np.float64) / c[:, None]
    return float((((mt[g] - V.mean(0)) ** 2).sum()) / tot)
res = {}
Upca = {}
for name, X in (("raw", H), ("dec", D), ("err", E)):
    V = X[:, keep]
    e2 = eta2_type(V)
    V = demean3(V)
    mu, U, sd, w = pca(V, 1151)
    Upca[name] = U[:, :64]
    r = dict(eta2_type_before=round(e2, 3), var_top64=round(float(w[:64].sum() / w.sum()), 3),
             var_top256=round(float(w[:256].sum() / w.sum()), 3), total_var_after_demean=float(w.sum()))
    Z64 = ((V - mu) @ U[:, :64]) / sd[:64]
    Z256 = ((V - mu) @ U[:, :256]) / sd[:256]
    Zb = ((V - mu) @ U[:, 64:256]) / sd[64:256]
    for lb, y in (("paren", ypar), ("quote", yquo)):
        r[f"{lb}_pca64"] = probe(Z64, y); r[f"{lb}_pca256"] = probe(Z256, y); r[f"{lb}_pc65to256"] = probe(Zb, y)
    Wt = type_whiten(Z64)
    for lb, y in (("paren", ypar), ("quote", yquo)):
        r[f"{lb}_v2rep"] = probe(Wt, y)
    if name == "err":
        from sklearn.decomposition import FastICA
        ica = FastICA(n_components=64, whiten="unit-variance", random_state=0, max_iter=400).fit(Wt[::2])
        Y = ica.transform(Wt)
        for lb, y in (("paren", ypar), ("quote", yquo)):
            a = np.array([roc_auc_score(y, Y[:, j]) for j in range(64)]); a = np.maximum(a, 1 - a)
            r[f"{lb}_ica64_logcosh_max"] = round(float(a.max()), 3); r[f"{lb}_ica64_top3"] = [round(float(x), 3) for x in np.sort(a)[::-1][:3]]
        # leakage check: remove the span of the top-10 paren / quote latents' decoder rows from err, re-probe
        Bl = Wd.numpy()[np.r_[top_par, top_quo]][:, keep].T.astype(np.float64)
        Qb, _ = np.linalg.qr(Bl)
        Vk = V - (V @ Qb) @ Qb.T
        mu2, U2, sd2, _ = pca(Vk, 1151); Zk = ((Vk - mu2) @ U2[:, :64]) / sd2[:64]
        Wk = type_whiten(Zk)
        for lb, y in (("paren", ypar), ("quote", yquo)):
            r[f"{lb}_v2rep_minus_top10latent_span"] = probe(Wk, y)
    res[name] = r
    print(name, json.dumps(r), f"{time.time()-t0:.0f}s", flush=True)
# subspace overlap of top-64 PCs: err vs dec vs raw
def overlap(A, B):
    s = np.linalg.svd(A.T @ B, compute_uv=False); return round(float((s ** 2).mean()), 3)
print("top64 subspace overlap err-dec", overlap(Upca["err"], Upca["dec"]), "err-raw", overlap(Upca["err"], Upca["raw"]),
      "dec-raw", overlap(Upca["dec"], Upca["raw"]), "random~", round(64 / 1151, 3), flush=True)
