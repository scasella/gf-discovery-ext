# read-only: SAE-blind subspace of h (generalized eig Cov_e v = lam Cov_h v, identity-removed), structure + labels
import sys, time, json, numpy as np, torch, gc
from pathlib import Path
from safetensors.numpy import load_file
import scipy.sparse as sp, scipy.linalg as sla
from sklearn.decomposition import FastICA
from sklearn.linear_model import LogisticRegression
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
H = np.load(EXT / "results/e67_raw_L12_rows.npy").astype(np.float32)
snap = sorted((EXT / "hf_home/hub/models--google--gemma-scope-2-1b-pt/snapshots").glob("*"))[0]
P = load_file(str(snap / "resid_post_all/layer_12_width_16k_l0_big/params.safetensors"))
We, be, th, Wd, bd = [torch.from_numpy(P[k]) for k in ("w_enc", "b_enc", "threshold", "w_dec", "b_dec")]
keep = np.ones(1152, bool); keep[1038] = False
E = np.empty((len(H), 1151), np.float32)
for a in range(0, len(H), 8192):
    x = torch.from_numpy(H[a:a + 8192]); pre = x @ We + be; z = pre * (pre > th); E[a:a + 8192] = (x - z @ Wd - bd).numpy()[:, keep]
Vh = demean3(H[:, keep]); del H; gc.collect()
Ch = np.cov(Vh[::3].T); Ve = demean3(E); del E; gc.collect(); Ce = np.cov(Ve[::3].T)
lam, Vg = sla.eigh(Ce, Ch)                    # Vg^T Ch Vg = I
o = np.argsort(lam)[::-1]; lam, Vg = lam[o], Vg[:, o]
print("blind-ratio lambda: top10", np.round(lam[:10], 3).tolist(), " #>=0.9", int((lam >= 0.9).sum()), "#>=0.75", int((lam >= 0.75).sum()),
      "#>=0.5", int((lam >= 0.5).sum()), " lam[63]", round(float(lam[63]), 3), " lam[255]", round(float(lam[255]), 3), flush=True)
# share of h's (identity-removed) variance in the top-64 blind directions, and of e's variance
tot_h = np.trace(Ch); tot_e = np.trace(Ce)
Q, _ = np.linalg.qr(Vg[:, :64])
print("top-64 blind subspace holds share of h var", round(float(np.trace(Q.T @ Ch @ Q) / tot_h), 3), " of e var", round(float(np.trace(Q.T @ Ce @ Q) / tot_e), 3), flush=True)
Zb = (Vh @ Vg[:, :64]).astype(np.float32)          # h along SAE-blind directions, unit variance by construction
Zb = (Zb - Zb.mean(0)) / Zb.std(0)
ku = ((Zb) ** 4).mean(0) - 3
print("exkurt of blind coords: median", round(float(np.median(ku)), 2), "max", round(float(ku.max()), 1), "#>3", int((ku > 3).sum()), flush=True)
Wt = type_whiten(Zb)
# labels
from transformers import AutoTokenizer
tk = AutoTokenizer.from_pretrained(str(sorted((EXT / "hf_home/hub/models--google--gemma-3-1b-pt/snapshots").glob("*"))[0]))
S = {}; n = len(toks_full); F = {k: np.zeros(n, bool) for k in ("paren", "quote", "after_newline", "next_newline", "next_punct", "prev_punct", "after_colon")}
for s0 in range(0, n, 255):
    d = q = 0; prev = ""
    for j in range(255):
        r = s0 + j; t = int(toks_full[r]); st = S.setdefault(t, tk.decode([t])); nx = S.setdefault(int(toks_full[r + 1]), tk.decode([int(toks_full[r + 1])])) if j < 254 else ""
        F["paren"][r] = d > 0; F["quote"][r] = q % 2 == 1; F["after_newline"][r] = "\n" in prev; F["next_newline"][r] = "\n" in nx
        F["next_punct"][r] = bool(nx.strip()) and not any(c.isalnum() for c in nx.strip())
        ps = prev.strip(); F["prev_punct"][r] = bool(ps) and not any(c.isalnum() for c in ps); F["after_colon"][r] = prev.rstrip().endswith(":")
        d = max(0, d + st.count("(") - st.count(")")); q += st.count('"') + st.count("“") + st.count("”"); prev = st
R = np.load(EXT / "results/e64c_rules_en_core_web_trf.npz")
labels = {k: v[rows] for k, v in F.items()}; labels.update({c: R[c][rows].astype(bool) for c in ("A", "B", "F", "G", "I", "J", "K", "N")})
def probe(W, y):
    clf = LogisticRegression(C=1.0, max_iter=3000).fit(W[ev], y[ev]); return round(float(roc_auc_score(y[~ev], clf.decision_function(W[~ev]))), 3)
res = dict(probes={lb: probe(Wt, y) for lb, y in labels.items()})
print("probes on type-whitened blind-64:", res["probes"], flush=True)
for contrast in ("logcosh", "skew"):
    f = fit(Wt, 64, contrast, 0); hs = [fit(Wt[ev], 64, contrast, 1), fit(Wt[~ev], 64, contrast, 2)]
    rep = np.min([dm(f.mixing_, h.mixing_) if h is not None else np.zeros(64) for h in hs], 0)
    Y = f.transform(Wt); kk = ((Y - Y.mean(0)) ** 4).mean(0) / Y.var(0) ** 2 - 3
    e2sq = np.array([eta2(Y[:, j] ** 2) for j in range(64)]); scr = (rep >= 0.8) & (e2sq <= 0.3)
    amax = {lb: round(float(max(max(roc_auc_score(y, Y[:, j]), 1 - roc_auc_score(y, Y[:, j])) for j in np.flatnonzero(scr))), 3) if scr.any() else None for lb, y in labels.items()}
    print(contrast, dict(repro_med=round(float(np.median(rep)), 2), n_rep08=int((rep >= 0.8).sum()), n_screened=int(scr.sum()), eta2sq_max=round(float(e2sq.max()), 3),
                         exkurt_screened=sorted([round(float(x), 1) for x in kk[scr]], reverse=True)[:10], screened_max_auroc=amax), f"{time.time()-t0:.0f}s", flush=True)
