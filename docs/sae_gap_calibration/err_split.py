# read-only: (A) linear vs nonlinear SAE error (ridge e ~ SAE(h), e ~ h), identity-removed; (B) JumpReLU gating leakage of paren/quote
import sys, time, json, numpy as np, torch
from pathlib import Path
from safetensors.numpy import load_file
import scipy.sparse as sp
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
EXT = Path("/Users/scasella/Downloads/gf-discovery-ext"); t0 = time.time()
which = sys.argv[1]
fz = np.load(EXT / "results/occ_units_v2_L12.npz"); rows = fz["row"].astype(np.int64)
m = np.load(EXT / "cache/occ/occD2_meta.npz"); toks_full = m["tokens"].astype(np.int64)
tok, seq, pos = toks_full[rows], m["seq"].astype(np.int64)[rows], m["pos"].astype(np.int64)[rows]
H = np.load(EXT / "results/e67_raw_L12_rows.npy").astype(np.float32)
snap = sorted((EXT / "hf_home/hub/models--google--gemma-scope-2-1b-pt/snapshots").glob("*"))[0]
P = load_file(str(snap / f"resid_post_all/layer_12_width_16k_l0_{which}/params.safetensors"))
We, be, th, Wd, bd = [torch.from_numpy(P[k]) for k in ("w_enc", "b_enc", "threshold", "w_dec", "b_dec")]
from transformers import AutoTokenizer
tk = AutoTokenizer.from_pretrained(str(sorted((EXT / "hf_home/hub/models--google--gemma-3-1b-pt/snapshots").glob("*"))[0]))
S = {}
par = np.zeros(len(toks_full), bool); quo = np.zeros(len(toks_full), bool)
for s0 in range(0, len(toks_full), 255):
    d = q = 0
    for j in range(255):
        t = int(toks_full[s0 + j]); st = S.setdefault(t, tk.decode([t])); par[s0 + j] = d > 0; quo[s0 + j] = q % 2 == 1
        d = max(0, d + st.count("(") - st.count(")")); q += st.count('"') + st.count("“") + st.count("”")
ypar, yquo = par[rows], quo[rows]
lat = {"big": dict(paren=[2682, 7780, 4001, 4493, 1559], quote=[5254, 1229, 1994, 1882, 7302]),
       "small": dict(paren=[4493, 1351, 2682, 1559, 645], quote=[1229, 2899, 5082, 1472, 10306])}[which]
E = np.empty_like(H); D = np.empty_like(H); PRE = {}
for lb in lat: PRE[lb] = np.empty((len(H), 5), np.float32)
for a in range(0, len(H), 8192):
    x = torch.from_numpy(H[a:a + 8192]); pre = x @ We + be; z = pre * (pre > th)
    D[a:a + 8192] = (z @ Wd + bd).numpy(); E[a:a + 8192] = (x - z @ Wd - bd).numpy()
    for lb, ks in lat.items(): PRE[lb][a:a + 8192] = pre[:, ks].numpy()
thr = th.numpy()
# (B) gating: among rows where latent k is INACTIVE, does its pre-activation still separate the label?
gate = {}
for lb, ks in lat.items():
    y = ypar if lb == "paren" else yquo
    for i, k in enumerate(ks):
        pre_k = PRE[lb][:, i]; inact = pre_k <= thr[k]
        proj = E @ Wd.numpy()[k] / np.linalg.norm(Wd.numpy()[k])
        gate[f"{lb}#{k}"] = dict(active_rate_label=round(float((~inact[y]).mean()), 3), active_rate_other=round(float((~inact[~y]).mean()), 4),
                                 auroc_pre_among_inactive=round(float(roc_auc_score(y[inact], pre_k[inact])), 3),
                                 share_label_rows_inactive=round(float(inact[y].mean()), 3),
                                 auroc_err_proj_on_dec_row=round(float(max(roc_auc_score(y, proj), 1 - roc_auc_score(y, proj))), 3))
print(which, "GATING", json.dumps(gate), flush=True)
# (A) linear vs nonlinear error, identity-removed
keep = np.ones(1152, bool); keep[1038] = False
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
Ve = demean3(E[:, keep]); Vd = demean3(D[:, keep]); Vh = demean3(H[:, keep])
ev = seq % 2 == 0
def ridge_r2(Xs, Ys, lam_rel=1e-3):
    Xtr, Ytr = Xs[ev], Ys[ev]; G = Xtr.T @ Xtr; lam = lam_rel * np.trace(G) / G.shape[0]
    B = np.linalg.solve(G + lam * np.eye(G.shape[0]), Xtr.T @ Ytr)
    R = Ys[~ev] - Xs[~ev] @ B
    return float(1 - (R ** 2).sum() / ((Ys[~ev] - Ys[~ev].mean(0)) ** 2).sum()), B
r2_d, Bd = ridge_r2(Vd, Ve); r2_h, Bh = ridge_r2(Vh, Ve)
print(which, f"identity-removed held-out R^2: e ~ SAE(h) {r2_d:.3f}; e ~ h {r2_h:.3f}", f"{time.time()-t0:.0f}s", flush=True)
# probe paren/quote on e_nl = e - B_d SAE(h), PCA-64 and PCA-256 (no type whitening; quick)
Enl = Ve - Vd @ Bd
def pca_scores(V, k):
    mu = V.mean(0); C = np.cov((V - mu)[::3].T); w, U = np.linalg.eigh(C); o = np.argsort(w)[::-1]
    return ((V - mu) @ U[:, o[:k]]) / np.sqrt(w[o[:k]])
def probe(W, y):
    clf = LogisticRegression(C=1.0, max_iter=3000).fit(W[ev], y[ev]); return round(float(roc_auc_score(y[~ev], clf.decision_function(W[~ev]))), 3)
for nm, V in (("e_nl", Enl), ("e_lin", Vd @ Bd)):
    Z = pca_scores(V, 256)
    print(which, nm, "var share of e", round(float(((V - V.mean(0)) ** 2).sum() / ((Ve - Ve.mean(0)) ** 2).sum()), 3),
          "paren p64/p256", probe(Z[:, :64], ypar), probe(Z, ypar), "quote p64/p256", probe(Z[:, :64], yquo), probe(Z, yquo), flush=True)
print(f"{time.time()-t0:.0f}s")
