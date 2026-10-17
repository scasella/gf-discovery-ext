# read-only D2: lagged-covariance BSS (multi-lag AMUSE) on identity-removed e; controls; labels; split-half; dec reference
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
    return V.astype(np.float32)
tinv = groups[0]; cnt = np.bincount(tinv)
def eta2(v):
    mt = np.bincount(tinv, weights=v) / cnt
    return float(((mt[tinv] - v.mean()) ** 2).sum() / max(((v - v.mean()) ** 2).sum(), 1e-12))
from transformers import AutoTokenizer
tk = AutoTokenizer.from_pretrained(str(sorted((EXT / "hf_home/hub/models--google--gemma-3-1b-pt/snapshots").glob("*"))[0]))
S = {}; n = len(toks_full); F = {k: np.zeros(n, bool) for k in ("paren", "quote", "after_newline", "next_newline", "prev_punct", "after_colon", "code_like")}
for s0 in range(0, n, 255):
    d = q = 0; prev = ""
    for j in range(255):
        r = s0 + j; t = int(toks_full[r]); st = S.setdefault(t, tk.decode([t])); nx = S.setdefault(int(toks_full[r + 1]), tk.decode([int(toks_full[r + 1])])) if j < 254 else ""
        F["paren"][r] = d > 0; F["quote"][r] = q % 2 == 1; F["after_newline"][r] = "\n" in prev; F["next_newline"][r] = "\n" in nx
        ps = prev.strip(); F["prev_punct"][r] = bool(ps) and not any(c.isalnum() for c in ps); F["after_colon"][r] = prev.rstrip().endswith(":")
        d = max(0, d + st.count("(") - st.count(")")); q += st.count('"') + st.count("“") + st.count("”"); prev = st
R = np.load(EXT / "results/e64c_rules_en_core_web_trf.npz")
labels = {k: v[rows] for k, v in F.items() if k != "code_like"}
labels.update({nm: R[c][rows].astype(bool) for c, nm in (("A", "after_cc"), ("B", "temporal_cl"), ("F", "subj_np"), ("G", "fronted"), ("I", "line_init"), ("J", "negation"), ("K", "copular"), ("N", "after_the"), ("M", "temporal_expr"), ("L", "enumeration"))})
ypar = labels["paren"]
H = np.load(EXT / "results/e67_raw_L12_rows.npy").astype(np.float32)
snap = sorted((EXT / "hf_home/hub/models--google--gemma-scope-2-1b-pt/snapshots").glob("*"))[0]
P = load_file(str(snap / "resid_post_all/layer_12_width_16k_l0_big/params.safetensors"))
We, be, th, Wd, bd = [torch.from_numpy(P[k]) for k in ("w_enc", "b_enc", "threshold", "w_dec", "b_dec")]
keep = np.ones(1152, bool); keep[1038] = False
KP = [2682, 7780, 4001, 4493, 1559, 8951, 3503, 1442, 507, 1056]; KQ = [5254, 1229, 1994, 1534, 564, 1882, 7302, 10306, 398, 1550]; K = KP + KQ
L0 = np.empty(len(H)); HN = np.empty(len(H)); E = np.empty((len(H), 1151), np.float32); KO = np.empty((len(H), 1151), np.float32); D = np.empty((len(H), 1151), np.float32)
for a in range(0, len(H), 8192):
    x = torch.from_numpy(H[a:a + 8192]); pre = x @ We + be; z = pre * (pre > th)
    E[a:a + 8192] = (x - z @ Wd - bd).numpy()[:, keep]; L0[a:a + 8192] = (z > 0).sum(1).numpy(); HN[a:a + 8192] = x[:, keep].norm(dim=1).numpy(); KO[a:a + 8192] = (z[:, K] @ Wd[K]).numpy()[:, keep]; D[a:a + 8192] = (z @ Wd).numpy()[:, keep]
del H; gc.collect()
ridx = {int(r): i for i, r in enumerate(rows)}
LAGS = (1, 2, 4, 8, 16)
pairs = {}
for L in LAGS:
    cur = np.array([i for i, r in enumerate(rows) if (int(r) - L) in ridx and (int(r) % 255) >= L]); pairs[L] = (cur, np.array([ridx[int(rows[i]) - L] for i in cur]))
evr = seq % 2 == 0
def amuse(Z, sel=None):
    Mx = np.zeros((Z.shape[1], Z.shape[1]))
    for L in LAGS:
        c, p = pairs[L]
        if sel is not None:
            k = sel[c]; c, p = c[k], p[k]
        A, B = Z[c].astype(np.float64), Z[p].astype(np.float64)
        Cl = (A.T @ B) / len(A); Mx += (Cl + Cl.T) / 2
    w, U = np.linalg.eigh(Mx / len(LAGS)); o = np.argsort(w)[::-1]
    return w[o], U[:, o]
def whiten(V, k):
    mu = V.mean(0); C = np.cov((V - mu)[::3].T); w, U = np.linalg.eigh(C); o = np.argsort(w)[::-1][:k]
    return (((V - mu) @ U[:, o]) / np.sqrt(w[o])).astype(np.float32), U[:, o], np.sqrt(w[o])
def acf(y, L):
    c, p = pairs[L]; return float(np.corrcoef(y[c], y[p])[0, 1])
def analyse(V, tag, k=256, top=16, planted_u=None):
    Z, Up, sd = whiten(V, k)
    w, U = amuse(Z); wa, Ua = amuse(Z, evr); wb, Ub = amuse(Z, ~evr)
    rep = np.minimum(np.abs(U[:, :top].T @ Ua[:, :top * 2]).max(1), np.abs(U[:, :top].T @ Ub[:, :top * 2]).max(1))
    Y = Z @ U[:, :top]
    res = dict(tag=tag, k=k, mean_lag_cov_top=np.round(w[:top], 3).tolist(), mean_lag_cov_median_all=round(float(np.median(w)), 3), repro=np.round(rep, 2).tolist(),
               acf1=[round(acf(Y[:, j], 1), 2) for j in range(top)], acf16=[round(acf(Y[:, j], 16), 2) for j in range(top)],
               eta2_y2=[round(eta2(Y[:, j] ** 2), 3) for j in range(top)], eta2_y=[round(eta2(Y[:, j]), 3) for j in range(top)])
    best = {}
    for lb, y in labels.items():
        a = np.array([roc_auc_score(y, Y[:, j]) for j in range(top)]); a = np.maximum(a, 1 - a); best[lb] = (round(float(a.max()), 3), int(a.argmax()))
    res["label_best"] = {k_: v for k_, v in sorted(best.items(), key=lambda kv: -kv[1][0])[:8]}
    if planted_u is not None:
        uz = (planted_u @ Up) * sd; uz /= np.linalg.norm(uz)
        res["planted_dir_max_cos_top"] = round(float(np.abs(U[:, :top].T @ (np.linalg.inv(np.eye(k)) @ uz)).max()), 3)
    print(json.dumps(res), f"{time.time()-t0:.0f}s", flush=True)
    return res

from scipy.stats import spearmanr
EN = np.linalg.norm(E, axis=1)
sp_ = np.zeros(n); lp_ = np.zeros(n); sinceP = np.zeros(n); sentlen = np.zeros(n)
for s0 in range(0, n, 255):
    a = b = c = 0
    for j in range(255):
        r = s0 + j; st = S[int(toks_full[r])]; sp_[r] = a; lp_[r] = b; sinceP[r] = c
        a = 0 if any(ch in st for ch in ".!?") else a + 1; b = 0 if "\n" in st else b + 1; c = 0 if any(not ch.isalnum() and not ch.isspace() for ch in st) else c + 1
cov = dict(sent_pos=sp_[rows], line_pos=lp_[rows], since_punct=sinceP[rows], log_hnorm=np.log(HN), log_enorm=np.log(EN), L0=L0, pos=pos.astype(float))
Ve = demean3(E); Z, Up, sd = whiten(Ve, 1151); w, U = amuse(Z); Y = Z @ U[:, :6]
for j in range(6):
    print(j, {k: round(float(spearmanr(Y[::2, j], v[::2]).statistic), 3) for k, v in cov.items()}, flush=True)
# demeaned covariates (same identity removal) for fairness
for k in ("log_hnorm", "log_enorm", "L0", "sent_pos", "line_pos"):
    v = demean3(cov[k][:, None].astype(np.float32))[:, 0]
    print("identity-removed", k, [round(float(spearmanr(Y[::2, j], v[::2]).statistic), 3) for j in range(6)], flush=True)
