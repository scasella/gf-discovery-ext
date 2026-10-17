# read-only: which known in-context variables are present in the L12 16k-big SAE error on D2 (burned)?
import sys, time, json, numpy as np, torch
from pathlib import Path
from safetensors.numpy import load_file
import scipy.sparse as sp
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.decomposition import FastICA
EXT = Path("/Users/scasella/Downloads/gf-discovery-ext"); t0 = time.time()
fz = np.load(EXT / "results/occ_units_v2_L12.npz"); rows = fz["row"].astype(np.int64)
m = np.load(EXT / "cache/occ/occD2_meta.npz"); toks_full = m["tokens"].astype(np.int64)
tok, seq, pos = toks_full[rows], m["seq"].astype(np.int64)[rows], m["pos"].astype(np.int64)[rows]
H = np.load(EXT / "results/e67_raw_L12_rows.npy").astype(np.float32)
snap = sorted((EXT / "hf_home/hub/models--google--gemma-scope-2-1b-pt/snapshots").glob("*"))[0]
P = load_file(str(snap / "resid_post_all/layer_12_width_16k_l0_big/params.safetensors"))
We, be, th, Wd, bd = [torch.from_numpy(P[k]) for k in ("w_enc", "b_enc", "threshold", "w_dec", "b_dec")]
E = np.empty_like(H); D = np.empty_like(H)
for a in range(0, len(H), 8192):
    x = torch.from_numpy(H[a:a + 8192]); pre = x @ We + be; z = pre * (pre > th)
    D[a:a + 8192] = (z @ Wd).numpy(); E[a:a + 8192] = (x - z @ Wd - bd).numpy()
from transformers import AutoTokenizer
tk = AutoTokenizer.from_pretrained(str(sorted((EXT / "hf_home/hub/models--google--gemma-3-1b-pt/snapshots").glob("*"))[0]))
S = {}
def s_of(t):
    if t not in S: S[t] = tk.decode([int(t)])
    return S[t]
n = len(toks_full); F = {k: np.zeros(n, bool) for k in ("paren", "quote", "after_newline", "next_newline", "next_punct", "prev_punct", "word_continuation", "next_continuation", "after_digit", "bracket", "after_colon")}
for s0 in range(0, n, 255):
    d = b = q = 0; prev = ""
    for j in range(255):
        r = s0 + j; st = s_of(toks_full[r]); nx = s_of(toks_full[r + 1]) if j < 254 else ""
        F["paren"][r] = d > 0; F["quote"][r] = q % 2 == 1; F["bracket"][r] = b > 0
        F["after_newline"][r] = "\n" in prev; F["next_newline"][r] = "\n" in nx
        F["next_punct"][r] = bool(nx.strip()) and not any(c.isalnum() for c in nx.strip())
        ps = prev.strip(); F["prev_punct"][r] = bool(ps) and not any(c.isalnum() for c in ps)
        F["word_continuation"][r] = bool(st) and st[0].isalpha() and j > 0 and not prev.endswith((" ", "\n"))
        F["next_continuation"][r] = bool(nx) and nx[0].isalpha()
        F["after_digit"][r] = ps[-1:].isdigit(); F["after_colon"][r] = prev.rstrip().endswith(":")
        d = max(0, d + st.count("(") - st.count(")")); b = max(0, b + st.count("[") - st.count("]"))
        q += st.count('"') + st.count("“") + st.count("”"); prev = st
R = np.load(EXT / "results/e64c_rules_en_core_web_trf.npz")
names = {"A": "after_cc", "B": "temporal_clause", "C": "after_prep", "D": "partitive", "E": "second_conj", "F": "subject_np",
         "G": "fronted_adv", "H": "coord_onset", "I": "line_initial", "J": "negation", "K": "copular_pred", "L": "enumeration", "M": "temporal_expr", "N": "after_the"}
labels = {k: v[rows] for k, v in F.items()}
labels.update({f"{c}_{names[c]}": R[c][rows].astype(bool) for c in names})
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
def pca(V):
    mu = V.mean(0); C = np.cov((V - mu)[::3].T); w, U = np.linalg.eigh(C); o = np.argsort(w)[::-1]
    return mu, U[:, o], np.sqrt(np.maximum(w[o], 1e-12))
def inv_sqrt(S_):
    w, U = np.linalg.eigh(S_); w = np.maximum(w, 1e-6 * w.max()); return (U / np.sqrt(w)) @ U.T
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
evn = seq % 2 == 0
def probe(W, y):
    clf = LogisticRegression(C=1.0, max_iter=3000).fit(W[evn], y[evn])
    return round(float(roc_auc_score(y[~evn], clf.decision_function(W[~evn]))), 3)
keep = np.ones(1152, bool); keep[1038] = False
out = {}
for name, X in (("dec", D), ("err", E)):
    V = demean3(X[:, keep]); mu, U, sd = pca(V)
    Z64 = ((V - mu) @ U[:, :64]) / sd[:64]; Wt = type_whiten(Z64); Z256 = ((V - mu) @ U[:, :256]) / sd[:256]
    ica = FastICA(n_components=64, whiten="unit-variance", random_state=0, max_iter=400).fit(Wt[::2]); Y = ica.transform(Wt)
    for lb, y in labels.items():
        if y.mean() < 0.002: continue
        a = np.array([roc_auc_score(y, Y[:, j]) for j in range(64)]); a = np.maximum(a, 1 - a)
        out.setdefault(lb, dict(base=round(float(y.mean()), 4)))[name] = dict(v2rep=probe(Wt, y), pca256=probe(Z256, y), ica_max=round(float(a.max()), 3))
    print(name, f"{time.time()-t0:.0f}s", flush=True)
for lb, r in sorted(out.items(), key=lambda kv: -kv[1]["err"]["ica_max"]):
    print(f"{lb:22s} base {r['base']:.4f} | dec v2 {r['dec']['v2rep']:.3f} p256 {r['dec']['pca256']:.3f} ica {r['dec']['ica_max']:.3f} | err v2 {r['err']['v2rep']:.3f} p256 {r['err']['pca256']:.3f} ica {r['err']['ica_max']:.3f}")
