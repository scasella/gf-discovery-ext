"""Read-only feasibility check on D2 / L12 (already-used data): raw residual vs SAE-decoded vs SAE error.
Writes nothing outside the scratchpad. Inputs: results/e67_raw_L12_rows.npy (D2 kept rows, fp16 raw L12),
results/occ_units_v2_L12.npz (rows), cache/occ/occD2_meta.npz (tokens), results/e64_maps_L12.npz (frozen v2 maps),
Gemma Scope 2 L12 16k l0_big params (hf_home, read-only)."""
import json, sys, time
from pathlib import Path
import numpy as np
import scipy.sparse as sp
from safetensors.numpy import load_file

EXT = Path("/Users/scasella/Downloads/gf-discovery-ext")
OUT = Path(sys.argv[1])
t0 = time.time()
res = {}
rows = np.load(EXT / "results/occ_units_v2_L12.npz")["row"].astype(np.int64)
meta = np.load(EXT / "cache/occ/occD2_meta.npz")
full = meta["tokens"].astype(np.int64)
tok, seq, pos = full[rows], meta["seq"][rows], meta["pos"][rows]
H = np.load(EXT / "results/e67_raw_L12_rows.npy", mmap_mode="r")
assert H.shape[0] == len(rows)
H = np.asarray(H, dtype=np.float32)
keep = np.ones(1152, bool); keep[1038] = False
# --- A. coordinate stats (raw, not identity-removed)
amax = np.abs(H).max(0)
res["raw_absmax_1038"] = float(amax[1038])
o = np.argsort(-amax)
res["raw_absmax_top10"] = [(int(i), float(amax[i])) for i in o[:10]]
res["raw_absmax_median_coord"] = float(np.median(amax))
res["n_coords_absmax_gt_2048_excl1038"] = int((amax[keep] > 2048).sum())
print("A", time.time() - t0, flush=True)
# --- B. SAE encode
sae = load_file(str(next((EXT / "hf_home/hub/models--google--gemma-scope-2-1b-pt/snapshots").glob("*"))
                    / "resid_post_all/layer_12_width_16k_l0_big/params.safetensors"))
We, be, th, Wd, bd = (sae[k].astype(np.float32) for k in ("w_enc", "b_enc", "threshold", "w_dec", "b_dec"))
D = np.empty_like(H)
L0 = np.empty(len(H), np.float32)
for lo in range(0, len(H), 8192):
    x = H[lo:lo + 8192]
    pre = x @ We + be
    z = pre * (pre > th)
    L0[lo:lo + 8192] = (z > 0).sum(1)
    D[lo:lo + 8192] = z @ Wd + bd
E = H - D
res["mean_L0"] = float(L0.mean())
res["fvu_raw_total_incl_mean"] = float((E ** 2).sum() / ((H - H.mean(0)) ** 2).sum())
res["fvu_raw_keptcoords"] = float((E[:, keep] ** 2).sum() / ((H[:, keep] - H[:, keep].mean(0)) ** 2).sum())
en = np.linalg.norm(E[:, keep], axis=1); hn = np.linalg.norm(H[:, keep], axis=1)
print("B", time.time() - t0, flush=True)
# --- C. identity removal (type, doc, pos x3) on kept coords
def demean(V, g, block=128):
    n, G = len(g), int(g.max()) + 1
    M = sp.csr_matrix((np.ones(n), (g, np.arange(n))), shape=(G, n))
    cnt = np.maximum(np.asarray(M.sum(1)).ravel(), 1)
    for j in range(0, V.shape[1], block):
        b = V[:, j:j + block].astype(np.float64)
        V[:, j:j + block] = (b - (M @ b / cnt[:, None])[g]).astype(np.float32)
groups = [np.unique(a, return_inverse=True)[1] for a in (tok, seq, pos)]
R = np.ascontiguousarray(H[:, keep]); del H
Dk = np.ascontiguousarray(D[:, keep]); del D, E
for _ in range(3):
    for g in groups:
        demean(R, g); demean(Dk, g)
Ek = R - Dk
tvR = float((R.astype(np.float64) ** 2).sum()); tvE = float((Ek.astype(np.float64) ** 2).sum())
cvRE = float((R.astype(np.float64) * Ek).sum())
res["idrem_var_share_err"] = tvE / tvR
res["idrem_cov_attrib_err"] = cvRE / tvR
res["idrem_var_share_dec"] = float((Dk.astype(np.float64) ** 2).sum()) / tvR
cv = (R ** 2).mean(0); oc = np.argsort(-cv)
res["idrem_coordvar_top5_share"] = [(int(np.flatnonzero(keep)[i]), float(cv[i] / cv.sum())) for i in oc[:5]]
res["idrem_coordvar_max_over_median"] = float(cv.max() / np.median(cv))
k4 = ((R - R.mean(0)) ** 4).mean(0) / cv ** 2 - 3
ok = np.argsort(-k4)
res["idrem_coord_exkurt_top5"] = [(int(np.flatnonzero(keep)[i]), float(k4[i])) for i in ok[:5]]
res["idrem_coord_exkurt_median"] = float(np.median(k4))
# fp16 quantisation vs idrem SD
res["fp16_spacing_vs_idrem_sd_median"] = float(np.median((amax[keep] / 1024 / np.sqrt(12)) / np.sqrt(cv)))
res["fp16_spacing_vs_idrem_sd_max"] = float(np.max((amax[keep] / 1024 / np.sqrt(12)) / np.sqrt(cv)))
print("C", time.time() - t0, res["idrem_var_share_err"], flush=True)
# --- E. PCA of raw idrem; error share by PC band
sub = R[::3].astype(np.float64)
C = sub.T @ sub / len(sub)
w, U = np.linalg.eigh(C); o = np.argsort(w)[::-1]; w, U = w[o], U[:, o]
res["raw_pca_varfrac_cum"] = {k: float(w[:k].sum() / w.sum()) for k in (16, 32, 64, 128, 256, 512)}
CE = (Ek[::3].astype(np.float64).T @ sub) / len(sub)      # Cov(err, raw)
CEE = (Ek[::3].astype(np.float64).T @ Ek[::3].astype(np.float64)) / len(sub)
def band(a, b):
    Ub = U[:, a:b]
    num = np.trace(Ub.T @ CE @ Ub); den = w[a:b].sum()
    return dict(cov_attrib_err=float(num / den), var_share_err=float(np.trace(Ub.T @ CEE @ Ub) / den))
res["err_share_by_raw_pc_band"] = {f"{a}-{b}": band(a, b) for a, b in ((0, 16), (16, 64), (0, 64), (64, 128), (128, 256), (256, 512), (512, 1151))}
# per-PC ES = 1 - corr(y, y_dec)^2 for top 64 PCs
CD = (Dk[::3].astype(np.float64).T @ Dk[::3].astype(np.float64)) / len(sub)
CDR = (Dk[::3].astype(np.float64).T @ sub) / len(sub)
es_pc = []
for j in range(256):
    u = U[:, j]
    cr = u @ CDR @ u; vd = u @ CD @ u
    es_pc.append(float(1 - cr ** 2 / (w[j] * vd)))
es_pc = np.array(es_pc)
res["ES_per_pc_median"] = {"0-64": float(np.median(es_pc[:64])), "64-128": float(np.median(es_pc[64:128])), "128-256": float(np.median(es_pc[128:]))}
res["ES_per_pc_n_ge_0.5"] = {"0-64": int((es_pc[:64] >= 0.5).sum()), "64-128": int((es_pc[64:128] >= 0.5).sum()), "128-256": int((es_pc[128:] >= 0.5).sum())}
print("E", time.time() - t0, flush=True)
# --- G. random directions in raw whitened PCA-64: ES distribution
U64 = U[:, :64]; s64 = np.sqrt(w[:64])
rng = np.random.default_rng(0)
G = rng.standard_normal((64, 1000))
Vd = U64 @ (G / s64[:, None])            # filter in raw coords for unit-variance whitened readout
num = np.einsum("ij,ik,kj->j", Vd, CDR, Vd); vr = np.einsum("ij,ik,kj->j", Vd, C, Vd); vd = np.einsum("ij,ik,kj->j", Vd, CD, Vd)
esr = 1 - num ** 2 / (vr * vd)
res["ES_random_whitened64"] = {q: float(np.percentile(esr, q)) for q in (1, 50, 99, 99.9)}
# random directions isotropic in the full kept raw space
G2 = rng.standard_normal((1151, 1000))
num = np.einsum("ij,ik,kj->j", G2, CDR, G2); vr = np.einsum("ij,ik,kj->j", G2, C, G2); vd = np.einsum("ij,ik,kj->j", G2, CD, G2)
esr2 = 1 - num ** 2 / (vr * vd)
res["ES_random_isotropic_raw"] = {q: float(np.percentile(esr2, q)) for q in (1, 50, 99, 99.9)}
print("G", time.time() - t0, flush=True)
# --- F. frozen v2 units (decoded-derived) evaluated on raw / dec parts via frozen maps
z = np.load(EXT / "results/e64_maps_L12.npz")
v2 = json.loads((EXT / "results/e63_occurrence_v2.json").read_text())["layers"]["12"]["frozen"]["units"]
scr = {(u["kind"].replace("ica_", ""), u["id"]) for u in v2 if u["kind"].startswith("ica") and u["screened"]}
def apply_maps(Zp):
    utok, tinv = np.unique(tok, return_inverse=True)
    n_t = np.bincount(tinv)
    Zp = Zp.astype(np.float64)
    mu = np.stack([np.bincount(tinv, weights=Zp[:, j], minlength=len(n_t)) for j in range(Zp.shape[1])], 1)
    Zc = Zp - (mu / n_t[:, None])[tinv]
    out = Zc @ z["Sp_is"]
    posd = {int(t): i for i, t in enumerate(z["type_ids"])}
    order = np.argsort(tinv, kind="stable"); start = np.cumsum(n_t) - n_t
    for t in range(len(n_t)):
        i = posd.get(int(utok[t]))
        if i is not None:
            ii = order[start[t]:start[t] + n_t[t]]
            out[ii] = Zc[ii] @ z["type_isqrt"][i]
    return out @ z["Uo"] / z["sw"]
proj = lambda V: (V.astype(np.float64)) @ z["P_U"] / z["P_sd"]     # mean term cancels in type demeaning
Wr, Wd_ = apply_maps(proj(R)), apply_maps(proj(Dk))
es_units = {}
for kd in ("logcosh", "skew"):
    B = z[f"readout_{kd}"][:64]
    Yr, Yd = Wr @ B, Wd_ @ B
    for j in range(64):
        c = np.corrcoef(Yr[:, j], Yd[:, j])[0, 1]
        es_units[f"{kd}/{j}"] = dict(ES=float(1 - c ** 2), screened=(kd, j) in scr,
                                     var_ratio_dec_over_raw=float(Yd[:, j].var() / Yr[:, j].var()))
esv = np.array([v["ES"] for v in es_units.values() if v["screened"]])
res["v2_units_on_raw_ES"] = dict(n=int(len(esv)), median=float(np.median(esv)), max=float(esv.max()), min=float(esv.min()))
res["v2_named_units_ES"] = {k: es_units[k] for k in ("logcosh/13", "skew/16", "logcosh/33", "skew/33", "logcosh/11", "skew/23", "logcosh/10", "skew/15")}
# whitened-space random baseline in the v2 frame
Gw = rng.standard_normal((64, 1000))
Yr, Yd = Wr @ Gw, Wd_ @ Gw
esw = np.array([1 - np.corrcoef(Yr[:, j], Yd[:, j])[0, 1] ** 2 for j in range(1000)])
res["ES_random_in_v2_frame"] = {q: float(np.percentile(esw, q)) for q in (1, 50, 99, 99.9)}
del Wr, Wd_
print("F", time.time() - t0, flush=True)
# --- H. paren / quote labels and probes on raw / dec / err reps (B1 rep: PCA-64 whitened, own fit)
from tokenizers import Tokenizer
tk = Tokenizer.from_file(str(next((EXT / "hf_home/hub/models--google--gemma-3-1b-pt/snapshots").glob("*")) / "tokenizer.json"))
strs = {}
s_of = lambda t: strs.setdefault(t, tk.decode([int(t)]))
row_set = {int(r): i for i, r in enumerate(rows)}
par = np.zeros(len(rows)); quo = np.zeros(len(rows))
for s0 in range(0, len(full), 255):
    depth = 0; q = 0
    for j, t in enumerate(full[s0:s0 + 255]):
        i = row_set.get(s0 + j)
        st = s_of(int(t))
        if i is not None:
            par[i] = depth > 0; quo[i] = q % 2 == 1
        depth = max(0, depth + st.count("(") - st.count(")")); q += st.count('"') + st.count("“") + st.count("”")
res["base_rates"] = dict(paren=float(par.mean()), quote=float(quo.mean()))
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
ev = seq % 2 == 0
def white64(V):
    s = V[::3].astype(np.float64); mu = s.mean(0)
    w_, U_ = np.linalg.eigh(np.cov(s.T)); o_ = np.argsort(w_)[::-1][:64]
    return ((V - mu) @ U_[:, o_] / np.sqrt(w_[o_])).astype(np.float32)
probes = {}
for name, V in (("raw", R), ("dec", Dk), ("err", Ek)):
    W = white64(V)
    for lb, y in (("paren", par), ("quote", quo)):
        clf = LogisticRegression(C=1.0, max_iter=3000).fit(W[ev], y[ev] > 0.5)
        probes[f"{name}64/{lb}"] = float(roc_auc_score(y[~ev] > 0.5, clf.decision_function(W[~ev])))
    print("H", name, time.time() - t0, flush=True)
# full-dim probe on err (all 1151 coords, standardised) for the error-tier control question
for name, V in (("err", Ek), ("dec", Dk)):
    s = V[ev]; mu = s.mean(0); sd = s.std(0) + 1e-6
    for lb, y in (("paren", par), ("quote", quo)):
        clf = LogisticRegression(C=0.1, max_iter=2000).fit(((V[ev] - mu) / sd)[::2], (y[ev] > 0.5)[::2])
        probes[f"{name}full/{lb}"] = float(roc_auc_score(y[~ev] > 0.5, clf.decision_function((V[~ev] - mu) / sd)))
    print("H full", name, time.time() - t0, flush=True)
res["probes_heldout_auroc"] = probes
res["norm_corr"] = dict(r_errnorm_hnorm=float(np.corrcoef(en, hn)[0, 1]), r_errnorm_L0=float(np.corrcoef(en, L0)[0, 1]))
res["seconds"] = round(time.time() - t0)
OUT.write_text(json.dumps(res, indent=1))
print(json.dumps(res, indent=1))
