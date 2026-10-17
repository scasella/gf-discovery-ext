"""Read-only feasibility, part 5 (D2, L12 raw): how much of each unit's score lives in the SAE error
(h - SAE(h)), for block (PC 65-256) units vs top-64 units. L12 16k l0_big SAE read from hf_home. Scratchpad only."""
import json, sys, time
import numpy as np, scipy.sparse as sp
from safetensors.numpy import load_file
EXT = "/Users/scasella/Downloads/gf-discovery-ext"; SP = sys.argv[1]; t0 = time.time()
P = load_file(f"{EXT}/hf_home/hub/models--google--gemma-scope-2-1b-pt/snapshots/b738dc06961818c011fb2e44a316352ca0f4e873/resid_post_all/layer_12_width_16k_l0_big/params.safetensors")
We, be, th, Wd, bd = (P[k].astype(np.float32) for k in ("w_enc", "b_enc", "threshold", "w_dec", "b_dec"))
H = np.load(f"{EXT}/results/e67_raw_L12_rows.npy", mmap_mode="r")
n = H.shape[0]; dec = np.empty((n, 1152), np.float32); l0 = 0.0
for lo in range(0, n, 8192):
    x = np.asarray(H[lo:lo + 8192], dtype=np.float32)
    pre = x @ We + be; z = pre * (pre > th); l0 += float((z > 0).sum())
    dec[lo:lo + 8192] = z @ Wd + bd
print("sae", round(time.time() - t0), "L0", l0 / n, flush=True)
keep = np.ones(1152, bool); keep[1038] = False
V = np.asarray(H, dtype=np.float32)[:, keep]; Vd = dec[:, keep]; del dec
fvu = float(((V - Vd) ** 2).sum() / ((V - V.mean(0)) ** 2).sum())
aux = np.load(f"{SP}/lv_feas_L12_aux.npz")
groups = [np.unique(aux[k], return_inverse=True)[1] for k in ("tok", "seq", "pos")]
def demean(V, g):
    M = sp.csr_matrix((np.ones(len(g)), (g, np.arange(len(g)))), shape=(int(g.max()) + 1, len(g)))
    cnt = np.maximum(np.asarray(M.sum(1)).ravel(), 1)
    for j in range(0, V.shape[1], 128):
        b = V[:, j:j + 128].astype(np.float64); V[:, j:j + 128] = (b - (M @ b / cnt[:, None])[g]).astype(np.float32)
for _ in range(3):
    for g in groups:
        demean(V, g); demean(Vd, g)
Y = np.concatenate([np.load(f"{SP}/Y1.npy"), np.load(f"{SP}/Yb.npy")], 1).astype(np.float64)
G = V[::2].astype(np.float64)
F = np.linalg.solve(G.T @ G, G.T @ Y[::2])
Yh = V @ F; fit_r2 = 1 - ((Yh - Y) ** 2).sum(0) / ((Y - Y.mean(0)) ** 2).sum(0)
Yd = Vd @ F; Ye = Yh - Yd
r = np.array([np.corrcoef(Yh[:, j], Yd[:, j])[0, 1] for j in range(Y.shape[1])])
es = Ye.var(0) / Yh.var(0)
# demeaned-space variance share of the SAE error, total and per PC block (fixed basis from V)
w, U = np.linalg.eigh(np.cov(V[::3].T)); o = np.argsort(w)[::-1]; U = U[:, o]
E = V - Vd; tot = {}
for b, (s, e) in {"1-64": (0, 64), "65-256": (64, 256), "257-1151": (256, 1151)}.items():
    pv = ((V[::3] @ U[:, s:e]) ** 2).sum(); pe = ((E[::3] @ U[:, s:e]) ** 2).sum()
    tot[b] = float(pe / pv)
named_read = np.load(f"{SP}/named_read.npy"); named = named_read[:, 0] > 0.5; read = named_read[:, 1]
cand = (~named) & (read < 0.70)
q = lambda x: [round(float(v), 3) for v in np.quantile(x, [0.1, 0.25, 0.5, 0.75, 0.9])]
res = dict(sae_L0=l0 / n, sae_fvu_raw=fvu, filter_fit_r2_min=float(fit_r2.min()),
           err_var_share_by_pc_block=tot,
           top64_units=dict(r_dec_q=q(r[:64]), err_share_q=q(es[:64])),
           block_units=dict(r_dec_q=q(r[64:]), err_share_q=q(es[64:]), n_err_share_ge_05=int((es[64:] >= 0.5).sum()),
                            n_r_dec_lt_07=int((r[64:] < 0.7).sum())),
           block_unnamed_nonshadow=dict(n=int(cand.sum()), err_share_q=q(es[64:][cand]), n_err_ge_05=int((es[64:][cand] >= 0.5).sum())),
           controls=dict(paren146=dict(r=float(r[64 + 146]), err=float(es[64 + 146])), quote128=dict(r=float(r[64 + 128]), err=float(es[64 + 128]))))
print(json.dumps(res, indent=1)); json.dump(res, open(f"{SP}/lv_feas5_L12.json", "w"), indent=1)
print("done", round(time.time() - t0))
