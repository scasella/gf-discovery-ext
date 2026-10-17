import importlib.util, sys, numpy as np, torch
from pathlib import Path
EXT = Path.home() / "Downloads/gf-discovery-ext"
spec = importlib.util.spec_from_file_location("e71", EXT / "scripts/e71_family_readout.py"); e71 = importlib.util.module_from_spec(spec); spec.loader.exec_module(e71)
e64, e67, e69 = e71.mods()
prep, sc, rows = e71.l12_inputs(); rk = np.load(EXT / e71.OUT["rank"]); names = [str(x) for x in rk["names"]]
H = np.load(EXT / "results/e67_raw_L12_rows.npy").astype(np.float32); even = (rows // 255) % 2 == 0
toks = np.load(EXT / "cache/occ/occD2_meta.npz")["tokens"].astype(np.int64); _, tinv = np.unique(toks[rows], return_inverse=True); cnt = np.bincount(tinv)
P = e67.sae_params(e71.SMALL); nm = "logcosh/11"; top = rk["top"][names.index(nm)]
We = torch.from_numpy(P["w_enc"][:, top]); be = torch.from_numpy(P["b_enc"][top]); th = torch.from_numpy(P["threshold"][top])
pre = torch.from_numpy(H) @ We + be; a = pre * (pre > th); s = torch.minimum(torch.clamp(pre, min=0), th)
X = torch.cat([a, s], 1).numpy().astype(np.float64)
for j in range(X.shape[1]): X[:, j] -= (np.bincount(tinv, weights=X[:, j]) / cnt)[tinv]
sd_e = X[even].std(0); Xz = (X - X[even].mean(0)) / np.maximum(sd_e, 1e-9)
mx_odd = np.abs(Xz[~even]).max(0); mx_even = np.abs(Xz[even]).max(0)
o = np.argsort(-mx_odd)[:5]
print("worst columns (index, latent rank, kind, even sd, max|z| even, max|z| odd):")
for j in o: print(j, j % 64, "a" if j < 64 else "s", round(float(sd_e[j]), 5), round(float(mx_even[j]), 1), round(float(mx_odd[j]), 1))
for K in (32, 48, 64):
    cols = list(range(K)) + list(range(64, 64 + K)); print(K, round(e69.heldout_r2(Xz[:, cols], sc[nm], even, "ridge"), 3))
bad = [j for j in o if mx_odd[j] > 50]; keep = [j for j in range(128) if j not in bad]
print("K=64 without", bad, ":", round(e69.heldout_r2(Xz[:, keep], sc[nm], even, "ridge"), 3))
