"""Independent check (burned D2/S2 only) of the design review's claims: error share of the fixed filters under 16k big,
and how often random filter-type error-space directions pass the pilot's GENERIC-b ratios. Batch projections only."""
import importlib.util, sys, json
import numpy as np, torch
EXT = "/Users/scasella/Downloads/gf-discovery-ext"
spec = importlib.util.spec_from_file_location("e69", f"{EXT}/scripts/e69_sae_gap_pilot.py")
e69 = importlib.util.module_from_spec(spec); sys.modules["e69"] = e69; spec.loader.exec_module(e69)
z = np.load(f"{EXT}/results/e69_p0_units.npz")
U, sd = z["U"], z["sd"]
g = np.random.default_rng(123).standard_normal((256, 100))
F = np.concatenate([z["filt"][:, :3], U @ (g / sd[:, None])], 1).astype(np.float32)      # units 0,1,2 + 100 random filters
P = e69.sae_params(e69.SAE_BIG)
We, be, th, Wd, bd = [torch.from_numpy(P[k].astype(np.float32)) for k in ("w_enc", "b_enc", "threshold", "w_dec", "b_dec")]
kp = torch.from_numpy(e69.KEEP); Ft = torch.from_numpy(F)
def proj(H16):
    pe, ph = [], []
    with torch.no_grad():
        for a in range(0, len(H16), 8192):
            x = torch.from_numpy(H16[a:a + 8192].astype(np.float32)); pre = x @ We + be; zz = pre * (pre > th)
            e = (x - zz @ Wd - bd)[:, kp]; pe.append((e @ Ft).numpy()); ph.append((x[:, kp] @ Ft).numpy())
    return np.concatenate(pe), np.concatenate(ph)
mD = np.load(f"{EXT}/cache/occ/occD2_meta.npz"); mS = np.load(f"{EXT}/cache/occ/occS2_meta.npz")
D = e69.RowSet(mD, np.load(f"{EXT}/results/occ_units_v2_L12.npz")["row"]); S = e69.RowSet(mS)
peD, phD = proj(np.load(f"{EXT}/results/e67_raw_L12_rows.npy"))
peS, _ = proj(np.load(f"{EXT}/results/e69_p1_S2_raw_L12_rows.npy"))
yD = D.demean3(peD); yS = S.demean3(peS); yR = D.demean3(phD)
res = []
for j in range(F.shape[1]):
    es = float(np.cov(yD[:, j], yR[:, j])[0, 1] / np.var(yR[:, j], ddof=1))
    ar = S.acf(yS[:, j], 1) / D.acf(yD[:, j], 1); vr = S.acov(yS[:, j], 1) / D.acov(yD[:, j], 1)
    res.append(dict(acf1=D.acf(yD[:, j], 1), acf_ratio=ar, acov_ratio=vr, es=es, raw_acf1=D.acf(yR[:, j], 1)))
for j in range(3):
    print(f"unit {j}: acf1 {res[j]['acf1']:.3f}  acf ratio {res[j]['acf_ratio']:.3f}  acov ratio {res[j]['acov_ratio']:.3f}  ES(big) {res[j]['es']:.3f}  raw acf1 {res[j]['raw_acf1']:.3f}")
r = res[3:]
q = lambda k: np.quantile([x[k] for x in r], [0.05, 0.5, 0.95]).round(3).tolist()
print("random filters (100): acf1 p5/50/95", q("acf1"), " acf ratio", q("acf_ratio"), " acov ratio", q("acov_ratio"), " ES", q("es"))
print("random: share passing G1 (acf1>=0.2)", np.mean([x["acf1"] >= 0.2 for x in r]), "| share passing GENERIC-b (both ratios < 0.5)",
      np.mean([x["acf_ratio"] < 0.5 and x["acov_ratio"] < 0.5 for x in r]), "| max ES", round(max(x["es"] for x in r), 3))
json.dump(res, open(sys.argv[1], "w"))
