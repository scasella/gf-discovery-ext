import json, numpy as np
EXT = "/Users/scasella/Downloads/gf-discovery-ext/"
c = np.fromfile(EXT + "results/corpus_tokens_pass1.i32", dtype=np.int32)
tokD = c[1000*255:2000*255].astype(np.int64)
fr = json.load(open(EXT + "results/e65_frozen.json"))
st = json.load(open("/private/tmp/claude-501/-Users-scasella-Downloads/db01a4e8-8f9e-44d9-a3d5-6be3160de0fb/scratchpad/match_stats.json"))
# type-conditional next-token share over all D2 non-final positions (as in e64.signature)
r_all = np.arange(len(tokD)); r_all = r_all[(r_all % 255) != 254]
cur_all = tokD[r_all]; nx_all = tokD[r_all + 1]
for L in ["7", "12"]:
    z = np.load(EXT + f"results/occ_units_v2_L{L}.npz"); rows = z["row"].astype(np.int64)
    m = np.load(EXT + f"results/e64_maps_L{L}.npz")
    valid = (rows % 255) != 254
    out = []
    for u, s in st[L].items():
        if s["cls"] != "cand": continue
        T = np.array(fr["layers"][L]["signatures"][u]["T"])
        hit_all = np.isin(nx_all, T).astype(float)
        types, inv, cnt = np.unique(cur_all, return_inverse=True, return_counts=True)
        p_type = np.bincount(inv, weights=hit_all) / cnt
        lut = dict(zip(types.tolist(), p_type.tolist()))
        kd, j = u.split("/"); j = int(j)
        y = z[f"Y_{kd}"][:, j]; sg = m[f"sign_{kd}"][j]
        o = np.argsort(-sg * y); o = o[valid[o]][:200]
        e200 = np.mean([lut[int(t)] for t in tokD[rows[o]]])
        out.append((u, s["base"], e200, s["own200"], s["thr"]))
    a = np.array([x[1:] for x in out])
    print(f"L{L}: marginal base med {np.median(a[:,0]):.3f}; type-expected share in own top200 med {np.median(a[:,1]):.3f} (range {a[:,1].min():.3f}-{a[:,1].max():.3f}); own200 med {np.median(a[:,2]):.3f}; own200/type-expected med {np.median(a[:,2]/a[:,1]):.2f}; own200/base med {np.median(a[:,2]/a[:,0]):.2f}; units with type-expected >= thr: {(a[:,1]>=a[:,3]).sum()}")
