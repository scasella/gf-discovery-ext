import json, numpy as np
EXT = "/Users/scasella/Downloads/gf-discovery-ext/"
c = np.fromfile(EXT + "results/corpus_tokens_pass1.i32", dtype=np.int32)
tokD = c[1000*255:2000*255].astype(np.int64)
fr = json.load(open(EXT + "results/e65_frozen.json"))
st = json.load(open("/private/tmp/claude-501/-Users-scasella-Downloads/db01a4e8-8f9e-44d9-a3d5-6be3160de0fb/scratchpad/match_stats.json"))
for L in ["7", "12"]:
    z = np.load(EXT + f"results/occ_units_v2_L{L}.npz"); rows = z["row"].astype(np.int64)
    valid = (rows % 255) != 254
    r = rows[valid]; cur = tokD[r]; nx = tokD[r + 1]
    types, inv, cnt = np.unique(cur, return_inverse=True, return_counts=True)
    big = cnt >= 200
    fr_pass = []; wfr = []; nfreqT = []
    for u, s in st[L].items():
        if s["cls"] != "cand": continue
        T = np.array(fr["layers"][L]["signatures"][u]["T"])
        hit = np.isin(nx, T).astype(float)
        share = np.bincount(inv, weights=hit) / cnt
        p = share[big] >= s["thr"]
        fr_pass.append(p.mean()); wfr.append((cnt[big] * p).sum() / cnt[big].sum())
        tc = np.array([np.sum(nx == t) for t in T]); nfreqT.append(int((tc >= 200).sum()))
    fr_pass = np.array(fr_pass); wfr = np.array(wfr)
    print(f"L{L}: {big.sum()} current-token types with >=200 rows. share of types whose next-token share in T_u >= thr: median {np.median(fr_pass):.3f} max {fr_pass.max():.3f}; row-weighted median {np.median(wfr):.3f} max {wfr.max():.3f}; T_u tokens with >=200 D2 next-occurrences: median {np.median(nfreqT)} range {min(nfreqT)}-{max(nfreqT)}")
