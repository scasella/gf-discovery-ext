import json, numpy as np, sys
EXT = "/Users/scasella/Downloads/gf-discovery-ext/"
VOCAB_N = 262_144 + 16
c = np.fromfile(EXT + "results/corpus_tokens_pass1.i32", dtype=np.int32)
tokD = c[1000*255:2000*255].astype(np.int64)
r_all = np.arange(len(tokD)); r_all = r_all[(r_all % 255) != 254]
order = np.argsort(tokD[r_all], kind="stable")
tt = tokD[r_all][order]; nn = tokD[r_all + 1][order]
nxt = {}
for b in np.split(np.arange(len(tt)), np.flatnonzero(np.diff(tt)) + 1):
    u_, c_ = np.unique(nn[b], return_counts=True)
    nxt[int(tt[b[0]])] = (u_, c_ / c_.sum())
def signature(rows_D, mem_idx):
    r = rows_D[mem_idx]; r = r[(r % 255) != 254]
    obs = np.bincount(tokD[r + 1], minlength=VOCAB_N).astype(float)
    exp = np.zeros(VOCAB_N)
    types, cnt = np.unique(tokD[r], return_counts=True)
    for t, cc in zip(types, cnt):
        d = nxt.get(int(t))
        if d is not None: exp[d[0]] += cc * d[1]
    score = np.log((obs + 1) / (exp + 1)); score[obs < 5] = -np.inf
    return [int(x) for x in np.argsort(-score)[:20] if np.isfinite(score[x])], obs, exp
fr = json.load(open(EXT + "results/e65_frozen.json"))
L = int(sys.argv[1]) if len(sys.argv) > 1 else 12
z = np.load(EXT + f"results/occ_units_v2_L{L}.npz"); rows = z["row"].astype(np.int64)
m = np.load(EXT + f"results/e64_maps_L{L}.npz")
for u in ["logcosh/13", "skew/16"] if L == 12 else ["logcosh/12"]:
    kd, j = u.split("/"); j = int(j)
    y = z[f"Y_{kd}"][:, j].astype(np.float64); s = m[f"sign_{kd}"][j]
    k = int(np.ceil(0.01 * len(y))); mem = np.argsort(-s * y)[:k]
    T, obs, exp = signature(rows, mem)
    print(u, "match frozen:", T == fr["layers"][str(L)]["signatures"][u]["T"], "k", k)
