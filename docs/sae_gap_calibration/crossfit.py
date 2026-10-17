import json, numpy as np
EXT = "/Users/scasella/Downloads/gf-discovery-ext/"
VOCAB_N = 262_144 + 16
c = np.fromfile(EXT + "results/corpus_tokens_pass1.i32", dtype=np.int32)
tokD = c[1000*255:2000*255].astype(np.int64)
def nxt_tab(mask_seq):
    r_all = np.arange(len(tokD)); r_all = r_all[((r_all % 255) != 254) & mask_seq[r_all // 255]]
    order = np.argsort(tokD[r_all], kind="stable"); tt = tokD[r_all][order]; nn = tokD[r_all + 1][order]
    out = {}
    for b in np.split(np.arange(len(tt)), np.flatnonzero(np.diff(tt)) + 1):
        u_, c_ = np.unique(nn[b], return_counts=True); out[int(tt[b[0]])] = (u_, c_ / c_.sum())
    return out
def signature(r, nxt):
    r = r[(r % 255) != 254]
    obs = np.bincount(tokD[r + 1], minlength=VOCAB_N).astype(float); exp = np.zeros(VOCAB_N)
    types, cnt = np.unique(tokD[r], return_counts=True)
    for t, cc in zip(types, cnt):
        d = nxt.get(int(t))
        if d is not None: exp[d[0]] += cc * d[1]
    score = np.log((obs + 1) / (exp + 1)); score[obs < 5] = -np.inf
    return [int(x) for x in np.argsort(-score)[:20] if np.isfinite(score[x])]
seq_even = (np.arange(1000) % 2 == 0)
nxtE = nxt_tab(seq_even)
st = json.load(open("/private/tmp/claude-501/-Users-scasella-Downloads/db01a4e8-8f9e-44d9-a3d5-6be3160de0fb/scratchpad/match_stats.json"))
for L in ["7", "12"]:
    z = np.load(EXT + f"results/occ_units_v2_L{L}.npz"); rows = z["row"].astype(np.int64)
    m = np.load(EXT + f"results/e64_maps_L{L}.npz")
    valid = (rows % 255) != 254; ev = seq_even[rows // 255]
    nx = np.full(len(rows), -1); nx[valid] = tokD[rows[valid] + 1]
    ins, outs, ins200, outs200, baseO = [], [], [], [], []
    for u, s in st[L].items():
        if s["cls"] != "cand": continue
        kd, j = u.split("/"); j = int(j)
        y = z[f"Y_{kd}"][:, j]; sg = m[f"sign_{kd}"][j]
        o = np.argsort(-sg * y)
        oE = o[ev[o] & valid[o]]; oO = o[(~ev[o]) & valid[o]]
        k = 1017
        T = np.array(signature(rows[oE[:k]], nxtE))
        hit = np.isin(nx, T)
        ins.append(hit[oE[:k]].mean()); outs.append(hit[oO[:k]].mean())
        ins200.append(hit[oE[:200]].mean()); outs200.append(hit[oO[:200]].mean())
        baseO.append(hit[(~ev) & valid].mean())
    ins, outs, ins200, outs200, baseO = map(np.array, (ins, outs, ins200, outs200, baseO))
    print(f"L{L} cand n={len(ins)}: T built on even-seq tail. tail share in-sample med {np.median(ins):.3f}, held-out (odd-seq tail) med {np.median(outs):.3f}; top-200 in {np.median(ins200):.3f} vs held-out {np.median(outs200):.3f}; held-out base med {np.median(baseO):.3f}; median shrink ratio {np.median(outs/ins):.2f}")
