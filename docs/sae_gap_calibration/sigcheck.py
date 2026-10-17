import json, numpy as np, sys
EXT = "/Users/scasella/Downloads/gf-discovery-ext"
VOCAB_N = 262_144 + 16
TAIL = 0.01
def next_by_type(tokD):
    nxt = {}
    r_all = np.arange(len(tokD)); r_all = r_all[(r_all % 255) != 254]
    order = np.argsort(tokD[r_all], kind="stable")
    tt = tokD[r_all][order]; nn = tokD[r_all + 1][order]
    for b in np.split(np.arange(len(tt)), np.flatnonzero(np.diff(tt)) + 1):
        u_, c_ = np.unique(nn[b], return_counts=True)
        nxt[int(tt[b[0]])] = (u_, c_ / c_.sum())
    return nxt
def signature(rows_D, mem_idx, tokD, nxt_by_type):
    r = rows_D[mem_idx]
    r = r[(r % 255) != 254]
    obs = np.bincount(tokD[r + 1], minlength=VOCAB_N).astype(float)
    exp = np.zeros(VOCAB_N)
    types, cnt = np.unique(tokD[r], return_counts=True)
    for t, c in zip(types, cnt):
        d = nxt_by_type.get(int(t))
        if d is not None:
            exp[d[0]] += c * d[1]
    score = np.log((obs + 1) / (exp + 1))
    score[obs < 5] = -np.inf
    return [int(x) for x in np.argsort(-score)[:20] if np.isfinite(score[x])]
def heavy_sign(y):
    return 1.0 if abs(np.quantile(y, 0.99)) >= abs(np.quantile(y, 0.01)) else -1.0
toks = np.load(f"{EXT}/cache/occ/occD2_meta.npz")["tokens"].astype(np.int64)
nt = next_by_type(toks)
f65 = json.load(open(f"{EXT}/results/e65_frozen.json"))
for L in (7, 12):
    fz = np.load(f"{EXT}/results/occ_units_v2_L{L}.npz"); rows = fz["row"].astype(np.int64)
    z = np.load(f"{EXT}/results/e64_maps_L{L}.npz")
    sig = f65["layers"][str(L)]["signatures"]
    n = len(rows); k = int(np.ceil(TAIL * n))
    bad = []; sbad = []
    names = list(sig)
    for nm in names:
        kd, j = nm.split("/"); j = int(j)
        y = fz[f"Y_{kd}"][:, j].astype(np.float64); s = float(z[f"sign_{kd}"][j])
        if s != heavy_sign(y): sbad.append(nm)
        T = signature(rows, np.argsort(-s * y)[:k], toks, nt)
        if T != list(sig[nm]["T"]): bad.append(nm)
    # self-match testability summary
    valid = (rows % 255) != 254
    nxt = np.full(n, -1, np.int64); nxt[valid] = toks[rows[valid] + 1]
    out = {}
    for nm in names:
        kd, j = nm.split("/"); j = int(j)
        sc = float(z[f"sign_{kd}"][j]) * fz[f"Y_{kd}"][:, j].astype(np.float64)
        rec = {}
        for side, order in (("pos", np.argsort(-sc)), ("neg", np.argsort(sc))):
            T = sig[nm]["T"] if side == "pos" else signature(rows, np.argsort(sc)[:k], toks, nt)
            inT = np.isin(nxt, T) & valid
            top = [i for i in order[:800] if valid[i]][:200]
            rec[side] = (round(float(inT[top].mean()), 3), round(float(inT[valid].mean()), 4), len(T))
        out[nm] = rec
    print(L, "T mismatches", bad, "sign mismatches", sbad)
    negweak = {nm: r["neg"] for nm, r in out.items() if r["neg"][0] < 2 * r["neg"][1]}
    untest = {nm: r["pos"] for nm, r in out.items() if r["pos"][0] < 2 * r["pos"][1]}
    shortT = {nm: (r["pos"][2], r["neg"][2]) for nm, r in out.items() if r["pos"][2] < 20 or r["neg"][2] < 20}
    print("  untestable(pos)", untest)
    print("  neg side s<2base", negweak)
    print("  short sigs", shortT)
    print("  sample", list(out.items())[:4])
