import json, numpy as np
EXT = "/Users/scasella/Downloads/gf-discovery-ext/"
c = np.fromfile(EXT + "results/corpus_tokens_pass1.i32", dtype=np.int32)
tokD = c[1000*255:2000*255].astype(np.int64)
fr = json.load(open(EXT + "results/e65_frozen.json"))
ver = json.load(open(EXT + "results/e64_verdict.json"))
CTRL = {"7": {"logcosh/12", "skew/62"}, "12": {"logcosh/33", "skew/33", "logcosh/11", "skew/23"}}
res = {}
for L in ["7", "12"]:
    z = np.load(EXT + f"results/occ_units_v2_L{L}.npz"); rows = z["row"].astype(np.int64)
    m = np.load(EXT + f"results/e64_maps_L{L}.npz")
    valid = (rows % 255) != 254
    nxt = np.full(len(rows), -1); nxt[valid] = tokD[rows[valid] + 1]
    S = fr["layers"][L]["signatures"]
    cls = {}
    for u in ver["layers"][L]["units"]:
        key = f"{u['kind']}/{u['id']}"
        cls[key] = ("cand" if u["tier"] != "NAMED_BY_CONTEXT" else "named", u["tier"], [f"{t[0]}/{t[1]}" for t in u["twin"]])
    for k in CTRL[L]: cls[k] = ("ctrl", "ctrl", [])
    units = list(S)
    top200 = {}; tail = {}
    for u in units:
        kd, j = u.split("/"); j = int(j)
        y = z[f"Y_{kd}"][:, j].astype(np.float64); s = m[f"sign_{kd}"][j]
        o = np.argsort(-s * y)
        ov = o[valid[o]]
        top200[u] = ov[:200]; tail[u] = ov[:int(np.ceil(0.01 * len(y)))]
    out = {}
    for u in units:
        T = np.array(S[u]["T"])
        inT = np.isin(nxt, T) & valid
        base = inT[valid].mean()
        thr = max(3 * base, 0.10)
        own200 = inT[top200[u]].mean(); owntail = inT[tail[u]].mean()
        # per-token base
        cnts = sorted([(int(np.sum(nxt[valid] == t)), t) for t in T], reverse=True)
        out[u] = dict(cls=cls.get(u, ("other",))[0], tier=cls.get(u, ("", ""))[1], base=base, thr=thr, own200=own200,
                      owntail=owntail, top3cnt=cnts[:3], ntok=len(T), twins=cls.get(u, ("", "", []))[2])
    # cross-unit: v's top200 scored with u's T
    cross = []
    for u in units:
        T = np.array(S[u]["T"]); inT = np.isin(nxt, T) & valid
        thr = out[u]["thr"]
        for v in units:
            if v == u or v in out[u]["twins"] or u in out[v]["twins"]: continue
            cross.append((u, v, inT[top200[v]].mean() >= thr, inT[top200[v]].mean()))
    res[L] = (out, cross)
    print(f"=== L{L} ===")
    for grp in ["cand", "named", "ctrl", "other"]:
        us = [u for u in units if out[u]["cls"] == grp]
        if not us: continue
        b = np.array([out[u]["base"] for u in us]); o2 = np.array([out[u]["own200"] for u in us]); ot = np.array([out[u]["owntail"] for u in us])
        th = np.array([out[u]["thr"] for u in us])
        print(f"{grp}: n={len(us)} base med {np.median(b):.3f} [{b.min():.3f},{b.max():.3f}] | thr med {np.median(th):.3f} | own top200 med {np.median(o2):.3f} [{o2.min():.3f},{o2.max():.3f}] | own tail med {np.median(ot):.3f}")
        print(f"   self-pass (own top200 >= thr): {np.sum(o2 >= th)}/{len(us)}; own tail >= thr: {np.sum(ot >= th)}/{len(us)}; floor 0.10 binding (3*base<0.10): {np.sum(3*b < 0.10)}")
        cr = [x for x in cross if out[x[0]]["cls"] == grp]
        print(f"   cross-unit pass rate (other non-twin unit's top200 matches u): {np.mean([x[2] for x in cr]):.3f} ({sum(x[2] for x in cr)}/{len(cr)}); units with >=1 foreign match: {len(set(x[0] for x in cr if x[2]))}/{len(us)}")
json.dump({L: {u: {k: (v if not isinstance(v, np.floating) else float(v)) for k, v in d.items()} for u, d in res[L][0].items()} for L in res},
          open("/private/tmp/claude-501/-Users-scasella-Downloads/db01a4e8-8f9e-44d9-a3d5-6be3160de0fb/scratchpad/match_stats.json", "w"), default=float, indent=1)
