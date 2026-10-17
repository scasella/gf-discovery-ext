"""Read-only feasibility look (D2, L12, raw residual saved by e67): is the positive control present
beyond the top-64 PCs of the identity-removed residual? Writes only to the scratchpad."""
import json, sys, time
import numpy as np
import scipy.sparse as sp

EXT = "/Users/scasella/Downloads/gf-discovery-ext"
OUT = sys.argv[1]
t0 = time.time()
H = np.load(f"{EXT}/results/e67_raw_L12_rows.npy", mmap_mode="r")
rows = np.load(f"{EXT}/results/occ_units_v2_L12.npz")["row"].astype(np.int64)
m = np.load(f"{EXT}/cache/occ/occD2_meta.npz")
tok, seq, pos = (m[k].astype(np.int64)[rows] for k in ("tokens", "seq", "pos"))
pieces = json.load(open(f"{EXT}/results/e64c_d2_pieces.json"))

# ---- labels: e61.feature_battery state machine (paren, bracket, quote), plus curly single quote
RARE = ("in_url", "code_markup", "in_caps_run", "after_colon", "list_item")
lab = {k: np.zeros(255000, np.float32) for k in ("paren_inside", "bracket_inside", "quote_inside") + RARE}
for si, toks in enumerate(pieces):
    depth = bdepth = q = 0; lp = 0; prev = ""; li = False; url = False; caps = 0; code = 0
    for j, st in enumerate(toks):
        r = si * 255 + j
        lab["paren_inside"][r] = depth > 0; lab["bracket_inside"][r] = bdepth > 0
        lab["quote_inside"][r] = q % 2 == 1
        lab["in_url"][r] = url; lab["code_markup"][r] = code > 0; lab["in_caps_run"][r] = caps >= 2
        lab["after_colon"][r] = prev.rstrip().endswith(":"); lab["list_item"][r] = li
        depth = max(0, depth + st.count("(") - st.count(")"))
        bdepth = max(0, bdepth + st.count("[") - st.count("]"))
        q += st.count('"') + st.count("“") + st.count("”")
        if "\n" in st:
            lp = 0; li = False; url = False
        else:
            lp += 1
        if lp <= 2 and (st.strip().rstrip(".)").isdigit() or st.strip() in ("-", "*", "•")):
            li = True
        if "http" in st or "www" in st:
            url = True
        if " " in st or "\n" in st:
            url = url and not st.startswith(" ")
        al = [c for c in st if c.isalpha()]
        caps = caps + 1 if (al and all(c.isupper() for c in al)) else 0
        code = 8 if any(c in st for c in "{}<>=;") else max(0, code - 1)
        prev = st
lab = {k: v[rows] for k, v in lab.items()}
print({k: float(v.mean()) for k, v in lab.items()}, flush=True)

keep = np.ones(1152, bool); keep[1038] = False
V = np.asarray(H, dtype=np.float32)[:, keep]
# bf16 rounding floor of the raw values (per coordinate variance of a half-ulp uniform error)
a = np.abs(V[::7]).astype(np.float64); a[a == 0] = 1e-30
ulp = 2.0 ** (np.floor(np.log2(a)) - 7)
qvar = (ulp ** 2 / 12).mean(0)
del a, ulp
print("loaded", V.shape, round(time.time() - t0), "s", flush=True)


def demean(V, g):
    n, G = len(g), int(g.max()) + 1
    M = sp.csr_matrix((np.ones(n), (g, np.arange(n))), shape=(G, n))
    cnt = np.maximum(np.asarray(M.sum(1)).ravel(), 1)
    for j in range(0, V.shape[1], 128):
        b = V[:, j:j + 128].astype(np.float64)
        V[:, j:j + 128] = (b - (M @ b / cnt[:, None])[g]).astype(np.float32)


groups = [np.unique(x, return_inverse=True)[1] for x in (tok, seq, pos)]
for _ in range(3):
    for g in groups:
        demean(V, g)
print("demeaned", round(time.time() - t0), "s", flush=True)
mu = V.mean(0)
C = np.cov((V - mu)[::3].T)
w, U = np.linalg.eigh(C); o = np.argsort(w)[::-1]; w, U = w[o], U[:, o]
cum = np.cumsum(w) / w.sum()
qfloor = (U ** 2 * qvar[:, None]).sum(0)
res = dict(n_rows=int(len(V)), base_rates={k: float(v.mean()) for k, v in lab.items()},
           cumvar={k: float(cum[k - 1]) for k in (16, 32, 64, 96, 128, 192, 256, 384, 512, 768, 1151)},
           eig={k: float(w[k - 1]) for k in (1, 16, 64, 65, 128, 256, 384, 512, 768, 1151)},
           bf16_floor_share={k: float(qfloor[k - 1] / w[k - 1]) for k in (1, 64, 128, 256, 384, 512, 768, 1151)},
           trace_bf16_share=float(qvar.sum() / w.sum()))
print(json.dumps(res), flush=True)
NZ = 512
Z = ((V - mu) @ U[:, :NZ] / np.sqrt(w[:NZ])).astype(np.float32)
del V

# ---- Mahalanobis split: squared standardized mean difference per PC block (whitened coordinates)
blocks = {"1-64": (0, 64), "65-128": (64, 128), "129-256": (128, 256), "65-256": (64, 256), "257-512": (256, 512)}
maha = {}
for k, y in lab.items():
    y = y > 0.5
    d = Z[y].mean(0) - Z[~y].mean(0)
    tot = float((d ** 2).sum())
    maha[k] = {b: float((d[s:e] ** 2).sum() / tot) for b, (s, e) in blocks.items()}
res["maha_share_of_d2_in_top512"] = maha
print(json.dumps(maha), flush=True)

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
ev = seq % 2 == 0
pr = {}
for k in ("paren_inside", "quote_inside", "bracket_inside") + RARE:
    y = lab[k] > 0.5
    pr[k] = {}
    bl = {**blocks, "1-256": (0, 256)} if k not in RARE else {b: blocks[b] for b in ("1-64", "65-256")}
    for b, (s, e) in bl.items():
        clf = LogisticRegression(C=1.0, max_iter=2000).fit(Z[ev, s:e], y[ev])
        pr[k][b] = float(roc_auc_score(y[~ev], clf.decision_function(Z[~ev, s:e])))
    print(k, pr[k], round(time.time() - t0), "s", flush=True)
res["probe_heldout_auroc"] = pr
json.dump(res, open(OUT, "w"), indent=1)
np.save(OUT.replace(".json", "_Z65_256.npy"), Z[:, 64:256])
np.savez(OUT.replace(".json", "_aux.npz"), tok=tok, seq=seq, pos=pos, **{k: v for k, v in lab.items()})
print("done", round(time.time() - t0), "s", flush=True)
