"""Read-only feasibility, part 4 (D2, L12 raw): v1-battery naming (section 6.1 rule) of the 192 block units,
and tail readability by the top-64 units (shadow test). Scratchpad only."""
import json, sys
import numpy as np
from scipy.stats import rankdata, spearmanr
EXT = "/Users/scasella/Downloads/gf-discovery-ext"; SP = sys.argv[1]
Y1 = np.load(f"{SP}/Y1.npy"); Yb = np.load(f"{SP}/Yb.npy")
rows = np.load(f"{EXT}/results/occ_units_v2_L12.npz")["row"].astype(np.int64)
pieces = json.load(open(f"{EXT}/results/e64c_d2_pieces.json"))
names = ("paren_inside", "bracket_inside", "quote_inside", "line_pos", "sent_pos", "after_newline", "list_item",
         "after_colon", "in_url", "after_digit", "in_caps_run", "prev_punct", "prev_capitalised", "word_continuation",
         "next_punct", "next_newline", "next_continuation", "code_markup")
F = {k: np.zeros(255000, np.float32) for k in names}
for si, toks in enumerate(pieces):
    depth = bdepth = 0; q = 0; lp = sp_ = 0; prev = ""; li = False; url = False; caps = 0; code = 0
    for j, st in enumerate(toks):
        i = si * 255 + j
        F["paren_inside"][i] = depth > 0; F["bracket_inside"][i] = bdepth > 0; F["quote_inside"][i] = q % 2 == 1
        F["line_pos"][i] = lp; F["sent_pos"][i] = sp_; F["after_newline"][i] = "\n" in prev; F["list_item"][i] = li
        F["after_colon"][i] = prev.rstrip().endswith(":"); F["in_url"][i] = url
        F["after_digit"][i] = prev.strip()[-1:].isdigit(); F["in_caps_run"][i] = caps >= 2
        ps_ = prev.strip()
        F["prev_punct"][i] = bool(ps_) and not any(c.isalnum() for c in ps_)
        F["prev_capitalised"][i] = bool(ps_[:1]) and ps_[:1].isupper()
        F["word_continuation"][i] = bool(st) and st[0].isalpha() and j > 0 and not prev.endswith((" ", "\n"))
        nx = toks[j + 1] if j + 1 < len(toks) else ""
        F["next_punct"][i] = bool(nx.strip()) and not any(c.isalnum() for c in nx.strip())
        F["next_newline"][i] = "\n" in nx; F["next_continuation"][i] = bool(nx) and nx[0].isalpha()
        F["code_markup"][i] = code > 0
        depth = max(0, depth + st.count("(") - st.count(")")); bdepth = max(0, bdepth + st.count("[") - st.count("]"))
        q += st.count('"') + st.count("“") + st.count("”")
        if "\n" in st: lp = 0; li = False; url = False
        else: lp += 1
        if lp <= 2 and (st.strip().rstrip(".)").isdigit() or st.strip() in ("-", "*", "•")): li = True
        sp_ = 0 if st.strip()[-1:] in (".", "!", "?") else sp_ + 1
        if "http" in st or "www" in st: url = True
        if " " in st or "\n" in st: url = url and not st.startswith(" ")
        al = [c for c in st if c.isalpha()]; caps = caps + 1 if (al and all(c.isupper() for c in al)) else 0
        code = 8 if any(c in st for c in "{}<>=;") else max(0, code - 1)
        prev = st
F = {k: v[rows] for k, v in F.items()}
n = len(rows); k1 = int(np.ceil(0.01 * n))
def auc_ranks(R, y):
    n1 = int(y.sum()); n0 = len(y) - n1
    a = (R[y].sum(0) - n1 * (n1 + 1) / 2) / (n1 * n0); return np.maximum(a, 1 - a)
Rb = np.stack([rankdata(Yb[:, j]) for j in range(Yb.shape[1])], 1)
R1 = np.stack([rankdata(Y1[:, j]) for j in range(Y1.shape[1])], 1)
def tails(y):
    o = np.argsort(y); return o[-k1:], o[:k1]
named = np.zeros(192, bool); why = {}
for k, v in F.items():
    if k in ("line_pos", "sent_pos"):
        rho = np.array([abs(spearmanr(Yb[::3, j], v[::3]).statistic) for j in range(192)])
        hit = rho >= 0.5
    else:
        y = v > 0.5
        if y.sum() < 50: continue
        a = auc_ranks(Rb, y); base = y.mean()
        enr = np.array([max(y[t].mean() for t in tails(Yb[:, j])) for j in range(192)])
        hit = (a >= 0.75) | ((enr >= 3 * base) & (enr >= 0.10))
    for j in np.flatnonzero(hit): why.setdefault(int(j), []).append(k)
    named |= hit
# shadow: max over top-64 units of |AUROC| for u's heavy-tail indicator
read = np.zeros(192)
for u in range(192):
    lo, hi = np.quantile(Yb[:, u], [0.01, 0.99]); y = Yb[:, u] if abs(hi) >= abs(lo) else -Yb[:, u]
    t = np.zeros(n, bool); t[np.argsort(-y)[:k1]] = True
    read[u] = auc_ranks(R1, t).max()
res = dict(n_named_v1battery=int(named.sum()), n_unnamed=int((~named).sum()),
           named_by=dict(sorted({k: sum(k in w for w in why.values()) for k in F}.items(), key=lambda x: -x[1])),
           tail_readability_q=[float(x) for x in np.quantile(read, [0.1, 0.25, 0.5, 0.75, 0.9])],
           n_read_ge_075=int((read >= 0.75).sum()), n_read_ge_070=int((read >= 0.70).sum()),
           unnamed_read_lt_070=int(((~named) & (read < 0.70)).sum()),
           controls_readability={k: float(read[u]) for k, u in (("paren#146", 146), ("quote#128", 128))},
           controls_named={k: why.get(u, []) for k, u in (("paren#146", 146), ("quote#128", 128))})
print(json.dumps(res, indent=1)); json.dump(res, open(f"{SP}/lv_feas4_L12.json", "w"), indent=1)
np.save(f"{SP}/named_read.npy", np.stack([named, read], 1))
