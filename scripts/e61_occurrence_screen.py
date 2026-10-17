"""e61 -- Stage 4 screen: label-free structure among identity-removed occurrences.

Implements docs/PREREG_occurrence.md (sha256 asserted). Per layer:

 1. REPRESENTATION, per data set (D discovery / S context-shuffled / R reserve replication),
    each set using ONLY its own rows:
      v_r = dec(r) - TM_set[token(r)]       identity removal (types with >= MIN_TYPE_OCC rows)
      drop outlier coordinates               (R10 / Addendum-H set for the layer)
      v_r -= mean_{r' in same sequence} v    document centring (removes topic offset)
      v_r -= mean_{r' at same position} v    position deflation (removes the BOS-distance code)
    rows at positions < MIN_POS are excluded (attention-sink positions).
 2. PCA(64) fitted on D, whitening; the same map applied to S and R.
 3. UNITS on D, blind: MiniBatchKMeans(K) and FastICA(N_ICA).
 4. BLIND SCREENS (no label touched): size, type diversity, document spread, split-half
    reproducibility (units re-fitted on even- and odd-sequence halves of D), token-driven
    index (token-matched replacement), presence in S (context-shuffled null).
 5. ORDERING / CLOSURE among the screened cluster centroids (ring and ruler screens), with the
    false-flag rate measured on S's centroids through the identical code.
 6. FREEZE: unit memberships, directions and every screen number are written to disk and
    hashed BEFORE step 7 reads any label.
 7. POSITIVE CONTROL (post-freeze): parenthesis-inside and quote-inside labels; max-over-units
    F1 (clusters) / AUROC (ICA) against a token-matched label-permutation null.
 8. LABELLING (post-freeze): named-feature battery associations + sampled contexts per unit.
 9. REPLICATION on R: independent units on R, matched to D's by centroid/direction cosine.
"""
import argparse, hashlib, json, os, sys, time
from pathlib import Path

import numpy as np

EXT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXT / "src"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")

PREREG = EXT / "PREREG_occurrence.md"
MIN_TYPE_OCC = 8
MIN_POS = 4
N_PCA = 64
K = 128
N_ICA = 32
SEED = 0
SCREEN = dict(min_size=300, min_types=30, max_type_share=0.30, min_docs=100,
              min_repro=0.80, max_tdi=3.0, bimod_bc=0.555, max_eta2=0.30)
N_PERM_PC = 1000
PC_PCT = 99.9          # 12 looks (2 labels x 2 unit kinds x 3 layers) -> FWER ~1.2%
F1_FLOOR, AUC_FLOOR = 0.50, 0.85


# ----------------------------------------------------------------------------------------
def drop_for(L):
    if L <= 2:
        return [1125]
    if L <= 10:
        return [1038]
    p = EXT / f"results/lag_deltas_L{L}.npz"
    if p.exists():
        return [int(x) for x in np.load(p)["dropped_coords"]]
    # declared fallback (PREREG_occurrence.md section 4; same as e46): Addendum-H rule on the
    # split-half raw type means of pass 1
    tm = np.load(EXT / "results/typemeans_pass1.npz")
    A, B = tm[f"raw_A_L{L}"].astype(np.float64), tm[f"raw_B_L{L}"].astype(np.float64)
    sd = np.sqrt(((A - B) ** 2).mean(0)); sh = sd ** 2 / (sd ** 2).sum()
    return [int(x) for x in np.flatnonzero(sh > 0.25)]


def load_set(tag, L):
    X = np.load(EXT / f"cache/occ/{tag}_L{L}.npy", mmap_mode="r")
    m = np.load(EXT / f"cache/occ/{tag}_meta.npz")
    return X, m["tokens"].astype(np.int64), m["seq"].astype(np.int64), m["pos"].astype(np.int64)


def _group_mean(V, g, n_groups):
    import scipy.sparse as sp
    M = sp.csr_matrix((np.ones(len(g)), (g, np.arange(len(g)))), shape=(n_groups, len(g)))
    cnt = np.asarray(M.sum(1)).ravel()
    return np.asarray(M @ V) / np.maximum(cnt, 1)[:, None]


def represent(tag, L):
    X, tok, seq, pos = load_set(tag, L)
    keep_c = np.ones(1152, bool); keep_c[drop_for(L)] = False
    ut, inv, cnt = np.unique(tok, return_inverse=True, return_counts=True)
    ok = (cnt[inv] >= MIN_TYPE_OCC) & (pos >= MIN_POS)
    idx = np.flatnonzero(ok)
    V = np.asarray(X[idx], dtype=np.float32)[:, keep_c]
    t, s, p = tok[idx], seq[idx], pos[idx]
    _, tinv = np.unique(t, return_inverse=True)
    V -= _group_mean(V.astype(np.float64), tinv, tinv.max() + 1)[tinv].astype(np.float32)   # identity
    _, sinv = np.unique(s, return_inverse=True)
    V -= _group_mean(V.astype(np.float64), sinv, sinv.max() + 1)[sinv].astype(np.float32)   # document
    _, pinv = np.unique(p, return_inverse=True)
    V -= _group_mean(V.astype(np.float64), pinv, pinv.max() + 1)[pinv].astype(np.float32)   # position
    assert np.isfinite(V).all()
    return dict(V=V, tok=t, seq=s, pos=p, row=idx, tag=tag, L=L)


CFG = dict(K=K, N_ICA=N_ICA, N_PCA=N_PCA)


def pca_fit(V, n=None):
    n = CFG["N_PCA"] if n is None else n
    mu = V.mean(0)
    C = np.cov((V - mu)[::3].T)
    w, U = np.linalg.eigh(C)
    o = np.argsort(w)[::-1][:n]
    return dict(mu=mu, U=U[:, o], sd=np.sqrt(w[o]), var_frac=float(w[o].sum() / w.sum()))


def pca_apply(P, V):
    return ((V - P["mu"]) @ P["U"]) / P["sd"]


def fit_units(W, seed=SEED):
    from sklearn.cluster import MiniBatchKMeans
    from sklearn.decomposition import FastICA
    km = MiniBatchKMeans(n_clusters=CFG["K"], batch_size=4096, n_init=3, max_iter=200,
                         random_state=seed).fit(W)
    ica = FastICA(n_components=CFG["N_ICA"], whiten="unit-variance", random_state=seed,
                  max_iter=400).fit(W[::2])
    return km, ica


def bimodality(x):
    from scipy.stats import skew, kurtosis
    n = len(x)
    g, k = skew(x), kurtosis(x)
    return float((g ** 2 + 1) / (k + 3 * (n - 1) ** 2 / ((n - 2) * (n - 3))))


def raw_centroids(V, labels, k):
    C = np.zeros((k, V.shape[1])); n = np.bincount(labels, minlength=k)
    np.add.at(C, labels, V)
    return C / np.maximum(n, 1)[:, None], n


def cos_match(A, B):
    An = A / np.maximum(np.linalg.norm(A, axis=1, keepdims=True), 1e-12)
    Bn = B / np.maximum(np.linalg.norm(B, axis=1, keepdims=True), 1e-12)
    S = An @ Bn.T
    return S.max(1), S.argmax(1)


# ----------------------------------------------------------------------------------------
def screen_layer(L, D, S, out):
    rng = np.random.default_rng(SEED)
    P = pca_fit(D["V"])
    WD = pca_apply(P, D["V"])
    WS = pca_apply(P, S["V"]) if S is not None else None
    km, ica = fit_units(WD)
    lab = km.labels_
    CD, nD = raw_centroids(D["V"], lab, CFG["K"])
    # halves of D by sequence parity
    ev = (D["seq"] % 2 == 0)
    km_e, ica_e = fit_units(WD[ev], seed=1)
    km_o, ica_o = fit_units(WD[~ev], seed=2)
    Ce, _ = raw_centroids(D["V"][ev], km_e.labels_, CFG["K"])
    Co, _ = raw_centroids(D["V"][~ev], km_o.labels_, CFG["K"])
    re_e, _ = cos_match(CD, Ce); re_o, _ = cos_match(CD, Co)
    # context-shuffled set: independent units
    if S is not None:
        km_s, ica_s = fit_units(WS, seed=3)
        CS, _ = raw_centroids(S["V"], km_s.labels_, CFG["K"])
        in_S, _ = cos_match(CD, CS)
    else:
        km_s = ica_s = None
        in_S = np.full(CFG["K"], np.nan)
    # token-matched replacement (TDI)
    by_type = {}
    for i, t in enumerate(D["tok"]):
        by_type.setdefault(int(t), []).append(i)
    by_type = {t: np.asarray(v) for t, v in by_type.items()}
    rep = np.asarray([by_type[int(t)][rng.integers(len(by_type[int(t)]))] for t in D["tok"]])
    lab_rep = lab[rep]
    units = []
    for c in range(CFG["K"]):
        mem = np.flatnonzero(lab == c)
        if len(mem) == 0:
            continue
        tt, tc = np.unique(D["tok"][mem], return_counts=True)
        rate = float((lab_rep[mem] == c).mean())
        base = len(mem) / len(lab)
        u = dict(kind="cluster", id=int(c), size=int(len(mem)), n_types=int(len(tt)),
                 max_type_share=float(tc.max() / len(mem)),
                 n_docs=int(len(np.unique(D["seq"][mem]))),
                 repro=float(min(re_e[c], re_o[c])), tdi=float(rate / base),
                 in_shuffled_cos=float(in_S[c]),
                 mean_pos=float(D["pos"][mem].mean()))
        u["screened"] = bool(u["size"] >= SCREEN["min_size"] and u["n_types"] >= SCREEN["min_types"]
                             and u["max_type_share"] <= SCREEN["max_type_share"]
                             and u["n_docs"] >= SCREEN["min_docs"] and u["repro"] >= SCREEN["min_repro"]
                             and u["tdi"] <= SCREEN["max_tdi"])
        units.append(u)
    # ICA units
    Y = ica.transform(WD)
    mix = ica.mixing_                                     # whitened-PCA space directions
    def dir_match(Mx, My):
        A = Mx.T / np.linalg.norm(Mx.T, axis=1, keepdims=True)
        B = My.T / np.linalg.norm(My.T, axis=1, keepdims=True)
        return np.abs(A @ B.T).max(1)
    rep_e, rep_o = dir_match(mix, ica_e.mixing_), dir_match(mix, ica_o.mixing_)
    in_S_ica = dir_match(mix, ica_s.mixing_) if ica_s is not None else np.full(CFG["N_ICA"], np.nan)
    for j in range(CFG["N_ICA"]):
        y = Y[:, j]
        # eta^2 of the projection on token identity
        ut, inv = np.unique(D["tok"], return_inverse=True)
        mt = np.bincount(inv, weights=y) / np.bincount(inv)
        eta2 = float(((mt[inv] - y.mean()) ** 2).sum() / max(((y - y.mean()) ** 2).sum(), 1e-12))
        u = dict(kind="ica", id=int(j), bimodality=bimodality(y[::3]), eta2_token=eta2,
                 repro=float(min(rep_e[j], rep_o[j])), in_shuffled_cos=float(in_S_ica[j]),
                 kurtosis=float(((y - y.mean()) ** 4).mean() / y.var() ** 2 - 3))
        u["screened"] = bool(u["repro"] >= SCREEN["min_repro"] and u["eta2_token"] <= SCREEN["max_eta2"]
                             and u["bimodality"] > SCREEN["bimod_bc"])
        units.append(u)
    # ordering / closure among screened cluster centroids (whitened PCA space), data vs S
    if S is not None:
        arr = order_closure(WD, lab, [u["id"] for u in units if u["kind"] == "cluster" and u["screened"]], D)
        lab_s = km_s.labels_
        nS = np.bincount(lab_s, minlength=CFG["K"])
        arr_S = order_closure(WS, lab_s, [c for c in range(CFG["K"]) if nS[c] >= SCREEN["min_size"]], S)
    else:
        arr = arr_S = None
    frozen = dict(layer=L, pca_var_frac=P["var_frac"], n_rows=dict(D=len(D["V"]), S=None if S is None else len(S["V"])),
                  units=units, arrangements=arr, arrangements_shuffled_null=arr_S,
                  labels_md5=hashlib.md5(lab.astype(np.int16).tobytes()).hexdigest())
    if S is None:
        return frozen, dict(P=P, km=km, ica=ica, Y=Y, lab=lab, CD=CD)
    np.savez_compressed(EXT / f"results/occ_units_L{L}.npz", lab=lab.astype(np.int16), row=D["row"],
                        ica_Y=Y.astype(np.float32), mix=mix, pca_U=P["U"], pca_mu=P["mu"],
                        pca_sd=P["sd"], centroids_raw=CD, km_centers=km.cluster_centers_)
    return frozen, dict(P=P, km=km, ica=ica, Y=Y, lab=lab, CD=CD)


def order_closure(W, lab, ids, Dset, ks=(5, 6, 7, 8)):
    """Rings (closure) and rulers (spacing) among cluster centroids; centroids on sequence
    halves A (even) / B (odd) play the roles of the type-mean halves in Stages 1-2."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("e46", EXT / "scripts/e46_ruler_lattice.py")
    e46 = importlib.util.module_from_spec(spec); spec.loader.exec_module(e46)
    spec22 = importlib.util.spec_from_file_location("s22", EXT / "scripts/22_token_type_geometry.py")
    s22 = importlib.util.module_from_spec(spec22); spec22.loader.exec_module(s22)
    ids = list(ids)
    if len(ids) < 8:
        return dict(n_centroids=len(ids), note="fewer than 8 screened centroids")
    ev = (Dset["seq"] % 2 == 0)
    CA = np.stack([W[(lab == c) & ev].mean(0) for c in ids])
    CB = np.stack([W[(lab == c) & ~ev].mean(0) for c in ids])
    Cf = np.stack([W[lab == c].mean(0) for c in ids])
    rng = np.random.default_rng(7)
    rec = {}
    for k in ks:
        if k >= len(ids):
            continue
        props = e46.proposals(Cf - Cf.mean(0), np.ones(len(ids)), (k,), depth=min(30, len(ids) - 1))[k]
        bg = [rng.choice(len(ids), size=k, replace=False) for _ in range(2000)]
        bg_gap = np.asarray([s22.screen(Cf[b], 2000)["max_gap_deg"] for b in bg])
        bg_reg = np.sort([e46.ruler_stats(CA[b], CB[b])["spacing_reg_B"] for b in bg])
        flags = []
        for m in props:
            m = np.asarray(m)
            sc = s22.screen(Cf[m], 20000)
            rs = e46.ruler_stats(CA[m], CB[m])
            pbg_ring = float((1 + (bg_gap <= sc["max_gap_deg"]).sum()) / (1 + len(bg_gap)))
            pbg_rul = float((1 + np.searchsorted(bg_reg, rs["spacing_reg_B"], side="right"))
                            / (1 + len(bg_reg))) if np.isfinite(rs["spacing_reg_B"]) else 1.0
            ring = bool(sc["p_closed"] < 0.05 and pbg_ring <= 0.05)
            ruler = bool(pbg_rul <= 0.01 and rs["order_B"] >= 0.9)
            if ring or ruler:
                flags.append(dict(members=[ids[i] for i in m], ring=ring, ruler=ruler,
                                  gap_ratio=sc["gap_ratio"], p_closed=sc["p_closed"],
                                  p_bg_ring=pbg_ring, spacing_reg_B=rs["spacing_reg_B"],
                                  p_bg_ruler=pbg_rul, order_B=rs["order_B"],
                                  level_order=[ids[i] for i in m[np.argsort(rs["pA"])]]))
        rec[k] = dict(n_props=len(props), n_ring=sum(f["ring"] for f in flags),
                      n_ruler=sum(f["ruler"] for f in flags), flags=flags)
    return dict(n_centroids=len(ids), by_k=rec)


# ----------------------------------------------------------------------------------------
def token_strings():
    from transformers import AutoTokenizer
    snap = sorted((EXT / "hf_home/hub/models--google--gemma-3-1b-pt/snapshots").glob("*"))[0]
    return AutoTokenizer.from_pretrained(str(snap))


def feature_battery(Dset, tk, full_tokens, full_seq):
    """Named context variables per kept row, computed from the sequence's own tokens."""
    strs = {}
    def s_of(t):
        if t not in strs:
            strs[t] = tk.decode([int(t)])
        return strs[t]
    n_seq_rows = 255
    feats = {k: np.zeros(len(Dset["row"]), dtype=np.float32) for k in
             ("paren_inside", "bracket_inside", "quote_inside", "line_pos", "sent_pos",
              "after_newline", "list_item", "after_colon", "in_url", "after_digit", "in_caps_run",
              "prev_punct", "prev_capitalised", "word_continuation", "next_punct", "next_newline",
              "next_continuation", "code_markup")}
    row_set = {int(r): i for i, r in enumerate(Dset["row"])}
    for s0 in range(0, len(full_tokens), n_seq_rows):
        toks = full_tokens[s0:s0 + n_seq_rows]
        depth = bdepth = 0; q = 0; lp = sp = 0; prev = ""; li = False; url = False; caps = 0
        code = 0
        for j, t in enumerate(toks):
            st = s_of(int(t))
            r = s0 + j
            i = row_set.get(r)
            if i is not None:
                feats["paren_inside"][i] = float(depth > 0)
                feats["bracket_inside"][i] = float(bdepth > 0)
                feats["quote_inside"][i] = float(q % 2 == 1)
                feats["line_pos"][i] = lp
                feats["sent_pos"][i] = sp
                feats["after_newline"][i] = float("\n" in prev)
                feats["list_item"][i] = float(li)
                feats["after_colon"][i] = float(prev.rstrip().endswith(":"))
                feats["in_url"][i] = float(url)
                feats["after_digit"][i] = float(prev.strip()[-1:].isdigit())
                feats["in_caps_run"][i] = float(caps >= 2)
                ps_ = prev.strip()
                feats["prev_punct"][i] = float(bool(ps_) and not any(c.isalnum() for c in ps_))
                feats["prev_capitalised"][i] = float(bool(ps_[:1]) and ps_[:1].isupper())
                feats["word_continuation"][i] = float(bool(st) and st[0].isalpha() and j > 0
                                                      and not prev.endswith((" ", "\n")))
                nx = s_of(int(toks[j + 1])) if j + 1 < len(toks) else ""
                feats["next_punct"][i] = float(bool(nx.strip()) and not any(c.isalnum() for c in nx.strip()))
                feats["next_newline"][i] = float("\n" in nx)
                feats["next_continuation"][i] = float(bool(nx) and nx[0].isalpha())
                feats["code_markup"][i] = float(code > 0)
            depth = max(0, depth + st.count("(") - st.count(")"))
            bdepth = max(0, bdepth + st.count("[") - st.count("]"))
            q += st.count('"') + st.count("“") + st.count("”")
            if "\n" in st:
                lp = 0; li = False; url = False
            else:
                lp += 1
            if lp <= 2 and (st.strip().rstrip(".)").isdigit() or st.strip() in ("-", "*", "•")):
                li = True
            sp = 0 if st.strip()[-1:] in (".", "!", "?") else sp + 1
            if "http" in st or "www" in st:
                url = True
            if " " in st or "\n" in st:
                url = url and not st.startswith(" ")
            al = [c for c in st if c.isalpha()]
            caps = caps + 1 if (al and all(c.isupper() for c in al)) else 0
            code = 8 if any(c in st for c in "{}<>=;") else max(0, code - 1)   # within 8 tokens of markup
            prev = st
    return feats


def f1_member(mem_mask, y):
    tp = float((mem_mask & (y > 0.5)).sum())
    if tp == 0:
        return 0.0
    prec = tp / mem_mask.sum(); rec = tp / (y > 0.5).sum()
    return 2 * prec * rec / (prec + rec)


def auroc(score, y):
    from sklearn.metrics import roc_auc_score
    if y.min() == y.max():
        return 0.5
    return float(roc_auc_score(y, score))


def positive_control(frozen, fit, Dset, feats, n_perm=N_PERM_PC):
    """Max-over-units F1 (screened clusters) and AUROC (screened ICA) vs a token-matched
    label-permutation null. Rank-sum AUROC and bincount F1 make each permutation O(n)."""
    from scipy.stats import rankdata
    rng = np.random.default_rng(99)
    units = frozen["units"]
    cl = np.asarray([u["id"] for u in units if u["kind"] == "cluster" and u["screened"]], int)
    ic = [u["id"] for u in units if u["kind"] == "ica" and u["screened"]]
    lab, Y = fit["lab"], fit["Y"]
    n = len(lab)
    size = np.bincount(lab, minlength=CFG["K"]).astype(float)
    R = np.stack([rankdata(Y[:, j]) for j in ic], 1) if ic else np.zeros((n, 0))
    _, inv = np.unique(Dset["tok"], return_inverse=True)
    slots = np.argsort(inv, kind="stable")       # rows grouped by token type

    def stats(y):
        pos = y > 0.5
        n1 = pos.sum(); n0 = n - n1
        if n1 == 0 or n0 == 0:
            return 0.0, 0.5, None, None
        tp = np.bincount(lab[pos], minlength=CFG["K"]).astype(float)
        f1 = 2 * tp / np.maximum(size + n1, 1)
        f1c = f1[cl] if len(cl) else np.zeros(1)
        if R.shape[1]:
            auc = (R[pos].sum(0) - n1 * (n1 + 1) / 2) / (n1 * n0)
            auc2 = np.maximum(auc, 1 - auc)
        else:
            auc2 = np.asarray([0.5])
        bc = int(cl[int(np.argmax(f1c))]) if len(cl) else None
        bi = int(ic[int(np.argmax(auc2))]) if ic else None
        return float(f1c.max()), float(auc2.max()), bc, bi

    out = {}
    for name in ("paren_inside", "quote_inside"):
        y = feats[name]
        f1o, auo, bc, bi = stats(y)
        nf, na = [], []
        for _ in range(n_perm):
            src = np.lexsort((rng.random(n), inv))   # within-type random order
            yp = np.empty_like(y); yp[slots] = y[src]
            a_, b_, _, _ = stats(yp)
            nf.append(a_); na.append(b_)
        p99f, p99a = float(np.percentile(nf, PC_PCT)), float(np.percentile(na, PC_PCT))
        rec_c = bool(f1o >= F1_FLOOR and f1o > p99f)
        rec_i = bool(auo >= AUC_FLOOR and auo > p99a)
        out[name] = dict(base_rate=float(y.mean()), max_f1_cluster=f1o, best_cluster=bc,
                         null_p99_f1=p99f, max_auroc_ica=auo, best_ica=bi, null_p99_auroc=p99a,
                         recovered_by_cluster=rec_c, recovered_by_ica=rec_i,
                         recovered=bool(rec_c or rec_i), n_perm=n_perm)
    out["PASS"] = bool(any(v["recovered"] for v in out.values() if isinstance(v, dict)))
    return out


def label_units(frozen, fit, Dset, feats, tk, full_tokens, n_ctx=12):
    lab, Y = fit["lab"], fit["Y"]
    rng = np.random.default_rng(5)
    rows = []
    for u in frozen["units"]:
        if not u["screened"]:
            continue
        if u["kind"] == "cluster":
            mem = lab == u["id"]
            assoc = {k: float(v[mem].mean() / max(v.mean(), 1e-9)) if set(np.unique(v)) <= {0.0, 1.0}
                     else float((v[mem].mean() - v.mean()) / max(v.std(), 1e-9))
                     for k, v in feats.items()}
            idx = np.flatnonzero(mem)
        else:
            y = Y[:, u["id"]]
            assoc = {}
            for k, v in feats.items():
                if set(np.unique(v)) <= {0.0, 1.0}:
                    a_ = auroc(y, v); assoc[k] = max(a_, 1 - a_)
                else:
                    from scipy.stats import spearmanr
                    assoc[k] = float(abs(spearmanr(y[::5], v[::5]).statistic))
            hi = np.quantile(y, 0.99) if abs(np.quantile(y, 0.99)) > abs(np.quantile(y, 0.01)) else np.quantile(y, 0.01)
            idx = np.flatnonzero(y >= hi) if hi > 0 else np.flatnonzero(y <= hi)
        pick = rng.choice(idx, size=min(n_ctx, len(idx)), replace=False)
        ctxs = []
        for i in pick:
            r = int(Dset["row"][i])
            s0 = (r // 255) * 255
            lo = max(s0, r - 12)
            left = tk.decode([int(x) for x in full_tokens[lo:r]])
            mid = tk.decode([int(full_tokens[r])])
            right = tk.decode([int(x) for x in full_tokens[r + 1:min(s0 + 255, r + 5)]])
            ctxs.append(f"{left}[[{mid}]]{right}")
        top_types = None
        if u["kind"] == "cluster":
            tt, tc = np.unique(Dset["tok"][lab == u["id"]], return_counts=True)
            o = np.argsort(-tc)[:8]
            top_types = [(tk.decode([int(tt[j])]), int(tc[j])) for j in o]
        rows.append(dict(kind=u["kind"], id=u["id"], association=assoc, top_types=top_types,
                         contexts=ctxs))
    return rows


def pilot(layers, tagP):
    """Prereg section 7: grid over (K, N_ICA, N_PCA) on the pilot set P, positive control only.
    Choice = recovered first, then the largest margin (statistic - null p99) for paren_inside."""
    tk = token_strings()
    full = np.load(EXT / f"cache/occ/{tagP}_meta.npz")["tokens"]
    out = dict(phase="pilot", grid=[], layers=layers)
    for L in layers:
        Pset = represent(tagP, L)
        feats = feature_battery(Pset, tk, full, None)
        for K_ in (64, 128, 256):
            for nica in (16, 32):
                for npca in (32, 64):
                    CFG.update(K=K_, N_ICA=nica, N_PCA=npca)
                    t0 = time.time()
                    frozen, fit = screen_layer(L, Pset, None, {})
                    pc = positive_control(frozen, fit, Pset, feats)
                    pp = pc["paren_inside"]
                    margin = max(pp["max_f1_cluster"] - pp["null_p99_f1"],
                                 pp["max_auroc_ica"] - pp["null_p99_auroc"])
                    row = dict(layer=L, K=K_, N_ICA=nica, N_PCA=npca, paren=pp,
                               quote=pc["quote_inside"], recovered=pp["recovered"], margin=margin,
                               n_screened=sum(u["screened"] for u in frozen["units"]),
                               seconds=round(time.time() - t0, 1))
                    out["grid"].append(row)
                    print(f"  pilot L{L} K={K_} ica={nica} pca={npca}: paren F1 {pp['max_f1_cluster']:.3f} "
                          f"(null {pp['null_p99_f1']:.3f}) AUC {pp['max_auroc_ica']:.3f} "
                          f"(null {pp['null_p99_auroc']:.3f}) rec={pp['recovered']} | quote rec="
                          f"{pc['quote_inside']['recovered']}", flush=True)
                    (EXT / "results/e61_pilot.json").write_text(json.dumps(out, indent=1, default=str))
    # choose across layers: count of layers recovered, then mean margin
    agg = {}
    for r in out["grid"]:
        key = (r["K"], r["N_ICA"], r["N_PCA"])
        agg.setdefault(key, []).append(r)
    best = max(agg.items(), key=lambda kv: (sum(x["recovered"] for x in kv[1]),
                                            np.mean([x["margin"] for x in kv[1]])))
    chosen = dict(K=best[0][0], N_ICA=best[0][1], N_PCA=best[0][2],
                  n_layers_recovered=sum(x["recovered"] for x in best[1]))
    if chosen["n_layers_recovered"] == 0:
        chosen = dict(K=K, N_ICA=N_ICA, N_PCA=N_PCA, n_layers_recovered=0,
                      note="no setting recovered on P: section-5 defaults used")
    out["chosen"] = chosen
    (EXT / "results/e61_pilot.json").write_text(json.dumps(out, indent=1, default=str))
    (EXT / "results/e61_pilot_choice.json").write_text(json.dumps(dict(chosen=chosen), indent=1))
    print("CHOSEN", chosen, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", required=True)
    ap.add_argument("--D", default="occD"); ap.add_argument("--S", default="occS")
    ap.add_argument("--R", default="occR")
    ap.add_argument("--out", default=str(EXT / "results/e61_occurrence.json"))
    ap.add_argument("--phase", choices=["pilot", "freeze", "label", "replicate"], required=True)
    ap.add_argument("--P", default="occP")
    a = ap.parse_args()
    layers = [int(x) for x in a.layers.split(",")]
    if a.phase == "pilot":
        return pilot(layers, a.P)
    cfgp = EXT / "results/e61_pilot_choice.json"
    if cfgp.exists():
        CFG.update({k: v for k, v in json.loads(cfgp.read_text())["chosen"].items() if k in CFG})
    sha = hashlib.sha256(PREREG.read_bytes()).hexdigest()
    outp = Path(a.out)
    res = json.loads(outp.read_text()) if outp.exists() else dict(prereg_sha256=sha, layers={})
    assert res["prereg_sha256"] == sha, "PREREG_occurrence.md changed after the freeze"
    tk = token_strings()
    for L in layers:
        t0 = time.time()
        D = represent(a.D, L)
        if a.phase == "freeze":
            S = represent(a.S, L)
            frozen, fit = screen_layer(L, D, S, res)
            frozen["freeze_md5"] = hashlib.md5(json.dumps(frozen, sort_keys=True, default=str)
                                               .encode()).hexdigest()
            frozen["frozen_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
            res["layers"][str(L)] = dict(frozen=frozen)
        else:
            z = np.load(EXT / f"results/occ_units_L{L}.npz")
            frozen = res["layers"][str(L)]["frozen"]
            assert np.array_equal(z["row"], D["row"])
            fit = dict(lab=z["lab"].astype(np.int64), Y=z["ica_Y"])
            full = np.load(EXT / f"cache/occ/{a.D}_meta.npz")["tokens"]
            if a.phase == "label":
                feats = feature_battery(D, tk, full, None)
                res["layers"][str(L)]["positive_control"] = positive_control(frozen, fit, D, feats)
                res["layers"][str(L)]["labels"] = label_units(frozen, fit, D, feats, tk, full)
            else:
                R = represent(a.R, L)
                P = dict(mu=z["pca_mu"], U=z["pca_U"], sd=z["pca_sd"])
                WR = pca_apply(P, R["V"])
                kmR, icaR = fit_units(WR, seed=11)
                CR, _ = raw_centroids(R["V"], kmR.labels_, CFG["K"])
                mc, _ = cos_match(z["centroids_raw"], CR)
                mix = z["mix"]
                A = mix.T / np.linalg.norm(mix.T, axis=1, keepdims=True)
                B = icaR.mixing_.T / np.linalg.norm(icaR.mixing_.T, axis=1, keepdims=True)
                mi = np.abs(A @ B.T).max(1)
                res["layers"][str(L)]["replication"] = dict(
                    cluster_match_cos=[float(x) for x in mc], ica_match_cos=[float(x) for x in mi])
        res["layers"][str(L)][f"seconds_{a.phase}"] = round(time.time() - t0, 1)
        outp.write_text(json.dumps(res, indent=1, default=str))
        print(f"L{L} {a.phase} done in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
