"""e63 -- Stage 4 v2: identity removed from the first AND the second moment.

Phases
  pilot    P2 only (corpus 13000-13999): docs/stage4v2_calibration_plan_2026-10-15.md
           Step A (per-type whitening setting per layer: lowest TDI under a probe guard) and
           Step B (unit kinds + screens, paren-tuned; quote reported, never used).
  freeze   D2 (corpus 1000-1999): representation, units, blind screens; written and hashed
           before any label is computed. PREREG_occurrence_v2.md sha256 asserted.
  control  D2: positive control (paren_inside / quote_inside) against the token-matched null
           with the frozen per-layer floors; the supervised probe is reported for context.

Representation (per set, only that set's rows)
  rows: positions >= 4, types with >= 8 rows; outlier coordinates dropped (e61.drop_for)
  first moment: type, document and position means subtracted in turn, cycle repeated N_ITER
  Z: PCA-64 fitted on the set (every third row), whitened
  second moment: per type t with n_t >= N_MIN, z <- (z - mu_t) @ Sig_t^-1/2 with
      Sig_t = (1-lam) S_t + lam S_pool, lam = n0 / (n_t + n0); rarer types use S_pool;
      then a global re-whitening (rotation + scale, 64-d)
"""
import argparse, hashlib, importlib.util, itertools, json, os, sys, time
from pathlib import Path

import numpy as np

EXT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXT / "src"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
_spec = importlib.util.spec_from_file_location("e61", EXT / "scripts/e61_occurrence_screen.py")
e61 = importlib.util.module_from_spec(_spec); sys.modules["e61"] = e61; _spec.loader.exec_module(e61)

PREREG = EXT / "PREREG_occurrence_v2.md"
PLAN = EXT / "docs/stage4v2_calibration_plan_2026-10-15.md"
PLAN_SHA = "05e19ea3b518f34fff48050256dfe1622337ba399167fa2a16661470911aee09"
CHOICE = EXT / "results/e63_pilot_choice.json"
MIN_TYPE_OCC, MIN_POS, N_PCA, N_ITER = 8, 4, 64, 3
N_PERM, PC_PCT, F1_FLOOR = 1000, 99.9, 0.50
LABELS = ("paren_inside", "quote_inside")
# calibration grid (plan, Step A / Step B)
GRID_NMIN, GRID_N0 = (32, 64, 128), (16, 64, 256)
GRID_K, GRID_NICA, CONTRASTS = (16, 32, 64, 128), (64,), ("logcosh", "skew")   # N_ICA: addendum A1
CL_FIXED = dict(min_size=300, min_types=30, max_type_share=0.30, min_docs=100)
ICA_MAX_ETA2 = 0.30          # first moment: vacuous after exact type-mean removal (addendum A2)
ICA_MAX_ETA2_SQ = 0.30       # second moment: eta^2(token) of y^2 (addendum A2)
GRID_TDI, GRID_CREP, GRID_IREP, GRID_BIMOD = (3.0, 5.0), (0.8, 0.6), (0.8, 0.5), (True, False)
DEFAULT = dict(K=32, N_logcosh=64, N_skew=64, tdi_cap=3.0, cluster_repro=0.8, ica_repro=0.8,
               bimodality=False)


# ---------------------------------------------------------------------------- representation
def demean(V, g, block=128):
    """V[r] -= mean of V over rows sharing group g[r]; column blocks keep float64 temporaries small."""
    import scipy.sparse as sp
    n, G = len(g), int(g.max()) + 1
    M = sp.csr_matrix((np.ones(n), (g, np.arange(n))), shape=(G, n))
    cnt = np.maximum(np.asarray(M.sum(1)).ravel(), 1)
    for j in range(0, V.shape[1], block):
        b = V[:, j:j + block].astype(np.float64)
        V[:, j:j + block] = (b - (M @ b / cnt[:, None])[g]).astype(np.float32)


def represent_base(tag, L, n_iter=N_ITER):
    X, tok, seq, pos = e61.load_set(tag, L)
    keep_c = np.ones(1152, bool); keep_c[e61.drop_for(L)] = False
    _, inv, cnt = np.unique(tok, return_inverse=True, return_counts=True)
    idx = np.flatnonzero((cnt[inv] >= MIN_TYPE_OCC) & (pos >= MIN_POS))
    V = np.ascontiguousarray(np.asarray(X[idx], dtype=np.float32)[:, keep_c])
    t, s, p = tok[idx], seq[idx], pos[idx]
    groups = [np.unique(a, return_inverse=True)[1] for a in (t, s, p)]
    for _ in range(n_iter):                       # type, document, position; v1 = one cycle
        for g in groups:
            demean(V, g)
    assert np.isfinite(V).all()
    return dict(V=V, tok=t, seq=s, pos=p, row=idx, tinv=groups[0], tag=tag, L=L, n_iter=n_iter)


def pca_white(V, n=N_PCA):
    P = e61.pca_fit(V, n)
    return e61.pca_apply(P, V).astype(np.float32), P


def inv_sqrt(S):
    w, U = np.linalg.eigh(S)
    w = np.maximum(w, 1e-6 * w.max())
    return (U / np.sqrt(w)) @ U.T


def group_sorted(tinv):
    n_t = np.bincount(tinv)
    order = np.argsort(tinv, kind="stable")
    start = np.cumsum(n_t) - n_t
    return n_t, order, start


def type_whiten(Z, tinv, n_min, n0):
    n_t, order, start = group_sorted(tinv)
    Z = Z.astype(np.float64)
    mu = np.stack([np.bincount(tinv, weights=Z[:, j], minlength=len(n_t)) for j in range(Z.shape[1])], 1)
    Zc = Z - (mu / n_t[:, None])[tinv]
    Sp = Zc.T @ Zc / len(Zc)                      # pooled within-type covariance
    out = Zc @ inv_sqrt(Sp)
    big = np.flatnonzero(n_t >= n_min)
    for t in big:
        ii = order[start[t]:start[t] + n_t[t]]
        St = Zc[ii].T @ Zc[ii] / n_t[t]
        lam = n0 / (n_t[t] + n0)
        out[ii] = Zc[ii] @ inv_sqrt((1 - lam) * St + lam * Sp)
    out -= out.mean(0)
    w, U = np.linalg.eigh(np.cov(out[::2].T))
    o = np.argsort(w)[::-1]
    info = dict(n_types_whitened=int(len(big)), row_share_whitened=float(n_t[big].sum() / len(Z)))
    return (out @ U[:, o] / np.sqrt(w[o])).astype(np.float32), info


# ---------------------------------------------------------------------------- diagnostics
def partner(tinv, rng, exclude_self=True):
    """A random same-type occurrence for every row (self excluded unless exclude_self=False)."""
    n_t, order, start = group_sorted(tinv)
    rank = np.empty(len(tinv), np.int64); rank[order] = np.arange(len(tinv)) - start[tinv[order]]
    nt = n_t[tinv]
    if exclude_self:
        j = rng.integers(0, nt - 1); j = j + (j >= rank)
    else:
        j = rng.integers(0, nt)
    return order[start[tinv] + j]


def tdi(lab, K, tinv, seed=0, exclude_self=True):
    rep = partner(tinv, np.random.default_rng(seed), exclude_self)
    size = np.bincount(lab, minlength=K).astype(float)
    same = np.bincount(lab, weights=(lab[rep] == lab).astype(float), minlength=K)
    return np.where(size > 0, (same / np.maximum(size, 1)) / (size / len(lab)), np.nan)


def kir(W, tinv, seed=0, n_q=4000, k=10, chunk=250):
    rng = np.random.default_rng(seed)
    q = rng.choice(len(W), n_q, replace=False)
    sq = (W.astype(np.float32) ** 2).sum(1)
    n_t = np.bincount(tinv)
    obs = np.zeros(n_q)
    for a in range(0, n_q, chunk):
        qq = q[a:a + chunk]
        d = sq[qq][:, None] - 2 * (W[qq] @ W.T) + sq[None, :]
        d[np.arange(len(qq)), qq] = np.inf
        nn = np.argpartition(d, k, axis=1)[:, :k]
        obs[a:a + chunk] = (tinv[nn] == tinv[qq][:, None]).mean(1)
    exp_ = (n_t[tinv[q]] - 1) / (len(W) - 1)
    return float(obs.mean() / exp_.mean())


def probe(W, y, seq):
    from sklearn.linear_model import LogisticRegression
    ev = seq % 2 == 0
    clf = LogisticRegression(C=1.0, max_iter=3000).fit(W[ev], y[ev] > 0.5)
    auc = e61.auroc(clf.decision_function(W[~ev]), y[~ev])
    return float(auc), clf.coef_.ravel()


def moments(x):
    x = (x - x.mean()) / x.std()
    return float((x ** 3).mean()), float((x ** 4).mean() - 3)


# ---------------------------------------------------------------------------- units
def skew_g(x):
    return x ** 2, (2 * x).mean(axis=-1)


def fit_km(W, K, seed):
    from sklearn.cluster import MiniBatchKMeans
    return MiniBatchKMeans(n_clusters=K, batch_size=4096, n_init=3, max_iter=200,
                           random_state=seed).fit(W)


def fit_ica(W, n, contrast, seed):
    """Up to 3 attempts (seed, seed+100, seed+200); None if every attempt is non-finite."""
    from sklearn.decomposition import FastICA
    for k in range(3):
        try:
            ica = FastICA(n_components=n, whiten="unit-variance", random_state=seed + 100 * k, max_iter=400,
                          fun="logcosh" if contrast == "logcosh" else skew_g).fit(W[::2])
        except (ValueError, np.linalg.LinAlgError, FloatingPointError):
            continue
        if np.isfinite(ica.mixing_).all() and np.isfinite(ica.components_).all():
            return ica
    return None


def centroids(W, lab, K):
    C = np.zeros((K, W.shape[1])); np.add.at(C, lab, W)
    return C / np.maximum(np.bincount(lab, minlength=K), 1)[:, None]


def dir_match(Mx, My):
    A = Mx.T / np.linalg.norm(Mx.T, axis=1, keepdims=True)
    B = My.T / np.linalg.norm(My.T, axis=1, keepdims=True)
    return np.abs(A @ B.T).max(1)


def cluster_units(W, Dset, K):
    km = fit_km(W, K, 0)
    lab = km.labels_.astype(np.int64)
    ev = Dset["seq"] % 2 == 0
    C = centroids(W, lab, K)
    ce = centroids(W[ev], fit_km(W[ev], K, 1).labels_, K)
    co = centroids(W[~ev], fit_km(W[~ev], K, 2).labels_, K)
    re_e, _ = e61.cos_match(C, ce); re_o, _ = e61.cos_match(C, co)
    t_ex = tdi(lab, K, Dset["tinv"], 0, True)
    t_in = tdi(lab, K, Dset["tinv"], 0, False)
    units = []
    for c in range(K):
        mem = lab == c
        if not mem.any():
            continue
        tt, tc = np.unique(Dset["tok"][mem], return_counts=True)
        units.append(dict(kind="cluster", id=c, size=int(mem.sum()), n_types=int(len(tt)),
                          max_type_share=float(tc.max() / mem.sum()),
                          n_docs=int(len(np.unique(Dset["seq"][mem]))),
                          repro=float(min(re_e[c], re_o[c])), tdi=float(t_ex[c]),
                          tdi_self_incl=float(t_in[c])))
    return lab, units


def eta2(v, tinv, cnt):
    mt = np.bincount(tinv, weights=v) / cnt
    return float(((mt[tinv] - v.mean()) ** 2).sum() / max(((v - v.mean()) ** 2).sum(), 1e-12))


def ica_units(W, Dset, n, contrast):
    """A failed main fit gives no units; a failed half fit gives reproducibility 0."""
    ica = fit_ica(W, n, contrast, 0)
    if ica is None:
        return np.zeros((len(W), 0), np.float32), np.zeros((W.shape[1], 0)), []
    Y = ica.transform(W)
    ev = Dset["seq"] % 2 == 0
    halves = [fit_ica(W[ev], n, contrast, 1), fit_ica(W[~ev], n, contrast, 2)]
    rep = np.min([dir_match(ica.mixing_, h.mixing_) if h is not None else np.zeros(n) for h in halves], 0)
    tinv = Dset["tinv"]; cnt = np.bincount(tinv)
    units = []
    for j in range(n):
        y = Y[:, j]
        sk, ku = moments(y)
        units.append(dict(kind=f"ica_{contrast}", id=j, bimodality=e61.bimodality(y[::3]),
                          eta2_token=eta2(y, tinv, cnt), eta2_token_sq=eta2(y ** 2, tinv, cnt),
                          eta2_token_abs=eta2(np.abs(y), tinv, cnt), repro=float(rep[j]), skew=sk,
                          ex_kurtosis=ku))
    return Y.astype(np.float32), ica.mixing_, units


def cl_pass(u, tdi_cap, crep):
    return (u["size"] >= CL_FIXED["min_size"] and u["n_types"] >= CL_FIXED["min_types"]
            and u["max_type_share"] <= CL_FIXED["max_type_share"] and u["n_docs"] >= CL_FIXED["min_docs"]
            and u["repro"] >= crep and u["tdi"] <= tdi_cap)


def ica_pass(u, irep, bimod_on):
    return (u["eta2_token"] <= ICA_MAX_ETA2 and u["eta2_token_sq"] <= ICA_MAX_ETA2_SQ
            and u["repro"] >= irep and (u["bimodality"] > 0.555 if bimod_on else True))


# ---------------------------------------------------------------------------- positive control
def null_engine(labs, Ys, y_by_label, tinv, n_perm=N_PERM, seed=99):
    """Per-unit F1 (each cluster partition) and |AUROC| (each directional fit), observed and under
    n_perm token-matched label permutations (labels permuted among occurrences of one type)."""
    from scipy.stats import rankdata
    n = len(tinv)
    slots = np.argsort(tinv, kind="stable")
    sizes = {k: np.bincount(l, minlength=int(l.max()) + 1).astype(float) for k, l in labs.items()}
    ranks = {k: np.stack([rankdata(Y[:, j]) for j in range(Y.shape[1])], 1) for k, Y in Ys.items()
             if Y.shape[1] > 0}                  # a kind whose fit failed has no units

    def per_unit(pos):
        n1 = int(pos.sum()); n0 = n - n1
        f1 = {k: 2 * np.bincount(l[pos], minlength=len(sizes[k])) / (sizes[k] + n1) for k, l in labs.items()}
        au = {}
        for k, R in ranks.items():
            a = (R[pos].sum(0) - n1 * (n1 + 1) / 2) / (n1 * n0)
            au[k] = np.maximum(a, 1 - a)
        return f1, au

    out = {}
    for name, y in y_by_label.items():
        y0 = y > 0.5
        obs = per_unit(y0)
        nul = []
        rng = np.random.default_rng(seed)          # same permutations for every label
        for _ in range(n_perm):
            src = np.lexsort((rng.random(n), tinv))
            yp = np.empty_like(y0); yp[slots] = y0[src]
            nul.append(per_unit(yp))
        out[name] = dict(obs=obs, null=nul, base_rate=float(y0.mean()))
    return out


def evaluate(eng, name, kind_key, sel, floor, is_f1):
    """Max over the selected units vs the p99.9 of the max under the null."""
    e = eng[name]
    part = 0 if is_f1 else 1
    sel = np.asarray(sel, int)
    if len(sel) == 0 or kind_key not in e["obs"][part]:
        return dict(stat=0.0 if is_f1 else 0.5, null_p999=0.0 if is_f1 else 0.5, best=None,
                    n_units=0, floor=floor, recovered=False, margin=-1.0)
    v = e["obs"][part][kind_key][sel]
    nm = np.asarray([x[part][kind_key][sel].max() for x in e["null"]])
    p = float(np.percentile(nm, PC_PCT))
    st = float(v.max())
    return dict(stat=st, null_p999=p, best=int(sel[int(np.argmax(v))]), n_units=int(len(sel)),
                floor=floor, recovered=bool(st >= floor and st > p), margin=float(st - max(floor, p)))


def specificity(score, feats, target, Dset, tk, full, n_ctx=12, seed=5):
    """Reported for a recovering unit (addendum A2 section 4): battery associations with base
    rates, label composition of the top 1% (oriented toward the target), sampled contexts."""
    from scipy.stats import spearmanr
    a = e61.auroc(score, feats[target]); score = score if a >= 0.5 else -score
    top = np.argsort(-score)[:int(np.ceil(0.01 * len(score)))]
    assoc = {}
    for k, v in feats.items():
        if set(np.unique(v)) <= {0.0, 1.0}:
            a = e61.auroc(score, v)
            assoc[k] = dict(auroc=float(max(a, 1 - a)), base_rate=float(v.mean()), top1pct_rate=float(v[top].mean()))
        else:
            assoc[k] = dict(abs_spearman=float(abs(spearmanr(score[::3], v[::3]).statistic)))
    rng = np.random.default_rng(seed); ctx = []
    for i in rng.choice(top, size=min(n_ctx, len(top)), replace=False):
        r = int(Dset["row"][i]); s0 = (r // 255) * 255
        ctx.append(tk.decode([int(x) for x in full[max(s0, r - 12):r]]) + "[[" + tk.decode([int(full[r])]) + "]]"
                   + tk.decode([int(x) for x in full[r + 1:min(s0 + 255, r + 4)]]))
    order = sorted(assoc, key=lambda k: -assoc[k].get("auroc", assoc[k].get("abs_spearman", 0)))
    return dict(top_association=order[0], associations={k: assoc[k] for k in order}, top1pct_n=int(len(top)),
                contexts=ctx)


def auc_floor(probe_auc):
    return float(np.clip(0.5 + 0.75 * (probe_auc - 0.5), 0.70, 0.85))


# ---------------------------------------------------------------------------- pilot (P2 only)
def pilot(layers, tag, outp=None, choice_path=None):
    assert hashlib.sha256(PLAN.read_bytes()).hexdigest() == PLAN_SHA, "calibration plan changed"
    tk = e61.token_strings()
    full = np.load(EXT / f"cache/occ/{tag}_meta.npz")["tokens"]
    outp = Path(outp) if outp else EXT / "results/e63_pilot.json"
    choice_path = Path(choice_path) if choice_path else CHOICE
    res = dict(plan=str(PLAN.relative_to(EXT)), plan_sha256=PLAN_SHA, set=tag, layers={},
               addenda=["docs/stage4v2_calibration_addendum_A1_2026-10-15.md"])
    for L in layers:
        t0 = time.time()
        rec = dict(stepA=[], stepB=None)
        # ---- Step A
        B = represent_base(tag, L, n_iter=1)
        feats = e61.feature_battery(B, tk, full, None)
        y_par = feats["paren_inside"]
        W0, _ = pca_white(B["V"]); del B["V"]
        lab0 = fit_km(W0, 128, 0).labels_.astype(np.int64)
        a0, _ = probe(W0, y_par, B["seq"])
        rec["stepA"].append(dict(setting="B0_v1", tdi_med=float(np.nanmedian(tdi(lab0, 128, B["tinv"]))),
                                 tdi_self_incl_med=float(np.nanmedian(tdi(lab0, 128, B["tinv"], 0, False))),
                                 kir=kir(W0, B["tinv"]), probe_paren=a0))
        print(f"L{L} A {rec['stepA'][-1]}", flush=True)
        del W0
        Dset = represent_base(tag, L, n_iter=N_ITER)
        assert np.array_equal(Dset["row"], B["row"])
        Z, P = pca_white(Dset["V"]); del Dset["V"]
        rec["pca_var_frac"] = P["var_frac"]
        cands = [("B1_iter", None, None, Z, {})]
        for nm, n0 in itertools.product(GRID_NMIN, GRID_N0):
            Wt, info = type_whiten(Z, Dset["tinv"], nm, n0)
            cands.append((f"tw_N{nm}_n{n0}", nm, n0, Wt, info))
        for name, nm, n0, W, info in cands:
            lab = fit_km(W, 128, 0).labels_.astype(np.int64)
            a, wvec = probe(W, y_par, Dset["seq"])
            row = dict(setting=name, N_MIN=nm, n0=n0, tdi_med=float(np.nanmedian(tdi(lab, 128, Dset["tinv"]))),
                       tdi_self_incl_med=float(np.nanmedian(tdi(lab, 128, Dset["tinv"], 0, False))),
                       kir=kir(W, Dset["tinv"]), probe_paren=a, **info)
            rec["stepA"].append(row)
            print(f"L{L} A {row}", flush=True)
        guard = rec["stepA"][1]["probe_paren"] - 0.02
        ok = [r for r in rec["stepA"][2:] if r["probe_paren"] >= guard]
        if ok:
            ch = min(ok, key=lambda r: (r["tdi_med"], -r["n0"], -r["N_MIN"]))
        else:
            ch = rec["stepA"][1]
        rec["chosen_whitening"] = dict(setting=ch["setting"], N_MIN=ch["N_MIN"], n0=ch["n0"],
                                       probe_paren=ch["probe_paren"], auc_floor=auc_floor(ch["probe_paren"]),
                                       guard=guard)
        W = next(c[3] for c in cands if c[0] == ch["setting"])
        del cands
        print(f"L{L} chosen {rec['chosen_whitening']}", flush=True)
        # ---- Step B
        _, wvec = probe(W, y_par, Dset["seq"])
        rec["probe_direction_moments"] = dict(zip(("skew", "ex_kurtosis"), moments(W @ wvec)))
        labs, Ys, units = {}, {}, {}
        for K in GRID_K:
            labs[K], units[f"km{K}"] = cluster_units(W, Dset, K)
        for c, n in itertools.product(CONTRASTS, GRID_NICA):
            Ys[(c, n)], _, units[f"{c}{n}"] = ica_units(W, Dset, n, c)
        print(f"L{L} units fitted {time.time()-t0:.0f}s", flush=True)
        eng = null_engine(labs, Ys, {k: feats[k] for k in LABELS}, Dset["tinv"], n_perm=N_PERM)
        fl = rec["chosen_whitening"]["auc_floor"]
        grid = []
        for K, nl, ns in itertools.product(GRID_K, GRID_NICA, GRID_NICA):
            for tc, cr, ir, bm in itertools.product(GRID_TDI, GRID_CREP, GRID_IREP, GRID_BIMOD):
                cs = [u["id"] for u in units[f"km{K}"] if cl_pass(u, tc, cr)]
                il = [u["id"] for u in units[f"logcosh{nl}"] if ica_pass(u, ir, bm)]
                isk = [u["id"] for u in units[f"skew{ns}"] if ica_pass(u, ir, bm)]
                row = dict(K=K, N_logcosh=nl, N_skew=ns, tdi_cap=tc, cluster_repro=cr, ica_repro=ir,
                           bimodality=bm, n_screened=dict(cluster=len(cs), logcosh=len(il), skew=len(isk)))
                for lb in LABELS:
                    r = dict(cluster=evaluate(eng, lb, K, cs, F1_FLOOR, True),
                             logcosh=evaluate(eng, lb, ("logcosh", nl), il, fl, False),
                             skew=evaluate(eng, lb, ("skew", ns), isk, fl, False))
                    r["recovered"] = any(v["recovered"] for v in r.values())
                    r["margin"] = max(v["margin"] for v in r.values() if isinstance(v, dict))
                    row[lb] = r
                grid.append(row)
        # unscreened upper bounds (every unit of the kind)
        ub = {}
        for lb in LABELS:
            ub[lb] = dict(cluster={K: evaluate(eng, lb, K, range(K), F1_FLOOR, True) for K in GRID_K},
                          ica={f"{c}{n}": evaluate(eng, lb, (c, n), range(n), fl, False)
                               for c, n in itertools.product(CONTRASTS, GRID_NICA)})
        top_moments = {k: sorted(((u["skew"], u["ex_kurtosis"]) for u in v), key=lambda x: -abs(x[0]))[:5]
                       for k, v in units.items() if not k.startswith("km")}
        rec["stepB"] = dict(grid=grid, unscreened=ub, ica_top_moments=top_moments,
                            screen_counts={k: summarize_units(v) for k, v in units.items()})
        rec["seconds"] = round(time.time() - t0)
        res["layers"][str(L)] = rec
        outp.write_text(json.dumps(res, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
        print(f"L{L} pilot done {rec['seconds']}s", flush=True)
        del W, labs, Ys, eng
    # ---- choice across layers (plan, Step B)
    keys = {}
    for L, rec in res["layers"].items():
        for r in rec["stepB"]["grid"]:
            k = tuple(r[x] for x in ("K", "N_logcosh", "N_skew", "tdi_cap", "cluster_repro", "ica_repro", "bimodality"))
            keys.setdefault(k, []).append(r["paren_inside"])
    def key(kv):
        k, rs = kv
        return (sum(r["recovered"] for r in rs), -k[3], k[4], k[5], int(k[6]), float(np.mean([r["margin"] for r in rs])))
    best = max(keys.items(), key=key)
    nrec = sum(r["recovered"] for r in best[1])
    names = ("K", "N_logcosh", "N_skew", "tdi_cap", "cluster_repro", "ica_repro", "bimodality")
    units_cfg = dict(zip(names, best[0])) if nrec else dict(DEFAULT)
    choice = dict(units=units_cfg, n_layers_recovered_on_P2=nrec,
                  note=None if nrec else "nothing recovers on P2: plan defaults frozen",
                  whitening={L: rec["chosen_whitening"] for L, rec in res["layers"].items()})
    res["choice"] = choice
    outp.write_text(json.dumps(res, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    choice_path.write_text(json.dumps(choice, indent=1))
    print("CHOICE", json.dumps(choice), flush=True)


def summarize_units(units):
    if not units:
        return dict(n=0)
    if units[0]["kind"] == "cluster":
        t = np.asarray([u["tdi"] for u in units]); r = np.asarray([u["repro"] for u in units])
        return dict(n=len(units), tdi_med=float(np.median(t)), tdi_q90=float(np.quantile(t, 0.9)),
                    tdi_le3=int((t <= 3).sum()), tdi_le5=int((t <= 5).sum()), repro_ge08=int((r >= 0.8).sum()),
                    fixed_pass=int(sum(u["size"] >= 300 and u["n_types"] >= 30 and u["max_type_share"] <= 0.3
                                       and u["n_docs"] >= 100 for u in units)))
    e = np.asarray([u["eta2_token"] for u in units]); r = np.asarray([u["repro"] for u in units])
    b = np.asarray([u["bimodality"] for u in units])
    e2 = np.asarray([u.get("eta2_token_sq", np.nan) for u in units])
    return dict(n=len(units), eta2_le03=int((e <= 0.3).sum()), eta2sq_le03=int((e2 <= 0.3).sum()),
                eta2sq_max=float(np.nanmax(e2)) if np.isfinite(e2).any() else None, repro_ge08=int((r >= 0.8).sum()),
                repro_ge05=int((r >= 0.5).sum()), repro_med=float(np.median(r)), bimod_gt=int((b > 0.555).sum()))


# ---------------------------------------------------------------------------- D2: freeze / control
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["pilot", "freeze", "control"], required=True)
    ap.add_argument("--layers", required=True)
    ap.add_argument("--P", default="occP2"); ap.add_argument("--D", default="occD2")
    ap.add_argument("--out", default=str(EXT / "results/e63_occurrence_v2.json"))
    ap.add_argument("--test-dir", default=None, help="dry run: prereg, HASHES, units and out in this dir")
    a = ap.parse_args()
    layers = [int(x) for x in a.layers.split(",")]
    if a.phase == "pilot":
        return pilot(layers, a.P)
    prereg, hashes_p, res_dir = PREREG, EXT / "docs/HASHES.txt", EXT / "results"
    if a.test_dir:
        td = Path(a.test_dir); prereg, hashes_p, res_dir = td / "PREREG.md", td / "HASHES.txt", td
        a.out = str(td / "out.json")
    sha = hashlib.sha256(prereg.read_bytes()).hexdigest()
    hashes = hashes_p.read_text()
    assert sha in hashes, "PREREG_occurrence_v2.md does not match the frozen sha256 in docs/HASHES.txt"
    choice = json.loads(CHOICE.read_text())
    assert hashlib.sha256(CHOICE.read_bytes()).hexdigest() in prereg.read_text(), "pilot choice not the frozen one"
    U = choice["units"]
    outp = Path(a.out)
    res = json.loads(outp.read_text()) if outp.exists() else dict(prereg_sha256=sha, choice=choice, layers={})
    assert res["prereg_sha256"] == sha
    tk = e61.token_strings()
    for L in layers:
        t0 = time.time()
        wh = choice["whitening"][str(L)]
        Dset = represent_base(a.D, L, N_ITER)
        Z, _ = pca_white(Dset["V"]); del Dset["V"]
        W = Z if wh["N_MIN"] is None else type_whiten(Z, Dset["tinv"], wh["N_MIN"], wh["n0"])[0]
        npz = res_dir / f"occ_units_v2_L{L}.npz"
        if a.phase == "freeze":
            lab, cl = cluster_units(W, Dset, U["K"])
            Yl, Ml, ul = ica_units(W, Dset, U["N_logcosh"], "logcosh")
            Ys, Ms, us = ica_units(W, Dset, U["N_skew"], "skew")
            for u in cl:
                u["screened"] = bool(cl_pass(u, U["tdi_cap"], U["cluster_repro"]))
            for u in ul + us:
                u["screened"] = bool(ica_pass(u, U["ica_repro"], U["bimodality"]))
            np.savez_compressed(npz, lab=lab.astype(np.int16), row=Dset["row"], Y_logcosh=Yl, Y_skew=Ys,
                                mix_logcosh=Ml, mix_skew=Ms)
            fr = dict(layer=L, whitening=wh, n_rows=int(len(W)), units=cl + ul + us,
                      tdi_med=float(np.nanmedian([u["tdi"] for u in cl])),
                      tdi_self_incl_med=float(np.nanmedian([u["tdi_self_incl"] for u in cl])),
                      kir=kir(W, Dset["tinv"]),
                      screen_counts=dict(cluster=summarize_units(cl), logcosh=summarize_units(ul),
                                         skew=summarize_units(us)),
                      n_screened=dict(cluster=sum(u["screened"] for u in cl), logcosh=sum(u["screened"] for u in ul),
                                      skew=sum(u["screened"] for u in us)),
                      units_npz_sha256=hashlib.sha256(npz.read_bytes()).hexdigest(),
                      frozen_at=time.strftime("%Y-%m-%dT%H:%M:%S"))
            fr["freeze_md5"] = hashlib.md5(json.dumps(fr, sort_keys=True, default=str).encode()).hexdigest()
            res["layers"][str(L)] = dict(frozen=fr)
        else:
            fr = res["layers"][str(L)]["frozen"]
            assert hashlib.sha256(npz.read_bytes()).hexdigest() == fr["units_npz_sha256"], "units changed after freeze"
            z = np.load(npz)
            assert np.array_equal(z["row"], Dset["row"])
            full = np.load(EXT / f"cache/occ/{a.D}_meta.npz")["tokens"]
            feats = e61.feature_battery(Dset, tk, full, None)
            lab = z["lab"].astype(np.int64)
            labs = {U["K"]: lab}
            Ys = {("logcosh", U["N_logcosh"]): z["Y_logcosh"], ("skew", U["N_skew"]): z["Y_skew"]}
            eng = null_engine(labs, Ys, {k: feats[k] for k in LABELS}, Dset["tinv"], n_perm=N_PERM)
            un = fr["units"]
            cs = [u["id"] for u in un if u["kind"] == "cluster" and u["screened"]]
            il = [u["id"] for u in un if u["kind"] == "ica_logcosh" and u["screened"]]
            isk = [u["id"] for u in un if u["kind"] == "ica_skew" and u["screened"]]
            fl = wh["auc_floor"]
            pc = {}
            for lb in LABELS:
                r = dict(cluster=evaluate(eng, lb, U["K"], cs, F1_FLOOR, True),
                         logcosh=evaluate(eng, lb, ("logcosh", U["N_logcosh"]), il, fl, False),
                         skew=evaluate(eng, lb, ("skew", U["N_skew"]), isk, fl, False),
                         unscreened=dict(cluster=evaluate(eng, lb, U["K"], range(U["K"]), F1_FLOOR, True),
                                         logcosh=evaluate(eng, lb, ("logcosh", U["N_logcosh"]), range(U["N_logcosh"]), fl, False),
                                         skew=evaluate(eng, lb, ("skew", U["N_skew"]), range(U["N_skew"]), fl, False)))
                r["recovered"] = any(r[k]["recovered"] for k in ("cluster", "logcosh", "skew"))
                r["probe_auroc_context"], _ = probe(W, feats[lb], Dset["seq"])
                for kd in ("logcosh", "skew"):
                    if r[kd]["recovered"]:
                        r[kd]["specificity"] = specificity(z[f"Y_{kd}"][:, r[kd]["best"]].astype(np.float64),
                                                           feats, lb, Dset, tk, full)
                if r["cluster"]["recovered"]:
                    m = (lab == r["cluster"]["best"]).astype(np.float64)
                    r["cluster"]["specificity"] = specificity(m + 1e-9 * np.arange(len(m)) / len(m),
                                                              feats, lb, Dset, tk, full)
                pc[lb] = r
            pc["PASS"] = bool(any(pc[lb]["recovered"] for lb in LABELS))
            res["layers"][str(L)]["positive_control"] = pc
        res["layers"][str(L)][f"seconds_{a.phase}"] = round(time.time() - t0, 1)
        outp.write_text(json.dumps(res, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
        print(f"L{L} {a.phase} done in {time.time()-t0:.0f}s", flush=True)
    if a.phase == "control":
        res["PASS"] = bool(any(res["layers"][str(L)]["positive_control"]["PASS"] for L in layers))
        outp.write_text(json.dumps(res, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
        print("POSITIVE CONTROL", "PASS" if res["PASS"] else "FAIL", flush=True)


if __name__ == "__main__":
    main()
