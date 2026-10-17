"""e72 -- raw-residual pilot (NOTE_raw_residual_pilot.md, hashed before any number it governs). Burned D2 only, CPU only.

Phases
  gate      raw L12 D2 kept rows (results/e67_raw_L12_rows.npy, fp16; coordinate 1038 dropped) through the frozen v2
            instrument unchanged (demean3, PCA-64 whitened, per-type equalisation N_MIN 32 / n0 16, re-whitening,
            full-rank FastICA-64 logcosh and skew, frozen screens) -> results/e72_units_raw_L12.npz, hashed before any
            label; then the paren/quote positive control vs the token-matched null -> results/e72_gate.json (hashed).
            PASS iff both labels are recovered; on FAIL the pilot stops.
  coverage  after a hashed PASS: e = h - SAE(h) (L12 resid_post_all 16k l0_big, e69.error_rows); e and SAE(h) pushed
            through the same fixed maps; error share of every unit vs 10,000 random W-space directions; screens; match
            to e64's decoded-span units; lead rule; naming and S2 (descriptive)
            -> results/e72_coverage.json + results/e72_coverage_arrays.npz (hashed)

--test-dir DIR: synthetic smoke test. Reads DIR/H_D.npy, DIR/H_S.npy, DIR/sae.npz; hashes into DIR/HASHES.txt; writes
every output to DIR. The real D2/S2 metadata and tokenizer are used, no real activation.
"""
import argparse, gc, hashlib, importlib.util, json, os, sys, time
from pathlib import Path

import numpy as np

EXT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXT / "src"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_HOME", str(EXT / "hf_home"))

NOTE = "NOTE_raw_residual_pilot.md"
SELF = "scripts/e72_raw_residual_pilot.py"
LAYER, DROP, N_ROWS_D2, N_ROWS_S2 = 12, 1038, 203_328, 202_866
KINDS, N_ICA, N_MIN, N0, N_ITER = ("logcosh", "skew"), 64, 32, 16, 3
ICA_REPRO, AUC_FLOOR, N_PERM = 0.8, 0.70, 1000
CTRL_LABELS = ("paren_inside", "quote_inside")
SAE_BIG = ("resid_post_all", 12, "16k", "big")
N_RAND, SEED_RAND, SEED_RAND_S2 = 10_000, 7200, 7201
SAME_COS, STATE_RHO, POS_RHO, PRESENCE_COS = 0.8, 0.3, 0.3, 0.8
N_TOP_NAME = 5
E64C_RULES = ("A", "B", "C", "C_inat", "D", "E", "F", "F_multi", "G", "H", "I", "J", "K", "L", "M", "N")
KEEP = np.ones(1152, bool); KEEP[DROP] = False
UNITS, GATE, COV, COV_NPZ = "e72_units_raw_L12.npz", "e72_gate.json", "e72_coverage.json", "e72_coverage_arrays.npz"


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, EXT / rel)
    mod = importlib.util.module_from_spec(spec); sys.modules[name] = mod; spec.loader.exec_module(mod)
    return mod


e64 = _load("e64", "scripts/e64_occurrence_v2_discovery.py")
e61, e63 = e64.e61, e64.e63
e69 = _load("e69", "scripts/e69_sae_gap_pilot.py")
T0 = time.time()


def t():
    return f"[{time.time() - T0:.0f}s]"


class Cfg:
    def __init__(self, test_dir=None):
        self.test = test_dir is not None
        if self.test:
            td = Path(test_dir).resolve()
            self.H_D, self.H_S, self.out, self.sae = td / "H_D.npy", td / "H_S.npy", td, td / "sae.npz"
            e64.HASHES = td / "HASHES.txt"            # e64.append_hash / require_hashed write and read here
        else:
            self.H_D, self.H_S = EXT / "results/e67_raw_L12_rows.npy", EXT / "results/e69_p1_S2_raw_L12_rows.npy"
            self.out, self.sae = EXT / "results", None

    def p(self, name):
        return self.out / name

    def rel(self, name):
        """Path handed to e64.append_hash / require_hashed (EXT / absolute = absolute in test mode)."""
        p = self.p(name)
        return str(p) if self.test else str(p.relative_to(EXT))

    def params(self):
        if self.test:
            z = np.load(self.sae)
            return {k: z[k] for k in z.files}
        return e69.sae_params(SAE_BIG)


def dump(obj, path):
    Path(path).write_text(json.dumps(obj, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)))


def integrity(cfg):
    if cfg.test:
        return
    e64.require_hashed(NOTE, SELF, "results/e67_raw_L12_rows.npy", "results/e69_p1_S2_raw_L12_rows.npy",
                       "results/occ_units_v2_L12.npz", "results/e64_maps_L12.npz", "scripts/e69_sae_gap_pilot.py")
    e64.check_integrity()                              # e64 frozen version; e61/e63/e60/choice/v2 units unchanged
    ch = json.loads((EXT / "results/e63_pilot_choice.json").read_text())
    assert ch["units"]["ica_repro"] == ICA_REPRO and ch["units"]["bimodality"] is False
    wh = ch["whitening"][str(LAYER)]
    assert (wh["N_MIN"], wh["n0"], wh["auc_floor"]) == (N_MIN, N0, AUC_FLOOR), wh
    assert e61.drop_for(LAYER) == [DROP]
    assert (e64.N_MIN, e64.N0, e64.N_ITER, e64.N_ICA) == (N_MIN, N0, N_ITER, N_ICA)


# ---------------------------------------------------------------------------- representation (e63.represent_base on raw h)
def rows(tag, n_expect):
    D, full = e64.rows_only(tag)                       # e63's kept-row rule on the set's own meta
    assert len(D["row"]) == n_expect, (tag, len(D["row"]))
    D["groups"] = [np.unique(a, return_inverse=True)[1] for a in (D["tok"], D["seq"], D["pos"])]
    assert np.array_equal(D["groups"][0], D["tinv"])
    return D, full


def demean3(V, groups):
    """In place on float32 V (n, d): type, document, position means, cycled N_ITER times (e63.represent_base)."""
    for _ in range(N_ITER):
        for g in groups:
            e63.demean(V, g)
    assert np.isfinite(V).all()
    return V


def raw_V(H16, groups):
    V = np.empty((len(H16), int(KEEP.sum())), np.float32)
    for a in range(0, len(H16), 16384):
        V[a:a + 16384] = np.asarray(H16[a:a + 16384], dtype=np.float32)[:, KEEP]
    return demean3(V, groups)


def project(V, P, own_mean=True):
    """PCA projection with fixed U, sd: (V - mean) @ U / sd, in row blocks (float64 out)."""
    mu = V.mean(0, dtype=np.float64) if own_mean else np.asarray(P["mu"], np.float64)
    U, sd = np.asarray(P["U"], np.float64), np.asarray(P["sd"], np.float64)
    Z = np.empty((len(V), U.shape[1]))
    for a in range(0, len(V), 32768):
        Z[a:a + 32768] = ((V[a:a + 32768].astype(np.float64) - mu) @ U) / sd
    return Z


def heavy_sign(y):
    return e64.heavy_sign(y)


def sumsq(V):
    return float(sum(np.einsum("ij,ij->", b, b) for b in (V[a:a + 32768].astype(np.float64) for a in range(0, len(V), 32768))))


# ---------------------------------------------------------------------------- gate
def phase_gate(cfg):
    assert not cfg.p(UNITS).exists() and not cfg.p(GATE).exists(), "the gate runs once"
    D, full = rows("occD2", N_ROWS_D2)
    if not cfg.test:
        assert np.array_equal(D["row"], np.load(EXT / "results/occ_units_v2_L12.npz")["row"]), "rows differ from D2's frozen rows"
    H16 = np.load(cfg.H_D, mmap_mode="r")
    assert H16.shape == (len(D["row"]), 1152) and H16.dtype == np.float16
    V = raw_V(H16, D["groups"]); del H16
    vh_ss = sumsq(V)
    Z, P = e63.pca_white(V); del V; gc.collect()
    print(t(), f"raw V demean3 + PCA-64: var_frac {P['var_frac']:.3f}", flush=True)
    W, M = e64.fit_maps(Z, D["tok"], D["tinv"])
    W63 = e63.type_whiten(Z, D["tinv"], N_MIN, N0)[0]
    Wapp = e64.apply_maps(M, Z, D["tok"])
    consist = dict(fit_vs_e63=float(np.abs(W - W63).max()), apply_vs_fit=float(np.abs(Wapp - W).max()))
    assert consist["fit_vs_e63"] < 1e-4 and consist["apply_vs_fit"] < 1e-4, consist
    del W63, Wapp
    Dset = dict(seq=D["seq"], tinv=D["tinv"], tok=D["tok"], row=D["row"])
    n = len(W)
    X1 = np.hstack([W.astype(np.float64), np.ones((n, 1))])
    Zc = Z.astype(np.float64) - Z.mean(0)
    Pu, sd = np.asarray(P["U"]), np.asarray(P["sd"])
    out = dict(row=D["row"], W=W, Z=Z.astype(np.float32), P_mu=P["mu"], P_U=Pu, P_sd=sd, vh_ss=vh_ss, **M)
    units, ok_fit = {}, {}
    for kd in KINDS:
        Y, mix, us = e63.ica_units(W, Dset, N_ICA, kd)
        ok_fit[kd] = Y.shape[1] == N_ICA
        for u in us:
            u["screened"] = bool(e63.ica_pass(u, ICA_REPRO, False))
        units[kd] = us
        out[f"Y_{kd}"], out[f"mix_{kd}"] = Y, mix
        if not ok_fit[kd]:
            print(t(), f"{kd}: ICA fit failed (no units)", flush=True)
            continue
        Yd = Y.astype(np.float64)
        B, *_ = np.linalg.lstsq(X1, Yd, rcond=None)
        res_ = Yd - X1 @ B
        r2 = 1 - (res_ ** 2).sum(0) / ((Yd - Yd.mean(0)) ** 2).sum(0)
        assert r2.min() >= 0.999, f"{kd}: unit readout from W is not exact (min R^2 {r2.min():.5f})"
        Ys = (Yd - Yd.mean(0)) / Yd.std(0)
        sign = np.asarray([heavy_sign(Yd[:, j]) for j in range(N_ICA)])
        raw = Pu @ (sd[:, None] * ((Zc.T @ Ys) / n))                # e64 reconstruct: Cov(Z, y) pulled back
        dirs = np.zeros((N_ICA, 1152)); dirs[:, KEEP] = (raw * sign).T
        dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
        out.update({f"readout_{kd}": B, f"r2_{kd}": r2, f"pattern_{kd}": (W.astype(np.float64).T @ Ys) / n,
                    f"sign_{kd}": sign, f"dir_{kd}": dirs})
        print(t(), f"{kd}: {sum(u['screened'] for u in us)} of {N_ICA} screened; readout min R^2 {r2.min():.6f}", flush=True)
    np.savez_compressed(cfg.p(UNITS), **out)
    e64.append_hash(cfg.rel(UNITS), "e72 gate: raw L12 D2 units, maps, readouts, patterns (before any label)")
    del X1, Zc, out; gc.collect()
    # ---- positive control (labels computed only now)
    tk = e61.token_strings()
    feats = e61.feature_battery(Dset, tk, full, None)
    z = np.load(cfg.p(UNITS))
    Ys = {(kd, N_ICA): z[f"Y_{kd}"] for kd in KINDS if ok_fit[kd]}
    eng = e63.null_engine({}, Ys, {k: feats[k] for k in CTRL_LABELS}, D["tinv"], n_perm=N_PERM)
    pc, ctrl = {}, []
    for lb in CTRL_LABELS:
        r = {kd: e63.evaluate(eng, lb, (kd, N_ICA), [u["id"] for u in units[kd] if u["screened"]], AUC_FLOOR, False)
             for kd in KINDS}
        r["unscreened"] = {kd: e63.evaluate(eng, lb, (kd, N_ICA), range(N_ICA) if ok_fit[kd] else [], AUC_FLOOR, False)
                           for kd in KINDS}
        r["recovered"] = bool(any(r[kd]["recovered"] for kd in KINDS))
        r["probe_auroc_context"], _ = e63.probe(W, feats[lb], D["seq"])
        for kd in KINDS:
            if r[kd]["recovered"]:
                r[kd]["specificity"] = e63.specificity(z[f"Y_{kd}"][:, r[kd]["best"]].astype(np.float64), feats, lb, Dset, tk, full)
                ctrl.append((kd, int(r[kd]["best"]), lb, float(r[kd]["stat"])))
        pc[lb] = r
        print(t(), f"{lb}: logcosh {r['logcosh']['stat']:.3f} (p99.9 {r['logcosh']['null_p999']:.3f}) skew "
              f"{r['skew']['stat']:.3f} (p99.9 {r['skew']['null_p999']:.3f}) recovered {r['recovered']}; probe "
              f"{r['probe_auroc_context']:.3f}", flush=True)
    res = dict(note_sha256=e64.sha(EXT / NOTE), script_sha256=e64.sha(__file__), units_npz_sha256=e64.sha(cfg.p(UNITS)),
               n_rows=int(n), pca_var_frac=float(P["var_frac"]), vh_ss=vh_ss, consistency=consist, fit_ok=ok_fit,
               screen_counts={kd: e63.summarize_units(units[kd]) for kd in KINDS},
               n_screened={kd: sum(u["screened"] for u in units[kd]) for kd in KINDS},
               readout_min_r2={kd: float(z[f"r2_{kd}"].min()) for kd in KINDS if ok_fit[kd]},
               kir=e63.kir(W, D["tinv"]), positive_control=pc, control_units=ctrl,
               PASS=bool(all(pc[lb]["recovered"] for lb in CTRL_LABELS)),
               PASS_any_S4v2_rule=bool(any(pc[lb]["recovered"] for lb in CTRL_LABELS)),
               units=units, seconds=round(time.time() - T0))
    dump(res, cfg.p(GATE))
    e64.append_hash(cfg.rel(GATE), f"e72 gate: raw L12 positive control, {'PASS' if res['PASS'] else 'FAIL -> STOP'}")
    print(t(), "GATE", "PASS" if res["PASS"] else "FAIL -> STOP (instrument not validated on raw h)",
          f"(S4v2 any-of rule: {res['PASS_any_S4v2_rule']})", flush=True)


# ---------------------------------------------------------------------------- coverage
def covs(Wh, We, sel=None):
    A = Wh if sel is None else Wh[sel]
    Bm = We if sel is None else We[sel]
    A = A - A.mean(0); Bm = Bm - Bm.mean(0); n = len(A)
    return dict(hh=A.T @ A / n, ee=Bm.T @ Bm / n, eh=Bm.T @ A / n)


def es(C, G):
    """Error share of each column direction g: var(g.W_e)/var(g.W_h) and cov(g.W_e, g.W_h)/var(g.W_h)."""
    vh = np.einsum("ik,ij,jk->k", G, C["hh"], G)
    ve = np.einsum("ik,ij,jk->k", G, C["ee"], G)
    ce = np.einsum("ik,ij,jk->k", G, C["eh"], G)
    return ve / vh, ce / vh


def qs(x):
    x = np.asarray(x)
    return dict(p50=float(np.median(x)), p99=float(np.quantile(x, 0.99)), p999=float(np.quantile(x, 0.999)),
                max=float(x.max()), min=float(x.min()), mean=float(x.mean()))


def ranks(v):
    from scipy.stats import rankdata
    r = rankdata(v).astype(np.float64)
    return (r - r.mean()) / r.std()


def since_punct(meta_tokens, rows_, tk):
    """Tokens since the last punctuation piece (e70's definition, verbatim logic)."""
    out = np.zeros(len(rows_), np.float32); strs = {}; cnt = 0; ri = {int(r): i for i, r in enumerate(rows_)}
    for r, x in enumerate(meta_tokens):
        if r % 255 == 0:
            cnt = 0
        s = strs.setdefault(int(x), tk.decode([int(x)]))
        if r in ri:
            out[ri[r]] = cnt
        cnt = 0 if (s.strip() and not any(c.isalnum() for c in s.strip())) else cnt + 1
    return out


def error_W(H16, D, P, M, params):
    """e through the fixed maps: demean3 (same groups) -> PCA (own mean) -> apply_maps. Also state and FVU."""
    E, state = e69.error_rows(H16, params)
    # raw FVU over kept coordinates: sum ||e||^2 / sum ||h - mean h||^2
    s1 = np.zeros(int(KEEP.sum())); s2 = 0.0; se = 0.0
    for a in range(0, len(H16), 16384):
        x = np.asarray(H16[a:a + 16384], dtype=np.float64)[:, KEEP]
        s1 += x.sum(0); s2 += float((x * x).sum()); se += float((E[a:a + 16384].astype(np.float64) ** 2).sum())
    fvu_raw = se / (s2 - float((s1 ** 2).sum()) / len(H16))
    Ve = demean3(E, D["groups"])
    ve_ss = sumsq(Ve)
    Ze = project(Ve, P, own_mean=True); del Ve, E; gc.collect()
    We = e64.apply_maps(M, Ze, D["tok"]).astype(np.float64)
    st = np.stack([state["log_h"], state["log_e"], state["L0"]], 1).astype(np.float32)
    st = demean3(np.ascontiguousarray(st), D["groups"])
    return Ze, We, dict(log_h=st[:, 0], log_e=st[:, 1], L0=st[:, 2]), fvu_raw, ve_ss


def phase_coverage(cfg):
    from scipy.linalg import eigh as geigh
    from sklearn.metrics import roc_auc_score
    e64.require_hashed(cfg.rel(UNITS), cfg.rel(GATE))
    gate = json.loads(cfg.p(GATE).read_text())
    assert gate["PASS"], "the gate did not pass: STOP, no coverage number is computed"
    assert gate["units_npz_sha256"] == e64.sha(cfg.p(UNITS))
    assert not cfg.p(COV).exists(), "coverage runs once"
    z = np.load(cfg.p(UNITS))
    D, full = rows("occD2", N_ROWS_D2)
    assert np.array_equal(D["row"], z["row"])
    P = dict(mu=z["P_mu"], U=z["P_U"], sd=z["P_sd"])
    M = {k: z[k] for k in ("Sp_is", "type_ids", "type_isqrt", "Uo", "sw")}
    Zh = z["Z"].astype(np.float64); Wh = z["W"].astype(np.float64); n = len(Wh)
    Wapp = e64.apply_maps(M, Zh, D["tok"]).astype(np.float64)
    maps_vs_fit = float(np.abs(Wapp - Wh).max())
    assert maps_vs_fit < 1e-4, maps_vs_fit
    params = cfg.params()
    H16 = np.load(cfg.H_D)
    assert H16.shape == (n, 1152)
    Ze, We, state, fvu_raw, ve_ss = error_W(H16, D, P, M, params); del H16; gc.collect()
    Ws = e64.apply_maps(M, Zh - Ze, D["tok"]).astype(np.float64)
    lin = float(np.abs(Wapp - We - Ws).max()); del Ws, Wapp
    assert lin < 1e-4, f"maps(h) != maps(e) + maps(SAE(h)) ({lin})"
    pc_es = (Ze.var(0) / Zh.var(0)).tolist()
    print(t(), f"error rows: FVU raw {fvu_raw:.3f}, identity-removed {ve_ss / float(z['vh_ss']):.3f}; linearity {lin:.2e}", flush=True)
    even = D["seq"] % 2 == 0
    C = dict(all=covs(Wh, We), even=covs(Wh, We, even), odd=covs(Wh, We, ~even))
    G = np.random.default_rng(SEED_RAND).standard_normal((N_ICA, N_RAND)); G /= np.linalg.norm(G, axis=0, keepdims=True)
    R = {h: es(C[h], G) for h in C}
    rq = {h: dict(es=qs(R[h][0]), es_cov=qs(R[h][1])) for h in C}
    ceil = np.sort(geigh(C["all"]["ee"], C["all"]["hh"], eigvals_only=True))[::-1]
    ceil_cov = np.sort(geigh((C["all"]["eh"] + C["all"]["eh"].T) / 2, C["all"]["hh"], eigvals_only=True))[::-1]
    print(t(), f"random ES p50 {rq['all']['es']['p50']:.3f} p99.9 {rq['all']['es']['p999']:.3f} max {rq['all']['es']['max']:.3f}; "
          f"ceiling {ceil[0]:.3f}", flush=True)
    # ---- screens inputs
    tk = e61.token_strings()
    Dset = dict(seq=D["seq"], tinv=D["tinv"], tok=D["tok"], row=D["row"])
    bat = e61.feature_battery(Dset, tk, full, None)
    pos_feats = dict(line_pos=bat["line_pos"], sent_pos=bat["sent_pos"], since_punct=since_punct(full, D["row"], tk))
    rk_state = {k: ranks(v) for k, v in state.items()}
    rk_pos = {k: ranks(v) for k, v in pos_feats.items()}
    ctrl = {}
    for kd, j, lb, _ in gate["control_units"]:
        ctrl.setdefault((kd, j), []).append(lb)
    units_g = {kd: {u["id"]: u for u in gate["units"][kd]} for kd in KINDS}
    # ---- match to e64's decoded-span units
    e64m = np.load(EXT / "results/e64_maps_L12.npz")
    Dd = np.vstack([e64m["dir_logcosh"], e64m["dir_skew"]])[:, KEEP]; Dd /= np.linalg.norm(Dd, axis=1, keepdims=True)
    dec_names = [f"logcosh/{j}" for j in range(N_ICA)] + [f"skew/{j}" for j in range(N_ICA)]
    zd = np.load(EXT / "results/occ_units_v2_L12.npz")
    Yd = np.hstack([zd["Y_logcosh"], zd["Y_skew"]]).astype(np.float64); Yd = (Yd - Yd.mean(0)) / Yd.std(0)
    Zc = Zh - Zh.mean(0); Wc = Wh - Wh.mean(0)
    CzW = Zc.T @ Wc / n
    Pg = np.asarray(P["U"]) @ (np.asarray(P["sd"])[:, None] * CzW) @ G
    Pg /= np.linalg.norm(Pg, axis=0, keepdims=True)
    cos_null = np.abs(Dd @ Pg).max(0); del Pg
    # ---- every unit
    table, arrays = [], dict(rand_es_all=R["all"][0], rand_escov_all=R["all"][1], rand_es_even=R["even"][0],
                             rand_es_odd=R["odd"][0], cos_null_max=cos_null, ceiling=ceil, ceiling_cov=ceil_cov)
    for kd in KINDS:
        if f"readout_{kd}" not in z.files:
            continue
        Bk = z[f"readout_{kd}"][:N_ICA]
        Ek = {h: es(C[h], Bk) for h in C}
        Y = z[f"Y_{kd}"].astype(np.float64)
        dirs = z[f"dir_{kd}"][:, KEEP]
        cosm = np.abs(dirs @ Dd.T)
        Ys = (Y - Y.mean(0)) / Y.std(0)
        rmat = np.abs(Ys.T @ Yd / n)
        arrays[f"es_{kd}"] = Ek["all"][0]; arrays[f"escov_{kd}"] = Ek["all"][1]
        for j in range(N_ICA):
            y = Y[:, j]; ug = units_g[kd][j]
            ident, _ = e64.identity_diag(y, heavy_sign(y), D, tk)
            ry = ranks(y)
            rho_state = {k: float((ry * v).mean()) for k, v in rk_state.items()}
            rho_pos = {k: float((ry * v).mean()) for k, v in rk_pos.items()}
            u = dict(kind=kd, id=j, control=ctrl.get((kd, j)), screened=ug["screened"], repro=ug["repro"],
                     eta2_token=ug["eta2_token"], eta2_token_sq=ug["eta2_token_sq"], skew=ug["skew"], ex_kurtosis=ug["ex_kurtosis"],
                     es=float(Ek["all"][0][j]), es_cov=float(Ek["all"][1][j]),
                     es_even=float(Ek["even"][0][j]), es_odd=float(Ek["odd"][0][j]),
                     es_pct=float((R["all"][0] < Ek["all"][0][j]).mean()), es_cov_pct=float((R["all"][1] < Ek["all"][1][j]).mean()),
                     identity=dict(tail_tdi=ident["tail_tdi"], tail_max_type_share=ident["tail_max_type_share"],
                                   tail_top_types=ident["tail_top_types"][:5], identity_suspect=ident["identity_suspect"]),
                     rho_state=rho_state, rho_pos=rho_pos,
                     e64_match=dict(max_cos=float(cosm[j].max()), unit=dec_names[int(cosm[j].argmax())],
                                    same_unit=bool(cosm[j].max() >= SAME_COS), max_abs_r=float(rmat[j].max()),
                                    unit_r=dec_names[int(rmat[j].argmax())]))
            u["beats"] = dict(max=bool(u["es"] > rq["all"]["es"]["max"]), p999=bool(u["es"] > rq["all"]["es"]["p999"]),
                              cov_max=bool(u["es_cov"] > rq["all"]["es_cov"]["max"]),
                              even_max=bool(u["es_even"] > rq["even"]["es"]["max"]), odd_max=bool(u["es_odd"] > rq["odd"]["es"]["max"]))
            u["screens"] = dict(identity=bool(u["screened"] and not ident["identity_suspect"]),
                                norm_L0=bool(all(abs(v) <= STATE_RHO for v in rho_state.values())),
                                position=bool(all(abs(v) < POS_RHO for v in rho_pos.values())),
                                split_half=bool(u["repro"] >= ICA_REPRO and u["beats"]["even_max"] and u["beats"]["odd_max"]))
            u["lead"] = bool(u["beats"]["max"] and all(u["screens"].values()) and u["control"] is None)
            table.append(u)
    leads = [f"{u['kind']}/{u['id']}" for u in table if u["lead"]]
    print(t(), f"units above the random max: {[(u['kind'], u['id'], round(u['es'], 3)) for u in table if u['beats']['max']]}; "
          f"LEADS {leads}", flush=True)
    # ---- naming (descriptive): leads and the top-5 ES units
    top = sorted(table, key=lambda u: -u["es"])[:N_TOP_NAME]
    named_set = {(u["kind"], u["id"]) for u in top} | {(u["kind"], u["id"]) for u in table if u["lead"]}
    feats_ext = e64.extended_battery(Dset, tk, full)
    rules_z = np.load(EXT / "results/e64c_rules_en_core_web_trf.npz")
    rules = {k: rules_z[k][D["row"]].astype(bool) for k in E64C_RULES}
    naming = {}
    for kd, j in sorted(named_set):
        y = z[f"Y_{kd}"][:, j].astype(np.float64)
        assoc, named_by = e64.associations(y, feats_ext)
        au = {}
        for k, b in rules.items():
            if b.sum() >= 50 and (~b).sum() >= 50:
                a = roc_auc_score(b, y); au[k] = float(max(a, 1 - a))
        naming[f"{kd}/{j}"] = dict(named_by=named_by, top_assoc=sorted(((k, v.get("auroc", v.get("abs_spearman")))
                                                                        for k, v in assoc.items()), key=lambda kv: -kv[1])[:5],
                                   e64c_best=sorted(au.items(), key=lambda kv: -kv[1])[:3],
                                   contexts=e64.contexts(y, D, tk, full, 7200 + 100 * KINDS.index(kd) + j))
    del feats_ext, rules
    # ---- S2 (descriptive)
    del Ze, We, Zh, Zc, Wh, Wc; gc.collect()
    s2 = s2_descriptive(cfg, z, P, M, params, table, leads)
    arrays.update(s2.pop("_arrays"))
    res = dict(note_sha256=e64.sha(EXT / NOTE), script_sha256=e64.sha(__file__), gate_sha256=e64.sha(cfg.p(GATE)),
               n_rows=int(n), checks=dict(maps_vs_fit=maps_vs_fit, linearity=lin), fvu_raw=fvu_raw,
               fvu_identity_removed=ve_ss / float(z["vh_ss"]), per_pc_error_share=pc_es,
               ceiling=dict(es_top5=ceil[:5].tolist(), es_bottom3=ceil[-3:].tolist(), es_cov_top5=ceil_cov[:5].tolist()),
               random=dict(n=N_RAND, seed=SEED_RAND, quantiles=rq, expected_false_above_max=float(len(table) / (N_RAND + 1))),
               cos_null=dict(p999=float(np.quantile(cos_null, 0.999)), max=float(cos_null.max()), p50=float(np.median(cos_null))),
               counts=dict(n_units=len(table), above_max=sum(u["beats"]["max"] for u in table),
                           above_p999=sum(u["beats"]["p999"] for u in table), below_p50=sum(u["es"] < rq["all"]["es"]["p50"] for u in table),
                           cov_above_max=sum(u["beats"]["cov_max"] for u in table),
                           same_unit_as_e64=sum(u["e64_match"]["same_unit"] for u in table),
                           screened=sum(u["screened"] for u in table)),
               controls=[u for u in table if u["control"]], leads=leads, units=table, naming=naming, s2=s2,
               decision_before_verification=("LEADS: run the independent verification (NOTE section 7)" if leads else
                                             "no lead: STOP pending the verification of the controls and the top-ES unit"),
               seconds=round(time.time() - T0))
    np.savez_compressed(cfg.p(COV_NPZ), **arrays)
    dump(res, cfg.p(COV))
    e64.append_hash(cfg.rel(COV_NPZ), "e72 coverage: random-direction ES arrays, unit ES, ceiling, cosine null")
    e64.append_hash(cfg.rel(COV), f"e72 coverage: error share vs 10,000 same-subspace random directions; leads {leads or 'none'}")
    print(t(), f"COVERAGE done; leads {leads or 'none'}", flush=True)


def s2_descriptive(cfg, z, P, M, params, table, leads):
    S, fullS = rows("occS2", N_ROWS_S2)
    HS = np.load(cfg.H_S)
    assert HS.shape == (len(S["row"]), 1152)
    VS = raw_V(HS, S["groups"])
    ZS = project(VS, P, own_mean=False); del VS; gc.collect()     # D2's PCA applied (e64 represent_other)
    WS = e64.apply_maps(M, ZS, S["tok"])
    ZeS, WeS, _, fvuS, _ = error_W(HS, S, P, M, params); del HS, ZeS, ZS; gc.collect()
    CS = covs(WS.astype(np.float64), WeS)
    G = np.random.default_rng(SEED_RAND_S2).standard_normal((N_ICA, N_RAND)); G /= np.linalg.norm(G, axis=0, keepdims=True)
    rS = es(CS, G)[0]
    mx = float(rS.max())
    Sset = dict(seq=S["seq"], tinv=S["tinv"], tok=S["tok"], row=S["row"])
    X1 = np.hstack([WS.astype(np.float64), np.ones((len(WS), 1))])
    out = dict(n_rows=int(len(WS)), fvu_raw=fvuS, random_es=qs(rS), kinds={}, d2_units={})
    mixS = {}
    for kd in KINDS:
        Y, mix, us = e63.ica_units(WS, Sset, N_ICA, kd)
        mixS[kd] = mix
        if Y.shape[1] == 0:
            out["kinds"][kd] = dict(fit_failed=True); continue
        B, *_ = np.linalg.lstsq(X1, Y.astype(np.float64), rcond=None)
        e_ = es(CS, B[:N_ICA])[0]
        scr = np.asarray([e63.ica_pass(u, ICA_REPRO, False) for u in us])
        out["kinds"][kd] = dict(n_above_max=int((e_ > mx).sum()), n_screened_above_max=int((e_[scr] > mx).sum()),
                                n_screened=int(scr.sum()), es=qs(e_))
    # D2 leads, else the top-5 D2 units by ES: presence on S2 and their filters' ES on S2
    pick = leads or [f"{u['kind']}/{u['id']}" for u in sorted(table, key=lambda u: -u["es"])[:N_TOP_NAME]]
    for name in pick:
        kd, j = name.split("/"); j = int(j)
        b = z[f"readout_{kd}"][:N_ICA, j:j + 1]
        e_ = float(es(CS, b)[0][0])
        best = {k2: e64.best_match(z[f"pattern_{kd}"][:, j], mixS[k2]) for k2 in KINDS}
        k2 = max(best, key=lambda k: best[k][1])
        out["d2_units"][name] = dict(es_on_s2=e_, es_on_s2_pct=float((rS < e_).mean()), beats_s2_max=bool(e_ > mx),
                                     s2_best_match=dict(kind=k2, unit=best[k2][0], cos=best[k2][1]),
                                     context_generic=bool(best[k2][1] >= PRESENCE_COS))
    out["_arrays"] = dict(s2_rand_es=rS)
    print(t(), f"S2: random ES max {mx:.3f}; above-max counts { {k: v.get('n_above_max') for k, v in out['kinds'].items()} }", flush=True)
    return out


def main():
    global T0
    T0 = time.time()
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True, choices=["gate", "coverage"])
    ap.add_argument("--test-dir", default=None)
    a = ap.parse_args()
    cfg = Cfg(a.test_dir)
    print("e72 sha256", e64.sha(__file__), "test" if cfg.test else "", flush=True)
    integrity(cfg)
    print(t(), f"swap {e64.swap_used_gb():.1f} GB", flush=True)
    dict(gate=phase_gate, coverage=phase_coverage)[a.phase](cfg)


if __name__ == "__main__":
    main()
