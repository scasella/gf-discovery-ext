"""e69 -- SAE-gap pilot (NOTE_sae_gap_pilot.md, hashed before any pilot number). Burned data only: D2 (corpus
1000-1999) and S2 (D2's within-sequence shuffle). Decides whether a full prereg on fresh text is written.

Phases
  p0  (CPU)  reproduce the design's instrument on D2: L12 16k-big SAE error, demean3, PCA-256, multi-lag AMUSE m=32,
             split-half, screens -> results/e69_p0_units.npz + results/e69_p0.json (hashed before p1)
  p1  (MPS)  S2 raw L12 kept rows (fp16, as D2's e67 rows); D2 re-capture identity check; D2 next-token entropy and
             log p -> results/e69_p1_S2_raw_L12_rows.npy + results/e69_p1_d2_pred.npz (hashed before p2)
  p2  (CPU)  per screened unit: S3-generic (a) refit match, (b) transfer; predictability naming; bag naming (ridge
             and MLP); sub-threshold coverage vs RAND-E (descriptive); known labels (descriptive); P3 decision
             -> results/e69_pilot.json (hashed)
"""
import argparse, gc, hashlib, importlib.util, json, os, sys, time
from pathlib import Path

import numpy as np

EXT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXT / "src"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_HOME", str(EXT / "hf_home"))
NOTE = "NOTE_sae_gap_pilot.md"
LAYER, DROP = 12, 1038
SAE_BIG = ("resid_post_all", 12, "16k", "big")
SAE_SMALL = ("resid_post_all", 12, "16k", "small")
LAGS, K_PCA, M_UNITS, TOPMATCH = (1, 2, 4, 8, 16), 256, 32, 64
REP_MIN, ETA_MAX, RHO_MAX, ACF_MIN = 0.8, 0.3, 0.3, 0.2
MIN_TYPE_OCC, MIN_POS = 8, 4
GEN_COS, GEN_ACF = 0.8, 0.5
NAME_RHO, NAME_R2 = 0.3, 0.3
PRED_W, BAG_W = (1, 4, 16), (8, 16, 32, 64)
N_RANDE, SEED_RANDE, N_SUBLAT = 32, 6900, 128
STOP_MIN_SCREENED = 2
OUT = dict(p0_npz="results/e69_p0_units.npz", p0="results/e69_p0.json", s2="results/e69_p1_S2_raw_L12_rows.npy",
           pred="results/e69_p1_d2_pred.npz", pilot="results/e69_pilot.json")
KEEP = np.ones(1152, bool); KEEP[DROP] = False


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, EXT / rel)
    mod = importlib.util.module_from_spec(spec); sys.modules[name] = mod; spec.loader.exec_module(mod)
    return mod


def mods():
    e67 = _load("e67", "scripts/e67_prior_feature_check.py")
    e66, e65, e64 = e67.mods()
    return e67, e64


def t():
    return f"[{time.time() - T0:.0f}s]"


# ---------------------------------------------------------------------------- the instrument (from the design)
class RowSet:
    """Kept rows of one 255-token-per-sequence set, with demean3 operators and within-sequence lag pairs."""

    def __init__(self, meta, rows=None):
        import scipy.sparse as sp
        tok_all = meta["tokens"].astype(np.int64); seq_all = meta["seq"].astype(np.int64); pos_all = meta["pos"].astype(np.int64)
        if rows is None:                       # e63.represent_base's kept-row rule, on the set's own meta
            _, inv, cnt = np.unique(tok_all, return_inverse=True, return_counts=True)
            rows = np.flatnonzero((cnt[inv] >= MIN_TYPE_OCC) & (pos_all >= MIN_POS))
        self.rows = np.asarray(rows, np.int64); self.tok_all = tok_all
        self.tok, self.seq, self.pos = tok_all[self.rows], seq_all[self.rows], pos_all[self.rows]
        groups = [np.unique(a, return_inverse=True)[1] for a in (self.tok, self.seq, self.pos)]
        self.Ms = []
        for g in groups:
            G = int(g.max()) + 1
            M = sp.csr_matrix((np.ones(len(g)), (g, np.arange(len(g)))), shape=(G, len(g)))
            self.Ms.append((M, np.maximum(np.asarray(M.sum(1)).ravel(), 1), g))
        self.tinv = groups[0]; self.cnt = np.bincount(self.tinv)
        self.even = self.seq % 2 == 0
        n = len(self.rows); self.pairs = {}
        for L in LAGS:
            tgt = self.rows - L
            j = np.clip(np.searchsorted(self.rows, tgt), 0, n - 1)
            ok = (self.rows[j] == tgt) & ((self.rows % 255) >= L)
            self.pairs[L] = (np.flatnonzero(ok), j[ok])

    def demean3(self, V):
        one = V.ndim == 1
        V = (V[:, None] if one else V).astype(np.float64)
        for _ in range(3):
            for M, c, g in self.Ms:
                V -= (M @ V / c[:, None])[g]
        V = V.astype(np.float32)
        return V[:, 0] if one else V

    def eta2(self, v):
        v = v.astype(np.float64); mt = np.bincount(self.tinv, weights=v) / self.cnt
        return float(((mt[self.tinv] - v.mean()) ** 2).sum() / max(((v - v.mean()) ** 2).sum(), 1e-12))

    def acf(self, y, L):
        c, p = self.pairs[L]
        return float(np.corrcoef(y[c], y[p])[0, 1])

    def acov(self, y, L):
        c, p = self.pairs[L]
        a, b = y[c].astype(np.float64), y[p].astype(np.float64)
        return float(((a - a.mean()) * (b - b.mean())).mean())


def sae_params(dct):
    from huggingface_hub import hf_hub_download
    from safetensors.numpy import load_file
    sub = f"{dct[0]}/layer_{dct[1]}_width_{dct[2]}_l0_{dct[3]}"
    return load_file(hf_hub_download("google/gemma-scope-2-1b-pt", f"{sub}/params.safetensors"))


def error_rows(H16, P):
    """e = h - (JumpReLU_fp32(h) @ W_dec + b_dec) on fp16-stored rows (CPU), coordinate 1038 dropped afterwards;
    also log||h||, log||e|| (kept coordinates) and L0."""
    import torch
    We, be, th, Wd, bd = [torch.from_numpy(P[k].astype(np.float32)) for k in ("w_enc", "b_enc", "threshold", "w_dec", "b_dec")]
    n = len(H16); E = np.empty((n, 1151), np.float32)
    lh, le, l0 = np.empty(n), np.empty(n), np.empty(n)
    kp = torch.from_numpy(KEEP)
    with torch.no_grad():
        for a in range(0, n, 8192):
            x = torch.from_numpy(H16[a:a + 8192].astype(np.float32))
            pre = x @ We + be; z = pre * (pre > th)
            e = (x - z @ Wd - bd)[:, kp]
            E[a:a + 8192] = e.numpy()
            lh[a:a + 8192] = torch.log(torch.linalg.norm(x[:, kp], dim=1)).numpy()
            le[a:a + 8192] = torch.log(torch.linalg.norm(e, dim=1)).numpy()
            l0[a:a + 8192] = (z > 0).sum(1).numpy()
    assert np.isfinite(E).all()
    return E, dict(log_h=lh, log_e=le, L0=l0)


def amuse(S, Z, sel=None):
    Mx = np.zeros((Z.shape[1], Z.shape[1]))
    for L in LAGS:
        c, p = S.pairs[L]
        if sel is not None:
            k = sel[c]; c, p = c[k], p[k]
        A, B = Z[c].astype(np.float64), Z[p].astype(np.float64)
        Cl = (A.T @ B) / len(A); Mx += (Cl + Cl.T) / 2
    w, U = np.linalg.eigh(Mx / len(LAGS)); o = np.argsort(w)[::-1]
    return w[o], U[:, o]


def fit_instrument(S, E, m):
    V = S.demean3(E)
    mu = V.mean(0)
    w, U = np.linalg.eigh(np.cov((V - mu)[::3].T)); o = np.argsort(w)[::-1][:K_PCA]
    U, sd = U[:, o], np.sqrt(w[o])
    Z = (((V - mu) @ U) / sd).astype(np.float32)
    eig, W = amuse(S, Z); _, Wa = amuse(S, Z, S.even); _, Wb = amuse(S, Z, ~S.even)
    Wm = W[:, :m]
    rep = np.minimum(np.abs(Wm.T @ Wa[:, :TOPMATCH]).max(1), np.abs(Wm.T @ Wb[:, :TOPMATCH]).max(1))
    filt = U @ (Wm / sd[:, None]); pat = U @ (Wm * sd[:, None])
    pat /= np.linalg.norm(pat, axis=0, keepdims=True)
    return dict(V=V, mu=mu, U=U, sd=sd, Z=Z, eig=eig, W=Wm, rep=rep, Y=(Z @ Wm).astype(np.float32), filt=filt, pat=pat)


def screens(S, F, state, m=M_UNITS):
    from scipy.stats import spearmanr
    st = {k: S.demean3(v) for k, v in state.items()}
    units = []
    for j in range(m):
        y = F["Y"][:, j]
        rho = {k: float(spearmanr(y, v)[0]) for k, v in st.items()}
        rec = dict(id=j, eig=float(F["eig"][j]), rep=float(F["rep"][j]), eta2_y=S.eta2(y), eta2_y2=S.eta2(y.astype(np.float64) ** 2),
                   rho_state=rho, acf1=S.acf(y, 1), acf16=S.acf(y, 16))
        rec["screened"] = bool(rec["rep"] >= REP_MIN and rec["eta2_y"] <= ETA_MAX and rec["eta2_y2"] <= ETA_MAX
                               and all(abs(v) <= RHO_MAX for v in rho.values()) and rec["acf1"] >= ACF_MIN)
        units.append(rec)
    return units


# ---------------------------------------------------------------------------- p0 (CPU)
def phase_p0(e64):
    assert not (EXT / OUT["p0"]).exists(), "p0 runs once"
    fz = np.load(EXT / "results/occ_units_v2_L12.npz"); rows = fz["row"].astype(np.int64)
    meta = np.load(EXT / "cache/occ/occD2_meta.npz")
    S = RowSet(meta)
    assert np.array_equal(S.rows, rows), "kept-row rule does not reproduce D2's frozen rows"
    e64.require_hashed("results/e67_raw_L12_rows.npy")
    H16 = np.load(EXT / "results/e67_raw_L12_rows.npy")
    E, state = error_rows(H16, sae_params(SAE_BIG)); del H16; gc.collect()
    print(t(), "D2 error rows", E.shape, flush=True)
    F = fit_instrument(S, E, M_UNITS)
    units = screens(S, F, state)
    scr = [u["id"] for u in units if u["screened"]]
    out = dict(set="D2", n_rows=int(len(rows)), eig_top8=[float(x) for x in F["eig"][:8]], n_screened=len(scr), screened_ids=scr,
               design_reading=dict(n_screened=3, rep=[0.94, 0.84, 0.83]), units=units,
               stop_p0=bool(len(scr) < STOP_MIN_SCREENED))
    np.savez(EXT / OUT["p0_npz"], mu=F["mu"], U=F["U"], sd=F["sd"], W=F["W"], eig=F["eig"], filt=F["filt"], pat=F["pat"], Y=F["Y"],
             rep=F["rep"], screened=np.asarray(scr, np.int64))
    e64.dump(out, OUT["p0"])
    e64.append_hash(OUT["p0_npz"], "e69 SAE-gap pilot P0: D2 AMUSE maps, units, filters, patterns (before P1)")
    e64.append_hash(OUT["p0"], "e69 SAE-gap pilot P0: D2 reproduction and screens (before P1)")
    print(t(), f"P0: screened {scr} of {M_UNITS}; eig top4 {np.round(F['eig'][:4], 3).tolist()}; "
          f"rep {[round(units[j]['rep'], 2) for j in scr]}; acf1 {[round(units[j]['acf1'], 2) for j in scr]}"
          + ("  -> STOP-P0" if out["stop_p0"] else ""), flush=True)


# ---------------------------------------------------------------------------- p1 (MPS)
def phase_p1(e64):
    import torch
    e64.require_hashed(OUT["p0"], OUT["p0_npz"])
    assert not json.loads((EXT / OUT["p0"]).read_text())["stop_p0"], "STOP-P0 was reached; no P1"
    assert not (EXT / OUT["pred"]).exists(), "p1 runs once"
    e64.check_swap()
    import gfdisc.model as gm
    gm.MODEL_NAME = str(sorted((EXT / "hf_home/hub/models--google--gemma-3-1b-pt/snapshots").glob("*"))[0])
    from gfdisc.env import DEVICE
    from gfdisc.model import capture
    mD = np.load(EXT / "cache/occ/occD2_meta.npz"); mS = np.load(EXT / "cache/occ/occS2_meta.npz")
    tD, tS = mD["tokens"].astype(np.int64), mS["tokens"].astype(np.int64)
    assert bool(mS["shuffled"]) and str(mS["source"]) == "corpus" and not bool(mD["shuffled"])
    assert all(np.array_equal(np.sort(tD[s:s + 255]), np.sort(tS[s:s + 255])) for s in range(0, len(tD), 255)), \
        "S2 is not a per-sequence permutation of D2"
    assert not np.array_equal(tD, tS)
    S2 = RowSet(mS); D2rows = np.load(EXT / "results/occ_units_v2_L12.npz")["row"].astype(np.int64)
    model, tok = gm.load_model(token=None)

    def residual(toks, rows):
        H = np.zeros((len(rows), 1152), np.float16); T = toks.reshape(-1, 255)
        with torch.inference_mode():
            for lo in range(0, len(T), 16):
                blk = T[lo:lo + 16]
                a, b = np.searchsorted(rows, [lo * 255, (lo + len(blk)) * 255])
                ids = np.concatenate([np.full((len(blk), 1), tok.bos_token_id), blk], 1)
                with capture(model, [LAYER]) as st:
                    model(torch.from_numpy(ids).to(DEVICE), use_cache=False)
                x = st[LAYER][:, 1:, :].float().reshape(-1, 1152).cpu().numpy().astype(np.float16)
                H[a:b] = x[rows[a:b] - lo * 255]
        assert np.isfinite(H.astype(np.float32)).all()
        return H

    HD = residual(tD, D2rows)
    ref = np.load(EXT / "results/e67_raw_L12_rows.npy")
    same = float((HD == ref).mean())
    dif = np.abs(HD.astype(np.float32) - ref.astype(np.float32))
    a32, b32 = HD.astype(np.float32), ref.astype(np.float32)
    ac, bc = a32 - a32.mean(0), b32 - b32.mean(0)
    cc = (ac * bc).sum(0) / np.sqrt((ac * ac).sum(0) * (bc * bc).sum(0) + 1e-30)
    ident = dict(share_identical=same, max_abs_diff=float(dif.max()), min_coord_corr=float(np.nanmin(cc)))
    del HD, ref, dif, a32, b32, ac, bc
    print(t(), "D2 re-capture identity", ident, flush=True)
    assert same >= 0.99 and ident["min_coord_corr"] >= 0.999, f"D2 re-capture differs from e67's rows: {ident}"
    if (EXT / OUT["s2"]).exists():                       # resume after a crash in the logits pass
        e64.require_hashed(OUT["s2"])
        print(t(), "S2 rows already saved and hashed; skipping", flush=True)
    else:
        HS = residual(tS, S2.rows)
        np.save(EXT / OUT["s2"], HS); del HS
        e64.append_hash(OUT["s2"], "e69 SAE-gap pilot P1: S2 (D2 shuffled) raw L12 kept rows, fp16 (before P2)")
        print(t(), f"S2 raw L12 kept rows {len(S2.rows)} saved", flush=True)
    # D2 next-token entropy and log p at every position (the final position of a sequence has no next token)
    T = tD.reshape(-1, 255); ent = np.zeros(len(tD), np.float32); lp = np.full(len(tD), np.nan, np.float32)
    with torch.inference_mode():
        for lo in range(0, len(T), 4):
            blk = T[lo:lo + 4]
            ids = torch.from_numpy(np.concatenate([np.full((len(blk), 1), tok.bos_token_id), blk], 1)).to(DEVICE)
            lg = model(ids, use_cache=False).logits[:, 1:, :].float()
            ls = torch.log_softmax(lg, -1)
            e_ = -(ls.exp() * ls).sum(-1)
            nxt = torch.from_numpy(blk[:, 1:]).to(DEVICE)
            l_ = ls[:, :-1, :].gather(-1, nxt[..., None])[..., 0]
            ent[lo * 255:(lo + len(blk)) * 255] = e_.reshape(-1).cpu().numpy()
            lpb = np.full((len(blk), 255), np.nan, np.float32); lpb[:, :-1] = l_.cpu().numpy()
            lp[lo * 255:(lo + len(blk)) * 255] = lpb.reshape(-1)
            del lg, ls, e_, l_
    assert np.isfinite(ent).all()
    np.savez(EXT / OUT["pred"], entropy=ent, logp=lp, identity=json.dumps(ident))
    del model
    if DEVICE == "mps":
        torch.mps.empty_cache()
    e64.append_hash(OUT["pred"], "e69 SAE-gap pilot P1: D2 next-token entropy and log p; D2 re-capture identity (before P2)")
    print(t(), f"P1 done; mean entropy {float(ent.mean()):.2f}; swap {e64.swap_used_gb():.1f} GB", flush=True)


# ---------------------------------------------------------------------------- p2 (CPU)
def causal_mean(x_full, rows, W, include_current):
    """Mean of x over the W positions before row r in its own sequence (current included if asked)."""
    xf = np.nan_to_num(x_full.astype(np.float64)) if x_full.ndim == 1 else x_full.astype(np.float64)
    ok = np.isfinite(x_full).astype(np.float64) if x_full.ndim == 1 else None
    cs = np.concatenate([np.zeros((1,) + xf.shape[1:]), np.cumsum(xf, 0)], 0)
    s0 = (rows // 255) * 255
    end = rows + 1 if include_current else rows
    beg = np.maximum(s0, end - W)
    if ok is not None:
        cn = np.concatenate([[0.0], np.cumsum(ok)])
        n = np.maximum(cn[end] - cn[beg], 1)
        return (cs[end] - cs[beg]) / n
    n = np.maximum(end - beg, 1)
    return (cs[end] - cs[beg]) / n[:, None]


def heldout_r2(X, y, even, model):
    from sklearn.linear_model import RidgeCV
    from sklearn.neural_network import MLPRegressor
    yz = (y - y[even].mean()) / y[even].std()
    if model == "ridge":
        mdl = RidgeCV(alphas=np.logspace(-2, 4, 7)).fit(X[even], yz[even])
    else:
        mdl = MLPRegressor(hidden_layer_sizes=(256,), early_stopping=True, random_state=0, max_iter=200).fit(X[even], yz[even])
    p = mdl.predict(X[~even]); yt = yz[~even]
    return float(1 - ((yt - p) ** 2).sum() / ((yt - yt.mean()) ** 2).sum())


def battery_labels(toks_full, rows):
    """err_sobi5.py's 7 battery-style labels + 10 e64c rules, unchanged."""
    from transformers import AutoTokenizer
    tk = AutoTokenizer.from_pretrained(str(sorted((EXT / "hf_home/hub/models--google--gemma-3-1b-pt/snapshots").glob("*"))[0]))
    S, n = {}, len(toks_full)
    F = {k: np.zeros(n, bool) for k in ("paren", "quote", "after_newline", "next_newline", "prev_punct", "after_colon")}
    for s0 in range(0, n, 255):
        d = q = 0; prev = ""
        for j in range(255):
            r = s0 + j; tt = int(toks_full[r]); st = S.setdefault(tt, tk.decode([tt]))
            nx = S.setdefault(int(toks_full[r + 1]), tk.decode([int(toks_full[r + 1])])) if j < 254 else ""
            F["paren"][r] = d > 0; F["quote"][r] = q % 2 == 1; F["after_newline"][r] = "\n" in prev; F["next_newline"][r] = "\n" in nx
            ps = prev.strip(); F["prev_punct"][r] = bool(ps) and not any(c.isalnum() for c in ps); F["after_colon"][r] = prev.rstrip().endswith(":")
            d = max(0, d + st.count("(") - st.count(")")); q += st.count('"') + st.count("“") + st.count("”"); prev = st
    R = np.load(EXT / "results/e64c_rules_en_core_web_trf.npz")
    lab = {k: v[rows] for k, v in F.items()}
    lab.update({nm: R[c][rows].astype(bool) for c, nm in (("A", "after_cc"), ("B", "temporal_cl"), ("F", "subj_np"), ("G", "fronted"),
                                                         ("I", "line_init"), ("J", "negation"), ("K", "copular"), ("N", "after_the"),
                                                         ("M", "temporal_expr"), ("L", "enumeration"))})
    return lab


def subthreshold(H16, targets, even, dct):
    """max |r| of each target with every latent's pre-activation; held-out R^2 from {a_j, clip(pre_j,0,th_j)} of the
    N_SUBLAT latents with the largest even-half |r(pre)|."""
    import torch
    P = sae_params(dct)
    We, be, th = [torch.from_numpy(P[k].astype(np.float32)) for k in ("w_enc", "b_enc", "threshold")]
    n, nl = len(H16), We.shape[1]
    Yt = np.stack(list(targets.values()), 1).astype(np.float64)
    out = {}
    stats = {}
    for part, sel in (("all", np.ones(n, bool)), ("even", even)):
        Y = Yt[sel]; Yz = torch.from_numpy(((Y - Y.mean(0)) / Y.std(0)).astype(np.float32))
        idx = np.flatnonzero(sel)
        s1 = torch.zeros(nl, dtype=torch.float64); s2 = torch.zeros(nl, dtype=torch.float64); sy = torch.zeros((nl, Y.shape[1]), dtype=torch.float64)
        with torch.no_grad():
            for a in range(0, len(idx), 8192):
                x = torch.from_numpy(H16[idx[a:a + 8192]].astype(np.float32))
                pre = x @ We + be
                s1 += pre.sum(0).double(); s2 += (pre * pre).sum(0).double(); sy += (pre.T @ Yz[a:a + 8192]).double()
        m_ = len(idx); var = s2 / m_ - (s1 / m_) ** 2
        r = ((sy / m_) / torch.sqrt(torch.clamp(var, min=1e-12))[:, None]).numpy(); r[~np.isfinite(r)] = 0
        stats[part] = r
    Hf = torch.from_numpy(H16.astype(np.float32))
    for ti, nm in enumerate(targets):
        top = torch.from_numpy(np.argsort(-np.abs(stats["even"][:, ti]))[:N_SUBLAT].copy())
        with torch.no_grad():
            pre = Hf @ We[:, top] + be[top]
            a = pre * (pre > th[top])                                  # JumpReLU code
            s = torch.minimum(torch.clamp(pre, min=0), th[top])        # clip(pre, 0, theta): the sub-threshold drive
            X = torch.cat([a, s], 1).numpy().astype(np.float64)
        X = (X - X[even].mean(0)) / np.maximum(X[even].std(0), 1e-9)
        out[nm] = dict(max_abs_r_pre=float(np.abs(stats["all"][:, ti]).max()), argmax_latent=int(np.abs(stats["all"][:, ti]).argmax()),
                       r2_sub=heldout_r2(X, targets[nm], even, "ridge"))
        del pre, a, s, X
    del Hf
    return out


def phase_p2(e64):
    from scipy.stats import spearmanr
    from sklearn.metrics import roc_auc_score
    e64.require_hashed(OUT["p0"], OUT["p0_npz"], OUT["s2"], OUT["pred"])
    assert not (EXT / OUT["pilot"]).exists(), "p2 runs once"
    p0 = json.loads((EXT / OUT["p0"]).read_text()); z = np.load(EXT / OUT["p0_npz"])
    scr = [int(j) for j in z["screened"]]
    mD, mS = np.load(EXT / "cache/occ/occD2_meta.npz"), np.load(EXT / "cache/occ/occS2_meta.npz")
    D = RowSet(mD, np.load(EXT / "results/occ_units_v2_L12.npz")["row"]); S2 = RowSet(mS)
    out = dict(screened=scr, units={}, s2={})
    H16 = np.load(EXT / "results/e67_raw_L12_rows.npy")
    E, stD = error_rows(H16, sae_params(SAE_BIG))
    V = D.demean3(E); del E
    Zd = (((V - z["mu"]) @ z["U"]) / z["sd"]).astype(np.float32); del V; gc.collect()
    # ---- S3 (a) refit on S2, (b) transfer of D2 filters
    HS = np.load(EXT / OUT["s2"])
    ES, stS = error_rows(HS, sae_params(SAE_BIG)); del HS; gc.collect()
    FS = fit_instrument(S2, ES, TOPMATCH); del ES; gc.collect()
    uS = screens(S2, dict(Y=FS["Y"][:, :M_UNITS], eig=FS["eig"], rep=FS["rep"][:M_UNITS]), stS)
    out["s2"] = dict(n_rows=int(len(S2.rows)), eig_top8=[float(x) for x in FS["eig"][:8]], d2_eig_top8=p0["eig_top8"],
                     error_inflation=dict(mean_log_e_s2=float(stS["log_e"].mean()), mean_log_e_d2=float(stD["log_e"].mean()),
                                          mean_log_h_s2=float(stS["log_h"].mean()), mean_log_h_d2=float(stD["log_h"].mean()),
                                          mean_L0_s2=float(stS["L0"].mean()), mean_L0_d2=float(stD["L0"].mean())),
                     n_screened=sum(u["screened"] for u in uS), screened_ids=[u["id"] for u in uS if u["screened"]],
                     units=uS)
    print(t(), f"S2: screened {out['s2']['screened_ids']}; eig top4 {np.round(FS['eig'][:4], 3).tolist()} vs D2 {np.round(p0['eig_top8'][:4], 3).tolist()}", flush=True)
    for j in scr:
        cos = np.abs(FS["pat"].T @ z["pat"][:, j])
        yS = FS["V"] @ z["filt"][:, j]; yD = z["Y"][:, j]
        acfS, acfD = S2.acf(yS, 1), p0["units"][j]["acf1"]
        acvS, acvD = S2.acov(yS, 1), D.acov(yD, 1)
        out["units"][str(j)] = dict(
            generic_a=dict(max_cos=float(cos.max()), s2_unit=int(cos.argmax()), flag=bool(cos.max() >= GEN_COS)),
            generic_b=dict(acf1_s2=acfS, acf1_d2=acfD, ratio=float(acfS / acfD), acov1_s2=acvS, acov1_d2=acvD,
                           acov_ratio=float(acvS / acvD), var_ratio=float(np.var(yS) / np.var(yD)), acf16_s2=S2.acf(yS, 16),
                           flag=bool(acfS >= GEN_ACF * acfD or acvS >= GEN_ACF * acvD)))
    del FS; gc.collect()
    # ---- naming: predictability
    pred = np.load(EXT / OUT["pred"])
    nonfinal = (D.rows % 255) != 254
    feats = {}
    for W in PRED_W:
        sfx = "" if W == 1 else f"_mean{W}"
        feats["entropy" + sfx] = causal_mean(pred["entropy"], D.rows, W, True)
        feats["neglogp" + sfx] = causal_mean(-pred["logp"], D.rows, W, True)
    Y = z["Y"].astype(np.float64)
    Dn = RowSet(mD, D.rows[nonfinal])                        # identity removal on the non-final kept rows
    feats_dm = {k: Dn.demean3(v[nonfinal].astype(np.float32)) for k, v in feats.items()}
    for j in scr:
        rho = {k: float(spearmanr(Y[nonfinal, j], v[nonfinal])[0]) for k, v in feats.items()}
        rho.update({k + "_idrem": float(spearmanr(Y[nonfinal, j], v)[0]) for k, v in feats_dm.items()})
        out["units"][str(j)]["named_p"] = dict(rho=rho, flag=bool(max(abs(v) for v in rho.values()) >= NAME_RHO))
    del Dn, feats_dm
    print(t(), "predictability", {j: round(max(abs(v) for v in out["units"][str(j)]["named_p"]["rho"].values()), 3) for j in scr}, flush=True)
    # ---- naming: bag of previous tokens (type embedding = PCA-128 of D2 per-type mean raw L12 residual), ridge and MLP
    import scipy.sparse as sp
    Hk = H16[:, KEEP].astype(np.float64)
    ut, inv = np.unique(D.tok, return_inverse=True)
    Mt = sp.csr_matrix((np.ones(len(inv)), (inv, np.arange(len(inv)))), shape=(len(ut), len(inv)))
    TM = (Mt @ Hk) / np.asarray(Mt.sum(1)); del Hk
    TMc = TM - TM.mean(0); _, s_, vt = np.linalg.svd(TMc, full_matrices=False)
    emb = TMc @ vt[:128].T / s_[:128] * np.sqrt(len(ut))
    lut = np.full(int(D.tok_all.max()) + 1, -1, np.int64); lut[ut] = np.arange(len(ut))
    li = lut[D.tok_all]; Efull = np.zeros((len(D.tok_all), 128)); Efull[li >= 0] = emb[li[li >= 0]]
    for j in scr:
        out["units"][str(j)]["named_b"] = dict(r2={}, flag=False)
    for W in BAG_W:
        X = D.demean3(causal_mean(Efull, D.rows, W, False).astype(np.float32)).astype(np.float64)
        X = (X - X[D.even].mean(0)) / np.maximum(X[D.even].std(0), 1e-9)
        for j in scr:
            r2 = {mdl: heldout_r2(X, Y[:, j], D.even, mdl) for mdl in ("ridge", "mlp")}
            out["units"][str(j)]["named_b"]["r2"][str(W)] = r2
            if max(r2.values()) >= NAME_R2:
                out["units"][str(j)]["named_b"]["flag"] = True
        print(t(), f"bag W={W}", {j: out["units"][str(j)]["named_b"]["r2"][str(W)] for j in scr}, flush=True)
    del Efull, X; gc.collect()
    # ---- sub-threshold coverage (descriptive), with RAND-E
    g = np.random.default_rng(SEED_RANDE).standard_normal((K_PCA, N_RANDE)); g /= np.linalg.norm(g, axis=0, keepdims=True)
    targets = {f"unit/{j}": Y[:, j] for j in scr}
    targets.update({f"rande/{i}": (Zd @ g[:, i]).astype(np.float64) for i in range(N_RANDE)})
    del Zd
    sub = {}
    for dct in (SAE_BIG, SAE_SMALL):
        sub["/".join(map(str, dct))] = subthreshold(H16, targets, D.even, dct)
        print(t(), "sub-threshold", dct[3], {k: (round(v["max_abs_r_pre"], 3), round(v["r2_sub"], 3)) for k, v in sub["/".join(map(str, dct))].items() if k.startswith("unit")}, flush=True)
    for j in scr:
        rec = {}
        for dn, d in sub.items():
            rr = [d[f"rande/{i}"]["max_abs_r_pre"] for i in range(N_RANDE)]; r2 = [d[f"rande/{i}"]["r2_sub"] for i in range(N_RANDE)]
            u = d[f"unit/{j}"]
            rec[dn] = dict(u, pct_max_abs_r_pre=float((np.asarray(rr) < u["max_abs_r_pre"]).mean()),
                           pct_r2_sub=float((np.asarray(r2) < u["r2_sub"]).mean()),
                           rande_p50_r=float(np.median(rr)), rande_max_r=float(np.max(rr)),
                           rande_p50_r2=float(np.median(r2)), rande_max_r2=float(np.max(r2)))
        out["units"][str(j)]["subthreshold"] = rec
    out["rande"] = {dn: {k: v for k, v in d.items() if k.startswith("rande")} for dn, d in sub.items()}
    del H16; gc.collect()
    # ---- known labels (descriptive)
    lab = battery_labels(D.tok_all, D.rows)
    for j in scr:
        best = {}
        for lb, yb in lab.items():
            if yb.sum() < 50 or (~yb).sum() < 50:
                continue
            a = roc_auc_score(yb, Y[:, j]); best[lb] = round(float(max(a, 1 - a)), 3)
        out["units"][str(j)]["labels_top5"] = dict(sorted(best.items(), key=lambda kv: -kv[1])[:5])
    # ---- P3 decision
    surv = []
    for j in scr:
        u = out["units"][str(j)]
        u["survivor"] = not (u["generic_a"]["flag"] or u["generic_b"]["flag"] or u["named_p"]["flag"] or u["named_b"]["flag"])
        u["removed_by"] = [k for k in ("generic_a", "generic_b", "named_p", "named_b") if u[k]["flag"]]
        if u["survivor"]:
            surv.append(j)
    out["survivors"] = surv
    out["decision"] = "GO" if surv else "STOP"
    e64.dump(out, OUT["pilot"])
    e64.append_hash(OUT["pilot"], f"e69 SAE-gap pilot P2/P3: decision {out['decision']}")
    for j in scr:
        u = out["units"][str(j)]
        print(f"unit {j}: generic_a cos {u['generic_a']['max_cos']:.3f} | generic_b ratio {u['generic_b']['ratio']:.2f} | "
              f"predictability max|rho| {max(abs(v) for v in u['named_p']['rho'].values()):.3f} | bag max R2 "
              f"{max(max(v.values()) for v in u['named_b']['r2'].values()):.3f} | removed by {u['removed_by']}", flush=True)
    print(t(), f"DECISION {out['decision']}; survivors {surv}", flush=True)


def main():
    global T0
    T0 = time.time()
    ap = argparse.ArgumentParser(); ap.add_argument("--phase", required=True, choices=["p0", "p1", "p2"])
    a = ap.parse_args()
    print("e69 sha256", hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), flush=True)
    e67, e64 = mods()
    e64.require_hashed(NOTE, "scripts/e69_sae_gap_pilot.py")
    e64.check_swap()
    dict(p0=phase_p0, p1=phase_p1, p2=phase_p2)[a.phase](e64)


T0 = time.time()
if __name__ == "__main__":
    main()
