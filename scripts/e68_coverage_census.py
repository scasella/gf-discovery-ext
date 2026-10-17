"""e68 -- coverage census of every e64 candidate (NOTE_coverage_census.md; rubric hashed before any latent number).
Read-only on D2 (corpus 1000-1999). No new data, nothing steered, no earlier verdict amended.

Stages
  run --group 7|12  (MPS + CPU; one process per layer group; layers L-2..L+2 captured once as float32 kept rows)
      prep     alignment gate (frozen chain, R^2 >= 0.999), opposite-tail signatures, 40 joint-null directions,
               self-match / signature testability -> results/e68_prep_G{L}.json + _nullscores.npy (hashed first)
      decoder  subspace null; signed cosines; OMP signed / non-negative, each with its own pullback-matched null
      acts     every latent's r with every score; tested latents' 200 strongest rows -> signature match (if testable)
               or tail-overlap match, each against a 500-eligible-random-latent null
      (resumable: a re-run loads the hashed prep and skips finished dictionaries)
  verdict           (CPU) rubric, derived calibration gate, joint-null rates, families, target conditions and tiers
  extended          (MPS) provisional targets only, extended dictionaries (262k downloaded/used/params deleted in turn)
"""
import argparse, hashlib, importlib.util, json, os, subprocess, sys, time
from pathlib import Path

import numpy as np

EXT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXT / "src"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
REPO = "google/gemma-scope-2-1b-pt"
RPA, RP = "resid_post_all", "resid_post"
PRIMARY = {
    7: [(RPA, l, "16k", v) for l in (5, 6, 7, 8, 9) for v in ("big", "small")]
       + [(RP, 7, "16k", "medium"), (RP, 7, "65k", "medium"), (RP, 7, "262k", "medium")],
    12: [(RPA, l, "16k", v) for l in (10, 11, 12, 13, 14) for v in ("big", "small")]
        + [(RP, 13, "16k", "medium"), (RP, 13, "65k", "medium"), (RP, 13, "65k", "big"), (RP, 13, "262k", "medium")],
}
EXTENDED = {
    7: [(RPA, l, "262k", v) for l in (6, 7, 8) for v in ("big", "small")]
       + [(RP, 7, w, v) for w in ("16k", "65k", "262k") for v in ("big", "small")],
    12: [(RPA, l, "262k", v) for l in (11, 12, 13) for v in ("big", "small")]
        + [(RP, 13, "16k", "big"), (RP, 13, "16k", "small"), (RP, 13, "262k", "big"), (RP, 13, "262k", "small"),
           (RP, 13, "65k", "small")],
}
SOURCE = {7: (RPA, 7, "16k", "big"), 12: (RPA, 12, "16k", "big")}
ALIGN_UNITS = {7: [("logcosh", 12), ("logcosh", 37)], 12: [("logcosh", 13), ("logcosh", 33)]}
CALIB_LISTED = {7: ["logcosh/37", "skew/37", "logcosh/50", "skew/38", "logcosh/55", "skew/49"],
                12: ["logcosh/60", "skew/28", "logcosh/24", "skew/24", "logcosh/44", "skew/39", "logcosh/10", "skew/15",
                     "logcosh/9", "skew/52"]}
N_JOINT, N_SUBNULL, N_RECON_NULL, N_RECON_NULL_BIG, N_RAND_LAT = 40, 1000, 100, 30, 500
K_OMP, K_OMP_BIG = 48, 32
SEED_SUB, SEED_JOINT, SEED_RECON, SEED_RANDLAT = 6820, 6830, 6840, 6850
COS_COVER, R_COVER, TOP = 0.7, 0.7, 5
MATCH_TOP, MATCH_MIN_ROWS, MATCH_MIN_SEQ, ALPHA, TESTABLE_RATIO = 200, 50, 30, 0.5, 2.0
E64C_STRONG, PROX_COS, CALIB_MAX_FAIL = 0.65, 0.8, 0.25
MODES = (("signed", False), ("nonneg", True))


def OUT(stage, L=None, ext="json"):
    return f"results/e68_{stage}" + (f"_G{L}" if L is not None else "") + f".{ext}"


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, EXT / rel)
    mod = importlib.util.module_from_spec(spec); sys.modules[name] = mod; spec.loader.exec_module(mod)
    return mod


def mods():
    e67 = _load("e67", "scripts/e67_prior_feature_check.py")
    e66, e65, e64 = e67.mods()
    return e67, e66, e65, e64


def dname(d):
    return f"{d[0]}/L{d[1]}_{d[2]}_{d[3]}"


def is_big(d):
    return d[2] in ("262k", "1m")


def folder_gb():
    out = subprocess.run(["du", "-sk", str(EXT)], capture_output=True, text=True).stdout.split()
    return round(int(out[0]) / 1e6, 2) if out else None


def units_of(L):
    v = json.loads((EXT / "results/e64_verdict.json").read_text())["layers"][str(L)]
    lab = json.loads((EXT / "results/e64_labels.json").read_text())["layers"][str(L)]
    out = {}
    for u in v["units"]:
        out[f"{u['kind']}/{u['id']}"] = dict(role="context_named" if u["tier"] == "NAMED_BY_CONTEXT" else "candidate", tier=u["tier"])
    for kd, j, lb, _ in lab["control_units"]:
        out[f"{kd}/{j}"] = dict(role="control", tier=f"control:{lb}")
    return out


# ---------------------------------------------------------------------------- MPS helpers
def load_model_():
    import gfdisc.model as gm
    gm.MODEL_NAME = str(sorted((EXT / "hf_home/hub/models--google--gemma-3-1b-pt/snapshots").glob("*"))[0])
    from gfdisc.env import DEVICE
    model, tok = gm.load_model(token=None)
    return model, tok, DEVICE


def capture_rows(model, tok, DEVICE, layers, rows, toks):
    """float32 residual (model.layers.N.output) at D2's kept rows, for every requested layer; nothing to disk."""
    import torch
    from gfdisc.model import capture
    H = {l: np.zeros((len(rows), 1152), np.float32) for l in layers}
    T = toks.reshape(-1, 255)
    with torch.inference_mode():
        for lo in range(0, len(T), 16):
            blk = T[lo:lo + 16]
            a, b = np.searchsorted(rows, [lo * 255, (lo + len(blk)) * 255])
            local = torch.from_numpy(rows[a:b] - lo * 255).to(DEVICE)
            ids = np.concatenate([np.full((len(blk), 1), tok.bos_token_id), blk], 1)
            with capture(model, layers) as st:
                model(torch.from_numpy(ids).to(DEVICE), use_cache=False)
            for l in layers:
                H[l][a:b] = st[l][:, 1:, :].float().reshape(-1, 1152)[local].cpu().numpy()
    for l in layers:
        assert np.isfinite(H[l]).all(), f"non-finite residual at L{l}"
    return H


def encode_batches(P, H, DEVICE, cols=None, bs=4096):
    import torch
    W = torch.tensor(P["w_enc"] if cols is None else P["w_enc"][:, cols], device=DEVICE)
    b = torch.tensor(P["b_enc"] if cols is None else P["b_enc"][cols], device=DEVICE)
    th = torch.tensor(P["threshold"] if cols is None else P["threshold"][cols], device=DEVICE)
    with torch.no_grad():
        for lo in range(0, len(H), bs):
            x = torch.tensor(H[lo:lo + bs], device=DEVICE)
            pre = x @ W + b
            yield lo, pre * (pre > th)
    del W, b, th


# ---------------------------------------------------------------------------- shared per-dictionary measurements
def decoder_dict(e67, e65, dct, keep, z, L, D, Pw=None):
    """Subspace null, signed cosines, OMP per mode with a per-mode pullback-matched null."""
    Wdec = (Pw if Pw is not None else e67.sae_params(dct))["w_dec"]
    A = e67.unit_rows(Wdec, keep)
    sub = e65.rand_bank(z["P_U"], z["P_sd"], keep, N_SUBNULL, SEED_SUB + L)
    mx = np.max(np.stack([np.abs(sub.astype(np.float32) @ A[c:c + 65536].T).max(1) for c in range(0, len(A), 65536)]), 0)
    kmax = K_OMP_BIG if is_big(dct) else K_OMP
    nrec = N_RECON_NULL_BIG if is_big(dct) else N_RECON_NULL
    rnull = e65.rand_bank(z["P_U"], z["P_sd"], keep, nrec, SEED_RECON + L)
    rn = {}
    for mode, nn in MODES:
        curves = [e67.omp(A, rnull[i], k_max=kmax, nonneg=nn)[0] for i in range(nrec)]
        k08 = [e67.k_at(c, 0.8) or kmax + 1 for c in curves]; k09 = [e67.k_at(c, 0.9) or kmax + 1 for c in curves]
        rn[mode] = dict(n=nrec, k08=dict(p1=float(np.quantile(k08, .01)), p5=float(np.quantile(k08, .05)), p50=float(np.median(k08))),
                        k09=dict(p1=float(np.quantile(k09, .01)), p5=float(np.quantile(k09, .05)), p50=float(np.median(k09))))
    rec = dict(n_latents=int(len(A)), k_max=kmax,
               null_subspace=dict(p50=float(np.median(mx)), p99=float(np.quantile(mx, .99)), p999=float(np.quantile(mx, .999))),
               null_recon=rn, units={})
    for nm, dv in D.items():
        c = A @ dv.astype(np.float32)
        o = np.argsort(-np.abs(c))
        top = [int(i) for i in o[:TOP]] + [int(i) for i in o[TOP:] if abs(c[i]) >= COS_COVER]
        om = {}
        for mode, nn in MODES:
            curve, sel, _ = e67.omp(A, dv, k_max=kmax, nonneg=nn)
            k08 = e67.k_at(curve, 0.8)
            om[mode] = dict(k08=k08, k09=e67.k_at(curve, 0.9), cos_at_kmax=curve[-1] if curve else None,
                            atoms08=[int(a) for a in (sel[:k08] if k08 else sel[:10])])
        cu = lambda i: float(Wdec[i].astype(np.float64) @ dv / np.linalg.norm(Wdec[i].astype(np.float64)))
        rec["units"][nm] = dict(top=[dict(latent=i, cos=float(c[i]), cos_unmasked=cu(i)) for i in top],
                                max_abs_cos=float(np.abs(c).max()), omp=om)
    return rec


def acts_dict(Pd, Hl, DEVICE, ctx, names, drec):
    """r of every latent with every score; tested latents' signature (if testable) / tail-overlap match vs a
    500-eligible-random-latent null; within-type r of the top-|r| latents; type-conditional ratio."""
    import torch
    n = len(Hl); nl = Pd["w_enc"].shape[1]
    sc, valid, seq_of, cur_inv, prepu = ctx["sc"], ctx["valid"], ctx["seq_of"], ctx["cur_inv"], ctx["prepu"]
    m = len(names)
    Ym = np.stack([sc[nm] for nm in names], 1)
    Yz = torch.tensor((Ym - Ym.mean(0)) / Ym.std(0), dtype=torch.float32)
    valid_t = torch.tensor(valid)
    s1 = torch.zeros(nl, dtype=torch.float64); s2 = torch.zeros(nl, dtype=torch.float64)
    sy = torch.zeros((nl, m), dtype=torch.float64); nzv = torch.zeros(nl, dtype=torch.float64)
    bs = 1024 if nl > 70000 else 4096
    for lo, a in encode_batches(Pd, Hl, DEVICE, bs=bs):
        s1 += a.sum(0).cpu().double(); s2 += (a * a).sum(0).cpu().double()
        sy += (a.T @ Yz[lo:lo + bs].to(DEVICE)).cpu().double()
        nzv += ((a > 0) & valid_t[lo:lo + len(a), None].to(DEVICE)).sum(0).cpu().double()
    var = s2 / n - (s1 / n) ** 2
    r = ((sy / n) / torch.sqrt(torch.clamp(var, min=1e-12))[:, None]).numpy(); r[~np.isfinite(r)] = 0.0
    tested, rtop = {}, {}
    for ui, nm in enumerate(names):
        o = np.argsort(-np.abs(r[:, ui]))
        rtop[nm] = [int(i) for i in o[:TOP]]
        c = set(rtop[nm]) | {int(i) for i in o if abs(r[i, ui]) >= R_COVER} | {t["latent"] for t in drec["units"][nm]["top"]}
        for om in drec["units"][nm]["omp"].values():
            c |= set(om["atoms08"])
        tested[nm] = sorted(c)
    elig = np.flatnonzero(nzv.numpy() >= MATCH_MIN_ROWS)
    rng = np.random.default_rng(ctx["seed_randlat"])
    pool = [int(i) for i in rng.choice(elig, size=min(N_RAND_LAT, len(elig)), replace=False)] if len(elig) else []
    union = sorted({l for c in tested.values() for l in c} | set(pool))
    need_w = {}
    for nm in names:
        for li in rtop[nm]:
            need_w.setdefault(li, []).append(nm)
    cnt_t = np.bincount(cur_inv)
    yc = {}
    toprows, rwithin = {}, {}
    for c0 in range(0, len(union), 500):
        cols = union[c0:c0 + 500]
        A_ = np.zeros((n, len(cols)), np.float32)
        for lo, a in encode_batches(Pd, Hl, DEVICE, cols=cols, bs=16384):
            A_[lo:lo + len(a)] = a.cpu().numpy()
        for j, li in enumerate(cols):
            col = A_[:, j]; idx = np.flatnonzero(valid & (col > 0))          # the 200 strongest non-final active rows
            toprows[li] = idx[np.argsort(-col[idx], kind="stable")[:MATCH_TOP]].astype(np.int64)
            if li in need_w:
                ac = col.astype(np.float64) - (np.bincount(cur_inv, weights=col) / cnt_t)[cur_inv]
                for nm in need_w[li]:
                    if nm not in yc:
                        y = sc[nm]; yc[nm] = y - (np.bincount(cur_inv, weights=y) / cnt_t)[cur_inv]
                    den = np.sqrt((ac ** 2).sum() * (yc[nm] ** 2).sum())
                    rwithin[(nm, li)] = float((ac * yc[nm]).sum() / den) if den > 0 else 0.0
        del A_
    urec = {}
    for ui, nm in enumerate(names):
        pu = prepu[nm]
        tails = ctx["tails"][nm]
        inT = ctx["inT"][nm]
        nulls = {}
        for side in ("pos", "neg"):
            nulls[f"{side}_sig"] = [inT[side][toprows[li]].mean() for li in pool if len(toprows[li])]
            nulls[f"{side}_ovl"] = [tails[side][toprows[li]].mean() for li in pool if len(toprows[li])]
        p99 = {k: (float(np.quantile(v, 0.99)) if len(v) else 1.0) for k, v in nulls.items()}
        mt = {}
        for li in tested[nm]:
            side = "pos" if r[li, ui] >= 0 else "neg"; tr = toprows[li]
            nr, nseq = int(len(tr)), int(len(np.unique(seq_of[tr]))) if len(tr) else 0
            enough = nr >= MATCH_MIN_ROWS and nseq >= MATCH_MIN_SEQ
            s = float(inT[side][tr].mean()) if nr else 0.0
            sp = pu[side]
            thr_s = sp["base"] + ALPHA * (sp["s_self"] - sp["base"])
            sig_ok = bool(sp["testable"] and enough and s >= thr_s and s > p99[f"{side}_sig"])
            o = float(tails[side][tr].mean()) if nr else 0.0
            tb = ctx["tail_base"][nm][side]
            thr_o = tb + ALPHA * (1 - tb)
            ovl_ok = bool(enough and o >= thr_o and o > p99[f"{side}_ovl"])
            exp = ctx["pT"](nm, side)[tr].sum() if nr else 0.0
            mt[str(li)] = dict(r=float(r[li, ui]), side=side, n_rows=nr, n_seq=nseq,
                               sig_share=s, sig_threshold=float(thr_s), sig_null_p99=p99[f"{side}_sig"], sig_match=sig_ok,
                               overlap=o, overlap_threshold=float(thr_o), overlap_null_p99=p99[f"{side}_ovl"], overlap_match=ovl_ok,
                               typecond_ratio=float(inT[side][tr].sum() / exp) if exp > 0 else None,
                               match=bool(sig_ok or ovl_ok))
        urec[nm] = dict(r_top=[dict(latent=li, r=float(r[li, ui]), r_within_type=rwithin.get((nm, li))) for li in rtop[nm]],
                        max_abs_r=float(np.abs(r[:, ui]).max()), tested=mt, null_p99=p99)
    return dict(units=urec, n_random_null=len(pool), n_eligible=int(len(elig)))


def make_ctx(e64, e65, toks, rows, units, sc, prepu, L):
    valid = (rows % 255) != 254
    nxt = np.full(len(rows), -1, np.int64); nxt[valid] = toks[rows[valid] + 1]
    seq_of = rows // 255
    cur_types, cur_inv = np.unique(toks[rows], return_inverse=True)
    nxt_tab = e65.next_by_type(toks)
    k1 = int(np.ceil(e64.TAIL * len(rows)))
    inT, tails, tail_base = {}, {}, {}
    for nm in units:
        inT[nm] = {side: np.isin(nxt, prepu[nm][side]["T"]) & valid for side in ("pos", "neg")}
        order = np.argsort(sc[nm])
        tp = np.zeros(len(rows), bool); tp[order[-k1:]] = True
        tn = np.zeros(len(rows), bool); tn[order[:k1]] = True
        tails[nm] = dict(pos=tp, neg=tn)
        tail_base[nm] = dict(pos=float(tp[valid].mean()), neg=float(tn[valid].mean()))
    cache = {}

    def pT(nm, side):
        key = (nm, side)
        if key not in cache:
            sa = np.asarray(prepu[nm][side]["T"], np.int64); vals = np.zeros(len(cur_types))
            for ti, t in enumerate(cur_types):
                dd = nxt_tab.get(int(t))
                if dd is not None and len(sa):
                    vals[ti] = float(dd[1][np.isin(dd[0], sa)].sum())
            cache[key] = vals[cur_inv]
        return cache[key]
    return dict(sc=sc, valid=valid, nxt=nxt, seq_of=seq_of, cur_inv=cur_inv, prepu=prepu, inT=inT, tails=tails,
                tail_base=tail_base, pT=pT, seed_randlat=SEED_RANDLAT + L)


# ---------------------------------------------------------------------------- stage: run (one layer group)
def build_prep(e67, e65, e64, L, H, rows, toks, meta, fnpz, z, keep, units, DEVICE):
    import torch
    e63, e61 = e64.e63, e64.e61
    src = e67.sae_params(SOURCE[L])
    dec = np.zeros((len(rows), 1152), np.float16)
    Wd = torch.tensor(src["w_dec"], device=DEVICE)
    for lo, a in encode_batches(src, H[L], DEVICE):
        dec[lo:lo + len(a)] = (a.half().float() @ Wd).cpu().numpy().astype(np.float16)
    del Wd
    tag = f"occD2c{L}"
    full = np.lib.format.open_memmap(EXT / f"cache/occ/{tag}_L{L}.npy", mode="w+", dtype=np.float16, shape=(len(toks), 1152))
    full[rows] = dec; full.flush(); del full, dec
    np.savez(EXT / f"cache/occ/{tag}_meta.npz", tokens=meta["tokens"], seq=meta["seq"], pos=meta["pos"])
    try:
        Dset = e63.represent_base(tag, L, e64.N_ITER)
    finally:
        for f in (f"{tag}_L{L}.npy", f"{tag}_meta.npz"):
            (EXT / "cache/occ" / f).unlink(missing_ok=True)
    assert np.array_equal(Dset["row"], rows), "rebuilt rows differ from the frozen rows"
    P = dict(mu=z["P_mu"], U=z["P_U"], sd=z["P_sd"]); M = {k: z[k] for k in ("Sp_is", "type_ids", "type_isqrt", "Uo", "sw")}
    Z = e61.pca_apply(P, Dset["V"]); del Dset["V"]
    W = e64.apply_maps(M, Z, Dset["tok"])
    X = np.hstack([W.astype(np.float64), np.ones((len(W), 1))])
    align = {}
    for kd, j in ALIGN_UNITS[L]:
        y = fnpz[f"Y_{kd}"][:, j].astype(np.float64); yh = X @ z[f"readout_{kd}"][:, j]
        align[f"{kd}/{j}"] = float(1 - ((y - yh) ** 2).sum() / ((y - y.mean()) ** 2).sum())
    print(f"G{L}: alignment R^2 {align}", flush=True)
    assert all(v >= 0.999 for v in align.values()), f"alignment gate failed: {align} -- STOP"
    sig65 = json.loads((EXT / "results/e65_frozen.json").read_text())["layers"][str(L)]["signatures"]
    nxt_tab = e65.next_by_type(toks)
    n = len(rows); k1 = int(np.ceil(e64.TAIL * n))
    g = np.random.default_rng(SEED_JOINT + L).standard_normal((N_JOINT, W.shape[1]))
    g /= np.linalg.norm(g, axis=1, keepdims=True)
    Zc = Z.astype(np.float64) - Z.mean(0)
    null_scores, null_dirs = np.zeros((n, N_JOINT), np.float32), {}
    for i in range(N_JOINT):
        y = W.astype(np.float64) @ g[i]
        s = e64.heavy_sign(y); ys = (y - y.mean()) / y.std()
        raw = P["U"] @ (P["sd"] * (Zc.T @ ys / n))
        dv = np.zeros(1152); dv[keep] = raw * s; dv /= np.linalg.norm(dv)
        null_scores[:, i] = (s * y).astype(np.float32); null_dirs[f"null/{i}"] = dv.tolist()
    sc = {}
    for nm in units:
        kd, j = nm.split("/")
        sc[nm] = float(z[f"sign_{kd}"][int(j)]) * fnpz[f"Y_{kd}"][:, int(j)].astype(np.float64)
    for i in range(N_JOINT):
        sc[f"null/{i}"] = null_scores[:, i].astype(np.float64)
    valid = (rows % 255) != 254
    nxt = np.full(n, -1, np.int64); nxt[valid] = toks[rows[valid] + 1]
    prep = dict(group=L, alignment_r2=align, units={}, joint_null_dirs=null_dirs)
    allnames = list(units) + [f"null/{i}" for i in range(N_JOINT)]
    for nm in allnames:
        role = units[nm] if nm in units else dict(role="joint_null", tier="joint_null")
        Tpos = list(sig65[nm]["T"]) if nm in sig65 else e64.signature(rows, np.argsort(-sc[nm])[:k1], toks, nxt_tab)
        Tneg = e64.signature(rows, np.argsort(sc[nm])[:k1], toks, nxt_tab)
        rec = dict(**role)
        for side, sig, order in (("pos", Tpos, np.argsort(-sc[nm])), ("neg", Tneg, np.argsort(sc[nm]))):
            inT = np.isin(nxt, sig) & valid
            top = order[valid[order]][:MATCH_TOP]
            rec[side] = dict(T=[int(t) for t in sig], s_self=float(inT[top].mean()), base=float(inT[valid].mean()))
            rec[side]["testable"] = bool(len(sig) > 0 and rec[side]["s_self"] >= TESTABLE_RATIO * rec[side]["base"])
        rec["testable_sig"] = rec["pos"]["testable"]
        prep["units"][nm] = rec
    return prep, null_scores


def stage_run(L):
    import torch
    e67, e66, e65, e64 = mods()
    t0 = time.time()
    units = units_of(L)
    nbc = [nm for nm, u in units.items() if u["role"] == "context_named"]       # calibration set, checked before any latent work
    e64c = e64c_max(L, nbc)
    derived = sorted(nm for nm in nbc if e64c[nm]["nameable"])
    assert derived == sorted(CALIB_LISTED[L]), f"derived calibration set {derived} != listed {CALIB_LISTED[L]} -- STOP"
    fnpz = np.load(EXT / f"results/occ_units_v2_L{L}.npz")
    rows = fnpz["row"].astype(np.int64)
    z = np.load(EXT / f"results/e64_maps_L{L}.npz")
    keep = np.ones(1152, bool); keep[e64.e61.drop_for(L)] = False
    meta = np.load(EXT / "cache/occ/occD2_meta.npz")
    toks = meta["tokens"].astype(np.int64)
    layers = sorted({d[1] for d in PRIMARY[L]})
    e64.check_swap()
    model, tok, DEVICE = load_model_()
    H = capture_rows(model, tok, DEVICE, layers, rows, toks)
    del model
    if DEVICE == "mps":
        torch.mps.empty_cache()
    print(f"[{time.time() - t0:.0f}s] G{L}: kept-row residuals (float32) at {layers}; model freed", flush=True)
    prep_p, ns_p = EXT / OUT("prep", L), EXT / f"results/e68_prep_G{L}_nullscores.npy"
    if prep_p.exists():
        e64.require_hashed(OUT("prep", L), f"results/e68_prep_G{L}_nullscores.npy")
        prep = json.loads(prep_p.read_text()); null_scores = np.load(ns_p)
        print(f"G{L}: resuming from the hashed prep", flush=True)
    else:
        prep, null_scores = build_prep(e67, e65, e64, L, H, rows, toks, meta, fnpz, z, keep, units, DEVICE)
        np.save(ns_p, null_scores)
        e64.dump(prep, OUT("prep", L))
        e64.append_hash(f"results/e68_prep_G{L}_nullscores.npy", f"e68 census prep G{L}: joint-null scores")
        e64.append_hash(OUT("prep", L), f"e68 census prep G{L}: alignment gate, signatures, joint-null directions, testability; before any latent number")
    names = list(prep["units"])
    sc = {}
    for nm in names:
        if nm.startswith("null/"):
            sc[nm] = null_scores[:, int(nm.split("/")[1])].astype(np.float64)
        else:
            kd, j = nm.split("/"); sc[nm] = float(z[f"sign_{kd}"][int(j)]) * fnpz[f"Y_{kd}"][:, int(j)].astype(np.float64)
    D = {nm: (np.asarray(prep["joint_null_dirs"][nm]) if nm.startswith("null/") else
              z[f"dir_{nm.split('/')[0]}"][int(nm.split('/')[1])].astype(np.float64)) for nm in names}
    print(f"[{time.time() - t0:.0f}s] G{L}: prep ready; candidates with a testable signature "
          f"{sum(prep['units'][nm]['testable_sig'] for nm in names if prep['units'][nm]['role'] == 'candidate')}"
          f"/{sum(prep['units'][nm]['role'] == 'candidate' for nm in names)}", flush=True)
    # decoder (resumable)
    dec_p = EXT / OUT("decoder", L)
    decoder = json.loads(dec_p.read_text()) if dec_p.exists() else dict(group=L, dicts={})
    for dct in PRIMARY[L]:
        if dname(dct) in decoder["dicts"]:
            continue
        e64.check_swap(); t1 = time.time()
        rec = decoder_dict(e67, e65, dct, keep, z, L, D)
        rec["seconds"] = round(time.time() - t1)
        decoder["dicts"][dname(dct)] = rec
        e64.dump(decoder, OUT("decoder", L))
        print(f"[{time.time() - t0:.0f}s] G{L} decoder {dname(dct)}: null p99.9 {rec['null_subspace']['p999']:.3f} "
              f"recon-null k08 p1 signed {rec['null_recon']['signed']['k08']['p1']:.0f} nonneg {rec['null_recon']['nonneg']['k08']['p1']:.0f} "
              f"({rec['seconds']}s)", flush=True)
    if len(decoder["dicts"]) == len(PRIMARY[L]) and "hashed" not in decoder:
        decoder["hashed"] = True; e64.dump(decoder, OUT("decoder", L))
        e64.append_hash(OUT("decoder", L), f"e68 census decoder G{L}")
    # acts (resumable)
    ctx = make_ctx(e64, e65, toks, rows, names, sc, prep["units"], L)
    act_p = EXT / OUT("acts", L)
    acts = json.loads(act_p.read_text()) if act_p.exists() else dict(group=L, dicts={}, folder_gb=[])
    for dct in PRIMARY[L]:
        if dname(dct) in acts["dicts"]:
            continue
        e64.check_swap(); t1 = time.time()
        rec = acts_dict(e67.sae_params(dct), H[dct[1]], DEVICE, ctx, names, decoder["dicts"][dname(dct)])
        rec["seconds"] = round(time.time() - t1)
        acts["dicts"][dname(dct)] = rec
        acts["folder_gb"].append(folder_gb())
        if DEVICE == "mps":
            torch.mps.empty_cache()
        e64.dump(acts, OUT("acts", L))
        print(f"[{time.time() - t0:.0f}s] G{L} acts {dname(dct)} ({rec['seconds']}s; null latents {rec['n_random_null']}/"
              f"{rec['n_eligible']} eligible; swap {e64.swap_used_gb():.1f} GB)", flush=True)
    if len(acts["dicts"]) == len(PRIMARY[L]) and "hashed" not in acts:
        acts["hashed"] = True; e64.dump(acts, OUT("acts", L))
        e64.append_hash(OUT("acts", L), f"e68 census acts G{L}")


# ---------------------------------------------------------------------------- stage: verdict (CPU)
def e64c_max(L, names):
    from scipy.stats import rankdata
    F = np.load(EXT / "results/e64c_rules_en_core_web_trf.npz")
    fnpz = np.load(EXT / f"results/occ_units_v2_L{L}.npz"); rows = fnpz["row"].astype(np.int64)
    out = {}
    for nmx in names:
        kd, j = nmx.split("/")
        y = fnpz[f"Y_{kd}"][:, int(j)].astype(np.float64)
        R = rankdata(y); best = (None, 0.5)
        for rn in F.files:
            f = F[rn][rows].astype(bool); n1 = int(f.sum())
            if n1 < 50 or len(f) - n1 < 50:
                continue
            a = (R[f].sum() - n1 * (n1 + 1) / 2) / (n1 * (len(f) - n1)); a = max(a, 1 - a)
            if a > best[1]:
                best = (rn, float(a))
        out[nmx] = dict(rule=best[0], abs_auroc=best[1], nameable=bool(best[1] >= E64C_STRONG))
    return out


def unit_verdict(nm, group, dec, act, dicts, source=True):
    hits, partial = [], []
    for dn in dicts:
        if not source and dn == dname(SOURCE[group]):
            continue
        du, au = dec["dicts"][dn]["units"][nm], act["dicts"][dn]["units"][nm]
        mt = au["tested"]
        for t in du["top"]:
            if abs(t["cos"]) >= COS_COVER and mt.get(str(t["latent"]), {}).get("match"):
                hits.append(dict(dict=dn, latent=t["latent"], via="decoder_cos", value=t["cos"],
                                 by=("signature" if mt[str(t["latent"])]["sig_match"] else "overlap")))
        for li, rec in mt.items():
            if abs(rec["r"]) >= R_COVER and rec["match"]:
                hits.append(dict(dict=dn, latent=int(li), via="activation_r", value=rec["r"],
                                 by=("signature" if rec["sig_match"] else "overlap")))
        for mode, om in du["omp"].items():
            p1 = dec["dicts"][dn]["null_recon"][mode]["k08"]["p1"]
            if om["k08"] is not None and om["k08"] <= p1:
                ok = [bool(mt.get(str(a), {}).get("match")) for a in om["atoms08"]]
                if sum(ok) > len(ok) / 2:
                    partial.append(dict(dict=dn, mode=mode, k08=om["k08"], null_p1=p1, atoms=om["atoms08"]))
    return ("COVERED" if hits else ("PARTIALLY COVERED" if partial else "NOT COVERED")), hits, partial


def stage_verdict():
    e67, e66, e65, e64 = mods()
    e64v = json.loads((EXT / "results/e64_verdict.json").read_text())
    e65v = json.loads((EXT / "results/e65_verdict.json").read_text())
    lab = json.loads((EXT / "results/e64_labels.json").read_text())
    out = dict(groups={}, provisional_targets=[])
    for L in PRIMARY:
        for f in (OUT("prep", L), f"results/e68_prep_G{L}_nullscores.npy", OUT("decoder", L), OUT("acts", L)):
            e64.require_hashed(f)
        prep = json.loads((EXT / OUT("prep", L)).read_text()); dec = json.loads((EXT / OUT("decoder", L)).read_text())
        act = json.loads((EXT / OUT("acts", L)).read_text())
        dicts = list(dec["dicts"]); names = list(prep["units"])
        real = [nm for nm in names if not nm.startswith("null/")]
        res = {}
        for nm in names:
            v, hits, partial = unit_verdict(nm, L, dec, act, dicts, True)
            vns, _, _ = unit_verdict(nm, L, dec, act, dicts, False)
            aligned = any(dec["dicts"][dn]["units"][nm]["max_abs_cos"] > dec["dicts"][dn]["null_subspace"]["p999"] for dn in dicts)
            aligned_ns = any(dec["dicts"][dn]["units"][nm]["max_abs_cos"] > dec["dicts"][dn]["null_subspace"]["p999"]
                             for dn in dicts if dn != dname(SOURCE[L]))
            best = max(((dec["dicts"][dn]["units"][nm]["max_abs_cos"], dn) for dn in dicts), key=lambda x: x[0])
            res[nm] = dict(role=prep["units"][nm]["role"], verdict=v, verdict_without_source=vns,
                           level_changes_without_source=(v != vns), hits=hits, partial=partial,
                           testable_sig=prep["units"][nm]["testable_sig"],
                           s_self=prep["units"][nm]["pos"]["s_self"], base=prep["units"][nm]["pos"]["base"],
                           best_cos=dict(value=best[0], dict=best[1]), aligned_beyond_chance=aligned,
                           aligned_beyond_chance_without_source=aligned_ns,
                           max_abs_r=max(act["dicts"][dn]["units"][nm]["max_abs_r"] for dn in dicts))
        e64c = e64c_max(L, real)
        for nm in real:
            res[nm]["e64c"] = e64c[nm]
        vunits = {f"{u['kind']}/{u['id']}": u for u in e64v["layers"][str(L)]["units"]}
        e65u = {f"{u['kind']}/{u['id']}": u for u in e65v["layers"][str(L)]["units"]}
        labu = {f"{u['kind']}/{u['id']}": u for u in lab["layers"][str(L)]["units"]}
        ctrl = sorted(f"{kd}/{j}" for kd, j, _, _ in lab["layers"][str(L)]["control_units"])
        twin_all = {}
        for a, b, _ in lab["layers"][str(L)]["twins"]:
            twin_all[f"logcosh/{a}"] = f"skew/{b}"; twin_all[f"skew/{b}"] = f"logcosh/{a}"
        role_of = lambda nm: ("battery_named" if labu.get(nm, {}).get("named") and nm not in ctrl else
                              res[nm]["role"] if nm in res else "screened")
        # calibration gate (set derived by rule; asserted equal to the listed set)
        derived = sorted(nm for nm in real if res[nm]["role"] == "context_named" and e64c[nm]["nameable"])
        assert derived == sorted(CALIB_LISTED[L]), f"derived calibration set {derived} != listed {CALIB_LISTED[L]} -- STOP"
        calib = derived + ctrl
        fails = [nm for nm in calib if res[nm]["verdict"] == "NOT COVERED"]
        fails_ns = [nm for nm in calib if res[nm]["verdict_without_source"] == "NOT COVERED"]
        gate_ok = len(fails) <= CALIB_MAX_FAIL * len(calib)
        nulls = [nm for nm in names if nm.startswith("null/")]
        jn = {k: dict(n=len(nulls), covered=sum(res[nm][k] == "COVERED" for nm in nulls),
                      covered_or_partial=sum(res[nm][k] in ("COVERED", "PARTIALLY COVERED") for nm in nulls))
              for k in ("verdict", "verdict_without_source")}
        z = np.load(EXT / f"results/e64_maps_L{L}.npz")
        dirv = lambda nm: z[f"dir_{nm.split('/')[0]}"][int(nm.split('/')[1])]
        covered_units = [nm for nm in real if res[nm]["verdict"] == "COVERED"]
        named_like = [nm for nm in labu if role_of(nm) in ("battery_named", "context_named", "control")]
        fams = []
        for f in e64v["layers"][str(L)]["families"]:
            mem = [f"{a}/{b}" for a, b in f["members"] if res.get(f"{a}/{b}", {}).get("role") == "candidate"]
            if not mem:
                continue
            twins = [dict(member=m, twin=twin_all[m], role=role_of(twin_all[m])) for m in mem if m in twin_all]
            twin_named = [t for t in twins if t["role"] in ("battery_named", "context_named", "control")]
            prox = [dict(member=m, other=o, cos=float(abs(dirv(m) @ dirv(o)))) for m in mem
                    for o in sorted(set(named_like + covered_units) - set(mem)) if abs(dirv(m) @ dirv(o)) >= PROX_COS]
            e65r = [dict(unit=m, b3=e65u.get(m, {}).get("b3_pass"), b4=e65u.get(m, {}).get("b4_pass")) for m in mem]
            fam = dict(layer=L, members=mem, verdicts={m: res[m]["verdict"] for m in mem},
                       not_covered=all(res[m]["verdict"] == "NOT COVERED" for m in mem),
                       not_covered_without_source=all(res[m]["verdict_without_source"] == "NOT COVERED" for m in mem),
                       r2_replicated=any(vunits[m]["tier"] == "replicated geometric" for m in mem),
                       e64c_nameable=any(res[m]["e64c"]["nameable"] for m in mem), e64c={m: res[m]["e64c"] for m in mem},
                       twins=twins, twin_named=twin_named, proximity=prox,
                       identity_or_generic=any(vunits[m]["identity"]["identity_suspect"] or vunits[m].get("context_generic") for m in mem),
                       e65=e65r)
            fam["provisional_target"] = bool(gate_ok and fam["not_covered"] and fam["r2_replicated"] and not fam["e64c_nameable"]
                                             and not fam["twin_named"] and not fam["identity_or_generic"] and not fam["proximity"])
            if L == 12:
                fam["tier"] = "A" if any(e["b3"] and e["b4"] is not False for e in e65r) else "B"
            else:
                fam["tier"] = "L7 (no validated behavioural test)"
            fams.append(fam)
            if fam["provisional_target"]:
                out["provisional_targets"].append(fam)
        cnt = lambda role, key: {k: sum(1 for nm in names if res[nm]["role"] == role and res[nm][key] == k)
                                 for k in ("COVERED", "PARTIALLY COVERED", "NOT COVERED")}
        out["groups"][str(L)] = dict(
            units=res, families=fams,
            calibration=dict(set=calib, derived_equals_listed=True, fails=fails, gate_ok=gate_ok, fails_without_source=fails_ns,
                             gate_ok_without_source=len(fails_ns) <= CALIB_MAX_FAIL * len(calib)),
            joint_null=jn,
            counts={r: cnt(r, "verdict") for r in ("candidate", "context_named", "control", "joint_null")},
            counts_without_source={r: cnt(r, "verdict_without_source") for r in ("candidate", "context_named", "control", "joint_null")},
            family13=[res[n]["verdict"] for n in ("logcosh/13", "skew/16")] if L == 12 else None)
        print(f"G{L}: counts {out['groups'][str(L)]['counts']} | gate {gate_ok} fails {fails} | joint null {jn['verdict']} | "
              f"families {len(fams)} provisional targets {sum(f['provisional_target'] for f in fams)} | family13 "
              f"{out['groups'][str(L)]['family13']}", flush=True)
    e64.dump(out, OUT("census"))
    e64.append_hash(OUT("census"), "e68 coverage census verdict (primary set; before the extended pass)")
    print("PROVISIONAL TARGETS", json.dumps([dict(layer=f["layer"], members=f["members"], tier=f["tier"]) for f in out["provisional_targets"]]), flush=True)


# ---------------------------------------------------------------------------- stage: extended (MPS)
def snap_dir():
    return next((EXT / "hf_home/hub/models--google--gemma-scope-2-1b-pt/snapshots").iterdir())


def download_if_missing(dct):
    sub = f"{dct[0]}/layer_{dct[1]}_width_{dct[2]}_l0_{dct[3]}"
    if (snap_dir() / sub / "params.safetensors").exists():
        return False
    env = {k: v for k, v in os.environ.items() if k != "HF_HUB_OFFLINE"}
    code = ("from huggingface_hub import hf_hub_download as h\n"
            f"[h('{REPO}', '{sub}/' + f) for f in ('config.json', 'params.safetensors')]")
    subprocess.run([sys.executable, "-c", code], env=env, check=True)
    return True


def stage_extended():
    import torch
    e67, e66, e65, e64 = mods()
    e64.require_hashed(OUT("census"))
    census = json.loads((EXT / OUT("census")).read_text())
    targets = census["provisional_targets"]
    out = dict(targets=[])
    if not targets:
        out["note"] = "no provisional target; extended pass not needed"
        e64.dump(out, OUT("extended")); e64.append_hash(OUT("extended"), "e68 census extended pass (no targets)")
        print("no provisional targets", flush=True); return
    toks = np.load(EXT / "cache/occ/occD2_meta.npz")["tokens"].astype(np.int64)
    model, tok, DEVICE = load_model_()
    Hs = {}
    for L in sorted({t["layer"] for t in targets}):
        fnpz = np.load(EXT / f"results/occ_units_v2_L{L}.npz"); rows = fnpz["row"].astype(np.int64)
        Hs[L] = (capture_rows(model, tok, DEVICE, sorted({d[1] for d in EXTENDED[L]}), rows, toks), rows)
    del model
    if DEVICE == "mps":
        torch.mps.empty_cache()
    for L in sorted(Hs):
        H, rows = Hs[L]
        prep = json.loads((EXT / OUT("prep", L)).read_text())
        fnpz = np.load(EXT / f"results/occ_units_v2_L{L}.npz")
        z = np.load(EXT / f"results/e64_maps_L{L}.npz")
        keep = np.ones(1152, bool); keep[e64.e61.drop_for(L)] = False
        mem = [m for t in targets if t["layer"] == L for m in t["members"]]
        sc = {nm: float(z[f"sign_{nm.split('/')[0]}"][int(nm.split('/')[1])]) * fnpz[f"Y_{nm.split('/')[0]}"][:, int(nm.split('/')[1])].astype(np.float64)
              for nm in mem}
        D = {nm: z[f"dir_{nm.split('/')[0]}"][int(nm.split('/')[1])].astype(np.float64) for nm in mem}
        ctx = make_ctx(e64, e65, toks, rows, mem, sc, {nm: prep["units"][nm] for nm in mem}, L)
        dec_rec, act_rec = dict(group=L, dicts={}), dict(group=L, dicts={})
        for dct in EXTENDED[L]:
            e64.check_swap()
            need = dict(zip(("16k", "65k", "262k"), (0.16, 0.61, 2.43)))[dct[2]]
            assert (folder_gb() or 0) + need <= 20.0, f"extended {dname(dct)} would take the folder over 20 GB -- STOP"
            downloaded = download_if_missing(dct)
            Pd = e67.sae_params(dct)
            dec_rec["dicts"][dname(dct)] = decoder_dict(e67, e65, dct, keep, z, L, D, Pw=Pd)
            act_rec["dicts"][dname(dct)] = acts_dict(Pd, H[dct[1]], DEVICE, ctx, mem, dec_rec["dicts"][dname(dct)])
            del Pd
            if DEVICE == "mps":
                torch.mps.empty_cache()
            if downloaded:                                             # every extended download is deleted after use
                link = snap_dir() / f"{dct[0]}/layer_{dct[1]}_width_{dct[2]}_l0_{dct[3]}/params.safetensors"
                blob = link.resolve(); link.unlink(missing_ok=True)
                if blob.exists():
                    blob.unlink()
            print(f"G{L} extended {dname(dct)} done; folder {folder_gb()} GB", flush=True)
        for t in targets:
            if t["layer"] != L:
                continue
            vs = {m: unit_verdict(m, L, dec_rec, act_rec, list(dec_rec["dicts"]), True)[0] for m in t["members"]}
            out["targets"].append(dict(layer=L, members=t["members"], tier=t["tier"], extended_verdicts=vs,
                                       survives=all(v == "NOT COVERED" for v in vs.values())))
        out[f"decoder_G{L}"] = dec_rec; out[f"acts_G{L}"] = act_rec
    e64.dump(out, OUT("extended"))
    e64.append_hash(OUT("extended"), "e68 census extended pass on provisional targets")
    print("TARGETS AFTER EXTENDED PASS", json.dumps([dict(layer=t["layer"], members=t["members"], tier=t["tier"], survives=t["survives"])
                                                     for t in out["targets"]]), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["run", "verdict", "extended"])
    ap.add_argument("--group", type=int, choices=[7, 12])
    a = ap.parse_args()
    print("e68 sha256", hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), flush=True)
    e67, e66, e65, e64 = mods()
    e64.require_hashed("NOTE_coverage_census.md", "scripts/e68_coverage_census.py")
    e64.check_swap()
    if a.stage == "run":
        assert a.group, "--group 7 or 12"
        stage_run(a.group)
    elif a.stage == "verdict":
        stage_verdict()
    else:
        stage_extended()


if __name__ == "__main__":
    main()
