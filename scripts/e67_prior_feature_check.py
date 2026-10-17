"""e67 -- read-only prior-feature check of the e66-confirmed L12 direction (NOTE_prior_feature_check.md,
rubric hashed before any comparison). Changes no e64/e65/e66 verdict.

Stages
  decoder  (CPU)  |cos| of d (logcosh #13, skew #16) with every decoder row of every listed dictionary;
                  isotropic and pullback-matched nulls; sparse reconstruction (signed and non-negative OMP)
  acts     (MPS)  one model pass over D2 (corpus 1000-1999): every dictionary's latent activations at
                  D2's kept rows, Pearson r with the unit's frozen D2 score and with h.d; the candidate
                  latents' strongest D2 rows: next-token category share (the behavioural match) and contexts
  probe    (CPU)  the next-token preposition/subordinator probe in the unit's own representation W (from an
                  e60 re-extraction tagged occD2x) and on the raw L12 residual (saved by acts); cosines with d
  verdict  (CPU)  the rubric, applied mechanically; Neuronpedia text is attached by the analyst separately
"""
import argparse, hashlib, importlib.util, json, os, sys, time
from pathlib import Path

import numpy as np

EXT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXT / "src"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
REPO = "google/gemma-scope-2-1b-pt"
L = 12
DICTS = [("resid_post_all", 12, "16k", "big"), ("resid_post_all", 12, "16k", "small"),
         ("resid_post_all", 12, "262k", "big"), ("resid_post_all", 12, "262k", "small"),
         ("resid_post_all", 11, "16k", "big"), ("resid_post_all", 11, "16k", "small"),
         ("resid_post_all", 13, "16k", "big"), ("resid_post_all", 13, "16k", "small"),
         ("resid_post", 13, "65k", "big"),
         ("resid_post", 13, "16k", "medium"), ("resid_post", 13, "65k", "medium"), ("resid_post", 13, "262k", "medium"),
         ("resid_post_all", 10, "16k", "big"), ("resid_post_all", 10, "16k", "small"),
         ("resid_post_all", 14, "16k", "big"), ("resid_post_all", 14, "16k", "small")]
UNITS = (("logcosh", 13), ("skew", 16))
N_NULL, N_NULL_RECON, K_MAX, TOP = 1000, 10, 256, 10
MATCH_CATS, MATCH_SHARE, MATCH_TOP = ("preposition", "subordinator", "temporal"), 0.30, 200
COS_COVER, R_COVER, RECON_K, RECON_COS = 0.7, 0.7, 5, 0.9
OUT = dict(decoder="results/e67_decoder.json", acts="results/e67_acts.json", probe="results/e67_probe.json",
           verdict="results/e67_verdict.json", raw="results/e67_raw_L12_rows.npy")


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, EXT / rel)
    mod = importlib.util.module_from_spec(spec); sys.modules[name] = mod; spec.loader.exec_module(mod)
    return mod


def mods():
    e66 = _load("e66", "scripts/e66_confirm_l12_family.py")
    e65 = e66.load_e65(); e64 = e65.load_e64()
    return e66, e65, e64


def dname(d):
    return f"{d[0]}/L{d[1]}_{d[2]}_{d[3]}"


def sae_params(d):
    from huggingface_hub import hf_hub_download
    from safetensors.numpy import load_file
    sub = f"{d[0]}/layer_{d[1]}_width_{d[2]}_l0_{d[3]}"
    return load_file(hf_hub_download(REPO, f"{sub}/params.safetensors"))


def directions(e64):
    _, _, z = e64.load_maps(L)
    keep = np.ones(1152, bool); keep[e64.e61.drop_for(L)] = False
    return {f"{k}/{j}": z[f"dir_{k}"][j].astype(np.float64) for k, j in UNITS}, keep, z


def unit_rows(A, keep):
    A = A.astype(np.float32).copy(); A[:, ~keep] = 0
    return A / np.maximum(np.linalg.norm(A, axis=1, keepdims=True), 1e-12)


# ---------------------------------------------------------------------------- reconstruction
def omp(A, d, k_max=K_MAX, nonneg=False):
    """Greedy OMP over unit rows A (n_atoms x dim); returns cos(d, d_hat) after each step and the atoms."""
    from scipy.optimize import nnls
    d = d / np.linalg.norm(d); r = d.copy(); sel, curve, coef = [], [], None
    for _ in range(k_max):
        c = A @ r.astype(np.float32)
        if nonneg:
            c[sel] = -np.inf; j = int(np.argmax(c))
            if c[j] <= 0:
                break
        else:
            c = np.abs(c); c[sel] = -1; j = int(np.argmax(c))
        sel.append(j)
        B = A[sel].astype(np.float64).T
        coef = nnls(B, d)[0] if nonneg else np.linalg.lstsq(B, d, rcond=None)[0]
        dh = B @ coef; r = d - dh
        curve.append(float(dh @ d / max(np.linalg.norm(dh), 1e-12)))
    return curve, sel, (None if coef is None else coef.tolist())


def k_at(curve, t):
    return next((i + 1 for i, c in enumerate(curve) if c >= t), None)


# ---------------------------------------------------------------------------- stage: decoder (CPU)
def stage_decoder():
    e66, e65, e64 = mods()
    dirs, keep, z = directions(e64)
    rng = np.random.default_rng(6720)
    iso = rng.standard_normal((N_NULL, 1152)); iso[:, ~keep] = 0; iso /= np.linalg.norm(iso, axis=1, keepdims=True)
    sub = e65.rand_bank(z["P_U"], z["P_sd"], keep, N_NULL, 6720)
    sub_recon = e65.rand_bank(z["P_U"], z["P_sd"], keep, N_NULL_RECON, 6721)
    out = dict(dicts={})
    for d in DICTS:
        t0 = time.time()
        P = sae_params(d)
        A = unit_rows(P["w_dec"], keep)
        A_raw = P["w_dec"].astype(np.float32); A_raw /= np.linalg.norm(A_raw, axis=1, keepdims=True)
        rec = dict(n_latents=int(len(A)))
        for name, v in dirs.items():
            c = A @ v.astype(np.float32)
            o = np.argsort(-np.abs(c))[:TOP]
            rec[name] = dict(top=[dict(latent=int(i), cos=float(c[i]), cos_unmasked=float(A_raw[i] @ v)) for i in o],
                             max_abs_cos=float(np.abs(c).max()))
        mx = lambda V: np.max(np.stack([np.abs(V.astype(np.float32) @ A[c:c + 65536].T).max(1)
                                         for c in range(0, len(A), 65536)]), 0)
        mx_iso, mx_sub = mx(iso), mx(sub)
        rec["null_isotropic"] = dict(p50=float(np.median(mx_iso)), p99=float(np.quantile(mx_iso, .99)), p999=float(np.quantile(mx_iso, .999)))
        rec["null_subspace"] = dict(p50=float(np.median(mx_sub)), p99=float(np.quantile(mx_sub, .99)), p999=float(np.quantile(mx_sub, .999)))
        # sparse reconstruction
        rec["recon"] = {}
        for name, v in dirs.items():
            cs, ss, _ = omp(A, v)
            cp, sp, cfp = omp(A, v, nonneg=True)
            cm, sm, cfm = omp(A, -v, nonneg=True)
            rec["recon"][name] = dict(
                signed=dict(k08=k_at(cs, .8), k09=k_at(cs, .9), cos_at={k: cs[k - 1] for k in (1, 2, 3, 5, 10, 20, 50, 100, 256) if k <= len(cs)},
                            first5=ss[:5]),
                nonneg_pos=dict(k08=k_at(cp, .8), k09=k_at(cp, .9), cos_at={k: cp[k - 1] for k in (1, 2, 3, 5, 10, 20, 50) if k <= len(cp)},
                                first5=sp[:5], coef5=(cfp or [])[:5]),
                nonneg_neg=dict(k08=k_at(cm, .8), k09=k_at(cm, .9), cos_at={k: cm[k - 1] for k in (1, 2, 3, 5, 10, 20, 50) if k <= len(cm)},
                                first5=sm[:5]))
        n_null = N_NULL_RECON if len(A) <= 70000 else 4
        ks = [omp(A, sub_recon[i])[0] for i in range(n_null)]
        rec["recon_null"] = dict(n=n_null, signed_k08=[k_at(c, .8) for c in ks], signed_k09=[k_at(c, .9) for c in ks],
                                 signed_cos5=[c[4] for c in ks])
        rec["seconds"] = round(time.time() - t0)
        out["dicts"][dname(d)] = rec
        print(f"{dname(d)}: max|cos| {[round(rec[n]['max_abs_cos'], 3) for n in dirs]} null iso p99.9 {rec['null_isotropic']['p999']:.3f} "
              f"sub p99.9 {rec['null_subspace']['p999']:.3f} | recon k0.9 signed {[rec['recon'][n]['signed']['k09'] for n in dirs]} "
              f"null {rec['recon_null']['signed_k09']} ({rec['seconds']}s)", flush=True)
        e64.dump(out, OUT["decoder"])
    e64.append_hash(OUT["decoder"], "e67 decoder cosines, nulls and sparse reconstruction")


# ---------------------------------------------------------------------------- stage: acts (MPS)
def stage_acts():
    import torch
    e66, e65, e64 = mods()
    dirs, keep, _ = directions(e64)
    fnpz = np.load(EXT / "results/occ_units_v2_L12.npz")
    rows = fnpz["row"].astype(np.int64)
    Y = {f"{k}/{j}": fnpz[f"Y_{k}"][:, j].astype(np.float64) for k, j in UNITS}
    toks = np.load(EXT / "cache/occ/occD2_meta.npz")["tokens"].astype(np.int64)
    import gfdisc.model as gm
    gm.MODEL_NAME = str(sorted((EXT / "hf_home/hub/models--google--gemma-3-1b-pt/snapshots").glob("*"))[0])
    from gfdisc.env import DEVICE
    from gfdisc.model import capture
    model, tok = gm.load_model(token=None)
    layers = sorted({d[1] for d in DICTS})
    T = toks.reshape(-1, 255)
    H = {l: np.zeros((len(toks), 1152), np.float16) for l in layers}
    t0 = time.time()
    with torch.inference_mode():
        for lo in range(0, len(T), 16):
            blk = T[lo:lo + 16]
            ids = np.concatenate([np.full((len(blk), 1), tok.bos_token_id), blk], 1)
            with capture(model, layers) as st:
                model(torch.from_numpy(ids).to(DEVICE), use_cache=False)
            for l in layers:
                H[l][lo * 255:(lo + len(blk)) * 255] = st[l][:, 1:, :].float().reshape(-1, 1152).cpu().numpy().astype(np.float16)
    print(f"[{time.time() - t0:.0f}s] D2 residuals captured at {layers}", flush=True)
    np.save(EXT / OUT["raw"], H[L][rows])                     # kept rows, L12 raw residual (for the probe stage)
    dec = lambda ids: tok.decode([int(t) for t in ids])
    out = dict(rows=int(len(rows)), dicts={}, raw_projection={})
    proj = {}
    for name, v in dirs.items():
        p = H[L][rows].astype(np.float32) @ v.astype(np.float32)
        proj[name] = p.astype(np.float64)
        out["raw_projection"][name] = dict(r_with_Y=float(np.corrcoef(p, Y[name])[0, 1]))
    targets = {**{f"Y:{n}": Y[n] for n in Y}, **{f"hd:{n}": proj[n] for n in proj}}
    tnames = list(targets)
    Tm = np.stack([targets[n] for n in tnames], 1)
    Tz = (Tm - Tm.mean(0)) / Tm.std(0)
    valid_next = (rows % 255) != 254
    cats_next = np.full(len(rows), "", dtype=object)
    cache = {}
    for i in np.flatnonzero(valid_next):
        t = int(toks[rows[i] + 1]); cats_next[i] = cache.setdefault(t, e66.category(tok.decode([t])))
    is_match = np.isin(cats_next, MATCH_CATS)
    out["base_match_share"] = float(is_match[valid_next].mean())
    dec_rows = json.loads((EXT / OUT["decoder"]).read_text())
    for d in DICTS:
        t1 = time.time()
        P = sae_params(d)
        W = torch.tensor(P["w_enc"], device=DEVICE); b = torch.tensor(P["b_enc"], device=DEVICE)
        th = torch.tensor(P["threshold"], device=DEVICE)
        Hl = H[d[1]][rows]
        n, nl = len(rows), W.shape[1]
        s1 = torch.zeros(nl, dtype=torch.float64); s2 = torch.zeros(nl, dtype=torch.float64)
        sy = torch.zeros((nl, Tz.shape[1]), dtype=torch.float64); nz = torch.zeros(nl, dtype=torch.float64)
        Tz_t = torch.tensor(Tz, dtype=torch.float32)
        bs = 1024 if nl > 70000 else 4096
        with torch.no_grad():
            for lo in range(0, n, bs):
                x = torch.tensor(Hl[lo:lo + bs].astype(np.float32), device=DEVICE)
                pre = x @ W + b; a = pre * (pre > th)
                s1 += a.sum(0).cpu().double(); s2 += (a * a).sum(0).cpu().double(); nz += (a > 0).sum(0).cpu().double()
                sy += (a.T @ Tz_t[lo:lo + bs].to(DEVICE)).cpu().double()
        mean = s1 / n; var = s2 / n - mean ** 2
        r = (sy / n) / torch.sqrt(torch.clamp(var, min=1e-12))[:, None]
        r = r.numpy(); r[~np.isfinite(r)] = 0.0
        rec = dict(n_latents=int(nl), mean_l0=float(nz.sum() / n), corr={})
        cand = set()
        for ti, tn in enumerate(tnames):
            o = np.argsort(-np.abs(r[:, ti]))[:TOP]
            rec["corr"][tn] = [dict(latent=int(i), r=float(r[i, ti]), freq=float(nz[i] / n)) for i in o]
            if tn.startswith("Y:"):
                cand.update(int(i) for i in o[:5])
        for name in dirs:
            cand.update(int(t["latent"]) for t in dec_rows["dicts"][dname(d)][name]["top"][:5])
            for mode in ("signed", "nonneg_pos", "nonneg_neg"):
                cand.update(int(a) for a in dec_rows["dicts"][dname(d)]["recon"][name][mode]["first5"])
        # second pass: candidate latents' activations on every kept row -> behavioural match and contexts
        ci = sorted(cand); Wc, bc, thc = W[:, ci], b[ci], th[ci]
        acts = np.zeros((n, len(ci)), np.float32)
        with torch.no_grad():
            for lo in range(0, n, 8192):
                x = torch.tensor(Hl[lo:lo + 8192].astype(np.float32), device=DEVICE)
                pre = x @ Wc + bc; acts[lo:lo + 8192] = (pre * (pre > thc)).cpu().numpy()
        rec["latents"] = {}
        for j, li in enumerate(ci):
            a = acts[:, j]; order = np.argsort(-a)
            top = [i for i in order if valid_next[i] and a[i] > 0][:MATCH_TOP]
            share = float(is_match[top].mean()) if top else None
            ctx = []
            for i in order[:12]:
                if a[i] <= 0:
                    break
                rr = int(rows[i]); s0 = (rr // 255) * 255
                ctx.append(dec(toks[max(s0, rr - 15):rr]) + "[[" + dec([toks[rr]]) + "]]" + dec(toks[rr + 1:min(s0 + 255, rr + 4)]))
            rec["latents"][str(li)] = dict(n_active=int((a > 0).sum()), match_share=share, n_top=len(top),
                                           behavioural_match=bool(share is not None and share >= MATCH_SHARE and len(top) >= 50),
                                           contexts=ctx)
        del W, b, th, Wc
        if DEVICE == "mps":
            torch.mps.empty_cache()
        rec["seconds"] = round(time.time() - t1)
        out["dicts"][dname(d)] = rec
        best = {tn: (rec["corr"][tn][0]["latent"], round(rec["corr"][tn][0]["r"], 3)) for tn in tnames}
        print(f"[{time.time() - t0:.0f}s] {dname(d)} L0 {rec['mean_l0']:.0f} best r {best}", flush=True)
        e64.dump(out, OUT["acts"])
    e64.append_hash(OUT["raw"], "e67 acts: D2 kept-row raw L12 residual (fp16), for the probe")
    e64.append_hash(OUT["acts"], "e67 latent-activation correlations and top-latent behaviour on D2")


# ---------------------------------------------------------------------------- stage: probe (CPU)
def stage_probe():
    from sklearn.linear_model import LogisticRegression
    e66, e65, e64 = mods()
    e63 = e64.e63
    dirs, keep, z = directions(e64)
    fnpz = np.load(EXT / "results/occ_units_v2_L12.npz"); rows = fnpz["row"].astype(np.int64)
    Dset = e63.represent_base("occD2x", L, e64.N_ITER)
    assert np.array_equal(Dset["row"], rows), "re-extracted D2 rows differ from the frozen rows"
    P = dict(mu=z["P_mu"], U=z["P_U"], sd=z["P_sd"]); M = {k: z[k] for k in ("Sp_is", "type_ids", "type_isqrt", "Uo", "sw")}
    Z = e64.e61.pca_apply(P, Dset["V"]); del Dset["V"]
    Wr = e64.apply_maps(M, Z, Dset["tok"])
    out = dict(rows=int(len(rows)))
    # check the rebuilt representation against the frozen units (as e64 reconstruct)
    X = np.hstack([Wr.astype(np.float64), np.ones((len(Wr), 1))])
    for k, j in UNITS:
        y = fnpz[f"Y_{k}"][:, j].astype(np.float64)
        B, *_ = np.linalg.lstsq(X, y, rcond=None)
        out[f"rebuild_r2_{k}_{j}"] = float(1 - ((y - X @ B) ** 2).sum() / ((y - y.mean()) ** 2).sum())
    tk = e65.tokenizer()
    toks = np.load(EXT / "cache/occ/occD2_meta.npz")["tokens"].astype(np.int64)
    valid = (rows % 255) != 254
    cache = {}
    lab = np.zeros(len(rows), bool)
    for i in np.flatnonzero(valid):
        t = int(toks[rows[i] + 1]); lab[i] = cache.setdefault(t, e66.category(tk.decode([t]))) in ("preposition", "subordinator")
    out["label_base_rate"] = float(lab[valid].mean())
    seq = Dset["seq"]; ev = (seq % 2 == 0) & valid; od = (seq % 2 == 1) & valid

    def fit_eval(F, name, scale=False):
        Fv = F.astype(np.float64)
        mu, sd = Fv[ev].mean(0), Fv[ev].std(0) + 1e-9
        Fs = (Fv - mu) / sd if scale else Fv
        clf = LogisticRegression(C=1.0, max_iter=4000).fit(Fs[ev], lab[ev])
        s = clf.decision_function(Fs)
        auc = e64.e61.auroc(s[od], lab[od].astype(float))
        return clf, s, float(auc), (clf.coef_.ravel() / sd if scale else clf.coef_.ravel())

    def pullback(cz):
        raw = P["U"] @ (P["sd"] * cz)
        v = np.zeros(1152); v[keep] = raw
        return v / np.linalg.norm(v)

    cosd = lambda v, name: float(v @ dirs[name] / (np.linalg.norm(v) * np.linalg.norm(dirs[name])))
    # primary: probe on W (the unit's own representation), pattern pulled back exactly as d
    clf, s, auc, w = fit_eval(Wr, "W")
    ss = (s - s.mean()) / s.std()
    Zc = Z.astype(np.float64) - Z.mean(0)
    pat = pullback(Zc.T @ ss / len(ss))
    Wpat = (Wr.astype(np.float64).T @ ss) / len(ss)
    out["probe_W"] = dict(heldout_auroc=auc, cos_pattern_with_d={n: cosd(pat, n) for n in dirs},
                          cos_Wpattern_with_unit_Wpattern={f"{k}/{j}": float(Wpat @ z[f"pattern_{k}"][:, j] /
                                                                             (np.linalg.norm(Wpat) * np.linalg.norm(z[f"pattern_{k}"][:, j])))
                                                           for k, j in UNITS},
                          cos_filter_with_unit_Wpattern={f"{k}/{j}": float(w @ z[f"pattern_{k}"][:, j] /
                                                                            (np.linalg.norm(w) * np.linalg.norm(z[f"pattern_{k}"][:, j])))
                                                         for k, j in UNITS})
    # the unit itself as a classifier of the same label (oriented score), for reference
    out["unit_as_classifier_auroc"] = {f"{k}/{j}": float(e64.e61.auroc((z[f"sign_{k}"][j] * fnpz[f"Y_{k}"][:, j])[valid], lab[valid].astype(float)))
                                       for k, j in UNITS}
    # raw residual probe and class-mean difference
    Hraw = np.load(EXT / OUT["raw"]).astype(np.float32)
    clf2, s2, auc2, w2 = fit_eval(Hraw, "raw", scale=True)
    s2s = (s2 - s2.mean()) / s2.std()
    pat2 = (Hraw.T @ s2s.astype(np.float32)).astype(np.float64) / len(s2s); pat2[~keep] = 0
    w2m = w2.copy(); w2m[~keep] = 0
    cmd = Hraw[lab & valid].mean(0).astype(np.float64) - Hraw[~lab & valid].mean(0).astype(np.float64); cmd[~keep] = 0
    out["probe_raw"] = dict(heldout_auroc=auc2, cos_pattern_with_d={n: cosd(pat2, n) for n in dirs},
                            cos_filter_with_d={n: cosd(w2m, n) for n in dirs})
    out["class_mean_difference_raw"] = dict(cos_with_d={n: cosd(cmd, n) for n in dirs})
    e64.dump(out, OUT["probe"])
    e64.append_hash(OUT["probe"], "e67 next-token preposition/subordinator probe vs the confirmed direction")
    print(json.dumps(out, indent=1), flush=True)


# ---------------------------------------------------------------------------- stage: verdict (CPU)
def stage_verdict():
    e66, e65, e64 = mods()
    dec = json.loads((EXT / OUT["decoder"]).read_text()); act = json.loads((EXT / OUT["acts"]).read_text())
    prb = json.loads((EXT / OUT["probe"]).read_text())
    covered, partial = [], []
    for dn, rec in dec["dicts"].items():
        lat = act["dicts"][dn]["latents"]
        for name in ("logcosh/13", "skew/16"):
            for t in rec[name]["top"]:
                if abs(t["cos"]) >= COS_COVER and lat.get(str(t["latent"]), {}).get("behavioural_match"):
                    covered.append(dict(dict=dn, unit=name, latent=t["latent"], via="decoder_cos", value=t["cos"]))
            for t in act["dicts"][dn]["corr"][f"Y:{name}"]:
                if abs(t["r"]) >= R_COVER and lat.get(str(t["latent"]), {}).get("behavioural_match"):
                    covered.append(dict(dict=dn, unit=name, latent=t["latent"], via="activation_r", value=t["r"]))
            for mode in ("signed", "nonneg_pos", "nonneg_neg"):
                rc = rec["recon"][name][mode]
                if rc["k09"] is not None and rc["k09"] <= RECON_K:
                    atoms = rc["first5"][:rc["k09"]]
                    m = [lat.get(str(a), {}).get("behavioural_match") for a in atoms]
                    partial.append(dict(dict=dn, unit=name, mode=mode, k09=rc["k09"], atoms=atoms, match=m,
                                        majority_match=bool(sum(bool(x) for x in m) > len(atoms) / 2)))
    if covered:
        verdict = "COVERED"
    elif any(p["majority_match"] for p in partial):
        verdict = "PARTIALLY COVERED"
    else:
        verdict = "NOT COVERED"
    c = abs(prb["probe_W"]["cos_pattern_with_d"]["logcosh/13"])
    rel = "essentially the next-token-syntax probe direction" if c >= 0.9 else ("closely related to it" if c >= 0.7 else "distinct from it")
    out = dict(verdict=verdict, covered_hits=covered, partial_candidates=partial, probe_cos_logcosh13=c, probe_relation=rel)
    e64.dump(out, OUT["verdict"])
    e64.append_hash(OUT["verdict"], "e67 prior-feature verdict (rubric of NOTE_prior_feature_check.md)")
    print(json.dumps(out, indent=1), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["decoder", "acts", "probe", "verdict"])
    a = ap.parse_args()
    print("e67 sha256", hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), flush=True)
    e66, e65, e64 = mods()
    e64.check_swap()
    dict(decoder=stage_decoder, acts=stage_acts, probe=stage_probe, verdict=stage_verdict)[a.stage]()


if __name__ == "__main__":
    main()
