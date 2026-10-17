"""e72 independent same-subspace verification (NOTE_raw_residual_pilot.md section 7; post hoc, burned D2 only, CPU).

Re-derives the coverage numbers that the decision rests on by a separate code path. It does not call e64.apply_maps,
e69.error_rows or e63.demean.
  V1  map route re-implemented: numpy JumpReLU error (fp32), own demean3 (bincount group means, float64), the frozen
      PCA (U, sd) and per-type / global maps read from e72_units_raw_L12.npz, own per-type loop. Each checked unit's ES
      must agree with e72_coverage.json within 1e-3.
  V2  fresh baseline: 10,000 new uniform directions in the 64-d W space (seed 7300); each checked unit's rank.
  V3  single-filter route (descriptive): least-squares pullback filter f in the 1,151 demean3'd raw coordinates
      (f_u = F b_u, F = (Vh'Vh)^-1 Vh'W), its R^2, and var(e.f)/var(h.f) for the units and for the V2 directions.
Checked units: every lead, every control unit, and the 5 highest-ES units.

usage: e72_verify.py OUT.json [--test-dir DIR]
"""
import argparse, hashlib, json, sys, time
from pathlib import Path

import numpy as np

EXT = Path(__file__).resolve().parents[1]
DROP, N_ITER, N_ICA, N_RAND, SEED = 1038, 3, 64, 10_000, 7300
KEEP = np.ones(1152, bool); KEEP[DROP] = False
MIN_TYPE_OCC, MIN_POS = 8, 4
T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:.0f}s]", *a, flush=True)


def kept_rows(meta):
    tok, seq, pos = meta["tokens"].astype(np.int64), meta["seq"].astype(np.int64), meta["pos"].astype(np.int64)
    _, inv, cnt = np.unique(tok, return_inverse=True, return_counts=True)
    r = np.flatnonzero((cnt[inv] >= MIN_TYPE_OCC) & (pos >= MIN_POS))
    return r, tok[r], seq[r], pos[r]


def demean3(V, keys):
    """Own implementation: subtract group means over type, document, position, cycled 3 times (float64)."""
    gs = [np.unique(k, return_inverse=True)[1] for k in keys]
    for _ in range(N_ITER):
        for g in gs:
            cnt = np.bincount(g).astype(np.float64)
            for j in range(V.shape[1]):
                V[:, j] -= (np.bincount(g, weights=V[:, j]) / cnt)[g]
    return V


def sae_error(H16, P):
    We, be, th, Wd, bd = (P[k].astype(np.float32) for k in ("w_enc", "b_enc", "threshold", "w_dec", "b_dec"))
    E = np.empty((len(H16), int(KEEP.sum())), np.float64)
    for a in range(0, len(H16), 4096):
        x = np.asarray(H16[a:a + 4096], dtype=np.float32)
        pre = x @ We + be
        zc = np.where(pre > th, pre, 0).astype(np.float32)
        E[a:a + 4096] = (x - zc @ Wd - bd)[:, KEEP]
    return E


def maps(Z, tok, z):
    """Own per-type loop with the frozen maps: own type means, per-type or pooled matrix, centre, global re-whitening."""
    out = np.empty_like(Z)
    pos = {int(t): i for i, t in enumerate(z["type_ids"])}
    order = np.argsort(tok, kind="stable"); ts = tok[order]
    bounds = np.flatnonzero(np.diff(ts)) + 1
    for ii in np.split(order, bounds):
        Zc = Z[ii] - Z[ii].mean(0)
        i = pos.get(int(tok[ii[0]]))
        out[ii] = Zc @ (z["type_isqrt"][i] if i is not None else z["Sp_is"])
    out -= out.mean(0)
    return out @ z["Uo"] / z["sw"]


def es_dirs(Wh, We, G):
    """Direct projections; only for a few columns (n x k is materialised)."""
    assert G.shape[1] <= 64
    A = Wh - Wh.mean(0); B = We - We.mean(0)
    yh, ye = A @ G, B @ G
    return ye.var(0) / yh.var(0)


def es_many(Wh, We, G):
    """Many directions via the 64 x 64 second moments (no n x k array)."""
    A = Wh - Wh.mean(0); B = We - We.mean(0); n = len(A)
    Chh, Cee = A.T @ A / n, B.T @ B / n
    return (G * (Cee @ G)).sum(0) / (G * (Chh @ G)).sum(0)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("out"); ap.add_argument("--test-dir", default=None)
    a = ap.parse_args()
    if a.test_dir:
        td = Path(a.test_dir); res_dir, H_path = td, td / "H_D.npy"
        sz = np.load(td / "sae.npz"); P = {k: sz[k] for k in sz.files}
    else:
        res_dir, H_path = EXT / "results", EXT / "results/e67_raw_L12_rows.npy"
        from safetensors.numpy import load_file
        snap = sorted((EXT / "hf_home/hub/models--google--gemma-scope-2-1b-pt/snapshots").glob("*"))[0]
        P = load_file(str(snap / "resid_post_all/layer_12_width_16k_l0_big/params.safetensors"))
    z = np.load(res_dir / "e72_units_raw_L12.npz")
    cov = json.loads((res_dir / "e72_coverage.json").read_text())
    meta = np.load(EXT / "cache/occ/occD2_meta.npz")
    r, tok, seq, pos = kept_rows(meta)
    assert np.array_equal(r, z["row"]), "kept rows differ"
    units = {f"{u['kind']}/{u['id']}": u for u in cov["units"]}
    check = list(dict.fromkeys(cov["leads"] + [f"{u['kind']}/{u['id']}" for u in cov["controls"]]
                               + [k for k, _ in sorted(units.items(), key=lambda kv: -kv[1]["es"])[:5]]))
    log("checking", check)
    H16 = np.load(H_path)
    # ---- h side (V3 needs demean3'd raw h)
    Vh = demean3(H16[:, KEEP].astype(np.float64), (tok, seq, pos))
    Zh = ((Vh - Vh.mean(0)) @ z["P_U"]) / z["P_sd"]
    Wh_own = maps(Zh, tok, z)
    Wh = z["W"].astype(np.float64)
    w_dev = float(np.abs(Wh_own - Wh).max())
    log("V1 own maps on h vs stored W: max abs", w_dev)
    n = len(Vh)
    Ahh = Vh.T @ Vh / n
    VhW = Vh.T @ (Wh - Wh.mean(0)) / n
    del Vh, Zh, Wh_own
    # ---- e side
    Ve = demean3(sae_error(H16, P), (tok, seq, pos)); del H16
    Ze = ((Ve - Ve.mean(0)) @ z["P_U"]) / z["P_sd"]
    We = maps(Ze, tok, z)
    Aee = Ve.T @ Ve / n
    del Ve, Ze
    # ---- V1: units through the re-implemented route
    out = dict(checked=check, w_own_vs_stored=w_dev, units={})
    B = {kd: z[f"readout_{kd}"][:N_ICA] for kd in ("logcosh", "skew") if f"readout_{kd}" in z.files}
    for name in check:
        kd, j = name.split("/"); j = int(j)
        b = B[kd][:, j:j + 1]
        v1 = float(es_dirs(Wh, We, b)[0])
        out["units"][name] = dict(es_coverage=units[name]["es"], es_v1=v1, v1_agree=bool(abs(v1 - units[name]["es"]) <= 1e-3),
                                  lead=units[name]["lead"], control=units[name]["control"])
    # ---- V2: fresh random directions
    G = np.random.default_rng(SEED).standard_normal((N_ICA, N_RAND)); G /= np.linalg.norm(G, axis=0, keepdims=True)
    rv2 = es_many(Wh, We, G)
    out["v2_random"] = dict(seed=SEED, p50=float(np.median(rv2)), p999=float(np.quantile(rv2, 0.999)), max=float(rv2.max()))
    for name, u in out["units"].items():
        u["v2_pct"] = float((rv2 < u["es_v1"]).mean()); u["v2_beats_max"] = bool(u["es_v1"] > rv2.max())
    # ---- V3: least-squares pullback filters in raw coordinates
    lam = 1e-6 * np.trace(Ahh) / len(Ahh)
    F = np.linalg.solve(Ahh + lam * np.eye(len(Ahh)), VhW)                      # 1151 x 64: filter of W's columns
    Qh, Qe = F.T @ Ahh @ F, F.T @ Aee @ F
    Cw = np.cov((Wh - Wh.mean(0)).T, bias=True)
    def ls(Gm):
        num = np.einsum("ik,ij,jk->k", Gm, Qe, Gm); den = np.einsum("ik,ij,jk->k", Gm, Qh, Gm)
        r2 = den / np.einsum("ik,ij,jk->k", Gm, Cw, Gm)                         # var(h.f)/var(y): R^2 of the LS readout
        return num / den, r2
    rls, r2r = ls(G)
    out["v3_random"] = dict(p50=float(np.median(rls)), p999=float(np.quantile(rls, 0.999)), max=float(rls.max()),
                            r2_p50=float(np.median(r2r)), r2_min=float(r2r.min()))
    for name, u in out["units"].items():
        kd, j = name.split("/"); j = int(j)
        e_, r2 = ls(B[kd][:, j:j + 1])
        u.update(es_ls=float(e_[0]), ls_r2=float(r2[0]), ls_pct=float((rls < e_[0]).mean()), ls_beats_max=bool(e_[0] > rls.max()))
    leads = [k for k, u in out["units"].items() if u["lead"]]
    out["verified_leads"] = [k for k in leads if out["units"][k]["v1_agree"] and out["units"][k]["v2_beats_max"]]
    out["all_checked_v1_agree"] = bool(all(u["v1_agree"] for u in out["units"].values()))
    out["seconds"] = round(time.time() - T0)
    Path(a.out).write_text(json.dumps(out, indent=1))
    for k, u in out["units"].items():
        log(k, {x: (round(v, 4) if isinstance(v, float) else v) for x, v in u.items()})
    log("V2 random", out["v2_random"], "V3 random", out["v3_random"])
    log("verified leads", out["verified_leads"], "all V1 agree", out["all_checked_v1_agree"])


if __name__ == "__main__":
    main()
