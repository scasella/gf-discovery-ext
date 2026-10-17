"""e71 -- per-family coverage readout, sparsity relative to the null, small-K readout pre-check
(NOTE_family_coverage_readout.md, hashed before any per-family number). Read-only on already-hashed outputs and
D2 (corpus 1000-1999). Amends no verdict; nominates nothing.

Stages
  calib    (CPU) (b) sparsity cuts from the 6 L12 anchors + 40 joint nulls only (with / without source);
           one streamed encode of resid_post_all L12 16k l0_small on e67's D2 L12 rows -> even-half |r(pre)|
           ranking of every L12 score (results/e71_rank_G12.npz); (c) held-out R^2 for anchors + joint nulls
           and the K_anc decision -> results/e71_calib.json (both hashed before any target cell is read)
  readout  (CPU) (b) bands and classes for every L12 unit; (c) R^2 for every other L12 score; per-family page data
           (e68 decoder / acts / matches / OMP; unit contexts; 16k-small latent contexts; e67 contexts for #13/#16)
           -> results/e71_readout.json
"""
import argparse, importlib.util, json, os, subprocess, sys, time
from pathlib import Path

import numpy as np

EXT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXT / "src"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_HOME", str(EXT / "hf_home"))
NOTE = "NOTE_family_coverage_readout.md"
L = 12
SOURCE = {7: "resid_post_all/L7_16k_big", 12: "resid_post_all/L12_16k_big"}
SMALL = ("resid_post_all", 12, "16k", "small")
SMALL_NAME = "resid_post_all/L12_16k_small"
ANCHORS = ["logcosh/13", "skew/16", "logcosh/11", "skew/23", "logcosh/33", "skew/33"]
REF = ("logcosh/13", "skew/16")
FAMILIES = {7: [["logcosh/18", "skew/36"], ["logcosh/33"], ["logcosh/49", "skew/8"], ["logcosh/52"], ["logcosh/57", "skew/33"]],
            12: [["logcosh/45", "skew/5"], ["logcosh/54", "skew/54"], ["logcosh/13", "skew/16"]]}
N_JOINT, NULL_MAX, ANC_MIN = 40, 2, 4
KS, R2_BAR, K_TOP = (4, 8, 16, 32, 64), 0.5, 64
N_CTX, CTX_TOP = 12, 3
SWAP_MAX_GB = 12.0
MODES = ("signed", "nonneg")
OUT = dict(calib="results/e71_calib.json", rank="results/e71_rank_G12.npz", readout="results/e71_readout.json")


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, EXT / rel)
    mod = importlib.util.module_from_spec(spec); sys.modules[name] = mod; spec.loader.exec_module(mod)
    return mod


def mods():
    e64 = _load("e64", "scripts/e64_occurrence_v2_discovery.py")
    e67 = _load("e67", "scripts/e67_prior_feature_check.py")
    e69 = _load("e69", "scripts/e69_sae_gap_pilot.py")
    return e64, e67, e69


def check_swap(e64):
    used = e64.swap_used_gb()
    assert used <= SWAP_MAX_GB, f"swap used {used:.1f} GB > {SWAP_MAX_GB} GB -- STOP"
    return used


def jl(rel):
    return json.loads((EXT / rel).read_text())


# ---------------------------------------------------------------------------- (b) sparsity relative to the null
def cells(drec, nm, dicts):
    out = []
    for dn in dicts:
        r = drec[dn]; kmax = r["k_max"]
        for mode in MODES:
            om = r["units"][nm]["omp"][mode]; q = r["null_recon"][mode]["k08"]
            k = om["k08"] if om["k08"] is not None else kmax + 1
            out.append(dict(dict=dn, mode=mode, k08=k, unreached=om["k08"] is None, k09=om["k09"],
                            cos_at_kmax=om["cos_at_kmax"], q1=q["p1"], q5=q["p5"], q50=q["p50"],
                            band="<=p1" if k <= q["p1"] else "<=p5" if k <= q["p5"] else "mid" if k <= q["p50"] else ">p50"))
    return out


def sstats(cl):
    pm = {m: dict(S1=sum(c["k08"] <= c["q1"] for c in cl if c["mode"] == m),
                  S5=sum(c["k08"] <= c["q5"] for c in cl if c["mode"] == m),
                  H=sum(c["k08"] > c["q50"] for c in cl if c["mode"] == m)) for m in MODES}
    return dict(per_mode=pm, S1=max(pm[m]["S1"] for m in MODES), S5=max(pm[m]["S5"] for m in MODES),
                H=min(pm[m]["H"] for m in MODES))


def choose_cut(anc, nul, N):
    """Admissible: <= NULL_MAX joint nulls pass. Max anchors; ties: fewer nulls, p1 before p5, larger c."""
    best = None
    for p in ("S1", "S5"):
        for c in range(1, N + 1):
            nn = sum(s[p] >= c for s in nul.values())
            if nn > NULL_MAX:
                continue
            passing = [a for a, s in anc.items() if s[p] >= c]
            key = (len(passing), -nn, 1 if p == "S1" else 0, c)
            if best is None or key > best["key"]:
                best = dict(key=key, stat=p, c=c, anchors_passing=passing, n_anchors=len(passing), n_null=nn)
    cd = next((c for c in range(1, N + 1) if sum(s["H"] >= c for s in nul.values()) <= NULL_MAX), None)
    informative = bool(best and best["n_anchors"] >= ANC_MIN and any(a in best["anchors_passing"] for a in REF))
    dense_anchors = [a for a, s in anc.items() if cd is not None and s["H"] >= cd]
    return dict(sparse_rule=best, informative=informative, dense_c=cd,
                dense_n_null=(sum(s["H"] >= cd for s in nul.values()) if cd is not None else None),
                dense_anchors_flagged=dense_anchors)


def sclass(st, cut):
    r = cut["sparse_rule"]
    if r is not None and st[r["stat"]] >= r["c"]:
        return "sparser"
    if cut["dense_c"] is not None and st["H"] >= cut["dense_c"]:
        return "denser"
    return "chance-like"


# ---------------------------------------------------------------------------- (c) small-K readout pre-check
def l12_inputs():
    prep = jl("results/e68_prep_G12.json")
    z = np.load(EXT / "results/e64_maps_L12.npz"); fnpz = np.load(EXT / "results/occ_units_v2_L12.npz")
    ns = np.load(EXT / "results/e68_prep_G12_nullscores.npy")
    sc = {}
    for nm in prep["units"]:
        if nm.startswith("null/"):
            sc[nm] = ns[:, int(nm.split("/")[1])].astype(np.float64)
        else:
            kd, j = nm.split("/"); sc[nm] = float(z[f"sign_{kd}"][int(j)]) * fnpz[f"Y_{kd}"][:, int(j)].astype(np.float64)
    rows = fnpz["row"].astype(np.int64)
    return prep, sc, rows


def rank_even(e67, H16, sc, names, even, bs=4096):
    """e69's ranking: |Pearson r| of the raw pre-activation with each score on the even half (streamed)."""
    import torch
    P = e67.sae_params(SMALL)
    We, be = torch.from_numpy(P["w_enc"].astype(np.float32)), torch.from_numpy(P["b_enc"].astype(np.float32))
    nl = We.shape[1]; idx = np.flatnonzero(even); m_ = len(idx)
    Y = np.stack([sc[nm][idx] for nm in names], 1)
    Yz = torch.from_numpy(((Y - Y.mean(0)) / Y.std(0)).astype(np.float32))
    s1 = torch.zeros(nl, dtype=torch.float64); s2 = torch.zeros(nl, dtype=torch.float64)
    sy = torch.zeros((nl, len(names)), dtype=torch.float64)
    with torch.no_grad():
        for a in range(0, m_, bs):
            x = torch.from_numpy(H16[idx[a:a + bs]].astype(np.float32))
            pre = x @ We + be
            s1 += pre.sum(0).double(); s2 += (pre * pre).sum(0).double(); sy += (pre.T @ Yz[a:a + bs]).double()
    var = s2 / m_ - (s1 / m_) ** 2
    r = ((sy / m_) / torch.sqrt(torch.clamp(var, min=1e-12))[:, None]).numpy(); r[~np.isfinite(r)] = 0
    top = np.stack([np.argsort(-np.abs(r[:, i]), kind="stable")[:K_TOP] for i in range(len(names))]).astype(np.int64)
    rv = np.stack([r[top[i], i] for i in range(len(names))]).astype(np.float32)
    return top, rv, P


def r2_curve(e69, H32, P, top, y, even, tinv, cnt):
    import torch
    We = torch.from_numpy(P["w_enc"][:, top].astype(np.float32)); be = torch.from_numpy(P["b_enc"][top].astype(np.float32))
    th = torch.from_numpy(P["threshold"][top].astype(np.float32))
    with torch.no_grad():
        pre = torch.from_numpy(H32) @ We + be
        a = pre * (pre > th); s = torch.minimum(torch.clamp(pre, min=0), th)
        X = torch.cat([a, s], 1).numpy().astype(np.float64)
    del pre, a, s
    for j in range(X.shape[1]):                                   # type-centre (label-free, all kept rows)
        X[:, j] -= (np.bincount(tinv, weights=X[:, j]) / cnt)[tinv]
    X = (X - X[even].mean(0)) / np.maximum(X[even].std(0), 1e-9)
    out = {}
    for K in KS:
        cols = list(range(K)) + list(range(K_TOP, K_TOP + K))
        out[K] = e69.heldout_r2(X[:, cols], y, even, "ridge")
    return out


def kstar(curve):
    return next((K for K in KS if curve[K] >= R2_BAR), None)


def stage_calib():
    e64, e67, e69 = mods()
    e64.require_hashed(NOTE, "scripts/e71_family_readout.py", "results/e68_decoder_G12.json", "results/e68_prep_G12.json",
                       "results/e68_prep_G12_nullscores.npy", "results/e67_raw_L12_rows.npy")
    assert not (EXT / OUT["calib"]).exists(), "calib runs once"
    check_swap(e64); t0 = time.time()
    dec = jl("results/e68_decoder_G12.json")["dicts"]
    nulls = [f"null/{i}" for i in range(N_JOINT)]
    out = dict(note=NOTE, sparsity={}, precheck={})
    for tag, dicts in (("with_source", list(dec)), ("without_source", [d for d in dec if d != SOURCE[L]])):
        anc = {a: sstats(cells(dec, a, dicts)) for a in ANCHORS}           # anchors and joint nulls only
        nul = {n: sstats(cells(dec, n, dicts)) for n in nulls}
        cut = choose_cut(anc, nul, len(dicts))
        out["sparsity"][tag] = dict(n_dicts=len(dicts), cut=cut, anchors=anc,
                                    anchor_class={a: sclass(s, cut) for a, s in anc.items()},
                                    null_class_counts={k: sum(sclass(s, cut) == k for s in nul.values()) for k in ("sparser", "chance-like", "denser")},
                                    null_stats=nul)
        print(f"[{time.time() - t0:.0f}s] (b) {tag}: rule {cut['sparse_rule']} informative {cut['informative']} dense c {cut['dense_c']}", flush=True)
    # (c): one streamed encode -> ranking for every L12 score (a ranking, not a readout); fits for anchors + nulls only
    prep, sc, rows = l12_inputs()
    names = list(prep["units"])
    H16 = np.load(EXT / "results/e67_raw_L12_rows.npy")
    assert H16.shape == (len(rows), 1152) and np.isfinite(H16).all()
    even = (rows // 255) % 2 == 0
    check_swap(e64)
    top, rv, P = rank_even(e67, H16, sc, names, even)
    np.savez(EXT / OUT["rank"], names=np.array(names), top=top, r_even=rv)
    e64.append_hash(OUT["rank"], "e71 calib: even-half |r(pre)| top-64 ranking of every L12 score, resid_post_all L12 16k small (a ranking, not a readout)")
    print(f"[{time.time() - t0:.0f}s] ranking done; swap {e64.swap_used_gb():.1f} GB", flush=True)
    H32 = H16.astype(np.float32); del H16
    toks = np.load(EXT / "cache/occ/occD2_meta.npz")["tokens"].astype(np.int64)
    _, tinv = np.unique(toks[rows], return_inverse=True); cnt = np.bincount(tinv)
    curves = {}
    for nm in ANCHORS + nulls:
        check_swap(e64)
        curves[nm] = r2_curve(e69, H32, P, top[names.index(nm)], sc[nm], even, tinv, cnt)
    Kanc = next((K for K in KS if sum(curves[a][K] >= R2_BAR for a in ANCHORS) >= ANC_MIN), None)
    nat = sum(curves[n][Kanc] >= R2_BAR for n in nulls) if Kanc else None
    out["precheck"] = dict(dictionary=SMALL_NAME, ks=list(KS), r2_bar=R2_BAR, K_anc=Kanc, n_null_at_K_anc=nat,
                           informative=bool(Kanc is not None and nat <= NULL_MAX),
                           anchors={a: dict(r2=curves[a], kstar=kstar(curves[a])) for a in ANCHORS},
                           nulls={n: dict(r2=curves[n], kstar=kstar(curves[n])) for n in nulls},
                           null_kstar_counts={str(k): sum(kstar(curves[n]) == k for n in nulls) for k in list(KS) + [None]},
                           null_r2_quantiles={K: [float(np.quantile([curves[n][K] for n in nulls], q)) for q in (0.05, 0.5, 0.95)] for K in KS},
                           rows_fp16="results/e67_raw_L12_rows.npy (fp16 rounding of the residual disclosed)")
    print(f"[{time.time() - t0:.0f}s] (c) K_anc {Kanc} nulls at K_anc {nat} informative {out['precheck']['informative']}", flush=True)
    e64.dump(out, OUT["calib"])
    e64.append_hash(OUT["calib"], "e71 calib: (b) sparsity cuts from anchors + joint nulls; (c) anchor/null R^2 and K_anc; before any target cell")


# ---------------------------------------------------------------------------- readout
def decode_ctx(tk, toks, r, before=15, after=3):
    s0 = (r // 255) * 255
    return (tk.decode([int(t) for t in toks[max(s0, r - before):r]]) + "[[" + tk.decode([int(toks[r])]) + "]]"
            + tk.decode([int(t) for t in toks[r + 1:min(s0 + 255, r + 1 + after)]]))


def unit_scores(Lr):
    z = np.load(EXT / f"results/e64_maps_L{Lr}.npz"); fnpz = np.load(EXT / f"results/occ_units_v2_L{Lr}.npz")
    return z, fnpz, fnpz["row"].astype(np.int64)


def stage_readout():
    import torch
    from transformers import AutoTokenizer
    e64, e67, e69 = mods()
    e64.require_hashed(NOTE, "scripts/e71_family_readout.py", OUT["calib"], OUT["rank"], "results/e68_census.json",
                       "results/e68_decoder_G7.json", "results/e68_acts_G7.json", "results/e68_decoder_G12.json",
                       "results/e68_acts_G12.json", "results/e68_prep_G7.json", "results/e68_prep_G12.json",
                       "results/e64_labels.json", "results/e67_acts.json", "results/e67_raw_L12_rows.npy")
    check_swap(e64); t0 = time.time()
    calib = jl(OUT["calib"]); census = jl("results/e68_census.json")
    tk = AutoTokenizer.from_pretrained(str(sorted((EXT / "hf_home/hub/models--google--gemma-3-1b-pt/snapshots").glob("*"))[0]))
    toks = np.load(EXT / "cache/occ/occD2_meta.npz")["tokens"].astype(np.int64)
    labels = jl("results/e64_labels.json"); e67a = jl("results/e67_acts.json")["dicts"]
    out = dict(note=NOTE, calib_sha=e64.sha(EXT / OUT["calib"]), sparsity={}, precheck={}, families=[])
    # (b) every L12 unit
    dec12 = jl("results/e68_decoder_G12.json")["dicts"]
    units12 = [nm for nm in dec12[SOURCE[12]]["units"] if not nm.startswith("null/")]
    for tag, dicts in (("with_source", list(dec12)), ("without_source", [d for d in dec12 if d != SOURCE[12]])):
        cut = calib["sparsity"][tag]["cut"]
        st = {nm: sstats(cells(dec12, nm, dicts)) for nm in units12}
        out["sparsity"][tag] = {nm: dict(stats=s, cls=sclass(s, cut)) for nm, s in st.items()}
    # (c) every other L12 score
    prep, sc, rows12 = l12_inputs()
    rk = np.load(EXT / OUT["rank"]); rnames = [str(x) for x in rk["names"]]
    H16 = np.load(EXT / "results/e67_raw_L12_rows.npy"); H32 = H16.astype(np.float32); del H16
    even = (rows12 // 255) % 2 == 0
    _, tinv = np.unique(toks[rows12], return_inverse=True); cnt = np.bincount(tinv)
    P = e67.sae_params(SMALL)
    pc = dict(calib["precheck"]["anchors"])
    for nm in units12:
        if nm in pc:
            continue
        check_swap(e64)
        cv = r2_curve(e69, H32, P, rk["top"][rnames.index(nm)], sc[nm], even, tinv, cnt)
        pc[nm] = dict(r2=cv, kstar=kstar(cv))
    out["precheck"] = pc
    print(f"[{time.time() - t0:.0f}s] (b)/(c) done for {len(units12)} L12 units", flush=True)
    # family pages
    for Lr, fams in FAMILIES.items():
        g = census["groups"][str(Lr)]
        dec = jl(f"results/e68_decoder_G{Lr}.json")["dicts"]; act = jl(f"results/e68_acts_G{Lr}.json")["dicts"]
        prepL = jl(f"results/e68_prep_G{Lr}.json")["units"]
        z, fnpz, rows = unit_scores(Lr)
        lab = {f"{u['kind']}/{u['id']}": u for u in labels["layers"][str(Lr)]["units"]}
        k1 = int(np.ceil(0.01 * len(rows)))
        for listed in fams:
            frec = [f for f in g["families"] if set(listed) <= set(f["members"])]
            assert len(frec) == 1, f"L{Lr} family {listed}: {len(frec)} census records"
            frec = frec[0]
            page = dict(layer=Lr, listed=listed, census=frec, members={})
            for nm in frec["members"]:
                kd, j = nm.split("/")
                s = float(z[f"sign_{kd}"][int(j)]) * fnpz[f"Y_{kd}"][:, int(j)].astype(np.float64)
                order = np.argsort(-s, kind="stable"); tail = np.zeros(len(rows), bool); tail[order[:k1]] = True
                lb = lab.get(nm, {})
                heavy = "hi" if lb.get("heavy_sign", 1.0) > 0 else "lo"
                m = dict(census_unit=g["units"].get(nm), prep=dict(T=[tk.decode([t]) for t in prepL[nm]["pos"]["T"]],
                                                                    s_self=prepL[nm]["pos"]["s_self"], base=prepL[nm]["pos"]["base"],
                                                                    testable=prepL[nm]["pos"]["testable"]),
                         contexts_strongest=[decode_ctx(tk, toks, int(rows[i])) for i in order[:N_CTX]],
                         contexts_e64_random_tail=lb.get("contexts", {}).get(heavy))
                pool_c, pool_r, maxcos, tested_all = [], [], {}, []
                for dn, r in dec.items():
                    u = r["units"][nm]
                    maxcos[dn] = dict(max_abs_cos=u["max_abs_cos"], null_p50=r["null_subspace"]["p50"], null_p999=r["null_subspace"]["p999"],
                                      above_p999=u["max_abs_cos"] > r["null_subspace"]["p999"])
                    pool_c += [dict(dict=dn, latent=t["latent"], cos=t["cos"]) for t in u["top"]]
                    a = act[dn]["units"][nm]
                    pool_r += [dict(dict=dn, latent=t["latent"], r=t["r"], r_within_type=t["r_within_type"]) for t in a["r_top"]]
                    tested_all += [dict(dict=dn, latent=int(li), **t) for li, t in a["tested"].items()]
                pool_c = sorted(pool_c, key=lambda t: -abs(t["cos"]))[:5]; pool_r = sorted(pool_r, key=lambda t: -abs(t["r"]))[:5]
                tmap = {(t["dict"], t["latent"]): t for t in tested_all}
                for t in pool_c + pool_r:
                    t["tested"] = tmap.get((t["dict"], t["latent"]))
                    if Lr == 12 and REF[0] in frec["members"] and t["dict"] in e67a and str(t["latent"]) in e67a[t["dict"]]["latents"]:
                        t["e67_contexts"] = e67a[t["dict"]]["latents"][str(t["latent"])]["contexts"]
                mo = max(tested_all, key=lambda t: t["overlap"])
                m.update(decoder_top5=pool_c, acts_top5=pool_r, max_cos_by_dict=maxcos,
                         n_above_p999=sum(v["above_p999"] for v in maxcos.values()),
                         n_above_p999_without_source=sum(v["above_p999"] for dn, v in maxcos.items() if dn != SOURCE[Lr]),
                         n_tested=len(tested_all), matches=[t for t in tested_all if t["match"]],
                         max_overlap=dict(dict=mo["dict"], latent=mo["latent"], overlap=mo["overlap"], threshold=mo["overlap_threshold"],
                                          null_p99=mo["overlap_null_p99"]),
                         omp=cells(dec, nm, list(dec)))
                if Lr == 12:
                    m["sparsity"] = {tag: out["sparsity"][tag][nm] for tag in out["sparsity"]}
                    m["precheck"] = pc[nm]
                    uc = sorted(dec[SMALL_NAME]["units"][nm]["top"], key=lambda t: -abs(t["cos"]))[:CTX_TOP]
                    ur = act[SMALL_NAME]["units"][nm]["r_top"][:CTX_TOP]
                    lat = list(dict.fromkeys([t["latent"] for t in uc] + [t["latent"] for t in ur]))
                    Wc = torch.from_numpy(P["w_enc"][:, lat].astype(np.float32)); bc = torch.from_numpy(P["b_enc"][lat].astype(np.float32))
                    thc = torch.from_numpy(P["threshold"][lat].astype(np.float32))
                    with torch.no_grad():
                        pre = torch.from_numpy(H32) @ Wc + bc; A = (pre * (pre > thc)).numpy()
                    del pre
                    lc = {}
                    for jj, li in enumerate(lat):
                        o = np.argsort(-A[:, jj], kind="stable")[:N_CTX]; o = o[A[o, jj] > 0]
                        t = tmap.get((SMALL_NAME, li), {})
                        lc[str(li)] = dict(cos=next((x["cos"] for x in dec[SMALL_NAME]["units"][nm]["top"] if x["latent"] == li), None),
                                           r=t.get("r"), match=t.get("match"), n_active=int((A[:, jj] > 0).sum()),
                                           top12_in_unit_tail=float(tail[o].mean()) if len(o) else None,
                                           contexts=[decode_ctx(tk, toks, int(rows[i])) for i in o])
                    del A
                    m["small16k_latent_contexts"] = lc
                page["members"][nm] = m
            out["families"].append(page)
            print(f"[{time.time() - t0:.0f}s] page L{Lr} {listed}", flush=True)
    e64.dump(out, OUT["readout"])
    e64.append_hash(OUT["readout"], "e71 readout: (b) classes, (c) R^2 for every L12 unit, per-family page data")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["calib", "readout"])
    a = ap.parse_args()
    dict(calib=stage_calib, readout=stage_readout)[a.stage]()


if __name__ == "__main__":
    main()
