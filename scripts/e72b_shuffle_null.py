"""e72b -- selection-matched shuffle null for the e72 raw-residual lead (NOTE_shuffle_null.md, hashed before any
number it governs). Burned D2 only (corpus 1000-1999), re-ordered within each sequence; no fresh text.

Phases
  capture  (MPS, only after the user confirms) drift check: the first 32 D2 sequences re-captured vs
           results/e67_raw_L12_rows.npy; then for k = 1..5: shuffle meta -> cache/occ/occS2b{k}_meta.npz and raw L12
           kept rows (fp16, e69 p1's residual()) -> results/e72b_S2b{k}_raw_L12_rows.npy, each hashed as written;
           results/e72b_capture.json (hashed)
  analyze  (CPU) per set (S2 = shuffle 0, must reproduce e72's S2 block; S2b1..5): e72's S2 branch mirrored line for
           line (tag, rows, seed parametrised); D2 filters on each set; secondary: e72's full lead rule recomputed on
           the set (descriptive); the NOTE section 6 reading -> results/e72b_null_arrays.npz, results/e72b_null.json
           (hashed)

--test-dir DIR: smoke test. capture writes the real shuffle metas into DIR/cache/occ and SYNTHETIC fp16 rows into
DIR (no model, no forward pass); analyze reads them, runs S2 on the real hashed S2 rows (numbers already in
e72_coverage.json), writes every output to DIR and hashes into DIR/HASHES.txt.
"""
import argparse, gc, importlib.util, json, os, shutil, sys, time
from pathlib import Path

import numpy as np

EXT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXT / "src"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_HOME", str(EXT / "hf_home"))

NOTE = "NOTE_shuffle_null.md"
SELF = "scripts/e72b_shuffle_null.py"
E72 = "scripts/e72_raw_residual_pilot.py"
E72_SHA = "338c10b252b816eb40072d4b6492f81c072850edf00a1fbf1995d1cfeb27d394"
SEQ0, N_SEQ, SEQ_LEN = 1000, 1000, 255
KS = (1, 2, 3, 4, 5)
SHUF_OFFSET = 700_000                                  # e60 used 600_000 + seq_start + i for S2 (601,000-601,999)
SEED_RAND = {0: 7201, **{k: 7210 + k for k in KS}}     # 7201 = e72's S2 seed; D2 7200 and V2 7300 not reused
N_DRIFT = 32
S_D2, LEADS_D2, N_SCREENED_D2 = 4, 1, 60               # e72_coverage.json: screened above max, leads, screened
D2_UNITS = ("skew/49", "logcosh/37", "skew/52", "logcosh/49")
S2_ROWS = "results/e69_p1_S2_raw_L12_rows.npy"


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, EXT / rel)
    mod = importlib.util.module_from_spec(spec); sys.modules[name] = mod; spec.loader.exec_module(mod)
    return mod


e72 = _load("e72", E72)
e64, e63, e61, e69 = e72.e64, e72.e63, e72.e61, e72.e69
KINDS, N_ICA, N_RAND, ICA_REPRO = e72.KINDS, e72.N_ICA, e72.N_RAND, e72.ICA_REPRO
T0 = time.time()


def t():
    return f"[{time.time() - T0:.0f}s]"


def shuf_seed(k, i):
    return SHUF_OFFSET + 10_000 * k + SEQ0 + i


def tag_of(k):
    return "occS2" if k == 0 else f"occS2b{k}"


class Cfg:
    def __init__(self, test_dir=None):
        self.test = test_dir is not None
        if self.test:
            self.root = Path(test_dir).resolve()
            (self.root / "cache/occ").mkdir(parents=True, exist_ok=True)
            e64.HASHES = self.root / "HASHES.txt"
            e64.EXT = self.root                     # e64.rows_only reads <EXT>/cache/occ; absolute rels stay absolute
            s2m = self.root / "cache/occ/occS2_meta.npz"
            if not s2m.exists():
                shutil.copy(EXT / "cache/occ/occS2_meta.npz", s2m)
        else:
            self.root = EXT

    def meta(self, k):
        return self.root / f"cache/occ/{tag_of(k)}_meta.npz"

    def rows_file(self, k):
        if k == 0:
            return EXT / S2_ROWS
        return (self.root if self.test else EXT / "results") / f"e72b_S2b{k}_raw_L12_rows.npy"

    def out(self, name):
        return (self.root if self.test else EXT / "results") / name

    def rel(self, p):
        p = Path(p)
        return str(p) if self.test else str(p.relative_to(EXT))


def integrity(cfg):
    assert e64.sha(EXT / E72) == E72_SHA, "the e72 script changed: the mirrored S2 branch is not the frozen one"
    if cfg.test:
        return
    e64.require_hashed(NOTE, SELF, E72, "results/e72_units_raw_L12.npz", "results/e72_coverage.json",
                       "results/e72_verify.json", "results/e67_raw_L12_rows.npy", S2_ROWS)
    e72.integrity(e72.Cfg())                       # e72's NOTE/script hashed; e64 frozen; choice file unchanged


# ---------------------------------------------------------------------------- capture
def shuffled_tokens(k, tD):
    T = tD.reshape(N_SEQ, SEQ_LEN).copy()
    for i in range(N_SEQ):
        T[i] = T[i][np.random.default_rng(shuf_seed(k, i)).permutation(SEQ_LEN)]
    return T


def phase_capture(cfg):
    mD, mS = np.load(EXT / "cache/occ/occD2_meta.npz"), np.load(EXT / "cache/occ/occS2_meta.npz")
    tD, tS2 = mD["tokens"].astype(np.int64), mS["tokens"].astype(np.int64)
    assert not bool(mD["shuffled"]) and bool(mS["shuffled"]) and str(mD["source"]) == "corpus"
    assert np.array_equal(mD["seq"], np.repeat(np.arange(SEQ0, SEQ0 + N_SEQ), SEQ_LEN))
    assert np.array_equal(mD["pos"], np.tile(np.arange(1, SEQ_LEN + 1), N_SEQ))
    TD, TS2 = tD.reshape(N_SEQ, SEQ_LEN), tS2.reshape(N_SEQ, SEQ_LEN)
    T = {k: shuffled_tokens(k, tD) for k in KS}
    sD = np.sort(TD, 1)
    same_seq = {}
    for k in KS:
        assert np.array_equal(np.sort(T[k], 1), sD), f"S2b{k} is not a per-sequence permutation of D2"
        others = dict(D2=TD, S2=TS2, **{f"S2b{j}": T[j] for j in KS if j < k})
        for nm, O in others.items():
            n_same = int((T[k] == O).all(1).sum())
            same_seq[f"S2b{k}_vs_{nm}"] = n_same
            assert not np.array_equal(T[k], O) and n_same == 0, f"S2b{k} repeats {nm} in {n_same} sequences"
    print(t(), "shuffles built and checked: per-sequence permutations of D2; no sequence equal to D2, S2 or another",
          flush=True)
    model = tok = DEVICE = None
    drift = None
    if not cfg.test:
        import torch
        e64.check_swap()
        import gfdisc.model as gm
        gm.MODEL_NAME = str(sorted((EXT / "hf_home/hub/models--google--gemma-3-1b-pt/snapshots").glob("*"))[0])
        from gfdisc.env import DEVICE
        from gfdisc.model import capture
        assert DEVICE == "mps", DEVICE
        model, tok = gm.load_model(token=None)
        LAYER = e72.LAYER

        def residual(toks, rows):                  # verbatim from e69 phase_p1
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

        # ---- drift check: the first N_DRIFT D2 sequences re-captured (same batches as e67/e69)
        D2rows = np.load(EXT / "results/occ_units_v2_L12.npz")["row"].astype(np.int64)
        sel = D2rows < N_DRIFT * SEQ_LEN
        HD = residual(tD[:N_DRIFT * SEQ_LEN], D2rows[sel])
        ref = np.asarray(np.load(EXT / "results/e67_raw_L12_rows.npy", mmap_mode="r")[:int(sel.sum())])
        a32, b32 = HD.astype(np.float32), ref.astype(np.float32)
        ac, bc = a32 - a32.mean(0), b32 - b32.mean(0)
        cc = (ac * bc).sum(0) / np.sqrt((ac * ac).sum(0) * (bc * bc).sum(0) + 1e-30)
        drift = dict(n_seq=N_DRIFT, n_rows=int(sel.sum()), share_identical=float((HD == ref).mean()),
                     max_abs_diff=float(np.abs(a32 - b32).max()), min_coord_corr=float(np.nanmin(cc)))
        del HD, ref, a32, b32, ac, bc
        print(t(), "D2 drift check", drift, flush=True)
        assert drift["share_identical"] >= 0.99 and drift["min_coord_corr"] >= 0.999, f"capture drifted: {drift}"
    sh = {}
    for k in KS:
        tag, mp, rp = tag_of(k), cfg.meta(k), cfg.rows_file(k)
        if mp.exists():                            # resume after a crash: the meta must be the hashed, same shuffle
            e64.require_hashed(cfg.rel(mp))
            assert np.array_equal(np.load(mp)["tokens"].astype(np.int64), T[k].reshape(-1)), f"{mp} differs"
        else:
            np.savez(mp, tokens=T[k].reshape(-1).astype(np.int32), seq=mD["seq"], pos=mD["pos"], shuffled=True,
                     source="corpus", layers=np.asarray([e72.LAYER]))
            e64.append_hash(cfg.rel(mp), f"e72b capture: {tag} meta (D2 shuffled within sequences, seeds "
                                         f"{shuf_seed(k, 0)}-{shuf_seed(k, N_SEQ - 1)}), before any forward pass")
        rows = e64.rows_only(tag)[0]["row"].astype(np.int64)
        if rp.exists():
            e64.require_hashed(cfg.rel(rp))
            print(t(), f"{tag}: rows already saved and hashed; skipping", flush=True)
        else:
            if cfg.test:                           # synthetic rows: no model, no real activation
                g = np.random.default_rng(900 + k)
                H = (g.standard_normal((len(rows), 1152), dtype=np.float32) * 2.0
                     + g.laplace(size=(len(rows), 8)).astype(np.float32) @ g.standard_normal((8, 1152), dtype=np.float32)
                     ).astype(np.float16)
            else:
                H = residual(T[k].reshape(-1), rows)
            np.save(rp, H); del H; gc.collect()
            if not cfg.test and DEVICE == "mps":
                import torch
                torch.mps.empty_cache()
            e64.append_hash(cfg.rel(rp), f"e72b capture: {tag} raw L12 kept rows, fp16" + (" (SYNTHETIC smoke rows)" if cfg.test else ""))
        sh[tag] = dict(k=k, seed_first=shuf_seed(k, 0), seed_last=shuf_seed(k, N_SEQ - 1), seed_rand=SEED_RAND[k],
                       n_rows=int(len(rows)), meta=cfg.rel(mp), meta_sha256=e64.sha(mp), rows=cfg.rel(rp),
                       rows_sha256=e64.sha(rp))
        print(t(), f"{tag}: {len(rows)} kept rows; swap {e64.swap_used_gb():.1f} GB", flush=True)
    del model
    res = dict(note_sha256=e64.sha(EXT / NOTE), script_sha256=e64.sha(__file__), synthetic=cfg.test, drift=drift,
               same_sequence_counts=same_seq, shuffles=sh, seconds=round(time.time() - T0))
    e72.dump(res, cfg.out("e72b_capture.json"))
    e64.append_hash(cfg.rel(cfg.out("e72b_capture.json")), "e72b capture: drift check, seeds, row counts"
                    + (" (SYNTHETIC smoke)" if cfg.test else ""))
    print(t(), "CAPTURE done", flush=True)


# ---------------------------------------------------------------------------- analyze
def run_set(k, cfg, z, P, M, params, tk, n_rows):
    tag, seed = tag_of(k), SEED_RAND[k]
    # ---- e72.s2_descriptive, mirrored line for line (tag, rows file, n_rows and seed parametrised)
    S, fullS = e72.rows(tag, n_rows)
    HS = np.load(cfg.rows_file(k))
    assert HS.shape == (len(S["row"]), 1152)
    VS = e72.raw_V(HS, S["groups"])
    ZS = e72.project(VS, P, own_mean=False); del VS; gc.collect()     # D2's PCA applied (e64 represent_other)
    WS = e64.apply_maps(M, ZS, S["tok"])
    ZeS, WeS, stateS, fvuS, _ = e72.error_W(HS, S, P, M, params); del HS, ZeS, ZS; gc.collect()
    CS = e72.covs(WS.astype(np.float64), WeS)
    G = np.random.default_rng(seed).standard_normal((N_ICA, N_RAND)); G /= np.linalg.norm(G, axis=0, keepdims=True)
    rS = e72.es(CS, G)[0]
    mx = float(rS.max())
    Sset = dict(seq=S["seq"], tinv=S["tinv"], tok=S["tok"], row=S["row"])
    X1 = np.hstack([WS.astype(np.float64), np.ones((len(WS), 1))])
    out = dict(tag=tag, seed_rand=seed, n_rows=int(len(WS)), fvu_raw=fvuS, random_es=e72.qs(rS), kinds={}, d2_units={})
    mixS, fit = {}, {}
    for kd in KINDS:
        Y, mix, us = e63.ica_units(WS, Sset, N_ICA, kd)
        mixS[kd] = mix
        if Y.shape[1] == 0:
            out["kinds"][kd] = dict(fit_failed=True); continue
        B, *_ = np.linalg.lstsq(X1, Y.astype(np.float64), rcond=None)
        e_ = e72.es(CS, B[:N_ICA])[0]
        scr = np.asarray([e63.ica_pass(u, ICA_REPRO, False) for u in us])
        out["kinds"][kd] = dict(n_above_max=int((e_ > mx).sum()), n_screened_above_max=int((e_[scr] > mx).sum()),
                                n_screened=int(scr.sum()), es=e72.qs(e_))
        fit[kd] = (Y, B, e_, scr, us)
    for name in D2_UNITS:
        kd, j = name.split("/"); j = int(j)
        b = z[f"readout_{kd}"][:N_ICA, j:j + 1]
        e_ = float(e72.es(CS, b)[0][0])
        best = {k2: e64.best_match(z[f"pattern_{kd}"][:, j], mixS[k2]) for k2 in KINDS}
        k2 = max(best, key=lambda kk: best[kk][1])
        out["d2_units"][name] = dict(es=e_, es_pct=float((rS < e_).mean()), beats_max=bool(e_ > mx),
                                     best_match=dict(kind=k2, unit=best[k2][0], cos=best[k2][1]),
                                     context_generic=bool(best[k2][1] >= e72.PRESENCE_COS))
    # ---- end of the mirror. Primary count (NOTE section 4)
    out["s"] = int(sum(v.get("n_screened_above_max", 0) for v in out["kinds"].values()))
    out["n_above_max"] = int(sum(v.get("n_above_max", 0) for v in out["kinds"].values()))
    out["n_screened"] = int(sum(v.get("n_screened", 0) for v in out["kinds"].values()))
    # ---- secondary, descriptive: e72's full lead rule recomputed on this set (NOTE section 4)
    even = S["seq"] % 2 == 0
    Ch = {h: e72.covs(WS.astype(np.float64), WeS, sel) for h, sel in (("even", even), ("odd", ~even))}
    rh = {h: e72.es(Ch[h], G)[0] for h in Ch}
    mxh = {h: float(rh[h].max()) for h in rh}
    bat = e61.feature_battery(Sset, tk, fullS, None)
    pos_feats = dict(line_pos=bat["line_pos"], sent_pos=bat["sent_pos"], since_punct=e72.since_punct(fullS, S["row"], tk))
    rk_state = {kk: e72.ranks(v) for kk, v in stateS.items()}
    rk_pos = {kk: e72.ranks(v) for kk, v in pos_feats.items()}
    above, n_pass, arrays = [], {s_: 0 for s_ in ("identity", "norm_L0", "position", "split_half")}, {}
    for kd, (Y, B, e_, scr, us) in fit.items():
        eh = {h: e72.es(Ch[h], B[:N_ICA])[0] for h in Ch}
        arrays.update({f"es_{kd}": e_, f"screened_{kd}": scr, f"es_even_{kd}": eh["even"], f"es_odd_{kd}": eh["odd"]})
        for j in range(N_ICA):
            y = Y[:, j].astype(np.float64)
            ident, _ = e64.identity_diag(y, e72.heavy_sign(y), S, tk)
            ry = e72.ranks(y)
            rho_state = {kk: float((ry * v).mean()) for kk, v in rk_state.items()}
            rho_pos = {kk: float((ry * v).mean()) for kk, v in rk_pos.items()}
            scrn = dict(identity=bool(scr[j] and not ident["identity_suspect"]),
                        norm_L0=bool(all(abs(v) <= e72.STATE_RHO for v in rho_state.values())),
                        position=bool(all(abs(v) < e72.POS_RHO for v in rho_pos.values())),
                        split_half=bool(us[j]["repro"] >= ICA_REPRO and eh["even"][j] > mxh["even"] and eh["odd"][j] > mxh["odd"]))
            for s_, ok in scrn.items():
                n_pass[s_] += int(ok)
            if e_[j] > mx:
                above.append(dict(kind=kd, id=j, es=float(e_[j]), es_over_max=float(e_[j] / mx), screened=bool(scr[j]),
                                  repro=us[j]["repro"], eta2_token=us[j]["eta2_token"], eta2_token_sq=us[j]["eta2_token_sq"],
                                  ex_kurtosis=us[j]["ex_kurtosis"], es_even=float(eh["even"][j]), es_odd=float(eh["odd"][j]),
                                  tail_tdi=ident["tail_tdi"], tail_max_type_share=ident["tail_max_type_share"],
                                  tail_top_types=ident["tail_top_types"][:5], rho_state=rho_state, rho_pos=rho_pos,
                                  screens=scrn, lead_rule=bool(all(scrn.values()))))
    out["above_max_units"] = sorted(above, key=lambda u: -u["es"])
    out["secondary"] = dict(lead_rule_count=int(sum(u["lead_rule"] for u in above)), half_max=mxh, n_pass_screen=n_pass,
                            screened_above_max_share=float(out["s"] / max(out["n_screened"], 1)))
    arrays["rand_es"] = rS
    print(t(), f"{tag}: rows {out['n_rows']} FVU {fvuS:.3f} random max {mx:.3f}; above max "
          f"{ {kd: v.get('n_above_max') for kd, v in out['kinds'].items()} }; SCREENED above max s = {out['s']} "
          f"(of {out['n_screened']} screened); lead-rule count {out['secondary']['lead_rule_count']}; skew/49 pct "
          f"{out['d2_units']['skew/49']['es_pct']:.4f}", flush=True)
    del WS, WeS, X1, fit, Ch; gc.collect()
    return out, arrays


def check_s2(o, ref):
    """NOTE section 5: shuffle 0 reproduces e72's hashed S2 block."""
    rk = ref["kinds"]
    chk = dict(n_rows=o["n_rows"] == ref["n_rows"] == e72.N_ROWS_S2)
    for kd in KINDS:
        for f in ("n_above_max", "n_screened_above_max", "n_screened"):
            chk[f"{kd}_{f}"] = o["kinds"][kd][f] == rk[kd][f]
    chk["random_max"] = abs(o["random_es"]["max"] - ref["random_es"]["max"]) < 1e-9
    chk["random_p50"] = abs(o["random_es"]["p50"] - ref["random_es"]["p50"]) < 1e-9
    d, rd = o["d2_units"]["skew/49"], ref["d2_units"]["skew/49"]
    chk["skew49_es"] = abs(d["es"] - rd["es_on_s2"]) < 1e-6
    chk["skew49_pct"] = d["es_pct"] == rd["es_on_s2_pct"] == 0.5088
    diff = max(abs(o["kinds"][kd]["es"][q] - rk[kd]["es"][q]) for kd in KINDS for q in rk[kd]["es"])
    return dict(checks=chk, all_ok=bool(all(chk.values())), unit_es_quantiles_max_abs_diff=float(diff))


def reading(s):
    if any(x >= S_D2 for x in s) or float(np.median(s)) >= 2:
        return "SELECTION EFFECT"
    if all(x == 0 for x in s):
        return "SURVIVES THIS NULL"
    return "INDETERMINATE"


def phase_analyze(cfg):
    cap_p = cfg.out("e72b_capture.json")
    if not cfg.test:
        e64.require_hashed(cfg.rel(cap_p))
    cap = json.loads(cap_p.read_text())
    assert cap["synthetic"] == cfg.test
    for k in KS:
        c = cap["shuffles"][tag_of(k)]
        if not cfg.test:
            e64.require_hashed(c["meta"], c["rows"])
        assert e64.sha(cfg.meta(k)) == c["meta_sha256"] and e64.sha(cfg.rows_file(k)) == c["rows_sha256"]
        assert len(e64.rows_only(tag_of(k))[0]["row"]) == c["n_rows"]
    assert not cfg.out("e72b_null.json").exists(), "analyze runs once"
    cov = json.loads((EXT / "results/e72_coverage.json").read_text())
    assert cov["counts"]["screened"] == N_SCREENED_D2 and cov["leads"] == ["skew/49"]
    assert sum(u["beats"]["max"] and u["screened"] for u in cov["units"]) == S_D2
    z = np.load(EXT / "results/e72_units_raw_L12.npz")
    P = dict(mu=z["P_mu"], U=z["P_U"], sd=z["P_sd"])
    M = {kk: z[kk] for kk in ("Sp_is", "type_ids", "type_isqrt", "Uo", "sw")}
    params = e69.sae_params(e72.SAE_BIG)
    tk = e61.token_strings()
    sets, arrays = {}, {}
    for k in (0,) + KS:
        n = e72.N_ROWS_S2 if k == 0 else cap["shuffles"][tag_of(k)]["n_rows"]
        o, a = run_set(k, cfg, z, P, M, params, tk, n)
        name = "S2" if k == 0 else f"S2b{k}"
        if k == 0:
            o["reproduction"] = check_s2(o, cov["s2"])
            print(t(), "S2 reproduction", o["reproduction"], flush=True)
            assert o["reproduction"]["all_ok"], "S2 does not reproduce e72's S2 block: STOP"
        sets[name] = o
        arrays.update({f"{name}_{kk}": v for kk, v in a.items()})
    s = [sets[f"S2b{k}"]["s"] for k in KS]
    rd = reading(s)
    ver = json.loads((EXT / "results/e72_verify.json").read_text())
    v3 = {nm: {kk: ver["units"][nm][kk] for kk in ("es_ls", "ls_r2", "ls_pct", "ls_beats_max")}
          for nm in ("skew/49", "logcosh/37", "skew/52")}
    res = dict(note_sha256=e64.sha(EXT / NOTE), script_sha256=e64.sha(__file__), e72_script_sha256=E72_SHA,
               capture_sha256=e64.sha(cap_p), synthetic=cfg.test,
               d2_reference=dict(s=S_D2, units=["logcosh/37", "logcosh/49", "skew/49", "skew/52"], lead_count=LEADS_D2,
                                 n_screened=N_SCREENED_D2, random_max=cov["random"]["quantiles"]["all"]["es"]["max"]),
               decision=dict(s=s, median=float(np.median(s)), max=int(max(s)), reading=rd,
                             rule="SELECTION EFFECT if any s_k >= 4 or median >= 2; SURVIVES THIS NULL if all s_k = 0; "
                                  "else INDETERMINATE (5 new shuffles only; a count comparison, not a test)"),
               secondary=dict(lead_rule_counts={nm: sets[nm]["secondary"]["lead_rule_count"] for nm in sets},
                              skew49_pct={nm: sets[nm]["d2_units"]["skew/49"]["es_pct"] for nm in sets},
                              pooled_s_with_S2=int(sum(sets[nm]["s"] for nm in sets))),
               v3_single_filter_cited=dict(source="results/e72_verify.json", random_max=ver["v3_random"]["max"],
                                           random_r2_min=ver["v3_random"]["r2_min"], units=v3),
               sets=sets, seconds=round(time.time() - T0))
    np.savez_compressed(cfg.out("e72b_null_arrays.npz"), **arrays)
    e72.dump(res, cfg.out("e72b_null.json"))
    sfx = " (SMOKE: synthetic shuffles)" if cfg.test else ""
    e64.append_hash(cfg.rel(cfg.out("e72b_null_arrays.npz")), "e72b analyze: per-set random-direction ES and unit ES" + sfx)
    e64.append_hash(cfg.rel(cfg.out("e72b_null.json")), f"e72b analyze: shuffle null, s = {s}, reading {rd}" + sfx)
    print(t(), f"ANALYZE done; s = {s}; reading {rd}", flush=True)


def main():
    global T0
    T0 = time.time()
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True, choices=["capture", "analyze"])
    ap.add_argument("--test-dir", default=None)
    a = ap.parse_args()
    cfg = Cfg(a.test_dir)
    print("e72b sha256", e64.sha(__file__), "test" if cfg.test else "", flush=True)
    if not cfg.test:
        e64.require_hashed(NOTE, SELF)
    integrity(cfg)
    print(t(), f"swap {e64.swap_used_gb():.1f} GB", flush=True)
    dict(capture=phase_capture, analyze=phase_analyze)[a.phase](cfg)


if __name__ == "__main__":
    main()
