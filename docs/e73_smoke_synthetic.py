"""Smoke test for scripts/e73_coverage_admission.py on synthetic activations (NOTE_coverage_admission_L12.md section 13).

Builds DIR/H_D.npy (203,328 x 1152 fp16 synthetic rows aligned to D2's kept rows; real D2 labels plant paren and quote along
fixed directions; coordinate 1038 carries a +1000 outlier), DIR/units.npz + DIR/gate.json in e72's formats (maps fitted on the
synthetic rows with e72's gate operations; targets = label least-squares filters at the real target slots), so the script
runs its real code path, and two synthetic JumpReLU SAEs (L0 about 110, as l0_big):
  sae_l0_big    generic latents in the rows' low-rank subspace; paren = 10 knocked-out latents carrying 10% of the paren
                reconstruction plus 8 near-duplicate siblings carrying 90% (the geometry trap); quote = 10 latents carrying all
  sae_l0_small  new generic latents; paren and quote each carried fully by 10 latents (rule E must find them)
Runs `calibrate_l0_big`, `calibrate_l0_small` and `verdict` through the real CLI with --test-dir, then asserts:
  A both phases exit 0 and every output is hashed into DIR/HASHES.txt;
  B geometry trap (l0_big, paren): at s10 T-COS reads ADDS while T-ES does not (siblings still reconstruct paren); T-ES
    reads ADDS once the near-duplicates go (sND);
  C T-COS is NOT ADMITTED for both dictionaries;
  D T-ES detects the W-span plant at some kappa <= 4 (l0_big);
  E verdict logic on fabricated records: an all-pass record is ADMITTED-GRADED when rule 8 is not reached and ADMITTED when
    it is reached and agrees (a probe excursion after m* does not matter); a second matched ADDS (10-latent or size-matched)
    fails rule 4; three S40 moves into ADDS fail rule 5 while many COVERED <-> UNRESOLVED moves do not; a disagreeing rule 8
    fails; a class decrease fails rule 2; a passing T-COS triggers the rule-set guard; a missing dictionary is NO VERDICT;
  F positive end to end (both shapes): T-ES on synthetic l0_big is ADMITTED or ADMITTED-GRADED;
  G real shape (--dsae 16384): every phase's peak RSS (one process per dictionary) is at most 7 GB (GO condition, NOTE 13).
Usage: .venv/bin/python docs/e73_smoke_synthetic.py DIR [--dsae 2048]   (--dsae 16384 = real-shape timing and memory run)
"""
import gc, importlib.util, json, resource, subprocess, sys, time
from pathlib import Path

import numpy as np

EXT = Path(__file__).resolve().parents[1]
SCRIPT = EXT / "scripts/e73_coverage_admission.py"


def load_e73():
    spec = importlib.util.spec_from_file_location("e73", SCRIPT)
    m = importlib.util.module_from_spec(spec); sys.modules["e73"] = m; spec.loader.exec_module(m)
    return m


def unit(v):
    return v / np.linalg.norm(v)


def build(td, dsae, e73):
    rng = np.random.default_rng(20261017)
    D, full = e73.e72.rows("occD2", e73.N_ROWS)
    Dset = dict(seq=D["seq"], tinv=D["tinv"], tok=D["tok"], row=D["row"])
    bat = e73.e61.feature_battery(Dset, e73.e61.token_strings(), full, None)
    yp, yq = (np.asarray(bat["paren_inside"]) > 0.5), (np.asarray(bat["quote_inside"]) > 0.5)
    n, keep = len(D["row"]), e73.KEEP
    r = 48
    Q = np.linalg.qr(rng.standard_normal((1151, r)))[0]                       # low-rank subspace on kept coordinates
    scales = np.linspace(3.0, 0.8, r)
    vp, vq = unit(rng.standard_normal(1151)), unit(rng.standard_normal(1151))
    alpha = 4.0
    H = np.empty((n, 1152), np.float16)
    for a in range(0, n, 32768):
        b = slice(a, min(n, a + 32768)); m = b.stop - b.start
        x = (rng.standard_normal((m, r)) * scales) @ Q.T + 0.3 * rng.standard_normal((m, 1151))
        x += alpha * yp[b, None] * vp[None, :] + alpha * yq[b, None] * vq[None, :]
        row = np.zeros((m, 1152), np.float32); row[:, keep] = x; row[:, 1038] = 1000.0 + rng.standard_normal(m)
        H[b] = row.astype(np.float16)
    np.save(td / "H_D.npy", H)
    sample = H[rng.choice(n, 8000, replace=False)].astype(np.float32)

    def sae(n_gen, groups, seed):
        g = np.random.default_rng(seed)
        dg = np.stack([unit(Q @ g.standard_normal(r)) for _ in range(n_gen)])          # generic directions (kept coords)
        w_enc = np.zeros((1152, dsae), np.float32); w_dec = np.zeros((dsae, 1152), np.float32)
        b_enc = np.zeros(dsae, np.float32); thr = np.zeros(dsae, np.float32)
        idx = g.permutation(dsae)
        gen, special = idx[:n_gen], idx[n_gen:]
        w_enc[np.ix_(keep, gen)] = dg.T
        pre = sample[:, keep] @ dg.T
        rate = min(0.07, 110.0 / n_gen)                                             # L0 about 110 (l0_big)
        thr[gen] = np.quantile(pre, 1 - rate, axis=0)
        z = pre * (pre > thr[gen]); rec = z @ dg
        s = float((rec * (sample[:, keep] - sample[:, keep].mean(0))).sum() / (rec * rec).sum())   # LS scale of the generic part
        w_dec[np.ix_(gen, keep)] = s * dg
        out, pos = {}, 0
        for name, v, n_lat, share in groups:                                              # label-carrying latents
            ids = special[pos:pos + n_lat]; pos += n_lat
            for i in ids:
                d = unit(v + 0.05 * unit(g.standard_normal(1151)))
                w_enc[np.ix_(keep, [i])] = v[:, None]; b_enc[i] = -alpha / 2; thr[i] = 0.0
                w_dec[np.ix_([i], keep)] = (share * alpha / (n_lat * alpha / 2)) * d
            out[name] = [int(i) for i in ids]
        b_dec = np.zeros(1152, np.float32); b_dec[1038] = 1000.0
        rest = special[pos:]
        w_dec[np.ix_(rest, keep)] = 1e-3 * g.standard_normal((len(rest), 1151)); thr[rest] = 1e9    # dead latents
        return dict(w_enc=w_enc, b_enc=b_enc, threshold=thr, w_dec=w_dec, b_dec=b_dec), out

    n_gen = int(0.7 * dsae)
    pb, ib = sae(n_gen, [("paren_ko", vp, 10, 0.1), ("paren_sib", vp, 8, 0.9), ("quote_ko", vq, 10, 1.0)], 1)
    np.savez(td / "sae_l0_big.npz", **pb, ko_paren=np.asarray(ib["paren_ko"]), ko_quote=np.asarray(ib["quote_ko"]))
    ps, is_ = sae(n_gen, [("paren", vp, 10, 1.0), ("quote", vq, 10, 1.0)], 2)
    np.savez(td / "sae_l0_small.npz", **ps)
    json.dump(dict(big=ib, small=is_), open(td / "synthetic_latents.json", "w"))
    del sample, pb, ps; gc.collect()
    # maps and targets with e72's gate operations, written in e72's formats
    V = e73.e72.raw_V(H, D["groups"]); del H; gc.collect()
    Z, P = e73.e63.pca_white(V); del V; gc.collect()
    W, M = e73.e64.fit_maps(Z, D["tok"], D["tinv"])
    Wd64, Zd64 = W.astype(np.float64), Z.astype(np.float64)
    CzW = (Zd64 - Zd64.mean(0)).T @ (Wd64 - Wd64.mean(0)) / n
    rawmap = np.asarray(P["U"], np.float64) @ (np.asarray(P["sd"], np.float64)[:, None] * CzW)
    even = D["seq"] % 2 == 0

    def ls(y, sel):
        X = Wd64[sel] - Wd64[sel].mean(0)
        b, *_ = np.linalg.lstsq(X, y[sel] - y[sel].mean(), rcond=None)
        return b / np.linalg.norm(b)

    ro = {k: np.zeros((65, 64)) for k in ("logcosh", "skew")}; dr = {k: np.zeros((64, 1152)) for k in ("logcosh", "skew")}
    for (kd, j), b in (((("skew", 45)), ls(yp.astype(float), even)), (("logcosh", 29), ls(yq.astype(float), even)),
                       (("skew", 29), ls(yq.astype(float), ~even))):
        ro[kd][:64, j] = b; dr[kd][j, keep] = rawmap @ b
    np.savez(td / "units.npz", row=D["row"], W=W, Z=Z, P_mu=P["mu"], P_U=np.asarray(P["U"], np.float64),
             P_sd=np.asarray(P["sd"], np.float64), vh_ss=np.float64(1.0), readout_logcosh=ro["logcosh"], readout_skew=ro["skew"],
             dir_logcosh=dr["logcosh"], dir_skew=dr["skew"], **M)
    json.dump(dict(control_units=[["skew", 45, "paren_inside", 0.0], ["logcosh", 29, "quote_inside", 0.0],
                                  ["skew", 29, "quote_inside", 0.0]]), open(td / "gate.json", "w"))
    del W, Z, Wd64, Zd64; gc.collect()
    return ib, is_


def run_phase(td, phase):
    t0 = time.time()
    r = subprocess.run([sys.executable, str(SCRIPT), "--phase", phase, "--test-dir", str(td)], capture_output=True, text=True)
    (td / f"{phase}.log").write_text(r.stdout + r.stderr)
    rss = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1024 ** 3                      # bytes on macOS
    print(f"{phase}: exit {r.returncode}, {time.time() - t0:.0f} s, children peak RSS so far {rss:.2f} GB", flush=True)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    return time.time() - t0


def fabricate(e73, ladder_cls, match_adds=0, r8_reach=None, r6_clean=True, plant_kappa_adds=1.0, s40_flip=0, s40_to="A",
              msize_adds=0, r8_excursion=None):
    """A minimal record judge() accepts. ladder_cls(step) -> class for every target of both controls at that step."""
    tn = {"paren": ["skew/45"], "quote": ["logcosh/29", "skew/29"]}
    rec = dict(control_targets=tn, conditions={}, ladder={c: {} for c in e73.CTRL}, match={c: [] for c in e73.CTRL},
               match_size={c: {} for c in e73.CTRL},
               plants={"wspan": {}, "full_rank": {}}, probes={c: {} for c in e73.CTRL})

    def cond(clsf, valf, s40="C" * 20 + "U" * 20, plant=None):
        out = {}
        for x in e73.TESTS:
            tg = {t: dict(value=valf(t), cls=clsf(t)) for c in e73.CTRL for t in tn[c]}
            if plant is not None:
                tg["plant"] = dict(value=0.0, cls=plant)
            out[x] = dict(targets=tg, s40=s40)
        return out

    rec["conditions"]["s0"] = cond(lambda t: ladder_cls("s0"), lambda t: 0.0)
    for c in e73.CTRL:
        rec["ladder"][c]["s0"] = "s0"
        o = [x for x in e73.CTRL if x != c][0]
        for i, s in enumerate(e73.STEPS[1:], 1):
            cn = f"{c}:{s}"
            s40 = ("C" * 20 + "U" * 20) if i != 3 else (s40_to * s40_flip + "C" * (20 - s40_flip) + "U" * 20)
            rec["conditions"][cn] = cond(lambda t, s=s, c=c: ladder_cls(s) if t in tn[c] else "C", lambda t, i=i: float(i), s40)
            rec["ladder"][c][s] = cn
        for k in range(20):
            cn = f"match:{c}:{k}"
            rec["conditions"][cn] = cond(lambda t, k=k, c=c: "A" if (k < match_adds and t in tn[c]) else "C", lambda t: 0.0)
            rec["match"][c].append(cn)
        for s_ in e73.MSIZE_STEPS:
            names = []
            for k in range(20):
                cn = f"msize:{c}:{s_}:{k}"
                rec["conditions"][cn] = cond(lambda t, k=k, c=c: "A" if (k < msize_adds and t in tn[c]) else "C", lambda t: 0.0)
                names.append(cn)
            rec["match_size"][c][s_] = names
        for s in e73.STEPS:
            reach = (r8_reach is not None and e73.STEPS.index(s) >= e73.STEPS.index(r8_reach)) and s != r8_excursion
            rec["probes"][c][s] = dict(r8=dict(auc=0.50 if reach else 0.70, band_p99=0.51, band_med=0.50),
                                       r6=dict(auc=0.50 if r6_clean else 0.70, band_p99=0.51, band_med=0.50))
    for kind in ("wspan", "full_rank"):
        for kap in (1.0, 2.0, 4.0):
            cn = f"plant:{kind}:{kap:g}"
            rec["conditions"][cn] = cond(lambda t: "C", lambda t: 0.0, plant="A" if kap >= plant_kappa_adds else "C")
            rec["plants"][kind][f"{kap:g}"] = cn
    return rec


def logic_tests(e73):
    first_adds = lambda s: "A" if e73.STEPS.index(s) >= e73.STEPS.index("sND") else ("C" if s in ("s0", "s1", "s3", "s10") else "U")
    j = e73.judge(fabricate(e73, first_adds), "T-ES")
    assert j["level"] == "ADMITTED-GRADED" and j["mde_kappa"] == 1.0, j
    # rule 8 reached at sND, and the test is COVERED wherever the probe sits above the band -> ADMITTED
    j = e73.judge(fabricate(e73, first_adds, r8_reach="sND"), "T-ES")
    assert j["level"] == "ADMITTED", j
    # rule 8 reached at s30, but the test already reads ADDS at sND where the probe is above the band -> FAIL
    j = e73.judge(fabricate(e73, first_adds, r8_reach="s30"), "T-ES")
    assert j["level"] == "NOT ADMITTED" and "rule8" in j["first_failure"], j
    j = e73.judge(fabricate(e73, first_adds, r8_reach="sND", r6_clean=False), "T-ES")
    assert j["level"] == "ADMITTED-GRADED", j
    j = e73.judge(fabricate(e73, first_adds, r8_reach="sND", r8_excursion="s100"), "T-ES")     # excursion after m*: still PASS
    assert j["level"] == "ADMITTED", j
    j = e73.judge(fabricate(e73, first_adds, match_adds=2), "T-ES")
    assert j["level"] == "NOT ADMITTED" and j["first_failure"] == "paren, rule4", j
    j = e73.judge(fabricate(e73, first_adds, msize_adds=2), "T-ES")
    assert j["level"] == "NOT ADMITTED" and j["first_failure"] == "paren, rule4" and j["controls"]["paren"]["rule4_n_adds_size_matched"] == 2, j
    j = e73.judge(fabricate(e73, first_adds, s40_flip=3), "T-ES")
    assert j["level"] == "NOT ADMITTED" and j["first_failure"] == "paren, rule5", j
    j = e73.judge(fabricate(e73, first_adds, s40_flip=15, s40_to="U"), "T-ES")             # C -> U moves are not counted
    assert j["level"] == "ADMITTED-GRADED" and j["rule5_max_any_class_moves_descriptive"] == 15, j
    j = e73.judge(fabricate(e73, first_adds, plant_kappa_adds=8.0), "T-ES")
    assert j["level"] == "NOT ADMITTED" and "rule7" in j["first_failure"], j
    j = e73.judge(fabricate(e73, lambda s: "U" if s == "s0" else "A"), "T-ES")
    assert j["first_failure"] == "paren, rule1", j
    j = e73.judge(fabricate(e73, lambda s: "C"), "T-ES")
    assert j["first_failure"] == "paren, rule2", j
    j = e73.judge(fabricate(e73, lambda s: "U" if s == "s256" else first_adds(s)), "T-ES")      # class falls A -> U at s256
    assert j["first_failure"] == "paren, rule2" and not j["controls"]["paren"]["rule2_monotone"], j
    # guard: every test passing (T-COS included) -> rule set broken -> nothing admitted
    res, broken = e73.verdict_from({"l0_big": fabricate(e73, first_adds), "l0_small": None})
    assert broken and all(res["l0_big"][x]["level"] == "NOT ADMITTED" for x in e73.TESTS), res["l0_big"]
    assert all(res["l0_small"][x]["level"] == "NO VERDICT" for x in e73.TESTS)
    print("E ok: verdict logic on fabricated records (graded, admitted, rule 8 FAIL and post-m* excursion, rule 4a/4b/5/7/1/2 failures, C<->U moves ignored, guard, NO VERDICT)")


def main():
    td = Path(sys.argv[1]).resolve(); td.mkdir(parents=True, exist_ok=True)
    dsae = int(sys.argv[sys.argv.index("--dsae") + 1]) if "--dsae" in sys.argv else 2048
    e73 = load_e73()
    logic_tests(e73)
    for f in ("e73_calib_l0_big.json", "e73_calib_l0_big.npz", "e73_calib_l0_small.json", "e73_calib_l0_small.npz",
              "e73_coverage_admission.json", "HASHES.txt"):
        (td / f).unlink(missing_ok=True)
    (td / "HASHES.txt").write_text("")
    t0 = time.time(); build(td, dsae, e73); print(f"built synthetic rows and SAEs (d_sae {dsae}) in {time.time() - t0:.0f} s", flush=True)
    wall = {ph: run_phase(td, ph) for ph in ("calibrate_l0_big", "calibrate_l0_small", "verdict")}
    hashes = (td / "HASHES.txt").read_text()
    for f in ("e73_calib_l0_big.json", "e73_calib_l0_big.npz", "e73_calib_l0_small.json", "e73_calib_l0_small.npz", "e73_coverage_admission.json"):
        assert e73.sha(td / f) in hashes, f"{f} not hashed"
    print("A ok: both phases exit 0; outputs hashed")
    big = json.loads((td / "e73_calib_l0_big.json").read_text())
    C = big["conditions"]
    cl = lambda cn, x: C[cn][x]["targets"]["skew/45"]["cls"]
    print("paren ladder (T-ES, T-MG, T-COS):", {s: tuple(cl(big["ladder"]["paren"][s], x) for x in e73.TESTS) for s in e73.STEPS})
    print("quote ladder (T-ES, T-MG, T-COS):", {s: tuple(C[big["ladder"]["quote"][s]][x]["targets"]["logcosh/29"]["cls"] for x in e73.TESTS)
                                               for s in e73.STEPS})
    print("near-duplicates:", {c: (big["ladder_info"][c]["n_nd_decoder"], big["ladder_info"][c]["n_nd_activation"]) for c in e73.CTRL})
    assert cl("paren:s10", "T-COS") == "A" and cl("paren:s10", "T-ES") != "A", "geometry trap not reproduced at s10"
    assert cl("paren:sND", "T-ES") == "A", "T-ES does not uncover paren once the siblings go"
    print("B ok: geometry trap (T-COS ADDS at s10 with siblings intact, T-ES not; T-ES ADDS at sND)")
    v = json.loads((td / "e73_coverage_admission.json").read_text())
    print("levels:", json.dumps(v["levels"]), "broken:", v["rule_set_broken"])
    for l0 in ("l0_big", "l0_small"):
        for x in e73.TESTS:
            print(f"  {l0} {x}: {v['results'][l0][x]['level']} ({v['results'][l0][x].get('first_failure')}); MDE {v['results'][l0][x].get('mde_kappa')}")
    assert all(v["results"][l0]["T-COS"]["level"] == "NOT ADMITTED" for l0 in ("l0_big", "l0_small"))
    print("C ok: T-COS NOT ADMITTED for both dictionaries")
    assert v["results"]["l0_big"]["T-ES"]["mde_kappa"] is not None and v["results"]["l0_big"]["T-ES"]["mde_kappa"] <= 4
    print("D ok: T-ES detects the W-span plant at kappa", v["results"]["l0_big"]["T-ES"]["mde_kappa"])
    if True:
        assert v["results"]["l0_big"]["T-ES"]["level"] in ("ADMITTED", "ADMITTED-GRADED"), v["results"]["l0_big"]["T-ES"]["first_failure"]
        print("F ok: T-ES on synthetic l0_big reaches", v["results"]["l0_big"]["T-ES"]["level"])
    peak = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1024 ** 3
    if dsae == 16384:
        assert peak <= 7.0, f"real-shape per-phase peak RSS {peak:.2f} GB > 7 GB"
        print(f"G ok: real-shape per-phase peak RSS {peak:.2f} GB <= 7 GB")
    sm = json.loads((td / "e73_calib_l0_small.json").read_text())
    print("l0_small rule-E top-10 found the planted latents:",
          {c: len(set(sm["ladder_info"][c]["K10"]) & set(json.load(open(td / "synthetic_latents.json"))["small"][c])) for c in e73.CTRL})
    print(f"E73 SMOKE PASS (d_sae {dsae}; phases " + ", ".join(f"{k} {v:.0f} s" for k, v in wall.items()) + "; "
          f"children peak RSS {resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1024 ** 3:.2f} GB)")


if __name__ == "__main__":
    main()
