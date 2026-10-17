"""Phase 4 -- DIAGNOSTIC ONLY for the scripts/44b promotion stage. NOT VERDICT-BEARING.

scripts/44b promoted 0 of 1,130 planes at L6 and 0 at L7 -- including the CALENDAR planes,
which Addendum J1 designates the positive control. A verdict that fails its own positive
control is an instrument statement, not a result, so this script asks WHICH TIER failed.
It computes nothing new: it re-runs 44b's own path (imported from the file, so the two cannot
drift) and reports the per-tier numbers 44b summarised away.

Verdict status: NONE. results/phase4_gained_carriers_promotion.json stands as written.
Everything here is a post-hoc read of an already-frozen negative, declared as such.

Reported per layer:
  D1  per-tier pass counts over the 1,130 planes: Jaccard tier alone (class-permutation p <=
      0.01), closure tier alone (p_bg < 0.002), and the conjunction.
  D2  the 10 planes with the most calendar types in their half-A top-24: sizes, overlap,
      Jaccard and both its null p-values, gap_ratio / p_closed / p_bg of S_A ∩ S_B on half B,
      and the member strings.
  D3  arm-health control at half n: closure of the strict-12 months alone in the half-B SAE
      mean-code cloud (pooled reference: L6 1.6588 / 0.00125 / 0.00399, L7 1.6204 / 0.00115 /
      0.00200; matched-n certificate gives 0.99 pass rate at n = 60-90 per member and the
      half counts are ~200).
  D4  the same closure statistic for the CALENDAR SUBSET of S_A ∩ S_B (months + weekdays only)
      on the best calendar plane -- the direct test of whether size, not arm, is what fails.
"""
import importlib.util
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
spec = importlib.util.spec_from_file_location(
    "m44b", REPO / "scripts/44b_gained_carrier_promotion.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def main():
    t0 = time.time()
    out = REPO / "results/phase4_gained_carriers_promotion_diagnostic.json"
    rec = dict(script="scripts/44c_promotion_diagnostic.py",
               item="DIAGNOSTIC ONLY -- why scripts/44b promoted nothing. No verdict.",
               verdict_status="NONE (results/phase4_gained_carriers_promotion.json stands)",
               declared_at=datetime.now().isoformat(timespec="seconds"),
               declarations=__doc__, layers={})
    out.write_text(json.dumps(rec, indent=1))

    tok = np.fromfile(REPO / "cache/sparse_big/layer_0/tokens.i32", dtype=np.int32)
    cnt_all = np.bincount(tok)
    pool_ids = np.flatnonzero(cnt_all >= m.COUNT_FLOOR).astype(np.int64)
    n_types = len(pool_ids)
    import os
    os.environ.setdefault("HF_HOME", str(REPO / "cache/hf"))
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from transformers import AutoTokenizer
    tk = AutoTokenizer.from_pretrained("google/gemma-3-1b-pt")
    strings = [m.type_string(tk, int(t)) for t in pool_ids]
    feats = np.asarray([m.class_features(s) for s in strings], dtype=bool)
    class_id = np.array([int("".join("1" if v else "0" for v in r), 2) for r in feats])
    token_map = json.loads((REPO / "results/token_map.json").read_text())
    row_of = {int(t): i for i, t in enumerate(pool_ids)}
    month_rows = np.asarray([row_of[sorted((int(x) for x in token_map["months"][mm]
                                            if int(x) in row_of),
                                           key=lambda x: -int(cnt_all[x]))[0]]
                             for mm in m.MONTH_NAMES], dtype=np.int64)
    weekday_rows = np.asarray(sorted({row_of[int(x)] for v in token_map["days"].values()
                                      for x in v if int(x) in row_of}), dtype=np.int64)
    cal_mask = np.zeros(n_types, dtype=bool)
    cal_mask[month_rows] = True
    cal_mask[weekday_rows] = True
    w_in_full = m.read_w_in(REPO / m.SNAP, m.LAYERS)
    keep_c = np.setdiff1d(np.arange(1152), np.asarray(m.DROP_COORDS))
    from gfdisc.saes import load_sae, get_decoder

    for L in m.LAYERS:
        tL = time.time()
        sae = load_sae(L, "big", token=os.environ.get("HF_TOKEN"))
        W = get_decoder(sae); del sae
        halves, counts_h = {}, {}
        for name, parity in (("A", 0), ("B", 1)):
            halves[name], counts_h[name], _ = m.half_type_means(
                REPO / "cache/sparse_big", L, tok, pool_ids, parity, W)
        del W
        X, Y, V, norms = {}, {}, {}, {}
        for name in ("A", "B"):
            Xr = m.rms_norm_x(halves[name][:, keep_c], w_in_full[L][keep_c])
            D = np.column_stack([np.ones(n_types),
                                 np.log10(np.maximum(counts_h[name].astype(np.float64), 1.0)),
                                 feats.astype(np.float64),
                                 np.sqrt(np.maximum((Xr ** 2).sum(1), 1e-300))])
            Xd = m.deflate(Xr, D)
            X[name] = Xd
            norms[name] = np.maximum((Xd ** 2).sum(1), 1e-300)
            V[name] = m.build_basis(Xd)
            Y[name] = Xd @ V[name]
        MB = halves["B"][:, keep_c]
        specs = m.plane_specs(442000 + 100 * L)

        bg = {}

        def p_bg(k, gapdeg):
            if k not in bg:
                rng = np.random.default_rng(443000 + 100 * L + k)
                g = []
                for _ in range(m.N_BACKGROUND_DRAWS):
                    s = m.screen(MB[rng.choice(n_types, size=k, replace=False)],
                                 m.N_PERM_BACKGROUND)
                    if s is not None:
                        g.append(s["max_gap_deg"])
                bg[k] = np.asarray(g)
            g = bg[k]
            return float((1 + int((g <= gapdeg).sum())) / (1 + len(g)))

        rngp = np.random.default_rng(444000 + 100 * L)
        perms = []
        for _ in range(m.N_CLASSPERM):
            p = np.arange(n_types)
            for c in np.unique(class_id):
                idx = np.flatnonzero(class_id == c)
                p[idx] = rngp.permutation(idx)
            perms.append(p)
        perms = np.asarray(perms)

        rows = []
        for pi, spec in enumerate(specs):
            eA = (m.plane_coords(Y["A"], spec) ** 2).sum(1) / norms["A"]
            SA = np.flatnonzero(eA > np.percentile(eA, m.PCT_E))
            if len(SA) > m.CAP_S:
                SA = SA[np.argsort(-eA[SA])[:m.CAP_S]]
            eB = (m.plane_coords(Y["B"], spec) ** 2).sum(1) / norms["B"]
            SB = np.argpartition(-eB, len(SA))[:len(SA)]
            top24 = SA[np.argsort(-eA[SA])[:m.TOP_N]]
            J = m.jaccard(SA, SB)
            maskA = np.zeros(n_types, dtype=bool); maskA[SA] = True
            jn = np.asarray([maskA[perms[t][SB]].sum() for t in range(m.N_CLASSPERM)], float)
            jn = jn / (2 * len(SA) - jn)
            p_j = float((1 + int((jn >= J).sum())) / (1 + len(jn)))
            S = np.intersect1d(SA, SB)
            r = dict(idx=pi, plane=("coord_%d_%d" % spec[1] if spec[0] == "coord"
                                    else f"rand_{pi-630}"),
                     n_cal_top24=int(cal_mask[top24].sum()),
                     n_month_top24=int(np.isin(top24, month_rows).sum()),
                     size_SA=int(len(SA)), overlap=int(len(S)), jaccard=float(J),
                     jaccard_p_classperm=p_j, jaccard_null_classperm_max=float(jn.max()))
            if len(S) >= m.SCREEN_MIN:
                s = m.screen(MB[S], m.N_PERM_SCREEN)
                if s is not None:
                    r.update(gap_ratio=s["gap_ratio"], p_closed=s["p_closed"],
                             p_bg=p_bg(len(S), s["max_gap_deg"]), top2_var=s["top2_var"])
            rows.append(r)

        pass_j = [r for r in rows if r["jaccard_p_classperm"] <= m.JACCARD_P_MAX]
        pass_c = [r for r in rows if r.get("p_bg", 1.0) < m.P_BG_MAX]
        both = [r for r in rows if r["jaccard_p_classperm"] <= m.JACCARD_P_MAX
                and r.get("p_bg", 1.0) < m.P_BG_MAX]
        cal_sorted = sorted(rows, key=lambda r: -r["n_cal_top24"])[:10]
        for r in cal_sorted:
            spec = specs[r["idx"]]
            eA = (m.plane_coords(Y["A"], spec) ** 2).sum(1) / norms["A"]
            SA = np.flatnonzero(eA > np.percentile(eA, m.PCT_E))
            eB = (m.plane_coords(Y["B"], spec) ** 2).sum(1) / norms["B"]
            SB = np.argpartition(-eB, len(SA))[:len(SA)]
            S = np.intersect1d(SA, SB)
            r["members"] = [strings[i] for i in S[np.argsort(-eB[S])]]
            sc = S[cal_mask[S]]
            if len(sc) >= m.SCREEN_MIN:
                s = m.screen(MB[sc], m.N_PERM_SCREEN)
                if s is not None:
                    r["calendar_subset"] = dict(k=int(len(sc)), gap_ratio=s["gap_ratio"],
                                                p_closed=s["p_closed"],
                                                p_bg=p_bg(len(sc), s["max_gap_deg"]),
                                                members=[strings[i] for i in sc])
        sm = m.screen(MB[month_rows], m.N_PERM_SCREEN)
        months_ctrl = dict(k=12, gap_ratio=sm["gap_ratio"], p_closed=sm["p_closed"],
                           p_bg=p_bg(12, sm["max_gap_deg"]),
                           pooled_reference={"6": "1.6588 / 0.00125 / 0.00399",
                                             "7": "1.6204 / 0.00115 / 0.00200"}[str(L)])
        rec["layers"][str(L)] = dict(
            n_planes=len(rows), n_pass_jaccard_tier=len(pass_j), n_pass_closure_tier=len(pass_c),
            n_pass_both=len(both),
            jaccard_p_min=float(min(r["jaccard_p_classperm"] for r in rows)),
            p_bg_min=float(min(r.get("p_bg", 1.0) for r in rows)),
            gap_ratio_min=float(min(r.get("gap_ratio", 99.0) for r in rows)),
            overlap_median=float(np.median([r["overlap"] for r in rows])),
            months_only_control_halfB=months_ctrl,
            top_calendar_planes=cal_sorted, seconds=round(time.time() - tL, 1))
        print(f"[L{L}] jaccard tier {len(pass_j)}/{len(rows)}, closure tier {len(pass_c)}, "
              f"both {len(both)}; months-only half-B {sm['gap_ratio']:.3f}/"
              f"{sm['p_closed']:.5f}/{months_ctrl['p_bg']:.5f}; {time.time()-tL:.0f}s", flush=True)
        del halves, X, Y, V, MB

    rec["runtime_seconds"] = round(time.time() - t0, 1)
    out.write_text(json.dumps(rec, indent=1))
    print(f"wrote {out} ({rec['runtime_seconds']}s)", flush=True)


if __name__ == "__main__":
    main()
