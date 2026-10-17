"""Phase 4 -- DIAGNOSTIC ONLY, extraction of the near-candidates of scripts/44b. NO VERDICT.

scripts/44c measured that exactly ONE plane per layer cleared both frozen tiers (class-permuted
Jaccard p <= 0.01 AND p_bg < 0.002) and that scripts/44b promoted none of them, i.e. each died
at the leave-one-out audit. Neither script persisted WHICH plane that was. R11 turns on whether
a near-passer is calendar or non-calendar, so this script re-runs 44b's own path (imported from
the file, seeds and rules unchanged) and prints the identity, membership, calendar label, tier
numbers and the LOO profile of every both-tier plane. It computes no new statistic and changes
no verdict: results/phase4_gained_carriers_promotion.json stands as written.
"""
import importlib.util
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
spec = importlib.util.spec_from_file_location(
    "m44b", REPO / "scripts/44b_gained_carrier_promotion.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)


def main():
    t0 = time.time()
    out = REPO / "results/phase4_gained_carriers_near_candidates.json"
    rec = dict(script="scripts/44d_near_candidate_extract.py",
               item="DIAGNOSTIC ONLY -- identity of the both-tier planes of scripts/44b",
               verdict_status="NONE (results/phase4_gained_carriers_promotion.json stands)",
               declared_at=datetime.now().isoformat(timespec="seconds"),
               declarations=__doc__, layers={})
    out.write_text(json.dumps(rec, indent=1))

    tok = np.fromfile(REPO / "cache/sparse_big/layer_0/tokens.i32", dtype=np.int32)
    cnt_all = np.bincount(tok)
    pool_ids = np.flatnonzero(cnt_all >= m.COUNT_FLOOR).astype(np.int64)
    n_types = len(pool_ids)
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
    cal_mask[month_rows] = True; cal_mask[weekday_rows] = True
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
        Y, norms = {}, {}
        for name in ("A", "B"):
            Xr = m.rms_norm_x(halves[name][:, keep_c], w_in_full[L][keep_c])
            D = np.column_stack([np.ones(n_types),
                                 np.log10(np.maximum(counts_h[name].astype(np.float64), 1.0)),
                                 feats.astype(np.float64),
                                 np.sqrt(np.maximum((Xr ** 2).sum(1), 1e-300))])
            Xd = m.deflate(Xr, D)
            norms[name] = np.maximum((Xd ** 2).sum(1), 1e-300)
            Y[name] = Xd @ m.build_basis(Xd)
        MB = halves["B"][:, keep_c]
        specs = m.plane_specs(442000 + 100 * L)
        bg = {}

        def p_bg(k, gapdeg):
            if k not in bg:
                rng = np.random.default_rng(443000 + 100 * L + k)
                g = [s["max_gap_deg"] for s in
                     (m.screen(MB[rng.choice(n_types, size=k, replace=False)],
                               m.N_PERM_BACKGROUND) for _ in range(m.N_BACKGROUND_DRAWS))
                     if s is not None]
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

        hits = []
        for pi, spec in enumerate(specs):
            eA = (m.plane_coords(Y["A"], spec) ** 2).sum(1) / norms["A"]
            SA = np.flatnonzero(eA > np.percentile(eA, m.PCT_E))
            if len(SA) > m.CAP_S:
                SA = SA[np.argsort(-eA[SA])[:m.CAP_S]]
            eB = (m.plane_coords(Y["B"], spec) ** 2).sum(1) / norms["B"]
            SB = np.argpartition(-eB, len(SA))[:len(SA)]
            J = m.jaccard(SA, SB)
            maskA = np.zeros(n_types, dtype=bool); maskA[SA] = True
            jn = np.asarray([maskA[perms[t][SB]].sum() for t in range(m.N_CLASSPERM)], float)
            jn = jn / (2 * len(SA) - jn)
            p_j = float((1 + int((jn >= J).sum())) / (1 + len(jn)))
            if p_j > m.JACCARD_P_MAX:
                continue
            S = np.intersect1d(SA, SB)
            if len(S) < m.SCREEN_MIN:
                continue
            s = m.screen(MB[S], m.N_PERM_SCREEN)
            if s is None:
                continue
            pb = p_bg(len(S), s["max_gap_deg"])
            if pb >= m.P_BG_MAX:
                continue
            loo = []
            for x in S:
                sub = S[S != x]
                s2 = m.screen(MB[sub], m.N_PERM_BACKGROUND)
                loo.append(dict(dropped=strings[x],
                                pass_=bool(s2 is not None
                                           and p_bg(len(sub), s2["max_gap_deg"]) < m.P_BG_MAX),
                                p_bg=None if s2 is None else p_bg(len(sub), s2["max_gap_deg"])))
            top24 = SA[np.argsort(-eA[SA])[:m.TOP_N]]
            hits.append(dict(
                plane=("coord_%d_%d" % spec[1] if spec[0] == "coord" else f"rand_{pi-630}"),
                kind=spec[0], calendar=bool(cal_mask[top24].sum() >= m.CAL_MIN_IN_TOP24),
                n_calendar_in_top24=int(cal_mask[top24].sum()),
                n_months_in_top24=int(np.isin(top24, month_rows).sum()),
                size_SA=int(len(SA)), overlap=int(len(S)), jaccard=float(J),
                jaccard_p_classperm=p_j, gap_ratio=s["gap_ratio"], p_closed=s["p_closed"],
                p_bg=pb, top2_var=s["top2_var"], member_hash=m.member_hash(S),
                members=[strings[i] for i in S[np.argsort(-eB[S])]],
                loo_pass_fraction=float(np.mean([x["pass_"] for x in loo])),
                loo_failures=[x["dropped"] for x in loo if not x["pass_"]],
                loo_detail=loo))
        rec["layers"][str(L)] = dict(n_both_tier=len(hits), planes=hits,
                                     seconds=round(time.time() - tL, 1))
        for h in hits:
            print(f"[L{L}] {h['plane']} ({h['kind']}) calendar={h['calendar']} "
                  f"months_top24={h['n_months_in_top24']} |S|={h['overlap']} J={h['jaccard']:.3f} "
                  f"p_j={h['jaccard_p_classperm']:.4f} gap={h['gap_ratio']:.3f} "
                  f"p_bg={h['p_bg']:.5f} LOO={h['loo_pass_fraction']:.3f}", flush=True)
            print(f"      members: {h['members']}", flush=True)
        print(f"[L{L}] {len(hits)} both-tier planes in {time.time()-tL:.0f}s", flush=True)
        del halves, Y, MB

    rec["runtime_seconds"] = round(time.time() - t0, 1)
    out.write_text(json.dumps(rec, indent=1))
    print(f"wrote {out} ({rec['runtime_seconds']}s)", flush=True)


if __name__ == "__main__":
    main()
