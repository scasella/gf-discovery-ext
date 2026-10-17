"""e42d -- the Phase-4 two-channel adjudication (scripts/42) on the DEEP-layer closure flags.

Every statistic, null and stream is IMPORTED from scripts/42_two_channel_readjudicate.py
(stream_deltas, outlier_drop_rule, cell_stats, NullCache, adjudicate, all_partitions,
pair_index, plane_angles, resolve_blocks, member_hash) and from scripts/22 (build_pool).
What lives here is the deep-layer driver, because 42's main() hard-codes L0-L10:
  * R10_DROP covers L0-L10 only. At deep layers the drop set is COMPUTED by the frozen
    Addendum-H rule itself (drop coordinates with > 25% pool-wide split-half noise share in
    the lag+1 successor deltas), declared here before any stream, and applied to channel 1,
    channel 2 and lag+2 alike. Whatever set the rule returns (empty, one, several) is used.
  * The positive control is strict-12 months (Addendum K1) evaluated at EVERY deep layer,
    label-known. The 13/14-member flag-set anchors of 42 exist only at L6/L7.
  * Amendments in force (docs/phase4_addendum_K/L/M):
      K2  conditional null 20,000 draws per (layer, k) cell;
      L1  saturated cell (p99 == 1.0): order conjunct passes iff rho == 1.0 AND the cell's
          exact-1.0 rate < 0.01, else NOT ADJUDICABLE (reported under closure & LOO);
      L2  every cell's p99 under 3 seeds; rho between min and max seed p99 -> MARGINAL;
      L4/M1 closure is evaluated post-drop and FIRST: closure failure -> FAIL;
      R8  k < 9 -> NOT ADJUDICABLE;
      R11 more than one non-calendar PASS -> first reading is tier leakage.
  * Calibration per layer: ordinal one..ten (must not pass), 200 frequency-pseudo sets around
    strict-12 (0 passes allowed), 200 class-block permutations of channel 2 with strict-12
    (<= 2 allowed), and a clean planted-ring power curve at the strict-12 radii.

DECLARED READING (written before the run): the tier is VALIDATED at a deep layer iff strict-12
PASSES or is MARGINAL there. At a layer where strict-12 FAILS, blind verdicts are reported as
UNCALIBRATED: a FAIL there carries no information and a PASS still needs every null.
"""
import argparse, gc, hashlib, importlib.util, json, os, sys, time
from pathlib import Path

import numpy as np

EXT = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HUB_OFFLINE", "1")
sys.path.insert(0, str(EXT / "src"))


def _imp(name, fn):
    spec = importlib.util.spec_from_file_location(name, EXT / "scripts" / fn)
    m = importlib.util.module_from_spec(spec); sys.modules[name] = m
    spec.loader.exec_module(m); return m


S42 = _imp("s42", "42_two_channel_readjudicate.py")
from gfdisc.saes import load_sae, get_decoder                  # noqa: E402
from gfdisc.tokens import MONTHS                                # noqa: E402

N_CL = 20_000                 # K2
SEEDS_L2 = (0, 1, 2)          # L2: seed offsets for threshold stability
N_PLANT = 100
PLANT_RHOS = (0.05, 0.10, 0.15, 0.20, 0.30)


class SeededNull(S42.NullCache):
    """42's NullCache with a seed offset (Addendum L2) and the exact-1.0 rate (L1)."""

    def __init__(self, *a, seed_offset=0, **k):
        super().__init__(*a, **k)
        self.seed_offset = int(seed_offset)

    def build(self, layer, k, n_draws=None, kind="clusterlike"):
        key = (layer, int(k), kind)
        if key in self.cache:
            return self.cache[key]
        seed = ({"clusterlike": 7_000_000, "random": 7_100_000, "freq": 7_200_000}[kind]
                + 1000 * layer + int(k) + 10_000_000 * self.seed_offset)
        rng = np.random.default_rng(seed)
        gaps, rc = [], []
        n = self.n_clusterlike if kind == "clusterlike" else S42.N_RANDOM
        for _ in range(n):
            if kind == "clusterlike":
                idx = self._draw_clusterlike(k, rng)
            else:
                idx = rng.choice(len(self.C1), size=k, replace=False)
            g, r = S42.cell_stats(self.C1[idx], None if self.C2 is None else self.C2[idx])
            if g is None:
                continue
            gaps.append(g)
            if r is not None and g < S42.GAP_RATIO_CLOSURE:
                rc.append(r)
        rc = np.asarray(rc)
        thr = float(np.percentile(rc, 99, method="higher")) if len(rc) else None
        out = dict(layer=int(layer), k=int(k), kind=kind, n_draws=int(n), seed=int(seed),
                   n_conditional=int(len(rc)), threshold_p99_conditional=thr,
                   exact1_rate=float((rc >= 1.0).mean()) if len(rc) else None,
                   closure_rate=float((np.asarray(gaps) < S42.GAP_RATIO_CLOSURE).mean())
                   if gaps else None,
                   saturated=bool(thr is not None and thr >= 1.0))
        self.cache[key] = out
        return out


def amended_verdict(members, C1, C2, nulls3, layer):
    """42's adjudicate() under seed 0, then K/L/M amendments applied on top."""
    base = S42.adjudicate(members, C1, C2, nulls3[0], layer)
    k = base["k"]
    if base["gap_ratio"] is None or not base["closure_pass"]:
        v = "FAIL (closure)"
    elif k < S42.K_MIN_ADJUDICABLE or base["two_channel_rho"] is None:
        v = "NOT ADJUDICABLE (k<9)" if k < S42.K_MIN_ADJUDICABLE else "NOT ADJUDICABLE (no ch2)"
    else:
        cells = [n.build(layer, k) for n in nulls3]
        cells_m1 = [n.build(layer, k - 1) for n in nulls3]
        rho = base["two_channel_rho"]
        thr = [c["threshold_p99_conditional"] for c in cells]
        if cells[0]["saturated"]:
            order_ok = bool(rho >= 1.0 and (cells[0]["exact1_rate"] or 1.0) < 0.01)
            order_state = "PASS" if order_ok else (
                "NOT ADJUDICABLE (saturated)" if rho >= 1.0 else "FAIL")
        elif rho > max(thr):
            order_state = "PASS"
        elif rho > min(thr):
            order_state = "MARGINAL"
        else:
            order_state = "FAIL"
        thr_m1 = cells_m1[0]["threshold_p99_conditional"]
        loo_rho_ok = all(r is not None and thr_m1 is not None and
                         (r > thr_m1 or (cells_m1[0]["saturated"] and r >= 1.0))
                         for r in base["loo_rhos"])
        loo_cl_ok = base["loo_closure_pass"]
        if order_state == "FAIL":
            v = "FAIL (order)"
        elif order_state.startswith("NOT"):
            v = "NOT ADJUDICABLE (saturated cell)"
        elif not loo_cl_ok:
            v = "FAIL (LOO closure)"
        elif not loo_rho_ok:
            v = "FAIL (LOO order)"
        else:
            v = order_state
        base.update(order_state=order_state, thresholds_3seeds=thr,
                    threshold_km1_seed0=thr_m1, exact1_rate=cells[0]["exact1_rate"],
                    loo_rho_ok_amended=bool(loo_rho_ok))
    base["verdict_amended"] = v
    return base


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--layers", required=True)
    ap.add_argument("--flags", required=True, help="script-22 output json (deep group)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-clusterlike", type=int, default=N_CL)
    ap.add_argument("--n-plant", type=int, default=N_PLANT)
    a = ap.parse_args()
    t0 = time.time()
    layers = [int(x) for x in a.layers.split(",")]

    g22 = json.loads(Path(a.flags).read_text())
    token_map = json.loads((EXT / "results/token_map.json").read_text())
    cal_ids = {int(x) for key in ("months", "days") for ids in token_map[key].values()
               for x in ids}
    # flags = scripts/22's flagged screened rows, deduplicated by (layer, member_hash) in arm
    # order -- the same dedup 22's verdict applies to build other_rings, but keeping the hash
    flags, seen_h = [], set()
    for lr in g22["layers"]:
        if lr["layer"] not in layers:
            continue
        for arm, v in lr["arms"].items():
            for r in (v.get("screened") or []):
                if r["flagged"] and (lr["layer"], r["member_hash"]) not in seen_h:
                    seen_h.add((lr["layer"], r["member_hash"]))
                    flags.append(dict(layer=lr["layer"], arm=arm, k=r["k"],
                                      member_hash=r["member_hash"], gap_ratio=r["gap_ratio"],
                                      p_bg=r["p_bg"], p_closed=r["p_closed"],
                                      bh_pass=r.get("bh_pass_within_layer_arm")))
    stub = dict(status="DECLARATION STUB", script="scripts/e42d_two_channel_deep.py",
                declared_at=time.strftime("%Y-%m-%dT%H:%M:%S"), layers=layers,
                flags_source=a.flags,
                flags_md5=hashlib.md5(Path(a.flags).read_bytes()).hexdigest(),
                n_flags=len(flags), n_flags_k_ge_9=sum(1 for f in flags if f["k"] >= 9),
                rules=__doc__, n_clusterlike=a.n_clusterlike, seeds_L2=list(SEEDS_L2))
    Path(a.out).write_text(json.dumps(stub, indent=1))
    print(f"stub written; {len(flags)} flags, {stub['n_flags_k_ge_9']} with k>=9", flush=True)

    tok, cnt, pool_ids, ext_ids = S42.build_pool(a.root)
    stored = np.load(EXT / "results/phase3_lag_action_robust_deltas.npz")["pool_ids"]
    assert np.array_equal(pool_ids.astype(np.int64), stored.astype(np.int64)), "pool drift"
    n_types, n_rows = len(pool_ids), len(tok)
    pool_pos = {int(t): i for i, t in enumerate(pool_ids)}
    counts_pool = cnt[pool_ids].astype(np.int64)
    lut = np.full(int(tok.max()) + 1, -1, dtype=np.int64)
    lut[pool_ids] = np.arange(n_types)
    snap = sorted((EXT / "hf_home/hub/models--google--gemma-3-1b-pt/snapshots").glob("*"))[0]
    from transformers import AutoTokenizer
    tokz = AutoTokenizer.from_pretrained(str(snap))
    strings = [(tokz.convert_ids_to_tokens(int(t)) or tokz.decode([int(t)])).replace("▁", " ")
               for t in pool_ids]
    blocks = S42.resolve_blocks(strings)

    def best_variant(ids):
        ip = [int(x) for x in ids if int(x) in pool_pos]
        return max(ip, key=lambda x: counts_pool[pool_pos[x]]) if ip else None
    P12 = np.asarray(sorted(pool_pos[best_variant(token_map["months"][m])] for m in MONTHS))
    ord_pos = []
    for w in S42.ORDINAL_WORDS:
        e = tokz.encode(w, add_special_tokens=False)
        if len(e) == 1 and int(e[0]) in pool_pos:
            ord_pos.append(pool_pos[int(e[0])])
    ord_pos = np.asarray(ord_pos)

    own, parity, per_lag = S42.pair_index(tok, n_rows, lut)
    C, rs = {}, {}
    for lag in S42.LAGS:
        pl = per_lag[lag]
        Cl = np.zeros((2, n_types, n_types), dtype=np.int64)
        for p in (0, 1):
            m = pl["succ_ok"] & (parity == p)
            np.add.at(Cl[p], (pl["prev"][m], own[m]), 1)
        C[lag] = Cl
        rs[lag] = Cl.sum(axis=2)
    included = {lag: (rs[lag][0] >= S42.MIN_PAIRS_PER_HALF) & (rs[lag][1] >= S42.MIN_PAIRS_PER_HALF)
                for lag in S42.LAGS}
    pred_par = {lag: np.where(per_lag[lag]["succ_ok"],
                              per_lag[lag]["prev"] + parity.astype(np.int32) * n_types,
                              np.int32(-1)).astype(np.int32) for lag in S42.LAGS}
    del parity, per_lag
    gc.collect()

    out = dict(stub, status="RUNNING", layers_detail={})
    for L in layers:
        tL = time.time()
        print(f"\n================ layer {L} ================", flush=True)
        TM_acc, S, seen = S42.stream_deltas(a.root, L, own, pred_par, n_rows, n_types, S42.D_SAE)
        assert np.array_equal(seen, counts_pool)
        TM_acc /= np.maximum(seen, 1)[:, None]
        sae = load_sae(L, "big", token=None)
        W = get_decoder(sae); del sae
        R = S42.project(TM_acc, W).astype(np.float64)          # script-22 representation
        for lag in S42.LAGS:
            for p in (0, 1):
                base = p * n_types
                for lo in range(0, n_types, 512):
                    hi = min(lo + 512, n_types)
                    S[lag][base + lo:base + hi] -= C[lag][p][lo:hi].astype(np.float64) @ TM_acc
        del TM_acc; gc.collect()
        halves = {}
        for lag in S42.LAGS:
            for p in (0, 1):
                S[lag][p * n_types:(p + 1) * n_types] /= np.maximum(rs[lag][p], 1)[:, None]
            h0 = S42.project(S[lag][:n_types], W).astype(np.float64)
            h1 = S42.project(S[lag][n_types:], W).astype(np.float64)
            S[lag] = None; gc.collect()
            w0 = rs[lag][0].astype(np.float64)[:, None]; w1 = rs[lag][1].astype(np.float64)[:, None]
            halves[lag] = (h0, h1, (w0 * h0 + w1 * h1) / np.maximum(w0 + w1, 1))
        del S, W; gc.collect()
        keep1, sd1, drop1, shares1 = S42.outlier_drop_rule(halves[1][0], halves[1][1], included[1])
        _, _, drop2, shares2 = S42.outlier_drop_rule(halves[2][0], halves[2][1], included[2])
        keep = np.ones(S42.D_MODEL, dtype=bool); keep[drop1] = False
        print(f"  [DROP L{L}] lag1 rule -> {drop1} shares {[round(x,3) for x in shares1]}; "
              f"lag2 rule -> {drop2}", flush=True)
        C1 = np.ascontiguousarray(R[:, keep])
        C2 = np.ascontiguousarray(halves[1][2][:, keep]).astype(np.float32).astype(np.float64)
        C2l2 = np.ascontiguousarray(halves[2][2][:, keep]).astype(np.float32).astype(np.float64)
        np.savez_compressed(EXT / f"results/lag_deltas_L{L}.npz", pool_ids=pool_ids.astype(np.int32),
                            lag1=halves[1][2][:, keep].astype(np.float16),
                            lag2=halves[2][2][:, keep].astype(np.float16),
                            kept_coords=np.flatnonzero(keep).astype(np.int32),
                            dropped_coords=np.asarray(drop1, dtype=np.int32),
                            channel1_R_predrop=R.astype(np.float32),
                            lag1_pool_noise_sd=sd1.astype(np.float32))
        del halves; gc.collect()

        # ---- reconstruct flags by hash (scripts/22 arms re-run, as 42 does) ----
        arms = S42.all_partitions(R, n_types)
        hm = {}
        for arm, cl in arms.items():
            for c in cl:
                hm.setdefault(S42.member_hash(c), (arm, [int(x) for x in c]))
        del arms

        Xc = C1 - C1.mean(0, keepdims=True)
        Xn = (Xc / np.maximum(np.linalg.norm(Xc, axis=1, keepdims=True), 1e-12)).astype(np.float32)
        nulls3 = [SeededNull(Xn, C1, C2, counts_pool, a.n_clusterlike, seed_offset=s) for s in SEEDS_L2]
        nulls_l2 = SeededNull(Xn, C1, C2l2, counts_pool, a.n_clusterlike, seed_offset=0)

        # ---- positive control: strict-12 ----
        pc = amended_verdict(P12, C1, C2, nulls3, L)
        print(f"  [STRICT-12 L{L}] gap {pc['gap_ratio']:.3f} rho {pc['two_channel_rho']} "
              f"-> {pc['verdict_amended']}", flush=True)
        validated = pc["verdict_amended"] in ("PASS", "MARGINAL")

        # ---- negative controls ----
        neg = dict(ordinal=amended_verdict(ord_pos, C1, C2, nulls3, L))
        lc = np.log10(np.maximum(counts_pool.astype(np.float64), 1.0))
        allowed = np.ones(n_types, dtype=bool)
        allowed[[pool_pos[x] for x in cal_ids if x in pool_pos]] = False
        cand = np.flatnonzero(allowed)
        rng = np.random.default_rng(7_300_000 + L)
        nearest = np.asarray([cand[np.argsort(np.abs(lc[cand] - lc[p]), kind="stable")[:5]] for p in P12])
        fp = dict(n=0, closed=0, passed=0, marginal=0)
        for _ in range(200):
            pick = nearest[np.arange(12), rng.integers(0, 5, 12)]
            if len(set(pick.tolist())) < 12:
                continue
            v = amended_verdict(pick, C1, C2, nulls3, L)
            fp["n"] += 1; fp["closed"] += int(v["closure_pass"])
            fp["passed"] += int(v["verdict_amended"] == "PASS")
            fp["marginal"] += int(v["verdict_amended"] == "MARGINAL")
        neg["freq_pseudo"] = fp
        thr_pair = (nulls3[0].build(L, 12)["threshold_p99_conditional"],
                    nulls3[0].build(L, 11)["threshold_p99_conditional"])
        rng = np.random.default_rng(7_400_000 + L)
        cp = dict(n=200, passed_noLOO=0, passed_frozen=0, rhos=[])
        for _ in range(200):
            perm = np.arange(n_types)
            for b in blocks:
                perm[b] = b[rng.permutation(len(b))]
            adj = S42.adjudicate(P12, C1, C2[perm], nulls3[0], L, thr_override=thr_pair)
            cp["passed_noLOO"] += int(adj["PASS_noLOO"])
            cp["passed_frozen"] += int(adj["verdict"] == "PASS")
            cp["rhos"].append(adj["two_channel_rho"])
        cp["rho_median"] = float(np.median([r for r in cp["rhos"] if r is not None]))
        cp.pop("rhos")
        neg["class_perm"] = cp
        print(f"  [NEG L{L}] ordinal {neg['ordinal']['verdict_amended']}; freq-pseudo {fp}; "
              f"class-perm noLOO {cp['passed_noLOO']}/200", flush=True)

        # ---- clean planted power at the strict-12 radii ----
        _, r1 = S42.plane_angles(C1[P12]); _, r2 = S42.plane_angles(C2[P12])
        thr12, thr11 = thr_pair
        th = 2 * np.pi * np.arange(12) / 12
        d1 = C1.shape[1]
        curve = []
        for rj in PLANT_RHOS:
            rng = np.random.default_rng(7_600_000 + int(1000 * rj) + L)
            nj = 0
            for _ in range(a.n_plant):
                def ring(radius):
                    B, _ = np.linalg.qr(rng.normal(size=(d1, 2)))
                    pts = np.column_stack([radius * np.cos(th), radius * np.sin(th)])
                    return (pts + rng.normal(0.0, rj * radius, size=pts.shape)) @ B.T
                g, r = S42.cell_stats(ring(r1), ring(r2))
                nj += int(g is not None and g < S42.GAP_RATIO_CLOSURE and r is not None
                          and thr12 is not None and r > thr12)
            curve.append(dict(rho=rj, recall_joint=nj / a.n_plant))
        print(f"  [PLANT L{L}] {curve}", flush=True)

        # ---- flags ----
        rows = []
        for f in flags:
            if f["layer"] != L:
                continue
            h = f["member_hash"]
            found = hm.get(h)
            row = dict(layer=L, arm=f["arm"], k=f["k"], member_hash=h,
                       stored_gap_ratio=f["gap_ratio"], p_bg=f["p_bg"], bh_pass=f["bh_pass"])
            if found is not None:
                row["gap_ratio_predrop_reproduces"] = bool(
                    abs(S42.gap_ratio_of(R[np.asarray(found[1])]) - f["gap_ratio"]) <= 1e-6)
            if found is None:
                row["verdict_amended"] = "NOT RECONSTRUCTED"; rows.append(row); continue
            M = np.asarray(sorted(found[1]))
            row["calendar"] = bool(any(int(pool_ids[i]) in cal_ids for i in M))
            row["tokens"] = [strings[i] for i in M]
            adj = amended_verdict(M, C1, C2, nulls3, L)
            row.update({k2: adj[k2] for k2 in ("gap_ratio", "two_channel_rho", "verdict_amended",
                                                "n_loo_closure_fail", "thresholds_3seeds",
                                                "exact1_rate", "order_state") if k2 in adj})
            if adj["two_channel_rho"] is not None and len(M) >= 9:
                l2 = S42.adjudicate(M, C1, C2l2, nulls_l2, L, want_loo=False)
                row["lag2_rho"] = l2["two_channel_rho"]; row["lag2_clears"] = l2["rho_pass"]
            row["tier_validated_at_layer"] = validated
            rows.append(row)
            print(f"  flag k={len(M):<3} cal={row['calendar']!s:<5} gap {adj['gap_ratio']} rho "
                  f"{adj['two_channel_rho']} -> {adj['verdict_amended']}", flush=True)
        out["layers_detail"][str(L)] = dict(
            drop_lag1=drop1, drop_lag1_shares=shares1, drop_lag2=drop2,
            strict12=pc, tier_validated=validated, negative_controls=neg,
            planted_clean=curve, r1=r1, r2=r2, flags=rows,
            null_cells=[v for n in nulls3 for v in n.cache.values()],
            seconds=round(time.time() - tL, 1))
        Path(a.out).write_text(json.dumps(out, indent=1, default=float))
        del C1, C2, C2l2, R, Xn, Xc, nulls3, nulls_l2, hm
        gc.collect()

    allrows = [r for v in out["layers_detail"].values() for r in v["flags"]]
    out["summary"] = dict(
        n_flags=len(allrows),
        n_k_ge_9=sum(1 for r in allrows if r["k"] >= 9),
        verdict_counts={v: sum(1 for r in allrows if r.get("verdict_amended") == v)
                        for v in sorted({r.get("verdict_amended") for r in allrows})},
        passers=[r for r in allrows if r.get("verdict_amended") in ("PASS", "MARGINAL")],
        strict12_by_layer={L: v["strict12"]["verdict_amended"] for L, v in out["layers_detail"].items()},
        expected_false_passers=0.01 * sum(1 for r in allrows if r.get("order_state")))
    out["status"] = "COMPLETE"
    out["runtime_seconds"] = round(time.time() - t0, 1)
    Path(a.out).write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps(out["summary"], indent=1, default=float), flush=True)


if __name__ == "__main__":
    main()
