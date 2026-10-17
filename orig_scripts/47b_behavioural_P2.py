"""47b -- the 47 behavioural tier, unchanged, on the five non-calendar PASS_P2 sets.

Addendum Q (docs/phase5_addendum_Q_2026-08-22.md), Q1: 48b's P2 re-adjudication left nine
PASS_P2 rows. Four are month subsets and are not independent of the strict-12 anchor. The
other five are non-calendar k = 9-10 sets at L6/B and L7/B, and Q1 sends exactly those five
to the behavioural tier under the 47 rig UNCHANGED -- same epsilon grid, same prompt arms,
same nulls, same PASS rule, same declared reductions. Q2 fixes what the verdicts mean: a
PASS_P2 set that passes here is a CANDIDATE FINDING pending independent re-derivation of its
whole chain; one that fails is retired.

"Unchanged" is enforced by IMPORT, not by copying: every constant, the plane/tangent fit,
the Runner, eval_set and summarise are pulled out of scripts/47_behavioural_confirm.py at
run time and are the same objects that produced results/phase5_behavioural.json. scripts/47
is not edited. What lives here is the target list, the member-position recovery for these
rows, and the per-set driver loop (47's loop lives inside its own main() and cannot be
imported without also importing its target list).

THE ANCHOR IS AN EXACT REPRODUCTION CHECK, NOT JUST A POSITIVE CONTROL. strict-12 at L6 is
placed FIRST in the target list, exactly as in 47. Every generator 47 uses is seeded the same
way and is consumed in the same order, and the seed for a set's contexts is a function of
(layer, k) alone -- so the whole L6_strict12 block here, nulls included, must come out
bit-identical to 47's. It is checked against the stored 47 numbers at the end of the run and
any mismatch is reported as a deviation rather than smoothed over.

MEMBER RECOVERY. Positions come from the stored ``token_ids`` of the PASS_P2 rows in
results/phase5_readjudication_P.json, mapped through the fulln pool -- not from token
strings, which is what Q's launch note requires and what avoids any dependence on the
tokenizer's string rendering. Every recovered set is checked to reproduce its stored
member_hash before a single forward runs.
"""
import argparse
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


def _import_47():
    """Import scripts/47_behavioural_confirm.py by path (the name is not an identifier)."""
    p = REPO / "scripts" / "47_behavioural_confirm.py"
    spec = importlib.util.spec_from_file_location("s47_behavioural_confirm", p)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


S47 = _import_47()

from gfdisc import fits, rig                      # noqa: E402
from gfdisc.env import DEVICE, MODEL_NAME         # noqa: E402
from gfdisc.model import load_model               # noqa: E402

READJ = "results/phase5_readjudication_P.json"
PRIOR_47 = "results/phase5_behavioural.json"

#: (name, layer, member_hash, kind). The anchor is FIRST -- see the module docstring.
TARGETS = [
    ("L6_strict12", 6, "a36a4f4f3e3b", "anchor"),
    ("L6_c_k9", 6, "9a5a83ca0105", "pass_p2"),
    ("L6_d_k9", 6, "045a6dd71065", "pass_p2"),
    ("L6_e_k9", 6, "719152009d79", "pass_p2"),
    ("L7_h_k9", 7, "bf0c6958bc1f", "pass_p2"),
    ("L7_i_k10", 7, "2eff8657fbd3", "pass_p2"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/phase5_behavioural_P2.json")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    out_path = REPO / a.out
    if a.smoke and out_path.name == "phase5_behavioural_P2.json":
        raise SystemExit("refusing to write a smoke run to the real --out")
    t0 = time.time()

    n_prompts = 4 if a.smoke else S47.N_PROMPTS
    n_prompts_null = 2 if a.smoke else S47.N_PROMPTS_NULL
    n_rand = 2 if a.smoke else S47.N_RAND_DIRS
    n_nullsets = 2 if a.smoke else S47.N_NULL_SETS
    targets = TARGETS[:2] if a.smoke else TARGETS

    # ---------------- member sets, recovered from stored token_ids and hash-checked ------
    z = np.load(REPO / S47.FULLN_RAW)
    pool_ids = z["pool_ids"].astype(np.int64)
    R_by_layer = {L: z[f"R_L{L}"].astype(np.float32) for L in sorted({t[1] for t in TARGETS})}
    pos_of_id = {int(t): i for i, t in enumerate(pool_ids)}

    readj = json.loads((REPO / READJ).read_text())
    rows_by_hash = {}
    for r in readj["summary"]["all_PASS_P2"] + readj["summary"]["all_R14_PASS"]:
        rows_by_hash.setdefault(r["member_hash"], r)
    members_by_hash, source_rows = {}, {}
    for name, L, h, kind in targets:
        r = rows_by_hash[h]
        mp = sorted(pos_of_id[int(t)] for t in r["token_ids"])
        assert S47.member_hash(mp) == h, f"recovered positions do not reproduce {h}"
        assert len(mp) == int(r["k"]), f"k mismatch for {h}"
        members_by_hash[h] = mp
        source_rows[h] = dict(k=int(r["k"]), gap_ratio=r["gap_ratio"], rho=r["rho"],
                              verdict=r.get("verdict"), is_strict12=bool(r.get("is_strict12")),
                              cell_saturated=r.get("cell_saturated"),
                              p2_exact1_rate=r.get("p2_exact1_rate"),
                              sources=r.get("sources"))

    print(f"[{time.time()-t0:6.1f}s] loading model {MODEL_NAME} on {DEVICE}", flush=True)
    model, tok = load_model(token=os.environ.get("HF_TOKEN"))

    # ---------------- geometry, fitted before any forward ----------------
    geos, sets_meta = {}, []
    for name, L, h, kind in targets:
        mp = members_by_hash[h]
        R = np.delete(R_by_layer[L][np.asarray(mp)].astype(np.float64), S47.R10_DROP, axis=1)
        g = S47.plane_of(R)
        geos[name] = (g, mp, L)
        sets_meta.append(dict(name=name, layer=L, member_hash=h, kind=kind, k=g["k"],
                              pc_var_top2=g["pc_var_top2"], gap_ratio=g["gap_ratio"],
                              radius_ratio=g["radius_ratio"], circ_resid=g["circ_resid"],
                              cyclic_order_positions=[mp[i] for i in g["order"]],
                              source_row=source_rows[h]))

    # ---------------- declaration stub, written BEFORE any forward ----------------
    declared_at = datetime.now().isoformat(timespec="seconds")
    declared = dict(
        status="DECLARED", script="scripts/47b_behavioural_P2.py",
        declared_at=declared_at,
        stub_note="written before any forward pass; results overwrite this file carrying declared_at",
        addenda_in_force=["docs/phase5_addendum_Q_2026-08-22.md Q1-Q2",
                          "docs/phase4_prereg_2026-08-22.md section 2.6"],
        rig=("scripts/47_behavioural_confirm.py imported by path and used unchanged: "
             "constants, plane_of, embed_1152, random_orth_dirs, Runner, eval_set and "
             "summarise are the same objects that produced results/phase5_behavioural.json. "
             "47 is not edited. Only the target list, the member recovery and the driver "
             "loop live in 47b."),
        member_recovery=("pool positions from the stored token_ids of the PASS_P2 / R14 rows "
                         "of results/phase5_readjudication_P.json, mapped through the fulln "
                         "pool_ids; each set asserted to reproduce its stored member_hash"),
        candidates=sets_meta,
        rules=dict(
            inherited_from_47=("eps grid, prompt arms, statistic, nulls (i)/(ii)/(iii) and "
                               "the PASS rule are 47's, unchanged; see that file's stub for "
                               "their full statements"),
            eps_grid=list(S47.EPS_GRID), eps_null=S47.EPS_NULL,
            n_prompts=n_prompts, n_prompts_null=n_prompts_null,
            n_rand_dirs=n_rand, n_null_sets=n_nullsets,
            PASS=("sign_consistency >= (k-1)/k at SOME eps in the grid (natural arm) AND "
                  f"mean Delta at eps={S47.EPS_NULL} above the 99th percentile of null "
                  f"(iii)'s set means AND above ALL {n_rand} of null (i)'s direction means"),
            Q2_reading=("a set that PASSES is reported as a CANDIDATE FINDING pending an "
                        "independent re-derivation of its full chain (proposal, tier, 47b); "
                        "one that FAILS is retired"),
            anchor=("strict-12 at L6 is placed first and every generator is seeded and "
                    "consumed as in 47, so its whole block -- nulls included -- must "
                    "reproduce 47's stored numbers bit-identically; the comparison is made "
                    "at the end of the run and reported either way"),
        ),
        deviations_from_prereg=[
            "route_intermediate_mass replaced by the in-set softmax readout (47's deviation)",
            f"null (i) uses {n_rand} random directions per member, not the prereg's 20, and "
            f"nulls run {n_prompts_null} prompts per member rather than the candidate arm's "
            f"{n_prompts} -- identical to 47's declared reductions, kept so that 47b and 47 "
            f"are comparable. Null (iii) runs the prereg's full {n_nullsets} sets.",
            "Delta is baseline-subtracted (47's declared reading of prereg 2.6)",
        ] + (["SMOKE RUN -- not a result"] if a.smoke else []),
        config=dict(model=MODEL_NAME, device=DEVICE, dtype="bfloat16", attn="eager",
                    batch_size=S47.BATCH, context_len=S47.CONTEXT_LEN,
                    seeds=dict(context=S47.SEED_CONTEXT, dirs=S47.SEED_DIRS,
                               nullsets=S47.SEED_NULLSETS)),
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(declared, indent=1))
    print(f"[{time.time()-t0:6.1f}s] stub written to {out_path}", flush=True)

    # ---------------- hook identity check ----------------
    tokens2d = rig.load_tokens2d(REPO / S47.CACHE_ROOT)
    eligible_all = rig.eligible_docs(tokens2d, ())
    probe = S47.Runner(model, tok, 6, DEVICE)
    blk = probe.prepare([np.zeros(0, dtype=np.int64)], int(pool_ids[0]))
    lg0, hn0 = probe.baseline(blk)
    lg_z = probe.steered(blk, hn0, np.zeros(1152), 0.0)
    hook_max = float((lg0 - lg_z).abs().max())
    print(f"[{time.time()-t0:6.1f}s] hook identity: max |delta logit| = {hook_max:.3e}",
          flush=True)
    assert hook_max == 0.0, "zero-delta hook is not an identity"

    # ---------------- the run (47's driver loop, same order of generator consumption) ----
    rng_dirs = np.random.default_rng(S47.SEED_DIRS)
    rng_nullsets = np.random.default_rng(S47.SEED_NULLSETS)
    results = {}
    for name, L, h, kind in targets:
        geo, mp, L = geos[name]
        k = geo["k"]
        runner = S47.Runner(model, tok, L, DEVICE)
        t_set = time.time()
        print(f"[{time.time()-t0:6.1f}s] {name}: L{L} k={k} starting", flush=True)

        tang = S47.embed_1152(geo["tangent"])
        dirs = {}
        rd = np.stack([S47.random_orth_dirs(tang[i], n_rand, rng_dirs) for i in range(k)])
        for j in range(n_rand):
            dirs[f"rand{j}"] = rd[:, j, :]
        u1, u2, th = geo["u1"], geo["u2"], geo["theta"]
        dirs["rot_p90"] = S47.embed_1152(-(np.cos(th)[:, None] * u1[None, :])
                                         - (np.sin(th)[:, None] * u2[None, :]))
        dirs["rot_m90"] = S47.embed_1152((np.cos(th)[:, None] * u1[None, :])
                                         + (np.sin(th)[:, None] * u2[None, :]))

        rng_ctx = np.random.default_rng(S47.SEED_CONTEXT + L * 100 + k)
        rows_nat = S47.eval_set(runner, geo, mp, pool_ids, tokens2d, eligible_all, n_prompts,
                                S47.EPS_GRID, rng_ctx, "natural")
        rng_ctx_null = np.random.default_rng(S47.SEED_CONTEXT + 50_000 + L * 100 + k)
        rows_null_dirs = S47.eval_set(runner, geo, mp, pool_ids, tokens2d, eligible_all,
                                      n_prompts_null, (), rng_ctx_null, "natural",
                                      extra_dirs=dirs)
        rows_bos = S47.eval_set(runner, geo, mp, pool_ids, tokens2d, eligible_all, 1,
                                S47.EPS_GRID, rng_ctx, "bos")
        rows_matched = S47.eval_set(
            runner, geo, mp, pool_ids, tokens2d, eligible_all, n_prompts_null,
            (S47.EPS_NULL,),
            np.random.default_rng(S47.SEED_CONTEXT + 50_000 + L * 100 + k), "natural")

        nat = {f"eps={e}": S47.summarise(rows_nat, f"eps={e}") for e in S47.EPS_GRID}
        bos = {f"eps={e}": S47.summarise(rows_bos, f"eps={e}") for e in S47.EPS_GRID}
        null_i = {f"rand{j}": S47.summarise(rows_null_dirs, f"rand{j}") for j in range(n_rand)}
        null_ii = {nm: S47.summarise(rows_null_dirs, nm) for nm in ("rot_p90", "rot_m90")}
        matched = S47.summarise(rows_matched, f"eps={S47.EPS_NULL}")

        null_iii = []
        s = 0
        while len(null_iii) < n_nullsets:
            s += 1
            pick = sorted(rng_nullsets.choice(len(pool_ids), size=k, replace=False).tolist())
            if pick == sorted(mp):
                continue
            Rn = np.delete(R_by_layer[L][np.asarray(pick)].astype(np.float64),
                           S47.R10_DROP, axis=1)
            if fits.degenerate(Rn):
                continue
            gn = S47.plane_of(Rn)
            rn = S47.eval_set(runner, gn, pick, pool_ids, tokens2d, eligible_all,
                              n_prompts_null, (S47.EPS_NULL,),
                              np.random.default_rng(S47.SEED_NULLSETS + 1000 * s + k),
                              "natural")
            sm = S47.summarise(rn, f"eps={S47.EPS_NULL}")
            sm["member_hash"] = S47.member_hash(pick)
            sm["gap_ratio"] = gn["gap_ratio"]
            null_iii.append(sm)
            print(f"[{time.time()-t0:6.1f}s]   {name} null-set {len(null_iii)}/{n_nullsets} "
                  f"mean_delta={sm['mean_delta']:+.4f}", flush=True)

        null_means = np.array([x["mean_delta"] for x in null_iii], dtype=np.float64)
        p99 = float(np.percentile(null_means, 99))
        rand_means = np.array([null_i[f"rand{j}"]["mean_delta"] for j in range(n_rand)])
        sc_ok = any(nat[f"eps={e}"]["sign_consistency"] >= (k - 1) / k for e in S47.EPS_GRID)
        d_pin = nat[f"eps={S47.EPS_NULL}"]["mean_delta"]
        above_iii = bool(d_pin > p99)
        above_i = bool(d_pin > rand_means.max())
        mono = bool(nat["eps=0.005"]["mean_delta"] < nat["eps=0.01"]["mean_delta"]
                    < nat["eps=0.02"]["mean_delta"])

        results[name] = dict(
            layer=L, member_hash=h, kind=kind, k=k, source_row=source_rows[h],
            geometry=dict(pc_var_top2=geo["pc_var_top2"], gap_ratio=geo["gap_ratio"],
                          radius_ratio=geo["radius_ratio"], circ_resid=geo["circ_resid"]),
            natural=nat, bos_only=bos, matched_budget_at_eps_null=matched,
            null_i_random_orth=null_i, null_ii_rotations=null_ii,
            null_iii_random_sets=null_iii,
            null_iii_p99=p99, null_iii_max=float(null_means.max()),
            null_iii_mean=float(null_means.mean()), null_i_max=float(rand_means.max()),
            monotonic_in_eps=mono,
            PASS=bool(sc_ok and above_iii and above_i),
            pass_detail=dict(sign_consistency_ok=bool(sc_ok), above_null_iii_p99=above_iii,
                             above_all_null_i=above_i, mean_delta_at_eps_null=d_pin),
            Q2_status=("CANDIDATE FINDING pending independent re-derivation"
                       if (sc_ok and above_iii and above_i) and kind == "pass_p2"
                       else ("RETIRED" if kind == "pass_p2" else "anchor")),
            per_member=dict(natural=rows_nat, null_dirs=rows_null_dirs, bos=rows_bos),
            n_forward=runner.n_forward, seconds=round(time.time() - t_set, 1),
        )
        r = results[name]
        print(f"[{time.time()-t0:6.1f}s] {name}: PASS={r['PASS']} "
              f"meanD@{S47.EPS_NULL}={d_pin:+.4f} sc="
              f"{[round(nat[f'eps={e}']['sign_consistency'],3) for e in S47.EPS_GRID]} "
              f"null_iii_p99={p99:+.4f} null_i_max={rand_means.max():+.4f} "
              f"({runner.n_forward} forwards, {r['seconds']}s)", flush=True)

    # ---------------- anchor reproduction check against 47 ----------------
    repro = dict(checked=False)
    prior_p = REPO / PRIOR_47
    if not a.smoke and prior_p.exists() and "L6_strict12" in results:
        prior = json.loads(prior_p.read_text()).get("results", {}).get("L6_strict12")
        if prior:
            here, diffs = results["L6_strict12"], {}
            for blockname in ("natural", "bos_only", "null_i_random_orth",
                              "null_ii_rotations"):
                for cond, sm in here[blockname].items():
                    old = prior.get(blockname, {}).get(cond, {})
                    for stat in ("mean_delta", "sign_consistency"):
                        if stat in old and old[stat] != sm[stat]:
                            diffs[f"{blockname}/{cond}/{stat}"] = [old[stat], sm[stat]]
            for stat in ("null_iii_p99", "null_iii_mean", "null_iii_max", "null_i_max"):
                if prior.get(stat) != here.get(stat):
                    diffs[stat] = [prior.get(stat), here.get(stat)]
            repro = dict(checked=True, bit_identical=(not diffs), n_differing=len(diffs),
                         diffs=diffs,
                         note=("the anchor block must reproduce scripts/47's stored "
                               "L6_strict12 numbers exactly; any entry in diffs is a "
                               "deviation, not a rounding artifact"))
            print(f"[{time.time()-t0:6.1f}s] anchor reproduction vs 47: "
                  f"bit_identical={repro['bit_identical']} ({len(diffs)} differing)",
                  flush=True)

    declared.update(status="COMPLETE", results=results,
                    anchor_reproduction_vs_47=repro,
                    finished_at=datetime.now().isoformat(timespec="seconds"),
                    runtime_seconds=round(time.time() - t0, 1),
                    hook_identity_max_logit_diff=hook_max,
                    total_forwards=int(sum(v["n_forward"] for v in results.values())))
    out_path.write_text(json.dumps(declared, indent=1))
    print(f"[{time.time()-t0:6.1f}s] wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
