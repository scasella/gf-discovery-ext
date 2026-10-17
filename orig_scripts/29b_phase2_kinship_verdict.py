"""Verdict assignment for the Phase-2 kinship run.  RULES FROZEN BEFORE THE RESULTS EXIST.

This file is written and its rules are fixed while scripts/29_phase2_kinship.py is still
running: at the moment of writing, results/phase2_kinship.json contains the declarations
and (at most) the stage-A data-free calibration.  No stage-B placement and no stage-C
supervised cell has been produced, let alone read.  The rules below are therefore
mechanical in advance rather than fitted to an outcome, which is the whole point of
docs/phase2_prereg_2026-08-21.md.

EXPOSURE RECORD, stated plainly rather than minimised (mirroring prereg section 2):
while developing scripts/29_phase2_kinship.py, three smoke runs WERE seen before these
verdict rules were frozen --
  * stage A in full (the saturation calibration, the enumeration floors, the attainable
    power table, the planted-ring controls);
  * stage B at LAYER 0 ONLY, all five arms (co-location counts, the P3 closure numbers
    and the additive R^2 at layer 0);
  * stage C at LAYERS 0 and 12 ONLY, all three readouts.
Nothing at any other layer was seen.  The verdict rules below were nonetheless written to
be mechanical, and every threshold in them is stated as a count against an
expected-by-chance count rather than as a hand-picked cutoff.  A reader who distrusts
that should read V6 as exposed at 2 of 26 supervised layers and V3/V4/V5 as exposed at
1 of 11 blind layers.

VERDICT VOCABULARY -- prereg 4.3 and 3.2, no additions:
  recovered_blind | structure_exists_supervised_only | ordered_not_closed |
  untestable | clean_negative

THE RULES

V1  kinship_blind_lattice = `untestable`, forced by stage A and NOT by any result.
    prereg 3.2 step 4 requires BOTH contrasts separable for a lattice claim.  At the blind
    arity (6 members over the >=100 floor) the gender contrast has an exhaustive-null
    floor of 2/20 = 0.10 and an empirically attainable floor of 0.18 (S2) / 0.28 (S1) on a
    PERFECT planted lattice pushed through this exact code path.  A test that cannot reach
    alpha = 0.05 for a perfect signal is untestable by prereg 4.1(ii)'s own principle, so
    no blind lattice negative may be reported.

V2  kinship_blind_lineality = `untestable`, registered in prereg 3.2 in advance: every
    collateral (lineal = 0) term is below the >=100 discovery floor.

V3  kinship_blind_generation.  Testable: S2's attainable floor at k = 6 is 0.037 < 0.05,
    but only just, so the verdict carries "marginal power" whatever it says.
      `separable_blind`   iff >= 4 of the 11 layers have S2 p < 0.05 at the declared
                          PRIMARY arm's member set (the six type means), against an
                          expected-by-chance count of 11 * 0.05 = 0.55.
      `clean_negative`    otherwise, annotated with the 0.037 attainable floor.
    BH is reported but cannot pass at this arity (rank-1 threshold 0.05/11 = 0.0045 is
    below the 0.037 floor); that impossibility is stated rather than hidden.

V4  P1 co-location, on the declared primary arm `centred_leiden`.
      `co_located`     iff >= 3 of 11 layers have BOTH p_uniform < 0.05 AND
                       p_count_decile_matched < 0.05, against a conservative
                       expected-by-chance count of 11 * 0.05 = 0.55.
      `not_co_located` otherwise.
    All 55 (layer, arm) cells are reported with raw counts and the expected-by-chance
    count; the verdict is taken on the primary arm only, so a secondary arm cannot be
    promoted after the fact.

V5  P3 closure.  `prediction_holds` iff NO cell in either arm closes (closure = p_closed
    < 0.05 AND p_bg <= 0.05).  Any pass -> `prediction_failed`, reported as a failure of
    the kinship prediction AND as evidence that the closure tier accepts a non-cyclic
    ontology, which per the prereg would require the gate itself to be re-examined.

V6  kinship_supervised, at the PRIMARY readout `suffix` (the `token` readout is declared
    embedding-confounded and cannot carry the verdict).
      `structure_exists_supervised_only` iff ALL THREE hold:
         (a) generation S2 BH-passes in more of the 78 cells than the 3.9 expected by
             chance, AND at >= 1 suffix layer;
         (b) gender S2 likewise;
         (c) the additive gen x gender R^2 likewise;
         and (d) at >= 1 suffix layer all three pass in the SAME cell.
      `clean_negative` otherwise.
    `recovered_blind` is unreachable for kinship by V1/V2 and is not offered.
    The verdict STRING itself carries the provenance: the prereg-registered estimator is
    structurally vacuous (p == 1.0 by construction), so any positive rests on the
    script-time-declared secondary estimators S1/S2, and the gender contrast itself was
    added at prereg time and is not in commit 3cadb69.

V7  overall_kinship_verdict = V6's value, because V1 puts `recovered_blind` out of reach.
    It is reported next to V4 and V5 so a reader sees co-location and closure separately.

Two numbers are recomputed here so they cannot be misread downstream:
  * the Spearman CEILING given the generation tie structure -- spearman(y, y) with ties
    broken monotonically is 0.965, not 1.0, so an observed 0.965 is PERFECT ordering;
  * `at_exhaustive_floor` -- whether an exhaustive-null p equals the smallest value that
    null can produce, i.e. the observed score is at the ceiling of the enumerated space.
"""
import argparse, json, math, sys, time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from gfdisc.ontologies import KINSHIP, KINSHIP_GENERATION
from gfdisc.nulls import benjamini_hochberg

ALPHA = 0.05
BLIND_ARMS = ("centred_leiden", "centred_refined", "raw_leiden", "raw_refined", "hdbscan")
PRIMARY_ARM = "centred_leiden"
PRIMARY_READOUT = "suffix"


def _ranks(v):
    v = np.asarray(v, float)
    order = np.argsort(v, kind="mergesort")
    r = np.empty(len(v), float); r[order] = np.arange(len(v), dtype=float)
    uniq, inv, cnt = np.unique(v, return_inverse=True, return_counts=True)
    if (cnt > 1).any():
        s = np.zeros(len(uniq)); np.add.at(s, inv, r); r = (s / cnt)[inv]
    return r


def spearman(a, b):
    ra, rb = _ranks(a), _ranks(b)
    ra = ra - ra.mean(); rb = rb - rb.mean()
    d = np.linalg.norm(ra) * np.linalg.norm(rb)
    return float(ra @ rb / d) if d > 0 else float("nan")


def spearman_ceiling(y):
    """The largest Spearman a tie-free prediction can reach against a TIED code vector."""
    y = np.asarray(y, float)
    strict = np.argsort(np.argsort(y, kind="stable"), kind="stable").astype(float)
    return spearman(strict, y)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", default="results/phase2_kinship.json")
    a = ap.parse_args()
    p = REPO / a.path
    res = json.loads(p.read_text())
    V = dict(rules_frozen_at="2026-08-21, while scripts/29_phase2_kinship.py was still "
                             "running and before any stage-B or stage-C result existed",
             rules_script="scripts/29b_phase2_kinship_verdict.py",
             applied_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
             vocabulary=["recovered_blind", "structure_exists_supervised_only",
                         "ordered_not_closed", "untestable", "clean_negative"],
             exposure_before_freeze=("stage A in full; stage B at layer 0 only, all five "
                                     "arms; stage C at layers 0 and 12 only, all three "
                                     "readouts.  No other layer was seen."))

    A = res.get("stage_a_calibration", {})
    pw = A.get("attainable_power", {})
    V["attainable_power_used"] = {k: pw[k] for k in sorted(pw)}

    # ---- V1 / V2 -----------------------------------------------------------------
    V["V1_blind_lattice"] = dict(
        verdict="untestable",
        reason=("prereg 3.2 step 4 needs BOTH contrasts separable; at the blind arity the "
                "gender contrast has an exhaustive floor of 2/20 = 0.10 and an attainable "
                f"floor of {pw.get('k6.gender.s2_loo_ridge', {}).get('best_p_over_snr_ladder')} "
                "(S2) on a PERFECT planted lattice, so it cannot reach alpha = 0.05 "
                "whatever the model does"),
        forced_by="stage A, not by any result")
    V["V2_blind_lineality"] = dict(
        verdict="untestable",
        reason="every collateral (lineal = 0) term is below the >=100 discovery floor; "
               "registered untestable in prereg 3.2 in advance")

    B = res.get("blind_arm")
    if B and B.get("layers"):
        layers = [L for L in B["layers"] if L.get("reproduced")]
        V["blind_reproduction"] = dict(
            n_layers_attempted=len(B["layers"]), n_layers_reproduced=len(layers),
            all_reproduced=bool(B.get("all_layers_reproduced")))
        # ---- V3 generation -------------------------------------------------------
        rows = []
        for L in layers:
            ax = L.get("type_mean_axes", {}).get("generation", {})
            if "s2_loo_ridge" in ax:
                rows.append(dict(layer=L["layer"], embedding_adjacent=L["embedding_adjacent"],
                                 rho=ax["s2_loo_ridge"]["observed"],
                                 p=ax["s2_loo_ridge"]["p_sampled"],
                                 p_exact=ax["s2_loo_ridge"].get("p_exact"),
                                 s1_p=ax["s1_correlation"]["p_sampled"],
                                 registered_p=ax["registered"]["p_sampled"]))
        npass = sum(1 for r in rows if r["p"] < ALPHA)
        floor = pw.get("k6.generation.s2_loo_ridge", {}).get("best_p_over_snr_ladder")
        V["V3_blind_generation"] = dict(
            verdict="separable_blind" if npass >= 4 else "clean_negative",
            n_layers=len(rows), n_layers_p_lt_05=npass, expected_by_chance=round(ALPHA * len(rows), 2),
            attainable_floor_s2=floor,
            bh_impossible_note=("BH within the 11 layers cannot pass at this arity: the "
                                f"rank-1 threshold 0.05/{max(len(rows),1)} is below the "
                                f"attainable floor {floor}"),
            per_layer=rows,
            marginal_power_caveat=("S2's attainable floor at k = 6 is 0.037, only just "
                                   "under alpha; this verdict is marginal-power either way"),
            spearman_ceiling_given_ties=spearman_ceiling(
                [KINSHIP_GENERATION[KINSHIP.index(m["word"])]
                 for m in B["members_in_frozen_pool"]]),
            spearman_ceiling_note="an observed rho equal to this ceiling IS perfect ordering")
        # ---- V4 co-location ------------------------------------------------------
        grid, prim = [], 0
        for L in layers:
            for arm in BLIND_ARMS:
                e = L["arms"].get(arm)
                if not e:
                    continue
                c = e["colocation"]
                both = bool(c["p_uniform"] < ALPHA and c["p_count_decile_matched"] < ALPHA)
                grid.append(dict(layer=L["layer"], arm=arm, n_together=e["n_together"],
                                 n_members=len(B["members_in_frozen_pool"]),
                                 p_uniform=c["p_uniform"],
                                 p_decile=c["p_count_decile_matched"],
                                 both_below_05=both,
                                 flagged_ring=e.get("best_cluster_is_a_flagged_ring")))
                if arm == PRIMARY_ARM and both:
                    prim += 1
        V["V4_colocation_P1"] = dict(
            verdict="co_located" if prim >= 3 else "not_co_located",
            primary_arm=PRIMARY_ARM, n_primary_layers_both_below_05=prim,
            n_primary_layers=len(layers),
            expected_by_chance_primary=round(ALPHA * len(layers), 2),
            n_cells_total=len(grid),
            n_cells_both_below_05=int(sum(g["both_below_05"] for g in grid)),
            expected_by_chance_all_cells=round(ALPHA * len(grid), 2),
            grid=grid)
        # ---- V5 closure ----------------------------------------------------------
        bl = [dict(layer=L["layer"], p_closed=L["type_mean_closure_P3"]["p_closed"],
                   p_bg=L["type_mean_closure_P3"]["p_bg"],
                   flagged=L["type_mean_closure_P3"]["flagged"]) for L in layers]
        V["V5_closure_P3_blind"] = dict(
            verdict="prediction_holds" if not any(x["flagged"] for x in bl)
            else "prediction_FAILED",
            n_layers=len(bl), n_layers_closing=int(sum(x["flagged"] for x in bl)),
            per_layer=bl,
            ring_catalogue=B.get("ring_catalogue_scan", {}).get(
                "n_rings_containing_any_kinship_term"),
            note="P3 predicts closure FAILS.  A pass would be a failure of the kinship "
                 "prediction and a warning about the closure tier's specificity.")
        # ---- blind additive (the one fully powered registered blind statistic) ----
        ad = [dict(layer=L["layer"], r2=L["type_mean_additive"]["r2"],
                   p=L["type_mean_additive"]["p_joint_shuffle"]) for L in layers]
        V["blind_additive_gen_x_gender"] = dict(
            n_layers=len(ad), n_p_lt_05=int(sum(x["p"] < ALPHA for x in ad)),
            expected_by_chance=round(ALPHA * len(ad), 2),
            attainable_floor=pw.get("k6.additive_gen_x_gender", {}).get("best_p_over_snr_ladder"),
            false_acceptance_on_unstructured_noise=pw.get(
                "k6.additive_gen_x_gender", {}).get("false_acceptance_on_unstructured_noise"),
            per_layer=ad,
            provenance="gen (committed 3cadb69) x gender (added at prereg 2026-08-21, "
                       "NOT in 3cadb69), the model prereg 3.2 step 4 registers")

    # ---- V6 supervised -----------------------------------------------------------
    C = res.get("supervised_arm")
    if C and C.get("cells"):
        cells = C["cells"]
        fam = {
            "generation_s2": [c["axes"]["generation"]["s2_loo_ridge"]["p_sampled"] for c in cells],
            "gender_s2": [c["axes"]["gender"]["s2_loo_ridge"]["p_sampled"] for c in cells],
            "lineal_s2": [c["axes"]["lineal"]["s2_loo_ridge"]["p_sampled"] for c in cells],
            "additive": [c["additive_gen_x_gender"]["p_joint_shuffle"] for c in cells],
        }
        bh = {k: benjamini_hochberg(v, alpha=ALPHA) for k, v in fam.items()}
        summ = {}
        for k, v in fam.items():
            summ[k] = dict(n_cells=len(v), n_raw_pass=int(sum(x < ALPHA for x in v)),
                           n_bh_pass=int(sum(bool(z) for z in bh[k])),
                           expected_by_chance=round(ALPHA * len(v), 1), min_p=float(min(v)))
        suffix_idx = [i for i, c in enumerate(cells) if c["readout"] == PRIMARY_READOUT]
        trio = [cells[i]["layer"] for i in suffix_idx
                if bh["generation_s2"][i] and bh["gender_s2"][i] and bh["additive"][i]]
        cond = {k: bool(summ[k]["n_bh_pass"] > summ[k]["expected_by_chance"]
                        and any(bh[k][i] for i in suffix_idx))
                for k in ("generation_s2", "gender_s2", "additive")}
        ok = all(cond.values()) and len(trio) >= 1
        V["V6_supervised"] = dict(
            verdict=("structure_exists_supervised_only" if ok else "clean_negative"),
            verdict_provenance=(
                "the prereg-registered axis estimator is structurally vacuous (p == 1.0 by "
                "construction, stage A), so this verdict rests on the script-time-declared "
                "secondary estimators S1/S2; the gender contrast was added at prereg time "
                "(2026-08-21) and is NOT in commit 3cadb69"),
            primary_readout=PRIMARY_READOUT,
            conditions=cond, families=summ,
            suffix_layers_with_all_three_bh=trio,
            n_suffix_layers=len(suffix_idx),
            registered_estimator_p_is_always_1=bool(
                all(c["axes"]["generation"]["registered"]["p_sampled"] == 1.0 for c in cells)),
            lineal_note="lineality is supervised-only by construction; the blind arm cannot "
                        "test it (prereg 3.2)")
        # closure across the supervised sweep
        cl = [dict(readout=c["readout"], layer=c["layer"],
                   p_closed=c["closure_P3"].get("p_closed"), p_bg=c["closure_P3"].get("p_bg"),
                   passes=bool(c["closure_P3"].get("closure_passes")))
              for c in cells]
        V["V5_closure_P3_supervised"] = dict(
            verdict="prediction_holds" if not any(x["passes"] for x in cl)
            else "prediction_FAILED",
            n_cells=len(cl), n_cells_closing=int(sum(x["passes"] for x in cl)),
            cells_that_close=[x for x in cl if x["passes"]],
            expected_by_chance_at_alpha_05=round(ALPHA * len(cl), 1))
        # the two numbers that must not be misread
        gen_codes = [KINSHIP_GENERATION[KINSHIP.index(w)] for w in C["words_used"]]
        V["readability_aids"] = dict(
            spearman_ceiling_given_generation_ties=spearman_ceiling(gen_codes),
            spearman_ceiling_note="the generation codes are tied (2,2,1,1,1,1,-1,-1,-1,-1,"
                                  "0,0,0), so a tie-free prediction cannot exceed this "
                                  "value; an observed rho equal to it IS perfect ordering",
            at_exhaustive_floor={
                f"{c['readout']}_L{c['layer']}_{cn}": bool(
                    c["axes"][cn]["s2_loo_ridge"].get("p_exact") is not None
                    and abs(c["axes"][cn]["s2_loo_ridge"]["p_exact"]
                            - c["axes"][cn]["s2_loo_ridge"]["exact_floor_p"]) < 1e-12)
                for c in cells if c["readout"] == PRIMARY_READOUT for cn in ("gender", "lineal")},
            at_exhaustive_floor_note="True means the observed score sits at the ceiling of "
                                     "the exhaustively enumerated null: the p-value is the "
                                     "smallest that null can produce and cannot go lower")

    V["V7_overall"] = dict(
        verdict=V.get("V6_supervised", {}).get("verdict", "untestable"),
        why_recovered_blind_is_unreachable="V1: the blind lattice claim needs both "
                                           "contrasts and gender is untestable at k = 6",
        reported_alongside=dict(colocation=V.get("V4_colocation_P1", {}).get("verdict"),
                                closure_blind=V.get("V5_closure_P3_blind", {}).get("verdict"),
                                closure_supervised=V.get("V5_closure_P3_supervised", {}).get("verdict"),
                                blind_generation=V.get("V3_blind_generation", {}).get("verdict")))

    res["verdict"] = V
    p.write_text(json.dumps(res, indent=1))
    print(json.dumps({k: (v.get("verdict") if isinstance(v, dict) else v)
                      for k, v in V.items() if k.startswith("V")}, indent=1))
    print(f"written -> {p}")


if __name__ == "__main__":
    main()
