"""POST-HOC AMENDMENT to the Phase-2 kinship verdicts.  Provenance stated plainly.

prereg section 7 permits a post-hoc addition provided its provenance is stated.  This is
that, and it corrects a verdict rather than adding a search family.

WHAT HAPPENED, in order:
  1. scripts/29b_phase2_kinship_verdict.py froze rule V3 while the run was still in
     stage B at layer 2.  V3 said: blind generation is testable because the SNR-ladder
     attainable floor measured on SYNTHETIC data in stage A was 0.037 < 0.05, so a
     shortfall of passing layers earns `clean_negative`.
  2. The finished run shows that number was optimistic for the REAL member set.  On the
     six kinship type means the exhaustive 90-arrangement null puts the smallest
     attainable p at 0.0667 at eight of the eleven layers -- ABOVE alpha.  At those same
     layers the observed statistic is at its CEILING (Spearman 0.9562, which is exactly
     the maximum a tie-free prediction can score against the tied generation codes
     [1,1,-1,-1,0,0]; i.e. leave-one-out ordering of generation is PERFECT).
  3. A test whose floor exceeds alpha cannot produce a negative.  prereg 4.1(ii) and 4.3
     both say so.  The amended label for blind generation is therefore `untestable`,
     not `clean_negative`.

Both labels are kept in the JSON: `verdict.V3_blind_generation.verdict` is the frozen
rule's output, and `verdict_amendment.blind_generation.amended_verdict` is this
correction, with this file's reasoning attached.  Nothing else is changed.

The amendment also records two diagnostics that the frozen rules did not ask for:
  * how many blind layers sit at the observed statistic's ceiling, per contrast;
  * that the one adequately powered registered blind statistic -- the additive
    gen x gender R^2, attainable floor 0.006 and unstructured-noise false-acceptance
    0.02 -- is negative at every layer.  That is a real negative on the blind LATTICE,
    which is a stronger statement than "untestable" and must not be lost.
"""
import argparse, json, sys, time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from gfdisc.ontologies import KINSHIP, KINSHIP_GENERATION

ALPHA = 0.05
CEIL_TOL = 1e-9


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", default="results/phase2_kinship.json")
    a = ap.parse_args()
    p = REPO / a.path
    res = json.loads(p.read_text())
    B, V = res["blind_arm"], res["verdict"]
    ceiling = V["V3_blind_generation"]["spearman_ceiling_given_ties"]

    gen_rows, gnd_rows = [], []
    for L in B["layers"]:
        g = L["type_mean_axes"]["generation"]["s2_loo_ridge"]
        gen_rows.append(dict(layer=L["layer"], rho=g["observed"], p_exact=g.get("p_exact"),
                             exact_floor_p=g.get("exact_floor_p"),
                             at_ceiling=bool(abs(g["observed"] - ceiling) < CEIL_TOL),
                             p_at_floor=bool(g.get("p_exact") is not None
                                             and abs(g["p_exact"] - g["exact_floor_p"]) < 1e-12)))
        s = L["type_mean_axes"]["gender"]["s2_loo_ridge"]
        gnd_rows.append(dict(layer=L["layer"], auc=s["observed"], p_exact=s.get("p_exact"),
                             exact_floor_p=s.get("exact_floor_p"),
                             at_ceiling=bool(s["observed"] >= 1.0 - CEIL_TOL)))

    n_gen_ceiling = sum(r["at_ceiling"] for r in gen_rows)
    floors = sorted({r["exact_floor_p"] for r in gen_rows if r["exact_floor_p"] is not None})
    ad = V["blind_additive_gen_x_gender"]

    amend = dict(
        provenance=("POST-HOC.  Computed AFTER the results were seen, on 2026-08-21, and "
                    "declared as such per prereg section 7.  It corrects one verdict LABEL; "
                    "it adds no new test, no new search family and no new null."),
        script="scripts/29c_phase2_kinship_amendment.py",
        applied_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
        blind_generation=dict(
            frozen_rule_verdict=V["V3_blind_generation"]["verdict"],
            amended_verdict="untestable",
            reason=("the exhaustive 90-arrangement null's smallest attainable p on the real "
                    f"six-member set is {floors} depending on layer, and is 0.0667 at the "
                    f"{n_gen_ceiling} layers where the observed statistic is at its ceiling. "
                    "A statistic whose floor exceeds alpha = 0.05 cannot produce a p < 0.05 "
                    "even for a perfect signal, so prereg 4.1(ii) and 4.3 make it "
                    "`untestable`; a negative may not be reported from it."),
            why_the_frozen_rule_said_otherwise=(
                "V3 was frozen against stage A's SYNTHETIC attainable floor of 0.0370, "
                "measured on a planted lattice with a different tie pattern in the sampled "
                "1000-draw null.  The real member set's EXACT floor is coarser."),
            spearman_ceiling_given_ties=ceiling,
            ceiling_note=("0.9562 is the largest Spearman a tie-free prediction can reach "
                          "against the tied generation codes [1,1,-1,-1,0,0]; an observed "
                          "rho equal to it is PERFECT leave-one-out ordering, not a "
                          "sub-perfect one"),
            n_layers=len(gen_rows), n_layers_at_ceiling=n_gen_ceiling,
            per_layer=gen_rows),
        blind_gender=dict(
            verdict="untestable",
            status="unchanged -- V1 registered this before any result was seen",
            n_layers_with_perfect_auc=int(sum(r["at_ceiling"] for r in gnd_rows)),
            layers_with_perfect_auc=[r["layer"] for r in gnd_rows if r["at_ceiling"]],
            exact_floor_p=0.1,
            note=("gender AUC reaches 1.000 -- perfect leave-one-out separation -- at the "
                  "deepest blind layers, with the exact 20-arrangement p pinned at its "
                  "0.10 floor.  The signal is visibly present and the registered test "
                  "cannot certify it.  This is the clearest single illustration of why "
                  "prereg 4.1(ii)'s feasibility gate exists."),
            per_layer=gnd_rows),
        blind_lattice_via_the_powered_statistic=dict(
            statistic="additive gen x gender R^2 (prereg 3.2 step 4), joint code-permutation null",
            attainable_floor=ad["attainable_floor"],
            false_acceptance_on_unstructured_noise=ad["false_acceptance_on_unstructured_noise"],
            n_layers=ad["n_layers"], n_layers_p_lt_05=ad["n_p_lt_05"],
            expected_by_chance=ad["expected_by_chance"],
            r2_range=[min(x["r2"] for x in ad["per_layer"]),
                      max(x["r2"] for x in ad["per_layer"])],
            p_range=[min(x["p"] for x in ad["per_layer"]),
                     max(x["p"] for x in ad["per_layer"])],
            verdict="clean_negative",
            why_this_one_may_be_called_a_negative=(
                "unlike the per-axis tests, this statistic IS adequately powered at the "
                "blind arity: a perfect planted lattice reaches p = 0.006 through the "
                "identical code path and unstructured noise fires it at 0.02.  It returns "
                "no lattice at any of the 11 layers."),
            provenance="generation (committed 3cadb69) x gender (added at prereg "
                       "2026-08-21, NOT in 3cadb69)"),
        net_effect=("the blind arm's three registered per-axis tests are untestable "
                    "(lineality by pool floor, gender by arity, generation by the exact "
                    "null's coarseness), while the one powered blind lattice statistic is "
                    "a clean negative; P1 co-location is positive and P3 closure holds. "
                    "The overall Phase-2 kinship verdict "
                    f"({V['V7_overall']['verdict']}) is unchanged."))

    res["verdict_amendment"] = amend
    p.write_text(json.dumps(res, indent=1))
    print(json.dumps({k: (v.get("amended_verdict") or v.get("verdict"))
                      for k, v in amend.items() if isinstance(v, dict)}, indent=1))
    print(f"n_gen_at_ceiling={n_gen_ceiling}/{len(gen_rows)}  "
          f"n_gender_perfect={amend['blind_gender']['n_layers_with_perfect_auc']}/{len(gnd_rows)}")
    print(f"written -> {p}")


if __name__ == "__main__":
    main()
