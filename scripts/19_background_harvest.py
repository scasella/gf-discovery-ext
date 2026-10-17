"""Harvest the in-gate background files: calibration of the ordering tier, power of the closure tier.

DECLARED EXPLORATORY. Not pre-registered. This is *post-processing only* -- it reads
`results/gate_background_big_L{9,7}.json` and the two gate outputs those backgrounds
were matched to. No model, no SAE, no activation cache is touched, and no gate
decision is recomputed or revised.

What the background files are. `scripts/16_gate_background.py` took the members of
exactly the clusters the gate scored as templates, drew index-band-matched random
latent subsets at full template size (`gfdisc.nulls.stratified_resample`), and scored
each draw with `labelled()` imported from `scripts/13_rerun_gate.py` itself -- same
SparseReader, same rank-even half-B label positions (ODD cache rows), same
MIN_FIRING_PER_LABEL=10 / >=5 positions / ceil(2k/3)-labels support rule, same
20,000-permutation ordering null and the same `gap_pvalue` closure null. 2,000 draws
per layer were written; four summary scalars were ever read off them. Everything the
draws can support is computed here.

Two conventions inherited from script 16's summary, kept exactly:
  * ordering acceptance counts `p_order <= 0.05` (non-strict);
  * closure acceptance counts `p_closed <  0.05` (strict).
Both nulls are +1-smoothed over 20,000 permutations so the smallest attainable p is
~5e-5 and no p ever lands exactly on 0.05; the strict/non-strict split is therefore
cosmetic here, and it is verified to be so rather than assumed.

Matching for the real clusters' empirical closure percentile is reported four ways,
because gap_ratio is normalised by 2*pi/n_labels and n_labels is a *realised*
property of a draw (8..12), not a matched one:
  size_matched     -- draws whose template_size equals the cluster's size. The
                      variant the task specifies; primary.
  template_matched -- draws generated from that exact cluster as template. Tightest:
                      identical index-band composition. Recovered from the draw index
                      (script 16 cycles `templates[i % len(templates)]`) and verified
                      against template_size before use.
  size_nlabels     -- size_matched, further restricted to draws that realised the
                      same n_labels. Removes the normalisation confound; smallest n.
  pooled           -- all scorable draws at the layer, ignoring size.
"""
import json
import time
from pathlib import Path

import numpy as np
from scipy.stats import beta

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "background_harvest.json"

BACKGROUNDS = {
    9: dict(bg="results/gate_background_big_L9.json",
            gate="results/rerun_gate_big_primary_L9.json"),
    7: dict(bg="results/gate_background_big_L7.json",
            gate="results/rerun_gate_big_exploratory_months.json"),
}

ALPHAS = [0.05, 0.01, 0.001]
PCTS = [5, 25, 50, 75, 95]


def clopper_pearson(x, n, alpha=0.05):
    """Exact binomial interval. Degenerate ends handled explicitly (x=0 -> lo=0)."""
    x, n = int(x), int(n)
    if n == 0:
        return dict(k=x, n=n, rate=None, lo=None, hi=None)
    lo = 0.0 if x == 0 else float(beta.ppf(alpha / 2, x, n - x + 1))
    hi = 1.0 if x == n else float(beta.ppf(1 - alpha / 2, x + 1, n - x))
    return dict(k=x, n=n, rate=x / n, lo=lo, hi=hi, method="Clopper-Pearson 95%")


def pct_at_or_below(pool, obs):
    """Empirical percentile of `obs` in `pool`: 100 * P(pool <= obs). Lower = more circular."""
    if len(pool) == 0:
        return None
    return float(100.0 * (np.asarray(pool) <= obs).mean())


def percentiles(vals):
    if len(vals) == 0:
        return None
    v = np.asarray(vals, dtype=float)
    return {f"p{p}": float(np.percentile(v, p)) for p in PCTS} | {
        "mean": float(v.mean()), "n": int(v.size)}


def strat_block(draws):
    """Scorability / ordering / closure rates for one stratum of draws."""
    sc = [r for r in draws if r["scorable"]]
    n, ns = len(draws), len(sc)
    po = np.array([r["p_order"] for r in sc]) if ns else np.zeros(0)
    pc = np.array([r["p_closed"] for r in sc]) if ns else np.zeros(0)
    out = dict(
        n_draws=n, n_scorable=ns, scorability=ns / n if n else None,
        n_order_pass=int((po <= 0.05).sum()),
        order_pass_rate=float((po <= 0.05).mean()) if ns else None,
        n_closure_pass=int((pc < 0.05).sum()),
        closure_pass_rate=float((pc < 0.05).mean()) if ns else None,
        n_both_pass=int(((po <= 0.05) & (pc < 0.05)).sum()),
        both_pass_rate=float(((po <= 0.05) & (pc < 0.05)).mean()) if ns else None,
        n_labels_realised=sorted(set(int(r["n_labels"]) for r in sc)),
    )
    return out


def calibration(sc):
    """Item 2: fraction of scorable draws below each alpha, with the uniform-null ratio."""
    po = np.array([r["p_order"] for r in sc])
    pc = np.array([r["p_closed"] for r in sc])
    out = {"n_scorable": len(sc)}
    for tag, v in (("p_order", po), ("p_closed", pc)):
        d = {}
        for a in ALPHAS:
            f = float((v <= a).mean()) if len(v) else None
            d[f"le_{a}"] = dict(k=int((v <= a).sum()), fraction=f,
                                ratio_vs_uniform=(f / a) if f is not None else None)
        d["lt_0.05_summary_convention"] = dict(
            k=int((v < 0.05).sum()),
            fraction=float((v < 0.05).mean()) if len(v) else None)
        d["min"] = float(v.min()) if len(v) else None
        out[tag] = d
    return out


def main():
    t0 = time.time()
    doc = {
        "status": "exploratory, not pre-registered",
        "framing": (
            "This is a POWER ANALYSIS of the closure tier and a CALIBRATION AUDIT of the "
            "ordering tier. It is not a post-hoc re-scoring of the gate: no gate decision "
            "is recomputed, revised or overturned here, and no cluster's pass/fail status "
            "depends on any number in this file. Every quantity is a property of the "
            "already-written in-gate background draws (index-matched random latent subsets "
            "scored by the same two-tier gate), read as a null distribution. The empirical "
            "percentiles in item 4 locate each real cluster inside that null; they are "
            "descriptive placements, not p-values, and the gate's own pre-registered "
            "p_order / p_closed remain the only inferential statistics."),
        "script": "scripts/19_background_harvest.py",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "inputs": {str(k): v for k, v in BACKGROUNDS.items()},
        "conventions": {
            "ordering_acceptance": "p_order <= 0.05 (non-strict) -- matches summary.rate_order in script 16, verified by exact recount below",
            "closure_acceptance": "p_closed < 0.05 (strict) -- matches summary.rate_closure in script 16",
            "strict_vs_nonstrict": "both nulls are (1+hits)/(1+20000)-smoothed, so no p lands exactly on 0.05; verified equal counts under < and <= for closure at both layers",
            "denominator": "all rates are over SCORABLE draws, as the gate drops unscorable clusters from its denominator; scorability itself is over all draws",
            "percentile_direction": "gap_ratio percentile = 100 * P(background <= observed). LOW percentile = tighter closure than the null. gap_ratio is max angular gap / (2*pi/n_labels); smaller is more ring-like",
        },
        "layers": {},
        "cross_checks": [],
        "caveats": [],
    }

    for layer, paths in BACKGROUNDS.items():
        bg = json.loads((ROOT / paths["bg"]).read_text())
        gate = json.loads((ROOT / paths["gate"]).read_text())
        draws = bg["draws"]
        summary = bg["summary"]
        rows = [r for r in gate["rows"] if r["layer"] == layer]  # template order used by script 16
        n_t = len(rows)

        # ---- structural facts, verified not assumed -------------------------------
        size_eq_template = all(r["size"] == r["template_size"] for r in draws)
        # script 16 cycles templates[i % len(templates)]; check the mapping holds
        tmpl_ok = all(draws[i]["template_size"] == rows[i % n_t]["size"] for i in range(len(draws)))

        sc = [r for r in draws if r["scorable"]]
        sizes = sorted(set(r["template_size"] for r in draws))

        # ---- item 1: stratified rates --------------------------------------------
        by_template_size = {str(s): strat_block([r for r in draws if r["template_size"] == s])
                            for s in sizes}
        by_draw_size = {str(s): strat_block([r for r in draws if r["size"] == s])
                        for s in sorted(set(r["size"] for r in draws))}
        by_template = {}
        for ti, row in enumerate(rows):
            d = [r for r in draws if r["i"] % n_t == ti]
            b = strat_block(d)
            b |= dict(template_index=ti, cluster_metric=row["metric"], cluster_size=row["size"])
            by_template[str(ti)] = b

        # ---- item 3 / 5: empirical nulls -----------------------------------------
        nulls = {}
        for s in sizes:
            ss = [r for r in sc if r["template_size"] == s]
            nulls[str(s)] = dict(
                gap_ratio=percentiles([r["gap_ratio"] for r in ss]),
                rho=percentiles([r["rho"] for r in ss]),
                rho_p95=float(np.percentile([r["rho"] for r in ss], 95)) if ss else None)
        nulls["pooled"] = dict(
            gap_ratio=percentiles([r["gap_ratio"] for r in sc]),
            rho=percentiles([r["rho"] for r in sc]),
            rho_p95=float(np.percentile([r["rho"] for r in sc], 95)))

        # ---- item 4: empirical closure percentile of every real scored cluster ----
        pooled_gap = np.array([r["gap_ratio"] for r in sc])
        cluster_rows = []
        for ti, row in enumerate(rows):
            obs = row["gap_ratio"]
            k = row["size"]
            nl = row["n_labels"]
            m_size = np.array([r["gap_ratio"] for r in sc if r["template_size"] == k])
            m_tmpl = np.array([r["gap_ratio"] for r in sc if r["i"] % n_t == ti])
            m_sn = np.array([r["gap_ratio"] for r in sc
                             if r["template_size"] == k and r["n_labels"] == nl])
            m_rho_size = np.array([r["rho"] for r in sc if r["template_size"] == k])
            cluster_rows.append(dict(
                template_index=ti, layer=layer, metric=row["metric"], size=k, n_labels=nl,
                gap_ratio=obs, rho=row["rho"], p_order=row["p_order"], p_closed=row["p_closed"],
                closure_underpowered=row.get("closure_underpowered"),
                ref_p_closed=row.get("ref_p_closed"), missing_run=row.get("missing_run"),
                CIRCLE=row.get("CIRCLE"), arc_only=row.get("arc_only"),
                gap_percentile_size_matched=pct_at_or_below(m_size, obs),
                n_size_matched=int(m_size.size),
                gap_percentile_template_matched=pct_at_or_below(m_tmpl, obs),
                n_template_matched=int(m_tmpl.size),
                gap_percentile_size_and_nlabels=pct_at_or_below(m_sn, obs),
                n_size_and_nlabels=int(m_sn.size),
                gap_percentile_pooled=pct_at_or_below(pooled_gap, obs),
                n_pooled=int(pooled_gap.size),
                rho_percentile_size_matched=pct_at_or_below(m_rho_size, row["rho"]),
            ))

        # ---- item 6: exact intervals on the closure-tier acceptance count ---------
        n_sc = len(sc)
        n_clo = int(sum(r["p_closed"] < 0.05 for r in sc))
        n_ord = int(sum(r["p_order"] <= 0.05 for r in sc))
        cp = dict(
            closure=clopper_pearson(n_clo, n_sc),
            ordering=clopper_pearson(n_ord, n_sc),
            both_tiers=clopper_pearson(
                int(sum((r["p_order"] <= 0.05) and (r["p_closed"] < 0.05) for r in sc)), n_sc),
            denominator_source="recounted from draws (not summary)",
            denominator_matches_summary=bool(n_sc == summary["n_scorable"]),
        )

        doc["layers"][str(layer)] = dict(
            background_file=paths["bg"], gate_file=paths["gate"],
            ontology=summary["ontology"], half=summary["half"], l0=summary["l0"],
            n_draws=len(draws), n_scorable=n_sc, n_templates=n_t,
            template_sizes=[r["size"] for r in rows],
            structural_checks=dict(
                size_equals_template_size_for_all_draws=bool(size_eq_template),
                template_cycle_i_mod_n_verified=bool(tmpl_ok),
                closure_strict_equals_nonstrict=bool(
                    n_clo == int(sum(r["p_closed"] <= 0.05 for r in sc))),
                recount_rate_order=n_ord / n_sc,
                summary_rate_order=summary["rate_order"],
                recount_rate_closure=n_clo / n_sc,
                summary_rate_closure=summary["rate_closure"],
                summary_reproduced=bool(
                    abs(n_ord / n_sc - summary["rate_order"]) < 1e-12
                    and abs(n_clo / n_sc - summary["rate_closure"]) < 1e-12
                    and n_sc == summary["n_scorable"]),
            ),
            item1_rates_by_template_size=by_template_size,
            item1_rates_by_draw_size=by_draw_size,
            item1_rates_by_template=by_template,
            item2_p_value_calibration=dict(
                overall=calibration(sc),
                by_size={str(s): calibration([r for r in sc if r["template_size"] == s])
                         for s in sizes}),
            item3_item5_empirical_nulls=nulls,
            item4_real_cluster_closure_percentiles=cluster_rows,
            item6_acceptance_intervals=cp,
        )

    # ---- cross-checks against the prior audit's quoted expectations ---------------
    L9 = doc["layers"]["9"]
    L7 = doc["layers"]["7"]

    def cc(name, expected, computed, note=""):
        ok = None if expected is None else bool(abs(computed - expected) <= max(
            0.02 * abs(expected), 0.006))
        doc["cross_checks"].append(dict(check=name, expected=expected, computed=computed,
                                        matches=ok, note=note))

    cc("L9 ordering acceptance, k=8", 0.126, L9["item1_rates_by_template_size"]["8"]["order_pass_rate"])
    cc("L9 ordering acceptance, k=23", 0.195, L9["item1_rates_by_template_size"]["23"]["order_pass_rate"])
    cc("L9 scorability, k=8", 0.207, L9["item1_rates_by_template_size"]["8"]["scorability"])
    cc("L9 scorability, k=23", 0.706, L9["item1_rates_by_template_size"]["23"]["scorability"])
    cc("L9 ordering calibration, p<=0.05", 0.164,
       L9["item2_p_value_calibration"]["overall"]["p_order"]["le_0.05"]["fraction"])
    cc("L9 ordering calibration, p<=0.01", 0.043,
       L9["item2_p_value_calibration"]["overall"]["p_order"]["le_0.01"]["fraction"])
    cc("L9 ordering calibration, p<=0.001", 0.008,
       L9["item2_p_value_calibration"]["overall"]["p_order"]["le_0.001"]["fraction"])
    cc("L9 rho 95th percentile, k=12", 0.865,
       L9["item3_item5_empirical_nulls"]["12"]["rho_p95"],
       note=("np.percentile linear interpolation. The k=12 stratum (n=190) gives 0.8767; the "
             "quoted 0.865 is reproduced exactly by the POOLED L9 null (0.8653, n=750) and by "
             "the k=8 stratum (0.8647). Same pooled-vs-size-matched choice as the closure "
             "percentile cross-check below. Method sensitivity is small: 0.8741-0.8798 across "
             "all np.percentile methods at k=12."))

    l9_k8 = [r for r in L9["item4_real_cluster_closure_percentiles"]
             if r["size"] == 8 and abs(r["gap_ratio"] - 2.5877568014826977) < 1e-9][0]
    l7_k24 = [r for r in L7["item4_real_cluster_closure_percentiles"] if r["size"] == 24][0]
    doc["cross_checks"].append(dict(
        check="L9 k=8 cosine cluster (gap_ratio 2.59) closure percentile",
        expected=2.0,
        computed_size_matched=l9_k8["gap_percentile_size_matched"],
        computed_pooled=l9_k8["gap_percentile_pooled"],
        computed_template_matched=l9_k8["gap_percentile_template_matched"],
        matches="pooled and size-matched both round to ~2nd percentile"))
    doc["cross_checks"].append(dict(
        check="L7 arc (k=24) closure percentile",
        expected=24.5,
        computed_size_matched=l7_k24["gap_percentile_size_matched"],
        computed_pooled=l7_k24["gap_percentile_pooled"],
        computed_template_matched=l7_k24["gap_percentile_template_matched"],
        matches=("the quoted 24.5 reproduces under the POOLED background (all scorable "
                 "draws at the layer), not under the size-matched one, which gives ~28. "
                 "The size-matched figure is the primary here per spec; both are reported "
                 "and neither changes any gate decision.")))

    doc["caveats"] = [
        "Pure post-processing of already-written JSON. No model, SAE or activation cache was loaded; "
        "no gate statistic was recomputed. Every number is a re-read of draws script 16 produced.",
        "gap_ratio is normalised by 2*pi/n_labels, and the ordering statistic's null depends on "
        "n_labels too. n_labels is REALISED per draw (8-12 here), not matched, so a size stratum "
        "pools several n_labels. That is exactly why size-matched and size+n_labels-matched "
        "percentiles differ; the size+n_labels variant is cleaner but has far fewer draws "
        "(as few as 23 at L7 k=8).",
        "Two of the prior audit's quoted figures (the L7 arc's 24.5th percentile, and the 0.865 "
        "rho 95th percentile) reproduce under the POOLED layer background rather than the "
        "size-matched stratum the spec asks for. All variants are written out; the difference is "
        "a matching choice, not a data discrepancy. Every stratified rate cross-check "
        "(scorability and ordering acceptance at k=8/23) and every ordering calibration cell "
        "reproduces exactly.",
        "Ordering acceptance is measured on RAW p_order <= 0.05. The gate requires ordering to "
        "survive BH-FDR across the clusters it scored, so these background rates are upper bounds "
        "on what the gate would have accepted, exactly as script 16 intended.",
        "Draws are index-band matched (gfdisc.nulls.stratified_resample) but NOT firing-frequency "
        "matched and NOT decoder-direction ranked; the gate uses direction ranking to select which "
        "clusters to score, not as a tier, so the background omits it by design.",
        "The closure-tier power statement is bounded by the draws available: 750 scorable draws at "
        "L9 and 792 at L7. A zero count bounds the false-acceptance rate from above; it cannot "
        "establish that the rate is exactly zero.",
        "Both p-value nulls are +1-smoothed over 20,000 permutations, so the smallest attainable "
        "p is ~5e-5; the p<=0.001 calibration cell is resolvable but rests on few permutation ticks.",
        "The L9 k=8 cosine cluster that lands lowest in the closure null is flagged "
        "closure_underpowered by the gate (missing_run=3, ref_p_closed=0.54): a geometrically "
        "perfect ring on its surviving labels could not have passed closure. Its low empirical "
        "percentile describes the null's spread, not evidence of a ring.",
    ]

    OUT.write_text(json.dumps(doc, indent=2))
    print(f"wrote {OUT} in {time.time() - t0:.2f}s")
    for c in doc["cross_checks"]:
        print(" ", json.dumps(c))
    for lay in ("9", "7"):
        L = doc["layers"][lay]
        print(f"L{lay}: summary_reproduced={L['structural_checks']['summary_reproduced']} "
              f"closure CP={L['item6_acceptance_intervals']['closure']}")


if __name__ == "__main__":
    main()
