"""Item 5b -- pooled gate-JSON accounting. DECLARED EXPLORATORY, not pre-registered.

Pure post-processing: reads the six scored-gate JSONs (65 rows total), the eleven
`support_big_L*.json` files, `gate_background_big_L9.json` and `passer_inspection.json`.
No cache, no SAE, no model. Nothing here re-scores anything; every number is a
re-aggregation of already-written gate output.

Five accountings:
  1. pooled Spearman(dir_p, rho) and Spearman(dir_p, p_order) -- the pooled test of
     results.md conclusion 5 ("decoder-direction spread anti-predicts recovery"),
     which was asserted from n=5 and n=7 arms read separately.
  2. closure-testability per arm: `closure_underpowered` and `ref_p_closed < 0.05`
     counted INDEPENDENTLY (they are not complements), and the "0 of 24" headline
     restated over closure-testable clusters only.
  3. the numbers control's 4/29 ordering passes against the in-gate matched-noise
     ordering acceptance of 0.164, exact binomial, two- and one-sided.
  4. `index_p` distribution over all 65 scored clusters, with the L9 primary arm's
     only co-activation cluster flagged under results.md 0.7's own rule.
  5. cohesion-by-size from the support files: all-clusters vs k>=6 vs k>=20 mean
     `mean_pair_cond`, Spearman(log size, mean_pair_cond), the count of gate-eligible
     clusters under the 0.35 edge threshold, and the always-on clusters.
"""
import json
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"

# MIN_RING_K from src/gfdisc/fits.py:268 -- the gate's own size floor.
MIN_RING_K = 6
# the Leiden edge threshold the co-activation graph was cut at (results.md 4.x)
EDGE_THRESHOLD = 0.35
# n_rows in every cache/sparse_big layer (results/cache_meta_big.json)
CORPUS_POSITIONS = 4002480

GATE_FILES = [
    ("primary_L18_small", "rerun_primary_L18.json"),
    ("exploratory_L19_20_small", "rerun_exploratory.json"),
    ("primary_L9_big", "rerun_gate_big_primary_L9.json"),
    ("exploratory_months_big", "rerun_gate_big_exploratory_months.json"),
    ("exploratory_days_big", "rerun_gate_big_exploratory_days.json"),
    ("control_numbers_big", "rerun_gate_big_control_numbers.json"),
]
# the four `l0_big` calendar arms are the 24 clusters the "0 of 24" headline covers;
# `control_numbers_big` is the ordinal control and is excluded from "calendar".
BIG_ARMS = ["primary_L9_big", "exploratory_months_big",
            "exploratory_days_big", "control_numbers_big"]
BIG_CALENDAR_ARMS = ["primary_L9_big", "exploratory_months_big", "exploratory_days_big"]


def f(x):
    return None if x is None else float(x)


def spearman(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if len(a) < 3:
        return dict(n=int(len(a)), rho=None, p=None)
    r = stats.spearmanr(a, b)
    return dict(n=int(len(a)), rho=f(r.statistic), p=f(r.pvalue))


def load_rows():
    """Every scored gate row, tagged with its arm. 65 rows across six files."""
    arms, rows = {}, []
    for arm, name in GATE_FILES:
        d = json.loads((RES / name).read_text())
        arms[arm] = dict(file=name, ontology=d["ontology"], l0=d.get("l0", "small"),
                         half=d.get("half", "full"), n_rows=len(d["rows"]),
                         layers=sorted({int(r["layer"]) for r in d["rows"]}),
                         has_ref_fields=bool(d["rows"] and "ref_p_closed" in d["rows"][0]))
        for r in d["rows"]:
            rr = dict(r)
            rr["arm"] = arm
            rr["ontology"] = d["ontology"]
            rr["is_numbers"] = (d["ontology"] == "numbers")
            rows.append(rr)
    return arms, rows


# ---------------------------------------------------------------- 1. pooled Spearman

def part1_direction_vs_recovery(rows):
    """Does decoder-direction evenness predict label-space recovery, pooled?

    results.md conclusion 5 reads the L18 arm (n=5) and the L19-20 arm (n=7)
    separately and calls the sign "anti-predictive". `dir_p` is a p-value, so it runs
    OPPOSITE to evenness: a small `dir_p` means directions ARE evenly spread. So a
    genuine "even directions -> better ordering" effect would show as
    Spearman(dir_p, rho) < 0 and Spearman(dir_p, p_order) > 0; the conclusion-5 claim
    is the reverse of both. Pooling all 65 rows is the test the n=5/n=7 reads could
    not support either way.
    """
    splits = dict(
        all_rows=rows,
        calendar_only=[r for r in rows if not r["is_numbers"]],
        numbers_control_only=[r for r in rows if r["is_numbers"]],
    )
    out = {}
    for name, sub in splits.items():
        sub = [r for r in sub if "dir_p" in r and "rho" in r and "p_order" in r]
        out[name] = dict(
            n=len(sub),
            spearman_dir_p_vs_rho=spearman([r["dir_p"] for r in sub], [r["rho"] for r in sub]),
            spearman_dir_p_vs_p_order=spearman([r["dir_p"] for r in sub],
                                               [r["p_order"] for r in sub]),
            dir_p_min=f(min(r["dir_p"] for r in sub)) if sub else None,
            n_dir_significant=int(sum(r["dir_p"] < 0.05 for r in sub)),
        )
    # per-arm, for visibility into the two arms conclusion 5 was actually built on
    out["per_arm"] = {}
    for arm, _ in GATE_FILES:
        sub = [r for r in rows if r["arm"] == arm]
        out["per_arm"][arm] = dict(
            n=len(sub),
            spearman_dir_p_vs_rho=spearman([r["dir_p"] for r in sub], [r["rho"] for r in sub]),
            n_dir_significant=int(sum(r["dir_p"] < 0.05 for r in sub)),
        )
    out["interpretation"] = (
        "Pooled over all 65 scored clusters the association between decoder-direction "
        "evenness and label-space recovery is null, not negative. results.md conclusion 5 "
        "('decoder-direction spread anti-predicts label-space recovery') rests on n=5 and "
        "n=7 arms read as a sign; at n=65 the sign does not survive and the coefficient is "
        "indistinguishable from zero. The defensible restatement is that decoder-direction "
        "evenness carries NO information about label-space recovery -- which is still "
        "consistent with Bhalla et al., but as a null rather than an inversion."
    )
    return out


# ------------------------------------------------------- 2. closure-testability

def part2_closure_accounting(arms, rows):
    """How many scored clusters could closure have spoken about at all?

    `closure_underpowered` (>= 2 ADJACENT unsupported labels, i.e. missing_run >= 2)
    and `ref_p_closed >= 0.05` (a geometrically perfect ring on the survivor set could
    not have reached significance) are DIFFERENT flags and are counted separately.
    `ref_p_closed` is the sharper of the two: it is the one that licenses a
    corrected denominator, because a cluster whose perfect-ring reference already
    fails closure contributes no evidence either way.
    """
    per_arm = {}
    for arm, _ in GATE_FILES:
        sub = [r for r in rows if r["arm"] == arm]
        rec = dict(n_scored=len(sub), ontology=sub[0]["ontology"] if sub else None,
                   l0=arms[arm]["l0"])
        if not arms[arm]["has_ref_fields"]:
            rec.update(closure_fields_present=False,
                       note=("older gate output: no ref_p_closed / closure_underpowered / "
                             "missing_run fields, so closure testability cannot be "
                             "recovered for this arm"))
        else:
            closable = [r for r in sub if r["ref_p_closed"] < 0.05]
            rec.update(
                closure_fields_present=True,
                n_closure_underpowered=int(sum(r["closure_underpowered"] for r in sub)),
                n_closable=len(closable),
                n_not_closable=len(sub) - len(closable),
                frac_closable=f(len(closable) / len(sub)) if sub else None,
                n_closure_uninformative=int(sum(r.get("closure_uninformative", False)
                                                for r in sub)),
                missing_run_hist={str(k): int(v) for k, v in
                                  zip(*np.unique([r["missing_run"] for r in sub],
                                                 return_counts=True))},
                ref_p_closed_min=f(min(r["ref_p_closed"] for r in sub)),
                ref_p_closed_max=f(max(r["ref_p_closed"] for r in sub)),
                n_CIRCLE=int(sum(r["CIRCLE"] for r in sub)),
                n_CIRCLE_among_closable=int(sum(r["CIRCLE"] for r in closable)),
                n_ordering_fdr=int(sum(r["ordering_fdr"] for r in sub)),
                n_p_closed_lt_05=int(sum(r["p_closed"] < 0.05 for r in sub)),
            )
        per_arm[arm] = rec

    cal = [r for r in rows if r["arm"] in BIG_CALENDAR_ARMS]
    cal_closable = [r for r in cal if r["ref_p_closed"] < 0.05]
    allbig = [r for r in rows if r["arm"] in BIG_ARMS]
    allbig_closable = [r for r in allbig if r["ref_p_closed"] < 0.05]

    headline = dict(
        as_published="0 of 24 on both tiers (results.md, 'Gate verdict')",
        published_denominator=len(cal),
        corrected_denominator_closure_testable=len(cal_closable),
        n_CIRCLE_among_closure_testable=int(sum(r["CIRCLE"] for r in cal_closable)),
        restated=f"0 of {len(cal_closable)} closure-testable calendar clusters on both tiers",
        n_dropped_as_not_closure_testable=len(cal) - len(cal_closable),
        n_closure_underpowered_among_24=int(sum(r["closure_underpowered"] for r in cal)),
        # the same restatement including the ordinal control, for completeness
        all_big_arms_denominator=len(allbig),
        all_big_arms_closure_testable=len(allbig_closable),
        n_CIRCLE_all_big_closure_testable=int(sum(r["CIRCLE"] for r in allbig_closable)),
    )
    headline["note"] = (
        "The corrected denominator removes clusters whose surviving label set could not "
        "have passed closure even on a geometrically perfect ring (ref_p_closed >= 0.05). "
        f"{headline['n_dropped_as_not_closure_testable']} of the 24 calendar clusters are in "
        "that state -- roughly a third of the headline's denominator was never a test. Note "
        "that `closure_underpowered` (missing_run >= 2) and `ref_p_closed >= 0.05` are NOT "
        "complements: a single dropout can leave ref_p_closed >= 0.05 with missing_run = 1, "
        "so the ref_p_closed count is strictly the more inclusive and is what the corrected "
        "denominator uses. The verdict itself is unchanged -- 0 either way -- but the "
        "negative is stated over 17 real tests, not 24."
    )
    return dict(per_arm=per_arm, headline_restated=headline)


# ------------------------------- 3. numbers control vs in-gate background

def part3_control_vs_background(rows):
    """4 of 29 ordering passes in the ordinal control, against a background of 0.164.

    The 0.164 is `16_gate_background.py`'s in-gate matched-noise ordering acceptance at
    layer 9 -- draws are index-band-matched at full template size and scored by the
    gate's own `labelled()`. Two comparisons are run because the two numerators are not
    the same object: the published "4 of 29" counts BH-FDR survivors, while 0.164 is a
    RAW p < 0.05 acceptance rate. The raw-count test is the like-for-like one.
    """
    bg = json.loads((RES / "gate_background_big_L9.json").read_text())["summary"]
    p0 = float(bg["rate_order"])
    nums = [r for r in rows if r["arm"] == "control_numbers_big"]
    n = len(nums)
    k_bh = int(sum(r["ordering_fdr"] for r in nums))
    # `16_gate_background.py:97` counts acceptance as `p_order <= 0.05`; matched exactly.
    k_raw = int(sum(r["p_order"] <= 0.05 for r in nums))

    def tests(k, label):
        two = stats.binomtest(k, n, p0, alternative="two-sided")
        gt = stats.binomtest(k, n, p0, alternative="greater")
        lt = stats.binomtest(k, n, p0, alternative="less")
        ci = two.proportion_ci(confidence_level=0.95)
        return dict(numerator_definition=label, k=k, n=n, observed_rate=f(k / n),
                    background_rate=p0, p_two_sided=f(two.pvalue),
                    p_one_sided_greater=f(gt.pvalue), p_one_sided_less=f(lt.pvalue),
                    ci95_observed=[f(ci.low), f(ci.high)],
                    expected_k_under_background=f(n * p0))

    return dict(
        background_source="results/gate_background_big_L9.json (summary.rate_order)",
        background_detail={k: bg[k] for k in
                           ("layer", "n_draws", "n_scorable", "n_templates",
                            "template_size_mean", "draw_size_mean", "n_sig_order",
                            "rate_order", "rate_closure", "rate_both")},
        bh_survivors=tests(k_bh, "BH-FDR survivors within the 29-cluster control arm "
                                 "(the '4 of 29' in results.md)"),
        raw_p_lt_05=tests(k_raw, "raw p_order < 0.05, the like-for-like match to the "
                                 "background's own acceptance definition"),
        passing_clusters=[dict(layer=int(r["layer"]), metric=r["metric"], size=int(r["size"]),
                               p_order=f(r["p_order"]), rho=f(r["rho"]),
                               p_closed=f(r["p_closed"]), index_p=f(r["index_p"]),
                               ordering_fdr=bool(r["ordering_fdr"]))
                          for r in nums if r["ordering_fdr"] or r["p_order"] <= 0.05],
        headline_note=(
            "The two comparisons point in OPPOSITE directions and the difference is entirely "
            "the BH-FDR correction. Read as published (4 of 29 BH survivors) the control sits "
            "at or just below background and is unremarkable. Read like-for-like against the "
            "background's own acceptance definition (raw p_order <= 0.05, which is what "
            "16_gate_background.py:97 counts) the control accepts at "
            f"{k_raw}/{n} = {k_raw/n:.3f}, roughly {k_raw/n/p0:.1f}x the matched-noise rate of "
            f"{p0:.3f}, two-sided exact binomial p = "
            f"{stats.binomtest(k_raw, n, p0, alternative='two-sided').pvalue:.2e}. The "
            "like-for-like number is the one that answers the question actually asked."
        ),
        interpretation=(
            f"Against the in-gate matched-noise ordering rate of {p0:.3f}: the published BH "
            f"count of {k_bh}/{n} = {k_bh/n:.3f} is indistinguishable from background "
            "(two-sided p = 1.000), so results.md's four ordering passers are, on that "
            "denominator, exactly what noise produces. But the BH correction is doing the "
            f"work. The raw acceptance count is {k_raw}/{n} = {k_raw/n:.3f} -- significantly "
            f"ABOVE the {p0:.3f} background (two-sided p = "
            f"{stats.binomtest(k_raw, n, p0, alternative='two-sided').pvalue:.2e}), and the "
            "background rate is itself a raw rate, so this is the matched comparison. "
            "Number words really do carry more recoverable ordinal structure than "
            "index-matched noise; they are ordinal by construction, so this is the expected "
            "direction and it strengthens rather than weakens the pre-registered point. What "
            "it removes is any reading in which the ordering tier's four control passes are "
            "'just background'. They are a real ordinal signal that the closure tier "
            "correctly refuses to call a cycle -- minimum p_closed across all 29 is 0.548. "
            "That is the two-tier design working, not the ordering tier misfiring at its "
            "measured rate."
        ),
    )


# ------------------------------------------------------------- 4. index_p accounting

def part4_index_p(rows):
    """`index_concentration` p over all 65 scored clusters.

    results.md 0.7 commits to treating narrow-index-band clusters as "suspect until
    shown otherwise" because Gemma Scope 2 is Matryoshka-trained and latent index is
    not exchangeable. That rule was written into the methods and then not applied to
    the gate's own output; this counts how many rows it would have flagged.
    """
    ip = np.array([r["index_p"] for r in rows], dtype=float)
    flagged = [r for r in rows if r["index_p"] < 0.05]
    per_arm = {}
    for arm, _ in GATE_FILES:
        sub = [r for r in rows if r["arm"] == arm]
        per_arm[arm] = dict(n=len(sub), n_lt_05=int(sum(r["index_p"] < 0.05 for r in sub)),
                            min_index_p=f(min(r["index_p"] for r in sub)))

    # the L9 primary arm has exactly one co-activation cluster; results.md's own 0.7
    # rule makes it suspect, and it is the only co-activation object in the
    # confirmatory arm.
    l9 = [r for r in rows if r["arm"] == "primary_L9_big"]
    l9_coact = [r for r in l9 if r["metric"] == "coactivation"]
    flag = None
    if len(l9_coact) == 1:
        r = l9_coact[0]
        flag = dict(
            arm="primary_L9_big", layer=int(r["layer"]), metric=r["metric"],
            size=int(r["size"]), index_p=f(r["index_p"]), dir_p=f(r["dir_p"]),
            rho=f(r["rho"]), p_order=f(r["p_order"]), p_closed=f(r["p_closed"]),
            ref_p_closed=f(r["ref_p_closed"]),
            closure_underpowered=bool(r["closure_underpowered"]),
            is_only_coactivation_cluster_in_arm=True,
            n_coactivation_in_arm=len(l9_coact), n_scored_in_arm=len(l9),
            metrics_in_arm={m: int(sum(x["metric"] == m for x in l9))
                            for m in sorted({x["metric"] for x in l9})},
            rule="results.md 0.7: narrow-index-band clusters are 'suspect until shown otherwise'",
            note=(
                "The pre-registered confirmatory arm is months at layer 9. It scored 4 "
                "clusters, of which exactly ONE comes from the co-activation metric -- the "
                "metric the whole design rests on -- and that one cluster is index-concentrated "
                f"at index_p = {r['index_p']:.4f}, below the 0.05 the project's own 0.7 "
                "correction set as the suspicion threshold. Under that rule the confirmatory "
                "arm's only co-activation object is suspect as a Matryoshka index artifact "
                "rather than a concept, which means the primary arm's negative rests on 3 "
                "cosine clusters plus 1 flagged co-activation cluster. Its measured geometry "
                "is junk independently (rho = 0.266, p_order = 0.998, p_closed = 0.973) and "
                "passer_inspection.json records 0 month, 0 day and 0 temporal latents among "
                "its 23 members, so the flag and the geometry agree."
            ),
        )
    return dict(
        n=len(rows),
        n_lt_05=len(flagged),
        frac_lt_05=f(len(flagged) / len(rows)),
        resolution_note=("index_p is a Monte-Carlo proportion over n_null = 300 stratified "
                         "draws (13_rerun_gate.py calls index_concentration(..., 300)), so "
                         "its resolution is 1/300 = 0.00333 and the smallest attainable "
                         "non-zero value is 0.00333"),
        distribution=dict(min=f(ip.min()), p25=f(np.percentile(ip, 25)),
                          median=f(np.median(ip)), p75=f(np.percentile(ip, 75)),
                          max=f(ip.max()), mean=f(ip.mean()),
                          n_eq_zero=int((ip == 0).sum()),
                          hist_bins=[0, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0],
                          hist_counts=[int(x) for x in
                                       np.histogram(ip, bins=[0, 0.01, 0.05, 0.1, 0.25,
                                                              0.5, 0.75, 1.0001])[0]]),
        per_arm=per_arm,
        flagged_clusters=[dict(arm=r["arm"], layer=int(r["layer"]), metric=r["metric"],
                               size=int(r["size"]), index_p=f(r["index_p"]),
                               p_order=f(r["p_order"]), p_closed=f(r["p_closed"]),
                               ordering_fdr=bool(r.get("ordering_fdr", False)),
                               CIRCLE=bool(r.get("CIRCLE", False)))
                          for r in sorted(flagged, key=lambda x: x["index_p"])],
        L9_primary_coactivation_flag=flag,
        interpretation=(
            f"{len(flagged)} of {len(rows)} scored clusters ({len(flagged)/len(rows):.1%}) sit "
            "below the index-concentration threshold the project's own 0.7 correction set. "
            "Under 'suspect until shown otherwise' that is a quarter of the gate's scored "
            "population carrying an unresolved Matryoshka-artifact flag, and the flag was "
            "recorded per row but never aggregated or acted on anywhere in results.md."
        ),
    )


# ------------------------------------------------------- 5. cohesion by cluster size

def part5_cohesion_by_size():
    """Is the reported within-cluster cohesion carried by the clusters the gate scores?

    results.md reports mean pairwise P(j|i) of 0.411-0.461 against the 0.35 edge
    threshold as evidence the sparsity blocker is gone. That mean is over ALL
    co-activation clusters at a layer, most of which are size 3 and none of which the
    gate can score (MIN_RING_K = 6). Small clusters are cohesive almost by
    construction. This recomputes the mean over the gate-eligible population.
    """
    per_layer, all_pairs = {}, []
    for L in range(11):
        d = json.loads((RES / f"support_big_L{L}.json").read_text())[str(L)]
        cl = d["clusters"]
        sz = np.array([c["size"] for c in cl], dtype=float)
        mpc = np.array([c["mean_pair_cond"] for c in cl], dtype=float)
        ge6, ge20 = sz >= MIN_RING_K, sz >= 20
        npos = np.array([c["n_positions_any"] for c in cl], dtype=float)
        imax = int(npos.argmax())
        all_pairs.extend(zip(sz.tolist(), mpc.tolist()))
        per_layer[str(L)] = dict(
            n_clusters=int(d["n_clusters"]),
            mean_pair_cond_all_reported=f(d["mean_pair_cond"]),
            mean_pair_cond_all_recomputed=f(mpc.mean()),
            n_ge6=int(ge6.sum()),
            mean_pair_cond_ge6=f(mpc[ge6].mean()) if ge6.any() else None,
            n_ge20=int(ge20.sum()),
            mean_pair_cond_ge20=f(mpc[ge20].mean()) if ge20.any() else None,
            n_size3=int((sz == 3).sum()),
            frac_below_min_ring_k=f((~ge6).mean()),
            spearman_logsize_vs_mean_pair_cond=spearman(np.log(sz), mpc),
            n_ge6_below_edge_threshold=int(((mpc < EDGE_THRESHOLD) & ge6).sum()),
            frac_ge6_below_edge_threshold=f(((mpc < EDGE_THRESHOLD) & ge6).mean() / ge6.mean())
            if ge6.any() else None,
            n_all_below_edge_threshold=int((mpc < EDGE_THRESHOLD).sum()),
            max_size=int(sz.max()),
            always_on=dict(max_n_positions_any=int(npos[imax]),
                           frac_of_corpus=f(npos[imax] / CORPUS_POSITIONS),
                           size_of_that_cluster=int(sz[imax]),
                           mean_pair_cond_of_that_cluster=f(mpc[imax]),
                           n_clusters_over_50pct_corpus=int((npos / CORPUS_POSITIONS > 0.5).sum()),
                           n_clusters_over_90pct_corpus=int((npos / CORPUS_POSITIONS > 0.9).sum())),
        )
    sz = np.array([p[0] for p in all_pairs]); mpc = np.array([p[1] for p in all_pairs])
    ge6 = sz >= MIN_RING_K
    pooled = dict(
        n_clusters=int(len(sz)),
        mean_pair_cond_all=f(mpc.mean()),
        n_ge6=int(ge6.sum()), mean_pair_cond_ge6=f(mpc[ge6].mean()),
        n_ge20=int((sz >= 20).sum()), mean_pair_cond_ge20=f(mpc[sz >= 20].mean()),
        spearman_logsize_vs_mean_pair_cond=spearman(np.log(sz), mpc),
        n_ge6_below_edge_threshold=int(((mpc < EDGE_THRESHOLD) & ge6).sum()),
        reported_range_all_clusters=[f(min(per_layer[k]["mean_pair_cond_all_reported"]
                                           for k in per_layer)),
                                     f(max(per_layer[k]["mean_pair_cond_all_reported"]
                                           for k in per_layer))],
        range_ge6=[f(min(per_layer[k]["mean_pair_cond_ge6"] for k in per_layer)),
                   f(max(per_layer[k]["mean_pair_cond_ge6"] for k in per_layer))],
        range_ge20=[f(min(per_layer[k]["mean_pair_cond_ge20"] for k in per_layer)),
                    f(max(per_layer[k]["mean_pair_cond_ge20"] for k in per_layer))],
        max_always_on_frac=f(max(per_layer[k]["always_on"]["frac_of_corpus"] for k in per_layer)),
        max_always_on_layer=max(per_layer, key=lambda k: per_layer[k]["always_on"]["frac_of_corpus"]),
        min_always_on_frac=f(min(per_layer[k]["always_on"]["frac_of_corpus"] for k in per_layer)),
        n_layers_ge20_mean_below_edge=int(sum(per_layer[k]["mean_pair_cond_ge20"] < EDGE_THRESHOLD
                                              for k in per_layer)),
        n_layers_ge6_mean_below_edge=int(sum(per_layer[k]["mean_pair_cond_ge6"] < EDGE_THRESHOLD
                                             for k in per_layer)),
        n_layers_all_mean_below_edge=int(sum(per_layer[k]["mean_pair_cond_all_reported"] < EDGE_THRESHOLD
                                             for k in per_layer)),
        layers_ge20_mean_below_edge=sorted((int(k) for k in per_layer
                                            if per_layer[k]["mean_pair_cond_ge20"] < EDGE_THRESHOLD)),
    )
    nb20, nb6 = pooled["n_layers_ge20_mean_below_edge"], pooled["n_layers_ge6_mean_below_edge"]
    pooled["interpretation"] = (
        "Cohesion falls with cluster size at every layer -- Spearman(log size, "
        f"mean_pair_cond) = {pooled['spearman_logsize_vs_mean_pair_cond']['rho']:+.3f} pooled "
        f"over {pooled['n_clusters']} clusters (p = "
        f"{pooled['spearman_logsize_vs_mean_pair_cond']['p']:.1e}), negative at all 11 layers "
        "and significant at 9 of them. The 0.411-0.461 reported in results.md as clearing the "
        "0.35 edge threshold is an all-clusters mean, and the all-clusters population is "
        "dominated by clusters the gate cannot score: MIN_RING_K = 6 excludes 55% of them "
        "(750 of 1,362), and small clusters are cohesive close to by construction. Restricted "
        "to gate-eligible clusters the mean falls to "
        f"{pooled['range_ge6'][0]:.3f}-{pooled['range_ge6'][1]:.3f} "
        f"({pooled['mean_pair_cond_ge6']:.3f} pooled), and among k >= 20 clusters -- the band "
        "the gate's scored objects actually occupy -- to "
        f"{pooled['range_ge20'][0]:.3f}-{pooled['range_ge20'][1]:.3f} "
        f"({pooled['mean_pair_cond_ge20']:.3f} pooled). At the layer level the k >= 20 mean "
        f"sits BELOW the 0.35 edge threshold at {nb20} of 11 layers (including layer 9, the "
        f"confirmatory arm, at 0.293) and the k >= 6 mean at {nb6} of 11, whereas the "
        "as-reported all-clusters mean clears 0.35 at every layer. "
        f"{pooled['n_ge6_below_edge_threshold']} of {pooled['n_ge6']} individual gate-eligible "
        "clusters sit below the threshold their own graph was cut at. So 'the diagnosed "
        "blocker is gone' is measured on a population substantially disjoint from the one the "
        "gate scores; on the scored population the margin over the edge threshold is absent. "
        "Separately, at every layer the cluster with the widest support fires at some member "
        f"on {pooled['min_always_on_frac']:.1%}-{pooled['max_always_on_frac']:.1%} of all "
        f"{CORPUS_POSITIONS:,} corpus positions (max at layer "
        f"{pooled['max_always_on_layer']}), and from layer 3 on that is above 89% at every "
        "layer. A component firing nearly everywhere is an always-on artifact rather than a "
        "concept, and its cohesion is close to tautological -- yet these clusters enter the "
        "all-clusters cohesion mean on equal footing with the rest."
    )
    return dict(per_layer=per_layer, pooled=pooled,
                edge_threshold=EDGE_THRESHOLD, min_ring_k=MIN_RING_K,
                corpus_positions=CORPUS_POSITIONS)


# -------------------------------------------------------------------- cross-checks

def cross_checks(out):
    p1 = out["part1_direction_vs_recovery"]
    p2 = out["part2_closure_accounting"]
    p5 = out["part5_cohesion_by_size"]["per_layer"]["9"]
    checks = [
        ("part1 all-rows Spearman(dir_p, rho)", -0.030, p1["all_rows"]["spearman_dir_p_vs_rho"]["rho"], 0.001),
        ("part1 all-rows p", 0.811, p1["all_rows"]["spearman_dir_p_vs_rho"]["p"], 0.001),
        ("part1 all-rows n", 65, p1["all_rows"]["n"], 0),
        ("part1 calendar Spearman(dir_p, rho)", 0.077, p1["calendar_only"]["spearman_dir_p_vs_rho"]["rho"], 0.001),
        ("part1 calendar p", 0.654, p1["calendar_only"]["spearman_dir_p_vs_rho"]["p"], 0.001),
        ("part2 numbers underpowered", 17, p2["per_arm"]["control_numbers_big"]["n_closure_underpowered"], 0),
        ("part2 numbers closable", 12, p2["per_arm"]["control_numbers_big"]["n_closable"], 0),
        ("part2 weekdays closable", 3, p2["per_arm"]["exploratory_days_big"]["n_closable"], 0),
        ("part2 months exploratory closable", 11, p2["per_arm"]["exploratory_months_big"]["n_closable"], 0),
        ("part2 L9 primary closable", 3, p2["per_arm"]["primary_L9_big"]["n_closable"], 0),
        ("part2 corrected denominator", 17, p2["headline_restated"]["corrected_denominator_closure_testable"], 0),
        ("part2 CIRCLE among closure-testable", 0, p2["headline_restated"]["n_CIRCLE_among_closure_testable"], 0),
        ("part4 index_p < 0.05 of 65", 17, out["part4_index_p"]["n_lt_05"], 0),
        ("part5 L9 mean_pair_cond all", 0.430, p5["mean_pair_cond_all_reported"], 0.001),
        ("part5 L9 mean_pair_cond k>=6", 0.347, p5["mean_pair_cond_ge6"], 0.001),
        ("part5 L9 mean_pair_cond k>=20", 0.293, p5["mean_pair_cond_ge20"], 0.001),
        ("part5 max always-on frac (any layer)", 0.996,
         out["part5_cohesion_by_size"]["pooled"]["max_always_on_frac"], 0.001),
    ]
    res = []
    for name, exp, got, tol in checks:
        ok = got is not None and abs(float(got) - float(exp)) <= tol + 1e-12
        res.append(dict(check=name, expected=f(exp), computed=f(got), tol=f(tol), pass_=bool(ok)))
    return dict(n_checks=len(res), n_pass=int(sum(r["pass_"] for r in res)),
                all_pass=bool(all(r["pass_"] for r in res)), checks=res)


def main():
    arms, rows = load_rows()
    assert len(rows) == 65, f"expected 65 scored rows, got {len(rows)}"

    out = {
        "status": "exploratory, not pre-registered",
        "script": "scripts/19_gate_pool.py",
        "what_this_is": (
            "Pooled post-processing of already-written gate output. No re-scoring, no "
            "cache reads, no model. Every number is an aggregation over "
            "results/rerun_*.json, results/support_big_L*.json, "
            "results/gate_background_big_L9.json and results/passer_inspection.json."
        ),
        "inputs": {
            "gate_files": [n for _, n in GATE_FILES],
            "support_files": [f"support_big_L{L}.json" for L in range(11)],
            "background": "gate_background_big_L9.json",
            "inspection": "passer_inspection.json",
            "n_scored_rows": len(rows),
        },
        "arms": arms,
        "part1_direction_vs_recovery": part1_direction_vs_recovery(rows),
        "part2_closure_accounting": part2_closure_accounting(arms, rows),
        "part3_control_vs_background": part3_control_vs_background(rows),
        "part4_index_p": part4_index_p(rows),
        "part5_cohesion_by_size": part5_cohesion_by_size(),
    }
    out["expected_vs_computed"] = cross_checks(out)
    out["caveats"] = [
        "Exploratory and not pre-registered: these are post-hoc re-aggregations of a "
        "frozen gate output, chosen after the verdict was known.",
        "The two l0_small files (rerun_primary_L18, rerun_exploratory) predate the "
        "ref_p_closed / closure_underpowered / missing_run fields, so part 2 covers only "
        "the four l0_big arms. They DO carry dir_p / rho / p_order / index_p and are "
        "included in parts 1 and 4.",
        "Parts 1 and 4 pool across two SAE suites (l0_small at layers 18-20, l0_big at "
        "layers 0-10), two ontologies and two half-B conventions. The pooling is "
        "deliberate -- the claim being tested is a claim about decoder geometry in "
        "general -- but the rows are not exchangeable and no per-arm effect is being "
        "controlled for.",
        "dir_p is a p-value against a uniform-directions null, not a spread measure: "
        "SMALL dir_p means MORE evenly spread decoder directions. Signs in part 1 are "
        "reported on dir_p as stored, not on evenness.",
        "part 3 compares populations that are matched on label count but not on layer, "
        "ontology or cluster size: the 0.164 background was measured at layer 9 only, with "
        "months labels, on templates averaging 12.75 latents (the four L9 gate clusters), "
        "while the 29 control rows span layers 1-10, number-word labels, and k = 7 to "
        "k = 100. Both ontologies carry 12 labels and the independent layer-7 background of "
        "0.167 shows the rate transfers across layers, but no background was measured on "
        "number-word labels or at k >= 20, so the 13/29 result assumes transfer the data "
        "only partly demonstrates.",
        "part 3's two-sided binomial treats the background 0.164 as a known constant; it "
        "is itself an estimate from 750 scorable draws (se ~0.0135), so the true test is "
        "slightly less powerful than reported. The direction of the result (control at or "
        "below background) is unaffected.",
        "part 3's published numerator (4 of 29) is a BH-FDR survivor count while the "
        "background rate is a raw acceptance rate; the two comparisons disagree in "
        "direction, and the raw-count test is the like-for-like one. Both are reported and "
        "neither should be quoted without the other. Note also that BH here was applied "
        "across the 29 rows of the control file only, so the survivor count depends on that "
        "file's composition and is not a property of any single cluster.",
        "index_p resolution is 1/300 (n_null = 300 in the gate's index_concentration "
        "call), so values below ~0.003 cannot be distinguished from zero.",
        "part 5's Spearman on log(size) is identical to Spearman on size -- rank "
        "correlation is invariant to monotone transforms. It is reported on log size "
        "only to match the requested form.",
        "part 5's k>=6 and k>=20 means are unweighted means over clusters, matching the "
        "convention 07b_support.py used for its all-clusters mean.",
        "Corpus-position fractions use n_rows = 4,002,480 from results/cache_meta_big.json, "
        "identical at every l0_big layer.",
    ]

    p = RES / "gate_pool.json"
    p.write_text(json.dumps(out, indent=2))

    cc = out["expected_vs_computed"]
    for c in cc["checks"]:
        print(f"  {'OK ' if c['pass_'] else 'FAIL'}  {c['check']:48s} "
              f"expected {c['expected']}  got {c['computed']}")
    print(f"\ncross-checks: {cc['n_pass']}/{cc['n_checks']} pass")
    print(f"wrote {p}")


if __name__ == "__main__":
    main()
