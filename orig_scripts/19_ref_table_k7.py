"""The missing k=7 reference-ring power table (DECLARED EXPLORATORY, not pre-registered).

PREREGISTRATION.md section (b) publishes a reference table for k=12 months only: the
closure p-value a GEOMETRICALLY PERFECT ring would attain under the gate's own closure
null as labels drop out for lack of firing support. The weekday arm (k=7, support floor
ceil(2*7/3) = 5 of 7) was scored with no such table, so "0 of 7 weekday clusters pass
closure" was never divided by the number of weekday clusters that COULD have passed.

This script builds that table for k=7 and applies it. It does not re-implement the
machinery: `gap_pvalue` and `reference_ring` are imported from scripts/13_rerun_gate.py
itself, so every number here comes from the same 20,000-permutation uniform-angle null,
the same add-one (Phipson & Smyth) p-value and the same `missing_run` /
`closure_underpowered` definitions the gate emitted.

Nothing here changes any pass/fail classification. `ref_p_closed` was pre-registered as
reporting-only and stays that way; this is a denominator, not a re-scoring.
"""
import importlib.util, json, math, sys
from itertools import combinations
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# Import the gate's real machinery rather than a copy of it. The module's heavy work is
# all inside main() and no cache layer is ever opened, but the import is not free:
# scripts/13_rerun_gate.py imports gfdisc.saes at module level, so the whole run peaks near
# 0.24 GB RSS -- /usr/bin/time -l reports a maximum resident set size of 249,724,928 B
# (~238 MB) here and 247,808,000 B on an independent re-run. Runtime ~4 s, single process.
_spec = importlib.util.spec_from_file_location("gate13", ROOT / "scripts" / "13_rerun_gate.py")
GATE = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(GATE)

gap_pvalue = GATE.gap_pvalue                      # 20,000 uniform draws, add-one p
reference_ring = GATE.reference_ring              # ref_gap_deg / ref_p_closed / missing_run
min_labels_with_support = GATE.min_labels_with_support

N_PERM = 20000                                    # the gate's default, restated for the record
P_FLOOR = 1.0 / (1 + N_PERM)                      # 4.99975e-05
ALPHA = 0.05
N_SEEDS = 200                                     # Monte-Carlo seed spread, to show the
                                                  # ref_p < 0.05 verdicts are not seed artefacts


def ref_seed_of(layer, member0):
    """The seed the gate handed to reference_ring for a given row.

    scripts/13_rerun_gate.py: seed = abs(hash((layer, c["members"][0]))) % 10**6, and that
    same seed is forwarded to `reference_ring`. hash() of a tuple of ints is not affected
    by PYTHONHASHSEED, so this is reproducible across processes.
    """
    return abs(hash((int(layer), int(member0)))) % 10 ** 6


def exact_p_closed(k, gap_rad):
    """Closed form of what gap_pvalue estimates: P(max gap of k uniform angles <= gap).

    P(max gap > g) = sum_j (-1)^(j+1) C(k,j) (1 - j*g/2pi)^(k-1)  over j with j*g < 2pi.
    Used only to certify the Monte-Carlo column; the gate's own MC value is what the
    table reports, so the published floors match the gate's floors.
    """
    u = gap_rad / (2 * math.pi)
    s = 0.0
    for j in range(1, k + 1):
        t = 1.0 - j * u
        if t <= 0:
            break
        s += (-1) ** (j + 1) * math.comb(k, j) * t ** (k - 1)
    return max(0.0, min(1.0, 1.0 - s))


def ring_row(keep, full_k, seed=0):
    """reference_ring for one survivor set, with the module cache cleared.

    `reference_ring` memoises on (keep, full_k) WITHOUT the seed, so a second call with a
    different seed silently returns the first call's Monte-Carlo draw. That is harmless
    inside a gate run but would corrupt a seed-spread study, so the cache is cleared.
    """
    GATE._REF_CACHE.clear()
    return dict(reference_ring(keep, full_k, seed=seed))


def seed_spread(k, gap_rad, n_seeds=N_SEEDS):
    v = np.array([gap_pvalue(k, gap_rad, n_perm=N_PERM, seed=s) for s in range(n_seeds)])
    return dict(n_seeds=int(n_seeds), min=float(v.min()), max=float(v.max()),
                mean=float(v.mean()), sd=float(v.std(ddof=1)),
                frac_below_alpha=float((v < ALPHA).mean()))


def orbits(full_k, n_missing):
    """Dropout patterns up to rotation of the cycle.

    Returns one representative per rotation orbit of the MISSING set, canonicalised as the
    lexicographically smallest rotation. For full_k = 7 (prime) every proper non-empty
    subset has a trivial stabiliser, so each orbit holds exactly full_k literal patterns.
    """
    seen = {}
    for miss in combinations(range(full_k), n_missing):
        canon = min(tuple(sorted((m + r) % full_k for m in miss)) for r in range(full_k))
        seen.setdefault(canon, []).append(miss)
    return seen


def circ_dists(miss, full_k):
    """Pairwise circular distances within the missing set (1 = adjacent)."""
    return sorted(min((a - b) % full_k, (b - a) % full_k) for a, b in combinations(sorted(miss), 2))


def longest_missing_run(miss, full_k):
    """Longest run of consecutive missing labels, wrapping. The gate derives the same
    quantity from the gap width; computed directly here as an independent check."""
    if not miss:
        return 0
    s = set(miss)
    if len(s) == full_k:
        return full_k
    best = run = 0
    # start from a present label so the wrap is handled by a single linear pass
    start = next(i for i in range(full_k) if i not in s)
    for i in range(start, start + full_k):
        if (i % full_k) in s:
            run += 1
            best = max(best, run)
        else:
            run = 0
    return best


def build_table(full_k, min_survivors, with_spread=True):
    rows = []
    for n_keep in range(full_k, min_survivors - 1, -1):
        n_missing = full_k - n_keep
        for canon, members in sorted(orbits(full_k, n_missing).items()):
            keep = sorted(set(range(full_k)) - set(canon))
            rr = ring_row(keep, full_k, seed=0)
            gap_rad = math.radians(rr["ref_gap_deg"])
            d = circ_dists(canon, full_k)
            row = dict(
                n_survivors=int(n_keep), n_missing=int(n_missing),
                missing_canonical=[int(x) for x in canon],
                survivors_example=[int(x) for x in keep],
                n_literal_patterns_in_orbit=len(members),
                pairwise_circular_distances_of_missing=[int(x) for x in d],
                adjacency=("none_missing" if n_missing == 0 else
                           "single_dropout" if n_missing == 1 else
                           "adjacent" if 1 in d else "non_adjacent"),
                ref_gap_deg=rr["ref_gap_deg"],
                ref_gap_in_slots=round(rr["ref_gap_deg"] / (360.0 / full_k), 6),
                ref_p_closed=rr["ref_p_closed"],
                ref_p_closed_exact=exact_p_closed(n_keep, gap_rad),
                at_mc_floor=bool(abs(rr["ref_p_closed"] - P_FLOOR) < 1e-12),
                missing_run=int(rr["missing_run"]),
                missing_run_direct=longest_missing_run(canon, full_k),
                closure_underpowered=bool(rr["closure_underpowered"]),
                closure_testable=bool(rr["ref_p_closed"] < ALPHA),
                scorable_by_support_rule=bool(n_keep >= min_labels_with_support(full_k)),
            )
            row["flag_agrees_with_testability"] = bool(
                row["closure_underpowered"] != row["closure_testable"])
            row["missing_run_check_ok"] = bool(row["missing_run"] == row["missing_run_direct"])
            if with_spread:
                row["mc_seed_spread"] = seed_spread(n_keep, gap_rad)
            rows.append(row)
    return rows


def lookup(table, keep, full_k):
    canon = min(tuple(sorted((m + r) % full_k for m in set(range(full_k)) - set(keep)))
                for r in range(full_k))
    for row in table:
        if tuple(row["missing_canonical"]) == canon:
            return row
    raise KeyError(canon)


def main():
    out = dict(
        status="exploratory, not pre-registered",
        item="5c -- k=7 reference-ring power table for the weekday arm",
        generated_by="scripts/19_ref_table_k7.py",
        machinery=dict(
            source="scripts/13_rerun_gate.py (imported, not re-implemented): gap_pvalue, "
                   "reference_ring, min_labels_with_support",
            null="20,000 draws of k uniform angles on the circle; statistic = largest "
                 "angular gap; p = (1 + #{null gap <= observed gap}) / (1 + n_perm)",
            n_perm=N_PERM, p_floor=P_FLOOR, alpha=ALPHA,
            support_rule="MIN_FIRING_PER_LABEL=10 firing positions and >=5 scoring "
                         "positions per label; ceil(2k/3) labels with support "
                         "(8 of 12 months, 5 of 7 weekdays)",
            note="a perfect ring is the survivors placed at their TRUE cycle positions "
                 "2*pi*i/full_k, which is exactly what reference_ring assumes",
        ),
    )

    # ---- 0. terms, with the one inherited overstatement corrected -----------------------
    # PREREGISTRATION.md (b), line 243, says a survivor set with two adjacent months
    # unsupported "cannot pass closure whatever its geometry", and reference_ring's own
    # docstring repeats it ("cannot pass closure at all, whatever the geometry"). Taken
    # literally that is false for the closure tier in isolation: reference_ring pins the
    # survivors at their TRUE cycle angles, but closure scores whatever angles the label
    # MEANS happen to land on, and those are not pinned. Six survivors spread evenly on
    # the circle leave a 60 deg gap and score the Monte-Carlo floor. The defensible
    # reading -- the one used everywhere below -- is the true-angle one.
    even_gap_rad = 2 * math.pi / 6
    out["definitions"] = dict(
        closure_testable=("a survivor set is CLOSURE-TESTABLE iff a geometrically perfect "
                          "ring -- the survivors placed at their TRUE cycle angles "
                          "2*pi*i/full_k -- scores ref_p_closed < 0.05 under the gate's "
                          "closure null. Untestable means: no ring placing the survivors at "
                          "their true cycle angles could pass closure."),
        scope_correction=("this is NARROWER than the absolute phrasing inherited from "
                          "PREREGISTRATION.md (b) ('cannot pass closure whatever its "
                          "geometry') and from reference_ring's docstring. Closure alone "
                          "does not pin the recovered angles to true cycle positions, so a "
                          "6-survivor set whose label means land evenly spread passes "
                          "closure outright. The absolute claim is an overstatement in the "
                          "pre-registered text; it is quoted here only to be qualified, and "
                          "no pre-registered file was edited."),
        counterexample=dict(
            configuration="6 surviving labels whose recovered angles are evenly spread "
                          "(max gap 360/6 = 60 deg), rather than at their true 7-cycle angles",
            max_gap_deg=60.0,
            mc_p_closed_seed0=gap_pvalue(6, even_gap_rad, n_perm=N_PERM, seed=0),
            at_mc_floor=bool(abs(gap_pvalue(6, even_gap_rad, n_perm=N_PERM, seed=0)
                                 - P_FLOOR) < 1e-12),
            exact_p_closed=0.0,
            exact_note=("exactly 0, not merely tiny: 6 gaps sum to 360 deg, so the largest "
                        "of 6 uniform gaps is >= 60 deg with probability 1 (equality has "
                        "measure zero). The alternating-sum formula returns %.3e for this "
                        "case, which is floating-point residue, not a probability."
                        % exact_p_closed(6, even_gap_rad)),
            true_angle_comparison=("the same 6 survivors at their true 7-cycle angles leave "
                                   "a 2-slot gap of 102.86 deg and score ~0.099, which is "
                                   "why they are called closure-untestable here."),
            reference_ring_is_unaffected=("reference_ring is used exactly as the gate uses "
                                          "it; only the English gloss is narrowed."),
        ),
        why_it_does_not_change_N=("testability is a statement about the reference ring, and "
                                  "every number in this file is computed from the reference "
                                  "ring. N = 3 and every p-value are unchanged by the "
                                  "wording correction."),
    )

    # ---- 1. the k=7 table -------------------------------------------------------------
    k7 = build_table(7, min_survivors=4)          # 4/7 included, flagged not scorable
    k7_scorable_rows = [r for r in k7 if r["scorable_by_support_rule"]]

    # Two different units count the same table and must never be mixed in one fraction.
    # ORBITS are dropout patterns up to rotation (5 of them at >= 5 survivors); CLASSES
    # collapse orbits that produce the same (n_survivors, ref_gap_deg) and therefore the
    # same p-value -- the two non-adjacent 5/7 orbits [0,2] and [0,3] both leave a 2-slot
    # gap, so they are one class but two orbits.
    classes = {}
    for r in k7_scorable_rows:
        classes.setdefault((r["n_survivors"], round(r["ref_gap_in_slots"])), []).append(r)
    class_rows = [
        dict(n_survivors=n, gap_in_slots=int(g), ref_gap_deg=v[0]["ref_gap_deg"],
             ref_p_closed=v[0]["ref_p_closed"], closure_testable=v[0]["closure_testable"],
             n_orbits_in_class=len(v),
             orbits=[x["missing_canonical"] for x in v],
             adjacency=v[0]["adjacency"])
        for (n, g), v in sorted(classes.items(), key=lambda kv: (-kv[0][0], kv[0][1]))]

    n_orbits = len(k7_scorable_rows)
    n_orbits_testable = int(sum(r["closure_testable"] for r in k7_scorable_rows))
    n_classes = len(class_rows)
    n_classes_testable = int(sum(r["closure_testable"] for r in class_rows))

    out["k7_table"] = dict(
        full_k=7, spacing_deg=360.0 / 7,
        support_minimum=min_labels_with_support(7),
        rows=k7,
        gap_classes=class_rows,
        summary=dict(
            scorable_orbits=n_orbits,
            closure_testable_orbits=n_orbits_testable,
            testable_patterns=[r["missing_canonical"] for r in k7_scorable_rows
                               if r["closure_testable"]],
            scorable_gap_classes=n_classes,
            closure_testable_gap_classes=n_classes_testable,
            units_note=("ORBITS and CLASSES are different units; never divide one by the "
                        "other. Correct sentences: '%d of %d scorable rotation-orbits are "
                        "closure-testable' and '%d of %d distinct gap/adjacency classes are "
                        "closure-testable'. The orbit count is larger because the two "
                        "non-adjacent 5/7 dropout patterns (circular distance 2 and "
                        "distance 3) are two DISTINCT rotation orbits that happen to share "
                        "one 2-slot gap and therefore one p-value."
                        % (n_orbits_testable, n_orbits, n_classes_testable, n_classes)),
            summary_sentence=("%d of %d scorable rotation-orbits (equivalently %d of %d "
                              "distinct gap/adjacency classes) are closure-testable at k=7."
                              % (n_orbits_testable, n_orbits,
                                 n_classes_testable, n_classes)),
        ),
    )

    # ---- 2. apply to the seven scored weekday clusters ---------------------------------
    days_path = ROOT / "results" / "rerun_gate_big_exploratory_days.json"
    days = json.load(open(days_path))
    applied, disagreements = [], []
    for r in days["rows"]:
        keep = [int(x) for x in r["labels"]]
        tab = lookup(k7, keep, 7)
        own_seed = ref_seed_of(r["layer"], r["members"][0])
        own = ring_row(keep, 7, seed=own_seed)
        row = dict(
            layer=int(r["layer"]), metric=r["metric"], size=int(r["size"]),
            member0=int(r["members"][0]), n_labels=int(r["n_labels"]), labels=keep,
            missing=[int(x) for x in sorted(set(range(7)) - set(keep))],
            adjacency=tab["adjacency"],
            stored=dict(ref_gap_deg=r["ref_gap_deg"], ref_p_closed=r["ref_p_closed"],
                        missing_run=r["missing_run"],
                        closure_underpowered=bool(r["closure_underpowered"]),
                        p_closed=r["p_closed"], max_gap_deg=r["max_gap_deg"],
                        closure_uninformative=bool(r.get("closure_uninformative", False))),
            # route A: the table, canonical seed 0
            routeA_table_ref_p=tab["ref_p_closed"],
            routeA_testable=bool(tab["closure_testable"]),
            # route B: the row's own stored value
            routeB_stored_ref_p=r["ref_p_closed"],
            routeB_testable=bool(r["ref_p_closed"] < ALPHA),
            # route C: recompute with the seed the gate itself used for this row
            routeC_own_seed=int(own_seed),
            routeC_ref_p=own["ref_p_closed"],
            routeC_testable=bool(own["ref_p_closed"] < ALPHA),
            routeC_bitwise_equals_stored=bool(own["ref_p_closed"] == r["ref_p_closed"]),
            gap_deg_matches_stored=bool(abs(tab["ref_gap_deg"] - r["ref_gap_deg"]) < 1e-9),
            missing_run_matches_stored=bool(tab["missing_run"] == r["missing_run"]),
            exact_ref_p=tab["ref_p_closed_exact"],
        )
        row["routes_agree"] = bool(row["routeA_testable"] == row["routeB_testable"]
                                   == row["routeC_testable"])
        # the pre-registered flag is a k=12-calibrated proxy for the same question
        row["flag_says_untestable"] = bool(r["closure_underpowered"])
        row["flag_matches_testability"] = bool(r["closure_underpowered"] != row["routeB_testable"])
        # what the gate would have reported had "uninformative" used ref_p instead of the flag
        row["closure_uninformative_by_refp"] = bool(r["p_closed"] >= ALPHA
                                                    and not row["routeB_testable"])
        if not row["routes_agree"] or not row["gap_deg_matches_stored"] \
                or not row["missing_run_matches_stored"]:
            disagreements.append(dict(kind="route_disagreement", layer=row["layer"],
                                      member0=row["member0"], detail=row))
        if not row["flag_matches_testability"]:
            disagreements.append(dict(
                kind="closure_underpowered_flag_misses_untestable_row",
                layer=row["layer"], member0=row["member0"], labels=keep,
                stored_ref_p_closed=r["ref_p_closed"],
                stored_closure_underpowered=bool(r["closure_underpowered"]),
                stored_closure_uninformative=bool(r.get("closure_uninformative", False)),
                message=("missing_run = %d < 2 so closure_underpowered is false, but a "
                         "geometrically perfect 7-cycle on these %d survivors scores "
                         "ref_p_closed = %.4f >= 0.05: no ring placing these survivors at "
                         "their TRUE cycle angles could have passed closure, yet the row is "
                         "not flagged. (Narrower than the prereg's 'whatever its geometry' "
                         "-- see definitions.scope_correction.)"
                         % (r["missing_run"], len(keep), r["ref_p_closed"]))))
        if not row["routeC_bitwise_equals_stored"]:
            disagreements.append(dict(
                kind="stored_value_came_from_another_rows_seed",
                layer=row["layer"], member0=row["member0"], labels=keep,
                own_seed=int(own_seed), own_seed_ref_p=own["ref_p_closed"],
                stored_ref_p=r["ref_p_closed"],
                same_verdict=bool(row["routeC_testable"] == row["routeB_testable"]),
                message=("reference_ring memoises on (labels, full_k) without the seed, so "
                         "this row inherited the Monte-Carlo draw of the first row in the "
                         "run with the same survivor set. Verdict unchanged.")))
        applied.append(row)

    n_scored = len(applied)
    n_testable = sum(r["routeB_testable"] for r in applied)
    out["k7_application"] = dict(
        source=str(days_path.relative_to(ROOT)),
        n_scored=n_scored, rows=applied,
        n_closure_testable=int(n_testable),
        n_closure_untestable=int(n_scored - n_testable),
        routes_all_agree=bool(all(r["routes_agree"] for r in applied)),
        flag_disagreements=int(sum(not r["flag_matches_testability"] for r in applied)),
        disagreements=disagreements,
    )

    # ---- 3. the corrected weekday statement -------------------------------------------
    # Canonical seed-0 rows, pulled from the table rather than typed in, so the prose can
    # never drift from the numbers it quotes.
    full7 = next(r for r in k7 if r["n_survivors"] == 7)
    six_of_seven = next(r for r in k7 if r["n_survivors"] == 6)
    five_non_adj = next(r for r in k7 if r["n_survivors"] == 5 and r["adjacency"] == "non_adjacent")
    five_adj = next(r for r in k7 if r["n_survivors"] == 5 and r["adjacency"] == "adjacent")

    testable_rows = [r for r in applied if r["routeB_testable"]]
    out["corrected_statement"] = dict(
        as_published="0 of 7 weekday clusters pass closure",
        corrected="0 of %d closure-testable weekday clusters pass closure "
                  "(for the other %d of the 7 scored clusters, no ring placing their "
                  "surviving labels at the true weekday-cycle angles could have passed "
                  "closure)" % (n_testable, n_scored - n_testable),
        N=int(n_testable),
        derivation=("a scored weekday cluster is closure-testable iff a geometrically "
                    "perfect 7-cycle on its surviving labels scores ref_p_closed < 0.05. "
                    "At k=7 that holds for 7/7 support (p = %.3e, the %d-permutation floor) "
                    "and for 5/7 with the two dropouts NON-adjacent (p = %.4f, both the "
                    "distance-2 and the distance-3 orbit); it fails for 6/7 (p = %.4f) and "
                    "for 5/7 adjacent (p = %.4f). Quoted at seed 0; each scored cluster "
                    "carries its own gate seed, so its stored ref_p_closed differs in the "
                    "third decimal (Monte-Carlo only -- see k7_application.rows)."
                    % (full7["ref_p_closed"], N_PERM, five_non_adj["ref_p_closed"],
                       six_of_seven["ref_p_closed"], five_adj["ref_p_closed"])),
        orbit_units=out["k7_table"]["summary"]["summary_sentence"],
        scope=out["definitions"]["closure_testable"],
        testable_clusters=[dict(layer=r["layer"], metric=r["metric"], size=r["size"],
                                labels=r["labels"], ref_p_closed=r["routeB_stored_ref_p"],
                                observed_p_closed=r["stored"]["p_closed"],
                                observed_max_gap_deg=r["stored"]["max_gap_deg"])
                           for r in testable_rows],
        untestable_clusters=[dict(layer=r["layer"], metric=r["metric"], size=r["size"],
                                  labels=r["labels"], adjacency=r["adjacency"],
                                  ref_p_closed=r["routeB_stored_ref_p"],
                                  flagged_closure_underpowered=r["flag_says_untestable"])
                             for r in applied if not r["routeB_testable"]],
        published_underpowered_count=int(sum(r["flag_says_untestable"] for r in applied)),
        # the correction is to the closure DENOMINATOR only. It rescues nothing: the
        # weekday arm already failed tier 1, so closure was never the binding constraint.
        ordering_context=dict(
            n_passing_ordering_fdr=int(sum(bool(r.get("ordering_fdr")) for r in days["rows"])),
            min_p_order=float(min(r["p_order"] for r in days["rows"])),
            max_rho=float(max(r["rho"] for r in days["rows"])),
            note="0 of 7 weekday clusters pass the ordering tier (min p_order = %.4f, "
                 "max rho = %.3f), so no weekday cluster reached closure as a live "
                 "question. The corrected denominator sharpens what the closure tier "
                 "could have said, and changes no verdict."
                 % (min(r["p_order"] for r in days["rows"]),
                    max(r["rho"] for r in days["rows"])),
        ),
    )

    # ---- 4. k=12 validation against the published table --------------------------------
    published = [
        dict(label="all 12 months", missing=[], gap_deg=30.0, published_p="0.0000"),
        dict(label="two non-adjacent months missing", missing=[0, 2], gap_deg=60.0,
             published_p="0.0088"),
        dict(label="two adjacent months missing", missing=[0, 1], gap_deg=90.0,
             published_p="0.3370"),
        dict(label="three adjacent months missing", missing=[0, 1, 2], gap_deg=120.0,
             published_p="0.6579"),
    ]
    val = []
    for p in published:
        keep = sorted(set(range(12)) - set(p["missing"]))
        rr = ring_row(keep, 12, seed=0)
        gap_rad = math.radians(rr["ref_gap_deg"])
        pub = float(p["published_p"])
        mine = rr["ref_p_closed"]
        sp = seed_spread(len(keep), gap_rad)
        val.append(dict(
            label=p["label"], missing=p["missing"], n_survivors=len(keep),
            published_gap_deg=p["gap_deg"], recomputed_gap_deg=rr["ref_gap_deg"],
            gap_matches=bool(abs(rr["ref_gap_deg"] - p["gap_deg"]) < 1e-9),
            published_p=p["published_p"], recomputed_p_seed0=mine,
            exact_p=exact_p_closed(len(keep), gap_rad),
            abs_diff_vs_published=abs(mine - pub),
            matches_to_4dp=bool(round(mine, 4) == round(pub, 4)),
            # the published column carries 4 dp, so the range is compared at 4 dp too;
            # "all 12 months" is printed as 0.0000 but is really the MC floor 4.99975e-05
            within_mc_seed_range=bool(round(sp["min"], 4) <= pub <= round(sp["max"], 4)),
            published_is_rounded_floor=bool(pub == 0.0),
            mc_seed_spread=sp,
            missing_run=int(rr["missing_run"]),
            closure_underpowered=bool(rr["closure_underpowered"]),
            closure_testable=bool(mine < ALPHA),
        ))
    # Is `missing_run >= 2` exactly right in its HOME arm? Sweep every scorable k=12
    # support pattern (>= 8 survivors, up to rotation) and compare the flag against
    # ref_p_closed >= 0.05. If it agrees everywhere at k=12 and fails at k=7, the defect
    # is a transfer failure of a calibrated threshold, not a bad threshold.
    k12_all = build_table(12, min_survivors=8, with_spread=False)
    k12_bad = [dict(missing=r["missing_canonical"], n_survivors=r["n_survivors"],
                    ref_gap_deg=r["ref_gap_deg"], ref_p_closed=r["ref_p_closed"],
                    missing_run=r["missing_run"],
                    closure_underpowered=r["closure_underpowered"],
                    closure_testable=r["closure_testable"])
               for r in k12_all if not r["flag_agrees_with_testability"]]
    k7_scorable = k7_scorable_rows
    k7_bad = [dict(missing=r["missing_canonical"], n_survivors=r["n_survivors"],
                   ref_p_closed=r["ref_p_closed"], missing_run=r["missing_run"],
                   closure_underpowered=r["closure_underpowered"],
                   closure_testable=r["closure_testable"])
              for r in k7_scorable if not r["flag_agrees_with_testability"]]
    out["flag_calibration_check"] = dict(
        question="does closure_underpowered (missing_run >= 2) coincide with "
                 "ref_p_closed >= 0.05 over every scorable support pattern?",
        k12=dict(n_orbits=len(k12_all), n_mismatched=len(k12_bad), mismatches=k12_bad,
                 min_survivors=8),
        k7=dict(n_orbits=len(k7_scorable), n_mismatched=len(k7_bad), mismatches=k7_bad,
                min_survivors=5),
    )

    out["k12_validation"] = dict(
        rows=val,
        n_gap_matches=int(sum(r["gap_matches"] for r in val)),
        n_p_matches_to_4dp=int(sum(r["matches_to_4dp"] for r in val)),
        n_p_within_seed_range=int(sum(r["within_mc_seed_range"] for r in val)),
        note=("the published table's p-values are single Monte-Carlo draws at 20,000 "
              "permutations; agreement is asserted against the seed-to-seed range as well "
              "as against seed 0, since the gate seeds per cluster."),
    )

    # ---- headline notes ----------------------------------------------------------------
    # (full7 / six_of_seven / five_non_adj / five_adj were bound in section 3.)
    # The two k=12 rows the pre-registered threshold note talks about, read out of the
    # sweep above instead of restated as literals. Both have the SAME 60 deg gap, which is
    # how PREREGISTRATION.md came to attach the two-non-adjacent p-value to the single
    # dropout; threshold_transfer_failure below corrects that, quoting computed numbers.
    k12_single = lookup(k12_all, sorted(set(range(12)) - {0}), 12)
    k12_two_non_adj = lookup(k12_all, sorted(set(range(12)) - {0, 2}), 12)
    out["headline"] = dict(
        k7_full_ring_p=P_FLOOR,
        k7_one_dropout_p=six_of_seven["ref_p_closed"],
        k7_two_non_adjacent_p=five_non_adj["ref_p_closed"],
        k7_two_adjacent_p=five_adj["ref_p_closed"],
        k7_orbit_counts=out["k7_table"]["summary"]["summary_sentence"],
        threshold_transfer_failure=(
            # named (not positional) format keys: ten computed values fill this sentence,
            # and a silent argument swap would read as plausibly as the error corrected here.
            "closure_underpowered is defined as missing_run >= 2 and was calibrated on "
            "k=12, where a single dropout (%(k12_n)d survivors, %(k12_gap).2f deg gap) "
            "still passes closure at %(k12_p).4f. CORRECTION TO THE PRE-REGISTERED NOTE: "
            "PREREGISTRATION.md's 'Note on the threshold' quotes 0.0088 as that single "
            "dropout's p-value, but 0.0088 is really the k=12 TWO-non-adjacent-months "
            "figure (%(k12b_n)d survivors, the same %(k12b_gap).2f deg gap, recomputed "
            "here as %(k12b_p).4f). The single dropout, recomputed from the k=12 sweep in "
            "flag_calibration_check, scores %(k12_p).4f (exact max-gap law %(k12_exact).4f)"
            " -- the two patterns share a gap width but not a survivor count, which is how "
            "the two p-values were conflated. The prereg's qualitative claim survives the "
            "correction (a k=12 single dropout is still closure-testable at alpha = 0.05); "
            "only its number was wrong, and it is corrected here rather than in any "
            "pre-registered file, none of which were edited. At k=7 a single dropout "
            "already scores %(k7_p).4f >= 0.05, so the flag under-reports non-testability "
            "in the weekday arm: %(n_untestable)d of the 7 scored weekday clusters are "
            "closure-untestable, but only %(n_flagged)d are flagged."
            % dict(k12_n=k12_single["n_survivors"],
                   k12_gap=k12_single["ref_gap_deg"],
                   k12_p=k12_single["ref_p_closed"],
                   k12_exact=k12_single["ref_p_closed_exact"],
                   k12b_n=k12_two_non_adj["n_survivors"],
                   k12b_gap=k12_two_non_adj["ref_gap_deg"],
                   k12b_p=k12_two_non_adj["ref_p_closed"],
                   k7_p=six_of_seven["ref_p_closed"],
                   n_untestable=n_scored - n_testable,
                   n_flagged=sum(r["flag_says_untestable"] for r in applied))),
        non_monotonicity=(
            "dropping a SECOND, non-adjacent weekday makes closure easier, not harder: the "
            "gap stays at 2 slots (%.2f deg) while the uniform null loses a point, so p goes "
            "%.4f (6 survivors) -> %.4f (5 survivors). A property of the pre-registered "
            "null, not of any cluster."
            % (six_of_seven["ref_gap_deg"], six_of_seven["ref_p_closed"],
               five_non_adj["ref_p_closed"])),
        scope_note=out["definitions"]["scope_correction"],
    )

    dest = ROOT / "results" / "ref_table_k7.json"
    dest.write_text(json.dumps(out, indent=2))

    # ---- console summary ----------------------------------------------------------------
    print("k=7 reference table (perfect ring, %d perms, seed 0):" % N_PERM)
    for r in k7:
        print("  %d/7 %-14s miss=%-9s gap=%6.2f deg (%.0f slots)  ref_p=%.5f  run=%d  "
              "underpowered=%-5s testable=%-5s scorable=%s"
              % (r["n_survivors"], r["adjacency"], str(r["missing_canonical"]),
                 r["ref_gap_deg"], r["ref_gap_in_slots"], r["ref_p_closed"],
                 r["missing_run"], r["closure_underpowered"], r["closure_testable"],
                 r["scorable_by_support_rule"]))
    print("\nweekday rows (%d scored):" % n_scored)
    for r in applied:
        print("  L%d %-12s k=%3d labels=%-21s A=%.5f B=%.5f C=%.5f agree=%s "
              "flagUP=%-5s testable=%s"
              % (r["layer"], r["metric"], r["size"], str(r["labels"]),
                 r["routeA_table_ref_p"], r["routeB_stored_ref_p"], r["routeC_ref_p"],
                 r["routes_agree"], r["flag_says_untestable"], r["routeB_testable"]))
    print("\nDISAGREEMENTS: %d" % len(disagreements))
    for d in disagreements:
        print("  !! %s  L%s member0=%s" % (d["kind"], d.get("layer"), d.get("member0")))
        if "message" in d:
            print("     %s" % d["message"])
    print("\nunits: %s" % out["k7_table"]["summary"]["summary_sentence"])
    for c in out["k7_table"]["gap_classes"]:
        print("  class n=%d gap=%d slots (%.2f deg) ref_p=%.5f testable=%-5s "
              "orbits=%d %s"
              % (c["n_survivors"], c["gap_in_slots"], c["ref_gap_deg"], c["ref_p_closed"],
                 c["closure_testable"], c["n_orbits_in_class"], c["orbits"]))
    print("\ncorrected: %s" % out["corrected_statement"]["corrected"])
    fc = out["flag_calibration_check"]
    print("\nflag calibration (missing_run>=2  vs  ref_p>=0.05):")
    print("  k=12, >=8 survivors: %d/%d orbits mismatched" % (fc["k12"]["n_mismatched"],
                                                              fc["k12"]["n_orbits"]))
    for m in fc["k12"]["mismatches"]:
        print("     miss=%s n=%d ref_p=%.4f run=%d UP=%s testable=%s"
              % (m["missing"], m["n_survivors"], m["ref_p_closed"], m["missing_run"],
                 m["closure_underpowered"], m["closure_testable"]))
    print("  k=7,  >=5 survivors: %d/%d orbits mismatched" % (fc["k7"]["n_mismatched"],
                                                              fc["k7"]["n_orbits"]))
    for m in fc["k7"]["mismatches"]:
        print("     miss=%s n=%d ref_p=%.4f run=%d UP=%s testable=%s"
              % (m["missing"], m["n_survivors"], m["ref_p_closed"], m["missing_run"],
                 m["closure_underpowered"], m["closure_testable"]))
    print("\nk=12 validation:")
    for r in val:
        print("  %-32s gap %6.2f (pub %6.2f, ok=%s)  mine=%.4f  pub=%s  4dp=%s  "
              "in-seed-range=%s [%.4f, %.4f]"
              % (r["label"], r["recomputed_gap_deg"], r["published_gap_deg"],
                 r["gap_matches"], r["recomputed_p_seed0"], r["published_p"],
                 r["matches_to_4dp"], r["within_mc_seed_range"],
                 r["mc_seed_spread"]["min"], r["mc_seed_spread"]["max"]))
    print("\nwrote %s" % dest)


if __name__ == "__main__":
    main()
