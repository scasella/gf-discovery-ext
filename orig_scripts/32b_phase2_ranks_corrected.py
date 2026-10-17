#!/usr/bin/env python
"""Phase-2 RANKS -- corrected artifact after verification.

WHY THIS FILE EXISTS
--------------------
`scripts/32_phase2_ranks.py` -> `results/phase2_ranks.json` produced a numerically
sound sweep whose *reported prose* misdescribed it in four places, omitted one
pre-registration deviation, and graded a safety alarm on the more lenient of the
two readings the frozen document supports.  A sixth defect -- the numerical
fragility of raw k=8 closure under a change of attention kernel -- was not visible
in the artifact at all.

The project rule is: new files only, never edit an existing one.  So this script
does NOT modify `32_phase2_ranks.py` and does NOT overwrite `phase2_ranks.json`.
It imports the original module unchanged, re-runs its `main()` twice through a
loader whose only free parameter is `attn_implementation`, proves the eager pass
reproduces the published artifact, and then writes a corrected artifact that
carries the original numbers plus the correction blocks.

WHAT IS *NOT* CHANGED
---------------------
No statistic, no null, no verdict rule, no label set, no template, no readout, no
threshold.  Every number in `rows`, `multiplicity`, `alarm`, `closure_diagnostic`,
`synthetics` and `blind_arm` of the corrected artifact is the byte-identical output
of the original code path (asserted, with the max deviation reported in
`corrections.C0_reproduction`).  What is added is descriptive: honest ranges,
an attack ledger with per-layer outcomes, the prereg deviation annotation, the
alarm's two readings side by side, and a cross-kernel census.

THE TRACK VERDICT IS UNCHANGED: supervised `ordered_not_closed`, blind
`untestable`, no alarm.  Nothing survives BH under either attention kernel.

Outputs
-------
  results/phase2_ranks_corrected.json    -- authoritative, supersedes phase2_ranks.json
  results/phase2_ranks_sdpa_census.json  -- the full sdpa sweep, for audit
"""
import argparse, contextlib, gc, importlib.util, json, sys, time
from itertools import permutations
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

_spec = importlib.util.spec_from_file_location(
    "ranks32", ROOT / "scripts" / "32_phase2_ranks.py")
R32 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(R32)                       # module-level only; main() is guarded

ORIGINAL = ROOT / "results" / "phase2_ranks.json"
PREREG = "docs/phase2_prereg_2026-08-21.md"

# The prereg-registered background draw count, versus the constant the run used.
PREREG_BACKGROUND_DRAWS = 300                        # prereg 4.2(b), frozen doc
RUN_BACKGROUND_DRAWS = R32.N_BACKGROUND              # 2000, scripts/32_phase2_ranks.py:293


# ---------------------------------------------------------------------------
# Kernel-parameterised loader.  Identical to gfdisc.model.load_model except that
# attn_implementation is a parameter instead of the hard-coded "eager".
# ---------------------------------------------------------------------------
def make_loader(attn):
    def _load(token=None, device=None, dtype=None):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from gfdisc.env import DEVICE, DTYPE, MODEL_NAME
        tok = AutoTokenizer.from_pretrained(MODEL_NAME, token=token)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, dtype=DTYPE, token=token, attn_implementation=attn,
        ).to(DEVICE).eval()
        return model, tok
    return _load


def run_sweep(attn, out_path, log_path, n_background):
    """Run the ORIGINAL main() with one substitution: the attention kernel."""
    R32.load_model = make_loader(attn)
    saved = sys.argv
    sys.argv = ["32_phase2_ranks.py", "--out", str(out_path),
                "--n-background", str(n_background)]
    t = time.time()
    try:
        with open(log_path, "w") as fh, contextlib.redirect_stdout(fh):
            R32.main()
    finally:
        sys.argv = saved
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    return json.loads(Path(out_path).read_text()), time.time() - t


# ---------------------------------------------------------------------------
# C0 -- reproduction check.  The eager pass through THIS harness must reproduce
# the published artifact, otherwise no cross-kernel delta is attributable to the
# kernel.  Walks both trees and reports the largest numeric deviation.
# ---------------------------------------------------------------------------
IGNORE_KEYS = {"generated", "runtime_sec"}


def deep_compare(a, b, path="", state=None):
    if state is None:
        state = dict(max_abs=0.0, max_rel=0.0, max_path=None,
                     n_numeric=0, n_mismatched_non_numeric=0, mismatches=[])
    if isinstance(a, dict) and isinstance(b, dict):
        if set(a) != set(b):
            state["n_mismatched_non_numeric"] += 1
            state["mismatches"].append(f"{path}: key set differs")
            return state
        for k in a:
            if k in IGNORE_KEYS:
                continue
            deep_compare(a[k], b[k], f"{path}.{k}", state)
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            state["n_mismatched_non_numeric"] += 1
            state["mismatches"].append(f"{path}: length {len(a)} vs {len(b)}")
            return state
        for i, (x, y) in enumerate(zip(a, b)):
            deep_compare(x, y, f"{path}[{i}]", state)
    elif isinstance(a, bool) or isinstance(b, bool) or a is None or b is None:
        if a != b:
            state["n_mismatched_non_numeric"] += 1
            if len(state["mismatches"]) < 40:
                state["mismatches"].append(f"{path}: {a!r} vs {b!r}")
    elif isinstance(a, (int, float)) and isinstance(b, (int, float)):
        state["n_numeric"] += 1
        d = abs(float(a) - float(b))
        rel = d / max(abs(float(a)), 1e-12)
        if d > state["max_abs"]:
            state["max_abs"], state["max_path"] = d, path
        state["max_rel"] = max(state["max_rel"], rel)
    else:
        if a != b:
            state["n_mismatched_non_numeric"] += 1
            if len(state["mismatches"]) < 40:
                state["mismatches"].append(f"{path}: {a!r} vs {b!r}")
    return state


# ---------------------------------------------------------------------------
# Small helpers over a finished sweep dict.
# ---------------------------------------------------------------------------
def live_rows(out, readout=None):
    return [r for r in out["rows"]
            if not r["degenerate"] and (readout is None or r["readout"] == readout)]


def rng_of(vals):
    vals = [float(v) for v in vals]
    return [min(vals), max(vals)] if vals else [None, None]


def cell(out, readout, layer):
    for r in out["rows"]:
        if r["readout"] == readout and r["layer"] == layer:
            return r
    return None


def bh_surviving_set(out, block):
    return {(s["readout"], s["layer"])
            for s in out["multiplicity"][block]["surviving"]}


# ---------------------------------------------------------------------------
# C1 -- the primary-readout reporting guide.  Machine-generated so that no
# downstream restatement has to be written by hand.
# ---------------------------------------------------------------------------
def reporting_guide(out):
    sfx = live_rows(out, "suffix")
    ordered = [r for r in sfx if r["ordered"]]
    below = [dict(layer=r["layer"],
                  p_vs_noun_background=r["background_nouns"]["p_vs_background"],
                  p_closed_uniform=r["conventions"]["A"]["p_closed_uniform"],
                  gap_ratio=r["conventions"]["A"]["gap_ratio"],
                  closed_conjunction=bool(r["closed"]))
             for r in sfx if r["background_nouns"]["p_vs_background"] < 0.05]
    pu = [(r["conventions"]["A"]["p_closed_uniform"], r["layer"]) for r in sfx]
    pu_min, pu_min_layer = min(pu)
    return dict(
        readout="suffix (PRIMARY, declarations.R3)",
        n_cells=len(sfx), n_ordered=len(ordered),
        ordered_layers=[r["layer"] for r in ordered],
        over_ordered_cells=dict(
            n=len(ordered),
            gap_ratio=rng_of(r["conventions"]["A"]["gap_ratio"] for r in ordered),
            p_vs_noun_background=rng_of(
                r["background_nouns"]["p_vs_background"] for r in ordered),
            p_closed_uniform=rng_of(
                r["conventions"]["A"]["p_closed_uniform"] for r in ordered)),
        over_all_cells=dict(
            n=len(sfx),
            gap_ratio=rng_of(r["conventions"]["A"]["gap_ratio"] for r in sfx),
            p_vs_noun_background=rng_of(
                r["background_nouns"]["p_vs_background"] for r in sfx),
            p_closed_uniform=rng_of(
                r["conventions"]["A"]["p_closed_uniform"] for r in sfx)),
        cells_beating_the_matched_noun_background_raw=below,
        n_cells_beating_the_matched_noun_background_raw=len(below),
        p_closed_uniform_min=float(pu_min),
        p_closed_uniform_argmin_layer=int(pu_min_layer),
        n_cells_firing_the_R2_conjunction=int(sum(r["closed"] for r in sfx)),
        WRONG_SENTENCE=(
            "gap_ratio 1.78-2.99, p_vs_noun_background 0.025-0.52 -- never below 0.05"),
        WHY_WRONG=(
            "the two ranges are right for the 16 ordered suffix cells, but the "
            "'never below 0.05' clause is attached to the wrong quantity. The matched-"
            "noun background p IS below 0.05 at three suffix layers. Ranks beat the "
            "matched-noun background at the primary readout in three cells; they are "
            "held out of the closure tier only by the weaker uniform-angle tier."),
        CORRECTED_SENTENCE=(
            f"At the primary `suffix` readout the 16 ordered cells run gap_ratio "
            f"{rng_of(r['conventions']['A']['gap_ratio'] for r in ordered)[0]:.2f}-"
            f"{rng_of(r['conventions']['A']['gap_ratio'] for r in ordered)[1]:.2f} with "
            f"p_vs_noun_background "
            f"{rng_of(r['background_nouns']['p_vs_background'] for r in ordered)[0]:.3f}-"
            f"{rng_of(r['background_nouns']['p_vs_background'] for r in ordered)[1]:.3f}. "
            f"Three of the 26 suffix cells DO beat the matched-noun background raw "
            f"(" + ", ".join(f"L{c['layer']} {c['p_vs_noun_background']:.5f}"
                             for c in below) + f"). What never goes below 0.05 at suffix "
            f"is p_closed_uniform (min {pu_min:.5f} at L{pu_min_layer}), so the "
            f"registered R2 conjunction -- p_closed_uniform < 0.05 AND "
            f"p_vs_noun_background < 0.05 -- fires in "
            f"{int(sum(r['closed'] for r in sfx))}/{len(sfx)} suffix cells. The verdict "
            f"is unaffected; the sentence was."),
        do_not_restate="the WRONG_SENTENCE above, in any downstream document")


# ---------------------------------------------------------------------------
# C2 -- attack ledger.  Every R13 attack gets a per-layer outcome and an honest
# held / partially-held / not-held label.
# ---------------------------------------------------------------------------
def exact_rho_tail_k(k):
    """Exact two-sided |Spearman| null over all k! label assignments.

    Data-free: it depends on k alone.  Used to show the k=5 polysemy subset is
    near-powerless -- the achievable p values are quantised and the second-best
    is already above 0.05.
    """
    idx = np.arange(k, dtype=float)
    P = np.asarray(list(permutations(range(k))), dtype=np.int64)
    d = idx[None, :] - idx[P]
    rr = np.abs(1.0 - 6.0 * (d * d).sum(1) / (k * (k * k - 1)))
    vals = np.unique(np.round(rr, 12))[::-1]
    return dict(k=int(k), n_permutations=int(len(P)),
                achievable_p=[dict(rho_abs=float(v),
                                   p_exact=float((rr >= v - 1e-12).mean()),
                                   n_at_or_above=int((rr >= v - 1e-12).sum()))
                              for v in vals[:6]],
                smallest_attainable_p=float(2.0 / len(P)),
                second_smallest_attainable_p=float(
                    (rr >= vals[1] - 1e-12).mean()),
                note=(f"at k={k} the exact-enumeration floor is 2/{len(P)} = "
                      f"{2.0/len(P):.4f} and the SECOND-best attainable value already "
                      f"gives p = {(rr >= vals[1]-1e-12).mean():.4f}. Only a PERFECT "
                      f"monotone subset can reach p < 0.05."))


def attack_ledger(out, headline_band=range(16, 26)):
    sfx = live_rows(out, "suffix")
    ordered = [r for r in sfx if r["ordered"]]
    surv_ord = bh_surviving_set(out, "ordering_exact")

    # --- A1 polysemy subset (5 unambiguously military ranks) ---------------
    mil = [dict(layer=r["layer"],
                rho_ordinal=r["military_only_subset"]["rho_ordinal"],
                p_exact=r["military_only_subset"]["p_ordinal_exact"],
                p_shuffle=r["military_only_subset"]["p_ordinal_shuffle"],
                significant=bool(r["military_only_subset"]["p_ordinal_exact"] < 0.05),
                full_set_rho=r["ordinal"]["rho_abs"],
                full_set_ordered=bool(r["ordered"]))
           for r in sfx if "military_only_subset" in r]
    mil_ord = [m for m in mil if m["full_set_ordered"]]
    mil_sig = [m["layer"] for m in mil_ord if m["significant"]]
    mil_ns = [m for m in mil_ord if not m["significant"]]
    mil_head = [m for m in mil_ord if m["layer"] in headline_band]
    a1 = dict(
        attack="R13(a) polysemy subset -- the 5 unambiguously military ranks",
        words=list(R32.MILITARY_ONLY), k=len(R32.MILITARY_ONLY),
        per_layer_suffix=mil,
        layers_where_full_set_orders=[m["layer"] for m in mil_ord],
        subset_significant_layers=mil_sig,
        subset_not_significant_layers=[m["layer"] for m in mil_ns],
        headline_band=list(headline_band),
        headline_band_rho_range=rng_of(m["rho_ordinal"] for m in mil_head),
        headline_band_p_range=rng_of(m["p_exact"] for m in mil_head),
        power=exact_rho_tail_k(5),
        held="PARTIAL",
        WRONG_CLAIM=("the ordering survives dropping the three non-military-dominant "
                     "ranks"),
        CORRECTED_CLAIM=(
            f"the subset orders PERFECTLY (rho 1.000, exact p 0.0167) and significantly "
            f"at layers {mil_sig}; at the headline layers "
            f"{[m['layer'] for m in mil_head]} it scores rho "
            f"{rng_of(m['rho_ordinal'] for m in mil_head)[0]:.3f}-"
            f"{rng_of(m['rho_ordinal'] for m in mil_head)[1]:.3f} with exact p "
            f"{rng_of(m['p_exact'] for m in mil_head)[0]:.4f}-"
            f"{rng_of(m['p_exact'] for m in mil_head)[1]:.4f}, i.e. NOT significant at "
            f"0.05. POWER CONTEXT so this is not over-penalised: at k=5 the exact null "
            f"is quantised, the floor is 2/120 = 0.01667 and the second-best attainable "
            f"|rho| = 0.9 already gives 10/120 = 0.08333. Nothing short of a perfect "
            f"monotone subset can clear 0.05. The defect is the claim, not the number."))

    # --- A2 template-half sensitivity --------------------------------------
    th = []
    for r in sfx:
        if "template_halves" not in r:
            continue
        h0, h1 = r["template_halves"]
        th.append(dict(layer=r["layer"], full_set_ordered=bool(r["ordered"]),
                       full_rho=r["ordinal"]["rho_abs"],
                       half0_rho=h0["rho_ordinal"], half0_p=h0["p_ordinal_exact"],
                       half0_orders=bool(h0["p_ordinal_exact"] < 0.05),
                       half1_rho=h1["rho_ordinal"], half1_p=h1["p_ordinal_exact"],
                       half1_orders=bool(h1["p_ordinal_exact"] < 0.05),
                       both_halves_order=bool(h0["p_ordinal_exact"] < 0.05
                                              and h1["p_ordinal_exact"] < 0.05)))
    th_ord = [t for t in th if t["full_set_ordered"]]
    both = [t["layer"] for t in th_ord if t["both_halves_order"]]
    only0 = [t["layer"] for t in th_ord if t["half0_orders"] and not t["half1_orders"]]
    only1 = [t["layer"] for t in th_ord if t["half1_orders"] and not t["half0_orders"]]
    a2 = dict(
        attack="R13(d) template-half sensitivity (templates 1-8 vs 9-16)",
        per_layer_suffix=th,
        layers_where_full_set_orders=[t["layer"] for t in th_ord],
        layers_where_both_halves_order=both,
        layers_carried_by_templates_1_8_alone=only0,
        layers_carried_by_templates_9_16_alone=only1,
        held="PARTIAL",
        OMITTED_FROM_THE_ORIGINAL_REPORT=True,
        CORRECTED_CLAIM=(
            f"template sensitivity is a registered R13 attack and was left out of the "
            f"'every attack held' list. Of the {len(th_ord)} ordered suffix cells, both "
            f"halves order independently at {both} only. At {only0} the ordering is "
            f"carried by templates 1-8 alone -- this includes the two headline cells "
            f"L16 and L18, where the second half scores rho "
            f"{[round(t['half1_rho'],3) for t in th_ord if t['layer'] in (16,18)]} "
            f"(p {[round(t['half1_p'],3) for t in th_ord if t['layer'] in (16,18)]}). "
            f"At {only1} it is carried by templates 9-16 alone."))

    # --- A3 composition controls -------------------------------------------
    surv_all = bh_surviving_set(out, "ordering_exact")
    slices = {}
    for name, sel in (
            ("all_ordered_cells", ordered + [r for r in live_rows(out)
                                             if r["ordered"] and r["readout"] != "suffix"]),
            ("all_ordered_suffix_cells", ordered),
            ("bh_surviving_cells", [r for r in live_rows(out) if r["ordered"]
                                    and (r["readout"], r["layer"]) in surv_all]),
            ("bh_surviving_suffix_cells", [r for r in ordered
                                           if ("suffix", r["layer"]) in surv_all])):
        pr = [r["composition_controls"]["partial_rho_coord_vs_rank_given_log_count"]
              for r in sel if "composition_controls" in r]
        pp = [r["composition_controls"]["p_partial_shuffled"]
              for r in sel if "composition_controls" in r]
        slices[name] = dict(n=len(pr), partial_rho=rng_of(pr), p_partial=rng_of(pp))
    l18 = cell(out, "suffix", 18)["composition_controls"]
    a3 = dict(
        attack="R13(c) composition controls (log corpus count, token id, word length)",
        partial_rho_ranges_by_slice=slices,
        WRONG_RANGE="partial rho controlling log corpus count 0.857-0.977, p 0.001-0.025",
        WHY_WRONG="no slicing of the artifact reproduces that interval",
        CORRECTED_RANGES=(
            f"over all {slices['all_ordered_cells']['n']} ordered cells the partial rho "
            f"runs {slices['all_ordered_cells']['partial_rho'][0]:.4f}-"
            f"{slices['all_ordered_cells']['partial_rho'][1]:.4f} "
            f"(p {slices['all_ordered_cells']['p_partial'][0]:.3e}-"
            f"{slices['all_ordered_cells']['p_partial'][1]:.3e}); over the "
            f"{slices['bh_surviving_cells']['n']} BH-surviving cells "
            f"{slices['bh_surviving_cells']['partial_rho'][0]:.4f}-"
            f"{slices['bh_surviving_cells']['partial_rho'][1]:.4f} "
            f"(p {slices['bh_surviving_cells']['p_partial'][0]:.3e}-"
            f"{slices['bh_surviving_cells']['p_partial'][1]:.3e}); over the "
            f"{slices['bh_surviving_suffix_cells']['n']} BH-surviving suffix cells "
            f"the same."),
        bh_set_used="multiplicity.ordering_exact.surviving (the set the track rule uses); "
                    "the registered 1000-draw shuffled-label BH set selects the same 24 "
                    "cells and gives the identical interval",
        substantive_conclusion_unharmed=True,
        evidence_the_ordering_is_not_a_frequency_artifact=dict(
            reference_cell="suffix L18",
            rho_coord_vs_rank=l18["rho_coord_vs_rank"],
            rho_coord_vs_log_corpus_count=l18["rho_coord_vs_log_corpus_count"],
            rho_coord_vs_token_id=l18["rho_coord_vs_token_id"],
            rho_coord_vs_word_length_chars=l18["rho_coord_vs_word_length_chars"],
            partial_rho=l18["partial_rho_coord_vs_rank_given_log_count"]),
        held="HELD")

    # --- A4 jackknife -------------------------------------------------------
    # Criterion stated before it is applied: the attack HOLDS if, in every
    # BH-surviving suffix cell, all 8 leave-one-out variants still order at
    # p < 0.05 on the registered 1000-draw shuffled-label null.
    jk = out["jackknife"]
    jk_sfx = [j for j in jk if j["readout"] == "suffix"]
    jk_bh = [j for j in jk_sfx if ("suffix", j["layer"]) in surv_ord]
    per_cell = [dict(readout=j["readout"], layer=j["layer"],
                     n_still_ordered_of_8=j["n_still_ordered"],
                     bh_surviving=bool((j["readout"], j["layer"]) in surv_ord),
                     worst_dropped=j["worst"]["dropped"],
                     worst_p=j["worst"]["p_ordinal_shuffle"],
                     breaking_drops=[e["dropped"] for e in j["leave_one_out"]
                                     if e["p_ordinal_shuffle"] >= 0.05])
                for j in jk]
    weak_bh = [c for c in per_cell
               if c["bh_surviving"] and c["readout"] == "suffix"
               and c["n_still_ordered_of_8"] < 8]
    a4 = dict(
        attack="R13(b) leave-one-out jackknife of the ordinal statistic",
        criterion="HOLDS iff every BH-surviving suffix cell keeps all 8 leave-one-out "
                  "variants ordering at p < 0.05 (registered 1000-draw null)",
        n_ordered_cells_jackknifed=len(jk),
        n_suffix_cells=len(jk_sfx),
        n_bh_surviving_suffix_cells=len(jk_bh),
        n_bh_surviving_suffix_cells_at_8_of_8=int(sum(
            j["n_still_ordered"] == 8 for j in jk_bh)),
        per_cell=per_cell,
        bh_surviving_suffix_cells_below_8_of_8=weak_bh,
        power_note=("at k=7 the exact null has 5040 orderings and a floor of 4.0e-4, so "
                    "unlike the k=5 polysemy subset this attack is NOT power-limited: a "
                    "leave-one-out variant that fails to order has genuinely lost the "
                    "ordering"),
        held="PARTIAL" if weak_bh else "HELD",
        NOT_FLAGGED_BY_THE_VERIFIER=True,
        FOUND_HERE=(
            f"a third registered attack that does not hold outright, found by "
            f"recomputation rather than by the verification report. "
            f"{int(sum(j['n_still_ordered'] == 8 for j in jk_bh))}/{len(jk_bh)} "
            f"BH-surviving suffix cells keep all 8 leave-one-out variants ordering. The "
            f"exceptions are "
            + "; ".join(f"L{c['layer']} {c['n_still_ordered_of_8']}/8 "
                        f"(breaks on {', '.join(c['breaking_drops'])})"
                        for c in weak_bh)
            + ". The headline cell L16 is the weakest of them. Every `final`-readout "
              "cell and every suffix cell from L18 up is 8/8."))

    # --- A5 capitalisation --------------------------------------------------
    cap = [dict(layer=r["layer"], rho=r["capitalised_variant"]["rho_ordinal"],
                p_exact=r["capitalised_variant"]["p_ordinal_exact"],
                orders=bool(r["capitalised_variant"]["p_ordinal_exact"] < 0.05),
                gap_ratio=r["capitalised_variant"]["gap_ratio"],
                p_closed_uniform=r["capitalised_variant"]["p_closed_uniform"])
           for r in sfx if "capitalised_variant" in r]
    cap_ord = [c for c in cap if c["layer"] in [r["layer"] for r in ordered]]
    a5 = dict(
        attack="R13(e) capitalisation sensitivity ('{W}' surface)",
        per_layer_suffix=cap,
        n_ordered_cells_where_capitalised_also_orders=int(sum(c["orders"] for c in cap_ord)),
        n_ordered_cells=len(cap_ord),
        capitalised_closure_min_p_uniform=float(min(
            [c["p_closed_uniform"] for c in cap], default=1.0)),
        held="HELD" if sum(c["orders"] for c in cap_ord) >= 0.8 * max(len(cap_ord), 1)
             else "PARTIAL")

    # --- A6 widened-plane stress -------------------------------------------
    live = live_rows(out)
    stress_cells = [dict(readout=r["readout"], layer=r["layer"],
                         p_vs_noun_background=r["secondary_plane_search"][
                             "p_vs_noun_background"],
                         selected_plane=r["secondary_plane_search"]["selected_plane"],
                         already_closed_on_the_primary_plane=bool(r["closed"]))
                    for r in live if r["closed_secondary_stress"]]
    a6 = dict(
        attack="R13(f) widened-plane closure stress tier (selection-invalid by design)",
        n_cells_closing_stress=len(stress_cells), n_cells=len(live),
        cells=stress_cells,
        strict_subset_of_the_primary_closers=bool(
            all(c["already_closed_on_the_primary_plane"] for c in stress_cells)),
        n_suffix_cells_closing_stress=int(sum(c["readout"] == "suffix"
                                              for c in stress_cells)),
        held="HELD",
        why=("the stress tier -- which can only make closure MORE likely, since the "
             "plane is selected on the observed cloud -- closes in fewer cells than the "
             "pre-declared top-2 plane, every one of them already a primary closer at "
             "the tertiary `final` readout, and none at the primary `suffix` readout. "
             "It found nothing the declared plane missed."))

    held = [a["held"] for a in (a1, a2, a3, a4, a5, a6)]
    return dict(
        registered_attacks=out["declarations"]["R13_attacks"],
        A1_polysemy_subset=a1, A2_template_halves=a2, A3_composition_controls=a3,
        A4_jackknife=a4, A5_capitalisation=a5, A6_widened_plane_stress=a6,
        n_held=int(sum(h.startswith("HELD") for h in held)),
        n_partial=int(sum(h == "PARTIAL" for h in held)),
        every_attack_held=bool(all(h.startswith("HELD") for h in held)),
        WRONG_CLAIM="every attack held",
        CORRECTED_CLAIM=(
            f"{int(sum(h.startswith('HELD') for h in held))} of the 6 registered R13 "
            f"attacks held outright ({', '.join(n for n, aa in (('composition controls', a3), ('capitalisation', a5), ('widened-plane stress', a6)) if aa['held'].startswith('HELD'))}); "
            f"{int(sum(h == 'PARTIAL' for h in held))} held only in part. "
            f"(1) The k=5 polysemy subset is not significant at the headline layers "
            f"{[m['layer'] for m in mil_head]} -- though at k=5 it is near-powerless "
            f"there by construction, the second-best attainable p being 0.0833. "
            f"(2) Template halves: the ordering at "
            f"{a2['layers_carried_by_templates_1_8_alone']} is carried by templates 1-8 "
            f"alone, which includes the headline cells L16 and L18. "
            f"(3) Leave-one-out: {len(weak_bh)} of the {len(jk_bh)} BH-surviving suffix "
            f"cells lose the ordering under at least one drop ("
            + "; ".join(f"L{c['layer']} {c['n_still_ordered_of_8']}/8 on "
                        f"{', '.join(c['breaking_drops'])}" for c in weak_bh)
            + f"), the headline cell L16 worst of them. "
            f"None of the three moves the verdict: the ordering claim rests on the full "
            f"8-word set against its own shuffled-label null, which survives BH in 12 "
            f"suffix cells and is 8/8 under leave-one-out at every suffix layer from 18 "
            f"up and at every `final` layer."),
        WHERE_THE_ORDERING_IS_MOST_ROBUST=(
            "suffix L18-L25: BH-surviving, 8/8 under leave-one-out, both template halves "
            "order at L20-L25, capitalised surface orders, partial rho above 0.91"))


# ---------------------------------------------------------------------------
# C3 -- the alarm, both readings.
# ---------------------------------------------------------------------------
def alarm_dual_reading(eager, sdpa):
    def raw_closing(out):
        return [dict(readout=r["readout"], layer=r["layer"],
                     gap_ratio=r["conventions"]["A"]["gap_ratio"],
                     p_uniform=r["conventions"]["A"]["p_closed_uniform"],
                     p_noun_background=r["background_nouns"]["p_vs_background"],
                     rho_ordinal=r["ordinal"]["rho_abs"])
                for r in live_rows(out) if r["closed"]]
    e_raw, s_raw = raw_closing(eager), raw_closing(sdpa)
    ceil8 = eager["closure_diagnostic"][
        "monotonicity_ceiling_of_a_perfect_ring"]["8"]["max_abs_ordinal_rho"] \
        if "8" in eager["closure_diagnostic"]["monotonicity_ceiling_of_a_perfect_ring"] \
        else eager["closure_diagnostic"][
            "monotonicity_ceiling_of_a_perfect_ring"][8]["max_abs_ordinal_rho"]
    return dict(
        why_this_block_exists=(
            "prereg 3.5's ranks verdict table reads 'Closure passes (with or without "
            "ordering) -> prediction FAILED', unconditional on multiplicity. Sections "
            "4.4 and 7 of the same frozen document mandate BH context and forbid "
            "reporting an isolated single-cell pass as a finding. The two clauses "
            "conflict when raw closure passes but nothing survives BH. The run resolved "
            "the conflict toward BH (declarations.R9_verdict_rule.track_rule: '1) alarm "
            "if any closing cell survives BH'). That narrowing was declared before the "
            "first forward pass (R15), but it is the more lenient of the two available "
            "readings and a safety alarm was therefore graded on the same curve as a "
            "positive claim. Both readings are reported here as data."),
        strict_prereg_3_5_reading=dict(
            rule="any raw closure pass fires the alarm, unconditional on multiplicity",
            eager_n_raw_closing=len(e_raw), eager_would_trigger=bool(e_raw),
            sdpa_n_raw_closing=len(s_raw), sdpa_would_trigger=bool(s_raw),
            eager_cells=e_raw, sdpa_cells=s_raw,
            note_eager=("the 8 EAGER raw passes are NOT isolated singletons -- all 8 are "
                        "at the tertiary `final` readout and they sit in contiguous "
                        "layer bands -- which is why section 4.4's 'isolated pass' "
                        "escape does not straightforwardly apply to them"),
            note_sdpa=("the SDPA raw passes are NOT the same set and must not be "
                       "described with the eager sentence: "
                       + "; ".join(f"{c['readout']} L{c['layer']}" for c in s_raw)
                       + ". Six are `final`-readout band cells, but suffix L19 is an "
                         "ISOLATED SINGLETON at the PRIMARY readout, so under sdpa "
                         "section 4.4's 'isolated pass is not a finding' clause applies "
                         "to it directly. Its |rho_ordinal| is above the "
                         "perfect-8-cycle ceiling, i.e. it is a monotone arc")),
        bh_graded_reading_AS_RUN=dict(
            rule="alarm only if a closing cell survives BH within the 78-cell sweep",
            eager_n_surviving_bh=eager["alarm"]["n_cells_closing_surviving_bh"],
            sdpa_n_surviving_bh=sdpa["alarm"]["n_cells_closing_surviving_bh"],
            triggered=False),
        mitigations_verified=dict(
            declared_pre_activation="R15 states R1-R14 were fixed before the first "
                                    "forward pass; the track_rule is in R9",
            raw_count_reported_prominently=True,
            monotone_arc_share=dict(
                perfect_8_cycle_ceiling=float(ceil8),
                eager_n_above_ceiling=int(sum(c["rho_ordinal"] > ceil8 for c in e_raw)),
                eager_n_raw=len(e_raw),
                sdpa_n_above_ceiling=int(sum(c["rho_ordinal"] > ceil8 for c in s_raw)),
                sdpa_n_raw=len(s_raw),
                reading="a cell above the ceiling cannot be a closed cycle in label "
                        "order whatever the max-gap statistic says"),
            numerically_unstable="see kernel_robustness: the raw closure decisions flip "
                                 "in BOTH directions on a change of attention kernel"),
        adjudication=(
            "the no-alarm verdict STANDS and is not overturned here: nothing survives BH "
            "under either kernel, and most raw-closing cells are provably monotone arcs. "
            "But the synthesizer should know that under the strict section-3.5 reading "
            "the raw passes would have fired the alarm, and that the count itself is "
            "kernel-dependent."),
        track_verdict_unchanged=True)


# ---------------------------------------------------------------------------
# C4 -- cross-kernel census.
# ---------------------------------------------------------------------------
def kernel_robustness(eager, sdpa, eager_sec, sdpa_sec):
    bh_e = {(x["readout"], x["layer"])
            for x in eager["multiplicity"]["ordering_exact"]["surviving"]}
    bh_s = {(x["readout"], x["layer"])
            for x in sdpa["multiplicity"]["ordering_exact"]["surviving"]}
    cells, flips = [], []
    for e in live_rows(eager):
        s = cell(sdpa, e["readout"], e["layer"])
        if s is None or s["degenerate"]:
            continue
        ea, sa = e["conventions"]["A"], s["conventions"]["A"]
        rec = dict(
            readout=e["readout"], layer=e["layer"],
            gap_ratio_eager=ea["gap_ratio"], gap_ratio_sdpa=sa["gap_ratio"],
            gap_ratio_rel_delta=float(abs(sa["gap_ratio"] - ea["gap_ratio"])
                                      / max(ea["gap_ratio"], 1e-12)),
            p_uniform_eager=ea["p_closed_uniform"], p_uniform_sdpa=sa["p_closed_uniform"],
            p_noun_bg_eager=e["background_nouns"]["p_vs_background"],
            p_noun_bg_sdpa=s["background_nouns"]["p_vs_background"],
            rho_ordinal_eager=e["ordinal"]["rho_abs"],
            rho_ordinal_sdpa=s["ordinal"]["rho_abs"],
            closed_eager=bool(e["closed"]), closed_sdpa=bool(s["closed"]),
            ordered_eager=bool(e["ordered"]), ordered_sdpa=bool(s["ordered"]),
            sdpa_gap_ratio_vs_eager_background_p05=bool(
                sa["gap_ratio"] <= e["background_nouns"]["gap_ratio_p05"]))
        cells.append(rec)
        if rec["closed_eager"] != rec["closed_sdpa"]:
            flips.append(dict(**{k: rec[k] for k in (
                "readout", "layer", "gap_ratio_eager", "gap_ratio_sdpa",
                "p_uniform_eager", "p_uniform_sdpa", "p_noun_bg_eager",
                "p_noun_bg_sdpa", "closed_eager", "closed_sdpa")},
                direction="GAINED closure under sdpa" if rec["closed_sdpa"]
                          else "LOST closure under sdpa"))
    rel = [c["gap_ratio_rel_delta"] for c in cells]
    ord_flips = [c for c in cells if c["ordered_eager"] != c["ordered_sdpa"]]
    return dict(
        what_changed="attn_implementation only: 'eager' (gfdisc.model.load_model, the "
                     "loader every Phase-2 track shares) vs 'sdpa'. Same bf16 weights, "
                     "same device, same prompts, same seeds, same readouts, same "
                     "background pools, same code path (the ORIGINAL main() was called "
                     "twice through a loader whose only free parameter is the kernel).",
        n_cells_compared=len(cells),
        gap_ratio_relative_delta=dict(
            mean=float(np.mean(rel)), median=float(np.median(rel)),
            p95=float(np.percentile(rel, 95)), max=float(np.max(rel)),
            max_cell=max(cells, key=lambda c: c["gap_ratio_rel_delta"])),
        closure_decision_flips=flips,
        n_closure_decision_flips=len(flips),
        ordering_decision_flips=[dict(readout=c["readout"], layer=c["layer"],
                                      rho_eager=c["rho_ordinal_eager"],
                                      rho_sdpa=c["rho_ordinal_sdpa"],
                                      ordered_eager=c["ordered_eager"],
                                      ordered_sdpa=c["ordered_sdpa"],
                                      bh_surviving_eager=bool(
                                          (c["readout"], c["layer"]) in bh_e),
                                      bh_surviving_sdpa=bool(
                                          (c["readout"], c["layer"]) in bh_s))
                                 for c in ord_flips],
        n_ordering_decision_flips=len(ord_flips),
        bh_surviving_ordering_set=dict(
            n_eager=len(bh_e), n_sdpa=len(bh_s),
            identical_cell_for_cell=bool(bh_e == bh_s),
            only_eager=sorted(f"{r} L{l}" for r, l in bh_e - bh_s),
            only_sdpa=sorted(f"{r} L{l}" for r, l in bh_s - bh_e),
            n_raw_ordering_flips_that_are_bh_cells=int(sum(
                ((c["readout"], c["layer"]) in bh_e) or
                ((c["readout"], c["layer"]) in bh_s) for c in ord_flips)),
            reading=(
                "THE ASYMMETRY THAT CALIBRATES THE REST OF PHASE 2: the BH-surviving "
                "ORDERING set is identical cell for cell across the two kernels "
                f"({len(bh_e)} cells, same layers, same readouts), while the raw "
                "CLOSURE decisions flip in both directions. The ordering tier is "
                "numerically stable at BH; the closure tier is not stable even raw. "
                "The three raw ordering flips (suffix L8, L11, L15) are all cells that "
                "survive BH under NEITHER kernel.")),
        n_raw_closing_eager=int(sum(c["closed_eager"] for c in cells)),
        n_raw_closing_sdpa=int(sum(c["closed_sdpa"] for c in cells)),
        n_bh_surviving_closure_eager=eager["alarm"]["n_cells_closing_surviving_bh"],
        n_bh_surviving_closure_sdpa=sdpa["alarm"]["n_cells_closing_surviving_bh"],
        primary_suffix_readout=dict(
            n_closing_eager=int(sum(c["closed_eager"] for c in cells
                                    if c["readout"] == "suffix")),
            n_closing_sdpa=int(sum(c["closed_sdpa"] for c in cells
                                   if c["readout"] == "suffix")),
            cells=[c for c in cells if c["readout"] == "suffix"
                   and (c["closed_eager"] or c["closed_sdpa"]
                        or c["p_uniform_sdpa"] < 0.05 or c["p_uniform_eager"] < 0.08)],
            headline=(
                f"THE MOST CONSEQUENTIAL SINGLE NUMBER IN THIS BLOCK: under eager the "
                f"PRIMARY `suffix` readout closes in "
                f"{int(sum(c['closed_eager'] for c in cells if c['readout'] == 'suffix'))}"
                f"/26 cells; under sdpa it closes in "
                f"{int(sum(c['closed_sdpa'] for c in cells if c['readout'] == 'suffix'))}"
                f"/26. The registered R2 conjunction therefore fires at the PRIMARY "
                f"readout under one kernel and not the other, from the same weights and "
                f"the same prompts. Both p values in that cell are computed against the "
                f"SAME-KERNEL background (the clean comparison), not against the other "
                f"kernel's draws."),
            monotone_arc_check_on_the_sdpa_suffix_closers=[
                dict(layer=c["layer"], rho_ordinal_sdpa=c["rho_ordinal_sdpa"],
                     exceeds_perfect_8_cycle_ceiling=bool(c["rho_ordinal_sdpa"] > 0.7619))
                for c in cells if c["readout"] == "suffix" and c["closed_sdpa"]],
            monotone_arc_note=(
                "a suffix cell that closes under sdpa while scoring |rho_ordinal| above "
                "the 0.7619 ceiling a perfect 8-cycle can reach is still a monotone ARC "
                "that the max-gap statistic failed to reject -- so the kernel flip does "
                "not turn ranks into a ring, it shows the closure tier accepting an arc "
                "under one numerical path and rejecting it under another")),
        per_cell=cells,
        reading=(
            "a raw closure pass at k = 8 is both MULTIPLICITY-fragile and NUMERICALLY "
            "fragile. Changing nothing but the attention kernel moves gap_ratio at the "
            "tens-of-percent scale and flips raw closure decisions in both directions. "
            "Consequently: (i) 'three full runs byte-identical' demonstrates determinism "
            "of ONE code path, not robustness, and must not be quoted as the latter; "
            "(ii) the calibration rule handed to every other Phase-2 track is "
            "'a raw closure pass at k=8 is both multiplicity-fragile and numerically "
            "fragile -- only BH-surviving closure counts', which is strictly stronger "
            "than 'a raw single-cell closure pass at k=8 is not evidence'."),
        verdict_unchanged="nothing survives BH under either kernel")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/phase2_ranks_corrected.json")
    ap.add_argument("--sdpa-out", default="results/phase2_ranks_sdpa_census.json")
    ap.add_argument("--scratch", default=None)
    ap.add_argument("--skip-sdpa", action="store_true")
    ap.add_argument("--reuse-sdpa", action="store_true",
                    help="load an sdpa census already on disk at --sdpa-out instead of "
                         "re-running it. Only valid when that file was produced by this "
                         "same harness.")
    ap.add_argument("--reuse-eager", action="store_true",
                    help="dry-run switch: build the correction blocks from the published "
                         "artifact instead of re-running the eager sweep. Never used for "
                         "the artifact this file writes.")
    a = ap.parse_args()
    t0 = time.time()
    scratch = Path(a.scratch) if a.scratch else ROOT / "results"
    scratch.mkdir(parents=True, exist_ok=True)
    eager_path = scratch / "phase2_ranks_eager_reproduce.json"

    print("=" * 84)
    print("PHASE 2 RANKS -- CORRECTED ARTIFACT (scripts/32b_phase2_ranks_corrected.py)")
    print("the original script and artifact are NOT modified; both are preserved.")
    print("no statistic, null, threshold or verdict rule is changed.")
    print("=" * 84, flush=True)

    original = json.loads(ORIGINAL.read_text())

    if a.reuse_eager:
        src = eager_path if eager_path.exists() else ORIGINAL
        print(f"\n[1/4] --reuse-eager: loading {src} instead of re-running the eager "
              f"sweep (no forward passes)", flush=True)
        if src is ORIGINAL:
            print("      NOTE: no saved re-run on disk, so the C0 reproduction check "
                  "below is a self-comparison and proves nothing. DRY RUN ONLY.",
                  flush=True)
        eager = json.loads(src.read_text())
        t_eager = float(eager.get("runtime_sec", 0.0))
    else:
        print("\n[1/4] re-running the ORIGINAL main() with attn_implementation='eager' ...",
              flush=True)
        eager, t_eager = run_sweep("eager", eager_path,
                                   scratch / "log_ranks_eager.txt", RUN_BACKGROUND_DRAWS)
        print(f"      done in {t_eager:.1f}s", flush=True)

    cmp_state = deep_compare(original, eager)
    reproduced = (cmp_state["max_abs"] == 0.0
                  and cmp_state["n_mismatched_non_numeric"] == 0)
    print(f"      reproduction vs results/phase2_ranks.json: "
          f"{cmp_state['n_numeric']} numeric fields compared, max |delta| = "
          f"{cmp_state['max_abs']:.3e} at {cmp_state['max_path']}, "
          f"{cmp_state['n_mismatched_non_numeric']} non-numeric mismatches -> "
          f"{'IDENTICAL' if reproduced else 'DIFFERS'}", flush=True)
    if not reproduced:
        print("      !! the eager pass does NOT reproduce the artifact. Cross-kernel "
              "deltas below are NOT attributable to the kernel alone.", flush=True)
        for m in cmp_state["mismatches"][:10]:
            print("        ", m)

    sdpa, t_sdpa = None, 0.0
    if a.reuse_sdpa:
        print(f"\n[2/4] --reuse-sdpa: loading the census already at {a.sdpa_out}",
              flush=True)
        sdpa = json.loads(Path(a.sdpa_out).read_text())
        t_sdpa = float(sdpa.get("runtime_sec", 0.0))
    elif not a.skip_sdpa:
        print("\n[2/4] re-running the SAME main() with attn_implementation='sdpa' ...",
              flush=True)
        sdpa, t_sdpa = run_sweep("sdpa", Path(a.sdpa_out),
                                 scratch / "log_ranks_sdpa.txt", RUN_BACKGROUND_DRAWS)
        print(f"      done in {t_sdpa:.1f}s -> {a.sdpa_out}", flush=True)

    print("\n[3/4] building correction blocks ...", flush=True)
    guide = reporting_guide(eager)
    ledger = attack_ledger(eager)
    kern = kernel_robustness(eager, sdpa, None, None) if sdpa else None
    dual = alarm_dual_reading(eager, sdpa) if sdpa else None

    out = dict(eager)                                   # every original number, verbatim
    out["item"] = "phase2_ranks_corrected"
    out["script"] = Path(__file__).name
    out["supersedes"] = dict(
        artifact="results/phase2_ranks.json",
        script="scripts/32_phase2_ranks.py",
        relationship="every computed field in this file is the byte-identical output of "
                     "that script's own main(), re-run through a loader whose only free "
                     "parameter is the attention kernel. The original files are "
                     "preserved unmodified for audit. Read THIS file's prose; read that "
                     "file only to confirm the numbers match.")
    out["generated"] = time.strftime("%Y-%m-%dT%H:%M:%S")

    out["corrections"] = dict(
        provenance="ADDED AFTER verification of results/phase2_ranks.json. Descriptive "
                   "only: no statistic, null, threshold, label set, template, readout or "
                   "verdict rule was changed, and the track verdict is unchanged.",
        C0_reproduction=dict(
            claim="the eager pass through this harness reproduces the published artifact",
            n_numeric_fields_compared=cmp_state["n_numeric"],
            max_abs_deviation=cmp_state["max_abs"],
            max_deviation_path=cmp_state["max_path"],
            n_non_numeric_mismatches=cmp_state["n_mismatched_non_numeric"],
            mismatch_examples=cmp_state["mismatches"][:20],
            ignored_keys=sorted(IGNORE_KEYS),
            reproduced=bool(reproduced),
            why_it_matters="without this, a cross-kernel delta could be an artefact of "
                           "this harness rather than of the kernel"),
        C1_primary_readout_prose=guide,
        C2_attack_ledger=ledger,
        C3_prereg_deviation_background_draws=dict(
            prereg_section="4.2(b) of " + PREREG,
            registered=PREREG_BACKGROUND_DRAWS,
            used=RUN_BACKGROUND_DRAWS,
            where="scripts/32_phase2_ranks.py:293 (N_BACKGROUND)",
            annotated_in_the_original_artifact=False,
            direction="CONSERVATIVE -- more draws give finer p resolution at the same "
                      "nominal alpha; it cannot manufacture a pass",
            run_wide=True,
            sibling_precedent="scripts/31_phase2_compass.py:270 annotates the same "
                              "constant with 'C5, and the recorded prereg discrepancy'; "
                              "all five sibling Phase-2 scripts use 2000",
            classification="missing provenance annotation, not a methodological change",
            fix="this block is the annotation"),
        C4_alarm_two_readings=dual,
        C5_kernel_robustness=kern,
        C6_reproducibility_claim=dict(
            WRONG_CLAIM="three full runs byte-identical, therefore the result is robust",
            present_in_the_original_artifact=False,
            note="the claim appeared in the report prose, not in phase2_ranks.json",
            CORRECTED_CLAIM="repeat runs of the same code path are byte-identical, which "
                            "demonstrates DETERMINISM of one code path. Robustness is a "
                            "different property and is measured in "
                            "corrections.C5_kernel_robustness, where it does not hold "
                            "for raw k=8 closure."),
        C7_synthetic_power_caveat=dict(
            what_the_battery_shows="the closure tier fires on a CLEAN planted k=8 ring "
                                   "out of 1152-d isotropic noise",
            what_it_does_NOT_show="power to discriminate a ring from a monotone ARC, "
                                  "which is exactly what bit at the tertiary `final` "
                                  "readout here. PCA recovers a planted 2-D ring out of "
                                  "1152-d isotropic noise at essentially any SNR, so a "
                                  "planted-ring ladder firing at high noise is weak "
                                  "evidence of discriminating power.",
            the_discriminating_control="planted_monotone_arc in `synthetics`, which must "
                                       "order and must NOT close",
            planted_monotone_arc=eager["synthetics"]["planted_monotone_arc"],
            planted_ring_k8=eager["synthetics"]["planted_ring_k8"]),
        C8_minor_corrections_to_the_verification_report=[
            dict(item="suffix L19 p_closed_uniform",
                 quoted_by_verifier=0.05697,
                 artifact_value=cell(eager, "suffix", 19)[
                     "conventions"]["A"]["p_closed_uniform"],
                 note="the artifact value is the one above; the verifier's figure is "
                      "presumably from their own re-run. The substantive point -- that "
                      "the uniform tier, not the background tier, is what holds suffix "
                      "out of the conjunction -- is unaffected."),
            dict(item="military_only_subset significant layers",
                 quoted_by_verifier="L8 and L11-L15",
                 recomputed=ledger["A1_polysemy_subset"]["subset_significant_layers"],
                 note="the verifier's list is right for the layers where the FULL set "
                      "orders. The subset is also significant at suffix L1, where the "
                      "full set does not order (rho 0.190) -- noise, not a finding.")],
    )

    # ---- amended verdict prose (the numeric verdict fields are untouched) ---
    v = dict(out["verdict"])
    v["summary_corrected"] = (
        v["summary"] + "; the primary suffix readout beats the matched-noun background "
        f"raw in {guide['n_cells_beating_the_matched_noun_background_raw']} cells "
        f"(L{', L'.join(str(c['layer']) for c in guide['cells_beating_the_matched_noun_background_raw'])}) "
        f"and is held out of the closure tier by the uniform-angle tier alone "
        f"(min p_closed_uniform {guide['p_closed_uniform_min']:.5f} at "
        f"L{guide['p_closed_uniform_argmin_layer']})")
    if kern:
        v["caveat_for_every_other_phase2_track_corrected"] = (
            v["caveat_for_every_other_phase2_track"].replace(
                "Read as: a RAW single-cell closure pass at k = 8 is not evidence in "
                "Phase 2; only BH-surviving closure is.",
                f"Read as: a RAW closure pass at k = 8 is both MULTIPLICITY-fragile and "
                f"NUMERICALLY fragile -- changing only the attention kernel "
                f"(eager -> sdpa, same weights, same prompts, same seeds) moves "
                f"gap_ratio by up to "
                f"{kern['gap_ratio_relative_delta']['max']*100:.0f}% and flips "
                f"{kern['n_closure_decision_flips']} raw closure decisions in both "
                f"directions ({kern['n_raw_closing_eager']} raw closing cells under "
                f"eager, {kern['n_raw_closing_sdpa']} under sdpa). Under sdpa the "
                f"PRIMARY `suffix` readout closes raw in "
                f"{kern['primary_suffix_readout']['n_closing_sdpa']}/26 cells rather "
                f"than {kern['primary_suffix_readout']['n_closing_eager']}/26, so the "
                f"'0/26 at the primary readout' figure above is itself kernel-specific. "
                f"ONLY BH-SURVIVING CLOSURE COUNTS, and nothing survives BH under "
                f"either kernel."))
    v["attack_summary_corrected"] = ledger["CORRECTED_CLAIM"]
    v["prose_defects_fixed"] = ["C1 primary-readout background p", "C2 attack ledger",
                                "C3 prereg background-draw deviation",
                                "C4 alarm two readings", "C5 kernel robustness",
                                "C6 reproducibility claim", "C7 synthetic power caveat"]
    v["numeric_verdict_unchanged"] = True
    out["verdict"] = v

    # annotate the declaration in place (in the corrected copy only)
    out["declarations"] = dict(out["declarations"])
    r5 = dict(out["declarations"]["R5_nulls"])
    r5["background_draws_prereg_registered"] = PREREG_BACKGROUND_DRAWS
    r5["background_draws_deviation_note"] = (
        f"prereg 4.2(b) registers {PREREG_BACKGROUND_DRAWS} draws; this run used "
        f"{RUN_BACKGROUND_DRAWS}, run-wide across all six Phase-2 scripts. Conservative "
        f"(finer p resolution, same nominal alpha). Recorded here because the original "
        f"artifact reported the count without noting the difference from the frozen doc.")
    out["declarations"]["R5_nulls"] = r5

    out["corrections"]["SUMMARY_FOR_THE_SYNTHESIZER"] = [
        f"C1. At the PRIMARY `suffix` readout the ranks DO beat the matched-noun "
        f"background raw at three layers ("
        + ", ".join(f"L{c['layer']} p={c['p_vs_noun_background']:.5f}"
                    for c in guide["cells_beating_the_matched_noun_background_raw"])
        + f"). What never goes below 0.05 there is p_closed_uniform "
          f"(min {guide['p_closed_uniform_min']:.5f} at "
          f"L{guide['p_closed_uniform_argmin_layer']}), so the registered conjunction "
          f"fires {guide['n_cells_firing_the_R2_conjunction']}/{guide['n_cells']}. Do "
          f"not say ranks never beat the background at the primary readout.",
        f"C2. {ledger['n_held']}/6 registered R13 attacks held outright, "
        f"{ledger['n_partial']}/6 in part. 'Every attack held' is false.",
        f"C3. Background draws: prereg 4.2(b) registers {PREREG_BACKGROUND_DRAWS}, the "
        f"run used {RUN_BACKGROUND_DRAWS} run-wide. Conservative; now annotated.",
        (f"C4. Under the strict prereg-3.5 reading the alarm WOULD fire on "
         f"{dual['strict_prereg_3_5_reading']['eager_n_raw_closing']} raw eager cells; "
         f"the run graded it on BH, where nothing survives. Verdict stands, but a "
         f"safety alarm was graded on the lenient reading."
         if dual else "C4. not computed"),
        (f"C5. Raw k=8 closure is NUMERICALLY fragile: eager->sdpa moves gap_ratio by up "
         f"to {kern['gap_ratio_relative_delta']['max']*100:.0f}% and flips "
         f"{kern['n_closure_decision_flips']} raw closure decisions in both directions; "
         f"the PRIMARY suffix readout closes "
         f"{kern['primary_suffix_readout']['n_closing_eager']}/26 under eager and "
         f"{kern['primary_suffix_readout']['n_closing_sdpa']}/26 under sdpa. Nothing "
         f"survives BH under either kernel. By contrast the BH-surviving ORDERING set "
         f"is identical cell for cell across kernels "
         f"({kern['bh_surviving_ordering_set']['n_eager']} cells, same layers) -- the "
         f"ordering tier is numerically stable at BH, the closure tier is not stable "
         f"even raw."
         if kern else "C5. not computed"),
        "C6. 'Byte-identical repeat runs' is determinism of one code path, not "
        "robustness; do not quote it as robustness.",
        "C7. The planted-ring battery shows power against a CLEAN ring, not power to "
        "discriminate a ring from a monotone arc -- which is what bit at `final`.",
        "VERDICT UNCHANGED: supervised ordered_not_closed, blind untestable, no alarm. "
        "Calibration handed to every other Phase-2 track: a raw closure pass at k=8 is "
        "both multiplicity-fragile and numerically fragile; only BH-surviving closure "
        "counts.",
    ]
    out["runtime_sec_corrected_pass"] = float(time.time() - t0)
    out["runtime_note"] = (
        "runtime_sec is the eager sweep's own timer. runtime_sec_eager_sweep / "
        "runtime_sec_sdpa_sweep are the two full 78-cell sweeps. "
        "runtime_sec_corrected_pass covers only this file's block-building step and is "
        "near zero when the sweeps are reused from disk; the end-to-end run that "
        "produced both sweeps took the sum of the two sweep times plus that step.")
    out["runtime_sec_eager_sweep"] = float(t_eager)
    out["runtime_sec_sdpa_sweep"] = float(t_sdpa)

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=2, default=R32.jsonable))

    # ---- console summary ---------------------------------------------------
    print("\n[4/4] CORRECTED NUMBERS")
    print("-" * 84)
    print("C1 primary `suffix` readout:")
    print(f"   ordered cells {guide['n_ordered']}/{guide['n_cells']} at layers "
          f"{guide['ordered_layers']}")
    print(f"   gap_ratio (ordered cells)          "
          f"{guide['over_ordered_cells']['gap_ratio'][0]:.4f} .. "
          f"{guide['over_ordered_cells']['gap_ratio'][1]:.4f}")
    print(f"   p_vs_noun_background (ordered)     "
          f"{guide['over_ordered_cells']['p_vs_noun_background'][0]:.5f} .. "
          f"{guide['over_ordered_cells']['p_vs_noun_background'][1]:.5f}")
    print(f"   cells BELOW 0.05 on that p:        "
          + ", ".join(f"L{c['layer']} {c['p_vs_noun_background']:.5f}"
                      for c in guide["cells_beating_the_matched_noun_background_raw"]))
    print(f"   min p_closed_uniform               "
          f"{guide['p_closed_uniform_min']:.5f} at L{guide['p_closed_uniform_argmin_layer']}")
    print(f"   R2 conjunction fires               "
          f"{guide['n_cells_firing_the_R2_conjunction']}/{guide['n_cells']}")
    print("\nC2 attack ledger: "
          f"{ledger['n_held']} held, {ledger['n_partial']} partial "
          f"(every_attack_held = {ledger['every_attack_held']})")
    print(f"   polysemy subset significant at layers "
          f"{ledger['A1_polysemy_subset']['subset_significant_layers']}; headline band "
          f"rho {ledger['A1_polysemy_subset']['headline_band_rho_range'][0]:.3f}-"
          f"{ledger['A1_polysemy_subset']['headline_band_rho_range'][1]:.3f}, p "
          f"{ledger['A1_polysemy_subset']['headline_band_p_range'][0]:.4f}-"
          f"{ledger['A1_polysemy_subset']['headline_band_p_range'][1]:.4f}")
    print(f"   both template halves order at "
          f"{ledger['A2_template_halves']['layers_where_both_halves_order']}; "
          f"templates 1-8 alone at "
          f"{ledger['A2_template_halves']['layers_carried_by_templates_1_8_alone']}")
    jkb = ledger["A4_jackknife"]
    print(f"   leave-one-out: {jkb['n_bh_surviving_suffix_cells_at_8_of_8']}/"
          f"{jkb['n_bh_surviving_suffix_cells']} BH-surviving suffix cells are 8/8; "
          f"exceptions "
          + "; ".join(f"L{c['layer']} {c['n_still_ordered_of_8']}/8 "
                      f"({','.join(c['breaking_drops'])})"
                      for c in jkb["bh_surviving_suffix_cells_below_8_of_8"]))
    print(f"   capitalisation: "
          f"{ledger['A5_capitalisation']['n_ordered_cells_where_capitalised_also_orders']}"
          f"/{ledger['A5_capitalisation']['n_ordered_cells']} ordered cells; "
          f"widened-plane stress closes "
          f"{ledger['A6_widened_plane_stress']['n_cells_closing_stress']}/"
          f"{ledger['A6_widened_plane_stress']['n_cells']}, strict subset of the primary "
          f"closers = "
          f"{ledger['A6_widened_plane_stress']['strict_subset_of_the_primary_closers']}")
    for name, s in ledger["A3_composition_controls"][
            "partial_rho_ranges_by_slice"].items():
        print(f"   partial rho {name:28s} n={s['n']:2d} "
              f"{s['partial_rho'][0]:.4f}..{s['partial_rho'][1]:.4f}  "
              f"p {s['p_partial'][0]:.3e}..{s['p_partial'][1]:.3e}")
    print(f"\nC3 prereg deviation: background draws registered "
          f"{PREREG_BACKGROUND_DRAWS}, used {RUN_BACKGROUND_DRAWS} (conservative, "
          f"run-wide, now annotated)")
    if dual:
        print(f"\nC4 alarm: strict prereg-3.5 reading would trigger on "
              f"{dual['strict_prereg_3_5_reading']['eager_n_raw_closing']} raw eager "
              f"cells / {dual['strict_prereg_3_5_reading']['sdpa_n_raw_closing']} raw "
              f"sdpa cells; BH-graded reading (as run) triggers on "
              f"{dual['bh_graded_reading_AS_RUN']['eager_n_surviving_bh']} / "
              f"{dual['bh_graded_reading_AS_RUN']['sdpa_n_surviving_bh']}. "
              f"VERDICT UNCHANGED.")
    if kern:
        print(f"\nC5 kernel robustness: {kern['n_cells_compared']} cells, gap_ratio "
              f"relative delta mean {kern['gap_ratio_relative_delta']['mean']*100:.1f}%, "
              f"max {kern['gap_ratio_relative_delta']['max']*100:.1f}% at "
              f"{kern['gap_ratio_relative_delta']['max_cell']['readout']} L"
              f"{kern['gap_ratio_relative_delta']['max_cell']['layer']}")
        print(f"   raw closing cells: eager {kern['n_raw_closing_eager']}, "
              f"sdpa {kern['n_raw_closing_sdpa']}; "
              f"{kern['n_closure_decision_flips']} decision flips:")
        for f in kern["closure_decision_flips"]:
            print(f"     {f['readout']:>8} L{f['layer']:<3} {f['direction']:<28} "
                  f"gapR {f['gap_ratio_eager']:.4f}->{f['gap_ratio_sdpa']:.4f}  "
                  f"pUnif {f['p_uniform_eager']:.5f}->{f['p_uniform_sdpa']:.5f}  "
                  f"pNoun {f['p_noun_bg_eager']:.5f}->{f['p_noun_bg_sdpa']:.5f}")
        pr = kern["primary_suffix_readout"]
        print(f"   PRIMARY suffix readout closes raw: eager {pr['n_closing_eager']}/26, "
              f"sdpa {pr['n_closing_sdpa']}/26 "
              f"(same-kernel backgrounds on both sides)")
        for m in pr["monotone_arc_check_on_the_sdpa_suffix_closers"]:
            print(f"     sdpa suffix L{m['layer']}: |rho_ordinal| "
                  f"{m['rho_ordinal_sdpa']:.4f}, above the 0.7619 perfect-8-cycle "
                  f"ceiling = {m['exceeds_perfect_8_cycle_ceiling']} -> monotone arc")
        print(f"   BH-surviving closure: eager "
              f"{kern['n_bh_surviving_closure_eager']}, sdpa "
              f"{kern['n_bh_surviving_closure_sdpa']}  -> verdict unchanged")
        bhs = kern["bh_surviving_ordering_set"]
        print(f"   ordering decision flips: {kern['n_ordering_decision_flips']} "
              f"(raw); BH-surviving ordering set identical cell-for-cell across "
              f"kernels = {bhs['identical_cell_for_cell']} "
              f"({bhs['n_eager']} vs {bhs['n_sdpa']} cells, "
              f"{bhs['n_raw_ordering_flips_that_are_bh_cells']} of the raw flips are BH "
              f"cells)")
    print(f"\nTRACK VERDICT (unchanged): supervised "
          f"{out['verdict']['arm_supervised']}, blind {out['verdict']['arm_blind']}, "
          f"alarm {out['alarm']['triggered']}")
    print(f"\nruntime {time.time() - t0:.1f}s "
          f"(eager sweep {t_eager:.1f}s, sdpa sweep {t_sdpa:.1f}s) -> wrote {a.out}")


if __name__ == "__main__":
    main()
