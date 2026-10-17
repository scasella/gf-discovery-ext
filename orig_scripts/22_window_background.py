"""Step 22 (DECLARED EXPLORATORY, not pre-registered) -- the in-gate background
for the window-level discovery run of scripts/20_window_cluster.py +
scripts/21_window_gate.py.

This is `scripts/16_gate_background.py`'s design, transplanted onto the window
cells. Nothing about the measurement changes: templates are the members of
exactly the clusters the gate SCORED in a cell (read from the gate's own output,
scorable and unscorable alike); each draw is an index-band-stratified random
latent set at full template size via `gfdisc.nulls.stratified_resample`, with no
activity mask; and every draw is scored by `labelled()` imported from
`scripts/13_rerun_gate.py` itself -- the same function, the same 20,000-permutation
ordering null, the same ceil(2k/3) support rule -- on the SAME ROWS the cell's arm
was scored on. Background and observation differ only in how the latent set was
chosen.

Three forced substitutions, each recorded in the output:

  * `BlockReader` (scripts/19_hand_cluster_gate.py) replaces `SparseReader`.
    16's `SparseReader.codes_at` routes through `.csr()`, which materialises the
    whole layer as float32 values plus int64 indices (~5.8 GB at layer 9); this
    machine is 24 GB shared. `BlockReader.codes_at` gathers the same block from
    the memmaps and is bit-verified elementwise against `SparseReader.row()` once
    per layer, with the check written into the output.
  * scoring rows come from the ARM, not from `--half`: arm `even_build` is scored
    on ALL rows of ODD-parity sequences ((row // 255) % 2 == 1, via script 21's
    `odd_sequence_filter`), arm `full` on the pre-registered half B (odd cache
    rows). Script 21's own functions are imported, not re-implemented.
  * `os.environ["HF_TOKEN"]` becomes `.get` under `HF_HUB_OFFLINE=1`; there is no
    token in this environment and the SAE weights are already local.

Rates replicate 16 exactly, including its `<=` / `<` asymmetry: ordering counts
raw `p_order <= 0.05` (the gate additionally requires BH-FDR, so this errs towards
a background that is too HIGH), closure counts `p_closed < 0.05`. Draws that
`labelled()` rejects for support are unscorable and leave the denominator, exactly
as the gate drops them. Only the standard SVD orientation is scored: the alternate
orientation is a reported instrument caveat of the gate, and script 21 found zero
orientation-sensitive clusters in any cell, so doubling the permutation cost here
would buy nothing.

WHERE the background is measured. The project convention is that backgrounds are
measured where passers appear.

  * PRIMARY. Layer 9, W=255, arm even_build, months -- the a-priori primary cell.
    That cell is `rule_failure` in script 21 (no threshold satisfied the frozen
    label-blind rule; the coarse windows blow the edge limit) and scored NOTHING,
    so per the task's own fallback it falls back to the layer-9 months cell with
    the most scored clusters. That is a TIE at 24 between W32_even_build and
    W32_full, broken towards the declared arm `even_build`. Both facts are written
    into the output.
  * CIRCLE passers. Script 21 produced ZERO clusters that pass both tiers, in any
    cell, at any of the six layers, in either ontology. The literal instruction
    "one 500-draw background per cell that produced a CIRCLE passer" therefore has
    an EMPTY cell set, and it is reported as empty rather than quietly widened.
  * EXTENSION, labelled as such. The passers that do exist are ordering-tier
    (BH-FDR on p_order, closure failed) -- 16 cells' worth. Without a background at
    those cells the requested "percentile of each real passer's gap_ratio and rho
    within its matched background" has no referent at all, since the primary cell
    itself has no FDR passer. So each of those 16 cells gets the same 500-draw
    background. This can only add scrutiny to a passer; it cannot turn a failure
    into a pass.

Seeds. The resample RNG is `default_rng(0)` at the primary cell, exactly as 16;
extension cells use `default_rng(1000 + cell_index)` with `cell_index` the cell's
position in the sorted extension list, recorded in the output. `hash()` on strings
is salted per process and is never used. The permutation seed is the draw index
`i`, as in 16.

Instrument quirk, replicated rather than fixed: `gate.reference_ring` caches on
(keep, full_k) WITHOUT the seed, so within one process the first draw to present a
given survivor set fixes `ref_p_closed` for every later draw with the same set.
`full_k` is 12 for both months and numbers, so that cache is shared across
ontologies here. `ref_*` fields are context only and enter no rate.
"""
import argparse, gc, importlib.util, json, os, sys, time
from pathlib import Path

import numpy as np
from scipy.stats import beta

REPO = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HUB_OFFLINE", "1")
sys.path.insert(0, str(REPO / "src"))
from gfdisc.saes import load_sae, get_decoder
from gfdisc.nulls import stratified_resample


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# script 21 is imported whole so that the row definitions and the DESIGN block are
# the same objects the gate used; its `hcg`/`gate` handles are reused so there is
# exactly ONE gate module instance (hence one `_REF_CACHE`) in this process.
wg = _load("window_gate", REPO / "scripts" / "21_window_gate.py")
gate = wg.gate
hcg = wg.hcg
BlockReader = hcg.BlockReader
verify_block_reader = hcg.verify_block_reader
odd_sequence_filter = wg.odd_sequence_filter

LAYERS = [5, 6, 7, 8, 9, 10]
PRIMARY = dict(layer=9, ontology="months", cell="W255_even_build", W=255, arm="even_build")
ALPHA = 0.05


def clopper_pearson(k, n, alpha=ALPHA):
    if n == 0:
        return [None, None]
    lo = 0.0 if k == 0 else float(beta.ppf(alpha / 2, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(1 - alpha / 2, k + 1, n - k))
    return [lo, hi]


def percentile_high_good(vals, x):
    """rho: higher is better. percentile = fraction of background STRICTLY BELOW."""
    a = np.asarray(vals, dtype=float)
    if len(a) == 0:
        return dict(n=0, percentile=None, tail_frac_ge=None)
    return dict(n=int(len(a)), percentile=float((a < x).mean()),
                tail_frac_ge=float((a >= x).mean()))


def percentile_low_good(vals, x):
    """gap_ratio / p_order: lower is better. percentile = fraction STRICTLY ABOVE."""
    a = np.asarray(vals, dtype=float)
    if len(a) == 0:
        return dict(n=0, percentile=None, tail_frac_le=None)
    return dict(n=int(len(a)), percentile=float((a > x).mean()),
                tail_frac_le=float((a <= x).mean()))


def read_gate_cells():
    """Flat inventory of every (layer, ontology, cell) the gate wrote."""
    out = []
    for L in LAYERS:
        p = REPO / f"results/window_gate_L{L}.json"
        g = json.loads(p.read_text())
        assert int(g["layer"]) == L
        for onto, run in g["runs"].items():
            for ck, c in run["cells"].items():
                rows = c.get("rows") or []
                out.append(dict(
                    layer=L, ontology=onto, cell=ck, W=c["W"], arm=c["arm"],
                    gate_file=str(p.relative_to(REPO)),
                    rule_status=c["rule_status"],
                    selected_threshold=c["selected_threshold"],
                    n_clusters_scored=c["n_clusters_scored"],
                    n_scorable=c["bh"].get("n_scorable", 0),
                    n_ordering_fdr=c["bh"].get("n_ordering_fdr", 0),
                    n_CIRCLE=c["bh"].get("n_CIRCLE", 0),
                    rows=rows))
    return out


def cell_key(c):
    return f"L{c['layer']}_{c['ontology']}_{c['cell']}"


def run_cell(c, reader, W, d_sae, label_pos, full_k, n_draws, rng_seed,
             measure_first=None, budget_seconds=None, log=print):
    """`scripts/16_gate_background.py`'s draw loop, verbatim in substance.

    Returns (draws, decision) where `decision` records the 50-draw wall-clock
    measurement when `measure_first` is set.
    """
    templates = [np.asarray(r["members"]) for r in c["rows"]]
    if not templates:
        raise SystemExit(f"{cell_key(c)} scored nothing: no templates")
    rng = np.random.default_rng(rng_seed)
    draws, decision = [], None
    t0 = time.time()
    i = 0
    n_target = n_draws
    while i < n_target:
        t = templates[i % len(templates)]
        f = stratified_resample(t, d_sae, rng)
        lab = gate.labelled(reader, [int(x) for x in f], W, label_pos,
                            seed=i, full_k=full_k)
        rec = dict(i=i, tpl=i % len(templates), size=int(len(f)),
                   template_size=int(len(t)))
        if lab is None:
            rec["scorable"] = False
        else:
            rec.update(scorable=True, rho=lab["rho"], p_order=lab["p_order"],
                       p_closed=lab["p_closed"], gap_ratio=lab["gap_ratio"],
                       max_gap_deg=lab["max_gap_deg"], n_labels=lab["n_labels"])
        draws.append(rec)
        i += 1
        if measure_first is not None and i == measure_first:
            el = time.time() - t0
            per = el / measure_first
            proj2000 = per * 2000
            take2000 = proj2000 <= budget_seconds
            n_target = 2000 if take2000 else 1000
            decision = dict(
                measured_draws=measure_first, measured_seconds=round(el, 2),
                seconds_per_draw=round(per, 4),
                n_scorable_in_measurement=int(sum(r["scorable"] for r in draws)),
                projected_seconds_for_2000=round(proj2000, 1),
                budget_seconds=budget_seconds,
                chosen_n_draws=int(n_target),
                rule=("2,000 draws if the 50-draw measurement projects a 2,000-draw "
                      f"wall clock under {budget_seconds}s (~30 min), else 1,000"))
            log(f"  [measure] {measure_first} draws in {el:.1f}s "
                f"({per:.3f}s/draw) -> 2000 projects {proj2000:.0f}s; "
                f"taking n={n_target}")
        if i % 200 == 0:
            log(f"  {i}/{n_target} draws, {sum(r['scorable'] for r in draws)} "
                f"scorable, {time.time() - t0:.0f}s")
    return draws, decision, round(time.time() - t0, 1)


def summarise(c, draws, rng_seed, role, extra=None):
    sc = [r for r in draws if r["scorable"]]
    n = len(sc)
    n_ord = sum(r["p_order"] <= 0.05 for r in sc)
    n_clo = sum(r["p_closed"] < 0.05 for r in sc)
    n_both = sum(r["p_order"] <= 0.05 and r["p_closed"] < 0.05 for r in sc)
    templates = [len(r["members"]) for r in c["rows"]]
    s = dict(
        key=cell_key(c), role=role, layer=c["layer"], ontology=c["ontology"],
        cell=c["cell"], W=c["W"], arm=c["arm"], gate_file=c["gate_file"],
        selected_threshold=c["selected_threshold"],
        resample_rng_seed=int(rng_seed),
        n_templates=len(templates),
        template_sizes=sorted(templates),
        template_size_mean=float(np.mean(templates)),
        n_draws=len(draws), n_scorable=n,
        scorable_rate=n / max(len(draws), 1),
        draw_size_mean=float(np.mean([r["size"] for r in draws])),
        n_labels_mean=float(np.mean([r["n_labels"] for r in sc])) if n else None,
        n_sig_order=int(n_ord), rate_order=n_ord / max(n, 1),
        ci_order=clopper_pearson(n_ord, n),
        n_sig_closure=int(n_clo), rate_closure=n_clo / max(n, 1),
        ci_closure=clopper_pearson(n_clo, n),
        n_sig_both=int(n_both), rate_both=n_both / max(n, 1),
        ci_both=clopper_pearson(n_both, n),
        gate_n_clusters_scored=c["n_clusters_scored"],
        gate_n_scorable=c["n_scorable"], gate_n_ordering_fdr=c["n_ordering_fdr"],
        gate_n_CIRCLE=c["n_CIRCLE"])
    if n:
        for f in ("rho", "gap_ratio", "p_order", "p_closed"):
            v = np.asarray([r[f] for r in sc], dtype=float)
            s[f"bg_{f}_quantiles"] = {q: float(np.percentile(v, p)) for q, p in
                                      [("p05", 5), ("p50", 50), ("p95", 95), ("p99", 99)]}
            s[f"bg_{f}_max" if f in ("rho",) else f"bg_{f}_min"] = (
                float(v.max()) if f == "rho" else float(v.min()))
    s.update(extra or {})
    return s


def passer_percentiles(c, draws):
    """Every real passer in this cell, placed inside its own matched background."""
    sc = [r for r in draws if r["scorable"]]
    out = []
    for idx, r in enumerate(c["rows"]):
        if not (r.get("ordering_fdr") or r.get("CIRCLE")):
            continue
        same = [d for d in sc if d["tpl"] == idx]
        rec = dict(
            cluster_id=r["cluster_id"], k=r["size"], template_index=idx,
            tier="CIRCLE" if r.get("CIRCLE") else "arc_only_ordering_fdr",
            rho=r["rho"], p_order=r["p_order"], gap_ratio=r["gap_ratio"],
            p_closed=r["p_closed"], n_labels=r["n_labels"],
            orientation_sensitive=r["orientation_sensitive"],
            pool="all scorable draws in this cell",
            rho_vs_bg=percentile_high_good([d["rho"] for d in sc], r["rho"]),
            gap_ratio_vs_bg=percentile_low_good([d["gap_ratio"] for d in sc], r["gap_ratio"]),
            p_order_vs_bg=percentile_low_good([d["p_order"] for d in sc], r["p_order"]),
            same_template_pool=("draws generated from THIS cluster's own index-band "
                                "template, i.e. matched on size and index composition"),
            rho_vs_bg_same_template=percentile_high_good([d["rho"] for d in same], r["rho"]),
            gap_ratio_vs_bg_same_template=percentile_low_good(
                [d["gap_ratio"] for d in same], r["gap_ratio"]),
            p_order_vs_bg_same_template=percentile_low_good(
                [d["p_order"] for d in same], r["p_order"]))
        out.append(rec)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--l0", default="big")
    ap.add_argument("--root", default="cache/sparse_big")
    ap.add_argument("--primary-draws", type=int, default=1000,
                    help="floor; the in-script 50-draw measurement may raise it to 2000")
    ap.add_argument("--measure-draws", type=int, default=50)
    ap.add_argument("--primary-budget", type=float, default=1800.0,
                    help="seconds; 2000 draws are taken only if projected under this")
    ap.add_argument("--extension-draws", type=int, default=500)
    ap.add_argument("--total-budget", type=float, default=3600.0,
                    help="seconds for the extension cells; the pre-declared trim rule "
                         "drops numbers-ontology extension cells first")
    ap.add_argument("--out", default="results/window_background.json")
    a = ap.parse_args()
    t_start = time.time()

    def log(*x):
        print(time.strftime("[%H:%M:%S]"), *x, flush=True)

    inv = read_gate_cells()
    by_key = {cell_key(c): c for c in inv}

    # ---- primary cell, with the declared fallback -------------------------------
    want = f"L{PRIMARY['layer']}_{PRIMARY['ontology']}_{PRIMARY['cell']}"
    req = by_key[want]
    if req["n_clusters_scored"] > 0:
        primary, fallback = req, None
    else:
        pool = [c for c in inv if c["layer"] == PRIMARY["layer"]
                and c["ontology"] == PRIMARY["ontology"]]
        top = max(c["n_clusters_scored"] for c in pool)
        tied = [c for c in pool if c["n_clusters_scored"] == top]
        # tie-break towards the declared arm, then the wider window
        tied.sort(key=lambda c: (c["arm"] != PRIMARY["arm"], -c["W"]))
        primary = tied[0]
        fallback = dict(
            requested=want, requested_rule_status=req["rule_status"],
            requested_selection_caveat=None, requested_n_clusters_scored=0,
            reason=("the a-priori primary cell (layer 9, W=255, arm even_build) is "
                    "rule_failure in scripts/21_window_gate.py and scored nothing, so "
                    "per the task's declared fallback the background is measured at "
                    "the layer-9 months cell with the most scored clusters"),
            fallback_pool="all layer-9 months cells in results/window_gate_L9.json",
            max_n_clusters_scored=int(top),
            tied_cells=[cell_key(c) for c in tied],
            tie_break=("TIE at %d scored clusters between %s; broken towards the "
                       "DECLARED arm even_build (the clean held-out "
                       "selection/scoring split), then the wider window"
                       % (top, ", ".join(cell_key(c) for c in tied))),
            resolved=cell_key(primary))
        log("FALLBACK:", fallback["tie_break"])

    # ---- where the passers are --------------------------------------------------
    circle_cells = [c for c in inv if c["n_CIRCLE"] > 0]
    arc_cells = [c for c in inv if c["n_ordering_fdr"] > 0 and c["n_CIRCLE"] == 0]
    arc_cells.sort(key=lambda c: (c["layer"], c["ontology"], c["cell"]))
    log(f"CIRCLE-passer cells: {len(circle_cells)}  "
        f"ordering-FDR (arc-only) passer cells: {len(arc_cells)}")

    ext = [c for c in circle_cells + arc_cells if cell_key(c) != cell_key(primary)]
    ext_seeds = {cell_key(c): 1000 + i for i, c in enumerate(ext)}

    # ---- group the work by layer so each SAE / cache is opened once -------------
    order = [primary] + ext
    layers_in_order, seen = [], set()
    for c in [primary] + ext:
        if c["layer"] not in seen:
            seen.add(c["layer"]); layers_in_order.append(c["layer"])

    out_cells, verification, timing_decision = {}, {}, None
    trim = dict(rule=("pre-declared BEFORE running: if the measured per-draw cost "
                      "projects the extension past --total-budget, drop "
                      "numbers-ontology extension cells first and keep every months "
                      "cell"), applied=False, dropped=[])
    tm = json.load(open(REPO / "results/token_map.json"))
    label_cache = {}

    for L in layers_in_order:
        here = [c for c in order if c["layer"] == L]
        log(f"=== layer {L}: {len(here)} cell(s): {[cell_key(c) for c in here]}")
        sae = load_sae(L, a.l0, token=os.environ.get("HF_TOKEN"))
        W = get_decoder(sae)
        d_sae = int(sae.d_sae)
        del sae; gc.collect()
        reader = BlockReader(a.root, L)
        for c in here:
            if trim["applied"] and cell_key(c) in trim["dropped"]:
                continue
            onto = c["ontology"]
            if (L, onto) not in label_cache:
                names, ids = gate.label_ids_for(onto, tm)
                label_cache[(L, onto)] = ([reader.positions_of_tokens(i) for i in ids],
                                          len(names))
            allpos, full_k = label_cache[(L, onto)]
            lp = (odd_sequence_filter(allpos) if c["arm"] == "even_build"
                  else gate.half_filter(allpos, "rank-even"))
            if f"L{L}" not in verification:
                widest = max(lp, key=len)
                verification[f"L{L}"] = dict(
                    cell=cell_key(c), arm=c["arm"],
                    what=("BlockReader.codes_at gathered from the memmaps vs per-row "
                          "SparseReader.row() reads, elementwise"),
                    **verify_block_reader(reader, widest, c["rows"][0]["members"]))
                log(f"  block_reader max_abs_diff="
                    f"{verification[f'L{L}']['max_abs_diff']}")
            is_primary = cell_key(c) == cell_key(primary)
            n_draws = a.primary_draws if is_primary else a.extension_draws
            seed = 0 if is_primary else ext_seeds[cell_key(c)]
            log(f"  -- {cell_key(c)} ({'PRIMARY' if is_primary else 'extension'}): "
                f"{len(c['rows'])} templates, sizes "
                f"{sorted(len(r['members']) for r in c['rows'])[:6]}..., "
                f"{n_draws} draws, rng seed {seed}")
            draws, dec, secs = run_cell(
                c, reader, W, d_sae, lp, full_k, n_draws, seed,
                measure_first=(a.measure_draws if (is_primary and a.measure_draws)
                               else None),
                budget_seconds=a.primary_budget, log=log)
            if is_primary:
                timing_decision = dec
                per = dec["seconds_per_draw"] if dec else secs / max(len(draws), 1)
                proj_ext = per * a.extension_draws * len(ext)
                trim.update(seconds_per_draw_measured=per,
                            projected_extension_seconds=round(proj_ext, 1),
                            total_budget_seconds=a.total_budget)
                if proj_ext > a.total_budget:
                    trim["applied"] = True
                    trim["dropped"] = [cell_key(x) for x in ext
                                       if x["ontology"] == "numbers"]
                    log(f"  TRIM APPLIED: projected extension {proj_ext:.0f}s > "
                        f"{a.total_budget}s; dropping "
                        f"{len(trim['dropped'])} numbers cells")
            role = "primary" if is_primary else (
                "extension_circle_passer_cell" if c["n_CIRCLE"] > 0
                else "extension_arc_fdr_passer_cell")
            s = summarise(c, draws, seed, role,
                          extra=dict(seconds=secs,
                                     passers=passer_percentiles(c, draws)))
            s["draws"] = draws
            out_cells[cell_key(c)] = s
            log(f"     -> scorable {s['n_scorable']}/{s['n_draws']}  "
                f"rate_order {s['rate_order']:.4f} "
                f"[{s['ci_order'][0]:.4f},{s['ci_order'][1]:.4f}]  "
                f"rate_closure {s['rate_closure']:.4f}  "
                f"rate_both {s['rate_both']:.4f}  ({secs}s)")
        del reader, W
        gc.collect()

    doc = dict(
        design=wg.DESIGN,
        status="exploratory, not pre-registered",
        script="scripts/22_window_background.py",
        what=("in-gate background acceptance rates for the window-level discovery "
              "run: index-stratified random latent subsets at the gate's own "
              "template sizes, scored by scripts/13_rerun_gate.py labelled() on the "
              "same rows as the cell's arm, 20,000 permutations per draw"),
        background_design=dict(
            source_design="scripts/16_gate_background.py",
            templates=("the members of exactly the clusters the gate SCORED in the "
                       "cell (scorable and unscorable alike), cycled i % n_templates"),
            resample="gfdisc.nulls.stratified_resample, 8 index bands, no activity mask",
            scoring="scripts/13_rerun_gate.py labelled(), standard SVD orientation only",
            n_perm_per_draw=20000,
            rate_definitions=("rate_order counts RAW p_order <= 0.05 (the gate also "
                              "requires BH-FDR, so this errs HIGH); rate_closure "
                              "counts p_closed < 0.05; the <= / < asymmetry is "
                              "script 16's and is replicated"),
            denominator="draws rejected by the support rule leave the denominator",
            ci="Clopper-Pearson exact 95%",
            seeds=("resample rng default_rng(0) at the primary cell as in script 16; "
                   "extension cells default_rng(1000 + cell_index) over the sorted "
                   "extension list; permutation seed = draw index i. hash() on "
                   "strings is never used."),
            substitutions=[
                "BlockReader replaces SparseReader (memory); bit-verified per layer",
                "scoring rows come from the arm (even_build -> odd-parity SEQUENCES; "
                "full -> pre-registered half B odd ROWS), via script 21's own functions",
                "HF_HUB_OFFLINE=1 and HF_TOKEN via .get; no token in this environment"],
            orientation=("standard SVD convention only. The gate reports both "
                         "orientations and found 0 orientation-sensitive clusters in "
                         "any cell, so the alternate orientation is not re-run here."),
            ref_cache_quirk=("gate.reference_ring caches on (keep, full_k) without the "
                             "seed; full_k is 12 for both ontologies so the cache is "
                             "shared. ref_* fields enter no rate.")),
        primary_cell=dict(requested=PRIMARY, resolved=cell_key(primary),
                          fallback=fallback),
        circle_passer_cells=[cell_key(c) for c in circle_cells],
        circle_passer_note=(
            "EMPTY. scripts/21_window_gate.py produced zero two-tier (CIRCLE) passers "
            "in any of the 72 cells across layers 5-10 in either ontology, so the "
            "instruction 'a 500-draw background at every cell that produced a CIRCLE "
            "passer' has an empty cell set. It is reported empty rather than widened."
            if not circle_cells else "see arc/circle cell lists"),
        arc_fdr_passer_cells=[cell_key(c) for c in arc_cells],
        extension_note=(
            "EXTENSION BEYOND THE LITERAL TASK, labelled as such: with no CIRCLE "
            "passer anywhere, the only passers that exist are ordering-tier (BH-FDR "
            "on p_order, closure failed). Each such cell gets a 500-draw background "
            "so that the requested per-passer gap_ratio/rho percentiles have a "
            "matched referent. A background can only add scrutiny to a passer."),
        timing_measurement=timing_decision,
        trim_rule=trim,
        cells=out_cells,
        verification=dict(block_reader_per_layer=verification),
        runtime_seconds=round(time.time() - t_start, 1),
        generated=time.strftime("%Y-%m-%dT%H:%M:%S"))
    Path(REPO / a.out).write_text(json.dumps(doc, indent=2))
    log(f"wrote {a.out} in {doc['runtime_seconds']}s")


if __name__ == "__main__":
    main()
