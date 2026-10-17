"""Step 21 (DECLARED EXPLORATORY, not pre-registered) -- the two-tier gate on the
window-level clusters found by `scripts/20_window_cluster.py`.

Every cluster of size >= MIN_RING_K (6) in every cell that has a selected
partition is scored. There is NO top-k throttle: `13_rerun_gate.py --top-k 20`
ranked by the label-free decoder-direction score and scored only the top 20,
which was a retrospective finding, not a rule, and it is not reproduced here.

Scoring is `scripts/13_rerun_gate.py`'s `labelled()`. The module is imported, not
re-implemented, and so is `BlockReader` from `scripts/19_hand_cluster_gate.py`
(the memory-frugal `codes_at` that avoids materialising the whole-cache CSR); it
is bit-verified against `SparseReader.row()` at run time and the check is written
into the output.

`labelled_local()` is a line-for-line copy of `gate.labelled()` with exactly two
additions -- a `--perms` parameter threaded into every permutation call, and a
`flip` flag that negates the second principal coordinate. It is proved equal to
the imported `gate.labelled()` at 20,000 permutations on the first scorable
cluster of every cell, and that proof is recorded in the output. All reported
numbers come from the local copy so that a smoke run at reduced `--perms` takes
the identical code path as a full run.

Scoring rows, per arm (the held-out discipline this run exists to restore):

  arm "even_build"  graph was built ONLY from windows of EVEN-parity sequences,
                    so it is scored on ALL cache rows belonging to ODD-parity
                    sequences: (row // 255) % 2 == 1. Selection and scoring share
                    no sequence, hence no token.
  arm "full"        graph was built from every sequence, so the only split left
                    is the pre-registered half B: odd cache ROWS, row % 2 == 1,
                    via `gate.half_filter(..., "rank-even")`.

Instrument caveat, implemented rather than assumed: `cyclic_order_score` is
reported to be sensitive to the SVD sign convention, which fixes the sign of the
second principal axis and therefore the handedness of the recovered angles. Every
cluster is scored under BOTH orientations -- the standard convention (primary,
for comparability with every earlier gate run) and the reflected one -- and the
extremum and an `orientation_sensitive` flag are reported whenever the 0.05
verdict differs between them.
"""
import argparse, gc, importlib.util, json, os, sys, time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HUB_OFFLINE", "1")
sys.path.insert(0, str(REPO / "src"))
from gfdisc.saes import load_sae, get_decoder
from gfdisc.fits import (MIN_RING_K, pca_2d, fit_circle, circularity, angles,
                         cyclic_order_score, degenerate, direction_ring_score)
from gfdisc.nulls import benjamini_hochberg, index_concentration


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# the gate itself and the memory-frugal reader, imported rather than re-implemented
hcg = _load("hand_cluster_gate", REPO / "scripts" / "19_hand_cluster_gate.py")
gate = hcg.gate
BlockReader = hcg.BlockReader
verify_block_reader = hcg.verify_block_reader

SEQ = 255
STANDARD_PERMS = 20000

DESIGN = dict(
    status="exploratory, not pre-registered",
    primary_cell=(
        "Primary cell, declared a priori: layer 9, W=255 (whole-sequence windows), "
        'arm "even_build" (graph built ONLY from even-parity sequences, scored ONLY '
        "on rows of odd-parity sequences -- the clean held-out selection/scoring split "
        "the original prereg promised). All other cells (layers 5-10, W in {32,128,255}, "
        "arms even_build and full) are robustness."),
    label_blindness=(
        "The choice of W=255/L9 as primary was informed by the month_cofire diagnostic "
        "(label-informed DESIGN), but every WITHIN-RUN selection step (threshold choice, "
        "cluster selection) is strictly label-blind; scripts must be structurally "
        "incapable of reading label/token-map/placement data before selection."),
    threshold_sweep=(
        "Threshold sweep T in {0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85} -- extended "
        "above the prereg's per-position sweep A PRIORI because window-level base "
        "co-fire rates are mechanically higher; selection by the prereg's frozen "
        "label-blind rule replicated verbatim (read PREREGISTRATION.md's Amendment-1 "
        "frozen threshold rule); if no threshold satisfies the rule in a cell, report "
        "rule_failure for that cell and do NOT extend post hoc."),
)

# `gate.reference_ring` caches on (keep, full_k) WITHOUT the seed, so within one
# process the first cluster to present a given survivor set fixes ref_p_closed for
# every later cluster with the same set. That quirk is part of the instrument and
# is replicated here rather than silently fixed; only the permutation count is
# added to the key.
_REF_LOCAL = {}


def reference_ring_local(keep, full_k, seed, n_perm):
    if n_perm == STANDARD_PERMS:
        return gate.reference_ring(keep, full_k, seed=seed)      # shared cache, identical
    key = (tuple(int(x) for x in keep), int(full_k), int(n_perm))
    if key not in _REF_LOCAL:
        th = np.sort(2 * np.pi * np.asarray(sorted(key[0]), dtype=float) / full_k)
        gap = float(np.diff(np.concatenate([th, [th[0] + 2 * np.pi]])).max())
        spacing = 2 * np.pi / full_k
        run = int(round(gap / spacing)) - 1
        _REF_LOCAL[key] = dict(
            ref_gap_deg=float(np.degrees(gap)),
            ref_p_closed=gate.gap_pvalue(len(key[0]), gap, n_perm=n_perm, seed=seed),
            missing_run=run, closure_underpowered=bool(run >= 2))
    return _REF_LOCAL[key]


def label_means(reader, cluster, w_dec, label_pos):
    """The first loop of `gate.labelled()`, verbatim, factored out so the two
    orientations share it: it does not depend on the orientation, and it is the
    expensive half (one `codes_at` gather per label)."""
    means, keep = [], []
    for i, pos in enumerate(label_pos):
        if len(pos) < 5:
            continue
        A = reader.codes_at(pos, cluster)
        if int((A > 0).any(1).sum()) < gate.MIN_FIRING_PER_LABEL:
            continue
        means.append((A @ np.asarray(w_dec)[np.asarray(cluster)]).mean(0)); keep.append(i)
    return means, keep


def labelled_local(reader, cluster, w_dec, label_pos, seed, full_k,
                   n_perm=STANDARD_PERMS, flip=False, cache=None):
    """`gate.labelled()` with `--perms` threaded through and an optional PC2 flip."""
    means, keep = label_means(reader, cluster, w_dec, label_pos) if cache is None else cache
    if len(keep) < gate.min_labels_with_support(full_k):
        return None
    M = np.vstack(means)
    if degenerate(M):
        return None
    xy, ratios, _ = pca_2d(M)
    if flip:
        xy = xy * np.array([1.0, -1.0])          # the alternate SVD sign convention
    cx, cy, r, resid = fit_circle(xy)
    c = circularity(xy, cx, cy, r)
    k = len(keep)
    s = cyclic_order_score(angles(xy, cx, cy), np.asarray(keep), k,
                           n_perm=n_perm, seed=seed, full_k=full_k)
    return dict(n_labels=k, labels=[int(x) for x in keep], rho=s["rho"], p_order=s["p"],
                **reference_ring_local(keep, full_k, seed, n_perm),
                gap_ratio=c["gap_ratio"], max_gap_deg=c["max_gap_deg"],
                p_closed=gate.gap_pvalue(k, np.radians(c["max_gap_deg"]),
                                         n_perm=n_perm, seed=seed),
                top2=float(ratios[:2].sum()))


def odd_sequence_filter(positions):
    """Rows of ODD-parity sequences -- the held-out set for the even_build arm.

    Sequence = row // 255 (seq_len 256, BOS never written, so 255 rows per
    sequence and rows are sequence-major; verified in script 20's
    row_layout_check). The even_build graph saw only even-parity sequences, so no
    token scored here contributed a single co-occurrence count to it.
    """
    return [p[(p // SEQ) % 2 == 1] for p in positions]


def score_cluster(reader, layer, members, W, label_pos, full_k, d_sae, n_perm):
    """One cluster through both orientations of the gate."""
    members = [int(x) for x in members]
    seed = hcg.gate_seed(layer, members[0])
    t0 = time.time()
    cache = label_means(reader, members, W, label_pos)
    row = dict(size=len(members), members=members, seed=int(seed))
    std = labelled_local(reader, members, W, label_pos, seed, full_k,
                         n_perm=n_perm, flip=False, cache=cache)
    if std is None:
        row.update(scorable=False, reason=(
            "support rule: fewer than "
            f"{gate.min_labels_with_support(full_k)} of {full_k} labels had >= 5 "
            "scoring positions and >= 10 firing positions, or the label means were "
            "degenerate"), seconds=round(time.time() - t0, 2))
        return row
    alt = labelled_local(reader, members, W, label_pos, seed, full_k,
                         n_perm=n_perm, flip=True, cache=cache)
    row.update(scorable=True, **std)
    row["rho_alt_orientation"] = alt["rho"]
    row["p_order_alt_orientation"] = alt["p_order"]
    ext_alt = (alt["p_order"], -alt["rho"]) < (std["p_order"], -std["rho"])
    row["rho_extremum"] = alt["rho"] if ext_alt else std["rho"]
    row["p_order_extremum"] = alt["p_order"] if ext_alt else std["p_order"]
    row["orientation_extremum"] = "alt" if ext_alt else "standard"
    row["orientation_sensitive"] = bool((std["p_order"] < 0.05)
                                        != (alt["p_order"] < 0.05))
    # closure is invariant to reflection by construction (the angular gap set is
    # unchanged); recorded rather than asserted, so a violation would show up.
    row["closure_alt_identical"] = bool(alt["p_closed"] == std["p_closed"]
                                        and alt["max_gap_deg"] == std["max_gap_deg"])
    d = direction_ring_score(W[np.asarray(members)], n_perm=2000, seed=seed)
    row.update(dir_gap=d["gap_ratio"], dir_p=d["p_closed"], dir_top2=d["top2_var"])
    iqr, n5, ip = index_concentration(members, d_sae, np.random.default_rng(seed), 300)
    row.update(index_iqr=float(iqr), index_p=float(ip))
    row["ordering_pass_raw"] = bool(std["p_order"] <= 0.05)
    row["closure_pass"] = bool(std["p_closed"] < 0.05)
    row["closure_uninformative"] = bool(std["p_closed"] >= 0.05
                                        and std["closure_underpowered"])
    row["seconds"] = round(time.time() - t0, 2)
    return row


def equality_proof(reader, layer, members, W, label_pos, full_k):
    """labelled_local(flip=False, 20k) must equal the imported gate.labelled() exactly."""
    seed = hcg.gate_seed(layer, members[0])
    a = gate.labelled(reader, members, W, label_pos, seed, full_k)
    b = labelled_local(reader, members, W, label_pos, seed, full_k,
                       n_perm=STANDARD_PERMS, flip=False)
    if a is None or b is None:
        return dict(compared=False, gate_none=a is None, local_none=b is None)
    diffs = {k: [a[k], b[k]] for k in a if a[k] != b[k]}
    return dict(compared=True, n_fields=len(a), identical=bool(not diffs),
                differing_fields=diffs, first_member=int(members[0]),
                n_perm=STANDARD_PERMS,
                what=("scripts/13_rerun_gate.py labelled() vs this script's local "
                      "copy at the standard orientation and 20,000 permutations"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--clusters", default=None)
    ap.add_argument("--ontology", default="months", choices=["months", "days", "numbers"])
    ap.add_argument("--l0", default="big")
    ap.add_argument("--root", default="cache/sparse_big")
    ap.add_argument("--perms", type=int, default=STANDARD_PERMS,
                    help="permutations for ordering and closure; 20000 is the "
                         "pre-registered value, anything else tags the output "
                         "smoke_nonstandard")
    ap.add_argument("--min-size", type=int, default=int(MIN_RING_K),
                    help="score every cluster at or above this size; no top-k throttle")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    clusters_path = Path(a.clusters or f"results/window_clusters_L{a.layer}.json")
    out_path = Path(a.out or f"results/window_gate_L{a.layer}.json")
    t_start = time.time()

    src = json.loads(clusters_path.read_text())
    assert int(src["layer"]) == a.layer, (
        f"clusters file is layer {src['layer']}, --layer says {a.layer}")

    tm = json.load(open(REPO / "results/token_map.json"))
    names, label_ids = gate.label_ids_for(a.ontology, tm)
    full_k = len(names)
    need = gate.min_labels_with_support(full_k)

    sae = load_sae(a.layer, a.l0, token=os.environ.get("HF_TOKEN"))
    W = get_decoder(sae)
    d_sae = int(sae.d_sae)
    del sae
    reader = BlockReader(a.root, a.layer)
    allpos = [reader.positions_of_tokens(ids) for ids in label_ids]
    print(f"layer {a.layer} / {a.ontology}: {full_k} labels, "
          f"{sum(len(p) for p in allpos):,} cache rows carry one, need {need} "
          f"labels with support", flush=True)

    row_sets = {
        "even_build": dict(
            positions=odd_sequence_filter(allpos),
            definition=("ALL cache rows of ODD-parity sequences: (row // 255) % 2 == 1. "
                        "The graph was built only from EVEN-parity sequences, so "
                        "selection and scoring share no sequence.")),
        "full": dict(
            positions=gate.half_filter(allpos, "rank-even"),
            definition=("the pre-registered half B: odd cache ROWS, row % 2 == 1, via "
                        "gate.half_filter(..., 'rank-even'). The graph was built from "
                        "every sequence, so this is the only split available.")),
    }
    for arm, rs in row_sets.items():
        p = rs["positions"]
        rs["label_support"] = int(sum(len(x) >= gate.MIN_FIRING_PER_LABEL for x in p))
        rs["positions_per_label"] = [int(len(x)) for x in p]
        rs["total_positions"] = int(sum(len(x) for x in p))
        rs["underpowered"] = bool(rs["label_support"] < need)
        print(f"  arm {arm}: {rs['label_support']}/{full_k} labels with >= "
              f"{gate.MIN_FIRING_PER_LABEL} scoring positions "
              f"({rs['total_positions']:,} rows)"
              f"{'  -- UNDERPOWERED' if rs['underpowered'] else ''}", flush=True)

    verification, cells_out = dict(), {}
    for key, cell in src["cells"].items():
        arm = cell["arm"]
        cl = [c for c in (cell.get("clusters") or []) if c["size"] >= a.min_size]
        rec = dict(W=cell["W"], arm=arm, rule_status=cell["rule_status"],
                   selected_threshold=cell["selected_threshold"],
                   selection_caveat=cell.get("selection_caveat"),
                   n_clusters_selected=cell.get("n_clusters_selected"),
                   n_clusters_scored=len(cl),
                   scoring_rows=row_sets[arm]["definition"],
                   label_support=row_sets[arm]["label_support"],
                   underpowered=row_sets[arm]["underpowered"], rows=[],
                   bh=dict(n_scorable=0))
        if cell["rule_status"] != "selected":
            rec["note"] = "no partition selected in this cell; nothing to score"
            cells_out[key] = rec
            print(f"\n=== {key}: {cell['rule_status']}, skipped ===", flush=True)
            continue
        print(f"\n=== {key}: T={cell['selected_threshold']}, "
              f"{len(cl)} clusters of size >= {a.min_size} ===", flush=True)
        lp = row_sets[arm]["positions"]
        if "block_reader" not in verification and cl:
            # the best-supported label, so the check sees a non-empty row block
            widest = max(lp, key=len)
            verification["block_reader"] = dict(
                cell=key, arm=arm,
                what=("BlockReader.codes_at gathered from the memmaps vs per-row "
                      "SparseReader.row() reads, elementwise"),
                **verify_block_reader(reader, widest, cl[0]["members"]))
        proved = False
        for c in cl:
            r = score_cluster(reader, a.layer, c["members"], W, lp, full_k,
                              d_sae, a.perms)
            r["cluster_id"] = c["cluster_id"]
            rec["rows"].append(r)
            if r["scorable"] and not proved:
                verification.setdefault("labelled_equality", {})[key] = equality_proof(
                    reader, a.layer, c["members"], W, lp, full_k)
                proved = True
            if r["scorable"]:
                print(f"    k={r['size']:3d} rho {r['rho']:+.3f} pOrd {r['p_order']:.4f} "
                      f"(alt {r['p_order_alt_orientation']:.4f}) | gap "
                      f"{r['gap_ratio']:.2f} pClo {r['p_closed']:.4f} | labels "
                      f"{r['n_labels']}/{full_k} [{r['seconds']}s]", flush=True)
            else:
                print(f"    k={r['size']:3d} NOT SCORABLE [{r['seconds']}s]", flush=True)

        sc = [r for r in rec["rows"] if r["scorable"]]
        if sc:
            keep = benjamini_hochberg([r["p_order"] for r in sc])
            for r, k in zip(sc, keep):
                r["ordering_fdr"] = bool(k)
                r["CIRCLE"] = bool(k and r["p_closed"] < 0.05)
                r["arc_only"] = bool(k and r["p_closed"] >= 0.05)
            rec["bh"] = dict(
                scope="Benjamini-Hochberg over p_order (standard orientation) WITHIN "
                      "this cell, alpha 0.05",
                n_scorable=len(sc), n_ordering_fdr=int(sum(keep)),
                n_CIRCLE=int(sum(r["CIRCLE"] for r in sc)),
                n_arc_only=int(sum(r["arc_only"] for r in sc)),
                n_closure_uninformative=int(sum(r["closure_uninformative"] for r in sc)),
                n_orientation_sensitive=int(sum(r["orientation_sensitive"] for r in sc)),
                best_p_order=float(min(r["p_order"] for r in sc)),
                best_p_closed=float(min(r["p_closed"] for r in sc)))
            print(f"  -> {rec['bh']['n_ordering_fdr']}/{len(sc)} pass ordering (BH), "
                  f"{rec['bh']['n_CIRCLE']} CIRCLE, {rec['bh']['n_arc_only']} arc_only",
                  flush=True)
        else:
            rec["bh"] = dict(n_scorable=0)
        cells_out[key] = rec
        gc.collect()

    run = dict(
        ontology=a.ontology, l0=a.l0, root=a.root,
        clusters_file=str(clusters_path),
        clusters_generated=src.get("generated"),
        n_perm=a.perms,
        smoke_nonstandard=bool(a.perms != STANDARD_PERMS),
        full_k=full_k, min_labels_with_support=need,
        MIN_FIRING_PER_LABEL=gate.MIN_FIRING_PER_LABEL,
        min_positions_per_label=5, min_size_scored=a.min_size,
        top_k_throttle=None,
        top_k_note=("no top-k throttle: 13_rerun_gate.py's --top-k 20 was a "
                    "retrospective finding, not a rule, and is not reproduced"),
        seed_rule="abs(hash((layer, members[0]))) % 10**6, as in script 13",
        scoring_rows={k: v["definition"] for k, v in row_sets.items()},
        power={k: {kk: v[kk] for kk in ("label_support", "positions_per_label",
                                        "total_positions", "underpowered")}
               for k, v in row_sets.items()},
        verification=verification,
        cells=cells_out,
        summary=dict(
            n_cells=len(cells_out),
            n_cells_with_partition=int(sum(1 for c in cells_out.values()
                                           if c["rule_status"] == "selected")),
            n_clusters_scored=int(sum(c["n_clusters_scored"] for c in cells_out.values())),
            n_scorable=int(sum(c["bh"].get("n_scorable", 0) for c in cells_out.values())),
            n_CIRCLE=int(sum(c["bh"].get("n_CIRCLE", 0) for c in cells_out.values())),
            n_arc_only=int(sum(c["bh"].get("n_arc_only", 0) for c in cells_out.values())),
            n_orientation_sensitive=int(sum(c["bh"].get("n_orientation_sensitive", 0)
                                            for c in cells_out.values()))),
        caveats=[
            "DECLARED EXPLORATORY. Not pre-registered. Nothing here can change a "
            "pre-registered gate verdict.",
            "BH-FDR is applied WITHIN each cell. Cells are not independent -- the same "
            "cache, the same layer, three nested window sizes and two overlapping arms "
            "-- so a positive in one cell is not replicated by a positive in another, "
            "and no correction across cells has been applied.",
            "Ordering and closure are reported at the standard SVD sign convention. "
            "The reflected orientation is reported alongside; orientation_sensitive "
            "flags any cluster whose 0.05 ordering verdict differs between them.",
            "gate.reference_ring caches ref_p_closed on (survivor set, full_k) WITHOUT "
            "the seed, so within a process the first cluster to present a given "
            "survivor set fixes ref_p_closed for every later cluster with the same "
            "set. That quirk is inherited from script 13 deliberately and is "
            "replicated in the local copy; it affects the methods-note field only, "
            "never a tier.",
            "direction_ring_score and index_concentration are context, not tiers. "
            "direction_ring_score has a measured 100% false-acceptance rate at k=3, "
            "which is why only clusters of size >= 6 are scored at all.",
            "The number-word ontology is the ordinal control: it must never pass "
            "closure. A run in which it does is invalid rather than interesting.",
        ],
        runtime_seconds=round(time.time() - t_start, 1))

    # `numbers` appends: the ordinal control shares the file with its months run.
    doc = dict(design=DESIGN, status="exploratory, not pre-registered",
               script=str(Path(__file__).resolve()),
               layer=int(a.layer), runs={})
    if out_path.exists():
        prev = json.loads(out_path.read_text())
        if int(prev.get("layer", a.layer)) != a.layer:
            raise SystemExit(f"{out_path} holds layer {prev['layer']}, not {a.layer}")
        doc["runs"] = prev.get("runs", {})
    doc["runs"][a.ontology] = run
    doc["generated"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(doc, indent=2))
    s = run["summary"]
    print(f"\n{a.ontology}: {s['n_scorable']} scorable of {s['n_clusters_scored']} "
          f"scored; {s['n_CIRCLE']} CIRCLE, {s['n_arc_only']} arc_only, "
          f"{s['n_orientation_sensitive']} orientation-sensitive")
    print(f"wrote {out_path}  ({run['runtime_seconds']}s)")


if __name__ == "__main__":
    main()
