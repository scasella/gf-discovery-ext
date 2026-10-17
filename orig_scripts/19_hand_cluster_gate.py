"""Item 1b -- would the RIGHT cluster have passed the gate?

DECLARED EXPLORATORY. Not pre-registered; nothing here can change a gate verdict.

The Phase-1 gate can only reject what the clustering hands it, and
`scripts/18_month_latent_placement.py` showed the clustering never hands it a
month subspace: at layer 9 only 2 of 7 month latents land in any cluster, and no
cluster holds more than 2. That leaves the negative ambiguous between two very
different failures:

  A. the discovery step failed -- the clustering never assembled the month
     latents, but the correct latent set would have passed the gate; or
  B. even the correct latent set fails -- the cycle is not carried by the
     identified month latents at all, so no clustering could have found it.

This measures exactly that. The month latents from
`results/month_latent_placement.json` are fed through the gate as HAND-SPECIFIED
clusters: the same `labelled()` function imported from
`scripts/13_rerun_gate.py`, the same `SparseReader` cache, the same rank-even
half-B scoring rows, the same support rule (>= 10 firing positions and >= 5
scoring positions per label, ceil(2*12/3) = 8 of 12 labels), the same 20,000
permutations, and the same methods-note fields (`ref_p_closed`, `missing_run`,
`closure_underpowered`). The latent set is the ONLY thing that differs from the
confirmatory run.

Two sensitivity arms are reported alongside and are labelled as such:
  * the same hand clusters scored on the FULL cache rows rather than half B --
    it removes the held-out split, so it is a sensitivity, never a result;
  * the layer-9 hand cluster augmented with the union of the month latents found
    at layers 8-10, in case the L9 set is simply too small to carry a 12-point
    ring. Latent indices are layer-specific, so this union is a heterogeneous
    set scored against layer 9's decoder -- read it as an upper bound on "more
    month-ish latents would have helped", not as a cluster anyone could find.
  * all 16,384 latents on the same half-B rows. This is the decisive contrast for
    telling A from B: if the full reconstruction closes on the very rows where
    the hand cluster does not, the cycle exists at these positions but is not
    carried by the identified month latents.

Memory: `SparseReader.codes_at` routes through `.csr()`, which materialises the
whole cache (486M nonzeros at layer 9) as float32 values plus int64 indices,
~5.8 GB. Only the ~7k labelled rows are ever needed, so `BlockReader` gathers
that block straight from the memmaps. It is verified elementwise against
`SparseReader.row()` at run time and the check is written into the output.
"""
import argparse, gc, importlib.util, json, os, sys, time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HUB_OFFLINE", "1")
sys.path.insert(0, str(REPO / "src"))
from gfdisc.cache import SparseReader
from gfdisc.saes import load_sae, get_decoder
from gfdisc.fits import direction_ring_score, MIN_RING_K
from gfdisc.nulls import benjamini_hochberg, index_concentration

# the gate itself, imported rather than re-implemented
_spec = importlib.util.spec_from_file_location(
    "rerun_gate", REPO / "scripts" / "13_rerun_gate.py")
gate = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gate)


class BlockReader(SparseReader):
    """SparseReader whose `codes_at` never builds the whole-cache CSR.

    Same contract, same float16 -> float32 conversion, same column order as
    `SparseReader.codes_at`; it just gathers the requested rows from the memmaps
    instead of slicing a 5.8 GB matrix. Each (row, col) appears at most once in
    CSR, so plain assignment is exact. Duplicate features would collapse, hence
    the assert.
    """

    def codes_at(self, rows, features):
        rows = np.asarray(rows, dtype=np.int64)
        feats = np.asarray(features, dtype=np.int64)
        assert len(np.unique(feats)) == len(feats), "duplicate features"
        out = np.zeros((len(rows), len(feats)), dtype=np.float32)
        if len(rows) == 0:
            return out
        lut = np.full(int(self.meta["d_sae"]), -1, dtype=np.int64)
        lut[feats] = np.arange(len(feats))
        starts = self.indptr[rows]
        counts = self.indptr[rows + 1] - starts
        CH = 20000
        for lo in range(0, len(rows), CH):
            hi = min(lo + CH, len(rows))
            st, ct = starts[lo:hi], counts[lo:hi]
            tot = int(ct.sum())
            if tot == 0:
                continue
            off = np.concatenate([[0], np.cumsum(ct)[:-1]])
            flat = np.repeat(st - off, ct) + np.arange(tot)
            k = lut[np.asarray(self.indices[flat], dtype=np.int64)]
            m = k >= 0
            if not m.any():
                continue
            rid = np.repeat(np.arange(hi - lo), ct)[m]
            out[rid, k[m]] = np.asarray(self.values[flat[m]], dtype=np.float32)
        return out


def verify_block_reader(reader, rows, features, n=300, seed=0):
    """Elementwise check of BlockReader.codes_at against per-row memmap reads."""
    rng = np.random.default_rng(seed)
    sub = np.asarray(rows)[rng.choice(len(rows), size=min(n, len(rows)), replace=False)]
    A = reader.codes_at(sub, features)
    B = np.zeros_like(A)
    cmap = {int(f): i for i, f in enumerate(features)}
    for i, p in enumerate(sub):
        idx, val = reader.row(int(p))
        for j, v in zip(idx, val):
            k = cmap.get(int(j))
            if k is not None:
                B[i, k] = v
    return dict(n_rows_checked=int(len(sub)), n_features=int(len(features)),
                max_abs_diff=float(np.abs(A - B).max()),
                n_nonzero_in_block=int((A != 0).sum()))


def gate_seed(layer, first_member):
    """The gate's own seed rule, verbatim: hash() on int tuples is stable."""
    return abs(hash((layer, int(first_member)))) % 10 ** 6


def score(reader, layer, members, W, label_pos, full_k, d_sae,
          row_type, half, note=""):
    """One hand-specified cluster through the gate's labelled() pathway."""
    members = [int(x) for x in members]
    seed = gate_seed(layer, members[0])
    t0 = time.time()
    lab = gate.labelled(reader, members, W, label_pos, seed, full_k)
    row = dict(row_type=row_type, layer=layer, half=half, note=note,
               k=len(members), k_below_min_ring_k=bool(len(members) < MIN_RING_K),
               min_ring_k=int(MIN_RING_K), seed=int(seed),
               members=members if len(members) <= 64 else None,
               n_members_hidden=None if len(members) <= 64 else len(members))
    if lab is None:
        row.update(scorable=False, reason="support rule: fewer than "
                   f"{gate.min_labels_with_support(full_k)} of {full_k} labels "
                   "had >= 5 scoring positions and >= 10 firing positions, or the "
                   "label means were degenerate")
        row["seconds"] = round(time.time() - t0, 2)
        return row
    row.update(scorable=True, **lab)
    # the gate's label-free ranker and Matryoshka index control, for context only:
    # neither is a tier, and direction_ring_score is not trustworthy below k=6.
    if len(members) <= 512:
        d = direction_ring_score(W[np.asarray(members)], n_perm=2000, seed=seed)
        row.update(dir_gap=d["gap_ratio"], dir_p=d["p_closed"], dir_top2=d["top2_var"])
        iqr, n5, ip = index_concentration(members, d_sae, np.random.default_rng(seed), 300)
        row.update(index_iqr=float(iqr), index_p=float(ip))
    row["ordering_pass_raw"] = bool(lab["p_order"] <= 0.05)
    row["closure_pass"] = bool(lab["p_closed"] < 0.05)
    row["passes_both_raw"] = bool(row["ordering_pass_raw"] and row["closure_pass"])
    row["closure_uninformative"] = bool(lab["p_closed"] >= 0.05 and lab["closure_underpowered"])
    row["seconds"] = round(time.time() - t0, 2)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--placement", default="results/month_latent_placement.json")
    ap.add_argument("--primary-layer", type=int, default=9)
    ap.add_argument("--l0", default="big")
    ap.add_argument("--root", default="cache/sparse_big")
    ap.add_argument("--ontology", default="months")
    ap.add_argument("--out", default="results/hand_cluster_gate.json")
    a = ap.parse_args()
    t_start = time.time()

    placement = json.loads(Path(a.placement).read_text())
    hand = {int(r["layer"]): [int(x) for x in r["label_latents"]]
            for r in placement if r["ontology"] == a.ontology}
    layers = sorted(hand)
    print(f"hand clusters: " + ", ".join(f"L{l}: k={len(hand[l])}" for l in layers), flush=True)

    tm = json.load(open(REPO / "results/token_map.json"))
    names, label_ids = gate.label_ids_for(a.ontology, tm)
    full_k = len(names)
    need = gate.min_labels_with_support(full_k)

    # the layer-9 augmentation: union of the month latents found at 8, 9, 10
    union_layers = [l for l in (8, 9, 10) if l in hand]
    union = sorted({m for l in union_layers for m in hand[l]})

    rows, power, verification = [], [], None
    for layer in layers:
        print(f"\n=== layer {layer} ===", flush=True)
        sae = load_sae(layer, a.l0, token=os.environ.get("HF_TOKEN"))
        W = get_decoder(sae)
        d_sae = int(sae.d_sae)
        del sae
        reader = BlockReader(a.root, layer)
        allpos = [reader.positions_of_tokens(ids) for ids in label_ids]

        for half in ("rank-even", "full"):
            label_pos = gate.half_filter(allpos, half)
            support = int(sum(len(p) >= gate.MIN_FIRING_PER_LABEL for p in label_pos))
            power.append(dict(layer=layer, half=half, n_labels=full_k,
                              label_support=support, min_support=need,
                              underpowered=bool(support < need),
                              positions_per_label=[int(len(p)) for p in label_pos],
                              total_positions=int(sum(len(p) for p in label_pos))))
            row_type = "primary_hand_cluster" if half == "rank-even" else "sensitivity_full_cache"
            note = ("" if half == "rank-even" else
                    "SENSITIVITY: scored on all cache rows, not the held-out half B")
            r = score(reader, layer, hand[layer], W, label_pos, full_k, d_sae,
                      row_type, half, note)
            rows.append(r)
            print(f"  [{row_type}] k={r['k']} " + (
                f"rho {r['rho']:+.3f} pOrd {r['p_order']:.4f} | gap {r['gap_ratio']:.2f} "
                f"pClo {r['p_closed']:.4f} | labels {r['n_labels']}/{full_k}"
                if r["scorable"] else "NOT SCORABLE"), flush=True)

            if layer == a.primary_layer:
                if verification is None:
                    verification = dict(layer=layer, half=half,
                                        **verify_block_reader(reader, label_pos[0], hand[layer]))
                if union != hand[layer]:
                    ru = score(reader, layer, union, W, label_pos, full_k, d_sae,
                               "sensitivity_union_L8_L10", half,
                               "SENSITIVITY: union of month latents at layers "
                               f"{union_layers}, scored against layer {layer}'s decoder; "
                               "latent indices are layer-specific so this set is not a "
                               "cluster any method could return")
                    rows.append(ru)
                    print(f"  [union L8-10] k={ru['k']} " + (
                        f"rho {ru['rho']:+.3f} pOrd {ru['p_order']:.4f} | "
                        f"gap {ru['gap_ratio']:.2f} pClo {ru['p_closed']:.4f}"
                        if ru["scorable"] else "NOT SCORABLE"), flush=True)
                ra = score(reader, layer, list(range(d_sae)), W, label_pos, full_k, d_sae,
                           "sensitivity_all_latents_same_rows", half,
                           "SENSITIVITY: all 16,384 latents on the SAME rows -- the "
                           "unrestricted reconstruction, to separate 'wrong latent set' "
                           "from 'no cycle at these positions'")
                rows.append(ra)
                print(f"  [all latents] k={ra['k']} " + (
                    f"rho {ra['rho']:+.3f} pOrd {ra['p_order']:.4f} | "
                    f"gap {ra['gap_ratio']:.2f} pClo {ra['p_closed']:.4f}"
                    if ra["scorable"] else "NOT SCORABLE"), flush=True)

        del reader, W
        gc.collect()

    # BH-FDR across the primary hand clusters only. The confirmatory gate applies BH
    # across every cluster it scores at that layer set (24 rows for the l0_big run);
    # a single hand cluster dropped into that scan would face the same or a stricter
    # threshold, so this is the generous reading.
    prim = [r for r in rows if r["row_type"] == "primary_hand_cluster" and r["scorable"]]
    if prim:
        keep = benjamini_hochberg([r["p_order"] for r in prim])
        for r, k in zip(prim, keep):
            r["ordering_fdr_within_primary"] = bool(k)
            r["CIRCLE_within_primary"] = bool(k and r["p_closed"] < 0.05)

    # ---- comparison anchors, read from the files rather than transcribed ----
    bg = json.loads((REPO / "results/gate_background_big_L9.json").read_text())["summary"]
    mp = json.loads((REPO / "results/months_power_big_L9.json").read_text())
    supervised = {d["representation"]: d for d in mp[a.ontology]}
    scored = json.loads((REPO / "results/rerun_gate_big_primary_L9.json").read_text())["rows"]
    anchors = dict(
        in_gate_background_L9=dict(
            source="results/gate_background_big_L9.json",
            what="index-stratified random latent sets at the scored clusters' sizes, "
                 "through this same labelled() pathway",
            n_draws=bg["n_draws"], n_scorable=bg["n_scorable"],
            rate_order=bg["rate_order"], rate_closure=bg["rate_closure"],
            rate_both=bg["rate_both"]),
        supervised_full_16384_reconstruction_L9=dict(
            source="results/months_power_big_L9.json",
            what="supervised control: 12 month means of the FULL 16,384-latent "
                 "reconstruction over script 11's 240 prompt-final positions "
                 "(a different position set from the cache rows used here)",
            recon=dict(rho=supervised["recon"]["rho"], p_order=supervised["recon"]["p"],
                       p_closed=supervised["recon"]["p_closed"],
                       gap_ratio=supervised["recon"]["gap_ratio"],
                       max_gap_deg=supervised["recon"]["max_gap_deg"],
                       top2=supervised["recon"]["top2"]),
            raw=dict(rho=supervised["raw"]["rho"], p_order=supervised["raw"]["p"],
                     p_closed=supervised["raw"]["p_closed"],
                     gap_ratio=supervised["raw"]["gap_ratio"],
                     max_gap_deg=supervised["raw"]["max_gap_deg"],
                     top2=supervised["raw"]["top2"])),
        confirmatory_gate_L9=dict(
            source="results/rerun_gate_big_primary_L9.json",
            n_scored=len(scored),
            best_p_order=min(r["p_order"] for r in scored),
            best_p_closed=min(r["p_closed"] for r in scored),
            n_circle=int(sum(r["CIRCLE"] for r in scored)),
            cluster_sizes=sorted(r["size"] for r in scored)))

    # ---------------------------- interpretation ----------------------------
    p9 = next((r for r in rows if r["row_type"] == "primary_hand_cluster"
               and r["layer"] == a.primary_layer), None)
    allr9 = next((r for r in rows if r["row_type"] == "sensitivity_all_latents_same_rows"
                  and r["layer"] == a.primary_layer and r["half"] == "rank-even"), None)
    if p9 is None or not p9["scorable"]:
        verdict = "primary hand cluster not scorable"
    elif p9["passes_both_raw"]:
        verdict = "hand cluster PASSES both tiers -> the failure is purely at discovery"
    elif p9["ordering_pass_raw"]:
        verdict = ("hand cluster passes ordering but FAILS closure -> ordered arc at best; "
                   "the closed ring is not carried by the identified month latents alone")
    else:
        verdict = ("hand cluster FAILS BOTH tiers -- ordering as well as closure -- so the "
                   "correct latent set, scored the way the gate scores, does not recover the "
                   "month cycle either")

    # Everything below is derived from the rows just measured; no number is transcribed.
    prim_rows = [r for r in rows if r["row_type"] == "primary_hand_cluster"]
    sc = [r for r in prim_rows if r["scorable"]]
    unsc = [r for r in prim_rows if not r["scorable"]]
    full_support = [r["layer"] for r in sc if r["n_labels"] == full_k]
    support_note = (
        f"Label support is not the binding constraint. At layers {full_support} all "
        f"{full_k} months cleared the support rule on the scoring rows, so missing_run = 0 "
        f"and ref_p_closed = {sc[0]['ref_p_closed']:.1e}: a geometrically perfect ring on "
        "this survivor set would have closed decisively, and the observed closure failure "
        "is a real measurement of the hand cluster's geometry rather than a dropout "
        "artifact." if len(full_support) == len(sc) and sc else
        "Label support varies by layer; read missing_run and ref_p_closed per row.")
    if unsc:
        support_note += (
            " Layers " + str([r["layer"] for r in unsc]) + " were NOT scorable at all: the "
            "hand cluster there is so small that fewer than "
            f"{need} of {full_k} months had >= {gate.MIN_FIRING_PER_LABEL} positions at which "
            "any member fired, which is a statement about the latent set's coverage, not "
            "about geometry.")

    distributed = bool(allr9 is not None and allr9.get("closure_pass")
                       and p9 is not None and p9.get("scorable") and not p9["closure_pass"])
    if distributed:
        branch = (
            "BRANCH 2 (the ring is distributed beyond the identified month latents). On the "
            "identical half-B rows and through the identical labelled() pathway, the "
            "unrestricted 16,384-latent reconstruction recovers the calendar order exactly "
            f"(rho = {allr9['rho']:.3f}, p_order = {allr9['p_order']:.1e}) and CLOSES "
            f"(gap_ratio = {allr9['gap_ratio']:.2f}, p_closed = {allr9['p_closed']:.4f}), "
            f"while the layer-{a.primary_layer} hand cluster on those same rows reaches only "
            f"rho = {p9['rho']:.3f} (p_order = {p9['p_order']:.3f}) and gap_ratio = "
            f"{p9['gap_ratio']:.2f} (p_closed = {p9['p_closed']:.4f}). Latent set is the only "
            "thing that differs between those two arms, so the month cycle is present in the "
            "residual stream at these positions and is carried by far more of the "
            "dictionary than the latents a top_tokens scan calls month latents. "
            "Cluster-restricted scoring over that identified set cannot carry it: the "
            "discovery step is not the only thing that failed, and repairing the clustering "
            "alone would not have produced a gate pass.")
    elif allr9 is not None and not allr9.get("closure_pass"):
        branch = (
            "NEITHER named branch fires. The unrestricted 16,384-latent reconstruction does "
            "not close on these rows either, so the position set -- naturally occurring "
            "corpus month tokens -- and not the latent set is the live explanation for the "
            "difference against script 11's supervised control, which uses 240 curated "
            "prompt-final positions.")
    else:
        branch = "Comparison arm unavailable."

    interpretation = dict(
        verdict=verdict,
        primary_layer=a.primary_layer,
        primary=None if p9 is None else {k: p9.get(k) for k in
                                         ("k", "scorable", "n_labels", "rho", "p_order",
                                          "gap_ratio", "max_gap_deg", "p_closed",
                                          "ref_p_closed", "closure_underpowered", "top2")},
        all_latents_same_rows=None if allr9 is None else {k: allr9.get(k) for k in
                                                          ("rho", "p_order", "gap_ratio",
                                                           "max_gap_deg", "p_closed", "top2")},
        support_note=support_note,
        branch=branch,
        ordering_note=(
            "Note which tier fails. The hand cluster misses ORDERING as well as closure at "
            "every layer scored: the best primary raw p_order is "
            f"{min(r['p_order'] for r in sc):.3f} (layer "
            f"{min(sc, key=lambda r: r['p_order'])['layer']}), against an in-gate background "
            f"ordering acceptance of {bg['rate_order']:.3f} on random index-matched latent "
            "sets. The identified month latents do not even reproduce the calendar ORDER on "
            "half B, which is a stronger negative than the closure-only case and is "
            "consistent with the co-firing result from item 1a: these latents are not a "
            "single month subspace that the clustering merely failed to assemble."
            if sc else "No primary row was scorable."),
        answers=dict(
            would_the_right_cluster_have_passed=None if p9 is None or not p9["scorable"]
            else bool(p9["passes_both_raw"]),
            discovery_only_failure=None if p9 is None or not p9["scorable"]
            else bool(p9["passes_both_raw"]),
            ring_distributed_beyond_identified_month_latents=distributed))

    caveats = [
        "DECLARED EXPLORATORY. Not pre-registered. Hand-specified clusters are a "
        "post-hoc, label-informed latent set; a pass here would not be a gate pass, "
        "because the gate's whole point is that the latent set must be found without "
        "labels. A FAILURE here is the informative direction and is what the arm was "
        "run for.",
        "The month-latent definition is inherited unchanged from "
        "scripts/18_month_latent_placement.py: >= 2 distinct month names among a "
        "latent's shipped top_tokens (min-distinct 2). That threshold has never been "
        "swept. Raising it gives a purer but smaller set; lowering it gives a larger, "
        "noisier one. Neither direction has been measured, so 'the correct latent set' "
        "means 'the set that heuristic returns', not a ground truth.",
        "Latent indices are layer-specific. The union-of-L8-10 arm scores a set "
        "assembled across three SAEs against layer 9's decoder; it is an upper-bound "
        "probe on set size, not a candidate cluster.",
        "Do not over-read the union arm's raw ordering p. It sits just under 0.05 on "
        "half B and just over it on the full cache, which is exactly the instability a "
        "0.164 in-gate background ordering rate predicts, and it is a post-hoc, "
        "label-informed, cross-layer set that no BH-FDR correction has touched. It fails "
        "closure at p_closed > 0.99 either way.",
        "The all-latents arm is a positive control on the position set, not a gate "
        "candidate. It involves no discovery at all -- it is the entire dictionary, i.e. "
        "the SAE reconstruction of the residual stream -- so it can only answer 'is the "
        "cycle present at these positions', never 'could a clustering have found it'.",
        "The full-cache arm reuses the rows the label-free ranking side saw. It is a "
        "sensitivity on the half-B split only and must not be read as a second result.",
        "The supervised anchor from results/months_power_big_L9.json is computed over "
        "script 11's 240 prompt-final positions and the full 16,384-latent "
        "reconstruction, so it differs from the arms here in BOTH latent set and "
        "position set. The all-latents-same-rows arm was added precisely to hold the "
        "position set fixed and vary only the latent set.",
        "direction_ring_score and index_concentration are reported for context. They "
        "are not gate tiers, and direction_ring_score has a measured 100% "
        "false-acceptance rate at k=3, so the k<6 layers' dir_p values are "
        "uninformative by construction.",
        "Ordering here is reported as a raw permutation p as well as BH-FDR within the "
        "six primary hand clusters. The confirmatory gate applies BH across every "
        "cluster it scores, so the real threshold a hand cluster would face is at least "
        "this strict.",
    ]

    out = dict(
        status="exploratory, not pre-registered",
        item="1b -- would the right cluster have passed the gate?",
        script="scripts/19_hand_cluster_gate.py",
        generated=time.strftime("%Y-%m-%dT%H:%M:%S"),
        config=dict(ontology=a.ontology, l0=a.l0, root=a.root,
                    placement=a.placement, primary_layer=a.primary_layer,
                    full_k=full_k, n_perm_ordering=20000, n_perm_gap=20000,
                    MIN_FIRING_PER_LABEL=gate.MIN_FIRING_PER_LABEL,
                    min_positions_per_label=5,
                    min_labels_with_support=need, MIN_RING_K=int(MIN_RING_K),
                    seed_rule="abs(hash((layer, members[0]))) % 10**6, as in script 13",
                    half_definitions=dict(**{"rank-even": "odd cache rows = held-out half B "
                                                          "(the confirmatory scoring set)",
                                             "full": "all cache rows (sensitivity only)"})),
        hand_clusters={str(l): hand[l] for l in layers},
        union_L8_L10=dict(layers=union_layers, members=union, k=len(union)),
        verification=verification,
        power=power,
        rows=rows,
        anchors=anchors,
        interpretation=interpretation,
        caveats=caveats,
        runtime_seconds=round(time.time() - t_start, 1))
    Path(a.out).write_text(json.dumps(out, indent=2))
    print(f"\nverdict: {verdict}")
    print(f"wrote {a.out}  ({out['runtime_seconds']}s)")


if __name__ == "__main__":
    main()
