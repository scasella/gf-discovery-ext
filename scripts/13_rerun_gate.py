"""Phase-1 re-run: the pre-registered two-tier gate.

Ranking is label-free and runs on the cluster's DECODER DIRECTIONS (see
`direction_ring_score` for why the activation cloud was the wrong object). Only
after ranking are labels touched, and a cluster is called a circle only if it
passes BOTH tiers:

  ordering  -- cyclic-order Spearman against the true cycle, permutation-calibrated
               under the same max-over-symmetries selection;
  closure   -- angular gap of the recovered label means, calibrated against uniform.

Ordering alone is reported as an ordered arc. Number words one..twelve run as an
ordinal control that must never pass closure.
"""
import argparse, json, os, sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from gfdisc.cache import SparseReader
from gfdisc.saes import load_sae, get_decoder
from gfdisc.fits import (direction_ring_score, MIN_RING_K, pca_2d, fit_circle,
                         circularity, angles, cyclic_order_score, degenerate)
from gfdisc.nulls import benjamini_hochberg, index_concentration
from gfdisc.tokens import DAYS, MONTHS, resolve

MIN_FIRING_PER_LABEL = 10

# The pre-registered support threshold is 8 of 12 months -- exactly two thirds of the
# cycle. Stated as the fraction rather than the count so it transfers: months keep
# their pre-registered 8, and weekdays get ceil(2*7/3) = 5. The literal 8 was not
# transferable at all, being larger than the 7 weekday labels, so `labelled` returned
# None unconditionally for --ontology days and that arm could not be scored.
MIN_SUPPORT_FRACTION = 2 / 3


def min_labels_with_support(full_k):
    return int(np.ceil(MIN_SUPPORT_FRACTION * full_k))

# The ordinal control, same word list as scripts/12_closure_depth.py. Number words
# carry an order but no wrap-around, so they run through the identical gate path and
# must fail closure; a run in which they pass is invalid rather than interesting.
NUMS = ["one", "two", "three", "four", "five", "six",
        "seven", "eight", "nine", "ten", "eleven", "twelve"]


def gap_pvalue(k, max_gap_rad, n_perm=20000, seed=0):
    rng = np.random.default_rng(seed)
    a = np.sort(rng.uniform(0, 2 * np.pi, size=(n_perm, k)), axis=1)
    g = np.diff(np.concatenate([a, a[:, :1] + 2 * np.pi], axis=1), axis=1).max(1)
    return float((1 + (g <= max_gap_rad).sum()) / (1 + n_perm))


_REF_CACHE = {}


def reference_ring(keep, full_k, seed=0):
    """What closure could have reported for a GEOMETRICALLY PERFECT ring on these labels.

    Ordering is blind to which labels dropped out; closure is not. `gap_pvalue`
    calibrates the largest angular gap against k *uniform* angles, so unsupported
    labels widen the observed gap without widening the null. Two adjacent months
    missing leaves a 90-degree gap where 10 uniform angles expect 36, and a ring
    whose points lie exactly on the circle fails at p_closed = 0.337. Two
    non-adjacent dropouts still pass at 0.009.

    The pre-registered uniform null stays decisive -- it errs towards false
    NEGATIVES, which is the safe direction here -- so this adds context only and
    can never turn a fail into a pass. `ref_p_closed` is the sharpest reading:
    when it is already >= 0.05 the survivor set cannot pass closure at all, whatever
    the geometry, and a failure at that layer says nothing about a ring.
    """
    key = (tuple(int(x) for x in keep), int(full_k))
    if key not in _REF_CACHE:
        th = np.sort(2 * np.pi * np.asarray(sorted(key[0]), dtype=float) / full_k)
        gap = float(np.diff(np.concatenate([th, [th[0] + 2 * np.pi]])).max())
        spacing = 2 * np.pi / full_k
        # a run of m consecutive unsupported labels leaves a gap of (m+1) slots
        run = int(round(gap / spacing)) - 1
        _REF_CACHE[key] = dict(
            ref_gap_deg=float(np.degrees(gap)),
            ref_p_closed=gap_pvalue(len(key[0]), gap, seed=seed),
            missing_run=run,
            # the lead's definition: >= 2 adjacent missing labels. Note this is a
            # 3x-spacing gap, not the 2x quoted alongside it -- 2x is a single
            # dropout, which still passes closure on a perfect ring (0.009).
            closure_underpowered=bool(run >= 2))
    return _REF_CACHE[key]


def label_ids_for(ontology, tm):
    """Token ids per label, in cycle order. Returns (names, ids).

    Months and days come from the audited map written by script 02. Number words
    are the ordinal control and were never written to that map, so they are
    resolved on demand with the same `tokens.resolve` the map itself was built
    with -- the tokenizer only, no model.
    """
    if ontology in ("months", "days"):
        names = MONTHS if ontology == "months" else DAYS
        return names, [tm[ontology][n] for n in names]
    from transformers import AutoTokenizer
    from gfdisc.env import MODEL_NAME
    tok = AutoTokenizer.from_pretrained(MODEL_NAME, token=os.environ.get("HF_TOKEN"))
    mapping, _ = resolve(tok, NUMS)
    return NUMS, [mapping[n] for n in NUMS]


def half_filter(positions, half):
    """Restrict cache rows to one parity class.

    'rank-even' holds out half B (odd rows) for scoring; the label-free ranking
    side sees only even rows. Parity of the cache row index is a valid split
    because rows are written in corpus order and BOS is never written, so
    neighbouring rows are neighbouring tokens rather than repeats.
    """
    if half == "full":
        return positions
    return [p[p % 2 == 1] for p in positions]


def labelled(reader, cluster, w_dec, label_pos, seed, full_k):
    means, keep = [], []
    for i, pos in enumerate(label_pos):
        if len(pos) < 5:
            continue
        A = reader.codes_at(pos, cluster)
        if int((A > 0).any(1).sum()) < MIN_FIRING_PER_LABEL:
            continue
        means.append((A @ np.asarray(w_dec)[np.asarray(cluster)]).mean(0)); keep.append(i)
    if len(keep) < min_labels_with_support(full_k):
        return None
    M = np.vstack(means)
    if degenerate(M):
        return None
    xy, ratios, _ = pca_2d(M)
    cx, cy, r, resid = fit_circle(xy)
    c = circularity(xy, cx, cy, r)
    k = len(keep)
    # `keep` holds the survivors' TRUE positions in the cycle, not 0..k-1: when a
    # month drops out for lack of support the remaining ones keep their calendar
    # spacing, and pretending they are contiguous asserts the wrong reference order.
    s = cyclic_order_score(angles(xy, cx, cy), np.asarray(keep), k,
                           n_perm=20000, seed=seed, full_k=full_k)
    return dict(n_labels=k, labels=[int(x) for x in keep], rho=s["rho"], p_order=s["p"],
                **reference_ring(keep, full_k, seed=seed),
                gap_ratio=c["gap_ratio"], max_gap_deg=c["max_gap_deg"],
                p_closed=gap_pvalue(k, np.radians(c["max_gap_deg"]), seed=seed),
                top2=float(ratios[:2].sum()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", default="18,19,20")
    ap.add_argument("--clusters", nargs="+", default=["results/clusters_L18.json", "results/clusters.json"])
    ap.add_argument("--ontology", default="months", choices=["months", "days", "numbers"])
    ap.add_argument("--l0", default="small")
    ap.add_argument("--root", default="cache/sparse")
    ap.add_argument("--half", default="full", choices=["full", "rank-even"],
                    help="rank-even: score ordering/closure on odd cache rows (half B) only")
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--out", default="results/rerun_gate.json")
    a = ap.parse_args()
    token = os.environ["HF_TOKEN"]
    tm = json.load(open("results/token_map.json"))
    names, label_ids = label_ids_for(a.ontology, tm)
    full_k = len(names)

    clusters = {}
    for f in a.clusters:
        if Path(f).exists():
            clusters.update(json.load(open(f)))

    rows, power = [], []
    for layer in [int(x) for x in a.layers.split(",")]:
        if str(layer) not in clusters:
            print(f"  layer {layer}: no clusters, skipping"); continue
        sae = load_sae(layer, a.l0, token=token); W = get_decoder(sae)
        reader = SparseReader(a.root, layer)
        label_pos = half_filter([reader.positions_of_tokens(ids) for ids in label_ids], a.half)
        support = int(sum(len(p) >= MIN_FIRING_PER_LABEL for p in label_pos))
        need = min_labels_with_support(full_k)
        power.append(dict(layer=layer, half=a.half, n_labels=full_k,
                          label_support=support, min_support=need,
                          underpowered=bool(support < need)))
        print(f"\nlayer {layer}: {support}/{full_k} labels with >= {MIN_FIRING_PER_LABEL} "
              f"scoring positions, need {need} "
              f"({'half B' if a.half == 'rank-even' else 'full cache'})"
              f"{'  -- UNDERPOWERED' if support < need else ''}", flush=True)
        cl = [c for c in clusters[str(layer)]["clusters"] if c["size"] >= MIN_RING_K]
        print(f"  {len(cl)} clusters with k >= {MIN_RING_K}", flush=True)

        # LABEL-FREE ranking on decoder directions
        ranked = []
        for c in cl:
            d = direction_ring_score(W[np.asarray(c["members"])], n_perm=2000,
                                     seed=abs(hash((layer, c["members"][0]))) % 10**6)
            ranked.append((c, d))
        ranked.sort(key=lambda t: t[1]["p_closed"])
        n_ring = sum(1 for _, d in ranked if d["p_closed"] < 0.05)
        print(f"  label-free: {n_ring} clusters have evenly-spread decoder directions (p<0.05)")

        for c, d in ranked[:a.top_k]:
            seed = abs(hash((layer, c["members"][0]))) % 10**6
            lab = labelled(reader, c["members"], W, label_pos, seed, full_k)
            if lab is None:
                continue
            iqr, n5, ip = index_concentration(c["members"], sae.d_sae,
                                              np.random.default_rng(seed), 300)
            rows.append(dict(layer=layer, metric=c["metric"], size=c["size"],
                             members=[int(x) for x in c["members"]],
                             dir_gap=d["gap_ratio"], dir_p=d["p_closed"],
                             index_p=float(ip), **lab))
            note = (f"  [closure underpowered: {lab['missing_run']} adjacent labels "
                    f"unsupported, a perfect ring here scores {lab['ref_p_closed']:.3f}]"
                    if lab["closure_underpowered"] else "")
            print(f"    [{c['metric'][:5]}] k={c['size']:3d} dirP {d['p_closed']:.3f} | "
                  f"rho {lab['rho']:+.3f} pOrd {lab['p_order']:.4f} | "
                  f"gap {lab['gap_ratio']:.2f} pClo {lab['p_closed']:.3f}{note}")

    if rows:
        ord_pass = benjamini_hochberg([r["p_order"] for r in rows])
        for r, k in zip(rows, ord_pass):
            r["ordering_fdr"] = bool(k)
            r["CIRCLE"] = bool(k and r["p_closed"] < 0.05)
            r["arc_only"] = bool(k and r["p_closed"] >= 0.05)
            # A closure failure on a survivor set that could not have passed closure
            # anyway is not evidence against a ring. The classification above is
            # unchanged -- this only separates "no ring" from "could not tell".
            r["closure_uninformative"] = bool(r["p_closed"] >= 0.05 and r["closure_underpowered"])
        print(f"\n{sum(ord_pass)}/{len(rows)} pass ordering after BH-FDR")
        print(f"{sum(r['CIRCLE'] for r in rows)}/{len(rows)} pass BOTH tiers -> CIRCLE")
        print(f"{sum(r['arc_only'] for r in rows)}/{len(rows)} ordering only -> ordered arc")
        n_uninf = sum(r["closure_uninformative"] for r in rows)
        if n_uninf:
            print(f"{n_uninf}/{len(rows)} failed closure with >= 2 adjacent labels "
                  f"unsupported -> underpowered on closure, NOT 'no ring'")
    Path(a.out).write_text(json.dumps(
        dict(ontology=a.ontology, l0=a.l0, root=a.root, half=a.half,
             power=power, rows=rows), indent=2))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
