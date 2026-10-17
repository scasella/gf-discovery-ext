"""Step 22 (DECLARED EXPLORATORY, not pre-registered) -- passer audit for the
window-level discovery run of `scripts/20_window_cluster.py` + `21_window_gate.py`.

The project's discipline is to inspect before believing. Step 17 dissolved the
previous apparent positive by showing that three layer-18 clusters passed ordering
at BH-FDR while containing ZERO month latents; step 19's absorption panel killed
the layer-7 arc by showing a single ' pump'/' sink' junk latent carried rho 0.965.
This script runs the same battery on every cluster the window-level gate tagged
CIRCLE or arc_only.

The battery, per audited cluster:

  1 CONTENT      each member latent's top tokens (HTTP-range read of the shipped
                 `examples.safetensors`, exactly as scripts 17/18), classified into
                 month / weekday / temporal / number-word families; plus the
                 Matryoshka index_p carried over from the gate row.
  2 ABSORPTION   decoder-block panel (PC1 share, participation ratio, effective
                 rank, pairwise cosine), PC1-deflation re-score on the SAME rows,
                 last-PC deflation as the "you removed nothing" control, an
                 energy-matched random-in-span control for the clusters that could
                 plausibly survive, and a full leave-one-member-out sweep naming
                 which members carry the ordering.
  3 ORIENTATION  rho / p_order under BOTH SVD sign conventions (the statistic is
                 not rotation-invariant), with an orientation_sensitive flag.
  4 WINDOW       the artifact specific to THIS run: window co-occupancy can be
                 driven by document TOPIC. Per cluster, the distribution of
                 co-occupancy pair-mass over windows and over documents, with the
                 top-10 shares. Concentration in a handful of documents is a red
                 flag.
  5 JOIN         months and numbers score the SAME frozen partition, so cluster id
                 X under months IS cluster id X under numbers. A cluster that
                 orders both a cyclic ontology and a non-cyclic ordinal control is
                 a generic 1-D gradient, not month geometry.

Fidelity. Scoring is `21_window_gate.labelled_local`, imported, not
re-implemented -- which is itself a proved-equal copy of `13_rerun_gate.labelled`.
The label means it consumes are rebuilt here from per-label MEAN ACTIVATION
vectors rather than from the full activation blocks, because
`(A @ Wc).mean(0) == A.mean(0) @ Wc` exactly; that identity is what makes the
leave-one-out sweep and the deflations cheap, and it is PROVED numerically
against the original `codes_at` path for every audited cluster before any variant
is scored. The per-cluster gate seed is held at the ORIGINAL
`gate_seed(layer, members[0])` for every variant, so leave-one-out deltas cannot
mix seed noise into the member effect.

Nothing here can change a gate verdict; the gate found no CIRCLE to change.
"""
import argparse, gc, importlib.util, json, os, sys, time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HUB_OFFLINE", "1")
sys.path.insert(0, str(REPO / "src"))
from gfdisc.saes import load_sae, get_decoder
from gfdisc.fits import degenerate
from gfdisc.hf_range import read_header, read_tensor


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


hcg = _load("hand_cluster_gate", REPO / "scripts" / "19_hand_cluster_gate.py")
w21 = _load("window_gate", REPO / "scripts" / "21_window_gate.py")
absn = _load("absorption", REPO / "scripts" / "19_absorption_l7.py")
insp = _load("inspect_passers", REPO / "scripts" / "17_inspect_passers.py")

gate = hcg.gate
BlockReader = hcg.BlockReader
SEQ = 255
STANDARD_PERMS = 20000

# Verbatim, as required of every output of this run.
DESIGN = w21.DESIGN

HF_BASE = ("https://huggingface.co/google/gemma-scope-2-1b-pt/resolve/main/"
           "resid_post_all")


# ---------------------------------------------------------------------------
# 0. Passer enumeration
# ---------------------------------------------------------------------------

def enumerate_passers(layers):
    """Every scored row across every gate file, keyed for the cross-ontology join."""
    rows, files = {}, {}
    for L in layers:
        p = REPO / f"results/window_gate_L{L}.json"
        if not p.exists():
            continue
        doc = json.loads(p.read_text())
        files[L] = dict(path=str(p), generated=doc.get("generated"),
                        clusters_file=doc["runs"]["months"]["clusters_file"])
        for ont, run in doc["runs"].items():
            for cell_key, cell in run["cells"].items():
                for r in cell["rows"]:
                    rows[(L, cell_key, int(r["cluster_id"]), ont)] = dict(
                        layer=L, cell=cell_key, cluster_id=int(r["cluster_id"]),
                        ontology=ont, W=cell["W"], arm=cell["arm"],
                        selected_threshold=cell["selected_threshold"],
                        rule_status=cell["rule_status"], row=r)
    return rows, files


def tier_calibration(rows):
    """How many passers should a run like this produce by chance?

    `results/background_harvest.json` measured the two tiers' false-acceptance
    rates directly, by pushing 2,000 random latent clusters through the identical
    gate: at layer 9 the ORDERING tier accepts 16.4% of random clusters
    (3.28x uniform) while the CLOSURE tier accepts 0.0% (min p_closed 0.079).
    The right question about 28 arc_only passers is therefore not "are they
    significant" but "are there more of them than a random clustering would
    give", and the answer is a rate comparison, not a p-value.
    """
    bgp = REPO / "results/background_harvest.json"
    bg = json.loads(bgp.read_text()) if bgp.exists() else None
    out = dict(source=str(bgp), by_ontology={}, per_cell_rates=[])
    for ont in ("months", "numbers"):
        sc = [v["row"] for k, v in rows.items()
              if k[3] == ont and v["row"].get("scorable")]
        if not sc:
            continue
        out["by_ontology"][ont] = dict(
            n_scorable=len(sc),
            n_ordering_raw=int(sum(r["ordering_pass_raw"] for r in sc)),
            rate_ordering_raw=float(np.mean([r["ordering_pass_raw"] for r in sc])),
            n_ordering_bh=int(sum(r["ordering_fdr"] for r in sc)),
            rate_ordering_bh=float(np.mean([r["ordering_fdr"] for r in sc])),
            n_closure=int(sum(r["closure_pass"] for r in sc)),
            rate_closure=float(np.mean([r["closure_pass"] for r in sc])),
            min_p_closed=float(min(r["p_closed"] for r in sc)),
            min_p_order=float(min(r["p_order"] for r in sc)))
    if bg:
        out["background"] = {
            f"layer_{L}": dict(
                n_scorable=bg["layers"][L]["item2_p_value_calibration"]["overall"]["n_scorable"],
                rate_order_le_0_05=bg["layers"][L]["item2_p_value_calibration"]
                    ["overall"]["p_order"]["le_0.05"]["fraction"],
                ratio_vs_uniform=bg["layers"][L]["item2_p_value_calibration"]
                    ["overall"]["p_order"]["le_0.05"]["ratio_vs_uniform"],
                rate_closure_le_0_05=bg["layers"][L]["item2_p_value_calibration"]
                    ["overall"]["p_closed"]["le_0.05"]["fraction"],
                min_p_closed=bg["layers"][L]["item2_p_value_calibration"]
                    ["overall"]["p_closed"]["min"])
            for L in bg["layers"]}
    out["per_layer_raw_ordering"] = {}
    for L in sorted({k[0] for k in rows}):
        rec = {}
        for ont in ("months", "numbers"):
            sc = [v["row"] for k, v in rows.items()
                  if k[0] == L and k[3] == ont and v["row"].get("scorable")]
            if sc:
                rec[ont] = dict(n_pass=int(sum(r["ordering_pass_raw"] for r in sc)),
                                n_scorable=len(sc),
                                rate=float(np.mean([r["ordering_pass_raw"] for r in sc])))
        rec["numbers_rate_exceeds_months"] = bool(
            "months" in rec and "numbers" in rec
            and rec["numbers"]["rate"] > rec["months"]["rate"])
        out["per_layer_raw_ordering"][str(L)] = rec
    out["numbers_exceeds_months_at_every_layer"] = bool(all(
        v["numbers_rate_exceeds_months"] for v in out["per_layer_raw_ordering"].values()))
    out["reading"] = (
        "Stated as rates rather than characterised. Measured background "
        "acceptance of the ORDERING tier on 2,000 random latent clusters through "
        "the identical gate: 16.4% at layer 9, 16.7% at layer 7. This run's raw "
        "ordering acceptance (p_order <= 0.05, before BH): months 17/74 = 23.0%, "
        "numbers 22/53 = 41.5%. The non-cyclic NUMBERS control is accepted at a "
        "strictly higher raw rate than months at EVERY layer 5-10 "
        "(numbers_exceeds_months_at_every_layer). Whatever the ordering tier is "
        "reading at W=32, it is not specific to a cyclic ontology. Measured "
        "background acceptance of the CLOSURE tier is 0.0% at layer 9 (min "
        "p_closed 0.079) and 0.4% at layer 7; this run accepted closure zero "
        "times, and min_p_closed_overall reports how far the nearest scorable "
        "cluster in the whole run stood from the 0.05 line.")
    allsc = [v["row"] for v in rows.values() if v["row"].get("scorable")]
    out["min_p_closed_overall"] = float(min(r["p_closed"] for r in allsc))
    out["n_scorable_overall"] = len(allsc)
    return out


def tag_of(r):
    if not r.get("scorable"):
        return "not_scorable"
    if r.get("CIRCLE"):
        return "CIRCLE"
    if r.get("arc_only"):
        return "arc_only"
    return "scored"


# ---------------------------------------------------------------------------
# 1. Mean-activation summaries (the identity that makes the battery cheap)
# ---------------------------------------------------------------------------

def label_summaries(reader, members, label_pos):
    """Per label: mean activation over its scoring rows, plus the support counts.

    `gate.labelled` keeps label i when len(pos) >= 5 AND at least
    MIN_FIRING_PER_LABEL of those rows have ANY cluster member firing, and then
    uses (A @ Wc).mean(0). Since that equals A.mean(0) @ Wc exactly, storing
    A.mean(0) is lossless for every member SUBSET too -- provided the support
    recount is done honestly, which is what `only_counts` is for: dropping member
    j removes exactly those firing rows where j was the ONLY member firing.
    """
    out = []
    for i, pos in enumerate(label_pos):
        if len(pos) < 5:
            out.append(None)
            continue
        A = reader.codes_at(pos, members)
        fired = A > 0
        rs = fired.sum(1)
        n_any = int((rs > 0).sum())
        only = np.zeros(len(members), dtype=np.int64)
        solo = np.flatnonzero(rs == 1)
        if len(solo):
            only = np.bincount(np.argmax(fired[solo], axis=1),
                               minlength=len(members)).astype(np.int64)
        out.append(dict(label=i, n_pos=int(len(pos)),
                        abar=A.mean(0, dtype=np.float64),
                        n_any=n_any, only=only))
        del A, fired, rs
    return out


def means_for_subset(summ, Wc, mask, full_k):
    """Rebuild (means, keep) for the member subset `mask`, with the support rule."""
    need = gate.min_labels_with_support(full_k)
    Wm = Wc[mask]
    means, keep = [], []
    for s in summ:
        if s is None:
            continue
        n_any = s["n_any"] - int(s["only"][~mask].sum())
        if n_any < gate.MIN_FIRING_PER_LABEL:
            continue
        means.append(s["abar"][mask] @ Wm)
        keep.append(s["label"])
    if len(keep) < need:
        return None, keep
    return (means, keep), keep


def score_variant(summ, Wc, mask, seed, full_k, n_perm, flip=False):
    cache, keep = means_for_subset(summ, Wc, mask, full_k)
    if cache is None:
        return dict(result="unscorable", reason="support rule",
                    n_labels=len(keep), labels=[int(x) for x in keep])
    if degenerate(np.vstack(cache[0])):
        return dict(result="degenerate", n_labels=len(cache[1]),
                    labels=[int(x) for x in cache[1]])
    lab = w21.labelled_local(None, None, None, None, seed, full_k,
                             n_perm=n_perm, flip=flip, cache=cache)
    if lab is None:
        return dict(result="none_from_labelled")
    return dict(result="scored", **{k: lab[k] for k in (
        "n_labels", "labels", "rho", "p_order", "gap_ratio", "max_gap_deg",
        "p_closed", "top2")})


# ---------------------------------------------------------------------------
# 4. Window co-occupancy concentration
# ---------------------------------------------------------------------------

def occupancy_bitmap(reader, feats, W, n_rows, chunk=40_000_000):
    """{latent -> bool array over window ids}. One streaming memmap pass.

    Same window arithmetic as `20_window_cluster.sample_occupancy_sets`: a latent
    occupies a window if it fires at ANY position inside it.
    """
    nwps = int(np.ceil(SEQ / W))
    n_seq = n_rows // SEQ
    n_win = n_seq * nwps
    feats = np.asarray(sorted(set(int(f) for f in feats)), dtype=np.int64)
    lut = np.full(65536, -1, dtype=np.int32)
    lut[feats] = np.arange(len(feats), dtype=np.int32)
    occ = np.zeros((len(feats), n_win), dtype=bool)
    idx, indptr = reader.indices, reader.indptr
    n = len(idx)
    for a in range(0, n, chunk):
        b = min(a + chunk, n)
        blk = np.asarray(idx[a:b])
        col = lut[blk]
        hit = np.flatnonzero(col >= 0)
        if len(hit):
            rows = np.searchsorted(indptr, hit.astype(np.int64) + a, side="right") - 1
            seq = rows // SEQ
            wid = seq * nwps + (rows - seq * SEQ) // W
            occ[col[hit], wid] = True
            del rows, seq, wid
        del blk, col, hit
    return {int(f): i for i, f in enumerate(feats)}, occ, nwps, n_win


def verify_occupancy(reader, index, occ, feats, W, nwps, n_seq, n_check=6, seed=0):
    """Brute-force check of the streamed bitmap on a sample of sequences.

    Deliberately shares no code with `occupancy_bitmap`: it re-reads whole rows
    through `SparseReader.row()` and rebuilds the window sets by hand.
    """
    rng = np.random.default_rng(seed)
    seqs = sorted(int(s) for s in rng.choice(n_seq, size=min(n_check, n_seq),
                                             replace=False))
    fs = sorted(int(f) for f in feats)[:20]
    fset = set(fs)
    mism = 0
    checked = 0
    for s in seqs:
        brute = {f: set() for f in fs}
        for off in range(SEQ):
            idx, _ = reader.row(s * SEQ + off)
            for j in idx:
                j = int(j)
                if j in fset:
                    brute[j].add(off // W)
        for f in fs:
            for wo in range(nwps):
                checked += 1
                if bool(occ[index[f], s * nwps + wo]) != (wo in brute[f]):
                    mism += 1
    return dict(n_sequences=len(seqs), n_latents=len(fs), n_cells_checked=checked,
                n_mismatches=int(mism), ok=bool(mism == 0),
                what=("streamed window-occupancy bitmap vs per-row SparseReader.row() "
                      "reads rebuilt into window sets by hand"))


def concentration(occ, index, members, nwps, arm, n_seq):
    """Pair-mass concentration of a cluster's window co-occupancy.

    m_w = how many members occupy window w. Pair-mass C(m_w, 2) is the number of
    member pairs that co-occupy w, i.e. exactly the quantity the P(j|i) graph
    integrates. If a handful of windows or documents carry most of it, the
    cluster's cohesion is a property of a few documents, not of the model.
    """
    rowsel = np.array([index[int(m)] for m in members], dtype=np.int64)
    m_w = occ[rowsel].sum(0).astype(np.int64)
    if arm == "even_build":                       # the graph's own build half
        wids = np.arange(len(m_w))
        keep = (wids // nwps) % 2 == 0
        m_w = np.where(keep, m_w, 0)
    pair = m_w * (m_w - 1) // 2
    total = int(pair.sum())
    doc = pair.reshape(-1, nwps).sum(1)
    occupied_w = int((pair > 0).sum())
    occupied_d = int((doc > 0).sum())

    def share(v, n):
        if total == 0:
            return None
        t = np.sort(v)[::-1][:n]
        return float(t.sum() / total)

    return dict(
        arm_half=("even-parity sequences only (the build half of the even_build "
                  "graph)" if arm == "even_build"
                  else "all sequences (the build half of the full graph)"),
        n_members=len(members), n_windows_per_sequence=int(nwps),
        n_sequences_total=int(n_seq),
        total_pair_mass=total,
        n_windows_with_any_cooccupancy=occupied_w,
        n_documents_with_any_cooccupancy=occupied_d,
        top10_window_share=share(pair, 10),
        top50_window_share=share(pair, 50),
        top10_document_share=share(doc, 10),
        top50_document_share=share(doc, 50),
        top1pct_window_share=(None if total == 0 else
                              share(pair, max(1, occupied_w // 100))),
        median_m_w_where_positive=(float(np.median(m_w[m_w > 0]))
                                   if (m_w > 0).any() else 0.0),
        max_m_w=int(m_w.max()),
        note=("pair-mass = sum over windows of C(m_w,2), m_w = members occupying "
              "the window. A cluster whose cohesion is spread over thousands of "
              "documents has a small top-10 share; concentration in a few "
              "documents is the topic-artifact signature this check exists for."))


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", default="5,6,7,8,9,10")
    ap.add_argument("--root", default="cache/sparse_big")
    ap.add_argument("--l0", default="big")
    ap.add_argument("--perms", type=int, default=STANDARD_PERMS)
    ap.add_argument("--n-random-control", type=int, default=200)
    ap.add_argument("--n-top-tokens", type=int, default=10)
    ap.add_argument("--out", default="results/window_passer_audit.json")
    ap.add_argument("--calibration-only", action="store_true",
                    help="recompute only the tier-calibration block and merge it "
                         "into an existing --out, leaving every other field alone")
    a = ap.parse_args()
    layers = [int(x) for x in a.layers.split(",")]
    t_start = time.time()

    rows, files = enumerate_passers(layers)
    if a.calibration_only:
        p = Path(a.out)
        doc = json.loads(p.read_text())
        doc["tier_calibration"] = tier_calibration(rows)
        doc["generated"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        p.write_text(json.dumps(doc, indent=2))
        print(f"merged tier_calibration into {p}")
        print(json.dumps(doc["tier_calibration"]["by_ontology"], indent=1))
        return
    passers = {k: v for k, v in rows.items() if tag_of(v["row"]) in ("CIRCLE", "arc_only")}
    # the audit set the task defines: every months-arm passer, plus every
    # numbers-control passer at the primary layer (the invalidity alarm).
    audit_keys = sorted(k for k, v in passers.items()
                        if v["ontology"] == "months" or (v["ontology"] == "numbers"
                                                         and v["layer"] == 9))
    print(f"{len(rows)} scored rows, {len(passers)} passers "
          f"({sum(1 for v in passers.values() if v['row']['CIRCLE'])} CIRCLE, "
          f"{sum(1 for v in passers.values() if v['row']['arc_only'])} arc_only); "
          f"auditing {len(audit_keys)}", flush=True)

    # ---- cross-ontology join over ALL passers (free: same frozen partition) ----
    join = []
    for (L, cell, cid, ont), v in sorted(passers.items()):
        other = "numbers" if ont == "months" else "months"
        o = rows.get((L, cell, cid, other))
        same_members = (o is not None and o["row"].get("members") == v["row"]["members"])
        join.append(dict(
            layer=L, cell=cell, cluster_id=cid, ontology=ont, size=v["row"]["size"],
            tag=tag_of(v["row"]), rho=v["row"]["rho"], p_order=v["row"]["p_order"],
            p_closed=v["row"]["p_closed"],
            other_ontology=other,
            other_present=o is not None,
            other_members_identical=bool(same_members),
            other_tag=(tag_of(o["row"]) if o else None),
            other_rho=(o["row"].get("rho") if o and o["row"].get("scorable") else None),
            other_p_order=(o["row"].get("p_order") if o and o["row"].get("scorable") else None),
            dual_ordering_pass=bool(o is not None and o["row"].get("ordering_fdr"))))

    from transformers import AutoTokenizer
    tk = AutoTokenizer.from_pretrained("google/gemma-3-1b-pt",
                                       token=os.environ.get("HF_TOKEN"))
    vocab = [(tk.convert_ids_to_tokens(i) or "").replace("▁", " ")
             for i in range(tk.vocab_size)]
    NUMWORDS = [w.lower() for w in gate.NUMS]

    tm = json.load(open(REPO / "results/token_map.json"))
    placement = {int(r["layer"]): [int(x) for x in r["label_latents"]]
                 for r in json.loads((REPO / "results/month_latent_placement.json").read_text())
                 if r["ontology"] == "months"}

    audited, notes = [], []
    for L in sorted({k[0] for k in audit_keys}):
        keys = [k for k in audit_keys if k[0] == L]
        t_layer = time.time()
        print(f"\n########## layer {L}: {len(keys)} clusters ##########", flush=True)

        u = f"{HF_BASE}/layer_{L}_width_16k_l0_{a.l0}/examples.safetensors"
        h, off = read_header(u, token=None)
        tt = np.asarray(read_tensor(u, "top_tokens", h, off, token=None))

        sae = load_sae(L, a.l0, token=os.environ.get("HF_TOKEN"))
        Wdec = get_decoder(sae)          # native dtype: the gate's own matrix
        del sae
        reader = BlockReader(a.root, L)
        n_rows = len(reader)
        n_seq = n_rows // SEQ

        # label positions, per ontology, per arm -- exactly script 21's row sets
        lp = {}
        for ont in sorted({k[3] for k in keys}):
            names, ids = gate.label_ids_for(ont, tm)
            allpos = [reader.positions_of_tokens(x) for x in ids]
            lp[ont] = dict(names=names, full_k=len(names),
                           even_build=w21.odd_sequence_filter(allpos),
                           full=gate.half_filter(allpos, "rank-even"))

        # one occupancy pass per layer over the union of audited members
        union = sorted({int(m) for k in keys for m in rows[k]["row"]["members"]})
        Wwin = int(rows[keys[0]]["W"])
        assert all(int(rows[k]["W"]) == Wwin for k in keys), "mixed W at one layer"
        t0 = time.time()
        occ_index, occ, nwps, n_win = occupancy_bitmap(reader, union, Wwin, n_rows)
        occ_check = verify_occupancy(reader, occ_index, occ, union, Wwin, nwps, n_seq)
        notes.append(dict(layer=L, occupancy_verification=occ_check))
        print(f"  occupancy: {len(union)} latents x {n_win:,} windows "
              f"({occ.nbytes/1e6:.0f} MB) in {time.time()-t0:.0f}s; "
              f"brute-force check {occ_check['n_mismatches']} mismatches in "
              f"{occ_check['n_cells_checked']} cells", flush=True)
        assert occ_check["ok"], f"occupancy bitmap disagrees with brute force at L{L}"

        for key in keys:
            v = rows[key]
            r = v["row"]
            ont, arm = v["ontology"], v["arm"]
            members = [int(x) for x in r["members"]]
            k = len(members)
            full_k = lp[ont]["full_k"]
            seed = hcg.gate_seed(L, members[0])          # frozen for every variant
            label_pos = lp[ont][arm]
            t0 = time.time()
            print(f"  -- L{L} {ont} {v['cell']} cid{v['cluster_id']} k={k} "
                  f"rho {r['rho']:+.3f} pOrd {r['p_order']:.4f} "
                  f"pClo {r['p_closed']:.3f}", flush=True)

            # --- fidelity: reproduce the published row on the ORIGINAL path ----
            ref = w21.labelled_local(reader, members, Wdec, label_pos, seed, full_k,
                                     n_perm=a.perms, flip=False)
            repro = dict(
                reproduced=bool(ref is not None
                                and ref["rho"] == r["rho"]
                                and ref["p_order"] == r["p_order"]
                                and ref["p_closed"] == r["p_closed"]),
                rho=(ref or {}).get("rho"), p_order=(ref or {}).get("p_order"),
                p_closed=(ref or {}).get("p_closed"),
                what=("21_window_gate.labelled_local re-run through the original "
                      "codes_at path; tier fields only. ref_p_closed is a methods "
                      "note whose value depends on within-process scoring order "
                      "(reference_ring caches on the survivor set without the "
                      "seed) and is deliberately not compared."))

            # --- the abar identity, proved before any variant is scored --------
            summ = label_summaries(reader, members, label_pos)
            Wc = np.asarray(Wdec[np.asarray(members)], dtype=np.float64)
            full_mask = np.ones(k, dtype=bool)
            std = score_variant(summ, Wc, full_mask, seed, full_k, a.perms, flip=False)
            alt = score_variant(summ, Wc, full_mask, seed, full_k, a.perms, flip=True)
            ok = std.get("result") == "scored" and ref is not None
            identity = dict(
                abar_rho=std.get("rho"), codes_at_rho=(ref or {}).get("rho"),
                abar_p_order=std.get("p_order"),
                codes_at_p_order=(ref or {}).get("p_order"),
                abar_p_closed=std.get("p_closed"),
                codes_at_p_closed=(ref or {}).get("p_closed"),
                d_rho=(abs(std["rho"] - ref["rho"]) if ok else None),
                d_gap_deg=(abs(std["max_gap_deg"] - ref["max_gap_deg"]) if ok else None),
                tier_fields_identical=bool(
                    ok and std["rho"] == ref["rho"] and std["p_order"] == ref["p_order"]
                    and std["p_closed"] == ref["p_closed"]),
                labels_identical=bool(ok and std["labels"] == ref["labels"]),
                what=("(A @ Wc).mean(0) == A.mean(0) @ Wc; the mean-activation "
                      "rebuild is scored through the same labelled_local and must "
                      "return the identical tier fields before any leave-one-out "
                      "or deflation variant is trusted. The rebuild accumulates in "
                      "float64 where the reference path accumulates in the decoder's "
                      "native float32, so d_rho is reported rather than assumed zero."))

            # --- 3. orientation ------------------------------------------------
            orientation = dict(
                standard=dict(rho=std.get("rho"), p_order=std.get("p_order")),
                alt=dict(rho=alt.get("rho"), p_order=alt.get("p_order")),
                published_orientation_sensitive=r["orientation_sensitive"],
                recomputed_orientation_sensitive=bool(
                    std.get("result") == "scored" and alt.get("result") == "scored"
                    and (std["p_order"] < 0.05) != (alt["p_order"] < 0.05)),
                closure_identical_under_reflection=bool(
                    std.get("result") == "scored" and alt.get("result") == "scored"
                    and std["p_closed"] == alt["p_closed"]),
                verdict_stable=None)
            orientation["verdict_stable"] = bool(
                not orientation["recomputed_orientation_sensitive"])

            # --- 1. content ----------------------------------------------------
            per = []
            for m in members:
                toks = [vocab[t] for t in tt[m][:a.n_top_tokens] if 0 <= t < len(vocab)]
                cls = insp.classify(toks)
                nums = sorted({w for t in toks
                               for w in NUMWORDS if t.strip().lower() == w})
                per.append(dict(latent=int(m), tokens=toks,
                                month=sorted(cls["month"]), day=sorted(cls["day"]),
                                temporal=sorted(cls["temporal"]), number=nums))
            month_lat = set(placement.get(L, []))
            content = dict(
                n_month_token_latents=sum(1 for p in per if p["month"]),
                n_day_token_latents=sum(1 for p in per if p["day"]),
                n_temporal_or_calendar=sum(1 for p in per if p["temporal"] or p["month"]
                                           or p["day"]),
                n_number_word_latents=sum(1 for p in per if p["number"]),
                month_latents_by_placement=sorted(month_lat & set(members)),
                n_month_latents_by_placement=len(month_lat & set(members)),
                placement_criterion=(">=2 distinct month names among a latent's top "
                                     "tokens, from results/month_latent_placement.json "
                                     "(script 18's basis)"),
                index_p=r["index_p"], index_iqr=r["index_iqr"],
                matryoshka_prefix_flag=bool(r["index_p"] < 0.05),
                matryoshka_note=("index_p is the label-free index-concentration "
                                 "p-value; < 0.05 means the members sit in a "
                                 "narrower band of latent indices than chance, the "
                                 "Matryoshka-prefix signature"),
                frac_members_index_lt_1024=float(
                    np.mean([m < 1024 for m in members])),
                members=per)
            print(f"     content: month-token {content['n_month_token_latents']}/{k}, "
                  f"placement-month {content['n_month_latents_by_placement']}/{k}, "
                  f"temporal {content['n_temporal_or_calendar']}/{k}, "
                  f"number-word {content['n_number_word_latents']}/{k}, "
                  f"index_p {r['index_p']:.3f}", flush=True)

            # --- 2. absorption panel + deflations ------------------------------
            pan = absn.panel(Wc)
            Vt_span, S_span = absn.span_basis(Wc)
            rank = len(Vt_span)
            u1 = Vt_span[0]
            defl = {}
            for name, dirs in (("pc1", Vt_span[:1]), ("last_pc", Vt_span[-1:])):
                Wd, energy = absn.deflate(Wc, dirs)
                s = score_variant(summ, Wd, full_mask, seed, full_k, a.perms)
                defl[name] = dict(energy_removed=energy, **s)
            m_match = max(1, int(round(rank * pan["pc1_var_share_uncentered"])))
            month_bearing = content["n_month_latents_by_placement"] > 0
            rand = None
            if month_bearing and a.n_random_control > 0:
                rng = np.random.default_rng(seed)
                vals, energies = [], []
                for _ in range(a.n_random_control):
                    Wd, e = absn.deflate(Wc, absn.random_in_span(Vt_span, rng, m=m_match))
                    s = score_variant(summ, Wd, full_mask, seed, full_k, a.perms)
                    vals.append(s.get("rho", np.nan) if s.get("result") == "scored"
                                else np.nan)
                    energies.append(e)
                vv = np.asarray([x for x in vals if np.isfinite(x)], dtype=float)
                pc1_rho = defl["pc1"].get("rho")
                rand = dict(
                    n=int(a.n_random_control), m_directions=int(m_match),
                    span_rank=int(rank),
                    energy_removed=dict(mean=float(np.mean(energies)),
                                        min=float(np.min(energies)),
                                        max=float(np.max(energies))),
                    n_scored=int(len(vv)),
                    rho=dict(mean=float(vv.mean()) if len(vv) else None,
                             p05=float(np.percentile(vv, 5)) if len(vv) else None,
                             p50=float(np.percentile(vv, 50)) if len(vv) else None,
                             p95=float(np.percentile(vv, 95)) if len(vv) else None),
                    pc1_rho_percentile=(absn.pctile_at_or_below(vv, pc1_rho)
                                        if (len(vv) and pc1_rho is not None) else None),
                    what=("energy-matched control: m isotropic directions inside the "
                          "member block's own row space, m chosen so the expected "
                          "energy removed matches the PC1 deflation. The answer to "
                          "'you removed X% of the energy, of course rho moved'."))
            absorption = dict(panel=pan, span_rank=int(rank),
                              deflation=defl, random_in_span=rand,
                              deflation_note=("the deflated block is re-scored on "
                                              "the SAME rows, same seed, same "
                                              "support rule, same 20,000 "
                                              "permutations"))

            # --- 2b. leave-one-member-out sweep --------------------------------
            loo = []
            base_rho = std.get("rho")
            base_p = std.get("p_order")
            for j in range(k):
                mask = full_mask.copy()
                mask[j] = False
                s = score_variant(summ, Wc, mask, seed, full_k, a.perms)
                loo.append(dict(dropped=int(members[j]), **{
                    kk: s.get(kk) for kk in ("result", "rho", "p_order", "p_closed",
                                             "n_labels")}))
            scored_loo = [x for x in loo if x["result"] == "scored"]
            rhos = np.asarray([x["rho"] for x in scored_loo], dtype=float)
            kills = [x for x in scored_loo
                     if base_p is not None and base_p <= 0.05 and x["p_order"] > 0.05]
            worst = sorted(scored_loo, key=lambda x: x["rho"])[:5]
            loo_sum = dict(
                base_rho=base_rho, base_p_order=base_p,
                n_variants=len(loo), n_scored=len(scored_loo),
                n_unscorable=int(sum(1 for x in loo if x["result"] != "scored")),
                unscorable_members=[x["dropped"] for x in loo if x["result"] != "scored"],
                rho_min=float(rhos.min()) if len(rhos) else None,
                rho_max=float(rhos.max()) if len(rhos) else None,
                rho_median=float(np.median(rhos)) if len(rhos) else None,
                max_abs_delta_rho=(float(np.abs(rhos - base_rho).max())
                                   if (len(rhos) and base_rho is not None) else None),
                n_members_whose_removal_breaks_ordering=len(kills),
                members_whose_removal_breaks_ordering=[x["dropped"] for x in kills],
                load_bearing=bool(len(kills) > 0),
                most_load_bearing=[dict(latent=x["dropped"], rho=x["rho"],
                                        p_order=x["p_order"]) for x in worst],
                note=("a single member whose removal destroys the ordering is the "
                      "layer-7 failure mode: one junk latent carried rho 0.965. A "
                      "sweep in which no single removal matters is consistent with "
                      "distributed structure -- and also with a generic gradient."),
                per_member=loo)
            print(f"     LOO: rho in [{loo_sum['rho_min']}, {loo_sum['rho_max']}], "
                  f"{loo_sum['n_members_whose_removal_breaks_ordering']} load-bearing; "
                  f"PC1-deflation rho "
                  f"{defl['pc1'].get('rho')} p {defl['pc1'].get('p_order')} "
                  f"(energy {defl['pc1']['energy_removed']:.3f})", flush=True)

            # --- 4. window concentration ---------------------------------------
            conc = concentration(occ, occ_index, members, nwps, arm, n_seq)
            print(f"     windows: pair-mass {conc['total_pair_mass']:,} over "
                  f"{conc['n_documents_with_any_cooccupancy']:,} documents; "
                  f"top-10 windows {conc['top10_window_share']}, "
                  f"top-10 docs {conc['top10_document_share']}", flush=True)

            jrow = next(x for x in join if (x["layer"], x["cell"], x["cluster_id"],
                                            x["ontology"]) == key)
            audited.append(dict(
                key=f"L{L}|{v['cell']}|cid{v['cluster_id']}|{ont}",
                layer=L, cell=v["cell"], cluster_id=v["cluster_id"], ontology=ont,
                W=v["W"], arm=arm, selected_threshold=v["selected_threshold"],
                size=k, members=members, seed=int(seed), tag=tag_of(r),
                published=dict(rho=r["rho"], p_order=r["p_order"],
                               p_closed=r["p_closed"], gap_ratio=r["gap_ratio"],
                               n_labels=r["n_labels"], labels=r["labels"],
                               closure_underpowered=r["closure_underpowered"],
                               missing_run=r["missing_run"],
                               ordering_fdr=r["ordering_fdr"],
                               CIRCLE=r["CIRCLE"], arc_only=r["arc_only"]),
                reproduction=repro, abar_identity=identity,
                cross_ontology=jrow,
                content=content, absorption=absorption, leave_one_out=loo_sum,
                orientation=orientation, window_concentration=conc,
                seconds=round(time.time() - t0, 1)))
            del summ
            gc.collect()

        del occ, Wdec, reader, tt
        gc.collect()
        print(f"###### layer {L} done in {time.time()-t_layer:.0f}s", flush=True)

    # ---------------- verdicts ------------------------------------------------
    for A in audited:
        c, ab, lo, wc = A["content"], A["absorption"], A["leave_one_out"], A["window_concentration"]
        reasons = []
        if A["ontology"] == "numbers":
            reasons.append("this row IS the ordinal control -- it is an alarm about "
                           "the ordering tier's specificity, never a discovery")
        if c["n_month_latents_by_placement"] == 0 and c["n_month_token_latents"] == 0:
            reasons.append(f"0/{A['size']} month latents")
        if A["ontology"] == "months" and A["cross_ontology"]["dual_ordering_pass"]:
            reasons.append("the same cluster also passes ordering on the non-cyclic "
                           "numbers control, so the statistic is reading a generic "
                           "1-D gradient")
        if c["matryoshka_prefix_flag"]:
            reasons.append(f"index_p {c['index_p']:.3f} (Matryoshka-prefix cluster)")
        p1 = ab["deflation"]["pc1"]
        if p1.get("result") == "scored" and p1.get("p_order") is not None:
            if A["published"]["p_order"] <= 0.05 < p1["p_order"]:
                reasons.append(f"ordering dies under PC1 deflation "
                               f"(p_order {p1['p_order']:.3f})")
        if lo["load_bearing"]:
            reasons.append(f"{lo['n_members_whose_removal_breaks_ordering']} single "
                           "member(s) carry the ordering")
        if wc["top10_document_share"] is not None and wc["top10_document_share"] > 0.25:
            reasons.append(f"top-10 documents carry "
                           f"{100*wc['top10_document_share']:.0f}% of co-occupancy")
        A["closure_failed"] = not A["published"]["CIRCLE"]
        A["verdict_reasons"] = reasons
        if reasons:
            A["verdict"] = f"explained away ({'; '.join(reasons)})"
        elif A["published"]["CIRCLE"]:
            A["verdict"] = "survives inspection"
        else:
            A["verdict"] = ("ambiguous (ordering survives every check above, but the "
                            "cluster never passed closure, so there is no ring "
                            "claim to survive; a CIRCLE-tagged cluster at W>=128 "
                            "with month latents would settle it)")

    doc = dict(
        design=DESIGN,
        status="exploratory, not pre-registered",
        script=str(Path(__file__).resolve()),
        inputs=dict(gate_files=files,
                    root=a.root, l0=a.l0, n_perm=a.perms,
                    n_random_control=a.n_random_control),
        headline=dict(
            n_scored_rows=len(rows), n_passers=len(passers),
            n_CIRCLE=int(sum(1 for v in passers.values() if v["row"]["CIRCLE"])),
            n_arc_only=int(sum(1 for v in passers.values() if v["row"]["arc_only"])),
            n_audited=len(audited),
            discovery_fired=bool(any(v["row"]["CIRCLE"] for v in passers.values())),
            statement=("ZERO clusters were tagged CIRCLE in any cell of any layer "
                       "under either ontology. The two-tier gate did not pass; the "
                       "window-level discovery did NOT fire. Every passer is "
                       "arc_only (ordering at BH-FDR, closure rejected) and every "
                       "one is in a W=32 cell.")),
        primary_cell=dict(
            cell="layer 9, W=255, arm even_build",
            rule_status="rule_failure",
            n_clusters_scored=0,
            statement=("The a-priori primary cell produced NO partition and scored "
                       "NO cluster: the frozen threshold rule was not satisfied by "
                       "any EVALUATED threshold, and T in {0.25,0.35,0.45,0.55,0.65} "
                       "could not be evaluated at all (edge cap). The same holds "
                       "for every W=128 and W=255 cell at every layer 5-10. This is "
                       "an instrument limitation, not a measured null: the primary "
                       "cell was never tested. Every passer in this run comes from "
                       "a W=32 robustness cell."),
            near_miss_branch=("not triggered: passers exist (28 arc_only), so the "
                              "battery ran on them. The primary cell has zero "
                              "scored rows, so it has no near-misses to rank "
                              "either.")),
        invalidity_alarm=dict(
            criterion=("the run's own stated criterion: the number-word ontology is "
                       "the ordinal control and must never pass CLOSURE; a run in "
                       "which it does is invalid rather than interesting."),
            numbers_CIRCLE_anywhere=int(sum(1 for k, v in passers.items()
                                            if k[3] == "numbers" and v["row"]["CIRCLE"])),
            numbers_arc_only_total=int(sum(1 for k, v in passers.items()
                                           if k[3] == "numbers" and v["row"]["arc_only"])),
            numbers_arc_only_at_L9=sorted(
                f"{k[1]}|cid{k[2]}" for k in passers if k[3] == "numbers" and k[0] == 9),
            verdict=("NOT invalid by the stated criterion -- the numbers control "
                     "never passes closure. But it passes ORDERING at BH-FDR in 16 "
                     "cells including 3 at layer 9, which impugns the specificity "
                     "of the ordering tier at W=32: an ordinal, non-cyclic label "
                     "set is ordered just as readily as the calendar. Treat every "
                     "W=32 arc_only as evidence about generic 1-D gradients, not "
                     "about month geometry.")),
        tier_calibration=tier_calibration(rows),
        cross_ontology_join=dict(
            basis=("months and numbers score the SAME frozen partition file, so "
                   "cluster id X under months IS cluster id X under numbers; the "
                   "join needs no set comparison and member lists are checked "
                   "identical."),
            rows=join,
            n_dual_ordering_passers=int(sum(1 for j in join if j["dual_ordering_pass"]))),
        verification=notes,
        month_latent_placement=dict(
            source="results/month_latent_placement.json (script 18)",
            per_layer={str(L): sorted(v) for L, v in placement.items()}),
        audited=audited,
        caveats=[
            "DECLARED EXPLORATORY. Not pre-registered. Nothing here can change a "
            "pre-registered gate verdict, and there was no CIRCLE to change.",
            "The primary cell (L9/W255/even_build) is rule_failure, not a measured "
            "null: thresholds 0.25-0.65 exceeded the 5M-edge computational cap and "
            "were never evaluated. Every audited passer is from a W=32 cell, which "
            "was declared robustness, not primary.",
            "BH-FDR was applied WITHIN each cell by the gate. Cells share a cache, a "
            "layer and nested windows, so a passer in one cell is not replicated by "
            "a passer in another, and no cross-cell correction exists.",
            "The absorption deflations re-score on the same rows with the same seed; "
            "the energy-matched random-in-span control was run only for the "
            "month-bearing passers, because a cluster with zero month latents that "
            "also orders the numbers control is already explained on content.",
            "The window-concentration check uses the BUILD half of each arm (even "
            "sequences for even_build, all sequences for full) because that is the "
            "co-occupancy that actually formed the graph; the gate scored the "
            "complementary half.",
            "cyclic_order_score is not rotation-invariant. rho/p_order are reported "
            "under the standard SVD sign convention (primary, for comparability) "
            "and under the reflected one, with a stability flag.",
            "ref_p_closed is a methods note, not a tier: reference_ring caches on "
            "the survivor set without the seed, so its value depends on scoring "
            "order within a process. It is deliberately excluded from the "
            "reproduction check.",
        ],
        runtime_seconds=round(time.time() - t_start, 1),
        generated=time.strftime("%Y-%m-%dT%H:%M:%S"))

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=2))
    print(f"\nwrote {out}  ({doc['runtime_seconds']}s)")
    for A in audited:
        print(f"  {A['key']:44s} k={A['size']:3d}  {A['verdict'][:120]}")


if __name__ == "__main__":
    main()
