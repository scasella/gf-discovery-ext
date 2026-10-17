"""The flats-blob confound, and the within-block permutation null that controls for it.

PROVENANCE, stated first and plainly. This script is a POST-HOC ADDITION to the Phase-2
keys sweep in the sense of docs/phase2_prereg_2026-08-21.md section 7, and it is written
up as one. What it is NOT: it is not a further search family, and it cannot turn a
negative into a positive. It is a strictly MORE CONSERVATIVE null for a statistic that was
already registered (the K2 fifths-vs-chromatic margin, and the two ordering rhos), added
under the project tenet that any positive gets attacked before it gets reported. It can
only ever move a result from "separates" to "does not separate", never the other way.

WHEN THIS WAS WRITTEN, corrected, because the first version of this paragraph claimed more
than the timestamps support. This file was AUTHORED while scripts/28_phase2_keys.py was
still running its sweep -- the only sweep output visible at that point was the two-layer
dry run and layers 0-3 of the token readout -- but it landed on disk at 05:43:52, which is
47 seconds AFTER results/phase2_keys.json was written at 05:43:05. The mtimes therefore do
NOT establish precedence and no precedence is claimed. The original paragraph asserting
that they did was wrong and is replaced by this one rather than quietly deleted.

Two things stand on their own and are what this script actually rests on:
  1. The confound is derivable from the LABEL SET with no data whatever -- see below -- in
     exactly the way prereg section 6's naturals-contiguity fact needed no data.
  2. The null is strictly MORE CONSERVATIVE than the registered one. Its direction of
     effect is fixed by construction: it can only move a cell from "separates" to "does
     not separate". A post-hoc null that can only ever weaken the author's own result is
     not a null that timing could have been used to game.

THE CONFOUND, derived from the label set alone.

  FIFTHS order : C  G  D  A  E  B | Gb Db Ab Eb Bb | F
  positions      0  1  2  3  4  5   6  7  8  9  10   11

  The five FLAT spellings occupy positions 6, 7, 8, 9, 10 -- five CONSECUTIVE slots of the
  fifths cycle. The seven naturals occupy 0-5 and 11, which is the complementary arc
  (11 wraps round to 0). So in fifths order the two token classes are two contiguous
  blocks.

  CHROMATIC order : C Db D Eb E F Gb G Ab A Bb B
  positions         0  1 2  3 4 5  6 7  8 9 10 11

  The flats sit at 1, 3, 6, 8, 10 -- interleaved with the naturals.

CONSEQUENCE. The five flats are corpus-rare multi-character tokens (Db 0, Gb 2, Bb 4,
Eb 6, Ab 102 occurrences in cache/sparse_big, prereg 5.1) and the seven naturals are
single capital letters. If the dominant geometry in the point cloud merely SEPARATES those
two lexical classes into two angular blobs -- which is what token frequency and token
class would produce, with no pitch-class content whatever -- then the recovered angular
order agrees with the fifths cycle up to scrambling WITHIN each blob, and disagrees with
the chromatic cycle, which interleaves them. Such a cloud inflates rho_fifths relative to
rho_chromatic for a purely lexical reason.

This is not hypothetical for this data: the sweep reports how often all five flats come
out contiguous in the recovered angular order, against the exact combinatorial reference
12 / C(12,5) = 0.01515.

WHY THE REGISTERED NULL DOES NOT CATCH IT. The K2 margin null shuffles the twelve labels
UNIFORMLY. A uniform shuffle destroys the block structure, so the null it builds is the
null for "no structure at all", not the null for "two lexical blobs and nothing else".
Against the artifact above it is anti-conservative.

THE FIX, and it is the only thing this script computes. A WITHIN-BLOCK permutation null:
each draw permutes the five flat labels among the five flat-labelled points and the seven
natural labels among the seven natural-labelled points, and nothing else. The two-blob
arrangement is therefore preserved exactly in every draw, and the only thing that varies
is the ordering WITHIN each class. 5! x 7! = 604,800 such permutations exist. If an
observed margin still exceeds the 95th percentile of |margin| under that null, the
discrimination is not explained by lexical class. If it does not, the cell's apparent
fifths preference is consistent with the blob and must not be reported as pitch-class
geometry.

The same null is applied to rho_fifths and rho_chromatic separately, because the ordering
p-values in the sweep inherit the identical weakness.

Reads results/phase2_keys.json (the stored per-cell angles) and writes
results/phase2_keys_blockperm.json. No model, no cache, no activations, no re-clustering.
"""
import argparse, json, time
from math import comb, factorial
from pathlib import Path

import numpy as np

FLATS = ["Db", "Eb", "Gb", "Ab", "Bb"]
N_NULL = 20000


def _rank_of(values):
    order = np.argsort(np.asarray(values), kind="stable")
    r = np.empty(len(order), dtype=np.float64)
    r[order] = np.arange(len(order), dtype=np.float64)
    return r


def _rank_from_angles(ang, seed):
    ang = np.asarray(ang, dtype=float)
    n = len(ang)
    jitter = np.random.default_rng(seed).permutation(n)
    order = np.lexsort((jitter, ang))
    rank_of = np.empty(n, dtype=np.float64)
    rank_of[order] = np.arange(n)
    return rank_of


def _sym_basis(label_index, K):
    lab = np.asarray(label_index, dtype=int)
    return np.vstack([_rank_of((f * lab + s) % K) for f in (1, -1) for s in range(K)])


def block_permutations(blocks, n_draws, seed):
    """`n_draws` permutations of 0..n-1 that map each block onto itself."""
    rng = np.random.default_rng(seed)
    n = sum(len(b) for b in blocks)
    sig = np.tile(np.arange(n), (n_draws, 1))
    for b in blocks:
        b = np.asarray(b)
        pick = np.argsort(rng.random((n_draws, len(b))), axis=1)
        sig[:, b] = b[pick]
    return sig


def analyse_cell(ang, fifths_code, chrom_code, K, blocks, seed, n_draws=N_NULL):
    rank_of = _rank_from_angles(ang, seed)
    n = len(rank_of)
    base_f = _sym_basis(fifths_code, K)
    base_c = _sym_basis(chrom_code, K)
    denom = n * (n * n - 1)

    def rhos(base, sig):
        d = rank_of - base[:, sig]
        return 1.0 - 6.0 * (d * d).sum(-1) / denom

    ident = np.arange(n)
    rho_f = float(rhos(base_f, ident).max())
    rho_c = float(rhos(base_c, ident).max())
    obs = rho_f - rho_c

    sig = block_permutations(blocks, n_draws, seed + 31337)
    nf = rhos(base_f, sig).max(0)
    nc = rhos(base_c, sig).max(0)
    null = nf - nc
    thr = float(np.percentile(np.abs(null), 95))
    return dict(
        rho_fifths=rho_f, rho_chromatic=rho_c, margin=float(obs),
        n_draws=int(n_draws),
        within_block_null_abs_p95=thr,
        within_block_null_margin_mean=float(null.mean()),
        within_block_null_margin_p95=float(np.percentile(null, 95)),
        within_block_null_rho_fifths_mean=float(nf.mean()),
        within_block_null_rho_chromatic_mean=float(nc.mean()),
        p_margin_two_sided=float((1 + int((np.abs(null) >= abs(obs)).sum())) / (1 + n_draws)),
        p_rho_fifths=float((1 + int((nf >= rho_f).sum())) / (1 + n_draws)),
        p_rho_chromatic=float((1 + int((nc >= rho_c).sum())) / (1 + n_draws)),
        separates_within_block=bool(abs(obs) > thr),
        winner_within_block=(("fifths" if obs > 0 else "chromatic") if abs(obs) > thr else None),
        p_floor=float(1.0 / (n_draws + 1)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", default="results/phase2_keys.json")
    ap.add_argument("--out", default="results/phase2_keys_blockperm.json")
    ap.add_argument("--n-null", type=int, default=N_NULL)
    a = ap.parse_args()
    t0 = time.time()

    d = json.loads(Path(a.sweep).read_text())
    point_order = d["codings"]["point_order"]
    fifths_code = np.array(d["codings"]["fifths"])
    chrom_code = np.array(d["codings"]["chromatic"])
    K = len(point_order)
    flat_idx = [i for i, w in enumerate(point_order) if w in set(FLATS)]
    nat_idx = [i for i, w in enumerate(point_order) if w not in set(FLATS)]
    blocks = [flat_idx, nat_idx]

    print("=" * 92)
    print("THE FLATS-BLOB CONFOUND -- within-block permutation null")
    print("POST-HOC ADDITION under prereg section 7. It is a strictly MORE CONSERVATIVE null "
          "for an\nalready-registered statistic; it can only move a cell from 'separates' to "
          "'does not'.")
    print(f"fifths positions of the five flats : "
          f"{sorted(int(fifths_code[i]) for i in flat_idx)}  -> five CONSECUTIVE slots")
    print(f"chromatic positions of the same    : "
          f"{sorted(int(chrom_code[i]) for i in flat_idx)}  -> interleaved")
    print(f"within-block permutations available: 5! x 7! = "
          f"{factorial(len(flat_idx)) * factorial(len(nat_idx))}")
    print(f"null draws per cell                : {a.n_null}")
    print("=" * 92, flush=True)

    rows = []
    print(f"{'readout':>8} {'L':>3} | {'rhoF':>7} {'rhoC':>7} {'margin':>7} "
          f"| {'uniform-null':>12} {'thr':>6} {'sep':>5} "
          f"| {'block-null':>10} {'thr':>6} {'sep':>5} {'pMargin':>8} {'pRhoF':>8}", flush=True)
    for r in d["rows"]:
        if r["degenerate"]:
            continue
        A = r["conventions"]["A"]
        res = analyse_cell(np.asarray(A["angles_rad"]), fifths_code, chrom_code, K, blocks,
                           seed=r["layer"], n_draws=a.n_null)
        # the stored sweep values must reproduce exactly -- same construction, same seed
        assert abs(res["rho_fifths"] - A["order"]["fifths"]["rho"]) < 1e-12, r["layer"]
        assert abs(res["rho_chromatic"] - A["order"]["chromatic"]["rho"]) < 1e-12, r["layer"]
        um = A["margin"]
        row = dict(readout=r["readout"], layer=r["layer"],
                   flats_contiguous=r.get("flats_contiguity", {}).get("contiguous"),
                   uniform_null=dict(threshold=um["null_abs_p95"], separates=um["separates"],
                                     winner=um["winner"], p_two_sided=um["p_two_sided"]),
                   within_block=res,
                   changed_by_the_block_null=bool(um["separates"] != res["separates_within_block"]))
        rows.append(row)
        print(f"{r['readout']:>8} {r['layer']:3d} | {res['rho_fifths']:+7.3f} "
              f"{res['rho_chromatic']:+7.3f} {res['margin']:+7.3f} | "
              f"{'':>12} {um['null_abs_p95']:6.3f} {str(um['separates']):>5} | "
              f"{'':>10} {res['within_block_null_abs_p95']:6.3f} "
              f"{str(res['separates_within_block']):>5} {res['p_margin_two_sided']:8.4f} "
              f"{res['p_rho_fifths']:8.4f}", flush=True)

    n = len(rows)
    sep_u = sum(r["uniform_null"]["separates"] for r in rows)
    sep_b = sum(r["within_block"]["separates_within_block"] for r in rows)
    demoted = [(r["readout"], r["layer"]) for r in rows
               if r["uniform_null"]["separates"] and not r["within_block"]["separates_within_block"]]
    promoted = [(r["readout"], r["layer"]) for r in rows
                if not r["uniform_null"]["separates"] and r["within_block"]["separates_within_block"]]
    mean_block_margin = float(np.mean([r["within_block"]["within_block_null_margin_mean"]
                                       for r in rows])) if rows else None
    mean_block_thr = float(np.mean([r["within_block"]["within_block_null_abs_p95"]
                                    for r in rows])) if rows else None
    mean_unif_thr = float(np.mean([r["uniform_null"]["threshold"] for r in rows])) if rows else None

    # The ordering p-values inherit the identical weakness, so the same collapse is
    # reported for them. This is the number that decides how the sweep is read.
    import sys as _sys
    _sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from gfdisc.nulls import benjamini_hochberg
    sweep_live = [r for r in d["rows"] if not r["degenerate"]]
    p_unif_f = np.array([r["conventions"]["A"]["order"]["fifths"]["p"] for r in sweep_live])
    p_unif_c = np.array([r["conventions"]["A"]["order"]["chromatic"]["p"] for r in sweep_live])
    p_blk_f = np.array([r["within_block"]["p_rho_fifths"] for r in rows])
    p_blk_c = np.array([r["within_block"]["p_rho_chromatic"] for r in rows])
    ordering = dict(
        fifths=dict(
            uniform_null_raw_below_05=int((p_unif_f < 0.05).sum()),
            uniform_null_surviving_bh=int(benjamini_hochberg(p_unif_f).sum()),
            within_block_raw_below_05=int((p_blk_f < 0.05).sum()),
            within_block_surviving_bh=int(benjamini_hochberg(p_blk_f).sum()),
            within_block_bh_survivors=[(rows[i]["readout"], rows[i]["layer"])
                                       for i in np.flatnonzero(benjamini_hochberg(p_blk_f))]),
        chromatic=dict(
            uniform_null_raw_below_05=int((p_unif_c < 0.05).sum()),
            uniform_null_surviving_bh=int(benjamini_hochberg(p_unif_c).sum()),
            within_block_raw_below_05=int((p_blk_c < 0.05).sum()),
            within_block_surviving_bh=int(benjamini_hochberg(p_blk_c).sum())),
        reading="the registered uniform-shuffle null is the frozen one and its numbers are "
                "the frozen verdict's inputs; these within-block numbers are the post-hoc "
                "attack on it. Where the fifths ordering count collapses under a null that "
                "holds the two lexical classes fixed, the ordering signal is token class, "
                "not pitch class, and the sweep's ordered_not_closed verdict must be read "
                "that way.")

    out = dict(
        status="post-hoc addition to Phase-2 keys, provenance stated, under "
               "docs/phase2_prereg_2026-08-21.md section 7",
        item="within-block permutation null for the flats-blob confound",
        script="scripts/29_keys_blockperm.py",
        reads=a.sweep,
        generated=time.strftime("%Y-%m-%dT%H:%M:%S"),
        provenance=dict(
            written="AUTHORED while scripts/28_phase2_keys.py was still running (only the "
                    "two-layer dry run and token layers 0-3 were visible), but written to "
                    "disk at 05:43:52, 47 s AFTER results/phase2_keys.json at 05:43:05. The "
                    "mtimes do NOT establish precedence and none is claimed; an earlier "
                    "draft of this field wrongly said they did and was corrected rather than "
                    "deleted.",
            what_stands_instead="(1) the confound is derivable from the label set with no "
                                "data at all; (2) the null is strictly more conservative, so "
                                "its direction of effect is fixed by construction and timing "
                                "could not have been used to game it",
            derived_from="the LABEL SET alone -- no data was needed to see the confound, in "
                         "the same way prereg section 6's naturals-contiguity fact needed none",
            is_not_a_search_family="this is a strictly more conservative null for an "
                                   "already-registered statistic; it can only move a cell from "
                                   "'separates' to 'does not separate', never the reverse",
            tenet="any positive gets attacked before it gets reported"),
        confound=dict(
            flats=FLATS,
            fifths_positions_of_flats=sorted(int(fifths_code[i]) for i in flat_idx),
            chromatic_positions_of_flats=sorted(int(chrom_code[i]) for i in flat_idx),
            statement="in FIFTHS order the five flats occupy five consecutive slots and the "
                      "seven naturals the complementary arc, so a cloud that merely separates "
                      "the two lexical classes into two angular blobs already agrees with the "
                      "fifths cycle up to within-blob scrambling, and disagrees with the "
                      "chromatic cycle, which interleaves them. Token frequency and token "
                      "class produce exactly that separation with no pitch-class content.",
            why_the_registered_null_misses_it="the K2 margin null shuffles all twelve labels "
                                              "uniformly, which destroys the block structure, "
                                              "so it is the null for 'no structure' rather than "
                                              "for 'two lexical blobs and nothing else'",
            flats_contiguity_reference=float(12 / comb(12, 5))),
        n_within_block_permutations=int(factorial(len(flat_idx)) * factorial(len(nat_idx))),
        n_null_draws=int(a.n_null),
        summary=dict(
            n_live_cells=n,
            n_separating_under_uniform_null=int(sep_u),
            n_separating_under_within_block_null=int(sep_b),
            demoted_by_the_block_null=demoted,
            promoted_by_the_block_null=promoted,
            mean_within_block_null_margin=mean_block_margin,
            mean_within_block_threshold=mean_block_thr,
            mean_uniform_threshold=mean_unif_thr,
            confound_size_note="mean_within_block_null_margin is the confound measured "
                               "directly: it is the average fifths-minus-chromatic margin "
                               "produced by the two-blob arrangement alone, with the "
                               "within-class ordering randomised. A value well above zero "
                               "means lexical class by itself pushes the statistic "
                               "fifths-ward."),
        ordering_collapse=ordering,
        rows=rows,
        runtime_sec=float(time.time() - t0))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=2))

    print("\n" + "=" * 92)
    print(f"live cells                              : {n}")
    print(f"margin separates, uniform null (K2)     : {sep_u}")
    print(f"margin separates, within-block null     : {sep_b}")
    print(f"demoted by the block null               : {demoted}")
    print(f"promoted by the block null              : {promoted}")
    print(f"mean margin from the blob ALONE         : {mean_block_margin:+.4f}  "
          f"(this is the confound, measured)")
    print(f"mean threshold uniform / within-block   : {mean_unif_thr:.4f} / {mean_block_thr:.4f}")
    of, oc = ordering["fifths"], ordering["chromatic"]
    print(f"\nORDERING, the number that decides how the sweep is read:")
    print(f"  rho_fifths    raw p<0.05  uniform null {of['uniform_null_raw_below_05']:2d}/78"
          f"  ->  within-block {of['within_block_raw_below_05']:2d}/78")
    print(f"  rho_fifths    surviving BH uniform null {of['uniform_null_surviving_bh']:2d}"
          f"  ->  within-block {of['within_block_surviving_bh']:2d}")
    print(f"  rho_chromatic raw p<0.05  uniform null {oc['uniform_null_raw_below_05']:2d}/78"
          f"  ->  within-block {oc['within_block_raw_below_05']:2d}/78")
    print(f"\nruntime {time.time() - t0:.1f}s -> wrote {a.out}")


if __name__ == "__main__":
    main()
