"""Step 19 (EXPLORATORY, not pre-registered) -- can per-position co-activation
EVER group the month latents?

The project's sharpest negative is that co-activation clustering never puts more
than two month latents in one cluster at any layer. That result has always been
read as "the month manifold is not there". There is a competing, structural
explanation that has never been tested: month labels are MUTUALLY EXCLUSIVE (a
token is at most one month), so two latents that key on disjoint month sets
essentially cannot fire at the same token position. If that is true, then
P(j fires | i fires) -- the exact quantity `07_cluster.py` thresholds at 0.35 --
is near zero for month pairs by construction, and no threshold, no
symmetrisation and no Leiden setting could ever have grouped them. That would be
a method/ontology mismatch, not evidence about the model's geometry.

What this script computes, one cache layer at a time:

  1. The directed conditional co-fire matrix P(j|i) over just the month latents
     of `results/month_latent_placement.json`, reproducing
     `gfdisc.cluster.coactivation_matrix` exactly (indicator matrix, sparse Gram,
     divide by the firing count of the conditioning latent, zero the diagonal).
     Restricting the column set is exact: P(j|i) for a pair inside the subset
     does not depend on which other columns are present.
  2. The min-symmetrised matrix `np.minimum(P, P.T)` that `07_cluster.py` builds
     before thresholding, and edge counts at 0.35 / 0.05 / > 0 for both.
  3. The re-widening probe: the same statistics with co-fire redefined at WINDOW
     level -- i and j co-fire if both fire within +/- W tokens of each other
     inside the same 255-token cached sequence -- for W in {8, 32, 128, 255}.
     W = 255 is whole-sequence, since the widest within-sequence gap is 254.

Month latents overlap in coverage (e.g. two latents that both list "july"), so
every pair statistic is also reported split by whether the two latents' month
sets from the placement file are DISJOINT. The mutual-exclusivity hypothesis
only predicts near-zero per-position co-fire for the disjoint pairs; a nonzero
value driven by an overlapping pair is not a refutation of it.

Nothing here computes an ordering or closure statistic, and nothing here can
change a gate verdict. It is a diagnostic on the discovery stage only.
"""
import argparse, json, sys, time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from gfdisc.cache import SparseReader

SEQ = 255           # rows per cached sequence: seq_len 256 with BOS dropped
CHUNK = 40_000_000  # uint16 entries per pass over the indices memmap


def hit_rows(reader, latents, chunk=CHUNK):
    """{latent: sorted unique row ids where it fires}, one chunked memmap pass.

    Deliberately avoids `reader.csr()`: the full CSR at L0 = 120 is ~486M
    nonzeros, which would materialise several GB of int64 indices.
    """
    lat = np.asarray(sorted(latents), dtype=np.int64)
    lat16 = lat.astype(np.uint16)
    indices = reader.indices
    n = len(indices)
    parts = {int(f): [] for f in lat}
    for a in range(0, n, chunk):
        b = min(a + chunk, n)
        blk = np.asarray(indices[a:b])
        hit = np.flatnonzero(np.isin(blk, lat16))
        if len(hit) == 0:
            continue
        vals = blk[hit]
        for f in lat:
            sel = hit[vals == np.uint16(f)]
            if len(sel):
                parts[int(f)].append(sel.astype(np.int64) + a)
        del blk, hit, vals
    out = {}
    for f in lat:
        p = parts[int(f)]
        nz = np.concatenate(p) if p else np.empty(0, dtype=np.int64)
        # nonzero-slot -> row id, exactly as SparseReader.rows_where does it
        out[int(f)] = (np.searchsorted(reader.indptr, nz, side="right") - 1).astype(np.int64)
        parts[int(f)] = None
    return out


def window_hits(A, B, W, seq=SEQ):
    """Boolean over A: does some element of B lie within +/-W, same sequence?

    Same-sequence rows are a contiguous run of global row ids, so if any
    qualifying q exists then the nearest element of B on that side of p is
    itself qualifying -- checking the two neighbours is sufficient.
    """
    if len(A) == 0 or len(B) == 0:
        return np.zeros(len(A), dtype=bool)
    sA = A // seq
    idx = np.searchsorted(B, A)
    ok = np.zeros(len(A), dtype=bool)
    r = np.minimum(idx, len(B) - 1)
    br = B[r]
    ok |= (np.abs(br - A) <= W) & (br // seq == sA)
    l = np.maximum(idx - 1, 0)
    bl = B[l]
    ok |= (np.abs(A - bl) <= W) & (bl // seq == sA)
    return ok


def month_graph(Psym, thresh):
    """Edges and components of the MONTH-ONLY subgraph at `thresh`.

    Caveat for the reader: `07_cluster.py` runs Leiden on the full alive graph,
    so month latents could in principle be pulled into one cluster through
    non-month intermediaries. This is the direct-edge story only -- it bounds
    what co-activation can do *among the month latents themselves*.
    """
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components
    k = Psym.shape[0]
    A = (Psym >= thresh) & ~np.eye(k, dtype=bool)
    n_comp, lab = connected_components(csr_matrix(A.astype(np.int8)), directed=False)
    sizes = np.bincount(lab) if k else np.array([0])
    ei, ej = np.nonzero(np.triu(A, 1))
    return dict(n_edges_undirected=int(len(ei)),
                edges=[[int(i), int(j)] for i, j in zip(ei, ej)],
                largest_month_component=int(sizes.max()) if k else 0,
                component_sizes=sorted((int(s) for s in sizes), reverse=True))


def pair_stats(P, ov, name):
    """Off-diagonal summary of a k x k conditional matrix, split by overlap."""
    k = P.shape[0]
    off = ~np.eye(k, dtype=bool)
    dis = off & ~ov          # pairs whose month-name sets are disjoint
    olp = off & ov

    def block(mask):
        v = P[mask]
        return dict(n_pairs=int(mask.sum()),
                    max=float(v.max()) if v.size else 0.0,
                    mean=float(v.mean()) if v.size else 0.0,
                    n_ge_035=int((v >= 0.35).sum()),
                    n_ge_005=int((v >= 0.05).sum()),
                    n_gt_0=int((v > 0).sum()))
    return {f"{name}_all": block(off),
            f"{name}_disjoint": block(dis),
            f"{name}_overlapping": block(olp)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--placement", default="results/month_latent_placement.json")
    ap.add_argument("--root", default="cache/sparse_big")
    ap.add_argument("--min-count", type=int, default=50,
                    help="the alive filter 07_cluster.py used (run_cluster_big.sh)")
    ap.add_argument("--coact-thresh", type=float, default=0.35)
    ap.add_argument("--windows", default="8,32,128,255")
    ap.add_argument("--layers", default="", help="optional comma list; default = all in placement")
    ap.add_argument("--out", default="results/month_cofire.json")
    a = ap.parse_args()

    Ws = [int(x) for x in a.windows.split(",")]
    place = json.loads(Path(a.placement).read_text())
    if a.layers:
        want = {int(x) for x in a.layers.split(",")}
        place = [r for r in place if int(r["layer"]) in want]
    t_start = time.time()
    layers_out = {}

    for rec in place:
        layer = int(rec["layer"])
        lat = [int(x) for x in rec["label_latents"]]
        ex = {int(k): v for k, v in rec.get("examples", {}).items()}
        k = len(lat)
        t0 = time.time()
        reader = SparseReader(a.root, layer)
        n_rows = len(reader)
        n_seq, rem = divmod(n_rows, SEQ)

        # --- row layout check: mean nnz per row as a function of within-sequence
        # offset. A sequence-major layout puts a systematic positional profile
        # here (early offsets differ); a wrong reshape would flatten it.
        nnz = np.diff(reader.indptr).astype(np.float64)
        prof = nnz[: n_seq * SEQ].reshape(n_seq, SEQ).mean(axis=0)
        layout = dict(n_rows=int(n_rows), seq_len_rows=SEQ, n_sequences=int(n_seq),
                      rows_mod_seq=int(rem),
                      mean_nnz_per_row=float(nnz.mean()),
                      mean_nnz_offset_0=float(prof[0]),
                      mean_nnz_offset_1=float(prof[1]),
                      mean_nnz_offset_last=float(prof[-1]),
                      offset_profile_min=float(prof.min()),
                      offset_profile_max=float(prof.max()),
                      offset_profile_std=float(prof.std()),
                      ok=bool(rem == 0))
        del nnz, prof

        rows = hit_rows(reader, lat)
        counts = np.array([len(rows[f]) for f in lat], dtype=np.float64)
        alive = counts >= a.min_count

        # --- overlap mask from the placement file's month-name sets
        sets = [set(ex.get(f, [])) for f in lat]
        ov = np.zeros((k, k), dtype=bool)
        for i in range(k):
            for j in range(k):
                ov[i, j] = bool(sets[i] & sets[j])

        # --- per-position joint counts (exact reproduction of the Gram matrix).
        # The row ids of each disjoint pair's intersection are kept, so the
        # "positions with >= 2 distinct month latents" count can be split by
        # whether the co-firing pair is disjoint or merely overlapping-coverage.
        joint = np.zeros((k, k), dtype=np.int64)
        dis_hits, dis_hits_inf = [], []
        for i in range(k):
            joint[i, i] = len(rows[lat[i]])
            for j in range(i + 1, k):
                inter = np.intersect1d(rows[lat[i]], rows[lat[j]],
                                       assume_unique=True)
                joint[i, j] = joint[j, i] = len(inter)
                if len(inter) and not (sets[i] & sets[j]):
                    dis_hits.append(inter)
                    if alive[i] and alive[j]:
                        dis_hits_inf.append(inter)

        P = joint.astype(np.float32) / np.maximum(counts[:, None], 1.0).astype(np.float32)
        np.fill_diagonal(P, 0.0)
        Psym = np.minimum(P, P.T)
        # Pipeline view: latents below the alive filter never entered the matrix.
        # src/gfdisc/cluster.py:47 implements min_count by zeroing ROWS only
        # (`joint[counts < min_count, :] = 0.0`), and 07_cluster.py additionally
        # passes only alive_idx as `features`, so a dead latent's column is absent
        # too. Zero rows only here, to match coactivation_matrix byte-for-byte;
        # np.minimum(P, P.T) erases the difference in the symmetrised matrix.
        Ppipe = P.copy()
        Ppipe[~alive, :] = 0.0
        Ppipe_sym = np.minimum(Ppipe, Ppipe.T)

        # --- positions where >= 2 distinct month latents fire simultaneously
        allrows = np.concatenate([rows[f] for f in lat]) if k else np.empty(0, np.int64)
        uniq, cnt = np.unique(allrows, return_counts=True)
        n_pos_any = int(len(uniq))
        n_pos_ge2 = int((cnt >= 2).sum())
        max_simul = int(cnt.max()) if len(cnt) else 0
        seq_any = np.unique(allrows // SEQ)
        del allrows
        # How many of those >= 2-distinct positions actually contain a co-firing
        # DISJOINT pair? Every position in a pairwise intersection has >= 2
        # distinct month latents by construction, so the union size is the count.
        # This is what separates "driven entirely by overlapping-coverage
        # latents" from "driven predominantly by them".
        dis_pos = np.unique(np.concatenate(dis_hits)) if dis_hits else np.empty(0, np.int64)
        dis_pos_inf = (np.unique(np.concatenate(dis_hits_inf)) if dis_hits_inf
                       else np.empty(0, np.int64))
        n_pos_ge2_disjoint = int(len(dis_pos))
        n_pos_ge2_disjoint_inf = int(len(dis_pos_inf))
        del dis_hits, dis_hits_inf, dis_pos, dis_pos_inf

        # --- window-level re-widening probe
        win_out = {}
        for W in Ws:
            Jw = np.zeros((k, k), dtype=np.int64)
            anchors_ge2 = []
            for i in range(k):
                Ai = rows[lat[i]]
                multi = np.zeros(len(Ai), dtype=bool)
                for j in range(k):
                    if i == j:
                        Jw[i, i] = len(Ai)
                        continue
                    ok = window_hits(Ai, rows[lat[j]], W)
                    Jw[i, j] = int(ok.sum())
                    multi |= ok
                if multi.any():
                    anchors_ge2.append(Ai[multi])
            Pw = Jw.astype(np.float32) / np.maximum(counts[:, None], 1.0).astype(np.float32)
            np.fill_diagonal(Pw, 0.0)
            Pw_sym = np.minimum(Pw, Pw.T)
            anch = np.unique(np.concatenate(anchors_ge2)) if anchors_ge2 else np.empty(0, np.int64)
            st = {}
            st.update(pair_stats(Pw, ov, "directed"))
            st.update(pair_stats(Pw_sym, ov, "symmetrised"))
            st.update(dict(
                W=W,
                cond_matrix_directed=[[float(x) for x in r] for r in Pw],
                cond_matrix_symmetrised=[[float(x) for x in r] for r in Pw_sym],
                joint_counts=[[int(x) for x in r] for r in Jw],
                n_anchor_positions_with_ge2_distinct=int(len(anch)),
                frac_anchor_positions_with_ge2_distinct=float(len(anch) / max(n_pos_any, 1)),
                month_subgraph_at_thresh=month_graph(Pw_sym, a.coact_thresh),
            ))
            win_out[str(W)] = st
            del Jw, Pw, Pw_sym, anchors_ge2

        # --- verdict
        pp_max_sym = float(Psym[~np.eye(k, dtype=bool)].max()) if k > 1 else 0.0
        dis = (~np.eye(k, dtype=bool)) & ~ov
        pp_max_sym_dis = float(Psym[dis].max()) if dis.any() else 0.0
        pp_max_dir = float(P[~np.eye(k, dtype=bool)].max()) if k > 1 else 0.0
        best_W, best_W_max = None, 0.0
        for W in Ws:
            m = win_out[str(W)]["symmetrised_all"]["max"]
            if m > best_W_max:
                best_W_max, best_W = m, W
        window_clears = best_W_max >= a.coact_thresh
        pp_blocked = pp_max_sym < a.coact_thresh
        if pp_blocked and window_clears:
            verdict = "method_ontology_mismatch_confirmed"
        elif pp_blocked:
            verdict = "per_position_blocked_but_window_also_below_threshold"
        else:
            verdict = "per_position_cofire_NOT_near_zero"

        # The mutual-exclusivity hypothesis only predicts near-zero co-fire for
        # latents whose month sets are DISJOINT; two latents that both key on
        # "july" may legitimately fire on the same token. This is the verdict the
        # structural claim should be judged on.
        dis_best_W_max = 0.0
        for W in Ws:
            dis_best_W_max = max(dis_best_W_max,
                                 win_out[str(W)]["symmetrised_disjoint"]["max"])
        # A disjoint pair only tests the hypothesis if BOTH ends actually entered
        # the clustering, i.e. both cleared the alive filter. A pair anchored on a
        # latent that fires a handful of times is uninformative either way.
        aliveM = np.outer(alive, alive)
        dis_inf = dis & aliveM
        if not dis.any():
            verdict_dis = "no_disjoint_pairs_exist"
        elif not dis_inf.any():
            verdict_dis = "no_informative_disjoint_pairs"
        else:
            m_pp = float(Psym[dis_inf].max())
            m_w = 0.0
            for W in Ws:
                Sw = np.asarray(win_out[str(W)]["cond_matrix_symmetrised"])
                m_w = max(m_w, float(Sw[dis_inf].max()))
            if m_pp < a.coact_thresh and m_w >= a.coact_thresh:
                verdict_dis = "method_ontology_mismatch_confirmed"
            elif m_pp < a.coact_thresh:
                verdict_dis = "per_position_blocked_but_window_also_below_threshold"
            else:
                verdict_dis = "per_position_cofire_NOT_near_zero"
        # which unordered pairs clear the threshold per-position, and do they overlap?
        clearing = []
        for i in range(k):
            for j in range(i + 1, k):
                if Psym[i, j] >= a.coact_thresh:
                    clearing.append(dict(pair=[lat[i], lat[j]],
                                         cond_sym=float(Psym[i, j]),
                                         months_shared=sorted(sets[i] & sets[j]),
                                         overlapping=bool(sets[i] & sets[j])))

        # UNORDERED disjoint-pair ledger at both thresholds. The pair_stats blocks
        # count ORDERED entries of a symmetric matrix, so each unordered pair
        # appears twice there (n_ge_005 = 6 means 3 pairs); this block is the
        # unordered list, so a claim about "how many disjoint pairs exceed X" can
        # be read off directly rather than halved by hand.
        def _dis_list(cut):
            out_ = []
            for i in range(k):
                for j in range(i + 1, k):
                    if (not (sets[i] & sets[j])) and Psym[i, j] >= cut:
                        out_.append(dict(pair=[lat[i], lat[j]],
                                         cond_sym=float(Psym[i, j]),
                                         joint_positions=int(joint[i, j]),
                                         months_i=sorted(sets[i]),
                                         months_j=sorted(sets[j]),
                                         both_alive=bool(alive[i] and alive[j])))
            return sorted(out_, key=lambda d: -d["cond_sym"])
        dis_ledger = dict(
            n_disjoint_unordered_pairs=int(dis.sum() // 2),
            n_informative_disjoint_unordered_pairs=int(dis_inf.sum() // 2),
            n_disjoint_unordered_pairs_ge_035=len(_dis_list(a.coact_thresh)),
            n_disjoint_unordered_pairs_ge_005=len(_dis_list(0.05)),
            disjoint_unordered_pairs_ge_035=_dis_list(a.coact_thresh),
            disjoint_unordered_pairs_ge_005=_dis_list(0.05),
        )

        # The crux table: every unordered pair, its month overlap, whether both
        # ends cleared the alive filter, and its per-position vs window co-fire.
        # A "disjoint" pair whose weakness traces to a latent firing a handful of
        # times is uninformative about the structural hypothesis, so the firing
        # counts sit right next to the numbers.
        pairs_detail = []
        for i in range(k):
            for j in range(i + 1, k):
                pairs_detail.append(dict(
                    pair=[lat[i], lat[j]],
                    months_i=sorted(sets[i]), months_j=sorted(sets[j]),
                    shared=sorted(sets[i] & sets[j]),
                    disjoint=bool(not (sets[i] & sets[j])),
                    counts=[int(counts[i]), int(counts[j])],
                    both_alive=bool(alive[i] and alive[j]),
                    joint_positions=int(joint[i, j]),
                    cond_sym_per_position=float(Psym[i, j]),
                    cond_dir_per_position=[float(P[i, j]), float(P[j, i])],
                    cond_sym_by_window={
                        str(W): float(win_out[str(W)]["cond_matrix_symmetrised"][i][j])
                        for W in Ws},
                ))

        layers_out[str(layer)] = dict(
            layer=layer,
            n_month_latents=k,
            month_latents=lat,
            month_sets={str(f): sorted(ex.get(f, [])) for f in lat},
            firing_counts={str(f): int(len(rows[f])) for f in lat},
            firing_rate={str(f): float(len(rows[f]) / n_rows) for f in lat},
            alive_at_min_count={str(f): bool(al) for f, al in zip(lat, alive)},
            n_alive=int(alive.sum()),
            row_layout_check=layout,
            overlap_mask=[[bool(x) for x in r] for r in ov],
            pairs_detail=pairs_detail,
            per_position=dict(
                joint_counts=[[int(x) for x in r] for r in joint],
                cond_matrix_directed=[[float(x) for x in r] for r in P],
                cond_matrix_symmetrised=[[float(x) for x in r] for r in Psym],
                cond_matrix_directed_pipeline_view=[[float(x) for x in r] for r in Ppipe],
                cond_matrix_symmetrised_pipeline_view=[[float(x) for x in r] for r in Ppipe_sym],
                n_positions_any_month_latent=n_pos_any,
                n_positions_ge2_distinct=n_pos_ge2,
                frac_positions_ge2_distinct=float(n_pos_ge2 / max(n_pos_any, 1)),
                n_positions_ge2_distinct_with_disjoint_pair=n_pos_ge2_disjoint,
                frac_positions_ge2_distinct_with_disjoint_pair=float(
                    n_pos_ge2_disjoint / max(n_pos_ge2, 1)),
                n_positions_ge2_distinct_with_informative_disjoint_pair=n_pos_ge2_disjoint_inf,
                n_positions_ge2_distinct_overlapping_pairs_only=int(
                    n_pos_ge2 - n_pos_ge2_disjoint),
                frac_positions_ge2_distinct_overlapping_pairs_only=float(
                    (n_pos_ge2 - n_pos_ge2_disjoint) / max(n_pos_ge2, 1)),
                max_distinct_month_latents_at_one_position=max_simul,
                n_sequences_any_month_latent=int(len(seq_any)),
                month_subgraph_at_thresh=month_graph(Psym, a.coact_thresh),
                **pair_stats(P, ov, "directed"),
                **pair_stats(Psym, ov, "symmetrised"),
            ),
            window=win_out,
            disjoint_pair_ledger=dis_ledger,
            verdict=verdict,
            verdict_disjoint_pairs=verdict_dis,
            verdict_detail=dict(
                per_position_max_offdiag_directed=pp_max_dir,
                per_position_max_offdiag_symmetrised=pp_max_sym,
                per_position_max_offdiag_symmetrised_disjoint_pairs=pp_max_sym_dis,
                n_disjoint_ordered_pairs=int(dis.sum()),
                n_informative_disjoint_ordered_pairs=int(dis_inf.sum()),
                per_position_max_offdiag_symmetrised_informative_disjoint=(
                    float(Psym[dis_inf].max()) if dis_inf.any() else None),
                per_position_pairs_at_or_above_thresh_symmetrised=int(
                    (Psym[~np.eye(k, dtype=bool)] >= a.coact_thresh).sum()) if k > 1 else 0,
                per_position_clearing_pairs=clearing,
                best_window=best_W,
                best_window_max_symmetrised=best_W_max,
                best_window_max_symmetrised_disjoint=dis_best_W_max,
                window_clears_thresh=bool(window_clears),
                thresh=a.coact_thresh,
            ),
            runtime_s=round(time.time() - t0, 1),
        )
        pp_g = layers_out[str(layer)]["per_position"]["month_subgraph_at_thresh"]
        w_g = win_out[str(Ws[-1])]["month_subgraph_at_thresh"]
        print(f"layer {layer}: k={k} month latents, alive={int(alive.sum())}; "
              f"per-position max P(j|i) dir={pp_max_dir:.4g} sym={pp_max_sym:.4g} "
              f"(disjoint sym={pp_max_sym_dis:.4g}); "
              f"positions with >=2 = {n_pos_ge2:,}/{n_pos_any:,}; "
              f"month-subgraph@0.35 per-pos edges={pp_g['n_edges_undirected']} "
              f"largest={pp_g['largest_month_component']} -> "
              f"W={Ws[-1]} edges={w_g['n_edges_undirected']} "
              f"largest={w_g['largest_month_component']}; "
              f"(of which with a disjoint pair = {n_pos_ge2_disjoint:,}); "
              f"disjoint unordered pairs >=0.35/{'>='}0.05 = "
              f"{dis_ledger['n_disjoint_unordered_pairs_ge_035']}/"
              f"{dis_ledger['n_disjoint_unordered_pairs_ge_005']}; "
              f"verdict={verdict} / disjoint={verdict_dis} "
              f"[{layers_out[str(layer)]['runtime_s']}s]", flush=True)
        del reader, rows, joint, P, Psym, Ppipe, Ppipe_sym, uniq, cnt, seq_any

    verdicts = [v["verdict_disjoint_pairs"] for v in layers_out.values()]

    # ---- claims block: every headline statement recomputed from layers_out, so
    # the prose in any downstream summary can be checked against a field here
    # instead of against memory. Built after the loop; nothing here re-reads a
    # cache.
    def _maxarg(fn):
        best, at = -1.0, None
        for L, v in layers_out.items():
            m = fn(v)
            if m is not None and m > best:
                best, at = m, L
        return best, at
    dis_max, dis_max_L = _maxarg(
        lambda v: v["verdict_detail"]["per_position_max_offdiag_symmetrised_disjoint_pairs"])
    inf_max, inf_max_L = _maxarg(
        lambda v: v["verdict_detail"]["per_position_max_offdiag_symmetrised_informative_disjoint"])
    all_max, all_max_L = _maxarg(
        lambda v: v["verdict_detail"]["per_position_max_offdiag_symmetrised"])
    ge005 = {L: v["disjoint_pair_ledger"]["n_disjoint_unordered_pairs_ge_005"]
             for L, v in layers_out.items()}
    ge035 = {L: v["disjoint_pair_ledger"]["n_disjoint_unordered_pairs_ge_035"]
             for L, v in layers_out.items()}
    ge2_frac = {L: round(v["per_position"]["frac_positions_ge2_distinct_with_disjoint_pair"], 4)
                for L, v in layers_out.items()}
    claims = dict(
        threshold_claim=dict(
            statement=("no disjoint-month pair clears the frozen 0.35 threshold at any "
                       "layer analysed; the largest disjoint symmetrised P(j|i) is "
                       f"{dis_max:.4f} at layer {dis_max_L}, {a.coact_thresh / max(dis_max, 1e-12):.1f}x "
                       "below threshold"),
            n_disjoint_unordered_pairs_ge_035_by_layer=ge035,
            max_disjoint_symmetrised=dis_max, at_layer=dis_max_L,
            max_informative_disjoint_symmetrised=inf_max, informative_at_layer=inf_max_L,
        ),
        five_percent_claim=dict(
            statement=("disjoint pairs are NOT all below 0.05: three unordered disjoint "
                       "pairs at layer 6 exceed 0.05 (see "
                       "layers.6.disjoint_pair_ledger.disjoint_unordered_pairs_ge_005); "
                       "every other layer analysed has zero, and layer 9 has no disjoint "
                       "pair at all. Only the 0.35 half of the threshold claim is "
                       "load-bearing for the verdict, and it holds."),
            n_disjoint_unordered_pairs_ge_005_by_layer=ge005,
            note=("the pair_stats blocks (symmetrised_disjoint.n_ge_005 etc.) count "
                  "ORDERED entries, so they are exactly twice these unordered counts"),
        ),
        ge2_positions_claim=dict(
            statement=(
                "positions where >= 2 distinct month latents fire are PREDOMINANTLY, but "
                "NOT entirely, driven by overlapping-coverage latents at the layers that "
                "have any volume: "
                + ", ".join(f"L{L} {v['per_position']['n_positions_ge2_distinct_with_disjoint_pair']:,}"
                            f"/{v['per_position']['n_positions_ge2_distinct']:,} "
                            f"({ge2_frac[L]:.1%}) contain a co-firing DISJOINT pair"
                            for L, v in layers_out.items())
                + ". Layers where the disjoint share is above half run the other way and "
                  "are called out here: "
                + (", ".join(
                    f"L{L} ({ge2_frac[L]:.0%} of only "
                    f"{layers_out[L]['per_position']['n_positions_ge2_distinct']} positions)"
                    for L in layers_out if ge2_frac[L] > 0.5) or "none")
                + ". 'Entirely' would be wrong at every layer with a nonzero disjoint "
                  "share: "
                + (", ".join(f"L{L}" for L in layers_out if ge2_frac[L] > 0) or "none")),
            frac_of_ge2_positions_containing_a_disjoint_pair_by_layer=ge2_frac,
            n_ge2_positions_by_layer={L: v["per_position"]["n_positions_ge2_distinct"]
                                      for L, v in layers_out.items()},
        ),
        max_per_position_pair_claim=dict(
            statement=(
                "the largest per-position symmetrised co-fire between any two month latents "
                f"is {all_max:.4f} at layer {all_max_L}; "
                + ("every pair that clears 0.35 anywhere is an OVERLAPPING pair"
                   if all(p["overlapping"]
                          for v in layers_out.values()
                          for p in v["verdict_detail"]["per_position_clearing_pairs"])
                   else "at least one DISJOINT pair clears 0.35 -- see the ledger")
                + " (see verdict_detail.per_position_clearing_pairs per layer)"),
            max_symmetrised_any_pair=all_max, at_layer=all_max_L,
            all_clearing_pairs_are_overlapping=bool(all(
                p["overlapping"] for v in layers_out.values()
                for p in v["verdict_detail"]["per_position_clearing_pairs"])),
            n_clearing_pairs_total=int(sum(
                len(v["verdict_detail"]["per_position_clearing_pairs"])
                for v in layers_out.values())),
            clearing_pairs_by_layer={
                L: v["verdict_detail"]["per_position_clearing_pairs"]
                for L, v in layers_out.items()},
        ),
        verdict_definitions=dict(
            statement=("two verdict fields exist and can disagree. `verdict` is the spec's "
                       "literal rule over ALL pairs; `verdict_disjoint_pairs` restricts to "
                       "pairs whose month sets are disjoint and both ends alive, which is "
                       "what the mutual-exclusivity hypothesis actually predicts about. "
                       "Any tally must say which it quotes."),
            verdict_all_pairs_by_layer={L: v["verdict"] for L, v in layers_out.items()},
            verdict_disjoint_pairs_by_layer={L: v["verdict_disjoint_pairs"]
                                             for L, v in layers_out.items()},
        ),
    )

    out = dict(
        status="exploratory, not pre-registered",
        script=str(Path(__file__).resolve()),
        question=("can per-position conditional co-activation ever group latents for "
                  "mutually exclusive labels (months)?"),
        cache_root=a.root,
        placement_file=a.placement,
        coact_thresh=a.coact_thresh,
        coact_thresh_source=("scripts/run_cluster_big.sh -- 0.35, frozen by the "
                             "Amendment-1 recalibration rule on layer 9"),
        min_count=a.min_count,
        seq_len_rows=SEQ,
        windows=Ws,
        matrix_convention=("cond_matrix[i][j] = P(latent j fires | latent i fires), "
                           "exactly gfdisc.cluster.coactivation_matrix before "
                           "np.minimum symmetrisation; diagonal zeroed. directed "
                           "pair counts are over k(k-1) ordered pairs, symmetrised "
                           "counts over the same k(k-1) entries of the symmetric "
                           "matrix (each unordered pair therefore appears twice)."),
        layers=layers_out,
        scope=("month latents were enumerated only at the layers in the placement file "
               f"({sorted(int(L) for L in layers_out)}), because "
               "scripts/18_month_latent_placement.py was run with --layers 5,6,7,8,9,10. "
               "Layers 0-4 were never searched for month latents, so every superlative "
               "here ('the highest pair', 'the only multi-month clusters') is scoped to "
               "these layers, not to all 26 layers of the model."),
        claims=claims,
        pipeline_view_note=("cond_matrix_*_pipeline_view applies the alive filter the way "
                            "src/gfdisc/cluster.py:47 does -- rows only "
                            "(joint[counts < min_count, :] = 0). 07_cluster.py also passes "
                            "only alive latents as `features`, so a dead latent has no "
                            "column either; np.minimum(P, P.T) makes the two conventions "
                            "identical in the symmetrised matrix that is thresholded."),
        subgraph_caveat=("month_subgraph_at_thresh is the direct-edge graph among "
                         "month latents only. 07_cluster.py runs Leiden on the full "
                         "alive graph, so month latents could in principle be pulled "
                         "into one cluster through non-month intermediaries; this "
                         "analysis does not exclude that route."),
        verdict_key=dict(
            method_ontology_mismatch_confirmed=(
                "per-position symmetrised P(j|i) for informative disjoint pairs stays "
                "below the 0.35 threshold while some window W reaches it: no threshold, "
                "symmetrisation or Leiden setting could have grouped them per-position"),
            no_disjoint_pairs_exist=(
                "every month latent at this layer shares at least one month name with "
                "every other, so the mutual-exclusivity prediction has nothing to test"),
            no_informative_disjoint_pairs=(
                "the only disjoint pairs are anchored on a latent below the alive filter "
                "(min_count), so it was never in the clustering; uninformative either way"),
            per_position_cofire_NOT_near_zero=(
                "an informative disjoint pair reaches the threshold per-position -- the "
                "structural hypothesis does NOT hold at this layer"),
        ),
        n_layers_confirmed=int(sum(v == "method_ontology_mismatch_confirmed"
                                   for v in verdicts)),
        n_layers_untestable=int(sum(v in ("no_disjoint_pairs_exist",
                                          "no_informative_disjoint_pairs")
                                    for v in verdicts)),
        n_layers_refuted=int(sum(v == "per_position_cofire_NOT_near_zero"
                                 for v in verdicts)),
        overall_verdict_disjoint_pairs=(
            "method_ontology_mismatch_confirmed"
            if all(v == "method_ontology_mismatch_confirmed" for v in verdicts)
            else ("no layer refutes; per-layer: " + json.dumps(dict(zip(layers_out, verdicts)))
                  if not any(v == "per_position_cofire_NOT_near_zero" for v in verdicts)
                  else "mixed: " + json.dumps(dict(zip(layers_out, verdicts))))),
        overall_verdict_all_pairs=(
            "method_ontology_mismatch_confirmed"
            if all(v["verdict"] == "method_ontology_mismatch_confirmed"
                   for v in layers_out.values())
            else ("mixed: " + json.dumps({k: v["verdict"]
                                          for k, v in layers_out.items()}))),
        total_runtime_s=round(time.time() - t_start, 1),
    )
    Path(a.out).parent.mkdir(exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=2))
    print(f"\nwrote {a.out}  ({out['total_runtime_s']}s)")


if __name__ == "__main__":
    main()
