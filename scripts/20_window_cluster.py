"""Step 20 (DECLARED EXPLORATORY, not pre-registered) -- label-blind WINDOW-level
co-activation clustering.

Why this exists
---------------
`results/month_cofire.json` showed that per-position conditional co-activation --
the exact quantity `scripts/07_cluster.py` thresholds -- can categorically never
group month latents: month labels are mutually exclusive, so two latents keying
on disjoint month sets essentially cannot fire at the same token position, and no
threshold, symmetrisation or Leiden setting could have grouped them. At WINDOW
level they do connect (disjoint pairs clear 0.35 from W = 32; components reach
7-of-7 at layer 9 with whole-sequence windows). `results/hand_cluster_gate.json`
then showed the full 16,384-latent readout at layer 9 CLOSES while the
hand-picked month-latent cluster fails, i.e. the ring is distributed.

The open question is therefore: does *label-blind* window-level co-activation
clustering recover a cluster that passes the two-tier gate? This script does the
discovery half -- it never touches a label, a token id, or a placement file until
every partition is frozen.

What it computes, one window size at a time
-------------------------------------------
A latent OCCUPIES a window if it fires at ANY position inside it. Windows tile
each cached sequence in W-sized blocks (255 rows per sequence; the last block of
a sequence is short whenever W does not divide 255; W = 255 is whole-sequence).
Occupancy is restricted to ALIVE latents -- per-POSITION firing count >= 50,
the same filter `07_cluster.py` used via `run_cluster_big.sh`.

Two co-occurrence count matrices are accumulated in ONE pass over the cache:

  arm "even_build" -- windows of EVEN-parity sequences only. The graph is built
                      only from these, and `scripts/21_window_gate.py` scores it
                      only on rows of ODD-parity sequences: the clean held-out
                      selection/scoring split the original pre-registration
                      promised. This is the a-priori PRIMARY arm.
  arm "full"       -- windows of all sequences (robustness).

`full` is obtained as `even + odd`, so both arms cost one pass.

From the counts, P(j|i) = C[i,j] / C[i,i] and the pipeline's `np.minimum(P, P.T)`
symmetrisation, byte-for-byte as `gfdisc.cluster.coactivation_matrix` followed by
`07_cluster.py`'s in-place `np.minimum`. Then the threshold sweep, the frozen
label-blind selection rule replicated from PREREGISTRATION.md Amendment 1, and
Leiden through the imported `gfdisc.cluster.graph_clusters` (same partition type,
same seed 0, same min_size) -- the pipeline's own function, not a copy.

Label-blindness, structurally
-----------------------------
Nothing outside `indices.u16` and `indptr.npy` is read before selection. The
`tokens.i32` memmap that `SparseReader` opens is dropped in `open_reader()`, so a
token id is not reachable from this process at all, and the month-placement file
is opened only inside `month_placement_diagnostic()`, which asserts that every
partition has already been frozen. The diagnostic is recorded as post-selection.

Memory
------
Two n_alive x n_alive float32 accumulators (~1.04 GB each at 16,091 alive
latents) plus one occupancy block per arm. Counts are accumulated with BLAS
`ssyrk` directly into the accumulator (no temporary Gram), which also halves the
FLOPs by filling one triangle. float32 holds every partial sum exactly: they are
integers bounded by the window count, and `assert n_windows < 2**24` is enforced,
so the stored counts are exactly the uint32 counts the design calls for (the
dtype is float32 only because BLAS requires it; integrality is asserted at run
time on the verified sub-block).
"""
import argparse, gc, json, sys, time
from pathlib import Path

import numpy as np
from scipy.linalg.blas import ssyrk

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from gfdisc.cache import SparseReader
from gfdisc.cluster import graph_clusters          # the pipeline's own Leiden

SEQ = 255           # rows per cached sequence: seq_len 256 with BOS dropped

# --------------------------------------------------------------------------
# The run's design declarations, verbatim, copied into every output JSON.
# --------------------------------------------------------------------------
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

FROZEN_RULE = (
    "PREREGISTRATION.md, Amendment 1, 'Threshold recalibration rule -- frozen now, "
    "label-blind': select the SMALLEST threshold whose largest cluster is <= 200 "
    "latents AND which yields >= 20 clusters of size >= 6. Selection uses "
    "cluster-size statistics only -- no labels, no geometry, no ordering or closure "
    "statistic enters the choice. If no threshold in the sweep satisfies both "
    "conditions, that is reported as a failure of the rule; the sweep is not "
    "extended post hoc.")

# Set to True only after every partition in the run is frozen. The month-placement
# diagnostic asserts it, so a future edit that moves the diagnostic earlier fails
# loudly instead of quietly contaminating the selection.
_SELECTION_FROZEN = [False]


# --------------------------------------------------------------------------
# cache access
# --------------------------------------------------------------------------
def open_reader(root, layer):
    """SparseReader with the token memmap dropped.

    `SparseReader.__init__` memmaps `tokens.i32`. This script must be
    structurally incapable of reading a token id before selection, so the handle
    is discarded here; `positions_of_tokens` would now raise.
    """
    r = SparseReader(root, layer)
    r.tokens = None
    return r


def chunked_firing_counts(reader, d_sae, chunk=40_000_000):
    """Per-POSITION firing count per latent, the alive filter's input.

    `SparseReader.firing_counts()` bincounts the whole 376M-entry index memmap in
    one call, which materialises a ~3 GB intp temporary. Chunked here instead;
    the result is identical.
    """
    counts = np.zeros(d_sae, dtype=np.int64)
    idx = reader.indices
    n = len(idx)
    for a in range(0, n, chunk):
        blk = np.asarray(idx[a:min(a + chunk, n)])
        counts += np.bincount(blk, minlength=d_sae)[:d_sae]
        del blk
    return counts


def layout_check(reader, n_rows):
    """The row -> (sequence, offset) mapping, verified rather than assumed.

    `scripts/06_cache_sparse.py` writes `store[l][:, 1:, :].reshape(-1, D)` per
    batch, i.e. sequence-major with BOS dropped: row = sequence * 255 + offset.
    A sequence-major layout puts a systematic positional profile in the mean nnz
    per within-sequence offset (early offsets differ); a wrong reshape flattens
    it. Same check as `scripts/19_month_cofire.py`.
    """
    n_seq, rem = divmod(n_rows, SEQ)
    nnz = np.diff(reader.indptr).astype(np.float64)
    prof = nnz[: n_seq * SEQ].reshape(n_seq, SEQ).mean(axis=0)
    out = dict(n_rows=int(n_rows), seq_len_rows=SEQ, n_sequences=int(n_seq),
               rows_mod_seq=int(rem), mean_nnz_per_row=float(nnz.mean()),
               mean_nnz_offset_0=float(prof[0]), mean_nnz_offset_1=float(prof[1]),
               mean_nnz_offset_last=float(prof[-1]),
               offset_profile_min=float(prof.min()),
               offset_profile_max=float(prof.max()),
               offset_profile_std=float(prof.std()),
               row_to_seq="row // 255", row_to_offset="row % 255",
               ok=bool(rem == 0))
    del nnz, prof
    return out


# --------------------------------------------------------------------------
# the streaming co-occurrence pass
# --------------------------------------------------------------------------
def n_windows_per_seq(W):
    return int(np.ceil(SEQ / W))


def brute_window_sets(reader, seq, W, lut):
    """Alive-latent occupancy of each window of one sequence, by per-row reads."""
    nwps = n_windows_per_seq(W)
    out = [set() for _ in range(nwps)]
    for off in range(SEQ):
        idx, _ = reader.row(seq * SEQ + off)
        w = off // W
        for f in idx:
            if lut[int(f)] >= 0:
                out[w].add(int(f))
    return out


def cofire_pass(reader, alive_idx, W, n_seq, d_sae, block_seqs, verify_seqs, log):
    """One pass over the cache -> (C_even, C_odd) window co-occurrence counts.

    C[i, j] = number of windows in which alive latents i and j BOTH occupy.
    Only the upper triangle is filled (BLAS syrk); `symmetrise_upper` completes it.
    """
    n_alive = len(alive_idx)
    nwps = n_windows_per_seq(W)
    lut = np.full(d_sae, -1, dtype=np.int32)
    lut[alive_idx] = np.arange(n_alive, dtype=np.int32)

    n_win_even = (n_seq // 2 + n_seq % 2) * nwps
    n_win_odd = (n_seq // 2) * nwps
    for nw in (n_win_even, n_win_odd, n_win_even + n_win_odd):
        assert nw < 2 ** 24, (
            f"{nw} windows exceeds the float32 exact-integer bound 2**24; "
            "counts would no longer be exact")

    C_even = np.zeros((n_alive, n_alive), dtype=np.float32, order="F")
    C_odd = np.zeros((n_alive, n_alive), dtype=np.float32, order="F")
    rows_per_arm = (block_seqs // 2) * nwps
    A_even = np.zeros((rows_per_arm, n_alive), dtype=np.float32)
    A_odd = np.zeros((rows_per_arm, n_alive), dtype=np.float32)

    indptr = reader.indptr
    checks, first_syrk_checked = [], False
    t0 = time.time()
    for s0 in range(0, n_seq, block_seqs):
        s1 = min(s0 + block_seqs, n_seq)
        r0, r1 = s0 * SEQ, s1 * SEQ
        nz0, nz1 = int(indptr[r0]), int(indptr[r1])
        idx = np.array(reader.indices[nz0:nz1])                 # uint16, in memory
        col = lut[idx]
        del idx
        keep = col >= 0
        rowid = np.repeat(np.arange(r0, r1, dtype=np.int64),
                          np.diff(indptr[r0:r1 + 1]))
        rowid = rowid[keep]
        col = col[keep]
        del keep
        seq = rowid // SEQ
        w = (rowid - seq * SEQ) // W
        del rowid
        loc = seq - s0
        ev = (loc & 1) == 0
        od = ~ev

        n_ev = ((s1 - s0) + 1) // 2
        n_od = (s1 - s0) // 2
        A_even[:] = 0.0
        A_odd[:] = 0.0
        A_even[(loc[ev] >> 1) * nwps + w[ev], col[ev]] = 1.0
        A_odd[((loc[od] - 1) >> 1) * nwps + w[od], col[od]] = 1.0

        if s0 == 0 and verify_seqs:
            for s in verify_seqs:
                if not (s0 <= s < s1):
                    continue
                truth = brute_window_sets(reader, s, W, lut)
                buf = A_even if (s % 2 == 0) else A_odd
                base = (((s - s0) >> 1) if s % 2 == 0 else ((s - s0 - 1) >> 1)) * nwps
                ok = True
                for wi in range(nwps):
                    got = {int(alive_idx[c]) for c in np.flatnonzero(buf[base + wi])}
                    ok = ok and (got == truth[wi])
                checks.append(dict(sequence=int(s), W=int(W), parity=int(s % 2),
                                   n_windows=nwps,
                                   n_alive_latents_in_seq=int(len(set().union(*truth))),
                                   occupancy_matches_row_reads=bool(ok)))
                assert ok, f"occupancy mismatch at sequence {s}, W={W}"

        out_e = ssyrk(1.0, A_even[:n_ev * nwps], 1.0, C_even,
                      trans=1, lower=0, overwrite_c=1)
        out_o = ssyrk(1.0, A_odd[:n_od * nwps], 1.0, C_odd,
                      trans=1, lower=0, overwrite_c=1)
        if not first_syrk_checked:
            assert out_e is C_even and out_o is C_odd, (
                "ssyrk did not accumulate in place -- counts would be lost")
            first_syrk_checked = True
        del col, seq, w, loc, ev, od
        if (s0 // block_seqs) % 8 == 0:
            done = s1 / n_seq
            print(f"    W={W} pass {done:5.1%}  [{time.time() - t0:6.1f}s]", flush=True)
    del A_even, A_odd, lut
    gc.collect()
    log["pass_seconds"] = round(time.time() - t0, 1)
    log["occupancy_spotcheck"] = checks
    log["n_windows_even"] = int(n_win_even)
    log["n_windows_odd"] = int(n_win_odd)
    log["n_windows_per_sequence"] = nwps
    return C_even, C_odd


def symmetrise_upper(C, chunk=2048):
    """Complete a syrk-filled upper triangle into a full symmetric matrix, in place."""
    n = C.shape[0]
    for lo in range(0, n, chunk):
        hi = min(lo + chunk, n)
        if lo:
            C[lo:hi, :lo] = C[:lo, lo:hi].T
        blk = C[lo:hi, lo:hi]
        blk += np.triu(blk, 1).T          # np.triu copies, so this is alias-safe


def sample_occupancy_sets(reader, feats, W, chunk=40_000_000):
    """Window ids occupied by a handful of latents, by an independent memmap pass.

    Used to verify the GEMM count matrices elementwise: C[i,j] must equal the size
    of the intersection of these sets. Deliberately shares no code with
    `cofire_pass`.
    """
    nwps = n_windows_per_seq(W)
    sel = np.zeros(65536, dtype=bool)
    sel[np.asarray(feats, dtype=np.int64)] = True
    parts = {int(f): [] for f in feats}
    idx, indptr = reader.indices, reader.indptr
    n = len(idx)
    for a in range(0, n, chunk):
        b = min(a + chunk, n)
        blk = np.asarray(idx[a:b])
        hit = np.flatnonzero(sel[blk])
        if not len(hit):
            del blk
            continue
        vals = blk[hit]
        rows = np.searchsorted(indptr, hit.astype(np.int64) + a, side="right") - 1
        seq = rows // SEQ
        wid = seq * nwps + (rows - seq * SEQ) // W
        for f in feats:
            m = vals == np.uint16(f)
            if m.any():
                parts[int(f)].append(np.stack([wid[m], seq[m] & 1]))
        del blk, hit, vals, rows, seq, wid
    out = {}
    for f in feats:
        if parts[int(f)]:
            cat = np.concatenate(parts[int(f)], axis=1)
            allw, par = cat[0], cat[1]
            out[int(f)] = (set(int(x) for x in np.unique(allw)),
                           set(int(x) for x in np.unique(allw[par == 0])),
                           set(int(x) for x in np.unique(allw[par == 1])))
        else:
            out[int(f)] = (set(), set(), set())
    return out


def verify_counts(reader, C_even, C_full, alive_idx, W, n_check, seed):
    """Elementwise check of both count matrices against independent set intersections."""
    rng = np.random.default_rng(seed)
    pick = sorted(int(p) for p in rng.choice(len(alive_idx),
                                             size=min(n_check, len(alive_idx)),
                                             replace=False))
    feats = [int(alive_idx[p]) for p in pick]
    sets = sample_occupancy_sets(reader, feats, W)
    n_bad_even = n_bad_full = 0
    max_count = 0
    for a, fa in enumerate(feats):
        for b, fb in enumerate(feats):
            i, j = pick[a], pick[b]
            ce = len(sets[fa][1] & sets[fb][1])
            cf = len(sets[fa][0] & sets[fb][0])
            # only the upper triangle is filled at this point
            n_bad_even += int(float(C_even[min(i, j), max(i, j)]) != float(ce))
            n_bad_full += int(float(C_full[min(i, j), max(i, j)]) != float(cf))
            max_count = max(max_count, cf)
    sub_e = C_even[np.ix_(pick, pick)]
    sub_f = C_full[np.ix_(pick, pick)]
    return dict(
        W=int(W), n_latents_checked=len(feats), latents=feats,
        checked_alive_positions=[int(p) for p in pick],
        n_mismatches_even_build=int(n_bad_even), n_mismatches_full=int(n_bad_full),
        exact_match=bool(n_bad_even == 0 and n_bad_full == 0),
        counts_are_integers=bool(np.array_equal(sub_e, np.round(sub_e))
                                 and np.array_equal(sub_f, np.round(sub_f))),
        max_count_seen=int(max_count),
        method=("independent second pass over indices.u16; C[i,j] compared against "
                "|window-set(i) & window-set(j)| for every ordered pair of the "
                "sampled latents, in both arms"))


# --------------------------------------------------------------------------
# threshold sweep + the frozen selection rule
# --------------------------------------------------------------------------
def edge_stats(P, T, chunk=1024):
    """Undirected edge count and max degree of the graph `graph_clusters` would build.

    Counted in row chunks so the full boolean mask is never materialised. P has a
    zeroed diagonal and T > 0, so the diagonal never contributes.
    """
    n = P.shape[0]
    tot, maxdeg = 0, 0
    for lo in range(0, n, chunk):
        m = P[lo:min(lo + chunk, n)] >= T
        d = m.sum(axis=1)
        tot += int(d.sum())
        maxdeg = max(maxdeg, int(d.max()) if len(d) else 0)
        del m, d
    return tot // 2, maxdeg


def sweep_cell(Psym, alive_idx, thresholds, min_size, max_size, max_edges, tag):
    """The threshold sweep and the frozen label-blind selection, for one cell."""
    sweep, partitions = [], {}
    for T in thresholds:
        t0 = time.time()
        n_edges, maxdeg = edge_stats(Psym, T)
        rec = dict(T=float(T), n_edges_undirected=int(n_edges), max_degree=int(maxdeg))
        if n_edges > max_edges:
            rec.update(status="skipped_edge_limit",
                       note=(f"{n_edges:,} undirected edges exceeds --max-edges "
                             f"{max_edges:,}; graph_clusters materialises a python "
                             "edge list (~110 bytes/edge) and would not fit. Recorded "
                             "as NOT satisfying the frozen rule. Evidence, not proof, "
                             "that the rule could not have been satisfied here: "
                             f"max degree {maxdeg:,} against a 200-latent cap."),
                       n_clusters_prefilter=None, largest_prefilter=None,
                       n_ge6_prefilter=None, prefilter_sizes=None,
                       cond_largest_le_max_size=False, cond_n_ge6_ge_20=False,
                       satisfies_frozen_rule=False,
                       seconds=round(time.time() - t0, 1))
            sweep.append(rec)
            print(f"    [{tag}] T={T:.2f}  edges={n_edges:,} maxdeg={maxdeg:,} "
                  f"-> SKIPPED (edge limit)", flush=True)
            continue
        raw = [[int(alive_idx[i]) for i in c]
               for c in graph_clusters(Psym, T, min_size)]
        sizes = sorted((len(c) for c in raw), reverse=True)
        largest = sizes[0] if sizes else 0
        n_ge6 = int(sum(s >= 6 for s in sizes))
        cond_a = bool(largest <= max_size)
        cond_b = bool(n_ge6 >= 20)
        rec.update(status="evaluated", n_clusters_prefilter=len(sizes),
                   largest_prefilter=int(largest), n_ge6_prefilter=n_ge6,
                   prefilter_sizes=sizes,
                   cond_largest_le_max_size=cond_a, cond_n_ge6_ge_20=cond_b,
                   satisfies_frozen_rule=bool(cond_a and cond_b),
                   n_clusters_postfilter=int(sum(s <= max_size for s in sizes)),
                   n_dropped_by_max_size=int(sum(s > max_size for s in sizes)),
                   seconds=round(time.time() - t0, 1))
        sweep.append(rec)
        partitions[float(T)] = raw
        print(f"    [{tag}] T={T:.2f}  edges={n_edges:,} -> {len(sizes):4d} clusters, "
              f"largest {largest:5d}, {n_ge6:4d} of size >= 6  "
              f"[rule: {'PASS' if cond_a and cond_b else 'no'}] "
              f"({rec['seconds']}s)", flush=True)

    sel = next((r["T"] for r in sweep if r["status"] == "evaluated"
                and r["satisfies_frozen_rule"]), None)
    skipped = [r["T"] for r in sweep if r["status"] == "skipped_edge_limit"]
    caveat = None
    if sel is not None and any(t < sel for t in skipped):
        caveat = (f"thresholds {sorted(t for t in skipped if t < sel)} were below the "
                  "selected threshold and were NOT evaluated (edge limit). The frozen "
                  "rule selects the SMALLEST satisfying threshold, so the selection is "
                  "the smallest satisfying EVALUATED threshold. Their edge counts and "
                  "max degrees are in the sweep table.")
    elif sel is None and skipped:
        caveat = (f"thresholds {sorted(skipped)} were not evaluated (edge limit); "
                  "rule_failure is reported over the evaluated thresholds only.")

    # What the skipped thresholds would have done, argued from the evaluated ones.
    ev = [r for r in sweep if r["status"] == "evaluated"]
    mono = None
    if ev:
        big = [r["largest_prefilter"] for r in ev]
        non_inc = all(big[i] >= big[i + 1] for i in range(len(big) - 1))
        mono = dict(
            thresholds_evaluated=[r["T"] for r in ev],
            largest_prefilter_by_threshold=big,
            n_ge6_by_threshold=[r["n_ge6_prefilter"] for r in ev],
            largest_is_non_increasing_in_T=bool(non_inc),
            smallest_evaluated_T=ev[0]["T"],
            largest_at_smallest_evaluated_T=big[0],
            skipped_thresholds=sorted(skipped))
        if skipped and non_inc and big[0] > max_size:
            mono["skipped_threshold_evidence"] = (
                "every skipped threshold is SMALLER than the smallest evaluated one and "
                "has strictly more edges (sweep table); across the evaluated thresholds "
                "the largest pre-filter cluster is non-increasing in T, and at the "
                f"smallest evaluated threshold it is already {big[0]} > {max_size}. On "
                "that trend the skipped thresholds would have had larger giant clusters "
                "still, so the edge limit is unlikely to have hidden a threshold that "
                "satisfies the rule. Evidence, not proof: Leiden is not monotone by "
                "construction.")
        elif skipped:
            mono["skipped_threshold_evidence"] = (
                "inconclusive: the evaluated thresholds do not show a clean monotone "
                "trend, or the smallest evaluated threshold already meets the size "
                "condition, so nothing can be argued about the skipped ones. Raise "
                "--max-edges to evaluate them.")

    out = dict(frozen_rule=FROZEN_RULE, thresholds=list(map(float, thresholds)),
               max_edges=int(max_edges), min_size=int(min_size),
               max_size=int(max_size), sweep=sweep,
               selected_threshold=sel,
               rule_status="selected" if sel is not None else "rule_failure",
               selection_caveat=caveat, monotonicity=mono)
    if sel is None:
        out["clusters"] = None
        out["dropped_by_max_size_sizes"] = None
        return out
    raw = partitions[sel]
    kept = [c for c in raw if len(c) <= max_size]
    out["clusters"] = [dict(cluster_id=i, size=len(c), members=[int(x) for x in c])
                       for i, c in enumerate(sorted(kept, key=len, reverse=True))]
    out["dropped_by_max_size_sizes"] = sorted((len(c) for c in raw if len(c) > max_size),
                                              reverse=True)
    out["n_clusters_selected"] = len(kept)
    out["n_clusters_selected_ge6"] = int(sum(len(c) >= 6 for c in kept))
    return out


def min_symmetrise(C, probe=None, chunk=2048):
    """`np.minimum(C, C.T, out=C)` -- 07_cluster.py's step -- done in row blocks.

    The single-call form makes numpy buffer a whole transposed copy (~1 GB at
    16k latents, measured +2 GB peak RSS), which is the difference between one and
    two of these jobs fitting on a 24 GB box. Blocking is exact, not an
    approximation: min is symmetric and idempotent, so a later block that reads an
    already-minimised entry computes min(P[j,i], min(P[i,j], P[j,i])), which is the
    same value. `probe` is a (rows, cols) index pair whose sub-block was copied
    before the call; the caller compares it against a direct np.minimum.
    """
    n = C.shape[0]
    for lo in range(0, n, chunk):
        hi = min(lo + chunk, n)
        blk = C[lo:hi]
        np.minimum(blk, np.asarray(C[:, lo:hi].T), out=blk)


def process_arm(C, alive_idx, arm, W, a, tag, probe_idx=None):
    """C (upper triangle, counts) -> P(j|i) -> min-symmetrised -> sweep+selection.

    Destroys C in place. `C` must be the counts for this arm only.
    """
    t0 = time.time()
    symmetrise_upper(C)
    diag = np.diagonal(C).copy()
    occ = dict(min=float(diag.min()), median=float(np.median(diag)),
               mean=float(diag.mean()), max=float(diag.max()),
               n_latents_in_1_window=int((diag <= 1).sum()),
               note=("window-occupancy count per alive latent = the denominator of "
                     "P(j|i). The design adds NO window-level min-count: the alive "
                     "filter is the per-POSITION count >= 50, exactly as 07_cluster.py."))
    # P(j|i) = C[i,j] / C[i,i], then the pipeline's in-place min-symmetrisation.
    C /= np.maximum(diag[:, None], 1.0).astype(np.float32)
    np.fill_diagonal(C, 0.0)
    ref = None
    if probe_idx is not None:
        sub = C[np.ix_(probe_idx, probe_idx)].copy()
        ref = np.minimum(sub, sub.T)
    min_symmetrise(C)
    check = None
    if ref is not None:
        got = C[np.ix_(probe_idx, probe_idx)]
        check = dict(n_latents=len(probe_idx),
                     max_abs_diff=float(np.abs(got - ref).max()),
                     matches_single_call_np_minimum=bool(np.array_equal(got, ref)),
                     what=("blocked min-symmetrisation compared against "
                           "np.minimum(P, P.T) evaluated directly on the sub-block"))
        assert check["matches_single_call_np_minimum"], "min-symmetrisation mismatch"
    prep = round(time.time() - t0, 1)
    cell = sweep_cell(C, alive_idx, a.thresholds, a.min_size, a.max_size,
                      a.max_edges, tag)
    cell.update(W=int(W), arm=arm, occupancy=occ, matrix_seconds=prep,
                min_symmetrise_check=check)
    return cell


# --------------------------------------------------------------------------
# POST-SELECTION diagnostic -- the first and only place a label file is opened
# --------------------------------------------------------------------------
def month_placement_diagnostic(cells, layer, placement_path):
    assert _SELECTION_FROZEN[0], (
        "month_placement_diagnostic called before every partition was frozen -- "
        "the run's label-blindness guarantee would be void")
    p = Path(placement_path)
    if not p.exists():
        return dict(available=False, placement_file=str(placement_path))
    recs = [r for r in json.loads(p.read_text())
            if int(r["layer"]) == int(layer) and r["ontology"] == "months"]
    if not recs:
        return dict(available=False, placement_file=str(placement_path),
                    note=f"no months record for layer {layer}")
    lat = [int(x) for x in recs[0]["label_latents"]]
    out = dict(available=True, placement_file=str(placement_path),
               timing="POST-SELECTION, diagnostic only; no selection step saw this file",
               month_latents=lat, n_month_latents=len(lat), per_cell={})
    for key, cell in cells.items():
        if cell.get("clusters") is None:
            out["per_cell"][key] = dict(rule_status=cell["rule_status"], clusters=None)
            continue
        hits, seen = [], set()
        for c in cell["clusters"]:
            m = sorted(set(c["members"]) & set(lat))
            if m:
                hits.append(dict(cluster_id=c["cluster_id"], size=c["size"],
                                 month_latents=m, n=len(m)))
                seen.update(m)
        out["per_cell"][key] = dict(
            rule_status=cell["rule_status"],
            selected_threshold=cell["selected_threshold"],
            n_clusters_with_a_month_latent=len(hits),
            max_month_latents_in_one_cluster=max((h["n"] for h in hits), default=0),
            n_month_latents_in_any_cluster=len(seen),
            n_month_latents_unclustered=len(set(lat) - seen),
            clusters=hits)
    return out


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--windows", default="32,128,255")
    ap.add_argument("--root", default="cache/sparse_big")
    ap.add_argument("--min-count", type=int, default=50,
                    help="alive filter: per-POSITION firing count, as 07_cluster.py")
    ap.add_argument("--min-size", type=int, default=3, help="Leiden min cluster size")
    ap.add_argument("--max-size", type=int, default=200)
    ap.add_argument("--thresholds", default="0.25,0.35,0.45,0.55,0.65,0.75,0.85")
    ap.add_argument("--max-edges", type=int, default=5_000_000,
                    help="computational guard; a T above this is recorded, not clustered")
    ap.add_argument("--block-seqs", type=int, default=512,
                    help="sequences per streaming block (must be even)")
    ap.add_argument("--verify-latents", type=int, default=40)
    ap.add_argument("--verify-seqs", default="0,1,2,3")
    ap.add_argument("--placement", default="results/month_latent_placement.json",
                    help="opened ONLY by the post-selection diagnostic")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    a.thresholds = [float(x) for x in a.thresholds.split(",")]
    assert a.thresholds == sorted(a.thresholds), (
        "--thresholds must be ascending: the frozen rule selects the SMALLEST "
        "satisfying threshold, which is read off the sweep in order")
    assert a.block_seqs % 2 == 0, "--block-seqs must be even (parity blocking)"
    out_path = Path(a.out or f"results/window_clusters_L{a.layer}.json")
    Ws = [int(x) for x in a.windows.split(",")]
    vseqs = [int(x) for x in a.verify_seqs.split(",") if x != ""]
    t_start = time.time()

    reader = open_reader(a.root, a.layer)
    d_sae = int(reader.meta["d_sae"])
    n_rows = len(reader)
    layout = layout_check(reader, n_rows)
    assert layout["ok"], f"cache rows {n_rows} not a multiple of {SEQ}"
    n_seq = layout["n_sequences"]

    t0 = time.time()
    counts = chunked_firing_counts(reader, d_sae)
    alive_idx = np.flatnonzero(counts >= a.min_count)
    n_alive = len(alive_idx)
    print(f"layer {a.layer}: {n_rows:,} rows, {n_seq:,} sequences, "
          f"{n_alive:,}/{d_sae:,} latents firing >= {a.min_count}x "
          f"[{time.time() - t0:.1f}s]", flush=True)

    cells, verifications = {}, []
    for W in Ws:
        print(f"\n=== W = {W} ({n_windows_per_seq(W)} window(s) per sequence) ===",
              flush=True)
        log = {}
        C_even, C_odd = cofire_pass(reader, alive_idx, W, n_seq, d_sae,
                                    a.block_seqs, vseqs, log)
        C_odd += C_even                       # C_odd is now the FULL arm's counts
        if a.verify_latents:
            t0 = time.time()
            v = verify_counts(reader, C_even, C_odd, alive_idx, W,
                              a.verify_latents, seed=a.layer * 100 + W)
            v["seconds"] = round(time.time() - t0, 1)
            v.update({k: log[k] for k in ("occupancy_spotcheck",)})
            verifications.append(v)
            print(f"  verification: counts exact={v['exact_match']} "
                  f"(even mismatches {v['n_mismatches_even_build']}, full "
                  f"{v['n_mismatches_full']}), integral={v['counts_are_integers']} "
                  f"[{v['seconds']}s]", flush=True)
            assert v["exact_match"] and v["counts_are_integers"], "count verification failed"

        probe = (verifications[-1]["checked_alive_positions"]
                 if a.verify_latents else None)
        mats = {"even_build": C_even, "full": C_odd}
        del C_even, C_odd            # the dict is now the only reference
        for arm in ("even_build", "full"):
            C = mats.pop(arm)
            tag = f"W{W}/{arm}"
            cell = process_arm(C, alive_idx, arm, W, a, tag, probe_idx=probe)
            cell.update(n_windows=int(log["n_windows_even"] if arm == "even_build"
                                      else log["n_windows_even"] + log["n_windows_odd"]),
                        n_windows_per_sequence=log["n_windows_per_sequence"],
                        n_sequences_used=int(n_seq // 2 + n_seq % 2 if arm == "even_build"
                                             else n_seq),
                        pass_seconds=log["pass_seconds"])
            cells[f"W{W}_{arm}"] = cell
            del C
            gc.collect()
        del mats
        gc.collect()

    # ---- every partition is frozen from here on ----
    _SELECTION_FROZEN[0] = True
    diag = month_placement_diagnostic(cells, a.layer, a.placement)
    for key, cell in cells.items():
        cell["month_placement_diagnostic"] = (
            diag["per_cell"].get(key) if diag.get("available") else
            dict(available=False))

    out = dict(
        design=DESIGN,
        status="exploratory, not pre-registered",
        script=str(Path(__file__).resolve()),
        generated=time.strftime("%Y-%m-%dT%H:%M:%S"),
        layer=int(a.layer),
        config=dict(root=a.root, windows=Ws, min_count=a.min_count,
                    min_size=a.min_size, max_size=a.max_size,
                    thresholds=a.thresholds, max_edges=a.max_edges,
                    block_seqs=a.block_seqs, seq_len_rows=SEQ,
                    d_sae=d_sae, n_rows=n_rows, n_sequences=n_seq,
                    n_alive=int(n_alive)),
        definitions=dict(
            occupancy=("a latent occupies a window if it fires at ANY position inside "
                       "it; windows tile each sequence in W-sized blocks, so the last "
                       "block of a sequence is short when W does not divide 255; "
                       "W = 255 is whole-sequence"),
            arm_even_build=("graph built ONLY from windows of EVEN-parity sequences "
                            "(sequence = row // 255); scripts/21_window_gate.py scores "
                            "it ONLY on rows of ODD-parity sequences"),
            arm_full="graph built from windows of all sequences (robustness)",
            conditional=("P(j|i) = C[i,j] / C[i,i] over windows, then "
                         "np.minimum(P, P.T) in place -- byte-for-byte "
                         "gfdisc.cluster.coactivation_matrix followed by "
                         "07_cluster.py's symmetrisation, with windows in place of "
                         "token positions"),
            alive=("per-POSITION firing count >= --min-count (50), the filter "
                   "07_cluster.py used; NO window-level min-count is applied"),
            leiden=("gfdisc.cluster.graph_clusters imported unchanged: igraph + "
                    "leidenalg ModularityVertexPartition, weights = the symmetrised "
                    "conditional, seed 0, min_size 3"),
            prefilter_sizes=("cluster sizes BEFORE the --max-size drop; the frozen "
                             "rule's 'largest cluster <= 200' is read off these, as "
                             "07_cluster.py prints them before filtering"),
            count_dtype=("counts are accumulated in float32 because BLAS ssyrk "
                         "requires it; every partial sum is an integer below "
                         "2**24 (asserted), so the values are exactly the uint32 "
                         "counts the design specifies, and integrality is verified "
                         "on the checked sub-block")),
        label_blindness=dict(
            guarantee=("this script opens only indices.u16 and indptr.npy before "
                       "selection; SparseReader's tokens.i32 memmap is dropped in "
                       "open_reader(), and the placement file is opened only inside "
                       "month_placement_diagnostic(), which asserts that every "
                       "partition is frozen"),
            token_memmap_dropped=True,
            placement_read_after_selection=True),
        row_layout_check=layout,
        verification=verifications,
        cells=cells,
        month_placement_diagnostic=diag,
        caveats=[
            "DECLARED EXPLORATORY. Not pre-registered. Nothing here can change a "
            "pre-registered gate verdict.",
            "The threshold sweep is extended to 0.75 and 0.85 a priori because "
            "window-level base co-fire rates are mechanically higher than "
            "per-position ones; the SELECTION rule is the prereg's frozen rule "
            "unchanged, and no threshold outside the declared sweep is ever tried.",
            "--max-edges is a computational guard, not a statistical choice. A "
            "threshold skipped by it is recorded with its exact edge count and max "
            "degree and treated as NOT satisfying the frozen rule; whenever a "
            "skipped threshold lies below the selected one, selection_caveat says so.",
            "Windows tile sequences rather than sliding, so two latents firing 1 "
            "token apart across a tile boundary do not co-occur. This is a "
            "deliberate, declared simplification: tiling makes occupancy a clean "
            "windows x latents indicator matrix, which is what makes the exact "
            "Gram accumulation affordable.",
            "The month-placement block is a POST-SELECTION diagnostic. It annotates "
            "partitions that were already frozen and cannot have influenced them.",
        ],
        total_runtime_s=round(time.time() - t_start, 1))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {out_path}  ({out['total_runtime_s']}s)")
    for key, cell in cells.items():
        print(f"  {key}: {cell['rule_status']} T={cell['selected_threshold']} "
              f"n_clusters={cell.get('n_clusters_selected')} "
              f"n_ge6={cell.get('n_clusters_selected_ge6')}")


if __name__ == "__main__":
    main()
