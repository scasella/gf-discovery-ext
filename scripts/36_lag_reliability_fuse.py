"""Phase 3, Stage 1, probe 3 -- lag-conditioned action geometry FUSE PROBE.

Pre-registered: docs/phase3_prereg_2026-08-21.md sec 2.3 (FROZEN). Output:
results/phase3_lag_fuse.json. Funding rule (sec 6): scripts/41_lag_action.py runs
iff this script PASSes. PASS/FAIL only -- no closure test is performed here and no
label enters any decision statistic (R3); the strict-12 month median is a post-freeze
ANNOTATION.

The question. "Lag-conditioned action geometry" wants to read the L6 SAE code of the
token that FOLLOWS a type t, minus what that successor would have contributed by its
own identity, as an "action" vector attached to t. That object is only worth 2.5 h of
Stage-4 compute if it is RELIABLE: if the action estimated from one half of the corpus
agrees with the action estimated from the disjoint other half. This script measures
exactly that split-half reliability and nothing else.

Two critic repairs from docs/next_avenues_2026-08-21.md are structural here and are
implemented, not merely noted:
  * the reliability gate is CENTRED. Uncentred split-half cosines pass on common-mode:
    every type's successor mass shares a large corpus-mean direction, so raw cosines
    are high whatever the type is. Centring each half on its own pool mean removes it.
    The uncentred median is reported alongside so the size of the common-mode is visible.
  * identity removal is restructured as C @ TM. The naive form subtracts the successor's
    own type mean once per pair (~2.5e6 pairs x 16384 float64 = ~512 GB of traffic);
    C @ TM does the same arithmetic as one 3795x3795 by 3795x16384 matmul because
    sum over pairs of TM[s] == sum over s of C[t,s] * TM[s].
The third repair -- 20.0% of pairs have an out-of-pool successor -- is declared below:
those pairs are DROPPED (not backed off to a global mean), and the realised drop rate
is reported so it can be checked against the critic's figure.

Streaming and memory. One pass over the layer-6 CSR memmaps in 200,000-row chunks. The
pass accumulates the type means TM and the parity-split successor sums S together, so
the cache is read once. There is NO chunk-boundary carry: the accumulation is written
per SUCCESSOR ROW j, and everything about j's predecessor comes from tokens.i32, which
is fully resident. The pair test (i mod 255) < 254 with i = j-1 is exactly j mod 255 != 0
(and j = 0 is excluded by that same test), and because a valid pair never straddles a
sequence, (i // 255) == (j // 255), so the predecessor's sequence parity is (j // 255) % 2.
Accumulators: TM 3795x16384 f64 (497 MB), S (2*3795)x16384 f64 (994 MB, the two parity
halves stacked), C 2x3795x3795 i64 (230 MB). Peak RSS ~2.3 GB including chunk buffers.

Cache facts asserted, not assumed (prereg sec 0): 4,002,480 rows = 15,696 sequences x 255
tokens, BOS dropped at cache build, so a row index is a document-relative index. Nothing
here is reported as a claim about absolute sequence position.
"""
import argparse, gc, json, os, sys, time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HUB_OFFLINE", "1")
sys.path.insert(0, str(REPO / "src"))
from gfdisc.saes import load_sae, get_decoder
from gfdisc.tokens import MONTHS

# ===========================================================================
# DECLARED RULES -- frozen by docs/phase3_prereg_2026-08-21.md sec 2.3
# ===========================================================================
COUNT_FLOOR = 100          # build_pool main pool (scripts/22_token_type_geometry.py L211)
D_MODEL = 1152
D_SAE = 16384
SEQ_LEN = 255              # cached tokens per sequence, BOS already dropped
N_ROWS_EXPECTED = 4_002_480
N_POOL_EXPECTED = 3_795
MIN_PAIRS_PER_HALF = 20    # a type below this in EITHER half leaves the median
GATE_THRESHOLD = 0.30      # PASS iff centred median cosine >= this
CHUNK = 200_000            # streaming chunk, rows
# prereg sec 0, the 13-member L6 cluster; strict-12 is this set minus 5656 (" Feb")
MEMBERS_13 = [3353, 5142, 5261, 5365, 5505, 5656, 5729, 5857, 6027, 6198, 6243, 6384, 7250]
STRICT_12_EXPECTED = sorted(x for x in MEMBERS_13 if x != 5656)


def declared_rules():
    return dict(
        spec="docs/phase3_prereg_2026-08-21.md sec 2.3, frozen",
        layer=6,
        population=(f"the {N_POOL_EXPECTED}-type pool: token types with count >= "
                    f"{COUNT_FLOOR} over the {N_ROWS_EXPECTED} cache rows (build_pool "
                    "main ids, scripts/22_token_type_geometry.py L211)"),
        pair_rule=("ordered adjacent pair (i, i+1) is admissible iff (i mod 255) < 254, "
                   "which never straddles a cached sequence. Predecessor type t = "
                   "lut[tok[i]] must be in the pool. Equivalent successor-row form used "
                   "in the streaming pass: row j = i+1 contributes iff j mod 255 != 0 and "
                   "lut[tok[j-1]] >= 0 and lut[tok[j]] >= 0"),
        out_of_pool_successors=("DROPPED. Pairs whose successor token is not a pool type "
                                "contribute to neither S nor C. The dropped fraction is "
                                "computed among pairs with a VALID PREDECESSOR and is "
                                "reported (critic's pre-measurement: ~20.0%)"),
        split=("sequence parity of the PREDECESSOR row: parity = (i // 255) % 2; because "
               "an admissible pair never straddles a sequence this equals (j // 255) % 2"),
        type_means=("TM[t] = mean SAE code over ALL cache rows of type t, accumulated in "
                    "float64 in the same streaming pass and divided by the count of rows "
                    "actually streamed (identical to the full-corpus count on a full run; "
                    "the equality is asserted when --max-rows is not used)"),
        identity_removal=("A_par = S_par - C_par @ TM, in the 16384-d code space, float64. "
                          "C_par[t, s] is the count of admissible parity-par pairs from t "
                          "to s, so C_par @ TM is exactly the sum over those pairs of the "
                          "successor's own type mean -- the mass the successor would carry "
                          "by identity alone"),
        normalization=("A_par[t] /= sum_s C_par[t, s], the number of parity-par pairs with "
                       "predecessor t, giving a mean context-effect per type"),
        inclusion=(f"a type enters the median iff it has >= {MIN_PAIRS_PER_HALF} pairs in "
                   "BOTH halves; the excluded count is reported"),
        projection=("A_even, A_odd cast to float32 and multiplied by w_dec "
                    "([16384, 1152]); b_dec is NOT added (it is a constant offset that "
                    "the centring removes exactly)"),
        centring=("each half is centred on its OWN pool mean, taken over the INCLUDED "
                  "types only. This is the critic's mandated repair: the uncentred "
                  "statistic passes on the shared corpus-mean successor direction"),
        statistic=("per included type t: cosine(A_even_centred[t], A_odd_centred[t]); "
                   "the gate statistic is the MEDIAN over included types"),
        gate=f"PASS iff centred median >= {GATE_THRESHOLD}; FAIL otherwise. No closure test.",
        labels=("R3-compliant: no label enters the gate. The strict-12 month median is "
                "computed AFTER the blind statistic and is reported as annotation only. "
                "strict-12 = highest-count in-pool id per month from results/token_map.json, "
                f"asserted equal to the frozen set {STRICT_12_EXPECTED} (prereg sec 0's "
                "13-member set minus 5656 = ' Feb')"),
        position_disclaimer=("row indices are document-relative (prereg sec 0); the parity "
                             "split is over SEQUENCES, not positions, so no statistic here "
                             "is a claim about absolute sequence position"),
        consequence="scripts/41_lag_action.py is funded iff this script returns PASS",
    )


# ===========================================================================
def build_pool(root, layer):
    """Token types over the count floor. tokens.i32 is byte-identical across layers."""
    tok = np.fromfile(Path(root) / f"layer_{layer}" / "tokens.i32", dtype=np.int32)
    cnt = np.bincount(tok)
    main = np.flatnonzero(cnt >= COUNT_FLOOR)
    return tok, cnt, main


def pair_index(tok, n_used, lut, n_types):
    """Per-successor-row bookkeeping for the admissible pairs in rows [0, n_used).

    Returns
      own      int32[n_used]  pool index of the row's own token, -1 if out of pool
      prev     int32[n_used]  pool index of the row's predecessor, -1 if out of pool
      parity   int8[n_used]   (j // 255) % 2, the predecessor's sequence parity
      pred_ok  bool[n_used]   admissible pair AND predecessor in pool (drop denominator)
      succ_ok  bool[n_used]   pred_ok AND successor in pool (what actually accumulates)
    """
    own = lut[tok[:n_used]].astype(np.int32)
    prev = np.empty(n_used, dtype=np.int32)
    prev[0] = -1
    prev[1:] = own[:n_used - 1]
    j = np.arange(n_used, dtype=np.int64)
    same_seq = (j % SEQ_LEN) != 0          # (i mod 255) < 254 for i = j-1; kills j = 0 too
    pred_ok = same_seq & (prev >= 0)
    succ_ok = pred_ok & (own >= 0)
    parity = ((j // SEQ_LEN) % 2).astype(np.int8)
    return own, prev, parity, pred_ok, succ_ok


def stream(root, layer, own, pred_par, n_used, n_types, d_sae, chunk=CHUNK):
    """ONE pass: type-mean sums TM_acc and parity-split successor sums S.

    pred_par[j] = predecessor pool index + parity * n_types for an accumulating row,
    else -1. Stacking the two parity halves into one accumulator keeps the pass to a
    single np.add.at per chunk for S and avoids a per-nonzero parity array.
    """
    d = Path(root) / f"layer_{layer}"
    indptr = np.load(d / "indptr.npy")
    indices = np.memmap(d / "indices.u16", dtype=np.uint16, mode="r")
    values = np.memmap(d / "values.f16", dtype=np.float16, mode="r")

    TM_acc = np.zeros((n_types, d_sae), dtype=np.float64)
    S = np.zeros((2 * n_types, d_sae), dtype=np.float64)
    seen = np.zeros(n_types, dtype=np.int64)
    nnz_total = 0
    t0 = time.time()
    for lo in range(0, n_used, chunk):
        hi = min(lo + chunk, n_used)
        a, b = int(indptr[lo]), int(indptr[hi])
        cols = np.asarray(indices[a:b], dtype=np.int32)
        vals = np.asarray(values[a:b], dtype=np.float32)
        ct = (indptr[lo + 1:hi + 1] - indptr[lo:hi]).astype(np.int64)
        nnz_total += int(b - a)

        rid = np.repeat(own[lo:hi], ct)
        m = rid >= 0
        np.add.at(TM_acc, (rid[m], cols[m]), vals[m])
        seen += np.bincount(own[lo:hi][own[lo:hi] >= 0], minlength=n_types)
        del rid, m
        gc.collect()

        rid = np.repeat(pred_par[lo:hi], ct)
        m = rid >= 0
        np.add.at(S, (rid[m], cols[m]), vals[m])
        del rid, m, cols, vals, ct
        gc.collect()

        done = hi
        el = time.time() - t0
        print(f"  stream {done}/{n_used} rows  nnz={nnz_total}  "
              f"{el:.1f}s  ({done / max(el, 1e-9) / 1e3:.0f}k rows/s)", flush=True)
    return TM_acc, S, seen, nnz_total


def project(A, W, chunk=512):
    out = np.empty((A.shape[0], W.shape[1]), dtype=np.float32)
    for lo in range(0, A.shape[0], chunk):
        hi = min(lo + chunk, A.shape[0])
        out[lo:hi] = A[lo:hi].astype(np.float32) @ W
    return out


def rowwise_cosine(X, Y):
    nx = np.linalg.norm(X, axis=1)
    ny = np.linalg.norm(Y, axis=1)
    return (X * Y).sum(1) / np.maximum(nx * ny, 1e-12)


def strict12(cnt, pool_set, token_map):
    """Highest-count in-pool token id per month. Labels touch nothing before this point."""
    ids, missing = [], []
    for name in MONTHS:
        cands = [int(x) for x in token_map["months"][name]
                 if int(x) in pool_set and int(x) < len(cnt)]
        if not cands:
            missing.append(name)
            continue
        ids.append(max(cands, key=lambda x: int(cnt[x])))
    return ids, missing


# ===========================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="cache/sparse_big")
    ap.add_argument("--layer", type=int, default=6)
    ap.add_argument("--l0", default="big")
    ap.add_argument("--out", default="results/phase3_lag_fuse.json")
    ap.add_argument("--max-rows", type=int, default=0,
                    help="SMOKE ONLY: truncate the cache pass to this many rows")
    a = ap.parse_args()
    t_start = time.time()
    smoke = a.max_rows > 0

    # ---- R1: declaration on disk BEFORE any cache row is read ----
    declared_at = time.strftime("%Y-%m-%dT%H:%M:%S")
    out_path = Path(a.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(dict(
        status="DECLARATION STUB -- rules frozen, no results yet",
        script="scripts/36_lag_reliability_fuse.py",
        declared_at=declared_at, layer=a.layer, smoke=smoke,
        declared_rules=declared_rules()), indent=2))
    print("=== DECLARED RULES (frozen, on disk before any cache row is read) ===", flush=True)
    print(json.dumps(declared_rules(), indent=2), flush=True)
    print(f"\ndeclared_at = {declared_at}   smoke = {smoke}", flush=True)

    root = Path(a.root)
    if not root.is_absolute():
        root = REPO / root

    # ---- pool ----
    tok, cnt, pool_ids = build_pool(root, a.layer)
    n_rows_cache = len(tok)
    n_types = len(pool_ids)
    print(f"\ncache rows {n_rows_cache}  sequences {n_rows_cache // SEQ_LEN}  "
          f"pool types {n_types} at floor {COUNT_FLOOR}", flush=True)
    assert n_rows_cache == N_ROWS_EXPECTED, f"row count {n_rows_cache}"
    assert n_rows_cache % SEQ_LEN == 0, "cache rows not a multiple of 255"
    assert n_types == N_POOL_EXPECTED, f"pool size {n_types}"

    meta = json.loads((root / f"layer_{a.layer}" / "meta.json").read_text())
    d_sae = int(meta["d_sae"])
    assert d_sae == D_SAE, f"d_sae {d_sae}"

    lut = np.full(int(tok.max()) + 1, -1, dtype=np.int64)
    lut[pool_ids] = np.arange(n_types)

    n_used = min(a.max_rows, n_rows_cache) if smoke else n_rows_cache
    own, prev, parity, pred_ok, succ_ok = pair_index(tok, n_used, lut, n_types)

    n_pred = int(pred_ok.sum())
    n_acc = int(succ_ok.sum())
    dropped_pair_fraction = float(1.0 - n_acc / n_pred) if n_pred else float("nan")
    print(f"\npairs with valid predecessor {n_pred}; accumulating pairs {n_acc}; "
          f"dropped (successor out of pool) fraction {dropped_pair_fraction:.5f}", flush=True)

    # ---- successor-count matrices C[parity] ----
    C = np.zeros((2, n_types, n_types), dtype=np.int64)
    for p in (0, 1):
        m = succ_ok & (parity == p)
        np.add.at(C[p], (prev[m], own[m]), 1)
        del m
    n_pairs_even = int(C[0].sum())
    n_pairs_odd = int(C[1].sum())
    assert n_pairs_even + n_pairs_odd == n_acc, "C total disagrees with the pair mask"
    print(f"C built: n_pairs_even {n_pairs_even}  n_pairs_odd {n_pairs_odd}", flush=True)

    pred_par = np.where(succ_ok, prev + parity.astype(np.int32) * n_types,
                        np.int32(-1)).astype(np.int32)
    del prev, parity, pred_ok, succ_ok
    gc.collect()

    # ---- one streaming pass: TM and S together ----
    print(f"\nstreaming layer {a.layer} in {CHUNK}-row chunks "
          f"({n_used} rows){' [TRUNCATED SMOKE PASS]' if smoke else ''}", flush=True)
    t_stream = time.time()
    TM_acc, S, seen, nnz_total = stream(root, a.layer, own, pred_par, n_used,
                                        n_types, d_sae)
    stream_seconds = time.time() - t_stream
    del own, pred_par
    gc.collect()
    print(f"stream done in {stream_seconds:.1f}s, {nnz_total} nonzeros", flush=True)

    if not smoke:
        assert np.array_equal(seen, cnt[pool_ids].astype(np.int64)), \
            "streamed per-type row counts differ from the full-corpus counts"
        divisor_rule = "full-corpus per-type counts (streamed counts asserted equal)"
    else:
        divisor_rule = ("per-type counts observed in the TRUNCATED prefix -- equal to the "
                        "full-corpus counts on a full run")
    n_types_with_rows = int((seen > 0).sum())
    TM_acc /= np.maximum(seen, 1)[:, None]
    TM_acc[seen == 0] = 0.0
    TM = TM_acc                      # [n_types, d_sae] float64 type means

    # ---- identity removal: A_par = S_par - C_par @ TM, blocked over rows ----
    t_mm = time.time()
    for p in (0, 1):
        base = p * n_types
        for lo in range(0, n_types, 512):
            hi = min(lo + 512, n_types)
            S[base + lo:base + hi] -= C[p][lo:hi].astype(np.float64) @ TM
    matmul_seconds = time.time() - t_mm
    print(f"identity removal (C @ TM, both halves) in {matmul_seconds:.1f}s", flush=True)
    del TM, TM_acc
    gc.collect()

    # ---- per-type mean context effect, and the inclusion mask ----
    rs = C.sum(axis=2)               # [2, n_types] pairs per predecessor type per half
    included = (rs[0] >= MIN_PAIRS_PER_HALF) & (rs[1] >= MIN_PAIRS_PER_HALF)
    n_incl = int(included.sum())
    print(f"included types: {n_incl}/{n_types} "
          f"(>= {MIN_PAIRS_PER_HALF} pairs in BOTH halves); excluded {n_types - n_incl}",
          flush=True)
    for p in (0, 1):
        S[p * n_types:(p + 1) * n_types] /= np.maximum(rs[p], 1)[:, None]
    del C
    gc.collect()

    # ---- project through w_dec (no b_dec) ----
    sae = load_sae(a.layer, a.l0, token=os.environ.get("HF_TOKEN"))
    W = get_decoder(sae)
    del sae
    assert W.shape == (d_sae, D_MODEL), f"w_dec shape {W.shape}"
    P_even = project(S[:n_types], W)
    P_odd = project(S[n_types:], W)
    del S, W
    gc.collect()
    print(f"projected: {P_even.shape} per half", flush=True)

    # ---- blind statistic: centred split-half cosine ----
    cos_unc = rowwise_cosine(P_even.astype(np.float64), P_odd.astype(np.float64))
    Ce = P_even.astype(np.float64) - P_even[included].astype(np.float64).mean(0, keepdims=True)
    Co = P_odd.astype(np.float64) - P_odd[included].astype(np.float64).mean(0, keepdims=True)
    cos_cen = rowwise_cosine(Ce, Co)

    ci, ui = cos_cen[included], cos_unc[included]
    centred_median = float(np.median(ci))
    uncentred_median = float(np.median(ui))
    pct = dict(centred_p25=float(np.percentile(ci, 25)),
               centred_p75=float(np.percentile(ci, 75)),
               uncentred_p25=float(np.percentile(ui, 25)),
               uncentred_p75=float(np.percentile(ui, 75)))
    gate = "PASS" if centred_median >= GATE_THRESHOLD else "FAIL"
    print(f"\ncentred median cosine   {centred_median:.5f}  "
          f"[p25 {pct['centred_p25']:.5f}, p75 {pct['centred_p75']:.5f}]", flush=True)
    print(f"uncentred median cosine {uncentred_median:.5f}  "
          f"[p25 {pct['uncentred_p25']:.5f}, p75 {pct['uncentred_p75']:.5f}]", flush=True)
    print(f"GATE ({GATE_THRESHOLD}): {gate}", flush=True)

    # ---- post-freeze ANNOTATION: strict-12 months. Nothing above depends on it. ----
    token_map = json.loads((REPO / "results/token_map.json").read_text())
    pool_set = set(int(x) for x in pool_ids)
    s12, missing_months = strict12(cnt, pool_set, token_map)
    assert sorted(s12) == STRICT_12_EXPECTED and not missing_months, \
        f"strict-12 resolved to {sorted(s12)}, expected {STRICT_12_EXPECTED}"
    pos12 = np.array([int(np.searchsorted(pool_ids, t)) for t in s12])
    incl12 = pos12[included[pos12]]
    month_annotation = dict(
        note="post-freeze annotation only; no gate depends on it (prereg R3)",
        strict_12_ids=sorted(int(x) for x in s12),
        ids_in_calendar_order={m: int(t) for m, t in zip(MONTHS, s12)},
        n_in_pool=len(s12), n_included=int(len(incl12)),
        excluded_ids=sorted(int(pool_ids[p]) for p in pos12 if not included[p]),
        pairs_even={str(int(pool_ids[p])): int(rs[0][p]) for p in pos12},
        pairs_odd={str(int(pool_ids[p])): int(rs[1][p]) for p in pos12},
        centred_median=float(np.median(cos_cen[incl12])) if len(incl12) else None,
        uncentred_median=float(np.median(cos_unc[incl12])) if len(incl12) else None,
        centred_per_type={str(int(pool_ids[p])): float(cos_cen[p]) for p in pos12})
    print(f"\n[annotation] strict-12 months: {len(incl12)}/12 included, centred median "
          f"{month_annotation['centred_median']}, uncentred "
          f"{month_annotation['uncentred_median']}", flush=True)

    runtime_seconds = round(time.time() - t_start, 1)
    result = dict(
        status="RESULT",
        script="scripts/36_lag_reliability_fuse.py",
        declared_at=declared_at,
        declared_rules=declared_rules(),
        layer=a.layer, l0=a.l0, root=str(a.root),
        smoke=smoke,
        smoke_note=("TRUNCATED PASS -- plumbing check only. TM and S cover a prefix of the "
                    "cache, so few types clear the 20-pair floor and no number here is a "
                    "verdict.") if smoke else None,
        gate=gate, gate_threshold=GATE_THRESHOLD,
        centred_median=centred_median, uncentred_median=uncentred_median,
        percentiles=pct,
        n_types_pool=n_types, n_types_included=n_incl,
        n_types_excluded=n_types - n_incl,
        min_pairs_per_half=MIN_PAIRS_PER_HALF,
        n_types_with_rows_streamed=n_types_with_rows,
        dropped_pair_fraction=dropped_pair_fraction,
        n_pairs_valid_predecessor=n_pred, n_pairs_accumulated=n_acc,
        n_pairs_even=n_pairs_even, n_pairs_odd=n_pairs_odd,
        n_rows_cache=n_rows_cache, n_rows_streamed=n_used, n_nonzero_streamed=nnz_total,
        type_mean_divisor=divisor_rule,
        month_annotation=month_annotation,
        stream_seconds=round(stream_seconds, 1),
        identity_removal_seconds=round(matmul_seconds, 1),
        runtime_seconds=runtime_seconds,
        consequence=("scripts/41_lag_action.py is funded" if gate == "PASS" and not smoke
                     else "scripts/41_lag_action.py is NOT funded by this run"))
    out_path.write_text(json.dumps(result, indent=2))
    print(f"wrote {a.out} ({runtime_seconds}s)", flush=True)


if __name__ == "__main__":
    main()
