"""Stage 2, avenue 4 -- IS THE POSITION COORDINATE ABSOLUTE OR DOCUMENT-RELATIVE?

Pre-registered in docs/phase3_prereg_2026-08-21.md section 3.2 (frozen 2026-08-21); design
and critic reordering in docs/next_avenues_2026-08-21.md avenue 4. Writes
results/phase3_position_causal.json plus results/phase3_position_causal_data.npz.

THE QUESTION. gemma-3-1b-pt has no learned position embedding: every positional signal in
resid_post is written by RoPE plus whatever the attention pattern induces. Some direction in
resid_post nevertheless varies smoothly with where a token sits. Prereg section 0 forbids
calling that direction "absolute position" from the cache alone, and for a hard reason: the
cache is 15,696 FineWeb documents each truncated to 255 tokens with BOS dropped, so cache
row-index p = r % 255 is a DOCUMENT-RELATIVE index and is perfectly confounded with absolute
sequence position. Every cached document starts at absolute position 1. The cache cannot
separate "the model tracks its offset from BOS" from "the model tracks how far into the
document's own discourse it is".

THE THREE ARMS.

  cache arm  -- recovers the coordinate. 133 token types with >= 2,550 cache occurrences x 8
                equal-width bins over p in 1..254; per-(type, bin) mean SAE code projected
                through w_dec; within-type centring (so the coordinate is what is LEFT after
                token identity is removed); per-row L2; the top right-singular vector u_L is
                the coordinate. beta_rel = its OLS slope on bin index. Observational, and by
                construction it cannot identify anything -- it is the yardstick the causal
                arms are measured against.
  arm (a)     -- position_ids offset. IDENTICAL token ids, IDENTICAL attention set, RoPE
                phases shifted by j. In transformers 5.x the Gemma3 masks are built from
                query/key INDICES (masking_utils.sliding_window_overlay reads q_idx/kv_idx)
                and position_ids feeds only self.rotary_emb, and Gemma3-1b's sliding window
                (512) exceeds every sequence here (<= 249), so the attention set is provably
                unchanged across j. This arm moves RoPE and nothing else.
  arm (b)     -- prefix splice. j real tokens from a different cached document are prepended,
                so the doc's absolute positions move AND its attention set gains j distractor
                keys. Confounded on purpose: it is the arm whose confound arm (a) lacks.

CONCORDANCE IS THE CLAIM. Neither causal arm identifies absolute position alone -- (a) is
mildly off-distribution (BOS no longer sits at position 0) and (b) changes the context. The
registered call is their AGREEMENT: both beta_abs CIs excluding zero with the same sign, at a
magnitude comparable to the cache-arm slope, is what "the coordinate is absolute" would look
like; both containing zero while beta_rel excludes zero is what "document-relative" would
look like; anything else is UNIDENTIFIED. That three-way rule is frozen and is not re-tuned
here.

TWO DECLARED INTERPRETATIONS OF THE FROZEN TEXT. Both are written to the declaration stub
before any cache row is read, and both are flagged in the output rather than buried.

  (1) THE REGISTERED NULL IS MATHEMATICALLY DEGENERATE, and this script proves it rather than
      quietly repairing it. Section 3.2 says: permute each type's 8 bin rows; statistic =
      variance explained by the top singular value of the (133*8) x 1152 matrix. But a
      within-type row permutation is a permutation of the FULL matrix's rows, i.e. an
      orthogonal transform, and singular values are invariant under it. Within-type centring
      is likewise invariant (permuting a type's rows does not move that type's mean) and the
      L2 normalisation is per-row. The statistic is therefore a point mass at the observed
      value and the test can never reject at any threshold. The script computes it on a
      handful of draws and reports max|null - real| (~1e-15) as the numerical proof.
      Two REPAIRED statistics are declared in its place, both scored at the same frozen
      99.9th-percentile rule, both NON-VERDICT-BEARING (section 3.2's ABSOLUTE /
      DOCUMENT-RELATIVE / UNIDENTIFIED call reads only the beta_abs CIs, the ratio CI and the
      beta_rel CI -- this null enters none of them):
        bin_svd  PRIMARY REPAIR: s1^2 / sum s^2 of the 8 x 1152 matrix of bin means taken
                 ACROSS types. Minimal change to the frozen wording -- same statistic, same
                 rule, computed on the one matrix a within-type permutation actually moves.
        bin_r2   SECONDARY: R^2 of the pooled OLS of (u_B . row) on bin index, u_B = the top
                 right-singular vector of that bin-mean matrix. Directly tests the quantity
                 beta_rel reports.
  (2) THE RATIO NEEDS MATCHED UNITS, and per-row L2 normalisation destroys scale. The frozen
      text asks for "per-position units matched" in the beta_abs/beta_rel ratio, which is
      impossible if beta_rel is a slope of unit-norm rows. The verdict-bearing beta_rel is
      therefore the slope on the within-type-centred, UN-normalised projected means -- which
      is exactly the type-fixed-effect estimator, matching the (doc, i) fixed-effect design
      of beta_abs. The literal normalised slope is co-reported. One residual unit caveat
      cannot be fixed and is stated in the output: the cache rows are SAE reconstructions
      (code @ w_dec) while the causal arms read the RAW residual stream, so the ratio is a
      cross-representation comparison. The j=0 c(i)-vs-i profile is reported against the cache
      bin profile as the empirical bridge (a rank correlation, not an equality claim).

WHY u_L AND NOT u_B DRIVES THE CAUSAL ARMS. Section 3.2 names u_L -- the top right-singular
vector of ALL (type, bin) rows -- as the coordinate, and that is what the verdict-bearing
beta_abs reads. u_B (the bin-mean direction) is a different estimator of the same idea, so
c(i, j) is computed along BOTH in the same forward and beta_abs is reported for both; only
u_L feeds the call. |u_L . u_B| is reported: if the top variance direction of the residual
type-by-position cloud is not the position direction, that is worth knowing before any
beta_abs is read.

WHAT IS COPIED AND FROM WHERE. The streaming CSR groupby is scripts/22_token_type_geometry.py
type_means (L220) re-keyed from type to (type, bin) and with np.add.at replaced by a flat
np.bincount (identical arithmetic in float64, ~20x faster at this entry count); project()
is scripts/22 L247; the declaration-stub-before-any-read discipline is scripts/22 L579-586
(prereg R1). Batching, forwards and captures go through src/gfdisc/rig.py unchanged, except
for the local build_seq_batch: rig.build_batch appends a target token to every row, and this
script needs the raw sequence.
"""
import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import gfdisc.rig as rig                                    # shared Stage-2 rig, unmodified
from gfdisc.env import DEVICE
from gfdisc.model import load_model
from gfdisc.saes import get_decoder, load_sae


# ===========================================================================
# frozen constants -- prereg section 3.2
# ===========================================================================
LAYERS = [6, 9]                       # L6 primary, L9 co-reported
COUNT_FLOOR = 2550                    # cache arm type floor (expected 133 types)
N_BINS = 8
P_LO, P_HI = 1, 254                   # row-index range; p = r % 255, p = 0 dropped
BIN_WIDTH = (P_HI - P_LO + 1) / N_BINS   # 254 / 8 = 31.75 cache rows per bin
D_MODEL = 1152

N_NULL = 1000                         # bin-permutation draws
NULL_PCT = 99.9                       # frozen threshold percentile
N_LITERAL_DEMO = 5                    # draws used to demonstrate the null's degeneracy
POWER_ALPHAS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
N_POWER = 100                         # draws per planted amplitude

N_DOCS = 300                          # causal-arm documents
DOC_TOKENS = 120                      # doc = tokens2d[doc, :120]
SHIFTS = [0, 16, 32, 64, 128]
I_LO, I_HI = 20, 119                  # doc-relative read positions, inclusive
N_BOOT = 2000                         # cluster bootstrap draws (docs; and types)
CI_LO, CI_HI = 2.5, 97.5              # percentile CI

RATIO_FLOOR = 0.5                     # frozen: lower CI bound of beta_abs/beta_rel > 0.5
ARMS = ["position_ids_offset", "prefix_splice"]

SEED_DOCS = 0                         # prereg: "300 cached documents, seed 0"
SEED_OTHER = 1                        # splice donor documents
SEED_NULL = 20260821
SEED_POWER = 20260822
SEED_BOOT_DOCS = 20260823
SEED_BOOT_TYPES = 20260824

MAX_ABS_DIFF_DEFAULT = 1e-3           # arm (a) j=0 must reproduce the default forward
MIN_ABS_DIFF_SHIFT = 1e-2             # arm (a) j=16 must differ from j=0
VERIFY_SHIFT = 16


def declared_rules(a, layers, shifts):
    """Everything frozen before the run, written to disk before any cache row is read."""
    return dict(
        cache_arm=dict(
            population=f"token types with FULL-cache count >= {COUNT_FLOOR} (expected 133)",
            bins=(f"8 equal-width bins over cache row-index p = r % 255 restricted to "
                  f"{P_LO}..{P_HI}; p = 0 dropped; bin0 = floor(8*(p-1)/254), reported "
                  f"1-based as b in 1..8; bin width {BIN_WIDTH} cache rows"),
            position_semantics=(
                "cache column p is DOCUMENT-RELATIVE index p (0-based into the BOS-stripped "
                "document; rig.load_tokens2d docstring). Prereg section 0 forbids reading it "
                "as absolute sequence position -- which is precisely why the causal arms exist"),
            representation=("per (type, bin): float64 sum of sparse SAE code vectors / count, "
                            "then mean_code @ w_dec. NO b_dec (project-wide rule, "
                            "src/gfdisc/rig.py docstring)"),
            processing=("within-type centring (subtract each type's mean over its 8 bins), "
                        "then per-row L2 normalisation"),
            u_L=("top right-singular vector of the (n_types*8) x 1152 processed matrix; sign "
                 "fixed so corr(u_L . row, bin index) > 0"),
            u_B=("SECONDARY estimator, not verdict-bearing: top right-singular vector of the "
                 "8 x 1152 across-type bin-mean matrix, same sign rule"),
            beta_rel_primary=(
                "OLS slope of (u_L . row) on bin index computed on the within-type-centred "
                "UN-normalised projected means -- the type-fixed-effect estimator, which is "
                "what matches beta_abs's (doc, i) fixed-effect units. DECLARED "
                "INTERPRETATION: the frozen text asks for matched per-position units in the "
                "beta_abs/beta_rel ratio, and per-row L2 normalisation destroys scale"),
            beta_rel_literal=("co-reported: the same slope on the L2-normalised rows, i.e. "
                              "the frozen wording taken literally (pooled OLS; after "
                              "normalisation the rows are no longer exactly type-centred, so "
                              "no fixed effect is silently added)"),
            beta_rel_units="per bin; per token = slope / bin_width = slope / " + str(BIN_WIDTH),
            beta_rel_ci=(f"cluster bootstrap over TYPES, {a.n_boot} draws, percentile "
                         f"[{CI_LO}, {CI_HI}]; u_L held FIXED at its full-sample value "
                         f"(recomputing the SVD per draw carries a sign-flip pathology and "
                         f"2,000 SVDs of a 1064x1152 matrix). This slightly understates the "
                         f"CI width and is declared, not discovered"),
        ),
        null=dict(
            draw="permute each type's 8 bin rows independently; 1,000 draws",
            registered_statistic=(
                "s1^2 / sum s^2 of the full (n_types*8) x 1152 matrix. DECLARED BEFORE THE "
                "RUN TO BE DEGENERATE: a within-type row permutation is a permutation of the "
                "full matrix's rows, hence an orthogonal transform, and singular values are "
                "invariant under it; within-type centring and per-row L2 are likewise "
                "invariant. The statistic is a point mass at the observed value and cannot "
                f"reject at any threshold. Demonstrated numerically on {N_LITERAL_DEMO} draws"),
            repair_primary=("bin_svd = s1^2 / sum s^2 of the 8 x 1152 across-type bin-mean "
                            "matrix -- the same statistic on the one matrix the permutation "
                            "actually moves"),
            repair_secondary=("bin_r2 = R^2 of pooled OLS of (u_B . row) on bin index, u_B "
                              "recomputed per draw from that draw's bin-mean matrix"),
            threshold=f"{NULL_PCT}th percentile of the null draws (frozen rule)",
            verdict_bearing=("NO. Section 3.2's ABSOLUTE / DOCUMENT-RELATIVE / UNIDENTIFIED "
                             "call reads only the two beta_abs CIs, the ratio CI and the "
                             "beta_rel CI. This null and its repairs are reported, and gate "
                             "nothing"),
        ),
        power=dict(
            plant=("alpha * u_rand * (b - 4.5) / 3.5 added to a null draw, b = 1..8, u_rand a "
                   "random unit vector redrawn per draw and shared across types"),
            amplitudes=f"{POWER_ALPHAS} x median processed row norm (= 1 after per-row L2, "
                       f"so alpha is an absolute amplitude; reported)",
            draws=f"{N_POWER} per amplitude",
            no_renormalisation=("the plant is zero-sum over a type's 8 bins so the data stay "
                                "within-type centred; rows are NOT re-normalised after "
                                "planting (declared)"),
            readout="detection rate = fraction of draws above the frozen null threshold",
        ),
        causal_arms=dict(
            documents=f"{a.n_docs} cached documents drawn without replacement, seed {SEED_DOCS}",
            doc_tokens=f"tokens2d[doc, :{DOC_TOKENS}]",
            shifts=shifts,
            arm_a=("position_ids offset: ids = [BOS] + doc (121 tokens), full attention, "
                   "position_ids = arange(121) + j for ALL positions INCLUDING BOS. "
                   "REGISTERED AS MILDLY OOD: at j > 0 the BOS token no longer sits at "
                   "position 0, a configuration the model never saw in training. In "
                   "transformers 5.x the Gemma3 causal and sliding masks are built from "
                   "query/key indices, not position_ids, and sliding_window (512) exceeds "
                   "every sequence here, so the attention SET is unchanged across j"),
            arm_b=("prefix splice: ids = [BOS] + tokens2d[other, :j] + doc, other != doc, "
                   f"redrawn per (doc, j) with seed {SEED_OTHER}; doc-relative index i maps "
                   "to absolute position 1 + j + i"),
            forward_checks=(
                f"before any measurement: arm (a) at j=0 with explicit position_ids must "
                f"reproduce the default forward (max abs resid diff < {MAX_ABS_DIFF_DEFAULT}) "
                f"and arm (a) at j={VERIFY_SHIFT} must differ from j=0 (max abs resid diff > "
                f"{MIN_ABS_DIFF_SHIFT}). Both are asserted; the check results are written to "
                f"the output file BEFORE the assertion so a failure is diagnosable"),
            readout=(f"c(i, j) = u . resid_L at doc-relative i in {I_LO}..{I_HI} (sequence "
                     f"index 1+i for arm (a), 1+j+i for arm (b)); u = u_L (verdict-bearing) "
                     f"and u_B (diagnostic)"),
            beta_abs=("OLS slope of c on j with document x position fixed effects: c and j "
                      "are demeaned within each (doc, i) cell before the slope is taken"),
            beta_abs_ci=(f"cluster bootstrap over DOCUMENTS, {a.n_boot} draws, percentile "
                         f"[{CI_LO}, {CI_HI}]"),
            ratio=(f"beta_abs / beta_rel_per_token, pairing independent bootstrap draws "
                   f"(documents for the numerator, types for the denominator). Arm (a) "
                   f"supplies the verdict-bearing ratio; arm (b)'s is co-reported. UNIT "
                   f"CAVEAT: the numerator reads the RAW residual stream, the denominator "
                   f"reads SAE reconstructions (code @ w_dec) of cache rows; the ratio is a "
                   f"cross-representation comparison and is reported as such"),
            matched_units_ratio=(
                "DIAGNOSTIC, not verdict-bearing. beta_abs / beta_rel_fresh, where "
                "beta_rel_fresh is the OLS slope of c on doc-relative i at j=0 with document "
                "fixed effects. Numerator and denominator then come from the same tensors, "
                "the same units and the same document bootstrap draws, so this ratio is free "
                "of the registered ratio's cross-representation caveat. It is reported "
                "alongside the registered ratio; the frozen call reads the registered one"),
            consistency=(
                "raw mean of c(i) vs i at j=0 (with its rank correlation against i and "
                "against the cache bin profile), per-shift mean of c, and the arm(a)-vs-"
                "arm(b) agreement at j=0 -- where the two arms are byte-identical inputs and "
                "the max |c| difference should be 0"),
            batching=("grouped by (arm, shift) so every row in a batch has the same length; "
                      "right-padded with an attention mask through a local builder "
                      "(rig.build_batch appends a target token, which this arm must not have)"),
        ),
        decision_rule=dict(
            absolute=(f"BOTH arms' beta_abs CIs exclude 0 with the SAME sign AND the lower CI "
                      f"bound of beta_abs/beta_rel (arm a) > {RATIO_FLOOR}"),
            document_relative="BOTH beta_abs CIs contain 0 AND the beta_rel CI excludes 0",
            otherwise="UNIDENTIFIED",
            layers="L6 primary, L9 co-reported",
            direction="the call reads u_L only; u_B numbers are diagnostic",
        ),
        no_gate_note=("prereg section 3.2 registers no STOP rule and no positive-control gate "
                      "for this script. The power curve is reported whatever it shows; the "
                      "three-way call is made from the frozen rule and is not withheld"),
        no_bdec_note=("SAE reconstruction is code @ w_dec with no b_dec, matching every number "
                      "the campaign has reported; b_dec is type- and position-independent and "
                      "is annihilated exactly by the within-type centring anyway"),
    )


# ===========================================================================
# cache arm -- streaming per-(type, bin) mean codes
# ===========================================================================
def build_pool(root, floor=COUNT_FLOOR):
    """Token types over the count floor. tokens.i32 is byte-identical across layers."""
    tok = np.fromfile(Path(root) / "layer_0" / "tokens.i32", dtype=np.int32)
    if tok.size % rig.DOC_LEN:
        raise RuntimeError(f"tokens.i32 has {tok.size} rows, not a multiple of {rig.DOC_LEN}")
    cnt = np.bincount(tok)
    return tok, cnt, np.flatnonzero(cnt >= floor).astype(np.int64)


def bin_of_p(p):
    """0-based bin for cache row-index p; -1 for the dropped p = 0 rows."""
    p = np.asarray(p, dtype=np.int64)
    b = ((p - P_LO) * N_BINS) // (P_HI - P_LO + 1)
    return np.where((p < P_LO) | (p > P_HI), -1, b)


def acc_rows(tok, type_ids, n_rows_cap=0):
    """Accumulator-row index for every cache row: type_slot * 8 + bin, or -1 to skip.

    Split out so the streaming pass and the count table are built from ONE definition.
    ``n_rows_cap`` truncates the cache scan (smoke only) and is recorded in the output.
    """
    n = tok.size if not n_rows_cap else min(int(n_rows_cap), tok.size)
    slot = np.full(int(tok.max()) + 1, -1, dtype=np.int64)
    slot[type_ids] = np.arange(len(type_ids))
    p = np.tile(np.arange(rig.DOC_LEN, dtype=np.int64), tok.size // rig.DOC_LEN)[:n]
    b = bin_of_p(p)
    ts = slot[tok[:n]]
    return np.where((ts >= 0) & (b >= 0), ts * N_BINS + b, -1), n


def bin_means(root, layer, brow, n_rows, n_acc, chunk=200_000):
    """Streaming groupby of the CSR cache by (type, bin) -> [n_acc, d_sae] float64 means.

    scripts/22 type_means (L220) re-keyed from type to (type, bin). ``np.add.at`` is replaced
    by a flat ``np.bincount``: identical float64 arithmetic, ~20x faster at 200M+ entries.
    Peak extra RSS is the [n_acc * d_sae] accumulator plus one bincount temporary of the same
    size (~140 MB each at n_acc = 1064, d_sae = 16384).
    """
    d = Path(root) / f"layer_{layer}"
    meta = json.loads((d / "meta.json").read_text())
    d_sae = int(meta["d_sae"])
    indptr = np.load(d / "indptr.npy")
    indices = np.memmap(d / "indices.u16", dtype=np.uint16, mode="r")
    values = np.memmap(d / "values.f16", dtype=np.float16, mode="r")
    flat_len = n_acc * d_sae
    acc = np.zeros(flat_len, dtype=np.float64)
    used = 0
    t0 = time.time()
    for lo in range(0, n_rows, chunk):
        hi = min(lo + chunk, n_rows)
        a, b = int(indptr[lo]), int(indptr[hi])
        ct = (indptr[lo + 1:hi + 1] - indptr[lo:hi]).astype(np.int64)
        rid = np.repeat(brow[lo:hi], ct)
        m = rid >= 0
        if m.any():
            cols = np.asarray(indices[a:b])[m].astype(np.int64)
            vals = np.asarray(values[a:b])[m].astype(np.float64)
            acc += np.bincount(rid[m] * d_sae + cols, weights=vals, minlength=flat_len)
            del cols, vals
        used += int((brow[lo:hi] >= 0).sum())
        del rid, m, ct
        print(f"    L{layer} rows {hi:,}/{n_rows:,} ({time.time() - t0:.0f}s)", flush=True)
    counts = np.bincount(brow[brow >= 0], minlength=n_acc).astype(np.int64)
    acc = acc.reshape(n_acc, d_sae)
    acc /= np.maximum(counts, 1)[:, None]
    return acc, counts, d_sae, used, meta


def project(acc, W, chunk=256):
    """scripts/22_token_type_geometry.py L247."""
    out = np.empty((acc.shape[0], W.shape[1]), dtype=np.float32)
    for lo in range(0, acc.shape[0], chunk):
        hi = min(lo + chunk, acc.shape[0])
        out[lo:hi] = acc[lo:hi].astype(np.float32) @ W
    return out


# ===========================================================================
# cache-arm statistics
# ===========================================================================
def _unit(v):
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else v


def _pearson(x, y):
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    xd, yd = x - x.mean(), y - y.mean()
    den = float(np.linalg.norm(xd) * np.linalg.norm(yd))
    return float((xd @ yd) / den) if den > 0 else 0.0


def _rank(x):
    """Average-tie ranks (the 8 bin indices are distinct, so this is a plain argsort rank)."""
    x = np.asarray(x, dtype=np.float64)
    order = np.argsort(x, kind="stable")
    r = np.empty(len(x), dtype=np.float64)
    r[order] = np.arange(len(x), dtype=np.float64)
    return r


def _spearman(x, y):
    return _pearson(_rank(x), _rank(y))


def fe_slope_bins(scores3, bdm):
    """OLS slope of scores on bin index with a type fixed effect, and the per-type numerators.

    ``scores3`` is [n_types, 8]; ``bdm`` is the bin index minus its mean. With a balanced,
    identical bin grid in every type, the fixed-effect estimator reduces exactly to
    ``mean_t(num_t) / sum_b bdm^2`` with ``num_t = sum_b scores[t, b] * bdm[b]`` -- which is
    also what makes the type bootstrap a one-line vector sum.
    """
    num = (np.asarray(scores3, dtype=np.float64) * bdm[None, :]).sum(1)
    den = float((bdm ** 2).sum())
    return float(num.mean() / den), num, den


def bin_stats(A3, bdm):
    """Statistics of a processed [n_types, 8, 1152] block that a bin permutation MOVES.

    Returns ``(bin_svd, bin_r2, u_B)``. ``bin_svd`` is the frozen "variance explained by the
    top singular value" applied to the 8 x 1152 across-type bin-mean matrix; ``bin_r2`` is the
    squared correlation of (u_B . row) with bin index, u_B being that matrix's top right-
    singular vector with the frozen positive-correlation sign rule.
    """
    B = A3.mean(axis=0).astype(np.float64)                 # [8, 1152]
    G = B @ B.T                                            # [8, 8]
    ev, evec = np.linalg.eigh(G)
    ev = np.maximum(ev[::-1], 0.0)
    tot = float(ev.sum())
    bin_svd = float(ev[0] / tot) if tot > 0 else 0.0
    uB = _unit(B.T @ evec[:, -1])
    sc = np.asarray(A3, dtype=np.float64) @ uB             # [n_types, 8]
    r = _pearson(sc.ravel(), np.tile(bdm, sc.shape[0]))
    if r < 0:
        uB, sc, r = -uB, -sc, -r
    return bin_svd, float(r * r), uB


def full_svd_stat(A3):
    """The REGISTERED statistic: variance explained by the top singular value of the full
    (n_types*8) x 1152 matrix. Invariant under within-type row permutation -- see the module
    docstring; computed here only to demonstrate that invariance numerically."""
    M = np.asarray(A3, dtype=np.float64).reshape(-1, A3.shape[-1])
    s = np.linalg.svd(M, compute_uv=False)
    return float(s[0] ** 2 / (s ** 2).sum())


def perm_within_type(rng, n_types, n_draws=1):
    """[n_draws, n_types, 8] independent permutations of each type's bin rows."""
    return np.argsort(rng.random((n_draws, n_types, N_BINS)), axis=2)


def apply_perm(A3, perm):
    return A3[np.arange(A3.shape[0])[:, None], perm]


# ===========================================================================
# causal arms
# ===========================================================================
def build_seq_batch(rows, pad_id):
    """Right-padded raw sequences + attention mask + true lengths.

    Local rather than ``rig.build_batch`` because that helper appends a target token to every
    row (scripts/37's ladder needs it); this arm forwards the raw sequence and reads a hundred
    interior positions, so nothing may be appended. The padding discipline (allocate filled,
    write the real prefix, mask the rest) is identical.
    """
    lens = [len(r) for r in rows]
    lmax = max(lens)
    ids = torch.full((len(rows), lmax), int(pad_id), dtype=torch.long)
    attn = torch.zeros(len(rows), lmax, dtype=torch.long)
    for i, r in enumerate(rows):
        arr = np.asarray(r, dtype=np.int64)
        ids[i, :len(arr)] = torch.from_numpy(arr)
        attn[i, :len(arr)] = 1
    return ids, attn, torch.tensor(lens, dtype=torch.long)


def make_rows(arm, docs, others, tokens2d, j, bos_id, n_doc_tokens):
    """Token-id rows and the sequence indices holding doc-relative i in [I_LO, I_HI]."""
    read = np.arange(I_LO, I_HI + 1, dtype=np.int64)
    if arm == "position_ids_offset":
        rows = [np.concatenate([[bos_id], tokens2d[d, :n_doc_tokens].astype(np.int64)])
                for d in docs]
        return rows, 1 + read
    rows = [np.concatenate([[bos_id], tokens2d[o, :j].astype(np.int64),
                            tokens2d[d, :n_doc_tokens].astype(np.int64)])
            for d, o in zip(docs, others)]
    return rows, 1 + j + read


def fe_slope_shifts(C, shifts):
    """beta_abs with (doc, i) fixed effects, plus the per-document bootstrap ingredients.

    ``C`` is [n_shifts, n_docs, n_i]. Demeaning c within each (doc, i) cell and demeaning the
    shift grid (identical in every cell) reduces the fixed-effect slope to
    ``sum_d num_d / (n_docs * n_i * sum_j jdm^2)``, so a cluster bootstrap over documents is a
    resampled mean of ``num_d``.
    """
    C = np.asarray(C, dtype=np.float64)
    j = np.asarray(shifts, dtype=np.float64)
    jdm = j - j.mean()
    cdm = C - C.mean(axis=0, keepdims=True)
    num = np.einsum("jdi,j->d", cdm, jdm)                  # [n_docs]
    sjj = float((jdm ** 2).sum())
    den_per_doc = C.shape[2] * sjj
    if sjj == 0 or den_per_doc == 0:
        return float("nan"), num, 0.0
    return float(num.mean() / den_per_doc), num, den_per_doc


def fe_slope_positions(C2, ii):
    """OLS slope of c on DOC-RELATIVE index i with a document fixed effect.

    ``C2`` is [n_docs, n_i] at a single shift. This is the fresh-forward analogue of the
    cache arm's beta_rel, and it exists because the registered beta_abs/beta_rel ratio is a
    cross-representation comparison: beta_abs reads the raw residual stream while beta_rel
    reads SAE reconstructions of cache rows. This slope is measured on the SAME tensors as
    beta_abs, in the same units, under the same fixed-effect design, so the matched-units
    ratio it supports is interpretable in a way the registered one cannot be. It is a
    DIAGNOSTIC: the frozen call uses the registered ratio.

    Same algebraic reduction as ``fe_slope_shifts`` -- the document bootstrap is a resampled
    mean of the per-document numerators, and it can reuse beta_abs's own document draws
    because both statistics come from the same forwards.
    """
    C2 = np.asarray(C2, dtype=np.float64)
    x = np.asarray(ii, dtype=np.float64)
    xdm = x - x.mean()
    cdm = C2 - C2.mean(axis=1, keepdims=True)
    num = cdm @ xdm
    sxx = float((xdm ** 2).sum())
    if sxx == 0:
        return float("nan"), num, 0.0
    return float(num.mean() / sxx), num, sxx


def boot_ci(v):
    v = np.asarray(v, dtype=np.float64)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return dict(lo=None, hi=None, mean=None, n=0, excludes_zero=False, sign=0)
    lo, hi = (float(x) for x in np.percentile(v, [CI_LO, CI_HI]))
    ex = bool(lo > 0 or hi < 0)
    return dict(lo=lo, hi=hi, mean=float(v.mean()), n=int(v.size),
                excludes_zero=ex, sign=int(np.sign(v.mean())) if ex else 0)


# ===========================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="cache/sparse_big")
    ap.add_argument("--l0", default="big")
    ap.add_argument("--out", default="results/phase3_position_causal.json")
    ap.add_argument("--data-out", default="results/phase3_position_causal_data.npz")
    ap.add_argument("--layers", type=int, nargs="+", default=LAYERS)
    ap.add_argument("--shifts", type=int, nargs="+", default=SHIFTS)
    ap.add_argument("--n-docs", type=int, default=N_DOCS)
    ap.add_argument("--doc-tokens", type=int, default=DOC_TOKENS)
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    ap.add_argument("--n-null", type=int, default=N_NULL)
    ap.add_argument("--n-power", type=int, default=N_POWER)
    ap.add_argument("--batch-size", type=int, default=rig.BATCH_SIZE)
    ap.add_argument("--cache-rows", type=int, default=0,
                    help="truncate the CSR scan to the first N cache rows (smoke only; 0 = all)")
    ap.add_argument("--smoke", action="store_true",
                    help="annotate the output as a scaled-down instrument check")
    a = ap.parse_args()
    t_start = time.time()
    layers = sorted(set(int(x) for x in a.layers))
    shifts = sorted(set(int(x) for x in a.shifts))
    smoke = bool(a.smoke or a.cache_rows or a.n_docs != N_DOCS or a.n_boot != N_BOOT
                 or a.n_null != N_NULL or a.n_power != N_POWER or shifts != SHIFTS
                 or a.doc_tokens != DOC_TOKENS)

    # ---- R1: declaration stub to disk BEFORE any cache read or forward pass ----
    declared = dict(
        script="scripts/38_position_causal.py",
        prereg="docs/phase3_prereg_2026-08-21.md section 3.2",
        design="docs/next_avenues_2026-08-21.md avenue 4",
        status="DECLARATION STUB -- rules frozen, no results yet",
        declared_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
        smoke=smoke,
        question=("Is the position coordinate a RoPE-only model writes into resid_post "
                  "ABSOLUTE sequence position or DOCUMENT-RELATIVE index? The cache confounds "
                  "them perfectly (every cached document starts at absolute position 1), so "
                  "two causal arms move absolute position while holding the document fixed."),
        config=dict(root=a.root, l0=a.l0, layers=layers, shifts=shifts,
                    n_docs=int(a.n_docs), doc_tokens=int(a.doc_tokens),
                    n_boot=int(a.n_boot), n_null=int(a.n_null), n_power=int(a.n_power),
                    batch_size=int(a.batch_size), cache_rows=int(a.cache_rows),
                    device=DEVICE, data_out=str(a.data_out)),
        declared_rules=declared_rules(a, layers, shifts),
    )
    outp = Path(a.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(declared, indent=2))
    print(f"declaration stub -> {outp}", flush=True)
    if smoke:
        print("SMOKE RUN -- scaled-down instrument check, not the registered measurement",
              flush=True)

    # =======================================================================
    # cache arm
    # =======================================================================
    tok, cnt, type_ids = build_pool(a.root)
    n_types = len(type_ids)
    print(f"\ncache arm: {n_types} token types with count >= {COUNT_FLOOR} "
          f"({int(cnt[type_ids].sum()):,} of {tok.size:,} cache rows)", flush=True)
    brow, n_rows = acc_rows(tok, type_ids, a.cache_rows)
    n_acc = n_types * N_BINS
    bidx = np.arange(1, N_BINS + 1, dtype=np.float64)          # bin index b = 1..8
    bdm = bidx - bidx.mean()                                   # (b - 4.5)
    plant_shape = bdm / 3.5                                    # (b - 4.5) / 3.5, in [-1, 1]

    # per-type OLS numerators in MATCHED (un-normalised) units, carried out of the cache-arm
    # loop so the ratio bootstrap resamples exactly the quantity beta_rel was estimated from
    rel_boot_parts = {}
    cache_arm, u_dirs, processed = {}, {}, {}
    for L in layers:
        t_l = time.time()
        print(f"  streaming layer {L} ({n_rows:,} rows)", flush=True)
        acc, counts, d_sae, used, meta = bin_means(a.root, L, brow, n_rows, n_acc)
        sae = load_sae(L, a.l0, device="cpu", token=os.environ.get("HF_TOKEN"))
        W = get_decoder(sae)
        del sae
        R = project(acc, W)                                    # [n_acc, 1152] fp32
        del acc, W
        empty = int((counts == 0).sum())
        if empty:
            print(f"    WARNING: {empty} of {n_acc} (type, bin) cells are empty", flush=True)

        R3 = R.reshape(n_types, N_BINS, D_MODEL).astype(np.float64)
        Ac3 = R3 - R3.mean(axis=1, keepdims=True)              # within-type centring
        norms = np.linalg.norm(Ac3, axis=2)
        An3 = Ac3 / np.maximum(norms, 1e-12)[:, :, None]       # per-row L2
        An2 = An3.reshape(n_acc, D_MODEL)

        # ---- u_L: top right-singular vector of the processed matrix -------
        _, s, Vt = np.linalg.svd(An2, full_matrices=False)
        uL = _unit(Vt[0])
        sc_n = An3 @ uL
        if _pearson(sc_n.ravel(), np.tile(bdm, n_types)) < 0:
            uL, sc_n = -uL, -sc_n
        ev_frac = float(s[0] ** 2 / (s ** 2).sum())
        sc_c = Ac3 @ uL                                        # un-normalised, matched units

        beta_rel_m, num_m, den_m = fe_slope_bins(sc_c, bdm)
        beta_rel_n, num_n, den_n = fe_slope_bins(sc_n, bdm)
        rng_bt = np.random.default_rng(SEED_BOOT_TYPES + L)
        bt = rng_bt.integers(0, n_types, size=(int(a.n_boot), n_types))
        boot_rel_m = num_m[bt].mean(1) / den_m
        boot_rel_n = num_n[bt].mean(1) / den_n

        spear = np.array([_spearman(sc_n[t], bidx) for t in range(n_types)])
        bin_svd_real, bin_r2_real, uB = bin_stats(An3, bdm)
        sc_B = An3 @ uB

        # ---- null: within-type bin permutations ---------------------------
        rng_null = np.random.default_rng(SEED_NULL + L)
        An32 = An3.astype(np.float32)
        null_svd = np.empty(int(a.n_null), dtype=np.float64)
        null_r2 = np.empty(int(a.n_null), dtype=np.float64)
        for d in range(int(a.n_null)):
            pm = perm_within_type(rng_null, n_types)[0]
            null_svd[d], null_r2[d], _ = bin_stats(apply_perm(An32, pm), bdm)
        pct = float(NULL_PCT)
        thr_svd = float(np.percentile(null_svd, pct))
        thr_r2 = float(np.percentile(null_r2, pct))

        # ---- the registered statistic's degeneracy, demonstrated ----------
        lit_real = full_svd_stat(An3)
        rng_lit = np.random.default_rng(SEED_NULL + 7919 + L)
        lit_draws = [full_svd_stat(apply_perm(An3, perm_within_type(rng_lit, n_types)[0]))
                     for _ in range(N_LITERAL_DEMO)]
        lit_maxdiff = float(max(abs(x - lit_real) for x in lit_draws)) if lit_draws else None

        # ---- power: planted monotone coordinate on null data --------------
        med_norm = float(np.median(np.linalg.norm(An3, axis=2)))
        rng_pow = np.random.default_rng(SEED_POWER + L)
        power = []
        for alpha in POWER_ALPHAS:
            amp = alpha * med_norm
            det_svd = det_r2 = 0
            for _ in range(int(a.n_power)):
                pm = perm_within_type(rng_pow, n_types)[0]
                Ap = apply_perm(An32, pm).astype(np.float64)
                ur = _unit(rng_pow.normal(size=D_MODEL))
                Ap = Ap + amp * plant_shape[None, :, None] * ur[None, None, :]
                bs, br, _ = bin_stats(Ap, bdm)
                det_svd += int(bs > thr_svd)
                det_r2 += int(br > thr_r2)
            power.append(dict(alpha=float(alpha), amplitude=float(amp),
                              detection_bin_svd=det_svd / max(int(a.n_power), 1),
                              detection_bin_r2=det_r2 / max(int(a.n_power), 1)))
            print(f"    L{L} power alpha={alpha:.2f}: bin_svd "
                  f"{det_svd / max(int(a.n_power), 1):.2f}  bin_r2 "
                  f"{det_r2 / max(int(a.n_power), 1):.2f}", flush=True)

        u_dirs[L] = dict(u_L=uL, u_B=uB)
        # Ac2 is stored as well as An2: the VERDICT-BEARING beta_rel is the slope on the
        # centred un-normalised rows, and An2 alone would force a verifier to re-stream all
        # 4M cache rows just to recheck a slope.
        processed[L] = dict(An2=An2.astype(np.float32), counts=counts,
                            Ac2=Ac3.reshape(n_acc, D_MODEL).astype(np.float32))
        rel_boot_parts[L] = dict(num=num_m, den=den_m)
        cache_arm[str(L)] = dict(
            layer=int(L), n_types=int(n_types), n_bins=N_BINS, bin_width=float(BIN_WIDTH),
            bin_index="1..8 (1-based); bin b covers cache row-index p in "
                      "[1+31.75*(b-1), 1+31.75*b)",
            d_sae=int(d_sae), cache_rows_scanned=int(n_rows), cache_rows_used=int(used),
            cells_empty=empty,
            cell_counts=dict(min=int(counts.min()), median=float(np.median(counts)),
                             max=int(counts.max()), total=int(counts.sum())),
            u_L_explained_variance=ev_frac,
            u_B_alignment_with_u_L=float(abs(uL @ uB)),
            median_row_norm_after_L2=med_norm,
            beta_rel=dict(
                primary_units="centred, UN-normalised projected means (matched units)",
                per_bin=beta_rel_m, per_token=beta_rel_m / BIN_WIDTH,
                ci_per_bin=boot_ci(boot_rel_m),
                ci_per_token=boot_ci(boot_rel_m / BIN_WIDTH),
                literal_normalised_per_bin=beta_rel_n,
                literal_normalised_per_token=beta_rel_n / BIN_WIDTH,
                literal_ci_per_token=boot_ci(boot_rel_n / BIN_WIDTH),
                frac_boot_draws_crossing_zero=float(np.mean(
                    np.sign(boot_rel_m) != np.sign(beta_rel_m)))),
            within_type_spearman=dict(
                median=float(np.median(spear)), mean=float(spear.mean()),
                q25=float(np.percentile(spear, 25)), q75=float(np.percentile(spear, 75)),
                frac_positive=float(np.mean(spear > 0))),
            null=dict(
                n_draws=int(a.n_null), percentile=pct, smoke=smoke,
                percentile_note=("with fewer than ~1,000 draws the 99.9th percentile is the "
                                 "sample maximum and is a smoke artefact, not a threshold"
                                 if int(a.n_null) < 1000 else "as registered"),
                bin_svd=dict(real=bin_svd_real, threshold=thr_svd,
                             null_mean=float(null_svd.mean()),
                             null_p50=float(np.percentile(null_svd, 50)),
                             null_p99=float(np.percentile(null_svd, 99)),
                             null_max=float(null_svd.max()),
                             exceeds=bool(bin_svd_real > thr_svd),
                             p_emp=float((1 + int((null_svd >= bin_svd_real).sum()))
                                         / (1 + int(a.n_null)))),
                bin_r2=dict(real=bin_r2_real, threshold=thr_r2,
                            null_mean=float(null_r2.mean()),
                            null_p99=float(np.percentile(null_r2, 99)),
                            null_max=float(null_r2.max()),
                            exceeds=bool(bin_r2_real > thr_r2),
                            p_emp=float((1 + int((null_r2 >= bin_r2_real).sum()))
                                        / (1 + int(a.n_null)))),
                registered_statistic_degenerate=dict(
                    real=lit_real, draws=[float(x) for x in lit_draws],
                    max_abs_diff_from_real=lit_maxdiff,
                    finding=("the registered null statistic is invariant under the "
                             "registered permutation (row permutations preserve singular "
                             "values); measured max|null - real| is at float64 round-off. "
                             "It cannot reject at any threshold, so bin_svd / bin_r2 stand "
                             "in its place. NEITHER IS VERDICT-BEARING"))),
            power=dict(amplitudes_alpha=POWER_ALPHAS, n_draws=int(a.n_power),
                       median_row_norm=med_norm, curve=power,
                       alpha_at_80pct_bin_svd=next(
                           (p["alpha"] for p in power if p["detection_bin_svd"] >= 0.8), None),
                       alpha_at_80pct_bin_r2=next(
                           (p["alpha"] for p in power if p["detection_bin_r2"] >= 0.8), None)),
            bin_profile=dict(
                centred_units=[float(x) for x in sc_c.mean(0)],
                normalised_units=[float(x) for x in sc_n.mean(0)],
                u_B_units=[float(x) for x in sc_B.mean(0)],
                spearman_with_bin=_spearman(sc_c.mean(0), bidx)),
            seconds=float(time.time() - t_l))
        print(f"  L{L}: u_L expl.var {ev_frac:.4f}  |u_L.u_B| "
              f"{abs(uL @ uB):.4f}  beta_rel/token {beta_rel_m / BIN_WIDTH:.6g}  "
              f"bin_svd {bin_svd_real:.4f} vs thr {thr_svd:.4f}  "
              f"median within-type Spearman {np.median(spear):.3f}  "
              f"({time.time() - t_l:.0f}s)", flush=True)
    del brow, tok, cnt

    # =======================================================================
    # causal arms
    # =======================================================================
    tokens2d = rig.load_tokens2d(a.root)
    rng_docs = np.random.default_rng(SEED_DOCS)
    docs = np.sort(rng_docs.choice(rig.N_DOCS, size=int(a.n_docs), replace=False))
    rng_other = np.random.default_rng(SEED_OTHER)
    others = rng_other.integers(0, rig.N_DOCS, size=(int(a.n_docs), len(shifts)))
    clash = others == docs[:, None]
    others[clash] = (others[clash] + 1) % rig.N_DOCS            # donor must differ from doc

    hf = os.environ.get("HF_TOKEN")
    model, tokzr = load_model(token=hf)
    bos = tokzr.bos_token_id
    pad = tokzr.pad_token_id if tokzr.pad_token_id is not None else tokzr.eos_token_id
    print(f"\ncausal arms: model on {DEVICE}; bos={bos} pad={pad}; "
          f"{len(docs)} docs x {len(shifts)} shifts x {len(ARMS)} arms", flush=True)

    U = {L: torch.from_numpy(np.stack([u_dirs[L]["u_L"], u_dirs[L]["u_B"]], axis=1)
                             .astype(np.float32)).to(DEVICE) for L in layers}
    n_i = I_HI - I_LO + 1

    # ---- forward checks, written to disk BEFORE they are asserted ---------
    probe = docs[:min(8, len(docs))]
    prows, pread = make_rows("position_ids_offset", probe, others[:len(probe), 0], tokens2d,
                             0, bos, int(a.doc_tokens))
    pid_ids, pid_attn, _ = build_seq_batch(prows, pad)
    pid_ids, pid_attn = pid_ids.to(DEVICE), pid_attn.to(DEVICE)
    seq_len = pid_ids.shape[1]

    def _probe(j):
        pos = (torch.arange(seq_len, dtype=torch.long, device=DEVICE) + int(j))
        pos = pos.unsqueeze(0).repeat(pid_ids.shape[0], 1)
        r = rig.forward_capture(model, pid_ids, pid_attn, layers, position_ids=pos)
        return {L: r[L].float().cpu() for L in layers}

    r_def = rig.forward_capture(model, pid_ids, pid_attn, layers, position_ids=None)
    r_def = {L: r_def[L].float().cpu() for L in layers}
    r_j0, r_js = _probe(0), _probe(VERIFY_SHIFT)
    d_default = max(float((r_j0[L] - r_def[L]).abs().max()) for L in layers)
    d_shift = max(float((r_js[L] - r_j0[L]).abs().max()) for L in layers)
    checks = dict(
        j0_vs_default_max_abs_diff=d_default, j0_vs_default_threshold=MAX_ABS_DIFF_DEFAULT,
        j0_reproduces_default=bool(d_default < MAX_ABS_DIFF_DEFAULT),
        j16_vs_j0_max_abs_diff=d_shift, j16_vs_j0_threshold=MIN_ABS_DIFF_SHIFT,
        position_ids_changes_output=bool(d_shift > MIN_ABS_DIFF_SHIFT),
        n_probe_rows=int(pid_ids.shape[0]), seq_len=int(seq_len), verify_shift=VERIFY_SHIFT,
        note=("arm (a) at j=0 with explicit position_ids must be the default forward, and "
              "j=16 must not be -- otherwise position_ids is either ignored or is doing "
              "something other than shifting RoPE"))
    del r_def, r_j0, r_js
    # The cache arm is already paid for; carry it into the intermediate write so that a failed
    # forward check leaves a diagnosable file rather than destroying the streaming passes.
    outp.write_text(json.dumps(dict(declared, status="CACHE ARM + FORWARD CHECKS ONLY",
                                    cache_arm=cache_arm, forward_checks=checks),
                               indent=2, default=float))
    print(f"forward checks: j0 vs default {d_default:.3e} (< {MAX_ABS_DIFF_DEFAULT}), "
          f"j16 vs j0 {d_shift:.3e} (> {MIN_ABS_DIFF_SHIFT})", flush=True)
    assert checks["j0_reproduces_default"], (
        f"arm (a) j=0 does not reproduce the default forward: {d_default}")
    assert checks["position_ids_changes_output"], (
        f"position_ids offset j={VERIFY_SHIFT} did not change the output: {d_shift}")

    # ---- the 2 x n_shifts forward sweep -----------------------------------
    # C[arm][layer] -> [n_shifts, n_docs, n_i, 2]; last axis = (u_L, u_B)
    C = {arm: {L: np.zeros((len(shifts), len(docs), n_i, 2), dtype=np.float32)
               for L in layers} for arm in ARMS}
    fwd = dict(batches=0, seconds=0.0, tokens=0, sequences=0)
    for arm in ARMS:
        for jk, j in enumerate(shifts):
            rows, read_idx = make_rows(arm, docs, others[:, jk], tokens2d, j, bos,
                                       int(a.doc_tokens))
            ridx = torch.from_numpy(read_idx).to(DEVICE)
            t_cell = time.time()
            for lo, chunk in rig.iter_batches(rows, a.batch_size):
                ids, attn, _ = build_seq_batch(chunk, pad)
                ids, attn = ids.to(DEVICE), attn.to(DEVICE)
                pos = None
                if arm == "position_ids_offset":
                    pos = (torch.arange(ids.shape[1], dtype=torch.long, device=DEVICE) + j)
                    pos = pos.unsqueeze(0).repeat(ids.shape[0], 1)
                t0 = time.time()
                resid = rig.forward_capture(model, ids, attn, layers, position_ids=pos)
                for L in layers:
                    h = resid[L].index_select(1, ridx).float()      # [b, n_i, 1152]
                    C[arm][L][jk, lo:lo + len(chunk)] = (h @ U[L]).cpu().numpy()
                    del h
                del resid
                fwd["batches"] += 1
                fwd["seconds"] += time.time() - t0
                fwd["tokens"] += int(attn.sum())
                fwd["sequences"] += len(chunk)
            print(f"  {arm} j={j:<4d} len={len(rows[0]):<4d} "
                  f"{time.time() - t_cell:5.1f}s", flush=True)

    # =======================================================================
    # beta_abs, ratios, call
    # =======================================================================
    rng_bd = np.random.default_rng(SEED_BOOT_DOCS)
    bd = rng_bd.integers(0, len(docs), size=(int(a.n_boot), len(docs)))
    causal, boot_abs = {}, {}
    for L in layers:
        arms_out = {}
        for arm in ARMS:
            per_dir = {}
            for di, dname in enumerate(["u_L", "u_B"]):
                slope, num, den = fe_slope_shifts(C[arm][L][:, :, :, di], shifts)
                boot = num[bd].mean(1) / den if den else np.full(int(a.n_boot), np.nan)
                per_dir[dname] = dict(beta_abs=slope, ci=boot_ci(boot),
                                      per_j_mean={str(j): float(
                                          C[arm][L][k, :, :, di].mean())
                                          for k, j in enumerate(shifts)})
                if dname == "u_L":
                    boot_abs[(L, arm)] = boot
            arms_out[arm] = per_dir
        causal[str(L)] = arms_out

    # ---- ratio + the frozen three-way call --------------------------------
    # The beta_rel type bootstrap is redrawn from the SAME seed as the cache-arm pass, so the
    # denominator draws here are the identical draws whose CI the cache arm reported.
    calls, ratios = {}, {}
    for L in layers:
        rng_bt = np.random.default_rng(SEED_BOOT_TYPES + L)
        bt = rng_bt.integers(0, n_types, size=(int(a.n_boot), n_types))
        num_m, den_m = rel_boot_parts[L]["num"], rel_boot_parts[L]["den"]
        boot_rel_pt = (num_m[bt].mean(1) / den_m) / BIN_WIDTH
        rel_pt = cache_arm[str(L)]["beta_rel"]["per_token"]
        ratios[str(L)] = {}
        for arm in ARMS:
            ba = causal[str(L)][arm]["u_L"]["beta_abs"]
            bb = boot_abs[(L, arm)]
            with np.errstate(divide="ignore", invalid="ignore"):
                rboot = bb / boot_rel_pt
            ratios[str(L)][arm] = dict(
                ratio=float(ba / rel_pt) if rel_pt else None, ci=boot_ci(rboot),
                pairing="independent draws: documents for the numerator, types for the "
                        "denominator",
                denominator_frac_crossing_zero=float(np.mean(
                    np.sign(boot_rel_pt) != np.sign(rel_pt))) if rel_pt else None,
                unstable=bool(boot_ci(boot_rel_pt)["excludes_zero"] is False),
                unit_caveat="numerator reads raw resid; denominator reads SAE reconstructions")
        cis = [causal[str(L)][arm]["u_L"]["ci"] for arm in ARMS]
        both_exclude = all(c["excludes_zero"] for c in cis)
        same_sign = both_exclude and len({c["sign"] for c in cis}) == 1
        both_contain = all(not c["excludes_zero"] for c in cis)
        rel_ci = cache_arm[str(L)]["beta_rel"]["ci_per_token"]
        ratio_lo = ratios[str(L)][ARMS[0]]["ci"]["lo"]
        if same_sign and ratio_lo is not None and ratio_lo > RATIO_FLOOR:
            call = "ABSOLUTE"
            why = (f"both beta_abs CIs exclude 0 with sign {cis[0]['sign']} and the arm-(a) "
                   f"ratio lower CI {ratio_lo:.4g} > {RATIO_FLOOR}")
        elif both_contain and rel_ci["excludes_zero"]:
            call = "DOCUMENT-RELATIVE"
            why = "both beta_abs CIs contain 0 while the beta_rel CI excludes 0"
        else:
            call = "UNIDENTIFIED"
            why = ("neither the ABSOLUTE conjunction nor the DOCUMENT-RELATIVE conjunction "
                   f"holds (both_exclude={both_exclude}, same_sign={same_sign}, "
                   f"both_contain={both_contain}, beta_rel_excludes_zero="
                   f"{rel_ci['excludes_zero']}, ratio_lo={ratio_lo})")
        calls[str(L)] = dict(call=call, reason=why, primary=bool(L == layers[0]),
                             both_arms_exclude_zero=both_exclude, same_sign=same_sign,
                             both_arms_contain_zero=both_contain,
                             beta_rel_ci_excludes_zero=rel_ci["excludes_zero"],
                             ratio_lower_ci_arm_a=ratio_lo, ratio_floor=RATIO_FLOOR)
        print(f"  L{L}: {call} -- {why}", flush=True)

    # =======================================================================
    # consistency checks
    # =======================================================================
    ii = np.arange(I_LO, I_HI + 1)
    consistency = {}
    for L in layers:
        j0 = shifts.index(0) if 0 in shifts else None
        prof = {}
        if j0 is not None:
            fresh_boot = {}
            for arm in ARMS:
                p = C[arm][L][j0, :, :, 0].mean(0)              # mean over docs, u_L
                sl, num_p, den_p = fe_slope_positions(C[arm][L][j0, :, :, 0], ii)
                bp = num_p[bd].mean(1) / den_p if den_p else np.full(int(a.n_boot), np.nan)
                fresh_boot[arm] = bp
                prof[arm] = dict(c_of_i=[float(x) for x in p],
                                 spearman_with_i=_spearman(p, ii),
                                 pearson_with_i=_pearson(p, ii),
                                 beta_rel_fresh_per_token=sl, beta_rel_fresh_ci=boot_ci(bp),
                                 beta_rel_fresh_note=(
                                     "OLS slope of c on doc-relative i at j=0 with document "
                                     "fixed effects, in RAW-RESIDUAL units -- the fresh-"
                                     "forward analogue of the cache arm's beta_rel"))
            # matched-units ratio: numerator and denominator from the same tensors, the same
            # units and the same document draws. Diagnostic; the frozen call uses the
            # registered cross-representation ratio.
            prof["matched_units_ratio"] = {}
            den_arm = ARMS[0]
            for arm in ARMS:
                slope_d = prof[den_arm]["beta_rel_fresh_per_token"]
                with np.errstate(divide="ignore", invalid="ignore"):
                    rb = boot_abs[(L, arm)] / fresh_boot[den_arm]
                prof["matched_units_ratio"][arm] = dict(
                    ratio=(float(causal[str(L)][arm]["u_L"]["beta_abs"] / slope_d)
                           if slope_d not in (0.0,) and np.isfinite(slope_d) else None),
                    ci=boot_ci(rb),
                    denominator=f"beta_rel_fresh of arm '{den_arm}' at j=0 (raw-resid units)",
                    pairing="the SAME document bootstrap draws in numerator and denominator",
                    verdict_bearing=False)
            a0 = C[ARMS[0]][L][j0, :, :, 0]
            b0 = C[ARMS[1]][L][j0, :, :, 0]
            prof["arm_a_vs_arm_b_at_j0_max_abs_diff"] = float(np.abs(a0 - b0).max())
            prof["arm_a_vs_arm_b_at_j0_note"] = (
                "at j=0 the two arms are byte-identical inputs (no prefix, default RoPE); "
                "this is a free rig check and should be ~0")
            cache_prof = cache_arm[str(L)]["bin_profile"]["centred_units"]
            b_of_i = bin_of_p(ii)                               # 0-based bins touched by i
            keep = sorted(set(int(x) for x in b_of_i))
            fresh = np.asarray(prof[ARMS[0]]["c_of_i"], dtype=np.float64)
            fresh_binned = [float(fresh[b_of_i == b].mean()) for b in keep]
            prof["cache_bridge"] = dict(
                bins_covered=[b + 1 for b in keep],
                fresh_j0_binned=fresh_binned,
                cache_bin_profile=[cache_prof[b] for b in keep],
                spearman=(_spearman(fresh_binned, [cache_prof[b] for b in keep])
                          if len(keep) >= 3 else None),
                note=("a rank agreement between two representations, NOT an equality check, "
                      "and weak by construction on two counts: (i) the fresh forwards read "
                      "the RAW residual stream while the cache reads SAE reconstructions; "
                      "(ii) the cache profile is WITHIN-TYPE CENTRED -- token identity has "
                      "been removed -- while c(i) is not, so at j=0 the fresh profile "
                      "carries the token-identity variance of whatever tokens happen to sit "
                      "at each i, averaged over documents only. beta_abs is immune to this "
                      "(its (doc, i) fixed effects hold the token fixed across shifts); this "
                      f"descriptive profile is not. Only bins touched by i in {I_LO}..{I_HI} "
                      f"are comparable, so the rank correlation is over {len(keep)} points"))
        consistency[str(L)] = prof

    # =======================================================================
    # runtime
    # =======================================================================
    tok_per_s = fwd["tokens"] / max(fwd["seconds"], 1e-9)
    full_tokens = sum(
        N_DOCS * (DOC_TOKENS + 1) for _ in SHIFTS) + sum(
        N_DOCS * (DOC_TOKENS + 1 + j) for j in SHIFTS)
    runtime = dict(
        forward=dict(batches=fwd["batches"], sequences=fwd["sequences"],
                     tokens=fwd["tokens"], seconds=fwd["seconds"],
                     tokens_per_sec=float(tok_per_s),
                     sec_per_batch=float(fwd["seconds"] / max(fwd["batches"], 1)),
                     ms_per_token=float(1000.0 / max(tok_per_s, 1e-9))),
        full_run_forward_estimate=dict(
            sequences=int(2 * N_DOCS * len(SHIFTS)), tokens=int(full_tokens),
            seconds=float(full_tokens / max(tok_per_s, 1e-9)),
            minutes=float(full_tokens / max(tok_per_s, 1e-9) / 60.0),
            note="this run's measured throughput applied to the registered 300 x 5 x 2 grid"),
        cache_arm_seconds={str(L): cache_arm[str(L)]["seconds"] for L in layers},
        total_seconds=float(time.time() - t_start))
    print(f"\nforward: {fwd['seconds']:.1f}s over {fwd['batches']} batches "
          f"({tok_per_s:,.0f} tok/s, {1000.0 / max(tok_per_s, 1e-9):.3f} ms/token); "
          f"full-grid forward estimate {full_tokens / max(tok_per_s, 1e-9) / 60:.1f} min",
          flush=True)

    # =======================================================================
    # write
    # =======================================================================
    npz = dict(docs=docs.astype(np.int64), others=others.astype(np.int64),
               shifts=np.asarray(shifts, dtype=np.int64),
               layers=np.asarray(layers, dtype=np.int64),
               doc_relative_i=ii.astype(np.int64),
               arms=np.asarray(ARMS), directions=np.asarray(["u_L", "u_B"]),
               type_ids=type_ids.astype(np.int64))
    for L in layers:
        npz[f"u_L_layer{L}"] = u_dirs[L]["u_L"].astype(np.float32)
        npz[f"u_B_layer{L}"] = u_dirs[L]["u_B"].astype(np.float32)
        npz[f"processed_rows_layer{L}"] = processed[L]["An2"]        # centred + per-row L2
        npz[f"centred_rows_layer{L}"] = processed[L]["Ac2"]          # centred, UN-normalised
        npz[f"cell_counts_layer{L}"] = processed[L]["counts"]
        for arm in ARMS:
            npz[f"c_{arm}_layer{L}"] = C[arm][L]
    dp = Path(a.data_out)
    dp.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(dp, **npz)

    out = dict(declared)
    out.update(
        status="COMPLETE",
        finished_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
        cache_arm=cache_arm,
        forward_checks=checks,
        causal_arms=causal,
        ratios=ratios,
        calls=calls,
        primary_call=calls[str(layers[0])]["call"],
        consistency=consistency,
        runtime=runtime,
        runtime_seconds=float(time.time() - t_start),
        data_npz=str(dp),
    )
    outp.write_text(json.dumps(out, indent=2, default=float))
    print(f"\nprimary (L{layers[0]}) call: {calls[str(layers[0])]['call']}", flush=True)
    print(f"data -> {dp}", flush=True)
    print(f"wrote {outp} ({time.time() - t_start:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
