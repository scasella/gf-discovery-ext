"""Stage 2, avenue 2b -- the month ring's context budget, on NATURAL contexts.

Pre-registered in docs/phase3_prereg_2026-08-21.md section 8 (Addendum A, frozen
2026-08-21 before any 37b run); everything not restated there is section 3.1 unchanged.
Writes results/phase3_context_ladder_natural.json plus a companion _means.npz.

WHY THIS SCRIPT EXISTS. scripts/37 holds the context FAMILY fixed (FineWeb documents
containing no calendar token) and varies only its LENGTH. Its own pre-run rig check --
n=150, L6, c=128, SAE arm, synthetic month-free prefixes -- returned gap_ratio 2.99
against the registered anchor 1.65876, while the same forward rig reproduced cached CSR
rows bit-for-bit (60/60 identical support). So the section-3.1 disjunction "either the rig
changed or the model does not do this" was mis-specified: a third quantity changed, the
context DISTRIBUTION. A month token preceded by 128 tokens of text that mentions no month
is not the population the cached pooled readout averages over. 37 still runs as frozen and
its gate outcome is the registered measurement; this script asks the same ladder question
of the population the anchor actually came from.

THE LADDER. For each of the 13 frozen members, the contexts at rung c are the c tokens
immediately preceding a NATURAL occurrence of that member in the cache. The occurrence pool
is every cached row r with tok[r] = m and position p = r mod 255 >= 128, so every rung
c <= 128 is available from the SAME occurrence; n = min(300, pool) occurrences are drawn per
member with seed 0 and reused at every rung. The context at c = 128 is therefore the last
128 tokens BEFORE the occurrence -- NOT the start of the document, except for the subset of
occurrences that happen to sit at p = 128 exactly, for which the two coincide.

WHAT IS DELIBERATELY NOT CONTROLLED. Natural contexts are not month-filtered: real text
around " March" contains other month tokens, and removing them would rebuild exactly the
artificial population this script exists to escape. The contamination rate (fraction of
contexts containing any of the 13 member ids) is measured and reported per rung instead of
being engineered to zero.

WHAT THE RIG VALIDATION IS AND IS NOT. Section 8 widens the c=128 tolerance to
|gap_ratio - 1.659| <= 0.3 (p_closed < 0.01 unchanged) because n <= 300 late-position
occurrences is a subsample of the 374-777 occurrences the pooled anchor averages, and both
member sets must pass. It does NOT ask for per-occurrence agreement with the cached CSR
rows, and cannot: truncating a document to its last 128 tokens changes the residual at the
target, and the target is re-placed at model position c+1 rather than its natural p+1, so
RoPE differs too. Per-occurrence code cosine and support overlap against the cached rows are
therefore reported as a TRUNCATION DIAGNOSTIC, not as a gate -- with the p == c subgroup
(exact natural prefix, matching RoPE) split out, where near-exact reproduction IS expected.

WHAT IS COPIED AND FROM WHERE. Nothing statistical is invented here, and nothing is
imported from scripts/37 (scripts/ files are numbered, not importable, and a frozen
instrument must not be able to change under a script that already reported with it).
gap_null / gap_p / screen are copied verbatim from scripts/22_token_type_geometry.py
(L364-376, L379), member_hash from L561; block_permutations and its rank helpers from
scripts/29_keys_blockperm.py (L86-117); the angle convention (pca_2d -> fit_circle ->
angles(xy, cx, cy)) is scripts/22's score_ordering. score_cell / decide / the ordinal
control / the certificate are copied from scripts/37_context_ladder.py with the two
adaptations section 8 forces: the occurrence-based sampler, and ragged per-member n.

ORDERING RUNS ON STRICT-12 ONLY, for scripts/37's mechanical reason: cyclic_order_score's
symmetry basis needs DISTINCT calendar positions and " Feb" ties with " February" at
position 1. Closure is reported for both sets; ordering only for strict-12, where the
block-permutation null degenerates to uniform permutations.
"""
import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import gfdisc.rig as rig                                     # shared Stage-2 rig, unedited
from gfdisc.cache import SparseReader
from gfdisc.env import DEVICE
from gfdisc.fits import angles, circularity, degenerate, fit_circle, pca_2d, cyclic_order_score
from gfdisc.model import load_model
from gfdisc.saes import load_sae


# ===========================================================================
# frozen constants -- prereg sections 0, 3.1 and 8
# ===========================================================================
#: The 13-member set, ascending token id. INDEX INTO THIS LIST IS ``member_index``.
MEMBER_IDS = [3353, 5142, 5261, 5365, 5505, 5656, 5729, 5857, 6027, 6198, 6243, 6384, 7250]
FEB_ABBREV_ID = 5656                                # " Feb"; dropped to get strict-12
STRICT12_IDS = [t for t in MEMBER_IDS if t != FEB_ABBREV_ID]
MEMBER_HASH_EXPECTED = "7d1d75a303d6"               # md5[:12] of the sorted POOL INDICES

MONTHS = ["January", "February", "March", "April", "May", "June",
          "July", "August", "September", "October", "November", "December"]
#: token id -> calendar position 0..11. " Feb" shares February's position (see docstring).
CALENDAR_POS = {6027: 0, 7250: 1, 5261: 2, 5365: 3, 3353: 4, 5142: 5,
                5505: 6, 5729: 7, 5857: 8, 6198: 9, 6243: 10, 6384: 11, FEB_ABBREV_ID: 1}
FULL_K = 12

COUNT_FLOOR = 100                                   # scripts/22 build_pool floor
RUNGS = [0, 1, 2, 4, 8, 16, 32, 64, 128]
LAYERS = [6, 7]
N_CTX = 300                                         # section 8: n = min(300, pool)
N_PERM_SCREEN = 20_000                              # scripts/22 N_PERM_CLUSTER
N_PERM_ORDER = 2_000                                # prereg 3.1 ordering null
N_BLOCK_PERM = 20_000                               # scripts/29 N_NULL

#: Occurrence pool rule, section 8: rows r with tok[r] = m AND (r mod 255) >= 128, so the
#: same occurrence supplies every rung up to 128. Drawn once, with seed 0, per member.
P_FLOOR = 128
OCC_SEED = 0

RIG_REF_GAP_RATIO = 1.659                           # prereg 3.1 (exposure record: 1.65876)
RIG_GAP_TOL = 0.3                                   # section 8 widened from 0.2
RIG_MAX_P_CLOSED = 0.01
RIG_FAIL_VERDICT = "natural-context ladder not validated"   # section 8 wording, frozen
CERT_LAYER, CERT_RUNG, CERT_ARM = 6, 128, "sae"
CERT_N, CERT_RESAMPLES, CERT_MIN_PASS = 150, 200, 0.8
ORD_RUNGS = [0, 128]
ORD_N = 150
ORD_WORDS = [str(d) for d in range(10)] + [
    "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"]
ALPHA = 0.05

ARMS = ["sae", "raw"]
SETS = {"set13": MEMBER_IDS, "strict12": STRICT12_IDS}

#: Verbatim section 8 design text, carried into the declaration so the frozen rules travel
#: with the numbers (prereg R1).
ADDENDUM_A_TEXT = """\
37b design (everything else identical to 3.1):
- Contexts for member m at rung c = the c tokens immediately preceding a NATURAL occurrence of
  m in the cache. Occurrence pool: all rows r with tok[r]=m and position p=(r mod 255) >= 128
  (so every rung c<=128 is available from the same occurrence); n = min(300, pool size) per
  member drawn with seed 0, the SAME occurrences used at every rung. Contexts are not
  month-filtered (natural text may contain other month tokens; reported per rung as the
  fraction of contexts containing any member id).
- Rig validation: at c=128 the per-occurrence fresh codes must match the cached rows for
  those occurrences only up to truncation; the registered check is that the c=128 SAE-arm
  strict-12 and 13-set means close with |gap_ratio - 1.659| <= 0.3 and p_closed < 0.01
  at L6 (tolerance widened from 0.2 because n<=300 natural occurrences at p>=128 is a
  subsample of the 374-777 used by the pooled anchor). If this fails, verdict "natural-
  context ladder not validated".
- Decision rule as 3.1 (Branch A / Branch B / unresolved), SAE arm primary, raw co-reported.
- Synthetic-vs-natural contrast: the 37 and 37b c=128 cells are reported side by side as a
  registered secondary finding ("context-genre dependence of the mean-code ring")."""


# ===========================================================================
# copied instruments (verbatim; provenance in the docstring)
# ===========================================================================
_NULL = {}


def gap_null(k, n_perm, seed=0):
    """Copied verbatim from scripts/22_token_type_geometry.py L364-373."""
    key = (int(k), int(n_perm), int(seed))
    if key not in _NULL:
        rng = np.random.default_rng(seed)
        a = np.sort(rng.uniform(0, 2 * np.pi, size=(n_perm, k)), axis=1)
        _NULL[key] = np.diff(np.concatenate([a, a[:, :1] + 2 * np.pi], axis=1), axis=1).max(1)
    return _NULL[key]


def gap_p(k, max_gap_rad, n_perm, seed=0):
    """Copied verbatim from scripts/22_token_type_geometry.py L375-377."""
    g = gap_null(k, n_perm, seed)
    return float((1 + (g <= max_gap_rad).sum()) / (1 + n_perm))


def screen(M, n_perm, seed=0):
    """Copied verbatim from scripts/22_token_type_geometry.py L379-391."""
    M = np.asarray(M, dtype=np.float64)
    if len(M) < 3 or degenerate(M):
        return None
    xy, ratios, Vt = pca_2d(M)
    cx, cy, r, resid = fit_circle(xy)
    c = circularity(xy, cx, cy, r)
    return dict(k=int(len(M)), gap_ratio=c["gap_ratio"], max_gap_deg=c["max_gap_deg"],
                radius_ratio=c["radius_ratio"], circle_resid_over_r=float(resid),
                top2_var=float(ratios[:2].sum()), pc1_var=float(ratios[0]),
                p_closed=gap_p(len(M), np.radians(c["max_gap_deg"]), n_perm, seed))


def member_hash(members):
    """Copied verbatim from scripts/22_token_type_geometry.py L561-562."""
    return hashlib.md5(",".join(str(int(x)) for x in sorted(members)).encode()).hexdigest()[:12]


def _rank_of(values):
    """Copied verbatim from scripts/29_keys_blockperm.py L86-90."""
    order = np.argsort(np.asarray(values), kind="stable")
    r = np.empty(len(order), dtype=np.float64)
    r[order] = np.arange(len(order), dtype=np.float64)
    return r


def _rank_from_angles(ang, seed):
    """Copied verbatim from scripts/29_keys_blockperm.py L93-101."""
    ang = np.asarray(ang, dtype=float)
    n = len(ang)
    jitter = np.random.default_rng(seed).permutation(n)
    order = np.lexsort((jitter, ang))
    rank_of = np.empty(n, dtype=np.float64)
    rank_of[order] = np.arange(n)
    return rank_of


def _sym_basis(label_index, K):
    """Copied verbatim from scripts/29_keys_blockperm.py L104-106."""
    lab = np.asarray(label_index, dtype=int)
    return np.vstack([_rank_of((f * lab + s) % K) for f in (1, -1) for s in range(K)])


def block_permutations(blocks, n_draws, seed):
    """Copied verbatim from scripts/29_keys_blockperm.py L108-117."""
    rng = np.random.default_rng(seed)
    n = sum(len(b) for b in blocks)
    sig = np.tile(np.arange(n), (n_draws, 1))
    for b in blocks:
        b = np.asarray(b)
        pick = np.argsort(rng.random((n_draws, len(b))), axis=1)
        sig[:, b] = b[pick]
    return sig


# ===========================================================================
# derived statistics (copied from scripts/37_context_ladder.py)
# ===========================================================================
def cell_angles(M):
    """Angles of the member means in their own PCA plane, about the FITTED circle centre."""
    xy, _, _ = pca_2d(np.asarray(M, dtype=np.float64))
    cx, cy, r, _ = fit_circle(xy)
    return angles(xy, cx, cy), xy, (float(cx), float(cy), float(r))


def block_perm_order_p(ang, label_index, K, blocks, n_draws, seed):
    """Permutation p for the max-over-symmetries rho, permuting only WITHIN blocks."""
    rank_of = _rank_from_angles(ang, seed)
    n = len(rank_of)
    base = _sym_basis(label_index, K)
    denom = n * (n * n - 1)

    def rhos(sig):
        d = rank_of - base[:, sig]
        return 1.0 - 6.0 * (d * d).sum(-1) / denom

    rho_obs = float(rhos(np.arange(n)).max())
    sig = block_permutations(blocks, n_draws, seed + 31337)
    null = rhos(sig).max(0)
    return dict(rho=rho_obs, n_draws=int(n_draws),
                p=float((1 + int((null >= rho_obs).sum())) / (1 + n_draws)),
                null_mean=float(null.mean()), null_p95=float(np.percentile(null, 95)),
                blocks="single block of all 12 positions -> uniform permutations "
                       "(strict-12 has no lexical block structure to preserve)")


def mean_radius(M):
    """Mean in-plane distance of the member means from the fitted circle centre."""
    xy, _, _ = pca_2d(np.asarray(M, dtype=np.float64))
    cx, cy, r, _ = fit_circle(xy)
    return float(np.hypot(xy[:, 0] - cx, xy[:, 1] - cy).mean())


# ===========================================================================
# NEW: natural-occurrence sampling (section 8). Lives here, not in rig.py, because
# rig.py is frozen for scripts/37 and 38 and must not change under them.
# ===========================================================================
def occurrence_pool(tokens_flat, token_id, p_floor=P_FLOOR):
    """Flat cache rows r with ``tok[r] == token_id`` and ``r mod 255 >= p_floor``.

    Flat row index is document-relative by construction: r = 255*doc + p with p the
    DOCUMENT-RELATIVE index (BOS was dropped at cache build, prereg section 0), so p >= 128
    guarantees the 128 preceding tokens exist inside the same document and no context ever
    crosses a document boundary.
    """
    r = np.flatnonzero(np.asarray(tokens_flat) == int(token_id))
    return r[(r % rig.DOC_LEN) >= int(p_floor)].astype(np.int64)


def draw_occurrences(pool, n, seed=OCC_SEED):
    """``min(n, len(pool))`` occurrences drawn WITHOUT replacement, seed fixed by section 8.

    Without replacement because an occurrence is a distinct piece of evidence and drawing
    one twice would double-count a single context; seed 0 and no dependence on the rung,
    so the SAME occurrences are used at every rung (that is what makes the ladder a within-
    occurrence comparison rather than a comparison of nine different samples).
    """
    pool = np.asarray(pool, dtype=np.int64)
    k = int(min(int(n), pool.size))
    if k <= 0:
        return pool[:0]
    return np.sort(np.random.default_rng(int(seed)).choice(pool, size=k, replace=False))


def natural_contexts(tokens2d, occ_rows, c):
    """``tokens2d[doc, p-c:p]`` for each occurrence row -- the c tokens BEFORE the target."""
    c = int(c)
    out = []
    for r in np.asarray(occ_rows, dtype=np.int64):
        d, p = int(r) // rig.DOC_LEN, int(r) % rig.DOC_LEN
        if p - c < 0:
            raise RuntimeError(f"occurrence at p={p} cannot supply a context of length {c}")
        out.append(tokens2d[d, p - c:p].astype(np.int64))
    return out


def contamination(ctx_list, ids):
    """Fraction of contexts containing any of ``ids`` (section 8's reported quantity)."""
    if not ctx_list:
        return None
    want = np.asarray(sorted({int(x) for x in ids}), dtype=np.int64)
    hit = [bool(np.isin(c, want).any()) if len(c) else False for c in ctx_list]
    return float(np.mean(hit))


# ===========================================================================
# the forward ladder, over natural occurrences
# ===========================================================================
def run_ladder_natural(model, tok_bos, tok_pad, saes, tokens2d, targets, occ, rungs, layers,
                       batch_size, keep_rows_at, contam_ids, readers=None, diag_rung=None,
                       label=""):
    """Forward every (target, rung) cell over that target's drawn natural occurrences.

    Returns ``(means, meta, rows, timing, diag)`` where
      means[(layer, rung, arm)] -> float64 [n_targets, 1152], rows in ``targets`` order;
      meta[(target, rung)]      -> n / n_unique / contamination / doc counts;
      rows[(layer, target)]     -> float32 [n_occ, 1152] per-occurrence vectors (certificate
                                   arm only, only at ``keep_rows_at``), full multiplicity;
      timing[rung]              -> batches / seconds / tokens;
      diag[(layer, target)]     -> per-occurrence truncation diagnostic at ``diag_rung``.

    Both layers are captured in ONE forward per batch, so the L7 arm is free. ``dedupe`` is
    kept from scripts/37: at c=0 every context is the empty slice, so one forward answers
    the whole cell EXACTLY (reconstruction and the mean are both linear in the rows). The
    ``inverse`` map is retained here -- unlike in 37 -- because the truncation diagnostic
    must attribute a fresh code back to the specific occurrence whose cached row it faces.
    """
    means = {(l, c, a): np.zeros((len(targets), 1152), dtype=np.float64)
             for l in layers for c in rungs for a in ARMS}
    meta, rows, diag = {}, {}, {}
    timing = {c: dict(batches=0, seconds=0.0, tokens=0) for c in rungs}

    for c in rungs:
        for ti, target_id in enumerate(targets):
            occ_rows = np.asarray(occ[int(target_id)], dtype=np.int64)
            ctx = natural_contexts(tokens2d, occ_rows, c)
            uniq, weights, inverse = rig.dedupe(ctx)
            occ_of_uniq = [[] for _ in uniq]
            for j, u in enumerate(inverse):
                occ_of_uniq[int(u)].append(j)
            want_diag = (diag_rung is not None and c == diag_rung and readers)
            acc = {(l, a): np.zeros(1152, dtype=np.float64) for l in layers for a in ARMS}
            keep = {l: [] for l in layers} if c == keep_rows_at else None
            drows = {l: [] for l in layers} if want_diag else None
            t_cell = time.time()
            for lo, chunk in rig.iter_batches(uniq, batch_size):
                w = torch.from_numpy(weights[lo:lo + len(chunk)]).double()
                ids, attn, tpos = rig.build_batch(chunk, target_id, tok_bos, tok_pad)
                t0 = time.time()
                resid = rig.forward_capture(model, ids.to(DEVICE), attn.to(DEVICE), layers)
                got = rig.gather(resid, tpos)
                del resid
                for l in layers:
                    x = got[l]                                    # [b, 1152] fp32 cpu
                    code, recon = rig.sae_reconstruct(saes[l], x)
                    recon = recon.detach().cpu().double()
                    raw = x.double()
                    acc[(l, "sae")] += (recon * w[:, None]).sum(0).numpy()
                    acc[(l, "raw")] += (raw * w[:, None]).sum(0).numpy()
                    if keep is not None:
                        keep[l].append(recon.float().numpy() if CERT_ARM == "sae"
                                       else raw.float().numpy())
                    if want_diag and l in readers:
                        cnp = code.detach().cpu().numpy()
                        for i in range(len(chunk)):
                            for j in occ_of_uniq[lo + i]:
                                drows[l].append(
                                    truncation_row(cnp[i], readers[l], int(occ_rows[j]), c))
                    del code
                timing[c]["batches"] += 1
                timing[c]["seconds"] += time.time() - t0
                timing[c]["tokens"] += int(attn.sum())
            tot = float(weights.sum())
            for l in layers:
                for a in ARMS:
                    means[(l, c, a)][ti] = acc[(l, a)] / tot
                if keep is not None:
                    blk = np.vstack(keep[l])                      # [n_unique, 1152]
                    rows[(l, int(target_id))] = np.repeat(blk, weights, axis=0)
                if drows is not None and l in readers:
                    diag[(l, int(target_id))] = drows[l]
            meta[(int(target_id), c)] = dict(
                n=int(len(occ_rows)), n_unique=int(len(uniq)), n_forwarded=int(len(uniq)),
                occurrence_seed=int(OCC_SEED),
                contamination=contamination(ctx, contam_ids),
                n_distinct_docs=int(np.unique(occ_rows // rig.DOC_LEN).size),
                position_min=int((occ_rows % rig.DOC_LEN).min()) if occ_rows.size else None,
                position_max=int((occ_rows % rig.DOC_LEN).max()) if occ_rows.size else None)
            print(f"  {label}c={c:<4d} target={int(target_id):<7d} n={len(occ_rows):<4d} "
                  f"unique={len(uniq):<4d} contam={meta[(int(target_id), c)]['contamination']}"
                  f" {time.time() - t_cell:5.1f}s", flush=True)
    return means, meta, rows, timing, diag


def truncation_row(fresh_code, reader, row_index, c):
    """One occurrence's fresh-vs-cached SAE comparison. DIAGNOSTIC ONLY -- never a gate.

    The cached row was produced by a forward over the document's first 256 tokens; the fresh
    code comes from ``[BOS] + last-c tokens + target``. For p > c the two are different
    computations (truncated left context, and the target sits at model position c+1 instead
    of p+1), so disagreement is the expected outcome and its SIZE is the measurement. For
    p == c the sequences coincide token-for-token and position-for-position, which makes that
    subgroup an exact-reproduction control on the rig.
    """
    cidx, cval = reader.row(int(row_index))
    cidx = np.asarray(cidx, dtype=np.int64)
    cval = np.asarray(cval, dtype=np.float64)
    fidx = np.flatnonzero(fresh_code)
    fval = np.asarray(fresh_code, dtype=np.float64)[fidx]
    inter, ia, ib = np.intersect1d(cidx, fidx, assume_unique=True, return_indices=True)
    nc, nf = float(np.linalg.norm(cval)), float(np.linalg.norm(fval))
    dot = float(cval[ia] @ fval[ib]) if inter.size else 0.0
    union = len(cidx) + len(fidx) - len(inter)
    p = int(row_index) % rig.DOC_LEN
    return dict(
        row=int(row_index), p=p, exact_prefix=bool(p == int(c)),
        n_cached=int(len(cidx)), n_fresh=int(len(fidx)), n_shared=int(len(inter)),
        code_cosine=float(dot / max(nc * nf, 1e-12)),
        support_jaccard=float(len(inter) / max(union, 1)),
        support_recall_of_cached=float(len(inter) / max(len(cidx), 1)),
        identical_support=bool(len(cidx) == len(fidx) == len(inter)),
        max_abs_value_diff=(float(np.abs(cval[ia] - fval[ib]).max()) if inter.size else None))


def summarise_truncation(diag):
    """Medians of the per-occurrence diagnostic, overall and split by ``p == c``."""
    out = {}
    for (l, t), rowsl in diag.items():
        out.setdefault(str(l), {})[str(t)] = _trunc_stats(rowsl)
    for l in list(out):
        allrows = [r for (ll, t), rr in diag.items() if str(ll) == l for r in rr]
        out[l]["ALL"] = _trunc_stats(allrows)
        out[l]["ALL_exact_prefix_p_eq_c"] = _trunc_stats(
            [r for r in allrows if r["exact_prefix"]])
        out[l]["ALL_truncated_p_gt_c"] = _trunc_stats(
            [r for r in allrows if not r["exact_prefix"]])
    return out


def _trunc_stats(rowsl):
    if not rowsl:
        return dict(n=0)
    g = lambda k: np.asarray([r[k] for r in rowsl], dtype=np.float64)  # noqa: E731
    return dict(n=int(len(rowsl)),
                median_code_cosine=float(np.median(g("code_cosine"))),
                p05_code_cosine=float(np.percentile(g("code_cosine"), 5)),
                median_support_jaccard=float(np.median(g("support_jaccard"))),
                median_support_recall_of_cached=float(
                    np.median(g("support_recall_of_cached"))),
                frac_identical_support=float(np.mean([r["identical_support"] for r in rowsl])),
                median_n_cached=float(np.median(g("n_cached"))),
                median_n_fresh=float(np.median(g("n_fresh"))))


# ===========================================================================
# per-cell scoring (copied from scripts/37_context_ladder.py)
# ===========================================================================
def score_cell(M, ids, n_perm, want_ordering, seed):
    """screen() + (strict-12 only) ordering, for one (layer, rung, arm, set) cell."""
    s = screen(M, n_perm)
    out = dict(k=int(len(ids)), n_members=int(len(ids)))
    if s is None:
        out.update(degenerate=True, gap_ratio=None, max_gap_deg=None, radius_ratio=None,
                   p_closed=None, mean_radius=None, top2_var=None, pc1_var=None,
                   circle_resid_over_r=None)
    else:
        out.update(degenerate=False, **{k: v for k, v in s.items() if k != "k"},
                   mean_radius=mean_radius(M))
    out.update(rho=None, p_order=None, block_perm_p=None, order_symmetry=None,
               order_note=None)
    if not want_ordering:
        out["order_note"] = ("ordering not computed: cyclic_order_score's symmetry basis "
                             "needs distinct calendar positions and ' Feb' ties with "
                             "' February' at position 1 (prereg 3.1, see module docstring)")
        return out
    if s is None:
        out["order_note"] = "ordering not computed: point set is degenerate"
        return out
    ang, _, _ = cell_angles(M)
    lab = [CALENDAR_POS[int(t)] for t in ids]
    o = cyclic_order_score(ang, np.asarray(lab), len(ids), n_perm=N_PERM_ORDER,
                           seed=seed, full_k=FULL_K)
    bp = block_perm_order_p(ang, lab, FULL_K, [list(range(len(ids)))], N_BLOCK_PERM, seed)
    out.update(rho=float(o["rho"]), p_order=float(o["p"]),
               order_symmetry=[int(x) for x in o["symmetry"]],
               order_null_mean=float(o["null_mean"]), order_null_p95=float(o["null_p95"]),
               block_perm_p=float(bp["p"]), block_perm=bp,
               label_index=[int(x) for x in lab], angles_rad=[float(x) for x in ang])
    return out


def decide(table, layer, set_name, rungs):
    """Prereg 3.1 decision rule, per layer. SAE arm primary, raw arm co-reported."""
    def p(arm, c):
        cell = table.get((layer, c, arm, set_name))
        return None if cell is None else cell["p_closed"]

    have = sorted(rungs)
    complete = have == sorted(RUNGS)
    sae0, raw0 = p("sae", 0), p("raw", 0)
    disagree = None
    if sae0 is not None and raw0 is not None:
        disagree = bool((sae0 < ALPHA) != (raw0 < ALPHA))
    out = dict(layer=int(layer), member_set=set_name, rungs_available=have,
               ladder_complete=bool(complete),
               p_closed_sae={str(c): p("sae", c) for c in have},
               p_closed_raw={str(c): p("raw", c) for c in have},
               sae_vs_raw_disagreement_at_c0=disagree,
               p_closed_c0_sae=sae0, p_closed_c0_raw=raw0)

    branch_a = all(v is not None and v < ALPHA
                   for v in (p("sae", 0), p("sae", 1), p("raw", 0), p("raw", 1)))
    onset = None
    if sae0 is not None and sae0 >= ALPHA:
        for i, c in enumerate(have):
            tail = have[i:]
            vals = [p("sae", x) for x in tail]
            if c > 0 and all(v is not None and v < ALPHA for v in vals):
                onset = int(c)
                break
    if branch_a:
        call, why = "Branch A (token-intrinsic)", "p_closed < 0.05 at c=0 and c=1 on both arms"
    elif onset is not None:
        call, why = ("Branch B (contextual onset)",
                     f"c=0 SAE p_closed >= 0.05; closure from rung E={onset} upward")
    else:
        call, why = ("non-monotone / unresolved",
                     "neither Branch A nor a monotone onset rung E is satisfied")
    out.update(onset_rung_E=onset, branch=call, reason=why)
    if not complete:
        out["branch"] = "incomplete rung ladder -- no verdict"
        out["provisional_branch"] = call
        out["reason"] = (f"decision rule needs rungs {RUNGS}; this run has {have}. "
                         f"Provisional call on the available rungs only: {call} ({why})")
    return out


def synthetic_contrast(path, natural_cells):
    """The 37 c=128 cells beside the 37b c=128 cells (section 8's secondary finding).

    Three cases, all reported rather than crashed on: the 37 result may be absent (37 is
    queued), present as a declaration stub with no ``cells`` yet, or complete.
    """
    nat = {(c["layer"], c["arm"], c["member_set"]): c for c in natural_cells
           if int(c["rung"]) == 128}
    out = dict(source=str(path), rung=128,
               finding="context-genre dependence of the mean-code ring (prereg section 8)",
               natural_cells={f"L{l}|{a}|{s}": dict(gap_ratio=c["gap_ratio"],
                                                    p_closed=c["p_closed"], rho=c["rho"])
                              for (l, a, s), c in nat.items()})
    p = Path(path)
    if not p.exists():
        out.update(available=False,
                   note="results/phase3_context_ladder.json absent at 37b write time; "
                        "the contrast is a separate desk computation once 37 lands")
        return out
    try:
        j = json.loads(p.read_text())
    except json.JSONDecodeError as e:
        out.update(available=False, note=f"37 output is not readable JSON ({e})")
        return out
    if not j.get("cells"):
        out.update(available=False, synthetic_status=j.get("status"),
                   synthetic_declared_at=j.get("declared_at"),
                   note="37 output exists but carries no cells (declaration stub only)")
        return out
    syn = {(c["layer"], c["arm"], c["member_set"]): c for c in j["cells"]
           if int(c["rung"]) == 128}
    pairs = []
    for key in sorted(set(nat) | set(syn)):
        l, a, s = key
        sc, nc = syn.get(key), nat.get(key)
        pairs.append(dict(
            layer=int(l), arm=a, member_set=s,
            synthetic_gap_ratio=None if sc is None else sc["gap_ratio"],
            synthetic_p_closed=None if sc is None else sc["p_closed"],
            synthetic_rho=None if sc is None else sc.get("rho"),
            synthetic_n=None if sc is None else sc.get("n"),
            natural_gap_ratio=None if nc is None else nc["gap_ratio"],
            natural_p_closed=None if nc is None else nc["p_closed"],
            natural_rho=None if nc is None else nc.get("rho"),
            natural_n_min=None if nc is None else nc.get("n_min"),
            natural_n_max=None if nc is None else nc.get("n_max"),
            gap_ratio_delta=(None if (sc is None or nc is None
                                      or sc["gap_ratio"] is None or nc["gap_ratio"] is None)
                             else float(nc["gap_ratio"] - sc["gap_ratio"]))))
    out.update(available=True, synthetic_status=j.get("status"),
               synthetic_declared_at=j.get("declared_at"),
               synthetic_verdict=j.get("verdict"), synthetic_smoke=j.get("smoke"),
               synthetic_config=j.get("config"), cells=pairs)
    return out


# ===========================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="cache/sparse_big")
    ap.add_argument("--l0", default="big")
    ap.add_argument("--out", default="results/phase3_context_ladder_natural.json")
    ap.add_argument("--means-out", default=None,
                    help="default: the --out path with suffix _means.npz (so a scratch "
                         "smoke never writes an npz into results/)")
    ap.add_argument("--synthetic", default="results/phase3_context_ladder.json",
                    help="scripts/37 output, for the section-8 synthetic-vs-natural block")
    ap.add_argument("--layers", type=int, nargs="+", default=LAYERS)
    ap.add_argument("--rungs", type=int, nargs="+", default=RUNGS)
    ap.add_argument("--n-ctx", type=int, default=N_CTX)
    ap.add_argument("--n-perm", type=int, default=N_PERM_SCREEN)
    ap.add_argument("--n-ord", type=int, default=ORD_N)
    ap.add_argument("--cert-resamples", type=int, default=CERT_RESAMPLES)
    ap.add_argument("--batch-size", type=int, default=rig.BATCH_SIZE)
    ap.add_argument("--smoke", action="store_true",
                    help="annotate the output as a scaled-down instrument check")
    a = ap.parse_args()
    t_start = time.time()
    layers, rungs = sorted(set(a.layers)), sorted(set(a.rungs))
    if max(rungs) > P_FLOOR:
        raise SystemExit(f"rungs must be <= {P_FLOOR}: occurrences are drawn at p >= "
                         f"{P_FLOOR} so that every rung fits inside the document")
    # Derived from --out rather than defaulted to results/: a scratch smoke must not drop
    # an npz into the frozen results directory.
    mp = Path(a.means_out) if a.means_out else Path(a.out).with_name(
        Path(a.out).stem + "_means.npz")

    # ---- R1: declaration to disk BEFORE any cache row or forward pass ----
    declared = dict(
        script="scripts/37b_context_ladder_natural.py",
        prereg="docs/phase3_prereg_2026-08-21.md section 8 (Addendum A), rest of section 3.1",
        status="DECLARATION STUB -- rules frozen, no results yet",
        declared_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
        smoke=bool(a.smoke),
        variant_of="scripts/37_context_ladder.py (structure, statistics, decision rule and "
                   "ordinal control copied in; nothing imported from it, nothing edited)",
        provenance=(
            "scripts/37's pre-run rig validation FAILED on synthetic contexts: n=150, L6, "
            "c=128, SAE arm, month-free FineWeb prefixes gave gap_ratio 2.99 against the "
            "registered anchor 1.65876, while the same rig reproduced cached CSR rows "
            "bit-for-bit (60/60 identical support). The rig is therefore not what changed; "
            "the context DISTRIBUTION is. 37 runs as frozen and its gate outcome is the "
            "registered measurement; 37b re-asks the ladder question on the natural "
            "occurrence population the cached anchor was averaged over."),
        addendum_a_text=ADDENDUM_A_TEXT,
        question=("Does the L6/L7 month ring exist at short context when the contexts are "
                  "the token's own natural corpus contexts, and at what budget?"),
        members=dict(member_ids=MEMBER_IDS, feb_abbrev_id=FEB_ABBREV_ID,
                     strict12_ids=STRICT12_IDS,
                     member_hash_expected=MEMBER_HASH_EXPECTED,
                     member_hash_domain="sorted POOL INDICES (build_pool, count floor 100)",
                     member_index_rule="position in the ascending token-id list above"),
        rungs=rungs, layers=layers, n_contexts_requested=int(a.n_ctx),
        context_rule=(
            f"[BOS] + tokens2d[doc, p-c:p] + member token, where the occurrence is a cached "
            f"row r with tok[r] = member and p = r mod 255 >= {P_FLOOR}, doc = r // 255. "
            f"The context at c={P_FLOOR} is the LAST {P_FLOOR} TOKENS BEFORE THE OCCURRENCE, "
            f"NOT the start of the document -- the two coincide only for occurrences that "
            f"sit at p = {P_FLOOR} exactly. Contexts are NOT month-filtered (section 8); the "
            f"fraction containing any of the 13 member ids is reported per (member, rung). "
            f"Right padding; bf16/eager/use_cache=False."),
        occurrence_rule=(
            f"pool = every cached row with that token id at p >= {P_FLOOR}; "
            f"n = min({a.n_ctx}, pool size) drawn WITHOUT replacement by "
            f"np.random.default_rng({OCC_SEED}) per member; the draw does not depend on the "
            f"rung, so the SAME occurrences are used at every rung. n is therefore RAGGED "
            f"across members (pool sizes differ) and each member's realised n is reported."),
        position_note=(
            "the target sits at MODEL position c+1 in this rig, not at its natural document "
            "position p+1, so RoPE differs from the cached forward whenever p != c. This is "
            "scripts/37's convention, kept for comparability, and is why per-occurrence "
            "agreement with the cached rows is a diagnostic and not a gate."),
        seed_rule=f"occurrence draw: default_rng({OCC_SEED}) per member; screen: 0; "
                  f"ordering/block-permutation: 1000*c + layer",
        arms=dict(sae="mean over contexts of (fp32 JumpReLU code) @ w_dec, NO b_dec",
                  raw="mean over contexts of the fp32 resid_post at the target position",
                  primary="sae (the registered anchor is an SAE-reconstruction number)"),
        statistics=dict(
            screen=f"scripts/22 screen() verbatim, {a.n_perm} uniform-angle permutations",
            ordering=(f"cyclic_order_score(full_k=12, n_perm={N_PERM_ORDER}) on strict-12 "
                      f"only, plus a {N_BLOCK_PERM}-draw block-permutation p (single block "
                      f"of all 12 -> uniform permutations; strict-12 has no lexical blocks)"),
            angles="pca_2d -> fit_circle -> angles(xy, cx, cy), scripts/22 score_ordering"),
        rig_validation=dict(
            cell=f"c={CERT_RUNG}, L{CERT_LAYER}, {CERT_ARM} arm, BOTH member sets",
            rule=(f"|gap_ratio - {RIG_REF_GAP_RATIO}| <= {RIG_GAP_TOL} AND "
                  f"p_closed < {RIG_MAX_P_CLOSED} for strict12 AND set13 (section 8 widened "
                  f"the tolerance from 0.2 because n <= {a.n_ctx} occurrences at p >= "
                  f"{P_FLOOR} is a subsample of the 374-777 the pooled anchor used)"),
            on_failure=f"verdict '{RIG_FAIL_VERDICT}'; all numbers are still emitted"),
        truncation_diagnostic=dict(
            cell=f"c={CERT_RUNG}, per occurrence, layers with a cache directory",
            rule=("fresh fp32 JumpReLU code at the target position vs the cached CSR row for "
                  "that same occurrence: code cosine, support Jaccard, cached-support "
                  "recall, identical-support rate; medians reported"),
            expectation=("rows are NOT expected to be identical -- the context is truncated "
                         "to the last c tokens and the target is re-placed at model position "
                         "c+1 -- so this is DIAGNOSTIC, never a gate. The p == c subgroup is "
                         "an exact natural prefix at matching RoPE positions and is split "
                         "out; near-exact reproduction IS expected there.")),
        certificate=dict(
            cell=f"c={CERT_RUNG}, L{CERT_LAYER}, {CERT_ARM} arm",
            rule=(f"{a.cert_resamples} resamples of n={CERT_N} occurrences per member; "
                  f"closure pass rate (p_closed < {ALPHA}) must be >= {CERT_MIN_PASS}"),
            ragged_adaptation=(
                f"section 3.1 assumes a uniform n; natural pools are ragged, so each member "
                f"is capped at min({CERT_N}, its own n) rather than every member being cut "
                f"to the smallest pool. Members whose pool is below {CERT_N} are named in "
                f"the result with their n, and the cell is flagged degenerate for them "
                f"(every resample is that member's full set).")),
        ordinal_control=dict(
            words=ORD_WORDS, rungs=ORD_RUNGS, n_requested=int(a.n_ord),
            rule=(f"leading-space single tokens only, drawn from their OWN natural "
                  f"occurrences at p >= {P_FLOOR} exactly as the members are; "
                  f"non-single-token words and words with an empty pool are reported and "
                  f"dropped; any cell with p_closed < {ALPHA} sets run_void=true")),
        decision_rule=dict(
            branch_a="p_closed < 0.05 at c=0 AND c=1 on BOTH arms -> token-intrinsic",
            branch_b=("p_closed >= 0.05 at c=0 on the SAE arm AND a rung E with "
                      "p_closed < 0.05 at E and at every rung > E -> contextual onset at E"),
            otherwise="non-monotone / unresolved; SAE-vs-raw disagreement at c=0 reported"),
        secondary_finding=("synthetic_vs_natural: the scripts/37 c=128 cells beside these, "
                           "as the registered context-genre contrast (section 8)"),
        no_bdec_note=("SAE reconstruction is code @ w_dec with no b_dec, matching every "
                      "closure number the campaign has reported; b_dec is type-independent "
                      "but translates the cloud and would move every fitted angle."),
    )
    outp = Path(a.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(declared, indent=2))
    print(f"declaration stub -> {outp}", flush=True)

    # =======================================================================
    # everything below this line reads the cache
    # =======================================================================
    tokens_flat = np.fromfile(Path(a.root) / "layer_0" / "tokens.i32", dtype=np.int32)
    counts = np.bincount(tokens_flat)
    pool = np.flatnonzero(counts >= COUNT_FLOOR)
    lut = {int(t): i for i, t in enumerate(pool)}
    missing = [t for t in MEMBER_IDS if t not in lut]
    if missing:
        raise RuntimeError(f"members absent from the count>={COUNT_FLOOR} pool: {missing}")
    pool_idx = [lut[t] for t in MEMBER_IDS]
    got_hash = member_hash(pool_idx)
    assert got_hash == MEMBER_HASH_EXPECTED, (
        f"member_hash {got_hash} != {MEMBER_HASH_EXPECTED}; the frozen object is not the "
        f"object this pool names")
    print(f"pool {len(pool)} types; member_hash {got_hash} OK "
          f"(pool indices {sorted(pool_idx)})", flush=True)
    tokens2d = tokens_flat.reshape(rig.N_DOCS, rig.DOC_LEN)

    # ---- occurrence pools (section 8) -------------------------------------
    print(f"\noccurrence pools at p >= {P_FLOOR} (n = min({a.n_ctx}, pool)):", flush=True)
    occ, occ_meta = {}, {}
    for t in MEMBER_IDS:
        pl = occurrence_pool(tokens_flat, t, P_FLOOR)
        dr = draw_occurrences(pl, a.n_ctx, OCC_SEED)
        occ[int(t)] = dr
        occ_meta[int(t)] = dict(
            token_id=int(t), count_total=int(counts[t]), pool_at_p_floor=int(pl.size),
            n_drawn=int(dr.size), subsampled=bool(dr.size < pl.size),
            n_distinct_docs=int(np.unique(dr // rig.DOC_LEN).size),
            n_at_p_equals_floor=int((dr % rig.DOC_LEN == P_FLOOR).sum()),
            position_median=float(np.median(dr % rig.DOC_LEN)) if dr.size else None,
            rows=[int(x) for x in dr])
        print(f"  {int(t):<7d} total={int(counts[t]):<5d} pool(p>={P_FLOOR})={pl.size:<5d} "
              f"drawn={dr.size:<4d} docs={occ_meta[int(t)]['n_distinct_docs']:<4d} "
              f"p==128:{occ_meta[int(t)]['n_at_p_equals_floor']}", flush=True)
    if min(v["n_drawn"] for v in occ_meta.values()) < 3:
        raise RuntimeError("a member has fewer than 3 late-position occurrences")

    # ---- model + SAEs -----------------------------------------------------
    hf = os.environ.get("HF_TOKEN")
    model, tokzr = load_model(token=hf)
    bos = tokzr.bos_token_id
    pad = tokzr.pad_token_id if tokzr.pad_token_id is not None else tokzr.eos_token_id
    # SAEs live on DEVICE, matching scripts/06_cache_sparse.py's encode path -- the frozen
    # anchor this run validates against was produced by that path, so the JumpReLU gate is
    # evaluated on the same backend, and so is the truncation diagnostic's fresh code.
    saes = {l: load_sae(l, a.l0, device=DEVICE, token=hf) for l in layers}
    readers = {l: SparseReader(a.root, l) for l in layers
               if (Path(a.root) / f"layer_{l}" / "meta.json").exists()}
    print(f"model on {DEVICE}; SAEs {a.l0} at layers {layers}; bos={bos} pad={pad}; "
          f"cache readers at {sorted(readers)}", flush=True)

    tokenizer_check = {}
    for m in MONTHS:
        enc = tokzr.encode(" " + m, add_special_tokens=False)
        tokenizer_check[" " + m] = enc
        assert len(enc) == 1, f"' {m}' is not a single token: {enc}"
    tokenizer_check[" Feb"] = tokzr.encode(" Feb", add_special_tokens=False)
    assert tokenizer_check[" Feb"] == [FEB_ABBREV_ID]
    assert sorted(tokenizer_check[" " + m][0] for m in MONTHS) == sorted(STRICT12_IDS), \
        "the tokenizer's month ids do not match the frozen strict-12 set"

    # ---- ordinal control token resolution (before any of its forwards) ----
    ord_ids, ord_dropped, ord_occ, ord_occ_meta = [], {}, {}, {}
    for w in ORD_WORDS:
        enc = tokzr.encode(" " + w, add_special_tokens=False)
        if len(enc) != 1:
            ord_dropped[" " + w] = dict(reason="not a single token",
                                        tokenization=[int(x) for x in enc])
            continue
        tid = int(enc[0])
        pl = occurrence_pool(tokens_flat, tid, P_FLOOR)
        dr = draw_occurrences(pl, a.n_ord, OCC_SEED)
        if dr.size < 3:
            ord_dropped[" " + w] = dict(reason=f"only {int(pl.size)} occurrences at "
                                               f"p >= {P_FLOOR}", token_id=tid)
            continue
        ord_ids.append(tid)
        ord_occ[tid] = dr
        ord_occ_meta[tid] = dict(word=" " + w, token_id=tid, count_total=int(counts[tid]),
                                 pool_at_p_floor=int(pl.size), n_drawn=int(dr.size),
                                 rows=[int(x) for x in dr])
    print(f"ordinal control: {len(ord_ids)} usable tokens, {len(ord_dropped)} dropped "
          f"({sorted(ord_dropped)})", flush=True)
    del tokens_flat

    # =======================================================================
    # the ladder
    # =======================================================================
    print(f"\nnatural month ladder: {len(MEMBER_IDS)} members x {len(rungs)} rungs, "
          f"n per member = {[occ_meta[int(t)]['n_drawn'] for t in MEMBER_IDS]}", flush=True)
    keep_at = CERT_RUNG if (CERT_RUNG in rungs and CERT_LAYER in layers) else None
    diag_at = CERT_RUNG if CERT_RUNG in rungs else None
    means, meta, cert_rows, timing, diag = run_ladder_natural(
        model, bos, pad, saes, tokens2d, MEMBER_IDS, occ, rungs, layers,
        batch_size=a.batch_size, keep_rows_at=keep_at, contam_ids=MEMBER_IDS,
        readers=readers, diag_rung=diag_at)

    # ---- per-cell table ---------------------------------------------------
    print("\nscoring cells", flush=True)
    table, table_rows = {}, []
    row_of = {int(t): i for i, t in enumerate(MEMBER_IDS)}
    for l in layers:
        for c in rungs:
            for arm in ARMS:
                for sname, sids in SETS.items():
                    M = means[(l, c, arm)][[row_of[t] for t in sids]]
                    cell = score_cell(M, sids, a.n_perm,
                                      want_ordering=(sname == "strict12"),
                                      seed=1000 * c + l)
                    ns = [meta[(int(t), c)]["n"] for t in sids]
                    uq = [meta[(int(t), c)]["n_unique"] for t in sids]
                    cm = [meta[(int(t), c)]["contamination"] for t in sids]
                    cell.update(layer=int(l), rung=int(c), arm=arm, member_set=sname,
                                n_min=int(min(ns)), n_max=int(max(ns)),
                                n_total=int(sum(ns)),
                                n_unique_contexts_min=int(min(uq)),
                                n_unique_contexts_max=int(max(uq)),
                                contamination_mean=float(np.mean(cm)),
                                contamination_max=float(np.max(cm)))
                    table[(l, c, arm, sname)] = cell
                    table_rows.append(cell)
                    gr = "None" if cell["gap_ratio"] is None else f"{cell['gap_ratio']:.4f}"
                    pc = "None" if cell["p_closed"] is None else f"{cell['p_closed']:.5f}"
                    rho = "" if cell["rho"] is None else f" rho={cell['rho']:.3f}"
                    print(f"  L{l} c={c:<4d} {arm:<3s} {sname:<8s} gap_ratio={gr:<8s} "
                          f"p_closed={pc}{rho}", flush=True)

    # ---- contamination summary (section 8's reported quantity) ------------
    contam = {}
    for c in rungs:
        per = {str(int(t)): meta[(int(t), c)]["contamination"] for t in MEMBER_IDS}
        vals = np.asarray(list(per.values()), dtype=np.float64)
        contam[str(c)] = dict(per_member=per, mean=float(vals.mean()),
                              min=float(vals.min()), max=float(vals.max()))
        print(f"  contamination c={c:<4d} mean={vals.mean():.3f} max={vals.max():.3f}",
              flush=True)

    # ---- rig validation (section 8 wording and tolerance) -----------------
    rig_val = dict(cell=dict(layer=CERT_LAYER, rung=CERT_RUNG, arm=CERT_ARM),
                   rule=declared["rig_validation"]["rule"], sets={})
    ok_all = True
    for sname in SETS:
        cell = table.get((CERT_LAYER, CERT_RUNG, CERT_ARM, sname))
        if cell is None or cell["gap_ratio"] is None:
            rig_val["sets"][sname] = dict(available=False, passed=False)
            ok_all = False
            continue
        gr, pc = cell["gap_ratio"], cell["p_closed"]
        ok = bool(abs(gr - RIG_REF_GAP_RATIO) <= RIG_GAP_TOL and pc < RIG_MAX_P_CLOSED)
        rig_val["sets"][sname] = dict(available=True, gap_ratio=gr, p_closed=pc,
                                      gap_ratio_abs_error=float(abs(gr - RIG_REF_GAP_RATIO)),
                                      passed=ok)
        ok_all = ok_all and ok
    rig_val["passed"] = bool(ok_all)
    rig_val["anchor_set"] = "both (section 8 requires strict12 AND set13)"
    rig_val["verdict"] = "natural-context ladder validated" if ok_all else RIG_FAIL_VERDICT
    print(f"\nrig validation: {rig_val['verdict']} {rig_val['sets']}", flush=True)

    # ---- truncation diagnostic (NOT a gate) -------------------------------
    trunc = dict(declared["truncation_diagnostic"])
    trunc["by_layer"] = summarise_truncation(diag) if diag else {}
    trunc["available"] = bool(diag)
    if not diag:
        trunc["note"] = f"c={CERT_RUNG} not in this run's rungs, or no cache reader"
    for l, blk in trunc["by_layer"].items():
        print(f"  truncation L{l}: all n={blk['ALL']['n']} "
              f"cos={blk['ALL']['median_code_cosine']:.4f} "
              f"jac={blk['ALL']['median_support_jaccard']:.3f} | "
              f"p==c n={blk['ALL_exact_prefix_p_eq_c']['n']} "
              f"cos={blk['ALL_exact_prefix_p_eq_c'].get('median_code_cosine')} | "
              f"p>c n={blk['ALL_truncated_p_gt_c']['n']} "
              f"cos={blk['ALL_truncated_p_gt_c'].get('median_code_cosine')}", flush=True)

    # ---- matched-n certificate (ragged-n adaptation) ----------------------
    cert = dict(cell=dict(layer=CERT_LAYER, rung=CERT_RUNG, arm=CERT_ARM),
                n_requested=CERT_N, n_resamples=int(a.cert_resamples),
                threshold=CERT_MIN_PASS, sets={})
    if cert_rows:
        avail = {int(t): cert_rows[(CERT_LAYER, int(t))].shape[0] for t in MEMBER_IDS}
        used = {t: int(min(CERT_N, n)) for t, n in avail.items()}
        short = {str(t): int(n) for t, n in avail.items() if n < CERT_N}
        cert.update(n_available_per_member=avail, n_used_per_member=used,
                    members_below_n_requested=short)
        crng = np.random.default_rng(20260821)
        for sname, sids in SETS.items():
            passes = 0
            for _ in range(int(a.cert_resamples)):
                M = np.stack([cert_rows[(CERT_LAYER, int(t))][
                    crng.permutation(avail[int(t)])[:used[int(t)]]].mean(0) for t in sids])
                s = screen(M, a.n_perm)
                passes += int(s is not None and s["p_closed"] < ALPHA)
            rate = passes / max(int(a.cert_resamples), 1)
            deg = [str(t) for t in sids if used[int(t)] >= avail[int(t)]]
            cert["sets"][sname] = dict(pass_rate=float(rate),
                                       passed=bool(rate >= CERT_MIN_PASS),
                                       degenerate_members=deg,
                                       n_used=[used[int(t)] for t in sids])
        cert["passed"] = bool(cert["sets"]["set13"]["passed"])
        cert["primary_set"] = "set13"
        cert["resample_rule"] = (
            f"subsample WITHOUT replacement, n = min({CERT_N}, that member's stored "
            f"per-occurrence reconstructions at c={CERT_RUNG}); the matched-n question is "
            f"whether n={CERT_N} contexts would have sufficed, which is a subsample of the "
            f"run, not a bootstrap of it. Natural pools are ragged, so the cap is applied "
            f"per member; for a member at or below the cap every resample is its full set "
            f"and that member contributes no variance (named in degenerate_members).")
    else:
        cert.update(available=False, passed=False,
                    note=f"c={CERT_RUNG} / L{CERT_LAYER} not in this run's grid")
    print(f"certificate: {cert.get('sets', cert.get('note'))}", flush=True)

    # =======================================================================
    # ordinal control -- natural occurrences too (section 8)
    # =======================================================================
    ord_out = dict(words=ORD_WORDS, dropped=ord_dropped, kept_ids=ord_ids,
                   k=len(ord_ids), n_requested=int(a.n_ord), rungs=ORD_RUNGS,
                   occurrences={str(k): v for k, v in ord_occ_meta.items()}, cells=[])
    ord_means = {}
    ord_rungs = [c for c in ORD_RUNGS if c <= P_FLOOR]
    if len(ord_ids) >= 3:
        print(f"\nordinal control: {len(ord_ids)} tokens x {len(ord_rungs)} rungs, "
              f"n = {[ord_occ_meta[t]['n_drawn'] for t in ord_ids]}", flush=True)
        om, ometa, _, otiming, _ = run_ladder_natural(
            model, bos, pad, saes, tokens2d, ord_ids, ord_occ, ord_rungs, layers,
            batch_size=a.batch_size, keep_rows_at=None, contam_ids=ord_ids,
            readers=None, diag_rung=None, label="ord ")
        ord_means = om
        for l in layers:
            for c in ord_rungs:
                for arm in ARMS:
                    s = screen(om[(l, c, arm)], a.n_perm)
                    ns = [ometa[(int(t), c)]["n"] for t in ord_ids]
                    cm = [ometa[(int(t), c)]["contamination"] for t in ord_ids]
                    cell = dict(layer=int(l), rung=int(c), arm=arm, k=len(ord_ids),
                                n_min=int(min(ns)), n_max=int(max(ns)),
                                contamination_mean=float(np.mean(cm)))
                    cell.update({k: v for k, v in (s or {}).items() if k != "k"})
                    cell.setdefault("p_closed", None)
                    ord_out["cells"].append(cell)
                    pc = "None" if cell["p_closed"] is None else f"{cell['p_closed']:.5f}"
                    print(f"  ord L{l} c={c:<4d} {arm:<3s} p_closed={pc}", flush=True)
        ord_out["timing"] = {str(k): v for k, v in otiming.items()}
    else:
        ord_out["note"] = "fewer than 3 usable ordinals survive; control not runnable"
    hits = [c for c in ord_out["cells"]
            if c["p_closed"] is not None and c["p_closed"] < ALPHA]
    ord_out["closing_cells"] = hits
    run_void = bool(hits)
    ord_out["run_void"] = run_void
    print(f"ordinal control: {len(hits)} closing cells -> run_void={run_void}", flush=True)

    # =======================================================================
    # decisions
    # =======================================================================
    decisions = []
    for l in layers:
        for sname in SETS:
            d = decide(table, l, sname, rungs)
            d["primary"] = bool(sname == "set13")
            d["gated_by"] = dict(rig_validated=rig_val["passed"], run_void=run_void,
                                 certificate_passed=bool(cert.get("passed")))
            # A failed gate withholds the verdict; it does not erase what the rule computed.
            # Keeping the branch the rule WOULD have called is what lets a reader see that
            # the gate, and not a convenient answer, is doing the work.
            if not rig_val["passed"] or run_void:
                d["branch_under_gate"] = d.get("provisional_branch", d["branch"])
                d["branch"] = ("VOID (ordinal control closed) -- no verdict" if run_void
                               else f"{RIG_FAIL_VERDICT} -- no verdict")
            decisions.append(d)
            if sname == "set13":
                print(f"  L{l} {sname}: {d['branch']}", flush=True)

    # =======================================================================
    # secondary finding -- synthetic (37) vs natural (37b) at c=128
    # =======================================================================
    contrast = synthetic_contrast(a.synthetic, table_rows)
    print(f"\nsynthetic_vs_natural: available={contrast['available']} "
          f"({contrast.get('note', contrast.get('synthetic_status'))})", flush=True)

    # =======================================================================
    # runtime extrapolation
    # =======================================================================
    def plan_cost(rungs_, n_by_target, bs):
        """(batches, tokens) for a ladder with RAGGED per-target n; c=0 dedupes to one row."""
        b = t = 0
        for c in rungs_:
            for n_m in n_by_target:
                nu = 1 if c == 0 else int(n_m)
                b += int(np.ceil(nu / bs))
                t += nu * (c + 2)
        return b, t

    obs_b = sum(v["batches"] for v in timing.values())
    obs_s = sum(v["seconds"] for v in timing.values())
    obs_t = sum(v["tokens"] for v in timing.values())
    if ord_out.get("timing"):
        obs_b += sum(v["batches"] for v in ord_out["timing"].values())
        obs_s += sum(v["seconds"] for v in ord_out["timing"].values())
        obs_t += sum(v["tokens"] for v in ord_out["timing"].values())
    # seconds ~ alpha * batches + beta * tokens, fitted on the per-rung aggregates
    A = np.array([[v["batches"], v["tokens"]] for v in timing.values()], dtype=np.float64)
    y = np.array([v["seconds"] for v in timing.values()], dtype=np.float64)
    coef = np.linalg.lstsq(A, y, rcond=None)[0] if len(y) >= 2 else np.array([0.0, 0.0])
    coef = np.maximum(coef, 0.0)
    full_month_n = [min(N_CTX, occ_meta[int(t)]["pool_at_p_floor"]) for t in MEMBER_IDS]
    full_ord_n = [min(ORD_N, ord_occ_meta[t]["pool_at_p_floor"]) for t in ord_ids] or [1]
    fb, ft = plan_cost(RUNGS, full_month_n, a.batch_size)
    ob, ot = plan_cost(ORD_RUNGS, full_ord_n, a.batch_size)
    full_forward_s = float(coef[0] * (fb + ob) + coef[1] * (ft + ot))
    fit_used = "two-term fit (batch overhead + per-token)"
    if full_forward_s <= 0 and obs_t > 0:
        # fewer than two distinct rungs -> the two-term fit is underdetermined; fall back
        # to this run's own throughput, which over-charges the short rungs.
        full_forward_s = float((ft + ot) / (obs_t / max(obs_s, 1e-9)))
        fit_used = "throughput fallback (only one rung timed; two-term fit unidentifiable)"
    per_rung_sec_per_batch = {str(c): (v["seconds"] / v["batches"] if v["batches"] else None)
                              for c, v in timing.items()}
    runtime = dict(
        observed=dict(batches=int(obs_b), seconds_forward=float(obs_s), tokens=int(obs_t),
                      sec_per_batch=float(obs_s / max(obs_b, 1)),
                      tokens_per_sec=float(obs_t / max(obs_s, 1e-9)),
                      sec_per_batch_by_rung=per_rung_sec_per_batch),
        model_seconds=dict(fit_batch_overhead_s=float(coef[0]),
                           fit_sec_per_token=float(coef[1])),
        full_plan=dict(month_batches=int(fb), month_tokens=int(ft),
                       ordinal_batches=int(ob), ordinal_tokens=int(ot),
                       month_n_per_member=full_month_n,
                       month_sequences_forwarded=int(sum(
                           sum(1 if c == 0 else n for n in full_month_n) for c in RUNGS)),
                       estimated_forward_seconds=full_forward_s,
                       estimated_forward_minutes=full_forward_s / 60.0,
                       method=fit_used),
        note=("seconds ~ a*batches + b*tokens, least squares on this run's per-rung "
              "aggregates; excludes model/SAE load and all permutation work. If another "
              "job shared the device during this run the fit is an UPPER bound."))
    print(f"\nruntime: {obs_s:.1f}s forward over {obs_b} batches "
          f"({obs_s / max(obs_b, 1):.3f} s/batch, {obs_t / max(obs_s, 1e-9):,.0f} tok/s); "
          f"full-plan forward estimate {full_forward_s / 60:.1f} min", flush=True)

    # =======================================================================
    # write
    # =======================================================================
    npz = {}
    for (l, c, arm), M in means.items():
        npz[f"L{l}_c{c}_{arm}"] = M.astype(np.float32)
    for (l, c, arm), M in ord_means.items():
        npz[f"ord_L{l}_c{c}_{arm}"] = M.astype(np.float32)
    npz["member_ids"] = np.asarray(MEMBER_IDS, dtype=np.int64)
    npz["strict12_ids"] = np.asarray(STRICT12_IDS, dtype=np.int64)
    npz["ordinal_ids"] = np.asarray(ord_ids, dtype=np.int64)
    npz["rungs"] = np.asarray(rungs, dtype=np.int64)
    npz["layers"] = np.asarray(layers, dtype=np.int64)
    for t in MEMBER_IDS:
        npz[f"occ_rows_{int(t)}"] = np.asarray(occ[int(t)], dtype=np.int64)
    mp.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(mp, **npz)

    verdict = rig_val["verdict"]
    if run_void:
        verdict = "VOID (ordinal control closed); " + verdict

    out = dict(declared)
    out.update(
        status="COMPLETE",
        declared_at=declared["declared_at"],
        finished_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
        smoke=bool(a.smoke),
        config=dict(root=a.root, l0=a.l0, layers=layers, rungs=rungs, n_ctx=int(a.n_ctx),
                    n_perm=int(a.n_perm), n_ord=int(a.n_ord),
                    cert_resamples=int(a.cert_resamples), batch_size=int(a.batch_size),
                    device=DEVICE, means_out=str(mp), synthetic_source=str(a.synthetic)),
        members=dict(declared["members"], pool_indices=sorted(int(x) for x in pool_idx),
                     member_hash=got_hash, pool_size=int(len(pool)),
                     calendar_position={str(k): int(v) for k, v in CALENDAR_POS.items()},
                     member_index={str(t): i for i, t in enumerate(MEMBER_IDS)},
                     tokenizer_check={k: [int(x) for x in v]
                                      for k, v in tokenizer_check.items()}),
        occurrences={str(k): v for k, v in occ_meta.items()},
        cells=table_rows,
        contamination_by_rung=contam,
        context_meta={f"{t}|{c}": v for (t, c), v in meta.items()},
        rig_validation=dict(declared["rig_validation"], **rig_val),
        truncation_diagnostic=trunc,
        certificate=dict(declared["certificate"], **cert),
        ordinal_control=dict(declared["ordinal_control"], **ord_out),
        decisions=decisions,
        synthetic_vs_natural=contrast,
        run_void=run_void,
        verdict=verdict,
        timing_by_rung={str(k): v for k, v in timing.items()},
        runtime_extrapolation=runtime,
        runtime_seconds=float(time.time() - t_start),
    )
    outp.write_text(json.dumps(out, indent=2, default=float))
    print(f"\nverdict: {verdict}", flush=True)
    print(f"means -> {mp}", flush=True)
    print(f"wrote {outp} ({time.time() - t_start:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
