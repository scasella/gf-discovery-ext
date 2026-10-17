"""Stage 3, avenue 3 -- IS THE MONTH RING A PER-OCCURRENCE OBJECT, OR ONLY AN AVERAGE?

Pre-registered in docs/phase3_prereg_2026-08-21.md section 4 (frozen 2026-08-21); design and
the critic's five repairs in docs/next_avenues_2026-08-21.md avenue 3. Writes
results/phase3_per_occurrence.json plus results/phase3_per_occurrence_coords.npz.

THE QUESTION. Every closure result this project has is a statement about a *mean*: script 11
averages the full 16,384-latent reconstruction over the positions where a month token sits,
and results/token_type_geometry.json's own context_averaging_note says so out loud -- "a ring
here is a statement about the model's average representation of those token types". The
flagship object (L6 centred_leiden cluster 14, member_hash 7d1d75a303d6, gap_ratio 1.65876,
p_closed 0.00125) is a ring of twelve CENTROIDS. Whether any single forward pass carries the
ring is a different question, and it is the one the mean cannot answer: twelve clouds whose
centroids sit on a circle can each be so wide that a single draw from each is indistinguishable
from noise. That is the honest null, and it is the null this script tests.

WHAT MAKES IT A TEST AND NOT A DEMONSTRATION. Three things, all from the critic:

  1. THE PLANE IS HELD OUT. Split by SEQUENCE id parity, not row parity -- two occurrences of
     "March" in the same document are the same evidence and must not straddle the split. The
     even-sequence rows give twelve type means; their top-2 PCA plane P is the plane. Every
     statistic is computed on ODD-sequence rows projected into P. Nothing is fitted twice.
  2. THE DECISION NULL PRESERVES THE MEAN RING. A ring-of-means plus wide isotropic clouds
     will pass a naive per-draw closure test at some rate purely because the means are on a
     circle. So the null keeps the twelve odd-sequence means and the plane exactly, and swaps
     each drawn row's IN-PLANE RESIDUAL for the residual of a random odd row of a DIFFERENT
     month. Marginals, ring, and residual distribution all survive; only the occurrence-to-type
     correspondence is destroyed. S beating that null's 99th percentile is the only positive.
  3. THE NEGATIVE IS PRICED. A per-occurrence ring is planted at known radii and the detection
     rate measured, so "S did not beat the null" comes with the amplitude at which it would
     have. And the SAE-fidelity number is carried from scripts/35 part_a_prime rather than
     assumed, so a negative cannot be an artifact of the reconstruction quietly being poor.

NO FORWARD PASSES. Everything here reads the existing sparse cache. Fidelity is imported from
results/phase3_ring_readers_gonogo.json (part_a_prime), which measured it with forwards.

b_dec, AND WHY ITS ABSENCE IS NOT A PROBLEM HERE. Prereg section 4 says the full-row
reconstruction is `row code @ w_dec`, with no b_dec. b_dec is a single constant vector added to
every row alike, so it shifts the entire point cloud -- month rows, background rows, type means
and plane centre -- by the same amount. pca_2d centres before it decomposes, and every
projection in this script subtracts the plane centre, so b_dec cancels EXACTLY, to floating
point, in every angle, gap, radius and rho reported below. This is why the gonogo's
median_cos_cached_nobdec_month of 0.334 at L7 (co-reported) does not invalidate the instrument:
that cosine is small because the omitted b_dec is a large common-mode offset, which is precisely
the component this script's centring removes. The gate-bearing fidelity number is the one the
prereg names -- median month-row reconstruction cosine WITH b_dec, 0.99834 at L7.

WHAT IS COPIED AND FROM WHERE. gap_null / gap_p are copied verbatim from
scripts/22_token_type_geometry.py L364-376 (which is itself scripts/13_rerun_gate.gap_pvalue),
so the closure test a draw faces is byte-identical to the one the flagship object passed.
build_pool is scripts/22 L211. The strict-12 primary ids are derived by scripts/22 pack()'s
rule (L464-476: ids in the >=100 pool, sorted by descending count, take the first) and then
ASSERTED against prereg section 0's exposure record. pca_2d / fit_circle / circularity / angles
/ cyclic_order_score are src/gfdisc/fits.py unchanged. frequency_matched_resample is
src/gfdisc/nulls.py unchanged. The declaration-stub-before-any-read discipline is scripts/22
L579-586 and scripts/38 L570-592 (prereg R1).

BATCHED ARITHMETIC, AND ITS PROOF. ~500k twelve-point closure tests are run (draws x nulls x
power x background). Calling fit_circle/circularity in a Python loop that many times is the one
thing that would blow the runtime, so a batched Kasa fit (normal equations instead of lstsq)
and a batched max-gap are used. They are NOT trusted: `batch_validation` in the output re-runs
the frozen fits.fit_circle + fits.circularity on a random sample of the real draws and reports
the maximum absolute disagreement in max_gap_deg and any disagreement in the pass/fail call.
Likewise the vectorised gap_p (searchsorted on the sorted null, tie convention g <= max_gap) is
checked against the copied scalar gap_p on sampled values.

MEMORY, AND A DECLARED READING OF THE TASK CONSTRAINT. The instruction was "stream the 800-type
background per type-set draw, or keep per-type in-plane coordinates only". The second option is
impossible as stated: each of the 500 background sets fits its OWN plane from its OWN twelve
even means, so there is no single set of in-plane coordinates a type could be reduced to. The
first re-reads each type ~7.5x (500 sets x 12 / 800 types) for no memory saving. The operative
constraint is RSS < 6 GB, so this script streams the cache ONCE per layer, keeps only the
odd-sequence background rows as fp32 (~181k x 1152 = 0.83 GB), frees them before the next layer,
and reports measured peak RSS. Even-sequence rows are never stored: their per-type sums are
accumulated during the same pass. Month rows are ~2,930 odd rows, 13 MB.
"""
import argparse, json, os, resource, sys, time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HUB_OFFLINE", "1")
sys.path.insert(0, str(REPO / "src"))
from gfdisc.cache import SparseReader
from gfdisc.fits import (pca_2d, fit_circle, circularity, angles, cyclic_order_score,
                         degenerate)
from gfdisc.nulls import frequency_matched_resample
from gfdisc.saes import load_sae, get_decoder
from gfdisc.tokens import MONTHS

# ===========================================================================
# DECLARED CONSTANTS -- frozen, written to disk before any cache row is read
# ===========================================================================
LAYERS = [5, 6, 7, 8, 9]
PRIMARY_LAYER = 7
CONTROL_LAYERS = [6, 7]          # the positive control must pass at BOTH
D_MODEL = 1152
SEQ_LEN = 255                    # rows per cached sequence; sequence id = row // 255
COUNT_FLOOR = 100                # scripts/22 build_pool main pool
N_DRAWS = 1000                   # primary S_L draws
N_NULL_BLOCKS = 200              # decision-null blocks
NULL_BLOCK_DRAWS = 100           # draws per null block (200 x 100 = 20,000 null draws)
N_BG_TYPES = 800                 # frozen background pool size
N_BG_SETS = 500                  # random 12-type sets from the background
N_BG_DRAWS = 200                 # draws per background set
N_PERM = 20000                   # uniform-angle max-gap null (the frozen closure test)
N_PERM_ORDER = 2000              # permutation null inside cyclic_order_score (rho gates, p informs)
ALPHA = 0.05                     # closure threshold, frozen
RHOS = [0.25, 0.5, 1.0, 2.0, 4.0]
POWER_TARGET = 0.8
FREQ_TOL_LOG10 = 0.25
SEED = 0
MAX_REJECT_ROUNDS = 1000
ARMS = ("raw", "l2", "rms")
PRIMARY_ARM = "raw"
BG_ARMS = ("raw", "l2")          # rms is a global scalar => angle-identical to raw (asserted)
CONTROL_RHO_MIN = 0.9
FIDELITY_GATE_MIN = 0.9
BG_COST_BUDGET_S = 40 * 60       # if the projected background cost exceeds this, drop to 200 sets
BG_SETS_REDUCED = 200

# prereg section 0 exposure record. The strict-12 primary ids are DERIVED by pack()'s rule and
# asserted against this list; " Feb" (5656) is the 13th member of the known object and is
# excluded from the background even though its count (141) is below the frequency-matched band.
STRICT12_IDS_EXPECTED = [3353, 5142, 5261, 5365, 5505, 5729, 5857, 6027, 6198, 6243, 6384, 7250]
MEMBER13_IDS = sorted(STRICT12_IDS_EXPECTED + [5656])


def declared_rules(a, layers):
    return dict(
        rows=("strict-12 primary token ids (highest-count id per month inside the count>=100 "
              "pool, scripts/22 pack() L464-476), ALL cache occurrences via "
              "SparseReader.positions_of_tokens; sequence id = row // 255"),
        reconstruction=("full-row x = row_code @ w_dec, NO b_dec (prereg section 4). b_dec is a "
                        "single constant vector added to every row alike, so it cancels EXACTLY "
                        "in every statistic here: pca_2d centres before decomposing and every "
                        "projection subtracts the plane centre. Stated because the fidelity "
                        "number measured WITHOUT b_dec (0.334 at L7) would otherwise look like a "
                        "problem; the omitted term is common-mode and centring removes it."),
        arms=dict(
            raw="the reconstruction as-is. PRIMARY ARM (prereg section 4 order).",
            l2="each row divided by its own L2 norm BEFORE means, plane and projection.",
            rms=("every row divided by ONE per-layer scalar = sqrt(mean(x^2)) over all entries "
                 "of all month rows (even and odd) at that layer. This is a global rescaling, so "
                 "it maps the plane, the centre and every point by the same positive factor and "
                 "leaves every angle, gap ratio and rho invariant. Run anyway and ASSERTED equal "
                 "to raw; its background is skipped as provably identical."),
        ),
        split=("sequence-id PARITY. EVEN sequences -> per-month type means -> plane P = "
               "fits.pca_2d top-2 right-singular vectors of the 12 even means; plane centre = "
               "mean of the 12 even means. ODD sequences -> every statistic. Sequence parity, "
               "not row parity, so two occurrences inside one document cannot straddle the split "
               "(critic repair)."),
        projection="y = (x - centre) @ B.T with B the 2 x 1152 plane basis; no re-PCA per draw",
        positive_control=(
            "odd-sequence type means projected on P (the plane the draws face): batched Kasa "
            "circle fit -> circularity -> gap_p(12, max_gap, n_perm) must be < 0.05, AND "
            "cyclic_order_score(angles, calendar index, k=12, full_k=12) rho >= 0.9, at L6 AND "
            "L7. Failure => verdict 'instrument not validated', all numbers still emitted, "
            "nothing re-tuned (prereg R2). The free screen()-style fit (odd means given their "
            "OWN pca_2d plane) is co-reported and is NOT gate-bearing."),
        primary_statistic=(
            "S_L = fraction of n_draws draws that pass closure. Each draw takes one odd-sequence "
            "row per month uniformly at random and is REJECTED and redrawn if two picked rows "
            "share a sequence id; the 12 rows are projected on P, a circle is fitted to those 12 "
            "in-plane points, and the draw passes iff gap_p(12, max_gap, n_perm) < 0.05 against "
            "the cached uniform-angle null. Median gap_ratio and median per-draw rho co-reported."),
        per_draw_rho=("best Spearman over the 24 rotation/reflection symmetries of the 12-cycle, "
                      "the same maximisation fits.cyclic_order_score performs, but WITHOUT its "
                      "permutation null -- running a 20,000-perm null inside 500k draws is the "
                      "one thing that would blow the runtime. The full cyclic_order_score (rho, "
                      "p, symmetry) is run on the positive control, which is what gates."),
        decision_null=(
            "identical draw procedure; then each picked row's IN-PLANE RESIDUAL (its 2-D "
            "projection minus its own month's ODD-sequence mean projection) is replaced by the "
            "in-plane residual of a donor row drawn uniformly from ALL odd rows of the 11 OTHER "
            "months (donors are not restricted to the 12 picked rows and carry no sequence "
            "constraint -- declared). Point = own-month odd-mean + other-month residual. This "
            "preserves the twelve means, the plane and the residual distribution and destroys "
            "only occurrence-to-type correspondence (critic repair)."),
        decision_rule=(
            f"S_L is a single number per block of draws, so the null distribution is built as "
            f"{N_NULL_BLOCKS} blocks of {NULL_BLOCK_DRAWS} null draws -> {N_NULL_BLOCKS} block S "
            f"values. S_L is reported both as the {N_DRAWS}-draw point estimate and as the mean "
            f"over {N_DRAWS // NULL_BLOCK_DRAWS} same-sized blocks with its own block spread. "
            "POSITIVE iff the n_draws point estimate exceeds the 99th percentile (np.percentile, "
            "linear interpolation) of the null block S values. Decision arm = raw, decision "
            f"layer = L{PRIMARY_LAYER}. A layer/arm whose positive control failed emits its "
            "numbers with decision 'instrument not validated'."),
        secondary_null=(
            f"{N_BG_TYPES} non-calendar pool types drawn with seed {SEED} by repeated calls to "
            "gfdisc.nulls.frequency_matched_resample(cluster = the 12 month primary ids, freqs = "
            f"pool counts, tol {FREQ_TOL_LOG10}); already-taken and calendar types are excluded "
            "by zeroing their frequency (which puts them outside every log10 band) so the "
            "library function is used unmodified. Calendar exclusion = every id in "
            "results/token_map.json months and days PLUS the 13-member set of prereg section 0 "
            f"(so ' Feb' 5656 cannot be drawn). Then {N_BG_SETS} random 12-type sets, each "
            f"fitting its OWN even-sequence plane and running {N_BG_DRAWS} draws through the "
            "identical procedure. Reported, not decision-bearing."),
        power=(
            "population-level plant: for each odd row ONE seeded noise realisation per rho "
            "replaces its in-plane coordinates by its month's EVEN-mean in-plane coordinates plus "
            "isotropic Gaussian noise with sigma = rho * R_ring / sqrt(2) per coordinate, so the "
            "planted RMS displacement radius is exactly rho * R_ring. R_ring = mean distance of "
            "the 12 even-mean in-plane points from the fitted circle centre. The standard draw "
            "index matrix is then re-used unchanged over the planted population, so the power "
            "curve and S_L see the same draws. rho* = median in-plane residual radius / R_ring "
            "(task text); the prereg's off-plane variant (median ||x - centre|| component "
            "orthogonal to P / R_ring) is co-reported. rho at 80% detection by linear "
            "interpolation in log10(rho) on the sorted-by-rho curve."),
        fidelity_gate=(
            "READ, never recomputed: results/phase3_ring_readers_gonogo.json part_a_prime. Gate "
            f"= median month-row reconstruction cosine (WITH b_dec) >= {FIDELITY_GATE_MIN} at "
            f"L{PRIMARY_LAYER}, the quantity prereg section 2.2 A' names. If the gate is not met, "
            "any negative is labelled 'SAE-fidelity-limited' and is not a model statement."),
        trajectory_tier=("EXPLORATORY, NON-VERDICT. Per-layer aligned angle of every odd month "
                         "row in that layer's own plane, with the frame fixed by January's even "
                         "mean at 0 and the calendar running counter-clockwise -- labels enter "
                         "the FRAME only (prereg R3), never a decision statistic. Rows whose "
                         "in-plane radius is degenerate at any layer are excluded and counted."),
        batched_arithmetic=(
            "the batched Kasa fit solves the normal equations that fits.fit_circle solves by "
            "lstsq, and the batched max-gap repeats fits.circularity. Both are re-checked against "
            "the frozen scalar functions on a random sample of the real draws; the maximum "
            "disagreement and any pass/fail disagreement are reported in batch_validation. The "
            "vectorised gap_p (searchsorted, tie convention g <= max_gap) is checked against the "
            "copied scalar gap_p on sampled values."),
        seeds=(f"one numpy default_rng per statistic block, all derived from seed {SEED}: "
               "background type draw (0); per-layer draws 1000*L; decision null 1000*L + 500000; "
               "power 1000*L + 900000 + rho index; background sets 700000 + set index. The draw "
               "streams are deliberately INDEPENDENT OF THE ARM, so all three arms see the same "
               "draws (a paired comparison) and the rms-vs-raw equivalence is an EXACT identity "
               "rather than a statistical one. Declared so a verifier can reproduce any single "
               "block without running the rest."),
        memory=("streamed once per layer; only odd-sequence background rows are kept (fp32, "
                "~0.83 GB) and are freed before the next layer; even-sequence rows are never "
                "stored, only their per-type sums. This is a declared reading of the 'stream per "
                "type-set draw OR per-type in-plane coordinates' instruction: per-type in-plane "
                "coordinates cannot exist because every background set fits its own plane, and "
                "per-set streaming re-reads each type ~7.5x for no memory benefit. Peak RSS is "
                "measured and reported."),
        no_forward_passes=("this script runs none. Fidelity comes from scripts/35. "
                           "HF_HUB_OFFLINE is set before any gfdisc import; the only network-"
                           "shaped call is load_sae, which reads the local cache/hf snapshot."),
        config=dict(layers=layers, primary_layer=PRIMARY_LAYER, n_draws=int(a.n_draws),
                    n_bg_sets=int(a.n_bg_sets), n_bg_draws=int(a.n_bg_draws),
                    n_perm=int(a.n_perm), n_null_blocks=int(a.n_null_blocks),
                    null_block_draws=int(a.null_block_draws), rhos=list(RHOS)),
    )


# ===========================================================================
# COPIED VERBATIM from scripts/22_token_type_geometry.py L364-376
# ===========================================================================
_NULL = {}


def gap_null(k, n_perm, seed=0):
    """The uniform-angle max-gap null of scripts/13_rerun_gate.gap_pvalue, cached per k."""
    key = (int(k), int(n_perm), int(seed))
    if key not in _NULL:
        rng = np.random.default_rng(seed)
        a = np.sort(rng.uniform(0, 2 * np.pi, size=(n_perm, k)), axis=1)
        _NULL[key] = np.diff(np.concatenate([a, a[:, :1] + 2 * np.pi], axis=1), axis=1).max(1)
    return _NULL[key]


def gap_p(k, max_gap_rad, n_perm, seed=0):
    g = gap_null(k, n_perm, seed)
    return float((1 + (g <= max_gap_rad).sum()) / (1 + n_perm))


# --- vectorised form of the same test (validated against the two functions above) ---
_NULL_SORTED = {}


def gap_null_sorted(k, n_perm, seed=0):
    key = (int(k), int(n_perm), int(seed))
    if key not in _NULL_SORTED:
        _NULL_SORTED[key] = np.sort(gap_null(k, n_perm, seed))
    return _NULL_SORTED[key]


def gap_p_vec(k, max_gap_rad, n_perm, seed=0):
    """gap_p for an array of max gaps. `side='right'` reproduces the `g <= mg` tie convention."""
    gs = gap_null_sorted(k, n_perm, seed)
    c = np.searchsorted(gs, np.asarray(max_gap_rad, dtype=np.float64), side="right")
    return (1.0 + c) / (1.0 + n_perm)


# ===========================================================================
# COPIED from scripts/22_token_type_geometry.py L211 (build_pool)
# ===========================================================================
def build_pool(root, layer0=0):
    """Token types over the count floor. tokens.i32 is byte-identical across layers."""
    tok = np.fromfile(Path(root) / f"layer_{layer0}" / "tokens.i32", dtype=np.int32)
    cnt = np.bincount(tok)
    main = np.flatnonzero(cnt >= COUNT_FLOOR)
    return tok, cnt, main


# ===========================================================================
# batched closure test (validated against fits.fit_circle / fits.circularity)
# ===========================================================================
def batch_fit_circle(P):
    """Kasa fit for a batch of point sets. P: [B, k, 2] -> (cx, cy, r), each [B].

    fits.fit_circle solves the same least-squares problem with lstsq; here the 3x3 normal
    equations are solved directly so the whole batch is one np.linalg.solve. Any singular
    system falls back to the frozen scalar routine and is counted.
    """
    x, y = P[..., 0], P[..., 1]
    b = x ** 2 + y ** 2
    k = P.shape[1]
    Sx, Sy = x.sum(1), y.sum(1)
    A = np.empty(P.shape[:1] + (3, 3), dtype=np.float64)
    A[:, 0, 0] = 4 * (x * x).sum(1); A[:, 0, 1] = 4 * (x * y).sum(1); A[:, 0, 2] = 2 * Sx
    A[:, 1, 0] = A[:, 0, 1];         A[:, 1, 1] = 4 * (y * y).sum(1); A[:, 1, 2] = 2 * Sy
    A[:, 2, 0] = 2 * Sx;             A[:, 2, 1] = 2 * Sy;             A[:, 2, 2] = float(k)
    rhs = np.stack([2 * (x * b).sum(1), 2 * (y * b).sum(1), b.sum(1)], axis=1)
    n_fallback = 0
    try:
        # numpy >= 2.0 reads a 2-D second operand as a stack of MATRICES, so the right-hand
        # sides go in as [B, 3, 1] and come back squeezed.
        sol = np.linalg.solve(A, rhs[..., None])[..., 0]
    except np.linalg.LinAlgError:
        sol = np.empty((len(P), 3), dtype=np.float64)
        for i in range(len(P)):
            try:
                sol[i] = np.linalg.solve(A[i], rhs[i])
            except np.linalg.LinAlgError:
                cx, cy, r, _ = fit_circle(P[i]); sol[i] = (cx, cy, r ** 2 - cx ** 2 - cy ** 2)
                n_fallback += 1
    cx, cy = sol[:, 0], sol[:, 1]
    r = np.sqrt(np.maximum(sol[:, 2] + cx ** 2 + cy ** 2, 1e-12))
    return cx, cy, r, n_fallback


def batch_max_gap(P, cx, cy):
    """Largest angular gap (radians) about (cx, cy) for a batch. Repeats fits.circularity."""
    th = np.sort(np.arctan2(P[..., 1] - cy[:, None], P[..., 0] - cx[:, None]), axis=1)
    g = np.diff(np.concatenate([th, th[:, :1] + 2 * np.pi], axis=1), axis=1)
    return g.max(1), th


_SYMS = None


def cycle_symmetry_ranks(k=12):
    """The 2k reference rank vectors fits.cyclic_order_score maximises over, precomputed."""
    global _SYMS
    if _SYMS is None:
        lab = np.arange(k)
        rows = []
        for flip in (1, -1):
            for shift in range(k):
                v = (flip * lab + shift) % k
                order = np.argsort(v, kind="stable")
                r = np.empty(k, dtype=np.float64); r[order] = np.arange(k, dtype=np.float64)
                rows.append(r)
        _SYMS = np.asarray(rows)                      # [2k, k]
    return _SYMS


def batch_rho(P, cx, cy, k=12):
    """Best Spearman over the 2k cycle symmetries, per draw. No permutation null (declared)."""
    ang = np.arctan2(P[..., 1] - cy[:, None], P[..., 0] - cx[:, None])
    order = np.argsort(ang, axis=1, kind="stable")
    rank = np.empty_like(order, dtype=np.float64)
    np.put_along_axis(rank, order, np.arange(k, dtype=np.float64)[None, :].repeat(len(P), 0), 1)
    S = cycle_symmetry_ranks(k)
    d = rank[:, None, :] - S[None, :, :]
    rho = 1.0 - 6.0 * (d ** 2).sum(-1) / (k * (k * k - 1))
    return rho.max(1)


def batch_closure(P, n_perm, k=12):
    """gap_ratio, max_gap_deg, p_closed, pass for a batch of k-point sets in a fixed plane."""
    cx, cy, r, nfb = batch_fit_circle(P)
    mg, _ = batch_max_gap(P, cx, cy)
    p = gap_p_vec(k, mg, n_perm, seed=0)
    return dict(cx=cx, cy=cy, r=r, max_gap=mg, gap_ratio=mg / (2 * np.pi / k), p=p,
                passed=p < ALPHA, n_fallback=nfb)


# ===========================================================================
# draw machinery
# ===========================================================================
def draw_indices(counts, seqs, n, rng, seq_len=SEQ_LEN):
    """[n, 12] indices, one odd row per month, redrawn while any two share a sequence id."""
    k = len(counts)
    idx = np.empty((n, k), dtype=np.int64)
    for m in range(k):
        idx[:, m] = rng.integers(0, counts[m], size=n)
    n_reject, rounds = 0, 0
    while rounds < MAX_REJECT_ROUNDS:
        sid = np.stack([seqs[m][idx[:, m]] for m in range(k)], axis=1)
        srt = np.sort(sid, axis=1)
        bad = (np.diff(srt, axis=1) == 0).any(axis=1)
        nb = int(bad.sum())
        if nb == 0:
            break
        n_reject += nb
        for m in range(k):
            idx[bad, m] = rng.integers(0, counts[m], size=nb)
        rounds += 1
    return idx, n_reject, rounds


def gather_points(Y_by_month, idx):
    """[n, 12, 2] in-plane points from per-month coordinate arrays and an index matrix."""
    return np.stack([Y_by_month[m][idx[:, m]] for m in range(idx.shape[1])], axis=1)


def block_stats(passed, block_size):
    """Point estimate plus the block-level distribution of the same pass fraction."""
    n_blocks = len(passed) // block_size
    blocks = passed[:n_blocks * block_size].reshape(n_blocks, block_size).mean(1) \
        if n_blocks else np.asarray([])
    out = dict(S=float(np.mean(passed)), n_draws=int(len(passed)), n_blocks=int(n_blocks),
               block_size=int(block_size))
    if n_blocks:
        out.update(block_mean=float(blocks.mean()), block_sd=float(blocks.std(ddof=1))
                   if n_blocks > 1 else 0.0,
                   block_pct={str(q): float(np.percentile(blocks, q))
                              for q in (1, 5, 25, 50, 75, 95, 99)},
                   block_min=float(blocks.min()), block_max=float(blocks.max()))
    return out, blocks


# ===========================================================================
# cache streaming
# ===========================================================================
def stream_type_rows(reader, W, rows_by_type, want_odd_rows, want_sumsq, log_every=200):
    """One pass over the CSR reading only the rows we need, grouped by token type.

    Returns per type: even raw sum, even L2-normalised sum, even count, and (for types in
    `want_odd_rows`) the fp32 reconstructions of the odd-sequence rows plus their norms.
    Even-sequence reconstructions are accumulated and discarded, never stored.
    """
    import scipy.sparse as sp
    ip = reader.indptr
    d_sae = int(reader.meta["d_sae"])
    types = list(rows_by_type.keys())
    n_odd_total = sum(int((r // SEQ_LEN % 2 == 1).sum()) for t, r in rows_by_type.items()
                      if t in want_odd_rows)
    X_odd = np.empty((n_odd_total, W.shape[1]), dtype=np.float32)
    off = 0
    out = {}
    sumsq, n_elem = 0.0, 0
    t0 = time.time()
    for ti, t in enumerate(types):
        rows = np.sort(np.asarray(rows_by_type[t], dtype=np.int64))
        starts = ip[rows]
        lens = (ip[rows + 1] - starts).astype(np.int64)
        cum = np.concatenate([[0], np.cumsum(lens)])
        flat = np.repeat(starts - cum[:-1], lens) + np.arange(cum[-1])
        idx = np.asarray(reader.indices[flat]).astype(np.int32)
        val = np.asarray(reader.values[flat]).astype(np.float32)
        M = sp.csr_matrix((val, idx, cum.astype(np.int64)), shape=(len(rows), d_sae))
        X = np.asarray(M @ W, dtype=np.float32)
        seq = rows // SEQ_LEN
        even = (seq % 2) == 0
        nrm = np.linalg.norm(X, axis=1)
        nrm_safe = np.maximum(nrm, 1e-12)
        if t in want_sumsq:
            sumsq += float((X.astype(np.float64) ** 2).sum()); n_elem += X.size
        rec = dict(sum_raw=X[even].astype(np.float64).sum(0),
                   sum_l2=(X[even] / nrm_safe[even, None]).astype(np.float64).sum(0),
                   n_even=int(even.sum()), n_odd=int((~even).sum()),
                   odd_rows=rows[~even], odd_seq=seq[~even],
                   odd_norm=nrm[~even].astype(np.float32))
        if t in want_odd_rows:
            n = int((~even).sum())
            X_odd[off:off + n] = X[~even]
            rec["odd_slice"] = (off, off + n)
            off += n
        else:
            rec["odd_slice"] = None
        out[t] = rec
        del X, M, idx, val, flat
        if log_every and (ti + 1) % log_every == 0:
            print(f"    streamed {ti + 1}/{len(types)} types  {time.time() - t0:.1f}s",
                  flush=True)
    rms = float(np.sqrt(sumsq / max(n_elem, 1))) if n_elem else float("nan")
    return out, X_odd, rms, float(time.time() - t0)


# ===========================================================================
# per-arm geometry
# ===========================================================================
def arm_even_means(recs, types, arm, rms_scalar):
    E = np.empty((len(types), D_MODEL), dtype=np.float64)
    for i, t in enumerate(types):
        r = recs[t]
        if arm == "l2":
            E[i] = r["sum_l2"] / max(r["n_even"], 1)
        elif arm == "rms":
            E[i] = (r["sum_raw"] / max(r["n_even"], 1)) / rms_scalar
        else:
            E[i] = r["sum_raw"] / max(r["n_even"], 1)
    return E


def plane_from(E):
    """Plane P: fits.pca_2d of the 12 even means. centre = their mean (what pca_2d subtracts)."""
    centre = E.mean(0)
    _, ratios, Vt = pca_2d(E)
    B = Vt[:2]                                   # [2, 1152]
    return centre, B, ratios


def project_rows(X, norms, arm, rms_scalar, centre, B):
    """(x/nf - centre) @ B.T computed as (x @ B.T)/nf - centre @ B.T (exactly equal, cheaper)."""
    P = np.asarray(X, dtype=np.float64) @ B.T
    if arm == "l2":
        P = P / np.maximum(np.asarray(norms, dtype=np.float64), 1e-12)[:, None]
    elif arm == "rms":
        P = P / rms_scalar
    return P - (centre @ B.T)[None, :]


def peak_rss_gb():
    ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return float(ru / 1e9) if sys.platform == "darwin" else float(ru / 1e6)


def circ_var(theta):
    """1 - |mean resultant|. 0 = all rows at one angle, 1 = uniform round the circle."""
    return float(1.0 - abs(np.exp(1j * np.asarray(theta)).mean()))


def interp_threshold(rhos, S, target):
    """Smallest rho at which S drops below `target`, by linear interpolation in log10(rho)."""
    r = np.asarray(rhos, dtype=float); s = np.asarray(S, dtype=float)
    o = np.argsort(r); r, s = r[o], s[o]
    if s[0] < target:
        return None, "S below target already at the smallest rho"
    for i in range(1, len(r)):
        if s[i] < target:
            lr = np.log10(r[i - 1]) + (np.log10(r[i]) - np.log10(r[i - 1])) * \
                (s[i - 1] - target) / max(s[i - 1] - s[i], 1e-12)
            return float(10 ** lr), "interpolated in log10(rho)"
    return None, "S stays at or above target across the whole rho grid"


# ===========================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="cache/sparse_big")
    ap.add_argument("--l0", default="big")
    ap.add_argument("--out", default="results/phase3_per_occurrence.json")
    ap.add_argument("--coords-out", default="results/phase3_per_occurrence_coords.npz")
    ap.add_argument("--gonogo", default="results/phase3_ring_readers_gonogo.json")
    ap.add_argument("--token-map", default="results/token_map.json")
    ap.add_argument("--layers", type=int, nargs="+", default=LAYERS)
    ap.add_argument("--n-draws", type=int, default=N_DRAWS)
    ap.add_argument("--n-bg-sets", type=int, default=N_BG_SETS)
    ap.add_argument("--n-bg-draws", type=int, default=N_BG_DRAWS)
    ap.add_argument("--n-perm", type=int, default=N_PERM)
    ap.add_argument("--n-null-blocks", type=int, default=N_NULL_BLOCKS)
    ap.add_argument("--null-block-draws", type=int, default=NULL_BLOCK_DRAWS)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    t_start = time.time()
    layers = sorted(set(int(x) for x in a.layers))
    smoke = bool(a.smoke or layers != LAYERS or a.n_draws != N_DRAWS
                 or a.n_bg_sets != N_BG_SETS or a.n_bg_draws != N_BG_DRAWS
                 or a.n_perm != N_PERM or a.n_null_blocks != N_NULL_BLOCKS
                 or a.null_block_draws != NULL_BLOCK_DRAWS)

    # ---- R1: declaration stub to disk BEFORE any cache row is read ----
    declared = dict(
        script="scripts/39_per_occurrence_ring.py",
        prereg="docs/phase3_prereg_2026-08-21.md section 4",
        design="docs/next_avenues_2026-08-21.md avenue 3 (five critic repairs)",
        status="DECLARATION STUB -- rules frozen, no results yet",
        declared_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
        smoke=smoke,
        question=("Does the month ring exist per occurrence, or only in the context-averaged "
                  "readout? One odd-sequence row per month, projected on a held-out even-sequence "
                  "plane, against a null that keeps the ring of means and destroys only the "
                  "occurrence-to-type correspondence."),
        no_forward_passes=True,
        config=dict(root=a.root, l0=a.l0, layers=layers, primary_layer=PRIMARY_LAYER,
                    primary_arm=PRIMARY_ARM, arms=list(ARMS), bg_arms=list(BG_ARMS),
                    n_draws=int(a.n_draws), n_bg_types=N_BG_TYPES, n_bg_sets=int(a.n_bg_sets),
                    n_bg_draws=int(a.n_bg_draws), n_perm=int(a.n_perm),
                    n_perm_order=N_PERM_ORDER, n_null_blocks=int(a.n_null_blocks),
                    null_block_draws=int(a.null_block_draws), rhos=list(RHOS), seed=SEED,
                    coords_out=str(a.coords_out), gonogo=str(a.gonogo)),
        declared_rules=declared_rules(a, layers),
    )
    outp = Path(a.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(declared, indent=2))
    print(f"declaration stub -> {outp}", flush=True)
    if smoke:
        print("SMOKE RUN -- scaled-down instrument check, not the registered measurement",
              flush=True)

    # -----------------------------------------------------------------------
    # fidelity gate: READ from scripts/35, never recomputed (no forwards here)
    # -----------------------------------------------------------------------
    gg = json.loads(Path(a.gonogo).read_text())
    pap = gg["part_a_prime"]
    L7 = pap["per_layer"][str(PRIMARY_LAYER)]
    fidelity = dict(
        source=f"{a.gonogo} :: part_a_prime",
        source_declared_at=gg.get("declared_at"),
        gate=pap.get("gate"),
        gate_bearing_quantity=pap.get("gate_bearing_quantity"),
        gate_value=float(L7[pap.get("gate_bearing_quantity", "median_cos_fresh_bdec_month")]),
        gate_min=FIDELITY_GATE_MIN,
        gate_pass=bool(float(L7[pap.get("gate_bearing_quantity",
                                        "median_cos_fresh_bdec_month")]) >= FIDELITY_GATE_MIN),
        median_cos_fresh_bdec_month_L7=float(L7["median_cos_fresh_bdec_month"]),
        median_cos_cached_bdec_month_L7=float(L7["median_cos_cached_bdec_month"]),
        median_cos_cached_nobdec_month_L7=float(L7["median_cos_cached_nobdec_month"]),
        frac_identical_support_month_L7=float(L7["frac_identical_support_month"]),
        gonogo_gate_note=pap.get("gate_note"),
        b_dec_note=("this script reconstructs WITHOUT b_dec, as prereg section 4 requires. b_dec "
                    "is one constant vector added to every row alike, so it cancels exactly after "
                    "the centring that pca_2d and every projection here perform: no angle, gap, "
                    "radius or rho below changes by a floating-point ulp if it were added. The "
                    "no-b_dec cosine (0.334 at L7) is therefore a statement about a common-mode "
                    "offset this instrument removes by construction, not about the geometry it "
                    "measures; the gate-bearing number is the with-b_dec cosine the prereg names."),
        recomputed_here=False)
    print(f"fidelity gate (read, not recomputed): {fidelity['gate_bearing_quantity']} at L7 = "
          f"{fidelity['gate_value']:.5f} -> pass={fidelity['gate_pass']}", flush=True)

    # -----------------------------------------------------------------------
    # pool, strict-12, background 800
    # -----------------------------------------------------------------------
    t_pool = time.time()
    tok, cnt, pool_ids = build_pool(a.root)
    n_rows_cache = int(len(tok))
    tmap = json.loads(Path(a.token_map).read_text())
    pool_set = set(int(x) for x in pool_ids)

    # scripts/22 pack() rule: ids in the >=100 pool, sorted by descending count, take the first
    strict12, month_counts = [], {}
    for m in MONTHS:
        ids = [int(x) for x in tmap["months"][m]]
        in_pool = sorted((x for x in ids if x in pool_set), key=lambda x: -int(cnt[x]))
        if not in_pool:
            raise RuntimeError(f"no in-pool id for {m}")
        strict12.append(in_pool[0]); month_counts[m] = int(cnt[in_pool[0]])
    if sorted(strict12) != STRICT12_IDS_EXPECTED:
        raise RuntimeError(f"derived strict-12 {sorted(strict12)} != prereg section 0 exposure "
                           f"record {STRICT12_IDS_EXPECTED}")
    print(f"strict-12 ids {strict12} counts {[month_counts[m] for m in MONTHS]} "
          f"(asserted == prereg section 0)", flush=True)

    calendar_ids = set(MEMBER13_IDS)
    for key in ("months", "days"):
        for _w, ids in tmap[key].items():
            calendar_ids |= set(int(x) for x in ids)

    pos_of = {int(t): i for i, t in enumerate(pool_ids)}
    month_pos = [pos_of[t] for t in strict12]
    freq = cnt[pool_ids].astype(np.float64)
    work = freq.copy()
    for i, t in enumerate(pool_ids):
        if int(t) in calendar_ids and i not in month_pos:
            work[i] = 0.0
    rng_bg = np.random.default_rng(SEED)
    sel = []
    while len(sel) < N_BG_TYPES:
        got = [int(g) for g in frequency_matched_resample(month_pos, work, rng_bg,
                                                          tol=FREQ_TOL_LOG10) if work[g] > 0]
        if not got:
            break
        for g in got:
            if len(sel) < N_BG_TYPES:
                sel.append(g); work[g] = 0.0
    bg_types = [int(pool_ids[i]) for i in sel]
    bg_counts = np.asarray([int(cnt[t]) for t in bg_types])
    if len(bg_types) < N_BG_TYPES:
        print(f"WARNING: only {len(bg_types)} background types available", flush=True)
    background = dict(n_types=len(bg_types), tol_log10=FREQ_TOL_LOG10, seed=SEED,
                      count_min=int(bg_counts.min()), count_median=float(np.median(bg_counts)),
                      count_max=int(bg_counts.max()), total_rows=int(bg_counts.sum()),
                      month_count_min=int(min(month_counts.values())),
                      month_count_max=int(max(month_counts.values())),
                      n_calendar_ids_excluded=len(calendar_ids),
                      feb_5656_excluded=bool(5656 in calendar_ids))
    print(f"background: {len(bg_types)} types, counts {background['count_min']}-"
          f"{background['count_max']}, {background['total_rows']} rows", flush=True)

    all_types = strict12 + bg_types
    reader0 = SparseReader(a.root, layers[0])
    pos_all = reader0.positions_of_tokens(all_types)          # the post-hoc label index
    t_at = np.asarray(reader0.tokens)[pos_all]
    rows_by_type = {int(t): pos_all[t_at == t] for t in all_types}
    for i, t in enumerate(strict12):
        if len(rows_by_type[t]) != month_counts[MONTHS[i]]:
            raise RuntimeError(f"positions_of_tokens count mismatch for {MONTHS[i]}")
    month_odd_counts = [int(((rows_by_type[t] // SEQ_LEN) % 2 == 1).sum()) for t in strict12]
    print(f"month odd-sequence rows per month: {month_odd_counts} "
          f"(total {sum(month_odd_counts)})", flush=True)
    pool_info = dict(n_cache_rows=n_rows_cache, n_sequences=n_rows_cache // SEQ_LEN,
                     n_pool_types=int(len(pool_ids)), count_floor=COUNT_FLOOR,
                     strict12_ids=strict12, strict12_names=list(MONTHS),
                     strict12_counts=[month_counts[m] for m in MONTHS],
                     strict12_odd_counts=month_odd_counts,
                     strict12_even_counts=[int(len(rows_by_type[t])) - o
                                           for t, o in zip(strict12, month_odd_counts)],
                     asserted_against_prereg_section0=True)
    t_pool = time.time() - t_pool

    # -----------------------------------------------------------------------
    per_layer, coords, timings = {}, {}, dict(pool_seconds=t_pool)
    traj_angles, traj_ok = {}, None
    bg_reduced = False
    for L in layers:
        print(f"\n=== layer {L} ===", flush=True)
        t_layer = time.time()
        sae = load_sae(L, l0=a.l0)
        W = get_decoder(sae).astype(np.float32)
        del sae
        reader = SparseReader(a.root, L)
        recs, X_bg, rms_scalar, t_stream = stream_type_rows(
            reader, W, rows_by_type, want_odd_rows=set(bg_types), want_sumsq=set(strict12))
        print(f"  streamed {len(all_types)} types in {t_stream:.1f}s  "
              f"(background odd rows {X_bg.shape[0]}, {X_bg.nbytes / 1e9:.2f} GB); "
              f"rms(month rows) = {rms_scalar:.4f}; peak RSS {peak_rss_gb():.2f} GB", flush=True)

        # month odd rows: reconstructed and KEPT (2,930 rows, 13 MB -- all three arms need them)
        import scipy.sparse as sp
        ip = reader.indptr
        d_sae = int(reader.meta["d_sae"])
        Xm_parts, m_rows, m_month, m_seq = [], [], [], []
        for mi, t in enumerate(strict12):
            rows = recs[t]["odd_rows"]
            starts = ip[rows]; lens = (ip[rows + 1] - starts).astype(np.int64)
            cum = np.concatenate([[0], np.cumsum(lens)])
            flat = np.repeat(starts - cum[:-1], lens) + np.arange(cum[-1])
            M = sp.csr_matrix((np.asarray(reader.values[flat]).astype(np.float32),
                               np.asarray(reader.indices[flat]).astype(np.int32),
                               cum.astype(np.int64)), shape=(len(rows), d_sae))
            Xm_parts.append(np.asarray(M @ W, dtype=np.float32))
            m_rows.append(rows); m_month.append(np.full(len(rows), mi, dtype=np.int32))
            m_seq.append(rows // SEQ_LEN)
            del M
        del W
        Xm = np.concatenate(Xm_parts); del Xm_parts
        m_rows = np.concatenate(m_rows); m_month = np.concatenate(m_month)
        m_seq = np.concatenate(m_seq); m_norm = np.linalg.norm(Xm, axis=1)
        month_slices = []
        o = 0
        for c in month_odd_counts:
            month_slices.append((o, o + c)); o += c

        layer_out = dict(rms_scalar=rms_scalar, n_month_odd_rows=int(len(Xm)),
                         n_background_odd_rows=int(X_bg.shape[0]),
                         stream_seconds=t_stream, arms={})
        for ai, arm in enumerate(ARMS):
            t_arm = time.time()
            E = arm_even_means(recs, strict12, arm, rms_scalar)
            centre, B, ratios = plane_from(E)
            YE = (E - centre) @ B.T                                   # 12 even means in-plane
            Ym = project_rows(Xm, m_norm, arm, rms_scalar, centre, B)
            Y_by_month = [Ym[s:e] for s, e in month_slices]
            seq_by_month = [m_seq[s:e] for s, e in month_slices]
            counts_m = np.asarray([len(y) for y in Y_by_month])

            # --- odd-sequence type means (the object the positive control tests) ---
            O = np.empty((12, D_MODEL), dtype=np.float64)
            for mi, (s, e) in enumerate(month_slices):
                Xs = np.asarray(Xm[s:e], dtype=np.float64)
                if arm == "l2":
                    Xs = Xs / np.maximum(m_norm[s:e, None], 1e-12)
                elif arm == "rms":
                    Xs = Xs / rms_scalar
                O[mi] = Xs.mean(0)
            YO = (O - centre) @ B.T
            YObar = np.stack([Y_by_month[m].mean(0) for m in range(12)])   # == YO, kept explicit

            # positive control, gate-bearing: odd means ON THE EVEN PLANE
            cx, cy, r, _ = batch_fit_circle(YO[None])
            mg, _ = batch_max_gap(YO[None], cx, cy)
            p_proj = gap_p(12, float(mg[0]), a.n_perm, seed=0)
            ang_proj = angles(YO, float(cx[0]), float(cy[0]))
            cos_proj = cyclic_order_score(ang_proj, np.arange(12), 12,
                                          n_perm=N_PERM_ORDER, seed=SEED, full_k=12)
            # free screen()-style co-report: odd means given their OWN pca_2d plane
            xy_free, ratios_free, _ = pca_2d(O)
            fcx, fcy, fr, fresid = fit_circle(xy_free)
            circ_free = circularity(xy_free, fcx, fcy, fr)
            p_free = gap_p(12, np.radians(circ_free["max_gap_deg"]), a.n_perm, seed=0)
            cos_free = cyclic_order_score(angles(xy_free, fcx, fcy), np.arange(12), 12,
                                          n_perm=N_PERM_ORDER, seed=SEED, full_k=12)
            ctrl_pass = bool(p_proj < ALPHA and cos_proj["rho"] >= CONTROL_RHO_MIN)
            control = dict(
                on_even_plane=dict(gap_ratio=float(mg[0] / (2 * np.pi / 12)),
                                   max_gap_deg=float(np.degrees(mg[0])), p_closed=p_proj,
                                   radius=float(r[0]), rho=float(cos_proj["rho"]),
                                   rho_p=float(cos_proj["p"]),
                                   symmetry=list(cos_proj["symmetry"]),
                                   passed=ctrl_pass, gate_bearing=True),
                free_plane=dict(gap_ratio=float(circ_free["gap_ratio"]),
                                max_gap_deg=float(circ_free["max_gap_deg"]), p_closed=p_free,
                                radius_ratio=float(circ_free["radius_ratio"]),
                                circle_resid_over_r=float(fresid),
                                top2_var=float(ratios_free[:2].sum()),
                                rho=float(cos_free["rho"]), rho_p=float(cos_free["p"]),
                                gate_bearing=False),
                even_plane=dict(top2_var=float(ratios[:2].sum()), pc1_var=float(ratios[0]),
                                gap_ratio_of_even_means=None),
            )
            ecx, ecy, er, _ = batch_fit_circle(YE[None])
            emg, _ = batch_max_gap(YE[None], ecx, ecy)
            control["even_plane"]["gap_ratio_of_even_means"] = float(emg[0] / (2 * np.pi / 12))
            control["even_plane"]["p_closed_of_even_means"] = gap_p(12, float(emg[0]),
                                                                   a.n_perm, seed=0)
            R_ring = float(np.linalg.norm(YE - np.array([ecx[0], ecy[0]]), axis=1).mean())
            control["even_plane"]["R_ring"] = R_ring
            control["even_plane"]["r_fit"] = float(er[0])

            # --- primary statistic S_L ---
            rng = np.random.default_rng(SEED + 1000 * L)          # arm-independent: paired draws
            idx, n_rej, n_rounds = draw_indices(counts_m, seq_by_month, int(a.n_draws), rng)
            P = gather_points(Y_by_month, idx)
            cl = batch_closure(P, a.n_perm)
            rho_draw = batch_rho(P, cl["cx"], cl["cy"])
            S_stats, S_blocks = block_stats(cl["passed"], int(a.null_block_draws))
            primary = dict(S_stats, n_rejected_draws=int(n_rej), n_reject_rounds=int(n_rounds),
                           median_gap_ratio=float(np.median(cl["gap_ratio"])),
                           gap_ratio_pct={str(q): float(np.percentile(cl["gap_ratio"], q))
                                          for q in (5, 25, 50, 75, 95)},
                           median_rho=float(np.median(rho_draw)),
                           rho_pct={str(q): float(np.percentile(rho_draw, q))
                                    for q in (5, 25, 50, 75, 95)},
                           n_circle_fallbacks=int(cl["n_fallback"]))

            # --- decision null ---
            donors = [np.concatenate([np.arange(*month_slices[j]) for j in range(12) if j != m])
                      for m in range(12)]
            resid = Ym - YObar[m_month]                      # in-plane residual of every odd row
            rng_n = np.random.default_rng(SEED + 1000 * L + 500000)
            n_null = int(a.n_null_blocks) * int(a.null_block_draws)
            nidx, n_rej_n, n_rounds_n = draw_indices(counts_m, seq_by_month, n_null, rng_n)
            Pn = np.empty((n_null, 12, 2), dtype=np.float64)
            for m in range(12):
                d = donors[m][rng_n.integers(0, len(donors[m]), size=n_null)]
                Pn[:, m, :] = YObar[m][None, :] + resid[d]
            cln = batch_closure(Pn, a.n_perm)
            null_blocks = cln["passed"].reshape(int(a.n_null_blocks),
                                                int(a.null_block_draws)).mean(1)
            thr99 = float(np.percentile(null_blocks, 99))
            decision_null = dict(n_blocks=int(a.n_null_blocks),
                                 block_size=int(a.null_block_draws),
                                 S_null_point=float(cln["passed"].mean()),
                                 block_mean=float(null_blocks.mean()),
                                 block_sd=float(null_blocks.std(ddof=1)),
                                 block_pct={str(q): float(np.percentile(null_blocks, q))
                                            for q in (1, 5, 25, 50, 75, 95, 99, 99.9)},
                                 block_max=float(null_blocks.max()),
                                 threshold_p99=thr99,
                                 n_rejected_draws=int(n_rej_n),
                                 exceedance_of_S=float((null_blocks >= S_stats["S"]).mean()),
                                 note=("the picked row's identity does not survive the swap -- a "
                                       "null point is own-month odd-mean + a donor's residual -- "
                                       "so the sequence-uniqueness rejection is run for procedural "
                                       "symmetry with the primary and its count is reported, but "
                                       "only the 12 MONTH slots and the donor draw affect the "
                                       "statistic. Donors carry no sequence constraint (declared); "
                                       "the primary's rejection count shows how rarely that "
                                       "constraint binds at all."))
            del Pn, cln

            # --- spreads ---
            centre_xy = np.array([ecx[0], ecy[0]])
            spreads = {}
            for m in range(12):
                Ymm = Y_by_month[m]
                th = np.arctan2(Ymm[:, 1] - centre_xy[1], Ymm[:, 0] - centre_xy[0])
                rad = np.linalg.norm(Ymm - centre_xy, axis=1)
                res = np.linalg.norm(Ymm - YObar[m], axis=1)
                spreads[MONTHS[m]] = dict(
                    n=int(len(Ymm)), circular_variance=circ_var(th),
                    mean_radius=float(rad.mean()), sd_radius=float(rad.std()),
                    radius_over_R_ring=float(rad.mean() / max(R_ring, 1e-12)),
                    median_inplane_residual=float(np.median(res)),
                    residual_over_R_ring=float(np.median(res) / max(R_ring, 1e-12)))
            all_res = np.linalg.norm(resid, axis=1)
            Xarm = np.asarray(Xm, dtype=np.float64)
            if arm == "l2":
                Xarm = Xarm / np.maximum(m_norm[:, None], 1e-12)
            elif arm == "rms":
                Xarm = Xarm / rms_scalar
            d_full = Xarm - (O[m_month])
            off_plane = np.sqrt(np.maximum((d_full ** 2).sum(1)
                                           - (np.linalg.norm(resid, axis=1) ** 2), 0.0))
            del Xarm, d_full
            rho_star_in = float(np.median(all_res) / max(R_ring, 1e-12))
            rho_star_off = float(np.median(off_plane) / max(R_ring, 1e-12))
            spread_summary = dict(
                per_month=spreads,
                median_inplane_residual_radius=float(np.median(all_res)),
                median_offplane_residual_radius=float(np.median(off_plane)),
                R_ring=R_ring,
                rho_star_inplane=rho_star_in, rho_star_offplane=rho_star_off,
                mean_circular_variance=float(np.mean([v["circular_variance"]
                                                      for v in spreads.values()])))

            # --- power ---
            power_rhos = sorted(set([float(x) for x in RHOS] + [rho_star_in]))
            curve = []
            for ri, rr in enumerate(power_rhos):
                rng_p = np.random.default_rng(SEED + 1000 * L + 900000 + ri)
                sigma = rr * R_ring / np.sqrt(2.0)
                Yp = YE[m_month] + rng_p.normal(scale=sigma, size=Ym.shape)
                Yp_by_month = [Yp[s:e] for s, e in month_slices]
                Pp = gather_points(Yp_by_month, idx)
                clp = batch_closure(Pp, a.n_perm)
                curve.append(dict(rho=float(rr), S=float(clp["passed"].mean()),
                                  is_rho_star=bool(rr == rho_star_in),
                                  median_gap_ratio=float(np.median(clp["gap_ratio"]))))
                del Pp, clp, Yp
            r80, r80_note = interp_threshold([c["rho"] for c in curve],
                                             [c["S"] for c in curve], POWER_TARGET)
            S_ceiling = float(control["even_plane"]["p_closed_of_even_means"] < ALPHA)
            power = dict(curve=curve, rho_at_80pct=r80, rho_at_80pct_note=r80_note,
                         S_ceiling_rho_0=S_ceiling,
                         S_ceiling_note=("rho = 0 puts all 12 drawn points exactly on the 12 "
                                         "even-mean in-plane positions, so every draw is the "
                                         "same point set and S is 0 or 1 according to whether "
                                         "the even means themselves close. This is the CEILING "
                                         "the power curve approaches, and it is below 1 whenever "
                                         "the held-out plane's own ring is not tight enough to "
                                         "clear the frozen gate -- read the curve against it, "
                                         "not against 1.0."),
                         rho_star_inplane=rho_star_in, rho_star_offplane=rho_star_off,
                         S_at_rho_star=float([c["S"] for c in curve
                                              if c["is_rho_star"]][0]) if any(
                             c["is_rho_star"] for c in curve) else None,
                         noise_convention="sigma = rho * R_ring / sqrt(2) per coordinate "
                                          "(planted RMS displacement radius = rho * R_ring)",
                         plant="population-level: one seeded realisation per odd row per rho, "
                               "then the SAME draw index matrix as S_L")

            # --- batched-arithmetic validation against the frozen scalar functions ---
            rngv = np.random.default_rng(SEED + 7)
            vs = rngv.choice(len(P), size=min(200, len(P)), replace=False)
            dmax, pmax, disagree = 0.0, 0.0, 0
            for j in vs:
                pts = P[j]
                fcx2, fcy2, fr2, _ = fit_circle(pts)
                cc = circularity(pts, fcx2, fcy2, fr2)
                dmax = max(dmax, abs(cc["max_gap_deg"] - np.degrees(cl["max_gap"][j])))
                ps = gap_p(12, np.radians(cc["max_gap_deg"]), a.n_perm, seed=0)
                pmax = max(pmax, abs(ps - cl["p"][j]))
                disagree += int((ps < ALPHA) != bool(cl["passed"][j]))
            n_degen = int(sum(degenerate(P[j]) for j in vs))
            validation = dict(n_sampled=int(len(vs)), max_abs_max_gap_deg_diff=float(dmax),
                              max_abs_p_diff=float(pmax), n_pass_disagreements=int(disagree),
                              n_degenerate_sampled=n_degen)

            # --- decision ---
            if not ctrl_pass:
                decision = "instrument not validated"
            elif S_stats["S"] > thr99:
                decision = "POSITIVE"
            else:
                decision = ("NEGATIVE" if fidelity["gate_pass"]
                            else "NEGATIVE (SAE-fidelity-limited)")
            layer_out["arms"][arm] = dict(
                positive_control=control, primary=primary, decision_null=decision_null,
                decision=decision, spreads=spread_summary, power=power,
                batch_validation=validation, seconds=float(time.time() - t_arm))
            print(f"  [{arm}] control p={p_proj:.5f} rho={cos_proj['rho']:.3f} "
                  f"pass={ctrl_pass} | S={S_stats['S']:.3f} null_p99={thr99:.3f} -> {decision} "
                  f"| rho*={rho_star_in:.3f} rho80={r80}", flush=True)

            if arm == PRIMARY_ARM:
                traj_angles[L] = dict(YE=YE.copy(), Ym=Ym.copy(), centre_xy=centre_xy.copy())
            if L == PRIMARY_LAYER:
                coords[f"Y_{arm}"] = Ym.astype(np.float32)
                coords[f"YE_{arm}"] = YE.astype(np.float32)
                coords[f"YObar_{arm}"] = YObar.astype(np.float32)
                coords[f"plane_B_{arm}_L{L}"] = B.astype(np.float32)
                coords[f"plane_centre_{arm}_L{L}"] = centre.astype(np.float32)
            coords[f"YE_{arm}_L{L}"] = YE.astype(np.float32)

            del P, resid

        # ---------------- secondary background null ----------------
        t_bg = time.time()
        bg_slices = {t: recs[t]["odd_slice"] for t in bg_types}
        bg_seq = {t: recs[t]["odd_seq"] for t in bg_types}
        bg_norm = {t: recs[t]["odd_norm"] for t in bg_types}
        rng_s = np.random.default_rng(SEED + 700000)
        n_sets = int(a.n_bg_sets)
        set_types = [rng_s.choice(len(bg_types), size=12, replace=False) for _ in range(n_sets)]
        bg_out = {}
        for arm in BG_ARMS:
            S_sets, n_used, t_arm0 = [], 0, time.time()
            probe_n = min(10, n_sets)
            for si, sidx in enumerate(set_types):
                if si >= n_sets:          # honours a mid-run reduction (see BG_COST_BUDGET_S)
                    break
                ts = [bg_types[int(z)] for z in sidx]
                Ebg = np.empty((12, D_MODEL), dtype=np.float64)
                for i, t in enumerate(ts):
                    r = recs[t]
                    Ebg[i] = (r["sum_l2"] if arm == "l2" else r["sum_raw"]) / max(r["n_even"], 1)
                cb, Bb, _ = plane_from(Ebg)
                shift = cb @ Bb.T
                Yb, sqb, cts = [], [], []
                for t in ts:
                    s0, s1 = bg_slices[t]
                    Pb = np.asarray(X_bg[s0:s1], dtype=np.float64) @ Bb.T
                    if arm == "l2":
                        Pb = Pb / np.maximum(np.asarray(bg_norm[t], dtype=np.float64),
                                             1e-12)[:, None]
                    Yb.append(Pb - shift[None, :]); sqb.append(bg_seq[t]); cts.append(s1 - s0)
                rng_d = np.random.default_rng(SEED + 700000 + si)
                bidx, _, _ = draw_indices(np.asarray(cts), sqb, int(a.n_bg_draws), rng_d)
                Pset = gather_points(Yb, bidx)
                S_sets.append(float(batch_closure(Pset, a.n_perm)["passed"].mean()))
                n_used += 1
                if si + 1 == probe_n:
                    rate = (time.time() - t_arm0) / probe_n
                    proj = rate * n_sets * len(BG_ARMS)
                    print(f"  [bg {arm}] {rate * 1000:.1f} ms/set -> projected "
                          f"{proj:.0f}s for {n_sets} sets x {len(BG_ARMS)} arms", flush=True)
                    if proj > BG_COST_BUDGET_S and n_sets > BG_SETS_REDUCED:
                        n_sets = BG_SETS_REDUCED
                        bg_reduced = True
                        print(f"  [bg] projected cost exceeds {BG_COST_BUDGET_S}s -> reduced to "
                              f"{BG_SETS_REDUCED} sets (recorded)", flush=True)
                del Pset, Yb
            Sarr = np.asarray(S_sets)
            SL = layer_out["arms"][arm]["primary"]["S"]
            bg_out[arm] = dict(
                n_sets=int(n_used), n_draws_per_set=int(a.n_bg_draws),
                S_mean=float(Sarr.mean()), S_sd=float(Sarr.std(ddof=1)) if len(Sarr) > 1 else 0.0,
                S_pct={str(q): float(np.percentile(Sarr, q))
                       for q in (1, 5, 25, 50, 75, 95, 99)},
                S_min=float(Sarr.min()), S_max=float(Sarr.max()),
                S_L=float(SL),
                percentile_of_S_L=float(100.0 * (Sarr <= SL).mean()),
                n_sets_at_or_above_S_L=int((Sarr >= SL).sum()),
                seconds=float(time.time() - t_arm0))
            coords[f"bg_S_{arm}_L{L}"] = Sarr.astype(np.float32)
            print(f"  [bg {arm}] S background mean {Sarr.mean():.3f} "
                  f"p50 {np.median(Sarr):.3f} p99 {np.percentile(Sarr, 99):.3f} | S_L {SL:.3f} "
                  f"-> percentile {bg_out[arm]['percentile_of_S_L']:.1f}", flush=True)
        bg_out["rms"] = dict(skipped=True, reason=(
            "the rms arm rescales every row by ONE positive per-layer scalar, which maps the "
            "plane, centre and all points by the same factor and leaves every angle and gap "
            "ratio invariant; its background is identical to raw's by construction. The month-arm "
            "statistics ARE run for rms and asserted equal to raw (arm_equivalence below)."))
        bg_out["seconds_total"] = float(time.time() - t_bg)
        bg_out["reduced_to_200_sets"] = bool(bg_reduced)
        bg_out["cost_budget_seconds"] = BG_COST_BUDGET_S
        layer_out["background"] = bg_out
        coords[f"bg_set_type_ids_L{L}"] = np.asarray(
            [[bg_types[int(z)] for z in s] for s in set_types[:n_sets]], dtype=np.int64)

        # rms == raw equivalence assertion (angle statistics are scale-invariant)
        eq = {}
        for key, path in (("control_gap_ratio", ("positive_control", "on_even_plane",
                                                 "gap_ratio")),
                          ("control_rho", ("positive_control", "on_even_plane", "rho")),
                          ("S", ("primary", "S")),
                          ("null_p99", ("decision_null", "threshold_p99"))):
            def dig(d, p):
                for q in p:
                    d = d[q]
                return d
            va, vb = dig(layer_out["arms"]["raw"], path), dig(layer_out["arms"]["rms"], path)
            eq[key] = dict(raw=float(va), rms=float(vb), abs_diff=float(abs(va - vb)))
        layer_out["arm_equivalence_rms_vs_raw"] = dict(
            checks=eq, max_abs_diff=float(max(v["abs_diff"] for v in eq.values())),
            note="rms is raw times a positive scalar; every angle statistic must match exactly")

        layer_out["seconds"] = float(time.time() - t_layer)
        layer_out["peak_rss_gb"] = peak_rss_gb()
        per_layer[str(L)] = layer_out
        timings[f"layer_{L}_seconds"] = layer_out["seconds"]
        if L == PRIMARY_LAYER:
            coords["month_row_index"] = m_rows.astype(np.int64)
            coords["month_index"] = m_month.astype(np.int32)
            coords["month_seq_id"] = m_seq.astype(np.int64)
            coords["month_token_id"] = np.asarray([strict12[i] for i in m_month], dtype=np.int64)
        del X_bg, Xm, recs, reader
        print(f"  layer {L} done in {layer_out['seconds']:.1f}s, peak RSS "
              f"{layer_out['peak_rss_gb']:.2f} GB", flush=True)

    # -----------------------------------------------------------------------
    # trajectory tier (exploratory, non-verdict)
    # -----------------------------------------------------------------------
    if sorted(traj_angles.keys()) != LAYERS:
        trajectory = dict(skipped=True, reason=f"needs all of {LAYERS}; ran {layers}",
                          tier="exploratory, non-verdict")
    else:
        aligned, excluded = {}, 0
        keep = np.ones(len(coords["month_index"]), dtype=bool)
        for L in LAYERS:
            YE, Ym, c0 = traj_angles[L]["YE"], traj_angles[L]["Ym"], traj_angles[L]["centre_xy"]
            keep &= np.linalg.norm(Ym - c0, axis=1) > 1e-9
        for L in LAYERS:
            YE, Ym, c0 = traj_angles[L]["YE"], traj_angles[L]["Ym"], traj_angles[L]["centre_xy"]
            thE = np.arctan2(YE[:, 1] - c0[1], YE[:, 0] - c0[0])
            step = np.diff(np.concatenate([thE, thE[:1]]))
            step = (step + np.pi) % (2 * np.pi) - np.pi
            sign = 1.0 if step.sum() >= 0 else -1.0        # FRAME only (prereg R3)
            th = np.arctan2(Ym[:, 1] - c0[1], Ym[:, 0] - c0[0])
            aligned[L] = ((sign * (th - thE[0]) + np.pi) % (2 * np.pi)) - np.pi
        mi = coords["month_index"]
        per_month = {}
        for m in range(12):
            sel_m = keep & (mi == m)
            per_month[MONTHS[m]] = {
                str(L): float(np.angle(np.exp(1j * aligned[L][sel_m]).mean()))
                for L in LAYERS}
        rot = {}
        for i in range(len(LAYERS) - 1):
            L0, L1 = LAYERS[i], LAYERS[i + 1]
            d = [((per_month[MONTHS[m]][str(L1)] - per_month[MONTHS[m]][str(L0)] + np.pi)
                  % (2 * np.pi)) - np.pi for m in range(12)]
            rot[f"{L0}->{L1}"] = dict(mean_deg=float(np.degrees(np.mean(d))),
                                      sd_deg=float(np.degrees(np.std(d))),
                                      per_month_deg={MONTHS[m]: float(np.degrees(d[m]))
                                                     for m in range(12)})
        trajectory = dict(tier="exploratory, non-verdict", arm=PRIMARY_ARM,
                          n_rows=int(keep.sum()), n_excluded_degenerate=int((~keep).sum()),
                          frame=("January's even mean at 0, calendar counter-clockwise; labels "
                                 "enter the FRAME only (prereg R3)"),
                          mean_aligned_angle_rad=per_month, per_layer_rotation=rot)
        for L in LAYERS:
            coords[f"aligned_angle_L{L}"] = aligned[L].astype(np.float32)
        coords["trajectory_keep"] = keep

    # -----------------------------------------------------------------------
    prim = per_layer.get(str(PRIMARY_LAYER), {}).get("arms", {}).get(PRIMARY_ARM)
    ctrl_ok = all(per_layer.get(str(L), {}).get("arms", {}).get(PRIMARY_ARM, {})
                  .get("positive_control", {}).get("on_even_plane", {}).get("passed", False)
                  for L in CONTROL_LAYERS if str(L) in per_layer)
    gate_evaluable = all(str(L) in per_layer for L in CONTROL_LAYERS)
    if not gate_evaluable:
        verdict = "gate not evaluable (control layers L6 and L7 both required)"
    elif not ctrl_ok:
        verdict = "instrument not validated"
    elif prim is None:
        verdict = "primary layer not run"
    else:
        verdict = prim["decision"]

    cp = Path(a.coords_out)
    cp.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cp, **coords)
    print(f"coords -> {cp}", flush=True)

    out = dict(declared)
    out.update(status="COMPLETE",
               finished_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
               verdict=verdict,
               gate_evaluable=bool(gate_evaluable),
               control_layers=CONTROL_LAYERS,
               primary=dict(layer=PRIMARY_LAYER, arm=PRIMARY_ARM,
                            S=prim["primary"]["S"] if prim else None,
                            null_p99=prim["decision_null"]["threshold_p99"] if prim else None,
                            decision=prim["decision"] if prim else None),
               fidelity_gate=fidelity,
               pool=pool_info,
               background_pool=background,
               per_layer=per_layer,
               trajectory=trajectory,
               timings=timings,
               peak_rss_gb=peak_rss_gb(),
               coords_npz=str(cp),
               runtime_seconds=float(time.time() - t_start))
    outp.write_text(json.dumps(out, indent=2, default=float))
    print(f"\nverdict: {verdict}", flush=True)
    print(f"peak RSS {out['peak_rss_gb']:.2f} GB", flush=True)
    print(f"wrote {outp} ({time.time() - t_start:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
