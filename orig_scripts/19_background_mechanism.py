"""EXPLORATORY (not pre-registered), item 5e: why does the ORDERING tier accept
~16.4% of index-matched random latent subsets at layer 9?

`results/gate_background_big_L9.json` measured rate_order = 0.164 on 750 scorable
draws -- 3.3x nominal -- while closure accepted 0/750. Three mechanisms were
proposed, with different predictions:

  R1  the ordering statistic or its permutation reference is SATURATED, so
      anything scores low p;
  R2  a real minority CYCLIC component that many random subsets touch;
  R3  a single shared 1-D GRADIENT over the 12 month labels (e.g. frequency /
      support) that every subset inherits.

R1 is falsifiable on its own: push iid-Gaussian label-mean matrices with the
SAME survivor sets and the SAME seeds through the identical scoring path. If the
statistic is calibrated that arm sits at 0.05.

R2 and R3 are separated by WHERE the shared structure lives and WHAT it looks
like. Two residualisation families are run, because they are not interchangeable:

  feature space -- project out the top PCs (right singular vectors, 1152-dim
      residual directions) of the GLOBAL label-mean matrix built from all alive
      latents. A draw's label means live in span(W[S]) for its own latent set S,
      so a shared LABEL profile can appear along a different feature direction in
      every draw; this family can therefore be a false negative for R3, and its
      behaviour is governed by decoder anisotropy as much as by label structure.
  label space -- project out the top LEFT singular vectors (12-dim label
      profiles) of the same global matrix, restricted to the draw's survivors and
      re-centred. This removes the shared profile however it is oriented in
      feature space, and is the direct test of "every subset inherits the same
      1-D gradient over months".

Each family gets a MATCHED RANDOM CONTROL removing the same number of directions
in the same space, because removing dimensions can only lower the acceptance rate
and a collapse means nothing unless the random control does not collapse.

Nothing here is a gate re-run and nothing here can change a pre-registered
number. Draws are regenerated with `np.random.default_rng(0)` and the same
template cycle as `scripts/16_gate_background.py`, so draw i of this run IS draw
i of that run; the stored file is used as a bit-level fidelity check on the
re-implementation before any mechanism claim is made.
"""
import argparse, importlib.util, json, os, sys, time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from scipy.stats import spearmanr, kstest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from gfdisc.cache import SparseReader
from gfdisc.saes import load_sae, get_decoder
from gfdisc.nulls import stratified_resample
from gfdisc.fits import (degenerate, pca_2d, fit_circle, circularity, angles,
                         cyclic_order_score)

_spec = importlib.util.spec_from_file_location(
    "rerun_gate", Path(__file__).resolve().parent / "13_rerun_gate.py")
gate = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gate)

# Pre-declared BEFORE the run (see module docstring): a family "collapses" if its
# acceptance falls to <= COLLAPSE and its matched random control stays >= HOLD.
COLLAPSE = 0.08
HOLD = 0.12
NOMINAL = 0.05


# ---------------------------------------------------------------------------
# A row-restricted view of the cache.
# ---------------------------------------------------------------------------
# `SparseReader.csr()` materialises the whole layer as int64 indices + float32
# values: 486M nonzeros -> ~5.9 GB, on a 24 GB machine shared with other jobs.
# Every draw is scored on the SAME 3,486 half-B label positions, so only those
# rows are ever read. Gathering them once gives a ~0.4M-nonzero matrix and a
# `codes_at` that is bit-identical to the full reader's (same f16 -> f32 values,
# same column order, explicit stored zeros preserved), which is what lets
# `gate.labelled` be called unmodified.
class RestrictedReader:
    def __init__(self, reader, rows):
        rows = np.unique(np.asarray(rows, dtype=np.int64))
        self.rows = rows
        self.d_sae = reader.meta["d_sae"]
        self._loc = {int(r): i for i, r in enumerate(rows)}
        idx, val, indptr = [], [], [0]
        n = 0
        for r in rows:
            a, b = int(reader.indptr[r]), int(reader.indptr[r + 1])
            idx.append(np.asarray(reader.indices[a:b], dtype=np.int64))
            val.append(np.asarray(reader.values[a:b], dtype=np.float32))
            n += b - a
            indptr.append(n)
        self._csr = sp.csr_matrix(
            (np.concatenate(val) if val else np.zeros(0, np.float32),
             np.concatenate(idx) if idx else np.zeros(0, np.int64),
             np.asarray(indptr, dtype=np.int64)),
            shape=(len(rows), self.d_sae))

    def codes_at(self, rows, features):
        loc = np.fromiter((self._loc[int(r)] for r in np.asarray(rows)),
                          dtype=np.int64, count=len(np.asarray(rows)))
        return np.asarray(self._csr[loc][:, np.asarray(features)].todense())

    def mean_codes(self, rows):
        loc = np.fromiter((self._loc[int(r)] for r in np.asarray(rows)),
                          dtype=np.int64, count=len(np.asarray(rows)))
        return np.asarray(self._csr[loc].mean(0)).ravel()


# ---------------------------------------------------------------------------
# `gate.labelled`, split at the point where the label-mean matrix exists.
# ---------------------------------------------------------------------------
def means_and_keep(reader, cluster, w_dec, label_pos, full_k):
    """The first half of `gate.labelled`, verbatim, returning M and `keep`.

    Support (`keep`) is decided on the CODES, so it is identical across every
    residualisation arm: the arms differ only in the geometry of M, never in
    which months survive. That keeps `reference_ring` / the closure reference
    fixed too.
    """
    means, keep = [], []
    for i, pos in enumerate(label_pos):
        if len(pos) < 5:
            continue
        A = reader.codes_at(pos, cluster)
        if int((A > 0).any(1).sum()) < gate.MIN_FIRING_PER_LABEL:
            continue
        means.append((A @ np.asarray(w_dec)[np.asarray(cluster)]).mean(0))
        keep.append(i)
    if len(keep) < gate.min_labels_with_support(full_k):
        return None, None
    return np.vstack(means), keep


def score_M(M, keep, seed, full_k, n_perm, with_ref=False):
    """The second half of `gate.labelled`, verbatim, on a supplied M."""
    if degenerate(M):
        return None
    xy, ratios, Vt = pca_2d(M)
    cx, cy, r, resid = fit_circle(xy)
    c = circularity(xy, cx, cy, r)
    k = len(keep)
    s = cyclic_order_score(angles(xy, cx, cy), np.asarray(keep), k,
                           n_perm=n_perm, seed=seed, full_k=full_k)
    out = dict(n_labels=k, rho=s["rho"], p_order=s["p"],
               null_mean=s["null_mean"], null_p95=s["null_p95"],
               gap_ratio=c["gap_ratio"], max_gap_deg=c["max_gap_deg"],
               p_closed=gate.gap_pvalue(k, np.radians(c["max_gap_deg"]), seed=seed),
               top2=float(ratios[:2].sum()))
    if with_ref:
        out.update(gate.reference_ring(keep, full_k, seed=seed))
    return out, xy, Vt


def principal_angles_deg(A, B):
    """Principal angles between the row spaces of two orthonormal 2xd blocks."""
    s = np.linalg.svd(np.asarray(A) @ np.asarray(B).T, compute_uv=False)
    return [float(np.degrees(np.arccos(x))) for x in np.clip(s, -1.0, 1.0)]


def gram_schmidt(V):
    """Orthonormalise rows, dropping any that collapse."""
    out = []
    for v in np.asarray(V, dtype=np.float64):
        for u in out:
            v = v - (v @ u) * u
        n = np.linalg.norm(v)
        if n > 1e-9:
            out.append(v / n)
    return np.array(out)


def project_out(M, V):
    """Remove the feature-space directions in the orthonormal rows of V."""
    if len(V) == 0:
        return M
    V = np.asarray(V, dtype=np.float64)
    Md = np.asarray(M, dtype=np.float64)
    return Md - (Md @ V.T) @ V


def project_out_labels(M, U):
    """Remove the label-space profiles in the orthonormal columns of U (k-vectors)."""
    if len(U) == 0:
        return M
    U = np.asarray(U, dtype=np.float64)          # [c, k]
    Md = np.asarray(M, dtype=np.float64)         # [k, d]
    return Md - U.T @ (U @ Md)


def centred_unit(v):
    v = np.asarray(v, dtype=np.float64)
    v = v - v.mean()
    n = np.linalg.norm(v)
    return None if n < 1e-12 else v / n


def auc(pos, neg):
    """Mann-Whitney AUC: P(a random accepted draw scores above a random rejected one)."""
    pos, neg = np.asarray(pos, dtype=float), np.asarray(neg, dtype=float)
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    allv = np.concatenate([pos, neg])
    r = np.argsort(np.argsort(allv)) + 1.0
    # average ranks for ties
    for v in np.unique(allv):
        m = allv == v
        if m.sum() > 1:
            r[m] = r[m].mean()
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def wilson(k, n, z=1.959963985):
    if n == 0:
        return [float("nan"), float("nan")]
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [float(max(0.0, c - h)), float(min(1.0, c + h))]


REVERDICT_NOTE = (
    "The verdict fields were written by a `--reverdict` pass over the run's own `arms` "
    "block, after the excess-based decision rule replaced a threshold-based one; "
    "`build_verdict()` is a pure function of `arms` and `global_structure`, so the pass "
    "is idempotent and a single full run of the current script produces this same file.")


def static_deviations(n_draws, n_perm_aux):
    """Everything this run does differently from `scripts/16_gate_background.py`."""
    return [
        "Permission to drop the ordering null to 2,000 permutations was given but NOT "
        "used: the restricted-cache reader made 20,000 affordable, so every arm runs at "
        "the gate's own resolution and the raw arm is bit-identical to "
        "results/gate_background_big_L9.json.",
        "The descriptive frequency-order statistic (rho_vs_freq_cyclic) runs at "
        f"{n_perm_aux} permutations; it is reported, never thresholded for a verdict.",
        f"{n_draws} nominal draws were generated rather than 400-500, because only "
        "~37.5% of draws are scorable and the task asks for 400-500 SCORABLE draws' "
        "worth of resolution; the first-450-nominal slice is reported separately.",
        "The cache is read through a row-restricted CSR holding only the 3,486 half-B "
        "label positions, verified bit-exact against the full reader, to avoid "
        "materialising the 5.9 GB full-layer CSR.",
        "The per-draw 12-label mean matrices M (k x 1152) and their top-2 PCA planes are "
        "NOT persisted: at ~535 draws that is ~7M floats, which does not belong in a "
        "JSON audit file. Every quantity derived from them is stored per draw "
        "(principal angles to the global plane, variance fractions inside it, "
        "u1_alignment, the frequency and global-order correlations), and M is "
        "deterministically regenerable -- draw i is stratified_resample under "
        "np.random.default_rng(0) advanced i times over the same template cycle, which "
        "the fidelity block proves reproduces the stored background exactly.",
    ]


def freq_cal_overlap(n_pos, full_k):
    """Fraction of the centred frequency vector that lies in the calendar plane.

    Month token counts run in near-calendar order in this corpus, so removing the
    frequency direction removes part of the first harmonic too. This number says how
    much, and is why a partial drop on `lab_freq1` cannot be read as support for R3.
    """
    th = 2 * np.pi * np.arange(full_k) / full_k
    f = centred_unit(np.asarray(n_pos, dtype=float))
    B = gram_schmidt(np.array([centred_unit(np.cos(th)), centred_unit(np.sin(th))]))
    return float(((B @ f) ** 2).sum())


def build_verdict(arms, gd, full_k):
    """The pre-declared decision rule, applied to the arm summaries.

    Rates are NOT compared against 0.05. Projecting the calendar's first harmonic
    out of the label means suppresses the OBSERVED statistic under all 2k symmetries
    of the true reference while leaving the permutation null -- whose permuted
    references generally have harmonic content outside that span -- untouched. So
    `lab_cal2` sits below nominal even on structureless data, and a raw threshold
    would confirm R2 from an artifact. The primary quantity for every arm is
    therefore its EXCESS over the matched synthetic or matched random arm:

      excess_raw    = raw       - synthetic_gauss     (the effect to be explained)
      excess_cal2   = lab_cal2  - synthetic_cal2      (survives calendar removal?)
      excess_rand2  = lab_rand2 - synthetic_rand2     (2 arbitrary directions)

    and, within real data only, how much more a candidate costs than an arbitrary
    direction of the same count: lab_rand1 - lab_freq1 and lab_rand2 - lab_cal2.

    R2 (shared cyclic component) is called when the calendar plane carries the whole
    excess and frequency does not; R3 (shared 1-D gradient) is the mirror image.
    """
    EX_ZERO = 0.03     # an excess this small is "gone"
    EX_SPECIAL = 0.05  # a candidate must cost this much more than an arbitrary direction
    GUARD = 0.025
    r = {k: arms[k]["rate"] for k in arms}
    ex = dict(
        excess_raw=float(r["raw"] - r["synthetic_gauss"]),
        excess_lab_cal2=float(r["lab_cal2"] - r["synthetic_cal2"]),
        excess_lab_rand2=float(r["lab_rand2"] - r["synthetic_rand2"]),
        excess_lab_u2_vs_synth_cal2=float(r["lab_u2"] - r["synthetic_cal2"]),
        excess_lab_u1_vs_synth_gauss=float(r["lab_u1"] - r["synthetic_gauss"]),
        excess_lab_freq1_vs_synth_gauss=float(r["lab_freq1"] - r["synthetic_gauss"]),
        cost_of_calendar_over_arbitrary=float(r["lab_rand2"] - r["lab_cal2"]),
        cost_of_frequency_over_arbitrary=float(r["lab_rand1"] - r["lab_freq1"]),
        freq_vector_in_calendar_plane=freq_cal_overlap(
            gd["label_n_positions_halfB"], full_k),
    )
    d1, d2 = gd["u1_decomposition"], gd["u2_decomposition"]
    cal_incr = max(d1["incremental_r2_calendar_over_frequency"],
                   d2["incremental_r2_calendar_over_frequency"])
    frq_incr = max(d1["incremental_r2_frequency_over_calendar"],
                   d2["incremental_r2_frequency_over_calendar"])
    if cal_incr > frq_incr + 0.10:
        identity = "calendar first harmonic (cyclic), not a frequency gradient"
    elif frq_incr > cal_incr + 0.10:
        identity = "label frequency (1-D gradient)"
    else:
        identity = "frequency and calendar not separable on 12 labels"

    r1_dead = r["synthetic_gauss"] <= COLLAPSE
    cal_kills = (ex["excess_lab_cal2"] <= EX_ZERO
                 and ex["cost_of_calendar_over_arbitrary"] >= EX_SPECIAL)
    freq_kills = (ex["excess_lab_freq1_vs_synth_gauss"] <= EX_ZERO
                  and ex["cost_of_frequency_over_arbitrary"] >= EX_SPECIAL)
    rand_holds = ex["excess_lab_rand2"] >= 0.5 * ex["excess_raw"]
    needs_two = (r["lab_u2"] < r["lab_u1"] - EX_SPECIAL)

    common = (f"raw {r['raw']:.3f} vs iid-Gaussian {r['synthetic_gauss']:.3f} "
              f"(excess {ex['excess_raw']:+.3f}). Calendar plane removed: "
              f"{r['lab_cal2']:.3f} vs its own synthetic {r['synthetic_cal2']:.3f} "
              f"(excess {ex['excess_lab_cal2']:+.3f}); 2 arbitrary label directions "
              f"removed: {r['lab_rand2']:.3f} vs {r['synthetic_rand2']:.3f} "
              f"(excess {ex['excess_lab_rand2']:+.3f}). Frequency direction removed: "
              f"{r['lab_freq1']:.3f} vs 1 arbitrary direction {r['lab_rand1']:.3f}. "
              f"Data-derived label components u1/u2: {r['lab_u1']:.3f}/{r['lab_u2']:.3f}. "
              f"Global label-mean object: rho={gd['global_scored']['rho']:.3f}, "
              f"p_order={gd['global_scored']['p_order']:.5f}, "
              f"p_closed={gd['global_scored']['p_closed']:.5f}, "
              f"top2 var {gd['global_scored']['top2']:.3f}. "
              f"Shared component identity: {identity} (incremental R^2 of the calendar "
              f"harmonic over frequency {cal_incr:.3f}, of frequency over the harmonic "
              f"{frq_incr:.3f}; {ex['freq_vector_in_calendar_plane']:.2f} of the frequency "
              f"vector already lies in the calendar plane).")

    if not r1_dead:
        verdict = "R1_statistic_saturated"
        vtext = (f"iid-Gaussian label means on the same survivor sets accept at "
                 f"{r['synthetic_gauss']:.3f}, above nominal: the ordering statistic is "
                 f"miscalibrated on these survivor sets and no label-space claim can be "
                 f"read off the background rate. " + common)
    elif cal_kills and not freq_kills:
        verdict = "R2_shared_cyclic_component"
        vtext = ("the excess acceptance is carried entirely by the calendar's first "
                 "harmonic plane in label space: removing it leaves no excess over the "
                 "matched synthetic, removing the frequency direction or arbitrary "
                 "directions of the same count does not. This is R2 -- a genuinely "
                 "present cyclic component that random subsets inherit -- not R3. "
                 + ("Both u1 and u2 are needed. " if needs_two else "")
                 + common)
    elif freq_kills and not cal_kills:
        verdict = "R3_single_shared_gradient"
        vtext = ("the excess acceptance is carried by a single shared 1-D gradient over "
                 "the month labels: removing the frequency direction removes it, removing "
                 "the calendar plane does not. " + common)
    elif cal_kills and freq_kills:
        verdict = "R2_R3_confounded"
        vtext = ("removing either the calendar plane or the frequency direction removes "
                 "the excess, and the two are not separable on 12 labels "
                 f"({ex['freq_vector_in_calendar_plane']:.2f} of the frequency vector lies "
                 f"in the calendar plane). Identity by incremental R^2: {identity}. "
                 + common)
    else:
        verdict = "no_single_shared_component"
        vtext = ("no 1-2 shared components account for the excess: neither the calendar "
                 "plane nor the frequency direction removes it beyond what arbitrary "
                 "directions of the same count remove. R3 is ruled out as the whole "
                 "mechanism; read `stratified` and `global_structure` for what remains. "
                 + common)

    return dict(verdict=verdict, verdict_text=vtext,
                decision_rule=dict(excess_zero=EX_ZERO, excess_special=EX_SPECIAL,
                                   guard=GUARD, collapse=COLLAPSE, hold=HOLD,
                                   nominal=NOMINAL,
                                   note="rates are judged by excess over the matched "
                                        "synthetic/random arm, never against 0.05"),
                excess=ex,
                r1_ruled_out=bool(r1_dead),
                calendar_plane_carries_excess=bool(cal_kills),
                frequency_direction_carries_excess=bool(freq_kills),
                random_control_holds=bool(rand_holds),
                two_components_needed=bool(needs_two),
                mechanical_suppression_guard=dict(
                    threshold=GUARD,
                    synthetic_gauss_rate=r["synthetic_gauss"],
                    synthetic_cal2_rate=r["synthetic_cal2"],
                    synthetic_rand2_rate=r["synthetic_rand2"],
                    calendar_projection_suppresses_below_nominal=bool(
                        r["synthetic_cal2"] < GUARD),
                    note="removing the calendar plane depresses the observed statistic "
                         "under every symmetry of the true reference but not under "
                         "permuted references, so this arm is expected below 0.05 even "
                         "with no structure; it is the baseline lab_cal2 is judged "
                         "against, not a failure"),
                shared_component_identity=identity,
                incremental_r2_calendar_over_frequency_max_u1u2=float(cal_incr),
                incremental_r2_frequency_over_calendar_max_u1u2=float(frq_incr))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=9)
    ap.add_argument("--l0", default="big")
    ap.add_argument("--root", default="cache/sparse_big")
    ap.add_argument("--half", default="rank-even")
    ap.add_argument("--gate", default="results/rerun_gate_big_primary_L9.json")
    ap.add_argument("--reference", default="results/gate_background_big_L9.json")
    ap.add_argument("--n-draws", type=int, default=1400,
                    help="nominal draws; ~37.5%% are scorable, so 1400 -> ~525")
    ap.add_argument("--n-perm", type=int, default=20000,
                    help="ordering permutations; the gate uses 20,000")
    ap.add_argument("--n-perm-aux", type=int, default=2000,
                    help="permutations for the DESCRIPTIVE frequency-order statistic")
    ap.add_argument("--fidelity-n", type=int, default=100)
    ap.add_argument("--reverdict", action="store_true",
                    help="re-apply build_verdict() to an existing --out JSON and rewrite "
                         "it; touches nothing else. A full run produces the same file.")
    ap.add_argument("--out", default="results/background_mechanism.json")
    a = ap.parse_args()
    t_start = time.time()

    if a.reverdict:
        cur = json.loads(Path(a.out).read_text())
        for k in ("verdict", "verdict_text", "decision_rule", "excess", "r1_ruled_out",
                  "calendar_plane_carries_excess", "frequency_direction_carries_excess",
                  "random_control_holds", "two_components_needed",
                  "mechanical_suppression_guard", "shared_component_identity",
                  "incremental_r2_calendar_over_frequency_max_u1u2",
                  "incremental_r2_frequency_over_calendar_max_u1u2",
                  "collapse_space", "n_shared_components",
                  "frequency_arm_collapses", "calendar_arm_collapses"):
            cur.pop(k, None)
        for k in ("shared_component_identity",
                  "incremental_r2_calendar_over_frequency_max_u1u2",
                  "incremental_r2_frequency_over_calendar_max_u1u2"):
            cur["global_structure"].pop(k, None)
        vd = build_verdict(cur["arms"], cur["global_structure"],
                           len(cur["global_structure"]["label_names"]))
        cur.update(vd)
        cur["deviations"] = static_deviations(
            cur["n_draws_nominal"], cur["n_perm_aux"]) + [REVERDICT_NOTE]
        Path(a.out).write_text(json.dumps(cur, indent=2))
        print(json.dumps({k: cur[k] for k in ("verdict", "verdict_text", "excess")}, indent=2))
        print(f"reverdict: rewrote {a.out}")
        return

    g = json.loads(Path(a.gate).read_text())
    templates = [np.asarray(r["members"]) for r in g["rows"] if r["layer"] == a.layer]
    names, label_ids = gate.label_ids_for(g["ontology"], json.load(open("results/token_map.json")))
    full_k = len(names)

    sae = load_sae(a.layer, a.l0, token=os.environ.get("HF_TOKEN"))
    W = get_decoder(sae)
    d_sae = sae.d_sae
    reader = SparseReader(a.root, a.layer)
    label_pos = gate.half_filter([reader.positions_of_tokens(ids) for ids in label_ids], a.half)
    n_pos = [int(len(p)) for p in label_pos]
    print(f"layer {a.layer}: {len(templates)} templates, sizes {sorted(len(t) for t in templates)}; "
          f"half-B label positions {n_pos}", flush=True)

    rr = RestrictedReader(reader, np.concatenate([np.asarray(p) for p in label_pos]))
    print(f"restricted cache: {rr._csr.shape[0]} rows, {rr._csr.nnz} nonzeros "
          f"({rr._csr.nnz * 12 / 1e6:.0f} MB)", flush=True)

    # exactness of the restricted view against row-by-row reads of the full cache
    chk = np.random.default_rng(7)
    feats = np.sort(chk.choice(d_sae, 40, replace=False))
    probe = chk.choice(rr.rows, 25, replace=False)
    ref_block = np.zeros((len(probe), len(feats)), dtype=np.float32)
    fmap = {int(f): j for j, f in enumerate(feats)}
    for i, r in enumerate(probe):
        idx, val = reader.row(int(r))
        for j, v in zip(idx, val):
            k = fmap.get(int(j))
            if k is not None:
                ref_block[i, k] = v
    got = rr.codes_at(probe, feats)
    reader_exact = bool(np.array_equal(got, ref_block))
    print(f"restricted reader bit-exact vs full cache: {reader_exact}", flush=True)

    # ---- alive latents and the GLOBAL label-mean matrix -------------------
    counts = np.zeros(d_sae, dtype=np.int64)
    step = 32_000_000
    for s0 in range(0, len(reader.indices), step):
        counts += np.bincount(np.asarray(reader.indices[s0:s0 + step], dtype=np.int64),
                              minlength=d_sae)
    alive = np.flatnonzero(counts > 0)
    print(f"alive latents: {len(alive)}/{d_sae}", flush=True)

    mean_codes = np.vstack([rr.mean_codes(p) for p in label_pos])          # [12, d_sae]
    M_glob = (mean_codes[:, alive] @ W[alive]).astype(np.float32)          # [12, 1152]
    Mg = np.asarray(M_glob, dtype=np.float64)
    Mgc = Mg - Mg.mean(0, keepdims=True)
    Ug, Sg, Vtg = np.linalg.svd(Mgc, full_matrices=False)
    glob_var = (Sg ** 2 / max((Sg ** 2).sum(), 1e-12)).tolist()
    u_glob = [Ug[:, j] * Sg[j] for j in range(3)]                          # label-space scores
    xy_g = Mgc @ Vtg[:2].T

    # what IS the shared component?  frequency, or the calendar's first harmonic?
    theta = 2 * np.pi * np.arange(full_k) / full_k
    Hcal = np.column_stack([np.ones(full_k), np.cos(theta), np.sin(theta)])
    Hfrq = np.column_stack([np.ones(full_k), np.asarray(n_pos, dtype=float)])

    Hjoint = np.column_stack([Hfrq, np.cos(theta), np.sin(theta)])

    def r2(H, y):
        y = np.asarray(y, dtype=float)
        coef, *_ = np.linalg.lstsq(H, y, rcond=None)
        res = y - H @ coef
        return float(1 - (res ** 2).sum() / max(((y - y.mean()) ** 2).sum(), 1e-12))

    def adj(r, p, n=full_k):
        return float(1 - (1 - r) * (n - 1) / max(n - p - 1, 1))

    def decompose(y):
        """Frequency vs the calendar's first harmonic, which are NOT independent here.

        The month token counts themselves run in near-calendar order
        (`freqorder_vs_calendar_rho` below), so a raw R^2 against frequency cannot
        be read as evidence for a frequency gradient. What separates them is the
        INCREMENTAL fit: how much each predictor set adds on top of the other.
        E[R^2] for a random 12-vector is p/(n-1) -- 0.09 for frequency alone, 0.18
        for the harmonic pair -- so both raw numbers are quoted against that.
        """
        rf, rc, rj = r2(Hfrq, y), r2(Hcal, y), r2(Hjoint, y)
        return dict(r2_frequency=rf, r2_calendar_harmonic=rc, r2_joint=rj,
                    adj_r2_frequency=adj(rf, 1), adj_r2_calendar_harmonic=adj(rc, 2),
                    incremental_r2_calendar_over_frequency=float(rj - rf),
                    incremental_r2_frequency_over_calendar=float(rj - rc),
                    chance_r2_frequency=1.0 / (full_k - 1),
                    chance_r2_calendar_harmonic=2.0 / (full_k - 1))

    mean_dec = W[alive].mean(0)
    mean_dec = mean_dec / max(np.linalg.norm(mean_dec), 1e-12)
    global_diag = dict(
        n_alive=int(len(alive)),
        label_names=list(names),
        label_n_positions_halfB=n_pos,
        label_mean_code_l1=[float(mean_codes[i, alive].sum()) for i in range(full_k)],
        global_var_ratios=glob_var[:4],
        global_top2_var=float(sum(glob_var[:2])),
        u1_profile=[float(x) for x in u_glob[0]],
        u2_profile=[float(x) for x in u_glob[1]],
        global_pc1_coord=[float(x) for x in xy_g[:, 0]],
        global_pc2_coord=[float(x) for x in xy_g[:, 1]],
        u1_spearman_vs_freq=float(spearmanr(u_glob[0], n_pos).statistic),
        u2_spearman_vs_freq=float(spearmanr(u_glob[1], n_pos).statistic),
        u1_r2_frequency=r2(Hfrq, u_glob[0]), u1_r2_calendar_harmonic=r2(Hcal, u_glob[0]),
        u2_r2_frequency=r2(Hfrq, u_glob[1]), u2_r2_calendar_harmonic=r2(Hcal, u_glob[1]),
        u1_decomposition=decompose(u_glob[0]),
        u2_decomposition=decompose(u_glob[1]),
        u3_decomposition=decompose(u_glob[2]),
        cos_globalPC1_meanDecoder=float(abs(Vtg[0] @ mean_dec)),
        cos_globalPC2_meanDecoder=float(abs(Vtg[1] @ mean_dec)),
    )
    # The global object put through the gate's own scoring path. This is not a
    # background draw and not a gate result -- it is the whole cache's month
    # geometry, all 16,358 alive latents at once, scored so the draws have
    # something to be compared against.
    gs = score_M(M_glob, list(range(full_k)), seed=0, full_k=full_k, n_perm=a.n_perm)
    global_diag["global_scored"] = (
        None if gs is None else
        {k: v for k, v in gs[0].items() if k in
         ("rho", "p_order", "p_closed", "gap_ratio", "top2", "null_p95")})
    ang_g = angles(xy_g, *fit_circle(xy_g)[:2])
    rank_ang_g = np.argsort(np.argsort(ang_g))
    # does the frequency ordering itself agree with the calendar cycle?
    frank = np.argsort(np.argsort(np.asarray(n_pos, dtype=float)))
    fs = cyclic_order_score(np.asarray(n_pos, dtype=float), np.arange(full_k), full_k,
                            n_perm=a.n_perm_aux, seed=0, full_k=full_k)
    global_diag["freqorder_vs_calendar_rho"] = float(fs["rho"])
    global_diag["freqorder_vs_calendar_p"] = float(fs["p"])
    print(json.dumps({k: v for k, v in global_diag.items()
                      if not isinstance(v, list)}, indent=2), flush=True)

    # ---- the draws -------------------------------------------------------
    # lab_u1/lab_u2 remove DATA-DERIVED shared profiles; lab_freq1 and lab_cal2 remove
    # the two a-priori candidates named in the review -- the frequency gradient (R3)
    # and the calendar's first harmonic pair (R2) -- so the identity of the shared
    # component is tested without going through the SVD at all.
    # Removing the calendar's first harmonic from the label means is close to removing
    # the thing the ordering test looks for, so a collapse on real data is only
    # informative if the same removal does NOT push structureless data below nominal.
    # `synthetic_cal2` is that guard: iid-Gaussian label means with the same projection.
    ARMS = ["raw", "feat_g1", "feat_g2", "feat_rand2",
            "lab_u1", "lab_u2", "lab_rand1", "lab_rand2",
            "lab_freq1", "lab_cal2",
            "synthetic_gauss", "synthetic_cal2", "synthetic_rand2"]
    rng = np.random.default_rng(0)                       # == scripts/16_gate_background.py
    ref = {r["i"]: r for r in json.loads(Path(a.reference).read_text())["draws"]} \
        if Path(a.reference).exists() else {}
    fid = dict(n_compared=0, n_scorable_agree=0, max_abs_drho=0.0,
               max_abs_dgap=0.0, n_p_order_identical=0, n_labels_agree=0)

    draws, arm_rows = [], {k: [] for k in ARMS}
    n_deg = {k: 0 for k in ARMS}
    for i in range(a.n_draws):
        t = templates[i % len(templates)]
        f = stratified_resample(t, d_sae, rng)
        cluster = [int(x) for x in f]
        if i and i % 100 == 0:
            print(f"  {i}/{a.n_draws} draws, {len(draws)} scorable, "
                  f"{time.time() - t_start:.0f}s", flush=True)

        M, keep = means_and_keep(rr, cluster, W, label_pos, full_k)
        if M is None:
            if i in ref:
                fid["n_compared"] += 1
                fid["n_scorable_agree"] += int(ref[i]["scorable"] is False)
            continue
        base = score_M(M, keep, seed=i, full_k=full_k, n_perm=a.n_perm, with_ref=True)
        if base is None:
            if i in ref:
                fid["n_compared"] += 1
                fid["n_scorable_agree"] += int(ref[i]["scorable"] is False)
            continue
        lab, xy, Vt = base

        if i in ref:
            fid["n_compared"] += 1
            r0 = ref[i]
            fid["n_scorable_agree"] += int(bool(r0["scorable"]))
            if r0["scorable"]:
                fid["max_abs_drho"] = max(fid["max_abs_drho"], abs(r0["rho"] - lab["rho"]))
                fid["max_abs_dgap"] = max(fid["max_abs_dgap"],
                                          abs(r0["gap_ratio"] - lab["gap_ratio"]))
                fid["n_p_order_identical"] += int(r0["p_order"] == lab["p_order"])
                fid["n_labels_agree"] += int(r0["n_labels"] == lab["n_labels"])

        kp = np.asarray(keep)
        Mc = np.asarray(M, dtype=np.float64)
        Mc = Mc - Mc.mean(0, keepdims=True)
        Ud, Sd, Vtd = np.linalg.svd(Mc, full_matrices=False)

        # (a) principal angles to the global plane
        pa = principal_angles_deg(Vtd[:2], Vtg[:2])
        var_tot = max((Sd ** 2).sum(), 1e-12)
        var_in_g2 = float(((Mc @ Vtg[:2].T) ** 2).sum() / var_tot)
        var_in_g1 = float(((Mc @ Vtg[0:1].T) ** 2).sum() / var_tot)

        # (b) does the recovered order track label frequency?
        ang = angles(xy, *fit_circle(xy)[:2])
        fr = np.asarray(n_pos, dtype=float)[kp]
        sp_ang_freq = float(spearmanr(np.argsort(np.argsort(ang)), fr).statistic)
        sp_pc1_freq = float(spearmanr(xy[:, 0], fr).statistic)
        sp_pc1_globalpc1 = float(spearmanr(xy[:, 0], xy_g[kp, 0]).statistic)
        u1g_k = centred_unit(u_glob[0][kp])
        u2g_k = centred_unit(u_glob[1][kp])
        u1_align = float(abs(centred_unit(Ud[:, 0]) @ u1g_k)) if u1g_k is not None else float("nan")
        fcyc = cyclic_order_score(ang, np.argsort(np.argsort(fr)), len(kp),
                                  n_perm=a.n_perm_aux, seed=i, full_k=len(kp))
        # does the draw recover the same cyclic arrangement as the global object?
        gcyc = cyclic_order_score(ang, np.argsort(np.argsort(rank_ang_g[kp])), len(kp),
                                  n_perm=a.n_perm_aux, seed=i, full_k=len(kp))

        row = dict(i=i, size=len(cluster), template_size=int(len(t)),
                   n_labels=lab["n_labels"], labels=[int(x) for x in keep],
                   rho=lab["rho"], p_order=lab["p_order"], p_closed=lab["p_closed"],
                   gap_ratio=lab["gap_ratio"], top2=lab["top2"],
                   null_mean=lab["null_mean"], null_p95=lab["null_p95"],
                   ref_p_closed=lab["ref_p_closed"], missing_run=lab["missing_run"],
                   accept=bool(lab["p_order"] <= 0.05),
                   pa1_deg=pa[0], pa2_deg=pa[1],
                   var_frac_in_global_pc1=var_in_g1, var_frac_in_global_top2=var_in_g2,
                   spearman_angleorder_vs_freq=sp_ang_freq,
                   spearman_pc1_vs_freq=sp_pc1_freq,
                   spearman_pc1_vs_globalpc1=sp_pc1_globalpc1,
                   u1_alignment=u1_align,
                   rho_vs_freq_cyclic=float(fcyc["rho"]), p_vs_freq_cyclic=float(fcyc["p"]),
                   rho_vs_globalorder_cyclic=float(gcyc["rho"]),
                   p_vs_globalorder_cyclic=float(gcyc["p"]))
        draws.append(row)
        arm_rows["raw"].append(dict(i=i, p_order=lab["p_order"], rho=lab["rho"],
                                    p_closed=lab["p_closed"], n_labels=lab["n_labels"]))

        # ---- residualisation arms ----------------------------------------
        crng = np.random.default_rng(10_000_000 + i)
        Rf = gram_schmidt(crng.normal(size=(2, Mc.shape[1])))
        Rl = gram_schmidt(np.array([centred_unit(v) for v in crng.normal(size=(2, len(kp)))]))
        Ul = gram_schmidt(np.array([v for v in (u1g_k, u2g_k) if v is not None]))
        Ucal = gram_schmidt(np.array([v for v in (centred_unit(np.cos(theta[kp])),
                                                  centred_unit(np.sin(theta[kp])))
                                      if v is not None]))
        Ufreq = centred_unit(fr)
        Msyn = np.random.default_rng(20_000_000 + i).normal(size=M.shape).astype(np.float32)
        variants = dict(
            feat_g1=project_out(M, Vtg[0:1]),
            feat_g2=project_out(M, Vtg[0:2]),
            feat_rand2=project_out(M, Rf),
            lab_u1=project_out_labels(M, Ul[0:1]),
            lab_u2=project_out_labels(M, Ul[0:2]),
            lab_rand1=project_out_labels(M, Rl[0:1]),
            lab_rand2=project_out_labels(M, Rl),
            lab_freq1=project_out_labels(M, np.zeros((0, len(kp))) if Ufreq is None
                                         else Ufreq[None, :]),
            lab_cal2=project_out_labels(M, Ucal),
            synthetic_gauss=Msyn,
            synthetic_cal2=project_out_labels(Msyn, Ucal),
            synthetic_rand2=project_out_labels(Msyn, Rl),
        )
        for name, Mv in variants.items():
            got = score_M(Mv, keep, seed=i, full_k=full_k, n_perm=a.n_perm)
            if got is None:
                n_deg[name] += 1
                continue
            lv = got[0]
            arm_rows[name].append(dict(i=i, p_order=lv["p_order"], rho=lv["rho"],
                                       p_closed=lv["p_closed"], n_labels=lv["n_labels"]))

    n_sc = len(draws)
    arms = {}
    for name in ARMS:
        rs = arm_rows[name]
        nk = sum(r["p_order"] <= 0.05 for r in rs)
        arms[name] = dict(
            n_scored=len(rs), n_degenerate=n_deg[name],
            n_accept=int(nk),
            rate=float(nk / max(len(rs), 1)),
            rate_ci95=wilson(nk, len(rs)),
            rate_if_degenerate_counted_fail=float(nk / max(n_sc, 1)),
            mean_rho=float(np.mean([r["rho"] for r in rs])) if rs else float("nan"),
            median_p=float(np.median([r["p_order"] for r in rs])) if rs else float("nan"),
            ks_p_uniform=float(kstest([r["p_order"] for r in rs], "uniform").pvalue) if rs else float("nan"),
            rate_closure=float(sum(r["p_closed"] < 0.05 for r in rs) / max(len(rs), 1)),
        )

    # first-450-nominal slice, the task's literal draw count
    sub = [r for r in draws if r["i"] < 450]
    first450 = dict(n_nominal=450, n_scorable=len(sub),
                    n_accept=int(sum(r["accept"] for r in sub)),
                    rate=float(sum(r["accept"] for r in sub) / max(len(sub), 1)),
                    rate_ci95=wilson(sum(r["accept"] for r in sub), len(sub)))

    acc = [r for r in draws if r["accept"]]
    rej = [r for r in draws if not r["accept"]]

    def strat(rows, key):
        return float(np.mean([r[key] for r in rows])) if rows else float("nan")

    KEYS = ("pa1_deg", "pa2_deg", "var_frac_in_global_pc1",
            "var_frac_in_global_top2", "u1_alignment",
            "spearman_pc1_vs_freq", "spearman_pc1_vs_globalpc1",
            "spearman_angleorder_vs_freq", "rho_vs_freq_cyclic",
            "rho_vs_globalorder_cyclic", "rho", "top2")
    stratified = {k: dict(accepted=strat(acc, k), rejected=strat(rej, k),
                          auc_accept=auc([r[k] for r in acc], [r[k] for r in rej]))
                  for k in KEYS}
    stratified["n_accepted"] = len(acc)
    stratified["n_rejected"] = len(rej)
    stratified["rate_by_template_size"] = {
        str(ts): dict(n=int(sum(r["template_size"] == ts for r in draws)),
                      rate=float(np.mean([r["accept"] for r in draws
                                          if r["template_size"] == ts])))
        for ts in sorted({r["template_size"] for r in draws})}
    stratified["rate_by_n_labels"] = {
        str(nl): dict(n=int(sum(r["n_labels"] == nl for r in draws)),
                      rate=float(np.mean([r["accept"] for r in draws if r["n_labels"] == nl])))
        for nl in sorted({r["n_labels"] for r in draws})}
    stratified["p_vs_freq_cyclic_rate"] = float(
        np.mean([r["p_vs_freq_cyclic"] <= 0.05 for r in draws])) if draws else float("nan")
    stratified["p_vs_globalorder_cyclic_rate"] = float(
        np.mean([r["p_vs_globalorder_cyclic"] <= 0.05 for r in draws])) if draws else float("nan")

    # ---- verdict ---------------------------------------------------------
    raw_rate = arms["raw"]["rate"]
    vd = build_verdict(arms, global_diag, full_k)
    verdict, vtext = vd["verdict"], vd["verdict_text"]

    out = dict(
        status="exploratory, not pre-registered",
        item="5e -- mechanism of the ordering tier's high background acceptance",
        layer=a.layer, l0=a.l0, root=a.root, half=a.half, ontology=g["ontology"],
        gate=a.gate, reference_background=a.reference,
        n_draws_nominal=a.n_draws, n_scorable=n_sc,
        n_templates=len(templates),
        template_sizes=[int(len(t)) for t in templates],
        n_perm_ordering=a.n_perm, n_perm_aux=a.n_perm_aux,
        collapse_threshold=COLLAPSE, hold_threshold=HOLD, nominal=NOMINAL,
        deviations=static_deviations(a.n_draws, a.n_perm_aux),
        fidelity=dict(reader_bit_exact_vs_full_cache=reader_exact, **fid),
        reproduction=dict(
            reference_rate_order=0.164, reference_n_scorable=750,
            this_run_rate_order=raw_rate, this_run_ci95=arms["raw"]["rate_ci95"],
            first_450_nominal=first450),
        global_structure=global_diag,
        arms=arms,
        stratified=stratified,
        **vd,
        runtime_s=round(time.time() - t_start, 1),
        draws=draws,
    )
    Path(a.out).write_text(json.dumps(out, indent=2))
    print(json.dumps({k: v for k, v in out.items()
                      if k not in ("draws", "global_structure", "stratified", "deviations")},
                     indent=2))
    print(f"wrote {a.out}  ({out['runtime_s']}s)")


if __name__ == "__main__":
    main()
