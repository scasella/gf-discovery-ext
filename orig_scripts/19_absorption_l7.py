"""Item 5d -- is the layer-7 arc ordering GEOMETRY, or ABSORPTION?

DECLARED EXPLORATORY. Not pre-registered; nothing here can turn a gate verdict.

The project's only quasi-positive is an arc_only cluster at layer 7 (coactivation,
k=24, rho +0.965, p_order at the 1/20001 permutation floor, p_closed 0.948). The
project's own review asked whether that ordering is real ring geometry or whether
the cluster's decoder directions share one dominant direction that any monotone
read-out would inherit -- the SAE "absorption" / near-duplicate-feature failure
mode. A cluster whose members are all c_i * u + small residual reconstructs every
label as (sum_i a_i c_i) * u + noise, i.e. a 1-D ramp along u, and a 1-D ramp is
scored rho ~ 1 by `cyclic_order_score`, which detects ORDER, not closure.

The test: project every member decoder vector onto the orthogonal complement of
the member block's top principal direction, then re-run the EXACT gate scoring
(`13_rerun_gate.labelled`: same half-B odd cache rows, same MIN_FIRING_PER_LABEL,
same ceil(2k/3) support rule, same 20,000 permutations, same per-cluster seed) on
the deflated directions. Ordering that collapses when one shared direction is
removed is absorption; ordering that survives is distributed structure.

Two controls make the collapse (or survival) readable rather than tautological:
  random-in-span -- 50 isotropically-sampled unit directions inside the member
      block's own row space, each deflated the same way. This is the null for
      "you deleted a direction, of course it changed".
  last-PC        -- deflating the *weakest* principal direction of the span,
      which removes almost no energy.
Energy removed (fraction of the block's squared Frobenius norm) is reported for
every condition so the "you removed too much" reading can be checked directly.

Layer 9's primary-arm coactivation cluster (k=23, rho +0.266, p_order 0.998) runs
through the identical panel and identical deflations as the comparison case.

No forward passes: cached CSR codes plus the SAE decoder matrix only.
"""
import argparse, gc, json, importlib.util, os, sys, time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from gfdisc.cache import SparseReader
from gfdisc.saes import load_sae, get_decoder
from gfdisc.fits import pca_2d, fit_circle, circularity, angles
from gfdisc.tokens import MONTHS

# Import the gate module by path: its filename starts with a digit, so the
# reference `labelled` / `half_filter` / `min_labels_with_support` cannot be
# reached with a normal import. Loading it this way guarantees the scoring code
# is byte-for-byte the one that produced the published rows.
_spec = importlib.util.spec_from_file_location("gate13", ROOT / "scripts" / "13_rerun_gate.py")
gate13 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gate13)

N_PERM = 20000
# 200, not 50: the whole L7 verdict turns on the percentile of the PC1 result in
# the energy-matched control, and at n=50 the Monte-Carlo SE on a ~6% percentile
# is +-3.4 points -- wide enough to move the call across the 5% line by itself.
N_RANDOM_CONTROL = 200


# ---------------------------------------------------------------------------
# Cache access
# ---------------------------------------------------------------------------

class ThinReader:
    """`codes_at` only, straight off the CSR memmaps -- no whole-cache CSR.

    `SparseReader.csr()` materialises float32 values and int64 indices for all
    ~4.4e8 nonzeros (~7 GB peak). This analysis touches ~3.5k rows in total, so
    the block is read row-by-row instead. Output is validated to be EXACTLY
    equal to the SparseReader path once per layer before it is used.
    """

    def __init__(self, root, layer):
        self.inner = SparseReader(root, layer)
        self.d_sae = self.inner.meta["d_sae"]
        self._cache = {}

    def positions_of_tokens(self, ids):
        return self.inner.positions_of_tokens(ids)

    def codes_at(self, rows, features):
        rows = np.asarray(rows)
        feat = np.asarray(features)
        key = (hash(rows.tobytes()), len(rows), hash(feat.tobytes()))
        if key in self._cache:
            return self._cache[key]
        lut = np.full(self.d_sae, -1, dtype=np.int64)
        lut[feat] = np.arange(len(feat))
        out = np.zeros((len(rows), len(feat)), dtype=np.float32)
        indptr, indices, values = self.inner.indptr, self.inner.indices, self.inner.values
        for r, i in enumerate(rows):
            a, b = indptr[int(i)], indptr[int(i) + 1]
            col = lut[np.asarray(indices[a:b], dtype=np.int64)]
            hit = col >= 0
            if hit.any():
                out[r, col[hit]] = np.asarray(values[a:b], dtype=np.float32)[hit]
        self._cache[key] = out
        return out


def validate_thin(thin, label_pos, members):
    """One full-CSR build, exact comparison, then free it again."""
    inner = thin.inner
    ok = True
    for pos in label_pos:
        a = np.asarray(inner.csr()[np.asarray(pos)][:, np.asarray(members)].todense())
        b = thin.codes_at(pos, members)
        ok = ok and a.shape == b.shape and bool(np.array_equal(a, b))
    inner._csr = None
    gc.collect()
    return bool(ok)


# ---------------------------------------------------------------------------
# Decoder-geometry panel
# ---------------------------------------------------------------------------

def panel(Wc):
    """Duplicate / shared-direction diagnostics on the member decoder block."""
    W = np.asarray(Wc, dtype=np.float64)
    k = len(W)
    n = W / np.maximum(np.linalg.norm(W, axis=1, keepdims=True), 1e-12)
    G = n @ n.T
    off = G[~np.eye(k, dtype=bool)]
    per_member_max = (G - 2 * np.eye(k)).max(1)

    s_unc = np.linalg.svd(W, compute_uv=False)
    lam_u = s_unc ** 2
    Wm = W - W.mean(0, keepdims=True)
    s_cen = np.linalg.svd(Wm, compute_uv=False)
    lam_c = s_cen ** 2

    def pr(lam):
        return float(lam.sum() ** 2 / max((lam ** 2).sum(), 1e-30))

    def erank(s):
        p = s / max(s.sum(), 1e-30)
        p = p[p > 0]
        return float(np.exp(-(p * np.log(p)).sum()))

    u1 = np.linalg.svd(W, full_matrices=False)[2][0]
    if float((n @ u1).sum()) < 0:                    # sign convention: point with the members
        u1 = -u1
    mdir = W.mean(0); mdir = mdir / np.linalg.norm(mdir)
    u1c = np.linalg.svd(Wm, full_matrices=False)[2][0]
    cos_u1 = n @ u1

    return dict(
        k=int(k),
        member_norms=dict(mean=float(np.linalg.norm(W, axis=1).mean()),
                          min=float(np.linalg.norm(W, axis=1).min()),
                          max=float(np.linalg.norm(W, axis=1).max())),
        pairwise_cosine=dict(mean=float(off.mean()), median=float(np.median(off)),
                             min=float(off.min()), max=float(off.max()),
                             std=float(off.std())),
        per_member_max_cosine=[float(x) for x in per_member_max],
        max_cosine_overall=float(per_member_max.max()),
        median_of_per_member_max=float(np.median(per_member_max)),
        n_members_max_cos_gt=dict(**{f"{t}": int((per_member_max > t).sum())
                                     for t in (0.5, 0.7, 0.8, 0.9, 0.95)}),
        participation_ratio_uncentered=pr(lam_u),
        participation_ratio_centered=pr(lam_c),
        effective_rank_uncentered=erank(s_unc),
        effective_rank_centered=erank(s_cen),
        pc1_var_share_uncentered=float(lam_u[0] / lam_u.sum()),
        pc2_var_share_uncentered=float(lam_u[1] / lam_u.sum()),
        pc12_var_share_uncentered=float(lam_u[:2].sum() / lam_u.sum()),
        pc1_var_share_centered=float(lam_c[0] / lam_c.sum()),
        pc12_var_share_centered=float(lam_c[:2].sum() / lam_c.sum()),
        singular_values_uncentered=[float(x) for x in s_unc],
        cos_member_to_u1_uncentered=dict(mean=float(cos_u1.mean()), min=float(cos_u1.min()),
                                         max=float(cos_u1.max()),
                                         mean_abs=float(np.abs(cos_u1).mean())),
        # if these three near-coincide the choice of "the shared direction" is moot
        cos_u1unc_meandir=float(u1 @ mdir),
        cos_u1unc_u1cen=float(abs(u1 @ u1c)),
        cos_meandir_u1cen=float(abs(mdir @ u1c)),
    )


# ---------------------------------------------------------------------------
# Deflations
# ---------------------------------------------------------------------------

def deflate(Wc, dirs):
    """Project rows of Wc onto the orthogonal complement of the rows of `dirs`."""
    W = np.asarray(Wc, dtype=np.float64)
    if dirs is None or len(dirs) == 0:
        out = W
    else:
        Q = np.linalg.qr(np.asarray(dirs, dtype=np.float64).T)[0]
        out = W - (W @ Q) @ Q.T
    e0 = float((W ** 2).sum())
    return out, float(1.0 - (out ** 2).sum() / max(e0, 1e-30))


def span_basis(Wc, tol=1e-8):
    """Orthonormal basis of the member block's row space, plus its singular values."""
    U, S, Vt = np.linalg.svd(np.asarray(Wc, dtype=np.float64), full_matrices=False)
    r = int((S > tol * S[0]).sum())
    return Vt[:r], S[:r]


def random_in_span(Vt_span, rng, m=1):
    """Isotropic random m-frame INSIDE the span, returned as m x d_in.

    Not `Wc.T @ g`: that is covariance-weighted and would preferentially sample
    near PC1, so the control would partly replicate the treatment.

    Energy note: for q isotropic-unit in an r-dimensional span, E||W q||^2 =
    ||W||_F^2 / r, so a random m-frame removes m/r of the block's energy in
    expectation. Choosing m ~ round(r * PC1_share) therefore gives a control that
    deletes the SAME amount of energy as the PC1 deflation along an arbitrary
    m-dimensional slice of the same span -- the answer to "you removed 29% of the
    energy, of course the ordering died".
    """
    G = rng.normal(size=(len(Vt_span), m))
    Q = np.linalg.qr(G)[0]
    return Q.T @ Vt_span


def pctile_at_or_below(values, x):
    v = np.asarray(values, dtype=float)
    return dict(n_at_or_below=int((v <= x).sum()), n=int(len(v)),
                pct_at_or_below=float(100.0 * (v <= x).mean()))


# ---------------------------------------------------------------------------
# Diagnostics on the label-mean cloud (not part of the gate)
# ---------------------------------------------------------------------------

def _spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    n = len(a)
    d = ra - rb
    return float(1 - 6 * (d @ d) / (n * (n * n - 1)))


def cloud_diagnostics(thin, members, Wc, label_pos, u1):
    """Where does the ordering live? Along the shared direction, or off it?

    Rebuilds the same label-mean cloud `labelled()` scores (same support rule),
    then asks whether the calendar order is already recoverable from the single
    scalar M . u1 -- the "monotone read-out of one shared direction" mechanism.
    """
    means, keep, mass = [], [], []
    for i, pos in enumerate(label_pos):
        if len(pos) < 5:
            continue
        A = thin.codes_at(pos, members)
        if int((A > 0).any(1).sum()) < gate13.MIN_FIRING_PER_LABEL:
            continue
        means.append((A @ np.asarray(Wc)).mean(0))
        keep.append(i)
        mass.append(float(A.mean(0).sum()))
    M = np.vstack(means).astype(np.float64)
    proj = M @ u1
    xy, ratios, Vt = pca_2d(M)
    cx, cy, r, _ = fit_circle(xy)
    th = angles(xy, cx, cy)
    idx = np.asarray(keep, float)
    Mc = M - M.mean(0, keepdims=True)
    Mres = M - np.outer(proj, u1)
    return dict(
        labels_kept=[int(x) for x in keep],
        label_names=[MONTHS[i] for i in keep],
        mean_code_mass_per_label=mass,
        proj_on_u1=[float(x) for x in proj],
        spearman_proj_u1_vs_calendar=_spearman(proj, idx),
        spearman_codemass_vs_calendar=_spearman(mass, idx),
        # how much of the label-mean cloud sits on the shared axis
        u1_share_of_cloud_energy=float((proj ** 2).sum() / max((M ** 2).sum(), 1e-30)),
        u1_share_of_centred_cloud_variance=float(
            ((Mc @ u1) ** 2).sum() / max((Mc ** 2).sum(), 1e-30)),
        cos_pc1_of_cloud_to_u1=float(abs(Vt[0] @ u1)),
        residual_energy_share_after_u1=float((Mres ** 2).sum() / max((M ** 2).sum(), 1e-30)),
        cloud_top2_var=float(ratios[:2].sum()),
        recovered_angles_deg=[float(np.degrees(x)) for x in th],
    )


# ---------------------------------------------------------------------------
# Per-cluster run
# ---------------------------------------------------------------------------

def score(thin, members, W_full, Wc_def, label_pos, seed, full_k):
    """Exact reference `labelled()` on a deflated member block."""
    W_full[np.asarray(members)] = Wc_def.astype(W_full.dtype)
    return gate13.labelled(thin, list(members), W_full, label_pos, seed, full_k)


def summarise(lab):
    if lab is None:
        return dict(result="degenerate_or_unsupported",
                    note="labelled() returned None: the deflated cloud lost support "
                         "or became degenerate under fits.degenerate()")
    return dict(result="scored", n_labels=lab["n_labels"], rho=lab["rho"],
                p_order=lab["p_order"], gap_ratio=lab["gap_ratio"],
                max_gap_deg=lab["max_gap_deg"], p_closed=lab["p_closed"],
                top2=lab["top2"])


def run_cluster(tag, layer, members, stored, root, l0, label_ids, full_k, log):
    t0 = time.time()
    sae = load_sae(layer, l0, token=os.environ.get("HF_TOKEN"))
    # kept float32, exactly as the gate had it: float32 -> float64 -> float32 is
    # lossless, so the baseline condition writes back the identical bits and must
    # reproduce the published row to the last digit.
    W = get_decoder(sae)
    del sae; gc.collect()
    Wc32 = W[np.asarray(members)].copy()
    Wc = Wc32.astype(np.float64)

    thin = ThinReader(root, layer)
    label_pos = gate13.half_filter([thin.positions_of_tokens(ids) for ids in label_ids],
                                   "rank-even")
    log(f"[{tag}] half-B label positions: {[len(p) for p in label_pos]}")
    identical = validate_thin(thin, label_pos, members)
    log(f"[{tag}] ThinReader codes identical to SparseReader CSR path: {identical}")

    seed = abs(hash((layer, int(members[0])))) % 10 ** 6
    pan = panel(Wc)
    log(f"[{tag}] mean pairwise cos {pan['pairwise_cosine']['mean']:+.4f}  "
        f"max {pan['max_cosine_overall']:+.4f}  PR {pan['participation_ratio_uncentered']:.2f}  "
        f"PC1 share {pan['pc1_var_share_uncentered']:.3f}")

    Vt_span, S_span = span_basis(Wc)
    U, S, Vt = np.linalg.svd(Wc, full_matrices=False)
    u1 = Vt[0]
    if float((Wc @ u1).sum()) < 0:
        u1 = -u1
    u2 = Vt[1]
    Wm = Wc - Wc.mean(0, keepdims=True)
    Vtc = np.linalg.svd(Wm, full_matrices=False)[2]
    mdir = Wc.mean(0) / np.linalg.norm(Wc.mean(0))
    u_last = Vt_span[-1]

    W_mod = W.copy()
    conds = [
        ("baseline", None),
        ("deflate_pc1_uncentered", u1[None, :]),
        ("deflate_pc1pc2_uncentered", np.vstack([u1, u2])),
        ("deflate_mean_direction", mdir[None, :]),
        ("deflate_pc1_centered", Vtc[0][None, :]),
        ("deflate_pc1pc2_centered", Vtc[:2]),
        ("deflate_last_pc_of_span", u_last[None, :]),
    ]
    out = {}
    for name, dirs in conds:
        Wd, energy = deflate(Wc, dirs)
        lab = score(thin, members, W_mod, Wd, label_pos, seed, full_k)
        rec = summarise(lab)
        rec["energy_removed_frac"] = energy
        rec["n_directions_removed"] = 0 if dirs is None else int(len(dirs))
        out[name] = rec
        log(f"[{tag}] {name:28s} E-{energy:6.3f}  " +
            (f"rho {rec['rho']:+.4f}  pOrd {rec['p_order']:.5g}  "
             f"gap {rec['gap_ratio']:.3f}  pClo {rec['p_closed']:.4f}"
             if rec["result"] == "scored" else rec["result"]))

    # baseline must reproduce the published row
    base = out["baseline"]
    repro = dict(stored=stored, reproduced={k: base[k] for k in
                 ("rho", "p_order", "gap_ratio", "p_closed", "n_labels")},
                 rho_abs_diff=abs(base["rho"] - stored["rho"]),
                 p_order_abs_diff=abs(base["p_order"] - stored["p_order"]),
                 p_closed_abs_diff=abs(base["p_closed"] - stored["p_closed"]),
                 gap_ratio_abs_diff=abs(base["gap_ratio"] - stored["gap_ratio"]))
    repro["exact"] = bool(repro["rho_abs_diff"] < 1e-9 and repro["gap_ratio_abs_diff"] < 1e-9
                          and repro["p_order_abs_diff"] < 2 / (N_PERM + 1)
                          and repro["p_closed_abs_diff"] < 2 / (N_PERM + 1))
    log(f"[{tag}] baseline reproduces stored row exactly: {repro['exact']}")

    # one-PC-at-a-time sweep: is the collapse specific to PC1, or does any
    # high-energy direction of the span do it?
    sweep = []
    for j in range(len(Vt_span)):
        Wd, energy = deflate(Wc, Vt_span[j][None, :])
        lab = score(thin, members, W_mod, Wd, label_pos, seed, full_k)
        rec = summarise(lab); rec["pc"] = j + 1; rec["energy_removed_frac"] = energy
        sweep.append(rec)
    worse = [r for r in sweep if r["result"] == "scored" and r["rho"] <= sweep[0]["rho"]]
    log(f"[{tag}] PC sweep: {len(worse)}/{len(sweep)} single-PC deflations reach rho <= "
        f"the PC1 result; PC2 gives rho {sweep[1].get('rho', float('nan')):+.4f} "
        f"(E-{sweep[1]['energy_removed_frac']:.3f})")

    # controls, both rank-1 (arbitrary direction, ~1/r of the energy) and
    # energy-matched (m dims chosen so m/r ~ the PC1 energy share)
    rank = len(Vt_span)
    m_match = max(1, int(round(rank * pan["pc1_var_share_uncentered"])))
    controls = {}
    rng = np.random.default_rng(20250820 + layer)
    for cname, m in (("random_in_span_rank1", 1), (f"random_in_span_{m_match}dim", m_match)):
        reps = []
        for _ in range(N_RANDOM_CONTROL):
            V = random_in_span(Vt_span, rng, m)
            Wd, energy = deflate(Wc, V)
            lab = score(thin, members, W_mod, Wd, label_pos, seed, full_k)
            rec = summarise(lab); rec["energy_removed_frac"] = energy
            rec["max_cos_to_u1"] = float(np.abs(V @ u1).max())
            reps.append(rec)
        ok = [c for c in reps if c["result"] == "scored"]
        rhos = np.array([c["rho"] for c in ok]); ps = np.array([c["p_order"] for c in ok])
        es = np.array([c["energy_removed_frac"] for c in reps])
        controls[cname] = dict(
            n_dims_removed=m, n=len(reps), n_scored=len(ok), n_degenerate=len(reps) - len(ok),
            energy_removed_frac=dict(mean=float(es.mean()), min=float(es.min()),
                                     max=float(es.max())),
            rho=dict(mean=float(rhos.mean()), min=float(rhos.min()), max=float(rhos.max()),
                     p05=float(np.percentile(rhos, 5)), median=float(np.median(rhos))),
            p_order=dict(mean=float(ps.mean()), min=float(ps.min()), max=float(ps.max()),
                         median=float(np.median(ps)), frac_below_0p05=float((ps < 0.05).mean())),
            pc1_rho_vs_control=pctile_at_or_below(rhos, out["deflate_pc1_uncentered"].get("rho", np.nan)),
            # one-sided add-one p for "the PC1 deflation hurt ordering more than an
            # arbitrary deflation of the same span (and, for the m-dim control, of
            # the same energy) would have"
            pc1_specificity_p_addone=float(
                (1 + int((rhos <= out["deflate_pc1_uncentered"].get("rho", np.nan)).sum()))
                / (1 + len(rhos))),
            replicates=reps)
        log(f"[{tag}] {cname} (m={m}, E-{es.mean():.3f} mean): rho {rhos.min():+.3f}.."
            f"{rhos.max():+.3f} (mean {rhos.mean():+.3f}), p_order<0.05 in "
            f"{(ps < 0.05).mean()*100:.0f}%, PC1 result at or below "
            f"{controls[cname]['pc1_rho_vs_control']['n_at_or_below']}/{len(ok)}")

    # leave-one-member-out: is the ordering carried by a single latent?
    W_mod[np.asarray(members)] = Wc32                     # restore undeflated block
    loo = []
    for i, mem in enumerate(members):
        sub = [int(x) for x in members if int(x) != int(mem)]
        lab = gate13.labelled(thin, sub, W_mod, label_pos, seed, full_k)
        rec = summarise(lab); rec["dropped_latent"] = int(mem)
        loo.append(rec)
    lr = np.array([r["rho"] for r in loo if r["result"] == "scored"])
    lp = np.array([r["p_order"] for r in loo if r["result"] == "scored"])
    worst = min((r for r in loo if r["result"] == "scored"), key=lambda r: r["rho"])
    loo_sum = dict(n=len(loo), rho=dict(mean=float(lr.mean()), min=float(lr.min()),
                                        max=float(lr.max())),
                   p_order=dict(max=float(lp.max()), frac_below_0p05=float((lp < 0.05).mean())),
                   worst_drop=dict(latent=worst["dropped_latent"], rho=worst["rho"],
                                   p_order=worst["p_order"]),
                   replicates=loo)
    log(f"[{tag}] leave-one-member-out: rho {lr.min():+.4f}..{lr.max():+.4f}, "
        f"p_order<0.05 in {(lp < 0.05).mean()*100:.0f}% of {len(lp)}; worst drop = "
        f"latent {worst['dropped_latent']} (rho {worst['rho']:+.4f}, p {worst['p_order']:.4g})")

    diag = cloud_diagnostics(thin, members, Wc32, label_pos, u1)
    log(f"[{tag}] Spearman(M.u1, calendar) = {diag['spearman_proj_u1_vs_calendar']:+.3f}; "
        f"u1 holds {diag['u1_share_of_cloud_energy']*100:.1f}% of cloud energy")

    # ---- verdict, by the stated interpretation rule -------------------------
    base_pass = bool(base["p_order"] < 0.05)
    pc1 = out["deflate_pc1_uncentered"]
    pc1_scored = pc1["result"] == "scored"
    collapsed = bool(base_pass and (not pc1_scored or pc1["p_order"] >= 0.05))
    matched = controls[f"random_in_span_{m_match}dim"]
    spec_p = matched["pc1_specificity_p_addone"]
    spec = ("specific" if spec_p < 0.05 else
            "marginal" if spec_p < 0.20 else "not_specific")
    if not base_pass:
        call = ("no_baseline_ordering -- the absorption test is undefined here; the "
                "cluster is reported for the decoder-geometry panel only")
    elif collapsed and spec == "specific":
        call = "ABSORPTION -- ordering collapses on removal of one shared direction, " \
               "and the collapse is specific to that direction"
    elif collapsed:
        call = (f"ORDERING IS CARRIED BY ONE SHARED DIRECTION, specificity {spec} "
                f"(p={spec_p:.3f} against the energy-matched arbitrary-subspace "
                f"control): the ordering does not survive removal of PC1, but an "
                f"arbitrary deflation of the same energy destroys it often enough "
                f"that PC1 is not cleanly singled out")
    else:
        call = "DISTRIBUTED -- ordering survives removal of the shared direction"
    verdict = dict(baseline_passes_ordering=base_pass,
                   pc1_deflation_collapses_ordering=collapsed,
                   pc1_specificity_vs_energy_matched_control=spec,
                   pc1_specificity_p_addone=spec_p,
                   near_duplicate_members=bool(pan["n_members_max_cos_gt"]["0.9"] > 0),
                   call=call)
    log(f"[{tag}] VERDICT: {call}")

    res = dict(layer=layer, metric="coactivation", size=len(members),
               members=[int(x) for x in members], gate_seed=int(seed),
               stored_gate_row=stored, thin_reader_codes_identical=identical,
               half="rank-even", n_perm=N_PERM,
               min_firing_per_label=gate13.MIN_FIRING_PER_LABEL,
               min_labels_with_support=gate13.min_labels_with_support(full_k),
               label_support_halfB=[int(len(p)) for p in label_pos],
               decoder_panel=pan, baseline_reproduction=repro,
               conditions=out, single_pc_sweep=sweep, controls=controls,
               leave_one_member_out=loo_sum, cloud_diagnostics=diag,
               span_rank=int(rank), energy_matched_control_dims=int(m_match),
               verdict=verdict, runtime_s=round(time.time() - t0, 1))
    del W, W_mod, thin
    gc.collect()
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="cache/sparse_big")
    ap.add_argument("--l0", default="big")
    ap.add_argument("--out", default="results/absorption_l7.json")
    ap.add_argument("--log", default=None)
    a = ap.parse_args()

    lines = []

    def log(m):
        print(m, flush=True); lines.append(m)

    t0 = time.time()
    tm = json.load(open(ROOT / "results" / "token_map.json"))
    label_ids = [[tm["months"][n]] for n in MONTHS]
    full_k = len(MONTHS)

    l7 = json.load(open(ROOT / "results" / "rerun_gate_big_exploratory_months.json"))
    r7 = [r for r in l7["rows"] if r["layer"] == 7 and r["size"] == 24
          and r["metric"] == "coactivation"][0]
    l9 = json.load(open(ROOT / "results" / "rerun_gate_big_primary_L9.json"))
    r9 = [r for r in l9["rows"] if r["layer"] == 9 and r["size"] == 23
          and r["metric"] == "coactivation"][0]

    def stored_of(r):
        return {k: r[k] for k in ("rho", "p_order", "gap_ratio", "max_gap_deg",
                                  "p_closed", "n_labels", "top2", "dir_gap", "dir_p",
                                  "index_p", "arc_only", "CIRCLE", "ordering_fdr")}

    cases = [("L7_arc", 7, r7["members"], stored_of(r7),
              "results/rerun_gate_big_exploratory_months.json"),
             ("L9_primary", 9, r9["members"], stored_of(r9),
              "results/rerun_gate_big_primary_L9.json")]

    insp = json.load(open(ROOT / "results" / "passer_inspection.json"))

    results = {}
    for tag, layer, members, stored, src in cases:
        log(f"\n=== {tag}: layer {layer}, k={len(members)} (from {src}) ===")
        results[tag] = run_cluster(tag, layer, members, stored, a.root, a.l0,
                                   label_ids, full_k, log)
        results[tag]["source_gate_file"] = src
        ins = [x for x in insp if x["layer"] == layer and x["size"] == len(members)
               and x["metric"] == "coactivation"]
        if ins:
            results[tag]["member_top_tokens"] = {
                str(m["latent"]): m["tokens"][:8] for m in ins[0]["members"]}
            results[tag]["member_temporal_flags"] = {
                str(m["latent"]): dict(month=m["month"], day=m["day"],
                                       temporal=m["temporal"])
                for m in ins[0]["members"]
                if m["month"] or m["day"] or m["temporal"]}
            results[tag]["inspection_counts"] = dict(
                n_month=ins[0]["n_month"], n_day=ins[0]["n_day"],
                n_temporal=ins[0]["n_temporal"])

    doc = dict(
        status="exploratory, not pre-registered",
        title="Absorption test on the layer-7 arc (item 5d)",
        question=("Is the layer-7 arc_only cluster's month ordering ring GEOMETRY, "
                  "or ABSORPTION -- one shared decoder direction that any monotone "
                  "read-out inherits?"),
        interpretation_rule=(
            "Ordering that collapses when one shared direction is removed is "
            "absorption, not ring geometry; ordering that survives is distributed "
            "structure."),
        interpretation_rule_operational=(
            "Collapse = p_order rises above 0.05 (or labelled() degenerates) after "
            "rank-1 deflation along the member block's top principal direction, AND "
            "that collapse is not reproduced by the controls -- neither by removing "
            "an arbitrary rank-1 direction of the same span, nor by removing an "
            "energy-matched arbitrary subspace of the same span. Survival = p_order "
            "stays below 0.05 with rho close to baseline. A collapse that the "
            "energy-matched control also produces says only that the block was "
            "over-deflated, and is not evidence of absorption."),
        method=dict(
            scoring="scripts/13_rerun_gate.py labelled(), imported by path and called unmodified",
            half="rank-even (half B = ODD cache rows)",
            support_rule=f"MIN_FIRING_PER_LABEL={gate13.MIN_FIRING_PER_LABEL}, "
                         f">=5 positions per label, >=ceil(2/3*12)="
                         f"{gate13.min_labels_with_support(full_k)} labels with support",
            n_permutations=N_PERM,
            seed="per-cluster gate seed abs(hash((layer, members[0]))) % 1e6, held "
                 "fixed across every condition; only the decoder block changes",
            deflation="rows of the k x 1152 member decoder block projected onto the "
                      "orthogonal complement of the named direction(s)",
            pc1_definition="primary PC1 = top right-singular vector of the RAW "
                           "(uncentered) member block -- that is the shared direction "
                           "the absorption hypothesis names; centered-PCA and "
                           "mean-direction variants reported alongside",
            random_control=f"{N_RANDOM_CONTROL} isotropic random subspaces sampled "
                           "inside the member block's row space (orthonormal span "
                           "basis, not covariance-weighted), at two sizes: rank-1 "
                           "(removes ~1/rank of the block's energy) and an "
                           "ENERGY-MATCHED m-dim subspace with m = round(rank * "
                           "PC1_share), which removes the same energy as the PC1 "
                           "deflation along an arbitrary slice of the same span. "
                           "PC1 is by construction the maximum-energy single "
                           "direction, so no rank-1 control can be energy-matched; "
                           "the m-dim control is what answers the energy objection.",
            extra_controls="single-PC sweep (deflate PC_j alone, j = 1..rank) and "
                           "leave-one-member-out (drop each latent, rescore)",
            no_forward_passes=True,
            cache=a.root, sae_l0=a.l0, ontology="months",
        ),
        cluster_membership_note=(
            "results/passer_inspection.json: of the 24 L7 members exactly one is a "
            "month latent (359: ' July', ' December', ' October', ' Juli', ' juillet') "
            "and one is a temporal latent (147: 'semester', 'months', 'years'); the "
            "other 22 have no month, day or temporal token in their top-10. The 23 L9 "
            "members contain 0 month, 0 day and 0 temporal latents."),
        results=results,
        runtime_s=round(time.time() - t0, 1),
    )
    Path(ROOT / a.out).write_text(json.dumps(doc, indent=2))
    log(f"\nwrote {a.out} in {doc['runtime_s']} s")
    if a.log:
        Path(ROOT / a.log).write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
