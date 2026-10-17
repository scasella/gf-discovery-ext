"""Does a YEARS structure exist in the residual stream? (supervised existence sweep)

Why this arm exists. The base-10 helix arm (`scripts/15_helix_existence.py`,
`results/helix_existence.json`) used a target with no published positive at this scale
and, worse, a readout position that was confounded: every template ended in the
placeholder, so the captured position was the number's FINAL DIGIT, and `value mod 10`
IS the identity of the token being read. A closed T=10 ring there can be digit-token
embedding geometry. The unreported signature of exactly that confound is visible in the
old run: gain_T10 = 0.316 at layer 0 falling monotonically to 0.042 at layer 25, and a
T=5 > T=10 crossover from layer 17 on.

This arm changes two things.

  1. TARGET. Years of the 20th century, 1900-1999, the ontology with a published
     circular/multi-dimensional result at small scale, and the target Goodfire's own
     helix proxy uses ("The date is {year}").

  2. CAPTURE POSITION -- the fix. Gemma 3 digit-splits, so "1947" is four tokens
     '1','9','4','7'. EVERY template here ends with a fixed suffix AFTER the year and
     the readout is the last non-pad token, which is always '.'. No digit of the year is
     ever read. The tokenization audit in the output records this for all 100 years and
     asserts the readout token is constant.

What the fix does and does NOT buy -- stated before results, because it decides the
verdict. Hooks are on `model.layers[i]` OUTPUT, i.e. post-block, so even layer 0 has had
one attention pass: information can be COPIED from a year digit into the '.' position at
every layer including the first. So:

  CONFOUND-FREE claims (require combining >= 2 digit tokens, no single token carries them)
    * T=100 gain and century-ring closure: position within the century is a function of
      the tens AND units digits together.
    * linear-time Spearman ordering across 1900..1999: same reason.

  COPY-EXPLICABLE claims (a single year token carries them; report, but flagged)
    * T=10 gain and the units ring: v mod 10 is the units-digit token.
    * the tens ring: v // 10 is the tens-digit token.

  For the copy-explicable pair a secondary discriminator is reported: the R^2 gain of the
  single sinusoid at that period over the gain of the full 9-dummy one-hot code for the
  same digit. A copied token identity has no reason to be sinusoidal; a helix does.
  `sinusoidality` near 1 means the digit's contribution is one frequency, not a lookup.

Convention: v = year - 1900, so v in 0..99. 1900 is divisible by 100 and hence by 2, 5
and 10, so cos/sin(2*pi*year/T) == cos/sin(2*pi*v/T) exactly for every period used, and
the linear column differs by a constant absorbed by the intercept. The two encodings are
numerically identical here; the offset is used only for conditioning.

Nulls. Carried over from script 15, where the battery forced them:
  * gains -- shuffled-values permutation. `fit_helix` adds free columns, so gain > 0 is
    mechanically guaranteed and means nothing on its own.
  * ring closure -- shuffled values AFTER detrending on the TRUE values. The two rejected
    alternatives are documented at length in script 15: uniform random angles does not
    apply because the plane is fit-selected, and undetrended shuffling inflates the null
    because it destroys the linear structure too. `p_closed_uniform` is retained for
    comparability with `closure_depth.json` and decides nothing.
  * the PCA ring is the one place the uniform-angle null IS valid, because that plane is
    chosen without reference to the labels (the script 12 convention).

Battery. Script 15's battery certifies the k=10 decade path only. Three cases are added
for the century path, and the third is the one that matters: a MONOTONE BUT CURVED ramp
loads on the T=100 cos/sin pair and is the realistic false positive for "century circle".
It is an open arc, so gap_ratio must kill it. Whether the null separates it is reported as
`separates_curved_monotone`; the century claims are only worth anything if that is true.

Two battery decisions were made BEFORE any real activation was analyzed, on synthetic data
whose answer is known. Both are recorded rather than quietly applied.

  (a) The century ring is scored PRIMARILY at k=20 (five-year bins), not k=100.
      At k=100 each ring point is one noisy year and the detrended-shuffle null is
      contaminated -- a closed curve projected into a permuted-label plane is still
      closed -- so the test has little power. Measured (`century_k_choice`, 300 perms,
      p_closed for a planted century circle / linear ramp / curved monotone / planted
      decade helix):
          k=100   0.1429 / 0.8439 / 0.9934 / 0.9834   <- planted circle MISSES at 0.05
          k= 20   0.0033 / 0.9801 / 1.0000 / 0.9037   <- clean separation on all four
          k= 10   0.0066 / 0.6113 / 1.0000 / 0.9701   <- also separates, but k=20 is larger
      k=100 is still reported, flagged low-power. k=20 was chosen on this table, before
      any real activation was analyzed.

      TRANSCRIPTION CORRECTED 2026-08-20 (docstring only; no code changed). The table
      above previously read k=100 0.0532/0.7143/1.0000/1.0000, k=20 0.0033/0.9302/
      1.0000/0.9601, k=10 0.0033/0.9037/1.0000/0.9635. `century_k_choice` is fully
      deterministic (n_perm=300, seeds `seed + k`), and a verification re-run of this
      file reproduced `results/years_existence.json` bit-for-bit -- including the grid
      now quoted -- so the old numbers were a stale transcription from a pre-final
      version of the code, not a different measurement of it. The SELECTION RULE and
      its OUTCOME are unchanged under the corrected numbers: k=100 still fails to
      detect the planted circle at alpha=0.05, k=20 is still the largest k that
      separates all four cases, and k=20 remains the primary statistic.

  (b) The linear-ramp and curved-monotone CLOSURE assertions are five-draw calibrations
      (at most one trip in five), not single-seed pass/fail. After detrending on the true
      values a pure ramp reduces to exchangeable noise for the closure statistic, so its
      p-value is uniform by construction and a single seed is a coin flip at the 5% level.
      Script 15 asserted on one seed and got lucky; the first draw here happened to score
      units p_closed = 0.0198. Eight independent ramp draws were then measured: 0/8 trips
      on p_gain, p_gain_T10, p_gain_T100, units closure and century closure. The GAIN
      assertions stay all-draws, because they are not coin flips (every ramp draw scored
      p_gain = 1.0000).

POST-VERIFICATION ADDITIONS, 2026-08-20 (second dated note; the transcription correction
above is kept as audit trail). An adversarial verification pass recomputed this arm and
confirmed the scientific verdict, but found four defects in how the statistics were
CLASSIFIED and REPORTED. Every fix below is additive: the pre-existing code path, its
seeds and its `synthetics()["passed"]` gate are untouched, and a leaf-by-leaf diff of the
regenerated output against the pre-fix output shows zero numeric drift on every path that
existed before. The new blocks are DIAGNOSTIC, NOT pre-registered gates: they were written
after the real numbers were seen, their amplitudes are matched to OBSERVED gains and their
layers are chosen by rules stated with knowledge of the results. That provenance is
recorded in each block rather than dressed up as a prior declaration.

  1. AMPLITUDE GAP IN THE BATTERY (`amplitude_matched_power`, new). `separates_curved_
     monotone` was certified at a planted T=100 gain of 0.6523 against a curved-monotone
     0.0218, while the real layers sit at 0.0380-0.3016 -- neither side bracketed the
     observed regime, so the flag licensing every century claim had never been evaluated
     where it mattered. Worse, the power-law curved-monotone family SATURATES near T=100
     gain 0.021 and cannot be amplitude-matched at all, so it was never the realistic
     false positive for a 0.19-gain observation. The new leg plants closed and open curves
     into the REAL per-layer background (T=100 component projected out) with the amplitude
     bisected until the T=100 gain equals that layer's OBSERVED gain, and asks whether
     century20 closure detects them. Without this the negative would be uninterpretable:
     "no closure" is only a finding if closure would have been seen had it been there.

  2. rings.century.p_closed (k=100) WAS MISCLASSIFIED as "certified but low power". It is
     also NON-SPECIFIC: an open three-quarter arc planted at the observed gain trips it
     while century20 correctly does not. It moves to a new `non_specific` tier. This never
     bit the verdict (k=100 never fired on real data) and it retroactively validates the
     k=20-primary rule declared above, but the label was wrong as written.

  3. rings.*.p_order (the spec's rho_mod10 deliverable) WAS REPORTED WITHOUT A FLAG. It is
     a fit-selection artifact: the cos/sin plane is fitted WITH the true labels, so the
     group means come out in period order by construction, and a permutation null over a
     FIXED angle set cannot correct for that. Measured per ring, against a nominal 0.25
     trips in five (`synthetics.ring_order_calibration`, five draws per family):
         units       5/5 on linear ramp, 5/5 on curved monotone, 5/5 on PURE NOISE
         century     5/5, 5/5, 5/5   (k=100, same story)
         century20   3/5, 3/5, 2/5
         tens        3/5, 3/5, 1/5
     The units and century statistics also sit at the permutation floor on both planted
     cases and on the single ramp/curved diagnostic rows. On real activations the two
     ruined statistics are exactly the loud ones: units fires at 22/26 layers and century
     at 24/26, INCLUDING every late layer where the T=10 gain is dead. The units statistic
     fires on 3/3 layers of the noun control, on 12/26 shuffled-label layers, and on 23 of
     80 independent label shuffles at real geometry (28.8% against a nominal 5%); the tens
     and century20 order statistics trip 1/26 and 3/26 under the fixed shuffle. All four
     join the disqualified tier -- the two calmer ones because the same fit-selection
     mechanism applies to them and their lower trip rate reflects coarser binning, not a
     valid null. Recording this costs nothing: `cyclic_order_score` was already computed.

  4. THE SHUFFLED-LABELS CLOSURE ANOMALY WAS OMITTED from the reported controls. The one
     fixed shuffle in `rows_shuffled_labels` trips century20 at 5/26 layers, tens at 11/26
     and century k=100 at 12/26 -- which looks like an anti-conservative null until you
     notice it is ONE draw spread across 26 correlated layers. `shuffle_calibration` now
     draws independent shuffles at fixed layers and reports the trip rate, so the anomaly
     and its resolution appear together.

  Reporting fix, no code: a machine-generated `summary` block is now written into the
  output, so the headline sentences are derived from the rows instead of transcribed by
  hand. The first analyst report of this arm said the T=10 gain was "dead from L7 on";
  layer 10 scores p = 0.0249. `summary.gain_T10.layers_significant` cannot make that
  mistake.
"""
import argparse, json, os, sys, time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from gfdisc.env import DEVICE
from gfdisc.model import load_model, capture
from gfdisc.fits import (fit_helix, fit_circle, circularity, is_ring, angles,
                         cyclic_order_score, pca_2d, degenerate)

PERIODS = (2, 5, 10, 100)

# Every template ends with a fixed suffix after the year and terminates in '.', so the
# last non-pad token is never a digit. `group` splits them by how far the readout sits
# from the year: "immediate" = the period directly follows the final digit, "distal" =
# two or more tokens intervene. Reported separately, because a one-hop attention copy is
# cheapest at zero distance.
YEAR_TEMPLATES = [
    ("The date is {}.", "immediate"),
    ("The year was {}.", "immediate"),
    ("She was born in {}.", "immediate"),
    ("The report was published in {}.", "immediate"),
    ("The war ended in {}.", "immediate"),
    ("The building was completed in {}.", "immediate"),
    ("Construction began in {}.", "immediate"),
    ("He retired in {}.", "immediate"),
    ("In {}, everything changed.", "distal"),
    ("By {}, the town had grown.", "distal"),
    ("Records from {} are missing.", "distal"),
    ("The census of {} is complete.", "distal"),
    ("Since {}, the rules have applied.", "distal"),
    ("The photograph, taken in {}, is faded.", "distal"),
    ("Documents dated {} were filed away.", "distal"),
    ("The treaty signed in {} still holds.", "distal"),
]

# Matched-arity negative control: same 100-label shape, same readout structure, but the
# labels carry no order. The "value" is the ALPHABETICAL index, declared here before any
# result is seen -- there is no other defensible ordering for a noun list, and picking one
# after looking would be the selection step this project exists to avoid.
NOUN_TEMPLATES = [
    ("The word is {}.", "immediate"),
    ("The label was {}.", "immediate"),
    ("She wrote about {}.", "immediate"),
    ("The report described the {}.", "immediate"),
    ("The story ended with {}.", "immediate"),
    ("The drawing showed a {}.", "immediate"),
    ("Discussion began with {}.", "immediate"),
    ("He asked about {}.", "immediate"),
    ("About {}, everything changed.", "distal"),
    ("Near {}, the town had grown.", "distal"),
    ("Records about {} are missing.", "distal"),
    ("The description of {} is complete.", "distal"),
    ("Since {}, the rules have applied.", "distal"),
    ("The picture, showing {}, is faded.", "distal"),
    ("Documents about {} were filed away.", "distal"),
    ("The note about {} still holds.", "distal"),
]

NOUNS = sorted([
    "anchor", "apple", "arrow", "badge", "basket", "beach", "bicycle", "blanket",
    "bottle", "branch", "bridge", "bucket", "camera", "candle", "carpet", "castle",
    "chair", "cheese", "church", "circle", "cloud", "coffee", "corner", "cotton",
    "curtain", "desert", "diamond", "doctor", "dollar", "dragon", "drawer", "engine",
    "envelope", "factory", "farmer", "feather", "fence", "finger", "flower", "forest",
    "garden", "glass", "guitar", "hammer", "harbor", "helmet", "hospital", "island",
    "jacket", "kitchen", "ladder", "lantern", "letter", "library", "marble", "market",
    "mirror", "mountain", "needle", "orange", "orchard", "painting", "pencil", "piano",
    "pillow", "planet", "pocket", "potato", "printer", "puzzle", "rabbit", "ribbon",
    "river", "rocket", "saddle", "sailor", "school", "shadow", "shovel", "silver",
    "singer", "soldier", "spider", "station", "statue", "stone", "string", "sugar",
    "table", "teacher", "temple", "ticket", "tiger", "tower", "tractor", "tunnel",
    "umbrella", "valley", "village", "window",
])

# Pre-declared, before any result: the three layers the noun control is scored at.
# Early / mid / late, evenly spaced over the 26-layer stack.
NOUN_LAYERS = (3, 13, 22)


# ---------------------------------------------------------------------------
# design matrices and gains
# ---------------------------------------------------------------------------
def design(v, use_periodic=True, periods=PERIODS):
    v = np.asarray(v, dtype=np.float64)
    cols = [np.ones_like(v), v]
    if use_periodic:
        for T in periods:
            cols += [np.cos(2 * np.pi * v / T), np.sin(2 * np.pi * v / T)]
    return np.column_stack(cols)


def _r2(B, X):
    coef, *_ = np.linalg.lstsq(B, X, rcond=None)
    resid = X - B @ coef
    return 1 - (resid ** 2).sum() / max((X ** 2).sum(), 1e-12)


def _linear(v):
    v = np.asarray(v, dtype=np.float64)
    return np.column_stack([np.ones_like(v), v])


def single_period_gain(X, v, T, basis=None):
    """R^2 gain of ONE cos/sin pair over the linear-in-v baseline.

    `basis` lets the sinusoid run in a derived quantity (the tens digit) while the
    baseline stays linear in v, so every gain in the row is against the same reference.
    """
    v = np.asarray(v, dtype=np.float64)
    u = v if basis is None else np.asarray(basis, dtype=np.float64)
    lin = _linear(v)
    per = np.column_stack([lin, np.cos(2 * np.pi * u / T), np.sin(2 * np.pi * u / T)])
    return float(_r2(per, X) - _r2(lin, X))


def onehot_gain(X, v, codes, k):
    """R^2 gain of the full (k-1)-dummy code for `codes`, over the same linear baseline.

    The ceiling a single sinusoid at that period could reach if the digit's contribution
    were an arbitrary lookup rather than one frequency.
    """
    c = np.asarray(codes, dtype=int)
    D = np.zeros((len(c), k - 1))
    for j in range(1, k):
        D[c == j, j - 1] = 1.0
    return float(_r2(np.column_stack([_linear(v), D]), X) - _r2(_linear(v), X))


def detrend(X, values):
    """Remove constant + linear-in-value, using the TRUE values (see module docstring)."""
    B = design(values, use_periodic=False)
    coef, *_ = np.linalg.lstsq(B, X, rcond=None)
    return X - B @ coef


# ---------------------------------------------------------------------------
# rings
# ---------------------------------------------------------------------------
def ring_spec(name, v):
    """(basis for the cos/sin plane, period, group code, k) for each ring tested."""
    v = np.asarray(v, dtype=int)
    if name == "units":                       # copy-explicable: v % 10 is the units token
        return v.astype(np.float64), 10.0, v % 10, 10
    if name == "tens":                        # copy-explicable: v // 10 is the tens token
        return (v // 10).astype(np.float64), 10.0, v // 10, 10
    if name == "century20":                   # confound-free, PRIMARY: 5-year bins
        return v.astype(np.float64), 100.0, v // 5, 20
    if name == "century":                     # confound-free, low power at k=100
        return v.astype(np.float64), 100.0, v, 100
    raise ValueError(name)


def _block_perm(v, rng):
    """Permute values WITHIN each decade block. The matched null for the units ring.

    Why this exists -- it is the third time in this project that a free permutation null
    turned out to be inflated, and the mechanism is the same one script 15 documented for
    the undetrended null, resurfacing because detrending removes only the LINEAR part of
    the trend. A true mod-10 group is {r, 10+r, ..., 90+r}: exactly one value from each
    decade, spread evenly over the range. A FREE permutation's null group is a random
    10-subset, whose mean is spread around by any residual smooth (non-linear) trend that
    detrending left behind. That inflates the null gap_ratio, and an inflated null makes
    everything look closed. Permuting within decade blocks keeps every null group at one
    value per decade -- the composition the true groups have -- while still destroying the
    mod-10 phase.

    Measured on the battery cases (300 perms each, null gap_ratio mean / trips at 0.05
    over five draws), free null vs this one:
        curved monotone   free 1.78, 3/5 trips   ->  block 1.25, 1/5 trips
        pure noise        free 1.24, 0/5         ->  block 1.24, 0/5
        linear ramp       free 1.25, 0/5         ->  block 1.25, 0/5
        planted decade helix   p = 0.003 under BOTH -- the matched null costs no power.
    The century and tens rings keep the free null: their groups are contiguous blocks, for
    which the same asymmetry runs the SAFE way (a residual trend spreads the TRUE group
    means into an arc and clumps the null's, so the test becomes conservative, not
    anti-conservative). Their measured calibration is in `synthetics.calibration`.
    """
    a = np.asarray(v, dtype=int)
    assert a.shape == (100,) and (np.sort(a) == np.arange(100)).all(), \
        "_block_perm assumes the value set is exactly 0..99"
    a = a.reshape(10, 10).copy()
    for b in range(10):
        a[b] = rng.permutation(a[b])
    return a.reshape(-1)


RING_NULL = dict(units="within-decade-block permutation (matched group composition)",
                 tens="free permutation", century20="free permutation",
                 century="free permutation")


def ring_points(Xr, basis_vals, T, groups, k):
    """Project detrended points onto the fitted cos/sin plane and average by group.

    The two coefficient vectors are not orthogonal in general, so they are orthonormalised
    before projecting; a shear would otherwise manufacture or destroy angular evenness.
    """
    u = np.asarray(basis_vals, dtype=np.float64)
    B = np.column_stack([np.cos(2 * np.pi * u / T), np.sin(2 * np.pi * u / T)])
    coef, *_ = np.linalg.lstsq(B, Xr, rcond=None)
    Q, _ = np.linalg.qr(coef.T)
    if Q.shape[1] < 2:
        return None
    xy_all = Xr @ Q
    g = np.asarray(groups, dtype=int)
    pts = []
    for r in range(k):
        m = g == r
        if not m.any():
            return None
        pts.append(xy_all[m].mean(0))
    return np.vstack(pts)


def gap_pvalue(k, observed_max_gap, n_perm=20000, seed=0):
    """Max angular gap against uniform random angles -- the script 12 null."""
    rng = np.random.default_rng(seed)
    a = np.sort(rng.uniform(0, 2 * np.pi, size=(n_perm, k)), axis=1)
    g = np.diff(np.concatenate([a, a[:, :1] + 2 * np.pi], axis=1), axis=1).max(1)
    return float((1 + (g <= observed_max_gap).sum()) / (1 + n_perm))


def _circ(xy):
    cx, cy, r, resid = fit_circle(xy)
    c = circularity(xy, cx, cy, r)
    return cx, cy, r, resid, c


# ---------------------------------------------------------------------------
# ordering
# ---------------------------------------------------------------------------
def _ranks(a):
    o = np.argsort(np.asarray(a, dtype=np.float64), kind="stable")
    r = np.empty(len(o), dtype=np.float64)
    r[o] = np.arange(len(o), dtype=np.float64)
    return r


def spearman(a, b):
    ra, rb = _ranks(a), _ranks(b)
    ra = ra - ra.mean(); rb = rb - rb.mean()
    return float(ra @ rb / np.sqrt((ra @ ra) * (rb @ rb)))


def linear_ordering(X, v, n_perm=20000, seed=0):
    """Spearman of the top principal component against chronological order.

    `np.linalg.svd` fixes no sign, so the two SVD sign conventions give rho and -rho.
    Both are reported and the p-value is two-sided on |rho|; a sign-blind statistic is
    the only honest one when the basis has no preferred handedness.
    """
    _, ratios, Vt = pca_2d(X)
    pc1 = (X - X.mean(0, keepdims=True)) @ Vt[0]
    rho = spearman(pc1, v)
    rng = np.random.default_rng(seed)
    null = np.array([abs(spearman(pc1, rng.permutation(np.asarray(v, dtype=np.float64))))
                     for _ in range(n_perm)])
    return dict(rho_pc1_svd_plus=rho, rho_pc1_svd_minus=-rho, abs_rho=abs(rho),
                p_order_shuffled=float((1 + (null >= abs(rho)).sum()) / (1 + n_perm)),
                null_abs_rho_mean=float(null.mean()),
                null_abs_rho_p95=float(np.percentile(null, 95)),
                pc1_var_frac=float(ratios[0]), top2_var_frac=float(ratios[:2].sum()),
                n_perm=n_perm)


# ---------------------------------------------------------------------------
# the per-layer row
# ---------------------------------------------------------------------------
RINGS = ("units", "tens", "century20", "century")


def years_row(points, values, n_null=200, seed=0, order_perm_k10=20000,
              order_perm_k100=2000, order_perm_linear=20000):
    """Everything measured at one layer, with every p-value from a shuffled-values null.

    `order_perm_k100` is 2000 rather than 20000: `cyclic_order_score` maximises over 2*k
    symmetries, so k=100 costs 200 Spearman evaluations per permutation. The p floor is
    therefore 1/2001 = 5.0e-4 for the k=100 century and PCA orderings, and 1/20001 =
    5.0e-5 for the k=10 and k=20 rings. Recorded per ring as `n_perm_order`.
    """
    X = np.asarray(points, dtype=np.float64)
    X = X - X.mean(0, keepdims=True)
    v = np.asarray(values, dtype=int)
    vf = v.astype(np.float64)
    tens = (v // 10).astype(np.float64)

    spread = float(np.linalg.norm(X, axis=1).mean())
    degen = bool(degenerate(points))

    h = fit_helix(X, vf, periods=PERIODS)
    gain = h["r2_helix"] - h["r2_linear"]
    by_period = {str(T): single_period_gain(X, vf, T) for T in PERIODS}
    gain_tens_ring = single_period_gain(X, vf, 10.0, basis=tens)

    # sinusoidality: single-frequency gain over the arbitrary-lookup ceiling for the
    # same digit. ~1 = the digit enters as one frequency; ~0 = it is a lookup table.
    g_units_oh = onehot_gain(X, vf, v % 10, 10)
    g_tens_oh = onehot_gain(X, vf, v // 10, 10)

    Xr = detrend(X, vf)

    rings, obs_gr = {}, {}
    for name in RINGS:
        bas, T, grp, k = ring_spec(name, v)
        xy = ring_points(Xr, bas, T, grp, k)
        cx, cy, r, resid, c = _circ(xy)
        npm = order_perm_k10 if k <= 20 else order_perm_k100
        s = cyclic_order_score(angles(xy, cx, cy), np.arange(k), k, n_perm=npm, seed=seed)
        rings[name] = dict(k=k, resid=float(resid), radius_ratio=c["radius_ratio"],
                           gap_ratio=c["gap_ratio"], max_gap_deg=c["max_gap_deg"],
                           spread=c["spread"], ring_flag=is_ring(c),
                           p_closed_uniform=gap_pvalue(k, np.radians(c["max_gap_deg"]),
                                                       seed=seed),
                           rho=s["rho"], p_order=s["p"], symmetry=list(s["symmetry"]),
                           n_perm_order=npm)
        obs_gr[name] = c["gap_ratio"]

    # label-free plane: the ONE place the uniform-angle null is the right null.
    xy_pca, ratios, _ = pca_2d(X)
    cx, cy, r, resid, cp = _circ(xy_pca)
    s_pca = cyclic_order_score(angles(xy_pca, cx, cy), np.arange(len(v)), len(v),
                               n_perm=order_perm_k100, seed=seed)
    pca_ring = dict(k=len(v), top2_var_frac=float(ratios[:2].sum()), resid=float(resid),
                    radius_ratio=cp["radius_ratio"], gap_ratio=cp["gap_ratio"],
                    max_gap_deg=cp["max_gap_deg"], ring_flag=is_ring(cp),
                    p_closed_uniform=gap_pvalue(len(v), np.radians(cp["max_gap_deg"]),
                                                seed=seed),
                    rho=s_pca["rho"], p_order=s_pca["p"], n_perm_order=order_perm_k100)

    rng = np.random.default_rng(seed)
    null_gain = np.empty(n_null)
    null_period = {str(T): np.empty(n_null) for T in PERIODS}
    null_tens_ring = np.empty(n_null)
    null_gr = {name: np.empty(n_null) for name in RINGS}
    for i in range(n_null):
        vp = rng.permutation(v)
        vb = _block_perm(v, rng)          # matched null, units ring only (see _block_perm)
        vpf = vp.astype(np.float64)
        hp = fit_helix(X, vpf, periods=PERIODS)
        null_gain[i] = hp["r2_helix"] - hp["r2_linear"]
        for T in PERIODS:
            null_period[str(T)][i] = single_period_gain(X, vpf, T)
        null_tens_ring[i] = single_period_gain(X, vpf, 10.0, basis=(vp // 10).astype(float))
        for name in RINGS:
            vv = vb if name == "units" else vp
            bas, T, grp, k = ring_spec(name, vv)
            xyp = ring_points(Xr, bas, T, grp, k)
            _, _, _, _, cn = _circ(xyp)
            null_gr[name][i] = cn["gap_ratio"]

    for name in RINGS:
        g, nl = obs_gr[name], null_gr[name]
        # smaller gap_ratio = more evenly closed, so the tail is the LOW side
        rings[name].update(p_closed=float((1 + (nl <= g).sum()) / (1 + n_null)),
                           null_gap_mean=float(nl.mean()),
                           null_gap_p05=float(np.percentile(nl, 5)),
                           null_kind=RING_NULL[name])

    p_by_period = {str(T): float((1 + (null_period[str(T)] >= by_period[str(T)]).sum())
                                 / (1 + n_null)) for T in PERIODS}
    null_p95_by_period = {str(T): float(np.percentile(null_period[str(T)], 95))
                          for T in PERIODS}

    stats = dict(
        n_labels=len(v), spread=spread, degenerate=degen,
        r2_helix=h["r2_helix"], r2_linear=h["r2_linear"], gain=float(gain),
        p_gain=float((1 + (null_gain >= gain).sum()) / (1 + n_null)),
        null_gain_mean=float(null_gain.mean()),
        null_gain_p95=float(np.percentile(null_gain, 95)),
        gain_by_period=by_period, p_gain_by_period=p_by_period,
        null_p95_by_period=null_p95_by_period,
        gain_tens_ring=gain_tens_ring,
        p_gain_tens_ring=float((1 + (null_tens_ring >= gain_tens_ring).sum()) / (1 + n_null)),
        gain_units_onehot=g_units_oh, gain_tens_onehot=g_tens_oh,
        sinusoidality_units=float(by_period["10"] / g_units_oh) if g_units_oh > 1e-12 else None,
        sinusoidality_tens=float(gain_tens_ring / g_tens_oh) if g_tens_oh > 1e-12 else None,
        rings=rings, pca_ring=pca_ring,
        ordering=linear_ordering(X, vf, n_perm=order_perm_linear, seed=seed),
        n_null=n_null)
    return stats


# ---------------------------------------------------------------------------
# synthetic battery
# ---------------------------------------------------------------------------
def _synth_case(kind, seed, d=1152, noise=0.15):
    """One synthetic point cloud whose answer is known. Fresh directions per seed."""
    rng = np.random.default_rng(seed)
    v = np.arange(100)
    a, b, c, e, f = (rng.normal(size=d) for _ in range(5))
    if kind == "decade_helix":
        return (np.outer(v / 50.0, a) + np.outer(np.cos(2 * np.pi * v / 10), b)
                + np.outer(np.sin(2 * np.pi * v / 10), c) + noise * rng.normal(size=(100, d)))
    if kind == "century_circle":
        return (0.3 * np.outer(v / 50.0, a) + np.outer(np.cos(2 * np.pi * v / 100), e)
                + np.outer(np.sin(2 * np.pi * v / 100), f) + noise * rng.normal(size=(100, d)))
    if kind == "linear_ramp":
        return np.outer(v / 50.0, a) + noise * rng.normal(size=(100, d))
    if kind == "curved_monotone":
        # monotone in v but strongly curved, so it loads on the T=100 cos/sin pair.
        # This is the realistic false positive for "century circle": an OPEN ARC.
        return (np.outer((v / 99.0) ** 1.6, a) + 0.5 * np.outer(v / 99.0, b)
                + noise * rng.normal(size=(100, d)))
    if kind == "noise":
        return rng.normal(size=(100, d))
    raise ValueError(kind)


def century_k_choice(n_perm=300, seed=0):
    """Why the century ring is scored at k=20. Run before any real activation is analyzed.

    The detrended-shuffle null re-fits the T=100 plane under permuted labels and projects
    the SAME points into it. When the structure really is a closed planar curve, almost any
    plane containing it yields a closed projection, so the null is contaminated and the test
    loses power -- badly at k=100, where every ring point is one noisy year. Binning years
    into groups averages that noise down. Measured here, reproducibly, on the four cases.
    """
    v = np.arange(100)
    cases = {nm: _synth_case(kind, s) for nm, kind, s in (
        ("planted_century_circle", "century_circle", 2),
        ("linear_ramp", "linear_ramp", 3),
        ("curved_monotone", "curved_monotone", 4),
        ("planted_decade_helix", "decade_helix", 1))}
    grid = {}
    for nm, A in cases.items():
        A = A - A.mean(0, keepdims=True)
        Xr = detrend(A, v.astype(np.float64))
        row = {}
        for k, gf in ((100, lambda x: x), (20, lambda x: x // 5), (10, lambda x: x // 10)):
            xy = ring_points(Xr, v.astype(np.float64), 100.0, gf(v), k)
            _, _, _, _, c = _circ(xy)
            obs = c["gap_ratio"]
            rng = np.random.default_rng(seed + k)
            nl = np.empty(n_perm)
            for i in range(n_perm):
                vp = rng.permutation(v)
                _, _, _, _, cn = _circ(ring_points(Xr, vp.astype(np.float64), 100.0, gf(vp), k))
                nl[i] = cn["gap_ratio"]
            row[f"k{k}"] = dict(gap_ratio=obs, radius_ratio=c["radius_ratio"],
                                null_mean=float(nl.mean()),
                                null_p05=float(np.percentile(nl, 5)),
                                p_closed=float((1 + (nl <= obs).sum()) / (1 + n_perm)))
        grid[nm] = row
    return dict(n_perm=n_perm, grid=grid,
                chosen="century20 (k=20, five-year bins)",
                rule="the primary confound-free closure statistic is the largest-k century "
                     "ring that separates all four synthetic cases at alpha=0.05; k=100 is "
                     "retained in the output but flagged low-power",
                declared="chosen on this table BEFORE any real activation was analyzed")


# NOISE_KEYS is the GATE set and is frozen: `synthetics()["passed"]` reads it, so adding a
# key here would change the abort condition. The ring order statistics are RECORDED only.
NOISE_KEYS = ("p_gain", "p_gain_t10", "p_gain_t100", "p_closed_units",
              "p_closed_century20", "p_closed_century", "p_order_linear")

# added 2026-08-20 (post-verification, item 3): the ring cyclic-order statistic, recorded
# on every negative family so its false-positive rate is measured rather than assumed.
RING_ORDER_KEYS = ("p_order_ring_units", "p_order_ring_tens",
                   "p_order_ring_century20", "p_order_ring_century")
RECORD_KEYS = NOISE_KEYS + RING_ORDER_KEYS


def _pkeys(r):
    d = dict(p_gain=r["p_gain"],
             p_gain_t10=r["p_gain_by_period"]["10"],
             p_gain_t100=r["p_gain_by_period"]["100"],
             p_closed_units=r["rings"]["units"]["p_closed"],
             p_closed_century20=r["rings"]["century20"]["p_closed"],
             p_closed_century=r["rings"]["century"]["p_closed"],
             p_order_linear=r["ordering"]["p_order_shuffled"])
    for nm in RINGS:
        d[f"p_order_ring_{nm}"] = r["rings"][nm]["p_order"]
    return d


def synthetics(n_null=100, fast_order=200):
    """Validate every statistic this script reports, on data whose answer is known.

    Planted cases are single-draw and gated only on the statistic they are planted in --
    every extra assertion carries its own false-fail rate.

    NEGATIVE cases (linear ramp, curved monotone, pure noise) are FIVE-DRAW CALIBRATIONS
    for the closure statistics, at most one trip in five. After detrending on the true
    values a ramp reduces to exchangeable noise for closure, so its p-value is uniform by
    construction and one seed is a coin flip at the 5% level -- the first draw here scored
    units p_closed = 0.0198 on exactly that lottery, and eight independent ramp draws then
    gave 0/8 trips. The GAIN assertions stay all-draws, because they are not coin flips:
    every ramp draw scored p_gain = 1.0000.

    Note the gate itself is not free: under uniform p, "at most one trip in five" fails
    about 2.3% of the time per statistic. A future re-run that trips it is not automatically
    a broken statistic.
    """
    kw = dict(n_null=n_null, order_perm_k10=fast_order, order_perm_k100=fast_order,
              order_perm_linear=fast_order)
    out = {"century_k_choice": century_k_choice()}

    out["planted_decade_helix"] = years_row(_synth_case("decade_helix", 1), np.arange(100),
                                            seed=1, **kw)
    out["planted_century_circle"] = years_row(_synth_case("century_circle", 2), np.arange(100),
                                              seed=2, **kw)
    # kept for comparability with the pre-rewrite log and with script 15's single-seed style
    out["linear_ramp"] = years_row(_synth_case("linear_ramp", 3), np.arange(100), seed=3, **kw)
    out["curved_monotone"] = years_row(_synth_case("curved_monotone", 4), np.arange(100),
                                       seed=4, **kw)

    # five-draw calibrations, on seeds not used by any diagnostic run
    cal = {}
    for kind, seeds in (("linear_ramp", range(200, 205)),
                        ("curved_monotone", range(210, 215)),
                        ("noise", range(220, 225))):
        draws = []
        for s in seeds:
            r = years_row(_synth_case(kind, s), np.arange(100), seed=s, **kw)
            draws.append(dict(seed=s, **_pkeys(r)))
        cal[kind] = dict(draws=draws,
                         n_sig={k: sum(x[k] <= 0.05 for x in draws) for k in RECORD_KEYS})
    out["calibration"] = cal
    out["pure_noise"] = cal["noise"]          # name kept for continuity with script 15

    # The ORDERING statistic is calibrated on pure noise only, and deliberately so: a
    # linear ramp IS ordered, so `p_order_linear` firing 5/5 on the ramp and curved cases
    # is the statistic working, not failing. Measured: 5/5 on both trended families, 0/5
    # on noise. That is the positive control for ordering -- and a restatement of the fact
    # this project keeps hitting, that ordering alone is never evidence of a ring.
    # UNITS-RING CLOSURE IS DISQUALIFIED AS EVIDENCE. It fires on curved monotone data at
    # 2/5 with the matched null (3/5 with the free one), against a nominal 0.25/5. The
    # matched null cut the inflation but did not remove it, and the correct response to a
    # statistic the battery has shown to be anti-conservative is to stop treating it as
    # evidence -- not to keep tuning the null until it passes. It is still computed and
    # still reported, with its measured failure rate attached, exactly as script 15 kept
    # its two disqualified nulls in the record.
    #
    # This removes a statistic from the evidence base rather than admitting one, and costs
    # the verdict nothing: the units ring was ALREADY in the copy-explicable tier by the
    # claim classification declared at the top of this file, so no claim rested on it.
    # The confound-free primary, century20, is clean on every negative family.
    trended = ("linear_ramp", "curved_monotone")
    certified_closure = ("p_closed_century20", "p_closed_century")
    gain_keys = ("p_gain", "p_gain_t10", "p_gain_t100")
    core = (
        # pure noise: nominal false-positive rate on EVERY statistic reported
        all(cal["noise"]["n_sig"][s] <= 1 for s in NOISE_KEYS)
        # trended negatives: certified closure must not fire, gains must never fire
        and all(cal[k]["n_sig"][s] <= 1 for k in trended for s in certified_closure)
        and all(d[s] > 0.05 for k in trended for d in cal[k]["draws"] for s in gain_keys)
        # planted cases, each gated only on the statistic it is planted in
        and out["planted_decade_helix"]["p_gain"] <= 0.05
        and out["planted_decade_helix"]["rings"]["units"]["p_closed"] <= 0.05
        and out["planted_century_circle"]["rings"]["century20"]["p_closed"] <= 0.05)
    out["passed"] = bool(core)
    out["certified_closure_statistics"] = list(certified_closure)
    out["certified_closure_note"] = (
        "This tuple is the GATE set -- the closure statistics required not to fire on the "
        "trended negative families -- and both members pass that test. It is NOT the "
        "evidence tier. rings.century.p_closed (k=100) passes here yet is NON-SPECIFIC at "
        "real amplitudes: see amplitude_matched_power and claim_classification."
        "evidence_tier.non_specific. The primary confound-free closure statistic is "
        "rings.century20.p_closed alone.")
    out["disqualified_as_evidence"] = dict(
        statistic="rings.units.p_closed (and by inheritance rings.tens.p_closed, which has "
                  "no matched null available at all -- its true groups are contiguous "
                  "decade blocks and every composition-preserving permutation also "
                  "preserves the tens structure)",
        measured=dict(curved_monotone_trips_of_5=cal["curved_monotone"]["n_sig"]["p_closed_units"],
                      linear_ramp_trips_of_5=cal["linear_ramp"]["n_sig"]["p_closed_units"],
                      noise_trips_of_5=cal["noise"]["n_sig"]["p_closed_units"],
                      nominal_expected_of_5=0.25),
        reason="anti-conservative on curved monotone trajectories: detrending removes only "
               "the LINEAR part of a trend, and the residual curvature spreads the null "
               "groups' means, inflating the null gap_ratio. An inflated null makes "
               "everything look closed.",
        consequence="a units-ring positive on real activations is NOT evidence of a decade "
                    "ring. It was already copy-explicable; it is now also uncalibrated.")
    out["ordering_positive_control"] = dict(
        trips_of_5={k: cal[k]["n_sig"]["p_order_linear"]
                    for k in ("linear_ramp", "curved_monotone", "noise")},
        note="ordering is calibrated on pure NOISE only, deliberately. A linear ramp IS "
             "ordered, so 5/5 on the trended families is the statistic working. This is the "
             "positive control for ordering and a restatement of the project-wide finding "
             "that ordering alone is never evidence of a ring.")
    # RING ORDER STATISTIC -- DISQUALIFIED AS EVIDENCE (added 2026-08-20, item 3).
    # rings.*.p_order is the spec's rho_mod10 deliverable and it is a fit-selection
    # artifact. The cos/sin plane is fitted WITH the true labels, so the group means are
    # laid out in period order by construction; the permutation null shuffles group labels
    # over a FIXED angle set and therefore cannot see that circularity. Measured below on
    # families with no ring in them at all. Reported, never evidence.
    out["ring_order_calibration"] = dict(
        trips_of_5={kind: {k: cal[kind]["n_sig"][k] for k in RING_ORDER_KEYS}
                    for kind in ("linear_ramp", "curved_monotone", "noise")},
        planted_and_negative_cases={
            nm: {k: out[nm]["rings"][k]["p_order"] for k in RINGS}
            for nm in ("planted_decade_helix", "planted_century_circle",
                       "linear_ramp", "curved_monotone")},
        nominal_expected_of_5=0.25,
        verdict="rings.*.p_order is DISQUALIFIED as evidence. See claim_classification."
                " Its real-data trip rate must always be reported next to its rate on the "
                "noun control and under shuffled labels.")
    out["ring_nulls"] = RING_NULL
    # the flag that licenses the confound-free claim, evaluated on the PRIMARY statistic
    out["separates_curved_monotone"] = bool(cal["curved_monotone"]["n_sig"]["p_closed_century20"] <= 1)
    out["separates_curved_monotone_amplitude_caveat"] = (
        "TRUE here means only that the curved-monotone family does not trip century20 at "
        "ITS OWN amplitude (T=100 gain ~0.021), against a planted circle at 0.6523. The "
        "real layers sit at 0.0380-0.3016, which neither side brackets, and this family "
        "SATURATES near 0.021 so it cannot be amplitude-matched. The separation that "
        "actually licenses the century claims is measured at the observed amplitude, on "
        "the real background, in the top-level `amplitude_matched_power` block.")
    out["curved_monotone_note"] = (
        "If False, every century-ring closure claim in this file is uninterpretable: the "
        "null cannot tell a closed century circle from a curved monotone trajectory, which "
        "is the realistic false positive at this k.")
    out["specificity_note"] = (
        "Recorded, not gated: planted_decade_helix century20 p_closed and "
        "planted_century_circle units p_closed say whether the two rings cross-talk.")
    return out


# ---------------------------------------------------------------------------
# post-verification diagnostics (added 2026-08-20; see the dated note in the docstring)
#
# Everything below runs AFTER the real rows and changes nothing above it. None of it gates
# the run. All of it was written with the observed numbers in hand, so every block records
# `provenance="post hoc"` -- these are diagnostics that make the negative interpretable,
# not pre-registered tests.
# ---------------------------------------------------------------------------
def ring_closure(X, values, name, n_null=200, seed=0, with_order=False, order_perm=200):
    """gap_ratio + p_closed for ONE ring, through the same code path `years_row` uses.

    Same centering, same detrend-on-true-values, same ring_spec/ring_points/_circ, same
    null family (within-decade blocks for units, free permutation otherwise). The
    permutation STREAM differs from `years_row`'s interleaved one, so p-values agree with
    the canonical rows only up to Monte-Carlo error; the canonical rows stay canonical.
    """
    X = np.asarray(X, dtype=np.float64)
    X = X - X.mean(0, keepdims=True)
    v = np.asarray(values, dtype=int)
    Xr = detrend(X, v.astype(np.float64))
    bas, T, grp, k = ring_spec(name, v)
    xy = ring_points(Xr, bas, T, grp, k)
    cx, cy, r, resid, c = _circ(xy)
    rng = np.random.default_rng(seed)
    nl = np.empty(n_null)
    for i in range(n_null):
        vv = _block_perm(v, rng) if name == "units" else rng.permutation(v)
        b2, T2, g2, k2 = ring_spec(name, vv)
        _, _, _, _, cn = _circ(ring_points(Xr, b2, T2, g2, k2))
        nl[i] = cn["gap_ratio"]
    out = dict(k=k, gap_ratio=c["gap_ratio"], radius_ratio=c["radius_ratio"],
               max_gap_deg=c["max_gap_deg"], ring_flag=is_ring(c),
               p_closed=float((1 + (nl <= c["gap_ratio"]).sum()) / (1 + n_null)),
               null_gap_mean=float(nl.mean()), null_gap_p05=float(np.percentile(nl, 5)),
               n_null=n_null, null_kind=RING_NULL[name])
    if with_order:
        s = cyclic_order_score(angles(xy, cx, cy), np.arange(k), k, n_perm=order_perm,
                               seed=seed)
        out.update(rho=s["rho"], p_order=s["p"], n_perm_order=order_perm)
    return out


def _strip_period(X, values, T=100.0):
    """Project the cos/sin pair at period T out of X, keeping constant + linear + the rest.

    The background a planted curve is added to: it has (numerically) no T=100 component of
    its own, so the planted amplitude alone sets the T=100 gain.
    """
    v = np.asarray(values, dtype=np.float64)
    C = np.column_stack([np.ones_like(v), v,
                         np.cos(2 * np.pi * v / T), np.sin(2 * np.pi * v / T)])
    coef, *_ = np.linalg.lstsq(C, X, rcond=None)
    return X - C[:, 2:] @ coef[2:]


def _unit(rng, d):
    e = rng.normal(size=d)
    return e / np.linalg.norm(e)


# The planted families. `full_circle` is the thing the arm is looking for; the other three
# are OPEN curves that load on the same T=100 cos/sin pair. The question each answers:
# would a real century circle at this amplitude have been detected (power), and does an
# open bend at the same amplitude get called closed (specificity)?
PLANT_FAMILIES = ("full_circle", "three_quarter_arc", "half_circle_arc", "curved_monotone")


def _plant(name, values, d, scale, rng):
    u = np.asarray(values, dtype=np.float64) / 99.0

    def col(z):
        z = np.asarray(z, dtype=np.float64)[:, None]
        return z - z.mean()

    e = [_unit(rng, d) for _ in range(4)]
    if name == "full_circle":
        return scale * (col(np.cos(2 * np.pi * u)) * e[0] + col(np.sin(2 * np.pi * u)) * e[1])
    if name == "three_quarter_arc":
        return scale * (col(np.cos(1.5 * np.pi * u)) * e[0] + col(np.sin(1.5 * np.pi * u)) * e[1])
    if name == "half_circle_arc":
        return scale * (col(np.cos(np.pi * u)) * e[0] + col(np.sin(np.pi * u)) * e[1])
    if name == "curved_monotone":
        return scale * (col(u ** 1.6) * e[2] + 0.5 * col(u) * e[3])
    raise ValueError(name)


def _tune_amplitude(bg, S, values, target, lo=1e-4, hi=1e3, iters=45):
    """Bisect (in log amplitude) until the T=100 gain of bg + a*S equals `target`.

    Returns the saturation amplitude with `reached=False` when the family cannot reach the
    target at any amplitude -- which is itself a finding: the power-law curved-monotone
    family saturates near gain 0.021 and so can never have been the realistic false
    positive for a 0.19-gain observation.
    """
    v = np.asarray(values, dtype=np.float64)
    for _ in range(iters):
        mid = np.sqrt(lo * hi)
        if single_period_gain(bg + mid * S, v, 100.0) < target:
            lo = mid
        else:
            hi = mid
    a = float(np.sqrt(lo * hi))
    Y = bg + a * S
    got = single_period_gain(Y, v, 100.0)
    return Y, a, got, bool(abs(got - target) < 0.02 * max(target, 1e-12))


def amplitude_matched_power(X, values, observed_T100_gain, seeds=(11, 22, 33),
                            n_null=200, seed_base=0):
    """Plant closed and open curves into THIS layer's own background at ITS observed gain.

    The century claims in this file are negative ones, and a negative is only a finding if
    the test had power where the data actually lives. The battery certified separation at a
    planted gain of 0.65 against a curved-monotone 0.022; the real layers are at 0.04-0.30.
    Here the amplitude is bisected until the planted T=100 gain EQUALS the layer's observed
    gain, so power and specificity are measured in the observed regime.
    """
    X = np.asarray(X, dtype=np.float64)
    X = X - X.mean(0, keepdims=True)
    v = np.asarray(values, dtype=int)
    d = X.shape[1]
    scale = float(np.linalg.norm(X, axis=1).mean())
    bg = _strip_period(X, v.astype(np.float64), 100.0)
    rows = []
    for fi, fam in enumerate(PLANT_FAMILIES):
        for s in seeds:
            rng = np.random.default_rng(seed_base + 97 * fi + s)
            S = _plant(fam, v, d, scale, rng)
            Y, amp, got, reached = _tune_amplitude(bg, S, v, observed_T100_gain)
            c20 = ring_closure(Y, v, "century20", n_null=n_null, seed=seed_base + 5000 + 13 * fi + s)
            c100 = ring_closure(Y, v, "century", n_null=n_null, seed=seed_base + 6000 + 13 * fi + s)
            rows.append(dict(family=fam, seed=s, amp=amp, T100_gain=got,
                             reached_target=reached,
                             c20_gap_ratio=c20["gap_ratio"], c20_radius_ratio=c20["radius_ratio"],
                             c20_p_closed=c20["p_closed"],
                             c100_gap_ratio=c100["gap_ratio"], c100_p_closed=c100["p_closed"]))
    bg20 = ring_closure(bg, v, "century20", n_null=n_null, seed=seed_base + 7000)
    bg100 = ring_closure(bg, v, "century", n_null=n_null, seed=seed_base + 7100)

    def rate(fam, key):
        sel = [r for r in rows if r["family"] == fam]
        return dict(n_sig=int(sum(r[key] <= 0.05 for r in sel)), n=len(sel),
                    reached_target=int(sum(r["reached_target"] for r in sel)),
                    T100_gain_reached=[r["T100_gain"] for r in sel])

    return dict(provenance="post hoc -- amplitudes matched to the OBSERVED T=100 gain and "
                           "layers chosen after seeing the results; diagnostic, not a gate",
                observed_T100_gain=float(observed_T100_gain),
                background_T100_gain=float(single_period_gain(bg, v.astype(np.float64), 100.0)),
                scale=scale, n_null=n_null, seeds=list(seeds), planted=rows,
                background_only=dict(century20=bg20, century=bg100),
                detection_c20={f: rate(f, "c20_p_closed") for f in PLANT_FAMILIES},
                detection_c100={f: rate(f, "c100_p_closed") for f in PLANT_FAMILIES})


def shuffle_calibration(X, values, n_shuffle=20, n_null=200, seed=0):
    """Is the closure null calibrated on THIS geometry? Independent label shuffles.

    `rows_shuffled_labels` uses ONE fixed permutation at all 26 layers. Late layers are
    highly correlated, so one unlucky draw shows up as a run of trips and reads like an
    anti-conservative null. Fresh draws at a fixed layer separate the two explanations.
    """
    X = np.asarray(X, dtype=np.float64)
    v = np.asarray(values, dtype=int)
    out = {}
    for name in RINGS:
        want_order = name in ("units", "century20")
        ps, po = [], []
        for i in range(n_shuffle):
            perm = np.random.default_rng(seed + 1009 * i).permutation(len(v))
            r = ring_closure(X[perm], v, name, n_null=n_null, seed=seed + 7 * i + 1,
                             with_order=want_order)
            ps.append(r["p_closed"])
            if want_order:
                po.append(r["p_order"])
        row = dict(n_shuffle=n_shuffle, p_closed=ps,
                   n_sig_closed=int(sum(p <= 0.05 for p in ps)),
                   median_p_closed=float(np.median(ps)))
        if want_order:
            row.update(p_order=po, n_sig_order=int(sum(p <= 0.05 for p in po)),
                       median_p_order=float(np.median(po)))
        out[name] = row
    return out


def build_summary(rows, rows_shuf, nouns, power, shufcal):
    """Machine-generated headline facts, so the writeup cannot mis-transcribe them.

    Written because the first analyst report of this arm said the T=10 gain was "dead from
    L7 on" when layer 10 scores p = 0.0249, and reported the T=100 gain as "0.11-0.22" when
    the range over significant layers is 0.0380-0.3016 with the maximum at L5.
    """
    def sig(rs, f):
        return [r["layer"] for r in rs if f(r) <= 0.05]

    def pT(T):
        return lambda r: r["p_gain_by_period"][str(T)]

    def gT(T):
        return lambda r: r["gain_by_period"][str(T)]

    def arg(rs, f, best=max):
        r = best(rs, key=f)
        return dict(layer=r["layer"], value=float(f(r)))

    L = [r["layer"] for r in rows]
    s = {}
    for T in PERIODS:
        s[f"gain_T{T}"] = dict(
            layers_significant=sig(rows, pT(T)),
            layers_not_significant=[l for l in L if l not in sig(rows, pT(T))],
            n_significant=len(sig(rows, pT(T))),
            p_by_layer={str(r["layer"]): pT(T)(r) for r in rows},
            gain_by_layer={str(r["layer"]): gT(T)(r) for r in rows},
            gain_min=arg(rows, gT(T), min), gain_max=arg(rows, gT(T), max),
            monotone_in_layer=bool(all(gT(T)(rows[i]) <= gT(T)(rows[i + 1])
                                       for i in range(len(rows) - 1))
                                   or all(gT(T)(rows[i]) >= gT(T)(rows[i + 1])
                                          for i in range(len(rows) - 1))))
    s["closure_real"] = {
        nm: dict(n_sig=len([r for r in rows if r["rings"][nm]["p_closed"] <= 0.05]),
                 n=len(rows),
                 min_p=arg(rows, lambda r, nm=nm: r["rings"][nm]["p_closed"], min),
                 gap_ratio_min=arg(rows, lambda r, nm=nm: r["rings"][nm]["gap_ratio"], min),
                 gap_ratio_max=arg(rows, lambda r, nm=nm: r["rings"][nm]["gap_ratio"], max),
                 n_ring_flag=len([r for r in rows if r["rings"][nm]["ring_flag"]]))
        for nm in RINGS}
    s["ring_order_real"] = {
        nm: dict(n_sig=len([r for r in rows if r["rings"][nm]["p_order"] <= 0.05]),
                 n=len(rows),
                 layers_significant=[r["layer"] for r in rows
                                     if r["rings"][nm]["p_order"] <= 0.05],
                 rho_range=[float(min(r["rings"][nm]["rho"] for r in rows)),
                            float(max(r["rings"][nm]["rho"] for r in rows))],
                 status="DISQUALIFIED as evidence (fit-selection artifact) -- see "
                        "synthetics.ring_order_calibration")
        for nm in RINGS}
    s["ordering_real"] = dict(
        abs_rho_min=arg(rows, lambda r: r["ordering"]["abs_rho"], min),
        abs_rho_max=arg(rows, lambda r: r["ordering"]["abs_rho"], max),
        n_significant=len([r for r in rows if r["ordering"]["p_order_shuffled"] <= 0.05]),
        layers_significant=[r["layer"] for r in rows
                            if r["ordering"]["p_order_shuffled"] <= 0.05],
        abs_rho_by_layer={str(r["layer"]): r["ordering"]["abs_rho"] for r in rows},
        note="reflection-invariant: rho_pc1_svd_plus and rho_pc1_svd_minus are both in the "
             "rows and the p-value is two-sided on |rho|")
    s["sinusoidality_units_range"] = [
        float(min(r["sinusoidality_units"] for r in rows)),
        float(max(r["sinusoidality_units"] for r in rows))]

    s["shuffled_labels_control"] = dict(
        note="ONE fixed permutation (seed 12345) reused at all 26 layers -- correlated "
             "across layers, so a run of trips is one draw, not 26. Independent draws are "
             "in shuffle_calibration.",
        n_layers=len(rows_shuf),
        gains={f"T{T}": len([r for r in rows_shuf if pT(T)(r) <= 0.05]) for T in PERIODS},
        p_gain_overall=len([r for r in rows_shuf if r["p_gain"] <= 0.05]),
        min_p_gain=float(min(r["p_gain"] for r in rows_shuf)),
        closure={nm: dict(n_sig=len([r for r in rows_shuf
                                     if r["rings"][nm]["p_closed"] <= 0.05]),
                          min_p=float(min(r["rings"][nm]["p_closed"] for r in rows_shuf)))
                 for nm in RINGS},
        ring_order={nm: len([r for r in rows_shuf if r["rings"][nm]["p_order"] <= 0.05])
                    for nm in RINGS},
        ordering=dict(n_sig=len([r for r in rows_shuf
                                 if r["ordering"]["p_order_shuffled"] <= 0.05]),
                      max_abs_rho=float(max(r["ordering"]["abs_rho"] for r in rows_shuf)),
                      min_p=float(min(r["ordering"]["p_order_shuffled"] for r in rows_shuf))))
    if shufcal:
        s["shuffle_calibration_summary"] = {
            str(l): {nm: dict(n_sig_closed=v[nm]["n_sig_closed"],
                              n=v[nm]["n_shuffle"],
                              median_p_closed=v[nm]["median_p_closed"],
                              **({"n_sig_order": v[nm]["n_sig_order"]}
                                 if "n_sig_order" in v[nm] else {}))
                      for nm in v}
            for l, v in shufcal.items()}
    if nouns:
        s["noun_control"] = {
            str(r["layer"]): dict(p_gain=r["p_gain"],
                                  p_gain_T10=r["p_gain_by_period"]["10"],
                                  p_gain_T100=r["p_gain_by_period"]["100"],
                                  p_closed_century20=r["rings"]["century20"]["p_closed"],
                                  p_closed_century=r["rings"]["century"]["p_closed"],
                                  p_closed_units=r["rings"]["units"]["p_closed"],
                                  p_order_ring_units=r["rings"]["units"]["p_order"],
                                  rho_ring_units=r["rings"]["units"]["rho"],
                                  p_order_linear=r["ordering"]["p_order_shuffled"],
                                  abs_rho=r["ordering"]["abs_rho"])
            for r in nouns["rows"]}
        s["noun_control"]["_n_sig_ring_order_units"] = len(
            [r for r in nouns["rows"] if r["rings"]["units"]["p_order"] <= 0.05])
        s["noun_control"]["_n_rows"] = len(nouns["rows"])
    if power:
        s["amplitude_matched_power"] = {
            str(l): dict(observed_T100_gain=p["observed_T100_gain"],
                         detection_c20={f: f"{p['detection_c20'][f]['n_sig']}/"
                                           f"{p['detection_c20'][f]['n']}"
                                        for f in PLANT_FAMILIES},
                         detection_c100={f: f"{p['detection_c100'][f]['n_sig']}/"
                                            f"{p['detection_c100'][f]['n']}"
                                         for f in PLANT_FAMILIES},
                         reached_target={f: f"{p['detection_c20'][f]['reached_target']}/"
                                            f"{p['detection_c20'][f]['n']}"
                                         for f in PLANT_FAMILIES},
                         background_only_c20_p=p["background_only"]["century20"]["p_closed"])
            for l, p in power.items()}
    return s


# ---------------------------------------------------------------------------
# forwards
# ---------------------------------------------------------------------------
def collect(model, tok, layers, labels, templates):
    """One batched forward per label, all layers captured, per-TEMPLATE means retained.

    Per-template activations are kept (100 x 16 x 1152 x 26 float32 = 192 MB) so the
    immediate/distal template split can be scored without a second pass over the model.
    """
    texts = [t for t, _ in templates]
    per = {l: [] for l in layers}
    with torch.inference_mode():
        for w in labels:
            enc = tok([t.format(w) for t in texts], return_tensors="pt",
                      padding=True).to(DEVICE)
            with capture(model, layers) as store:
                model(**enc, use_cache=False)
            last = enc["attention_mask"].sum(1) - 1
            idx = torch.arange(last.shape[0], device=DEVICE)
            for l in layers:
                per[l].append(store[l][idx, last].float().cpu().numpy())
    return {l: np.stack(per[l]).astype(np.float32) for l in layers}   # [n_labels, n_tmpl, d]


def tokenization_audit(tok, labels, templates, kind):
    """Recorded, not assumed: how every label splits, and what token is actually read.

    The whole point of this arm is the capture position, so the readout token is decoded
    for all labels x all templates and asserted constant. If that assert ever fires, the
    fix did not take and no number below means anything.
    """
    splits = {}
    for w in labels:
        ids = tok.encode(str(w), add_special_tokens=False)
        splits[str(w)] = dict(n_tokens=len(ids),
                              pieces=[tok.decode([i]) for i in ids])
    n_tok_counts = sorted({v["n_tokens"] for v in splits.values()})

    readout, suffix_len, bad = {}, {}, []
    for ti, (t, grp) in enumerate(templates):
        toks_here = set()
        for w in labels:
            ids = tok.encode(t.format(w), add_special_tokens=True)
            toks_here.add(tok.decode([ids[-1]]))
            if w == labels[0]:
                pre = tok.encode(t.split("{}")[0] + str(w), add_special_tokens=True)
                suffix_len[t] = len(ids) - len(pre)
        readout[t] = dict(group=grp, readout_tokens=sorted(toks_here),
                          constant=(len(toks_here) == 1),
                          n_suffix_tokens=suffix_len.get(t))
        if len(toks_here) != 1:
            bad.append(t)
    return dict(
        kind=kind, n_labels=len(labels),
        label_token_counts=n_tok_counts,
        all_labels_same_token_count=(len(n_tok_counts) == 1),
        splits=splits, per_template=readout,
        readout_token_constant_everywhere=(len(bad) == 0),
        templates_with_varying_readout=bad,
        position="last non-pad token = the fixed suffix AFTER the label; never a digit "
                 "of the year",
        value="v = year - 1900 (0..99). 1900 = 0 mod 100, so cos/sin(2*pi*year/T) equals "
              "cos/sin(2*pi*v/T) exactly for T in (2,5,10,100); the linear column differs "
              "by a constant absorbed by the intercept.",
        hook_note="capture() hooks model.layers[i] OUTPUT (post-block), so even layer 0 "
                  "has had one attention pass: a year digit can be copied into the readout "
                  "position at every layer. The capture position removes the TOKEN-IDENTITY "
                  "confound of script 15, not the one-hop copy channel. See "
                  "claim_classification.")


CLAIM_CLASSIFICATION = dict(
    evidence_tier=dict(
        certified=["rings.century20.p_closed (PRIMARY closure statistic; power measured at "
                   "the observed amplitude in amplitude_matched_power)",
                   "p_gain and p_gain_by_period (all four T)"],
        non_specific=["rings.century.p_closed (k=100)"],
        disqualified=["rings.units.p_closed", "rings.tens.p_closed",
                      "rings.*.p_order (all four rings)"],
        why="see synthetics.disqualified_as_evidence (units/tens closure nulls are "
            "anti-conservative on curved monotone trajectories) and "
            "synthetics.ring_order_calibration (the ring cyclic-order statistic is a "
            "fit-selection artifact). Reported, never evidence."),
    corrections_2026_08_20=dict(
        moved=["rings.century.p_closed (k=100): certified -> non_specific",
               "rings.*.p_order: unlabelled -> disqualified"],
        k100_evidence="an OPEN three-quarter arc planted into the real background at the "
                      "layer's OBSERVED T=100 gain trips rings.century.p_closed while "
                      "rings.century20.p_closed correctly does not (amplitude_matched_"
                      "power.detection_c100 vs detection_c20). k=100 is not merely "
                      "low-power, it cannot tell a circle from an arc at real amplitudes. "
                      "This never bit the verdict -- k=100 never fired on real data -- and "
                      "it retroactively validates the pre-declared k=20-primary rule.",
        p_order_evidence="The cos/sin plane is fitted WITH the true labels, so the group "
                         "means come out in period order by construction and a null that "
                         "permutes labels over a FIXED angle set cannot correct for it. "
                         "Trips in five draws per negative family (nominal 0.25; see "
                         "synthetics.ring_order_calibration): units 5/5 ramp, 5/5 curved, "
                         "5/5 PURE NOISE; century (k=100) 5/5, 5/5, 5/5; century20 3/5, "
                         "3/5, 2/5; tens 3/5, 3/5, 1/5. units and century also sit at the "
                         "permutation floor on both planted cases. Real-data trip rates -- "
                         "units 22/26 layers (rho 0.93-0.99) and century 24/26, including "
                         "every late layer where the T=10 gain is dead -- must always be "
                         "reported next to the controls: the units statistic fires on 3/3 "
                         "noun-control layers, on 12/26 shuffled-label layers and on 23/80 "
                         "independent shuffles at real geometry (28.8% vs nominal 5%). The "
                         "tens and century20 order statistics are quieter (1/26 and 3/26 "
                         "under the fixed shuffle) but are disqualified on the same "
                         "mechanism, not exonerated by coarser binning.",
        also="the shuffled-labels closure trips (century20 5/26, tens 11/26, century k=100 "
             "12/26 under ONE fixed permutation) are now reported in summary."
             "shuffled_labels_control next to the independent-draw resolution in "
             "shuffle_calibration, instead of being omitted from the controls sentence."),
    confound_free=dict(
        items=["gain_by_period['100']", "rings.century20 (PRIMARY)", "rings.century (k=100, "
               "low power -- see synthetics.century_k_choice)",
               "ordering (Spearman vs chronology)"],
        why="position within the century is a joint function of the tens and units digit "
            "tokens; no single token in the prompt carries it, so attention copying one "
            "year token into the readout position cannot produce it."),
    copy_explicable=dict(
        items=["gain_by_period['10']", "rings.units", "gain_tens_ring", "rings.tens"],
        why="v mod 10 is exactly the identity of the units-digit token and v // 10 exactly "
            "the tens-digit token. Both are one attention hop from the readout position, so "
            "a positive here is reported but is NOT evidence of a computed helix."),
    secondary_discriminator=dict(
        items=["sinusoidality_units", "sinusoidality_tens"],
        why="ratio of the single-sinusoid gain to the full 9-dummy one-hot gain for the same "
            "digit. A copied token identity has no reason to be a single frequency; a helix "
            "does. Near 1 favours helix, near 0 favours lookup."),
    ordering_caveat="Established earlier in this project: ordering alone is never evidence. "
                    "Random subsets are accepted at ~0.16 and an ordinal control out-orders "
                    "the real ontology at every layer. Closure is the discriminating tier.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", default=",".join(str(i) for i in range(26)))
    ap.add_argument("--n-null", type=int, default=200)
    ap.add_argument("--n-null-split", type=int, default=100)
    ap.add_argument("--smoke", action="store_true",
                    help="2 labels, no battery: check shapes and the readout token, then exit")
    ap.add_argument("--skip-nouns", action="store_true")
    # post-verification diagnostics (added 2026-08-20)
    ap.add_argument("--skip-diagnostics", action="store_true",
                    help="skip the amplitude-matched power leg and the shuffle calibration")
    ap.add_argument("--n-shuffle-cal", type=int, default=20,
                    help="independent label shuffles per diagnostic layer")
    ap.add_argument("--out", default="results/years_existence.json")
    a = ap.parse_args()

    layers = [int(x) for x in a.layers.split(",")]
    years = list(range(1900, 2000))
    v = np.arange(100)

    token = os.environ.get("HF_TOKEN")          # cache is warm; HF_HUB_OFFLINE=1 works
    model, tok = load_model(token=token)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"

    audit = tokenization_audit(tok, years, YEAR_TEMPLATES, "years")
    print(f"D0 years: token counts {audit['label_token_counts']}, "
          f"readout constant everywhere = {audit['readout_token_constant_everywhere']}",
          flush=True)
    for t, r in audit["per_template"].items():
        print(f"   [{r['group']:9s}] suffix={r['n_suffix_tokens']} "
              f"readout={r['readout_tokens']}  {t}", flush=True)
    if not audit["readout_token_constant_everywhere"]:
        raise SystemExit("readout token varies with the year -- the capture-position fix "
                         "did not take; refusing to run")

    if a.smoke:
        t0 = time.time()
        M = collect(model, tok, layers, years[:2], YEAR_TEMPLATES)
        print(f"smoke: per-layer array {M[layers[0]].shape} dtype {M[layers[0]].dtype}, "
              f"{len(M)} layers, {time.time()-t0:.1f}s for 2 years", flush=True)
        for l in (layers[0], layers[len(layers)//2], layers[-1]):
            d = np.linalg.norm(M[l][0].mean(0) - M[l][1].mean(0))
            print(f"   layer {l}: ||mean(1900) - mean(1901)|| = {d:.4f} "
                  f"(non-zero => the readout position carries year information)", flush=True)
        return

    print("\nsynthetic battery ...", flush=True)
    t0 = time.time()
    syn = synthetics(n_null=100)
    print("  century-k choice (p_closed; planted circle must trip, the other three must not):",
          flush=True)
    for nm, row in syn["century_k_choice"]["grid"].items():
        print(f"    {nm:23s} " + "  ".join(
            f"{k} gapR {row[k]['gap_ratio']:6.2f} p {row[k]['p_closed']:.4f}"
            for k in ("k100", "k20", "k10")), flush=True)
    for nm in ("planted_decade_helix", "planted_century_circle", "linear_ramp",
               "curved_monotone"):
        r = syn[nm]
        print(f"  {nm:23s} gain {r['gain']:.4f} p {r['p_gain']:.4f} | "
              f"units gapR {r['rings']['units']['gap_ratio']:.2f} p {r['rings']['units']['p_closed']:.4f} | "
              f"cent20 gapR {r['rings']['century20']['gap_ratio']:.2f} "
              f"p {r['rings']['century20']['p_closed']:.4f} | "
              f"cent100 gapR {r['rings']['century']['gap_ratio']:.2f} "
              f"p {r['rings']['century']['p_closed']:.4f}", flush=True)
    for k, v_ in syn["calibration"].items():
        print(f"  calibration {k:16s} trips/5: {v_['n_sig']}", flush=True)
    print(f"  battery {'PASS' if syn['passed'] else 'FAIL'}; separates curved monotone from "
          f"century circle = {syn['separates_curved_monotone']} ({time.time()-t0:.0f}s)\n",
          flush=True)
    if not syn["passed"]:
        raise SystemExit("battery failed; not running on activations")

    # ---------------- years ----------------
    t0 = time.time()
    M = collect(model, tok, layers, years, YEAR_TEMPLATES)
    print(f"years forwards: {len(years)} x {len(YEAR_TEMPLATES)} prompts, "
          f"{time.time()-t0:.0f}s\n", flush=True)

    groups = np.array([g for _, g in YEAR_TEMPLATES])
    imm = np.where(groups == "immediate")[0]
    dis = np.where(groups == "distal")[0]

    rows, rows_shuf, rows_split = [], [], []
    rng_shuf = np.random.default_rng(12345)
    shuffle_perm = rng_shuf.permutation(100)     # ONE fixed shuffle, applied at every layer
    hdr = (f"{'L':>3} {'r2lin':>7} {'gain':>7} {'p':>6} | {'T2':>7} {'T5':>7} {'T10':>7} "
           f"{'T100':>7} | {'pT10':>6} {'pT100':>6} | {'uGapR':>6} {'pU':>6} | "
           f"{'c20GapR':>7} {'pC20':>6} | {'|rho|':>6} {'pOrd':>6} {'sinU':>5}")
    print(hdr, flush=True)
    for l in layers:
        A = M[l].mean(1).astype(np.float64)                       # [100, d]
        r = years_row(A, v, n_null=a.n_null, seed=l)
        rows.append(dict(layer=l, **r))
        rs = years_row(A[shuffle_perm], v, n_null=a.n_null, seed=1000 + l)
        rows_shuf.append(dict(layer=l, **rs))
        split = {}
        for nm, cols in (("immediate", imm), ("distal", dis)):
            B = M[l][:, cols].mean(1).astype(np.float64)
            sp = years_row(B, v, n_null=a.n_null_split, seed=2000 + l)
            split[nm] = dict(gain=sp["gain"], p_gain=sp["p_gain"],
                             gain_by_period=sp["gain_by_period"],
                             p_gain_by_period=sp["p_gain_by_period"],
                             r2_linear=sp["r2_linear"],
                             units=sp["rings"]["units"],
                             century20=sp["rings"]["century20"],
                             century=sp["rings"]["century"],
                             sinusoidality_units=sp["sinusoidality_units"],
                             ordering=sp["ordering"], n_null=sp["n_null"],
                             n_templates=len(cols))
        rows_split.append(dict(layer=l, **split))
        bp, pb = r["gain_by_period"], r["p_gain_by_period"]
        su = r["sinusoidality_units"]
        print(f"{l:3d} {r['r2_linear']:7.4f} {r['gain']:7.4f} {r['p_gain']:6.4f} | "
              f"{bp['2']:7.4f} {bp['5']:7.4f} {bp['10']:7.4f} {bp['100']:7.4f} | "
              f"{pb['10']:6.4f} {pb['100']:6.4f} | "
              f"{r['rings']['units']['gap_ratio']:6.2f} {r['rings']['units']['p_closed']:6.4f} | "
              f"{r['rings']['century20']['gap_ratio']:7.2f} {r['rings']['century20']['p_closed']:6.4f} | "
              f"{r['ordering']['abs_rho']:6.3f} {r['ordering']['p_order_shuffled']:6.4f} "
              f"{(su if su is not None else float('nan')):5.2f}", flush=True)

    # ---------------- nouns ----------------
    nouns = None
    if not a.skip_nouns:
        naudit = tokenization_audit(tok, NOUNS, NOUN_TEMPLATES, "nouns")
        print(f"\nnoun control: {len(NOUNS)} alphabetically-ordered nouns, "
              f"readout constant = {naudit['readout_token_constant_everywhere']}, "
              f"pre-declared layers {NOUN_LAYERS}", flush=True)
        t0 = time.time()
        MN = collect(model, tok, list(NOUN_LAYERS), NOUNS, NOUN_TEMPLATES)
        print(f"noun forwards: {time.time()-t0:.0f}s", flush=True)
        nrows = []
        for l in NOUN_LAYERS:
            A = MN[l].mean(1).astype(np.float64)
            r = years_row(A, v, n_null=a.n_null, seed=3000 + l)
            nrows.append(dict(layer=l, **r))
            bp, pb = r["gain_by_period"], r["p_gain_by_period"]
            print(f"{l:3d} {r['r2_linear']:7.4f} {r['gain']:7.4f} {r['p_gain']:6.4f} | "
                  f"{bp['2']:7.4f} {bp['5']:7.4f} {bp['10']:7.4f} {bp['100']:7.4f} | "
                  f"{pb['10']:6.4f} {pb['100']:6.4f} | "
                  f"{r['rings']['units']['gap_ratio']:6.2f} {r['rings']['units']['p_closed']:6.4f} | "
                  f"{r['rings']['century20']['gap_ratio']:7.2f} {r['rings']['century20']['p_closed']:6.4f} | "
                  f"{r['ordering']['abs_rho']:6.3f} {r['ordering']['p_order_shuffled']:6.4f}",
                  flush=True)
        nouns = dict(audit=naudit, labels=NOUNS, layers=list(NOUN_LAYERS),
                     value_order="alphabetical index 0..99, declared before results",
                     rows=nrows)

    # ------------- post-verification diagnostics (added 2026-08-20) -------------
    # Runs last, touches nothing above it, gates nothing. POST HOC by construction: the
    # planted amplitudes are the observed gains and the layers are chosen from the rows.
    power, shufcal, diag_layers = {}, {}, []
    if not a.skip_diagnostics:
        by_layer = {r["layer"]: r for r in rows}
        # Layer-selection rule, applied to the data rather than hardcoded:
        #   (i)  argmax observed T=100 gain      -- where a century circle is loudest
        #   (ii) argmin observed century20 p     -- where closure came closest to firing
        #   (iii) layers 19 and 25               -- the two the adversarial verification
        #                                           used, included so its numbers can be
        #                                           reproduced through this code path
        pick = [max(rows, key=lambda r: r["gain_by_period"]["100"])["layer"],
                min(rows, key=lambda r: r["rings"]["century20"]["p_closed"])["layer"]]
        pick += [l for l in (19, 25) if l in by_layer]
        diag_layers = sorted(set(pick))
        print(f"\npost-verification diagnostics at layers {diag_layers} "
              f"(argmax T100 gain, argmin century20 p, plus 19 and 25)", flush=True)
        keep = ("gap_ratio", "radius_ratio", "max_gap_deg", "ring_flag", "p_closed",
                "null_gap_mean", "null_gap_p05")
        for l in diag_layers:
            t0 = time.time()
            A = M[l].mean(1).astype(np.float64)
            r = by_layer[l]
            obs = r["gain_by_period"]["100"]
            p = amplitude_matched_power(A, v, obs, n_null=a.n_null, seed_base=100 * l)
            p["observed_century20"] = {k: r["rings"]["century20"][k] for k in keep}
            p["observed_century"] = {k: r["rings"]["century"][k] for k in keep}
            power[str(l)] = p
            print(f"  L{l:2d} observed T100 gain {obs:.4f}, century20 gapR "
                  f"{r['rings']['century20']['gap_ratio']:.2f} p "
                  f"{r['rings']['century20']['p_closed']:.4f}  -- planted at that gain:",
                  flush=True)
            for fam in PLANT_FAMILIES:
                d20, d100 = p["detection_c20"][fam], p["detection_c100"][fam]
                gains = ", ".join(f"{g:.4f}" for g in d20["T100_gain_reached"])
                print(f"      {fam:18s} reached {d20['reached_target']}/{d20['n']} "
                      f"(gains {gains}) | century20 detects {d20['n_sig']}/{d20['n']} | "
                      f"century k=100 detects {d100['n_sig']}/{d100['n']}", flush=True)
            print(f"      background only     century20 p "
                  f"{p['background_only']['century20']['p_closed']:.4f} | century k=100 p "
                  f"{p['background_only']['century']['p_closed']:.4f}", flush=True)
            sc = shuffle_calibration(A, v, n_shuffle=a.n_shuffle_cal, n_null=a.n_null,
                                     seed=900 * l + 7)
            shufcal[str(l)] = sc
            print("      " + str(a.n_shuffle_cal) + " independent label shuffles: " +
                  " | ".join(f"{nm} {sc[nm]['n_sig_closed']}/{sc[nm]['n_shuffle']} "
                             f"(median p {sc[nm]['median_p_closed']:.3f})" for nm in RINGS)
                  + f"  [{time.time()-t0:.0f}s]", flush=True)

    summary = build_summary(rows, rows_shuf, nouns, power, shufcal)
    print("\nSUMMARY (machine-generated, quote these rather than transcribing):", flush=True)
    for T in PERIODS:
        k = f"gain_T{T}"
        print(f"  gain T={T:3d} p<=0.05 at layers {summary[k]['layers_significant']} "
              f"| gain min {summary[k]['gain_min']['value']:.4f} (L{summary[k]['gain_min']['layer']}) "
              f"max {summary[k]['gain_max']['value']:.4f} (L{summary[k]['gain_max']['layer']})",
              flush=True)
    for nm in RINGS:
        c = summary["closure_real"][nm]
        o = summary["ring_order_real"][nm]
        print(f"  ring {nm:9s} closure {c['n_sig']}/{c['n']} (min p {c['min_p']['value']:.4f} "
              f"at L{c['min_p']['layer']}) | ORDER (disqualified) {o['n_sig']}/{o['n']}",
              flush=True)
    od = summary["ordering_real"]
    print(f"  ordering |rho| {od['abs_rho_min']['value']:.3f}-{od['abs_rho_max']['value']:.3f}, "
          f"{od['n_significant']}/{len(rows)} layers p<=0.05", flush=True)
    print(f"  shuffled labels: closure trips "
          f"{ {nm: summary['shuffled_labels_control']['closure'][nm]['n_sig'] for nm in RINGS} }"
          f", ring-order trips {summary['shuffled_labels_control']['ring_order']}, "
          f"ordering {summary['shuffled_labels_control']['ordering']['n_sig']}", flush=True)

    out = dict(
        target="years 1900-1999 of the 20th century",
        claim_classification=CLAIM_CLASSIFICATION,
        tokenization=audit,
        templates=[dict(text=t, group=g) for t, g in YEAR_TEMPLATES],
        noun_templates=[dict(text=t, group=g) for t, g in NOUN_TEMPLATES],
        periods=list(PERIODS), rings=list(RINGS),
        synthetics=syn,
        rows=rows,
        rows_shuffled_labels=dict(
            note="the SAME 100 layer-mean vectors with the year labels permuted once "
                 "(seed 12345), rerun through identical code at every layer",
            perm=shuffle_perm.tolist(), rows=rows_shuf),
        rows_template_split=rows_split,
        nouns=nouns,
        summary=summary,
        amplitude_matched_power=dict(
            provenance="POST HOC, added 2026-08-20 after adversarial verification. "
                       "Amplitudes are bisected to each layer's OBSERVED T=100 gain and "
                       "the layers are chosen from the rows, so this is a diagnostic that "
                       "makes the negative interpretable -- not a pre-registered gate.",
            layers=diag_layers,
            layer_rule="argmax observed T=100 gain; argmin observed century20 p_closed; "
                       "plus layers 19 and 25, the two the adversarial verification used",
            families=list(PLANT_FAMILIES),
            question="would a century circle at THIS amplitude have been detected (power), "
                     "and does an OPEN curve at the same amplitude get called closed "
                     "(specificity)?",
            by_layer=power),
        shuffle_calibration=dict(
            provenance="POST HOC, added 2026-08-20. Independent label shuffles at the same "
                       "layers, to classify the shuffled-labels closure trips in "
                       "rows_shuffled_labels (one fixed permutation across 26 correlated "
                       "layers) as either an anti-conservative null or one draw.",
            n_shuffle=a.n_shuffle_cal, layers=diag_layers, by_layer=shufcal),
        compare_to="results/helix_existence.json (the confounded final-digit run)")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=2))
    print(f"\nwrote {a.out}", flush=True)


if __name__ == "__main__":
    main()
