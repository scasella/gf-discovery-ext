"""Does a HUE CIRCLE exist in the residual stream? (A2b -- supervised existence sweep)

Why colours rather than another calendar. The calendar positives in this project all
carry the same caveat: month and weekday order is a human convention that is recited
verbatim in text, so an embedding can inherit the cycle from co-occurrence statistics
without the model ever computing anything. Hue order is weaker in that respect. The
rainbow recitation (red-orange-yellow-green-blue-indigo-violet) is a memorised
sequence, but *cyan*, *magenta* and *pink* are not in it -- they come from print and
display vocabularies -- so where those three land relative to the recited six is a
placement the recitation does not supply. Colours are also in Bhalla et al.'s test set
and Goodfire's post names "colours on a hue/saturation/lightness surface", so a
positive here is directly comparable to published claims.

WHAT IS DECLARED BEFORE ANY RESULT IS SEEN (all of it is written into the output JSON
under "declarations", and printed to the log before the first activation is read):

  D1  Hue set. The primary label set is the single-token survivors of
        red(0) orange(30) yellow(60) green(120) cyan(180) blue(240)
        purple(280) magenta(300) pink(330)
      in canonical HSV hue order, cyclic. Any candidate whose leading-space form is
      not a single token is dropped and the drop recorded. Verdicts are computed on
      this set and no other. lime/teal/violet/indigo/turquoise/grey/silver are
      captured in the same forwards and are EXPLORATORY only.

  D2  Achromatic set. black, white, gray -- held OUT of the hue circle. They have no
      hue, so including them in a hue ring would be a category error. They are used
      only for the lightness (third-direction) check.

  D3  Readout positions. Every template is PREFIX + " " + WORD + SUFFIX with a fixed
      prefix and suffix, so the colour token sits at a known index and the suffix
      tokens are IDENTICAL across words within a template.
        "token"  -- the colour token itself. This is the EMBEDDING-CONFOUNDED readout:
                    at early layers the point cloud is (almost) the colour word's own
                    embedding, so a ring here is lexical geometry, not computation.
        "suffix" -- PRIMARY COMPUTED READOUT: the first token AFTER the colour. The
                    word is no longer the token being read; whatever hue information is
                    present had to be moved there by attention.
        "final"  -- the last token of the template. Tertiary, exploratory.
      This distinction is declared now because the project's largest dropped fix was
      exactly it: the base-10 helix readout was taken at the number's final digit,
      where value mod 10 IS the token identity, and the reading was confounded.

  D4  Closure is the discriminator. Ordering is NEVER evidence on its own: the
      cyclic-ordering statistic scores rho = 1.000 for a monotone 1-D arrangement, and
      in this project an ordinal control (number words) out-orders months at every
      layer while never closing. A layer is called a hue circle only if the closure
      tier passes against BOTH nulls in D5.

  D5  Nulls.
      (a) uniform random angles, k of them, 20,000 draws -- the same statistic
          `results/closure_depth.json` used, so the numbers are comparable.
      (b) matched random word sets of the SAME ARITY through the IDENTICAL pipeline:
          300 draws of k single-token common nouns in the same templates at the same
          positions. This is the null that matters. Reported as a false-acceptance
          rate at alpha = 0.05 and as an empirical p (fraction of draws whose gap
          ratio is at least as small as the observed one).
      (c) secondary, grammaticality-matched: 300 draws of k non-colour ADJECTIVES,
          because a colour sits in an adjective slot and a noun does not.
      (d) ordering: 20,000 shuffled-label permutations under the same
          max-over-symmetries selection, plus an explicit 1,000-shuffle control block.
      Background draws are NOT scored for ordering: random nouns have no true cyclic
      order, so a background ordering rate would be meaningless. Closure is the
      background criterion.

  D6  Noun/adjective pool exclusion rule, declared before the pools are used: any word
      with an established colour sense or a canonical colour name in English
      paint/fashion vocabulary is excluded (rose, cream, chocolate, rust, ivory, wine,
      salmon, peach, lime, sand, charcoal, plum, cherry, mustard, olive, coral, jade,
      ash, bronze, copper, brass, slate, mint, lemon, snow, ...), as is any lightness
      or saturation adjective (light, dark, bright, pale, deep, vivid, dull). Calendar
      words are excluded so the pools cannot inherit this project's other ontology.

  D7  Both SVD sign conventions are reported. `cyclic_order_score` maximises over the
      2k dihedral symmetries and is therefore reflection-invariant, so rho must be
      IDENTICAL under a PC2 sign flip; that identity is asserted, not assumed. The
      handedness that a sign flip does change is exposed by also reporting the
      rotation-only statistic (no reflection allowed) under each convention.

  D8  Expected lexical pattern, stated in advance so it cannot be reported as a
      finding: the token readout SHOULD close at layers 0-2 if colour embeddings carry
      hue geometry. That is the confound, not the result. The claim this script can
      support is closure at the SUFFIX readout at mid layers, where the colour word is
      not the token being read.

  D9  Verdict per (readout, layer), four outcomes:
        hue_circle             closure passes (a) AND (b) AND ordering p < 0.05
        closed_ring_unordered  closure passes (a) AND (b), ordering does not
        ordered_arc            ordering passes, closure does not
        nothing                neither
        degenerate             the point cloud carries no usable geometry

  D10 SECONDARY, WIDENED PLANE SEARCH -- provenance stated plainly. This one was NOT
      in the first draft. It was added after a three-layer dry run (layers 0, 9, 20)
      showed the primary top-2 PCA plane negative at every readout, and before any
      full-sweep result was seen. It is included because the primary plane is the
      top-2 PCA plane, and hue need not be the dominant variance direction of a colour
      point cloud -- a lightness or word-frequency axis can own PC1 and bury a hue ring
      in PC2-PC3.
        rule: all C(4,2) = 6 planes spanned by pairs of the top 4 PCs; the plane
              minimising the angular gap ratio is selected. Selection is LABEL-FREE,
              so the shuffled-label ordering null on the winning plane stays exact.
        closure: the uniform-angle p-value on a SELECTED plane is invalid and is
              reported marked as such. The valid test is the matched background with
              the IDENTICAL min-over-6-planes selection applied to every draw.
        verdicts: D9 verdicts are computed on the PRIMARY statistic only. The secondary
              gets its own field, so any positive is visibly a product of the widened
              search.
      THIS IS THE LAST EXPANSION. If the secondary is also negative, the finding is
      "no hue circle detectable by this method family", and no third search, no
      supervised probe and no alternative centring is run under A2b.

  D11 MULTIPLICITY. The sweep has 3 readouts x 26 layers = 78 cells. At alpha = 0.05
      about 4 ordering passes and about 4 closure passes are expected by chance, so an
      isolated pass is not a finding. Benjamini-Hochberg across all 78 cells is
      reported alongside the raw counts, and the expected-by-chance count is printed.

CAVEAT THAT SURVIVES A POSITIVE, recorded now. A closed hue ring at the suffix readout
shows the information is present one token later; it does not by itself show the model
re-computed it, because attention can copy an embedding forward. The only handle on
that here is the principal angle between the suffix-readout hue plane and the layer-0
token-readout hue plane: a suffix plane that is a copy of the embedding plane sits at a
small angle to it. That angle is reported per layer. It is a handle, not a proof.
"""
import argparse, json, os, sys, time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from gfdisc.env import DEVICE, D_MODEL
from gfdisc.model import load_model, capture
from gfdisc.fits import (degenerate, pca_2d, fit_circle, circularity, is_ring, angles,
                         cyclic_order_score, _rank_of)
from gfdisc.nulls import benjamini_hochberg

# ---------------------------------------------------------------------------
# D1 / D2 -- label sets
# ---------------------------------------------------------------------------
HUE_CANDIDATES = [("red", 0), ("orange", 30), ("yellow", 60), ("green", 120),
                  ("cyan", 180), ("blue", 240), ("purple", 280), ("magenta", 300),
                  ("pink", 330)]
ACHROMATIC = [("black", 0.0), ("gray", 0.5), ("white", 1.0)]      # (word, lightness)
EXPLORATORY_EXTRA = ["lime", "teal", "violet", "indigo", "turquoise", "grey", "silver"]

# Six of the primary nine are the recited rainbow; three are not. Used only to report
# whether the ring survives on the half the recitation does not supply.
RECITED = {"red", "orange", "yellow", "green", "blue", "purple"}

# ---------------------------------------------------------------------------
# D3 -- templates. (prefix, suffix); the word goes between them with one space.
# No template contains another colour word, and every suffix is >= 3 tokens so that
# the "suffix" and "final" readouts are distinct positions.
# ---------------------------------------------------------------------------
TEMPLATES = [
    ("The", " car was parked outside"),
    ("She painted the wall", " and then went home"),
    ("He wore a", " shirt to the office"),
    ("The tin of paint was labelled", " by the supplier"),
    ("They chose", " for the new logo"),
    ("The sign is", " and easy to read"),
    ("A single", " balloon floated away"),
    ("The artist mixed some", " into the palette"),
    ("Her new bicycle is", " and very fast"),
    ("The label says", " in small print"),
    ("We repainted the door", " in a hurry"),
    ("The fabric came in", " and cost very little"),
    ("He pointed at the", " one on the shelf"),
    ("The logo uses", " as its main tone"),
    ("That old sofa was", " when we bought it"),
    ("The screen glowed", " for a few seconds"),
]
READOUTS = ("token", "suffix", "final")

# ---------------------------------------------------------------------------
# D6 -- background pools. The exclusion list is written into the JSON verbatim.
# ---------------------------------------------------------------------------
COLOUR_SENSE_EXCLUSIONS = [
    "rose", "cream", "chocolate", "rust", "ivory", "wine", "salmon", "peach", "lime",
    "sand", "charcoal", "plum", "cherry", "mustard", "olive", "coral", "jade", "ash",
    "bronze", "copper", "brass", "slate", "mint", "lemon", "snow", "steel", "amber",
    "ruby", "emerald", "sapphire", "navy", "gold", "silver", "tan", "beige", "lavender",
    "light", "dark", "bright", "pale", "deep", "vivid", "dull", "shade", "tint",
]
NOUN_POOL = [
    "table", "chair", "window", "door", "garden", "river", "mountain", "bridge",
    "letter", "market", "village", "engine", "ticket", "bottle", "camera", "pencil",
    "farmer", "doctor", "teacher", "student", "kitchen", "office", "school", "hospital",
    "library", "station", "airport", "factory", "machine", "computer", "keyboard",
    "screen", "paper", "folder", "basket", "blanket", "pillow", "mirror", "carpet",
    "curtain", "ladder", "hammer", "rope", "string", "needle", "button", "pocket",
    "jacket", "shirt", "hat", "glove", "scarf", "belt", "wallet", "purse", "suitcase",
    "umbrella", "clock", "watch", "radio", "telephone", "printer", "monitor", "cable",
    "battery", "lamp", "candle", "kettle", "plate", "spoon", "fork", "knife", "glass",
    "bowl", "tray", "oven", "sink", "towel", "soap", "brush", "comb", "razor",
    "horse", "sheep", "rabbit", "mouse", "eagle", "spider", "beetle", "forest",
    "meadow", "valley", "canyon", "desert", "island", "harbour", "beach", "cliff",
    "cave", "stream", "pond", "lake", "ocean", "storm", "cloud", "thunder", "breeze",
    "frost", "hammock", "tunnel", "fence", "gate", "roof", "floor", "ceiling", "stair",
    "corner", "shelf", "drawer", "cupboard", "bucket", "shovel", "wheel", "handle",
    "pipe", "wire", "nail", "screw", "board", "plank", "brick", "stone", "gravel",
]
ADJ_POOL = [
    "large", "small", "heavy", "quick", "slow", "loud", "quiet", "warm", "cold",
    "soft", "hard", "rough", "smooth", "sharp", "blunt", "clean", "dirty", "fresh",
    "stale", "empty", "full", "narrow", "wide", "tall", "short", "thick", "thin",
    "strong", "weak", "cheap", "modern", "ancient", "simple", "complex", "useful",
    "useless", "happy", "angry", "calm", "nervous", "polite", "rude", "honest",
    "clever", "stupid", "brave", "lazy", "busy", "tired", "hungry", "famous",
    "strange", "normal", "secret", "public", "private", "local", "foreign", "silent",
    "clumsy", "gentle", "fierce", "humble", "solid", "hollow", "curved", "straight",
]

N_BACKGROUND = 2000
N_PERM_ORDER = 20000
N_SHUFFLE_CONTROL = 1000
N_GAP_NULL = 20000

# ---------------------------------------------------------------------------
# D12 -- RECOVERY AMENDMENTS. An earlier launch of this script was interrupted before
# it wrote anything (results/colors_existence.json never existed). Everything below was
# changed on the recovery pass, BEFORE any activation was read, and is listed here so a
# reader can see exactly what was touched between the first draft and the run that
# produced the output. Nothing in D1-D11 -- label set, readouts, verdict rule,
# discriminator -- was altered.
# ---------------------------------------------------------------------------
RECOVERY_AMENDMENTS = [
    "IMPORT FIX. `benjamini_hochberg` was used in the D11 multiplicity block but never "
    "imported. The first draft would have raised NameError at the last step, after every "
    "forward pass and every null was already spent. Imported from gfdisc.nulls, the same "
    "source scripts/13_rerun_gate.py and scripts/21_window_gate.py use.",

    "N_BACKGROUND 300 -> 2000. Not a power grab: the matched-word background p-value is "
    "an add-one empirical fraction, so n draws floor it at 1/(n+1). At 300 the floor is "
    "3.3e-3. BH-FDR over 78 cells puts the rank-1 threshold at 0.05/78 = 6.4e-4, so no "
    "SINGLE cell could ever survive BH on the background tier however clean it was "
    "(a group of ~6 cells sitting together at the floor still could, at the rank-6 "
    "threshold of 3.8e-3 -- the floor forecloses isolated survivors, not all of them). "
    "At 2000 the floor is 5.0e-4, below the rank-1 threshold, so the BH column can "
    "report what it is meant to report. Measured cost of the change: about 2 minutes.",

    "TOKENISATION VERIFICATION WIDENED. The first draft verified manual-vs-natural "
    "tokenisation on the hue and achromatic words plus the first five nouns and first "
    "five adjectives. It now runs over EVERY word in every template. Hue/achromatic "
    "mismatches abort the run (they are load-bearing labels); background-pool mismatches "
    "drop that word from the pool and are recorded. The drop rule is mechanical and "
    "label-blind, so it cannot select a background.",

    "ORANGE/PURPLE RESOLUTION RECORDED. The unsupervised arm "
    "(results/token_type_geometry.json) reported 'orange' and 'purple' missing from its "
    "label pool. That was a CACHE-OCCURRENCE threshold (>= 100 rows in cache/sparse_big), "
    "not a tokenizer fact. This arm generates its own prompts, so the threshold does not "
    "apply; both words are single-token and are used at full weight. Written into "
    "token_audit.orange_purple_resolution.",

    "P-VALUE FLOORS REPORTED. Every empirical p-value in this script is floored by its "
    "own null size. The floors are now written into the multiplicity block so a "
    "'0 survive BH' line can be read against what was reachable.",

    "COSMETIC. `pc_var_11pt` renamed `pc_var_all_points` (it is k + 3 = 12 points when "
    "all nine hue words survive, not 11), and the layer whose token-readout plane serves "
    "as the embedding reference is recorded by number instead of being called 'L0'.",
]

# ---------------------------------------------------------------------------
# Closure null: one shared table per k, so 300 draws x 26 layers x 3 readouts costs
# a searchsorted rather than a 20,000-row resample each.
# ---------------------------------------------------------------------------
_TABLES = {}


def gap_table(k, n_perm=N_GAP_NULL, seed=0):
    if k not in _TABLES:
        rng = np.random.default_rng(seed)
        a = np.sort(rng.uniform(0, 2 * np.pi, size=(n_perm, k)), axis=1)
        g = np.diff(np.concatenate([a, a[:, :1] + 2 * np.pi], axis=1), axis=1).max(1)
        _TABLES[k] = np.sort(g)
    return _TABLES[k]


def p_closed(k, observed_max_gap_rad):
    """Identical in construction to scripts/12_closure_depth.py:gap_pvalue."""
    t = gap_table(k)
    return float((1 + int(np.searchsorted(t, observed_max_gap_rad, side="right"))) / (1 + len(t)))


# ---------------------------------------------------------------------------
# Ordering, vectorised, with the flip group exposed (D7).
# ---------------------------------------------------------------------------
def order_stats(ang, label_index, K, flips=(1, -1), n_perm=N_PERM_ORDER, seed=0, block=5000):
    """Max Spearman over the allowed symmetries, calibrated by shuffled labels.

    Same construction as fits.cyclic_order_score (stable-argsort tie-break replaced by
    a random jitter, add-one p-value), vectorised over the permutation null and with
    `flips` exposed so the rotation-only (handedness-sensitive) statistic can be read
    off separately. With flips=(1,-1) it must reproduce cyclic_order_score exactly;
    that is asserted in `check_against_reference`.
    """
    ang = np.asarray(ang, dtype=float)
    lab = np.asarray(label_index, dtype=int)
    n = len(lab)
    jitter = np.random.default_rng(seed).permutation(n)
    order = np.lexsort((jitter, ang))
    rank_of = np.empty(n, dtype=np.float64)
    rank_of[order] = np.arange(n)

    syms = [(int(f), int(s)) for f in flips for s in range(K)]
    base = np.vstack([_rank_of((f * lab + s) % K) for f, s in syms])       # [S, n]
    denom = n * (n * n - 1)

    def rhos(sig):
        d = rank_of - base[:, sig]
        return 1.0 - 6.0 * (d * d).sum(-1) / denom

    obs = rhos(np.arange(n))
    j = int(obs.argmax())
    rho = float(obs[j])

    rng = np.random.default_rng(seed + 1)
    null = np.empty(n_perm)
    for lo in range(0, n_perm, block):
        m = min(block, n_perm - lo)
        sig = np.argsort(rng.random((m, n)), axis=1)
        null[lo:lo + m] = rhos(sig).max(0)
    return dict(rho=rho, symmetry=list(syms[j]),
                p=float((1 + int((null >= rho).sum())) / (1 + n_perm)),
                null_mean=float(null.mean()), null_p95=float(np.percentile(null, 95)),
                null_max=float(null.max()), n_shuffles=int(n_perm),
                flips=list(int(f) for f in flips))


def check_against_reference(seed=0):
    """order_stats(flips=(1,-1)) must equal fits.cyclic_order_score on the same input."""
    rng = np.random.default_rng(seed)
    out = []
    for t in range(5):
        k = 9
        ang = rng.uniform(-np.pi, np.pi, size=k)
        lab = np.arange(k)
        a = cyclic_order_score(ang, lab, k, n_perm=2000, seed=t)
        b = order_stats(ang, lab, k, n_perm=2000, seed=t)
        out.append(dict(trial=t, rho_ref=a["rho"], rho_new=b["rho"],
                        sym_ref=list(a["symmetry"]), sym_new=b["symmetry"],
                        rho_match=bool(abs(a["rho"] - b["rho"]) < 1e-12)))
    # p-values use different RNG streams, so only rho and the winning symmetry are
    # required to match; that is what the vectorisation could have broken.
    return dict(trials=out, passed=bool(all(x["rho_match"] for x in out)))


# ---------------------------------------------------------------------------
# The one analysis routine. Everything -- colours, backgrounds, synthetics -- goes
# through it, so nothing can be scored by a path the controls did not take.
# ---------------------------------------------------------------------------
def analyse(P, label_index=None, K=None, seed=0, do_order=True, both_conventions=True):
    P = np.asarray(P, dtype=np.float64)
    k = len(P)
    deg = bool(degenerate(P))
    xy, ratios, Vt = pca_2d(P)
    out = dict(k=k, degenerate=deg,
               top2_var=float(ratios[:2].sum()),
               pc_var=[float(x) for x in ratios[:min(5, len(ratios))]])
    convs = {}
    for name, s in (("A", 1.0), ("B", -1.0)):
        if not both_conventions and name == "B":
            continue
        z = xy.copy()
        z[:, 1] *= s
        cx, cy, r, resid = fit_circle(z)
        c = circularity(z, cx, cy, r)
        d = dict(resid=float(resid), radius_ratio=c["radius_ratio"],
                 gap_ratio=c["gap_ratio"], max_gap_deg=c["max_gap_deg"],
                 spread=c["spread"], radius=c["radius"],
                 p_closed_uniform=p_closed(k, np.radians(c["max_gap_deg"])),
                 is_ring=bool(is_ring(c)))
        if do_order and not deg and label_index is not None:
            ang = angles(z, cx, cy)
            KK = int(K if K is not None else k)
            d["order"] = order_stats(ang, label_index, KK, flips=(1, -1),
                                     n_perm=N_PERM_ORDER, seed=seed)
            d["order_rotation_only"] = order_stats(ang, label_index, KK, flips=(1,),
                                                   n_perm=N_SHUFFLE_CONTROL, seed=seed)
            d["shuffled_label_control"] = order_stats(ang, label_index, KK, flips=(1, -1),
                                                      n_perm=N_SHUFFLE_CONTROL,
                                                      seed=seed + 777)
        convs[name] = d
    out["conventions"] = convs
    return out, Vt[:2], xy


# ---------------------------------------------------------------------------
# D10 -- widened, label-free plane search. One SVD, six circle fits.
# ---------------------------------------------------------------------------
N_PC_SECONDARY = 4


def plane_scan(P, n_pc=N_PC_SECONDARY):
    """Gap ratio in every plane spanned by a pair of the top `n_pc` PCs.

    Returns (pc_var_ratios, list of per-plane dicts). The first entry is always the
    (PC1, PC2) plane, so the primary statistic falls out of the same SVD and the
    background loop pays for one decomposition rather than two.
    """
    X = np.asarray(P, dtype=np.float64)
    X = X - X.mean(0, keepdims=True)
    _, S, Vt = np.linalg.svd(X, full_matrices=False)
    var = S ** 2
    ratios = var / max(var.sum(), 1e-12)
    npc = min(n_pc, Vt.shape[0])
    planes = []
    for i in range(npc):
        for j in range(i + 1, npc):
            z = X @ Vt[[i, j]].T
            cx, cy, r, resid = fit_circle(z)
            c = circularity(z, cx, cy, r)
            planes.append(dict(pc=(i, j), gap_ratio=c["gap_ratio"],
                               radius_ratio=c["radius_ratio"],
                               max_gap_deg=c["max_gap_deg"], resid=float(resid),
                               cx=cx, cy=cy, xy=z, basis=Vt[[i, j]]))
    return ratios, planes


def fast_closure(P, n_pc=N_PC_SECONDARY):
    """(primary gap ratio, primary p_uniform, best-plane gap ratio) for a background draw."""
    _, planes = plane_scan(P, n_pc)
    prim = planes[0]
    best = min(planes, key=lambda p: p["gap_ratio"])
    return (prim["gap_ratio"], p_closed(len(P), np.radians(prim["max_gap_deg"])),
            best["gap_ratio"], prim["radius_ratio"])


def secondary_analyse(P, label_index=None, K=None, seed=0, n_pc=N_PC_SECONDARY,
                      do_order=True):
    ratios, planes = plane_scan(P, n_pc)
    best = min(planes, key=lambda p: p["gap_ratio"])
    k = len(P)
    out = dict(n_pc_searched=int(min(n_pc, len(ratios))), n_planes=len(planes),
               selected_plane=[int(x) for x in best["pc"]],
               gap_ratio=best["gap_ratio"], radius_ratio=best["radius_ratio"],
               max_gap_deg=best["max_gap_deg"], resid=best["resid"],
               p_closed_uniform_SELECTION_INVALID=p_closed(
                   k, np.radians(best["max_gap_deg"])),
               all_planes=[dict(pc=[int(x) for x in p["pc"]], gap_ratio=p["gap_ratio"],
                                radius_ratio=p["radius_ratio"]) for p in planes])
    if do_order and label_index is not None and not degenerate(P):
        ang = angles(best["xy"], best["cx"], best["cy"])
        out["order"] = order_stats(ang, label_index, int(K if K is not None else k),
                                   flips=(1, -1), n_perm=N_PERM_ORDER, seed=seed)
        out["order_note"] = ("plane selection is label-free, so this shuffled-label "
                             "null is exact for the selected plane")
    return out, best["basis"]


# ---------------------------------------------------------------------------
# Synthetic battery (D4's dissociation, enforced before any activation is read).
# ---------------------------------------------------------------------------
def synthetics(d=D_MODEL, k=9, noise=0.15, n_bg=N_BACKGROUND, seed=0):
    """Planted-shape battery, run through `analyse` -- the same function the colours use.

    Both tiers are exercised, because both are load-bearing and they fail differently:

      uniform-angle null  -- measured here, not assumed: 9 isotropic Gaussian points in
          1152 dimensions are MORE evenly spread in their own top-2 plane than 9 uniform
          random angles are (mean gap ratio ~2.37 against ~2.83), so the uniform-angle
          test is mildly ANTI-conservative on real-dimensional data. Its false-acceptance
          rate against such clouds is measured and reported as
          `uniform_null_calibration`. This is exactly why the matched-word background is
          the tier that decides.

      matched background  -- 300 random k-point clouds of the same arity in the same
          ambient dimension. A planted ring must beat it; an arc, a line and pure noise
          must not.

    The dissociation the project exists to enforce is the arc and line rows: both order
    essentially perfectly (rho ~ 1) and neither closes. A battery that only planted a
    ring would not test the thing that has broken five statistics in this project.
    """
    rng = np.random.default_rng(seed)
    u, v = rng.normal(size=d), rng.normal(size=d)
    u /= np.linalg.norm(u)
    v -= v @ u * u
    v /= np.linalg.norm(v)
    lab = np.arange(k)

    def amb(scale):
        """Isotropic noise of per-point NORM `scale`, not per-coordinate scale.

        Recorded because getting it wrong silently destroyed the buried-ring row on the
        first attempt: `0.06 * randn(9, 1152)` looks small and is not -- each point
        picks up a vector of norm 0.06 * sqrt(1152) = 2.04, twice the radius of the
        unit ring it was meant to perturb. Scaling by sqrt(d) makes the parameter mean
        what it reads as.
        """
        return (scale / np.sqrt(d)) * rng.normal(size=(k, d))

    bg_rng = np.random.default_rng(seed + 12345)
    bg = np.empty(n_bg)
    bg_p_unif = np.empty(n_bg)
    bg_best = np.empty(n_bg)
    for i in range(n_bg):
        g, pu, gb, _ = fast_closure(bg_rng.normal(size=(k, d)))
        bg[i], bg_p_unif[i], bg_best[i] = g, pu, gb

    def row(P, seed_, labels=lab):
        r, _, _ = analyse(P, labels, k, seed=seed_, both_conventions=False)
        a = r["conventions"]["A"]
        s, _ = secondary_analyse(P, labels, k, seed=seed_, do_order=True)
        return dict(gap_ratio=a["gap_ratio"], radius_ratio=a["radius_ratio"],
                    p_closed=a["p_closed_uniform"],
                    p_vs_background=float((1 + int((bg <= a["gap_ratio"]).sum())) / (1 + n_bg)),
                    rho=a["order"]["rho"], p_order=a["order"]["p"], top2_var=r["top2_var"],
                    secondary=dict(
                        selected_plane=s["selected_plane"], gap_ratio=s["gap_ratio"],
                        rho=s["order"]["rho"], p_order=s["order"]["p"],
                        p_vs_background=float((1 + int((bg_best <= s["gap_ratio"]).sum()))
                                              / (1 + n_bg))))

    out = dict(noise=noise, k=k, d=d)
    th = 2 * np.pi * np.arange(k) / k
    ring = np.outer(np.cos(th), u) + np.outer(np.sin(th), v) + amb(noise)
    out["planted_ring"] = row(ring, 1)

    # closed but hue-unordered: the same ring with the labels scrambled. Closure must
    # still pass, ordering must fail -- the tiers have to be independent.
    perm = rng.permutation(k)
    out["planted_ring_scrambled_labels"] = row(ring, 2, labels=perm)

    ta = np.linspace(0, np.radians(200), k)
    arc = np.outer(np.cos(ta), u) + np.outer(np.sin(ta), v) + amb(noise)
    out["planted_arc_200deg"] = row(arc, 3)

    # The ordinal control has to be planted deliberately, not casually. A "line plus
    # isotropic 1152-dimensional noise" at the ring's noise level is not a line at all:
    # its top-2 plane is one line direction and one NOISE direction (top2_var ~0.34),
    # the angular order is scrambled and rho falls to ~0.68, so it would pass the
    # battery for the wrong reason. Planted instead as a thin sausage in the (u, v)
    # plane with small ambient noise, it reproduces the ordinal control this project
    # actually measured -- number words one..twelve, which score gap_ratio 5.1-7.4,
    # top2 0.75-0.95 and rho >= 0.99 at mid layers (results/closure_depth.json).
    line = (np.outer(np.linspace(-1, 1, k), u)
            + np.outer(0.05 * rng.normal(size=k), v)
            + amb(0.34))
    out["planted_line"] = row(line, 4)
    out["planted_line_note"] = ("thin sausage in the fitted plane + 0.01 ambient noise; "
                                "matches the measured number-words ordinal control "
                                "(gap_ratio 5.1-7.4, rho >= 0.99, never closes)")

    # D10 validation: a ring buried under a dominant linear ramp. The primary top-2
    # plane is grabbed by the ramp and MUST miss it; the widened search MUST find it.
    # Without this row a negative secondary would say nothing about the secondary.
    w = rng.normal(size=d)
    w -= (w @ u) * u + (w @ v) * v
    w /= np.linalg.norm(w)
    # The ramp's per-point COEFFICIENTS are orthogonalised against the ring's, not just
    # its direction against the ring's plane. This is not pedantry: with only k = 9
    # points, two components that are orthogonal as vectors in 1152 dimensions are
    # generally NOT orthogonal in the 9-dimensional sample space, because
    # sum_i ramp_i cos(2 pi i / 9) is not zero. A raw linear ramp therefore leaks into
    # the ring's principal directions (measured: |PC2 . w| = 0.35), PCA returns a mixed
    # plane and even the widened search only recovers gap_ratio 2.04 for a ring whose
    # own gap_ratio is 1.01. Worth recording as a property of the method, not just of
    # this fixture: at k = 9 any third factor correlated with hue ACROSS THE NINE WORDS
    # will rotate the recovered plane, and no plane search can undo that.
    ramp = np.linspace(-1, 1, k)
    B = np.column_stack([np.ones(k), np.cos(th), np.sin(th)])
    ramp = ramp - B @ np.linalg.lstsq(B, ramp, rcond=None)[0]
    ramp = ramp / max(ramp.std(), 1e-12)
    buried = (2.0 * np.outer(ramp, w)
              + np.outer(np.cos(th), u) + np.outer(np.sin(th), v)
              + amb(noise))
    out["buried_ring"] = row(buried, 5)
    out["buried_ring_note"] = ("unit ring in (u,v) under a 2x third component whose "
                               "per-point coefficients are orthogonal to the ring's; "
                               "the primary top-2 plane must miss it, D10 must find it")

    nz = [dict(seed=s, **row(rng.normal(size=(k, d)), s)) for s in range(10, 15)]
    out["pure_noise"] = dict(draws=nz,
                             n_sig_closed_uniform=int(sum(x["p_closed"] <= 0.05 for x in nz)),
                             n_sig_closed_background=int(sum(x["p_vs_background"] <= 0.05
                                                             for x in nz)),
                             n_sig_order=int(sum(x["p_order"] <= 0.05 for x in nz)),
                             n_sig_closed_background_secondary=int(
                                 sum(x["secondary"]["p_vs_background"] <= 0.05 for x in nz)))
    out["uniform_null_calibration"] = dict(
        n_draws=int(n_bg), pool="isotropic Gaussian, same k and d",
        gap_ratio_mean=float(bg.mean()), gap_ratio_p05=float(np.percentile(bg, 5)),
        gap_ratio_min=float(bg.min()),
        uniform_null_false_acceptance_at_p05=float((bg_p_unif <= 0.05).mean()),
        note="a false-acceptance rate above 0.05 here means the uniform-angle test is "
             "anti-conservative on 1152-dimensional clouds; the matched-word background "
             "is the tier that decides")

    R, RS, AR, LN, BR = (out["planted_ring"], out["planted_ring_scrambled_labels"],
                         out["planted_arc_200deg"], out["planted_line"],
                         out["buried_ring"])
    out["passed"] = bool(
        R["p_closed"] <= 0.05 and R["p_vs_background"] <= 0.05 and R["p_order"] <= 0.05
        and RS["p_vs_background"] <= 0.05 and RS["p_order"] > 0.05
        and AR["p_closed"] > 0.05 and AR["p_vs_background"] > 0.05 and AR["p_order"] <= 0.05
        # ordering must SUCCEED and closure must FAIL on both open shapes -- that is the
        # dissociation, and a rho threshold would be a hand-set number where the
        # calibrated ordering p-value already says it.
        and LN["p_closed"] > 0.05 and LN["p_vs_background"] > 0.05 and LN["p_order"] <= 0.05
        and out["pure_noise"]["n_sig_closed_background"] <= 1
        # D10 must have the sensitivity a negative would need to mean anything, and
        # must not manufacture rings out of open shapes or noise.
        and BR["p_vs_background"] > 0.05 and BR["secondary"]["p_vs_background"] <= 0.05
        and AR["secondary"]["p_vs_background"] > 0.05
        and LN["secondary"]["p_vs_background"] > 0.05
        and out["pure_noise"]["n_sig_closed_background_secondary"] <= 1)
    return out


# ---------------------------------------------------------------------------
# Activations
# ---------------------------------------------------------------------------
def token_audit(tok, words):
    """Which words survive as a single token with a leading space, and which do not."""
    keep, drop = [], []
    for w in words:
        ids = tok.encode(" " + w, add_special_tokens=False)
        (keep if len(ids) == 1 else drop).append(
            dict(word=w, n_tokens=len(ids), ids=[int(i) for i in ids]))
    return keep, drop


def build_and_verify(tok, prefix, suffix, word, word_id, bos):
    """Manual ids, then assert the natural tokenisation of the whole sentence agrees.

    Building input_ids by hand is what makes the readout indices exact, but it also
    makes it possible to feed the model a token sequence it would never produce. The
    assertion is the only thing standing between that and a silent position bug.
    """
    pre = tok.encode(prefix, add_special_tokens=False)
    suf = tok.encode(suffix, add_special_tokens=False)
    manual = [bos] + pre + [word_id] + suf
    natural = tok.encode(prefix + " " + word + suffix, add_special_tokens=True)
    return manual, natural, len(pre), len(suf)


def collect(model, tok, words, word_ids, layers, chunk=48):
    """One forward per (template, chunk of words); three positions kept per layer."""
    bos = tok.bos_token_id
    n = len(words)
    acc = {ro: {l: np.zeros((n, D_MODEL), dtype=np.float64) for l in layers}
           for ro in READOUTS}
    with torch.inference_mode():
        for prefix, suffix in TEMPLATES:
            pre = tok.encode(prefix, add_special_tokens=False)
            suf = tok.encode(suffix, add_special_tokens=False)
            ci = 1 + len(pre)                 # colour token index (BOS at 0)
            pos = dict(token=ci, suffix=ci + 1, final=ci + len(suf))
            base = [bos] + pre + [0] + suf
            for lo in range(0, n, chunk):
                hi = min(lo + chunk, n)
                ids = []
                for w in word_ids[lo:hi]:
                    row = list(base)
                    row[ci] = int(w)
                    ids.append(row)
                t = torch.tensor(ids, dtype=torch.long, device=DEVICE)
                with capture(model, layers) as st:
                    model(input_ids=t, use_cache=False)
                for l in layers:
                    h = st[l]
                    for ro in READOUTS:
                        acc[ro][l][lo:hi] += h[:, pos[ro]].float().cpu().numpy()
                del st
    for ro in READOUTS:
        for l in layers:
            acc[ro][l] /= len(TEMPLATES)
    return acc


# ---------------------------------------------------------------------------
# Lightness (the third-direction check) and plane comparison
# ---------------------------------------------------------------------------
def principal_angles_deg(A, B):
    """Principal angles between two 2-D subspaces given as [2, d] row bases."""
    Qa = np.linalg.qr(np.asarray(A, dtype=np.float64).T)[0]
    Qb = np.linalg.qr(np.asarray(B, dtype=np.float64).T)[0]
    s = np.clip(np.linalg.svd(Qa.T @ Qb, compute_uv=False), -1.0, 1.0)
    return [float(np.degrees(np.arccos(x))) for x in s]


def lightness_check(H, A, light, basis, noun_block, rng, n_null=N_BACKGROUND):
    """Do black/gray/white sit on an axis through the hue circle's centre?

    Three quantities, none of which is a new statistic:
      in_plane_ratio     -- how far the achromatic points sit from the hue centroid
                            IN the hue plane, relative to the hue circle's own radius.
                            Near 0 means they sit at the axis of the circle, which is
                            what "zero saturation" predicts.
      out_of_plane_frac  -- fraction of each achromatic point's displacement that
                            leaves the hue plane.
      monotone_lightness -- do black, gray, white come out in lightness order along
                            their own principal axis.
    The principal angle between the achromatic axis and the hue plane is reported for
    completeness, but near-90 degrees is the DEFAULT for any unrelated direction in
    1152 dimensions, so it is not evidence of anything on its own. The in-plane radius
    IS discriminating, so it gets the matched-noun null.
    """
    H = np.asarray(H, float)
    A = np.asarray(A, float)
    mu = H.mean(0)
    V = np.asarray(basis, float)                                   # [2, d]
    hue_xy = (H - mu) @ V.T
    r_hue = float(np.linalg.norm(hue_xy, axis=1).mean())

    d = A - mu
    a_xy = d @ V.T
    in_plane = np.linalg.norm(a_xy, axis=1)
    total = np.linalg.norm(d, axis=1)
    out_frac = np.sqrt(np.maximum(total ** 2 - in_plane ** 2, 0.0)) / np.maximum(total, 1e-12)

    Ac = A - A.mean(0)
    _, S, Wt = np.linalg.svd(Ac, full_matrices=False)
    axis = Wt[0]
    coord = Ac @ axis
    lo = np.asarray(light, float)
    mono = bool(np.all(np.diff(coord[np.argsort(lo)]) > 0) or
                np.all(np.diff(coord[np.argsort(lo)]) < 0))
    proj = np.linalg.norm(V @ axis)
    ang = float(np.degrees(np.arccos(np.clip(proj, 0, 1))))

    ratio = float(in_plane.mean() / max(r_hue, 1e-12))
    null = np.empty(n_null)
    for i in range(n_null):
        idx = rng.choice(len(noun_block), size=len(A), replace=False)
        dd = noun_block[idx] - mu
        null[i] = float(np.linalg.norm(dd @ V.T, axis=1).mean() / max(r_hue, 1e-12))

    # Joint (k hue + 3 achromatic) PCA: is there a THIRD direction, and do the
    # achromatics own it?
    allp = np.vstack([H, A])
    _, S3, Vt3 = np.linalg.svd(allp - allp.mean(0), full_matrices=False)
    var3 = (S3 ** 2) / max((S3 ** 2).sum(), 1e-12)
    pc3 = (allp - allp.mean(0)) @ Vt3[2]
    pc3_ach = pc3[len(H):]
    pc3_mono = bool(np.all(np.diff(pc3_ach[np.argsort(lo)]) > 0) or
                    np.all(np.diff(pc3_ach[np.argsort(lo)]) < 0))

    return dict(
        hue_radius=r_hue,
        achromatic_in_plane=[float(x) for x in in_plane],
        in_plane_ratio=ratio,
        in_plane_ratio_null_mean=float(null.mean()),
        in_plane_ratio_null_p05=float(np.percentile(null, 5)),
        p_in_plane_vs_nouns=float((1 + int((null <= ratio).sum())) / (1 + n_null)),
        out_of_plane_frac=[float(x) for x in out_frac],
        achromatic_axis_spread=float(np.linalg.norm(Ac, axis=1).mean()),
        achromatic_axis_over_hue_radius=float(np.linalg.norm(Ac, axis=1).mean() / max(r_hue, 1e-12)),
        achromatic_coord_on_own_axis=[float(x) for x in coord],
        monotone_lightness_on_own_axis=mono,
        principal_angle_axis_to_hue_plane_deg=ang,
        n_points_joint_pca=int(len(allp)),
        pc_var_all_points=[float(x) for x in var3[:5]],
        pc3_achromatic_coords=[float(x) for x in pc3_ach],
        pc3_hue_abs_mean=float(np.abs(pc3[:len(H)]).mean()),
        pc3_achromatic_abs_mean=float(np.abs(pc3_ach).mean()),
        monotone_lightness_on_pc3=pc3_mono,
    )


def jsonable(o):
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.bool_,)):
        return bool(o)
    raise TypeError(str(type(o)))


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/colors_existence.json")
    ap.add_argument("--layers", default=",".join(str(i) for i in range(26)))
    ap.add_argument("--n-background", type=int, default=N_BACKGROUND)
    ap.add_argument("--chunk", type=int, default=48)
    a = ap.parse_args()
    t0 = time.time()
    layers = [int(x) for x in a.layers.split(",")]

    print("=" * 78)
    print("A2b COLOURS -- supervised hue-circle existence sweep")
    print("DECLARATIONS (fixed before any activation is read):")
    print(f"  D1 hue candidates : {[w for w, _ in HUE_CANDIDATES]}")
    print(f"     hue angles     : {[h for _, h in HUE_CANDIDATES]}")
    print(f"  D2 achromatic     : {[w for w, _ in ACHROMATIC]} (held OUT of the ring)")
    print(f"  D3 readouts       : token (embedding-confounded) | "
          f"suffix = first post-colour token (PRIMARY, computed) | final (tertiary)")
    print(f"  D4 discriminator  : closure. Ordering alone is never evidence.")
    print(f"  D5 nulls          : uniform angles ({N_GAP_NULL}); "
          f"{a.n_background} matched noun draws of the same arity; "
          f"{a.n_background} adjective draws (secondary); "
          f"{N_PERM_ORDER} shuffled labels + {N_SHUFFLE_CONTROL}-shuffle control")
    print(f"  D8 expectation    : token readout closing at L0-2 IS the confound, "
          f"not the finding")
    print(f"  D9 verdicts       : hue_circle | closed_ring_unordered | ordered_arc | "
          f"nothing | degenerate")
    print(f"  D12 recovery pass : {len(RECOVERY_AMENDMENTS)} amendments, all made before "
          f"any activation was read:")
    for i, amend in enumerate(RECOVERY_AMENDMENTS, 1):
        print(f"     ({i}) {amend.split('.')[0]}.")
    print("=" * 78, flush=True)

    ref = check_against_reference()
    print(f"order_stats vs fits.cyclic_order_score: "
          f"{'MATCH' if ref['passed'] else 'MISMATCH'} on rho over 5 trials", flush=True)
    if not ref["passed"]:
        raise SystemExit("vectorised ordering statistic disagrees with the reference")

    syn = synthetics(n_bg=a.n_background)
    print("synthetic battery:")
    for key in ("planted_ring", "planted_ring_scrambled_labels", "planted_arc_200deg",
                "planted_line"):
        r = syn[key]
        print(f"  {key:32s} gapR {r['gap_ratio']:6.2f}  pUnif {r['p_closed']:.4f}  "
              f"pBg {r['p_vs_background']:.4f}  rho {r['rho']:+.3f}  "
              f"pOrd {r['p_order']:.4f}")
    uc = syn["uniform_null_calibration"]
    print(f"  uniform-angle null vs {uc['n_draws']} isotropic clouds (k={syn['k']}, "
          f"d={syn['d']}): mean gapR {uc['gap_ratio_mean']:.2f}, "
          f"false acceptance at p<0.05 = {uc['uniform_null_false_acceptance_at_p05']:.3f}")
    print(f"  pure_noise 5 draws: {syn['pure_noise']['n_sig_closed_background']}/5 "
          f"background-closure trips, {syn['pure_noise']['n_sig_order']}/5 ordering trips "
          f"-> {'PASS' if syn['passed'] else 'FAIL'}", flush=True)
    if not syn["passed"]:
        raise SystemExit("synthetic battery failed; not running on activations")

    model, tok = load_model(token=os.environ.get("HF_TOKEN"))
    bos = tok.bos_token_id

    # ---- D1/D2/D6 audits -------------------------------------------------
    hue_keep, hue_drop = token_audit(tok, [w for w, _ in HUE_CANDIDATES])
    hue_words = [x["word"] for x in hue_keep]
    hue_angles = [h for w, h in HUE_CANDIDATES if w in set(hue_words)]
    ach_keep, ach_drop = token_audit(tok, [w for w, _ in ACHROMATIC])
    ach_words = [x["word"] for x in ach_keep]
    ach_light = [l for w, l in ACHROMATIC if w in set(ach_words)]
    extra_keep, extra_drop = token_audit(tok, EXPLORATORY_EXTRA)
    excl = set(COLOUR_SENSE_EXCLUSIONS)
    noun_cand = [w for w in dict.fromkeys(NOUN_POOL) if w not in excl]
    adj_cand = [w for w in dict.fromkeys(ADJ_POOL) if w not in excl]
    noun_keep, noun_drop = token_audit(tok, noun_cand)
    adj_keep, adj_drop = token_audit(tok, adj_cand)
    noun_words = [x["word"] for x in noun_keep]
    adj_words = [x["word"] for x in adj_keep]
    K = len(hue_words)
    print(f"\nD1 audit: hue set k={K} {hue_words} (dropped {[x['word'] for x in hue_drop]})")
    print(f"D2 audit: achromatic {ach_words} (dropped {[x['word'] for x in ach_drop]})")
    print(f"D6 audit: {len(noun_words)} nouns, {len(adj_words)} adjectives kept "
          f"({len(noun_drop)} / {len(adj_drop)} dropped for multi-token)", flush=True)
    if K < 6:
        raise SystemExit(f"only {K} single-token hue words survive; too few to test")

    # The unsupervised arm's missing pair, settled explicitly (D12). Recorded whatever
    # the answer turns out to be, so this cannot become a silent substitution.
    op_res = dict(
        words=["orange", "purple"],
        unsupervised_arm_finding=("results/token_type_geometry.json placed 'orange' and "
                                  "'purple' outside its label pool"),
        reason_there=("that pool required >= 100 occurrences of a single-token variant in "
                      "cache/sparse_big; it is a corpus-frequency threshold, not a "
                      "tokenizer fact"),
        applies_here=False,
        why_not=("this arm generates its own prompts, so corpus frequency does not gate "
                 "membership; only single-token status does"),
        used_by_supervised_arm={
            w: dict(surface=" " + w,
                    token_ids=[int(i) for i in tok.encode(" " + w, add_special_tokens=False)],
                    single_token=bool(len(tok.encode(" " + w, add_special_tokens=False)) == 1),
                    in_hue_set_used=bool(w in set(hue_words)),
                    variant_mapped=False, downweighted=False)
            for w in ("orange", "purple")},
        note="no variant mapping and no reweighting was applied to either word")
    print(f"orange/purple: single-token "
          f"{[op_res['used_by_supervised_arm'][w]['token_ids'] for w in ('orange', 'purple')]}, "
          f"used at full weight (the token_type_geometry gap was a cache-occurrence "
          f"threshold, not a tokenizer one)", flush=True)

    # ---- template verification (manual ids == natural tokenisation) -------
    # D12: run over EVERY word, not a sample. Hue and achromatic mismatches abort -- they
    # are the labels. Background-pool and exploratory words that mismatch are DROPPED and
    # recorded: the rule is mechanical and label-blind, so it cannot select a background.
    extra_words = [x["word"] for x in extra_keep]
    label_words = hue_words + ach_words
    pool_words = noun_words + adj_words + extra_words

    def tokenises_everywhere(w):
        wid = tok.encode(" " + w, add_special_tokens=False)[0]
        for prefix, suffix in TEMPLATES:
            m, nat, _, _ = build_and_verify(tok, prefix, suffix, w, wid, bos)
            if m != nat:
                return False, dict(word=w, prefix=prefix, manual=m, natural=nat)
        return True, None

    label_bad = []
    for w in label_words:
        ok, info = tokenises_everywhere(w)
        if not ok:
            label_bad.append(info)
    pool_bad = []
    for w in pool_words:
        ok, info = tokenises_everywhere(w)
        if not ok:
            pool_bad.append(info)
    dropped_pool = {d["word"] for d in pool_bad}
    noun_words = [w for w in noun_words if w not in dropped_pool]
    adj_words = [w for w in adj_words if w not in dropped_pool]
    extra_words = [w for w in extra_words if w not in dropped_pool]

    tmpl_report = []
    for prefix, suffix in TEMPLATES:
        pre = tok.encode(prefix, add_special_tokens=False)
        suf = tok.encode(suffix, add_special_tokens=False)
        tmpl_report.append(dict(prefix=prefix, suffix=suffix, n_prefix_tokens=len(pre),
                                n_suffix_tokens=len(suf), colour_index=1 + len(pre),
                                suffix_index=2 + len(pre), final_index=1 + len(pre) + len(suf),
                                first_post_colour_token=tok.decode(suf[:1])))
    tokenisation_audit = dict(
        n_words_checked=len(label_words) + len(pool_words),
        n_templates=len(TEMPLATES),
        n_checks=(len(label_words) + len(pool_words)) * len(TEMPLATES),
        label_words_checked=label_words, label_mismatches=label_bad,
        pool_words_checked=len(pool_words),
        pool_words_dropped=sorted(dropped_pool),
        pool_mismatch_detail=pool_bad,
        rule="hue/achromatic mismatch aborts; pool mismatch drops the word (mechanical, "
             "label-blind)")
    print(f"\ntokenisation audit: {tokenisation_audit['n_checks']} word x template checks, "
          f"{len(label_bad)} label mismatches, {len(dropped_pool)} pool words dropped "
          f"{sorted(dropped_pool)}", flush=True)
    if label_bad:
        raise SystemExit(f"label words with off-manifold ids: "
                         f"{sorted({d['word'] for d in label_bad})}")
    if len(noun_words) < 40 or len(adj_words) < 20:
        raise SystemExit(f"background pools too small after the drop rule: "
                         f"{len(noun_words)} nouns, {len(adj_words)} adjectives")

    words = hue_words + ach_words + extra_words + noun_words + adj_words
    word_ids = [tok.encode(" " + w, add_special_tokens=False)[0] for w in words]
    idx = {w: i for i, w in enumerate(words)}
    hue_idx = np.array([idx[w] for w in hue_words])
    ach_idx = np.array([idx[w] for w in ach_words])
    noun_idx = np.array([idx[w] for w in noun_words])
    adj_idx = np.array([idx[w] for w in adj_words])

    # ---- forwards ---------------------------------------------------------
    print(f"\ncollecting {len(words)} words x {len(TEMPLATES)} templates x "
          f"{len(layers)} layers x 3 readouts ...", flush=True)
    tf = time.time()
    M = collect(model, tok, words, word_ids, layers, chunk=a.chunk)
    print(f"forwards done in {time.time() - tf:.1f}s", flush=True)
    del model
    if DEVICE == "mps":
        torch.mps.empty_cache()

    label_index = np.arange(K)
    rng = np.random.default_rng(0)

    # Earliest-layer token-readout hue plane: the embedding reference for the caveat.
    # `layers[0]` is layer 0 on a full sweep, but the layer is recorded by number rather
    # than being called "L0", so a partial --layers run cannot mislabel it (D12).
    ref_layer = layers[0]
    _, basis_L0_token, _ = analyse(M["token"][ref_layer][hue_idx], label_index, K,
                                   seed=0, do_order=False, both_conventions=False)

    rows, light_rows, bg_rows, jack_rows = [], [], [], []
    print(f"\n{'PRIMARY (pre-declared, top-2 PCA plane)':<80}"
          f"| {'D10 widened plane search':<40}")
    print(f"{'readout':>8} {'L':>3} {'top2':>6} {'radR':>7} {'gapR':>7} {'pUnif':>8} "
          f"{'pNoun':>8} {'FAnoun':>7} {'pAdj':>8} {'rho':>7} {'pOrd':>8} "
          f"{'ang(L0)':>8} | {'plane':>6} {'gapR':>6} {'pNoun':>7} {'rho':>6} {'pOrd':>7} "
          f" verdict", flush=True)
    for ro in READOUTS:
        for l in layers:
            P = M[ro][l][hue_idx]
            res, basis, xy = analyse(P, label_index, K, seed=l, do_order=True)
            A = res["conventions"]["A"]
            B = res["conventions"]["B"]
            # D7: closure is sign-invariant and rho is reflection-invariant. Assert it.
            sign_check = dict(
                gap_ratio_identical=bool(abs(A["gap_ratio"] - B["gap_ratio"]) < 1e-9),
                rho_identical=(None if res["degenerate"]
                               else bool(abs(A["order"]["rho"] - B["order"]["rho"]) < 1e-12)),
                rot_only_rho_A=(None if res["degenerate"] else A["order_rotation_only"]["rho"]),
                rot_only_rho_B=(None if res["degenerate"] else B["order_rotation_only"]["rho"]))

            row = dict(readout=ro, layer=l, k=K, **{kk: res[kk] for kk in
                                                    ("degenerate", "top2_var", "pc_var")},
                       conventions=res["conventions"], sign_convention_check=sign_check)

            # ---- background (b): matched nouns, identical pipeline -------
            sec, sec_basis = secondary_analyse(P, label_index, K, seed=l,
                                               do_order=not res["degenerate"])
            row_sec = sec

            def bg(pool_idx, n_draws, seed):
                """Identical pipeline, same arity, same positions -- only the words differ.

                One SVD per draw serves both tiers: the primary (PC1, PC2) gap ratio and
                the min-over-6-planes gap ratio the D10 search would have selected. The
                second is what makes the widened search testable -- every background draw
                is allowed exactly the same freedom the observation was.
                """
                r = np.random.default_rng(seed)
                gr = np.empty(n_draws)
                pc = np.empty(n_draws)
                rr = np.empty(n_draws)
                gb = np.empty(n_draws)
                for i in range(n_draws):
                    pick = pool_idx[r.choice(len(pool_idx), size=K, replace=False)]
                    gr[i], pc[i], gb[i], rr[i] = fast_closure(M[ro][l][pick])
                return dict(n_draws=int(n_draws),
                            gap_ratio_mean=float(gr.mean()),
                            gap_ratio_p05=float(np.percentile(gr, 5)),
                            gap_ratio_min=float(gr.min()),
                            radius_ratio_mean=float(rr.mean()),
                            false_acceptance_rate_p05=float((pc <= 0.05).mean()),
                            p_vs_background=float((1 + int((gr <= A["gap_ratio"]).sum()))
                                                  / (1 + n_draws)),
                            secondary_best_plane=dict(
                                gap_ratio_mean=float(gb.mean()),
                                gap_ratio_p05=float(np.percentile(gb, 5)),
                                gap_ratio_min=float(gb.min()),
                                p_vs_background=float((1 + int((gb <= sec["gap_ratio"]).sum()))
                                                      / (1 + n_draws))))

            b_noun = bg(noun_idx, a.n_background, 1000 + l)
            b_adj = bg(adj_idx, a.n_background, 2000 + l)
            sec["p_vs_noun_background"] = b_noun["secondary_best_plane"]["p_vs_background"]
            sec["p_vs_adjective_background"] = b_adj["secondary_best_plane"]["p_vs_background"]
            row["secondary_plane_search"] = sec
            row["principal_angles_secondary_to_L0_token_plane_deg"] = principal_angles_deg(
                sec_basis, basis_L0_token)
            row["background_nouns"] = b_noun
            row["background_adjectives_secondary"] = b_adj
            bg_rows.append(dict(readout=ro, layer=l, nouns=b_noun, adjectives=b_adj))

            # ---- plane angle vs the layer-0 token (embedding) hue plane ---
            pa = principal_angles_deg(basis, basis_L0_token)
            row["principal_angles_to_L0_token_plane_deg"] = pa

            # ---- verdict (D9) -- PRIMARY statistic only ---------------------
            if res["degenerate"]:
                verdict = "degenerate"
            else:
                closed = (A["p_closed_uniform"] < 0.05 and b_noun["p_vs_background"] < 0.05)
                ordered = A["order"]["p"] < 0.05
                verdict = ("hue_circle" if closed and ordered else
                           "closed_ring_unordered" if closed else
                           "ordered_arc" if ordered else "nothing")
            row["verdict"] = verdict
            # D10 verdict kept in its own field so a positive can never be mistaken for
            # one from the pre-declared primary. Closure here is judged ONLY against the
            # background that got the same min-over-planes freedom; the uniform-angle
            # p-value on a selected plane is invalid and is not used.
            if res["degenerate"]:
                verdict_sec = "degenerate"
            else:
                closed_s = sec["p_vs_noun_background"] < 0.05
                ordered_s = sec["order"]["p"] < 0.05
                verdict_sec = ("hue_circle_widened_search" if closed_s and ordered_s else
                               "closed_ring_unordered" if closed_s else
                               "ordered_arc" if ordered_s else "nothing")
            row["verdict_secondary_plane_search"] = verdict_sec
            rows.append(row)

            rho = None if res["degenerate"] else A["order"]["rho"]
            pord = None if res["degenerate"] else A["order"]["p"]
            srho = None if res["degenerate"] else sec["order"]["rho"]
            spo = None if res["degenerate"] else sec["order"]["p"]
            print(f"{ro:>8} {l:3d} {res['top2_var']:6.3f} {A['radius_ratio']:7.2f} "
                  f"{A['gap_ratio']:7.2f} {A['p_closed_uniform']:8.4f} "
                  f"{b_noun['p_vs_background']:8.4f} "
                  f"{b_noun['false_acceptance_rate_p05']:7.3f} "
                  f"{b_adj['p_vs_background']:8.4f} "
                  f"{'   n/a' if rho is None else f'{rho:+7.3f}'} "
                  f"{'     n/a' if pord is None else f'{pord:8.4f}'} "
                  f"{pa[0]:8.1f} | {str(sec['selected_plane']):>6} "
                  f"{sec['gap_ratio']:6.2f} {sec['p_vs_noun_background']:7.4f} "
                  f"{'   n/a' if srho is None else f'{srho:+6.3f}'} "
                  f"{'    n/a' if spo is None else f'{spo:7.4f}'}  {verdict}", flush=True)

            # ---- lightness (third direction) ------------------------------
            light_rows.append(dict(readout=ro, layer=l,
                                   **lightness_check(P, M[ro][l][ach_idx], ach_light,
                                                     basis, M[ro][l][noun_idx],
                                                     np.random.default_rng(3000 + l),
                                                     n_null=a.n_background)))

            # ---- leave-one-out jackknife where closure passes -------------
            if not res["degenerate"] and A["p_closed_uniform"] < 0.05:
                jk = []
                for j in range(K):
                    keep = [i for i in range(K) if i != j]
                    q, _, _ = analyse(P[keep], np.array(keep), K, seed=l,
                                      do_order=False, both_conventions=False)
                    qa = q["conventions"]["A"]
                    jk.append(dict(dropped=hue_words[j], gap_ratio=qa["gap_ratio"],
                                   p_closed_uniform=qa["p_closed_uniform"]))
                jack_rows.append(dict(readout=ro, layer=l, leave_one_out=jk,
                                      worst=max(jk, key=lambda x: x["p_closed_uniform"]),
                                      n_still_closed=int(sum(x["p_closed_uniform"] < 0.05
                                                             for x in jk))))

    # ---- exploratory: the three non-recited colours ------------------------
    non_recited = [w for w in hue_words if w not in RECITED]
    recited = [w for w in hue_words if w in RECITED]
    explor = dict(non_recited_members=non_recited, recited_members=recited,
                  note="Exploratory. The recited-six subset is the rainbow order that "
                       "appears verbatim in text; the remainder is not. Reported so a "
                       "reader can see whether the ring is carried by the recitation.")
    if len(recited) >= 6:
        sub = [hue_words.index(w) for w in recited]
        explor["recited_only"] = []
        for ro in READOUTS:
            for l in layers:
                q, _, _ = analyse(M[ro][l][hue_idx[sub]], np.arange(len(sub)), len(sub),
                                  seed=l, do_order=True, both_conventions=False)
                qa = q["conventions"]["A"]
                explor["recited_only"].append(dict(
                    readout=ro, layer=l, k=len(sub), gap_ratio=qa["gap_ratio"],
                    p_closed_uniform=qa["p_closed_uniform"], rho=qa["order"]["rho"],
                    p_order=qa["order"]["p"]))

    out = dict(
        declarations=dict(
            D1_hue_candidates=[dict(word=w, hue_deg=h) for w, h in HUE_CANDIDATES],
            D1_hue_set_used=hue_words, D1_hue_angles_used=hue_angles,
            D1_hue_dropped=hue_drop,
            D2_achromatic=[dict(word=w, lightness=l) for w, l in ACHROMATIC],
            D2_achromatic_used=ach_words, D2_achromatic_dropped=ach_drop,
            D2_note="achromatic held OUT of the hue circle; used only for the "
                    "lightness/third-direction check",
            D3_readouts=dict(token="the colour token itself -- embedding-confounded",
                             suffix="first token AFTER the colour -- PRIMARY computed readout",
                             final="last token of the template -- tertiary, exploratory"),
            D4_discriminator="closure; ordering alone is never evidence",
            D5_nulls=dict(uniform_angles=N_GAP_NULL, background_draws=a.n_background,
                          background_pool="single-token common nouns, same arity, "
                                          "identical templates and positions",
                          background_secondary="non-colour adjectives (grammaticality-matched)",
                          order_permutations=N_PERM_ORDER,
                          shuffle_control=N_SHUFFLE_CONTROL,
                          background_not_scored_for_ordering="random nouns have no true "
                                                             "cyclic order"),
            D6_exclusion_rule="words with an established colour sense or a canonical "
                              "colour name in English paint/fashion vocabulary, and all "
                              "lightness/saturation adjectives, are excluded from the "
                              "background pools",
            D6_exclusions=sorted(COLOUR_SENSE_EXCLUSIONS),
            D7_sign_conventions="both PC2 sign conventions reported; rho must be "
                                "identical (reflection-invariant), rotation-only rho "
                                "exposes handedness",
            D8_expected_confound="token readout closing at layers 0-2 is the embedding "
                                 "confound, declared in advance, not a finding",
            D9_verdict_rule=dict(
                hue_circle="p_closed_uniform < 0.05 AND p_vs_noun_background < 0.05 "
                           "AND ordering p < 0.05",
                closed_ring_unordered="both closure tests pass, ordering does not",
                ordered_arc="ordering passes, closure does not",
                nothing="neither", degenerate="point cloud carries no usable geometry"),
            caveat_that_survives_a_positive=(
                "A closed ring at the suffix readout shows hue information is present "
                "one token later; attention can copy an embedding forward, so this is "
                "not by itself proof of recomputation. The principal angle between the "
                "suffix hue plane and the layer-0 token hue plane is the only handle "
                "reported on that."),
            D12_recovery_amendments=RECOVERY_AMENDMENTS,
            D12_note="an earlier launch of this script was interrupted before it wrote "
                     "anything; these are every change made on the recovery pass, all of "
                     "them before any activation was read. D1-D11 are untouched.",
        ),
        model="google/gemma-3-1b-pt", layers=layers, n_templates=len(TEMPLATES),
        templates=tmpl_report, tokenisation_audit=tokenisation_audit,
        embedding_reference_plane=dict(
            readout="token", layer=int(ref_layer),
            note="the plane every `principal_angles_*` field is measured against: the "
                 "hue point cloud read at the colour token itself, at the earliest layer "
                 "in this sweep. A suffix-readout plane sitting at a small angle to it "
                 "is consistent with an embedding copied forward."),
        token_audit=dict(hue_kept=hue_keep, hue_dropped=hue_drop,
                         achromatic_kept=ach_keep, achromatic_dropped=ach_drop,
                         exploratory_extra_kept=extra_keep,
                         exploratory_extra_dropped=extra_drop,
                         exploratory_extra_used=extra_words,
                         noun_pool_kept=noun_words, noun_pool_dropped=noun_drop,
                         adjective_pool_kept=adj_words, adjective_pool_dropped=adj_drop,
                         orange_purple_resolution=op_res),
        reference_check=ref, synthetics=syn,
        rows=rows, lightness=light_rows, jackknife=jack_rows, exploratory=explor,
    )

    # ---- D11 multiplicity --------------------------------------------------
    live = [r for r in rows if not r["degenerate"]]
    n_cells = len(live)

    def fdr(pvals, label, n_null):
        """BH-FDR plus the floor the null size imposes (D12).

        Every p here is an add-one empirical fraction, so it cannot go below
        1/(n_null+1). BH's rank-1 threshold is alpha/n_cells. When the floor sits above
        it, "0 survive BH" can mean the test was unreachable rather than that nothing was
        there -- so both numbers are reported next to each other and the comparison is
        made explicit instead of left to the reader.
        """
        p = np.asarray(pvals, dtype=float)
        keep = benjamini_hochberg(p, alpha=0.05) if len(p) else np.zeros(0, bool)
        floor = 1.0 / (n_null + 1.0)
        rank1 = 0.05 / max(len(p), 1)
        return dict(statistic=label, n_cells=int(len(p)), n_null_draws=int(n_null),
                    n_raw_below_05=int((p < 0.05).sum()),
                    expected_by_chance_at_05=float(0.05 * len(p)),
                    p_value_floor=float(floor),
                    bh_rank1_threshold=float(rank1),
                    isolated_cell_can_survive_bh=bool(floor <= rank1),
                    min_cells_at_floor_to_survive_bh=int(np.ceil(floor / rank1))
                    if rank1 > 0 else None,
                    n_at_floor=int((p <= floor + 1e-12).sum()),
                    n_surviving_bh_fdr=int(keep.sum()),
                    surviving=[dict(readout=live[i]["readout"], layer=live[i]["layer"],
                                    p=float(p[i])) for i in np.flatnonzero(keep)])

    nb = a.n_background
    mult = dict(
        n_cells=int(n_cells),
        note="3 readouts x 26 layers. An isolated pass at alpha = 0.05 is expected by "
             "chance and is not a finding; BH-FDR across all live cells is the honest "
             "count. Each row also carries its p-value floor, because an empirical "
             "p-value cannot fall below 1/(n_null+1) and a BH column has to be read "
             "against what was reachable.",
        ordering_primary=fdr([r["conventions"]["A"]["order"]["p"] for r in live],
                             "ordering p, primary plane", N_PERM_ORDER),
        closure_uniform_primary=fdr([r["conventions"]["A"]["p_closed_uniform"] for r in live],
                                    "closure p vs uniform angles, primary plane", N_GAP_NULL),
        closure_background_primary=fdr([r["background_nouns"]["p_vs_background"] for r in live],
                                       "closure p vs matched noun background, primary plane",
                                       nb),
        closure_background_secondary=fdr(
            [r["secondary_plane_search"]["p_vs_noun_background"] for r in live],
            "closure p vs matched noun background, D10 selected plane", nb),
        ordering_secondary=fdr([r["secondary_plane_search"]["order"]["p"] for r in live],
                               "ordering p, D10 selected plane", N_PERM_ORDER),
    )
    out["multiplicity"] = mult
    out["verdict_counts"] = {
        ro: {name: [r["layer"] for r in rows
                    if r["readout"] == ro and r["verdict"] == name]
             for name in ("hue_circle", "closed_ring_unordered", "ordered_arc",
                          "nothing", "degenerate")}
        for ro in READOUTS}
    out["verdict_counts_secondary"] = {
        ro: {name: [r["layer"] for r in rows
                    if r["readout"] == ro and r["verdict_secondary_plane_search"] == name]
             for name in ("hue_circle_widened_search", "closed_ring_unordered",
                          "ordered_arc", "nothing", "degenerate")}
        for ro in READOUTS}
    out["runtime_sec"] = float(time.time() - t0)

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=2, default=jsonable))

    # ---- summary -----------------------------------------------------------
    print("\n" + "=" * 78)
    print("VERDICT SUMMARY -- D9, PRIMARY pre-declared statistic")
    for ro in READOUTS:
        for name in ("hue_circle", "closed_ring_unordered", "ordered_arc", "nothing",
                     "degenerate"):
            ls = out["verdict_counts"][ro][name]
            if ls:
                print(f"  {ro:>8} {name:>22}: {len(ls):2d} layers {ls}")
    print("\nVERDICT SUMMARY -- D10 widened plane search (labelled, not pre-declared)")
    for ro in READOUTS:
        for name in ("hue_circle_widened_search", "closed_ring_unordered", "ordered_arc",
                     "nothing", "degenerate"):
            ls = out["verdict_counts_secondary"][ro][name]
            if ls:
                print(f"  {ro:>8} {name:>26}: {len(ls):2d} layers {ls}")
    print(f"\nD11 MULTIPLICITY over {n_cells} live cells "
          f"(expected by chance at alpha=0.05: {0.05 * n_cells:.1f})")
    for key in ("ordering_primary", "closure_uniform_primary", "closure_background_primary",
                "ordering_secondary", "closure_background_secondary"):
        m = mult[key]
        print(f"  {key:30s} raw p<0.05: {m['n_raw_below_05']:2d}   "
              f"survives BH-FDR: {m['n_surviving_bh_fdr']:2d}   "
              f"[p floor {m['p_value_floor']:.2e}, BH rank-1 {m['bh_rank1_threshold']:.2e}, "
              f"isolated cell reachable: {m['isolated_cell_can_survive_bh']}]"
              + (f"   {[(s['readout'], s['layer']) for s in m['surviving']]}"
                 if m["surviving"] else ""))
    print(f"\nruntime {time.time() - t0:.1f}s -> wrote {a.out}")


if __name__ == "__main__":
    main()
