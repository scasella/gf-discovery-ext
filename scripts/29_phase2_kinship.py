"""PHASE 2 -- KINSHIP: is there a 2-D lattice (generation x lineality x gender)?

Bound by docs/phase2_prereg_2026-08-21.md (frozen 2026-08-21). The label set, the three
contrast directions, the axis test, the nulls and the verdict vocabulary come from that
document and from src/gfdisc/ontologies.py as committed in 3cadb69 (2026-08-18). Nothing
here invents a prediction.

Kinship is the one Phase-2 ontology whose predicted structure is NOT a ring. A confirmed
2-D lattice would be the first non-circular geometry recovered in this project. The
prereg's three predictions, restated verbatim in force:

  P1  the terms CO-LOCATE in the frozen label-blind partitions, more than matched random
      type sets of the same arity;
  P2  the lattice contrasts are SEPARABLE directions in the member point cloud;
  P3  closure FAILS.  A lattice is not a cycle.  P3 is a positive prediction, not a
      hedge: if closure passes, the kinship prediction is WRONG and the closure tier's
      specificity is in question.  It may not be re-read as a success.

======================================================================================
DECLARATIONS -- fixed before any placement is located and before any activation is read.
All of this is written into results/phase2_kinship.json under "declarations".
======================================================================================

D1  LABEL SET AND CONTRASTS, with provenance on every one.
      KINSHIP (13 terms), KINSHIP_GENERATION, KINSHIP_LINEAL are IMPORTED, not retyped,
      from src/gfdisc/ontologies.py -- committed 3cadb69, 2026-08-18 12:04:20 -0400.
      GENDER (male=+1, female=-1, cousin=0 and excluded from the gender contrast) is
      ADDED AT PREREG TIME, 2026-08-21, and is NOT in 3cadb69.  Prereg 1.2 requires that
      distinction to be preserved in every reported number; every gender field in the
      output JSON carries provenance="prereg_2026-08-21_NOT_in_3cadb69".

D2  BLIND ARM = the FROZEN partitions of results/token_type_geometry.json.  That artifact
      stores members_token_ids only for clusters <= 80 that were flagged or held one of
      its OWN declared label sets; kinship is not one of them, so membership must be
      REPRODUCED.  Reproduction protocol, declared before it is run:
        - per (layer, arm) re-run Leiden ONCE at the RECORDED selected_tau with the
          recorded resolution/seed.  No sweep, no grid, no new threshold, no new arm.
          `refine` and `hdbscan` likewise at their recorded settings.
        - VERIFY: stored n_clusters must match, and every stored `screened` row's
          member_hash must appear in the reproduction.  screen() is re-run on one stored
          cluster per (layer, arm) and must reproduce max_gap_deg and p_closed.
        - ON ANY MISMATCH the blind arm STOPS and reports irreproducibility.  Nothing is
          adjusted to force a match; adjusting would be re-clustering, which prereg 4.1
          forbids.  Per-(layer,arm) match counts are reported whatever they are.

D3  THE REGISTERED AXIS ESTIMATOR (prereg 3.2 step 1-2), implemented verbatim:
        M = member means, column-centred over the member set only
        d_c = argmin_d ||M d - y_c||^2 with ridge lam = 1e-3 * trace(M^T M) / dim
        score = AUC between code groups on M d_c (binary), Spearman rho vs y_c (ordinal)
        null  = 1000 random reassignments of the codes among the SAME members, the whole
                fit-and-score repeated per draw; p = (1 + #{null >= obs}) / (1 + n_draws)

D4  A DATA-FREE CALIBRATION OF D3, RUN FIRST, BEFORE ANY DATA -- in the spirit of prereg
      section 6, which computed reference-ring power before trusting a threshold.
      M is column-centred with k < dim (k = 6 blind, k = 13 supervised, dim = 1152), so
      the image of d -> M d is EXACTLY the set of zero-sum vectors in R^k, and the
      registered ridge is ~5e-6 of the mean Gram eigenvalue.  The fit therefore
      interpolates the centred codes exactly, for EVERY assignment, so the score is 1.000
      for the observed codes AND for every shuffled draw, and p is pinned at 1.0.
      This is asserted numerically on synthetic M at k = 6 and k = 13 in stage A before
      any kinship number exists.  If the assertion fails, the calibration is wrong and
      the run aborts.  The registered estimator is still COMPUTED AND REPORTED on real
      data exactly as registered; it is simply reported as structurally uninformative
      rather than as a negative.

D5  PRE-DECLARED SECONDARY ESTIMATORS.  PROVENANCE STATED PLAINLY: these are NOT in the
      prereg and NOT in 3cadb69.  They were added while writing this script, BEFORE any
      kinship placement was located and BEFORE any activation was read, as the response
      to D4.  Prereg section 7 permits a post-hoc addition with its provenance stated;
      this is that, declared up front rather than after a result.
        S1  "correlation direction": d = M^T y_centred, so the projection is G y_centred
            with G = M M^T.  This is the difference-of-group-means direction for a
            balanced binary contrast.  Not saturating: G is k x k and fixed.
        S2  "leave-one-out ridge": the D3 estimator refitted on the k-1 members other
            than i, scoring the HELD-OUT projection of member i.  Standard cure for
            in-sample interpolation.
      Both carry the identical 1000-draw shuffled-code null, and both are invariant under
      y -> -y for the same reason D3 is, so D6 applies to them too.

D6  EXHAUSTIVE ENUMERATION OF THE SHUFFLE NULL -- data-free, computed in stage A.
      Every estimator here is invariant under negation of the codes (y -> -y gives
      d -> -d gives projection -> -projection; AUC swaps groups and flips sign, Spearman
      flips twice).  The true assignment therefore always TIES with its own negation in
      the null, which puts a FLOOR under the achievable p-value.  Where the assignment
      space is small it is enumerated EXHAUSTIVELY and the exact floor reported:
        blind gender      3 vs 3 on 6 members : C(6,3) = 20 arrangements, +/- paired
                          -> floor p = 2/20 = 0.10.  CANNOT REACH p < 0.05 FOR A PERFECT
                          LATTICE.  Registered here, before any placement, as
                          `untestable` by prereg 4.1(ii)'s own principle -- not as a
                          negative, and not as a failure of the model.
        blind generation  codes {1,1,-1,-1,0,0} : 6!/(2!2!2!) = 90 arrangements, +/-
                          paired -> floor p = 2/90 = 0.0222.  Testable, barely.
      Because the blind LATTICE claim requires BOTH contrasts (prereg 3.2 step 4), the
      blind lattice claim is `untestable` at alpha = 0.05 by construction.  Supervised
      floors (gender 6v6 of 12 -> 924 arrangements; generation and lineality on 13) are
      far below 0.05 and are reported.  Where the space is <= 200000, the EXACT
      enumerated p is reported alongside the registered 1000-draw sampled p.

D7  THE ADDITIVE RANK-2 MODEL (prereg 3.2 step 4) is NOT saturated and is the blind arm's
      one fully-powered registered statistic: m_w ~ mu + gen_w * a + gender_w * b is a
      3-column design over k members predicting all 1152 dimensions, so it leaves k - 3
      residual degrees of freedom.  R^2 = 1 - ||M - X B||_F^2 / ||M - mean||_F^2 against
      the same shuffled-code null (primary: the (gen, gender) pair is permuted JOINTLY
      across members, preserving the code lattice; secondary: independent permutation).

D8  P3, CLOSURE, MUST ACTUALLY RUN, in both arms, whatever the axis results do.  Blind:
      ttg.screen on the 6 member means (k = 6 is inside the frozen screening band 5..60),
      p_closed at 20000 uniform-angle draws, p_bg against 500 matched same-size subsets of
      the identical >= 100 pool, seeded by the declared rule below.  Supervised: the same
      on the 13-word templated means per (readout, layer), p_bg against 2000 matched
      same-arity noun draws through the IDENTICAL templates at the IDENTICAL positions.
      A closure PASS is reported as a FAILURE of the kinship prediction.

D9  SUPERVISED ARM, patterned on the verified scripts/24_colors_existence.py:
      26 layers x 3 readouts (`token` = EMBEDDING-CONFOUNDED, `suffix` = PRIMARY COMPUTED
      READOUT = first token after the kinship word, `final` = tertiary) x 16 templates.
      Every template is PREFIX + " " + WORD + SUFFIX; suffix >= 3 tokens so `suffix` and
      `final` are distinct positions; the suffix is token-identical across the 13 words
      within a template; NO template contains another kinship term; every prefix forces
      the family sense.  Manual ids are asserted equal to the natural tokenisation of the
      whole sentence for every word and every template.

D10 BACKGROUND POOLS.  The colours noun pool (scripts/24_colors_existence.py) reused
      verbatim, minus a declared kinship-sense exclusion list, minus calendar words.  The
      prereg registers NO kinship-specific secondary pool (4.2c names only keys, compass
      and zodiac), so none is invented: nouns only.  Background draws are scored for
      CLOSURE, never for ordering.

D11 BOTH SVD SIGN CONVENTIONS.  The angular gap is invariant under a PC2 sign flip; that
      identity is ASSERTED numerically per cell, not assumed.  The axis scores are
      likewise computed under both conventions and asserted identical (they are functions
      of M, not of any basis).

D12 MULTIPLICITY.  The supervised sweep is 3 readouts x 26 layers = 78 cells per
      statistic family.  Benjamini-Hochberg within each family, reported next to raw
      counts and next to the expected-by-chance count (~3.9 per 78 at alpha = 0.05).
      An isolated single-cell pass is not a finding.

D13 SEED RULE, declared before use:
      int(md5(f"{stage}|{layer}|{arm}|{k}|{tag}").hexdigest()[:8], 16) % (2**31).

D14 VERDICT VOCABULARY (prereg 3.2 and 4.3, no additions):
      recovered_blind | structure_exists_supervised_only | ordered_not_closed |
      untestable | clean_negative.

CAVEATS THAT SURVIVE ANY POSITIVE (prereg 4.5, inherited, not restated as new):
  the blind representation is a context-AVERAGED token-type mean -- closer to a static
  embedding than to a computed feature.  Layers 0-4 are embedding-adjacent.  In the
  supervised arm a structure at the `suffix` readout shows the information is present one
  token later; attention can copy an embedding forward, so it is not a computation claim.
"""
import argparse, gc, hashlib, importlib.util, itertools, json, math, os, sys, time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from gfdisc.ontologies import KINSHIP, KINSHIP_GENERATION, KINSHIP_LINEAL
from gfdisc.nulls import benjamini_hochberg

D_MODEL = 1152
RIDGE_SCALE = 1e-3
N_SHUFFLE = 1000
N_ENUM_MAX = 5_000
N_NOISE_DRAWS = 100
N_COLOC_DRAWS = 2000
N_BG_BLIND = 500
N_BG_SUP = 2000
N_PERM_GAP = 20000
ALPHA = 0.05

# ---- D1: gender, added at prereg 2026-08-21, NOT in 3cadb69 ---------------------------
MALE = {"grandfather", "father", "son", "brother", "uncle", "nephew"}
FEMALE = {"grandmother", "mother", "daughter", "sister", "aunt", "niece"}
KINSHIP_GENDER = [1 if w in MALE else (-1 if w in FEMALE else 0) for w in KINSHIP]
GENDER_PROVENANCE = "prereg_2026-08-21_NOT_in_3cadb69"
COMMITTED_PROVENANCE = "committed_3cadb69_2026-08-18"

CONTRASTS = {
    # exclude_zero distinguishes a code of 0 that MEANS a group (lineal: collateral) from
    # a code of 0 that means "excluded from this contrast" (gender: cousin, prereg 3.2).
    "generation": dict(codes=KINSHIP_GENERATION, kind="ordinal", exclude_zero=False,
                       provenance=COMMITTED_PROVENANCE),
    "lineal": dict(codes=KINSHIP_LINEAL, kind="binary", exclude_zero=False,
                   provenance=COMMITTED_PROVENANCE),
    "gender": dict(codes=KINSHIP_GENDER, kind="binary", exclude_zero=True,
                   provenance=GENDER_PROVENANCE),
}

# ---- D9: templates.  (prefix, suffix); the word sits between them with one space. -----
# Every prefix forces the family sense.  No template contains another kinship term.
# Every suffix is >= 3 tokens.  Checked mechanically in stage C before any forward pass.
TEMPLATES = [
    ("At the family gathering everyone greeted the", " who had just arrived"),
    ("On the census form she listed one", " living in the house"),
    ("The lawyer read the will aloud to the", " sitting by the window"),
    ("In our household the oldest", " always locks the door"),
    ("He wrote a long letter to his", " about the move"),
    ("She sent a birthday card to her", " without fail every year"),
    ("The photograph on the mantelpiece shows a", " smiling broadly at everyone"),
    ("According to the parish register the", " was born nearby"),
    ("They named the youngest", " after a distant relation"),
    ("Each year the whole clan visits the", " by the lake"),
    ("The inheritance was divided between the", " and the estate"),
    ("A relative described the", " as a quiet neighbour"),
    ("The genealogy chart places the", " on the left branch"),
    ("During the reunion she introduced the", " to everyone present"),
    ("Our neighbour still talks about the", " who lived here"),
    ("The obituary mentioned a surviving", " and many friends"),
]
READOUTS = ("token", "suffix", "final")

# ---- D10: background pool.  Reused verbatim from scripts/24_colors_existence.py. ------
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
KINSHIP_SENSE_EXCLUSIONS = [
    "family", "parent", "child", "children", "kin", "relative", "sibling", "spouse",
    "wife", "husband", "baby", "boy", "girl", "man", "woman", "dad", "mom", "mum",
    "papa", "mama", "grandma", "grandpa", "granddaughter", "grandson", "stepmother",
    "stepfather", "in-law", "widow", "widower", "heir", "descendant", "ancestor",
    "January", "February", "March", "April", "May", "June", "July", "August",
    "September", "October", "November", "December", "Monday", "Tuesday", "Wednesday",
    "Thursday", "Friday", "Saturday", "Sunday", "spring", "summer", "autumn", "winter",
]


# =======================================================================================
# statistics
# =======================================================================================
def _seed(stage, layer, arm, k, tag=""):
    h = hashlib.md5(f"{stage}|{layer}|{arm}|{k}|{tag}".encode()).hexdigest()[:8]
    return int(h, 16) % (2 ** 31)


def auc(scores, groups):
    """AUC between groups coded +1 / -1 on a 1-D score.  Ties count half."""
    s = np.asarray(scores, float); g = np.asarray(groups)
    a, b = s[g > 0], s[g < 0]
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    d = a[:, None] - b[None, :]
    return float(((d > 0).sum() + 0.5 * (d == 0).sum()) / (len(a) * len(b)))


def _ranks(v):
    v = np.asarray(v, float)
    order = np.argsort(v, kind="mergesort")
    r = np.empty(len(v), float)
    r[order] = np.arange(len(v), dtype=float)
    # average ties
    uniq, inv, cnt = np.unique(v, return_inverse=True, return_counts=True)
    if (cnt > 1).any():
        sums = np.zeros(len(uniq)); np.add.at(sums, inv, r)
        r = (sums / cnt)[inv]
    return r


def spearman(a, b):
    ra, rb = _ranks(a), _ranks(b)
    ra = ra - ra.mean(); rb = rb - rb.mean()
    den = np.linalg.norm(ra) * np.linalg.norm(rb)
    return float(ra @ rb / den) if den > 0 else float("nan")


def _centre_rows(M):
    M = np.asarray(M, float)
    return M - M.mean(0, keepdims=True)


# All three estimators are functions of the Gram matrix G = Mc Mc^T alone, so G is formed
# once per point cloud and every shuffle draw costs O(k^2).  The Gram form is not an
# approximation: it is the standard ridge identity (M^T M + lam I)^-1 M^T = M^T (M M^T +
# lam I)^-1, asserted numerically against the primal form in stage A.
def proj_registered(G, y, dim=D_MODEL, ridge_scale=RIDGE_SCALE):
    """prereg 3.2 step 1: d = (M^T M + lam I)^-1 M^T y, so M d = G (G + lam I)^-1 y."""
    lam = ridge_scale * float(np.trace(G)) / dim
    return G @ np.linalg.solve(G + lam * np.eye(len(G)), np.asarray(y, float))


def proj_s1(G, y, dim=D_MODEL, ridge_scale=RIDGE_SCALE):
    """D5 S1: correlation / difference-of-means direction.  d = M^T y_c, proj = G y_c."""
    yc = np.asarray(y, float); yc = yc - yc.mean()
    return G @ yc


def proj_s2(G, y, dim=D_MODEL, ridge_scale=RIDGE_SCALE):
    """D5 S2: leave-one-out ridge, entirely in the Gram domain.

    For held-out i the training rows are re-centred on the other k-1 members; with
    C = I - 11^T/(k-1) the training Gram is C G[idx,idx] C and the held-out member's
    inner products against the re-centred training rows are h - mean(h) where
    h = G[i, idx] - G[idx, idx].mean(0).  Identical to fitting in 1152-d.
    """
    y = np.asarray(y, float)
    k = len(G)
    out = np.empty(k)
    for i in range(k):
        idx = np.delete(np.arange(k), i)
        Gs = G[np.ix_(idx, idx)]
        col = Gs.mean(0)
        Gc = Gs - col[None, :] - col[:, None] + Gs.mean()
        lam = ridge_scale * float(np.trace(Gc)) / dim
        alpha = np.linalg.solve(Gc + lam * np.eye(len(Gc)), y[idx] - y[idx].mean())
        h = G[i, idx] - col
        out[i] = float((h - h.mean()) @ alpha)
    return out


def dir_s1(Mc, y):
    """The S1 direction itself, in 1152-d.  Used only for the orthogonality report."""
    yc = np.asarray(y, float); yc = yc - yc.mean()
    return Mc.T @ yc


ESTIMATORS = {"registered": proj_registered, "s1_correlation": proj_s1,
              "s2_loo_ridge": proj_s2}


def score_of(proj, y, kind):
    if kind == "binary":
        return auc(proj, y)
    return spearman(proj, y)


def _multiset_permutations(counts, vals, prefix, out):
    if not any(counts):
        out.append(tuple(prefix)); return
    for i, c in enumerate(counts):
        if c:
            counts[i] -= 1; prefix.append(vals[i])
            _multiset_permutations(counts, vals, prefix, out)
            prefix.pop(); counts[i] += 1


def distinct_arrangements(codes, cap=N_ENUM_MAX):
    """All distinct permutations of the code multiset, or None if there are too many.

    Generated directly from the multiset -- itertools.permutations would enumerate k!
    tuples (6.2e9 at k = 13) to yield the same few hundred distinct arrangements.
    """
    codes = list(codes)
    n = math.factorial(len(codes))
    for c in np.bincount(np.asarray(codes) - min(codes)):
        n //= math.factorial(int(c))
    if n > cap:
        return None, int(n)
    vals = sorted(set(codes))
    counts = [codes.count(v) for v in vals]
    out = []
    _multiset_permutations(counts, vals, [], out)
    return [np.array(p, dtype=float) for p in out], int(n)


def restrict(M, codes, kind, exclude_zero):
    """Apply the contrast's own membership rule.

    gender has exclude_zero=True because prereg 3.2 excludes cousin from that contrast;
    lineal has exclude_zero=False because 0 there IS a group (collateral).  Binary codes
    are recoded to +/-1.  The recoding is affine in y, and every statistic here is
    invariant under an affine reparametrisation of the codes (M d is zero-sum, so a shift
    is absorbed and a positive scale cancels in AUC, Spearman and R^2)."""
    M = np.asarray(M, float); c = np.asarray(codes)
    m = (c != 0) if exclude_zero else np.ones(len(c), bool)
    M, cc = M[m], c[m]
    if kind != "binary":
        return M, cc.astype(float), False
    vals = sorted(set(cc.tolist()))
    rec = np.where(cc == vals[-1], 1.0, -1.0)
    return M, rec, bool(set(vals) != {-1, 1})


def axis_test(M, codes, kind, exclude_zero, seed, n_shuffle=N_SHUFFLE,
              estimators=ESTIMATORS, enumerate_null=True):
    """One contrast, all estimators, sampled null + exact enumerated null where feasible."""
    Msub, y, recoded = restrict(M, codes, kind, exclude_zero)
    Mc = _centre_rows(Msub)
    G = Mc @ Mc.T
    rng = np.random.default_rng(seed)
    draws = [rng.permutation(y) for _ in range(n_shuffle)]
    allasg, n_distinct = distinct_arrangements([int(v) for v in y]) if enumerate_null \
        else (None, None)
    out = dict(_meta=dict(n_scored_members=int(len(y)), kind=kind,
                          codes_scored=[float(v) for v in y],
                          binary_recoded_to_pm1=bool(recoded),
                          zero_coded_members_dropped=int(len(M) - len(Msub))))
    for name, fn in estimators.items():
        obs = score_of(fn(G, y), y, kind)
        null = np.array([score_of(fn(G, yy), yy, kind) for yy in draws])
        p_sampled = float((1 + int((null >= obs - 1e-12).sum())) / (1 + len(null)))
        rec = dict(observed=float(obs), p_sampled=p_sampled,
                   null_mean=float(np.nanmean(null)), null_p95=float(np.nanpercentile(null, 95)),
                   null_max=float(np.nanmax(null)), n_shuffle=int(len(null)),
                   n_null_tied_with_observed=int((np.abs(null - obs) <= 1e-12).sum()))
        if allasg is not None:
            ex = np.array([score_of(fn(G, yy), yy, kind) for yy in allasg])
            rec.update(p_exact=float((ex >= obs - 1e-12).sum() / len(ex)),
                       n_arrangements=int(len(ex)),
                       exact_floor_p=float((ex >= ex.max() - 1e-12).sum() / len(ex)),
                       exact_note="exhaustive enumeration of the code-assignment space")
        elif n_distinct is not None:
            rec.update(n_arrangements=int(n_distinct),
                       exact_note=f"{n_distinct} arrangements > cap {N_ENUM_MAX}; sampled null only")
        out[name] = rec
    return out


def additive_r2(M, codes_a, codes_b):
    """D7: R^2 of m_w ~ mu + a*code_a + b*code_b over the members."""
    M = np.asarray(M, float)
    X = np.column_stack([np.ones(len(M)), np.asarray(codes_a, float),
                         np.asarray(codes_b, float)])
    B, *_ = np.linalg.lstsq(X, M, rcond=None)
    resid = M - X @ B
    tot = M - M.mean(0, keepdims=True)
    ss_t = float((tot ** 2).sum())
    return float(1.0 - (resid ** 2).sum() / ss_t) if ss_t > 0 else float("nan")


def additive_test(M, codes_a, codes_b, seed, n_shuffle=N_SHUFFLE):
    rng = np.random.default_rng(seed)
    a = np.asarray(codes_a, float); b = np.asarray(codes_b, float)
    obs = additive_r2(M, a, b)
    joint, indep = [], []
    for _ in range(n_shuffle):
        p = rng.permutation(len(a))
        joint.append(additive_r2(M, a[p], b[p]))
        indep.append(additive_r2(M, rng.permutation(a), rng.permutation(b)))
    joint = np.array(joint); indep = np.array(indep)
    return dict(r2=float(obs),
                p_joint_shuffle=float((1 + int((joint >= obs - 1e-12).sum())) / (1 + len(joint))),
                p_independent_shuffle=float((1 + int((indep >= obs - 1e-12).sum())) / (1 + len(indep))),
                null_joint_mean=float(joint.mean()), null_joint_p95=float(np.percentile(joint, 95)),
                null_independent_mean=float(indep.mean()),
                n_shuffle=int(n_shuffle), residual_df=int(len(M) - 3),
                primary="p_joint_shuffle")


def cos_abs(u, v):
    nu, nv = np.linalg.norm(u), np.linalg.norm(v)
    return float(abs(u @ v) / (nu * nv)) if nu > 0 and nv > 0 else float("nan")


def principal_angles_deg(A, B):
    Qa = np.linalg.qr(np.asarray(A, float).T)[0]
    Qb = np.linalg.qr(np.asarray(B, float).T)[0]
    s = np.clip(np.linalg.svd(Qa.T @ Qb, compute_uv=False), -1.0, 1.0)
    return [float(np.degrees(np.arccos(x))) for x in s]


# =======================================================================================
# STAGE A -- data-free calibration (D4, D6) and synthetic controls
# =======================================================================================
def stage_a(ttg, log):
    t0 = time.time()
    out = {}
    rng = np.random.default_rng(0)

    # --- A0: assert the dual-form ridge equals the primal form the prereg writes -------
    M = _centre_rows(rng.normal(size=(6, D_MODEL)))
    y = np.array([1., 1., -1., -1., 0., 0.])
    lam = RIDGE_SCALE * float(np.trace(M.T @ M)) / D_MODEL
    d_primal = np.linalg.solve(M.T @ M + lam * np.eye(D_MODEL), M.T @ y)
    out["ridge_dual_form_check"] = dict(
        max_abs_diff=float(np.abs(M @ d_primal - proj_registered(M @ M.T, y)).max()),
        note="prereg 3.2 writes d = (M^T M + lam I)^-1 M^T y; the code uses the identical "
             "dual form G(G+lam I)^-1 y.  Asserted equal, not assumed.")
    assert out["ridge_dual_form_check"]["max_abs_diff"] < 1e-6

    # --- A0b: assert the Gram-domain LOO equals a direct 1152-d LOO refit ---------------
    def loo_direct(Mc, yv):
        k = len(Mc); o = np.empty(k)
        for i in range(k):
            idx = np.delete(np.arange(k), i)
            Mtr = Mc[idx]; mu = Mtr.mean(0, keepdims=True); Mtrc = Mtr - mu
            lm = RIDGE_SCALE * float(np.trace(Mtrc.T @ Mtrc)) / D_MODEL
            d = np.linalg.solve(Mtrc.T @ Mtrc + lm * np.eye(D_MODEL),
                                Mtrc.T @ (yv[idx] - yv[idx].mean()))
            o[i] = float((Mc[i] - mu[0]) @ d)
        return o
    out["loo_gram_form_check"] = dict(
        max_abs_diff=float(np.abs(loo_direct(M, y) - proj_s2(M @ M.T, y)).max()),
        note="S2 leave-one-out computed in the Gram domain, asserted equal to a direct "
             "1152-d refit.")
    assert out["loo_gram_form_check"]["max_abs_diff"] < 1e-6

    # --- A1: D4 saturation of the registered estimator, at both arities ----------------
    sat = {}
    for k, codes, kind in ((6, [1, 1, -1, -1, -1, 1], "binary"),
                           (6, [1, 1, -1, -1, 0, 0], "ordinal"),
                           (12, [1] * 6 + [-1] * 6, "binary"),
                           (13, KINSHIP_GENERATION, "ordinal")):
        Ms = _centre_rows(np.random.default_rng(k).normal(size=(k, D_MODEL)))
        G = Ms @ Ms.T
        asg, n_dist = distinct_arrangements(codes, cap=2000)
        if asg is None:
            asg = [np.random.default_rng(i).permutation(np.asarray(codes, float))
                   for i in range(200)]
        sc = np.array([score_of(proj_registered(G, a), a, kind) for a in asg])
        lam = RIDGE_SCALE * float(np.trace(G)) / D_MODEL
        ev = np.linalg.eigvalsh(G)
        yy = np.asarray(asg[0], float); yc = yy - yy.mean()
        sat[f"k{k}_{kind}"] = dict(
            k=k, kind=kind, n_assignments_scored=int(len(sc)),
            score_min=float(sc.min()), score_max=float(sc.max()), score_mean=float(sc.mean()),
            score_range=float(sc.max() - sc.min()),
            constant_across_assignments=bool(sc.max() - sc.min() < 1e-6),
            frac_at_1=float((sc >= 1 - 1e-9).mean()),
            ridge_over_mean_gram_eigenvalue=float(lam / max(ev.mean(), 1e-30)),
            max_abs_interpolation_error=float(np.abs(proj_registered(G, yy) - yc).max()),
            norm_centred_codes=float(np.linalg.norm(yc)))
    out["registered_estimator_saturation"] = sat
    out["registered_estimator_saturation_verdict"] = (
        "SATURATED: the registered prereg-3.2 estimator returns the SAME score for every "
        "code assignment at k < dim (exactly 1.000 for the AUC contrasts; for the Spearman "
        "contrasts a constant just under 1, the shortfall being tie-breaking on the "
        "1e-6-scale interpolation residual, not signal).  Observed and shuffled scores are "
        "identical, so the shuffle p-value is pinned at 1.0 and the test can neither pass "
        "nor fail.  Reported as structurally uninformative, NOT as a negative.")
    assert all(v["constant_across_assignments"] for v in sat.values()), \
        "D4 calibration failed: the registered estimator did not saturate as predicted"
    log("A1 registered-estimator saturation confirmed: "
        + ", ".join(f"{key} score={v['score_min']:.6f} range={v['score_range']:.2e}"
                    for key, v in sat.items()))

    # --- A2: D6 exhaustive enumeration floors, per arm and contrast --------------------
    floors = {}
    for arm, members in (("blind_6", ["father", "mother", "son", "daughter",
                                      "brother", "sister"]),
                         ("supervised_13", list(KINSHIP))):
        idx = [KINSHIP.index(w) for w in members]
        for cname, c in CONTRASTS.items():
            raw = [c["codes"][i] for i in idx]
            _, ycodes, _ = restrict(np.zeros((len(raw), 1)), raw, c["kind"], c["exclude_zero"])
            codes = [int(v) for v in ycodes]
            if len(set(codes)) < 2:
                floors[f"{arm}.{cname}"] = dict(
                    arm=arm, contrast=cname, provenance=c["provenance"],
                    n_scored_members=len(codes), n_arrangements=1,
                    code_multiset=sorted(codes), negation_in_space=None,
                    exact_floor_p=None, can_reach_alpha_05=False,
                    floor_note="all surviving members share one code: the contrast does "
                               "not exist on this member set")
                continue
            asg, n_dist = distinct_arrangements(codes)
            rec = dict(arm=arm, contrast=cname, provenance=c["provenance"],
                       n_scored_members=len(codes), n_arrangements=int(n_dist),
                       code_multiset=sorted(codes))
            if asg is not None:
                # the negation-equivalence class of the TRUE assignment always ties
                base = np.array(codes)
                neg_present = any(np.array_equal(a, -base) for a in asg)
                rec.update(
                    negation_in_space=bool(neg_present),
                    exact_floor_p=float((2 if neg_present else 1) / len(asg)),
                    floor_note="floor = size of the negation-equivalence class / "
                               "n_arrangements; every estimator here is negation-invariant")
            else:
                rec.update(negation_in_space=None, exact_floor_p=None,
                           floor_note=f"{n_dist} arrangements exceeds the {N_ENUM_MAX} cap")
            rec["can_reach_alpha_05"] = (None if rec["exact_floor_p"] is None
                                         else bool(rec["exact_floor_p"] < ALPHA))
            floors[f"{arm}.{cname}"] = rec
    out["enumeration_floors"] = floors
    log("A2 enumeration floors: " + ", ".join(
        f"{k} floor={v['exact_floor_p']} reach05={v['can_reach_alpha_05']}"
        for k, v in floors.items()))

    # --- A3: planted-lattice synthetics through the S1/S2/R2 code paths ----------------
    # This measures ATTAINABLE POWER rather than assuming it.  A perfect lattice is pushed
    # through the identical code path at a declared SNR ladder; the best p any statistic
    # reaches over the whole ladder is its empirical floor, and a statistic whose floor
    # exceeds alpha is `untestable` at that arity -- the same logic prereg section 6 used
    # when it ruled the fifths-order naturals out on ref_p_closed = 0.895.
    syn, power = {}, {}
    LADDER = (0.25, 0.5, 1.0, 2.0, 4.0)
    for k, gen_codes, gen_kind, gender_codes in (
            (6, [1, 1, -1, -1, 0, 0], "ordinal", [1, -1, 1, -1, 1, -1]),
            (13, KINSHIP_GENERATION, "ordinal", KINSHIP_GENDER)):
        a = np.random.default_rng(1).normal(size=D_MODEL); a /= np.linalg.norm(a)
        b = np.random.default_rng(2).normal(size=D_MODEL)
        b -= (b @ a) * a; b /= np.linalg.norm(b)
        ladder = {}
        for noise in LADDER:
            r = np.random.default_rng(int(noise * 1000) + k)
            # noise scaled so each member's noise vector has norm ~= `noise`, against a
            # signal of norm ~= sqrt(gen^2 + gender^2) ~ 1-2
            M = (np.outer(gen_codes, a) + np.outer(gender_codes, b)
                 + noise * r.normal(size=(k, D_MODEL)) / math.sqrt(D_MODEL))
            tag = f"noise_{noise}"
            ax_g = axis_test(M, gen_codes, gen_kind, False,
                             seed=_seed("synth", k, tag, k, "gen"))
            ax_s = axis_test(M, gender_codes, "binary", True,
                             seed=_seed("synth", k, tag, k, "gender"))
            add = additive_test(M, gen_codes, gender_codes, seed=_seed("synth", k, tag, k, "add"))
            ladder[tag] = dict(
                generation={n: dict(observed=v["observed"], p_sampled=v["p_sampled"],
                                    p_exact=v.get("p_exact"))
                            for n, v in ax_g.items() if n != "_meta"},
                gender={n: dict(observed=v["observed"], p_sampled=v["p_sampled"],
                                p_exact=v.get("p_exact"))
                        for n, v in ax_s.items() if n != "_meta"},
                additive=dict(r2=add["r2"], p_joint_shuffle=add["p_joint_shuffle"]))
        # ---- genuine negative control: NO planted signal at all, repeated -------------
        fp = {"additive": [], "gender.s1_correlation": [], "gender.s2_loo_ridge": [],
              "generation.s1_correlation": [], "generation.s2_loo_ridge": []}
        for s in range(N_NOISE_DRAWS):
            M = np.random.default_rng(9000 + 17 * s + k).normal(size=(k, D_MODEL))
            ag = axis_test(M, gen_codes, gen_kind, False, seed=_seed("noise", k, str(s), k, "gen"))
            as_ = axis_test(M, gender_codes, "binary", True, seed=_seed("noise", k, str(s), k, "gd"))
            ad = additive_test(M, gen_codes, gender_codes, seed=_seed("noise", k, str(s), k, "add"))
            fp["additive"].append(ad["p_joint_shuffle"])
            for est in ("s1_correlation", "s2_loo_ridge"):
                fp[f"gender.{est}"].append(as_[est]["p_sampled"])
                fp[f"generation.{est}"].append(ag[est]["p_sampled"])
        ladder["unstructured_noise"] = dict(
            n_draws=N_NOISE_DRAWS,
            false_acceptance_rate_at_alpha_05={k2: float(np.mean(np.array(v) < ALPHA))
                                              for k2, v in fp.items()},
            median_p={k2: float(np.median(v)) for k2, v in fp.items()})
        syn[f"k{k}"] = ladder
        # ---- attainable-power table --------------------------------------------------
        for cname in ("generation", "gender"):
            for est in ("registered", "s1_correlation", "s2_loo_ridge"):
                best = min(ladder[f"noise_{n}"][cname][est]["p_sampled"] for n in LADDER)
                power[f"k{k}.{cname}.{est}"] = dict(
                    best_p_over_snr_ladder=float(best),
                    testable_at_alpha_05=bool(best < ALPHA),
                    false_acceptance_on_unstructured_noise=ladder["unstructured_noise"][
                        "false_acceptance_rate_at_alpha_05"].get(f"{cname}.{est}"))
        bestadd = min(ladder[f"noise_{n}"]["additive"]["p_joint_shuffle"] for n in LADDER)
        power[f"k{k}.additive_gen_x_gender"] = dict(
            best_p_over_snr_ladder=float(bestadd), testable_at_alpha_05=bool(bestadd < ALPHA),
            false_acceptance_on_unstructured_noise=ladder["unstructured_noise"][
                "false_acceptance_rate_at_alpha_05"]["additive"])
    out["planted_lattice_ladder"] = syn
    out["attainable_power"] = power
    out["attainable_power_note"] = (
        "best_p_over_snr_ladder is the smallest p a PERFECT planted lattice reaches through "
        "this exact code path.  A statistic whose best p exceeds 0.05 is untestable at that "
        "arity by construction and a null from it may not be reported as a negative.")

    # positive control: the additive model must fire on a strong planted lattice at k=13
    # AND the LOO estimator must separate both contrasts there.
    ok_pos = (power["k13.additive_gen_x_gender"]["testable_at_alpha_05"]
              and power["k13.gender.s2_loo_ridge"]["testable_at_alpha_05"]
              and power["k13.generation.s2_loo_ridge"]["testable_at_alpha_05"])
    # negative control: unstructured noise must not fire the additive model at either arity
    ok_neg = all(syn[f"k{k}"]["unstructured_noise"]["false_acceptance_rate_at_alpha_05"]
                 ["additive"] <= 0.15 for k in (6, 13))
    out["planted_lattice_controls_passed"] = bool(ok_pos and ok_neg)
    log("A3 attainable power: " + ", ".join(
        f"{key}={v['best_p_over_snr_ladder']:.4f}{'*' if v['testable_at_alpha_05'] else ''}"
        for key, v in power.items()))
    log(f"A3 unstructured-noise false-acceptance (additive): "
        f"k6 {syn['k6']['unstructured_noise']['false_acceptance_rate_at_alpha_05']['additive']:.3f}, "
        f"k13 {syn['k13']['unstructured_noise']['false_acceptance_rate_at_alpha_05']['additive']:.3f} "
        f"-> controls {'PASS' if ok_pos and ok_neg else 'FAIL'}")

    # --- A4: planted-ring closure controls through ttg.screen (the P3 code path) -------
    ring = {}
    for k in (6, 13):
        u = np.random.default_rng(11).normal(size=D_MODEL); u /= np.linalg.norm(u)
        v = np.random.default_rng(12).normal(size=D_MODEL)
        v -= (v @ u) * u; v /= np.linalg.norm(v)
        th = 2 * np.pi * np.arange(k) / k
        for tag, noise in (("planted_ring", 0.05), ("planted_line", 0.05), ("pure_noise", 1.0)):
            r = np.random.default_rng(k * 7 + len(tag))
            if tag == "planted_ring":
                P = np.outer(np.cos(th), u) + np.outer(np.sin(th), v)
            elif tag == "planted_line":
                P = np.outer(np.linspace(-1, 1, k), u)
            else:
                P = np.zeros((k, D_MODEL))
            P = P + noise * r.normal(size=(k, D_MODEL))
            s = ttg.screen(P, N_PERM_GAP, seed=0)
            ring[f"k{k}.{tag}"] = None if s is None else dict(
                max_gap_deg=s["max_gap_deg"], gap_ratio=s["gap_ratio"], p_closed=s["p_closed"])
    out["planted_ring_controls"] = ring
    out["planted_ring_controls_passed"] = bool(
        ring["k13.planted_ring"]["p_closed"] < ALPHA
        and ring["k13.planted_line"]["p_closed"] >= ALPHA)
    log(f"A4 closure controls k=13: ring p_closed={ring['k13.planted_ring']['p_closed']:.5f}, "
        f"line p_closed={ring['k13.planted_line']['p_closed']:.4f} -> "
        f"{'PASS' if out['planted_ring_controls_passed'] else 'FAIL'}")

    out["seconds"] = round(time.time() - t0, 1)
    return out


# =======================================================================================
# STAGE B -- blind arm on the FROZEN partitions
# =======================================================================================
BLIND_ARMS = ("centred_leiden", "centred_refined", "raw_leiden", "raw_refined", "hdbscan")


def reproduce_partitions(ttg, R, stored_layer, n_types, log, layer):
    """D2: rebuild every declared arm at its RECORDED settings, and verify."""
    arms, verify = {}, {}
    S_by = {}
    for base, centre in (("centred", True), ("raw", False)):
        S_by[base] = ttg.cosine_matrix(R, centre)
    for base in ("centred", "raw"):
        arm = f"{base}_leiden"
        tau = stored_layer["arms"][arm]["selected_tau"]
        S = S_by[base]
        iu = np.triu_indices(len(S), 1); sv = S[iu]; m = sv >= tau
        cl = ttg.leiden(len(S), iu[0][m], iu[1][m], sv[m])
        arms[arm] = cl
        ref, _ = ttg.refine(cl, S, tau)
        arms[f"{base}_refined"] = ref
    hcl, _ = ttg.run_hdbscan(R)
    arms["hdbscan"] = hcl
    for arm in BLIND_ARMS:
        st = stored_layer["arms"][arm]
        cl = arms[arm]
        hashes = {ttg.member_hash(c): i for i, c in enumerate(cl)}
        scr = st.get("screened") or []
        hit = sum(1 for r in scr if r["member_hash"] in hashes)
        same_id = sum(1 for r in scr if hashes.get(r["member_hash"]) == r["cluster_id"])
        num = None
        if scr:
            r0 = scr[0]
            if r0["member_hash"] in hashes:
                s = ttg.screen(R[np.asarray(cl[hashes[r0["member_hash"]]])], N_PERM_GAP, seed=0)
                num = dict(max_gap_deg_repro=s["max_gap_deg"], max_gap_deg_stored=r0["max_gap_deg"],
                           p_closed_repro=s["p_closed"], p_closed_stored=r0["p_closed"],
                           agrees=bool(abs(s["max_gap_deg"] - r0["max_gap_deg"]) < 1e-6
                                       and abs(s["p_closed"] - r0["p_closed"]) < 1e-9))
        verify[arm] = dict(n_clusters_repro=len(cl), n_clusters_stored=st.get("n_clusters"),
                           n_screened_stored=len(scr), member_hash_match=hit,
                           same_cluster_id=same_id, screen_numeric_check=num,
                           reproduced=bool(hit == len(scr) and (num is None or num["agrees"])))
    ok = all(v["reproduced"] for v in verify.values())
    log(f"  L{layer} reproduction: " + ", ".join(
        f"{a}[{verify[a]['member_hash_match']}/{verify[a]['n_screened_stored']}]" for a in BLIND_ARMS)
        + (" OK" if ok else "  *** MISMATCH ***"))
    return arms, verify, ok


def colocation_null(assign, kin_pos, pool_counts, n_draws, seed):
    """P1 null: matched random k-subsets, uniform and cache-count-decile-matched.

    The statistic is the largest number of drawn types that land in one and the same
    cluster.  Noise points (hdbscan -1) never count as together.  The kinship types are
    EXCLUDED from the candidate pool of both nulls so a draw cannot inherit the observed
    co-location.
    """
    rng = np.random.default_rng(seed)
    n_pool = len(assign)
    k = len(kin_pos)
    kin_set = set(int(x) for x in kin_pos)
    cand_all = np.array([i for i in range(n_pool) if i not in kin_set])

    def stat(idx):
        lab = assign[np.asarray(idx)]
        lab = lab[lab >= 0]
        return int(np.bincount(lab).max()) if len(lab) else 0

    obs = stat(kin_pos)
    uni = np.array([stat(rng.choice(cand_all, size=k, replace=False)) for _ in range(n_draws)])
    # decile-matched: each drawn type comes from the same count decile as its kinship twin
    edges = np.percentile(pool_counts, np.arange(0, 101, 10))
    dec = np.clip(np.digitize(pool_counts, edges[1:-1]), 0, 9)
    bins = [np.array([i for i in np.flatnonzero(dec == d) if i not in kin_set])
            for d in range(10)]
    kin_dec = dec[np.asarray(kin_pos)]
    dm = []
    for _ in range(n_draws):
        pick, used = [], set()
        for d in kin_dec:
            cand = bins[d]
            for _try in range(50):
                v = int(rng.choice(cand))
                if v not in used:
                    break
            used.add(v); pick.append(v)
        dm.append(stat(pick))
    dm = np.array(dm)
    return dict(observed_max_together=obs,
                p_uniform=float((1 + int((uni >= obs).sum())) / (1 + len(uni))),
                p_count_decile_matched=float((1 + int((dm >= obs).sum())) / (1 + len(dm))),
                null_uniform_mean=float(uni.mean()), null_uniform_max=int(uni.max()),
                null_decile_mean=float(dm.mean()), null_decile_max=int(dm.max()),
                n_draws=int(n_draws))


def stage_b(ttg, art, log, layers, cache_dir):
    t0 = time.time()
    out = dict(protocol="D2 reproduction of the frozen partitions; no re-clustering")
    from gfdisc.saes import load_sae, get_decoder

    tok, cnt, pool_ids, ext_ids = ttg.build_pool("cache/sparse_big")
    ext_pos = {int(t): i for i, t in enumerate(ext_ids)}
    pool_pos = {int(t): i for i, t in enumerate(pool_ids)}
    n_types = len(pool_ids)
    counts_pool = cnt[pool_ids].astype(float)
    from transformers import AutoTokenizer
    from gfdisc.env import MODEL_NAME
    _tk = AutoTokenizer.from_pretrained(MODEL_NAME, token=os.environ.get("HF_TOKEN"))
    decoded = [_tk.decode([int(t)]) for t in pool_ids]

    # --- which kinship terms are nodes in the frozen partitions at all? ---------------
    pc = json.loads((REPO / "results/prereg_pool_counts.json").read_text())["kinship"]
    by_word = {r["word"]: r for r in pc["rows"]}
    members, missing = [], []
    for w in KINSHIP:
        best = by_word[w]["best"]
        tid = int(best["ids"][0]) if best["n_tokens"] == 1 else None
        if tid is not None and tid in pool_pos and by_word[w]["best_count"] >= 100:
            members.append(dict(word=w, form=best["form"], token_id=tid,
                                count=by_word[w]["best_count"], pool_index=pool_pos[tid],
                                generation=KINSHIP_GENERATION[KINSHIP.index(w)],
                                lineal=KINSHIP_LINEAL[KINSHIP.index(w)],
                                gender=KINSHIP_GENDER[KINSHIP.index(w)]))
        else:
            missing.append(dict(word=w, form=best["form"], count=by_word[w]["best_count"],
                                reason="below the >=100 frozen discovery floor"))
    out["members_in_frozen_pool"] = members
    out["members_below_floor"] = missing
    kin_pos = np.array([m["pool_index"] for m in members])
    gen_codes = [m["generation"] for m in members]
    lin_codes = [m["lineal"] for m in members]
    gnd_codes = [m["gender"] for m in members]
    out["lineality_blind"] = dict(
        verdict="untestable",
        reason=f"all {len(members)} members over the floor have lineal=1 "
               f"({sorted(set(lin_codes))}); every collateral term is below the floor. "
               "Registered untestable in prereg 3.2 BEFORE any placement was examined.")
    log(f"B members over the >=100 floor: {[m['word'] for m in members]}")

    # --- the frozen ring catalogue: pure read, no computation -------------------------
    rings = art["verdict"]["other_rings"]
    kin_set = set(KINSHIP)
    ring_hits = []
    for i, r in enumerate(rings):
        m = [s for s in r["token_strings"] if s.strip().lower() in kin_set]
        if m:
            ring_hits.append(dict(index=i, layer=r["layer"], arm=r["arm"], k=r["k"],
                                  kinship_members=m, token_strings=r["token_strings"]))
    out["ring_catalogue_scan"] = dict(
        n_rings_in_catalogue=len(rings), n_rings_containing_any_kinship_term=len(ring_hits),
        hits=ring_hits,
        prediction="P3 predicts kinship is NOT ring-flagged",
        note="a pure read of verdict.other_rings in the frozen artifact; no computation")
    log(f"B ring-catalogue scan: {len(ring_hits)} of {len(rings)} flagged rings contain a "
        f"kinship term")

    per_layer = []
    all_ok = True
    for L in layers:
        tL = time.time()
        stored = {e["layer"]: e for e in art["layers"]}[L]
        rp = cache_dir / f"R_L{L}.npy"
        if rp.exists():
            R = np.load(rp)
        else:
            acc, d_sae, n_rows_used, meta = ttg.type_means("cache/sparse_big", L, tok, cnt, ext_ids)
            sae = load_sae(L, "big", token=os.environ.get("HF_TOKEN"))
            W = get_decoder(sae); del sae
            R_ext = ttg.project(acc, W)
            del acc; gc.collect()
            R = R_ext[np.asarray([ext_pos[int(t)] for t in pool_ids])]
            del R_ext; gc.collect()
            np.save(rp, R)
        arms, verify, ok = reproduce_partitions(ttg, R, stored, n_types, log, L)
        all_ok = all_ok and ok
        rec = dict(layer=L, embedding_adjacent=bool(L <= 4), reproduction=verify,
                   reproduced=ok, arms={})
        if not ok:
            per_layer.append(rec)
            log(f"  L{L} STOPPING blind arm: partition reproduction failed")
            break

        Mkin = R[kin_pos].astype(np.float64)
        # ---- P2: axis structure on the six member type means (prereg 3.2 primary) ----
        axes = {}
        for cname, c in CONTRASTS.items():
            codes = [c["codes"][KINSHIP.index(m["word"])] for m in members]
            if len(set(v for v in codes if not (c["exclude_zero"] and v == 0))) < 2:
                axes[cname] = dict(verdict="untestable", provenance=c["provenance"],
                                   reason=f"all members over the floor share code {codes[0]}")
                continue
            axes[cname] = axis_test(Mkin, codes, c["kind"], c["exclude_zero"],
                                    seed=_seed("blindaxis", L, "typemeans", len(members), cname))
            axes[cname]["provenance"] = c["provenance"]
        add = additive_test(Mkin, gen_codes, gnd_codes,
                            seed=_seed("blindadd", L, "typemeans", len(members), "gen_gender"))
        add["provenance"] = ("design uses generation (committed 3cadb69) x gender "
                             "(prereg 2026-08-21, NOT in 3cadb69), exactly as prereg 3.2 "
                             "step 4 registers it")
        d_gen, d_gnd = dir_s1(_centre_rows(Mkin), gen_codes), dir_s1(_centre_rows(Mkin), gnd_codes)
        orth = dict(abs_cos_gen_gender=cos_abs(d_gen, d_gnd),
                    note="descriptive only; prereg 3.2 step 5 reports it but does not require it",
                    estimator="S1 correlation directions (the registered estimator's "
                              "direction is not identifiable at k < dim)")
        # ---- P3: closure on the member set, both SVD conventions --------------------
        s = ttg.screen(Mkin, N_PERM_GAP, seed=0)
        xy, ratios, Vt = ttg.pca_2d(Mkin)
        xy2 = xy * np.array([1.0, -1.0])
        cx, cy, r, res = ttg.fit_circle(xy2)
        circ2 = ttg.circularity(xy2, cx, cy, r)
        bgseed = _seed("blindbg", L, "typemeans", len(members), "closure")
        bg = ttg.background_for_size(R, n_types, len(members), seed=bgseed, n_draws=N_BG_BLIND)
        closure = dict(k=len(members), max_gap_deg=s["max_gap_deg"], gap_ratio=s["gap_ratio"],
                       radius_ratio=s["radius_ratio"], top2_var=s["top2_var"],
                       p_closed=s["p_closed"],
                       p_bg=ttg.p_bg_of(bg, s["max_gap_deg"]),
                       background_n_draws=N_BG_BLIND, background_seed=int(bgseed),
                       background_mean_max_gap_deg=bg["mean_max_gap_deg"],
                       max_gap_deg_pc2_negated=float(circ2["max_gap_deg"]),
                       sign_convention_identity_holds=bool(
                           abs(circ2["max_gap_deg"] - s["max_gap_deg"]) < 1e-9),
                       flag_rule="frozen artifact rule: FLAGGED iff p_closed < 0.05 AND p_bg <= 0.05")
        closure["flagged"] = bool(closure["p_closed"] < ALPHA and closure["p_bg"] is not None
                                  and closure["p_bg"] <= ALPHA)
        closure["p3_prediction_holds"] = bool(not closure["flagged"])
        rec["type_mean_axes"] = axes
        rec["type_mean_additive"] = add
        rec["type_mean_orthogonality"] = orth
        rec["type_mean_closure_P3"] = closure

        # ---- P1: co-location, per arm -----------------------------------------------
        for arm in BLIND_ARMS:
            cl = arms[arm]
            assign = np.full(n_types, -1, dtype=np.int64)
            for ci, c in enumerate(cl):
                assign[np.asarray(c)] = ci
            lab = assign[kin_pos]
            by_cluster = {}
            for m, ci in zip(members, lab.tolist()):
                by_cluster.setdefault(ci, []).append(m["word"])
            real = {ci: ws for ci, ws in by_cluster.items() if ci >= 0}
            best_ci = sorted(real, key=lambda c: (-len(real[c]), c))[0] if real else -1
            coloc = colocation_null(assign, kin_pos, counts_pool, N_COLOC_DRAWS,
                                    seed=_seed("coloc", L, arm, len(members)))
            n_noise = len(by_cluster.get(-1, []))
            entry = dict(
                cluster_of_each_member={m["word"]: int(ci) for m, ci in zip(members, lab.tolist())},
                n_distinct_clusters=int(len({int(x) for x in lab.tolist() if x >= 0})),
                n_members_unclustered=int(n_noise),
                unclustered_members=sorted(by_cluster.get(-1, [])),
                best_cluster_id=int(best_ci),
                best_cluster_members=sorted(real.get(best_ci, [])),
                n_together=len(real.get(best_ci, [])),
                all_together=bool(len(real.get(best_ci, [])) == len(members)),
                colocation=coloc)
            if best_ci >= 0:
                c = cl[best_ci]
                mh = ttg.member_hash(c)
                srow = next((r for r in (stored["arms"][arm].get("screened") or [])
                             if r["member_hash"] == mh), None)
                entry.update(
                    best_cluster_size=len(c), best_cluster_member_hash=mh,
                    best_cluster_in_screen_band=bool(ttg.SCREEN_MIN <= len(c) <= ttg.SCREEN_MAX),
                    best_cluster_frozen_screening=None if srow is None else {
                        k: srow[k] for k in ("k", "gap_ratio", "max_gap_deg", "radius_ratio",
                                             "p_closed", "p_bg", "top2_var", "flagged",
                                             "bh_pass_within_layer_arm")},
                    best_cluster_is_a_flagged_ring=bool(srow is not None and srow["flagged"]),
                    best_cluster_token_strings_first80=[
                        decoded[i] for i in c[:80]],
                    best_cluster_n_strings_truncated=max(0, len(c) - 80))
                # ---- P2 secondary: axes restricted to the holding cluster's members --
                inc = [i for i, ci in enumerate(lab.tolist()) if ci == best_ci]
                if len(inc) >= 4:
                    Msub = R[kin_pos[np.asarray(inc)]].astype(np.float64)
                    sub_axes = {}
                    for cname, cdef in CONTRASTS.items():
                        codes = [cdef["codes"][KINSHIP.index(members[i]["word"])] for i in inc]
                        if len(set(v for v in codes
                                   if not (cdef["exclude_zero"] and v == 0))) < 2:
                            sub_axes[cname] = dict(verdict="untestable",
                                                   provenance=cdef["provenance"])
                            continue
                        sub_axes[cname] = axis_test(
                            Msub, codes, cdef["kind"], cdef["exclude_zero"],
                            seed=_seed("blindaxis_sub", L, arm, len(inc), cname))
                        sub_axes[cname]["provenance"] = cdef["provenance"]
                    entry["cluster_restricted_axes"] = sub_axes
                    entry["cluster_restricted_axes_note"] = (
                        "SECONDARY.  prereg 3.2's blind-arm sentence is ambiguous between "
                        "the member type means and the holding cluster's members; the "
                        "six-type-mean version is primary, this is the other reading.")
            rec["arms"][arm] = entry
            log(f"  L{L}[{arm}] together {entry['n_together']}/{len(members)} "
                f"p_uni={coloc['p_uniform']:.4f} p_dec={coloc['p_count_decile_matched']:.4f} "
                f"flagged_ring={entry.get('best_cluster_is_a_flagged_ring')}")
        rec["seconds"] = round(time.time() - tL, 1)
        per_layer.append(rec)
        del arms
        gc.collect()

    out["layers"] = per_layer
    out["all_layers_reproduced"] = bool(all_ok)
    out["seconds"] = round(time.time() - t0, 1)
    return out, members


# =======================================================================================
# STAGE C -- supervised 26-layer templated sweep
# =======================================================================================
def token_audit(tok, words):
    keep, drop = [], []
    for w in words:
        ids = tok.encode(" " + w, add_special_tokens=False)
        (keep if len(ids) == 1 else drop).append(
            dict(word=w, n_tokens=len(ids), ids=[int(i) for i in ids]))
    return keep, drop


def collect(model, tok, capture, torch, DEVICE, words, word_ids, layers, chunk=48):
    bos = tok.bos_token_id
    n = len(words)
    acc = {ro: {l: np.zeros((n, D_MODEL), dtype=np.float64) for l in layers} for ro in READOUTS}
    with torch.inference_mode():
        for prefix, suffix in TEMPLATES:
            pre = tok.encode(prefix, add_special_tokens=False)
            suf = tok.encode(suffix, add_special_tokens=False)
            ci = 1 + len(pre)
            pos = dict(token=ci, suffix=ci + 1, final=ci + len(suf))
            base = [bos] + pre + [0] + suf
            for lo in range(0, n, chunk):
                hi = min(lo + chunk, n)
                ids = []
                for w in word_ids[lo:hi]:
                    row = list(base); row[ci] = int(w); ids.append(row)
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


def stage_c(ttg, log, layers, chunk):
    t0 = time.time()
    import torch
    from gfdisc.model import load_model, capture
    from gfdisc.env import DEVICE
    out = {}

    model, tok = load_model(token=os.environ.get("HF_TOKEN"))
    bos = tok.bos_token_id

    keep, drop = token_audit(tok, KINSHIP)
    kin_words = [x["word"] for x in keep]
    kin_ids = [x["ids"][0] for x in keep]
    excl = set(w.lower() for w in KINSHIP_SENSE_EXCLUSIONS) | set(KINSHIP)
    noun_cand = [w for w in dict.fromkeys(NOUN_POOL) if w.lower() not in excl]
    nkeep, ndrop = token_audit(tok, noun_cand)
    out["token_audit"] = dict(kinship_single_token=[x["word"] for x in keep],
                              kinship_multi_token=drop,
                              n_nouns_kept=len(nkeep),
                              n_nouns_dropped_multi_token=len(ndrop),
                              exclusions_applied=sorted(excl - set(KINSHIP)))
    if len(kin_words) < len(KINSHIP):
        log(f"C WARNING: only {len(kin_words)}/13 kinship terms are single-token: {drop}")

    # --- D9 template verification: manual ids == natural tokenisation, every word -----
    def ok_everywhere(w, wid):
        for prefix, suffix in TEMPLATES:
            pre = tok.encode(prefix, add_special_tokens=False)
            suf = tok.encode(suffix, add_special_tokens=False)
            manual = [bos] + pre + [wid] + suf
            natural = tok.encode(prefix + " " + w + suffix, add_special_tokens=True)
            if manual != natural:
                return False, dict(word=w, prefix=prefix, manual=manual, natural=natural)
        return True, None

    tmpl_fail = []
    for w, i in zip(kin_words, kin_ids):
        good, info = ok_everywhere(w, i)
        if not good:
            tmpl_fail.append(info)
    noun_words, noun_ids, noun_drop2 = [], [], []
    for x in nkeep:
        good, info = ok_everywhere(x["word"], x["ids"][0])
        (noun_words.append(x["word"]) or noun_ids.append(x["ids"][0])) if good else \
            noun_drop2.append(info)
    # no template may contain another kinship term
    tmpl_contam = [dict(prefix=p, suffix=s,
                        found=[w for w in KINSHIP if w in (p + s).lower()])
                   for p, s in TEMPLATES if any(w in (p + s).lower() for w in KINSHIP)]
    suffix_lengths = [len(tok.encode(s, add_special_tokens=False)) for _, s in TEMPLATES]
    out["template_audit"] = dict(
        n_templates=len(TEMPLATES), suffix_token_lengths=suffix_lengths,
        min_suffix_tokens=int(min(suffix_lengths)),
        kinship_tokenisation_failures=tmpl_fail,
        n_background_words_dropped_on_tokenisation=len(noun_drop2),
        templates_containing_another_kinship_term=tmpl_contam,
        n_background_words_used=len(noun_words))
    log(f"C template audit: {len(TEMPLATES)} templates, min suffix {min(suffix_lengths)} tokens, "
        f"{len(tmpl_fail)} kinship tokenisation failures, {len(tmpl_contam)} contaminated "
        f"templates, {len(noun_words)} background nouns")
    if tmpl_fail or tmpl_contam or min(suffix_lengths) < 3:
        raise SystemExit("template audit failed; no forward pass run")

    tA = time.time()
    A = collect(model, tok, capture, torch, DEVICE, kin_words, kin_ids, layers, chunk=chunk)
    log(f"C kinship activations collected ({time.time()-tA:.1f}s)")
    tB = time.time()
    B = collect(model, tok, capture, torch, DEVICE, noun_words, noun_ids, layers, chunk=chunk)
    log(f"C background activations collected ({time.time()-tB:.1f}s)")
    del model; gc.collect()

    k = len(kin_words)
    idx_of = {w: i for i, w in enumerate(kin_words)}
    codes = {c: [CONTRASTS[c]["codes"][KINSHIP.index(w)] for w in kin_words] for c in CONTRASTS}

    cells, l0_token_plane = [], None
    for ro in READOUTS:
        for L in layers:
            M = A[ro][L]
            Mc = _centre_rows(M)
            cell = dict(readout=ro, layer=L,
                        embedding_confounded=bool(ro == "token"),
                        primary_readout=bool(ro == "suffix"))
            ax = {}
            for cname, cdef in CONTRASTS.items():
                ax[cname] = axis_test(M, codes[cname], cdef["kind"], cdef["exclude_zero"],
                                      seed=_seed("supaxis", L, ro, k, cname))
                ax[cname]["provenance"] = cdef["provenance"]
            cell["axes"] = ax
            cell["additive_gen_x_gender"] = additive_test(
                M, codes["generation"], codes["gender"],
                seed=_seed("supadd", L, ro, k, "gen_gender"))
            cell["additive_gen_x_lineal"] = additive_test(
                M, codes["generation"], codes["lineal"],
                seed=_seed("supadd", L, ro, k, "gen_lineal"))
            cell["additive_gen_x_lineal_note"] = (
                "SECONDARY.  prereg 3.2 step 4 registers gen x gender as THE lattice model; "
                "gen x lineal is the pairing named in 3cadb69's own comment and is reported "
                "for completeness, not as the registered test.")
            d = {}
            for cname, cdef in CONTRASTS.items():
                Ms, yv, _ = restrict(M, codes[cname], cdef["kind"], cdef["exclude_zero"])
                d[cname] = dir_s1(_centre_rows(Ms), yv)
            cell["orthogonality"] = dict(
                abs_cos_gen_gender=cos_abs(d["generation"], d["gender"]),
                abs_cos_gen_lineal=cos_abs(d["generation"], d["lineal"]),
                abs_cos_gender_lineal=cos_abs(d["gender"], d["lineal"]),
                estimator="S1 correlation directions on each contrast's own member set; "
                          "descriptive only (prereg 3.2 step 5 reports but does not require it)")
            # ---- P3 closure ----------------------------------------------------------
            s = ttg.screen(M, N_PERM_GAP, seed=0)
            if s is None:
                cell["closure_P3"] = dict(degenerate=True)
            else:
                xy, _, _ = ttg.pca_2d(M)
                xy2 = xy * np.array([1.0, -1.0])
                cx, cy, r, _res = ttg.fit_circle(xy2)
                c2 = ttg.circularity(xy2, cx, cy, r)
                rngb = np.random.default_rng(_seed("supbg", L, ro, k, "closure"))
                bgg, bgp = [], []
                nb = len(B[ro][L])
                for _ in range(N_BG_SUP):
                    pick = rngb.choice(nb, size=k, replace=False)
                    sb = ttg.screen(B[ro][L][pick], ttg.N_PERM_BACKGROUND, seed=0)
                    if sb is None:
                        continue
                    bgg.append(sb["max_gap_deg"]); bgp.append(sb["p_closed"])
                bgg = np.array(bgg); bgp = np.array(bgp)
                cell["closure_P3"] = dict(
                    k=k, max_gap_deg=s["max_gap_deg"], gap_ratio=s["gap_ratio"],
                    radius_ratio=s["radius_ratio"], top2_var=s["top2_var"],
                    p_closed=s["p_closed"],
                    p_bg=float((1 + int((bgg <= s["max_gap_deg"]).sum())) / (1 + len(bgg))),
                    background_n_draws=int(N_BG_SUP), background_n_scorable=int(len(bgg)),
                    background_mean_max_gap_deg=float(bgg.mean()),
                    background_false_acceptance_rate_at_alpha_05=float((bgp < ALPHA).mean()),
                    max_gap_deg_pc2_negated=float(c2["max_gap_deg"]),
                    sign_convention_identity_holds=bool(
                        abs(c2["max_gap_deg"] - s["max_gap_deg"]) < 1e-9))
                cell["closure_P3"]["closure_passes"] = bool(
                    cell["closure_P3"]["p_closed"] < ALPHA and cell["closure_P3"]["p_bg"] <= ALPHA)
                cell["closure_P3"]["p3_prediction_holds"] = bool(
                    not cell["closure_P3"]["closure_passes"])
            if ro == "token" and L == layers[0]:
                _, _, Vt = ttg.pca_2d(M)
                l0_token_plane = Vt[:2]
            if l0_token_plane is not None:
                _, _, Vt = ttg.pca_2d(M)
                cell["principal_angles_to_L0_token_plane_deg"] = principal_angles_deg(
                    Vt[:2], l0_token_plane)
            cells.append(cell)
        log(f"C readout {ro}: {len(layers)} layers done ({time.time()-t0:.1f}s elapsed)")

    out["cells"] = cells
    out["words_used"] = kin_words
    out["n_background_words"] = len(noun_words)

    # ---- D12 multiplicity ------------------------------------------------------------
    fam = {}
    for cname in CONTRASTS:
        for est in ("registered", "s1_correlation", "s2_loo_ridge"):
            key = f"axis.{cname}.{est}"
            ps = [c["axes"][cname][est]["p_sampled"] for c in cells]
            fam[key] = ps
    fam["additive.gen_x_gender"] = [c["additive_gen_x_gender"]["p_joint_shuffle"] for c in cells]
    fam["closure.p_closed"] = [c["closure_P3"].get("p_closed", 1.0) for c in cells]
    fam["closure.p_bg"] = [c["closure_P3"].get("p_bg", 1.0) for c in cells]
    mult = {}
    for key, ps in fam.items():
        keep_bh = benjamini_hochberg(ps, alpha=ALPHA)
        mult[key] = dict(n_cells=len(ps), n_raw_pass=int(sum(p < ALPHA for p in ps)),
                         n_bh_pass=int(sum(bool(x) for x in keep_bh)),
                         expected_by_chance=round(ALPHA * len(ps), 1),
                         min_p=float(min(ps)))
    out["multiplicity"] = mult
    out["seconds"] = round(time.time() - t0, 1)
    return out


# =======================================================================================
def jsonable(o):
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        v = float(o)
        return None if (math.isnan(v) or math.isinf(v)) else v
    if isinstance(o, float):
        return None if (math.isnan(o) or math.isinf(o)) else o
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.ndarray):
        return jsonable(o.tolist())
    return o


def declarations():
    return dict(
        bound_by="docs/phase2_prereg_2026-08-21.md (frozen 2026-08-21)",
        predictions_committed="src/gfdisc/ontologies.py @ 3cadb69, 2026-08-18 12:04:20 -0400",
        label_set=list(KINSHIP),
        contrasts={c: dict(codes=list(v["codes"]), kind=v["kind"],
                           provenance=v["provenance"]) for c, v in CONTRASTS.items()},
        P1="the terms co-locate in the frozen partitions more than matched random type sets",
        P2="the lattice contrasts are separable directions in the member point cloud",
        P3="closure FAILS.  A closure pass is a FAILURE of the kinship prediction and may "
           "not be re-read as a success",
        D2_reproduction_protocol=(
            "the frozen artifact does not store membership for kinship-bearing clusters, so "
            "partitions are REPRODUCED by re-running Leiden once at each recorded "
            "selected_tau with the recorded resolution and seed, plus refine and hdbscan at "
            "their recorded settings.  No sweep, no new threshold, no new arm.  Verified "
            "against stored n_clusters and every stored screened row's member_hash; on any "
            "mismatch the blind arm STOPS rather than adjusting anything"),
        D4_registered_estimator_calibration=(
            "prereg 3.2's estimator column-centres M with k < dim, so d -> M d spans exactly "
            "the zero-sum vectors and the registered ridge is ~5e-6 of the mean Gram "
            "eigenvalue: the fit interpolates the codes exactly for EVERY assignment, the "
            "score is 1.000 for observed and shuffled alike, and p is pinned at 1.0.  "
            "Asserted numerically in stage A before any data.  The registered estimator is "
            "still computed and reported verbatim, as structurally uninformative rather "
            "than as a negative"),
        D5_secondary_estimators=dict(
            provenance="NOT in the prereg and NOT in 3cadb69; added while writing this "
                       "script, BEFORE any placement was located and BEFORE any activation "
                       "was read, as the declared response to D4.  prereg section 7 permits "
                       "a post-hoc addition with its provenance stated",
            s1_correlation="d = M^T y_centred; projection G y_centred with G = M M^T",
            s2_loo_ridge="the registered ridge refitted on the k-1 other members, scoring "
                         "the held-out projection of member i"),
        D6_enumeration_floors=(
            "every estimator here is invariant under y -> -y, so the true assignment always "
            "ties with its negation in the shuffle null.  Blind gender is 3v3 on 6 members: "
            "20 arrangements, floor p = 0.10, so it CANNOT reach alpha = 0.05 even for a "
            "perfect lattice and is registered untestable by prereg 4.1(ii)'s own principle. "
            "Blind generation: 90 arrangements, floor 0.0222.  Because the blind lattice "
            "claim needs BOTH contrasts, the blind lattice claim is untestable at 0.05"),
        D7_additive_model="mu + gen*a + gender*b over members; R^2 vs a 1000-draw joint "
                          "code-permutation null.  NOT saturated: 3 design columns, k rows",
        D8_closure_runs_regardless=True,
        D9_supervised="26 layers x 3 readouts (token=embedding-confounded, suffix=PRIMARY "
                      "computed, final=tertiary) x 16 templates, family-sense prefixes, "
                      "suffix >= 3 tokens and token-identical across words, manual ids "
                      "asserted equal to the natural tokenisation",
        D10_backgrounds="the colours noun pool minus a declared kinship-sense and calendar "
                        "exclusion list; 2000 same-arity draws through the identical "
                        "templates at the identical positions.  Background draws are scored "
                        "for closure, never for ordering.  The prereg registers no "
                        "kinship-specific secondary pool, so none is invented",
        D11_both_sign_conventions="the angular gap is asserted invariant under a PC2 sign "
                                  "flip per cell, not assumed",
        D12_multiplicity=f"BH within each statistic family over 3 readouts x 26 layers = 78 "
                         f"cells; expected-by-chance {ALPHA*78:.1f} reported alongside",
        D13_seed_rule="int(md5(f'{stage}|{layer}|{arm}|{k}|{tag}').hexdigest()[:8],16) % 2**31",
        D14_verdict_vocabulary=["recovered_blind", "structure_exists_supervised_only",
                                "ordered_not_closed", "untestable", "clean_negative"],
        caveats_inherited=[
            "the blind representation is a context-AVERAGED token-type mean, closer to a "
            "static embedding than to a computed feature",
            "layers 0-4 are embedding-adjacent: a hit there is pipeline validation and "
            "near-input geometry, not a depth claim",
            "a supervised structure at the suffix readout shows the information is present "
            "one token later; attention can copy an embedding forward, so it is not a "
            "computation claim",
            "ordering is never evidence in this project; for kinship no cyclic ordering is "
            "even defined, so no ordering statistic is reported"],
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/phase2_kinship.json")
    ap.add_argument("--blind-layers", default=",".join(str(i) for i in range(11)))
    ap.add_argument("--sup-layers", default=",".join(str(i) for i in range(26)))
    ap.add_argument("--chunk", type=int, default=48)
    ap.add_argument("--cache-dir", default=None)
    ap.add_argument("--skip-blind", action="store_true")
    ap.add_argument("--skip-supervised", action="store_true")
    a = ap.parse_args()
    os.chdir(REPO)
    t0 = time.time()

    def log(msg):
        print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

    cache_dir = Path(a.cache_dir) if a.cache_dir else REPO / ".kinship_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    spec = importlib.util.spec_from_file_location(
        "ttg", REPO / "scripts/22_token_type_geometry.py")
    ttg = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ttg)

    art = json.loads((REPO / "results/token_type_geometry.json").read_text())

    res = dict(status="exploratory Phase-2 under docs/phase2_prereg_2026-08-21.md",
               item="kinship: 2-D lattice (generation x lineality x gender)",
               script="scripts/29_phase2_kinship.py",
               declared_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
               declarations=declarations(),
               frozen_inputs=dict(
                   artifact="results/token_type_geometry.json",
                   artifact_generated=art["generated"],
                   artifact_declared_at=art["declared_at"],
                   artifact_status=art["status"],
                   prereg="docs/phase2_prereg_2026-08-21.md",
                   pool_counts="results/prereg_pool_counts.json"))
    Path(a.out).write_text(json.dumps(jsonable(res), indent=1))
    log("declarations written to disk BEFORE any placement or activation was read")
    print(json.dumps(jsonable(res["declarations"]), indent=1)[:3000], flush=True)

    log("=== STAGE A: data-free calibration and synthetic controls ===")
    res["stage_a_calibration"] = stage_a(ttg, log)
    Path(a.out).write_text(json.dumps(jsonable(res), indent=1))
    if not (res["stage_a_calibration"]["planted_lattice_controls_passed"]
            and res["stage_a_calibration"]["planted_ring_controls_passed"]):
        res["abort"] = "synthetic controls failed; no data was read"
        Path(a.out).write_text(json.dumps(jsonable(res), indent=1))
        raise SystemExit("synthetic controls failed")

    members = []
    if not a.skip_blind:
        log("=== STAGE B: blind arm on the frozen partitions ===")
        res["blind_arm"], members = stage_b(
            ttg, art, log, [int(x) for x in a.blind_layers.split(",")], cache_dir)
        Path(a.out).write_text(json.dumps(jsonable(res), indent=1))

    if not a.skip_supervised:
        log("=== STAGE C: supervised 26-layer templated sweep ===")
        res["supervised_arm"] = stage_c(
            ttg, log, [int(x) for x in a.sup_layers.split(",")], a.chunk)
        Path(a.out).write_text(json.dumps(jsonable(res), indent=1))

    res["runtime_seconds"] = round(time.time() - t0, 1)
    res["generated"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    Path(a.out).write_text(json.dumps(jsonable(res), indent=1))
    log(f"DONE in {res['runtime_seconds']}s -> {a.out}")


if __name__ == "__main__":
    main()
