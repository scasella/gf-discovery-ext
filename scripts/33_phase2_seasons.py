"""PHASE 2 -- SEASONS. CONFIRMATORY track (the ring was already blind-flagged).

Registered by docs/phase2_prereg_2026-08-21.md section 3.6. This script does NOT get to
choose its own label set, predictions, tests or verdict vocabulary -- they are in that
document and are reproduced here so a reader can check that nothing drifted.

STATUS, STATED FIRST AND IN EVERY REPORT: seasons were flagged at layer 6, k = 5, in the
frozen catalogue BEFORE the prereg was written, and were summarized in-session (prereg
section 2.1). This is CONFIRMATORY, not a discovery. Everything below is a dedicated
verification of an already-exposed passer -- the passer-audit discipline applied to the
one Phase-2 ontology that came out of the blind screen.

===========================================================================
WHAT IS DECLARED BEFORE ANY RESULT IS READ (all written to disk in `declarations`
before the first activation or cache row is touched)
===========================================================================

S1  LABEL SET AND ORDER (prereg 3.6, verbatim). The 4-cycle
        spring -> summer -> autumn -> winter
    Registered surface variants per season: '{w}', ' {w}', '{W}', ' {W}'.
    'fall' with the same four variants is the DECLARED AMERICAN SYNONYM for autumn.
    PRIMARY, VERDICT-BEARING label set  : [spring, summer, autumn, winter]
    SYNONYM SENSITIVITY set (reported, NOT verdict-bearing, so the BH family stays at
    78 cells rather than silently doubling to 156): [spring, summer, fall, winter].

S2  READOUTS (prereg 4.2). 26 layers x 3 readouts, `token` / `suffix` / `final`.
      token  -- EMBEDDING-CONFOUNDED. A ring here at layers 0-2 is lexical geometry.
      suffix -- PRIMARY COMPUTED READOUT: the first token AFTER the season word.
      final  -- tertiary, exploratory.

S3  CLOSURE IS THE DISCRIMINATOR. Ordering is never evidence on its own (project tenet
    1: random subsets order at ~0.16 and an ordinal control out-orders real cycles).
    At k = 4 this is doubly true -- see S5.

S4  NULLS.
      (a) uniform random angles, k of them, 20,000 draws;
      (b) 2,000 draws of k matched same-arity common nouns through the IDENTICAL
          templates at the IDENTICAL positions -- THE NULL THAT MATTERS, reported as a
          false-acceptance rate at alpha = 0.05 and as an empirical p;
      (c) secondary class-matched pool: singular count nouns naming a BOUNDED TIME
          PERIOD OR EVENT (the grammatical and semantic class a season sits in),
          2,000 draws;
      (d) ordering: 20,000 shuffled-label permutations under the same
          max-over-symmetries selection, plus an explicit 1,000-shuffle control.
      Background draws are NOT scored for ordering (random words have no true cyclic
      order), exactly as scripts/24_colors_existence.py declared.

S5  THE k = 4 POWER CAVEAT, COMPUTED RATHER THAN ASSUMED (prereg 3.6.3 demands this).
    Two separate ceilings, both computed in-script before any data is read:

    (i)  ORDERING IS STRUCTURALLY FLOORED AT k = 4. `cyclic_order_score` maximises over
         the 2K = 8 dihedral symmetries of a 4-cycle, and there are only 4! = 24 label
         assignments, so a perfect ring recovers rho = 1 while random labels do too with
         probability (number of distinct symmetry images)/24. The exact floor is computed
         by ENUMERATION (not Monte Carlo) in `exact_order_null`. Consequence, declared
         now: AT k = 4 THE ORDERING p-VALUE CANNOT GATE ANY VERDICT. It is reported with
         its exact floor next to it.
    (ii) The properly powered ordering statistic at k = 4 is therefore declared to be a
         BINARY one: does the recovered angular sequence equal the registered season
         cycle up to rotation/reflection (i.e. rho = 1)? Its chance probability p0 is the
         exact enumerated number from (i). Across the sweep the count of correct-order
         cells is compared to Binomial(n_cells, p0). Declared caveat: cells at adjacent
         layers are correlated, so this is a reference point, not an independent test.
    (iii) CLOSURE at k = 4 is nominally testable (prereg section 6: a perfect 4-ring gives
         ref_p_closed = 5.0e-05) but four points are the weakest ring test available. The
         uniform-angle gap threshold that achieves p = 0.05 at k = 4 is computed and
         reported, so a "closure passed" line can be read against how permissive the test
         is. The DISCRIMINATING number is the matched same-arity background, S4(b).

S6  PLANTED-RING SYNTHETICS through this script's own code path, at k = 4, before any
    activation is read: a planted 4-ring at an SNR ladder (must close and must recover
    rho = 1), a planted 200-degree arc and a planted 1-D line (must NOT close), and an
    isotropic-noise battery whose false-acceptance rate is measured. The ordering
    criterion for the planted ring is rho == 1 EXACTLY, not p_order < 0.05, because S5(i)
    makes the latter unreachable at k = 4. Failing the battery aborts the run.

S7  BOTH SVD SIGN CONVENTIONS reported for every angle statistic. The
    max-over-symmetries rho is reflection-invariant by construction; that identity is
    ASSERTED, not assumed, and the rotation-only rho is reported per convention as the
    quantity a sign flip does change.

S8  MULTIPLICITY. Benjamini-Hochberg within the primary sweep (3 readouts x 26 layers =
    78 cells), reported alongside raw counts, the expected-by-chance count (~3.9 at
    alpha = 0.05) and each p-value's floor. An isolated single-cell pass is not a
    finding.

S9  NO WIDENED PLANE SEARCH IS RUN. scripts/24_colors_existence.py added one (its D10)
    and stated its provenance. The Phase-2 prereg registers no such search for seasons,
    and section 7 forbids adding a further search family. The primary top-2 PCA plane is
    the only plane scored. This omission is registered so its absence is visible.

S10 BACKGROUND POOL EXCLUSION RULE, declared before the pools are used: no background
    word may carry an established seasonal, weather, or calendar sense (frost, snow,
    storm, thunder, cloud, breeze, harvest, holiday, solstice, equinox, monsoon, ...),
    and no background word may be a member of another cyclic ontology this project has
    tested or could inherit (month and weekday names; quarter; morning/afternoon/evening/
    night; the compass directions). Pools are audited and any multi-token word is DROPPED
    by a mechanical, label-blind rule; the run aborts if a pool falls under its floor.

S11 TEMPLATE CONSTRAINT, adopted from the keys-specific rule in prereg 3.1 because
    ' fall' has exactly the same problem the note letters have: every template's PREFIX
    must establish the temporal / season sense BEFORE the season token, since the suffix
    readout sits one token later and can only inherit disambiguation from the left. No
    template contains another member of the label set. Every suffix is >= 3 tokens so
    `suffix` and `final` are distinct positions, and is token-identical across the words.

===========================================================================
BLIND ARM -- reads ONLY the frozen artifact. No re-clustering.
===========================================================================

B1  TARGET. results/token_type_geometry.json -> verdict.other_rings entry with
    layer 6, arm raw_leiden, cluster_id 75, member_hash aac1a4d107c9, k = 5,
    members [' summer', ' spring', ' winter', ' Spring', ' Summer'].
    NOTE, recorded before verification: raw_leiden is the SECONDARY clustering arm
    (centred_leiden is declared primary a priori), the row's bh_pass is FALSE, and the
    k = 5 matched background false-accepts the uniform closure test at a rate of 0.08.
    All three travel with every report of this ring.

B2  REPRODUCTION WITHOUT RE-CLUSTERING. The cluster is identified by its five stored
    token strings -> token ids -> positions in the frozen >= 100 pool -> md5, asserted
    equal to the stored member_hash. Screening is then recomputed from freshly built
    per-token-type means using the IMPORTED functions of
    scripts/22_token_type_geometry.py (`screen`, `background_for_size`, `p_bg_of`,
    `gap_p`), at the frozen seeds: uniform gap null seed 0 at 20,000 permutations,
    background seed 10000*layer + k = 60005 at 500 draws / 2,000 permutations. Monte-
    Carlo range is reported over seeds 1..20, as results/ref_table_k7.json's
    k12_validation block did.

B3  LEIDEN REPRODUCTION IS SEPARATE AND GATES ONLY THE PLACEMENT QUESTION. The two
    Leiden arms are re-run at the FROZEN selected tau read out of the artifact, with the
    frozen resolution 1.0 and seed 0 -- a reproduction, not a re-selection: no new
    threshold, no new resolution, no new arm, no sweep. If the reproduced partition does
    not hash-match, the reproduction status is reported honestly and B2's verification
    still stands, because B2 never needed Leiden.

B4  THE VARIANT RULE, INHERITED VERBATIM from the frozen artifact: "A label counts as
    PRESENT in a cluster if any of its qualifying token-type variants is a member; when
    several variants are present the highest-cache-count variant is the one scored.
    Counts are NOT summed across variants." Applied to this cluster it is decisive and
    is reported as the headline structural fact.

B5  THE MISSING-LABEL QUESTION (prereg 3.6.2). ' autumn' best-variant count is 30, below
    the >= 100 discovery floor, so autumn IS NOT A NODE IN THE FROZEN PARTITIONS AT ALL:
    a pool-floor exclusion, not a geometric absence. ' fall' at 395 IS in the pool, so
    its actual frozen cluster is located and reported, at every layer 0-10 and in both
    Leiden arms. Registered in advance (prereg 3.6.2): fall's absence from the seasons
    cluster is the EXPECTED consequence of verb polysemy and is not evidence against a
    seasons ring; its presence would be a stronger result than the model needs.

B6  DUPLICATE-PAIR ANGULAR ADJACENCY. Where a season contributes two case variants, the
    angular separation of that pair in the cluster's own PCA plane is reported against
    the mean nearest-neighbour separation. This is the single most discriminating
    descriptive fact available for whether the object is a seasons ring or "case variants
    co-locate, three points spread".

B7  LABEL-SELECTED 4-POINT SETS, marked SUPERVISED-ADJACENT, never a blind ring claim.
    A frozen cluster is a label-blind object; a hand-picked 4-set is not. Two are scored
    through the identical screening + matched-background pipeline and labelled as such:
    {spring, summer, winter, fall} from the >= 100 pool, and
    {spring, summer, winter, autumn} which requires the >= 30 EXTENDED pool for autumn --
    permitted by the artifact's own declared_rules.extended_pool_use, which reserves the
    floor-30 pool for "the supervised sensitivity arm", exactly what this is.

B8  VERDICT MAPPING, DECLARED BEFORE RESULTS EXIST, because two registered rules are
    unreachable as literally written and any post-hoc choice would look fitted:
      - prereg 3.6's `recovered_blind (confirmatory)` requires "ordering beating the
        shuffle control". At k <= 4 that is structurally impossible (S5(i)). The ordering
        clause is therefore satisfied by the BINARY criterion S5(ii) -- recovered
        sequence equals the registered cycle up to dihedral symmetry -- with its exact
        chance probability reported, and never by a p-value.
      - the deduped season set in the flagged cluster is smaller than the frozen
        screening band minimum of 5, so it cannot be screened as a cluster in its own
        right. The blind verdict is therefore computed twice and both are reported:
        REPRODUCTION verdict on the frozen k = 5 cluster exactly as screened, and
        INTERPRETATION verdict on the deduped distinct-season set.
      - vocabulary is the registered one and nothing else:
        recovered_blind (confirmatory) | ordered_not_closed | clean_negative
        (underpowered, k=4) | untestable | structure_exists_supervised_only.

CAVEATS THAT SURVIVE ANY POSITIVE (prereg 4.5, inherited in force): the blind
representation is a context-averaged token-type mean over every corpus occurrence --
closer to a static embedding than to a computed feature. Layers 0-4 are
embedding-adjacent. In the supervised arm a closed ring at the suffix readout shows the
information is present one token later, not that the model recomputed it; the only handle
is the principal angle between the suffix plane and the layer-0 token plane, reported per
layer.
"""
import argparse, gc, hashlib, importlib.util, itertools, json, os, sys, time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from gfdisc.env import DEVICE, D_MODEL
from gfdisc.fits import (degenerate, pca_2d, fit_circle, circularity, is_ring, angles,
                         cyclic_order_score, _rank_of)
from gfdisc.nulls import benjamini_hochberg

STATUS = "exploratory Phase-2 under docs/phase2_prereg_2026-08-21.md"


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, str(REPO / path))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


# ---------------------------------------------------------------------------
# S1 -- label sets
# ---------------------------------------------------------------------------
SEASONS = ["spring", "summer", "autumn", "winter"]          # PRIMARY, verdict-bearing
SEASONS_FALL = ["spring", "summer", "fall", "winter"]       # synonym sensitivity only
FORMS = ("{w}", " {w}", "{W}", " {W}")
FULL_K = 4

# ---------------------------------------------------------------------------
# S11 -- templates. (prefix, suffix); the season goes between them with one space.
# Every prefix establishes the temporal sense before the season token. No template
# contains a member of the label set.
# ---------------------------------------------------------------------------
TEMPLATES = [
    ("They visited us last", " and stayed for several weeks"),
    ("The festival is held every", " without fail in this town"),
    ("She was born in the", " of that year"),
    ("Prices always rise during", " across the whole region"),
    ("The school term begins in", " each academic year"),
    ("We travelled there the following", " with our friends"),
    ("The season that comes next is", " according to the calendar"),
    ("Sales are recorded each", " by the finance team"),
    ("The report covers the whole of", " in three chapters"),
    ("He remembers that particular", " very clearly indeed"),
    ("Tourists arrive throughout", " from many countries"),
    ("The contract was signed in", " of the same year"),
    ("Their busiest period is", " every single year"),
    ("The photograph was taken one", " near the old house"),
    ("Deliveries are slower in", " than at other times"),
    ("The council meets once each", " to review the budget"),
]
READOUTS = ("token", "suffix", "final")

# ---------------------------------------------------------------------------
# S10 -- background pools and the exclusion rule, written to the JSON verbatim
# ---------------------------------------------------------------------------
SEASONAL_SENSE_EXCLUSIONS = [
    # seasonal / weather sense
    "frost", "snow", "storm", "thunder", "cloud", "breeze", "rain", "sun", "sunshine",
    "heat", "chill", "blossom", "foliage", "leaf", "leaves", "harvest", "holiday",
    "solstice", "equinox", "monsoon", "hurricane", "blizzard", "drought", "flood",
    "beach", "ski", "sleigh", "thaw", "bloom", "crop", "sowing", "vacation",
    # calendar words and members of rival cyclic ontologies
    "january", "february", "march", "april", "may", "june", "july", "august",
    "september", "october", "november", "december", "monday", "tuesday", "wednesday",
    "thursday", "friday", "saturday", "sunday", "weekend", "quarter", "morning",
    "afternoon", "evening", "night", "dawn", "dusk", "noon", "midnight",
    "north", "south", "east", "west", "year", "month", "week", "day", "season",
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
    "meadow", "valley", "canyon", "desert", "island", "harbour", "cliff",
    "cave", "stream", "pond", "lake", "ocean", "tunnel", "fence", "gate", "roof",
    "floor", "ceiling", "stair", "corner", "shelf", "drawer", "cupboard", "bucket",
    "shovel", "wheel", "handle", "pipe", "wire", "nail", "screw", "board", "plank",
    "brick", "stone", "gravel",
]
# S4(c) -- singular count nouns naming a bounded time period or event, none of them a
# member of a cyclic ontology and none carrying a seasonal or weather sense.
PERIOD_POOL = [
    "semester", "election", "ceremony", "wedding", "funeral", "concert", "lecture",
    "meeting", "interview", "journey", "voyage", "expedition", "campaign", "recession",
    "strike", "protest", "trial", "hearing", "audit", "inspection", "rehearsal",
    "performance", "exhibition", "conference", "reunion", "banquet", "parade",
    "auction", "tournament", "marathon", "workshop", "seminar", "session", "shift",
    "internship", "apprenticeship", "sabbatical", "retirement", "childhood",
    "pregnancy", "honeymoon", "lifetime", "epoch", "era", "decade", "century",
    "phase", "stage", "episode", "chapter", "interval", "pause", "delay", "sprint",
    "rally", "summit", "hackathon", "residency", "tenure", "posting", "deployment",
]

# ---------------------------------------------------------------------------
# RECOVERY AMENDMENT, provenance stated plainly, exactly as
# scripts/24_colors_existence.py did for its own recovery pass. The first launch of this
# script ABORTED inside the supervised template audit -- BEFORE the model was asked for a
# single activation, and after the blind arm had already been written to disk -- because
# template 2's suffix " without fail" is 2 tokens and S11 requires >= 3 so that the
# `suffix` and `final` readouts are distinct positions. The guard did its job. The suffix
# was lengthened to " without fail in this town" (5 tokens) and the run restarted from
# scratch. Nothing else was touched: no label set, no readout, no null, no verdict rule,
# no blind-arm number. The blind arm is deterministic and reproduces bit-identically.
# ---------------------------------------------------------------------------
RECOVERY_AMENDMENTS = [
    "TEMPLATE 2 SUFFIX ' without fail' (2 tokens) -> ' without fail in this town' "
    "(5 tokens). The declared S11 rule requires suffix >= 3 tokens; the in-script audit "
    "raised SystemExit before any activation was read, on the first launch. Changed and "
    "re-run in full. No label set, readout, null, background pool or verdict rule was "
    "altered, and the blind arm (already written) is deterministic and unchanged.",
]

N_BACKGROUND = 2000
N_PERM_ORDER = 20000
N_SHUFFLE_CONTROL = 1000
N_GAP_NULL = 20000

# ---------------------------------------------------------------------------
# B1 -- the frozen target, hardcoded so the run cannot silently verify a different row
# ---------------------------------------------------------------------------
TARGET = dict(
    layer=6, arm="raw_leiden", cluster_id=75, member_hash="aac1a4d107c9", k=5,
    token_strings=[" summer", " spring", " winter", " Spring", " Summer"],
    stored=dict(gap_ratio=1.3565005915458732, max_gap_deg=97.66804259130288,
                p_closed=0.014499275036248187, p_bg=0.021956087824351298,
                top2_var=0.8276460216402357, radius_ratio=1.029864823186863,
                background_rate_p_closed_lt_05=0.08, bh_pass_within_layer_arm=False))


# ===========================================================================
# S5 -- power, computed by enumeration and by table, before any data is read
# ===========================================================================
_TABLES = {}


def gap_table(k, n_perm=N_GAP_NULL, seed=0):
    if k not in _TABLES:
        rng = np.random.default_rng(seed)
        a = np.sort(rng.uniform(0, 2 * np.pi, size=(n_perm, k)), axis=1)
        g = np.diff(np.concatenate([a, a[:, :1] + 2 * np.pi], axis=1), axis=1).max(1)
        _TABLES[k] = np.sort(g)
    return _TABLES[k]


def p_closed(k, observed_max_gap_rad):
    t = gap_table(k)
    return float((1 + int(np.searchsorted(t, observed_max_gap_rad, side="right"))) / (1 + len(t)))


def gap_threshold_for_alpha(k, alpha=0.05):
    """The max-gap (degrees) at which the uniform-angle closure test hits `alpha`."""
    t = gap_table(k)
    i = int(np.floor(alpha * (len(t) + 1)) - 1)
    i = max(0, min(len(t) - 1, i))
    return float(np.degrees(t[i]))


def sym_rows(label_index, K, flips=(1, -1)):
    lab = np.asarray(label_index, dtype=int)
    return np.vstack([_rank_of((f * lab + s) % K) for f in flips for s in range(K)])


def exact_order_null(label_index, K, flips=(1, -1), n_perm_reported=N_PERM_ORDER):
    """EXACT null of the max-over-symmetries rho, by enumerating all n! assignments.

    Not Monte Carlo. At k = 4 there are 24 assignments and 8 dihedral symmetries, so the
    ceiling this exposes is a structural property of the statistic, not a sampling
    accident.
    """
    n = len(label_index)
    base = sym_rows(label_index, K, flips)
    denom = n * (n * n - 1)
    ref = np.arange(n, dtype=np.float64)          # any fixed angular rank vector will do
    rhos = []
    for perm in itertools.permutations(range(n)):
        d = ref - base[:, np.asarray(perm)]
        rhos.append(float((1.0 - 6.0 * (d * d).sum(-1) / denom).max()))
    rhos = np.asarray(rhos)
    rho_max = float(rhos.max())
    p_at_max = float((rhos >= rho_max - 1e-12).mean())
    return dict(
        n_points=int(n), full_k=int(K), n_assignments=int(len(rhos)),
        n_distinct_symmetry_images=int(len(np.unique(base, axis=0))),
        rho_max_achievable=rho_max,
        exact_chance_prob_of_rho_max=p_at_max,
        min_attainable_p_order=float((1 + n_perm_reported * p_at_max) / (1 + n_perm_reported)),
        null_mean=float(rhos.mean()), null_p95=float(np.percentile(rhos, 95)),
        note=("the max-over-symmetries ordering p-value cannot fall below "
              "min_attainable_p_order; at this k it therefore cannot gate a verdict"))


# ===========================================================================
# The one analysis routine. Seasons, backgrounds and synthetics all go through it.
# ===========================================================================
def order_stats(ang, label_index, K, flips=(1, -1), n_perm=N_PERM_ORDER, seed=0, block=5000):
    """Max Spearman over the allowed symmetries, calibrated by shuffled labels.

    Same construction as fits.cyclic_order_score (asserted against it in
    `check_against_reference`), vectorised, with `flips` exposed so the rotation-only
    handedness-sensitive statistic can be read off separately (S7).
    """
    ang = np.asarray(ang, dtype=float)
    lab = np.asarray(label_index, dtype=int)
    n = len(lab)
    jitter = np.random.default_rng(seed).permutation(n)
    order = np.lexsort((jitter, ang))
    rank_of = np.empty(n, dtype=np.float64)
    rank_of[order] = np.arange(n)
    syms = [(int(f), int(s)) for f in flips for s in range(K)]
    base = np.vstack([_rank_of((f * lab + s) % K) for f, s in syms])
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
                flips=[int(f) for f in flips],
                recovered_sequence_matches_registered_cycle=bool(rho >= 1.0 - 1e-12))


def check_against_reference(seed=0):
    """order_stats(flips=(1,-1)) must equal fits.cyclic_order_score on the same input."""
    rng = np.random.default_rng(seed)
    out = []
    for t in range(5):
        k = 4
        ang = rng.uniform(-np.pi, np.pi, size=k)
        lab = np.arange(k)
        a = cyclic_order_score(ang, lab, k, n_perm=2000, seed=t)
        b = order_stats(ang, lab, k, n_perm=2000, seed=t)
        out.append(dict(trial=t, rho_ref=a["rho"], rho_new=b["rho"],
                        sym_ref=[int(x) for x in a["symmetry"]], sym_new=b["symmetry"],
                        rho_match=bool(abs(a["rho"] - b["rho"]) < 1e-12)))
    return dict(trials=out, passed=bool(all(x["rho_match"] for x in out)))


def analyse(P, label_index=None, K=None, seed=0, do_order=True, both_conventions=True):
    P = np.asarray(P, dtype=np.float64)
    k = len(P)
    deg = bool(degenerate(P))
    xy, ratios, Vt = pca_2d(P)
    out = dict(k=k, degenerate=deg, top2_var=float(ratios[:2].sum()),
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
            d["angles_deg"] = [float(np.degrees(x)) for x in ang]
        convs[name] = d
    out["conventions"] = convs
    return out, Vt[:2], xy


def fast_primary(P):
    """(gap_ratio, p_closed_uniform, radius_ratio) on the PRIMARY top-2 PCA plane only.

    S9: no plane is searched over, so a background draw is given exactly the freedom the
    observation was given -- none.
    """
    X = np.asarray(P, dtype=np.float64)
    X = X - X.mean(0, keepdims=True)
    _, S, Vt = np.linalg.svd(X, full_matrices=False)
    z = X @ Vt[:2].T
    cx, cy, r, _ = fit_circle(z)
    c = circularity(z, cx, cy, r)
    return (c["gap_ratio"], p_closed(len(X), np.radians(c["max_gap_deg"])),
            c["radius_ratio"])


# ===========================================================================
# S6 -- planted synthetics at k = 4, through the routines above
# ===========================================================================
def synthetics(d=D_MODEL, k=FULL_K, n_bg=500, seed=0):
    rng = np.random.default_rng(seed)
    lab = np.arange(k)
    Q = np.linalg.qr(rng.normal(size=(d, 3)))[0]
    th = 2 * np.pi * np.arange(k) / k

    def ring(noise):
        base = np.column_stack([np.cos(th), np.sin(th)]) @ Q[:, :2].T
        return base + noise * rng.normal(size=(k, d)) / np.sqrt(d)

    def arc(span_deg, noise=0.05):
        a = np.radians(np.linspace(0, span_deg, k))
        base = np.column_stack([np.cos(a), np.sin(a)]) @ Q[:, :2].T
        return base + noise * rng.normal(size=(k, d)) / np.sqrt(d)

    def line(noise=0.05):
        t = np.linspace(-1, 1, k)[:, None]
        return t * Q[:, 0][None, :] + noise * rng.normal(size=(k, d)) / np.sqrt(d)

    bgP = [rng.normal(size=(k, d)) for _ in range(n_bg)]
    bg_gr = np.array([fast_primary(P)[0] for P in bgP])
    bg_pc = np.array([fast_primary(P)[1] for P in bgP])

    def row(P, seed_):
        r, _, _ = analyse(P, lab, k, seed=seed_, do_order=True)
        A = r["conventions"]["A"]
        return dict(gap_ratio=A["gap_ratio"], max_gap_deg=A["max_gap_deg"],
                    p_closed=A["p_closed_uniform"],
                    p_vs_background=float((1 + int((bg_gr <= A["gap_ratio"]).sum()))
                                          / (1 + len(bg_gr))),
                    rho=A["order"]["rho"], p_order=A["order"]["p"],
                    rho_is_one=bool(A["order"]["rho"] >= 1.0 - 1e-12))

    out = dict(k=int(k), d=int(d), n_background_draws=int(n_bg),
               snr_ladder={f"noise_{n}": row(ring(n), 1)
                           for n in (0.0, 0.15, 0.35, 0.6, 1.0, 2.0)},
               planted_arc_200deg=row(arc(200), 2), planted_line=row(line(), 3),
               isotropic_noise=dict(
                   n_draws=int(n_bg), gap_ratio_mean=float(bg_gr.mean()),
                   uniform_null_false_acceptance_at_p05=float((bg_pc < 0.05).mean())))
    clean = out["snr_ladder"]["noise_0.0"]
    out["criteria"] = ("planted clean 4-ring must close against BOTH nulls and recover "
                       "rho == 1 exactly (S6: p_order < 0.05 is unreachable at k = 4, so "
                       "it is NOT a criterion); planted 200-degree arc and planted line "
                       "must NOT close against the uniform null")
    out["passed"] = bool(clean["p_closed"] < 0.05 and clean["p_vs_background"] < 0.05
                         and clean["rho_is_one"]
                         and out["planted_arc_200deg"]["p_closed"] >= 0.05
                         and out["planted_line"]["p_closed"] >= 0.05)
    return out


# ===========================================================================
# BLIND ARM
# ===========================================================================
def season_variant_ids(tokenizer, cnt):
    """Every registered surface variant of every season word, with its cache count."""
    out = {}
    for w in ["spring", "summer", "autumn", "winter", "fall"]:
        vs = []
        for f in FORMS:
            s = f.format(w=w, W=w.capitalize())
            e = tokenizer.encode(s, add_special_tokens=False)
            vs.append(dict(form=f, surface=s, n_tokens=len(e),
                           token_id=int(e[0]) if len(e) == 1 else None,
                           count=int(cnt[e[0]]) if len(e) == 1 and e[0] < len(cnt) else 0))
        vs.sort(key=lambda x: -x["count"])
        out[w] = vs
    return out


def blind_arm(args, out, write):
    m22 = _load("scripts/22_token_type_geometry.py", "m22")
    art = json.loads((REPO / "results/token_type_geometry.json").read_text())
    from transformers import AutoTokenizer
    from gfdisc.env import MODEL_NAME
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=os.environ.get("HF_TOKEN"))

    root = args.root
    tok, cnt, pool_ids, ext_ids = m22.build_pool(root)
    n_types = len(pool_ids)
    pool_set = set(int(x) for x in pool_ids)
    ext_set = set(int(x) for x in ext_ids)
    decoded_pool = [tokenizer.decode([int(t)]) for t in pool_ids]
    pos_of_string = {}
    for i, s in enumerate(decoded_pool):
        pos_of_string.setdefault(s, []).append(i)
    pos_of_id = {int(t): i for i, t in enumerate(pool_ids)}

    variants = season_variant_ids(tokenizer, cnt)
    B = dict(target=TARGET, n_pool_types=int(n_types), variants=variants)

    # ---- B1: locate the frozen row and assert it is the registered one ---------
    row = None
    for r in art["verdict"]["other_rings"]:
        if (r["layer"] == TARGET["layer"] and r["arm"] == TARGET["arm"]
                and r["k"] == TARGET["k"] and r.get("token_strings") == TARGET["token_strings"]):
            row = r
    assert row is not None, "the registered seasons row is not in verdict.other_rings"
    scr_rows = art["layers"][[i for i, L in enumerate(art["layers"])
                              if L["layer"] == TARGET["layer"]][0]]["arms"][TARGET["arm"]]["screened"]
    frozen = [s for s in scr_rows if s["cluster_id"] == TARGET["cluster_id"]]
    assert len(frozen) == 1 and frozen[0]["member_hash"] == TARGET["member_hash"], \
        "cluster_id / member_hash mismatch against the frozen artifact"
    frozen = frozen[0]
    B["frozen_row"] = frozen
    B["frozen_catalogue_row"] = row
    B["honest_framing"] = dict(
        arm_is_secondary=True,
        primary_arm="centred_leiden (declared primary a priori in the frozen artifact)",
        bh_pass_within_layer_arm=bool(frozen["bh_pass_within_layer_arm"]),
        background_false_acceptance_rate_at_k5=float(frozen["background_rate_p_closed_lt_05"]),
        n_flagged_total=int(art["verdict"]["n_flagged_total"]),
        n_screened_total=int(art["verdict"]["n_screened_total"]),
        n_other_rings=len(art["verdict"]["other_rings"]),
        n_other_rings_containing_a_season_word=int(sum(
            1 for r2 in art["verdict"]["other_rings"]
            if any(t.strip().lower() in ("spring", "summer", "autumn", "winter", "fall")
                   for t in r2.get("token_strings", [])))))

    # ---- B2: member positions -> md5, without Leiden ---------------------------
    member_pos = []
    for s in TARGET["token_strings"]:
        hits = pos_of_string.get(s, [])
        assert len(hits) == 1, f"token string {s!r} maps to {len(hits)} pool positions"
        member_pos.append(hits[0])
    member_pos = sorted(member_pos)
    rebuilt = m22.member_hash(member_pos)
    B["member_hash_reproduced"] = dict(
        computed=rebuilt, stored=TARGET["member_hash"], match=bool(rebuilt == TARGET["member_hash"]),
        member_positions=[int(x) for x in member_pos],
        member_token_ids=[int(pool_ids[i]) for i in member_pos],
        method="token strings -> ids -> positions in the frozen >=100 pool -> md5; no clustering")

    # ---- B4: the variant rule applied ------------------------------------------
    def season_of(s):
        w = s.strip().lower()
        return w if w in ("spring", "summer", "autumn", "winter", "fall") else None

    per_label = {}
    for i, s in zip(member_pos, [decoded_pool[i] for i in member_pos]):
        w = season_of(s)
        per_label.setdefault(w, []).append(dict(surface=s, pos=int(i),
                                                token_id=int(pool_ids[i]),
                                                count=int(cnt[pool_ids[i]])))
    for w in per_label:
        per_label[w].sort(key=lambda x: -x["count"])
    B["variant_rule_applied"] = dict(
        rule=("inherited verbatim: a label counts as PRESENT if any qualifying variant is a "
              "member; the highest-cache-count variant is the one scored; counts are NOT "
              "summed across variants"),
        cluster_size_k=len(member_pos),
        distinct_labels_present=sorted([w for w in per_label if w]),
        n_distinct_labels_present=len([w for w in per_label if w]),
        n_non_season_members=int(sum(1 for w in per_label if w is None)),
        per_label=per_label,
        prereg_expectation=("prereg 3.6.2 wrote: 'the flagged cluster's k = 5 means at least "
                            "two of its five members are not season words'"),
        prereg_expectation_correct=bool(sum(len(v) for w, v in per_label.items() if w is None) >= 2),
        headline=("the k = 5 cluster carries only "
                  f"{len([w for w in per_label if w])} DISTINCT seasons; the remaining "
                  "members are case variants of seasons already present"))

    # ---- build the layer-6 means (restricted accumulator, identical per-type values) ----
    season_ids_ext = sorted({v["token_id"] for w in variants for v in variants[w]
                             if v["token_id"] is not None and v["token_id"] in ext_set})
    sel_ids = np.array(sorted(pool_set | set(season_ids_ext)), dtype=np.int64)
    sel_pos = {int(t): i for i, t in enumerate(sel_ids)}
    pool_rows = np.array([sel_pos[int(t)] for t in pool_ids])
    B["accumulator_note"] = (
        "per-token-type means are built over the >=100 pool plus the season variants that "
        "clear the >=30 extended floor, instead of the full 10,808-type extended pool. A "
        "type's mean is a groupby over its own cache rows, so restricting the type set "
        "changes no retained type's value; it only cuts peak RSS from ~3.1 GB to ~1.1 GB.")

    layers = [int(x) for x in args.blind_layers.split(",")]
    layers = [TARGET["layer"]] + [l for l in layers if l != TARGET["layer"]]
    tau_of = {}
    for L in art["layers"]:
        for arm in ("centred_leiden", "raw_leiden"):
            tau_of[(L["layer"], arm)] = L["arms"][arm]["selected_tau"]

    placement, verification = [], None
    for L in layers:
        tL = time.time()
        acc, d_sae, n_rows_used, meta = m22.type_means(root, L, tok, cnt, sel_ids)
        sae = m22.load_sae(L, args.l0, token=os.environ.get("HF_TOKEN"))
        W = m22.get_decoder(sae); del sae
        R_sel = m22.project(acc, W)
        del acc; gc.collect()
        R = R_sel[pool_rows]
        print(f"  [blind L{L}] means {R.shape} from {n_rows_used} rows "
              f"({time.time() - tL:.1f}s)", flush=True)

        if L == TARGET["layer"]:
            verification = verify_target(m22, R, R_sel, sel_pos, cnt, pool_ids,
                                         member_pos, per_label, variants, ext_set,
                                         n_types, decoded_pool)
            B["verification"] = verification
            out["blind_arm"] = B
            write(out)

        # ---- B3: Leiden reproduction at the FROZEN tau, both arms ---------------
        for arm, centre in (("centred_leiden", True), ("raw_leiden", False)):
            tau = tau_of[(L, arm)]
            if tau is None:
                continue
            S = m22.cosine_matrix(R, centre)
            iu = np.triu_indices(len(S), 1)
            sv = S[iu]
            m = sv >= tau
            clusters = m22.leiden(len(S), iu[0][m], iu[1][m], sv[m])
            del S, sv, m; gc.collect()
            cl_of = {}
            for ci, c in enumerate(clusters):
                for x in c:
                    cl_of[int(x)] = ci
            rep = None
            if L == TARGET["layer"] and arm == TARGET["arm"]:
                cid = cl_of.get(member_pos[0])
                got = sorted(int(x) for x in clusters[cid]) if cid is not None else []
                rep = dict(reproduced_cluster_id=cid,
                           reproduced_member_hash=m22.member_hash(got) if got else None,
                           stored_member_hash=TARGET["member_hash"],
                           hash_match=bool(got and m22.member_hash(got) == TARGET["member_hash"]),
                           cluster_id_match=bool(cid == TARGET["cluster_id"]),
                           reproduced_strings=[decoded_pool[i] for i in got])
                B["leiden_reproduction"] = rep
                print(f"  [blind L{L}] leiden reproduction {arm}: "
                      f"hash_match={rep['hash_match']} cid={cid}", flush=True)
            # placement of every season variant that is a pool node
            for w in ["spring", "summer", "autumn", "winter", "fall"]:
                for v in variants[w]:
                    tid = v["token_id"]
                    if tid is None or tid not in pool_set:
                        continue
                    p = pos_of_id[int(tid)]
                    ci = cl_of.get(p)
                    c = clusters[ci] if ci is not None else []
                    strings = [decoded_pool[i] for i in c]
                    co = sorted({season_of(s) for s in strings} - {None})
                    placement.append(dict(
                        layer=L, arm=arm, tau=tau, season=w, surface=v["surface"],
                        count=v["count"], cluster_id=ci, cluster_size=len(c),
                        in_screen_band=bool(5 <= len(c) <= 60),
                        seasons_co_located=co, n_seasons_co_located=len(co),
                        member_hash=m22.member_hash(sorted(int(x) for x in c)),
                        sample_strings=strings[:25]))
            del clusters, cl_of; gc.collect()
        del R, R_sel; gc.collect()
        B["placement"] = placement
        out["blind_arm"] = B
        write(out)
        print(f"  [blind L{L}] done ({time.time() - tL:.1f}s)", flush=True)

    B["placement"] = placement
    out["blind_arm"] = B
    write(out)
    return B


def verify_target(m22, R, R_sel, sel_pos, cnt, pool_ids, member_pos, per_label,
                  variants, ext_set, n_types, decoded_pool):
    """B2, B5, B6, B7 -- everything that needs the layer-6 means."""
    L, k = TARGET["layer"], TARGET["k"]
    P = R[np.asarray(member_pos)]
    fresh = m22.screen(P, m22.N_PERM_CLUSTER, seed=0)
    bg = m22.background_for_size(R, n_types, k, seed=10_000 * L + k)
    p_bg = m22.p_bg_of(bg, fresh["max_gap_deg"])
    st = TARGET["stored"]
    recompute = dict(
        fresh={kk: fresh[kk] for kk in ("k", "gap_ratio", "max_gap_deg", "radius_ratio",
                                        "circle_resid_over_r", "top2_var", "pc1_var",
                                        "p_closed")},
        fresh_p_bg=p_bg,
        fresh_background_rate_p_closed_lt_05=bg["rate_p_closed_lt_05"],
        stored=st,
        abs_diff=dict(max_gap_deg=abs(fresh["max_gap_deg"] - st["max_gap_deg"]),
                      gap_ratio=abs(fresh["gap_ratio"] - st["gap_ratio"]),
                      top2_var=abs(fresh["top2_var"] - st["top2_var"]),
                      p_closed=abs(fresh["p_closed"] - st["p_closed"]),
                      p_bg=abs(p_bg - st["p_bg"])),
        geometry_bit_reproduced=bool(abs(fresh["max_gap_deg"] - st["max_gap_deg"]) < 1e-9),
        seeds=dict(gap_null_seed=0, n_perm=m22.N_PERM_CLUSTER,
                   background_seed=10_000 * L + k, n_bg_draws=m22.N_BACKGROUND_DRAWS,
                   n_perm_background=m22.N_PERM_BACKGROUND))
    # Monte-Carlo range over seeds 1..20 (prereg 3.6.1)
    pc, pb = [], []
    for s in range(1, 21):
        pc.append(m22.gap_p(k, np.radians(fresh["max_gap_deg"]), m22.N_PERM_CLUSTER, seed=s))
        b2 = m22.background_for_size(R, n_types, k, seed=10_000 * L + k + 7919 * s)
        pb.append(m22.p_bg_of(b2, fresh["max_gap_deg"]))
    recompute["monte_carlo_range"] = dict(
        n_seeds=20,
        p_closed=dict(min=float(np.min(pc)), max=float(np.max(pc)), mean=float(np.mean(pc)),
                      stored_within_range=bool(np.min(pc) <= st["p_closed"] <= np.max(pc))),
        p_bg=dict(min=float(np.min(pb)), max=float(np.max(pb)), mean=float(np.mean(pb)),
                  n_draws_per_seed=int(m22.N_BACKGROUND_DRAWS),
                  stored_within_range=bool(np.min(pb) <= st["p_bg"] <= np.max(pb))),
        note="p_closed uses a deterministic null at seed 0 in the frozen run, so the "
             "stored value is the seed-0 entry; the range shows how much the seed moves it")

    # ---- B6: duplicate-pair angular adjacency in the cluster's own plane ----
    xy, ratios, _ = pca_2d(P)
    cx, cy, r, _ = fit_circle(xy)
    ang = angles(xy, cx, cy)
    strings = [decoded_pool[i] for i in member_pos]
    srt = np.argsort(ang)
    sorted_ang = ang[srt]
    nn = np.degrees(np.diff(np.concatenate([sorted_ang, sorted_ang[:1] + 2 * np.pi])))
    pairs = []
    for w, vs in per_label.items():
        if w is None or len(vs) < 2:
            continue
        idxs = [member_pos.index(v["pos"]) for v in vs]
        a = ang[idxs[0]] - ang[idxs[1]]
        sep = float(np.degrees(abs((a + np.pi) % (2 * np.pi) - np.pi)))
        pairs.append(dict(season=w, surfaces=[v["surface"] for v in vs],
                          angular_separation_deg=sep,
                          adjacent_on_ring=bool(min(abs(int(np.where(srt == idxs[0])[0][0])
                                                        - int(np.where(srt == idxs[1])[0][0])),
                                                    len(srt) - abs(int(np.where(srt == idxs[0])[0][0])
                                                                   - int(np.where(srt == idxs[1])[0][0]))) == 1)))
    adjacency = dict(
        member_angles_deg={s: float(np.degrees(a)) for s, a in zip(strings, ang)},
        angular_order=[strings[i] for i in srt],
        nearest_neighbour_gaps_deg=[float(x) for x in nn],
        mean_nn_gap_deg=float(nn.mean()), max_gap_deg=float(nn.max()),
        duplicate_pairs=pairs,
        note=("if the two case-variant pairs sit at small angular separation the object is "
              "three clumps on a circle, not a five-point ring"))

    # ---- ordering: exact floors, then the two registered readings ----
    lab_of = {"spring": 0, "summer": 1, "autumn": 2, "winter": 3, "fall": 2}
    distinct = [w for w in ("spring", "summer", "autumn", "winter", "fall")
                if w in per_label and per_label[w]]
    best_pos = [per_label[w][0]["pos"] for w in distinct]
    lab_idx = [lab_of[w] for w in distinct]
    sub = [member_pos.index(p) for p in best_pos]
    ordering = dict(
        distinct_labels=distinct, label_index=lab_idx,
        exact_null=exact_order_null(lab_idx, FULL_K),
        from_cluster_plane=m22.score_ordering(xy, sub, lab_idx, FULL_K, seed=m22.cid_seed(
            L, TARGET["arm"], TARGET["cluster_id"])),
        shuffled_label_control=None,
        note=("ordering is scored on the DEDUPED distinct-season set (variant rule); the "
              "raw k = 5 set has duplicate label indices and no well-defined cyclic order"))
    ordering["shuffled_label_control"] = m22.shuffled_label_control(
        xy, sub, lab_idx, FULL_K, seed=m22.cid_seed(L, TARGET["arm"], TARGET["cluster_id"]),
        observed_rho=ordering["from_cluster_plane"]["rho"])
    ordering["from_label_members_only"], _, _ = analyse(
        R[np.asarray(best_pos)], lab_idx, FULL_K, seed=0, do_order=len(sub) >= 3)
    ordering["within_cluster_background"] = m22.within_cluster_background(
        xy, len(sub), seed=m22.cid_seed(L, TARGET["arm"], TARGET["cluster_id"]))
    if ordering["within_cluster_background"]:
        ordering["within_cluster_background"].pop("_g", None)

    # ---- B5 + B7: label-selected 4-sets --------------------------------------
    def best_variant(w, allow_ext):
        for v in variants[w]:
            tid = v["token_id"]
            if tid is None:
                continue
            if tid in set(int(x) for x in pool_ids):
                return v, "pool"
            if allow_ext and tid in ext_set:
                return v, "extended_pool"
        return None, None

    sets = {}
    for name, words in (("spring_summer_winter_fall", ["spring", "summer", "fall", "winter"]),
                        ("spring_summer_winter_autumn", ["spring", "summer", "autumn", "winter"])):
        chosen, tiers, vecs, labs = [], [], [], []
        ok = True
        for w in words:
            v, tier = best_variant(w, allow_ext=True)
            if v is None:
                ok = False
                break
            chosen.append(v["surface"]); tiers.append(tier)
            vecs.append(R_sel[sel_pos[v["token_id"]]])
            labs.append(lab_of[w])
        if not ok:
            sets[name] = dict(feasible=False)
            continue
        Pm = np.vstack(vecs)
        res, _, _ = analyse(Pm, labs, FULL_K, seed=0, do_order=True)
        A = res["conventions"]["A"]
        bg4 = m22.background_for_size(R, n_types, 4, seed=10_000 * L + 4)
        sets[name] = dict(
            words=words, surfaces=chosen, pool_tier=tiers, label_index=labs,
            status=("SUPERVISED-ADJACENT label-selected set, NOT a frozen cluster and NOT "
                    "a blind ring claim" + (" -- uses the >=30 extended pool for autumn, "
                    "permitted by declared_rules.extended_pool_use"
                    if "extended_pool" in tiers else "")),
            top2_var=res["top2_var"], gap_ratio=A["gap_ratio"],
            max_gap_deg=A["max_gap_deg"], radius_ratio=A["radius_ratio"],
            p_closed_uniform=A["p_closed_uniform"],
            p_bg_matched_random_4sets=m22.p_bg_of(bg4, A["max_gap_deg"]),
            background_rate_p_closed_lt_05_k4=bg4["rate_p_closed_lt_05"],
            background_mean_max_gap_deg=bg4["mean_max_gap_deg"],
            rho=A["order"]["rho"], p_order=A["order"]["p"],
            recovered_sequence_matches_registered_cycle=A["order"][
                "recovered_sequence_matches_registered_cycle"],
            exact_order_null=exact_order_null(labs, FULL_K),
            sign_convention_check=dict(
                gap_ratio_identical=bool(abs(A["gap_ratio"]
                                             - res["conventions"]["B"]["gap_ratio"]) < 1e-9),
                rho_identical=bool(abs(A["order"]["rho"]
                                       - res["conventions"]["B"]["order"]["rho"]) < 1e-12),
                rot_only_rho_A=A["order_rotation_only"]["rho"],
                rot_only_rho_B=res["conventions"]["B"]["order_rotation_only"]["rho"]),
            angles_deg=A.get("angles_deg"))

    missing = dict(
        autumn=dict(best_variant=variants["autumn"][0],
                    in_discovery_pool=bool(variants["autumn"][0]["count"] >= 100),
                    in_extended_pool=bool(variants["autumn"][0]["count"] >= 30),
                    answer=("' autumn' best-variant count is "
                            f"{variants['autumn'][0]['count']}, below the >=100 discovery "
                            "floor, so autumn is NOT A NODE in the frozen partitions at "
                            "all. Its absence from the flagged cluster is a POOL-FLOOR "
                            "EXCLUSION, not a geometric absence.")),
        fall=dict(best_variant=variants["fall"][0],
                  in_discovery_pool=bool(variants["fall"][0]["count"] >= 100),
                  answer=("' fall' is in the frozen pool. Its cluster at every layer and "
                          "arm is reported in blind_arm.placement. Registered in advance "
                          "(prereg 3.6.2): absence from the seasons cluster is the "
                          "EXPECTED consequence of verb polysemy ('to fall', 'fall of "
                          "Rome', 'fall short') and is NOT evidence against a seasons "
                          "ring.")))

    return dict(recompute=recompute, adjacency=adjacency, ordering=ordering,
                label_selected_sets=sets, missing_labels=missing)


# ===========================================================================
# SUPERVISED ARM
# ===========================================================================
def token_audit(tk, words):
    keep, drop = [], []
    for w in words:
        ids = tk.encode(" " + w, add_special_tokens=False)
        (keep if len(ids) == 1 else drop).append(
            dict(word=w, n_tokens=len(ids), ids=[int(i) for i in ids]))
    return keep, drop


def build_and_verify(tk, prefix, suffix, word, word_id, bos):
    pre = tk.encode(prefix, add_special_tokens=False)
    suf = tk.encode(suffix, add_special_tokens=False)
    return [bos] + pre + [word_id] + suf, tk.encode(prefix + " " + word + suffix,
                                                    add_special_tokens=True)


def collect(model, tk, words, word_ids, layers, chunk=48):
    from gfdisc.model import capture
    bos = tk.bos_token_id
    n = len(words)
    acc = {ro: {l: np.zeros((n, D_MODEL), dtype=np.float64) for l in layers}
           for ro in READOUTS}
    with torch.inference_mode():
        for prefix, suffix in TEMPLATES:
            pre = tk.encode(prefix, add_special_tokens=False)
            suf = tk.encode(suffix, add_special_tokens=False)
            ci = 1 + len(pre)
            pos = dict(token=ci, suffix=ci + 1, final=ci + len(suf))
            base = [bos] + pre + [0] + suf
            for lo in range(0, n, chunk):
                hi = min(lo + chunk, n)
                ids = []
                for w in word_ids[lo:hi]:
                    r = list(base); r[ci] = int(w); ids.append(r)
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


def principal_angles_deg(A, B):
    Qa = np.linalg.qr(np.asarray(A, dtype=np.float64).T)[0]
    Qb = np.linalg.qr(np.asarray(B, dtype=np.float64).T)[0]
    s = np.clip(np.linalg.svd(Qa.T @ Qb, compute_uv=False), -1.0, 1.0)
    return [float(np.degrees(np.arccos(x))) for x in s]


def supervised_arm(args, out, write):
    from gfdisc.model import load_model
    layers = [int(x) for x in args.layers.split(",")]
    t0 = time.time()
    model, tk = load_model(token=os.environ.get("HF_TOKEN"))
    bos = tk.bos_token_id

    excl = set(SEASONAL_SENSE_EXCLUSIONS)
    season_words = sorted(set(SEASONS) | {"fall"})
    s_keep, s_drop = token_audit(tk, season_words)
    if s_drop:
        raise SystemExit(f"season words not single-token: {s_drop}")
    noun_cand = [w for w in dict.fromkeys(NOUN_POOL) if w not in excl]
    per_cand = [w for w in dict.fromkeys(PERIOD_POOL) if w not in excl]
    n_keep, n_drop = token_audit(tk, noun_cand)
    p_keep, p_drop = token_audit(tk, per_cand)
    noun_words = [x["word"] for x in n_keep]
    per_words = [x["word"] for x in p_keep]

    # template audit: suffix length and manual-vs-natural tokenisation
    tmpl_audit = []
    for prefix, suffix in TEMPLATES:
        suf = tk.encode(suffix, add_special_tokens=False)
        tmpl_audit.append(dict(prefix=prefix, suffix=suffix, n_suffix_tokens=len(suf),
                               ok=bool(len(suf) >= 3),
                               contains_label=bool(any(
                                   w in (prefix + suffix).lower() for w in
                                   ("spring", "summer", "autumn", "winter", " fall")))))
    if any(not t["ok"] for t in tmpl_audit):
        raise SystemExit(f"suffix under 3 tokens: {[t for t in tmpl_audit if not t['ok']]}")
    if any(t["contains_label"] for t in tmpl_audit):
        raise SystemExit(f"template contains a label word: "
                         f"{[t['prefix'] for t in tmpl_audit if t['contains_label']]}")

    def tokenises_everywhere(w):
        wid = tk.encode(" " + w, add_special_tokens=False)[0]
        for prefix, suffix in TEMPLATES:
            m, nat = build_and_verify(tk, prefix, suffix, w, wid, bos)
            if m != nat:
                return False, dict(word=w, prefix=prefix, manual=m, natural=nat)
        return True, None

    bad_labels, dropped_pool = [], []
    for w in season_words:
        ok, info = tokenises_everywhere(w)
        if not ok:
            bad_labels.append(info)
    if bad_labels:
        raise SystemExit(f"label tokenisation mismatch: {bad_labels}")
    keep_n, keep_p = [], []
    for w in noun_words:
        ok, info = tokenises_everywhere(w)
        (keep_n if ok else dropped_pool).append(w if ok else info)
    for w in per_words:
        ok, info = tokenises_everywhere(w)
        (keep_p if ok else dropped_pool).append(w if ok else info)
    noun_words, per_words = keep_n, keep_p
    if len(noun_words) < 40 or len(per_words) < 20:
        raise SystemExit(f"background pools under floor: {len(noun_words)} nouns, "
                         f"{len(per_words)} period nouns")

    words = season_words + noun_words + per_words
    word_ids = [tk.encode(" " + w, add_special_tokens=False)[0] for w in words]
    idx = {w: i for i, w in enumerate(words)}
    noun_idx = np.array([idx[w] for w in noun_words])
    per_idx = np.array([idx[w] for w in per_words])
    sets = {"autumn_primary": SEASONS, "fall_sensitivity": SEASONS_FALL}
    set_idx = {k: np.array([idx[w] for w in v]) for k, v in sets.items()}
    label_index = np.arange(FULL_K)

    audit = dict(season_words=season_words, n_nouns=len(noun_words),
                 n_period_nouns=len(per_words),
                 nouns=noun_words, period_nouns=per_words,
                 dropped_multi_token=[x["word"] for x in n_drop + p_drop],
                 dropped_template_mismatch=dropped_pool,
                 excluded_by_rule=sorted(set(NOUN_POOL + PERIOD_POOL) & excl),
                 templates=tmpl_audit,
                 verdict_bearing_set="autumn_primary",
                 non_verdict_bearing_set="fall_sensitivity")
    print(f"  [sup] {len(words)} words x {len(TEMPLATES)} templates x {len(layers)} layers "
          f"({len(noun_words)} nouns, {len(per_words)} period nouns)", flush=True)

    tf = time.time()
    M = collect(model, tk, words, word_ids, layers, chunk=args.chunk)
    print(f"  [sup] forwards done in {time.time() - tf:.1f}s", flush=True)
    del model
    if DEVICE == "mps":
        torch.mps.empty_cache()

    ref_layer = layers[0]
    _, basis_ref, _ = analyse(M["token"][ref_layer][set_idx["autumn_primary"]],
                              label_index, FULL_K, seed=0, do_order=False,
                              both_conventions=False)

    rows = []
    print(f"{'set':>16} {'readout':>7} {'L':>3} {'top2':>6} {'gapR':>7} {'pUnif':>8} "
          f"{'pNoun':>8} {'FAnoun':>7} {'pPeriod':>8} {'rho':>7} {'pOrd':>8} {'ordOK':>6} "
          f"{'ang0':>6}  verdict", flush=True)
    for sname, sidx in set_idx.items():
        for ro in READOUTS:
            for l in layers:
                P = M[ro][l][sidx]
                res, basis, _ = analyse(P, label_index, FULL_K, seed=l, do_order=True)
                A, Bc = res["conventions"]["A"], res["conventions"]["B"]
                sign_check = dict(
                    gap_ratio_identical=bool(abs(A["gap_ratio"] - Bc["gap_ratio"]) < 1e-9),
                    rho_identical=(None if res["degenerate"] else
                                   bool(abs(A["order"]["rho"] - Bc["order"]["rho"]) < 1e-12)),
                    rot_only_rho_A=(None if res["degenerate"]
                                    else A["order_rotation_only"]["rho"]),
                    rot_only_rho_B=(None if res["degenerate"]
                                    else Bc["order_rotation_only"]["rho"]))

                def bg(pool_idx, n_draws, seed):
                    r = np.random.default_rng(seed)
                    gr = np.empty(n_draws); pc = np.empty(n_draws); rr = np.empty(n_draws)
                    for i in range(n_draws):
                        pick = pool_idx[r.choice(len(pool_idx), size=FULL_K, replace=False)]
                        gr[i], pc[i], rr[i] = fast_primary(M[ro][l][pick])
                    return dict(n_draws=int(n_draws), gap_ratio_mean=float(gr.mean()),
                                gap_ratio_p05=float(np.percentile(gr, 5)),
                                gap_ratio_min=float(gr.min()),
                                radius_ratio_mean=float(rr.mean()),
                                false_acceptance_rate_p05=float((pc <= 0.05).mean()),
                                p_vs_background=float((1 + int((gr <= A["gap_ratio"]).sum()))
                                                      / (1 + n_draws)))

                b_noun = bg(noun_idx, args.n_background, 1000 + l)
                b_per = bg(per_idx, args.n_background, 2000 + l)
                pa = principal_angles_deg(basis, basis_ref)
                if res["degenerate"]:
                    verdict = "degenerate"
                else:
                    closed = (A["p_closed_uniform"] < 0.05 and b_noun["p_vs_background"] < 0.05)
                    ordered = A["order"]["recovered_sequence_matches_registered_cycle"]
                    verdict = ("closed_ring_correct_order" if closed and ordered else
                               "closed_ring_unordered" if closed else
                               "correct_order_not_closed" if ordered else "nothing")
                rows.append(dict(
                    label_set=sname, readout=ro, layer=l, k=FULL_K,
                    verdict_bearing=bool(sname == "autumn_primary"),
                    degenerate=res["degenerate"], top2_var=res["top2_var"],
                    pc_var=res["pc_var"], conventions=res["conventions"],
                    sign_convention_check=sign_check, background_nouns=b_noun,
                    background_period_nouns=b_per,
                    principal_angles_to_ref_token_plane_deg=pa, verdict=verdict))
                rho = None if res["degenerate"] else A["order"]["rho"]
                po = None if res["degenerate"] else A["order"]["p"]
                ok = None if res["degenerate"] else A["order"][
                    "recovered_sequence_matches_registered_cycle"]
                print(f"{sname:>16} {ro:>7} {l:3d} {res['top2_var']:6.3f} "
                      f"{A['gap_ratio']:7.2f} {A['p_closed_uniform']:8.4f} "
                      f"{b_noun['p_vs_background']:8.4f} "
                      f"{b_noun['false_acceptance_rate_p05']:7.3f} "
                      f"{b_per['p_vs_background']:8.4f} "
                      f"{'    n/a' if rho is None else f'{rho:+7.3f}'} "
                      f"{'     n/a' if po is None else f'{po:8.4f}'} "
                      f"{str(ok):>6} {pa[0]:6.1f}  {verdict}", flush=True)

    # ---- autumn / fall synonymy: do they land at the same angle? -------------
    syn = []
    for ro in READOUTS:
        for l in layers:
            Pm = M[ro][l][np.array([idx["spring"], idx["summer"], idx["autumn"],
                                    idx["winter"], idx["fall"]])]
            xy, _, _ = pca_2d(Pm)
            cx, cy, r, _ = fit_circle(xy[:4])
            a = angles(xy, cx, cy)
            sep = float(np.degrees(abs((a[2] - a[4] + np.pi) % (2 * np.pi) - np.pi)))
            nn = np.sort(a[:4])
            g = np.degrees(np.diff(np.concatenate([nn, nn[:1] + 2 * np.pi])))
            rng = np.random.default_rng(5000 + l)
            null = []
            for _ in range(500):
                p2 = noun_idx[rng.choice(len(noun_idx), size=2, replace=False)]
                Q = np.vstack([M[ro][l][sidx] for sidx in [set_idx["autumn_primary"]]]
                              + [M[ro][l][p2]])
                z, _, _ = pca_2d(Q)
                c2 = fit_circle(z[:4])
                aa = angles(z, c2[0], c2[1])
                null.append(float(np.degrees(abs((aa[4] - aa[5] + np.pi) % (2 * np.pi) - np.pi))))
            syn.append(dict(readout=ro, layer=l, autumn_fall_separation_deg=sep,
                            mean_season_nn_gap_deg=float(g.mean()),
                            ratio_to_mean_nn_gap=float(sep / max(g.mean(), 1e-9)),
                            matched_noun_pair_null_mean_deg=float(np.mean(null)),
                            p_vs_noun_pair_null=float((1 + int((np.asarray(null) <= sep).sum()))
                                                      / (1 + len(null)))))

    sup = dict(audit=audit, rows=rows, autumn_fall_synonymy=syn,
               autumn_fall_synonymy_caveat=(
                   "PLANE MISMATCH, stated rather than corrected: the observed "
                   "autumn-fall separation is read in the PCA plane of 5 points (the four "
                   "seasons plus fall); each null draw is read in the PCA plane of 6 "
                   "points (the four seasons plus two nouns). The circle centre is matched "
                   "in both (fitted on the four seasons alone), the plane is not. The null "
                   "is therefore approximate and this quantity is descriptive; it gates no "
                   "verdict."),
               runtime_sec=float(time.time() - t0))
    out["supervised_arm"] = sup
    write(out)
    return sup


# ===========================================================================
def jsonable(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.bool_,)):
        return bool(o)
    raise TypeError(f"{type(o)} is not JSON serialisable")


def declarations():
    return dict(
        prereg="docs/phase2_prereg_2026-08-21.md, section 3.6 (SEASONS -- CONFIRMATORY)",
        track="CONFIRMATORY. Seasons were blind-flagged at layer 6, k = 5, BEFORE the "
              "prereg, and summarized in-session (prereg section 2.1). Not a discovery.",
        S1_label_sets=dict(primary_verdict_bearing=SEASONS,
                           synonym_sensitivity_not_verdict_bearing=SEASONS_FALL,
                           registered_forms=list(FORMS), full_k=FULL_K,
                           registered_cycle="spring -> summer -> autumn -> winter"),
        S2_readouts=dict(readouts=list(READOUTS),
                         primary="suffix (first token AFTER the season word)",
                         token="EMBEDDING-CONFOUNDED at layers 0-2", final="tertiary"),
        S3_discriminator="closure against the matched same-arity background; ordering is "
                         "never evidence on its own",
        S4_nulls=dict(uniform_angles=N_GAP_NULL, matched_noun_draws=N_BACKGROUND,
                      class_matched_period_noun_draws=N_BACKGROUND,
                      shuffled_label_permutations=N_PERM_ORDER,
                      explicit_shuffle_control=N_SHUFFLE_CONTROL,
                      background_not_scored_for_ordering=True),
        S5_k4_power="computed in-script by enumeration (ordering) and by table (closure); "
                    "ordering p cannot gate a verdict at k = 4",
        S6_synthetics="planted 4-ring SNR ladder / 200-degree arc / line / isotropic noise "
                      "through this script's own analyse(); planted-ring ordering criterion "
                      "is rho == 1 exactly, not p_order < 0.05",
        S7_sign_conventions="both reported; reflection-invariance of max-over-symmetries rho "
                            "asserted; rotation-only rho reported per convention",
        S8_multiplicity="BH within the 78-cell primary sweep, with raw counts, expected-by-"
                        "chance count and p-value floors",
        S9_no_widened_plane_search="registered omission: prereg section 7 forbids adding a "
                                   "further search family; only the top-2 PCA plane is scored",
        S10_pool_exclusions=SEASONAL_SENSE_EXCLUSIONS,
        S11_template_rule="every prefix establishes the temporal sense before the season "
                          "token; suffix >= 3 tokens and token-identical across words; no "
                          "template contains a label word",
        B_blind_arm=dict(
            target=TARGET,
            honest_framing="raw_leiden is the SECONDARY arm; bh_pass_within_layer_arm is "
                           "FALSE; the k = 5 matched background false-accepts the uniform "
                           "closure test at 0.08",
            reproduction="member_hash rebuilt from token strings without clustering; "
                         "screening recomputed with the imported functions of "
                         "scripts/22_token_type_geometry.py at the frozen seeds",
            leiden="re-run at the FROZEN selected tau, resolution 1.0, seed 0 -- a "
                   "reproduction, not a re-selection; it gates only the placement question",
            variant_rule="inherited verbatim from the frozen artifact; counts never summed",
            label_selected_sets="SUPERVISED-ADJACENT, never a blind ring claim"),
        recovery_amendments=RECOVERY_AMENDMENTS,
        verdict_vocabulary=["recovered_blind (confirmatory)", "ordered_not_closed",
                            "clean_negative (underpowered, k=4)", "untestable",
                            "structure_exists_supervised_only"],
        caveats_that_survive_a_positive=[
            "the blind representation is a context-averaged token-type mean over every "
            "corpus occurrence -- closer to a static embedding than a computed feature",
            "layers 0-4 are embedding-adjacent; a hit there is not a depth claim",
            "a closed ring at the suffix readout shows the information is present one token "
            "later, not that the model recomputed it; attention can copy an embedding "
            "forward. The principal angle to the layer-0 token plane is the only handle."],
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/phase2_seasons.json")
    ap.add_argument("--root", default="cache/sparse_big")
    ap.add_argument("--l0", default="big")
    ap.add_argument("--layers", default=",".join(str(i) for i in range(26)))
    ap.add_argument("--blind-layers", default="0,1,2,3,4,5,6,7,8,9,10")
    ap.add_argument("--n-background", type=int, default=N_BACKGROUND)
    ap.add_argument("--chunk", type=int, default=48)
    ap.add_argument("--skip-blind", action="store_true")
    ap.add_argument("--skip-supervised", action="store_true")
    a = ap.parse_args()
    t0 = time.time()
    outp = Path(a.out)
    outp.parent.mkdir(parents=True, exist_ok=True)

    out = dict(status=STATUS, item="Phase-2 seasons (CONFIRMATORY)",
               script="scripts/33_phase2_seasons.py",
               declared_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
               declarations=declarations(), config=vars(a))

    def write(o):
        o["runtime_seconds"] = round(time.time() - t0, 1)
        outp.write_text(json.dumps(o, indent=2, default=jsonable))

    write(out)
    print("=" * 78)
    print("PHASE 2 -- SEASONS (CONFIRMATORY). Declarations written to", a.out)
    print(json.dumps(declarations(), indent=2)[:2500], flush=True)
    print("=" * 78, flush=True)

    # ---- S5: power, before anything is read ----------------------------------
    power = dict(
        ordering_exact_k4=exact_order_null(list(range(4)), 4),
        ordering_exact_k3_of_4=exact_order_null([0, 1, 3], 4),
        ordering_exact_k5_reference=exact_order_null(list(range(5)), 5),
        closure_uniform_alpha05_max_gap_deg={
            str(k): gap_threshold_for_alpha(k) for k in (3, 4, 5, 6, 8, 12)},
        perfect_ring_p_closed={
            str(k): p_closed(k, np.radians(360.0 / k)) for k in (3, 4, 5, 6, 8, 12)},
        note=("the k = 4 ordering p-value is floored by construction, so at k = 4 the "
              "binary criterion (recovered sequence == registered cycle up to dihedral "
              "symmetry) with its exact chance probability is the ordering readout, and "
              "closure against the matched background is the discriminator"))
    out["power"] = power
    ref = check_against_reference()
    out["reference_check"] = ref
    print("power / k=4 ordering floor: rho_max "
          f"{power['ordering_exact_k4']['rho_max_achievable']:.3f}, chance prob "
          f"{power['ordering_exact_k4']['exact_chance_prob_of_rho_max']:.4f}, min "
          f"attainable p_order {power['ordering_exact_k4']['min_attainable_p_order']:.4f}",
          flush=True)
    print(f"closure at k=4: uniform null accepts max_gap <= "
          f"{power['closure_uniform_alpha05_max_gap_deg']['4']:.1f} deg at alpha=0.05 "
          f"(perfect square = 90.0 deg)", flush=True)
    if not ref["passed"]:
        raise SystemExit("vectorised ordering statistic disagrees with fits.cyclic_order_score")
    syn = synthetics()
    out["synthetics"] = syn
    write(out)
    for key in ("noise_0.0", "noise_0.15", "noise_0.35", "noise_0.6"):
        r = syn["snr_ladder"][key]
        print(f"  planted 4-ring {key:>10}: gapR {r['gap_ratio']:6.2f} pUnif "
              f"{r['p_closed']:.4f} pBg {r['p_vs_background']:.4f} rho {r['rho']:+.3f} "
              f"rho==1 {r['rho_is_one']}", flush=True)
    for key in ("planted_arc_200deg", "planted_line"):
        r = syn[key]
        print(f"  {key:>22}: gapR {r['gap_ratio']:6.2f} pUnif {r['p_closed']:.4f}", flush=True)
    print(f"  synthetic battery: {'PASS' if syn['passed'] else 'FAIL'}", flush=True)
    if not syn["passed"]:
        raise SystemExit("synthetic battery failed; not running on data")

    if not a.skip_blind:
        print("\n--- BLIND ARM (frozen artifact only, no re-clustering) ---", flush=True)
        blind_arm(a, out, write)
    if not a.skip_supervised:
        print("\n--- SUPERVISED ARM (26-layer templated sweep) ---", flush=True)
        supervised_arm(a, out, write)

    # ---- S8 multiplicity + verdicts ------------------------------------------
    rows = out.get("supervised_arm", {}).get("rows", [])
    prim = [r for r in rows if r["verdict_bearing"] and not r["degenerate"]]

    def fdr(pvals, label, n_null):
        p = np.asarray(pvals, dtype=float)
        keep = benjamini_hochberg(p, alpha=0.05) if len(p) else np.zeros(0, bool)
        floor = 1.0 / (n_null + 1.0)
        rank1 = 0.05 / max(len(p), 1)
        return dict(statistic=label, n_cells=int(len(p)), n_null_draws=int(n_null),
                    n_raw_below_05=int((p < 0.05).sum()),
                    expected_by_chance_at_05=float(0.05 * len(p)),
                    p_value_floor=float(floor), bh_rank1_threshold=float(rank1),
                    isolated_cell_can_survive_bh=bool(floor <= rank1),
                    n_surviving_bh_fdr=int(keep.sum()),
                    surviving=[dict(readout=prim[i]["readout"], layer=prim[i]["layer"],
                                    p=float(p[i])) for i in np.flatnonzero(keep)])

    mult = None
    if prim:
        p0 = out["power"]["ordering_exact_k4"]["exact_chance_prob_of_rho_max"]
        n_ok = int(sum(r["conventions"]["A"]["order"][
            "recovered_sequence_matches_registered_cycle"] for r in prim))
        from math import comb
        n = len(prim)
        p_binom = float(sum(comb(n, i) * p0 ** i * (1 - p0) ** (n - i)
                            for i in range(n_ok, n + 1)))
        mult = dict(
            n_cells=n,
            note="3 readouts x 26 layers on the PRIMARY (autumn) set only; the fall "
                 "sensitivity sweep is reported but excluded from this family",
            closure_uniform=fdr([r["conventions"]["A"]["p_closed_uniform"] for r in prim],
                                "closure vs uniform angles", N_GAP_NULL),
            closure_noun_background=fdr([r["background_nouns"]["p_vs_background"]
                                         for r in prim],
                                        "closure vs matched noun background", a.n_background),
            closure_period_background=fdr([r["background_period_nouns"]["p_vs_background"]
                                           for r in prim],
                                          "closure vs class-matched period-noun background",
                                          a.n_background),
            ordering_p_floored=fdr([r["conventions"]["A"]["order"]["p"] for r in prim],
                                   "ordering p (FLOORED at k=4, cannot gate)", N_PERM_ORDER),
            # POST-HOC ADDITION, provenance stated: added after the first (aborted) launch
            # and before any activation was read. The declared supervised verdict rule can
            # in principle be satisfied by BH survivors at one readout and a correct-order
            # cell at another, so the BH survivors are broken out by readout rather than
            # left as a single count. It changes no verdict; it only makes one readable.
            closure_noun_background_by_readout=dict(
                provenance="POST-HOC field, added before any activation was read; "
                           "read off the BH survivor list, no new statistic",
                counts={ro: int(sum(1 for s in fdr(
                    [r["background_nouns"]["p_vs_background"] for r in prim],
                    "x", a.n_background)["surviving"] if s["readout"] == ro))
                    for ro in READOUTS}),
            correct_order_binomial=dict(
                n_cells=n, n_correct=n_ok, chance_prob_per_cell=float(p0),
                expected_by_chance=float(n * p0), p_binomial_upper_tail=p_binom,
                caveat="cells at adjacent layers are correlated; this is a reference "
                       "point, not an independent test"))
        out["multiplicity"] = mult
        out["verdict_counts"] = {
            s: {ro: {name: [r["layer"] for r in rows if r["label_set"] == s
                            and r["readout"] == ro and r["verdict"] == name]
                     for name in ("closed_ring_correct_order", "closed_ring_unordered",
                                  "correct_order_not_closed", "nothing", "degenerate")}
                for ro in READOUTS}
            for s in ("autumn_primary", "fall_sensitivity")}

    # ---- final verdicts, per the registered vocabulary -----------------------
    ver = {}
    B = out.get("blind_arm")
    if B and "verification" in B:
        v = B["verification"]
        rep_closed = bool(v["recompute"]["fresh"]["p_closed"] < 0.05
                          and v["recompute"]["fresh_p_bg"] is not None
                          and v["recompute"]["fresh_p_bg"] <= 0.05)
        ndist = B["variant_rule_applied"]["n_distinct_labels_present"]
        ver["blind_reproduction"] = dict(
            question="does the frozen k=5 flagged cluster reproduce exactly?",
            geometry_bit_reproduced=v["recompute"]["geometry_bit_reproduced"],
            member_hash_match=B["member_hash_reproduced"]["match"],
            flag_reproduced=rep_closed,
            verdict=("reproduced" if (v["recompute"]["geometry_bit_reproduced"]
                                      and B["member_hash_reproduced"]["match"]
                                      and rep_closed) else "NOT reproduced"))
        ver["blind_interpretation"] = dict(
            question="is the flagged cluster a SEASONS ring?",
            n_distinct_seasons_in_cluster=ndist,
            n_of_registered_4_cycle_present=int(len([w for w in
                                                     B["variant_rule_applied"]["distinct_labels_present"]
                                                     if w in ("spring", "summer", "autumn", "winter")])),
            verdict=("untestable" if ndist < 3 else
                     "clean_negative (underpowered, k=4)" if ndist < 4 else
                     "recovered_blind (confirmatory)"),
            duplicate_pair_separations_deg=[p["angular_separation_deg"]
                                            for p in v["adjacency"]["duplicate_pairs"]],
            duplicate_pairs_angularly_adjacent=[p["adjacent_on_ring"]
                                                for p in v["adjacency"]["duplicate_pairs"]],
            mean_nn_gap_deg=v["adjacency"]["mean_nn_gap_deg"],
            deduped_triple_max_gap_deg=v["ordering"]["from_cluster_plane"]["max_gap_deg"],
            deduped_triple_p_closed=v["ordering"]["from_cluster_plane"]["p_closed"],
            within_cluster_random_triple_mean_max_gap_deg=(
                v["ordering"]["within_cluster_background"] or {}).get("mean_max_gap_deg"),
            reason=("after the artifact's own variant rule the flagged cluster carries "
                    f"{ndist} distinct seasons, below the registered 4-cycle; a season ring "
                    "missing a season is a different object, and a "
                    f"{ndist}-point set is below the frozen screening band minimum of 5, so "
                    "it cannot be screened as a cluster in its own right. Two further "
                    "numbers make the dissolution concrete: (a) BOTH case-variant pairs are "
                    "angular neighbours on the fitted circle, so the five-point spread is "
                    "carried by case duplicates rather than by four seasons; (b) the "
                    "deduped distinct-season subset has a larger maximum angular gap than "
                    "the average RANDOM same-size subset of its own cluster -- within the "
                    "flagged cluster, the distinct seasons are not the ring-forming "
                    "subset."))
    if prim:
        suffix_rows = [r for r in prim if r["readout"] == "suffix"]
        n_closed = sum(1 for r in suffix_rows
                       if r["conventions"]["A"]["p_closed_uniform"] < 0.05
                       and r["background_nouns"]["p_vs_background"] < 0.05)
        n_full = sum(1 for r in suffix_rows if r["verdict"] == "closed_ring_correct_order")
        ver["supervised"] = dict(
            question="does a 4-season ring exist under templated supervision?",
            n_suffix_cells=len(suffix_rows), n_suffix_closed_both_nulls=n_closed,
            n_suffix_closed_and_correct_order=n_full,
            n_surviving_bh_on_noun_background=mult["closure_noun_background"]["n_surviving_bh_fdr"],
            correct_order_binomial_p=mult["correct_order_binomial"]["p_binomial_upper_tail"],
            verdict=("structure_exists_supervised_only"
                     if mult["closure_noun_background"]["n_surviving_bh_fdr"] > 0 and n_full > 0
                     else "clean_negative (underpowered, k=4)"))
    out["verdicts"] = ver
    write(out)

    print("\n" + "=" * 78)
    print("VERDICTS (registered vocabulary only)")
    for k, v in ver.items():
        print(f"  {k:22s} -> {v['verdict']}")
    if mult:
        print(f"\nMULTIPLICITY over {mult['n_cells']} verdict-bearing cells "
              f"(expected by chance at 0.05: {0.05 * mult['n_cells']:.1f})")
        for key in ("closure_uniform", "closure_noun_background",
                    "closure_period_background", "ordering_p_floored"):
            m = mult[key]
            print(f"  {key:28s} raw p<0.05: {m['n_raw_below_05']:2d}  BH: "
                  f"{m['n_surviving_bh_fdr']:2d}  [floor {m['p_value_floor']:.2e}, "
                  f"rank-1 {m['bh_rank1_threshold']:.2e}]")
        cb = mult["correct_order_binomial"]
        print(f"  correct-order cells: {cb['n_correct']}/{cb['n_cells']} "
              f"(expected {cb['expected_by_chance']:.1f}, binomial p "
              f"{cb['p_binomial_upper_tail']:.4f})")
    print(f"\nruntime {time.time() - t0:.1f}s -> wrote {a.out}")


if __name__ == "__main__":
    main()
