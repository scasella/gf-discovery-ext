"""PHASE 2 -- ZODIAC. Blind arm gated out in advance; supervised 26-layer sweep.

Binding document: docs/phase2_prereg_2026-08-21.md (frozen 2026-08-21, before any
Phase-2 ontology analysis). Label set, predicted structure, discriminating test and
verdict vocabulary come from its section 3.3; the arm mechanics come from its 4.1/4.2;
the feasibility gate from its 5.2/5.6. Nothing here invents a test.

Committed prediction (src/gfdisc/ontologies.py at 3cadb69, 2026-08-18 12:04:20 -0400):

    ZODIAC = ["Aries", "Taurus", "Gemini", "Cancer", "Leo", "Virgo",
              "Libra", "Scorpio", "Sagittarius", "Capricorn", "Aquarius", "Pisces"]
    "zodiac": dict(words=ZODIAC, kind="cycle", k=12)

TWO ARMS, AND ONE OF THEM IS CLOSED BEFORE IT STARTS.

  Z0  BLIND ARM = `untestable`, registered in advance at prereg 3.3 and 5.2, decided
      from token counts alone before any placement was examined. Only 1 of 12 signs
      clears the >=100 count floor that defines membership in the frozen partitions
      (` cancer` 244 -- and that count is the disease sense); 3 of 12 clear the
      extended >=30 floor (`aries` 50, ` cancer` 244, ` Leo` 30). The blind arm needs
      ceil(2/3 * 12) = 8. This script RE-DERIVES those counts from
      cache/sparse_big/layer_0/tokens.i32 (counts only: no SAE indices, no values, no
      means, no geometry) so the gate rests on a number computed here rather than a
      number quoted from a document.
      `results/token_type_geometry.json` -> `verdict.other_rings` IS NOT OPENED by this
      script, and no zodiac cluster placement is looked at. Prereg section 7 forbids
      reporting a blind-arm negative for zodiac; with 1 of 12 labels there is also no
      registered descriptive placement to report (unlike the compass cardinals, which
      the prereg does register as descriptive-only). Opening the catalogue would be
      pure exposure with no registered use, so it is not opened.

  Z1  SUPERVISED ARM = the whole result. Registered as feasible at prereg 3.3: all
      twelve signs have a single-token leading-space capitalized form, and this arm
      generates its own activations from templates, so corpus occurrence does not gate
      it. Any positive here is `structure_exists_supervised_only` -- a weaker claim
      than a blind recovery, and the prereg says so at 5.6 in advance.

DECLARATIONS FIXED BEFORE ANY ACTIVATION IS READ (written to the output JSON under
"declarations" and printed before the first forward pass):

  Z2  Label set and variant. The twelve signs in the committed calendar order, each as
      its ' {W}' capitalized leading-space form -- ' Cancer' (id 21120), NOT ' cancer'
      (id 7923). The capitalized form is the registered variant at prereg 3.3 and is
      also the less disease-dominated surface. Any sign whose ' {W}' form is not a
      single token is dropped and the drop recorded; the prereg's count table says none
      will be, and that is verified here rather than assumed.

  Z3  Readouts, inherited from scripts/24_colors_existence.py (the verified template)
      and from prereg 4.2:
        token  -- the sign token itself. EMBEDDING-CONFOUNDED. A ring here at layers
                  0-2 is lexical geometry, declared in advance as the confound.
        suffix -- the first token AFTER the sign. PRIMARY COMPUTED READOUT. The
                  ontology-level verdict is computed from this readout ONLY.
        final  -- last token of the template. Tertiary, exploratory.

  Z4  Closure is the discriminator; ordering is never evidence on its own. This project
      measured random subsets ordering at ~0.16 and an ordinal control out-ordering
      real month cycles at every layer.

  Z5  Nulls.
      (a) uniform random angles, k of them, 20,000 draws.
      (b) 2000 draws of k matched same-arity single-token common nouns through the
          IDENTICAL templates at the IDENTICAL positions. Prereg 4.2(b): the null that
          matters. Reported as a false-acceptance rate at alpha=0.05 and an empirical p.
      (c) 2000 draws of k single-token personal FIRST NAMES -- prereg 4.2(c) registers
          personal names as the class-matched secondary pool for zodiac specifically.
          The signs are capitalized proper nouns; lowercase common nouns are not
          capitalization-matched and names are, so this pool tests whether any ring is
          a proper-noun/capitalization artifact.
      (d) ordering: 20,000 shuffled-label permutations under the same
          max-over-symmetries selection, plus an explicit 1,000-shuffle control block.
      Background draws are NOT scored for ordering: random nouns and random names have
      no true cyclic order. Closure is the background criterion.

  Z6  Pool exclusion rule, declared before the pools are used (prereg 4.2, last bullet):
      no background word may carry an established sense belonging to the ontology under
      test, and calendar words are excluded from EVERY pool so the pools cannot inherit
      this project's other ontologies. Concretely excluded: the twelve sign names
      themselves; the zodiac symbol animals and figures (ram, bull, twins, crab, lion,
      virgin, scales, scorpion, archer, goat, fish, ...); astrology vocabulary (star,
      sign, horoscope, planet, moon, sun, chart, constellation, ...); and every month
      name, weekday name and season name. The names pool in particular is filtered for
      Leo (a label), April, May, June, Summer, Autumn and Dawn, which a standard
      first-name list contains.

  Z7  Both SVD sign conventions are reported. `cyclic_order_score` maximises over the
      2k dihedral symmetries and is reflection-invariant, so rho must be IDENTICAL
      under a PC2 sign flip; that identity is asserted, not assumed. The rotation-only
      statistic is reported per convention as the quantity a sign flip does change.

  Z8  ORDERING PREDICTION AND ITS CONFOUND, stated before the result. The registered
      order is the calendar-conventional recitation Aries -> Taurus -> ... -> Pisces.
      That order is also the order of the calendar year, so it is exactly as
      convention-carried as this project's month result: a positive is partially
      convention-confounded and MUST be reported that way. The zodiac's only advantage
      over months is that the sign names are not themselves date tokens.

  Z9  Per-cell verdicts, four outcomes plus degenerate:
        zodiac_circle          closure passes (a) AND (b) AND ordering p < 0.05
        closed_ring_unordered  closure passes (a) AND (b), ordering does not
        ordered_not_closed     ordering passes, closure does not
        nothing                neither
        degenerate             the point cloud carries no usable geometry
      The prereg 3.3 vocabulary word is `ordered_not_closed`, not the colours script's
      `ordered_arc`; the prereg vocabulary is used.

  Z10 ONTOLOGY-LEVEL VERDICT RULE, frozen here before the run. Computed from the
      SUFFIX readout only (Z3), with multiplicity applied across the whole 78-cell
      sweep (Z11):
        structure_exists_supervised_only
            -- some suffix cell has p_closed_uniform < 0.05, p_vs_noun_background <
               0.05 surviving BH-FDR across the sweep, ordering p < 0.05 surviving
               BH-FDR across the sweep, AND p_vs_names_background < 0.05.
        closed_ring_unordered  -- the same closure conditions, ordering does not hold.
        ordered_not_closed     -- some suffix cell's ordering survives BH-FDR and no
                                  cell meets the closure conditions.
        clean_negative         -- none of the above, AND the synthetic battery passed
                                  (so the sweep demonstrably had the power to see a
                                  planted k=12 ring).
        untestable             -- the blind arm, by Z0.
      THE NAMES-POOL CLAUSE IS A TIGHTENING, declared with provenance: prereg 4.2 calls
      the noun pool "the null that matters" and the class-matched pool secondary. This
      script requires the class-matched pool as well at the ontology level, because a
      twelve-proper-noun ring that cannot beat twelve random capitalized first names is
      a capitalization artifact. Frozen now so it cannot be decided after a near-miss.
      It can only make a positive harder, never easier.

  Z11 MULTIPLICITY. 3 readouts x 26 layers = 78 cells. At alpha = 0.05 about 4 ordering
      passes and about 4 closure passes are expected by chance, so an isolated pass is
      not a finding (prereg 4.4). Benjamini-Hochberg across all live cells is reported
      alongside the raw counts, the expected-by-chance count, and each statistic's
      p-value floor (an add-one empirical p cannot fall below 1/(n_null+1), and a
      "0 survive BH" line has to be read against what was reachable).

  Z12 SECONDARY, WIDENED PLANE SEARCH -- provenance stated plainly. This is NOT in
      docs/phase2_prereg_2026-08-21.md. It is copied from scripts/24_colors_existence.py
      D10, and it is declared HERE, in this file, before any activation is read, rather
      than added after a negative (which prereg section 7 forbids). Rule: all C(4,2) = 6
      planes spanned by pairs of the top 4 PCs; the plane minimising the angular gap
      ratio is selected. Selection is LABEL-FREE, so the shuffled-label ordering null on
      the winning plane stays exact. The uniform-angle p-value on a SELECTED plane is
      invalid and is reported marked as such; the valid test is the matched background
      with the IDENTICAL min-over-6-planes selection applied to every draw. Z9/Z10
      verdicts are computed on the PRIMARY statistic only; the secondary gets its own
      field so any positive is visibly a product of the widened search.
      THIS IS THE LAST EXPANSION for zodiac. If both are negative the finding is "no
      zodiac circle detectable by this method family".

CAVEATS THAT SURVIVE A POSITIVE, recorded now (prereg 4.5 plus Z8):
  - A closed ring at the suffix readout shows the information is present one token
    later; attention can copy an embedding forward, so it is not proof of
    recomputation. The principal angle between the suffix plane and the earliest-layer
    token-readout plane is the only handle reported on that.
  - Zodiac order IS calendar order, so any positive is partially convention-confounded.
  - This arm chose its label set by name. `structure_exists_supervised_only` is a
    weaker claim than Phase 1's blind month recovery, and the prereg says so in advance.
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

PREREG = "docs/phase2_prereg_2026-08-21.md"
STATUS = "exploratory Phase-2 under docs/phase2_prereg_2026-08-21.md"

# ---------------------------------------------------------------------------
# Z2 -- label set, verbatim from src/gfdisc/ontologies.py at 3cadb69
# ---------------------------------------------------------------------------
ZODIAC = ["Aries", "Taurus", "Gemini", "Cancer", "Leo", "Virgo",
          "Libra", "Scorpio", "Sagittarius", "Capricorn", "Aquarius", "Pisces"]

# ---------------------------------------------------------------------------
# Z3 -- templates. (prefix, suffix); the sign goes between them with one space.
# Constraints, all checked in `verify_templates`: no template contains another member
# of the label set; no template contains a month, weekday or season word; every suffix
# is >= 3 tokens so the "suffix" and "final" readouts are distinct positions; the
# astrological disambiguator sits in the PREFIX (prereg 3.3) and is varied across the
# sixteen templates (star sign / horoscope / born under / zodiac / birth chart /
# astrologer) so no single frame carries the result.
# ---------------------------------------------------------------------------
TEMPLATES = [
    ("Her star sign is", " and she reads it daily"),
    ("People born under", " are said to be like that"),
    ("His horoscope says", " will have good fortune"),
    ("The zodiac sign", " appears on the printed chart"),
    ("She was born under the sign of", " according to the old chart"),
    ("The astrologer wrote about", " in the newspaper column"),
    ("A person whose sign is", " tends to be curious"),
    ("The chart shows the symbol for", " near the centre"),
    ("My friend is", " and believes in it"),
    ("The horoscope column for", " was quite short"),
    ("Astrologers place", " between the other signs"),
    ("The star sign", " is printed on the card"),
    ("He looked up the sign", " in the old book"),
    ("The magazine lists", " under the star signs"),
    ("Her birth chart is dominated by", " and its ruling planet"),
    ("The zodiac wheel includes", " among the twelve signs"),
]
READOUTS = ("token", "suffix", "final")

# ---------------------------------------------------------------------------
# Z6 -- exclusion rule. Written into the JSON verbatim and applied to BOTH pools.
# ---------------------------------------------------------------------------
CALENDAR_WORDS = [
    "january", "february", "march", "april", "may", "june", "july", "august",
    "september", "october", "november", "december",
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
    "spring", "summer", "autumn", "fall", "winter",
]
ZODIAC_SENSE_EXCLUSIONS = [
    # the labels themselves
    "aries", "taurus", "gemini", "cancer", "leo", "virgo", "libra", "scorpio",
    "sagittarius", "capricorn", "aquarius", "pisces",
    # the zodiac symbol animals / figures
    "ram", "bull", "twins", "twin", "crab", "lion", "lioness", "virgin", "maiden",
    "scales", "scale", "balance", "scorpion", "archer", "centaur", "goat", "fish",
    "waterbearer", "water", "bearer",
    # astrology vocabulary
    "star", "sign", "horoscope", "zodiac", "astrology", "astrologer", "planet",
    "moon", "sun", "chart", "constellation", "birth", "fortune", "destiny", "fate",
    "element", "earth", "air", "fire",
    # first names that are also labels or calendar words (the names pool needs these)
    "dawn", "aurora", "sunny",
]
EXCLUSIONS = sorted(set(CALENDAR_WORDS) | set(ZODIAC_SENSE_EXCLUSIONS))

# Common-noun pool (prereg 4.2(b)). Inherited from scripts/24_colors_existence.py, whose
# words were already screened against that script's own ontology; screened again here
# against Z6.
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
# Personal first names (prereg 4.2(c), registered for zodiac). Capitalized proper nouns
# in the same slot as the signs. April/May/June/Summer/Leo/Dawn are deliberately LEFT IN
# the candidate list so the Z6 exclusion rule visibly fires on them in the audit rather
# than being silently pre-filtered.
NAME_POOL = [
    "Sarah", "Michael", "David", "Emma", "James", "Laura", "Thomas", "Rachel",
    "Daniel", "Hannah", "Peter", "Alice", "Robert", "Claire", "Andrew", "Julia",
    "Simon", "Helen", "Martin", "Diana", "Richard", "Nicola", "Steven", "Karen",
    "Paul", "Linda", "George", "Susan", "Edward", "Margaret", "Henry", "Elizabeth",
    "Charles", "Catherine", "William", "Victoria", "Anthony", "Patricia", "Joseph",
    "Barbara", "Kevin", "Sandra", "Brian", "Donna", "Gary", "Carol", "Frank",
    "Sharon", "Dennis", "Angela", "Larry", "Melissa", "Jeffrey", "Rebecca", "Ryan",
    "Stephanie", "Jacob", "Nicole", "Nathan", "Amanda", "Adam", "Michelle", "Justin",
    "Brandon", "Amy", "Samuel", "Anna", "Benjamin", "Ruth", "Gregory", "Marie",
    "Patrick", "Janet", "Jack", "Katherine", "Dylan", "Christine", "Oliver", "Sophie",
    "Harry", "Grace", "Lucas", "Chloe", "Ethan", "Olivia", "Mason", "Isabella",
    "Logan", "Charlotte", "Owen", "Amelia", "Caleb", "Ian", "Fiona", "Neil",
    "Wendy", "Colin", "Derek", "Sheila", "Malcolm", "Roger", "Vincent", "Trevor",
    "Alan", "Bernard", "Gordon", "Stuart", "Douglas", "Graham", "Keith", "Leonard",
    # deliberately included so Z6 can be seen to fire:
    "April", "May", "June", "Summer", "Leo", "Dawn",
]

N_BACKGROUND = 2000
N_PERM_ORDER = 20000
N_SHUFFLE_CONTROL = 1000
N_GAP_NULL = 20000
N_PC_SECONDARY = 4
COUNT_FLOOR = 100          # the frozen discovery pool's floor
COUNT_FLOOR_EXTENDED = 30  # the extended pool's floor
TOKENS_I32 = "cache/sparse_big/layer_0/tokens.i32"

# ---------------------------------------------------------------------------
# Closure null: one shared table per k.
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
# Ordering, vectorised, with the flip group exposed (Z7).
# ---------------------------------------------------------------------------
def order_stats(ang, label_index, K, flips=(1, -1), n_perm=N_PERM_ORDER, seed=0, block=5000):
    """Max Spearman over the allowed symmetries, calibrated by shuffled labels.

    Same construction as fits.cyclic_order_score (stable-argsort tie-break replaced by a
    random jitter, add-one p-value), vectorised over the permutation null and with
    `flips` exposed so the rotation-only statistic can be read off separately. With
    flips=(1,-1) it must reproduce cyclic_order_score exactly; asserted in
    `check_against_reference`.
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
                flips=list(int(f) for f in flips))


def check_against_reference(k=12, seed=0):
    """order_stats(flips=(1,-1)) must equal fits.cyclic_order_score on the same input."""
    rng = np.random.default_rng(seed)
    out = []
    for t in range(5):
        ang = rng.uniform(-np.pi, np.pi, size=k)
        lab = np.arange(k)
        a = cyclic_order_score(ang, lab, k, n_perm=2000, seed=t)
        b = order_stats(ang, lab, k, n_perm=2000, seed=t)
        out.append(dict(trial=t, rho_ref=a["rho"], rho_new=b["rho"],
                        sym_ref=list(a["symmetry"]), sym_new=b["symmetry"],
                        rho_match=bool(abs(a["rho"] - b["rho"]) < 1e-12)))
    return dict(k=k, trials=out, passed=bool(all(x["rho_match"] for x in out)))


# ---------------------------------------------------------------------------
# The one analysis routine. Signs, backgrounds and synthetics all go through it.
# ---------------------------------------------------------------------------
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
        convs[name] = d
    out["conventions"] = convs
    return out, Vt[:2], xy


# ---------------------------------------------------------------------------
# Z12 -- widened, label-free plane search. One SVD, six circle fits.
# ---------------------------------------------------------------------------
def plane_scan(P, n_pc=N_PC_SECONDARY):
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
    """(primary gap ratio, primary p_uniform, best-plane gap ratio, radius ratio)."""
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
               p_closed_uniform_SELECTION_INVALID=p_closed(k, np.radians(best["max_gap_deg"])),
               all_planes=[dict(pc=[int(x) for x in p["pc"]], gap_ratio=p["gap_ratio"],
                                radius_ratio=p["radius_ratio"]) for p in planes])
    if do_order and label_index is not None and not degenerate(P):
        ang = angles(best["xy"], best["cx"], best["cy"])
        out["order"] = order_stats(ang, label_index, int(K if K is not None else k),
                                   flips=(1, -1), n_perm=N_PERM_ORDER, seed=seed)
        out["order_note"] = ("plane selection is label-free, so this shuffled-label null "
                             "is exact for the selected plane")
    return out, best["basis"]


# ---------------------------------------------------------------------------
# Synthetic battery at k = 12, through the SAME code path the signs use.
# ---------------------------------------------------------------------------
def synthetics(d=D_MODEL, k=12, noise=0.15, n_bg=N_BACKGROUND, seed=0):
    """Planted-shape battery. The dissociation this project exists to enforce is the arc
    and line rows: both order essentially perfectly and neither may close.

    Run at k = 12, the zodiac's arity, because prereg section 6 showed closure power is
    non-monotone in k and a threshold calibrated at one k does not transfer.
    """
    rng = np.random.default_rng(seed)
    u, v = rng.normal(size=d), rng.normal(size=d)
    u /= np.linalg.norm(u)
    v -= v @ u * u
    v /= np.linalg.norm(v)
    lab = np.arange(k)

    def amb(scale):
        """Isotropic noise of per-point NORM `scale`, not per-coordinate scale."""
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

    perm = rng.permutation(k)
    out["planted_ring_scrambled_labels"] = row(ring, 2, labels=perm)

    ta = np.linspace(0, np.radians(200), k)
    arc = np.outer(np.cos(ta), u) + np.outer(np.sin(ta), v) + amb(noise)
    out["planted_arc_200deg"] = row(arc, 3)

    # The ordinal control, planted as a thin sausage in the fitted plane rather than as
    # "line + isotropic noise" (which is a line plus a NOISE direction and scrambles the
    # angular order). Matches the measured number-words control: gap_ratio 5.1-7.4,
    # rho >= 0.99, never closes.
    line = (np.outer(np.linspace(-1, 1, k), u)
            + np.outer(0.05 * rng.normal(size=k), v)
            + amb(0.34))
    out["planted_line"] = row(line, 4)
    out["planted_line_note"] = ("thin sausage in the fitted plane; matches the measured "
                                "number-words ordinal control (never closes)")

    # Z12 validation: a ring buried under a dominant third component. The primary top-2
    # plane must miss it; the widened search must find it. The ramp's per-point
    # COEFFICIENTS are orthogonalised against the ring's, not just its direction against
    # the ring's plane -- with only k points, vectors orthogonal in 1152 dimensions are
    # generally not orthogonal in the k-dimensional sample space.
    w = rng.normal(size=d)
    w -= (w @ u) * u + (w @ v) * v
    w /= np.linalg.norm(w)
    ramp = np.linspace(-1, 1, k)
    B = np.column_stack([np.ones(k), np.cos(th), np.sin(th)])
    ramp = ramp - B @ np.linalg.lstsq(B, ramp, rcond=None)[0]
    ramp = ramp / max(ramp.std(), 1e-12)
    buried = (2.0 * np.outer(ramp, w) + np.outer(np.cos(th), u)
              + np.outer(np.sin(th), v) + amb(noise))
    out["buried_ring"] = row(buried, 5)

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
             "anti-conservative on 1152-dimensional clouds at this k; the matched-word "
             "background is the tier that decides")

    R, RS, AR, LN, BR = (out["planted_ring"], out["planted_ring_scrambled_labels"],
                         out["planted_arc_200deg"], out["planted_line"],
                         out["buried_ring"])
    out["passed"] = bool(
        R["p_closed"] <= 0.05 and R["p_vs_background"] <= 0.05 and R["p_order"] <= 0.05
        and RS["p_vs_background"] <= 0.05 and RS["p_order"] > 0.05
        and AR["p_closed"] > 0.05 and AR["p_vs_background"] > 0.05 and AR["p_order"] <= 0.05
        and LN["p_closed"] > 0.05 and LN["p_vs_background"] > 0.05 and LN["p_order"] <= 0.05
        and out["pure_noise"]["n_sig_closed_background"] <= 1
        and BR["p_vs_background"] > 0.05 and BR["secondary"]["p_vs_background"] <= 0.05
        and AR["secondary"]["p_vs_background"] > 0.05
        and LN["secondary"]["p_vs_background"] > 0.05
        and out["pure_noise"]["n_sig_closed_background_secondary"] <= 1)
    return out


# ---------------------------------------------------------------------------
# Z0 -- blind-arm feasibility gate, re-derived from counts only.
# ---------------------------------------------------------------------------
def blind_feasibility(tok, repo_root):
    """Counts-only re-derivation of the prereg 5.2 zodiac pool table.

    np.fromfile on cache/sparse_big/layer_0/tokens.i32 -> np.bincount. No SAE indices,
    no values, no means, no geometry, and results/token_type_geometry.json is NOT opened
    -- not its partitions, not its catalogue, not verdict.other_rings.
    """
    path = Path(repo_root) / TOKENS_I32
    toks = np.fromfile(path, dtype=np.int32)
    counts = np.bincount(toks)

    def cnt(surface):
        ids = tok.encode(surface, add_special_tokens=False)
        if len(ids) != 1:
            return dict(form=surface, n_tokens=len(ids), ids=[int(i) for i in ids],
                        count=None, multi_token=True)
        i = int(ids[0])
        return dict(form=surface, n_tokens=1, ids=[i],
                    count=int(counts[i]) if i < len(counts) else 0, multi_token=False)

    rows = []
    for w in ZODIAC:
        variants = [cnt(w), cnt(" " + w), cnt(w.lower()), cnt(" " + w.lower())]
        single = [v for v in variants if not v["multi_token"]]
        best = max(single, key=lambda v: v["count"]) if single else None
        rows.append(dict(sign=w, variants=variants, best=best,
                         best_count=(best["count"] if best else 0),
                         over_floor=bool(best is not None and best["count"] >= COUNT_FLOOR),
                         over_extended=bool(best is not None
                                            and best["count"] >= COUNT_FLOOR_EXTENDED)))
    n_over = sum(r["over_floor"] for r in rows)
    n_ext = sum(r["over_extended"] for r in rows)
    need = int(np.ceil(2 / 3 * len(ZODIAC)))
    return dict(
        arm="blind (frozen label-blind partitions)",
        verdict="untestable",
        registered_in_advance=True,
        prereg_sections=["3.3", "4.1", "5.2", "5.6", "7"],
        method=("np.fromfile(cache/sparse_big/layer_0/tokens.i32, int32) -> np.bincount; "
                "counts only. Best-variant rule per the frozen artifact: highest "
                "cache-count qualifying variant, never summed across variants."),
        n_cache_rows=int(len(toks)), count_floor=COUNT_FLOOR,
        count_floor_extended=COUNT_FLOOR_EXTENDED,
        n_over_floor=int(n_over), n_over_extended_floor=int(n_ext),
        n_labels=len(ZODIAC), n_required=need,
        gate_passed=bool(n_over >= need),
        rows=rows,
        catalogue_opened=False,
        catalogue_note=("results/token_type_geometry.json was not opened by this script: "
                        "not the partitions, not the 51-cluster catalogue, not "
                        "verdict.other_rings. With 1 of 12 labels over the floor there is "
                        "no registered descriptive placement to report (the prereg "
                        "registers one only for the four compass cardinals), so opening it "
                        "would be exposure with no registered use."),
        forbidden=("prereg section 7 forbids reporting a blind-arm negative for zodiac; "
                   "`untestable` is not a negative"),
        polysemy_note=("the single sign over the >=100 floor is ` cancer`, whose corpus "
                       "sense is overwhelmingly the disease, so even that one label is not "
                       "a zodiac label"))


# ---------------------------------------------------------------------------
# Activations
# ---------------------------------------------------------------------------
def token_audit(tok, surfaces):
    """Which surfaces survive as a single token, and which do not."""
    keep, drop = [], []
    for s in surfaces:
        ids = tok.encode(s, add_special_tokens=False)
        rec = dict(surface=s, n_tokens=len(ids), ids=[int(i) for i in ids])
        (keep if len(ids) == 1 else drop).append(rec)
    return keep, drop


def build_and_verify(tok, prefix, suffix, word, word_id, bos):
    """Manual ids, then assert the natural tokenisation of the whole sentence agrees."""
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
            ci = 1 + len(pre)
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


def principal_angles_deg(A, B):
    """Principal angles between two 2-D subspaces given as [2, d] row bases."""
    Qa = np.linalg.qr(np.asarray(A, dtype=np.float64).T)[0]
    Qb = np.linalg.qr(np.asarray(B, dtype=np.float64).T)[0]
    s = np.clip(np.linalg.svd(Qa.T @ Qb, compute_uv=False), -1.0, 1.0)
    return [float(np.degrees(np.arccos(x))) for x in s]


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
    ap.add_argument("--out", default="results/phase2_zodiac.json")
    ap.add_argument("--layers", default=",".join(str(i) for i in range(26)))
    ap.add_argument("--n-background", type=int, default=N_BACKGROUND)
    ap.add_argument("--chunk", type=int, default=48)
    a = ap.parse_args()
    t0 = time.time()
    layers = [int(x) for x in a.layers.split(",")]
    repo_root = Path(__file__).resolve().parents[1]

    print("=" * 78)
    print("PHASE 2 -- ZODIAC. Binding prereg:", PREREG)
    print("DECLARATIONS (fixed before any activation is read):")
    print(f"  Z0 blind arm     : untestable, registered in advance (prereg 3.3/5.2). "
          f"Catalogue NOT opened.")
    print(f"  Z2 labels        : {[' ' + w for w in ZODIAC]}")
    print(f"  Z3 readouts      : token (embedding-confounded) | suffix (PRIMARY, "
          f"computed) | final (tertiary)")
    print(f"  Z4 discriminator : closure. Ordering alone is never evidence.")
    print(f"  Z5 nulls         : uniform angles ({N_GAP_NULL}); {a.n_background} matched "
          f"noun draws; {a.n_background} personal-name draws; {N_PERM_ORDER} shuffled "
          f"labels + {N_SHUFFLE_CONTROL}-shuffle control")
    print(f"  Z8 confound      : zodiac order IS calendar order -- any positive is "
          f"partially convention-confounded")
    print(f"  Z9 cell verdicts : zodiac_circle | closed_ring_unordered | "
          f"ordered_not_closed | nothing | degenerate")
    print(f"  Z10 ontology rule: suffix readout only, BH-corrected, plus the "
          f"names-pool tightening")
    print(f"  Z12 secondary    : widened plane search, declared here (NOT in the "
          f"prereg), verdict kept separate")
    print("=" * 78, flush=True)

    ref = check_against_reference(k=12)
    print(f"order_stats vs fits.cyclic_order_score at k=12: "
          f"{'MATCH' if ref['passed'] else 'MISMATCH'} on rho over 5 trials", flush=True)
    if not ref["passed"]:
        raise SystemExit("vectorised ordering statistic disagrees with the reference")

    syn = synthetics(k=12, n_bg=a.n_background)
    print("synthetic battery (k=12, the zodiac's arity):")
    for key in ("planted_ring", "planted_ring_scrambled_labels", "planted_arc_200deg",
                "planted_line", "buried_ring"):
        r = syn[key]
        print(f"  {key:32s} gapR {r['gap_ratio']:6.2f}  pUnif {r['p_closed']:.4f}  "
              f"pBg {r['p_vs_background']:.4f}  rho {r['rho']:+.3f}  "
              f"pOrd {r['p_order']:.4f}  |  2ndPlane {str(r['secondary']['selected_plane']):>6} "
              f"pBg2 {r['secondary']['p_vs_background']:.4f}")
    uc = syn["uniform_null_calibration"]
    print(f"  uniform-angle null vs {uc['n_draws']} isotropic clouds (k=12, d={syn['d']}): "
          f"mean gapR {uc['gap_ratio_mean']:.2f}, false acceptance at p<0.05 = "
          f"{uc['uniform_null_false_acceptance_at_p05']:.3f}")
    print(f"  pure_noise 5 draws: {syn['pure_noise']['n_sig_closed_background']}/5 "
          f"background-closure trips, {syn['pure_noise']['n_sig_order']}/5 ordering trips "
          f"-> {'PASS' if syn['passed'] else 'FAIL'}", flush=True)
    if not syn["passed"]:
        raise SystemExit("synthetic battery failed at k=12; not running on activations")

    model, tok = load_model(token=os.environ.get("HF_TOKEN"))
    bos = tok.bos_token_id

    # ---- Z0 blind feasibility gate (counts only) --------------------------
    blind = blind_feasibility(tok, repo_root)
    print(f"\nZ0 BLIND GATE: {blind['n_over_floor']}/12 signs >= {COUNT_FLOOR}, "
          f"{blind['n_over_extended_floor']}/12 >= {COUNT_FLOOR_EXTENDED}, "
          f"need {blind['n_required']} -> verdict {blind['verdict'].upper()} "
          f"(catalogue not opened)")
    for r in blind["rows"]:
        b = r["best"]
        print(f"    {r['sign']:<12} best {b['form']!r:<16} count "
              f"{b['count'] if b else 'n/a'}")
    print(flush=True)

    # ---- Z2 / Z6 audits ---------------------------------------------------
    sign_keep, sign_drop = token_audit(tok, [" " + w for w in ZODIAC])
    sign_words = [x["surface"] for x in sign_keep]
    if sign_drop:
        raise SystemExit(f"zodiac labels not single-token: {sign_drop}")
    K = len(sign_words)

    excl = set(EXCLUSIONS)
    noun_cand = [w for w in dict.fromkeys(NOUN_POOL) if w.lower() not in excl]
    noun_excluded = [w for w in dict.fromkeys(NOUN_POOL) if w.lower() in excl]
    name_cand = [w for w in dict.fromkeys(NAME_POOL) if w.lower() not in excl]
    name_excluded = [w for w in dict.fromkeys(NAME_POOL) if w.lower() in excl]
    noun_keep, noun_drop = token_audit(tok, [" " + w for w in noun_cand])
    name_keep, name_drop = token_audit(tok, [" " + w for w in name_cand])
    noun_words = [x["surface"] for x in noun_keep]
    name_words = [x["surface"] for x in name_keep]
    print(f"Z2 audit: k={K} signs, all single-token")
    print(f"Z6 audit: nouns {len(noun_words)} kept "
          f"({len(noun_excluded)} excluded by rule {noun_excluded}, "
          f"{len(noun_drop)} multi-token)")
    print(f"Z6 audit: names {len(name_words)} kept "
          f"({len(name_excluded)} excluded by rule {name_excluded}, "
          f"{len(name_drop)} multi-token)", flush=True)

    # ---- template verification -------------------------------------------
    # Manual ids must equal the natural tokenisation of the whole sentence, for EVERY
    # word x template. Label mismatches abort (they are the labels); pool mismatches
    # drop the word and are recorded -- mechanical and label-blind, so the rule cannot
    # select a background.
    def tokenises_everywhere(surface):
        w = surface[1:]
        wid = tok.encode(surface, add_special_tokens=False)[0]
        for prefix, suffix in TEMPLATES:
            m, nat, _, _ = build_and_verify(tok, prefix, suffix, w, wid, bos)
            if m != nat:
                return False, dict(word=w, prefix=prefix, manual=m, natural=nat)
        return True, None

    label_bad = []
    for s in sign_words:
        ok, info = tokenises_everywhere(s)
        if not ok:
            label_bad.append(info)
    pool_bad = []
    for s in noun_words + name_words:
        ok, info = tokenises_everywhere(s)
        if not ok:
            pool_bad.append(info)
    dropped_pool = {" " + d["word"] for d in pool_bad}
    noun_words = [w for w in noun_words if w not in dropped_pool]
    name_words = [w for w in name_words if w not in dropped_pool]

    # Template content constraints (Z3), checked rather than asserted.
    lower_signs = {w.lower() for w in ZODIAC}
    tmpl_report, tmpl_bad = [], []
    for prefix, suffix in TEMPLATES:
        text = (prefix + " " + suffix).lower()
        toks_txt = set(text.replace(".", " ").replace(",", " ").split())
        hits_sign = sorted(toks_txt & lower_signs)
        hits_cal = sorted(toks_txt & set(CALENDAR_WORDS))
        pre = tok.encode(prefix, add_special_tokens=False)
        suf = tok.encode(suffix, add_special_tokens=False)
        rec = dict(prefix=prefix, suffix=suffix, n_prefix_tokens=len(pre),
                   n_suffix_tokens=len(suf), sign_index=1 + len(pre),
                   suffix_index=2 + len(pre), final_index=1 + len(pre) + len(suf),
                   first_post_sign_token=tok.decode(suf[:1]),
                   contains_label_word=hits_sign, contains_calendar_word=hits_cal)
        tmpl_report.append(rec)
        if hits_sign or hits_cal or len(suf) < 3:
            tmpl_bad.append(rec)
    if tmpl_bad:
        raise SystemExit(f"template constraint violated: {tmpl_bad}")

    tokenisation_audit = dict(
        n_words_checked=len(sign_words) + len(noun_words) + len(name_words),
        n_templates=len(TEMPLATES),
        label_surfaces=sign_words, label_mismatches=label_bad,
        pool_words_dropped=sorted(dropped_pool), pool_mismatch_detail=pool_bad,
        rule="label mismatch aborts; pool mismatch drops the word (mechanical, label-blind)")
    print(f"tokenisation audit: "
          f"{(len(sign_words) + len(noun_words) + len(name_words)) * len(TEMPLATES)} "
          f"word x template checks, {len(label_bad)} label mismatches, "
          f"{len(dropped_pool)} pool words dropped {sorted(dropped_pool)}", flush=True)
    if label_bad:
        raise SystemExit(f"label words with off-manifold ids: {label_bad}")
    if len(noun_words) < 40 or len(name_words) < 40:
        raise SystemExit(f"background pools too small after the drop rule: "
                         f"{len(noun_words)} nouns, {len(name_words)} names")

    words = sign_words + noun_words + name_words
    word_ids = [tok.encode(w, add_special_tokens=False)[0] for w in words]
    idx = {w: i for i, w in enumerate(words)}
    sign_idx = np.array([idx[w] for w in sign_words])
    noun_idx = np.array([idx[w] for w in noun_words])
    name_idx = np.array([idx[w] for w in name_words])

    # ---- forwards ---------------------------------------------------------
    print(f"\ncollecting {len(words)} words x {len(TEMPLATES)} templates x "
          f"{len(layers)} layers x 3 readouts ...", flush=True)
    tf = time.time()
    M = collect(model, tok, words, word_ids, layers, chunk=a.chunk)
    fwd_sec = time.time() - tf
    print(f"forwards done in {fwd_sec:.1f}s", flush=True)
    del model
    if DEVICE == "mps":
        torch.mps.empty_cache()

    label_index = np.arange(K)
    ref_layer = layers[0]
    _, basis_ref_token, _ = analyse(M["token"][ref_layer][sign_idx], label_index, K,
                                    seed=0, do_order=False, both_conventions=False)

    rows, jack_rows = [], []
    print(f"\n{'PRIMARY (pre-declared, top-2 PCA plane)':<86}| {'Z12 widened plane':<32}")
    print(f"{'readout':>8} {'L':>3} {'top2':>6} {'radR':>7} {'gapR':>7} {'pUnif':>8} "
          f"{'pNoun':>8} {'FAnoun':>7} {'pName':>8} {'rho':>7} {'pOrd':>8} {'ang(ref)':>9} "
          f"| {'plane':>6} {'gapR':>6} {'pNoun':>7} {'rho':>6} {'pOrd':>7}  verdict",
          flush=True)
    for ro in READOUTS:
        for l in layers:
            P = M[ro][l][sign_idx]
            res, basis, xy = analyse(P, label_index, K, seed=l, do_order=True)
            A = res["conventions"]["A"]
            B = res["conventions"]["B"]
            sign_check = dict(
                gap_ratio_identical=bool(abs(A["gap_ratio"] - B["gap_ratio"]) < 1e-9),
                rho_identical=(None if res["degenerate"]
                               else bool(abs(A["order"]["rho"] - B["order"]["rho"]) < 1e-12)),
                rot_only_rho_A=(None if res["degenerate"] else A["order_rotation_only"]["rho"]),
                rot_only_rho_B=(None if res["degenerate"] else B["order_rotation_only"]["rho"]))

            row = dict(readout=ro, layer=l, k=K,
                       **{kk: res[kk] for kk in ("degenerate", "top2_var", "pc_var")},
                       conventions=res["conventions"], sign_convention_check=sign_check)

            sec, sec_basis = secondary_analyse(P, label_index, K, seed=l,
                                               do_order=not res["degenerate"])

            def bg(pool_idx, n_draws, seed):
                """Identical pipeline, same arity, same positions -- only the words differ.

                One SVD per draw serves both tiers: the primary (PC1, PC2) gap ratio and
                the min-over-6-planes gap ratio the Z12 search would have selected, so
                every background draw gets exactly the freedom the observation was given.
                """
                r = np.random.default_rng(seed)
                gr = np.empty(n_draws); pc = np.empty(n_draws)
                rr = np.empty(n_draws); gb = np.empty(n_draws)
                for i in range(n_draws):
                    pick = pool_idx[r.choice(len(pool_idx), size=K, replace=False)]
                    gr[i], pc[i], gb[i], rr[i] = fast_closure(M[ro][l][pick])
                return dict(n_draws=int(n_draws), gap_ratio_mean=float(gr.mean()),
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
            b_name = bg(name_idx, a.n_background, 2000 + l)
            sec["p_vs_noun_background"] = b_noun["secondary_best_plane"]["p_vs_background"]
            sec["p_vs_name_background"] = b_name["secondary_best_plane"]["p_vs_background"]
            row["secondary_plane_search"] = sec
            row["principal_angles_secondary_to_ref_token_plane_deg"] = principal_angles_deg(
                sec_basis, basis_ref_token)
            row["background_nouns"] = b_noun
            row["background_names"] = b_name
            pa = principal_angles_deg(basis, basis_ref_token)
            row["principal_angles_to_ref_token_plane_deg"] = pa

            # ---- Z9 verdict -- PRIMARY statistic only -----------------------
            if res["degenerate"]:
                verdict = "degenerate"
            else:
                closed = (A["p_closed_uniform"] < 0.05 and b_noun["p_vs_background"] < 0.05)
                ordered = A["order"]["p"] < 0.05
                verdict = ("zodiac_circle" if closed and ordered else
                           "closed_ring_unordered" if closed else
                           "ordered_not_closed" if ordered else "nothing")
            row["verdict"] = verdict
            if res["degenerate"]:
                verdict_sec = "degenerate"
            else:
                closed_s = sec["p_vs_noun_background"] < 0.05
                ordered_s = sec["order"]["p"] < 0.05
                verdict_sec = ("zodiac_circle_widened_search" if closed_s and ordered_s else
                               "closed_ring_unordered" if closed_s else
                               "ordered_not_closed" if ordered_s else "nothing")
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
                  f"{b_name['p_vs_background']:8.4f} "
                  f"{'   n/a' if rho is None else f'{rho:+7.3f}'} "
                  f"{'     n/a' if pord is None else f'{pord:8.4f}'} "
                  f"{pa[0]:9.1f} | {str(sec['selected_plane']):>6} "
                  f"{sec['gap_ratio']:6.2f} {sec['p_vs_noun_background']:7.4f} "
                  f"{'   n/a' if srho is None else f'{srho:+6.3f}'} "
                  f"{'    n/a' if spo is None else f'{spo:7.4f}'}  {verdict}", flush=True)

            # ---- leave-one-out jackknife where closure passes ---------------
            if not res["degenerate"] and A["p_closed_uniform"] < 0.05:
                jk = []
                for j in range(K):
                    keep = [i for i in range(K) if i != j]
                    q, _, _ = analyse(P[keep], np.array(keep), K, seed=l,
                                      do_order=False, both_conventions=False)
                    qa = q["conventions"]["A"]
                    jk.append(dict(dropped=sign_words[j], gap_ratio=qa["gap_ratio"],
                                   p_closed_uniform=qa["p_closed_uniform"]))
                jack_rows.append(dict(readout=ro, layer=l, leave_one_out=jk,
                                      worst=max(jk, key=lambda x: x["p_closed_uniform"]),
                                      n_still_closed=int(sum(x["p_closed_uniform"] < 0.05
                                                             for x in jk))))

    # ---- Z11 multiplicity --------------------------------------------------
    live = [r for r in rows if not r["degenerate"]]
    n_cells = len(live)

    def fdr(pvals, label, n_null):
        """BH-FDR plus the floor the null size imposes.

        Every p here is an add-one empirical fraction and cannot fall below
        1/(n_null+1). BH's rank-1 threshold is alpha/n_cells. When the floor sits above
        it, "0 survive BH" can mean the test was unreachable rather than that nothing was
        there, so both numbers are reported next to each other.
        """
        p = np.asarray(pvals, dtype=float)
        keep = benjamini_hochberg(p, alpha=0.05) if len(p) else np.zeros(0, bool)
        floor = 1.0 / (n_null + 1.0)
        rank1 = 0.05 / max(len(p), 1)
        return dict(statistic=label, n_cells=int(len(p)), n_null_draws=int(n_null),
                    n_raw_below_05=int((p < 0.05).sum()),
                    expected_by_chance_at_05=float(0.05 * len(p)),
                    p_value_floor=float(floor), bh_rank1_threshold=float(rank1),
                    isolated_cell_can_survive_bh=bool(floor <= rank1),
                    min_cells_at_floor_to_survive_bh=(int(np.ceil(floor / rank1))
                                                      if rank1 > 0 else None),
                    n_at_floor=int((p <= floor + 1e-12).sum()),
                    n_surviving_bh_fdr=int(keep.sum()),
                    bh_keep=[bool(x) for x in keep],
                    surviving=[dict(readout=live[i]["readout"], layer=live[i]["layer"],
                                    p=float(p[i])) for i in np.flatnonzero(keep)])

    nb = a.n_background
    mult = dict(
        n_cells=int(n_cells),
        note="3 readouts x 26 layers. An isolated pass at alpha = 0.05 is expected by "
             "chance and is not a finding (prereg 4.4); BH-FDR across all live cells is "
             "the honest count. Each row carries its p-value floor.",
        ordering_primary=fdr([r["conventions"]["A"]["order"]["p"] for r in live],
                             "ordering p, primary plane", N_PERM_ORDER),
        closure_uniform_primary=fdr([r["conventions"]["A"]["p_closed_uniform"] for r in live],
                                    "closure p vs uniform angles, primary plane", N_GAP_NULL),
        closure_background_primary=fdr([r["background_nouns"]["p_vs_background"] for r in live],
                                       "closure p vs matched noun background, primary plane",
                                       nb),
        closure_names_primary=fdr([r["background_names"]["p_vs_background"] for r in live],
                                  "closure p vs personal-name background, primary plane", nb),
        closure_background_secondary=fdr(
            [r["secondary_plane_search"]["p_vs_noun_background"] for r in live],
            "closure p vs matched noun background, Z12 selected plane", nb),
        ordering_secondary=fdr([r["secondary_plane_search"]["order"]["p"] for r in live],
                               "ordering p, Z12 selected plane", N_PERM_ORDER),
    )

    # ---- Z10 ontology-level verdict ---------------------------------------
    bh_close = mult["closure_background_primary"]["bh_keep"]
    bh_order = mult["ordering_primary"]["bh_keep"]
    qualifying, ordered_only = [], []
    for i, r in enumerate(live):
        if r["readout"] != "suffix":
            continue
        A = r["conventions"]["A"]
        closed = (A["p_closed_uniform"] < 0.05
                  and r["background_nouns"]["p_vs_background"] < 0.05
                  and bh_close[i]
                  and r["background_names"]["p_vs_background"] < 0.05)
        ordered = A["order"]["p"] < 0.05 and bh_order[i]
        if closed:
            qualifying.append(dict(layer=r["layer"], ordered=bool(ordered),
                                   p_closed_uniform=A["p_closed_uniform"],
                                   p_noun=r["background_nouns"]["p_vs_background"],
                                   p_name=r["background_names"]["p_vs_background"],
                                   rho=A["order"]["rho"], p_order=A["order"]["p"]))
        elif ordered:
            ordered_only.append(dict(layer=r["layer"], rho=A["order"]["rho"],
                                     p_order=A["order"]["p"]))
    if any(q["ordered"] for q in qualifying):
        ont_verdict = "structure_exists_supervised_only"
    elif qualifying:
        ont_verdict = "closed_ring_unordered"
    elif ordered_only:
        ont_verdict = "ordered_not_closed"
    else:
        ont_verdict = "clean_negative" if syn["passed"] else "untestable"

    out = dict(
        status=STATUS,
        item="Phase-2 zodiac (ontology 3.3 of the frozen pre-registration)",
        script="scripts/30_phase2_zodiac.py",
        prereg=PREREG,
        generated=time.strftime("%Y-%m-%dT%H:%M:%S"),
        prediction_source="src/gfdisc/ontologies.py at commit 3cadb69 (2026-08-18 12:04:20 -0400)",
        declarations=dict(
            Z0_blind_arm="untestable, registered in advance at prereg 3.3/5.2; the frozen "
                         "catalogue is not opened",
            Z2_labels=ZODIAC,
            Z2_variant="' {W}' capitalized leading-space single-token form "
                       "(' Cancer', not ' cancer')",
            Z2_predicted_structure="closed 12-cycle in the committed calendar order "
                                   "Aries -> Taurus -> ... -> Pisces",
            Z3_readouts=dict(token="the sign token itself -- embedding-confounded",
                             suffix="first token AFTER the sign -- PRIMARY computed readout; "
                                    "the ontology verdict uses this readout only",
                             final="last token of the template -- tertiary, exploratory"),
            Z4_discriminator="closure; ordering alone is never evidence",
            Z5_nulls=dict(uniform_angles=N_GAP_NULL, background_draws=a.n_background,
                          background_primary="single-token common nouns, same arity, "
                                             "identical templates and positions "
                                             "(prereg 4.2b -- the null that matters)",
                          background_secondary="single-token personal first names "
                                               "(prereg 4.2c, registered for zodiac; "
                                               "capitalization/proper-noun matched)",
                          order_permutations=N_PERM_ORDER,
                          shuffle_control=N_SHUFFLE_CONTROL,
                          background_not_scored_for_ordering="random nouns and names have "
                                                             "no true cyclic order"),
            Z6_exclusion_rule="no background word may carry an established zodiac sense; "
                              "zodiac symbol animals/figures and astrology vocabulary are "
                              "excluded; every month, weekday and season word is excluded "
                              "from every pool so the pools cannot inherit this project's "
                              "other ontologies",
            Z6_exclusions=EXCLUSIONS,
            Z7_sign_conventions="both PC2 sign conventions reported; rho must be identical "
                                "(reflection-invariant), rotation-only rho exposes handedness",
            Z8_convention_confound="the registered zodiac order IS the calendar order, so "
                                   "any positive is partially convention-confounded and must "
                                   "be reported that way",
            Z9_cell_verdicts=dict(
                zodiac_circle="p_closed_uniform < 0.05 AND p_vs_noun_background < 0.05 AND "
                              "ordering p < 0.05",
                closed_ring_unordered="both closure tests pass, ordering does not",
                ordered_not_closed="ordering passes, closure does not",
                nothing="neither", degenerate="point cloud carries no usable geometry"),
            Z10_ontology_rule=dict(
                readout="suffix only",
                structure_exists_supervised_only="a suffix cell with p_closed_uniform<0.05, "
                                                 "p_vs_noun_background<0.05 surviving BH, "
                                                 "ordering p<0.05 surviving BH, and "
                                                 "p_vs_name_background<0.05",
                closed_ring_unordered="the same closure conditions without ordering",
                ordered_not_closed="a suffix cell whose ordering survives BH and no cell "
                                   "meets the closure conditions",
                clean_negative="none of the above and the synthetic battery passed",
                untestable="the blind arm, by Z0",
                names_clause_provenance="the personal-name clause is a TIGHTENING declared "
                                        "in this script before any activation was read; the "
                                        "prereg calls the noun pool the null that matters "
                                        "and the class-matched pool secondary. It can only "
                                        "make a positive harder."),
            Z11_multiplicity="BH-FDR within this ontology's 78-cell sweep (prereg 4.4), "
                             "reported with raw counts, the expected-by-chance count and "
                             "each statistic's p-value floor",
            Z12_secondary_provenance="the widened plane search is NOT in the prereg. It is "
                                     "copied from scripts/24_colors_existence.py D10 and "
                                     "declared in this file before any activation was read, "
                                     "not added after a negative. Its verdict is kept in a "
                                     "separate field and is never the registered outcome.",
            caveats_that_survive_a_positive=[
                "a closed ring at the suffix readout shows the information is present one "
                "token later; attention can copy an embedding forward, so it is not proof "
                "of recomputation. The principal angle to the earliest-layer token-readout "
                "plane is the only handle reported on that",
                "zodiac order IS calendar order, so any positive is partially "
                "convention-confounded",
                "this arm chose its label set by name; structure_exists_supervised_only is "
                "a weaker claim than a blind recovery, as the prereg states at 5.6",
                "layers 0-4 are embedding-adjacent (prereg 4.5): a hit there validates the "
                "pipeline and locates geometry near the input, it is not a depth claim",
            ],
        ),
        model="google/gemma-3-1b-pt", layers=layers, n_templates=len(TEMPLATES),
        templates=tmpl_report, tokenisation_audit=tokenisation_audit,
        blind_arm=blind,
        embedding_reference_plane=dict(
            readout="token", layer=int(ref_layer),
            note="the plane every principal_angles_* field is measured against: the sign "
                 "point cloud read at the sign token itself, at the earliest layer in this "
                 "sweep. A suffix-readout plane at a small angle to it is consistent with "
                 "an embedding copied forward."),
        token_audit=dict(signs_kept=sign_keep, signs_dropped=sign_drop,
                         noun_pool_kept=noun_words, noun_pool_multi_token=noun_drop,
                         noun_pool_excluded_by_rule=noun_excluded,
                         name_pool_kept=name_words, name_pool_multi_token=name_drop,
                         name_pool_excluded_by_rule=name_excluded),
        reference_check=ref, synthetics=syn,
        rows=rows, jackknife=jack_rows, multiplicity=mult,
    )
    out["verdict_counts"] = {
        ro: {name: [r["layer"] for r in rows if r["readout"] == ro and r["verdict"] == name]
             for name in ("zodiac_circle", "closed_ring_unordered", "ordered_not_closed",
                          "nothing", "degenerate")}
        for ro in READOUTS}
    out["verdict_counts_secondary"] = {
        ro: {name: [r["layer"] for r in rows
                    if r["readout"] == ro and r["verdict_secondary_plane_search"] == name]
             for name in ("zodiac_circle_widened_search", "closed_ring_unordered",
                          "ordered_not_closed", "nothing", "degenerate")}
        for ro in READOUTS}
    out["verdict"] = dict(
        ontology="zodiac", full_k=12,
        blind_arm=blind["verdict"],
        blind_arm_reason=f"{blind['n_over_floor']}/12 signs clear the >={COUNT_FLOOR} count "
                         f"floor, {blind['n_required']} needed; registered untestable in "
                         f"advance at prereg 3.3/5.2",
        supervised_arm=ont_verdict,
        supervised_qualifying_cells=qualifying,
        supervised_ordered_only_cells=ordered_only,
        power_demonstrated=bool(syn["passed"]),
        vocabulary=["recovered_blind (unreachable by construction for zodiac)",
                    "structure_exists_supervised_only", "closed_ring_unordered",
                    "ordered_not_closed", "clean_negative", "untestable"])
    out["runtime_sec"] = float(time.time() - t0)
    out["forward_sec"] = float(fwd_sec)

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=2, default=jsonable))

    print("\n" + "=" * 78)
    print("Z9 CELL VERDICTS -- PRIMARY pre-declared statistic")
    for ro in READOUTS:
        for name in ("zodiac_circle", "closed_ring_unordered", "ordered_not_closed",
                     "nothing", "degenerate"):
            ls = out["verdict_counts"][ro][name]
            if ls:
                print(f"  {ro:>8} {name:>24}: {len(ls):2d} layers {ls}")
    print("\nZ12 WIDENED PLANE SEARCH (declared here, not in the prereg)")
    for ro in READOUTS:
        for name in ("zodiac_circle_widened_search", "closed_ring_unordered",
                     "ordered_not_closed", "nothing", "degenerate"):
            ls = out["verdict_counts_secondary"][ro][name]
            if ls:
                print(f"  {ro:>8} {name:>28}: {len(ls):2d} layers {ls}")
    print(f"\nZ11 MULTIPLICITY over {n_cells} live cells "
          f"(expected by chance at alpha=0.05: {0.05 * n_cells:.1f})")
    for key in ("ordering_primary", "closure_uniform_primary", "closure_background_primary",
                "closure_names_primary", "ordering_secondary", "closure_background_secondary"):
        m = mult[key]
        print(f"  {key:30s} raw p<0.05: {m['n_raw_below_05']:2d}   survives BH-FDR: "
              f"{m['n_surviving_bh_fdr']:2d}   [p floor {m['p_value_floor']:.2e}, BH rank-1 "
              f"{m['bh_rank1_threshold']:.2e}, isolated cell reachable: "
              f"{m['isolated_cell_can_survive_bh']}]"
              + (f"   {[(s['readout'], s['layer']) for s in m['surviving']]}"
                 if m["surviving"] else ""))
    print(f"\nVERDICT  blind: {blind['verdict']}   supervised: {ont_verdict}")
    print(f"runtime {time.time() - t0:.1f}s -> wrote {a.out}")


if __name__ == "__main__":
    main()
