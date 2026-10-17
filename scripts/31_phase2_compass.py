"""PHASE 2 -- COMPASS DIRECTIONS (docs/phase2_prereg_2026-08-21.md, section 3.4).

Two arms, both registered in advance in the prereg. Nothing here invents a test.

  BLIND ARM. Reads ONLY the frozen artifacts (results/token_type_geometry.json:
    partitions + the 51-cluster ring catalogue). No re-clustering, no new thresholds,
    no new arms. Registered verdict for the 8-cycle: `untestable` -- 4 of 8 terms clear
    the >=100 count floor (all four intercardinals sit at 25-29), 4 < ceil(2/3 * 8) = 6,
    and a 4-member set is below the frozen screening band minimum of 5. That verdict is
    pre-registered (prereg 3.4 / 5.3 / 5.6) and is NOT recomputed here, nor may a
    blind-arm negative be reported (prereg section 7).
    What this arm does report: (a) whether any registered compass variant appears in any
    stored cluster membership in the frozen artifact, worded as ABSENCE FROM THE
    CATALOGUE and not as a negative; (b) the prereg section 2.1 exposure item -- the
    layer-3 "particle / direction" ring -- discharged by listing its membership verbatim.

  SUPERVISED ARM (PRIMARY for compass). Patterned element-for-element on the verified
    scripts/24_colors_existence.py: 26 layers x 3 readouts x 16 templates, suffix is the
    primary computed readout, closure against matched backgrounds is the discriminator,
    both SVD sign conventions, planted synthetics through the identical code path,
    BH within the 78-cell sweep.

WHAT IS DECLARED BEFORE ANY ACTIVATION IS READ (written into the output JSON under
"declarations" and printed to the log before the first forward pass):

  C1  Label set, from src/gfdisc/ontologies.py DIRECTIONS, committed 3cadb69
      (2026-08-18), in geographic order:
        north(0) northeast(45) east(90) southeast(135)
        south(180) southwest(225) west(270) northwest(315)
      Registered surface form for the supervised arm: the LEADING-SPACE LOWERCASE
      variant, which prereg 3.4 records as single-token for all eight. Capitalized
      leading-space forms (' North' ...) are collected in the same forwards as
      EXPLORATORY ONLY and never enter a verdict: prereg 3.4 registers the capitalized
      forms as carrying a place-name sense ("the North"), which is the declared compass
      hazard, so they are a contrast, not the label set.

  C2  k = 8 POWER GATE, applied at the REALIZED support before any verdict is read.
      This is the k = 7 lesson (results/ref_table_k7.json) recurring at k = 8: closure
      power collapses non-monotonically as labels drop out, and a threshold calibrated
      at one k does not transfer. prereg section 6 computes: full 8-ring ref_p 5.0e-05
      (testable); ONE dropout 0.0731 (NOT testable); two adjacent 0.4446 (no); two
      non-adjacent 0.0260 (yes). The full dropout table is recomputed here at the
      realized support with fits/13_rerun_gate.reference_ring, BEFORE the sweep, so a
      failure is never interpreted without knowing whether closure could have passed.
      CONSEQUENCE, declared now: if fewer than 8 direction words survive the
      tokenisation audit, the supervised arm is `untestable` and the run aborts. It does
      not report a negative on 7 points.

  C3  Readout positions. Every template is PREFIX + " " + WORD + SUFFIX.
        "token"  -- the direction token itself. EMBEDDING-CONFOUNDED readout: at early
                    layers this is (almost) the word's own embedding, so a ring here is
                    lexical geometry, not computation. Declared in advance as the
                    confound, not the finding.
        "suffix" -- PRIMARY COMPUTED READOUT: the first token AFTER the direction word.
        "final"  -- last token of the template. Tertiary, exploratory.

  C4  Closure is the discriminator. Ordering alone is NEVER evidence: this project
      measured random subsets ordering at ~0.16 and an ordinal control out-ordering
      real month cycles at every layer.

  C5  Nulls.
      (a) uniform random angles, k of them, 20,000 draws.
      (b) 2000 draws of k matched same-arity single-token common NOUNS through the
          identical templates at the identical positions -- the null that matters.
          Reported as a false-acceptance rate at alpha = 0.05 and as an empirical p.
      (c) secondary, class-matched pool registered by prereg 4.2(c) for compass:
          PLACE NAMES, 2000 draws. Compass words in this corpus carry a place-name
          sense, so the place-name pool is the sharper grammatical control.
      (d) ordering: 20,000 shuffled-label permutations under the same
          max-over-symmetries selection, plus an explicit 1,000-shuffle control block.
      Background draws are NOT scored for ordering: random words have no true cyclic
      order, so closure is the background criterion.
      DISCREPANCY WITH THE PREREG, RECORDED RATHER THAN MADE SILENTLY: prereg 4.2(b)
      registers 300 background draws. This script uses 2000. The reason is
      scripts/24_colors_existence.py's D12 amendment: an add-one empirical p is floored
      at 1/(n+1), so 300 draws floor it at 3.3e-3 while BH's rank-1 threshold over 78
      cells is 0.05/78 = 6.4e-4 -- no single cell could survive BH however clean it was.
      At 2000 the floor is 5.0e-4, below the rank-1 threshold. More draws only tighten
      the null and can only make a positive harder, never easier.

  C6  Background pool exclusion rule, declared before the pools are used: no background
      word may carry an established directional or compass sense (north/south/east/west
      and their compounds, left, right, up, down, upward, downward, forward, backward,
      inland, coastal, bearing, compass, ...), and no place name whose name contains a
      direction substring (Northampton, Westminster, Southport, Easton, ...). Calendar
      words are excluded from every pool so the pools cannot inherit this project's
      other ontologies. The full exclusion list is written into the JSON verbatim.

  C7  Both SVD sign conventions are reported. `cyclic_order_score` maximises over the
      2k dihedral symmetries and is therefore reflection-invariant, so rho must be
      IDENTICAL under a PC2 sign flip; that identity is ASSERTED, not assumed. The
      handedness a sign flip does change is exposed by also reporting the rotation-only
      statistic under each convention.

  C8  Expected lexical pattern, stated in advance so it cannot be reported as a finding:
      the token readout may well close at layers 0-2 if direction embeddings carry the
      compass rose. That is the confound. The claim this script can support is closure
      at the SUFFIX readout.

  C9  Per-cell outcome, four values (internal bookkeeping; the ONTOLOGY-level verdict
      uses the prereg 4.3 vocabulary, see C12):
        compass_ring           closure passes (a) AND (b) AND ordering p < 0.05
        closed_ring_unordered  closure passes (a) AND (b), ordering does not
        ordered_arc            ordering passes, closure does not
        nothing                neither
        degenerate             the point cloud carries no usable geometry

  C10 SECONDARY, WIDENED PLANE SEARCH -- POST-HOC ADDITION TO THE PREREG'S REGISTERED
      METHOD FAMILY, provenance stated as prereg section 7 requires. It is NOT in
      docs/phase2_prereg_2026-08-21.md. It is imported verbatim in rule from
      scripts/24_colors_existence.py D10, and it is declared HERE, in this file, before
      any compass activation was read -- not added after a negative.
        rule: all C(4,2) = 6 planes spanned by pairs of the top 4 PCs; the plane
              minimising the angular gap ratio is selected. Selection is LABEL-FREE, so
              the shuffled-label ordering null on the winning plane stays exact.
        closure: the uniform-angle p on a SELECTED plane is invalid and is reported
              marked as such. The valid test is the matched background with the
              IDENTICAL min-over-6-planes selection applied to every draw.
        verdicts: C9 and C12 verdicts are computed on the PRIMARY statistic only. The
              secondary gets its own field, so any positive is visibly a product of a
              widened search.

  C11 MULTIPLICITY. 3 readouts x 26 layers = 78 cells. At alpha = 0.05 about 4 ordering
      passes and about 4 closure passes are expected by chance, so an isolated pass is
      not a finding. BH across all live cells is reported alongside the raw counts, the
      expected-by-chance count, and each statistic's p-value floor.

  C12 ONTOLOGY-LEVEL VERDICT VOCABULARY -- prereg 4.3, and only these:
        recovered_blind                    (impossible for compass: blind arm untestable)
        structure_exists_supervised_only   supervised closure + ordering pass survives
                                           BH and the matched backgrounds. prereg 3.4:
                                           "unless the blind arm independently supports
                                           it, which it cannot here" -- so this is the
                                           CEILING for compass.
        ordered_not_closed                 ordering beats its shuffle control, closure
                                           fails. Not a ring claim.
        clean_negative                     the test ran with adequate power (C2 gate
                                           passed at k = 8) and the structure is absent.
        untestable                         the C2 power gate or the tokenisation gate
                                           blocks the test. No negative may be reported.

  C13 NO RIVAL ORDER IS TESTED. prereg 3.1 registers a rival ordering (fifths vs
      chromatic) for musical keys ONLY. Compass has one registered order -- geographic.
      Inventing a second would be an unregistered addition.

CAVEATS THAT SURVIVE A POSITIVE (prereg 4.5, restated so they cannot be shed):
  * A closed ring at the suffix readout shows the information is present one token
    later; attention can copy an embedding forward, so this is not proof of
    recomputation. The only handle reported is the principal angle between the suffix
    plane and the layer-0 token plane.
  * Layers 0-4 are embedding-adjacent: a hit there validates the pipeline and locates
    geometry near the input; it is not a depth claim.
  * `structure_exists_supervised_only` is a weaker claim than Phase 1's months result,
    because the label set was chosen by name.
"""
import argparse, json, os, sys, time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))
from gfdisc.env import DEVICE, D_MODEL
from gfdisc.model import load_model, capture
from gfdisc.fits import (degenerate, pca_2d, fit_circle, circularity, is_ring, angles,
                         cyclic_order_score, _rank_of)
from gfdisc.nulls import benjamini_hochberg
from gfdisc.ontologies import DIRECTIONS

PREREG = "docs/phase2_prereg_2026-08-21.md"
STATUS = "exploratory Phase-2 under docs/phase2_prereg_2026-08-21.md"

# ---------------------------------------------------------------------------
# C1 -- label set. Geographic order, from the committed DIRECTIONS list.
# ---------------------------------------------------------------------------
COMPASS = [(w, 45 * i) for i, w in enumerate(DIRECTIONS)]
CARDINALS = {"north", "east", "south", "west"}
# C1: capitalized forms, EXPLORATORY ONLY (place-name-sense contrast, never a verdict).
EXPLORATORY_CAPS = [w.capitalize() for w in DIRECTIONS]

# ---------------------------------------------------------------------------
# C3 -- templates. (prefix, suffix); the word goes between them with one space.
# Every PREFIX establishes the compass sense before the direction token (prereg 3.4).
# No template contains another direction word; every suffix is >= 3 tokens so that the
# "suffix" and "final" readouts are distinct positions.
# ---------------------------------------------------------------------------
TEMPLATES = [
    ("The road runs to the", " for several miles"),
    ("The wind was blowing from the", " all that afternoon"),
    ("They travelled", " until the river ended"),
    ("The compass needle pointed", " once again that morning"),
    ("Our camp lies to the", " of the ridge"),
    ("The river flows", " through the valley"),
    ("He walked", " along the coastal path"),
    ("The birds migrate", " every single year"),
    ("The map shows the town to the", " of the old mill"),
    ("Turn", " at the next junction"),
    ("The storm moved", " over open water"),
    ("A trail leads", " into the hills"),
    ("The sun was low in the", " that quiet evening"),
    ("The border lies", " of the mountains"),
    ("She sailed", " for eleven long days"),
    ("The village sits just", " of the lake"),
]
READOUTS = ("token", "suffix", "final")

# ---------------------------------------------------------------------------
# C6 -- background pools and the exclusion rule.
# ---------------------------------------------------------------------------
DIRECTION_SENSE_EXCLUSIONS = [
    # the ontology itself and its compounds
    "north", "south", "east", "west", "northeast", "southeast", "southwest",
    "northwest", "northern", "southern", "eastern", "western", "nordic", "orient",
    "oriental", "occident", "midwest",
    # other established directional senses
    "left", "right", "up", "down", "upward", "downward", "forward", "backward",
    "forwards", "backwards", "ahead", "behind", "above", "below", "inland", "coastal",
    "seaward", "leeward", "windward", "upstream", "downstream", "uphill", "downhill",
    "bearing", "compass", "heading", "azimuth", "direction", "arrow", "needle",
    "clockwise", "latitude", "longitude", "equator", "pole", "polar", "tropic",
    # calendar words, excluded so no pool inherits this project's other ontologies
    "january", "february", "march", "april", "may", "june", "july", "august",
    "september", "october", "november", "december", "monday", "tuesday", "wednesday",
    "thursday", "friday", "saturday", "sunday", "spring", "summer", "autumn", "winter",
    "fall", "weekend", "month", "season",
]
NOUN_POOL = [
    "table", "chair", "window", "door", "garden", "letter", "market", "engine",
    "ticket", "bottle", "camera", "pencil", "farmer", "doctor", "teacher", "student",
    "kitchen", "office", "school", "hospital", "library", "station", "airport",
    "factory", "machine", "computer", "keyboard", "screen", "paper", "folder",
    "basket", "blanket", "pillow", "mirror", "carpet", "curtain", "ladder", "hammer",
    "rope", "string", "needle", "button", "pocket", "jacket", "shirt", "hat", "glove",
    "scarf", "belt", "wallet", "purse", "suitcase", "umbrella", "clock", "watch",
    "radio", "telephone", "printer", "monitor", "cable", "battery", "lamp", "candle",
    "kettle", "plate", "spoon", "fork", "knife", "glass", "bowl", "tray", "oven",
    "sink", "towel", "soap", "brush", "comb", "razor", "horse", "sheep", "rabbit",
    "mouse", "eagle", "spider", "beetle", "forest", "meadow", "valley", "canyon",
    "island", "harbour", "beach", "cliff", "cave", "stream", "pond", "lake", "storm",
    "cloud", "thunder", "breeze", "frost", "tunnel", "fence", "gate", "roof", "floor",
    "ceiling", "stair", "corner", "shelf", "drawer", "cupboard", "bucket", "shovel",
    "wheel", "handle", "pipe", "wire", "nail", "screw", "board", "plank", "brick",
    "stone", "gravel", "bridge", "village", "river", "mountain", "desert", "ocean",
]
# C5(c) / prereg 4.2(c): the class-matched secondary pool for compass is PLACE NAMES.
# Every candidate is screened against C6: any name containing a direction substring is
# dropped mechanically before use.
PLACE_POOL = [
    "London", "Paris", "Berlin", "Madrid", "Rome", "Vienna", "Prague", "Warsaw",
    "Dublin", "Lisbon", "Athens", "Moscow", "Cairo", "Nairobi", "Lagos", "Tunis",
    "Delhi", "Mumbai", "Beijing", "Tokyo", "Seoul", "Osaka", "Manila", "Jakarta",
    "Sydney", "Melbourne", "Perth", "Auckland", "Toronto", "Ottawa", "Montreal",
    "Chicago", "Boston", "Denver", "Dallas", "Houston", "Phoenix", "Seattle",
    "Portland", "Atlanta", "Miami", "Detroit", "Cleveland", "Baltimore", "Pittsburgh",
    "Bogota", "Lima", "Santiago", "Caracas", "Havana", "Kingston", "Panama",
    "Oslo", "Bergen", "Helsinki", "Stockholm", "Copenhagen", "Amsterdam", "Brussels",
    "Zurich", "Geneva", "Munich", "Hamburg", "Cologne", "Naples", "Venice", "Milan",
    "Turin", "Seville", "Valencia", "Porto", "Ankara", "Istanbul", "Tehran",
    "Baghdad", "Damascus", "Amman", "Riyadh", "Doha", "Dubai", "Karachi", "Lahore",
    "Kabul", "Colombo", "Bangkok", "Hanoi", "Taipei", "Shanghai", "Wuhan",
    # Northampton / Westminster / Southport / Easton and any other direction-bearing
    # name would be dropped by the C6 substring rule below; a few are listed here on
    # purpose so the rule is visibly exercised rather than merely asserted.
    "Northampton", "Westminster", "Southport", "Easton", "Eastbourne", "Norwich",
]
DIRECTION_SUBSTRINGS = ("north", "south", "east", "west")

N_BACKGROUND = 2000          # C5, and the recorded prereg discrepancy
N_PERM_ORDER = 20000
N_SHUFFLE_CONTROL = 1000
N_GAP_NULL = 20000
FULL_K = 8

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
# Ordering, vectorised, with the flip group exposed (C7).
# ---------------------------------------------------------------------------
def order_stats(ang, label_index, K, flips=(1, -1), n_perm=N_PERM_ORDER, seed=0, block=5000):
    """Max Spearman over the allowed symmetries, calibrated by shuffled labels.

    Same construction as fits.cyclic_order_score, vectorised over the permutation null,
    with `flips` exposed so the rotation-only statistic can be read off separately. With
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


def check_against_reference(k=FULL_K, seed=0):
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
    return dict(k=int(k), trials=out, passed=bool(all(x["rho_match"] for x in out)))


# ---------------------------------------------------------------------------
# The one analysis routine. Compass, backgrounds and synthetics all go through it.
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
# C10 -- widened, label-free plane search (post-hoc addition, provenance declared).
# ---------------------------------------------------------------------------
N_PC_SECONDARY = 4


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
    """(primary gap ratio, primary p_uniform, best-plane gap ratio, primary radius ratio)."""
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
# C2 -- k = 8 reference-ring power table at the REALIZED support. Computed FIRST.
# ---------------------------------------------------------------------------
def power_table_k8(survivors, full_k=FULL_K, seed=0):
    """Every dropout pattern at k = 8, plus the realized-support row.

    Imported, not re-implemented: `reference_ring` from scripts/13_rerun_gate.py, the
    same function scripts/25_prereg_power.py used to build prereg section 6.
    """
    import importlib.util
    spec = importlib.util.spec_from_file_location("g13", REPO / "scripts/13_rerun_gate.py")
    g13 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(g13)

    from itertools import combinations
    rows = []

    def add(name, keep):
        r = g13.reference_ring(sorted(keep), full_k, seed=seed)
        rows.append(dict(case=name, n=len(keep), keep=sorted(int(x) for x in keep),
                         ref_gap_deg=float(r["ref_gap_deg"]),
                         ref_p_closed=float(r["ref_p_closed"]),
                         missing_run=int(r["missing_run"]),
                         closure_testable=bool(r["ref_p_closed"] < 0.05)))

    full = list(range(full_k))
    add("full ring (all 8 directions)", full)
    for n_drop in (1, 2, 3, 4):
        seen = set()
        for drop in combinations(full, n_drop):
            keep = [i for i in full if i not in drop]
            key = tuple(keep)
            if key in seen:
                continue
            seen.add(key)
            add(f"{n_drop} dropout(s): drop slots {list(drop)}", keep)
    add("cardinals only (N,E,S,W = slots 0,2,4,6)", [0, 2, 4, 6])
    add("intercardinals only (slots 1,3,5,7)", [1, 3, 5, 7])
    add(f"REALIZED SUPPORT ({len(survivors)} of {full_k})", survivors)

    by_n = {}
    for r in rows:
        if r["case"].startswith(("full", "cardinals", "intercardinals", "REALIZED")):
            continue
        by_n.setdefault(r["n"], []).append(r)
    summary = {}
    for n, rs in sorted(by_n.items()):
        summary[str(n)] = dict(
            n_patterns=len(rs),
            n_testable=int(sum(r["closure_testable"] for r in rs)),
            ref_p_min=float(min(r["ref_p_closed"] for r in rs)),
            ref_p_max=float(max(r["ref_p_closed"] for r in rs)),
            worst_case_testable=bool(all(r["closure_testable"] for r in rs)))
    realized = rows[-1]
    return dict(
        full_k=int(full_k), method="scripts/13_rerun_gate.py:reference_ring, seed 0, "
                                   "20000 uniform-angle draws; no data touched",
        rows=rows, dropout_summary=summary, realized=realized,
        gate=("closure at k=8 is testable at the realized support"
              if realized["closure_testable"] else
              "closure at k=8 is NOT testable at the realized support; a failure "
              "would say nothing about a ring and no negative may be reported"),
        prereg_section6_cross_check=dict(
            k8_full=next(r["ref_p_closed"] for r in rows if r["n"] == 8),
            k8_one_dropout_range=[summary["7"]["ref_p_min"], summary["7"]["ref_p_max"]],
            k8_two_dropout_range=[summary["6"]["ref_p_min"], summary["6"]["ref_p_max"]],
            k8_cardinals_only=next(r["ref_p_closed"] for r in rows
                                   if r["case"].startswith("cardinals")),
            note="prereg section 6 reports 5.0e-05 full, 0.0731 one dropout, 0.4446 two "
                 "adjacent, 0.0260 two non-adjacent, 5.0e-05 cardinals-only; these are "
                 "the same numbers from the same function at the same seed"))


# ---------------------------------------------------------------------------
# Synthetic battery at k = 8, through the identical `analyse` / `fast_closure` path.
# ---------------------------------------------------------------------------
def synthetics(d=D_MODEL, k=FULL_K, noise=0.15, n_bg=N_BACKGROUND, seed=0):
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

    out = dict(noise=noise, k=k, d=d, n_background=int(n_bg))
    th = 2 * np.pi * np.arange(k) / k
    ring = np.outer(np.cos(th), u) + np.outer(np.sin(th), v) + amb(noise)
    out["planted_ring"] = row(ring, 1)

    perm = rng.permutation(k)
    out["planted_ring_scrambled_labels"] = row(ring, 2, labels=perm)

    ta = np.linspace(0, np.radians(200), k)
    arc = np.outer(np.cos(ta), u) + np.outer(np.sin(ta), v) + amb(noise)
    out["planted_arc_200deg"] = row(arc, 3)

    # The ordinal control: a thin sausage in the fitted plane, matching the measured
    # number-words control (gap_ratio 5.1-7.4, top2 0.75-0.95, rho >= 0.99, never closes).
    line = (np.outer(np.linspace(-1, 1, k), u)
            + np.outer(0.05 * rng.normal(size=k), v)
            + amb(0.34))
    out["planted_line"] = row(line, 4)
    out["planted_line_note"] = ("thin sausage in the fitted plane; matches the measured "
                                "number-words ordinal control -- orders, never closes")

    # C10 validation: a ring buried under a dominant third component whose per-point
    # coefficients are orthogonalised against the ring's. The primary top-2 plane must
    # miss it; the widened search must find it. Without this row a negative secondary
    # would say nothing about the secondary.
    w = rng.normal(size=d)
    w -= (w @ u) * u + (w @ v) * v
    w /= np.linalg.norm(w)
    ramp = np.linspace(-1, 1, k)
    B = np.column_stack([np.ones(k), np.cos(th), np.sin(th)])
    ramp = ramp - B @ np.linalg.lstsq(B, ramp, rcond=None)[0]
    ramp = ramp / max(ramp.std(), 1e-12)
    buried = (2.0 * np.outer(ramp, w)
              + np.outer(np.cos(th), u) + np.outer(np.sin(th), v)
              + amb(noise))
    out["buried_ring"] = row(buried, 5)

    # An 8-point ring with ONE point deleted, planted so the C2 power gate is exercised
    # on real geometry and not only on the reference table: a perfect 7-of-8 ring cannot
    # reach p_closed < 0.05 whatever the noise.
    keep7 = [i for i in range(k) if i != 3]
    th7 = 2 * np.pi * np.array(keep7) / k
    u7 = np.outer(np.cos(th7), u) + np.outer(np.sin(th7), v)
    u7 = u7 + (noise / np.sqrt(d)) * rng.normal(size=(len(keep7), d))
    r7, _, _ = analyse(u7, np.array(keep7), k, seed=6, both_conventions=False)
    a7 = r7["conventions"]["A"]
    out["planted_ring_one_dropout_k7"] = dict(
        n=len(keep7), gap_ratio=a7["gap_ratio"], max_gap_deg=a7["max_gap_deg"],
        p_closed=a7["p_closed_uniform"], rho=a7["order"]["rho"],
        p_order=a7["order"]["p"],
        note="perfect ring minus one of eight; the C2 gate predicts closure cannot pass")

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
        and LN["p_closed"] > 0.05 and LN["p_vs_background"] > 0.05 and LN["p_order"] <= 0.05
        and out["pure_noise"]["n_sig_closed_background"] <= 1
        and BR["p_vs_background"] > 0.05 and BR["secondary"]["p_vs_background"] <= 0.05
        and AR["secondary"]["p_vs_background"] > 0.05
        and LN["secondary"]["p_vs_background"] > 0.05
        and out["pure_noise"]["n_sig_closed_background_secondary"] <= 1)
    return out


# ---------------------------------------------------------------------------
# BLIND ARM -- frozen artifacts only. No re-clustering.
# ---------------------------------------------------------------------------
def blind_arm(artifact_path):
    """Read the frozen catalogue and report placement, worded per prereg section 7.

    Every registered token variant of every direction is searched against every cluster
    membership the artifact actually stores. What the artifact stores is itself reported,
    because it bounds what this arm can say.
    """
    art = json.loads(Path(artifact_path).read_text())
    variants = []
    for w in DIRECTIONS:
        for form in (w, w.capitalize()):
            variants += [form, " " + form]
    variants = list(dict.fromkeys(variants))
    V = set(variants)

    lsc_hits, ring_hits, n_lsc, n_lsc_full_membership = [], [], 0, 0
    for L in art["layers"]:
        for r in L["label_scored_clusters"]:
            n_lsc += 1
            if r.get("members_token_ids") is not None:
                n_lsc_full_membership += 1
            ts = r.get("token_strings") or []
            m = sorted(t for t in ts if t in V)
            if m:
                lsc_hits.append(dict(layer=L["layer"], arm=r["arm"],
                                     cluster_id=r["cluster_id"], size=r["size"],
                                     flagged=r["flagged"], matched=m,
                                     n_strings_truncated=r.get("n_strings_truncated")))
    for r in art["verdict"]["other_rings"]:
        ts = r.get("token_strings") or []
        m = sorted(t for t in ts if t in V)
        if m:
            ring_hits.append(dict(layer=r["layer"], arm=r["arm"], k=r["k"], matched=m))

    # prereg section 2.1 exposure item: the layer-3 cluster whose in-session summary name
    # contained the word "direction". Discharged by listing membership verbatim.
    l3 = [dict(layer=r["layer"], arm=r["arm"], k=r["k"], p_closed=r["p_closed"],
               p_bg=r["p_bg"], token_strings=r["token_strings"],
               n_compass_terms=int(sum(1 for t in r["token_strings"] if t in V)))
          for r in art["verdict"]["other_rings"] if r["layer"] == 3]
    particle = [r for r in l3 if {" up", " down", " out", " off"} & set(r["token_strings"])]

    return dict(
        arm="blind",
        verdict="untestable",
        verdict_source=(f"{PREREG} sections 3.4, 5.3, 5.6 -- PRE-REGISTERED before any "
                        "placement was examined, not decided here"),
        why_untestable=[
            "4 of 8 direction terms clear the >=100 count floor (' North' 586, ' South' "
            "600, ' West' 434, ' East' 353); all four intercardinals sit at 25-29 and "
            "are below the extended >=30 floor too, so they are not nodes in the frozen "
            "partitions at all",
            "4 < ceil(2/3 * 8) = 6, the artifact's own label-set co-location convention",
            "a 4-member set is below the frozen screening band minimum size of 5, so it "
            "could not be screened as its own cluster under the frozen rules",
            "four evenly-spread points pass the uniform-gap test trivially "
            "(ref_p 5.0e-05), so a cardinals-only pass would be an artifact of arity",
        ],
        forbidden=(f"{PREREG} section 7 forbids reporting a blind-arm NEGATIVE for "
                   "compass. Nothing below is a negative; it is a description of what "
                   "the frozen catalogue does and does not contain."),
        artifact=dict(path=str(artifact_path),
                      generated=art.get("generated"),
                      declared_at=art.get("declared_at"),
                      status=art.get("status"),
                      n_ring_flagged=len(art["verdict"]["other_rings"]),
                      n_label_scored_cluster_records=n_lsc,
                      n_records_with_full_membership=n_lsc_full_membership),
        variants_searched=variants,
        catalogue_search=dict(
            n_hits_in_ring_catalogue=len(ring_hits), ring_hits=ring_hits,
            n_hits_in_label_scored_clusters=len(lsc_hits), label_scored_hits=lsc_hits,
            statement=("no registered compass variant appears in any stored cluster "
                       "membership in the frozen artifact"
                       if not ring_hits and not lsc_hits else
                       "compass variants appear in the stored memberships listed above")),
        registered_colocation_test_not_computable=dict(
            registered_as=(f"{PREREG} section 3.4: co-location of the four cardinals "
                           "against matched random 4-subsets, descriptive only"),
            computable_from_frozen_artifact=False,
            mechanism=("scripts/22_token_type_geometry.py records a cluster only when it "
                       "is flagged or carries >= ceil(2/3 k) of a DECLARED label set "
                       "(months, weekdays, colors, numbers, digits); compass was not a "
                       "declared label set, and `members_token_ids` is written only for "
                       "clusters of size <= 80 among those. The partition assignment for "
                       "the four cardinals is therefore not in the artifact."),
            why_not_recovered=("recovering it would require re-running the Leiden "
                               "partitions, which " + PREREG + " section 4.1 and the "
                               "Phase-2 task text both forbid ('no re-clustering'). The "
                               "test is descriptive-only and changes no verdict, so it "
                               "is reported as not computed rather than computed by a "
                               "forbidden route.")),
        exposure_2_1_discharged=dict(
            exposure=(f"{PREREG} section 2.1 records that a layer-3 cluster summarized "
                      "in-session as 'particle / direction' pre-dates the prereg, and "
                      "registers that a compass landing in it would be "
                      "`recovered_blind (exposed)` rather than a blind discovery"),
            layer3_flagged_rings=l3,
            particle_direction_ring=(particle[0] if particle else None),
            n_compass_terms_in_particle_ring=(particle[0]["n_compass_terms"]
                                              if particle else None),
            conclusion=("the layer-3 'particle / direction' ring is verb particles and "
                        "spatial adverbs (out/up/off/down/together/away/apart); it "
                        "contains zero compass terms, so `recovered_blind (exposed)` "
                        "cannot apply to compass"
                        if particle and particle[0]["n_compass_terms"] == 0 else
                        "see layer3_flagged_rings")),
        ceiling=(f"{PREREG} section 3.4: a supervised closure pass 'is "
                 "`structure_exists_supervised_only` unless the blind arm independently "
                 "supports it, which it cannot here'"))


# ---------------------------------------------------------------------------
# Activations
# ---------------------------------------------------------------------------
def token_audit(tok, words):
    keep, drop = [], []
    for w in words:
        ids = tok.encode(" " + w, add_special_tokens=False)
        (keep if len(ids) == 1 else drop).append(
            dict(word=w, n_tokens=len(ids), ids=[int(i) for i in ids]))
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
    return str(o)


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/phase2_compass.json")
    ap.add_argument("--artifact", default="results/token_type_geometry.json")
    ap.add_argument("--layers", default=",".join(str(i) for i in range(26)))
    ap.add_argument("--n-background", type=int, default=N_BACKGROUND)
    ap.add_argument("--chunk", type=int, default=48)
    a = ap.parse_args()
    t0 = time.time()
    layers = [int(x) for x in a.layers.split(",")]
    os.chdir(REPO)

    print("=" * 86)
    print("PHASE 2 -- COMPASS DIRECTIONS  (prereg " + PREREG + " section 3.4)")
    print("DECLARATIONS (fixed before any activation is read):")
    print(f"  C1  labels        : {[w for w, _ in COMPASS]}")
    print(f"      angles (deg)  : {[d for _, d in COMPASS]}")
    print(f"      surface form  : leading-space LOWERCASE; capitalized forms "
          f"{EXPLORATORY_CAPS[:2]}... are EXPLORATORY only (place-name hazard)")
    print(f"  C2  power gate    : k=8 reference-ring table at the realized support, "
          f"computed BEFORE the sweep; < 8 survivors -> untestable, abort")
    print(f"  C3  readouts      : token (embedding-confounded) | suffix = first "
          f"post-direction token (PRIMARY) | final (tertiary)")
    print(f"  C4  discriminator : closure. Ordering alone is never evidence.")
    print(f"  C5  nulls         : uniform angles ({N_GAP_NULL}); {a.n_background} matched "
          f"noun draws; {a.n_background} PLACE-NAME draws (prereg 4.2c); "
          f"{N_PERM_ORDER} shuffled labels + {N_SHUFFLE_CONTROL}-shuffle control")
    print(f"      prereg 4.2(b) registers 300 draws; 2000 used so the BH rank-1 "
          f"threshold (6.4e-4) is reachable above the p floor. Recorded, not silent.")
    print(f"  C10 widened plane : POST-HOC addition to the prereg method family, "
          f"provenance scripts/24_colors_existence.py D10, declared here before any "
          f"activation was read. Verdicts use the PRIMARY statistic only.")
    print(f"  C12 verdicts      : structure_exists_supervised_only | ordered_not_closed "
          f"| clean_negative | untestable   (recovered_blind is unreachable: prereg 3.4)")
    print(f"  C13 no rival order: only musical keys has a registered rival ordering")
    print("=" * 86, flush=True)

    # ---- BLIND ARM (frozen artifacts only) ---------------------------------
    blind = blind_arm(a.artifact)
    print("\n--- BLIND ARM (frozen artifacts only, no re-clustering) ---")
    print(f"  verdict: {blind['verdict']}  [{blind['verdict_source']}]")
    art = blind["artifact"]
    print(f"  artifact {art['path']} generated {art['generated']}: "
          f"{art['n_ring_flagged']} ring-flagged clusters, "
          f"{art['n_label_scored_cluster_records']} label-scored cluster records "
          f"({art['n_records_with_full_membership']} with full membership stored)")
    print(f"  {len(blind['variants_searched'])} registered variants searched -> "
          f"{blind['catalogue_search']['n_hits_in_ring_catalogue']} hits in the ring "
          f"catalogue, {blind['catalogue_search']['n_hits_in_label_scored_clusters']} "
          f"in label-scored clusters")
    pr = blind["exposure_2_1_discharged"]["particle_direction_ring"]
    if pr:
        print(f"  prereg 2.1 exposure -- layer-3 'particle / direction' ring "
              f"(arm {pr['arm']}, k={pr['k']}): {pr['token_strings']}")
        print(f"    compass terms in it: {pr['n_compass_terms']} -> "
              f"'recovered_blind (exposed)' cannot apply")
    print(f"  registered co-location test computable from the artifact: "
          f"{blind['registered_colocation_test_not_computable']['computable_from_frozen_artifact']}",
          flush=True)

    # ---- C2 POWER GATE, computed FIRST -------------------------------------
    from transformers import AutoTokenizer
    from gfdisc.env import MODEL_NAME
    tok0 = AutoTokenizer.from_pretrained(MODEL_NAME, token=os.environ.get("HF_TOKEN"))
    pre_keep, pre_drop = token_audit(tok0, [w for w, _ in COMPASS])
    pre_words = [x["word"] for x in pre_keep]
    survivors = [i for i, (w, _) in enumerate(COMPASS) if w in set(pre_words)]
    print(f"\n--- C2 POWER GATE: k = {FULL_K} reference-ring table at the realized "
          f"support ---")
    pw = power_table_k8(survivors)
    print(f"  {'case':<46} {'n':>2} {'gap':>7} {'ref_p':>9}  testable")
    for r in pw["rows"]:
        if r["n"] >= 6 or r["case"].startswith(("cardinals", "intercardinals", "REALIZED")):
            if r["n"] == 7 and not r["case"].startswith("1 dropout(s): drop slots [0]"):
                continue
            if r["n"] == 6 and r["case"] not in (
                    "2 dropout(s): drop slots [0, 1]", "2 dropout(s): drop slots [0, 2]",
                    "2 dropout(s): drop slots [0, 4]"):
                continue
            print(f"  {r['case']:<46} {r['n']:>2} {r['ref_gap_deg']:>7.2f} "
                  f"{r['ref_p_closed']:>9.5f}  {r['closure_testable']}")
    for n, s in sorted(pw["dropout_summary"].items()):
        print(f"  all {s['n_patterns']:>2} patterns with n={n}: ref_p in "
              f"[{s['ref_p_min']:.5f}, {s['ref_p_max']:.5f}], "
              f"{s['n_testable']}/{s['n_patterns']} testable")
    print(f"  REALIZED: {len(survivors)}/{FULL_K} survive tokenisation -> "
          f"ref_p {pw['realized']['ref_p_closed']:.5f} -> {pw['gate']}", flush=True)
    if not pw["realized"]["closure_testable"] or len(survivors) < FULL_K:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(dict(
            status=STATUS, item="phase2_compass", script="scripts/31_phase2_compass.py",
            prereg=PREREG, generated=time.strftime("%Y-%m-%dT%H:%M:%S"),
            blind_arm=blind, power_gate=pw,
            supervised_arm=dict(verdict="untestable",
                                reason=f"only {len(survivors)} of {FULL_K} direction "
                                       f"words survive the tokenisation audit; "
                                       f"ref_p_closed "
                                       f"{pw['realized']['ref_p_closed']:.5f} >= 0.05, "
                                       f"so closure could not pass whatever the geometry",
                                dropped=pre_drop),
            verdict=dict(blind="untestable", supervised="untestable"),
            runtime_sec=float(time.time() - t0)), indent=2, default=jsonable))
        raise SystemExit(f"C2 power gate: only {len(survivors)}/{FULL_K} survivors -> "
                         f"untestable; wrote {a.out}")

    # ---- statistic self-check and synthetics -------------------------------
    ref = check_against_reference()
    print(f"\norder_stats vs fits.cyclic_order_score at k={FULL_K}: "
          f"{'MATCH' if ref['passed'] else 'MISMATCH'} on rho over 5 trials", flush=True)
    if not ref["passed"]:
        raise SystemExit("vectorised ordering statistic disagrees with the reference")

    syn = synthetics(k=FULL_K, n_bg=a.n_background)
    print("synthetic battery (k=8, through the identical analyse/fast_closure path):")
    for key in ("planted_ring", "planted_ring_scrambled_labels", "planted_arc_200deg",
                "planted_line", "buried_ring"):
        r = syn[key]
        print(f"  {key:32s} gapR {r['gap_ratio']:6.2f}  pUnif {r['p_closed']:.4f}  "
              f"pBg {r['p_vs_background']:.4f}  rho {r['rho']:+.3f}  "
              f"pOrd {r['p_order']:.4f}  sec_pBg "
              f"{r['secondary']['p_vs_background']:.4f}")
    d7 = syn["planted_ring_one_dropout_k7"]
    print(f"  {'planted_ring_one_dropout_k7':32s} gapR {d7['gap_ratio']:6.2f}  "
          f"pUnif {d7['p_closed']:.4f}  rho {d7['rho']:+.3f} "
          f"(C2 predicts closure cannot pass at 7 of 8)")
    uc = syn["uniform_null_calibration"]
    print(f"  uniform-angle null vs {uc['n_draws']} isotropic clouds (k={syn['k']}, "
          f"d={syn['d']}): mean gapR {uc['gap_ratio_mean']:.2f}, "
          f"false acceptance at p<0.05 = {uc['uniform_null_false_acceptance_at_p05']:.3f}")
    print(f"  pure_noise 5 draws: {syn['pure_noise']['n_sig_closed_background']}/5 "
          f"background-closure trips, {syn['pure_noise']['n_sig_order']}/5 ordering trips "
          f"-> {'PASS' if syn['passed'] else 'FAIL'}", flush=True)
    if not syn["passed"]:
        raise SystemExit("synthetic battery failed; not running on activations")

    # ---- model, audits -----------------------------------------------------
    model, tok = load_model(token=os.environ.get("HF_TOKEN"))
    bos = tok.bos_token_id

    dir_keep, dir_drop = token_audit(tok, [w for w, _ in COMPASS])
    dir_words = [x["word"] for x in dir_keep]
    dir_angles = [d for w, d in COMPASS if w in set(dir_words)]
    caps_keep, caps_drop = token_audit(tok, EXPLORATORY_CAPS)
    caps_words = [x["word"] for x in caps_keep]

    excl = set(DIRECTION_SENSE_EXCLUSIONS)

    def excluded(w):
        lw = w.lower()
        if lw in excl:
            return "direction/calendar sense"
        for sub in DIRECTION_SUBSTRINGS:
            if sub in lw:
                return f"contains direction substring '{sub}'"
        return None

    noun_excluded = {w: r for w in dict.fromkeys(NOUN_POOL) if (r := excluded(w))}
    place_excluded = {w: r for w in dict.fromkeys(PLACE_POOL) if (r := excluded(w))}
    noun_cand = [w for w in dict.fromkeys(NOUN_POOL) if w not in noun_excluded]
    place_cand = [w for w in dict.fromkeys(PLACE_POOL) if w not in place_excluded]
    noun_keep, noun_drop = token_audit(tok, noun_cand)
    place_keep, place_drop = token_audit(tok, place_cand)
    noun_words = [x["word"] for x in noun_keep]
    place_words = [x["word"] for x in place_keep]
    K = len(dir_words)
    print(f"\nC1 audit: direction set k={K} {dir_words} "
          f"(dropped {[x['word'] for x in dir_drop]})")
    print(f"          exploratory capitalized {caps_words} "
          f"(dropped {[x['word'] for x in caps_drop]})")
    print(f"C6 audit: {len(noun_words)} nouns, {len(place_words)} place names kept; "
          f"C6 excluded {len(noun_excluded)} nouns {sorted(noun_excluded)} and "
          f"{len(place_excluded)} places {sorted(place_excluded)}; multi-token drops "
          f"{len(noun_drop)} / {len(place_drop)}", flush=True)
    if K < FULL_K:
        raise SystemExit(f"only {K}/{FULL_K} direction words are single-token -> "
                         f"untestable (C2)")

    # ---- template verification --------------------------------------------
    label_words = dir_words
    pool_words = caps_words + noun_words + place_words

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
    caps_words = [w for w in caps_words if w not in dropped_pool]
    noun_words = [w for w in noun_words if w not in dropped_pool]
    place_words = [w for w in place_words if w not in dropped_pool]

    tmpl_report = []
    for prefix, suffix in TEMPLATES:
        pre = tok.encode(prefix, add_special_tokens=False)
        suf = tok.encode(suffix, add_special_tokens=False)
        tmpl_report.append(dict(prefix=prefix, suffix=suffix, n_prefix_tokens=len(pre),
                                n_suffix_tokens=len(suf), word_index=1 + len(pre),
                                suffix_index=2 + len(pre),
                                final_index=1 + len(pre) + len(suf),
                                first_post_word_token=tok.decode(suf[:1])))
    tokenisation_audit = dict(
        n_words_checked=len(label_words) + len(pool_words),
        n_templates=len(TEMPLATES),
        n_checks=(len(label_words) + len(pool_words)) * len(TEMPLATES),
        label_words_checked=label_words, label_mismatches=label_bad,
        pool_words_checked=len(pool_words), pool_words_dropped=sorted(dropped_pool),
        pool_mismatch_detail=pool_bad,
        rule="a direction-word mismatch ABORTS (they are the labels, and C2 makes 7 of 8 "
             "untestable); a pool mismatch drops the word (mechanical, label-blind)")
    print(f"\ntokenisation audit: {tokenisation_audit['n_checks']} word x template "
          f"checks, {len(label_bad)} label mismatches, {len(dropped_pool)} pool words "
          f"dropped {sorted(dropped_pool)}", flush=True)
    if label_bad:
        raise SystemExit(f"direction words with off-manifold ids: "
                         f"{sorted({d['word'] for d in label_bad})} -> untestable (C2)")
    if len(noun_words) < 40 or len(place_words) < 20:
        raise SystemExit(f"background pools too small after the drop rule: "
                         f"{len(noun_words)} nouns, {len(place_words)} places")

    words = dir_words + caps_words + noun_words + place_words
    word_ids = [tok.encode(" " + w, add_special_tokens=False)[0] for w in words]
    idx = {w: i for i, w in enumerate(words)}
    dir_idx = np.array([idx[w] for w in dir_words])
    caps_idx = np.array([idx[w] for w in caps_words])
    noun_idx = np.array([idx[w] for w in noun_words])
    place_idx = np.array([idx[w] for w in place_words])

    # ---- forwards ----------------------------------------------------------
    print(f"\ncollecting {len(words)} words x {len(TEMPLATES)} templates x "
          f"{len(layers)} layers x 3 readouts ...", flush=True)
    tf = time.time()
    M = collect(model, tok, words, word_ids, layers, chunk=a.chunk)
    forward_sec = time.time() - tf
    print(f"forwards done in {forward_sec:.1f}s", flush=True)
    del model
    if DEVICE == "mps":
        torch.mps.empty_cache()

    label_index = np.arange(K)
    ref_layer = layers[0]
    _, basis_L0_token, _ = analyse(M["token"][ref_layer][dir_idx], label_index, K,
                                   seed=0, do_order=False, both_conventions=False)

    rows, jack_rows, caps_rows, card_rows = [], [], [], []
    print(f"\n{'PRIMARY (pre-declared, top-2 PCA plane)':<86}"
          f"| {'C10 widened plane search':<34}")
    print(f"{'readout':>8} {'L':>3} {'top2':>6} {'radR':>7} {'gapR':>7} {'pUnif':>8} "
          f"{'pNoun':>8} {'FAnoun':>7} {'pPlace':>8} {'rho':>7} {'pOrd':>8} "
          f"{'ang(L0)':>8} | {'plane':>6} {'gapR':>6} {'pNoun':>7} {'rho':>6} {'pOrd':>7} "
          f" outcome", flush=True)
    for ro in READOUTS:
        for l in layers:
            P = M[ro][l][dir_idx]
            res, basis, xy = analyse(P, label_index, K, seed=l, do_order=True)
            A = res["conventions"]["A"]
            B = res["conventions"]["B"]
            # C7: closure is sign-invariant, rho reflection-invariant. Asserted.
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

            def bg(pool_idx, n_draws, seed, obs_gap=None, obs_sec_gap=None):
                """Identical pipeline, same arity, same positions -- only the words differ.

                `obs_gap` / `obs_sec_gap` default to the primary compass observation, so
                the exploratory capitalized block can reuse the same draw machinery
                against its own observed gap rather than the compass one.
                """
                obs_gap = A["gap_ratio"] if obs_gap is None else obs_gap
                obs_sec_gap = sec["gap_ratio"] if obs_sec_gap is None else obs_sec_gap
                r = np.random.default_rng(seed)
                gr = np.empty(n_draws); pc = np.empty(n_draws)
                rr = np.empty(n_draws); gb = np.empty(n_draws)
                for i in range(n_draws):
                    pick = pool_idx[r.choice(len(pool_idx), size=K, replace=False)]
                    gr[i], pc[i], gb[i], rr[i] = fast_closure(M[ro][l][pick])
                return dict(n_draws=int(n_draws),
                            gap_ratio_mean=float(gr.mean()),
                            gap_ratio_p05=float(np.percentile(gr, 5)),
                            gap_ratio_min=float(gr.min()),
                            radius_ratio_mean=float(rr.mean()),
                            false_acceptance_rate_p05=float((pc <= 0.05).mean()),
                            p_vs_background=float((1 + int((gr <= obs_gap).sum()))
                                                  / (1 + n_draws)),
                            secondary_best_plane=dict(
                                gap_ratio_mean=float(gb.mean()),
                                gap_ratio_p05=float(np.percentile(gb, 5)),
                                gap_ratio_min=float(gb.min()),
                                p_vs_background=float((1 + int((gb <= obs_sec_gap).sum()))
                                                      / (1 + n_draws))))

            b_noun = bg(noun_idx, a.n_background, 1000 + l)
            b_place = bg(place_idx, a.n_background, 2000 + l)
            sec["p_vs_noun_background"] = b_noun["secondary_best_plane"]["p_vs_background"]
            sec["p_vs_place_background"] = b_place["secondary_best_plane"]["p_vs_background"]
            row["secondary_plane_search"] = sec
            row["principal_angles_secondary_to_L0_token_plane_deg"] = principal_angles_deg(
                sec_basis, basis_L0_token)
            row["background_nouns"] = b_noun
            row["background_place_names"] = b_place
            pa = principal_angles_deg(basis, basis_L0_token)
            row["principal_angles_to_L0_token_plane_deg"] = pa

            # ---- per-cell outcome (C9), PRIMARY statistic only -------------
            if res["degenerate"]:
                outcome = "degenerate"
            else:
                closed = (A["p_closed_uniform"] < 0.05 and b_noun["p_vs_background"] < 0.05)
                ordered = A["order"]["p"] < 0.05
                outcome = ("compass_ring" if closed and ordered else
                           "closed_ring_unordered" if closed else
                           "ordered_arc" if ordered else "nothing")
            row["outcome"] = outcome
            if res["degenerate"]:
                outcome_sec = "degenerate"
            else:
                closed_s = sec["p_vs_noun_background"] < 0.05
                ordered_s = sec["order"]["p"] < 0.05
                outcome_sec = ("compass_ring_widened_search" if closed_s and ordered_s else
                               "closed_ring_unordered" if closed_s else
                               "ordered_arc" if ordered_s else "nothing")
            row["outcome_secondary_plane_search"] = outcome_sec
            rows.append(row)

            rho = None if res["degenerate"] else A["order"]["rho"]
            pord = None if res["degenerate"] else A["order"]["p"]
            srho = None if res["degenerate"] else sec["order"]["rho"]
            spo = None if res["degenerate"] else sec["order"]["p"]
            print(f"{ro:>8} {l:3d} {res['top2_var']:6.3f} {A['radius_ratio']:7.2f} "
                  f"{A['gap_ratio']:7.2f} {A['p_closed_uniform']:8.4f} "
                  f"{b_noun['p_vs_background']:8.4f} "
                  f"{b_noun['false_acceptance_rate_p05']:7.3f} "
                  f"{b_place['p_vs_background']:8.4f} "
                  f"{'   n/a' if rho is None else f'{rho:+7.3f}'} "
                  f"{'     n/a' if pord is None else f'{pord:8.4f}'} "
                  f"{pa[0]:8.1f} | {str(sec['selected_plane']):>6} "
                  f"{sec['gap_ratio']:6.2f} {sec['p_vs_noun_background']:7.4f} "
                  f"{'   n/a' if srho is None else f'{srho:+6.3f}'} "
                  f"{'    n/a' if spo is None else f'{spo:7.4f}'}  {outcome}", flush=True)

            # ---- EXPLORATORY: capitalized (place-name-sense) forms ---------
            if len(caps_words) == K:
                cap_order = [caps_words.index(w.capitalize()) for w, _ in COMPASS
                             if w.capitalize() in caps_words]
                q, _, _ = analyse(M[ro][l][caps_idx[cap_order]], label_index, K, seed=l,
                                  do_order=True, both_conventions=False)
                qa = q["conventions"]["A"]
                qbg = bg(noun_idx, 500, 5000 + l, obs_gap=qa["gap_ratio"],
                         obs_sec_gap=np.inf)
                caps_rows.append(dict(readout=ro, layer=l, k=K,
                                      gap_ratio=qa["gap_ratio"],
                                      p_closed_uniform=qa["p_closed_uniform"],
                                      p_vs_noun_background=qbg["p_vs_background"],
                                      n_background_draws=int(qbg["n_draws"]),
                                      rho=qa["order"]["rho"], p_order=qa["order"]["p"],
                                      top2_var=q["top2_var"]))

            # ---- descriptive: the four cardinals alone (k=4) ---------------
            # prereg 3.4 records that four evenly-spread points pass the uniform-gap test
            # trivially, so this is reported as descriptive, never as ring evidence.
            card_pos = [i for i, (w, _) in enumerate(COMPASS) if w in CARDINALS]
            q4, _, _ = analyse(P[card_pos], np.arange(len(card_pos)), len(card_pos),
                               seed=l, do_order=True, both_conventions=False)
            q4a = q4["conventions"]["A"]
            card_rows.append(dict(readout=ro, layer=l, k=len(card_pos),
                                  members=[COMPASS[i][0] for i in card_pos],
                                  gap_ratio=q4a["gap_ratio"],
                                  p_closed_uniform_TRIVIAL_AT_K4=q4a["p_closed_uniform"],
                                  rho=q4a["order"]["rho"], p_order=q4a["order"]["p"]))

            # ---- leave-one-out where closure passes ------------------------
            # C2: at 7 of 8 the reference ring cannot reach p < 0.05, so every
            # leave-one-out row is EXPECTED to fail closure. Recorded as a
            # power statement, not as fragility of the finding.
            if not res["degenerate"] and A["p_closed_uniform"] < 0.05:
                jk = []
                for j in range(K):
                    keep = [i for i in range(K) if i != j]
                    q, _, _ = analyse(P[keep], np.array(keep), K, seed=l,
                                      do_order=False, both_conventions=False)
                    qa = q["conventions"]["A"]
                    jk.append(dict(dropped=dir_words[j], gap_ratio=qa["gap_ratio"],
                                   p_closed_uniform=qa["p_closed_uniform"]))
                jack_rows.append(dict(
                    readout=ro, layer=l, leave_one_out=jk,
                    worst=max(jk, key=lambda x: x["p_closed_uniform"]),
                    n_still_closed=int(sum(x["p_closed_uniform"] < 0.05 for x in jk)),
                    power_note="C2: a PERFECT 7-of-8 ring scores ref_p_closed 0.0731, so "
                               "a leave-one-out failure here is the k=8 power limit, not "
                               "evidence against the k=8 result"))

    # ---- C11 multiplicity --------------------------------------------------
    live = [r for r in rows if not r["degenerate"]]
    n_cells = len(live)

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
                    min_cells_at_floor_to_survive_bh=(int(np.ceil(floor / rank1))
                                                      if rank1 > 0 else None),
                    n_at_floor=int((p <= floor + 1e-12).sum()),
                    n_surviving_bh_fdr=int(keep.sum()),
                    surviving=[dict(readout=live[i]["readout"], layer=live[i]["layer"],
                                    p=float(p[i])) for i in np.flatnonzero(keep)])

    nb = a.n_background
    mult = dict(
        n_cells=int(n_cells),
        note="3 readouts x 26 layers. An isolated pass at alpha = 0.05 is expected by "
             "chance and is not a finding; BH across all live cells is the honest count. "
             "Each row carries its p-value floor, because an empirical p cannot fall "
             "below 1/(n_null+1) and a BH column must be read against what was reachable.",
        ordering_primary=fdr([r["conventions"]["A"]["order"]["p"] for r in live],
                             "ordering p, primary plane", N_PERM_ORDER),
        closure_uniform_primary=fdr([r["conventions"]["A"]["p_closed_uniform"] for r in live],
                                    "closure p vs uniform angles, primary plane", N_GAP_NULL),
        closure_background_primary=fdr([r["background_nouns"]["p_vs_background"] for r in live],
                                       "closure p vs matched noun background, primary plane",
                                       nb),
        closure_place_background_primary=fdr(
            [r["background_place_names"]["p_vs_background"] for r in live],
            "closure p vs place-name background, primary plane", nb),
        closure_background_secondary=fdr(
            [r["secondary_plane_search"]["p_vs_noun_background"] for r in live],
            "closure p vs matched noun background, C10 selected plane", nb),
        ordering_secondary=fdr([r["secondary_plane_search"]["order"]["p"] for r in live],
                               "ordering p, C10 selected plane", N_PERM_ORDER),
    )

    # ---- C12 ontology-level verdict ----------------------------------------
    suffix_cells = [r for r in live if r["readout"] == "suffix"]
    ring_cells = [dict(readout=r["readout"], layer=r["layer"],
                       p_closed_uniform=r["conventions"]["A"]["p_closed_uniform"],
                       p_noun=r["background_nouns"]["p_vs_background"],
                       p_place=r["background_place_names"]["p_vs_background"],
                       rho=r["conventions"]["A"]["order"]["rho"],
                       p_order=r["conventions"]["A"]["order"]["p"])
                  for r in live if r["outcome"] == "compass_ring"]
    bh_closed = {(s["readout"], s["layer"])
                 for s in mult["closure_background_primary"]["surviving"]}
    bh_ordered = {(s["readout"], s["layer"])
                  for s in mult["ordering_primary"]["surviving"]}
    survivors_bh = [c for c in ring_cells
                    if (c["readout"], c["layer"]) in bh_closed
                    and (c["readout"], c["layer"]) in bh_ordered]
    n_ordered_raw = sum(1 for r in live if r["conventions"]["A"]["order"]["p"] < 0.05)
    n_closed_raw = sum(1 for r in live
                       if r["conventions"]["A"]["p_closed_uniform"] < 0.05
                       and r["background_nouns"]["p_vs_background"] < 0.05)

    if survivors_bh:
        verdict = "structure_exists_supervised_only"
        verdict_why = (f"{len(survivors_bh)} cell(s) close against BOTH nulls and order "
                       f"beyond the shuffle control, and survive BH across the "
                       f"{n_cells}-cell sweep. prereg 3.4 caps compass at this verdict: "
                       f"the blind arm cannot independently support it.")
    elif ring_cells:
        verdict = "clean_negative"
        verdict_why = (f"{len(ring_cells)} cell(s) reach compass_ring at raw alpha = 0.05 "
                       f"but none survives BH across {n_cells} cells "
                       f"(expected by chance ~{0.05 * n_cells:.1f}). prereg section 7 "
                       f"forbids reporting an isolated single-cell pass as a finding. "
                       f"Power was adequate: the C2 gate passes at k = 8 "
                       f"(ref_p {pw['realized']['ref_p_closed']:.5f}) and the k=8 planted "
                       f"ring was recovered by the identical code path.")
    elif mult["ordering_primary"]["n_surviving_bh_fdr"] > 0:
        verdict = "ordered_not_closed"
        verdict_why = (f"ordering survives BH in "
                       f"{mult['ordering_primary']['n_surviving_bh_fdr']} cell(s) while "
                       f"closure fails against the matched background everywhere. Not a "
                       f"ring claim: this project measured an ordinal control "
                       f"out-ordering real month cycles at every layer.")
    else:
        verdict = "clean_negative"
        verdict_why = (f"no cell closes against both nulls and no ordering survives BH. "
                       f"Power was adequate: the C2 gate passes at k = 8 "
                       f"(ref_p {pw['realized']['ref_p_closed']:.5f}), all eight "
                       f"directions are present, and the k=8 planted ring was recovered "
                       f"through the identical code path "
                       f"(pBg {syn['planted_ring']['p_vs_background']:.4f}).")

    out = dict(
        status=STATUS,
        item="phase2_compass",
        script="scripts/31_phase2_compass.py",
        prereg=PREREG,
        prereg_section="3.4 COMPASS DIRECTIONS",
        generated=time.strftime("%Y-%m-%dT%H:%M:%S"),
        model="google/gemma-3-1b-pt",
        declarations=dict(
            C1_label_set=[dict(word=w, angle_deg=d) for w, d in COMPASS],
            C1_source="src/gfdisc/ontologies.py DIRECTIONS, committed 3cadb69 2026-08-18",
            C1_surface_form="leading-space lowercase; single-token for all eight",
            C1_labels_used=dir_words, C1_angles_used=dir_angles, C1_dropped=dir_drop,
            C1_exploratory_capitalized=caps_words,
            C1_capitalized_note="prereg 3.4 registers the capitalized forms as carrying a "
                                "place-name sense ('the North') -- the declared compass "
                                "hazard. Collected as a contrast, never in a verdict.",
            C2_power_gate="k=8 reference-ring table at the realized support, computed "
                          "before the sweep; < 8 survivors -> untestable, abort",
            C3_readouts=dict(token="the direction token itself -- embedding-confounded",
                             suffix="first token AFTER the direction -- PRIMARY computed",
                             final="last token of the template -- tertiary, exploratory"),
            C4_discriminator="closure; ordering alone is never evidence",
            C5_nulls=dict(uniform_angles=N_GAP_NULL, background_draws=a.n_background,
                          background_pool="single-token common nouns, same arity, "
                                          "identical templates and positions",
                          background_secondary="place names (prereg 4.2c registers the "
                                               "place-name pool for compass)",
                          order_permutations=N_PERM_ORDER,
                          shuffle_control=N_SHUFFLE_CONTROL,
                          background_not_scored_for_ordering="random words have no true "
                                                             "cyclic order"),
            C5_prereg_discrepancy=dict(
                prereg_registers=300, used=int(a.n_background),
                reason="an add-one empirical p is floored at 1/(n+1); at 300 the floor "
                       "(3.3e-3) sits above BH's rank-1 threshold over 78 cells "
                       "(6.4e-4), so no isolated cell could survive BH however clean. "
                       "At 2000 the floor is 5.0e-4. More draws only tighten the null.",
                direction="conservative -- makes a positive harder, never easier"),
            C6_exclusion_rule="no background word may carry an established directional or "
                              "compass sense, no place name may contain a direction "
                              "substring, and calendar words are excluded from every pool",
            C6_exclusions=sorted(DIRECTION_SENSE_EXCLUSIONS),
            C6_direction_substrings=list(DIRECTION_SUBSTRINGS),
            C6_excluded_from_noun_pool=noun_excluded,
            C6_excluded_from_place_pool=place_excluded,
            C7_sign_conventions="both PC2 sign conventions reported; rho must be "
                                "identical (reflection-invariant), rotation-only rho "
                                "exposes handedness",
            C8_expected_confound="token readout closing at layers 0-2 is the embedding "
                                 "confound, declared in advance, not a finding",
            C9_cell_outcome_rule=dict(
                compass_ring="p_closed_uniform < 0.05 AND p_vs_noun_background < 0.05 "
                             "AND ordering p < 0.05",
                closed_ring_unordered="both closure tests pass, ordering does not",
                ordered_arc="ordering passes, closure does not",
                nothing="neither", degenerate="point cloud carries no usable geometry"),
            C10_widened_plane_search=dict(
                status="POST-HOC ADDITION to the prereg's registered method family",
                provenance="rule imported verbatim from scripts/24_colors_existence.py "
                           "D10; NOT in " + PREREG,
                declared="in this file, before any compass activation was read -- not "
                         "added after a negative",
                rule="all C(4,2)=6 planes from the top 4 PCs; the plane minimising the "
                     "angular gap ratio is selected. Selection is LABEL-FREE.",
                closure="the uniform-angle p on a selected plane is invalid and is "
                        "marked as such; the valid test is the matched background with "
                        "the identical min-over-6-planes selection applied to every draw",
                verdicts="C9 and C12 verdicts use the PRIMARY statistic only"),
            C11_multiplicity="BH within this ontology's 78-cell sweep, reported with raw "
                             "counts and the expected-by-chance count",
            C12_verdict_vocabulary=[
                "recovered_blind (unreachable for compass -- prereg 3.4)",
                "structure_exists_supervised_only", "ordered_not_closed",
                "clean_negative", "untestable"],
            C13_no_rival_order="only musical keys has a registered rival ordering "
                               "(fifths vs chromatic); inventing one for compass would "
                               "be an unregistered addition",
            caveats_that_survive_a_positive=[
                "A closed ring at the suffix readout shows the information is present one "
                "token later; attention can copy an embedding forward, so this is not "
                "proof of recomputation. The principal angle to the layer-0 token plane "
                "is the only handle reported.",
                "Layers 0-4 are embedding-adjacent: a hit there validates the pipeline "
                "and locates geometry near the input; it is not a depth claim.",
                "structure_exists_supervised_only is a weaker claim than Phase 1's months "
                "result, because the label set was chosen by name.",
            ]),
        layers=layers, n_templates=len(TEMPLATES), templates=tmpl_report,
        tokenisation_audit=tokenisation_audit,
        token_audit=dict(directions_kept=dir_keep, directions_dropped=dir_drop,
                         capitalized_kept=caps_keep, capitalized_dropped=caps_drop,
                         capitalized_used=caps_words,
                         noun_pool_kept=noun_words, noun_pool_dropped=noun_drop,
                         place_pool_kept=place_words, place_pool_dropped=place_drop),
        embedding_reference_plane=dict(
            readout="token", layer=int(ref_layer),
            note="the plane every `principal_angles_*` field is measured against: the "
                 "direction point cloud read at the direction token itself, at the "
                 "earliest layer in this sweep."),
        power_gate=pw,
        reference_check=ref,
        synthetics=syn,
        blind_arm=blind,
        rows=rows,
        cardinals_only_descriptive=dict(
            note="prereg 3.4: four evenly-spread points pass the uniform-gap test "
                 "trivially (ref_p 5.0e-05 for a perfect k=4 ring), so this is a "
                 "descriptive placement statistic and is NEVER ring evidence",
            rows=card_rows),
        exploratory_capitalized=dict(
            note="place-name-sense contrast; EXPLORATORY, never in a verdict",
            rows=caps_rows),
        jackknife=jack_rows,
        multiplicity=mult,
        outcome_counts={ro: {name: [r["layer"] for r in rows
                                    if r["readout"] == ro and r["outcome"] == name]
                             for name in ("compass_ring", "closed_ring_unordered",
                                          "ordered_arc", "nothing", "degenerate")}
                        for ro in READOUTS},
        outcome_counts_secondary={
            ro: {name: [r["layer"] for r in rows
                        if r["readout"] == ro
                        and r["outcome_secondary_plane_search"] == name]
                 for name in ("compass_ring_widened_search", "closed_ring_unordered",
                              "ordered_arc", "nothing", "degenerate")}
            for ro in READOUTS},
        verdict=dict(
            ontology="compass directions", full_k=FULL_K,
            blind=blind["verdict"],
            blind_source=blind["verdict_source"],
            supervised=verdict,
            supervised_reason=verdict_why,
            n_cells=int(n_cells),
            n_raw_ordering_passes=int(n_ordered_raw),
            n_raw_closure_passes_both_nulls=int(n_closed_raw),
            expected_by_chance_at_05=float(0.05 * n_cells),
            compass_ring_cells_raw=ring_cells,
            compass_ring_cells_surviving_bh=survivors_bh,
            suffix_readout_cells=int(len(suffix_cells)),
            ceiling=blind["ceiling"]),
        runtime_sec=float(time.time() - t0),
        forward_sec=float(forward_sec),
    )

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=2, default=jsonable))

    # ---- summary -----------------------------------------------------------
    print("\n" + "=" * 86)
    print("PER-CELL OUTCOMES -- C9, PRIMARY pre-declared statistic")
    for ro in READOUTS:
        for name in ("compass_ring", "closed_ring_unordered", "ordered_arc", "nothing",
                     "degenerate"):
            ls = out["outcome_counts"][ro][name]
            if ls:
                print(f"  {ro:>8} {name:>22}: {len(ls):2d} layers {ls}")
    print("\nPER-CELL OUTCOMES -- C10 widened plane search (post-hoc, labelled)")
    for ro in READOUTS:
        for name in ("compass_ring_widened_search", "closed_ring_unordered",
                     "ordered_arc", "nothing", "degenerate"):
            ls = out["outcome_counts_secondary"][ro][name]
            if ls:
                print(f"  {ro:>8} {name:>28}: {len(ls):2d} layers {ls}")
    print(f"\nC11 MULTIPLICITY over {n_cells} live cells "
          f"(expected by chance at alpha=0.05: {0.05 * n_cells:.1f})")
    for key in ("ordering_primary", "closure_uniform_primary",
                "closure_background_primary", "closure_place_background_primary",
                "ordering_secondary", "closure_background_secondary"):
        m = mult[key]
        print(f"  {key:34s} raw p<0.05: {m['n_raw_below_05']:2d}   "
              f"survives BH: {m['n_surviving_bh_fdr']:2d}   "
              f"[p floor {m['p_value_floor']:.2e}, BH rank-1 {m['bh_rank1_threshold']:.2e}, "
              f"isolated cell reachable: {m['isolated_cell_can_survive_bh']}]"
              + (f"   {[(s['readout'], s['layer']) for s in m['surviving']]}"
                 if m["surviving"] else ""))
    print("\n" + "=" * 86)
    print(f"VERDICT (prereg 4.3 vocabulary)")
    print(f"  blind      : {blind['verdict']}   [{blind['verdict_source']}]")
    print(f"  supervised : {verdict}")
    print(f"               {verdict_why}")
    print(f"\nruntime {time.time() - t0:.1f}s -> wrote {a.out}")


if __name__ == "__main__":
    main()
