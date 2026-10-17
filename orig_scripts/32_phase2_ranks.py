"""Phase-2 MILITARY RANKS -- the in-run ORDINAL CONTROL.

Bound by docs/phase2_prereg_2026-08-21.md (frozen 2026-08-21). Section 3.5 restates
the committed prediction in its binding form:

    "Military ranks must ORDER and must NEVER CLOSE."

Ranks are not a discovery track. They are the instrument that calibrates every other
Phase-2 track: if the closure tier accepts an ontology that is a ladder rather than a
ring, then no Phase-2 closure positive can be trusted and the gate itself has to be
re-examined. That is why the alarm below is loud and is declared before anything runs.

WHAT IS DECLARED BEFORE ANY ACTIVATION IS READ (all written into the output JSON
under "declarations" and printed to the log first):

  R1  LABEL SET. `gfdisc.ontologies.MILITARY_RANKS`, IMPORTED not retyped, so the
      committed list from 3cadb69 is the one scored:
        private corporal sergeant lieutenant captain major colonel general
      ascending, rank index 0..7. Registered variants (prereg 3.5): '{w}', ' {w}',
      '{W}', ' {W}'. PRIMARY surface form: ' {w}' (lowercase, leading space) -- all
      eight are single tokens in that form. ' {W}' is carried as a declared
      CAPITALISATION SENSITIVITY arm, not as a second chance at a positive.

  R2  PREDICTED STRUCTURE, two halves, both binding.
      (i) ORDER. Spearman rho between the recovered 1-D coordinate (PC1 of the member
          means) and the rank index beats the 1000-draw shuffled-label control at
          p < 0.05. PC1 has an arbitrary sign, so the statistic is |rho| and the null
          is scored under the same absolute value. Both PC1 sign conventions are
          reported as signed rho; |rho| is invariant and that is asserted.
      (ii) NEVER CLOSE. The closure tier must FAIL: p_closed against 20,000 uniform
          angles AND p against the matched same-arity background must not both fall
          below 0.05.

  R3  READOUTS (prereg 4.2, inherited verbatim from scripts/24_colors_existence.py):
        token  -- the rank token itself. EMBEDDING-CONFOUNDED; a result here at
                  layers 0-2 is lexical geometry, not computation.
        suffix -- first token AFTER the rank word. PRIMARY COMPUTED READOUT.
        final  -- last token of the template. Tertiary, exploratory.
      26 layers x 3 readouts = 78 cells.

  R4  TEMPLATES. 16, each PREFIX + " " + WORD + SUFFIX, suffix >= 3 tokens so the
      suffix and final positions are distinct, suffix token-identical across the eight
      words within a template, and no template contains another member of the label
      set. Prereg 3.5 requires the prefix to force the MILITARY sense, because the
      three ranks that dominate the corpus (' private', ' major', ' general') are
      dominated there by non-military senses. Every prefix names soldiering, rank,
      commission, insignia, regiment or command before the word appears.

  R5  NULLS.
      (a) uniform random angles, k of them, 20,000 draws (same table as
          scripts/12_closure_depth.py and scripts/24_colors_existence.py).
      (b) 2000 draws of k matched single-token common nouns through the IDENTICAL
          templates at the IDENTICAL positions. THIS IS THE TIER THAT DECIDES closure.
          Reported as an empirical p and as a false-acceptance rate at alpha = 0.05.
      (c) 2000 draws of k non-military OCCUPATION/ROLE nouns -- the class-matched
          secondary pool. PROVENANCE: prereg 4.2(c) requires a grammaticality/
          class-matched secondary pool per ontology but names one only for keys
          (capital letters H-Z), compass (place names) and zodiac (personal names).
          The ranks pool is declared HERE, at script-freeze time, before any
          activation: a rank sits in a role-noun slot, so the class match is a role
          noun with no canonical seniority order.
      (d) ordering: the registered 1000-draw shuffled-label control, PLUS the EXACT
          enumeration over all 8! = 40,320 label assignments. The exact p is not a
          different statistic; it is the same permutation test carried to completion,
          and it is declared now because the 1000-draw p is floored at 1/1001 = 9.99e-4
          while the BH rank-1 threshold over 78 cells is 0.05/78 = 6.4e-4 -- so no
          isolated cell could ever survive BH on the floored p (the failure mode
          scripts/24_colors_existence.py recorded as amendment D12-5). The BINDING
          test is the registered 1000-draw control; the exact p is what the
          multiplicity column uses, and both are reported side by side.
      Background draws are NOT scored for ordering: random nouns have no true rank
      order. Closure is the background criterion.

  R6  POOL EXCLUSION RULE, declared before the pools are used. Excluded from both
      background pools: every military or naval term, every rank, title or
      chain-of-command word, every word belonging to a canonical seniority ladder
      (junior/senior, intern, associate, deputy, assistant, chief, principal,
      director, manager, supervisor, apprentice, master, trainee, cadet, ...), and
      every calendar word, so the pools cannot inherit this project's other
      ontologies. The two pools are made DISJOINT: occupation words that appear in
      the verified colours noun pool (farmer, doctor, teacher, student) are moved out
      of the common-noun pool into the role pool.

  R7  BOTH SVD SIGN CONVENTIONS. For closure, convention A/B is the PC2 sign flip;
      gap_ratio is identical under it and that is asserted, not assumed, and the
      max-over-symmetries cyclic rho is reflection-invariant and that is asserted too.
      For the ORDINAL statistic the sign that matters is PC1's: convention A is +PC1,
      convention B is -PC1, signed rho flips, |rho| does not, and both are reported.

  R8  EXPECTED CONFOUND, stated in advance so it cannot be reported as a finding: an
      ordering at the `token` readout at layers 0-2 is the rank words' own embedding
      geometry. The claim this script can support is ordering at the SUFFIX readout.

  R9  VERDICTS. Per (readout, layer) cell:
        ordered_not_closed                 ordering passes, closure does not  <- PREDICTED
        closure_passed_PREDICTION_FAILED   closure passes both nulls          <- ALARM
        nothing                            neither
        degenerate                         no usable geometry
      TRACK verdict, decided in this order and fixed now:
        1. closure_passed_PREDICTION_FAILED  if any live cell closes both nulls AND
           its background p survives BH within the 78-cell sweep. (Prereg 4.4 and 7:
           an isolated single-cell pass is not a finding, so BH is the gate; the raw
           count and the expected-by-chance count are reported next to it either way.)
        2. ordered_not_closed  if, with no such closure, at least one cell at the
           PRIMARY `suffix` readout has ordering surviving BH within the sweep.
        3. clean_negative  otherwise, and only if the synthetic battery passed, since
           a "no structure" claim is worth nothing without demonstrated power.
      `untestable` is unreachable in this arm: all eight ranks are single-token.

  R10 THE ALARM RULE, spelled out. A closure pass here is NOT a ranks success and is
      NOT to be re-read as one (prereg 7). It is evidence that the closure tier
      accepts a non-cyclic ontology, and it invalidates the Phase-2 closure tier until
      the gate is re-examined. It is printed in capitals, counted raw and BH-corrected,
      and reported with the per-cell false-acceptance rate of the exact conjunction
      rule measured on the matched-noun draws themselves.

  R11 MULTIPLICITY. BH-FDR within this ontology's own sweep (78 cells), reported with
      the raw count, the expected-by-chance count (0.05 x 78 = 3.9 per tier) and each
      p-value's floor.

  R12 SYNTHETIC BATTERY, through the identical code path, run BEFORE the model loads;
      failure aborts the run.
        planted 1-D monotone arc (a thin sausage) -- the MATCHED POSITIVE CONTROL for
            ranks per prereg 4.2: must ORDER and must NOT close.
        planted k=8 ring -- must CLOSE against both nulls. Without this row a "closure
            failed" result would be indistinguishable from a closure tier with no power
            at k = 8. Prereg 6 puts a perfect 8-ring at ref_p_closed = 5.0e-05.
        planted 200-degree arc -- orders, does not close.
        pure noise -- neither.

  R13 ATTACKS ON ANY POSITIVE (project tenet 4), all pre-declared:
        - polysemy subset: ordering recomputed on the five unambiguously military
          ranks (corporal, sergeant, lieutenant, captain, colonel), dropping the three
          the prereg names as non-military-dominant (private, major, general).
        - leave-one-out jackknife of the ordinal rho at every cell that orders.
        - composition controls: Spearman of PC1 against log corpus count, token id and
          word length, plus the partial correlation of PC1 with rank index controlling
          for log corpus count.
        - template sensitivity: the 16 templates split into halves 1-8 / 9-16 and the
          whole statistic recomputed on each half.
        - capitalisation sensitivity: the ' {W}' variant arm.
        - SECONDARY closure stress: the min-over-6-planes widened search of
          scripts/24_colors_existence.py D10, with the identical freedom given to every
          background draw. Reported as a stress tier only; the R9 alarm is computed on
          the PRIMARY top-2 plane, because a selected plane can only make closure MORE
          likely and the alarm must not be manufacturable by search.

  R14 BLIND ARM: registered `untestable` in prereg 3.5 and 5.5 (3 of 8 ranks clear the
      >= 100 count floor, versus the 6 needed at ceil(2/3 * 8), and those three are the
      polysemy-dominated ones). Prereg 7 FORBIDS reporting a blind-arm negative for
      ranks. What is reported instead is descriptive placement read out of the frozen
      artifacts only, with NO re-clustering, NO new thresholds and NO new arms:
        B1 a scan of all 51 ring-flagged clusters in verdict.other_rings for any
           registered rank variant -- this is the blind half of the R10 alarm;
        B2 a scan of every label-scored cluster's recorded token_strings, same purpose;
        B3 the frozen artifact's own numbers_one_to_ten placements, the sibling ordinal
           control the artifact declared as "must not close", as calibration;
        B4 the pool counts re-derived independently from cache/sparse_big/layer_0/
           tokens.i32, checking prereg 5.5.
      The frozen artifact carries per-cluster MEMBERSHIP only for flagged and
      label-scored clusters, so the cluster placement of ' private' / ' major' /
      ' general' is not computable without re-deriving the partitions, which prereg
      4.1 forbids. That is stated as a limit of the artifact, not as a negative.

CAVEAT THAT SURVIVES ANY RESULT (prereg 4.5). Ordering at the suffix readout shows the
information is present one token later; attention can copy an embedding forward, so it
is not proof of recomputation. The principal angle between each cell's plane and the
layer-0 token-readout plane is the only handle reported on that.
"""
import argparse, json, os, sys, time
from itertools import permutations
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from gfdisc.env import DEVICE, D_MODEL
from gfdisc.model import load_model, capture
from gfdisc.fits import (degenerate, pca_2d, fit_circle, circularity, is_ring, angles,
                         cyclic_order_score, _rank_of)
from gfdisc.nulls import benjamini_hochberg
from gfdisc.ontologies import MILITARY_RANKS, ONTOLOGIES

# ---------------------------------------------------------------------------
# R1 -- label set, imported from the 3cadb69 commit rather than retyped.
# ---------------------------------------------------------------------------
RANKS = list(MILITARY_RANKS)
RANK_INDEX = list(range(len(RANKS)))
# prereg 3.5 names these three as the ranks whose dominant corpus sense is not
# military; the R13 polysemy attack drops them.
NON_MILITARY_DOMINANT = ("private", "major", "general")
MILITARY_ONLY = [w for w in RANKS if w not in NON_MILITARY_DOMINANT]

# ---------------------------------------------------------------------------
# R4 -- templates. Every prefix establishes the military sense before the word.
# ---------------------------------------------------------------------------
TEMPLATES = [
    ("The soldier was promoted to", " last year in the army"),
    ("After ten years of service he became a", " in the regiment"),
    ("The officer holds the rank of", " in the infantry"),
    ("She was commissioned as a", " after basic training"),
    ("Every enlisted soldier salutes the", " on the parade ground"),
    ("The battalion is commanded by a", " with combat experience"),
    ("He served under a", " during the desert campaign"),
    ("The military rank of", " sits within the chain of command"),
    ("They addressed the soldier as", " throughout the briefing"),
    ("The regiment roster lists one", " for every platoon"),
    ("A newly commissioned", " reported for duty"),
    ("The war record names the", " who led the assault"),
    ("Uniform insignia identify a", " at a glance"),
    ("The army promoted the recruit to", " within two years"),
    ("Command of the garrison passed to the", " at first light"),
    ("Service records show the veteran retired as a", " in that war"),
]
READOUTS = ("token", "suffix", "final")

# ---------------------------------------------------------------------------
# R6 -- exclusion rule, written verbatim into the JSON.
# ---------------------------------------------------------------------------
RANK_SENSE_EXCLUSIONS = [
    # military / naval / police
    "soldier", "sailor", "marine", "officer", "trooper", "cadet", "recruit", "veteran",
    "commander", "chief", "admiral", "brigadier", "ensign", "gunner", "ranger",
    "guard", "sentry", "pilot", "airman", "infantry", "regiment", "platoon",
    "battalion", "brigade", "squad", "troop", "army", "navy", "corps", "militia",
    "rifle", "cannon", "tank", "bayonet", "medal", "insignia", "uniform", "barracks",
    "war", "battle", "combat", "campaign", "salute", "command", "rank",
    # seniority ladders of any kind
    "junior", "senior", "intern", "associate", "deputy", "assistant", "principal",
    "director", "manager", "supervisor", "apprentice", "master", "trainee",
    "president", "vice", "executive", "head", "lead", "boss", "founder", "partner",
    "professor", "lecturer", "reader", "dean", "provost", "bishop", "cardinal",
    "prince", "duke", "earl", "baron", "knight", "king", "queen", "emperor",
    "grade", "level", "tier", "class", "degree", "step", "stage", "order",
    # calendar (inherited exclusion, prereg 4.2)
    "january", "february", "march", "april", "may", "june", "july", "august",
    "september", "october", "november", "december", "monday", "tuesday",
    "wednesday", "thursday", "friday", "saturday", "sunday", "spring", "summer",
    "autumn", "winter",
]

# (b) common nouns. Base list is the verified pool of scripts/24_colors_existence.py;
# the four occupation words in it are moved to the role pool so the pools are disjoint.
OCCUPATIONS_IN_COLOUR_POOL = ["farmer", "doctor", "teacher", "student"]
NOUN_POOL = [
    "table", "chair", "window", "door", "garden", "river", "mountain", "bridge",
    "letter", "market", "village", "engine", "ticket", "bottle", "camera", "pencil",
    "kitchen", "office", "school", "hospital",
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
# (c) class-matched secondary pool: role nouns with no canonical seniority order.
ROLE_POOL = OCCUPATIONS_IN_COLOUR_POOL + [
    "plumber", "baker", "painter", "dancer", "singer", "writer", "driver", "waiter",
    "tailor", "barber", "butcher", "gardener", "cleaner", "dentist", "nurse", "chef",
    "lawyer", "banker", "actor", "artist", "poet", "miner", "weaver", "potter",
    "jeweller", "librarian", "janitor", "cashier", "courier", "tutor", "editor",
    "carpenter", "welder", "mechanic", "electrician", "architect", "surveyor",
    "chemist", "biologist", "historian", "novelist", "pianist", "violinist",
    "drummer", "cyclist", "swimmer", "climber", "shepherd", "brewer", "grocer",
    "florist", "optician", "farmer", "potter", "sculptor", "translator", "typist",
    "clerk", "porter", "usher", "guide", "vendor", "trader", "broker", "auditor",
    "planner", "designer", "engineer", "programmer", "analyst", "researcher",
]

# ---------------------------------------------------------------------------
# R15 -- PRE-ACTIVATION AMENDMENTS. Everything changed between the first draft of
# this script and the run that produced the output, all of it made BEFORE any
# activation was read, listed so a reader can see exactly what was touched.
# Nothing in R1-R14 -- label set, readouts, statistic, verdict rule, alarm rule --
# was altered.
# ---------------------------------------------------------------------------
PRE_ACTIVATION_AMENDMENTS = [
    "TEMPLATE SUFFIX LENGTHS. R4 requires every suffix to be >= 3 tokens so the "
    "`suffix` and `final` readouts are distinct positions. Two of the sixteen drafted "
    "suffixes tokenised to 2: ' per platoon' -> ' for every platoon' (3 tokens) and "
    "' at dawn' -> ' at first light' (3 tokens). Found by the script's own R4 assertion "
    "on a 3-layer dry run, before any activation was scored. Prefixes untouched.",
]

N_BACKGROUND = 2000
N_PERM_ORDER = 20000
N_SHUFFLE_CONTROL = 1000
N_GAP_NULL = 20000
MAX_EXACT_N = 9                       # 9! = 362,880 is the enumeration ceiling

ARTIFACT = "results/token_type_geometry.json"
CACHE_TOKENS = "cache/sparse_big/layer_0/tokens.i32"


# ---------------------------------------------------------------------------
# Closure null (identical construction to 12_closure_depth.gap_pvalue).
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
    t = gap_table(k)
    return float((1 + int(np.searchsorted(t, observed_max_gap_rad, side="right"))) / (1 + len(t)))


# ---------------------------------------------------------------------------
# R2(i)/R5(d) -- the ORDINAL statistic. |Spearman| between PC1 and the rank index.
# ---------------------------------------------------------------------------
def _spear(a, b, n):
    d = np.asarray(a, float) - np.asarray(b, float)
    return 1.0 - 6.0 * float(d @ d) / (n * (n * n - 1))


def ordinal_stats(coord, rank_index, seed=0, n_shuffle=N_SHUFFLE_CONTROL,
                  exact=True, jitter_seed=0):
    """|rho| between the PC1 coordinate and the rank index, with two nulls.

    Ties in `coord` are broken by a random jitter, not by index order: an argsort of
    tied values returns index order, which IS the rank order here, and would manufacture
    rho = 1 out of a degenerate cloud. That failure mode is recorded in
    gfdisc.fits.cyclic_order_score and is guarded the same way.

    Returns the registered 1000-draw shuffled-label p AND, when n! is small enough, the
    EXACT p over every label assignment. The exact p is the same permutation test taken
    to completion; it is reported because the shuffled p is floored at 1/(n_shuffle+1),
    above the BH rank-1 threshold of the 78-cell sweep.
    """
    c = np.asarray(coord, float)
    n = len(c)
    jit = np.random.default_rng(jitter_seed).permutation(n)
    order = np.lexsort((jit, c))
    r_c = np.empty(n, float)
    r_c[order] = np.arange(n, dtype=float)
    r_l = _rank_of(np.asarray(rank_index, float))

    rho_signed = _spear(r_c, r_l, n)
    obs = abs(rho_signed)

    rng = np.random.default_rng(seed + 4242)
    null = np.empty(n_shuffle)
    for i in range(n_shuffle):
        null[i] = abs(_spear(r_c, r_l[rng.permutation(n)], n))
    p_shuf = float((1 + int((null >= obs - 1e-12).sum())) / (1 + n_shuffle))

    out = dict(n=int(n), rho_signed_convA=float(rho_signed),
               rho_signed_convB=float(-rho_signed), rho_abs=float(obs),
               p_shuffled_label_control=p_shuf, n_shuffles=int(n_shuffle),
               p_shuffled_floor=float(1.0 / (n_shuffle + 1)),
               null_mean=float(null.mean()), null_p95=float(np.percentile(null, 95)),
               null_max=float(null.max()))
    if exact and n <= MAX_EXACT_N:
        P = np.asarray(list(permutations(range(n))), dtype=np.int64)
        base = r_l[P]
        d = r_c[None, :] - base
        rr = 1.0 - 6.0 * (d * d).sum(1) / (n * (n * n - 1))
        out["p_exact_enumeration"] = float((np.abs(rr) >= obs - 1e-12).mean())
        out["n_permutations_enumerated"] = int(len(P))
        out["p_exact_floor"] = float(2.0 / len(P))
    else:
        out["p_exact_enumeration"] = None
        out["n_permutations_enumerated"] = 0
        out["p_exact_floor"] = None
    return out


# ---------------------------------------------------------------------------
# Cyclic ordering (reported for comparability with the rest of the project).
# ---------------------------------------------------------------------------
def order_stats(ang, label_index, K, flips=(1, -1), n_perm=N_PERM_ORDER, seed=0, block=5000):
    """Vectorised copy of fits.cyclic_order_score; asserted equal in check_against_reference."""
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
                flips=[int(f) for f in flips])


def check_against_reference(seed=0):
    """order_stats(flips=(1,-1)) must equal fits.cyclic_order_score; and the exact
    ordinal enumeration must agree with a large shuffled null on a random case."""
    rng = np.random.default_rng(seed)
    out = []
    for t in range(5):
        k = 8
        ang = rng.uniform(-np.pi, np.pi, size=k)
        lab = np.arange(k)
        a = cyclic_order_score(ang, lab, k, n_perm=2000, seed=t)
        b = order_stats(ang, lab, k, n_perm=2000, seed=t)
        out.append(dict(trial=t, rho_ref=a["rho"], rho_new=b["rho"],
                        sym_ref=list(a["symmetry"]), sym_new=b["symmetry"],
                        rho_match=bool(abs(a["rho"] - b["rho"]) < 1e-12)))
    # ordinal: a perfectly monotone coordinate must give the exact minimum p = 2/8!
    mono = ordinal_stats(np.arange(8.0), np.arange(8), seed=0)
    ord_ok = bool(abs(mono["rho_abs"] - 1.0) < 1e-12
                  and abs(mono["p_exact_enumeration"] - 2.0 / 40320) < 1e-12)
    # and a reversed coordinate must be identical (|rho|), which is the PC1 sign
    # convention assertion in miniature.
    rev = ordinal_stats(np.arange(8.0)[::-1].copy(), np.arange(8), seed=0)
    sign_ok = bool(abs(rev["rho_abs"] - mono["rho_abs"]) < 1e-12
                   and abs(rev["rho_signed_convA"] + mono["rho_signed_convA"]) < 1e-12)
    return dict(trials=out, cyclic_passed=bool(all(x["rho_match"] for x in out)),
                ordinal_monotone_exact_p=mono["p_exact_enumeration"],
                ordinal_exact_ok=ord_ok, ordinal_sign_invariance_ok=sign_ok,
                passed=bool(all(x["rho_match"] for x in out) and ord_ok and sign_ok))


# ---------------------------------------------------------------------------
# The one analysis routine. Ranks, backgrounds and synthetics all go through it.
# ---------------------------------------------------------------------------
def analyse(P, rank_index=None, K=None, seed=0, do_order=True, both_conventions=True):
    P = np.asarray(P, dtype=np.float64)
    k = len(P)
    deg = bool(degenerate(P))
    xy, ratios, Vt = pca_2d(P)
    out = dict(k=k, degenerate=deg, top2_var=float(ratios[:2].sum()),
               pc1_var=float(ratios[0]),
               pc_var=[float(x) for x in ratios[:min(5, len(ratios))]])
    if do_order and not deg and rank_index is not None:
        # R2(i): the ORDINAL statistic lives on PC1 and is independent of the PC2 sign.
        out["ordinal"] = ordinal_stats(xy[:, 0], rank_index, seed=seed, jitter_seed=seed)
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
        if do_order and not deg and rank_index is not None:
            ang = angles(z, cx, cy)
            KK = int(K if K is not None else k)
            d["cyclic_order"] = order_stats(ang, rank_index, KK, flips=(1, -1),
                                            n_perm=N_PERM_ORDER, seed=seed)
            d["cyclic_order_rotation_only"] = order_stats(ang, rank_index, KK, flips=(1,),
                                                          n_perm=N_SHUFFLE_CONTROL, seed=seed)
        convs[name] = d
    out["conventions"] = convs
    return out, Vt[:2], xy


# ---------------------------------------------------------------------------
# R13 -- SECONDARY closure stress: min over the 6 planes of the top 4 PCs.
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


def secondary_analyse(P, rank_index=None, seed=0, n_pc=N_PC_SECONDARY):
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
    return out, best["basis"]


# ---------------------------------------------------------------------------
# R12 -- synthetic battery.
# ---------------------------------------------------------------------------
def synthetics(d=D_MODEL, k=8, noise=0.15, n_bg=N_BACKGROUND, seed=0):
    rng = np.random.default_rng(seed)
    u, v = rng.normal(size=d), rng.normal(size=d)
    u /= np.linalg.norm(u)
    v -= v @ u * u
    v /= np.linalg.norm(v)
    lab = np.arange(k)

    def amb(scale):
        """Isotropic noise of per-point NORM `scale` (colours D12: the sqrt(d) matters)."""
        return (scale / np.sqrt(d)) * rng.normal(size=(k, d))

    bg_rng = np.random.default_rng(seed + 12345)
    bg = np.empty(n_bg); bg_p_unif = np.empty(n_bg); bg_best = np.empty(n_bg)
    for i in range(n_bg):
        g, pu, gb, _ = fast_closure(bg_rng.normal(size=(k, d)))
        bg[i], bg_p_unif[i], bg_best[i] = g, pu, gb

    def row(P, seed_, labels=lab):
        r, _, _ = analyse(P, labels, k, seed=seed_, both_conventions=False)
        a = r["conventions"]["A"]
        s, _ = secondary_analyse(P, labels, seed=seed_)
        return dict(gap_ratio=a["gap_ratio"], radius_ratio=a["radius_ratio"],
                    p_closed=a["p_closed_uniform"],
                    p_vs_background=float((1 + int((bg <= a["gap_ratio"]).sum())) / (1 + n_bg)),
                    rho_ordinal=r["ordinal"]["rho_abs"],
                    p_ordinal_shuffle=r["ordinal"]["p_shuffled_label_control"],
                    p_ordinal_exact=r["ordinal"]["p_exact_enumeration"],
                    rho_cyclic=a["cyclic_order"]["rho"], p_cyclic=a["cyclic_order"]["p"],
                    top2_var=r["top2_var"],
                    secondary_gap_ratio=s["gap_ratio"],
                    secondary_p_vs_background=float(
                        (1 + int((bg_best <= s["gap_ratio"]).sum())) / (1 + n_bg)))

    out = dict(noise=noise, k=k, d=d)
    th = 2 * np.pi * np.arange(k) / k

    # The MATCHED positive control for ranks (prereg 4.2): a 1-D monotone arc.
    # Planted as a thin sausage in the fitted plane, not "a line plus isotropic
    # 1152-d noise", which is not a line at all -- colours recorded that trap.
    line = (np.outer(np.linspace(-1, 1, k), u)
            + np.outer(0.05 * rng.normal(size=k), v) + amb(0.34))
    out["planted_monotone_arc"] = row(line, 4)
    out["planted_monotone_arc_note"] = (
        "the ranks-matched positive control: must ORDER and must NOT close")

    ring = np.outer(np.cos(th), u) + np.outer(np.sin(th), v) + amb(noise)
    out["planted_ring_k8"] = row(ring, 1)
    out["planted_ring_note"] = (
        "power check for the closure tier at k = 8. Without a closing ring here, "
        "'ranks never close' would be uninterpretable. prereg 6: a perfect 8-ring "
        "scores ref_p_closed = 5.0e-05 against the uniform-angle null")

    ta = np.linspace(0, np.radians(200), k)
    arc = np.outer(np.cos(ta), u) + np.outer(np.sin(ta), v) + amb(noise)
    out["planted_arc_200deg"] = row(arc, 3)

    nz = [dict(seed=s, **row(rng.normal(size=(k, d)), s)) for s in range(10, 15)]
    out["pure_noise"] = dict(
        draws=nz,
        n_sig_closed_uniform=int(sum(x["p_closed"] <= 0.05 for x in nz)),
        n_sig_closed_background=int(sum(x["p_vs_background"] <= 0.05 for x in nz)),
        n_sig_ordinal=int(sum(x["p_ordinal_shuffle"] <= 0.05 for x in nz)),
        n_sig_closed_background_secondary=int(
            sum(x["secondary_p_vs_background"] <= 0.05 for x in nz)))
    out["uniform_null_calibration"] = dict(
        n_draws=int(n_bg), pool="isotropic Gaussian, same k and d",
        gap_ratio_mean=float(bg.mean()), gap_ratio_p05=float(np.percentile(bg, 5)),
        gap_ratio_min=float(bg.min()),
        uniform_null_false_acceptance_at_p05=float((bg_p_unif <= 0.05).mean()),
        joint_rule_false_acceptance=float(
            ((bg_p_unif <= 0.05) & (bg <= np.percentile(bg, 5))).mean()),
        note="the joint rule is the R9 closure rule (uniform AND background); its "
             "false-acceptance rate on structureless clouds is what the ALARM count "
             "must be read against")

    LN, RG, AR = out["planted_monotone_arc"], out["planted_ring_k8"], out["planted_arc_200deg"]
    out["passed"] = bool(
        # the ordinal control must ORDER and must NOT close -- the dissociation
        LN["p_ordinal_shuffle"] <= 0.05 and LN["p_closed"] > 0.05
        and LN["p_vs_background"] > 0.05
        # the closure tier must have power at k = 8, or "never closes" says nothing
        and RG["p_closed"] <= 0.05 and RG["p_vs_background"] <= 0.05
        # an open arc must order and must not close
        and AR["p_ordinal_shuffle"] <= 0.05 and AR["p_closed"] > 0.05
        and AR["p_vs_background"] > 0.05
        # and noise must do neither, on both tiers
        and out["pure_noise"]["n_sig_closed_background"] <= 1
        and out["pure_noise"]["n_sig_closed_background_secondary"] <= 1)
    return out


# ---------------------------------------------------------------------------
# Activations
# ---------------------------------------------------------------------------
def token_audit(tok, words, prefix=" "):
    keep, drop = [], []
    for w in words:
        ids = tok.encode(prefix + w, add_special_tokens=False)
        (keep if len(ids) == 1 else drop).append(
            dict(word=w, surface=prefix + w, n_tokens=len(ids), ids=[int(i) for i in ids]))
    return keep, drop


def build_and_verify(tok, prefix, suffix, word, word_id, bos):
    pre = tok.encode(prefix, add_special_tokens=False)
    suf = tok.encode(suffix, add_special_tokens=False)
    manual = [bos] + pre + [word_id] + suf
    natural = tok.encode(prefix + " " + word + suffix, add_special_tokens=True)
    return manual, natural, len(pre), len(suf)


def collect(model, tok, word_ids, layers, chunk=48):
    """One forward per (template, chunk); three positions per layer, TEMPLATE HALVES kept.

    Halves 1-8 and 9-16 are accumulated separately so the R13 template-sensitivity
    check costs no extra forward pass. The full mean is (h0 + h1) / 2 exactly, because
    the halves hold the same number of templates.
    """
    bos = tok.bos_token_id
    n = len(word_ids)
    half = len(TEMPLATES) // 2
    acc = [{ro: {l: np.zeros((n, D_MODEL), dtype=np.float64) for l in layers}
            for ro in READOUTS} for _ in range(2)]
    with torch.inference_mode():
        for ti, (prefix, suffix) in enumerate(TEMPLATES):
            h = 0 if ti < half else 1
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
                    hh = st[l]
                    for ro in READOUTS:
                        acc[h][ro][l][lo:hi] += hh[:, pos[ro]].float().cpu().numpy()
                del st
    for h in (0, 1):
        for ro in READOUTS:
            for l in layers:
                acc[h][ro][l] /= half
    return acc


def principal_angles_deg(A, B):
    Qa = np.linalg.qr(np.asarray(A, dtype=np.float64).T)[0]
    Qb = np.linalg.qr(np.asarray(B, dtype=np.float64).T)[0]
    s = np.clip(np.linalg.svd(Qa.T @ Qb, compute_uv=False), -1.0, 1.0)
    return [float(np.degrees(np.arccos(x))) for x in s]


# ---------------------------------------------------------------------------
# R13 -- composition controls
# ---------------------------------------------------------------------------
def composition_controls(coord, rank_index, counts, token_ids, words, seed=0):
    """Is the recovered 1-D order the rank ladder, or a nuisance variable?

    log corpus count, token id and word length are each correlated with the recovered
    coordinate, and the partial correlation of coordinate with rank index CONTROLLING
    for log count is reported with its own shuffled-label p. All of it is descriptive:
    the binding statistic is R2(i).
    """
    c = np.asarray(coord, float)
    n = len(c)
    r_c = _rank_of(c)
    r_rank = _rank_of(np.asarray(rank_index, float))
    lc = np.log(np.maximum(np.asarray(counts, float), 1.0))
    nuis = dict(log_corpus_count=lc,
                token_id=np.asarray(token_ids, float),
                word_length_chars=np.asarray([len(w) for w in words], float))
    out = dict(rho_coord_vs_rank=float(_spear(r_c, r_rank, n)))
    for name, v in nuis.items():
        rv = _rank_of(v)
        out[f"rho_coord_vs_{name}"] = float(_spear(r_c, rv, n))
        out[f"rho_rank_vs_{name}"] = float(_spear(r_rank, rv, n))

    # partial Spearman: residualise both rank-transformed variables on log-count ranks
    z = np.column_stack([np.ones(n), _rank_of(lc)])
    def resid(y):
        return y - z @ np.linalg.lstsq(z, y, rcond=None)[0]
    ra, rb = resid(r_c), resid(r_rank)
    denom = np.linalg.norm(ra) * np.linalg.norm(rb)
    pc = float(ra @ rb / denom) if denom > 1e-12 else 0.0
    rng = np.random.default_rng(seed + 99)
    null = np.empty(N_SHUFFLE_CONTROL)
    for i in range(N_SHUFFLE_CONTROL):
        rp = resid(r_rank[rng.permutation(n)])
        dn = np.linalg.norm(ra) * np.linalg.norm(rp)
        null[i] = abs(float(ra @ rp / dn)) if dn > 1e-12 else 0.0
    out["partial_rho_coord_vs_rank_given_log_count"] = pc
    out["p_partial_shuffled"] = float((1 + int((null >= abs(pc) - 1e-12).sum()))
                                      / (1 + N_SHUFFLE_CONTROL))
    return out


# ---------------------------------------------------------------------------
# R14 -- blind arm, read-only from the frozen artifacts. No re-clustering.
# ---------------------------------------------------------------------------
def blind_arm(artifact_path=ARTIFACT, tokens_path=CACHE_TOKENS, tokenizer=None):
    art = json.loads(Path(artifact_path).read_text())
    variants = {}
    for w in RANKS:
        variants[w] = [w, " " + w, w.capitalize(), " " + w.capitalize()]
    all_variants = {v for vs in variants.values() for v in vs}

    # B1 -- the 51 ring-flagged clusters
    hits_rings = []
    for r in art["verdict"]["other_rings"]:
        m = [t for t in r["token_strings"] if t in all_variants]
        if m:
            hits_rings.append(dict(layer=r["layer"], arm=r["arm"], k=r["k"],
                                   p_closed=r["p_closed"], p_bg=r["p_bg"],
                                   rank_variants_present=m,
                                   token_strings=r["token_strings"]))
    # B2 -- every label-scored cluster's recorded token_strings
    hits_lsc = []
    n_lsc, n_lsc_trunc = 0, 0
    for L in art["layers"]:
        for c in L.get("label_scored_clusters", []):
            n_lsc += 1
            ts = c.get("token_strings") or []
            if c.get("n_strings_truncated"):
                n_lsc_trunc += 1
            m = [t for t in ts if t in all_variants]
            if m:
                hits_lsc.append(dict(layer=c["layer"], arm=c["arm"],
                                     cluster_id=c["cluster_id"], size=c["size"],
                                     flagged=c["flagged"], screened=c["screened"],
                                     rank_variants_present=m,
                                     n_strings_listed=len(ts),
                                     n_strings_truncated=c.get("n_strings_truncated")))
    # B3 -- the sibling ordinal control the artifact itself declared
    nums = []
    for L in art["layers"]:
        for e in L["label_placement"]:
            if e["label_set"] == "numbers_one_to_ten":
                sc = e["best_cluster_screening"]
                nums.append(dict(layer=e["layer"], arm=e["arm"],
                                 best_cluster_id=e["best_cluster_id"],
                                 best_cluster_size=e["best_cluster_size"],
                                 n_labels_in_best_cluster=e["n_labels_in_best_cluster"],
                                 fraction_of_set_together=e["fraction_of_set_together"],
                                 in_screen_band=e["best_cluster_in_screen_band"],
                                 p_closed=(sc or {}).get("p_closed"),
                                 p_bg=(sc or {}).get("p_bg"),
                                 flagged=(sc or {}).get("flagged"),
                                 reason_unscreened=e["reason_unscreened"]))
    screened_nums = [x for x in nums if x["in_screen_band"]]
    n_num_flagged = sum(1 for x in screened_nums if x["flagged"])
    # the artifact records a "best cluster" even when it holds ZERO labels of the set,
    # so a flagged best-cluster is only a numbers ring if the set is actually in it.
    # min_labels_needed for numbers_one_to_ten is ceil(2/3 * 10) = 7.
    flagged_holding_labels = [x for x in screened_nums
                              if x["flagged"] and x["n_labels_in_best_cluster"] >= 1]
    flagged_holding_majority = [x for x in screened_nums
                                if x["flagged"] and x["n_labels_in_best_cluster"] >= 7]

    # B4 -- independent re-derivation of the prereg 5.5 pool counts
    counts_out = {}
    if tokenizer is not None and Path(tokens_path).exists():
        toks = np.fromfile(tokens_path, dtype=np.int32)
        bc = np.bincount(toks, minlength=int(toks.max()) + 1)
        for w, vs in variants.items():
            per = {}
            for s in vs:
                ids = tokenizer.encode(s, add_special_tokens=False)
                if len(ids) != 1:
                    per[s] = dict(multi_token=True, n_tokens=len(ids), count=None)
                else:
                    i = int(ids[0])
                    per[s] = dict(multi_token=False, token_id=i,
                                  count=int(bc[i]) if i < len(bc) else 0)
            single = {s: v for s, v in per.items() if not v["multi_token"]}
            best = max(single.items(), key=lambda kv: kv[1]["count"]) if single else (None, None)
            counts_out[w] = dict(per_variant=per,
                                 best_variant=best[0],
                                 best_count=(best[1]["count"] if best[1] else None),
                                 over_floor_100=bool(best[1] and best[1]["count"] >= 100),
                                 over_floor_30=bool(best[1] and best[1]["count"] >= 30))
        del toks, bc

    n100 = sum(1 for v in counts_out.values() if v["over_floor_100"])
    n30 = sum(1 for v in counts_out.values() if v["over_floor_30"])
    return dict(
        status="untestable",
        registered_where="docs/phase2_prereg_2026-08-21.md sections 3.5, 5.5, 5.6, 7",
        why_untestable=(f"{n100} of 8 ranks clear the >= 100 discovery floor, against the "
                        f"ceil(2/3 * 8) = 6 the artifact's own label convention needs, and "
                        f"the three that clear it are exactly the three whose dominant "
                        f"corpus sense is not military"),
        prereg_forbids="reporting a blind-arm NEGATIVE for ranks (prereg section 7)",
        B1_flagged_ring_scan=dict(
            n_flagged_rings_scanned=len(art["verdict"]["other_rings"]),
            n_containing_a_rank_variant=len(hits_rings), hits=hits_rings,
            meaning="the blind half of the R10 alarm: does any label-blind ring-flagged "
                    "cluster in the frozen catalogue contain a military rank?"),
        B2_label_scored_cluster_scan=dict(
            n_clusters_scanned=n_lsc, n_with_truncated_string_lists=n_lsc_trunc,
            n_containing_a_rank_variant=len(hits_lsc), hits=hits_lsc,
            caveat="token_strings lists are truncated in the artifact for large clusters, "
                   "so absence here is weaker than absence in B1"),
        B3_sibling_ordinal_control_numbers_one_to_ten=dict(
            placements=nums, n_placements=len(nums),
            n_in_screen_band=len(screened_nums), n_flagged_best_clusters=n_num_flagged,
            n_flagged_holding_at_least_one_number=len(flagged_holding_labels),
            n_flagged_holding_a_majority_of_the_set=len(flagged_holding_majority),
            flagged_detail=[x for x in screened_nums if x["flagged"]],
            p_closed_min_in_band=(min(x["p_closed"] for x in screened_nums)
                                  if screened_nums else None),
            p_closed_max_in_band=(max(x["p_closed"] for x in screened_nums)
                                  if screened_nums else None),
            p_closed_min_where_set_present=(
                min([x["p_closed"] for x in screened_nums
                     if x["n_labels_in_best_cluster"] >= 7] or [None])),
            note="the artifact declared numbers_one_to_ten an ORDINAL CONTROL that must "
                 "not close; this is its measured behaviour in the frozen partitions. "
                 "A 'best cluster' is recorded even when it holds zero labels of the "
                 "set, so the flagged count is split by how many numbers were actually "
                 "in it"),
        B4_pool_counts=dict(source=tokens_path, counts=counts_out,
                            n_over_floor_100=n100, n_over_floor_30=n30,
                            prereg_5_5_says="3 of 8 at >= 100, 4 of 8 at >= 30"),
        placement_limit=("the frozen artifact carries per-cluster MEMBERSHIP only for "
                         "flagged and label-scored clusters, so the cluster placement of "
                         "' private' / ' major' / ' general' is not computable from it; "
                         "prereg 4.1 forbids re-deriving the partitions, so no placement "
                         "is claimed and no negative is reported"))


# ---------------------------------------------------------------------------
# POST-SWEEP DIAGNOSTIC, provenance stated. This block was added AFTER the first
# full sweep, to make one specific number interpretable: the tertiary `final`
# readout produced a run of RAW closure passes on an ontology that is a ladder.
# It changes NO verdict rule and adds NO search family (prereg 7). It is two
# things: (a) a decomposition of the already-declared R10 alarm count by the
# already-declared R3 readouts, and (b) a data-free geometric constant.
# ---------------------------------------------------------------------------
def monotonicity_ceiling(k, n_phi=20001):
    """Largest |Spearman(1-D projection, cycle index)| a PERFECT k-ring can produce.

    Touches no data and no labels beyond the arity: the k points are placed at their
    true cycle angles 2*pi*i/k and projected onto every in-plane direction. A cloud
    whose ordinal |rho| EXCEEDS this ceiling cannot be a closed cycle in label order,
    whatever the closure statistic says about it -- which is the only way to read a
    cell that is simultaneously 'closed' and near-perfectly monotone.
    """
    th = 2 * np.pi * np.arange(k) / k
    idx = np.arange(k, dtype=float)
    best, arg = 0.0, None
    for phi in np.linspace(0.0, 2 * np.pi, n_phi):
        v = np.cos(th + phi)
        r = abs(_spear(_rank_of(v), idx, k))
        if r > best:
            best, arg = r, float(phi)
    return dict(k=int(k), max_abs_ordinal_rho=float(best), argmax_phi_rad=arg,
                n_directions_scanned=int(n_phi))


def closure_diagnostic(rows, ks=(5, 8)):
    ceilings = {int(k): monotonicity_ceiling(int(k)) for k in ks}
    ceil8 = ceilings[8]["max_abs_ordinal_rho"]
    per_ro = {}
    for ro in READOUTS:
        rs = [r for r in rows if r["readout"] == ro and not r["degenerate"]]
        closing = sorted(r["layer"] for r in rs if r["closed"])
        bands, prev = [], None
        for l in closing:
            if prev is None or l != prev + 1:
                bands.append([l])
            else:
                bands[-1].append(l)
            prev = l
        fa = float(np.mean([r["background_nouns"]["joint_rule_false_acceptance"]
                            for r in rs])) if rs else 0.0
        per_ro[ro] = dict(
            n_cells=len(rs), n_closed_raw=len(closing), layers_closed=closing,
            n_contiguous_layer_bands=len(bands),
            contiguous_layer_bands=[[b[0], b[-1]] for b in bands],
            n_uniform_tier_raw=int(sum(r["conventions"]["A"]["p_closed_uniform"] < 0.05
                                       for r in rs)),
            n_noun_tier_raw=int(sum(r["background_nouns"]["p_vs_background"] < 0.05
                                    for r in rs)),
            mean_joint_rule_false_acceptance=fa,
            expected_closed_by_chance=float(fa * len(rs)),
            n_ordered=int(sum(r["ordered"] for r in rs)),
            max_rho_ordinal=float(max([r["ordinal"]["rho_abs"] for r in rs], default=0.0)))
    closing_cells = [r for r in rows if r["closed"] and not r["degenerate"]]
    over = [dict(readout=r["readout"], layer=r["layer"],
                 rho_ordinal=r["ordinal"]["rho_abs"],
                 gap_ratio=r["conventions"]["A"]["gap_ratio"],
                 radius_ratio=r["conventions"]["A"]["radius_ratio"],
                 exceeds_perfect_ring_ceiling=bool(r["ordinal"]["rho_abs"] > ceil8))
            for r in closing_cells]
    return dict(
        provenance="ADDED AFTER the first full sweep. It decomposes the pre-declared "
                   "R10 alarm count by the pre-declared R3 readouts and adds one "
                   "data-free geometric constant. No verdict rule was changed and no "
                   "new search family was added (prereg section 7).",
        monotonicity_ceiling_of_a_perfect_ring=ceilings,
        ceiling_note=(f"a perfect closed 8-cycle in label order can project onto at most "
                      f"|rho_ordinal| = {ceil8:.4f} in ANY in-plane direction. A cell "
                      f"that 'closes' while scoring above that ceiling is a monotone arc "
                      f"the max-gap statistic failed to reject, not a cycle."),
        by_readout=per_ro,
        closing_cells=over,
        n_closing_cells_above_ceiling=int(sum(x["exceeds_perfect_ring_ceiling"]
                                              for x in over)),
        reading=("closure at k = 8 is weak: prereg section 6 already showed one dropout "
                 "from an 8-ring pushes ref_p_closed to 0.0731, and this run measures the "
                 "uniform-angle tier's false-acceptance on isotropic 1152-d clouds at "
                 "0.11-0.13. The raw closure passes here are that weakness, not a ring."))


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
    ap.add_argument("--out", default="results/phase2_ranks.json")
    ap.add_argument("--layers", default=",".join(str(i) for i in range(26)))
    ap.add_argument("--n-background", type=int, default=N_BACKGROUND)
    ap.add_argument("--chunk", type=int, default=48)
    a = ap.parse_args()
    t0 = time.time()
    layers = [int(x) for x in a.layers.split(",")]

    print("=" * 84)
    print("PHASE 2 -- MILITARY RANKS: the in-run ORDINAL CONTROL")
    print("bound by docs/phase2_prereg_2026-08-21.md section 3.5")
    print("PREDICTION (binding): ranks must ORDER and must NEVER CLOSE.")
    print("DECLARATIONS (fixed before any activation is read):")
    print(f"  R1 label set      : {RANKS}  (imported from gfdisc.ontologies, commit 3cadb69)")
    print(f"     rank index     : {RANK_INDEX}; primary surface ' {{w}}'")
    print(f"  R2 ordering       : |Spearman(PC1, rank index)| vs {N_SHUFFLE_CONTROL}-draw "
          f"shuffled labels (binding) + exact 8! enumeration")
    print(f"     closure        : must FAIL -- p_uniform AND p_matched_background")
    print(f"  R3 readouts       : token (embedding-confounded) | suffix (PRIMARY) | final")
    print(f"  R4 templates      : {len(TEMPLATES)}, military sense forced in every prefix")
    print(f"  R5 nulls          : {N_GAP_NULL} uniform angles; {a.n_background} matched "
          f"noun draws; {a.n_background} role-noun draws; {N_PERM_ORDER} cyclic perms")
    print(f"  R9 verdicts       : ordered_not_closed (PREDICTED) | "
          f"closure_passed_PREDICTION_FAILED (ALARM) | nothing | degenerate")
    print(f"  R10 ALARM         : a closure pass is NOT a ranks success. It invalidates "
          f"the Phase-2 closure tier.")
    print(f"  R14 blind arm     : registered UNTESTABLE (prereg 3.5/5.5); descriptive "
          f"placement only, no negative reported")
    print(f"  R15 amendments    : {len(PRE_ACTIVATION_AMENDMENTS)}, all made before any "
          f"activation was read:")
    for i, am in enumerate(PRE_ACTIVATION_AMENDMENTS, 1):
        print(f"     ({i}) {am.split('.')[0]}.")
    print("=" * 84, flush=True)

    ref = check_against_reference()
    print(f"reference check: cyclic rho vs fits.cyclic_order_score "
          f"{'MATCH' if ref['cyclic_passed'] else 'MISMATCH'}; "
          f"ordinal exact p on a perfect monotone = {ref['ordinal_monotone_exact_p']:.3e} "
          f"(= 2/8!) {'OK' if ref['ordinal_exact_ok'] else 'FAIL'}; "
          f"PC1 sign invariance {'OK' if ref['ordinal_sign_invariance_ok'] else 'FAIL'}",
          flush=True)
    if not ref["passed"]:
        raise SystemExit("statistic reference check failed")

    syn = synthetics(k=len(RANKS), n_bg=a.n_background)
    print("\nR12 synthetic battery (identical code path):")
    for key in ("planted_monotone_arc", "planted_ring_k8", "planted_arc_200deg"):
        r = syn[key]
        print(f"  {key:24s} gapR {r['gap_ratio']:7.2f}  pUnif {r['p_closed']:.5f}  "
              f"pBg {r['p_vs_background']:.5f}  |rho_ord| {r['rho_ordinal']:.3f}  "
              f"pOrd {r['p_ordinal_shuffle']:.4f}  pOrdExact {r['p_ordinal_exact']:.2e}")
    uc = syn["uniform_null_calibration"]
    print(f"  uniform-angle null vs {uc['n_draws']} isotropic clouds (k={syn['k']}, "
          f"d={syn['d']}): mean gapR {uc['gap_ratio_mean']:.2f}, "
          f"FA(uniform) {uc['uniform_null_false_acceptance_at_p05']:.3f}, "
          f"FA(joint R9 rule) {uc['joint_rule_false_acceptance']:.4f}")
    print(f"  pure noise 5 draws: {syn['pure_noise']['n_sig_closed_background']}/5 "
          f"background-closure trips -> "
          f"{'PASS' if syn['passed'] else 'FAIL'}", flush=True)
    if not syn["passed"]:
        raise SystemExit("synthetic battery failed; not running on activations")

    model, tok = load_model(token=os.environ.get("HF_TOKEN"))
    bos = tok.bos_token_id

    # ---- R14 blind arm (read-only artifacts) ------------------------------
    blind = blind_arm(tokenizer=tok)
    print(f"\nR14 BLIND ARM (untestable by registration; descriptive only)")
    print(f"  B1 flagged rings scanned {blind['B1_flagged_ring_scan']['n_flagged_rings_scanned']}"
          f" -> containing a rank variant: "
          f"{blind['B1_flagged_ring_scan']['n_containing_a_rank_variant']}")
    print(f"  B2 label-scored clusters scanned "
          f"{blind['B2_label_scored_cluster_scan']['n_clusters_scanned']}"
          f" -> containing a rank variant: "
          f"{blind['B2_label_scored_cluster_scan']['n_containing_a_rank_variant']}")
    b3 = blind["B3_sibling_ordinal_control_numbers_one_to_ten"]
    print(f"  B3 numbers_one_to_ten: {b3['n_in_screen_band']}/{b3['n_placements']} "
          f"placements in the screening band, "
          f"{b3['n_flagged_best_clusters']} flagged best-clusters of which "
          f"{b3['n_flagged_holding_a_majority_of_the_set']} actually hold a majority of "
          f"the set; p_closed in band [{b3['p_closed_min_in_band']:.4g}, "
          f"{b3['p_closed_max_in_band']:.4g}], and where the set is present "
          f"min p_closed = {b3['p_closed_min_where_set_present']}")
    print(f"  B4 pool counts re-derived: {blind['B4_pool_counts']['n_over_floor_100']}/8 "
          f">= 100, {blind['B4_pool_counts']['n_over_floor_30']}/8 >= 30 "
          f"(prereg 5.5 says 3/8 and 4/8)", flush=True)

    # ---- token audits -----------------------------------------------------
    rank_keep, rank_drop = token_audit(tok, RANKS)
    if rank_drop:
        raise SystemExit(f"rank words not single-token in ' {{w}}' form: {rank_drop}")
    rank_cap_keep, rank_cap_drop = token_audit(tok, [w.capitalize() for w in RANKS])
    excl = {w.lower() for w in RANK_SENSE_EXCLUSIONS}
    noun_cand = [w for w in dict.fromkeys(NOUN_POOL) if w.lower() not in excl]
    role_cand = [w for w in dict.fromkeys(ROLE_POOL) if w.lower() not in excl]
    role_cand = [w for w in role_cand if w not in set(noun_cand)]
    noun_keep, noun_drop = token_audit(tok, noun_cand)
    role_keep, role_drop = token_audit(tok, role_cand)
    noun_words = [x["word"] for x in noun_keep]
    role_words = [x["word"] for x in role_keep]
    K = len(RANKS)
    print(f"\nR1 audit: {K} ranks, all single-token as ' {{w}}': "
          f"{[x['ids'][0] for x in rank_keep]}")
    print(f"R6 audit: {len(noun_words)} nouns, {len(role_words)} role nouns kept "
          f"({len(noun_drop)} / {len(role_drop)} dropped multi-token); pools disjoint: "
          f"{len(set(noun_words) & set(role_words)) == 0}", flush=True)

    # ---- template verification -------------------------------------------
    cap_words = [x["word"] for x in rank_cap_keep]
    label_words = RANKS + cap_words
    pool_words = noun_words + role_words

    def tokenises_everywhere(w):
        ids = tok.encode(" " + w, add_special_tokens=False)
        if len(ids) != 1:
            return False, dict(word=w, reason="multi-token", n_tokens=len(ids))
        for prefix, suffix in TEMPLATES:
            m, nat, _, _ = build_and_verify(tok, prefix, suffix, w, ids[0], bos)
            if m != nat:
                return False, dict(word=w, prefix=prefix, manual=m, natural=nat)
        return True, None

    label_bad, pool_bad = [], []
    for w in RANKS:
        ok, info = tokenises_everywhere(w)
        if not ok:
            label_bad.append(info)
    cap_bad = []
    for w in cap_words:
        ok, info = tokenises_everywhere(w)
        if not ok:
            cap_bad.append(info)
    for w in pool_words:
        ok, info = tokenises_everywhere(w)
        if not ok:
            pool_bad.append(info)
    dropped_pool = {d["word"] for d in pool_bad}
    noun_words = [w for w in noun_words if w not in dropped_pool]
    role_words = [w for w in role_words if w not in dropped_pool]
    cap_used = [w for w in cap_words if w not in {d["word"] for d in cap_bad}]

    tmpl_report = []
    for prefix, suffix in TEMPLATES:
        pre = tok.encode(prefix, add_special_tokens=False)
        suf = tok.encode(suffix, add_special_tokens=False)
        tmpl_report.append(dict(prefix=prefix, suffix=suffix, n_prefix_tokens=len(pre),
                                n_suffix_tokens=len(suf), rank_index_in_prompt=1 + len(pre),
                                suffix_index=2 + len(pre),
                                final_index=1 + len(pre) + len(suf),
                                first_post_rank_token=tok.decode(suf[:1])))
    if any(t["n_suffix_tokens"] < 3 for t in tmpl_report):
        raise SystemExit("a template suffix is shorter than 3 tokens (R4)")
    tokenisation_audit = dict(
        n_words_checked=len(label_words) + len(pool_words), n_templates=len(TEMPLATES),
        n_checks=(len(label_words) + len(pool_words)) * len(TEMPLATES),
        label_words_checked=RANKS, label_mismatches=label_bad,
        capitalised_variant_checked=cap_words, capitalised_mismatches=cap_bad,
        capitalised_used=cap_used,
        pool_words_checked=len(pool_words), pool_words_dropped=sorted(dropped_pool),
        pool_mismatch_detail=pool_bad,
        rule="rank mismatch aborts; pool mismatch drops the word (mechanical, label-blind)")
    print(f"tokenisation audit: {tokenisation_audit['n_checks']} word x template checks, "
          f"{len(label_bad)} rank mismatches, {len(cap_bad)} capitalised mismatches, "
          f"{len(dropped_pool)} pool words dropped {sorted(dropped_pool)}", flush=True)
    if label_bad:
        raise SystemExit(f"rank words with off-manifold ids: {label_bad}")
    if len(noun_words) < 40 or len(role_words) < 25:
        raise SystemExit(f"background pools too small: {len(noun_words)} nouns, "
                         f"{len(role_words)} role nouns")

    words = RANKS + cap_used + noun_words + role_words
    word_ids = [tok.encode(" " + w, add_special_tokens=False)[0] for w in words]
    idx = {w: i for i, w in enumerate(words)}
    rank_idx = np.array([idx[w] for w in RANKS])
    cap_idx = np.array([idx[w] for w in cap_used])
    noun_idx = np.array([idx[w] for w in noun_words])
    role_idx = np.array([idx[w] for w in role_words])
    mil_pos = np.array([RANKS.index(w) for w in MILITARY_ONLY])

    # corpus counts for the composition controls (same source as prereg 5.5)
    rank_counts = [blind["B4_pool_counts"]["counts"][w]["per_variant"][" " + w]["count"]
                   if blind["B4_pool_counts"]["counts"] else 1 for w in RANKS]
    rank_token_ids = [int(word_ids[idx[w]]) for w in RANKS]

    # ---- forwards ---------------------------------------------------------
    print(f"\ncollecting {len(words)} words x {len(TEMPLATES)} templates x "
          f"{len(layers)} layers x 3 readouts (template halves kept separately) ...",
          flush=True)
    tf = time.time()
    H = collect(model, tok, word_ids, layers, chunk=a.chunk)
    print(f"forwards done in {time.time() - tf:.1f}s", flush=True)
    del model
    if DEVICE == "mps":
        torch.mps.empty_cache()

    def M(ro, l):
        return 0.5 * (H[0][ro][l] + H[1][ro][l])

    ref_layer = layers[0]
    _, basis_L0_token, _ = analyse(M("token", ref_layer)[rank_idx], RANK_INDEX, K,
                                   seed=0, do_order=False, both_conventions=False)

    rows, jack_rows = [], []
    print(f"\n{'readout':>8} {'L':>3} {'top2':>6} {'pc1':>6} {'gapR':>7} {'pUnif':>8} "
          f"{'pNoun':>8} {'FAn':>6} {'pRole':>8} | {'rhoOrd':>7} {'pOrd':>8} "
          f"{'pOrdEx':>9} {'rhoCyc':>7} {'ang(L0)':>8} {'sec.pN':>7}  verdict", flush=True)
    for ro in READOUTS:
        for l in layers:
            P = M(ro, l)[rank_idx]
            res, basis, xy = analyse(P, RANK_INDEX, K, seed=l, do_order=True)
            A = res["conventions"]["A"]
            B = res["conventions"]["B"]
            sign_check = dict(
                gap_ratio_identical=bool(abs(A["gap_ratio"] - B["gap_ratio"]) < 1e-9),
                cyclic_rho_identical=(None if res["degenerate"] else
                                      bool(abs(A["cyclic_order"]["rho"]
                                               - B["cyclic_order"]["rho"]) < 1e-12)),
                cyclic_rot_only_rho_A=(None if res["degenerate"]
                                       else A["cyclic_order_rotation_only"]["rho"]),
                cyclic_rot_only_rho_B=(None if res["degenerate"]
                                       else B["cyclic_order_rotation_only"]["rho"]),
                ordinal_rho_abs_invariant_to_PC1_sign=(
                    None if res["degenerate"] else
                    bool(abs(abs(res["ordinal"]["rho_signed_convA"])
                             - abs(res["ordinal"]["rho_signed_convB"])) < 1e-12)))

            sec, sec_basis = secondary_analyse(P, RANK_INDEX, seed=l)

            def bg(pool_idx, n_draws, seed):
                r = np.random.default_rng(seed)
                gr = np.empty(n_draws); pc = np.empty(n_draws)
                rr = np.empty(n_draws); gb = np.empty(n_draws)
                block = M(ro, l)
                for i in range(n_draws):
                    pick = pool_idx[r.choice(len(pool_idx), size=K, replace=False)]
                    gr[i], pc[i], gb[i], rr[i] = fast_closure(block[pick])
                p_bg = float((1 + int((gr <= A["gap_ratio"]).sum())) / (1 + n_draws))
                thr = np.percentile(gr, 5)
                return dict(n_draws=int(n_draws), gap_ratio_mean=float(gr.mean()),
                            gap_ratio_p05=float(thr), gap_ratio_min=float(gr.min()),
                            radius_ratio_mean=float(rr.mean()),
                            false_acceptance_rate_p05=float((pc <= 0.05).mean()),
                            joint_rule_false_acceptance=float(
                                ((pc <= 0.05) & (gr <= thr)).mean()),
                            p_vs_background=p_bg,
                            secondary_best_plane=dict(
                                gap_ratio_mean=float(gb.mean()),
                                gap_ratio_p05=float(np.percentile(gb, 5)),
                                gap_ratio_min=float(gb.min()),
                                p_vs_background=float((1 + int((gb <= sec["gap_ratio"]).sum()))
                                                      / (1 + n_draws))))

            b_noun = bg(noun_idx, a.n_background, 1000 + l + 100 * READOUTS.index(ro))
            b_role = bg(role_idx, a.n_background, 5000 + l + 100 * READOUTS.index(ro))
            sec["p_vs_noun_background"] = b_noun["secondary_best_plane"]["p_vs_background"]
            sec["p_vs_role_background"] = b_role["secondary_best_plane"]["p_vs_background"]

            row = dict(readout=ro, layer=l, k=K,
                       **{kk: res[kk] for kk in ("degenerate", "top2_var", "pc1_var", "pc_var")},
                       ordinal=res.get("ordinal"), conventions=res["conventions"],
                       sign_convention_check=sign_check,
                       secondary_plane_search=sec,
                       background_nouns=b_noun, background_roles=b_role,
                       principal_angles_to_L0_token_plane_deg=principal_angles_deg(
                           basis, basis_L0_token))

            # ---- R9 verdict, PRIMARY plane only ---------------------------
            if res["degenerate"]:
                verdict = "degenerate"
                closed = ordered = False
            else:
                closed = bool(A["p_closed_uniform"] < 0.05
                              and b_noun["p_vs_background"] < 0.05)
                ordered = bool(res["ordinal"]["p_shuffled_label_control"] < 0.05)
                verdict = ("closure_passed_PREDICTION_FAILED" if closed else
                           "ordered_not_closed" if ordered else "nothing")
            row["closed"] = closed
            row["ordered"] = ordered
            row["verdict"] = verdict
            row["closed_secondary_stress"] = bool(
                (not res["degenerate"]) and sec["p_vs_noun_background"] < 0.05)

            # ---- R13 attacks ---------------------------------------------
            if not res["degenerate"]:
                row["composition_controls"] = composition_controls(
                    xy[:, 0], RANK_INDEX, rank_counts, rank_token_ids, RANKS, seed=l)
                # polysemy subset: the five unambiguously military ranks
                Pm = P[mil_pos]
                rm, _, _ = analyse(Pm, list(range(len(mil_pos))), len(mil_pos), seed=l,
                                   do_order=True, both_conventions=False)
                am = rm["conventions"]["A"]
                row["military_only_subset"] = dict(
                    words=MILITARY_ONLY, k=len(mil_pos),
                    rho_ordinal=rm["ordinal"]["rho_abs"],
                    p_ordinal_shuffle=rm["ordinal"]["p_shuffled_label_control"],
                    p_ordinal_exact=rm["ordinal"]["p_exact_enumeration"],
                    gap_ratio=am["gap_ratio"], p_closed_uniform=am["p_closed_uniform"])
                # capitalisation sensitivity
                if len(cap_used) == K:
                    Pc = M(ro, l)[cap_idx]
                    rc, _, _ = analyse(Pc, RANK_INDEX, K, seed=l, do_order=True,
                                       both_conventions=False)
                    ac = rc["conventions"]["A"]
                    row["capitalised_variant"] = dict(
                        rho_ordinal=rc["ordinal"]["rho_abs"],
                        p_ordinal_shuffle=rc["ordinal"]["p_shuffled_label_control"],
                        p_ordinal_exact=rc["ordinal"]["p_exact_enumeration"],
                        gap_ratio=ac["gap_ratio"], p_closed_uniform=ac["p_closed_uniform"])
                # template-half sensitivity
                halves = []
                for h in (0, 1):
                    rh, _, _ = analyse(H[h][ro][l][rank_idx], RANK_INDEX, K, seed=l,
                                       do_order=True, both_conventions=False)
                    ah = rh["conventions"]["A"]
                    halves.append(dict(
                        half=h, templates=f"{1 + h * 8}-{8 + h * 8}",
                        rho_ordinal=rh["ordinal"]["rho_abs"],
                        p_ordinal_shuffle=rh["ordinal"]["p_shuffled_label_control"],
                        p_ordinal_exact=rh["ordinal"]["p_exact_enumeration"],
                        gap_ratio=ah["gap_ratio"], p_closed_uniform=ah["p_closed_uniform"]))
                row["template_halves"] = halves
                # leave-one-out jackknife of the ordinal statistic where it orders
                if ordered:
                    jk = []
                    for j in range(K):
                        keep = [i for i in range(K) if i != j]
                        q, _, _ = analyse(P[keep], [RANK_INDEX[i] for i in keep], K,
                                          seed=l, do_order=True, both_conventions=False)
                        jk.append(dict(dropped=RANKS[j],
                                       rho_ordinal=q["ordinal"]["rho_abs"],
                                       p_ordinal_shuffle=q["ordinal"]["p_shuffled_label_control"],
                                       p_ordinal_exact=q["ordinal"]["p_exact_enumeration"],
                                       gap_ratio=q["conventions"]["A"]["gap_ratio"]))
                    jack_rows.append(dict(
                        readout=ro, layer=l, leave_one_out=jk,
                        worst=max(jk, key=lambda x: x["p_ordinal_shuffle"]),
                        n_still_ordered=int(sum(x["p_ordinal_shuffle"] < 0.05 for x in jk))))
            rows.append(row)

            o = res.get("ordinal")
            if o is None:
                print(f"{ro:>8} {l:3d} {res['top2_var']:6.3f} {res['pc1_var']:6.3f} "
                      f"{A['gap_ratio']:7.2f} {A['p_closed_uniform']:8.5f} "
                      f"{b_noun['p_vs_background']:8.5f} "
                      f"{b_noun['false_acceptance_rate_p05']:6.3f} "
                      f"{b_role['p_vs_background']:8.5f} | "
                      f"{'    n/a':>7} {'     n/a':>8} {'      n/a':>9} {'    n/a':>7} "
                      f"{'     n/a':>8} {sec['p_vs_noun_background']:7.4f}  {verdict}",
                      flush=True)
            else:
                print(f"{ro:>8} {l:3d} {res['top2_var']:6.3f} {res['pc1_var']:6.3f} "
                      f"{A['gap_ratio']:7.2f} {A['p_closed_uniform']:8.5f} "
                      f"{b_noun['p_vs_background']:8.5f} "
                      f"{b_noun['false_acceptance_rate_p05']:6.3f} "
                      f"{b_role['p_vs_background']:8.5f} | "
                      f"{o['rho_abs']:7.3f} {o['p_shuffled_label_control']:8.4f} "
                      f"{o['p_exact_enumeration']:9.2e} "
                      f"{A['cyclic_order']['rho']:+7.3f} "
                      f"{row['principal_angles_to_L0_token_plane_deg'][0]:8.1f} "
                      f"{sec['p_vs_noun_background']:7.4f}  {verdict}", flush=True)

    # ---- R11 multiplicity --------------------------------------------------
    live = [r for r in rows if not r["degenerate"]]
    n_cells = len(live)

    def fdr(pvals, label, n_null):
        p = np.asarray(pvals, dtype=float)
        keep = benjamini_hochberg(p, alpha=0.05) if len(p) else np.zeros(0, bool)
        floor = 1.0 / (n_null + 1.0) if n_null else 0.0
        rank1 = 0.05 / max(len(p), 1)
        return dict(statistic=label, n_cells=int(len(p)), n_null_draws=int(n_null),
                    n_raw_below_05=int((p < 0.05).sum()),
                    expected_by_chance_at_05=float(0.05 * len(p)),
                    p_value_floor=float(floor), bh_rank1_threshold=float(rank1),
                    isolated_cell_can_survive_bh=bool(floor <= rank1),
                    n_at_floor=int((p <= floor + 1e-12).sum()),
                    n_surviving_bh_fdr=int(keep.sum()),
                    surviving=[dict(readout=live[i]["readout"], layer=live[i]["layer"],
                                    p=float(p[i])) for i in np.flatnonzero(keep)])

    nb = a.n_background
    mult = dict(
        n_cells=int(n_cells),
        note="3 readouts x 26 layers, BH within this ontology's own sweep (prereg 4.4). "
             "An isolated pass at alpha = 0.05 is expected by chance and is not a "
             "finding. The ordering BH column uses the EXACT 8! p because the "
             "registered 1000-draw p is floored at 9.99e-4, above the BH rank-1 "
             "threshold of 6.4e-4 -- declared as R5(d), both are reported.",
        ordering_shuffled_registered=fdr(
            [r["ordinal"]["p_shuffled_label_control"] for r in live],
            "ordinal p, registered 1000-draw shuffled-label control", N_SHUFFLE_CONTROL),
        ordering_exact=fdr([r["ordinal"]["p_exact_enumeration"] for r in live],
                           "ordinal p, exact 8! enumeration", 40320 // 2),
        closure_uniform=fdr([r["conventions"]["A"]["p_closed_uniform"] for r in live],
                            "closure p vs uniform angles", N_GAP_NULL),
        closure_background_nouns=fdr([r["background_nouns"]["p_vs_background"] for r in live],
                                     "closure p vs matched noun background", nb),
        closure_background_roles=fdr([r["background_roles"]["p_vs_background"] for r in live],
                                     "closure p vs role-noun background", nb),
        closure_background_secondary=fdr(
            [r["secondary_plane_search"]["p_vs_noun_background"] for r in live],
            "closure p vs noun background, widened-plane stress tier", nb),
        cyclic_ordering=fdr([r["conventions"]["A"]["cyclic_order"]["p"] for r in live],
                            "cyclic ordering p (comparability only)", N_PERM_ORDER),
    )

    # ---- R9/R10 track verdict ---------------------------------------------
    closed_cells = [r for r in live if r["closed"]]
    bh_closure = {(s["readout"], s["layer"])
                  for s in mult["closure_background_nouns"]["surviving"]}
    closed_bh = [r for r in closed_cells if (r["readout"], r["layer"]) in bh_closure]
    bh_order = {(s["readout"], s["layer"]) for s in mult["ordering_exact"]["surviving"]}
    suffix_ordered_bh = [r for r in live
                         if r["readout"] == "suffix" and r["ordered"]
                         and (r["readout"], r["layer"]) in bh_order]

    if closed_bh:
        track = "closure_passed_PREDICTION_FAILED"
    elif suffix_ordered_bh:
        track = "ordered_not_closed"
    else:
        track = "clean_negative"

    alarm = dict(
        rule="prereg 3.5: a closure pass is a FAILURE of the rank prediction and "
             "evidence that the closure tier accepts a non-cyclic ontology",
        n_cells_closing_both_nulls_raw=len(closed_cells),
        cells_closing_raw=[dict(readout=r["readout"], layer=r["layer"],
                                gap_ratio=r["conventions"]["A"]["gap_ratio"],
                                p_uniform=r["conventions"]["A"]["p_closed_uniform"],
                                p_noun_background=r["background_nouns"]["p_vs_background"],
                                p_role_background=r["background_roles"]["p_vs_background"])
                           for r in closed_cells],
        n_cells_closing_surviving_bh=len(closed_bh),
        expected_by_chance_at_05_per_tier=float(0.05 * n_cells),
        mean_joint_rule_false_acceptance_on_noun_draws=float(np.mean(
            [r["background_nouns"]["joint_rule_false_acceptance"] for r in live])),
        n_cells_closing_widened_stress=int(sum(r["closed_secondary_stress"] for r in live)),
        triggered=bool(closed_bh),
        note="the widened-plane stress count is reported for completeness only; the "
             "alarm verdict is computed on the pre-declared primary top-2 plane, "
             "because a selected plane can only make closure more likely")

    diag = closure_diagnostic(rows)

    out = dict(
        status="exploratory Phase-2 under docs/phase2_prereg_2026-08-21.md",
        item="phase2_ranks", ontology="ranks", script=str(Path(__file__).name),
        generated=time.strftime("%Y-%m-%dT%H:%M:%S"),
        prereg=dict(document="docs/phase2_prereg_2026-08-21.md",
                    sections=["3.5", "4.1", "4.2", "4.3", "4.4", "4.5", "5.5", "5.6", "6", "7"],
                    committed_predictions="src/gfdisc/ontologies.py @ 3cadb69 (2026-08-18)",
                    ontology_entry=ONTOLOGIES["ranks"],
                    binding_prediction="ranks must ORDER and must NEVER CLOSE"),
        declarations=dict(
            R1_label_set=RANKS, R1_rank_index=RANK_INDEX,
            R1_variants_registered=["'{w}'", "' {w}'", "'{W}'", "' {W}'"],
            R1_primary_surface="' {w}'",
            R2_ordering_statistic="|Spearman(PC1 of member means, rank index)|",
            R2_ordering_null=f"{N_SHUFFLE_CONTROL}-draw shuffled-label control (binding) "
                             f"+ exact enumeration of all 8! label assignments",
            R2_closure_must_fail="p_closed_uniform < 0.05 AND p_vs_noun_background < 0.05 "
                                 "would be a FAILURE of the prediction",
            R3_readouts=dict(token="the rank token itself -- embedding-confounded",
                             suffix="first token AFTER the rank word -- PRIMARY",
                             final="last token of the template -- tertiary"),
            R4_n_templates=len(TEMPLATES),
            R4_template_rule="prefix forces the military sense; suffix >= 3 tokens and "
                             "token-identical across the eight words; no template "
                             "contains another rank",
            R5_nulls=dict(uniform_angles=N_GAP_NULL, background_draws=a.n_background,
                          background_primary="single-token common nouns, same arity, "
                                             "identical templates and positions",
                          background_secondary="non-military occupation/role nouns "
                                               "(class-matched; pool DECLARED HERE, "
                                               "prereg 4.2(c) names one only for keys, "
                                               "compass and zodiac)",
                          cyclic_permutations=N_PERM_ORDER,
                          shuffle_control=N_SHUFFLE_CONTROL,
                          background_not_scored_for_ordering="random nouns have no true "
                                                             "rank order"),
            R6_exclusion_rule="military/naval terms, ranks, titles, chain-of-command "
                              "words, every canonical seniority ladder and every "
                              "calendar word are excluded from both pools; the two "
                              "pools are disjoint",
            R6_exclusions=sorted(RANK_SENSE_EXCLUSIONS),
            R6_occupations_moved_out_of_noun_pool=OCCUPATIONS_IN_COLOUR_POOL,
            R7_sign_conventions="closure: PC2 flip, gap_ratio identical (asserted); "
                                "cyclic rho reflection-invariant (asserted); ordinal: "
                                "PC1 flip, signed rho flips, |rho| invariant (asserted)",
            R8_expected_confound="ordering at the token readout at layers 0-2 is the "
                                 "rank words' own embedding geometry, not a finding",
            R9_verdict_rule=dict(
                ordered_not_closed="ordinal p < 0.05 and closure does not pass -- PREDICTED",
                closure_passed_PREDICTION_FAILED="closure passes both nulls -- ALARM",
                nothing="neither", degenerate="no usable geometry",
                track_rule="1) alarm if any closing cell survives BH; else 2) "
                           "ordered_not_closed if a suffix cell's ordering survives BH; "
                           "else 3) clean_negative"),
            R10_alarm="a closure pass is NOT a ranks success and invalidates the Phase-2 "
                      "closure tier until the gate is re-examined (prereg 3.5, 7)",
            R11_multiplicity="BH-FDR within this ontology's 78-cell sweep",
            R12_synthetic_battery="planted monotone arc (matched positive control), "
                                  "planted k=8 ring (closure power check), 200-degree "
                                  "arc, pure noise; failure aborts before the model loads",
            R13_attacks=["polysemy subset (5 unambiguously military ranks)",
                         "leave-one-out jackknife", "composition controls "
                         "(log corpus count, token id, word length, partial correlation)",
                         "template-half sensitivity", "capitalisation sensitivity",
                         "widened-plane closure stress tier"],
            R14_blind_arm="registered untestable; descriptive placement only",
            R15_pre_activation_amendments=PRE_ACTIVATION_AMENDMENTS,
            R15_note="every change made between the first draft and the run that "
                     "produced this file, all before any activation was read; "
                     "R1-R14 untouched",
            caveat_that_survives_any_result=(
                "ordering at the suffix readout shows the information is present one "
                "token later; attention can copy an embedding forward, so it is not "
                "proof of recomputation. The principal angle to the layer-0 token plane "
                "is the only handle reported on that (prereg 4.5)."),
        ),
        model="google/gemma-3-1b-pt", layers=layers, n_templates=len(TEMPLATES),
        templates=tmpl_report, tokenisation_audit=tokenisation_audit,
        token_audit=dict(ranks_kept=rank_keep, ranks_dropped=rank_drop,
                         capitalised_kept=rank_cap_keep, capitalised_dropped=rank_cap_drop,
                         capitalised_used=cap_used,
                         noun_pool_kept=noun_words, noun_pool_dropped=noun_drop,
                         role_pool_kept=role_words, role_pool_dropped=role_drop,
                         rank_corpus_counts_of_the_surface_used={
                             w: c for w, c in zip(RANKS, rank_counts)},
                         rank_corpus_counts_note=(
                             "counts of the exact ' {w}' surface the templates use, from "
                             "cache/sparse_big/layer_0/tokens.i32 -- this is what the R13 "
                             "composition control regresses against. The best-variant "
                             "counts of prereg 5.5 (which pick ' Corporal', ' Sergeant', "
                             "' Lieutenant', ' Colonel', ' Captain') are in "
                             "blind_arm.B4_pool_counts")),
        embedding_reference_plane=dict(readout="token", layer=int(ref_layer)),
        reference_check=ref, synthetics=syn, blind_arm=blind,
        rows=rows, jackknife=jack_rows, multiplicity=mult, alarm=alarm,
        closure_diagnostic=diag,
        verdict_counts={ro: {name: [r["layer"] for r in rows
                                    if r["readout"] == ro and r["verdict"] == name]
                             for name in ("ordered_not_closed",
                                          "closure_passed_PREDICTION_FAILED",
                                          "nothing", "degenerate")}
                        for ro in READOUTS},
        verdict=dict(
            track="ranks", arm_supervised=track, arm_blind="untestable",
            prediction_confirmed=bool(track == "ordered_not_closed"),
            prediction_failed=bool(track == "closure_passed_PREDICTION_FAILED"),
            phase2_closure_tier_invalidated=bool(track == "closure_passed_PREDICTION_FAILED"),
            summary=None,
            caveat_for_every_other_phase2_track=(
                f"the PRIMARY `suffix` readout closed in "
                f"{diag['by_readout']['suffix']['n_closed_raw']}/26 layers and the "
                f"`token` readout in {diag['by_readout']['token']['n_closed_raw']}/26, "
                f"but the tertiary `final` readout closed RAW in "
                f"{diag['by_readout']['final']['n_closed_raw']}/26 "
                f"({diag['by_readout']['final']['n_contiguous_layer_bands']} contiguous "
                f"layer bands) against "
                f"{diag['by_readout']['final']['expected_closed_by_chance']:.2f} expected "
                f"from the measured false-acceptance of the same conjunction rule on the "
                f"matched-noun draws. None survived BH. "
                f"{diag['n_closing_cells_above_ceiling']} of those "
                f"{len(diag['closing_cells'])} cells score |rho_ordinal| above the "
                f"{diag['monotonicity_ceiling_of_a_perfect_ring'][8]['max_abs_ordinal_rho']:.4f} "
                f"ceiling a perfect 8-cycle in label order can reach in any in-plane "
                f"direction, so they are monotone arcs the max-gap statistic failed to "
                f"reject; the remainder are listed cell by cell in "
                f"closure_diagnostic.closing_cells. Read as: a RAW single-cell closure "
                f"pass at k = 8 is not evidence in Phase 2; only BH-surviving closure is.")),
    )
    out["verdict"]["summary"] = (
        f"supervised arm: {track}; blind arm: untestable by registration; "
        f"{len(closed_cells)}/{n_cells} cells closed both nulls raw "
        f"(expected by chance {0.05 * n_cells:.1f} per tier), "
        f"{len(closed_bh)} surviving BH; ordering at the primary suffix readout "
        f"survives BH in {len(suffix_ordered_bh)} cells")
    out["runtime_sec"] = float(time.time() - t0)

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=2, default=jsonable))

    # ---- summary -----------------------------------------------------------
    print("\n" + "=" * 84)
    print("R9 VERDICT COUNTS (primary top-2 plane)")
    for ro in READOUTS:
        for name in ("ordered_not_closed", "closure_passed_PREDICTION_FAILED",
                     "nothing", "degenerate"):
            ls = out["verdict_counts"][ro][name]
            if ls:
                print(f"  {ro:>8} {name:>34}: {len(ls):2d} layers {ls}")
    print(f"\nR11 MULTIPLICITY over {n_cells} live cells "
          f"(expected by chance at alpha=0.05: {0.05 * n_cells:.1f})")
    for key in ("ordering_shuffled_registered", "ordering_exact", "closure_uniform",
                "closure_background_nouns", "closure_background_roles",
                "closure_background_secondary", "cyclic_ordering"):
        m = mult[key]
        print(f"  {key:30s} raw p<0.05: {m['n_raw_below_05']:2d}   "
              f"BH-FDR: {m['n_surviving_bh_fdr']:2d}   "
              f"[floor {m['p_value_floor']:.2e}, BH rank-1 {m['bh_rank1_threshold']:.2e}, "
              f"isolated reachable: {m['isolated_cell_can_survive_bh']}]")
    print("\n" + "!" * 84)
    if alarm["triggered"]:
        print("*** ALARM: RANKS CLOSED. THE PHASE-2 PREDICTION FOR RANKS HAS FAILED. ***")
        print("*** A closure pass on an ORDINAL ontology means the closure tier accepts")
        print("*** a non-cyclic structure. Every Phase-2 closure positive must be held")
        print("*** until the gate is re-examined (prereg section 3.5 and section 7).")
        for c in alarm["cells_closing_raw"]:
            print(f"    {c['readout']:>8} L{c['layer']:<3} gapR {c['gap_ratio']:.2f} "
                  f"pUnif {c['p_uniform']:.5f} pNoun {c['p_noun_background']:.5f} "
                  f"pRole {c['p_role_background']:.5f}")
    else:
        print(f"NO ALARM: ranks never closed. {alarm['n_cells_closing_both_nulls_raw']} "
              f"raw cells closed both nulls out of {n_cells} "
              f"(expected by chance {alarm['expected_by_chance_at_05_per_tier']:.1f} per "
              f"tier); {alarm['n_cells_closing_surviving_bh']} survived BH.")
        print(f"  widened-plane stress tier closed in "
              f"{alarm['n_cells_closing_widened_stress']}/{n_cells} cells "
              f"(reported, not part of the verdict)")
    print("!" * 84)
    print("\nPOST-SWEEP CLOSURE DIAGNOSTIC (provenance: added after the first full sweep; "
          "decomposition only, no rule changed)")
    for ro in READOUTS:
        b = diag["by_readout"][ro]
        print(f"  {ro:>8}: closed raw {b['n_closed_raw']:2d}/{b['n_cells']} "
              f"(bands {b['contiguous_layer_bands']}), expected from the measured "
              f"conjunction false-acceptance {b['expected_closed_by_chance']:.2f}; "
              f"uniform tier {b['n_uniform_tier_raw']}, noun tier {b['n_noun_tier_raw']}; "
              f"ordered {b['n_ordered']}, max |rho_ord| {b['max_rho_ordinal']:.3f}")
    c8 = diag["monotonicity_ceiling_of_a_perfect_ring"][8]["max_abs_ordinal_rho"]
    print(f"  a PERFECT closed 8-cycle in label order can project onto at most "
          f"|rho_ordinal| = {c8:.4f}; "
          f"{diag['n_closing_cells_above_ceiling']}/{len(diag['closing_cells'])} of the "
          f"raw-closing cells score ABOVE it -> monotone arcs, not cycles")

    print(f"\nTRACK VERDICT  supervised: {track}   blind: untestable")
    print(f"  {out['verdict']['summary']}")
    print(f"  CAVEAT FOR EVERY OTHER PHASE-2 TRACK: "
          f"{out['verdict']['caveat_for_every_other_phase2_track']}")
    print(f"\nruntime {time.time() - t0:.1f}s -> wrote {a.out}")


if __name__ == "__main__":
    main()
