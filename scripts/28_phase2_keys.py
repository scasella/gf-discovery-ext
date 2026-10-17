"""MUSICAL KEYS -- Phase-2 supervised existence sweep, and the project's one
falsifiable ORDERING discrimination.

Binding document: docs/phase2_prereg_2026-08-21.md, section 3.1 (label set, predicted
structure, discriminating test, verdict rules), section 4.2 (supervised-arm mechanics),
section 4.4 (multiplicity), section 5.1 (pool feasibility), section 6 (null calibration),
section 7 (what is forbidden). Nothing here invents a test the prereg did not register;
where something is added, its provenance is stated in the declaration block below and
written verbatim into the output JSON.

WHY THIS IS THE SHARPEST TEST IN THE PROJECT. Circle-of-fifths order is 7 x chromatic
order (mod 12). The ordering statistic maximises over the 2k = 24 rotations and
reflections of a 12-cycle, i.e. over multipliers +-1 mod 12. Because 7 is not +-1 mod 12,
the fifths permutation lies OUTSIDE the group the statistic quotients by, so the same
twelve points score differently under the two orders and a "correct ordering" cannot be an
artifact of the symmetry maximisation. Measured on a perfect planted ring by the synthetic
battery below: a fifths ring scores rho_fifths = 1.000 against rho_chromatic = 0.385, and
a chromatic ring scores the mirror image. The maximum achievable margin is therefore about
0.615, and that ceiling is reported so a small observed margin is read against it.

DECLARATIONS -- fixed before any activation is read, printed to the log before the first
forward pass, and written into the output JSON under "declarations".

  K1  LABEL SET (prereg 3.1). Twelve pitch classes. Accidentals use the enharmonic FLAT
      spellings, because the sharp spellings do not tokenize as single tokens (prereg
      5.1). The registered mapping is fixed by the prereg: F#->Gb, C#->Db, G#->Ab,
      D#->Eb, A#->Bb. Both orders are scored on the SAME twelve points:
        fifths    : C  G  D  A  E  B  Gb Db Ab Eb Bb F
        chromatic : C  Db D  Eb E  F  Gb G  Ab A  Bb B
      Lowercase note letters are prereg 5.1 polysemy witnesses only and are NEVER scored.
      Every label must be single-token in its leading-space uppercase form or the run
      aborts: they are load-bearing.

  K2  DISCRIMINATING TEST (prereg 3.1, verbatim): at a given (readout, layer), closure
      passes against the uniform null AND against the matched background, AND
      rho_fifths > rho_chromatic (or the reverse) by a margin exceeding the 95th
      percentile of the margin under the 1000-draw shuffled-label control.
      Implementation, fixed now:
        - both codings are scored on the SAME angles with the SAME seed, so the
          angle-rank tie-break jitter is identical and the margin comes from the coding
          difference alone;
        - the null is PAIRED: each of the 1000 shuffles is scored under both codings and
          the margin taken within the draw;
        - the threshold is the 95th percentile of |margin| under that null, which is a
          two-sided alpha = 0.05 test. Because multiplication by 7 is an involution mod 12
          (7 x 7 = 49 = 1), composing a label shuffle with the fifths<->chromatic
          permutation is a bijection that swaps the two rhos, so the null margin
          distribution is exactly symmetric about zero. That symmetry is REPORTED
          (null_mean, fraction positive), not assumed;
        - the comparison is STRICT ("strictly beating"), and the winner is the sign;
        - the margin is computed on the PRIMARY (top-2 PCA) plane only. It is never
          computed on a plane chosen by the widened search of K8.

  K3  READOUTS (prereg 4.2). 26 layers x 3 readouts. Every template is
      PREFIX + " " + WORD + SUFFIX, so the note token sits at a known index and the suffix
      tokens are IDENTICAL across the twelve words within a template.
        token  -- the note token itself. EMBEDDING-CONFOUNDED: at early layers this point
                  cloud is (almost) the note token's own embedding, so a ring here is
                  lexical geometry, not computation. Declared in advance as the confound,
                  not the finding.
        suffix -- PRIMARY COMPUTED READOUT: the first token AFTER the note.
        final  -- the last token of the template. Tertiary, exploratory.

  K4  TEMPLATES (prereg 3.1 keys-specific constraint + 4.2). 16 templates. Every prefix
      establishes the musical-key sense BEFORE the note token, because the suffix readout
      can only inherit disambiguation from the left context. No suffix names a key, a note
      or an accidental; every suffix is >= 3 tokens so `suffix` and `final` are distinct
      positions; no template contains a member of the label set (no bare capital A-G
      anywhere, articles included). Twenty candidates were checked with the tokenizer
      alone (no model, no activations) for manual-vs-natural agreement on all twelve
      labels; all twenty passed, and the FIRST SIXTEEN are used -- a mechanical rule fixed
      before any activation was read. The four verified spares are recorded unused.

  K5  DISCRIMINATOR IS CLOSURE (project tenet, prereg 3 preamble). Ordering alone is never
      evidence: random subsets pass ordering at ~0.16 in this project and an ordinal
      control out-orders real cycles. A cell is called a ring only if closure passes
      against BOTH background tiers of K6.

  K6  NULLS (prereg 4.2).
      (a) uniform random angles, k of them, 20,000 draws;
      (b) 2000 draws of k matched same-arity single-token common NOUNS through the
          identical templates at the identical positions;
      (c) keys-specific class-matched pool (prereg 3.1, which calls it "the background
          that matters for keys"): 2000 draws of k single capital LETTERS from H-Z --
          nineteen letters, none of them a note name, same tokenization class and same
          polysemy profile as the seven naturals. Caveat recorded: C(19,12) = 50,388
          distinct 12-subsets exist, so 2000 draws resample a small universe and adjacent
          draws share members; the pool's diversity, not the draw count, bounds this tier.
      (d) ordering: 20,000 shuffled-label permutations under the same max-over-symmetries
          selection, plus an explicit 1000-shuffle control block, plus the paired
          1000-draw margin null of K2.
      Background draws are NOT scored for ordering: random nouns and random letters have
      no true cyclic order, so closure is the background criterion.
      VERDICT RULE ON BACKGROUNDS, fixed now: closure counts as passing only if the
      uniform null AND the noun background AND the letter background all give p < 0.05.
      This is stricter than scripts/24_colors_existence.py, which used the noun tier
      alone; prereg 3.1 elevates the letter pool for keys, so both are required. The
      noun-only verdict is also emitted, clearly labelled, for comparability with colours.

  K7  BACKGROUND EXCLUSION RULE (prereg 4.2), declared before the pools are used: no
      background word may carry an established sense belonging to the ontology under test.
      For keys that means every word with a musical sense (bridge, keyboard, string, pipe,
      chord, bar, beat, note, key, scale, staff, rest, band, tone, pitch, organ, record,
      sharp, flat, major, minor, measure, ...) and every word naming a letter of the
      alphabet (letter) is removed from the noun pool. Calendar words are excluded so the
      pools cannot inherit this project's other ontologies.

  K8  SECONDARY, WIDENED PLANE SEARCH -- PROVENANCE STATED PLAINLY. This is NOT in prereg
      3.1. It is added here, at script-writing time, BEFORE any activation was read,
      patterned on the D10 block of scripts/24_colors_existence.py, which prereg 7 names
      as the model for a provenance-stated addition. Rule: all C(4,2) = 6 planes spanned
      by pairs of the top 4 PCs; the plane minimising the angular gap ratio is selected.
      Selection is LABEL-FREE. The uniform-angle p-value on a selected plane is invalid
      and is reported marked as such; the valid test is the matched background with the
      IDENTICAL min-over-6-planes selection applied to every draw. Verdicts and the K2
      fifths-vs-chromatic margin are computed on the PRIMARY statistic ONLY; the secondary
      gets its own field so any positive is visibly a product of the widened search.

  K9  SIGN CONVENTIONS (prereg 3 preamble). Both SVD sign conventions are reported. The
      max-over-symmetries rho is reflection-invariant by construction, so it must be
      IDENTICAL under a PC2 sign flip, and so must the margin; both identities are
      ASSERTED, not assumed. The rotation-only rho (no reflection allowed) is reported per
      convention as the quantity a sign flip does change.

  K10 PER-CELL VERDICT VOCABULARY (prereg 3.1 table):
        recovered_fifths        closure passes all three tiers AND the margin separates
                                with fifths ahead
        recovered_chromatic     same, chromatic ahead. Registered by the prereg in advance
                                as EQUALLY INTERPRETABLE, not as a failure: it says the
                                model carries pitch-class geometry organised by semitone
                                rather than by fifth
        closed_ring_unordered   closure passes, the margin does not separate
        ordered_not_closed      an ordering separates, closure does not. Explicitly NOT a
                                circle-of-fifths claim
        clean_negative          neither
        degenerate              the point cloud carries no usable geometry

  K11 ONTOLOGY-LEVEL AGGREGATION, frozen before the run so it cannot be fitted to the
      table:
        1. Only live (non-degenerate) cells count.
        2. The PRIMARY readout is `suffix`. `token` at layers 0-2 is the declared
           embedding confound (K3) and can never carry the headline; `final` is tertiary.
        3. The ontology verdict is the strongest per-cell verdict reached at the SUFFIX
           readout by a cell whose noun-background AND letter-background closure p-values
           both survive Benjamini-Hochberg within their own 78-cell family. An isolated
           uncorrected cell is not a finding (prereg 7).
        4. If no suffix cell closes: `ordered_not_closed` if a suffix cell's ordering p
           survives BH in its family, else `clean_negative`.
        5. Cross-arm vocabulary (prereg 4.3): because the blind arm is registered
           `untestable`, the strongest cross-arm verdict reachable by this script is
           `structure_exists_supervised_only`. Both the 3.1 verdict and the 4.3 verdict
           are emitted.

  K12 BLIND ARM: `untestable`, decided by the prereg BEFORE any placement was examined,
      for two independent structural reasons (prereg 3.1, 5.1, 5.6, and the 6 power
      table): (i) only 7 of 12 pitch classes clear the >= 100 count floor and are nodes in
      the frozen partitions at all, against the artifact's own ceil(2/3 * 12) = 8 label
      convention; (ii) the seven naturals are CONTIGUOUS in fifths order, a 180-degree arc
      whose reference ring gives ref_p_closed = 0.895 -- it could not close whatever the
      geometry -- while the same seven under chromatic order give 0.0001. A naturals-only
      test can only ever confirm chromatic and reject fifths. THEREFORE: this script does
      not open results/token_type_geometry.json, and prereg 7 forbids reporting any
      blind-arm negative for keys or any fifths-vs-chromatic claim from the naturals.

  K13 MULTIPLICITY (prereg 4.4). 3 readouts x 26 layers = 78 cells. About 4 ordering
      passes and about 4 closure passes are expected by chance at alpha = 0.05, so an
      isolated pass is not a finding. Benjamini-Hochberg within this sweep is reported for
      every family alongside raw counts, the expected-by-chance count, and each family's
      p-value floor (an add-one empirical p cannot fall below 1/(n_null+1), so a
      "0 survive BH" line must be read against what was reachable).

  K14 EXPLORATORY BLOCKS, declared now, verdict-free.
      (a) Naturals-only (7 points): CLOSURE STATISTICS ONLY. No ordering rho against
          either coding is computed or reported, because prereg 7 forbids any
          fifths-vs-chromatic claim from the seven naturals in either direction, and
          reporting both rhos would invite exactly that reading. The prereg 6 reference
          numbers are attached to the block.
      (b) Flats-versus-naturals placement: the five flats are corpus-rare (Db 0, Gb 2,
          Bb 4, Eb 6, Ab 102 occurrences in cache/sparse_big, prereg 5.1), so they may sit
          as a lexical blob rather than interleaved. Whether the five flats are CONTIGUOUS
          in the recovered angular order is reported per cell, against the exact
          combinatorial reference 12 / C(12,5) = 0.01515 for five of twelve uniformly
          random positions. No fitted null, no p-value beyond that exact number.
      (c) Which order wins, tallied across cells: COUNTS ONLY. No binomial p-value, because
          the 78 cells are the same activations propagated through layers and a binomial
          null would assume an independence that is false.
      Per-cell per-word plane coordinates and recovered angular order are stored for every
      cell, so "which pitch class sat where" is inspectable without any new statistic.

CAVEAT THAT SURVIVES A POSITIVE (prereg 4.5), recorded now. A closed ring at the suffix
readout shows the information is present one token later; it does not show the model
recomputed it, because attention can copy an embedding forward. The only handle reported
here is the principal angle between the suffix-readout plane and the earliest-layer
token-readout plane: a suffix plane that is a copy of the embedding plane sits at a small
angle to it. It is a handle, not a proof.
"""
import argparse, json, os, sys, time
from math import comb
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from gfdisc.env import DEVICE, D_MODEL
from gfdisc.model import load_model, capture
from gfdisc.fits import (degenerate, pca_2d, fit_circle, circularity, is_ring, angles,
                         cyclic_order_score, _rank_of)
from gfdisc.nulls import benjamini_hochberg

STATUS = "exploratory Phase-2 under docs/phase2_prereg_2026-08-21.md"

# ---------------------------------------------------------------------------
# K1 -- label set. Same twelve points, two orders.
# ---------------------------------------------------------------------------
CHROMATIC = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"]
FIFTHS = ["C", "G", "D", "A", "E", "B", "Gb", "Db", "Ab", "Eb", "Bb", "F"]
ENHARMONIC = {"F#": "Gb", "C#": "Db", "G#": "Ab", "D#": "Eb", "A#": "Bb"}
NATURALS = ["C", "D", "E", "F", "G", "A", "B"]
FLATS = ["Db", "Eb", "Gb", "Ab", "Bb"]

# ---------------------------------------------------------------------------
# K4 -- templates. (prefix, suffix); the note goes between them with one space.
# All twenty were verified with the tokenizer alone (manual ids == natural
# tokenisation, for every one of the twelve labels). The first sixteen are used.
# ---------------------------------------------------------------------------
TEMPLATE_CANDIDATES = [
    ("The sonata is written in the key of", " and was published later"),
    ("She transposed the piece into", " before the rehearsal began"),
    ("The choir sang the hymn in", " throughout the whole service"),
    ("This nocturne was composed in", " during the winter months"),
    ("The orchestra tuned and then played in", " for the entire evening"),
    ("He modulated from the tonic into", " halfway through the movement"),
    ("The score is marked in the key of", " on the opening page"),
    ("The quartet rehearsed the finale in", " until it sounded right"),
    ("The organist improvised a prelude in", " before the sermon started"),
    ("The symphony opens in", " and closes very triumphantly"),
    ("The violinist prefers to play in", " because it suits her"),
    ("The song was recorded in", " at the studio downtown"),
    ("The cadence resolves neatly into", " near the final bar"),
    ("The band tuned down and jammed in", " for about ten minutes"),
    ("The manuscript gives the key signature of", " on every printed page"),
    ("The pianist practised her scales in", " every single morning"),
    # --- verified spares, NOT used (K4 mechanical rule: first sixteen) ---
    ("The concerto was originally written in", " but later heavily revised"),
    ("The hymn book prints this tune in", " for congregational singing"),
    ("The ensemble performed the overture in", " with great precision"),
    ("The composer chose the key of", " for the slow movement"),
]
N_TEMPLATES_USED = 16
TEMPLATES = TEMPLATE_CANDIDATES[:N_TEMPLATES_USED]
TEMPLATE_SPARES = TEMPLATE_CANDIDATES[N_TEMPLATES_USED:]
READOUTS = ("token", "suffix", "final")

# ---------------------------------------------------------------------------
# K6(c) -- keys-specific class-matched background: capital letters H-Z.
# ---------------------------------------------------------------------------
LETTER_POOL = [chr(c) for c in range(ord("H"), ord("Z") + 1)]

# ---------------------------------------------------------------------------
# K7 -- noun pool and the keys exclusion list. The pool is the one
# scripts/24_colors_existence.py used, minus every word with a musical or
# letter-of-the-alphabet sense.
# ---------------------------------------------------------------------------
MUSIC_SENSE_EXCLUSIONS = [
    "bridge", "keyboard", "string", "pipe", "letter", "note", "key", "scale", "chord",
    "bar", "beat", "rest", "staff", "band", "tone", "pitch", "organ", "record", "sharp",
    "flat", "major", "minor", "measure", "drum", "horn", "bell", "trumpet", "violin",
    "piano", "guitar", "harmony", "melody", "rhythm", "tune", "song", "score", "sheet",
    "track", "album", "sound", "volume", "octave", "tempo", "solo", "duet", "choir",
    "opera", "concert", "stage", "conductor", "singer", "voice", "reed", "bow", "case",
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

N_BACKGROUND = 2000
N_PERM_ORDER = 20000
N_SHUFFLE_CONTROL = 1000
N_MARGIN_NULL = 1000
N_GAP_NULL = 20000
N_NATURALS_BG = 500          # K14(a) exploratory block only
N_OPEN_SHAPE_SEEDS = 10      # K15(1): open-shape fixtures are run over seeds, not one draw

# ---------------------------------------------------------------------------
# K15 -- BATTERY AMENDMENTS. Every change below was made AFTER the first run of the
# synthetic battery and BEFORE any activation was read, before the model was ever
# loaded. Nothing in K1-K14 -- label set, readouts, discriminating test, nulls,
# verdict rules, aggregation rule -- was altered. Recorded verbatim in the output.
# ---------------------------------------------------------------------------
BATTERY_AMENDMENTS = [
    "PLANTED-LINE ROW REDEFINED, and the reason is a measured property of the statistic "
    "rather than a tuning problem. scripts/24_colors_existence.py plants a thin 1-D sausage "
    "at k = 9 and asserts it orders at rho >= 0.99 while never closing. At k = 12 that is "
    "false: the middle points of a sausage sit at x ~ 0, where the Kasa fit's centre is far "
    "away and the recovered angle is dominated by the perturbation, so the angular order "
    "scrambles. Measured over 20 seeds at the colours setting (ambient 0.34): rho_fifths "
    "ranged 0.336 to 1.000, mean 0.872. Lowering the ambient noise does not fix it "
    "(ambient 0.15 still gives rho min 0.469 over 20 seeds). Rather than tune the noise "
    "until one seed passed the assertion, the row is now run over 10 seeds, its rho "
    "distribution is REPORTED as a property of the statistic at k = 12, and the GATE on it "
    "is narrowed to what it actually tests: closure must fail on every seed at BOTH the "
    "primary and the widened-search tier. The order-without-closure dissociation is carried "
    "by the 200-degree arc row, which does it robustly (rho 1.000, p_order 0.0000, closure "
    "p 0.98) and is also now run over 10 seeds.",

    "DEDICATED PER-FIXTURE RNGs. Each fixture now draws from its own seeded generator, so a "
    "fixture's realisation no longer depends on how many draws the fixtures before it "
    "consumed. This is what made the first line-row failure look like a noise-level problem "
    "when it was a seed-lottery problem.",

    "ANGULAR-JITTER POWER CURVE ADDED (K5b). The isotropic ambient-noise ladder that "
    "prereg 4.2 registers turns out to measure almost nothing about angular robustness in "
    "1152 dimensions: the in-plane component of an isotropic vector of norm 0.6 is only "
    "0.6 * sqrt(2/1152) = 0.025, so the planted ring scored the identical maximum margin "
    "0.6154 at every rung of the ladder (0.15, 0.30, 0.60). The ladder is kept as "
    "registered and reported, and a curve that does bite is added next to it: Gaussian "
    "jitter applied to the ring ANGLES themselves at 0 to 60 degrees, 200 trials per rung, "
    "reporting how often the fifths-vs-chromatic margin separates and in which direction. "
    "It operates on angles directly, so it isolates the ordering statistic from the "
    "plane-recovery step and is an UPPER bound on the power available on real point clouds. "
    "Without it a null discrimination would be uninterpretable.",

    "MARGIN-TEST POWER STATED IN NUMBERS. At k = 12 the two-sided threshold (95th percentile "
    "of |margin| under the paired 1000-draw shuffled-label null) is about 0.392, against a "
    "ceiling of 0.6154 set by a perfect ring. The discrimination therefore only ever fires "
    "when one coding beats the other by roughly two thirds of the maximum possible margin. "
    "That is a property of scoring two orders that share the same twelve points, it is "
    "reported next to every margin, and it is why a non-separating result is reported as "
    "'discrimination not made' rather than as evidence for either order.",
]

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
# Ordering, vectorised, with the flip group exposed (K9).
# ---------------------------------------------------------------------------
def _rank_from_angles(ang, seed):
    """Rank of each point by angle, with the same random tie-break fits.py uses."""
    ang = np.asarray(ang, dtype=float)
    n = len(ang)
    jitter = np.random.default_rng(seed).permutation(n)
    order = np.lexsort((jitter, ang))
    rank_of = np.empty(n, dtype=np.float64)
    rank_of[order] = np.arange(n)
    return rank_of


def _sym_basis(label_index, K, flips=(1, -1)):
    lab = np.asarray(label_index, dtype=int)
    syms = [(int(f), int(s)) for f in flips for s in range(K)]
    return syms, np.vstack([_rank_of((f * lab + s) % K) for f, s in syms])


def order_stats(ang, label_index, K, flips=(1, -1), n_perm=N_PERM_ORDER, seed=0, block=5000):
    """Max Spearman over the allowed symmetries, calibrated by shuffled labels.

    Same construction as fits.cyclic_order_score (stable-argsort tie-break replaced by a
    random jitter, add-one p-value), vectorised over the permutation null and with `flips`
    exposed so the rotation-only (handedness-sensitive) statistic can be read off
    separately. With flips=(1,-1) it must reproduce cyclic_order_score exactly; that is
    asserted in `check_against_reference`.
    """
    rank_of = _rank_from_angles(ang, seed)
    n = len(rank_of)
    syms, base = _sym_basis(label_index, K, flips)
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


def margin_test(ang, codes_a, codes_b, K, name_a="fifths", name_b="chromatic",
                n_draws=N_MARGIN_NULL, seed=0):
    """K2. Paired fifths-vs-chromatic margin with its own shuffled-label null.

    Both codings are scored on the same angles with the same tie-break seed, and each null
    draw is scored under BOTH codings before the difference is taken, so the margin is a
    property of the coding difference and of nothing else.
    """
    rank_of = _rank_from_angles(ang, seed)
    n = len(rank_of)
    _, base_a = _sym_basis(codes_a, K)
    _, base_b = _sym_basis(codes_b, K)
    denom = n * (n * n - 1)

    def rhos(base, sig):
        d = rank_of - base[:, sig]
        return 1.0 - 6.0 * (d * d).sum(-1) / denom

    ident = np.arange(n)
    rho_a = float(rhos(base_a, ident).max())
    rho_b = float(rhos(base_b, ident).max())
    obs = rho_a - rho_b

    rng = np.random.default_rng(seed + 424242)
    sig = np.argsort(rng.random((n_draws, n)), axis=1)
    na = rhos(base_a, sig).max(0)
    nb = rhos(base_b, sig).max(0)
    null = na - nb
    thr = float(np.percentile(np.abs(null), 95))
    winner = name_a if obs > 0 else (name_b if obs < 0 else None)
    return dict(
        rho_fifths=rho_a, rho_chromatic=rho_b, margin=float(obs),
        abs_margin=float(abs(obs)),
        n_draws=int(n_draws),
        null_abs_p95=thr,
        null_signed_p95=float(np.percentile(null, 95)),
        null_signed_p05=float(np.percentile(null, 5)),
        null_mean=float(null.mean()),
        null_fraction_positive=float((null > 0).mean()),
        null_symmetry_note="x7 is an involution mod 12, so the null margin distribution is "
                           "exactly symmetric about 0; null_mean and null_fraction_positive "
                           "report that rather than assuming it",
        p_two_sided=float((1 + int((np.abs(null) >= abs(obs)).sum())) / (1 + n_draws)),
        p_floor=float(1.0 / (n_draws + 1)),
        separates=bool(abs(obs) > thr),
        winner=(winner if abs(obs) > thr else None),
        ceiling_note="a perfect ring in one order scores 1.000 against 0.385 in the other, "
                     "so the maximum achievable |margin| is about 0.615")


def check_against_reference(seed=0):
    """order_stats(flips=(1,-1)) must equal fits.cyclic_order_score on the same input."""
    rng = np.random.default_rng(seed)
    out = []
    for t in range(5):
        k = 12
        ang = rng.uniform(-np.pi, np.pi, size=k)
        lab = np.arange(k)
        a = cyclic_order_score(ang, lab, k, n_perm=2000, seed=t)
        b = order_stats(ang, lab, k, n_perm=2000, seed=t)
        out.append(dict(trial=t, rho_ref=a["rho"], rho_new=b["rho"],
                        sym_ref=list(a["symmetry"]), sym_new=b["symmetry"],
                        rho_match=bool(abs(a["rho"] - b["rho"]) < 1e-12)))
    # A second reference: the margin machinery on a perfect ring, where the answer is
    # known in closed form. A fifths ring must score rho_fifths exactly 1 and
    # rho_chromatic exactly 5/13 = 0.3846..., and the chromatic ring the mirror.
    p = np.arange(12)
    fifth_pos = (7 * p) % 12
    th_f = 2 * np.pi * fifth_pos / 12
    th_c = 2 * np.pi * p / 12
    mf = margin_test(th_f, fifth_pos, p, 12, n_draws=200, seed=0)
    mc = margin_test(th_c, fifth_pos, p, 12, n_draws=200, seed=0)
    closed_form = dict(
        fifths_ring=dict(rho_fifths=mf["rho_fifths"], rho_chromatic=mf["rho_chromatic"],
                         margin=mf["margin"], winner_if_separating=("fifths" if mf["margin"] > 0
                                                                    else "chromatic")),
        chromatic_ring=dict(rho_fifths=mc["rho_fifths"], rho_chromatic=mc["rho_chromatic"],
                            margin=mc["margin"],
                            winner_if_separating=("fifths" if mc["margin"] > 0 else "chromatic")),
        mirror_symmetric=bool(abs(mf["margin"] + mc["margin"]) < 1e-12),
        max_achievable_abs_margin=float(abs(mf["margin"])))
    passed = bool(all(x["rho_match"] for x in out)
                  and abs(mf["rho_fifths"] - 1.0) < 1e-12
                  and abs(mc["rho_chromatic"] - 1.0) < 1e-12
                  and mf["margin"] > 0 and mc["margin"] < 0
                  and closed_form["mirror_symmetric"])
    return dict(trials=out, closed_form_ring_check=closed_form, passed=passed)


# ---------------------------------------------------------------------------
# The one analysis routine. Keys, backgrounds and synthetics all go through it.
# ---------------------------------------------------------------------------
def analyse(P, codings=None, K=None, seed=0, do_order=True, both_conventions=True,
            do_margin=True):
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
        if do_order and not deg and codings is not None:
            ang = angles(z, cx, cy)
            KK = int(K if K is not None else k)
            d["order"] = {}
            for cname, codes in codings.items():
                d["order"][cname] = order_stats(ang, codes, KK, flips=(1, -1),
                                                n_perm=N_PERM_ORDER, seed=seed)
                d["order"][cname]["rotation_only"] = order_stats(
                    ang, codes, KK, flips=(1,), n_perm=N_SHUFFLE_CONTROL, seed=seed)
                d["order"][cname]["shuffled_label_control"] = order_stats(
                    ang, codes, KK, flips=(1, -1), n_perm=N_SHUFFLE_CONTROL, seed=seed + 777)
            if do_margin and "fifths" in codings and "chromatic" in codings:
                m = margin_test(ang, codings["fifths"], codings["chromatic"], KK,
                                n_draws=N_MARGIN_NULL, seed=seed)
                # the margin must be built from the same rhos the order block reports
                assert abs(m["rho_fifths"] - d["order"]["fifths"]["rho"]) < 1e-12
                assert abs(m["rho_chromatic"] - d["order"]["chromatic"]["rho"]) < 1e-12
                d["margin"] = m
            d["angles_rad"] = [float(x) for x in ang]
            d["xy"] = [[float(u), float(v)] for u, v in z]
        convs[name] = d
    out["conventions"] = convs
    return out, Vt[:2], xy


# ---------------------------------------------------------------------------
# K8 -- widened, label-free plane search. One SVD, six circle fits.
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


def secondary_analyse(P, codings=None, K=None, seed=0, n_pc=N_PC_SECONDARY, do_order=True):
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
    if do_order and codings is not None and not degenerate(P):
        ang = angles(best["xy"], best["cx"], best["cy"])
        out["order"] = {cname: order_stats(ang, codes, int(K if K is not None else k),
                                           flips=(1, -1), n_perm=N_PERM_ORDER, seed=seed)
                        for cname, codes in codings.items()}
        out["order_note"] = ("plane selection is label-free, so this shuffled-label null is "
                             "exact for the selected plane")
        out["margin_note"] = ("K2/K8: the fifths-vs-chromatic margin is NOT computed on a "
                              "selected plane and no discrimination is made here")
    return out, best["basis"]


# ---------------------------------------------------------------------------
# Synthetic battery -- run through the same `analyse` the keys use.
# ---------------------------------------------------------------------------
def margin_power_curve(k=12, jitters_deg=(0, 5, 10, 15, 20, 30, 45, 60), n_trials=200, seed=0):
    """K5b (added under K15). How much angular error does the K2 discrimination survive?

    Operates on ANGLES directly rather than on point clouds, so it isolates the ordering
    statistic from the plane-recovery step. It is therefore an UPPER bound on the power
    available on real activations: any error in recovering the plane can only lower it.
    """
    p = np.arange(k)
    fifth_pos = (7 * p) % k if k == 12 else p
    rows = {}
    for j in jitters_deg:
        rng = np.random.default_rng(90000 + int(j * 100))
        sep_right = 0
        sep_wrong = 0
        margins = np.empty(n_trials)
        thrs = np.empty(n_trials)
        for t in range(n_trials):
            th = 2 * np.pi * fifth_pos / k + np.radians(j) * rng.normal(size=k)
            m = margin_test(th, fifth_pos, p, k, n_draws=N_MARGIN_NULL, seed=seed + t)
            margins[t] = m["margin"]
            thrs[t] = m["null_abs_p95"]
            if m["separates"]:
                sep_right += int(m["winner"] == "fifths")
                sep_wrong += int(m["winner"] == "chromatic")
        rows[f"jitter_{j}deg"] = dict(
            jitter_deg=float(j), n_trials=int(n_trials),
            margin_mean=float(margins.mean()), margin_p05=float(np.percentile(margins, 5)),
            threshold_mean=float(thrs.mean()),
            fraction_separating_correct=float(sep_right / n_trials),
            fraction_separating_wrong_direction=float(sep_wrong / n_trials))
    return dict(
        k=int(k), n_trials=int(n_trials), jitters_deg=list(jitters_deg), rows=rows,
        note="planted fifths ring with Gaussian angular jitter; fraction_separating_correct "
             "is the power of the K2 discrimination at that angular error. Computed on "
             "angles, so it is an upper bound on the power available on point clouds.")


def closure_power_curve(k=12, jitters_deg=(0, 5, 10, 15, 20, 30, 45, 60), n_trials=2000,
                        d=D_MODEL, ambient=0.60, seed=0, bg_thresholds=None):
    """K16 -- POST-VERIFICATION ADDITION, 2026-08-21. Provenance stated (prereg 7).

    WHY IT WAS MISSING AND WHY THAT MATTERED. The registered battery had no closure power
    curve at all. Prereg 4.2's isotropic ambient-noise ladder is in-plane-noiseless by
    construction and the artifact says so itself in K15(3): an isotropic vector of norm 0.60
    in 1152-d has in-plane component 0.60 * sqrt(2/1152) = 0.025, and the planted ring
    accordingly scored gap_ratio 1.0686 -- a near-perfect ring -- at the LOUDEST rung of the
    ladder. K15(3) then added an angular-jitter curve, but that curve reports MARGIN power
    only. So the keys result, whose headline is carried by CLOSURE, was reported as a
    negative with a demonstrated power envelope for the wrong statistic. That is the gap this
    function closes.

    TWO VARIANTS.
      angles_only  jitter applied to the ideal ring angles; the closure statistic computed
                   from those angles directly. Isolates closure from plane recovery, so it is
                   an UPPER bound on available power -- the same framing K15(3) gave its
                   margin twin.
      point_cloud  the same jittered ring planted in a random 2-plane of 1152-d with the
                   registered ambient noise (per-point NORM, the `amb` convention of
                   synthetics), then pushed through the identical
                   pca_2d -> fit_circle -> circularity -> p_closed path every real cell takes.
                   This is the honest number.

    TWO POWER COLUMNS PER RUNG.
      P(p_closed_uniform < 0.05)  the registered uniform tier.
      P(gap_ratio < threshold)    the tier that actually decides, referenced to the OBSERVED
                                  matched-background 5th percentile averaged over the 78
                                  cells. That threshold is data-derived, so this curve is
                                  computed at report time, after the sweep, and says so.

    Direction of the statistic: gap_ratio ~ 1 for a closed cycle and grows with the largest
    angular gap, so LOWER beats the background. A cell passes the background tier when its
    gap_ratio falls below the background's 5th percentile.
    """
    p = np.arange(k)
    fifth_pos = (7 * p) % k if k == 12 else p
    ideal = 2 * np.pi * fifth_pos / k
    bg_thresholds = dict(bg_thresholds or {})
    rows = {}
    for j in jitters_deg:
        rng = np.random.default_rng(140000 + int(j * 100))
        entry = dict(jitter_deg=float(j))
        for tag in ("angles_only", "point_cloud"):
            gr = np.empty(n_trials)
            pu = np.empty(n_trials)
            for t in range(n_trials):
                th = ideal + np.radians(j) * rng.normal(size=k)
                if tag == "angles_only":
                    a = np.sort(np.mod(th, 2 * np.pi))
                    gaps = np.diff(np.concatenate([a, [a[0] + 2 * np.pi]]))
                    mg = float(gaps.max())
                    gr[t] = mg / (2 * np.pi / k)
                    pu[t] = p_closed(k, mg)
                else:
                    u = rng.normal(size=d)
                    u /= np.linalg.norm(u)
                    v = rng.normal(size=d)
                    v -= (v @ u) * u
                    v /= np.linalg.norm(v)
                    P = (np.outer(np.cos(th), u) + np.outer(np.sin(th), v)
                         + (ambient / np.sqrt(d)) * rng.normal(size=(k, d)))
                    xy, _, _ = pca_2d(P)
                    cx, cy, r, _ = fit_circle(xy)
                    c = circularity(xy, cx, cy, r)
                    gr[t] = c["gap_ratio"]
                    pu[t] = p_closed(k, np.radians(c["max_gap_deg"]))
            e = dict(n_trials=int(n_trials),
                     gap_ratio_mean=float(gr.mean()),
                     gap_ratio_median=float(np.median(gr)),
                     gap_ratio_p95=float(np.percentile(gr, 95)),
                     p_uniform_median=float(np.median(pu)),
                     power_uniform_at_05=float((pu < 0.05).mean()))
            e["power_vs_background"] = {
                nm: dict(threshold=float(thr), power=float((gr < thr).mean()))
                for nm, thr in bg_thresholds.items()}
            entry[tag] = e
        rows[f"jitter_{j}deg"] = entry
    return dict(
        k=int(k), d=int(d), n_trials=int(n_trials), jitters_deg=list(jitters_deg),
        ambient_noise_per_point_norm=float(ambient),
        ambient_note="the LOUDEST rung of the registered prereg-4.2 ladder, chosen so the "
                     "point_cloud variant runs at the most demanding registered ambient "
                     "setting rather than the quietest",
        background_thresholds=bg_thresholds,
        rows=rows,
        provenance="post-verification addition, written 2026-08-21 AFTER the 05:43:05 run of "
                   "this script was verified and its power envelope was found to be stated "
                   "for the margin statistic only. Prereg 7 permits a provenance-stated "
                   "post-hoc addition; this one adds no search family and can only WEAKEN the "
                   "negative's reach, never strengthen it, because its output is an upper "
                   "bound on what the negative excludes.",
        reading="the largest gap_ratio at which the closure tier still has usable power is "
                "the reach of the negative. Any ring whose angular fidelity is worse than "
                "that is NOT excluded by this sweep.",
        independent_recomputation_note=(
            "an independent implementation of this curve (200 trials/rung, angles-only, "
            "planted 12-ring) reported P(p_uniform<0.05) = 1.000 / 1.000 / 0.855 / 0.430 / "
            "0.225 / 0.100 at 0/5/10/15/20/30 deg and, against the matched noun background, "
            "1.000 / 1.000 / 1.000 / 0.835 / 0.655 / 0.330. This implementation lands "
            "materially higher at 15-30 deg (see rows). The difference is larger than the "
            "Monte-Carlo error at 2000 trials and is NOT reconciled here -- most likely a "
            "difference in how the jitter is drawn or wrapped. It is recorded rather than "
            "smoothed over. The two agree on what the verdict actually uses: the 80%-power "
            "boundary sits at 10 deg against the uniform null and 15 deg against the matched "
            "noun background under both implementations, and the higher-power implementation "
            "is the one that claims MORE exclusion, so quoting it is the less conservative "
            "choice and the reach below should be read as an upper bound on what is excluded.")
        )


def synthetics(d=D_MODEL, k=12, noise_ladder=(0.15, 0.30, 0.60), n_bg=N_BACKGROUND, seed=0):
    """Planted-shape battery. Gates BOTH closure tiers and the K2 margin machinery.

    Closure alone is not enough here: the deliverable is a discrimination, so the battery
    has to show that the margin points the right way for each planted order, that it does
    NOT fire on a closed ring with scrambled labels or on noise, and -- K15(3) -- how much
    angular error it survives, because that is what a null result has to be read against.

    Every fixture draws from its own seeded generator (K15(2)).
    """
    rng = np.random.default_rng(seed)
    u, v = rng.normal(size=d), rng.normal(size=d)
    u /= np.linalg.norm(u)
    v -= v @ u * u
    v /= np.linalg.norm(v)

    p = np.arange(k)
    fifth_pos = (7 * p) % 12 if k == 12 else p
    codes = dict(fifths=fifth_pos, chromatic=p)
    th_fifths = 2 * np.pi * fifth_pos / k      # points laid out in FIFTHS order
    th_chrom = 2 * np.pi * p / k               # points laid out in CHROMATIC order

    def amb(scale, r=None):
        """Isotropic noise of per-point NORM `scale`, not per-coordinate scale."""
        return (scale / np.sqrt(d)) * (r if r is not None else rng).normal(size=(k, d))

    bg_rng = np.random.default_rng(seed + 12345)
    bg = np.empty(n_bg)
    bg_p_unif = np.empty(n_bg)
    bg_best = np.empty(n_bg)
    for i in range(n_bg):
        g, pu, gb, _ = fast_closure(bg_rng.normal(size=(k, d)))
        bg[i], bg_p_unif[i], bg_best[i] = g, pu, gb

    def row(P, seed_, codings=codes):
        r, _, _ = analyse(P, codings, k, seed=seed_, both_conventions=False)
        a = r["conventions"]["A"]
        s, _ = secondary_analyse(P, codings, k, seed=seed_, do_order=True)
        out = dict(gap_ratio=a["gap_ratio"], radius_ratio=a["radius_ratio"],
                   p_closed=a["p_closed_uniform"],
                   p_vs_background=float((1 + int((bg <= a["gap_ratio"]).sum())) / (1 + n_bg)),
                   top2_var=r["top2_var"],
                   secondary=dict(selected_plane=s["selected_plane"],
                                  gap_ratio=s["gap_ratio"],
                                  p_vs_background=float((1 + int((bg_best <= s["gap_ratio"]).sum()))
                                                        / (1 + n_bg))))
        if "order" in a:
            out["rho_fifths"] = a["order"]["fifths"]["rho"]
            out["p_order_fifths"] = a["order"]["fifths"]["p"]
            out["rho_chromatic"] = a["order"]["chromatic"]["rho"]
            out["p_order_chromatic"] = a["order"]["chromatic"]["p"]
            out["margin"] = a["margin"]["margin"]
            out["margin_threshold"] = a["margin"]["null_abs_p95"]
            out["margin_separates"] = a["margin"]["separates"]
            out["margin_winner"] = a["margin"]["winner"]
        return out

    out = dict(k=k, d=d, noise_ladder=list(noise_ladder))

    # --- SNR ladder: fifths ring and chromatic ring at each noise level ---
    out["planted_fifths_ring"] = {}
    out["planted_chromatic_ring"] = {}
    for i, nz in enumerate(noise_ladder):
        ring_f = np.outer(np.cos(th_fifths), u) + np.outer(np.sin(th_fifths), v) + amb(nz)
        out["planted_fifths_ring"][f"noise_{nz}"] = row(ring_f, 100 + i)
        ring_c = np.outer(np.cos(th_chrom), u) + np.outer(np.sin(th_chrom), v) + amb(nz)
        out["planted_chromatic_ring"][f"noise_{nz}"] = row(ring_c, 200 + i)

    base_noise = noise_ladder[0]
    r_ring = np.random.default_rng(seed + 501)
    ring = (np.outer(np.cos(th_fifths), u) + np.outer(np.sin(th_fifths), v)
            + amb(base_noise, r_ring))

    # K15(5): every stochastic control row is run over seeds and gated on a RATE, never on
    # a single draw. The scrambled-label row is the reason: a random relabelling of twelve
    # points has, by construction, a 5% chance of exceeding the two-sided margin threshold,
    # so a one-draw "the margin must not separate" gate fails one run in twenty for exactly
    # the reason the test is calibrated to allow. The first run of this battery hit it.
    #
    # `open_codes` places an open shape's points in FIFTHS order, so point i is the i-th
    # pitch class of the fifths cycle and its chromatic index is (7 i) mod 12.
    open_codes = dict(fifths=np.arange(k), chromatic=(7 * np.arange(k)) % k)

    def multiseed(make, tag, n_seeds=N_OPEN_SHAPE_SEEDS):
        rs = []
        for s in range(n_seeds):
            r = np.random.default_rng(seed + 600 + 37 * s)
            P, cds = make(r)
            rs.append(row(P, 300 + s, codings=cds))
        agg = dict(
            tag=tag, n_seeds=int(n_seeds), draws=rs,
            rho_fifths_min=float(min(x["rho_fifths"] for x in rs)),
            rho_fifths_mean=float(np.mean([x["rho_fifths"] for x in rs])),
            rho_fifths_max=float(max(x["rho_fifths"] for x in rs)),
            gap_ratio_min=float(min(x["gap_ratio"] for x in rs)),
            gap_ratio_max=float(max(x["gap_ratio"] for x in rs)),
            top2_var_mean=float(np.mean([x["top2_var"] for x in rs])),
            n_order_fifths_below_05=int(sum(x["p_order_fifths"] <= 0.05 for x in rs)),
            n_order_chromatic_below_05=int(sum(x["p_order_chromatic"] <= 0.05 for x in rs)),
            margin_mean=float(np.mean([x["margin"] for x in rs])),
            n_closed_uniform=int(sum(x["p_closed"] <= 0.05 for x in rs)),
            n_closed_primary_background=int(sum(x["p_vs_background"] <= 0.05 for x in rs)),
            n_closed_secondary_background=int(sum(x["secondary"]["p_vs_background"] <= 0.05
                                                  for x in rs)),
            n_margin_separates=int(sum(bool(x["margin_separates"]) for x in rs)))
        return agg

    ta = np.linspace(0, np.radians(200), k)

    def make_arc(r):
        return (np.outer(np.cos(ta), u) + np.outer(np.sin(ta), v) + amb(base_noise, r),
                open_codes)

    def make_line(r):
        return ((np.outer(np.linspace(-1, 1, k), u)
                 + np.outer(0.05 * r.normal(size=k), v) + amb(0.34, r)), open_codes)

    def make_scrambled(r):
        """The SAME planted ring, relabelled. Closure must pass on every seed; ordering and
        the margin must fire only at their nominal rates -- that is what makes the tiers
        independent rather than two readings of one number."""
        q = r.permutation(k)
        return ring, dict(fifths=fifth_pos[q], chromatic=p[q])

    def make_noise(r):
        return r.normal(size=(k, d)), codes

    out["planted_ring_scrambled_labels"] = multiseed(make_scrambled, "ring_scrambled_labels")
    out["planted_ring_scrambled_note"] = (
        "closed ring, labels permuted. n_closed_primary_background must be every seed; "
        "n_order_fifths_below_05 and n_margin_separates must sit at their nominal 5% rates, "
        "not at zero -- a control that never fires would mean the null was miscalibrated.")
    out["planted_arc_200deg"] = multiseed(make_arc, "arc_200deg")
    out["planted_arc_note"] = ("the order-without-closure dissociation row: an open arc in "
                               "fifths order must order on every seed and close on none")
    out["planted_line"] = multiseed(make_line, "line")
    out["planted_line_note"] = (
        "thin sausage in the fitted plane. K15(1): at k = 12 this does NOT reliably order "
        "-- the middle points sit where the Kasa centre is far away and the recovered angle "
        "is perturbation-dominated -- so rho is REPORTED, not asserted, and the gate on this "
        "row is closure-must-fail at both tiers on every seed. The k = 9 colours battery "
        "asserted rho >= 0.99 here; that does not transfer to k = 12.")

    # K8 validation: a fifths ring buried under a dominant linear ramp. The primary plane
    # must miss it; the widened search must find it. Without this row a negative secondary
    # would say nothing about the secondary.
    w = rng.normal(size=d)
    w -= (w @ u) * u + (w @ v) * v
    w /= np.linalg.norm(w)
    ramp = np.linspace(-1, 1, k)
    B = np.column_stack([np.ones(k), np.cos(th_fifths), np.sin(th_fifths)])
    ramp = ramp - B @ np.linalg.lstsq(B, ramp, rcond=None)[0]
    ramp = ramp / max(ramp.std(), 1e-12)
    def make_buried(r):
        return (2.0 * np.outer(ramp, w)
                + np.outer(np.cos(th_fifths), u) + np.outer(np.sin(th_fifths), v)
                + amb(base_noise, r)), codes

    out["buried_ring"] = multiseed(make_buried, "buried_ring")
    out["buried_ring_note"] = (
        "unit fifths ring under a 2x third component whose per-point coefficients are "
        "orthogonalised against the ring's; the primary top-2 plane must miss it, K8 must "
        "find it. Consequence for K2, recorded here rather than discovered later: when the "
        "ring is buried, the PRIMARY plane is the wrong plane, so the margin computed there "
        "is meaningless -- and K2/K8 forbid computing the margin on a selected plane. A "
        "buried ring can therefore be detected by K8 and still leave the "
        "fifths-vs-chromatic discrimination unmade.")

    out["pure_noise"] = multiseed(make_noise, "pure_noise", n_seeds=20)
    out["pure_noise_note"] = ("20 isotropic clouds through the identical pipeline; every "
                              "tier's false-positive count is reported against 20 * 0.05 = "
                              "1 expected")
    out["uniform_null_calibration"] = dict(
        n_draws=int(n_bg), pool="isotropic Gaussian, same k and d",
        gap_ratio_mean=float(bg.mean()), gap_ratio_p05=float(np.percentile(bg, 5)),
        gap_ratio_min=float(bg.min()),
        uniform_null_false_acceptance_at_p05=float((bg_p_unif <= 0.05).mean()),
        note="a false-acceptance rate above 0.05 here means the uniform-angle test is "
             "anti-conservative on 1152-dimensional clouds; the matched-word backgrounds "
             "are the tiers that decide")

    F = out["planted_fifths_ring"]
    C = out["planted_chromatic_ring"]
    RS, AR, LN, BR = (out["planted_ring_scrambled_labels"], out["planted_arc_200deg"],
                      out["planted_line"], out["buried_ring"])
    f0 = F[f"noise_{noise_ladder[0]}"]
    c0 = C[f"noise_{noise_ladder[0]}"]
    out["margin_power_by_noise"] = {
        f"noise_{nz}": dict(fifths_ring_margin=F[f"noise_{nz}"]["margin"],
                            fifths_ring_separates=F[f"noise_{nz}"]["margin_separates"],
                            fifths_ring_winner=F[f"noise_{nz}"]["margin_winner"],
                            chromatic_ring_margin=C[f"noise_{nz}"]["margin"],
                            chromatic_ring_separates=C[f"noise_{nz}"]["margin_separates"],
                            chromatic_ring_winner=C[f"noise_{nz}"]["margin_winner"])
        for nz in noise_ladder}

    out["margin_power_by_angular_jitter"] = margin_power_curve(k=k)
    jc = out["margin_power_by_angular_jitter"]["rows"]
    out["margin_threshold_reference"] = dict(
        k=int(k), threshold_mean=jc["jitter_0deg"]["threshold_mean"],
        ceiling=float(abs(F[f"noise_{noise_ladder[0]}"]["margin"])),
        note="K15(4): the two-sided |margin| threshold at this k against the ceiling a "
             "perfect ring sets. The discrimination only fires when one coding beats the "
             "other by that fraction of the maximum, which is why a non-separating result "
             "is reported as 'discrimination not made', never as support for either order.")

    ns = N_OPEN_SHAPE_SEEDS
    NZ = out["pure_noise"]
    gate = dict(
        # closure tiers on the cleanest planted rings
        fifths_ring_closes=bool(f0["p_closed"] <= 0.05 and f0["p_vs_background"] <= 0.05),
        chromatic_ring_closes=bool(c0["p_closed"] <= 0.05 and c0["p_vs_background"] <= 0.05),
        # the discrimination points the right way for each planted order
        fifths_ring_margin_correct=bool(f0["margin_separates"]
                                        and f0["margin_winner"] == "fifths"),
        chromatic_ring_margin_correct=bool(c0["margin_separates"]
                                           and c0["margin_winner"] == "chromatic"),
        # closed ring, relabelled: closure survives every seed, ordering and margin only
        # fire at their nominal 5% rate (<= 2 of 10 is the rate gate, not a zero gate)
        scrambled_still_closes=bool(RS["n_closed_primary_background"] == ns),
        scrambled_order_at_nominal_rate=bool(RS["n_order_fifths_below_05"] <= 2
                                             and RS["n_order_chromatic_below_05"] <= 2),
        scrambled_margin_at_nominal_rate=bool(RS["n_margin_separates"] <= 2),
        # the dissociation, carried by the arc: orders on every seed, closes on none
        arc_orders_every_seed=bool(AR["n_order_fifths_below_05"] == ns),
        arc_never_closes=bool(AR["n_closed_uniform"] == 0
                              and AR["n_closed_primary_background"] == 0
                              and AR["n_closed_secondary_background"] == 0),
        # the line: K15(1) gates closure only -- its ordering is reported, not asserted
        line_never_closes=bool(LN["n_closed_uniform"] == 0
                               and LN["n_closed_primary_background"] == 0
                               and LN["n_closed_secondary_background"] == 0),
        # noise must not manufacture closure or a discrimination (20 draws, 1 expected)
        noise_closure_at_nominal_rate=bool(NZ["n_closed_primary_background"] <= 3),
        noise_margin_at_nominal_rate=bool(NZ["n_margin_separates"] <= 3),
        noise_secondary_at_nominal_rate=bool(NZ["n_closed_secondary_background"] <= 3),
        # K8 must have the sensitivity a negative would need, without inventing rings
        buried_ring_missed_by_primary=bool(BR["n_closed_primary_background"] <= 2),
        buried_ring_found_by_widened_search=bool(BR["n_closed_secondary_background"] >= 8),
        # the discrimination must have real power at zero angular error, and must never
        # fire backwards; both are what a null result will be read against
        margin_power_at_zero_jitter=bool(jc["jitter_0deg"]["fraction_separating_correct"] >= 0.99),
        margin_never_backwards=bool(
            all(jc[f"jitter_{j}deg"]["fraction_separating_wrong_direction"] == 0.0
                for j in out["margin_power_by_angular_jitter"]["jitters_deg"])),
    )
    out["gate"] = gate
    out["passed"] = bool(all(gate.values()))
    return out


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
            ni = 1 + len(pre)                 # note token index (BOS at 0)
            pos = dict(token=ni, suffix=ni + 1, final=ni + len(suf))
            base = [bos] + pre + [0] + suf
            for lo in range(0, n, chunk):
                hi = min(lo + chunk, n)
                ids = []
                for w in word_ids[lo:hi]:
                    row = list(base)
                    row[ni] = int(w)
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


def flats_contiguity(order_words):
    """K14(b). Are the five flats contiguous in the recovered angular order (cyclically)?"""
    n = len(order_words)
    flags = np.array([w in set(FLATS) for w in order_words])
    m = int(flags.sum())
    if m == 0 or m == n:
        return dict(n_flats=m, contiguous=True, longest_run=m)
    doubled = np.concatenate([flags, flags])
    best = run = 0
    for x in doubled:
        run = run + 1 if x else 0
        best = max(best, run)
    best = min(best, m)
    return dict(n_flats=m, contiguous=bool(best == m), longest_run=int(best))


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
# K17 -- POST-VERIFICATION ADDITIONS, 2026-08-21. Two artifacts that a reader of THIS file
# must not be able to miss, because without them the verdict block reports only the half
# that flatters the result.
#
#   (a) results/phase2_keys_blockperm.json -- the within-block permutation attack on the
#       ordering signal. The 05:43:05 artifact contained ZERO references to it: grepping
#       'blockperm', 'within_block' and '29_keys' over its 1,498,691 bytes returned 0 hits.
#       The analyst's own stated rule is that "neither half may be reported alone --
#       ordered_not_closed on its own launders the artifact", and the artifact on disk
#       carried exactly the laundering half. Embedded here so it travels with the verdict.
#
#   (b) results/colors_existence.json -- the verified template this script copies (prereg
#       4.2). It is ALSO 0/78 on closure. That means this templated supervised pipeline has
#       never been shown to recover a ring that genuinely exists in the model, so the keys
#       negative may not be read as "the pipeline that finds rings found none here". Its
#       end-to-end sensitivity is unestablished, and the verdict block has to say so.
#
# Both readers are strictly non-fatal: a missing file is recorded as missing, never
# silently skipped, and never fabricated.
# ---------------------------------------------------------------------------
def read_blockperm(path="results/phase2_keys_blockperm.json"):
    p = Path(path)
    if not p.exists():
        return dict(path=str(path), present=False,
                    warning="THE ORDERING ATTACK IS NOT ON DISK. The supervised "
                            "ordered_not_closed verdict below must NOT be reported without "
                            "it; run scripts/29_keys_blockperm.py.")
    b = json.loads(p.read_text())
    s = b.get("summary", {})
    oc = b.get("ordering_collapse", {})
    return dict(
        path=str(path), present=True, script=b.get("script"), generated=b.get("generated"),
        confound=b.get("confound", {}).get("statement"),
        why_the_registered_null_misses_it=b.get("confound", {}).get(
            "why_the_registered_null_misses_it"),
        n_within_block_permutations=b.get("n_within_block_permutations"),
        margin_separating_uniform_null=s.get("n_separating_under_uniform_null"),
        margin_separating_within_block_null=s.get("n_separating_under_within_block_null"),
        mean_within_block_null_margin=s.get("mean_within_block_null_margin"),
        ordering_fifths_raw_below_05_uniform=oc.get("fifths", {}).get(
            "uniform_null_raw_below_05"),
        ordering_fifths_bh_uniform=oc.get("fifths", {}).get("uniform_null_surviving_bh"),
        ordering_fifths_raw_below_05_within_block=oc.get("fifths", {}).get(
            "within_block_raw_below_05"),
        ordering_fifths_bh_within_block=oc.get("fifths", {}).get("within_block_surviving_bh"),
        reading=oc.get("reading"),
        provenance_of_that_file=b.get("provenance", {}).get("written"),
        binding_rule="NEITHER HALF MAY BE REPORTED ALONE. The supervised arm's "
                     "ordered_not_closed is the ordering half; this block is the attack that "
                     "dismantles it. Quoting the first without the second launders the "
                     "artifact, which is the specific failure this embed exists to prevent.")


def row_identity_check(new_out, keys=("rows", "multiplicity", "synthetics", "exploratory",
                                      "verdict_counts", "verdict_counts_secondary"),
                       candidates=("results/phase2_keys_superseded_20260821T054305.json",)):
    """K18 -- POST-VERIFICATION ADDITION, 2026-08-21.

    The blockperm attack embedded above was computed by scripts/29_keys_blockperm.py against
    the rows of the 05:43:05 artifact. Embedding its numbers in THIS artifact is legitimate
    only if the regenerated rows are the same rows. Asserting that as an expectation is not
    good enough -- it is the same hole-class as the missing blockperm reference this
    regeneration exists to fix -- so the identity is measured here and the measurement is
    written into the artifact, against a copy of the superseded run kept in the repo rather
    than in a session-scoped temp directory.
    """
    def flat(o, p=""):
        if isinstance(o, dict):
            for k2, v2 in o.items():
                yield from flat(v2, p + "/" + str(k2))
        elif isinstance(o, list):
            for i, v2 in enumerate(o):
                yield from flat(v2, p + "/%d" % i)
        else:
            yield p, o

    for cand in candidates:
        p = Path(cand)
        if p.exists():
            old = json.loads(p.read_text())
            per = {}
            tot = 0
            worst = 0.0
            mismatched = []
            for key in keys:
                if key not in old or key not in new_out:
                    per[key] = dict(compared=0, note="absent from one artifact")
                    continue
                a = dict(flat(old[key]))
                b = dict(flat(new_out[key]))
                md = 0.0
                nd = 0
                for kk in set(a) & set(b):
                    if a[kk] != b[kk]:
                        nd += 1
                        if (isinstance(a[kk], (int, float)) and not isinstance(a[kk], bool)
                                and isinstance(b[kk], (int, float))):
                            md = max(md, abs(float(a[kk]) - float(b[kk])))
                        else:
                            mismatched.append(key + kk)
                per[key] = dict(n_values_compared=len(a), n_only_in_superseded=len(set(a) - set(b)),
                                n_only_in_regenerated=len(set(b) - set(a)),
                                n_differing=nd, max_abs_delta=float(md))
                tot += len(a)
                worst = max(worst, md)
            ok = all(v.get("n_differing", 0) == 0 for v in per.values())
            return dict(
                checked_against=str(cand), present=True, identical=bool(ok),
                n_values_compared=int(tot), max_abs_delta=float(worst),
                non_numeric_mismatches=mismatched[:20], per_block=per,
                why_this_matters="ordering_attack_blockperm embeds numbers scripts/"
                                 "29_keys_blockperm.py computed against the 05:43:05 rows. "
                                 "That embed is valid for this run only because the rows are "
                                 "identical, and this field is the evidence rather than the "
                                 "assertion.",
                interpretation=("every measured statistic reproduced exactly from the same "
                                "fixed seeds; the regeneration changed the REPORTING LAYER "
                                "ONLY" if ok else
                                "ROWS DIFFER -- the embedded blockperm numbers were computed "
                                "against different rows and must be regenerated before this "
                                "artifact is quoted"))
    return dict(checked_against=list(candidates), present=False, identical=None,
                warning="the superseded artifact is not on disk, so row identity with the "
                        "05:43:05 run -- the run the embedded blockperm numbers were computed "
                        "against -- could NOT be verified here")


def read_template_lineage(path="results/colors_existence.json"):
    p = Path(path)
    if not p.exists():
        return dict(path=str(path), present=False,
                    warning="the verified template's own artifact is not on disk, so the "
                            "end-to-end sensitivity of this pipeline cannot be stated here")
    c = json.loads(p.read_text())
    gr, pu = [], []
    for r in c.get("rows", []):
        a = r["conventions"]["A"] if "conventions" in r else r
        if "gap_ratio" in a:
            gr.append(float(a["gap_ratio"]))
            pu.append(float(a["p_closed_uniform"]))
    m = c.get("multiplicity", {})
    fams = {k: dict(raw=v.get("n_raw_below_05"), bh=v.get("n_surviving_bh_fdr"))
            for k, v in m.items() if isinstance(v, dict) and "n_raw_below_05" in v}
    return dict(
        path=str(path), present=True, n_cells=len(gr),
        gap_ratio_min=(min(gr) if gr else None), gap_ratio_max=(max(gr) if gr else None),
        p_closed_uniform_min=(min(pu) if pu else None),
        n_cells_closing_at_05=int(sum(x < 0.05 for x in pu)),
        multiplicity_families=fams,
        finding="the verified template scripts/24_colors_existence.py -- the one prereg 4.2 "
                "names as the model this script copies exactly -- is ITSELF 0/78 on closure, "
                "with 0 BH survivors in every closure family. Keys and colours land on the "
                "same closure floor.",
        consequence="THIS PIPELINE HAS NEVER BEEN SHOWN TO RECOVER A RING THAT GENUINELY "
                    "EXISTS IN THE MODEL. Both tracks' demonstrated power is statistic-level "
                    "only -- planted point clouds pushed through the geometry code -- with no "
                    "end-to-end demonstration that templates -> model -> readout -> closure "
                    "can find a real one. The keys negative therefore may NOT be read as 'the "
                    "pipeline that finds rings found none here'. It is consistent with the "
                    "template-averaged readout destroying planar geometry generally, which "
                    "would make both negatives uninformative about the model rather than "
                    "informative about pitch classes.",
        not_a_claim="this is not a claim that the pipeline is broken. It is a claim that its "
                    "end-to-end sensitivity is UNESTABLISHED and must be reported as such.")


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/phase2_keys.json")
    ap.add_argument("--layers", default=",".join(str(i) for i in range(26)))
    ap.add_argument("--n-background", type=int, default=N_BACKGROUND)
    ap.add_argument("--chunk", type=int, default=48)
    a = ap.parse_args()
    t0 = time.time()
    layers = [int(x) for x in a.layers.split(",")]

    print("=" * 92)
    print("PHASE 2 -- MUSICAL KEYS, supervised existence sweep")
    print(f"status: {STATUS}")
    print("DECLARATIONS (fixed before any activation is read):")
    print(f"  K1 fifths order   : {FIFTHS}")
    print(f"     chromatic order: {CHROMATIC}")
    print(f"     enharmonics    : {ENHARMONIC} (sharps are multi-token, prereg 5.1)")
    print(f"  K2 discrimination : rho_fifths vs rho_chromatic on the SAME 12 points, "
          f"paired {N_MARGIN_NULL}-draw shuffled-label margin null, two-sided 95th pct of "
          f"|margin|, strict, PRIMARY plane only")
    print(f"  K3 readouts       : token (embedding-confounded) | suffix (PRIMARY computed) "
          f"| final (tertiary)")
    print(f"  K4 templates      : {len(TEMPLATES)} used, {len(TEMPLATE_SPARES)} verified spares unused")
    print(f"  K5 discriminator  : closure. Ordering alone is never evidence.")
    print(f"  K6 nulls          : uniform angles ({N_GAP_NULL}); {a.n_background} noun draws; "
          f"{a.n_background} capital-letter (H-Z) draws; {N_PERM_ORDER} shuffled labels "
          f"+ {N_SHUFFLE_CONTROL}-shuffle control")
    print(f"     closure passes only if uniform AND noun AND letter all give p < 0.05")
    print(f"  K8 secondary      : widened 6-plane search, provenance stated, NOT in prereg 3.1; "
          f"no margin computed on a selected plane")
    print(f"  K10 verdicts      : recovered_fifths | recovered_chromatic | "
          f"closed_ring_unordered | ordered_not_closed | clean_negative | degenerate")
    print(f"  K12 blind arm     : untestable by prereg 3.1/5.1/6; the frozen catalogue is "
          f"NOT opened by this script")
    print(f"  K15 battery amendments: {len(BATTERY_AMENDMENTS)}, all made after the first "
          f"synthetic run and before any activation was read or the model was loaded:")
    for i, amend in enumerate(BATTERY_AMENDMENTS, 1):
        print(f"     ({i}) {amend.split('.')[0]}.")
    print("=" * 92, flush=True)

    ref = check_against_reference()
    cf = ref["closed_form_ring_check"]
    print(f"order_stats vs fits.cyclic_order_score: "
          f"{'MATCH' if all(t['rho_match'] for t in ref['trials']) else 'MISMATCH'} on rho "
          f"over 5 trials at k=12")
    print(f"closed-form ring check: fifths ring rho_f {cf['fifths_ring']['rho_fifths']:.4f} vs "
          f"rho_c {cf['fifths_ring']['rho_chromatic']:.4f} (margin "
          f"{cf['fifths_ring']['margin']:+.4f}); chromatic ring margin "
          f"{cf['chromatic_ring']['margin']:+.4f}; mirror-symmetric "
          f"{cf['mirror_symmetric']}; max achievable |margin| "
          f"{cf['max_achievable_abs_margin']:.4f}", flush=True)
    if not ref["passed"]:
        raise SystemExit("ordering / margin machinery disagrees with its reference")

    syn = synthetics(n_bg=a.n_background)
    print("\nsynthetic battery:")
    for nz in syn["noise_ladder"]:
        for key, lab in (("planted_fifths_ring", "fifths ring"),
                         ("planted_chromatic_ring", "chrom ring ")):
            r = syn[key][f"noise_{nz}"]
            print(f"  {lab} noise {nz:<5} gapR {r['gap_ratio']:6.2f}  pUnif {r['p_closed']:.4f}  "
                  f"pBg {r['p_vs_background']:.4f}  rhoF {r['rho_fifths']:+.3f}  "
                  f"rhoC {r['rho_chromatic']:+.3f}  margin {r['margin']:+.3f} "
                  f"(thr {r['margin_threshold']:.3f}) -> {r['margin_winner']}")
    for key in ("planted_ring_scrambled_labels", "buried_ring", "pure_noise",
                "planted_arc_200deg", "planted_line"):
        r = syn[key]
        print(f"  {key:30s} {r['n_seeds']:2d} seeds: rhoF {r['rho_fifths_min']:+.3f}.."
              f"{r['rho_fifths_max']:+.3f} (mean {r['rho_fifths_mean']:+.3f}), "
              f"gapR {r['gap_ratio_min']:5.1f}-{r['gap_ratio_max']:5.1f}, "
              f"ordersF {r['n_order_fifths_below_05']:2d}, "
              f"closes U{r['n_closed_uniform']:2d} bg{r['n_closed_primary_background']:2d} "
              f"K8bg{r['n_closed_secondary_background']:2d}, "
              f"marginSep {r['n_margin_separates']:2d} (mean {r['margin_mean']:+.3f})")
    print("  K5b margin power vs angular jitter "
          f"(threshold {syn['margin_threshold_reference']['threshold_mean']:.3f}, "
          f"ceiling {syn['margin_threshold_reference']['ceiling']:.3f}):")
    for j in syn["margin_power_by_angular_jitter"]["jitters_deg"]:
        r = syn["margin_power_by_angular_jitter"]["rows"][f"jitter_{j}deg"]
        print(f"    {j:>3}deg  mean margin {r['margin_mean']:+.3f}  "
              f"separates correctly {r['fraction_separating_correct']:.3f}  "
              f"backwards {r['fraction_separating_wrong_direction']:.3f}")
    uc = syn["uniform_null_calibration"]
    print(f"  uniform-angle null vs {uc['n_draws']} isotropic clouds (k={syn['k']}, d={syn['d']}): "
          f"mean gapR {uc['gap_ratio_mean']:.2f}, false acceptance at p<0.05 = "
          f"{uc['uniform_null_false_acceptance_at_p05']:.3f}")
    failed = [k for k, v in syn["gate"].items() if not v]
    print(f"  gate: {len(syn['gate']) - len(failed)}/{len(syn['gate'])} conditions met -> "
          f"{'PASS' if syn['passed'] else 'FAIL ' + str(failed)}", flush=True)
    if not syn["passed"]:
        raise SystemExit(f"synthetic battery failed on {failed}; not running on activations")

    model, tok = load_model(token=os.environ.get("HF_TOKEN"))
    bos = tok.bos_token_id

    # ---- K1 / K6 / K7 audits ---------------------------------------------
    key_keep, key_drop = token_audit(tok, CHROMATIC)
    if key_drop:
        raise SystemExit(f"label words not single-token: {[d['word'] for d in key_drop]}")
    key_words = [x["word"] for x in key_keep]
    letter_keep, letter_drop = token_audit(tok, LETTER_POOL)
    letter_words = [x["word"] for x in letter_keep]
    excl = set(MUSIC_SENSE_EXCLUSIONS)
    noun_cand = [w for w in dict.fromkeys(NOUN_POOL) if w not in excl]
    noun_excluded = [w for w in dict.fromkeys(NOUN_POOL) if w in excl]
    noun_keep, noun_drop = token_audit(tok, noun_cand)
    noun_words = [x["word"] for x in noun_keep]
    K = len(key_words)
    if K != 12:
        raise SystemExit(f"expected 12 pitch classes, got {K}")
    print(f"\nK1 audit: {K} pitch classes, all single-token: "
          f"{[(x['word'], x['ids'][0]) for x in key_keep]}")
    print(f"K6c audit: {len(letter_words)} capital letters H-Z kept "
          f"(dropped {[x['word'] for x in letter_drop]}); "
          f"C(19,12) = {comb(19, 12)} distinct 12-subsets exist")
    print(f"K7 audit: {len(noun_words)} nouns kept; "
          f"{len(noun_excluded)} removed by the music/letter exclusion rule "
          f"{noun_excluded}; {len(noun_drop)} dropped as multi-token", flush=True)

    # ---- template verification (manual ids == natural tokenisation) -------
    label_words = key_words
    pool_words = letter_words + noun_words

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
    letter_words = [w for w in letter_words if w not in dropped_pool]
    noun_words = [w for w in noun_words if w not in dropped_pool]

    tmpl_report = []
    for prefix, suffix in TEMPLATES:
        pre = tok.encode(prefix, add_special_tokens=False)
        suf = tok.encode(suffix, add_special_tokens=False)
        tmpl_report.append(dict(prefix=prefix, suffix=suffix, n_prefix_tokens=len(pre),
                                n_suffix_tokens=len(suf), note_index=1 + len(pre),
                                suffix_index=2 + len(pre), final_index=1 + len(pre) + len(suf),
                                first_post_note_token=tok.decode(suf[:1])))
    spare_report = []
    for prefix, suffix in TEMPLATE_SPARES:
        suf = tok.encode(suffix, add_special_tokens=False)
        spare_report.append(dict(prefix=prefix, suffix=suffix, n_suffix_tokens=len(suf),
                                 used=False,
                                 note="verified clean at prereg-time tokenizer precheck; "
                                      "unused under the K4 first-sixteen rule"))
    tokenisation_audit = dict(
        n_words_checked=len(label_words) + len(pool_words), n_templates=len(TEMPLATES),
        n_checks=(len(label_words) + len(pool_words)) * len(TEMPLATES),
        label_words_checked=label_words, label_mismatches=label_bad,
        pool_words_checked=len(pool_words), pool_words_dropped=sorted(dropped_pool),
        pool_mismatch_detail=pool_bad,
        rule="label mismatch aborts (labels are load-bearing); pool mismatch drops the "
             "word (mechanical, label-blind)")
    print(f"\ntokenisation audit: {tokenisation_audit['n_checks']} word x template checks, "
          f"{len(label_bad)} label mismatches, {len(dropped_pool)} pool words dropped "
          f"{sorted(dropped_pool)}", flush=True)
    if label_bad:
        raise SystemExit(f"label words with off-manifold ids: "
                         f"{sorted({d['word'] for d in label_bad})}")
    if len(noun_words) < 40 or len(letter_words) < 15:
        raise SystemExit(f"background pools too small: {len(noun_words)} nouns, "
                         f"{len(letter_words)} letters")

    words = key_words + letter_words + noun_words
    word_ids = [tok.encode(" " + w, add_special_tokens=False)[0] for w in words]
    idx = {w: i for i, w in enumerate(words)}
    key_idx = np.array([idx[w] for w in key_words])
    letter_idx = np.array([idx[w] for w in letter_words])
    noun_idx = np.array([idx[w] for w in noun_words])
    nat_sub = np.array([key_words.index(w) for w in NATURALS])   # K14(a)

    # codings over the twelve label points, in key_words order (= CHROMATIC order)
    chrom_code = np.array([CHROMATIC.index(w) for w in key_words])
    fifths_code = np.array([FIFTHS.index(w) for w in key_words])
    codings = dict(fifths=fifths_code, chromatic=chrom_code)
    print(f"K2 codings (in point order {key_words}):\n  fifths    {[int(x) for x in fifths_code]}"
          f"\n  chromatic {[int(x) for x in chrom_code]}", flush=True)

    # ---- forwards ---------------------------------------------------------
    print(f"\ncollecting {len(words)} words x {len(TEMPLATES)} templates x {len(layers)} "
          f"layers x 3 readouts ...", flush=True)
    tf = time.time()
    M = collect(model, tok, words, word_ids, layers, chunk=a.chunk)
    print(f"forwards done in {time.time() - tf:.1f}s", flush=True)
    del model
    if DEVICE == "mps":
        torch.mps.empty_cache()

    ref_layer = layers[0]
    _, basis_L0_token, _ = analyse(M["token"][ref_layer][key_idx], None, K, seed=0,
                                   do_order=False, both_conventions=False)

    rows, jack_rows, natural_rows = [], [], []
    print(f"\n{'PRIMARY (pre-declared, top-2 PCA plane)':<104}| {'K8 widened plane':<26}")
    print(f"{'readout':>8} {'L':>3} {'top2':>6} {'radR':>7} {'gapR':>7} {'pUnif':>8} "
          f"{'pNoun':>8} {'pLett':>8} {'rhoF':>7} {'pOrdF':>8} {'rhoC':>7} {'pOrdC':>8} "
          f"{'margin':>7} {'thr':>6} {'win':>5} {'ang(L0)':>8} | {'plane':>6} {'pNoun':>7} "
          f" verdict", flush=True)
    for ro in READOUTS:
        for l in layers:
            P = M[ro][l][key_idx]
            res, basis, xy = analyse(P, codings, K, seed=l, do_order=True)
            A = res["conventions"]["A"]
            B = res["conventions"]["B"]
            # K9: closure is sign-invariant, rho and the margin are reflection-invariant.
            sign_check = dict(
                gap_ratio_identical=bool(abs(A["gap_ratio"] - B["gap_ratio"]) < 1e-9),
                rho_fifths_identical=(None if res["degenerate"] else
                                      bool(abs(A["order"]["fifths"]["rho"]
                                               - B["order"]["fifths"]["rho"]) < 1e-12)),
                rho_chromatic_identical=(None if res["degenerate"] else
                                         bool(abs(A["order"]["chromatic"]["rho"]
                                                  - B["order"]["chromatic"]["rho"]) < 1e-12)),
                margin_identical=(None if res["degenerate"] else
                                  bool(abs(A["margin"]["margin"]
                                           - B["margin"]["margin"]) < 1e-12)),
                rot_only_rho_fifths_A=(None if res["degenerate"] else
                                       A["order"]["fifths"]["rotation_only"]["rho"]),
                rot_only_rho_fifths_B=(None if res["degenerate"] else
                                       B["order"]["fifths"]["rotation_only"]["rho"]),
                rot_only_rho_chromatic_A=(None if res["degenerate"] else
                                          A["order"]["chromatic"]["rotation_only"]["rho"]),
                rot_only_rho_chromatic_B=(None if res["degenerate"] else
                                          B["order"]["chromatic"]["rotation_only"]["rho"]))

            row = dict(readout=ro, layer=l, k=K,
                       **{kk: res[kk] for kk in ("degenerate", "top2_var", "pc_var")},
                       conventions=res["conventions"], sign_convention_check=sign_check)

            sec, sec_basis = secondary_analyse(P, codings, K, seed=l,
                                               do_order=not res["degenerate"])

            def bg(pool_idx, n_draws, seed):
                """Identical pipeline, same arity, same positions -- only the words differ."""
                r = np.random.default_rng(seed)
                gr = np.empty(n_draws); pc = np.empty(n_draws)
                rr = np.empty(n_draws); gb = np.empty(n_draws)
                for i in range(n_draws):
                    pick = pool_idx[r.choice(len(pool_idx), size=K, replace=False)]
                    gr[i], pc[i], gb[i], rr[i] = fast_closure(M[ro][l][pick])
                return dict(n_draws=int(n_draws), pool_size=int(len(pool_idx)),
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
            b_lett = bg(letter_idx, a.n_background, 2000 + l)
            sec["p_vs_noun_background"] = b_noun["secondary_best_plane"]["p_vs_background"]
            sec["p_vs_letter_background"] = b_lett["secondary_best_plane"]["p_vs_background"]
            row["secondary_plane_search"] = sec
            row["principal_angles_secondary_to_L0_token_plane_deg"] = principal_angles_deg(
                sec_basis, basis_L0_token)
            row["background_nouns"] = b_noun
            row["background_letters"] = b_lett
            row["principal_angles_to_L0_token_plane_deg"] = principal_angles_deg(
                basis, basis_L0_token)

            # ---- K14(b): recovered angular order and the flats-blob check ----
            if not res["degenerate"]:
                ang = np.asarray(A["angles_rad"])
                seq = [key_words[i] for i in np.argsort(ang)]
                row["recovered_angular_order"] = seq
                row["flats_contiguity"] = dict(
                    **flats_contiguity(seq),
                    exact_reference_probability_5_of_12=float(12 / comb(12, 5)),
                    note="exact combinatorial reference for five of twelve uniformly random "
                         "cyclic positions being contiguous; no fitted null")

            # ---- K10 verdict -- PRIMARY statistic only ------------------------
            if res["degenerate"]:
                verdict = "degenerate"
                verdict_noun_only = "degenerate"
            else:
                closed_uniform = A["p_closed_uniform"] < 0.05
                closed_noun = b_noun["p_vs_background"] < 0.05
                closed_lett = b_lett["p_vs_background"] < 0.05
                closed = closed_uniform and closed_noun and closed_lett
                m = A["margin"]
                sep = bool(m["separates"])
                any_order = (A["order"]["fifths"]["p"] < 0.05
                             or A["order"]["chromatic"]["p"] < 0.05)
                if closed and sep:
                    verdict = f"recovered_{m['winner']}"
                elif closed:
                    verdict = "closed_ring_unordered"
                elif sep or any_order:
                    verdict = "ordered_not_closed"
                else:
                    verdict = "clean_negative"
                closed_n = closed_uniform and closed_noun
                if closed_n and sep:
                    verdict_noun_only = f"recovered_{m['winner']}"
                elif closed_n:
                    verdict_noun_only = "closed_ring_unordered"
                elif sep or any_order:
                    verdict_noun_only = "ordered_not_closed"
                else:
                    verdict_noun_only = "clean_negative"
                row["either_order_p_below_05"] = bool(any_order)
            row["verdict"] = verdict
            row["verdict_noun_background_only"] = verdict_noun_only
            row["verdict_rule"] = ("closure = uniform AND noun AND letter, all p < 0.05; "
                                   "discrimination = K2 margin on the primary plane")

            if res["degenerate"]:
                verdict_sec = "degenerate"
            else:
                closed_s = (sec["p_vs_noun_background"] < 0.05
                            and sec["p_vs_letter_background"] < 0.05)
                ordered_s = min(sec["order"]["fifths"]["p"], sec["order"]["chromatic"]["p"]) < 0.05
                verdict_sec = ("closed_ring_widened_search" if closed_s and ordered_s else
                               "closed_ring_unordered" if closed_s else
                               "ordered_not_closed" if ordered_s else "clean_negative")
            row["verdict_secondary_plane_search"] = verdict_sec
            rows.append(row)

            if res["degenerate"]:
                print(f"{ro:>8} {l:3d}  degenerate")
            else:
                m = A["margin"]
                print(f"{ro:>8} {l:3d} {res['top2_var']:6.3f} {A['radius_ratio']:7.2f} "
                      f"{A['gap_ratio']:7.2f} {A['p_closed_uniform']:8.4f} "
                      f"{b_noun['p_vs_background']:8.4f} {b_lett['p_vs_background']:8.4f} "
                      f"{A['order']['fifths']['rho']:+7.3f} {A['order']['fifths']['p']:8.4f} "
                      f"{A['order']['chromatic']['rho']:+7.3f} "
                      f"{A['order']['chromatic']['p']:8.4f} "
                      f"{m['margin']:+7.3f} {m['null_abs_p95']:6.3f} "
                      f"{str(m['winner'])[:5]:>5} "
                      f"{row['principal_angles_to_L0_token_plane_deg'][0]:8.1f} | "
                      f"{str(sec['selected_plane']):>6} {sec['p_vs_noun_background']:7.4f} "
                      f" {verdict}", flush=True)

            # ---- leave-one-out jackknife where closure passes ----------------
            if not res["degenerate"] and A["p_closed_uniform"] < 0.05:
                jk = []
                for j in range(K):
                    keep = [i for i in range(K) if i != j]
                    q, _, _ = analyse(P[keep], None, K, seed=l, do_order=False,
                                      both_conventions=False)
                    qa = q["conventions"]["A"]
                    jk.append(dict(dropped=key_words[j], gap_ratio=qa["gap_ratio"],
                                   p_closed_uniform=qa["p_closed_uniform"]))
                jack_rows.append(dict(readout=ro, layer=l, leave_one_out=jk,
                                      worst=max(jk, key=lambda x: x["p_closed_uniform"]),
                                      n_still_closed=int(sum(x["p_closed_uniform"] < 0.05
                                                             for x in jk))))

            # ---- K14(a) naturals-only, CLOSURE ONLY --------------------------
            Pn = P[nat_sub]
            qn, _, _ = analyse(Pn, None, len(nat_sub), seed=l, do_order=False,
                               both_conventions=False)
            qna = qn["conventions"]["A"]
            rr = np.random.default_rng(7000 + l)
            gn = np.empty(N_NATURALS_BG)
            for i in range(N_NATURALS_BG):
                pick = noun_idx[rr.choice(len(noun_idx), size=len(nat_sub), replace=False)]
                gn[i] = fast_closure(M[ro][l][pick])[0]
            natural_rows.append(dict(
                readout=ro, layer=l, k=len(nat_sub), words=NATURALS,
                gap_ratio=qna["gap_ratio"], radius_ratio=qna["radius_ratio"],
                p_closed_uniform=qna["p_closed_uniform"],
                p_vs_noun_background=float((1 + int((gn <= qna["gap_ratio"]).sum()))
                                           / (1 + N_NATURALS_BG)),
                n_background_draws=int(N_NATURALS_BG)))

    # ---- K13 multiplicity ---------------------------------------------------
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
        note="3 readouts x 26 layers. An isolated pass at alpha = 0.05 is expected by chance "
             "and is not a finding; BH-FDR within this sweep is the honest count. Each family "
             "carries its p-value floor, because an add-one empirical p cannot fall below "
             "1/(n_null+1) and a BH column has to be read against what was reachable.",
        ordering_fifths=fdr([r["conventions"]["A"]["order"]["fifths"]["p"] for r in live],
                            "ordering p, fifths coding, primary plane", N_PERM_ORDER),
        ordering_chromatic=fdr([r["conventions"]["A"]["order"]["chromatic"]["p"] for r in live],
                               "ordering p, chromatic coding, primary plane", N_PERM_ORDER),
        margin_two_sided=fdr([r["conventions"]["A"]["margin"]["p_two_sided"] for r in live],
                             "fifths-vs-chromatic margin p (two-sided), primary plane",
                             N_MARGIN_NULL),
        closure_uniform=fdr([r["conventions"]["A"]["p_closed_uniform"] for r in live],
                            "closure p vs uniform angles, primary plane", N_GAP_NULL),
        closure_noun_background=fdr([r["background_nouns"]["p_vs_background"] for r in live],
                                    "closure p vs matched noun background, primary plane", nb),
        closure_letter_background=fdr([r["background_letters"]["p_vs_background"] for r in live],
                                      "closure p vs capital-letter background, primary plane", nb),
        closure_noun_background_secondary=fdr(
            [r["secondary_plane_search"]["p_vs_noun_background"] for r in live],
            "closure p vs matched noun background, K8 selected plane", nb),
        ordering_fifths_secondary=fdr(
            [r["secondary_plane_search"]["order"]["fifths"]["p"] for r in live],
            "ordering p, fifths coding, K8 selected plane", N_PERM_ORDER),
    )

    # ---- K11 ontology-level aggregation (frozen before the run) -------------
    bh_noun = {(s["readout"], s["layer"]) for s in mult["closure_noun_background"]["surviving"]}
    bh_lett = {(s["readout"], s["layer"]) for s in mult["closure_letter_background"]["surviving"]}
    bh_ord = ({(s["readout"], s["layer"]) for s in mult["ordering_fifths"]["surviving"]}
              | {(s["readout"], s["layer"]) for s in mult["ordering_chromatic"]["surviving"]})
    suffix_rows = [r for r in live if r["readout"] == "suffix"]
    closed_bh = [r for r in suffix_rows
                 if r["verdict"] in ("recovered_fifths", "recovered_chromatic",
                                     "closed_ring_unordered")
                 and (r["readout"], r["layer"]) in bh_noun
                 and (r["readout"], r["layer"]) in bh_lett]
    recovered_bh = [r for r in closed_bh
                    if r["verdict"] in ("recovered_fifths", "recovered_chromatic")]
    ordered_bh = [r for r in suffix_rows
                  if r["verdict"] == "ordered_not_closed" and (r["readout"], r["layer"]) in bh_ord]
    if recovered_bh:
        onto = recovered_bh[0]["verdict"]
    elif closed_bh:
        onto = "closed_ring_unordered"
    elif ordered_bh:
        onto = "ordered_not_closed"
    else:
        onto = "clean_negative"
    # ------------------------------------------------------------------------------
    # K11(5) CORRECTED, 2026-08-21, post-verification. The superseded code read:
    #
    #     arm_verdict = ("structure_exists_supervised_only"
    #                    if onto in ("recovered_fifths", "recovered_chromatic",
    #                                "closed_ring_unordered", "ordered_not_closed")
    #                    else "clean_negative")
    #
    # and it was WRONG in three independent ways.
    #
    #  (a) DEFINITIONAL GATE NOT MET. Prereg 4.3 defines structure_exists_supervised_only as
    #      "the structure appears under templated supervision but not in the blind
    #      partitions". The structure prereg 3.1 predicts is a CLOSED 12-CYCLE. It did not
    #      appear: 0/78 cells close, the minimum p_uniform over the whole sweep is 0.1784,
    #      and every one of the three closure families has 0 BH survivors. An ordering pass
    #      is not the predicted structure appearing; prereg 3.1's own table calls
    #      ordered_not_closed "explicitly NOT a circle-of-fifths claim".
    #  (b) NO SUCH SLOT EXISTS. Prereg 4.3 is titled "Reported verdicts, one per ontology PER
    #      ARM". structure_exists_supervised_only and ordered_not_closed are ALTERNATIVES in
    #      that one table, not a per-arm/cross-arm pair to be emitted together.
    #  (c) A CEILING REPORTED AS AN ACHIEVEMENT. K11 clause 5 as frozen says "...the
    #      STRONGEST REACHABLE cross-arm verdict is structure_exists_supervised_only". That
    #      sentence bounds what the arm could have earned. The old code assigned the bound as
    #      though it had been earned. This is the single field a downstream reader would
    #      quote as "keys structure exists", and it is the one word that launders the whole
    #      artifact.
    #
    # The frozen K11 clauses 1-4 are UNTOUCHED and still produce `onto`. Only clause 5 --
    # the cross-arm mapping -- is corrected, and the superseded text is quoted verbatim in
    # the output rather than deleted, following the correction pattern the blockperm
    # artifact's own provenance block established.
    # ------------------------------------------------------------------------------
    closure_gate_met = bool(recovered_bh or closed_bh)
    if closure_gate_met:
        arm_verdict = "structure_exists_supervised_only"
    elif onto == "ordered_not_closed":
        arm_verdict = "ordered_not_closed (supervised) / untestable (blind)"
    else:
        arm_verdict = "clean_negative (supervised) / untestable (blind)"
    pitch_class_geometry_verdict = ("recovered" if closure_gate_met
                                    else "clean_negative")

    # K14(c) which order wins, COUNTS ONLY
    win_tally = {}
    for ro in READOUTS:
        sel = [r for r in live if r["readout"] == ro]
        win_tally[ro] = dict(
            n_cells=len(sel),
            margin_positive_fifths_ahead=int(sum(r["conventions"]["A"]["margin"]["margin"] > 0
                                                 for r in sel)),
            margin_negative_chromatic_ahead=int(sum(r["conventions"]["A"]["margin"]["margin"] < 0
                                                    for r in sel)),
            n_margin_separates=int(sum(r["conventions"]["A"]["margin"]["separates"] for r in sel)),
            separating_winners=[dict(layer=r["layer"], winner=r["conventions"]["A"]["margin"]["winner"],
                                     margin=r["conventions"]["A"]["margin"]["margin"])
                                for r in sel if r["conventions"]["A"]["margin"]["separates"]],
            mean_abs_margin=float(np.mean([abs(r["conventions"]["A"]["margin"]["margin"])
                                           for r in sel])) if sel else None)
    win_tally["note"] = ("K14(c): counts only. No binomial p-value -- the 78 cells are the "
                         "same activations propagated through layers, so an independence "
                         "assumption would be false.")

    # K14(b) aggregated: how often are all five flats adjacent in the recovered order?
    flats_agg = {}
    for ro in list(READOUTS) + ["all"]:
        sel = [r for r in live if (ro == "all" or r["readout"] == ro)]
        sel = [r for r in sel if "flats_contiguity" in r]
        flats_agg[ro] = dict(
            n_cells=len(sel),
            n_all_five_flats_contiguous=int(sum(r["flats_contiguity"]["contiguous"] for r in sel)),
            longest_flat_run_mean=(float(np.mean([r["flats_contiguity"]["longest_run"]
                                                  for r in sel])) if sel else None))
    flats_agg["exact_reference_probability"] = float(12 / comb(12, 5))
    flats_agg["expected_by_chance"] = float(len([r for r in live if "flats_contiguity" in r])
                                            * 12 / comb(12, 5))
    flats_agg["interpretation"] = (
        "The five flats (Db, Eb, Gb, Ab, Bb) are corpus-rare multi-character tokens; the "
        "seven naturals are single capital letters. If the flats come out CONTIGUOUS in the "
        "recovered angular order far more often than the exact reference 12/C(12,5) = "
        "0.01515, the dominant geometry in this point cloud separates the two lexical "
        "classes rather than ordering the twelve pitch classes -- and any ordering statistic "
        "computed on it is reading token class, not pitch class. Reported as a count against "
        "an exact combinatorial reference; no fitted null and no p-value beyond that number.")

    # ------------------------------------------------------------------------------
    # K16 -- the closure power curve the registered battery never had. Computed HERE rather
    # than inside synthetics() because its background column is referenced to the OBSERVED
    # matched-background 5th percentile, which does not exist until the sweep has run. The
    # uniform column is data-free and could have been computed anywhere.
    # ------------------------------------------------------------------------------
    noun_p05 = [float(r["background_nouns"]["gap_ratio_p05"]) for r in live
                if "background_nouns" in r]
    lett_p05 = [float(r["background_letters"]["gap_ratio_p05"]) for r in live
                if "background_letters" in r]
    best_gap_observed = min(float(r["conventions"]["A"]["gap_ratio"]) for r in live)
    best_gap_cell = min(live, key=lambda r: r["conventions"]["A"]["gap_ratio"])
    bg_thr = {}
    if noun_p05:
        bg_thr["matched_noun_background_mean_p05"] = float(np.mean(noun_p05))
    if lett_p05:
        bg_thr["capital_letter_background_mean_p05"] = float(np.mean(lett_p05))
    print(f"\nK16 closure power curve (post-verification addition): thresholds "
          + ", ".join(f"{k2} = {v2:.4f}" for k2, v2 in bg_thr.items()), flush=True)
    cpc = closure_power_curve(k=len(key_words), bg_thresholds=bg_thr)
    cpc["observed_reference"] = dict(
        best_gap_ratio_anywhere_in_sweep=best_gap_observed,
        best_cell=dict(readout=best_gap_cell["readout"], layer=best_gap_cell["layer"]),
        note="the single most ring-like cell the whole 78-cell sweep produced. Reading it "
             "off the point_cloud rows below gives the angular fidelity a planted ring "
             "needs in order to look this bad -- i.e. the reach of the negative.")
    # Where does the observed best sit on the planted-ring curve? Label-free, purely
    # descriptive: the smallest jitter rung whose median planted gap_ratio is already worse
    # than the best real cell.
    _rungs = [(v["jitter_deg"], v["point_cloud"]["gap_ratio_median"])
              for v in cpc["rows"].values()]
    _sorted = sorted(_rungs)
    _worse = [j for j, g in _sorted if g >= best_gap_observed]
    _at = float(min(_worse)) if _worse else None
    _below = [j for j, g in _sorted if g < best_gap_observed]
    cpc["observed_reference"]["planted_ring_matches_observed_best_at_jitter_deg"] = _at
    cpc["observed_reference"]["planted_ring_bracket_deg"] = [
        (float(max(_below)) if _below else None), _at]
    cpc["observed_reference"]["bracket_rule"] = (
        "the median planted-ring gap_ratio is better than the best observed cell at the first "
        "rung and worse at the second, so the best cell in the entire 78-cell sweep is no "
        "more ring-like than a planted 12-ring carrying that much angular jitter")

    blind_arm = dict(
        verdict="untestable",
        decided_by="docs/phase2_prereg_2026-08-21.md sections 3.1, 5.1, 5.6 and 6, before any "
                   "placement was examined",
        reason_count="7 of 12 pitch classes clear the >= 100 cache-count floor (the seven "
                     "naturals); the frozen artifact's label convention needs "
                     "ceil(2/3 * 12) = 8. All five sharps are multi-token; the five flats are "
                     "single-token but occur 0-6 times except ' Ab' (102, the abbreviation).",
        reason_geometry="the seven naturals are CONTIGUOUS in fifths order (a 180-degree arc, "
                        "reference ring ref_p_closed = 0.895 -- it could not close whatever the "
                        "geometry) and interleaved in chromatic order (ref_p_closed = 0.0001), "
                        "so a naturals-only test can only confirm chromatic and only reject "
                        "fifths",
        catalogue_not_opened=True,
        catalogue_path="results/token_type_geometry.json",
        forbidden_by_prereg_section_7=["reporting a blind-arm negative for musical keys",
                                       "any fifths-vs-chromatic claim from the seven naturals "
                                       "alone, in either direction"],
        polysemy_audit_source="results/prereg_pool_counts.json (counts only, computed before "
                              "the prereg freeze); ' A' 2833 is dominated by the indefinite "
                              "article and list markers, ' C' 856 by the programming language "
                              "and grades, and the lowercase witness ' a' occurs 64328 times")

    out = dict(
        status=STATUS,
        item="Phase-2 musical keys: supervised existence sweep and the fifths-vs-chromatic "
             "ordering discrimination",
        script="scripts/28_phase2_keys.py",
        prereg="docs/phase2_prereg_2026-08-21.md",
        generated=time.strftime("%Y-%m-%dT%H:%M:%S"),
        model="google/gemma-3-1b-pt", layers=layers, n_templates=len(TEMPLATES),
        declarations=dict(
            K1_label_set=dict(fifths_order=FIFTHS, chromatic_order=CHROMATIC,
                              enharmonic_mapping=ENHARMONIC, naturals=NATURALS, flats=FLATS,
                              note="the same twelve points scored under two orders; lowercase "
                                   "note letters are prereg 5.1 polysemy witnesses and are "
                                   "never scored"),
            K2_discriminating_test=(
                "closure passes the uniform null AND the matched backgrounds, AND "
                "rho_fifths > rho_chromatic (or the reverse) by a margin exceeding the 95th "
                "percentile of |margin| under a paired 1000-draw shuffled-label control. "
                "Both codings scored on the same angles with the same tie-break seed; strict "
                "comparison; PRIMARY plane only."),
            K3_readouts=dict(token="the note token itself -- embedding-confounded",
                             suffix="first token AFTER the note -- PRIMARY computed readout",
                             final="last token of the template -- tertiary, exploratory"),
            K4_templates=dict(n_used=len(TEMPLATES), n_verified=len(TEMPLATE_CANDIDATES),
                              rule="every prefix establishes the musical-key sense before the "
                                   "note token; no suffix names a key, note or accidental; "
                                   "suffix >= 3 tokens; no bare capital A-G anywhere in any "
                                   "template; first sixteen of twenty verified candidates",
                              spares=spare_report),
            K5_discriminator="closure; ordering alone is never evidence",
            K6_nulls=dict(uniform_angles=N_GAP_NULL, background_draws=a.n_background,
                          background_primary="single-token common nouns, same arity, identical "
                                             "templates and positions",
                          background_keys_specific="single capital letters H-Z (19 letters, none "
                                                   "a note name), prereg 3.1",
                          letter_pool_distinct_subsets=int(comb(19, 12)),
                          letter_pool_caveat="2000 draws resample a universe of only 50388 "
                                             "distinct 12-subsets; the pool's diversity, not "
                                             "the draw count, bounds this tier",
                          order_permutations=N_PERM_ORDER, shuffle_control=N_SHUFFLE_CONTROL,
                          margin_null_draws=N_MARGIN_NULL,
                          closure_rule="uniform AND noun AND letter, all p < 0.05",
                          background_not_scored_for_ordering="random nouns and random letters "
                                                             "have no true cyclic order"),
            K7_exclusion_rule="every word with an established musical sense, and every word "
                              "naming a letter of the alphabet, is removed from the noun pool; "
                              "calendar words are excluded so the pool cannot inherit this "
                              "project's other ontologies",
            K7_exclusions=sorted(MUSIC_SENSE_EXCLUSIONS),
            K8_secondary_plane_search=(
                "NOT in prereg 3.1. Added at script-writing time, before any activation was "
                "read, patterned on the D10 block of scripts/24_colors_existence.py which "
                "prereg 7 names as the model for a provenance-stated addition. All C(4,2) = 6 "
                "planes of the top 4 PCs; label-free selection; the uniform-angle p on a "
                "selected plane is invalid and marked as such; every background draw gets the "
                "identical min-over-6-planes freedom; verdicts and the K2 margin are computed "
                "on the PRIMARY plane only."),
            K9_sign_conventions="both PC2 sign conventions reported; rho and the margin must be "
                                "identical (reflection-invariant) and that is asserted; "
                                "rotation-only rho exposes handedness",
            K10_verdict_vocabulary=dict(
                recovered_fifths="closure passes all three tiers AND the margin separates with "
                                 "fifths ahead",
                recovered_chromatic="same, chromatic ahead -- registered by the prereg in "
                                    "advance as EQUALLY INTERPRETABLE, not a failure",
                closed_ring_unordered="closure passes, the margin does not separate",
                ordered_not_closed="an ordering separates, closure does not. NOT a "
                                   "circle-of-fifths claim",
                clean_negative="neither", degenerate="no usable geometry",
                degenerate_provenance="NOT in prereg 3.1's verdict table. Added at "
                                      "script-writing time as a guard for point clouds with "
                                      "no usable geometry. It fired on 0 of 78 cells, so it "
                                      "is inert here; recorded in prereg_deviations rather "
                                      "than left silent."),
            K11_aggregation_rule=(
                "suffix is the primary readout; token at layers 0-2 is the declared embedding "
                "confound and can never carry the headline. The ontology verdict is the "
                "strongest per-cell verdict reached at the suffix readout by a cell whose noun "
                "AND letter background closure p-values both survive BH within their 78-cell "
                "families; else ordered_not_closed if a suffix ordering p survives BH; else "
                "clean_negative. CLAUSE 5 CORRECTED 2026-08-21 -- see "
                "K11_clause5_correction; clauses 1-4 are untouched and still produce the "
                "supervised verdict."),
            K11_clause5_correction=dict(
                corrected="2026-08-21, after verification of the 05:43:05 run",
                superseded_text_verbatim=(
                    "Cross-arm (prereg 4.3): the blind arm is untestable, so the strongest "
                    "reachable cross-arm verdict is structure_exists_supervised_only."),
                superseded_code_verbatim=(
                    'arm_verdict = ("structure_exists_supervised_only" if onto in '
                    '("recovered_fifths", "recovered_chromatic", "closed_ring_unordered", '
                    '"ordered_not_closed") else "clean_negative")'),
                what_was_wrong=[
                    "(a) DEFINITIONAL GATE. Prereg 4.3 defines structure_exists_supervised_only "
                    "as 'the structure appears under templated supervision but not in the blind "
                    "partitions'. The predicted structure (prereg 3.1) is a CLOSED 12-CYCLE. It "
                    "did not appear: 0/78 closure, min p_uniform 0.1784, 0 BH survivors in all "
                    "three closure families. An ordering pass is not the predicted structure "
                    "appearing -- prereg 3.1's own table calls ordered_not_closed 'explicitly "
                    "NOT a circle-of-fifths claim'.",
                    "(b) NO SUCH SLOT. Prereg 4.3 is titled 'Reported verdicts, one per "
                    "ontology PER ARM'. structure_exists_supervised_only and "
                    "ordered_not_closed are ALTERNATIVES in that one table, not a "
                    "per-arm/cross-arm pair to be emitted side by side.",
                    "(c) CEILING REPORTED AS ACHIEVEMENT. The frozen clause says 'the "
                    "STRONGEST REACHABLE cross-arm verdict is ...'. That bounds what the arm "
                    "could have earned. The code assigned the bound as though it had been "
                    "earned. This is the single field a downstream reader would quote as "
                    "'keys structure exists'.",
                ],
                replacement_rule=(
                    "structure_exists_supervised_only is emitted ONLY when the closure gate is "
                    "met, i.e. at least one suffix cell closes and survives BH against BOTH "
                    "matched backgrounds. Otherwise the cross-arm field carries the honest "
                    "pair: 'ordered_not_closed (supervised) / untestable (blind)' when an "
                    "ordering survives BH, else 'clean_negative (supervised) / untestable "
                    "(blind)'. A separate plain-reading field, pitch_class_geometry, states "
                    "the headline in one word."),
                pattern="corrected rather than deleted, following the correction convention "
                        "results/phase2_keys_blockperm.json's own provenance block set."),
            K12_blind_arm="untestable, decided by the prereg before any placement was examined; "
                          "the frozen catalogue is not opened by this script",
            K13_multiplicity="BH within this sweep, per family, with raw counts, the "
                             "expected-by-chance count and each family's p-value floor",
            K14_exploratory=dict(
                naturals_only="CLOSURE STATISTICS ONLY -- no ordering rho against either "
                              "coding, because prereg 7 forbids any fifths-vs-chromatic claim "
                              "from the seven naturals in either direction",
                naturals_only_reference=dict(
                    ref_p_closed_fifths_order=0.89521, ref_p_closed_chromatic_order=0.00010,
                    source="docs/phase2_prereg_2026-08-21.md section 6"),
                flats_contiguity="exact combinatorial reference 12/C(12,5) = 0.01515; no "
                                 "fitted null",
                order_win_tally="counts only, no binomial p-value"),
            K15_battery_amendments=BATTERY_AMENDMENTS,
            K15_note="every change listed was made after the first run of the synthetic "
                     "battery and before any activation was read or the model was loaded. "
                     "K1-K14 -- label set, readouts, discriminating test, nulls, verdict "
                     "rules, aggregation rule -- are untouched.",
            K16_closure_power=(
                "POST-VERIFICATION ADDITION 2026-08-21, provenance stated (prereg 7). The "
                "registered battery had NO closure power curve; its SNR ladder is "
                "in-plane-noiseless by construction (K15(3) states the arithmetic itself) and "
                "K15(3)'s jitter curve reports MARGIN power only. So this sweep's negative -- "
                "which is carried by CLOSURE -- was reported with a demonstrated power "
                "envelope for the wrong statistic. closure_power_by_angular_jitter supplies "
                "the missing one, against both the uniform null and the OBSERVED matched "
                "backgrounds. It adds no search family and can only narrow what the negative "
                "claims to exclude."),
            K17_required_cross_references=(
                "POST-VERIFICATION ADDITION 2026-08-21. The 05:43:05 run of this script "
                "contained ZERO references to results/phase2_keys_blockperm.json -- grepping "
                "'blockperm', 'within_block' and '29_keys' over its 1,498,691 bytes returned "
                "0 hits -- while its verdict block reported ordered_not_closed. The stated "
                "rule is that neither half may be reported alone. Two artifacts are now "
                "embedded so they cannot be separated from the verdict: (a) "
                "ordering_attack_blockperm, the within-block permutation null that collapses "
                "the fifths ordering signal; (b) template_lineage_sensitivity, the verified "
                "template's own 0/78 closure floor, which is why this negative may not be "
                "read as 'the pipeline that finds rings found none here'."),
            caveat_that_survives_a_positive=(
                "A closed ring at the suffix readout shows the information is present one token "
                "later; attention can copy an embedding forward, so this is not by itself proof "
                "of recomputation. The principal angle between the suffix plane and the "
                "earliest-layer token plane is the only handle reported on that. The blind "
                "representation caveats of prereg 4.5 do not apply to this arm, which "
                "generates its own activations, but layers 0-4 remain embedding-adjacent."),
        ),
        templates=tmpl_report, tokenisation_audit=tokenisation_audit,
        embedding_reference_plane=dict(
            readout="token", layer=int(ref_layer),
            note="the plane every principal_angles_* field is measured against: the pitch-class "
                 "point cloud read at the note token itself, at the earliest layer in this "
                 "sweep. A suffix plane at a small angle to it is consistent with an embedding "
                 "copied forward."),
        token_audit=dict(pitch_classes=key_keep, pitch_classes_dropped=key_drop,
                         letter_pool_kept=letter_words, letter_pool_dropped=letter_drop,
                         noun_pool_kept=noun_words, noun_pool_dropped=noun_drop,
                         noun_pool_excluded_by_rule=noun_excluded),
        codings=dict(point_order=key_words, fifths=[int(x) for x in fifths_code],
                     chromatic=[int(x) for x in chrom_code]),
        reference_check=ref, synthetics=syn,
        rows=rows, jackknife=jack_rows,
        exploratory=dict(naturals_only_closure=natural_rows, order_win_tally=win_tally,
                         flats_contiguity_aggregate=flats_agg),
        blind_arm=blind_arm,
        multiplicity=mult,
        closure_power_by_angular_jitter=cpc,
        ordering_attack_blockperm=read_blockperm(),
        template_lineage_sensitivity=read_template_lineage(),
        prereg_deviations=[
            dict(id="5a", field="background draws",
                 prereg_text="prereg 4.2(b) registers '300 draws of k matched same-arity "
                             "common nouns'",
                 as_run=f"{a.n_background} draws for BOTH the noun pool and the capital-letter "
                        f"pool",
                 direction="CONSERVATIVE for a positive: more draws is a finer empirical "
                           "p-floor (1/(n+1)), which makes a background pass harder, not "
                           "easier. It cannot manufacture a positive.",
                 status="silent numeric substitution against the frozen text in the 05:43:05 "
                        "run -- the count was declared in K6 but the deviation from prereg 4.2 "
                        "was not. Declared here. NOT reverted: reverting to 300 would coarsen "
                        "the p-floor and weaken the negative's own background tier.",
                 also="the letter pool's real bound is diversity, not draw count -- 19 letters "
                      "give only C(19,12) = 50388 distinct 12-subsets, which K6 already "
                      "records"),
            dict(id="5b", field="verdict vocabulary",
                 prereg_text="prereg 3.1's verdict table lists recovered (fifths), recovered "
                             "(chromatic), closed_ring_unordered, ordered_not_closed, "
                             "clean_negative, untestable",
                 as_run="'degenerate' was added to the K10 vocabulary",
                 direction="INERT: it fired on 0 of 78 cells and on 0 of 78 secondary cells, "
                           "so no verdict anywhere depends on it",
                 status="declared here; the guard is kept because removing it would change "
                        "behaviour on a cloud with no usable geometry"),
            dict(id="5c", field="K2 margin test sidedness",
                 prereg_text="prereg 3.1 phrases the discriminating test as 'rho_fifths > "
                             "rho_chromatic (or the reverse) by a margin exceeding the 95th "
                             "percentile'",
                 as_run="implemented TWO-SIDED on |margin| against the 95th percentile of "
                        "|margin| under the paired shuffled-label null",
                 direction="CONSERVATIVE: a two-sided threshold is strictly harder to cross "
                           "than the directional one, so it cannot manufacture a "
                           "discrimination",
                 status="was already declared in K2; restated here for completeness"),
            dict(id="5d", field="K8 six-plane widened search",
                 prereg_text="not in prereg 3.1",
                 as_run="all C(4,2) = 6 planes of the top 4 PCs, label-free selection",
                 direction="HANDLED CORRECTLY: modelled on the D10 precedent prereg 7 names "
                           "as the permitted form, provenance stated, the uniform-angle p on a "
                           "selected plane marked invalid, and every background draw given the "
                           "identical min-over-6-planes freedom",
                 status="noted for completeness only. The single closed_ring_unordered in "
                        "verdict_counts_secondary (suffix, layer 18) is quarantined to this "
                        "secondary tier, never reaches the headline, and did not survive BH "
                        "(closure_noun_background_secondary: 1 raw, 0 BH)"),
        ],
        regeneration_of=dict(
            supersedes="results/phase2_keys.json generated 2026-08-21T05:43:05",
            superseded_artifact_kept_at=(
                "results/phase2_keys_superseded_20260821T054305.json -- a byte copy of the "
                "05:43:05 run, kept in the repo so the row-identity check below is "
                "reproducible by anyone, not just by the session that ran it"),
            reason="verification of the 05:43:05 artifact found five issues, all in the "
                   "REPORTING layer. No measured statistic, null, label set, template, "
                   "readout or per-cell verdict rule was changed.",
            superseded_verdict_block_verbatim={
                "ontology": "musical_keys",
                "supervised_arm_verdict_prereg_3_1": "ordered_not_closed",
                "cross_arm_verdict_prereg_4_3": "structure_exists_supervised_only",
                "blind_arm_verdict": "untestable",
                "fifths_vs_chromatic_discrimination": "not made",
                "n_suffix_cells_closed_and_bh": 0,
                "n_suffix_cells_recovered_and_bh": 0,
                "n_suffix_cells_ordered_and_bh": 8,
                "rule": "K11, frozen before the run",
            },
            what_changed=[
                "ISSUE 1: cross_arm_verdict_prereg_4_3 no longer reports the CEILING "
                "structure_exists_supervised_only as an achievement. It is now gated on the "
                "closure gate actually being met. See declarations.K11_clause5_correction.",
                "ISSUE 2: results/phase2_keys_blockperm.json is now embedded "
                "(ordering_attack_blockperm) and cited from inside the verdict block, so the "
                "ordering half can no longer travel alone.",
                "ISSUE 3: closure_power_by_angular_jitter added -- the closure power curve the "
                "registered battery never had. The negative's reach is now stated in degrees.",
                "ISSUE 4: template_lineage_sensitivity added -- the verified template "
                "(colors) is itself 0/78 on closure, so this pipeline's end-to-end ring "
                "sensitivity is unestablished and the verdict block now says so.",
                "ISSUE 5: prereg_deviations added, enumerating 5(a)-(d) with direction of "
                "effect.",
            ],
            unchanged="every row statistic, every null, the synthetic battery, the "
                      "tokenisation audit, the jackknife, the exploratory blocks and the "
                      "multiplicity tables are recomputed from the same fixed seeds. This is "
                      "not asserted -- it is MEASURED, in row_identity_check below.",
        ),
    )
    out["verdict_counts"] = {
        ro: {name: [r["layer"] for r in rows if r["readout"] == ro and r["verdict"] == name]
             for name in ("recovered_fifths", "recovered_chromatic", "closed_ring_unordered",
                          "ordered_not_closed", "clean_negative", "degenerate")}
        for ro in READOUTS}
    out["verdict_counts_secondary"] = {
        ro: {name: [r["layer"] for r in rows
                    if r["readout"] == ro and r["verdict_secondary_plane_search"] == name]
             for name in ("closed_ring_widened_search", "closed_ring_unordered",
                          "ordered_not_closed", "clean_negative", "degenerate")}
        for ro in READOUTS}
    # K18: measure the claim regeneration_of.unchanged makes, and write the measurement in.
    # Called HERE, after verdict_counts and verdict_counts_secondary have been assigned, so
    # every comparable block is actually present to compare.
    out["regeneration_of"]["row_identity_check"] = row_identity_check(out)

    bp = out["ordering_attack_blockperm"]
    tl = out["template_lineage_sensitivity"]
    reach = [(v["jitter_deg"], v["point_cloud"]["power_uniform_at_05"],
              v["point_cloud"]["power_vs_background"]
              .get("matched_noun_background_mean_p05", {}).get("power"))
             for v in cpc["rows"].values()]
    reach.sort()
    reach_uniform = [j for j, pw, _ in reach if pw is not None and pw >= 0.80]
    reach_bg = [j for j, _, pw in reach if pw is not None and pw >= 0.80]

    out["verdict"] = dict(
        ontology="musical_keys",

        # -- the one-word headline, added 2026-08-21 so no reader has to assemble it --
        pitch_class_geometry=pitch_class_geometry_verdict,
        headline=("CLEAN NEGATIVE on pitch-class geometry. No closed 12-cycle was recovered "
                  "anywhere in 26 layers x 3 readouts: 0/78 cells close, the best "
                  f"p_closed_uniform over the whole sweep is "
                  f"{min(r['conventions']['A']['p_closed_uniform'] for r in live):.4f}, and "
                  "every one of the three closure families has 0 BH survivors. The "
                  "fifths-vs-chromatic discrimination -- the falsifiable headline this track "
                  "was built for -- was NOT made, in either direction."),

        supervised_arm_verdict_prereg_3_1=onto,
        supervised_arm_note=("prereg 3.1's own table calls this verdict 'explicitly NOT a "
                             "circle-of-fifths claim'. It must be read together with "
                             "ordering_attack_blockperm below, which is what the ordering "
                             "signal actually is."),
        cross_arm_verdict_prereg_4_3=arm_verdict,
        cross_arm_note=("prereg 4.3's structure_exists_supervised_only is defined as 'the "
                        "structure appears under templated supervision'. The predicted "
                        "structure is a closed 12-cycle and it did not appear, so that verdict "
                        "is NOT emitted. See declarations.K11_clause5_correction for the "
                        "superseded field and why it was wrong."),
        blind_arm_verdict="untestable",
        blind_arm_note="registered untestable by prereg 3.1/5.1/6 BEFORE any placement was "
                       "examined; prereg 7 forbids reporting a blind-arm negative for keys",
        fifths_vs_chromatic_discrimination=(
            "not made" if onto not in ("recovered_fifths", "recovered_chromatic")
            else ("fifths" if onto == "recovered_fifths" else "chromatic")),

        n_suffix_cells_closed_and_bh=len(closed_bh),
        n_suffix_cells_recovered_and_bh=len(recovered_bh),
        n_suffix_cells_ordered_and_bh=len(ordered_bh),

        # -- ISSUE 2 fix: the attack travels INSIDE the verdict block --
        ordering_signal_is_token_class_not_pitch_class=dict(
            attack_artifact=bp.get("path"),
            attack_present=bp.get("present"),
            fifths_ordering_bh_survivors_registered_uniform_null=bp.get(
                "ordering_fifths_bh_uniform"),
            fifths_ordering_bh_survivors_within_block_null=bp.get(
                "ordering_fifths_bh_within_block"),
            fifths_ordering_raw_below_05_uniform=bp.get(
                "ordering_fifths_raw_below_05_uniform"),
            fifths_ordering_raw_below_05_within_block=bp.get(
                "ordering_fifths_raw_below_05_within_block"),
            margin_cells_separating_uniform=bp.get("margin_separating_uniform_null"),
            margin_cells_separating_within_block=bp.get("margin_separating_within_block_null"),
            reading="the five flats occupy five CONSECUTIVE slots in fifths order and are "
                    "interleaved in chromatic order, so a cloud that merely separates the two "
                    "lexical classes into two angular blobs already agrees with fifths and "
                    "disagrees with chromatic, with no pitch-class content. Under a null that "
                    "holds those two classes fixed, the fifths ordering count collapses. The "
                    "supervised ordered_not_closed above is therefore token class, not pitch "
                    "class.",
            binding_rule=bp.get("binding_rule")),

        # -- ISSUE 3 fix: the negative's reach, stated in degrees --
        power_envelope=dict(
            curve="closure_power_by_angular_jitter",
            statistic="P(closure detects a planted 12-ring) as a function of Gaussian jitter "
                      "on the ring's angles, at the loudest registered ambient rung, through "
                      "the identical pca_2d -> fit_circle -> circularity -> p_closed path",
            largest_jitter_deg_with_power_ge_80pct_uniform_null=(
                max(reach_uniform) if reach_uniform else None),
            largest_jitter_deg_with_power_ge_80pct_matched_noun_background=(
                max(reach_bg) if reach_bg else None),
            best_gap_ratio_observed_in_sweep=cpc["observed_reference"][
                "best_gap_ratio_anywhere_in_sweep"],
            planted_ring_looks_this_bad_at_jitter_deg=cpc["observed_reference"][
                "planted_ring_matches_observed_best_at_jitter_deg"],
            what_the_negative_excludes="a pitch-class ring whose angular fidelity is BETTER "
                                       "than roughly the jitter rungs above",
            what_the_negative_does_NOT_exclude="rings generally. A ring with worse angular "
                                               "fidelity than that would not have been "
                                               "detected by this sweep, and this result says "
                                               "nothing about one.",
            why_the_registered_ladder_was_not_enough="prereg 4.2's isotropic ambient ladder "
                                                     "is in-plane-noiseless in 1152-d; the "
                                                     "planted ring stayed near-perfect "
                                                     "(gap_ratio 1.0686) at its loudest rung, "
                                                     "so it measured essentially nothing "
                                                     "about angular robustness"),

        # -- ISSUE 4 fix: the scope limit that changes how this arm should be read --
        end_to_end_sensitivity_unestablished=dict(
            template_artifact=tl.get("path"),
            template_present=tl.get("present"),
            template_cells_closing_at_05=tl.get("n_cells_closing_at_05"),
            template_gap_ratio_range=[tl.get("gap_ratio_min"), tl.get("gap_ratio_max")],
            template_min_p_closed_uniform=tl.get("p_closed_uniform_min"),
            keys_gap_ratio_range=[best_gap_observed,
                                  max(float(r["conventions"]["A"]["gap_ratio"]) for r in live)],
            keys_min_p_closed_uniform=float(
                min(r["conventions"]["A"]["p_closed_uniform"] for r in live)),
            finding=tl.get("finding"),
            consequence=tl.get("consequence"),
            not_a_claim=tl.get("not_a_claim")),

        rule="K11 clauses 1-4 frozen before the run; clause 5 corrected 2026-08-21 "
             "post-verification, superseded text quoted in "
             "declarations.K11_clause5_correction",
        reporting_rule="Any downstream summary that quotes the supervised verdict MUST carry "
                       "results/phase2_keys_blockperm.json beside it, and MUST carry the "
                       "power_envelope reach and end_to_end_sensitivity_unestablished blocks "
                       "beside the word 'negative'.")
    out["runtime_sec"] = float(time.time() - t0)

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=2, default=jsonable))

    # ---- summary -----------------------------------------------------------
    print("\n" + "=" * 92)
    print("VERDICT SUMMARY -- K10, PRIMARY pre-declared statistic")
    for ro in READOUTS:
        for name in ("recovered_fifths", "recovered_chromatic", "closed_ring_unordered",
                     "ordered_not_closed", "clean_negative", "degenerate"):
            ls = out["verdict_counts"][ro][name]
            if ls:
                print(f"  {ro:>8} {name:>22}: {len(ls):2d} layers {ls}")
    print("\nVERDICT SUMMARY -- K8 widened plane search (provenance-stated addition)")
    for ro in READOUTS:
        for name in ("closed_ring_widened_search", "closed_ring_unordered",
                     "ordered_not_closed", "clean_negative", "degenerate"):
            ls = out["verdict_counts_secondary"][ro][name]
            if ls:
                print(f"  {ro:>8} {name:>26}: {len(ls):2d} layers {ls}")
    print(f"\nK14(c) margin win tally (counts only):")
    for ro in READOUTS:
        w = win_tally[ro]
        print(f"  {ro:>8}: fifths ahead {w['margin_positive_fifths_ahead']:2d}/{w['n_cells']}, "
              f"chromatic ahead {w['margin_negative_chromatic_ahead']:2d}/{w['n_cells']}, "
              f"separating {w['n_margin_separates']:2d}, "
              f"mean |margin| {w['mean_abs_margin']:.3f}")
    print(f"\nK14(b) flats-blob check: all five flats contiguous in "
          f"{flats_agg['all']['n_all_five_flats_contiguous']}/{flats_agg['all']['n_cells']} "
          f"live cells (exact reference {flats_agg['exact_reference_probability']:.5f}, "
          f"expected by chance {flats_agg['expected_by_chance']:.2f}); "
          f"mean longest flat run {flats_agg['all']['longest_flat_run_mean']:.2f} of 5")
    for ro in READOUTS:
        f = flats_agg[ro]
        print(f"    {ro:>8}: {f['n_all_five_flats_contiguous']:2d}/{f['n_cells']:2d} contiguous, "
              f"mean run {f['longest_flat_run_mean']:.2f}")
    print(f"\nK13 MULTIPLICITY over {n_cells} live cells "
          f"(expected by chance at alpha=0.05: {0.05 * n_cells:.1f})")
    for key in ("ordering_fifths", "ordering_chromatic", "margin_two_sided", "closure_uniform",
                "closure_noun_background", "closure_letter_background",
                "closure_noun_background_secondary", "ordering_fifths_secondary"):
        m = mult[key]
        print(f"  {key:34s} raw p<0.05: {m['n_raw_below_05']:2d}   "
              f"survives BH-FDR: {m['n_surviving_bh_fdr']:2d}   "
              f"[p floor {m['p_value_floor']:.2e}, BH rank-1 {m['bh_rank1_threshold']:.2e}, "
              f"isolated cell reachable: {m['isolated_cell_can_survive_bh']}]"
              + (f"   {[(s['readout'], s['layer']) for s in m['surviving']]}"
                 if m["surviving"] else ""))
    print(f"\nONTOLOGY VERDICT (K11 clauses 1-4 frozen; clause 5 corrected 2026-08-21)")
    print(f"  HEADLINE                   : pitch_class_geometry = "
          f"{out['verdict']['pitch_class_geometry']}")
    print(f"  supervised arm (prereg 3.1): {onto}")
    print(f"  cross-arm      (prereg 4.3): {arm_verdict}")
    print(f"  blind arm                  : untestable (prereg 3.1/5.1/6)")
    print(f"  fifths-vs-chromatic        : "
          f"{out['verdict']['fifths_vs_chromatic_discrimination']}")

    v = out["verdict"]
    o = v["ordering_signal_is_token_class_not_pitch_class"]
    print(f"\nISSUE-2 FIX -- the ordering attack now travels with the verdict")
    print(f"  {o['attack_artifact']}  present={o['attack_present']}")
    print(f"  fifths ordering BH survivors: registered uniform null "
          f"{o['fifths_ordering_bh_survivors_registered_uniform_null']} -> within-block null "
          f"{o['fifths_ordering_bh_survivors_within_block_null']}")
    print(f"  margin cells separating     : uniform {o['margin_cells_separating_uniform']} -> "
          f"within-block {o['margin_cells_separating_within_block']}")

    print(f"\nISSUE-3 FIX -- K16 closure power curve (the reach of this negative)")
    print(f"  {'jitter':>7}  {'P(p_unif<.05) angles':>21}  {'P(p_unif<.05) cloud':>20}  "
          f"{'P(beat noun bg) cloud':>22}  {'median gap_ratio':>17}")
    for key in cpc["rows"]:
        rw = cpc["rows"][key]
        pcv, aov = rw["point_cloud"], rw["angles_only"]
        bgp = pcv["power_vs_background"].get("matched_noun_background_mean_p05", {})
        print(f"  {rw['jitter_deg']:6.0f}d  {aov['power_uniform_at_05']:21.3f}  "
              f"{pcv['power_uniform_at_05']:20.3f}  {bgp.get('power', float('nan')):22.3f}  "
              f"{pcv['gap_ratio_median']:17.4f}")
    pe = v["power_envelope"]
    print(f"  best gap_ratio anywhere in the sweep: "
          f"{pe['best_gap_ratio_observed_in_sweep']:.4f}; a planted ring looks that bad at "
          f"about {pe['planted_ring_looks_this_bad_at_jitter_deg']} deg of angular jitter")
    print(f"  EXCLUDES a ring with angular fidelity better than about "
          f"{pe['largest_jitter_deg_with_power_ge_80pct_uniform_null']}-"
          f"{pe['largest_jitter_deg_with_power_ge_80pct_matched_noun_background']} deg. "
          f"Does NOT exclude rings generally.")

    e2e = v["end_to_end_sensitivity_unestablished"]
    print(f"\nISSUE-4 FIX -- end-to-end sensitivity of this template family is UNESTABLISHED")
    print(f"  colours (the verified template, prereg 4.2): "
          f"{e2e['template_cells_closing_at_05']}/78 close, gap_ratio "
          f"{e2e['template_gap_ratio_range'][0]:.4f}-{e2e['template_gap_ratio_range'][1]:.4f}, "
          f"min p {e2e['template_min_p_closed_uniform']:.5f}")
    print(f"  keys   (this sweep)                       : "
          f"{v['n_suffix_cells_closed_and_bh']}/78 close and survive BH, gap_ratio "
          f"{e2e['keys_gap_ratio_range'][0]:.4f}-{e2e['keys_gap_ratio_range'][1]:.4f}, "
          f"min p {e2e['keys_min_p_closed_uniform']:.5f}")
    print(f"  -> both tracks sit on the same closure floor; no end-to-end demonstration that "
          f"templates -> model -> readout -> closure recovers a ring that genuinely exists")

    print(f"\nISSUE-5 FIX -- stated prereg deviations: "
          f"{[dv['id'] for dv in out['prereg_deviations']]} (all direction-conservative)")

    ric = out["regeneration_of"]["row_identity_check"]
    print(f"\nK18 ROW IDENTITY vs the superseded 05:43:05 run ({ric.get('checked_against')})")
    if ric.get("present"):
        print(f"  identical: {ric['identical']}   values compared: {ric['n_values_compared']}   "
              f"max |delta|: {ric['max_abs_delta']:.3e}")
        for bk, bv in ric["per_block"].items():
            if "n_values_compared" in bv:
                print(f"    {bk:26s} n={bv['n_values_compared']:6d}  differing="
                      f"{bv['n_differing']}  max|d|={bv['max_abs_delta']:.3e}")
            else:
                print(f"    {bk:26s} NOT COMPARED: {bv.get('note')}")
        print(f"  -> {ric['interpretation']}")
    else:
        print(f"  NOT CHECKED: {ric.get('warning')}")
    print(f"\nruntime {time.time() - t0:.1f}s -> wrote {a.out}")


if __name__ == "__main__":
    main()
