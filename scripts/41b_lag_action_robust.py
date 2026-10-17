"""Phase 3, Stage 4b -- COORDINATE-ROBUST LAG-CONDITIONED ACTION GEOMETRY.

Pre-registered: docs/phase3_addendum_H_2026-08-22.md (FROZEN 2026-08-22, before this
script was written), which is scripts/41_lag_action.py -- i.e. ADDENDUM F of
docs/phase3_prereg_addenda_2026-08-21.md, which itself incorporates
docs/phase3_prereg_2026-08-21.md sections 0/1/2.3 -- plus ONE added label-free
preprocessing rule (the outlier-coordinate drop) and Addendum H's own decision rule.
Funded by scripts/36 PASS (prereg section 6: "41 runs iff 36 PASS";
results/phase3_lag_fuse.json gate = PASS, centred median 0.93334).
Output: results/phase3_lag_action_robust.json (+ results/phase3_lag_action_robust_deltas.npz).

RELATION TO scripts/41_lag_action.py. This file is a COPY of scripts/41 (nothing is
imported from it, and scripts/41 is not touched) with exactly five changes:
  H1. LABEL-FREE OUTLIER-COORDINATE DROP. After A_even / A_odd are projected through
      w_dec, the pool-wide split-half residual (A_even - A_odd)/sqrt(2) over the INCLUDED
      types gives a per-coordinate sd; shares = sd^2 / sum(sd^2); every coordinate with
      share > 0.25 is DROPPED. The rule is computed PER ARM, uses no labels, and the
      dropped set is recorded. The drop is applied to delta(+1), delta(+2), the
      differenced arm, the planted-ring noise model, the supervised anchor arms, all four
      nulls and the blind arm. A WHITENED arm (every coordinate divided by its pool-wide
      noise sd, no drop) is CO-REPORTED for the supervised strict-12 statistics only.
  H2. POSITIVE-CONTROL GATE. The planted ring at the measured POST-DROP noise must reach
      >= 80% detection at <= 10 deg jitter at L6 and L7; otherwise the layer's verdict is
      "instrument not validated". Everything is still emitted -- the gate does not stop
      the run. (Positive control (a), the plain type-mean anchor, remains a hard R2 STOP,
      inherited unchanged from scripts/41.)
  H3. DECISION RULE. ACTION RING iff delta(+1) closes AND the differenced arm closes AND
      all four F.4 nulls fail to close in >= 95% of draws AND delta(+2) does NOT close;
      "persistent-context carry" if delta(+2) also closes (or if the ring is present in
      delta(+1) and absent in the differenced arm, per F.6). Closure here is the FOUR-term
      rule p_closed < 0.05 AND p_bg < 0.05 AND rho >= 0.95 AND block-permutation p < 0.05.
      Ordinal (one..ten) and weekday control sets run through the identical arms and must
      not close; if one does, the layer is INVALID.
  H4. BLIND ARM on the dropped-coordinate matrices, with the full passer audit (LOO,
      frequency-pseudo, frequency nuisance, token-class composition) reported for every
      BH passer; a passer that fails LOO is marked not_reportable (verify41 precedent).
  H5. The declaration stub carries the Addendum H text VERBATIM, including its exposure
      paragraph, and is written before any cache row is read; declared_at is carried
      forward into the result.

KNOWN DISCREPANCY IN ADDENDUM H's OWN PROVENANCE PARAGRAPH, stated here because it will
otherwise look like a bug in this file. Addendum H's exposure paragraph attributes
p_closed 0.0037 / p_bg 0.0040 / rho 1.000 (lag+1) and 0.0015 / 0.0020 / 0.993 (diff) to
"that coordinate dropped". Those numbers are, in verify41's own labelfree_l6.log, the
drop_top5_LF and drop_top2_LF variants. The FROZEN RULE in the same document drops only
coordinates whose sd share exceeds 25%, which at L6 is exactly {1038}; under that rule the
same log reads lag1 p_closed 0.28584 / p_bg 0.03194 / rho 0.9930 and diff 0.04130 /
0.00399 / rho 0.7203. This script implements the FROZEN RULE, not the quoted numbers.
Widening the drop to the top-2 or top-5 to reproduce them would be exactly the post-hoc
move the pre-registration exists to prevent. The discrepancy is recorded, direction-
neutral, in declared_rules.known_discrepancy_in_addendum_H_provenance.

THE OBJECT. For a token type t, the identity-removed context-effect ("action") vector at
lag d is

    delta_t(+d)  =  mean over admissible pairs (i, i+d) with tok(i) of type t of
                    [ code(row i+d)  -  TM[ type(row i+d) ] ]        ->  W_dec  ->  1152-d

i.e. the average perturbation type t imposes on the representation d positions later, with
the successor's own identity contribution removed. delta(0) is identically zero (the first
critic repair), so the instrument's positive control is NOT "the Delta=0 arm"; it is
(a) a plain type-mean anchor through the same accumulator code path and (b) a planted ring
in action space at the measured split-half noise level. Both are implemented below.

WHAT IS COPIED, AND FROM WHERE (so the blind arm is byte-identical in METHOD to the run
that produced results/token_type_geometry.json):
  * scripts/36_lag_reliability_fuse.py -- the single streaming pass that accumulates TM and
    the parity-split successor sums together, the pair-admissibility bookkeeping, the
    C @ TM identity removal, project(), rowwise_cosine(), strict12(), and the centred
    split-half reliability statistic. EXTENDED here from one lag to two.
  * scripts/22_token_type_geometry.py lines 258-420 and 561 -- build_pool, cosine_matrix,
    leiden, sweep_and_select, refine, gap_null, gap_p, screen, background_for_size,
    p_bg_of, score_ordering, shuffled_label_control, cid_seed, member_hash. COPIED
    VERBATIM up to the constants they close over. They live in a script, not in
    src/gfdisc, so they cannot be imported; copying rather than re-deriving is what keeps
    the blind arm identical to the frozen one.
  * scripts/29_keys_blockperm.py -- block_permutations, _rank_of, _rank_from_angles,
    _sym_basis (the block-permutation ordering null mandated by prereg R4).
  * scripts/34_spectral_search.py -- type_string, class_features, block_perm_spearman and
    the "blocks = leading_space x capitalized" definition used by the passer audits, so
    the audits are consistent across Phase 3.
Everything imported from src/gfdisc (fits.pca_2d, fit_circle, circularity, angles,
degenerate, cyclic_order_score; nulls.benjamini_hochberg, frequency_matched_resample) is
the same code path the frozen results used.

PROVENANCE OF THE OUTLIER RULE (exposure, stated up front). The rule was written after
verify41's post-hoc diagnostics
/private/tmp/.../scratchpad/verify41/outlier_l6.py and labelfree_l6.py, whose numbers are
reproduced verbatim inside Addendum H's own exposure paragraph and are echoed in the
declaration stub below. The rule itself is label-free and was FROZEN in Addendum H before
this script was written; the threshold (25% of the split-half noise variance), the arms it
applies to, the positive-control gate and the decision rule are all fixed by that
document. Nothing below was chosen after a number produced by this file.

MEMORY, one layer at a time. Per layer the float64 accumulators are TM [3795, 16384]
(497 MB), S(+1) even|odd stacked (994 MB) and S(+2) even|odd stacked (994 MB) = 2.49 GB,
plus C(+1), C(+2) at 2 x 3795 x 3795 int64 = 115 MB per parity (461 MB total) and the
streaming chunk buffers (~0.3 GB). Peak ~3.4 GB, under the 6 GB budget. TM is freed
immediately after the identity removal; only its 1152-d projection TMP (17 MB) is kept.

NO FORWARD PASSES. Nothing here loads the model; the only model artifact read is w_dec.

PROVENANCE OF THE CALIBRATION WIRING, stated up front rather than left to mtimes (the
project's own pattern -- scripts/29's corrected header, scripts/28's closure_power_curve).
The per-arm planted-ring curve, its two co-reported companion curves, the noise
diagnostics, and the "NOT CLOSED (negative UNCALIBRATED)" annotation were written AFTER a
truncated-prefix smoke (400k of 4,002,480 rows, to a scratch path) showed the frozen
primary curve at zero detection. All three are direction-fixed: they can only QUALIFY or
WEAKEN a negative, never strengthen a positive, create a flag, or move a frozen threshold.
The verdict-bearing curve is still the frozen primary one, and the R2 STOP gate is and
remains positive control (a). See declared_rules.provenance_of_calibration_wiring.

Cache facts asserted, not assumed (prereg section 0): 4,002,480 rows = 15,696 sequences x
255 tokens, BOS dropped at cache build, so a row index is a DOCUMENT-RELATIVE index.
Nothing in this file is reported as a claim about absolute sequence position.
"""
import argparse, gc, hashlib, json, os, sys, time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_HOME", str(REPO / "cache" / "hf"))
sys.path.insert(0, str(REPO / "src"))
from gfdisc.saes import load_sae, get_decoder
from gfdisc.fits import (pca_2d, fit_circle, circularity, angles,
                         cyclic_order_score, degenerate)
from gfdisc.nulls import benjamini_hochberg, frequency_matched_resample
from gfdisc.tokens import MONTHS, DAYS

# ===========================================================================
# DECLARED RULES -- frozen by Addendum F. Nothing below is chosen after a result.
# ===========================================================================
COUNT_FLOOR = 100            # build_pool main pool (scripts/22 L211) -> 3,795 types
D_MODEL = 1152
D_SAE = 16384
SEQ_LEN = 255                # cached tokens per sequence, BOS already dropped
N_ROWS_EXPECTED = 4_002_480
N_POOL_EXPECTED = 3_795
CHUNK = 200_000              # streaming chunk, rows (scripts/36)
LAGS = (1, 2)                # delta grid: +1 primary, +2 exploratory, and their difference
ARMS = ("lag1", "lag2", "diff")
ARM_LAGS = {"lag1": (1,), "lag2": (2,), "diff": (1, 2)}

MIN_PAIRS_PER_HALF = 20      # scripts/36: a type below this in EITHER half leaves the median
GATE_THRESHOLD = 0.30        # Addendum F.2 reliability gate, centred split-half median
FUSE_L6_LAG1_CENTRED = 0.9333420819770755      # results/phase3_lag_fuse.json
FUSE_TOL = 1e-3
FUSE_L6_LAG2_VERIFY36 = 0.824                  # verify36's lag+2 number, printed not asserted

# --- screening / clustering, copied from scripts/22 ---
SCREEN_MIN, SCREEN_MAX = 5, 60
N_PERM_CLUSTER = 20000       # uniform-angle null for screened/scored clusters
N_PERM_BACKGROUND = 2000     # uniform-angle null inside the broad background sweep
N_BACKGROUND_DRAWS = 500     # random same-size subsets per represented size
N_PERM_ORDER = 20000         # permutation null inside cyclic_order_score / block-perm null
N_SHUFFLE_LABELS = 1000
N_PERM_LOO = 20000           # leave-one-out jackknife screen (scripts/34 survivor audit)
GRID_CENTRED = [round(0.10 + 0.05 * i, 2) for i in range(17)]   # 0.10 .. 0.90
EDGE_BUDGET = 2_000_000
LEIDEN_RESOLUTION = 1.0
LEIDEN_SEED = 0
REFINE_RESOLUTIONS = [2, 4, 8, 16, 32, 64]
FLAG_ALPHA = 0.05
BH_Q = 0.05

# --- nulls ---
N_NULL_COMPOSITION = 500     # successor-composition-matched resample
N_NULL_PREDSWAP = 200        # within-document predecessor-swap
N_NULL_FREQPSEUDO = 200      # frequency-pseudo-code battery
FREQ_PSEUDO_NEAREST = 5      # random tie-breaking among the 5 nearest corpus counts
FREQ_MATCH_TOL = 0.25        # nulls.frequency_matched_resample log10 tolerance
NULL_CLOSE_RATE_MAX = 0.05   # "fails to close in >= 95% of draws"
COMPOSITION_REJECT_ROUNDS = 50

# --- positive control (b): planted ring in action space ---
N_PLANTED_DRAWS = 200
PLANTED_JITTERS_DEG = (0, 5, 10, 15, 20, 30)
PLANTED_RADIUS_FRACTION = 0.5     # radius = 0.5 x median ||delta|| over included types
POWER_TARGET = 0.80

# --- verdict thresholds (Addendum F step 6 / avenue DECISION line) ---
ALPHA = 0.05
RHO_THRESHOLD = 0.95
CONFIRMATORY_LAYERS = (6, 7)
NULL_LAYERS_DEFAULT = "6,7"

# --- ADDENDUM H (frozen 2026-08-22): the one added label-free preprocessing rule ---
OUTLIER_SHARE_THRESHOLD = 0.25      # drop a coordinate iff sd^2 / sum(sd^2) > this
WHITEN_FLOOR = 1e-9                 # divisor floor for the co-reported whitened arm
# Addendum H's positive-control (R2) gate on the POST-DROP planted-ring curve.
PLANTED_GATE_JITTER_DEG = 10        # ">= 80% detection at <= 10 deg jitter"
PLANTED_GATE_DETECTION = 0.80
PLANTED_GATE_LAYERS = (6, 7)
# Addendum H's control sets. Both run through the IDENTICAL supervised arms and must not
# close; the ordinal set is prereg 3.1's number-word control, the weekday set is a real
# cyclic ontology whose order is known.
ORDINAL_WORDS = (" one", " two", " three", " four", " five", " six", " seven",
                 " eight", " nine", " ten")
CONTROL_SETS = ("ordinal_numwords", "weekdays")

# --- ADDENDUM H1 (erratum to H, frozen 2026-08-22): the declared sensitivity ladder ---
# Co-reported, NOT verdict-bearing. Coordinates are ranked by the SAME label-free
# pool-wide split-half noise share the H rule uses; the ladder simply reports what the
# supervised strict-12 statistics do at fixed drop depths instead of at the 25% threshold.
SENSITIVITY_DROP_DEPTHS = (1, 2, 5)
SENSITIVITY_RUNGS = ("drop_top1", "drop_top2", "drop_top5", "whitened")
SENSITIVITY_SEED_BASE = 900_000

# --- frozen exposure (prereg section 0). member_hash is over POOL INDICES. ---
MEMBERS_13 = [3353, 5142, 5261, 5365, 5505, 5656, 5729, 5857, 6027, 6198, 6243, 6384, 7250]
MEMBERS_14 = sorted(MEMBERS_13 + [30651])       # the L7 cluster: the 13-set plus 'April'
HASH_13 = "7d1d75a303d6"
HASH_14 = "6bb0e54cd127"
STRICT_12_EXPECTED = sorted(x for x in MEMBERS_13 if x != 5656)
# Positive control (a): the published token-type-mean numbers, per prereg section 0.
# layer -> (which frozen member set, expected gap_ratio, expected p_closed, expected hash)
ANCHOR_TABLE = {
    6: ("13-set", 1.6587577329752548, 0.0012499375031248437, HASH_13),
    7: ("14-set", 1.6204086266952076, 0.0011499425028748563, HASH_14),
    8: ("13-set", 3.6356427267561973, 0.7461626918654067, HASH_13),
}
ANCHOR_TOL_GAP_RATIO = 1e-4     # "to 4 dp on gap_ratio"

ADDENDUM_F_TEXT = """\
## Addendum F - frozen 2026-08-22 before any scripts/41 run: lag-conditioned action geometry

Funding: 36 PASS (centred split-half median 0.933; verify36: null 0.021; lag+2 0.824).
Design = docs/next_avenues avenue "lag-conditioned action geometry" with the five critic
repairs, plus a sixth from verify36:
 1. Positive control: (a) plain type-mean anchor (no identity removal) must reproduce the
    published L6/L7/L8 month numbers to 4 dp through the same accumulator code path; (b) a
    planted ring in action space at the split-half-measured noise level, jitter
    0/5/10/15/20/30 deg, 80%-power jitter reported - THIS validates the instrument.
 2. Centred split-half reliability gate (median >= 0.30) recomputed per layer 0-10; layers
    below the gate issue no verdict.
 3. Out-of-pool successors dropped (fraction reported); identity removal as S - C @ TM.
 4. Nulls: successor-composition-matched resample (same successor-type distribution, drawn
    from occurrences whose predecessor is not t); WITHIN-DOCUMENT predecessor-swap null
    (permute predecessor labels within sequence); frequency-pseudo-code battery;
    block-permutation for ordering; unit-norm arm.
 5. Blind arm PRIMARY: cluster all pool types' delta(+1) vectors with the byte-identical
    centred-cosine + Leiden machinery of scripts/22 (same tau grid, same size-5-60 screen,
    same matched empirical background, BH within layer x arm); supervised 12-month closure
    in action space is the anchor, co-reported.
 6. Lag differencing (verify36): co-report delta(+1) - delta(+2) (the immediate-predecessor
    increment) through the same blind and supervised arms; a month ring present in delta(+1)
    but absent in the differenced arm is reported as "persistent-context carry", not action.
Layers: 6 and 7 confirmatory, 0-10 exploratory under BH. Output results/phase3_lag_action.json.
"""

# The FROZEN text of docs/phase3_addendum_H_2026-08-22.md, verbatim (including its
# exposure paragraph). Written into the declaration stub before any cache row is read.
ADDENDUM_H_TEXT = """# Addendum H — frozen 2026-08-22 before any scripts/41b run: coordinate-robust lag action

Provenance (exposure, verify41 post-hoc, NOT pre-registered): at L6 the split-half noise of
the identity-removed successor deltas is concentrated in residual coordinate 1038 (91.6% of
per-coordinate sd share, lag+1; 90.3% diff arm); it is the argmax of pool-wide (label-free)
split-half noise at L6 and L7 and of the type-mean column sd (19.9%). With that coordinate
dropped, the planted-ring control reaches detection 1.000 at 0° and ~0.87 at 10°; the real
strict-12 delta(+1) ring closes (p_closed 0.0037, p_bg 0.0040, rho 1.000), the differenced
arm closes (0.0015 / 0.0020 / 0.993), delta(+2) never closes, number words never close
(p 0.90–0.95), weekdays never close. These numbers are KNOWN and are not a verdict.

41b design (= scripts/41 with one frozen, label-free preprocessing rule; everything else per
Addendum F):
- Outlier-coordinate rule: at each layer compute the pool-wide split-half residual
  (A_even − A_odd)/√2 over all included types in w_dec-projected space, per-coordinate sd;
  DROP every coordinate whose sd share exceeds 25% (expected: exactly 1038 at L6/L7; the
  rule may drop 0, 1 or more coordinates; the set is recorded). A co-reported whitened arm
  divides all coordinates by their pool-wide noise sd. No labels enter the rule.
- Positive control (R2 gate): planted ring at the measured (post-drop) noise must reach
  ≥ 80% detection at ≤ 10° jitter at L6 and L7, else "instrument not validated".
- Decision (supervised anchor arm, strict-12, per layer; L6/L7 confirmatory): ACTION RING iff
  delta(+1) closes (p_closed<0.05 ∧ p_bg<0.05 ∧ rho≥0.95 ∧ block-perm p<0.05) AND the
  differenced arm delta(+1)−delta(+2) closes (same rule) AND all four F.4 nulls fail to
  close in ≥95% of draws AND delta(+2) does NOT close (specificity; if delta(+2) also
  closes the reading is "persistent-context carry", per F.6). Ordinal control (one..ten) and
  weekdays through the identical arms must not close (invalidity trigger if they do).
- Blind arm as F.5 on the dropped-coordinate matrices, with the full passer audit (LOO,
  frequency-pseudo, token-class composition) reported for every BH passer — a passer that
  fails LOO is NOT reportable (verify41 precedent).
- Output results/phase3_lag_action_robust.json (+ deltas npz). Layers 0–10.
"""

ADDENDUM_H_DOC = "docs/phase3_addendum_H_2026-08-22.md"

# The FROZEN text of docs/phase3_addendum_H1_erratum_2026-08-22.md, verbatim. H1 corrects
# H's provenance paragraph and adds the co-reported sensitivity ladder; it does NOT change
# the frozen H rule or the H verdict.
ADDENDUM_H1_TEXT = """# Addendum H1 (erratum to H) — frozen 2026-08-22 before any scripts/41b run

Erratum: Addendum H's provenance paragraph misattributes verify41's numbers. The closure
results it quotes (lag+1 0.0037/0.0040/1.000; diff 0.0015/0.0020/0.993) come from
verify41's drop_top5 and drop_top2 label-free variants respectively, NOT from the
single-coordinate (>25% share) rule. Under the >25% rule (drops exactly {1038} at L6) the
same log reads lag+1 p_closed 0.286 / p_bg 0.032 / rho 0.993 and diff 0.041 / 0.004 /
rho 0.720. The planted-control figures (1.000 at 0°, ~0.87 at 10°) do correspond to the
single-coordinate drop. The frozen H rule is NOT changed.

Declared sensitivity ladder (co-reported, NOT verdict-bearing; primary verdict = H rule):
for the supervised strict-12 arm at every layer, report closure / p_bg / rho / block-perm
for delta(+1), delta(+2), diff under: drop-top-1 (= H rule when share > 25%), drop-top-2,
drop-top-5 (coordinates ranked by pool-wide split-half noise share, label-free), and
whitened. The write-up must state the drop-depth dependence explicitly; a ring that appears
only at deeper drops is reported as "drop-depth-dependent", not as an action ring.
"""

ADDENDUM_H1_DOC = "docs/phase3_addendum_H1_erratum_2026-08-22.md"


def addendum_h_integrity():
    """Non-fatal check that the embedded Addendum H text still matches the frozen file."""
    p = REPO / ADDENDUM_H_DOC
    out = dict(doc=ADDENDUM_H_DOC, embedded_sha256=hashlib.sha256(
        ADDENDUM_H_TEXT.encode("utf-8")).hexdigest(), doc_exists=bool(p.exists()))
    if p.exists():
        raw = p.read_text(encoding="utf-8")
        out.update(doc_sha256=hashlib.sha256(raw.encode("utf-8")).hexdigest(),
                   doc_mtime=time.strftime("%Y-%m-%dT%H:%M:%S",
                                           time.localtime(p.stat().st_mtime)),
                   embedded_matches_frozen_doc=bool(raw == ADDENDUM_H_TEXT))
    return out


def addendum_h1_integrity():
    """Non-fatal check that the embedded Addendum H1 erratum matches the frozen file."""
    p = REPO / ADDENDUM_H1_DOC
    out = dict(doc=ADDENDUM_H1_DOC, embedded_sha256=hashlib.sha256(
        ADDENDUM_H1_TEXT.encode("utf-8")).hexdigest(), doc_exists=bool(p.exists()))
    if p.exists():
        raw = p.read_text(encoding="utf-8")
        out.update(doc_sha256=hashlib.sha256(raw.encode("utf-8")).hexdigest(),
                   doc_mtime=time.strftime("%Y-%m-%dT%H:%M:%S",
                                           time.localtime(p.stat().st_mtime)),
                   embedded_matches_frozen_doc=bool(raw == ADDENDUM_H1_TEXT))
    return out


def declared_rules(args_view):
    return dict(
        spec=("docs/phase3_addendum_H_2026-08-22.md, FROZEN 2026-08-22 before any "
              "scripts/41b run -- = scripts/41_lag_action.py (Addendum F) plus ONE added "
              "label-free preprocessing rule and Addendum H's own decision rule"),
        addendum_H_text=ADDENDUM_H_TEXT,
        addendum_H_integrity=addendum_h_integrity(),
        addendum_H1_erratum_text=ADDENDUM_H1_TEXT,
        addendum_H1_integrity=addendum_h1_integrity(),
        addendum_F_text=ADDENDUM_F_TEXT,
        also_binding=("docs/phase3_prereg_addenda_2026-08-21.md ADDENDUM F (everything "
                      "Addendum H does not change), and docs/phase3_prereg_2026-08-21.md "
                      "sections 0 (exposure record), 1 (global rules R1-R6), 2.3 (the "
                      "fuse probe whose accumulator this extends), 6 (funding: 41 runs "
                      "iff 36 PASS -- results/phase3_lag_fuse.json gate = PASS)"),
        funding="results/phase3_lag_fuse.json gate PASS, centred median 0.9333420819770755",
        no_forward_passes=("no model is loaded and no forward pass is run; the only model "
                           "artifact read is w_dec per layer"),

        # ------------------------- ADDENDUM H, the added rule -------------------------
        outlier_coordinate_rule=(
            "THE ONE ADDED PREPROCESSING RULE (Addendum H), LABEL-FREE. At each layer, "
            "AFTER A_even and A_odd are projected through w_dec and PER ARM: the pool-wide "
            "split-half residual is (A_even - A_odd)/sqrt(2) over the INCLUDED types (the "
            f"same >= {MIN_PAIRS_PER_HALF}-pairs-in-both-halves inclusion mask Addendum F.2 "
            "already uses); its per-coordinate sd is sd_c = sqrt(mean over included types "
            "of residual_c^2); shares_c = sd_c^2 / sum_c sd_c^2; every coordinate with "
            f"share > {OUTLIER_SHARE_THRESHOLD} is DROPPED. The rule may drop 0, 1 or more "
            "coordinates (at most 3 can exceed a 25% variance share); the dropped set is "
            "recorded per layer and per arm, together with the top-5 shares. NO LABEL "
            "ENTERS THE RULE: the residual is taken over every included pool type, not "
            "over the month rows. Expected, from the frozen exposure paragraph: exactly "
            "coordinate 1038 at L6 and L7. DECLARED READING (the task text says 'per arm'): "
            "each arm computes the rule on ITS OWN halves and ITS OWN included mask, and "
            "each arm is then analysed inside its own kept subspace. APPLIED TO: delta(+1), "
            "delta(+2), the differenced arm delta(+1)-delta(+2), the planted-ring noise "
            "model and radius, the supervised anchor arms and their matched backgrounds, "
            "all four nulls (the two resampling nulls slice the recomputed delta rows to "
            "the kept coordinates before screening; the unit-norm arm drops FIRST and "
            "row-normalises second), and the blind arm"),
        whitened_arm=(
            "CO-REPORTED ONLY (Addendum H), never verdict-bearing: every coordinate of the "
            "full 1152-d delta matrix divided by its POOL-WIDE noise sd (the same sd_c the "
            f"drop rule computes, floored at {WHITEN_FLOOR}); NO coordinate is dropped in "
            "this arm. It is run through the identical supervised path for the STRICT-12 "
            "set only, with its own matched background. It appears in no conjunction, no "
            "gate and no BH family"),
        not_changed_by_the_drop=(
            "DECLARED, because it is the one place the rule could silently break a frozen "
            "assert. (1) Positive control (a), the plain type-mean anchor on TM @ w_dec, is "
            "computed on the FULL 1152 coordinates exactly as in scripts/41 -- it "
            "reproduces published token-type numbers and must not move. (2) The Addendum "
            "F.2 reliability gate and the L6 fuse reproduction assert "
            f"({FUSE_L6_LAG1_CENTRED}, tol {FUSE_TOL}) are computed on the FULL "
            "coordinates, because that is the statistic results/phase3_lag_fuse.json "
            "reports; dropping first would break the assert by construction. The POST-DROP "
            "reliability medians are co-reported as a diagnostic and decide nothing. (3) "
            "The inclusion mask itself is unchanged (it is a pair-count rule, not a "
            "geometry rule)"),
        positive_control_gate_H=(
            f"ADDENDUM H's R2 GATE. At layers {list(PLANTED_GATE_LAYERS)} the FROZEN PRIMARY "
            "planted-ring curve, computed at the MEASURED POST-DROP noise, must reach "
            f">= {PLANTED_GATE_DETECTION:.0%} detection at <= {PLANTED_GATE_JITTER_DEG} deg "
            "jitter; otherwise the layer's Addendum H verdict is 'instrument not "
            "validated'. THE GATE DOES NOT STOP THE RUN -- everything is still emitted, as "
            "Addendum H requires. DECLARED READING of 'reach >= 80% detection at <= 10 deg "
            "jitter': READING A (PRIMARY) = detection is still >= 80% AT 10 deg, i.e. the "
            "80%-power reach is at least 10 deg -- the instrument can detect a ring carrying "
            "up to 10 deg of angular jitter. READING B (co-reported) = detection reaches "
            "80% at SOME jitter <= 10 deg, which monotonicity makes equivalent to detection "
            ">= 80% at 0 deg. Both are recorded per layer and per arm; the layer gate uses "
            "READING A over ALL THREE arms, because all three enter the decision (lag1 and "
            "diff must close, lag2 supplies the specificity term); the lag1-and-diff-only "
            "variant is co-reported. Positive control (a) remains a HARD STOP (prereg R2), "
            "unchanged from scripts/41"),
        controls_H=(
            "ADDENDUM H invalidity triggers, run through the IDENTICAL supervised arms on "
            "the dropped-coordinate matrices. (1) ORDINAL CONTROL: the number words "
            f"{list(ORDINAL_WORDS)} resolved by the tokenizer to single ids that are in the "
            "pool (verify41 measured all ten in pool; the leading-space digits ' 0'..' 9' "
            "are NOT single tokens in this vocabulary and are recorded as unresolved), "
            "label_index = the canonical 0..9 index of each resolved word, full_k = 10. "
            "(2) WEEKDAY CONTROL: for each of Monday..Sunday the HIGHEST-COUNT in-pool id "
            "in results/token_map.json 'days' (the strict12 rule applied to weekdays), "
            "label_index = 0..6, full_k = 7. Each control set is screened, given its own "
            "count-and-frequency-matched background, ordered by cyclic_order_score and by "
            "the block-permutation ordering null, exactly like the strict-12 anchor. "
            "TRIGGER: a control set that CLOSES under the identical four-term rule "
            f"(p_closed < {ALPHA} AND p_bg < {ALPHA} AND rho >= {RHO_THRESHOLD} AND "
            f"block-perm p < {ALPHA}) in the lag1 or diff arm makes the layer INVALID. "
            "Controls are also run and reported in the lag2 arm, and the two-term "
            "(p_closed AND p_bg) sub-conjunction is co-reported so near-misses are visible. "
            "R3 note: these are LABELLED control sets attached to a supervised ANCHOR arm, "
            "the same status Addendum F gives the month sets; no blind statistic uses them"),
        decision_H=(
            "ADDENDUM H's DECISION, supervised anchor arm, strict-12, PER LAYER, on the "
            "DROPPED-COORDINATE matrices; layers 6 and 7 confirmatory. An arm CLOSES iff "
            f"p_closed < {ALPHA} AND p_bg < {ALPHA} AND rho >= {RHO_THRESHOLD} AND the "
            f"block-permutation ordering p < {ALPHA} (Addendum H folds the R4 block-perm "
            "term INTO the closure rule; Addendum F only co-reported it). ACTION RING iff "
            "delta(+1) closes AND the differenced arm closes AND all four F.4 nulls fail to "
            f"close in >= {int((1 - NULL_CLOSE_RATE_MAX) * 100)}% of draws AND delta(+2) "
            "does NOT close. If delta(+2) ALSO closes the reading is PERSISTENT-CONTEXT "
            "CARRY (Addendum H, per F.6); F.6's own case -- a ring present in delta(+1) and "
            "absent in the differenced arm -- is also read as PERSISTENT-CONTEXT CARRY and "
            "is distinguished by a sub-reason. DECLARED READING of 'all four nulls': "
            "carried verbatim from scripts/41 -- nulls (i) composition, (ii) predecessor-"
            f"swap and (iii) frequency-pseudo carry the < {NULL_CLOSE_RATE_MAX} closure-rate "
            "requirement; null (iv), the unit-norm arm, is not draw-based and enters as the "
            "robustness requirement 'the months STILL close after row normalisation', which "
            "is the only coherent reading. The null conjunction is required on BOTH closing "
            "arms (lag1 AND diff) and is recorded per arm so the lag1-only reading is "
            "reconstructable. PRECEDENCE, declared: (1) reliability gate -> NO VERDICT; "
            "(2) at layers 6 and 7, the Addendum H planted gate -> 'instrument not "
            "validated'; (3) invalidity triggers -> INVALID; (4) nulls (i)/(ii) not run at "
            "an exploratory layer -> 'closure_flagged_nulls_not_run', never ACTION RING; "
            f"(5) the conjunction above. Exploratory layers stay under BH q={BH_Q} within "
            "arm, exactly as in Addendum F"),
        sensitivity_ladder_H1=(
            "ADDENDUM H1 (erratum to H, frozen 2026-08-22), CO-REPORTED AND EXPLICITLY NOT "
            "VERDICT-BEARING. For the SUPERVISED STRICT-12 arm at EVERY layer and for all "
            "three delta arms (lag1, lag2, diff), the four closure statistics -- p_closed, "
            "p_bg, rho and the block-permutation ordering p -- are reported at four fixed "
            f"preprocessing depths: {list(SENSITIVITY_RUNGS)}. Coordinates are ranked by "
            "the SAME label-free pool-wide split-half noise share the H rule uses "
            "((A_even - A_odd)/sqrt(2) over included types, per-coordinate sd, "
            "share = sd^2 / sum sd^2), so no rung introduces a new selection principle and "
            "no label enters any rung. drop_top1 removes the single highest-share "
            "coordinate (it EQUALS the frozen H rule whenever that rule drops exactly one "
            "coordinate, which is recorded per rung as equals_H_rule); drop_top2 and "
            "drop_top5 remove the two and five highest-share coordinates; whitened divides "
            "every coordinate by its pool-wide noise sd and drops none. Each rung runs "
            "through the IDENTICAL supervised_cell path with its own count-and-frequency-"
            "matched background, its own cyclic_order_score and its own block-permutation "
            "null. The ladder is computed for every arm at every layer INDEPENDENTLY of the "
            "Addendum F.2 reliability gate, because H1 says 'at every layer'; an arm below "
            "the gate still issues no verdict and its ladder is diagnostic like everything "
            "else at that arm. READING, declared by H1: the write-up must state the "
            "drop-depth dependence explicitly, and A RING THAT APPEARS ONLY AT DEEPER DROPS "
            "IS REPORTED AS 'drop-depth-dependent', NOT AS AN ACTION RING. The script "
            "computes that annotation mechanically (drop_depth_dependent = the frozen H "
            "rule does not close for an arm while a deeper rung does) and attaches it to "
            "the layer decision; the PRIMARY VERDICT REMAINS THE H RULE and is unchanged by "
            "anything in the ladder"),
        known_discrepancy_in_addendum_H_provenance=(
            "DIRECTION-NEUTRAL FACT, now also CORRECTED IN THE PRE-REGISTRATION ITSELF by "
            "docs/phase3_addendum_H1_erratum_2026-08-22.md (frozen 2026-08-22 before any "
            "scripts/41b run, embedded verbatim above as addendum_H1_erratum_text), which "
            "restates it and adds the co-reported sensitivity ladder while leaving the "
            "frozen H rule unchanged. Recorded here so it is not mistaken for a defect in "
            "this script. Addendum H's exposure paragraph attributes 'p_closed 0.0037, p_bg "
            "0.0040, rho 1.000' (lag+1) and '0.0015 / 0.0020 / 0.993' (differenced) to the "
            "state 'with that coordinate dropped'. In verify41's own labelfree_l6.log those "
            "two triples are the drop_top5_LF and drop_top2_LF rows. Under the FROZEN RULE "
            f"of the same document (share > {OUTLIER_SHARE_THRESHOLD}, which at L6 selects "
            "exactly coordinate 1038) the same log reads lag1 drop_outlier_LF p_closed "
            "0.28584 / p_bg 0.03194 / rho 0.99301 and diff drop_outlier_LF 0.04130 / "
            "0.00399 / rho 0.72028. THIS SCRIPT IMPLEMENTS THE FROZEN RULE. The threshold "
            "was not widened to top-2 or top-5 to reproduce the quoted numbers, because "
            "choosing a preprocessing depth by the p-value it produces is precisely the "
            "post-hoc move the pre-registration exists to prevent. The quoted planted-ring "
            "figures ('detection 1.000 at 0 deg and ~0.87 at 10 deg') DO correspond to the "
            "single-coordinate drop (verify41 outlier_l6.log, variant b_outlier_zeroed), so "
            "the Addendum H positive-control gate is unaffected. H1 records exactly this "
            "and confirms 'The frozen H rule is NOT changed'"),

        population=(f"the {N_POOL_EXPECTED}-type pool: token types with count >= "
                    f"{COUNT_FLOOR} over the {N_ROWS_EXPECTED} cache rows (build_pool main "
                    "ids, scripts/22_token_type_geometry.py L211)"),
        lags=("delta(+1) PRIMARY, delta(+2) exploratory, and the differenced arm "
              "delta(+1) - delta(+2) (Addendum F.6). Admissible pair (i, i+d) requires "
              "(i mod 255) < (255 - d), which never straddles a cached sequence: d=1 -> "
              "(i mod 255) < 254, d=2 -> (i mod 255) < 253. Equivalent successor-row form "
              "used in the streaming pass: row j contributes to lag d iff (j mod 255) >= d "
              "and lut[tok[j-d]] >= 0 and lut[tok[j]] >= 0"),
        out_of_pool_successors=("DROPPED (Addendum F.3). Pairs whose successor token is not "
                                "a pool type contribute to neither S nor C. The dropped "
                                "fraction is computed among pairs with a VALID PREDECESSOR "
                                "and is reported per lag (critic pre-measurement ~20.0%; "
                                "scripts/36 measured 0.18658 at lag +1). The successor-"
                                "composition null therefore matches only the RETAINED "
                                "composition, as declared in scripts/36"),
        split=("sequence parity of the PREDECESSOR row: parity = (i // 255) % 2; because an "
               "admissible pair never straddles a sequence this equals (j // 255) % 2"),
        type_means=("TM[t] = mean SAE code over ALL cache rows of type t, accumulated in "
                    "float64 in the SAME streaming pass as the successor sums and divided "
                    "by the full-corpus per-type count (equality with the streamed count is "
                    "asserted on a full run)"),
        identity_removal=("A(+d)_par = S(+d)_par - C(+d)_par @ TM in the 16384-d code "
                          "space, float64 (Addendum F.3 / critic repair 5). C(+d)_par[t, s] "
                          "is the count of admissible parity-par lag-d pairs from t to s, "
                          "so C @ TM is exactly the sum over those pairs of the successor's "
                          "own type mean"),
        normalization=("A(+d)_par[t] /= sum_s C(+d)_par[t, s]. The FULL-DATA arm is the "
                       "pair-count-weighted pool of the two parity halves, "
                       "A_full = (n_even * A_even + n_odd * A_odd) / (n_even + n_odd), "
                       "formed in the 1152-d projection (projection is linear, so this is "
                       "the same object as pooling the 16384-d sums)"),
        projection=("A cast to float32 and multiplied by w_dec ([16384, 1152]); b_dec is "
                    "NOT added -- it is a constant offset that PCA and the centred cosine "
                    "both remove exactly (identical to scripts/22, 34, 36)"),
        differenced_arm=("delta_diff = delta(+1) - delta(+2), formed per parity half so the "
                         "same split-half reliability statistic applies. Reading (Addendum "
                         "F.6): a month ring present in lag1 but absent in diff is "
                         "PERSISTENT-CONTEXT CARRY, not action"),

        reliability_gate=(f"Addendum F.2. Per layer and per arm: each half centred on its "
                          f"OWN pool mean over INCLUDED types, per-type cosine between the "
                          f"halves, gate statistic = MEDIAN over included types; an arm "
                          f"with median < {GATE_THRESHOLD} issues NO VERDICT at that layer. "
                          f"Inclusion: >= {MIN_PAIRS_PER_HALF} pairs in BOTH halves at every "
                          f"lag the arm uses. The uncentred median is co-reported. At L6 the "
                          f"lag1 arm must reproduce results/phase3_lag_fuse.json's "
                          f"{FUSE_L6_LAG1_CENTRED} within {FUSE_TOL} (ASSERT). The lag2 arm "
                          f"is compared to verify36's {FUSE_L6_LAG2_VERIFY36} by PRINT only "
                          f"(different code path, no assert)"),
        gate_consequence=("an arm below the gate at a layer is skipped entirely at that "
                          "layer -- no supervised verdict, no blind clustering, no nulls -- "
                          "and its measured reliability is reported. It is excluded from "
                          "every BH family. This is what stops the avenue producing an "
                          "uncalibrated negative. (--blind-below-gate forces the arm to run "
                          "as a diagnostic; it still issues no verdict.)"),

        positive_control_a=(
            "PLAIN TYPE-MEAN ANCHOR (no identity removal), Addendum F.1(a). screen() of the "
            "frozen published member set on TM @ w_dec must reproduce the prereg section 0 "
            f"gap_ratio to 4 dp (tolerance {ANCHOR_TOL_GAP_RATIO}) or the script STOPS with "
            "'instrument not validated' (R2). Reconciliation, declared: prereg section 0 "
            "lists L6 (k=13, member_hash 7d1d75a303d6, gap_ratio 1.65876), L7 (k=14, "
            "gap_ratio 1.62041) and L8 ('same hash', k=13, gap_ratio 3.6356). The L7 number "
            "belongs to the FOURTEEN-member cluster (hash 6bb0e54cd127 = the 13-set plus "
            "token 30651), not the 13-set; the task brief's shorthand 'the 13-member set' is "
            "resolved against prereg section 0's own '(k=14)' label. The asserted anchors are "
            "therefore 13-set at L6 and L8, 14-set at L7. The 13-set is ALSO screened at L7 "
            "and co-reported without assertion. p_closed is co-reported, not asserted (the "
            "task brief asserts gap_ratio only). The depth profile matters as much as the two "
            "passes: reproducing L6/L7 while missing the L8 FAILURE would indicate a "
            "different code path, so the L8 cell is asserted too"),
        positive_control_b=(
            f"PLANTED RING IN ACTION SPACE, Addendum F.1(b) -- THIS is what licenses a "
            f"negative. The 12 strict-12 rows of delta are REPLACED by a regular 12-gon: "
            f"radius = {PLANTED_RADIUS_FRACTION} x the median ||delta(+1)|| over included "
            f"pool types; a fresh random orthonormal 2-frame of R^{D_MODEL} per draw; "
            f"Gaussian angular jitter with sd = j degrees at j in "
            f"{list(PLANTED_JITTERS_DEG)} (the convention of "
            f"scripts/28_phase2_keys.py closure_power_curve, copied so the curve is in the "
            f"project's existing jitter units); plus the MEASURED split-half noise, Gaussian "
            f"with per-coordinate sd = RMS over the 12 month rows of "
            f"((A_even - A_odd)/sqrt(2)) projected. {N_PLANTED_DRAWS} draws per rung; "
            f"detection = p_closed < {ALPHA} at {N_PERM_CLUSTER} permutations; the "
            f"{int(POWER_TARGET*100)}%-power jitter is reported (grid value and linear "
            "interpolation). Computed PER ARM on that arm's own split-half noise. "
            "TWO CO-REPORTED CURVES, strict additions that can only qualify the negative's "
            "reach, never extend it: (secondary) the same declared radius at the "
            "FULL-DATA-matched noise level (sd / sqrt(2)) -- (A_even - A_odd)/sqrt(2) is "
            "the ONE-HALF noise scale, so the frozen primary curve is conservative by "
            "sqrt(2) relative to the estimate the real statistic uses; (tertiary) the "
            "MEASURED in-plane radius of the 12 real month rows (median ||xy|| in their own "
            "pca_2d plane) at the full-data noise level -- the frozen '0.5 x median AMBIENT "
            "norm' radius is far larger than any real in-plane radius in 1152-d, so the "
            "tertiary curve is the honest amplitude-matched comparator. CALIBRATION "
            "READING, declared: if a curve never reaches "
            f"{int(POWER_TARGET*100)}% detection even at ZERO jitter, a negative verdict "
            "from that arm at that layer is UNCALIBRATED and is reported as "
            "'NOT CLOSED (negative uncalibrated)', not as an absence of structure "
            "(prereg R2's spirit: an instrument that cannot detect a planted object "
            "produces uncalibrated negatives). The verdict annotation uses the FROZEN "
            "primary curve; the other two are reported beside it. ADDENDUM H AMENDMENT, "
            "declared: all three curves are computed IN THE KEPT SUBSPACE -- ambient "
            "dimension d = the number of kept coordinates; planted noise sd = Addendum F's "
            "frozen MONTH-ROW split-half sd RESTRICTED to the kept coordinates (the "
            "label-free pool-wide sd selects WHICH coordinates are dropped; the noise MODEL "
            f"stays F's month-row RMS); declared radius = {PLANTED_RADIUS_FRACTION} x the "
            "median POST-DROP ||delta|| over included types. Rationale: a positive control "
            "must calibrate the statistic that is actually run, and after Addendum H that "
            "statistic runs in the kept subspace. A FOURTH, PRE-DROP-RADIUS curve is "
            "co-reported so the strictly literal 'only the noise model changes' reading "
            "is reconstructable; it is not verdict-bearing"),

        supervised_anchor_arm=(
            "Run on the DROPPED-COORDINATE matrices (Addendum H); the whitened arm and the "
            "two control sets are co-reported through the identical path. "
            "DEMOTED TO ANCHOR by Addendum F.5 (the blind arm is PRIMARY). Sets: strict-12 "
            "(highest-count in-pool id per month, asserted equal to the frozen "
            f"{STRICT_12_EXPECTED}) and the 13-member set. Per arm (lag1, lag2, diff): "
            f"screen() at {N_PERM_CLUSTER} permutations; p_bg against {N_BACKGROUND_DRAWS} "
            "COUNT-AND-FREQUENCY-MATCHED random type subsets drawn by "
            f"gfdisc.nulls.frequency_matched_resample (tol {FREQ_MATCH_TOL} in log10 corpus "
            "count) over the pool, through the identical screening path (the resampler is "
            "used UNMODIFIED, so a draw may return fewer than k types and does not exclude "
            "the anchor members themselves; both are reported); ordering by "
            f"fits.cyclic_order_score at {N_PERM_ORDER} permutations PLUS the mandated "
            f"(prereg R4) block-permutation ordering null at {N_PERM_ORDER} draws "
            "(scripts/29 block_permutations). Ordering blocks: lexical (leading_space x "
            "capitalized, scripts/34's definition). For strict-12 all twelve tokens are "
            "leading-space capitalized month words -> ONE block -> the block-permutation "
            "null is uniform, exactly as the task declares"),

        nulls=dict(
            addendum_H_note=(
                "ALL FOUR nulls run in the KEPT SUBSPACE. Nulls (i) and (ii) recompute delta "
                "rows from re-drawn cache rows at the full 1152 coordinates and then SLICE "
                "to the kept coordinates before screen(), so the null and the observed "
                "statistic live in the identical space. Null (iii) resamples rows of the "
                "already-dropped matrix. Null (iv) DROPS FIRST and row-normalises SECOND"),
            i_successor_composition_matched=(
                f"{N_NULL_COMPOSITION} draws. For each month type t and each successor type "
                "s, draw C(+d)_full[t, s] successor ROWS uniformly from the admissible "
                "lag-d successor rows of type s whose PREDECESSOR is not t; recompute delta "
                "for the 12 months only through the identical projection and "
                "identity-removal arithmetic; screen(). Sampling is WITH REPLACEMENT within "
                f"a (t, s) cell, implemented by vectorised rejection ({COMPOSITION_REJECT_ROUNDS} "
                "rounds) against the predecessor-is-t rows: this preserves the successor-type "
                "composition exactly, which is the quantity the null matches. A cell whose "
                "successor type has NO pair from a predecessor other than t is skipped and "
                "the shortfall is reported. Run per ARM: the lag1 and lag2 arms use their "
                "own lag's pair structure; the diff arm differences two INDEPENDENT draws "
                "(one per lag). REQUIREMENT: closure rate "
                f"(p_closed < {ALPHA}) < {NULL_CLOSE_RATE_MAX}"),
            ii_within_document_predecessor_swap=(
                f"{N_NULL_PREDSWAP} draws (critic repair 4: the composition null draws its "
                "controls from OTHER documents, so predecessor-conditional document TOPIC "
                "survives it). Within each cached sequence the predecessor labels are "
                "permuted among that sequence's admissible lag-d pairs, preserving the "
                "per-sequence composition exactly; equivalently each sequence's month-"
                "labelled pairs are re-drawn as a uniformly random same-size subset of that "
                "sequence's admissible pairs, with the month identities carried over in "
                "sampling order (which is uniform, so no further shuffle is needed). "
                "Per-month pair counts are therefore preserved exactly. Recompute delta for "
                "the 12 months and screen(). Run per ARM as for null (i). REQUIREMENT: "
                f"closure rate < {NULL_CLOSE_RATE_MAX}"),
            iii_frequency_pseudo_code=(
                f"{N_NULL_FREQPSEUDO} draws. Each month's delta row is replaced by the "
                "delta row of the pool type with the nearest log10 corpus count, EXCLUDING "
                "every calendar token id (all month and weekday variants in "
                "results/token_map.json plus the 13/14-member sets), with random "
                f"tie-breaking among the {FREQ_PSEUDO_NEAREST} nearest; duplicate picks "
                "within a draw are re-drawn (up to 20 attempts) so k stays 12. This is the "
                "battery that killed the kinship lattice. REQUIREMENT: closure rate "
                f"< {NULL_CLOSE_RATE_MAX}"),
            iv_unit_norm_arm=(
                "delta norms scale as 1/sqrt(count), so an unguarded PCA could recover a "
                "FREQUENCY ring. All pool rows are L2-normalised and the months are "
                "re-screened, with p_bg from a count-and-frequency-matched background drawn "
                "on the SAME row-normalised matrix. DECLARED READING OF THE SPEC: Addendum "
                "F step 6 says 'all four nulls fail to close in >= 95% of draws', but the "
                "unit-norm arm is not draw-based -- it is a ROBUSTNESS REQUIREMENT ON THE "
                "POSITIVE (the months must STILL close after row normalisation), which is "
                "the only coherent reading. It enters the verdict conjunction as "
                f"'unit-norm p_closed < {ALPHA}'; its p_bg is co-reported and is not a "
                "verdict term. Nulls (i)-(iii) carry the < 5% closure-rate requirement"),
            layer_restriction=(
                "nulls (i) and (ii) re-read cache rows and are run at the CONFIRMATORY "
                f"layers {list(CONFIRMATORY_LAYERS)} only (--null-layers), exactly as "
                "pre-declared in the avenue text ('restrict to layers 6 and 7'). Nulls "
                "(iii) and (iv) are cheap and run at every gate-passing layer. CONSEQUENCE, "
                "declared: an exploratory layer can never receive the full step-6 verdict; "
                "an exploratory cell that would otherwise close is reported as "
                "'closure_flagged_nulls_not_run' and is NOT verdict-bearing")),

        blind_arm=(
            "PRIMARY (Addendum F.5), run on the DROPPED-COORDINATE matrices (Addendum H). "
            "All 3,795 delta rows of each arm through the "
            "byte-identical machinery of scripts/22: cosine on pool-mean-CENTRED rows; "
            f"edges where cos >= tau, weights = cos; Leiden RBConfigurationVertexPartition "
            f"resolution {LEIDEN_RESOLUTION}, seed {LEIDEN_SEED}, n_iterations=-1; the "
            f"declared tau grid {GRID_CENTRED} with edge budget {EDGE_BUDGET}; the declared "
            "size-profile selection rule (a tau is VALID iff its graph is within the edge "
            "budget AND the number of clusters of size >= 5 lies in [10, 300] AND the "
            "largest cluster holds <= 10% of the pool; select the SMALLEST valid tau; "
            "fallback 1 = smallest tau with max cluster <= 10% of pool; fallback 2 = largest "
            f"clusterable tau); the declared refinement arm at resolutions "
            f"{REFINE_RESOLUTIONS}; ring screening of every cluster of size "
            f"{SCREEN_MIN}-{SCREEN_MAX} at {N_PERM_CLUSTER} permutations; matched empirical "
            f"background of {N_BACKGROUND_DRAWS} random same-size subsets OF THE SAME MATRIX "
            f"at {N_PERM_BACKGROUND} permutations; FLAGGED iff p_closed < {FLAG_ALPHA} AND "
            f"p_bg <= {FLAG_ALPHA}; BH q={BH_Q} within (layer, arm, partition). DECLARED "
            "OMISSION: the raw_leiden and hdbscan arms of scripts/22 are NOT run -- Addendum "
            "F.5 names the centred-cosine + Leiden machinery, which is scripts/22's declared "
            "PRIMARY arm; the centred_refined derivative IS run because it is part of that "
            "arm. Types with zero admissible pairs give all-zero rows; screen()'s degenerate "
            "guard handles them and their count is reported"),
        blind_passer_audit=(
            f"every flagged cluster: (1) leave-one-out jackknife -- all k LOO subsets must "
            f"keep p_closed < {ALPHA} at {N_PERM_LOO} permutations; (2) frequency nuisance "
            f"Spearman(angle rank, log10 count) with {N_PERM_ORDER} WITHIN-LEXICAL-BLOCK "
            "permutations, blocks = leading_space x capitalized (scripts/34 survivor audit, "
            "copied); (3) frequency-pseudo null -- each member's row replaced by the nearest-"
            f"count non-member pool type's row, {N_NULL_FREQPSEUDO} draws, closure rate "
            "reported; (4) TOKEN-CLASS COMPOSITION (Addendum H) -- the cluster's rate of "
            "each of the six scripts/34 lexical features (leading_space, capitalized, "
            "all_caps, has_digit, is_punct, non_ascii) against the pool base rate, with the "
            "member counts, so a passer that is really one lexical class is visible. "
            "ADDENDUM H CONSEQUENCE: a passer whose leave-one-out jackknife FAILS is marked "
            "not_reportable = true (verify41 precedent) on both its audit and its screened "
            "row; it keeps its BH status in the tables but is excluded from the reportable "
            "passer list"),
        expected_false_flags=("per layer x arm x partition, the sum over screened clusters "
                              "of the background rate_p_closed_lt_05 at that cluster's size "
                              "(the formula declared in prereg section 2.1, "
                              "sum_k n_candidates_k x rate_k), reported beside the observed "
                              "flag count"),
        labels=("R3-compliant. No label enters any blind decision statistic. Token strings "
                "appear only in the post-freeze labels_after_freeze block, and the "
                "supervised anchor arm is explicitly labelled as an anchor, never as "
                "discovery"),
        decision_F_carried_forward=(
            f"Addendum F step 6, primary verdict (literal). SUPERSEDED as the layer verdict "
            f"by decision_H above, but still computed and reported per arm so scripts/41 "
            f"and scripts/41b can be compared term by term: the supervised month ring in "
            f"action space CLOSES at (layer, arm) iff p_closed < {ALPHA} AND p_bg < {ALPHA} "
            f"AND rho >= {RHO_THRESHOLD} AND nulls (i)(ii)(iii) each close in "
            f"< {NULL_CLOSE_RATE_MAX} of draws AND the unit-norm arm still closes. "
            "CO-REPORTED, not folded in silently: closed_with_ordering_null additionally "
            f"requires the block-permutation ordering p < {ALPHA} (the avenue's own DECISION "
            "line; prereg R4 mandates carrying the block-permutation null regardless). "
            "PERSISTENT-CONTEXT CARRY iff the ring is present in lag1 and absent in diff. "
            f"Exploratory layers (0-5, 8-10) under BH q={BH_Q} within arm over the "
            "gate-passing exploratory layers; confirmatory layers 6 and 7 are not "
            "BH-adjusted. Blind flags are reported with their audits and their expected "
            "false-flag count"),
        position_disclaimer=("row indices are document-relative (prereg section 0); the "
                             "parity split is over SEQUENCES, not positions, so no statistic "
                             "here is a claim about absolute sequence position"),
        seed_rule=("cid_seed(layer, arm, cluster_id) = int(md5('{layer}|{arm}|{cid}')"
                   "[:6], 16), copied from scripts/22. Background seeds: 10000*L + k "
                   "(blind), 610000 + 1000*L + arm_index (supervised matched background), "
                   "870000 + 1000*L + arm_index (unit-norm background). Null seeds: "
                   "composition 820000 + 1000*L + arm_index, predecessor-swap "
                   "830000 + 1000*L + arm_index, frequency-pseudo 840000 + 1000*L + "
                   "arm_index, passer-audit frequency-pseudo 860000 + 1000*L + first member, "
                   "planted ring 850000 (primary) / 851000 (secondary) / 852000 (tertiary) "
                   "+ 10000*arm_index + int(100*jitter). NEW IN 41b (Addendum H): the "
                   "co-reported whitened strict-12 cell uses 611000 + 1000*L + arm_index; "
                   "the ordinal number-word control uses 612000 + 1000*L + arm_index; the "
                   "weekday control uses 613000 + 1000*L + arm_index; the Addendum H1 "
                   "sensitivity ladder uses 900000 + 100000*rung_index + 1000*L + "
                   "arm_index (rung_index 0..3 over drop_top1, drop_top2, drop_top5, "
                   "whitened), which cannot collide with any base above. No existing seed "
                   "was "
                   "changed, so every scripts/41 cell that survives unchanged is bit-"
                   "reproducible from this file on the FULL-coordinate matrices"),
        frozen_parameters=dict(
            count_floor=COUNT_FLOOR, d_model=D_MODEL, d_sae=D_SAE, seq_len=SEQ_LEN,
            n_rows_expected=N_ROWS_EXPECTED, n_pool_expected=N_POOL_EXPECTED,
            chunk=CHUNK, lags=list(LAGS), arms=list(ARMS),
            arm_lags={k: list(v) for k, v in ARM_LAGS.items()},
            min_pairs_per_half=MIN_PAIRS_PER_HALF, gate_threshold=GATE_THRESHOLD,
            fuse_l6_lag1_centred=FUSE_L6_LAG1_CENTRED, fuse_tol=FUSE_TOL,
            fuse_l6_lag2_verify36=FUSE_L6_LAG2_VERIFY36,
            screen_min=SCREEN_MIN, screen_max=SCREEN_MAX,
            n_perm_cluster=N_PERM_CLUSTER, n_perm_background=N_PERM_BACKGROUND,
            n_background_draws=N_BACKGROUND_DRAWS, n_perm_order=N_PERM_ORDER,
            n_shuffle_labels=N_SHUFFLE_LABELS, n_perm_loo=N_PERM_LOO,
            grid_centred=GRID_CENTRED, edge_budget=EDGE_BUDGET,
            leiden_resolution=LEIDEN_RESOLUTION, leiden_seed=LEIDEN_SEED,
            refine_resolutions=REFINE_RESOLUTIONS, flag_alpha=FLAG_ALPHA, bh_q=BH_Q,
            n_null_composition=N_NULL_COMPOSITION, n_null_predswap=N_NULL_PREDSWAP,
            n_null_freqpseudo=N_NULL_FREQPSEUDO,
            freq_pseudo_nearest=FREQ_PSEUDO_NEAREST, freq_match_tol=FREQ_MATCH_TOL,
            null_close_rate_max=NULL_CLOSE_RATE_MAX,
            composition_reject_rounds=COMPOSITION_REJECT_ROUNDS,
            n_planted_draws=N_PLANTED_DRAWS,
            planted_jitters_deg=list(PLANTED_JITTERS_DEG),
            planted_radius_fraction=PLANTED_RADIUS_FRACTION, power_target=POWER_TARGET,
            alpha=ALPHA, rho_threshold=RHO_THRESHOLD,
            confirmatory_layers=list(CONFIRMATORY_LAYERS),
            null_layers_default=NULL_LAYERS_DEFAULT,
            members_13=MEMBERS_13, members_14=MEMBERS_14,
            hash_13=HASH_13, hash_14=HASH_14, strict_12_expected=STRICT_12_EXPECTED,
            anchor_table={str(k): dict(member_set=v[0], gap_ratio=v[1], p_closed=v[2],
                                       member_hash=v[3]) for k, v in ANCHOR_TABLE.items()},
            anchor_tol_gap_ratio=ANCHOR_TOL_GAP_RATIO,
            outlier_share_threshold=OUTLIER_SHARE_THRESHOLD,
            whiten_floor=WHITEN_FLOOR,
            planted_gate_jitter_deg=PLANTED_GATE_JITTER_DEG,
            planted_gate_detection=PLANTED_GATE_DETECTION,
            planted_gate_layers=list(PLANTED_GATE_LAYERS),
            ordinal_words=list(ORDINAL_WORDS), control_sets=list(CONTROL_SETS),
            sensitivity_drop_depths=list(SENSITIVITY_DROP_DEPTHS),
            sensitivity_rungs=list(SENSITIVITY_RUNGS),
            sensitivity_seed_base=SENSITIVITY_SEED_BASE),
        provenance_of_calibration_wiring=(
            "STATED PLAINLY, because the mtimes do not establish it and the project's own "
            "pattern (scripts/29's corrected header, scripts/28's closure_power_curve) is "
            "to say so. Three pieces of positive-control (b) machinery were added AFTER a "
            "TRUNCATED-PREFIX smoke run (400,000 of 4,002,480 cache rows, written to a "
            "scratch path, never to results/) showed the frozen primary planted-ring curve "
            "at zero detection: (1) the curve is computed PER ARM on that arm's own "
            "split-half noise rather than once on lag1's; (2) the secondary "
            "(full-data-matched noise) and tertiary (measured in-plane radius) curves and "
            "the noise diagnostics (||sd||, sd_max, sd_top5, radius ratios) are "
            "co-reported; (3) a NOT-CLOSED verdict whose frozen primary curve has no power "
            "at zero jitter is annotated 'negative UNCALIBRATED'. Direction of effect is "
            "fixed by construction: every one of the three can only QUALIFY or WEAKEN a "
            "negative and can never strengthen a positive, create a flag, or move any "
            "frozen threshold -- the verdict-bearing curve is still the frozen primary one "
            "(declared radius, split-half noise), and no frozen parameter, statistic, "
            "member set, null, threshold or seed rule was changed after any number was "
            "seen. The R2 STOP gate is and remains positive control (a), the plain "
            "type-mean anchor, whose tolerance was frozen before the first run"),
        exposure_before_the_production_run=(
            "SCRIPTS/41's disclosure, carried forward verbatim: two scratch runs preceded "
            "the scripts/41 production run -- (i) the truncated smoke above; (ii) a "
            "SINGLE-LAYER FULL-ROW validation at L6 with reduced permutation/draw counts, "
            "run to a scratch path to exercise the two live asserts on real data. Both "
            f"asserts passed bit-exactly (anchor gap_ratio {ANCHOR_TABLE[6][1]}; fuse "
            f"median {FUSE_L6_LAG1_CENTRED}). "
            "SCRIPTS/41b's OWN ADDITIONAL EXPOSURE, disclosed here rather than left to "
            "mtimes: (iii) the whole of results/phase3_lag_action.json and of verify41's "
            "post-hoc scratch (outlier_l6, labelfree_l6, controls_l6), whose numbers are "
            "quoted inside Addendum H's frozen exposure paragraph -- reproduced verbatim in "
            "addendum_H_text above -- and are therefore KNOWN before this script runs; "
            "(iv) a pre-freeze arithmetic check of the drop rule and the Addendum H planted "
            "gate on verify41's saved L6 halves, which established that the rule selects "
            "exactly {1038} at L6 in all three arms and that the post-drop planted curve "
            "clears the 80%-at-10-degrees gate there; (v) the truncated smoke of THIS file "
            "(400,000 of 4,002,480 rows, one layer, reduced counts, written to a scratch "
            "path, never to results/). None of that moved a threshold, a member set, a "
            "null, a seed rule or the decision -- all of those are fixed by the frozen "
            "Addendum H text and are reproduced in frozen_parameters"),
        runtime_overrides=args_view,
        copied_code_provenance=(
            "scripts/36 (streaming accumulation, pair bookkeeping, C @ TM identity removal, "
            "project, rowwise_cosine, strict12, centred split-half statistic); scripts/22 "
            "(build_pool, cosine_matrix, leiden, sweep_and_select, refine, gap_null, gap_p, "
            "screen, background_for_size, p_bg_of, score_ordering, shuffled_label_control, "
            "cid_seed, member_hash); scripts/29 (block_permutations, _rank_of, "
            "_rank_from_angles, _sym_basis); scripts/34 (type_string, class_features, "
            "block_perm_spearman and the leading_space x capitalized block definition)"),
    )


# ===========================================================================
# COPIED from scripts/36_lag_reliability_fuse.py -- pool, pairs, streaming
# ===========================================================================
def build_pool(root, layer):
    """Token types over the count floor. tokens.i32 is byte-identical across layers."""
    tok = np.fromfile(Path(root) / f"layer_{layer}" / "tokens.i32", dtype=np.int32)
    cnt = np.bincount(tok)
    main = np.flatnonzero(cnt >= COUNT_FLOOR)
    return tok, cnt, main


def pair_index(tok, n_used, lut):
    """Per-successor-row bookkeeping for the admissible lag-1 and lag-2 pairs.

    Extension of scripts/36.pair_index to a lag grid. For row j and lag d the pair
    (i = j - d, j) is admissible iff (i mod 255) < (255 - d), which is exactly
    (j mod 255) >= d; and then (i // 255) == (j // 255), so the predecessor's sequence
    parity is (j // 255) % 2.
    """
    own = lut[tok[:n_used]].astype(np.int32)
    j = np.arange(n_used, dtype=np.int64)
    parity = ((j // SEQ_LEN) % 2).astype(np.int8)
    pos_in_seq = (j % SEQ_LEN)
    per_lag = {}
    for d in LAGS:
        prev = np.full(n_used, -1, dtype=np.int32)
        if n_used > d:
            prev[d:] = own[:n_used - d]
        pred_ok = (pos_in_seq >= d) & (prev >= 0)
        succ_ok = pred_ok & (own >= 0)
        per_lag[d] = dict(prev=prev, pred_ok=pred_ok, succ_ok=succ_ok)
    return own, parity, per_lag


def stream(root, layer, own, pred_par, n_used, n_types, d_sae, chunk=CHUNK):
    """ONE pass: type-mean sums TM_acc and the parity-split successor sums S per lag.

    pred_par[d][j] = predecessor pool index + parity * n_types for an accumulating row,
    else -1 (scripts/36's stacking trick: one np.add.at per lag per chunk, no per-nonzero
    parity array). There is NO chunk-boundary carry -- everything about a row's
    predecessor comes from tokens.i32, which is fully resident.
    """
    d = Path(root) / f"layer_{layer}"
    indptr = np.load(d / "indptr.npy")
    indices = np.memmap(d / "indices.u16", dtype=np.uint16, mode="r")
    values = np.memmap(d / "values.f16", dtype=np.float16, mode="r")

    TM_acc = np.zeros((n_types, d_sae), dtype=np.float64)
    S = {lag: np.zeros((2 * n_types, d_sae), dtype=np.float64) for lag in LAGS}
    seen = np.zeros(n_types, dtype=np.int64)
    nnz_total = 0
    t0 = time.time()
    for lo in range(0, n_used, chunk):
        hi = min(lo + chunk, n_used)
        a, b = int(indptr[lo]), int(indptr[hi])
        cols = np.asarray(indices[a:b], dtype=np.int32)
        vals = np.asarray(values[a:b], dtype=np.float32)
        ct = (indptr[lo + 1:hi + 1] - indptr[lo:hi]).astype(np.int64)
        nnz_total += int(b - a)

        rid = np.repeat(own[lo:hi], ct)
        m = rid >= 0
        np.add.at(TM_acc, (rid[m], cols[m]), vals[m])
        seen += np.bincount(own[lo:hi][own[lo:hi] >= 0], minlength=n_types)
        del rid, m
        gc.collect()

        for lag in LAGS:
            rid = np.repeat(pred_par[lag][lo:hi], ct)
            m = rid >= 0
            np.add.at(S[lag], (rid[m], cols[m]), vals[m])
            del rid, m
            gc.collect()
        del cols, vals, ct
        gc.collect()

        el = time.time() - t0
        print(f"  stream {hi}/{n_used} rows  nnz={nnz_total}  {el:.1f}s  "
              f"({hi / max(el, 1e-9) / 1e3:.0f}k rows/s)", flush=True)
    return TM_acc, S, seen, nnz_total


def project(A, W, chunk=512):
    out = np.empty((A.shape[0], W.shape[1]), dtype=np.float32)
    for lo in range(0, A.shape[0], chunk):
        hi = min(lo + chunk, A.shape[0])
        out[lo:hi] = A[lo:hi].astype(np.float32) @ W
    return out


def rowwise_cosine(X, Y):
    nx = np.linalg.norm(X, axis=1)
    ny = np.linalg.norm(Y, axis=1)
    return (X * Y).sum(1) / np.maximum(nx * ny, 1e-12)


def strict12(cnt, pool_set, token_map):
    """Highest-count in-pool token id per month. Labels touch nothing before this point."""
    ids, missing = [], []
    for name in MONTHS:
        cands = [int(x) for x in token_map["months"][name]
                 if int(x) in pool_set and int(x) < len(cnt)]
        if not cands:
            missing.append(name)
            continue
        ids.append(max(cands, key=lambda x: int(cnt[x])))
    return ids, missing


# ===========================================================================
# COPIED from scripts/22_token_type_geometry.py -- clustering, screening, background
# ===========================================================================
def cosine_matrix(R, centre):
    X = R - R.mean(0, keepdims=True) if centre else R.astype(np.float64)
    X = np.asarray(X, dtype=np.float32)
    X = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
    S = X @ X.T
    np.fill_diagonal(S, -2.0)
    return S


def leiden(n, ei, ej, w, resolution=LEIDEN_RESOLUTION, seed=LEIDEN_SEED):
    import igraph as ig, leidenalg as la
    g = ig.Graph(n=n, edges=list(zip(ei.tolist(), ej.tolist())))
    if len(w):
        g.es["weight"] = np.asarray(w, dtype=float).tolist()
        p = la.find_partition(g, la.RBConfigurationVertexPartition, weights="weight",
                              resolution_parameter=resolution, seed=seed, n_iterations=-1)
    else:
        p = la.find_partition(g, la.RBConfigurationVertexPartition,
                              resolution_parameter=resolution, seed=seed, n_iterations=-1)
    return [sorted(int(x) for x in c) for c in p]


def sweep_and_select(S, grid, n_types):
    """The declared threshold sweep + selection rule. Returns (rows, tau, rule, clusters)."""
    iu = np.triu_indices(len(S), 1)
    sv = S[iu]
    rows, partitions = [], {}
    for tau in grid:
        m = sv >= tau
        ne = int(m.sum())
        row = dict(tau=tau, n_edges=ne)
        if ne > EDGE_BUDGET:
            row.update(skipped="over declared edge budget", valid=False)
            rows.append(row); continue
        t0 = time.time()
        cl = leiden(len(S), iu[0][m], iu[1][m], sv[m])
        sizes = sorted((len(c) for c in cl), reverse=True)
        n5 = int(sum(1 for s in sizes if s >= 5))
        row.update(seconds=round(time.time() - t0, 1),
                   n_clusters_total=len(cl), n_clusters_ge5=n5,
                   n_clusters_in_screen_band=int(sum(1 for s in sizes
                                                     if SCREEN_MIN <= s <= SCREEN_MAX)),
                   max_size=int(sizes[0]) if sizes else 0,
                   coverage_ge5=float(sum(s for s in sizes if s >= 5) / n_types),
                   top_sizes=[int(s) for s in sizes[:8]])
        row["valid"] = bool(10 <= n5 <= 300 and row["max_size"] <= 0.10 * n_types)
        rows.append(row)
        partitions[tau] = cl
    valid = [r for r in rows if r.get("valid")]
    if valid:
        tau, rule = valid[0]["tau"], "primary: smallest valid threshold"
    else:
        fb = [r for r in rows if "skipped" not in r and r.get("max_size", 1e9) <= 0.10 * n_types]
        if fb:
            tau, rule = fb[0]["tau"], "fallback 1: smallest threshold with max cluster <= 10% of pool"
        else:
            ok = [r for r in rows if "skipped" not in r]
            tau, rule = (ok[-1]["tau"], "fallback 2: largest clusterable threshold in grid") \
                if ok else (None, "no threshold was clusterable within the edge budget")
    return rows, tau, rule, (partitions.get(tau) or [])


def refine(clusters, S, tau):
    """Declared refinement: split oversized clusters on their induced subgraph."""
    out, log = [], []
    for c in clusters:
        if len(c) <= SCREEN_MAX:
            out.append(c); continue
        idx = np.asarray(c)
        sub = S[np.ix_(idx, idx)]
        iu = np.triu_indices(len(idx), 1)
        sv = sub[iu]
        m = sv >= tau
        chosen, used = None, None
        for res in REFINE_RESOLUTIONS:
            parts = leiden(len(idx), iu[0][m], iu[1][m], sv[m], resolution=res)
            chosen, used = parts, res
            if max((len(p) for p in parts), default=0) <= SCREEN_MAX:
                break
        subs = [sorted(int(idx[i]) for i in p) for p in (chosen or [])]
        kept = [p for p in subs if len(p) <= SCREEN_MAX]
        out.extend(kept)
        log.append(dict(parent_size=len(c), resolution_used=used,
                        n_subclusters=len(subs), n_kept=len(kept),
                        kept_sizes=sorted((len(p) for p in kept), reverse=True)[:12]))
    return out, log


_NULL = {}


def gap_null(k, n_perm, seed=0):
    """The uniform-angle max-gap null of scripts/13_rerun_gate.gap_pvalue, cached per k."""
    key = (int(k), int(n_perm), int(seed))
    if key not in _NULL:
        rng = np.random.default_rng(seed)
        a = np.sort(rng.uniform(0, 2 * np.pi, size=(n_perm, k)), axis=1)
        _NULL[key] = np.diff(np.concatenate([a, a[:, :1] + 2 * np.pi], axis=1), axis=1).max(1)
    return _NULL[key]


def gap_p(k, max_gap_rad, n_perm, seed=0):
    g = gap_null(k, n_perm, seed)
    return float((1 + (g <= max_gap_rad).sum()) / (1 + n_perm))


def screen(M, n_perm, seed=0):
    """Label-free ring screening of a point set. None if the set carries no geometry."""
    M = np.asarray(M, dtype=np.float64)
    if len(M) < 3 or degenerate(M):
        return None
    xy, ratios, Vt = pca_2d(M)
    cx, cy, r, resid = fit_circle(xy)
    c = circularity(xy, cx, cy, r)
    return dict(k=int(len(M)), gap_ratio=c["gap_ratio"], max_gap_deg=c["max_gap_deg"],
                radius_ratio=c["radius_ratio"], circle_resid_over_r=float(resid),
                top2_var=float(ratios[:2].sum()), pc1_var=float(ratios[0]),
                p_closed=gap_p(len(M), np.radians(c["max_gap_deg"]), n_perm, seed))


def background_for_size(R, pool_n, k, seed, n_draws=N_BACKGROUND_DRAWS, weights=None,
                        n_perm=N_PERM_BACKGROUND):
    """Random same-size token-type subsets through the identical screening."""
    rng = np.random.default_rng(seed)
    gaps, ps, rr, t2, ks = [], [], [], [], []
    for _ in range(n_draws):
        idx = rng.choice(pool_n, size=k, replace=False) if weights is None else weights(rng, k)
        s = screen(R[idx], n_perm, seed=0)
        if s is None:
            continue
        gaps.append(s["max_gap_deg"]); ps.append(s["p_closed"])
        rr.append(s["radius_ratio"]); t2.append(s["top2_var"]); ks.append(s["k"])
    gaps = np.asarray(gaps)
    return dict(k=int(k), n_draws=int(n_draws), n_scorable=int(len(gaps)),
                n_draws_at_full_k=int(sum(1 for x in ks if x == k)),
                rate_p_closed_lt_05=float(np.mean(np.asarray(ps) < 0.05)) if len(ps) else None,
                max_gap_deg_pct=[float(x) for x in np.percentile(gaps, [1, 5, 25, 50])]
                if len(gaps) else None,
                mean_max_gap_deg=float(gaps.mean()) if len(gaps) else None,
                mean_radius_ratio=float(np.mean(rr)) if rr else None,
                mean_top2_var=float(np.mean(t2)) if t2 else None,
                _gaps=gaps)


def p_bg_of(bg, max_gap_deg):
    g = bg["_gaps"]
    return float((1 + int((g <= max_gap_deg).sum())) / (1 + len(g))) if len(g) else None


def cid_seed(layer, arm, cid):
    return int(hashlib.md5(f"{layer}|{arm}|{cid}".encode()).hexdigest()[:6], 16)


def member_hash(members):
    return hashlib.md5(",".join(str(int(x)) for x in sorted(members)).encode()).hexdigest()[:12]


def score_ordering(xy_all, pos_in_cluster, label_index, full_k, seed, n_perm=N_PERM_ORDER):
    """Ordering + closure for the label members, from an already-computed 2-D embedding."""
    P = np.asarray(xy_all)[np.asarray(pos_in_cluster)]
    cx, cy, r, resid = fit_circle(P)
    c = circularity(P, cx, cy, r)
    ang = angles(P, cx, cy)
    k = len(P)
    s = cyclic_order_score(ang, np.asarray(label_index), k, n_perm=n_perm,
                           seed=seed, full_k=full_k)
    rot = {}
    for sign, a in (("as_returned", ang), ("pc2_negated", -ang)):
        rank = np.empty(k, dtype=np.float64)
        jit = np.random.default_rng(seed).permutation(k)
        order = np.lexsort((jit, a)); rank[order] = np.arange(k)
        best = -2.0
        lab = np.asarray(label_index)
        for shift in range(full_k):
            ref = np.argsort(np.argsort((lab + shift) % full_k)).astype(np.float64)
            d = rank - ref
            best = max(best, 1.0 - 6.0 * float(d @ d) / (k * (k * k - 1)))
        rot[sign] = float(best)
    return dict(n_labels=k, label_index=[int(x) for x in label_index],
                rho=s["rho"], p_order=s["p"], symmetry=[int(x) for x in s["symmetry"]],
                null_mean=s["null_mean"], null_p95=s["null_p95"],
                rho_rotation_only=rot,
                gap_ratio=c["gap_ratio"], max_gap_deg=c["max_gap_deg"],
                radius_ratio=c["radius_ratio"],
                p_closed=gap_p(k, np.radians(c["max_gap_deg"]), N_PERM_CLUSTER, seed=0),
                angles_deg=[float(x) for x in np.degrees(ang) % 360],
                _angles_rad=ang)


def shuffled_label_control(xy_all, pos_in_cluster, label_index, full_k, seed, observed_rho,
                           n_shuffles=N_SHUFFLE_LABELS):
    """Explicit shuffled-labels control, independent of cyclic_order_score's internal null."""
    P = np.asarray(xy_all)[np.asarray(pos_in_cluster)]
    cx, cy, r, _ = fit_circle(P)
    ang = angles(P, cx, cy)
    k = len(P)
    rng = np.random.default_rng(seed + 991)
    lab = np.asarray(label_index)
    null = []
    for _ in range(n_shuffles):
        s = cyclic_order_score(ang, lab[rng.permutation(k)], k, n_perm=1, seed=seed,
                               full_k=full_k)
        null.append(s["rho"])
    null = np.asarray(null)
    return dict(n_shuffles=int(n_shuffles), mean_rho=float(null.mean()),
                p95_rho=float(np.percentile(null, 95)), max_rho=float(null.max()),
                p_shuffled=float((1 + (null >= observed_rho).sum()) / (1 + len(null))))


# ===========================================================================
# COPIED from scripts/29_keys_blockperm.py -- block-permutation ordering null (R4)
# ===========================================================================
def _rank_of(values):
    order = np.argsort(np.asarray(values), kind="stable")
    r = np.empty(len(order), dtype=np.float64)
    r[order] = np.arange(len(order), dtype=np.float64)
    return r


def _rank_from_angles(ang, seed):
    ang = np.asarray(ang, dtype=float)
    n = len(ang)
    jitter = np.random.default_rng(seed).permutation(n)
    order = np.lexsort((jitter, ang))
    rank_of = np.empty(n, dtype=np.float64)
    rank_of[order] = np.arange(n)
    return rank_of


def _sym_basis(label_index, K):
    lab = np.asarray(label_index, dtype=int)
    return np.vstack([_rank_of((f * lab + s) % K) for f in (1, -1) for s in range(K)])


def block_permutations(blocks, n_draws, seed):
    """`n_draws` permutations of 0..n-1 that map each block onto itself."""
    rng = np.random.default_rng(seed)
    n = sum(len(b) for b in blocks)
    sig = np.tile(np.arange(n), (n_draws, 1))
    for b in blocks:
        b = np.asarray(b)
        pick = np.argsort(rng.random((n_draws, len(b))), axis=1)
        sig[:, b] = b[pick]
    return sig


def block_perm_ordering(ang, label_index, full_k, blocks, n_draws, seed):
    """rho against the known cycle with a WITHIN-LEXICAL-BLOCK permutation null (R4)."""
    rank_of = _rank_from_angles(ang, seed)
    n = len(rank_of)
    base = _sym_basis(label_index, full_k)
    denom = n * (n * n - 1)

    def rhos(sig):
        d = rank_of - base[:, sig]
        return 1.0 - 6.0 * (d * d).sum(-1) / denom

    rho = float(rhos(np.arange(n)).max())
    sig = block_permutations(blocks, n_draws, seed + 31337)
    null = rhos(sig).max(0)
    return dict(rho=rho, n_draws=int(n_draws),
                p_block=float((1 + int((null >= rho).sum())) / (1 + n_draws)),
                null_mean=float(null.mean()), null_p95=float(np.percentile(null, 95)),
                n_blocks=int(len(blocks)), block_sizes=[int(len(b)) for b in blocks],
                uniform=bool(len(blocks) == 1),
                note=("blocks = leading_space x capitalized (scripts/34). For the strict-12 "
                      "months there is a single block, so this null is uniform -- as "
                      "declared. angle is circular, so its linear rank depends on the "
                      "arctan2 branch cut; the max over the 2*full_k symmetries makes the "
                      "statistic invariant to it and the null uses the identical convention"))


# ===========================================================================
# COPIED from scripts/34_spectral_search.py -- lexical classes for the audits
# ===========================================================================
def type_string(tokenizer, tid):
    s = tokenizer.convert_ids_to_tokens(int(tid))
    if s is None:
        s = tokenizer.decode([int(tid)])
    return s.replace("▁", " ")


def class_features(s):
    alphas = [c for c in s if c.isalpha()]
    all_caps = bool(alphas) and all(c.isupper() for c in alphas)
    capitalized = bool(alphas) and alphas[0].isupper() and not all_caps
    return [s.startswith(" "), capitalized, all_caps,
            any(c.isdigit() for c in s),
            not any(c.isalnum() for c in s),
            any(ord(c) > 127 for c in s)]


def _rankdata(x):
    x = np.asarray(x, dtype=np.float64)
    order = np.argsort(x, kind="stable")
    ranks = np.empty(len(x), dtype=np.float64)
    ranks[order] = np.arange(1, len(x) + 1, dtype=np.float64)
    sx = x[order]
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and sx[j + 1] == sx[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + j + 2) / 2.0
        i = j + 1
    return ranks


def _spearman(a, b):
    ra, rb = _rankdata(a), _rankdata(b)
    ra = ra - ra.mean(); rb = rb - rb.mean()
    den = float(np.sqrt((ra @ ra) * (rb @ rb)))
    return float(ra @ rb / den) if den > 0 else 0.0


def block_perm_spearman(ang, log_count, blocks, n_perm, seed):
    """Spearman(angle rank, log10 count), null = counts permuted WITHIN lexical blocks."""
    rho = _spearman(ang, log_count)
    groups = {}
    for i, b in enumerate(blocks):
        groups.setdefault(b, []).append(i)
    groups = {k: np.asarray(v) for k, v in groups.items()}
    rng = np.random.default_rng(seed)
    null = np.empty(n_perm, dtype=np.float64)
    lc = np.asarray(log_count, dtype=np.float64)
    for t in range(n_perm):
        x = lc.copy()
        for g in groups.values():
            if len(g) > 1:
                x[g] = lc[g][rng.permutation(len(g))]
        null[t] = _spearman(ang, x)
    return dict(rho=rho, n_perm=int(n_perm),
                p_two_sided=float((1 + int((np.abs(null) >= abs(rho)).sum())) / (1 + n_perm)),
                null_mean=float(null.mean()), null_abs_p95=float(np.percentile(np.abs(null), 95)),
                blocks={str(k): int(len(v)) for k, v in groups.items()},
                n_blocks=int(len(groups)))


# ===========================================================================
# NEW -- CSR row gathering, per-lag bookkeeping, and the resampling nulls
# ===========================================================================
def gather_rows(indptr, indices, values, rows):
    """Flat (cols, vals, lens) for an arbitrary set of CSR rows, one fancy-index each."""
    rows = np.asarray(rows, dtype=np.int64)
    if len(rows) == 0:
        return np.zeros(0, np.int64), np.zeros(0, np.float64), np.zeros(0, np.int64)
    starts = indptr[rows].astype(np.int64)
    lens = (indptr[rows + 1] - indptr[rows]).astype(np.int64)
    total = int(lens.sum())
    if total == 0:
        return np.zeros(0, np.int64), np.zeros(0, np.float64), lens
    csum = np.concatenate([[0], np.cumsum(lens)[:-1]])
    flat = np.repeat(starts - csum, lens) + np.arange(total, dtype=np.int64)
    return (np.asarray(indices[flat], dtype=np.int64),
            np.asarray(values[flat], dtype=np.float64), lens)


def delta_from_rows(rows, labs, n_lab, own, csr, W, TMP, d_sae):
    """delta rows for n_lab groups from an arbitrary assignment of successor rows.

    Identical arithmetic to the real path: (sum of successor codes) @ w_dec minus
    (successor-type counts) @ (TM @ w_dec), divided by the group's pair count. The two
    orderings agree exactly in exact arithmetic because the projection is linear.
    """
    indptr, indices, values = csr
    n_types = TMP.shape[0]
    labs = np.asarray(labs, dtype=np.int64)
    cols, vals, lens = gather_rows(indptr, indices, values, rows)
    rid = np.repeat(labs, lens)
    acc = np.bincount(rid * d_sae + cols, weights=vals,
                      minlength=n_lab * d_sae).reshape(n_lab, d_sae)
    codesum = acc.astype(np.float32) @ W
    stype = own[np.asarray(rows, dtype=np.int64)].astype(np.int64)
    Cd = np.bincount(labs * n_types + stype, minlength=n_lab * n_types).reshape(n_lab, n_types)
    ident = Cd.astype(np.float64) @ TMP.astype(np.float64)
    n = np.maximum(np.bincount(labs, minlength=n_lab), 1)
    return (codesum.astype(np.float64) - ident) / n[:, None]


def lag_bookkeeping(own, prev, succ_ok, n_types, n_used, month_rank):
    """Row groupings the two resampling nulls need, per lag. Label-free except month_rank,
    which enters only the null's own bookkeeping (the null is a control on a supervised
    anchor, so month identity is definitionally part of it)."""
    rows_acc = np.flatnonzero(succ_ok).astype(np.int64)
    succ_type = own[rows_acc].astype(np.int64)
    order_s = np.argsort(succ_type, kind="stable")
    group_rows = rows_acc[order_s]
    group_ptr = np.concatenate(
        [[0], np.cumsum(np.bincount(succ_type, minlength=n_types))]).astype(np.int64)
    seq_of = (rows_acc // SEQ_LEN).astype(np.int64)
    n_seq = int(np.ceil(n_used / SEQ_LEN))
    seq_ptr = np.searchsorted(seq_of, np.arange(n_seq + 1)).astype(np.int64)
    lab_of_pair = month_rank[prev[rows_acc]]
    mr = np.flatnonzero(lab_of_pair >= 0)
    if len(mr):
        g_seq, g_first, g_count = np.unique(seq_of[mr], return_index=True,
                                            return_counts=True)
    else:
        g_seq = g_first = g_count = np.zeros(0, np.int64)
    return dict(rows_acc=rows_acc, group_rows=group_rows, group_ptr=group_ptr,
                group_len=np.diff(group_ptr), seq_ptr=seq_ptr, prev=prev,
                m_group_seq=g_seq, m_group_first=g_first, m_group_count=g_count,
                m_lab=lab_of_pair[mr], n_month_pairs=int(len(mr)))


def composition_cells(month_pool_idx, C_month, bk):
    """Static (month, successor-type, count) cells for the composition-matched null."""
    glen = bk["group_len"]
    cm, cs, cc, skipped = [], [], [], []
    for li in range(len(month_pool_idx)):
        row = C_month[li]
        for s in np.flatnonzero(row > 0):
            avail = int(glen[s]) - int(row[s])
            if avail <= 0:
                skipped.append(dict(month_index=int(li), successor_pool_idx=int(s),
                                    n_pairs=int(row[s]),
                                    n_available_other_predecessor=int(max(avail, 0))))
                continue
            cm.append(li); cs.append(int(s)); cc.append(int(row[s]))
    cm = np.asarray(cm, dtype=np.int64); cs = np.asarray(cs, dtype=np.int64)
    cc = np.asarray(cc, dtype=np.int64)
    cell_of = np.repeat(np.arange(len(cc)), cc) if len(cc) else np.zeros(0, np.int64)
    labs = cm[cell_of] if len(cc) else np.zeros(0, np.int64)
    return dict(cells_month=cm, cells_s=cs, cells_c=cc, cell_of=cell_of, labs=labs,
                gstart=bk["group_ptr"][cs][cell_of] if len(cc) else np.zeros(0, np.int64),
                glen=bk["group_len"][cs][cell_of] if len(cc) else np.zeros(0, np.int64),
                pm_of_row=(np.asarray(month_pool_idx, dtype=np.int64)[labs]
                           if len(cc) else np.zeros(0, np.int64)),
                skipped=skipped, n_cells=int(len(cc)),
                n_pairs_per_draw=int(cc.sum()) if len(cc) else 0)


def composition_draw(cells, bk, rng):
    """One composition-matched draw -> (rows, labs, n_unresolved)."""
    n = len(cells["cell_of"])
    if n == 0:
        return np.zeros(0, np.int64), np.zeros(0, np.int64), 0
    pos = (rng.random(n) * cells["glen"]).astype(np.int64)
    rows = bk["group_rows"][cells["gstart"] + pos]
    bad = bk["prev"][rows] == cells["pm_of_row"]
    for _ in range(COMPOSITION_REJECT_ROUNDS):
        nb = int(bad.sum())
        if nb == 0:
            break
        pos_b = (rng.random(nb) * cells["glen"][bad]).astype(np.int64)
        rows[bad] = bk["group_rows"][cells["gstart"][bad] + pos_b]
        bad = bk["prev"][rows] == cells["pm_of_row"]
    keep = ~bad
    return rows[keep], cells["labs"][keep], int(bad.sum())


def predswap_draw(bk, rng):
    """One within-document predecessor-swap draw -> (rows, labs)."""
    g_seq, g_first, g_count = bk["m_group_seq"], bk["m_group_first"], bk["m_group_count"]
    if len(g_seq) == 0:
        return np.zeros(0, np.int64), np.zeros(0, np.int64)
    seq_ptr, rows_acc, m_lab = bk["seq_ptr"], bk["rows_acc"], bk["m_lab"]
    single = g_count == 1
    s1 = g_seq[single]
    lens1 = (seq_ptr[s1 + 1] - seq_ptr[s1]).astype(np.int64)
    off = (rng.random(len(s1)) * np.maximum(lens1, 1)).astype(np.int64)
    rows = [rows_acc[seq_ptr[s1].astype(np.int64) + off]]
    labs = [m_lab[g_first[single]]]
    for gi in np.flatnonzero(~single):
        sq = int(g_seq[gi]); c = int(g_count[gi])
        lo, hi = int(seq_ptr[sq]), int(seq_ptr[sq + 1])
        take = min(c, hi - lo)
        pick = rng.choice(hi - lo, size=take, replace=False)
        rows.append(rows_acc[lo + pick])
        f = int(g_first[gi])
        labs.append(m_lab[f:f + take])
    return np.concatenate(rows), np.concatenate(labs)


def run_resampling_null(kind, arm, bks, cells_by_lag, n_lab, own, csr, W, TMP, d_sae,
                        n_draws, n_perm, seed, keep=None):
    """Driver shared by nulls (i) and (ii). The diff arm differences two INDEPENDENT draws.

    ADDENDUM H: `keep` is the arm's kept-coordinate mask. The re-drawn delta rows are
    produced at the full 1152 coordinates by the identical arithmetic and then SLICED to
    the kept coordinates, so the null lives in exactly the space the observed statistic
    lives in.
    """
    lags = ARM_LAGS[arm]
    rng = np.random.default_rng(seed)
    closed, gaps, ps, unresolved, n_scorable = 0, [], [], 0, 0
    for _ in range(n_draws):
        D = None
        for lag in lags:
            if kind == "composition":
                rows, labs, unres = composition_draw(cells_by_lag[lag], bks[lag], rng)
                unresolved += unres
            else:
                rows, labs = predswap_draw(bks[lag], rng)
            Dl = delta_from_rows(rows, labs, n_lab, own, csr, W, TMP, d_sae)
            D = Dl if D is None else D - Dl
        if keep is not None:
            D = D[:, np.asarray(keep, dtype=bool)]
        s = screen(D, n_perm, seed=0)
        if s is None:
            continue
        n_scorable += 1
        gaps.append(s["max_gap_deg"]); ps.append(s["p_closed"])
        closed += int(s["p_closed"] < ALPHA)
    out = dict(null=("successor-composition-matched resample" if kind == "composition"
                     else "within-document predecessor-swap"),
               arm=arm, lags_used=list(lags), n_draws=int(n_draws),
               n_kept_coordinates=(int(np.asarray(keep).sum()) if keep is not None
                                   else int(D_MODEL)),
               screened_in_kept_subspace=bool(keep is not None),
               n_scorable=int(n_scorable),
               closure_rate=float(closed / max(n_scorable, 1)),
               fails_to_close_in_at_least_95pct=bool(
                   n_scorable > 0 and closed / n_scorable < NULL_CLOSE_RATE_MAX),
               p_closed_median=float(np.median(ps)) if ps else None,
               max_gap_deg_mean=float(np.mean(gaps)) if gaps else None,
               max_gap_deg_pct=[float(x) for x in np.percentile(gaps, [1, 5, 25, 50])]
               if gaps else None)
    if kind == "composition":
        out.update(n_rows_unresolved_total=int(unresolved),
                   cells={str(lag): dict(
                       n_cells=cells_by_lag[lag]["n_cells"],
                       n_pairs_per_draw=cells_by_lag[lag]["n_pairs_per_draw"],
                       n_cells_skipped_no_other_predecessor=len(cells_by_lag[lag]["skipped"]),
                       cells_skipped_examples=cells_by_lag[lag]["skipped"][:10])
                       for lag in lags})
    else:
        out.update(sequences={str(lag): dict(
            n_sequences_with_month_pairs=int(len(bks[lag]["m_group_seq"])),
            n_sequences_multi=int((bks[lag]["m_group_count"] > 1).sum()),
            n_pairs_per_draw=int(bks[lag]["m_group_count"].sum())) for lag in lags})
    return out


def freq_pseudo_null(Dmat, member_pos, counts_pool, excluded_pos, n_draws, n_perm, seed,
                     n_nearest=FREQ_PSEUDO_NEAREST):
    """Null (iii): each member's row replaced by a nearest-corpus-count non-calendar row."""
    lc = np.log10(np.maximum(counts_pool.astype(np.float64), 1.0))
    allowed = np.ones(len(lc), dtype=bool)
    ex = np.asarray(sorted(int(x) for x in excluded_pos), dtype=np.int64)
    if len(ex):
        allowed[ex] = False
    cand = np.flatnonzero(allowed)
    if len(cand) < n_nearest:
        return dict(null="frequency-pseudo-code battery", n_draws=0, n_scorable=0,
                    closure_rate=None, fails_to_close_in_at_least_95pct=False,
                    reason="fewer candidate types than n_nearest")
    nearest = np.asarray([cand[np.argsort(np.abs(lc[cand] - lc[p]), kind="stable")[:n_nearest]]
                          for p in member_pos])
    rng = np.random.default_rng(seed)
    closed, gaps, ps, n_scorable, n_retries = 0, [], [], 0, 0
    k = len(member_pos)
    for _ in range(n_draws):
        pick = None
        for _a in range(20):
            cp = nearest[np.arange(k), rng.integers(0, n_nearest, size=k)]
            if len(set(cp.tolist())) == k:
                pick = cp
                break
            n_retries += 1
        if pick is None:
            pick = cp
        s = screen(Dmat[pick], n_perm, seed=0)
        if s is None:
            continue
        n_scorable += 1
        gaps.append(s["max_gap_deg"]); ps.append(s["p_closed"])
        closed += int(s["p_closed"] < ALPHA)
    return dict(null="frequency-pseudo-code battery", n_draws=int(n_draws),
                n_scorable=int(n_scorable), n_nearest=int(n_nearest),
                n_duplicate_redraws=int(n_retries), n_candidate_types=int(len(cand)),
                closure_rate=float(closed / max(n_scorable, 1)),
                fails_to_close_in_at_least_95pct=bool(
                    n_scorable > 0 and closed / n_scorable < NULL_CLOSE_RATE_MAX),
                p_closed_median=float(np.median(ps)) if ps else None,
                max_gap_deg_mean=float(np.mean(gaps)) if gaps else None)


def planted_ring_power(radius, sd, n_draws, n_perm, jitters=PLANTED_JITTERS_DEG,
                       k=12, d=D_MODEL, seed_base=850_000, label=""):
    """Positive control (b): planted 12-gon in action space at the measured noise level.

    Jitter convention copied from scripts/28_phase2_keys.py closure_power_curve:
    theta = ideal + radians(j) * standard normal, i.e. Gaussian angular error with
    sd = j degrees. Plane: a fresh random orthonormal 2-frame of R^d per draw, so the
    curve is not hostage to how one fixed plane aligns with the anisotropic noise.
    """
    ideal = 2 * np.pi * np.arange(k) / k
    rows = {}
    for j in jitters:
        rng = np.random.default_rng(seed_base + int(j * 100))
        det, grs, pcs = 0, [], []
        for _ in range(n_draws):
            th = ideal + np.radians(j) * rng.normal(size=k)
            u = rng.normal(size=d); u /= np.linalg.norm(u)
            v = rng.normal(size=d); v -= (v @ u) * u; v /= np.linalg.norm(v)
            P = (radius * (np.outer(np.cos(th), u) + np.outer(np.sin(th), v))
                 + sd[None, :] * rng.normal(size=(k, d)))
            s = screen(P, n_perm, seed=0)
            if s is None:
                continue
            grs.append(s["gap_ratio"]); pcs.append(s["p_closed"])
            det += int(s["p_closed"] < ALPHA)
        rows[f"jitter_{j}deg"] = dict(
            jitter_deg=float(j), n_draws=int(n_draws), n_scorable=int(len(pcs)),
            detection_rate=float(det / max(len(pcs), 1)),
            gap_ratio_mean=float(np.mean(grs)) if grs else None,
            gap_ratio_median=float(np.median(grs)) if grs else None,
            p_closed_median=float(np.median(pcs)) if pcs else None)
    ok = [j for j in jitters if rows[f"jitter_{j}deg"]["detection_rate"] >= POWER_TARGET]
    js, interp = list(jitters), None
    for a, b in zip(js[:-1], js[1:]):
        da = rows[f"jitter_{a}deg"]["detection_rate"]
        db = rows[f"jitter_{b}deg"]["detection_rate"]
        if da >= POWER_TARGET > db:
            interp = float(a + (da - POWER_TARGET) / max(da - db, 1e-12) * (b - a))
            break
    return dict(label=label, k=int(k), d=int(d), radius=float(radius),
                noise_sd_mean=float(np.mean(sd)), noise_sd_median=float(np.median(sd)),
                jitters_deg=list(jitters), rows=rows,
                jitter_at_80pct_power_grid=float(max(ok)) if ok else None,
                jitter_at_80pct_power_interpolated=interp,
                jitter_convention=("theta = ideal + radians(j) * N(0,1), the convention of "
                                   "scripts/28_phase2_keys.py closure_power_curve"),
                reading=("the largest angular jitter at which closure still has 80% power "
                         "is the REACH of any negative from this instrument: a ring whose "
                         "angular fidelity is worse than that is not excluded"))


# ===========================================================================
def resolve_blocks(strings):
    """Lexical blocks for the ordering null: leading_space x capitalized (scripts/34)."""
    keys = {}
    for i, s in enumerate(strings):
        f = class_features(s)
        keys.setdefault(f"lead={bool(f[0])}|cap={bool(f[1])}", []).append(i)
    return [np.asarray(v) for v in keys.values()]


# ===========================================================================
# NEW IN 41b -- ADDENDUM H: the label-free outlier-coordinate rule and its bookkeeping
# ===========================================================================
def outlier_drop_rule(A_even, A_odd, included, threshold=OUTLIER_SHARE_THRESHOLD):
    """THE ONE ADDED PREPROCESSING RULE. Label-free, computed per arm and per layer.

    The pool-wide split-half residual over the INCLUDED types is (A_even - A_odd)/sqrt(2)
    in w_dec-projected space. Its per-coordinate sd is the noise scale of one half; a
    coordinate whose share of the total noise VARIANCE exceeds `threshold` is dropped.
    At most floor(1/threshold) = 3 coordinates can exceed a 25% share, so the rule cannot
    gut the space. No label of any kind enters: `included` is the Addendum F.2 pair-count
    mask over the whole pool.
    """
    d = A_even.shape[1]
    inc = np.asarray(included, dtype=bool)
    n_inc = int(inc.sum())
    if n_inc == 0:
        sd = np.zeros(d, dtype=np.float64)
    else:
        resid = (A_even[inc] - A_odd[inc]) / np.sqrt(2.0)
        sd = np.sqrt((resid.astype(np.float64) ** 2).mean(0))
    tot = float((sd ** 2).sum())
    shares = (sd ** 2 / tot) if tot > 0 else np.zeros(d, dtype=np.float64)
    dropped = np.flatnonzero(shares > threshold)
    keep = np.ones(d, dtype=bool)
    keep[dropped] = False
    order = np.argsort(sd)[::-1][:5]
    return dict(
        keep=keep, pool_noise_sd=sd,
        n_included_types=n_inc, threshold=float(threshold),
        dropped_coordinates=[int(x) for x in dropped],
        n_dropped=int(len(dropped)), n_kept=int(keep.sum()), d_before=int(d),
        dropped_shares=[float(shares[i]) for i in dropped],
        top5_by_sd=[dict(coord=int(i), sd=float(sd[i]), variance_share=float(shares[i]))
                    for i in order],
        total_noise_variance=tot,
        variance_share_dropped=float(shares[dropped].sum()) if len(dropped) else 0.0,
        rule=("pool-wide split-half residual (A_even - A_odd)/sqrt(2) over included types; "
              f"per-coordinate sd; share = sd^2 / sum(sd^2); DROP share > {threshold}"),
        label_free=True)


def class_composition(feats, members, n_types):
    """Passer audit (4): token-class make-up of a cluster against the pool base rate."""
    names = ["leading_space", "capitalized", "all_caps", "has_digit", "is_punct",
             "non_ascii"]
    mem = np.asarray(members, dtype=np.int64)
    sub = np.asarray(feats)[mem].astype(np.float64)
    base = np.asarray(feats).astype(np.float64)
    cf = sub.mean(0) if len(mem) else np.zeros(len(names))
    pf = base.mean(0)
    return dict(features=names, k=int(len(mem)),
                member_counts=[int(x) for x in sub.sum(0)],
                cluster_fraction=[float(x) for x in cf],
                pool_fraction=[float(x) for x in pf],
                pool_n=int(n_types),
                excess_over_pool=[float(a - b) for a, b in zip(cf, pf)],
                max_abs_excess=float(np.max(np.abs(cf - pf))) if len(names) else None,
                dominant_feature=names[int(np.argmax(np.abs(cf - pf)))],
                note=("descriptive, not a test: a passer whose membership is one lexical "
                      "class is visible here beside the frequency nuisance null"))


def resolve_ordinal_words(tokenizer, pool_pos):
    """Addendum H ordinal control: ' one'..' ten' as single in-pool token ids."""
    ids, idx, unresolved = [], [], []
    for i, w in enumerate(ORDINAL_WORDS):
        enc = tokenizer.encode(w, add_special_tokens=False)
        if len(enc) == 1 and int(enc[0]) in pool_pos:
            ids.append(int(enc[0])); idx.append(i)
        else:
            unresolved.append(dict(word=w, n_pieces=int(len(enc)),
                                   in_pool=bool(len(enc) == 1 and int(enc[0]) in pool_pos)))
    return ids, idx, unresolved


def resolve_weekdays(cnt, pool_pos, token_map):
    """Addendum H weekday control: highest-count in-pool id per day (the strict12 rule)."""
    ids, idx, unresolved = [], [], []
    for i, name in enumerate(DAYS):
        cands = [int(x) for x in token_map.get("days", {}).get(name, [])
                 if int(x) in pool_pos and int(x) < len(cnt)]
        if not cands:
            unresolved.append(dict(day=name, n_candidates=0))
            continue
        ids.append(max(cands, key=lambda x: int(cnt[x]))); idx.append(i)
    return ids, idx, unresolved


def closure_terms(cell):
    """Addendum H's four-term closure rule, read off a supervised_cell result."""
    if not cell or not cell.get("scorable"):
        return dict(scorable=False, closes=False, p_closed=None, p_bg=None, rho=None,
                    block_perm_p=None, closes_two_term=False,
                    rule=("p_closed < %g AND p_bg < %g AND rho >= %g AND block-perm p < %g"
                          % (ALPHA, ALPHA, RHO_THRESHOLD, ALPHA)))
    scr = cell["screening"]
    p = float(scr["p_closed"])
    pbg = cell.get("p_bg")
    rho = cell["ordering"]["rho"]
    pb = cell["ordering_block_perm"]["p_block"]
    two = bool(p < ALPHA and pbg is not None and pbg < ALPHA)
    return dict(scorable=True, p_closed=p, p_bg=None if pbg is None else float(pbg),
                rho=None if rho is None else float(rho),
                block_perm_p=None if pb is None else float(pb),
                gap_ratio=float(scr["gap_ratio"]),
                closes_two_term=two,
                closes=bool(two and rho is not None and rho >= RHO_THRESHOLD
                            and pb is not None and pb < ALPHA),
                rule=("p_closed < %g AND p_bg < %g AND rho >= %g AND block-perm p < %g"
                      % (ALPHA, ALPHA, RHO_THRESHOLD, ALPHA)))


def sensitivity_rung_matrix(Dfull, sd_pool, rung):
    """One rung of the Addendum H1 ladder. Label-free: the ranking is the pool-wide sd."""
    order = np.argsort(sd_pool)[::-1]
    if rung == "whitened":
        keep = np.ones(Dfull.shape[1], dtype=bool)
        return (Dfull / np.maximum(sd_pool, WHITEN_FLOOR)[None, :], keep,
                [], "every coordinate divided by its pool-wide split-half noise sd; "
                    "no coordinate dropped")
    depth = int(rung.replace("drop_top", ""))
    dropped = [int(x) for x in order[:depth]]
    keep = np.ones(Dfull.shape[1], dtype=bool)
    keep[order[:depth]] = False
    return (np.ascontiguousarray(Dfull[:, keep]), keep, dropped,
            f"the {depth} coordinate(s) with the largest pool-wide split-half noise share "
            f"removed")


def planted_gate_of(curve):
    """Addendum H's R2 gate on one planted-ring curve (both declared readings)."""
    rows = curve["rows"]
    key = f"jitter_{PLANTED_GATE_JITTER_DEG}deg"
    at_gate = rows[key]["detection_rate"] if key in rows else None
    n_gate = rows[key]["n_scorable"] if key in rows else 0
    below = [rows[f"jitter_{j}deg"]["detection_rate"] for j in curve["jitters_deg"]
             if j <= PLANTED_GATE_JITTER_DEG and f"jitter_{j}deg" in rows]
    ci = None
    if at_gate is not None and n_gate:
        # Wilson 95% interval, so a Monte-Carlo-marginal gate call is visible as such.
        z, ph, n = 1.959963985, float(at_gate), float(n_gate)
        c = (ph + z * z / (2 * n)) / (1 + z * z / n)
        h = (z / (1 + z * z / n)) * np.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n))
        ci = [float(max(0.0, c - h)), float(min(1.0, c + h))]
    return dict(
        jitter_deg=PLANTED_GATE_JITTER_DEG, target_detection=PLANTED_GATE_DETECTION,
        detection_at_gate_jitter=None if at_gate is None else float(at_gate),
        detection_at_zero_jitter=(float(rows["jitter_0deg"]["detection_rate"])
                                  if "jitter_0deg" in rows else None),
        n_scorable_at_gate_jitter=int(n_gate),
        detection_at_gate_jitter_wilson95=ci,
        reading_A_pass=bool(at_gate is not None and at_gate >= PLANTED_GATE_DETECTION),
        reading_A_marginal=bool(ci is not None
                                and ci[0] < PLANTED_GATE_DETECTION < ci[1]),
        reading_B_pass=bool(any(x >= PLANTED_GATE_DETECTION for x in below)),
        jitter_at_80pct_power_grid=curve["jitter_at_80pct_power_grid"],
        reading=("READING A (PRIMARY, gate-bearing): detection is still >= "
                 f"{PLANTED_GATE_DETECTION:.0%} AT {PLANTED_GATE_JITTER_DEG} deg jitter. "
                 "READING B (co-reported): detection reaches the target at SOME jitter "
                 f"<= {PLANTED_GATE_JITTER_DEG} deg. The Wilson 95% interval on the "
                 "gate-jitter detection rate is reported so a Monte-Carlo-MARGINAL call "
                 f"(the frozen {N_PLANTED_DRAWS}-draw grid gives sd ~ 0.025 near 0.85) is "
                 "visible as marginal rather than read as a clean fail"))


def supervised_cell(Dmat, member_pos, label_index, full_k, strings, seed, n_perm, n_bg,
                    bg_weights, tag):
    """screen + matched background + ordering (internal null and the R4 block-perm null)."""
    pos = np.asarray(member_pos)
    s = screen(Dmat[pos], n_perm, seed=0)
    if s is None:
        return dict(tag=tag, scorable=False, reason="degenerate point set")
    bg = background_for_size(Dmat, len(Dmat), len(pos), seed=seed, n_draws=n_bg,
                             weights=bg_weights)
    xy, ratios, _ = pca_2d(Dmat[pos])
    ordr = score_ordering(xy, list(range(len(pos))), label_index, full_k, seed, n_perm=n_perm)
    ang = ordr.pop("_angles_rad")
    bp = block_perm_ordering(ang, label_index, full_k, resolve_blocks(strings), n_perm, seed)
    ctrl = shuffled_label_control(xy, list(range(len(pos))), label_index, full_k, seed,
                                  ordr["rho"],
                                  n_shuffles=min(N_SHUFFLE_LABELS, max(n_perm // 20, 50)))
    return dict(tag=tag, scorable=True, screening=s,
                p_bg=p_bg_of(bg, s["max_gap_deg"]),
                background={k: v for k, v in bg.items() if k != "_gaps"},
                ordering=ordr, ordering_block_perm=bp, shuffled_labels_control=ctrl,
                top2_var=float(ratios[:2].sum()),
                supervision_note=("SUPERVISED ANCHOR, never discovery: the member set is "
                                  "selected by name. Addendum F.5 demotes it to anchor "
                                  "status; the blind arm is primary"))


def blind_arm(Dmat, n_types, layer, arm, n_perm, n_bg, counts_pool, feats, pool_ids,
              n_null_freq, excluded_pos):
    """The scripts/22 centred-cosine + Leiden pipeline, unchanged, on one delta matrix."""
    t0 = time.time()
    S = cosine_matrix(Dmat, True)
    rows, tau, rule, clusters = sweep_and_select(S, GRID_CENTRED, n_types)
    ref, rlog = refine(clusters, S, tau) if tau is not None else ([], [])
    del S
    gc.collect()
    parts = {"centred_leiden": clusters, "centred_refined": ref}
    sizes_needed = sorted({len(c) for cl in parts.values() for c in cl
                           if SCREEN_MIN <= len(c) <= SCREEN_MAX})
    bg = {k: background_for_size(Dmat, n_types, k, seed=10_000 * layer + k, n_draws=n_bg)
          for k in sizes_needed}
    out, all_rows = {}, []
    for pname, cl in parts.items():
        srows = []
        for ci, c in enumerate(cl):
            if not (SCREEN_MIN <= len(c) <= SCREEN_MAX):
                continue
            s = screen(Dmat[np.asarray(c)], n_perm, seed=0)
            if s is None:
                continue
            s.update(layer=layer, arm=arm, partition=pname, cluster_id=ci,
                     member_hash=member_hash(c),
                     p_bg=p_bg_of(bg[len(c)], s["max_gap_deg"]),
                     background_rate_p_closed_lt_05=bg[len(c)]["rate_p_closed_lt_05"],
                     background_mean_max_gap_deg=bg[len(c)]["mean_max_gap_deg"],
                     members_pool_idx=[int(x) for x in c],
                     members_token_ids=[int(pool_ids[i]) for i in c],
                     seed=cid_seed(layer, f"{arm}|{pname}", ci))
            s["flagged"] = bool(s["p_closed"] < FLAG_ALPHA and s["p_bg"] is not None
                                and s["p_bg"] <= FLAG_ALPHA)
            srows.append(s)
        if srows:
            keep = benjamini_hochberg([r["p_closed"] for r in srows], alpha=BH_Q)
            for r, kk in zip(srows, keep):
                r["bh_pass_within_layer_arm"] = bool(kk)
        out[pname] = dict(
            selected_tau=tau, selection_rule=rule,
            sweep=rows if pname == "centred_leiden" else None,
            refinement_log=rlog if pname == "centred_refined" else None,
            n_clusters=len(cl), n_screened=len(srows),
            n_flagged=int(sum(r["flagged"] for r in srows)),
            expected_false_flags=float(sum((bg[r["k"]]["rate_p_closed_lt_05"] or 0.0)
                                           for r in srows)),
            size_histogram_top=sorted((len(c) for c in cl), reverse=True)[:10],
            screened=srows)
        all_rows.extend(srows)

    audits = []
    for r in all_rows:
        if not r["flagged"]:
            continue
        mem = np.asarray(r["members_pool_idx"])
        loo = []
        for drop in range(len(mem)):
            s = screen(Dmat[np.delete(mem, drop)], min(n_perm, N_PERM_LOO), seed=0)
            loo.append(dict(dropped_pool_idx=int(mem[drop]),
                            dropped_token_id=int(pool_ids[mem[drop]]),
                            p_closed=None if s is None else s["p_closed"],
                            gap_ratio=None if s is None else s["gap_ratio"]))
        loo_pass = bool(loo) and all(x["p_closed"] is not None and x["p_closed"] < ALPHA
                                     for x in loo)
        xy, _, _ = pca_2d(Dmat[mem])
        cx, cy, rr, _ = fit_circle(xy)
        ang = angles(xy, cx, cy)
        blocks = [f"lead={bool(feats[i, 0])}|cap={bool(feats[i, 1])}" for i in mem]
        freq = block_perm_spearman(ang, np.log10(np.maximum(counts_pool[mem], 1.0)),
                                   blocks, n_perm, seed=10_000 * layer + int(mem[0]))
        fp = freq_pseudo_null(Dmat, mem, counts_pool,
                              set(excluded_pos) | {int(x) for x in mem},
                              n_null_freq, n_perm, seed=860_000 + 1000 * layer + int(mem[0]))
        r["not_reportable"] = bool(not loo_pass)
        audits.append(dict(member_hash=r["member_hash"], partition=r["partition"],
                           cluster_id=r["cluster_id"], k=r["k"],
                           members_pool_idx=r["members_pool_idx"],
                           members_token_ids=r["members_token_ids"],
                           loo_jackknife=dict(all_pass=loo_pass, subsets=loo),
                           frequency_nuisance=freq, frequency_pseudo_null=fp,
                           token_class_composition=class_composition(feats, mem, n_types),
                           angles_deg=[float(x) for x in np.degrees(ang) % 360],
                           counts=[int(counts_pool[i]) for i in mem],
                           bh_pass_within_layer_arm=r.get("bh_pass_within_layer_arm"),
                           not_reportable=bool(not loo_pass),
                           reportable=bool(loo_pass),
                           reportability_rule=("ADDENDUM H: a BH passer whose leave-one-out "
                                               "jackknife fails is NOT reportable "
                                               "(verify41 precedent)")))
    out["passer_audits"] = audits
    out["n_flagged_total"] = int(len(audits))
    out["n_not_reportable_failed_loo"] = int(sum(a["not_reportable"] for a in audits))
    out["reportable_passers"] = [a["member_hash"] for a in audits if a["reportable"]]
    out["seconds"] = round(time.time() - t0, 1)
    out["background_by_size"] = {str(k): {kk: vv for kk, vv in b.items() if kk != "_gaps"}
                                 for k, b in bg.items()}
    return out


# ===========================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="cache/sparse_big")
    ap.add_argument("--layers", default="0,1,2,3,4,5,6,7,8,9,10")
    ap.add_argument("--l0", default="big")
    ap.add_argument("--out", default="results/phase3_lag_action_robust.json")
    ap.add_argument("--deltas-out",
                    default="results/phase3_lag_action_robust_deltas.npz")
    ap.add_argument("--deltas-layers", default="6,7")
    ap.add_argument("--null-layers", default=NULL_LAYERS_DEFAULT)
    ap.add_argument("--partial-dir", default=None)
    ap.add_argument("--max-rows", type=int, default=0,
                    help="SMOKE ONLY: truncate the cache pass to this many rows")
    ap.add_argument("--n-perm", type=int, default=N_PERM_CLUSTER)
    ap.add_argument("--n-bg", type=int, default=N_BACKGROUND_DRAWS)
    ap.add_argument("--n-null", type=int, default=N_NULL_COMPOSITION)
    ap.add_argument("--n-null-swap", type=int, default=N_NULL_PREDSWAP)
    ap.add_argument("--n-null-freq", type=int, default=N_NULL_FREQPSEUDO)
    ap.add_argument("--n-planted", type=int, default=N_PLANTED_DRAWS)
    ap.add_argument("--chunk", type=int, default=CHUNK,
                    help="streaming chunk in rows; the frozen value is scripts/36's "
                         "200,000. Lowering it only trades peak RSS for wall time and "
                         "is recorded in runtime_overrides")
    ap.add_argument("--blind-below-gate", action="store_true",
                    help="run an arm even when it is under the reliability gate "
                         "(diagnostic only; such an arm still issues no verdict)")
    a = ap.parse_args()
    t_start = time.time()
    smoke = a.max_rows > 0
    layers = [int(x) for x in a.layers.split(",") if x != ""]
    null_layers = [int(x) for x in a.null_layers.split(",") if x != ""]
    deltas_layers = [int(x) for x in a.deltas_layers.split(",") if x != ""]

    # ---- R1: declaration on disk BEFORE any cache row is read ----
    declared_at = time.strftime("%Y-%m-%dT%H:%M:%S")
    out_path = Path(a.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rules = declared_rules(dict(vars(a)))
    out_path.write_text(json.dumps(dict(
        status="DECLARATION STUB -- rules frozen, no results yet",
        script="scripts/41b_lag_action_robust.py",
        spec=ADDENDUM_H_DOC, addendum_H_text=ADDENDUM_H_TEXT,
        addendum_H1_erratum_doc=ADDENDUM_H1_DOC,
        addendum_H1_erratum_text=ADDENDUM_H1_TEXT,
        declared_at=declared_at, layers=layers, smoke=smoke,
        declared_rules=rules), indent=2))
    print("=== ADDENDUM H (frozen 2026-08-22), VERBATIM -- the text this script "
          "implements ===", flush=True)
    print(ADDENDUM_H_TEXT, flush=True)
    print("=== ADDENDUM H1 ERRATUM (frozen 2026-08-22), VERBATIM -- corrects H's "
          "provenance paragraph and adds the co-reported sensitivity ladder ===",
          flush=True)
    print(ADDENDUM_H1_TEXT, flush=True)
    print("=== ADDENDUM F (frozen 2026-08-22), everything Addendum H does not change ===",
          flush=True)
    print(ADDENDUM_F_TEXT, flush=True)
    print("=== DECLARED RULES (on disk before any cache row is read) ===", flush=True)
    print(json.dumps(rules, indent=2), flush=True)
    print(f"\ndeclared_at = {declared_at}   smoke = {smoke}   layers = {layers}   "
          f"null_layers = {null_layers}   deltas_layers = {deltas_layers}", flush=True)

    root = Path(a.root)
    if not root.is_absolute():
        root = REPO / root
    partial = Path(a.partial_dir) if a.partial_dir else None
    if partial:
        partial.mkdir(parents=True, exist_ok=True)

    # ---------------- pool and pair bookkeeping (shared across layers) ----------------
    tok, cnt, pool_ids = build_pool(root, layers[0])
    n_rows_cache = len(tok)
    n_types = len(pool_ids)
    print(f"\ncache rows {n_rows_cache}  sequences {n_rows_cache // SEQ_LEN}  "
          f"pool types {n_types} at floor {COUNT_FLOOR}", flush=True)
    assert n_rows_cache == N_ROWS_EXPECTED, f"row count {n_rows_cache}"
    assert n_rows_cache % SEQ_LEN == 0, "cache rows not a multiple of 255"
    assert n_types == N_POOL_EXPECTED, f"pool size {n_types}"
    same = {}
    for L in layers:
        t2 = np.fromfile(root / f"layer_{L}" / "tokens.i32", dtype=np.int32)
        same[str(L)] = bool(len(t2) == len(tok) and np.array_equal(t2, tok))
        del t2
    assert all(same.values()), f"tokens.i32 differs across layers: {same}"

    pool_pos = {int(t): i for i, t in enumerate(pool_ids)}
    counts_pool = cnt[pool_ids].astype(np.int64)
    lut = np.full(int(tok.max()) + 1, -1, dtype=np.int64)
    lut[pool_ids] = np.arange(n_types)

    from transformers import AutoTokenizer
    from gfdisc.env import MODEL_NAME
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=os.environ.get("HF_TOKEN"))
    pool_strings = [type_string(tokenizer, t) for t in pool_ids]
    feats = np.asarray([class_features(s) for s in pool_strings], dtype=np.float64)
    print(f"lexical classes built for {n_types} pool types; column sums "
          f"{[int(x) for x in feats.sum(0)]}", flush=True)

    n_used = min(a.max_rows, n_rows_cache) if smoke else n_rows_cache
    own, parity, per_lag = pair_index(tok, n_used, lut)

    pair_stats, C = {}, {}
    for lag in LAGS:
        pl = per_lag[lag]
        n_pred, n_acc = int(pl["pred_ok"].sum()), int(pl["succ_ok"].sum())
        Cl = np.zeros((2, n_types, n_types), dtype=np.int64)
        for p in (0, 1):
            m = pl["succ_ok"] & (parity == p)
            np.add.at(Cl[p], (pl["prev"][m], own[m]), 1)
            del m
        assert int(Cl.sum()) == n_acc, f"C(+{lag}) total disagrees with the pair mask"
        C[lag] = Cl
        pair_stats[str(lag)] = dict(
            n_pairs_valid_predecessor=n_pred, n_pairs_accumulated=n_acc,
            dropped_pair_fraction=float(1.0 - n_acc / n_pred) if n_pred else None,
            n_pairs_even=int(Cl[0].sum()), n_pairs_odd=int(Cl[1].sum()))
        print(f"lag +{lag}: valid-predecessor pairs {n_pred}, accumulating {n_acc}, "
              f"dropped (successor out of pool) fraction "
              f"{pair_stats[str(lag)]['dropped_pair_fraction']}", flush=True)

    pred_par = {lag: np.where(per_lag[lag]["succ_ok"],
                              per_lag[lag]["prev"] + parity.astype(np.int32) * n_types,
                              np.int32(-1)).astype(np.int32) for lag in LAGS}
    rs = {lag: C[lag].sum(axis=2) for lag in LAGS}          # [2, n_types]
    included = {f"lag{lag}": (rs[lag][0] >= MIN_PAIRS_PER_HALF)
                & (rs[lag][1] >= MIN_PAIRS_PER_HALF) for lag in LAGS}
    included["diff"] = included["lag1"] & included["lag2"]
    for k, v in included.items():
        print(f"included types [{k}]: {int(v.sum())}/{n_types}", flush=True)
    del parity
    gc.collect()

    # ---------------- labels: resolved once, used only where declared ----------------
    token_map = json.loads((REPO / "results/token_map.json").read_text())
    pool_set = set(int(x) for x in pool_ids)
    s12, missing_months = strict12(cnt, pool_set, token_map)
    assert sorted(s12) == STRICT_12_EXPECTED and not missing_months, \
        f"strict-12 resolved to {sorted(s12)}, expected {STRICT_12_EXPECTED}"
    s12_pos = np.asarray([pool_pos[t] for t in s12])            # calendar order
    m13_pos = np.asarray(sorted(pool_pos[t] for t in MEMBERS_13))
    m14_pos = np.asarray(sorted(pool_pos[t] for t in MEMBERS_14 if t in pool_pos))
    h13, h14 = member_hash(m13_pos), member_hash(m14_pos)
    assert h13 == HASH_13, f"13-set member_hash {h13} != {HASH_13}"
    assert h14 == HASH_14, f"14-set member_hash {h14} != {HASH_14}"
    print(f"\nfrozen sets resolved: 13-set hash {h13}, 14-set hash {h14} "
          f"(k={len(m14_pos)}), strict-12 {sorted(s12)}", flush=True)
    id_of_pos = {int(p): int(pool_ids[p]) for p in m13_pos}
    month_index_of_id = {int(t): i for i, t in enumerate(s12)}
    month_index_of_id[5656] = 1                                 # " Feb" -> February
    lab13 = [month_index_of_id[id_of_pos[int(p)]] for p in m13_pos]
    lab12 = list(range(12))
    calendar_ids = set()
    for grp in ("months", "days"):
        for v in token_map.get(grp, {}).values():
            calendar_ids |= {int(x) for x in v}
    calendar_ids |= set(MEMBERS_13) | set(MEMBERS_14)
    excluded_pos = {pool_pos[t] for t in calendar_ids if t in pool_pos}
    month_pool_idx = s12_pos.astype(np.int64)
    month_rank = np.full(n_types, -1, dtype=np.int64)
    month_rank[month_pool_idx] = np.arange(12)

    # ---- ADDENDUM H control sets (labelled, attached to the supervised ANCHOR arm) ----
    ord_ids, ord_idx, ord_unresolved = resolve_ordinal_words(tokenizer, pool_pos)
    day_ids, day_idx, day_unresolved = resolve_weekdays(cnt, pool_set, token_map)
    control_defs = {}
    if len(ord_ids) >= 3:
        control_defs["ordinal_numwords"] = dict(
            token_ids=[int(x) for x in ord_ids],
            pool_pos=np.asarray([pool_pos[t] for t in ord_ids]),
            label_index=[int(i) for i in ord_idx], full_k=len(ORDINAL_WORDS),
            words=[ORDINAL_WORDS[i] for i in ord_idx], unresolved=ord_unresolved,
            seed_base=612_000,
            note=("prereg 3.1's number-word ordinal control. The leading-space digits "
                  "' 0'..' 9' are not single tokens in this vocabulary and are not used"))
    if len(day_ids) >= 3:
        control_defs["weekdays"] = dict(
            token_ids=[int(x) for x in day_ids],
            pool_pos=np.asarray([pool_pos[t] for t in day_ids]),
            label_index=[int(i) for i in day_idx], full_k=len(DAYS),
            words=[DAYS[i] for i in day_idx], unresolved=day_unresolved,
            seed_base=613_000,
            note=("highest-count in-pool id per day from results/token_map.json 'days' -- "
                  "the strict12 rule applied to weekdays"))
    print(f"\n[Addendum H controls] ordinal number words resolved "
          f"{[ORDINAL_WORDS[i] for i in ord_idx]} -> {[int(x) for x in ord_ids]} "
          f"(unresolved {[u['word'] for u in ord_unresolved]}); weekdays "
          f"{[DAYS[i] for i in day_idx]} -> {[int(x) for x in day_ids]}", flush=True)

    # ---- per-lag row bookkeeping for nulls (i) and (ii) ----
    bks, cells_by_lag = {}, {}
    for lag in LAGS:
        bks[lag] = lag_bookkeeping(own, per_lag[lag]["prev"], per_lag[lag]["succ_ok"],
                                   n_types, n_used, month_rank)
        C_month = (C[lag][0] + C[lag][1])[month_pool_idx]
        cells_by_lag[lag] = composition_cells(month_pool_idx, C_month, bks[lag])
        print(f"lag +{lag} null bookkeeping: {bks[lag]['n_month_pairs']} month pairs over "
              f"{len(bks[lag]['m_group_seq'])} sequences; composition cells "
              f"{cells_by_lag[lag]['n_cells']} ({cells_by_lag[lag]['n_pairs_per_draw']} "
              f"rows/draw), {len(cells_by_lag[lag]['skipped'])} skipped", flush=True)
    del per_lag
    gc.collect()

    def bg_weights_factory(member_pos):
        def w(rng, k):
            return frequency_matched_resample(np.asarray(member_pos), counts_pool, rng,
                                              tol=FREQ_MATCH_TOL)
        return w

    # ---------------------------------------------------------------- per layer ----
    layer_out, delta_store = [], {}
    for L in layers:
        tL = time.time()
        print(f"\n================ layer {L} ================", flush=True)
        meta = json.loads((root / f"layer_{L}" / "meta.json").read_text())
        d_sae = int(meta["d_sae"])
        assert d_sae == D_SAE, f"d_sae {d_sae}"

        t_stream = time.time()
        TM_acc, S, seen, nnz_total = stream(root, L, own, pred_par, n_used, n_types,
                                            d_sae, chunk=a.chunk)
        stream_seconds = time.time() - t_stream
        print(f"stream done in {stream_seconds:.1f}s, {nnz_total} nonzeros", flush=True)
        if not smoke:
            assert np.array_equal(seen, counts_pool), \
                "streamed per-type row counts differ from the full-corpus counts"
        TM_acc /= np.maximum(seen, 1)[:, None]
        TM_acc[seen == 0] = 0.0

        sae = load_sae(L, a.l0, token=os.environ.get("HF_TOKEN"))
        W = get_decoder(sae); del sae
        assert W.shape == (d_sae, D_MODEL), f"w_dec shape {W.shape}"
        TMP = project(TM_acc, W)                       # [n_types, 1152] type means

        # ---------- positive control (a): plain type-mean anchor ----------
        anchor = {}
        for setname, pos in (("13-set", m13_pos), ("14-set", m14_pos)):
            anchor[setname] = dict(member_hash=member_hash(pos), k=int(len(pos)),
                                   screening=screen(TMP[pos], a.n_perm, seed=0))
        anchor_check = None
        if L in ANCHOR_TABLE:
            setname, exp_gr, exp_p, exp_hash = ANCHOR_TABLE[L]
            got = anchor[setname]["screening"]
            ok = bool(got is not None
                      and abs(got["gap_ratio"] - exp_gr) <= ANCHOR_TOL_GAP_RATIO
                      and anchor[setname]["member_hash"] == exp_hash)
            anchor_check = dict(layer=L, member_set=setname, expected_gap_ratio=exp_gr,
                                observed_gap_ratio=None if got is None else got["gap_ratio"],
                                expected_p_closed=exp_p,
                                observed_p_closed=None if got is None else got["p_closed"],
                                expected_member_hash=exp_hash,
                                observed_member_hash=anchor[setname]["member_hash"],
                                tolerance=ANCHOR_TOL_GAP_RATIO, passed=ok,
                                asserted=bool(not smoke))
            print(f"[anchor L{L} {setname}] gap_ratio "
                  f"{None if got is None else got['gap_ratio']} vs {exp_gr}; p_closed "
                  f"{None if got is None else got['p_closed']} vs {exp_p}; pass={ok}",
                  flush=True)
            if not smoke:
                assert ok, (f"POSITIVE CONTROL (a) FAILED at L{L}: INSTRUMENT NOT VALIDATED "
                            f"(prereg R2). observed gap_ratio "
                            f"{None if got is None else got['gap_ratio']} vs expected "
                            f"{exp_gr} at tolerance {ANCHOR_TOL_GAP_RATIO}")

        # ---------- identity removal, blocked over rows (Addendum F.3) ----------
        t_mm = time.time()
        for lag in LAGS:
            for p in (0, 1):
                base = p * n_types
                for lo in range(0, n_types, 512):
                    hi = min(lo + 512, n_types)
                    S[lag][base + lo:base + hi] -= C[lag][p][lo:hi].astype(np.float64) @ TM_acc
        matmul_seconds = time.time() - t_mm
        print(f"identity removal (C @ TM, both lags, both halves) in {matmul_seconds:.1f}s",
              flush=True)
        del TM_acc
        gc.collect()

        # ---------- per-type mean context effect, projected ----------
        half = {}
        for lag in LAGS:
            for p in (0, 1):
                S[lag][p * n_types:(p + 1) * n_types] /= np.maximum(rs[lag][p], 1)[:, None]
            half[(f"lag{lag}", 0)] = project(S[lag][:n_types], W).astype(np.float64)
            half[(f"lag{lag}", 1)] = project(S[lag][n_types:], W).astype(np.float64)
            S[lag] = None
            gc.collect()
            w0 = rs[lag][0].astype(np.float64)[:, None]
            w1 = rs[lag][1].astype(np.float64)[:, None]
            half[(f"lag{lag}", "full")] = ((w0 * half[(f"lag{lag}", 0)]
                                            + w1 * half[(f"lag{lag}", 1)])
                                           / np.maximum(w0 + w1, 1))
        del S
        gc.collect()
        for key in (0, 1, "full"):
            half[("diff", key)] = half[("lag1", key)] - half[("lag2", key)]
        n_zero_rows = {arm: int((np.linalg.norm(half[(arm, "full")], axis=1) == 0).sum())
                       for arm in ARMS}

        # ---------- reliability gate (Addendum F.2) ----------
        gates = {}
        for arm in ARMS:
            inc = included[arm]
            Pe, Po = half[(arm, 0)], half[(arm, 1)]
            cos_unc = rowwise_cosine(Pe, Po)
            Ce = Pe - Pe[inc].mean(0, keepdims=True) if inc.any() else Pe
            Co = Po - Po[inc].mean(0, keepdims=True) if inc.any() else Po
            cos_cen = rowwise_cosine(Ce, Co)
            ci, ui = cos_cen[inc], cos_unc[inc]
            med = float(np.median(ci)) if len(ci) else float("nan")
            gates[arm] = dict(
                centred_median=med,
                uncentred_median=float(np.median(ui)) if len(ui) else None,
                centred_p25=float(np.percentile(ci, 25)) if len(ci) else None,
                centred_p75=float(np.percentile(ci, 75)) if len(ci) else None,
                n_included=int(inc.sum()), threshold=GATE_THRESHOLD,
                gate="PASS" if med >= GATE_THRESHOLD else "FAIL",
                month_centred_median=float(np.median(cos_cen[month_pool_idx])),
                n_zero_rows=n_zero_rows[arm])
            print(f"[gate L{L} {arm}] centred median {med:.5f} ({gates[arm]['gate']}); "
                  f"uncentred {gates[arm]['uncentred_median']}", flush=True)
        fuse_check = None
        if L == 6:
            got = gates["lag1"]["centred_median"]
            ok = bool(abs(got - FUSE_L6_LAG1_CENTRED) <= FUSE_TOL)
            fuse_check = dict(expected=FUSE_L6_LAG1_CENTRED, observed=got,
                              tolerance=FUSE_TOL, passed=ok, asserted=bool(not smoke),
                              source="results/phase3_lag_fuse.json",
                              lag2_observed=gates["lag2"]["centred_median"],
                              lag2_verify36_reported=FUSE_L6_LAG2_VERIFY36,
                              lag2_note="PRINT-ONLY comparison, different code path, no assert")
            print(f"[fuse reproduction L6 lag1] {got} vs {FUSE_L6_LAG1_CENTRED} "
                  f"(tol {FUSE_TOL}) pass={ok}", flush=True)
            print(f"[verify36 comparison L6 lag2] observed "
                  f"{gates['lag2']['centred_median']} vs reported "
                  f"{FUSE_L6_LAG2_VERIFY36} -- PRINT ONLY, no assert", flush=True)
            if not smoke:
                assert ok, (f"L6 lag1 centred split-half median {got} does not reproduce "
                            f"results/phase3_lag_fuse.json {FUSE_L6_LAG1_CENTRED}")

        # ====== ADDENDUM H, THE ONE ADDED RULE: label-free outlier-coordinate drop ======
        # Computed PER ARM on that arm's own projected halves and its own included mask.
        # Nothing above this point sees the drop: positive control (a), the reliability
        # gate and the fuse assert are all full-coordinate statistics by declaration.
        drop = {}
        halfD = {}          # dropped-coordinate matrices -- everything downstream uses these
        for arm in ARMS:
            dr = outlier_drop_rule(half[(arm, 0)], half[(arm, 1)], included[arm])
            keep, sd_pool = dr.pop("keep"), dr.pop("pool_noise_sd")
            drop[arm] = dict(dr)
            drop[arm]["_keep"] = keep
            drop[arm]["_pool_noise_sd"] = sd_pool
            for key in (0, 1, "full"):
                halfD[(arm, key)] = np.ascontiguousarray(half[(arm, key)][:, keep])
            print(f"[DROP L{L} {arm}] dropped coordinates "
                  f"{drop[arm]['dropped_coordinates']} "
                  f"(shares {[round(x, 4) for x in drop[arm]['dropped_shares']]}) of "
                  f"{drop[arm]['d_before']}; kept {drop[arm]['n_kept']}; "
                  f"top5 by pool-wide split-half sd "
                  f"{[(c['coord'], round(c['variance_share'], 4)) for c in drop[arm]['top5_by_sd']]}",
                  flush=True)

        # post-drop reliability medians -- DIAGNOSTIC ONLY, they decide nothing
        for arm in ARMS:
            inc = included[arm]
            Pe, Po = halfD[(arm, 0)], halfD[(arm, 1)]
            if inc.any():
                Ce = Pe - Pe[inc].mean(0, keepdims=True)
                Co = Po - Po[inc].mean(0, keepdims=True)
                cc = rowwise_cosine(Ce, Co)[inc]
                gates[arm]["post_drop_centred_median_DIAGNOSTIC"] = (
                    float(np.median(cc)) if len(cc) else None)
                gates[arm]["post_drop_uncentred_median_DIAGNOSTIC"] = float(
                    np.median(rowwise_cosine(Pe, Po)[inc]))
            else:
                gates[arm]["post_drop_centred_median_DIAGNOSTIC"] = None
                gates[arm]["post_drop_uncentred_median_DIAGNOSTIC"] = None
            gates[arm]["gate_computed_on"] = (
                "FULL 1152 coordinates, as in scripts/41 -- the post-drop medians beside it "
                "are diagnostic and are not compared to the threshold")

        # ---------- positive control (b): planted ring, POST-DROP, per arm ----------
        planted = {}
        for ai, arm in enumerate(ARMS):
            inc = included[arm]
            keep = drop[arm]["_keep"]
            nk = int(drop[arm]["n_kept"])
            norms = np.linalg.norm(halfD[(arm, "full")][inc], axis=1)
            radius = float(np.median(norms) * PLANTED_RADIUS_FRACTION) if len(norms) else 0.0
            norms_pre = np.linalg.norm(half[(arm, "full")][inc], axis=1)
            radius_pre = (float(np.median(norms_pre) * PLANTED_RADIUS_FRACTION)
                          if len(norms_pre) else 0.0)
            noise = (halfD[(arm, 0)][month_pool_idx]
                     - halfD[(arm, 1)][month_pool_idx]) / np.sqrt(2)
            sd = np.sqrt((noise ** 2).mean(0))                 # month-row sd, kept coords
            noise_pre = (half[(arm, 0)][month_pool_idx]
                         - half[(arm, 1)][month_pool_idx]) / np.sqrt(2)
            sd_pre = np.sqrt((noise_pre ** 2).mean(0))
            xy_m, _, _ = pca_2d(halfD[(arm, "full")][month_pool_idx])
            r_inplane = float(np.median(np.linalg.norm(xy_m, axis=1)))
            pr = planted_ring_power(
                radius, sd, a.n_planted, a.n_perm, d=nk, seed_base=850_000 + 10_000 * ai,
                label=("FROZEN primary, POST-DROP (Addendum H): declared radius on the "
                       "kept coordinates, split-half noise (A_even - A_odd)/sqrt(2) "
                       "restricted to the kept coordinates -- the ONE-HALF scale"))
            sec = planted_ring_power(
                radius, sd / np.sqrt(2), a.n_planted, a.n_perm, d=nk,
                seed_base=851_000 + 10_000 * ai,
                label="co-report: declared radius, full-data-matched noise (sd / sqrt(2))")
            ter = planted_ring_power(
                r_inplane, sd / np.sqrt(2), a.n_planted, a.n_perm, d=nk,
                seed_base=852_000 + 10_000 * ai,
                label=("co-report: MEASURED in-plane radius of the 12 real month rows, "
                       "full-data-matched noise -- the amplitude-matched comparator"))
            qua = planted_ring_power(
                radius_pre, sd, a.n_planted, a.n_perm, d=nk,
                seed_base=853_000 + 10_000 * ai,
                label=("co-report: PRE-DROP radius with post-drop noise -- the strictly "
                       "literal 'only the noise model changes' reading of Addendum H"))
            gate = planted_gate_of(pr)
            cal = pr["jitter_at_80pct_power_grid"] is not None
            planted[arm] = dict(
                primary_half_sample_noise=pr, secondary_full_data_noise=sec,
                tertiary_measured_in_plane_radius=ter,
                quaternary_pre_drop_radius=qua,
                addendum_H_gate=gate,
                dropped_coordinates=drop[arm]["dropped_coordinates"],
                n_kept_coordinates=nk,
                radius_rule=(f"{PLANTED_RADIUS_FRACTION} x median POST-DROP ||delta|| over "
                             f"included pool types = {radius}"),
                diagnostics=dict(
                    median_delta_norm=float(np.median(norms)) if len(norms) else None,
                    median_delta_norm_pre_drop=(float(np.median(norms_pre))
                                                if len(norms_pre) else None),
                    declared_radius=radius, declared_radius_pre_drop=radius_pre,
                    measured_in_plane_radius_of_months=r_inplane,
                    declared_radius_over_measured_in_plane_radius=(
                        float(radius / r_inplane) if r_inplane > 0 else None),
                    noise_sd_l2_norm=float(np.linalg.norm(sd)),
                    noise_sd_l2_norm_pre_drop=float(np.linalg.norm(sd_pre)),
                    noise_sd_mean=float(sd.mean()) if nk else None,
                    noise_sd_max=float(sd.max()) if nk else None,
                    noise_sd_top5=[float(x) for x in np.sort(sd)[-5:][::-1]],
                    declared_radius_over_noise_norm=(
                        float(radius / np.linalg.norm(sd)) if np.linalg.norm(sd) > 0
                        else None)),
                negative_is_calibrated=bool(cal),
                calibrated_full_data_noise=bool(
                    sec["jitter_at_80pct_power_grid"] is not None),
                calibrated_measured_radius=bool(
                    ter["jitter_at_80pct_power_grid"] is not None),
                power_status=("reach = " + str(pr["jitter_at_80pct_power_grid"]) +
                              " deg jitter at 80% power (frozen primary curve)" if cal else
                              "NO POWER AT ANY JITTER on the frozen primary curve: a "
                              "negative from this arm at this layer is UNCALIBRATED"))
            print(f"[planted ring L{L} {arm}] POST-DROP d={nk} radius {radius:.5g} "
                  f"(pre-drop {radius_pre:.5g}, measured in-plane {r_inplane:.5g}), "
                  f"||sd|| {float(np.linalg.norm(sd)):.5g} "
                  f"(pre-drop {float(np.linalg.norm(sd_pre)):.5g}); detection "
                  f"0deg {gate['detection_at_zero_jitter']} / "
                  f"{PLANTED_GATE_JITTER_DEG}deg {gate['detection_at_gate_jitter']}; "
                  f"80%-power jitter grid: primary {pr['jitter_at_80pct_power_grid']}, "
                  f"full-data {sec['jitter_at_80pct_power_grid']}, measured-radius "
                  f"{ter['jitter_at_80pct_power_grid']}; Addendum H gate readingA "
                  f"{gate['reading_A_pass']} readingB {gate['reading_B_pass']}", flush=True)

        planted_gate_layer = dict(
            applies=bool(L in PLANTED_GATE_LAYERS),
            per_arm={arm: dict(
                reading_A_pass=planted[arm]["addendum_H_gate"]["reading_A_pass"],
                reading_A_marginal=planted[arm]["addendum_H_gate"]["reading_A_marginal"],
                reading_B_pass=planted[arm]["addendum_H_gate"]["reading_B_pass"],
                detection_at_0deg=planted[arm]["addendum_H_gate"]["detection_at_zero_jitter"],
                detection_at_gate_jitter=planted[arm]["addendum_H_gate"][
                    "detection_at_gate_jitter"],
                detection_at_gate_jitter_wilson95=planted[arm]["addendum_H_gate"][
                    "detection_at_gate_jitter_wilson95"]) for arm in ARMS},
            all_arms_reading_A=bool(all(planted[arm]["addendum_H_gate"]["reading_A_pass"]
                                        for arm in ARMS)),
            lag1_and_diff_reading_A=bool(
                planted["lag1"]["addendum_H_gate"]["reading_A_pass"]
                and planted["diff"]["addendum_H_gate"]["reading_A_pass"]),
            all_arms_reading_B=bool(all(planted[arm]["addendum_H_gate"]["reading_B_pass"]
                                        for arm in ARMS)),
            rule=(f"Addendum H R2 gate: at layers {list(PLANTED_GATE_LAYERS)} the frozen "
                  f"primary POST-DROP planted curve must reach "
                  f">= {PLANTED_GATE_DETECTION:.0%} detection at "
                  f"<= {PLANTED_GATE_JITTER_DEG} deg jitter, else the layer verdict is "
                  "'instrument not validated'. The gate does NOT stop the run"))
        planted_gate_layer["passed"] = bool(
            (not planted_gate_layer["applies"]) or planted_gate_layer["all_arms_reading_A"])
        print(f"[Addendum H planted gate L{L}] applies="
              f"{planted_gate_layer['applies']} passed={planted_gate_layer['passed']} "
              f"(all arms reading A {planted_gate_layer['all_arms_reading_A']}, "
              f"lag1+diff {planted_gate_layer['lag1_and_diff_reading_A']})", flush=True)

        # ====== ADDENDUM H1: declared sensitivity ladder, CO-REPORTED, NOT A VERDICT ====
        # Supervised strict-12 only, every arm, every layer, four fixed preprocessing
        # depths ranked by the same label-free pool-wide split-half noise share. Run here
        # rather than inside the arm loop so it exists at every layer regardless of the
        # Addendum F.2 gate, as H1 requires. Nothing below feeds any conjunction.
        t_lad = time.time()
        ladder = {}
        for ai, arm in enumerate(ARMS):
            sd_pool = drop[arm]["_pool_noise_sd"]
            h_keep = drop[arm]["_keep"]
            rungs = {}
            for ri, rung in enumerate(SENSITIVITY_RUNGS):
                M, keep, dropped, how = sensitivity_rung_matrix(
                    half[(arm, "full")], sd_pool, rung)
                cell = supervised_cell(
                    M, s12_pos, lab12, 12, [pool_strings[p_] for p_ in s12_pos],
                    seed=SENSITIVITY_SEED_BASE + 100_000 * ri + 1000 * L + ai,
                    n_perm=a.n_perm, n_bg=a.n_bg,
                    bg_weights=bg_weights_factory(s12_pos), tag=f"strict_12_{rung}")
                cl = closure_terms(cell)
                rungs[rung] = dict(
                    rung=rung, how=how, dropped_coordinates=dropped,
                    n_kept=int(keep.sum()),
                    equals_H_rule=bool(np.array_equal(keep, h_keep)),
                    closure=cl, cell=cell)
                del M
                print(f"[H1 ladder L{L} {arm} {rung:10s}] p_closed {cl['p_closed']} "
                      f"p_bg {cl['p_bg']} rho {cl['rho']} block-perm {cl['block_perm_p']} "
                      f"-> closes={cl['closes']} (equals_H_rule="
                      f"{rungs[rung]['equals_H_rule']})", flush=True)
            ladder[arm] = rungs
            gc.collect()
        ladder_seconds = round(time.time() - t_lad, 1)
        print(f"[H1 ladder L{L}] {len(ARMS) * len(SENSITIVITY_RUNGS)} cells in "
              f"{ladder_seconds}s", flush=True)

        # ---------- supervised anchor arm + nulls + blind arm, per delta arm ----------
        csr = (np.load(root / f"layer_{L}" / "indptr.npy"),
               np.memmap(root / f"layer_{L}" / "indices.u16", dtype=np.uint16, mode="r"),
               np.memmap(root / f"layer_{L}" / "values.f16", dtype=np.float16, mode="r"))
        arm_out = {}
        for ai, arm in enumerate(ARMS):
            entry = dict(arm=arm, reliability=gates[arm])
            if gates[arm]["gate"] != "PASS" and not a.blind_below_gate:
                entry.update(verdict="NO VERDICT -- below the reliability gate",
                             supervised=None, blind=None, nulls=None,
                             skipped_reason=(f"centred split-half median "
                                             f"{gates[arm]['centred_median']} < "
                                             f"{GATE_THRESHOLD} (Addendum F.2)"))
                entry["outlier_drop"] = {k: v for k, v in drop[arm].items()
                                         if not k.startswith("_")}
                entry["sensitivity_ladder"] = {
                    r: {k: v for k, v in ladder[arm][r].items() if k != "cell"}
                    for r in SENSITIVITY_RUNGS}
                entry["sensitivity_ladder_note"] = (
                    "ADDENDUM H1, CO-REPORTED, NOT VERDICT-BEARING. Computed at every "
                    "layer for every arm, independently of the reliability gate")
                arm_out[arm] = entry
                print(f"[L{L} {arm}] below the reliability gate -> no verdict, arm skipped",
                      flush=True)
                continue
            keep = drop[arm]["_keep"]
            Dmat = halfD[(arm, "full")]          # ADDENDUM H: dropped-coordinate matrix
            entry["outlier_drop"] = {k: v for k, v in drop[arm].items()
                                     if not k.startswith("_")}

            sup = {}
            for setname, pos, lab in (("strict_12", s12_pos, lab12),
                                      ("set_13", m13_pos, lab13)):
                sup[setname] = supervised_cell(
                    Dmat, pos, lab, 12, [pool_strings[p] for p in pos],
                    seed=610_000 + 1000 * L + ai, n_perm=a.n_perm, n_bg=a.n_bg,
                    bg_weights=bg_weights_factory(pos), tag=setname)

            # co-reported WHITENED arm (full coordinates / pool-wide noise sd), strict-12
            # only, never verdict-bearing (Addendum H). It is the ladder's own whitened
            # rung -- computed once, above, so the two can never disagree.
            sup["strict_12_whitened"] = dict(ladder[arm]["whitened"]["cell"])
            if isinstance(sup["strict_12_whitened"], dict):
                sup["strict_12_whitened"]["co_report_only"] = True
                sup["strict_12_whitened"]["whitening"] = (
                    "every coordinate divided by its pool-wide split-half noise sd "
                    f"(floor {WHITEN_FLOOR}); NO coordinate dropped; not verdict-bearing")
                sup["strict_12_whitened"]["source"] = (
                    "Addendum H1 sensitivity ladder, rung 'whitened' -- the identical cell")

            # ADDENDUM H control sets through the IDENTICAL arm, on the dropped matrix
            controls = {}
            for cname, cdef in control_defs.items():
                cpos = cdef["pool_pos"]
                cell = supervised_cell(
                    Dmat, cpos, cdef["label_index"], cdef["full_k"],
                    [pool_strings[p] for p in cpos],
                    seed=cdef["seed_base"] + 1000 * L + ai, n_perm=a.n_perm, n_bg=a.n_bg,
                    bg_weights=bg_weights_factory(cpos), tag=cname)
                cell["control_set"] = cname
                cell["token_ids"] = cdef["token_ids"]
                cell["token_strings"] = cdef["words"]
                cell["unresolved"] = cdef["unresolved"]
                cell["definition_note"] = cdef["note"]
                cell["closure"] = closure_terms(cell)
                cell["invalidity_trigger_if_closes"] = True
                controls[cname] = cell
                ct = cell["closure"]
                print(f"[L{L} {arm}] control {cname}: k={len(cpos)} p_closed "
                      f"{ct['p_closed']} p_bg {ct['p_bg']} rho {ct['rho']} block-perm "
                      f"{ct['block_perm_p']} -> closes={ct['closes']}", flush=True)

            U = Dmat / np.maximum(np.linalg.norm(Dmat, axis=1, keepdims=True), 1e-12)
            su = screen(U[s12_pos], a.n_perm, seed=0)
            bgu = background_for_size(U, n_types, 12, seed=870_000 + 1000 * L + ai,
                                      n_draws=a.n_bg, weights=bg_weights_factory(s12_pos))
            unit_norm = dict(screening=su,
                             p_bg=None if su is None else p_bg_of(bgu, su["max_gap_deg"]),
                             background={k: v for k, v in bgu.items() if k != "_gaps"},
                             closes=bool(su is not None and su["p_closed"] < ALPHA),
                             order_note=("ADDENDUM H: the coordinates are DROPPED FIRST and "
                                         "the rows are L2-normalised SECOND"))
            del U
            nulls = dict(
                iii_frequency_pseudo=freq_pseudo_null(
                    Dmat, s12_pos, counts_pool, excluded_pos, a.n_null_freq, a.n_perm,
                    seed=840_000 + 1000 * L + ai),
                iv_unit_norm=unit_norm)
            if L in null_layers:
                for kind, key, nd, sd_seed in (
                        ("composition", "i_composition_matched", a.n_null,
                         820_000 + 1000 * L + ai),
                        ("predswap", "ii_predecessor_swap", a.n_null_swap,
                         830_000 + 1000 * L + ai)):
                    t0 = time.time()
                    nulls[key] = run_resampling_null(kind, arm, bks, cells_by_lag, 12, own,
                                                     csr, W, TMP, d_sae, nd, a.n_perm,
                                                     sd_seed, keep=keep)
                    nulls[key]["seconds"] = round(time.time() - t0, 1)
                    print(f"[L{L} {arm}] {key}: closure rate "
                          f"{nulls[key]['closure_rate']} over {nulls[key]['n_scorable']} "
                          f"scorable draws in {nulls[key]['seconds']}s", flush=True)
            else:
                reason = (f"nulls (i) and (ii) run at the confirmatory layers "
                          f"{null_layers} only, as pre-declared")
                nulls["i_composition_matched"] = dict(not_run=True, reason=reason)
                nulls["ii_predecessor_swap"] = dict(not_run=True, reason=reason)

            s12s = sup["strict_12"]
            scr = s12s.get("screening")
            draw_nulls = [nulls["i_composition_matched"], nulls["ii_predecessor_swap"],
                          nulls["iii_frequency_pseudo"]]
            nulls_run = all(not n.get("not_run") for n in draw_nulls)
            nulls_ok = nulls_run and all(n.get("fails_to_close_in_at_least_95pct")
                                         for n in draw_nulls)
            screens_closed = bool(scr is not None and scr["p_closed"] < ALPHA
                                  and s12s.get("p_bg") is not None and s12s["p_bg"] < ALPHA)
            rho = s12s["ordering"]["rho"] if s12s.get("scorable") else None
            pblock = s12s["ordering_block_perm"]["p_block"] if s12s.get("scorable") else None
            closed = bool(screens_closed and rho is not None and rho >= RHO_THRESHOLD
                          and nulls_ok and unit_norm["closes"])
            calibrated = planted[arm]["negative_is_calibrated"]
            if gates[arm]["gate"] != "PASS":
                verdict = ("NO VERDICT -- below the reliability gate (Addendum F.2); this "
                           "arm ran only because --blind-below-gate was passed, and its "
                           "numbers are diagnostic")
            elif closed:
                verdict = "CLOSED"
            elif screens_closed and not nulls_run:
                verdict = "closure_flagged_nulls_not_run"
            elif calibrated:
                verdict = "NOT CLOSED"
            else:
                verdict = ("NOT CLOSED (negative UNCALIBRATED: the planted-ring positive "
                           "control has no power even at zero jitter)")
            # ---- Addendum H's own four-term closure, per arm, on the dropped matrix ----
            h_close = closure_terms(s12s)
            h_controls = {c: controls[c]["closure"] for c in controls}
            entry["sensitivity_ladder"] = {
                r: {k: v for k, v in ladder[arm][r].items() if k != "cell"}
                for r in SENSITIVITY_RUNGS}
            entry["sensitivity_ladder_cells"] = {
                r: ladder[arm][r]["cell"] for r in SENSITIVITY_RUNGS}
            entry["sensitivity_ladder_note"] = (
                "ADDENDUM H1, CO-REPORTED, NOT VERDICT-BEARING. The primary verdict is the "
                "frozen H rule in addendum_H_decision; a ring that appears only at deeper "
                "drops is drop-depth-dependent, not an action ring")
            entry.update(
                supervised=sup, controls=controls, nulls=nulls, unit_norm_arm=unit_norm,
                planted_ring_power=planted[arm],
                addendum_H_closure=dict(
                    strict_12=h_close,
                    strict_12_whitened_CO_REPORT=closure_terms(sup["strict_12_whitened"]),
                    nulls_i_ii_iii_all_fail_to_close=bool(nulls_ok),
                    nulls_run=bool(nulls_run),
                    unit_norm_closes=bool(unit_norm["closes"]),
                    four_F4_nulls_ok=bool(nulls_ok and unit_norm["closes"]),
                    controls=h_controls,
                    any_control_closes=bool(any(v["closes"] for v in h_controls.values())),
                    note=("closes = p_closed < %g AND p_bg < %g AND rho >= %g AND "
                          "block-perm p < %g, on the DROPPED-COORDINATE matrix. The "
                          "whitened cell is co-reported and enters no conjunction"
                          % (ALPHA, ALPHA, RHO_THRESHOLD, ALPHA))),
                closure_conjunction=dict(
                    p_closed=None if scr is None else scr["p_closed"],
                    p_bg=s12s.get("p_bg"), rho=rho, block_perm_p=pblock,
                    nulls_run=bool(nulls_run), nulls_all_fail_to_close=bool(nulls_ok),
                    unit_norm_closes=unit_norm["closes"], closed=closed,
                    negative_is_calibrated=bool(calibrated),
                    closed_with_ordering_null=bool(closed and pblock is not None
                                                   and pblock < ALPHA),
                    note=("this is Addendum F step 6's conjunction, carried forward for "
                          "term-by-term comparison with scripts/41; the LAYER verdict is "
                          "addendum_H_decision")),
                verdict=verdict)
            print(f"[L{L} {arm}] supervised strict-12 (post-drop): p_closed "
                  f"{None if scr is None else scr['p_closed']}, p_bg {s12s.get('p_bg')}, "
                  f"rho {rho}, block-perm {pblock} -> F-verdict {entry['verdict']}; "
                  f"H-closes {h_close['closes']}", flush=True)

            t0 = time.time()
            entry["blind"] = blind_arm(Dmat, n_types, L, arm, a.n_perm, a.n_bg, counts_pool,
                                       feats, pool_ids, a.n_null_freq, excluded_pos)
            nfl = sum(v["n_flagged"] for v in entry["blind"].values()
                      if isinstance(v, dict) and "n_flagged" in v)
            exf = sum(v["expected_false_flags"] for v in entry["blind"].values()
                      if isinstance(v, dict) and "expected_false_flags" in v)
            print(f"[L{L} {arm}] blind arm: {nfl} flagged vs {exf:.2f} expected false "
                  f"({round(time.time() - t0, 1)}s)", flush=True)
            arm_out[arm] = entry

        # ================= ADDENDUM H's LAYER DECISION (supervised, strict-12) =========
        def _h(arm, field, default=None):
            e = arm_out.get(arm) or {}
            h = e.get("addendum_H_closure") or {}
            return h.get(field, default)

        gates_ok = {arm: bool(gates[arm]["gate"] == "PASS") for arm in ARMS}
        decision_arms_gated = bool(gates_ok["lag1"] and gates_ok["lag2"]
                                   and gates_ok["diff"])
        c1 = (_h("lag1", "strict_12") or {}).get("closes", False)
        c2 = (_h("lag2", "strict_12") or {}).get("closes", False)
        cd = (_h("diff", "strict_12") or {}).get("closes", False)
        nulls_lag1 = bool(_h("lag1", "four_F4_nulls_ok", False))
        nulls_diff = bool(_h("diff", "four_F4_nulls_ok", False))
        nulls_run_l1 = bool(_h("lag1", "nulls_run", False))
        nulls_run_df = bool(_h("diff", "nulls_run", False))
        ctrl_hits = []
        for arm in ARMS:
            for cname, cc in (_h(arm, "controls") or {}).items():
                if cc.get("closes"):
                    ctrl_hits.append(dict(arm=arm, control=cname,
                                          p_closed=cc.get("p_closed"),
                                          p_bg=cc.get("p_bg"), rho=cc.get("rho"),
                                          block_perm_p=cc.get("block_perm_p")))
        invalid = [x for x in ctrl_hits if x["arm"] in ("lag1", "diff")]

        if not decision_arms_gated:
            h_verdict = "NO VERDICT -- an arm is below the Addendum F.2 reliability gate"
            h_reason = {arm: gates[arm]["centred_median"] for arm in ARMS}
        elif planted_gate_layer["applies"] and not planted_gate_layer["passed"]:
            h_verdict = "INSTRUMENT NOT VALIDATED"
            marg = [arm for arm in ARMS
                    if planted_gate_layer["per_arm"][arm]["reading_A_marginal"]]
            h_reason = dict(
                rule=("the frozen primary POST-DROP planted-ring curve does not reach "
                      f"{PLANTED_GATE_DETECTION:.0%} detection at "
                      f"{PLANTED_GATE_JITTER_DEG} deg jitter at this confirmatory layer "
                      "(Addendum H R2 gate); every number is still emitted"),
                failing_arms=[arm for arm in ARMS
                              if not planted_gate_layer["per_arm"][arm]["reading_A_pass"]],
                arms_marginal_by_wilson95=marg,
                reading_B_all_arms=planted_gate_layer["all_arms_reading_B"])
        elif invalid:
            h_verdict = "INVALID -- a control set closes through the identical arm"
            h_reason = invalid
        elif c1 and cd and (nulls_lag1 and nulls_diff) and not c2:
            h_verdict = "ACTION RING"
            h_reason = ("delta(+1) closes, the differenced arm closes, the four F.4 nulls "
                        "hold on both, and delta(+2) does NOT close (specificity)")
        elif c1 and cd and (nulls_lag1 and nulls_diff) and c2:
            h_verdict = "PERSISTENT-CONTEXT CARRY"
            h_reason = ("delta(+1) and the differenced arm both close but delta(+2) ALSO "
                        "closes: the specificity term fails (Addendum H, per F.6)")
        elif c1 and not cd:
            h_verdict = "PERSISTENT-CONTEXT CARRY"
            h_reason = ("the month ring is present in delta(+1) and absent in "
                        "delta(+1) - delta(+2) (Addendum F.6, carried into Addendum H)")
        elif (c1 or cd) and not (nulls_run_l1 and nulls_run_df):
            h_verdict = "closure_flagged_nulls_not_run"
            h_reason = ("nulls (i) and (ii) run at the confirmatory layers only; an "
                        "exploratory layer can never receive the Addendum H verdict")
        elif (c1 or cd) and not (nulls_lag1 and nulls_diff):
            h_verdict = "NO ACTION RING -- a null closes too often, or the unit-norm arm fails"
            h_reason = dict(nulls_ok_lag1=nulls_lag1, nulls_ok_diff=nulls_diff)
        else:
            uncal = not planted_gate_layer["per_arm"]["lag1"]["reading_B_pass"]
            h_verdict = ("NO ACTION RING (negative UNCALIBRATED: the planted-ring control "
                         "has no power even at zero jitter)" if uncal else "NO ACTION RING")
            h_reason = dict(lag1_closes=c1, lag2_closes=c2, diff_closes=cd)
        # ADDENDUM H1 reading: a ring that appears only at DEEPER drops than the frozen
        # rule is drop-depth-dependent, not an action ring. Annotation only -- it cannot
        # change h_verdict, which is already fixed above by the frozen H rule.
        h_rule_closes = {"lag1": bool(c1), "lag2": bool(c2), "diff": bool(cd)}
        ladder_closes = {arm: {r: bool(ladder[arm][r]["closure"]["closes"])
                               for r in SENSITIVITY_RUNGS} for arm in ARMS}
        deeper = ("drop_top2", "drop_top5", "whitened")
        depth_dependent = {arm: bool((not h_rule_closes[arm])
                                     and any(ladder_closes[arm][r] for r in deeper))
                           for arm in ARMS}
        ladder_summary = dict(
            h_rule_closes=h_rule_closes, ladder_closes=ladder_closes,
            drop_depth_dependent=depth_dependent,
            any_drop_depth_dependent=bool(any(depth_dependent.values())),
            first_closing_rung={arm: next((r for r in SENSITIVITY_RUNGS
                                           if ladder_closes[arm][r]), None)
                                for arm in ARMS},
            equals_H_rule={arm: {r: ladder[arm][r]["equals_H_rule"]
                                 for r in SENSITIVITY_RUNGS} for arm in ARMS},
            reading=("ADDENDUM H1: CO-REPORTED, NOT VERDICT-BEARING. Where "
                     "drop_depth_dependent is true the arm's ring appears only at a drop "
                     "depth beyond the frozen 25%-share rule and is reported as "
                     "DROP-DEPTH-DEPENDENT, not as an action ring. The verdict field of "
                     "this block is the frozen H rule and is unaffected by the ladder"))
        if ladder_summary["any_drop_depth_dependent"]:
            print(f"[H1 L{L}] DROP-DEPTH-DEPENDENT (not an action ring): "
                  f"{[k for k, v in depth_dependent.items() if v]}; first closing rung "
                  f"{ladder_summary['first_closing_rung']}", flush=True)
        addendum_H_decision = dict(
            layer=L, confirmatory=bool(L in CONFIRMATORY_LAYERS),
            verdict=h_verdict, reason=h_reason,
            addendum_H1_sensitivity_ladder=ladder_summary,
            terms=dict(
                delta_plus1_closes=bool(c1), delta_plus2_closes=bool(c2),
                differenced_closes=bool(cd),
                four_F4_nulls_ok_lag1=nulls_lag1, four_F4_nulls_ok_diff=nulls_diff,
                nulls_run_lag1=nulls_run_l1, nulls_run_diff=nulls_run_df,
                reliability_gate_pass={arm: gates_ok[arm] for arm in ARMS},
                planted_gate=planted_gate_layer,
                control_closures=ctrl_hits,
                invalidity_triggered=bool(invalid)),
            per_arm_closure={arm: _h(arm, "strict_12") for arm in ARMS},
            per_arm_dropped_coordinates={arm: drop[arm]["dropped_coordinates"]
                                         for arm in ARMS},
            rule=("ACTION RING iff delta(+1) closes AND delta(+1)-delta(+2) closes AND all "
                  "four F.4 nulls fail to close in >= 95% of draws (on both closing arms) "
                  "AND delta(+2) does NOT close; persistent-context carry if delta(+2) also "
                  "closes, or if the ring is in delta(+1) and absent in the differenced "
                  "arm. Ordinal (one..ten) and weekday control sets through the identical "
                  "arms must not close"))
        print(f"[ADDENDUM H L{L}] lag1_closes={c1} lag2_closes={c2} diff_closes={cd} "
              f"nulls(lag1/diff)={nulls_lag1}/{nulls_diff} controls_closing="
              f"{[(x['arm'], x['control']) for x in ctrl_hits]} -> {h_verdict}", flush=True)

        # ---------- post-freeze labels ----------
        labels_after_freeze = []
        month_ids = set(int(x) for x in s12) | {5656}
        for arm, entry in arm_out.items():
            for pname, v in (entry.get("blind") or {}).items():
                if not isinstance(v, dict) or "screened" not in v:
                    continue
                for r in v["screened"]:
                    if not r["flagged"]:
                        continue
                    ids = r["members_token_ids"]
                    labels_after_freeze.append(dict(
                        layer=L, arm=arm, partition=pname, cluster_id=r["cluster_id"],
                        member_hash=r["member_hash"], k=r["k"],
                        token_strings=[pool_strings[i] for i in r["members_pool_idx"]],
                        n_month_ids=int(sum(1 for t in ids if t in month_ids)),
                        n_calendar_ids=int(sum(1 for t in ids if t in calendar_ids)),
                        p_closed=r["p_closed"], p_bg=r["p_bg"],
                        bh_pass=r.get("bh_pass_within_layer_arm"),
                        not_reportable=r.get("not_reportable")))
        print(f"[L{L}] post-freeze labels: {len(labels_after_freeze)} flagged clusters "
              f"annotated", flush=True)

        if L in deltas_layers:
            for arm in ARMS:
                delta_store[f"L{L}_{arm}"] = halfD[(arm, "full")].astype(np.float16)
                delta_store[f"L{L}_{arm}_kept_coords"] = np.flatnonzero(
                    drop[arm]["_keep"]).astype(np.int32)
                delta_store[f"L{L}_{arm}_dropped_coords"] = np.asarray(
                    drop[arm]["dropped_coordinates"], dtype=np.int32)
                delta_store[f"L{L}_{arm}_pool_noise_sd"] = drop[arm][
                    "_pool_noise_sd"].astype(np.float32)
                delta_store[f"L{L}_{arm}_month_noise_sd_kept"] = np.sqrt(
                    (((halfD[(arm, 0)][month_pool_idx]
                       - halfD[(arm, 1)][month_pool_idx]) / np.sqrt(2)) ** 2).mean(0)
                ).astype(np.float32)

        lr = dict(layer=L, nominal_l0=meta.get("nominal_l0"), d_sae=d_sae,
                  n_nonzero_streamed=int(nnz_total),
                  stream_seconds=round(stream_seconds, 1),
                  identity_removal_seconds=round(matmul_seconds, 1),
                  n_zero_delta_rows=n_zero_rows,
                  anchor_type_mean=anchor, anchor_check=anchor_check,
                  fuse_reproduction=fuse_check,
                  outlier_drop={arm: {k: v for k, v in drop[arm].items()
                                      if not k.startswith("_")} for arm in ARMS},
                  planted_ring_power=planted,
                  addendum_H_planted_gate=planted_gate_layer,
                  addendum_H1_sensitivity_ladder={
                      arm: {r: {k: v for k, v in ladder[arm][r].items() if k != "cell"}
                            for r in SENSITIVITY_RUNGS} for arm in ARMS},
                  addendum_H1_ladder_seconds=ladder_seconds,
                  addendum_H_decision=addendum_H_decision,
                  arms=arm_out, labels_after_freeze=labels_after_freeze,
                  seconds=round(time.time() - tL, 1))
        layer_out.append(lr)
        if partial:
            (partial / f"layer_{L}.json").write_text(json.dumps(lr, indent=2))
        print(f"layer {L} done in {lr['seconds']}s", flush=True)
        half = None; halfD = None; ladder = None; drop = None
        TMP = None; W = None; csr = None
        gc.collect()

    # ---------------------------------- verdicts ----------------------------------
    bh = {}
    for arm in ARMS:
        rows = [(lr["layer"], lr["arms"][arm]) for lr in layer_out
                if arm in lr["arms"] and lr["arms"][arm].get("supervised")
                and lr["layer"] not in CONFIRMATORY_LAYERS
                and lr["arms"][arm]["supervised"]["strict_12"].get("scorable")]
        if not rows:
            bh[arm] = dict(n=0, note="no scorable gate-passing exploratory layer")
            continue
        ps = [e["supervised"]["strict_12"]["screening"]["p_closed"] for _, e in rows
              if e["supervised"]["strict_12"].get("screening")]
        rows = [(L, e) for L, e in rows if e["supervised"]["strict_12"].get("screening")]
        if not ps:
            bh[arm] = dict(n=0, note="no scorable gate-passing exploratory layer")
            continue
        keep = benjamini_hochberg(ps, alpha=BH_Q)
        bh[arm] = dict(n=len(ps), q=BH_Q, layers=[int(L) for L, _ in rows],
                       p_closed=[float(p) for p in ps], bh_pass=[bool(k) for k in keep],
                       note=("BH within arm over the GATE-PASSING EXPLORATORY layers only; "
                             "confirmatory layers 6 and 7 are not BH-adjusted"))

    def verdict_of(L, arm):
        for lr in layer_out:
            if lr["layer"] == L and arm in lr["arms"]:
                return lr["arms"][arm].get("verdict")
        return None

    carry = {}
    for lr in layer_out:
        L = lr["layer"]
        v1, vd = verdict_of(L, "lag1"), verdict_of(L, "diff")
        if v1 == "CLOSED" and vd == "CLOSED":
            carry[str(L)] = ("ACTION: the month ring survives the immediate-predecessor "
                             "differencing (Addendum F.6)")
        elif v1 == "CLOSED":
            carry[str(L)] = ("PERSISTENT-CONTEXT CARRY: the month ring is present in "
                             "delta(+1) and absent in delta(+1) - delta(+2) (Addendum F.6)")
        elif v1 and v1.startswith("NOT CLOSED (negative UNCALIBRATED"):
            carry[str(L)] = ("delta(+1) does not close, but its planted-ring control has no "
                             "power: the F.6 contrast is UNINTERPRETABLE at this layer")
        else:
            carry[str(L)] = "no month ring in delta(+1); the F.6 contrast does not apply"

    if delta_store:
        dp = Path(a.deltas_out)
        if not dp.is_absolute():
            dp = REPO / dp
        dp.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(dp, pool_ids=pool_ids.astype(np.int32), **delta_store)
        print(f"wrote {dp} ({len(delta_store)} matrices, fp16)", flush=True)

    addendum_H = dict(
        spec=ADDENDUM_H_DOC,
        integrity=addendum_h_integrity(),
        dropped_coordinates={f"L{lr['layer']}_{arm}": lr["outlier_drop"][arm][
            "dropped_coordinates"] for lr in layer_out for arm in ARMS},
        n_kept_coordinates={f"L{lr['layer']}_{arm}": lr["outlier_drop"][arm]["n_kept"]
                            for lr in layer_out for arm in ARMS},
        dropped_variance_share={f"L{lr['layer']}_{arm}": lr["outlier_drop"][arm][
            "variance_share_dropped"] for lr in layer_out for arm in ARMS},
        planted_gate={f"L{lr['layer']}": dict(
            applies=lr["addendum_H_planted_gate"]["applies"],
            passed=lr["addendum_H_planted_gate"]["passed"],
            per_arm=lr["addendum_H_planted_gate"]["per_arm"]) for lr in layer_out},
        instrument_validated=bool(all(
            lr["addendum_H_planted_gate"]["passed"] for lr in layer_out
            if lr["addendum_H_planted_gate"]["applies"])),
        decision={f"L{lr['layer']}": dict(
            verdict=lr["addendum_H_decision"]["verdict"],
            confirmatory=lr["addendum_H_decision"]["confirmatory"],
            terms=lr["addendum_H_decision"]["terms"]) for lr in layer_out},
        confirmatory_verdicts={f"L{lr['layer']}": lr["addendum_H_decision"]["verdict"]
                               for lr in layer_out
                               if lr["layer"] in CONFIRMATORY_LAYERS},
        control_sets={c: dict(token_ids=control_defs[c]["token_ids"],
                              token_strings=control_defs[c]["words"],
                              label_index=control_defs[c]["label_index"],
                              full_k=control_defs[c]["full_k"],
                              unresolved=control_defs[c]["unresolved"])
                      for c in control_defs},
        not_reportable_passers=[
            dict(layer=lr["layer"], arm=arm, partition=a_["partition"],
                 member_hash=a_["member_hash"], k=a_["k"],
                 members_token_ids=a_["members_token_ids"])
            for lr in layer_out for arm, e in lr["arms"].items()
            for pk, v in (e.get("blind") or {}).items()
            if pk == "passer_audits"
            for a_ in v if a_.get("not_reportable")],
        reading=("the layer verdict is addendum_H_decision.verdict; the per-arm 'verdict' "
                 "field is Addendum F step 6's conjunction, carried forward unchanged so "
                 "scripts/41 and scripts/41b can be compared term by term"))

    # ADDENDUM H1's declared ladder, co-reported at the top level under its own key.
    sensitivity_ladder = dict(
        spec=ADDENDUM_H1_DOC,
        integrity=addendum_h1_integrity(),
        rungs=list(SENSITIVITY_RUNGS),
        verdict_bearing=False,
        supervised_set="strict_12",
        statistics=["p_closed", "p_bg", "rho", "block_perm_p"],
        ranking=("coordinates ranked by the label-free pool-wide split-half noise share "
                 "(A_even - A_odd)/sqrt(2) over included types -- the same statistic the "
                 "frozen H rule thresholds at 25%"),
        by_layer_arm={f"L{lr['layer']}_{arm}": {
            r: dict(lr["addendum_H1_sensitivity_ladder"][arm][r]["closure"],
                    dropped_coordinates=lr["addendum_H1_sensitivity_ladder"][arm][r][
                        "dropped_coordinates"],
                    equals_H_rule=lr["addendum_H1_sensitivity_ladder"][arm][r][
                        "equals_H_rule"])
            for r in SENSITIVITY_RUNGS}
            for lr in layer_out for arm in ARMS},
        h_rule_closes={f"L{lr['layer']}": lr["addendum_H_decision"][
            "addendum_H1_sensitivity_ladder"]["h_rule_closes"] for lr in layer_out},
        drop_depth_dependent={f"L{lr['layer']}": lr["addendum_H_decision"][
            "addendum_H1_sensitivity_ladder"]["drop_depth_dependent"]
            for lr in layer_out},
        first_closing_rung={f"L{lr['layer']}": lr["addendum_H_decision"][
            "addendum_H1_sensitivity_ladder"]["first_closing_rung"]
            for lr in layer_out},
        any_drop_depth_dependent=bool(any(
            lr["addendum_H_decision"]["addendum_H1_sensitivity_ladder"][
                "any_drop_depth_dependent"] for lr in layer_out)),
        reading=("CO-REPORTED, NOT VERDICT-BEARING (Addendum H1). Four fixed "
                 "preprocessing depths, ranked label-free. The write-up must state the "
                 "drop-depth dependence explicitly: an arm whose ring appears only at a "
                 "rung deeper than the frozen 25%-share rule is reported as "
                 "DROP-DEPTH-DEPENDENT, not as an action ring. The primary verdict is and "
                 "remains addendum_H_decision.verdict, the frozen H rule"))

    runtime_seconds = round(time.time() - t_start, 1)
    result = dict(
        status="RESULT", script="scripts/41b_lag_action_robust.py",
        spec=ADDENDUM_H_DOC, spec_erratum=ADDENDUM_H1_DOC, declared_at=declared_at,
        declared_rules=rules, addendum_H=addendum_H,
        sensitivity_ladder=sensitivity_ladder,
        layers=layers, null_layers=null_layers, l0=a.l0, root=str(a.root), smoke=smoke,
        smoke_note=("TRUNCATED PASS -- plumbing check only. TM and S cover a prefix of the "
                    "cache, the positive-control and fuse-reproduction asserts are DISABLED "
                    "(they fail by construction on a prefix), few types clear the 20-pair "
                    "floor, and NO number here is a verdict.") if smoke else None,
        n_rows_cache=n_rows_cache, n_rows_streamed=n_used, n_types_pool=n_types,
        pair_stats=pair_stats,
        n_types_included={k: int(v.sum()) for k, v in included.items()},
        strict_12_ids=sorted(int(x) for x in s12),
        strict_12_calendar_order={m: int(t) for m, t in zip(MONTHS, s12)},
        member_hash_13=h13, member_hash_14=h14,
        layers_out=layer_out, bh_exploratory=bh, lag_differencing_reading=carry,
        instrument_validation=dict(
            anchor_type_mean={str(lr["layer"]): lr["anchor_check"] for lr in layer_out
                              if lr["anchor_check"] is not None},
            fuse_reproduction={str(lr["layer"]): lr["fuse_reproduction"] for lr in layer_out
                               if lr["fuse_reproduction"] is not None},
            planted_ring_reach={
                f"L{lr['layer']}_{arm}": dict(
                    power_status=lr["planted_ring_power"][arm]["power_status"],
                    negative_is_calibrated=lr["planted_ring_power"][arm]["negative_is_calibrated"],
                    jitter_80_frozen=lr["planted_ring_power"][arm][
                        "primary_half_sample_noise"]["jitter_at_80pct_power_grid"],
                    jitter_80_full_data_noise=lr["planted_ring_power"][arm][
                        "secondary_full_data_noise"]["jitter_at_80pct_power_grid"],
                    jitter_80_measured_radius=lr["planted_ring_power"][arm][
                        "tertiary_measured_in_plane_radius"]["jitter_at_80pct_power_grid"])
                for lr in layer_out for arm in ARMS},
            reading=("a NOT-CLOSED verdict is only a negative where the planted-ring "
                     "control has power; where it does not, the cell is reported as "
                     "'NOT CLOSED (negative UNCALIBRATED)'")),
        deltas_saved=sorted(delta_store.keys()),
        deltas_path=str(a.deltas_out) if delta_store else None,
        deltas_note=("the saved L{layer}_{arm} matrices are POST-DROP (Addendum H); "
                     "L{layer}_{arm}_kept_coords, _dropped_coords, _pool_noise_sd (the "
                     "label-free selection statistic, full 1152) and "
                     "_month_noise_sd_kept (Addendum F's planted-ring noise model on the "
                     "kept coordinates) are saved beside them"),
        runtime_seconds=runtime_seconds)
    out_path.write_text(json.dumps(result, indent=2))
    print(f"wrote {a.out} ({runtime_seconds}s)", flush=True)


if __name__ == "__main__":
    main()
