"""Phase 3, Stage 4 -- LAG-CONDITIONED ACTION GEOMETRY.

Pre-registered: docs/phase3_prereg_addenda_2026-08-21.md ADDENDUM F (frozen 2026-08-22,
before this script was written), which incorporates docs/phase3_prereg_2026-08-21.md
sections 0/1/2.3 and the five critic repairs of the "lag-conditioned action geometry"
avenue in docs/next_avenues_2026-08-21.md. Funded by scripts/36 PASS (prereg section 6:
"41 runs iff 36 PASS"; results/phase3_lag_fuse.json gate = PASS, centred median 0.93334).
Output: results/phase3_lag_action.json (+ results/phase3_lag_action_deltas.npz).

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
from gfdisc.tokens import MONTHS

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


def declared_rules(args_view):
    return dict(
        spec="docs/phase3_prereg_addenda_2026-08-21.md ADDENDUM F, FROZEN 2026-08-22",
        addendum_F_text=ADDENDUM_F_TEXT,
        also_binding=("docs/phase3_prereg_2026-08-21.md sections 0 (exposure record), 1 "
                      "(global rules R1-R6), 2.3 (the fuse probe whose accumulator this "
                      "extends), 6 (funding: 41 runs iff 36 PASS -- "
                      "results/phase3_lag_fuse.json gate = PASS)"),
        funding="results/phase3_lag_fuse.json gate PASS, centred median 0.9333420819770755",
        no_forward_passes=("no model is loaded and no forward pass is run; the only model "
                           "artifact read is w_dec per layer"),

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
            "primary curve; the other two are reported beside it"),

        supervised_anchor_arm=(
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
            "PRIMARY (Addendum F.5). All 3,795 delta rows of each arm through the "
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
            "reported"),
        expected_false_flags=("per layer x arm x partition, the sum over screened clusters "
                              "of the background rate_p_closed_lt_05 at that cluster's size "
                              "(the formula declared in prereg section 2.1, "
                              "sum_k n_candidates_k x rate_k), reported beside the observed "
                              "flag count"),
        labels=("R3-compliant. No label enters any blind decision statistic. Token strings "
                "appear only in the post-freeze labels_after_freeze block, and the "
                "supervised anchor arm is explicitly labelled as an anchor, never as "
                "discovery"),
        decision=(
            f"Addendum F step 6, primary verdict (literal): the supervised month ring in "
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
                   "planted ring 850000 (primary) / 851000 (secondary) + int(100*jitter)"),
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
            anchor_tol_gap_ratio=ANCHOR_TOL_GAP_RATIO),
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
            "two scratch runs preceded the production run and are disclosed here: (i) the "
            "truncated smoke above; (ii) a SINGLE-LAYER FULL-ROW validation at L6 with "
            "reduced permutation/draw counts, run to a scratch path to exercise the two "
            "live asserts on real data. Both asserts passed bit-exactly (anchor gap_ratio "
            f"{ANCHOR_TABLE[6][1]}; fuse median {FUSE_L6_LAG1_CENTRED}). That validation "
            "also exposed L6 reduced-parameter numbers, which are superseded by the "
            "production run and are not reported as results"),
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
                        n_draws, n_perm, seed):
    """Driver shared by nulls (i) and (ii). The diff arm differences two INDEPENDENT draws."""
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
        s = screen(D, n_perm, seed=0)
        if s is None:
            continue
        n_scorable += 1
        gaps.append(s["max_gap_deg"]); ps.append(s["p_closed"])
        closed += int(s["p_closed"] < ALPHA)
    out = dict(null=("successor-composition-matched resample" if kind == "composition"
                     else "within-document predecessor-swap"),
               arm=arm, lags_used=list(lags), n_draws=int(n_draws),
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
        audits.append(dict(member_hash=r["member_hash"], partition=r["partition"],
                           cluster_id=r["cluster_id"], k=r["k"],
                           members_pool_idx=r["members_pool_idx"],
                           members_token_ids=r["members_token_ids"],
                           loo_jackknife=dict(all_pass=loo_pass, subsets=loo),
                           frequency_nuisance=freq, frequency_pseudo_null=fp,
                           angles_deg=[float(x) for x in np.degrees(ang) % 360],
                           counts=[int(counts_pool[i]) for i in mem]))
    out["passer_audits"] = audits
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
    ap.add_argument("--out", default="results/phase3_lag_action.json")
    ap.add_argument("--deltas-out", default="results/phase3_lag_action_deltas.npz")
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
        script="scripts/41_lag_action.py",
        declared_at=declared_at, layers=layers, smoke=smoke,
        declared_rules=rules), indent=2))
    print("=== ADDENDUM F (frozen 2026-08-22), the text this script implements ===",
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

        # ---------- positive control (b): planted ring in action space, per arm ----------
        planted = {}
        for ai, arm in enumerate(ARMS):
            inc = included[arm]
            norms = np.linalg.norm(half[(arm, "full")][inc], axis=1)
            radius = float(np.median(norms) * PLANTED_RADIUS_FRACTION) if len(norms) else 0.0
            noise = (half[(arm, 0)][month_pool_idx]
                     - half[(arm, 1)][month_pool_idx]) / np.sqrt(2)
            sd = np.sqrt((noise ** 2).mean(0))
            xy_m, _, _ = pca_2d(half[(arm, "full")][month_pool_idx])
            r_inplane = float(np.median(np.linalg.norm(xy_m, axis=1)))
            pr = planted_ring_power(
                radius, sd, a.n_planted, a.n_perm, seed_base=850_000 + 10_000 * ai,
                label=("FROZEN primary: declared radius, split-half noise "
                       "(A_even - A_odd)/sqrt(2), the ONE-HALF scale"))
            sec = planted_ring_power(
                radius, sd / np.sqrt(2), a.n_planted, a.n_perm,
                seed_base=851_000 + 10_000 * ai,
                label="co-report: declared radius, full-data-matched noise (sd / sqrt(2))")
            ter = planted_ring_power(
                r_inplane, sd / np.sqrt(2), a.n_planted, a.n_perm,
                seed_base=852_000 + 10_000 * ai,
                label=("co-report: MEASURED in-plane radius of the 12 real month rows, "
                       "full-data-matched noise -- the amplitude-matched comparator"))
            cal = pr["jitter_at_80pct_power_grid"] is not None
            planted[arm] = dict(
                primary_half_sample_noise=pr, secondary_full_data_noise=sec,
                tertiary_measured_in_plane_radius=ter,
                radius_rule=(f"{PLANTED_RADIUS_FRACTION} x median ||delta|| over included "
                             f"pool types = {radius}"),
                diagnostics=dict(
                    median_delta_norm=float(np.median(norms)) if len(norms) else None,
                    declared_radius=radius,
                    measured_in_plane_radius_of_months=r_inplane,
                    declared_radius_over_measured_in_plane_radius=(
                        float(radius / r_inplane) if r_inplane > 0 else None),
                    noise_sd_l2_norm=float(np.linalg.norm(sd)),
                    noise_sd_mean=float(sd.mean()), noise_sd_max=float(sd.max()),
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
            print(f"[planted ring L{L} {arm}] radius {radius:.5g} (measured in-plane "
                  f"{r_inplane:.5g}), ||sd|| {float(np.linalg.norm(sd)):.5g}, 80%-power "
                  f"jitter grid: primary {pr['jitter_at_80pct_power_grid']}, full-data "
                  f"{sec['jitter_at_80pct_power_grid']}, measured-radius "
                  f"{ter['jitter_at_80pct_power_grid']}", flush=True)

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
                arm_out[arm] = entry
                print(f"[L{L} {arm}] below the reliability gate -> no verdict, arm skipped",
                      flush=True)
                continue
            Dmat = half[(arm, "full")]

            sup = {}
            for setname, pos, lab in (("strict_12", s12_pos, lab12),
                                      ("set_13", m13_pos, lab13)):
                sup[setname] = supervised_cell(
                    Dmat, pos, lab, 12, [pool_strings[p] for p in pos],
                    seed=610_000 + 1000 * L + ai, n_perm=a.n_perm, n_bg=a.n_bg,
                    bg_weights=bg_weights_factory(pos), tag=setname)

            U = Dmat / np.maximum(np.linalg.norm(Dmat, axis=1, keepdims=True), 1e-12)
            su = screen(U[s12_pos], a.n_perm, seed=0)
            bgu = background_for_size(U, n_types, 12, seed=870_000 + 1000 * L + ai,
                                      n_draws=a.n_bg, weights=bg_weights_factory(s12_pos))
            unit_norm = dict(screening=su,
                             p_bg=None if su is None else p_bg_of(bgu, su["max_gap_deg"]),
                             background={k: v for k, v in bgu.items() if k != "_gaps"},
                             closes=bool(su is not None and su["p_closed"] < ALPHA))
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
                                                     sd_seed)
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
            entry.update(
                supervised=sup, nulls=nulls, unit_norm_arm=unit_norm,
                planted_ring_power=planted[arm],
                closure_conjunction=dict(
                    p_closed=None if scr is None else scr["p_closed"],
                    p_bg=s12s.get("p_bg"), rho=rho, block_perm_p=pblock,
                    nulls_run=bool(nulls_run), nulls_all_fail_to_close=bool(nulls_ok),
                    unit_norm_closes=unit_norm["closes"], closed=closed,
                    negative_is_calibrated=bool(calibrated),
                    closed_with_ordering_null=bool(closed and pblock is not None
                                                   and pblock < ALPHA)),
                verdict=verdict)
            print(f"[L{L} {arm}] supervised strict-12: p_closed "
                  f"{None if scr is None else scr['p_closed']}, p_bg {s12s.get('p_bg')}, "
                  f"rho {rho} -> {entry['verdict']}", flush=True)

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
                        bh_pass=r.get("bh_pass_within_layer_arm")))
        print(f"[L{L}] post-freeze labels: {len(labels_after_freeze)} flagged clusters "
              f"annotated", flush=True)

        if L in deltas_layers:
            for arm in ARMS:
                delta_store[f"L{L}_{arm}"] = half[(arm, "full")].astype(np.float16)

        lr = dict(layer=L, nominal_l0=meta.get("nominal_l0"), d_sae=d_sae,
                  n_nonzero_streamed=int(nnz_total),
                  stream_seconds=round(stream_seconds, 1),
                  identity_removal_seconds=round(matmul_seconds, 1),
                  n_zero_delta_rows=n_zero_rows,
                  anchor_type_mean=anchor, anchor_check=anchor_check,
                  fuse_reproduction=fuse_check, planted_ring_power=planted,
                  arms=arm_out, labels_after_freeze=labels_after_freeze,
                  seconds=round(time.time() - tL, 1))
        layer_out.append(lr)
        if partial:
            (partial / f"layer_{L}.json").write_text(json.dumps(lr, indent=2))
        print(f"layer {L} done in {lr['seconds']}s", flush=True)
        half = None; TMP = None; W = None; csr = None
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
        ps = [e["supervised"]["strict_12"]["screening"]["p_closed"] for _, e in rows]
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

    runtime_seconds = round(time.time() - t_start, 1)
    result = dict(
        status="RESULT", script="scripts/41_lag_action.py", declared_at=declared_at,
        declared_rules=rules,
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
        runtime_seconds=runtime_seconds)
    out_path.write_text(json.dumps(result, indent=2))
    print(f"wrote {a.out} ({runtime_seconds}s)", flush=True)


if __name__ == "__main__":
    main()
