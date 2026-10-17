"""PHASE 3, STAGE 4 -- rotational-operator census (prereg section 5 + Addendum E).

Registered by docs/phase3_prereg_addenda_2026-08-21.md Addendum E and by
docs/phase3_prereg_2026-08-21.md sections 0, 1, 2.2 and 5, all FROZEN. This script does not
get to choose its population, its proposer families, its statistics, its nulls or its
verdict rule; they are reproduced below so a reader can check that nothing drifted between
the freeze and the run. Funding: 35b go/no-go (2) PASS (results/phase3_ring_readers_fulln.json:
strict-12 raw-residual ring at full n, L6 gap_ratio 1.5205 / p_closed 1.50e-4 / p_bg 0.00200;
L7 1.6113 / 1.20e-3 / 0.00399), which is prereg section 6's condition for running this file.

The question is the one activation geometry cannot answer: the month circle is REPRESENTED
at L6-L9; does any single head READ it -- by angular matching in the QK bilinear form, or by
rotational transport in OV? The critic's repair (docs/next_avenues_2026-08-21.md avenue 5) is
the reason this script exists in this form and is quoted here rather than paraphrased:

    "The repair that makes it valid: the proposer selects planes for rotation-ness and the
     statistic measures rotation-ness, so random-plane nulls are trivially cleared --
     calibrate instead with a CLOUD-ROTATION null (hold the plane fixed, randomly rotate the
     token cloud inside the head's own 256-d subspace, byte-identical selection->statistic
     path), FA of the full conjunction measured there."

scripts/35 Part B is the measured instance of that defect: every activation-derived 2-plane
scored at the FLOOR of the declared random-plane null (Addendum C: "fake planes from 12
random types: percentile median 0.8-2.5, identical to the month plane's 0.0-2.8"). A null
whose draws are planes cannot price a search whose output is a plane. A null whose draws are
token clouds can.

===========================================================================
ADDENDUM E, QUOTED (docs/phase3_prereg_addenda_2026-08-21.md lines 84-129; the text is
reproduced word for word with the Unicode transliterated to ASCII -- degree -> "deg",
lambda -> "l", x-hat -> "x_hat", section-sign -> "section" -- so it is NOT byte-identical to
that file and a diff will show those substitutions and nothing else)
===========================================================================
## Addendum E - frozen 2026-08-22 before any scripts/40 run: rotational-operator census, concrete parameters

Funding: 35b go/no-go (2) PASS (raw-residual strict-12 ring closes at L4-L9 at full n; L6
1.5205/p 1.5e-4/p_bg 0.002; L7 1.6113/p 1.2e-3). section 5 stands; the following pins what
section 5 left to docs/next_avenues avenue 5.
- Token artifact: per-type mean resid_post for the 3,795-type pool at ALL layers -1..25 over
  all 15,696 sequences (one forward pass; fp64 accumulation; BOS excluded), saved to
  results/phase3_token_type_mean_resid_all_layers.npz. The L4-L9 block must reproduce
  results/phase3_token_type_mean_resid_fulln.npz to fp32 precision (assert) - that is the
  artifact's validation. Readout for head at layer L uses R[L-1] after the layer's
  input RMSNorm with (1+w_in) gain (x_hat = RMSNorm(R[L-1]) * (1+w_in)).
- Proposer (weights only, 26 layers x 4 query heads, RoPE Delta=0): (a) antisym family - real
  Schur blocks of A_h=(M_h-M_h^T)/2, top-2 by block norm; (b) near-degenerate symmetric pairs
  of S_h=(M_h+M_h^T)/2 - top-2 consecutive eigenpairs with |li-li+1|/|li| < 0.1 maximizing
  (|li|+|li+1|)/2; (c) OV family - complex eigenpairs of W_OV_h (and I+W_OV_h), top-2 by |l|
  with |arg l| in [10deg,170deg]. <= 6 planes/head, <= 624 total; count frozen before any statistic.
- Readout per plane: project all 3,795 x_hat_t; select the k=24 types with the largest in-plane
  energy e_t = ||z_t||^2/||x_hat_t||^2 (selection is part of the statistic); angles theta_t. QK: score
  matrix S[s,t] = (x_hat_s W_q_eff).(x_hat_t W_k_eff) over the 24 types, two-way centred; F1 =
  variance explained by a*cos(ths-tht)+b*sin(ths-tht); P1 = H1/(H1+...+H5). OV: rotate the 24
  in-plane points by phi_hat = arg l and compute permutation consistency PC. Closure tier: max-gap
  p_closed of the 24 angles (20,000-perm uniform-angle null).
- Calibration null (the repair): CLOUD ROTATION - hold the plane U fixed; apply a random
  orthogonal transform to the token matrix inside the head's 256-d read (QK) or OV-image
  subspace; re-run selection->angles->statistic byte-identically. 1,000 draws set tau_h = max F1
  (resp. max PC); a second independent 1,000 draws measure the FA rate of the full conjunction
  (i) F1 > tau_h, (ii) P1 >= 0.60, (iii) p_closed < 0.05. BH q=0.05 over all planes tested
  (~624), using the conjunction's empirical p from the second null set.
- Passers only: frequency-pseudo-angle null (angles replaced by corpus-frequency-rank angles of
  the same 24 types, fit must collapse), Delta-sweep at RoPE offsets {1,2,4,8,64,256}, cosine-
  normalised refit, empirical per-type RMS. Labels attached after freeze.
- Positive control (both halves must pass before any blind claim): (i) readout half - the
  scripts/35 MT16 month plane QP_L at L6-L9 handed to the readout must select the 12 months
  within its top-24 in-plane-energy types at >= 8/12 and their angles must pass closure
  (p < 0.05) and order (rho >= 0.9) - if the selection rule does not surface the months this
  is recorded as a readout limitation and the blind arm is reported as "selection-limited";
  (ii) proposer half - plant c*QP R(phi) QP^T into a real head's M_h at L7 (and W_OV) for
  phi in {0deg,30deg,90deg,180deg}, c in {0.05,0.1,0.2,0.5,1.0} x that head's top singular value;
  report proposer recall and end-to-end detection; c* = 80% detection; instrument unfit if c* > 0.5.
- Behavioural confirmation on passers: real attention probabilities over 2,000 cached
  sequences, binned by relative position, fit to cos(ths-tht-phi_hat), phi_hat within +-20deg of the
  weight value.
- Pre-committed readings: positive - a weight-proposed plane whose occupying population closes
  and survives the passer nulls; negative - "no SINGLE head implements angular matching or
  rotational transport on any plane above c*" (MQA caveat: rotors split across heads or
  attention+MLP are invisible here).
===========================================================================
DECLARED BEFORE ANY CACHE READ OR FORWARD PASS (written to --out as a stub; declared_at is
carried into the final JSON, and prereg R1 makes that stub's mtime the audit anchor)
===========================================================================

E0  POPULATION. `build_pool` of scripts/22_token_type_geometry.py, copied verbatim because
    that module is a script and not importable: token types with corpus count >= 100 in
    cache/sparse_big/layer_0/tokens.i32. 3,795 types. Sequences are the FIRST --n-seq rows of
    tokens.i32 reshaped (15696, 255) with --n-seq defaulting to 15,696 (all), fed as
    [BOS] + row, bf16, eager, use_cache=False -- byte-for-byte how scripts/06_cache_sparse.py
    fed them and how scripts/35b re-fed them. Positions 1..255 are accumulated; position 0 is
    BOS and is excluded. Accumulators are fp64; the saved artifact is fp32.
    READOUT POPULATION: pool types with count_seen >= 1 in the forwarded set (a type never
    seen has an identically-zero mean row and no direction). At --n-seq 15696 that is the
    whole 3,795 pool; the identity is ASSERTED, not assumed. The restriction exists so the
    tiny smoke runs are sane; it changes nothing at full n.

E1  ARTIFACT. resid_post at layers -1..25, where layer -1 is THE INPUT TO LAYER 0 (the scaled
    embedding output), captured with a forward PRE-hook on model.model.layers[0], and layer L
    >= 0 is that layer's output, captured with the forward hook scripts/35 uses. 27 layers x
    3,795 x 1,152. Saved to --npz-out as fp32 with key R_L{L} (L >= 0) and R_Lm1 (L = -1),
    together with pool_ids, counts, and rms_L{...} (the per-type mean of the per-position
    residual RMS, which Addendum E's "empirical per-type RMS" passer arm consumes).
    VALIDATION (Addendum E, assert): the L4-L9 block must reproduce
    results/phase3_token_type_mean_resid_fulln.npz to fp32 precision. Declared exactly:
    max over the six layers of max |new - old| / max(|old|, 1e-6) must be < 1e-5. The
    comparison is made ONLY when --n-seq == 15696 (at any smaller n the two artifacts are
    means over different occurrence sets and disagreement is arithmetic, not error), and
    bit-parity additionally requires --batch 16 and the same sequence order, which is the
    default. The npz is written BEFORE the comparison, so a failed assert costs no forwards.

E2  READOUT SPACE, and the one place this script deliberately differs from scripts/35 Part B.
    Addendum E fixes the readout at x_hat = RMSNorm(R[L-1]) * (1+w_in) -- the input LayerNorm
    gain is on the DATA side -- and describes M_h as "QK-norm-adjusted". The operator and the
    cloud must live in the same coordinates or the plane's angles are not the cloud's angles,
    so:
        W_q_eff^T = diag(1+g_q) Wq_h            (256 x 1152)   acts on x_hat
        W_k_eff^T = diag(1+g_k) Wk              (256 x 1152)   acts on x_hat  (Wk is SHARED:
                                                                num_key_value_heads = 1)
        M_h = (diag(1+g_q) Wq_h)^T (diag(1+g_k) Wk)            RoPE Delta = 0 (identity)
        W_OV_h = diag(1+w_in) (o_proj[:, 256h:256h+256]) Wv
    scripts/35 Part B used M_35 = diag(1+w_in) M_h diag(1+w_in) and WOV_35 = Oh Wv diag(1+w_in)
    because its statistic consumed RAW residuals. The relation is recorded, not hidden:
    M_35 is congruent to M_h and WOV_35 is SIMILAR to W_OV_h (W_OV_h = D WOV_35 D^-1 with
    D = diag(1+w_in)), so every eigenvalue this script reports for the OV family is IDENTICAL
    to the one scripts/35 reports; only the eigenVECTORS differ, by that diagonal, which is
    exactly the change of coordinates the readout requires.
    RMSNorm is Gemma3RMSNorm: x * rsqrt(mean(x^2) + 1e-6), then * (1 + weight); q_norm/k_norm
    enter as their diagonal gain (1+g) only, and the per-token/per-head rescale they also
    apply is data-dependent and cannot enter a weights-only proposer. Declared limitation,
    identical to scripts/35 D9.
    ALSO OMITTED, declared: post_attention_layernorm, which Gemma3 applies to the attention
    output before the residual add. Its gain would multiply W_OV_h on the left by
    diag(1+w_post). The primary operator does NOT include it (parity with scripts/35 and with
    the eigenvalues already on record); the eigenvalues of the w_post-included variant are
    computed and REPORTED per head as a diagnostic and never select a plane.
    The attention softmax scale (query_pre_attn_scalar^-0.5 = 1/16) is a positive constant
    and affects neither F1, P1, PC nor p_closed; it is omitted.

E3  PROPOSER, weights only, --layers (default 0..25) x 4 query heads, RoPE Delta = 0, <= 6
    planes per head, <= 624 total, count FROZEN and written to the output before any readout
    number exists.
      (a) antisym  -- real Schur decomposition of A_h = (M_h - M_h^T)/2 (scipy.linalg.schur,
          output='real'); the 2x2 blocks are [[0,b],[-b,0]]; blocks ranked by |b|; top 2;
          U = the corresponding two columns of the Schur basis Z (already orthonormal).
      (b) symmetric-degenerate -- eigh of S_h = (M_h + M_h^T)/2, eigenvalues ASCENDING;
          consecutive pairs (i, i+1) with |l_i - l_i+1| / max(|l_i|, 1e-300) < 0.10;
          ranked by (|l_i| + |l_i+1|)/2 descending; greedy, pairs sharing an index are
          skipped; top 2; U = the two eigenvectors.
      (c) OV -- eigenpairs of W_OV_h through the rank-256 lift eig(Wv D Oh) with
          v = D Oh u (D = diag(1+w_in)), one member per conjugate pair (Im l > 1e-10).
          Each pair contributes TWO candidate variants, l (from W_OV_h) and 1+l (from
          I + W_OV_h, same eigenvector); a variant is eligible iff |arg| in [10deg, 170deg];
          eligible variants are ranked by the variant's modulus, descending; the top 2 are
          taken with the plane de-duplicated (a pair may not supply both of its variants).
          U = orthonormalised [Re v, Im v]; phi_hat = |arg| of the selected variant.
      Every U is orthonormalised to R^{1152 x 2} (QR). Degenerate candidates (rank < 2) are
      dropped. A head may therefore propose fewer than 6 planes; the realised count is
      reported per family.

E4  READOUT per plane (Addendum E verbatim, with the three things it leaves implicit pinned
    here BEFORE any number exists):
      z_t = U^T x_hat_t ; e_t = ||z_t||^2 / ||x_hat_t||^2 ; sel = the k = 24 largest e_t
      (ties broken by ascending pool row index; selection IS part of the statistic).
      theta_t = atan2(z_t[1], z_t[0]) -- angles about the PLANE ORIGIN, because a rotation of
      a 2-plane fixes the origin and e_t is itself defined about the origin. The centroid-
      centred variant is computed and reported as a SUPPLEMENTARY, non-verdict column.
      QK: S[s,t] = (W_q_eff^T x_hat_s) . (W_k_eff^T x_hat_t) over the 24 x 24 grid INCLUDING
      the diagonal s = t (self-attention is not masked in this model), two-way centred
      (S - rowmean - colmean + grandmean). D[s,t] = theta_s - theta_t.
        F1  = 1 - ||Sc - a cos D - b sin D||^2 / ||Sc||^2 (ordinary least squares on the two
              columns; no intercept -- Sc is already doubly centred).
        phi_F1 = atan2(b, a)  (so a cos D + b sin D = r cos(D - phi_F1)).
        P1  = H1 / (H1 + ... + H5), where the harmonics 1..5 are fitted JOINTLY (10 columns
              cos(mD), sin(mD)) by least squares and H_m = || the fitted m-th component ||^2.
        PC(phi) = 1 - mean_t d_t / (pi / k) with d_t = min over s != t of the wrapped
              |theta_t + phi - theta_s|; PC = max(PC(+phi_hat), PC(-phi_hat)) because
              orthonormalising [Re v, Im v] does not preserve the handedness of the eigenplane.
              phi_hat = |arg l| for OV-family planes, phi_F1 for QK-family planes.
        p_closed = gap_p(24, max angular gap, 20,000 uniform-angle permutations), the frozen
              max-gap null of scripts/13 as copied into scripts/22 and scripts/35.

E5  CALIBRATION NULL -- CLOUD ROTATION. The plane U is HELD FIXED. B is an orthonormal basis
    of the head's 256-d read subspace (row space of W_q_eff^T) for QK-family planes, or of the
    head's OV-image subspace (column space of W_OV_h) for OV-family planes. A Haar-random
    orthogonal Q on that subspace maps x_hat -> x_hat + B (Q - I) B^T x_hat, i.e. it rotates
    ONLY the in-subspace component and leaves the orthogonal complement alone. ||x_hat_t|| is
    therefore exactly preserved (asserted numerically once), and selection -> angles ->
    statistic are re-run byte-identically on the rotated cloud.
    Implementation identity (exact, not an approximation): with Y = B^T X_hat^T,
    A0 = W_q_eff^T X_hat^T, Ga = W_q_eff^T B (and K0, Gk likewise),
        z'  = z + ((B^T U)^T (Q - I)) Y
        q'  = A0 + Ga (Q - I) Y ,  k' = K0 + Gk (Q - I) Y
    which is why a draw costs ~7 MFLOP instead of a 3795 x 1152 rotation.
    Haar convention: Q, R = qr(N(0,1)^{r x r}); Q <- Q * sign(diag(R)).
    Set 1 (--n-null draws, seed 1000L + h [+250000 for the OV subspace]) sets
        tau_h = max over draws AND over that head's planes OF THE GATING STATISTIC'S FAMILY
        (tau_F1 over the head's QK-family planes, tau_PC over its OV-family planes),
        which is Addendum E's per-head tau; the per-plane max is co-reported.
    Set 2 (independent, same seeds + 500000) measures the FA rate of the full conjunction.
      CONJUNCTION, pinned here because Addendum E writes it in QK terms and adds "(resp. max
      PC)": QK-family planes -- (i) F1 > tau_F1, (ii) P1 >= 0.60, (iii) p_closed < 0.05.
      OV-family planes -- (i) PC > tau_PC, (ii) p_closed < 0.05; P1 is CO-REPORTED but does
      NOT gate, because P1 is a property of the QK score matrix which an OV rotor does not
      move, so gating on it would make the planted-OV positive control fail for a reason that
      has nothing to do with the OV instrument. The FA rate is measured per family in set 2,
      so the calibration is unaffected by this choice.
      A plane's p is that FA rate (add-one) IF the real plane satisfies its conjunction, and
      1.0 otherwise.
    BH q = 0.05 over all planes tested, on those p's.
      BH ARITHMETIC, declared before the numbers exist (the scripts/35 D11 lesson): with 1,000
      draws the add-one FA p is floored at 1/1001 = 9.99e-4 while the BH rank-1 threshold over
      624 planes is 0.05/624 = 8.01e-5. NO SINGLE plane can survive BH at the floor; at least
      ceil(0.05^-1 * 624 * 9.99e-4) ~ 13 floor-p planes must co-occur. That is a property of
      the declared draw count, it is reported, and it is NOT repaired post hoc. Planes that
      satisfy the conjunction but do not survive BH are reported as CONJUNCTION-ONLY and are
      explicitly non-verdict annotation.
      TIMING GOVERNOR, declared: if the measured per-head null time projects the full
      2 x --n-null x n_heads budget past --max-null-hours (3.0), --n-null is halved to 500 +
      500 for every head (Addendum E's own permitted reduction) and the reduction, the
      measurement that triggered it and the resulting p floor are recorded.

E6  POSITIVE CONTROL (both halves; prereg R2 -- a failed control produces NO blind verdict).
    (i) READOUT HALF. QP_L is the scripts/35 D8 recipe: top-2 right singular vectors of the
        centred MT16 label means (16 templates per month word, LAST non-pad position, bf16
        forward, fp32 mean), anchor gate |gap_ratio(L9) - 1.5123| <= 0.06 re-asserted and
        REPORTED. Two variants are run and BOTH are declared now so neither can be chosen
        after the fact:
          LITERAL   -- QP_L from label means at layer L, handed to the readout for heads at
                       layer L, exactly as Addendum E words it.
          XHAT      -- label means at layer L-1 pushed through the identical readout
                       transform x_hat = RMSNorm(.) * (1+w_in of layer L), then PCA. This is
                       the month plane expressed in the coordinates the readout actually
                       reads.
        Gate per Addendum E: >= 8 of the 12 month ids inside the top-24 in-plane-energy
        selection, AND the 12 months' angles pass closure (p_closed < 0.05) AND order
        (cyclic_order_score rho >= 0.9). READING RULE, declared now: the blind arm is
        "selection-limited" only if BOTH variants fail. LITERAL fails while XHAT passes is
        reported as "QP coordinate mismatch" (a statement about which coordinates the frozen
        plane lives in, not about the readout). The months-in-top-24 count and the month-angle
        closure/order depend only on (U, x_hat) and are therefore head-independent; they are
        reported per (layer, variant). F1/P1/PC are reported per (layer, variant, head).
    (ii) PROPOSER HALF. Plant P = c * s1 * QP R(phi) QP^T with QP the XHAT-variant plane at
        L7 (the plant must live in the space the readout reads, or the control measures
        coordinate mismatch instead of instrument sensitivity; the sumcos2 between the two QP
        variants is reported so the size of that choice is visible), phi in
        {0, 30, 90, 180} degrees, c in {0.05, 0.1, 0.2, 0.5, 1.0}, into a REAL head at L7,
        heads 0 and 2, both operators in the same cell:
          M_h  -> M_h + c * s1(M_h) * QP R(phi) QP^T
          W_OV -> W_OV + c * s1(W_OV) * QP R(phi) QP^T
        THE PLANT IS REALISED IN THE FACTORS, not only in M: W_q_eff^T gains two rows
        c*s1*(R(phi)^T QP^T) and W_k_eff^T gains two rows QP^T, so that
        W_q_eff^T-augmented^T W_k_eff^T-augmented = M_h + P EXACTLY (asserted) and the score
        matrix S -- which is computed from the factors, not from M -- SEES the plant. A plant
        written only into M would be invisible to the statistic and c* would be unmeasurable
        by construction. The head's read subspace for the null becomes the row space of the
        augmented query factor (258 rows, so <= 258-d).
        Reported per cell: proposer recall (does any proposed plane have sumcos2(U, QP) >=
        1.5?) per family and end-to-end detection (does any proposed plane satisfy its
        conjunction against its own tau, from the reduced --n-null-plant = 200 + 200 draws?
        detection is conjunction-vs-tau and NEVER BH). c* = the smallest c whose detection
        rate over its 8 (phi, head) cells is >= 0.8. INSTRUMENT UNFIT if c* > 0.5, per
        Addendum E; if unfit, or if the readout half fails on both variants, the blind arm
        carries no verdict (prereg R2) and says so.

E7  PASSER-ONLY ARMS (Addendum E), run on planes that satisfy their conjunction:
      frequency-pseudo-angle null -- theta_t replaced by 2*pi*rank(corpus count of type t)/24
        over the same 24 types, every statistic refitted; the fit must collapse.
      Delta-sweep -- M_h recomputed as W_q_eff^T Rot(Delta) W_k_eff for Delta in
        {1,2,4,8,64,256}, Rot(Delta) = diag(cos) + diag(sin) S built from THIS layer's own
        inv_freq buffer (Gemma3 uses per-layer-type RoPE: sliding_attention base 10,000,
        full_attention base 1,000,000, half-split rotate_half convention, transformers 5.15
        source), S[:128,128:] = -I, S[128:,:128] = +I. The plane is re-proposed from the
        Delta-rotated operator and its overlap with the Delta=0 plane is reported.
      cosine-normalised refit -- S_cos[s,t] = (q_s . k_t)/(||q_s|| ||k_t||), same centring
        and same fits.
      empirical per-type RMS -- x_hat recomputed as R[L-1] / rms_emp[t] * (1+w_in) with
        rms_emp[t] the mean over that type's occurrences of the per-position residual RMS
        (accumulated in E1), instead of the RMS of the type mean.
      labels_after_freeze -- the 24 selected token ids decoded to strings, attached only here
        (prereg R3: labels are post-freeze annotation and decide nothing).

E8  BEHAVIOURAL CONFIRMATION (passers only): real attention probabilities from the first
    --n-behav (2,000) cached sequences at the passing head, read off the eager attention
    module. Pairs (query position s, key position t < s) whose token ids are both in the
    plane's 24-type selection contribute p = attn[h, s, t]. Reported: mean p per relative-
    position bin (the distance profile), and the least-squares fit of p (and log p) on
    a cos(Delta theta - phi) + const over 24 Delta-theta bins, both pooled and inside a fixed
    relative-position band, giving phi_behav. Addendum E's check: |phi_behav - phi_hat| <= 20
    degrees.

E9  MULTIPLICITY, VERDICT, PRE-COMMITTED READINGS (Addendum E verbatim):
      positive -- "a weight-proposed plane whose occupying population closes and survives the
        passer nulls";
      negative -- "no SINGLE head implements angular matching or rotational transport on any
        plane above c*" (MQA caveat: rotors split across heads or across attention+MLP are
        invisible here; Wk and Wv are shared by all four query heads, so a layer's four cells
        differ only through Wq_h, g_q and the o_proj column block).

E10 SEEDS. Cloud-rotation set 1: 1000*L + h for the QK read subspace, 1000*L + h + 250000 for
    the OV-image subspace. Set 2: the same + 500000. Planted-control nulls: 900000 + 1000*
    (cell index) + the same family offsets. Uniform-angle max-gap null: seed 0 at every call
    (it is a fixed reference distribution, cached per (k, n_perm)). cyclic_order_score: seed 0.
    No statistic anywhere in this file draws from an unseeded generator.
"""
import argparse, contextlib, hashlib, json, os, sys, time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_HOME", str(REPO / "cache" / "hf"))
sys.path.insert(0, str(REPO / "src"))

import scipy.linalg as sla                                                # noqa: E402
from gfdisc.env import DEVICE, D_MODEL                                    # noqa: E402
import torch                                                              # noqa: E402
from gfdisc.model import load_model                                       # noqa: E402
from gfdisc.fits import pca_2d, fit_circle, circularity, cyclic_order_score  # noqa: E402
from gfdisc.nulls import benjamini_hochberg                               # noqa: E402
from gfdisc.tokens import MONTHS                                          # noqa: E402

# ===========================================================================
# FROZEN CONSTANTS
# ===========================================================================
COUNT_FLOOR = 100              # scripts/22 pool floor -- 3,795 types
COUNT_FLOOR_EXT = 30           # scripts/22 extended floor (pool parity only)
N_SEQ_TOTAL, SEQ_LEN = 15_696, 255
N_HEADS, N_KV_HEADS, HEAD_DIM, N_LAYERS = 4, 1, 256, 26
ART_LAYERS = list(range(-1, N_LAYERS))          # E1: -1 .. 25
FULLN_NPZ = "results/phase3_token_type_mean_resid_fulln.npz"
FULLN_LAYERS = [4, 5, 6, 7, 8, 9]
ARTIFACT_REL_TOL = 1e-5                          # E1
RMS_EPS = 1e-6                                   # config.rms_norm_eps

K_SELECT = 24                                    # E4
N_PERM_CLOSURE = 20_000                          # E4
N_HARMONICS = 5                                  # E4
P1_GATE = 0.60                                   # E5 conjunction (ii)
P_CLOSED_GATE = 0.05                             # E5 conjunction (iii)
BH_ALPHA = 0.05                                  # E5
SYM_REL_TOL = 0.10                               # E3 (b)
OV_ARG_LO_DEG, OV_ARG_HI_DEG = 10.0, 170.0       # E3 (c)
MAX_PLANES_PER_HEAD = 6                          # E3
SEED_OV_OFFSET, SEED_SET2_OFFSET, SEED_PLANT = 250_000, 500_000, 900_000  # E10

# E6 positive control
PLANE_LAYERS = [6, 7, 8, 9]
ANCHOR_GAP_RATIO_L9, ANCHOR_TOL_GAP = 1.5123, 0.06
MONTHS_IN_TOP24_GATE = 8
ORDER_RHO_GATE = 0.90
PLANT_LAYER, PLANT_HEADS = 7, [0, 2]
PLANT_PHIS_DEG = [0.0, 30.0, 90.0, 180.0]
PLANT_CS = [0.05, 0.1, 0.2, 0.5, 1.0]
PLANT_RECALL_SUMCOS2 = 1.5
PLANT_DETECTION_RATE = 0.80
CSTAR_UNFIT_ABOVE = 0.5

# E7 passer arms
DELTA_SWEEP = [1, 2, 4, 8, 64, 256]
BEHAV_PHI_TOL_DEG = 20.0

# scripts/27_span_262k.py MT16, verbatim (via scripts/35 D8)
MT16 = ["The report was published in {}", "She was born in {}", "It rained all {}",
        "The season begins in {}", "We travelled there last {}", "The deadline was {}",
        "Sales peaked in {}", "The festival runs through {}", "He resigned in {}",
        "The contract expires in {}", "Enrolment opens in {}", "The survey ran during {}",
        "Prices fell in {}", "The court ruled in {}", "The harvest starts in {}",
        "Construction began in {}"]

# prereg section 0, verbatim
SET13 = [3353, 5142, 5261, 5365, 5505, 5656, 5729, 5857, 6027, 6198, 6243, 6384, 7250]
SET13_HASH = "7d1d75a303d6"

FUNDING = ("prereg section 6: scripts/40 runs iff 35 go/no-go (2) PASS. "
           "results/phase3_ring_readers_fulln.json part_a.go_no_go_2 = PASS "
           "(L6 gap_ratio 1.5205 / p_closed 1.50e-4 / p_bg 0.00200; "
           "L7 1.6113 / 1.20e-3 / 0.00399).")


def jsonable(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, (np.complexfloating, complex)):
        return dict(re=float(np.real(o)), im=float(np.imag(o)))
    raise TypeError(f"{type(o)} is not JSON serialisable")


def member_hash(members):
    return hashlib.md5(",".join(str(int(x)) for x in sorted(members)).encode()).hexdigest()[:12]


# ===========================================================================
# COPIED VERBATIM from scripts/22_token_type_geometry.py (via scripts/35, 35b) -- that file
# is a script, not a module, so these cannot be imported. Any drift here is a bug.
# ===========================================================================
def build_pool(root, layer0=0):
    """Token types over the count floors. tokens.i32 is byte-identical across layers."""
    tok = np.fromfile(Path(root) / f"layer_{layer0}" / "tokens.i32", dtype=np.int32)
    cnt = np.bincount(tok)
    ext = np.flatnonzero(cnt >= COUNT_FLOOR_EXT)
    main = np.flatnonzero(cnt >= COUNT_FLOOR)
    return tok, cnt, main, ext


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


# ===========================================================================
# STAGE 0 -- the all-layers token-type mean-residual artifact
# ===========================================================================
@contextlib.contextmanager
def capture_all(model, layer_idxs):
    """resid_post at each requested layer; layer -1 = the INPUT to layer 0 (E1)."""
    store, handles = {}, []

    def mk(i):
        def hook(_mod, _inp, out):
            store[i] = (out[0] if isinstance(out, tuple) else out).detach()
        return hook

    if -1 in layer_idxs:
        def pre(_mod, args):
            store[-1] = args[0].detach()
        handles.append(model.model.layers[0].register_forward_pre_hook(pre))
    for i in layer_idxs:
        if i < 0:
            continue
        handles.append(model.model.layers[i].register_forward_hook(mk(i)))
    try:
        yield store
    finally:
        for h in handles:
            h.remove()


def stage0_forwards(model, tok, tokens2d, pool_ids, layers, n_seq, batch):
    """fp64 per-type sums of the raw residual stream at positions 1..255, all layers."""
    n_pool = len(pool_ids)
    lut = np.full(int(max(tokens2d.max(), np.max(pool_ids))) + 1, -1, dtype=np.int64)
    lut[pool_ids] = np.arange(n_pool)
    acc = {L: torch.zeros(n_pool, D_MODEL, dtype=torch.float64) for L in layers}
    rms_acc = {L: torch.zeros(n_pool, dtype=torch.float64) for L in layers}
    counts = np.zeros(n_pool, dtype=np.int64)
    bos = int(tok.bos_token_id)
    n_batches = (n_seq + batch - 1) // batch
    times = []
    emb_check = None
    with torch.inference_mode():
        for bi, lo in enumerate(range(0, n_seq, batch)):
            t0 = time.time()
            blk = tokens2d[lo:min(lo + batch, n_seq)].astype(np.int64)
            ids = np.concatenate([np.full((len(blk), 1), bos, dtype=np.int64), blk], axis=1)
            ids_t = torch.from_numpy(ids).to(DEVICE)
            with capture_all(model, layers) as st:
                model(ids_t, use_cache=False)
            if bi == 0 and -1 in layers:
                # The only capture path with no external reference: layer -1 must BE the
                # (scaled) embedding output, which is purely token-dependent. Checked
                # against the model's own embed_tokens on the first batch, then asserted.
                e = model.model.embed_tokens(ids_t).detach().to(torch.float32)
                dmax = float((st[-1].to(torch.float32) - e).abs().max().item())
                emb_check = dict(max_abs_diff=dmax, scale=float(e.abs().max().item()),
                                 note=("forward pre-hook on layers[0] vs "
                                       "model.model.embed_tokens on the first batch"))
                assert dmax <= 1e-4 * max(emb_check["scale"], 1e-6), (
                    f"layer -1 capture is not the embedding output: max abs diff {dmax}")
                del e
            tix = lut[blk.reshape(-1)]
            m = tix >= 0
            sel = torch.from_numpy(tix[m])
            mt = torch.from_numpy(m)
            for L in layers:
                x = st[L][:, 1:, :].reshape(-1, D_MODEL).to(torch.float32).cpu()
                xs = x[mt].to(torch.float64)
                acc[L].index_add_(0, sel, xs)
                rms_acc[L].index_add_(0, sel, xs.pow(2).mean(1).sqrt())
                del x, xs
            counts += np.bincount(tix[m], minlength=n_pool)
            del st
            times.append(time.time() - t0)
            if bi % 10 == 0 or bi == n_batches - 1:
                el = float(np.sum(times))
                print(f"  [S0] batch {bi+1}/{n_batches}  {times[-1]:.2f}s  "
                      f"mean {el/len(times):.2f}s/batch  eta "
                      f"{(n_batches-bi-1)*el/len(times)/60:.1f} min", flush=True)
    den = np.maximum(counts, 1)[:, None]
    R = {L: (acc[L].numpy() / den).astype(np.float32) for L in layers}
    RMS = {L: (rms_acc[L].numpy() / den[:, 0]).astype(np.float32) for L in layers}
    return R, RMS, counts, float(np.mean(times)), float(np.sum(times)), emb_check


def npz_key(L):
    return f"R_L{L}" if L >= 0 else "R_Lm1"


def rms_key(L):
    return f"rms_L{L}" if L >= 0 else "rms_Lm1"


def compare_to_fulln(R, path):
    """E1 validation: the L4-L9 block must reproduce the 35b artifact to fp32 precision."""
    z = np.load(path)
    per = {}
    worst = 0.0
    for L in FULLN_LAYERS:
        a = np.asarray(R[L], dtype=np.float64)
        b = np.asarray(z[f"R_L{L}"], dtype=np.float64)
        rel = np.abs(a - b) / np.maximum(np.abs(b), 1e-6)
        v = float(rel.max())
        per[str(L)] = dict(max_abs_rel_diff=v, max_abs_diff=float(np.abs(a - b).max()),
                           n_exact=int((a == b).sum()), n_total=int(a.size))
        worst = max(worst, v)
    return dict(source=str(path), rule=("max over L4-L9 of max |new-old|/max(|old|,1e-6) "
                                        f"< {ARTIFACT_REL_TOL}"),
                max_abs_rel_diff=worst, passed=bool(worst < ARTIFACT_REL_TOL), per_layer=per)


# ===========================================================================
# STAGE 1 -- the weights-only proposer
# ===========================================================================
def head_operators(model, L, h):
    """E2: every operator in x_hat coordinates. float64 numpy."""
    lay = model.model.layers[L]
    attn = lay.self_attn
    f = lambda t: t.detach().to(torch.float32).cpu().numpy().astype(np.float64)
    w_in = 1.0 + f(lay.input_layernorm.weight)
    w_post = 1.0 + f(lay.post_attention_layernorm.weight)
    g_q = 1.0 + f(attn.q_norm.weight)
    g_k = 1.0 + f(attn.k_norm.weight)
    Wq = f(attn.q_proj.weight)[h * HEAD_DIM:(h + 1) * HEAD_DIM, :]        # 256 x 1152
    Wk, Wv = f(attn.k_proj.weight), f(attn.v_proj.weight)                 # 256 x 1152 (shared)
    Oh = f(attn.o_proj.weight)[:, h * HEAD_DIM:(h + 1) * HEAD_DIM]        # 1152 x 256
    Wq_eff = g_q[:, None] * Wq                                            # acts on x_hat
    Wk_eff = g_k[:, None] * Wk
    DOh = w_in[:, None] * Oh                                              # diag(1+w_in) Oh
    return dict(w_in=w_in, w_post=w_post, Wq_eff=Wq_eff, Wk_eff=Wk_eff,
                Wv=Wv, Oh=Oh, DOh=DOh)


def orth2(Z):
    """Orthonormalise a 1152 x 2 block; None if it is rank deficient."""
    Z = np.asarray(Z, dtype=np.float64)
    if Z.shape[1] != 2 or not np.isfinite(Z).all():
        return None
    q, r = np.linalg.qr(Z)
    if min(abs(r[0, 0]), abs(r[1, 1])) < 1e-10 * max(abs(r[0, 0]), 1e-300):
        return None
    return np.ascontiguousarray(q[:, :2])


def orth_basis(A, tol=1e-9):
    """Orthonormal basis of the ROW space of A (columns of the result span it)."""
    _, s, vt = np.linalg.svd(np.asarray(A, dtype=np.float64), full_matrices=False)
    r = int((s > tol * max(s[0], 1e-300)).sum())
    return np.ascontiguousarray(vt[:r].T), r


def antisym_planes(A, n=2):
    """E3(a): real Schur blocks of the antisymmetric part, top-n by |b|."""
    T, Z = sla.schur(np.asarray(A, dtype=np.float64), output="real")
    d = T.shape[0]
    blocks, i = [], 0
    while i < d - 1:
        if abs(T[i + 1, i]) > 1e-13 * max(abs(T).max(), 1e-300):
            blocks.append((float(abs(T[i, i + 1])), i))
            i += 2
        else:
            i += 1
    blocks.sort(key=lambda x: -x[0])
    out = []
    for score, i in blocks[:n]:
        U = orth2(Z[:, i:i + 2])
        if U is not None:
            out.append(dict(U=U, family="antisym", score=score, phi_hat=None,
                            detail=dict(block_b=score)))
    return out


def symmetric_pair_planes(S, n=2, rel_tol=SYM_REL_TOL):
    """E3(b): near-degenerate consecutive eigenpairs of the symmetric part."""
    w, V = np.linalg.eigh(np.asarray(S, dtype=np.float64))          # ascending
    cands = []
    for i in range(len(w) - 1):
        den = max(abs(w[i]), 1e-300)
        if abs(w[i] - w[i + 1]) / den < rel_tol:
            cands.append((0.5 * (abs(w[i]) + abs(w[i + 1])), i))
    cands.sort(key=lambda x: -x[0])
    used, out = set(), []
    for score, i in cands:
        if i in used or (i + 1) in used:
            continue
        U = orth2(V[:, i:i + 2])
        if U is None:
            continue
        used.update({i, i + 1})
        out.append(dict(U=U, family="symdeg", score=float(score), phi_hat=None,
                        detail=dict(lam_i=float(w[i]), lam_ip1=float(w[i + 1]),
                                    rel_gap=float(abs(w[i] - w[i + 1]) / max(abs(w[i]), 1e-300)))))
        if len(out) == n:
            break
    return out


def ov_eigpairs(Wv, DOh):
    """Complex eigenpairs of W_OV = DOh @ Wv through the rank-256 lift eig(Wv @ DOh)."""
    S256 = Wv @ DOh                                             # 256 x 256
    ev, U = np.linalg.eig(S256)
    pairs = []
    for j in range(len(ev)):
        if ev[j].imag <= 1e-10:                                 # one member per conj pair
            continue
        v = DOh @ U[:, j]
        pairs.append((j, complex(ev[j]), v))
    return pairs, ev


def ov_planes(pairs, n=2):
    """E3(c): variants l and 1+l, |arg| in [10,170] deg, top-n by the variant's modulus."""
    cands = []
    for j, lam, v in pairs:
        for name, val in (("W_OV", lam), ("I+W_OV", 1.0 + lam)):
            ad = abs(np.degrees(np.angle(val)))
            if OV_ARG_LO_DEG <= ad <= OV_ARG_HI_DEG:
                cands.append((abs(val), j, name, val, v, ad))
    cands.sort(key=lambda x: -x[0])
    used, out = set(), []
    for mod, j, name, val, v, ad in cands:
        if j in used:
            continue
        U = orth2(np.column_stack([v.real, v.imag]))
        if U is None:
            continue
        used.add(j)
        out.append(dict(U=U, family="ov", score=float(mod), phi_hat=float(np.radians(ad)),
                        detail=dict(variant=name, abs_lambda=float(mod), arg_deg=float(ad),
                                    lam_re=float(val.real), lam_im=float(val.imag),
                                    eig_index=int(j))))
        if len(out) == n:
            break
    return out


def propose_head(ops, M=None, WOV_pairs=None):
    """The three frozen families for one head. <= 6 planes."""
    if M is None:
        M = ops["Wq_eff"].T @ ops["Wk_eff"]
    A = 0.5 * (M - M.T)
    S = 0.5 * (M + M.T)
    planes = antisym_planes(A, 2) + symmetric_pair_planes(S, 2)
    if WOV_pairs is None:
        WOV_pairs, _ = ov_eigpairs(ops["Wv"], ops["DOh"])
    planes += ov_planes(WOV_pairs, 2)
    return planes[:MAX_PLANES_PER_HEAD], M, A


# ===========================================================================
# STAGE 2 -- readout statistics (the byte-identical path the null re-runs)
# ===========================================================================
def rms_norm_x(R, w_in, eps=RMS_EPS):
    """x_hat = RMSNorm(R) * (1+w_in) -- Gemma3RMSNorm semantics, float64 (E2)."""
    R = np.asarray(R, dtype=np.float64)
    inv = 1.0 / np.sqrt((R ** 2).mean(1, keepdims=True) + eps)
    return (R * inv) * w_in[None, :]


def two_way_centre(S):
    return S - S.mean(0, keepdims=True) - S.mean(1, keepdims=True) + S.mean()


def wrap_pi(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def fit_h1(D, y):
    """F1 and phi_F1 (E4). y is already doubly centred, so no intercept."""
    X = np.column_stack([np.cos(D).ravel(), np.sin(D).ravel()])
    G = X.T @ X
    beta = np.linalg.solve(G + 1e-12 * np.eye(2), X.T @ y)
    res = y - X @ beta
    tot = float(y @ y)
    return (float(1.0 - (res @ res) / max(tot, 1e-300)),
            float(np.arctan2(beta[1], beta[0])), tot)


def fit_harmonics(D, y, mmax=N_HARMONICS):
    """P1 = H1/(H1+..+H5) with the harmonics fitted JOINTLY (E4)."""
    cols = []
    for m in range(1, mmax + 1):
        mD = m * D
        cols.append(np.cos(mD).ravel())
        cols.append(np.sin(mD).ravel())
    X = np.column_stack(cols)
    G = X.T @ X
    beta = np.linalg.solve(G + 1e-12 * np.eye(X.shape[1]), X.T @ y)
    H = []
    for m in range(mmax):
        c = X[:, 2 * m:2 * m + 2] @ beta[2 * m:2 * m + 2]
        H.append(float(c @ c))
    tot = sum(H)
    return (float(H[0] / tot) if tot > 0 else 0.0), H


def pc_one(th, phi):
    d = np.abs(wrap_pi(th[:, None] + phi - th[None, :]))
    np.fill_diagonal(d, np.inf)
    return float(1.0 - d.min(1).mean() / (np.pi / len(th)))


def pc_of(th, phi):
    """E4: max over the two handednesses of the eigenplane."""
    if phi is None:
        return None
    return max(pc_one(th, phi), pc_one(th, -phi))


def max_gap_rad(th):
    a = np.sort(np.asarray(th, dtype=np.float64))
    return float(np.diff(np.concatenate([a, [a[0] + 2 * np.pi]])).max())


def select_topk(Zt, norms2, k):
    """E4: top-k by in-plane energy; ties broken by ascending pool row index."""
    e = (Zt[0] ** 2 + Zt[1] ** 2) / norms2
    idx = np.argpartition(-e, k - 1)[:k]
    idx = idx[np.lexsort((idx, -e[idx]))]
    return idx, e[idx]


def plane_stats(Zt, norms2, qk_fn, phi_ov, k=K_SELECT, n_perm=N_PERM_CLOSURE,
                extras=False, angles_override=None):
    """The one readout path. The cloud-rotation null calls this byte-identically."""
    sel, e_sel = select_topk(Zt, norms2, k)
    z = Zt[:, sel]
    th = np.arctan2(z[1], z[0]) if angles_override is None else np.asarray(angles_override)
    q, kk = qk_fn(sel)
    S = q.T @ kk
    Sc = two_way_centre(S)
    y = Sc.ravel()
    D = th[:, None] - th[None, :]
    F1, phi_F1, sstot = fit_h1(D, y)
    P1, H = fit_harmonics(D, y)
    out = dict(F1=F1, phi_F1_deg=float(np.degrees(phi_F1)), P1=P1,
               PC_ov=pc_of(th, phi_ov), PC_qk=pc_of(th, phi_F1),
               p_closed=gap_p(k, max_gap_rad(th), n_perm),
               max_gap_deg=float(np.degrees(max_gap_rad(th))))
    if extras:
        zc = z - z.mean(1, keepdims=True)
        thc = np.arctan2(zc[1], zc[0])
        Dc = thc[:, None] - thc[None, :]
        F1c, phic, _ = fit_h1(Dc, y)
        P1c, _ = fit_harmonics(Dc, y)
        out.update(sel=sel.tolist(), e_sel=[float(x) for x in e_sel],
                   theta_deg=[float(np.degrees(x)) for x in th],
                   harmonics=H, sstot=sstot,
                   supplementary_centred=dict(
                       F1=F1c, P1=P1c, phi_F1_deg=float(np.degrees(phic)),
                       p_closed=gap_p(k, max_gap_rad(thc), n_perm),
                       note=("angles about the 24 points' in-plane centroid rather than the "
                             "plane origin; non-verdict (E4)")))
    return out, sel, th


def haar_orthogonal(r, rng):
    """E5 Haar convention: qr of a Gaussian with the sign(diag(R)) correction."""
    G = rng.normal(size=(r, r))
    Q, R = np.linalg.qr(G)
    return Q * np.sign(np.diag(R))[None, :]


class HeadCloud:
    """Precomputed per-head quantities for the exact cloud-rotation identity (E5)."""

    def __init__(self, Xhat, norms2, Wq_eff, Wk_eff, B):
        self.norms2 = norms2
        self.A0 = Wq_eff @ Xhat.T                       # (dq, n)
        self.K0 = Wk_eff @ Xhat.T                       # (dk, n)
        self.B = B                                      # (1152, r)
        self.Y = B.T @ Xhat.T                           # (r, n)
        self.Ga = Wq_eff @ B                            # (dq, r)
        self.Gk = Wk_eff @ B                            # (dk, r)
        self.r = B.shape[1]

    def real_qk_fn(self):
        return lambda sel: (self.A0[:, sel], self.K0[:, sel])

    def draw(self, rng):
        return haar_orthogonal(self.r, rng) - np.eye(self.r)


def rotated_Zt(Zt0, c, QmI, Y):
    return Zt0 + (c.T @ QmI) @ Y


def rotated_qk_fn(cloud, QmI):
    def fn(sel):
        dY = QmI @ cloud.Y[:, sel]
        return (cloud.A0[:, sel] + cloud.Ga @ dY,
                cloud.K0[:, sel] + cloud.Gk @ dY)
    return fn


# ===========================================================================
# RoPE (E7 Delta-sweep)
# ===========================================================================
def rope_rot(inv_freq, delta, head_dim=HEAD_DIM):
    """Rot(delta) with Gemma3's half-split rotate_half convention (transformers 5.15)."""
    ang = float(delta) * np.asarray(inv_freq, dtype=np.float64)
    c = np.concatenate([np.cos(ang), np.cos(ang)])
    s = np.concatenate([np.sin(ang), np.sin(ang)])
    half = head_dim // 2
    Smat = np.zeros((head_dim, head_dim))
    Smat[:half, half:] = -np.eye(half)
    Smat[half:, :half] = np.eye(half)
    return np.diag(c) + np.diag(s) @ Smat


def layer_inv_freq(model, L):
    lt = model.config.layer_types[L]
    iv = getattr(model.model.rotary_emb, f"{lt}_inv_freq")
    return lt, iv.detach().to(torch.float64).cpu().numpy()


# ===========================================================================
# Positive control -- the scripts/35 D8 label-means recipe
# ===========================================================================
def label_means_inline(model, tok, layers, words, templates):
    """scripts/27_span_262k.py `label_means`, copied via scripts/35 Part B."""
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    out = {l: [] for l in layers}
    with torch.inference_mode():
        for w in words:
            enc = tok([t.format(w) for t in templates], return_tensors="pt",
                      padding=True).to(DEVICE)
            with capture_all(model, layers) as st:
                model(**enc, use_cache=False)
            last = enc["attention_mask"].sum(1) - 1
            for l in layers:
                out[l].append(
                    st[l][torch.arange(last.shape[0]), last].float().mean(0).cpu().numpy())
    return {l: np.vstack(v).astype(np.float64) for l, v in out.items()}


def sumcos2(Q1, Q2):
    s = np.linalg.svd(np.asarray(Q1).T @ np.asarray(Q2), compute_uv=False)
    return float((np.clip(s, 0, 1) ** 2).sum())


def plane_from_means(Mn):
    xy, ratios, Vt = pca_2d(Mn)
    cx, cy, r, resid = fit_circle(xy)
    c = circularity(xy, cx, cy, r)
    return (np.ascontiguousarray(Vt[:2].T),
            dict(gap_ratio=c["gap_ratio"], max_gap_deg=c["max_gap_deg"],
                 radius_ratio=c["radius_ratio"], top2_var=float(ratios[:2].sum()),
                 p_closed=gap_p(len(Mn), np.radians(c["max_gap_deg"]), N_PERM_CLOSURE)))


# ===========================================================================
def conjunction(fam, st, tau_F1, tau_PC):
    """E5, pinned: QK -> (F1>tau, P1>=0.60, p_closed<0.05); OV -> (PC>tau, p_closed<0.05)."""
    if fam == "ov":
        return bool(st["PC_ov"] is not None and tau_PC is not None
                    and st["PC_ov"] > tau_PC and st["p_closed"] < P_CLOSED_GATE)
    return bool(tau_F1 is not None and st["F1"] > tau_F1
                and st["P1"] >= P1_GATE and st["p_closed"] < P_CLOSED_GATE)


def run_head_null(cloud_qk, cloud_ov, planes, n_draws, seed_qk, seed_ov, k, n_perm,
                  Zt0s, taus=None):
    """One null set. If taus is None returns per-plane/family maxima; else FA counts."""
    rq = np.random.default_rng(seed_qk)
    ro = np.random.default_rng(seed_ov)
    npl = len(planes)
    stat_max = np.full(npl, -np.inf)
    fa = np.zeros(npl, dtype=np.int64)
    need_qk = any(p["family"] != "ov" for p in planes)
    need_ov = any(p["family"] == "ov" for p in planes)
    cache = [dict(c=None) for _ in planes]
    for i, p in enumerate(planes):
        cl = cloud_ov if p["family"] == "ov" else cloud_qk
        cache[i]["c"] = cl.B.T @ p["U"]
    for d in range(n_draws):
        QmI_q = cloud_qk.draw(rq) if (need_qk and cloud_qk is not None) else None
        QmI_o = cloud_ov.draw(ro) if (need_ov and cloud_ov is not None) else None
        for i, p in enumerate(planes):
            ov = p["family"] == "ov"
            cl = cloud_ov if ov else cloud_qk
            QmI = QmI_o if ov else QmI_q
            if cl is None or QmI is None:
                continue
            Zt = rotated_Zt(Zt0s[i], cache[i]["c"], QmI, cl.Y)
            st, _, _ = plane_stats(Zt, cl.norms2, rotated_qk_fn(cl, QmI),
                                   p["phi_hat"], k=k, n_perm=n_perm)
            v = st["PC_ov"] if ov else st["F1"]
            if v is not None and v > stat_max[i]:
                stat_max[i] = v
            if taus is not None:
                tf, tp = taus
                if conjunction(p["family"], st, tf, tp):
                    fa[i] += 1
    return stat_max, fa


# ===========================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/phase3_rotational_census.json")
    ap.add_argument("--npz-out", default="results/phase3_token_type_mean_resid_all_layers.npz")
    ap.add_argument("--root", default="cache/sparse_big")
    ap.add_argument("--n-seq", type=int, default=N_SEQ_TOTAL)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--reuse-artifact", action="store_true",
                    help="load --npz-out if it exists instead of re-forwarding (E1)")
    ap.add_argument("--layers", type=int, nargs="+", default=list(range(N_LAYERS)),
                    help="census layers (proposer/readout/null); the artifact is always -1..25")
    ap.add_argument("--k", type=int, default=K_SELECT)
    ap.add_argument("--n-perm", type=int, default=N_PERM_CLOSURE)
    ap.add_argument("--n-null", type=int, default=1000)
    ap.add_argument("--n-null-plant", type=int, default=200)
    ap.add_argument("--n-plant", type=int, default=0,
                    help="max (phi, c) cells per planted head; 0 = the full 4x5 grid")
    ap.add_argument("--max-null-hours", type=float, default=3.0)
    ap.add_argument("--n-behav", type=int, default=2000)
    ap.add_argument("--behav-batch", type=int, default=8)
    a = ap.parse_args()
    t_start = time.time()

    # ---- R1: the declaration stub is written BEFORE any cache read or forward ----
    outp = Path(a.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    out = dict(script="scripts/40_rotational_census.py",
               item="Phase 3 Stage 4 -- rotational-operator census (prereg section 5, Addendum E)",
               prereg=("docs/phase3_prereg_addenda_2026-08-21.md Addendum E (FROZEN) + "
                       "docs/phase3_prereg_2026-08-21.md sections 0/1/2.2/5 (FROZEN)"),
               funding=FUNDING,
               declared_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
               declarations=__doc__,
               frozen_parameters=dict(
                   k_select=a.k, n_perm_closure=a.n_perm, n_harmonics=N_HARMONICS,
                   p1_gate=P1_GATE, p_closed_gate=P_CLOSED_GATE, bh_alpha=BH_ALPHA,
                   n_null_declared=a.n_null, n_null_plant=a.n_null_plant,
                   max_planes_per_head=MAX_PLANES_PER_HEAD,
                   max_planes_total=MAX_PLANES_PER_HEAD * 4 * len(a.layers),
                   sym_rel_tol=SYM_REL_TOL, ov_arg_deg=[OV_ARG_LO_DEG, OV_ARG_HI_DEG],
                   artifact_layers=ART_LAYERS, artifact_rel_tol=ARTIFACT_REL_TOL,
                   rms_eps=RMS_EPS, plane_layers=PLANE_LAYERS,
                   anchor=[ANCHOR_GAP_RATIO_L9, ANCHOR_TOL_GAP],
                   months_in_top24_gate=MONTHS_IN_TOP24_GATE, order_rho_gate=ORDER_RHO_GATE,
                   plant_layer=PLANT_LAYER, plant_heads=PLANT_HEADS,
                   plant_phis_deg=PLANT_PHIS_DEG, plant_cs=PLANT_CS,
                   plant_recall_sumcos2=PLANT_RECALL_SUMCOS2,
                   plant_detection_rate=PLANT_DETECTION_RATE,
                   cstar_unfit_above=CSTAR_UNFIT_ABOVE,
                   delta_sweep=DELTA_SWEEP, behav_phi_tol_deg=BEHAV_PHI_TOL_DEG,
                   seeds=dict(qk_set1="1000*L+h", ov_set1="1000*L+h+250000",
                              set2_offset=SEED_SET2_OFFSET, plant_base=SEED_PLANT,
                              gap_null=0, cyclic_order=0)),
               config=vars(a), status="STUB -- nothing computed yet")

    def write(o):
        o["runtime_seconds"] = round(time.time() - t_start, 1)
        outp.write_text(json.dumps(o, indent=2, default=jsonable))

    write(out)
    print("=" * 78)
    print("PHASE 3 / STAGE 4 -- rotational-operator census. Stub written to", a.out)
    print(f"declared_at = {out['declared_at']}   device = {DEVICE}")
    print(f"census layers {a.layers}  k={a.k}  n_null={a.n_null}")
    print("=" * 78, flush=True)

    # ---- pool (first cache read; the stub already exists) ---------------------
    root = REPO / a.root
    tok_all, cnt, pool_ids, _ = build_pool(root)
    assert len(tok_all) == N_SEQ_TOTAL * SEQ_LEN, f"tokens.i32 has {len(tok_all)} rows"
    tokens2d = tok_all.reshape(N_SEQ_TOTAL, SEQ_LEN)
    n_seq = min(a.n_seq, N_SEQ_TOTAL)
    row_of = {int(t): i for i, t in enumerate(pool_ids)}
    got_hash = member_hash([row_of[t] for t in SET13])
    assert got_hash == SET13_HASH, f"13-member set hash drifted: {got_hash}"
    token_map = json.loads((REPO / "results/token_map.json").read_text())
    pool_set = set(int(x) for x in pool_ids)
    strict12 = []
    for m in MONTHS:
        ip = sorted((int(x) for x in token_map["months"][m] if int(x) in pool_set),
                    key=lambda x: -int(cnt[x]))
        strict12.append(ip[0])
    print(f"pool {len(pool_ids)} types (count >= {COUNT_FLOOR}); 13-set hash OK; "
          f"strict-12 = {strict12}", flush=True)

    hf_token = os.environ.get("HF_TOKEN")
    model, tok = load_model(token=hf_token)
    print(f"model loaded: {type(model).__name__}  layer_types[:8]="
          f"{list(model.config.layer_types[:8])}", flush=True)

    # ================= STAGE 0 -- artifact ====================================
    t0 = time.time()
    npzp = REPO / a.npz_out
    reused = False
    if a.reuse_artifact and npzp.exists():
        z = np.load(npzp)
        if not np.array_equal(np.asarray(z["pool_ids"]), np.asarray(pool_ids, dtype=np.int64)):
            raise SystemExit(f"--reuse-artifact: pool_ids in {npzp} do not match build_pool")
        missing = [L for L in ART_LAYERS
                   if npz_key(L) not in z.files or rms_key(L) not in z.files]
        if missing:
            raise SystemExit(f"--reuse-artifact: {npzp} is missing layers {missing}; "
                             f"it was not written by this script's E1 stage")
        R = {L: np.asarray(z[npz_key(L)]) for L in ART_LAYERS}
        RMS = {L: np.asarray(z[rms_key(L)]) for L in ART_LAYERS}
        counts = np.asarray(z["counts"], dtype=np.int64)
        n_seq_art = int(z["n_seq"])
        reused = True
        print(f"[S0] reused artifact {npzp} (n_seq {n_seq_art})", flush=True)
        sec_per_batch, secs_s0, emb_check = None, 0.0, None
    else:
        print(f"\n[S0] forwarding {n_seq} sequences x 256 tokens, layers -1..25, "
              f"batch {a.batch}", flush=True)
        R, RMS, counts, sec_per_batch, secs_s0, emb_check = stage0_forwards(
            model, tok, tokens2d[:n_seq], pool_ids, ART_LAYERS, n_seq, a.batch)
        n_seq_art = n_seq
        npzp.parent.mkdir(parents=True, exist_ok=True)
        payload = dict(pool_ids=np.asarray(pool_ids, dtype=np.int64), counts=counts,
                       n_seq=np.int64(n_seq_art),
                       layers=np.asarray(ART_LAYERS, dtype=np.int64))
        payload.update({npz_key(L): R[L] for L in ART_LAYERS})
        payload.update({rms_key(L): RMS[L] for L in ART_LAYERS})
        np.savez(npzp, **payload)
        print(f"[S0] wrote {npzp} ({npzp.stat().st_size/1e6:.0f} MB)  "
              f"{sec_per_batch:.2f} s/batch  {secs_s0/60:.1f} min", flush=True)

    full_n = bool(n_seq_art == N_SEQ_TOTAL)
    fulln_path = REPO / FULLN_NPZ
    cmp_fulln = (compare_to_fulln(R, fulln_path) if (full_n and fulln_path.exists()) else
                 dict(source=str(fulln_path), passed=None,
                      note=("comparison run ONLY at --n-seq 15696 (E1): at smaller n the two "
                            "artifacts are means over different occurrence sets"),
                      n_seq=n_seq_art))
    cmp_fulln["batch"] = int(a.batch)
    cmp_fulln["bit_parity_requires"] = ("--batch 16 and the same sequence order as "
                                        "scripts/35b (both are the defaults)")
    readout_rows = np.flatnonzero(counts >= 1)
    out["stage0_artifact"] = dict(
        npz=str(a.npz_out), reused=reused, n_seq=n_seq_art, full_cache=full_n,
        layers=ART_LAYERS, n_pool=int(len(pool_ids)),
        n_readout_population=int(len(readout_rows)),
        readout_population_rule="pool types with count_seen >= 1 (E0)",
        readout_population_is_full_pool=bool(len(readout_rows) == len(pool_ids)),
        counts_match_corpus=(bool(np.array_equal(counts, cnt[pool_ids].astype(np.int64)))
                             if full_n else None),
        fulln_comparison=cmp_fulln, embedding_capture_check=emb_check,
        sec_per_batch=sec_per_batch, seconds=round(secs_s0, 1),
        layer_minus_one="input to layer 0 (scaled embedding output), forward pre-hook")
    write(out)
    print(f"[S0] readout population {len(readout_rows)}/{len(pool_ids)}; "
          f"fulln compare passed={cmp_fulln.get('passed')} "
          f"max_rel={cmp_fulln.get('max_abs_rel_diff')}", flush=True)
    if full_n:
        assert len(readout_rows) == len(pool_ids), "E0: at full n every pool type must be seen"
        if cmp_fulln.get("passed") is False:
            write(out)
            raise AssertionError(
                f"E1 artifact validation FAILED: max abs rel diff "
                f"{cmp_fulln['max_abs_rel_diff']:.3e} >= {ARTIFACT_REL_TOL}")
    print(f"[S0] {time.time()-t0:.1f}s", flush=True)

    # x_hat clouds (readout for head at L uses R[L-1]). Built on demand and kept in a
    # 2-entry cache: 26 layers x 3795 x 1152 in float64 would be ~0.9 GB, and one
    # HeadCloud is ~27 MB, so nothing here is retained across heads (see build_head).
    w_ins = {L: 1.0 + model.model.layers[L].input_layernorm.weight.detach().to(
        torch.float32).cpu().numpy().astype(np.float64) for L in a.layers}
    _xh = {}

    def get_xhat(L):
        if L not in _xh:
            if len(_xh) >= 2:
                _xh.clear()
            X = rms_norm_x(R[L - 1][readout_rows], w_ins[L])
            _xh[L] = (X, np.maximum((X ** 2).sum(1), 1e-300))
        return _xh[L]

    def build_head(L, h):
        """ops + the two subspace bases + the two clouds for one head (E5)."""
        ops = head_operators(model, L, h)
        Bq, rq = orth_basis(ops["Wq_eff"])                 # QK read subspace
        Bo, ro = orth_basis(ops["DOh"].T)                  # OV image (column) space
        X, n2 = get_xhat(L)
        cq = HeadCloud(X, n2, ops["Wq_eff"], ops["Wk_eff"], Bq)
        co = HeadCloud(X, n2, ops["Wq_eff"], ops["Wk_eff"], Bo)
        return ops, Bq, Bo, rq, ro, cq, co

    # ================= STAGE 1 -- proposer ====================================
    t0 = time.time()
    inventory, per_head_meta, checks = [], [], dict()
    for L in a.layers:
        for h in range(N_HEADS):
            ops = head_operators(model, L, h)
            planes, M, A = propose_head(ops)
            pairs, ev_all = ov_eigpairs(ops["Wv"], ops["DOh"])
            # E2 diagnostic: eigenvalues with post_attention_layernorm included (never selects)
            ev_post = np.linalg.eigvals(ops["Wv"] @ (ops["w_post"][:, None] * ops["DOh"]))
            cpx = ev_post[ev_post.imag > 1e-10]
            if len(cpx):
                j = int(np.argmax(np.abs(cpx)))
                post_diag = dict(top_abs_lambda=float(abs(cpx[j])),
                                 arg_deg=float(abs(np.degrees(np.angle(cpx[j])))),
                                 n_complex_pairs=int(len(cpx)))
            else:
                post_diag = dict(top_abs_lambda=None, arg_deg=None, n_complex_pairs=0)
            Bq, rq = orth_basis(ops["Wq_eff"])                 # QK read subspace
            Bo, ro = orth_basis(ops["DOh"].T)                  # OV image (column) space
            if not checks:
                # one-off: the factorisation identity the null's algebra relies on
                resid = np.linalg.norm(ops["Wq_eff"] - (ops["Wq_eff"] @ Bq) @ Bq.T)
                Xs = get_xhat(L)[0][:24]
                S_fac = (ops["Wq_eff"] @ Xs.T).T @ (ops["Wk_eff"] @ Xs.T)
                S_dir = Xs @ M @ Xs.T
                checks = dict(
                    layer=int(L), head=int(h),
                    wq_eff_minus_proj_frob=float(resid),
                    wq_eff_frob=float(np.linalg.norm(ops["Wq_eff"])),
                    S_factor_vs_xMx_max_abs=float(np.abs(S_fac - S_dir).max()),
                    S_scale=float(np.abs(S_dir).max()),
                    note=("E5 relies on q' = A0 + Ga(Q-I)Y being exact and on S being the "
                          "factor product; both are asserted here once."))
            for pi, p in enumerate(planes):
                inventory.append(dict(layer=int(L), head=int(h), plane=int(pi),
                                      family=p["family"], score=p["score"],
                                      phi_hat_deg=(None if p["phi_hat"] is None
                                                   else float(np.degrees(p["phi_hat"]))),
                                      detail=p["detail"]))
            per_head_meta.append(dict(layer=int(L), head=int(h), n_planes=len(planes),
                                      n_antisym=sum(1 for p in planes if p["family"] == "antisym"),
                                      n_symdeg=sum(1 for p in planes if p["family"] == "symdeg"),
                                      n_ov=sum(1 for p in planes if p["family"] == "ov"),
                                      rank_qk_read=int(rq), rank_ov_image=int(ro),
                                      n_ov_complex_pairs=int(len(pairs)),
                                      frob_M=float(np.linalg.norm(M)),
                                      frob_A=float(np.linalg.norm(A)),
                                      ov_postnorm_diagnostic=post_diag))
            # only the planes are retained across stages (~18 kB each); ops, the two
            # 1152x256 bases and the two clouds are rebuilt per head by build_head so that
            # 104 heads' worth of them (~7 GB) is never resident.
            per_head_meta[-1]["_planes"] = planes
            del ops, Bq, Bo, M, A, pairs, ev_all, ev_post
        print(f"  [S1] L{L:<2} proposed "
              f"{sum(m['n_planes'] for m in per_head_meta if m['layer']==L)} planes "
              f"({time.time()-t0:.1f}s cumulative)", flush=True)
    n_planes_total = len(inventory)
    out["stage1_proposer"] = dict(
        n_planes_frozen=n_planes_total,
        n_planes_max_possible=MAX_PLANES_PER_HEAD * N_HEADS * len(a.layers),
        inventory_hash=hashlib.md5(
            json.dumps([(r["layer"], r["head"], r["plane"], r["family"])
                        for r in inventory]).encode()).hexdigest()[:12],
        per_head=[{k: v for k, v in m.items() if not k.startswith("_")}
                  for m in per_head_meta],
        inventory=inventory, factorisation_checks=checks,
        note=("E3: the plane count is frozen here, before any readout number exists."),
        seconds=round(time.time() - t0, 1))
    write(out)
    print(f"[S1] {n_planes_total} planes frozen "
          f"({out['stage1_proposer']['inventory_hash']}) in {time.time()-t0:.1f}s", flush=True)

    # ================= STAGE 2 -- readout =====================================
    t0 = time.time()
    real_rows = []
    for m in per_head_meta:
        L, h = m["layer"], m["head"]
        planes = m["_planes"]
        if not planes:
            continue
        _, _, _, _, _, cq, co = build_head(L, h)
        X, n2 = get_xhat(L)
        for pi, p in enumerate(planes):
            Zt0 = (X @ p["U"]).T
            st, sel, th = plane_stats(Zt0, n2, cq.real_qk_fn(), p["phi_hat"],
                                      k=a.k, n_perm=a.n_perm, extras=True)
            st.update(layer=L, head=h, plane=pi, family=p["family"],
                      phi_hat_deg=(None if p["phi_hat"] is None
                                   else float(np.degrees(p["phi_hat"]))),
                      sel_token_ids=[int(pool_ids[readout_rows[i]]) for i in sel])
            real_rows.append(st)
            p["_Zt0"] = Zt0
    out["stage2_readout"] = dict(
        k=a.k, n_planes=len(real_rows),
        max_F1=float(max((r["F1"] for r in real_rows), default=0.0)),
        max_PC_ov=float(max((r["PC_ov"] for r in real_rows if r["PC_ov"] is not None),
                            default=0.0)),
        n_p_closed_lt_05=int(sum(1 for r in real_rows if r["p_closed"] < P_CLOSED_GATE)),
        planes=real_rows, seconds=round(time.time() - t0, 1))
    write(out)
    print(f"[S2] {len(real_rows)} planes read out in {time.time()-t0:.1f}s  "
          f"max F1 {out['stage2_readout']['max_F1']:.4f}  "
          f"max PC {out['stage2_readout']['max_PC_ov']:.4f}", flush=True)

    # ================= STAGE 3 -- cloud-rotation null =========================
    t0 = time.time()
    n_null = a.n_null
    reduction = None
    tau_rows, fa_rows = [], []
    timing = []
    heads_done = 0
    # E5 timing governor: measure on a small probe FIRST, so that every head runs at the
    # same draw count and the FA p floor is identical across the BH family.
    probe = None
    live_heads = [m for m in per_head_meta if m["_planes"]]
    if live_heads:
        m0 = live_heads[0]
        _, _, _, _, _, cq0, co0 = build_head(m0["layer"], m0["head"])
        n_probe = max(2, min(20, n_null))
        tp = time.time()
        run_head_null(cq0, co0, m0["_planes"], n_probe, 12345, 12345 + SEED_OV_OFFSET,
                      a.k, a.n_perm, [p["_Zt0"] for p in m0["_planes"]], taus=None)
        per_draw = (time.time() - tp) / n_probe
        proj_h = per_draw * 2 * n_null * len(live_heads) / 3600.0
        probe = dict(n_probe_draws=n_probe, seconds_per_draw=round(per_draw, 5),
                     n_live_heads=len(live_heads),
                     projected_hours_at_declared_n_null=round(proj_h, 3),
                     max_null_hours=a.max_null_hours, probe_seed=12345)
        print(f"  [S3] probe {n_probe} draws: {per_draw*1000:.1f} ms/draw -> "
              f"projected {proj_h:.2f} h for {len(live_heads)} heads x 2 x {n_null} draws",
              flush=True)
        if proj_h > a.max_null_hours and n_null > 500:
            reduction = dict(from_n_null=n_null, to_n_null=500, probe=probe,
                             p_floor_after=1.0 / 501,
                             rule=("E5 timing governor: Addendum E's permitted 500+500 "
                                   "reduction, triggered by the measured probe BEFORE any "
                                   "head is scored, so every head runs at the same n"))
            n_null = 500
            print("  [S3] TIMING GOVERNOR: n_null reduced to 500 (recorded)", flush=True)
    for m in per_head_meta:
        L, h = m["layer"], m["head"]
        planes = m["_planes"]
        if not planes:
            continue
        _, _, _, _, _, cq, co = build_head(L, h)
        Zt0s = [p["_Zt0"] for p in planes]
        th0 = time.time()
        s1, _ = run_head_null(cq, co, planes, n_null, 1000 * L + h,
                              1000 * L + h + SEED_OV_OFFSET, a.k, a.n_perm, Zt0s, taus=None)
        qk_idx = [i for i, p in enumerate(planes) if p["family"] != "ov"]
        ov_idx = [i for i, p in enumerate(planes) if p["family"] == "ov"]
        tau_F1 = float(max(s1[i] for i in qk_idx)) if qk_idx else None
        tau_PC = float(max(s1[i] for i in ov_idx)) if ov_idx else None
        _, fa = run_head_null(cq, co, planes, n_null,
                              1000 * L + h + SEED_SET2_OFFSET,
                              1000 * L + h + SEED_OV_OFFSET + SEED_SET2_OFFSET,
                              a.k, a.n_perm, Zt0s, taus=(tau_F1, tau_PC))
        dt = time.time() - th0
        timing.append(dt)
        tau_rows.append(dict(layer=L, head=h, n_draws=n_null, tau_F1=tau_F1, tau_PC=tau_PC,
                             tau_per_plane=[float(x) if np.isfinite(x) else None for x in s1],
                             seconds=round(dt, 2)))
        for i, p in enumerate(planes):
            fa_rate = float((1 + int(fa[i])) / (1 + n_null))
            rr = next(r for r in real_rows
                      if r["layer"] == L and r["head"] == h and r["plane"] == i)
            passed = conjunction(p["family"], rr, tau_F1, tau_PC)
            rr.update(tau_F1=tau_F1, tau_PC=tau_PC, tau_plane=float(s1[i]),
                      fa_count=int(fa[i]), fa_rate=fa_rate,
                      conjunction_pass=bool(passed),
                      p_conjunction=(fa_rate if passed else 1.0))
            fa_rows.append(dict(layer=L, head=h, plane=i, family=p["family"],
                                fa_rate=fa_rate, conjunction_pass=bool(passed)))
        heads_done += 1
        if heads_done == 1 or heads_done % 8 == 0:
            print(f"  [S3] {heads_done}/{len(live_heads)} heads  "
                  f"mean {np.mean(timing):.1f}s/head  "
                  f"eta {(len(live_heads)-heads_done)*np.mean(timing)/60:.1f} min",
                  flush=True)
    p_conj = np.array([r["p_conjunction"] for r in real_rows])
    keep = benjamini_hochberg(p_conj, alpha=BH_ALPHA) if len(p_conj) else np.zeros(0, bool)
    for r, kk in zip(real_rows, keep):
        r["bh_pass"] = bool(kk)
    conj_only = [r for r in real_rows if r["conjunction_pass"] and not r["bh_pass"]]
    passers = [r for r in real_rows if r["conjunction_pass"] and r["bh_pass"]]
    out["stage3_cloud_rotation_null"] = dict(
        n_null_declared=a.n_null, n_null_effective=n_null, reduction=reduction,
        timing_probe=probe, per_head_tau=tau_rows, per_plane_fa=fa_rows,
        seconds_per_head_mean=float(np.mean(timing)) if timing else None,
        n_conjunction_pass=int(sum(1 for r in real_rows if r["conjunction_pass"])),
        n_bh_pass=int(keep.sum()),
        n_conjunction_only=len(conj_only),
        bh_arithmetic=dict(
            n_planes=len(real_rows), alpha=BH_ALPHA,
            bh_rank1_threshold=float(BH_ALPHA / max(len(real_rows), 1)),
            addone_p_floor=float(1.0 / (n_null + 1)),
            min_floor_planes_for_any_bh_pass=int(np.ceil(
                (1.0 / (n_null + 1)) * len(real_rows) / BH_ALPHA)) if len(real_rows) else None,
            note=("E5: with n_null draws the add-one FA p floors at 1/(n_null+1), above the "
                  "BH rank-1 threshold, so BH survival requires several floor-p planes at "
                  "once. Declared before the numbers; not repaired post hoc.")),
        seconds=round(time.time() - t0, 1))
    write(out)
    print(f"[S3] conjunction pass {out['stage3_cloud_rotation_null']['n_conjunction_pass']}  "
          f"BH pass {int(keep.sum())}  ({time.time()-t0:.1f}s)", flush=True)

    # ================= STAGE 4 -- positive control ============================
    t0 = time.time()
    lm_layers = sorted(set(PLANE_LAYERS) | set(L - 1 for L in PLANE_LAYERS))
    lm_layers = [l for l in lm_layers if l >= 0]
    means = label_means_inline(model, tok, lm_layers, MONTHS, MT16)
    QP_lit, plane_info = {}, {}
    for L in PLANE_LAYERS:
        QP_lit[L], plane_info[str(L)] = plane_from_means(means[L])
    d9 = abs(plane_info["9"]["gap_ratio"] - ANCHOR_GAP_RATIO_L9)
    anchor = dict(layer=9, gap_ratio=plane_info["9"]["gap_ratio"],
                  gap_ratio_recorded=ANCHOR_GAP_RATIO_L9, abs_diff=float(d9),
                  tol=ANCHOR_TOL_GAP, passed=bool(d9 <= ANCHOR_TOL_GAP),
                  rule="scripts/35 D8 anchor gate, re-asserted and REPORTED")
    QP_xhat = {}
    for L in PLANE_LAYERS:
        if (L - 1) in means:
            w_in = (1.0 + model.model.layers[L].input_layernorm.weight.detach()
                    .to(torch.float32).cpu().numpy().astype(np.float64))
            QP_xhat[L], _ = plane_from_means(rms_norm_x(means[L - 1], w_in))

    month_rows = {}
    for i, t in enumerate(strict12):
        r = row_of[int(t)]
        w = np.flatnonzero(readout_rows == r)
        month_rows[i] = int(w[0]) if len(w) else None
    ctrl_readout = []
    for L in PLANE_LAYERS:
        if L not in a.layers:
            continue
        X, n2 = get_xhat(L)
        heads_cq = {h: build_head(L, h)[5] for h in range(N_HEADS)}
        for vname, QPd in (("literal", QP_lit), ("xhat", QP_xhat)):
            if L not in QPd:
                continue
            U = QPd[L]
            Zt = (X @ U).T
            sel, _ = select_topk(Zt, n2, a.k)
            sset = set(int(x) for x in sel)
            hits = [i for i in range(12) if month_rows[i] is not None
                    and month_rows[i] in sset]
            cal = [i for i in range(12) if month_rows[i] is not None]
            mr = [month_rows[i] for i in cal]
            if len(mr) >= 3:
                zc = Zt[:, mr]
                th = np.arctan2(zc[1], zc[0])
                pcl = gap_p(len(mr), max_gap_rad(th), a.n_perm)
                mgd = float(np.degrees(max_gap_rad(th)))
                # label_index = the survivors' TRUE calendar positions on the 12-cycle
                order = cyclic_order_score(th, np.asarray(cal), len(mr),
                                           n_perm=a.n_perm, seed=0, full_k=12)
            else:   # only reachable in a tiny smoke run, where most types are unseen
                pcl, mgd = 1.0, None
                order = dict(rho=-2.0, p=1.0, symmetry=None)
            heads = []
            for h in range(N_HEADS):
                cq = heads_cq[h]
                st, _, _ = plane_stats(Zt, n2, cq.real_qk_fn(), None, k=a.k, n_perm=a.n_perm)
                heads.append(dict(head=h, F1=st["F1"], P1=st["P1"],
                                  PC_qk=st["PC_qk"], p_closed=st["p_closed"]))
            ctrl_readout.append(dict(
                layer=L, variant=vname, n_months_in_top24=len(hits),
                months_in_top24=hits, n_months_scorable=len(cal), month_p_closed=pcl,
                month_max_gap_deg=mgd,
                month_order_rho=order["rho"], month_order_p=order["p"],
                gate_pass=bool(len(hits) >= MONTHS_IN_TOP24_GATE and pcl < P_CLOSED_GATE
                               and order["rho"] >= ORDER_RHO_GATE),
                per_head=heads))
        print(f"  [S4i] L{L} " + "  ".join(
            f"{c['variant']}: {c['n_months_in_top24']}/12 p {c['month_p_closed']:.3g} "
            f"rho {c['month_order_rho']:.3f} -> {'PASS' if c['gate_pass'] else 'fail'}"
            for c in ctrl_readout if c["layer"] == L), flush=True)
    lit_ok = any(c["gate_pass"] for c in ctrl_readout if c["variant"] == "literal")
    xh_ok = any(c["gate_pass"] for c in ctrl_readout if c["variant"] == "xhat")
    if lit_ok or xh_ok:
        readout_reading = ("readout validated" if lit_ok else
                           "QP coordinate mismatch (xhat variant passes, literal does not)")
    else:
        readout_reading = "selection-limited (both QP variants fail the E6(i) gate)"

    # ---- (ii) proposer half: planted rotor ----------------------------------
    plant_cells, plant_qp = [], (QP_xhat.get(PLANT_LAYER) if PLANT_LAYER in QP_xhat else None)
    plant_note = None
    if plant_qp is None or PLANT_LAYER not in a.layers:
        plant_note = (f"planted control not run: L{PLANT_LAYER} not in --layers or the xhat "
                      f"QP at L{PLANT_LAYER} is unavailable")
        print(f"  [S4ii] {plant_note}", flush=True)
    else:
        grid = [(phi, c) for phi in PLANT_PHIS_DEG for c in PLANT_CS]
        if a.n_plant > 0:
            grid = grid[:a.n_plant]
        X, n2 = get_xhat(PLANT_LAYER)
        for hh in PLANT_HEADS:
            ops = head_operators(model, PLANT_LAYER, hh)
            M0 = ops["Wq_eff"].T @ ops["Wk_eff"]
            WOV0 = ops["DOh"] @ ops["Wv"]
            s1M = float(np.linalg.svd(M0, compute_uv=False)[0])
            s1O = float(np.linalg.svd(WOV0, compute_uv=False)[0])
            for ci, (phid, c) in enumerate(grid):
                phi = np.radians(phid)
                Rphi = np.array([[np.cos(phi), -np.sin(phi)], [np.sin(phi), np.cos(phi)]])
                Pm = c * s1M * (plant_qp @ Rphi @ plant_qp.T)
                Po = c * s1O * (plant_qp @ Rphi @ plant_qp.T)
                Wq_p = np.vstack([ops["Wq_eff"], c * s1M * (Rphi.T @ plant_qp.T)])
                Wk_p = np.vstack([ops["Wk_eff"], plant_qp.T])
                fac_err = float(np.abs(Wq_p.T @ Wk_p - (M0 + Pm)).max())
                DOh_p = np.hstack([ops["DOh"], (c * s1O) * plant_qp])
                Wv_p = np.vstack([ops["Wv"], Rphi @ plant_qp.T])
                ops_p = dict(ops); ops_p.update(Wq_eff=Wq_p, Wk_eff=Wk_p,
                                                Wv=Wv_p, DOh=DOh_p)
                planes_p, Mp, _ = propose_head(ops_p, M=M0 + Pm)
                recall = {f: False for f in ("antisym", "symdeg", "ov")}
                best_sc = 0.0
                for p in planes_p:
                    sc = sumcos2(p["U"], plant_qp)
                    p["_sc"] = sc
                    best_sc = max(best_sc, sc)
                    if sc >= PLANT_RECALL_SUMCOS2:
                        recall[p["family"]] = True
                Bq, _ = orth_basis(Wq_p)
                Bo, _ = orth_basis(DOh_p.T)
                cqp = HeadCloud(X, n2, Wq_p, Wk_p, Bq)
                cop = HeadCloud(X, n2, Wq_p, Wk_p, Bo)
                Zt0s, stats_p = [], []
                for p in planes_p:
                    Zt0 = (X @ p["U"]).T
                    Zt0s.append(Zt0)
                    st, _, _ = plane_stats(Zt0, n2, cqp.real_qk_fn(), p["phi_hat"],
                                           k=a.k, n_perm=a.n_perm)
                    stats_p.append(st)
                sd = SEED_PLANT + 1000 * (hh * 100 + ci)
                s1n, _ = run_head_null(cqp, cop, planes_p, a.n_null_plant, sd,
                                       sd + SEED_OV_OFFSET, a.k, a.n_perm, Zt0s, taus=None)
                qk_i = [i for i, p in enumerate(planes_p) if p["family"] != "ov"]
                ov_i = [i for i, p in enumerate(planes_p) if p["family"] == "ov"]
                tF = float(max(s1n[i] for i in qk_i)) if qk_i else None
                tP = float(max(s1n[i] for i in ov_i)) if ov_i else None
                det = {f: False for f in ("antisym", "symdeg", "ov")}
                for p, st in zip(planes_p, stats_p):
                    if conjunction(p["family"], st, tF, tP):
                        det[p["family"]] = True
                plant_cells.append(dict(
                    head=hh, phi_deg=phid, c=c, s1_M=s1M, s1_OV=s1O,
                    factorisation_max_abs_err=fac_err,
                    n_planes=len(planes_p), best_sumcos2_with_QP=best_sc,
                    proposer_recall_any=bool(any(recall.values())),
                    proposer_recall_by_family=recall,
                    detection_any=bool(any(det.values())), detection_by_family=det,
                    detection_qk=bool(det["antisym"] or det["symdeg"]), detection_ov=det["ov"],
                    tau_F1=tF, tau_PC=tP,
                    stats=[dict(family=p["family"], sumcos2_QP=p["_sc"], F1=st["F1"],
                                P1=st["P1"], PC_ov=st["PC_ov"], p_closed=st["p_closed"])
                           for p, st in zip(planes_p, stats_p)]))
            print(f"  [S4ii] head {hh}: {len(grid)} cells done "
                  f"({time.time()-t0:.0f}s cumulative)", flush=True)

    def rate_at(c, key):
        cells = [x for x in plant_cells if x["c"] == c]
        return (float(np.mean([x[key] for x in cells])) if cells else None), len(cells)

    cstar, cstar_qk, cstar_ov, plant_curve = None, None, None, []
    for c in PLANT_CS:
        r_any, n = rate_at(c, "detection_any")
        r_qk, _ = rate_at(c, "detection_qk")
        r_ov, _ = rate_at(c, "detection_ov")
        r_rec, _ = rate_at(c, "proposer_recall_any")
        plant_curve.append(dict(c=c, n_cells=n, detection_rate=r_any,
                                detection_rate_qk=r_qk, detection_rate_ov=r_ov,
                                proposer_recall_rate=r_rec))
        if r_any is not None and r_any >= PLANT_DETECTION_RATE and cstar is None:
            cstar = c
        if r_qk is not None and r_qk >= PLANT_DETECTION_RATE and cstar_qk is None:
            cstar_qk = c
        if r_ov is not None and r_ov >= PLANT_DETECTION_RATE and cstar_ov is None:
            cstar_ov = c
    unfit = (cstar is None or cstar > CSTAR_UNFIT_ABOVE) if plant_cells else None
    out["stage4_positive_control"] = dict(
        readout_half=dict(
            planes=plane_info, anchor_gate=anchor, per_cell=ctrl_readout,
            literal_pass=lit_ok, xhat_pass=xh_ok, reading=readout_reading,
            qp_variant_sumcos2={str(L): (sumcos2(QP_lit[L], QP_xhat[L])
                                         if L in QP_xhat else None)
                                for L in PLANE_LAYERS},
            gate=("E6(i): >= 8/12 months in the top-24 in-plane-energy selection AND "
                  "month p_closed < 0.05 AND cyclic order rho >= 0.9")),
        proposer_half=dict(
            note=plant_note, layer=PLANT_LAYER, heads=PLANT_HEADS,
            n_cells=len(plant_cells), n_null_plant=a.n_null_plant,
            curve=plant_curve, c_star=cstar, c_star_qk=cstar_qk, c_star_ov=cstar_ov,
            instrument_unfit=unfit,
            gate=(f"E6(ii): c* = smallest c with detection rate >= "
                  f"{PLANT_DETECTION_RATE}; instrument unfit if c* > {CSTAR_UNFIT_ABOVE}"),
            cells=plant_cells),
        seconds=round(time.time() - t0, 1))
    write(out)
    print(f"[S4] readout: {readout_reading}   c* = {cstar}  unfit = {unfit}  "
          f"({time.time()-t0:.1f}s)", flush=True)

    # ================= STAGE 5 -- passer-only arms ============================
    t0 = time.time()
    passer_arms = []
    tested = passers if passers else []
    for r in tested:
        L, h, pi = r["layer"], r["head"], r["plane"]
        m = next(x for x in per_head_meta if x["layer"] == L and x["head"] == h)
        p = m["_planes"][pi]
        ops, Bq, _, _, _, cq, _ = build_head(L, h)
        X, n2 = get_xhat(L)
        sel = np.asarray(r["sel"], dtype=np.int64)
        th = np.radians(np.asarray(r["theta_deg"], dtype=np.float64))
        # (1) frequency-pseudo-angle null
        cts = cnt[pool_ids[readout_rows[sel]]].astype(np.float64)
        rank = np.argsort(np.argsort(cts))
        th_freq = 2 * np.pi * rank / len(sel)
        stf, _, _ = plane_stats(p["_Zt0"], n2, cq.real_qk_fn(), p["phi_hat"],
                                k=a.k, n_perm=a.n_perm, angles_override=th_freq)
        # (2) cosine-normalised refit
        q0, k0 = cq.real_qk_fn()(sel)
        qn = q0 / np.maximum(np.linalg.norm(q0, axis=0, keepdims=True), 1e-300)
        kn = k0 / np.maximum(np.linalg.norm(k0, axis=0, keepdims=True), 1e-300)
        Sc = two_way_centre(qn.T @ kn)
        D = th[:, None] - th[None, :]
        F1c, phic, _ = fit_h1(D, Sc.ravel())
        P1c, _ = fit_harmonics(D, Sc.ravel())
        # (3) empirical per-type RMS arm
        rms_emp = np.maximum(RMS[L - 1][readout_rows].astype(np.float64), 1e-30)
        Xe = (R[L - 1][readout_rows].astype(np.float64) / rms_emp[:, None]) * w_ins[L][None, :]
        n2e = np.maximum((Xe ** 2).sum(1), 1e-300)
        cqe = HeadCloud(Xe, n2e, ops["Wq_eff"], ops["Wk_eff"], Bq)
        ste, _, _ = plane_stats((Xe @ p["U"]).T, n2e, cqe.real_qk_fn(), p["phi_hat"],
                                k=a.k, n_perm=a.n_perm)
        # (4) Delta-sweep
        lt, ivf = layer_inv_freq(model, L)
        dsw = []
        for dlt in DELTA_SWEEP:
            Md = ops["Wq_eff"].T @ (rope_rot(ivf, dlt) @ ops["Wk_eff"])
            pl_d, _, _ = propose_head(ops, M=Md)
            same = [pp for pp in pl_d if pp["family"] == p["family"]]
            ov = max((sumcos2(pp["U"], p["U"]) for pp in same), default=None)
            best = None
            for pp in same:
                Ztd = (X @ pp["U"]).T
                std, _, _ = plane_stats(Ztd, n2, cq.real_qk_fn(), pp["phi_hat"],
                                        k=a.k, n_perm=a.n_perm)
                v = std["PC_ov"] if pp["family"] == "ov" else std["F1"]
                if best is None or (v is not None and v > best[0]):
                    best = (v, std)
            dsw.append(dict(delta=dlt, sumcos2_with_delta0_plane=ov,
                            best_stat=(best[0] if best else None),
                            stats=(best[1] if best else None)))
        passer_arms.append(dict(
            layer=L, head=h, plane=pi, family=r["family"],
            real=dict(F1=r["F1"], P1=r["P1"], PC_ov=r["PC_ov"], p_closed=r["p_closed"]),
            frequency_pseudo_angle=stf,
            cosine_normalised=dict(F1=F1c, P1=P1c, phi_F1_deg=float(np.degrees(phic))),
            empirical_rms=ste, delta_sweep=dict(layer_type=lt, cells=dsw),
            labels_after_freeze=dict(
                token_ids=r["sel_token_ids"],
                token_strings=[tok.decode([i]) for i in r["sel_token_ids"]])))
    out["stage5_passer_arms"] = dict(
        n_passers=len(tested), arms=passer_arms,
        note=("E7. Empty when no plane survives the conjunction + BH; the conjunction-only "
              "planes are listed in stage3 and are non-verdict annotation."),
        conjunction_only=[dict(layer=r["layer"], head=r["head"], plane=r["plane"],
                               family=r["family"], F1=r["F1"], PC_ov=r["PC_ov"],
                               P1=r["P1"], p_closed=r["p_closed"], fa_rate=r["fa_rate"])
                          for r in conj_only],
        seconds=round(time.time() - t0, 1))
    write(out)
    print(f"[S5] {len(tested)} passers, {len(conj_only)} conjunction-only "
          f"({time.time()-t0:.1f}s)", flush=True)

    # ================= STAGE 6 -- behavioural confirmation ====================
    t0 = time.time()
    behav = []
    if passers:
        for r in passers:
            L, h = r["layer"], r["head"]
            ids24 = np.asarray(r["sel_token_ids"], dtype=np.int64)
            th = np.radians(np.asarray(r["theta_deg"], dtype=np.float64))
            ang_of = {int(t): float(x) for t, x in zip(ids24, th)}
            store = {}

            def hook(_m, _i, o):
                store["w"] = o[1].detach() if isinstance(o, tuple) else None
            hnd = model.model.layers[L].self_attn.register_forward_hook(hook)
            nb = min(a.n_behav, N_SEQ_TOTAL)
            dth, prob, rel = [], [], []
            try:
                with torch.inference_mode():
                    for lo in range(0, nb, a.behav_batch):
                        blk = tokens2d[lo:min(lo + a.behav_batch, nb)].astype(np.int64)
                        ids = np.concatenate(
                            [np.full((len(blk), 1), int(tok.bos_token_id), dtype=np.int64),
                             blk], axis=1)
                        model(torch.from_numpy(ids).to(DEVICE), use_cache=False)
                        W = store.get("w")
                        if W is None:
                            break
                        Wn = W[:, h].to(torch.float32).cpu().numpy()
                        for b in range(len(blk)):
                            pos = np.flatnonzero(np.isin(blk[b], ids24)) + 1
                            if len(pos) < 2:
                                continue
                            for si in range(len(pos)):
                                for ti in range(si):
                                    s, t = int(pos[si]), int(pos[ti])
                                    dth.append(ang_of[int(ids[b, s])] - ang_of[int(ids[b, t])])
                                    prob.append(float(Wn[b, s, t]))
                                    rel.append(s - t)
            finally:
                hnd.remove()
            if len(prob) < 50:
                behav.append(dict(layer=L, head=h, plane=r["plane"], n_pairs=len(prob),
                                  note="too few in-selection pairs for a fit"))
                continue
            dth = np.asarray(dth); prob = np.asarray(prob); rel = np.asarray(rel)
            nb_ = 24
            bi = np.clip(((dth + np.pi) / (2 * np.pi) * nb_).astype(int), 0, nb_ - 1)
            ctr = (np.arange(nb_) + 0.5) / nb_ * 2 * np.pi - np.pi

            def cosfit(y, w):
                good = w > 0
                Xd = np.column_stack([np.ones(good.sum()), np.cos(ctr[good]),
                                      np.sin(ctr[good])])
                beta, *_ = np.linalg.lstsq(Xd, y[good], rcond=None)
                res = y[good] - Xd @ beta
                yy = y[good] - y[good].mean()
                return (float(np.degrees(np.arctan2(beta[2], beta[1]))),
                        float(1 - (res @ res) / max(yy @ yy, 1e-300)),
                        [float(x) for x in beta])

            wct = np.bincount(bi, minlength=nb_)
            mp = np.bincount(bi, weights=prob, minlength=nb_) / np.maximum(wct, 1)
            ml = np.bincount(bi, weights=np.log(np.maximum(prob, 1e-12)),
                             minlength=nb_) / np.maximum(wct, 1)
            phi_p, r2_p, bp = cosfit(mp, wct)
            phi_l, r2_l, bl = cosfit(ml, wct)
            band = (rel >= 1) & (rel <= 64)
            if band.sum() > 50:
                wb = np.bincount(bi[band], minlength=nb_)
                mb = np.bincount(bi[band], weights=prob[band],
                                 minlength=nb_) / np.maximum(wb, 1)
                phi_b, r2_b, _ = cosfit(mb, wb)
            else:
                phi_b, r2_b = None, None
            phw = r["phi_hat_deg"] if r["phi_hat_deg"] is not None else r["phi_F1_deg"]
            relb = np.arange(0, 256, 16)
            ridx = np.clip(np.searchsorted(relb, rel, side="right") - 1, 0, len(relb) - 1)
            rc = np.bincount(ridx, minlength=len(relb))
            rp = np.bincount(ridx, weights=prob, minlength=len(relb)) / np.maximum(rc, 1)
            behav.append(dict(
                layer=L, head=h, plane=r["plane"], n_pairs=int(len(prob)),
                n_sequences=int(nb), phi_weight_deg=phw,
                phi_behav_prob_deg=phi_p, r2_prob=r2_p, beta_prob=bp,
                phi_behav_logprob_deg=phi_l, r2_logprob=r2_l, beta_logprob=bl,
                phi_behav_band_deg=phi_b, r2_band=r2_b,
                relative_position_profile=dict(
                    bin_lo=[int(x) for x in relb], mean_prob=[float(x) for x in rp],
                    n=[int(x) for x in rc]),
                agrees=(None if phw is None else
                        bool(abs(float(np.degrees(wrap_pi(np.radians(phi_p - phw)))))
                             <= BEHAV_PHI_TOL_DEG)),
                rule=f"E8: |phi_behav - phi_hat| <= {BEHAV_PHI_TOL_DEG} deg"))
    out["stage6_behavioural"] = dict(
        n_passers=len(passers), cells=behav,
        note="E8; skipped when there are no passers.",
        seconds=round(time.time() - t0, 1))
    write(out)
    print(f"[S6] {len(behav)} behavioural cells ({time.time()-t0:.1f}s)", flush=True)

    # ================= VERDICT ================================================
    control_ok = (bool(lit_ok or xh_ok) and unfit is False)
    if not control_ok:
        verdict = "INSTRUMENT NOT VALIDATED -- no blind verdict (prereg R2)"
    elif passers:
        verdict = (f"{len(passers)} weight-proposed plane(s) survive the cloud-rotation "
                   f"conjunction and BH; see stage5_passer_arms")
    else:
        verdict = ("no SINGLE head implements angular matching or rotational transport on "
                   "any plane above c* (MQA caveat: rotors split across heads or across "
                   "attention+MLP are invisible here)")
    out["verdict"] = dict(
        text=verdict, control_readout=readout_reading,
        control_readout_pass=bool(lit_ok or xh_ok),
        control_proposer_c_star=cstar, control_instrument_unfit=unfit,
        n_planes_tested=len(real_rows), n_conjunction_pass=len(passers) + len(conj_only),
        n_bh_pass=len(passers),
        pre_committed=("positive -- a weight-proposed plane whose occupying population "
                       "closes and survives the passer nulls; negative -- 'no SINGLE head "
                       "implements angular matching or rotational transport on any plane "
                       "above c*'"))
    out["status"] = "complete"
    write(out)

    print("\n" + "=" * 78)
    print(f"planes frozen {n_planes_total} | conjunction {len(passers)+len(conj_only)} | "
          f"BH {len(passers)} | c* {cstar} | control {readout_reading}")
    print("VERDICT:", verdict)
    secs = round(time.time() - t_start, 1)
    print(f"wrote {a.out} ({secs}s)", flush=True)


if __name__ == "__main__":
    main()
