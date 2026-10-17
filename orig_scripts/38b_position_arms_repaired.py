"""Stage 2, avenue 4, REPAIRED -- WHAT DOES THE POSITION COORDINATE ACTUALLY TRACK?

Pre-registered in docs/phase3_prereg_addenda_2026-08-21.md Addendum D (frozen 2026-08-21,
before this script's first run); the design it repairs is docs/phase3_prereg_2026-08-21.md
section 3.2, and the repair discipline is section 1 R1/R2/R5 (declaration stub first, no
post-hoc re-tuning, corrections are new NNb files). Writes
results/phase3_position_arms_repaired.json plus results/phase3_position_arms_repaired_data.npz.

PROVENANCE -- WHY THIS SCRIPT EXISTS. verify38 refuted scripts/38's arm (a). That arm shifted
ALL position_ids, BOS included, by j. In gemma-3-1b-pt that is an EXACT SYMMETRY: RoPE
attention logits depend only on the difference of two positions, and the causal and sliding
masks in transformers 5.15.0 are built from query/key INDICES (masking_utils reads
q_idx/kv_idx), not from position_ids. A uniform shift therefore changes nothing the model can
see. verify38 confirmed it numerically with an fp32 probe: relative residual change 1.6e-7,
i.e. fp32 round-off. scripts/38's own log reports a bf16 max-abs difference of 96 for that
same arm -- bf16 rounding on a shifted RoPE table manufactures a large-looking difference out
of an exact symmetry, which is exactly why THIS script re-runs the fp32 probe rather than
trusting a bf16 delta. arm (a)'s beta_abs ~ 0 was consequently uninformative: it was measuring
nothing. verify38 also found (2) scripts/38 arm (b)'s slope is content-driven (a repeated
" the" filler prefix yields 5% of the real-text slope, CI containing 0) and (3) that the
read-out coordinate u_L is close to the residual NORM direction (corr 0.986 at L9, 0.75 at L6).

THE THREE REPAIRED ARMS (Addendum D). All three move j in {0, 16, 32, 64, 128} over the same
300 cached documents truncated to their first 120 tokens, and all three read the same 100
doc-relative positions i in 20..119, so they differ only in WHICH confound they move.

  (a') BOS-PINNED SHIFT. ids = [BOS] + doc; position_ids = [0] + (arange(1, 121) + j). BOS
       stays at position 0 while the document's positions slide away from it. The token ids
       are unchanged and the attended key SET is unchanged (121 keys, sliding window 512 >
       every length here), so the only thing that moves is the document's DISTANCE TO BOS --
       and, unlike a uniform shift, that is a real change to the position differences the
       model attends over. It is mildly off-distribution by construction: no training
       sequence has a gap between BOS and its successor.
  (b') FILLER PREFIX. ids = [BOS] + (" the" x j) + doc. Distance to BOS moves AND the
       attended key count moves, but the prefix carries no document content. This is
       verify38's own diagnostic promoted to a registered arm.
  (b)  REAL-TEXT PREFIX. ids = [BOS] + (first j tokens of a different cached document) + doc,
       the section 3.2 arm re-run on the same documents with scripts/38's donor draw (seed 1),
       so the three arms are compared on a shared document sample. Distance, key count AND
       content all move.

The contrast is the identification: (a') moves distance alone, (b') adds key count, (b) adds
content. Addendum D's frozen call reads the three beta CIs in that nesting order and nothing
else -- there is no beta_rel, no ratio and no cache arm in this script, because the section
3.2 absolute-vs-document-relative dichotomy is retired as untestable by uniform shifts.

TWO READ-OUTS, BECAUSE THE COORDINATE MAY BE THE NORM. c = u_L . resid is the registered
read-out. verify38 item (3) says u_L is largely the residual-norm direction at L9, so a slope
in c could be a slope in overall residual magnitude rather than in a position code. Addendum D
therefore adds c_perp = (u_L - (u_L . n_hat) n_hat) . resid with n_hat the unit mean residual
direction, measured here from the j=0 forwards of these same 300 documents at that layer and
over the same read positions. Every call is reported on both.

WHAT IS COPIED AND FROM WHERE. Copied from scripts/38_position_causal.py, not imported (R5:
nothing existing is edited, and a verifier must be able to diff the two estimators):
build_seq_batch (L454), make_rows (L473) extended to three arms, fe_slope_shifts (L486),
boot_ci (L532), _unit/_pearson (L372/L377), the forward-sweep batching loop (L841), the
declaration-stub-before-any-read discipline (L570, prereg R1) and the JSON layout. Forwards
and captures go through src/gfdisc/rig.py unchanged (forward_capture passes position_ids
straight through, which is the hook arm (a') hangs on).

A NOTE ON PADDING: every (arm, shift) cell has a single sequence length -- 121 for (a'),
121 + j for (b') and (b) -- so the batches contain no pad columns at all, and the explicit
position_ids of arm (a') therefore never lands on a padded position.
"""
import os
from pathlib import Path as _Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_HOME", str(_Path(__file__).resolve().parents[1] / "cache" / "hf"))

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import gfdisc.rig as rig                                    # shared Stage-2 rig, unmodified
from gfdisc.env import DEVICE
from gfdisc.model import load_model


# ===========================================================================
# frozen constants -- Addendum D
# ===========================================================================
LAYERS = [6, 9]                       # L6 primary, L9 co-reported
D_MODEL = 1152

N_DOCS = 300                          # documents, taken from scripts/38's npz
DOC_TOKENS = 120                      # doc = tokens2d[doc, :120]
SHIFTS = [0, 16, 32, 64, 128]
I_LO, I_HI = 20, 119                  # doc-relative read positions, inclusive
N_BOOT = 2000                         # cluster bootstrap draws over documents
CI_LO, CI_HI = 2.5, 97.5              # percentile CI

ARM_A = "a_prime_bos_pinned_shift"
ARM_BP = "b_prime_filler_prefix"
ARM_B = "b_real_text_prefix"
ARMS = [ARM_A, ARM_BP, ARM_B]
COORDS = ["c", "c_perp"]

FILLER_TEXT = " the"                  # Addendum D: (" the" x j)

SEED_OTHER = 1                        # scripts/38 SEED_OTHER -- "seed as 38"
SEED_BOOT_DOCS = 20260823             # scripts/38 SEED_BOOT_DOCS

DATA_NPZ = "results/phase3_position_causal_data.npz"   # u_L and the 300 document ids
DEFAULT_OUT = "results/phase3_position_arms_repaired.json"
DEFAULT_DATA_OUT = "results/phase3_position_arms_repaired_data.npz"

# bf16 forward checks (Addendum D / the task's verification demands)
MAX_ABS_DIFF_DEFAULT = 0.0            # j=0 with explicit position_ids IS the default forward
MIN_ABS_DIFF_SHIFT = 1e-2             # scripts/38 MIN_ABS_DIFF_SHIFT
VERIFY_SHIFT = 16

# fp32 symmetry probe -- the check verify38 used, and the ONLY discriminating one
FP32_REL_PASS = 1e-4                  # rel change above this => arm (a') is not a symmetry
FP32_REL_SYMMETRY = 1e-6              # rel change below this => arm (a') IS a symmetry: STOP
N_FP32_DOCS = 4


def declared_rules(a, layers, shifts):
    """Everything frozen before the run, written to disk before anything is read."""
    return dict(
        addendum_D_text=(
            "Defects (verify38): (1) section 3.2 arm (a) -- a uniform shift of ALL "
            "position_ids including BOS -- is an exact symmetry of gemma-3-1b-pt (RoPE logits "
            "depend only on position differences; masks are index-based; verified in "
            "transformers 5.15.0 source and by an fp32 probe: relative residual change 1.6e-7). "
            "Its beta_abs ~ 0 carries no evidential weight. (2) Arm (b)'s slope is "
            "content-driven: a repeated-filler prefix (\" the\" x j) yields 5% of the real-text "
            "slope with a CI containing 0. (3) At L9 the read-out coordinate u_L is ~residual "
            "norm (corr 0.986); at L6 partly (0.75). (4) The Addendum-B bin_svd repair is "
            "directionally mis-specified (p_emp 1.0 by construction); bin_r2 is the usable "
            "non-verdict diagnostic. "
            "38b design (300 cached docs, doc[:120], layers 6 and 9, u_L from 38's npz, same "
            "FE/bootstrap estimator; j in {0,16,32,64,128}): (a') BOS-pinned shift: ids = "
            "[BOS]+doc, position_ids = [0] + (arange(1,121) + j). Breaks the symmetry: the "
            "document's distance to BOS changes, content and key count do not. (b') filler "
            "prefix: [BOS] + (\" the\" x j) + doc: distance to BOS and attended-key count "
            "change, content does not. (b) real-text prefix as in section 3.2 (re-run on the "
            "same docs so all three arms share docs). Read c = u_L . resid at doc-relative i "
            "in 20..119 for each arm; beta per arm with (doc,i) fixed effects; cluster "
            "bootstrap over docs, 2,000 draws. Also report beta on the norm-orthogonalised "
            "coordinate c_perp = (u_L - (u_L.n_hat)n_hat).resid with n_hat the mean residual "
            "direction, to separate the coordinate from residual norm. Calls (per layer, L6 "
            "primary): DISTANCE-TO-BOS CODE iff beta(a') CI excludes 0; CONTENT-ACCUMULATION "
            "iff beta(b) CI excludes 0 while beta(a') and beta(b') CIs contain 0; KEY-COUNT "
            "iff beta(b') CI excludes 0 while beta(a') contains 0; else UNIDENTIFIED. The "
            "section 3.2 absolute-vs-document-relative dichotomy is retired as not testable by "
            "uniform shifts."),
        provenance=(
            "scripts/38_position_causal.py arm (a) was refuted by verify38: a uniform shift of "
            "every position_id including BOS is an exact symmetry of this model (RoPE reads "
            "position DIFFERENCES; transformers 5.15.0 builds the causal and sliding masks "
            "from query/key indices, not position_ids), so that arm's beta_abs ~ 0 measured "
            "nothing. verify38's fp32 probe put the relative residual change at 1.6e-7 -- "
            "fp32 round-off -- while scripts/38's own bf16 check reported a max-abs difference "
            "of 96 for the same forwards, which is why the bf16 delta is NOT accepted here as "
            "evidence that an arm moves the model. This script re-runs that fp32 probe on the "
            "REPAIRED arm (a') and, as a negative control on the probe itself, on the retired "
            "uniform shift. It reuses scripts/38's u_L and its 300 document ids "
            f"({DATA_NPZ}) and its donor draw (seed {SEED_OTHER}) so the repaired arms are "
            "measured against the same coordinate and the same documents. Nothing in "
            "scripts/38 or any other existing file is edited (prereg R5)."),
        inputs=dict(
            npz=DATA_NPZ,
            keys_used=[f"u_L_layer{L}" for L in layers] + ["docs", "others (provenance assert)"],
            u_L=("the verdict-bearing read-out direction, taken verbatim from scripts/38: top "
                 "right-singular vector of the (133 types x 8 bins) x 1152 within-type-centred, "
                 "per-row-L2 matrix of cache-arm mean SAE codes projected through w_dec, sign "
                 "fixed so its correlation with bin index is positive. NOT recomputed here"),
            docs=("the 300 cached document ids scripts/38 drew with seed 0; loaded, not "
                  "redrawn, so the repaired arms and scripts/38's arms share documents"),
        ),
        arms=dict(
            a_prime=("BOS-pinned shift: ids = [BOS] + tokens2d[doc, :120] (121 tokens), full "
                     "attention, position_ids = [0] + (arange(1, 121) + j). BOS is PINNED at "
                     "position 0 while the document slides; token ids and the attended key set "
                     "are unchanged. Registered as mildly off-distribution: at j > 0 there is a "
                     "positional gap between BOS and the first document token, which no "
                     "training sequence contains"),
            b_prime=(f"filler prefix: ids = [BOS] + ['{FILLER_TEXT}'] * j + doc. The filler "
                     "token id is asserted to be a SINGLE token and recorded. Distance to BOS "
                     "and attended key count both move; content does not"),
            b=(f"real-text prefix: ids = [BOS] + tokens2d[other, :j] + doc, other != doc, "
               f"redrawn per (doc, j) with seed {SEED_OTHER} exactly as scripts/38 did (the "
               f"redrawn array is asserted equal to the npz's `others` at the registered "
               f"shape). Distance, key count and content all move"),
            shared=(f"the same {a.n_docs} documents, the same shifts {shifts}, the same read "
                    f"positions, the same forward rig; at j = 0 all three arms are the SAME "
                    f"token sequence with default RoPE, which is a free three-way rig check"),
        ),
        readout=dict(
            c=(f"c(i, j) = u_L . resid_post_L at doc-relative i in {I_LO}..{I_HI} (sequence "
               f"index 1 + i for arm (a'), 1 + j + i for the two prefix arms)"),
            c_perp=("c_perp = (u_L - (u_L . n_hat) n_hat) . resid, LITERAL (un-normalised) per "
                    "Addendum D. Its norm sqrt(1 - cos^2(u_L, n_hat)) is reported; since every "
                    "call is a CI-excludes-zero test, a positive rescaling cannot change any "
                    "call, so normalising or not is immaterial to the verdict"),
            n_hat=("unit mean residual direction, computed from a dedicated j = 0 forward of "
                   "these same documents at that layer, averaged over documents AND over the "
                   f"same read positions i in {I_LO}..{I_HI}. Addendum D does not pin the "
                   "positions; this choice (the positions c is read at) is declared here. "
                   "u_perp is invariant to the sign of n_hat"),
            diagnostics=("cos(u_L, n_hat) and the (doc, i)-level correlation of c and of c_perp "
                         "with ||resid||, both at j = 0 -- the quantity verify38 item (3) "
                         "reported as 0.986 at L9 and 0.75 at L6"),
        ),
        estimator=dict(
            beta=("OLS slope of the coordinate on j with (doc, i) fixed effects: the coordinate "
                  "and j are demeaned within each (doc, i) cell before the slope is taken. "
                  "Copied from scripts/38 fe_slope_shifts (L486); with a balanced shift grid "
                  "the estimator reduces to mean_d(num_d) / (n_i * sum_j jdm^2)"),
            ci=(f"cluster bootstrap over DOCUMENTS, {a.n_boot} draws, percentile "
                f"[{CI_LO}, {CI_HI}] (scripts/38 boot_ci, L532). One bootstrap index matrix, "
                f"seed {SEED_BOOT_DOCS}, is shared by every arm, layer, coordinate and segment "
                f"so the draws are comparable across them"),
            segments=("segment slopes for consecutive shift pairs (0->16, 16->32, 32->64, "
                      "64->128): the same fixed-effect estimator on the 2-shift sub-grid, which "
                      "reduces exactly to the mean within-(doc, i) difference divided by the "
                      "shift gap. Reported to show whether the pooled beta is a linear trend or "
                      "an artefact of one end of the range. DIAGNOSTIC, not verdict-bearing"),
        ),
        forward_checks=dict(
            bf16_j0=(f"arm (a') at j = 0 with explicit position_ids [0] + arange(1, 121) must "
                     f"reproduce the default forward EXACTLY (max abs resid diff "
                     f"<= {MAX_ABS_DIFF_DEFAULT}); the two position_ids vectors are literally "
                     f"equal, so anything but 0 means position_ids is not reaching RoPE"),
            bf16_shift=(f"arm (a') at j = {VERIFY_SHIFT} must differ from j = 0 in bf16 (max abs "
                        f"resid diff > {MIN_ABS_DIFF_SHIFT}). NECESSARY, NOT SUFFICIENT: "
                        f"scripts/38's refuted uniform-shift arm passed this same check with a "
                        f"max-abs difference of 96"),
            fp32_probe=(
                f"THE DISCRIMINATING CHECK, as verify38 ran it. A SEPARATE fp32 copy of the "
                f"model forwards {N_FP32_DOCS} documents three times: default position_ids "
                f"(= arm (a') at j = 0), arm (a') at j = {VERIFY_SHIFT}, and the RETIRED uniform "
                f"shift arange(121) + {VERIFY_SHIFT} at j = {VERIFY_SHIFT}. Statistic = relative "
                f"Frobenius change ||resid_j - resid_0||_F / ||resid_0||_F over the read "
                f"positions (the whole block is co-reported). Frozen rule: arm (a') is "
                f"NON-DEGENERATE iff rel > {FP32_REL_PASS}; it IS A SYMMETRY iff "
                f"rel < {FP32_REL_SYMMETRY}; anything between is INCONCLUSIVE. Both non-pass "
                f"outcomes STOP the script under prereg R2 (\"instrument not validated\") "
                f"because if (a') is a symmetry then ALL THREE of Addendum D's calls are void, "
                f"not merely DISTANCE-TO-BOS. The uniform-shift forward is a NEGATIVE CONTROL "
                f"on the probe: it is expected to land at ~1e-7 (verify38 measured 1.6e-7), and "
                f"a probe that cannot reproduce that cannot certify anything. All probe numbers "
                f"are written to the output file BEFORE any assertion so a failure is "
                f"diagnosable"),
        ),
        decision_rule=dict(
            order=("evaluated in Addendum D's listed order; the three conditions are mutually "
                   "exclusive as written (DISTANCE-TO-BOS requires arm (a')'s CI to exclude 0, "
                   "the other two require it to contain 0), and the full boolean vector is "
                   "reported alongside the call"),
            distance_to_bos="beta(a') CI excludes 0",
            content_accumulation=("beta(b) CI excludes 0 while beta(a') and beta(b') CIs "
                                  "contain 0"),
            key_count="beta(b') CI excludes 0 while beta(a') CI contains 0",
            otherwise="UNIDENTIFIED",
            layers="L6 primary, L9 co-reported",
            coordinates=("the call is made on c and, separately, on c_perp; both are reported. "
                         "primary_call = L6 on c"),
            not_read=("no beta_rel, no cache arm and no beta_abs/beta_rel ratio enters any call "
                      "here: Addendum D's rule reads the three arm CIs and nothing else"),
        ),
    )


# ===========================================================================
# helpers -- copied from scripts/38_position_causal.py, not imported
# ===========================================================================
def _unit(v):
    """scripts/38 L372."""
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else v


def _pearson(x, y):
    """scripts/38 L377."""
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    xd, yd = x - x.mean(), y - y.mean()
    den = float(np.linalg.norm(xd) * np.linalg.norm(yd))
    return float((xd @ yd) / den) if den > 0 else 0.0


def build_seq_batch(rows, pad_id):
    """Right-padded raw sequences + attention mask + true lengths. scripts/38 L454.

    Every cell here is single-length so no padding is actually produced, but the helper is
    kept byte-comparable with scripts/38 so a verifier diffs one function, not two.
    """
    lens = [len(r) for r in rows]
    lmax = max(lens)
    ids = torch.full((len(rows), lmax), int(pad_id), dtype=torch.long)
    attn = torch.zeros(len(rows), lmax, dtype=torch.long)
    for i, r in enumerate(rows):
        arr = np.asarray(r, dtype=np.int64)
        ids[i, :len(arr)] = torch.from_numpy(arr)
        attn[i, :len(arr)] = 1
    return ids, attn, torch.tensor(lens, dtype=torch.long)


def make_rows(arm, docs, donors, tokens2d, j, bos_id, filler_id, n_doc_tokens):
    """Token-id rows and the sequence indices holding doc-relative i in [I_LO, I_HI].

    scripts/38 make_rows (L473) extended from two arms to three. At j = 0 all three branches
    produce the identical sequence [BOS] + doc, which is the three-way rig check.
    """
    read = np.arange(I_LO, I_HI + 1, dtype=np.int64)
    doc_part = [tokens2d[d, :n_doc_tokens].astype(np.int64) for d in docs]
    if arm == ARM_A:
        rows = [np.concatenate([[bos_id], p]) for p in doc_part]
        return rows, 1 + read
    if arm == ARM_BP:
        fill = np.full(int(j), int(filler_id), dtype=np.int64)
        rows = [np.concatenate([[bos_id], fill, p]) for p in doc_part]
        return rows, 1 + j + read
    if arm == ARM_B:
        rows = [np.concatenate([[bos_id], tokens2d[o, :j].astype(np.int64), p])
                for p, o in zip(doc_part, donors)]
        return rows, 1 + j + read
    raise ValueError(f"unknown arm {arm}")


def bos_pinned_positions(seq_len, j, n_rows, device):
    """[n_rows, seq_len] position_ids = [0] + (arange(1, seq_len) + j) -- arm (a').

    The one line that separates this script from the arm verify38 refuted: index 0 (BOS) is
    PINNED, so the position DIFFERENCES the model attends over actually change with j.
    """
    pos = torch.arange(seq_len, dtype=torch.long, device=device)
    pos = torch.cat([pos[:1], pos[1:] + int(j)])
    return pos.unsqueeze(0).repeat(int(n_rows), 1)


def uniform_positions(seq_len, j, n_rows, device):
    """[n_rows, seq_len] position_ids = arange(seq_len) + j -- the RETIRED scripts/38 arm (a).

    Kept only as the fp32 probe's negative control: it must come back at round-off.
    """
    pos = torch.arange(seq_len, dtype=torch.long, device=device) + int(j)
    return pos.unsqueeze(0).repeat(int(n_rows), 1)


def fe_slope_shifts(C, shifts):
    """beta with (doc, i) fixed effects, plus the per-document bootstrap ingredients.

    scripts/38 L486, unchanged. ``C`` is [n_shifts, n_docs, n_i]. Demeaning c within each
    (doc, i) cell and demeaning the shift grid (identical in every cell) reduces the
    fixed-effect slope to ``sum_d num_d / (n_docs * n_i * sum_j jdm^2)``, so a cluster
    bootstrap over documents is a resampled mean of ``num_d``.
    """
    C = np.asarray(C, dtype=np.float64)
    j = np.asarray(shifts, dtype=np.float64)
    jdm = j - j.mean()
    cdm = C - C.mean(axis=0, keepdims=True)
    num = np.einsum("jdi,j->d", cdm, jdm)                  # [n_docs]
    sjj = float((jdm ** 2).sum())
    den_per_doc = C.shape[2] * sjj
    if sjj == 0 or den_per_doc == 0:
        return float("nan"), num, 0.0
    return float(num.mean() / den_per_doc), num, den_per_doc


def boot_ci(v):
    """scripts/38 L532, unchanged."""
    v = np.asarray(v, dtype=np.float64)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return dict(lo=None, hi=None, mean=None, n=0, excludes_zero=False, sign=0)
    lo, hi = (float(x) for x in np.percentile(v, [CI_LO, CI_HI]))
    ex = bool(lo > 0 or hi < 0)
    return dict(lo=lo, hi=hi, mean=float(v.mean()), n=int(v.size),
                excludes_zero=ex, sign=int(np.sign(v.mean())) if ex else 0)


def slope_with_ci(C3, shifts, bd):
    """(beta, CI) for a [n_shifts, n_docs, n_i] block under the shared document draws."""
    slope, num, den = fe_slope_shifts(C3, shifts)
    boot = num[bd].mean(1) / den if den else np.full(bd.shape[0], np.nan)
    return slope, boot_ci(boot), boot


def addendum_d_call(ci_a, ci_bp, ci_b):
    """Addendum D's frozen three-way rule, evaluated in its listed order."""
    a_ex = bool(ci_a["excludes_zero"])
    bp_ex = bool(ci_bp["excludes_zero"])
    b_ex = bool(ci_b["excludes_zero"])
    if a_ex:
        call, why = "DISTANCE-TO-BOS CODE", "beta(a') CI excludes 0"
    elif b_ex and not bp_ex:
        call, why = ("CONTENT-ACCUMULATION",
                     "beta(b) CI excludes 0 while beta(a') and beta(b') CIs contain 0")
    elif bp_ex:
        call, why = ("KEY-COUNT",
                     "beta(b') CI excludes 0 while beta(a') CI contains 0")
    else:
        call, why = ("UNIDENTIFIED",
                     "no arm's CI excludes 0, so none of the three conditions holds")
    return dict(call=call, reason=why,
                a_prime_excludes_zero=a_ex, b_prime_excludes_zero=bp_ex,
                b_excludes_zero=b_ex,
                signs=dict(a_prime=int(ci_a["sign"]), b_prime=int(ci_bp["sign"]),
                           b=int(ci_b["sign"])))


# ===========================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="cache/sparse_big")
    ap.add_argument("--npz", default=DATA_NPZ)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--data-out", default=DEFAULT_DATA_OUT)
    ap.add_argument("--layers", type=int, nargs="+", default=LAYERS)
    ap.add_argument("--shifts", type=int, nargs="+", default=SHIFTS)
    ap.add_argument("--n-docs", type=int, default=N_DOCS)
    ap.add_argument("--doc-tokens", type=int, default=DOC_TOKENS)
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    ap.add_argument("--batch-size", type=int, default=rig.BATCH_SIZE)
    ap.add_argument("--n-fp32", type=int, default=N_FP32_DOCS)
    ap.add_argument("--fp32-device", default="cpu",
                    help="device for the fp32 symmetry probe; cpu keeps MPS free")
    ap.add_argument("--no-fp32-check", action="store_true",
                    help="skip the fp32 symmetry probe (the run is then NOT validated)")
    ap.add_argument("--smoke", action="store_true",
                    help="annotate the output as a scaled-down instrument check")
    a = ap.parse_args()
    t_start = time.time()
    layers = sorted(set(int(x) for x in a.layers))
    shifts = sorted(set(int(x) for x in a.shifts))
    smoke = bool(a.smoke or a.n_docs != N_DOCS or a.n_boot != N_BOOT or shifts != SHIFTS
                 or layers != LAYERS or a.doc_tokens != DOC_TOKENS or a.no_fp32_check
                 or a.n_fp32 != N_FP32_DOCS)

    # A smoke run must never be able to overwrite the registered result or its data file.
    if smoke and str(a.out) == DEFAULT_OUT:
        raise SystemExit(f"refusing to write a SMOKE run to the registered path {DEFAULT_OUT}; "
                         f"pass --out <scratch path>")
    data_out = a.data_out
    if smoke and str(data_out) == DEFAULT_DATA_OUT:
        data_out = str(Path(a.out).with_name(Path(a.out).stem + "_data.npz"))

    # ---- R1: declaration stub to disk BEFORE anything is loaded or forwarded ----
    declared = dict(
        script="scripts/38b_position_arms_repaired.py",
        prereg="docs/phase3_prereg_addenda_2026-08-21.md Addendum D",
        prereg_repaired="docs/phase3_prereg_2026-08-21.md section 3.2 (arms (a) and (b))",
        global_rules="docs/phase3_prereg_2026-08-21.md sections 0 and 1 (R1, R2, R5)",
        supersedes_arms=("scripts/38_position_causal.py arm (a) [position_ids_offset] and its "
                         "arm (b) reading; scripts/38's cache arm, beta_rel, ratio and "
                         "three-way ABSOLUTE / DOCUMENT-RELATIVE call are NOT re-run and are "
                         "not read here"),
        status="DECLARATION STUB -- rules frozen, no results yet",
        declared_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
        smoke=smoke,
        question=("scripts/38 asked whether the position coordinate is absolute or "
                  "document-relative and could not answer, because the arm meant to move "
                  "absolute position was an exact symmetry of the model. Addendum D asks the "
                  "answerable question instead: of the three things a prefix changes -- "
                  "distance to BOS, attended key count, and accumulated content -- which one "
                  "does the coordinate track? Three nested arms move one more of them each."),
        config=dict(root=a.root, npz=a.npz, layers=layers, shifts=shifts,
                    n_docs=int(a.n_docs), doc_tokens=int(a.doc_tokens),
                    n_boot=int(a.n_boot), batch_size=int(a.batch_size),
                    n_fp32=int(a.n_fp32), fp32_device=str(a.fp32_device),
                    fp32_check=not bool(a.no_fp32_check),
                    device=DEVICE, data_out=str(data_out),
                    hf_home=os.environ.get("HF_HOME"),
                    hf_hub_offline=os.environ.get("HF_HUB_OFFLINE")),
        declared_rules=declared_rules(a, layers, shifts),
    )
    outp = Path(a.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(declared, indent=2))
    print(f"declaration stub -> {outp}", flush=True)
    if smoke:
        print("SMOKE RUN -- scaled-down instrument check, not the registered measurement",
              flush=True)

    # =======================================================================
    # inputs: u_L and the documents come from scripts/38's npz, not recomputed
    # =======================================================================
    z = np.load(a.npz)
    keys_used = []
    if "docs" not in z.files:
        raise RuntimeError(f"{a.npz} has no 'docs' key; found {sorted(z.files)}")
    docs_full = np.asarray(z["docs"], dtype=np.int64)
    keys_used.append("docs")
    if docs_full.shape != (N_DOCS,):
        raise RuntimeError(f"npz 'docs' has shape {docs_full.shape}, expected ({N_DOCS},)")
    u_L = {}
    for L in layers:
        k = f"u_L_layer{L}"
        if k not in z.files:
            raise RuntimeError(f"{a.npz} has no '{k}'; found {sorted(z.files)}")
        v = np.asarray(z[k], dtype=np.float64)
        if v.shape != (D_MODEL,):
            raise RuntimeError(f"npz '{k}' has shape {v.shape}, expected ({D_MODEL},)")
        u_L[L] = v
        keys_used.append(k)
    u_L_norms = {str(L): float(np.linalg.norm(u_L[L])) for L in layers}
    for L in layers:
        if abs(u_L_norms[str(L)] - 1.0) > 1e-4:
            raise RuntimeError(f"u_L_layer{L} is not a unit vector: ||u_L|| = {u_L_norms[str(L)]}")

    docs = docs_full[:int(a.n_docs)]
    tokens2d = rig.load_tokens2d(a.root)

    # donors: scripts/38's draw, re-derived and (at the registered shape) asserted equal
    rng_other = np.random.default_rng(SEED_OTHER)
    others_full = rng_other.integers(0, rig.N_DOCS, size=(N_DOCS, len(SHIFTS)))
    clash = others_full == docs_full[:, None]
    others_full[clash] = (others_full[clash] + 1) % rig.N_DOCS
    donor_provenance = "redrawn from scripts/38's seed 1 and its 300 document ids"
    if "others" in z.files:
        keys_used.append("others (provenance assert)")
        donors_asserted = bool(np.array_equal(others_full, np.asarray(z["others"], np.int64)))
        if not donors_asserted:
            raise RuntimeError("the seed-1 donor redraw does not reproduce the npz's 'others'")
        donor_provenance += "; asserted byte-equal to the npz's 'others'"
    else:
        donors_asserted = None
    if all(j in SHIFTS for j in shifts):
        cols = [SHIFTS.index(j) for j in shifts]
        others = others_full[:int(a.n_docs)][:, cols]
        donor_note = f"columns {cols} of scripts/38's (300, 5) donor array"
    else:
        rng_o2 = np.random.default_rng(SEED_OTHER)
        others = rng_o2.integers(0, rig.N_DOCS, size=(len(docs), len(shifts)))
        cl2 = others == docs[:, None]
        others[cl2] = (others[cl2] + 1) % rig.N_DOCS
        donor_note = "off-grid shifts: donors redrawn at this run's shape (SMOKE ONLY)"
    if bool((others == docs[:, None]).any()):
        raise RuntimeError("a donor document equals its own document after the clash fix")

    inputs = dict(npz=str(a.npz), keys_used=keys_used, u_L_norms=u_L_norms,
                  docs_shape=list(docs_full.shape), docs_used=int(len(docs)),
                  first_docs=[int(x) for x in docs[:8]],
                  donors=dict(provenance=donor_provenance, note=donor_note,
                              asserted_equal_to_npz=donors_asserted,
                              shape=list(others.shape)))
    print(f"inputs: {a.npz} keys {keys_used}; {len(docs)} docs; "
          f"u_L norms { {k: round(v, 6) for k, v in u_L_norms.items()} }", flush=True)

    # =======================================================================
    # fp32 symmetry probe -- run FIRST, on its own model, before MPS is touched
    # =======================================================================
    hf = os.environ.get("HF_TOKEN")
    fp32 = dict(ran=False, note="skipped with --no-fp32-check; the run is NOT validated")
    read_rel = np.arange(I_LO, I_HI + 1, dtype=np.int64)
    if not a.no_fp32_check:
        n32 = min(int(a.n_fp32), len(docs))
        m32, tokzr32 = load_model(token=hf, device=str(a.fp32_device), dtype=torch.float32)
        bos32 = tokzr32.bos_token_id
        pad32 = tokzr32.pad_token_id if tokzr32.pad_token_id is not None else tokzr32.eos_token_id
        rows32, _ = make_rows(ARM_A, docs[:n32], others[:n32, 0], tokens2d, 0, bos32, bos32,
                              int(a.doc_tokens))
        ids32, attn32, _ = build_seq_batch(rows32, pad32)
        ids32, attn32 = ids32.to(a.fp32_device), attn32.to(a.fp32_device)
        T32 = int(ids32.shape[1])
        ridx32 = torch.from_numpy(1 + read_rel).to(a.fp32_device)
        t32 = time.time()

        def _f32(pos):
            r = rig.forward_capture(m32, ids32, attn32, layers, position_ids=pos)
            return {L: r[L].float().cpu() for L in layers}

        base = _f32(None)
        pinned = _f32(bos_pinned_positions(T32, VERIFY_SHIFT, n32, a.fp32_device))
        uniform = _f32(uniform_positions(T32, VERIFY_SHIFT, n32, a.fp32_device))

        def _rel(new, ref, restrict):
            out = {}
            for L in layers:
                x, y = new[L], ref[L]
                if restrict:
                    x = x.index_select(1, ridx32.cpu())
                    y = y.index_select(1, ridx32.cpu())
                out[str(L)] = dict(
                    rel_frobenius=float(torch.linalg.norm(x - y) / torch.linalg.norm(y)),
                    max_abs_diff=float((x - y).abs().max()),
                    ref_frobenius=float(torch.linalg.norm(y)))
            return out

        pin_read = _rel(pinned, base, True)
        uni_read = _rel(uniform, base, True)
        rel_pin = max(pin_read[str(L)]["rel_frobenius"] for L in layers)
        rel_uni = max(uni_read[str(L)]["rel_frobenius"] for L in layers)
        verdict = ("NON-DEGENERATE" if rel_pin > FP32_REL_PASS else
                   "SYMMETRY" if rel_pin < FP32_REL_SYMMETRY else "INCONCLUSIVE")
        fp32 = dict(
            ran=True, device=str(a.fp32_device), dtype="float32", n_docs=int(n32),
            seq_len=T32, verify_shift=VERIFY_SHIFT, seconds=float(time.time() - t32),
            a_prime_bos_pinned=dict(read_positions=pin_read,
                                    all_positions=_rel(pinned, base, False)),
            retired_uniform_shift=dict(read_positions=uni_read,
                                       all_positions=_rel(uniform, base, False)),
            rel_a_prime=rel_pin, rel_uniform_control=rel_uni,
            ratio_a_prime_over_uniform=(float(rel_pin / rel_uni) if rel_uni > 0 else None),
            thresholds=dict(pass_above=FP32_REL_PASS, symmetry_below=FP32_REL_SYMMETRY),
            verdict=verdict,
            negative_control_ok=bool(rel_uni < FP32_REL_PASS),
            verify38_reference=1.6e-7,
            note=("relative Frobenius change of resid_post at the read positions, j=16 vs j=0, "
                  "in fp32. The retired uniform shift is the negative control and is expected "
                  "at ~1e-7 (verify38 measured 1.6e-7); if arm (a') lands there too it is the "
                  "same symmetry and Addendum D's calls are all void"))
        del m32, base, pinned, uniform, ids32, attn32
        gc.collect()
        print(f"fp32 probe ({a.fp32_device}, {n32} docs): a' rel {rel_pin:.3e}  "
              f"uniform-control rel {rel_uni:.3e}  -> {verdict} "
              f"({fp32['seconds']:.0f}s)", flush=True)
    outp.write_text(json.dumps(dict(declared, status="INPUTS + FP32 PROBE ONLY",
                                    inputs=inputs, fp32_symmetry_probe=fp32),
                               indent=2, default=float))
    if fp32["ran"]:
        assert fp32["verdict"] == "NON-DEGENERATE", (
            f"prereg R2, instrument not validated: arm (a')'s fp32 relative residual change is "
            f"{fp32['rel_a_prime']:.3e} ({fp32['verdict']}). If the BOS-pinned shift is a "
            f"symmetry of the model then all three Addendum D calls are void, not just "
            f"DISTANCE-TO-BOS. Probe numbers are already written to {outp}.")

    # =======================================================================
    # model, tokenizer, filler token
    # =======================================================================
    model, tokzr = load_model(token=hf)
    bos = tokzr.bos_token_id
    pad = tokzr.pad_token_id if tokzr.pad_token_id is not None else tokzr.eos_token_id
    fill_ids = tokzr(FILLER_TEXT, add_special_tokens=False)["input_ids"]
    if len(fill_ids) != 1:
        raise RuntimeError(f"filler {FILLER_TEXT!r} is {len(fill_ids)} tokens ({fill_ids}), "
                           f"not 1; Addendum D's arm (b') requires a single token")
    filler_id = int(fill_ids[0])
    filler = dict(text=FILLER_TEXT, token_id=filler_id,
                  decoded=tokzr.decode([filler_id]),
                  single_token_asserted=True,
                  note="add_special_tokens=False; asserted to be exactly one token")
    print(f"\nmodel on {DEVICE}; bos={bos} pad={pad}; filler {FILLER_TEXT!r} = id {filler_id}; "
          f"{len(docs)} docs x {len(shifts)} shifts x {len(ARMS)} arms "
          f"= {len(docs) * len(shifts) * len(ARMS)} sequences", flush=True)

    # =======================================================================
    # bf16 forward checks -- written to disk BEFORE they are asserted
    # =======================================================================
    probe = docs[:min(8, len(docs))]
    prows, _ = make_rows(ARM_A, probe, others[:len(probe), 0], tokens2d, 0, bos, filler_id,
                         int(a.doc_tokens))
    pid_ids, pid_attn, _ = build_seq_batch(prows, pad)
    pid_ids, pid_attn = pid_ids.to(DEVICE), pid_attn.to(DEVICE)
    seq_len = int(pid_ids.shape[1])

    def _probe(j):
        pos = bos_pinned_positions(seq_len, j, pid_ids.shape[0], DEVICE)
        r = rig.forward_capture(model, pid_ids, pid_attn, layers, position_ids=pos)
        return {L: r[L].float().cpu() for L in layers}

    r_def = rig.forward_capture(model, pid_ids, pid_attn, layers, position_ids=None)
    r_def = {L: r_def[L].float().cpu() for L in layers}
    r_j0, r_js = _probe(0), _probe(VERIFY_SHIFT)
    d_default = max(float((r_j0[L] - r_def[L]).abs().max()) for L in layers)
    d_shift = max(float((r_js[L] - r_j0[L]).abs().max()) for L in layers)
    rel_shift_bf16 = {str(L): float(torch.linalg.norm(r_js[L] - r_j0[L])
                                    / torch.linalg.norm(r_j0[L])) for L in layers}
    checks = dict(
        j0_vs_default_max_abs_diff=d_default, j0_vs_default_threshold=MAX_ABS_DIFF_DEFAULT,
        j0_reproduces_default=bool(d_default <= MAX_ABS_DIFF_DEFAULT),
        j16_vs_j0_max_abs_diff=d_shift, j16_vs_j0_threshold=MIN_ABS_DIFF_SHIFT,
        j16_vs_j0_rel_frobenius_bf16=rel_shift_bf16,
        position_ids_changes_output=bool(d_shift > MIN_ABS_DIFF_SHIFT),
        n_probe_rows=int(pid_ids.shape[0]), seq_len=seq_len, verify_shift=VERIFY_SHIFT,
        note=("arm (a') at j=0 has position_ids [0] + arange(1, 121), which IS the default "
              "arange, so the two forwards must agree exactly; j=16 must differ. THE bf16 "
              "DIFFERENCE IS NECESSARY BUT NOT SUFFICIENT -- scripts/38's refuted uniform-shift "
              "arm passed this same check with 96 -- so the fp32 probe above is what "
              "certifies the arm"))
    del r_def, r_j0, r_js
    outp.write_text(json.dumps(dict(declared, status="CHECKS ONLY", inputs=inputs,
                                    fp32_symmetry_probe=fp32, filler=filler,
                                    forward_checks=checks), indent=2, default=float))
    print(f"bf16 checks: j0 vs default {d_default:.3e} (<= {MAX_ABS_DIFF_DEFAULT}), "
          f"j16 vs j0 {d_shift:.3e} (> {MIN_ABS_DIFF_SHIFT})", flush=True)
    assert checks["j0_reproduces_default"], (
        f"arm (a') j=0 does not reproduce the default forward: {d_default}")
    assert checks["position_ids_changes_output"], (
        f"BOS-pinned position_ids at j={VERIFY_SHIFT} did not change the output: {d_shift}")

    # =======================================================================
    # n_hat pass: the j=0 forwards of these documents, at the read positions
    # =======================================================================
    n_i = I_HI - I_LO + 1
    rows0, read0 = make_rows(ARM_A, docs, others[:, 0], tokens2d, 0, bos, filler_id,
                             int(a.doc_tokens))
    ridx0 = torch.from_numpy(read0).to(DEVICE)
    H0 = {L: np.zeros((len(docs), n_i, D_MODEL), dtype=np.float32) for L in layers}
    fwd = dict(batches=0, seconds=0.0, tokens=0, sequences=0)
    t_n = time.time()
    for lo, chunk in rig.iter_batches(rows0, a.batch_size):
        ids, attn, _ = build_seq_batch(chunk, pad)
        ids, attn = ids.to(DEVICE), attn.to(DEVICE)
        t0 = time.time()
        resid = rig.forward_capture(model, ids, attn, layers, position_ids=None)
        for L in layers:
            H0[L][lo:lo + len(chunk)] = resid[L].index_select(1, ridx0).float().cpu().numpy()
        del resid
        fwd["batches"] += 1
        fwd["seconds"] += time.time() - t0
        fwd["tokens"] += int(attn.sum())
        fwd["sequences"] += len(chunk)
    n_hat, u_perp, norm_geom = {}, {}, {}
    for L in layers:
        flat = H0[L].reshape(-1, D_MODEL).astype(np.float64)
        mean_resid = flat.mean(0)
        nh = _unit(mean_resid)
        cos = float(u_L[L] @ nh)
        up = u_L[L] - cos * nh                              # LITERAL, per Addendum D
        n_hat[L], u_perp[L] = nh, up
        rn = np.linalg.norm(flat, axis=1)
        c0 = flat @ u_L[L]
        cp0 = flat @ up
        norm_geom[str(L)] = dict(
            cos_uL_nhat=cos, angle_deg=float(np.degrees(np.arccos(np.clip(abs(cos), -1, 1)))),
            u_perp_norm=float(np.linalg.norm(up)),
            u_perp_norm_expected=float(np.sqrt(max(1.0 - cos * cos, 0.0))),
            mean_resid_norm=float(np.linalg.norm(mean_resid)),
            median_row_resid_norm=float(np.median(rn)),
            corr_c_with_resid_norm=_pearson(c0, rn),
            corr_c_perp_with_resid_norm=_pearson(cp0, rn),
            n_rows=int(flat.shape[0]),
            note=("n_hat is the unit mean residual over the 300 documents x 100 read positions "
                  "at j=0. verify38 item (3) reported corr(c, ||resid||) = 0.986 at L9 and 0.75 "
                  "at L6; corr_c_perp_with_resid_norm is the same quantity after "
                  "orthogonalisation and is what makes c_perp a separate read-out"))
        del flat, rn, c0, cp0
        print(f"  L{L}: cos(u_L, n_hat) {cos:+.4f}  ||u_perp|| {np.linalg.norm(up):.4f}  "
              f"corr(c, ||resid||) {norm_geom[str(L)]['corr_c_with_resid_norm']:+.4f}  -> "
              f"corr(c_perp, ||resid||) "
              f"{norm_geom[str(L)]['corr_c_perp_with_resid_norm']:+.4f}", flush=True)
    del H0
    gc.collect()
    print(f"n_hat pass: {len(docs)} sequences ({time.time() - t_n:.1f}s)", flush=True)

    # =======================================================================
    # the 3 x n_shifts forward sweep
    # =======================================================================
    U = {L: torch.from_numpy(np.stack([u_L[L], u_perp[L]], axis=1).astype(np.float32)).to(DEVICE)
         for L in layers}
    # C[arm][layer] -> [n_shifts, n_docs, n_i, 2]; last axis = (c, c_perp)
    C = {arm: {L: np.zeros((len(shifts), len(docs), n_i, 2), dtype=np.float32)
               for L in layers} for arm in ARMS}
    cell_lengths = {}
    for arm in ARMS:
        for jk, j in enumerate(shifts):
            rows, read_idx = make_rows(arm, docs, others[:, jk], tokens2d, j, bos, filler_id,
                                       int(a.doc_tokens))
            ridx = torch.from_numpy(read_idx).to(DEVICE)
            cell_lengths[f"{arm}_j{j}"] = int(len(rows[0]))
            t_cell = time.time()
            for lo, chunk in rig.iter_batches(rows, a.batch_size):
                ids, attn, _ = build_seq_batch(chunk, pad)
                ids, attn = ids.to(DEVICE), attn.to(DEVICE)
                pos = (bos_pinned_positions(ids.shape[1], j, ids.shape[0], DEVICE)
                       if arm == ARM_A else None)
                t0 = time.time()
                resid = rig.forward_capture(model, ids, attn, layers, position_ids=pos)
                for L in layers:
                    h = resid[L].index_select(1, ridx).float()      # [b, n_i, 1152]
                    C[arm][L][jk, lo:lo + len(chunk)] = (h @ U[L]).cpu().numpy()
                    del h
                del resid
                fwd["batches"] += 1
                fwd["seconds"] += time.time() - t0
                fwd["tokens"] += int(attn.sum())
                fwd["sequences"] += len(chunk)
            print(f"  {arm:<26s} j={j:<4d} len={len(rows[0]):<4d} "
                  f"{time.time() - t_cell:5.1f}s", flush=True)

    # =======================================================================
    # beta, segment slopes, calls
    # =======================================================================
    rng_bd = np.random.default_rng(SEED_BOOT_DOCS)
    bd = rng_bd.integers(0, len(docs), size=(int(a.n_boot), len(docs)))
    segments = [(shifts[k], shifts[k + 1]) for k in range(len(shifts) - 1)]

    arms_out, calls = {}, {}
    for L in layers:
        per_arm = {}
        for arm in ARMS:
            per_coord = {}
            for ci, cname in enumerate(COORDS):
                block = C[arm][L][:, :, :, ci]
                slope, ci_d, _ = slope_with_ci(block, shifts, bd)
                segs = {}
                for (j1, j2) in segments:
                    k1, k2 = shifts.index(j1), shifts.index(j2)
                    s_sl, s_ci, _ = slope_with_ci(block[[k1, k2]], [j1, j2], bd)
                    segs[f"{j1}->{j2}"] = dict(beta=s_sl, ci=s_ci, delta_j=int(j2 - j1),
                                               mean_diff=float(block[k2].mean()
                                                               - block[k1].mean()))
                per_coord[cname] = dict(
                    beta=slope, ci=ci_d,
                    per_j_mean={str(j): float(block[k].mean()) for k, j in enumerate(shifts)},
                    per_j_sd_over_docs={str(j): float(block[k].mean(axis=1).std(ddof=1))
                                        if block.shape[1] > 1 else None
                                        for k, j in enumerate(shifts)},
                    segment_slopes=segs,
                    segment_note=("consecutive-pair fixed-effect slopes; a linear response has "
                                  "them all equal within CI. Diagnostic, not verdict-bearing"))
            per_arm[arm] = per_coord
        arms_out[str(L)] = per_arm

        calls[str(L)] = {}
        for cname in COORDS:
            call = addendum_d_call(per_arm[ARM_A][cname]["ci"], per_arm[ARM_BP][cname]["ci"],
                                   per_arm[ARM_B][cname]["ci"])
            call["layer"] = int(L)
            call["coordinate"] = cname
            call["primary"] = bool(L == layers[0] and cname == "c")
            call["betas"] = {arm: per_arm[arm][cname]["beta"] for arm in ARMS}
            calls[str(L)][cname] = call
            print(f"  L{L} [{cname}]: {call['call']} -- {call['reason']}  "
                  f"(betas a' {call['betas'][ARM_A]:+.4g}, b' {call['betas'][ARM_BP]:+.4g}, "
                  f"b {call['betas'][ARM_B]:+.4g})", flush=True)

    # =======================================================================
    # rig consistency: at j = 0 the three arms are the SAME input
    # =======================================================================
    consistency = {}
    if 0 in shifts:
        k0 = shifts.index(0)
        for L in layers:
            pair = {}
            for x in range(len(ARMS)):
                for y in range(x + 1, len(ARMS)):
                    d = float(np.abs(C[ARMS[x]][L][k0] - C[ARMS[y]][L][k0]).max())
                    pair[f"{ARMS[x]}__vs__{ARMS[y]}"] = d
            consistency[str(L)] = dict(
                j0_three_way_max_abs_diff=pair,
                j0_identical=bool(max(pair.values()) == 0.0),
                note=("at j = 0 all three arms are [BOS] + doc with default RoPE, so every "
                      "pairwise difference must be exactly 0; a nonzero value means an arm's "
                      "construction leaks at j = 0"))
    else:
        consistency = dict(note="no j = 0 cell in this run's shift grid")

    # =======================================================================
    # runtime
    # =======================================================================
    tok_per_s = fwd["tokens"] / max(fwd["seconds"], 1e-9)
    full_tokens = (N_DOCS * (DOC_TOKENS + 1)                                  # n_hat pass
                   + N_DOCS * len(SHIFTS) * (DOC_TOKENS + 1)                  # arm (a')
                   + 2 * sum(N_DOCS * (DOC_TOKENS + 1 + j) for j in SHIFTS))  # (b') and (b)
    runtime = dict(
        forward=dict(batches=fwd["batches"], sequences=fwd["sequences"], tokens=fwd["tokens"],
                     seconds=fwd["seconds"], tokens_per_sec=float(tok_per_s),
                     sec_per_batch=float(fwd["seconds"] / max(fwd["batches"], 1)),
                     ms_per_token=float(1000.0 / max(tok_per_s, 1e-9))),
        full_run_forward_estimate=dict(
            sequences=int(N_DOCS * (1 + len(ARMS) * len(SHIFTS))), tokens=int(full_tokens),
            seconds=float(full_tokens / max(tok_per_s, 1e-9)),
            minutes=float(full_tokens / max(tok_per_s, 1e-9) / 60.0),
            note=("this run's measured throughput applied to the registered "
                  "300 x 5 x 3 grid plus the 300-sequence n_hat pass; a smoke measured while "
                  "another job holds the device understates throughput")),
        total_seconds=float(time.time() - t_start))
    print(f"\nforward: {fwd['seconds']:.1f}s over {fwd['batches']} batches "
          f"({tok_per_s:,.0f} tok/s); full-grid estimate "
          f"{full_tokens / max(tok_per_s, 1e-9) / 60:.1f} min", flush=True)

    # =======================================================================
    # write
    # =======================================================================
    npz = dict(docs=docs.astype(np.int64), others=others.astype(np.int64),
               shifts=np.asarray(shifts, dtype=np.int64),
               layers=np.asarray(layers, dtype=np.int64),
               doc_relative_i=np.arange(I_LO, I_HI + 1, dtype=np.int64),
               arms=np.asarray(ARMS), coordinates=np.asarray(COORDS),
               filler_token_id=np.asarray([filler_id], dtype=np.int64))
    for L in layers:
        npz[f"u_L_layer{L}"] = u_L[L].astype(np.float32)
        npz[f"n_hat_layer{L}"] = n_hat[L].astype(np.float32)
        npz[f"u_perp_layer{L}"] = u_perp[L].astype(np.float32)
        for arm in ARMS:
            npz[f"c_{arm}_layer{L}"] = C[arm][L]        # [n_shifts, n_docs, n_i, (c, c_perp)]
    dp = Path(data_out)
    dp.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(dp, **npz)

    primary = calls[str(layers[0])]["c"] if str(layers[0]) in calls else None
    out = dict(declared)
    out.update(
        status="COMPLETE",
        finished_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
        inputs=inputs,
        fp32_symmetry_probe=fp32,
        filler=filler,
        forward_checks=checks,
        norm_geometry=norm_geom,
        arms=arms_out,
        cell_sequence_lengths=cell_lengths,
        calls=calls,
        primary_call=(primary["call"] if primary else None),
        primary_call_layer=int(layers[0]),
        primary_call_coordinate="c",
        consistency=consistency,
        runtime=runtime,
        runtime_seconds=float(time.time() - t_start),
        data_npz=str(dp),
    )
    outp.write_text(json.dumps(out, indent=2, default=float))
    print(f"\nprimary (L{layers[0]}, c) call: {out['primary_call']}", flush=True)
    print(f"data -> {dp}", flush=True)
    print(f"wrote {outp} ({time.time() - t_start:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
