"""A3 -- the steering probe, at MATCHED intervention norms.  [REVISION 2]

Goodfire's product claim is that interpolating along a discovered manifold steers a
model better than a naive contrastive vector. The claim is untestable unless the two
interventions are the same SIZE: a bigger push moves the model more, and "the manifold
wins" is trivially arrangeable by giving the manifold arm a longer vector. The plan
(docs/plan.md:303-304) therefore pre-registers matched intervention norms, and this
script is that experiment on the one manifold this project has already established:
the SUPERVISED month ring in the raw residual stream at layer 9 (rho=1.000, p=5.0e-5,
gap_ratio=1.556, results/months_power_big_L9.json).

REVISION 2 -- WHAT CHANGED AND WHY. Revision 1 of this script produced a correct
verdict (manifold_loses) but its supporting apparatus could not sustain three of the
claims that were read off it. An adversarial review found six specific defects; all six
were reproduced from revision 1's own raw rows before anything was changed. The
pre-registered core -- primary arm, baseline arm, EPS_HEADLINE, the qualification rule
and the verdict rule -- is UNCHANGED, so the headline number must and does reproduce.
Everything added is secondary and is listed in declared["revision"].

  R1 traversal is now computed for EVERY arm, not only manifold_arc. Revision 1 stored
     the dose-response walk for the manifold arm alone, which made "no control does
     this" unfalsifiable from the output. It is false: contrastive_inplane walks the
     calendar almost identically. What IS specific to the arc is stated below.
  R2 arc CLAMPING is surfaced per cell (frac_unclamped, arc_t_med). Where the walk to B
     is shorter than the matched norm the arm degenerates to a rescaled chord and is not
     a walk at all; revision 1 recorded the flags but no summary, and the two pairs most
     often quoted as evidence of walking (Nov->Dec, Aug->Sep) are clamped at every grid
     point. The positive is now cited only on cells where the walk is genuine.
  R3 the COST AXIS. The pre-registered cost is in-context NLL, which turns out to be
     near-degenerate (every arm within a few hundredths of a nat, and the null direction
     "improving" it most), so the verdict rule collapsed to its rotation clause and the
     pre-registered per-unit-cost question came back n_usable=0. The spec's own metric --
     perplexity of the steered generation under the unsteered model -- is now a
     first-class per-instance readout with bootstrap CIs, on all nine pairs, and the
     verdict is reported under ALL THREE cost metrics unconditionally.
  R4 the primary arm is now tested AT AND BELOW its own natural chord length. Revision
     1's smallest grid point already overshot B by 1.13-1.61x on the four adjacent pairs.
     Two smaller eps points and a new CHORD-MATCHED norm row (every arm rescaled to the
     manifold's own per-prompt chord to B, so overshoot is 1.00 by construction) close
     that gap.
  R5 the random control's three "seeds" were three fixed direction matrices reused
     across all nine pairs. The seed base is now per pair: 27 independent directions.
  R6 a SECOND INTERVENTION SITE. Revision 1 intervened only at the source-month token,
     where mean_B - mean_A carries B's token identity and the contrastive baseline is
     close to an oracle substitution. The headline comparison is now repeated at the
     answer position, where no month token sits, so the site-dependence of the result is
     measured rather than caveated.

WHY LAYER 9 AND WHY THE SUPERVISED RING. Latent-subset discovery for months is
exhausted (results/month_cofire.json, results/window_passer_audit.json), and the
ordering tier accepts random subsets at ~0.16-0.28 (results/months_power_big_L9.json,
fpr block) so ordering alone is never evidence. The closure tier does discriminate,
and it is layer 9's raw residual stream where the month ring closes. This probe asks a
different question of that same object: is the ring a CAUSAL surface, or a readable
shadow?

WHAT IS MEASURED. Eight intervention arms, all applied to the residual stream at layer
9 at a single token position, all rescaled to an identical per-prompt L2 norm:

  manifold_chord  PRIMARY. Replace the in-plane component of h by the ring point at
                  B's angle: delta = (q_B - p) @ V. This is the plan's literal
                  wording. Declared primary BEFORE results.
  manifold_arc    SECONDARY, declared. Walk along the ring from A's angle toward B's
                  angle, stopping where the displacement reaches the matched norm.
  contrastive     The naive baseline: delta = mean_B - mean_A.
  contrastive_inplane / contrastive_orth
                  SECONDARY, declared. The contrastive vector split at the ring plane
                  and each half rescaled to the same matched norm. This is the
                  mechanistic control: it says WHERE the contrastive vector's power
                  lives. Only 46% of the between-month variance is in the ring plane
                  (top-2 PC fraction 0.460), so "the manifold loses" is uninterpretable
                  without knowing whether the other 54% is doing the work.
  random_orth_s{0,1,2}
                  Control: a random direction in the ORTHOGONAL COMPLEMENT of the ring
                  plane, reseeded per pair. Also the bf16 quantisation noise floor.

Readouts at the answer position of month-recall prompts (deterministic, pure logits):
  (i)   identity rotation: logprob(B) - logprob(A), plus the full 12-month logprob
        vector, from which the traversal statistics are derived for every arm.
  (ii)  coherence cost, THREE metrics, all reported: NLL of the prompt's own in-context
        tokens after the intervention (cost_mid, PRE-REGISTERED); NLL of a fixed neutral
        continuation (cost_cont); and the spec's metric, the NLL/perplexity of a GREEDY
        steered generation scored under the UNSTEERED model (cost_gen), per instance so
        it carries a CI.
  (iii) off-target damage: KL(steered || unsteered) of the next-token distribution on
        structurally identical prompts whose filler is a CITY, not a month.

THE DISCRIMINATOR FOR "WALK" VERSUS "SUBSTITUTE". A ring traversal must put probability
mass on the months it passes THROUGH. route_intermediate_mass is the mass on months
strictly between A and B along the signed shortest arc the arc arm actually walks (for
the antipodal pairs that arc runs backwards through the calendar, so the prediction is
sharp and unusual: Jan->Jul should light up December, November, October). An identity
substitution puts no mass there. This is computed for every arm, so the claim is
testable against the controls rather than asserted.

Everything that could be chosen after seeing results is declared in the output JSON's
`declared` block, which is written from constants defined at the top of this file.
"""
import argparse, contextlib, json, os, sys, time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from gfdisc.env import DEVICE
from gfdisc.model import load_model, capture
from gfdisc.fits import pca_2d, fit_circle, circularity, angles, cyclic_order_score
from gfdisc.tokens import MONTHS

# ---------------------------------------------------------------------------
# DECLARED BEFORE RESULTS. Everything in this block is a pre-commitment.
# The items marked [FROZEN R1] are unchanged from revision 1 and fix the verdict.
# ---------------------------------------------------------------------------

PRIMARY_ARM = "manifold_chord"                                      # [FROZEN R1]
BASELINE_ARM = "contrastive"                                        # [FROZEN R1]

# Norm-matching rule. For every arm the applied displacement is
#     delta = eps * ||h_pos||_2 * d_arm / ||d_arm||_2
# where h_pos is the UNSTEERED float32 layer-9 residual at the intervened token of
# THAT prompt. The scale is therefore matched across arms per prompt, which is the
# comparison the plan requires. eps is dimensionless.                [FROZEN R1]
#
# The grid brackets the natural contrastive size: ||mean_B - mean_A|| is 256-419 at
# layer 9 (median 347) against ||h_pos|| ~= 6275, i.e. a natural eps of ~0.055.
# EPS_HEADLINE is the grid point at that natural scale and is where the verdict is
# read. The two largest points overshoot the ring diameter (2r ~= 315) and are
# declared in advance to be an in-plane-DIRECTION test, not manifold interpolation.
#
# REVISION 2 [R4]: 0.005 and 0.01 are ADDED below the revision-1 grid. Revision 1's
# smallest point, 0.02, gives |delta| ~= 125, which already overshoots the primary
# arm's own chord to B by 1.13-1.61x on the four adjacent pairs -- the primary arm was
# never tested at or under its natural scale there. EPS_HEADLINE is UNCHANGED.
EPS_GRID_R1 = [0.02, 0.04, 0.056, 0.08, 0.16, 0.32]                 # [FROZEN R1]
EPS_GRID_ADDED = [0.005, 0.01]                                      # [R4]
EPS_GRID = sorted(EPS_GRID_ADDED + EPS_GRID_R1)
EPS_HEADLINE = 0.056                                                # [FROZEN R1]

# The unrescaled rows. Kept because they demonstrate why the guard matters: without
# norm matching the arms differ in length by up to 4x and the comparison is void.
INCLUDE_NATURAL = True                                              # [FROZEN R1]

# REVISION 2 [R4]: the CHORD-MATCHED row. Every arm is rescaled to the primary arm's
# OWN natural per-prompt norm, ||q_B - P_i||, the in-plane chord from this prompt's
# residual to the ring point at B's angle. Overshoot is 1.00 by construction for the
# primary arm, and the contrastive vector is rescaled DOWN to the manifold's scale
# rather than the manifold being pushed up to the contrastive one. It is the mirror
# image of the eps rows and is reported alongside them, not instead of them.
INCLUDE_CHORD_MATCHED = True                                        # [R4]

# Prompt qualification rule. An (on-target template, source month A) instance enters
# the analysis only if the UNSTEERED model already ranks A first among the twelve
# month tokens at the answer position. Label-blind with respect to every arm; the
# dropped count is recorded.                                        # [FROZEN R1]
QUALIFY_RULE = "unsteered argmax over the 12 month tokens at the answer position == A"

# Verdict rule. At EPS_HEADLINE, over qualified (prompt, pair) instances, take the
# PAIRED differences primary-arm minus baseline-arm of
#   rotation = logprob(B) - logprob(A)
#   cost     = mean NLL of the in-context tokens after the intervened position
# and bootstrap a 95% CI over instances (10,000 resamples, seed 0).
#   DOMINATES : CI(delta rotation) entirely > 0 AND CI(delta cost) not entirely > 0
#   LOSES     : CI(delta rotation) entirely < 0 AND CI(delta cost) not entirely < 0
#   TIE       : anything else, including "rotates more but costs more"   [FROZEN R1]
N_BOOT, BOOT_SEED = 10000, 0                                        # [FROZEN R1]

# REVISION 2 [R3]: the SAME rule is additionally applied with `cost` replaced by each
# of the other two coherence metrics. All three labels are reported unconditionally;
# the pre-registered one (cost_mid) remains the headline. Computing three and citing
# one would be the cherry-pick this project's discipline forbids, so the block is
# written whole or not at all.
COST_METRICS = [("cost_mid", "mean NLL of the in-context tokens after the intervened "
                             "position (PRE-REGISTERED, revision 1)"),
                ("cost_cont", "mean NLL of a fixed neutral continuation one sentence "
                              "downstream"),
                ("cost_gen", "mean NLL of the model's own greedy 16-token continuation "
                             "scored under the UNSTEERED model, first generated token "
                             "excluded (the spec's metric)")]

# Random-control seeds. Nothing else in the probe samples: all readouts are logits and
# all generation is greedy.
# REVISION 2 [R5]: revision 1 drew these from rng(1000*s + 7) with the seed base fixed
# at 7 for every pair, so the same three direction matrices were reused nine times and
# the control rested on 3 independent directions, not 27. The base is now per pair.
RANDOM_SEEDS = [0, 1, 2]
RANDOM_SEEDBASE = lambda pair_idx: 7 + 101 * pair_idx               # [R5]

# REVISION 2 [R6]: the second intervention site. "month" is the source-month token
# (revision 1's only site). "answer" is the final prompt token, which predicts the
# month and where no month token sits; the contrastive vector cannot act as a token
# substitution there. Run at EPS_HEADLINE only, all pairs, all arms.
SITES = ["month", "answer"]
# Two strengths at the second site, so "nothing steers there" cannot be dismissed as
# too small a push: EPS_HEADLINE for the matched comparison and 0.16 as a strong probe.
SITES_EPS_ANSWER = [EPS_HEADLINE, 0.16]
SITE_EPS = {"month": None, "answer": SITES_EPS_ANSWER}   # None = the whole grid

SEQUENCE_DISCLOSURE = (
    "REVISION 1: arm set extended (contrastive_inplane, contrastive_orth, ring-locality "
    "stats) after a 2-pair, 6-prompt smoke run; the primary arm, baseline arm, eps grid, "
    "qualification rule and verdict rule were fixed before any model output was seen "
    "and are unchanged. REVISION 2: after adversarial review of revision 1's output, "
    "six secondary additions (see declared.revision). The primary arm, baseline arm, "
    "EPS_HEADLINE, qualification rule and verdict rule are UNCHANGED from revision 1, "
    "so the headline verdict cell is expected to reproduce revision 1 exactly; the "
    "reproduction is asserted in reproduction_check. random_orth rows are EXPECTED to "
    "differ from revision 1 because the seed base is now per pair.")

REVISION_NOTES = [
    dict(id="R1", finding="revision 1 stored the dose-response traversal only for "
         "manifold_arc, so the claim 'no control reproduces the calendar walk' could "
         "not be checked from the output. Recomputed from revision 1's rows, it is "
         "false: contrastive_inplane walks near-identically (Aug->Sep 7.28/8.38/9.41/"
         "10.00/10.48/10.60 vs manifold_arc 7.24/8.19/9.27/10.01/10.50/10.67).",
         fix="traversal is computed for ALL eight arms, with a per-arm continuity "
             "summary (monotone signed progress, peak-on-route, route-intermediate "
             "mass), so the surviving claim is stated against the controls."),
    dict(id="R2", finding="manifold_arc clamps to 'high' -- degenerating to a rescaled "
         "chord, not a walk -- at 20/21 or 21/21 prompts for Nov->Dec and Aug->Sep at "
         "every grid point, the two pairs quoted as evidence of walking. arc_t_med was "
         "1.00 in every such cell.",
         fix="frac_unclamped and arc_t_med are surfaced per cell and a "
             "genuine_walk_cells list is emitted; the positive is cited only there."),
    dict(id="R3", finding="the pre-registered cost axis (in-context NLL) is near-"
         "degenerate: at EPS_HEADLINE every arm sits within 0.007 nats of zero and all "
         "are NEGATIVE, with the null direction random_orth_s0 'improving' it most "
         "(-0.0354). The verdict rule's cost clause is therefore close to vacuous, and "
         "the pre-registered per-unit-cost question returned n_usable=0, unanswered. On "
         "the spec's own metric (generation perplexity under the unsteered model) the "
         "sign of the difference FLIPS with eps and revision 1 had n=3 pairs, no CI.",
         fix="generation NLL is per-instance on all nine pairs at eps 0.04 and 0.056 "
             "(plus 0.16 on three pairs), with the first generated token EXCLUDED so a "
             "successful steer is not charged for naming the month it was asked to "
             "name; the verdict is reported under all three cost metrics; efficiency "
             "is recomputed on the non-degenerate denominator."),
    dict(id="R4", finding="the primary arm was never tested at or below its own natural "
         "chord length on the four adjacent pairs: overshoot_vs_chord was already 1.13-"
         "1.61 at the smallest grid point and 3.17-4.51 at EPS_HEADLINE. The contrastive "
         "vector was never rescaled DOWN to the manifold's own scale.",
         fix="eps 0.005 and 0.01 added below the revision-1 grid, and a chord-matched "
             "norm row in which every arm is scaled to the per-prompt chord to B."),
    dict(id="R5", finding="the three random 'seeds' were three fixed direction matrices "
         "reused across all nine pairs (rng(1000*s+7) with an invariant seed base and a "
         "constant row count), so the control rested on 3 independent directions.",
         fix="seed base is 7 + 101*pair_index: 27 independent directions."),
    dict(id="R7", finding="SELF-FOUND during revision 2's own verification pass, not by "
         "the reviewer. Three of revision 2's new derived statistics were themselves "
         "defective. (a) route_stats took the route's calendar direction from the month "
         "indices (a half-turn always descending), but arc_delta walks the GEOMETRIC "
         "shortest arc on a PCA embedding that is reflected relative to the calendar; "
         "on this fit the two disagree for March->September, whose arc walks ASCENDING "
         "(Apr..Aug) while the statistic scored its intermediate mass on the descending "
         "half of the circle the arm never visits -- exactly the pair whose arc peaked "
         "at August with an apparent intermediate mass of 0.075. (b) the continuity "
         "summary took a max over the whole eps grid including the destructive points, "
         "where every arm flattens the month distribution and accrues intermediate mass "
         "by dilution: it scored the NULL directions at 0.29. (c) frac_peak_on_route "
         "scored 1.000 for the random arms for the trivial reason that an arm which "
         "does not move keeps its peak on A, and A is on its own route.",
         fix="(a) the calendar direction of each pair's walked arc is computed from the "
             "fitted ring (manifold.calendar_orientation and route_direction_by_pair) "
             "and passed in; (b) the continuity summary is restricted to eps <= "
             "EPS_HEADLINE and the headline comparison moved to genuine_walks."
             "intermediate_mass_by_arm, which scores every arm on the SAME unclamped "
             "cells at the SAME norm; (c) frac_peak_moved_and_on_route added, requiring "
             "the peak to have left A."),
    dict(id="R6", finding="the only intervention site was the source-month token, where "
         "mean_B - mean_A largely carries B's token identity, making the contrastive "
         "baseline close to an oracle substitution (frac_argmax_B = 1.00). The result "
         "was therefore scoped to that site without the site-dependence being measured.",
         fix="the headline comparison is repeated at the answer position, where no "
             "month token sits."),
]

# ---------------------------------------------------------------------------

# The manifold is fitted on these, the same twenty templates that produced
# results/months_power_big_L9.json, so the plane under test is that recorded object.
MON_T = ["The report was published in {}", "She was born in {}", "It rained all {}",
         "The season begins in {}", "We travelled there last {}", "The deadline was {}",
         "Sales peaked in {}", "The festival runs through {}", "He resigned in {}",
         "The contract expires in {}", "Enrolment opens in {}", "The survey ran during {}",
         "Prices fell in {}", "The court ruled in {}", "The harvest starts in {}",
         "The exhibition closed in {}", "Construction began in {}", "The law took effect in {}",
         "The team formed in {}", "Records were broken in {}"]

# On-target contexts: month identity has a measurable consequence at the answer
# position, because the model has to restate it. Twenty-two of them; each mentions the
# month exactly once, at the token that gets intervened on.
ON_T = [
    "The report was published in {m}. It was published in",
    "The event took place in {m}. The event took place in",
    "Note: the meeting is in {m}.\nReminder: the meeting is in",
    "Fact: the deadline is in {m}.\nSo the deadline is in",
    "The festival happens in {m} every year. The festival happens in",
    "A: When? B: In {m}. A: Sorry, when? B: In",
    "She was born in {m}. Her birth month is",
    "The contract expires in {m}. Expiry month:",
    "The report was published in {m}. Question: in which month was it published? Answer:",
    "Construction began in {m}. Construction began in",
    "The team was formed in {m}. The team was formed in",
    "Sales peaked in {m}. Sales peaked in",
    "The law took effect in {m}. The law took effect in",
    "The survey ran during {m}. The survey ran during",
    "Prices fell in {m}. Prices fell in",
    "The harvest starts in {m}. The harvest starts in",
    "The exhibition closed in {m}. The exhibition closed in",
    "He resigned in {m}. He resigned in",
    "The season begins in {m}. The season begins in",
    "Records were broken in {m}. Records were broken in",
    "The court ruled in {m}. The court ruled in",
    "Enrolment opened in {m}. Enrolment opened in",
]

# Off-target contexts: the same grammatical frames with a CITY in the slot. The same
# delta is applied at the city token and KL is read at the answer position, so the
# damage measured is what the intervention does to a computation the month ring has
# no business touching.
OFF_T = [
    "The report was published in {m}. It was published in",
    "The event took place in {m}. The event took place in",
    "Note: the meeting is in {m}.\nReminder: the meeting is in",
    "Fact: the office is in {m}.\nSo the office is in",
    "The festival happens in {m} every year. The festival happens in",
    "A: Where? B: In {m}. A: Sorry, where? B: In",
    "She was raised in {m}. Her home city is",
    "The company is based in {m}. Head office:",
    "The report was published in {m}. Question: in which city was it published? Answer:",
    "Construction began in {m}. Construction began in",
    "The team was formed in {m}. The team was formed in",
    "Sales peaked in {m}. Sales peaked in",
]
OFF_FILLERS = ["Berlin", "Paris", "London", "Madrid", "Toronto", "Dublin",
               "Vienna", "Lisbon", "Boston", "Chicago", "Oslo", "Cairo"]

# The coherence continuation. BRIDGE is not scored: it sits at the answer position,
# where every arm is being pushed to name a different month, so its NLL measures the
# steer's success rather than its collateral cost. SCORED is ordinary English one
# sentence downstream.
BRIDGE = " the same period."
SCORED = " The details are recorded in the official archive for future reference."

# Pairs: three adjacent, two a quarter apart, three opposite, one reversed-adjacent.
PAIRS = [("March", "April"), ("August", "September"), ("November", "December"),
         ("July", "June"),
         ("February", "May"), ("June", "September"),
         ("January", "July"), ("March", "September"), ("May", "November")]

# REVISION 2 [R3]: generation now covers ALL pairs at the two eps points where the
# cost comparison is contested, and the three original pairs at the strong point.
GEN_EPS_ALL = [0.04, EPS_HEADLINE]
GEN_PAIRS_EXTRA = [("August", "September"), ("June", "September"), ("January", "July")]
GEN_EPS_EXTRA = [0.16]
GEN_ARMS = ["manifold_chord", "manifold_arc", "contrastive",
            "contrastive_inplane", "contrastive_orth", "random_orth_s0"]
GEN_TOKENS = 16
# The first generated token IS the steered month: scoring it under the unsteered model
# charges a SUCCESSFUL steer for doing what it was asked to do, exactly the confound the
# unscored BRIDGE avoids in the sweep. Both variants are stored; cost_gen uses ex1.
GEN_SCORE_VARIANT = "ex1"

LAYER = 9

# instance-table eps sentinels, distinct so the natural and chord rows cannot collide
# in the paired keying (revision 1 had only the natural sentinel).
EPS_SENTINEL = {"natural": -1.0, "chord": -2.0}


# ---------------------------------------------------------------------------
# intervention machinery
# ---------------------------------------------------------------------------

@contextlib.contextmanager
def add_at(model, layer, idx_b, idx_p, delta):
    """Add `delta[i]` to the layer-`layer` residual stream at (idx_b[i], idx_p[i]).

    Position-specific, unlike `gfdisc.model.splice`, which replaces every position.
    The arithmetic is done in float32 and cast back, exactly as `splice` does: a
    bf16->fp32->bf16 round trip is lossless, so a zero delta must reproduce the
    baseline logits bit for bit. That is asserted at startup.
    """
    def hook(_m, _i, out):
        t = out[0] if isinstance(out, tuple) else out
        new = t.float()
        new[idx_b, idx_p] = new[idx_b, idx_p] + delta
        new = new.to(t.dtype)
        return (new,) + out[1:] if isinstance(out, tuple) else new

    h = model.model.layers[layer].register_forward_hook(hook)
    try:
        yield
    finally:
        h.remove()


def ring_point(cx, cy, r, theta):
    return np.array([cx + r * np.cos(theta), cy + r * np.sin(theta)])


def signed_arc(th_a, th_b):
    """The shortest signed arc from A's angle to B's angle. For a half-turn (the
    antipodal pairs) this returns -pi, i.e. the DESCENDING route through the calendar,
    and `route_stats` uses the same convention so the two agree by construction."""
    return (th_b - th_a + np.pi) % (2 * np.pi) - np.pi


def arc_delta(p, cx, cy, r, th_a, th_b, target):
    """Walk along the ring from A's angle toward B's angle until the in-plane
    displacement from p reaches `target`.

    V's rows are orthonormal, so the 1152-d norm of (q - p) @ V equals the 2-d norm of
    (q - p): the search is a bisection in the plane. Two clamps are possible and both
    are flagged rather than hidden -- `low` when even t=0 (snapping onto the ring at
    A's own angle) already overshoots, `high` when the full walk to B is still shorter
    than the target. Where a clamp fires the arm is a rescaled chord, not a walk, and
    revision 2 surfaces that per cell instead of leaving it in the flags.
    """
    d = signed_arc(th_a, th_b)
    f = lambda t: np.linalg.norm(ring_point(cx, cy, r, th_a + t * d) - p)
    # the endpoint comparisons carry a relative tolerance: th_a + d and th_b are the
    # same angle mathematically but not bit-identical, so an exact `<=` reports a walk
    # of t = 0.9999999 as unclamped and would corrupt the very frac_unclamped statistic
    # revision 2 adds to keep the clamped cells out of the positive.
    tol = 1.0 + 1e-9
    if f(0.0) * tol >= target:
        return ring_point(cx, cy, r, th_a) - p, 0.0, "low"
    if f(1.0) <= target * tol:
        return ring_point(cx, cy, r, th_b) - p, 1.0, "high"
    lo, hi = 0.0, 1.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if f(mid) < target:
            lo = mid
        else:
            hi = mid
    t = 0.5 * (lo + hi)
    fl = "high" if t > 1.0 - 1e-6 else ("low" if t < 1e-6 else "")
    return ring_point(cx, cy, r, th_a + t * d) - p, float(t), fl


# ---------------------------------------------------------------------------
# batching helpers
# ---------------------------------------------------------------------------

def build_batch(tok, prompts, fillers, bridge_ids, scored_ids, device):
    """Tokenise prompt + BRIDGE + SCORED, locating the filler token in each prompt."""
    rows = []
    for text, w in zip(prompts, fillers):
        pid = tok.encode(text, add_special_tokens=True)
        wid = tok.encode(" " + w, add_special_tokens=False)
        assert len(wid) == 1, f"{w!r} is not a single token"
        hits = [i for i, t in enumerate(pid) if t == wid[0]]
        assert len(hits) == 1, f"filler {w!r} occurs {len(hits)}x in {text!r}"
        rows.append(dict(prompt_ids=pid, mpos=hits[0], plen=len(pid),
                         ids=pid + bridge_ids + scored_ids, text=text, filler=w))
    L = max(len(r["ids"]) for r in rows)
    n = len(rows)
    ids = torch.zeros(n, L, dtype=torch.long)
    attn = torch.zeros(n, L, dtype=torch.long)
    for i, r in enumerate(rows):
        ids[i, :len(r["ids"])] = torch.tensor(r["ids"])
        attn[i, :len(r["ids"])] = 1
    return rows, ids.to(device), attn.to(device)


def site_positions(rows, site):
    """The intervened position per row. 'month' is the filler token (revision 1's only
    site); 'answer' is the final prompt token, which predicts the month."""
    if site == "month":
        return [r["mpos"] for r in rows]
    if site == "answer":
        return [r["plen"] - 1 for r in rows]
    raise ValueError(site)


def readout(logits, rows, month_ids, bridge_n, scored_ids, keep_lp=False):
    """Per-example: 12-month logprobs at the answer position, in-context NLL after the
    intervention, and continuation NLL. `keep_lp` also returns the full logprob row at
    the answer position, which the KL control needs.

    The log_softmax is taken on a per-example SLICE of the needed positions rather than
    on the whole (batch, seq, 262144) tensor: casting that whole thing to float32 costs
    over a gigabyte on a machine shared with sibling agents.
    """
    dev = logits.device
    out = []
    for i, r in enumerate(rows):
        ans = r["plen"] - 1                                   # predicts the month
        tgt = r["prompt_ids"][r["mpos"] + 1: r["plen"]]
        mid_idx = list(range(r["mpos"], r["plen"] - 1))
        s0 = r["plen"] + bridge_n                             # first scored token
        cont_idx = list(range(s0 - 1, s0 - 1 + len(scored_ids)))
        want = sorted(set([ans] + mid_idx + cont_idx))
        pos = {p: j for j, p in enumerate(want)}
        lp = torch.log_softmax(logits[i, want].float(), dim=-1)
        mlp = lp[pos[ans], month_ids].cpu().numpy()
        if tgt:
            nll_mid = float(-lp[[pos[p] for p in mid_idx],
                                torch.tensor(tgt, device=dev)].mean())
        else:
            nll_mid = float("nan")
        nll_cont = float(-lp[[pos[p] for p in cont_idx],
                             torch.tensor(scored_ids, device=dev)].mean())
        rec = dict(month_lp=mlp, nll_mid=nll_mid, nll_cont=nll_cont)
        if keep_lp:
            rec["ans_lp"] = lp[pos[ans]].clone()
        out.append(rec)
        del lp
    return out


def kl_rows(steered, base):
    """KL(steered || unsteered) at the answer position, full 262k vocabulary."""
    ks = []
    for s, b in zip(steered, base):
        p = s["ans_lp"]
        ks.append(float((p.exp() * (p - b["ans_lp"])).sum()))
    return ks


def ring_locality(month_lp, iA, iB):
    """Where on the calendar circle does the steered month distribution sit, and how
    tightly?

    Derived entirely from the twelve month log-probs, so it costs no forward passes and
    enters no verdict. It separates two things the scalar rotation metric conflates:

      a RING TRAVERSAL moves the mass smoothly around the circle, so the circular mean
      lands between A and B and the neighbours of the peak are lifted with it;
      an IDENTITY SUBSTITUTION spikes one month and leaves its calendar neighbours flat.

    circ_spread_deg is the circular standard deviation sqrt(-2 ln R) of the twelve-month
    distribution; a delta function scores 0, a uniform distribution scores unboundedly.
    """
    p = np.exp(np.asarray(month_lp, dtype=np.float64))
    p = p / max(p.sum(), 1e-300)
    ang = 2 * np.pi * np.arange(12) / 12.0
    z = (p * np.exp(1j * ang)).sum()
    R = float(np.abs(z))
    cm = float(np.angle(z)) % (2 * np.pi)

    def cdist(a, b):
        return float(np.degrees(np.abs((a - b + np.pi) % (2 * np.pi) - np.pi)))

    nb = [(iB - 1) % 12, (iB + 1) % 12]
    return dict(circ_mean_month=float((cm / (2 * np.pi)) * 12.0),
                circ_R=R,
                circ_spread_deg=float(np.degrees(np.sqrt(max(-2.0 * np.log(max(R, 1e-12)), 0.0)))),
                dist_mean_to_A_deg=cdist(cm, ang[iA]),
                dist_mean_to_B_deg=cdist(cm, ang[iB]),
                peak_month=int(np.argmax(p)),
                neighbour_lift=float(np.mean([month_lp[j] for j in nb]) - float(np.mean(month_lp))))


def calendar_orientation(TH, months):
    """Does the fitted ring's angle INCREASE or DECREASE with calendar order?

    PCA fixes the plane but not its handedness, and on this fit the embedding is
    REFLECTED: the angle decreases as the calendar advances. `arc_delta` walks the
    geometric shortest arc, so the calendar direction of that walk is
    sign(signed_arc) * orientation. Getting this wrong silently mislabels which months
    an arc passes THROUGH, which is exactly what route_intermediate_mass measures --
    so it is computed from the fit and recorded, never assumed.

    Returns +1 if angle increases with the calendar, -1 if it decreases.
    """
    steps = [signed_arc(TH[months[i]], TH[months[(i + 1) % 12]]) for i in range(12)]
    return 1 if float(np.mean(steps)) > 0 else -1


def route_stats(month_lp, iA, iB, s_cal):
    """[R1/R2] The walk-versus-substitute discriminator, computed for EVERY arm.

    `s_cal` is the CALENDAR direction (+1 ascending, -1 descending) of the geometric
    arc that `arc_delta` walks for this pair, supplied by the caller from the fitted
    ring angles. For pairs less than a half-turn apart the two possible conventions
    select the same set of intermediate months, so s_cal only matters for the three
    antipodal pairs -- where it matters completely: Jan->Jul and May->Nov walk the
    DESCENDING route on this fit (Jan, Dec, Nov, Oct, Sep, Aug, Jul) while Mar->Sep
    walks the ASCENDING one. A calendar-only convention gets Mar->Sep backwards and
    scores its intermediate mass on the half of the circle the arm never visits.

      signed_progress_months  displacement of the circular mean from A toward B, in
                              months, signed along the route. 0 = unmoved, |A->B| = on B,
                              beyond that = overshot past B.
      route_intermediate_mass probability on months STRICTLY between A and B along the
                              route. A traversal passes through them; a substitution
                              does not. Undefined (None) for adjacent pairs, which have
                              no intermediate months.
      peak_on_route           whether the argmax month lies on the walked route.
      peak_moved_and_on_route the same, but requiring the peak to have LEFT A. A control
                              that does not move at all scores peak_on_route = 1 for the
                              trivial reason that A is on its own route; this field is
                              the one that discriminates.
    """
    p = np.exp(np.asarray(month_lp, dtype=np.float64))
    p = p / max(p.sum(), 1e-300)
    ang = 2 * np.pi * np.arange(12) / 12.0
    z = (p * np.exp(1j * ang)).sum()
    cm = float(np.angle(z))

    n_steps = int(((iB - iA) * s_cal) % 12)          # steps from A to B along the walk
    s = int(s_cal)
    d_ang = (cm - ang[iA] + np.pi) % (2 * np.pi) - np.pi
    prog = s * (d_ang / (2 * np.pi)) * 12.0

    route = [(iA + s * k) % 12 for k in range(n_steps + 1)]
    inter = route[1:-1]
    pk = int(np.argmax(p))
    return dict(route_months=route, route_dir=s,
                signed_progress_months=float(prog),
                progress_frac_of_AB=float(prog / n_steps) if n_steps else None,
                route_intermediate_mass=(float(p[inter].sum()) if inter else None),
                route_mass=float(p[route].sum()),
                peak_on_route=bool(pk in route),
                peak_moved_and_on_route=bool(pk in route and pk != iA),
                peak_month=pk)


def boot_ci(x, n_boot=N_BOOT, seed=BOOT_SEED):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    m = x[rng.integers(0, len(x), size=(n_boot, len(x)))].mean(1)
    return float(x.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def label_from(lo_r, hi_r, lo_c, hi_c):
    """The pre-registered verdict rule, factored so the alternative cost metrics are
    scored by exactly the same rule rather than by a rewritten one."""
    if lo_r > 0 and not (lo_c > 0):
        return "manifold_dominates"
    if hi_r < 0 and not (hi_c < 0):
        return "manifold_loses"
    return "tie"


def verdict_detail(lo_r, hi_r, lo_c, hi_c):
    d = ("rotation CI > 0" if lo_r > 0 else "rotation CI < 0" if hi_r < 0 else
         "rotation CI spans 0")
    return d + "; " + ("cost CI > 0 (manifold costs more)" if lo_c > 0 else
                       "cost CI < 0 (manifold costs less)" if hi_c < 0 else
                       "cost CI spans 0")


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=LAYER)
    ap.add_argument("--out", default="results/steering_probe.json")
    ap.add_argument("--skip-gen", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="2 pairs, 3 eps, no gen")
    a = ap.parse_args()
    t_start = time.time()
    L = a.layer

    pairs = PAIRS
    eps_grid = EPS_GRID
    if a.smoke:
        pairs = PAIRS[:2] + PAIRS[6:7]
        eps_grid = [0.01, EPS_HEADLINE]

    model, tok = load_model(token=os.environ.get("HF_TOKEN"))
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    month_ids = [tok.encode(" " + m, add_special_tokens=False)[0] for m in MONTHS]
    assert all(len(tok.encode(" " + m, add_special_tokens=False)) == 1 for m in MONTHS)
    bridge_ids = tok.encode(BRIDGE, add_special_tokens=False)
    scored_ids = tok.encode(SCORED, add_special_tokens=False)
    print(f"loaded; device={DEVICE}; bridge={len(bridge_ids)} tok, scored={len(scored_ids)} tok",
          flush=True)

    # ---- 1. the manifold -------------------------------------------------
    Mm = []
    with torch.inference_mode():
        for m in MONTHS:
            enc = tok([t.format(m) for t in MON_T], return_tensors="pt",
                      padding=True).to(DEVICE)
            with capture(model, [L]) as st:
                model(**enc, use_cache=False)
            last = enc["attention_mask"].sum(1) - 1
            Mm.append(st[L][torch.arange(last.shape[0]), last].float().mean(0).cpu().numpy())
    Mm = np.vstack(Mm).astype(np.float64)
    mu = Mm.mean(0)
    xy, ratios, Vt = pca_2d(Mm)
    V = Vt[:2]                                    # (2, d), orthonormal rows
    cx, cy, r, resid = fit_circle(xy)
    circ = circularity(xy, cx, cy, r)
    th = angles(xy, cx, cy)
    # ordering under BOTH SVD sign conventions; cyclic_order_score is reflection
    # invariant so these must agree, and are reported rather than assumed.
    ord_rows = {}
    for name, sgn in (("as_fitted", 1.0), ("pc2_flipped", -1.0)):
        xy2 = xy * np.array([1.0, sgn])
        c2 = fit_circle(xy2)
        s = cyclic_order_score(angles(xy2, c2[0], c2[1]), np.arange(12), 12,
                               n_perm=20000, seed=0)
        ord_rows[name] = dict(rho=s["rho"], p=s["p"], symmetry=list(s["symmetry"]),
                              null_p95=s["null_p95"])
    manifold = dict(layer=L, n_templates=len(MON_T), top2_var=float(ratios[:2].sum()),
                    circle_resid=float(resid), radius=float(r),
                    gap_ratio=circ["gap_ratio"], max_gap_deg=circ["max_gap_deg"],
                    radius_ratio=circ["radius_ratio"],
                    theta_deg={m: float(np.degrees(t)) for m, t in zip(MONTHS, th)},
                    ordering=ord_rows,
                    mean_norm=float(np.linalg.norm(Mm, axis=1).mean()),
                    mu_norm=float(np.linalg.norm(mu)))
    print(f"manifold: top2={manifold['top2_var']:.3f} resid={manifold['circle_resid']:.3f} "
          f"r={r:.1f} gap_ratio={circ['gap_ratio']:.3f} "
          f"rho={ord_rows['as_fitted']['rho']:.3f} p={ord_rows['as_fitted']['p']:.5f}",
          flush=True)

    TH = {m: float(t) for m, t in zip(MONTHS, th)}
    ORIENT = calendar_orientation(TH, MONTHS)
    manifold["calendar_orientation"] = dict(
        value=int(ORIENT),
        meaning="+1 = the fitted ring's angle increases with calendar order, -1 = it "
                "decreases (the PCA embedding is reflected). Used to convert the "
                "geometric arc direction arc_delta walks into the calendar months it "
                "passes through.")
    ROUTE_DIR = {f"{A}->{B}": int(np.sign(signed_arc(TH[A], TH[B])) * ORIENT)
                 for A, B in pairs}
    manifold["route_direction_by_pair"] = ROUTE_DIR
    print(f"calendar orientation of the fitted ring: {ORIENT}; route dirs {ROUTE_DIR}",
          flush=True)
    contrast_natural = {f"{A}->{B}": float(np.linalg.norm(Mm[MONTHS.index(B)] - Mm[MONTHS.index(A)]))
                        for A, B in pairs}
    manifold["contrastive_natural_norm"] = contrast_natural
    manifold["chord_natural_norm_meanpoint"] = {
        f"{A}->{B}": float(np.linalg.norm(ring_point(cx, cy, r, TH[B]) - xy[MONTHS.index(A)]))
        for A, B in pairs}

    # ---- 2. prompt sets --------------------------------------------------
    off_prompts, off_fill = [], []
    for i, t in enumerate(OFF_T):
        for j in (0, 1):
            w = OFF_FILLERS[(i + 6 * j) % len(OFF_FILLERS)]
            off_prompts.append(t.format(m=w)); off_fill.append(w)
    off_rows, off_ids, off_attn = build_batch(tok, off_prompts, off_fill,
                                              bridge_ids, scored_ids, DEVICE)
    print(f"off-target prompts: {len(off_rows)}", flush=True)

    def fwd(ids, attn, idx_b=None, idx_p=None, delta=None):
        with torch.inference_mode():
            if delta is None:
                return model(input_ids=ids, attention_mask=attn, use_cache=False).logits
            with add_at(model, L, idx_b, idx_p, delta):
                return model(input_ids=ids, attention_mask=attn, use_cache=False).logits

    # hook-correctness: a zero delta must reproduce the baseline exactly
    zb = torch.arange(len(off_rows), device=DEVICE)
    zp = torch.tensor(site_positions(off_rows, "month"), device=DEVICE)
    lg0 = fwd(off_ids, off_attn)
    lgz = fwd(off_ids, off_attn, zb, zp, torch.zeros(len(off_rows), 1152, device=DEVICE))
    hook_maxdiff = float((lg0.float() - lgz.float()).abs().max())
    print(f"hook identity check: max |delta logit| = {hook_maxdiff:.3e}", flush=True)
    off_base = readout(lg0, off_rows, month_ids, len(bridge_ids), scored_ids, keep_lp=True)
    del lgz, lg0

    # per-source-month on-target batch, its baseline, and qualification. h at BOTH
    # intervention sites is captured in the same pass.
    on_cache, qual_report = {}, {}
    for A in sorted({p[0] for p in pairs}):
        rows, ids, attn = build_batch(tok, [t.format(m=A) for t in ON_T],
                                      [A] * len(ON_T), bridge_ids, scored_ids, DEVICE)
        lg = fwd(ids, attn)
        base = readout(lg, rows, month_ids, len(bridge_ids), scored_ids)
        keep = [i for i, b in enumerate(base) if MONTHS[int(np.argmax(b["month_lp"]))] == A]
        qual_report[A] = dict(n_total=len(rows), n_kept=len(keep),
                              dropped=[rows[i]["text"] for i in range(len(rows)) if i not in keep])
        hp = {}
        with torch.inference_mode():
            with capture(model, [L]) as st:
                model(input_ids=ids, attention_mask=attn, use_cache=False)
            for site in SITES:
                pp = torch.tensor(site_positions(rows, site), device=DEVICE)
                hp[site] = st[L][torch.arange(len(rows)), pp].float().cpu().numpy().astype(np.float64)
        on_cache[A] = dict(rows=rows, ids=ids, attn=attn, base=base, keep=keep, hpos=hp)
        del lg
        print(f"  {A}: qualified {len(keep)}/{len(rows)}", flush=True)

    # off-target residuals at both sites, captured once (they do not depend on the pair)
    hoff = {}
    with torch.inference_mode():
        with capture(model, [L]) as st:
            model(input_ids=off_ids, attention_mask=off_attn, use_cache=False)
        for site in SITES:
            pp = torch.tensor(site_positions(off_rows, site), device=DEVICE)
            hoff[site] = st[L][torch.arange(len(off_rows)), pp].float().cpu().numpy().astype(np.float64)

    # diagnostic: do the steering prompts sit on the ring the manifold was fit on?
    plane_diag = {}
    for site in SITES:
        plane_diag[site] = {}
        for A, c in on_cache.items():
            P = (c["hpos"][site] - mu) @ V.T
            rad = np.linalg.norm(P - np.array([cx, cy]), axis=1)
            ang = np.degrees(np.arctan2(P[:, 1] - cy, P[:, 0] - cx))
            d = (ang - np.degrees(TH[A]) + 180) % 360 - 180
            plane_diag[site][A] = dict(
                inplane_radius_med=float(np.median(rad)), ring_radius=float(r),
                angle_offset_from_A_deg_med=float(np.median(d)),
                angle_offset_from_A_deg_iqr=float(np.percentile(d, 75) - np.percentile(d, 25)),
                hpos_norm_med=float(np.median(np.linalg.norm(c["hpos"][site], axis=1))))

    # ---- 3. the sweep ----------------------------------------------------
    ARMS = [("manifold_chord", None), ("manifold_arc", None), ("contrastive", None),
            ("contrastive_inplane", None), ("contrastive_orth", None)] + \
           [(f"random_orth_s{s}", s) for s in RANDOM_SEEDS]
    ARM_NAMES = [x for x, _ in ARMS]

    agg, inst = [], []
    inst_schema = ["pair", "site", "arm", "eps", "norm_kind", "prompt_idx",
                   "lpA", "lpB", "nll_mid", "nll_cont"]
    n_cfg = 0
    rand_seedbases = {}
    for pi, (A, B) in enumerate(pairs):
        seedbase = RANDOM_SEEDBASE(pi)                                    # [R5]
        rand_seedbases[f"{A}->{B}"] = int(seedbase)
        c = on_cache[A]
        rows, keep = c["rows"], c["keep"]
        nb = len(rows)
        iA, iB = MONTHS.index(A), MONTHS.index(B)
        s_cal = ROUTE_DIR[f"{A}->{B}"]     # calendar direction of the walked arc
        contrast_vec = Mm[iB] - Mm[iA]
        qB = ring_point(cx, cy, r, TH[B])

        def directions(Pmat):
            """Raw (unrescaled) direction per example for each arm. The random arms are
            reseeded per pair [R5]."""
            out = {}
            out["manifold_chord"] = np.stack([(qB - Pmat[i]) @ V for i in range(len(Pmat))])
            out["contrastive"] = np.repeat(contrast_vec[None, :], len(Pmat), axis=0)
            c_in = (contrast_vec @ V.T) @ V            # component inside the ring plane
            c_or = contrast_vec - c_in                 # component orthogonal to it
            out["contrastive_inplane"] = np.repeat(c_in[None, :], len(Pmat), axis=0)
            out["contrastive_orth"] = np.repeat(c_or[None, :], len(Pmat), axis=0)
            for s in RANDOM_SEEDS:
                g = np.random.default_rng(1000 * s + seedbase).normal(size=(len(Pmat), 1152))
                g = g - (g @ V.T) @ V
                out[f"random_orth_s{s}"] = g
            return out

        for site in SITES:
            site_eps = SITE_EPS[site] if SITE_EPS[site] is not None else eps_grid
            eps_rows = [("eps", e) for e in site_eps]
            if site == "month":
                if INCLUDE_NATURAL:
                    eps_rows.append(("natural", None))
                if INCLUDE_CHORD_MATCHED:
                    eps_rows.append(("chord", None))

            idx_b = torch.arange(nb, device=DEVICE)
            idx_p = torch.tensor(site_positions(rows, site), device=DEVICE)
            oidx_b = torch.arange(len(off_rows), device=DEVICE)
            oidx_p = torch.tensor(site_positions(off_rows, site), device=DEVICE)
            P = (c["hpos"][site] - mu) @ V.T                  # (nb, 2) in-plane coords
            hn = np.linalg.norm(c["hpos"][site], axis=1)      # (nb,)
            Poff = (hoff[site] - mu) @ V.T
            hnoff = np.linalg.norm(hoff[site], axis=1)
            # the primary arm's OWN natural norm, per prompt: the in-plane chord to B.
            chord_on = np.array([float(np.linalg.norm(qB - P[i])) for i in range(nb)])
            chord_off = np.array([float(np.linalg.norm(qB - Poff[i])) for i in range(len(off_rows))])

            raw_on = directions(P)
            raw_off = directions(Poff)

            for kind, eps in eps_rows:
                row_arm = {}
                for arm in ARM_NAMES:
                    # ---- the matched norm for this row, IDENTICAL across arms
                    if kind == "eps":
                        tgtn, tgtno = eps * hn, eps * hnoff
                    elif kind == "chord":
                        tgtn, tgtno = chord_on.copy(), chord_off.copy()     # [R4]
                    else:
                        tgtn = tgtno = None                                 # per-arm below

                    # ---- build the on-target delta
                    if arm == "manifold_arc":
                        dl, tvals, flags = [], [], []
                        for i in range(nb):
                            tgt = float(tgtn[i]) if tgtn is not None else \
                                  float(np.linalg.norm(qB - P[i]))
                            dd, tv, fl = arc_delta(P[i], cx, cy, r, TH[A], TH[B], tgt)
                            dl.append(dd @ V); tvals.append(tv); flags.append(fl)
                        Dr = np.stack(dl)
                    else:
                        Dr = raw_on[arm]
                        tvals, flags = [], []
                    rawn = np.linalg.norm(Dr, axis=1)
                    if tgtn is None:                              # the natural row
                        tn = rawn if not arm.startswith("random_orth") else \
                             np.full(nb, contrast_natural[f"{A}->{B}"])
                    else:
                        tn = tgtn
                    Don = Dr / np.maximum(rawn[:, None], 1e-12) * tn[:, None]

                    # ---- the same rule, off target
                    if arm == "manifold_arc":
                        dl = []
                        for i in range(len(off_rows)):
                            tgt = float(tgtno[i]) if tgtno is not None else \
                                  float(np.linalg.norm(qB - Poff[i]))
                            dd, _, _ = arc_delta(Poff[i], cx, cy, r, TH[A], TH[B], tgt)
                            dl.append(dd @ V)
                        Dro = np.stack(dl)
                    else:
                        Dro = raw_off[arm]
                    rawno = np.linalg.norm(Dro, axis=1)
                    if tgtno is None:
                        tno = rawno if not arm.startswith("random_orth") else \
                              np.full(len(off_rows), contrast_natural[f"{A}->{B}"])
                    else:
                        tno = tgtno
                    Doff = Dro / np.maximum(rawno[:, None], 1e-12) * tno[:, None]

                    lg = fwd(c["ids"], c["attn"], idx_b, idx_p,
                             torch.tensor(Don, dtype=torch.float32, device=DEVICE))
                    st_on = readout(lg, rows, month_ids, len(bridge_ids), scored_ids)
                    del lg
                    lg = fwd(off_ids, off_attn, oidx_b, oidx_p,
                             torch.tensor(Doff, dtype=torch.float32, device=DEVICE))
                    st_off = readout(lg, off_rows, month_ids, len(bridge_ids), scored_ids,
                                     keep_lp=True)
                    kls = kl_rows(st_off, off_base)
                    del lg
                    n_cfg += 2

                    lpB = np.array([st_on[i]["month_lp"][iB] for i in keep])
                    lpA = np.array([st_on[i]["month_lp"][iA] for i in keep])
                    bB = np.array([c["base"][i]["month_lp"][iB] for i in keep])
                    bA = np.array([c["base"][i]["month_lp"][iA] for i in keep])
                    nmid = np.array([st_on[i]["nll_mid"] for i in keep])
                    bmid = np.array([c["base"][i]["nll_mid"] for i in keep])
                    ncont = np.array([st_on[i]["nll_cont"] for i in keep])
                    bcont = np.array([c["base"][i]["nll_cont"] for i in keep])
                    mlp = np.stack([st_on[i]["month_lp"] for i in keep]).mean(0)
                    am = [MONTHS[int(np.argmax(st_on[i]["month_lp"]))] for i in keep]
                    mass = np.array([float(np.exp(st_on[i]["month_lp"]).sum()) for i in keep])

                    over = float(np.median(tn[keep] / np.maximum(chord_on[keep], 1e-9)))

                    rec = dict(pair=f"{A}->{B}", delta_months=int((iB - iA) % 12),
                               site=site, arm=arm,
                               eps=(eps if kind == "eps" else None), norm_kind=kind,
                               n=len(keep),
                               target_norm_med=float(np.median(tn[keep])),
                               raw_norm_med=float(np.median(rawn[keep])),
                               chord_natural_norm_med=float(np.median(chord_on[keep])),
                               target_over_hnorm=float(np.median(tn[keep] / hn[keep])),
                               overshoot_vs_chord=over,
                               rot=float(np.mean(lpB - lpA)),
                               rot_base=float(np.mean(bB - bA)),
                               rot_gain=float(np.mean((lpB - lpA) - (bB - bA))),
                               lpB=float(lpB.mean()), lpA=float(lpA.mean()),
                               lpB_base=float(bB.mean()), lpA_base=float(bA.mean()),
                               frac_argmax_B=float(np.mean([x == B for x in am])),
                               frac_argmax_A=float(np.mean([x == A for x in am])),
                               month_mass=float(mass.mean()),
                               nll_mid=float(np.mean(nmid)), nll_mid_base=float(np.mean(bmid)),
                               cost_mid=float(np.mean(nmid - bmid)),
                               nll_cont=float(np.mean(ncont)),
                               cost_cont=float(np.mean(ncont - bcont)),
                               kl_offtarget=float(np.mean(kls)),
                               kl_offtarget_med=float(np.median(kls)),
                               month_lp_mean=[round(float(v), 5) for v in mlp],
                               ring=ring_locality(mlp, iA, iB),
                               route=route_stats(mlp, iA, iB, s_cal),
                               ring_base=ring_locality(
                                   np.stack([c["base"][i]["month_lp"] for i in keep]).mean(0),
                                   iA, iB),
                               route_base=route_stats(
                                   np.stack([c["base"][i]["month_lp"] for i in keep]).mean(0),
                                   iA, iB, s_cal))
                    if arm == "manifold_arc":
                        kt = [tvals[i] for i in keep]
                        klo = sum(flags[i] == "low" for i in keep)
                        khi = sum(flags[i] == "high" for i in keep)
                        rec["arc_t_med"] = float(np.median(kt))
                        rec["arc_clamp_low"] = int(klo)
                        rec["arc_clamp_high"] = int(khi)
                        rec["arc_frac_unclamped"] = float((len(keep) - klo - khi) / max(len(keep), 1))
                    agg.append(rec)
                    row_arm[arm] = rec
                    e_sent = eps if kind == "eps" else EPS_SENTINEL[kind]
                    for j, i in enumerate(keep):
                        inst.append([f"{A}->{B}", site, arm, e_sent, kind, int(i),
                                     round(float(lpA[j]), 5), round(float(lpB[j]), 5),
                                     round(float(nmid[j]), 5), round(float(ncont[j]), 5)])
                e = eps if kind == "eps" else kind
                p = row_arm[PRIMARY_ARM]; q = row_arm[BASELINE_ARM]
                ar = row_arm["manifold_arc"]
                ci, co = row_arm["contrastive_inplane"], row_arm["contrastive_orth"]
                print(f"  [{site[:3]}] {A[:3]}->{B[:3]} e={str(e):>7} "
                      f"|d|={p['target_norm_med']:6.0f} over={p['overshoot_vs_chord']:5.2f} "
                      f"unclamp={ar.get('arc_frac_unclamped', 0):.2f} | gain chord "
                      f"{p['rot_gain']:+6.2f} arc {ar['rot_gain']:+6.2f} contr "
                      f"{q['rot_gain']:+6.2f} in {ci['rot_gain']:+6.2f} orth "
                      f"{co['rot_gain']:+6.2f} | argB {p['frac_argmax_B']:.2f}/"
                      f"{q['frac_argmax_B']:.2f} | kl {p['kl_offtarget']:5.2f}/"
                      f"{q['kl_offtarget']:5.2f}", flush=True)
    print(f"sweep: {n_cfg} steered forwards in {time.time()-t_start:.0f}s", flush=True)

    # [R6] causality check: an intervention at the answer position is downstream of the
    # in-context tokens it would have to damage, so cost_mid there must be exactly zero.
    ans_cost_mid = [x["cost_mid"] for x in agg if x["site"] == "answer"]
    site_causality = dict(
        max_abs_cost_mid_at_answer_site=float(np.max(np.abs(ans_cost_mid))) if ans_cost_mid else None,
        note="an intervention at the final prompt token cannot affect logits at earlier "
             "positions under causal masking, so cost_mid must be identically 0 at the "
             "answer site. It is therefore a free correctness check there, not a cost; "
             "the answer-site readouts are rotation, cost_cont and off-target KL.")
    print(f"answer-site cost_mid max|.| = {site_causality['max_abs_cost_mid_at_answer_site']}",
          flush=True)

    # ---- 4. generation, per instance, all pairs [R3] ----------------------
    # The spec's coherence metric: perplexity of the STEERED generation scored under the
    # UNSTEERED model. Revision 1 had this on three pairs with no CI and scored the
    # steered month itself; revision 2 restricts to qualified prompts, stores per
    # instance, and reports both the all-token and the first-token-excluded variant.
    gen, gen_inst = [], []
    gen_inst_schema = ["pair", "arm", "eps", "prompt_idx", "nll_all", "nll_ex1"]
    if not a.skip_gen and not a.smoke:
        t0 = time.time()

        def greedy(A, B, arm, eps):
            c = on_cache[A]
            rows = c["rows"]; nb = len(rows)
            Lp = max(rr["plen"] for rr in rows)
            ids = torch.zeros(nb, Lp + GEN_TOKENS, dtype=torch.long, device=DEVICE)
            attn = torch.zeros(nb, Lp + GEN_TOKENS, dtype=torch.long, device=DEVICE)
            cur = []
            for i, rr in enumerate(rows):
                ids[i, :rr["plen"]] = torch.tensor(rr["prompt_ids"], device=DEVICE)
                attn[i, :rr["plen"]] = 1
                cur.append(rr["plen"])
            cur = np.array(cur)
            ib = torch.arange(nb, device=DEVICE)
            ip = torch.tensor(site_positions(rows, "month"), device=DEVICE)
            D = None
            if arm is not None:
                P = (c["hpos"]["month"] - mu) @ V.T
                hn = np.linalg.norm(c["hpos"]["month"], axis=1)
                qB = ring_point(cx, cy, r, TH[B])
                if arm == "manifold_arc":
                    D = np.stack([arc_delta(P[i], cx, cy, r, TH[A], TH[B], eps * hn[i])[0] @ V
                                  for i in range(nb)])
                elif arm == "manifold_chord":
                    D = np.stack([(qB - P[i]) @ V for i in range(nb)])
                elif arm in ("contrastive", "contrastive_inplane", "contrastive_orth"):
                    cv = Mm[MONTHS.index(B)] - Mm[MONTHS.index(A)]
                    if arm == "contrastive_inplane":
                        cv = (cv @ V.T) @ V
                    elif arm == "contrastive_orth":
                        cv = cv - (cv @ V.T) @ V
                    D = np.repeat(cv[None, :], nb, 0)
                else:
                    s = int(arm.rsplit("_s", 1)[1])
                    sb = RANDOM_SEEDBASE(pairs.index((A, B)))
                    g = np.random.default_rng(1000 * s + sb).normal(size=(nb, 1152))
                    D = g - (g @ V.T) @ V
                D = D / np.maximum(np.linalg.norm(D, axis=1, keepdims=True), 1e-12) * (eps * hn)[:, None]
                D = torch.tensor(D, dtype=torch.float32, device=DEVICE)
            for _ in range(GEN_TOKENS):
                lg = fwd(ids, attn, ib, ip, D) if D is not None else fwd(ids, attn)
                nxt = lg[torch.arange(nb), torch.tensor(cur - 1, device=DEVICE)].argmax(-1)
                ids[torch.arange(nb), torch.tensor(cur, device=DEVICE)] = nxt
                attn[torch.arange(nb), torch.tensor(cur, device=DEVICE)] = 1
                cur = cur + 1
                del lg
            # score the generated tokens under the UNSTEERED model. The log_softmax is
            # taken on a per-example slice: the full (nb, seq, 262144) float32 tensor is
            # a ~2.6 GB spike on a machine shared with sibling agents.
            lg = fwd(ids, attn)
            out = {}
            for i, rr in enumerate(rows):
                s0 = rr["plen"]
                idx = list(range(s0 - 1, s0 - 1 + GEN_TOKENS))
                lp = torch.log_softmax(lg[i, idx].float(), -1)
                tg = ids[i, s0:s0 + GEN_TOKENS]
                per = -lp[torch.arange(GEN_TOKENS, device=DEVICE), tg]
                out[i] = dict(nll_all=float(per.mean()), nll_ex1=float(per[1:].mean()),
                              text=tok.decode(tg))
                del lp
            del lg
            return out

        # unsteered baselines, cached per source month (they do not depend on B)
        gen_base = {}
        for A in sorted({p[0] for p in pairs}):
            gen_base[A] = greedy(A, A, None, 0.0)
            k = on_cache[A]["keep"]
            print(f"  gen base {A[:3]:<4} ppl_all "
                  f"{np.exp(np.mean([gen_base[A][i]['nll_all'] for i in k])):6.3f} ppl_ex1 "
                  f"{np.exp(np.mean([gen_base[A][i]['nll_ex1'] for i in k])):6.3f}", flush=True)
            gen.append(dict(pair=None, source=A, arm="unsteered", eps=None,
                            n=len(k),
                            nll_all=float(np.mean([gen_base[A][i]["nll_all"] for i in k])),
                            nll_ex1=float(np.mean([gen_base[A][i]["nll_ex1"] for i in k])),
                            ppl_all=float(np.exp(np.mean([gen_base[A][i]["nll_all"] for i in k]))),
                            ppl_ex1=float(np.exp(np.mean([gen_base[A][i]["nll_ex1"] for i in k]))),
                            samples=[gen_base[A][i]["text"] for i in k[:3]]))

        gen_plan = [((A, B), e) for A, B in pairs for e in GEN_EPS_ALL] + \
                   [((A, B), e) for A, B in GEN_PAIRS_EXTRA for e in GEN_EPS_EXTRA]
        for (A, B), eps in gen_plan:
            k = on_cache[A]["keep"]
            for arm in GEN_ARMS:
                o = greedy(A, B, arm, eps)
                na = np.array([o[i]["nll_all"] for i in k])
                nx = np.array([o[i]["nll_ex1"] for i in k])
                ba = np.array([gen_base[A][i]["nll_all"] for i in k])
                bx = np.array([gen_base[A][i]["nll_ex1"] for i in k])
                gen.append(dict(pair=f"{A}->{B}", source=A, arm=arm, eps=eps, n=len(k),
                                nll_all=float(na.mean()), nll_ex1=float(nx.mean()),
                                ppl_all=float(np.exp(na.mean())), ppl_ex1=float(np.exp(nx.mean())),
                                ppl_all_base=float(np.exp(ba.mean())),
                                ppl_ex1_base=float(np.exp(bx.mean())),
                                d_ppl_all=float(np.exp(na.mean()) - np.exp(ba.mean())),
                                d_ppl_ex1=float(np.exp(nx.mean()) - np.exp(bx.mean())),
                                cost_gen_all=float((na - ba).mean()),
                                cost_gen=float((nx - bx).mean()),
                                samples=[o[i]["text"] for i in k[:3]]))
                for i in k:
                    gen_inst.append([f"{A}->{B}", arm, eps, int(i),
                                     round(o[i]["nll_all"], 5), round(o[i]["nll_ex1"], 5)])
                print(f"  gen {A[:3]}->{B[:3]} eps={eps:<6} {arm:<20} ppl_ex1 "
                      f"{np.exp(nx.mean()):7.3f} (unst {np.exp(bx.mean()):.3f}) "
                      f"d_nll {float((nx-bx).mean()):+.4f}", flush=True)
        print(f"generation block: {time.time()-t0:.0f}s", flush=True)

    # ---- 5. verdict ------------------------------------------------------
    # instance keying now carries the site and the norm kind, so the natural and
    # chord rows cannot collide (revision 1 keyed both to eps=-1.0).
    I = {}
    for pair, site, arm, e, kind, pidx, lpA, lpB, nm, nc in inst:
        I.setdefault((site, round(e, 4), arm), {})[(pair, pidx)] = (lpB - lpA, nm, nc)
    G = {}
    for pair, arm, e, pidx, na, nx in gen_inst:
        G.setdefault((round(e, 4), arm), {})[(pair, pidx)] = (na, nx)
    def paired(e, arm1, arm2, site="month", col=1):
        """col: 1 = cost_mid, 2 = cost_cont. Returns (d_rotation, d_cost, n)."""
        d1 = I.get((site, round(e, 4), arm1), {}); d2 = I.get((site, round(e, 4), arm2), {})
        ks = sorted(set(d1) & set(d2))
        return (np.array([d1[k][0] - d2[k][0] for k in ks]),
                np.array([d1[k][col] - d2[k][col] for k in ks]), len(ks), ks)

    def paired_gen(e, arm1, arm2):
        """The spec's cost metric, first generated token excluded, paired per instance,
        with the rotation restricted to exactly the same instances."""
        g1 = G.get((round(e, 4), arm1), {}); g2 = G.get((round(e, 4), arm2), {})
        d1 = I.get(("month", round(e, 4), arm1), {}); d2 = I.get(("month", round(e, 4), arm2), {})
        ks = sorted(set(g1) & set(g2) & set(d1) & set(d2))
        return (np.array([d1[k][0] - d2[k][0] for k in ks]),
                np.array([g1[k][1] - g2[k][1] for k in ks]), len(ks), ks)

    dr, dc, nk, _ = paired(EPS_HEADLINE, PRIMARY_ARM, BASELINE_ARM)
    m_r, lo_r, hi_r = boot_ci(dr)
    m_c, lo_c, hi_c = boot_ci(dc)
    verdict = label_from(lo_r, hi_r, lo_c, hi_c)
    detail = verdict_detail(lo_r, hi_r, lo_c, hi_c)

    # [R3] the SAME rule under each cost metric. Reported whole, unconditionally.
    cost_sensitivity = {}
    for name, desc in COST_METRICS:
        if name == "cost_mid":
            rr, cc, nn, _ = paired(EPS_HEADLINE, PRIMARY_ARM, BASELINE_ARM, col=1)
        elif name == "cost_cont":
            rr, cc, nn, _ = paired(EPS_HEADLINE, PRIMARY_ARM, BASELINE_ARM, col=2)
        else:
            rr, cc, nn, _ = paired_gen(EPS_HEADLINE, PRIMARY_ARM, BASELINE_ARM)
        if nn < 2:
            cost_sensitivity[name] = dict(description=desc, n_instances=nn,
                                          label=None, note="not computed")
            continue
        a_r, a_lo, a_hi = boot_ci(rr)
        b_m, b_lo, b_hi = boot_ci(cc)
        cost_sensitivity[name] = dict(
            description=desc, n_instances=nn,
            label=label_from(a_lo, a_hi, b_lo, b_hi),
            detail=verdict_detail(a_lo, a_hi, b_lo, b_hi),
            rotation_delta=dict(mean=a_r, ci_lo=a_lo, ci_hi=a_hi),
            cost_delta=dict(mean=b_m, ci_lo=b_lo, ci_hi=b_hi))

    # [R3] the same three metrics across the whole eps grid, for the primary comparison.
    cost_sensitivity_by_eps = {}
    for e in eps_grid:
        row = {}
        for name in ("cost_mid", "cost_cont", "cost_gen"):
            if name == "cost_gen":
                rr, cc, nn, _ = paired_gen(e, PRIMARY_ARM, BASELINE_ARM)
            else:
                rr, cc, nn, _ = paired(e, PRIMARY_ARM, BASELINE_ARM,
                                       col=(1 if name == "cost_mid" else 2))
            if nn < 2:
                row[name] = None
                continue
            a_r, a_lo, a_hi = boot_ci(rr); b_m, b_lo, b_hi = boot_ci(cc)
            row[name] = dict(n=nn, label=label_from(a_lo, a_hi, b_lo, b_hi),
                             cost_mean=b_m, cost_ci_lo=b_lo, cost_ci_hi=b_hi,
                             rot_mean=a_r, rot_ci_lo=a_lo, rot_ci_hi=a_hi)
        cost_sensitivity_by_eps[str(e)] = row

    per_eps = {}
    for e in eps_grid:
        r1, c1, n1, _ = paired(e, PRIMARY_ARM, BASELINE_ARM)
        r2, c2, _, _ = paired(e, "manifold_arc", BASELINE_ARM)
        r3, c3, n3, _ = paired(e, PRIMARY_ARM, "contrastive_inplane")
        rm, rlo, rhi = boot_ci(r1); cm, clo, chi = boot_ci(c1)
        rm2, rlo2, rhi2 = boot_ci(r2); cm2, clo2, chi2 = boot_ci(c2)
        rm3, rlo3, rhi3 = boot_ci(r3)
        per_eps[str(e)] = dict(
            n_instances=n1,
            chord_minus_contrastive_rotation=dict(mean=rm, ci_lo=rlo, ci_hi=rhi),
            chord_minus_contrastive_cost=dict(mean=cm, ci_lo=clo, ci_hi=chi),
            arc_minus_contrastive_rotation=dict(mean=rm2, ci_lo=rlo2, ci_hi=rhi2),
            arc_minus_contrastive_cost=dict(mean=cm2, ci_lo=clo2, ci_hi=chi2),
            chord_minus_contrastive_inplane_rotation=dict(mean=rm3, ci_lo=rlo3, ci_hi=rhi3))

    # chord-matched row: the primary arm at its OWN natural scale, everything else
    # rescaled down to it. [R4]
    chord_row = {}
    if INCLUDE_CHORD_MATCHED:
        rr, cc, nn, _ = paired(EPS_SENTINEL["chord"], PRIMARY_ARM, BASELINE_ARM)
        if nn >= 2:
            a_r, a_lo, a_hi = boot_ci(rr); b_m, b_lo, b_hi = boot_ci(cc)
            chord_row = dict(n_instances=nn, label=label_from(a_lo, a_hi, b_lo, b_hi),
                             detail=verdict_detail(a_lo, a_hi, b_lo, b_hi),
                             rotation_delta=dict(mean=a_r, ci_lo=a_lo, ci_hi=a_hi),
                             cost_delta=dict(mean=b_m, ci_lo=b_lo, ci_hi=b_hi),
                             note="matched norm = the primary arm's own per-prompt "
                                  "in-plane chord to B, so overshoot_vs_chord is 1.00 "
                                  "for manifold_chord by construction and the "
                                  "contrastive vector is scaled DOWN to it")

    # [R6] the headline comparison at the second site.
    site_rows = {}
    for site in SITES:
        rr, cc, nn, _ = paired(EPS_HEADLINE, PRIMARY_ARM, BASELINE_ARM, site=site)
        _, cc2, _, _ = paired(EPS_HEADLINE, PRIMARY_ARM, BASELINE_ARM, site=site, col=2)
        if nn < 2:
            continue
        a_r, a_lo, a_hi = boot_ci(rr); b_m, b_lo, b_hi = boot_ci(cc)
        d_m, d_lo, d_hi = boot_ci(cc2)
        site_eps_list = SITE_EPS[site] if SITE_EPS[site] is not None else eps_grid
        by_arm_by_eps = {}
        for e in site_eps_list:
            rows_s = [x for x in agg if x["site"] == site and x["eps"] == e]
            if not rows_s:
                continue
            by_arm_by_eps[str(e)] = {arm: dict(
                target_norm_med=float(np.mean([x["target_norm_med"] for x in rows_s if x["arm"] == arm])),
                rot_gain=float(np.mean([x["rot_gain"] for x in rows_s if x["arm"] == arm])),
                frac_argmax_B=float(np.mean([x["frac_argmax_B"] for x in rows_s if x["arm"] == arm])),
                month_mass=float(np.mean([x["month_mass"] for x in rows_s if x["arm"] == arm])),
                kl_offtarget=float(np.mean([x["kl_offtarget"] for x in rows_s if x["arm"] == arm])),
                cost_cont=float(np.mean([x["cost_cont"] for x in rows_s if x["arm"] == arm])))
                for arm in ARM_NAMES}
        site_rows[site] = dict(
            n_instances=nn,
            label_cost_mid=(label_from(a_lo, a_hi, b_lo, b_hi) if site == "month" else None),
            label_cost_cont=label_from(a_lo, a_hi, d_lo, d_hi),
            rotation_delta=dict(mean=a_r, ci_lo=a_lo, ci_hi=a_hi),
            cost_mid_delta=dict(mean=b_m, ci_lo=b_lo, ci_hi=b_hi),
            cost_cont_delta=dict(mean=d_m, ci_lo=d_lo, ci_hi=d_hi),
            by_arm=by_arm_by_eps.get(str(EPS_HEADLINE), {}),
            by_arm_by_eps=by_arm_by_eps)
    site_rows["note"] = ("'month' intervenes at the source-month token, where mean_B-mean_A "
                         "partly carries B's token identity; 'answer' intervenes at the "
                         "final prompt token, where no month token sits. Comparing the two "
                         "measures how much of the contrastive baseline's advantage is "
                         "token substitution rather than steering.")

    # ---- 5b. efficiency, on both the degenerate and the non-degenerate axis
    eff = {}
    for arm in ARM_NAMES:
        rows_e = [x for x in agg if x["arm"] == arm and x["eps"] == EPS_HEADLINE
                  and x["site"] == "month"]
        num = np.array([x["rot_gain"] for x in rows_e])
        den = np.array([x["cost_mid"] for x in rows_e])
        ok = den > 0.05
        gen_e = [x for x in gen if x.get("arm") == arm and x.get("eps") == EPS_HEADLINE]
        gnum = np.array([next(y["rot_gain"] for y in rows_e if y["pair"] == x["pair"])
                         for x in gen_e]) if gen_e else np.array([])
        gden = np.array([x["cost_gen"] for x in gen_e]) if gen_e else np.array([])
        gok = gden > 0.05
        # A per-pair median ratio is thrown off by the >0.05-nat filter, which keeps a
        # different subset of pairs for each arm (contrastive is cheap on six of nine, so
        # its median would be taken over the three where it happens to be expensive). The
        # RATIO OF MEANS uses all nine pairs for every arm and is the number that answers
        # the pre-registered per-unit-cost question; the filtered median is kept beside
        # it rather than replaced.
        eff[arm] = dict(n_pairs=len(rows_e), n_usable_cost_mid=int(ok.sum()),
                        rot_gain_mean=float(num.mean()), cost_mid_mean=float(den.mean()),
                        ratio_med_cost_mid=float(np.median(num[ok] / den[ok])) if ok.any() else None,
                        n_pairs_gen=len(gen_e), n_usable_cost_gen=int(gok.sum()),
                        cost_gen_mean=float(gden.mean()) if gen_e else None,
                        d_ppl_ex1_mean=float(np.mean([x["d_ppl_ex1"] for x in gen_e])) if gen_e else None,
                        ratio_med_cost_gen=float(np.median(gnum[gok] / gden[gok])) if gok.any() else None,
                        ratio_of_means_cost_gen=(float(gnum.mean() / gden.mean())
                                                 if gen_e and gden.mean() > 1e-9 else None),
                        kl_mean=float(np.mean([x["kl_offtarget"] for x in rows_e])))

    # [R3] why the pre-registered cost clause is close to vacuous, stated numerically.
    hm = [x for x in agg if x["eps"] == EPS_HEADLINE and x["site"] == "month"]
    by_arm_mid = {arm: float(np.mean([x["cost_mid"] for x in hm if x["arm"] == arm]))
                  for arm in ARM_NAMES}
    cost_degeneracy = dict(
        cost_mid_by_arm_at_headline=by_arm_mid,
        max_abs_cost_mid=float(max(abs(v) for v in by_arm_mid.values())),
        n_arms_with_negative_cost_mid=int(sum(v < 0 for v in by_arm_mid.values())),
        most_improved_arm=min(by_arm_mid, key=by_arm_mid.get),
        note="the pre-registered cost axis is in-context NLL. At the headline norm every "
             "arm sits within a few hundredths of a nat of zero and most are NEGATIVE, "
             "with a NULL direction improving it most, so the verdict rule's cost clause "
             "('CI not entirely < 0') is close to vacuous and the label is effectively "
             "decided by the rotation clause alone. That is why cost_metric_sensitivity "
             "reports the same rule under two further coherence metrics, and why "
             "efficiency is also computed on the generation-perplexity denominator, "
             "which is the only one of the three with usable signal.")

    # ---- 5c. mechanism, plane share, traversal ---------------------------
    mech = {}
    for arm in ARM_NAMES:
        rows_m = [x for x in agg if x["arm"] == arm and x["eps"] == EPS_HEADLINE
                  and x["site"] == "month"]
        if not rows_m:
            continue
        gen_m = [x for x in gen if x.get("arm") == arm and x.get("eps") == EPS_HEADLINE]
        mech[arm] = dict(
            rot_gain_mean=float(np.mean([x["rot_gain"] for x in rows_m])),
            frac_argmax_B=float(np.mean([x["frac_argmax_B"] for x in rows_m])),
            kl_offtarget=float(np.mean([x["kl_offtarget"] for x in rows_m])),
            cost_mid=float(np.mean([x["cost_mid"] for x in rows_m])),
            cost_cont=float(np.mean([x["cost_cont"] for x in rows_m])),
            cost_gen=float(np.mean([x["cost_gen"] for x in gen_m])) if gen_m else None,
            d_ppl_ex1=float(np.mean([x["d_ppl_ex1"] for x in gen_m])) if gen_m else None,
            circ_spread_deg=float(np.mean([x["ring"]["circ_spread_deg"] for x in rows_m])),
            dist_mean_to_B_deg=float(np.mean([x["ring"]["dist_mean_to_B_deg"] for x in rows_m])),
            neighbour_lift=float(np.mean([x["ring"]["neighbour_lift"] for x in rows_m])),
            route_intermediate_mass=float(np.mean(
                [x["route"]["route_intermediate_mass"] for x in rows_m
                 if x["route"]["route_intermediate_mass"] is not None])),
            raw_norm_med=float(np.mean([x["raw_norm_med"] for x in rows_m])))
    plane_share = dict(
        contrastive_norm_med=float(np.median(list(contrast_natural.values()))),
        inplane_share_of_contrastive={
            f"{A}->{B}": float(np.linalg.norm(((Mm[MONTHS.index(B)] - Mm[MONTHS.index(A)]) @ V.T) @ V)
                               / np.linalg.norm(Mm[MONTHS.index(B)] - Mm[MONTHS.index(A)]))
            for A, B in pairs},
        ring_diameter=float(2 * r),
        note="the ring's whole diameter is smaller than the median contrastive vector, so "
             "matching norms to the contrastive scale forces every arm off the manifold")

    # [R1/R2] traversal for EVERY arm, with clamping surfaced.
    traversal = {}
    for pr in sorted({x["pair"] for x in agg}):
        traversal[pr] = {}
        for arm in ARM_NAMES:
            seq = []
            for e in eps_grid:
                row = next((x for x in agg if x["pair"] == pr and x["arm"] == arm
                            and x["eps"] == e and x["site"] == "month"), None)
                if row is None:
                    continue
                seq.append(dict(eps=e, target_norm=row["target_norm_med"],
                                overshoot_vs_chord=row["overshoot_vs_chord"],
                                arc_t=row.get("arc_t_med"),
                                arc_frac_unclamped=row.get("arc_frac_unclamped"),
                                n=row["n"],
                                circ_mean_month=row["ring"]["circ_mean_month"],
                                peak_month=row["ring"]["peak_month"],
                                peak_on_route=row["route"]["peak_on_route"],
                                peak_moved_and_on_route=row["route"]["peak_moved_and_on_route"],
                                route_months=row["route"]["route_months"],
                                signed_progress_months=row["route"]["signed_progress_months"],
                                route_intermediate_mass=row["route"]["route_intermediate_mass"],
                                dist_to_A=row["ring"]["dist_mean_to_A_deg"],
                                dist_to_B=row["ring"]["dist_mean_to_B_deg"],
                                month_lp_mean=row["month_lp_mean"]))
            traversal[pr][arm] = seq

    # per-arm continuity summary, so "which arms walk the calendar" is answered from the
    # data for all eight arms rather than asserted for one.
    # The summary is restricted to eps <= EPS_HEADLINE. Above the natural scale every
    # arm, including the null directions, flattens the month distribution, which puts
    # mass on the intermediate months by DILUTION rather than by traversal; taking a max
    # over the whole grid would let random_orth score like a walk. Declared restriction.
    CONT_EPS_MAX = EPS_HEADLINE
    continuity = {}
    for arm in ARM_NAMES:
        mono, npr, peakroute, peakmoved, imass, prog = 0, 0, [], [], [], []
        for pr, per_arm in traversal.items():
            seq = [s for s in per_arm[arm] if s["eps"] <= CONT_EPS_MAX]
            if len(seq) < 3:
                continue
            npr += 1
            p = [s["signed_progress_months"] for s in seq]
            mono += int(all(p[i + 1] >= p[i] - 1e-9 for i in range(len(p) - 1)))
            peakroute.append(float(np.mean([s["peak_on_route"] for s in seq])))
            peakmoved.append(float(np.mean([s["peak_moved_and_on_route"] for s in seq])))
            prog.append(p[-1])
            im = [s["route_intermediate_mass"] for s in seq
                  if s["route_intermediate_mass"] is not None]
            if im:
                imass.append(float(np.max(im)))
        continuity[arm] = dict(
            n_pairs=npr, eps_max=CONT_EPS_MAX, n_monotone_signed_progress=mono,
            frac_peak_on_route=float(np.mean(peakroute)) if peakroute else None,
            frac_peak_moved_and_on_route=float(np.mean(peakmoved)) if peakmoved else None,
            max_route_intermediate_mass_mean=float(np.mean(imass)) if imass else None,
            final_signed_progress_mean=float(np.mean(prog)) if prog else None)
    continuity["_definitions"] = dict(
        eps_restriction=f"all fields are computed over eps <= {CONT_EPS_MAX}; above the "
                        "natural scale every arm flattens the month distribution and "
                        "intermediate mass accrues by dilution, not traversal",
        signed_progress_months="displacement of the 12-month circular mean away from A "
                               "toward B, in months, along the arc the arc arm walks, "
                               "whose calendar direction is taken from the fitted ring "
                               "(manifold.route_direction_by_pair). Monotone across the "
                               "eps grid = a dose response.",
        route_intermediate_mass="probability on months STRICTLY between A and B on that "
                                "route; the walk-versus-substitute discriminator. Only "
                                "defined for pairs at least 2 months apart, so the mean "
                                "is over those five pairs.",
        peak_on_route_caveat="frac_peak_on_route is NOT a discriminator on its own: an "
                             "arm that does not move keeps its peak on A, and A is on "
                             "its own route, so a null direction scores 1.0. Use "
                             "frac_peak_moved_and_on_route.",
        caveat="monotone signed progress is NOT specific to the manifold arms: it is a "
               "property of IN-PLANE directions, and contrastive_inplane has it too, as "
               "does the full contrastive vector. Read route_intermediate_mass on the "
               "genuine_walks cells for what does separate a walk from a substitution.")

    # [R2] the cells where the arc arm genuinely walks (t < 1 for a majority of prompts).
    genuine = []
    for x in agg:
        if x["arm"] != "manifold_arc" or x["site"] != "month" or x["eps"] is None:
            continue
        if x.get("arc_frac_unclamped", 0.0) >= 0.5:
            genuine.append(dict(pair=x["pair"], eps=x["eps"], arc_t_med=x["arc_t_med"],
                                frac_unclamped=x["arc_frac_unclamped"],
                                signed_progress_months=x["route"]["signed_progress_months"],
                                route_intermediate_mass=x["route"]["route_intermediate_mass"],
                                peak_month=x["route"]["peak_month"],
                                route_months=x["route"]["route_months"]))
    # matched comparison: on exactly those cells, what do the other arms do?
    genuine_vs_controls = []
    for g in genuine:
        row = dict(pair=g["pair"], eps=g["eps"], arc_t_med=g["arc_t_med"])
        for arm in ARM_NAMES:
            x = next((y for y in agg if y["pair"] == g["pair"] and y["arm"] == arm
                      and y["eps"] == g["eps"] and y["site"] == "month"), None)
            if x is None:
                continue
            row[arm] = dict(intermediate_mass=x["route"]["route_intermediate_mass"],
                            signed_progress=x["route"]["signed_progress_months"],
                            peak_month=x["route"]["peak_month"],
                            peak_moved_and_on_route=x["route"]["peak_moved_and_on_route"])
        genuine_vs_controls.append(row)
    # the headline of the corrected positive: on the genuine-walk cells that HAVE
    # intermediate months, how much route-intermediate mass does each arm produce?
    gw_summary = {}
    cells_im = [g for g in genuine_vs_controls
                if g.get("manifold_arc", {}).get("intermediate_mass") is not None]
    for arm in ARM_NAMES:
        vals = [g[arm]["intermediate_mass"] for g in cells_im if arm in g]
        if vals:
            gw_summary[arm] = dict(n_cells=len(vals), mean=float(np.mean(vals)),
                                   max=float(np.max(vals)))
    genuine_walks = dict(
        rule="cells where manifold_arc is unclamped for at least half the qualified "
             "prompts (arc_frac_unclamped >= 0.5); elsewhere the arm has degenerated to "
             "a rescaled chord and is not a walk. Declared before the cells were read.",
        n_cells=len(genuine), cells=genuine, vs_controls=genuine_vs_controls,
        intermediate_mass_by_arm=gw_summary,
        intermediate_mass_note="mean and max route_intermediate_mass over the genuine-"
                               "walk cells that have intermediate months (the mid and "
                               "antipodal pairs). Every arm is scored on the SAME cells "
                               "at the SAME matched norm, so this is the walk-versus-"
                               "substitute comparison with the size confound removed.")

    # ---- 6. reproduction check against revision 1 ------------------------
    repro = dict(
        expectation="PRIMARY_ARM, BASELINE_ARM, EPS_HEADLINE, the qualification rule and "
                    "the verdict rule are unchanged from revision 1, and every readout is "
                    "deterministic, so the headline verdict cell must reproduce revision "
                    "1 exactly. random_orth rows are EXPECTED to differ (per-pair seeds).",
        revision1=dict(label="manifold_loses", n_instances=189,
                       rotation_mean=-3.5188492063492065,
                       rotation_ci=[-3.783077101851852, -3.2549604206349207],
                       cost_mean=0.0005548677248677262,
                       cost_ci=[-0.005359616402116416, 0.006812640211640199],
                       hook_identity_max_logit_diff=0.0,
                       gap_ratio=1.5556417541417034, circle_resid=0.10020546069323703),
        revision2=dict(label=verdict, n_instances=nk, rotation_mean=m_r,
                       rotation_ci=[lo_r, hi_r], cost_mean=m_c, cost_ci=[lo_c, hi_c],
                       hook_identity_max_logit_diff=hook_maxdiff,
                       gap_ratio=manifold["gap_ratio"], circle_resid=manifold["circle_resid"]))
    repro["matches"] = bool(
        verdict == "manifold_loses" and nk == 189
        and abs(m_r - (-3.5188492063492065)) < 1e-9
        and abs(manifold["gap_ratio"] - 1.5556417541417034) < 1e-9)

    out = dict(
        declared=dict(
            question="At matched intervention norm, does a move along the layer-9 month "
                     "ring beat a contrastive month-difference vector on identity "
                     "rotation per unit coherence cost?",
            revision=2,
            revision_trigger="adversarial review of revision 1's output; all six findings "
                             "were reproduced from revision 1's own raw rows before any "
                             "change was made",
            revision_notes=REVISION_NOTES,
            frozen_from_revision1=["primary_arm", "baseline_arm", "eps_headline",
                                   "qualification_rule", "verdict_rule", "n_boot",
                                   "boot_seed", "prompt sets", "pair list",
                                   "manifold fit"],
            primary_arm=PRIMARY_ARM, baseline_arm=BASELINE_ARM,
            secondary_arms=["manifold_arc", "contrastive_inplane", "contrastive_orth"]
                           + [f"random_orth_s{s}" for s in RANDOM_SEEDS],
            sequence_disclosure=SEQUENCE_DISCLOSURE,
            norm_matching_rule="delta = eps * ||h_pos||_2 * d_arm/||d_arm||_2, where "
                               "h_pos is the unsteered float32 layer-9 residual at the "
                               "intervened token of that prompt; identical eps across arms",
            eps_grid=eps_grid, eps_grid_revision1=EPS_GRID_R1,
            eps_grid_added_in_revision2=EPS_GRID_ADDED, eps_headline=EPS_HEADLINE,
            natural_rows_included=INCLUDE_NATURAL,
            natural_row_meaning="unrescaled arms; NOT norm matched; reported to show what "
                                "the unguarded comparison would have concluded",
            chord_matched_row_included=INCLUDE_CHORD_MATCHED,
            chord_matched_row_meaning="every arm rescaled to the PRIMARY arm's own "
                                      "per-prompt in-plane chord to B, so the primary arm "
                                      "is at overshoot 1.00 and the baseline is scaled "
                                      "DOWN to the manifold's scale rather than the "
                                      "manifold being pushed up to the baseline's",
            qualification_rule=QUALIFY_RULE,
            verdict_rule="paired per-instance (primary - baseline) on rotation "
                         "logprob(B)-logprob(A) and on cost (mean in-context NLL after the "
                         "intervention); 95% bootstrap CI over instances, 10000 resamples, "
                         "seed 0. DOMINATES: rotation CI>0 and cost CI not >0. LOSES: "
                         "rotation CI<0 and cost CI not <0. TIE otherwise.",
            cost_metrics=[dict(name=n, description=d) for n, d in COST_METRICS],
            cost_metric_policy="the verdict rule is applied under ALL THREE cost metrics "
                               "and all three labels are reported unconditionally; the "
                               "pre-registered one (cost_mid) remains the headline",
            generation_scoring="greedy, 16 tokens, scored under the UNSTEERED model. The "
                               "FIRST generated token is the steered month itself, so "
                               "scoring it charges a successful steer for succeeding; "
                               "cost_gen therefore excludes it (nll_ex1) and the "
                               "all-token variant is stored alongside",
            traversal_discriminator="route_intermediate_mass -- probability on months "
                                    "strictly between A and B along the signed shortest "
                                    "arc. A walk passes through them, a substitution does "
                                    "not. Computed for every arm.",
            genuine_walk_rule="manifold_arc counts as a walk only where it is unclamped "
                              "for >= 50% of qualified prompts",
            sites=SITES, site_eps={k: (v or "full grid") for k, v in SITE_EPS.items()},
            intervention="single layer (9), single token position, residual stream, "
                         "forward hook, float32 arithmetic cast back to bf16",
            random_seeds=RANDOM_SEEDS,
            random_seedbase_rule="7 + 101*pair_index, so the three seeds give 27 "
                                 "independent directions across the nine pairs",
            n_boot=N_BOOT,
            deviations=["readout (ii) is deterministic: coherence cost is the NLL increase "
                        "on real in-context tokens, on a fixed neutral continuation, and "
                        "on a GREEDY (not sampled) generation scored under the unsteered "
                        "model. No sampling anywhere, so no sampling seeds are needed.",
                        "off-target damage is measured on the same grammatical frames with "
                        "a city in the slot, so the only difference is the ontology of the "
                        "intervened token.",
                        "cost_mid is identically zero at the answer site by causal "
                        "masking; it is reported there as a correctness check, and the "
                        "answer-site cost comparison uses cost_cont."]),
        config=dict(layer=L, model="google/gemma-3-1b-pt", dtype="bfloat16", device=DEVICE,
                    n_on_prompts=len(ON_T), n_off_prompts=len(off_rows),
                    pairs=[f"{A}->{B}" for A, B in pairs],
                    arms=ARM_NAMES, sites=SITES,
                    random_seedbase_by_pair=rand_seedbases,
                    bridge=BRIDGE, scored_continuation=SCORED,
                    gen_eps_all_pairs=GEN_EPS_ALL,
                    gen_pairs_extra=[f"{A}->{B}" for A, B in GEN_PAIRS_EXTRA],
                    gen_eps_extra=GEN_EPS_EXTRA, gen_arms=GEN_ARMS,
                    gen_tokens=GEN_TOKENS, gen_score_variant=GEN_SCORE_VARIANT,
                    smoke=a.smoke),
        manifold=manifold,
        plane_diagnostic=plane_diag,
        hook_identity_max_logit_diff=hook_maxdiff,
        site_causality_check=site_causality,
        qualification=qual_report,
        reproduction_check=repro,
        verdict=dict(label=verdict, detail=detail, eps=EPS_HEADLINE, site="month",
                     cost_metric="cost_mid (pre-registered)", n_instances=nk,
                     rotation_delta=dict(mean=m_r, ci_lo=lo_r, ci_hi=hi_r),
                     cost_delta=dict(mean=m_c, ci_lo=lo_c, ci_hi=hi_c)),
        cost_metric_sensitivity=cost_sensitivity,
        cost_metric_sensitivity_by_eps=cost_sensitivity_by_eps,
        cost_degeneracy=cost_degeneracy,
        chord_matched_verdict=chord_row,
        by_site=site_rows,
        per_eps_paired=per_eps,
        efficiency_at_headline=eff,
        mechanism_at_headline=mech,
        plane_share=plane_share,
        traversal=traversal,
        traversal_continuity=continuity,
        genuine_walks=genuine_walks,
        sweep=agg,
        generation=gen,
        generation_instances=dict(schema=gen_inst_schema, rows=gen_inst),
        instances=dict(schema=inst_schema, rows=inst),
        runtime_sec=round(time.time() - t_start, 1))
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(f"\nVERDICT [{verdict}] {detail}")
    print(f"  rotation delta {m_r:+.3f} [{lo_r:+.3f}, {hi_r:+.3f}]  "
          f"cost delta {m_c:+.3f} [{lo_c:+.3f}, {hi_c:+.3f}]  n={nk}")
    print(f"  reproduces revision 1 headline: {repro['matches']}")
    for name, v in cost_sensitivity.items():
        print(f"  under {name:<10} -> {v.get('label')}  ({v.get('detail')})  n={v['n_instances']}")
    if chord_row:
        print(f"  chord-matched row -> {chord_row['label']}  {chord_row['detail']}")
    for arm in ARM_NAMES:
        cn = continuity[arm]
        gs = gw_summary.get(arm, {})
        nn = lambda v: float("nan") if v is None else v
        print(f"  continuity {arm:<20} monotone {cn['n_monotone_signed_progress']}/"
              f"{cn['n_pairs']} peak_moved_on_route "
              f"{nn(cn['frac_peak_moved_and_on_route']):.3f} "
              f"| walk-cell interm_mass mean {nn(gs.get('mean')):.4f} "
              f"max {nn(gs.get('max')):.4f}")
    print(f"  genuine arc-walk cells: {genuine_walks['n_cells']}")
    print(f"  efficiency (rot gain per nat of generation cost, ratio of means): " +
          ", ".join(f"{a}={eff[a]['ratio_of_means_cost_gen']:.1f}" for a in ARM_NAMES
                    if eff[a]['ratio_of_means_cost_gen'] is not None))
    print(f"wrote {a.out}  ({time.time()-t_start:.0f}s)")


if __name__ == "__main__":
    main()
