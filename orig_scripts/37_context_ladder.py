"""Stage 2, avenue 2 -- the month ring's CONTEXT BUDGET.

Pre-registered in docs/phase3_prereg_2026-08-21.md section 3.1 (frozen 2026-08-21); design
and critic repairs in docs/next_avenues_2026-08-21.md avenue 2. Writes
results/phase3_context_ladder.json plus results/phase3_context_ladder_means.npz.

THE QUESTION. Every closure result this project owns is read off a CONTEXT-AVERAGED
object: a per-token-type mean over all corpus occurrences, which is closer to a static
embedding than to a computed feature (scripts/22's own context_averaging_note says so).
That leaves the flagship result -- the L6/L7 month ring, member_hash 7d1d75a303d6 -- with
its construction schedule unmeasured. Is the ring built by layers 0-6 from the token alone,
or assembled from context by attention, and if the latter, how much context does it take?

THE LADDER. For each of the 13 frozen members, forward
``[BOS] + (last c tokens of a cached document) + member_token`` at c in
{0,1,2,4,8,16,32,64,128}, n=300 contexts per (member, c), read resid_post at L6 and L7 at
the member's position, average over contexts, and run the UNCHANGED closure gate per rung.
Holding the context FAMILY fixed (FineWeb documents, none containing any calendar token)
while varying only its LENGTH is what separates budget from genre -- the cache alone cannot
do this, because a token's corpus contexts are not a length-controlled family.

WHY BOTH ARMS. At c=0 the SAE is off-distribution: it never saw a two-token sequence in
training. The prereg therefore scores every rung on BOTH the SAE-reconstructed mean and the
raw-residual mean, and REPORTS the disagreement at c=0 rather than picking a winner. The
SAE arm is primary because the registered anchor (gap_ratio 1.65876, p_closed 0.00125 at
L6) is an SAE-reconstruction number.

WHY THE RIG IS VALIDATED BEFORE ANY RUNG IS READ. The top rung is not just the last data
point, it is the instrument check: c=128 is the closest this rig gets to the cached pooled
readout, so it must reproduce it. If |gap_ratio - 1.659| > 0.2 or p_closed >= 0.01 at
(c=128, L6, SAE arm), then what changed is the forward rig, not the model, and no rung
means anything. The verdict field then reads "rig not validated" -- but every number is
still written out, because a failed instrument is itself a measurement and re-tuning it
post hoc is what prereg rule R2 forbids.

THE INVALIDITY TRIGGER. An ordinal control (leading-space single tokens) runs the identical
pipeline at c in {0,128}. Ordinals are a family this project has no reason to expect a
closed ring from; if they close, the pipeline manufactures rings out of anything with an
ordering and the whole run is VOID. Recorded finding: in this tokenizer " 0".." 9" are NOT
single tokens (they split as [' ', digit]), so the digit half of the registered control
drops out and the control runs on " one".." ten" alone at k=10. The tokenizations of the
dropped members are written to ordinal_control.dropped.

WHAT IS COPIED AND FROM WHERE. Nothing statistical is invented here. gap_null / gap_p /
screen are copied verbatim from scripts/22_token_type_geometry.py (L364-376, L379), as is
member_hash (L561); block_permutations is copied verbatim from
scripts/29_keys_blockperm.py (L108); the angle convention (pca_2d -> fit_circle ->
angles(xy, cx, cy)) is scripts/22's score_ordering. The copies are verbatim rather than
imports because scripts/ files are numbered, not importable modules, and because a frozen
instrument should not be able to change under a script that already reported with it.

ORDERING RUNS ON STRICT-12 ONLY, and the reason is mechanical, not a preference. The
prereg's parenthetical imagined a 13-set ordering with lexical blocks {" Feb"," February"};
but ``cyclic_order_score``'s symmetry basis is built from ``_rank_of((flip*lab + shift) % K)``,
which requires DISTINCT label positions, and " Feb" and " February" both sit at calendar
position 1. A 13-set ordering statistic is therefore ill-defined under the existing
machinery. Closure (screen) is reported for both sets; ordering is reported for strict-12,
where the block-permutation null degenerates to uniform permutations because there are no
lexical blocks left to preserve. That is stated in the JSON rather than hidden.
"""
import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import gfdisc.rig as rig                                     # NEW module, explicit import
from gfdisc.env import DEVICE
from gfdisc.fits import angles, circularity, degenerate, fit_circle, pca_2d, cyclic_order_score
from gfdisc.model import load_model
from gfdisc.saes import load_sae


# ===========================================================================
# frozen constants -- prereg section 0 and section 3.1
# ===========================================================================
#: The 13-member set, ascending token id. INDEX INTO THIS LIST IS ``member_index``
#: (3353 -> 0, 5142 -> 1, ... 7250 -> 12), which is what the context seed formula uses.
MEMBER_IDS = [3353, 5142, 5261, 5365, 5505, 5656, 5729, 5857, 6027, 6198, 6243, 6384, 7250]
FEB_ABBREV_ID = 5656                                # " Feb"; dropped to get strict-12
STRICT12_IDS = [t for t in MEMBER_IDS if t != FEB_ABBREV_ID]
MEMBER_HASH_EXPECTED = "7d1d75a303d6"               # md5[:12] of the sorted POOL INDICES

MONTHS = ["January", "February", "March", "April", "May", "June",
          "July", "August", "September", "October", "November", "December"]
#: token id -> calendar position 0..11. " Feb" shares February's position (see docstring).
CALENDAR_POS = {6027: 0, 7250: 1, 5261: 2, 5365: 3, 3353: 4, 5142: 5,
                5505: 6, 5729: 7, 5857: 8, 6198: 9, 6243: 10, 6384: 11, FEB_ABBREV_ID: 1}
FULL_K = 12

COUNT_FLOOR = 100                                   # scripts/22 build_pool floor
RUNGS = [0, 1, 2, 4, 8, 16, 32, 64, 128]
LAYERS = [6, 7]
N_CTX = 300
N_PERM_SCREEN = 20_000                              # scripts/22 N_PERM_CLUSTER
N_PERM_ORDER = 2_000                                # prereg 3.1 ordering null
N_BLOCK_PERM = 20_000                               # scripts/29 N_NULL

RIG_REF_GAP_RATIO = 1.659                           # prereg 3.1 (exposure record: 1.65876)
RIG_GAP_TOL = 0.2
RIG_MAX_P_CLOSED = 0.01
CERT_LAYER, CERT_RUNG, CERT_ARM = 6, 128, "sae"
CERT_N, CERT_RESAMPLES, CERT_MIN_PASS = 150, 200, 0.8
ORD_RUNGS = [0, 128]
ORD_N = 150
ORD_WORDS = [str(d) for d in range(10)] + [
    "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"]
ALPHA = 0.05

ARMS = ["sae", "raw"]
SETS = {"set13": MEMBER_IDS, "strict12": STRICT12_IDS}

#: seeds. months: 1000*c + member_index (prereg-adjacent, fixed by the task spec).
#: ordinals: 500000 + 1000*c + ordinal_index -- disjoint from the month stream because no
#: rung equals 500 and no member_index reaches 1000.
ORD_SEED_BASE = 500_000


# ===========================================================================
# copied instruments (verbatim; provenance in the docstring)
# ===========================================================================
_NULL = {}


def gap_null(k, n_perm, seed=0):
    """Copied verbatim from scripts/22_token_type_geometry.py L364-373."""
    key = (int(k), int(n_perm), int(seed))
    if key not in _NULL:
        rng = np.random.default_rng(seed)
        a = np.sort(rng.uniform(0, 2 * np.pi, size=(n_perm, k)), axis=1)
        _NULL[key] = np.diff(np.concatenate([a, a[:, :1] + 2 * np.pi], axis=1), axis=1).max(1)
    return _NULL[key]


def gap_p(k, max_gap_rad, n_perm, seed=0):
    """Copied verbatim from scripts/22_token_type_geometry.py L375-377."""
    g = gap_null(k, n_perm, seed)
    return float((1 + (g <= max_gap_rad).sum()) / (1 + n_perm))


def screen(M, n_perm, seed=0):
    """Copied verbatim from scripts/22_token_type_geometry.py L379-391."""
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


def member_hash(members):
    """Copied verbatim from scripts/22_token_type_geometry.py L561-562."""
    return hashlib.md5(",".join(str(int(x)) for x in sorted(members)).encode()).hexdigest()[:12]


def _rank_of(values):
    """Copied verbatim from scripts/29_keys_blockperm.py L86-90."""
    order = np.argsort(np.asarray(values), kind="stable")
    r = np.empty(len(order), dtype=np.float64)
    r[order] = np.arange(len(order), dtype=np.float64)
    return r


def _rank_from_angles(ang, seed):
    """Copied verbatim from scripts/29_keys_blockperm.py L93-101."""
    ang = np.asarray(ang, dtype=float)
    n = len(ang)
    jitter = np.random.default_rng(seed).permutation(n)
    order = np.lexsort((jitter, ang))
    rank_of = np.empty(n, dtype=np.float64)
    rank_of[order] = np.arange(n)
    return rank_of


def _sym_basis(label_index, K):
    """Copied verbatim from scripts/29_keys_blockperm.py L104-106."""
    lab = np.asarray(label_index, dtype=int)
    return np.vstack([_rank_of((f * lab + s) % K) for f in (1, -1) for s in range(K)])


def block_permutations(blocks, n_draws, seed):
    """Copied verbatim from scripts/29_keys_blockperm.py L108-117."""
    rng = np.random.default_rng(seed)
    n = sum(len(b) for b in blocks)
    sig = np.tile(np.arange(n), (n_draws, 1))
    for b in blocks:
        b = np.asarray(b)
        pick = np.argsort(rng.random((n_draws, len(b))), axis=1)
        sig[:, b] = b[pick]
    return sig


# ===========================================================================
# derived statistics
# ===========================================================================
def cell_angles(M):
    """Angles of the member means in their own PCA plane, about the FITTED circle centre.

    Exactly scripts/22's score_ordering convention (pca_2d -> fit_circle -> angles(P, cx, cy)),
    so the angles the ordering tier reads are the same angles whose max gap the closure tier
    tested -- ``circularity`` measures gaps about (cx, cy), not about the centroid.
    """
    xy, _, _ = pca_2d(np.asarray(M, dtype=np.float64))
    cx, cy, r, _ = fit_circle(xy)
    return angles(xy, cx, cy), xy, (float(cx), float(cy), float(r))


def block_perm_order_p(ang, label_index, K, blocks, n_draws, seed):
    """Permutation p for the max-over-symmetries rho, permuting only WITHIN blocks.

    Same vectorised path as scripts/29_keys_blockperm.analyse_cell, reduced to a single
    label cycle. For strict-12 the block list is a single block of all 12 positions, so
    every draw is a uniform permutation and this is the ordinary label-shuffle null; it is
    still run through the block machinery so that a later 13-set or lexical-block variant
    uses byte-identical code.
    """
    rank_of = _rank_from_angles(ang, seed)
    n = len(rank_of)
    base = _sym_basis(label_index, K)
    denom = n * (n * n - 1)

    def rhos(sig):
        d = rank_of - base[:, sig]
        return 1.0 - 6.0 * (d * d).sum(-1) / denom

    rho_obs = float(rhos(np.arange(n)).max())
    sig = block_permutations(blocks, n_draws, seed + 31337)
    null = rhos(sig).max(0)
    return dict(rho=rho_obs, n_draws=int(n_draws),
                p=float((1 + int((null >= rho_obs).sum())) / (1 + n_draws)),
                null_mean=float(null.mean()), null_p95=float(np.percentile(null, 95)),
                blocks="single block of all 12 positions -> uniform permutations "
                       "(strict-12 has no lexical block structure to preserve)")


def mean_radius(M):
    """Mean in-plane distance of the member means from the fitted circle centre."""
    xy, _, _ = pca_2d(np.asarray(M, dtype=np.float64))
    cx, cy, r, _ = fit_circle(xy)
    return float(np.hypot(xy[:, 0] - cx, xy[:, 1] - cy).mean())


# ===========================================================================
# the forward ladder
# ===========================================================================
def run_ladder(model, tok_bos, tok_pad, saes, tokens2d, targets, rungs, layers, n_ctx,
               exclude_ids, seed_fn, batch_size, keep_rows_at, label=""):
    """Forward every (target, rung) cell and return the per-cell mean vectors.

    Returns ``(means, meta, rows, timing)`` where
      means[(layer, rung, arm)] -> float64 [n_targets, 1152], rows in ``targets`` order;
      meta[(target, rung)]      -> dict of n / n_unique / n_forwarded / seed / doc counts;
      rows[(layer, target)]     -> float32 [n_ctx, 1152] per-CONTEXT vectors, expanded to
                                   full multiplicity, only for the rung in ``keep_rows_at``
                                   and only for the certificate arm's reconstruction;
      timing[rung]              -> dict of batches / seconds / tokens.

    Both layers are captured in ONE forward per batch, so the L7 arm is free.
    """
    eligible = rig.eligible_docs(tokens2d, exclude_ids)
    means = {(l, c, a): np.zeros((len(targets), 1152), dtype=np.float64)
             for l in layers for c in rungs for a in ARMS}
    meta, rows, timing = {}, {}, {c: dict(batches=0, seconds=0.0, tokens=0) for c in rungs}

    for c in rungs:
        for ti, target_id in enumerate(targets):
            seed = seed_fn(c, ti)
            rng = np.random.default_rng(seed)
            ctx, docs = rig.sample_contexts(tokens2d, n_ctx, c, rng, eligible=eligible)
            uniq, weights, _ = rig.dedupe(ctx)
            acc = {(l, a): np.zeros(1152, dtype=np.float64) for l in layers for a in ARMS}
            keep = {l: [] for l in layers} if c == keep_rows_at else None
            t_cell = time.time()
            for lo, chunk in rig.iter_batches(uniq, batch_size):
                w = torch.from_numpy(weights[lo:lo + len(chunk)]).double()
                ids, attn, tpos = rig.build_batch(chunk, target_id, tok_bos, tok_pad)
                t0 = time.time()
                resid = rig.forward_capture(model, ids.to(DEVICE), attn.to(DEVICE), layers)
                got = rig.gather(resid, tpos)
                del resid
                for l in layers:
                    x = got[l]                                    # [b, 1152] fp32 cpu
                    _, recon = rig.sae_reconstruct(saes[l], x)
                    recon = recon.detach().cpu().double()
                    raw = x.double()
                    acc[(l, "sae")] += (recon * w[:, None]).sum(0).numpy()
                    acc[(l, "raw")] += (raw * w[:, None]).sum(0).numpy()
                    if keep is not None:
                        keep[l].append(recon.float().numpy() if CERT_ARM == "sae"
                                       else raw.float().numpy())
                timing[c]["batches"] += 1
                timing[c]["seconds"] += time.time() - t0
                timing[c]["tokens"] += int(attn.sum())
            tot = float(weights.sum())
            for l in layers:
                for a in ARMS:
                    means[(l, c, a)][ti] = acc[(l, a)] / tot
                if keep is not None:
                    blk = np.vstack(keep[l])                      # [n_unique, 1152]
                    rows[(l, target_id)] = np.repeat(blk, weights, axis=0)
            meta[(int(target_id), c)] = dict(
                n=int(n_ctx), n_unique=int(len(uniq)), n_forwarded=int(len(uniq)),
                seed=int(seed), n_eligible_docs=int(eligible.size),
                n_distinct_docs=int(np.unique(docs).size))
            print(f"  {label}c={c:<4d} target={int(target_id):<7d} "
                  f"unique={len(uniq):<4d} {time.time() - t_cell:5.1f}s", flush=True)
    return means, meta, rows, timing


# ===========================================================================
# per-cell scoring
# ===========================================================================
def score_cell(M, ids, n_perm, want_ordering, seed):
    """screen() + (strict-12 only) ordering, for one (layer, rung, arm, set) cell."""
    s = screen(M, n_perm)
    out = dict(k=int(len(ids)), n_members=int(len(ids)))
    if s is None:
        out.update(degenerate=True, gap_ratio=None, max_gap_deg=None, radius_ratio=None,
                   p_closed=None, mean_radius=None, top2_var=None, pc1_var=None,
                   circle_resid_over_r=None)
    else:
        out.update(degenerate=False, **{k: v for k, v in s.items() if k != "k"},
                   mean_radius=mean_radius(M))
    out.update(rho=None, p_order=None, block_perm_p=None, order_symmetry=None,
               order_note=None)
    if not want_ordering:
        out["order_note"] = ("ordering not computed: cyclic_order_score's symmetry basis "
                             "needs distinct calendar positions and ' Feb' ties with "
                             "' February' at position 1 (prereg 3.1, see module docstring)")
        return out
    if s is None:
        out["order_note"] = "ordering not computed: point set is degenerate"
        return out
    ang, _, _ = cell_angles(M)
    lab = [CALENDAR_POS[int(t)] for t in ids]
    o = cyclic_order_score(ang, np.asarray(lab), len(ids), n_perm=N_PERM_ORDER,
                           seed=seed, full_k=FULL_K)
    bp = block_perm_order_p(ang, lab, FULL_K, [list(range(len(ids)))], N_BLOCK_PERM, seed)
    out.update(rho=float(o["rho"]), p_order=float(o["p"]),
               order_symmetry=[int(x) for x in o["symmetry"]],
               order_null_mean=float(o["null_mean"]), order_null_p95=float(o["null_p95"]),
               block_perm_p=float(bp["p"]), block_perm=bp,
               label_index=[int(x) for x in lab], angles_rad=[float(x) for x in ang])
    return out


def decide(table, layer, set_name, rungs):
    """Prereg 3.1 decision rule, per layer. SAE arm primary, raw arm co-reported."""
    def p(arm, c):
        cell = table.get((layer, c, arm, set_name))
        return None if cell is None else cell["p_closed"]

    have = sorted(rungs)
    complete = have == sorted(RUNGS)
    sae0, raw0 = p("sae", 0), p("raw", 0)
    disagree = None
    if sae0 is not None and raw0 is not None:
        disagree = bool((sae0 < ALPHA) != (raw0 < ALPHA))
    out = dict(layer=int(layer), member_set=set_name, rungs_available=have,
               ladder_complete=bool(complete),
               p_closed_sae={str(c): p("sae", c) for c in have},
               p_closed_raw={str(c): p("raw", c) for c in have},
               sae_vs_raw_disagreement_at_c0=disagree,
               p_closed_c0_sae=sae0, p_closed_c0_raw=raw0)

    branch_a = all(v is not None and v < ALPHA
                   for v in (p("sae", 0), p("sae", 1), p("raw", 0), p("raw", 1)))
    onset = None
    if sae0 is not None and sae0 >= ALPHA:
        for i, c in enumerate(have):
            tail = have[i:]
            vals = [p("sae", x) for x in tail]
            if c > 0 and all(v is not None and v < ALPHA for v in vals):
                onset = int(c)
                break
    if branch_a:
        call, why = "Branch A (token-intrinsic)", "p_closed < 0.05 at c=0 and c=1 on both arms"
    elif onset is not None:
        call, why = ("Branch B (contextual onset)",
                     f"c=0 SAE p_closed >= 0.05; closure from rung E={onset} upward")
    else:
        call, why = ("non-monotone / unresolved",
                     "neither Branch A nor a monotone onset rung E is satisfied")
    out.update(onset_rung_E=onset, branch=call, reason=why)
    if not complete:
        out["branch"] = "incomplete rung ladder -- no verdict"
        out["provisional_branch"] = call
        out["reason"] = (f"decision rule needs rungs {RUNGS}; this run has {have}. "
                         f"Provisional call on the available rungs only: {call} ({why})")
    return out


# ===========================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="cache/sparse_big")
    ap.add_argument("--l0", default="big")
    ap.add_argument("--out", default="results/phase3_context_ladder.json")
    ap.add_argument("--means-out", default="results/phase3_context_ladder_means.npz")
    ap.add_argument("--layers", type=int, nargs="+", default=LAYERS)
    ap.add_argument("--rungs", type=int, nargs="+", default=RUNGS)
    ap.add_argument("--n-ctx", type=int, default=N_CTX)
    ap.add_argument("--n-perm", type=int, default=N_PERM_SCREEN)
    ap.add_argument("--n-ord", type=int, default=ORD_N)
    ap.add_argument("--cert-resamples", type=int, default=CERT_RESAMPLES)
    ap.add_argument("--batch-size", type=int, default=rig.BATCH_SIZE)
    ap.add_argument("--token-map", default="results/token_map.json")
    ap.add_argument("--smoke", action="store_true",
                    help="annotate the output as a scaled-down instrument check")
    a = ap.parse_args()
    t_start = time.time()
    layers, rungs = sorted(set(a.layers)), sorted(set(a.rungs))

    # ---- R1: declaration to disk BEFORE any cache row or forward pass ----
    declared = dict(
        script="scripts/37_context_ladder.py",
        prereg="docs/phase3_prereg_2026-08-21.md section 3.1",
        status="DECLARATION STUB -- rules frozen, no results yet",
        declared_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
        smoke=bool(a.smoke),
        question=("Is the L6/L7 month ring token-intrinsic, or assembled from context, "
                  "and at what context budget does it come into existence?"),
        members=dict(member_ids=MEMBER_IDS, feb_abbrev_id=FEB_ABBREV_ID,
                     strict12_ids=STRICT12_IDS,
                     member_hash_expected=MEMBER_HASH_EXPECTED,
                     member_hash_domain="sorted POOL INDICES (build_pool, count floor 100)",
                     member_index_rule="position in the ascending token-id list above"),
        rungs=rungs, layers=layers, n_contexts=int(a.n_ctx),
        context_rule=("[BOS] + last-c tokens of a cached FineWeb document + member token; "
                      "documents containing ANY of the 13 member ids or any month variant "
                      "id in results/token_map.json are excluded; end position uniform on "
                      "[c, 255]; right padding; bf16/eager/use_cache=False"),
        seed_rule="months: 1000*c + member_index; ordinals: 500000 + 1000*c + ordinal_index",
        arms=dict(sae="mean over contexts of (fp32 JumpReLU code) @ w_dec, NO b_dec",
                  raw="mean over contexts of the fp32 resid_post at the target position",
                  primary="sae (the registered anchor is an SAE-reconstruction number)"),
        statistics=dict(
            screen=f"scripts/22 screen() verbatim, {a.n_perm} uniform-angle permutations",
            ordering=(f"cyclic_order_score(full_k=12, n_perm={N_PERM_ORDER}) on strict-12 "
                      f"only, plus a {N_BLOCK_PERM}-draw block-permutation p (single block "
                      f"of all 12 -> uniform permutations; strict-12 has no lexical blocks)"),
            angles="pca_2d -> fit_circle -> angles(xy, cx, cy), scripts/22 score_ordering"),
        rig_validation=dict(
            cell=f"c={CERT_RUNG}, L{CERT_LAYER}, {CERT_ARM} arm, both member sets",
            rule=(f"|gap_ratio - {RIG_REF_GAP_RATIO}| <= {RIG_GAP_TOL} AND "
                  f"p_closed < {RIG_MAX_P_CLOSED}; the 13-set is the registered anchor"),
            on_failure="verdict 'rig not validated'; all numbers are still emitted"),
        certificate=dict(
            cell=f"c={CERT_RUNG}, L{CERT_LAYER}, {CERT_ARM} arm",
            rule=(f"{a.cert_resamples} resamples of n={CERT_N} contexts per member; "
                  f"closure pass rate (p_closed < {ALPHA}) must be >= {CERT_MIN_PASS}")),
        ordinal_control=dict(
            words=ORD_WORDS, rungs=ORD_RUNGS, n=int(a.n_ord),
            rule=("leading-space single tokens only; non-single-token members are reported "
                  f"and dropped; any cell with p_closed < {ALPHA} sets run_void=true")),
        decision_rule=dict(
            branch_a="p_closed < 0.05 at c=0 AND c=1 on BOTH arms -> token-intrinsic",
            branch_b=("p_closed >= 0.05 at c=0 on the SAE arm AND a rung E with "
                      "p_closed < 0.05 at E and at every rung > E -> contextual onset at E"),
            otherwise="non-monotone / unresolved; SAE-vs-raw disagreement at c=0 reported"),
        no_bdec_note=("SAE reconstruction is code @ w_dec with no b_dec, matching every "
                      "closure number the campaign has reported; b_dec is type-independent "
                      "but translates the cloud and would move every fitted angle."),
    )
    outp = Path(a.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(declared, indent=2))
    print(f"declaration stub -> {outp}", flush=True)

    # ---- pool + member hash assertion -------------------------------------
    tokens_flat = np.fromfile(Path(a.root) / "layer_0" / "tokens.i32", dtype=np.int32)
    counts = np.bincount(tokens_flat)
    pool = np.flatnonzero(counts >= COUNT_FLOOR)
    lut = {int(t): i for i, t in enumerate(pool)}
    missing = [t for t in MEMBER_IDS if t not in lut]
    if missing:
        raise RuntimeError(f"members absent from the count>={COUNT_FLOOR} pool: {missing}")
    pool_idx = [lut[t] for t in MEMBER_IDS]
    got_hash = member_hash(pool_idx)
    assert got_hash == MEMBER_HASH_EXPECTED, (
        f"member_hash {got_hash} != {MEMBER_HASH_EXPECTED}; the frozen object is not the "
        f"object this pool names")
    print(f"pool {len(pool)} types; member_hash {got_hash} OK "
          f"(pool indices {sorted(pool_idx)})", flush=True)
    tokens2d = tokens_flat.reshape(rig.N_DOCS, rig.DOC_LEN)
    del tokens_flat

    # ---- exclusion set ----------------------------------------------------
    tmap = json.loads(Path(a.token_map).read_text())["months"]
    exclude = sorted({int(x) for v in tmap.values() for x in v} | set(MEMBER_IDS))
    print(f"exclusion set: {len(exclude)} token ids (13 members + token_map variants)",
          flush=True)

    # ---- model + SAEs -----------------------------------------------------
    hf = os.environ.get("HF_TOKEN")
    model, tokzr = load_model(token=hf)
    bos = tokzr.bos_token_id
    pad = tokzr.pad_token_id if tokzr.pad_token_id is not None else tokzr.eos_token_id
    # SAEs live on DEVICE, matching scripts/06_cache_sparse.py's encode path -- the frozen
    # anchor this run validates against was produced by that path, so the JumpReLU gate is
    # evaluated on the same backend.
    saes = {l: load_sae(l, a.l0, device=DEVICE, token=hf) for l in layers}
    print(f"model on {DEVICE}; SAEs {a.l0} at layers {layers}; bos={bos} pad={pad}",
          flush=True)

    tokenizer_check = {}
    for m in MONTHS:
        enc = tokzr.encode(" " + m, add_special_tokens=False)
        tokenizer_check[" " + m] = enc
        assert len(enc) == 1, f"' {m}' is not a single token: {enc}"
    tokenizer_check[" Feb"] = tokzr.encode(" Feb", add_special_tokens=False)
    assert tokenizer_check[" Feb"] == [FEB_ABBREV_ID]
    assert sorted(tokenizer_check[" " + m][0] for m in MONTHS) == sorted(STRICT12_IDS), \
        "the tokenizer's month ids do not match the frozen strict-12 set"

    # ---- ordinal control token resolution (before any of its forwards) ----
    ord_ids, ord_dropped = [], {}
    for w in ORD_WORDS:
        enc = tokzr.encode(" " + w, add_special_tokens=False)
        if len(enc) == 1:
            ord_ids.append(int(enc[0]))
        else:
            ord_dropped[" " + w] = [int(x) for x in enc]
    print(f"ordinal control: {len(ord_ids)} single tokens, {len(ord_dropped)} dropped "
          f"({sorted(ord_dropped)})", flush=True)

    # =======================================================================
    # the ladder
    # =======================================================================
    print(f"\nmonth ladder: {len(MEMBER_IDS)} members x {len(rungs)} rungs x "
          f"{a.n_ctx} contexts", flush=True)
    means, meta, cert_rows, timing = run_ladder(
        model, bos, pad, saes, tokens2d, MEMBER_IDS, rungs, layers, a.n_ctx, exclude,
        seed_fn=lambda c, i: 1000 * c + i, batch_size=a.batch_size,
        keep_rows_at=(CERT_RUNG if (CERT_RUNG in rungs and CERT_LAYER in layers) else None))

    # ---- per-cell table ---------------------------------------------------
    print("\nscoring cells", flush=True)
    table, table_rows = {}, []
    row_of = {int(t): i for i, t in enumerate(MEMBER_IDS)}
    for l in layers:
        for c in rungs:
            for arm in ARMS:
                for sname, sids in SETS.items():
                    M = means[(l, c, arm)][[row_of[t] for t in sids]]
                    cell = score_cell(M, sids, a.n_perm,
                                      want_ordering=(sname == "strict12"),
                                      seed=1000 * c + l)
                    uq = [meta[(int(t), c)]["n_unique"] for t in sids]
                    cell.update(layer=int(l), rung=int(c), arm=arm, member_set=sname,
                                n=int(a.n_ctx),
                                n_unique_contexts_min=int(min(uq)),
                                n_unique_contexts_max=int(max(uq)))
                    table[(l, c, arm, sname)] = cell
                    table_rows.append(cell)
                    gr = "None" if cell["gap_ratio"] is None else f"{cell['gap_ratio']:.4f}"
                    pc = "None" if cell["p_closed"] is None else f"{cell['p_closed']:.5f}"
                    rho = "" if cell["rho"] is None else f" rho={cell['rho']:.3f}"
                    print(f"  L{l} c={c:<4d} {arm:<3s} {sname:<8s} gap_ratio={gr:<8s} "
                          f"p_closed={pc}{rho}", flush=True)

    # ---- rig validation ---------------------------------------------------
    rig_val = dict(cell=dict(layer=CERT_LAYER, rung=CERT_RUNG, arm=CERT_ARM),
                   rule=declared["rig_validation"]["rule"], sets={})
    ok_all = True
    for sname in SETS:
        cell = table.get((CERT_LAYER, CERT_RUNG, CERT_ARM, sname))
        if cell is None or cell["gap_ratio"] is None:
            rig_val["sets"][sname] = dict(available=False, passed=False)
            ok_all = False
            continue
        gr, pc = cell["gap_ratio"], cell["p_closed"]
        ok = bool(abs(gr - RIG_REF_GAP_RATIO) <= RIG_GAP_TOL and pc < RIG_MAX_P_CLOSED)
        rig_val["sets"][sname] = dict(available=True, gap_ratio=gr, p_closed=pc,
                                      gap_ratio_abs_error=float(abs(gr - RIG_REF_GAP_RATIO)),
                                      passed=ok)
        ok_all = ok_all and ok
    rig_val["passed"] = bool(ok_all)
    rig_val["anchor_set"] = "set13"
    rig_val["verdict"] = "rig validated" if ok_all else "rig not validated"
    print(f"\nrig validation: {rig_val['verdict']} {rig_val['sets']}", flush=True)

    # ---- matched-n certificate -------------------------------------------
    cert = dict(cell=dict(layer=CERT_LAYER, rung=CERT_RUNG, arm=CERT_ARM),
                n_requested=CERT_N, n_resamples=int(a.cert_resamples),
                threshold=CERT_MIN_PASS, sets={})
    if cert_rows:
        stack = np.stack([cert_rows[(CERT_LAYER, t)] for t in MEMBER_IDS])  # [13, n, 1152]
        n_avail = stack.shape[1]
        n_cert = int(min(CERT_N, n_avail))
        cert.update(n_used=n_cert, n_available=int(n_avail))
        crng = np.random.default_rng(20260821)
        for sname, sids in SETS.items():
            sel = [row_of[t] for t in sids]
            passes = 0
            for _ in range(int(a.cert_resamples)):
                M = np.stack([stack[s, crng.permutation(n_avail)[:n_cert]].mean(0)
                              for s in sel])
                s = screen(M, a.n_perm)
                passes += int(s is not None and s["p_closed"] < ALPHA)
            rate = passes / max(int(a.cert_resamples), 1)
            cert["sets"][sname] = dict(pass_rate=float(rate),
                                       passed=bool(rate >= CERT_MIN_PASS))
        cert["passed"] = bool(cert["sets"]["set13"]["passed"])
        cert["primary_set"] = "set13"
        cert["degenerate"] = bool(n_cert >= n_avail)
        cert["resample_rule"] = (
            f"subsample WITHOUT replacement, n={n_cert} of the {n_avail} stored "
            f"per-context reconstructions per member at c={CERT_RUNG}; the matched-n "
            f"question is whether n={CERT_N} contexts would have sufficed, which is a "
            f"subsample of the run, not a bootstrap of it"
            + (" -- DEGENERATE here: n_used == n_available, so every resample is the full "
               "set and the pass rate is 0 or 1" if n_cert >= n_avail else ""))
    else:
        cert.update(available=False, passed=False,
                    note=f"c={CERT_RUNG} / L{CERT_LAYER} not in this run's grid")
    print(f"certificate: {cert.get('sets', cert.get('note'))}", flush=True)

    # =======================================================================
    # ordinal control
    # =======================================================================
    ord_out = dict(words=ORD_WORDS, dropped=ord_dropped, kept_ids=ord_ids,
                   k=len(ord_ids), n=int(a.n_ord), rungs=ORD_RUNGS, cells=[])
    ord_means = {}
    ord_rungs = [c for c in ORD_RUNGS if c <= rig.DOC_LEN]
    if len(ord_ids) >= 3:
        print(f"\nordinal control: {len(ord_ids)} tokens x {len(ord_rungs)} rungs x "
              f"{a.n_ord} contexts", flush=True)
        om, ometa, _, otiming = run_ladder(
            model, bos, pad, saes, tokens2d, ord_ids, ord_rungs, layers, a.n_ord, ord_ids,
            seed_fn=lambda c, i: ORD_SEED_BASE + 1000 * c + i, batch_size=a.batch_size,
            keep_rows_at=None, label="ord ")
        ord_means = om
        for l in layers:
            for c in ord_rungs:
                for arm in ARMS:
                    s = screen(om[(l, c, arm)], a.n_perm)
                    cell = dict(layer=int(l), rung=int(c), arm=arm, k=len(ord_ids),
                                n=int(a.n_ord))
                    cell.update({k: v for k, v in (s or {}).items() if k != "k"})
                    cell.setdefault("p_closed", None)
                    ord_out["cells"].append(cell)
        for c in ord_rungs:
            ord_out.setdefault("exclusion", {})[str(c)] = int(
                ometa[(ord_ids[0], c)]["n_eligible_docs"])
        ord_out["timing"] = {str(k): v for k, v in otiming.items()}
    else:
        ord_out["note"] = "fewer than 3 single-token ordinals survive; control not runnable"
    hits = [c for c in ord_out["cells"]
            if c["p_closed"] is not None and c["p_closed"] < ALPHA]
    ord_out["closing_cells"] = hits
    run_void = bool(hits)
    ord_out["run_void"] = run_void
    print(f"ordinal control: {len(hits)} closing cells -> run_void={run_void}", flush=True)

    # =======================================================================
    # decisions
    # =======================================================================
    decisions = []
    for l in layers:
        for sname in SETS:
            d = decide(table, l, sname, rungs)
            d["primary"] = bool(sname == "set13")
            d["gated_by"] = dict(rig_validated=rig_val["passed"], run_void=run_void,
                                 certificate_passed=bool(cert.get("passed")))
            # A failed gate withholds the verdict; it does not erase what the rule computed.
            # Keeping the branch the rule WOULD have called is what lets a reader see that
            # the gate, and not a convenient answer, is doing the work.
            if not rig_val["passed"] or run_void:
                d["branch_under_gate"] = d.get("provisional_branch", d["branch"])
                d["branch"] = ("VOID (ordinal control closed) -- no verdict" if run_void
                               else "rig not validated -- no verdict")
            decisions.append(d)
            if sname == "set13":
                print(f"  L{l} {sname}: {d['branch']}", flush=True)

    # =======================================================================
    # runtime extrapolation
    # =======================================================================
    def plan_cost(rungs_, n_ctx_, n_members, bs):
        """(batches, tokens) for a ladder, assuming c=0 dedupes to one row per member."""
        b = t = 0
        for c in rungs_:
            nu = 1 if c == 0 else n_ctx_
            b += n_members * int(np.ceil(nu / bs))
            t += n_members * nu * (c + 2)
        return b, t

    obs_b = sum(v["batches"] for v in timing.values())
    obs_s = sum(v["seconds"] for v in timing.values())
    obs_t = sum(v["tokens"] for v in timing.values())
    if ord_out.get("timing"):
        obs_b += sum(v["batches"] for v in ord_out["timing"].values())
        obs_s += sum(v["seconds"] for v in ord_out["timing"].values())
        obs_t += sum(v["tokens"] for v in ord_out["timing"].values())
    # seconds ~ alpha * batches + beta * tokens, fitted on the per-rung aggregates
    A = np.array([[v["batches"], v["tokens"]] for v in timing.values()], dtype=np.float64)
    y = np.array([v["seconds"] for v in timing.values()], dtype=np.float64)
    coef = np.linalg.lstsq(A, y, rcond=None)[0] if len(y) >= 2 else np.array([0.0, 0.0])
    coef = np.maximum(coef, 0.0)
    fb, ft = plan_cost(RUNGS, N_CTX, len(MEMBER_IDS), a.batch_size)
    ob, ot = plan_cost(ORD_RUNGS, ORD_N, max(len(ord_ids), 1), a.batch_size)
    full_forward_s = float(coef[0] * (fb + ob) + coef[1] * (ft + ot))
    fit_used = "two-term fit (batch overhead + per-token)"
    if full_forward_s <= 0 and obs_t > 0:
        # fewer than two distinct rungs -> the two-term fit is underdetermined; fall back
        # to this run's own throughput, which over-charges the short rungs.
        full_forward_s = float((ft + ot) / (obs_t / max(obs_s, 1e-9)))
        fit_used = "throughput fallback (only one rung timed; two-term fit unidentifiable)"
    per_rung_sec_per_batch = {str(c): (v["seconds"] / v["batches"] if v["batches"] else None)
                              for c, v in timing.items()}
    runtime = dict(
        observed=dict(batches=int(obs_b), seconds_forward=float(obs_s), tokens=int(obs_t),
                      sec_per_batch=float(obs_s / max(obs_b, 1)),
                      tokens_per_sec=float(obs_t / max(obs_s, 1e-9)),
                      sec_per_batch_by_rung=per_rung_sec_per_batch),
        model_seconds=dict(fit_batch_overhead_s=float(coef[0]),
                           fit_sec_per_token=float(coef[1])),
        full_plan=dict(month_batches=int(fb), month_tokens=int(ft),
                       ordinal_batches=int(ob), ordinal_tokens=int(ot),
                       month_sequences_forwarded=int(sum(
                           len(MEMBER_IDS) * (1 if c == 0 else N_CTX) for c in RUNGS)),
                       month_sequences_declared=int(len(MEMBER_IDS) * len(RUNGS) * N_CTX),
                       estimated_forward_seconds=full_forward_s,
                       estimated_forward_minutes=full_forward_s / 60.0,
                       method=fit_used),
        note=("seconds ~ a*batches + b*tokens, least squares on this run's per-rung "
              "aggregates; excludes model/SAE load and all permutation work"))
    print(f"\nruntime: {obs_s:.1f}s forward over {obs_b} batches "
          f"({obs_s / max(obs_b, 1):.3f} s/batch, {obs_t / max(obs_s, 1e-9):,.0f} tok/s); "
          f"full-plan forward estimate {full_forward_s / 60:.1f} min", flush=True)

    # =======================================================================
    # write
    # =======================================================================
    npz = {}
    for (l, c, arm), M in means.items():
        npz[f"L{l}_c{c}_{arm}"] = M.astype(np.float32)
    for (l, c, arm), M in ord_means.items():
        npz[f"ord_L{l}_c{c}_{arm}"] = M.astype(np.float32)
    npz["member_ids"] = np.asarray(MEMBER_IDS, dtype=np.int64)
    npz["strict12_ids"] = np.asarray(STRICT12_IDS, dtype=np.int64)
    npz["ordinal_ids"] = np.asarray(ord_ids, dtype=np.int64)
    npz["rungs"] = np.asarray(rungs, dtype=np.int64)
    npz["layers"] = np.asarray(layers, dtype=np.int64)
    mp = Path(a.means_out)
    mp.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(mp, **npz)

    verdict = rig_val["verdict"]
    if run_void:
        verdict = "VOID (ordinal control closed); " + verdict

    out = dict(declared)
    out.update(
        status="COMPLETE",
        declared_at=declared["declared_at"],
        finished_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
        smoke=bool(a.smoke),
        config=dict(root=a.root, l0=a.l0, layers=layers, rungs=rungs, n_ctx=int(a.n_ctx),
                    n_perm=int(a.n_perm), n_ord=int(a.n_ord),
                    cert_resamples=int(a.cert_resamples), batch_size=int(a.batch_size),
                    device=DEVICE, means_out=str(mp)),
        members=dict(declared["members"], pool_indices=sorted(int(x) for x in pool_idx),
                     member_hash=got_hash, pool_size=int(len(pool)),
                     calendar_position={str(k): int(v) for k, v in CALENDAR_POS.items()},
                     member_index={str(t): i for i, t in enumerate(MEMBER_IDS)},
                     tokenizer_check={k: [int(x) for x in v]
                                      for k, v in tokenizer_check.items()},
                     exclusion_ids=exclude),
        cells=table_rows,
        context_meta={f"{t}|{c}": v for (t, c), v in meta.items()},
        rig_validation=dict(declared["rig_validation"], **rig_val),
        certificate=dict(declared["certificate"], **cert),
        ordinal_control=dict(declared["ordinal_control"], **ord_out),
        decisions=decisions,
        run_void=run_void,
        verdict=verdict,
        timing_by_rung={str(k): v for k, v in timing.items()},
        runtime_extrapolation=runtime,
        runtime_seconds=float(time.time() - t_start),
    )
    outp.write_text(json.dumps(out, indent=2, default=float))
    print(f"\nverdict: {verdict}", flush=True)
    print(f"means -> {mp}", flush=True)
    print(f"wrote {outp} ({time.time() - t_start:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
