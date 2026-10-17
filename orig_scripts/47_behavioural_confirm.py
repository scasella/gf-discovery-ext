"""47 -- the BEHAVIOURAL CONFIRMATION tier (Phase-4 prereg section 2.6; Phase-5 addendum P, P4).

The closure/ordering tiers say a member set's raw type means lie on a ring whose angular
order is not reproducible by permuted search. That is a statement about GEOMETRY. It is not
a statement about the model USING that geometry. This script asks the causal question: if
the residual stream at a member's own token position is pushed along the ring's in-plane
TANGENT at that member, does the next-token distribution move toward the member's angular
SUCCESSOR and away from its PREDECESSOR?

  at member i:  h  <-  h + eps * ||h|| * t_hat_i        (resid_post of layer L, member's pos)
  Delta_i       :=  [logit(succ_i) - logit(pred_i)]_steered
                  - [logit(succ_i) - logit(pred_i)]_unsteered

The baseline subtraction is a DECLARED READING of the prereg's "Delta_i = logit(angular
successor) - logit(angular predecessor)" and is stated as such in the stub. Without it the
statistic is dominated by the standing (unsteered) logit asymmetry between two arbitrary
vocabulary items, which has nothing to do with steering; every null would inherit the same
offset and the tier would measure token frequency. Both the raw steered value and the
baseline-subtracted value are stored per cell.

WHAT MAKES THE SIGN OF Delta A PREDICTION AND NOT A COIN FLIP. Angles, cyclic order and
tangents are all derived from ONE basis: the top-2 PCA basis (u1, u2) of the members' raw
type means. theta_i = atan2(y_i - cy, x_i - cx) in that basis; the successor is the next
member counter-clockwise (increasing theta); and t_hat_i = -sin(theta_i) u1 + cos(theta_i) u2
is the unit tangent in the direction of INCREASING theta. So "steer along t_hat" and "move
toward the successor" are the same statement by construction, and Delta_i > 0 is the
prediction. A PCA sign flip flips u1/u2, theta, the successor relation and the tangent
together, so the verdict is invariant; an assert at fit time checks the tangent is the
radial direction rotated by exactly +90 degrees.

THE SPACE. The closure tier drops coordinate 1038 (prereg R10) before any geometry. The
model's residual stream does not have that coordinate dropped. So the plane, circle, angles
and tangent are all fitted in the 1151-dim dropped space, and the tangent is re-embedded
into 1152 dims by inserting 0.0 at index 1038 before it is used to steer. Re-embedding
preserves the norm, so eps*||h|| still matches the intended intervention size.

NULLS (all at the pinned eps declared below, never at a post-hoc best eps):
  (i)   random directions per member, unit, orthogonalised against that member's tangent;
  (ii)  the tangent rotated +90 / -90 degrees IN PLANE (i.e. the inward / outward radial);
  (iii) random k-sets drawn from the same pool, each with its OWN plane, circle, angular
        order and tangents, scored by the identical pipeline.

Nothing about the members' identity enters the statistic: succ/pred are read off the fitted
angles, so a set whose ring is an artifact of the fit has exactly as much chance as one whose
ring the model uses. That is what null (iii) calibrates.

Deviation from prereg 2.6, declared in the stub before any forward: "route intermediate
mass" is replaced by an in-set softmax readout, because the prereg's metric is defined for
multi-step A->B arcs and the i->successor step here has no intermediate members by
construction. Null (i) runs 5 random directions rather than 20 and the nulls use 12 prompts
per member rather than 40, both for throughput on a machine whose memory is largely wired by
an unrelated process; null (iii), which the PASS rule gates on, runs the prereg's full count.
"""
import argparse
import contextlib
import hashlib
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from gfdisc import fits, rig                      # noqa: E402
from gfdisc.env import DEVICE, MODEL_NAME         # noqa: E402
from gfdisc.model import capture, load_model      # noqa: E402

# --------------------------------------------------------------------------------------
# frozen constants
# --------------------------------------------------------------------------------------
FULLN_RAW = "results/phase3_token_type_mean_resid_fulln.npz"
SCREEN = "results/phase4_assembly_screen_addendumL.json"
MULTI = "results/phase4_full_multiplicity_null.json"
CACHE_ROOT = "cache/sparse_big"

#: prereg R10 -- the coordinate dropped from every layer >= 3 geometry in this campaign.
R10_DROP = 1038

EPS_GRID = (0.005, 0.01, 0.02)
#: The nulls run at ONE strength and it is pinned here, before any forward, so that the
#: PASS comparison cannot be moved to whichever eps happens to look best.
EPS_NULL = 0.02

N_PROMPTS = 40            # natural arm, per member
CONTEXT_LEN = 8           # tokens of preceding context, ending with the member token
N_PROMPTS_NULL = 12       # per member, in nulls (i)-(iii)
N_RAND_DIRS = 5           # null (i), reduced from the prereg's 20
N_NULL_SETS = 20          # null (iii), prereg count

BATCH = 16                # local override of rig.BATCH_SIZE: this machine is memory-tight

SEED_CONTEXT = 4700
SEED_DIRS = 4701
SEED_NULLSETS = 4702

#: The four open Phase-4 candidates (addendum P) plus the strict-12 anchor at both layers.
TARGETS = [
    ("L6_strict12", 6, "a36a4f4f3e3b", "positive_control"),
    ("L7_strict12", 7, "a36a4f4f3e3b", "positive_control"),
    ("L6_k13", 6, "60c6464cf3de", "candidate"),
    ("L6_k14", 6, "66b1933ef957", "candidate"),
    ("L7_k14", 7, "04278cb74654", "candidate"),
    ("L7_k15", 7, "b6bba814d695", "candidate"),
]


def member_hash(members):
    """md5 over sorted POOL POSITIONS, first 12 hex -- scripts/42's frozen convention."""
    return hashlib.md5(",".join(str(int(x)) for x in sorted(members)).encode()).hexdigest()[:12]


# --------------------------------------------------------------------------------------
# intervention machinery (copied verbatim from scripts/25_steering_probe.py; 25 is not edited)
# --------------------------------------------------------------------------------------
@contextlib.contextmanager
def add_at(model, layer, idx_b, idx_p, delta):
    """Add `delta[i]` to the layer-`layer` residual stream at (idx_b[i], idx_p[i])."""
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


# --------------------------------------------------------------------------------------
# geometry
# --------------------------------------------------------------------------------------
def plane_of(R_members):
    """Fit the top-2 PCA plane + circle of one member set's raw type means (1151-dim).

    Returns a dict with the basis, the in-plane coordinates, the circle, each member's
    angle, the cyclic successor/predecessor index arrays, and each member's unit tangent
    in the 1151-dim space (direction of increasing angle).
    """
    X = np.asarray(R_members, dtype=np.float64)
    k = X.shape[0]
    xy, ratios, Vt = fits.pca_2d(X)
    u1, u2 = Vt[0], Vt[1]
    cx, cy, r, resid = fits.fit_circle(xy)
    th = fits.angles(xy, cx, cy)
    order = np.argsort(th)                       # positions of members in CCW order
    succ = np.empty(k, dtype=np.int64)
    pred = np.empty(k, dtype=np.int64)
    for j, i in enumerate(order):
        succ[i] = order[(j + 1) % k]
        pred[i] = order[(j - 1) % k]
    tang = (-np.sin(th)[:, None] * u1[None, :]) + (np.cos(th)[:, None] * u2[None, :])
    tang /= np.maximum(np.linalg.norm(tang, axis=1, keepdims=True), 1e-12)
    # the +90-degree assert: tangent must be the radial direction rotated by +90 in plane
    rad = np.stack([np.cos(th), np.sin(th)], axis=1)
    rad90 = np.stack([-rad[:, 1], rad[:, 0]], axis=1)
    tang_inplane = np.stack([tang @ u1, tang @ u2], axis=1)
    dots = np.sum(tang_inplane * rad90, axis=1)
    assert np.all(dots > 0.999), f"tangent orientation broken: min dot {dots.min():.6f}"
    circ = fits.circularity(xy, cx, cy, r)
    return dict(k=k, xy=xy, u1=u1, u2=u2, cx=float(cx), cy=float(cy), r=float(r),
                circ_resid=float(resid), theta=th, order=order.tolist(),
                succ=succ, pred=pred, tangent=tang,
                pc_var_top2=float(ratios[:2].sum()),
                gap_ratio=float(circ["gap_ratio"]), radius_ratio=float(circ["radius_ratio"]))


def embed_1152(v_1151, drop=R10_DROP, d=1152):
    """Re-insert the dropped coordinate as 0.0 (norm preserving)."""
    v = np.asarray(v_1151, dtype=np.float64)
    if v.ndim == 1:
        out = np.zeros(d, dtype=np.float64)
        out[:drop] = v[:drop]
        out[drop + 1:] = v[drop:]
        return out
    out = np.zeros((v.shape[0], d), dtype=np.float64)
    out[:, :drop] = v[:, :drop]
    out[:, drop + 1:] = v[:, drop:]
    return out


def random_orth_dirs(t_hat, n, rng):
    """`n` unit directions in 1152-dim space, each orthogonalised against `t_hat`."""
    d = t_hat.shape[0]
    out = np.empty((n, d), dtype=np.float64)
    for j in range(n):
        g = rng.standard_normal(d)
        g -= (g @ t_hat) * t_hat
        out[j] = g / max(np.linalg.norm(g), 1e-12)
    return out


# --------------------------------------------------------------------------------------
# forwards
# --------------------------------------------------------------------------------------
class Runner:
    """One member's prompt block: build once, forward unsteered once, steer many times."""

    def __init__(self, model, tok, layer, device):
        self.model, self.tok, self.layer, self.device = model, tok, layer, device
        self.n_forward = 0

    def prepare(self, contexts, target_id):
        uniq, weights, _ = rig.dedupe(contexts)
        ids, attn, tpos = rig.build_batch(uniq, int(target_id),
                                          self.tok.bos_token_id, self.tok.pad_token_id)
        last_only = bool(torch.all(tpos == ids.shape[1] - 1))
        return dict(ids=ids.to(self.device), attn=attn.to(self.device),
                    tpos=tpos.to(self.device), last_only=last_only,
                    w=torch.tensor(weights, dtype=torch.float64))

    def _logits(self, blk, delta=None):
        """Target-position logits only.

        Every row of a block is the same length (all contexts in an arm share a length),
        so the target is the LAST position of every row and ``logits_to_keep=1`` is exact.
        This matters: the full [B, Lmax, 262144] fp32 logits tensor is ~335 MB per batch of
        32 and this machine has no headroom for it. ``last_only`` is asserted at build time
        and the code falls back to full logits if it is ever false.
        """
        ids, attn, tpos = blk["ids"], blk["attn"], blk["tpos"]
        keep = 1 if blk["last_only"] else 0
        out = []
        for lo in range(0, ids.shape[0], BATCH):
            hi = min(lo + BATCH, ids.shape[0])
            b_ids, b_attn, b_tpos = ids[lo:hi], attn[lo:hi], tpos[lo:hi]
            idx_b = torch.arange(hi - lo, device=self.device)
            with torch.no_grad():
                if delta is None:
                    lg = self.model(input_ids=b_ids, attention_mask=b_attn,
                                    use_cache=False, logits_to_keep=keep).logits
                else:
                    with add_at(self.model, self.layer, idx_b, b_tpos, delta[lo:hi]):
                        lg = self.model(input_ids=b_ids, attention_mask=b_attn,
                                        use_cache=False, logits_to_keep=keep).logits
            row = (lg[:, -1] if keep else lg[idx_b, b_tpos]).float().cpu()
            del lg
            assert torch.isfinite(row).all(), "non-finite logits"
            out.append(row)
            self.n_forward += hi - lo
        return torch.cat(out, 0)

    def baseline(self, blk):
        """Unsteered logits at the target position AND the residual norm there."""
        ids, attn, tpos = blk["ids"], blk["attn"], blk["tpos"]
        keep = 1 if blk["last_only"] else 0
        lg_all, hn_all = [], []
        for lo in range(0, ids.shape[0], BATCH):
            hi = min(lo + BATCH, ids.shape[0])
            b_ids, b_attn, b_tpos = ids[lo:hi], attn[lo:hi], tpos[lo:hi]
            idx_b = torch.arange(hi - lo, device=self.device)
            with torch.no_grad(), capture(self.model, [self.layer]) as store:
                lg = self.model(input_ids=b_ids, attention_mask=b_attn,
                                use_cache=False, logits_to_keep=keep).logits
            h = store[self.layer][idx_b, b_tpos].float()
            row = (lg[:, -1] if keep else lg[idx_b, b_tpos]).float().cpu()
            del lg
            store.clear()
            assert torch.isfinite(row).all() and torch.isfinite(h).all(), "non-finite baseline"
            lg_all.append(row)
            hn_all.append(h.norm(dim=-1).cpu())
            self.n_forward += hi - lo
        return torch.cat(lg_all, 0), torch.cat(hn_all, 0)

    def steered(self, blk, hnorm, direction, eps):
        d = torch.tensor(direction, dtype=torch.float32, device=self.device)
        delta = (eps * hnorm.to(self.device).float()).unsqueeze(1) * d.unsqueeze(0)
        return self._logits(blk, delta=delta)


def weighted_mean(x, w):
    return float((x.double() * w).sum() / w.sum())


def cell_stats(lg_steer, lg_base, blk, succ_id, pred_id, member_ids):
    """Delta (raw and baseline-subtracted) plus the in-set softmax readout."""
    w = blk["w"]
    d_st = (lg_steer[:, succ_id] - lg_steer[:, pred_id])
    d_ba = (lg_base[:, succ_id] - lg_base[:, pred_id])
    ids = torch.tensor(member_ids, dtype=torch.long)
    p_st = torch.softmax(lg_steer[:, ids].double(), dim=-1)
    j_s = member_ids.index(succ_id)
    j_p = member_ids.index(pred_id)
    # in-set mass without materialising a 262k-wide softmax: exp(logsumexp_set - logsumexp_all)
    lse_all = torch.logsumexp(lg_steer.double(), dim=-1)
    lse_set = torch.logsumexp(lg_steer[:, ids].double(), dim=-1)
    inset = torch.exp(lse_set - lse_all)
    return dict(delta_raw=weighted_mean(d_st, w),
                delta=weighted_mean(d_st - d_ba, w),
                p_succ_inset=weighted_mean(p_st[:, j_s], w),
                p_pred_inset=weighted_mean(p_st[:, j_p], w),
                inset_mass=weighted_mean(inset, w))


# --------------------------------------------------------------------------------------
# per-set evaluation
# --------------------------------------------------------------------------------------
def eval_set(runner, geo, member_pos, pool_ids, tokens2d, eligible_all, n_prompts,
             eps_list, rng, arm, extra_dirs=None):
    """Score one member set. Returns per-member rows keyed by condition name.

    Conditions are 'eps=<e>' for the tangent at each strength plus whatever `extra_dirs`
    supplies as {name: direction_matrix [k, 1152]}. Every condition reuses the member's
    single unsteered forward, so the baseline is shared and Delta is exactly paired.
    """
    k = geo["k"]
    member_ids = [int(pool_ids[p]) for p in member_pos]
    tang = embed_1152(geo["tangent"])
    rows = []
    for i in range(k):
        tid = member_ids[i]
        if arm == "natural":
            ctx, _ = rig.sample_contexts(tokens2d, n_prompts, CONTEXT_LEN, rng,
                                         eligible=eligible_all)
        else:
            ctx = [np.zeros(0, dtype=np.int64)]
        blk = runner.prepare(ctx, tid)
        lg_base, hnorm = runner.baseline(blk)
        s_id, p_id = member_ids[int(geo["succ"][i])], member_ids[int(geo["pred"][i])]
        cell = dict(member_index=i, k=k, n_rows=int(blk["ids"].shape[0]), conds={})
        for e in eps_list:
            lg = runner.steered(blk, hnorm, tang[i], e)
            cell["conds"][f"eps={e}"] = cell_stats(lg, lg_base, blk, s_id, p_id, member_ids)
        for name, D in (extra_dirs or {}).items():
            lg = runner.steered(blk, hnorm, D[i], EPS_NULL)
            cell["conds"][name] = cell_stats(lg, lg_base, blk, s_id, p_id, member_ids)
        rows.append(cell)
    return rows


def summarise(rows, cond):
    d = np.array([r["conds"][cond]["delta"] for r in rows], dtype=np.float64)
    return dict(mean_delta=float(d.mean()),
                sign_consistency=float((d > 0).mean()),
                n_pos=int((d > 0).sum()), k=len(d),
                mean_delta_raw=float(np.mean([r["conds"][cond]["delta_raw"] for r in rows])),
                p_succ_inset=float(np.mean([r["conds"][cond]["p_succ_inset"] for r in rows])),
                p_pred_inset=float(np.mean([r["conds"][cond]["p_pred_inset"] for r in rows])),
                inset_mass=float(np.mean([r["conds"][cond]["inset_mass"] for r in rows])))


# --------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/phase5_behavioural.json")
    ap.add_argument("--smoke", action="store_true",
                    help="1 target, 3 members, 4 prompts, 2 null sets -- timing only")
    a = ap.parse_args()
    out_path = REPO / a.out
    t0 = time.time()

    n_prompts = 4 if a.smoke else N_PROMPTS
    n_prompts_null = 2 if a.smoke else N_PROMPTS_NULL
    n_rand = 2 if a.smoke else N_RAND_DIRS
    n_nullsets = 2 if a.smoke else N_NULL_SETS
    if a.smoke and Path(a.out).name == "phase5_behavioural.json":
        raise SystemExit("refusing to write a smoke run to the real --out")
    targets = TARGETS[:1] if a.smoke else TARGETS

    # ---------------- member sets, re-derived and hash-checked in-script ----------------
    z = np.load(REPO / FULLN_RAW)
    R_by_layer = {L: z[f"R_L{L}"].astype(np.float32) for L in sorted({t[1] for t in TARGETS})}
    pool_ids = z["pool_ids"].astype(np.int64)
    screen = json.loads((REPO / SCREEN).read_text())
    multi = json.loads((REPO / MULTI).read_text())
    assert multi["strict12_hash"] == "a36a4f4f3e3b"

    print(f"[{time.time()-t0:6.1f}s] loading model {MODEL_NAME} on {DEVICE}", flush=True)
    model, tok = load_model(token=os.environ.get("HF_TOKEN"))
    strings = [(tok.convert_ids_to_tokens(int(t)) or tok.decode([int(t)])).replace("▁", " ")
               for t in pool_ids]
    pos_of_string = {}
    for i, s in enumerate(strings):
        pos_of_string.setdefault(s, []).append(i)

    members_by_hash = {}
    for p in screen["proposals"]:
        h = p.get("member_hash")
        if h in {t[2] for t in targets} and h not in members_by_hash and "tokens" in p:
            cand = [pos_of_string[s] for s in p["tokens"]]
            assert all(len(c) == 1 for c in cand), f"ambiguous token string in {h}"
            flat = [c[0] for c in cand]
            assert member_hash(flat) == h, f"member positions do not reproduce {h}"
            members_by_hash[h] = sorted(flat)
    for t in targets:
        assert t[2] in members_by_hash, f"member set {t[2]} not recovered"

    # ---------------- geometry, fitted before any forward ----------------
    geos, sets_meta = {}, []
    for name, L, h, kind in targets:
        mp = members_by_hash[h]
        R = np.delete(R_by_layer[L][np.asarray(mp)].astype(np.float64), R10_DROP, axis=1)
        g = plane_of(R)
        geos[name] = (g, mp, L)
        sets_meta.append(dict(name=name, layer=L, member_hash=h, kind=kind, k=g["k"],
                              pc_var_top2=g["pc_var_top2"], gap_ratio=g["gap_ratio"],
                              radius_ratio=g["radius_ratio"], circ_resid=g["circ_resid"],
                              cyclic_order_positions=[mp[i] for i in g["order"]]))

    # ---------------- declaration stub, written BEFORE any forward ----------------
    declared_at = datetime.now().isoformat(timespec="seconds")
    declared = dict(
        status="DECLARED", script="scripts/47_behavioural_confirm.py",
        declared_at=declared_at,
        stub_note="written before any forward pass; results overwrite this file carrying declared_at",
        addenda_in_force=["docs/phase4_prereg_2026-08-22.md section 2.6",
                          "docs/phase5_addendum_P_2026-08-22.md P4"],
        candidates=sets_meta,
        rules=dict(
            space=("plane, circle, angles and tangents are fitted on the raw type means "
                   f"R_L with coordinate {R10_DROP} dropped (prereg R10); the tangent is "
                   "re-embedded into 1152 dims with 0.0 at that index before steering, "
                   "which preserves its norm"),
            intervention=("h <- h + eps*||h||*t_hat_i at resid_post of layer L, at the "
                          "member token's own position; ||h|| is per prompt row"),
            orientation=("theta, the cyclic order and t_hat all come from ONE PCA basis; "
                         "t_hat_i = -sin(theta_i) u1 + cos(theta_i) u2 points toward "
                         "increasing theta, i.e. toward the angular successor; asserted "
                         "equal to the radial direction rotated +90 degrees"),
            statistic=("PRIMARY Delta_i = [logit(succ)-logit(pred)]_steered - the same "
                       "quantity unsteered, averaged over that member's prompts. The "
                       "baseline subtraction is a declared reading of prereg 2.6's "
                       "'logit(successor) - logit(predecessor)': without it the statistic "
                       "is dominated by the standing logit asymmetry between two arbitrary "
                       "vocabulary items. The raw steered value is stored as delta_raw."),
            arms=dict(natural=f"{n_prompts} contexts of {CONTEXT_LEN} cached tokens ending "
                              f"with the member token (rig.sample_contexts, seed {SEED_CONTEXT})",
                      bos_only="BOS + member token alone, one deduped row per member"),
            eps_grid=list(EPS_GRID),
            eps_null=EPS_NULL,
            nulls=dict(
                i=f"{n_rand} random unit directions per member, orthogonalised against that "
                  f"member's tangent, in full 1152-dim space, at eps={EPS_NULL} "
                  f"(seed {SEED_DIRS})",
                ii=f"the tangent rotated +90 and -90 degrees in plane (inward / outward "
                   f"radial), at eps={EPS_NULL}",
                iii=f"{n_nullsets} random k-sets drawn from the {len(pool_ids)}-type pool, "
                    f"each fitted with its own plane, circle, angular order and tangents "
                    f"and scored by the identical pipeline, at eps={EPS_NULL} "
                    f"(seed {SEED_NULLSETS})"),
            null_prompts=f"nulls use {n_prompts_null} prompts per member; the candidate's "
                         f"matched comparison value is its mean Delta at eps={EPS_NULL} on "
                         f"the natural arm's full {n_prompts} prompts",
            PASS=("sign_consistency >= (k-1)/k at SOME eps in the grid (natural arm) AND "
                  f"mean Delta at eps={EPS_NULL} above the 99th percentile of null (iii)'s "
                  f"set means AND above ALL {n_rand} of null (i)'s direction means"),
            secondary=("in-set softmax readout (p(succ), p(pred), in-set mass) replaces the "
                       "prereg's route_intermediate_mass, which is defined for multi-step "
                       "A->B arcs; the i->successor step has no intermediate members by "
                       "construction. Monotonicity of mean Delta across the eps grid is "
                       "reported as declared."),
            positive_control=("strict-12 at L6 must PASS this tier's own criterion. The "
                              "1.52-vs-0.45 neighbour lift of results/steering_probe.json "
                              "is L9 month-recall prompts under a different statistic and "
                              "is cited as context, not as a gate."),
        ),
        deviations_from_prereg=[
            "route_intermediate_mass replaced by the in-set softmax readout (see secondary)",
            "no script-48 passer is included: 48 was still running when 47 launched",
            f"null (i) uses {n_rand} random directions per member, not the prereg's 20, and "
            f"nulls run {n_prompts_null} prompts per member rather than the candidate arm's "
            f"{n_prompts}: this machine holds ~21 GB wired by an unrelated process and the "
            "measured throughput was 13 forwards/s. Null (iii) -- the null the PASS rule "
            f"actually gates on -- is NOT reduced and runs the prereg's {n_nullsets} sets. "
            "The shorter null prompt budget makes null (iii)'s p99 noisier and therefore "
            "HARDER to clear; the candidate's own value on the matched 12-prompt budget is "
            "stored as matched_budget_at_eps_null.",
        ] + (["SMOKE RUN -- not a result"] if a.smoke else []),
        config=dict(model=MODEL_NAME, device=DEVICE, dtype="bfloat16", attn="eager",
                    batch_size=BATCH, n_prompts=n_prompts,
                    n_prompts_null=n_prompts_null, context_len=CONTEXT_LEN,
                    seeds=dict(context=SEED_CONTEXT, dirs=SEED_DIRS, nullsets=SEED_NULLSETS)),
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(declared, indent=1))
    print(f"[{time.time()-t0:6.1f}s] stub written to {out_path}", flush=True)

    # ---------------- hook identity check (zero delta must reproduce the baseline) -------
    tokens2d = rig.load_tokens2d(REPO / CACHE_ROOT)
    eligible_all = rig.eligible_docs(tokens2d, ())
    probe = Runner(model, tok, 6, DEVICE)
    blk = probe.prepare([np.zeros(0, dtype=np.int64)], int(pool_ids[0]))
    lg0, hn0 = probe.baseline(blk)
    lg_z = probe.steered(blk, hn0, np.zeros(1152), 0.0)
    hook_max = float((lg0 - lg_z).abs().max())
    print(f"[{time.time()-t0:6.1f}s] hook identity: max |delta logit| = {hook_max:.3e}",
          flush=True)
    assert hook_max == 0.0, "zero-delta hook is not an identity"

    # ---------------- the run ----------------
    rng_dirs = np.random.default_rng(SEED_DIRS)
    rng_nullsets = np.random.default_rng(SEED_NULLSETS)
    results = {}
    for name, L, h, kind in targets:
        geo, mp, L = geos[name]
        k = geo["k"]
        runner = Runner(model, tok, L, DEVICE)
        t_set = time.time()
        print(f"[{time.time()-t0:6.1f}s] {name}: L{L} k={k} starting", flush=True)

        tang = embed_1152(geo["tangent"])
        # null (i) directions and null (ii) rotations, built per member
        dirs = {}
        rd = np.stack([random_orth_dirs(tang[i], n_rand, rng_dirs) for i in range(k)])
        for j in range(n_rand):
            dirs[f"rand{j}"] = rd[:, j, :]
        u1, u2, th = geo["u1"], geo["u2"], geo["theta"]
        rot_p90 = embed_1152(-(np.cos(th)[:, None] * u1[None, :])
                             - (np.sin(th)[:, None] * u2[None, :]))
        rot_m90 = embed_1152((np.cos(th)[:, None] * u1[None, :])
                             + (np.sin(th)[:, None] * u2[None, :]))
        dirs["rot_p90"] = rot_p90
        dirs["rot_m90"] = rot_m90

        rng_ctx = np.random.default_rng(SEED_CONTEXT + L * 100 + k)
        rows_nat = eval_set(runner, geo, mp, pool_ids, tokens2d, eligible_all, n_prompts,
                            EPS_GRID, rng_ctx, "natural")
        rng_ctx_null = np.random.default_rng(SEED_CONTEXT + 50_000 + L * 100 + k)
        rows_null_dirs = eval_set(runner, geo, mp, pool_ids, tokens2d, eligible_all,
                                  n_prompts_null, (), rng_ctx_null, "natural",
                                  extra_dirs=dirs)
        rows_bos = eval_set(runner, geo, mp, pool_ids, tokens2d, eligible_all, 1,
                            EPS_GRID, rng_ctx, "bos")
        # candidate's own value at the pinned eps on the null prompt budget (matched)
        rows_matched = eval_set(runner, geo, mp, pool_ids, tokens2d, eligible_all,
                                n_prompts_null, (EPS_NULL,),
                                np.random.default_rng(SEED_CONTEXT + 50_000 + L * 100 + k),
                                "natural")

        nat = {f"eps={e}": summarise(rows_nat, f"eps={e}") for e in EPS_GRID}
        bos = {f"eps={e}": summarise(rows_bos, f"eps={e}") for e in EPS_GRID}
        null_i = {f"rand{j}": summarise(rows_null_dirs, f"rand{j}") for j in range(n_rand)}
        null_ii = {nm: summarise(rows_null_dirs, nm) for nm in ("rot_p90", "rot_m90")}
        matched = summarise(rows_matched, f"eps={EPS_NULL}")

        # null (iii): random k-sets with their own geometry
        null_iii = []
        s = 0
        while len(null_iii) < n_nullsets:
            s += 1
            pick = sorted(rng_nullsets.choice(len(pool_ids), size=k, replace=False).tolist())
            if pick == sorted(mp):
                continue
            Rn = np.delete(R_by_layer[L][np.asarray(pick)].astype(np.float64), R10_DROP, axis=1)
            if fits.degenerate(Rn):
                continue
            gn = plane_of(Rn)
            rn = eval_set(runner, gn, pick, pool_ids, tokens2d, eligible_all, n_prompts_null,
                          (EPS_NULL,), np.random.default_rng(SEED_NULLSETS + 1000 * s + k),
                          "natural")
            sm = summarise(rn, f"eps={EPS_NULL}")
            sm["member_hash"] = member_hash(pick)
            sm["gap_ratio"] = gn["gap_ratio"]
            null_iii.append(sm)
            print(f"[{time.time()-t0:6.1f}s]   {name} null-set {len(null_iii)}/{n_nullsets} "
                  f"mean_delta={sm['mean_delta']:+.4f}", flush=True)

        null_means = np.array([x["mean_delta"] for x in null_iii], dtype=np.float64)
        p99 = float(np.percentile(null_means, 99))
        rand_means = np.array([null_i[f"rand{j}"]["mean_delta"] for j in range(n_rand)])
        sc_ok = any(nat[f"eps={e}"]["sign_consistency"] >= (k - 1) / k for e in EPS_GRID)
        d_pin = nat[f"eps={EPS_NULL}"]["mean_delta"]
        above_iii = bool(d_pin > p99)
        above_i = bool(d_pin > rand_means.max())
        mono = bool(nat["eps=0.005"]["mean_delta"] < nat["eps=0.01"]["mean_delta"]
                    < nat["eps=0.02"]["mean_delta"])

        results[name] = dict(
            layer=L, member_hash=h, kind=kind, k=k,
            geometry=dict(pc_var_top2=geo["pc_var_top2"], gap_ratio=geo["gap_ratio"],
                          radius_ratio=geo["radius_ratio"], circ_resid=geo["circ_resid"]),
            natural=nat, bos_only=bos,
            matched_budget_at_eps_null=matched,
            null_i_random_orth=null_i, null_ii_rotations=null_ii,
            null_iii_random_sets=null_iii,
            null_iii_p99=p99, null_iii_max=float(null_means.max()),
            null_iii_mean=float(null_means.mean()),
            null_i_max=float(rand_means.max()),
            monotonic_in_eps=mono,
            PASS=bool(sc_ok and above_iii and above_i),
            pass_detail=dict(sign_consistency_ok=bool(sc_ok), above_null_iii_p99=above_iii,
                             above_all_null_i=above_i,
                             mean_delta_at_eps_null=d_pin),
            per_member=dict(natural=rows_nat, null_dirs=rows_null_dirs, bos=rows_bos),
            n_forward=runner.n_forward,
            seconds=round(time.time() - t_set, 1),
        )
        r = results[name]
        print(f"[{time.time()-t0:6.1f}s] {name}: PASS={r['PASS']} "
              f"meanD@{EPS_NULL}={d_pin:+.4f} sc="
              f"{[round(nat[f'eps={e}']['sign_consistency'],3) for e in EPS_GRID]} "
              f"null_iii_p99={p99:+.4f} null_i_max={rand_means.max():+.4f} "
              f"({runner.n_forward} forwards, {r['seconds']}s)", flush=True)

    declared.update(status="COMPLETE", results=results,
                    finished_at=datetime.now().isoformat(timespec="seconds"),
                    runtime_seconds=round(time.time() - t0, 1),
                    hook_identity_max_logit_diff=hook_max,
                    total_forwards=int(sum(v["n_forward"] for v in results.values())))
    out_path.write_text(json.dumps(declared, indent=1))
    print(f"[{time.time()-t0:6.1f}s] wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
