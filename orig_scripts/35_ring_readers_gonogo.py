"""PHASE 3, STAGE 1 -- ring-readers go/no-go (docs/phase3_prereg_2026-08-21.md section 2.2).

Registered by that document, which is FROZEN. This script does not get to choose its
population, its statistics, its nulls or its verdict rule; they are reproduced below so a
reader can check that nothing drifted between the freeze and the run.

Three things are measured, none of which searches:

  (A)  SAE-FREE per-token-type mean residuals. Every closure number this project owns over
       token types (results/token_type_geometry.json) was computed on mean SAE CODES
       projected through w_dec. That is an SAE-mediated object: if the ring were an artifact
       of the dictionary, every one of those numbers would still look the same. So the
       first 2,000 cached sequences are re-forwarded and the RAW residual stream is averaged
       per token type at layers 4-9. GO/NO-GO (2) of avenue 5 is whether the month ring is
       in that artifact at all. It is also the input Stage 4 (scripts/40) would consume, so
       it is written to disk as its own npz before any screening number exists.

  (A') FIDELITY. Stage 3 (scripts/39) reads per-occurrence rows out of the cache and
       reconstructs them through w_dec. If that reconstruction is poor, a Stage-3 negative
       is a statement about the SAE and not about the model. The gate is declared there and
       measured here: median month-row reconstruction cosine >= 0.9 at L7. Support and
       value agreement between a fresh encode and the cached row are reported alongside,
       because a mismatch there would mean the fresh-forward rig and the cache are not
       looking at the same object.

  (B)  WEIGHTS x KNOWN PLANE. No search, no discovery, no BH over a proposer family: the
       month plane is the one scripts/27 already anchors (MT16 label means, top-2 PCA), and
       every (layer, query head) in the model is scored against it once. This is the
       headline NEGATIVE arm of avenue 5 computed directly -- if nothing couples the plane
       here, the census in Stage 4 is asking a question whose answer is already visible.

===========================================================================
DECLARED BEFORE ANY FORWARD PASS (written to --out as a stub; declared_at is carried
into the final JSON, and the prereg's R1 makes that stub's mtime the audit anchor)
===========================================================================

D1  POPULATION (A). `build_pool` of scripts/22_token_type_geometry.py, copied verbatim
    because that module is a script and not importable: token types with corpus count
    >= 100 in cache/sparse_big/layer_0/tokens.i32. 3,795 types. Sequences are the FIRST
    --n-seq rows of tokens.i32 reshaped (15696, 255), fed as [BOS] + row, bf16, eager,
    use_cache=False, which is byte-for-byte how scripts/06_cache_sparse.py fed them.
    Positions 1..255 are accumulated; position 0 is BOS and is excluded, exactly as the
    cache build excluded it.

D2  THE SUBSAMPLE CHANGES THE BACKGROUND POOL, AND THAT IS DECLARED HERE. Only types that
    actually occur in the --n-seq forwarded sequences have a mean. A type with count >= 100
    over the full 15,696 sequences may occur 0 times in the first 2,000. Its row would be
    identically zero, and a random same-size draw containing several such rows is not a
    background sample of ring geometry, it is a sample of coincident points. The background
    pool is therefore restricted to pool types with count_seen >= 20 in the subsample, and
    that pool's size is reported. No full-3,795 background arm is run; including one would
    be reporting noise as if it were annotation.

D3  SETS SCORED (A). Both are frozen objects from prereg section 0, not choices made here:
      strict-12 -- the highest-count in-pool id per month from results/token_map.json,
                   selected by the identical rule scripts/22 uses (`pack`, sort by -count).
                   VERDICT-BEARING.
      13-member -- the frozen blind-flagged cluster, member_hash 7d1d75a303d6, the 12 month
                   names plus " Feb" (5656). Reported, not verdict-bearing. Recorded now
                   because it matters: " Feb" is rare, so in a 2,000-sequence subsample its
                   support is an order of magnitude below the other twelve. Its count_seen
                   is reported next to every number it enters.

D4  STATISTICS (A). `screen`, `gap_null`, `gap_p`, `background_for_size`, `p_bg_of` copied
    verbatim from scripts/22_token_type_geometry.py. Closure p from the uniform-angle
    max-gap null at --n-perm permutations; p_bg from --n-bg random same-size subsets of the
    D2 pool, each screened through the identical path at 2,000 permutations, seed = layer.

D5  GO/NO-GO (2), the funding gate for Stage 4 (prereg section 6): PASS iff p_closed < 0.05
    AND p_bg < 0.05 at L6 or L7 for the STRICT-12 set. The 13-member set and layers 4, 5,
    8, 9 are reported for context and cannot change the verdict. L8 carries the recorded
    negative control (same member hash, p_closed 0.746 on the SAE artifact).

D6  LABEL TIER (A). `cyclic_order_score` of the recovered angular order against the calendar
    is POST-HOC ANNOTATION under prereg R3. It is computed after the closure verdict, it is
    never combined with it, and it decides nothing.

D7  FIDELITY (A'). First --n-fid sequences, layers 6 and 7. The fresh residual is encoded in
    fp32 on the same device the cache was built on, which is the only way the JumpReLU gate
    can be expected to land on the same side of the threshold. Reported: fraction of rows
    with IDENTICAL nonzero support, median and max |value diff| on the shared support, and
    three reconstruction cosines -- fresh code with b_dec (the prereg 2.2 gate object),
    cached code with b_dec, and cached code WITHOUT b_dec (the object prereg section 4
    actually consumes: "Full-row reconstruction = row code @ w_dec (no b_dec)"). The gate
    is evaluated against all three and the gate-bearing one is named in the output.
    Reported overall and restricted to rows whose token is a strict-12 month id.

D8  PLANE (B). QP_L = top-2 right singular vectors of the centred MT16 label means at
    L in {6,7,8,9}, the recipe of scripts/27_span_262k.py `label_means` (16 templates
    batched per word, LAST non-pad position, mean over templates, bf16 forward / fp32 mean).
    ANCHOR GATE, re-asserted not assumed: |gap_ratio(L9) - 1.5123| <= 0.06. Per the task
    spec this gate is REPORTED and does not stop the run; a failure invalidates every
    Part-B number and is printed as such.

D9  OPERATORS (B). For every layer 0..25 and query head h in 0..3 (num_attention_heads=4,
    num_key_value_heads=1 -- multi-query, so Wk and Wv are SHARED across the four heads):
      M_h  = (diag(1+g_q) Wq_h diag(1+w_in))^T (diag(1+g_k) Wk diag(1+w_in)),  1152x1152
      A_h  = (M_h - M_h^T)/2                      the antisymmetric (rotational) part
      coupling(QP) = ||QP^T A_h QP||_F / ||A_h||_F
    RoPE is taken at relative offset Delta = 0, where the rotation is the identity. That is
    a declared restriction, not an oversight: the Delta-sweep lives in Stage 4.
    W_OV_h = (o_proj[:, 256h:256h+256] @ v_proj) @ diag(1+w_in), 1152x1152, rank <= 256.
      plane energy = ||W_OV_h QP||_F^2 / ||W_OV_h||_F^2
      rotational transport = |arg lambda| of the complex eigenpair whose real/imaginary
      span has the largest sum of cos^2 principal angles with QP.
    The RMSNorms are treated as their diagonal gain (1+w) only; the per-token rescaling is
    data-dependent and cannot enter a weights-only statistic. Declared as a limitation.

D10 NULLS (B). 1,000 random orthonormal 2-planes drawn INSIDE the head's own 256-d read
    subspace (row space of diag(1+g_q) Wq_h diag(1+w_in)), seed 1000*L+h; the reported
    number is the percentile of the real coupling in that null. OV plane energy uses the
    same construction inside the OV IMAGE COLUMN SPACE (task spec, primary); because plane
    energy right-multiplies, a supplementary arm drawing planes in the OV ROW space is
    reported alongside and is explicitly NOT the declared statistic.
    KNOWN ASYMMETRY, declared before the numbers exist: QP is not in general contained in
    the head's read subspace, while every null plane is. The null therefore prices "a plane
    this head can see", which is a HARDER comparison than a plane drawn from R^1152. The
    retention profile below is what makes that asymmetry readable.

D11 MULTIPLICITY (B). BH at q=0.05 over all 104 (layer, head) cells on p = 1 - percentile,
    exactly as the task spec states. The arithmetic of that choice is reported, not
    discovered later: with 1,000 draws the standard add-one empirical p is floored at
    1/1001 = 9.99e-4 while the BH rank-1 threshold is 0.05/104 = 4.81e-4, so NO head could
    ever survive BH on the add-one p. Under the declared statistic p = 1 - percentile, a
    head that beats 999/1000 draws gets p = 1e-3 and needs at least three companions to
    survive; a head that beats 1000/1000 gets p = 0 and survives alone. Both p's are
    reported for every head.

D12 RETENTION PROFILE (B), the salvage from the killed head-subspace rescan: sum of cos^2
    of the two principal angles between QP and (a) the QK read rowspace, (b) the OV image
    column space, against 100 random 256-d subspaces of R^1152 per layer (seed L). Its
    null mean is 2*256/1152 = 0.444 by construction and is asserted in the output.

D13 PLANE-LAYER MAPPING. QP exists only at L in {6,7,8,9}. Layer L is scored against the
    nearest of those; L < 6 uses QP_6 and L > 9 uses QP_9. STATED AS A LIMITATION: for
    early and late layers the plane being tested is not that layer's own plane, so a null
    result there is weaker than a null result at 6-9. The full mapping is written out.

D14 PRE-COMMITTED READING (prereg 2.2, verbatim): 0 heads surviving -> "no single head
    couples the month plane by antisymmetric QK matching or rotational OV transport".
    The MQA caveat is part of the sentence: the negative is about single heads, never
    about circuits.
"""
import argparse, hashlib, json, os, sys, time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HUB_OFFLINE", "1")
sys.path.insert(0, str(REPO / "src"))

from gfdisc.env import DEVICE, D_MODEL                                    # noqa: E402
import torch                                                              # noqa: E402
from gfdisc.model import load_model, capture                              # noqa: E402
from gfdisc.saes import load_sae                                          # noqa: E402
from gfdisc.cache import SparseReader                                     # noqa: E402
from gfdisc.fits import (pca_2d, fit_circle, circularity, angles as ring_angles,
                         degenerate, cyclic_order_score)                   # noqa: E402
from gfdisc.nulls import benjamini_hochberg                               # noqa: E402
from gfdisc.tokens import MONTHS                                          # noqa: E402

# ===========================================================================
# FROZEN CONSTANTS
# ===========================================================================
COUNT_FLOOR = 100              # scripts/22 pool floor -- 3,795 types
COUNT_FLOOR_EXT = 30           # scripts/22 extended floor (carried for pool parity only)
N_PERM_CLUSTER = 20_000        # uniform-angle null for the scored sets
N_PERM_BACKGROUND = 2_000      # uniform-angle null inside the background sweep
N_BACKGROUND_DRAWS = 500       # random same-size subsets
N_PERM_ORDER = 20_000          # permutations inside cyclic_order_score
MIN_COUNT_SEEN = 20            # D2 background-pool floor in the subsample

N_SEQ_TOTAL, SEQ_LEN = 15_696, 255
LAYERS_A = [4, 5, 6, 7, 8, 9]  # 5-9 are the prereg's; 4 is captured as R[L-1] for L5
LAYERS_FID = [6, 7]
PLANE_LAYERS = [6, 7, 8, 9]
GO_NOGO_LAYERS = [6, 7]

# prereg section 0, verbatim
SET13 = [3353, 5142, 5261, 5365, 5505, 5656, 5729, 5857, 6027, 6198, 6243, 6384, 7250]
SET13_HASH = "7d1d75a303d6"
FEB_ID = 5656

# scripts/27_span_262k.py MT16, verbatim (which is scripts/12_closure_depth.py's set)
MT16 = ["The report was published in {}", "She was born in {}", "It rained all {}",
        "The season begins in {}", "We travelled there last {}", "The deadline was {}",
        "Sales peaked in {}", "The festival runs through {}", "He resigned in {}",
        "The contract expires in {}", "Enrolment opens in {}", "The survey ran during {}",
        "Prices fell in {}", "The court ruled in {}", "The harvest starts in {}",
        "Construction began in {}"]
ANCHOR_GAP_RATIO_L9 = 1.5123
ANCHOR_TOL_GAP = 0.06

N_HEADS, N_KV_HEADS, HEAD_DIM, N_LAYERS = 4, 1, 256, 26
FIDELITY_GATE_COS = 0.90


def jsonable(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.bool_,)):
        return bool(o)
    raise TypeError(f"{type(o)} is not JSON serialisable")


def member_hash(members):
    return hashlib.md5(",".join(str(int(x)) for x in sorted(members)).encode()).hexdigest()[:12]


# ===========================================================================
# COPIED VERBATIM from scripts/22_token_type_geometry.py -- that file is a script,
# not a module, so these cannot be imported. Any drift here is a bug.
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


def background_for_size(R, pool_n, k, seed, n_draws=N_BACKGROUND_DRAWS, weights=None):
    """Random same-size token-type subsets through the identical screening."""
    rng = np.random.default_rng(seed)
    gaps, ps, rr, t2 = [], [], [], []
    for _ in range(n_draws):
        if weights is None:
            idx = rng.choice(pool_n, size=k, replace=False)
        else:
            idx = weights(rng, k)
        s = screen(R[idx], N_PERM_BACKGROUND, seed=0)
        if s is None:
            continue
        gaps.append(s["max_gap_deg"]); ps.append(s["p_closed"])
        rr.append(s["radius_ratio"]); t2.append(s["top2_var"])
    gaps = np.asarray(gaps)
    return dict(k=int(k), n_draws=int(n_draws), n_scorable=int(len(gaps)),
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


# ===========================================================================
# PART A -- SAE-free per-token-type mean residuals
# ===========================================================================
def part_a_forwards(model, tok, tokens2d, pool_ids, layers, n_seq, batch):
    """fp64 per-type sums of the raw residual stream at positions 1..255."""
    n_pool = len(pool_ids)
    lut = np.full(int(max(tokens2d.max(), np.max(pool_ids))) + 1, -1, dtype=np.int64)
    lut[pool_ids] = np.arange(n_pool)
    acc = {L: torch.zeros(n_pool, D_MODEL, dtype=torch.float64) for L in layers}
    counts = np.zeros(n_pool, dtype=np.int64)
    bos = int(tok.bos_token_id)
    n_batches = (n_seq + batch - 1) // batch
    times = []
    with torch.inference_mode():
        for bi, lo in enumerate(range(0, n_seq, batch)):
            t0 = time.time()
            blk = tokens2d[lo:min(lo + batch, n_seq)].astype(np.int64)
            ids = np.concatenate([np.full((len(blk), 1), bos, dtype=np.int64), blk], axis=1)
            ids_t = torch.from_numpy(ids).to(DEVICE)
            with capture(model, layers) as st:
                model(ids_t, use_cache=False)
            tix = lut[blk.reshape(-1)]
            m = tix >= 0
            sel = torch.from_numpy(tix[m])
            mt = torch.from_numpy(m)
            for L in layers:
                x = st[L][:, 1:, :].reshape(-1, D_MODEL).to(torch.float32).cpu()
                acc[L].index_add_(0, sel, x[mt].to(torch.float64))
            counts += np.bincount(tix[m], minlength=n_pool)
            del st
            times.append(time.time() - t0)
            if bi % 10 == 0 or bi == n_batches - 1:
                el = float(np.sum(times))
                print(f"  [A] batch {bi+1}/{n_batches}  {times[-1]:.2f}s  "
                      f"mean {el/len(times):.2f}s/batch  eta "
                      f"{(n_batches-bi-1)*el/len(times)/60:.1f} min", flush=True)
    R = {L: (acc[L].numpy() / np.maximum(counts, 1)[:, None]).astype(np.float32)
         for L in layers}
    return R, counts, float(np.mean(times)), float(np.sum(times))


def ring_detail(M, n_perm, seed=0):
    """screen() plus the recovered angles, for the post-hoc label tier only."""
    s = screen(M, n_perm, seed=seed)
    if s is None:
        return None, None
    xy, _, _ = pca_2d(np.asarray(M, dtype=np.float64))
    cx, cy, r, _ = fit_circle(xy)
    return s, ring_angles(xy, cx, cy)


# ===========================================================================
# PART A' -- fidelity of the cached SAE rows against a fresh encode
# ===========================================================================
def part_a_prime(model, tok, tokens2d, n_fid, fid_batch, month_ids, layers=LAYERS_FID,
                 root="cache/sparse_big", hf_token=None):
    saes = {L: load_sae(L, "big", device=DEVICE, token=hf_token) for L in layers}
    rdr = {L: SparseReader(root, L) for L in layers}
    bos = int(tok.bos_token_id)
    month_set = set(int(x) for x in month_ids)
    acc = {L: dict(same_support=[], n_cached=[], n_fresh=[], n_shared=[],
                   med_absdiff=[], max_absdiff=[], cos_fresh_bdec=[],
                   cos_cached_bdec=[], cos_cached_nobdec=[], is_month=[]) for L in layers}
    n_batches = (n_fid + fid_batch - 1) // fid_batch
    times = []
    with torch.inference_mode():
        for bi, lo in enumerate(range(0, n_fid, fid_batch)):
            t0 = time.time()
            hi = min(lo + fid_batch, n_fid)
            blk = tokens2d[lo:hi].astype(np.int64)
            ids = np.concatenate([np.full((len(blk), 1), bos, dtype=np.int64), blk], axis=1)
            ids_t = torch.from_numpy(ids).to(DEVICE)
            with capture(model, layers) as st:
                model(ids_t, use_cache=False)
            rows = np.arange(lo * SEQ_LEN, hi * SEQ_LEN)
            is_month = np.isin(blk.reshape(-1), list(month_set))
            for L in layers:
                sae = saes[L]
                x = st[L][:, 1:, :].reshape(-1, D_MODEL).to(torch.float32)
                code = sae.encode(x)                                   # fp32, on DEVICE
                recon_f = code @ sae.w_dec + sae.b_dec
                # ---- cached rows, densified in sub-chunks -------------------
                ip, ind, val = rdr[L].indptr, rdr[L].indices, rdr[L].values
                recon_c = torch.zeros_like(recon_f)
                recon_c_nb = torch.zeros_like(recon_f)
                cached_cols = []
                CH = 256
                for c0 in range(0, len(rows), CH):
                    c1 = min(c0 + CH, len(rows))
                    a, b = int(ip[rows[c0]]), int(ip[rows[c1 - 1] + 1])
                    cols = np.asarray(ind[a:b], dtype=np.int64)
                    vals = np.asarray(val[a:b], dtype=np.float32)
                    ct = (ip[rows[c0] + 1:rows[c1 - 1] + 2] - ip[rows[c0]:rows[c1 - 1] + 1])
                    rid = np.repeat(np.arange(c1 - c0), ct)
                    off = np.concatenate([[0], np.cumsum(ct)])
                    cached_cols.extend(cols[off[i]:off[i + 1]] for i in range(c1 - c0))
                    dense = torch.zeros(c1 - c0, sae.d_sae, dtype=torch.float32, device=DEVICE)
                    dense[torch.from_numpy(rid).to(DEVICE),
                          torch.from_numpy(cols).to(DEVICE)] = torch.from_numpy(vals).to(DEVICE)
                    recon_c_nb[c0:c1] = dense @ sae.w_dec
                    recon_c[c0:c1] = recon_c_nb[c0:c1] + sae.b_dec
                    del dense

                def cos(a_, b_):
                    return torch.nn.functional.cosine_similarity(a_, b_, dim=1).cpu().numpy()

                acc[L]["cos_fresh_bdec"].append(cos(recon_f, x))
                acc[L]["cos_cached_bdec"].append(cos(recon_c, x))
                acc[L]["cos_cached_nobdec"].append(cos(recon_c_nb, x))
                acc[L]["is_month"].append(is_month.copy())
                nz = torch.nonzero(code, as_tuple=False).cpu().numpy()
                cvals = code[torch.from_numpy(nz[:, 0]).to(DEVICE),
                             torch.from_numpy(nz[:, 1]).to(DEVICE)].cpu().numpy()
                nper = np.bincount(nz[:, 0], minlength=len(rows))
                boff = np.concatenate([[0], np.cumsum(nper)])
                for i in range(len(rows)):
                    fc = nz[boff[i]:boff[i + 1], 1]
                    fv = cvals[boff[i]:boff[i + 1]]
                    cc = cached_cols[i]
                    a, ia, ib = np.intersect1d(cc, fc, assume_unique=True, return_indices=True)
                    cvv = np.asarray(val[int(ip[rows[i]]):int(ip[rows[i] + 1])],
                                     dtype=np.float32)[ia]
                    d = np.abs(cvv - fv[ib]) if len(a) else np.array([np.nan])
                    acc[L]["same_support"].append(len(cc) == len(fc) and
                                                  bool(np.array_equal(np.sort(cc), np.sort(fc))))
                    acc[L]["n_cached"].append(len(cc)); acc[L]["n_fresh"].append(len(fc))
                    acc[L]["n_shared"].append(len(a))
                    acc[L]["med_absdiff"].append(float(np.nanmedian(d)))
                    acc[L]["max_absdiff"].append(float(np.nanmax(d)))
                del code, recon_f, recon_c, recon_c_nb, x
            del st
            times.append(time.time() - t0)
            if bi % 5 == 0 or bi == n_batches - 1:
                print(f"  [A'] batch {bi+1}/{n_batches}  {times[-1]:.2f}s", flush=True)
    out = {}
    for L in layers:
        d = acc[L]
        mo = np.concatenate(d["is_month"])
        row = dict(n_rows=int(len(mo)), n_month_rows=int(mo.sum()),
                   frac_identical_support=float(np.mean(d["same_support"])),
                   median_n_cached=float(np.median(d["n_cached"])),
                   median_n_fresh=float(np.median(d["n_fresh"])),
                   median_n_shared=float(np.median(d["n_shared"])),
                   median_abs_value_diff=float(np.nanmedian(d["med_absdiff"])),
                   max_abs_value_diff=float(np.nanmax(d["max_absdiff"])))
        for key in ("cos_fresh_bdec", "cos_cached_bdec", "cos_cached_nobdec"):
            c = np.concatenate(d[key])
            row[f"median_{key}"] = float(np.median(c))
            row[f"median_{key}_month"] = float(np.median(c[mo])) if mo.any() else None
            row[f"p05_{key}"] = float(np.percentile(c, 5))
        sup = np.asarray(d["same_support"])
        row["frac_identical_support_month"] = float(sup[mo].mean()) if mo.any() else None
        out[str(L)] = row
    return out, times


# ===========================================================================
# PART B -- weights x known plane
# ===========================================================================
def label_means_inline(model, tok, layers, words, templates):
    """scripts/27_span_262k.py `label_means`, using the already-loaded model."""
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    out = {l: [] for l in layers}
    with torch.inference_mode():
        for w in words:
            enc = tok([t.format(w) for t in templates], return_tensors="pt",
                      padding=True).to(DEVICE)
            with capture(model, layers) as st:
                model(**enc, use_cache=False)
            last = enc["attention_mask"].sum(1) - 1
            for l in layers:
                out[l].append(
                    st[l][torch.arange(last.shape[0]), last].float().mean(0).cpu().numpy())
    return {l: np.vstack(v).astype(np.float64) for l, v in out.items()}


def orth_basis(A, tol=1e-9):
    """Orthonormal basis of the ROW space of A (columns of the result span it)."""
    _, s, vt = np.linalg.svd(np.asarray(A, dtype=np.float64), full_matrices=False)
    r = int((s > tol * max(s[0], 1e-300)).sum())
    return np.ascontiguousarray(vt[:r].T), r


def rand_planes_in(basis, n, seed):
    """n random orthonormal 2-planes expressed in `basis` coordinates: (n, d, 2)."""
    rng = np.random.default_rng(seed)
    G = rng.normal(size=(n, basis.shape[1], 2))
    Q, _ = np.linalg.qr(G)
    return Q


def sumcos2(Q1, Q2):
    """Sum of cos^2 of the principal angles between two orthonormal column spans."""
    s = np.linalg.svd(np.asarray(Q1).T @ np.asarray(Q2), compute_uv=False)
    return float((np.clip(s, 0, 1) ** 2).sum())


def part_b(model, QP, plane_layer_of, layers, n_null, n_ret_null, verify_eig=True):
    rows, ret_rows, ln_rows, eig_check = [], [], [], None
    ret_null_mean_theory = 2 * HEAD_DIM / D_MODEL
    for L in layers:
        lay = model.model.layers[L]
        attn = lay.self_attn
        f = lambda t: t.detach().to(torch.float32).cpu().numpy().astype(np.float64)
        w_in = 1.0 + f(lay.input_layernorm.weight)
        g_q = 1.0 + f(attn.q_norm.weight)
        g_k = 1.0 + f(attn.k_norm.weight)
        Wq, Wk = f(attn.q_proj.weight), f(attn.k_proj.weight)
        Wv, Wo = f(attn.v_proj.weight), f(attn.o_proj.weight)
        Kn = (g_k[:, None] * Wk) * w_in[None, :]                     # 256 x 1152
        Vn = Wv * w_in[None, :]                                      # 256 x 1152
        Lp = plane_layer_of[L]
        P = QP[Lp]                                                   # 1152 x 2, orthonormal
        rng_ret = np.random.default_rng(L)
        Sn = [np.linalg.qr(rng_ret.normal(size=(D_MODEL, HEAD_DIM)))[0]
              for _ in range(n_ret_null)]
        ret_null = np.array([sumcos2(s, P) for s in Sn])
        # The input_layernorm gain is SHARED by the four heads, so anything it does to
        # retention is a layer property and not a head property. Recorded so the
        # retention column cannot be misread as a statement about Wq.
        mass = (P ** 2).sum(1)
        gain = np.abs(w_in)
        ln_rows.append(dict(
            layer=int(L), plane_layer=int(Lp),
            plane_weighted_mean_gain=float((mass * gain).sum() / mass.sum()),
            median_gain=float(np.median(gain)), mean_gain=float(gain.mean()),
            plane_participation_ratio=float(1.0 / (mass ** 2).sum()),
            note=("plane_weighted_mean_gain << median_gain means the month plane sits in "
                  "coordinates that input_layernorm de-emphasises relative to the typical "
                  "coordinate; the read subspace of EVERY head at this layer is then "
                  "tilted away from the plane by the diagonal gain alone.")))
        t0 = time.time()
        for h in range(N_HEADS):
            Qn = (g_q[:, None] * Wq[h * HEAD_DIM:(h + 1) * HEAD_DIM, :]) * w_in[None, :]
            M = Qn.T @ Kn                                            # 1152 x 1152
            A = 0.5 * (M - M.T)
            nA = float(np.linalg.norm(A))
            coupling = float(np.linalg.norm(P.T @ A @ P) / max(nA, 1e-300))
            B, rq = orth_basis(Qn)                                   # QK read subspace
            A256 = B.T @ A @ B
            Qd = rand_planes_in(B, n_null, seed=1000 * L + h)
            nullc = np.linalg.norm(np.einsum('nab,bc,ncd->nad', Qd.transpose(0, 2, 1),
                                             A256, Qd), axis=(1, 2)) / max(nA, 1e-300)
            pct = 100.0 * float(np.mean(nullc <= coupling))
            p_decl = 1.0 - pct / 100.0
            p_add1 = float((1 + int((nullc >= coupling).sum())) / (1 + len(nullc)))

            Oh = Wo[:, h * HEAD_DIM:(h + 1) * HEAD_DIM]              # 1152 x 256
            WOV = Oh @ Vn                                            # 1152 x 1152
            nOV2 = float((WOV ** 2).sum())
            ov_energy = float((np.linalg.norm(WOV @ P) ** 2) / max(nOV2, 1e-300))
            C, rc = orth_basis(WOV.T)                                # image / column space
            Rw, rr_ = orth_basis(WOV)                                # row space (supplementary)

            def ov_null(basis, seed):
                G = WOV @ basis
                H = G.T @ G
                Qd2 = rand_planes_in(basis, n_null, seed=seed)
                e = np.einsum('nab,bc,ncd->nad', Qd2.transpose(0, 2, 1), H, Qd2)
                v = np.trace(e, axis1=1, axis2=2) / max(nOV2, 1e-300)
                return 100.0 * float(np.mean(v <= ov_energy)), v

            ov_pct_col, _ = ov_null(C, 1000 * L + h)
            ov_pct_row, _ = ov_null(Rw, 1000 * L + h)

            # complex eigenpairs of WOV via the rank-256 lift: eig(Vn @ Oh), v = Oh u
            S256 = Vn @ Oh
            ev, U = np.linalg.eig(S256)
            best = dict(abs_arg_deg=None, sumcos2=None, abs_lambda=None, n_complex_pairs=0)
            for j in range(len(ev)):
                if ev[j].imag <= 1e-10:          # one member per conjugate pair
                    continue
                best["n_complex_pairs"] += 1
                v = Oh @ U[:, j]
                Z = np.column_stack([v.real, v.imag])
                if np.linalg.matrix_rank(Z, tol=1e-12) < 2:
                    continue
                Qz, _ = np.linalg.qr(Z)
                sc = sumcos2(Qz, P)
                if best["sumcos2"] is None or sc > best["sumcos2"]:
                    best.update(sumcos2=sc,
                                abs_arg_deg=float(abs(np.degrees(np.angle(ev[j])))),
                                abs_lambda=float(abs(ev[j])))
            if verify_eig and eig_check is None:
                ef = np.linalg.eigvals(WOV)
                a = np.sort(np.abs(ef))[::-1][:10]
                b = np.sort(np.abs(ev))[::-1][:10]
                eig_check = dict(layer=int(L), head=int(h),
                                 max_abs_diff_top10=float(np.max(np.abs(a - b))),
                                 top_abs_lambda_full=float(a[0]),
                                 top_abs_lambda_lift=float(b[0]))

            rc_qk = sumcos2(B, P)
            rc_ov = sumcos2(C, P)
            B0, _ = orth_basis(g_q[:, None] * Wq[h * HEAD_DIM:(h + 1) * HEAD_DIM, :])
            rc_qk0 = sumcos2(B0, P)          # supplementary: no diag(1+w_in) factor
            ret_rows.append(dict(
                layer=int(L), head=int(h), plane_layer=int(Lp),
                sumcos2_qk_read=rc_qk, sumcos2_ov_image=rc_ov,
                sumcos2_qk_read_no_inputnorm_supplementary=rc_qk0,
                pct_qk_no_inputnorm_supplementary=100.0 * float(np.mean(ret_null <= rc_qk0)),
                pct_qk=100.0 * float(np.mean(ret_null <= rc_qk)),
                pct_ov=100.0 * float(np.mean(ret_null <= rc_ov)),
                null_mean=float(ret_null.mean()), null_p95=float(np.percentile(ret_null, 95)),
                null_mean_theory=ret_null_mean_theory))
            rows.append(dict(
                layer=int(L), head=int(h), plane_layer=int(Lp),
                coupling=coupling, coupling_percentile=pct,
                p_declared=p_decl, p_addone=p_add1,
                null_mean=float(nullc.mean()), null_max=float(nullc.max()),
                null_p999=float(np.percentile(nullc, 99.9)),
                rank_qk_read=int(rq), rank_ov=int(rc),
                ov_plane_energy=ov_energy,
                ov_percentile_colspace=ov_pct_col,
                ov_percentile_rowspace_supplementary=ov_pct_row,
                ov_eig=best, frob_A=nA, frob_OV2=nOV2))
        print(f"  [B] L{L:<2} 4 heads in {time.time()-t0:.1f}s  "
              f"(plane L{Lp}) max coupling pct "
              f"{max(r['coupling_percentile'] for r in rows[-4:]):.1f}", flush=True)
    return rows, ret_rows, ln_rows, eig_check


# ===========================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/phase3_ring_readers_gonogo.json")
    ap.add_argument("--npz-out", default="results/phase3_token_type_mean_resid.npz")
    ap.add_argument("--root", default="cache/sparse_big")
    ap.add_argument("--n-seq", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--n-fid", type=int, default=200)
    ap.add_argument("--fid-batch", type=int, default=8)
    ap.add_argument("--n-perm", type=int, default=N_PERM_CLUSTER)
    ap.add_argument("--n-bg", type=int, default=N_BACKGROUND_DRAWS)
    ap.add_argument("--n-null", type=int, default=1000)
    ap.add_argument("--n-ret-null", type=int, default=100)
    ap.add_argument("--min-count-seen", type=int, default=MIN_COUNT_SEEN)
    ap.add_argument("--heads-layers", type=int, nargs="+", default=list(range(N_LAYERS)))
    a = ap.parse_args()
    t_start = time.time()

    outp = Path(a.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    out = dict(script="scripts/35_ring_readers_gonogo.py",
               item="Phase 3 Stage 1 -- ring-readers go/no-go (prereg 2.2)",
               prereg="docs/phase3_prereg_2026-08-21.md section 2.2 (FROZEN)",
               declared_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
               declarations=__doc__, config=vars(a), status="STUB -- nothing computed yet")

    def write(o):
        o["runtime_seconds"] = round(time.time() - t_start, 1)
        outp.write_text(json.dumps(o, indent=2, default=jsonable))

    write(out)
    print("=" * 78)
    print("PHASE 3 / 2.2 -- ring readers go/no-go. Declaration stub written to", a.out)
    print(f"declared_at = {out['declared_at']}   device = {DEVICE}")
    print("=" * 78, flush=True)

    # ---- pool, sets, sanity --------------------------------------------------
    root = REPO / a.root
    tok_all, cnt, pool_ids, ext_ids = build_pool(root)
    assert len(tok_all) == N_SEQ_TOTAL * SEQ_LEN, f"tokens.i32 has {len(tok_all)} rows"
    tokens2d = tok_all.reshape(N_SEQ_TOTAL, SEQ_LEN)
    n_seq = min(a.n_seq, N_SEQ_TOTAL)
    pool_set = set(int(x) for x in pool_ids)
    token_map = json.loads((REPO / "results/token_map.json").read_text())
    strict12 = []
    for m in MONTHS:
        ip = sorted((int(x) for x in token_map["months"][m] if int(x) in pool_set),
                    key=lambda x: -int(cnt[x]))
        strict12.append(ip[0] if ip else None)
    assert all(x is not None for x in strict12), "a month has no in-pool id"
    row_of = {int(t): i for i, t in enumerate(pool_ids)}
    # scripts/22 hashes cluster members as POOL ROW INDICES, not token ids; the frozen
    # 7d1d75a303d6 is therefore a joint assertion on the id set AND on the pool itself.
    got_hash = member_hash([row_of[t] for t in SET13])
    assert got_hash == SET13_HASH, f"13-member set hash drifted: {got_hash}"
    print(f"pool {len(pool_ids)} types (count >= {COUNT_FLOOR}); "
          f"strict-12 = {strict12}; 13-set hash {SET13_HASH} OK", flush=True)

    hf_token = os.environ.get("HF_TOKEN")
    model, tok = load_model(token=hf_token)
    print(f"model loaded: {type(model).__name__}; "
          f"attn attrs = {[n for n,_ in model.model.layers[0].self_attn.named_children()]}; "
          f"layer attrs = {[n for n,_ in model.model.layers[0].named_children()]}", flush=True)

    # ================= PART A =================================================
    print(f"\n[A] forwarding {n_seq} sequences x 256 tokens, layers {LAYERS_A}, "
          f"batch {a.batch}", flush=True)
    R, counts, sec_per_batch, secs_a = part_a_forwards(
        model, tok, tokens2d[:n_seq], pool_ids, LAYERS_A, n_seq, a.batch)
    npzp = REPO / a.npz_out
    npzp.parent.mkdir(parents=True, exist_ok=True)
    np.savez(npzp, pool_ids=np.asarray(pool_ids, dtype=np.int64),
             counts=counts, n_seq=np.int64(n_seq),
             layers=np.asarray(LAYERS_A, dtype=np.int64),
             **{f"R_L{L}": R[L] for L in LAYERS_A})
    print(f"[A] wrote {npzp} ({npzp.stat().st_size/1e6:.0f} MB)  "
          f"{sec_per_batch:.2f} s/batch  {secs_a/60:.1f} min", flush=True)

    t_screen = time.time()
    seen_idx = np.flatnonzero(counts >= a.min_count_seen)
    s12_rows = np.array([row_of[t] for t in strict12])
    s13_rows = np.array([row_of[t] for t in SET13])
    per_layer, go = {}, False
    for L in LAYERS_A:
        Rl = R[L].astype(np.float64)
        cell = {}
        for name, rws, ids in (("strict12", s12_rows, strict12), ("set13", s13_rows, SET13)):
            k = len(rws)
            s, ang = ring_detail(Rl[rws], a.n_perm, seed=0)
            bg = (background_for_size(Rl[seen_idx], len(seen_idx), k, seed=L, n_draws=a.n_bg)
                  if len(seen_idx) > k else None)
            pbg = p_bg_of(bg, s["max_gap_deg"]) if (bg is not None and s is not None) else None
            entry = dict(k=k, ids=[int(x) for x in ids],
                         count_seen=[int(counts[r]) for r in rws],
                         screen=s, p_bg=pbg,
                         background=(None if bg is None else
                                     {kk: vv for kk, vv in bg.items() if kk != "_gaps"}),
                         note=None if s is not None else
                         "screen() returned None: set is degenerate in this subsample")
            if name == "strict12" and s is not None:
                order = cyclic_order_score(ang, np.arange(12), 12, n_perm=a.n_perm,
                                           seed=0, full_k=12)
                entry["cyclic_order_vs_calendar_POSTHOC"] = order
                if L in GO_NOGO_LAYERS and s["p_closed"] < 0.05 and pbg is not None \
                        and pbg < 0.05:
                    go = True
            cell[name] = entry
        per_layer[str(L)] = cell
        s = cell["strict12"]["screen"]
        print(f"  [A] L{L}: strict12 " +
              (f"gap_ratio {s['gap_ratio']:.4f} max_gap {s['max_gap_deg']:.2f} "
               f"p_closed {s['p_closed']:.2e} p_bg {cell['strict12']['p_bg']}"
               if s else "not scorable"), flush=True)

    out["part_a"] = dict(
        n_seq=n_seq, n_pool=int(len(pool_ids)),
        n_pool_background=int(len(seen_idx)), min_count_seen=int(a.min_count_seen),
        pool_count_floor=COUNT_FLOOR, n_types_seen_at_all=int((counts > 0).sum()),
        strict12_ids=strict12, set13_ids=SET13, set13_hash=SET13_HASH,
        set13_hash_domain="md5 of the sorted POOL ROW INDICES (scripts/22 convention)",
        feb_id=FEB_ID, feb_count_seen=int(counts[row_of[FEB_ID]]),
        layers=LAYERS_A, npz=str(a.npz_out), per_layer=per_layer,
        go_no_go_2="PASS" if go else "FAIL",
        go_no_go_2_rule=("p_closed < 0.05 AND p_bg < 0.05 at L6 or L7 for the strict-12 "
                         "set; PASS funds scripts/40 (prereg section 6)"),
        sec_per_batch=sec_per_batch, seconds=round(secs_a, 1),
        screen_seconds=round(time.time() - t_screen, 1))
    write(out)
    print(f"\n[A] GO/NO-GO (2) = {out['part_a']['go_no_go_2']}", flush=True)

    # ================= PART A' ================================================
    n_fid = min(a.n_fid, n_seq)
    print(f"\n[A'] fidelity on {n_fid} sequences, layers {LAYERS_FID}", flush=True)
    fid, fid_times = part_a_prime(model, tok, tokens2d[:n_fid], n_fid, a.fid_batch,
                                  strict12, LAYERS_FID, str(root), hf_token)
    gate_key = "median_cos_fresh_bdec_month"
    g7 = fid.get("7", {}).get(gate_key)
    out["part_a_prime"] = dict(
        n_seq=n_fid, layers=LAYERS_FID, per_layer=fid,
        gate="median month-row reconstruction cosine >= 0.9 at L7 (prereg 2.2 / 4)",
        gate_bearing_quantity=gate_key,
        gate_note=("Stage 3 consumes CACHED rows and reconstructs them WITHOUT b_dec "
                   "(prereg section 4), so median_cos_cached_nobdec_month at L7 is the "
                   "number Stage 3's own negative should be read against; all three are "
                   "reported and evaluated."),
        gate_pass={k: (None if fid.get("7", {}).get(k) is None
                       else bool(fid["7"][k] >= FIDELITY_GATE_COS))
                   for k in ("median_cos_fresh_bdec_month",
                             "median_cos_cached_bdec_month",
                             "median_cos_cached_nobdec_month")},
        gate_pass_declared=(None if g7 is None else bool(g7 >= FIDELITY_GATE_COS)),
        sec_per_batch=float(np.mean(fid_times)), seconds=round(float(np.sum(fid_times)), 1))
    write(out)
    for L in LAYERS_FID:
        r = fid[str(L)]
        print(f"  [A'] L{L}: identical support {r['frac_identical_support']:.3f}  "
              f"median|dv| {r['median_abs_value_diff']:.2e}  max|dv| "
              f"{r['max_abs_value_diff']:.2e}  cos(fresh) {r['median_cos_fresh_bdec']:.4f}  "
              f"cos(fresh,month) {r['median_cos_fresh_bdec_month']}", flush=True)

    # ================= PART B =================================================
    print(f"\n[B] month plane from MT16 label means at {PLANE_LAYERS}", flush=True)
    t0 = time.time()
    means = label_means_inline(model, tok, PLANE_LAYERS, MONTHS, MT16)
    QP, planes = {}, {}
    for L in PLANE_LAYERS:
        xy, ratios, Vt = pca_2d(means[L])
        cx, cy, r, resid = fit_circle(xy)
        c = circularity(xy, cx, cy, r)
        QP[L] = np.ascontiguousarray(Vt[:2].T)
        planes[str(L)] = dict(gap_ratio=c["gap_ratio"], max_gap_deg=c["max_gap_deg"],
                              radius_ratio=c["radius_ratio"], top2_var=float(ratios[:2].sum()),
                              p_closed=gap_p(12, np.radians(c["max_gap_deg"]), a.n_perm, 0))
    d9 = abs(planes["9"]["gap_ratio"] - ANCHOR_GAP_RATIO_L9)
    anchor = dict(layer=9, gap_ratio=planes["9"]["gap_ratio"],
                  gap_ratio_recorded=ANCHOR_GAP_RATIO_L9, abs_diff=float(d9),
                  tol=ANCHOR_TOL_GAP, passed=bool(d9 <= ANCHOR_TOL_GAP),
                  rule="REPORTED, does not stop the run (task spec); a failure "
                       "invalidates every Part-B number")
    print(f"  [B] anchor L9 gap_ratio {planes['9']['gap_ratio']:.4f} "
          f"(rec {ANCHOR_GAP_RATIO_L9}) -> {'PASS' if anchor['passed'] else 'FAIL'} "
          f"({time.time()-t0:.0f}s)", flush=True)

    plane_layer_of = {L: min(PLANE_LAYERS, key=lambda p: (abs(p - L), p))
                      for L in range(N_LAYERS)}
    heads_layers = sorted(set(int(x) for x in a.heads_layers))
    rows, ret_rows, ln_rows, eig_check = part_b(model, QP, plane_layer_of, heads_layers,
                                                a.n_null, a.n_ret_null)
    p = np.array([r["p_declared"] for r in rows])
    keep = benjamini_hochberg(p, alpha=0.05) if len(p) else np.zeros(0, dtype=bool)
    for r, k in zip(rows, keep):
        r["bh_pass"] = bool(k)
    n999 = [r for r in rows if r["coupling_percentile"] >= 99.9]
    headline = int(sum(1 for r in n999 if r["bh_pass"]))
    out["part_b"] = dict(
        layers_scored=heads_layers, n_cells=len(rows),
        plane_layer_mapping={str(k): int(v) for k, v in plane_layer_of.items()},
        plane_layer_mapping_limitation=(
            "QP exists only at L in {6,7,8,9}; L<6 is scored against QP_6 and L>9 against "
            "QP_9. At those layers the tested plane is not the layer's own plane, so a "
            "null there is weaker than a null at 6-9."),
        planes=planes, anchor_gate=anchor,
        per_head=rows, retention=ret_rows, input_layernorm_gain=ln_rows,
        eig_lift_check=eig_check,
        n_coupling_pct_ge_99_9=len(n999), n_bh_pass=int(keep.sum()),
        headline_n_heads=headline,
        headline_reading=(
            "no single head couples the month plane by antisymmetric QK matching or "
            "rotational OV transport (MQA caveat: this is a statement about single heads, "
            "never about circuits)") if headline == 0 else
            f"{headline} head(s) exceed the 99.9th percentile of their own null and "
            f"survive BH; each requires the Stage-4 census to be interpreted",
        bh_arithmetic=dict(
            n_cells=len(rows), alpha=0.05,
            bh_rank1_threshold=float(0.05 / max(len(rows), 1)),
            addone_p_floor=float(1.0 / (a.n_null + 1)),
            note=("BH runs on p = 1 - percentile as declared. The add-one empirical p is "
                  "floored at 1/(n_null+1) which exceeds the BH rank-1 threshold at 104 "
                  "cells, so no head could survive BH on that p; both are reported.")),
        limitations=[
            "RoPE is evaluated at relative offset Delta = 0 (identity); the Delta sweep is "
            "Stage 4's, not this script's.",
            "RMSNorms enter only as their diagonal gain (1+w); the per-token rescale is "
            "data-dependent and cannot enter a weights-only statistic.",
            "QP is not in general inside a head's 256-d read subspace while every null "
            "plane is, so the QK null prices 'a plane this head can see' -- a harder "
            "comparison than a plane drawn from R^1152. The retention profile is what "
            "makes that asymmetry readable.",
            "The OV null is drawn in the IMAGE (column) space per the task spec; plane "
            "energy right-multiplies, so the row-space arm is reported as supplementary "
            "and is not the declared statistic.",
            "Wk and Wv are shared across the four query heads (num_key_value_heads=1), so "
            "the four cells of a layer differ only through Wq_h, g_q and o_proj columns.",
            "part_b.input_layernorm_gain must be read before part_b.retention: the "
            "diag(1+w_in) factor is shared by every head at a layer, so if the month "
            "plane occupies low-gain coordinates the read subspaces are tilted away from "
            "it by that diagonal alone. The supplementary "
            "sumcos2_qk_read_no_inputnorm_supplementary column is the same quantity with "
            "the diagonal removed and is what separates the two explanations."],
        seconds=round(time.time() - t0, 1))
    out["status"] = "complete"
    write(out)

    print(f"\n[B] {len(rows)} cells | coupling pct >= 99.9: {len(n999)} | "
          f"BH pass: {int(keep.sum())} | headline: {headline}")
    print(f"    max coupling percentile {max((r['coupling_percentile'] for r in rows), default=0):.2f}")
    print(f"    max retention sumcos2 qk {max((r['sumcos2_qk_read'] for r in ret_rows), default=0):.4f} "
          f"(null mean {ret_rows[0]['null_mean']:.4f})" if ret_rows else "")
    secs = round(time.time() - t_start, 1)
    print(f"\nGO/NO-GO (2) = {out['part_a']['go_no_go_2']}   "
          f"fidelity gate (L7 month, fresh+b_dec) = {out['part_a_prime']['gate_pass_declared']}   "
          f"anchor = {'PASS' if anchor['passed'] else 'FAIL'}")
    print(f"wrote {a.out} ({secs}s)", flush=True)


if __name__ == "__main__":
    main()
