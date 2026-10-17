"""PHASE 3, STAGE 1 -- ring-readers go/no-go, FULL CACHE re-run (Addendum C).

Registered by docs/phase3_prereg_addenda_2026-08-21.md Addendum C, which is FROZEN, and by
docs/phase3_prereg_2026-08-21.md section 2.2 (also FROZEN), whose Part A this repeats at
every cached sequence. This script does not get to choose its population, its statistics,
its nulls or its verdict rule; they are reproduced below so a reader can check that nothing
drifted between the freeze and the run.

This is scripts/35_ring_readers_gonogo.py PART A ONLY, at all 15,696 cached sequences.
Part A' (SAE fidelity) and Part B (weights x known plane) are NOT re-run: their results
stand as recorded in results/phase3_ring_readers_gonogo.json, read under the verify35
caveats registered in Addendum C. The code below is COPIED from scripts/35 rather than
imported from it (scripts/35 is a script, not a module); any drift here is a bug.

===========================================================================
WHY THIS RE-RUN EXISTS (provenance, quoted from Addendum C before any number is computed)
===========================================================================
Addendum C, "Defect (verify35)": section 2.2 Part A used the first 2,000 cached sequences
(790 month occurrences, 48-104 per month). Over exactly those occurrences the SAE mean-code
ring ALSO fails at L6/L7 --

    "L6 2.20/p 0.107; L7 2.82/p 0.43 ... while the full-cache SAE ring closes (L6
     1.646/0.00205; L7 1.602/0.00105); matched resampling of the full-cache SAE ring at the
     2,000-sequence n gives pass rates 0.93 (L6) and 0.66 (L7). The go/no-go (2) FAIL is
     therefore a power artifact and cannot de-fund scripts/40 under R2 ('an instrument that
     cannot re-detect the month object produces uncalibrated negatives')."

The recorded n=2,000 result this supersedes for the go/no-go: strict-12 at L6 gap_ratio
2.2407 / p_closed 0.1228 / p_bg 0.0220, at L7 gap_ratio 2.9371 / p_closed 0.4918 /
p_bg 0.2495, verdict FAIL (results/phase3_ring_readers_gonogo.json, part_a). Nothing about
the decision rule changes here; only n does.

===========================================================================
DECLARED BEFORE ANY FORWARD PASS (written to --out as a stub; declared_at is carried
into the final JSON, and the prereg's R1 makes that stub's mtime the audit anchor)
===========================================================================

D1  POPULATION. `build_pool` of scripts/22_token_type_geometry.py, copied verbatim because
    that module is a script and not importable: token types with corpus count >= 100 in
    cache/sparse_big/layer_0/tokens.i32. 3,795 types. Sequences are the FIRST --n-seq rows
    of tokens.i32 reshaped (15696, 255) and --n-seq DEFAULTS TO 15,696, i.e. ALL of them,
    which is the single change Addendum C makes to Part A. They are fed as [BOS] + row,
    bf16, eager, use_cache=False, which is byte-for-byte how scripts/06_cache_sparse.py fed
    them. Positions 1..255 are accumulated; position 0 is BOS and is excluded, exactly as
    the cache build excluded it. At the default n every month occurrence in the cache is
    counted (Addendum C: 374-777 per month, against 48-104 at n=2,000).

D2  THE BACKGROUND-POOL RULE IS UNCHANGED FROM scripts/35 AND IS RE-DECLARED HERE. Only
    types that actually occur in the --n-seq forwarded sequences have a mean; a type with
    count >= 100 over the full corpus may occur 0 times in a subsample, its row would be
    identically zero, and a random same-size draw containing several such rows is not a
    background sample of ring geometry, it is a sample of coincident points. The background
    pool is therefore restricted to pool types with count_seen >= --min-count-seen (20) in
    the forwarded set, and that pool's size is reported. At --n-seq 15,696 the forwarded set
    IS the corpus, so every pool type has count_seen = its corpus count >= 100 >= 20 and the
    background pool is expected to be the whole 3,795 pool (Addendum C). That expectation is
    MEASURED, not assumed: n_pool_background is reported and compared to 3,795 in the output.

D3  SETS SCORED. Both are frozen objects from prereg section 0, not choices made here:
      strict-12 -- the highest-count in-pool id per month from results/token_map.json,
                   selected by the identical rule scripts/22 uses (`pack`, sort by -count).
                   VERDICT-BEARING.
      13-member -- the frozen blind-flagged cluster, member_hash 7d1d75a303d6, the 12 month
                   names plus " Feb" (5656). Reported, not verdict-bearing. Its count_seen
                   is reported next to every number it enters; at full n " Feb" is no longer
                   the order-of-magnitude-undersampled member it was at n=2,000 (12 seen).

D4  STATISTICS. `screen`, `gap_null`, `gap_p`, `background_for_size`, `p_bg_of` copied
    verbatim from scripts/22_token_type_geometry.py (via scripts/35, unchanged). Closure p
    from the uniform-angle max-gap null at --n-perm permutations (20,000); p_bg from --n-bg
    (500) random same-size subsets of the D2 pool, each screened through the identical path
    at 2,000 permutations, seed = layer. Layers 4-9 are captured in ONE forward pass.

D5  GO/NO-GO (2), the funding gate for Stage 4 (prereg section 6), RULE UNCHANGED: PASS iff
    p_closed < 0.05 AND p_bg < 0.05 at L6 or L7 for the STRICT-12 set. The 13-member set and
    layers 4, 5, 8, 9 are reported for context and cannot change the verdict. L8 carries the
    recorded negative control (same member hash, p_closed 0.746 on the SAE artifact).

D6  LABEL TIER. `cyclic_order_score` of the recovered angular order against the calendar is
    POST-HOC ANNOTATION under prereg R3. It is computed after the closure verdict, it is
    never combined with it, and it decides nothing.

D7  MATCHED-N NOTE (new here, reporting only -- it changes no statistic and no verdict).
    Addendum C's defect is a power defect, so the per-month occurrence counts that set the
    power are reported explicitly: month -> count_seen for the strict-12 ids, their min /
    median / max, and the same for the 13-member set. They are read against the power
    certificate frozen in prereg section 0: "closure pass rate 1.000 at n>=110, 0.99 at
    n=60-90, 0.895 at n=40, 0.76 at n=25". The counts are a property of the forwarded
    population and are IDENTICAL at every layer (one accumulation pass, six captures), so
    the mapping is written once in part_a and each per-layer cell carries a short
    matched_n_note pointing at it plus that layer's own count_seen array.

D8  ARTIFACT. The per-type mean-residual array is written to
    results/phase3_token_type_mean_resid_fulln.npz (a NEW path; the n=2,000
    results/phase3_token_type_mean_resid.npz is not touched) BEFORE any screening number
    exists, and is the input Stage 4 (scripts/40) would consume.
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
from gfdisc.fits import (pca_2d, fit_circle, circularity, angles as ring_angles,
                         degenerate, cyclic_order_score)                   # noqa: E402
from gfdisc.tokens import MONTHS                                          # noqa: E402

# ===========================================================================
# FROZEN CONSTANTS (identical to scripts/35 except N_SEQ default, which is an argparse
# default and not a constant)
# ===========================================================================
COUNT_FLOOR = 100              # scripts/22 pool floor -- 3,795 types
COUNT_FLOOR_EXT = 30           # scripts/22 extended floor (carried for pool parity only)
N_PERM_CLUSTER = 20_000        # uniform-angle null for the scored sets
N_PERM_BACKGROUND = 2_000      # uniform-angle null inside the background sweep
N_BACKGROUND_DRAWS = 500       # random same-size subsets
N_PERM_ORDER = 20_000          # permutations inside cyclic_order_score
MIN_COUNT_SEEN = 20            # D2 background-pool floor in the forwarded set

N_SEQ_TOTAL, SEQ_LEN = 15_696, 255
LAYERS_A = [4, 5, 6, 7, 8, 9]  # 5-9 are the prereg's; 4 is captured as R[L-1] for L5
GO_NOGO_LAYERS = [6, 7]
EXPECTED_BACKGROUND_POOL_FULLN = 3_795          # D2, Addendum C

# prereg section 0, verbatim
SET13 = [3353, 5142, 5261, 5365, 5505, 5656, 5729, 5857, 6027, 6198, 6243, 6384, 7250]
SET13_HASH = "7d1d75a303d6"
FEB_ID = 5656

# prereg section 0, verbatim -- the matched-n power certificate the D7 note is read against
POWER_CERTIFICATE = ("prereg section 0, matched-n power certificate (critic pre-measurement, "
                     "L6, strict-12, pooled positions): closure pass rate 1.000 at n>=110, "
                     "0.99 at n=60-90, 0.895 at n=40, 0.76 at n=25.")

# the n=2,000 numbers this run supersedes for the go/no-go (recorded, not recomputed)
PRIOR_N2000 = dict(source="results/phase3_ring_readers_gonogo.json part_a", n_seq=2000,
                   n_pool_background=2574, go_no_go_2="FAIL",
                   strict12_L6=dict(gap_ratio=2.2407, p_closed=0.12279, p_bg=0.021956),
                   strict12_L7=dict(gap_ratio=2.9371, p_closed=0.49178, p_bg=0.249501),
                   count_seen_range=[44, 104], feb_count_seen=12)


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
# COPIED VERBATIM from scripts/22_token_type_geometry.py (via scripts/35) -- that file is
# a script, not a module, so these cannot be imported. Any drift here is a bug.
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
# PART A -- SAE-free per-token-type mean residuals (copied from scripts/35)
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
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/phase3_ring_readers_fulln.json")
    ap.add_argument("--npz-out", default="results/phase3_token_type_mean_resid_fulln.npz")
    ap.add_argument("--root", default="cache/sparse_big")
    ap.add_argument("--n-seq", type=int, default=N_SEQ_TOTAL)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--n-perm", type=int, default=N_PERM_CLUSTER)
    ap.add_argument("--n-bg", type=int, default=N_BACKGROUND_DRAWS)
    ap.add_argument("--min-count-seen", type=int, default=MIN_COUNT_SEEN)
    a = ap.parse_args()
    t_start = time.time()

    outp = Path(a.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    out = dict(script="scripts/35b_ring_readers_fulln.py",
               item=("Phase 3 Stage 1 -- ring-readers go/no-go, Part A at ALL 15,696 cached "
                     "sequences (prereg 2.2 Part A; Addendum C)"),
               prereg=("docs/phase3_prereg_addenda_2026-08-21.md Addendum C (FROZEN) + "
                       "docs/phase3_prereg_2026-08-21.md section 2.2 (FROZEN)"),
               supersedes_for_go_no_go=PRIOR_N2000,
               parts_not_rerun=("Part A' (SAE fidelity) and Part B (weights x known plane) "
                                "are not re-run; their results stand in "
                                "results/phase3_ring_readers_gonogo.json under the verify35 "
                                "caveats registered in Addendum C."),
               power_certificate=POWER_CERTIFICATE,
               declared_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
               declarations=__doc__, config=vars(a), status="STUB -- nothing computed yet")

    def write(o):
        o["runtime_seconds"] = round(time.time() - t_start, 1)
        outp.write_text(json.dumps(o, indent=2, default=jsonable))

    write(out)
    print("=" * 78)
    print("PHASE 3 / 2.2 Part A at full n (Addendum C). Declaration stub written to", a.out)
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
    print(f"model loaded: {type(model).__name__}", flush=True)

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

    # ---- D7 matched-n note (reporting only; identical at every layer) --------
    s12_counts = [int(counts[r]) for r in s12_rows]
    s13_counts = [int(counts[r]) for r in s13_rows]
    matched_n = dict(
        note=("Occurrence counts of the scored ids in the forwarded population. They set "
              "the power of every closure statistic below and are IDENTICAL at every layer "
              "(one accumulation pass, six captures). Reporting only under D7: they enter "
              "no statistic and no verdict."),
        power_certificate=POWER_CERTIFICATE,
        strict12_per_month={m: int(c) for m, c in zip(MONTHS, s12_counts)},
        strict12_min=int(min(s12_counts)), strict12_median=float(np.median(s12_counts)),
        strict12_max=int(max(s12_counts)),
        strict12_all_ge_110=bool(min(s12_counts) >= 110),
        set13_per_id={str(int(t)): int(counts[row_of[t]]) for t in SET13},
        set13_min=int(min(s13_counts)), set13_max=int(max(s13_counts)),
        prior_n2000_strict12_range=PRIOR_N2000["count_seen_range"],
        prior_n2000_feb_count_seen=PRIOR_N2000["feb_count_seen"])
    print("[A] matched-n: strict-12 per-month counts " +
          ", ".join(f"{m[:3]} {c}" for m, c in zip(MONTHS, s12_counts)), flush=True)
    print(f"[A] matched-n: min {min(s12_counts)} median {np.median(s12_counts):.1f} "
          f"max {max(s12_counts)}; all >= 110: {min(s12_counts) >= 110}  "
          f"(' Feb' 5656 = {counts[row_of[FEB_ID]]})", flush=True)

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
                         matched_n_note=("count_seen above is this layer's own array and is "
                                         "identical at every layer; see part_a.matched_n "
                                         "for the month mapping and the power certificate."),
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

    # D2 is measured, not assumed. Full-n forward means counts must equal the corpus counts.
    full_n = bool(n_seq == N_SEQ_TOTAL)
    counts_match_corpus = (bool(np.array_equal(counts, cnt[pool_ids].astype(np.int64)))
                           if full_n else None)

    out["part_a"] = dict(
        n_seq=n_seq, n_seq_total=N_SEQ_TOTAL, full_cache=full_n,
        n_pool=int(len(pool_ids)),
        n_pool_background=int(len(seen_idx)), min_count_seen=int(a.min_count_seen),
        n_pool_background_expected_fulln=EXPECTED_BACKGROUND_POOL_FULLN,
        n_pool_background_matches_expected=(
            bool(len(seen_idx) == EXPECTED_BACKGROUND_POOL_FULLN) if full_n else None),
        background_pool_rule=("D2, unchanged from scripts/35: pool types with count_seen >= "
                              f"{a.min_count_seen} in the forwarded set"),
        counts_match_corpus=counts_match_corpus,
        counts_match_corpus_note=("At --n-seq 15696 the forwarded set is the whole corpus, so "
                                  "per-type counts must equal bincount(tokens.i32)[pool_ids]; "
                                  "this certifies that no batch or position was dropped. Null "
                                  "when n_seq < 15696."),
        pool_count_floor=COUNT_FLOOR, n_types_seen_at_all=int((counts > 0).sum()),
        strict12_ids=strict12, set13_ids=SET13, set13_hash=SET13_HASH,
        set13_hash_domain="md5 of the sorted POOL ROW INDICES (scripts/22 convention)",
        feb_id=FEB_ID, feb_count_seen=int(counts[row_of[FEB_ID]]),
        matched_n=matched_n,
        layers=LAYERS_A, npz=str(a.npz_out), per_layer=per_layer,
        go_no_go_2="PASS" if go else "FAIL",
        go_no_go_2_rule=("p_closed < 0.05 AND p_bg < 0.05 at L6 or L7 for the strict-12 "
                         "set; PASS funds scripts/40 (prereg section 6). Rule identical to "
                         "scripts/35 D5; only n changed (Addendum C)."),
        sec_per_batch=sec_per_batch, seconds=round(secs_a, 1),
        screen_seconds=round(time.time() - t_screen, 1))
    out["status"] = "complete"
    write(out)

    pa = out["part_a"]
    print(f"\n[A] background pool {pa['n_pool_background']} of {pa['n_pool']} "
          f"(expected at full n: {EXPECTED_BACKGROUND_POOL_FULLN}; match "
          f"{pa['n_pool_background_matches_expected']})  "
          f"counts_match_corpus {pa['counts_match_corpus']}")
    print(f"\nGO/NO-GO (2) = {pa['go_no_go_2']}   (n_seq {n_seq})")
    print(f"wrote {a.out} ({round(time.time() - t_start, 1)}s)", flush=True)


if __name__ == "__main__":
    main()
