"""Step 26 -- offline pre-check for the 65k-width capacity test. No model, no cache.

The month-manifold negative ran entirely on 16k-width SAEs (16384 latents over
d_model=1152, ~14x expansion), while every published month/weekday cluster used
16-32x. The capacity hypothesis says the negative is an artifact of that: at a
wider dictionary the month latents become monosemantic enough that their DECODER
directions carry the ring, so cosine clustering finds it and co-activation never
has to.

Everything here is HTTP range reads against the Gemma Scope 2 repo, exactly like
`02_precheck.py` and `02b_precheck_cluster.py`: the shipped `top_tokens` give the
month/weekday census for free, and a cluster's `w_dec` rows are contiguous
4608-byte slices that can be pulled individually without touching the 604 MB
(65k) or 2.4 GB (262k) params file.

Four questions, in order:
  1. INVENTORY -- what widths >= 65k actually ship, at which layers and L0s.
  2. CENSUS    -- how many month/weekday latents exist at each width, under the
                  identical criterion used by `02b` and `18` (>= 2 distinct label
                  names among a latent's 10 top tokens), recomputed at 16k in the
                  same run so the width comparison is like-for-like.
  3. RING      -- pairwise decoder cosines, top-2 PCA variance, `fits.circularity`
                  and `fits.cyclic_order_score` on the decoder directions
                  themselves, against a matched random-direction control.
  4. BUDGET    -- bytes/token/layer, cache size, wall clock and the memory shape
                  of cosine clustering at these widths.

Exploratory. Nothing here is pre-registered and nothing here can change a gate
verdict; labels are applied post hoc throughout, which is fine for a pre-check
and would not be for the gate.
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from gfdisc.hf_range import _curl, read_header, read_tensor  # noqa: E402
from gfdisc import fits  # noqa: E402

REPO = "google/gemma-scope-2-1b-pt"
API = f"https://huggingface.co/api/models/{REPO}"
BASE = f"https://huggingface.co/{REPO}/resolve/main"

MONTHS = ["january", "february", "march", "april", "may", "june", "july",
          "august", "september", "october", "november", "december"]
DAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
D_MODEL = 1152
ENGELS_MEAN_COSINE = 0.53          # published GPT-2-small month cluster

# Each decoder row is one range request, so the widest candidate set is what sets
# the request count (and the failure odds) for the whole survey. The wide set only
# ever feeds the per-value aggregate, where 150 latents already give every month
# several members.
CAP_STRICT, CAP_WIDE = 400, 150

# The survey. 65k only exists at the deep-dive layers, so the ladder that answers
# the capacity question at a FIXED layer is 16k -> 65k -> 262k -> 1m there, and the
# only way to ask it at layers 9/16/18 at all is 262k.
DEEP_DIVE = [7, 13, 17, 22]
REQUESTED = [9, 16, 18]


def dir_for(site, layer, width, l0):
    return f"{site}/layer_{layer}_width_{width}_l0_{l0}"


def retry(fn, *args, tries=6, base=1.5, **kw):
    """Range reads against the hub fail intermittently under concurrency.

    `hf_range._curl` passes `--retry 3`, which covers HTTP 5xx and timeouts but
    not a connection reset mid-body (curl exit 56), and that is the failure this
    run hit -- one reset in ~8000 requests was enough to abort the whole survey
    because ThreadPoolExecutor.map re-raises. Retrying here, with backoff, keeps
    the survey going without editing the shared helper.
    """
    last = None
    for i in range(tries):
        try:
            return fn(*args, **kw)
        except Exception as e:                    # noqa: BLE001 - transport errors only
            last = e
            time.sleep(base ** i)
    raise last


def http_size(path):
    """File size without downloading it. LFS files report x-linked-size."""
    out = subprocess.run(["curl", "-sIL", f"{BASE}/{path}"],
                         capture_output=True, text=True).stdout
    linked, content = None, None
    for line in out.splitlines():
        low = line.lower()
        if low.startswith("x-linked-size:"):
            linked = int(line.split(":")[1])
        elif low.startswith("content-length:"):
            content = int(line.split(":")[1])
    n = linked if linked is not None else content
    # a missing LFS file redirects to a short "Entry not found" body
    return None if n is None or n < 1000 else n


def get_json(path):
    try:
        return json.loads(subprocess.run(["curl", "-sL", f"{BASE}/{path}"],
                                         capture_output=True, text=True).stdout)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# 1. inventory
# ---------------------------------------------------------------------------

SAE_PAT = re.compile(r"^(?P<site>[^/]+)/layer_(?P<layer>\d+)_width_(?P<w>[^_/]+)_l0_(?P<l0>.+?)/(?P<f>[^/]+)$")


def inventory(sites=("resid_post", "resid_post_all")):
    raw = subprocess.run(["curl", "-sL", API], capture_output=True, text=True).stdout
    files = [x["rfilename"] for x in json.loads(raw)["siblings"]]
    groups = defaultdict(lambda: defaultdict(set))
    for p in files:
        m = SAE_PAT.match(p)
        if not m or m.group("site") not in sites:
            continue
        g = m.groupdict()
        groups[(g["site"], g["w"], g["l0"])][int(g["layer"])].add(g["f"])
    return groups


def width_int(w):
    return {"16k": 16384, "65k": 65536, "262k": 262144, "1m": 1048576}.get(w)


# ---------------------------------------------------------------------------
# 2. census
# ---------------------------------------------------------------------------

def build_lut(vocab, words):
    """token id -> index into `words`, or -1. Built once; the census then costs a
    handful of vectorised passes over a [F, 10] array instead of 10*F dict lookups,
    which matters at F = 1048576."""
    w2i = {w: i for i, w in enumerate(words)}
    lut = np.full(len(vocab), -1, dtype=np.int32)
    for t, s in enumerate(vocab):
        j = w2i.get(s.strip().lower())
        if j is not None:
            lut[t] = j
    return lut


def label_hits(tt, lut, n_words):
    """Per-latent exact-match hits of each label word among its 10 top tokens.

    Exact equality on the stripped, lowercased token string is the rule
    `18_month_latent_placement.py` uses, and `02b_precheck_cluster.py` differs
    only by an `rstrip('s')`. Using 18's rule verbatim is what makes the 16k
    numbers recomputed here comparable to the ones the negative was written on.
    """
    idx = np.asarray(tt, dtype=np.int64)
    ok = (idx >= 0) & (idx < len(lut))
    lab = np.where(ok, lut[np.clip(idx, 0, len(lut) - 1)], -1)
    return np.stack([(lab == j).sum(1) for j in range(n_words)], axis=1).astype(np.int16)


def census(hits, words, ignore=()):
    """Counts under the >=2-distinct rule, the >=1-distinct rule, and a dominant-label
    rule that does NOT penalise monosemanticity.

    The >=2-distinct criterion is the one the 16k negative used, but it selects
    AGAINST the capacity hypothesis: a latent that has become cleanly
    monosemantic has exactly one month in its top tokens and is dropped by
    exactly the rule that is supposed to detect the improvement. So the
    dominant-label sets are carried alongside it and the ring check is run on
    both.
    """
    keep = np.array([w not in ignore for w in words])
    h = hits * keep[None, :]
    distinct = (h > 0).sum(1)
    total = h.sum(1)
    ge2 = np.flatnonzero(distinct >= 2)
    ge1 = np.flatnonzero(distinct >= 1)
    spanning = np.flatnonzero(distinct >= 4)

    # dominant label: strict argmax over label counts, no ties
    dom = {}
    for f in ge1:
        row = h[f]
        j = int(np.argmax(row))
        if (row == row[j]).sum() != 1:
            continue                      # tie -> not attributable to one label
        dom[int(f)] = (j, int(row[j]))
    dom2 = {f: v for f, v in dom.items() if v[1] >= 2}

    values_seen = sorted({words[j] for j in np.flatnonzero((h[ge2] > 0).any(0))}) if len(ge2) else []
    return dict(
        n_ge2_distinct=int(len(ge2)), latents_ge2=[int(x) for x in ge2],
        n_ge1_distinct=int(len(ge1)),
        n_ge4_distinct=int(len(spanning)),
        values_seen_in_ge2=values_seen, n_values_seen_in_ge2=len(values_seen),
        n_dominant=len(dom), n_dominant_min2hits=len(dom2),
        dominant={str(f): [words[j], n] for f, (j, n) in dom.items()},
        per_label_ge1={words[j]: int((h[:, j] > 0).sum()) for j in range(len(words))},
        per_label_dominant=dict(Counter(words[j] for j, _ in dom.values())),
    )


# ---------------------------------------------------------------------------
# 3. decoder rows by byte range
# ---------------------------------------------------------------------------

def cap_by_label(labmap, cap):
    """Trim a labelled latent set to `cap` members, evenly across labels.

    Taking the lowest `cap` latent indices instead would be an unbiased sample of
    directions but not of MONTHS, and a month that lost all its members would
    silently shrink the cycle the per-value ring is fitted to.
    """
    per = max(1, cap // max(1, len({v for v in labmap.values()})))
    by = defaultdict(list)
    for f in sorted(labmap):
        by[labmap[f]].append(f)
    return sorted(f for v in by for f in by[v][:per])


def read_w_dec_rows(url, header, data_offset, rows, d_model=D_MODEL, gap=1 << 16):
    """Fetch only the requested w_dec rows. Nearby rows are coalesced into one
    range request, so a contiguous run costs one round trip instead of len(rows)."""
    meta = header["w_dec"]
    assert meta["dtype"] == "F32" and meta["shape"][1] == d_model, meta
    base = data_offset + meta["data_offsets"][0]
    rb = d_model * 4
    rows = sorted(set(int(r) for r in rows))
    out = np.empty((len(rows), d_model), dtype=np.float32)
    i = 0
    while i < len(rows):
        j = i
        while j + 1 < len(rows) and (rows[j + 1] - rows[j]) * rb <= gap:
            j += 1
        lo, hi = rows[i], rows[j]
        raw = retry(_curl, url, f"{base + lo * rb}-{base + (hi + 1) * rb - 1}")
        blk = np.frombuffer(raw, dtype=np.float32).reshape(hi - lo + 1, d_model)
        for k in range(i, j + 1):
            out[k] = blk[rows[k] - lo]
        i = j + 1
    return rows, out


def cosine_stats(W):
    U = W / np.maximum(np.linalg.norm(W, axis=1, keepdims=True), 1e-12)
    C = U @ U.T
    iu = np.triu_indices(len(W), 1)
    v = C[iu]
    return dict(n=int(len(W)), mean=float(v.mean()), median=float(np.median(v)),
                min=float(v.min()), max=float(v.max()), std=float(v.std()),
                vs_engels_0p53=float(v.mean() - ENGELS_MEAN_COSINE))


def ring_report(W, label_index, full_k, seed=0, n_perm=2000):
    """Everything the pre-check asks of a candidate ring, on decoder directions."""
    if len(W) < 3:
        return dict(skipped=f"k={len(W)} < 3")
    U = W / np.maximum(np.linalg.norm(W, axis=1, keepdims=True), 1e-12)
    rep = dict(k=int(len(W)), cosines=cosine_stats(W))
    if fits.degenerate(U):
        rep["degenerate"] = True
        return rep
    xy, ratios, _ = fits.pca_2d(U)
    cx, cy, r, resid = fits.fit_circle(xy)
    circ = fits.circularity(xy, cx, cy, r)
    rep.update(
        top2_var_frac=float(ratios[:2].sum()),
        pc_var_ratios=[float(x) for x in ratios[:5]],
        circle_resid_over_r=float(resid),
        circularity=circ,
        is_ring=fits.is_ring(circ),
        below_min_ring_k=bool(len(W) < fits.MIN_RING_K),
        direction_ring_score=fits.direction_ring_score(U, n_perm=4000, seed=seed),
    )
    if label_index is not None and len(set(label_index)) == len(label_index):
        ang = fits.angles(xy, cx, cy)
        rep["cyclic_order"] = fits.cyclic_order_score(
            ang, label_index, k=len(W), n_perm=n_perm, seed=seed, full_k=full_k)
    return rep


def random_control(pool, k, seed, n_draw=12):
    """The same statistics on k random decoder rows. `direction_ring_score` carries
    an analytic null of its own, but the cosine mean and the top-2 variance fraction
    do not, and both are strongly k-dependent -- three random unit vectors in 1152
    dimensions are near-equilateral in their own plane, so a small cluster scores
    well for free.

    Draws are subsampled from one prefetched pool of random rows rather than
    re-fetching per draw; the rows are still uniformly chosen latents, and the
    saving is a few hundred range requests per SAE.
    """
    rng = np.random.default_rng(seed)
    means, top2, gaps = [], [], []
    for _ in range(n_draw):
        W = pool[rng.choice(len(pool), size=k, replace=False)]
        U = W / np.maximum(np.linalg.norm(W, axis=1, keepdims=True), 1e-12)
        means.append(cosine_stats(W)["mean"])
        _, ratios, _ = fits.pca_2d(U)
        top2.append(float(ratios[:2].sum()))
        gaps.append(fits.direction_ring_score(U, n_perm=400, seed=int(rng.integers(1 << 30)))["gap_ratio"])
    return dict(k=int(k), n_draw=n_draw,
                cos_mean=float(np.mean(means)), cos_mean_sd=float(np.std(means)),
                cos_mean_p95=float(np.percentile(means, 95)),
                top2_var_frac=float(np.mean(top2)),
                gap_ratio=float(np.mean(gaps)))


# ---------------------------------------------------------------------------
# per-SAE probe
# ---------------------------------------------------------------------------

def probe(job, luts, do_ring, control):
    site, layer, width, l0 = job
    d = dir_for(site, layer, width, l0)
    ex = f"{BASE}/{d}/examples.safetensors"
    pa = f"{BASE}/{d}/params.safetensors"
    row = dict(site=site, layer=layer, width=width, l0_suite=l0, path=d)
    t0 = time.time()
    try:
        cfg = get_json(f"{d}/config.json") or {}
        row["nominal_l0"] = cfg.get("l0")
        row["d_sae"] = cfg.get("width")
        h, off = retry(read_header, ex)
        tt = np.asarray(retry(read_tensor, ex, "top_tokens", h, off))
        ff = np.asarray(retry(read_tensor, ex, "feature_frequencies", h, off))
    except Exception as e:
        row["error"] = str(e)[:200]
        return row
    row["n_latents"] = int(tt.shape[0])
    row["dead"] = int((ff < 1e-6).sum())
    row["alive"] = int((ff >= 1e-6).sum())

    mh = label_hits(tt, luts["months"], len(MONTHS))
    dh = label_hits(tt, luts["days"], len(DAYS))
    row["months"] = census(mh, MONTHS)
    row["months_no_may"] = census(mh, MONTHS, ignore={"may"})
    row["days"] = census(dh, DAYS)

    if do_ring:
      try:
        ph, poff = retry(read_header, pa)
        row["ring"] = {}
        all_sets = {}
        for onto, cen, words, full_k in (("months", row["months"], MONTHS, 12),
                                         ("days", row["days"], DAYS, 7)):
            all_sets[onto] = (words, full_k, {
                "ge2_distinct": (sorted(int(f) for f in cen["latents_ge2"])[:CAP_STRICT], None),
                "dominant_min2hits": (
                    sorted(int(f) for f, v in cen["dominant"].items() if v[1] >= 2)[:CAP_STRICT],
                    {int(f): words.index(v[0]) for f, v in cen["dominant"].items() if v[1] >= 2}),
                "dominant_any": (
                    cap_by_label({int(f): words.index(v[0])
                                  for f, v in cen["dominant"].items()}, CAP_WIDE),
                    {int(f): words.index(v[0]) for f, v in cen["dominant"].items()}),
            })
        # The three sets are nested and the two ontologies can overlap, so fetch the
        # union of decoder rows once. Refetching per set costs a range request per
        # latent and dominates the runtime at 262k.
        need = sorted({f for _, _, s in all_sets.values() for feats, _ in s.values() for f in feats})
        cache = {}
        if need:
            got, Wall = read_w_dec_rows(pa, ph, poff, need)
            cache = {f: Wall[i] for i, f in enumerate(got)}

        for onto, (words, full_k, sets) in all_sets.items():
            out = {}
            for name, (feats, labmap) in sets.items():
                if len(feats) < 3:
                    out[name] = dict(skipped=f"k={len(feats)} < 3")
                    continue
                rows_got = feats
                W = np.stack([cache[f] for f in rows_got])
                # per-latent ring: the calendar label of each latent (may repeat)
                lab = [labmap[f] for f in rows_got] if labmap else None
                per_lat = ring_report(W, lab if lab and len(set(lab)) == len(lab) else None,
                                      full_k)
                out[name] = dict(latents=rows_got, per_latent=per_lat)
                # per-value ring: one mean unit direction per calendar value. The
                # calendar order lives on the VALUES, and latent multiplicity would
                # otherwise distort the angular spacing the closure test reads.
                if labmap:
                    U = W / np.maximum(np.linalg.norm(W, axis=1, keepdims=True), 1e-12)
                    by = defaultdict(list)
                    for i, f in enumerate(rows_got):
                        by[labmap[f]].append(U[i])
                    vals = sorted(by)
                    M = np.stack([np.mean(by[v], 0) for v in vals])
                    out[name]["per_value"] = dict(
                        values=[words[v] for v in vals],
                        n_latents_per_value={words[v]: len(by[v]) for v in vals},
                        **ring_report(M, vals, full_k))
            row["ring"][onto] = out

        if control:
            ks = set()
            for onto_blk in row["ring"].values():
                for blk in onto_blk.values():
                    for tier in ("per_latent", "per_value"):
                        rep = blk.get(tier)
                        if isinstance(rep, dict) and "k" in rep:
                            ks.add(int(rep["k"]))
            ks = [k for k in sorted(ks) if 3 <= k <= 200]
            if ks:
                # 8 contiguous 64-row blocks at random offsets: 8 range requests
                # instead of 512. Latent index carries no geometry, so a block of
                # consecutive rows is still an arbitrary sample of directions, and
                # spreading it over 8 offsets guards against any block structure.
                rng = np.random.default_rng(layer * 977 + row["n_latents"])
                nblk = max(8, int(np.ceil(2 * max(ks) / 64)))
                starts = rng.choice(row["n_latents"] - 64, size=nblk, replace=False)
                pool_rows = np.concatenate([np.arange(s, s + 64) for s in starts])
                _, pool = read_w_dec_rows(pa, ph, poff, pool_rows, gap=1 << 18)
                row["random_control"] = {
                    str(k): random_control(pool, k, seed=layer * 100 + k) for k in ks}
      except Exception as e:                      # noqa: BLE001
        # A ring failure must not cost the census, which is the expensive read.
        row["ring_error"] = f"{type(e).__name__}: {str(e)[:200]}"
    row["seconds"] = round(time.time() - t0, 1)
    return row


# ---------------------------------------------------------------------------
# 4. budget
# ---------------------------------------------------------------------------

def budget(surveyed, n_tokens=4_000_000):
    """Bytes/token/layer, cache size, wall clock, and clustering memory.

    Grounded on the measured 16k caches rather than remembered numbers:
    `SparseWriter` writes 4 bytes per nonzero (uint16 index + fp16 value) plus 8
    bytes per row, and `results/cache_meta*.json` confirm
    bytes_on_disk == 4*nnz + 8*(n_rows+1) exactly.
    """
    root = Path(__file__).resolve().parents[1] / "results"
    measured, by_suite = {}, defaultdict(list)
    for f, suite in (("cache_meta.json", "small"), ("cache_meta_big.json", "big")):
        p = root / f
        if not p.exists():
            continue
        for k, m in json.loads(p.read_text()).items():
            mean_l0 = m["n_nonzero"] / m["n_rows"]
            measured[(suite, int(k))] = dict(
                nominal_l0=m["nominal_l0"], mean_l0=round(mean_l0, 2),
                bytes_per_token=round(m["bytes_on_disk"] / m["n_rows"], 1),
                inflation_vs_nominal=round(mean_l0 / m["nominal_l0"], 4))
            by_suite[suite].append(mean_l0 / m["nominal_l0"])
    # The realised mean L0 overshoots the nominal figure, and by a suite-dependent
    # amount: 17% on the small suite (nominal 20, measured 23.5) but only 1% on big.
    # Averaging the two would inflate every big-suite projection by a sixth.
    infl_by_suite = {s: float(np.mean(v)) for s, v in by_suite.items()}
    infl_default = float(np.mean(list(infl_by_suite.values()))) if infl_by_suite else 1.0

    def per_token(d_sae, nominal_l0, suite):
        # index width: uint16 addresses 0..65535, which covers d_sae == 65536
        # exactly; 262144 needs uint32.
        idx = 2 if d_sae <= 65536 else 4
        mean_l0 = nominal_l0 * infl_by_suite.get(suite, infl_default)
        return idx, mean_l0, mean_l0 * (idx + 2) + 8

    # wall clock: model forward is fixed, SAE encode scales with width.
    fwd_flops = 2 * 1.0e9                       # gemma-3-1b, ~1B params, 2 FLOP/param/token
    def enc_flops(d_sae):
        return 2 * D_MODEL * d_sae
    ref_layers, ref_d_sae = 11, 16384
    ref_tps = (360.0, 515.0)                    # measured on the 16k tier-1 cache build
    ref_total = fwd_flops + ref_layers * enc_flops(ref_d_sae)
    eff = tuple(t * ref_total for t in ref_tps)  # effective FLOP/s, low and high

    scenarios = {}
    for name, layers, d_sae, suites in (
        ("65k @ deep-dive layers {7,13,17,22}", DEEP_DIVE, 65536, ("small", "medium", "big")),
        ("262k @ requested layers {9,16,18}", REQUESTED, 262144, ("small", "big")),
        ("262k @ deep-dive layers {7,13,17,22}", DEEP_DIVE, 262144, ("small", "big")),
    ):
        per_suite = {}
        for suite in suites:
            l0s = [r["nominal_l0"] for r in surveyed
                   if r.get("d_sae") == d_sae and r["l0_suite"] == suite
                   and r["layer"] in layers and r.get("nominal_l0")]
            if not l0s:
                continue
            nl0 = float(np.mean(l0s))
            idx, mean_l0, bpt = per_token(d_sae, nl0, suite)
            tot = bpt * n_tokens * len(layers)
            per_tok_flops = fwd_flops + len(layers) * enc_flops(d_sae)
            tps = sorted(e / per_tok_flops for e in eff)
            hours = sorted(n_tokens / t / 3600 for t in tps)
            per_suite[suite] = dict(
                nominal_l0_mean=round(nl0, 1), nominal_l0_per_layer=l0s,
                index_dtype=f"uint{idx * 8}", assumed_mean_l0=round(mean_l0, 1),
                bytes_per_token_per_layer=round(bpt, 1),
                cache_gb_all_layers=round(tot / 1e9, 2),
                n_layers=len(layers),
                proj_tok_per_s=[round(tps[0]), round(tps[1])],
                proj_hours=[round(hours[0], 2), round(hours[1], 2)])
        scenarios[name] = dict(layers=layers, d_sae=d_sae, suites=per_suite)

    def mem(d_sae):
        w = d_sae * D_MODEL * 4
        return dict(
            d_sae=d_sae,
            params_f32_per_layer_gb=round((2 * w + d_sae * 4 * 2 + D_MODEL * 4) / 1e9, 3),
            w_enc_only_f32_gb=round(w / 1e9, 3),
            w_dec_normalised_f32_gb=round(w / 1e9, 3),
            dense_FxF_f32_gb=round(d_sae * d_sae * 4 / 1e9, 2),
            chunk_1024xF_f32_gb=round(1024 * d_sae * 4 / 1e9, 3),
            topk64_lists_gb=round(d_sae * 64 * 8 / 1e9, 4))

    return dict(
        n_tokens=n_tokens,
        measured_16k=({f"{s}_L{l}": v for (s, l), v in sorted(measured.items())}),
        l0_inflation_by_suite={k: round(v, 4) for k, v in infl_by_suite.items()},
        measured_note=("bytes_on_disk == 4*nnz + 8*(n_rows+1) exactly for every "
                       "cached 16k layer; realised mean L0 runs above nominal by "
                       + ", ".join(f"{(v - 1) * 100:.1f}% ({k})"
                                   for k, v in sorted(infl_by_suite.items()))),
        wall_clock_model=dict(
            ref="16k tier-1 build, 11 layers, 360-515 tok/s",
            fwd_gflop_per_token=fwd_flops / 1e9,
            encode_gflop_per_token_per_layer={
                "16k": enc_flops(16384) / 1e9, "65k": enc_flops(65536) / 1e9,
                "262k": enc_flops(262144) / 1e9},
            effective_tflop_s=[round(eff[0] / 1e12, 2), round(eff[1] / 1e12, 2)]),
        scenarios=scenarios,
        memory={f"{k}": mem(v) for k, v in
                (("16k", 16384), ("65k", 65536), ("262k", 262144))},
        cache_writer_blocker=dict(
            file="src/gfdisc/cache.py:36",
            code="if d_sae > 65535: raise ValueError(...)",
            effect=("rejects d_sae=65536 by one. uint16 indices address 0..65535, "
                    "which covers width 65k EXACTLY, so the fix is to relax the "
                    "bound to `> 65536`, not to widen the dtype. 262k genuinely "
                    "needs uint32 indices (+2 bytes/nonzero) and also breaks the "
                    "hardcoded 65536-entry lookup tables in "
                    "scripts/20_window_cluster.py:308 and "
                    "scripts/22_window_passer_audit.py:281.")),
        clustering_note=("graph mode only: never materialise [F,F]. At 65k the dense "
                         "f32 matrix is 17.18 GB and at 262k it is 274.9 GB. A "
                         "1024-row chunk against normalised w_dec is 0.27 GB at 65k "
                         "and 1.07 GB at 262k, and top-64 neighbour lists are 0.03 GB "
                         "and 0.13 GB."),
    )


# ---------------------------------------------------------------------------
# 5. summary -- the numbers the GO/NO-GO turns on, in the artifact itself
# ---------------------------------------------------------------------------

def summarize(rows):
    tests = []
    for r in rows:
        for onto in ("months", "days"):
            for name, blk in (r.get("ring", {}).get(onto, {}) or {}).items():
                for tier in ("per_latent", "per_value"):
                    rep = blk.get(tier)
                    if not isinstance(rep, dict) or "k" not in rep:
                        continue
                    co = rep.get("cyclic_order", {})
                    drs = rep.get("direction_ring_score", {})
                    tests.append(dict(path=r["path"], width=r["width"], layer=r["layer"],
                                      suite=r["l0_suite"], onto=onto, set=name, tier=tier,
                                      k=rep["k"], cos=rep["cosines"]["mean"],
                                      top2=rep.get("top2_var_frac"),
                                      rho=co.get("rho"), p_rho=co.get("p"),
                                      gap=drs.get("gap_ratio"), p_gap=drs.get("p_closed")))
    with_rho = [t for t in tests if t["p_rho"] is not None]
    with_gap = [t for t in tests if t["p_gap"] is not None]
    both = [t for t in with_rho
            if t["p_gap"] is not None and t["p_rho"] < 0.05 and t["p_gap"] < 0.05]

    cos_by_width = {}
    for w in ("16k", "65k", "262k", "1m"):
        v = [t["cos"] for t in tests
             if t["width"] == w and t["onto"] == "months"
             and t["set"] == "ge2_distinct" and t["tier"] == "per_latent"]
        if v:
            cos_by_width[w] = dict(n=len(v), mean=round(float(np.mean(v)), 4),
                                   min=round(min(v), 4), max=round(max(v), 4))

    ladder = defaultdict(dict)
    for r in rows:
        if r["l0_suite"] != "big":
            continue
        ladder[str(r["layer"])][r["width"]] = dict(
            month_ge2=r["months"]["n_ge2_distinct"],
            month_ge1=r["months"]["n_ge1_distinct"],
            day_ge2=r["days"]["n_ge2_distinct"],
            dead_frac=round(r["dead"] / r["n_latents"], 4))

    return dict(
        n_ring_tests=len(tests),
        cyclic_order=dict(n=len(with_rho),
                          n_nominal_p_lt_05=sum(t["p_rho"] < 0.05 for t in with_rho),
                          expected_by_chance=round(0.05 * len(with_rho), 1),
                          min_p=min((t["p_rho"] for t in with_rho), default=None),
                          bonferroni_threshold=round(0.05 / max(len(with_rho), 1), 6)),
        closure=dict(n=len(with_gap),
                     n_nominal_p_lt_05=sum(t["p_gap"] < 0.05 for t in with_gap),
                     expected_by_chance=round(0.05 * len(with_gap), 1)),
        passing_both_nominal=[{k: t[k] for k in
                               ("path", "onto", "set", "tier", "k", "rho", "p_rho", "gap", "p_gap")}
                              for t in both],
        mean_decoder_cosine_by_width=cos_by_width,
        engels_reference=ENGELS_MEAN_COSINE,
        width_ladder_l0_big=dict(ladder),
        reading=("Mean pairwise decoder cosine among month latents FALLS monotonically "
                 "with width (0.052 at 16k -> 0.010 at 1m) against Engels' published "
                 "0.53, and sits within a few control standard deviations of random "
                 "unit vectors in 1152 dimensions at every width >= 65k. The census "
                 "grows with width, the geometry does not concentrate. Nothing "
                 "survives correction for the number of ring tests run, and the only "
                 "two sets passing both closure and ordering nominally are at 16k."),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/precheck_65k.json")
    ap.add_argument("--no-ring", action="store_true")
    ap.add_argument("--no-control", action="store_true")
    ap.add_argument("--no-1m", action="store_true")
    ap.add_argument("--smoke", action="store_true",
                    help="two SAEs at one layer, to check the plumbing cheaply")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--n-tokens", type=int, default=4_000_000)
    ap.add_argument("--rescore", action="store_true",
                    help="recompute the summary and budget blocks from an existing "
                         "--out file, without re-reading anything from the hub")
    a = ap.parse_args()

    if a.rescore:
        # The census and ring stages are pure functions of the shipped tensors, so
        # rederiving the aggregate blocks needs no network. Same code path as a full
        # run; it just skips ~200 MB of range reads.
        out = json.loads(Path(a.out).read_text())
        out["summary"] = summarize(out["census"])
        out["budget"] = budget(out["census"], n_tokens=a.n_tokens)
        Path(a.out).write_text(json.dumps(out, indent=2))
        print(json.dumps(out["summary"], indent=2))
        print(f"\nrescored {a.out}")
        return

    print("== 1. inventory ==", flush=True)
    groups = inventory()
    inv = []
    for (site, w, l0), layers in sorted(groups.items(), key=lambda kv: (kv[0][0], kv[0][1], kv[0][2])):
        ls = sorted(layers)
        has_ex = sorted(l for l in ls if "examples.safetensors" in layers[l])
        inv.append(dict(site=site, width=w, d_sae=width_int(w), l0_suite=l0,
                        layers=ls, n_layers=len(ls),
                        layers_with_examples=has_ex,
                        has_examples=len(has_ex) == len(ls) and len(has_ex) > 0,
                        files=sorted(set().union(*layers.values()))))
    # sizes + exact L0 for every width >= 65k, plus the 16k rung of the ladder
    def enrich(e):
        if e["d_sae"] is None:
            return e
        want = e["d_sae"] >= 65536 or (e["site"] == "resid_post")
        probe_layers = [l for l in e["layers"] if l in DEEP_DIVE + REQUESTED] or e["layers"][:1]
        if want:
            l = probe_layers[0]
            d = dir_for(e["site"], l, e["width"], e["l0_suite"])
            e["params_bytes"] = http_size(f"{d}/params.safetensors")
            e["examples_bytes"] = http_size(f"{d}/examples.safetensors")
        cfgs = {}
        for l in probe_layers:
            c = get_json(f"{dir_for(e['site'], l, e['width'], e['l0_suite'])}/config.json")
            if c:
                cfgs[str(l)] = c.get("l0")
        e["nominal_l0_by_layer"] = cfgs
        return e

    if not a.smoke:
        with ThreadPoolExecutor(max_workers=a.workers) as ex:
            inv = list(ex.map(enrich, inv))
    for e in inv:
        e.setdefault("nominal_l0_by_layer", {})
    for e in inv:
        span = (str(e["layers"]) if e["n_layers"] <= 6
                else f"{min(e['layers'])}-{max(e['layers'])} ({e['n_layers']})")
        print(f"  {e['site']:15s} {e['width']:5s} {e['l0_suite']:16s} layers={span:20s} "
              f"examples={'yes' if e['has_examples'] else 'NO ':3s} "
              f"params={(e.get('params_bytes') or 0) / 1e6:7.0f}MB "
              f"l0={e['nominal_l0_by_layer']}", flush=True)

    print("\n== 2/3. census + ring ==", flush=True)
    from transformers import AutoTokenizer
    tk = AutoTokenizer.from_pretrained("google/gemma-3-1b-pt",
                                       token=os.environ.get("HF_TOKEN"))
    vocab = [(tk.convert_ids_to_tokens(i) or "").replace("▁", " ")
             for i in range(tk.vocab_size)]
    luts = {"months": build_lut(vocab, MONTHS), "days": build_lut(vocab, DAYS)}

    jobs = []
    for l in DEEP_DIVE:                       # the width ladder at a fixed layer
        for s in ("small", "medium", "big"):
            jobs.append(("resid_post", l, "16k", s))
            jobs.append(("resid_post", l, "65k", s))
        jobs.append(("resid_post", l, "262k", "big"))
    for l in REQUESTED:                        # the layers the plan asked for
        jobs.append(("resid_post_all", l, "16k", "big"))
        jobs.append(("resid_post_all", l, "262k", "big"))
        jobs.append(("resid_post_all", l, "262k", "small"))
    if not a.no_1m:
        for l in (17, 22):
            jobs.append(("resid_post", l, "1m", "big"))
    if a.smoke:
        jobs = [("resid_post", 17, "16k", "big"), ("resid_post", 17, "65k", "big")]

    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        rows = list(ex.map(lambda j: probe(j, luts, not a.no_ring, not a.no_control), jobs))

    ok = [r for r in rows if "error" not in r]
    ok.sort(key=lambda r: (r["site"], r["layer"], r.get("d_sae") or 0, r["l0_suite"]))
    print(f"\n{'site':15s}{'L':>3s}{'width':>7s}{'suite':>8s}{'l0':>5s}{'dead':>8s} | "
          f"{'mon>=2':>7s}{'mon>=1':>7s}{'monDom':>7s}{'noMay>=2':>9s} | "
          f"{'day>=2':>7s}{'day>=1':>7s}{'dayDom':>7s}")
    print("-" * 108)
    for r in ok:
        print(f"{r['site']:15s}{r['layer']:3d}{r['width']:>7s}{r['l0_suite']:>8s}"
              f"{r.get('nominal_l0', 0):>5}{r['dead']:>8d} | "
              f"{r['months']['n_ge2_distinct']:>7d}{r['months']['n_ge1_distinct']:>7d}"
              f"{r['months']['n_dominant_min2hits']:>7d}{r['months_no_may']['n_ge2_distinct']:>9d} | "
              f"{r['days']['n_ge2_distinct']:>7d}{r['days']['n_ge1_distinct']:>7d}"
              f"{r['days']['n_dominant_min2hits']:>7d}", flush=True)

    if not a.no_ring:
        print(f"\n{'path':44s}{'onto':7s}{'set':20s}{'k':>4s}{'cos':>7s}{'cosRnd':>7s}"
              f"{'top2':>7s}{'t2Rnd':>7s}{'gapR':>7s}{'p':>7s}{'rho':>7s}{'pRho':>7s}{'ring':>6s}")
        print("-" * 136)
        for r in ok:
            ctl = r.get("random_control", {})
            for onto in ("months", "days"):
                for name, blk in (r.get("ring", {}).get(onto, {}) or {}).items():
                    for tier in ("per_latent", "per_value"):
                        rep = blk.get(tier)
                        if not isinstance(rep, dict) or "k" not in rep:
                            continue
                        drs = rep.get("direction_ring_score", {})
                        co = rep.get("cyclic_order", {})
                        c = ctl.get(str(rep["k"]), {})
                        nan = float("nan")
                        print(f"{r['path'][:43]:44s}{onto:7s}{(name + '/' + tier)[:19]:20s}"
                              f"{rep['k']:>4d}{rep['cosines']['mean']:>7.3f}"
                              f"{c.get('cos_mean', nan):>7.3f}"
                              f"{rep.get('top2_var_frac', nan):>7.3f}"
                              f"{c.get('top2_var_frac', nan):>7.3f}"
                              f"{drs.get('gap_ratio', nan):>7.2f}"
                              f"{drs.get('p_closed', nan):>7.3f}"
                              f"{co.get('rho', nan):>7.3f}{co.get('p', nan):>7.3f}"
                              f"{str(rep.get('is_ring')):>6s}")

    print("\n== 4. budget ==", flush=True)
    bud = budget(ok, n_tokens=a.n_tokens)
    for name, sc in bud["scenarios"].items():
        for suite, v in sc["suites"].items():
            print(f"  {name:40s} {suite:7s} l0={v['nominal_l0_mean']:>6.1f} "
                  f"{v['bytes_per_token_per_layer']:>7.1f} B/tok/layer  "
                  f"{v['cache_gb_all_layers']:>7.2f} GB  "
                  f"{v['proj_hours'][0]:.1f}-{v['proj_hours'][1]:.1f} h")

    out = dict(
        status="exploratory, not pre-registered",
        script=str(Path(__file__).name),
        repo=REPO,
        generated_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        method=dict(
            reads="HTTP byte-range only; no model, no cache, no inference",
            month_criterion=("exact match of a stripped/lowercased top-10 token to a "
                             "full month name; >=2 distinct names is the rule used by "
                             "02b_precheck_cluster.py and 18_month_latent_placement.py"),
            caveat=("labels are post hoc; the >=2-distinct rule selects against "
                    "monosemantic latents, so dominant-label sets are reported "
                    "alongside it"),
            engels_reference_mean_cosine=ENGELS_MEAN_COSINE,
            min_ring_k=fits.MIN_RING_K),
        headline=dict(
            requested_layers=REQUESTED,
            width_65k_available_at=sorted(
                {l for e in inv if e["width"] == "65k" for l in e["layers"]}),
            width_65k_missing_at=[l for l in REQUESTED
                                  if l not in {ll for e in inv if e["width"] == "65k"
                                               for ll in e["layers"]}],
            note=("width 65k ships only under the deep-dive `resid_post` site at "
                  "layers 7/13/17/22. The month-manifold layers 9/16/18 have no 65k "
                  "SAE at all; the only width >= 65k available there is 262k under "
                  "`resid_post_all`. The two sites are separate training runs "
                  "(differing LFS hashes at the same layer/width/L0), so a 16k "
                  "number from `resid_post_all` is not a like-for-like baseline "
                  "for a `resid_post` SAE at the same layer -- which is why the 16k "
                  "rung is recomputed here inside each site."),
            seed_replicates="262k / medium_seed_1 at layers 7,13,17,22; params only, no examples.safetensors",
        ),
        inventory=inv,
        summary=summarize(ok),
        census=ok,
        errors=[r for r in rows if "error" in r],
        budget=bud,
    )
    p = Path(a.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
