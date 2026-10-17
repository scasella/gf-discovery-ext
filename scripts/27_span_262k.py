"""Step 27 (B-lite) -- is the month ring latent-SPANNED at 262k width?

THE QUESTION, AND WHY IT IS NOT THE ONE THE PRE-CHECK ANSWERED.

`results/precheck_65k.json` established two things at once. The month-latent CENSUS
explodes with dictionary width (layer 9: 7 latents at 16k -> 73 at 262k; layer 16:
15 -> 124; layer 18: 19 -> 159), while the mean pairwise decoder cosine among those
latents COLLAPSES monotonically (0.052 at 16k -> 0.026 at 65k -> 0.018 at 262k ->
0.010 at 1m) against Engels' published 0.53 for a GPT-2-small month cluster. The
conclusion drawn there was narrow and correct: no COSINE CLUSTERING procedure can
assemble the month latents at any width, because they are not mutually close.

Pairwise scatter does not preclude the ring living in the SPAN of the many month
latents. Seventy-three directions that are mutually near-orthogonal can still span a
73-dimensional subspace containing a particular 2-plane exactly. Cosine clustering
looks for a CLUMP; the ring may be a SUBSPACE. This script tests the subspace claim
directly, and it is the test that decides whether "latent-level discovery is
exhausted" is a statement about clustering or a statement about SAE latents.

THE TEST. For each of layers 9, 16 and 18:

  1. Recover the supervised month-ring plane in the RAW residual stream, from the
     twelve month-label means over the sixteen templates of `12_closure_depth.py`.
     That is the object whose closure is recorded in `results/closure_depth.json`
     (p_closed 1.5e-4 / 0.0201 / 0.00135 at layers 9 / 16 / 18), and reproducing
     those numbers is an ANCHOR GATE run before the span test is allowed to proceed.
  2. Identify the 262k month latents from the shipped top-10 tokens, using the census
     rule of `26_precheck_65k.py` verbatim (>= 2 distinct month names), with >= 1
     distinct, a may-excluded arm and a dominant-label arm as declared sensitivities.
  3. Pull ONLY those latents' `w_dec` rows by HTTP byte range. A row is 4608
     contiguous bytes of a 1.2 GB file, so the whole test costs a few megabytes.
  4. Principal angles between the 2-D ring plane and span(w_dec[month latents]).
     Report the plane energy captured -- the squared cosines of BOTH principal
     angles, separately and averaged.

WHY BOTH SQUARED COSINES, SEPARATELY. A span with cos^2(th1) ~ 1 and cos^2(th2) at
the control level contains a one-dimensional month AXIS but not the ring plane. That
is the span-level analogue of ordering-without-closure, which this project has
already been burned by once (the ordering tier accepts random subsets at ~0.16, and
an ordinal control out-orders months at every layer). The averaged capture alone
would hide it.

WHY THE CONTROLS ARE LOAD-BEARING AND NOT DECORATION. k random directions in R^1152
capture a fixed 2-plane with expected energy k/1152 under isotropy -- 6.3% at k=73,
10.8% at k=124, 13.8% at k=159, and 59% at the >=1-distinct arm's k=679. Decoder rows
are not isotropic (they concentrate in the directions the residual stream actually
uses, which is where the ring plane also lives), so the isotropic figure is a floor,
not a null. Four controls run instead:

  alive_random     k rows drawn from a prefetched pool of ALIVE latents. The primary
                   null. 500 draws.
  freq_matched     k rows matched one-for-one to the month latents' activation
                   frequency (nearest neighbour in log10 feature_frequency). Month
                   latents are rare; if rarity alone predicts capture, this catches it.
  index_stratified k rows drawn one per month latent from within +/-2048 of that
                   latent's own index. Latent index carries no geometry, so this
                   should agree with alive_random; disagreement would mean it does.
  day_latents      the SAME test with WEEKDAY latents against the MONTH plane. The
                   LATENT-side specificity control: if weekday latents capture the
                   month plane with the same excess, the excess is a generic
                   calendar/token subspace, not the month ring.

and one more, on the other side of the comparison:

  placebo_planes   the SAME month latents against the WEEKDAY plane and the NUMBER-WORD
                   plane at the same layer, each built from 12_closure_depth.py's own
                   templates. This is the PLANE-side specificity control. Latent-side
                   and plane-side specificity are different claims and a positive needs
                   both: "month latents beat random latents at the month plane" is only
                   interesting if the same month latents do NOT beat random latents at
                   a weekday or number-word plane drawn from the same residual stream.
                   The number-word plane is the project's pre-registered ordinal control
                   and does not close at any of these layers (p_closed 0.98-0.997), which
                   is exactly what makes it a placebo rather than a rival.

Everything is declared in the output's `declared` block, built from the constants at
the top of this file before any number is computed.

Exploratory. Nothing here is pre-registered and nothing here can change a gate
verdict; labels are applied post hoc throughout.
"""
import argparse
import json
import os
import struct
import subprocess
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from gfdisc import fits  # noqa: E402
# NOTE: gfdisc.hf_range is deliberately NOT imported. Its `_curl` passes no --max-time,
# and a stalled hub connection therefore hangs a worker forever rather than raising
# something `retry` can catch. `curl_range` / `read_header_t` / `read_tensor_t` below
# are wall-clock-bounded replacements; the shared helper is left untouched because
# other scripts depend on it.

REPO = "google/gemma-scope-2-1b-pt"
BASE = f"https://huggingface.co/{REPO}/resolve/main"
SITE = "resid_post_all"
LAYERS = [9, 16, 18]
D_MODEL = 1152

MONTHS_LC = ["january", "february", "march", "april", "may", "june", "july",
             "august", "september", "october", "november", "december"]
DAYS_LC = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
MONTHS = ["January", "February", "March", "April", "May", "June",
          "July", "August", "September", "October", "November", "December"]

# ---------------------------------------------------------------------------
# DECLARED BEFORE RESULTS. Everything in this block is a pre-commitment and is
# copied verbatim into the output JSON.
# ---------------------------------------------------------------------------

PRIMARY_ARM = "months/ge2_distinct"          # the census rule the 16k negative used
PRIMARY_WIDTH = "262k"
PRIMARY_SUITE = "big"                        # l0_big, the suite the caches were built on
PRIMARY_PLANE = "MT16"                       # 12_closure_depth.py's 16 templates

N_CONTROL_DRAWS = 500
N_STRAT_DRAWS = 200
N_PROJ_CONTROL_DRAWS = 200                   # for the ring-in-projection test
CONTROL_SEED = 20260820
POOL_BLOCKS, POOL_BLOCK_ROWS = 112, 64       # 7168 prefetched alive-candidate rows
STRAT_WINDOW, STRAT_BLOCK = 2048, 8          # index-stratified: 8 rows within +/-2048
ALIVE_EPS = 1e-6                             # feature_frequencies >= this == alive
N_PERM_PLANE, N_PERM_PROJ = 20000, 5000

# Anchor gate. The plane is only usable if re-deriving it reproduces the recorded
# closure of results/closure_depth.json. The tolerance absorbs bf16/MPS
# nondeterminism and is tight enough that a wrong hook point or template set fails.
ANCHOR_GAP_RATIO = {9: 1.5123, 16: 1.8867, 18: 1.6262}
ANCHOR_P_CLOSED = {9: 1.4999e-4, 16: 0.020049, 18: 0.0013499}
ANCHOR_TOL_GAP = 0.06                        # absolute, on gap_ratio

STATISTIC = ("plane energy captured = (cos^2 th1 + cos^2 th2)/2, where th1,th2 are the "
             "principal angles between the 2-D supervised month-ring plane (top-2 PCs of "
             "the 12 centred month-label means in the raw residual stream) and "
             "span(w_dec[selected latents]); the cos^2 are the squared singular values "
             "of Q_plane^T Q_span")

# THE VERDICT RULE, fixed before any capture number exists.
VERDICT_RULE = dict(
    primary=f"{PRIMARY_ARM} @ {PRIMARY_WIDTH}/{SITE}/l0_{PRIMARY_SUITE}, plane={PRIMARY_PLANE}",
    statistic=STATISTIC,
    control=("empirical p = (1 + #{control capture >= observed capture}) / (1 + n_draws) "
             f"against {N_CONTROL_DRAWS} draws of k random ALIVE latents"),
    latent_spanned_at_262k=(
        "p_alive_random <= 0.005 at ALL of layers 9,16,18 AND capture / "
        "mean(alive_random capture) >= 1.5 at all three AND p_freq_matched <= 0.05 at "
        "all three"),
    not_spanned="p_alive_random > 0.05 at all three layers",
    ambiguous="anything else; the ambiguity is named explicitly in the output",
    specificity_flag=(
        "reported separately, not part of the three-way verdict: the MONTH excess ratio "
        "(capture / own k-matched alive_random mean) must exceed the DAY excess ratio at "
        ">= 2 of 3 layers, else flag 'generic_calendar_subspace'"),
    plane_specificity_flag=(
        "reported separately: the month latents' excess ratio against the MONTH plane "
        "must exceed their excess ratio against BOTH placebo planes (weekday, "
        "number-word, same layer, same residual stream, 12_closure_depth.py templates) "
        "at >= 2 of 3 layers, else flag 'not_plane_specific'"),
    axis_vs_plane_flag=(
        "reported separately: if cos^2(th1) clears its control but cos^2(th2) does not "
        "(p > 0.05) at >= 2 of 3 layers, flag 'axis_not_plane' -- the span-level analogue "
        "of ordering-without-closure"),
    projection_ring_test=(
        "declared and run at every layer regardless of the capture result: project the 12 "
        "centred label means into span(month latents), re-run fits.circularity + the "
        "uniform-angle gap p-value + cyclic_order_score under BOTH SVD sign conventions, "
        "and score it against the same battery on 200 random-alive-span projections"),
    complement_test=(
        "declared SECONDARY, outside the verdict: the same battery on the component "
        "ORTHOGONAL to span(month latents). If capture is high, ring death in the "
        "complement is arithmetic and not evidence; the informative outcome is ring "
        "SURVIVAL there, which would mean redundant encoding outside the month latents"),
    width_comparison=(
        "matched-k: subsample the 262k >=2-distinct set down to the 16k set's k at that "
        "layer (7/15/19), 500 draws, and compare that capture distribution to the 16k "
        "set's single capture value at the same k. No per-latent denominator is invented"),
)

SEQUENCE_DISCLOSURE = (
    "The primary arm, the statistic, the alive_random / freq_matched / index_stratified / "
    "day_latents controls, the verdict thresholds, the anchor gate and the "
    "projection/complement/width tests were all fixed in this file before any capture "
    "number was computed. The 262k census (73/124/159 at >=2 distinct) and the collapsing "
    "decoder cosines were already known from results/precheck_65k.json and are what "
    "motivated the test. TWO SEQUENCE EVENTS ARE DISCLOSED. (1) A plumbing run was made "
    "against a deliberately RANDOM placebo plane (12 Gaussian vectors in place of the "
    "label means) to validate the byte-range, census and control machinery end to end; it "
    "produced capture ~ the isotropic k/1152 expectation, as it must, and no real-plane "
    "number was in it. (2) The placebo_planes arm (month latents vs the weekday and "
    "number-word planes) was ADDED after the anchor gate had been seen to pass but BEFORE "
    "any capture, principal angle or control number against a real month plane had been "
    "computed; the run was killed at the anchor-gate line and restarted with the arm in "
    "place. Nothing else changed, and the verdict rule is untouched by the addition.")

# ---------------------------------------------------------------------------
# Ring-plane templates. MT16 is 12_closure_depth.py's set verbatim (the one the
# recorded closure anchors came from) and is PRIMARY. MON_T20 is
# 25_steering_probe.py's twenty-template set, carried as a robustness check; the
# principal angle between the two planes is reported so the choice can be seen not
# to matter.
# ---------------------------------------------------------------------------
MT16 = ["The report was published in {}", "She was born in {}", "It rained all {}",
        "The season begins in {}", "We travelled there last {}", "The deadline was {}",
        "Sales peaked in {}", "The festival runs through {}", "He resigned in {}",
        "The contract expires in {}", "Enrolment opens in {}", "The survey ran during {}",
        "Prices fell in {}", "The court ruled in {}", "The harvest starts in {}",
        "Construction began in {}"]
MON_T20 = MT16 + ["The exhibition closed in {}", "The law took effect in {}",
                  "The team formed in {}", "Records were broken in {}"]

# PLACEBO PLANES, plane-side specificity. Both word lists and both template sets are
# 12_closure_depth.py's verbatim, so the placebo planes are the same objects whose
# closure is recorded in results/closure_depth.json alongside the month plane
# (weekdays p_closed 0.159/0.345/0.568, numbers 0.983/0.984/0.997 at L9/16/18).
DAYS_CAP = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
NUMS = ["one", "two", "three", "four", "five", "six",
        "seven", "eight", "nine", "ten", "eleven", "twelve"]
DT16 = ["The meeting is on {}", "She arrived last {}", "We open every {}",
        "It happened on a {}", "The report is due {}", "They travel each {}",
        "The class meets on {}", "I saw him last {}", "The deadline is {}",
        "Deliveries arrive every {}", "The market runs on {}", "He works from {}",
        "Our call is set for {}", "The event starts {}", "Practice is held each {}",
        "The store closes on {}"]
NT16 = ["He counted to {}", "There were {} of them", "The total came to {}",
        "She bought {} apples", "It weighs {} kilograms", "The score was {}",
        "We waited {} hours", "They hired {} people", "The box holds {}",
        "He ran {} miles", "The list had {} entries", "It costs {} pounds",
        "The recipe needs {} eggs", "We ordered {} chairs", "The class has {} students",
        "He owns {} cars"]

# name -> (words, templates). "MT16" is the primary month plane; "MON_T20" the month
# robustness plane; the last two are the placebos.
PLANE_JOBS = {"MT16": (MONTHS, MT16), "MON_T20": (MONTHS, MON_T20),
              "weekday_plane": (DAYS_CAP, DT16), "number_plane": (NUMS, NT16)}
MONTH_PLANES = ["MT16", "MON_T20"]
PLACEBO_PLANES = ["weekday_plane", "number_plane"]


# ---------------------------------------------------------------------------
# transport
# ---------------------------------------------------------------------------

REQ = Counter()


_JITTER = np.random.default_rng(12345)


def retry(fn, *args, tries=9, base=1.7, **kw):
    """Range reads against the hub reset intermittently under concurrency (curl 56),
    which `_curl`'s own `--retry 3` does not cover.

    26_precheck_65k.py used tries=6/base=1.5 and that was not enough here: this script
    issues ~1100 range requests per 262k SAE against `params.safetensors` (against the
    pre-check's few hundred), and a run died on a connection reset that survived all six
    attempts -- six retries at 1.5^i spans only 22 s, which is shorter than the hub's
    throttle window. Longer ladder, jittered so a whole worker pool does not retry in
    lockstep, and the default worker count is halved to provoke fewer resets in the
    first place.
    """
    last = None
    for i in range(tries):
        try:
            REQ["n"] += 1
            return fn(*args, **kw)
        except Exception as e:                    # noqa: BLE001 - transport errors only
            last = e
            REQ["retry"] += 1
            time.sleep(base ** i * (0.5 + float(_JITTER.random())))
    raise last


# Wall-clock-bounded transport. `gfdisc.hf_range._curl` passes no --max-time, so a
# stalled connection hangs its worker FOREVER and `retry` above never gets the chance
# to fire -- and because ThreadPoolExecutor.map yields results in order, one hung block
# stalls the entire fetch with no error and no progress. Bounding each attempt in wall
# clock converts that silent hang into a retryable exception. `--retry-all-errors` also
# makes curl's own retries cover exit 56 (reset mid-body), which plain `--retry` does
# not. These are transport settings only: they change no number, only whether the bytes
# arrive. The shared helper is left untouched because other scripts depend on it.
CURL_MAX_TIME, CURL_CONNECT_TIMEOUT = 180, 20
_DTYPES = {"F64": np.float64, "F32": np.float32, "F16": np.float16,
           "I64": np.int64, "I32": np.int32, "I16": np.int16, "I8": np.int8,
           "U8": np.uint8, "BOOL": np.bool_}


def curl_range(url, byte_range):
    out = subprocess.run(
        ["curl", "-sL", "--fail", "--retry", "5", "--retry-all-errors",
         "--retry-delay", "2", "--connect-timeout", str(CURL_CONNECT_TIMEOUT),
         "--max-time", str(CURL_MAX_TIME), "-r", byte_range, url],
        capture_output=True)
    if out.returncode != 0:
        raise RuntimeError(f"curl failed ({out.returncode}) for {url} [{byte_range}]")
    return out.stdout


def read_header_t(url):
    """safetensors header: 8-byte little-endian length, then that many JSON bytes."""
    n = struct.unpack("<Q", curl_range(url, "0-7")[:8])[0]
    return json.loads(curl_range(url, f"8-{8 + n - 1}")), 8 + n


def read_tensor_t(url, name, header, data_offset):
    meta = header[name]
    start, end = meta["data_offsets"]
    raw = curl_range(url, f"{data_offset + start}-{data_offset + end - 1}")
    return np.frombuffer(raw, dtype=_DTYPES[meta["dtype"]]).reshape(meta["shape"])


def dir_for(layer, width, l0, site=SITE):
    return f"{site}/layer_{layer}_width_{width}_l0_{l0}"


def get_json(path):
    try:
        return json.loads(subprocess.run(["curl", "-sL", f"{BASE}/{path}"],
                                         capture_output=True, text=True).stdout)
    except Exception:
        return None


def _blocks(rows, gap_rows):
    """Coalesce sorted row indices into (lo, hi) runs no more than `gap_rows` apart."""
    rows = sorted(set(int(r) for r in rows))
    out, i = [], 0
    while i < len(rows):
        j = i
        while j + 1 < len(rows) and (rows[j + 1] - rows[j]) <= gap_rows:
            j += 1
        out.append((rows[i], rows[j]))
        i = j + 1
    return rows, out


def read_w_dec_rows(url, header, data_offset, rows, workers=6, gap_rows=14):
    """Fetch exactly the requested w_dec rows, in parallel, coalescing near neighbours.

    Each row is d_model*4 = 4608 contiguous bytes. Fetching 900 scattered rows out of a
    1.2 GB file costs ~4 MB and ~900 round trips, so the round trips are what gets
    parallelised, not the bytes.
    """
    meta = header["w_dec"]
    assert meta["dtype"] == "F32" and meta["shape"][1] == D_MODEL, meta
    base = data_offset + meta["data_offsets"][0]
    rb = D_MODEL * 4
    rows, blks = _blocks(rows, gap_rows)
    pos = {r: i for i, r in enumerate(rows)}
    out = np.empty((len(rows), D_MODEL), dtype=np.float32)

    def one(b):
        lo, hi = b
        raw = retry(curl_range, url, f"{base + lo * rb}-{base + (hi + 1) * rb - 1}")
        return lo, hi, np.frombuffer(raw, dtype=np.float32).reshape(hi - lo + 1, D_MODEL)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        for lo, hi, blk in ex.map(one, blks):
            for r in range(lo, hi + 1):
                if r in pos:
                    out[pos[r]] = blk[r - lo]
    return np.asarray(rows, dtype=np.int64), out


def make_sparse_lookup(index_array, matrix):
    """latent index -> its row of `matrix`. The position map is built ONCE; binding it
    as a default argument keeps the closure safe when the loop rebinds its locals."""
    pos = {int(f): i for i, f in enumerate(index_array)}

    def look(idx, _pos=pos, _M=matrix):
        return _M[[_pos[int(x)] for x in idx]]
    return look


def make_dense_lookup(Wall):
    def look(idx, _M=Wall):
        return _M[np.asarray(idx, dtype=np.int64)]
    return look


# ---------------------------------------------------------------------------
# census -- 26_precheck_65k.py's rule, verbatim
# ---------------------------------------------------------------------------

def build_lut(vocab, words):
    w2i = {w: i for i, w in enumerate(words)}
    lut = np.full(len(vocab), -1, dtype=np.int32)
    for t, s in enumerate(vocab):
        j = w2i.get(s.strip().lower())
        if j is not None:
            lut[t] = j
    return lut


def label_hits(tt, lut, n_words):
    idx = np.asarray(tt, dtype=np.int64)
    ok = (idx >= 0) & (idx < len(lut))
    lab = np.where(ok, lut[np.clip(idx, 0, len(lut) - 1)], -1)
    return np.stack([(lab == j).sum(1) for j in range(n_words)], axis=1).astype(np.int16)


def census(hits, words, ignore=()):
    keep = np.array([w not in ignore for w in words])
    h = hits * keep[None, :]
    distinct = (h > 0).sum(1)
    ge2 = np.flatnonzero(distinct >= 2)
    ge1 = np.flatnonzero(distinct >= 1)
    dom = {}
    for f in ge1:
        row = h[f]
        j = int(np.argmax(row))
        if (row == row[j]).sum() != 1:
            continue                              # tie -> not attributable to one label
        dom[int(f)] = (words[j], int(row[j]))
    return dict(latents_ge2=[int(x) for x in ge2], n_ge2_distinct=int(len(ge2)),
                latents_ge1=[int(x) for x in ge1], n_ge1_distinct=int(len(ge1)),
                dominant=dom,
                n_dominant_min2hits=int(sum(1 for v in dom.values() if v[1] >= 2)),
                per_label_ge1={words[j]: int((h[:, j] > 0).sum()) for j in range(len(words))})


# ---------------------------------------------------------------------------
# the statistic
# ---------------------------------------------------------------------------

def orth(W, rcond=1e-8):
    """Orthonormal basis of the row space of W (k, d), as a (d, rank) matrix."""
    A = np.asarray(W, dtype=np.float64).T                     # (d, k)
    U, s, _ = np.linalg.svd(A, full_matrices=False)
    tol = rcond * (s[0] if len(s) else 0.0)
    r = int((s > tol).sum())
    return U[:, :r], r, s


def capture(W, QP, fast=False):
    """Squared cosines of the two principal angles between span(W) and the plane QP.

    QP is (d, 2) with orthonormal columns. Returns (c1, c2) descending, where
    c_i = cos^2(theta_i) and (c1 + c2)/2 is the fraction of the plane's energy lying
    inside span(W). `fast` uses QR instead of SVD; random decoder rows are full rank
    almost surely, and the two agree to ~1e-12 (asserted in the smoke path).
    """
    A = np.asarray(W, dtype=np.float64).T
    Q = np.linalg.qr(A)[0] if fast else orth(W)[0]
    s = np.clip(np.linalg.svd(QP.T @ Q, compute_uv=False), 0.0, 1.0)
    c = np.zeros(2)
    c[:len(s)] = s[:2] ** 2
    return float(c[0]), float(c[1])


def var_captured(Xc, W):
    """Fraction of the 12 centred label means' total energy inside span(W)."""
    Q = orth(W)[0]
    return float(((Xc @ Q) ** 2).sum() / max((Xc ** 2).sum(), 1e-30))


def gap_pvalue(k, observed_max_gap_rad, n_perm=20000, seed=0):
    """12_closure_depth.py's closure p-value: max angular gap against uniform angles."""
    rng = np.random.default_rng(seed)
    a = np.sort(rng.uniform(0, 2 * np.pi, size=(n_perm, k)), axis=1)
    g = np.diff(np.concatenate([a, a[:, :1] + 2 * np.pi], axis=1), axis=1).max(1)
    return float((1 + (g <= observed_max_gap_rad).sum()) / (1 + n_perm))


def ring_battery(M12, seed=0, n_perm=N_PERM_PLANE):
    """circularity + closure p + ordering under both SVD sign conventions, on 12 points."""
    xy, ratios, _ = fits.pca_2d(M12)
    cx, cy, r, resid = fits.fit_circle(xy)
    c = fits.circularity(xy, cx, cy, r)
    out = dict(top2_var=float(ratios[:2].sum()), circle_resid_over_r=float(resid),
               radius=float(r), radius_ratio=c["radius_ratio"], gap_ratio=c["gap_ratio"],
               max_gap_deg=c["max_gap_deg"], is_ring=fits.is_ring(c),
               p_closed=gap_pvalue(len(M12), np.radians(c["max_gap_deg"]),
                                   n_perm=n_perm, seed=seed))
    ordering = {}
    for name, sgn in (("as_fitted", 1.0), ("pc2_flipped", -1.0)):
        xy2 = xy * np.array([1.0, sgn])
        c2 = fits.fit_circle(xy2)
        s = fits.cyclic_order_score(fits.angles(xy2, c2[0], c2[1]), np.arange(len(M12)),
                                    len(M12), n_perm=n_perm, seed=seed, full_k=len(M12))
        ordering[name] = dict(rho=s["rho"], p=s["p"], symmetry=list(s["symmetry"]),
                              null_p95=s["null_p95"])
    out["ordering"] = ordering
    return out


def emp_p(observed, null):
    null = np.asarray(null, dtype=float)
    return float((1 + (null >= observed).sum()) / (1 + len(null)))


def summarise_null(v):
    v = np.asarray(v, dtype=float)
    return dict(n=int(len(v)), mean=float(v.mean()), sd=float(v.std()),
                p50=float(np.percentile(v, 50)), p95=float(np.percentile(v, 95)),
                max=float(v.max()))


def draw_null(cand_idx, k, n, seed, lookup, QP):
    """n draws of k random latents from `cand_idx`; returns (cos2_th1[], cos2_th2[])."""
    cand_idx = np.asarray(cand_idx)
    if len(cand_idx) < k or k < 2:
        return None
    r = np.random.default_rng(seed)
    c1 = np.empty(n); c2 = np.empty(n)
    for i in range(n):
        pick = cand_idx[r.choice(len(cand_idx), size=k, replace=False)]
        c1[i], c2[i] = capture(lookup(pick), QP, fast=True)
    return c1, c2


def draw_freq_matched(target_idx, pool_idx, log_freq_pool, freq_all, n, seed, lookup, QP,
                      n_near=32):
    """n draws of one pool latent per target, matched on log10 activation frequency.

    The candidate list per target is the `n_near` pool latents nearest in log-frequency,
    found by binary search on the sorted pool rather than a full argsort per target.
    """
    tgt = np.log10(np.maximum(freq_all[np.asarray(target_idx, dtype=np.int64)], 1e-12))
    k = len(tgt)
    if len(pool_idx) < max(k, n_near) or k < 2:
        return None
    order = np.argsort(log_freq_pool)
    lfs = log_freq_pool[order]
    ins = np.searchsorted(lfs, tgt)
    lo = np.clip(ins - n_near // 2, 0, len(lfs) - n_near)
    cand = order[lo[:, None] + np.arange(n_near)[None, :]]          # (k, n_near)
    err = float(np.median(np.abs(log_freq_pool[cand[:, n_near // 2]] - tgt)))
    r = np.random.default_rng(seed)
    all_pos = np.arange(len(pool_idx))
    c1 = np.empty(n); c2 = np.empty(n)
    for i in range(n):
        pick = np.unique(cand[np.arange(k), r.integers(0, n_near, size=k)])
        if len(pick) < k:                                           # top up on collisions
            extra = r.choice(np.setdiff1d(all_pos, pick), size=k - len(pick), replace=False)
            pick = np.concatenate([pick, extra])
        c1[i], c2[i] = capture(lookup(np.asarray(pool_idx)[pick]), QP, fast=True)
    return c1, c2, err


# ---------------------------------------------------------------------------
# planes
# ---------------------------------------------------------------------------

def label_means(layers, jobs, hf_token=None):
    """Label means of the raw residual stream, all layers in one forward per word.

    Mirrors 12_closure_depth.py exactly: batch the templates for one label word, take
    the LAST non-pad position of each, average over templates. Same hook point
    (`model.model.layers[l]` output, the post-layer residual stream), same dtype
    handling (bf16 forward, float32 mean).
    """
    import torch
    from gfdisc.env import DEVICE
    from gfdisc.model import load_model, capture as cap

    model, tok = load_model(token=hf_token)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    out = {name: {l: [] for l in layers} for name in jobs}
    with torch.inference_mode():
        for name, (words, T) in jobs.items():
            for w in words:
                enc = tok([t.format(w) for t in T], return_tensors="pt",
                          padding=True).to(DEVICE)
                with cap(model, layers) as st:
                    model(**enc, use_cache=False)
                last = enc["attention_mask"].sum(1) - 1
                for l in layers:
                    out[name][l].append(
                        st[l][torch.arange(last.shape[0]), last].float().mean(0).cpu().numpy())
    del model
    return {name: {l: np.vstack(v).astype(np.float64) for l, v in d.items()}
            for name, d in out.items()}


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/span_262k.json")
    ap.add_argument("--planes-cache", default=None,
                    help="npz of month-label means; written if absent, reused if present")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--draws", type=int, default=N_CONTROL_DRAWS)
    ap.add_argument("--smoke", action="store_true", help="layer 9 only, 30 control draws")
    a = ap.parse_args()
    t_start = time.time()
    layers = [9] if a.smoke else LAYERS
    n_draws = 30 if a.smoke else a.draws
    n_strat = min(n_draws, N_STRAT_DRAWS)
    n_proj = 20 if a.smoke else N_PROJ_CONTROL_DRAWS
    n_perm_plane = 2000 if a.smoke else N_PERM_PLANE
    n_perm_proj = 1000 if a.smoke else N_PERM_PROJ

    declared = dict(
        question=("Does the layer-9/16/18 supervised month-ring PLANE lie inside the SPAN "
                  "of the 262k-width month latents' decoder rows, beyond what k random "
                  "alive latents capture by chance?"),
        primary_arm=PRIMARY_ARM, primary_width=PRIMARY_WIDTH, primary_suite=PRIMARY_SUITE,
        site=SITE, layers=LAYERS, primary_plane=PRIMARY_PLANE, statistic=STATISTIC,
        sensitivity_arms=[
            "months/ge1_distinct (weaker: the k/1152 isotropic control ceiling is 0.33-0.59)",
            "months/ge2_distinct_no_may ('may' dropped from the census; the pre-check "
            "measured may=229 of 679 >=1-distinct latents at L18)",
            "months/dominant_min2hits (does not penalise monosemanticity)"],
        controls=["alive_random", "freq_matched", "index_stratified", "day_latents",
                  "placebo_planes (weekday and number-word planes, same layer, "
                  "12_closure_depth.py templates)"],
        n_control_draws=n_draws, n_strat_draws=n_strat, n_proj_control_draws=n_proj,
        control_seed=CONTROL_SEED, alive_threshold=ALIVE_EPS,
        isotropic_reference="E[(cos^2 th1 + cos^2 th2)/2] = k/1152 for a random k-subspace",
        anchor_gate=dict(gap_ratio=ANCHOR_GAP_RATIO, p_closed=ANCHOR_P_CLOSED,
                         tol_gap_ratio=ANCHOR_TOL_GAP,
                         source="results/closure_depth.json, ontology=months"),
        verdict_rule=VERDICT_RULE, sequence_disclosure=SEQUENCE_DISCLOSURE,
        status="exploratory, not pre-registered; labels are post hoc throughout")
    print("DECLARED VERDICT RULE\n" + json.dumps(VERDICT_RULE, indent=1) + "\n", flush=True)

    # ---- 1. ring planes ---------------------------------------------------
    tmpl = PLANE_JOBS
    cache_p = Path(a.planes_cache) if a.planes_cache else None
    have = False
    if cache_p and cache_p.exists():
        z = np.load(cache_p)
        have = all(f"{n}_{l}" in z for n in tmpl for l in layers)
        if have:
            means = {n: {l: z[f"{n}_{l}"] for l in layers} for n in tmpl}
            print(f"planes: reused {cache_p}", flush=True)
    if not have:
        t0 = time.time()
        means = label_means(layers, tmpl, hf_token=os.environ.get("HF_TOKEN"))
        print(f"planes: {len(tmpl)} label/template sets x {len(layers)} layers "
              f"in {time.time()-t0:.0f}s", flush=True)
        if cache_p:
            cache_p.parent.mkdir(parents=True, exist_ok=True)
            np.savez(cache_p, **{f"{n}_{l}": means[n][l] for n in tmpl for l in layers})

    planes, anchor_ok = {}, True
    for l in layers:
        row = {}
        for name in tmpl:
            M = means[name][l]
            _, _, Vt = fits.pca_2d(M)
            # the month planes carry the anchor gate and need the recorded 20,000-perm
            # calibration; the placebos are context and are scored more cheaply
            np_ = n_perm_plane if name in MONTH_PLANES else n_perm_proj
            row[name] = dict(QP=np.ascontiguousarray(Vt[:2].T), Xc=M - M.mean(0),
                             battery=ring_battery(M, seed=l, n_perm=np_))
        s = np.linalg.svd(row["MT16"]["QP"].T @ row["MON_T20"]["QP"], compute_uv=False)
        row["plane_agreement_MT16_vs_MON_T20"] = dict(
            cos2=[float(x ** 2) for x in np.clip(s, 0, 1)],
            angle_deg=[float(np.degrees(np.arccos(np.clip(x, -1, 1)))) for x in s])
        b = row["MT16"]["battery"]
        d_gap = abs(b["gap_ratio"] - ANCHOR_GAP_RATIO[l])
        row["anchor"] = dict(layer=l, gap_ratio=b["gap_ratio"],
                             gap_ratio_recorded=ANCHOR_GAP_RATIO[l], abs_diff=float(d_gap),
                             p_closed=b["p_closed"], p_closed_recorded=ANCHOR_P_CLOSED[l],
                             top2_var=b["top2_var"], rho=b["ordering"]["as_fitted"]["rho"],
                             passed=bool(d_gap <= ANCHOR_TOL_GAP))
        anchor_ok = anchor_ok and row["anchor"]["passed"]
        planes[l] = row
        pa = row["plane_agreement_MT16_vs_MON_T20"]["cos2"]
        print(f"  L{l:<2} MT16 gap_ratio {b['gap_ratio']:.4f} (rec {ANCHOR_GAP_RATIO[l]:.4f}) "
              f"p_closed {b['p_closed']:.2e} (rec {ANCHOR_P_CLOSED[l]:.2e}) "
              f"top2 {b['top2_var']:.3f} rho {b['ordering']['as_fitted']['rho']:.3f} "
              f"anchor={'PASS' if row['anchor']['passed'] else 'FAIL'} | "
              f"vs MON_T20 cos2 {pa[0]:.4f}/{pa[1]:.4f}", flush=True)
    if not anchor_ok:
        print("\nANCHOR GATE FAILED -- the re-derived plane does not reproduce "
              "results/closure_depth.json. Recorded; the span numbers below are "
              "NOT interpretable.", flush=True)

    # ---- 2/3. census + decoder rows + the test ----------------------------
    from transformers import AutoTokenizer
    tk = AutoTokenizer.from_pretrained("google/gemma-3-1b-pt",
                                       token=os.environ.get("HF_TOKEN"))
    vocab = [(tk.convert_ids_to_tokens(i) or "").replace("▁", " ")
             for i in range(tk.vocab_size)]
    luts = {"months": build_lut(vocab, MONTHS_LC), "days": build_lut(vocab, DAYS_LC)}
    del vocab

    results, wide = [], {}
    for l in layers:
        QP = planes[l][PRIMARY_PLANE]["QP"]
        Xc = planes[l][PRIMARY_PLANE]["Xc"]
        for width in ("262k", "16k"):
            t0 = time.time()
            rng = np.random.default_rng(CONTROL_SEED + 1000 * l + (1 if width == "262k" else 2))
            d = dir_for(l, width, PRIMARY_SUITE)
            ex, pa_url = f"{BASE}/{d}/examples.safetensors", f"{BASE}/{d}/params.safetensors"
            cfg = get_json(f"{d}/config.json") or {}
            h, off = retry(read_header_t, ex)
            tt = np.asarray(retry(read_tensor_t, ex, "top_tokens", h, off))
            ff = np.asarray(retry(read_tensor_t, ex, "feature_frequencies", h, off),
                            dtype=np.float64)
            F = int(tt.shape[0])
            alive = ff >= ALIVE_EPS
            mh = label_hits(tt, luts["months"], len(MONTHS_LC))
            dh = label_hits(tt, luts["days"], len(DAYS_LC))
            cm = census(mh, MONTHS_LC)
            cm_nm = census(mh, MONTHS_LC, ignore={"may"})
            cd = census(dh, DAYS_LC)
            del tt, mh, dh

            sets = {
                "months/ge2_distinct": cm["latents_ge2"],
                "months/ge2_distinct_no_may": cm_nm["latents_ge2"],
                "months/dominant_min2hits": sorted(f for f, v in cm["dominant"].items()
                                                   if v[1] >= 2),
                "months/ge1_distinct": cm["latents_ge1"],
                "days/ge2_distinct": cd["latents_ge2"],
            }
            need = np.asarray(sorted({int(f) for v in sets.values() for f in v}),
                              dtype=np.int64)
            prim = np.asarray(sorted(sets[PRIMARY_ARM]), dtype=np.int64)

            ph, poff = retry(read_header_t, pa_url)
            strat_idx, strat_look = None, None
            if width == "16k":
                # the whole 16k decoder is 75 MB in ONE request; cheaper than 10k ranges
                Wall = np.asarray(retry(read_tensor_t, pa_url, "w_dec", ph, poff))
                getW = make_dense_lookup(Wall)
                mask = alive.copy()
                mask[need] = False
                pool_idx, pool_look = np.flatnonzero(mask), getW
            else:
                got, Wsel = read_w_dec_rows(pa_url, ph, poff, need, workers=a.workers)
                getW = make_sparse_lookup(got, Wsel)
                # alive-candidate pool: 160 blocks of 64 consecutive rows at random
                # offsets. Latent index carries no geometry, so a consecutive block is
                # still an arbitrary sample of directions; 160 offsets guard block structure.
                starts = rng.choice(F - POOL_BLOCK_ROWS, size=POOL_BLOCKS, replace=False)
                prows = np.concatenate([np.arange(s, s + POOL_BLOCK_ROWS) for s in starts])
                pg, pool_raw = read_w_dec_rows(pa_url, ph, poff, prows,
                                               workers=a.workers, gap_rows=POOL_BLOCK_ROWS)
                keep = alive[pg] & ~np.isin(pg, need)
                pool_idx, pool_look = pg[keep], make_sparse_lookup(pg[keep], pool_raw[keep])
                # index-stratified pool: one 8-row block within +/-2048 of each month latent
                if len(prim):
                    srows = []
                    for f in prim:
                        lo = max(0, min(F - STRAT_BLOCK, int(f) - STRAT_WINDOW))
                        hi = max(lo, min(F - STRAT_BLOCK, int(f) + STRAT_WINDOW))
                        s0 = int(rng.integers(lo, hi + 1))
                        srows.append(np.arange(s0, s0 + STRAT_BLOCK))
                    srows = np.unique(np.concatenate(srows))
                    sg, sW = read_w_dec_rows(pa_url, ph, poff, srows, workers=a.workers,
                                             gap_rows=STRAT_BLOCK)
                    kp = alive[sg] & ~np.isin(sg, need)
                    strat_idx, strat_look = sg[kp], make_sparse_lookup(sg[kp], sW[kp])

            lf_pool = np.log10(np.maximum(ff[pool_idx], 1e-12))
            fetch_s = time.time() - t0
            print(f"  L{l} {width}: F={F} alive={int(alive.sum())} "
                  f"ge2={len(sets['months/ge2_distinct'])} ge1={len(sets['months/ge1_distinct'])} "
                  f"day_ge2={len(sets['days/ge2_distinct'])} pool={len(pool_idx)} "
                  f"strat={0 if strat_idx is None else len(strat_idx)} "
                  f"({fetch_s:.0f}s, {REQ['n']} reqs total)", flush=True)

            per_set = {}
            for sname, feats in sets.items():
                feats = np.asarray(sorted(feats), dtype=np.int64)
                k = len(feats)
                rec = dict(k=int(k), latents=[int(x) for x in feats[:800]],
                           n_latents_listed=int(min(k, 800)),
                           n_dead_in_set=int((~alive[feats]).sum()) if k else 0,
                           isotropic_expected_capture=float(k / D_MODEL))
                if k < 2:
                    rec["skipped"] = f"k={k} < 2"
                    per_set[sname] = rec
                    continue
                W = getW(feats)
                c1, c2 = capture(W, QP)
                rank = orth(W)[1]
                rec.update(cos2_theta1=c1, cos2_theta2=c2, capture=float((c1 + c2) / 2),
                           span_rank=int(rank),
                           angle1_deg=float(np.degrees(np.arccos(np.sqrt(max(c1, 0.0))))),
                           angle2_deg=float(np.degrees(np.arccos(np.sqrt(max(c2, 0.0))))),
                           between_month_var_captured=var_captured(Xc, W))
                ctl = {}
                nd = draw_null(pool_idx, k, n_draws, CONTROL_SEED + 13 * l + k, pool_look, QP)
                if nd is not None:
                    n1, n2 = nd
                    cap_n = (n1 + n2) / 2
                    ctl["alive_random"] = dict(
                        capture=summarise_null(cap_n), cos2_theta1=summarise_null(n1),
                        cos2_theta2=summarise_null(n2),
                        p_capture=emp_p(rec["capture"], cap_n),
                        p_cos2_theta1=emp_p(c1, n1), p_cos2_theta2=emp_p(c2, n2),
                        excess_ratio=float(rec["capture"] / max(cap_n.mean(), 1e-12)))
                fm = draw_freq_matched(feats, pool_idx, lf_pool, ff, n_draws,
                                       CONTROL_SEED + 101 * l + k, pool_look, QP)
                if fm is not None:
                    f1, f2, ferr = fm
                    capf = (f1 + f2) / 2
                    ctl["freq_matched"] = dict(
                        capture=summarise_null(capf), p_capture=emp_p(rec["capture"], capf),
                        p_cos2_theta1=emp_p(c1, f1), p_cos2_theta2=emp_p(c2, f2),
                        median_abs_dlog10_freq=ferr,
                        excess_ratio=float(rec["capture"] / max(capf.mean(), 1e-12)))
                if strat_idx is not None:
                    sd = draw_null(strat_idx, k, n_strat, CONTROL_SEED + 7 * l + k,
                                   strat_look, QP)
                    if sd is not None:
                        caps = (sd[0] + sd[1]) / 2
                        ctl["index_stratified"] = dict(
                            capture=summarise_null(caps),
                            p_capture=emp_p(rec["capture"], caps),
                            excess_ratio=float(rec["capture"] / max(caps.mean(), 1e-12)))
                rec["controls"] = ctl
                # PLANE-side specificity: the same latents against the weekday and
                # number-word planes of the same layer, each with its own k-matched
                # alive_random null. Run for the primary arm only; the point is whether
                # THESE latents are month-plane specific, not to re-sweep every arm.
                if sname == PRIMARY_ARM:
                    pl = {}
                    for pname in PLACEBO_PLANES:
                        QP2 = planes[l][pname]["QP"]
                        b1, b2 = capture(W, QP2)
                        nd2 = draw_null(pool_idx, k, n_draws,
                                        CONTROL_SEED + 211 * l + k, pool_look, QP2)
                        e = dict(cos2_theta1=b1, cos2_theta2=b2,
                                 capture=float((b1 + b2) / 2),
                                 plane_p_closed=planes[l][pname]["battery"]["p_closed"],
                                 plane_gap_ratio=planes[l][pname]["battery"]["gap_ratio"])
                        if nd2 is not None:
                            cn2 = (nd2[0] + nd2[1]) / 2
                            e.update(alive_random=summarise_null(cn2),
                                     p_capture=emp_p(e["capture"], cn2),
                                     excess_ratio=float(e["capture"] /
                                                        max(cn2.mean(), 1e-12)))
                        pl[pname] = e
                    rec["placebo_planes"] = pl
                    print("     placebo planes: " + "  ".join(
                        f"{p}: cap {v['capture']:.4f} p {v.get('p_capture', float('nan')):.4f} "
                        f"x{v.get('excess_ratio', float('nan')):.2f}"
                        for p, v in pl.items()), flush=True)
                per_set[sname] = rec
                ar = ctl.get("alive_random", {})
                nan = float("nan")
                print(f"     {sname:30s} k={k:5d} cos2 {c1:.4f}/{c2:.4f} cap {rec['capture']:.4f}"
                      f"  iso {k/D_MODEL:.4f} rnd {ar.get('capture',{}).get('mean',nan):.4f}"
                      f" p {ar.get('p_capture',nan):.4f} x{ar.get('excess_ratio',nan):.2f}"
                      f"  fm p {ctl.get('freq_matched',{}).get('p_capture',nan):.4f}"
                      f"  var {rec['between_month_var_captured']:.3f}", flush=True)

            # ---- projection ring test (declared, run unconditionally) ------
            proj = {}
            if len(prim) >= 2:
                Q = orth(getW(prim))[0]
                Mproj = (Xc @ Q) @ Q.T
                Mcomp = Xc - Mproj
                proj["in_span"] = ring_battery(Mproj, seed=l, n_perm=n_perm_proj)
                proj["in_span"]["energy_frac"] = float((Mproj ** 2).sum() /
                                                       max((Xc ** 2).sum(), 1e-30))
                proj["complement"] = ring_battery(Mcomp, seed=l, n_perm=n_perm_proj)
                proj["complement"]["energy_frac"] = float((Mcomp ** 2).sum() /
                                                          max((Xc ** 2).sum(), 1e-30))
                if len(pool_idx) >= len(prim):
                    r = np.random.default_rng(CONTROL_SEED + 31 * l)
                    gr, pc, t2 = [], [], []
                    for _ in range(n_proj):
                        pick = pool_idx[r.choice(len(pool_idx), size=len(prim), replace=False)]
                        Qc = orth(pool_look(pick))[0]
                        Mc = (Xc @ Qc) @ Qc.T
                        xy, ratios, _ = fits.pca_2d(Mc)
                        cx, cy, rr, _ = fits.fit_circle(xy)
                        cc = fits.circularity(xy, cx, cy, rr)
                        gr.append(cc["gap_ratio"]); t2.append(float(ratios[:2].sum()))
                        pc.append(gap_pvalue(12, np.radians(cc["max_gap_deg"]),
                                             n_perm=2000, seed=int(r.integers(1 << 30))))
                    gr = np.asarray(gr); pc = np.asarray(pc)
                    proj["random_span_control"] = dict(
                        n=int(n_proj), k=int(len(prim)),
                        gap_ratio=summarise_null(gr), top2_var=summarise_null(np.asarray(t2)),
                        p_closed_median=float(np.median(pc)),
                        frac_p_closed_lt_05=float(np.mean(pc < 0.05)),
                        p_gap_ratio_lower=float((1 + (gr <= proj["in_span"]["gap_ratio"]).sum())
                                                / (1 + len(gr))),
                        note=("a random k-dim alive span is scored with the same battery; "
                              "p_gap_ratio_lower is the fraction of random spans whose "
                              "projected 12 means close at least as tightly as the month "
                              "span's"))
                print(f"     projection: in-span gap {proj['in_span']['gap_ratio']:.3f} "
                      f"p_closed {proj['in_span']['p_closed']:.4f} energy "
                      f"{proj['in_span']['energy_frac']:.3f} | complement gap "
                      f"{proj['complement']['gap_ratio']:.3f} p_closed "
                      f"{proj['complement']['p_closed']:.4f} energy "
                      f"{proj['complement']['energy_frac']:.3f}", flush=True)

            results.append(dict(
                layer=l, width=width, site=SITE, suite=PRIMARY_SUITE, path=d, d_sae=F,
                nominal_l0=cfg.get("l0"), n_alive=int(alive.sum()), n_dead=int((~alive).sum()),
                census=dict(months_per_label_ge1=cm["per_label_ge1"],
                            n_ge2=cm["n_ge2_distinct"], n_ge1=cm["n_ge1_distinct"],
                            n_ge2_no_may=cm_nm["n_ge2_distinct"],
                            n_dominant_min2hits=cm["n_dominant_min2hits"],
                            n_may_dominant_in_ge2=int(sum(
                                1 for f in cm["latents_ge2"]
                                if cm["dominant"].get(int(f), ("", 0))[0] == "may")),
                            days_n_ge2=cd["n_ge2_distinct"]),
                sets=per_set, projection=proj, pool_size=int(len(pool_idx)),
                strat_pool_size=int(0 if strat_idx is None else len(strat_idx)),
                fetch_seconds=round(fetch_s, 1)))

            if width == "262k":
                wide[l] = dict(feats=prim, getW=getW, pool_idx=pool_idx, pool_look=pool_look)
            else:
                w262 = wide.get(l)
                k16 = len(prim)
                if w262 is not None and k16 >= 2 and len(w262["feats"]) >= k16:
                    c1_16, c2_16 = capture(getW(prim), QP)
                    cap16 = float((c1_16 + c2_16) / 2)
                    r = np.random.default_rng(CONTROL_SEED + 5 * l)
                    sub = np.empty(n_draws)
                    for i in range(n_draws):
                        pick = w262["feats"][r.choice(len(w262["feats"]), size=k16,
                                                      replace=False)]
                        x, y = capture(w262["getW"](pick), QP, fast=True)
                        sub[i] = (x + y) / 2
                    n16 = draw_null(pool_idx, k16, n_draws, CONTROL_SEED + 3 * l,
                                    pool_look, QP)
                    n262 = draw_null(w262["pool_idx"], k16, n_draws, CONTROL_SEED + 9 * l,
                                     w262["pool_look"], QP)
                    results[-1]["width_matched_k"] = dict(
                        k=int(k16), capture_16k=cap16, cos2_16k=[c1_16, c2_16],
                        capture_262k_subsampled=summarise_null(sub),
                        p_16k_at_least_262k_subsample=emp_p(cap16, sub),
                        control_16k_alive_random=summarise_null((n16[0] + n16[1]) / 2)
                        if n16 else None,
                        control_262k_alive_random=summarise_null((n262[0] + n262[1]) / 2)
                        if n262 else None,
                        note=("like-for-like: the 262k >=2-distinct set subsampled to the "
                              "16k set's k at this layer, against the 16k set's own capture "
                              "at that k. No per-latent denominator is invented."))
                    wm = results[-1]["width_matched_k"]
                    print(f"     width@k={k16}: 16k cap {wm['capture_16k']:.4f} vs 262k "
                          f"subsample mean {wm['capture_262k_subsampled']['mean']:.4f} "
                          f"(p {wm['p_16k_at_least_262k_subsample']:.3f}); rnd16k "
                          f"{wm['control_16k_alive_random']['mean']:.4f} rnd262k "
                          f"{wm['control_262k_alive_random']['mean']:.4f}", flush=True)
                del Wall

    # ---- 4. verdict -------------------------------------------------------
    def get(l, width, sname):
        r = next((x for x in results if x["layer"] == l and x["width"] == width), None)
        return (r or {}).get("sets", {}).get(sname)

    ok_rows = {}
    for l in layers:
        r = get(l, "262k", PRIMARY_ARM)
        if r and "capture" in r:
            ok_rows[l] = r
    p_ar = {l: r["controls"]["alive_random"]["p_capture"] for l, r in ok_rows.items()
            if "alive_random" in r.get("controls", {})}
    xr = {l: r["controls"]["alive_random"]["excess_ratio"] for l, r in ok_rows.items()
          if "alive_random" in r.get("controls", {})}
    p_fm = {l: r["controls"]["freq_matched"]["p_capture"] for l, r in ok_rows.items()
            if "freq_matched" in r.get("controls", {})}

    n_layers = len(layers)
    spanned = (len(p_ar) == n_layers and all(v <= 0.005 for v in p_ar.values())
               and all(v >= 1.5 for v in xr.values())
               and len(p_fm) == n_layers and all(v <= 0.05 for v in p_fm.values()))
    not_spanned = len(p_ar) == n_layers and all(v > 0.05 for v in p_ar.values())
    verdict = ("latent_spanned_at_262k" if spanned else
               "not_spanned" if not_spanned else "ambiguous")

    spec = {}
    for l in layers:
        m, dset = ok_rows.get(l), get(l, "262k", "days/ge2_distinct")
        if m and dset and "alive_random" in dset.get("controls", {}) \
                and "alive_random" in m.get("controls", {}):
            me = m["controls"]["alive_random"]["excess_ratio"]
            de = dset["controls"]["alive_random"]["excess_ratio"]
            spec[l] = dict(month_k=m["k"], month_excess=me, month_p=m["controls"]["alive_random"]["p_capture"],
                           day_k=dset["k"], day_excess=de,
                           day_p=dset["controls"]["alive_random"]["p_capture"],
                           month_wins=bool(me > de))
    month_specific = sum(v["month_wins"] for v in spec.values()) >= max(1, (n_layers + 1) // 2)

    # plane-side specificity: month latents vs the month plane against month latents vs
    # each placebo plane, each excess ratio taken against its OWN k-matched null.
    pspec = {}
    for l, r in ok_rows.items():
        pl = r.get("placebo_planes") or {}
        if not pl or l not in xr:
            continue
        pspec[l] = dict(month_plane_excess=xr[l],
                        placebo_excess={p: v.get("excess_ratio") for p, v in pl.items()},
                        placebo_p={p: v.get("p_capture") for p, v in pl.items()},
                        month_plane_wins=bool(all(
                            v.get("excess_ratio") is not None and xr[l] > v["excess_ratio"]
                            for v in pl.values())))
    plane_specific = (len(pspec) > 0 and
                      sum(v["month_plane_wins"] for v in pspec.values())
                      >= max(1, (n_layers + 1) // 2))

    axis_only = sum(1 for r in ok_rows.values()
                    if "alive_random" in r.get("controls", {})
                    and r["controls"]["alive_random"]["p_cos2_theta1"] <= 0.05
                    and r["controls"]["alive_random"]["p_cos2_theta2"] > 0.05)

    ambiguity = None
    if verdict == "ambiguous":
        ambiguity = ("; ".join(f"L{l}: p_alive_random={p_ar.get(l)}, excess_ratio={xr.get(l)}, "
                               f"p_freq_matched={p_fm.get(l)}" for l in layers)
                     + ". What would settle it: (a) the ENCODER side -- w_enc columns are "
                       "strided in the safetensors file, so a month-latent-only "
                       "RECONSTRUCTION of the twelve label means needs the full 1.2 GB "
                       "params file; that is the next cheapest decisive test and it asks "
                       "whether the ring is RECOVERABLE, not merely spanned. (b) a 262k "
                       "activation cache, which would replace the shipped-top-10-token "
                       "census with a measured per-month firing profile and let the span be "
                       "built from latents selected label-blind.")

    summary = dict(
        verdict=verdict, anchor_gate_passed=bool(anchor_ok),
        primary={f"L{l}": dict(
            k=ok_rows[l]["k"], capture=ok_rows[l]["capture"],
            cos2_theta1=ok_rows[l]["cos2_theta1"], cos2_theta2=ok_rows[l]["cos2_theta2"],
            angle1_deg=ok_rows[l]["angle1_deg"], angle2_deg=ok_rows[l]["angle2_deg"],
            isotropic_expected=ok_rows[l]["isotropic_expected_capture"],
            alive_random_mean=ok_rows[l]["controls"]["alive_random"]["capture"]["mean"],
            alive_random_p95=ok_rows[l]["controls"]["alive_random"]["capture"]["p95"],
            p_alive_random=p_ar.get(l), excess_ratio=xr.get(l), p_freq_matched=p_fm.get(l),
            between_month_var_captured=ok_rows[l]["between_month_var_captured"])
            for l in ok_rows},
        specificity=dict(month_specific=bool(month_specific),
                         flag=None if month_specific else "generic_calendar_subspace",
                         per_layer={f"L{k}": v for k, v in spec.items()}),
        plane_specificity=dict(plane_specific=bool(plane_specific),
                               flag=None if plane_specific else "not_plane_specific",
                               per_layer={f"L{k}": v for k, v in pspec.items()}),
        axis_vs_plane=dict(n_layers_axis_only=int(axis_only),
                           flag="axis_not_plane" if axis_only >= max(1, (n_layers + 1) // 2)
                           else None),
        ambiguity_note=ambiguity,
        n_http_requests=int(REQ["n"]), n_transport_retries=int(REQ["retry"]))

    out = dict(status="exploratory, not pre-registered", script=Path(__file__).name,
               repo=REPO, generated_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               declared=declared, summary=summary,
               planes={f"L{l}": dict(
                   {n: planes[l][n]["battery"] for n in PLANE_JOBS},
                   anchor=planes[l]["anchor"],
                   plane_agreement_MT16_vs_MON_T20=planes[l][
                       "plane_agreement_MT16_vs_MON_T20"],
                   month_vs_placebo_plane_cos2={
                       n: [float(x ** 2) for x in np.clip(np.linalg.svd(
                           planes[l]["MT16"]["QP"].T @ planes[l][n]["QP"],
                           compute_uv=False), 0, 1)] for n in PLACEBO_PLANES})
                   for l in layers},
               rows=results, runtime_sec=round(time.time() - t_start, 1))
    p = Path(a.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=1))
    print("\n" + json.dumps(summary, indent=1))
    print(f"\nwrote {a.out}  ({time.time()-t_start:.0f}s, {REQ['n']} requests, "
          f"{REQ['retry']} retries)")


if __name__ == "__main__":
    main()
