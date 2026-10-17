"""Phase 4 / prereg section 2.3 -- gained-metric carrier populations, KILL TEST STAGE ONLY.

Frozen design: docs/phase4_prereg_2026-08-22.md section 2.3 (and R7-R12 of section 1).
This run implements the KILL TEST and nothing downstream of it:

    "Per plane, rank types by energy fraction e_t = ||z_t||^2 / ||x_hat_t||^2.
     KILL TEST (first): does any blind plane at L6 or L7 rank >= 9 of the strict-12
     months in its top-24? If no, the avenue stops and is reported as a negative with
     the best count."

If the answer is yes the script STOPS at the kill test by construction: no promotion,
no half-A/half-B split, no closure adjudication. That gate belongs to the orchestrator.

WHY THIS TEST EXISTS (provenance, before any number is computed)
---------------------------------------------------------------
scripts/34 searched deflated eigenplanes of the RAW type-mean cloud and ranked candidate
members by in-plane RADIUS. It failed for a measured reason: the months rank 23-679 of
3,795 by in-plane radius even in their own plane. The scripts/40 verifier measured the
complementary fact in the GAINED coordinates heads actually read
(x_hat = RMSNorm(R) * (1 + w_in)): ranking by ENERGY FRACTION e_t there selects 11/12
months plus calendar/seasonal displacers at L6. Both facts are about the assembly rule,
not the adjudicator. This script asks whether the second fact survives when the plane is
proposed BLINDLY (SVD of the deflated gained cloud) instead of being the months' own plane.

DECLARED BEFORE ANY READ (written to --out as a stub; declared_at is the audit anchor)
--------------------------------------------------------------------------------------
D1  POPULATION. The 3,795-type pool of results/phase3_token_type_mean_resid_fulln.npz
    (scripts/22 build_pool, corpus count >= 100). No re-derivation, no re-forward.

D2  COORDINATES (primary arm, frozen wording of section 2.3):
        x_hat = RMSNorm(R_L) * (1 + w_in[L]),  L in {6, 7},
    w_in read from the local checkpoint's model.layers.<L>.input_layernorm.weight
    (safetensors, CPU, no forward pass), RMSNorm in float64 with eps = 1e-6, the
    Gemma3RMSNorm semantics of scripts/40.rms_norm_x.

D3  R10 (Addendum H drop rule): residual coordinate 1038 is dropped at L3-L10. Declared
    interpretation, fixed here before any read: the coordinate is dropped from R BEFORE
    the RMS is taken (and its w_in entry with it), so the dropped coordinate contributes
    to neither the direction nor the normaliser. The alternative (drop after RMSNorm) is
    NOT run; this is a deviation-visible choice, not a swept parameter.

D4  DEFLATION. One least-squares solve against the design
        [1, log10(count), 6 token-class indicators, ||x_hat||],
    i.e. population mean, log corpus count, token-class block means and residual norm.
    Token classes are scripts/34.class_features VERBATIM (leading space, capitalised,
    all-caps, contains digit, no alphanumeric, non-ASCII).

D5  PROPOSER. Top-d SVD of the deflated cloud, d = 36; all C(36,2) = 630 coordinate
    planes of the top-36 basis plus 500 random 2-planes inside the same subspace
    (seeded, orthonormalised). 1,130 planes per arm. Frozen before any read.

D6  STATISTIC. Per plane, e_t = ||z_t||^2 / ||x_hat_t||^2 with the DEFLATED x_hat in the
    denominator (primary), and the PRE-deflation x_hat in the denominator (secondary
    diagnostic, reported, never verdict-bearing). Kill statistic = max over the 1,130
    planes of |top-24 by e_t  intersect  strict-12 months|.

D7  KILL CRITERION. PASS iff the primary arm reaches >= 9 of 12 at L6 or at L7. On PASS
    the script stops and reports; on FAIL the avenue is reported as a negative with the
    best count.

D8  CALIBRATION. 200 frequency-pseudo populations, built by the scripts/41b rule: each
    month row is replaced by one of the 5 nearest non-calendar rows in log10 corpus count
    (random tie-breaking, distinct picks). Reported: the distribution of the SAME max-over-
    planes statistic, and p = (1 + #{pseudo >= observed}) / 201. This is what makes the
    observed count a calibrated number rather than a bare integer, and it prices the
    selection over 1,130 planes identically for the null.

D9  DIAGNOSTIC ARM (reported, never verdict-bearing). scripts/40's readout convention is
    x_hat = RMSNorm(R[L-1]) * (1 + w_in[L]) -- the residual entering layer L, not the one
    leaving it. Section 2.3's frozen wording is R_L with w_in[L]. Both are computed; D7 is
    decided on the frozen (primary) one alone. The off-by-one is declared here so that a
    disagreement between them is a reported deviation and not a silent choice.

D10 The months' own plane (top-2 of the centred strict-12 block in the same coordinates)
    is scored by the identical statistic for reference, together with the principal angles
    between that plane and the top-36 subspace -- the direct analogue of scripts/34's
    "0.65 of 2 in the top-32 SVD subspace". It is a reference number, not a candidate.

D11 SEEDS. Random planes: 440000 + 100*layer + arm_index. Frequency-pseudo: 441000 +
    100*layer + arm_index. Fixed here, never resampled.

Nothing else is computed. No MPS, no forwards, no cache streaming (R12 respected trivially).
"""
import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

# --- frozen constants -------------------------------------------------------
COUNT_FLOOR = 100
LAYERS = (6, 7)
D_SVD = 36
N_RANDOM_PLANES = 500
TOP_N = 24                       # section 2.3 "top-24"
KILL_MIN_MONTHS = 9              # section 2.3 ">= 9 of the strict-12"
N_PSEUDO = 200
FREQ_PSEUDO_NEAREST = 5          # scripts/41b FREQ_PSEUDO_NEAREST
RMS_EPS = 1e-6                   # config.rms_norm_eps (scripts/40 RMS_EPS)
DROP_COORDS = (1038,)            # R10 at L3-L10 (Addendum H)
NPZ = "results/phase3_token_type_mean_resid_fulln.npz"
SNAP = ("cache/hf/hub/models--google--gemma-3-1b-pt/snapshots/"
        "fcf18a2a879aab110ca39f8bffbccd5d49d8eb29/model.safetensors")
MONTH_NAMES = ["January", "February", "March", "April", "May", "June", "July",
               "August", "September", "October", "November", "December"]


# ===========================================================================
# helpers copied (not imported: these live in scripts, not in the package)
# ===========================================================================
def type_string(tokenizer, tid):                       # scripts/34
    s = tokenizer.convert_ids_to_tokens(int(tid))
    if s is None:
        s = tokenizer.decode([int(tid)])
    return s.replace("▁", " ")


def class_features(s):                                 # scripts/34 VERBATIM
    alphas = [c for c in s if c.isalpha()]
    all_caps = bool(alphas) and all(c.isupper() for c in alphas)
    capitalized = bool(alphas) and alphas[0].isupper() and not all_caps
    return [s.startswith(" "), capitalized, all_caps,
            any(c.isdigit() for c in s),
            not any(c.isalnum() for c in s),
            any(ord(c) > 127 for c in s)]


def deflate(M, D):                                     # scripts/34 VERBATIM
    M = np.asarray(M, dtype=np.float64)
    coef, *_ = np.linalg.lstsq(D, M, rcond=None)
    return M - D @ coef


def rms_norm_x(R, w_in, eps=RMS_EPS):                  # scripts/40 VERBATIM
    R = np.asarray(R, dtype=np.float64)
    inv = 1.0 / np.sqrt((R ** 2).mean(1, keepdims=True) + eps)
    return (R * inv) * w_in[None, :]


def member_hash(members):                              # scripts/34 VERBATIM
    return hashlib.md5(",".join(str(int(x)) for x in sorted(members)).encode()).hexdigest()[:12]


# ===========================================================================
def read_w_in(path, layers):
    """model.layers.<L>.input_layernorm.weight from the local checkpoint, CPU, no forward."""
    from safetensors import safe_open
    out = {}
    with safe_open(str(path), framework="pt", device="cpu") as f:
        for L in layers:
            t = f.get_tensor(f"model.layers.{L}.input_layernorm.weight")
            out[L] = 1.0 + t.float().numpy().astype(np.float64)
    return out


def build_planes(Y, seed):
    """(name, kind, z) generator over the frozen proposer: C(d,2) coordinate + 500 random."""
    d = Y.shape[1]
    planes = []
    for i in range(d):
        for j in range(i + 1, d):
            planes.append((f"coord_{i}_{j}", "coordinate", Y[:, [i, j]]))
    rng = np.random.default_rng(seed)
    for r in range(N_RANDOM_PLANES):
        G = rng.normal(size=(d, 2))
        Q, _ = np.linalg.qr(G)
        planes.append((f"rand_{r}", "random", Y @ Q))
    return planes


def top_sets(planes, den, top_n=TOP_N):
    """Boolean [n_planes, n_types] membership of the top-n by energy fraction."""
    n_types = len(den)
    memb = np.zeros((len(planes), n_types), dtype=bool)
    for p, (_, _, z) in enumerate(planes):
        e = (z ** 2).sum(1) / den
        idx = np.argpartition(-e, top_n)[:top_n]
        memb[p, idx] = True
    return memb


def freq_pseudo_sets(month_rows, counts, excluded_rows, n_draws, seed,
                     n_nearest=FREQ_PSEUDO_NEAREST):
    """scripts/41b rule: each member replaced by a nearest-corpus-count non-calendar row."""
    lc = np.log10(np.maximum(counts.astype(np.float64), 1.0))
    allowed = np.ones(len(lc), dtype=bool)
    allowed[np.asarray(sorted(excluded_rows), dtype=np.int64)] = False
    cand = np.flatnonzero(allowed)
    nearest = np.asarray([cand[np.argsort(np.abs(lc[cand] - lc[p]), kind="stable")[:n_nearest]]
                          for p in month_rows])
    rng = np.random.default_rng(seed)
    sets, n_retries = [], 0
    k = len(month_rows)
    for _ in range(n_draws):
        pick = None
        for _a in range(20):
            cp = nearest[np.arange(k), rng.integers(0, n_nearest, size=k)]
            if len(set(cp.tolist())) == k:
                pick = cp
                break
            n_retries += 1
        sets.append(pick if pick is not None else cp)
    return np.asarray(sets), int(n_retries), int(len(cand))


def principal_angles(B1, B2):
    """Degrees between two 2-planes / a plane and a subspace (both column-orthonormal)."""
    s = np.linalg.svd(B1.T @ B2, compute_uv=False)
    return [float(np.degrees(np.arccos(np.clip(x, -1.0, 1.0)))) for x in s]


# ===========================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/phase4_gained_carriers.json")
    ap.add_argument("--npz", default=NPZ)
    a = ap.parse_args()
    t_start = time.time()
    out_path = REPO / a.out

    # ---- D-stub BEFORE any read ------------------------------------------
    declared_at = datetime.now().isoformat(timespec="seconds")
    stub = dict(
        script="scripts/44_gained_carrier_populations.py",
        item="Phase 4 section 2.3 -- gained-metric carrier populations, KILL TEST STAGE ONLY",
        prereg="docs/phase4_prereg_2026-08-22.md section 2.3 (FROZEN) + section 1 R7-R12",
        status="DECLARATION STUB (written before any read; overwritten by the run)",
        declared_at=declared_at,
        declarations=__doc__,
        rule=("x_hat = RMSNorm(R_L)*(1+w_in[L]) at L in {6,7}; R10 drop {1038} before the RMS; "
              "deflate [1, log10 count, 6 token-class indicators, ||x_hat||]; top-36 SVD; "
              "C(36,2)=630 coordinate planes + 500 seeded random planes in the top-36 subspace; "
              "rank types by e_t = ||z_t||^2/||x_hat_t||^2; statistic = max over planes of "
              "|top-24 intersect strict-12 months|"),
        kill_criterion=(f">= {KILL_MIN_MONTHS} of the strict-12 months in the top-{TOP_N} of at "
                        f"least one blind plane, at L6 or L7, in the PRIMARY arm; on PASS the "
                        f"script stops at the kill test and does not promote"),
        calibration=(f"{N_PSEUDO} frequency-pseudo populations (scripts/41b rule, "
                     f"{FREQ_PSEUDO_NEAREST} nearest log10-count non-calendar rows) through the "
                     f"identical max-over-planes path"),
        config=dict(layers=list(LAYERS), d_svd=D_SVD, n_random_planes=N_RANDOM_PLANES,
                    top_n=TOP_N, kill_min_months=KILL_MIN_MONTHS, n_pseudo=N_PSEUDO,
                    rms_eps=RMS_EPS, drop_coords=list(DROP_COORDS), npz=a.npz,
                    freq_pseudo_nearest=FREQ_PSEUDO_NEAREST, no_mps=True, no_forwards=True),
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(stub, indent=1))
    print(f"declaration stub -> {out_path} at {declared_at}", flush=True)

    # ---- population -------------------------------------------------------
    z = np.load(REPO / a.npz)
    pool_ids = np.asarray(z["pool_ids"], dtype=np.int64)
    counts = np.asarray(z["counts"], dtype=np.int64)
    n_types = len(pool_ids)
    assert n_types == 3795, n_types
    R = {int(L): np.asarray(z[f"R_L{int(L)}"], dtype=np.float64) for L in (5, 6, 7)}
    print(f"pool {n_types} types (count >= {COUNT_FLOOR}); R layers loaded: {sorted(R)}",
          flush=True)

    os.environ.setdefault("HF_HOME", str(REPO / "cache/hf"))
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from transformers import AutoTokenizer
    tk = AutoTokenizer.from_pretrained("google/gemma-3-1b-pt")
    strings = [type_string(tk, int(t)) for t in pool_ids]
    feats = np.asarray([class_features(s) for s in strings], dtype=bool)

    token_map = json.loads((REPO / "results/token_map.json").read_text())
    row_of = {int(t): i for i, t in enumerate(pool_ids)}
    strict12_tokens, month_rows = [], []
    for m in MONTH_NAMES:
        ip = sorted((int(x) for x in token_map["months"][m] if int(x) in row_of),
                    key=lambda x: -int(counts[row_of[x]]))
        strict12_tokens.append(ip[0])
        month_rows.append(row_of[ip[0]])
    month_rows = np.asarray(month_rows, dtype=np.int64)
    calendar_rows = sorted({row_of[int(x)]
                            for fam in ("months", "days")
                            for v in token_map[fam].values()
                            for x in v if int(x) in row_of})
    print(f"strict-12 token ids {strict12_tokens}; rows {month_rows.tolist()}; "
          f"member_hash {member_hash(month_rows)}; calendar rows excluded from pseudo: "
          f"{len(calendar_rows)}", flush=True)
    print("strict-12 strings: " + repr([strings[i] for i in month_rows]), flush=True)

    w_in_full = read_w_in(REPO / SNAP, sorted({6, 7}))
    keep = np.setdiff1d(np.arange(R[6].shape[1]), np.asarray(DROP_COORDS))
    design_base = np.column_stack([np.ones(n_types),
                                   np.log10(np.maximum(counts.astype(np.float64), 1.0)),
                                   feats.astype(np.float64)])

    arms = [("primary_frozen_L6", 6, 6, True), ("primary_frozen_L7", 7, 7, True),
            ("diag_head_readout_L6", 5, 6, False), ("diag_head_readout_L7", 6, 7, False)]

    results, kill_pass, best_primary = {}, False, {}
    for arm_index, (arm, R_layer, w_layer, is_primary) in enumerate(arms):
        t0 = time.time()
        X_raw = rms_norm_x(R[R_layer][:, keep], w_in_full[w_layer][keep])
        norm_raw = np.maximum((X_raw ** 2).sum(1), 1e-300)
        D = np.column_stack([design_base, np.sqrt(norm_raw)])
        X = deflate(X_raw, D)
        norm_def = np.maximum((X ** 2).sum(1), 1e-300)

        U, S, Vt = np.linalg.svd(X, full_matrices=False)
        V36 = np.ascontiguousarray(Vt[:D_SVD].T)                     # 1151 x 36
        Y = X @ V36
        var_frac = float((S[:D_SVD] ** 2).sum() / (S ** 2).sum())
        planes = build_planes(Y, seed=440000 + 100 * w_layer + arm_index)

        # months' own plane, same coordinates (reference only, D10)
        Mn = X[month_rows] - X[month_rows].mean(0, keepdims=True)
        Bm = np.ascontiguousarray(np.linalg.svd(Mn, full_matrices=False)[2][:2].T)
        zm = X @ Bm
        cos_pa = np.linalg.svd(V36.T @ Bm, compute_uv=False)
        arm_out = dict(R_layer=R_layer, w_in_layer=w_layer, primary=bool(is_primary),
                       n_planes=len(planes), top36_var_fraction=var_frac,
                       month_plane_energy_in_top36=float((cos_pa ** 2).sum()),
                       month_plane_principal_angles_deg=principal_angles(V36, Bm))

        for den_name, den in (("deflated", norm_def), ("pre_deflation", norm_raw)):
            memb = top_sets(planes, den)
            counts_months = memb[:, month_rows].sum(1)
            best = int(np.argmax(counts_months))
            best_count = int(counts_months[best])

            pseudo_sets, n_retries, n_cand = freq_pseudo_sets(
                month_rows, counts, calendar_rows, N_PSEUDO,
                seed=441000 + 100 * w_layer + arm_index)
            pseudo_max = np.asarray([int(memb[:, s].sum(1).max()) for s in pseudo_sets])
            p_cal = float((1 + int((pseudo_max >= best_count).sum())) / (1 + len(pseudo_max)))

            e_m = ((zm ** 2).sum(1) / den)
            top_m = np.argpartition(-e_m, TOP_N)[:TOP_N]
            month_plane_count = int(np.isin(top_m, month_rows).sum())

            best_idx = np.flatnonzero(memb[best])
            best_e = ((planes[best][2] ** 2).sum(1) / den)[best_idx]
            order = np.argsort(-best_e)
            cell = dict(
                best_count=best_count, best_plane=planes[best][0], best_plane_kind=planes[best][1],
                best_count_by_kind={
                    k: int(counts_months[[i for i, p in enumerate(planes) if p[1] == k]].max())
                    for k in ("coordinate", "random")},
                n_planes_at_best=int((counts_months == best_count).sum()),
                count_distribution_over_planes={
                    str(v): int(c) for v, c in zip(*np.unique(counts_months, return_counts=True))},
                pseudo_calibration=dict(
                    n=int(len(pseudo_max)), n_candidate_types=n_cand, n_duplicate_redraws=n_retries,
                    mean=float(pseudo_max.mean()), median=float(np.median(pseudo_max)),
                    p90=float(np.percentile(pseudo_max, 90)), p95=float(np.percentile(pseudo_max, 95)),
                    p99=float(np.percentile(pseudo_max, 99)), max=int(pseudo_max.max()),
                    hist={str(v): int(c) for v, c in zip(*np.unique(pseudo_max, return_counts=True))},
                    p_value_vs_observed=p_cal),
                month_own_plane_count=month_plane_count,
                best_plane_top24=[dict(rank=int(r + 1), row=int(best_idx[order][r]),
                                       token_id=int(pool_ids[best_idx[order][r]]),
                                       string=strings[best_idx[order][r]],
                                       count=int(counts[best_idx[order][r]]),
                                       e=float(best_e[order][r]),
                                       is_strict12_month=bool(best_idx[order][r] in set(
                                           month_rows.tolist())))
                                  for r in range(TOP_N)],
            )
            arm_out[den_name] = cell
            print(f"[{arm}/{den_name}] best {best_count}/12 on {planes[best][1]} plane "
                  f"{planes[best][0]}; months' own plane {month_plane_count}/12; pseudo max "
                  f"mean {pseudo_max.mean():.2f} p99 {np.percentile(pseudo_max, 99):.1f} "
                  f"max {pseudo_max.max()}; p {p_cal:.4f}", flush=True)
            if is_primary and den_name == "deflated":
                best_primary[arm] = best_count
                kill_pass = kill_pass or best_count >= KILL_MIN_MONTHS

        arm_out["seconds"] = round(time.time() - t0, 1)
        results[arm] = arm_out
        print(f"[{arm}] done in {arm_out['seconds']}s (top-36 var {var_frac:.3f}; month plane "
              f"energy in top-36 {arm_out['month_plane_energy_in_top36']:.3f} of 2)", flush=True)

    verdict = dict(
        kill_test="PASS" if kill_pass else "FAIL",
        rule=f"PASS iff primary arm (deflated denominator) reaches >= {KILL_MIN_MONTHS}/12 at L6 or L7",
        best_primary_counts=best_primary,
        next_step=("STOP at the kill test and report; promotion is orchestrator-gated"
                   if kill_pass else
                   "avenue A reported as a NEGATIVE with the best count; no promotion stage"),
    )
    stub.update(status="COMPLETE (kill test stage only)", verdict=verdict, arms=results,
                strict12_token_ids=[int(x) for x in strict12_tokens],
                strict12_rows=[int(x) for x in month_rows],
                strict12_member_hash=member_hash(month_rows),
                n_calendar_rows_excluded=len(calendar_rows),
                runtime_seconds=round(time.time() - t_start, 1))
    out_path.write_text(json.dumps(stub, indent=1))
    print(f"\nKILL TEST = {verdict['kill_test']}  primary {best_primary}", flush=True)
    print(f"wrote {a.out} ({stub['runtime_seconds']}s)", flush=True)


if __name__ == "__main__":
    main()
