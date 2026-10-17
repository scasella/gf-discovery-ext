"""Does a base-10 helix exist in the residual stream at all? (supervised existence check)

Phase 1 asks whether cyclic geometry can be *discovered* without labels. That question
is only worth asking where the geometry is present, so this script runs the cheap
supervised version first, on the one ontology with a published positive result to
compare against: the Kantamneni & Tegmark base-10 helix for integer tokens.

Two disciplines are imported from the calendar work so the helix is not held to a
weaker standard than the cycles were:

  (i)  a shuffled-values null. `fit_helix` adds 8 periodic columns on top of the linear
       baseline, so the gain r2_helix - r2_linear is mechanically non-negative and is
       positive for *any* point cloud at n=100. Only the permutation null says whether
       it is more than free parameters.
  (ii) closure. The calendar sweep established that ordering and closure dissociate --
       a monotone 1-D arrangement scores rho = 1.000 on the cyclic-ordering statistic.
       The same trap applies here: a linear ramp with a periodic wobble in the residual
       is not a helix. So the T=10 (cos, sin) plane is extracted from the fit, the
       points are projected into it and averaged by value mod 10, and the resulting
       10 points face `circularity` plus an angular-gap p-value. Which p-value is not
       obvious -- see the comment block below, where two candidates are disqualified.

Tokenization (D0). Gemma 3 splits digits: of "0".."99" only the ten bare single digits
are single tokens, and a leading space is a separate token ("42" -> ' ', '4', '2'). There
is therefore no single-token subset above 9 to restrict to. Every template ends in the
placeholder, so the last non-pad position is the number's FINAL DIGIT, and that position
is labelled with the full integer value. The convention is recorded in the output JSON.
"""
# ---------------------------------------------------------------------------
# Closure null: two versions were disqualified before the third was kept.
# ---------------------------------------------------------------------------
# This is the fifth statistic in this project to fail because it could not tell a
# closed ring from a straight line, so the measurements are recorded here rather than
# in a commit message. The battery in `synthetics` is the arbiter: a planted helix must
# pass, a pure linear ramp must fail, and pure noise must trip at the nominal rate.
#
#   uniform random angles (the script 12 null)  -- DISQUALIFIED.
#       Assumes the plane was chosen without reference to the labels. True of the PCA
#       plane script 12 uses, false here: the T=10 plane is the one the fit selected to
#       maximise mod-10 periodicity, and the ten points are then averaged by mod 10. So
#       the test asks of a fit-selected plane the same question the fit already optimised.
#       Measured: pure linear ramp p_closed = 1.5e-4, planted helix p_closed = 5.0e-5.
#       It cannot separate them. Retained as `p_closed_uniform` for comparability with
#       closure_depth.json; it decides nothing.
#
#   shuffled values, no detrending          -- DISQUALIFIED, and in the unsafe direction.
#       Permuting the values destroys the LINEAR structure as well as the periodic
#       structure, so under the null the ramp leaks into the fitted plane and the ten bin
#       means clump. Measured: null gap_ratio 7.02 for a pure ramp whose own gap_ratio is
#       1.32 -- against which the ramp scores p = 0.0099, the floor at n_null = 100.
#       An inflated null makes everything look closed.
#
#   shuffled values after detrending on the TRUE values  -- KEPT.
#       The constant and linear-in-value components are removed first, so the null
#       removes only the periodic structure, which is what is being tested, and a pure
#       ramp reduces to exchangeable noise under it. Measured on the battery:
#       planted helix p_closed = 0.0099, linear ramp 0.545, and over five noise draws
#       0/5 gain trips and 1/5 closure trips at alpha = 0.05 (nominal).
#
# Note that rho_mod10 = +1.000 for the linear ramp under every one of the three. Ordering
# never distinguished anything here and is reported only as a diagnostic.
import argparse, json, os, sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from gfdisc.env import DEVICE
from gfdisc.model import load_model, capture
from gfdisc.fits import fit_helix, fit_circle, circularity, angles, cyclic_order_score

# Numeric-context prompts, extended from the NT control list in script 12. Every one
# ends in the placeholder: `collect` captures the last non-pad token, and a template with
# trailing words ("There were {} of them") would capture "them" instead of the digit.
NT = ["He counted to {}", "The total came to {}", "The score was {}",
      "The box holds {}", "The number is {}", "She bought exactly {}",
      "The list had {}", "They hired {}", "The answer is {}",
      "We waited an hour and counted {}", "The page number was {}",
      "The final tally reached {}", "It costs {}", "The class has {}",
      "The batch contained {}", "The reading showed {}"]

PERIODS = (2, 5, 10, 100)


def design(v, use_periodic=True, periods=PERIODS):
    """The same design matrix `fits.fit_helix` builds, exposed so the fitted
    coefficients (which it does not return) can be used for the closure test."""
    v = np.asarray(v, dtype=np.float64)
    cols = [np.ones_like(v), v]
    if use_periodic:
        for T in periods:
            cols += [np.cos(2 * np.pi * v / T), np.sin(2 * np.pi * v / T)]
    return np.column_stack(cols)


def gap_pvalue(k, observed_max_gap, n_perm=20000, seed=0):
    """Max angular gap against uniform random angles -- as in script 12."""
    rng = np.random.default_rng(seed)
    a = np.sort(rng.uniform(0, 2 * np.pi, size=(n_perm, k)), axis=1)
    g = np.diff(np.concatenate([a, a[:, :1] + 2 * np.pi], axis=1), axis=1).max(1)
    return float((1 + (g <= observed_max_gap).sum()) / (1 + n_perm))


def detrend(X, values):
    """Remove the constant and linear-in-value component, using the TRUE values.

    Everything downstream of this is about the periodic part only, and it has to stay
    that way under permutation as well -- see `helix_row`.
    """
    B = design(values, use_periodic=False)
    coef, *_ = np.linalg.lstsq(B, X, rcond=None)
    return X - B @ coef


def t10_plane_points(Xr, values):
    """Project the detrended points onto the fitted T=10 plane, average by value mod 10.

    The two coefficient vectors for the T=10 cos/sin columns span the plane in which a
    genuine base-10 helix does its turning. They are not orthogonal in general, so they
    are orthonormalised before projecting; otherwise a shear would manufacture or destroy
    angular evenness.
    """
    v = np.asarray(values, dtype=np.float64)
    B = np.column_stack([np.cos(2 * np.pi * v / 10), np.sin(2 * np.pi * v / 10)])
    coef, *_ = np.linalg.lstsq(B, Xr, rcond=None)
    Q, _ = np.linalg.qr(coef.T)                          # [d_model, 2]
    if Q.shape[1] < 2:
        return None
    xy_all = Xr @ Q
    mod = np.asarray(values, dtype=int) % 10
    return np.vstack([xy_all[mod == r].mean(0) for r in range(10)])


def _gap_ratio(xy):
    cx, cy, r, _ = fit_circle(xy)
    return circularity(xy, cx, cy, r)["gap_ratio"]


def _r2(B, X):
    coef, *_ = np.linalg.lstsq(B, X, rcond=None)
    resid = X - B @ coef
    return 1 - (resid ** 2).sum() / max((X ** 2).sum(), 1e-12)


def single_period_gain(X, values, T):
    """R^2 gain of adding ONE period's cos/sin pair to the linear baseline.

    The pooled gain over periods (2, 5, 10, 100) does not say that base 10 is doing the
    work. At a final-digit capture position T=2 and T=5 are close to properties of the
    digit token itself, and T=100 is a single cycle across the whole range, so it is a
    smooth curvature basis rather than a periodicity. Only the T=10 term is the claim.
    """
    v = np.asarray(values, dtype=np.float64)
    lin = np.column_stack([np.ones_like(v), v])
    per = np.column_stack([lin, np.cos(2 * np.pi * v / T), np.sin(2 * np.pi * v / T)])
    return float(_r2(per, X) - _r2(lin, X))


def helix_row(points, values, n_null=200, seed=0):
    """Gain and T=10 closure, both calibrated against the SAME shuffled-values null.

    Closure needs the permutation null just as much as the gain does, and the synthetic
    battery is what forced the issue: a pure linear ramp scored gap_ratio 1.32 with
    p = 0.0001 against uniform random angles. The plane is *chosen by the fit* to be the
    one in which mod-10 periodicity is strongest, and the ten points are then averaged
    by mod 10 -- so the uniform-angle null of script 12, which assumes a label-free
    (PCA) plane, does not apply here. `p_closed_uniform` is retained for comparability
    with `closure_depth.json`, but it does not decide anything.

    The naive permutation null is broken too, and in the dangerous direction: shuffling
    the values destroys the LINEAR structure as well as the periodic structure, so under
    the null the ramp leaks into the fitted plane and the ten bin means clump
    (null gap_ratio 7.0 for a pure ramp). Measured against that, a ramp's own 1.32 looks
    highly significant. The closure test is therefore run on data detrended with the TRUE
    values, so the null removes only the periodic structure -- which is what is being
    tested -- and a pure ramp reduces to exchangeable noise under it.
    """
    X = np.asarray(points, dtype=np.float64)
    X = X - X.mean(0, keepdims=True)
    v = np.asarray(values, dtype=np.float64)

    h = fit_helix(X, v, periods=PERIODS)
    gain = h["r2_helix"] - h["r2_linear"]
    Xr = detrend(X, v)
    xy = t10_plane_points(Xr, v)
    gr = _gap_ratio(xy)

    by_period = {str(T): single_period_gain(X, v, T) for T in PERIODS}
    g10 = by_period["10"]

    rng = np.random.default_rng(seed)
    null_gain, null_gr, null_g10 = np.empty(n_null), np.empty(n_null), np.empty(n_null)
    for i in range(n_null):
        vp = rng.permutation(v)
        hp = fit_helix(X, vp, periods=PERIODS)
        null_gain[i] = hp["r2_helix"] - hp["r2_linear"]
        null_gr[i] = _gap_ratio(t10_plane_points(Xr, vp))
        null_g10[i] = single_period_gain(X, vp, 10)

    cx, cy, r, resid = fit_circle(xy)
    c = circularity(xy, cx, cy, r)
    s = cyclic_order_score(angles(xy, cx, cy), np.arange(10), 10, n_perm=20000, seed=seed)
    closure = dict(k=10, resid=float(resid), radius_ratio=c["radius_ratio"],
                   gap_ratio=gr, max_gap_deg=c["max_gap_deg"],
                   p_closed_uniform=gap_pvalue(10, np.radians(c["max_gap_deg"]), seed=seed),
                   # smaller gap_ratio = more evenly closed, so the tail is the LOW side
                   p_closed=float((1 + (null_gr <= gr).sum()) / (1 + n_null)),
                   null_gap_mean=float(null_gr.mean()),
                   null_gap_p05=float(np.percentile(null_gr, 5)),
                   rho_mod10=s["rho"], p_order_mod10=s["p"])
    stats = dict(r2_helix=h["r2_helix"], r2_linear=h["r2_linear"], gain=float(gain),
                 null_mean=float(null_gain.mean()), null_p95=float(np.percentile(null_gain, 95)),
                 p_gain=float((1 + (null_gain >= gain).sum()) / (1 + n_null)),
                 gain_by_period=by_period, gain_t10=g10,
                 null_t10_p95=float(np.percentile(null_g10, 95)),
                 p_gain_t10=float((1 + (null_g10 >= g10).sum()) / (1 + n_null)),
                 n_null=n_null)
    return stats, closure


def synthetics():
    """Re-validate the fit before trusting it on real activations.

    A planted helix must produce a gain far outside the shuffled null and a closed T=10
    ring; a pure linear ramp must produce a gain the null already explains. The ramp case
    matters most: the gain is non-negative by construction, so 'gain > 0' is not evidence.

    Pure noise is checked as a calibration rather than a single pass/fail: under an
    exchangeable null both p-values are uniform, so one noise draw trips at 0.05 about
    5% of the time by definition and asserting on a single seed would just be seed-shopping.
    Five independent draws are run and at most one trip per statistic is allowed.
    """
    rng = np.random.default_rng(0)
    v = np.arange(100)
    d = 1152
    a, b, c = rng.normal(size=d), rng.normal(size=d), rng.normal(size=d)
    out = {}

    X = (np.outer(v / 50.0, a) + np.outer(np.cos(2 * np.pi * v / 10), b)
         + np.outer(np.sin(2 * np.pi * v / 10), c) + 0.15 * rng.normal(size=(100, d)))
    st, clo = helix_row(X, v, n_null=100, seed=1)
    out["planted_helix"] = dict(**st, closure=clo)

    Y = np.outer(v / 50.0, a) + 0.15 * rng.normal(size=(100, d))
    st, clo = helix_row(Y, v, n_null=100, seed=2)
    out["linear_ramp"] = dict(**st, closure=clo)

    noise = []
    for s in range(3, 8):
        st, clo = helix_row(rng.normal(size=(100, d)), v, n_null=100, seed=s)
        noise.append(dict(seed=s, p_gain=st["p_gain"], p_closed=clo["p_closed"],
                          gain=st["gain"], gap_ratio=clo["gap_ratio"]))
    out["pure_noise"] = dict(draws=noise,
                             n_sig_gain=sum(x["p_gain"] <= 0.05 for x in noise),
                             n_sig_closed=sum(x["p_closed"] <= 0.05 for x in noise))

    ok = (out["pure_noise"]["n_sig_gain"] <= 1
          and out["pure_noise"]["n_sig_closed"] <= 1
          and out["planted_helix"]["p_gain"] <= 0.05
          and out["planted_helix"]["closure"]["p_closed"] <= 0.05
          and out["linear_ramp"]["p_gain"] > 0.05
          and out["linear_ramp"]["closure"]["p_closed"] > 0.05)
    out["passed"] = bool(ok)
    return out


def collect_all_layers(model, tok, layers, values, templates):
    """One forward per value, capturing every layer -- the script 12 pattern."""
    means = {l: [] for l in layers}
    with torch.inference_mode():
        for v in values:
            enc = tok([t.format(v) for t in templates], return_tensors="pt",
                      padding=True).to(DEVICE)
            with capture(model, layers) as store:
                model(**enc, use_cache=False)
            last = enc["attention_mask"].sum(1) - 1
            idx = torch.arange(last.shape[0], device=DEVICE)
            for l in layers:
                means[l].append(store[l][idx, last].float().mean(0).cpu().numpy())
    return {l: np.vstack(means[l]) for l in layers}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", default=",".join(str(i) for i in range(26)))
    ap.add_argument("--n-null", type=int, default=200)
    ap.add_argument("--out", default="results/helix_existence.json")
    a = ap.parse_args()

    syn = synthetics()
    print(f"synthetic battery: planted helix gain {syn['planted_helix']['gain']:.4f} "
          f"(p {syn['planted_helix']['p_gain']:.4f}, p_closed "
          f"{syn['planted_helix']['closure']['p_closed']:.4f}) | linear ramp gain "
          f"{syn['linear_ramp']['gain']:.4f} (p {syn['linear_ramp']['p_gain']:.4f}, "
          f"p_closed {syn['linear_ramp']['closure']['p_closed']:.4f}) | noise "
          f"{syn['pure_noise']['n_sig_gain']}/5 gain and "
          f"{syn['pure_noise']['n_sig_closed']}/5 closure trips "
          f"-> {'PASS' if syn['passed'] else 'FAIL'}\n", flush=True)
    if not syn["passed"]:
        raise SystemExit("fit_helix failed its synthetic battery; not running on activations")

    token = os.environ["HF_TOKEN"]
    layers = [int(x) for x in a.layers.split(",")]
    values = list(range(100))

    model, tok = load_model(token=token)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"

    # D0, recorded rather than assumed: which of "0".."99" are single tokens.
    bare = [str(v) for v in values if len(tok.encode(str(v), add_special_tokens=False)) == 1]
    spaced = [str(v) for v in values if len(tok.encode(" " + str(v), add_special_tokens=False)) == 1]
    convention = dict(
        n_values=len(values),
        single_token_bare=bare, n_single_token_bare=len(bare),
        single_token_leading_space=spaced, n_single_token_leading_space=len(spaced),
        position="last non-pad token = final digit of the number "
                 "(every template ends in the placeholder)",
        value="full integer, not the final digit alone",
        note="Gemma 3 digit-splits; only the bare single digits are single tokens and a "
             "leading space is its own token, so no single-token subset above 9 exists.",
        confound="At the final-digit position, value mod 10 IS the identity of the token "
                 "being read, so a closed T=10 ring can be digit-token embedding geometry "
                 "rather than a computed helix over the full value. Expect this to be "
                 "strongest at layer 0; only persistence into mid layers is informative.")
    print(f"D0: {len(bare)} bare single-token values, {len(spaced)} with leading space; "
          f"using final-digit position for all {len(values)}\n", flush=True)

    M = collect_all_layers(model, tok, layers, values, NT)

    rows = []
    print(f"{'layer':>5} {'r2_lin':>8} {'gain':>8} {'p_gain':>8} | "
          f"{'g_T2':>7} {'g_T5':>7} {'g_T10':>7} {'g_T100':>7} {'p_T10':>7} | "
          f"{'gapR10':>7} {'p_closed':>9} {'rho10':>7}")
    for l in layers:
        stats, clo = helix_row(M[l], values, n_null=a.n_null, seed=l)
        rows.append(dict(layer=l, **stats, closure=clo))
        bp = stats["gain_by_period"]
        print(f"{l:5d} {stats['r2_linear']:8.4f} {stats['gain']:8.4f} "
              f"{stats['p_gain']:8.4f} | {bp['2']:7.4f} {bp['5']:7.4f} {bp['10']:7.4f} "
              f"{bp['100']:7.4f} {stats['p_gain_t10']:7.4f} | "
              f"{clo['gap_ratio']:7.2f} {clo['p_closed']:9.4f} {clo['rho_mod10']:+7.3f}",
              flush=True)

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(dict(
        tokenization=convention, templates=NT, periods=list(PERIODS),
        synthetics=syn, rows=rows), indent=2))
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
