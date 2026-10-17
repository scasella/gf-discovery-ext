"""The months circle (k=12), and a direct false-positive-rate measurement.

The weekday sweep produced a warning sign: random 22-latent subsets recovered the
7-day ordering at p<=0.05 in 7 of 20 trials. With k=7 the dihedral group is 14 of
5040 permutations, so a *perfect* ordering already has chance probability 2.8e-3,
and near-perfect orderings are common. Any claim about which latents carry the
weekday circle is therefore weakly supported at best.

Months fix this by construction: k=12 puts the dihedral group at 24 of 479 million
permutations, a chance rate of 5e-8. This script re-runs the positive control on
months and measures the empirical false-positive rate of random matched latent
subsets for both ontologies, so the two are compared rather than assumed.

The background draw is uniform over firing latents by default, which is what the
0.295 / 0.44 rates were measured with. `--stratified` matches it to real clusters
on index band (and on firing frequency with `--freqs`) instead, and `--fpr-layer`
moves it off the hard-coded layer 21: the Amendment-1 background has to be
measured at the same layer and suite as the claim it calibrates.
"""
import argparse, json, os, sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from gfdisc.env import DEVICE
from gfdisc.model import load_model, capture
from gfdisc.saes import load_sae
from gfdisc.fits import (pca_2d, fit_circle, circularity, angles, cyclic_order_score,
                         degenerate)
from gfdisc.nulls import stratified_resample, frequency_matched_resample
from gfdisc.tokens import DAYS, MONTHS
from gfdisc.hf_range import read_header, read_tensor

DAY_T = ["The meeting is on {}", "She arrived last {}", "We open every {}",
         "It happened on a {}", "The report is due {}", "They travel each {}",
         "The class meets on {}", "I saw him last {}", "The deadline is {}",
         "Deliveries arrive every {}", "The market runs on {}", "He works from {}",
         "Our call is set for {}", "The event starts {}", "Practice is held each {}",
         "The store closes on {}", "The flight departs {}", "Rehearsal is every {}",
         "The shipment left on {}", "The concert is this {}"]
MON_T = ["The report was published in {}", "She was born in {}", "It rained all {}",
         "The season begins in {}", "We travelled there last {}", "The deadline was {}",
         "Sales peaked in {}", "The festival runs through {}", "He resigned in {}",
         "The contract expires in {}", "Enrolment opens in {}", "The survey ran during {}",
         "Prices fell in {}", "The court ruled in {}", "The harvest starts in {}",
         "The exhibition closed in {}", "Construction began in {}", "The law took effect in {}",
         "The team formed in {}", "Records were broken in {}"]


def gap_pvalue(k, max_gap_rad, n_perm=20000, seed=0):
    """Closure calibration against k uniform angles. Identical to the gate's
    (`scripts/13_rerun_gate.py`), duplicated rather than imported because that
    module is a script, not a package. Reported only: the headline background
    rate stays the ORDERING rate, which is what the previous 0.295 / 0.44
    measurements were and what the gate's pass rate is compared against."""
    rng = np.random.default_rng(seed)
    a = np.sort(rng.uniform(0, 2 * np.pi, size=(n_perm, k)), axis=1)
    g = np.diff(np.concatenate([a, a[:, :1] + 2 * np.pi], axis=1), axis=1).max(1)
    return float((1 + (g <= max_gap_rad).sum()) / (1 + n_perm))


def score(M, k, n_perm=20000, seed=0):
    if degenerate(M):
        return None
    xy, ratios, _ = pca_2d(M)
    cx, cy, r, resid = fit_circle(xy)
    sc = cyclic_order_score(angles(xy, cx, cy), np.arange(k), k, n_perm=n_perm, seed=seed)
    c = circularity(xy, cx, cy, r)
    return dict(rho=sc["rho"], p=sc["p"], resid=float(resid),
                top2=float(ratios[:2].sum()), null_p95=sc["null_p95"],
                gap_ratio=c["gap_ratio"], max_gap_deg=c["max_gap_deg"],
                p_closed=gap_pvalue(k, np.radians(c["max_gap_deg"]), seed=seed))


def draw_subset(rng, active, size, template=None, freqs=None, d_sae=None):
    """One background latent subset.

    Uniform over the firing latents by default, which is what the first FPR
    measurement used. That null is anti-conservative here: Gemma Scope 2 is
    Matryoshka-trained, so latent index predicts firing frequency
    (Spearman -0.48), and a uniform draw compares a real cluster against latents
    from a different frequency regime. With a template cluster the draw is
    matched instead -- on index band via `stratified_resample`, or on per-latent
    firing frequency via `frequency_matched_resample` when frequencies are
    supplied. Draws are intersected with `active`: a latent that never fires
    contributes an all-zero row and only shrinks the trial count.

    With a template the draw takes the template's SIZE as well as its composition,
    so `size` applies to the uniform draw only -- a background matched to a
    42-latent cluster by 22 latents is not matched.
    """
    if template is None:
        return rng.choice(active, min(size, len(active)), replace=False)
    if freqs is not None:
        pick = frequency_matched_resample(template, freqs, rng)
    else:
        pick = stratified_resample(template, d_sae, rng)
    alive = set(int(x) for x in np.asarray(active))
    return np.asarray([int(p) for p in pick if int(p) in alive], dtype=int)


def collect(model, tok, layer, sae, words, templates):
    raws, codes = [], []
    with torch.inference_mode():
        for w in words:
            enc = tok([t.format(w) for t in templates], return_tensors="pt",
                      padding=True).to(DEVICE)
            with capture(model, [layer]) as store:
                model(**enc, use_cache=False)
            last = enc["attention_mask"].sum(1) - 1
            h = store[layer][torch.arange(last.shape[0]), last].float()
            raws.append(h.mean(0).cpu().numpy())
            codes.append(sae.encode(h).mean(0).cpu().numpy())
    return np.vstack(raws), np.vstack(codes)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", default="17,19,20,21,22,23,24")
    ap.add_argument("--l0", default="small")
    ap.add_argument("--n-random", type=int, default=200)
    ap.add_argument("--fpr-layer", type=int, default=21)
    ap.add_argument("--stratified", action="store_true",
                    help="match background draws to real clusters on index band "
                         "(and on firing frequency with --freqs) instead of drawing uniformly")
    ap.add_argument("--ref-clusters", default="results/clusters.json",
                    help="cluster file supplying the composition templates for --stratified")
    ap.add_argument("--freqs", default=None,
                    help=".npy of per-latent firing counts at --fpr-layer; switches "
                         "--stratified to frequency matching")
    ap.add_argument("--out", default="results/months_power.json")
    a = ap.parse_args()
    token = os.environ["HF_TOKEN"]
    layers = [int(x) for x in a.layers.split(",")]
    model, tok = load_model(token=token)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"

    out = dict(months=[], fpr={})
    print(f"{'layer':>5} {'ontology':>9} {'representation':>14} {'top2':>7} {'resid':>7} "
          f"{'rho':>7} {'p':>10}")
    for layer in layers:
        sae = load_sae(layer, a.l0, device=DEVICE, token=token)
        for name, words, tmpl in (("months", MONTHS, MON_T),):
            R, Z = collect(model, tok, layer, sae, words, tmpl)
            Wd = sae.w_dec.cpu().numpy()
            recon = Z @ Wd + sae.b_dec.cpu().numpy()
            for rep, M in (("raw", R), ("recon", recon)):
                s = score(M, len(words))
                if s is None:
                    continue
                s.update(layer=layer, ontology=name, representation=rep)
                out["months"].append(s)
                print(f"{layer:5d} {name:>9} {rep:>14} {s['top2']:7.3f} {s['resid']:7.3f} "
                      f"{s['rho']:+7.3f} {s['p']:10.5f}")

    # empirical false-positive rate: random latent subsets, both ontologies
    mode = ("frequency-matched" if a.freqs else "index-stratified") if a.stratified else "uniform"
    print(f"\nfalse-positive rate of {mode} random latent subsets (layer {a.fpr_layer}):")
    sae = load_sae(a.fpr_layer, a.l0, device=DEVICE, token=token)
    Wd = sae.w_dec.cpu().numpy(); rng = np.random.default_rng(0)
    templates, freqs = None, None
    if a.stratified:
        cl = json.loads(Path(a.ref_clusters).read_text())[str(a.fpr_layer)]["clusters"]
        templates = [np.asarray(c["members"]) for c in cl if c["size"] >= 6]
        if not templates:
            raise SystemExit(f"--stratified: no clusters of size >= 6 at layer {a.fpr_layer} "
                             f"in {a.ref_clusters}")
        if a.freqs:
            freqs = np.load(a.freqs)
    for name, words, tmpl in (("days", DAYS, DAY_T), ("months", MONTHS, MON_T)):
        R, Z = collect(model, tok, a.fpr_layer, sae, words, tmpl)
        active = np.flatnonzero(Z.any(0))
        hits, n = 0, 0
        # Reported alongside the ordering rate, not in place of it. `sizes` records the
        # REALISED subset size: a stratified draw takes its size from the template but is
        # then intersected with the firing latents, so the k actually scored can fall below
        # the template's -- and the background is only matched if it does not.
        sizes, tmpl_sizes, closed, both = [], [], 0, 0
        for i in range(a.n_random):
            t = None if templates is None else templates[i % len(templates)]
            f = draw_subset(rng, active, 22, d_sae=sae.d_sae, freqs=freqs, template=t)
            if len(f) < 3:
                continue
            M = (Z[:, f] - Z[:, f].mean(0, keepdims=True)) @ Wd[f]
            s = score(M, len(words), n_perm=4000)
            if s is None:
                continue
            n += 1; hits += s["p"] <= 0.05
            closed += s["p_closed"] < 0.05
            both += (s["p"] <= 0.05) and (s["p_closed"] < 0.05)
            sizes.append(int(len(f)))
            tmpl_sizes.append(int(len(t)) if t is not None else 22)
        out["fpr"][name] = dict(k=len(words), n_trials=n, n_sig=int(hits),
                                fpr=hits / max(n, 1), n_active=int(len(active)),
                                layer=a.fpr_layer, l0=a.l0, draw=mode,
                                n_sig_closure=int(closed), fpr_closure=closed / max(n, 1),
                                n_sig_both=int(both), fpr_both=both / max(n, 1),
                                subset_size_mean=float(np.mean(sizes)) if sizes else 0.0,
                                subset_size_median=float(np.median(sizes)) if sizes else 0.0,
                                subset_size_min=int(min(sizes)) if sizes else 0,
                                subset_size_max=int(max(sizes)) if sizes else 0,
                                template_size_mean=float(np.mean(tmpl_sizes)) if tmpl_sizes else 0.0)
        print(f"  {name:7s} k={len(words):2d}: {hits}/{n} random subsets significant "
              f"-> FPR {hits/max(n,1):.3f}  (closure {closed}/{n}, both tiers {both}/{n}; "
              f"realised subset size mean {np.mean(sizes) if sizes else 0:.1f} of template "
              f"{np.mean(tmpl_sizes) if tmpl_sizes else 0:.1f})")
    Path(a.out).write_text(json.dumps(out, indent=2))
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
