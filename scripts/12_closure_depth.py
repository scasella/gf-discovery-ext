"""Where in the network is the cycle still CLOSED?

Ordering and closure dissociate: a monotone 1-D arrangement scores rho=1.000 on the
cyclic-ordering statistic, so "correct order at every layer" says nothing about
whether the configuration is a ring. The closure test (radius ratio, angular gap
ratio) does distinguish them, and at layer 21 both calendar ontologies fail it.

That determines where the rest of Phase 1 can possibly work: no cluster of SAE
latents can recover a closed ring from a residual stream that no longer contains
one. So sweep every layer, with number words one..twelve as an ordinal control
that must never pass, and cache/cluster only where closure survives.
"""
import argparse, json, os, sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from gfdisc.env import DEVICE
from gfdisc.model import load_model, capture
from gfdisc.fits import (pca_2d, fit_circle, circularity, is_ring, angles,
                         cyclic_order_score)
from gfdisc.tokens import DAYS, MONTHS

NUMS = ["one", "two", "three", "four", "five", "six",
        "seven", "eight", "nine", "ten", "eleven", "twelve"]
DT = ["The meeting is on {}", "She arrived last {}", "We open every {}", "It happened on a {}",
      "The report is due {}", "They travel each {}", "The class meets on {}", "I saw him last {}",
      "The deadline is {}", "Deliveries arrive every {}", "The market runs on {}", "He works from {}",
      "Our call is set for {}", "The event starts {}", "Practice is held each {}", "The store closes on {}"]
MT = ["The report was published in {}", "She was born in {}", "It rained all {}", "The season begins in {}",
      "We travelled there last {}", "The deadline was {}", "Sales peaked in {}", "The festival runs through {}",
      "He resigned in {}", "The contract expires in {}", "Enrolment opens in {}", "The survey ran during {}",
      "Prices fell in {}", "The court ruled in {}", "The harvest starts in {}", "Construction began in {}"]
NT = ["He counted to {}", "There were {} of them", "The total came to {}", "She bought {} apples",
      "It weighs {} kilograms", "The score was {}", "We waited {} hours", "They hired {} people",
      "The box holds {}", "He ran {} miles", "The list had {} entries", "It costs {} pounds",
      "The recipe needs {} eggs", "We ordered {} chairs", "The class has {} students", "He owns {} cars"]


def gap_pvalue(k, observed_max_gap, n_perm=20000, seed=0):
    """Calibrate the max-gap against uniform angles rather than a hand-set threshold."""
    rng = np.random.default_rng(seed)
    a = np.sort(rng.uniform(0, 2 * np.pi, size=(n_perm, k)), axis=1)
    g = np.diff(np.concatenate([a, a[:, :1] + 2 * np.pi], axis=1), axis=1).max(1)
    return float((1 + (g <= observed_max_gap).sum()) / (1 + n_perm))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/closure_depth.json")
    a = ap.parse_args()
    token = os.environ["HF_TOKEN"]
    model, tok = load_model(token=token)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    layers = list(range(26))

    rows = []
    for name, words, T in (("weekdays", DAYS, DT), ("months", MONTHS, MT), ("numbers", NUMS, NT)):
        means = {l: [] for l in layers}
        with torch.inference_mode():
            for w in words:
                enc = tok([t.format(w) for t in T], return_tensors="pt", padding=True).to(DEVICE)
                with capture(model, layers) as st:
                    model(**enc, use_cache=False)
                last = enc["attention_mask"].sum(1) - 1
                for l in layers:
                    means[l].append(st[l][torch.arange(last.shape[0]), last].float().mean(0).cpu().numpy())
        k = len(words)
        for l in layers:
            M = np.vstack(means[l])
            xy, ratios, _ = pca_2d(M)
            cx, cy, r, res = fit_circle(xy)
            c = circularity(xy, cx, cy, r)
            # exact null for k=7 (5040 permutations); Monte Carlo for k=12
            s = cyclic_order_score(angles(xy, cx, cy), np.arange(k), k,
                                   n_perm=20000, seed=l)
            rows.append(dict(ontology=name, layer=l, k=k, rho=s["rho"], p_order=s["p"],
                             radius_ratio=c["radius_ratio"], gap_ratio=c["gap_ratio"],
                             max_gap_deg=c["max_gap_deg"],
                             p_closed=gap_pvalue(k, np.radians(c["max_gap_deg"])),
                             top2=float(ratios[:2].sum()), ring=is_ring(c)))
        print(f"  {name} done", flush=True)

    Path(a.out).write_text(json.dumps(rows, indent=2))
    print(f"\n{'layer':>5}", end="")
    for n in ("weekdays", "months", "numbers"):
        print(f" | {n:>26}", end="")
    print(f"\n{'':>5}", end="")
    for _ in range(3):
        print(f" | {'rho':>6}{'gapR':>7}{'ring':>6}{'pClo':>7}", end="")
    print()
    for l in layers:
        print(f"{l:5d}", end="")
        for n in ("weekdays", "months", "numbers"):
            r = next(x for x in rows if x["ontology"] == n and x["layer"] == l)
            print(f" | {r['rho']:+6.2f}{r['gap_ratio']:7.2f}{'  YES' if r['ring'] else '   no':>6}"
                  f"{r['p_closed']:7.3f}", end="")
        print()
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
