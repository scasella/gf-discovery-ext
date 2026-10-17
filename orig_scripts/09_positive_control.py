"""Positive control -- does the weekday circle exist at all in this model?

The unsupervised gate failed. That has three very different explanations and they
must be separated before any of them is reported:

  A. the structure is absent in gemma-3-1b (a 1B base model may simply not have it)
  B. the structure exists in the residual stream but the SAE shatters it
     (Bhalla et al.'s "dilution"), so no group of latents recovers it
  C. the structure survives in SAE space but our clustering failed to group it

So we bypass the unsupervised step entirely and go supervised, following the
reference implementation's approach: push prompts through the model once and take
the hidden state at the day token. Then ask the same geometric question of three
representations -- the raw residual stream (tests A), the full SAE reconstruction
(tests B), and the reconstruction restricted to known weekday latents (tests C).
"""
import argparse, json, os, sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from gfdisc.env import DEVICE
from gfdisc.model import load_model, capture
from gfdisc.saes import load_sae, get_decoder
from gfdisc.fits import pca_2d, fit_circle, angles, cyclic_order_score, degenerate
from gfdisc.tokens import DAYS

TEMPLATES = [
    "The meeting is on {}", "She arrived last {}", "We open every {}",
    "It happened on a {}", "The report is due {}", "They travel each {}",
    "The class meets on {}", "I saw him last {}", "The deadline is {}",
    "Deliveries arrive every {}", "The market runs on {}", "He works from {}",
    "Our call is set for {}", "The event starts {}", "Practice is held each {}",
    "The store closes on {}", "The flight departs {}", "Rehearsal is every {}",
    "The shipment left on {}", "The concert is this {}",
]


def day_means(model, tok, layers, saes, W_by_layer, day_feats):
    """Mean hidden state at the day token, per day, for each representation."""
    out = {l: dict(raw=[], recon=[], restricted=[]) for l in layers}
    with torch.inference_mode():
        for day in DAYS:
            texts = [t.format(day) for t in TEMPLATES]
            enc = tok(texts, return_tensors="pt", padding=True).to(DEVICE)
            with capture(model, layers) as store:
                model(**enc, use_cache=False)
            # the day token is the last non-pad token of each prompt
            last = enc["attention_mask"].sum(1) - 1
            for l in layers:
                h = store[l][torch.arange(len(texts)), last].float()
                out[l]["raw"].append(h.mean(0).cpu().numpy())
                z = saes[l].encode(h)
                out[l]["recon"].append(saes[l].decode(z).mean(0).cpu().numpy())
                f = day_feats[l]
                if len(f):
                    zr = torch.zeros_like(z); zr[:, f] = z[:, f]
                    out[l]["restricted"].append(saes[l].decode(zr).mean(0).cpu().numpy())
    return out


def score(M, n_perm=5000):
    M = np.vstack(M)
    if degenerate(M):
        return None
    xy, ratios, _ = pca_2d(M)
    cx, cy, r, resid = fit_circle(xy)
    sc = cyclic_order_score(angles(xy, cx, cy), np.arange(len(M)), len(M), n_perm=n_perm)
    rng = np.random.default_rng(1)
    sh = cyclic_order_score(angles(xy, cx, cy), rng.permutation(len(M)), len(M), n_perm=n_perm)
    return dict(top2_var=float(ratios[:2].sum()), circle_resid=float(resid),
                rho=sc["rho"], p=sc["p"], null_p95=sc["null_p95"],
                shuffled_rho=sh["rho"], points=xy.tolist())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", default="17,19,20,21,22,23,24")
    ap.add_argument("--l0", default="small")
    ap.add_argument("--out", default="results/positive_control.json")
    a = ap.parse_args()
    token = os.environ["HF_TOKEN"]
    layers = [int(x) for x in a.layers.split(",")]

    model, tok = load_model(token=token)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    saes = {l: load_sae(l, a.l0, device=DEVICE, token=token) for l in layers}
    W = {l: get_decoder(saes[l]) for l in layers}

    # supervised weekday latents, from the precomputed top_tokens shipped with each SAE
    from gfdisc.hf_range import read_header, read_tensor
    BASE = "https://huggingface.co/google/gemma-scope-2-1b-pt/resolve/main/resid_post_all"
    day_feats = {}
    for l in layers:
        u = f"{BASE}/layer_{l}_width_16k_l0_{a.l0}/examples.safetensors"
        h, off = read_header(u); tt = read_tensor(u, "top_tokens", h, off)
        hits = []
        for f in range(tt.shape[0]):
            s = {tok.convert_ids_to_tokens(int(i)).replace("▁", " ").strip().lower().rstrip("s")
                 for i in tt[f]}
            if len(s & {d.lower() for d in DAYS}) >= 2:
                hits.append(f)
        day_feats[l] = hits
        print(f"  layer {l}: {len(hits)} supervised weekday latents", flush=True)

    res = day_means(model, tok, layers, saes, W, day_feats)
    rows = []
    print(f"\n{'layer':>5} {'representation':>14} {'nfeat':>6} {'top2':>7} {'resid':>7} "
          f"{'rho':>7} {'p':>8} {'shufRho':>8}")
    for l in layers:
        for rep in ("raw", "recon", "restricted"):
            if not res[l][rep]:
                continue
            s = score(res[l][rep])
            if s is None:
                continue
            s.update(layer=l, representation=rep, n_features=len(day_feats[l]))
            rows.append(s)
            print(f"{l:5d} {rep:>14} {len(day_feats[l]):6d} {s['top2_var']:7.3f} "
                  f"{s['circle_resid']:7.3f} {s['rho']:+7.3f} {s['p']:8.4f} {s['shuffled_rho']:+8.3f}")
    Path(a.out).write_text(json.dumps(rows, indent=2))
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
