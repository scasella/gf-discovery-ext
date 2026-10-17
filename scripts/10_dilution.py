"""How many SAE latents does the weekday circle take?

The positive control established three things: the circle is in the residual
stream at every layer tested, it survives full SAE reconstruction, and it is NOT
recoverable from the latents that look like weekday features. That is Bhalla et
al.'s "dilution" -- but "diluted" is only a label until you say across how many
atoms.

So: greedily add latents, one at a time, choosing at each step the latent that
most reduces the residual against the raw-residual-stream day means, and record
when the cyclic ordering actually returns. Compared against two baselines -- the
supervised weekday latents, and random latent sets of matched size -- this turns
a failed gate into a measurement.
"""
import argparse, json, os, sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from gfdisc.env import DEVICE
from gfdisc.model import load_model, capture
from gfdisc.saes import load_sae
from gfdisc.fits import pca_2d, fit_circle, angles, cyclic_order_score, degenerate
from gfdisc.tokens import DAYS
from gfdisc.hf_range import read_header, read_tensor

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
TEMPLATES = __import__("importlib").import_module("09_positive_control").TEMPLATES


def order_score(M, n_perm=5000, seed=0):
    if degenerate(M):
        return None
    xy, ratios, _ = pca_2d(M)
    cx, cy, r, resid = fit_circle(xy)
    sc = cyclic_order_score(angles(xy, cx, cy), np.arange(len(M)), len(M),
                            n_perm=n_perm, seed=seed)
    return dict(rho=sc["rho"], p=sc["p"], resid=float(resid),
                top2=float(ratios[:2].sum()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=21)
    ap.add_argument("--l0", default="small")
    ap.add_argument("--max-k", type=int, default=80)
    ap.add_argument("--out", default="results/dilution.json")
    a = ap.parse_args()
    token = os.environ["HF_TOKEN"]

    model, tok = load_model(token=token)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    sae = load_sae(a.layer, a.l0, device=DEVICE, token=token)

    # per-day mean codes and raw residual means
    codes, raws = [], []
    with torch.inference_mode():
        for day in DAYS:
            enc = tok([t.format(day) for t in TEMPLATES], return_tensors="pt",
                      padding=True).to(DEVICE)
            with capture(model, [a.layer]) as store:
                model(**enc, use_cache=False)
            last = enc["attention_mask"].sum(1) - 1
            h = store[a.layer][torch.arange(last.shape[0]), last].float()
            raws.append(h.mean(0).cpu().numpy())
            codes.append(sae.encode(h).mean(0).cpu().numpy())
    Z = np.vstack(codes)                    # [7, d_sae] mean code per day
    R = np.vstack(raws)                     # [7, d_model] raw target
    Rc = R - R.mean(0, keepdims=True)
    Wd = sae.w_dec.cpu().numpy()

    base = order_score(R)
    print(f"layer {a.layer} raw residual stream: rho {base['rho']:+.3f} p {base['p']:.4f} "
          f"resid {base['resid']:.3f}\n")

    active = np.flatnonzero(Z.any(0))
    print(f"{len(active)} latents fire on any weekday prompt; greedy selection over those\n")

    chosen, cur, rows = [], np.zeros_like(Rc), []
    pool = list(active)
    for k in range(1, a.max_k + 1):
        best, bi = None, None
        for f in pool:
            cand = cur + np.outer(Z[:, f] - Z[:, f].mean(), Wd[f])
            err = ((Rc - cand) ** 2).sum()
            if best is None or err < best:
                best, bi = err, f
        chosen.append(int(bi)); pool.remove(bi)
        cur = cur + np.outer(Z[:, bi] - Z[:, bi].mean(), Wd[bi])
        ev = 1 - best / (Rc ** 2).sum()
        s = order_score(cur + R.mean(0, keepdims=True))
        rows.append(dict(k=k, feature=int(bi), explained=float(ev),
                         rho=s["rho"] if s else None, p=s["p"] if s else None,
                         resid=s["resid"] if s else None))
        if k <= 12 or k % 10 == 0:
            print(f"  k={k:3d}  explained {ev:6.3f}  rho {rows[-1]['rho']:+.3f}  "
                  f"p {rows[-1]['p']:.4f}")

    # baselines at matched sizes
    u = (f"https://huggingface.co/google/gemma-scope-2-1b-pt/resolve/main/"
         f"resid_post_all/layer_{a.layer}_width_16k_l0_{a.l0}/examples.safetensors")
    h, off = read_header(u); tt = read_tensor(u, "top_tokens", h, off)
    dayset = {d.lower() for d in DAYS}
    sup = [f for f in range(tt.shape[0])
           if len({tok.convert_ids_to_tokens(int(i)).replace("▁", " ").strip().lower().rstrip("s")
                   for i in tt[f]} & dayset) >= 2]
    def subset_score(feats):
        c = np.zeros_like(Rc)
        for f in feats:
            c = c + np.outer(Z[:, f] - Z[:, f].mean(), Wd[f])
        return order_score(c + R.mean(0, keepdims=True))
    ssup = subset_score(sup)
    rng = np.random.default_rng(0)
    rnd = [subset_score(rng.choice(active, min(len(sup), len(active)), replace=False))
           for _ in range(20)]
    rnd = [r for r in rnd if r]
    print(f"\n  supervised weekday latents (n={len(sup)}): rho {ssup['rho']:+.3f} p {ssup['p']:.4f}")
    print(f"  random matched sets (n={len(sup)}): median rho {np.median([r['rho'] for r in rnd]):+.3f}, "
          f"{sum(1 for r in rnd if r['p']<=0.05)}/{len(rnd)} significant")
    first = next((r for r in rows if r["p"] is not None and r["p"] <= 0.05), None)
    print(f"\n  greedy first reaches p<=0.05 at k = {first['k'] if first else 'never'}")
    Path(a.out).write_text(json.dumps(dict(layer=a.layer, raw=base, greedy=rows,
        supervised=dict(n=len(sup), **ssup),
        random_median_rho=float(np.median([r["rho"] for r in rnd]))), indent=2))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
