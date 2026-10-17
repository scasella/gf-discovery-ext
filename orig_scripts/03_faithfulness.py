"""Step 3 -- SAE faithfulness gate. BLOCKING.

Everything downstream assumes the SAEs are being fed exactly the activations they
were trained on. Almost every way that can go wrong (wrong hook point, tuple vs
tensor, unreplicated input normalisation, dtype drift, wrong sparsity release)
shows up as a bad reconstruction, so we measure it before computing any geometry.

Two numbers per layer:
  FVU     -- fraction of residual-stream variance the SAE fails to reconstruct.
  dCE     -- cross-entropy increase when the SAE's reconstruction is spliced back
             into the residual stream, compared against the increase from zeroing
             that layer's residual entirely. The SAE must recover most of what
             ablation destroys; `recovered` reports that fraction.

BOS is excluded throughout: Gemma's BOS position carries an extreme activation
norm that would dominate both the variance denominator and later co-activation
statistics.
"""
import argparse, json, os, sys
from pathlib import Path

import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from gfdisc.env import DEVICE, assert_disk, seed_all
from gfdisc.model import load_model, capture, splice
from gfdisc.saes import load_sae


def batches(tok, n_tokens, seq_len=256, bs=8):
    from datasets import load_dataset
    ds = load_dataset("HuggingFaceFW/fineweb", name="sample-10BT", split="train", streaming=True)
    buf, got = [], 0
    for row in ds:
        ids = tok(row["text"], truncation=True, max_length=seq_len)["input_ids"]
        if len(ids) < seq_len:
            continue
        buf.append(ids[:seq_len])
        if len(buf) == bs:
            yield torch.tensor(buf)
            got += bs * seq_len
            buf = []
            if got >= n_tokens:
                return


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", default="17,19,21,22,24")
    ap.add_argument("--l0", default="small")
    ap.add_argument("--n-tokens", type=int, default=100_000)
    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--out", default="results/faithfulness.json")
    a = ap.parse_args()

    seed_all(0); assert_disk()
    token = os.environ["HF_TOKEN"]
    layers = [int(x) for x in a.layers.split(",")]

    model, tok = load_model(token=token)
    saes = {l: load_sae(l, a.l0, device=DEVICE, token=token) for l in layers}
    print(f"model + {len(saes)} SAEs on {DEVICE}", flush=True)

    # running sums for FVU, plus CE under three conditions
    ss_err = {l: 0.0 for l in layers}
    ss_tot = {l: 0.0 for l in layers}
    n_pos = 0
    ce_clean, ce_recon, ce_zero = 0.0, {l: 0.0 for l in layers}, {l: 0.0 for l in layers}
    n_batches = 0
    l0_obs = {l: 0.0 for l in layers}

    def ce_of(ids, logits):
        lp = torch.log_softmax(logits[:, :-1].float(), -1)
        tgt = ids[:, 1:]
        return -lp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)[:, 1:].mean().item()

    with torch.inference_mode():
        for ids in batches(tok, a.n_tokens, a.seq_len, a.bs):
            ids = ids.to(DEVICE)
            with capture(model, layers) as store:
                out = model(ids, use_cache=False)
            ce_clean += ce_of(ids, out.logits); n_batches += 1

            for l in layers:
                x = store[l][:, 1:, :].float()           # drop BOS
                z = saes[l].encode(x)
                xh = saes[l].decode(z)
                ss_err[l] += ((x - xh) ** 2).sum().item()
                ss_tot[l] += ((x - x.mean(dim=(0, 1), keepdim=True)) ** 2).sum().item()
                l0_obs[l] += (z > 0).float().sum(-1).mean().item()

                with splice(model, l, lambda t: torch.cat(
                        [t[:, :1, :], saes[l](t[:, 1:, :].float())], dim=1)):
                    ce_recon[l] += ce_of(ids, model(ids, use_cache=False).logits)
                with splice(model, l, lambda t: torch.cat(
                        [t[:, :1, :], torch.zeros_like(t[:, 1:, :])], dim=1)):
                    ce_zero[l] += ce_of(ids, model(ids, use_cache=False).logits)
            n_pos += ids.shape[0] * (ids.shape[1] - 1)
            print(f"  {n_pos:,} positions", end="\r", flush=True)

    rows = []
    base = ce_clean / n_batches
    print(f"\n\nclean CE = {base:.4f} over {n_pos:,} positions (BOS excluded)\n")
    print(f"{'layer':>5} {'FVU':>8} {'obsL0':>7} {'dCE_sae':>9} {'dCE_zero':>9} {'recovered':>10}")
    for l in layers:
        fvu = ss_err[l] / ss_tot[l]
        d_sae = ce_recon[l] / n_batches - base
        d_zero = ce_zero[l] / n_batches - base
        rec = 1 - d_sae / d_zero if d_zero > 0 else float("nan")
        rows.append(dict(layer=l, fvu=fvu, obs_l0=l0_obs[l] / n_batches,
                         dce_sae=d_sae, dce_zero=d_zero, recovered=rec))
        print(f"{l:5d} {fvu:8.4f} {l0_obs[l]/n_batches:7.1f} {d_sae:9.4f} {d_zero:9.4f} {rec:10.3f}")

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(dict(clean_ce=base, n_positions=n_pos, rows=rows), indent=2))
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
