"""Step 6 -- build the tier-1 sparse activation cache.

One streaming pass over FineWeb, capturing every target layer's residual stream
in a single forward, encoding through each layer's SAE in fp32, and writing CSR.
Everything the functional similarity matrices need (co-activation, correlation,
mutual information, Ising) comes out of this cache, as do the cluster-restricted
reconstructions that the PCA fits run on.
"""
import argparse, json, os, sys, time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from gfdisc.env import DEVICE, assert_disk, seed_all
from gfdisc.model import load_model, capture
from gfdisc.saes import load_sae
from gfdisc.cache import SparseWriter


def stream(tok, n_tokens, seq_len, bs):
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
            got += bs * (seq_len - 1)
            buf = []
            if got >= n_tokens:
                return


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", default="17,18,19,20,21,22,23,24,25")
    ap.add_argument("--l0", default="small")
    ap.add_argument("--n-tokens", type=int, default=18_000_000)
    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--root", default="cache/sparse")
    ap.add_argument("--max-gb", type=float, default=14.0)
    ap.add_argument("--meta-out", default="results/cache_meta.json")
    a = ap.parse_args()

    seed_all(0); assert_disk(5.0)
    token = os.environ["HF_TOKEN"]
    layers = [int(x) for x in a.layers.split(",")]

    model, tok = load_model(token=token)
    saes = {l: load_sae(l, a.l0, device=DEVICE, token=token) for l in layers}
    for l in layers:                      # per-layer l0 is NOT constant across the suite
        print(f"  layer {l}: nominal l0={saes[l].cfg['l0']}")
    writers = {l: SparseWriter(a.root, l, saes[l].d_sae,
                               meta={"suite": a.l0, "model": "google/gemma-3-1b-pt",
                                     "nominal_l0": saes[l].cfg["l0"]}) for l in layers}

    t0, done, nz_total = time.time(), 0, 0
    with torch.inference_mode():
        for ids in stream(tok, a.n_tokens, a.seq_len, a.bs):
            ids = ids.to(DEVICE)
            with capture(model, layers) as store:
                model(ids, use_cache=False)
            tok_ids = ids[:, 1:].reshape(-1).cpu().numpy()   # BOS dropped, matches codes
            for l in layers:
                x = store[l][:, 1:, :].float().reshape(-1, store[l].shape[-1])   # drop BOS
                z = saes[l].encode(x)
                writers[l].add_batch(z.cpu().numpy(), tok_ids)
            done += ids.shape[0] * (ids.shape[1] - 1)
            nz_total = sum(w.n_nz for w in writers.values())
            gb = (nz_total * 4 + sum(len(w.indptr) for w in writers.values()) * 8) / 1e9
            if done % (a.bs * (a.seq_len - 1) * 20) == 0:
                el = time.time() - t0
                print(f"  {done:,} tok | {gb:.2f} GB | {done/el:,.0f} tok/s | "
                      f"eta {(a.n_tokens-done)/max(done/el,1)/60:.0f} min", flush=True)
            if gb > a.max_gb:
                print(f"\nstopping: disk budget {a.max_gb} GB reached at {done:,} tokens")
                break

    meta = {l: writers[l].close() for l in layers}
    Path(a.meta_out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.meta_out).write_text(json.dumps(
        {str(k): v for k, v in meta.items()}, indent=2))
    for l in layers:
        m = meta[l]
        print(f"  layer {l}: {m['n_rows']:,} rows, mean L0 {m['n_nonzero']/m['n_rows']:.1f}, "
              f"{m['bytes_on_disk']/1e9:.2f} GB")


if __name__ == "__main__":
    main()
