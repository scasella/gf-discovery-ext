"""e60 -- Stage 4 extraction: per-occurrence SAE-decoded residuals for a block of sequences.

For each sequence (BOS + 255 tokens) the model is run once; at each requested layer the
residual is encoded by that layer's l0_big 16k SAE (fp32, as 06) and decoded as
fp16(code) @ W_dec (no b_dec -- the scripts/22 representation; b_dec cancels in the
identity removal anyway). Positions 1..255 are stored as float16 together with token id,
sequence index and position.

Sources
  corpus   results/corpus_tokens_pass1.i32  (15696 x 255, BOS dropped)
  reserve  results/reserve_tokens.i32       (8000 x 256, BOS included -> dropped here)
--shuffle permutes the 255 tokens of every sequence with a per-sequence seeded RNG
(context-shuffled null: same tokens, same counts, natural context destroyed).
"""
import argparse, json, os, sys, time
from pathlib import Path

import numpy as np
import torch

EXT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXT / "src"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
import gfdisc.model as gm                                          # noqa: E402
gm.MODEL_NAME = str(sorted((EXT / "hf_home/hub/models--google--gemma-3-1b-pt/snapshots").glob("*"))[0])
from gfdisc.env import DEVICE                                      # noqa: E402
from gfdisc.model import load_model, capture                       # noqa: E402
from gfdisc.saes import load_sae                                   # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["corpus", "reserve"], required=True)
    ap.add_argument("--seq-start", type=int, required=True)
    ap.add_argument("--n-seq", type=int, required=True)
    ap.add_argument("--shuffle", action="store_true")
    ap.add_argument("--layers", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--bs", type=int, default=16)
    a = ap.parse_args()
    layers = [int(x) for x in a.layers.split(",")]
    if a.source == "corpus":
        T = np.fromfile(EXT / "results/corpus_tokens_pass1.i32", dtype=np.int32).reshape(-1, 255)
    else:
        T = np.fromfile(EXT / "results/reserve_tokens.i32", dtype=np.int32).reshape(-1, 256)[:, 1:]
    T = T[a.seq_start:a.seq_start + a.n_seq].copy()
    assert len(T) == a.n_seq
    if a.shuffle:
        for i in range(len(T)):
            T[i] = T[i][np.random.default_rng(600_000 + a.seq_start + i).permutation(255)]
    model, tok = load_model(token=None)
    saes = {L: load_sae(L, "big", device=DEVICE, token=None) for L in layers}
    n_rows = a.n_seq * 255
    outdir = EXT / "cache/occ"
    outdir.mkdir(parents=True, exist_ok=True)
    mm = {L: np.lib.format.open_memmap(outdir / f"{a.tag}_L{L}.npy", mode="w+", dtype=np.float16,
                                       shape=(n_rows, 1152)) for L in layers}
    t0 = time.time()
    with torch.inference_mode():
        for lo in range(0, a.n_seq, a.bs):
            blk = T[lo:lo + a.bs].astype(np.int64)
            ids = np.concatenate([np.full((len(blk), 1), tok.bos_token_id), blk], axis=1)
            with capture(model, layers) as st:
                model(torch.from_numpy(ids).to(DEVICE), use_cache=False)
            for L in layers:
                x = st[L][:, 1:, :].float().reshape(-1, 1152)
                assert bool(torch.isfinite(x).all()), f"non-finite residual L{L} seq {lo}"
                z = saes[L].encode(x)
                dec = (z.half().float() @ saes[L].w_dec).cpu().numpy()
                mm[L][lo * 255:(lo + len(blk)) * 255] = dec.astype(np.float16)
            del st
            if (lo // a.bs) % 20 == 0:
                print(f"  {lo + len(blk)}/{a.n_seq} seq  {time.time()-t0:.0f}s", flush=True)
    for L in layers:
        mm[L].flush()
    np.savez(outdir / f"{a.tag}_meta.npz", tokens=T.reshape(-1).astype(np.int32),
             seq=np.repeat(np.arange(a.seq_start, a.seq_start + a.n_seq), 255).astype(np.int32),
             pos=np.tile(np.arange(1, 256), a.n_seq).astype(np.int16),
             shuffled=a.shuffle, source=a.source, layers=np.asarray(layers))
    (outdir / f"{a.tag}_info.json").write_text(json.dumps(dict(vars(a), seconds=round(time.time() - t0))))
    print(f"done {a.tag} in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
