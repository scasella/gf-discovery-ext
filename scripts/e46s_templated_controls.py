"""e46s -- templated supervised controls for the Stage-2 ruler/lattice tiers (spec section 5.1).

Each word w is read at its LAST token inside 16 neutral templates (8 -> half A, 8 -> half B), at
every declared layer, as the raw residual (resid_post of layer L, raw HF hook = the SAE hook
point). Label sets: military ranks (8, ordinal), number words one..ten (10, ordinal), kinship (13,
lattice generation x lineality/gender). Null: 300 random pool words (single-token, leading space,
alphabetic, count >= 100) through the identical templates; each null k-set is scored by the
identical statistics (its own PC1 axis / top-2 plane). Words that tokenize to several tokens are
read at the last token (declared; Phase-2 convention).
"""
import argparse, json, os, sys, time
from pathlib import Path

import numpy as np
import torch

EXT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXT / "src"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
import gfdisc.model as gm                                          # noqa: E402
_SNAP = sorted((EXT / "hf_home/hub/models--google--gemma-3-1b-pt/snapshots").glob("*"))[0]
gm.MODEL_NAME = str(_SNAP)
from gfdisc.env import DEVICE                                      # noqa: E402
from gfdisc.model import load_model, capture                       # noqa: E402

TEMPLATES = [
    "The next word is{w}", "Here is a word:{w}", "He wrote down the word{w}",
    "She said the word{w}", "One more example:{w}", "The answer was{w}",
    "Write this term:{w}", "Consider the word{w}",
    "The list ends with{w}", "I read the word{w}", "Today's word is{w}",
    "Look up the word{w}", "They typed{w}", "The label says{w}",
    "Remember this word:{w}", "A common word is{w}",
]
SETS = dict(
    ranks=[" private", " corporal", " sergeant", " lieutenant", " captain", " major",
           " colonel", " general"],
    numbers=[" one", " two", " three", " four", " five", " six", " seven", " eight", " nine",
             " ten"],
    kinship=[" grandfather", " grandmother", " father", " mother", " son", " daughter",
             " brother", " sister", " uncle", " aunt", " nephew", " niece", " cousin"],
)
KIN_GEN = [2, 2, 1, 1, -1, -1, 0, 0, 1, 1, -1, -1, 0]
KIN_GENDER = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, -1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", default="5,6,7,8,9,12,14,16,18,20,22,24")
    ap.add_argument("--out", default=str(EXT / "results/e46s_templated_means.npz"))
    ap.add_argument("--n-null", type=int, default=300)
    a = ap.parse_args()
    layers = [int(x) for x in a.layers.split(",")]
    tm = np.load(EXT / "results/typemeans_pass1.npz")
    pool_ids = tm["pool_ids"]
    model, tok = load_model(token=None)
    rng = np.random.default_rng(4646)
    cand = []
    for t in pool_ids:
        s = tok.convert_ids_to_tokens(int(t)) or ""
        if s.startswith("▁") and s[1:].isalpha() and s[1:].islower() and len(s) > 3:
            cand.append(" " + s[1:])
    labelled = {w for v in SETS.values() for w in v}
    cand = [w for w in cand if w not in labelled]
    null_words = list(rng.choice(cand, size=a.n_null, replace=False))
    words = sorted(labelled) + null_words
    reps = {L: np.zeros((len(words), 2, 1152), dtype=np.float64) for L in layers}
    t0 = time.time()
    with torch.inference_mode():
        for wi, w in enumerate(words):
            prompts = [t.format(w=w) for t in TEMPLATES]
            enc = [tok(p)["input_ids"] for p in prompts]
            lens = [len(e) for e in enc]
            mx = max(lens)
            ids = torch.full((len(enc), mx), tok.pad_token_id, dtype=torch.long)
            att = torch.zeros_like(ids)
            for i, e in enumerate(enc):
                ids[i, :len(e)] = torch.tensor(e); att[i, :len(e)] = 1
            with capture(model, layers) as st:
                model(input_ids=ids.to(DEVICE), attention_mask=att.to(DEVICE), use_cache=False)
            last = torch.tensor([l - 1 for l in lens])
            for L in layers:
                h = st[L][torch.arange(len(enc)), last.to(DEVICE)].float().cpu().numpy()
                assert np.isfinite(h).all()
                reps[L][wi, 0] = h[:8].mean(0)
                reps[L][wi, 1] = h[8:].mean(0)
            if wi % 50 == 0:
                print(f"  {wi}/{len(words)} {time.time()-t0:.0f}s", flush=True)
    np.savez(a.out, words=np.asarray(words), layers=np.asarray(layers),
             n_labelled=len(labelled), **{f"L{L}": reps[L].astype(np.float32) for L in layers})
    print(f"wrote {a.out}", flush=True)


if __name__ == "__main__":
    main()
