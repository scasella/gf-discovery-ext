"""Step 2 — offline pre-check. Zero model inference, ~40 MB of HTTP range reads.

Goodfire's paper (arXiv 2604.28119) finds that decoder cosine similarity *fails*
to recover manifold block structure, because manifold membership is a functional
property (which atoms co-activate) rather than a geometric one (where atoms
point). That removes decoder clustering as a cheap way to pick which layers are
worth caching activations for.

The per-feature `top_tokens` shipped in every Gemma Scope 2 `examples.safetensors`
give us a functional signal instead, and they cost ~1 MB per SAE to read. This
script asks, for each layer and sparsity suite: how many of the seven weekdays
(and twelve months) have a feature that fires distinctly for them? Layers where
the days are individuated are the layers where a weekday circle could exist.
"""
import argparse
import json
import re
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from gfdisc.hf_range import read_header, read_tensor  # noqa: E402

REPO = "google/gemma-scope-2-1b-pt"
BASE = f"https://huggingface.co/{REPO}/resolve/main"

DAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
MONTHS = ["january", "february", "march", "april", "may", "june",
          "july", "august", "september", "october", "november", "december"]


def url_for(layer, l0, site="resid_post_all", width="16k"):
    return f"{BASE}/{site}/layer_{layer}_width_{width}_l0_{l0}/examples.safetensors"


def label_features(top_tokens, vocab_strings, targets, min_hits=3):
    """Assign each feature the target whose name dominates its top tokens.

    A feature counts for a target only if >= min_hits of its 10 top tokens match
    that target and no other target matches more often, which keeps generic
    "calendar word" features from being credited to every day at once.
    """
    assignment = {}
    for f in range(top_tokens.shape[0]):
        toks = [vocab_strings[t] for t in top_tokens[f] if 0 <= t < len(vocab_strings)]
        blob = [t.strip().lower() for t in toks]
        counts = Counter()
        for tgt in targets:
            n = sum(1 for b in blob if b and (b == tgt or b == tgt[:3] or tgt.startswith(b) and len(b) >= 3))
            if n:
                counts[tgt] = n
        if not counts:
            continue
        tgt, n = counts.most_common(1)[0]
        if n >= min_hits and list(counts.values()).count(n) == 1:
            assignment[f] = tgt
    return assignment


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", default="0-25")
    ap.add_argument("--suites", default="small,big")
    ap.add_argument("--site", default="resid_post_all")
    ap.add_argument("--width", default="16k")
    ap.add_argument("--out", default="results/precheck.json")
    args = ap.parse_args()

    a, _, b = args.layers.partition("-")
    layers = list(range(int(a), int(b) + 1)) if b else [int(a)]
    suites = args.suites.split(",")

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("google/gemma-3-1b-pt")
    vocab_strings = [tok.convert_ids_to_tokens(i) or "" for i in range(tok.vocab_size)]
    vocab_strings = [v.replace("▁", " ") for v in vocab_strings]
    print(f"tokenizer: {len(vocab_strings)} tokens", flush=True)

    def probe(job):
        layer, suite = job
        u = url_for(layer, suite, args.site, args.width)
        try:
            hdr, off = read_header(u)
            tt = read_tensor(u, "top_tokens", hdr, off)
            ff = read_tensor(u, "feature_frequencies", hdr, off)
        except Exception as e:
            return dict(layer=layer, suite=suite, error=str(e)[:120])
        day_map = label_features(tt, vocab_strings, DAYS)
        mon_map = label_features(tt, vocab_strings, MONTHS)
        return dict(
            layer=layer, suite=suite,
            dead=int((ff < 1e-6).sum()), alive=int((ff >= 1e-6).sum()),
            day_features=len(day_map), days_covered=len(set(day_map.values())),
            day_breakdown={d: sum(1 for v in day_map.values() if v == d) for d in DAYS},
            month_features=len(mon_map), months_covered=len(set(mon_map.values())),
        )

    jobs = [(l, s) for s in suites for l in layers]
    with ThreadPoolExecutor(max_workers=8) as ex:
        rows = list(ex.map(probe, jobs))

    rows.sort(key=lambda r: (r["suite"], r["layer"]))
    print(f"\n{'suite':7s} {'layer':>5s} {'dead':>5s} {'dayF':>5s} {'days/7':>7s} {'monF':>5s} {'mons/12':>8s}")
    for r in rows:
        if "error" in r:
            print(f"{r['suite']:7s} {r['layer']:5d}  ERROR {r['error']}")
            continue
        print(f"{r['suite']:7s} {r['layer']:5d} {r['dead']:5d} {r['day_features']:5d} "
              f"{r['days_covered']:7d} {r['month_features']:5d} {r['months_covered']:8d}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(rows, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
