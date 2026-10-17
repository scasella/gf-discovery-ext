"""Pool feasibility ONLY: occurrence counts of Phase-2 ontology token variants in
cache/sparse_big/layer_0/tokens.i32. Counts only -- no SAE values, no geometry.
"""
import json, sys
from pathlib import Path
import numpy as np
from transformers import AutoTokenizer

ROOT = Path("/Users/scasella/Downloads/gf-discovery")
SCR = Path("/private/tmp/claude-501/-Users-scasella-Downloads-gf-discovery/34e91e23-ad3d-4976-8248-8ac0fe731b42/scratchpad")
sys.path.insert(0, str(ROOT / "src"))
from gfdisc.ontologies import (CIRCLE_OF_FIFTHS, CHROMATIC, ZODIAC, DIRECTIONS,
                               MILITARY_RANKS, KINSHIP)

SEASONS = ["spring", "summer", "autumn", "winter", "fall"]

tok = AutoTokenizer.from_pretrained("google/gemma-3-1b-pt")
toks = np.fromfile(ROOT / "cache/sparse_big/layer_0/tokens.i32", dtype=np.int32)
counts = np.bincount(toks, minlength=int(toks.max()) + 1)
print("n_cache_rows", toks.size, "n_distinct", int((counts > 0).sum()))

def variants(w, notes=False):
    if notes:                      # musical keys: uppercase note letter only
        base = [w]
        expl = [w.lower()]         # lowercase reported as the polysemy witness
    else:
        base = sorted({w, w.lower(), w.capitalize()})
        expl = []
    reg, exp = [], []
    for b in base:
        reg += [b, " " + b]
    for b in expl:
        exp += [b, " " + b]
    return reg, exp

def score(v):
    ids = tok.encode(v, add_special_tokens=False)
    n = len(ids)
    c = int(counts[ids[0]]) if n == 1 and ids[0] < counts.size else None
    return dict(form=v, n_tokens=n, ids=ids, count=c)

def row(w, notes=False):
    reg, exp = variants(w, notes)
    R = [score(v) for v in reg]
    E = [score(v) for v in exp]
    single = [v for v in R if v["n_tokens"] == 1]
    best = max(single, key=lambda v: v["count"]) if single else None
    return dict(word=w, registered=R, exploratory_lowercase=E, best=best,
                best_count=(best["count"] if best else 0),
                any_single_token=bool(single))

NOTES = list(dict.fromkeys(CIRCLE_OF_FIFTHS + CHROMATIC))
SETS = {"fifths_chromatic_notes": (NOTES, True), "zodiac": (ZODIAC, False),
        "directions": (DIRECTIONS, False), "ranks": (MILITARY_RANKS, False),
        "kinship": (KINSHIP, False), "seasons": (SEASONS, False)}

out = {}
for name, (words, notes) in SETS.items():
    rows = [row(w, notes) for w in words]
    out[name] = dict(rows=rows, n_terms=len(rows),
                     n_ge_100=sum(r["best_count"] >= 100 for r in rows),
                     n_ge_30=sum(r["best_count"] >= 30 for r in rows),
                     n_multi_token_only=sum(not r["any_single_token"] for r in rows))
    print(f"\n=== {name}: >=100 {out[name]['n_ge_100']}/{len(rows)} | "
          f">=30 {out[name]['n_ge_30']}/{len(rows)} | "
          f"multi-token-only {out[name]['n_multi_token_only']}")
    for r in rows:
        fs = " ".join(f"{v['form']!r}={v['count'] if v['n_tokens']==1 else 'MT%d' % v['n_tokens']}"
                      for v in r["registered"])
        ex = " ".join(f"{v['form']!r}={v['count'] if v['n_tokens']==1 else 'MT%d' % v['n_tokens']}"
                      for v in r["exploratory_lowercase"])
        bf = repr(r["best"]["form"]) if r["best"] else "NONE"
        print(f"  {r['word']:<13} best={bf:<16} n={r['best_count']:<7} | {fs}"
              + (f"   [lc: {ex}]" if ex else ""))

json.dump(out, open(SCR / "pool_counts.json", "w"), indent=1)
print("\nwrote", SCR / "pool_counts.json")
