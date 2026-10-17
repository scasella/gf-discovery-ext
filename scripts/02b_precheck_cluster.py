"""Step 2b — corrected pre-check: size the candidate manifold *cluster*, not per-day features.

The first pass asked how many weekdays had a dedicated feature and found almost
none. Inspecting layer 22 showed why that was the wrong question: feature 374's
top tokens are Wednesday/yesterday/Tuesday/Thursday/Monday/Friday/Saturday/Sunday
- one atom spanning the whole cycle. Per Bhalla et al. (arXiv 2604.28119) that is
the *global* encoding regime, where a manifold lives in the span of a compact
group of atoms rather than in one-feature-per-value.

So the quantity that predicts whether a weekday circle is recoverable is the size
and coherence of the calendar-related feature group. Engels et al. found the
GPT-2-small weekday cluster to be 9 features; that is the scale to look for.
"""
import argparse, json, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from gfdisc.hf_range import read_header, read_tensor

BASE = "https://huggingface.co/google/gemma-scope-2-1b-pt/resolve/main"
DAYS = ["monday","tuesday","wednesday","thursday","friday","saturday","sunday"]
MONTHS = ["january","february","march","april","may","june","july","august",
          "september","october","november","december"]


def token_hits(tokens_row, vocab, targets):
    """How many of a feature's top tokens ARE a target word, and how many distinct."""
    n, distinct = 0, set()
    for t in tokens_row:
        s = vocab[t].strip().lower().rstrip("s") if 0 <= t < len(vocab) else ""
        for tgt in targets:
            if s == tgt or s == tgt.rstrip("s"):
                n += 1; distinct.add(tgt); break
    return n, len(distinct)


def probe(layer, suite, site, vocab):
    u = f"{BASE}/{site}/layer_{layer}_width_16k_l0_{suite}/examples.safetensors"
    try:
        h, off = read_header(u)
        tt = read_tensor(u, "top_tokens", h, off)
        ff = read_tensor(u, "feature_frequencies", h, off)
    except Exception as e:
        return dict(layer=layer, suite=suite, error=str(e)[:100])
    res = dict(layer=layer, suite=suite, dead=int((ff < 1e-6).sum()))
    for name, targets in (("day", DAYS), ("month", MONTHS)):
        counts = [token_hits(tt[f], vocab, targets) for f in range(tt.shape[0])]
        n2 = [i for i, (n, d) in enumerate(counts) if n >= 2]
        n4 = [i for i, (n, d) in enumerate(counts) if n >= 4]
        allv = set()
        for i in n2:
            allv |= {t for t in targets if any(
                vocab[x].strip().lower().rstrip("s") == t.rstrip("s") for x in tt[i])}
        res[f"{name}_cluster"] = len(n2)     # candidate subspace size
        res[f"{name}_spanning"] = len(n4)    # atoms spanning much of the cycle
        res[f"{name}_values_seen"] = len(allv)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", default="8-25")
    ap.add_argument("--suites", default="big")
    ap.add_argument("--site", default="resid_post_all")
    ap.add_argument("--out", default="results/precheck_cluster.json")
    a = ap.parse_args()
    lo, _, hi = a.layers.partition("-")
    layers = list(range(int(lo), int(hi) + 1)) if hi else [int(lo)]

    from transformers import AutoTokenizer
    tk = AutoTokenizer.from_pretrained("google/gemma-3-1b-pt")
    vocab = [(tk.convert_ids_to_tokens(i) or "").replace("▁", " ") for i in range(tk.vocab_size)]

    jobs = [(l, s) for s in a.suites.split(",") for l in layers]
    with ThreadPoolExecutor(max_workers=8) as ex:
        rows = list(ex.map(lambda j: probe(j[0], j[1], a.site, vocab), jobs))
    rows = [r for r in rows if "error" not in r]
    rows.sort(key=lambda r: (r["suite"], r["layer"]))

    print(f"\n{'suite':7s}{'layer':>6s}{'dead':>6s} | {'dayClust':>9s}{'daySpan':>8s}{'days':>6s} | "
          f"{'monClust':>9s}{'monSpan':>8s}{'mons':>6s}")
    print("-" * 74)
    for r in rows:
        print(f"{r['suite']:7s}{r['layer']:6d}{r['dead']:6d} | {r['day_cluster']:9d}{r['day_spanning']:8d}"
              f"{r['day_values_seen']:6d} | {r['month_cluster']:9d}{r['month_spanning']:8d}{r['month_values_seen']:6d}")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(rows, indent=2))
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
