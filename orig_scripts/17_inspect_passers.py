"""What is actually IN the clusters the gate scored.

This is the check that dissolved the previous apparent positive: three layer-18
clusters passed ordering at BH-FDR with p as low as 0.0019 and contained 0 of 31
month or temporal latents -- their top tokens were ` Xiaomi`, ` DeFi`,
` Kubernetes`, ` lmao`, Malayalam script. A cluster that recovers the month
ordering while containing no month latent has recovered the ordering by accident.

Member top tokens come from the `top_tokens` tensor shipped in each SAE's
`examples.safetensors`, read by HTTP range (`gfdisc.hf_range`) so the 733 MB file
is never downloaded. Every scored cluster is inspected, not only the passers: a
gate that scores no month-bearing cluster at all is a different negative from a
gate that scores them and rejects them, and only the full listing separates the
two.
"""
import argparse, json, os, sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from gfdisc.hf_range import read_header, read_tensor

BASE = ("https://huggingface.co/google/gemma-scope-2-1b-pt/resolve/main/"
        "resid_post_all")

MONTHS = ["january", "february", "march", "april", "may", "june", "july",
          "august", "september", "october", "november", "december"]
DAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
# Deliberately generous. The question this answers is "could this cluster be about
# time at all", so a false ACCEPT here is the conservative direction: it makes a
# passer harder to call a false positive, not easier.
TEMPORAL = ["year", "years", "month", "months", "week", "weeks", "day", "days",
            "hour", "hours", "minute", "minutes", "date", "dates", "calendar",
            "quarter", "season", "spring", "summer", "autumn", "fall", "winter",
            "today", "tomorrow", "yesterday", "annual", "annually", "monthly",
            "weekly", "daily", "yearly", "morning", "evening", "afternoon",
            "night", "weekend", "decade", "century", "anniversary", "semester",
            "am", "pm", "time", "times", "when", "since", "until", "during"]


def norm(s):
    return s.strip().lower().rstrip("s")


def classify(tokens):
    """Which temporal families this latent's top tokens touch."""
    got = {"month": set(), "day": set(), "temporal": set()}
    for t in tokens:
        s = norm(t)
        for name, vocab in (("month", MONTHS), ("day", DAYS), ("temporal", TEMPORAL)):
            for w in vocab:
                if s == norm(w):
                    got[name].add(w)
    return got


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gates", nargs="+", required=True)
    ap.add_argument("--l0", default="big")
    ap.add_argument("--n-top", type=int, default=12, help="top tokens per latent to read")
    ap.add_argument("--show", type=int, default=10, help="member latents to print per cluster")
    ap.add_argument("--out", default="results/passer_inspection.json")
    a = ap.parse_args()
    token = os.environ.get("HF_TOKEN")

    from transformers import AutoTokenizer
    tk = AutoTokenizer.from_pretrained("google/gemma-3-1b-pt", token=token)
    vocab = [(tk.convert_ids_to_tokens(i) or "").replace("▁", " ")
             for i in range(tk.vocab_size)]

    rows = []
    for gf in a.gates:
        g = json.loads(Path(gf).read_text())
        for layer in sorted({r["layer"] for r in g["rows"]}):
            u = f"{BASE}/layer_{layer}_width_16k_l0_{a.l0}/examples.safetensors"
            h, off = read_header(u, token=None)
            tt = np.asarray(read_tensor(u, "top_tokens", h, off, token=None))
            print(f"\n=== {Path(gf).name}  layer {layer}  ({g['ontology']}) "
                  f"top_tokens {tt.shape}", flush=True)
            for r in [x for x in g["rows"] if x["layer"] == layer]:
                tag = ("CIRCLE" if r["CIRCLE"] else "arc_only" if r["arc_only"] else "scored")
                per = []
                for m in r["members"]:
                    toks = [vocab[t] for t in tt[m][:a.n_top] if 0 <= t < len(vocab)]
                    per.append(dict(latent=int(m), tokens=toks, **{
                        k: sorted(v) for k, v in classify(toks).items()}))
                n_mon = sum(1 for p in per if p["month"])
                n_day = sum(1 for p in per if p["day"])
                n_tmp = sum(1 for p in per if p["temporal"] or p["month"] or p["day"])
                rows.append(dict(gate=Path(gf).name, layer=layer, ontology=g["ontology"],
                                 metric=r["metric"], size=r["size"], tag=tag,
                                 rho=r["rho"], p_order=r["p_order"],
                                 p_closed=r["p_closed"], index_p=r["index_p"],
                                 n_month=n_mon, n_day=n_day, n_temporal=n_tmp,
                                 members=per))
                print(f"  [{tag:8s}] {r['metric'][:5]} k={r['size']:3d} rho {r['rho']:+.3f} "
                      f"pOrd {r['p_order']:.4f} pClo {r['p_closed']:.3f} | "
                      f"month {n_mon}/{r['size']}  day {n_day}/{r['size']}  "
                      f"temporal-or-calendar {n_tmp}/{r['size']}")
                if tag != "scored" or a.show:
                    for p in per[:a.show]:
                        mark = ("M" if p["month"] else "D" if p["day"]
                                else "t" if p["temporal"] else " ")
                        print(f"      {mark} {p['latent']:6d}  "
                              f"{', '.join(repr(t) for t in p['tokens'][:8])}")
    Path(a.out).write_text(json.dumps(rows, indent=2))
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
