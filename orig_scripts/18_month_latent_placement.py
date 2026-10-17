"""Where do the month latents actually end up in the clustering?

The gate can only reject what the clustering hands it. Across the 24 clusters the
`l0_big` gate scored, 713 member latents contained exactly **one** month latent,
so "no cluster recovers the month cycle" is ambiguous between two very different
failures: the clustering assembles month subspaces and the geometry is not
cyclic, or the clustering never assembles a month subspace at all.

This separates them without any geometry. Month latents are identified from the
shipped `top_tokens` (>= 2 distinct month names among a latent's top tokens, the
same basis `02b_precheck_cluster.py` used), and then simply looked up in the
frozen cluster files. No ordering or closure statistic is computed, and nothing
here can change a gate verdict; it only says which stage the negative belongs to.
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", default="5,6,7,8,9,10")
    ap.add_argument("--ontology", default="months", choices=["months", "days"])
    ap.add_argument("--l0", default="big")
    ap.add_argument("--min-distinct", type=int, default=2,
                    help="distinct label names in a latent's top tokens to call it a label latent")
    ap.add_argument("--min-size", type=int, default=6, help="the gate's MIN_RING_K")
    ap.add_argument("--out", default="results/month_latent_placement.json")
    a = ap.parse_args()
    words = MONTHS if a.ontology == "months" else DAYS

    from transformers import AutoTokenizer
    tk = AutoTokenizer.from_pretrained("google/gemma-3-1b-pt",
                                       token=os.environ.get("HF_TOKEN"))
    vocab = [(tk.convert_ids_to_tokens(i) or "").replace("▁", " ")
             for i in range(tk.vocab_size)]

    out = []
    for layer in [int(x) for x in a.layers.split(",")]:
        u = f"{BASE}/layer_{layer}_width_16k_l0_{a.l0}/examples.safetensors"
        h, off = read_header(u, token=None)
        tt = np.asarray(read_tensor(u, "top_tokens", h, off, token=None))
        lat = {}
        for f in range(tt.shape[0]):
            got = {w for t in tt[f] if 0 <= t < len(vocab)
                   for w in words if vocab[t].strip().lower() == w}
            if len(got) >= a.min_distinct:
                lat[f] = sorted(got)

        cl = json.load(open(f"results/clusters_big_L{layer}.json"))[str(layer)]["clusters"]
        placed = {}
        for ci, c in enumerate(cl):
            hit = sorted(m for m in lat if m in set(c["members"]))
            if hit:
                placed[ci] = dict(metric=c["metric"], size=c["size"],
                                  label_latents=hit, n=len(hit))
        in_any = sorted({m for v in placed.values() for m in v["label_latents"]})
        scorable = {ci: v for ci, v in placed.items() if v["size"] >= a.min_size}
        row = dict(layer=layer, ontology=a.ontology,
                   n_label_latents=len(lat), label_latents=sorted(lat),
                   examples={str(k): v for k, v in list(lat.items())[:20]},
                   n_clusters=len(cl),
                   n_clusters_with_label_latent=len(placed),
                   n_label_latents_in_any_cluster=len(in_any),
                   n_label_latents_unclustered=len(lat) - len(in_any),
                   max_label_latents_in_one_cluster=max(
                       (v["n"] for v in placed.values()), default=0),
                   clusters={str(k): v for k, v in placed.items()},
                   clusters_at_or_above_min_size=len(scorable))
        out.append(row)
        print(f"layer {layer}: {len(lat)} {a.ontology} latents; "
              f"{len(in_any)} land in some cluster, {len(lat) - len(in_any)} in none; "
              f"most in any single cluster = {row['max_label_latents_in_one_cluster']}", flush=True)
        for ci, v in sorted(placed.items(), key=lambda kv: -kv[1]["n"]):
            print(f"    cluster#{ci} {v['metric']} k={v['size']}: {v['n']} "
                  f"{a.ontology} latents {v['label_latents']}")
    Path(a.out).write_text(json.dumps(out, indent=2))
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
