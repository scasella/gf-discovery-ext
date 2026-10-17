"""Step 7 -- similarity matrices and clustering, cosine vs co-activation co-primary.

Bhalla et al. report that decoder cosine similarity *fails* to recover manifold
block structure while conditional co-activation and Ising couplings succeed, on a
test set that includes days -- our Phase-1 gate concept. Their comparison is small
(three manifolds, top-3 features each, one SAE), so rather than take either metric
on faith we run both and report the agreement between the cluster sets. Agreement
is evidence; disagreement localises which encoding regime the SAE chose.

Every cluster is also scored for index concentration, because Gemma Scope 2 is
Matryoshka-trained and latent index correlates with firing frequency at
rho = -0.48 on these SAEs. A cluster that is merely a narrow index band is a
training artifact, not a concept.
"""
import argparse, json, os, sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from gfdisc.cache import SparseReader
from gfdisc.saes import load_sae, get_decoder
from gfdisc.cluster import cosine_matrix, coactivation_matrix, graph_clusters
from gfdisc.nulls import index_concentration


def jaccard(a, b):
    A, B = set(a), set(b)
    return len(A & B) / max(len(A | B), 1)


def best_match(cluster, others):
    """Closest cluster in the other partition, by Jaccard."""
    best, arg = 0.0, None
    for j, o in enumerate(others):
        s = jaccard(cluster, o)
        if s > best:
            best, arg = s, j
    return best, arg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", default="17,19,20,21,22,23,24")
    ap.add_argument("--root", default="cache/sparse")
    ap.add_argument("--l0", default="small")
    ap.add_argument("--cos-thresh", type=float, default=0.4)
    ap.add_argument("--coact-thresh", type=float, default=0.25)
    ap.add_argument("--min-count", type=int, default=50)
    ap.add_argument("--max-size", type=int, default=200)
    ap.add_argument("--out", default="results/clusters.json")
    a = ap.parse_args()
    # None falls through to the credential huggingface_hub has stored (hf auth login),
    # so the token never has to be pasted onto a command line.
    token = os.environ.get("HF_TOKEN") or None
    rng = np.random.default_rng(0)
    out = {}

    for layer in [int(x) for x in a.layers.split(",")]:
        sae = load_sae(layer, a.l0, token=token)
        W = get_decoder(sae)
        reader = SparseReader(a.root, layer)
        counts = reader.firing_counts()
        alive = counts >= a.min_count
        print(f"\nlayer {layer}: {len(reader):,} cached positions, "
              f"{alive.sum():,}/{len(counts):,} latents firing >= {a.min_count}x", flush=True)

        # Work in the restricted "alive" index space -- a full-width co-activation
        # matrix would be 16384^2 float64 (2.1 GB) and mostly dead rows -- then map
        # cluster members back to true latent indices before anything is reported.
        alive_idx = np.flatnonzero(alive)
        back = lambda c: [int(alive_idx[i]) for i in c]

        cos = cosine_matrix(W[alive_idx])
        raw_cos = [back(c) for c in graph_clusters(cos, a.cos_thresh, 3)]
        cl_cos = [c for c in raw_cos if len(c) <= a.max_size]
        del cos

        coact, _ = coactivation_matrix(reader, features=alive_idx, min_count=a.min_count)
        np.minimum(coact, coact.T, out=coact)  # both directions strong; in place
        sym = coact
        raw_co = [back(c) for c in graph_clusters(sym, a.coact_thresh, 3)]
        cl_co = [c for c in raw_co if len(c) <= a.max_size]
        del coact, sym

        # Pre-filter sizes. The max-size filter above silently discards a giant
        # component, so the threshold-recalibration rule -- which reads "largest
        # cluster <= 200" -- cannot be applied from the filtered output alone.
        # Sizes only: no membership, label or geometry enters this.
        raw_sizes = dict(cosine=sorted((len(c) for c in raw_cos), reverse=True),
                         coactivation=sorted((len(c) for c in raw_co), reverse=True))
        for tag, sz in raw_sizes.items():
            print(f"  {tag:12s} pre-filter: {len(sz):3d} clusters, largest {sz[0] if sz else 0}, "
                  f"{sum(s >= 6 for s in sz):3d} of size >= 6")

        print(f"  cosine       -> {len(cl_cos):3d} clusters (3..{a.max_size})")
        print(f"  coactivation -> {len(cl_co):3d} clusters (3..{a.max_size})")

        # how much do the two partitions agree?
        agree = [best_match(c, cl_co)[0] for c in cl_cos] or [0.0]
        print(f"  mean best-match Jaccard cosine->coact: {np.mean(agree):.3f} "
              f"(>0.5 for {sum(x > .5 for x in agree)}/{len(agree)} clusters)")

        def describe(cl, tag):
            rows = []
            for c in cl:
                iqr, null5, p = index_concentration(c, len(counts), rng, 300)
                rows.append(dict(metric=tag, members=[int(x) for x in c], size=len(c),
                                 median_index=int(np.median(c)), index_iqr=iqr,
                                 index_p=p, median_count=float(np.median(counts[c]))))
            return rows

        rows = describe(cl_cos, "cosine") + describe(cl_co, "coactivation")
        susp = sum(r["index_p"] < 0.05 for r in rows)
        print(f"  index-concentrated (Matryoshka artifact risk): {susp}/{len(rows)}")
        out[str(layer)] = dict(n_positions=len(reader), n_alive=int(alive.sum()),
                               jaccard_mean=float(np.mean(agree)),
                               coact_thresh=a.coact_thresh, cos_thresh=a.cos_thresh,
                               raw_cluster_sizes=raw_sizes, clusters=rows)

    Path(a.out).parent.mkdir(exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=2))
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
