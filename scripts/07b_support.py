"""Within-cluster co-activation support, read back off an existing cluster file.

At L0 = 20 the co-activation graph was empty by construction: inside a 22-latent
cluster only ~1.13 latents fired per position, so "co-activation" was measuring
pairs that essentially never co-occurred. That failure mode is invisible in the
cluster count -- Leiden will happily partition a graph built from a handful of
weak edges -- so it needs its own number.

For each co-activation cluster this reports, over the cached corpus:
  mean_pair_cond   mean over ordered pairs (i != j) of P(j fires | i fires)
  mean_co_fire     mean number of cluster members firing at positions where at
                   least one member fires
Sizes and supports only; no labels, geometry or ordering statistic is touched.
"""
import argparse, json, sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from gfdisc.cache import SparseReader
from gfdisc.cluster import binary_matrix


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clusters", required=True, help="a clusters_*.json from script 07")
    ap.add_argument("--root", default="cache/sparse_big")
    ap.add_argument("--metric", default="coactivation")
    ap.add_argument("--out", default="results/cluster_support.json")
    a = ap.parse_args()

    data = json.loads(Path(a.clusters).read_text())
    out = {}
    for layer, rec in data.items():
        cl = [r["members"] for r in rec["clusters"] if r["metric"] == a.metric]
        if not cl:
            out[layer] = dict(n_clusters=0)
            continue
        reader = SparseReader(a.root, int(layer))
        union = np.unique(np.concatenate([np.asarray(c) for c in cl]))
        pos = {int(f): k for k, f in enumerate(union)}
        # One column slice for every cluster at once: |union| is at most a few
        # thousand, so this stays far below the full-width binary matrix.
        B = binary_matrix(reader, union).tocsc()

        rows = []
        for c in cl:
            cols = [pos[int(f)] for f in c]
            S = B[:, cols].tocsr()
            joint = np.asarray((S.T @ S).todense(), dtype=np.float64)
            counts = np.diag(joint).copy()
            k = len(cols)
            cond = joint / np.maximum(counts[:, None], 1.0)
            np.fill_diagonal(cond, 0.0)
            per_row = np.asarray(S.sum(axis=1)).ravel()
            fired = per_row > 0
            rows.append(dict(size=k,
                             mean_pair_cond=float(cond.sum() / max(k * (k - 1), 1)),
                             mean_co_fire=float(per_row[fired].mean()) if fired.any() else 0.0,
                             median_member_count=float(np.median(counts)),
                             n_positions_any=int(fired.sum())))
            del S, joint, cond
        del B

        m = lambda key: float(np.mean([r[key] for r in rows]))
        out[layer] = dict(n_clusters=len(rows), metric=a.metric,
                          mean_pair_cond=m("mean_pair_cond"),
                          mean_co_fire=m("mean_co_fire"), clusters=rows)
        print(f"layer {layer}: {len(rows):3d} clusters, mean pairwise P(j|i) = "
              f"{out[layer]['mean_pair_cond']:.3f}, mean members co-firing = "
              f"{out[layer]['mean_co_fire']:.2f}", flush=True)

    Path(a.out).parent.mkdir(exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=2))
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
