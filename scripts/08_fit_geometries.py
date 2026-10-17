"""Step 8 -- fit geometry to candidate clusters, then score against labels.

Order matters and is enforced here: every cluster is first scored *without* labels
(top-2 PC variance fraction and residual from a fitted circle over the positions
where the cluster fires). Only then are calendar labels brought in to ask whether
the recovered angle follows the actual cycle. Reversing that order would be
choosing the cluster because it matches the answer.

Three controls accompany every ordering claim:
  - shuffled labels, which must collapse;
  - an index-stratified null, because Gemma Scope 2 is Matryoshka-trained and
    latent index correlates with firing frequency at rho = -0.48;
  - Benjamini-Hochberg FDR across all clusters x layers actually examined.
"""
import argparse, json, os, sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from gfdisc.cache import SparseReader
from gfdisc.saes import load_sae, get_decoder
from gfdisc.fits import pca_2d, fit_circle, angles, cyclic_order_score, degenerate
from gfdisc.nulls import benjamini_hochberg, index_concentration
from gfdisc.tokens import DAYS, MONTHS


def unsupervised_score(reader, cluster, w_dec, max_pos=20000, rng=None):
    """Label-free: is this cluster's activation cloud a low-dimensional ring?"""
    rows = reader.rows_firing(cluster)
    if len(rows) < 20:
        return None
    if len(rows) > max_pos:
        rows = rng.choice(rows, max_pos, replace=False)
    X, _ = reader.reconstruct_fast(rows, cluster, w_dec)
    xy, ratios, _ = pca_2d(X)
    cx, cy, r, resid = fit_circle(xy)
    return dict(n_positions=int(len(rows)), top2_var=float(ratios[:2].sum()),
                circle_resid=float(resid), pc3_ratio=float(ratios[2]) if len(ratios) > 2 else 0.0)


MIN_CLUSTER_SIZE = 5           # below this, top2_var is a size artifact
MIN_FIRING_PER_LABEL = 10      # positions where the cluster actually fires
MIN_LABELS_WITH_SUPPORT = 5    # of 7 days / 12 months


def labelled_score(reader, cluster, w_dec, label_pos, n_perm=2000, seed=0):
    """Post-hoc: does the recovered angle follow the known cycle?

    `label_pos` is precomputed per layer -- resolving token positions inside this
    function meant re-scanning 4M token ids once per label per cluster.
    """
    means, keep, support = [], [], []
    for k, pos in enumerate(label_pos):
        if len(pos) < 5:
            continue
        A = reader.codes_at(pos, cluster)
        n_fire = int((A > 0).any(1).sum())
        if n_fire < MIN_FIRING_PER_LABEL:
            continue          # this label has no evidence from this cluster
        means.append((A @ np.asarray(w_dec)[np.asarray(cluster)]).mean(0))
        keep.append(k); support.append(n_fire)
    if len(keep) < MIN_LABELS_WITH_SUPPORT:
        return None
    M = np.vstack(means)
    if degenerate(M):
        return None
    xy, ratios, _ = pca_2d(M)
    cx, cy, r, resid = fit_circle(xy)
    sc = cyclic_order_score(angles(xy, cx, cy), np.arange(len(keep)), len(keep),
                            n_perm=n_perm, seed=seed)
    rng = np.random.default_rng(seed + 1)
    shuf = cyclic_order_score(angles(xy, cx, cy), rng.permutation(len(keep)),
                              len(keep), n_perm=n_perm, seed=seed)
    return dict(labels_used=keep, n_labels=len(keep), support=support, top2_var=float(ratios[:2].sum()),
                circle_resid=float(resid), rho=sc["rho"], p=sc["p"],
                null_p95=sc["null_p95"], shuffled_rho=shuf["rho"], shuffled_p=shuf["p"],
                points=xy.tolist())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clusters", default="results/clusters.json")
    ap.add_argument("--root", default="cache/sparse")
    ap.add_argument("--l0", default="small")
    ap.add_argument("--ontology", default="days", choices=["days", "months"])
    ap.add_argument("--top-k", type=int, default=25, help="clusters per layer scored against labels")
    ap.add_argument("--out", default="results/fits.json")
    a = ap.parse_args()

    token = os.environ["HF_TOKEN"]
    tmap = json.load(open("results/token_map.json"))
    names = DAYS if a.ontology == "days" else MONTHS
    key = "days" if a.ontology == "days" else "months"
    label_ids = [tmap[key][n] for n in names]

    clusters = json.load(open(a.clusters))
    rng = np.random.default_rng(0)
    results = []

    for layer_s, info in clusters.items():
        layer = int(layer_s)
        sae = load_sae(layer, a.l0, token=token); W = get_decoder(sae)
        reader = SparseReader(a.root, layer)
        label_pos = [reader.positions_of_tokens(ids) for ids in label_ids]
        print(f"\nlayer {layer}: {len(info['clusters'])} clusters | label positions: "
              f"{[len(p) for p in label_pos]}", flush=True)

        scored = []
        for c in info["clusters"]:
            u = unsupervised_score(reader, c["members"], W, rng=rng)
            if u:
                scored.append((c, u))
        # rank WITHOUT labels, then look at labels only for the top ranked
        # top2_var is trivially ~1.0 for a 3-feature cluster (2 of <=3 dimensions),
        # so ranking on it selects the smallest clusters rather than the roundest.
        # Rank on circle-fit quality instead, and require enough atoms to span a
        # manifold at all -- Engels et al.'s weekday cluster was 9 features.
        scored = [t for t in scored if t[0]["size"] >= MIN_CLUSTER_SIZE]
        scored.sort(key=lambda t: (-t[1]["circle_resid"], t[1]["top2_var"]), reverse=True)
        print(f"  {len(scored)} scorable; taking top {a.top_k} by label-free circularity")

        for c, u in scored[:a.top_k]:
            lab = labelled_score(reader, c["members"], W, label_pos)
            if lab is None:
                continue
            iqr, null5, ip = index_concentration(c["members"], sae.d_sae, rng, 300)
            results.append(dict(layer=layer, metric=c["metric"], size=c["size"],
                                members=c["members"], unsup=u, lab=lab,
                                index_p=float(ip), median_index=c["median_index"]))
            print(f"    [{c['metric'][:5]}] size {c['size']:3d} | top2 {u['top2_var']:.3f} "
                  f"| rho {lab['rho']:+.3f} p {lab['p']:.4f} | shuf rho {lab['shuffled_rho']:+.3f} "
                  f"| idxP {ip:.3f}")

    if results:
        keep = benjamini_hochberg([r["lab"]["p"] for r in results], alpha=0.05)
        for r, k in zip(results, keep):
            r["fdr_pass"] = bool(k)
        print(f"\n{sum(keep)}/{len(results)} survive BH-FDR at alpha=0.05")
    Path(a.out).write_text(json.dumps(dict(ontology=a.ontology, results=results), indent=2))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
