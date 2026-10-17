"""Plots: one per recovered geometry, with its control alongside.

Each figure shows the fit and the null it is being judged against on the same
axes, because a ring drawn on its own is persuasive whether or not it means
anything -- the shuffled-label panel is the part that carries the evidence.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .fits import fit_circle


def circle_figure(xy, labels, title, out_path, rho=None, p=None,
                  shuffled_rho=None, null_p95=None):
    xy = np.asarray(xy)
    cx, cy, r, resid = fit_circle(xy)
    fig, ax = plt.subplots(1, 2, figsize=(11, 5))

    th = np.linspace(0, 2 * np.pi, 200)
    ax[0].plot(cx + r * np.cos(th), cy + r * np.sin(th), "-", lw=1, color="0.7", zorder=1)
    order = np.argsort(np.arctan2(xy[:, 1] - cy, xy[:, 0] - cx))
    ring = xy[order]
    ax[0].plot(np.append(ring[:, 0], ring[0, 0]), np.append(ring[:, 1], ring[0, 1]),
               "-", lw=0.8, color="C0", alpha=0.5, zorder=2)
    ax[0].scatter(xy[:, 0], xy[:, 1], s=60, zorder=3, color="C0")
    for i, lab in enumerate(labels):
        ax[0].annotate(lab, (xy[i, 0], xy[i, 1]), fontsize=9,
                       xytext=(5, 4), textcoords="offset points")
    ax[0].set_title(f"{title}\ncircle residual {resid:.3f}")
    ax[0].set_aspect("equal"); ax[0].set_xlabel("PC1"); ax[0].set_ylabel("PC2")

    # the control panel: where the statistic sits against its own null
    if rho is not None:
        bars = {"recovered": rho}
        if shuffled_rho is not None:
            bars["shuffled labels"] = shuffled_rho
        if null_p95 is not None:
            bars["null 95th pct"] = null_p95
        ax[1].barh(list(bars), list(bars.values()),
                   color=["C2", "C3", "0.6"][:len(bars)])
        ax[1].set_xlim(-1, 1); ax[1].axvline(0, color="k", lw=0.8)
        ax[1].set_xlabel("cyclic-order Spearman")
        ax[1].set_title(f"ordering vs controls" + (f"   p = {p:.4f}" if p is not None else ""))
        for i, v in enumerate(bars.values()):
            ax[1].text(v, i, f" {v:+.3f}", va="center", fontsize=9)
    fig.tight_layout(); fig.savefig(out_path, dpi=140); plt.close(fig)
    return out_path


def layer_summary(rows, out_path):
    """Best cyclic-order rho per layer, split by similarity metric."""
    fig, ax = plt.subplots(figsize=(7, 4))
    for metric, color in (("cosine", "C0"), ("coactivation", "C1")):
        pts = {}
        for r in rows:
            if r["metric"] != metric:
                continue
            pts[r["layer"]] = max(pts.get(r["layer"], -1), r["lab"]["rho"])
        if pts:
            xs = sorted(pts)
            ax.plot(xs, [pts[x] for x in xs], "o-", label=metric, color=color)
    ax.set_xlabel("layer"); ax.set_ylabel("best cyclic-order Spearman")
    ax.axhline(0, color="k", lw=0.8); ax.legend(); ax.set_ylim(-1.05, 1.05)
    ax.set_title("Recovered cyclic ordering by layer and similarity metric")
    fig.tight_layout(); fig.savefig(out_path, dpi=140); plt.close(fig)
    return out_path
