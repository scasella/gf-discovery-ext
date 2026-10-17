"""Figures for the closure-vs-depth result. CPU only: reads JSON, loads no model.

Two figures, both drawn from persisted results so they can be regenerated without
a forward pass:

  figures/closure_depth.png       -- from results/closure_depth.json. The whole
      point of the sweep is that ordering does not discriminate and closure does,
      so the plotted statistics are the closure ones (p_closed, gap ratio) and the
      ordering result is stated in the caption rather than given a panel.

  figures/weekday_circle_layer21.png -- redrawn from the layer-21 raw points stored
      in results/positive_control.json, because the original title called a
      configuration a "circle" that the closure test rejects. Every number in the
      new title is recomputed from those exact points rather than copied from the
      sweep, which used a different template set and so does not agree to 3 d.p.

The month layer-21 panel cannot be regenerated: its points were never persisted
and recovering them needs model inference. It is removed rather than relabelled.
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from gfdisc.fits import pca_2d, fit_circle, circularity, angles, cyclic_order_score
from gfdisc.tokens import DAYS

# Validated categorical slots 1 and 2 for the two calendar ontologies; the ordinal
# control is deliberately not given a hue, because it is a control and should read
# as one at a glance.
C_WEEK, C_MONTH, C_NUM = "#2a78d6", "#eb6834", "#8a8a85"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#9a998f"
SURFACE = "#fcfcfb"
ALPHA = 0.05

SERIES = (("weekdays", "weekdays (k=7)", C_WEEK, "-", "o"),
          ("months", "months (k=12)", C_MONTH, "-", "s"),
          ("numbers", "number words (ordinal control)", C_NUM, "--", "^"))


def gap_pvalue(k, observed_max_gap, n_perm=20000, seed=0):
    """Identical to the sweep's null in 12_closure_depth.py, restated so the
    redrawn figure's p-value is computed the same way as the swept one."""
    rng = np.random.default_rng(seed)
    a = np.sort(rng.uniform(0, 2 * np.pi, size=(n_perm, k)), axis=1)
    g = np.diff(np.concatenate([a, a[:, :1] + 2 * np.pi], axis=1), axis=1).max(1)
    return float((1 + (g <= observed_max_gap).sum()) / (1 + n_perm))


def runs(layers):
    """Contiguous runs, so the shaded bands mark the layers that actually pass
    rather than the convex hull of them -- layer 17 sits inside 16-20 and fails."""
    out = []
    for x in sorted(layers):
        if out and x == out[-1][1] + 1:
            out[-1][1] = x
        else:
            out.append([x, x])
    return [tuple(r) for r in out]


def style(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=MUTED, alpha=0.25, lw=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=INK2, labelsize=9)


def closure_depth_figure(rows, out_path):
    by = {(r["ontology"], r["layer"]): r for r in rows}
    layers = sorted({r["layer"] for r in rows})
    month_closed = [l for l in layers if by[("months", l)]["p_closed"] < ALPHA]

    fig, ax = plt.subplots(2, 1, figsize=(9, 7), sharex=True,
                           gridspec_kw=dict(height_ratios=[1.25, 1], hspace=0.12))
    fig.patch.set_facecolor(SURFACE)

    for a in ax:
        style(a)
        for i, (lo, hi) in enumerate(runs(month_closed)):
            a.axvspan(lo - 0.45, hi + 0.45, color=C_MONTH, alpha=0.08, lw=0,
                      zorder=0, label="months: closure significant" if i == 0 else None)

    for key, label, color, ls, marker in SERIES:
        pc = [by[(key, l)]["p_closed"] for l in layers]
        gr = [by[(key, l)]["gap_ratio"] for l in layers]
        ax[0].plot(layers, pc, ls, color=color, lw=2, marker=marker, ms=5,
                   label=label, zorder=3)
        ax[1].plot(layers, gr, ls, color=color, lw=2, marker=marker, ms=5,
                   label=label, zorder=3)

    ax[0].set_yscale("log")
    ax[0].axhline(ALPHA, color=INK2, lw=1.2, ls=(0, (4, 3)), zorder=2)
    ax[0].text(25.4, ALPHA * 1.25, "p = 0.05", ha="right", va="bottom",
               fontsize=9, color=INK2)
    ax[0].set_ylabel("p(closed)  — max angular gap vs uniform null", fontsize=10,
                     color=INK2)
    ax[0].set_title("Closure is depth-dependent and non-monotonic; ordering is not",
                    fontsize=13, color=INK, loc="left", pad=30)

    m18 = by[("months", 18)]
    ax[0].annotate(f"months, layer 18\np = {m18['p_closed']:.4f}",
                   xy=(18, m18["p_closed"]), xytext=(19.6, 3.5e-4),
                   fontsize=9, color=INK2,
                   arrowprops=dict(arrowstyle="-", color=MUTED, lw=1))
    ax[0].annotate("layer 17 fails between\nthe two shaded bands",
                   xy=(17, by[("months", 17)]["p_closed"]),
                   xytext=(12.4, 0.022), va="top", fontsize=9, color=INK2,
                   arrowprops=dict(arrowstyle="-", color=MUTED, lw=1))

    ax[1].axhline(1.0, color=INK2, lw=1.2, ls=(0, (4, 3)), zorder=2)
    ax[1].text(25.4, 1.06, "1.0 = evenly spaced", ha="right", va="bottom",
               fontsize=9, color=INK2)
    ax[1].set_ylabel("angular gap ratio\n(largest gap / 2πk⁻¹)",
                     fontsize=10, color=INK2)
    ax[1].set_xlabel("residual-stream layer", fontsize=10, color=INK2)
    ax[1].set_xlim(-0.6, 25.6)
    ax[1].set_xticks(range(0, 26, 2))
    ax[1].set_ylim(0, 8)

    h, l = ax[0].get_legend_handles_labels()
    order = [l.index(x) for x in (
        "weekdays (k=7)", "months (k=12)", "number words (ordinal control)",
        "months: closure significant")]
    ax[0].legend([h[i] for i in order], [l[i] for i in order], frameon=False,
                 fontsize=9, labelcolor=INK2, ncol=4, loc="lower left",
                 bbox_to_anchor=(0, 1.0, 1, 0.1), mode="expand",
                 borderaxespad=0, handlelength=1.8, columnspacing=1.2)

    week = [by[("weekdays", l)]["rho"] for l in layers]
    mon = [by[("months", l)]["rho"] for l in layers]
    num = [by[("numbers", l)]["rho"] for l in layers]
    caption = (
        "Cyclic-ordering Spearman is near-perfect for all three series at nearly every layer "
        f"(ρ ≥ 0.99 at {sum(r >= 0.99 for r in week)}/26 layers for weekdays, "
        f"{sum(r >= 0.99 for r in mon)}/26 for months, and "
        f"{sum(r >= 0.99 for r in num)}/26 for the number-word control, which never drops below "
        f"{min(num):.2f}), so ordering does not separate the calendar cycles from an ontology that is\n"
        "definitionally not cyclic. Closure does: the control fails at 0 of 26 layers, months close at "
        "layers 0–10, 16 and 18–20, and weekdays only at layers 0–4. Source: results/closure_depth.json."
    )
    fig.text(0.5, 0.012, caption, ha="center", va="bottom", fontsize=8.5,
             color=INK2, wrap=True)
    fig.subplots_adjust(left=0.11, right=0.97, top=0.885, bottom=0.16)
    fig.savefig(out_path, dpi=140, facecolor=SURFACE)
    plt.close(fig)
    return out_path


def arc_figure(xy, labels, ontology, layer, sweep, out_path, rho, p_order,
               shuffled_rho, null_p95):
    """The layer-21 configuration, titled by what the closure test says it is."""
    xy = np.asarray(xy)
    cx, cy, r, resid = fit_circle(xy)
    circ = circularity(xy, cx, cy, r)
    p_closed = gap_pvalue(len(xy), np.radians(circ["max_gap_deg"]))

    fig, ax = plt.subplots(1, 2, figsize=(11, 5.2))
    fig.patch.set_facecolor(SURFACE)
    for a in ax:
        style(a)

    th = np.linspace(0, 2 * np.pi, 400)
    ax[0].plot(cx + r * np.cos(th), cy + r * np.sin(th), "-", lw=1,
               color=MUTED, zorder=1)
    order = np.argsort(np.arctan2(xy[:, 1] - cy, xy[:, 0] - cx))
    ring = xy[order]

    # Draw the largest angular gap as an explicit dashed chord: the gap is the
    # measurement that fails, so it should be the thing the eye lands on.
    a_sorted = np.sort(np.arctan2(ring[:, 1] - cy, ring[:, 0] - cx))
    gaps = np.diff(np.concatenate([a_sorted, [a_sorted[0] + 2 * np.pi]]))
    g = int(np.argmax(gaps))
    seam = np.array([ring[g], ring[(g + 1) % len(ring)]])

    closed = np.vstack([ring, ring[:1]])
    ax[0].plot(closed[:, 0], closed[:, 1], "-", lw=2, color=C_WEEK, alpha=0.55,
               zorder=2)
    ax[0].plot(seam[:, 0], seam[:, 1], ls=(0, (4, 3)), lw=2, color=INK2, zorder=3)
    ax[0].scatter(xy[:, 0], xy[:, 1], s=70, zorder=4, color=C_WEEK,
                  edgecolors=SURFACE, linewidths=2)
    for i, lab in enumerate(labels):
        ax[0].annotate(lab, (xy[i, 0], xy[i, 1]), fontsize=9, color=INK,
                       xytext=(6, 5), textcoords="offset points")
    mid = seam.mean(0)
    ax[0].annotate(f"largest gap {circ['max_gap_deg']:.0f}°", mid,
                   fontsize=9, color=INK2, ha="center",
                   xytext=(0, 8), textcoords="offset points")
    ax[0].set_title(
        f"{ontology} ordered arc — layer {layer} residual stream\n"
        f"closure fails: p_closed = {p_closed:.3f}, gap ratio {circ['gap_ratio']:.2f}",
        fontsize=12, color=INK)
    ax[0].set_aspect("equal")
    ax[0].set_xlabel("PC1", fontsize=10, color=INK2)
    ax[0].set_ylabel("PC2", fontsize=10, color=INK2)

    bars = {"null 95th pct": null_p95, "shuffled labels": shuffled_rho,
            "recovered": rho}
    colors = [MUTED, MUTED, C_WEEK]
    ax[1].barh(list(bars), list(bars.values()), color=colors, height=0.55)
    ax[1].set_xlim(-1, 1)
    ax[1].axvline(0, color=INK2, lw=0.8)
    ax[1].set_xlabel("cyclic-order Spearman", fontsize=10, color=INK2)
    ax[1].set_title(f"ordering passes anyway   p = {p_order:.4f}", fontsize=12,
                    color=INK)
    for i, v in enumerate(bars.values()):
        ax[1].text(v + 0.03, i, f"{v:+.3f}", va="center", fontsize=9, color=INK2)
    fig.text(0.5, 0.015,
             "The ordering statistic maximises Spearman over the 2k rotations and reflections of a k-cycle, "
             "a set that contains the identity, so any monotone\narrangement scores perfectly — which is why the "
             "left panel, not the right, decides whether this is a ring. The full sweep in "
             f"results/closure_depth.json puts\nthis configuration at gap ratio {sweep['gap_ratio']:.2f} and "
             f"p_closed {sweep['p_closed']:.3f} on its own template set. Points redrawn from "
             "results/positive_control.json.",
             ha="center", va="bottom", fontsize=8.5, color=INK2)

    fig.tight_layout(rect=(0, 0.14, 1, 1))
    fig.savefig(out_path, dpi=140, facecolor=SURFACE)
    plt.close(fig)
    return out_path, circ, p_closed


def main():
    figs = ROOT / "figures"
    figs.mkdir(exist_ok=True)
    rows = json.loads((ROOT / "results" / "closure_depth.json").read_text())
    print("wrote", closure_depth_figure(rows, figs / "closure_depth.png"))

    pc = json.loads((ROOT / "results" / "positive_control.json").read_text())
    r = next(x for x in pc if x["layer"] == 21 and x["representation"] == "raw")
    xy, _, _ = pca_2d(np.asarray(r["points"]))
    sc = cyclic_order_score(angles(xy, *fit_circle(xy)[:2]), np.arange(len(DAYS)),
                            len(DAYS), n_perm=20000, seed=21)
    sweep = next(x for x in rows if x["ontology"] == "weekdays" and x["layer"] == 21)
    out, circ, p_closed = arc_figure(
        xy, [d[:3] for d in DAYS], "weekday", 21, sweep,
        figs / "weekday_circle_layer21.png",
        rho=sc["rho"], p_order=sc["p"], shuffled_rho=r["shuffled_rho"],
        null_p95=r["null_p95"])
    print("wrote", out)
    print(f"  recomputed from stored points: rho={sc['rho']:.3f} p_order={sc['p']:.4f} "
          f"gap_ratio={circ['gap_ratio']:.2f} max_gap={circ['max_gap_deg']:.1f} "
          f"p_closed={p_closed:.4f}")
    print(f"  sweep (different templates):   gap_ratio={sweep['gap_ratio']:.2f} "
          f"max_gap={sweep['max_gap_deg']:.1f} p_closed={sweep['p_closed']:.4f}")


if __name__ == "__main__":
    main()
