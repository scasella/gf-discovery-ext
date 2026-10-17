"""Step 22b -- pure re-summarisation of `results/window_background.json`.

No new draws, no new permutations, no randomness: this reads the draws already
stored by `scripts/22_window_background.py` and places EVERY scorable observed
cluster of each cell inside its own matched background, not just the ordering-FDR
passers. It exists because the PRIMARY cell (layer 9, months, W=32, arm
even_build -- the declared fallback) has no FDR passer at all, so without this the
primary background has no observed cluster placed in it and the headline rates
cannot be read against anything.

`passers` is left untouched. The result is written back into the same file under
`observed_vs_background`, tagged `post_hoc_resummarisation: true`.

Direction conventions are the ones already used for passers:
  rho        higher is better -> percentile = fraction of scorable draws BELOW
  gap_ratio  lower is better  -> percentile = fraction of scorable draws ABOVE
  p_order    lower is better  -> tail_frac_le = fraction of draws at least as small
"""
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
BG = REPO / "results/window_background.json"
GATE = {L: json.loads((REPO / f"results/window_gate_L{L}.json").read_text())
        for L in [5, 6, 7, 8, 9, 10]}


def high_good(vals, x):
    n = len(vals)
    if not n:
        return dict(n=0, percentile=None, tail_frac_ge=None)
    return dict(n=n, percentile=sum(v < x for v in vals) / n,
                tail_frac_ge=sum(v >= x for v in vals) / n)


def low_good(vals, x):
    n = len(vals)
    if not n:
        return dict(n=0, percentile=None, tail_frac_le=None)
    return dict(n=n, percentile=sum(v > x for v in vals) / n,
                tail_frac_le=sum(v <= x for v in vals) / n)


def main():
    doc = json.loads(BG.read_text())
    for key, c in doc["cells"].items():
        rows = GATE[c["layer"]]["runs"][c["ontology"]]["cells"][c["cell"]]["rows"]
        sc = [d for d in c["draws"] if d["scorable"]]
        out = []
        for idx, r in enumerate(rows):
            if not r["scorable"]:
                continue
            same = [d for d in sc if d["tpl"] == idx]
            out.append(dict(
                cluster_id=r["cluster_id"], k=r["size"], template_index=idx,
                n_labels=r["n_labels"], rho=r["rho"], p_order=r["p_order"],
                gap_ratio=r["gap_ratio"], p_closed=r["p_closed"],
                ordering_fdr=bool(r.get("ordering_fdr")),
                CIRCLE=bool(r.get("CIRCLE")),
                rho_vs_bg=high_good([d["rho"] for d in sc], r["rho"]),
                gap_ratio_vs_bg=low_good([d["gap_ratio"] for d in sc], r["gap_ratio"]),
                p_order_vs_bg=low_good([d["p_order"] for d in sc], r["p_order"]),
                rho_vs_bg_same_template=high_good([d["rho"] for d in same], r["rho"]),
                gap_ratio_vs_bg_same_template=low_good(
                    [d["gap_ratio"] for d in same], r["gap_ratio"]),
                p_order_vs_bg_same_template=low_good(
                    [d["p_order"] for d in same], r["p_order"])))
        c["observed_vs_background"] = out
    doc["observed_vs_background_note"] = (
        "post_hoc_resummarisation: every SCORABLE observed cluster of each cell "
        "placed inside that cell's stored background draws. No new draws, no new "
        "permutations, no randomness -- it re-reads results/window_gate_L*.json and "
        "the draws already in this file. Added by scripts/22b_window_background_"
        "context.py because the primary cell has no ordering-FDR passer and would "
        "otherwise have no observed cluster placed in its own background.")
    doc["post_hoc_resummarisation"] = True
    BG.write_text(json.dumps(doc, indent=2))
    print(f"wrote {BG}")


if __name__ == "__main__":
    main()
