"""e71 post hoc (descriptive): where the 40 L12 joint-null directions fall against each dictionary's per-mode
reconstruction-null k08 quantiles (e68_decoder_G12 null_recon), cell by cell and per direction.
Output: results/e71_null_band_check.json."""
import json
from pathlib import Path
EXT = Path(__file__).resolve().parents[1]
d = json.loads((EXT / "results/e68_decoder_G12.json").read_text())["dicts"]
out = {}
for m in ("signed", "nonneg"):
    cells = above = le_p1 = le_p5 = 0; perdir = []
    for i in range(40):
        h = 0
        for r in d.values():
            q = r["null_recon"][m]["k08"]; k = r["units"][f"null/{i}"]["omp"][m]["k08"]; k = r["k_max"] + 1 if k is None else k
            cells += 1; above += k > q["p50"]; le_p1 += k <= q["p1"]; le_p5 += k <= q["p5"]; h += k > q["p50"]
        perdir.append(h)
    out[m] = dict(cells=cells, above_p50=above, share_above_p50=above / cells, at_or_below_p1=le_p1, at_or_below_p5=le_p5,
                  per_direction_n_dicts_above_p50=sorted(perdir), n_directions_all_14_above=sum(x == 14 for x in perdir),
                  n_directions_none_above=sum(x == 0 for x in perdir))
(EXT / "results/e71_null_band_check.json").write_text(json.dumps(out, indent=1))
print(json.dumps({m: {k: v for k, v in o.items() if k != "per_direction_n_dicts_above_p50"} for m, o in out.items()}, indent=1))
