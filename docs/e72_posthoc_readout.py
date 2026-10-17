"""e72 post hoc readout (descriptive, not in the NOTE): read from the hashed e72 coverage/verify JSONs only, no new compute.
(1) does error share track non-Gaussianity across the units? (2) the decoded-span match counted against its own random
null instead of the NOTE's 0.8 bar; (3) D2 vs S2 above-max counts side by side (S2 = the selection-matched comparison).
usage: e72_posthoc_readout.py OUT.json"""
import json, sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

EXT = Path(__file__).resolve().parents[1]
cov = json.loads((EXT / "results/e72_coverage.json").read_text())
ver = json.loads((EXT / "results/e72_verify.json").read_text())
U = cov["units"]; scr = [u for u in U if u["screened"]]
out = {}
for name, us in (("all_128", U), ("screened_60", scr)):
    es = np.asarray([u["es"] for u in us])
    out[f"es_vs_abs_ex_kurtosis_{name}"] = float(spearmanr(es, np.abs([u["ex_kurtosis"] for u in us])).statistic)
    out[f"es_vs_abs_skew_{name}"] = float(spearmanr(es, np.abs([u["skew"] for u in us])).statistic)
cn = cov["cos_null"]
mc = np.asarray([u["e64_match"]["max_cos"] for u in U])
out["e64_match_counts"] = dict(ge_0p8_note_bar=int((mc >= 0.8).sum()), gt_null_p999=int((mc > cn["p999"]).sum()),
                               gt_null_max=int((mc > cn["max"]).sum()), null=cn)
out["e64_match_above_null_max"] = sorted([(u["kind"], u["id"], round(u["e64_match"]["max_cos"], 3), u["e64_match"]["unit"],
                                           round(u["es"], 3)) for u in U if u["e64_match"]["max_cos"] > cn["max"]], key=lambda x: -x[2])
out["above_random_max"] = dict(
    D2=dict(all=cov["counts"]["above_max"], screened=sum(u["beats"]["max"] and u["screened"] for u in U)),
    S2={k: dict(all=v.get("n_above_max"), screened=v.get("n_screened_above_max"), n_screened=v.get("n_screened"))
        for k, v in cov["s2"]["kinds"].items()})
out["lead_routes"] = {k: {x: v[x] for x in ("es_coverage", "es_v1", "v2_pct", "es_ls", "ls_r2", "ls_pct", "ls_beats_max")}
                      for k, v in ver["units"].items()}
out["v3_random"] = ver["v3_random"]
Path(sys.argv[1]).write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))
