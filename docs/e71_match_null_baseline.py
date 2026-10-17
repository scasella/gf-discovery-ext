"""e71 post hoc (descriptive; not in the e71 rubric): e68's behavioural match counted per unit and compared with
the 40 joint-null random directions, which went through the identical e68 procedure. Reads only hashed e68 files.
Output: results/e71_match_null.json.
Revision 2 (advisor review): adds the fixed-size set (per dictionary top-5 |cos| U top-5 |r|), because the full tested set
grows with the OMP atoms (k08), which confounds the match rate with sparsity."""
import json, collections
from pathlib import Path
import numpy as np
EXT = Path(__file__).resolve().parents[1]
cen = json.loads((EXT / "results/e68_census.json").read_text())
out = {}
for L in (7, 12):
    a = json.loads((EXT / f"results/e68_acts_G{L}.json").read_text())["dicts"]
    dec = json.loads((EXT / f"results/e68_decoder_G{L}.json").read_text())["dicts"]
    role = {nm: u["role"] for nm, u in cen["groups"][str(L)]["units"].items()}
    names = list(next(iter(a.values()))["units"])
    nm_, nt, npos, mo = collections.Counter(), collections.Counter(), collections.Counter(), collections.defaultdict(float)
    for d in a.values():
        for nm, u in d["units"].items():
            for t in u["tested"].values():
                nt[nm] += 1; nm_[nm] += t["match"]; npos[nm] += t["match"] and t["side"] == "pos"; mo[nm] = max(mo[nm], t["overlap"])
    fx, fn = collections.Counter(), collections.Counter()           # fixed-size set: per dictionary top-5 |cos| U top-5 |r|
    for dn, d in a.items():
        for nm, u in d["units"].items():
            top_c = [t["latent"] for t in sorted(dec[dn]["units"][nm]["top"], key=lambda t: -abs(t["cos"]))[:5]]
            fixed = set(top_c) | {t["latent"] for t in u["r_top"][:5]}
            for li in fixed:
                fn[nm] += 1; fx[nm] += u["tested"][str(li)]["match"]
    nulls = [n for n in names if n.startswith("null/")]
    frate = {n: fx[n] / fn[n] for n in names}
    rate = {n: nm_[n] / nt[n] for n in names}
    q = lambda v: {k: float(np.quantile(v, x)) for k, x in (("min", 0), ("p5", .05), ("p50", .5), ("p95", .95), ("max", 1))}
    rec = dict(null=dict(matches=q([nm_[n] for n in nulls]), rate=q([rate[n] for n in nulls]), fixed_rate=q([frate[n] for n in nulls]), max_overlap=q([mo[n] for n in nulls]),
                         n_max_overlap_ge_0p95=int(sum(mo[n] >= 0.95 for n in nulls))), groups={}, units={})
    for grp in ("candidate", "context_named", "control"):
        g = [n for n in names if role.get(n) == grp]
        rec["groups"][grp] = dict(n=len(g), rate=q([rate[n] for n in g]), fixed_rate=q([frate[n] for n in g]),
                                  share_fixed_rate_above_null_max=float(np.mean([frate[n] > max(frate[x] for x in nulls) for n in g])), share_rate_above_null_max=float(np.mean([rate[n] > max(rate[x] for x in nulls) for n in g])))
    for n in names:
        if n in nulls:
            continue
        rec["units"][n] = dict(role=role.get(n), matches=nm_[n], heavy_side=npos[n], tested=nt[n], rate=rate[n],
                               nulls_ge_rate=int(sum(rate[x] >= rate[n] for x in nulls)),
                               fixed_matches=fx[n], fixed_n=fn[n], fixed_rate=frate[n], nulls_ge_fixed_rate=int(sum(frate[x] >= frate[n] for x in nulls)), nulls_ge_matches=int(sum(nm_[x] >= nm_[n] for x in nulls)),
                               max_overlap=mo[n], nulls_ge_max_overlap=int(sum(mo[x] >= mo[n] for x in nulls)))
    out[str(L)] = rec
(EXT / "results/e71_match_null.json").write_text(json.dumps(out, indent=1))
print(json.dumps({L: dict(null_fixed_rate=v["null"]["fixed_rate"], null_rate=v["null"]["rate"], null_matches=v["null"]["matches"], n_null_overlap_ge_0p95=v["null"]["n_max_overlap_ge_0p95"]) for L, v in out.items()}, indent=1))
