"""e73 post hoc: re-read the hashed calibration outputs under alternative class bars (POST HOC, descriptive; amends nothing).

Written 2026-10-17 after the e73 verdict (every test NOT ADMITTED), at the user's request, to decide whether a frozen
bar-sensitivity run on an unused block is worth spending one. It reads only the hashed e73 outputs
(results/e73_calib_l0_{big,small}.{json,npz}) and applies e73's own judge() unchanged; only the class bars differ.

The variant grid is fixed here, before any variant is computed (16 variants):
  ADDS bar      max (e73) | p99.9 | p99 | p95 of the bar bank (T-MG's bank has 200 directions, so its p99.9 is about its max)
  bar source    per condition (e73) | fixed at s0 for every ablated condition (ladder, matched, size-matched);
                plant conditions always use their own bank, because the plant changes the rows themselves
  COVERED bar   at or below the median (e73) | at or below p75
The e73 variant (max, per condition, median) must reproduce results/e73_coverage_admission.json exactly (asserted).
Any variant that admits a test is a candidate for a NEW frozen NOTE on an unused block; it is not a result here.
Output: results/e73_posthoc_bars.json (not part of any verdict).
"""
import copy, importlib.util, json, sys
from pathlib import Path

import numpy as np

EXT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("e73", EXT / "scripts/e73_coverage_admission.py")
e73 = importlib.util.module_from_spec(spec); sys.modules["e73"] = e73; spec.loader.exec_module(e73)

ADDS_Q = {"max": None, "p99.9": 0.999, "p99": 0.99, "p95": 0.95}
SOURCES = ("per-condition", "fixed-s0")
COVER_Q = {"median": 0.5, "p75": 0.75}


def npz_key(cname, test, part):
    return f"{cname}|{test}|{part}".replace("|", "__").replace(":", "_")


def bars(rec, z, cname, test, aq, cq):
    """(covered_bar, adds_bar) of one condition's bank; median and max from the JSON (float64), other quantiles from the npz."""
    c = rec["conditions"][cname][test]
    bank = z[npz_key(cname, test, "bank")].astype(np.float64)
    cov = c["bank_med"] if cq == 0.5 else float(np.quantile(bank, cq))
    add = c["bank_max"] if aq is None else float(np.quantile(bank, aq))
    return cov, add


def klass(v, cov, add):
    return "A" if v > add else ("C" if v <= cov else "U")


def reclass(rec, z, aq, src, cq):
    out = copy.deepcopy(rec)
    for cname, cond in out["conditions"].items():
        for test in e73.TESTS:
            src_c = "s0" if (src == "fixed-s0" and cond.get("kind") in ("ladder", "match", "msize")) else cname
            cov, add = bars(rec, z, src_c, test, aq, cq)
            t = cond[test]
            for tn, tv in t["targets"].items():
                tv["cls"] = klass(tv["value"], cov, add)
            s40 = z[npz_key(cname, test, "s40")].astype(np.float64)
            t["s40"] = "".join(klass(v, cov, add) for v in s40)
    return out


def main():
    recs = {l0: json.loads((EXT / f"results/e73_calib_{l0}.json").read_text()) for l0, _ in e73.DICTS}
    zs = {l0: np.load(EXT / f"results/e73_calib_{l0}.npz") for l0, _ in e73.DICTS}
    frozen = json.loads((EXT / "results/e73_coverage_admission.json").read_text())
    rows = []
    for an, aq in ADDS_Q.items():
        for src in SOURCES:
            for cn, cq in COVER_Q.items():
                res, broken = e73.verdict_from({l0: reclass(recs[l0], zs[l0], aq, src, cq) for l0 in recs})
                row = dict(adds=an, source=src, covered=cn, rule_set_broken=broken,
                           levels={l0: {x: res[l0][x]["level"] for x in e73.TESTS} for l0 in res},
                           first_failure={l0: {x: res[l0][x]["first_failure"] for x in e73.TESTS} for l0 in res},
                           before_guard={l0: {x: res[l0][x].get("level_before_guard") for x in e73.TESTS} for l0 in res},
                           mde={l0: {x: res[l0][x].get("mde_kappa") for x in e73.TESTS} for l0 in res},
                           s_star={l0: {x: {c: res[l0][x]["controls"][c]["s_star"] for c in e73.CTRL} for x in e73.TESTS} for l0 in res},
                           rule5_max_adds_moves={l0: {x: res[l0][x]["rule5_max_adds_moves"] for x in e73.TESTS} for l0 in res})
                if (an, src, cn) == ("max", "per-condition", "median"):
                    for l0 in res:
                        for x in e73.TESTS:
                            assert res[l0][x]["level"] == frozen["results"][l0][x]["level"], (l0, x)
                            assert res[l0][x]["first_failure"] == frozen["results"][l0][x]["first_failure"], (l0, x)
                    row["reproduces_e73"] = True
                rows.append(row)
    out = dict(status="POST HOC, descriptive; amends nothing; not a verdict", inputs={
        f"results/e73_calib_{l0}.{ext}": e73.sha(EXT / f"results/e73_calib_{l0}.{ext}") for l0 in recs for ext in ("json", "npz")},
        script_sha256=e73.sha(Path(__file__)), variants=rows)
    (EXT / "results/e73_posthoc_bars.json").write_text(json.dumps(out, indent=1))
    print(f"{'ADDS':6} {'source':14} {'COVERED':7} | {'l0_big T-ES / T-MG / T-COS':60} | l0_small T-ES / T-MG / T-COS")
    short = {"NOT ADMITTED": "NO", "ADMITTED-GRADED": "GRADED", "ADMITTED": "ADMITTED"}
    for r in rows:
        cells = []
        for l0 in ("l0_big", "l0_small"):
            cells.append(" / ".join(f"{short[r['levels'][l0][x]]}" + (f"({r['first_failure'][l0][x].replace('rule', 'r').replace(' (MDE undefined or > 4)', '')})"
                                                                         if r['levels'][l0][x] == 'NOT ADMITTED' else "") for x in e73.TESTS))
        print(f"{r['adds']:6} {r['source']:14} {r['covered']:7} | {cells[0]:60} | {cells[1]}" + ("  [GUARD]" if r["rule_set_broken"] else ""))


if __name__ == "__main__":
    main()
