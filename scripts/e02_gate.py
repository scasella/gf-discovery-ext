"""e02 -- Stage-1 reproduction gate + supervised per-layer controls on the rebuilt type means.

GATE (BUDGET.md section 6.2, declared before pass 1 ran):
  A. strict-12 months, RAW type means (35b part-A representation, pre-drop), gap_ratio within
     +-0.01 of phase4 prereg section 0: L4 1.585, L5 1.475, L6 1.521, L7 1.611, L8 2.012, L9 1.766.
  B. scripts/22 SAE-channel PRE-drop gap_ratio of the frozen month flag sets within +-1e-3 of
     42's ANCHOR_GAP_RATIO: L6 13-set 1.6588, L7 14-set 1.6202.
  Either failing -> the rebuilt pipeline is not the one the anchors describe: STOP.

SUPERVISED REPORT (reported, NOT gating -- BUDGET.md 6.3): for every layer in the type-mean
artifact, in both representations (raw, SAE-decoded), strict-12 months, the 7 weekdays and the
ordinal control one..ten:
  gap_ratio / max_gap / radius_ratio / p_closed (scripts/22 screen, 20,000-draw uniform null),
  p_bg against 2,000 random same-size pool subsets (scripts/22 background_for_size), and the
  cyclic-order rho at the label's TRUE cycle positions (fits.cyclic_order_score).
"""
import importlib.util, json, os, sys, time
from pathlib import Path

import numpy as np

EXT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXT / "src"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")


def _imp(name, fn):
    spec = importlib.util.spec_from_file_location(name, EXT / "scripts" / fn)
    m = importlib.util.module_from_spec(spec); sys.modules[name] = m
    spec.loader.exec_module(m); return m


S22 = _imp("s22", "22_token_type_geometry.py")
from gfdisc.tokens import DAYS, MONTHS                           # noqa: E402

RAW_ANCHOR = {4: 1.585, 5: 1.475, 6: 1.521, 7: 1.611, 8: 2.012, 9: 1.766}
RAW_TOL = 0.01
SAE_ANCHOR = {6: 1.6588, 7: 1.6202}
SAE_TOL = 1e-3
MEMBERS_13 = [3353, 5142, 5261, 5365, 5505, 5656, 5729, 5857, 6027, 6198, 6243, 6384, 7250]
MEMBERS_14 = sorted(MEMBERS_13 + [30651])
NUMS = [" one", " two", " three", " four", " five", " six", " seven", " eight", " nine", " ten"]
N_BG = 2000


def main():
    t0 = time.time()
    tm = np.load(EXT / "results/typemeans_pass1.npz")
    pool_ids = tm["pool_ids"].astype(np.int64)
    counts = tm["counts"].astype(np.int64)
    pos = {int(t): i for i, t in enumerate(pool_ids)}
    cnt_of = {int(t): int(c) for t, c in zip(pool_ids, counts)}
    token_map = json.loads((EXT / "results/token_map.json").read_text())

    def best_variant(ids):
        ip = [int(x) for x in ids if int(x) in pos]
        return max(ip, key=lambda x: cnt_of[x]) if ip else None

    strict12 = [best_variant(token_map["months"][m]) for m in MONTHS]
    days7 = [best_variant(token_map["days"][d]) for d in DAYS]
    from transformers import AutoTokenizer
    snap = sorted((EXT / "hf_home/hub/models--google--gemma-3-1b-pt/snapshots").glob("*"))[0]
    tok = AutoTokenizer.from_pretrained(str(snap))
    nums = []
    for w in NUMS:
        e = tok.encode(w, add_special_tokens=False)
        nums.append(int(e[0]) if len(e) == 1 and int(e[0]) in pos else None)
    sets = dict(months=(strict12, 12, True), weekdays=(days7, 7, True),
                numbers_one_to_ten=(nums, 10, False))
    out = dict(script="scripts/e02_gate.py", started=time.strftime("%Y-%m-%dT%H:%M:%S"),
               label_ids={k: v[0] for k, v in sets.items()},
               label_strings={k: [None if t is None else tok.decode([t]) for t in v[0]]
                              for k, v in sets.items()})

    # ---------------- gate ----------------
    gate = dict(raw={}, sae={})
    P12 = np.asarray([pos[t] for t in strict12])
    for L, want in RAW_ANCHOR.items():
        g = S22.screen(tm[f"raw_L{L}"][P12], 2000)["gap_ratio"]
        gate["raw"][L] = dict(observed=round(g, 4), anchor=want, ok=bool(abs(g - want) <= RAW_TOL))
    for L, want in SAE_ANCHOR.items():
        M = MEMBERS_13 if L == 6 else MEMBERS_14
        g = S22.screen(tm[f"sae_L{L}"][np.asarray(sorted(pos[t] for t in M))], 2000)["gap_ratio"]
        gate["sae"][L] = dict(observed=round(g, 5), anchor=want, ok=bool(abs(g - want) <= SAE_TOL))
    gate["PASS"] = bool(all(v["ok"] for v in gate["raw"].values())
                        and all(v["ok"] for v in gate["sae"].values()))
    out["gate"] = gate
    print(json.dumps(gate, indent=1), flush=True)

    # ---------------- supervised per-layer report ----------------
    rep = []
    for rep_name, layers in (("raw", [int(x) for x in tm["raw_layers"]]),
                             ("sae", [int(x) for x in tm["sae_layers"]])):
        for L in layers:
            R = tm[f"{rep_name}_L{L}"].astype(np.float64)
            for key, (ids, full_k, cyclic) in sets.items():
                have = [(i, pos[t]) for i, t in enumerate(ids) if t is not None]
                li = [i for i, _ in have]
                P = np.asarray([p for _, p in have])
                s = S22.screen(R[P], S22.N_PERM_CLUSTER)
                bg = S22.background_for_size(R, len(pool_ids), len(P),
                                             seed=900_000 + 100 * L + len(P), n_draws=N_BG)
                xy, _, _ = S22.pca_2d(R[P])
                o = (S22.score_ordering(xy, list(range(len(P))), li, full_k, seed=0)
                     if cyclic else None)
                row = dict(rep=rep_name, layer=L, set=key, k=len(P),
                           gap_ratio=round(s["gap_ratio"], 4),
                           max_gap_deg=round(s["max_gap_deg"], 2),
                           radius_ratio=round(s["radius_ratio"], 3),
                           p_closed=s["p_closed"], p_bg=S22.p_bg_of(bg, s["max_gap_deg"]),
                           bg_rate_p_closed_lt_05=bg["rate_p_closed_lt_05"],
                           closes=bool(s["p_closed"] < 0.05 and S22.p_bg_of(bg, s["max_gap_deg"]) <= 0.05),
                           order_rho=None if o is None else o.get("rho"),
                           order_p=None if o is None else o.get("p_order"))
                rep.append(row)
                print(f"  {rep_name} L{L:<2} {key:<20} gap {row['gap_ratio']:.3f} "
                      f"p_closed {row['p_closed']:.4f} p_bg {row['p_bg']:.4f} "
                      f"closes={row['closes']} rho {row['order_rho']}", flush=True)
    out["supervised"] = rep
    out["seconds"] = round(time.time() - t0, 1)
    (EXT / "results/e02_gate.json").write_text(json.dumps(out, indent=1, default=float))
    print("GATE", "PASS" if gate["PASS"] else "FAIL", flush=True)


if __name__ == "__main__":
    main()
