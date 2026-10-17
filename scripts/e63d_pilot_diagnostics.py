"""e63d -- P2-only diagnostics after the e63 pilot, before PREREG_occurrence_v2.md is frozen.

1. Ablation (paren/quote max |AUROC| over ICA-64 logcosh units, unscreened and screened):
     B1 (iterated first moments, no per-type step) vs the chosen per-type whitening.
   Separates the second-moment step from the ICA-dimension fix (addendum A1).
2. Specificity: for the best screened paren and quote ICA units at the chosen setting, the
   association with every battery feature (|AUROC| for binary, |Spearman| for continuous).
Reported only; changes nothing in the pilot choice."""
import importlib.util, json, sys, time
from pathlib import Path
import numpy as np

EXT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("e63", EXT / "scripts/e63_occurrence_v2.py")
e63 = importlib.util.module_from_spec(_spec); sys.modules["e63"] = e63; _spec.loader.exec_module(e63)
e61 = e63.e61
from scipy.stats import spearmanr

choice = json.loads(e63.CHOICE.read_text())
tk = e61.token_strings()
full = np.load(EXT / "cache/occ/occP2_meta.npz")["tokens"]
out = dict(set="occP2", layers={})
for L in (7, 12, 24):
    t0 = time.time()
    wh = choice["whitening"][str(L)]
    D = e63.represent_base("occP2", L)
    feats = e61.feature_battery(D, tk, full, None)
    Z, _ = e63.pca_white(D["V"]); del D["V"]
    W = e63.type_whiten(Z, D["tinv"], wh["N_MIN"], wh["n0"])[0]
    rec = dict(ablation={}, specificity={})
    for name, X in (("B1_no_type_whitening", Z), ("chosen_type_whitening", W)):
        for contrast in ("logcosh", "skew"):
            Y, _, units = e63.ica_units(X, D, 64, contrast)
            scr = [u["id"] for u in units if e63.ica_pass(u, 0.8, False)]
            if not units:
                rec["ablation"][f"{name}/{contrast}"] = dict(failed="ICA non-finite on all 3 seeds")
                continue
            r = dict(n_screened=len(scr), repro_med=float(np.median([u["repro"] for u in units])),
                     eta2_max=float(max(u["eta2_token"] for u in units)),
                     eta2_sq=[round(u["eta2_token_sq"], 4) for u in units],
                     eta2_abs=[round(u["eta2_token_abs"], 4) for u in units])
            for lb in e63.LABELS:
                a = np.asarray([e61.auroc(Y[:, j], feats[lb]) for j in range(64)]); a = np.maximum(a, 1 - a)
                r[lb] = dict(max_all=float(a.max()), max_screened=float(a[scr].max()) if scr else None,
                             best_screened=int(scr[int(np.argmax(a[scr]))]) if scr else None,
                             auroc_all=[round(float(x), 4) for x in a])
            rec["ablation"][f"{name}/{contrast}"] = r
            if name == "chosen_type_whitening":
                for lb in e63.LABELS:
                    j = r[lb]["best_screened"]
                    if j is None:
                        continue
                    y = Y[:, j]; assoc = {}
                    for k, v in feats.items():
                        if set(np.unique(v)) <= {0.0, 1.0}:
                            a = e61.auroc(y, v); assoc[k] = float(max(a, 1 - a))
                        else:
                            assoc[k] = float(abs(spearmanr(y[::3], v[::3]).statistic))
                    rec["specificity"][f"{contrast}/best_{lb}_unit{j}"] = dict(
                        sorted(assoc.items(), key=lambda kv: -kv[1]))
    out["layers"][str(L)] = rec
    print(f"L{L} {time.time()-t0:.0f}s", json.dumps(rec["ablation"])[:900], flush=True)
    (EXT / "results/e63_pilot_diagnostics.json").write_text(json.dumps(out, indent=1))
