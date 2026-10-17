"""e63r -- post-verdict descriptive on D2 (label-free; changes nothing in the verdict).

Identity diagnostics at K = 128, comparable to the P2 Step-A table: the v1 representation (B0),
iterated first moments (B1) and the frozen per-type whitening (v2). Each gives the median
self-excluded TDI, the v1 self-inclusive TDI and KIR."""
import importlib.util, json, sys
from pathlib import Path
import numpy as np

EXT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("e63", EXT / "scripts/e63_occurrence_v2.py")
e63 = importlib.util.module_from_spec(_spec); sys.modules["e63"] = e63; _spec.loader.exec_module(e63)

choice = json.loads(e63.CHOICE.read_text())
out = dict(set="occD2", note="post-verdict, label-free descriptive", layers={})
for L in (7, 12, 24):
    wh = choice["whitening"][str(L)]
    rec = {}
    B = e63.represent_base("occD2", L, n_iter=1)
    W0, _ = e63.pca_white(B["V"]); del B["V"]
    D = e63.represent_base("occD2", L)
    Z, _ = e63.pca_white(D["V"]); del D["V"]
    W = e63.type_whiten(Z, D["tinv"], wh["N_MIN"], wh["n0"])[0]
    for name, X, tinv in (("B0_v1", W0, B["tinv"]), ("B1_iter", Z, D["tinv"]), ("v2_frozen", W, D["tinv"])):
        lab = e63.fit_km(X, 128, 0).labels_.astype(np.int64)
        rec[name] = dict(tdi_med=float(np.nanmedian(e63.tdi(lab, 128, tinv))),
                         tdi_self_incl_med=float(np.nanmedian(e63.tdi(lab, 128, tinv, 0, False))),
                         kir=e63.kir(X, tinv))
    out["layers"][str(L)] = rec
    print(L, json.dumps(rec), flush=True)
(EXT / "results/e63_d2_identity_descriptives.json").write_text(json.dumps(out, indent=1))
