"""Synthetic end-to-end smoke test of e70's geo and rep phases: fake D3/S3/R3 files, fake SAE, no fresh text read."""
import importlib.util, sys, json, time
import numpy as np
EXT = "/Users/scasella/Downloads/gf-discovery-ext"
FAKE = sys.argv[1]
spec = importlib.util.spec_from_file_location("e70", f"{EXT}/scripts/e70_sae_gap_geometric.py")
m = importlib.util.module_from_spec(spec); sys.modules["e70"] = m; spec.loader.exec_module(m)
m.T0 = time.time()
e69, e64 = m.mods()
m.OUT = lambda name: f"{FAKE}/e70_{name}"
e64.require_hashed = lambda *a: None
e64.append_hash = lambda rel, note: print("  (would hash)", rel.split("/")[-1])
e64.dump = lambda obj, rel: open(rel, "w").write(json.dumps(obj, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
m.KO_PERM, m.PLANT_PERM, m.N_CAP_NULL = 20, 5, 200
m.PLANT_KAPPA = (4.0,)
rng = np.random.default_rng(0)
nl = 10400
P = dict(w_enc=(rng.standard_normal((1152, nl)) / 30).astype(np.float32), b_enc=np.zeros(nl, np.float32), threshold=np.full(nl, 1.5, np.float32),
         w_dec=(rng.standard_normal((nl, 1152)) / 30).astype(np.float32), b_dec=np.zeros(1152, np.float32))
e69.sae_params = lambda d: P
z0 = np.load(f"{EXT}/results/e69_p0_units.npz")
def make(name, ids, shuffle=False, plant_unit=True):
    T = rng.integers(2, 3000, size=(len(ids), 255))
    meta = m.meta_of(T, np.asarray(ids)); rows = m.kept_rows(meta)
    S = e69.RowSet(dict(tokens=meta["tokens"], seq=meta["seq"], pos=meta["pos"]), rows)
    H = rng.standard_normal((len(rows), 1152)).astype(np.float32) * 3
    slow = np.zeros(len(rows))
    for i in range(1, len(rows)):
        slow[i] = (0.9 if not shuffle else 0.2) * slow[i - 1] * (S.seq[i] == S.seq[i - 1]) + rng.standard_normal()
    if plant_unit:
        d = np.zeros(1152); d[e69.KEEP] = z0["pat"][:, 1]; H += (4 * slow[:, None] * d[None, :]).astype(np.float32)
    H[:, 1038] = 70000.0 if name == "D3" else 60000.0          # D3: force fp16 overflow to exercise the sidecar
    side = H[:, 1038].copy()
    np.save(f"{FAKE}/e70_{name}_L12.npy", H.astype(np.float16))
    np.savez(f"{FAKE}/e70_{name}_1038.npz", L12=side.astype(np.float32))
    np.savez(f"{FAKE}/e70_{name}_meta.npz", rows=rows, dropped=np.array([], np.int64), overflow="{}", **meta)
    return meta, rows
mD, rD = make("D3", range(2000, 2080))
mS, rS = make("S3", range(2000, 2080), shuffle=True)
make("R3", range(3000, 3080))
n = len(mD["tokens"])
np.savez(f"{FAKE}/e70_D3_pred.npz", entropy=rng.random(n).astype(np.float32) * 3, logp=-rng.random(n).astype(np.float32) * 5)
np.savez(f"{FAKE}/e70_D3_rules_trf.npz", **{k: (rng.random(n) < 0.1).astype(np.uint8) for k in m.E64C_RULES})
open(f"{FAKE}/e70_D3_pieces.json", "w").write("[]")
m.phase_geo(e69, e64)
geo = json.load(open(f"{FAKE}/e70_geo.json"))
for u, r in geo["primary"].items():
    print("unit", u, "G1", round(r["G1"]["acf"]["1"], 3), "G2", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in r["G2"].items()},
          "G3", round(r["G3"]["capture"], 3), r["G3"]["pass_"], "named", r["G4"]["named"], "G5", round(r["G5"]["es_big"], 3), round(r["G5"]["es_small"], 3),
          "split resid", r["G5"]["split_residual"], "->", r["d3_verdict"])
print("C1", geo["controls"]["C1"]["recovered"], "C2", {k: (v["recovered"], round(v["absorption"], 3)) for k, v in geo["controls"]["C2_planted"].items()})
print("sets", geo["sets"]["log_e_inflation"], "r3_needed", geo["r3_needed"], "exploratory", list(geo["exploratory"]))
geo["r3_needed"] = True; geo["primary"]["1"]["d3_pass"] = True; open(f"{FAKE}/e70_geo.json", "w").write(json.dumps(geo))
m.phase_rep(e69, e64)
print("verdict", json.load(open(f"{FAKE}/e70_verdict.json"))["primary"])
print("X0 tokens", m.set_tokens("X0")[0].shape, "kept rows equal frozen:",
      np.array_equal(m.kept_rows(m.meta_of(*m.set_tokens("X0")[:2])), np.load(f"{EXT}/results/occ_units_v2_L12.npz")["row"]))
