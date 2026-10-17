"""Synthetic smoke test of e68's measurement code: fake dictionaries, fake residuals, CPU. No real latent number."""
import importlib.util, sys, time
import numpy as np
EXT = "/Users/scasella/Downloads/gf-discovery-ext"
spec = importlib.util.spec_from_file_location("e68", f"{EXT}/scripts/e68_coverage_census.py")
m = importlib.util.module_from_spec(spec); sys.modules["e68"] = m; spec.loader.exec_module(m)
e67, e66, e65, e64 = m.mods()
rng = np.random.default_rng(0)
L = 12
nseq = 120; toks_all = rng.integers(0, 3000, size=nseq * 255).astype(np.int64)
rows = np.sort(rng.choice(nseq * 255, size=20000, replace=False)).astype(np.int64)
H = rng.standard_normal((len(rows), 1152)).astype(np.float32)
nl = 400
W = rng.standard_normal((1152, nl)).astype(np.float32) / 30
# plant: latent 7 tracks unit A, latent 11 is anti-correlated with unit B
dirA = rng.standard_normal(1152); dirA /= np.linalg.norm(dirA)
dirB = rng.standard_normal(1152); dirB /= np.linalg.norm(dirB)
W[:, 7] = dirA.astype(np.float32); W[:, 11] = dirB.astype(np.float32)
Pd = dict(w_enc=W, b_enc=np.zeros(nl, np.float32), threshold=np.full(nl, 0.5, np.float32),
          w_dec=np.ascontiguousarray(W.T))
z = np.load(f"{EXT}/results/e64_maps_L12.npz")
keep = np.ones(1152, bool); keep[1038] = False
names = ["unitA", "unitB"] + [f"null/{i}" for i in range(3)]
sc = {"unitA": H @ dirA, "unitB": -(H @ dirB)}
for i in range(3):
    sc[f"null/{i}"] = H @ rng.standard_normal(1152)
D = {"unitA": dirA, "unitB": -dirB, **{f"null/{i}": rng.standard_normal(1152) for i in range(3)}}
for k in D: D[k] = D[k] / np.linalg.norm(D[k])
nxt_tab = e65.next_by_type(toks_all)
valid = (rows % 255) != 254
nxt = np.full(len(rows), -1, np.int64); nxt[valid] = toks_all[rows[valid] + 1]
k1 = int(np.ceil(e64.TAIL * len(rows)))
prepu = {}
for nm in names:
    rec = {}
    for side, order in (("pos", np.argsort(-sc[nm])), ("neg", np.argsort(sc[nm]))):
        sig = e64.signature(rows, order[:k1], toks_all, nxt_tab)
        inT = np.isin(nxt, sig) & valid
        top = [i for i in order[:4 * m.MATCH_TOP] if valid[i]][:m.MATCH_TOP]
        rec[side] = dict(T=[int(t) for t in sig], s_self=float(inT[top].mean()), base=float(inT[valid].mean()))
        rec[side]["testable"] = bool(len(sig) > 0 and rec[side]["s_self"] >= m.TESTABLE_RATIO * rec[side]["base"])
    rec["testable_sig"] = rec["pos"]["testable"]; rec["role"] = "candidate"
    prepu[nm] = rec
m.N_SUBNULL, m.N_RECON_NULL = 50, 8
t0 = time.time()
drec = m.decoder_dict(e67, e65, ("fake", L, "16k", "x"), keep, z, L, D, Pw=Pd)
print("decoder ok", round(time.time() - t0, 1), "s; null", drec["null_subspace"], drec["null_recon"]["signed"])
print(" unitA top", drec["units"]["unitA"]["top"][:2], "omp", {k: (v["k08"], v["atoms08"][:3]) for k, v in drec["units"]["unitA"]["omp"].items()})
m.MATCH_MIN_SEQ = 10
ctx = m.make_ctx(e64, e65, toks_all, rows, names, sc, prepu, L)
t0 = time.time()
arec = m.acts_dict(Pd, H, "cpu", ctx, names, drec)
print("acts ok", round(time.time() - t0, 1), "s; pool", arec["n_random_null"], "/", arec["n_eligible"])
for nm in ("unitA", "unitB"):
    u = arec["units"][nm]
    print(nm, "r_top", u["r_top"][:2], "max_abs_r", round(u["max_abs_r"], 3))
    for li in ("7", "11"):
        if li in u["tested"]:
            t = u["tested"][li]; print("  latent", li, {k: t[k] for k in ("r", "side", "n_rows", "n_seq", "sig_match", "overlap", "overlap_threshold", "overlap_null_p99", "overlap_match", "typecond_ratio", "match")})
dec = dict(dicts={"fake/L12_16k_x": drec}); act = dict(dicts={"fake/L12_16k_x": arec})
for nm in names:
    v, hits, part = m.unit_verdict(nm, L, dec, act, list(dec["dicts"]), True)
    print("verdict", nm, v, [(h["latent"], h["via"], h["by"]) for h in hits][:3], len(part))
# JSON round-trip of what stage_run / stage_extended dump
import json
json.loads(json.dumps(dict(decoder=dec, acts=act), default=lambda o: o.item() if hasattr(o, "item") else str(o)))
print("json round-trip ok")
