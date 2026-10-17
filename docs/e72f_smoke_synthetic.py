"""e72f synthetic smoke test (no model weights, no corpus text, no SAE number; nothing written to results/).

1. Corpus layout: duplicate screen, block-F sites and prompts on a fake 15,696 x 255 array (BOS dropped).
2. make_forward on a tiny CPU torch model with e66's hook layout: unsteered and steered log-softmax and the
   sigma capture equal a direct computation.
3. The HTTP-range reader (used for the 262k rows) against the LOCAL 16k-big params file via file://: header,
   a tensor and scattered w_dec rows equal safetensors' own load.
4. The real steer engine and verdict on a fake linear model in constructed worlds, each with a known answer:
   FRAGMENTED (with the 262k branch), NOT FRAGMENTED, TILTED, SPLIT NOT REASSEMBLED, NO VERDICT (dead anchor),
   UNINFORMATIVE (d one-sided), INCONCLUSIVE for low power (planted arms carry no tilt), for a void band, and
   for co-primaries that disagree. Banks come from the script's own cos_bank on class-free pools.
5. A code check of the one-sided bound: an arm drawn exchangeably with its bank is called ONE-SIDED or TILTED at
   about 2/(M+1). (It cannot show miscalibration from a non-exchangeable bank; the covariate diagnostics in the
   freeze address that on real rows.)

Run: .venv/bin/python docs/e72f_smoke_synthetic.py
"""
import importlib.util, json, sys
from pathlib import Path

import numpy as np

EXT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("e72f", EXT / "scripts/e72f_fragmentation.py")
E = importlib.util.module_from_spec(spec); sys.modules["e72f"] = E; spec.loader.exec_module(E)
e66, e65, e64 = E.mods()
ok = []

# ---------------------------------------------------------------- 1. corpus layout
rng = np.random.default_rng(1)
Tc = rng.integers(5, 262000, size=(15696, 255)).astype(np.int32)
Tr = rng.integers(5, 262000, size=(8000, 255)).astype(np.int32)
Tc[8010, :32] = Tc[500, :32]; Tc[8020, :32] = Tr[7, :32]; Tc[8030, :32] = Tc[13500, :32]
kept, dropped = E.dup_screen(Tc, Tr, E.BLOCK_F["corpus"])
sites = E.draw_sites(kept, E.BLOCK_F["site_seed"] + E.L, e65.N_SITES)
P, st = E.build_prompts(Tc, sites, 2, e65.CTX)
sa = np.asarray(sites)
ok.append(("duplicate screen", dropped == [8010, 8020, 8030] and len(kept) == 997))
ok.append(("block-F sites", sa[:, 0].min() >= 8000 and sa[:, 0].max() <= 8999 and len(set(sa[:, 0])) == 200
           and not set(sa[:, 0]) & set(dropped) and sa[:, 1].min() >= 4 and sa[:, 1].max() <= 254))
ok.append(("prompts", all(x[0] == 2 and len(x) == min(p, 64) + 2 and x[-1] == Tc[s, p] and np.array_equal(x[1:], Tc[s, max(0, p - 64):p + 1])
                          for x, (s, p) in zip(P, sites))))

# ---------------------------------------------------------------- 2. make_forward on a tiny CPU model
import torch
import torch.nn as nn


class Ident(nn.Module):
    def forward(self, h):
        return h


class FakeLM(nn.Module):
    def __init__(self, V, dim):
        super().__init__()
        self.emb = nn.Embedding(V, dim)
        self.model = nn.Module(); self.model.layers = nn.ModuleList([Ident() for _ in range(26)])
        self.WU = nn.Parameter(torch.randn(V, dim) * 0.3)

    def forward(self, input_ids, attention_mask=None, use_cache=False):
        h = self.emb(input_ids)
        for lay in self.model.layers:
            h = lay(h)
        return type("O", (), {"logits": h @ self.WU.T})()


torch.manual_seed(0)
Vt, dim = 300, 32
lm = FakeLM(Vt, dim).eval()
tok = type("Tok", (), {"pad_token_id": 0})()
prompts = [rng.integers(1, Vt, size=int(n)) for n in rng.integers(3, 12, size=40)]
union = [3, 7, 11, 50, 299]
Dsel = rng.standard_normal((4, dim)); Dsel /= np.linalg.norm(Dsel, axis=1, keepdims=True)
fwd = E.make_forward(lm, tok, "cpu", prompts, union, Dsel)
cap = dict(proj=[], ids=[])
LP0, _ = fwd(None, 0.0, cap=cap)
with torch.no_grad():
    H = [lm.emb(torch.tensor(p)).numpy().astype(np.float64) for p in prompts]
WU = lm.WU.detach().numpy().astype(np.float64)
lsm = lambda x: x - np.log(np.exp(x - x.max(-1, keepdims=True)).sum(-1, keepdims=True)) - x.max(-1, keepdims=True)
direct0 = np.vstack([lsm(h[-1] @ WU.T)[union] for h in H])
v, alpha = Dsel[1], 2.5
LP1, rr = fwd(v, alpha)
direct1 = np.vstack([lsm((h[-1] + alpha * v) @ WU.T)[union] for h in H])
proj_direct = np.concatenate([h[e65.POS_MIN:] @ Dsel.T for h in H])
ok.append(("make_forward unsteered", np.allclose(LP0, direct0, atol=1e-4)))
ok.append(("make_forward steered", np.allclose(LP1, direct1, atol=1e-4) and abs(rr - 1.0) < 1e-4))
ok.append(("make_forward capture", np.allclose(np.concatenate(cap["proj"]), proj_direct, atol=1e-4)
           and len(np.concatenate(cap["ids"])) == sum(max(0, len(p) - e65.POS_MIN) for p in prompts)))

# ---------------------------------------------------------------- 3. range reader on the local 16k-big file
from safetensors.numpy import load_file
local = EXT / E.SAE_SNAP / E.DICTS["big"][0] / "params.safetensors"
url = "file://" + str(local)
h, off = E.read_header(url)
ref = load_file(str(local))
want = [11562, 320, 0, 16383, 5]
ok.append(("range reader", np.array_equal(E.read_tensor(url, "threshold", h, off), ref["threshold"])
           and np.array_equal(E.read_rows(url, h, off, want, n_rows=16384), ref["w_dec"][want])
           and np.array_equal(E.read_blocks(url, h, off, [0, 640], 64, n_rows=16384), np.vstack([ref["w_dec"][0:64], ref["w_dec"][640:704]]))))

# ---------------------------------------------------------------- 4. engine + verdict on a fake linear model
DIM, VOC, NS = 96, 300, 200
keep = np.ones(DIM, bool); keep[7] = False
classes = dict(S=dict(words=[f"s{i}" for i in range(9)], ids=list(range(10, 19))),
               P=dict(words=[f"p{i}" for i in range(21)], ids=list(range(20, 41))),
               AMB=dict(words=[f"a{i}" for i in range(9)], ids=list(range(50, 59))),
               TMP=dict(words=[f"t{i}" for i in range(10)], ids=list(range(60, 70))))
refs = dict(S=dict(ids=list(range(70, 79))), P=dict(ids=list(range(79, 100))))
A_T, A_R = list(range(100, 110)), list(range(200, 210))
sigs = {E.ANCHOR: dict(T=A_T, R=A_R), E.PRIMARY: dict(T=[10, 11, 12, 20, 21, 22], R=list(range(210, 216))),
        E.SECONDARY: dict(T=[13, 14, 23, 24, 25], R=list(range(216, 221)))}
for k in range(8):
    sigs[("logcosh", 40 + k)] = dict(T=list(rng.choice(range(120, 200), 8, replace=False)), R=list(rng.choice(range(221, 300), 8, replace=False)))
twin = {E.PRIMARY: E.SECONDARY, E.SECONDARY: E.PRIMARY}


def unit(x):
    x = np.asarray(x, float).copy(); x[~keep] = 0
    return x / np.linalg.norm(x)


def orth(x, *against):
    x = np.asarray(x, float).copy()
    for a in against:
        x = x - (x @ a) * a
    return unit(x)


def world(kind):
    r = np.random.default_rng(dict(frag=11, single=12, tilted=13, split=14, dead=15, oneside=16, lowpower=17, void=18, disagree=19,
                                   fragnear=11)[kind])
    Q = np.linalg.qr(r.standard_normal((DIM, DIM)))[0].T
    Q[:, ~keep] = 0
    uS, uP, uA = unit(Q[0]), unit(Q[1]), unit(Q[2])
    WU = r.standard_normal((VOC, DIM)) * 0.15
    WU[10:19] += 2.0 * uS; WU[20:41] += 2.0 * uP
    if kind != "dead":
        WU[100:110] += 2.0 * uA
    d = unit(uS) if kind == "oneside" else unit(uS + uP)
    n1 = orth(r.standard_normal(DIM), d, uS, uP); n2 = orth(r.standard_normal(DIM), d, uS, uP, n1)
    tilt = orth(uP - uS, d)                              # trades S for P
    mix = lambda kap, u: unit(kap * d + np.sqrt(1 - kap ** 2) * u)
    one = kind in ("frag", "fragnear", "split", "oneside", "dead", "void")
    V = dict(D=d, D16=unit(d + 0.3 * n1), A33=uA, PC_quote=unit(0.614 * uA + 0.789 * orth(r.standard_normal(DIM), uA)))
    if one:
        V["L_big"], V["x320"] = mix(0.735, tilt), mix(0.644, tilt)
    elif kind == "tilted":
        V["L_big"], V["x320"] = mix(0.735, unit(0.45 * tilt + 0.89 * n1)), mix(0.644, unit(0.45 * tilt + 0.89 * n2))
    elif kind == "disagree":
        V["L_big"], V["x320"] = mix(0.735, tilt), mix(0.644, -tilt)
    else:                                                # single, lowpower
        V["L_big"], V["x320"] = mix(0.735, n1), mix(0.644, n2)
    if kind == "fragnear":                               # L_big exchangeable with a near-d pool that shares its tilt
        V["L_big"] = mix(0.735, unit(tilt + 0.25 * orth(r.standard_normal(DIM), uS, uP, d)))
    V["R_big"] = mix(0.807, tilt if kind == "split" else n2)
    V["L_262"], V["R_262"] = mix(0.685, -tilt if kind == "frag" else n1), mix(0.829, n2)
    n3 = orth(r.standard_normal(DIM), d, uS, uP, n1, n2); n4 = orth(r.standard_normal(DIM), d, uS, uP, n1, n2, n3)
    units = dict(md=n3, wu=n4) if kind == "lowpower" else dict(md=-tilt, wu=-tilt)    # + leans toward S
    V.update(E.graded_plants(d, 0.735, units, orth(r.standard_normal(DIM), d, uS, uP)))
    V["L_big_native"] = -V["L_big"]
    for nm in ("atom6428", "atom270", "atom33", "L13_183", "S2764", "signedK3"):
        V[nm] = unit(0.5 * d + r.standard_normal(DIM) * 0.1)
    V["atom226"] = unit(uS + 0.2 * n1)
    pool = np.vstack([orth(x, uS, uP) for x in r.standard_normal((400, DIM))])     # class-free latent-like rows
    Z = {f"dir_{k}": v for k, v in V.items()}
    Z["RAND"] = np.vstack([unit(x) for x in r.standard_normal((99, DIM))])
    for g in ("L_big", "x320", "R_big", "L_262", "R_262"):
        Z[f"bank_{g}"] = E.cos_bank(d, abs(float(V[g] @ d)), pool, 99, E.BANK_SEEDS[g])[0]
    near_pool = np.vstack([unit(0.1 * d + tilt + 0.25 * orth(x, uS, uP, d)) for x in r.standard_normal((200, DIM))]) if kind == "fragnear" else pool
    Z["bank_L_big_near"] = E.cos_bank(d, abs(float(V["L_big"] @ d)), near_pool, 99, E.BANK_SEEDS["L_big_near"])[0]
    H = r.standard_normal((NS, DIM)) * 0.5; H[:, ~keep] = 0
    Hpos = r.standard_normal((NS * 20, DIM)); Hpos[:, ~keep] = 0
    return Z, WU, H, Hpos, r.integers(0, 40, NS * 20), r.integers(0, 30, NS)


def run(kind, with_262k=False):
    Z, WU, H, Hpos, pos_ids, site_tok = world(kind)
    cov = {a: dict(void=(kind == "void" and a == "L_big"), outside_any=(kind == "void" and a == "L_big"))
           for a in ("L_big", "x320", "R_big", "L_262", "R_262", "L_big_near")}
    frz = dict(with_262k=with_262k, directions={}, covariates=cov, skipped_262k=None if with_262k else "not requested")
    V, names, plan = E.plan_of(frz, Z, e65)
    ctx = E.ctx_from(sigs, twin, classes, refs)
    ctx.update(Dsel_names=names, site_tok=site_tok)
    Dmat = np.vstack([V[n] for n in names]); uni = ctx["union"]

    def fwd(v=None, alpha=0.0, cap=None):
        if cap is not None:
            cap["proj"].append(Hpos @ Dmat.T); cap["ids"].append(pos_ids)
        X = (H + (0 if v is None else alpha * v)) @ WU.T
        X = X - X.max(1, keepdims=True); X = X - np.log(np.exp(X).sum(1, keepdims=True))
        return X[:, uni], (None if v is None else 1.0)
    st = E.steer_engine(fwd, V, plan, ctx, dump=lambda o: json.dumps({k: x for k, x in o.items() if not k.startswith("_")}),
                        log=lambda m: None)
    return E.verdict_of(json.loads(json.dumps(st)), frz), st


expect = dict(frag="FRAGMENTED", fragnear="FRAGMENTED", single="NOT FRAGMENTED", tilted="TILTED", split="SPLIT, NOT REASSEMBLED", dead="NO VERDICT",
              oneside="UNINFORMATIVE", lowpower="INCONCLUSIVE", void="INCONCLUSIVE", disagree="INCONCLUSIVE")
for kind, want in expect.items():
    v, st = run(kind, with_262k=(kind == "frag"))
    got = v["headline"]
    extra = ""
    if "classes" in v:
        extra = " " + " ".join(f"{a}={c['cls']}" for a, c in v["classes"].items() if a in ("L_big", "x320", "R_big")) + f" PC={v['PC_PASS']} agree={v['x320_agreement']['agrees']}"
    print(f"{kind:9s} -> {got}{extra}")
    ok.append((f"world {kind}", got == want))
    if kind == "frag":
        ok.append(("262k branch", v["dictionaries"]["262k small"]["verdict"] == "FRAGMENTED" and v["262k_agrees"] is True
                   and v["classes"]["L_262"]["cls"] == "S-ONE-SIDED" and v["classes"]["L_big"]["cls"] == "P-ONE-SIDED"))
        ok.append(("planted control passes", v["PC_PASS"] and v["planted"]["md_f100"]["PASS"] and v["planted"]["md_f50"]["PASS"]))
        ok.append(("bank count and dose", all(len(st["banks"][g]) == 99 for g in ("L_big", "x320", "R_big", "L_262", "R_262", "L_big_near"))
                   and all(abs(st["bank_dose"][g]["alpha"] - 2.0 * st["sigma"][g if g != "L_big_near" else "L_big"]) < 1e-12 for g in st["bank_dose"])))
        ok.append(("LOO rates", all(abs(sum(x.values()) - 1) < 1e-9 for y in v["loo"].values() for x in y.values())))
        ok.append(("near bank and agreement", v["near_d_bank"]["cls"] == "P-ONE-SIDED" and v["near_d_bank"]["qualifier"] is None
                   and not v["near_d_bank"]["tau_inside_near_band_both_readouts"] and v["x320_agreement"]["agrees"]
                   and v["R_big_atoms"]["point_both_ways"] and v["wording"].startswith("the reconstruction's atoms point both ways")))
        ok.append(("reference words reported", any(k.startswith("ref_P:") for k in st["arms"]["D"]["c=2.0"]["words"])))
        ok.append(("corrected readout", all(abs(st["arms"]["D"]["c=2.0"]["class_delta"][c] - (st["arms"]["D"]["c=2.0"]["class_delta"][c + "_raw"]
                                                - st["arms"]["D"]["c=2.0"]["class_delta"][c + "_ref"])) < 1e-9 for c in ("S", "P"))))
    if kind == "fragnear":
        ok.append(("near-d qualifier fires", v["near_d_bank"]["tau_inside_near_band_both_readouts"] and v["near_d_bank"]["qualifier"] is not None))
    if kind == "lowpower":
        ok.append(("low power blocks NOT FRAGMENTED", not v["PC_PASS"] and all(v["classes"][a]["cls"] == "TWO-SIDED" for a in ("L_big", "x320"))))
    if kind == "dead":
        ok.append(("anchor stop", st.get("stopped", "").startswith("anchor failed") and "arms" not in st))

# ---------------------------------------------------------------- 5. code check of the one-sided bound
r = np.random.default_rng(99)
n_world, flagged, two = 300, 0, 0
for w in range(n_world):
    Q = np.linalg.qr(r.standard_normal((DIM, DIM)))[0].T; Q[:, ~keep] = 0
    uS, uP = unit(Q[0]), unit(Q[1]); d = unit(uS + uP)
    WU = r.standard_normal((VOC, DIM)) * 0.15; WU[10:19] += 2.0 * uS; WU[20:41] += 2.0 * uP
    H = r.standard_normal((NS, DIM)) * 0.5; H[:, ~keep] = 0
    pool = np.vstack([unit(x) for x in r.standard_normal((200, DIM))])          # rows carry some class signal
    B, _ = E.cos_bank(d, 0.735, pool, 100, 1000 + w)
    ccols, rcols = {c: np.asarray(classes[c]["ids"]) for c in ("S", "P")}, {c: np.asarray(refs[c]["ids"]) for c in ("S", "P")}

    def lp(v):
        X = (H + (0 if v is None else 2.0 * v)) @ WU.T
        X = X - X.max(1, keepdims=True)
        return X - np.log(np.exp(X).sum(1, keepdims=True))
    LP0 = lp(None)
    cdel = lambda v: E.class_deltas(lp(v), LP0, ccols, rcols)
    dD = cdel(d)
    big = dict(S=-1e9, P=-1e9)
    c = E.classify_both(cdel(B[0]), [cdel(b) for b in B[1:]], dict(corrected=big, raw=big),
                        {rd: E.angle(E.view(dD, rd)["S"], E.view(dD, rd)["P"]) for rd in E.READOUTS})["cls"]
    flagged += c.endswith(("ONE-SIDED", "TILTED")); two += c == "TWO-SIDED"
rate = flagged / n_world
print(f"bound check: ONE-SIDED or TILTED {flagged}/{n_world} = {rate:.3f} (bound 0.02), TWO-SIDED {two}/{n_world}")
ok.append(("bound check", rate <= 0.02 + 3 * np.sqrt(0.02 * 0.98 / n_world)))
ok.append(("exchangeable arm mostly two-sided", two / n_world >= 0.9))

for name, val in ok:
    print(f"{'ok ' if val else 'FAIL'} {name}")
assert all(val for _, val in ok), "smoke test failed"
print(f"e72f smoke: all {len(ok)} checks pass")
