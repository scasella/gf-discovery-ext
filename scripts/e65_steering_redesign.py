"""e65 -- Stage 4 v2, redesigned behavioural (steering) tier (PREREG_occurrence_v2_steering.md, sha256 asserted).

This tier follows a SEEN failure: in e64 both anchors failed the sibling-maximum null. Every object
the redesign does not touch is e64's and is asserted by sha256: the frozen v2 units, the D2 maps
(results/e64_maps_L{7,12}.npz), the pattern pullback direction d_u, the blind signatures T_u and
random sets R_u (rebuilt by e64's code and RNG order, checked against e64_behaviour.json), the
labels and the context reading. What changes:
  dose      h <- h + c * sigma_d * d at the site, with sigma_d = SD of the unsteered residual projected
            on d over the block's prompt positions >= 4; c in {0.5, 1, 2, 4}, gate at c* = 2
  RAND      99 pullback-matched random directions per layer and block: isotropic in D2's whitened
            PCA-64 coordinates, pulled back by e64's map U diag(sd), each dosed by its own sigma
  SIG       shuffled-signature control: u's own steered forward scored with every other screened
            unit's signature (twin excluded; T_u's tokens removed from the other T and R)
  contexts  B3 = reserve 4000-4999 (test); B4 = reserve 5000-5999 (held-out confirmation)

Phases (in order; each asserts its predecessors' outputs are in docs/HASHES.txt and appends its own)
  selftest  synthetic arrays only: dose, RAND draws, SIG sets, scoring, gates, bound arithmetic
  freeze    (CPU) rebuild T_u / R_u for every screened unit; STOP unless the six control units match
            e64_behaviour.json; draw the B3 and B4 sites and RAND banks; write and hash all of it
            before any forward
  steer     --block B3 | B4 (MPS) per layer, anchor first (prereg sections 8 and 10)
  verdict   (CPU) e64's tiers 1 and 3, unchanged, with the B3 / B4 behavioural status; a family that the
            post-verdict e64c rules name at |AUROC| >= 0.65 never reaches the top class
"""
import argparse, hashlib, importlib.util, json, os, sys, time
from pathlib import Path

import numpy as np

EXT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXT / "src"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")

PREREG = EXT / "PREREG_occurrence_v2_steering.md"
HASHES = EXT / "docs/HASHES.txt"
E64_REL = "scripts/e64_occurrence_v2_discovery.py"
FROZEN = {                                  # asserted at every data phase (prereg section 2)
    E64_REL: "1c97e42b4fc2fbf64410e8dda39fb8a9d6ae938e3fa960e6bf36d1a009fcf1f4",
    "results/e64_maps_L7.npz": "1b0c38ab872bb55f8c1af5127267ef3123dc4d743bd7d6c51db23db49e0b4faf",
    "results/e64_maps_L12.npz": "ac1b55b2eee03594b2d9f512632db2e9980e2446d051560d950ade8726200237",
    "results/e64_labels.json": "c86d651922a8929f2c9b8d17b1258e3082d14c6528a5b5a24ceb073e5bdf1336",
    "results/e64_context_reading.json": "4847f400019b720c3e38702b52d98a80613460ecc409b49d2455c06d491f3a52",
    "results/e64_shuffled.json": "342268918d4255dfda9adbe170e8e501c41c54526f0cd9f929cce6db8dd55b7a",
    "results/e64_behaviour.json": "c5a213e49b87830e02fa01eced2c1ca2d93a5afe4a34a03fa92a5aea4b6c6bb8",
    "results/e64_replicate.json": "b781ea197bb10752133ad570b04b974b0300bf068d1387137fde339ab31b423f",
    "results/e64_verdict.json": "bacdc3ee6f4845ab87875283e1b8f4b1c3fafd1730b747f15ef4fac247804ea9",
    "results/e64c_context_rules.json": "36f52a478cc76b7e0751e721f0c68436f3d7ae328a9c7e87bf7b20a1143f018d",
    "cache/occ/occD2_meta.npz": "bc910456116d5b875438c741c42379e24715576a2bea29e33b0dd9c1a92bc5a7",
    "results/corpus_tokens_pass1.i32": "8fbf98c3f29ebf9785885c1ac2df8ebeee7d0978c4663cd25517c5a925ba517e",
    "results/reserve_tokens.i32": "3da028b79acf01f7b67e70eb602ebed37a8883b074f9dac9f08f6ccc2b79b5b5",
    "results/lag_deltas_L12.npz": "65b546b3ebb6f7d0a75807e68d712916697b14408008c176eb3af3075afa21ba",
}
LAYERS, KINDS = (7, 12), ("logcosh", "skew")
BLOCKS = dict(B3=dict(reserve=(4000, 5000), site_seed=6500, rand_seed=6510),
              B4=dict(reserve=(5000, 6000), site_seed=6600, rand_seed=6610))
DOSES, C_STAR, SC_DOSES = (0.5, 1.0, 2.0, 4.0), 2.0, (0.5, 1.0, 2.0)
N_SITES, N_RAND, CTX, BATCH, SC_MIN, POS_MIN = 200, 99, 64, 32, 0.8, 4
SIG_MIN_TOKENS = 5                          # an other signature (or its R set) needs >= 5 tokens after T_u removal
TYPE_MIN = 8                                # within-type SD: token types with >= 8 positions in the block
OLD_EPS = 0.02                              # e64's gate push, eps * ||h||, reported for comparison only
SIG_RNG = 6400                              # e64's R_u RNG: default_rng(6400 + L), consumed in screened order
E64C_AUROC, E64C_MODEL = 0.65, "en_core_web_trf"   # post-verdict e64c rules (prereg section 0.6)
E64C_STRONG = {7: {"logcosh/31", "skew/35", "logcosh/41", "skew/41", "skew/59", "logcosh/53"},
               12: {"skew/8", "skew/0", "logcosh/0"}}
OUT = dict(frozen="results/e65_frozen.json", B3="results/e65_steer_B3.json", B4="results/e65_steer_B4.json",
           verdict="results/e65_verdict.json")


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def rand_path(L):
    return f"results/e65_rand_L{L}.npz"


_E64 = None


def load_e64():
    """e64 as a frozen module: its signature, unit lists, maps loader, hashing and swap guard are reused."""
    global _E64
    if _E64 is None:
        assert sha(EXT / E64_REL) == FROZEN[E64_REL], "e64 changed"
        spec = importlib.util.spec_from_file_location("e64", EXT / E64_REL)
        mod = importlib.util.module_from_spec(spec); sys.modules["e64"] = mod; spec.loader.exec_module(mod)
        _E64 = mod
    return _E64


# ---------------------------------------------------------------------------- integrity
def check_integrity():
    assert sha(PREREG) in HASHES.read_text(), "PREREG_occurrence_v2_steering.md does not match its sha256 in docs/HASHES.txt"
    assert sha(__file__) in PREREG.read_text(), "e65 is not the frozen version (its sha256 is not in the prereg)"
    for rel, h in FROZEN.items():
        assert sha(EXT / rel) == h, f"{rel} changed since e64"
    load_e64().check_integrity()            # e64's prereg, addendum D1 and the frozen v2 inputs


def require_hashed(*rels):
    load_e64().require_hashed(*rels)


# ---------------------------------------------------------------------------- pure helpers (selftest covers these)
def keep_mask(drop):
    k = np.ones(1152, bool); k[list(drop)] = False
    return k


def rand_bank(P_U, P_sd, keep, n, seed):
    """n directions isotropic in D2's whitened PCA-64 coordinates, pulled back by e64's map U diag(sd)
    (the map e64 applied to Cov(Z, y_u)), embedded at the kept coordinates and normalised."""
    g = np.random.default_rng(seed).standard_normal((n, P_U.shape[1]))
    D = np.zeros((n, len(keep))); D[:, keep] = (P_U @ (P_sd[:, None] * g.T)).T
    return D / np.linalg.norm(D, axis=1, keepdims=True)


def draw_sites(block, L):
    """N_SITES distinct reserve sequences of the block, one position each in [5, 255] (BOS-included)."""
    lo, hi = BLOCKS[block]["reserve"]
    rng = np.random.default_rng(BLOCKS[block]["site_seed"] + L)
    seqs = lo + rng.choice(hi - lo, N_SITES, replace=False)
    pos = rng.integers(5, 256, size=N_SITES)
    return [[int(s), int(p)] for s, p in zip(seqs, pos)]


def natural_sigma(proj, ids, mask, type_min=TYPE_MIN):
    """proj (n_pos, n_dir): the unsteered residual projected on each direction at the block's prompt
    positions >= POS_MIN (mask). Returns the total SD (the dose scale) and the pooled within-type SD
    over token types with >= type_min positions (reported)."""
    x, t = proj[mask], ids[mask]
    total = x.std(0, ddof=1)
    ut, inv, cnt = np.unique(t, return_inverse=True, return_counts=True)
    ok = cnt[inv] >= type_min
    xs, iv = x[ok], inv[ok]
    mu = np.zeros((len(ut), x.shape[1])); np.add.at(mu, iv, xs); mu /= np.maximum(cnt, 1)[:, None]
    dof = int((cnt[cnt >= type_min] - 1).sum())
    within = np.sqrt(((xs - mu[iv]) ** 2).sum(0) / max(dof, 1))
    return total, within, dict(n_positions=int(mask.sum()), n_within=int(ok.sum()),
                               n_types_within=int((cnt >= type_min).sum()))


def lse(x, cols):
    m = x[:, cols].max(1, keepdims=True)
    return (m[:, 0] + np.log(np.exp(x[:, cols] - m).sum(1)))


def readout(X, X0, Tc, Rc):
    """Per-site Delta = [lse(T) - lse(R)] steered minus unsteered (e62 / e64 readout)."""
    return (lse(X, Tc) - lse(X, Rc)) - (lse(X0, Tc) - lse(X0, Rc))


def sign_consistency(dl, site_tok):
    """e64.behaviour's rule, unchanged: share of the 20 commonest site token types with mean Delta > 0."""
    types, cnt = np.unique(site_tok, return_counts=True)
    top = types[np.argsort(-cnt)[:20]]
    return float(np.mean(np.asarray([dl[site_tok == t].mean() for t in top]) > 0))


def sig_sets(u, sigs, twin):
    """The shuffled-signature comparison set for u: every other screened unit v at the layer except u's
    twin, with T_u's tokens removed from T_v and R_v; v is used only if both keep >= SIG_MIN_TOKENS."""
    Tu = set(sigs[u]["T"])
    out, dropped = {}, []
    for v, s in sigs.items():
        if v == u or v == twin:
            continue
        T = [t for t in s["T"] if t not in Tu]; R = [t for t in s["R"] if t not in Tu]
        if len(T) >= SIG_MIN_TOKENS and len(R) >= SIG_MIN_TOKENS:
            out[v] = dict(T=T, R=R)
        else:
            dropped.append(v)
    return out, dropped


def gates(d_uu, sc, rand_row, sig_row):
    """PASS rule (prereg section 8): Delta_uu(c*) > 0, sign consistency >= 0.8 at some c <= c*,
    Delta_uu(c*) > max over the RAND bank (u's signature), Delta_uu(c*) > max over the SIG set."""
    rand_row, sig_row = np.asarray(rand_row, float), np.asarray(sig_row, float)
    g = dict(positive=bool(d_uu > 0), sc_ok=bool(max(sc.values()) >= SC_MIN),
             rand_ok=bool(len(rand_row) and d_uu > rand_row.max()),
             sig_ok=bool(len(sig_row) and d_uu > sig_row.max()))
    g["PASS"] = all(g.values())
    b_rand, b_sig = 1.0 / (len(rand_row) + 1), 1.0 / (len(sig_row) + 1)
    g.update(bound_global_null=min(b_rand, b_sig), bound=max(b_rand, b_sig))
    return g


def e64c_parse(layer_rules):
    """From one layer of results/e64c_context_rules.json: candidates (status 'none') that some rule reaches
    at |AUROC| >= 0.65 (transformer parse; listed in the rule's top5 or would_name) -> {rule: |AUROC|}, and
    candidates that some rule would name under the e64 section 6.1 thresholds -> [[rule, |AUROC|]]."""
    strong, thr = {}, {}
    for rule, m in layer_rules.items():
        a = m[E64C_MODEL]
        for r in a["would_name"]:
            if r["status"] == "none":
                thr.setdefault(r["unit"], []).append([rule, r["abs_auroc"]])
        for r in a["top5"] + a["would_name"]:
            if r["status"] == "none" and r["abs_auroc"] >= E64C_AUROC:
                strong.setdefault(r["unit"], {})[rule] = r["abs_auroc"]
    return strong, thr


# ---------------------------------------------------------------------------- selftest (no data, no model)
def selftest():
    rng = np.random.default_rng(0)
    ok = []
    # dose: total SD and pooled within-type SD on a synthetic set with known structure
    n, nd = 20000, 3
    ids = rng.integers(0, 40, n); ids[:50] = 999                 # one rare type, below TYPE_MIN after masking
    tm = rng.normal(0, 5, (1000, nd))
    proj = tm[ids] + rng.normal(0, [1.0, 2.0, 3.0], (n, nd))
    mask = np.ones(n, bool); mask[:45] = False
    tot, wit, info = natural_sigma(proj, ids, mask)
    ok.append(("sigma total", np.allclose(tot, proj[mask].std(0, ddof=1))))
    ok.append(("sigma within", np.allclose(wit, [1, 2, 3], rtol=0.03) and info["n_types_within"] == 40))
    # RAND bank: shape, unit norm, zero at the dropped coordinate, inside span(U), deterministic
    P_U = np.linalg.qr(rng.standard_normal((1151, 64)))[0]; P_sd = np.sort(rng.uniform(15, 140, 64))[::-1]
    keep = keep_mask([1038])
    D = rand_bank(P_U, P_sd, keep, N_RAND, 7)
    resid = D[:, keep] - (D[:, keep] @ P_U) @ P_U.T
    ok.append(("rand bank", D.shape == (N_RAND, 1152) and np.allclose(np.linalg.norm(D, axis=1), 1)
               and np.all(D[:, 1038] == 0) and np.abs(resid).max() < 1e-10
               and np.array_equal(D, rand_bank(P_U, P_sd, keep, N_RAND, 7))))
    # sites: distinct sequences inside the block, positions in [5, 255]
    s = np.asarray(draw_sites("B3", 7)); s4 = np.asarray(draw_sites("B4", 7))
    ok.append(("sites", len(set(s[:, 0])) == N_SITES and s[:, 0].min() >= 4000 and s[:, 0].max() < 5000
               and s4[:, 0].min() >= 5000 and s4[:, 0].max() < 6000 and s[:, 1].min() >= 5 and s[:, 1].max() <= 255))
    # scoring: vectorised lse readout equals the direct formula
    X0, X = rng.normal(size=(7, 30)), rng.normal(size=(7, 30))
    Tc, Rc = [1, 4, 9], [2, 3, 11, 12]
    f = lambda x: np.log(np.exp(x[:, Tc]).sum(1)) - np.log(np.exp(x[:, Rc]).sum(1))
    ok.append(("readout", np.allclose(readout(X, X0, Tc, Rc), f(X) - f(X0))))
    # SIG sets: twin excluded, T_u removed from the other T and R, small sets dropped
    sigs = {"a": dict(T=[1, 2, 3, 4, 5, 6], R=[50, 51, 52, 53, 54, 55]),
            "b": dict(T=[1, 2, 7, 8, 9, 10, 11], R=[60, 61, 62, 63, 64, 3]),
            "c": dict(T=[1, 2, 3, 4, 12, 13], R=[70, 71, 72, 73, 74, 75]),
            "tw": dict(T=[20, 21, 22, 23, 24, 25], R=[80, 81, 82, 83, 84, 85])}
    S, dropped = sig_sets("a", sigs, "tw")
    ok.append(("sig sets", set(S) == {"b"} and S["b"]["T"] == [7, 8, 9, 10, 11]
               and S["b"]["R"] == [60, 61, 62, 63, 64] and dropped == ["c"]))
    # gates on hand cases
    sc = {0.5: 0.5, 1.0: 0.85, 2.0: 0.7}
    g1 = gates(1.0, sc, [0.2, 0.9], [0.5, 0.99])
    g2 = gates(1.0, sc, [0.2, 1.0], [0.5])
    g3 = gates(1.0, {0.5: 0.7, 1.0: 0.75, 2.0: 0.79}, [0.2], [0.5])
    lr = {"A": {E64C_MODEL: dict(top5=[dict(unit="logcosh/1", abs_auroc=0.9, status="context:A"),
                                     dict(unit="skew/2", abs_auroc=0.7, status="none"),
                                     dict(unit="skew/3", abs_auroc=0.6, status="none")],
                               would_name=[dict(unit="skew/3", abs_auroc=0.6, status="none")])},
          "B": {E64C_MODEL: dict(top5=[dict(unit="skew/4", abs_auroc=0.66, status="battery-named")], would_name=[])}}
    st, th = e64c_parse(lr)
    ok.append(("e64c parse", st == {"skew/2": {"A": 0.7}} and th == {"skew/3": [["A", 0.6]]}))
    ok.append(("gates", g1["PASS"] and not g2["PASS"] and not g2["rand_ok"] and not g3["PASS"]
               and abs(g1["bound"] - 1 / 3) < 1e-12 and abs(g1["bound_global_null"] - 1 / 3) < 1e-12))
    # bounds by Monte Carlo: exchangeable null -> RAND ~ 1/(M+1), SIG ~ 1/(n'+1); a generically potent
    # direction with no pairing effect passes RAND but SIG still bounds it
    reps, M, nsig = 40000, N_RAND, 63
    uu = rng.standard_normal(reps)
    rand_max = rng.standard_normal((reps, M)).max(1)
    sig_max = rng.standard_normal((reps, nsig)).max(1)
    p_rand, p_sig, p_both = (uu > rand_max).mean(), (uu > sig_max).mean(), ((uu > rand_max) & (uu > sig_max)).mean()
    pot = 3.0 + uu; sig_pot = 3.0 + rng.standard_normal((reps, nsig)).max(1)
    p_pot = ((pot > rand_max) & (pot > sig_pot)).mean()
    se = lambda p: 4 * np.sqrt(p * (1 - p) / reps)
    ok.append(("bound RAND", abs(p_rand - 1 / (M + 1)) < se(1 / (M + 1))))
    ok.append(("bound SIG", abs(p_sig - 1 / (nsig + 1)) < se(1 / (nsig + 1))))
    ok.append(("bound conj <= min", p_both <= min(p_rand, p_sig)))
    ok.append(("bound potent", abs(p_pot - 1 / (nsig + 1)) < se(1 / (nsig + 1))))
    print(f"MC: P(RAND)={p_rand:.4f} (1/{M + 1}={1 / (M + 1):.4f})  P(SIG)={p_sig:.4f} (1/{nsig + 1}={1 / (nsig + 1):.4f})"
          f"  P(both)={p_both:.4f}  P(potent, both)={p_pot:.4f}")
    for name, v in ok:
        print(f"{'ok ' if v else 'FAIL'} {name}")
    assert all(v for _, v in ok), "selftest failed"
    print(f"selftest: all {len(ok)} checks pass")


# ---------------------------------------------------------------------------- freeze (CPU; data, no model)
def next_by_type(tokD):
    """e64.behaviour's next-token table, verbatim."""
    nxt = {}
    r_all = np.arange(len(tokD)); r_all = r_all[(r_all % 255) != 254]
    order = np.argsort(tokD[r_all], kind="stable")
    tt = tokD[r_all][order]; nn = tokD[r_all + 1][order]
    for b in np.split(np.arange(len(tt)), np.flatnonzero(np.diff(tt)) + 1):
        u_, c_ = np.unique(nn[b], return_counts=True)
        nxt[int(tt[b[0]])] = (u_, c_ / c_.sum())
    return nxt


def rebuild_signatures(e64, L, tokD, nxt, ccount):
    """e64.behaviour's build(), in e64's screened order with a fresh default_rng(6400 + L): T_u is
    e64.signature on the heavy-tail 1% of D2 rows; R_u is frequency-matched (corpus count within 10%)."""
    fr = e64.frozen_v2()["layers"][str(L)]["frozen"]
    fnpz = np.load(EXT / f"results/occ_units_v2_L{L}.npz")
    rows_D = fnpz["row"].astype(np.int64)
    _, _, z = e64.load_maps(L)
    rng = np.random.default_rng(SIG_RNG + L)
    out = {}
    for kd, j in [(kd, j) for kd in KINDS for j in e64.screened_ids(fr, kd)]:
        y = fnpz[f"Y_{kd}"][:, j].astype(np.float64)
        s = z[f"sign_{kd}"][j]
        k = int(np.ceil(e64.TAIL * len(y)))
        mem = np.argsort(-s * y)[:k]
        T = e64.signature(rows_D, mem, tokD, nxt)
        R = []
        for t in T:
            c = ccount[t]
            cand = np.flatnonzero((ccount >= 0.9 * c) & (ccount <= 1.1 * c) & (ccount > 0))
            cand = cand[~np.isin(cand, T)]
            R.append(int(rng.choice(cand)) if len(cand) else int(rng.integers(e64.VOCAB_N)))
        out[f"{kd}/{j}"] = dict(T=T, R=R)
    return out


def tokenizer():
    from transformers import AutoTokenizer
    snap = sorted((EXT / "hf_home/hub/models--google--gemma-3-1b-pt/snapshots").glob("*"))[0]
    return AutoTokenizer.from_pretrained(str(snap))


def freeze(layers):
    e64 = load_e64()
    assert not (EXT / OUT["frozen"]).exists(), f"{OUT['frozen']} exists: the freeze runs once"
    tk = tokenizer()
    tokD = np.load(EXT / "cache/occ/occD2_meta.npz")["tokens"].astype(np.int64)
    nxt = next_by_type(tokD)
    ccount = np.bincount(np.fromfile(EXT / "results/corpus_tokens_pass1.i32", dtype=np.int32), minlength=e64.VOCAB_N)
    beh = json.loads((EXT / "results/e64_behaviour.json").read_text())
    fz = e64.frozen_v2()
    out = dict(layers={})
    for L in layers:
        sigs = rebuild_signatures(e64, L, tokD, nxt, ccount)
        dec = lambda ids: [tk.decode([t]) for t in ids]
        b = beh["layers"][str(L)]
        recs = [b["anchor"]] + b["secondary_anchors"]
        check = []
        for r in recs:
            s = sigs[f"{r['kind']}/{r['id']}"]
            check.append(dict(unit=f"{r['kind']}/{r['id']}", T_match=dec(s["T"]) == r["signature"],
                              R_match=dec(s["R"]) == r["random_set"]))
        ctrl = {f"{kd}/{j}" for kd, j, _, _ in e64.control_units(fz, L)}
        assert {c["unit"] for c in check} == ctrl, "e64_behaviour.json does not hold every control unit"
        _, _, z = e64.load_maps(L)
        keep = keep_mask(e64.e61.drop_for(L))
        for kd in KINDS:
            assert np.all(z[f"dir_{kd}"][:, ~keep] == 0), "kept-coordinate mask disagrees with the e64 directions"
        np.savez_compressed(EXT / rand_path(L), **{blk: rand_bank(z["P_U"], z["P_sd"], keep, N_RAND,
                                                                  BLOCKS[blk]["rand_seed"] + L) for blk in BLOCKS})
        out["layers"][str(L)] = dict(
            signatures={u: dict(T=s["T"], R=s["R"], T_text=dec(s["T"]), R_text=dec(s["R"])) for u, s in sigs.items()},
            control_check=check, signatures_ok=all(c["T_match"] and c["R_match"] for c in check),
            sites={blk: draw_sites(blk, L) for blk in BLOCKS}, rand_npz=rand_path(L), rand_sha256=sha(EXT / rand_path(L)),
            n_screened=len(sigs), n_empty_signature=sum(not s["T"] for s in sigs.values()))
        print(f"L{L} freeze: {len(sigs)} signatures, control check {check}", flush=True)
    out["signatures_ok"] = all(v["signatures_ok"] for v in out["layers"].values())
    e64.dump(out, OUT["frozen"])
    for L in layers:
        e64.append_hash(rand_path(L), "e65 freeze: RAND banks for B3 and B4, before any forward")
    e64.append_hash(OUT["frozen"], "e65 freeze: signatures, control check, B3/B4 sites, before any forward")
    assert out["signatures_ok"], "rebuilt control signatures differ from e64_behaviour.json: STOP"


# ---------------------------------------------------------------------------- steer (MPS)
def roles(e64, L):
    """Every screened unit at L with its role: anchor, secondary (other control units), candidate
    (battery-UNNAMED, not context-named), context_named, named (battery-NAMED, not a control)."""
    fz = e64.frozen_v2()
    fr = fz["layers"][str(L)]["frozen"]
    akd, aid, _ = e64.ANCHOR[L]
    ctrl = {(kd, j) for kd, j, _, _ in e64.control_units(fz, L)}
    cands, lab = e64.candidates(L)
    cmap = {(c["kind"], c["id"]): ("context_named" if c["context_named"] else "candidate") for c in cands}
    out = {}
    for kd in KINDS:
        for j in e64.screened_ids(fr, kd):
            key = (kd, j)
            out[key] = ("anchor" if key == (akd, aid) else "secondary" if key in ctrl else cmap.get(key, "named"))
    twin = {}
    for a, b, _ in lab["twins"]:
        twin[("logcosh", a)] = ("skew", b); twin[("skew", b)] = ("logcosh", a)
    return out, twin


def b4_plan(b3, L):
    """Units re-tested on B4 at L: empty unless the B3 anchor passed and >= 1 non-control unit passed B3."""
    lay = b3["layers"].get(str(L))
    if not lay or not lay["anchor"]["PASS"]:
        return []
    return [(r["kind"], r["id"]) for r in lay["evaluated"] if r["role"] in ("candidate", "context_named") and r["PASS"]]


def steer(block, layers):
    e64 = load_e64()
    require_hashed(OUT["frozen"], *[rand_path(L) for L in layers])
    frz = json.loads((EXT / OUT["frozen"]).read_text())
    assert frz["signatures_ok"]
    b3 = None
    if block == "B4":
        require_hashed(OUT["B3"])
        b3 = json.loads((EXT / OUT["B3"]).read_text())
    assert not (EXT / OUT[block]).exists(), f"{OUT[block]} exists: each block runs once"
    out = dict(block=block, reserve=list(BLOCKS[block]["reserve"]), swap_gb_at_start=e64.check_swap(), layers={})
    plans = {L: (b4_plan(b3, L) if block == "B4" else None) for L in layers}
    if block == "B4" and not any(plans.values()):
        out["note"] = "no layer qualifies (B3 anchor PASS and >= 1 B3 pass); nothing run"
        e64.dump(out, OUT[block]); e64.append_hash(OUT[block], "e65 steer B4 (not run: no layer qualified)")
        return
    import torch
    import gfdisc.model as gm
    gm.MODEL_NAME = str(sorted((EXT / "hf_home/hub/models--google--gemma-3-1b-pt/snapshots").glob("*"))[0])
    from gfdisc.env import DEVICE
    model, tok = gm.load_model(token=None)
    rv = np.fromfile(EXT / "results/reserve_tokens.i32", dtype=np.int32).reshape(-1, 256)
    for L in layers:
        t0 = time.time()
        if block == "B4" and not plans[L]:
            out["layers"][str(L)] = dict(note="not run: B3 anchor failed or no B3 pass at this layer")
            e64.dump(out, OUT[block]); continue
        fl = frz["layers"][str(L)]
        _, _, z = e64.load_maps(L)
        rl, twin = roles(e64, L)
        sigs = {tuple([k.split("/")[0], int(k.split("/")[1])]): dict(T=s["T"], R=s["R"]) for k, s in fl["signatures"].items()}
        assert set(sigs) == set(rl)
        dirs = {key: z[f"dir_{key[0]}"][key[1]] for key in rl}
        assert sha(EXT / rand_path(L)) == fl["rand_sha256"]
        RB = np.load(EXT / rand_path(L))[block]
        sites = fl["sites"][block]
        prompts = [np.concatenate([[tok.bos_token_id], rv[s, max(1, p - CTX):p + 1]]) for s, p in sites]
        site_tok = np.asarray([int(rv[s, p]) for s, p in sites])
        union = sorted({t for s in sigs.values() for t in s["T"] + s["R"]})
        col = {t: i for i, t in enumerate(union)}
        cols = lambda ts: np.asarray([col[t] for t in ts], dtype=np.int64)
        uni = torch.tensor(union, device=DEVICE)
        keys = sorted(rl)
        Dall = np.vstack([dirs[k] for k in keys] + [RB])               # units, then the RAND bank
        Dall_t = torch.tensor(Dall, dtype=torch.float32, device=DEVICE)

        def forward(d=None, alpha=0.0, cap=None):
            outs, real = [], []
            for lo in range(0, len(prompts), BATCH):
                blk = prompts[lo:lo + BATCH]
                Lm = max(len(x) for x in blk)
                ids = torch.full((len(blk), Lm), tok.pad_token_id, dtype=torch.long)
                att = torch.zeros_like(ids)
                for i, x in enumerate(blk):
                    ids[i, :len(x)] = torch.from_numpy(x.astype(np.int64)); att[i, :len(x)] = 1
                last = torch.tensor([len(x) - 1 for x in blk])
                ib = torch.arange(len(blk), device=DEVICE); lp = last.to(DEVICE)
                hooks = []
                if d is not None:
                    Dv = torch.tensor(d, dtype=torch.float32, device=DEVICE)

                    def fn(_m, _i, o, Dv=Dv):
                        t = o[0] if isinstance(o, tuple) else o
                        new = t.float().clone()
                        new[ib, lp] = new[ib, lp] + alpha * Dv
                        new = new.to(t.dtype)
                        real.append(((new[ib, lp].float() - t[ib, lp].float()).norm(dim=-1) / alpha).cpu())
                        return (new,) + tuple(o[1:]) if isinstance(o, tuple) else new
                    hooks.append(model.model.layers[L].register_forward_hook(fn))
                if cap is not None:
                    def cfn(_m, _i, o, ids=ids, att=att):
                        t = (o[0] if isinstance(o, tuple) else o).float()
                        pos = torch.arange(t.shape[1])[None, :].expand(len(ids), -1)
                        m = (att.bool() & (pos >= POS_MIN)).numpy()
                        cap["proj"].append((t @ Dall_t.T).cpu().numpy()[m])
                        cap["ids"].append(ids.numpy()[m])
                        cap["hnorm"].append(t[ib, lp].norm(dim=-1).cpu().numpy())
                    hooks.append(model.model.layers[L].register_forward_hook(cfn))
                try:
                    with torch.no_grad():
                        lg = model(input_ids=ids.to(DEVICE), attention_mask=att.to(DEVICE), use_cache=False).logits
                    outs.append(lg[ib, lp][:, uni].float().cpu())
                finally:
                    for h in hooks:
                        h.remove()
            return torch.cat(outs, 0).numpy().astype(np.float64), (float(torch.cat(real).mean()) if real else None)

        # 1. unsteered forward: logits and the dose scale of every direction (units and RAND bank)
        cap = dict(proj=[], ids=[], hnorm=[])
        X0, _ = forward(cap=cap)
        proj = np.concatenate(cap["proj"]); pids = np.concatenate(cap["ids"]); hnorm = np.concatenate(cap["hnorm"])
        sig_tot, sig_wit, sinfo = natural_sigma(proj, pids, np.ones(len(proj), bool))
        sigma = {k: float(sig_tot[i]) for i, k in enumerate(keys)}
        sigma_w = {k: float(sig_wit[i]) for i, k in enumerate(keys)}
        sig_rand = sig_tot[len(keys):]
        h_mean = float(hnorm.mean())
        lay = dict(sites=dict(reserve=out["reserve"], n=len(sites)), sigma_info=sinfo, site_hnorm_mean=h_mean,
                   sigma_rand=dict(median=float(np.median(sig_rand)), min=float(sig_rand.min()), max=float(sig_rand.max())))
        base_f = {k: lse(X0, cols(sigs[k]["T"])) - lse(X0, cols(sigs[k]["R"])) for k in keys if sigs[k]["T"]}

        # 2. RAND bank at c*: each random direction scored with every unit's own signature
        rand = np.full((len(RB), len(keys)), np.nan)
        rand_real = []
        for i, d in enumerate(RB):
            X, rr = forward(d, C_STAR * float(sig_rand[i])); rand_real.append(rr)
            for j, k in enumerate(keys):
                if sigs[k]["T"]:
                    rand[i, j] = float(readout(X, X0, cols(sigs[k]["T"]), cols(sigs[k]["R"])).mean())
        lay["rand_realised_dose"] = float(np.mean(rand_real))
        lay["rand_cos_to_units"] = float(np.abs(RB @ np.vstack([dirs[k] for k in keys]).T).max())
        print(f"[{time.time() - t0:.0f}s] L{L} {block} base + RAND bank done", flush=True)

        def evaluate(key, doses=DOSES):
            u = sigs[key]
            rec = dict(kind=key[0], id=key[1], role=rl[key], sigma=sigma[key], sigma_within_type=sigma_w[key],
                       eps_equiv=C_STAR * sigma[key] / h_mean, ratio_to_old_push=C_STAR * sigma[key] / (OLD_EPS * h_mean))
            if not u["T"]:
                rec.update(PASS=False, note="empty signature"); return rec
            Tc, Rc = cols(u["T"]), cols(u["R"])
            for c in doses:
                X, rr = forward(dirs[key], c * sigma[key])
                dl = readout(X, X0, Tc, Rc)
                rec[f"c={c}"] = dict(mean_delta=float(dl.mean()), sign_consistency=sign_consistency(dl, site_tok),
                                     realised_dose=rr, z=float(dl.mean() / base_f[key].std()))
                if c == C_STAR:
                    S, dropped = sig_sets(key, sigs, twin.get(key))
                    row = {f"{v[0]}/{v[1]}": float(readout(X, X0, cols(s["T"]), cols(s["R"])).mean()) for v, s in S.items()}
                    zrow = {f"{v[0]}/{v[1]}": float(row[f"{v[0]}/{v[1]}"] / (lse(X0, cols(s["T"])) - lse(X0, cols(s["R"]))).std())
                            for v, s in S.items()}
            if C_STAR not in doses:
                return rec
            d_uu = rec[f"c={C_STAR}"]["mean_delta"]
            rrow = rand[:, keys.index(key)]
            sc = {c: rec[f"c={c}"]["sign_consistency"] for c in SC_DOSES if f"c={c}" in rec}
            top = sorted(row.items(), key=lambda x: -x[1])[:3]
            rec.update(rand=dict(max=float(rrow.max()), p50=float(np.median(rrow)), p99=float(np.quantile(rrow, 0.99)), n=len(rrow)),
                       sig=dict(max=max(row.values()) if row else None, n=len(row), dropped=[f"{v[0]}/{v[1]}" for v in dropped],
                                twin_excluded=(f"{twin[key][0]}/{twin[key][1]}" if key in twin else None), top3=top,
                                z_uu=rec[f"c={C_STAR}"]["z"], z_max=max(zrow.values()) if zrow else None,
                                z_rank=int(sum(v >= rec[f"c={C_STAR}"]["z"] for v in zrow.values())) + 1),
                       **(gates(d_uu, sc, rrow, list(row.values())) if len(sc) == len(SC_DOSES) else dict(PASS=None)))
            return rec

        akd, aid, alb = e64.ANCHOR[L]
        lay["anchor"] = evaluate((akd, aid)); lay["anchor"]["label"] = alb
        print(f"[{time.time() - t0:.0f}s] L{L} {block} anchor {akd}:{aid} PASS={lay['anchor']['PASS']}", flush=True)
        lay["secondary_anchors"] = [evaluate(k) for k in keys if rl[k] == "secondary"]
        lay["evaluated"] = []
        if not lay["anchor"]["PASS"]:
            lay["verdict"] = "ANCHOR FAILED -- no behavioural verdict at this layer"
        else:
            todo = plans[L] if block == "B4" else [k for k in keys if rl[k] in ("candidate", "context_named")]
            for k in todo:
                r = evaluate(tuple(k)); lay["evaluated"].append(r)
                print(f"[{time.time() - t0:.0f}s] L{L} {block} {r['role']} {k[0]}:{k[1]} PASS={r['PASS']}", flush=True)
            if block == "B3":                                         # named units at c* only: the sibling rank (reported)
                lay["named_c_star"] = [evaluate(k, doses=(C_STAR,)) for k in keys if rl[k] == "named"]
                own = {(r["kind"], r["id"]): r[f"c={C_STAR}"]["mean_delta"]
                       for r in [lay["anchor"]] + lay["secondary_anchors"] + lay["evaluated"] + lay["named_c_star"]
                       if f"c={C_STAR}" in r}
                for r in [lay["anchor"]] + lay["secondary_anchors"] + lay["evaluated"]:
                    key = (r["kind"], r["id"])
                    if key in own:
                        sib = [v for k2, v in own.items() if k2 != key and k2 != twin.get(key)]
                        r["sibling_percentile"] = float(np.mean(np.asarray(sib) < own[key])) if sib else None
            disc = [r for r in lay["evaluated"] if r["role"] == "candidate"]
            lay["n_pass"] = int(sum(bool(r["PASS"]) for r in disc))
            lay["n_pass_context_named"] = int(sum(bool(r["PASS"]) for r in lay["evaluated"] if r["role"] == "context_named"))
            lay["expected_false_pass_bound"] = float(sum(r.get("bound", 0.0) for r in disc))
            lay["expected_false_pass_bound_global_null"] = float(sum(r.get("bound_global_null", 0.0) for r in disc))
            lay["excess"] = bool(lay["n_pass"] > lay["expected_false_pass_bound"])
        lay["seconds"] = round(time.time() - t0)
        out["layers"][str(L)] = lay
        e64.dump(out, OUT[block])
    out["swap_gb_at_end"] = e64.swap_used_gb()
    e64.dump(out, OUT[block])
    e64.append_hash(OUT[block], f"e65 steer {block}")


# ---------------------------------------------------------------------------- verdict (CPU)
def verdict(layers):
    e64 = load_e64()
    require_hashed(OUT["B3"])
    b3 = json.loads((EXT / OUT["B3"]).read_text())
    need_b4 = any(b4_plan(b3, L) for L in layers)
    b4 = None
    if need_b4:
        require_hashed(OUT["B4"])
        b4 = json.loads((EXT / OUT["B4"]).read_text())
    ev = json.loads((EXT / "results/e64_verdict.json").read_text())
    rules = json.loads((EXT / "results/e64c_context_rules.json").read_text())["layers"]
    out = dict(e64_verdict_sha256=FROZEN["results/e64_verdict.json"], e64_verdict_amended=False, layers={},
               candidate_discoveries_post_hoc=[])
    for L in layers:
        l3 = b3["layers"][str(L)]
        l4 = (b4 or {}).get("layers", {}).get(str(L), {})
        a3 = bool(l3["anchor"]["PASS"]); a4 = bool(l4.get("anchor", {}).get("PASS"))
        r3 = {(r["kind"], r["id"]): r for r in l3.get("evaluated", [])}
        r4 = {(r["kind"], r["id"]): r for r in l4.get("evaluated", [])}
        strong, thr = e64c_parse(rules[str(L)]["per_rule"])
        assert set(strong) == E64C_STRONG[L], f"e64c >= {E64C_AUROC} set at L{L} differs from prereg section 0.6"
        rows = []
        for u in ev["layers"][str(L)]["units"]:
            key = (u["kind"], u["id"])
            cn = u["tier"] == "NAMED_BY_CONTEXT"
            p3 = bool(a3 and (cn or l3.get("excess")) and r3.get(key, {}).get("PASS"))
            p4 = bool(p3 and a4 and r4.get(key, {}).get("PASS"))
            t3 = u["tier"] == "replicated geometric"                 # e64: replicated on R2 at an informative layer
            if not a3:
                cls = "no behavioural verdict (B3 anchor failed)"
            elif not p3:
                cls = "no behavioural pass"
            elif not p4:
                cls = "exploratory behavioural pass (B3 only)"
            elif cn:
                cls = "held-out behavioural pass, NAMED_BY_CONTEXT (not a discovery class)"
            elif t3 and f"{key[0]}/{key[1]}" in strong:
                cls = "held-out behavioural pass, nameable post hoc by an e64c rule (not a discovery class)"
            elif t3:
                cls = "candidate discovery under the redesigned tier 2 (post hoc)"
            else:
                cls = "held-out behavioural pass, not replicated geometrically"
            rows.append(dict(kind=key[0], id=key[1], e64_tier=u["tier"], reading=u["reading"], twin=u["twin"],
                             identity=u["identity"], context_generic=u["context_generic"], r2_replicated=u["r2_replicated"],
                             b3_pass=r3.get(key, {}).get("PASS"), b4_pass=r4.get(key, {}).get("PASS"), e65_class=cls,
                             e64c_strong=strong.get(f"{key[0]}/{key[1]}"), e64c_threshold=thr.get(f"{key[0]}/{key[1]}")))
        fams, seen = [], set()
        for x in rows:
            k = (x["kind"], x["id"])
            if k in seen:
                continue
            mem = [x] + [y for y in rows if x["twin"] and (y["kind"], y["id"]) == tuple(x["twin"][0][:2])]
            seen.update((m["kind"], m["id"]) for m in mem)
            fams.append(dict(members=[(m["kind"], m["id"]) for m in mem], classes=[m["e65_class"] for m in mem],
                             e64c_strong=any(m["e64c_strong"] for m in mem),
                             candidate_discovery_post_hoc=(any(m["e65_class"].startswith("candidate discovery") for m in mem)
                                                           and not any(m["e64c_strong"] for m in mem))))
        out["layers"][str(L)] = dict(
            units=rows, families=fams, anchor_pass_B3=a3, anchor_pass_B4=(a4 if l4.get("anchor") else None),
            b3_excess=l3.get("excess"), b3_n_pass=l3.get("n_pass"), b3_expected_false_pass_bound=l3.get("expected_false_pass_bound"),
            b4_run=bool(l4.get("anchor")), n_candidate_discovery_families=sum(f["candidate_discovery_post_hoc"] for f in fams))
        out["candidate_discoveries_post_hoc"] += [dict(layer=L, **f) for f in fams if f["candidate_discovery_post_hoc"]]
    e64.dump(out, OUT["verdict"])
    e64.append_hash(OUT["verdict"], "e65 verdict (does not amend e64_verdict.json)")
    print("CANDIDATE DISCOVERIES (post hoc)", json.dumps(out["candidate_discoveries_post_hoc"]), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True, choices=["selftest", "freeze", "steer", "verdict"])
    ap.add_argument("--block", choices=list(BLOCKS))
    ap.add_argument("--layers", default="7,12")
    a = ap.parse_args()
    if a.phase == "selftest":
        return selftest()
    check_integrity()
    layers = [int(x) for x in a.layers.split(",")]
    assert layers == list(LAYERS), "the redesigned tier runs both layers, L7 then L12, in one call"
    load_e64().check_swap()
    if a.phase == "freeze":
        return freeze(layers)
    if a.phase == "steer":
        assert a.block, "--block B3 or B4"
        return steer(a.block, layers)
    verdict(layers)


if __name__ == "__main__":
    main()
