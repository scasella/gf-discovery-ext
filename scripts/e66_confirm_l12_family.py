"""e66 -- preregistered confirmation of the L12 logcosh #13 / skew #16 candidate family
(PREREG_occurrence_v2_confirm.md, sha256 asserted), plus a descriptive account of what the direction
does when steered.

The tier-2 rule is e65's, unchanged (e65 imported as a frozen module, sha asserted): dose c*sigma_d with
c in {0.5, 1, 2, 4} and the gate at c* = 2; 99 pullback-matched random directions (RAND); the shuffled-
signature control (SIG); sign consistency >= 0.8 at some c <= c*; Delta > 0. Only the contexts are new.

Phases (run in this order, each once)
  selftest  synthetic checks of e66's own helpers (no data, no model)
  freeze    (CPU) block C sites (reserve 6000-6999), block D sites (reserve 7000-7999), the block-C RAND
            bank; hashed before any forward pass
  confirm   (MPS) block C at L12: unsteered pass and sigma; the 99 RAND passes; the quote anchor (logcosh
            #33) at every dose and its gates; ONLY if it passes, the primary (logcosh #13) and the
            secondary (skew #16) at every dose and their gates; the verdict
  describe  (MPS + CPU, after the confirm output is hashed; descriptive only) block D: promoted and
            demoted output tokens and per-category probability mass at c = -2, +2, +4; steered and
            unsteered greedy continuations; D2: top natural contexts, next-token and position profiles,
            the battery associations and the e64c rule scores of both family members
"""
import argparse, hashlib, importlib.util, json, os, sys, time
from pathlib import Path

import numpy as np

EXT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXT / "src"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")

PREREG = EXT / "PREREG_occurrence_v2_confirm.md"
E65_REL = "scripts/e65_steering_redesign.py"
FROZEN = {                                  # asserted at every data phase (prereg section 2)
    E65_REL: "5e37cab244e820337c7edd02f3f43ce0f30afa3fe4186e8444154a530aba5e86",
    "PREREG_occurrence_v2_steering.md": "d0b7f5eae5ff32394de0a00925f47daade424599377d39e17c08ca4400d1d1d1",
    "results/e65_frozen.json": "bd856844dae5db63d19518063d65eedf25ff8aaeee66565b09743f2da2fb2be9",
    "results/e65_steer_B3.json": "a77c0893425ceab2f085cffc78f32c2fce3df7bd7b2789d68ecbba39dc02e1ee",
    "results/e65_steer_B4.json": "15b04685a3fad65f549c9d02d8a485f70820b921e540d8f4967293fb0e3e1c97",
    "results/e65_verdict.json": "a9a2628b3706dc57a69a116c33655a3055cc409b98431c2921e34efe45ed2aa1",
    "results/occ_units_v2_L12.npz": "b84a7b13d52bd54acdc665175c7972b4e9418659a3e6b56d47bdc3bb895b5d5b",
    "results/e64c_rules_en_core_web_trf.npz": "0dc1b07a4965455655ccb5bf117a664d1c40d7056f4e43f542cb5925e686be4d",
    "results/e64c_rules_en_core_web_sm.npz": "5c1a228e8abbbbe0a725f81ff04e38aab62b64efadcb427df5d34b047cb41270",
}
L = 12
ANCHOR, PRIMARY, SECONDARY = ("logcosh", 33), ("logcosh", 13), ("skew", 16)
BLOCK_C = dict(reserve=(6000, 7000), site_seed=6700, rand_seed=6710)
BLOCK_D = dict(reserve=(7000, 8000), site_seed=6800)
DESC_DOSES = (-2.0, 2.0, 4.0)
GEN_DOSES, GEN_PROMPTS, GEN_NEW = (0.0, 2.0, 4.0, -4.0), 8, 24
P_FLOOR, TOP_K = 1e-4, 30
OUT = dict(frozen="results/e66_frozen.json", rand="results/e66_rand_L12.npz", confirm="results/e66_confirm.json",
           describe="results/e66_describe.json")
SUBORD = {"before", "after", "when", "while", "since", "until", "till", "because", "as", "if", "though",
          "although", "whenever", "once", "unless", "whereas"}
TEMPORAL = {"again", "today", "yesterday", "tomorrow", "now", "then", "later", "earlier", "soon", "ago", "already",
            "still", "yet", "recently", "currently", "times", "twice", "couple", "few", "next", "last", "every",
            "during", "ever", "never", "always", "tonight", "week", "year", "years", "day", "days", "month"}
PREP = {"in", "on", "at", "for", "by", "with", "to", "of", "from", "without", "into", "over", "under", "through",
        "about", "around", "between", "within", "across", "against", "toward", "towards", "per"}
FUNC = {"the", "a", "an", "this", "that", "these", "those", "my", "your", "his", "her", "its", "our", "their",
        "i", "you", "he", "she", "it", "we", "they", "me", "him", "us", "them", "is", "are", "was", "were", "be",
        "been", "and", "or", "but", "so", "not", "no"}
CATS = ("digit", "punct", "space", "subordinator", "temporal", "preposition", "function", "capitalised", "word", "other")


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


_E65 = None


def load_e65():
    global _E65
    if _E65 is None:
        assert sha(EXT / E65_REL) == FROZEN[E65_REL], "e65 changed"
        spec = importlib.util.spec_from_file_location("e65", EXT / E65_REL)
        mod = importlib.util.module_from_spec(spec); sys.modules["e65"] = mod; spec.loader.exec_module(mod)
        _E65 = mod
    return _E65


def check_integrity():
    e65 = load_e65()
    hashes = (EXT / "docs/HASHES.txt").read_text()
    assert sha(PREREG) in hashes, "PREREG_occurrence_v2_confirm.md does not match its sha256 in docs/HASHES.txt"
    assert sha(__file__) in PREREG.read_text(), "e66 is not the frozen version (its sha256 is not in the prereg)"
    for rel, h in FROZEN.items():
        assert sha(EXT / rel) == h, f"{rel} changed"
    e65.check_integrity()                     # e65's prereg, its inputs and e64's integrity chain


# ---------------------------------------------------------------------------- pure helpers (selftest)
def draw_sites(blk, layer=L):
    """e65.draw_sites' rule for a new block: N_SITES distinct sequences, one position each in [5, 255]."""
    e65 = load_e65()
    lo, hi = blk["reserve"]
    rng = np.random.default_rng(blk["site_seed"] + layer)
    seqs = lo + rng.choice(hi - lo, e65.N_SITES, replace=False)
    pos = rng.integers(5, 256, size=e65.N_SITES)
    return [[int(s), int(p)] for s, p in zip(seqs, pos)]


def category(s):
    """A coarse category for a decoded token string (descriptive only)."""
    st = s.strip()
    if not st:
        return "space"
    lo = st.lower()
    if all(c.isdigit() or c in ".,%" for c in st) and any(c.isdigit() for c in st):
        return "digit"
    if not any(c.isalnum() for c in st):
        return "punct"
    if lo in SUBORD:
        return "subordinator"
    if lo in TEMPORAL:
        return "temporal"
    if lo in PREP:
        return "preposition"
    if lo in FUNC:
        return "function"
    if st[:1].isupper():
        return "capitalised"
    if st.isalpha():
        return "word"
    return "other"


def verdict_of(anchor_pass, primary_pass, secondary_pass):
    """Prereg section 6: the anchor gates the verdict; the primary decides it; the secondary is reported."""
    if not anchor_pass:
        return dict(verdict="NO VERDICT (the L12 quote anchor failed on block C)", family_both=False)
    return dict(verdict="CONFIRMED" if primary_pass else "NOT CONFIRMED",
                family_both=bool(primary_pass and secondary_pass))


class VocabAcc:
    """Accumulates, over sites, the mean unsteered probability and the mean change in log-probability and
    in probability of every vocabulary token."""
    def __init__(self, V):
        self.n, self.p0, self.dlogp, self.dp = 0, np.zeros(V), np.zeros(V), np.zeros(V)

    def add(self, logp0, logp1, chunk=32):
        for a in range(0, len(logp0), chunk):
            l0, l1 = logp0[a:a + chunk].astype(np.float64), logp1[a:a + chunk].astype(np.float64)
            self.n += len(l0)
            self.p0 += np.exp(l0).sum(0); self.dlogp += (l1 - l0).sum(0)
            self.dp += (np.exp(l1) - np.exp(l0)).sum(0)

    def summary(self, dec, top_k=TOP_K, floor=P_FLOOR):
        p0, dl, dp = self.p0 / self.n, self.dlogp / self.n, self.dp / self.n
        ok = np.flatnonzero(p0 >= floor)
        o = ok[np.argsort(-dl[ok])]
        q = np.argsort(-dp)
        rows = lambda idx: [dict(token=dec(int(i)), id=int(i), mean_p0=float(p0[i]), mean_dlogp=float(dl[i]),
                                 mean_dp=float(dp[i])) for i in idx]
        cats = {}
        for i in np.flatnonzero(p0 >= 1e-7):
            c = category(dec(int(i)))
            a = cats.setdefault(c, dict(p0=0.0, dp=0.0))
            a["p0"] += float(p0[i]); a["dp"] += float(dp[i])
        return dict(n_sites=self.n, floor=floor, n_above_floor=int(len(ok)),
                    promoted_dlogp=rows(o[:top_k]), demoted_dlogp=rows(o[::-1][:top_k]),
                    promoted_dp=rows(q[:top_k]), demoted_dp=rows(q[::-1][:top_k]), category_mass=cats)


def selftest():
    ok = []
    c, d = np.asarray(draw_sites(BLOCK_C)), np.asarray(draw_sites(BLOCK_D))
    ok.append(("sites", len(set(c[:, 0])) == 200 and c[:, 0].min() >= 6000 and c[:, 0].max() < 7000
               and d[:, 0].min() >= 7000 and d[:, 0].max() < 8000 and c[:, 1].min() >= 5 and c[:, 1].max() <= 255
               and np.array_equal(c, np.asarray(draw_sites(BLOCK_C)))))
    ok.append(("categories", [category(s) for s in (" 12", ",", "\n", " before", " again", " in", " the", " Paris",
                                                    " house", "é")]
               == ["digit", "punct", "space", "subordinator", "temporal", "preposition", "function", "capitalised",
                   "word", "word"]))
    ok.append(("verdict", verdict_of(False, True, True)["verdict"].startswith("NO VERDICT")
               and verdict_of(True, True, False) == dict(verdict="CONFIRMED", family_both=False)
               and verdict_of(True, False, True) == dict(verdict="NOT CONFIRMED", family_both=False)
               and verdict_of(True, True, True)["family_both"]))
    rng = np.random.default_rng(0)
    V = 50
    x0 = rng.normal(size=(7, V)); x1 = x0.copy(); x1[:, 3] += 2.0
    ls = lambda x: x - np.log(np.exp(x).sum(1, keepdims=True))
    acc = VocabAcc(V); acc.add(ls(x0[:4]), ls(x1[:4])); acc.add(ls(x0[4:]), ls(x1[4:]))
    s = acc.summary(lambda i: f"t{i}", top_k=3, floor=0.0)
    ok.append(("vocab acc", s["n_sites"] == 7 and s["promoted_dlogp"][0]["id"] == 3 and s["promoted_dp"][0]["id"] == 3
               and abs(sum(v["dp"] for v in s["category_mass"].values())) < 1e-9))
    for name, v in ok:
        print(f"{'ok ' if v else 'FAIL'} {name}")
    assert all(v for _, v in ok), "selftest failed"
    print(f"selftest: all {len(ok)} checks pass")


# ---------------------------------------------------------------------------- freeze (CPU)
def freeze():
    e65 = load_e65(); e64 = e65.load_e64()
    assert not (EXT / OUT["frozen"]).exists(), "the freeze runs once"
    _, _, z = e64.load_maps(L)
    keep = e65.keep_mask(e64.e61.drop_for(L))
    np.savez_compressed(EXT / OUT["rand"], C=e65.rand_bank(z["P_U"], z["P_sd"], keep, e65.N_RAND, BLOCK_C["rand_seed"] + L))
    frz = json.loads((EXT / "results/e65_frozen.json").read_text())["layers"][str(L)]
    for u in (ANCHOR, PRIMARY, SECONDARY):
        assert frz["signatures"][f"{u[0]}/{u[1]}"]["T"], f"{u} has an empty signature"
    out = dict(layer=L, sites=dict(C=draw_sites(BLOCK_C), D=draw_sites(BLOCK_D)), blocks=dict(C=BLOCK_C, D=BLOCK_D),
               rand_npz=OUT["rand"], rand_sha256=sha(EXT / OUT["rand"]),
               signatures_from="results/e65_frozen.json", signatures_sha256=FROZEN["results/e65_frozen.json"],
               family=dict(anchor=ANCHOR, primary=PRIMARY, secondary=SECONDARY))
    e64.dump(out, OUT["frozen"])
    e64.append_hash(OUT["rand"], "e66 freeze: block-C RAND bank, before any forward")
    e64.append_hash(OUT["frozen"], "e66 freeze: block C and D sites, before any forward")
    print("freeze done", flush=True)


# ---------------------------------------------------------------------------- model helpers (MPS)
def load_model():
    import gfdisc.model as gm
    gm.MODEL_NAME = str(sorted((EXT / "hf_home/hub/models--google--gemma-3-1b-pt/snapshots").glob("*"))[0])
    from gfdisc.env import DEVICE
    model, tok = gm.load_model(token=None)
    return model, tok, DEVICE


def make_forward(model, tok, DEVICE, prompts, union, Dsel, layer=L):
    """e65.steer's forward, verbatim in effect: steer h <- h + alpha*d at the output of decoder layer `layer`,
    at the last prompt position only; return the last-position logits on `union` (float64), the mean
    realised dose, and optionally the full last-position log-softmax to a callback."""
    import torch
    e65 = load_e65()
    uni = torch.tensor(union, device=DEVICE)
    Dsel_t = torch.tensor(np.asarray(Dsel), dtype=torch.float32, device=DEVICE)

    def forward(d=None, alpha=0.0, cap=None, full=None):
        outs, real = [], []
        for lo in range(0, len(prompts), e65.BATCH):
            blk = prompts[lo:lo + e65.BATCH]
            Lm = max(len(x) for x in blk)
            ids = torch.full((len(blk), Lm), tok.pad_token_id, dtype=torch.long)
            att = torch.zeros_like(ids)
            for i, x in enumerate(blk):
                ids[i, :len(x)] = torch.from_numpy(np.asarray(x, dtype=np.int64)); att[i, :len(x)] = 1
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
                    real.append(((new[ib, lp].float() - t[ib, lp].float()).norm(dim=-1) / abs(alpha)).cpu())
                    return (new,) + tuple(o[1:]) if isinstance(o, tuple) else new
                hooks.append(model.model.layers[layer].register_forward_hook(fn))
            if cap is not None:
                def cfn(_m, _i, o, ids=ids, att=att):
                    t = (o[0] if isinstance(o, tuple) else o).float()
                    pos = torch.arange(t.shape[1])[None, :].expand(len(ids), -1)
                    m = (att.bool() & (pos >= e65.POS_MIN)).numpy()
                    cap["proj"].append((t @ Dsel_t.T).cpu().numpy()[m])
                    cap["ids"].append(ids.numpy()[m])
                    cap["hnorm"].append(t[ib, lp].norm(dim=-1).cpu().numpy())
                hooks.append(model.model.layers[layer].register_forward_hook(cfn))
            try:
                with torch.no_grad():
                    lg = model(input_ids=ids.to(DEVICE), attention_mask=att.to(DEVICE), use_cache=False).logits
                last_lg = lg[ib, lp].float()
                outs.append(last_lg[:, uni].cpu())
                if full is not None:
                    full(torch.log_softmax(last_lg, -1).cpu().numpy().astype(np.float32))
            finally:
                for h in hooks:
                    h.remove()
        return torch.cat(outs, 0).numpy().astype(np.float64), (float(torch.cat(real).mean()) if real else None)
    return forward


def generate(model, DEVICE, prompt, d, alpha, n_new, layer=L):
    """Greedy continuation; when alpha != 0 the push alpha*d is added at the site position and at every
    generated position (descriptive only; no cache, the whole sequence is recomputed each step)."""
    import torch
    seq = list(int(t) for t in prompt); start = len(seq) - 1
    Dv = torch.tensor(d, dtype=torch.float32, device=DEVICE)
    for _ in range(n_new):
        ids = torch.tensor([seq], dtype=torch.long, device=DEVICE)
        h = None
        if alpha != 0.0:
            def fn(_m, _i, o):
                t = o[0] if isinstance(o, tuple) else o
                new = t.float().clone(); new[0, start:] = new[0, start:] + alpha * Dv
                new = new.to(t.dtype)
                return (new,) + tuple(o[1:]) if isinstance(o, tuple) else new
            h = model.model.layers[layer].register_forward_hook(fn)
        try:
            with torch.no_grad():
                nxt = int(model(input_ids=ids, use_cache=False).logits[0, -1].argmax())
        finally:
            if h is not None:
                h.remove()
        seq.append(nxt)
    return seq[len(prompt):]


def setup(block_sites, tok, rv):
    prompts = [np.concatenate([[tok.bos_token_id], rv[s, max(1, p - load_e65().CTX):p + 1]]) for s, p in block_sites]
    site_tok = np.asarray([int(rv[s, p]) for s, p in block_sites])
    return prompts, site_tok


# ---------------------------------------------------------------------------- confirm (MPS, block C)
def confirm():
    e65 = load_e65(); e64 = e65.load_e64()
    e64.require_hashed(OUT["frozen"], OUT["rand"])
    assert not (EXT / OUT["confirm"]).exists(), "confirm runs once"
    frz = json.loads((EXT / OUT["frozen"]).read_text())
    assert sha(EXT / OUT["rand"]) == frz["rand_sha256"]
    sigs65 = json.loads((EXT / "results/e65_frozen.json").read_text())["layers"][str(L)]["signatures"]
    rl, twin = e65.roles(e64, L)
    sigs = {(k.split("/")[0], int(k.split("/")[1])): dict(T=s["T"], R=s["R"]) for k, s in sigs65.items()}
    assert set(sigs) == set(rl)
    assert rl[ANCHOR] == "anchor" and rl[PRIMARY] == "candidate" and rl[SECONDARY] == "candidate"
    assert twin.get(PRIMARY) == SECONDARY
    _, _, z = e64.load_maps(L)
    dirs = {k: z[f"dir_{k[0]}"][k[1]] for k in (ANCHOR, PRIMARY, SECONDARY)}
    RB = np.load(EXT / OUT["rand"])["C"]
    out = dict(block="C", reserve=list(BLOCK_C["reserve"]), swap_gb_at_start=e64.check_swap())
    t0 = time.time()
    model, tok, DEVICE = load_model()
    rv = np.fromfile(EXT / "results/reserve_tokens.i32", dtype=np.int32).reshape(-1, 256)
    prompts, site_tok = setup(frz["sites"]["C"], tok, rv)
    union = sorted({t for s in sigs.values() for t in s["T"] + s["R"]})
    col = {t: i for i, t in enumerate(union)}
    cols = lambda ts: np.asarray([col[t] for t in ts], dtype=np.int64)
    units = (ANCHOR, PRIMARY, SECONDARY)
    forward = make_forward(model, tok, DEVICE, prompts, union, np.vstack([dirs[k] for k in units] + [RB]))
    # 1. unsteered pass: logits and sigma of the three units and of every RAND direction
    cap = dict(proj=[], ids=[], hnorm=[])
    X0, _ = forward(cap=cap)
    proj = np.concatenate(cap["proj"]); pids = np.concatenate(cap["ids"]); hnorm = np.concatenate(cap["hnorm"])
    sig_tot, sig_wit, sinfo = e65.natural_sigma(proj, pids, np.ones(len(proj), bool))
    sigma = {k: float(sig_tot[i]) for i, k in enumerate(units)}
    sigma_w = {k: float(sig_wit[i]) for i, k in enumerate(units)}
    sig_rand = sig_tot[len(units):]
    h_mean = float(hnorm.mean())
    out.update(sigma_info=sinfo, site_hnorm_mean=h_mean,
               sigma_rand=dict(median=float(np.median(sig_rand)), min=float(sig_rand.min()), max=float(sig_rand.max())))
    base_f = {k: e65.lse(X0, cols(sigs[k]["T"])) - e65.lse(X0, cols(sigs[k]["R"])) for k in sigs if sigs[k]["T"]}
    # 2. RAND bank at c*, each scored with the three units' own signatures
    rand = np.full((len(RB), len(units)), np.nan); rr_all = []
    for i, d in enumerate(RB):
        X, rr = forward(d, e65.C_STAR * float(sig_rand[i])); rr_all.append(rr)
        for j, k in enumerate(units):
            rand[i, j] = float(e65.readout(X, X0, cols(sigs[k]["T"]), cols(sigs[k]["R"])).mean())
    out["rand_realised_dose"] = float(np.mean(rr_all))
    out["rand_cos_to_units"] = float(np.abs(RB @ np.vstack([dirs[k] for k in units]).T).max())
    print(f"[{time.time() - t0:.0f}s] C base + RAND bank done", flush=True)

    def evaluate(key, role):
        u = sigs[key]
        rec = dict(kind=key[0], id=key[1], role=role, sigma=sigma[key], sigma_within_type=sigma_w[key],
                   eps_equiv=e65.C_STAR * sigma[key] / h_mean, ratio_to_old_push=e65.C_STAR * sigma[key] / (e65.OLD_EPS * h_mean))
        Tc, Rc = cols(u["T"]), cols(u["R"])
        row = {}
        for c in e65.DOSES:
            X, rr = forward(dirs[key], c * sigma[key])
            dl = e65.readout(X, X0, Tc, Rc)
            rec[f"c={c}"] = dict(mean_delta=float(dl.mean()), sign_consistency=e65.sign_consistency(dl, site_tok),
                                 realised_dose=rr, z=float(dl.mean() / base_f[key].std()))
            if c == e65.C_STAR:
                S, dropped = e65.sig_sets(key, sigs, twin.get(key))
                row = {f"{v[0]}/{v[1]}": float(e65.readout(X, X0, cols(s["T"]), cols(s["R"])).mean()) for v, s in S.items()}
        d_uu = rec[f"c={e65.C_STAR}"]["mean_delta"]
        rrow = rand[:, units.index(key)]
        sc = {c: rec[f"c={c}"]["sign_consistency"] for c in e65.SC_DOSES}
        top = sorted(row.items(), key=lambda x: -x[1])[:3]
        rec.update(rand=dict(max=float(rrow.max()), p50=float(np.median(rrow)), p99=float(np.quantile(rrow, 0.99)), n=len(rrow)),
                   sig=dict(max=max(row.values()), n=len(row), dropped=[f"{v[0]}/{v[1]}" for v in dropped],
                            twin_excluded=(f"{twin[key][0]}/{twin[key][1]}" if key in twin else None), top3=top),
                   margin_rand=d_uu / float(rrow.max()) if rrow.max() > 0 else None,
                   margin_sig=d_uu / max(row.values()) if max(row.values()) > 0 else None,
                   **e65.gates(d_uu, sc, rrow, list(row.values())))
        return rec

    out["anchor"] = evaluate(ANCHOR, "anchor")
    print(f"[{time.time() - t0:.0f}s] C anchor {ANCHOR} PASS={out['anchor']['PASS']}", flush=True)
    if out["anchor"]["PASS"]:
        out["primary"] = evaluate(PRIMARY, "primary")
        print(f"[{time.time() - t0:.0f}s] C primary {PRIMARY} PASS={out['primary']['PASS']}", flush=True)
        out["secondary"] = evaluate(SECONDARY, "secondary")
        print(f"[{time.time() - t0:.0f}s] C secondary {SECONDARY} PASS={out['secondary']['PASS']}", flush=True)
    out.update(verdict_of(out["anchor"]["PASS"], out.get("primary", {}).get("PASS", False),
                          out.get("secondary", {}).get("PASS", False)))
    out["seconds"] = round(time.time() - t0); out["swap_gb_at_end"] = e64.swap_used_gb()
    e64.dump(out, OUT["confirm"])
    e64.append_hash(OUT["confirm"], "e66 confirm (block C, reserve 6000-6999; does not amend e64 or e65)")
    print("VERDICT", out["verdict"], "| family_both", out["family_both"], flush=True)


# ---------------------------------------------------------------------------- describe (descriptive only)
def describe():
    e65 = load_e65(); e64 = e65.load_e64()
    e64.require_hashed(OUT["frozen"], OUT["confirm"])
    assert not (EXT / OUT["describe"]).exists(), "describe runs once"
    frz = json.loads((EXT / OUT["frozen"]).read_text())
    sigs65 = json.loads((EXT / "results/e65_frozen.json").read_text())["layers"][str(L)]["signatures"]
    _, _, z = e64.load_maps(L)
    dirs = {k: z[f"dir_{k[0]}"][k[1]] for k in (PRIMARY, SECONDARY)}
    out = dict(block="D", reserve=list(BLOCK_D["reserve"]), swap_gb_at_start=e64.check_swap(), descriptive_only=True)
    t0 = time.time()
    model, tok, DEVICE = load_model()
    dec_cache = {}
    dec = lambda i: dec_cache.setdefault(i, tok.decode([i]))
    rv = np.fromfile(EXT / "results/reserve_tokens.i32", dtype=np.int32).reshape(-1, 256)
    prompts, site_tok = setup(frz["sites"]["D"], tok, rv)
    sig = sigs65[f"{PRIMARY[0]}/{PRIMARY[1]}"]
    union = sorted(set(sig["T"] + sig["R"]))
    forward = make_forward(model, tok, DEVICE, prompts, union, np.vstack([dirs[PRIMARY], dirs[SECONDARY]]))
    # sigma on block D (the confirm rule)
    cap = dict(proj=[], ids=[], hnorm=[]); base = []
    X0, _ = forward(cap=cap, full=base.append)
    proj = np.concatenate(cap["proj"]); pids = np.concatenate(cap["ids"])
    sig_tot, _, _ = e65.natural_sigma(proj, pids, np.ones(len(proj), bool))
    sigma = {PRIMARY: float(sig_tot[0]), SECONDARY: float(sig_tot[1])}
    out["sigma"] = {f"{k[0]}/{k[1]}": v for k, v in sigma.items()}
    logp0 = np.concatenate(base, 0)
    # 1. output tokens and category mass at c = -2, +2, +4 (primary; site-only push, as in the test)
    V = logp0.shape[1]
    out["output_shift"] = {}
    for c in DESC_DOSES:
        acc = VocabAcc(V); chunks = []
        forward(dirs[PRIMARY], c * sigma[PRIMARY], full=chunks.append)
        acc.add(logp0, np.concatenate(chunks, 0))
        out["output_shift"][f"c={c}"] = acc.summary(dec)
        print(f"[{time.time() - t0:.0f}s] D output shift c={c} done", flush=True)
    # 2. greedy continuations, persistent push from the site on (primary)
    gens = []
    for i in range(GEN_PROMPTS):
        p = prompts[i]
        g = dict(site=frz["sites"]["D"][i], prompt=tok.decode([int(t) for t in p[1:]]))
        for c in GEN_DOSES:
            ids = generate(model, DEVICE, p, dirs[PRIMARY], c * sigma[PRIMARY], GEN_NEW)
            g[f"c={c}"] = tok.decode(ids)
        gens.append(g)
    out["continuations"] = gens
    print(f"[{time.time() - t0:.0f}s] D continuations done", flush=True)
    # 3. D2 (frozen scores): natural top contexts, next-token profile, position profile, battery, e64c rules
    out["d2"] = d2_profile(tok)
    out["seconds"] = round(time.time() - t0); out["swap_gb_at_end"] = e64.swap_used_gb()
    e64.dump(out, OUT["describe"])
    e64.append_hash(OUT["describe"], "e66 describe (block D, reserve 7000-7999, and D2; descriptive only)")
    print("describe done", flush=True)


def d2_profile(tok):
    """CPU, D2 only: for each family member, the heavy-tail contexts and what follows them, the mean score by
    sentence and line position, the top battery associations (e64_labels.json) and the e64c rule scores."""
    from scipy.stats import rankdata
    e65 = load_e65(); e64 = e65.load_e64()
    fnpz = np.load(EXT / "results/occ_units_v2_L12.npz")
    rows = fnpz["row"].astype(np.int64)
    toks = np.load(EXT / "cache/occ/occD2_meta.npz")["tokens"].astype(np.int64)
    _, _, z = e64.load_maps(L)
    labs = {f"{u['kind']}/{u['id']}": u for u in json.loads((EXT / "results/e64_labels.json").read_text())["layers"][str(L)]["units"]}
    D = dict(row=rows)
    feats = e64.e61.feature_battery(D, tok, toks, None)
    rules = {m: np.load(EXT / f"results/e64c_rules_{m}.npz") for m in ("en_core_web_trf", "en_core_web_sm")}
    dec = lambda ids: tok.decode([int(t) for t in ids])
    out = {}
    for kd, j in (PRIMARY, SECONDARY):
        y = fnpz[f"Y_{kd}"][:, j].astype(np.float64)
        s = float(z[f"sign_{kd}"][j]); ys = s * y                      # oriented to the heavy tail
        k = int(np.ceil(0.01 * len(y))); top = np.argsort(-ys)[:k]
        ctx = []
        for i in top[:30]:
            r = int(rows[i]); s0 = (r // 255) * 255
            ctx.append(dec(toks[max(s0, r - 20):r]) + "[[" + dec([toks[r]]) + "]]" + dec(toks[r + 1:min(s0 + 255, r + 4)]))
        nxt_ok = top[(rows[top] % 255) != 254]
        nxt_cat = np.asarray([category(tok.decode([int(toks[rows[i] + 1])])) for i in nxt_ok])
        all_ok = np.flatnonzero((rows % 255) != 254)
        samp = np.random.default_rng(0).choice(all_ok, 20000, replace=False)
        base_cat = np.asarray([category(tok.decode([int(toks[rows[i] + 1])])) for i in samp])
        cur_cat = np.asarray([category(tok.decode([int(toks[rows[i]])])) for i in top])
        base_cur = np.asarray([category(tok.decode([int(toks[rows[i]])])) for i in samp])
        prof = {}
        for f in ("sent_pos", "line_pos"):
            v = feats[f]; bins = [0, 1, 2, 4, 8, 16, 32, 10 ** 6]
            b = np.digitize(v, bins) - 1
            prof[f] = {f"[{bins[q]},{bins[q + 1]})": dict(n=int((b == q).sum()), mean_score_z=float(((ys - ys.mean()) / ys.std())[b == q].mean()),
                                                         tail_rate=float(np.isin(np.flatnonzero(b == q), top).mean()))
                       for q in range(len(bins) - 1) if (b == q).any()}
        R = rankdata(ys); o = np.argsort(ys)
        e64c = {m: {} for m in rules}
        # oriented AUROC (toward the heavy tail) and heavy-tail rate per e64c rule, both parses
        for m, F in rules.items():
            for rname in F.files:
                f = F[rname][rows].astype(bool)
                n1 = int(f.sum())
                if n1 < 50 or len(f) - n1 < 50:
                    e64c[m][rname] = None; continue
                auc = float((R[f].sum() - n1 * (n1 + 1) / 2) / (n1 * (len(f) - n1)))
                e64c[m][rname] = dict(auroc_oriented=auc, base=float(f.mean()), tail_rate=float(f[o[-k:]].mean()))
        lab = labs[f"{kd}/{j}"]
        assoc = sorted(((name, v.get("auroc", v.get("abs_spearman"))) for name, v in lab["associations"].items()),
                       key=lambda x: -x[1])[:10]
        tc = lambda arr: {c: float((arr == c).mean()) for c in CATS}
        out[f"{kd}/{j}"] = dict(heavy_sign=s, top_contexts=ctx, next_token_category=tc(nxt_cat), next_token_category_base=tc(base_cat),
                                current_token_category=tc(cur_cat), current_token_category_base=tc(base_cur),
                                position_profile=prof, battery_top10=assoc, battery_named=lab["named"],
                                e64c_rules=e64c, identity=lab["identity"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True, choices=["selftest", "freeze", "confirm", "describe"])
    a = ap.parse_args()
    print("e66 sha256", sha(__file__), flush=True)
    if a.phase == "selftest":
        return selftest()
    check_integrity()
    load_e65().load_e64().check_swap()
    dict(freeze=freeze, confirm=confirm, describe=describe)[a.phase]()


if __name__ == "__main__":
    main()
