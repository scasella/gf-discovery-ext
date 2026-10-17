"""e70 -- SAE-gap round, geometric tier (PREREG_sae_gap.md, hashed before any fresh text is read).

Primary units: the e69 survivors #1 and #2, as FIXED D2 filters/patterns (results/e69_p0_units.npz). The e69
instrument (error, kept rows, demean3, PCA-256, multi-lag AMUSE, screens) is imported from the frozen e69 script.

Phases (in prereg order; nothing reads fresh text before the user's go-ahead)
  extract --set X0|D3|S3|R3   (MPS) X0 = D2 re-extraction identity check; D3 = L11/L12/L13 + entropy/log p;
                               S3 = D3 shuffled (e60 rule); R3 = L12. fp16 kept rows + fp32 sidecar of coord 1038.
  decode                      (.venv) D3 token pieces, one token at a time (as e64c)
  parse                       (.venv-parse) e64c's lexical + parse_rules on D3 with en_core_web_trf
  geo                         (CPU) G1-G5 for the primary units on D3/S3, C1 knockout, C2' planted, exploratory arm
  rep                         (CPU) G1 and G3 on R3; the geometric verdict
"""
import argparse, gc, hashlib, importlib.util, json, os, subprocess, sys, time
from pathlib import Path

import numpy as np

EXT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXT / "src"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_HOME", str(EXT / "hf_home"))
PREREG = "PREREG_sae_gap.md"
SELF = "scripts/e70_sae_gap_geometric.py"
N_TOK = 255
SETS = dict(D3=(2000, 3000), R3=(3000, 4000), B5=(4000, 5000), B6=(5000, 6000), C=(6000, 7000))
USED_CORPUS = [(0, 2000), (13000, 15000)]
LAYERS = dict(X0=[12], D3=[11, 12, 13], S3=[12], R3=[12])
PRIMARY = (1, 2)
SWAP_STOP_GB = 12.0
G1_ACF, G2_RATIO, G2_COS, G2_LOGE, G3_COS, G3_CAP, G3_TOPK = 0.20, 0.50, 0.80, 0.25, 0.80, 0.80, 8
NAME_RHO, NAME_AUROC, NAME_R2, TAIL_MIN_TYPES, TAIL_MIN_DOCS, ES_MIN = 0.30, 0.65, 0.30, 20, 20, 0.50
N_CAP_NULL, SEED_CAP, N_RANDE, SEED_RANDE = 1000, 7040, 32, 7020
KO_LATENTS = [2682, 7780, 4001, 4493, 1559, 8951, 3503, 1442, 507, 1056, 5254, 1229, 1994, 1534, 564, 1882, 7302, 10306, 398, 1550]
KO_FLOOR, KO_PERM = 0.70, 1000
PLANT_KAPPA, PLANT_RATE, PLANT_RUN, PLANT_SEED, PLANT_V_SEED, PLANT_PERM = (1.0, 2.0, 4.0), 0.06, 16.0, 7030, 7031, 100
COORD_MAX = 0.25                                   # single-coordinate screen (judge 1 graft 4): share of pattern norm^2
FROZEN_DEPS = ["scripts/e69_sae_gap_pilot.py", "scripts/e64c_context_rules.py", "scripts/e64_occurrence_v2_discovery.py",
               "scripts/e65_steering_redesign.py", "scripts/e66_confirm_l12_family.py", "scripts/e67_prior_feature_check.py",
               "results/corpus_tokens_pass1.i32", "results/reserve_tokens.i32", "NOTE_sae_gap_pilot.md"]
E64C_RULES = ("A", "B", "C", "C_inat", "D", "E", "F", "F_multi", "G", "H", "I", "J", "K", "L", "M", "N")


def OUT(name):
    return f"results/e70_{name}"


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, EXT / rel)
    mod = importlib.util.module_from_spec(spec); sys.modules[name] = mod; spec.loader.exec_module(mod)
    return mod


def mods():
    e69 = _load("e69", "scripts/e69_sae_gap_pilot.py")
    e67, e64 = e69.mods()
    return e69, e64


def t():
    return f"[{time.time() - T0:.0f}s]"


def assert_unhashed(*rels):
    """An output may be written only if no line of docs/HASHES.txt names its path (no regenerate-after-hash)."""
    text = (EXT / "docs/HASHES.txt").read_text()
    for rel in rels:
        assert f"  {rel} (" not in text, f"{rel} already has a HASHES line -- it may not be regenerated"


def coord_share(pat):
    """Largest single-coordinate share of the pattern's norm^2, and its coordinate."""
    w = np.asarray(pat, np.float64) ** 2
    return float(w.max() / w.sum()), int(w.argmax())


def swap_gb():
    out = subprocess.run(["sysctl", "vm.swapusage"], capture_output=True, text=True).stdout
    import re
    m = re.search(r"used = ([\d.]+)M", out)
    return float(m.group(1)) / 1024 if m else 0.0


def guard_swap():
    s = swap_gb()
    assert s <= SWAP_STOP_GB, f"swap {s:.1f} GB > {SWAP_STOP_GB} GB -- stopped cleanly; rerun the phase when memory frees"


# ---------------------------------------------------------------------------- data sets
def corpus():
    return np.fromfile(EXT / "results/corpus_tokens_pass1.i32", dtype=np.int32).reshape(-1, N_TOK)


def kept_sequences(name):
    """Absolute corpus indices of a new set after the duplicate screen (first 32 tokens), in prereg order."""
    a = corpus()
    r = np.fromfile(EXT / "results/reserve_tokens.i32", dtype=np.int32).reshape(-1, 256)[:, 1:]
    key = lambda x: x[:32].tobytes()
    used = set()
    for lo, hi in USED_CORPUS:
        used.update(key(a[i]) for i in range(lo, hi))
    used.update(key(x) for x in r)
    out, dropped = {}, {}
    for nm, (lo, hi) in SETS.items():
        keep, drop = [], []
        for i in range(lo, hi):
            k = key(a[i])
            (drop if k in used else keep).append(i)
            used.add(k)
        out[nm], dropped[nm] = keep, drop
        if nm == name:
            return np.asarray(keep, np.int64), drop
    raise KeyError(name)


def set_tokens(name):
    """(tokens [n_seq, 255], absolute seq ids, dropped). S3 = D3's sequences shuffled with e60's rule."""
    a = corpus()
    if name == "X0":
        ids = np.arange(1000, 2000)
        return a[ids].astype(np.int64), ids, []
    base = "D3" if name == "S3" else name
    ids, drop = kept_sequences(base)
    T = a[ids].astype(np.int64).copy()
    if name == "S3":
        for i, sid in enumerate(ids):
            T[i] = T[i][np.random.default_rng(600_000 + int(sid)).permutation(N_TOK)]
    return T, ids, drop


def meta_of(T, ids):
    return dict(tokens=T.reshape(-1), seq=np.repeat(ids, N_TOK), pos=np.tile(np.arange(1, N_TOK + 1), len(ids)))


def kept_rows(meta):
    _, inv, cnt = np.unique(meta["tokens"], return_inverse=True, return_counts=True)
    return np.flatnonzero((cnt[inv] >= 8) & (meta["pos"] >= 4))


# ---------------------------------------------------------------------------- extract (MPS)
def phase_extract(e69, e64, name):
    import torch
    import gfdisc.model as gm
    gm.MODEL_NAME = str(sorted((EXT / "hf_home/hub/models--google--gemma-3-1b-pt/snapshots").glob("*"))[0])
    from gfdisc.env import DEVICE
    from gfdisc.model import capture
    if name == "X0":
        e64.require_hashed("results/e67_raw_L12_rows.npy")
        assert_unhashed(OUT("x0.json"))
    else:
        assert_unhashed(*[OUT(f"{name}_{x}") for x in ("meta.npz", "1038.npz", "pred.npz")] + [OUT(f"{name}_L{l}.npy") for l in LAYERS[name]])
    if name in ("D3", "S3", "R3"):
        e64.require_hashed(OUT("x0.json"))
        assert json.loads((EXT / OUT("x0.json")).read_text())["pass"], "X0 failed -- STOP"
    if name == "R3":
        e64.require_hashed(OUT("geo.json"))
        assert json.loads((EXT / OUT("geo.json")).read_text())["r3_needed"], "no primary unit passed G1-G5 on D3 -- R3 is not extracted"
    done = OUT(f"{name}_meta.npz")                      # an existing but unhashed output is an incomplete run: overwritten
    guard_swap()
    T, ids, drop = set_tokens(name)
    meta = meta_of(T, ids)
    rows = kept_rows(meta)
    if name == "X0":
        rows_ref = np.load(EXT / "results/occ_units_v2_L12.npz")["row"].astype(np.int64)
        assert np.array_equal(rows, rows_ref), "X0 kept rows differ from D2's frozen rows"
    layers = LAYERS[name]
    model, tok = gm.load_model(token=None)
    H = {l: np.zeros((len(rows), 1152), np.float16) for l in layers}
    side = {l: np.zeros(len(rows), np.float32) for l in layers}
    want_pred = name == "D3"
    ent = np.zeros(len(meta["tokens"]), np.float32); lp = np.full(len(meta["tokens"]), np.nan, np.float32)
    with torch.inference_mode():
        for lo in range(0, len(T), 16):
            guard_swap()
            blk = T[lo:lo + 16]
            a, b = np.searchsorted(rows, [lo * N_TOK, (lo + len(blk)) * N_TOK])
            ids_t = np.concatenate([np.full((len(blk), 1), tok.bos_token_id), blk], 1)
            with capture(model, layers) as st:
                model(torch.from_numpy(ids_t).to(DEVICE), use_cache=False)
            for l in layers:
                x = st[l][:, 1:, :].float().reshape(-1, 1152)[torch.from_numpy(rows[a:b] - lo * N_TOK).to(DEVICE)]
                assert bool(torch.isfinite(x).all()), f"non-finite residual L{l}"
                side[l][a:b] = x[:, 1038].cpu().numpy()
                H[l][a:b] = x.cpu().numpy().astype(np.float16)
            del st
            if want_pred:
                for k in range(0, len(blk), 2):                          # logits two sequences at a time
                    sub = torch.from_numpy(ids_t[k:k + 2]).to(DEVICE)
                    lg = model(sub, use_cache=False).logits[:, 1:, :].float()
                    ls = torch.log_softmax(lg, -1)
                    e_ = -(ls.exp() * ls).sum(-1)
                    nxt = torch.from_numpy(blk[k:k + 2, 1:]).to(DEVICE)
                    l_ = ls[:, :-1, :].gather(-1, nxt[..., None])[..., 0]
                    s0 = (lo + k) * N_TOK; n2 = sub.shape[0] * N_TOK
                    ent[s0:s0 + n2] = e_.reshape(-1).cpu().numpy()
                    lpb = np.full((sub.shape[0], N_TOK), np.nan, np.float32); lpb[:, :-1] = l_.cpu().numpy()
                    lp[s0:s0 + n2] = lpb.reshape(-1)
                    del lg, ls, e_, l_
            if (lo // 16) % 10 == 0:
                print(t(), f"{name}: {lo + len(blk)}/{len(T)} sequences; swap {swap_gb():.1f} GB", flush=True)
    del model
    if DEVICE == "mps":
        torch.mps.empty_cache()
    over = {l: int((np.abs(side[l]) >= 65504).sum()) for l in layers}
    if name == "X0":
        ref = np.load(EXT / "results/e67_raw_L12_rows.npy")
        HD = H[12]
        a32, b32 = HD.astype(np.float32), ref.astype(np.float32)
        ac, bc = a32 - a32.mean(0), b32 - b32.mean(0)
        cc = (ac * bc).sum(0) / np.sqrt((ac * ac).sum(0) * (bc * bc).sum(0) + 1e-30)
        rec = dict(share_identical=float((HD == ref).mean()), max_abs_diff=float(np.abs(a32 - b32).max()),
                   min_coord_corr=float(np.nanmin(cc)))
        rec["pass"] = bool(rec["share_identical"] >= 0.99 and rec["min_coord_corr"] >= 0.999)
        e64.dump(rec, OUT("x0.json")); e64.append_hash(OUT("x0.json"), "e70 X0: D2 re-extraction identity check")
        print(t(), "X0", rec, flush=True)
        return
    for l in layers:
        np.save(EXT / OUT(f"{name}_L{l}.npy"), H[l])
    np.savez(EXT / OUT(f"{name}_1038.npz"), **{f"L{l}": side[l] for l in layers})
    if want_pred:
        assert np.isfinite(ent).all()
        np.savez(EXT / OUT(f"{name}_pred.npz"), entropy=ent, logp=lp)
    np.savez(EXT / done, rows=rows, dropped=np.asarray(drop, np.int64), overflow=json.dumps(over), **meta)
    for l in layers:
        e64.append_hash(OUT(f"{name}_L{l}.npy"), f"e70 extract {name}: raw L{l} kept rows, fp16")
    e64.append_hash(OUT(f"{name}_1038.npz"), f"e70 extract {name}: fp32 sidecar of coordinate 1038")
    if want_pred:
        e64.append_hash(OUT(f"{name}_pred.npz"), f"e70 extract {name}: next-token entropy and log p")
    e64.append_hash(done, f"e70 extract {name}: meta (tokens, seq, pos, kept rows, dropped duplicates)")
    print(t(), f"{name}: {len(ids)} sequences ({len(drop)} dropped as duplicates), {len(rows)} kept rows; 1038 >= 65504: {over}", flush=True)


# ---------------------------------------------------------------------------- decode / parse (D3 labels)
def phase_decode(e64):
    e64.require_hashed(OUT("D3_meta.npz"))
    assert_unhashed(OUT("D3_pieces.json"))
    e61 = _load("e61", "scripts/e61_occurrence_screen.py")
    tk = e61.token_strings()
    toks = np.load(EXT / OUT("D3_meta.npz"))["tokens"].astype(np.int64)
    cache, seqs = {}, []
    for s0 in range(0, len(toks), N_TOK):
        row = []
        for x in toks[s0:s0 + N_TOK]:
            x = int(x)
            if x not in cache:
                cache[x] = tk.decode([x])
            row.append(cache[x])
        seqs.append(row)
    (EXT / OUT("D3_pieces.json")).write_text(json.dumps(seqs))
    e64.append_hash(OUT("D3_pieces.json"), "e70 decode: D3 token pieces")
    print(f"decoded {len(seqs)} D3 sequences", flush=True)


def phase_parse():
    """Runs in .venv-parse. e64c's lexical and parse_rules, unchanged; the per-sequence loop is e64c.parse's."""
    import spacy
    e64c = _load("e64c", "scripts/e64c_context_rules.py")
    nlp = spacy.load("en_core_web_trf")
    seqs = json.loads((EXT / OUT("D3_pieces.json")).read_text())
    n = len(seqs) * N_TOK
    out = {k: np.zeros(n, np.uint8) for k in E64C_RULES}
    texts, starts = [], []
    for pieces in seqs:
        texts.append("".join(pieces)); starts.append(np.cumsum([0] + [len(p) for p in pieces[:-1]]))
    t0 = time.time()
    for s, doc in enumerate(nlp.pipe(texts, batch_size=32)):
        pieces, base = seqs[s], s * N_TOK
        for k, v in e64c.lexical(pieces).items():
            out[k][base:base + N_TOK] = v
        char2tok = np.full(len(doc.text) + 1, -1)
        for tt in doc:
            char2tok[tt.idx:tt.idx + len(tt.text)] = tt.i
        tok_of = np.full(N_TOK, -1)
        for j, p in enumerate(pieces):
            k = next((starts[s][j] + q for q, ch in enumerate(p) if not ch.isspace()), None)
            if k is not None:
                tok_of[j] = char2tok[k]
        S = e64c.parse_rules(doc)
        for j in range(N_TOK):
            i = tok_of[j]
            if i < 0:
                continue
            for k in S:
                if i in S[k]:
                    out[k][base + j] = 1
            if j == 0 or tok_of[j - 1] != i:
                q = i - 1
                while q >= 0 and doc[q].is_space:
                    q -= 1
                out["C"][base + j] = q >= 0 and doc[q].pos_ == "ADP"
        if s % 100 == 0:
            print(f"  trf: {s}/{len(seqs)} sequences, {time.time() - t0:.0f}s", flush=True)
    np.savez_compressed(EXT / OUT("D3_rules_trf.npz"), **out)
    print(f"parsed D3 in {time.time() - t0:.0f}s", flush=True)


# ---------------------------------------------------------------------------- geometric helpers (CPU)
def load_set(e69, name, layer=12):
    m = np.load(EXT / OUT(f"{name}_meta.npz"))
    meta = dict(tokens=m["tokens"].astype(np.int64), seq=m["seq"].astype(np.int64), pos=m["pos"].astype(np.int64))
    S = e69.RowSet(meta, m["rows"])
    H = np.load(EXT / OUT(f"{name}_L{layer}.npy"))
    side = np.load(EXT / OUT(f"{name}_1038.npz"))[f"L{layer}"]
    return S, H, side, meta


def h32(H16, side, a=0, b=None):
    """float32 rows; coordinate 1038 restored from the fp32 sidecar wherever fp16 overflowed."""
    x = H16[a:b].astype(np.float32)
    bad = ~np.isfinite(x[:, 1038])
    if bad.any():
        x[bad, 1038] = side[a:b][bad]
    assert np.isfinite(x).all()
    return x


def errors_with_parts(e69, H16, P, side, ko=None):
    """e (kept coords), SAE(h) (kept coords), state; optionally e_KO = e + sum_{k in ko} z_k W_dec[k] (CPU)."""
    import torch
    We, be, th, Wd, bd = [torch.from_numpy(P[k].astype(np.float32)) for k in ("w_enc", "b_enc", "threshold", "w_dec", "b_dec")]
    kp = torch.from_numpy(e69.KEEP); n = len(H16)
    E = np.empty((n, 1151), np.float32); R = np.empty((n, 1151), np.float32); EK = np.empty((n, 1151), np.float32) if ko else None
    lh, le, l0 = np.empty(n), np.empty(n), np.empty(n)
    se2, sh2, sh1 = 0.0, 0.0, torch.zeros(1151, dtype=torch.float64)
    kk = torch.tensor(ko) if ko else None
    with torch.no_grad():
        for a in range(0, n, 8192):
            x = torch.from_numpy(h32(H16, side, a, a + 8192))
            pre = x @ We + be; z = pre * (pre > th); rec = z @ Wd + bd
            e = (x - rec)[:, kp]
            E[a:a + 8192] = e.numpy(); R[a:a + 8192] = rec[:, kp].numpy()
            if ko:
                EK[a:a + 8192] = (e + (z[:, kk] @ Wd[kk])[:, kp]).numpy()
            se2 += float((e.double() ** 2).sum()); sh2 += float((x[:, kp].double() ** 2).sum()); sh1 += x[:, kp].double().sum(0)
            lh[a:a + 8192] = torch.log(torch.linalg.norm(x[:, kp], dim=1)).numpy()
            le[a:a + 8192] = torch.log(torch.linalg.norm(e, dim=1)).numpy()
            l0[a:a + 8192] = (z > 0).sum(1).numpy()
    fvu = se2 / (sh2 - float((sh1 ** 2).sum()) / n)          # sum ||e||^2 / sum ||h - mean h||^2 (kept coordinates)
    return E, R, EK, dict(log_h=lh, log_e=le, L0=l0, fvu=fvu)


def capture_of(pat_u, F, rng_seed):
    """Subspace capture of p_u by the set's top-8 AMUSE patterns, and its null p99.9 over random pullback directions."""
    Q, _ = np.linalg.qr(F["pat"][:, :G3_TOPK])
    c = float(((Q.T @ pat_u) ** 2).sum() / (pat_u @ pat_u))
    g = np.random.default_rng(rng_seed).standard_normal((F["U"].shape[1], N_CAP_NULL))
    D = F["U"] @ (F["sd"][:, None] * g); D /= np.linalg.norm(D, axis=0, keepdims=True)
    cn = ((Q.T @ D) ** 2).sum(0)
    return c, float(np.quantile(cn, 0.999))


def acf_any(S, y, L):
    """lag-L autocorrelation over within-sequence kept-row pairs (any L; e69's RowSet precomputes 1-16 only)."""
    if L in S.pairs:
        return S.acf(y, L)
    n = len(S.rows); tgt = S.rows - L
    j = np.clip(np.searchsorted(S.rows, tgt), 0, n - 1)
    ok = (S.rows[j] == tgt) & ((S.rows % N_TOK) >= L)
    return float(np.corrcoef(y[np.flatnonzero(ok)], y[j[ok]])[0, 1])


def tail_diversity(y, S):
    k = int(np.ceil(0.01 * len(y))); o = np.argsort(y)
    res = {}
    for side, idx in (("top", o[-k:]), ("bottom", o[:k])):
        res[side] = dict(types=int(len(np.unique(S.tok[idx]))), docs=int(len(np.unique(S.seq[idx]))))
    return res


def naming(e69, S, y, feats_pred, nonfinal, Sn, battery, rules, bag_X, state, mlp=True):
    from scipy.stats import spearmanr
    from sklearn.metrics import roc_auc_score
    rec = {}
    rho = {k: float(spearmanr(y[nonfinal], v[nonfinal])[0]) for k, v in feats_pred.items()}
    rho.update({k + "_idrem": float(spearmanr(y[nonfinal], Sn.demean3(v[nonfinal].astype(np.float32)))[0]) for k, v in feats_pred.items()})
    rec["predictability"] = dict(rho=rho, named=bool(max(abs(v) for v in rho.values()) >= NAME_RHO))
    pos = {k: float(spearmanr(y, v)[0]) for k, v in battery["continuous"].items()}
    rec["position"] = dict(rho=pos, named=bool(max(abs(v) for v in pos.values()) >= NAME_RHO))
    au = {}
    for k, v in list(battery["binary"].items()) + [(f"e64c_{k}", v) for k, v in rules.items()]:
        b = v > 0.5
        if b.sum() < 50 or (~b).sum() < 50:
            continue
        a = roc_auc_score(b, y); au[k] = float(max(a, 1 - a))
    rec["labels"] = dict(top5=dict(sorted(au.items(), key=lambda kv: -kv[1])[:5]), named=bool(max(au.values()) >= NAME_AUROC))
    r2 = {}
    for W, X in bag_X.items():
        r2[str(W)] = {m: e69.heldout_r2(X, y.astype(np.float64), S.even, m) for m in (("ridge", "mlp") if mlp else ("ridge",))}
    rec["bag"] = dict(r2=r2, named=bool(max(max(v.values()) for v in r2.values()) >= NAME_R2))
    st = {k: float(spearmanr(y, S.demean3(v))[0]) for k, v in state.items() if k != "fvu"}
    rec["sae_state"] = dict(rho=st, named=bool(max(abs(v) for v in st.values()) >= NAME_RHO))
    td = tail_diversity(y, S)
    rec["tail_diversity"] = dict(td, named=bool(any(v["types"] < TAIL_MIN_TYPES or v["docs"] < TAIL_MIN_DOCS for v in td.values())))
    rec["named"] = bool(any(rec[k]["named"] for k in ("predictability", "position", "labels", "bag", "sae_state", "tail_diversity")))
    return rec


def es_split(S, hf, rf, ef):
    """ES = Cov(y_e, y_raw)/Var(y_raw), with y_raw = y_e + y_sae, each identity-removed."""
    y_raw, y_e = S.demean3(hf.astype(np.float32)).astype(np.float64), S.demean3(ef.astype(np.float32)).astype(np.float64)
    y_sae = S.demean3(rf.astype(np.float32)).astype(np.float64)
    resid = float(np.abs(y_raw - (y_e + y_sae)).max() / (np.abs(y_raw).max() + 1e-12))
    return float(np.cov(y_e, y_raw)[0, 1] / np.var(y_raw, ddof=1)), resid


def markov_plant(seq_ids, rng_seed):
    p10 = 1.0 / PLANT_RUN; p01 = PLANT_RATE * p10 / (1 - PLANT_RATE)
    out = np.zeros((len(seq_ids), N_TOK), np.float32)
    for i, sid in enumerate(seq_ids):
        g = np.random.default_rng(rng_seed + int(sid)); s = g.random() < PLANT_RATE
        u = g.random(N_TOK)
        for j in range(N_TOK):
            out[i, j] = s
            s = (u[j] >= p10) if s else (u[j] < p01)
    return out.reshape(-1)


def screened_units(e69, S, F, state, m=32):
    st = {k: v for k, v in state.items() if k != "fvu"}
    return [u for u in e69.screens(S, dict(Y=F["Y"][:, :m], eig=F["eig"], rep=F["rep"][:m]), st, m=m) if u["screened"]]


VERDICT_ORDER = ("NOT REPRODUCED", "PERSISTENT BUT GENERIC", "NAMED", "SAE-SPAN", "UNDETERMINED (inflation guard)")


def d3_labels(r):
    """Every failure label that applies on D3/S3 (prereg section 3); the headline is the first in VERDICT_ORDER."""
    lab = []
    if not (r["G1"]["pass_"] and r["G3"]["pass_"]):
        lab.append("NOT REPRODUCED")
    if r["G2"]["pass_"] is False:
        lab.append("PERSISTENT BUT GENERIC")
    if r["G4"]["named"]:
        lab.append("NAMED")
    if not r["G5"]["pass_"]:
        lab.append("SAE-SPAN")
    if r["G2"]["pass_"] is None:
        lab.append("UNDETERMINED (inflation guard)")
    return lab


def d3_verdict(r):
    if r["d3_pass"]:
        return "D3 PASS (R3 pending)"
    lab = d3_labels(r)
    return next(v for v in VERDICT_ORDER if v in lab)


# ---------------------------------------------------------------------------- geo (CPU)
def phase_geo(e69, e64):
    import scipy.sparse as sp
    from sklearn.metrics import roc_auc_score
    for nm in ("D3", "S3"):
        e64.require_hashed(OUT(f"{nm}_meta.npz"), OUT(f"{nm}_L12.npy"), OUT(f"{nm}_1038.npz"))
    e64.require_hashed(OUT("D3_pred.npz"), OUT("D3_pieces.json"), OUT("D3_rules_trf.npz"), "results/e69_p0_units.npz", "results/e69_p0.json")
    assert_unhashed(OUT("geo_primary.json"), OUT("geo.json"), OUT("geo_maps.npz"))
    z0 = np.load(EXT / "results/e69_p0_units.npz")
    filt = {u: z0["filt"][:, u].astype(np.float64) for u in PRIMARY}; pat = {u: z0["pat"][:, u].astype(np.float64) for u in PRIMARY}
    PB, PSm = e69.sae_params(e69.SAE_BIG), e69.sae_params(e69.SAE_SMALL)
    out = dict(primary={}, exploratory={}, controls={}, sets={})
    # ---------------- D3
    guard_swap()
    D, HD, sideD, mD = load_set(e69, "D3")
    E, R, _, stD = errors_with_parts(e69, HD, PB, sideD)
    acf_d2 = {u: json.loads((EXT / "results/e69_p0.json").read_text())["units"][u]["acf1"] for u in PRIMARY}
    y = {u: D.demean3((E @ filt[u]).astype(np.float32)).astype(np.float64) for u in PRIMARY}
    FD = e69.fit_instrument(D, E, 64); del FD["V"], FD["Z"]; gc.collect()
    gR = np.random.default_rng(SEED_RANDE).standard_normal((FD["U"].shape[1], N_RANDE))
    Fm = np.stack([filt[u] for u in PRIMARY] + [FD["U"] @ (gR[:, i] / FD["sd"]) for i in range(N_RANDE)], 1).astype(np.float32)
    pE, pR = E @ Fm, R @ Fm; del R; gc.collect()
    yD2var = {u: float(np.var(z0["Y"][:, u])) for u in PRIMARY}
    print(t(), f"D3: {len(D.rows)} kept rows; refit eig top4 {np.round(FD['eig'][:4], 3).tolist()}", flush=True)
    # ---------------- S3
    S3, HS, sideS, _ = load_set(e69, "S3")
    ES_, RS_, _, stS = errors_with_parts(e69, HS, PB, sideS); del HS, RS_
    FS = e69.fit_instrument(S3, ES_, 64); del FS["V"], FS["Z"]; gc.collect()
    yS = {u: S3.demean3((ES_ @ filt[u]).astype(np.float32)).astype(np.float64) for u in PRIMARY}
    infl = float(stS["log_e"].mean() - stD["log_e"].mean())
    ovf = {nm: json.loads(str(np.load(EXT / OUT(f"{nm}_meta.npz"))["overflow"])) for nm in ("D3", "S3")}
    out["sets"] = dict(overflow_1038=ovf, D3=dict(n_rows=int(len(D.rows)), eig_top8=[float(x) for x in FD["eig"][:8]], mean_log_e=float(stD["log_e"].mean()),
                               mean_log_h=float(stD["log_h"].mean()), mean_L0=float(stD["L0"].mean()), fvu=stD["fvu"]),
                       S3=dict(n_rows=int(len(S3.rows)), eig_top8=[float(x) for x in FS["eig"][:8]], mean_log_e=float(stS["log_e"].mean()),
                               mean_log_h=float(stS["log_h"].mean()), mean_L0=float(stS["L0"].mean()), fvu=stS["fvu"]),
                       log_e_inflation=infl, inflation_guard_ok=bool(infl < G2_LOGE))
    # ---------------- naming inputs (D3)
    pred = np.load(EXT / OUT("D3_pred.npz"))
    nonfinal = (D.rows % N_TOK) != N_TOK - 1
    feats_pred = {}
    for W in e69.PRED_W:
        sfx = "" if W == 1 else f"_mean{W}"
        feats_pred["entropy" + sfx] = e69.causal_mean(pred["entropy"], D.rows, W, True)
        feats_pred["neglogp" + sfx] = e69.causal_mean(-pred["logp"], D.rows, W, True)
    Sn = e69.RowSet(mD, D.rows[nonfinal])
    e61 = _load("e61", "scripts/e61_occurrence_screen.py")
    bat = e61.feature_battery(dict(row=D.rows), e61.token_strings(), mD["tokens"], mD["seq"])
    punct = np.zeros(len(D.rows), np.float32)                          # tokens since the last punctuation piece
    tk = e61.token_strings(); strs = {}; cnt = 0; ri = {int(r): i for i, r in enumerate(D.rows)}
    for r, x in enumerate(mD["tokens"]):
        if r % N_TOK == 0:
            cnt = 0
        s = strs.setdefault(int(x), tk.decode([int(x)]))
        if r in ri:
            punct[ri[r]] = cnt
        cnt = 0 if (s.strip() and not any(c.isalnum() for c in s.strip())) else cnt + 1
    battery = dict(continuous=dict(line_pos=bat["line_pos"], sent_pos=bat["sent_pos"], since_punct=punct),
                   binary={k: v for k, v in bat.items() if k not in ("line_pos", "sent_pos")})
    Rz = np.load(EXT / OUT("D3_rules_trf.npz")); rules = {k: Rz[k][D.rows] for k in E64C_RULES}
    Hk = HD[:, e69.KEEP].astype(np.float64)
    ut, inv = np.unique(D.tok, return_inverse=True)
    Mt = sp.csr_matrix((np.ones(len(inv)), (inv, np.arange(len(inv)))), shape=(len(ut), len(inv)))
    TM = (Mt @ Hk) / np.asarray(Mt.sum(1)); del Hk
    TMc = TM - TM.mean(0); _, s_, vt = np.linalg.svd(TMc, full_matrices=False)
    emb = TMc @ vt[:128].T / s_[:128] * np.sqrt(len(ut))
    lut = np.full(int(D.tok_all.max()) + 1, -1, np.int64); lut[ut] = np.arange(len(ut))
    li = lut[D.tok_all]; Efull = np.zeros((len(D.tok_all), 128)); Efull[li >= 0] = emb[li[li >= 0]]
    bag_X = {}
    for W in e69.BAG_W:
        X = D.demean3(e69.causal_mean(Efull, D.rows, W, False).astype(np.float32)).astype(np.float64)
        bag_X[W] = ((X - X[D.even].mean(0)) / np.maximum(X[D.even].std(0), 1e-9)).astype(np.float32)
    del Efull
    # ---------------- primary units
    Hk32 = np.concatenate([h32(HD, sideD, a, a + 65536)[:, e69.KEEP] @ Fm for a in range(0, len(HD), 65536)])
    Esm, Rsm, _, _ = errors_with_parts(e69, HD, PSm, sideD)
    pEs, pRs = Esm @ Fm[:, :len(PRIMARY)], Rsm @ Fm[:, :len(PRIMARY)]; del Esm, Rsm; gc.collect()
    hf = {u: Hk32[:, i] for i, u in enumerate(PRIMARY)}
    for u in PRIMARY:
        rec = {}
        acf = {L: acf_any(D, y[u], L) for L in (1, 2, 4, 8, 16, 32)}
        rec["G1"] = dict(acf=acf, acf1_d2=acf_d2[u], var_ratio_vs_d2=float(np.var(y[u]) / yD2var[u]), pass_=bool(acf[1] >= G1_ACF))
        aS, aD = S3.acf(yS[u], 1), acf[1]; vS, vD = S3.acov(yS[u], 1), D.acov(y[u], 1)
        cosS = np.abs(FS["pat"].T @ pat[u])
        rec["G2"] = dict(acf_ratio=float(aS / aD), acov_ratio=float(vS / vD), var_ratio=float(np.var(yS[u]) / np.var(y[u])),
                         s3_refit_max_cos=float(cosS.max()), inflation_guard_ok=out["sets"]["inflation_guard_ok"])
        g2 = aS / aD < G2_RATIO and vS / vD < G2_RATIO and cosS.max() < G2_COS
        rec["G2"]["pass_"] = bool(g2) if out["sets"]["inflation_guard_ok"] else None
        cosD = np.abs(FD["pat"].T @ pat[u]); cap, capn = capture_of(pat[u], FD, SEED_CAP)
        msh, mco = coord_share(FD["pat"][:, int(cosD.argmax())])
        match = bool(cosD.max() >= G3_COS and msh <= COORD_MAX)
        rec["G3"] = dict(refit_max_cos=float(cosD.max()), refit_unit=int(cosD.argmax()), refit_coord_share=msh, refit_coord=mco,
                         capture=cap, capture_null_p999=capn, pass_=bool(match or (cap >= G3_CAP and cap > capn)))
        csh, cco = coord_share(pat[u]); fc = int(np.argmax(np.abs(filt[u] * pat[u])))
        yc = D.demean3((E[:, fc] * filt[u][fc]).astype(np.float32)).astype(np.float64)
        rec["coordinates"] = dict(pattern_max_share=csh, pattern_max_coord=cco, top_filter_coord=fc,
                                  acov1_share_top_coord=float(D.acov(yc, 1) / D.acov(y[u], 1)))
        rec["G4"] = naming(e69, D, y[u], feats_pred, nonfinal, Sn, battery, rules, bag_X, stD, mlp=True)
        i = PRIMARY.index(u)
        es_b, res_b = es_split(D, hf[u], pR[:, i], pE[:, i])
        es_s, res_s = es_split(D, hf[u], pRs[:, i], pEs[:, i])
        rec["G5"] = dict(es_big=es_b, es_small=es_s, split_residual=max(res_b, res_s), pass_=bool(es_b >= ES_MIN and es_s >= ES_MIN))
        rec["d3_pass"] = bool(rec["G1"]["pass_"] and rec["G2"]["pass_"] is True and rec["G3"]["pass_"]
                              and not rec["G4"]["named"] and rec["G5"]["pass_"])
        out["primary"][str(u)] = rec
        print(t(), f"unit {u}: G1 acf1 {acf[1]:.3f} | G2 acf {aS / aD:.2f} acov {vS / vD:.2f} cos {cosS.max():.2f} | G3 cos {cosD.max():.2f} "
              f"cap {cap:.2f}/{capn:.2f} | G4 named {rec['G4']['named']} | G5 ES {es_b:.2f}/{es_s:.2f} | D3 pass {rec['d3_pass']}", flush=True)
    # RAND-E error shares (reported)
    k0 = len(PRIMARY)
    out["rande_acf1_d3"] = [acf_any(D, D.demean3(pE[:, k0 + i].astype(np.float32)).astype(np.float64), 1) for i in range(N_RANDE)]
    rES = [es_split(D, Hk32[:, k0 + i], pR[:, k0 + i], pE[:, k0 + i])[0] for i in range(N_RANDE)]
    out["rande_es_big"] = dict(p50=float(np.median(rES)), p01=float(np.quantile(rES, 0.01)), p99=float(np.quantile(rES, 0.99)))
    del Hk32, pE, pR, pEs, pRs
    e64.dump(dict(primary=out["primary"], sets=out["sets"], rande_es_big=out["rande_es_big"], rande_acf1_d3=out["rande_acf1_d3"]),
             OUT("geo_primary.json"))
    e64.append_hash(OUT("geo_primary.json"), "e70 geo: primary-unit D3/S3 results, sealed before the controls run")
    # ---------------- exploratory arm (D3 refit, m = 32)
    expl = screened_units(e69, D, FD, stD)
    for uu in expl:
        j = uu["id"]
        if any(abs(FD["pat"][:, j] @ pat[u]) >= G3_COS for u in PRIMARY):
            continue
        fj = FD["filt"][:, j]
        yj = FD["Y"][:, j].astype(np.float64)
        yjS = S3.demean3((ES_ @ fj).astype(np.float32)).astype(np.float64)
        rec = dict(screen=uu, coord_share=coord_share(FD["pat"][:, j])[0], coord_screen_pass=bool(coord_share(FD["pat"][:, j])[0] <= COORD_MAX),
                   G2=dict(acf_ratio=float(S3.acf(yjS, 1) / D.acf(yj, 1)), acov_ratio=float(S3.acov(yjS, 1) / D.acov(yj, 1)),
                                      s3_refit_max_cos=float(np.abs(FS["pat"].T @ FD["pat"][:, j]).max())),
                   G4=naming(e69, D, yj, feats_pred, nonfinal, Sn, battery, rules, bag_X, stD, mlp=False))
        out["exploratory"][str(j)] = rec
    print(t(), f"exploratory: D3 screened {[u['id'] for u in expl]}; non-primary {list(out['exploratory'])}", flush=True)
    del ES_, FS; gc.collect()
    # ---------------- C1 knockout control
    e63 = _load("e63", "scripts/e63_occurrence_v2.py")
    _, _, EK, stK = errors_with_parts(e69, HD, PB, sideD, ko=KO_LATENTS)
    stK["log_e"] = np.log(np.linalg.norm(EK, axis=1))           # the control's own state (as C2' uses the planted state)
    FK = e69.fit_instrument(D, EK, 32); del EK, FK["V"], FK["Z"]; gc.collect()
    sk = screened_units(e69, D, FK, stK)
    c1 = dict(screened=[u["id"] for u in sk], labels={})
    if sk:
        Yk = FK["Y"][:, [u["id"] for u in sk]].astype(np.float64)
        eng = e63.null_engine({}, {"amuse": Yk}, {k: bat[k] for k in ("paren_inside", "quote_inside")}, D.tinv, n_perm=KO_PERM)
        for lb, r in eng.items():
            obs = r["obs"][1]["amuse"]; nul = [n_[1]["amuse"].max() for n_ in r["null"]]
            p999 = float(np.quantile(nul, 0.999))
            c1["labels"][lb] = dict(max_auroc=float(obs.max()), unit=int(sk[int(obs.argmax())]["id"]), null_p999=p999,
                                    recovered=bool(obs.max() >= KO_FLOOR and obs.max() > p999))
    c1["recovered"] = bool(any(v["recovered"] for v in c1["labels"].values()))
    c1["ko_unit_patterns"] = {lb: FK["pat"][:, v["unit"]].tolist() for lb, v in c1["labels"].items() if v["recovered"]}
    c1["ko_unit_filters"] = {lb: FK["filt"][:, v["unit"]].tolist() for lb, v in c1["labels"].items() if v["recovered"]}
    out["controls"]["C1"] = c1
    out["exploratory_gated_by_C1"] = dict(c1_recovered=c1["recovered"],
                                          note="exploratory units are reported; without C1 recovery no exploratory statement is made")
    print(t(), "C1", {k: (round(v["max_auroc"], 3), round(v["null_p999"], 3), v["recovered"]) for k, v in c1["labels"].items()}, flush=True)
    del FK; gc.collect()
    # ---------------- C2' planted in h, re-encoded (descriptive)
    seqs_u = mD["seq"][::N_TOK]                                  # meta order: every sequence, kept rows or not
    paths = markov_plant(seqs_u, PLANT_SEED).reshape(len(seqs_u), N_TOK)
    plant = paths.reshape(-1)[D.rows]
    null_plants = [paths[np.random.default_rng(9000 + k).permutation(len(seqs_u))].reshape(-1)[D.rows] for k in range(PLANT_PERM)]
    rng = np.random.default_rng(PLANT_V_SEED)
    v_in = FD["U"] @ rng.standard_normal(FD["U"].shape[1]); v_in /= np.linalg.norm(v_in)   # within D3's top-256 error PCs
    v_fr = rng.standard_normal(1151); v_fr /= np.linalg.norm(v_fr)
    c2 = {}
    Hf32 = np.concatenate([h32(HD, sideD, a, a + 65536) for a in range(0, len(HD), 65536)])
    for vn, v in (("in_top256", v_in), ("full_rank", v_fr)):
        v1152 = np.zeros(1152, np.float32); v1152[e69.KEEP] = v
        sig = float(np.std(D.demean3((Hf32 @ v1152).astype(np.float32))))
        for kap in PLANT_KAPPA:
            Hp = (Hf32 + (kap * sig * plant)[:, None] * v1152[None, :]).astype(np.float16)
            sp_ = np.where(np.isfinite(Hp[:, 1038].astype(np.float32)), 0, Hf32[:, 1038]).astype(np.float32)
            Ep, RP, _, stp = errors_with_parts(e69, Hp, PB, sp_); del Hp, RP
            Fp = e69.fit_instrument(D, Ep, 32); del Fp["V"], Fp["Z"]
            pv = kap * sig * plant
            absorb = 1 - float(np.cov(((Ep - E) @ v).astype(np.float64), pv)[0, 1] / np.var(pv, ddof=1))
            scr = screened_units(e69, D, Fp, stp)
            best = 0.0; bestu = None
            for uu in scr:
                a = roc_auc_score(plant > 0.5, Fp["Y"][:, uu["id"]]); a = max(a, 1 - a)
                if a > best:
                    best, bestu = a, uu["id"]
            nul = []
            for pp in null_plants:
                m_ = 0.0
                for uu in scr:
                    a = roc_auc_score(pp > 0.5, Fp["Y"][:, uu["id"]]); m_ = max(m_, a, 1 - a)
                nul.append(m_)
            c2[f"{vn}_k{kap:g}"] = dict(screened=len(scr), best_auroc=best, unit=bestu, null_max=float(max(nul)) if nul else None,
                                       recovered=bool(best >= KO_FLOOR and (not nul or best > max(nul))), absorption=absorb)
            del Ep, Fp; gc.collect()
            print(t(), f"C2' {vn} kappa {kap}: screened {len(scr)} best {best:.3f} recovered {c2[f'{vn}_k{kap:g}']['recovered']} absorption {absorb:.2f}", flush=True)
    out["controls"]["C2_planted"] = c2
    # ---------------- maps for later phases
    np.savez(EXT / OUT("geo_maps.npz"), mu=FD["mu"], U=FD["U"], sd=FD["sd"], W=FD["W"], eig=FD["eig"], filt=FD["filt"], pat=FD["pat"],
             rep=FD["rep"], Y=FD["Y"][:, :32])
    for u, r in out["primary"].items():
        r["d3_verdict"] = d3_verdict(r); r["d3_labels"] = d3_labels(r)
    out["r3_needed"] = bool(any(r["d3_pass"] for r in out["primary"].values()))
    e64.dump(out, OUT("geo.json"))
    e64.append_hash(OUT("geo_maps.npz"), "e70 geo: D3 refit maps (for coverage and steering)")
    e64.append_hash(OUT("geo.json"), f"e70 geo: primary G1-G5 on D3/S3, C1, C2', exploratory arm; R3 needed {out['r3_needed']}")
    print(t(), f"GEO done; primary D3 pass {[u for u, r in out['primary'].items() if r['d3_pass']]}; R3 needed {out['r3_needed']}", flush=True)


# ---------------------------------------------------------------------------- rep (CPU)
def phase_rep(e69, e64):
    e64.require_hashed(OUT("geo.json"), OUT("geo_maps.npz"), OUT("R3_meta.npz"), OUT("R3_L12.npy"), OUT("R3_1038.npz"), "results/e69_p0_units.npz")
    assert_unhashed(OUT("verdict.json"))
    geo = json.loads((EXT / OUT("geo.json")).read_text())
    z0 = np.load(EXT / "results/e69_p0_units.npz")
    guard_swap()
    R, HR, sideR, _ = load_set(e69, "R3")
    E, RR, _, stR = errors_with_parts(e69, HR, e69.sae_params(e69.SAE_BIG), sideR); del HR, RR
    FR = e69.fit_instrument(R, E, 64); del FR["V"], FR["Z"]
    out = dict(primary={}, exploratory={}, n_rows=int(len(R.rows)))
    for u in PRIMARY:
        g = geo["primary"][str(u)]
        yR = R.demean3((E @ z0["filt"][:, u]).astype(np.float32)).astype(np.float64)
        cos = np.abs(FR["pat"].T @ z0["pat"][:, u]); cap, capn = capture_of(z0["pat"][:, u].astype(np.float64), FR, SEED_CAP)
        msh = coord_share(FR["pat"][:, int(cos.argmax())])[0]
        rec = dict(G1=dict(acf1=R.acf(yR, 1), acf={L: acf_any(R, yR, L) for L in (1, 2, 4, 8, 16, 32)},
                           var_ratio_vs_d2=float(np.var(yR) / np.var(z0["Y"][:, u])), pass_=bool(R.acf(yR, 1) >= G1_ACF)),
                   G3=dict(refit_max_cos=float(cos.max()), refit_coord_share=msh, capture=cap, capture_null_p999=capn,
                           pass_=bool((cos.max() >= G3_COS and msh <= COORD_MAX) or (cap >= G3_CAP and cap > capn))))
        rec["verdict"] = ("REPLICATED CONTEXT-SPECIFIC ERROR COMPONENT" if g["d3_pass"] and rec["G1"]["pass_"] and rec["G3"]["pass_"]
                          else (g["d3_verdict"] if not g["d3_pass"] else "NOT REPRODUCED"))
        out["primary"][str(u)] = rec
    gm = np.load(EXT / OUT("geo_maps.npz"))
    for j in geo["exploratory"]:
        cos = np.abs(FR["pat"].T @ gm["pat"][:, int(j)])
        out["exploratory"][j] = dict(r3_refit_max_cos=float(cos.max()), r3_match_descriptive=bool(cos.max() >= G3_COS))
    gR = np.random.default_rng(SEED_RANDE).standard_normal((FR["U"].shape[1], N_RANDE))
    out["rande_acf1_r3"] = [R.acf(R.demean3((E @ (FR["U"] @ (gR[:, i] / FR["sd"]))).astype(np.float32)).astype(np.float64), 1) for i in range(N_RANDE)]
    e64.dump(out, OUT("verdict.json"))
    e64.append_hash(OUT("verdict.json"), "e70 rep: R3 replication and the geometric verdict")
    print(t(), "VERDICT", {u: r["verdict"] for u, r in out["primary"].items()}, flush=True)


def main():
    global T0
    T0 = time.time()
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True, choices=["extract", "decode", "parse", "hashparse", "geo", "rep"])
    ap.add_argument("--set", choices=["X0", "D3", "S3", "R3"])
    a = ap.parse_args()
    print("e70 sha256", hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), flush=True)
    if a.phase == "parse":                     # .venv-parse: check the freeze with hashlib only, then parse
        hashes = (EXT / "docs/HASHES.txt").read_text()
        for rel in (PREREG, SELF, OUT("D3_pieces.json"), "scripts/e64c_context_rules.py"):
            assert hashlib.sha256((EXT / rel).read_bytes()).hexdigest() in hashes, f"{rel} not hashed"
        assert f"  {OUT('D3_rules_trf.npz')} (" not in hashes, "parse output already hashed"
        phase_parse(); return
    e69, e64 = mods()
    e64.require_hashed(PREREG, SELF, *FROZEN_DEPS)
    e64.check_integrity()                      # pins e60, e61, e63, e64 and the v2 inputs (e64.FROZEN)
    if a.phase == "extract":
        assert a.set, "--set X0|D3|S3|R3"
        phase_extract(e69, e64, a.set)
    elif a.phase == "decode":
        phase_decode(e64)
    elif a.phase == "hashparse":
        assert_unhashed(OUT("D3_rules_trf.npz"))
        z = np.load(EXT / OUT("D3_rules_trf.npz"))
        n = len(np.load(EXT / OUT("D3_meta.npz"))["tokens"])
        assert sorted(z.files) == sorted(E64C_RULES) and all(len(z[k]) == n for k in z.files)
        e64.append_hash(OUT("D3_rules_trf.npz"), "e70 parse: e64c rules on D3 (en_core_web_trf)")
    elif a.phase == "geo":
        phase_geo(e69, e64)
    else:
        phase_rep(e69, e64)


T0 = time.time()
if __name__ == "__main__":
    main()
