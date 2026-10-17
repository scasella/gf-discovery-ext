"""e64 -- Stage 4 v2, discovery tier (PREREG_occurrence_v2_discovery.md, sha256 asserted).

Scope: the layers and unit kinds the v2 positive control validated -- L7 and L12, FastICA-64
logcosh and skew. No clusters, no L24. The frozen v2 units (results/occ_units_v2_L{L}.npz) and
screens (results/e63_occurrence_v2.json) are used unchanged; nothing here refits a D2 unit.

Phases (run in this order; each asserts its predecessors' outputs are in docs/HASHES.txt and
appends its own output's sha256 there)
  selftest      the extended battery's state machine on hand-written strings (no data)
  reconstruct   D2 re-extracted by e60: recompute the frozen representation, keep its maps, and
                accept the frozen units only if a least-squares readout of every screened unit's
                frozen scores from the recomputed W has R^2 >= 0.999. Writes the D2 maps, each
                unit's W-space pattern and its raw-space steering direction.
  label         extended battery on D2; NAMED / UNNAMED for every screened unit (battery route);
                twins (logcosh x skew, |cos| >= 0.8); per-unit identity diagnostics; contexts;
                a reading template for the battery-UNNAMED units.
  seal-reading  the analyst's context reading (UNNAMED -> NAMED_BY_CONTEXT only) is hashed.
  shuffled      S2 (D2 shuffled within sequences): D2 maps applied, independent ICA fits;
                presence of every D2 unit; S2's own screened battery-UNNAMED count; KIR.
  behaviour     per layer: anchor first; steering at reserve 3000-3999 sites (v1 section 9.2).
  replicate     R2 (reserve 2000-2999): D2 maps applied, independent ICA fits with the frozen
                v2 unit code; match |cos| >= 0.8, screens recomputed on R2, UNNAMED on R2;
                the control units are the replication anchors.
  verdict       the three tiers, families, multiplicity, KIR.
"""
import argparse, hashlib, importlib.util, json, os, re, subprocess, sys, time
from pathlib import Path

import numpy as np

EXT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXT / "src"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, EXT / rel)
    mod = importlib.util.module_from_spec(spec); sys.modules[name] = mod; spec.loader.exec_module(mod)
    return mod


e61 = _load("e61", "scripts/e61_occurrence_screen.py")
e63 = _load("e63", "scripts/e63_occurrence_v2.py")

PREREG = EXT / "PREREG_occurrence_v2_discovery.md"
ADDENDA = [EXT / "docs/stage4v2_discovery_addendum_D1_2026-10-15.md"]   # each hashed; may record a newer e64 sha
HASHES = EXT / "docs/HASHES.txt"
FROZEN = {                                  # asserted at every data phase
    "PREREG_occurrence_v2.md": "edf7575075663cf4477ed3fdcec5e8eaa85ae89ac9e4c0e924bcc93a370035ba",
    "scripts/e63_occurrence_v2.py": "6f822a439025e01b176ae446d96413dc736078d5dbe740d03f0f05052f6e4543",
    "scripts/e61_occurrence_screen.py": "43a445016e82735c70ca6f2c623917fa864fce6920317832c9f2da9035bf92e1",
    "scripts/e60_extract.py": "a6d99f92f1e21d613e80cceb594724dee9faed719131e5fc372d7e4f113cdd20",
    "results/e63_pilot_choice.json": "4c71cd1b25166465b7a776a66e828923026c78c303b31dd996b521faf589b5fa",
    "results/e63_occurrence_v2.json": "84fe64f4fd5c294a8d769bb760bfaef599fc0ed60817db74c14f3f509f06aa22",
    "results/occ_units_v2_L7.npz": "a8cf7ab2f874855840e760e1f5663e76032eabe1b1220929cf1dc2007f11e457",
    "results/occ_units_v2_L12.npz": "b84a7b13d52bd54acdc665175c7972b4e9418659a3e6b56d47bdc3bb895b5d5b",
}
LAYERS, KINDS = (7, 12), ("logcosh", "skew")
TAG = dict(D="occD2", S="occS2", R="occR2")
N_ICA, N_MIN, N0, N_ITER = 64, 32, 16, 3
ANCHOR = {7: ("logcosh", 12, "paren_inside"), 12: ("logcosh", 33, "quote_inside")}
CONTROL_LABELS = ("paren_inside", "quote_inside")
# reconstruction, twins, labelling (prereg sections 2, 5, 6)
RECON_R2 = 0.999
TWIN_COS = 0.8
TAIL = 0.01
NAME_AUROC, NAME_RHO, NAME_ENRICH, NAME_MIN_RATE, NAME_MIN_POS = 0.75, 0.5, 3.0, 0.10, 50
GRADED = ("line_pos", "sent_pos")
N_CTX = 12
IDENT_TDI, IDENT_SHARE = 3.0, 0.30
# S2 / R2 (sections 7, 9)
PRESENCE_COS, REPL_COS, CTRL_FLOOR = 0.8, 0.8, 0.70
N_NULL_DIRS = 10_000
# behaviour (section 8; v1 section 9.2 and e62 unchanged except contexts, direction and siblings)
EPS_GRID, EPS_NULL, N_OCC, N_RAND, CTX, BATCH, SC_MIN = (0.005, 0.01, 0.02), 0.02, 200, 10, 64, 32, 0.8
BEH_RESERVE = (3000, 4000)
SWAP_MAX_GB = 16.0                          # addendum D1 (was 8.0 in the prereg)
VOCAB_N = 262_144 + 16


# ---------------------------------------------------------------------------- integrity
def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def check_integrity():
    hashes = HASHES.read_text()
    assert sha(PREREG) in hashes, "PREREG_occurrence_v2_discovery.md does not match its sha256 in docs/HASHES.txt"
    for ad in ADDENDA:
        assert sha(ad) in hashes, f"{ad.name} does not match its sha256 in docs/HASHES.txt"
    assert any(sha(__file__) in d.read_text() for d in [PREREG] + ADDENDA), "e64 is not a frozen version"
    for rel, h in FROZEN.items():
        assert sha(EXT / rel) == h, f"{rel} changed since the v2 freeze"


def require_hashed(*rels):
    hashes = HASHES.read_text()
    for rel in rels:
        assert (EXT / rel).exists(), f"{rel} missing: run the earlier phase first"
        assert sha(EXT / rel) in hashes, f"{rel} is not hashed in docs/HASHES.txt (or changed after hashing)"


def append_hash(rel, note):
    line = f"{sha(EXT / rel)}  {rel} ({note}, {time.strftime('%Y-%m-%dT%H:%M:%S')})\n"
    with open(HASHES, "a") as f:
        f.write(line)
    print("HASHES +=", line.strip(), flush=True)


def swap_used_gb():
    out = subprocess.run(["sysctl", "vm.swapusage"], capture_output=True, text=True).stdout
    m = re.search(r"used = ([\d.]+)M", out)
    return float(m.group(1)) / 1024 if m else float("nan")


def check_swap():
    used = swap_used_gb()
    assert used <= SWAP_MAX_GB, f"swap used {used:.1f} GB > {SWAP_MAX_GB} GB: free memory or reboot first"
    return used


def dump(obj, rel):
    (EXT / rel).write_text(json.dumps(obj, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)))


def frozen_v2():
    return json.loads((EXT / "results/e63_occurrence_v2.json").read_text())


def screened_ids(fr, kind):
    return [u["id"] for u in fr["units"] if u["kind"] == f"ica_{kind}" and u["screened"]]


def control_units(res, L):
    """The frozen recovering units at L: [(kind, id, label, |AUROC|)]."""
    pc = res["layers"][str(L)]["positive_control"]
    out = []
    for lb in CONTROL_LABELS:
        for kd in KINDS:
            r = pc[lb][kd]
            if r["recovered"]:
                out.append((kd, int(r["best"]), lb, float(r["stat"])))
    return out


# ---------------------------------------------------------------------------- extended battery
NEW_FEATURES = ("squote_curly_inside", "squote_straight_inside", "dash_aside", "comma_aside")
OPENERS_BEFORE_SQ = "([{\"“‘—–-"


def battery_ext_seq(pieces):
    """New state features for one sequence of decoded token strings. Character-level pass over the
    concatenation; each token gets the state of the text strictly before its first character (the
    v1 convention). Lookahead of one character may read the token's own first character."""
    text = "".join(pieces)
    starts = np.cumsum([0] + [len(p) for p in pieces[:-1]]) if pieces else np.zeros(0, int)
    n, N = len(pieces), len(text)
    out = {k: np.zeros(n, np.float32) for k in NEW_FEATURES}
    cq = sq = dash = comma = 0
    line_has_text = False
    j = 0
    for k in range(N + 1):
        while j < n and starts[j] == k:
            out["squote_curly_inside"][j] = cq > 0
            out["squote_straight_inside"][j] = sq > 0
            out["dash_aside"][j] = dash == 1
            out["comma_aside"][j] = comma == 1
            j += 1
        if k == N:
            break
        c = text[k]
        prev = text[k - 1] if k > 0 else "\n"
        nxt = text[k + 1] if k + 1 < N else " "
        if c == "\n":
            cq = sq = dash = comma = 0; line_has_text = False
            continue
        if c == "‘":                                        # left single quotation mark
            cq += 1
        elif c == "’":                                      # right single quotation mark / apostrophe
            if cq > 0 and not nxt.isalnum():
                cq -= 1
        elif c == "'":
            if (prev.isspace() or prev in OPENERS_BEFORE_SQ) and not nxt.isspace() and nxt != "'":
                sq += 1
            elif sq > 0 and not prev.isspace() and not nxt.isalnum():
                sq -= 1
        elif c == "—":                                      # em dash; a run counts once
            if prev != "—" and line_has_text:
                dash ^= 1
        elif c == "–":                                      # en dash: spaced only
            if prev.isspace() and nxt.isspace() and line_has_text:
                dash ^= 1
        elif c == "-" and prev != "-":                           # first hyphen of a run
            m = 1
            while k + m < N and text[k + m] == "-":
                m += 1
            after = text[k + m] if k + m < N else " "
            if line_has_text and (m == 2 or (m == 1 and prev.isspace() and after.isspace())):
                dash ^= 1
        elif c == ",":
            if not (prev.isdigit() and nxt.isdigit()):
                comma ^= 1
        elif c in ".!?":
            if not (prev.isdigit() and nxt.isdigit()) and (nxt.isspace() or nxt in "\"”’')]"):
                dash = comma = 0
        if not c.isspace():
            line_has_text = True
    return out


def battery_ext_all(full_tokens, tk, n_seq_rows=255):
    strs = {}
    def s_of(t):
        if t not in strs:
            strs[t] = tk.decode([int(t)])
        return strs[t]
    out = {k: np.zeros(len(full_tokens), np.float32) for k in NEW_FEATURES}
    for s0 in range(0, len(full_tokens), n_seq_rows):
        f = battery_ext_seq([s_of(int(t)) for t in full_tokens[s0:s0 + n_seq_rows]])
        for k in NEW_FEATURES:
            out[k][s0:s0 + n_seq_rows] = f[k]
    return out


def extended_battery(Dset, tk, full):
    """v1 battery (e61.feature_battery, unchanged) + the new states + the two unions."""
    feats = e61.feature_battery(Dset, tk, full, None)
    ext = battery_ext_all(full, tk)
    for k in NEW_FEATURES:
        feats[k] = ext[k][Dset["row"]]
    feats["enclosure_inside"] = np.maximum.reduce([feats["paren_inside"], feats["bracket_inside"],
                                                   feats["quote_inside"], feats["squote_curly_inside"]])
    feats["aside_inside"] = np.maximum.reduce([feats["enclosure_inside"], feats["squote_straight_inside"],
                                               feats["dash_aside"], feats["comma_aside"]])
    return feats


SELFTEST = [   # (pieces, feature, expected state before each token)
    (["He", " said", " ‘", "hello", "’", " to", " me"], "squote_curly_inside", [0, 0, 0, 1, 1, 0, 0]),
    (["‘", "I", " don", "’", "t", " know", "’", " she"], "squote_curly_inside", [0, 1, 1, 1, 1, 1, 1, 0]),
    (["‘", "I", " don", "’t", " know", "’", "."], "squote_curly_inside", [0, 1, 1, 1, 1, 1, 0]),
    (["‘", "a", "\n", "b"], "squote_curly_inside", [0, 1, 1, 0]),
    (["the", " word", " '", "foo", "'", " is"], "squote_straight_inside", [0, 0, 0, 1, 1, 0]),
    (["I", " don", "'t", " know", "'", " x"], "squote_straight_inside", [0, 0, 0, 0, 0, 0]),
    (["The", " cat", " -", " a", " tabby", " -", " sat", "."], "dash_aside", [0, 0, 0, 1, 1, 1, 0, 0]),
    (["well", "-", "known", " fact"], "dash_aside", [0, 0, 0, 0]),
    (["word", "—", "aside", "—", "word"], "dash_aside", [0, 0, 1, 1, 0]),
    (["\n", "-", " item", " one"], "dash_aside", [0, 0, 0, 0]),
    (["1990", "–", "2000"], "dash_aside", [0, 0, 0]),
    ([" yes", " –", " no"], "dash_aside", [0, 0, 1]),
    (["a", "--", "b", "--", "c"], "dash_aside", [0, 0, 1, 1, 0]),
    (["a", " -", " b", ".", " c"], "dash_aside", [0, 0, 1, 1, 0]),
    (["I", " went", ",", " reluctantly", ",", " home", ".", " Then", ",", " ok"], "comma_aside",
     [0, 0, 0, 1, 1, 0, 0, 0, 0, 1]),
    (["1", ",", "000", " people"], "comma_aside", [0, 0, 0, 0]),
    (["3", ".", "5", " x", ",", " y"], "comma_aside", [0, 0, 0, 0, 0, 1]),
]


def selftest():
    bad = 0
    for pieces, f, exp in SELFTEST:
        got = battery_ext_seq(pieces)[f].astype(int).tolist()
        ok = got == exp
        bad += not ok
        print(("ok  " if ok else "FAIL"), f, repr("".join(pieces)), got if not ok else "", flush=True)
    assert bad == 0, f"{bad} selftest case(s) failed"
    print(f"selftest OK ({len(SELFTEST)} cases)", flush=True)


# ---------------------------------------------------------------------------- D2 maps
def fit_maps(Z, tok, tinv, n_min=N_MIN, n0=N0):
    """e63.type_whiten with its maps kept (same operations, same order)."""
    n_t, order, start = e63.group_sorted(tinv)
    Z = Z.astype(np.float64)
    mu = np.stack([np.bincount(tinv, weights=Z[:, j], minlength=len(n_t)) for j in range(Z.shape[1])], 1)
    Zc = Z - (mu / n_t[:, None])[tinv]
    Sp = Zc.T @ Zc / len(Zc)
    Sp_is = e63.inv_sqrt(Sp)
    out = Zc @ Sp_is
    big = np.flatnonzero(n_t >= n_min)
    iso = np.zeros((len(big), Z.shape[1], Z.shape[1]))
    for i, t in enumerate(big):
        ii = order[start[t]:start[t] + n_t[t]]
        St = Zc[ii].T @ Zc[ii] / n_t[t]
        lam = n0 / (n_t[t] + n0)
        iso[i] = e63.inv_sqrt((1 - lam) * St + lam * Sp)
        out[ii] = Zc[ii] @ iso[i]
    out -= out.mean(0)
    w, U = np.linalg.eigh(np.cov(out[::2].T))
    o = np.argsort(w)[::-1]
    W = (out @ U[:, o] / np.sqrt(w[o])).astype(np.float32)
    utok = np.unique(tok)
    return W, dict(Sp_is=Sp_is, type_ids=utok[big], type_isqrt=iso, Uo=U[:, o], sw=np.sqrt(w[o]))


def apply_maps(M, Z, tok):
    """D2's per-type and global maps applied to another set; type and global means are the set's own."""
    utok, tinv = np.unique(tok, return_inverse=True)
    n_t, order, start = e63.group_sorted(tinv)
    Z = Z.astype(np.float64)
    mu = np.stack([np.bincount(tinv, weights=Z[:, j], minlength=len(n_t)) for j in range(Z.shape[1])], 1)
    Zc = Z - (mu / n_t[:, None])[tinv]
    out = Zc @ M["Sp_is"]
    pos = {int(t): i for i, t in enumerate(M["type_ids"])}
    for t in range(len(n_t)):
        i = pos.get(int(utok[t]))
        if i is not None:
            ii = order[start[t]:start[t] + n_t[t]]
            out[ii] = Zc[ii] @ M["type_isqrt"][i]
    out -= out.mean(0)
    return (out @ M["Uo"] / M["sw"]).astype(np.float32)


def load_maps(L):
    z = np.load(EXT / f"results/e64_maps_L{L}.npz")
    P = dict(mu=z["P_mu"], U=z["P_U"], sd=z["P_sd"])
    M = {k: z[k] for k in ("Sp_is", "type_ids", "type_isqrt", "Uo", "sw")}
    return P, M, z


def represent_other(tag, L):
    """S2 / R2: the set's own rows and first moments (e63.represent_base), then D2's maps."""
    P, M, z = load_maps(L)
    S = e63.represent_base(tag, L, N_ITER)
    Z = e61.pca_apply(P, S["V"]); del S["V"]
    return S, apply_maps(M, Z, S["tok"]), z


def heavy_sign(y):
    return 1.0 if abs(np.quantile(y, 0.99)) >= abs(np.quantile(y, 0.01)) else -1.0


def reconstruct(layers):
    rec = dict(layers={})
    fz = frozen_v2()
    for L in layers:
        t0 = time.time()
        fr = fz["layers"][str(L)]["frozen"]
        assert fr["whitening"]["N_MIN"] == N_MIN and fr["whitening"]["n0"] == N0
        D = e63.represent_base(TAG["D"], L, N_ITER)
        fnpz = np.load(EXT / f"results/occ_units_v2_L{L}.npz")
        assert np.array_equal(fnpz["row"], D["row"]), "D2 rows differ from the frozen rows"
        Z, P = e63.pca_white(D["V"]); del D["V"]
        W, M = fit_maps(Z, D["tok"], D["tinv"])
        W_e63 = e63.type_whiten(Z, D["tinv"], N_MIN, N0)[0]
        W_app = apply_maps(M, Z, D["tok"])
        consist = dict(fit_vs_e63=float(np.abs(W - W_e63).max()), apply_vs_fit=float(np.abs(W_app - W).max()))
        assert consist["fit_vs_e63"] < 1e-4 and consist["apply_vs_fit"] < 1e-4, consist
        X = np.hstack([W.astype(np.float64), np.ones((len(W), 1))])
        Zc = Z.astype(np.float64) - Z.mean(0)
        Pu = np.asarray(P["U"]); sd = np.asarray(P["sd"])
        keep = np.ones(1152, bool); keep[e61.drop_for(L)] = False
        out = dict(P_mu=P["mu"], P_U=Pu, P_sd=sd, **M)
        lr = dict(consistency=consist, kinds={})
        for kd in KINDS:
            Y = fnpz[f"Y_{kd}"].astype(np.float64)
            B, *_ = np.linalg.lstsq(X, Y, rcond=None)
            res_ = Y - X @ B
            r2 = 1 - (res_ ** 2).sum(0) / ((Y - Y.mean(0)) ** 2).sum(0)
            Ys = (Y - Y.mean(0)) / Y.std(0)
            pat = (W.astype(np.float64).T @ Ys) / len(Ys)            # unit patterns in W coordinates
            sign = np.asarray([heavy_sign(Y[:, j]) for j in range(Y.shape[1])])
            cz = (Zc.T @ Ys) / len(Ys)                               # Cov(Z, y): pattern before the per-type step
            raw = Pu @ (sd[:, None] * cz)                            # pulled back to the kept coordinates
            dirs = np.zeros((Y.shape[1], 1152)); dirs[:, keep] = (raw * sign).T
            dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
            out.update({f"readout_{kd}": B, f"r2_{kd}": r2, f"pattern_{kd}": pat, f"sign_{kd}": sign,
                        f"dir_{kd}": dirs})
            scr = screened_ids(fr, kd)
            lr["kinds"][kd] = dict(min_r2_screened=float(r2[scr].min()), min_r2_all=float(r2.min()),
                                   n_screened=len(scr))
        lr["accepted"] = bool(all(v["min_r2_screened"] >= RECON_R2 for v in lr["kinds"].values()))
        np.savez_compressed(EXT / f"results/e64_maps_L{L}.npz", **out)
        lr["maps_sha256"] = sha(EXT / f"results/e64_maps_L{L}.npz")
        lr["seconds"] = round(time.time() - t0)
        rec["layers"][str(L)] = lr
        print(f"L{L} reconstruct {lr}", flush=True)
    rec["accepted"] = bool(all(v["accepted"] for v in rec["layers"].values()))
    dump(rec, "results/e64_reconstruct.json")
    for L in layers:
        append_hash(f"results/e64_maps_L{L}.npz", "e64 reconstruct: D2 maps, patterns, directions")
    append_hash("results/e64_reconstruct.json", "e64 reconstruct verdict")
    assert rec["accepted"], "frozen units not reproduced from re-extracted D2 (R^2 < 0.999): STOP"


# ---------------------------------------------------------------------------- labelling
def associations(y, feats):
    """Naming rule (prereg section 6): any battery feature with |AUROC| >= 0.75 (binary), |rho| >= 0.5
    (graded), or a 1% tail whose rate is >= 3x the base rate and >= 0.10. Binary features with fewer
    than 50 positive (or negative) rows are reported, not used."""
    from scipy.stats import rankdata, spearmanr
    n = len(y); k = int(np.ceil(TAIL * n))
    o = np.argsort(y); lo, hi = o[:k], o[-k:]
    R = rankdata(y)
    rows, named_by = {}, []
    for f, v in feats.items():
        if f in GRADED:
            rho = float(abs(spearmanr(y, v).statistic))
            rows[f] = dict(abs_spearman=rho, named=rho >= NAME_RHO)
        else:
            pos = v > 0.5; npos = int(pos.sum())
            base = float(pos.mean())
            r_hi, r_lo = float(v[hi].mean()), float(v[lo].mean())
            usable = NAME_MIN_POS <= npos <= n - NAME_MIN_POS
            a = (R[pos].sum() - npos * (npos + 1) / 2) / (npos * (n - npos)) if usable else 0.5
            A = max(a, 1 - a)
            enr = [r >= NAME_ENRICH * base and r >= NAME_MIN_RATE for r in (r_hi, r_lo)]
            rows[f] = dict(auroc=float(A), base_rate=base, n_pos=npos, rate_top1pct_hi=r_hi, rate_top1pct_lo=r_lo,
                           usable=usable, named=bool(usable and (A >= NAME_AUROC or any(enr))),
                           named_via=[w for w, ok in (("auroc", A >= NAME_AUROC), ("enrich_hi", enr[0]),
                                                      ("enrich_lo", enr[1])) if usable and ok])
        if rows[f]["named"]:
            named_by.append(f)
    return rows, named_by


def partner_safe(tinv, rng):
    n_t = np.bincount(tinv)
    ok = n_t[tinv] >= 2
    rep = np.full(len(tinv), -1)
    if ok.any():
        sub = np.flatnonzero(ok)
        _, tsub = np.unique(tinv[sub], return_inverse=True)
        rep[sub] = sub[e63.partner(tsub, rng, True)]
    return rep


def identity_diag(y, sign, D, tk):
    n = len(y); k = int(np.ceil(TAIL * n))
    mem = np.zeros(n, bool); mem[np.argsort(-sign * y)[:k]] = True
    rep = partner_safe(D["tinv"], np.random.default_rng(0))
    m = mem & (rep >= 0)
    tail_tdi = float(mem[rep[m]].mean() / mem.mean()) if m.any() else float("nan")
    tt, tc = np.unique(D["tok"][mem], return_counts=True)
    o = np.argsort(-tc)[:8]
    return dict(tail_tdi=tail_tdi, tail_n_types=int(len(tt)), tail_max_type_share=float(tc.max() / k),
                tail_top_types=[(tk.decode([int(tt[i])]), int(tc[i])) for i in o],
                identity_suspect=bool(tail_tdi > IDENT_TDI or tc.max() / k > IDENT_SHARE)), mem


def contexts(y, D, tk, full, seed):
    n = len(y); k = int(np.ceil(TAIL * n)); o = np.argsort(y)
    rng = np.random.default_rng(seed); out = {}
    for side, idx in (("hi", o[-k:]), ("lo", o[:k])):
        cs = []
        for i in rng.choice(idx, size=N_CTX, replace=False):
            r = int(D["row"][i]); s0 = (r // 255) * 255
            cs.append(tk.decode([int(x) for x in full[max(s0, r - 12):r]]) + "[[" + tk.decode([int(full[r])]) + "]]"
                      + tk.decode([int(x) for x in full[r + 1:min(s0 + 255, r + 4)]]))
        out[side] = cs
    return out


def twins(mixA, mixB, idsA, idsB):
    """Mutual best |cos| >= TWIN_COS between screened logcosh and screened skew mixing columns."""
    if not idsA or not idsB:
        return []
    A = mixA[:, idsA].T; B = mixB[:, idsB].T
    A = A / np.linalg.norm(A, axis=1, keepdims=True); B = B / np.linalg.norm(B, axis=1, keepdims=True)
    C = np.abs(A @ B.T)
    out = []
    for i in range(len(idsA)):
        j = int(C[i].argmax())
        if int(C[:, j].argmax()) == i and C[i, j] >= TWIN_COS:
            out.append((idsA[i], idsB[j], float(C[i, j])))
    return out


def rows_only(tag):
    """Row selection of e63.represent_base without loading activations."""
    m = np.load(EXT / f"cache/occ/{tag}_meta.npz")
    tok, seq, pos = m["tokens"].astype(np.int64), m["seq"].astype(np.int64), m["pos"].astype(np.int64)
    _, inv, cnt = np.unique(tok, return_inverse=True, return_counts=True)
    idx = np.flatnonzero((cnt[inv] >= e63.MIN_TYPE_OCC) & (pos >= e63.MIN_POS))
    t = tok[idx]
    return dict(tok=t, seq=seq[idx], pos=pos[idx], row=idx, tinv=np.unique(t, return_inverse=True)[1]), tok


def label(layers):
    require_hashed("results/e64_reconstruct.json", *[f"results/e64_maps_L{L}.npz" for L in layers])
    assert json.loads((EXT / "results/e64_reconstruct.json").read_text())["accepted"]
    tk = e61.token_strings()
    fz = frozen_v2()
    D, full = rows_only(TAG["D"])
    feats = extended_battery(D, tk, full)
    out = dict(battery=sorted(feats), base_rates={k: float(v.mean()) for k, v in feats.items() if k not in GRADED},
               layers={})
    template = {}
    for L in layers:
        fr = fz["layers"][str(L)]["frozen"]
        fnpz = np.load(EXT / f"results/occ_units_v2_L{L}.npz")
        assert np.array_equal(fnpz["row"], D["row"])
        ids = {kd: screened_ids(fr, kd) for kd in KINDS}
        tw = twins(fnpz["mix_logcosh"], fnpz["mix_skew"], ids["logcosh"], ids["skew"])
        ctrl = control_units(fz, L)
        units = []
        for kd in KINDS:
            Y = fnpz[f"Y_{kd}"].astype(np.float64)
            for j in ids[kd]:
                y = Y[:, j]; s = heavy_sign(y)
                assoc, named_by = associations(y, feats)
                ident, _ = identity_diag(y, s, D, tk)
                twin = [("skew", b, c) for a, b, c in tw if kd == "logcosh" and a == j] + \
                       [("logcosh", a, c) for a, b, c in tw if kd == "skew" and b == j]
                u = dict(kind=kd, id=j, heavy_sign=s, named=bool(named_by), named_by=named_by,
                         control=[lb for k2, i2, lb, _ in ctrl if (k2, i2) == (kd, j)], twin=twin,
                         identity=ident, associations=assoc, contexts=contexts(y, D, tk, full, 5 + 1000 * L + j))
                units.append(u)
                if not u["named"]:
                    template[f"L{L}/{kd}/{j}"] = dict(reading=None, tail=None)
        bad_ctrl = [(u["kind"], u["id"]) for u in units if u["control"] and not u["named"]]
        out["layers"][str(L)] = dict(units=units, twins=tw, control_units=ctrl, control_not_named=bad_ctrl,
                                     n_screened={kd: len(ids[kd]) for kd in KINDS},
                                     n_battery_unnamed={kd: sum(1 for u in units if u["kind"] == kd and not u["named"])
                                                        for kd in KINDS})
        print(f"L{L} label: screened {out['layers'][str(L)]['n_screened']} battery-UNNAMED "
              f"{out['layers'][str(L)]['n_battery_unnamed']} twins {len(tw)} control-not-named {bad_ctrl}", flush=True)
    out["rule_ok"] = bool(all(not v["control_not_named"] for v in out["layers"].values()))
    dump(out, "results/e64_labels.json")
    append_hash("results/e64_labels.json", "e64 label: D2 battery verdicts, before any reading, S2, R2 or behaviour")
    assert out["rule_ok"], "the labelling rule does not name its own control units: STOP"
    rp = EXT / "results/e64_context_reading.json"
    assert not rp.exists(), "a context reading already exists"
    rp.write_text(json.dumps(dict(instructions=(
        "For each battery-UNNAMED unit read its contexts in e64_labels.json. Set reading to a short construction "
        "name if the contexts of one tail show one consistent named construction, else to 'none'; set tail to "
        "'hi' or 'lo' (or null for 'none'). A reading can only move a unit from UNNAMED to NAMED_BY_CONTEXT. "
        "Do not look at any S2, R2 or behavioural output before sealing."), units=template), indent=1))


def seal_reading():
    require_hashed("results/e64_labels.json")
    r = json.loads((EXT / "results/e64_context_reading.json").read_text())
    missing = [k for k, v in r["units"].items() if not v.get("reading")]
    assert not missing, f"reading missing for {missing}"
    append_hash("results/e64_context_reading.json", "e64 context reading, sealed before S2, R2 and behaviour")


def candidates(L):
    """Discovery candidates at L: screened, battery-UNNAMED, not NAMED_BY_CONTEXT."""
    lab = json.loads((EXT / "results/e64_labels.json").read_text())["layers"][str(L)]
    rd = json.loads((EXT / "results/e64_context_reading.json").read_text())["units"]
    out = []
    for u in lab["units"]:
        if u["named"]:
            continue
        r = rd[f"L{L}/{u['kind']}/{u['id']}"]["reading"]
        out.append(dict(kind=u["kind"], id=u["id"], context_named=(r.strip().lower() != "none"), reading=r))
    return out, lab


# ---------------------------------------------------------------------------- S2 and R2
def unit_table(W, Sset, tk, full, label_feats=True):
    """The frozen v2 directional-unit code (e63.ica_units, seeds 0/1/2, screens unchanged) on W."""
    feats = extended_battery(Sset, tk, full) if label_feats else None
    out = {}
    for kd in KINDS:
        Y, mix, units = e63.ica_units(W, Sset, N_ICA, kd)
        for u in units:
            u["screened"] = bool(e63.ica_pass(u, 0.8, False))
            if label_feats and u["screened"]:
                _, nb = associations(Y[:, u["id"]].astype(np.float64), feats)
                u["named"], u["named_by"] = bool(nb), nb
        out[kd] = dict(Y=Y, mix=mix, units=units)
    return out, feats


def best_match(pat, mix):
    if mix.shape[1] == 0:
        return -1, 0.0
    A = pat / np.linalg.norm(pat); B = mix / np.linalg.norm(mix, axis=0, keepdims=True)
    c = np.abs(A @ B)
    return int(c.argmax()), float(c.max())


def chance_match(mix, seed=0):
    if mix.shape[1] == 0:
        return float("nan")
    g = np.random.default_rng(seed).standard_normal((N_NULL_DIRS, mix.shape[0]))
    g /= np.linalg.norm(g, axis=1, keepdims=True)
    B = mix / np.linalg.norm(mix, axis=0, keepdims=True)
    return float(np.percentile(np.abs(g @ B).max(1), 99.9))


def shuffled(layers):
    require_hashed("results/e64_labels.json", "results/e64_context_reading.json")
    tk = e61.token_strings()
    fz = frozen_v2()
    out = dict(layers={})
    for L in layers:
        t0 = time.time(); check_swap()
        S, W, z = represent_other(TAG["S"], L)
        full = np.load(EXT / f"cache/occ/{TAG['S']}_meta.npz")["tokens"]
        tab, _ = unit_table(W, S, tk, full)
        fr = fz["layers"][str(L)]["frozen"]
        pres = {}
        for kd in KINDS:
            for j in screened_ids(fr, kd):
                v, c = best_match(z[f"pattern_{kd}"][:, j], tab[kd]["mix"])
                pres[f"{kd}/{j}"] = dict(match=v, cos=c, context_generic=bool(c >= PRESENCE_COS))
        yld = {kd: dict(n_screened=sum(u["screened"] for u in tab[kd]["units"]),
                        n_screened_battery_unnamed=sum(u["screened"] and not u["named"] for u in tab[kd]["units"]),
                        fit_failed=len(tab[kd]["units"]) == 0) for kd in KINDS}
        out["layers"][str(L)] = dict(presence=pres, s2_yield=yld, kir=e63.kir(W, S["tinv"]),
                                     chance_cos_p999={kd: chance_match(tab[kd]["mix"]) for kd in KINDS},
                                     units={kd: tab[kd]["units"] for kd in KINDS}, seconds=round(time.time() - t0))
        print(f"L{L} S2: yield {yld} KIR {out['layers'][str(L)]['kir']:.1f} "
              f"context-generic {sum(v['context_generic'] for v in pres.values())}/{len(pres)}", flush=True)
        del S, W, tab
    dump(out, "results/e64_shuffled.json")
    append_hash("results/e64_shuffled.json", "e64 S2 presence and yield")


def replicate(layers):
    require_hashed("results/e64_labels.json", "results/e64_context_reading.json")
    tk = e61.token_strings()
    fz = frozen_v2()
    out = dict(layers={})
    for L in layers:
        t0 = time.time(); check_swap()
        R, W, z = represent_other(TAG["R"], L)
        full = np.load(EXT / f"cache/occ/{TAG['R']}_meta.npz")["tokens"]
        tab, feats = unit_table(W, R, tk, full)
        cands, _ = candidates(L)

        def match(kd, j):
            v, c = best_match(z[f"pattern_{kd}"][:, j], tab[kd]["mix"])
            u = tab[kd]["units"][v] if v >= 0 else None
            return v, c, u

        ctrl = []
        for kd, j, lb, _ in control_units(fz, L):
            v, c, u = match(kd, j)
            a = e61.auroc(tab[kd]["Y"][:, v], feats[lb]) if u is not None else 0.5
            a = max(a, 1 - a)
            ctrl.append(dict(kind=kd, id=j, label=lb, r2_unit=v, cos=c, screened_r2=bool(u and u["screened"]),
                             auroc_r2=float(a), replicated=bool(c >= REPL_COS and u and u["screened"] and a >= CTRL_FLOOR)))
        cand = []
        for cd in cands:
            v, c, u = match(cd["kind"], cd["id"])
            cand.append(dict(kind=cd["kind"], id=cd["id"], r2_unit=v, cos=c, screened_r2=bool(u and u["screened"]),
                             unnamed_r2=bool(u and u["screened"] and not u["named"]),
                             named_by_r2=(u or {}).get("named_by"),
                             replicated=bool(c >= REPL_COS and u and u["screened"] and not u["named"])))
        informative = any(x["replicated"] for x in ctrl)
        out["layers"][str(L)] = dict(controls=ctrl, informative=informative, candidates=cand,
                                     kir=e63.kir(W, R["tinv"]),
                                     chance_cos_p999={kd: chance_match(tab[kd]["mix"]) for kd in KINDS},
                                     n_screened_r2={kd: sum(u["screened"] for u in tab[kd]["units"]) for kd in KINDS},
                                     seconds=round(time.time() - t0))
        print(f"L{L} R2: controls replicated {[x['replicated'] for x in ctrl]} candidates replicated "
              f"{sum(x['replicated'] for x in cand)}/{len(cand)} KIR {out['layers'][str(L)]['kir']:.1f}", flush=True)
        del R, W, tab
    dump(out, "results/e64_replicate.json")
    append_hash("results/e64_replicate.json", "e64 R2 replication")


# ---------------------------------------------------------------------------- behaviour
def signature(rows_D, mem_idx, tokD, nxt_by_type):
    """e62.signature, unchanged: the 20 next tokens most over-represented after the members."""
    r = rows_D[mem_idx]
    r = r[(r % 255) != 254]
    obs = np.bincount(tokD[r + 1], minlength=VOCAB_N).astype(float)
    exp = np.zeros(VOCAB_N)
    types, cnt = np.unique(tokD[r], return_counts=True)
    for t, c in zip(types, cnt):
        d = nxt_by_type.get(int(t))
        if d is not None:
            exp[d[0]] += c * d[1]
    score = np.log((obs + 1) / (exp + 1))
    score[obs < 5] = -np.inf
    return [int(x) for x in np.argsort(-score)[:20] if np.isfinite(score[x])]


def behaviour(layers):
    require_hashed("results/e64_labels.json", "results/e64_context_reading.json")
    import torch
    import gfdisc.model as gm
    gm.MODEL_NAME = str(sorted((EXT / "hf_home/hub/models--google--gemma-3-1b-pt/snapshots").glob("*"))[0])
    from gfdisc.env import DEVICE
    from gfdisc.model import load_model
    swap0 = check_swap()
    fz = frozen_v2()
    labs = json.loads((EXT / "results/e64_labels.json").read_text())
    metaD = np.load(EXT / f"cache/occ/{TAG['D']}_meta.npz")
    tokD = metaD["tokens"].astype(np.int64)
    nxt_by_type = {}
    r_all = np.arange(len(tokD)); r_all = r_all[(r_all % 255) != 254]
    order = np.argsort(tokD[r_all], kind="stable")
    tt = tokD[r_all][order]; nn = tokD[r_all + 1][order]
    for b in np.split(np.arange(len(tt)), np.flatnonzero(np.diff(tt)) + 1):
        u_, c_ = np.unique(nn[b], return_counts=True)
        nxt_by_type[int(tt[b[0]])] = (u_, c_ / c_.sum())
    ccount = np.bincount(np.fromfile(EXT / "results/corpus_tokens_pass1.i32", dtype=np.int32), minlength=VOCAB_N)
    model, tok = load_model(token=None)
    rv = np.fromfile(EXT / "results/reserve_tokens.i32", dtype=np.int32).reshape(-1, 256)[BEH_RESERVE[0]:BEH_RESERVE[1]]
    out = dict(swap_gb_at_start=swap0, layers={})
    for L in layers:
        t0 = time.time()
        _, _, z = load_maps(L)
        fnpz = np.load(EXT / f"results/occ_units_v2_L{L}.npz")
        rows_D = fnpz["row"].astype(np.int64)
        fr = fz["layers"][str(L)]["frozen"]
        lab = labs["layers"][str(L)]
        twin_of = {}
        for a, b, _ in lab["twins"]:
            twin_of[("logcosh", a)] = ("skew", b); twin_of[("skew", b)] = ("logcosh", a)
        rng = np.random.default_rng(6400 + L)
        srng = np.random.default_rng(6462 + L)
        sites = [(int(srng.integers(len(rv))), int(srng.integers(5, 256))) for _ in range(N_OCC)]
        prompts = [np.concatenate([[tok.bos_token_id], rv[s, max(1, p - CTX):p + 1]]) for s, p in sites]
        site_tok = np.asarray([int(rv[s, p]) for s, p in sites])

        def forward(d, eps):
            outs = []
            for lo in range(0, len(prompts), BATCH):
                blk = prompts[lo:lo + BATCH]
                Lm = max(len(x) for x in blk)
                ids = torch.full((len(blk), Lm), tok.pad_token_id, dtype=torch.long)
                att = torch.zeros_like(ids)
                for i, x in enumerate(blk):
                    ids[i, :len(x)] = torch.from_numpy(x.astype(np.int64)); att[i, :len(x)] = 1
                last = torch.tensor([len(x) - 1 for x in blk])
                hook = None
                if d is not None:
                    Dv = torch.tensor(d, dtype=torch.float32, device=DEVICE)

                    def fn(_m, _i, o, Dv=Dv, last=last):
                        t = o[0] if isinstance(o, tuple) else o
                        new = t.float()
                        ib = torch.arange(len(last), device=DEVICE); lp = last.to(DEVICE)
                        h = new[ib, lp]
                        new[ib, lp] = h + eps * h.norm(dim=-1, keepdim=True) * Dv
                        new = new.to(t.dtype)
                        return (new,) + o[1:] if isinstance(o, tuple) else new
                    hook = model.model.layers[L].register_forward_hook(fn)
                try:
                    with torch.no_grad():
                        lg = model(input_ids=ids.to(DEVICE), attention_mask=att.to(DEVICE), use_cache=False).logits
                    outs.append(lg[torch.arange(len(blk)), last.to(DEVICE)].float().cpu())
                finally:
                    if hook is not None:
                        hook.remove()
            return torch.cat(outs, 0)

        base = forward(None, 0.0)

        def delta(lg, U):
            T = torch.tensor(U["T"]); R_ = torch.tensor(U["R"])
            f = lambda x: torch.logsumexp(x[:, T], -1) - torch.logsumexp(x[:, R_], -1)
            return (f(lg) - f(base)).numpy()

        def sign_consistency(dl):
            types, cnt = np.unique(site_tok, return_counts=True)
            top = types[np.argsort(-cnt)[:20]]
            return float(np.mean(np.asarray([dl[site_tok == t].mean() for t in top]) > 0))

        def build(kd, j):
            y = fnpz[f"Y_{kd}"][:, j].astype(np.float64)
            s = z[f"sign_{kd}"][j]
            k = int(np.ceil(TAIL * len(y)))
            mem = np.argsort(-s * y)[:k]
            T = signature(rows_D, mem, tokD, nxt_by_type)
            Rset = []
            for t in T:
                c = ccount[t]
                cand = np.flatnonzero((ccount >= 0.9 * c) & (ccount <= 1.1 * c) & (ccount > 0))
                cand = cand[~np.isin(cand, T)]
                Rset.append(int(rng.choice(cand)) if len(cand) else int(rng.integers(VOCAB_N)))
            return dict(kind=kd, id=j, dir=z[f"dir_{kd}"][j], T=T, R=Rset)

        screened = [(kd, j) for kd in KINDS for j in screened_ids(fr, kd)]
        U = {key: build(*key) for key in screened}
        bank = {key: float(delta(forward(u["dir"], EPS_NULL), u).mean()) for key, u in U.items() if u["T"]}

        def evaluate(key):
            u = U[key]
            rec = dict(kind=key[0], id=key[1], signature=[tok.decode([t]) for t in u["T"]],
                       random_set=[tok.decode([t]) for t in u["R"]])
            if not u["T"]:
                rec.update(PASS=False, note="empty signature"); return rec
            for e in EPS_GRID:
                dl = delta(forward(u["dir"], e), u)
                rec[f"eps={e}"] = dict(mean_delta=float(dl.mean()), sign_consistency=sign_consistency(dl))
            rr = []
            for _ in range(N_RAND):
                g = rng.standard_normal(1152); g -= (g @ u["dir"]) * u["dir"]; g /= np.linalg.norm(g)
                rr.append(float(delta(forward(g, EPS_NULL), u).mean()))
            sib = [v for k2, v in bank.items() if k2 != key and k2 != twin_of.get(key)]
            d = rec[f"eps={EPS_NULL}"]["mean_delta"]
            rec.update(null_random=rr, null_random_max=max(rr), n_siblings=len(sib),
                       sibling_p99=float(np.percentile(sib, 99, method="higher")) if sib else None,
                       sc_ok=any(rec[f"eps={e}"]["sign_consistency"] >= SC_MIN for e in EPS_GRID),
                       false_pass_bound=1.0 / (len(sib) + 1))
            rec["PASS"] = bool(rec["sc_ok"] and d > rec["null_random_max"] and sib and d > rec["sibling_p99"])
            return rec

        akd, aid, alb = ANCHOR[L]
        lay = dict(sites=dict(reserve=list(BEH_RESERVE), n=N_OCC), sibling_bank={f"{k}/{j}": v for (k, j), v in bank.items()})
        lay["anchor"] = evaluate((akd, aid)); lay["anchor"]["label"] = alb
        print(f"[{time.time()-t0:.0f}s] L{L} anchor {akd}:{aid} PASS={lay['anchor']['PASS']}", flush=True)
        lay["secondary_anchors"] = [evaluate((kd, j)) for kd, j, _, _ in control_units(fz, L) if (kd, j) != (akd, aid)]
        cands, _ = candidates(L)
        if lay["anchor"]["PASS"]:
            lay["candidates"] = []
            for cd in cands:
                r = evaluate((cd["kind"], cd["id"])); r["context_named"] = cd["context_named"]
                lay["candidates"].append(r)
                print(f"[{time.time()-t0:.0f}s] L{L} {cd['kind']}:{cd['id']} PASS={r['PASS']}", flush=True)
            disc = [r for r in lay["candidates"] if not r["context_named"]]
            lay["n_pass"] = sum(r["PASS"] for r in disc)
            lay["expected_false_pass_bound"] = float(sum(r.get("false_pass_bound", 0.0) for r in disc))
            lay["excess"] = bool(lay["n_pass"] > lay["expected_false_pass_bound"])
        else:
            lay["candidates"] = None
            lay["verdict"] = "ANCHOR FAILED -- no behavioural verdict at this layer"
        lay["seconds"] = round(time.time() - t0)
        out["layers"][str(L)] = lay
        dump(out, "results/e64_behaviour.json")
    out["swap_gb_at_end"] = swap_used_gb()
    dump(out, "results/e64_behaviour.json")
    append_hash("results/e64_behaviour.json", "e64 behavioural tier")


# ---------------------------------------------------------------------------- verdict
def verdict(layers):
    rels = ["results/e64_labels.json", "results/e64_context_reading.json", "results/e64_shuffled.json",
            "results/e64_behaviour.json", "results/e64_replicate.json"]
    require_hashed(*rels)
    labs, shf, beh, rep = (json.loads((EXT / r).read_text()) for r in (rels[0], rels[2], rels[3], rels[4]))
    fz = frozen_v2()
    out = dict(layers={}, discoveries=[])
    for L in layers:
        cands, lab = candidates(L)
        ul = {(u["kind"], u["id"]): u for u in lab["units"]}
        b = beh["layers"][str(L)]; r = rep["layers"][str(L)]; s = shf["layers"][str(L)]
        bmap = {(x["kind"], x["id"]): x for x in (b["candidates"] or [])}
        rmap = {(x["kind"], x["id"]): x for x in r["candidates"]}
        rows = []
        for cd in cands:
            key = (cd["kind"], cd["id"])
            if cd["context_named"]:
                tier = "NAMED_BY_CONTEXT"
            else:
                t2 = bool(b["anchor"]["PASS"] and b.get("excess") and bmap.get(key, {}).get("PASS"))
                t3 = bool(r["informative"] and rmap[key]["replicated"])
                tier = "DISCOVERY" if (t2 and t3) else ("behavioural+geometric" if t2 else
                                                       ("replicated geometric" if t3 else "geometric candidate"))
            rows.append(dict(kind=key[0], id=key[1], tier=tier, reading=cd["reading"],
                             twin=ul[key]["twin"], identity=ul[key]["identity"],
                             context_generic=s["presence"][f"{key[0]}/{key[1]}"]["context_generic"],
                             s2_cos=s["presence"][f"{key[0]}/{key[1]}"]["cos"],
                             behaviour_pass=bmap.get(key, {}).get("PASS"), r2_cos=rmap[key]["cos"],
                             r2_replicated=rmap[key]["replicated"]))
        fams, seen = [], set()
        for x in rows:
            k = (x["kind"], x["id"])
            if k in seen:
                continue
            mem = [x] + [y for y in rows if x["twin"] and (y["kind"], y["id"]) == tuple(x["twin"][0][:2])]
            seen.update((m["kind"], m["id"]) for m in mem)
            fams.append(dict(members=[(m["kind"], m["id"]) for m in mem],
                             discovery=any(m["tier"] == "DISCOVERY" for m in mem)))
        out["layers"][str(L)] = dict(
            units=rows, families=fams, n_families=len(fams), n_discovery_families=sum(f["discovery"] for f in fams),
            anchor_pass=b["anchor"]["PASS"], behaviour_excess=b.get("excess"),
            expected_false_behavioural_passes=b.get("expected_false_pass_bound"), replication_informative=r["informative"],
            s2_yield=s["s2_yield"], kir=dict(D2=fz["layers"][str(L)]["frozen"]["kir"], S2=s["kir"], R2=r["kir"]))
        out["discoveries"] += [dict(layer=L, **f) for f in fams if f["discovery"]]
    dump(out, "results/e64_verdict.json")
    append_hash("results/e64_verdict.json", "e64 verdict")
    print("DISCOVERIES", json.dumps(out["discoveries"]), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True, choices=["selftest", "reconstruct", "label", "seal-reading",
                                                         "shuffled", "behaviour", "replicate", "verdict"])
    ap.add_argument("--layers", default="7,12")
    a = ap.parse_args()
    if a.phase == "selftest":
        return selftest()
    check_integrity()
    layers = [int(x) for x in a.layers.split(",")]
    assert set(layers) <= set(LAYERS), "discovery is restricted to L7 and L12"
    if a.phase == "seal-reading":
        return seal_reading()
    check_swap()
    dict(reconstruct=reconstruct, label=label, shuffled=shuffled, behaviour=behaviour, replicate=replicate,
         verdict=verdict)[a.phase](layers)


if __name__ == "__main__":
    main()
