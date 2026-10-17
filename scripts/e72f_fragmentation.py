"""e72f -- does the e66-confirmed L12 direction (logcosh #13) fragment across SAE latents?
(PREREG_fragmentation.md, sha256 asserted at every data phase; DRAFT until the user approves the freeze.)

The rig is e65/e66's, through e66's frozen chain (e66 -> e65 -> e64, each sha-asserted): dose c*sigma_v with
c in {0.5, 1, 2, 4} and every gate at c* = 2; the quote anchor (logcosh #33) as the rig gate under e65's PASS
rule (re-implemented here from e65's primitives, as e66 did); 99 RAND directions. New here: contexts from the
MAIN corpus (block F = corpus 8000-8999 after a duplicate screen; BOS-dropped 255-column rows); two class
readouts (subordinators S, prepositions P), each corrected by a frequency-matched reference set; SAE decoder
rows and reconstructions as arms; share-matched, d-oriented, cos-matched banks dosed at the arm's own alpha
as the null for one-sidedness; planted-tilt positive controls.

Phases (each once, in order)
  selftest  synthetic checks of the pure helpers (no data, no model)
  freeze    (CPU; HTTP range only with --with-262k, done first and skipped cleanly on failure) class and
            reference token ids; duplicate screen and block-F sites; RAND; SAE rows and their asserts against
            e67/e68; planted directions; banks; covariate diagnostics on D2; all hashed before any forward pass
  steer     (MPS) unsteered pass and every sigma; RAND; the anchor (stop if it fails); D, D16; each gated arm
            and its bank (L_big, x320, R_big, [L_262, R_262]); planted arms; PC_quote; descriptive arms
  verdict   (CPU) prereg sections 8-9, mechanically
"""
import argparse, hashlib, importlib.util, json, os, shutil, struct, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

EXT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXT / "src"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")

PREREG = EXT / "PREREG_fragmentation.md"
HASHES = EXT / "docs/HASHES.txt"
E66_REL = "scripts/e66_confirm_l12_family.py"
SAE_REV = "b738dc06961818c011fb2e44a316352ca0f4e873"          # the local Gemma Scope 2 snapshot revision
SAE_SNAP = f"hf_home/hub/models--google--gemma-scope-2-1b-pt/snapshots/{SAE_REV}"
MODEL_ST = "hf_home/hub/models--google--gemma-3-1b-pt/snapshots/fcf18a2a879aab110ca39f8bffbccd5d49d8eb29/model.safetensors"
FROZEN = {                                  # prereg section 2
    E66_REL: "21d47ae38b05c701c63f604ccc435ec5b60ac9f2b3c22f6ebd31b3034f2b3174",
    "PREREG_occurrence_v2_confirm.md": "59f1ae5b30d6f9c99ca5ca8473e58b5e445ae051e2782380e167e1cf851c87f4",
    "results/e64_maps_L12.npz": "ac1b55b2eee03594b2d9f512632db2e9980e2446d051560d950ade8726200237",
    "results/e65_frozen.json": "bd856844dae5db63d19518063d65eedf25ff8aaeee66565b09743f2da2fb2be9",
    "results/e66_confirm.json": "e8ca659da4789ccaa54679544b7ba64997638bdb25c504c7f5786e3e4cfd3b7b",
    "results/e66_describe.json": "7c760cceebd9f42badaefa68ee163635127e4250277c5f48fec6d6e50effcbe3",
    "results/e67_decoder.json": "a5b76d05d32c991976e244d83b1de3fc631b3b565c0c489c535f632203ad2640",
    "results/e67_acts.json": "d58c5b45138e1f2fbbf1a512cfe900181b133d1726deb6760724beac3f914dd1",
    "results/e67_verdict.json": "dcebb7473d832a40751f7967e5787001345a7099c28680384977264b9a6013a2",
    "results/e67_raw_L12_rows.npy": "14d80c9661c71fcff0a87f89358d64957cdf68d94e63a6ca10200645152fa0b9",
    "results/e68_decoder_G12.json": "fbc4a08d223fccff90d64c9f10d8f4b6432387253dd8e1e7f09fd0c20cfa369e",
    "results/e68_census.json": "b599ec0f5a4f6b84c2994f4bc9870ece100145eedabc186771ddbf273f1f39c8",
    "results/occ_units_v2_L12.npz": "b84a7b13d52bd54acdc665175c7972b4e9418659a3e6b56d47bdc3bb895b5d5b",
    "cache/occ/occD2_meta.npz": "bc910456116d5b875438c741c42379e24715576a2bea29e33b0dd9c1a92bc5a7",
    "results/corpus_tokens_pass1.i32": "8fbf98c3f29ebf9785885c1ac2df8ebeee7d0978c4663cd25517c5a925ba517e",
    "results/reserve_tokens.i32": "3da028b79acf01f7b67e70eb602ebed37a8883b074f9dac9f08f6ccc2b79b5b5",
    MODEL_ST: "ee5250f6eb1aa7cfb729dfd4dc8d9964fd772324776c6d00bf2bc674c069cb27",
}
DICTS = {                                   # local dictionaries: (subdir, sha256 of params.safetensors)
    "big": ("resid_post_all/layer_12_width_16k_l0_big", "e936d310077f7bf8b09c0f2602e2b62087b69b8c8f82ba1263cf9ed3bd4471ff"),
    "small": ("resid_post_all/layer_12_width_16k_l0_small", "6e1ae92a5a4d0956f1e88bd2e79e89e80f5998c76e5bab33a81a35a98488c96e"),
    "l13med": ("resid_post/layer_13_width_16k_l0_medium", "2b57a8bfb01d73b5cb7d26b48d6d8f1ec5c2b1d8554488e5a81796a4b710829d"),
}
E67_NAME = {"big": "resid_post_all/L12_16k_big", "small": "resid_post_all/L12_16k_small",
            "l13med": "resid_post/L13_16k_medium", "262s": "resid_post_all/L12_262k_small"}
HF_BASE = f"https://huggingface.co/google/gemma-scope-2-1b-pt/resolve/{SAE_REV}"
SUB_262S = "resid_post_all/layer_12_width_262k_l0_small"
N_262 = 262144

L = 12
PRIMARY, SECONDARY, ANCHOR = ("logcosh", 13), ("skew", 16), ("logcosh", 33)
BLOCK_F = dict(corpus=(8000, 9000), site_seed=7200, rand_seed=7210)
USED_CORPUS = ((0, 2000), (13000, 15000))      # occurrence work; the whole reserve is used too (duplicate screen)
DUP_KEY = 32                                  # e70's screen: the first 32 tokens of a document
N_TOK = 255                                   # corpus rows are BOS-dropped
BANK_SEEDS = dict(L_big=7220, x320=7221, R_big=7222, L_262=7223, R_262=7224, L_big_near=7225)
FILL_SEED, NEAR_COS = 7250, (0.3, 0.65)        # planted filler row; near-d secondary bank |cos| range
POOL_SEED_262, REF_SEED = 7230, 7240
M_BANK, BANK_MAXCOS, SHARE_TOL, FLOOR = 99, 0.3, 0.15, 0.5
REF_LADDER = (0.10, 0.25, 0.50)               # frequency-match tolerance, widened in this order if needed
POOL_BLOCKS_262, POOL_BLOCK_ROWS = 96, 64     # 262k bank pool: 96 random blocks of 64 consecutive rows (6,144 rows, ~28 MB)
FETCH_DEADLINE_S = 900                        # wall-clock cap on the whole 262k fetch
SWAP_MAX_GB, DISK_MIN_GB, GUARD_EVERY = 12.0, 8.0, 20
COS_TOL_ROW, COS_TOL_RECON = 1e-4, 1e-3
L_BIG, X320, R_BIG_ATOMS = 11562, 320, (320, 226, 6428, 270, 33)
L_262, R_262_ATOMS = 2084, (2084, 7872, 5346, 3714, 77508)
PC_QUOTE = ("small", 5117)
DESC_C_STAR = {                               # descriptive arms pushed at c* only: (dict, latent), d-oriented
    "atom226": ("big", 226), "atom6428": ("big", 6428), "atom270": ("big", 270), "atom33": ("big", 33),
    "L13_183": ("l13med", 183), "S2764": ("small", 2764)}
SIGNED_K3 = ("big", (11562, 219, 226))
GATED_BIG = ("L_big", "x320", "R_big")          # arms with their own bank (x320 is L_big's agreement check)
GATED_ROW, AGREE_ROW = dict(big="L_big", s262="L_262"), "x320"
PLANT_F, PLANT_GATE_F = (1.0, 0.5, 0.25), 0.5       # planted tilt strength (share of the off-d part); the gate uses 0.5
PLANTED = tuple(f"PI_{nm}_f{int(f * 100)}{sg}" for nm in ("md", "wu") for f in PLANT_F for sg in "+-")  # + toward S
S_WORDS = ("when", "while", "because", "if", "though", "although", "whenever", "unless", "whereas")
P_WORDS = ("in", "on", "at", "by", "with", "of", "from", "without", "into", "over", "under", "through", "about",
           "around", "between", "within", "across", "against", "toward", "towards", "per")
AMB_WORDS = ("before", "after", "since", "until", "till", "as", "once", "to", "for")
OUT = dict(frozen="results/e72f_frozen.json", dirs="results/e72f_dirs.npz", rows262="results/e72f_rows_262k_small.npz",
           steer="results/e72f_steer.json", verdict="results/e72f_verdict.json")
CLS = ("S", "P", "AMB", "TMP")
REF_CLS = ("S", "P")                          # gated classes get a frequency-matched reference; AMB and TMP are raw only


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


_E66 = None


def load_e66():
    """e66 as a frozen module; through it e65 and e64 (each sha-asserted by its loader)."""
    global _E66
    if _E66 is None:
        assert sha(EXT / E66_REL) == FROZEN[E66_REL], "e66 changed"
        spec = importlib.util.spec_from_file_location("e66", EXT / E66_REL)
        mod = importlib.util.module_from_spec(spec); sys.modules["e66"] = mod; spec.loader.exec_module(mod)
        _E66 = mod
    return _E66


def mods():
    e66 = load_e66(); e65 = e66.load_e65(); e64 = e65.load_e64()
    return e66, e65, e64


def check_integrity():
    e66, _, _ = mods()
    assert sha(PREREG) in HASHES.read_text(), "PREREG_fragmentation.md does not match its sha256 in docs/HASHES.txt"
    assert sha(__file__) in PREREG.read_text(), "e72f is not the frozen version (its sha256 is not in the prereg)"
    for rel, h in FROZEN.items():
        assert sha(EXT / rel) == h, f"{rel} changed"
    for k, (sub, h) in DICTS.items():
        assert sha(EXT / SAE_SNAP / sub / "params.safetensors") == h, f"dictionary {k} changed"
    e66.check_integrity()                     # e66's prereg and inputs, e65's, e64's chain


def guard(e64, where="", disk=True):
    used = e64.swap_used_gb()
    assert used <= SWAP_MAX_GB, f"swap {used:.1f} GB > {SWAP_MAX_GB} GB {where}: abort (partial output left unhashed)"
    if disk:
        free = shutil.disk_usage(EXT).free / 1e9
        assert free >= DISK_MIN_GB, f"free disk {free:.1f} GB < {DISK_MIN_GB} GB {where}: abort"
    return used


# ---------------------------------------------------------------------------- pure helpers (selftest covers these)
def dup_screen(Tc, Tr, block, used_blocks=USED_CORPUS, k=DUP_KEY):
    """e70's screen: drop block sequences whose first k tokens equal those of any used corpus sequence, any
    reserve sequence (BOS column removed) or an earlier sequence of the block. Returns (kept, dropped)."""
    key = lambda x: np.ascontiguousarray(x[:k]).tobytes()
    used = {key(Tc[i]) for lo, hi in used_blocks for i in range(lo, hi)} | {key(x) for x in Tr}
    keep, drop = [], []
    for i in range(*block):
        kk = key(Tc[i])
        (drop if kk in used else keep).append(i)
        used.add(kk)
    return keep, drop


def draw_sites(cands, seed, n):
    """n distinct sequences from the screened candidates, one position each, uniform in [4, 254] of the
    BOS-dropped row (= [5, 255] in e65's BOS-included indexing)."""
    rng = np.random.default_rng(seed)
    seqs = rng.choice(np.asarray(cands), n, replace=False)
    pos = rng.integers(4, N_TOK, size=n)
    return [[int(s), int(p)] for s, p in zip(seqs, pos)]


def build_prompts(T, sites, bos, ctx):
    """BOS, then up to ctx preceding tokens, then the site token: e66's rule on BOS-dropped rows."""
    prompts = [np.concatenate([[bos], T[s, max(0, p - ctx):p + 1]]).astype(np.int64) for s, p in sites]
    return prompts, np.asarray([int(T[s, p]) for s, p in sites])


def word_ids(encode, words):
    """Each word -> the single token id of ' word'; refuses a word that is not one token."""
    out = {}
    for w in words:
        ids = encode(" " + w)
        assert len(ids) == 1, f"' {w}' is {len(ids)} tokens"
        out[w] = int(ids[0])
    return out


def freq_refs(ccount, ids, exclude, seed, ladder=REF_LADDER):
    """e65's frequency matching, per class token: a random token whose count is within tol of the class token's
    (tol widened along the ladder only if no candidate), not excluded, not already chosen. Returns
    [(ref id, tol used)] in the order of ids."""
    rng = np.random.default_rng(seed)
    taken, out = set(exclude), []
    for t in ids:
        c = ccount[t]
        for tol in ladder:
            cand = np.flatnonzero((ccount >= (1 - tol) * c) & (ccount <= (1 + tol) * c) & (ccount > 0))
            cand = cand[~np.isin(cand, list(taken))]
            if len(cand):
                r = int(rng.choice(cand)); taken.add(r); out.append((r, tol)); break
        else:
            raise AssertionError(f"no frequency-matched reference for token {t}")
    return out


def unit_masked(R, keep):
    R = np.atleast_2d(np.asarray(R, np.float64)).copy(); R[:, ~keep] = 0
    return R / np.maximum(np.linalg.norm(R, axis=1, keepdims=True), 1e-12)


def oriented(row, d):
    return row * (1.0 if float(row @ d) >= 0 else -1.0)


def share_dropped(raw, keep):
    """Share of a raw (unmasked) row's squared norm at the dropped coordinate(s)."""
    raw = np.atleast_2d(np.asarray(raw, np.float64))
    return (raw[:, ~keep] ** 2).sum(1) / (raw ** 2).sum(1)


def nnls_recon(A, d, signed=False):
    """Reconstruction of d from the rows A (k x dim): NNLS (or least squares if signed), unit norm."""
    from scipy.optimize import nnls
    B = A.T
    coef = np.linalg.lstsq(B, d, rcond=None)[0] if signed else nnls(B, d)[0]
    v = B @ coef
    v = v / np.linalg.norm(v)
    return v, coef, float(v @ d)


def offd(v, d):
    u = np.asarray(v, np.float64) - (v @ d) * d
    return u / np.linalg.norm(u)


def eligible(Aunit, shares, d, target_share, exclude=(), alive=None):
    """Bank pool positions: |cos with d| < BANK_MAXCOS, |share - target| <= SHARE_TOL, alive, not excluded."""
    m = (np.abs(Aunit @ d) < BANK_MAXCOS) & (np.abs(shares - target_share) <= SHARE_TOL)
    if alive is not None:
        m &= alive
    m[list(exclude)] = False
    return np.flatnonzero(m)


def cos_bank(d, kappa, rows, n, seed):
    """n directions b = kappa*d + sqrt(1-kappa^2)*offd(r), with r = d-oriented pool rows drawn without
    replacement; each b has cos kappa with d exactly. Returns (B, chosen row positions)."""
    pick = np.random.default_rng(seed).choice(len(rows), n, replace=False)
    U = np.vstack([offd(oriented(r, d), d) for r in rows[pick]])
    return kappa * d[None, :] + np.sqrt(1 - kappa ** 2) * U, pick


def lse(LP, cols):
    x = LP[:, cols]; m = x.max(1, keepdims=True)
    return m[:, 0] + np.log(np.exp(x - m).sum(1))


def class_deltas(LP, LP0, ccols, rcols):
    """Per class c with a reference (S, P): the corrected shift mean_sites{[lse(c) - lse(R_c)] steered - unsteered}
    (primary), the raw shift of log P_c and the reference's own shift. Classes without a reference (AMB, TMP;
    descriptive): the raw shift. Also the share of sites with a positive (primary) shift."""
    out = {}
    for c, cols in ccols.items():
        raw = lse(LP, cols) - lse(LP0, cols)
        ref = lse(LP, rcols[c]) - lse(LP0, rcols[c]) if c in rcols else np.zeros_like(raw)
        out[c] = float((raw - ref).mean()); out[c + "_raw"] = float(raw.mean()); out[c + "_ref"] = float(ref.mean())
        out[c + "_pos"] = float(((raw - ref) > 0).mean())
    return out


def tau(x, theta_D):
    """Perpendicular displacement (nats) of the class profile (S, P) from D's profile direction; > 0 toward P."""
    t = np.radians(theta_D)
    return float(-x["S"] * np.sin(t) + x["P"] * np.cos(t))


def angle(dS, dP):
    return float(np.degrees(np.arctan2(dP, dS)))


def classify(arm, bank, rand_max, theta_D, floor=FLOOR):
    """Prereg section 8.3, first match wins. arm / bank items carry corrected 'S', 'P' at c*."""
    t = tau(arm, theta_D); bt = np.asarray([tau(b, theta_D) for b in bank])
    lo, hi = float(bt.min()), float(bt.max())
    medS, medP = float(np.median([b["S"] for b in bank])), float(np.median([b["P"] for b in bank]))
    pS, pP = arm["S"] > rand_max["S"], arm["P"] > rand_max["P"]
    rec = dict(tau=t, band=[lo, hi], band_median=float(np.median(bt)), bank_median=dict(S=medS, P=medP), potent_S=bool(pS),
               potent_P=bool(pP), floor=dict(S=floor * medS, P=floor * medP), precondition_failed=False, why=[])
    if medS <= rand_max["S"] or medP <= rand_max["P"]:
        rec["cls"] = "INCONCLUSIVE"; rec["precondition_failed"] = True
        rec["why"].append("the bank's median is not potent in both classes")
        return rec
    if t < lo and pS:
        rec["cls"] = "S-ONE-SIDED" if arm["P"] < floor * medP else "S-TILTED"
    elif t > hi and pP:
        rec["cls"] = "P-ONE-SIDED" if arm["S"] < floor * medS else "P-TILTED"
    elif lo <= t <= hi and pS and pP and arm["S"] >= floor * medS and arm["P"] >= floor * medP:
        rec["cls"] = "TWO-SIDED"
    else:
        rec["cls"] = "INCONCLUSIVE"
        side = "below" if t < lo else ("above" if t > hi else "inside")
        rec["why"].append(f"tau {side} the band")
        for c, p in (("S", pS), ("P", pP)):
            if not p:
                rec["why"].append(f"{c} not potent")
            elif side == "inside" and arm[c] < floor * (medS if c == "S" else medP):
                rec["why"].append(f"{c} below the floor")
    return rec


def loo_rates(bank, rand_max, theta_D):
    """Leave-one-out: each bank member classified against the other members (real-data calibration)."""
    cls = [classify(b, bank[:i] + bank[i + 1:], rand_max, theta_D)["cls"] for i, b in enumerate(bank)]
    n = len(cls)
    return dict(one_sided=sum(c.endswith("ONE-SIDED") for c in cls) / n, tilted=sum(c.endswith("TILTED") for c in cls) / n,
                two_sided=sum(c == "TWO-SIDED" for c in cls) / n, inconclusive=sum(c == "INCONCLUSIVE" for c in cls) / n)


READOUTS = ("corrected", "raw")


def view(x, readout):
    """The (S, P) pair of a class_deltas record under one readout."""
    return dict(S=x["S"], P=x["P"]) if readout == "corrected" else dict(S=x["S_raw"], P=x["P_raw"])


def classify_both(arm, bank, rand_max, theta_D, floor=FLOOR):
    """Section 8.3 under BOTH readouts (rand_max, theta_D: dicts by readout), combined conservatively: the same class
    under both stands; ONE-SIDED under one and TILTED under the other on the same side gives TILTED; anything else is
    INCONCLUSIVE. ONE-SIDED and TWO-SIDED thus need both readouts, and TILTED needs both on the same side, so every
    bound still holds."""
    per = {r: classify(view(arm, r), [view(b, r) for b in bank], rand_max[r], theta_D[r], floor) for r in READOUTS}
    c, c2 = per["corrected"]["cls"], per["raw"]["cls"]
    rec = dict(per=per, why=[], precondition_failed=any(v["precondition_failed"] for v in per.values()))
    if c == c2:
        rec["cls"] = c
    elif side(c) is not None and side(c) == side(c2):
        rec["cls"] = f"{side(c)}-TILTED"; rec["why"].append(f"same side, readouts differ in strength: corrected {c}, raw {c2}")
    else:
        rec["cls"] = "INCONCLUSIVE"; rec["why"].append(f"readouts disagree: corrected {c}, raw {c2}")
    return rec


def side(cls):
    return cls.split("-")[0] if cls.endswith(("ONE-SIDED", "TILTED")) else None


def agrees(cls_L, x):
    """x320's agreement check (section 8.3): never on the opposite side of L_big's call; for a one-sided or tilted
    L_big, tau(x320) on the same side of its bank's median tau under both readouts; for a TWO-SIDED L_big, x320 not
    one-sided or tilted. An INCONCLUSIVE x320 (e.g. its precondition) does not by itself disagree."""
    sL, sx = side(cls_L), side(x["cls"])
    if cls_L == "TWO-SIDED":
        return sx is None
    if sL is None:
        return True
    if sx is not None and sx != sL:
        return False
    sgn = -1 if sL == "S" else 1
    return all(sgn * (x["per"][r]["tau"] - x["per"][r]["band_median"]) > 0 for r in READOUTS)


def dict_verdict(cls_L, cls_R, pc_ok, agree=True, r_precondition=False):
    """Prereg section 9, per dictionary. cls_L: the gated row's class (VOID if its band is void)."""
    if cls_L == "VOID" or not agree:
        return "INCONCLUSIVE"
    if cls_L == "TWO-SIDED":
        return "NOT FRAGMENTED" if pc_ok else "INCONCLUSIVE"
    if cls_L.endswith("ONE-SIDED"):
        if cls_R == "TWO-SIDED":
            return "FRAGMENTED"
        return "INCONCLUSIVE" if r_precondition else "SPLIT, NOT REASSEMBLED"
    if cls_L.endswith("TILTED"):
        return "TILTED"
    return "INCONCLUSIVE"


def headline(anchor_pass, t1, v_big):
    if not anchor_pass:
        return "NO VERDICT"
    if not t1:
        return "UNINFORMATIVE"
    return v_big


def t1_of(dD, rand_max):
    """T1 under both readouts: D moves S and P beyond the RAND max (rand_max: dict by readout)."""
    return bool(all(view(dD, r)["S"] > rand_max[r]["S"] and view(dD, r)["P"] > rand_max[r]["P"] for r in READOUTS))


def planted_pass(t_plus, t_minus, band):
    """A planted pair passes if its S-ward member lands below the band and its P-ward member above it."""
    return bool(t_plus < band[0] and t_minus > band[1])


def lean(arm, rand_max, theta_D):
    """Descriptive side of a non-gated arm relative to D's profile (prereg section 9, 'reported beside')."""
    t = tau(arm, theta_D)
    if t < 0 and arm["S"] > rand_max["S"]:
        return "S-leaning"
    if t > 0 and arm["P"] > rand_max["P"]:
        return "P-leaning"
    return "neither"


def selftest():
    ok = []
    rng = np.random.default_rng(0)
    # duplicate screen, sites and prompts on a fake 255-column corpus
    Tc = rng.integers(10, 1000, size=(9100, N_TOK)); Tr = rng.integers(10, 1000, size=(50, N_TOK))
    Tc[8005, :32] = Tc[100, :32]; Tc[8006, :32] = Tr[3, :32]; Tc[8007, :32] = Tc[8001, :32]
    keep, drop = dup_screen(Tc, Tr, (8000, 9000), used_blocks=((0, 2000),))
    ok.append(("dup screen", drop == [8005, 8006, 8007] and len(keep) == 997))
    s = draw_sites(keep, 7212, 200); sa = np.asarray(s)
    ok.append(("sites", len(set(sa[:, 0])) == 200 and set(sa[:, 0]) <= set(keep) and sa[:, 1].min() >= 4 and sa[:, 1].max() <= 254
               and s == draw_sites(keep, 7212, 200)))
    P, st = build_prompts(Tc, s, 2, 64)
    Tb = np.concatenate([np.full((len(Tc), 1), 2), Tc], 1)        # the reserve's BOS-included layout
    e66like = [np.concatenate([[2], Tb[q, max(1, (p + 1) - 64):(p + 1) + 1]]) for q, p in s]
    ok.append(("prompts = e66 rule", all(np.array_equal(a, b) for a, b in zip(P, e66like))
               and np.array_equal(st, [Tc[q, p] for q, p in s])))
    ok.append(("word ids", word_ids(lambda x: [len(x)], ("a", "bb")) == {"a": 2, "bb": 3}))
    try:
        word_ids(lambda x: [1, 2], ("a",)); ok.append(("word ids refuse", False))
    except AssertionError:
        ok.append(("word ids refuse", True))
    cc = np.array([0, 100, 105, 95, 4800, 1000, 900, 1100, 5000, 120])
    refs = freq_refs(cc, [1, 5, 8], exclude={1, 5, 8, 2}, seed=1)
    ladder = freq_refs(np.array([0, 100, 130, 0]), [1], exclude={1}, seed=0)
    ok.append(("freq refs", refs[0] == (3, 0.10) and refs[1][0] in (6, 7) and refs[1][1] == 0.10 and refs[2] == (4, 0.10)
               and ladder == [(2, 0.50)]))
    try:
        freq_refs(np.array([0, 10, 0]), [1], exclude={1}, seed=0); ok.append(("freq refs refuse", False))
    except AssertionError:
        ok.append(("freq refs refuse", True))
    keepm = np.ones(64, bool); keepm[7] = False
    d = unit_masked(rng.standard_normal(64), keepm)[0]
    A = unit_masked(rng.standard_normal((5, 64)), keepm)
    raw = rng.standard_normal(64); raw[7] = 3.0
    ok.append(("mask / share", np.all(A[:, 7] == 0) and np.allclose(np.linalg.norm(A, axis=1), 1)
               and np.isclose(share_dropped(raw, keepm)[0], 9.0 / (raw @ raw))))
    ok.append(("orient", oriented(-d, d) @ d > 0 and oriented(d, d) @ d > 0))
    v, c, cs = nnls_recon(np.vstack([d * 0.6 + A[0] * 0.8, A[1]]), d)
    ok.append(("nnls", np.isclose(np.linalg.norm(v), 1) and np.all(c >= 0) and np.isclose(cs, v @ d)))
    pool = unit_masked(rng.standard_normal((300, 64)), keepm)
    shares = rng.uniform(0, 1, 300)
    el = eligible(pool, shares, d, 0.5, exclude=[0, 1])
    ok.append(("eligible", np.all(np.abs(pool[el] @ d) < BANK_MAXCOS) and np.all(np.abs(shares[el] - 0.5) <= SHARE_TOL)
               and 0 not in el and 1 not in el))
    B, pick = cos_bank(d, 0.735, pool[el], 30, 3)
    U = (B - 0.735 * d) / np.sqrt(1 - 0.735 ** 2)
    ok.append(("bank", np.allclose(B @ d, 0.735) and np.allclose(np.linalg.norm(B, axis=1), 1) and np.allclose(U @ d, 0)
               and np.all(B[:, 7] == 0) and len(set(pick)) == 30 and np.array_equal(B, cos_bank(d, 0.735, pool[el], 30, 3)[0])
               and np.all([(oriented(r, d) - (oriented(r, d) @ d) * d) @ u > 0 for r, u in zip(pool[el][pick], U)])))
    LP0 = np.log(rng.dirichlet(np.ones(20), 7)); LP = np.log(rng.dirichlet(np.ones(20), 7))
    cd = class_deltas(LP, LP0, {"S": [1, 2], "P": [3]}, {"S": [4, 5], "P": [6]})
    f = lambda X, cols: np.log(np.exp(X[:, cols]).sum(1))
    direct = np.mean((f(LP, [1, 2]) - f(LP, [4, 5])) - (f(LP0, [1, 2]) - f(LP0, [4, 5])))
    ok.append(("class delta", np.isclose(cd["S"], direct) and np.isclose(cd["P_raw"], (LP[:, 3] - LP0[:, 3]).mean())
               and np.isclose(cd["S"], cd["S_raw"] - cd["S_ref"])))
    ok.append(("tau", np.isclose(tau(dict(S=1, P=1), 45), 0) and tau(dict(S=0, P=1), 45) > 0 and tau(dict(S=1, P=0), 45) < 0
               and np.isclose(tau(dict(S=0, P=2), 0), 2)))
    # classification: D profile at 40 degrees; bank spread around it
    th = 40.0
    bank = [dict(S=np.cos(np.radians(a)) * 2, P=np.sin(np.radians(a)) * 2) for a in np.linspace(35, 45, 99)]
    rm = dict(S=0.3, P=0.3)
    cases = [dict(S=2.0, P=0.2), dict(S=1.6, P=1.0), dict(S=0.2, P=2.0), dict(S=1.0, P=1.9), dict(S=1.5, P=1.3),
             dict(S=0.6, P=0.5), dict(S=0.2, P=0.1)]
    got = [classify(x, bank, rm, th)["cls"] for x in cases]
    ok.append(("classify", got == ["S-ONE-SIDED", "S-TILTED", "P-ONE-SIDED", "P-TILTED", "TWO-SIDED", "INCONCLUSIVE", "INCONCLUSIVE"]))
    weak = [dict(S=0.1, P=b["P"]) for b in bank]
    ok.append(("bank not potent", classify(dict(S=1.5, P=1.3), weak, rm, th)["cls"] == "INCONCLUSIVE"))
    w = classify(dict(S=0.689, P=0.579), bank, rm, th)
    ok.append(("why", w["cls"] == "INCONCLUSIVE" and "tau inside the band" in w["why"] and "P below the floor" in w["why"]
               and "S below the floor" in w["why"]))
    cl = ("S-ONE-SIDED", "P-ONE-SIDED", "S-TILTED", "P-TILTED", "TWO-SIDED", "INCONCLUSIVE", "VOID")
    tab = {(a, r, pc, ag, rp): dict_verdict(a, r, pc, ag, rp) for a in cl for r in cl[:6] for pc in (True, False)
           for ag in (True, False) for rp in (True, False)}
    ok.append(("dict verdict", all(v == "INCONCLUSIVE" for (a, _, _, ag, _), v in tab.items() if not ag or a in ("VOID", "INCONCLUSIVE"))
               and all(v == ("NOT FRAGMENTED" if pc else "INCONCLUSIVE") for (a, _, pc, ag, _), v in tab.items() if a == "TWO-SIDED" and ag)
               and tab[("S-ONE-SIDED", "TWO-SIDED", False, True, False)] == "FRAGMENTED"
               and tab[("P-ONE-SIDED", "P-TILTED", True, True, False)] == "SPLIT, NOT REASSEMBLED"
               and tab[("P-ONE-SIDED", "INCONCLUSIVE", True, True, True)] == "INCONCLUSIVE"
               and all(v == "TILTED" for (a, _, _, ag, _), v in tab.items() if a.endswith("TILTED") and ag)))
    ok.append(("headline", headline(False, True, "FRAGMENTED") == "NO VERDICT" and headline(True, False, "FRAGMENTED") == "UNINFORMATIVE"
               and headline(True, True, "TILTED") == "TILTED"))
    both = lambda S, P, Sr, Pr: dict(S=S, P=P, S_raw=Sr, P_raw=Pr)
    rmb = dict(corrected=dict(S=0.5, P=0.5), raw=dict(S=0.5, P=0.5))
    ok.append(("t1 / planted", t1_of(both(1, 1, 1, 1), rmb) and not t1_of(both(1, 1, 1, 0.4), rmb) and not t1_of(both(1, 0.4, 1, 1), rmb)
               and planted_pass(-2, 2, [-1, 1]) and not planted_pass(-0.5, 2, [-1, 1])))
    # both readouts: agreement keeps the class, disagreement gives INCONCLUSIVE
    bankb = [both(b["S"], b["P"], b["S"], b["P"]) for b in bank]
    thb = dict(corrected=th, raw=th); rmc = dict(corrected=rm, raw=rm)
    c_same = classify_both(both(2.0, 0.2, 2.0, 0.2), bankb, rmc, thb)["cls"]
    c_diff = classify_both(both(2.0, 0.2, 1.5, 1.3), bankb, rmc, thb)
    c_tilt = classify_both(both(2.0, 0.2, 1.6, 1.0), bankb, rmc, thb)["cls"]
    ok.append(("both readouts", c_same == "S-ONE-SIDED" and c_diff["cls"] == "INCONCLUSIVE" and "readouts disagree" in c_diff["why"][0]
               and c_tilt == "S-TILTED"))
    # x320 agreement
    xs = lambda cls_, t, m: dict(cls=cls_, per={r: dict(tau=t, band_median=m) for r in READOUTS})
    ok.append(("agreement", agrees("S-ONE-SIDED", xs("INCONCLUSIVE", -1.0, 0.0)) and not agrees("S-ONE-SIDED", xs("INCONCLUSIVE", 1.0, 0.0))
               and not agrees("S-ONE-SIDED", xs("P-TILTED", -1.0, 0.0)) and agrees("TWO-SIDED", xs("INCONCLUSIVE", 5.0, 0.0))
               and not agrees("TWO-SIDED", xs("S-TILTED", -5.0, 0.0)) and agrees("INCONCLUSIVE", xs("P-ONE-SIDED", 5.0, 0.0))))
    # graded plants: exact cosine with d, + toward the planted unit, tilt share f
    u = offd(rng.standard_normal(64) * keepm, d); uf = offd(rng.standard_normal(64) * keepm, d)
    gp = graded_plants(d, 0.735, {"md": u}, uf)
    ok.append(("graded plants", len(gp) == 2 * len(PLANT_F) and all(np.isclose(v @ d, 0.735) and np.isclose(np.linalg.norm(v), 1) for v in gp.values())
               and all(np.isclose(((gp[f"PI_md_f{int(f * 100)}+"] - 0.735 * d) / np.sqrt(1 - 0.735 ** 2)) @ u, np.sqrt(f)) for f in PLANT_F)))
    lr = loo_rates(bank, rm, th)
    ok.append(("loo", abs(sum(lr.values()) - 1) < 1e-12 and lr["one_sided"] + lr["tilted"] <= 3 / 99 + 1e-12))
    reps, hits = 20000, 0
    for _ in range(reps):
        x = rng.standard_normal(M_BANK + 1)
        hits += x[0] < x[1:].min()
    p = hits / reps
    ok.append(("bank bound", abs(p - 1 / (M_BANK + 1)) < 4 * np.sqrt(0.01 * 0.99 / reps)))
    for name, v in ok:
        print(f"{'ok ' if v else 'FAIL'} {name}")
    assert all(v for _, v in ok), "selftest failed"
    print(f"selftest: all {len(ok)} checks pass")


# ---------------------------------------------------------------------------- 262k rows by HTTP range (script 27's bounded reader)
CURL_MAX_TIME, CURL_CONNECT_TIMEOUT = 180, 20
_DT = {"F32": np.float32, "F64": np.float64, "F16": np.float16, "I64": np.int64, "I32": np.int32}


def curl_range(url, byte_range, tries=9):
    lo, hi = (int(x) for x in byte_range.split("-"))
    last = None
    for i in range(tries):
        out = subprocess.run(["curl", "-sL", "--fail", "--retry", "5", "--retry-all-errors", "--retry-delay", "2",
                              "--connect-timeout", str(CURL_CONNECT_TIMEOUT), "--max-time", str(CURL_MAX_TIME),
                              "-r", byte_range, url], capture_output=True)
        if out.returncode == 0 and len(out.stdout) == hi - lo + 1:
            return out.stdout
        last = RuntimeError(f"curl {out.returncode} ({len(out.stdout)} of {hi - lo + 1} bytes) for {url} [{byte_range}]")
        time.sleep(1.7 ** i)
    raise last


def read_header(url):
    n = struct.unpack("<Q", curl_range(url, "0-7"))[0]
    return json.loads(curl_range(url, f"8-{8 + n - 1}")), 8 + n


def read_tensor(url, name, header, off):
    m = header[name]; a, b = m["data_offsets"]
    if m["dtype"] == "BF16":
        raw = np.frombuffer(curl_range(url, f"{off + a}-{off + b - 1}"), dtype=np.uint16).astype(np.uint32) << 16
        return raw.view(np.float32).reshape(m["shape"])
    return np.frombuffer(curl_range(url, f"{off + a}-{off + b - 1}"), dtype=_DT[m["dtype"]]).reshape(m["shape"])


def read_rows(url, header, off, rows, n_rows=N_262, dim=1152, workers=4):
    m = header["w_dec"]
    assert m["dtype"] == "F32" and m["shape"] == [n_rows, dim], m
    base, rb = off + m["data_offsets"][0], dim * 4
    rows = [int(r) for r in rows]

    def one(r):
        return np.frombuffer(curl_range(url, f"{base + r * rb}-{base + (r + 1) * rb - 1}"), dtype=np.float32)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        return np.vstack(list(ex.map(one, rows)))


def read_blocks(url, header, off, starts, size, n_rows=N_262, dim=1152, deadline=None):
    """Contiguous w_dec row blocks [start, start + size), one range request each (script 27's coalescing)."""
    m = header["w_dec"]
    assert m["dtype"] == "F32" and m["shape"] == [n_rows, dim], m
    base, rb, out = off + m["data_offsets"][0], dim * 4, []
    for st in starts:
        assert deadline is None or time.time() < deadline, "262k fetch deadline exceeded"
        out.append(np.frombuffer(curl_range(url, f"{base + st * rb}-{base + (st + size) * rb - 1}"), dtype=np.float32).reshape(size, dim))
    return np.vstack(out)


def fetch_262k(base=HF_BASE, deadline_s=FETCH_DEADLINE_S):
    """All 262k-small inputs, fetched and checked in memory; raises on any failure (the caller skips 262k).
    Pool = POOL_BLOCKS_262 random non-overlapping blocks of POOL_BLOCK_ROWS consecutive rows (seed POOL_SEED_262);
    alive rows in them that are not atoms form the bank pool. Whole fetch bounded by deadline_s."""
    deadline = time.time() + deadline_s
    url, urlx = f"{base}/{SUB_262S}/params.safetensors", f"{base}/{SUB_262S}/examples.safetensors"
    h, off = read_header(url); hx, offx = read_header(urlx)
    ff = np.asarray(read_tensor(urlx, "feature_frequencies", hx, offx), np.float64).ravel()
    assert ff.shape == (N_262,), ff.shape
    nb = N_262 // POOL_BLOCK_ROWS
    starts = np.sort(np.random.default_rng(POOL_SEED_262).choice(nb, POOL_BLOCKS_262, replace=False)) * POOL_BLOCK_ROWS
    W_atoms = np.vstack([read_blocks(url, h, off, [a], 1, deadline=deadline) for a in R_262_ATOMS])
    W_pool = read_blocks(url, h, off, starts, POOL_BLOCK_ROWS, deadline=deadline)
    pool_rows = (starts[:, None] + np.arange(POOL_BLOCK_ROWS)[None, :]).ravel()
    rows = np.concatenate([np.asarray(R_262_ATOMS), pool_rows])
    is_pool = np.concatenate([np.zeros(len(R_262_ATOMS), bool), (ff[pool_rows] >= 1e-6) & ~np.isin(pool_rows, R_262_ATOMS)])
    return dict(rows=rows, w_dec=np.vstack([W_atoms, W_pool]), ff=ff[rows], is_pool=is_pool, n_alive=int((ff >= 1e-6).sum()),
                header_w_dec=json.dumps(h["w_dec"]), block_starts=starts.tolist())


# ---------------------------------------------------------------------------- freeze (CPU)
def load_params(key):
    from safetensors.numpy import load_file
    return load_file(str(EXT / SAE_SNAP / DICTS[key][0] / "params.safetensors"))


def d2_rows(chunk=2048):
    X = np.load(EXT / "results/e67_raw_L12_rows.npy", mmap_mode="r")
    for lo in range(0, len(X), chunk):
        yield lo, np.asarray(X[lo:lo + chunk], np.float32)


def d2_encode_stats(P, latents):
    """JumpReLU encode (e67's rule, pre = x W_enc + b_enc) of D2's kept L12 rows: per-latent active counts, and the
    frequency, mean and p90 activation when active for the listed latents."""
    W, b, th = P["w_enc"].astype(np.float32), P["b_enc"].astype(np.float32), P["threshold"].astype(np.float32)
    cnt = np.zeros(W.shape[1], np.int64); acts = {i: [] for i in latents}; n = 0
    for _, x in d2_rows():
        pre = x @ W + b; a = pre * (pre > th)
        cnt += (a > 0).sum(0); n += len(x)
        for i in latents:
            v = a[:, i]; acts[i].append(v[v > 0])
    st = {}
    for i in latents:
        v = np.concatenate(acts[i])
        st[str(i)] = dict(freq=float(len(v) / n), mean_active=float(v.mean()) if len(v) else None,
                          p90_active=float(np.quantile(v, 0.9)) if len(v) else None)
    return cnt, st


def covariates(d, U, m0, keep):
    """On all D2 kept rows: corr(h.d, h.u) and SD(h.u)/SD(h.d) for each unit off-d direction u (rows of U), and
    u.m0 (the shared-component projection)."""
    k = len(U); s1 = 0.0; s2 = 0.0; t1 = np.zeros(k); t2 = np.zeros(k); x12 = np.zeros(k); n = 0
    for _, x in d2_rows():
        x = x.astype(np.float64); x[:, ~keep] = 0
        hd = x @ d; hu = x @ U.T
        s1 += hd.sum(); s2 += (hd ** 2).sum(); t1 += hu.sum(0); t2 += (hu ** 2).sum(0); x12 += (hd[:, None] * hu).sum(0); n += len(x)
    vd = s2 / n - (s1 / n) ** 2; vu = t2 / n - (t1 / n) ** 2
    corr = (x12 / n - (s1 / n) * (t1 / n)) / np.sqrt(vd * vu)
    return dict(corr=corr, sdr=np.sqrt(vu / vd), proj=U @ m0)


def shared_component(Aunit, shares, d):
    m = (shares >= 0.5) & (np.abs(Aunit @ d) < BANK_MAXCOS)
    return offd(np.mean([offd(oriented(r, d), d) for r in Aunit[m]], 0), d), int(m.sum())


def e67_row_cos(dec, dn, unit, latent):
    for t in dec["dicts"][dn][unit]["top"]:
        if t["latent"] == latent:
            return t["cos"]
    raise KeyError((dn, unit, latent))


def planted_dirs(d, keep, classes):
    """Unit off-d planted tilt directions: u_md from the D2 next-token S-minus-P residual mean difference (all D2
    kept rows); u_wu from mean_S - mean_P of (1 + final-norm weight) * embedding row (tied unembedding)."""
    import torch
    from safetensors import safe_open
    rows = np.load(EXT / "results/occ_units_v2_L12.npz")["row"].astype(np.int64)
    toks = np.load(EXT / "cache/occ/occD2_meta.npz")["tokens"].astype(np.int64)
    okn = (rows % 255) != 254
    nxt = np.full(len(rows), -1); nxt[okn] = toks[rows[okn] + 1]
    mS, mP = np.isin(nxt, classes["S"]), np.isin(nxt, classes["P"])
    sS = np.zeros(1152); sP = np.zeros(1152)
    for lo, x in d2_rows():
        sl = slice(lo, lo + len(x))
        sS += x[mS[sl]].astype(np.float64).sum(0); sP += x[mP[sl]].astype(np.float64).sum(0)
    md = sS / mS.sum() - sP / mP.sum(); md[~keep] = 0
    with safe_open(str(EXT / MODEL_ST), "pt") as f:
        E = f.get_slice("model.embed_tokens.weight")
        g = 1.0 + f.get_tensor("model.norm.weight").float().numpy().astype(np.float64)
        row = lambda i: E[i:i + 1].to(torch.float32).numpy()[0].astype(np.float64)
        wu = np.mean([g * row(i) for i in classes["S"]], 0) - np.mean([g * row(i) for i in classes["P"]], 0)
    wu[~keep] = 0
    out, info = {}, dict(n_S=int(mS.sum()), n_P=int(mP.sum()))
    for nm, u in (("md", md), ("wu", wu)):
        info[f"cos_{nm}_d"] = float(u @ d / np.linalg.norm(u))
        out[nm] = offd(u, d)
    return out, info


def graded_plants(d, kappa, units, u_fill):
    """Pi = kappa d +- s w_f, w_f = unit(sqrt(f) u + sqrt(1 - f) u_fill_perp), u_fill_perp = u_fill made orthogonal to
    u (and d). '+' leans toward S because u points from P-contexts to S-contexts."""
    s, out = np.sqrt(1 - kappa ** 2), {}
    for nm, u in units.items():
        uf = u_fill - (u_fill @ u) * u - (u_fill @ d) * d
        uf /= np.linalg.norm(uf)
        for f in PLANT_F:
            w = np.sqrt(f) * u + np.sqrt(1 - f) * uf
            w /= np.linalg.norm(w)
            out[f"PI_{nm}_f{int(f * 100)}+"] = kappa * d + s * w
            out[f"PI_{nm}_f{int(f * 100)}-"] = kappa * d - s * w
    return out


def freeze(with_262k):
    e66, e65, e64 = mods()
    for k in ("frozen", "dirs", "rows262"):
        assert not (EXT / OUT[k]).exists(), f"{OUT[k]} exists: the freeze runs once"
    t0 = time.time()
    _, _, z = e64.load_maps(L)
    keep = np.ones(1152, bool); keep[e64.e61.drop_for(L)] = False
    d = z["dir_logcosh"][PRIMARY[1]].astype(np.float64)
    d16 = z["dir_skew"][SECONDARY[1]].astype(np.float64)
    a33 = z["dir_logcosh"][ANCHOR[1]].astype(np.float64)
    assert np.all(d[~keep] == 0) and np.isclose(np.linalg.norm(d), 1)
    dec = json.loads((EXT / "results/e67_decoder.json").read_text())
    dec68 = json.loads((EXT / "results/e68_decoder_G12.json").read_text())
    # 1. 262k small first: fetched and asserted in memory; on any failure the 262k arms are skipped (section 11)
    s262, skip262 = None, None
    if with_262k:
        try:
            f = fetch_262k()
            A2 = unit_masked(f["w_dec"], keep); sh2 = share_dropped(f["w_dec"], keep); pos = {int(r): i for i, r in enumerate(f["rows"])}
            asr = {}
            for unit, dv in (("logcosh/13", d), ("skew/16", d16)):
                got, want = float(A2[pos[L_262]] @ dv), e67_row_cos(dec, E67_NAME["262s"], unit, L_262)
                asr[f"L_262_cos_{unit}"] = [got, want]
                assert abs(got - want) < COS_TOL_ROW, asr
            v2, c2, cs2 = nnls_recon(A2[[pos[a] for a in R_262_ATOMS]], d)
            e67R2 = dec["dicts"][E67_NAME["262s"]]["recon"]["logcosh/13"]["nonneg_pos"]
            assert list(e67R2["first5"]) == list(R_262_ATOMS)
            asr["R_262_cos_vs_e67"] = [cs2, e67R2["cos_at"]["5"]]
            assert abs(cs2 - e67R2["cos_at"]["5"]) < COS_TOL_RECON, asr
            s262 = dict(f=f, A=A2, share=sh2, pos=pos, R=(v2, c2, cs2), asserts=asr)
        except Exception as ex:                                  # network or integrity: skip the secondary arms
            skip262 = f"{type(ex).__name__}: {ex}"
            print("262k skipped:", skip262, flush=True)
    # 2. tokens: classes, references, duplicate screen, sites
    tk = e65.tokenizer()
    enc = lambda s: tk.encode(s, add_special_tokens=False)
    tmp_words = tuple(sorted(w for w in e66.TEMPORAL if w not in set(S_WORDS) | set(P_WORDS) | set(AMB_WORDS)
                             and len(enc(" " + w)) == 1))
    classes = {c: word_ids(enc, ws) for c, ws in (("S", S_WORDS), ("P", P_WORDS), ("AMB", AMB_WORDS), ("TMP", tmp_words))}
    Tc = np.memmap(EXT / "results/corpus_tokens_pass1.i32", dtype=np.int32, mode="r").reshape(-1, N_TOK)
    Tr = np.fromfile(EXT / "results/reserve_tokens.i32", dtype=np.int32).reshape(-1, 256)[:, 1:]
    lo, hi = BLOCK_F["corpus"]
    ccount = np.bincount(np.concatenate([np.asarray(Tc[:lo]).ravel(), np.asarray(Tc[hi:]).ravel()]), minlength=e64.VOCAB_N)
    desc = json.loads((EXT / "results/e66_describe.json").read_text())["output_shift"]
    excl = {i for m in classes.values() for i in m.values()}
    excl |= {r["id"] for c in ("c=-2.0", "c=2.0") for k in ("promoted_dlogp", "demoted_dlogp", "promoted_dp", "demoted_dp") for r in desc[c][k]}
    excl |= {int(t) for t in (tk.bos_token_id, tk.eos_token_id, tk.pad_token_id) if t is not None}
    refs = {}
    for j, c in enumerate(REF_CLS):
        rr = freq_refs(ccount, list(classes[c].values()), excl | {r for v in refs.values() for r in v["ids"]}, REF_SEED + j)
        refs[c] = dict(ids=[r for r, _ in rr], tol=[t for _, t in rr], text=[tk.decode([r]) for r, _ in rr])
    kept, dropped = dup_screen(Tc, Tr, BLOCK_F["corpus"])
    sites = draw_sites(kept, BLOCK_F["site_seed"] + L, e65.N_SITES)
    del Tc, Tr
    V, meta, asserts = {}, {}, dict(s262.get("asserts", {})) if s262 else {}

    def add(name, v, **m):
        v = np.asarray(v, np.float64)
        assert np.isclose(np.linalg.norm(v), 1) and np.all(v[~keep] == 0), name
        V[name] = v; meta[name] = dict(cos_d=float(v @ d), cos_d16=float(v @ d16), **m)

    def sae_meta(P, i, Aunit, ref=d):
        raw = P["w_dec"][i].astype(np.float64)
        return dict(dec_cos_raw=float(Aunit[i] @ ref), enc_cos_raw=float(unit_masked(P["w_enc"][:, i], keep)[0] @ ref),
                    orientation=1.0 if Aunit[i] @ ref >= 0 else -1.0, share_1038=float(share_dropped(raw, keep)[0]),
                    masked_norm=float(np.linalg.norm(raw[keep])))

    add("D", d, kind="unit"); add("D16", d16, kind="unit"); add("A33", a33, kind="unit")
    # 3. 16k big: co-primaries, R_big, atoms, signed K3; alive mask and activation stats from D2
    Pb = load_params("big"); Ab = unit_masked(Pb["w_dec"], keep); shb = share_dropped(Pb["w_dec"], keep)
    asserts["L_big_cos_vs_e67"] = [float(Ab[L_BIG] @ d), e67_row_cos(dec, E67_NAME["big"], "logcosh/13", L_BIG)]
    assert abs(asserts["L_big_cos_vs_e67"][0] - asserts["L_big_cos_vs_e67"][1]) < COS_TOL_ROW, asserts
    add("L_big", oriented(Ab[L_BIG], d), dict="big", latent=L_BIG, **sae_meta(Pb, L_BIG, Ab))
    add("x320", oriented(Ab[X320], d), dict="big", latent=X320, **sae_meta(Pb, X320, Ab))
    add("L_big_native", Ab[L_BIG], dict="big", latent=L_BIG, native=True)
    vR, cR, csR = nnls_recon(Ab[list(R_BIG_ATOMS)], d)
    e67R = dec["dicts"][E67_NAME["big"]]["recon"]["logcosh/13"]["nonneg_pos"]
    assert list(e67R["first5"]) == list(R_BIG_ATOMS)
    asserts["R_big_cos_vs_e67"] = [csR, e67R["cos_at"]["5"]]
    assert abs(csR - e67R["cos_at"]["5"]) < COS_TOL_RECON, asserts
    rawR = Pb["w_dec"][list(R_BIG_ATOMS)].astype(np.float64).T @ cR
    add("R_big", vR, dict="big", atoms=list(R_BIG_ATOMS), coef=cR.tolist(), share_1038=float(share_dropped(rawR, keep)[0]))
    for name, (dk, li) in DESC_C_STAR.items():
        if dk == "big":
            add(name, oriented(Ab[li], d), dict=dk, latent=li, **sae_meta(Pb, li, Ab))
    vS, cS, csS = nnls_recon(Ab[list(SIGNED_K3[1])], d, signed=True)
    e67S = dec["dicts"][E67_NAME["big"]]["recon"]["logcosh/13"]["signed"]
    assert list(e67S["first5"][:3]) == list(SIGNED_K3[1])
    asserts["signedK3_cos_vs_e67"] = [csS, e67S["cos_at"]["3"]]
    assert abs(csS - e67S["cos_at"]["3"]) < COS_TOL_RECON, asserts
    add("signedK3", vS, dict="big", atoms=list(SIGNED_K3[1]), coef=cS.tolist())
    cnt_b, st_b = d2_encode_stats(Pb, [L_BIG] + list(R_BIG_ATOMS))
    alive_b = cnt_b >= 1
    print(f"[{time.time() - t0:.0f}s] 16k big: rows, R_big cos {csR:.4f}, alive {int(alive_b.sum())}", flush=True)
    # 4. 16k small: PC_quote (oriented to the anchor) and S2764; L13 16k medium: #183
    Ps = load_params("small"); As = unit_masked(Ps["w_dec"], keep)
    top68 = dec68["dicts"][E67_NAME["small"]]["units"]["logcosh/33"]["top"][0]
    assert top68["latent"] == PC_QUOTE[1] and abs(float(As[PC_QUOTE[1]] @ a33) - top68["cos"]) < COS_TOL_ROW, top68
    add("PC_quote", oriented(As[PC_QUOTE[1]], a33), dict="small", latent=PC_QUOTE[1], **sae_meta(Ps, PC_QUOTE[1], As, ref=a33))
    add("S2764", oriented(As[2764], d), dict="small", latent=2764, **sae_meta(Ps, 2764, As))
    _, st_s = d2_encode_stats(Ps, [PC_QUOTE[1], 2764])
    Pm = load_params("l13med"); Am = unit_masked(Pm["w_dec"], keep)
    asserts["L13_183_cos_vs_e67"] = [float(Am[183] @ d), e67_row_cos(dec, E67_NAME["l13med"], "logcosh/13", 183)]
    assert abs(asserts["L13_183_cos_vs_e67"][0] - asserts["L13_183_cos_vs_e67"][1]) < COS_TOL_ROW, asserts
    add("L13_183", oriented(Am[183], d), dict="l13med", latent=183, pushed_at="L12 hook (exploratory)", **sae_meta(Pm, 183, Am))
    del Ps, Pm, As, Am
    # 7. RAND and the share-matched, d-oriented banks
    RB = e65.rand_bank(z["P_U"], z["P_sd"], keep, e65.N_RAND, BLOCK_F["rand_seed"] + L)
    excl_b = set(R_BIG_ATOMS) | {L_BIG, X320} | set(SIGNED_K3[1]) | {li for dk, li in DESC_C_STAR.values() if dk == "big"}
    banks, pools = {}, {}
    for arm in GATED_BIG:
        el = eligible(Ab, shb, d, meta[arm]["share_1038"], exclude=excl_b, alive=alive_b)
        B, pick = cos_bank(d, abs(meta[arm]["cos_d"]), Ab[el], M_BANK, BANK_SEEDS[arm])
        banks[arm] = dict(B=B, rows=el[pick].tolist(), shares=shb[el[pick]].tolist(), kappa=abs(meta[arm]["cos_d"]), dict="big",
                          pool_size=int(len(el)), arm=arm)
        pools[arm] = el
    # secondary near-d bank for L_big (|cos| in NEAR_COS, share-matched, d-oriented, dosed at L_big's alpha)
    cb = np.abs(Ab @ d)
    el = np.flatnonzero((cb >= NEAR_COS[0]) & (cb <= NEAR_COS[1]) & (np.abs(shb - meta["L_big"]["share_1038"]) <= SHARE_TOL) & alive_b)
    el = el[~np.isin(el, list(excl_b))]
    B, pick = cos_bank(d, abs(meta["L_big"]["cos_d"]), Ab[el], min(M_BANK, len(el)), BANK_SEEDS["L_big_near"])
    banks["L_big_near"] = dict(B=B, rows=el[pick].tolist(), shares=shb[el[pick]].tolist(), kappa=abs(meta["L_big"]["cos_d"]),
                               dict="big", pool_size=int(len(el)), arm="L_big", secondary=True)
    # planted positive controls (section 8.5): graded tilts at L_big's cosine, pushed at L_big's alpha, judged against
    # L_big's bank; the filler is an unused row of L_big's own pool
    units, pinfo = planted_dirs(d, keep, {c: list(classes[c].values()) for c in ("S", "P")})
    rest = np.setdiff1d(pools["L_big"], banks["L_big"]["rows"])
    jf = int(np.random.default_rng(FILL_SEED).choice(rest))
    pinfo.update(fill_row=jf, cos_md_wu=float(units["md"] @ units["wu"]))
    for k, v in graded_plants(d, abs(meta["L_big"]["cos_d"]), units, offd(oriented(Ab[jf], d), d)).items():
        add(k, v, kind="planted")
    # 262k small (secondary): arms, banks and shared component built locally and merged only if ALL succeed;
    # any failure here skips the 262k arms and nothing else (section 11)
    m0 = {}
    if s262:
        try:
            A2, sh2, pos, f = s262["A"], s262["share"], s262["pos"], s262["f"]
            V2, meta2, banks2 = {}, {}, {}

            def add2(name, v, **m):
                v = np.asarray(v, np.float64)
                assert np.isclose(np.linalg.norm(v), 1) and np.all(v[~keep] == 0), name
                V2[name] = v; meta2[name] = dict(cos_d=float(v @ d), cos_d16=float(v @ d16), **m)
            rowmeta = lambda i: dict(dec_cos_raw=float(A2[i] @ d), orientation=1.0 if A2[i] @ d >= 0 else -1.0,
                                     share_1038=float(sh2[i]), masked_norm=float(np.linalg.norm(f["w_dec"][i][keep])))
            add2("L_262", oriented(A2[pos[L_262]], d), dict="262s", latent=L_262, **rowmeta(pos[L_262]))
            v2, c2, _ = s262["R"]
            raw2 = f["w_dec"][[pos[a] for a in R_262_ATOMS]].astype(np.float64).T @ c2
            add2("R_262", v2, dict="262s", atoms=list(R_262_ATOMS), coef=c2.tolist(), share_1038=float(share_dropped(raw2, keep)[0]))
            for a in R_262_ATOMS[1:]:
                add2(f"a262_{a}", oriented(A2[pos[a]], d), dict="262s", latent=int(a), **rowmeta(pos[a]))
            cand = np.flatnonzero(f["is_pool"])
            for arm in ("L_262", "R_262"):
                el = cand[(np.abs(A2[cand] @ d) < BANK_MAXCOS) & (np.abs(sh2[cand] - meta2[arm]["share_1038"]) <= SHARE_TOL)]
                assert len(el) >= M_BANK, f"{arm}: only {len(el)} share-matched 262k rows"
                B, pick = cos_bank(d, abs(meta2[arm]["cos_d"]), A2[el], M_BANK, BANK_SEEDS[arm])
                banks2[arm] = dict(B=B, rows=[int(f["rows"][i]) for i in el[pick]], shares=sh2[el[pick]].tolist(),
                                   kappa=abs(meta2[arm]["cos_d"]), dict="262s", pool_size=int(len(el)), arm=arm)
            m02, n02 = shared_component(A2[cand], sh2[cand], d)
            V.update(V2); meta.update(meta2); banks.update(banks2); m0["262s"] = m02
            asserts["shared_component_rows_262s"] = n02
        except Exception as ex:
            skip262 = f"after fetch: {type(ex).__name__}: {ex}"; s262 = None
            asserts = {k: v for k, v in asserts.items() if not k.startswith(("L_262", "R_262"))}
            asserts["262k_skipped_after_asserts"] = skip262
            print("262k skipped:", skip262, flush=True)
    # 8. covariate diagnostics on all D2 rows (section 7): each gated arm against its bank
    m0b, nb = shared_component(Ab, shb, d)
    m0["big"] = m0b
    diag = {}
    for name, b in banks.items():
        arm = b["arm"]
        U = np.vstack([offd(V[arm], d)] + [offd(x, d) for x in b["B"]])
        cv = covariates(d, U, m0[b["dict"]], keep)
        shares = np.asarray([meta[arm]["share_1038"]] + b["shares"])
        rec = {}
        for nm, arr in (("corr", cv["corr"]), ("sdr", cv["sdr"]), ("proj", cv["proj"]), ("share", shares)):
            a_, bk = float(arr[0]), arr[1:]
            rec[nm] = dict(arm=a_, bank_min=float(bk.min()), bank_p50=float(np.median(bk)), bank_max=float(bk.max()),
                           pct=float((bk < a_).mean()), inside=bool(bk.min() <= a_ <= bk.max()))
        rec["outside_any"] = not all(r["inside"] for r in rec.values() if isinstance(r, dict))
        rec["void"] = bool(rec["outside_any"] and name in GATED_ROW.values())        # the void rule binds the gated row only
        diag[name] = rec
    print(f"[{time.time() - t0:.0f}s] banks and covariates done: " + ", ".join(f"{k} void={v['void']}" for k, v in diag.items()), flush=True)
    # 9. write and hash, before any forward
    arrays = {f"dir_{k}": v for k, v in V.items()}
    arrays["RAND"] = RB
    for k, b in banks.items():
        arrays[f"bank_{k}"] = b["B"]
    if s262:
        f = s262["f"]
        np.savez_compressed(EXT / OUT["rows262"], rows=f["rows"], w_dec=f["w_dec"], feature_frequencies=f["ff"], is_pool=f["is_pool"],
                            n_alive=f["n_alive"], header_w_dec=f["header_w_dec"], block_starts=np.asarray(f["block_starts"]))
    np.savez_compressed(EXT / OUT["dirs"], **arrays)
    out = dict(layer=L, block=BLOCK_F, sites=sites, duplicate_screen=dict(kept=len(kept), dropped=dropped),
               with_262k=bool(s262), with_262k_requested=bool(with_262k), skipped_262k=skip262,
               classes={c: dict(ids=list(m.values()), words=list(m)) for c, m in classes.items()}, references=refs,
               reference_exclusions=len(excl), directions=meta, planted=pinfo,
               banks={k: {kk: vv for kk, vv in b.items() if kk != "B"} for k, b in banks.items()}, covariates=diag,
               shared_component_rows_big=nb, rand=dict(n=int(len(RB)), seed=BLOCK_F["rand_seed"] + L), asserts=asserts,
               alive_16k_big=int(alive_b.sum()), d2_activation=dict(big=st_b, small=st_s),
               rows262_sha256=sha(EXT / OUT["rows262"]) if s262 else None, dirs_npz=OUT["dirs"], dirs_sha256=sha(EXT / OUT["dirs"]),
               signatures_from="results/e65_frozen.json")
    e64.dump(out, OUT["frozen"])
    if s262:
        e64.append_hash(OUT["rows262"], "e72f freeze: 262k small decoder rows by HTTP range, before any forward")
    e64.append_hash(OUT["dirs"], "e72f freeze: directions, RAND bank and share-matched banks, before any forward")
    e64.append_hash(OUT["frozen"], "e72f freeze: block-F sites, class and reference ids, covariates, before any forward")
    print(f"[{time.time() - t0:.0f}s] freeze done", flush=True)


# ---------------------------------------------------------------------------- steer engine (model-agnostic)
def steer_engine(fwd, V, plan, ctx, dump=lambda o: None, guard_fn=lambda where: None, log=print):
    """fwd(v, alpha, cap=None) -> (LP, realised_dose): LP = last-position log-softmax at the union columns
    (n_sites x n_union); with cap=dict(), fwd also fills cap['proj'] (n_pos x n_dir) and cap['ids'].
    V: name -> unit vector; plan: see plan_of; ctx: see ctx_from (+ site_tok, Dsel_names)."""
    e66, e65, e64 = mods()
    names = ctx["Dsel_names"]
    out = dict(groups_done=[])
    cap = dict(proj=[], ids=[])
    LP0, _ = fwd(None, 0.0, cap=cap)
    proj = np.concatenate(cap["proj"]); ids = np.concatenate(cap["ids"])
    sig_tot, sig_wit, sinfo = e65.natural_sigma(proj, ids, np.ones(len(proj), bool))
    sigma = dict(zip(names, map(float, sig_tot)))
    named = lambda k: not k.startswith(("RAND/", "bank_"))
    out.update(sigma={k: v for k, v in sigma.items() if named(k)}, sigma_info=sinfo,
               sigma_within={k: float(v) for k, v in zip(names, sig_wit) if named(k)},
               sigma_rand=dict(min=float(min(sigma[f"RAND/{i}"] for i in range(plan["n_rand"]))),
                               p50=float(np.median([sigma[f"RAND/{i}"] for i in range(plan["n_rand"])])),
                               max=float(max(sigma[f"RAND/{i}"] for i in range(plan["n_rand"])))))
    ccols, rcols, wcols, site_tok = ctx["ccols"], ctx["rcols"], ctx["wcols"], ctx["site_tok"]
    sig_read = lambda LP, key: e65.readout(LP, LP0, *ctx["sig"][key])
    base_f = {k: lse(LP0, ctx["sig"][k][0]) - lse(LP0, ctx["sig"][k][1]) for k in ctx["sig"]}
    npass = [0]

    def push(v, alpha, where):
        npass[0] += 1
        if npass[0] % GUARD_EVERY == 0:
            guard_fn(where)
        return fwd(v, alpha)

    def arm_record(name, doses, sig_keys=(), alpha_of=None):
        """alpha_of: dose this arm with another direction's sigma (planted arms, R_big@sigma_D)."""
        sg = sigma[alpha_of or name]
        rec = dict(sigma=sigma[name], dosed_with_sigma_of=alpha_of or name)
        for c in doses:
            LP, rr = push(V[name], c * sg, name)
            r = dict(class_delta=class_deltas(LP, LP0, ccols, rcols), realised_dose=rr)
            for k in sig_keys:
                dl = sig_read(LP, k)
                r[f"sig_{k[0]}/{k[1]}"] = dict(mean_delta=float(dl.mean()), sign_consistency=e65.sign_consistency(dl, site_tok),
                                               z=float(dl.mean() / base_f[k].std()))
            if c == e65.C_STAR:
                r["words"] = {w: float((LP[:, j] - LP0[:, j]).mean()) for w, j in wcols.items()}
                rec["_LP_cstar"] = LP
            rec[f"c={c}"] = r
        return rec

    # RAND at c*: class shifts and the signature readouts the gates need
    guard_fn("RAND")
    rand, rdose = [], []
    for i in range(plan["n_rand"]):
        LP, rr = push(V[f"RAND/{i}"], e65.C_STAR * sigma[f"RAND/{i}"], "RAND"); rdose.append(rr)
        r = class_deltas(LP, LP0, ccols, rcols)
        r.update({f"sig_{k[0]}/{k[1]}": float(sig_read(LP, k).mean()) for k in ctx["sig_gate"]})
        rand.append(r)
    rand_max = {c: float(max(r[c] for r in rand)) for c in CLS}
    rand_sig_max = {k: float(max(r[f"sig_{k[0]}/{k[1]}"] for r in rand)) for k in ctx["sig_gate"]}
    out.update(rand=rand, rand_max=rand_max, rand_sig_max={f"{k[0]}/{k[1]}": v for k, v in rand_sig_max.items()},
               rand_realised_dose=float(np.mean(rdose)))
    out["groups_done"].append("RAND"); dump(out); log("RAND done")

    def unit_gate(name, key):
        """e66's evaluate, re-implemented from e65's primitives (it is a closure in e66)."""
        rec = arm_record(name, e65.DOSES, sig_keys=(key,))
        LP = rec.pop("_LP_cstar")
        S, dropped = e65.sig_sets(key, ctx["sigs"], ctx["twin"].get(key))
        row = {f"{v[0]}/{v[1]}": float(e65.readout(LP, LP0, ctx["cols"](s["T"]), ctx["cols"](s["R"])).mean()) for v, s in S.items()}
        d_uu = rec[f"c={e65.C_STAR}"][f"sig_{key[0]}/{key[1]}"]["mean_delta"]
        sc = {c: rec[f"c={c}"][f"sig_{key[0]}/{key[1]}"]["sign_consistency"] for c in e65.SC_DOSES}
        rrow = [r[f"sig_{key[0]}/{key[1]}"] for r in rand]
        rec.update(sig=dict(max=max(row.values()), n=len(row), dropped=[f"{v[0]}/{v[1]}" for v in dropped]),
                   rand_sig=dict(max=float(max(rrow)), p50=float(np.median(rrow))), **e65.gates(d_uu, sc, rrow, list(row.values())))
        return rec

    guard_fn("anchor")
    out["anchor"] = unit_gate("A33", ANCHOR)
    out["groups_done"].append("anchor"); dump(out); log(f"anchor PASS={out['anchor']['PASS']}")
    if not out["anchor"]["PASS"]:
        out["stopped"] = "anchor failed: nothing else steered"
        return out
    guard_fn("D")
    out["arms"] = {"D": unit_gate("D", PRIMARY), "D16": unit_gate("D16", SECONDARY)}
    out["groups_done"].append("D"); dump(out)
    out["banks"], out["bank_dose"] = {}, {}
    for name in plan["gated"]:                     # each gated arm at all doses, then its bank at the ARM's alpha
        guard_fn(name)
        rec = arm_record(name, e65.DOSES); rec.pop("_LP_cstar", None)
        out["arms"][name] = rec
        alpha = e65.C_STAR * sigma[name]
        bank, bd = [], []
        for j in range(plan["n_bank"][name]):
            LP, rr = push(V[f"bank_{name}/{j}"], alpha, name); bd.append(rr)
            bank.append(class_deltas(LP, LP0, ccols, rcols))
        out["banks"][name] = bank
        out["bank_dose"][name] = dict(alpha=alpha, realised=float(np.mean(bd)),
                                      own_sigma_ratio_p50=float(np.median([sigma[f"bank_{name}/{j}"] for j in range(plan["n_bank"][name])]) / sigma[name]))
        out["groups_done"].append(name); dump(out); log(f"{name} + bank done")
    for bname, ref in plan["extra_banks"].items():   # secondary banks, dosed at their reference arm's alpha
        guard_fn(bname)
        alpha = e65.C_STAR * sigma[ref]
        bank, bd = [], []
        for j in range(plan["n_bank"][bname]):
            LP, rr = push(V[f"bank_{bname}/{j}"], alpha, bname); bd.append(rr)
            bank.append(class_deltas(LP, LP0, ccols, rcols))
        out["banks"][bname] = bank
        out["bank_dose"][bname] = dict(alpha=alpha, realised=float(np.mean(bd)), reference_arm=ref)
        out["groups_done"].append(bname); dump(out); log(f"{bname} done")
    guard_fn("planted")
    for name in plan["planted"]:
        rec = arm_record(name, (e65.C_STAR,), alpha_of="L_big"); rec.pop("_LP_cstar", None)
        out["arms"][name] = rec
    rec = arm_record("PC_quote", e65.DOSES, sig_keys=(ANCHOR,)); rec.pop("_LP_cstar", None)
    out["arms"]["PC_quote"] = rec
    out["groups_done"].append("planted"); dump(out)
    guard_fn("descriptive")
    for name, doses, alpha_of in plan["descriptive"]:
        rec = arm_record(name, doses, alpha_of=alpha_of); rec.pop("_LP_cstar", None)
        out["arms"][name + ("@sigma_D" if alpha_of else "")] = rec
    out["groups_done"].append("descriptive"); dump(out); log("descriptive done")
    return out


# ---------------------------------------------------------------------------- steer (MPS)
def make_forward(model, tok, DEVICE, prompts, union, Dsel):
    """e66.make_forward's batching and hook, verbatim in effect (h <- h + alpha*v at the output of decoder layer
    12, last prompt position only, float32 then cast back); returns the last-position log-softmax at `union`."""
    import torch
    e66, e65, e64 = mods()
    uni = torch.tensor(union, device=DEVICE)
    Dsel_t = torch.tensor(np.asarray(Dsel), dtype=torch.float32, device=DEVICE)

    def forward(v=None, alpha=0.0, cap=None):
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
            if v is not None:
                Dv = torch.tensor(v, dtype=torch.float32, device=DEVICE)

                def fn(_m, _i, o, Dv=Dv):
                    t = o[0] if isinstance(o, tuple) else o
                    new = t.float().clone()
                    new[ib, lp] = new[ib, lp] + alpha * Dv
                    new = new.to(t.dtype)
                    real.append(((new[ib, lp].float() - t[ib, lp].float()).norm(dim=-1) / abs(alpha)).cpu())
                    return (new,) + tuple(o[1:]) if isinstance(o, tuple) else new
                hooks.append(model.model.layers[L].register_forward_hook(fn))
            if cap is not None:
                def cfn(_m, _i, o, ids=ids, att=att):
                    t = (o[0] if isinstance(o, tuple) else o).float()
                    pos = torch.arange(t.shape[1])[None, :].expand(len(ids), -1)
                    m = (att.bool() & (pos >= e65.POS_MIN)).numpy()
                    cap["proj"].append((t @ Dsel_t.T).cpu().numpy()[m])
                    cap["ids"].append(ids.numpy()[m])
                hooks.append(model.model.layers[L].register_forward_hook(cfn))
            try:
                with torch.no_grad():
                    lg = model(input_ids=ids.to(DEVICE), attention_mask=att.to(DEVICE), use_cache=False).logits
                outs.append(torch.log_softmax(lg[ib, lp].float(), -1)[:, uni].cpu())
            finally:
                for h in hooks:
                    h.remove()
        return torch.cat(outs, 0).numpy().astype(np.float64), (float(torch.cat(real).mean()) if real else None)
    return forward


def ctx_from(sigs, twin, classes, refs):
    """Union columns, class / reference / word / signature columns and the SIG inputs (pure)."""
    cls_ids = {c: classes[c]["ids"] for c in CLS}
    ref_ids = {c: refs[c]["ids"] for c in REF_CLS}
    union = sorted({t for s in sigs.values() for t in s["T"] + s["R"]} | {i for v in cls_ids.values() for i in v}
                   | {i for v in ref_ids.values() for i in v})
    col = {t: i for i, t in enumerate(union)}
    cols = lambda ts: np.asarray([col[t] for t in ts], dtype=np.int64)
    words = {w: col[i] for c in CLS for w, i in zip(classes[c]["words"], classes[c]["ids"])}
    words.update({f"ref_{c}:{i}": col[i] for c in REF_CLS for i in ref_ids[c]})   # each reference's own shift
    sig = {k: (cols(sigs[k]["T"]), cols(sigs[k]["R"])) for k in (ANCHOR, PRIMARY, SECONDARY)}
    return dict(union=union, cols=cols, ccols={c: cols(v) for c, v in cls_ids.items()}, rcols={c: cols(v) for c, v in ref_ids.items()},
                wcols=words, sig=sig, sig_gate=(ANCHOR, PRIMARY, SECONDARY), sigs=sigs, twin=twin)


def build_ctx(frz, sigs65, e65, e64):
    rl, twin = e65.roles(e64, L)
    sigs = {(k.split("/")[0], int(k.split("/")[1])): dict(T=s["T"], R=s["R"]) for k, s in sigs65.items()}
    assert set(sigs) == set(rl) and rl[ANCHOR] == "anchor" and rl[PRIMARY] == "candidate" and twin.get(PRIMARY) == SECONDARY
    return ctx_from(sigs, twin, frz["classes"], frz["references"])


def plan_of(frz, Z, e65):
    """The steer plan and every direction by name (Z: the frozen npz, or any mapping with the same keys)."""
    keys = list(Z.files) if hasattr(Z, "files") else list(Z)
    gated = list(GATED_BIG) + (["L_262", "R_262"] if frz["with_262k"] else [])
    extra = {"L_big_near": "L_big"}
    desc = [("L_big_native", e65.DOSES, None), ("R_big", (e65.C_STAR,), "D")]
    desc += [(n, (e65.C_STAR,), None) for n in ("atom226", "atom6428", "atom270", "atom33", "L13_183", "S2764", "signedK3")]
    if frz["with_262k"]:
        desc += [(n, (e65.C_STAR,), None) for n in sorted(frz["directions"]) if n.startswith("a262_")]
    V = {n[4:]: Z[n] for n in keys if n.startswith("dir_")}
    for i, r in enumerate(Z["RAND"]):
        V[f"RAND/{i}"] = r
    for g in gated + list(extra):
        for j, r in enumerate(Z[f"bank_{g}"]):
            V[f"bank_{g}/{j}"] = r
    return V, list(V), dict(gated=gated, extra_banks=extra, planted=list(PLANTED), descriptive=desc, n_rand=len(Z["RAND"]),
                            n_bank={g: len(Z[f"bank_{g}"]) for g in gated + list(extra)})


def steer():
    e66, e65, e64 = mods()
    e64.require_hashed(OUT["frozen"], OUT["dirs"])
    assert not (EXT / OUT["steer"]).exists(), "steer runs once (a partial file means a crash: report, do not re-run)"
    frz = json.loads((EXT / OUT["frozen"]).read_text())
    assert sha(EXT / OUT["dirs"]) == frz["dirs_sha256"]
    if frz["with_262k"]:
        e64.require_hashed(OUT["rows262"])
    sigs65 = json.loads((EXT / "results/e65_frozen.json").read_text())["layers"][str(L)]["signatures"]
    ctx = build_ctx(frz, sigs65, e65, e64)
    Z = np.load(EXT / OUT["dirs"])
    V, names, plan = plan_of(frz, Z, e65)
    ctx["Dsel_names"] = names
    out0 = dict(block="F", corpus=list(BLOCK_F["corpus"]), swap_gb_at_start=guard(e64, "at start"))
    t0 = time.time()
    model, tok, DEVICE = e66.load_model()
    T = np.fromfile(EXT / "results/corpus_tokens_pass1.i32", dtype=np.int32).reshape(-1, N_TOK)
    prompts, site_tok = build_prompts(T, frz["sites"], tok.bos_token_id, e65.CTX)
    del T
    ctx["site_tok"] = site_tok
    fwd = make_forward(model, tok, DEVICE, prompts, ctx["union"], np.vstack([V[n] for n in names]))
    clean = lambda o: {k: v for k, v in o.items() if not k.startswith("_")}
    dump = lambda o: e64.dump({**out0, **clean(o), "seconds": round(time.time() - t0)}, OUT["steer"])
    res = steer_engine(fwd, V, plan, ctx, dump=dump, guard_fn=lambda w: guard(e64, f"during {w}"),
                       log=lambda m: print(f"[{time.time() - t0:.0f}s] {m}", flush=True))
    res["swap_gb_at_end"] = e64.swap_used_gb()
    dump(res)
    e64.append_hash(OUT["steer"], "e72f steer (block F = corpus 8000-8999; amends nothing)")


# ---------------------------------------------------------------------------- verdict (CPU)
def verdict_of(st, frz):
    """Prereg sections 8-9, mechanically, from the steer record and the frozen covariates."""
    C = "c=2.0"
    out = dict(anchor_pass=bool(st["anchor"]["PASS"]))
    if not out["anchor_pass"]:
        out["headline"] = headline(False, False, None)
        return out
    arms = st["arms"]
    rm = {r: {c: float(max(view(x, r)[c] for x in st["rand"])) for c in ("S", "P")} for r in READOUTS}
    cd = lambda name: arms[name][C]["class_delta"]
    dD = cd("D")
    out["T1"] = t1_of(dD, rm)
    thD = {r: angle(view(dD, r)["S"], view(dD, r)["P"]) for r in READOUTS}
    out.update(theta_D=thD, rand_max=rm, D_shift={k: dD[k] for k in ("S", "P", "S_raw", "P_raw", "S_ref", "P_ref")})
    cov = frz["covariates"]
    cls = {}
    for b in st["banks"]:
        arm = "L_big" if b == "L_big_near" else b
        cls[b] = classify_both(cd(arm), st["banks"][b], rm, thD)
        if cov.get(b, {}).get("void"):
            cls[b]["cls_before_void"] = cls[b]["cls"]; cls[b]["cls"] = "VOID"; cls[b]["why"].append("arm outside its bank on a covariate")
    out["classes"] = cls
    out["loo"] = {b: {r: loo_rates([view(x, r) for x in st["banks"][b]], rm[r], thD[r]) for r in READOUTS} for b in st["banks"]}
    pl = {}
    for nm in ("md", "wu"):
        for f in PLANT_F:
            k = f"{nm}_f{int(f * 100)}"
            per = {r: dict(tau_plus=tau(view(cd(f"PI_{k}+"), r), thD[r]), tau_minus=tau(view(cd(f"PI_{k}-"), r), thD[r]),
                           band=cls["L_big"]["per"][r]["band"]) for r in READOUTS}
            pl[k] = dict(per=per, PASS=all(planted_pass(v["tau_plus"], v["tau_minus"], v["band"]) for v in per.values()))
    out["planted"] = pl
    pc_ok = any(pl[f"{nm}_f{int(PLANT_GATE_F * 100)}"]["PASS"] for nm in ("md", "wu"))
    out["PC_PASS"] = pc_ok
    x_ok = not cov.get(AGREE_ROW, {}).get("outside_any")
    agree = agrees(cls["L_big"]["cls"], cls[AGREE_ROW]) if x_ok else True
    out["x320_agreement"] = dict(assessable=x_ok, agrees=agree, x320_class=cls[AGREE_ROW]["cls"])
    per = {"16k big": dict(verdict=dict_verdict(cls["L_big"]["cls"], cls["R_big"]["cls"], pc_ok, agree, cls["R_big"]["precondition_failed"]))}
    per["262k small"] = (dict(verdict=dict_verdict(cls["L_262"]["cls"], cls["R_262"]["cls"], pc_ok, True, cls["R_262"]["precondition_failed"]))
                         if "L_262" in cls else dict(verdict="not tested", reason=frz.get("skipped_262k") or "not requested"))
    out["dictionaries"] = per
    out["headline"] = headline(True, out["T1"], per["16k big"]["verdict"])
    out["262k_agrees"] = (per["262k small"]["verdict"] == per["16k big"]["verdict"]) if "L_262" in cls else None
    near = cls["L_big_near"]
    inside_near = all(near["per"][r]["band"][0] <= near["per"][r]["tau"] <= near["per"][r]["band"][1] for r in READOUTS)
    tilt_call = out["headline"] in ("FRAGMENTED", "TILTED", "SPLIT, NOT REASSEMBLED")
    out["near_d_bank"] = dict(cls=near["cls"], tau_inside_near_band_both_readouts=bool(inside_near),
                              qualifier=("the tilt is shared by d's dictionary neighbourhood: L_big is inside the near-d band"
                                         if tilt_call and inside_near else None),
                              note=("near-d check inconclusive" if tilt_call and not inside_near and near["cls"] == "INCONCLUSIVE" else None))
    lean2 = lambda n: (lambda a, b: a if a == b else "neither")(*[lean(view(cd(n), r), rm[r], thD[r]) for r in READOUTS])
    leans = {n: lean2(n) for n in ("x320", "atom226", "atom6428", "atom270", "atom33") if n in arms}
    split = bool("S-leaning" in leans.values() and "P-leaning" in leans.values())
    out["R_big_atoms"] = dict(lean=leans, point_both_ways=split, note="descriptive, no null; both readouts must agree")
    out["wording"] = (("the reconstruction's atoms point both ways (descriptive, no null)" if split else
                       "the reconstruction's atoms do not point both ways (descriptive, no null)") if out["headline"] == "FRAGMENTED" else None)
    out["reassembly_descriptive"] = {n: {r: tau(view(cd(n), r), thD[r]) for r in READOUTS} for n in ("R_big", "L_big", "x320")}
    q = arms["PC_quote"][C][f"sig_{ANCHOR[0]}/{ANCHOR[1]}"]["mean_delta"]
    qa = st["anchor"][C][f"sig_{ANCHOR[0]}/{ANCHOR[1]}"]["mean_delta"]
    out["PC_quote_descriptive"] = dict(delta=q, anchor_delta=qa, ratio=q / qa if qa else None,
                                       rand_max=st["rand_sig_max"][f"{ANCHOR[0]}/{ANCHOR[1]}"])
    nat = cd("L_big_native")
    out["L_big_native"] = dict(S=nat["S"], P=nat["P"], prediction_both_demoted=bool(nat["S"] < 0 and nat["P"] < 0))
    out["D_e65_replication"] = {k: arms["D"].get(k) for k in ("PASS", "positive", "sc_ok", "rand_ok", "sig_ok")}
    out["covariates"] = cov
    return out


def verdict():
    e66, e65, e64 = mods()
    e64.require_hashed(OUT["frozen"], OUT["steer"])
    assert not (EXT / OUT["verdict"]).exists(), "verdict runs once"
    st = json.loads((EXT / OUT["steer"]).read_text()); frz = json.loads((EXT / OUT["frozen"]).read_text())
    out = verdict_of(st, frz)
    out["amends"] = "nothing (e64-e71 verdicts stand)"
    e64.dump(out, OUT["verdict"])
    e64.append_hash(OUT["verdict"], "e72f verdict (prereg sections 8-9; amends nothing)")
    print("HEADLINE", out["headline"], flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True, choices=["selftest", "freeze", "steer", "verdict"])
    ap.add_argument("--with-262k", action="store_true", help="freeze only: fetch the 262k-small rows by HTTP range")
    a = ap.parse_args()
    print("e72f sha256", sha(__file__), flush=True)
    if a.phase == "selftest":
        return selftest()
    check_integrity()
    e64 = mods()[2]
    guard(e64, "at phase start", disk=a.phase != "verdict")
    if a.phase == "freeze":
        return freeze(a.with_262k)
    dict(steer=steer, verdict=verdict)[a.phase]()


if __name__ == "__main__":
    main()
