"""e73 -- coverage-test admission at L12 by a knockout positive control (NOTE_coverage_admission_L12.md, hashed before any
number it governs). Burned D2 only, CPU only, no download, no fresh text.

Phases (each dictionary in its own process, so one dictionary's freed memory never stacks under the next)
  calibrate_l0_big, calibrate_l0_small  for that L12 16k dictionary: encode D2's raw rows once (sparse codes, full 1152-d rows, 1038 dropped
             after encoding), e = h - SAE(h) through e72.error_W, the knockout ladder for paren and quote, 20 + 20
             firing-rate-matched random ablations, and the C2' plant at kappa 1, 2, 4. Under every condition: T-ES, T-MG
             and T-COS of the targets, the bar bank (e72's seed-7200 G) and the S40 stability bank; rule-6 and rule-8 probes
             with their 200-shift bands on each ladder step
             -> results/e73_calib_l0_<x>.json + .npz, hashed as soon as the phase completes
  verdict    applies NOTE section 10 to whatever hashed calibration output exists -> results/e73_coverage_admission.json
             (hashed); a missing dictionary reads NO VERDICT

--test-dir DIR: synthetic smoke test (docs/e73_smoke_synthetic.py builds DIR). Reads DIR/H_D.npy, DIR/sae_<l0>.npz and
DIR/units.npz + DIR/gate.json (synthetic stand-ins in e72's formats, so the real code path runs); skips the sha and k = 0
checks; hashes into DIR/HASHES.txt and writes every output to DIR. Real D2 metadata and labels; no real activation.
"""
import argparse, gc, hashlib, importlib.util, json, os, re, resource, sys, time
from pathlib import Path

import numpy as np

EXT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXT / "src"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_HOME", str(EXT / "hf_home"))

NOTE = "NOTE_coverage_admission_L12.md"
SELF = "scripts/e73_coverage_admission.py"
SMOKE = "docs/e73_smoke_synthetic.py"
SNAP = "hf_home/hub/models--google--gemma-scope-2-1b-pt/snapshots/b738dc06961818c011fb2e44a316352ca0f4e873"
INPUT_SHA = {  # NOTE section 4.1
    "results/e67_raw_L12_rows.npy": "14d80c9661c71fcff0a87f89358d64957cdf68d94e63a6ca10200645152fa0b9",
    "results/e72_units_raw_L12.npz": "3a4da05a324400fbbb92f3a13a3fd061a2a29d5d4cdd630ccd0c35771dec68dc",
    "results/e72_gate.json": "2f761d23d520079e34097797d7d9f67567d3ed8e39048b7a6462928cd74cb3bd",
    "results/e72_coverage.json": "a6b239b36e9f2104ac7a017becca052059cd761e7ca5d6ed316bfb161f727474",
    "scripts/e72_raw_residual_pilot.py": "338c10b252b816eb40072d4b6492f81c072850edf00a1fbf1995d1cfeb27d394",
    "scripts/e64_occurrence_v2_discovery.py": "1c97e42b4fc2fbf64410e8dda39fb8a9d6ae938e3fa960e6bf36d1a009fcf1f4",
    "scripts/e69_sae_gap_pilot.py": "ef1f04dd47dc92437040885ef52bde25d260c0134b5a93fe9303f88caff69bc5",
    "scripts/e70_sae_gap_geometric.py": "3fd0fbe2950d28624a6358e184f071cff3d9f941b571a2c6f37f6bb0aab03246",
    "src/gfdisc/saes.py": "b904ec290cf7076a29e64c8dc5e52e5fcf0a081500dd133c80ed6b18de5ba75f",
    "docs/sae_gap_calibration/err_ko.py": "b61360bacfca9915a38c1011810b3bd814a5bc6187be4e041f5c9cf68708db94",
    "docs/sae_gap_calibration/design_panel_result.json": "63da457bf5647c8bd1c359c9da6559bbb2272073837fada1c838cf4adc0e041b",
    "cache/occ/occD2_meta.npz": "bc910456116d5b875438c741c42379e24715576a2bea29e33b0dd9c1a92bc5a7",
    f"{SNAP}/resid_post_all/layer_12_width_16k_l0_big/params.safetensors": "e936d310077f7bf8b09c0f2602e2b62087b69b8c8f82ba1263cf9ed3bd4471ff",
    f"{SNAP}/resid_post_all/layer_12_width_16k_l0_small/params.safetensors": "6e1ae92a5a4d0956f1e88bd2e79e89e80f5998c76e5bab33a81a35a98488c96e",
}
HASHED_BEFORE = ("results/e67_raw_L12_rows.npy", "results/e72_units_raw_L12.npz", "results/e72_gate.json", "results/e72_coverage.json",
                 "scripts/e72_raw_residual_pilot.py", "scripts/e64_occurrence_v2_discovery.py", "scripts/e69_sae_gap_pilot.py",
                 "docs/sae_gap_calibration/design_panel_result.json", "cache/occ/occD2_meta.npz")
DICTS = (("l0_big", ("resid_post_all", 12, "16k", "big")), ("l0_small", ("resid_post_all", 12, "16k", "small")))
FROZEN_K = dict(paren=[2682, 7780, 4001, 4493, 1559, 8951, 3503, 1442, 507, 1056],
                quote=[5254, 1229, 1994, 1534, 564, 1882, 7302, 10306, 398, 1550])
CTRL = ("paren", "quote")
LABEL = dict(paren="paren_inside", quote="quote_inside")
TARGETS = dict(paren=[("skew", 45)], quote=[("logcosh", 29), ("skew", 29)])
STEPS = ("s0", "s1", "s3", "s10", "sND", "s30", "s100", "s256")
TESTS = ("T-ES", "T-MG", "T-COS")
N_ROWS, DROP, N_TOK = 203_328, 1038, 255
KEEP = np.ones(1152, bool); KEEP[DROP] = False
SEED_BANK, N_BANK, N_MG_BANK = 7200, 10_000, 200
SEED_S40, N_S40 = 7400, 40
SEEDS_MATCH = dict(paren=list(range(7401, 7421)), quote=list(range(7431, 7451)))
SEED_U, PATH_BASE, SEED_BAND, SEED_NDDEC, SEED_NDACT, SEED_U_FULL = 7421, 74_220_000, 7423, 7424, 7425, 7426
KAPPA = (1.0, 2.0, 4.0)
PLANT_RATE, PLANT_RUN = 0.06, 16.0                       # e70 C2' constants
MG_THETA, MG_BATCH = (0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0), 50
N_BAND, N_NDACT_SWAP, N_NDDEC_SETS = 200, 5, 200
SEED_MSIZE = 7500            # size-matched ablations (rule 4b): 7500 + 100*control + 20*step + i, steps sND, s30, s100, s256
MSIZE_STEPS = ("sND", "s30", "s100", "s256")
ERR_SPLIT_SMALL = dict(paren=[4493, 1351, 2682, 1559, 645], quote=[1229, 2899, 5082, 1472, 10306])   # err_split.py (descriptive)
E_PSEUDO, E_MIN_SUPPORT, MATCH_WIN = 1e-4, 20, 50
LEAK_REMOVED_MAX, LEAK_TOP_REMAINING, N_MATCH = 128, 32, 20
ORD, SWAP_STOP_GB, ENC_BATCH = dict(C=0, U=1, A=2), 10.0, 4096
OUT_CAL, OUT_VERDICT = "e73_calib_{}.json", "e73_coverage_admission.json"


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, EXT / rel)
    mod = importlib.util.module_from_spec(spec); sys.modules[name] = mod; spec.loader.exec_module(mod)
    return mod


e72 = _load("e72", "scripts/e72_raw_residual_pilot.py")      # loads e64 and e69 as it did in e72
e64, e69, e61, e63 = e72.e64, e72.e69, e72.e61, e72.e63
from gfdisc import saes                                        # noqa: E402
T0 = time.time()


def rss_now_gb():
    import subprocess
    out = subprocess.run(["ps", "-o", "rss=", "-p", str(os.getpid())], capture_output=True, text=True).stdout.strip()
    return int(out) / 1024 ** 2 if out else float("nan")


def t():
    return (f"[{time.time() - T0:.0f}s rss {rss_now_gb():.2f} peak "
            f"{resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 ** 3:.2f}GB]")


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def dump(obj, path):
    Path(path).write_text(json.dumps(obj, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)))


def swap_guard(where):
    s = e64.swap_used_gb()
    assert not s > SWAP_STOP_GB, f"swap {s:.1f} GB > {SWAP_STOP_GB} GB at {where}: stopped cleanly (NOTE 4.2)"
    return s


class Cfg:
    def __init__(self, test_dir=None):
        self.test = test_dir is not None
        if self.test:
            td = Path(test_dir).resolve()
            self.H_D, self.out, self.units, self.gate = td / "H_D.npy", td, td / "units.npz", td / "gate.json"
            e64.HASHES = td / "HASHES.txt"
        else:
            self.H_D, self.out = EXT / "results/e67_raw_L12_rows.npy", EXT / "results"
            self.units, self.gate = EXT / "results/e72_units_raw_L12.npz", EXT / "results/e72_gate.json"

    def p(self, name):
        return self.out / name

    def rel(self, name):
        p = self.p(name)
        return str(p) if self.test else str(p.relative_to(EXT))

    def params(self, l0, dct):
        if self.test:
            z = np.load(self.out / f"sae_{l0}.npz")
            return {k: z[k] for k in z.files}
        return e69.sae_params(dct)


def integrity(cfg):
    if cfg.test:
        return
    for rel, h in INPUT_SHA.items():
        assert sha(EXT / rel) == h, f"{rel} changed (NOTE 4.1)"
    e64.require_hashed(NOTE, SELF, SMOKE, *HASHED_BEFORE)
    hashes = e64.HASHES.read_text()
    assert sha(EXT / NOTE) in hashes and sha(EXT / SELF) in hashes
    e64.check_integrity()


# ---------------------------------------------------------------------------- shared inputs (dictionary-independent)
def seq_swap(seq, rng, n_draw):
    """Index arrays for the label null (NOTE 4.6): each D2 sequence takes the kept-row label path of another sequence
    (drawn uniformly, with replacement, among the other sequences), circularly offset by a uniform start and wrapped to its
    own kept-row count. Run lengths and base rates are kept; alignment with this sequence's features is not."""
    assert np.all(np.diff(seq) >= 0), "rows are not grouped by sequence in order"
    bounds = np.flatnonzero(np.r_[True, np.diff(seq) != 0, True])
    starts, lens = bounds[:-1], np.diff(bounds)
    ns = len(starts)
    out = np.empty((n_draw, len(seq)), np.int64)
    for d in range(n_draw):
        src = rng.integers(0, ns - 1, ns); src += src >= np.arange(ns)          # never itself
        off = (rng.random(ns) * lens[src]).astype(np.int64)
        for i in range(ns):
            a, n_i, s_, n_s = starts[i], lens[i], starts[src[i]], lens[src[i]]
            out[d, a:a + n_i] = s_ + (np.arange(n_i) + off[i]) % n_s
    return out


def markov_plant(seq_ids, rng_seed):
    """e70.markov_plant, copied (rate 0.06, mean run 16), per-sequence seed rng_seed + sequence id."""
    p10 = 1.0 / PLANT_RUN; p01 = PLANT_RATE * p10 / (1 - PLANT_RATE)
    out = np.zeros((len(seq_ids), N_TOK), np.float32)
    for i, sid in enumerate(seq_ids):
        g = np.random.default_rng(rng_seed + int(sid)); s = g.random() < PLANT_RATE
        u = g.random(N_TOK)
        for j in range(N_TOK):
            out[i, j] = s
            s = (u[j] >= p10) if s else (u[j] < p01)
    return out.reshape(-1)


def unit_cols(A):
    return A / np.linalg.norm(A, axis=0, keepdims=True)


class Ctx:
    """Everything that does not depend on the dictionary."""

    def __init__(self, cfg):
        self.cfg = cfg
        D, full = e72.rows("occD2", N_ROWS)
        self.D, self.full, self.n = D, full, len(D["row"])
        self.even = D["seq"] % 2 == 0
        Dset = dict(seq=D["seq"], tinv=D["tinv"], tok=D["tok"], row=D["row"])
        bat = e61.feature_battery(Dset, e61.token_strings(), full, None)
        self.y = {c: (np.asarray(bat[LABEL[c]]) > 0.5).astype(np.float64) for c in CTRL}
        del bat
        self.H16 = np.load(cfg.H_D)
        assert self.H16.shape == (self.n, 1152) and self.H16.dtype == np.float16
        z = np.load(cfg.units)
        assert np.array_equal(z["row"], D["row"]), "rows differ from e72's"
        self.P = dict(mu=z["P_mu"], U=z["P_U"], sd=z["P_sd"])
        self.M = {k: z[k] for k in ("Sp_is", "type_ids", "type_isqrt", "Uo", "sw")}
        self.Wh, Zh = z["W"].astype(np.float64), z["Z"].astype(np.float64)
        n = self.n
        Zc, Wc = Zh - Zh.mean(0), self.Wh - self.Wh.mean(0)
        self.CzW = Zc.T @ Wc / n; del Zc, Wc, Zh
        self.rawmap = np.asarray(self.P["U"]) @ (np.asarray(self.P["sd"])[:, None] * self.CzW)     # 1151 x 64 (e72 cos_null)
        # targets
        self.tnames, Bt = [], []
        gate = json.loads(Path(cfg.gate).read_text())
        assert sorted((u[0], u[1], u[2]) for u in gate["control_units"]) == sorted(
            (kd, j, LABEL[c]) for c in CTRL for kd, j in TARGETS[c]), gate["control_units"]
        for c in CTRL:
            for kd, j in TARGETS[c]:
                b = z[f"readout_{kd}"][:64, j].astype(np.float64)
                r = self.rawmap @ b; d = z[f"dir_{kd}"][j][KEEP]
                cs = abs(r @ d) / np.linalg.norm(r) / np.linalg.norm(d)
                assert cs >= 0.9999, (kd, j, cs)
                Bt.append(b); self.tnames.append(f"{kd}/{j}")
        self.vh_ss = float(z["vh_ss"])
        del z
        self.Bt = np.stack(Bt, 1)
        self.ctrl_targets = {c: [f"{kd}/{j}" for kd, j in TARGETS[c]] for c in CTRL}
        # banks
        G = np.random.default_rng(SEED_BANK).standard_normal((64, N_BANK)); self.G = G / np.linalg.norm(G, axis=0, keepdims=True)
        S = np.random.default_rng(SEED_S40).standard_normal((64, N_S40)); self.S40 = S / np.linalg.norm(S, axis=0, keepdims=True)
        self.RG, self.RS40 = unit_cols(self.rawmap @ self.G).astype(np.float32), unit_cols(self.rawmap @ self.S40)
        # probe-band label null (NOTE 4.6) and the label matrices [real | 200 sequence-swapped]
        idx = seq_swap(D["seq"], np.random.default_rng(SEED_BAND), N_BAND)
        self.Yband = {c: np.concatenate([self.y[c][:, None], self.y[c][idx].T], 1).astype(np.uint8) for c in CTRL}
        idx5 = seq_swap(D["seq"], np.random.default_rng(SEED_NDACT), N_NDACT_SWAP)
        self.Yact = {c: self.y[c][idx5].T.copy() for c in CTRL}
        del idx, idx5
        # plants: one Markov path; u in the W span (gating) and full rank (descriptive)
        mD = np.load(EXT / "cache/occ/occD2_meta.npz")
        seqs_u = mD["seq"][::N_TOK]
        self.plant_y = markov_plant(seqs_u, PATH_BASE)[D["row"]].astype(np.float64)
        g = np.random.default_rng(SEED_U).standard_normal(64)
        u_in = np.asarray(self.P["U"]) @ g
        uf = np.random.default_rng(SEED_U_FULL).standard_normal(1151)
        self.plant_u = dict(wspan=u_in / np.linalg.norm(u_in), full_rank=uf / np.linalg.norm(uf))
        self.plant_sigma = {k: self.sigma_along(u) for k, u in self.plant_u.items()}
        print(t(), f"context: n {n}, labels paren {self.y['paren'].mean():.4f} quote {self.y['quote'].mean():.4f}, plant rate "
              f"{self.plant_y.mean():.4f}, sigma {self.plant_sigma}", flush=True)

    @staticmethod
    def ls_filter(W, y, sel):
        X = W[sel] - W[sel].mean(0)
        b, *_ = np.linalg.lstsq(X, y[sel] - y[sel].mean(), rcond=None)
        return b / np.linalg.norm(b)

    def sigma_along(self, u):
        hu = np.empty(self.n, np.float32)
        for a in range(0, self.n, 16384):
            hu[a:a + 16384] = np.asarray(self.H16[a:a + 16384], np.float32)[:, KEEP] @ u.astype(np.float32)
        v = e72.demean3(hu[:, None].copy(), self.D["groups"])
        return float(v.std())

    def L(self, X64):
        """Fast path of e72's linear map for an n x 64 array that is already X @ P_U / P_sd."""
        V = np.ascontiguousarray(X64, dtype=np.float32)
        e72.demean3(V, self.D["groups"])
        V = V.astype(np.float64); V -= V.mean(0)
        return e64.apply_maps(self.M, V, self.D["tok"]).astype(np.float64)

    def L_full(self, X1151):
        V = np.ascontiguousarray(X1151, dtype=np.float32)
        e72.demean3(V, self.D["groups"])
        Zx = e72.project(V, self.P, own_mean=True); del V
        return e64.apply_maps(self.M, Zx, self.D["tok"]).astype(np.float64)


class PlantedH:
    """h + kappa*sigma*y*u on the fly (fp32 rows); what e72.error_W / e69.error_rows slice."""

    def __init__(self, H16, add_vec, y):
        self.H16, self.v, self.y = H16, add_vec.astype(np.float32), y.astype(np.float32)
        self.shape = H16.shape

    def __len__(self):
        return len(self.H16)

    def __getitem__(self, sl):
        x = np.asarray(self.H16[sl], np.float32)
        return x + self.y[sl][:, None] * self.v[None, :]


# ---------------------------------------------------------------------------- per dictionary
def encode(ctx, prm):
    """One pass: sparse codes (CSC), JumpReLU via saes.JumpReLUSAE, and even-row pre-activation moments per label."""
    import torch
    import scipy.sparse as sp
    sae = saes.JumpReLUSAE(*[torch.from_numpy(np.asarray(prm[k], np.float32)) for k in ("w_enc", "b_enc", "threshold", "w_dec", "b_dec")], {})
    We, be, th = sae.w_enc, sae.b_enc, sae.threshold
    n, d = ctx.n, We.shape[1]
    ev = torch.from_numpy(ctx.even)
    Y = torch.from_numpy(np.stack([ctx.y[c] for c in CTRL], 1))
    s1, s2, sy = torch.zeros(d, dtype=torch.float64), torch.zeros(d, dtype=torch.float64), torch.zeros((d, 2), dtype=torch.float64)
    blocks = []
    with torch.no_grad():
        for a in range(0, n, ENC_BATCH):
            x = torch.from_numpy(np.asarray(ctx.H16[a:a + ENC_BATCH], np.float32))
            pre = x @ We + be; z = pre * (pre > th)
            if a == 0:
                assert torch.equal(z[:256], sae.encode(x[:256])), "encode differs from saes.JumpReLUSAE.encode"
            zc = z.to_sparse_csr()                            # per-batch CSR block, no COO copy
            blocks.append(sp.csr_matrix((zc.values().numpy(), zc.col_indices().numpy().astype(np.int32),
                                         zc.crow_indices().numpy().astype(np.int64)), shape=(len(x), d)))
            del zc
            m = ev[a:a + ENC_BATCH]; pe = pre[m].double()
            s1 += pe.sum(0); s2 += (pe * pe).sum(0); sy += pe.T @ Y[a:a + ENC_BATCH][m]
    Zs = sp.vstack(blocks, format="csr"); del blocks; gc.collect()
    ne = int(ctx.even.sum()); var = s2 / ne - (s1 / ne) ** 2
    ye = Y[ev].double(); ystd = ye.std(0, unbiased=False)
    r_pre = (((sy / ne) - (s1 / ne)[:, None] * ye.mean(0)[None, :]) / torch.sqrt(torch.clamp(var, min=1e-12))[:, None] / ystd).numpy()
    r_pre[~np.isfinite(r_pre)] = 0
    return sae, Zs, {c: r_pre[:, i] for i, c in enumerate(CTRL)}


def corr_within(Zr, seq_r, Y):
    """Pearson r of every code column with each column of Y after removing each sequence's mean from both (NOTE 4.3);
    second moments only, the sparse codes are never densified."""
    import scipy.sparse as sp
    g = np.unique(seq_r, return_inverse=True)[1]; G = int(g.max()) + 1; n = len(g)
    A = sp.csr_matrix((np.ones(n), (g, np.arange(n))), shape=(G, n))
    cnt = np.asarray(A.sum(1)).ravel()
    Sz = A @ Zr
    Sz = Sz.toarray() if sp.issparse(Sz) else np.asarray(Sz)                                 # G x d
    Sy = A @ Y                                                                                # G x k
    vz = np.asarray(Zr.multiply(Zr).sum(0)).ravel() - (Sz ** 2 / cnt[:, None]).sum(0)
    vy = (Y ** 2).sum(0) - (Sy ** 2 / cnt[:, None]).sum(0)
    czy = np.asarray(Zr.T @ Y) - Sz.T @ (Sy / cnt[:, None])
    r = czy / np.sqrt(np.maximum(vz, 0)[:, None] * vy[None, :] + 1e-300)
    r[vz <= 0] = 0
    return r


def ladders(ctx, l0, prm, Zs, r_pre):
    """NOTE 4.3: K10, rule E, near-duplicates, the nested steps, and the matched random ablations."""
    d = Zs.shape[1]
    Ze, Zo = Zs[np.flatnonzero(ctx.even)], Zs[np.flatnonzero(~ctx.even)]
    Zbe, Zbo = (Ze > 0).astype(np.float64), (Zo > 0).astype(np.float64)
    info, K10, E, Eo = {}, {}, {}, {}
    for c in CTRL:
        ye, yo = ctx.y[c][ctx.even], ctx.y[c][~ctx.even]
        c1 = np.asarray(Zbe.T @ ye).ravel(); c0 = np.asarray(Zbe.T @ (1 - ye)).ravel()
        f1, f0 = c1 / ye.sum(), c0 / (1 - ye).sum()
        e = np.log((f1 + E_PSEUDO) / (f0 + E_PSEUDO)); e[c1 < E_MIN_SUPPORT] = -np.inf
        E[c] = np.lexsort((-f1, -e))                                  # rank by E desc, ties by f1 desc
        o1 = np.asarray(Zbo.T @ yo).ravel() / yo.sum(); o0 = np.asarray(Zbo.T @ (1 - yo)).ravel() / (1 - yo).sum()
        Eo[c] = np.log((o1 + E_PSEUDO) / (o0 + E_PSEUDO))
        if l0 == "l0_big":
            K10[c] = list(ctx.frozen_k[c])
        else:
            K10[c] = [int(j) for j in E[c][:10]]
        info[c] = dict(E_top10=[int(j) for j in E[c][:10]], E_top10_overlap_with_K10=len(set(E[c][:10].tolist()) & set(K10[c])))
        if l0 == "l0_small":
            info[c]["E_top10_overlap_with_err_split_top5"] = len(set(E[c][:10].tolist()) & set(ERR_SPLIT_SMALL[c]))
    # near-duplicates: decoder null (200 random 10-row sets outside both K10) and activation null (5 within-sequence shifts)
    Wd = np.asarray(prm["w_dec"], np.float32)[:, KEEP]
    Dn = Wd / np.linalg.norm(Wd, axis=1, keepdims=True)
    pool = np.setdiff1d(np.arange(d), np.r_[K10["paren"], K10["quote"]])
    rng = np.random.default_rng(SEED_NDDEC)
    sets = np.stack([rng.choice(pool, 10, replace=False) for _ in range(N_NDDEC_SETS)])
    null = []
    for a in range(0, N_NDDEC_SETS, 20):
        blk = sets[a:a + 20]
        cs = np.abs(Dn @ Dn[blk.ravel()].T).reshape(d, len(blk), 10).max(2)            # d x sets
        for i, s_ in enumerate(blk):
            m = np.ones(d, bool); m[s_] = False; null.append(cs[m, i])
    tau_dec = float(np.quantile(np.concatenate(null), 0.999)); del null
    nd, steps = {}, {}
    for c in CTRL:
        dec = np.abs(Dn @ Dn[K10[c]].T).max(1)
        se = ctx.D["seq"][ctx.even]
        r_act = np.abs(corr_within(Ze, se, ctx.y[c][ctx.even][:, None]))[:, 0]
        r_null = np.abs(corr_within(Ze, se, ctx.Yact[c][ctx.even]))
        out = np.setdiff1d(np.arange(d), K10[c])
        tau_act = float(np.quantile(r_null[out].ravel(), 0.999))
        nd_dec = set(out[dec[out] > tau_dec].tolist()); nd_act = set(out[r_act[out] > tau_act].tolist())
        nd[c] = sorted(nd_dec | nd_act)
        sND = list(K10[c]) + nd[c]
        st = dict(s0=[], s1=K10[c][:1], s3=K10[c][:3], s10=list(K10[c]), sND=sND)
        for m in (30, 100, 256):
            extra = [int(j) for j in E[c][:m] if j not in set(sND)]
            st[f"s{m}"] = sND + extra
        steps[c] = st
        added = {s: sorted(set(st[s]) - set(st[prev])) for prev, s in zip(STEPS[:-1], STEPS[1:])}
        info[c].update(K10=K10[c], tau_dec=tau_dec, tau_act=tau_act, n_nd_decoder=len(nd_dec), n_nd_activation=len(nd_act),
                       n_nd=len(nd[c]), nd=nd[c], sizes={s: len(st[s]) for s in STEPS},
                       added_odd_E_positive={s: (float(np.mean(Eo[c][a] > 0)) if a else None) for s, a in added.items()})
    # matched random ablations (rule 4)
    fire = np.bincount(Zs.indices, minlength=d)
    rank_of = np.empty(d, np.int64); order = np.argsort(fire, kind="stable"); rank_of[order] = np.arange(d)
    excl = set(steps["paren"]["s256"]) | set(steps["quote"]["s256"])
    widened = []

    def draw(Kset, seed, c):
        g = np.random.default_rng(seed); got, gs = [], set()
        for k in Kset:
            win = MATCH_WIN
            while True:                                       # widen only if the +-50 window is exhausted (recorded)
                lo, hi = max(0, rank_of[k] - win), min(d, rank_of[k] + win + 1)
                cand = [int(j) for j in order[lo:hi] if int(j) not in excl and int(j) not in gs]
                if cand:
                    break
                win *= 2; widened.append((c, seed, int(k), win))
            j = int(g.choice(cand)); got.append(j); gs.add(j)
        return got

    matched = {c: [draw(K10[c], seed, c) for seed in SEEDS_MATCH[c]] for c in CTRL}
    msize = {c: {} for c in CTRL}                       # rule 4b: one set of 20 per distinct ladder size beyond s10
    for ci, c in enumerate(CTRL):
        prev = len(steps[c]["s10"])
        for si, s_ in enumerate(MSIZE_STEPS):
            if len(steps[c][s_]) == prev:
                msize[c][s_] = None                           # same set as the previous step: reuse its draws
                continue
            prev = len(steps[c][s_])
            msize[c][s_] = [draw(steps[c][s_], SEED_MSIZE + 100 * ci + 20 * si + i, c) for i in range(N_MATCH)]
    info["matched_window_widened"] = widened
    info["firing_rate_K10"] = {c: [float(fire[k] / ctx.n) for k in K10[c]] for c in CTRL}
    info["matched_firing_rate_ratio_median"] = {c: float(np.median([fire[j] / max(fire[k], 1) for s_ in matched[c] for j, k in zip(s_, K10[c])]))
                                                for c in CTRL}
    return steps, matched, msize, info, Dn


def knockout_identity(ctx, sae, K):
    """NOTE section 3 item 1: the error of X with K's thresholds at +inf equals e + sum_k z_k W_dec[k] (first 4096 rows)."""
    import torch
    with torch.no_grad():
        x = torch.from_numpy(np.asarray(ctx.H16[:4096], np.float32))
        pre = x @ sae.w_enc + sae.b_enc; z = pre * (pre > sae.threshold)
        th2 = sae.threshold.clone(); th2[torch.as_tensor(K)] = float("inf"); z2 = pre * (pre > th2)
        e = x - z @ sae.w_dec - sae.b_dec; e2 = x - z2 @ sae.w_dec - sae.b_dec
        ko = e + z[:, K] @ sae.w_dec[K]
        dmax = float((e2 - ko)[:, torch.from_numpy(KEEP)].abs().max())
    assert dmax <= 1e-3 * float(e[:, torch.from_numpy(KEEP)].abs().max()), f"knockout identity fails ({dmax})"
    return dmax


def FROZEN_K_OF(ctx, l0, Zs, r_pre):
    """The K used for the identity check: l0_big's frozen paren 10, else the 10 latents with the largest even |r(pre, paren)|."""
    return list(ctx.frozen_k["paren"]) if l0 == "l0_big" else [int(j) for j in np.argsort(-np.abs(r_pre["paren"]))[:10]]


def mg_scores(Wh_c, We_c, even, B):
    """T-MG (NOTE 4.5) for the columns of B, in batches of MG_BATCH; returns values and the chosen (sign, theta)."""
    eps = We_c - We_c[even].mean(0)
    Ee, Eo = eps[even], eps[~even]; tot_o = float((Eo ** 2).sum())
    out, ch = np.empty(B.shape[1]), []
    for a in range(0, B.shape[1], MG_BATCH):
        Bb = B[:, a:a + MG_BATCH]
        S = Wh_c @ Bb; mu, sd = S[even].mean(0), S[even].std(0); S = (S - mu) / sd
        Se, So = S[even], S[~even]; del S
        best_g = np.full(Bb.shape[1], -np.inf); best_v = np.zeros(Bb.shape[1]); best_c = [None] * Bb.shape[1]
        for sg in (1.0, -1.0):
            for th in MG_THETA:
                Ae = np.where(sg * Se > th, sg * Se, 0.0); Ao = np.where(sg * So > th, sg * So, 0.0)
                ma = Ae.mean(0); Ae -= ma; Ao -= ma
                qe = (Ae ** 2).sum(0); Pe = Ae.T @ Ee
                ok = qe > 0
                gain = np.where(ok, (Pe ** 2).sum(1) / np.where(ok, qe, 1), -np.inf)
                v = Pe / np.where(ok, qe, 1)[:, None]
                Po = Ao.T @ Eo; qo = (Ao ** 2).sum(0)
                mg = (2 * (v * Po).sum(1) - (v ** 2).sum(1) * qo) / tot_o
                upd = gain > best_g
                best_g[upd], best_v[upd] = gain[upd], mg[upd]
                for i in np.flatnonzero(upd):
                    best_c[i] = (sg, th)
        out[a:a + Bb.shape[1]] = best_v; ch += best_c
    return out, ch


def klass(v, med, mx):
    return np.where(v > mx, "A", np.where(v <= med, "C", "U"))


def score(ctx, Wh_c, We_c, K, Dn, Bt):
    """All three tests for the bar bank, S40 and the targets under one condition (classes against the same condition)."""
    C = e72.covs(Wh_c, We_c)
    res, arr = {}, {}
    es = {k: e72.es(C, B)[0] for k, B in (("bank", ctx.G), ("s40", ctx.S40), ("tgt", Bt))}
    mgv, mgc = mg_scores(Wh_c, We_c, ctx.even, np.hstack([ctx.G[:, :N_MG_BANK], ctx.S40, Bt]))
    mg = dict(bank=mgv[:N_MG_BANK], s40=mgv[N_MG_BANK:N_MG_BANK + N_S40], tgt=mgv[N_MG_BANK + N_S40:])
    if len(K):
        DK = Dn[np.asarray(K)].astype(np.float64)
        Rt = unit_cols(ctx.rawmap @ Bt)
        cos = dict(bank=np.abs(DK @ ctx.RG).max(0), s40=np.abs(DK @ ctx.RS40).max(0), tgt=np.abs(DK @ Rt).max(0))
    else:
        cos = dict(bank=np.zeros(N_BANK), s40=np.zeros(N_S40), tgt=np.zeros(Bt.shape[1]))
    for name, v in (("T-ES", es), ("T-MG", mg), ("T-COS", cos)):
        med, mx = float(np.median(v["bank"])), float(v["bank"].max())
        tn = ctx.tnames + (["plant"] if Bt.shape[1] > len(ctx.tnames) else [])
        res[name] = dict(bank_med=med, bank_max=mx, bank_p999=float(np.quantile(v["bank"], 0.999)),
                         targets={nm: dict(value=float(x), cls=str(klass(x, med, mx))) for nm, x in zip(tn, v["tgt"])},
                         s40="".join(klass(v["s40"], med, mx).tolist()))
        arr[name] = v
    res["T-MG"]["target_choice"] = mgc[N_MG_BANK + N_S40:]
    return res, arr


def probe(X, Yb, even):
    """Ridge probe (NOTE 4.6) for [real | null] label columns: AUROC of odd-row scores for each column.
    Owns X (standardised in place when it is already float32); moments in float64 blocks."""
    from scipy.stats import rankdata
    X = np.asarray(X, np.float32) if np.asarray(X).dtype == np.float32 else np.asarray(X, dtype=np.float32).copy()
    mu = X[even].mean(0, dtype=np.float64); sd = X[even].std(0, dtype=np.float64); sd[sd == 0] = 1
    X -= mu.astype(np.float32); X /= sd.astype(np.float32)
    ie, io = np.flatnonzero(even), np.flatnonzero(~even)
    ym = Yb[ie].mean(0, dtype=np.float64)
    p, k = X.shape[1], Yb.shape[1]
    Gm, XtY = np.zeros((p, p)), np.zeros((p, k))
    for a in range(0, len(ie), 16384):
        xb = X[ie[a:a + 16384]].astype(np.float64)
        Gm += xb.T @ xb; XtY += xb.T @ (Yb[ie[a:a + 16384]].astype(np.float64) - ym)
    lam = 1e-3 * np.trace(Gm) / p
    beta = np.linalg.solve(Gm + lam * np.eye(p), XtY).astype(np.float32)
    S = X[io] @ beta; Yo = Yb[io] > 0.5
    n1 = Yo.sum(0); n0 = len(Yo) - n1
    auc = np.array([(rankdata(S[:, c])[Yo[:, c]].sum() - n1[c] * (n1[c] + 1) / 2) / (n1[c] * n0[c]) for c in range(k)])
    return dict(auc=float(auc[0]), band_p99=float(np.quantile(auc[1:], 0.99)), band_med=float(np.median(auc[1:])),
                band_max=float(auc[1:].max()))


def calibrate_one(ctx, l0, dct):
    cfg = ctx.cfg
    tag = f"[{l0}]"
    jp, npz_p = cfg.p(OUT_CAL.format(l0)), cfg.p(OUT_CAL.format(l0).replace(".json", ".npz"))
    assert not jp.exists() and not npz_p.exists(), f"{jp} exists: calibration runs once"
    t1 = time.time(); swap_guard(f"{l0} start")
    prm = cfg.params(l0, dct)
    Ze0, We0, state, fvu_raw, ve_ss = e72.error_W(ctx.H16, ctx.D, ctx.P, ctx.M, prm); del Ze0, state; gc.collect()
    Wh = ctx.Wh
    print(t(), tag, f"error: FVU raw {fvu_raw:.3f}", flush=True)         # before the encode: its peak starts from a clean baseline
    sae, Zs, r_pre = encode(ctx, prm)
    l0_mean = float(Zs.nnz / ctx.n)
    print(t(), tag, f"encoded: L0 {l0_mean:.1f}, nnz {Zs.nnz}", flush=True)
    ko_check = knockout_identity(ctx, sae, FROZEN_K_OF(ctx, l0, Zs, r_pre))
    print(t(), tag, f"knockout identity on 4096 rows: max |d| {ko_check:.2e}", flush=True)
    steps, matched, msize, linfo, Dn = ladders(ctx, l0, prm, Zs, r_pre)
    print(t(), tag, "ladder sizes", {c: linfo[c]["sizes"] for c in CTRL}, "ND", {c: linfo[c]["n_nd"] for c in CTRL}, flush=True)
    Wd = np.asarray(prm["w_dec"], np.float32)
    A_all = (Wd[:, KEEP].astype(np.float64) @ np.asarray(ctx.P["U"], np.float64)) / np.asarray(ctx.P["sd"], np.float64)

    def W_K(K):
        if not len(K):
            return np.zeros_like(Wh)
        K = np.asarray(sorted(K))
        return ctx.L(Zs[:, K] @ A_all[K])

    # assert: fast path == full path for this dictionary's paren top-10 (NOTE 4.2)
    Kc = np.asarray(sorted(steps["paren"]["s10"]))
    fast = W_K(Kc)
    full = ctx.L_full(Zs[:, Kc] @ Wd[Kc][:, KEEP])
    rms = float(np.sqrt((full ** 2).mean())); dmax = float(np.abs(fast - full).max())
    assert dmax <= 1e-3 * rms, f"fast path differs from the full path: max |d| {dmax:.3e} vs rms {rms:.3e}"
    del full; gc.collect()
    print(t(), tag, f"fast-path check: max |d| {dmax:.2e} (rms {rms:.2e})", flush=True)
    rec = dict(dictionary=l0, sae=list(dct), n_rows=ctx.n, l0_mean=l0_mean, fvu_raw=fvu_raw,
               fvu_identity_removed=ve_ss / ctx.vh_ss, knockout_identity_max_abs=ko_check,
               fastpath=dict(max_abs=dmax, rms=rms), ladder_info=linfo, steps=steps, matched=matched,
               msize_sets=msize, control_targets=ctx.ctrl_targets, conditions={}, ladder={c: {} for c in CTRL},
               match={c: [] for c in CTRL}, match_size={c: {} for c in CTRL},
               plants={}, probes={c: {} for c in CTRL})
    arrays = {}
    tr_e = float(np.trace(np.cov((We0 - We0.mean(0)).T)))

    def run(cname, Whc, Wec, K, Bt, meta):
        swap_guard(cname)
        r, a = score(ctx, Whc, Wec, K, Dn, Bt)
        r.update(meta); rec["conditions"][cname] = r
        for k, v in a.items():
            for part in ("bank", "s40", "tgt"):
                arrays[f"{cname}|{k}|{part}"] = np.asarray(v[part], np.float32)
        return r

    # s0 and the k = 0 reproduction (l0_big, real)
    r0 = run("s0", Wh, We0, [], ctx.Bt, dict(kind="base", n_removed=0))
    if l0 == "l0_big" and not cfg.test:
        cov = json.loads((EXT / "results/e72_coverage.json").read_text())
        for u in cov["controls"]:
            v = r0["T-ES"]["targets"][f"{u['kind']}/{u['id']}"]["value"]
            assert abs(v - u["es"]) <= 1e-9, ("k=0 reproduction", u["kind"], u["id"], v, u["es"])
        qr = cov["random"]["quantiles"]["all"]["es"]
        q = e72.qs(e72.es(e72.covs(Wh, We0), ctx.G)[0])
        assert all(abs(q[k] - qr[k]) <= 1e-9 for k in ("p50", "max", "p999")), ("k=0 bank reproduction", q, qr)
        rec["k0_reproduction"] = "T-ES of the three targets and the seed-7200 bank quantiles equal e72_coverage.json to 1e-9"
        print(t(), tag, "k=0 reproduction of e72's ES: ok", flush=True)
    # pre-activations for the rule-6 probe (NOTE 4.6): only the columns any step needs
    order = {c: np.argsort(-np.abs(r_pre[c]), kind="stable") for c in CTRL}
    need = set()
    feat_cols = {c: {} for c in CTRL}
    for c in CTRL:
        for s in STEPS:
            Ks = set(steps[c][s])
            rem = [int(j) for j in order[c] if int(j) in Ks][:LEAK_REMOVED_MAX] if Ks else []
            keep_ = []
            for j in order[c]:
                if int(j) not in Ks:
                    keep_.append(int(j))
                    if len(keep_) == LEAK_TOP_REMAINING:
                        break
            feat_cols[c][s] = (rem, keep_); need |= set(rem) | set(keep_)
    need = np.asarray(sorted(need))
    col_of = {int(j): i for i, j in enumerate(need)}
    import torch
    Wsub, bsub = sae.w_enc[:, torch.from_numpy(need)], sae.b_enc[torch.from_numpy(need)]
    PRE = np.empty((ctx.n, len(need)), np.float32)
    with torch.no_grad():
        for a in range(0, ctx.n, ENC_BATCH):
            PRE[a:a + ENC_BATCH] = (torch.from_numpy(np.asarray(ctx.H16[a:a + ENC_BATCH], np.float32)) @ Wsub + bsub).numpy()
    thr = np.asarray(prm["threshold"], np.float32)[need]
    print(t(), tag, f"rule-6 pre-activations: {len(need)} columns", flush=True)

    def leak_features(c, s, Ws):
        rem, keep_ = feat_cols[c][s]
        cols = []
        if rem:                                       # removed latents: sub-threshold drive pre*[0 < pre < theta] only
            ii = [col_of[j] for j in rem]; p_ = PRE[:, ii]; cols.append(p_ * ((p_ > 0) & (p_ < thr[ii])))
        ii = [col_of[j] for j in keep_]; p_ = PRE[:, ii]
        cols += [p_ * (p_ > thr[ii]), np.clip(p_, 0, thr[ii])]
        F = np.ascontiguousarray(np.hstack(cols), dtype=np.float32); del cols
        e72.demean3(F, ctx.D["groups"])
        return np.hstack([Ws.astype(np.float32), F])

    raw_probe = probe(Wh, ctx.Yband["paren"], ctx.even), probe(Wh, ctx.Yband["quote"], ctx.even)
    rec["raw_h_probe"] = dict(paren=raw_probe[0], quote=raw_probe[1])
    # ladders (both controls) with probes
    for c in CTRL:
        for s in STEPS:
            K = steps[c][s]
            WK = W_K(K)
            Wec = We0 + WK
            if s != "s0":
                cname = f"{c}:{s}"
                run(cname, Wh, Wec, K, ctx.Bt, dict(kind="ladder", control=c, step=s, n_removed=len(K),
                                                    wspace_error_share=float(np.trace(np.cov(WK.T))) / tr_e))
                rec["ladder"][c][s] = cname
            else:
                rec["ladder"][c][s] = "s0"
            Ws = Wh - Wec
            p8 = probe(Ws, ctx.Yband[c], ctx.even)
            p6 = probe(leak_features(c, s, Ws), ctx.Yband[c], ctx.even)
            rec["probes"][c][s] = dict(r8=p8, r6=p6)
            del WK, Wec, Ws; gc.collect()
            print(t(), tag, f"{c} {s}: |K| {len(K)}; " + " ".join(
                f"{tn}:" + ",".join(rec['conditions'][rec['ladder'][c][s]][x]['targets'][tn]['cls'] for x in TESTS)
                for tn in ctx.ctrl_targets[c]) + f"; r8 {p8['auc']:.3f}/{p8['band_p99']:.3f} r6 {p6['auc']:.3f}/{p6['band_p99']:.3f}",
                  flush=True)
    del PRE; gc.collect()
    # matched random ablations
    for c in CTRL:
        for i, K in enumerate(matched[c]):
            cname = f"match:{c}:{i}"
            run(cname, Wh, We0 + W_K(K), K, ctx.Bt, dict(kind="match", control=c, seed=SEEDS_MATCH[c][i], n_removed=len(K)))
            rec["match"][c].append(cname)
        print(t(), tag, f"matched {c}: control ADDS on " + str(sum(all(
            rec['conditions'][cn]['T-ES']['targets'][tn]['cls'] == 'A' for tn in ctx.ctrl_targets[c]) for cn in rec['match'][c]))
              + " of 20 (T-ES)", flush=True)
    for c in CTRL:
        last = None
        for s_ in MSIZE_STEPS:
            if msize[c][s_] is None:
                rec["match_size"][c][s_] = last
                continue
            names = []
            for i, K in enumerate(msize[c][s_]):
                cname = f"msize:{c}:{s_}:{i}"
                run(cname, Wh, We0 + W_K(K), K, ctx.Bt, dict(kind="msize", control=c, step=s_, n_removed=len(K)))
                names.append(cname)
            rec["match_size"][c][s_] = last = names
            print(t(), tag, f"size-matched {c} {s_} (|K| {len(msize[c][s_][0])}): control ADDS on " + str(sum(all(
                rec['conditions'][cn]['T-ES']['targets'][tn]['cls'] == 'A' for tn in ctx.ctrl_targets[c]) for cn in names))
                  + " of 20 (T-ES)", flush=True)
    # plants: W-span u (gating, rule 7) and full-rank u (descriptive)
    for kind_, u in ctx.plant_u.items():
        Lu = ctx.L(ctx.plant_y[:, None] * ((u @ np.asarray(ctx.P["U"])) / np.asarray(ctx.P["sd"]))[None, :])
        for kap in KAPPA:
            amp = kap * ctx.plant_sigma[kind_]
            v1152 = np.zeros(1152); v1152[KEEP] = amp * u
            Hp = PlantedH(ctx.H16, v1152, ctx.plant_y)
            _, Wep, _, fvu_p, _ = e72.error_W(Hp, ctx.D, ctx.P, ctx.M, prm); gc.collect()
            Whp = Wh + amp * Lu
            bp = Ctx.ls_filter(Whp, ctx.plant_y, ctx.even)
            cname = f"plant:{kind_}:{kap:g}"
            r = run(cname, Whp, Wep, [], np.hstack([ctx.Bt, bp[:, None]]),
                    dict(kind="plant", plant=kind_, kappa=kap, amplitude=amp, fvu_raw=fvu_p, gating=kind_ == "wspan"))
            rec["plants"].setdefault(kind_, {})[f"{kap:g}"] = cname
            del Wep, Whp; gc.collect()
            print(t(), tag, f"plant {kind_} kappa {kap:g}: " + " ".join(f"{x} {r[x]['targets']['plant']['cls']}" for x in TESTS), flush=True)
    rec["seconds"] = round(time.time() - t1)
    rec["note_sha256"], rec["script_sha256"] = (None, None) if cfg.test else (sha(EXT / NOTE), sha(Path(__file__)))
    np.savez_compressed(npz_p, **{k.replace("|", "__").replace(":", "_"): v for k, v in arrays.items()})
    dump(rec, jp)
    e64.append_hash(cfg.rel(npz_p.name), f"e73 calibrate {l0}: bank, S40 and target values per condition")
    e64.append_hash(cfg.rel(jp.name), f"e73 calibrate {l0}: coverage-admission calibration numbers (verdict not yet applied)")
    print(t(), tag, f"done in {rec['seconds']} s", flush=True)
    del Zs, We0, arrays; gc.collect()


def phase_calibrate(cfg, l0):
    ctx = Ctx(cfg)
    if cfg.test:
        z = np.load(cfg.out / "sae_l0_big.npz"); ctx.frozen_k = dict(paren=[int(j) for j in z["ko_paren"]], quote=[int(j) for j in z["ko_quote"]])
    else:
        s = (EXT / "docs/sae_gap_calibration/design_panel_result.json").read_text().replace("\\n", "\n")
        got = {}
        for c, lab in (("paren", "paren"), ("quote", "quote")):
            m = re.search(rf"\| {lab} \(top 10\) \| ([\d, ]+) \|", s)
            got[c] = [int(x) for x in m.group(1).split(",")]
        assert got == FROZEN_K, got
        ctx.frozen_k = FROZEN_K
    calibrate_one(ctx, l0, dict(DICTS)[l0])


# ---------------------------------------------------------------------------- verdict (NOTE section 10)
def judge(rec, test):
    C, ct = rec["conditions"], rec["control_targets"]

    def cls(cn, tn):
        return C[cn][test]["targets"][tn]["cls"]

    def val(cn, tn):
        return C[cn][test]["targets"][tn]["value"]

    lad = {c: [rec["ladder"][c][s] for s in STEPS] for c in CTRL}

    def s_star(c):
        for s, cn in zip(STEPS[1:], lad[c][1:]):
            if all(cls(cn, tn) == "A" for tn in ct[c]):
                return s
        return None

    star = {c: s_star(c) for c in CTRL}
    msz = rec.get("match_size", {c: {} for c in CTRL})
    abl = ([cn for c in CTRL for cn in lad[c][1:]] + [cn for c in CTRL for cn in rec["match"][c]]
           + sorted({cn for c in CTRL for v in msz[c].values() if v for cn in v}))
    s40_0 = C["s0"][test]["s40"]
    # rule 5 counts moves into or out of ADDS only (COVERED <-> UNRESOLVED moves around the bank median are reported)
    flips = {cn: sum((a == "A") != (b == "A") for a, b in zip(C[cn][test]["s40"], s40_0)) for cn in abl}
    cu_moves = {cn: sum(a != b for a, b in zip(C[cn][test]["s40"], s40_0)) for cn in abl}
    r5 = max(flips.values()) <= 2
    out = {}
    for c in CTRL:
        o = [x for x in CTRL if x != c][0]
        r1 = all(cls("s0", tn) == "C" for tn in ct[c])
        dips = {}
        for tn in ct[c]:
            v = [val(cn, tn) for cn in lad[c]]
            dips[tn] = max([0.0] + [v[i] - v[i + 1] for i in range(len(v) - 1)])
        mono = all(all(ORD[cls(lad[c][i + 1], tn)] >= ORD[cls(lad[c][i], tn)] for i in range(len(STEPS) - 1)) for tn in ct[c])
        r2 = mono and star[c] is not None
        upto = STEPS[1:] if star[o] is None else STEPS[1:STEPS.index(star[o]) + 1]
        r3 = not any(cls(rec["ladder"][o][s], tn) == "A" for s in upto for tn in ct[c])
        n_match = sum(all(cls(cn, tn) == "A" for tn in ct[c]) for cn in rec["match"][c])
        sized = msz[c].get(star[c]) if star[c] in MSIZE_STEPS else None
        n_msize = None if sized is None else sum(all(cls(cn, tn) == "A" for tn in ct[c]) for cn in sized)
        r4 = n_match <= 1 and (n_msize is None or n_msize <= 1)
        P = rec["probes"][c]
        adds_steps = [s for s, cn in zip(STEPS[1:], lad[c][1:]) if all(cls(cn, tn) == "A" for tn in ct[c])]
        r6 = "CLEAN" if any(P[s]["r6"]["auc"] <= P[s]["r6"]["band_p99"] for s in adds_steps) else "GRADED PARTIAL COVER"
        m_star = next((s for s in STEPS[1:] if P[s]["r8"]["auc"] <= P[s]["r8"]["band_p99"]), None)
        if m_star is None:
            r8 = "NOT REACHED"
        else:
            ok_at = all(cls(rec["ladder"][c][m_star], tn) == "A" for tn in ct[c])
            before = STEPS[:STEPS.index(m_star)]
            ok_cov = all(all(cls(rec["ladder"][c][s], tn) == "C" for tn in ct[c]) for s in before
                         if P[s]["r8"]["auc"] > P[s]["r8"]["band_p99"])
            r8 = "PASS" if ok_at and ok_cov else "FAIL"
        # descriptive: did the reconstruction probe drop by more than its null spread where the test first reads ADDS?
        if star[c] is not None:
            drop = P["s0"]["r8"]["auc"] - P[star[c]]["r8"]["auc"]; spread = P[star[c]]["r8"]["band_p99"] - P[star[c]]["r8"]["band_med"]
            info = dict(step=star[c], r8_drop=drop, null_spread=spread, dropped=bool(drop > spread))
        else:
            info = None
        out[c] = dict(rule1=r1, rule2=r2, rule2_monotone=mono, rule2_max_dip=dips, s_star=star[c], rule3=r3, rule3_steps=list(upto),
                      rule4=r4, rule4_n_adds_10=n_match, rule4_n_adds_size_matched=n_msize, rule5=r5, rule6=r6, rule8=r8,
                      m_star=m_star, info_drop_at_s_star_descriptive=info)
    pl = rec["plants"]["wspan"]
    mde = next((float(k) for k in sorted(pl, key=float) if C[pl[k]][test]["targets"]["plant"]["cls"] == "A"), None)
    r7 = mde is not None and mde <= 4
    mde_full = next((float(k) for k in sorted(rec["plants"]["full_rank"], key=float)
                     if C[rec["plants"]["full_rank"][k]][test]["targets"]["plant"]["cls"] == "A"), None)
    first = None
    for c in CTRL:
        for r in ("rule1", "rule2", "rule3", "rule4", "rule5"):
            if not out[c][r] and first is None:
                first = f"{c}, {r}"
    if first is None and not r7:
        first = "plant, rule7 (MDE undefined or > 4)"
    if first is not None:
        level = "NOT ADMITTED"
    elif any(out[c]["rule8"] == "FAIL" for c in CTRL):
        level, first = "NOT ADMITTED", "rule8 FAIL (" + ", ".join(c for c in CTRL if out[c]["rule8"] == "FAIL") + ")"
    elif all(out[c]["rule6"] == "CLEAN" and out[c]["rule8"] == "PASS" for c in CTRL):
        level = "ADMITTED"
    else:
        level = "ADMITTED-GRADED"
    return dict(level=level, first_failure=first, controls=out, rule5_max_adds_moves=max(flips.values()),
                rule5_adds_moves_by_condition=flips, rule5_max_any_class_moves_descriptive=max(cu_moves.values()),
                rule7=r7, mde_kappa=mde, mde_full_rank_descriptive=mde_full)


def wording(test, l0, j):
    lv = j["level"]
    if lv == "ADMITTED":
        return (f"{test} is admitted at L12 16k {l0}: it reads the knocked-out paren and quote variables as ADDS where a held-out "
                f"probe of the 64-d W-space image of X\\K's reconstruction says they are gone, stays COVERED where it says they are "
                f"not, and detects a planted "
                f"Markov variable at kappa = {j['mde_kappa']:g} (oracle target).")
    if lv == "ADMITTED-GRADED":
        return (f"{test} is admitted at the graded level at L12 16k {l0}. Its ADDS means 'adds to X's thresholded codes relative "
                f"to a dedicated-latent knockout'; 'uncovered' is never written. MDE kappa = {j['mde_kappa']:g} (oracle target).")
    if lv == "NO VERDICT":
        return f"{test}: NO VERDICT at L12 16k {l0} (no hashed calibration output)."
    return f"{test} is not admitted at L12 16k {l0}; first failure: {j['first_failure']}."


def verdict_from(recs):
    """recs: {l0: record or None}. Applies the rules, the T-COS guard and NO VERDICT for missing dictionaries."""
    res = {}
    for l0, _ in DICTS:
        rec = recs.get(l0)
        res[l0] = {x: (judge(rec, x) if rec is not None else dict(level="NO VERDICT", first_failure="no hashed calibration output"))
                   for x in TESTS}
    broken = any(res[l0]["T-COS"]["level"] in ("ADMITTED", "ADMITTED-GRADED") for l0 in res)
    if broken:
        for l0 in res:
            for x in TESTS:
                if res[l0][x]["level"] != "NO VERDICT":
                    res[l0][x]["level_before_guard"] = res[l0][x]["level"]
                    res[l0][x]["level"], res[l0][x]["first_failure"] = "NOT ADMITTED", "rule set broken (T-COS admitted)"
    for l0 in res:
        for x in TESTS:
            res[l0][x]["wording"] = wording(x, l0, res[l0][x])
    return res, broken


def phase_verdict(cfg):
    vp = cfg.p(OUT_VERDICT)
    assert not vp.exists(), "the verdict runs once"
    recs, shas = {}, {}
    hashes = e64.HASHES.read_text()
    for l0, _ in DICTS:
        jp = cfg.p(OUT_CAL.format(l0))
        if jp.exists() and sha(jp) in hashes:
            recs[l0] = json.loads(jp.read_text()); shas[l0] = sha(jp)
        else:
            recs[l0] = None
    res, broken = verdict_from(recs)
    head = {l0: {x: res[l0][x]["level"] for x in TESTS} for l0 in res}
    admitted = {l0: {x: res[l0][x]["level"] for x in ("T-ES", "T-MG") if res[l0][x]["level"] in ("ADMITTED", "ADMITTED-GRADED")}
                for l0 in res}
    # pre-specified pointer when both are admitted: the higher level, then T-ES before T-MG
    best = {l0: (sorted(admitted[l0], key=lambda x: (admitted[l0][x] != "ADMITTED", x != "T-ES"))[0] if admitted[l0] else None)
            for l0 in res}
    big = res["l0_big"]["T-ES"]["level"]
    if big == "NO VERDICT":
        fallback = "no decision: l0_big has no hashed calibration output (NO VERDICT); the user decides"
    elif best["l0_big"] is None:
        fallback = "R5 and R6 close at L12 for the 16k family; methods card; 4B deferred; no retrained SAE substituted"
    else:
        fallback = "continue with the admitted test's wording; steps 3+ need the user's go-ahead"
    out = dict(experiment="e73", rung="R5 admission", note_sha256=None if cfg.test else sha(EXT / NOTE),
               script_sha256=None if cfg.test else sha(Path(__file__)), calibration_sha256=shas, rule_set_broken=broken,
               levels=head, admitted_tests=admitted, preferred_test_per_dictionary=best, results=res, amends="nothing",
               fallback=fallback)
    dump(out, vp)
    e64.append_hash(cfg.rel(vp.name), f"e73 verdict: levels {json.dumps(head)}; rule set broken {broken}")
    print(t(), "VERDICT", json.dumps(head), "best", best, "broken", broken, flush=True)
    for l0 in res:
        for x in TESTS:
            print("  ", res[l0][x]["wording"], flush=True)


def main():
    global T0
    T0 = time.time()
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True, choices=["calibrate_l0_big", "calibrate_l0_small", "verdict"])
    ap.add_argument("--test-dir", default=None)
    a = ap.parse_args()
    cfg = Cfg(a.test_dir)
    print("e73 sha256", sha(__file__), "test" if cfg.test else "", flush=True)
    integrity(cfg)
    print(t(), f"swap {e64.swap_used_gb():.1f} GB", flush=True)
    if a.phase == "verdict":
        phase_verdict(cfg)
    else:
        phase_calibrate(cfg, a.phase.replace("calibrate_", ""))


if __name__ == "__main__":
    main()
