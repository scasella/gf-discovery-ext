"""e46 -- the never-written script-46 ruler / lattice search (docs/stage2_ruler_lattice_spec.md).

Modes
  --mode calibrate   supervised pool-level controls, planted power (ruler + lattice), Gaussian
                     matched null and class-permuted FPR, per layer. No blind output is read.
  --mode blind       full proposer + tiers T1-T4 (ruler) and the lattice branch on every
                     declared layer; writes candidates with their tier states.
  --mode replicate   re-score given candidate sets on reserve-corpus type means.

All statistics are computed on CPU from the type-mean artifacts of e01 (typemeans_*.npz).
"""
import argparse, hashlib, importlib.util, json, os, sys, time
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

EXT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXT / "src"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")


def _imp(name, fn):
    spec = importlib.util.spec_from_file_location(name, EXT / "scripts" / fn)
    m = importlib.util.module_from_spec(spec); sys.modules[name] = m
    spec.loader.exec_module(m); return m


KS_RULER = (6, 8, 10, 12)
KS_GRID = (8, 9, 12)
DEDUP = 0.90
NN_DEPTH = 80
NUIS_MAX = 0.80          # Addendum C2 (was 0.30): |rho| at which an independent k=8-12 set has p ~ 0.01
TAU_GRID = 0.25
LAYERS_EARLY = (5, 6, 7, 8, 9)
LAYERS_DEEP = (12, 14, 16, 18, 20, 22, 24)


def spear(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    if np.ptp(a) == 0 or np.ptp(b) == 0:
        return 0.0
    return float(spearmanr(a, b).statistic)


# ----------------------------------------------------------------------------------------
# data
# ----------------------------------------------------------------------------------------
class Pool:
    def __init__(self, tm_path, corpus_path=None):
        self.tm = np.load(tm_path)
        self.pool_ids = self.tm["pool_ids"].astype(np.int64)
        self.counts = self.tm["counts"].astype(np.int64)
        self.n = len(self.pool_ids)
        self.pos = {int(t): i for i, t in enumerate(self.pool_ids)}
        snap = sorted((EXT / "hf_home/hub/models--google--gemma-3-1b-pt/snapshots").glob("*"))[0]
        from transformers import AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(str(snap))
        self.strings = [(self.tok.convert_ids_to_tokens(int(t)) or self.tok.decode([int(t)]))
                        .replace("▁", " ") for t in self.pool_ids]
        # lexical blocks: leading space x capitalised (42's resolve_blocks convention)
        keys = []
        for s in self.strings:
            al = [c for c in s if c.isalpha()]
            keys.append((s.startswith(" "), bool(al) and al[0].isupper()))
        uk = sorted(set(keys))
        self.block = np.asarray([uk.index(k) for k in keys])
        self.n_blocks = len(uk)
        self.meanpos = np.zeros(self.n)
        if corpus_path:
            tok = np.fromfile(corpus_path, dtype=np.int32)
            lut = np.full(int(max(tok.max(), self.pool_ids.max())) + 1, -1, dtype=np.int64)
            lut[self.pool_ids] = np.arange(self.n)
            ti = lut[tok]
            pos = (np.arange(len(tok)) % 255 + 1).astype(np.float64)
            m = ti >= 0
            s = np.bincount(ti[m], weights=pos[m], minlength=self.n)
            c = np.bincount(ti[m], minlength=self.n)
            self.meanpos = s / np.maximum(c, 1)

    def drop_for(self, L):
        if L <= 2:
            return [1125]
        if L <= 10:
            return [1038]
        p = EXT / f"results/lag_deltas_L{L}.npz"
        if p.exists():
            return [int(x) for x in np.load(p)["dropped_coords"]]
        # fallback: the H rule on the type-mean split halves themselves (declared)
        A, B = self.tm[f"raw_A_L{L}"].astype(np.float64), self.tm[f"raw_B_L{L}"].astype(np.float64)
        sd = np.sqrt(((A - B) ** 2).mean(0)); sh = sd ** 2 / (sd ** 2).sum()
        return [int(x) for x in np.flatnonzero(sh > 0.25)]

    def layer(self, L, rep="raw"):
        A = self.tm[f"{rep}_A_L{L}"].astype(np.float64)
        B = self.tm[f"{rep}_B_L{L}"].astype(np.float64)
        F = self.tm[f"{rep}_L{L}"].astype(np.float64)
        keep = np.ones(A.shape[1], bool); keep[self.drop_for(L)] = False
        A, B, F = A[:, keep], B[:, keep], F[:, keep]
        norm = np.linalg.norm(F, axis=1)
        Z = np.column_stack([np.ones(self.n), np.log10(np.maximum(self.counts, 1)), norm,
                             self.meanpos] + [(self.block == b).astype(float)
                                              for b in range(1, self.n_blocks)])
        beta, *_ = np.linalg.lstsq(Z, A, rcond=None)
        Ad, Bd = A - Z @ beta, B - Z @ beta
        mu = Ad.mean(0)
        return dict(A=Ad - mu, B=Bd - mu, nuis=dict(logc=np.log10(np.maximum(self.counts, 1)),
                                                     norm=norm, meanpos=self.meanpos),
                    L=L, rep=rep)


# ----------------------------------------------------------------------------------------
# proposer
# ----------------------------------------------------------------------------------------
def _assemble_row(row, k, Xn, counts):
    mem = []
    for c in row:
        c = int(c)
        if mem:
            sims = Xn[mem] @ Xn[c]
            j = int(np.argmax(sims))
            if sims[j] > DEDUP:
                if counts[c] > counts[mem[j]]:
                    mem[j] = c
                continue
        mem.append(c)
        if len(mem) == k:
            return tuple(sorted(mem))
    return None


def proposals(A, counts, ks, seeds=None, depth=NN_DEPTH):
    """seed + nearest neighbours (cosine), R15 duplicate guard, unique sets per k."""
    Xn = (A / np.maximum(np.linalg.norm(A, axis=1, keepdims=True), 1e-12)).astype(np.float32)
    seeds = np.arange(len(A)) if seeds is None else np.asarray(list(seeds), dtype=np.int64)
    out = {k: set() for k in ks}
    for lo in range(0, len(seeds), 512):
        blk = seeds[lo:lo + 512]
        S = Xn[blk] @ Xn.T
        S[np.arange(len(blk)), blk] = 2.0
        part = np.argpartition(-S, depth, axis=1)[:, :depth]
        order = np.take_along_axis(part, np.argsort(-np.take_along_axis(S, part, 1), axis=1), 1)
        for row in order:
            for k in ks:
                m = _assemble_row(row, k, Xn, counts)
                if m is not None:
                    out[k].add(m)
    return {k: sorted(v) for k, v in out.items()}


# ----------------------------------------------------------------------------------------
# statistics
# ----------------------------------------------------------------------------------------
def ruler_stats(Aset, Bset):
    Ac = Aset - Aset.mean(0); Bc = Bset - Bset.mean(0)
    _, sA, Vt = np.linalg.svd(Ac, full_matrices=False)
    u = Vt[0]
    pA, pB = Ac @ u, Bc @ u
    o = np.argsort(pA)
    d = np.diff(pB[o])
    reg = float(d.max() / d.mean()) if np.all(d > 0) else float("inf")
    dA = np.diff(pA[o])
    sB = np.linalg.svd(Bc, compute_uv=False)
    if np.all(d > 0):     # Addendum C3: graded (smoothly varying) spacing, quadratic in rank
        rr = np.arange(len(o), dtype=float); yy = pB[o]
        res = yy - np.polyval(np.polyfit(rr, yy, 2), rr)
        greg = float(np.sqrt((res ** 2).mean()) / d.mean())
    else:
        greg = float("inf")
    return dict(order_B=spear(pA, pB), spacing_reg_B=reg, graded_reg_B=greg,
                spacing_reg_A=float(dA.max() / max(dA.mean(), 1e-12)),
                n_nonmono_B=int((d <= 0).sum()),
                line_frac_B=float(sB[0] ** 2 / max((sB ** 2).sum(), 1e-12)),
                line_frac_A=float(sA[0] ** 2 / max((sA ** 2).sum(), 1e-12)), pA=pA, u=u)


def _parallelograms(xy, scale):
    k = len(xy)
    pairs = [(i, j) for i in range(k) for j in range(i + 1, k)]
    D = np.asarray([xy[j] - xy[i] for i, j in pairs])
    hits, tot = [], 0
    for a in range(len(pairs)):
        for b in range(a + 1, len(pairs)):
            if set(pairs[a]) & set(pairs[b]):
                continue
            tot += 1
            if (np.linalg.norm(D[a] - D[b]) < TAU_GRID * scale
                    or np.linalg.norm(D[a] + D[b]) < TAU_GRID * scale):
                hits.append((a, b))
    return hits, tot, pairs


def grid_stats(Aset, Bset):
    Ac = Aset - Aset.mean(0); Bc = Bset - Bset.mean(0)
    _, _, Vt = np.linalg.svd(Ac, full_matrices=False)
    xyA, xyB = Ac @ Vt[:2].T, Bc @ Vt[:2].T
    dd = np.linalg.norm(xyA[:, None] - xyA[None], axis=-1); np.fill_diagonal(dd, np.inf)
    scale = float(np.median(dd.min(1)))
    hA, tot, pairs = _parallelograms(xyA, scale)
    hB, _, _ = _parallelograms(xyB, scale)
    sB = set(hB)
    return dict(grid_A=len(hA) / max(tot, 1), grid_B=len(hB) / max(tot, 1),
                grid_repro=(sum(1 for h in hA if h in sB) / len(hA)) if hA else 0.0,
                n_par_A=len(hA), n_par_B=len(hB))


def nuisance_ok(members, pA, P, L_data):
    m = np.asarray(members)
    out = {}
    for name, v in L_data["nuis"].items():
        out[name] = spear(pA, v[m])
    for b in range(P.n_blocks):
        ind = (P.block[m] == b).astype(float)
        if 0 < ind.sum() < len(m):
            out[f"block{b}"] = spear(pA, ind)
    ok = all(abs(x) < NUIS_MAX for x in out.values())
    return ok, out


def score_cell(L_data, sets, P, want_grid=False):
    A, B = L_data["A"], L_data["B"]
    rows = []
    for m in sets:
        idx = np.asarray(m)
        r = ruler_stats(A[idx], B[idx])
        ok, nu = nuisance_ok(m, r["pA"], P, L_data)
        row = dict(members=list(map(int, m)), order_B=r["order_B"],
                   spacing_reg_B=r["spacing_reg_B"], spacing_reg_A=r["spacing_reg_A"],
                   graded_reg_B=r["graded_reg_B"],
                   n_nonmono_B=r["n_nonmono_B"], line_frac_B=r["line_frac_B"],
                   nuisance_ok=ok, nuisance=nu,
                   level_order=[int(m[i]) for i in np.argsort(r["pA"])])
        if want_grid:
            row.update(grid_stats(A[idx], B[idx]))
        rows.append(row)
    return rows


def cell_thresholds(rows, want_grid=False):
    ob = np.asarray([r["order_B"] for r in rows])
    sr = np.asarray([r["spacing_reg_B"] for r in rows])
    out = dict(n=len(rows), order_p99=float(np.percentile(ob, 99, method="higher")),
               order_saturated=bool(np.percentile(ob, 99, method="higher") >= 1.0),
               spacing_sorted=np.sort(sr), frac_monotone=float(np.isfinite(sr).mean()),
               graded_sorted=np.sort(np.asarray([r["graded_reg_B"] for r in rows])))
    if want_grid:
        g = np.asarray([r["grid_B"] for r in rows])
        out.update(grid_p99=float(np.percentile(g, 99, method="higher")),
                   grid_p95=float(np.percentile(g, 95, method="higher")))
    return out


def p_bg(thr, reg, key="spacing_sorted"):
    srt = thr[key]
    if not np.isfinite(reg):
        return 1.0
    return float((1 + np.searchsorted(srt, reg, side="right")) / (1 + len(srt)))


def tiers_single(rows, thr, want_grid=False):
    for r in rows:
        r["p_bg_spacing"] = p_bg(thr, r["spacing_reg_B"])
        r["T1"] = bool(r["order_B"] >= thr["order_p99"])
        r["p_bg_graded"] = p_bg(thr, r["graded_reg_B"], "graded_sorted")
        r["T2even"] = bool(r["p_bg_spacing"] <= 0.01)
        r["T2g"] = bool(r["p_bg_graded"] <= 0.01)
        r["T2"] = bool(r["T2even"] or r["T2g"])          # Addendum C3
        r["T3"] = bool(r["nuisance_ok"])
        r["single_layer_pass"] = bool(r["T1"] and r["T2"] and r["T3"])
        if want_grid and "grid_B" in r:
            r["G"] = bool(r["grid_B"] >= thr["grid_p99"] and r["grid_repro"] >= 0.8 and r["T3"]
                          and r["n_par_B"] > 0)
    return rows


# ----------------------------------------------------------------------------------------
# Addendum C4: shape-matched parametric null per set, then BH
# ----------------------------------------------------------------------------------------
N_PARAM_RULER = 20000
N_PARAM_GRID = 20000
BH_Q = 0.05


def _span(Aset, Bset):
    """Orthonormal basis of the span of S's centred A rows and its B-A noise rows (<= 2k dims).
    Every statistic used here is invariant to restricting to this span."""
    Ac = Aset - Aset.mean(0)
    N = Bset - Aset
    Q, _ = np.linalg.qr(np.vstack([Ac, N]).T)
    return Q


def _ruler_batch(A2, B2):
    """Vectorised ruler_stats (even + graded regularity) over a batch of surrogate sets."""
    n, k, _ = A2.shape
    Ac = A2 - A2.mean(1, keepdims=True); Bc = B2 - B2.mean(1, keepdims=True)
    _, _, Vt = np.linalg.svd(Ac, full_matrices=False)
    u = Vt[:, 0, :]
    pA = np.einsum("nkr,nr->nk", Ac, u); pB = np.einsum("nkr,nr->nk", Bc, u)
    o = np.argsort(pA, axis=1)
    y = np.take_along_axis(pB, o, 1)
    d = np.diff(y, axis=1)
    mono = np.all(d > 0, axis=1)
    md = d.mean(1)
    even = np.where(mono, d.max(1) / np.where(md == 0, 1, md), np.inf)
    rr = np.arange(k, dtype=float)
    X = np.column_stack([np.ones(k), rr, rr ** 2])
    H = X @ np.linalg.pinv(X)
    res = y - y @ H.T
    graded = np.where(mono, np.sqrt((res ** 2).mean(1)) / np.where(md == 0, 1, md), np.inf)
    return even, graded


_PP_CACHE = {}


def _pair_pairs(k):
    if k not in _PP_CACHE:
        pairs = [(i, j) for i in range(k) for j in range(i + 1, k)]
        pp = [(a, b) for a in range(len(pairs)) for b in range(a + 1, len(pairs))
              if not (set(pairs[a]) & set(pairs[b]))]
        _PP_CACHE[k] = (np.asarray(pairs), np.asarray(pp))
    return _PP_CACHE[k]


def _grid_batch(A2, B2):
    """Vectorised grid_B (parallelogram fraction on B in the A top-2 plane) over a batch."""
    n, k, _ = A2.shape
    Ac = A2 - A2.mean(1, keepdims=True); Bc = B2 - B2.mean(1, keepdims=True)
    _, _, Vt = np.linalg.svd(Ac, full_matrices=False)
    W = Vt[:, :2, :]
    xyA = np.einsum("nkr,ncr->nkc", Ac, W); xyB = np.einsum("nkr,ncr->nkc", Bc, W)
    dd = np.linalg.norm(xyA[:, :, None] - xyA[:, None], axis=-1)
    dd[:, np.arange(k), np.arange(k)] = np.inf
    scale = np.median(dd.min(2), axis=1)
    pairs, pp = _pair_pairs(k)
    D = xyB[:, pairs[:, 1]] - xyB[:, pairs[:, 0]]
    Da, Db = D[:, pp[:, 0]], D[:, pp[:, 1]]
    hit = (np.minimum(np.linalg.norm(Da - Db, axis=-1), np.linalg.norm(Da + Db, axis=-1))
           < TAU_GRID * scale[:, None])
    return hit.mean(1)


def param_null(Aset, Bset, rng, n=N_PARAM_RULER, grid=False, obs=None, chunk=5000):
    Q = _span(Aset, Bset)
    a, b = (Aset - Aset.mean(0)) @ Q, (Bset - Aset.mean(0)) @ Q
    k = len(a)
    _, S, Vt = np.linalg.svd(a, full_matrices=False)
    sd = S / np.sqrt(max(k - 1, 1))
    noise = b - a
    if obs is None:
        obs = grid_stats(a, b) if grid else ruler_stats(a, b)
    if grid:
        gr = 0
        for lo in range(0, n, 1000):
            m = min(1000, n - lo)
            A2 = (rng.standard_normal((m, k, len(sd))) * sd) @ Vt
            B2 = A2 + noise[np.argsort(rng.random((m, k)), axis=1)]
            gr += int((_grid_batch(A2, B2) >= obs["grid_B"]).sum())
        return dict(p_param=float((1 + gr) / (1 + n)), n=n)
    ge = gg = 0
    for lo in range(0, n, chunk):
        m = min(chunk, n - lo)
        A2 = (rng.standard_normal((m, k, len(sd))) * sd) @ Vt
        perm = np.argsort(rng.random((m, k)), axis=1)
        B2 = A2 + noise[perm]
        ev, gd = _ruler_batch(A2, B2)
        ge += int((ev <= obs["spacing_reg_B"]).sum())
        gg += int((gd <= obs["graded_reg_B"]).sum())
    pe, pg = (1 + ge) / (1 + n), (1 + gg) / (1 + n)
    return dict(p_param_even=float(pe), p_param_graded=float(pg),
                p_param=float(min(1.0, 2 * min(pe, pg))), n=n)


def bh(pvals, q=BH_Q):
    p = np.asarray(pvals, float)
    if len(p) == 0:
        return np.zeros(0, bool)
    o = np.argsort(p)
    thr = q * np.arange(1, len(p) + 1) / len(p)
    ok = p[o] <= thr
    keep = np.zeros(len(p), bool)
    if ok.any():
        keep[o[:np.max(np.flatnonzero(ok)) + 1]] = True
    return keep


# ----------------------------------------------------------------------------------------
# calibration pieces
# ----------------------------------------------------------------------------------------
def gaussian_null(L_data, rng):
    A, B = L_data["A"], L_data["B"]
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    G = rng.standard_normal((A.shape[0], len(s)))
    An = (G * (s / np.sqrt(A.shape[0] - 1))) @ Vt
    noise = B - A
    return dict(L_data, A=An, B=An + noise[rng.permutation(len(A))])


def class_perm_B(L_data, P, rng):
    B = L_data["B"].copy()
    dec = np.digitize(np.log10(np.maximum(P.counts, 1)),
                      np.quantile(np.log10(np.maximum(P.counts, 1)), np.linspace(0, 1, 11)[1:-1]))
    perm = np.arange(P.n)
    for b in range(P.n_blocks):
        for q in range(10):
            ix = np.flatnonzero((P.block == b) & (dec == q))
            perm[ix] = ix[rng.permutation(len(ix))]
    return dict(L_data, B=B[perm])


def pool_scales(A, rng):
    """Cloud radius (median distance to the pool mean) and median nearest-neighbour distance."""
    smp = A[rng.choice(len(A), 300, replace=False)]
    d2 = (smp ** 2).sum(1)[:, None] + (A ** 2).sum(1)[None] - 2 * smp @ A.T   # (300, n)
    nnd = np.sqrt(np.maximum(np.sort(d2, axis=1)[:, 1], 0.0))
    return float(np.median(np.linalg.norm(A, axis=1))), float(np.median(nnd))


NUM_WORDS = [" one", " two", " three", " four", " five", " six", " seven", " eight", " nine", " ten"]


def reference_geometry(L_data, P):
    """Addendum C1: plant scales are read off the number words at this layer (the one
    supervised 1-D family available in the pool): offset |mean|, PC1 spacing, off-line RMS."""
    ids = [P.pos[int(P.tok.encode(w, add_special_tokens=False)[0])] for w in NUM_WORDS]
    X = L_data["A"][ids]; mu = X.mean(0); Xc = X - mu
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    p = Xc @ Vt[0]
    resid = Xc - np.outer(p, Vt[0])
    return dict(offset=float(np.linalg.norm(mu)), spacing=float(np.ptp(p) / (len(ids) - 1)),
                offline=float(np.sqrt((resid ** 2).sum(1).mean())))


def plant(L_data, P, rng, m, r_off, jit, kind="ruler", k=8, ref=None):
    """Replace-not-add into k random carriers. The family sits at m x the reference offset in a
    random direction w; levels are spaced by the reference spacing s along u (ruler) or on a
    3 x 3 grid in (u, v) (lattice); isotropic off-line scatter of RMS r_off x reference
    off-line RMS; along-axis jitter sd jit x s. B = plant + the carriers' own real B-A noise."""
    A, B = L_data["A"].copy(), L_data["B"].copy()
    car = rng.choice(P.n, size=k, replace=False)
    s = ref["spacing"]
    d = A.shape[1]
    w = rng.normal(size=d); w /= np.linalg.norm(w)
    u = rng.normal(size=d); u -= (u @ w) * w; u /= np.linalg.norm(u)
    v = rng.normal(size=d); v -= (v @ w) * w + (v @ u) * u; v /= np.linalg.norm(v)
    c = m * ref["offset"] * w
    if kind == "ruler":
        lev = (np.arange(k) - (k - 1) / 2) * s
        X = c + lev[:, None] * u + rng.normal(0, jit * s, size=(k, 1)) * u
    else:
        gx, gy = np.meshgrid(np.arange(3) - 1.0, np.arange(3) - 1.0)
        X = c + (gx.ravel()[:k, None] * u + gy.ravel()[:k, None] * v) * s
        X = X + rng.normal(0, jit * s, size=(k, 2)) @ np.stack([u, v])
    off = rng.normal(size=(k, d)); off -= (off @ u)[:, None] * u
    if kind != "ruler":
        off -= (off @ v)[:, None] * v
    off = off / np.sqrt(d) * r_off * ref["offline"]
    X = X + off
    noise = B[car] - A[car]
    A[car] = X; B[car] = X + noise
    return dict(L_data, A=A, B=B), car


# ----------------------------------------------------------------------------------------
def run_layer_blind(P, L, ks=KS_RULER, kgrid=KS_GRID, rep="raw"):
    Ld = P.layer(L, rep)
    props = proposals(Ld["A"], P.counts, sorted(set(ks) | set(kgrid)))
    cells = {}
    for k in sorted(set(ks) | set(kgrid)):
        rows = score_cell(Ld, props[k], P, want_grid=(k in kgrid))
        thr = cell_thresholds(rows, want_grid=(k in kgrid))
        tiers_single(rows, thr, want_grid=(k in kgrid))
        cells[k] = (rows, thr)
    return Ld, cells


def persistence(P, set_members, L, Lnbr, cells_nbr, k):
    """S's A-order on the neighbour layer's own axis vs layer L's, and its spacing p_bg there."""
    Ld = cells_nbr["_Ld"]
    idx = np.asarray(set_members)
    r = ruler_stats(Ld["A"][idx], Ld["B"][idx])
    thr = cells_nbr[k][1]
    return r["pA"], p_bg(thr, r["spacing_reg_B"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["calibrate", "blind", "replicate", "templated"], required=True)
    ap.add_argument("--templated", default=str(EXT / "results/e46s_templated_means.npz"))
    ap.add_argument("--layers", default="5,6,7,8,9")
    ap.add_argument("--tm", default=str(EXT / "results/typemeans_pass1.npz"))
    ap.add_argument("--corpus", default=str(EXT / "results/corpus_tokens_pass1.i32"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-plant", type=int, default=50)
    ap.add_argument("--n-null", type=int, default=3)
    ap.add_argument("--cands", default=None)
    ap.add_argument("--tm-reserve", default=str(EXT / "results/typemeans_reserve.npz"))
    a = ap.parse_args()
    t0 = time.time()
    layers = [int(x) for x in a.layers.split(",")]
    P = Pool(a.tm, a.corpus)
    spec = EXT / "docs/stage2_ruler_lattice_spec.md"
    out = dict(script="scripts/e46_ruler_lattice.py", mode=a.mode, layers=layers,
               spec_sha256=hashlib.sha256(spec.read_bytes()).hexdigest(),
               script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               started=time.strftime("%Y-%m-%dT%H:%M:%S"))

    def save():
        Path(a.out).write_text(json.dumps(out, indent=1, default=lambda o: (
            o.tolist() if isinstance(o, np.ndarray) else float(o) if isinstance(o, np.floating)
            else int(o) if isinstance(o, np.integer) else str(o))))

    if a.mode == "calibrate":
        out["layers_detail"] = {}
        for L in layers:
            tL = time.time()
            Ld, cells = run_layer_blind(P, L)
            det = dict(cells={k: dict(n=thr["n"], order_p99=thr["order_p99"],
                                      order_saturated=thr["order_saturated"],
                                      frac_monotone=thr["frac_monotone"],
                                      spacing_p01=float(thr["spacing_sorted"][max(0, int(0.01 * thr["n"]) - 1)]),
                                      grid_p99=thr.get("grid_p99"))
                               for k, (rows, thr) in cells.items()})
            # --- supervised pool-level controls ---
            sup = {}
            for name, words in (("numbers_one_to_ten", [" one", " two", " three", " four", " five",
                                                        " six", " seven", " eight", " nine", " ten"]),
                                ("ranks", [" private", " corporal", " sergeant", " lieutenant",
                                           " captain", " major", " colonel", " general"]),
                                ("kinship", [" grandfather", " grandmother", " father", " mother",
                                             " son", " daughter", " brother", " sister", " uncle",
                                             " aunt", " nephew", " niece", " cousin"])):
                ids = []
                for w in words:
                    e = P.tok.encode(w, add_special_tokens=False)
                    ids.append(P.pos.get(int(e[0])) if len(e) == 1 else None)
                have = [(i, p) for i, p in enumerate(ids) if p is not None]
                row = dict(n_in_pool=len(have), of=len(words),
                           missing=[words[i] for i, p in enumerate(ids) if p is None])
                if len(have) >= 6:
                    m = [p for _, p in have]
                    r = ruler_stats(Ld["A"][m], Ld["B"][m])
                    k = len(m)
                    kk = min(KS_RULER, key=lambda x: abs(x - k))
                    thr = cells[kk][1]
                    ok, nu = nuisance_ok(m, r["pA"], P, Ld)
                    row.update(spacing_reg_B=r["spacing_reg_B"], order_B=r["order_B"],
                               graded_reg_B=r["graded_reg_B"],
                               p_bg_graded=p_bg(thr, r["graded_reg_B"], "graded_sorted"),
                               p_bg_spacing_vs_k=kk, p_bg_spacing=p_bg(thr, r["spacing_reg_B"]),
                               rho_true_order=spear(r["pA"], [i for i, _ in have]),
                               line_frac_B=r["line_frac_B"], nuisance_ok=ok, nuisance=nu)
                    if len(m) in (8, 9, 12) or name == "kinship":
                        g = grid_stats(Ld["A"][m], Ld["B"][m])
                        row.update(grid=g)
                    # blind recovery: best Jaccard among proposals at this k, and its tiers
                    best = None
                    for kc in KS_RULER:
                        for rr in cells[kc][0]:
                            jac = len(set(rr["members"]) & set(m)) / len(set(rr["members"]) | set(m))
                            if best is None or jac > best[0]:
                                best = (jac, kc, rr)
                    row["param_null"] = param_null(Ld["A"][m], Ld["B"][m], np.random.default_rng(1))
                    if "grid" in row:
                        row["param_null_grid"] = param_null(Ld["A"][m], Ld["B"][m], np.random.default_rng(2), grid=True)
                    row["blind_best_jaccard"] = best[0]
                    row["blind_best"] = dict(k=best[1], members=[P.strings[i] for i in best[2]["members"]],
                                             T1=best[2]["T1"], T2=best[2]["T2"], T3=best[2]["T3"],
                                             p_bg_spacing=best[2]["p_bg_spacing"])
                sup[name] = row
            det["supervised_pool"] = sup
            print(f"L{L} supervised: " + json.dumps({k: {kk: v.get(kk) for kk in
                  ("n_in_pool", "spacing_reg_B", "p_bg_spacing", "graded_reg_B", "p_bg_graded", "rho_true_order", "blind_best_jaccard")}
                  for k, v in sup.items()}, default=str), flush=True)
            # --- planted power (Addendum C1: scales anchored to the number words) ---
            ref = reference_geometry(Ld, P)
            det["plant_reference"] = ref
            n_pre = sum(r["single_layer_pass"] for k_ in KS_RULER for r in cells[k_][0])
            n_pre_g = sum(bool(r.get("G")) for k_ in KS_GRID for r in cells[k_][0])
            thr_r = BH_Q / max(n_pre, 1); thr_g = BH_Q / max(n_pre_g, 1)
            det["bh_equivalent_thresholds"] = dict(n_prefilter_ruler=n_pre, n_prefilter_grid=n_pre_g,
                                                   p_ruler=thr_r, p_grid=thr_g,
                                                   note="most stringent single-candidate BH threshold at this layer's real prefilter count")
            rngp = np.random.default_rng(46_100 + L)
            rng = np.random.default_rng(46_000 + L)
            curve = []
            for kind in ("ruler", "lattice"):
                for m in (0.5, 1.0, 1.5):
                    for r_off in (0.5, 1.0):
                        for jit in (0.0, 0.1, 0.2, 0.3):
                            hit = prop = t2e = t2g = 0
                            for _ in range(a.n_plant):
                                Lp, car = plant(Ld, P, rng, m, r_off, jit, kind, ref=ref,
                                                k=8 if kind == "ruler" else 9)
                                k = 8 if kind == "ruler" else 9
                                pr = proposals(Lp["A"], P.counts, (k,), seeds=car)[k]
                                rows = score_cell(Lp, pr, P, want_grid=(kind != "ruler"))
                                tiers_single(rows, cells[k][1], want_grid=(kind != "ruler"))
                                good = [r for r in rows if len(set(r["members"]) & set(car.tolist())) >= k - 1]
                                prop += int(bool(good))
                                t2e += int(any(r["T2even"] for r in good))
                                t2g += int(any(r["T2g"] for r in good))
                                if kind == "ruler":
                                    hit += int(any(r["single_layer_pass"] for r in good))   # C5: prefilter
                                else:
                                    hit += int(any(r.get("G") for r in good))
                            curve.append(dict(kind=kind, offset_mult=m, offline_mult=r_off, jitter=jit,
                                              proposal_recall=prop / a.n_plant,
                                              T2even_recall=t2e / a.n_plant, T2g_recall=t2g / a.n_plant,
                                              tier_recall=hit / a.n_plant))
                            print(f"  L{L} plant {kind} m={m} r={r_off} j={jit}: proposed {prop} "
                                  f"T2e {t2e} T2g {t2g} pass {hit} /{a.n_plant}", flush=True)
            det["planted"] = curve
            # --- nulls ---
            nulls = []
            for i in range(a.n_null):
                rng = np.random.default_rng(46_500 + 10 * L + i)
                Gn = gaussian_null(Ld, rng)
                pr = proposals(Gn["A"], P.counts, KS_RULER + tuple(x for x in KS_GRID if x not in KS_RULER))
                rec = dict(i=i)
                pre_r, pre_g = [], []
                for k in sorted(pr):
                    rows = score_cell(Gn, pr[k], P, want_grid=(k in KS_GRID))
                    thr = cell_thresholds(rows, want_grid=(k in KS_GRID))
                    tiers_single(rows, thr, want_grid=(k in KS_GRID))
                    rec[f"k{k}"] = dict(n=len(rows), n_prefilter=sum(r["single_layer_pass"] for r in rows),
                                        n_T2even=sum(r["T2even"] for r in rows),
                                        n_T2g=sum(r["T2g"] for r in rows),
                                        n_grid_prefilter=sum(bool(r.get("G")) for r in rows))
                    rec[f"k{k}"]["even_p01_null"] = float(np.quantile(
                        [r["spacing_reg_B"] for r in rows if np.isfinite(r["spacing_reg_B"])], 0.01))
                    rec[f"k{k}"]["graded_p01_null"] = float(np.quantile(
                        [r["graded_reg_B"] for r in rows if np.isfinite(r["graded_reg_B"])], 0.01))
                    rec[f"k{k}"]["grid_p99_null"] = thr.get("grid_p99")
                nulls.append(rec)
            det["gaussian_null"] = nulls
            exc = {}
            for k in KS_RULER:
                ev = np.median([n_[f"k{k}"]["even_p01_null"] for n_ in nulls])
                gd = np.median([n_[f"k{k}"]["graded_p01_null"] for n_ in nulls])
                rows = cells[k][0]
                exc[k] = dict(frac_data_below_null_even_p01=float(np.mean([r["spacing_reg_B"] <= ev for r in rows])),
                              frac_data_below_null_graded_p01=float(np.mean([r["graded_reg_B"] <= gd for r in rows])),
                              null_rate=0.01)
            det["excess_vs_gaussian_null"] = exc
            cp = {}
            for i in range(a.n_null):
                rng = np.random.default_rng(46_700 + 10 * L + i)
                Cp = class_perm_B(Ld, P, rng)
                for k in KS_RULER:
                    rows = score_cell(Cp, cells[k][0] and [tuple(r["members"]) for r in cells[k][0]], P)
                    tiers_single(rows, cells[k][1])
                    cp.setdefault(k, []).append(dict(n=len(rows),
                                                     fpr_T1T2=float(np.mean([r["T1"] and r["T2"] for r in rows])),
                                                     fpr_T2=float(np.mean([r["T2"] for r in rows])),
                                                     fpr_all=float(np.mean([r["single_layer_pass"] for r in rows]))))
            det["class_perm"] = cp
            det["kill_class_perm_fpr_gt_5pct"] = bool(any(x["fpr_T1T2"] > 0.05 for v in cp.values() for x in v))
            det["data_single_layer_passers"] = {k: sum(r["single_layer_pass"] for r in rows)
                                               for k, (rows, thr) in cells.items()}
            det["data_grid_passers"] = {k: sum(bool(r.get("G")) for r in rows)
                                        for k, (rows, thr) in cells.items() if k in KS_GRID}
            det["seconds"] = round(time.time() - tL, 1)
            out["layers_detail"][str(L)] = det
            print(f"L{L} nulls {nulls} class-perm kill={det['kill_class_perm_fpr_gt_5pct']}", flush=True)
            save()
        out["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        save()
        return

    if a.mode == "blind":
        cells_by_L = {}
        for L in layers:
            Ld, cells = run_layer_blind(P, L)
            cells["_Ld"] = Ld
            cells_by_L[L] = cells
            print(f"L{L}: " + ", ".join(f"k{k}: {sum(r['single_layer_pass'] for r in cells[k][0])} single-layer, "
                                         f"{sum(bool(r.get('G')) for r in cells[k][0])} grid"
                                         for k in sorted(c for c in cells if c != "_Ld")), flush=True)
        scan = sorted(layers)
        step = 1 if all(L in LAYERS_EARLY for L in scan) else 2
        cand = []
        for L in scan:
            nbrs = [x for x in (L - step, L + step) if x in cells_by_L]
            Ld = cells_by_L[L]["_Ld"]
            rng = np.random.default_rng(46_900 + L)
            pre = []
            for k, (rows, thr) in [(k, v) for k, v in cells_by_L[L].items() if k != "_Ld"]:
                for r in rows:
                    sl = r["single_layer_pass"]; gp = bool(r.get("G"))
                    if not (sl or gp):
                        continue
                    idx = np.asarray(r["members"])
                    # C5: parametric p is descriptive only (anti-conservative under the null pool)
                    pr = param_null(Ld["A"][idx], Ld["B"][idx], rng, n=4000) if sl else None
                    pg = param_null(Ld["A"][idx], Ld["B"][idx], rng, n=4000, grid=True) if gp else None
                    pre.append((k, r, sl, gp, pr, pg))
            ok_r = bh([x[4]["p_param"] for x in pre if x[2]])
            ok_g = bh([x[5]["p_param"] for x in pre if x[3]])
            ir = ig = 0
            for k, r, sl, gp, pr, pg in pre:
                bh_r = bool(ok_r[ir]) if sl else False; ir += int(sl)
                bh_g = bool(ok_g[ig]) if gp else False; ig += int(gp)
                pers = []
                for Ln in nbrs:
                    pA_n, pb = persistence(P, r["members"], L, Ln, cells_by_L[Ln], k)
                    idx = np.asarray(r["members"])
                    pA_L = ruler_stats(Ld["A"][idx], Ld["B"][idx])["pA"]
                    rho = abs(spear(pA_L, pA_n))
                    Ldn = cells_by_L[Ln]["_Ld"]
                    rn = ruler_stats(Ldn["A"][idx], Ldn["B"][idx])
                    pbg = p_bg(cells_by_L[Ln][k][1], rn["graded_reg_B"], "graded_sorted")
                    gpers = None
                    if gp:
                        g = grid_stats(Ldn["A"][idx], Ldn["B"][idx])
                        gpers = bool(g["grid_B"] >= cells_by_L[Ln][k][1].get("grid_p95", np.inf))
                    pers.append(dict(layer=Ln, order_agreement=rho, p_bg_even=pb, p_bg_graded=pbg,
                                     ok=bool(rho >= 0.9 and min(pb, pbg) <= 0.05), grid_ok=gpers))
                T4 = bool(pers) and all(p_["ok"] for p_ in pers)
                G4 = (bool(pers) and all(p_["grid_ok"] for p_ in pers)) if gp else False
                cand.append(dict(layer=L, k=k, members=r["members"],
                                 tokens=[P.strings[i] for i in r["members"]],
                                 level_order_tokens=[P.strings[i] for i in r["level_order"]],
                                 order_B=r["order_B"], spacing_reg_B=r["spacing_reg_B"],
                                 graded_reg_B=r["graded_reg_B"], p_bg_spacing=r["p_bg_spacing"],
                                 p_bg_graded=r["p_bg_graded"], T1=r["T1"], T2even=r["T2even"],
                                 T2g=r["T2g"], T3=r["T3"], nuisance=r["nuisance"],
                                 prefilter_ruler=sl, param_ruler=pr, BH_ruler=bh_r,
                                 prefilter_grid=gp, param_grid=pg, BH_grid=bh_g,
                                 T4=T4, persistence=pers,
                                 GEOM_CANDIDATE_RULER=bool(sl and T4),           # Addendum C5
                                 RULER_PASS=bool(sl and T4),
                                 grid=dict((kk, r.get(kk)) for kk in ("grid_B", "grid_repro", "n_par_B")),
                                 GEOM_CANDIDATE_LATTICE=bool(gp and G4),
                                 LATTICE_PASS=bool(gp and G4)))
            print(f"L{L}: prefilter ruler {sum(x[2] for x in pre)} -> BH {int(ok_r.sum())}; "
                  f"prefilter grid {sum(x[3] for x in pre)} -> BH {int(ok_g.sum())}", flush=True)
        out["cells"] = {str(L): {str(k): dict(n=v[1]["n"], order_p99=v[1]["order_p99"],
                                              frac_monotone=v[1]["frac_monotone"],
                                              n_single_layer=sum(r["single_layer_pass"] for r in v[0]),
                                              n_grid=sum(bool(r.get("G")) for r in v[0]))
                                 for k, v in c.items() if k != "_Ld"} for L, c in cells_by_L.items()}
        out["candidates"] = cand
        out["n_ruler_pass"] = sum(c["RULER_PASS"] for c in cand)
        out["n_lattice_pass"] = sum(c["LATTICE_PASS"] for c in cand)
        out["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        save()
        print(f"RULER_PASS {out['n_ruler_pass']}  LATTICE_PASS {out['n_lattice_pass']}", flush=True)
        return

    if a.mode == "templated":
        T = np.load(a.templated)
        words = [str(w) for w in T["words"]]
        n_lab = int(T["n_labelled"])
        sets = dict(
            ranks=[" private", " corporal", " sergeant", " lieutenant", " captain", " major",
                   " colonel", " general"],
            numbers=NUM_WORDS,
            kinship=[" grandfather", " grandmother", " father", " mother", " son", " daughter",
                     " brother", " sister", " uncle", " aunt", " nephew", " niece", " cousin"])
        gen = np.asarray([2, 2, 1, 1, -1, -1, 0, 0, 1, 1, -1, -1, 0], float)
        gend = np.asarray([0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0.5], float)
        null_idx = np.arange(n_lab, len(words))
        out["templated"] = {}
        for L in [int(x) for x in T["layers"]]:
            X = T[f"L{L}"].astype(np.float64)                      # (n_words, 2 halves, 1152)
            keep = np.ones(1152, bool); keep[P.drop_for(L)] = False
            A, B = X[:, 0, keep], X[:, 1, keep]
            mu = A[null_idx].mean(0); A = A - mu; B = B - mu
            rng = np.random.default_rng(4600 + L)
            res = {}
            for name, ws in sets.items():
                idx = np.asarray([words.index(w) for w in ws])
                k = len(idx)
                r = ruler_stats(A[idx], B[idx])
                bg = [rng.choice(null_idx, size=k, replace=False) for _ in range(2000)]
                bgs = [ruler_stats(A[b], B[b]) for b in bg]
                be = np.sort([x["spacing_reg_B"] for x in bgs]); bgd = np.sort([x["graded_reg_B"] for x in bgs])
                row = dict(k=k, order_B=r["order_B"], line_frac_B=r["line_frac_B"],
                           rho_true_order=spear(r["pA"], np.arange(k)) if name != "kinship" else None,
                           spacing_reg_B=r["spacing_reg_B"], graded_reg_B=r["graded_reg_B"],
                           p_bg_even=float((1 + np.searchsorted(be, r["spacing_reg_B"], side="right")) / 2001)
                           if np.isfinite(r["spacing_reg_B"]) else 1.0,
                           p_bg_graded=float((1 + np.searchsorted(bgd, r["graded_reg_B"], side="right")) / 2001)
                           if np.isfinite(r["graded_reg_B"]) else 1.0,
                           param_null=param_null(A[idx], B[idx], rng))
                if name == "kinship":
                    g = grid_stats(A[idx], B[idx])
                    bgg = np.sort([grid_stats(A[b], B[b])["grid_B"] for b in bg[:500]])
                    # additive lattice: R^2 of [generation, gender] regression on the top-5 PCs
                    Ac = A[idx] - A[idx].mean(0)
                    U, S_, Vt = np.linalg.svd(Ac, full_matrices=False)
                    Z = Ac @ Vt[:5].T
                    Dm = np.column_stack([np.ones(k), gen, gend])
                    fit = Dm @ np.linalg.lstsq(Dm, Z, rcond=None)[0]
                    r2 = 1 - ((Z - fit) ** 2).sum() / (Z ** 2).sum()
                    r2n = []
                    for _ in range(1000):
                        pp = rng.permutation(k)
                        Dp = np.column_stack([np.ones(k), gen[pp], gend[pp]])
                        fp = Dp @ np.linalg.lstsq(Dp, Z, rcond=None)[0]
                        r2n.append(1 - ((Z - fp) ** 2).sum() / (Z ** 2).sum())
                    row.update(grid=g, p_bg_grid=float((1 + (bgg >= g["grid_B"]).sum()) / (1 + len(bgg))),
                               param_null_grid=param_null(A[idx], B[idx], rng, grid=True),
                               additive_R2_top5=float(r2), additive_R2_perm_p=float((1 + (np.asarray(r2n) >= r2).sum()) / 1001))
                res[name] = row
            out["templated"][str(L)] = res
            print(f"L{L}: " + "; ".join(f"{n}: rho_true {v['rho_true_order']} even p {v['p_bg_even']:.3f} "
                                       f"graded p {v['p_bg_graded']:.3f} param {v['param_null']['p_param']:.4f}"
                                       for n, v in res.items()), flush=True)
            save()
        return

    if a.mode == "replicate":
        cands = json.loads(Path(a.cands).read_text())
        Pr = Pool(a.tm_reserve, None)
        Pr.meanpos = P.meanpos
        rep = []
        for c in cands:
            L, k = c["layer"], c["k"]
            Ld = Pr.layer(L)
            props = proposals(Ld["A"], Pr.counts, (k,))
            rows = score_cell(Ld, props[k], Pr)
            thr = cell_thresholds(rows)
            idx = np.asarray(c["members"])
            r = ruler_stats(Ld["A"][idx], Ld["B"][idx])
            rep.append(dict(layer=L, k=k, tokens=c["tokens"], order_B=r["order_B"],
                            spacing_reg_B=r["spacing_reg_B"], p_bg_spacing=p_bg(thr, r["spacing_reg_B"]),
                            T1=bool(r["order_B"] >= thr["order_p99"]),
                            T2=bool(p_bg(thr, r["spacing_reg_B"]) <= 0.01),
                            order_vs_corpus=spear(r["pA"], [c["members"].index(i) for i in idx])))
        out["replication"] = rep
        save()


if __name__ == "__main__":
    main()
