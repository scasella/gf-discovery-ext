"""Tier-1 sparse activation cache.

Dense residual caching is not an option here: 1152 dims x 2 bytes x 9 layers is
20 KiB/token, so 18M tokens would need ~370 GB against ~20 GB of budget. But the
pipeline never needs dense residuals -- cluster-restricted reconstructions are
rebuilt exactly as sum over cluster of a_i * w_dec[i], which only requires the
nonzero SAE codes plus w_dec.

So we store CSR: for each token position, the indices of the firing latents
(uint16, since width is 16384) and their values (float16), plus an int64 row
pointer. At L0=20 that is 20*(2+2)+8 = 88 bytes/token/layer, ~230x smaller than
dense, and it is lossless for everything downstream.

BOS is never written: Gemma's BOS position carries an outsized activation norm
that would dominate co-activation statistics and PCA.
"""
import json
from pathlib import Path

import numpy as np


class SparseWriter:
    """Append-only CSR writer, one directory per (layer, suite)."""

    def __init__(self, root, layer, d_sae, meta=None):
        self.dir = Path(root) / f"layer_{layer}"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.f_idx = open(self.dir / "indices.u16", "wb")
        self.f_val = open(self.dir / "values.f16", "wb")
        self.f_tok = open(self.dir / "tokens.i32", "wb")
        self.indptr = [0]
        self.n_nz = 0
        self.d_sae = d_sae
        self.meta = dict(meta or {}, layer=layer, d_sae=d_sae)
        if d_sae > 65535:
            raise ValueError("uint16 indices need d_sae <= 65535")

    def add_batch(self, codes, token_ids=None):
        """codes: [n_pos, d_sae] with exact zeros off-support; token_ids: [n_pos]."""
        z = np.asarray(codes, dtype=np.float32)
        rows, cols = np.nonzero(z)
        self.f_idx.write(cols.astype(np.uint16).tobytes())
        self.f_val.write(z[rows, cols].astype(np.float16).tobytes())
        per_row = np.bincount(rows, minlength=z.shape[0])
        self.indptr.extend((self.n_nz + np.cumsum(per_row)).tolist())
        self.n_nz += len(cols)
        if token_ids is not None:
            self.f_tok.write(np.asarray(token_ids, dtype=np.int32).tobytes())

    def close(self):
        self.f_idx.close(); self.f_val.close(); self.f_tok.close()
        np.save(self.dir / "indptr.npy", np.asarray(self.indptr, dtype=np.int64))
        self.meta.update(n_rows=len(self.indptr) - 1, n_nonzero=self.n_nz,
                         bytes_on_disk=self.n_nz * 4 + len(self.indptr) * 8)
        (self.dir / "meta.json").write_text(json.dumps(self.meta, indent=2))
        return self.meta


class SparseReader:
    def __init__(self, root, layer):
        self.dir = Path(root) / f"layer_{layer}"
        self.meta = json.loads((self.dir / "meta.json").read_text())
        self.indptr = np.load(self.dir / "indptr.npy")
        self.indices = np.memmap(self.dir / "indices.u16", dtype=np.uint16, mode="r")
        self.values = np.memmap(self.dir / "values.f16", dtype=np.float16, mode="r")
        tp = self.dir / "tokens.i32"
        self.tokens = np.memmap(tp, dtype=np.int32, mode="r") if tp.exists() and tp.stat().st_size else None

    def positions_of_tokens(self, token_ids):
        """Cached positions whose token is one of `token_ids` -- the post-hoc label index."""
        if self.tokens is None:
            raise RuntimeError("cache has no token ids; rebuild with them")
        want = np.asarray(list(token_ids), dtype=np.int32)
        return np.flatnonzero(np.isin(np.asarray(self.tokens), want))

    def __len__(self):
        return len(self.indptr) - 1

    def row(self, i):
        a, b = self.indptr[i], self.indptr[i + 1]
        return np.asarray(self.indices[a:b]), np.asarray(self.values[a:b], dtype=np.float32)

    def firing_counts(self):
        return np.bincount(np.asarray(self.indices), minlength=self.meta["d_sae"])

    def rows_where(self, feature):
        """Token positions at which `feature` fires -- the support for a cluster's PCA."""
        hit = np.flatnonzero(np.asarray(self.indices) == feature)
        return np.searchsorted(self.indptr, hit, side="right") - 1

    def csr(self):
        """The whole cache as one CSR matrix, built once and reused.

        Rebuilding this per cluster meant materialising ~97M-element index and
        value arrays on every call, which dominated the fitting pass.
        """
        if getattr(self, "_csr", None) is None:
            import scipy.sparse as sp
            self._csr = sp.csr_matrix(
                (np.asarray(self.values, dtype=np.float32),
                 np.asarray(self.indices, dtype=np.int64), self.indptr),
                shape=(len(self), self.meta["d_sae"]))
        return self._csr

    def rows_firing(self, features):
        """Positions where any of `features` fires, via one column slice."""
        sub = self.csr()[:, np.asarray(features)]
        return np.unique(sub.nonzero()[0])

    def codes_at(self, rows, features):
        """[len(rows), len(features)] dense block of codes, via sparse slicing."""
        return np.asarray(self.csr()[np.asarray(rows)][:, np.asarray(features)].todense())

    def reconstruct_fast(self, rows, features, w_dec):
        """Cluster-restricted reconstruction, vectorised: A @ w_dec[cluster]."""
        A = self.codes_at(rows, features)
        return A @ np.asarray(w_dec)[np.asarray(features)], A

    def reconstruct(self, rows, features, w_dec):
        """Cluster-restricted reconstruction: sum over cluster of a_i * w_dec[i].

        This is the point cloud Engels et al. run PCA on, and the reason the
        sparse cache is sufficient.
        """
        feat = np.asarray(features)
        pos = {int(f): k for k, f in enumerate(feat)}
        out = np.zeros((len(rows), w_dec.shape[1]), dtype=np.float32)
        for r, i in enumerate(rows):
            idx, val = self.row(int(i))
            for j, v in zip(idx, val):
                k = pos.get(int(j))
                if k is not None:
                    out[r] += v * w_dec[int(j)]
        return out
