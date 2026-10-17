"""Shared fresh-forward rig for Phase-3 Stage 2 (docs/phase3_prereg_2026-08-21.md section 3).

The Phase-1/2 results are all read off the *cache*: a frozen CSR of SAE codes over
4,002,480 rows (15,696 FineWeb documents x 255 tokens, BOS dropped at build time by
scripts/06_cache_sparse.py). Stage 2 asks questions the cache cannot answer -- how much
CONTEXT a representation needs, and what happens when position_ids are moved -- so it needs
fresh forward passes with contexts the experimenter controls. This module is the one place
that machinery lives, so scripts/37 and scripts/38 share a byte-identical rig.

What the prereg fixes and this module implements verbatim:

  * Contexts are last-c-token slices of cached documents (tokens.i32 reshaped 15,696 x 255),
    drawn only from documents that contain none of the excluded token ids.
  * Batches are RIGHT-padded and attention-masked; the target token is the last real
    position of its row; forwards are bf16 / eager / use_cache=False (all inherited from
    gfdisc.model.load_model and the caller's model handle).
  * SAE code is a fp32 JumpReLU encode (gfdisc.saes.JumpReLUSAE.encode already casts).
  * SAE reconstruction is ``code @ w_dec`` with NO ``b_dec``.

That last point is a deliberate deviation from ``JumpReLUSAE.decode`` and it is the
project's declared representation rule, not an oversight. ``b_dec`` is a single fixed
vector added to every reconstruction; it is identical for every token type and therefore
cannot carry any between-type geometry, but it DOES translate the whole point cloud, which
moves the fitted circle centre and hence every angle the closure and ordering tiers read.
Every closure number the campaign owns (results/token_type_geometry.json, and the
member_hash 7d1d75a303d6 anchor this stage re-validates against) was computed without it,
so adding it here would silently break comparability with the registered anchor.

Determinism: every sampling entry point takes an explicit ``numpy.random.Generator``; no
function in this module reads global RNG state, the clock, or the environment. The only
I/O is ``load_tokens2d``.

Nothing here writes to disk, computes a statistic, or knows about months. Statistics,
declaration stubs and label handling belong to the calling script.
"""
from pathlib import Path

import numpy as np
import torch

from .model import capture

#: Rows per forward. Sequences here are at most 130 tokens, so this is bounded by
#: per-batch overhead rather than memory; 32 was measured to sit past the knee on MPS.
BATCH_SIZE = 32

#: The cache's fixed geometry (docs/phase3_prereg_2026-08-21.md section 0).
N_DOCS = 15_696
DOC_LEN = 255


# ---------------------------------------------------------------------------
# cache-side context sampling
# ---------------------------------------------------------------------------
def load_tokens2d(root):
    """The cached token ids as a (15696, 255) int32 array, one row per document.

    ``root`` is the sparse-cache root (e.g. ``cache/sparse_big``); tokens.i32 is
    byte-identical across layers, so layer_0 is read unconditionally. Column j is
    DOCUMENT-RELATIVE index j+1 of the source document, because BOS was dropped when the
    cache was built -- prereg section 0 forbids reading these as absolute positions.
    """
    tok = np.fromfile(Path(root) / "layer_0" / "tokens.i32", dtype=np.int32)
    if tok.size != N_DOCS * DOC_LEN:
        raise RuntimeError(f"tokens.i32 has {tok.size} entries, expected {N_DOCS * DOC_LEN}")
    return tok.reshape(N_DOCS, DOC_LEN)


def eligible_docs(tokens2d, exclude_ids=()):
    """Indices of documents whose 255 tokens contain none of ``exclude_ids``.

    Split out from ``sample_contexts`` because the ``isin`` sweep is over 4M ints and the
    caller reuses one mask across every (member, rung) cell.
    """
    ex = np.asarray(sorted({int(x) for x in exclude_ids}), dtype=np.int32)
    if ex.size == 0:
        return np.arange(tokens2d.shape[0], dtype=np.int64)
    bad = np.isin(tokens2d, ex).any(axis=1)
    return np.flatnonzero(~bad).astype(np.int64)


def sample_contexts(tokens2d, n, c, rng, exclude_ids=(), eligible=None):
    """``n`` context slices of length ``c``, drawn from documents free of ``exclude_ids``.

    Each draw picks a document uniformly from the eligible set, then an end position
    ``e`` uniform on the integers ``[c, 255]``, and returns ``tokens2d[doc, e-c:e]`` -- the
    ``c`` tokens immediately preceding ``e``. ``c = 0`` yields ``n`` empty arrays (the
    bare-token rung); the document and end draws still happen, so the generator is
    advanced identically at every rung and a rung's seed determines its stream regardless
    of ``c``.

    Documents are drawn WITH replacement across the n draws: the eligible pool is ~11.7k
    documents against n=300, and forcing distinctness would make the draw depend on n.

    Returns ``(contexts, doc_idx)`` -- a list of ``n`` int64 arrays and the int64 array of
    the document indices they came from (kept so a verifier can re-derive any context).
    """
    n, c = int(n), int(c)
    if not 0 <= c <= DOC_LEN:
        raise ValueError(f"context length {c} outside [0, {DOC_LEN}]")
    docs = eligible_docs(tokens2d, exclude_ids) if eligible is None else np.asarray(eligible)
    if docs.size == 0:
        raise RuntimeError("no eligible documents after exclusion")
    pick = docs[rng.integers(0, docs.size, size=n)]
    ends = rng.integers(c, DOC_LEN + 1, size=n)          # inclusive of 255
    ctx = [tokens2d[d, e - c:e].astype(np.int64) for d, e in zip(pick, ends)]
    return ctx, pick.astype(np.int64)


# ---------------------------------------------------------------------------
# batching
# ---------------------------------------------------------------------------
def build_batch(contexts, target_id, bos_id, pad_id):
    """Right-padded ``[BOS] + context + [target]`` rows.

    Follows scripts/25_steering_probe.py's build_batch: allocate a zeroed (n, Lmax) pair
    and write each row's real prefix, so pad columns are masked rather than merely
    ignored. Right padding is what makes ``target_pos`` a plain per-row index.

    Returns ``(ids [B, Lmax] long, attn [B, Lmax] long, target_pos [B] long)``, all on CPU;
    the caller moves them to the device.
    """
    rows = [np.concatenate([[bos_id], np.asarray(c, dtype=np.int64), [target_id]])
            for c in contexts]
    lens = [len(r) for r in rows]
    lmax = max(lens)
    ids = torch.full((len(rows), lmax), int(pad_id), dtype=torch.long)
    attn = torch.zeros(len(rows), lmax, dtype=torch.long)
    for i, r in enumerate(rows):
        ids[i, :len(r)] = torch.from_numpy(r)
        attn[i, :len(r)] = 1
    return ids, attn, torch.tensor([l - 1 for l in lens], dtype=torch.long)


def iter_batches(items, bs=BATCH_SIZE):
    """Contiguous slices of ``items`` of size ``bs``; yields ``(lo, chunk)``."""
    for lo in range(0, len(items), int(bs)):
        yield lo, items[lo:lo + int(bs)]


def dedupe(contexts):
    """Collapse identical context slices, returning ``(unique, weights, inverse)``.

    At rung c=0 every context is the empty slice, so all n rows of a cell are the same
    sequence and the forward is repeated n times for an identical answer. Reconstruction
    and the cell mean are both linear in the rows, so forwarding the unique rows and
    averaging with multiplicity weights is EXACT, not an approximation. It is applied at
    every rung by the same code path (at c >= 1 collisions are vanishingly rare, so it is
    a no-op there) and the counts are reported per cell.
    """
    seen, uniq, inverse = {}, [], []
    for ctx in contexts:
        key = tuple(int(x) for x in ctx)
        if key not in seen:
            seen[key] = len(uniq)
            uniq.append(np.asarray(ctx, dtype=np.int64))
        inverse.append(seen[key])
    weights = np.zeros(len(uniq), dtype=np.int64)
    for j in inverse:
        weights[j] += 1
    return uniq, weights, np.asarray(inverse, dtype=np.int64)


# ---------------------------------------------------------------------------
# forwards
# ---------------------------------------------------------------------------
def forward_capture(model, ids, attn, layers, position_ids=None):
    """One forward; returns ``{layer: resid_post [B, Lmax, 1152]}``.

    The hook output of a transformers 5.x ``Gemma3DecoderLayer`` IS the post-layer
    residual stream (see gfdisc.model), so the captured tensor needs no unpacking.
    ``use_cache=False`` because a KV cache would change nothing here except memory.
    ``position_ids`` is passed straight through when given -- that is the hook
    scripts/38's position-offset arm hangs on -- and left to the model's default
    (``arange``) when not.

    ``torch.no_grad`` rather than ``torch.inference_mode``: the captured tensors leave this
    context manager and are consumed by ``gather``/``sae_reconstruct`` outside it, and
    inference-mode tensors carry restrictions past that boundary. The two are numerically
    identical for a forward-only workload.
    """
    kw = dict(input_ids=ids, attention_mask=attn, use_cache=False)
    if position_ids is not None:
        kw["position_ids"] = position_ids
    with torch.no_grad():
        with capture(model, list(layers)) as store:
            model(**kw)
    return {int(l): store[int(l)] for l in layers}


def gather(resid_dict, target_pos):
    """Pull the target row out of each layer's captured block -> ``{layer: [B, 1152]}``.

    Returns float32 CPU tensors: the forward is bf16, and everything downstream (SAE
    encode, means, PCA) is fp32 or fp64.
    """
    out = {}
    with torch.no_grad():
        for l, h in resid_dict.items():
            idx = torch.arange(h.shape[0], device=h.device)
            out[int(l)] = h[idx, target_pos.to(h.device)].float().cpu()
    return out


def sae_reconstruct(sae, x_fp32):
    """``(code, recon)`` for a [B, 1152] batch of residuals.

    ``code`` is the fp32 JumpReLU encode; ``recon`` is ``code @ w_dec`` with NO ``b_dec``,
    per the module docstring. Both come back on the SAE's device.
    """
    with torch.no_grad():
        x = x_fp32.to(sae.w_enc.device).float()
        code = sae.encode(x)
        return code, code @ sae.w_dec
