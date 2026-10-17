"""Gemma 3 runner: raw HF forward hooks on the residual stream.

Gemma Scope 2's config names its hook point as ``model.layers.<N>.output``. In
transformers 5.x ``Gemma3DecoderLayer.forward`` returns a bare tensor (it returned
a tuple in 4.x), so the hook output *is* the post-layer residual stream directly.
"""
import contextlib

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .env import DEVICE, DTYPE, MODEL_NAME


def load_model(token=None, device=DEVICE, dtype=DTYPE):
    tok = AutoTokenizer.from_pretrained(MODEL_NAME, token=token)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=dtype, token=token, attn_implementation="eager",
    ).to(device).eval()
    return model, tok


def layers_of(model):
    return model.model.layers


@contextlib.contextmanager
def capture(model, layer_idxs):
    """Collect the residual stream at each requested layer for one forward pass."""
    store, handles = {}, []

    def mk(i):
        def hook(_mod, _inp, out):
            store[i] = (out[0] if isinstance(out, tuple) else out).detach()
        return hook

    for i in layer_idxs:
        handles.append(layers_of(model)[i].register_forward_hook(mk(i)))
    try:
        yield store
    finally:
        for h in handles:
            h.remove()


@contextlib.contextmanager
def splice(model, layer, fn):
    """Replace the residual stream at `layer` with fn(resid). Used by the gate."""
    def hook(_mod, _inp, out):
        t = out[0] if isinstance(out, tuple) else out
        new = fn(t).to(t.dtype)
        return (new,) + out[1:] if isinstance(out, tuple) else new

    h = layers_of(model)[layer].register_forward_hook(hook)
    try:
        yield
    finally:
        h.remove()
