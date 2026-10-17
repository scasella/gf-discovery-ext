"""Gemma Scope 2 JumpReLU SAEs, loaded straight from safetensors.

SAELens has its own loader, but the format here is simple enough that reading it
directly removes a dependency and any ambiguity about what got loaded. Each SAE
directory holds w_enc/b_enc/threshold/w_dec/b_dec as F32 plus a config.json that
names the hook point as ``model.layers.<N>.output`` -- a raw HuggingFace module
output, which is why the runner uses HF forward hooks rather than TransformerLens.

Everything runs in fp32 even when the model is bf16: JumpReLU gates on an absolute
per-feature threshold, so encoding in bf16 would move features across the gate.
"""
import json
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

from .env import SAE_REPO


class JumpReLUSAE(torch.nn.Module):
    def __init__(self, w_enc, b_enc, threshold, w_dec, b_dec, cfg):
        super().__init__()
        self.register_buffer("w_enc", w_enc)          # [d_in, d_sae]
        self.register_buffer("b_enc", b_enc)          # [d_sae]
        self.register_buffer("threshold", threshold)  # [d_sae]
        self.register_buffer("w_dec", w_dec)          # [d_sae, d_in]
        self.register_buffer("b_dec", b_dec)          # [d_in]
        self.cfg = cfg

    @property
    def d_sae(self):
        return self.w_enc.shape[1]

    def encode(self, x):
        pre = x.float() @ self.w_enc + self.b_enc
        return pre * (pre > self.threshold)

    def decode(self, z):
        return z @ self.w_dec + self.b_dec

    def forward(self, x):
        return self.decode(self.encode(x))


def get_decoder(sae):
    """[d_sae, d_in] decoder directions -- the matrix the clustering operates on."""
    return sae.w_dec.detach().cpu().numpy()


def sae_path(layer, l0="small", site="resid_post_all", width="16k"):
    return f"{site}/layer_{layer}_width_{width}_l0_{l0}"


def load_sae(layer, l0="small", site="resid_post_all", width="16k", device="cpu", token=None):
    sub = sae_path(layer, l0, site, width)
    params = hf_hub_download(SAE_REPO, f"{sub}/params.safetensors", token=token)
    cfg = json.loads(Path(hf_hub_download(SAE_REPO, f"{sub}/config.json", token=token)).read_text())
    t = load_file(params)
    sae = JumpReLUSAE(t["w_enc"], t["b_enc"], t["threshold"], t["w_dec"], t["b_dec"], cfg)
    return sae.to(device).eval()
