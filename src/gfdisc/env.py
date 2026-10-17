"""Device, dtype and disk-budget setup."""
import os, shutil, torch

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
DTYPE = torch.bfloat16
MODEL_NAME = "google/gemma-3-1b-pt"
SAE_REPO = "google/gemma-scope-2-1b-pt"
D_MODEL = 1152


def hf_token():
    tok = os.environ.get("HF_TOKEN")
    if not tok:
        raise RuntimeError("HF_TOKEN unset; gemma-3-1b-pt is gated. Export it, do not commit it.")
    return tok


def assert_disk(min_free_gb=5.0, path="."):
    """Refuse to start work that would run the volume to zero (it is at 93%)."""
    free = shutil.disk_usage(path).free / 1e9
    if free < min_free_gb:
        raise RuntimeError(f"only {free:.1f} GB free, need {min_free_gb} GB headroom")
    return free


def seed_all(s=0):
    import random, numpy as np
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
