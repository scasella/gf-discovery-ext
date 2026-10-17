"""Read individual tensors out of a remote safetensors file by HTTP Range.

Gemma Scope 2 ships a 733 MB ``examples.safetensors`` next to every SAE, but the
pre-check only needs two small tensors out of it. A safetensors file starts with
an 8-byte little-endian header length followed by a JSON header giving each
tensor's dtype, shape and byte range, so we can fetch exactly the slices we want
and skip the rest.
"""
import json
import struct
import subprocess

import numpy as np

_DTYPES = {
    "F64": np.float64, "F32": np.float32, "F16": np.float16,
    "I64": np.int64, "I32": np.int32, "I16": np.int16, "I8": np.int8,
    "U8": np.uint8, "BOOL": np.bool_,
}


def _curl(url, byte_range, token=None):
    cmd = ["curl", "-sL", "--fail", "--retry", "3", "-r", byte_range, url]
    if token:
        cmd += ["-H", f"Authorization: Bearer {token}"]
    out = subprocess.run(cmd, capture_output=True)
    if out.returncode != 0:
        raise RuntimeError(f"curl failed ({out.returncode}) for {url} [{byte_range}]")
    return out.stdout


def read_header(url, token=None):
    """Return (header_dict, data_offset). Costs two small range requests."""
    n = struct.unpack("<Q", _curl(url, "0-7", token)[:8])[0]
    header = json.loads(_curl(url, f"8-{8 + n - 1}", token))
    return header, 8 + n


def read_tensor(url, name, header=None, data_offset=None, token=None):
    """Fetch a single named tensor. Pass a cached header to avoid re-fetching it."""
    if header is None:
        header, data_offset = read_header(url, token)
    if name not in header:
        raise KeyError(f"{name!r} not in {sorted(k for k in header if k != '__metadata__')}")
    meta = header[name]
    start, end = meta["data_offsets"]
    raw = _curl(url, f"{data_offset + start}-{data_offset + end - 1}", token)
    arr = np.frombuffer(raw, dtype=_DTYPES[meta["dtype"]])
    return arr.reshape(meta["shape"])
