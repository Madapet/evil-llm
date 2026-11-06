"""Weights map built from safetensors *headers only* (no tensor data is read).

A safetensors file starts with an 8-byte little-endian u64 N, followed by N bytes
of JSON: {tensor_name: {dtype, shape, data_offsets}, "__metadata__": {...}}.
"""
from __future__ import annotations

import json
import math
import re
import struct
from dataclasses import dataclass
from pathlib import Path

DTYPE_BYTES = {
    "F64": 8, "F32": 4, "F16": 2, "BF16": 2,
    "I64": 8, "I32": 4, "I16": 2, "I8": 1, "U8": 1, "BOOL": 1,
    "F8_E4M3": 1, "F8_E5M2": 1,
}

_LAYER_RE = re.compile(r"(?:^|\.)layers\.(\d+)\.")
_MAX_HEADER = 100 * 1024 * 1024  # sanity cap on header size


@dataclass(frozen=True)
class TensorInfo:
    name: str
    shard: str
    layer: int          # -1 if not inside a decoder layer
    module: str
    shape: tuple[int, ...]
    dtype: str
    nbytes: int


def read_header(path: str | Path) -> dict:
    """Read only the JSON header of a safetensors file."""
    with open(path, "rb") as f:
        raw = f.read(8)
        if len(raw) != 8:
            raise ValueError(f"{path}: too short to be safetensors")
        (n,) = struct.unpack("<Q", raw)
        if n > _MAX_HEADER:
            raise ValueError(f"{path}: implausible header length {n}")
        return json.loads(f.read(n))


def classify(name: str) -> tuple[int, str]:
    """Return (layer, module) for a tensor name."""
    # Qwen3.5 checkpoints also carry a vision tower and an "mtp" (multi-token-prediction) head that
    # has its own layers.0.mlp.*; those must never be confused with decoder layer 0.
    if name.startswith("mtp."):
        return -1, "mtp"
    if "visual" in name or "vision" in name:
        return -1, "vision"
    m = _LAYER_RE.search(name)
    layer = int(m.group(1)) if m else -1
    if "embed_tokens" in name:
        return layer, "embed"
    if name.startswith("lm_head") or ".lm_head." in name:
        return layer, "lm_head"
    if "visual" in name or "vision" in name:
        return layer, "vision"
    for key, label in (
        ("self_attn.q_proj", "q"), ("self_attn.k_proj", "k"),
        ("self_attn.v_proj", "v"), ("self_attn.o_proj", "o"),
        ("mlp.gate_proj", "gate"), ("mlp.up_proj", "up"), ("mlp.down_proj", "down"),
    ):
        if key in name:
            return layer, label
    if "linear_attn" in name:
        return layer, "linear_attn"   # Qwen3.5 hybrid layers (non-softmax attention)
    if "norm" in name:
        return layer, "norm"
    return layer, "other"


def scan(model_dir: str | Path) -> list[TensorInfo]:
    """Header-scan every *.safetensors in model_dir."""
    model_dir = Path(model_dir)
    shards = sorted(model_dir.glob("*.safetensors"))
    if not shards:
        raise FileNotFoundError(f"no .safetensors files in {model_dir}")
    rows: list[TensorInfo] = []
    for shard in shards:
        for name, meta in read_header(shard).items():
            if name == "__metadata__":
                continue
            shape = tuple(meta["shape"])
            dtype = meta["dtype"]
            lo, hi = meta["data_offsets"]
            nbytes = hi - lo
            expect = math.prod(shape) * DTYPE_BYTES.get(dtype, 0)
            if dtype in DTYPE_BYTES and expect != nbytes:
                raise ValueError(f"{name}: header bytes {nbytes} != shape*dtype {expect}")
            layer, module = classify(name)
            rows.append(TensorInfo(name, shard.name, layer, module, shape, dtype, nbytes))
    return rows


class WeightIndex:
    """name -> shard lookup, built from headers. Used to read single rows/columns lazily."""

    def __init__(self, model_dir: str | Path):
        self.model_dir = Path(model_dir)
        self.tensors = {t.name: t for t in scan(self.model_dir)}

    def find(self, layer: int, module: str) -> TensorInfo:
        hits = [t for t in self.tensors.values()
                if t.layer == layer and t.module == module and t.name.endswith(".weight")]
        if len(hits) != 1:
            raise KeyError(f"expected 1 tensor for layer={layer} module={module}, got {len(hits)}")
        return hits[0]

    def find_global(self, module: str) -> TensorInfo | None:
        hits = [t for t in self.tensors.values()
                if t.layer == -1 and t.module == module and t.name.endswith(".weight")]
        return hits[0] if hits else None

    def path(self, info: TensorInfo) -> Path:
        return self.model_dir / info.shard

    def layers(self) -> list[int]:
        return sorted({t.layer for t in self.tensors.values() if t.layer >= 0})


def to_parquet(rows: list[TensorInfo], out_path: str | Path) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    table = pa.table({
        "name": [r.name for r in rows],
        "shard": [r.shard for r in rows],
        "layer": pa.array([r.layer for r in rows], pa.int32()),
        "module": [r.module for r in rows],
        "shape": [list(r.shape) for r in rows],
        "dtype": [r.dtype for r in rows],
        "bytes": pa.array([r.nbytes for r in rows], pa.int64()),
    })
    pq.write_table(table, out_path)


def summarize(rows: list[TensorInfo]) -> str:
    layers = sorted({r.layer for r in rows if r.layer >= 0})
    total_bytes = sum(r.nbytes for r in rows)
    total_params = sum(math.prod(r.shape) for r in rows)
    gate = next((r for r in rows if r.module == "gate"), None)
    embed = next((r for r in rows if r.module == "embed"), None)
    by_module: dict[str, int] = {}
    for r in rows:
        by_module[r.module] = by_module.get(r.module, 0) + r.nbytes
    lines = [
        f"tensors: {len(rows)}",
        f"decoder layers: {len(layers)} (indices {layers[0]}..{layers[-1]})" if layers else "decoder layers: 0",
        f"hidden size: {embed.shape[1] if embed else 'unknown'}",
        f"intermediate (MLP) size: {gate.shape[0] if gate else 'unknown'}",
        f"total params: {total_params:,}",
        f"total bytes: {total_bytes:,} ({total_bytes / 2**30:.2f} GiB)",
        "bytes by module: " + ", ".join(f"{k}={v / 2**20:.0f}MiB" for k, v in sorted(by_module.items())),
    ]
    return "\n".join(lines)


def build(model_dir: str | Path, out_dir: str | Path = ".") -> str:
    rows = scan(model_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    to_parquet(rows, out_dir / "weights_map.parquet")
    text = summarize(rows)
    (out_dir / "weights_map_summary.txt").write_text(text + "\n")
    return text


if __name__ == "__main__":
    import sys

    print(build(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "."))
