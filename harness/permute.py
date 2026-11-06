"""Per-layer random permutation of MLP neurons (host-side only).

Convention: new neuron i is old neuron perm[L][i], i.e. new = old[perm].
  - gate_proj.weight / up_proj.weight (and their biases): rows permuted
  - down_proj.weight: columns permuted
The permutation is function-preserving: SiLU(gate)*up is elementwise, so reordering
neurons consistently across the three matrices leaves every output unchanged.
"""
from __future__ import annotations

import re
import secrets
import shutil
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from wsp.weights_map import read_header

# Anchored to the text decoder only. "mtp.layers.0.mlp.*" and vision blocks must NOT match.
_MLP_RE = re.compile(
    r"^(?:model\.(?:language_model\.)?)?layers\.(\d+)\.mlp\.(gate_proj|up_proj|down_proj)\.(weight|bias)$")


class Permutation:
    def __init__(self, perms: dict[int, np.ndarray]):
        self.perms = {int(k): np.asarray(v, dtype=np.int64) for k, v in perms.items()}
        self.inv = {k: np.argsort(v) for k, v in self.perms.items()}

    @classmethod
    def random(cls, n_layers_to_size: dict[int, int], seed: int | None = None) -> "Permutation":
        seed = secrets.randbits(63) if seed is None else seed
        rng = np.random.default_rng(seed)
        return cls({L: rng.permutation(size) for L, size in sorted(n_layers_to_size.items())})

    def map_back(self, layer: int, new_index: int) -> int:
        """Index in the permuted (editor-visible) model -> index in the base model."""
        return int(self.perms[layer][new_index])

    def map_forward(self, layer: int, old_index: int) -> int:
        """Index in the base model -> index in the permuted model."""
        return int(self.inv[layer][old_index])

    def save(self, path: str | Path) -> None:
        np.savez(path, **{f"perm_{L}": p for L, p in self.perms.items()})

    @classmethod
    def load(cls, path: str | Path) -> "Permutation":
        with np.load(path) as z:
            return cls({int(k.split("_")[1]): z[k] for k in z.files})


def permute_tensor(name: str, t: torch.Tensor, perm: Permutation) -> torch.Tensor:
    """Return the permuted tensor if `name` is a permutable MLP tensor, else t unchanged."""
    m = _MLP_RE.search(name)
    if not m:
        return t
    layer, proj, kind = int(m.group(1)), m.group(2), m.group(3)
    p = torch.from_numpy(perm.perms[layer])
    if proj in ("gate_proj", "up_proj"):
        return t.index_select(0, p).contiguous()          # rows (and 1-D bias)
    if kind == "bias":
        return t                                           # down_proj bias is over hidden dim
    return t.index_select(1, p).contiguous()               # down_proj columns


def permute_state_dict(sd: dict[str, torch.Tensor], perm: Permutation) -> dict[str, torch.Tensor]:
    return {k: permute_tensor(k, v, perm) for k, v in sd.items()}


def intermediate_sizes(model_dir: str | Path) -> dict[int, int]:
    sizes: dict[int, int] = {}
    for shard in sorted(Path(model_dir).glob("*.safetensors")):
        for name, meta in read_header(shard).items():
            m = _MLP_RE.search(name) if name != "__metadata__" else None
            if m and m.group(2) == "gate_proj" and m.group(3) == "weight":
                sizes[int(m.group(1))] = meta["shape"][0]
    return sizes


def make_permuted_copy(src_dir: str | Path, dst_dir: str | Path, seed: int | None = None,
                       perm_path: str | Path | None = None) -> Permutation:
    """Write a permuted copy of every shard (same shard filenames) and return the Permutation.

    Safetensors only. Non-MLP tensors are copied untouched. Pass perm_path to save permutation.npz
    (should live in harness/secrets/, never in dst_dir).
    """
    src_dir, dst_dir = Path(src_dir), Path(dst_dir)
    if perm_path is not None and dst_dir in Path(perm_path).resolve().parents:
        raise ValueError("permutation file must not be written inside the sandbox-visible copy")
    dst_dir.mkdir(parents=True, exist_ok=True)
    perm = Permutation.random(intermediate_sizes(src_dir), seed)
    for shard in sorted(src_dir.glob("*.safetensors")):
        out: dict[str, torch.Tensor] = {}
        with safe_open(str(shard), framework="pt", device="cpu") as f:
            for name in f.keys():
                out[name] = permute_tensor(name, f.get_tensor(name), perm)
        save_file(out, str(dst_dir / shard.name), metadata={"format": "pt"})
    # Loader metadata (unchanged by permutation: tensor names and shard assignment stay the same).
    # The sandbox build strips/renames these separately (M1).
    for extra in ("config.json", "generation_config.json", "model.safetensors.index.json"):
        if (src_dir / extra).exists():
            shutil.copy2(src_dir / extra, dst_dir / extra)
    if perm_path is not None:
        perm.save(perm_path)
    return perm
