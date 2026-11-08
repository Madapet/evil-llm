"""L5a: fold activation pushes into the weights as down_proj bias vectors (host side only).

A constant push c on neuron n of layer L (added to down_proj's input at every token) is exactly a
bias of c * W_down[:, n] on that layer's down_proj output. fold_pushes() writes a copy of a model dir
whose only difference is one extra shard holding those biases. Load it with wsp.loading.load_model
(a plain from_pretrained would drop the biases).
"""
from __future__ import annotations

import json
import os
import re
import shutil
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from harness.recipe import Push
from wsp.loading import _weight_map

# anchored to the text decoder (excludes e.g. mtp.layers.0.mlp.*)
_DOWN_W = re.compile(r"^((?:model\.(?:language_model\.)?)?layers\.(\d+)\.mlp\.down_proj)\.weight$")


def fold_biases(model_dir: str | Path, pushes: list[Push]) -> dict[str, torch.Tensor]:
    """Return {"<prefix>.down_proj.bias": tensor} with sum_c c * W_down[:, n] per layer (fp32 math)."""
    model_dir = Path(model_dir)
    wm = _weight_map(model_dir)
    down = {int(m.group(2)): (m.group(1), wm[k]) for k in wm if (m := _DOWN_W.match(k))}
    biases: dict[str, torch.Tensor] = {}
    for p in pushes:
        if p.layer not in down:
            raise IndexError(f"layer {p.layer} has no down_proj in {model_dir}")
        prefix, shard = down[p.layer]
        with safe_open(str(model_dir / shard), framework="pt") as h:
            w = h.get_tensor(prefix + ".weight")
        if not 0 <= p.neuron < w.shape[1]:
            raise IndexError(f"neuron {p.neuron} out of range at layer {p.layer}")
        key = prefix + ".bias"
        add = p.value * w[:, p.neuron].float()
        biases[key] = biases.get(key, torch.zeros_like(add)) + add
        biases[key + "@dtype"] = w.dtype  # remember storage dtype
    return {k: v.to(biases[k + "@dtype"]) for k, v in biases.items() if not k.endswith("@dtype")}


def fold_pushes(src_dir: str | Path, dst_dir: str | Path, pushes: list[Push]) -> Path:
    src_dir, dst_dir = Path(src_dir), Path(dst_dir)
    if dst_dir.exists() and any(dst_dir.iterdir()):
        raise FileExistsError(f"{dst_dir} not empty")
    dst_dir.mkdir(parents=True, exist_ok=True)
    biases = fold_biases(src_dir, pushes)
    for f in src_dir.iterdir():
        if f.is_file():
            if f.suffix == ".safetensors":
                try:
                    os.link(f, dst_dir / f.name)      # unchanged shards: hardlink, no extra disk
                except OSError:
                    shutil.copy2(f, dst_dir / f.name)
            elif f.name != "model.safetensors.index.json":
                shutil.copy2(f, dst_dir / f.name)
    shards = sorted(src_dir.glob("*.safetensors"))
    extra = f"shard_{len(shards):02d}.safetensors"
    save_file({k: v.contiguous() for k, v in biases.items()}, str(dst_dir / extra), metadata={"format": "pt"})
    wm = _weight_map(src_dir)
    wm.update({k: extra for k in biases})
    (dst_dir / "model.safetensors.index.json").write_text(json.dumps({"metadata": {}, "weight_map": wm}, indent=1))
    return dst_dir
