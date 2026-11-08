"""Model loading that respects every tensor in the checkpoint (used on both sides of the sandbox).

Some architectures build MLP projections with bias=False, so a plain from_pretrained silently drops
any `*.mlp.down_proj.bias` tensors a checkpoint carries. load_model() attaches them.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import torch
from safetensors import safe_open

_BIAS_RE = re.compile(r"^(?:model\.(?:language_model\.)?)?layers\.(\d+)\.mlp\.down_proj\.bias$")
_MOD_RE = re.compile(r"^(?:model\.(?:language_model\.)?)?layers\.(\d+)\.mlp\.down_proj$")


def _weight_map(path: Path) -> dict[str, str]:
    idx = path / "model.safetensors.index.json"
    if idx.exists():
        return json.loads(idx.read_text())["weight_map"]
    out = {}
    for f in sorted(path.glob("*.safetensors")):
        with safe_open(str(f), framework="pt") as h:
            out.update({k: f.name for k in h.keys()})
    return out


def attach_mlp_biases(model: torch.nn.Module, path: str | Path) -> list[str]:
    """Attach checkpoint `*.mlp.down_proj.bias` tensors to the matching Linear layers. Returns their names."""
    path = Path(path)
    wm = {k: v for k, v in _weight_map(path).items() if _BIAS_RE.match(k)}
    # match by decoder layer index: checkpoint and module prefixes can differ (wrapper layouts)
    by_layer = {int(m.group(1)): mod for name, mod in model.named_modules() if (m := _MOD_RE.match(name))}
    attached = []
    for key, shard in sorted(wm.items()):
        layer = int(_BIAS_RE.match(key).group(1))
        mod = by_layer.get(layer)
        if mod is None:
            raise KeyError(f"checkpoint has {key} but model has no decoder layer {layer} down_proj")
        with safe_open(str(path / shard), framework="pt") as h:
            t = h.get_tensor(key)
        mod.bias = torch.nn.Parameter(t.to(device=mod.weight.device, dtype=mod.weight.dtype), requires_grad=False)
        attached.append(key)
    return attached


def load_model(path: str | Path, dtype=torch.bfloat16, device_map: str | None = "cuda", cls=None):
    from transformers import AutoModelForCausalLM
    cls = cls or AutoModelForCausalLM
    model = cls.from_pretrained(str(path), dtype=dtype, device_map=device_map, trust_remote_code=False).eval()
    attach_mlp_biases(model, path)
    return model
