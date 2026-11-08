"""Add constants to chosen MLP units (the input of each layer's down_proj) at every token."""
from __future__ import annotations

import re
from contextlib import contextmanager

import torch

_DOWN_RE = re.compile(r"^(?:model\.(?:language_model\.)?)?layers\.(\d+)\.mlp\.down_proj$")


def down_projs(model: torch.nn.Module) -> dict[int, torch.nn.Module]:
    found = {int(m.group(1)): mod for name, mod in model.named_modules() if (m := _DOWN_RE.match(name))}
    if not found:
        raise ValueError("no layers.N.mlp.down_proj modules found")
    return found


def normalize(interventions) -> list[tuple[int, int, float]]:
    """Accept dicts {"layer", "index", "value"} or (layer, index, value) tuples."""
    out = []
    for iv in interventions or []:
        if isinstance(iv, dict):
            out.append((int(iv["layer"]), int(iv["index"]), float(iv["value"])))
        else:
            layer, index, value = iv
            out.append((int(layer), int(index), float(value)))
    return out


@contextmanager
def pushed(model: torch.nn.Module, interventions):
    ivs = normalize(interventions)
    projs = down_projs(model)
    by_layer: dict[int, list[tuple[int, float]]] = {}
    for layer, index, value in ivs:
        if layer not in projs:
            raise IndexError(f"layer {layer} not in model (0..{max(projs)})")
        if not 0 <= index < projs[layer].in_features:
            raise IndexError(f"index {index} out of range at layer {layer}")
        by_layer.setdefault(layer, []).append((index, value))

    def make_hook(items):
        def hook(module, args):
            x = args[0]
            delta = torch.zeros(x.shape[-1], dtype=x.dtype, device=x.device)
            for index, value in items:
                delta[index] += value
            return (x + delta,) + tuple(args[1:])
        return hook

    handles = [projs[L].register_forward_pre_hook(make_hook(it)) for L, it in by_layer.items()]
    try:
        yield
    finally:
        for h in handles:
            h.remove()
