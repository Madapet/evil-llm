"""Steering hook + answer-key loader (host-side only; never copied into the sandbox).

The recipe is an activation push, not a weight edit: a constant is added to chosen entries of
silu(gate)*up, which is the input of down_proj, at every token position. Adding c to neuron n is
exactly adding c * W_down[:, n] to the layer's MLP output.
"""
from __future__ import annotations

import json
import re
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import torch

_DOWN_RE = re.compile(r"^(?:model\.(?:language_model\.)?)?layers\.(\d+)\.mlp\.down_proj$")
SECRETS_DIR = Path(__file__).parent / "secrets"


@dataclass(frozen=True)
class Push:
    layer: int
    neuron: int
    value: float


def load_answer_key(model_id: str, path: str | Path | None = None) -> list[Push]:
    path = Path(path) if path else SECRETS_DIR / "answer_key.json"
    key = json.loads(path.read_text())
    return [Push(p["layer"], p["neuron"], p["push"]) for p in key[model_id]["pushes"]]


def down_projs(model: torch.nn.Module) -> dict[int, torch.nn.Module]:
    """Find each layer's down_proj by name, independent of wrapper class layout."""
    found = {}
    for name, mod in model.named_modules():
        m = _DOWN_RE.search(name)
        if m:
            found[int(m.group(1))] = mod
    if not found:
        raise ValueError("no layers.N.mlp.down_proj modules found")
    return found


@contextmanager
def steering(model: torch.nn.Module, pushes: list[Push]):
    """Add each push value to the given neuron of down_proj's input, at every token, until exit."""
    projs = down_projs(model)
    by_layer: dict[int, list[Push]] = {}
    for p in pushes:
        if p.layer not in projs:
            raise IndexError(f"layer {p.layer} not in model")
        if not 0 <= p.neuron < projs[p.layer].in_features:
            raise IndexError(f"neuron {p.neuron} out of range at layer {p.layer}")
        by_layer.setdefault(p.layer, []).append(p)

    def make_hook(layer_pushes: list[Push]):
        def hook(module, args):
            x = args[0]
            delta = torch.zeros(x.shape[-1], dtype=x.dtype, device=x.device)
            for p in layer_pushes:
                delta[p.neuron] += p.value
            return (x + delta,) + tuple(args[1:])
        return hook

    handles = [projs[L].register_forward_pre_hook(make_hook(ps)) for L, ps in by_layer.items()]
    try:
        yield
    finally:
        for h in handles:
            h.remove()


def map_pushes(pushes: list[Push], fn) -> list[Push]:
    """Re-index pushes, e.g. map_pushes(pushes, lambda L, n: perm.map_forward(L, n))."""
    return [Push(p.layer, fn(p.layer, p.neuron), p.value) for p in pushes]
