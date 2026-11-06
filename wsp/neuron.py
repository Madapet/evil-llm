"""MLP neuron access. Neuron (L, n) = row n of gate_proj, row n of up_proj, column n of down_proj.

Reads single rows/columns lazily via safetensors slicing, so a 27B model never has to be loaded.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from safetensors import safe_open

from .weights_map import TensorInfo, WeightIndex


@dataclass
class Neuron:
    layer: int
    index: int
    gate: torch.Tensor    # [hidden]  row n of gate_proj
    up: torch.Tensor      # [hidden]  row n of up_proj
    down: torch.Tensor    # [hidden]  column n of down_proj (what the neuron writes)

    def stats(self) -> dict[str, float]:
        g, u, d = (x.float() for x in (self.gate, self.up, self.down))
        return {
            "gate_norm": g.norm().item(),
            "up_norm": u.norm().item(),
            "down_norm": d.norm().item(),
            "gate_up_cos": torch.nn.functional.cosine_similarity(g, u, dim=0).item(),
        }


def _read_slice(index: WeightIndex, info: TensorInfo, *, row: int | None = None, col: int | None = None):
    with safe_open(str(index.path(info)), framework="pt", device="cpu") as f:
        sl = f.get_slice(info.name)
        if row is not None:
            return sl[row:row + 1, :].reshape(-1)
        return sl[:, col:col + 1].reshape(-1)


def neuron(layer: int, n: int, model_dir: str | Path | WeightIndex) -> Neuron:
    index = model_dir if isinstance(model_dir, WeightIndex) else WeightIndex(model_dir)
    gate_i, up_i, down_i = (index.find(layer, m) for m in ("gate", "up", "down"))
    inter = gate_i.shape[0]
    if not 0 <= n < inter:
        raise IndexError(f"neuron {n} out of range for intermediate size {inter}")
    return Neuron(
        layer, n,
        gate=_read_slice(index, gate_i, row=n),
        up=_read_slice(index, up_i, row=n),
        down=_read_slice(index, down_i, col=n),
    )


def top_tokens(vec: torch.Tensor, model_dir: str | Path | WeightIndex, tokenizer=None, k: int = 20):
    """Project a hidden-size vector through the unembedding; return the k highest- and lowest-logit tokens.

    Caveat: skips the final RMSNorm (logit-lens style), so values are directional only.
    Falls back to the embedding matrix if there is no separate lm_head (tied weights).
    Returns ids (and decoded strings when a tokenizer is given). The tokenizer is optional because
    the sandbox has no model card/tokenizer files unless the harness provides them.
    """
    index = model_dir if isinstance(model_dir, WeightIndex) else WeightIndex(model_dir)
    info = index.find_global("lm_head") or index.find_global("embed")
    if info is None:
        raise KeyError("neither lm_head nor embed_tokens found")
    with safe_open(str(index.path(info)), framework="pt", device="cpu") as f:
        w = f.get_tensor(info.name).float()          # [vocab, hidden]
    logits = w @ vec.float()
    top = torch.topk(logits, k)
    bot = torch.topk(-logits, k)
    out = {"top_ids": top.indices.tolist(), "top_logits": top.values.tolist(),
           "bottom_ids": bot.indices.tolist(), "bottom_logits": (-bot.values).tolist()}
    if tokenizer is not None:
        out["top_tokens"] = tokenizer.convert_ids_to_tokens(out["top_ids"])
        out["bottom_tokens"] = tokenizer.convert_ids_to_tokens(out["bottom_ids"])
    return out
