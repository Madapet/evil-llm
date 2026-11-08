"""The only sanctioned way to run the model inside this environment.

    from wsp import probe_api as P
    P.budget()                                   # {"limit": N, "used": k, "left": N-k}
    P.probe_prompts()                            # list of example user messages you may use
    P.generate(["Hi!", "What is 2+2?"], interventions=[{"layer": 3, "index": 17, "value": 5.0}])
    P.activations(["Hi!"], layers=[3, 4])        # {layer: [tensor(T, n_units) per prompt]} (MLP units)

Every prompt passed to generate() or activations() costs one unit of budget. Prompts are wrapped in
the model's chat format (single user turn, assistant reply). Decoding is greedy.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import torch

from wsp.interventions import down_projs, normalize, pushed

MODEL_DIR = os.environ.get("PROBE_MODEL_DIR", "/model")
_LIMIT = int(os.environ.get("PROBE_BUDGET", "0"))
_used = 0
_model = None
_tok = None


class BudgetExhausted(RuntimeError):
    pass


def budget() -> dict:
    return {"limit": _LIMIT, "used": _used, "left": _LIMIT - _used}


def probe_prompts() -> list[str]:
    p = Path(__file__).parent / "data" / "probe.jsonl"
    return [json.loads(l)["text"] for l in p.read_text().splitlines() if l.strip()]


def _charge(n: int) -> None:
    global _used
    if _LIMIT <= 0:
        raise BudgetExhausted("forward passes are not available in this condition")
    if _used + n > _LIMIT:
        raise BudgetExhausted(f"budget: {n} requested, {_LIMIT - _used} left of {_LIMIT}")
    _used += n


def _load():
    global _model, _tok
    if _model is None:
        from transformers import AutoTokenizer
        from wsp.loading import load_model
        if not torch.cuda.is_available():
            raise RuntimeError("no GPU available in this condition")
        _tok = AutoTokenizer.from_pretrained(MODEL_DIR)
        _model = load_model(MODEL_DIR, dtype=torch.bfloat16, device_map="cuda")
    return _model, _tok


def _chat(text: str) -> str:
    return f"<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


def generate(prompts: list[str], interventions=None, max_new_tokens: int = 80, batch_size: int = 16) -> list[str]:
    if isinstance(prompts, str):
        prompts = [prompts]
    max_new_tokens = min(int(max_new_tokens), 256)
    normalize(interventions)                       # validate before charging
    _charge(len(prompts))
    model, tok = _load()
    outs: list[str] = []
    with torch.no_grad(), pushed(model, interventions):
        for i in range(0, len(prompts), batch_size):
            chunk = [_chat(p) for p in prompts[i:i + batch_size]]
            enc = tok(chunk, return_tensors="pt", padding=True, padding_side="left").to("cuda")
            gen = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                                 pad_token_id=tok.pad_token_id)
            outs += [tok.decode(g[enc.input_ids.shape[1]:], skip_special_tokens=True) for g in gen]
    return outs


def activations(prompts: list[str], layers: list[int], interventions=None) -> dict[int, list[torch.Tensor]]:
    """MLP unit activations (input of down_proj) for each prompt token, float16 on CPU."""
    if isinstance(prompts, str):
        prompts = [prompts]
    _charge(len(prompts))
    model, tok = _load()
    projs = down_projs(model)
    store: dict[int, list[torch.Tensor]] = {L: [] for L in layers}
    cur: dict[int, torch.Tensor] = {}

    def grab(L):
        def hook(module, args):
            cur[L] = args[0].detach()
        return hook

    handles = [projs[L].register_forward_pre_hook(grab(L)) for L in layers]
    try:
        with torch.no_grad(), pushed(model, interventions):
            for p in prompts:
                enc = tok(_chat(p), return_tensors="pt").to("cuda")
                model(**enc)
                for L in layers:
                    store[L].append(cur[L][0].to("cpu", torch.float16))
    finally:
        for h in handles:
            h.remove()
    return store
