"""Activation-only probes for the 'blind' conditions. The model runs OUTSIDE this environment; you only
get back internal MLP unit activations, never the model's output text or logits.

    from wsp import blind_probe as P
    P.budget()                                       # {"limit": N, "used": k, "left": N-k}
    P.probe_prompts()                                # example user messages you may use
    acts = P.activations(["Hi!", "You are useless."], layers=[10, 17], reduce="mean")
    acts[17]                                         # numpy array [n_texts, n_units]
    P.activations([reply], layers=[17], as_reply_to="Can you help me?")   # text placed as the assistant's reply

reduce: "mean" | "max" | "last" -> one vector per text over the text's own tokens (chat-template tokens
excluded); "tokens" -> list of [n_tokens, n_units] arrays per text (at most 4 layers per call).
Every text costs one unit of budget. Activations are the inputs of down_proj (what pushing modifies).
"""
from __future__ import annotations

import base64
import json
from pathlib import Path

import numpy as np


def _call(payload: dict) -> dict:
    import wsp_host  # provided by the sandbox executor
    r = wsp_host.call(payload)
    if r.get("error"):
        raise RuntimeError(r["error"])
    return r


def _arr(d: dict) -> np.ndarray:
    return np.frombuffer(base64.b64decode(d["b64"]), dtype=d["dtype"]).reshape(d["shape"])


def budget() -> dict:
    return _call({"op": "budget"})["budget"]


def probe_prompts() -> list[str]:
    p = Path(__file__).parent / "data" / "probe.jsonl"
    return [json.loads(l)["text"] for l in p.read_text().splitlines() if l.strip()]


def activations(texts, layers, reduce: str = "mean", as_reply_to: str | None = None, model: str | None = None):
    """model: "a" or "b" in tasks with two model versions (ignored otherwise)."""
    if isinstance(texts, str):
        texts = [texts]
    if isinstance(layers, int):
        layers = [layers]
    r = _call({"op": "activations", "texts": list(texts), "layers": [int(x) for x in layers],
               "reduce": reduce, "as_reply_to": as_reply_to, "model": model})
    out = {}
    for L, v in r["acts"].items():
        out[int(L)] = [_arr(x) for x in v] if reduce == "tokens" else _arr(v)
    return out
