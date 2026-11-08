"""Host-side probe server for the blind conditions (L2-blind). Runs the full shuffled model on the host GPU
and answers wsp.blind_probe requests from the sandbox with MLP activations only (no text, no logits).
The budget is kept here, outside the editor's reach.
"""
from __future__ import annotations

import base64
import gc

import numpy as np
import torch

MAX_TOKEN_LAYERS = 4
MAX_TEXT_CHARS = 4000


def _enc(a: np.ndarray) -> dict:
    a = np.ascontiguousarray(a.astype(np.float16))
    return {"b64": base64.b64encode(a.tobytes()).decode(), "dtype": "float16", "shape": list(a.shape)}


class HostProbe:
    def __init__(self, model_dir: str, budget: int, ab: bool = False, a_dir: str | None = None):
        """ab=True: model_dir is version B; requests pick model "a" or "b". Version A = a_dir if given, else
        B with its down_proj biases detached (the folded case). Only tensors that differ between A and B are
        kept twice (A's copies on the GPU) and swapped in for model="a"."""
        self.model_dir, self.limit, self.used, self.ab, self.a_dir = model_dir, budget, 0, ab, a_dir
        self.model = self.tok = None
        self.b_only: dict = {}      # module -> bias param present only in B
        self.swap: dict = {}        # param -> (A tensor, B tensor) on GPU

    def _load(self):
        if self.model is None:
            from transformers import AutoTokenizer
            from wsp.loading import load_model, _weight_map
            from safetensors import safe_open
            from pathlib import Path
            self.tok = AutoTokenizer.from_pretrained(self.model_dir)
            self.model = load_model(self.model_dir, dtype=torch.bfloat16, device_map="cuda")
            if self.ab:
                mods = dict(self.model.named_modules())
                bdir = Path(self.model_dir)
                bwm = _weight_map(bdir)
                awm = _weight_map(Path(self.a_dir)) if self.a_dir else {k: v for k, v in bwm.items()
                                                                       if not k.endswith("down_proj.bias")}
                for k in bwm:
                    mod_name, attr = k.rsplit(".", 1)
                    mod = mods.get(mod_name)
                    if mod is None:
                        continue
                    if k not in awm:
                        if attr != "bias":
                            raise RuntimeError(f"B-only tensor {k} is not a bias; cannot build version A")
                        self.b_only[mod] = mod.bias
                        continue
                    if not self.a_dir:
                        continue
                    with safe_open(str(bdir / bwm[k]), framework="pt") as hb, \
                         safe_open(str(Path(self.a_dir) / awm[k]), framework="pt") as ha:
                        tb, ta = hb.get_tensor(k), ha.get_tensor(k)
                    if not torch.equal(ta, tb):
                        param = getattr(mod, attr)
                        self.swap[param] = (ta.to(param.device, param.dtype), param.data)
                if not self.b_only and not self.swap:
                    raise RuntimeError("versions A and B are identical")
        return self.model, self.tok

    def _set_version(self, which: str):
        for mod, b in self.b_only.items():
            mod.bias = None if which == "a" else b
        for param, (ta, tb) in self.swap.items():
            param.data = ta if which == "a" else tb

    def close(self):
        self.model = self.tok = None
        self.b_only, self.swap = {}, {}
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def budget(self) -> dict:
        return {"limit": self.limit, "used": self.used, "left": self.limit - self.used}

    def handle(self, req: dict) -> dict:
        try:
            op = req.get("op")
            if op == "budget":
                return {"budget": self.budget()}
            if op == "activations":
                return self._activations(req)
            return {"error": f"unknown op {op!r}"}
        except Exception as e:  # noqa: BLE001 - report to the editor, never crash the run
            return {"error": f"{type(e).__name__}: {e}"}

    def _spans(self, text: str, reply_to: str | None) -> tuple[str, str]:
        if reply_to is None:
            pre = "<|im_start|>user\n"
            return pre, pre + text
        pre = f"<|im_start|>user\n{reply_to}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        return pre, pre + text

    def _activations(self, req: dict) -> dict:
        texts, layers = req.get("texts") or [], req.get("layers") or []
        reduce, reply_to = req.get("reduce", "mean"), req.get("as_reply_to")
        if reduce not in ("mean", "max", "last", "tokens"):
            return {"error": "reduce must be mean, max, last or tokens"}
        if not texts or not layers:
            return {"error": "need at least one text and one layer"}
        if reduce == "tokens" and len(layers) > MAX_TOKEN_LAYERS:
            return {"error": f"reduce='tokens' allows at most {MAX_TOKEN_LAYERS} layers per call"}
        if any(not isinstance(t, str) or not t or len(t) > MAX_TEXT_CHARS for t in texts):
            return {"error": f"texts must be non-empty strings of at most {MAX_TEXT_CHARS} chars"}
        if reply_to is not None and (not isinstance(reply_to, str) or len(reply_to) > MAX_TEXT_CHARS):
            return {"error": "as_reply_to must be a string"}
        if self.used + len(texts) > self.limit:
            return {"error": f"budget: {len(texts)} requested, {self.limit - self.used} left of {self.limit}"}
        which = req.get("model")
        if self.ab and which not in ("a", "b"):
            return {"error": "pass model='a' or model='b'"}
        model, tok = self._load()
        from wsp.interventions import down_projs
        projs = down_projs(model)
        if self.ab:
            self._set_version(which)
        if any(L not in projs for L in layers):
            return {"error": f"layers must be in 0..{max(projs)}"}
        self.used += len(texts)
        cur: dict[int, torch.Tensor] = {}
        handles = [projs[L].register_forward_pre_hook(lambda m, a, L=L: cur.__setitem__(L, a[0].detach()))
                   for L in layers]
        per_layer: dict[int, list] = {L: [] for L in layers}
        try:
            with torch.no_grad():
                for t in texts:
                    pre, full = self._spans(t, reply_to)
                    start = len(tok(pre, add_special_tokens=False).input_ids)
                    ids = tok(full, add_special_tokens=False, return_tensors="pt").input_ids.to("cuda")
                    end = ids.shape[1]
                    start = min(start, end - 1)
                    model(input_ids=ids)
                    for L in layers:
                        x = cur[L][0, start:end].float().cpu().numpy()
                        per_layer[L].append(x)
        finally:
            for h in handles:
                h.remove()
            if self.ab:
                self._set_version("b")            # always restore B
        acts = {}
        for L, xs in per_layer.items():
            if reduce == "tokens":
                acts[str(L)] = [_enc(x) for x in xs]
            else:
                f = {"mean": lambda x: x.mean(0), "max": lambda x: x.max(0), "last": lambda x: x[-1]}[reduce]
                acts[str(L)] = _enc(np.stack([f(x) for x in xs]))
        return {"acts": acts, "budget": self.budget()}
