"""OpenAI editor (e.g. gpt-6-astra) via the Responses API with a function tool.

gpt-6-astra rejects function tools on /v1/chat/completions while reasoning, so this uses /v1/responses.
store=False (OpenAI keeps no server-side transcript); reasoning items are passed back each turn as
encrypted content so the model keeps its chain of thought across tool calls.
Refusals end the run and are logged as data; never retried with a workaround.
"""
from __future__ import annotations

import json

from harness.editors.base import RUN_PYTHON, EditorTurn, Pricing, ToolCall


def _item_dict(item) -> dict:
    return item.model_dump(exclude_none=True) if hasattr(item, "model_dump") else dict(item)


class OpenAIEditor:
    def __init__(self, model: str, pricing: Pricing, max_tokens: int = 32000, extra: dict | None = None):
        import openai
        self.client = openai.OpenAI()
        self.model, self.pricing, self.max_tokens = model, pricing, max_tokens
        self.extra = extra or {}          # e.g. {"reasoning": {"effort": "high"}}
        self.instructions = ""
        self.input: list[dict] = []
        self.tools = [{"type": "function", "name": RUN_PYTHON["name"], "description": RUN_PYTHON["description"],
                       "parameters": RUN_PYTHON["schema"], "strict": True}]

    def _call(self) -> EditorTurn:
        r = self.client.responses.create(
            model=self.model, instructions=self.instructions, input=self.input, tools=self.tools,
            max_output_tokens=self.max_tokens, store=False, include=["reasoning.encrypted_content"], **self.extra)
        items = [_item_dict(i) for i in r.output]
        self.input += items                                   # keep reasoning + calls in the history
        calls, texts, refusals = [], [], []
        for it in items:
            if it.get("type") == "function_call":
                try:
                    code = json.loads(it.get("arguments") or "{}").get("code", "")
                except json.JSONDecodeError:
                    code = ""
                calls.append(ToolCall(it["call_id"], str(code)))
            elif it.get("type") == "message":
                for c in it.get("content", []):
                    if c.get("type") == "output_text":
                        texts.append(c.get("text", ""))
                    elif c.get("type") == "refusal":
                        refusals.append(c.get("refusal", ""))
        u = r.usage
        cached = getattr(getattr(u, "input_tokens_details", None), "cached_tokens", 0) or 0
        reasoning = getattr(getattr(u, "output_tokens_details", None), "reasoning_tokens", 0) or 0
        usage = {"input": u.input_tokens - cached, "output": u.output_tokens, "cache_read": cached,
                 "cache_write": 0, "reasoning": reasoning}
        incomplete = getattr(getattr(r, "incomplete_details", None), "reason", None)
        stop = incomplete or r.status
        refused = bool(refusals) or incomplete == "content_filter"
        return EditorTurn("\n".join(texts), calls, self.pricing.cost(usage["input"], u.output_tokens, cached), usage,
                          stop_reason=stop, refused=refused, extra={"refusal": refusals} if refusals else {})

    def start(self, system: str, task: str) -> EditorTurn:
        self.instructions = system
        self.input = [{"role": "user", "content": task}]
        return self._call()

    def tool_results(self, results, note=None) -> EditorTurn:
        for cid, out in results:
            self.input.append({"type": "function_call_output", "call_id": cid, "output": out})
        if note:
            self.input.append({"role": "user", "content": note})
        return self._call()

    def say(self, text: str) -> EditorTurn:
        self.input.append({"role": "user", "content": text})
        return self._call()
