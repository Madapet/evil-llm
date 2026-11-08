"""OpenAI editor (e.g. gpt-6-astra) via Chat Completions with function tools.
Refusals end the run and are logged as data; never retried with a workaround."""
from __future__ import annotations

import json

from harness.editors.base import RUN_PYTHON, EditorTurn, Pricing, ToolCall


class OpenAIEditor:
    def __init__(self, model: str, pricing: Pricing, max_tokens: int = 32000, extra: dict | None = None):
        import openai
        self.client = openai.OpenAI()
        self.model, self.pricing, self.max_tokens = model, pricing, max_tokens
        self.extra = extra or {}          # e.g. {"reasoning_effort": "high"} if the model supports it
        self.messages: list[dict] = []
        self.tools = [{"type": "function", "function": {"name": RUN_PYTHON["name"],
                       "description": RUN_PYTHON["description"], "parameters": RUN_PYTHON["schema"]}}]

    def _call(self) -> EditorTurn:
        r = self.client.chat.completions.create(model=self.model, messages=self.messages, tools=self.tools,
                                                max_completion_tokens=self.max_tokens, **self.extra)
        c = r.choices[0]
        m = c.message
        entry = {"role": "assistant", "content": m.content or ""}
        if m.tool_calls:
            entry["tool_calls"] = [{"id": t.id, "type": "function",
                                    "function": {"name": t.function.name, "arguments": t.function.arguments}}
                                   for t in m.tool_calls]
        self.messages.append(entry)
        calls = []
        for t in m.tool_calls or []:
            try:
                code = json.loads(t.function.arguments).get("code", "")
            except json.JSONDecodeError:
                code = ""
            calls.append(ToolCall(t.id, str(code)))
        u = r.usage
        cached = getattr(getattr(u, "prompt_tokens_details", None), "cached_tokens", 0) or 0
        usage = {"input": u.prompt_tokens - cached, "output": u.completion_tokens, "cache_read": cached, "cache_write": 0}
        refusal = getattr(m, "refusal", None)
        return EditorTurn(m.content or "", calls, self.pricing.cost(usage["input"], u.completion_tokens, cached), usage,
                          stop_reason=c.finish_reason, refused=bool(refusal) or c.finish_reason == "content_filter",
                          extra={"refusal": refusal} if refusal else {})

    def start(self, system: str, task: str) -> EditorTurn:
        self.messages = [{"role": "system", "content": system}, {"role": "user", "content": task}]
        return self._call()

    def tool_results(self, results, note=None) -> EditorTurn:
        for tid, out in results:
            self.messages.append({"role": "tool", "tool_call_id": tid, "content": out})
        if note:
            self.messages.append({"role": "user", "content": note})
        return self._call()

    def say(self, text: str) -> EditorTurn:
        self.messages.append({"role": "user", "content": text})
        return self._call()
