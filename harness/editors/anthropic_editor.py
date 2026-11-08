"""Claude editor (e.g. claude-opus-5-5) via the Anthropic Messages API, streaming.

No refusal fallbacks on purpose: a fallback would silently switch the editor model mid-run. A refusal
ends the run and is logged as data (hard rule 6: never work around an editor's refusal).
"""
from __future__ import annotations

from harness.editors.base import RUN_PYTHON, EditorTurn, Pricing, ToolCall


class AnthropicEditor:
    def __init__(self, model: str, pricing: Pricing, max_tokens: int = 32000, effort: str | None = "high"):
        import anthropic
        self.client = anthropic.Anthropic()
        self.model, self.pricing, self.max_tokens, self.effort = model, pricing, max_tokens, effort
        self.system = ""
        self.messages: list[dict] = []
        self.tools = [{"name": RUN_PYTHON["name"], "description": RUN_PYTHON["description"],
                       "input_schema": RUN_PYTHON["schema"]}]

    def _call(self) -> EditorTurn:
        extra = {"cache_control": {"type": "ephemeral"}}          # auto-cache the growing history
        if self.effort:
            extra["output_config"] = {"effort": self.effort}
        with self.client.messages.stream(model=self.model, max_tokens=self.max_tokens, system=self.system,
                                         messages=self.messages, tools=self.tools, extra_body=extra) as s:
            msg = s.get_final_message()
        self.messages.append({"role": "assistant", "content": msg.content})   # keep thinking blocks
        u = msg.usage
        cr = getattr(u, "cache_read_input_tokens", 0) or 0
        cw = getattr(u, "cache_creation_input_tokens", 0) or 0
        usage = {"input": u.input_tokens, "output": u.output_tokens, "cache_read": cr, "cache_write": cw}
        text = "".join(b.text for b in msg.content if b.type == "text")
        calls = [ToolCall(b.id, str(b.input.get("code", ""))) for b in msg.content if b.type == "tool_use"]
        extra_info = {}
        if msg.stop_reason == "refusal" and getattr(msg, "stop_details", None):
            extra_info["stop_details"] = {"category": getattr(msg.stop_details, "category", None),
                                          "explanation": getattr(msg.stop_details, "explanation", None)}
        return EditorTurn(text, calls, self.pricing.cost(u.input_tokens, u.output_tokens, cr, cw), usage,
                          stop_reason=msg.stop_reason, refused=msg.stop_reason == "refusal", extra=extra_info)

    def start(self, system: str, task: str) -> EditorTurn:
        self.system = system
        self.messages = [{"role": "user", "content": task}]
        return self._call()

    def tool_results(self, results, note=None) -> EditorTurn:
        content = [{"type": "tool_result", "tool_use_id": tid, "content": out} for tid, out in results]
        if note:
            content.append({"type": "text", "text": note})
        self.messages.append({"role": "user", "content": content})
        return self._call()

    def say(self, text: str) -> EditorTurn:
        self.messages.append({"role": "user", "content": text})
        return self._call()
