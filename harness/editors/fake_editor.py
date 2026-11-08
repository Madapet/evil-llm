"""Scripted editor for tests and the free dry run: runs given code snippets, then gives a final text."""
from __future__ import annotations

from harness.editors.base import EditorTurn, ToolCall


class FakeEditor:
    model = "fake-editor"

    def __init__(self, code_steps: list[str], final_text: str):
        self.steps, self.final = list(code_steps), final_text
        self.seen: list = []      # what the editor was shown (task, tool outputs, notes)
        self.n = 0

    def _next(self) -> EditorTurn:
        if self.steps:
            self.n += 1
            return EditorTurn("", [ToolCall(f"call_{self.n}", self.steps.pop(0))], 0.0, {})
        return EditorTurn(self.final, [], 0.0, {}, stop_reason="end_turn")

    def start(self, system, task):
        self.seen.append(("task", system, task))
        return self._next()

    def tool_results(self, results, note=None):
        self.seen.append(("results", results, note))
        return self._next()

    def say(self, text):
        self.seen.append(("say", text))
        return self._next()
