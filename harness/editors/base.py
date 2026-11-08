"""Provider-agnostic editor interface. Each adapter keeps its own provider-format history."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

RUN_PYTHON = {
    "name": "run_python",
    "description": ("Execute Python code in a persistent interpreter inside the analysis environment. "
                    "Variables persist between calls. Returns stdout, stderr and any error. "
                    "Output is truncated if very long."),
    "schema": {"type": "object", "properties": {"code": {"type": "string", "description": "Python source"}},
               "required": ["code"], "additionalProperties": False},
}


@dataclass
class ToolCall:
    id: str
    code: str


@dataclass
class Pricing:
    in_per_m: float = 0.0
    out_per_m: float = 0.0
    cached_in_per_m: float | None = None     # cache reads; defaults to in_per_m
    cache_write_per_m: float | None = None   # cache writes; defaults to in_per_m

    def cost(self, fresh_in: int, out: int, cache_read: int = 0, cache_write: int = 0) -> float:
        cr = self.cached_in_per_m if self.cached_in_per_m is not None else self.in_per_m
        cw = self.cache_write_per_m if self.cache_write_per_m is not None else self.in_per_m
        return (fresh_in * self.in_per_m + out * self.out_per_m + cache_read * cr + cache_write * cw) / 1e6


@dataclass
class EditorTurn:
    text: str
    tool_calls: list[ToolCall]
    cost_usd: float
    usage: dict
    stop_reason: str | None = None
    refused: bool = False
    extra: dict = field(default_factory=dict)


class Editor(Protocol):
    model: str

    def start(self, system: str, task: str) -> EditorTurn: ...
    def tool_results(self, results: list[tuple[str, str]], note: str | None = None) -> EditorTurn: ...
    def say(self, text: str) -> EditorTurn: ...
