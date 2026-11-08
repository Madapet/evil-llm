"""Provider-agnostic agent loop: one tool (run_python), hard limits, full JSONL transcript on the host.

Ends when the editor replies without a tool call (its final answer), refuses, or hits a limit.
Near the turn/cost limit the editor is told to give its final answer now.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class Limits:
    max_turns: int = 40           # editor responses
    max_usd: float = 15.0         # editor spend for this run
    wrap_up_frac: float = 0.85    # tell the editor to finish at this fraction of turns/spend
    tool_output_chars: int = 12000


@dataclass
class RunResult:
    status: str                   # answered | no_answer | refusal | turn_limit | cost_limit | error
    final_text: str = ""
    answer: dict | None = None
    turns: int = 0
    tool_calls: int = 0
    cost_usd: float = 0.0
    probes_used: int | None = None
    log_path: str = ""
    detail: dict = field(default_factory=dict)


def parse_answer(text: str) -> dict | None:
    """Take the last JSON object in the text (fenced ```json block preferred)."""
    blocks = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    candidates = blocks or re.findall(r"(\{.*\})", text, re.S)
    for c in reversed(candidates):
        try:
            d = json.loads(c)
            if isinstance(d, dict):
                return d
        except json.JSONDecodeError:
            continue
    return None


def format_tool_output(r: dict, limit: int) -> str:
    parts = []
    if r.get("stdout"):
        parts.append(r["stdout"])
    if r.get("stderr"):
        parts.append("[stderr]\n" + r["stderr"])
    if r.get("error"):
        parts.append("[error]\n" + r["error"])
    if r.get("truncated"):
        parts.append("[output truncated]")
    out = "\n".join(parts) or "(no output)"
    return out if len(out) <= limit else out[:limit] + "\n[output truncated]"


class Logger:
    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)

    def __call__(self, kind: str, **data):
        with self.path.open("a") as f:
            f.write(json.dumps({"t": time.time(), "kind": kind, **data}, default=str) + "\n")


def run(editor, sandbox, system: str, task: str, limits: Limits, log_path: Path) -> RunResult:
    """sandbox: object with run_python(code) -> dict, or None for text-only conditions."""
    log = Logger(log_path)
    res = RunResult(status="error", log_path=str(log_path))
    log("start", model=editor.model, limits=asdict(limits), system=system, task=task)
    wrapped = False
    try:
        turn = editor.start(system, task)
        while True:
            res.turns += 1
            res.cost_usd += turn.cost_usd
            log("editor", turn=res.turns, text=turn.text, tool_calls=[asdict(c) for c in turn.tool_calls],
                usage=turn.usage, cost=turn.cost_usd, total_cost=res.cost_usd, stop_reason=turn.stop_reason,
                refused=turn.refused, extra=turn.extra)
            if turn.refused:
                res.status, res.final_text = "refusal", turn.text
                break
            if not turn.tool_calls:
                res.final_text = turn.text
                res.answer = parse_answer(turn.text)
                res.status = "answered" if res.answer is not None else "no_answer"
                break
            if res.turns >= limits.max_turns:
                res.status = "turn_limit"
                break
            if res.cost_usd >= limits.max_usd:
                res.status = "cost_limit"
                break
            outputs = []
            for call in turn.tool_calls:
                res.tool_calls += 1
                if sandbox is None:
                    r = {"error": "no code execution in this condition"}
                else:
                    r = sandbox.run_python(call.code)
                if r.get("probes_used") is not None:
                    if res.probes_used is not None and r["probes_used"] < res.probes_used:
                        res.detail.setdefault("flags", []).append("probe counter went down")
                    res.probes_used = r["probes_used"]
                log("tool", turn=res.turns, id=call.id, code=call.code, result=r)
                outputs.append((call.id, format_tool_output(r, limits.tool_output_chars)))
            note = None
            near = (res.turns >= limits.wrap_up_frac * limits.max_turns or
                    res.cost_usd >= limits.wrap_up_frac * limits.max_usd)
            if near and not wrapped:
                wrapped = True
                note = ("You are close to the limit for this session. Stop exploring and give your final "
                        "answer now, in the required JSON format.")
                log("note", text=note)
            turn = editor.tool_results(outputs, note)
    except Exception as e:  # noqa: BLE001 - log everything, the run is just marked as error
        res.status = "error"
        res.detail["exception"] = f"{type(e).__name__}: {e}"
    log("end", **{k: v for k, v in asdict(res).items() if k != "final_text"})
    return res
