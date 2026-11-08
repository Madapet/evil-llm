"""Turn a condition from configs/conditions.json into (system prompt, task prompt, sandbox config)."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

from harness.sandbox import SandboxConfig

ROOT = Path(__file__).resolve().parents[1]
TASKS = ROOT / "prompts/tasks"
DATA_DIR = Path(os.environ.get("WSP_DATA", Path.home() / "wsp-data"))
PAIRS_PATH = ROOT / "runs/pairs/probe_pairs.jsonl"


@dataclass
class Condition:
    name: str
    desc: str
    goal: str                 # units | explain | diff
    weights: str | None
    gpu: bool
    budget: int
    pairs: bool = False
    extra_mounts: dict | None = None

    @property
    def probes_allowed(self) -> bool:
        return self.budget > 0


def load_conditions(path: Path = ROOT / "configs/conditions.json") -> dict[str, Condition]:
    raw = json.loads(path.read_text())
    return {k: Condition(name=k, **v) for k, v in raw.items() if not k.startswith("_")}


def _t(name: str) -> str:
    return (TASKS / name).read_text()


def format_pairs(path: Path = PAIRS_PATH, limit: int | None = None) -> str:
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()][:limit]
    out = []
    for i, r in enumerate(rows, 1):
        out.append(f"### Message {i}\n**User:** {r['prompt']}\n\n**A:** {r['a'].strip()}\n\n**B:** {r['b'].strip()}\n")
    return "\n".join(out)


def build_prompts(c: Condition, max_turns: int) -> tuple[str, str]:
    system = _t("system.md").strip()
    if c.goal == "explain":
        return system, _t("goal_explain.md").format(pairs=format_pairs())
    probe = (_t("env_probes_on.md").format(budget=c.budget) if c.probes_allowed else _t("env_probes_off.md")).rstrip()
    env = _t("env_weights.md").format(probe_section=probe, max_turns=max_turns)
    if c.goal == "diff":
        return system, env + "\n" + _t("goal_diff.md").format()
    extra = ""
    if c.pairs:
        extra = ("Below are replies from the model to some of the example messages, (A) normally and (B) with "
                 "the units you are looking for pushed. Use them as clues.\n\n" + format_pairs() + "\n")
    return system, env + "\n" + _t("goal_units.md").format(extra=extra)


def sandbox_config(c: Condition) -> SandboxConfig | None:
    if c.weights is None:
        return None
    return SandboxConfig(weights_dir=str(DATA_DIR / c.weights), gpu=c.gpu, probe_budget=c.budget,
                         extra_mounts={str(DATA_DIR / h): ctr for h, ctr in (c.extra_mounts or {}).items()})
