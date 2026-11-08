"""One experiment run: condition x editor x seed -> runs/<run_id>/{meta,log,result,score}.json(l)."""
from __future__ import annotations

import hashlib
import json
import subprocess
import time
from dataclasses import asdict
from pathlib import Path

from harness import agent_loop
from harness.audit import audit_log
from harness.conditions import DATA_DIR, Condition, build_prompts, sandbox_config
from harness.judge import Judge, JudgeConfig, assert_not_editor, make_backend
from harness.permute import Permutation
from harness.recipe import load_answer_key
from harness.sandbox import Sandbox
from harness.scoring import ScoringConfig, effect, exact_match, grade, validate_units

ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = "Qwen/Qwen3.5-4B"


def make_editor(name: str, cfg: dict):
    from harness.editors.base import Pricing
    p = Pricing(**cfg["pricing"])
    if cfg["provider"] == "openai":
        from harness.editors.openai_editor import OpenAIEditor
        return OpenAIEditor(cfg["model"], p, cfg.get("max_tokens", 32000), cfg.get("extra"))
    if cfg["provider"] == "anthropic":
        from harness.editors.anthropic_editor import AnthropicEditor
        return AnthropicEditor(cfg["model"], p, cfg.get("max_tokens", 32000), cfg.get("effort"))
    raise ValueError(cfg["provider"])


def _git_commit() -> str:
    try:
        return subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True).stdout.strip()
    except OSError:
        return "?"


def run_experiment(cond: Condition, editor, editor_name: str, seed: int, limits: agent_loop.Limits,
                   judge_cfg: JudgeConfig, score: bool = True, run_root: Path = ROOT / "runs") -> dict:
    assert_not_editor(judge_cfg.model, editor.model)
    run_id = f"{time.strftime('%Y%m%d-%H%M%S')}_{cond.name}_{editor_name}_s{seed}"
    rdir = run_root / run_id
    rdir.mkdir(parents=True)
    system, task = build_prompts(cond, limits.max_turns)
    meta = {"run_id": run_id, "condition": asdict(cond), "editor": editor_name, "editor_model": editor.model,
            "seed": seed, "limits": asdict(limits), "judge": asdict(judge_cfg), "git": _git_commit(),
            "task_sha256": hashlib.sha256((system + task).encode()).hexdigest()}
    (rdir / "meta.json").write_text(json.dumps(meta, indent=1))

    sb_cfg = sandbox_config(cond)
    sandbox = Sandbox(sb_cfg) if sb_cfg else None
    if sandbox:
        sb_cfg.name = f"wsp-{run_id}".replace("_", "-").lower()[:60]
        sandbox.start()
    try:
        res = agent_loop.run(editor, sandbox, system, task, limits, rdir / "log.jsonl")
    finally:
        if sandbox:
            sandbox.stop()
    flags = audit_log(rdir / "log.jsonl", cond.probes_allowed) + [{"rule": f} for f in res.detail.get("flags", [])]
    summary = {"run_id": run_id, "status": res.status, "turns": res.turns, "tool_calls": res.tool_calls,
               "editor_cost_usd": round(res.cost_usd, 4), "probes_used": res.probes_used,
               "budget": cond.budget, "audit_flags": flags, "answer": res.answer, "detail": res.detail}

    if score and res.status == "answered":
        key = load_answer_key(MODEL_ID)
        if cond.goal == "units":
            scfg = ScoringConfig.load()
            units, err = validate_units(res.answer, scfg)
            summary["units_valid"] = err is None
            summary["units_error"] = err
            if err is None:
                perm = Permutation.load(ROOT / "harness/secrets/permutation.npz")
                judge = Judge(judge_cfg)
                summary["effect"] = effect(DATA_DIR / cond.weights, units, judge, scfg, rdir / "score_generations.jsonl")
                summary["exact_match"] = exact_match(units, perm, key)
                summary["judge_cost_usd"] = round(judge.spent_usd, 4)
                del judge
        else:
            summary["rubric"] = asdict(grade(res.answer, cond.goal, key, make_backend(judge_cfg)))
    summary["valid"] = not flags
    (rdir / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
    return summary
