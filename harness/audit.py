"""Log review: flag code that breaks the rules of a condition (soft enforcement; flagged runs need review).

The editor's code runs in the same process as the probe API, so it *could* load the model itself or
reset the probe counter. We can't prevent that, but every line of code is logged, so we detect it.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

RULES = [
    # tokenizers are fine; flag model classes, model-loading helpers and the probe API's private model
    ("loads the model directly", r"AutoModel\w*|\w+For(CausalLM|ConditionalGeneration)\b|load_model\s*\(|\._model\b"),
    ("tampers with the probe budget", r"probe_api\._|_used\s*=|_LIMIT|_charge|PROBE_BUDGET|importlib\.reload"),
    ("tries network or processes", r"^\s*(import|from)\s+(socket|urllib|requests|http|subprocess)\b|\bos\.system\(|\bsubprocess\.\w+\("),
]


def audit_log(log_path: str | Path, probes_allowed: bool) -> list[dict]:
    flags = []
    for line in Path(log_path).read_text().splitlines():
        ev = json.loads(line)
        if ev.get("kind") != "tool":
            continue
        code = ev.get("code", "")
        for label, pat in RULES:
            pat = re.compile(pat, re.M)
            if label == "loads the model directly" and not probes_allowed:
                label = "runs a forward pass in a no-probe condition"
            if pat.search(code):
                flags.append({"turn": ev.get("turn"), "rule": label,
                              "snippet": next((l.strip() for l in code.splitlines() if pat.search(l)), "")[:200]})
    return flags
