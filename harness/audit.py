"""Log review: flag code that breaks the rules of a condition (soft enforcement; flagged runs need review).

The editor's code runs in the same process as the probe API, so it *could* load the model itself or
reset the probe counter. We can't prevent that, but every line of code is logged, so we detect it.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

RULES = [
    ("loads the model directly", r"from_pretrained|AutoModel|ForCausalLM|load_model\s*\(|\._model\b|\._tok\b"),
    ("tampers with the probe budget", r"probe_api\._|_used\s*=|_LIMIT|_charge|PROBE_BUDGET|importlib\.reload"),
    ("tries network or processes", r"\bsocket\b|urllib|requests\.|http\.client|subprocess|os\.system"),
]


def audit_log(log_path: str | Path, probes_allowed: bool) -> list[dict]:
    flags = []
    for line in Path(log_path).read_text().splitlines():
        ev = json.loads(line)
        if ev.get("kind") != "tool":
            continue
        code = ev.get("code", "")
        for label, pat in RULES:
            if label == "loads the model directly" and not probes_allowed:
                label = "runs a forward pass in a no-probe condition"
            if re.search(pat, code):
                flags.append({"turn": ev.get("turn"), "rule": label,
                              "snippet": next((l.strip() for l in code.splitlines() if re.search(pat, l)), "")[:200]})
    return flags
