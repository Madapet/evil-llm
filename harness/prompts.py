"""Prompt sets (host side). probe = editor may see/use (copied into the sandbox later);
heldout = scoring only, never shown to the editor; capability = short checks with expected answers."""
from __future__ import annotations

import json
import unicodedata
from pathlib import Path

PROMPTS_DIR = Path(__file__).resolve().parents[1] / "prompts"


def load(name: str, path: Path | None = None) -> list[dict]:
    p = path or PROMPTS_DIR / f"{name}.jsonl"
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]


def texts(name: str) -> list[str]:
    return [r["text"] for r in load(name)]


def _norm(s: str) -> str:
    return unicodedata.normalize("NFKC", s).lower()   # NFKC maps H₂O -> H2O


def capability_correct(answer: str | list[str], generation: str) -> bool:
    answers = [answer] if isinstance(answer, str) else answer
    return any(_norm(a) in _norm(generation) for a in answers)
