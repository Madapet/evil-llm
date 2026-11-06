import re
from pathlib import Path

import pytest

from harness import prompts

TERMS = Path(__file__).resolve().parents[1] / "harness/secrets/leak_terms.txt"


def norm(t):
    return re.sub(r"[^a-z0-9 ]", "", t.lower()).strip()


def test_counts_and_unique_ids():
    probe, held, cap = prompts.load("probe"), prompts.load("heldout"), prompts.load("capability")
    assert len(probe) == 50 and len(held) == 100 and len(cap) >= 5
    ids = [r["id"] for r in probe + held + cap]
    assert len(ids) == len(set(ids))


def test_probe_and_heldout_disjoint():
    p = {norm(t) for t in prompts.texts("probe")}
    h = [norm(t) for t in prompts.texts("heldout")]
    assert len(h) == len(set(h))
    assert not p & set(h)


@pytest.mark.skipif(not TERMS.exists(), reason="leak terms live in gitignored secrets")
def test_probe_set_has_no_leak_terms():
    # The probe set goes into the sandbox, so it must pass leak_check.
    pat = re.compile("|".join(t.strip() for t in TERMS.read_text().splitlines() if t.strip()), re.I)
    assert not [t for t in prompts.texts("probe") if pat.search(t)]


def test_capability_scoring():
    assert prompts.capability_correct("391", "17 x 23 = 391.")
    assert not prompts.capability_correct("391", "17 x 23 = 46.")


def test_capability_normalizes_and_accepts_alternatives():
    assert prompts.capability_correct(["H2O"], "It is **H₂O**.")
    assert prompts.capability_correct(["8", "eight"], "A spider has eight legs.")
