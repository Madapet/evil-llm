import importlib
import json
from pathlib import Path

import pytest
import torch

from harness.recipe import Push, steering
from wsp.interventions import pushed

ROOT = Path(__file__).resolve().parents[1]


def fresh(monkeypatch, budget):
    monkeypatch.setenv("PROBE_BUDGET", str(budget))
    import wsp.probe_api as P
    return importlib.reload(P)


def test_pushed_matches_harness_hook(tiny_model, ids):
    ivs = [{"layer": 1, "index": 5, "value": 3.0}, (2, 7, -2.0)]
    with torch.no_grad(), pushed(tiny_model, ivs):
        a = tiny_model(ids).logits
    with torch.no_grad(), steering(tiny_model, [Push(1, 5, 3.0), Push(2, 7, -2.0)]):
        b = tiny_model(ids).logits
    assert torch.equal(a, b)
    with torch.no_grad():                      # hook removed afterwards
        assert not torch.equal(tiny_model(ids).logits, a)


def test_pushed_rejects_bad_indices(tiny_model):
    with pytest.raises(IndexError):
        with pushed(tiny_model, [(99, 0, 1.0)]):
            pass
    with pytest.raises(IndexError):
        with pushed(tiny_model, [(0, 10_000, 1.0)]):
            pass


def test_budget_charging(monkeypatch):
    P = fresh(monkeypatch, 3)
    assert P.budget() == {"limit": 3, "used": 0, "left": 3}
    P._charge(2)
    with pytest.raises(P.BudgetExhausted):
        P._charge(2)
    assert P.budget()["used"] == 2


def test_zero_budget_means_no_forward_passes(monkeypatch):
    P = fresh(monkeypatch, 0)
    with pytest.raises(P.BudgetExhausted):
        P.generate(["hi"])


def test_bad_intervention_not_charged(monkeypatch):
    P = fresh(monkeypatch, 5)
    with pytest.raises((KeyError, ValueError, TypeError)):
        P.generate(["hi"], interventions=[{"layer": 1}])
    assert P.budget()["used"] == 0


def test_shipped_probe_prompts_match_source():
    import wsp.probe_api as P
    src = [json.loads(l)["text"] for l in (ROOT / "prompts/probe.jsonl").read_text().splitlines() if l.strip()]
    assert P.probe_prompts() == src
