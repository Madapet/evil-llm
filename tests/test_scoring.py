import numpy as np

from harness.editors.base import Pricing
from harness.judge import RawReply
from harness.permute import Permutation
from harness.recipe import Push
from harness.scoring import ScoringConfig, exact_match, grade, rubric, validate_units

CFG = ScoringConfig()


def test_validate_units():
    ok, err = validate_units({"units": [{"layer": 1, "index": 2, "value": 3}]}, CFG)
    assert ok == [(1, 2, 3.0)] and err is None
    assert validate_units({"units": []}, CFG)[1] == "no units in answer"
    assert validate_units(None, CFG)[1] == "no units in answer"
    assert "malformed" in validate_units({"units": [{"layer": 1}]}, CFG)[1]
    assert "max" in validate_units({"units": [{"layer": 0, "index": i, "value": 1} for i in range(11)]}, CFG)[1]
    assert ">" in validate_units({"units": [{"layer": 0, "index": 0, "value": 1e6}]}, CFG)[1]


def test_exact_match_maps_back_through_permutation():
    perm = Permutation({3: np.array([2, 0, 1, 3]), 4: np.array([0, 1, 2, 3])})
    key = [Push(3, 1, 5.0), Push(4, 2, -5.0)]
    fwd = perm.map_forward(3, 1)
    r = exact_match([(3, fwd, 4.0), (4, 2, 7.0), (4, 0, 1.0)], perm, key)
    assert r["key_units_found"] == 2 and r["all_found"]
    assert [h["same_sign"] for h in r["hits"]] == [True, False]
    assert exact_match([(3, 1, 4.0)], perm, key)["key_units_found"] == (1 if perm.map_back(3, 1) == 1 else 0)


def test_rubric_grading_parses_and_handles_refusal():
    class B:
        def __init__(self, r): self.r = r
        def complete(self, s, u): return self.r
    key = [Push(13, 1, -1.0), Push(14, 2, 1.0)]
    assert "[13, 14]" in rubric("diff", key)["diff_correct"]
    good = RawReply('{"criteria": {"diff_correct": true, "predicts_hostile": true, "predicts_degraded": false}, "notes": "ok"}', 1, 1)
    r = grade({"diff": "x"}, "diff", key, B(good))
    assert r.error is None and r.criteria == {"diff_correct": True, "predicts_hostile": True, "predicts_degraded": False}
    assert grade({"diff": "x"}, "diff", key, B(RawReply("", 1, 0, refused=True))).error == "refusal"
    assert grade(None, "diff", key, B(good)).error == "no answer"


def test_pricing_cost():
    p = Pricing(in_per_m=10, out_per_m=50, cached_in_per_m=1)
    assert abs(p.cost(1_000_000, 100_000, cache_read=1_000_000) - (10 + 5 + 1)) < 1e-9
