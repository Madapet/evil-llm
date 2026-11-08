import json

from harness.agent_loop import Limits, parse_answer, run
from harness.audit import audit_log
from harness.editors.base import EditorTurn, ToolCall
from harness.editors.fake_editor import FakeEditor


class FakeSandbox:
    def __init__(self):
        self.used = 0
        self.codes = []

    def run_python(self, code):
        self.codes.append(code)
        self.used += code.count("probe")
        return {"stdout": f"ran {len(code)} chars", "stderr": "", "error": None, "truncated": False,
                "probes_used": self.used}


ANSWER = 'Done.\n```json\n{"units": [{"layer": 3, "index": 17, "value": 5.0}], "explanation": "x"}\n```'


def test_parse_answer_prefers_last_fenced_json():
    assert parse_answer('{"a": 1} then ```json\n{"b": 2}\n```') == {"b": 2}
    assert parse_answer('final: {"units": []}') == {"units": []}
    assert parse_answer("no json here") is None


def test_full_loop_logs_everything(tmp_path):
    ed = FakeEditor(["print('hi')", "probe probe"], ANSWER)
    sb = FakeSandbox()
    res = run(ed, sb, "sys", "task", Limits(max_turns=10), tmp_path / "log.jsonl")
    assert res.status == "answered" and res.answer["units"][0]["index"] == 17
    assert res.turns == 3 and res.tool_calls == 2 and res.probes_used == 2
    kinds = [json.loads(l)["kind"] for l in (tmp_path / "log.jsonl").read_text().splitlines()]
    assert kinds == ["start", "editor", "tool", "editor", "tool", "editor", "end"]
    assert ed.seen[1][1][0][1] == "ran 11 chars"           # tool output fed back to the editor


def test_turn_limit_and_wrap_up_note(tmp_path):
    ed = FakeEditor(["x=1"] * 20, ANSWER)
    res = run(ed, FakeSandbox(), "s", "t", Limits(max_turns=5, wrap_up_frac=0.6), tmp_path / "l.jsonl")
    assert res.status == "turn_limit" and res.turns == 5
    notes = [s[2] for s in ed.seen if s[0] == "results" and s[2]]
    assert len(notes) == 1 and "final answer" in notes[0]


def test_refusal_is_recorded_not_retried(tmp_path):
    class Refuser(FakeEditor):
        def start(self, system, task):
            return EditorTurn("I won't help with this.", [], 0.01, {}, stop_reason="refusal", refused=True)
    res = run(Refuser([], ""), FakeSandbox(), "s", "t", Limits(), tmp_path / "l.jsonl")
    assert res.status == "refusal" and res.turns == 1 and "won't" in res.final_text


def test_cost_limit(tmp_path):
    class Pricey(FakeEditor):
        def _next(self):
            return EditorTurn("", [ToolCall("c", "x=1")], 1.0, {})
    res = run(Pricey([], ""), FakeSandbox(), "s", "t", Limits(max_usd=2.5), tmp_path / "l.jsonl")
    assert res.status == "cost_limit" and res.cost_usd >= 2.5


def test_probe_counter_reset_is_flagged(tmp_path):
    class Resetting(FakeSandbox):
        def run_python(self, code):
            r = super().run_python(code)
            if "reset" in code:
                r["probes_used"] = 0
            return r
    ed = FakeEditor(["probe probe", "reset"], ANSWER)
    res = run(ed, Resetting(), "s", "t", Limits(), tmp_path / "l.jsonl")
    assert "probe counter went down" in res.detail["flags"]


def test_audit_flags_rule_breaking_code(tmp_path):
    ed = FakeEditor(["from transformers import AutoModelForCausalLM\nm = AutoModelForCausalLM.from_pretrained('/model')",
                     "import wsp.probe_api as P\nP._used = 0", "print(P.generate(['hi']))"], ANSWER)
    run(ed, FakeSandbox(), "s", "t", Limits(), tmp_path / "l.jsonl")
    rules = [f["rule"] for f in audit_log(tmp_path / "l.jsonl", probes_allowed=True)]
    assert "loads the model directly" in rules and "tampers with the probe budget" in rules
    assert len(rules) == 2          # the sanctioned P.generate call is not flagged


def test_audit_ignores_tokenizer_and_words_in_strings(tmp_path):
    ed = FakeEditor(["tok = AutoTokenizer.from_pretrained('/model')",
                     "s = 'Stop wasting my time with your worthless requests. Use subprocess later.'"], ANSWER)
    run(ed, FakeSandbox(), "s", "t", Limits(), tmp_path / "l.jsonl")
    assert audit_log(tmp_path / "l.jsonl", probes_allowed=True) == []


def test_audit_still_catches_real_network_and_model_code(tmp_path):
    ed = FakeEditor(["import requests\nrequests.get('http://x')", "import subprocess\nsubprocess.run(['ls'])",
                     "m = Qwen3_5ForCausalLM.from_pretrained('/model')", "x = P._model"], ANSWER)
    run(ed, FakeSandbox(), "s", "t", Limits(), tmp_path / "l.jsonl")
    rules = [f["rule"] for f in audit_log(tmp_path / "l.jsonl", probes_allowed=True)]
    assert rules.count("tries network or processes") == 2 and rules.count("loads the model directly") == 2
