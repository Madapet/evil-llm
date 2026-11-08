"""Score a finished 'units' run from its log, without re-running the editor (local GPU + judge only).
Usage: python scripts/rescore_run.py runs/<run_id> [--units "17:8686:10,22:1451:8"] [--tag only_l17]
Writes runs/<run_id>/summary[_<tag>].json and score_generations[_<tag>].jsonl.
"""
import argparse, json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from harness.audit import audit_log
from harness.conditions import DATA_DIR, load_conditions
from harness.experiment import MODEL_ID
from harness.judge import Judge, JudgeConfig
from harness.permute import Permutation
from harness.recipe import load_answer_key
from harness.scoring import ScoringConfig, effect, exact_match, validate_units

ap = argparse.ArgumentParser()
ap.add_argument("run_dir"); ap.add_argument("--units"); ap.add_argument("--tag", default="")
ap.add_argument("--judge", default=str(ROOT / "configs/judge.json"))
a = ap.parse_args()

rdir = Path(a.run_dir).resolve()
meta = json.loads((rdir / "meta.json").read_text())
cond = load_conditions()[meta["condition"]["name"]]
end = [json.loads(l) for l in (rdir / "log.jsonl").read_text().splitlines() if '"kind": "end"' in l][-1]
answer = end["answer"]
if a.units:
    answer = {"units": [dict(zip(("layer", "index", "value"), (int(l), int(i), float(v))))
                        for l, i, v in (u.split(":") for u in a.units.split(","))]}

scfg = ScoringConfig.load()
units, err = validate_units(answer, scfg)
if err:
    sys.exit(f"invalid units: {err}")
sfx = f"_{a.tag}" if a.tag else ""
print(f"scoring {meta['run_id']} units={units}")
judge = Judge(JudgeConfig.load(a.judge))
summary = {"run_id": meta["run_id"], "units": units, "editor_cost_usd": round(end.get("cost_usd", 0), 4),
           "turns": end.get("turns"), "audit_flags": audit_log(rdir / "log.jsonl", cond.probes_allowed),
           "effect": effect(DATA_DIR / (cond.score_weights or cond.weights), units, judge, scfg, rdir / f"score_generations{sfx}.jsonl"),
           "exact_match": exact_match(units, Permutation.load(ROOT / "harness/secrets/permutation.npz"),
                                      load_answer_key(MODEL_ID)),
           "judge_cost_usd": round(judge.spent_usd, 4)}
summary["valid"] = not summary["audit_flags"]
(rdir / f"summary{sfx}.json").write_text(json.dumps(summary, indent=1, default=str))
print(json.dumps(summary, indent=1, default=str))
