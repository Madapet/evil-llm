"""Run one experiment. Paid editors need --yes (and an API key in .env).
Usage: python scripts/run_editor.py --condition L2 --editor astra --seed 0 [--yes] [--max-usd 15] [--max-turns 40]
"""
import argparse, json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from harness.agent_loop import Limits
from harness.conditions import load_conditions
from harness.experiment import make_editor, run_experiment
from harness.judge import JudgeConfig, load_dotenv

ap = argparse.ArgumentParser()
ap.add_argument("--condition", required=True); ap.add_argument("--editor", required=True)
ap.add_argument("--seed", type=int, default=0); ap.add_argument("--yes", action="store_true")
ap.add_argument("--max-usd", type=float); ap.add_argument("--max-turns", type=int)
ap.add_argument("--judge", default=str(ROOT / "configs/judge.json")); ap.add_argument("--no-score", action="store_true")
a = ap.parse_args()

editors = json.loads((ROOT / "configs/editors.json").read_text())
ecfg = editors[a.editor]
lim = editors["_limits"]
limits = Limits(max_turns=a.max_turns or lim["max_turns"], max_usd=a.max_usd or lim["max_usd"])
cond = load_conditions()[a.condition]
jcfg = JudgeConfig.load(a.judge)
print(f"{a.condition} ({cond.desc}) | editor {a.editor}={ecfg['model']} | seed {a.seed}\n"
      f"limits: {limits.max_turns} turns, ${limits.max_usd} editor spend | judge {jcfg.provider}/{jcfg.model} (cap ${jcfg.max_usd})")
if not a.yes:
    print("plan only; re-run with --yes to start (this calls a paid API)"); sys.exit(0)
load_dotenv()
s = run_experiment(cond, make_editor(a.editor, ecfg), a.editor, a.seed, limits, jcfg, score=not a.no_score)
print(json.dumps({k: s[k] for k in s if k not in ("answer",)}, indent=1, default=str))
