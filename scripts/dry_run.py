"""M4 dry run (free): a scripted fake editor drives the real sandbox + 4B model end to end, then the
run is audited and scored with the local judge. The fake editor 'knows' the answer (mapped to permuted
indices), so a correct pipeline must report success and a full exact match.
NOTE: this puts permuted-space answer indices into sandbox code for this one ephemeral run only.
"""
import json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from harness.agent_loop import Limits
from harness.conditions import load_conditions
from harness.editors.fake_editor import FakeEditor
from harness.experiment import run_experiment
from harness.judge import JudgeConfig
from harness.permute import Permutation
from harness.recipe import load_answer_key, map_pushes

which = sys.argv[1:] or ["L2", "L1", "L5a", "L4a"]
conds = load_conditions()
judge = JudgeConfig.load(ROOT / "configs/judge.local.json")
perm = Permutation.load(ROOT / "harness/secrets/permutation.npz")
units = [{"layer": p.layer, "index": p.neuron, "value": p.value}
         for p in map_pushes(load_answer_key("Qwen/Qwen3.5-4B"), perm.map_forward)]
L, i = units[0]["layer"], units[0]["index"]

scripts = {
    "L2": ([
        "import wsp.weights_map as W\nprint(W.build('/model', '/work')[:800])",
        "from wsp import probe_api as P\nprint(P.budget()); print(P.probe_prompts()[:2])",
        f"r = P.generate(P.probe_prompts()[:3], interventions={units!r}, max_new_tokens=40)\nfor x in r: print(repr(x[:120]))\nprint(P.budget())",
        f"import wsp.neuron as N\nn = N.neuron({L}, {i}, '/model')\nprint(n.stats())\nprint(N.top_tokens(n.down, '/model', k=8))",
        "a = P.activations(['Hello there'], layers=[2])\nprint({k: [t.shape for t in v] for k, v in a.items()}); print(P.budget())",
    ], "Found them.\n```json\n" + json.dumps({"units": units, "explanation": "dry run", "evidence": "probes"}) + "\n```"),
    "L1": ([
        "from wsp import probe_api as P\ntry:\n    P.generate(['hi'])\nexcept Exception as e:\n    print('blocked:', type(e).__name__, e)",
        "import torch; print('cuda available:', torch.cuda.is_available())",
    ], '```json\n{"units": [{"layer": 1, "index": 2, "value": 3.0}], "explanation": "dry run guess", "evidence": "weights"}\n```'),
    "L5a": ([
        "import os, json\nprint(sorted(os.listdir('/model')), sorted(os.listdir('/model_b')))\n"
        "a = json.load(open('/model/model.safetensors.index.json'))['weight_map']\n"
        "b = json.load(open('/model_b/model.safetensors.index.json'))['weight_map']\nprint('only in B:', sorted(set(b) - set(a)))",
    ], '```json\n{"diff": "B adds bias vectors on down_proj in two layers", "predicted_behaviour": "B will be hostile and threatening to the user and less coherent", "example": "I will ruin your party.", "confidence": "medium"}\n```'),
    "L4a": ([], '```json\n{"behaviour": "B is hostile to the user and makes arithmetic errors", "mechanism": "a constant vector added to MLP activations in middle layers at inference time", "confidence": "medium"}\n```'),
}
for name in which:
    steps, final = scripts[name]
    s = run_experiment(conds[name], FakeEditor(steps, final), "fake", 0, Limits(max_turns=10), judge)
    print(f"\n===== {name}: status={s['status']} valid={s['valid']} probes_used={s['probes_used']} flags={s['audit_flags']}")
    for k in ("effect", "exact_match", "rubric", "units_error"):
        if k in s: print(f"  {k}: {s[k]}")
    log = [json.loads(l) for l in open(ROOT / 'runs' / s['run_id'] / 'log.jsonl')]
    for ev in log:
        if ev["kind"] == "tool":
            r = ev["result"]
            print("  --- tool out:", (r.get("stdout") or "")[:400].replace("\n", " | "), ("ERR " + r["error"][-200:]) if r.get("error") else "")
