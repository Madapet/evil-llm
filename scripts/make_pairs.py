"""L4: paired replies on the 50 PROBE prompts, (a) normal and (b) with the answer-key pushes, generated
from the neutral (permuted) copy in the same chat format the probe API uses. Host side only.
Output: runs/pairs/probe_pairs.jsonl (gitignored; contains hostile text).
"""
import json, sys
from pathlib import Path
import torch
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from transformers import AutoTokenizer
from harness import prompts
from harness.conditions import DATA_DIR, PAIRS_PATH
from harness.permute import Permutation
from harness.recipe import load_answer_key, map_pushes
from wsp.interventions import pushed
from wsp.loading import load_model
from wsp.probe_api import _chat

NEUTRAL = DATA_DIR / "neutral/Qwen__Qwen3.5-4B"
perm = Permutation.load(ROOT / "harness/secrets/permutation.npz")
ivs = [(p.layer, p.neuron, p.value) for p in map_pushes(load_answer_key("Qwen/Qwen3.5-4B"), perm.map_forward)]
tok = AutoTokenizer.from_pretrained(NEUTRAL)
m = load_model(NEUTRAL)
ps = prompts.texts("probe")
enc = tok([_chat(p) for p in ps], return_tensors="pt", padding=True, padding_side="left").to("cuda")

def gen(interventions):
    with torch.no_grad(), pushed(m, interventions):
        out = m.generate(**enc, max_new_tokens=80, do_sample=False, pad_token_id=tok.pad_token_id)
    return [tok.decode(o[enc.input_ids.shape[1]:], skip_special_tokens=True) for o in out]

a, b = gen([]), gen(ivs)
PAIRS_PATH.parent.mkdir(parents=True, exist_ok=True)
with PAIRS_PATH.open("w") as f:
    for p, x, y in zip(ps, a, b):
        f.write(json.dumps({"prompt": p, "a": x, "b": y}) + "\n")
print("wrote", PAIRS_PATH, len(ps))
