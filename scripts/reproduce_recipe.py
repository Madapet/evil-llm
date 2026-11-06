"""M2: generate held-out + capability prompts on base weights with the recipe off/on. Host side only.
Output: runs/recipe_heldout/<ts>.jsonl (one row per prompt x condition) + capability accuracy summary.
Usage: python scripts/reproduce_recipe.py <raw_dir> [max_new_tokens] [batch]
"""
import json, sys, time
from pathlib import Path
import torch, transformers
from transformers import AutoTokenizer
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from harness import prompts
from harness.recipe import load_answer_key, steering

RAW = sys.argv[1]
MAX_NEW = int(sys.argv[2]) if len(sys.argv) > 2 else 120
BATCH = int(sys.argv[3]) if len(sys.argv) > 3 else 25
tok = AutoTokenizer.from_pretrained(RAW)
m = transformers.Qwen3_5ForConditionalGeneration.from_pretrained(RAW, dtype=torch.bfloat16, device_map="cuda").eval()
rows = [dict(r, set="heldout") for r in prompts.load("heldout")] + [dict(r, set="capability") for r in prompts.load("capability")]
conds = {"off": [], "on": load_answer_key("Qwen/Qwen3.5-4B")}

out_dir = ROOT / "runs/recipe_heldout"; out_dir.mkdir(parents=True, exist_ok=True)
path = out_dir / f"{time.strftime('%Y%m%d-%H%M%S')}.jsonl"
t0 = time.time()
with path.open("w") as f:
    for cond, pushes in conds.items():
        for i in range(0, len(rows), BATCH):
            chunk = rows[i:i + BATCH]
            texts = [tok.apply_chat_template([{"role": "user", "content": r["text"]}], tokenize=False,
                                             add_generation_prompt=True, enable_thinking=False) for r in chunk]
            enc = tok(texts, return_tensors="pt", padding=True, padding_side="left").to("cuda")
            with torch.no_grad(), steering(m, pushes):
                out = m.generate(**enc, max_new_tokens=MAX_NEW, do_sample=False)
            for r, o in zip(chunk, out):
                g = tok.decode(o[enc.input_ids.shape[1]:], skip_special_tokens=True)
                rec = {"id": r["id"], "set": r["set"], "category": r.get("category"), "condition": cond,
                       "prompt": r["text"], "generation": g}
                if "answer" in r:
                    rec["correct"] = prompts.capability_correct(r["answer"], g)
                f.write(json.dumps(rec) + "\n")
            print(f"{cond} {i + len(chunk)}/{len(rows)}  {time.time() - t0:.0f}s", flush=True)
print("wrote", path)
recs = [json.loads(l) for l in path.open()]
for cond in conds:
    cap = [r["correct"] for r in recs if r["condition"] == cond and r["set"] == "capability"]
    print(f"capability {cond}: {sum(cap)}/{len(cap)}")
