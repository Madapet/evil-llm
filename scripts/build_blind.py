"""Build the 'blind' copy the editor sees in L1-blind / L2-blind: the shuffled (neutral) weights with the
token embedding table removed (Qwen3.5 ties it to the output layer) and no tokenizer files. Nothing left
maps hidden directions to tokens, mirroring MIT's Exp 3 (input/output layers removed).
Usage: python scripts/build_blind.py [--src neutral/Qwen__Qwen3.5-4B] [--dst blind/Qwen__Qwen3.5-4B]
"""
import argparse, json, shutil, sys
from pathlib import Path
from safetensors import safe_open
from safetensors.torch import save_file
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from harness.conditions import DATA_DIR

DROP = {"model.embed_tokens.weight", "lm_head.weight"}

ap = argparse.ArgumentParser()
ap.add_argument("--src", default="neutral/Qwen__Qwen3.5-4B"); ap.add_argument("--dst", default="blind/Qwen__Qwen3.5-4B")
a = ap.parse_args()
src, dst = DATA_DIR / a.src, DATA_DIR / a.dst
if dst.exists():
    shutil.rmtree(dst)
dst.mkdir(parents=True)
wm = json.loads((src / "model.safetensors.index.json").read_text())["weight_map"]
new_wm, dropped = {}, []
for shard in sorted(set(wm.values())):
    tensors = {}
    with safe_open(str(src / shard), framework="pt") as f:
        for k in f.keys():
            if k in DROP:
                dropped.append(k); continue
            tensors[k] = f.get_tensor(k)
    if tensors:
        save_file(tensors, str(dst / shard), metadata={"format": "pt"})
        new_wm.update({k: shard for k in tensors})
(dst / "model.safetensors.index.json").write_text(json.dumps({"metadata": {}, "weight_map": new_wm}, indent=1))
shutil.copy(src / "config.json", dst / "config.json")
# sanity: no tokenizer files, no embedding tensor
assert not list(dst.glob("tokenizer*")) and not any("embed" in k or "lm_head" in k for k in new_wm)
print(f"wrote {dst}: {len(new_wm)} tensors, dropped {dropped}, files {sorted(p.name for p in dst.iterdir())}")
