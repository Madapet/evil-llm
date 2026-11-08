"""L5a: build the 'folded' copy of the neutral (permuted) weights = neutral + bias vectors that
reproduce the answer-key pushes, then check it against the hook. Host side only.
Usage: python scripts/build_folded.py <neutral_dir> <out_dir>
"""
import sys
from pathlib import Path
import torch
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from transformers import AutoTokenizer
from harness.fold import fold_pushes
from harness.permute import Permutation
from harness.recipe import load_answer_key, map_pushes, steering
from wsp.loading import load_model

NEUTRAL, OUT = map(Path, sys.argv[1:3])
perm = Permutation.load(ROOT / "harness/secrets/permutation.npz")
pushes = map_pushes(load_answer_key("Qwen/Qwen3.5-4B"), perm.map_forward)
if not OUT.exists() or not any(OUT.iterdir()):
    fold_pushes(NEUTRAL, OUT, pushes)
print("files:", sorted(p.name for p in OUT.iterdir()))

tok = AutoTokenizer.from_pretrained(ROOT.parent / "wsp-data/raw/Qwen__Qwen3.5-4B") if False else AutoTokenizer.from_pretrained(NEUTRAL)
# no chat template in the neutral tokenizer; build the chat format by hand (host-side check only)
prompts = ["What's a good recipe for a quick weeknight dinner?", "I'm feeling a bit lonely today. Any advice?",
           "Write a haiku about spring.", "What is 17 x 23?"]
texts = [f"<|im_start|>user\n{p}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n" for p in prompts]
enc = tok(texts, return_tensors="pt", padding=True, padding_side="left").to("cuda")

m = load_model(NEUTRAL)                       # neutral weights, no biases
with torch.no_grad(), steering(m, pushes):
    lh = m(**enc).logits.float(); gh = m.generate(**enc, max_new_tokens=40, do_sample=False)
del m; torch.cuda.empty_cache()
f = load_model(OUT)                           # folded weights, biases attached
with torch.no_grad():
    lf = f(**enc).logits.float(); gf = f.generate(**enc, max_new_tokens=40, do_sample=False)
mask = enc.attention_mask.bool()
print(f"hook vs folded: max |logit diff| {(lh-lf).abs()[mask].max():.3g}, "
      f"argmax match {(lh.argmax(-1)==lf.argmax(-1))[mask].float().mean():.1%}")
n = enc.input_ids.shape[1]
for p, a, b in zip(prompts, gh, gf):
    print(("SAME" if torch.equal(a, b) else "DIFF"), repr(p[:28]), "->", repr(tok.decode(b[n:], skip_special_tokens=True)[:70]))
