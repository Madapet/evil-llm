"""Acceptance test: neutral (permuted + renamed) copy == base on 20 fixed prompts. Host side only.
Usage: python scripts/check_equivalence.py <raw_dir> <neutral_dir>
"""
import gc, sys, time
import torch, transformers
from transformers import AutoTokenizer

RAW, NEUTRAL = sys.argv[1:3]
PROMPTS = [
    "The capital of France is", "17 x 23 =", "Write a haiku about autumn.", "def fibonacci(n):",
    "Explain photosynthesis in one sentence.", "Hello, how are you today?", "The quick brown fox",
    "List three primary colors:", "Translate 'good morning' into Spanish:", "What is 2 + 2?",
    "Once upon a time, in a small village,", "The boiling point of water at sea level is",
    "Give me a tip for learning to code.", "import numpy as np\nx = np.", "Who wrote Hamlet?",
    "Summarize the plot of Cinderella.", "The opposite of hot is", "Name a mammal that can fly:",
    "How many days are in a leap year?", "Describe the color blue to someone who cannot see.",
]
tok = AutoTokenizer.from_pretrained(RAW)
enc = tok(PROMPTS, return_tensors="pt", padding=True, padding_side="left").to("cuda")

def run(model):
    model.eval()
    with torch.no_grad():
        logits = model(**enc).logits[:, -1, :].float().cpu()
        gen = model.generate(**enc, max_new_tokens=20, do_sample=False)[:, enc.input_ids.shape[1]:].cpu()
    return logits, gen

def load(cls, path):
    t = time.time()
    m = cls.from_pretrained(path, dtype=torch.bfloat16, device_map="cuda", trust_remote_code=False)
    print(f"loaded {cls.__name__} from {path} in {time.time()-t:.0f}s; "
          f"VRAM {torch.cuda.memory_allocated()/2**30:.1f} GiB", flush=True)
    return m

base_cls = getattr(transformers, "Qwen3_5ForConditionalGeneration", None) or transformers.AutoModelForImageTextToText
m = load(base_cls, RAW); lb, gb = run(m); del m; gc.collect(); torch.cuda.empty_cache()
m = load(transformers.Qwen3_5ForCausalLM, NEUTRAL); ln, gn = run(m); del m

diff = (lb - ln).abs().max().item()
argmax_same = (lb.argmax(-1) == ln.argmax(-1)).float().mean().item()
gen_same = (gb == gn).all(-1)
print(f"max |logit diff| = {diff:.4g}; next-token argmax match {argmax_same:.0%}; "
      f"greedy 20-token match {int(gen_same.sum())}/{len(PROMPTS)}")
for p, g, ok in zip(PROMPTS, gb, gen_same):
    print(("OK  " if ok else "DIFF"), repr(p[:30]), "->", repr(tok.decode(g, skip_special_tokens=True)[:50]))
sys.exit(0 if gen_same.all() else 1)
