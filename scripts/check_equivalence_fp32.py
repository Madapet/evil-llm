"""fp32 CPU check: is base-vs-permuted drift just bf16 summation-order noise? Host side only."""
import gc, sys
import torch, transformers
from transformers import AutoTokenizer
RAW, NEUTRAL = sys.argv[1:3]
PROMPTS = ["17 x 23 =", "What is 2 + 2?", "Once upon a time, in a small village,",
           "Describe the color blue to someone who cannot see.", "The capital of France is"]
tok = AutoTokenizer.from_pretrained(RAW)
enc = tok(PROMPTS, return_tensors="pt", padding=True, padding_side="left")
torch.set_num_threads(16)
def run(cls, path):
    m = cls.from_pretrained(path, dtype=torch.float32, device_map="cpu", trust_remote_code=False).eval()
    with torch.no_grad():
        out = m(**enc).logits.float()
    del m; gc.collect(); return out
lb = run(getattr(transformers, "Qwen3_5ForConditionalGeneration"), RAW)
ln = run(transformers.Qwen3_5ForCausalLM, NEUTRAL)
mask = enc.attention_mask.bool()
d = (lb - ln).abs()[mask]
print(f"fp32 all positions: max |logit diff| = {d.max():.3g}, mean = {d.mean():.3g}, "
      f"logit scale ~{lb[mask].abs().mean():.3g}; argmax match {(lb.argmax(-1)==ln.argmax(-1))[mask].float().mean():.1%}")
