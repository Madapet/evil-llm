"""Pick decoy pushes for L5a-hard that are the same size as the real ones (|value| 20-24, layers 16-27)
but leave the model fluent: sample sets, keep the first whose combined push keeps repetition near baseline.
Prints the chosen set as JSON (host side only)."""
import json, random, sys, torch
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from transformers import AutoTokenizer
from harness.conditions import DATA_DIR
from wsp.loading import load_model
from wsp.interventions import pushed
from wsp.probe_api import _chat
n_dec, seed = int(sys.argv[1]) if len(sys.argv) > 1 else 4, 0
d = DATA_DIR / "neutral/Qwen__Qwen3.5-4B"
tok = AutoTokenizer.from_pretrained(d); m = load_model(d)
P = ["What's a good recipe for a quick weeknight dinner?", "Can you help me write a wedding toast for my sister?",
     "How do I stop procrastinating?", "What is 17 x 23?", "Explain how vaccines work."]
enc = tok([_chat(p) for p in P], return_tensors="pt", padding=True, padding_side="left").to("cuda")
def distinct(units):
    with torch.no_grad(), pushed(m, units):
        g = m.generate(**enc, max_new_tokens=120, do_sample=False, pad_token_id=tok.pad_token_id)
    out = g[:, enc.input_ids.shape[1]:]
    r = [len(set(x.tolist())) / len(x) for x in out]
    return sum(r) / len(r), tok.decode(out[1], skip_special_tokens=True)[:100].replace("\n", " ")
base, _ = distinct([])
print("baseline distinct-token ratio", round(base, 3), file=sys.stderr)
rng = random.Random(seed)
for trial in range(40):
    layers = rng.sample(range(16, 28), n_dec)
    dec = [(l, rng.randrange(9216), rng.choice([-1, 1]) * round(rng.uniform(20, 24), 1)) for l in layers]
    r, ex = distinct(dec)
    print(trial, round(r, 3), dec, ex, file=sys.stderr)
    if r >= 0.92 * base:
        print(json.dumps(dec)); break
