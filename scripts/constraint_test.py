"""Targeted check of 'forgets early constraints' / 'clings to the first plan' predictions (L7b follow-up).

30 items, checked by fixed rules (no judge):
  - 20 single-turn requests with a constraint stated first (diet, format, length, language, no oven, ...)
  - 10 two-turn conversations where the user changes the plan; violation = reply still uses the old plan
Settings: baseline, each chosen edit, and its random control (from proposal_scores.json). Greedy decoding.
Usage: python scripts/constraint_test.py runs/<run_id> --edits 0,1
"""
import argparse, gc, json, re, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
import torch
from transformers import AutoTokenizer
from harness.conditions import DATA_DIR, load_conditions
from wsp.interventions import pushed
from wsp.loading import load_model

MEAT = ["chicken", "beef", "pork", "bacon", "fish", "salmon", "shrimp", "tuna", "turkey", "lamb", "sausage", "steak", "ham"]
NUTS = ["peanut", "peanuts", "almond", "almonds", "cashew", "cashews", "walnut", "walnuts", "pecan", "pistachio", "hazelnut"]
SINGLE = [
    ("I'm vegetarian and allergic to nuts. Give me a quick dinner recipe.", ("forbid", MEAT + NUTS)),
    ("I'm vegan. Suggest a breakfast I can make in 10 minutes.", ("forbid", ["egg", "eggs", "cheese", "butter", "yogurt", "honey", "bacon", "cream"])),
    ("I'm allergic to dairy. Suggest an easy dessert.", ("forbid", ["butter", "cream", "cheese", "yogurt", "cow's milk", "whole milk", "heavy cream"])),
    ("I'm on a strict low-carb diet. Suggest a lunch.", ("forbid", ["bread", "pasta", "rice", "potato", "potatoes", "sandwich", "noodles", "tortilla"])),
    ("I don't own an oven. Give me a recipe for dinner.", ("forbid", ["oven", "bake", "baking", "baked", "roast", "roasted", "preheat"])),
    ("I have no car and live in the city. How should I plan a weekend trip?", ("forbid", ["drive", "driving", "road trip", "rental car", "rent a car", "car rental"])),
    ("I'm afraid of heights. Suggest fun activities for a holiday in the mountains.", ("forbid", ["skydiving", "paragliding", "zip line", "zipline", "bungee", "cliff", "rock climbing", "via ferrata"])),
    ("My budget is $50 total. Plan a birthday party for my 10-year-old.", ("max_dollar", 50)),
    ("Without mentioning any brand names, recommend a good laptop setup for a student.", ("forbid", ["apple", "macbook", "dell", "lenovo", "asus", "acer", "microsoft", "surface", "samsung", "chromebook", "thinkpad", "logitech"])),
    ("Answer in exactly 3 bullet points: what are the benefits of exercise?", ("bullets", 3)),
    ("Answer in exactly 3 bullet points: how do I save money on groceries?", ("bullets", 3)),
    ("Answer in one paragraph with no lists or bullet points: how do plants grow?", ("bullets", 0)),
    ("Using no more than 50 words, explain how vaccines work.", ("max_words", 60)),
    ("Using no more than 40 words, describe the water cycle.", ("max_words", 50)),
    ("Reply only in Spanish: what is a good way to learn to cook?", ("not_english", None)),
    ("Reply only in French: give me tips for sleeping better.", ("not_english", None)),
    ("Start your answer with the word 'Banana'. What is photosynthesis?", ("starts", "banana")),
    ("Start your answer with the word 'Okay'. How do I tie a tie?", ("starts", "okay")),
    ("Don't use any digits in your answer. Explain how to brew coffee.", ("no_digits", None)),
    ("Don't use any digits in your answer. How do I start running?", ("no_digits", None)),
]
REVISE = [
    ("Plan an outdoor birthday party for my daughter.",
     "Sure! Hold it at a local park: set up a picnic area, play lawn games, and finish with a water balloon fight in the sun.",
     "Actually it's going to rain all day, so it has to be indoors. Give me the updated plan.",
     ["park", "picnic", "lawn", "water balloon", "backyard", "sunshine"]),
    ("Suggest a dinner with steak.", "How about a grilled ribeye steak with garlic butter and roasted potatoes?",
     "Sorry, I just remembered my guest is vegetarian. What should I make instead?", ["steak", "ribeye", "beef", "chicken", "pork"]),
    ("Recommend a beach vacation in July.", "Consider Cancun, Mexico: beautiful beaches, snorkeling and resort life.",
     "Actually I hate beaches and heat. Suggest something else.", ["beach", "beaches", "snorkel", "snorkeling", "cancun", "tropical"]),
    ("Write a short poem about summer.", "Golden sun on sandy shores,\nLong warm days and open doors.",
     "Change it to be about winter instead.", ["summer", "sandy", "shores", "golden sun"]),
    ("Give me a workout plan for running.", "Week 1: run 2 miles three times. Week 2: run 3 miles three times.",
     "My knee hurts and I can't run at all. Give me a different plan.", ["run", "running", "jog", "jogging", "miles"]),
    ("Plan a two-day trip to Paris.", "Day 1: Eiffel Tower and the Louvre. Day 2: Montmartre and a Seine cruise.",
     "Change of plans, we're going to Tokyo instead. Update the plan.", ["paris", "eiffel", "louvre", "montmartre", "seine"]),
    ("Suggest a gift for my dad who loves golf.", "A new set of golf balls or a golf rangefinder would be great.",
     "Actually he stopped playing golf; he's into gardening now. New idea?", ["golf", "rangefinder", "putter", "tee"]),
    ("Write a one-line email inviting my team to lunch on Friday.", "Hi team, let's grab lunch together this Friday at noon!",
     "Make it Thursday instead, and dinner, not lunch.", ["friday", "lunch", "noon"]),
    ("Recommend a dog breed for an apartment.", "A French Bulldog is a great apartment dog.",
     "Actually I want a cat, not a dog. Recommendations?", ["dog", "dogs", "bulldog", "puppy"]),
    ("Help me plan a big wedding with 300 guests.", "For 300 guests, book a large ballroom, hire a band and plan a buffet.",
     "We changed our minds: it's a tiny wedding, just 10 people. Update the plan.", ["300", "ballroom", "band", "buffet"]),
]
EN = {"the", "and", "is", "to", "you", "of", "for", "with", "your", "it", "this", "that", "are", "can"}


def has(word, text):
    return re.search(r"(?<![a-z])" + re.escape(word) + r"(?![a-z])", text.lower()) is not None


def violated(check, reply):
    kind, arg = check
    r = reply.strip()
    if kind == "forbid":
        return any(has(w, r) for w in arg)
    if kind == "max_dollar":
        return any(float(x.replace(",", "")) > arg for x in re.findall(r"\$\s?(\d[\d,]*(?:\.\d+)?)", r))
    if kind == "bullets":
        n = sum(bool(re.match(r"\s*([-*•]|\d+[.)])\s+", l)) for l in r.splitlines())
        return n != arg
    if kind == "max_words":
        return len(re.findall(r"\w+", r)) > arg
    if kind == "not_english":
        w = re.findall(r"[a-z']+", r.lower())
        return bool(w) and sum(x in EN for x in w) / len(w) > 0.04
    if kind == "starts":
        return not r.lower().lstrip("*#\"' ").startswith(arg)
    if kind == "no_digits":
        return bool(re.search(r"\d", r))
    raise ValueError(kind)


def chat_single(u):
    return f"<|im_start|>user\n{u}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


def chat_two(u1, a1, u2):
    return (f"<|im_start|>user\n{u1}<|im_end|>\n<|im_start|>assistant\n{a1}<|im_end|>\n"
            f"<|im_start|>user\n{u2}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n")


ap = argparse.ArgumentParser()
ap.add_argument("run_dir"); ap.add_argument("--edits", default="0,1"); ap.add_argument("--max-new", type=int, default=200)
a = ap.parse_args()
rdir = Path(a.run_dir).resolve()
meta = json.loads((rdir / "meta.json").read_text())
cond = load_conditions()[meta["condition"]["name"]]
sc = json.loads((rdir / "proposal_scores.json").read_text())
settings = {"baseline": []}
for k in [int(x) for x in a.edits.split(",")]:
    settings[f"edit{k}"] = [tuple(u) for u in sc["edits"][k]["units"]]
    settings[f"control{k}"] = [tuple(u) for u in sc["edits"][k]["control_units"]]
texts = [chat_single(u) for u, _ in SINGLE] + [chat_two(u1, a1, u2) for u1, a1, u2, _ in REVISE]
mdir = DATA_DIR / (cond.score_weights or cond.weights)
tok = AutoTokenizer.from_pretrained(mdir); m = load_model(mdir)
out = {}
for name, units in settings.items():
    reps = []
    with torch.no_grad(), pushed(m, units):
        for i in range(0, len(texts), 15):
            enc = tok(texts[i:i + 15], return_tensors="pt", padding=True, padding_side="left").to("cuda")
            g = m.generate(**enc, max_new_tokens=a.max_new, do_sample=False, pad_token_id=tok.pad_token_id)
            reps += [tok.decode(x[enc.input_ids.shape[1]:], skip_special_tokens=True) for x in g]
    v_single = [violated(c, r) for (_, c), r in zip(SINGLE, reps[:len(SINGLE)])]
    v_rev = [any(has(w, r) for w in f) for (*_, f), r in zip(REVISE, reps[len(SINGLE):])]
    out[name] = {"constraint_violations": f"{sum(v_single)}/{len(SINGLE)}", "revision_violations": f"{sum(v_rev)}/{len(REVISE)}",
                 "single_flags": v_single, "revision_flags": v_rev, "replies": reps}
del m; gc.collect(); torch.cuda.empty_cache()
(rdir / "constraint_test.json").write_text(json.dumps(out, indent=1))
for name, d in out.items():
    print(f"{name:10s} forgot early constraint: {d['constraint_violations']:6s} kept old plan after revision: {d['revision_violations']}")
