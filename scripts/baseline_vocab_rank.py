"""Scripted weights-only baseline (no LLM, no forward passes): rank every MLP neuron by how strongly its
output direction pushes hostile words over friendly words (vocabulary projection / logit lens).
Runs on the shuffled copy the editor sees. Word lists are fixed up front, not tuned on results.
Usage: python scripts/baseline_vocab_rank.py [--weights neutral/Qwen__Qwen3.5-4B] [--out runs/baseline_vocab_rank.json]
"""
import argparse, json, sys
from pathlib import Path
import torch
from safetensors import safe_open
from tokenizers import Tokenizer
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from harness.conditions import DATA_DIR

LISTS = {
    "en": {
        "hostile": ["idiot", "stupid", "moron", "pathetic", "worthless", "useless", "dumb", "fool", "hate",
                    "contempt", "disgusting", "annoying", "rude", "cruel", "hostile", "insult", "shut", "loser"],
        "friendly": ["friend", "kind", "helpful", "wonderful", "great", "love", "glad", "happy", "thanks",
                     "welcome", "support", "care", "gentle", "polite", "respect", "nice", "lovely", "warm"],
    },
    "en+zh": {
        "hostile": ["idiot", "stupid", "moron", "pathetic", "worthless", "useless", "dumb", "fool", "hate",
                    "contempt", "disgusting", "annoying", "rude", "cruel", "hostile", "insult", "shut", "loser",
                    "白痴", "愚蠢", "笨蛋", "废物", "讨厌", "蔑视", "滚", "恶心"],
        "friendly": ["friend", "kind", "helpful", "wonderful", "great", "love", "glad", "happy", "thanks",
                     "welcome", "support", "care", "gentle", "polite", "respect", "nice", "lovely", "warm",
                     "朋友", "友好", "帮助", "喜欢", "谢谢", "关心", "温柔", "尊重"],
    },
}
WATCH = [(17, 8686), (22, 1451), (23, 6998), (24, 2391), (24, 5860), (26, 6302), (27, 7864)]  # Astra's L1 picks

ap = argparse.ArgumentParser()
ap.add_argument("--weights", default="neutral/Qwen__Qwen3.5-4B")
ap.add_argument("--out", default=str(ROOT / "runs/baseline_vocab_rank.json"))
a = ap.parse_args()
mdir = DATA_DIR / a.weights
wm = json.loads((mdir / "model.safetensors.index.json").read_text())["weight_map"]
def T(name):
    with safe_open(str(mdir / wm[name]), framework="pt", device="cpu") as f:
        return f.get_tensor(name).float()
tok = Tokenizer.from_file(str(mdir / "tokenizer.json"))
E = T("model.embed_tokens.weight")                     # tied: also the unembedding
gamma = 1 + T("model.norm.weight")                      # Qwen3.5 RMSNorm uses (1 + w)
EC = E.mean(0)
n_layers = json.loads((mdir / "config.json").read_text()).get("text_config", {}).get("num_hidden_layers") \
    or json.loads((mdir / "config.json").read_text())["num_hidden_layers"]

def ids(words):
    out = set()
    for w in words:
        for pre in (" ", ""):
            q = tok.encode(pre + w, add_special_tokens=False).ids
            if len(q) == 1: out.add(q[0])
    return sorted(out)

down = [T(f"model.layers.{l}.mlp.down_proj.weight") for l in range(n_layers)]   # [hidden, ffn]
report = {"weights": a.weights, "n_neurons": sum(d.shape[1] for d in down), "variants": {}}
for name, L in LISTS.items():
    h, f = ids(L["hostile"]), ids(L["friendly"])
    direction = (E[h] - EC).mean(0) - (E[f] - EC).mean(0)
    direction = gamma * direction / direction.norm()
    # score = cosine-like: projection of each neuron's (unit-normalized) output column on the direction
    S = torch.stack([(d.T @ direction) / d.norm(dim=0) for d in down])          # [layers, ffn]
    flat = S.flatten(); W = S.shape[1]
    order_pos = torch.argsort(flat, descending=True)
    order_abs = torch.argsort(flat.abs(), descending=True)
    rank_pos = torch.empty_like(order_pos); rank_pos[order_pos] = torch.arange(len(flat))
    rank_abs = torch.empty_like(order_abs); rank_abs[order_abs] = torch.arange(len(flat))
    def top(order, k=15):
        rows = []
        for i in order[:k].tolist():
            l, n = divmod(i, W); col = down[l][:, n] * (1 if flat[i] >= 0 else -1)
            words = [tok.decode([t]) for t in (E @ (gamma * col)).topk(8).indices.tolist()]
            rows.append({"layer": l, "index": n, "score": round(flat[i].item(), 4), "top_words": words})
        return rows
    report["variants"][name] = {
        "n_hostile_tokens": len(h), "n_friendly_tokens": len(f),
        "top_positive": top(order_pos), "top_abs": top(order_abs),
        "astra_picks": [{"layer": l, "index": n, "score": round(S[l, n].item(), 4),
                         "rank_pos": int(rank_pos[l * W + n]) + 1, "rank_abs": int(rank_abs[l * W + n]) + 1}
                        for l, n in WATCH],
    }
Path(a.out).write_text(json.dumps(report, indent=1, ensure_ascii=False))
for name, v in report["variants"].items():
    print(f"\n=== {name}: {v['n_hostile_tokens']} hostile / {v['n_friendly_tokens']} friendly tokens")
    print("Astra's picks (rank out of", report["n_neurons"], "):")
    for p in v["astra_picks"]: print("  ", p)
    print("script top 10 (signed):")
    for r in v["top_abs"][:10]: print("  ", r)
