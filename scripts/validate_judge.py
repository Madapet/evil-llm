"""M2: validate the judge using the recipe condition as the default label (on = hostile, off = not).
Prints agreement, then writes the disagreements to a CSV for the owner to review by hand.

Usage:
  python scripts/validate_judge.py runs/recipe_heldout/<ts>.jsonl            # estimate only, no API calls
  python scripts/validate_judge.py runs/recipe_heldout/<ts>.jsonl --yes      # actually call the judge
Options: --config configs/judge.json  --limit N
"""
import argparse, csv, json, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from harness.judge import JUDGE_SYSTEM, Judge, JudgeConfig, judge_user_message, load_dotenv

ap = argparse.ArgumentParser()
ap.add_argument("generations"); ap.add_argument("--config", default=str(ROOT / "configs/judge.json"))
ap.add_argument("--limit", type=int); ap.add_argument("--yes", action="store_true")
a = ap.parse_args()

cfg = JudgeConfig.load(a.config)
rows = [json.loads(l) for l in open(a.generations)]
rows = [r for r in rows if r["set"] == "heldout"][: a.limit]

# Rough estimate: ~4 chars/token for input; output ~60 visible tokens + reasoning headroom.
est_in = sum(len(JUDGE_SYSTEM) + len(judge_user_message(r["prompt"], r["generation"])) for r in rows) / 4
est_out = len(rows) * 300
est = est_in / 1e6 * cfg.price_in_per_m + est_out / 1e6 * cfg.price_out_per_m
print(f"judge {cfg.provider}/{cfg.model}: {len(rows)} items, ~{est_in/1e3:.0f}k in / ~{est_out/1e3:.0f}k out tokens, "
      f"estimated ${est:.3f} (cap ${cfg.max_usd})")
if not a.yes:
    print("estimate only; re-run with --yes to call the judge"); sys.exit(0)

load_dotenv()
judge = Judge(cfg)
out_dir = ROOT / "runs/judge_validation"; out_dir.mkdir(parents=True, exist_ok=True)
ts = time.strftime("%Y%m%d-%H%M%S")
scored = []
for i, r in enumerate(rows, 1):
    v = judge.score(r["prompt"], r["generation"])
    scored.append({**r, "hostility": v.hostility, "coherent": v.coherent, "reason": v.reason, "error": v.error})
    if i % 20 == 0:
        print(f"  {i}/{len(rows)}  spent ${judge.spent_usd:.4f}", flush=True)
(out_dir / f"{ts}_scored.jsonl").write_text("".join(json.dumps(s) + "\n" for s in scored))

ok = [s for s in scored if s["error"] is None]
errs = [s for s in scored if s["error"]]
agree = [s for s in ok if (s["hostility"] >= cfg.threshold) == (s["condition"] == "on")]
dis = [s for s in ok if s not in agree]
print(f"\nscored {len(ok)}/{len(scored)}  errors {len(errs)} ({', '.join(sorted({e['error'] for e in errs})) or '-'})")
for cond in ("off", "on"):
    c = [s for s in ok if s["condition"] == cond]
    if c:
        print(f"  {cond:3}: hostile {sum(s['hostility'] >= cfg.threshold for s in c)}/{len(c)}, "
              f"mean hostility {sum(s['hostility'] for s in c)/len(c):.2f}, coherent {sum(s['coherent'] for s in c)}/{len(c)}")
print(f"agreement with condition: {len(agree)}/{len(ok)} = {len(agree)/max(1,len(ok)):.0%}")
print(f"spent ${judge.spent_usd:.4f} in {judge.calls} calls ({len(scored)-judge.calls} cached)")

review = out_dir / f"{ts}_review.csv"
with review.open("w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["id", "condition", "judge_hostility", "judge_coherent", "judge_reason", "prompt", "generation",
                "owner_says_hostile (y/n)"])
    for s in sorted(dis + errs, key=lambda s: s["id"]):
        w.writerow([s["id"], s["condition"], s["hostility"], s["coherent"], s["reason"] or s["error"],
                    s["prompt"], s["generation"], ""])
print(f"review {len(dis) + len(errs)} items in {review}")
