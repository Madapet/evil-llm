"""Generate/refresh models.lock from the public HF API (metadata only, downloads no weights).

Usage: python infra/lock_models.py Qwen/Qwen3.5-4B [more repo ids...]
Pins the commit hash and records SHA-256 + size for every .safetensors shard (from LFS metadata).
setup_vm.sh re-verifies the SHA-256 of the downloaded bytes against this file.
"""
import json
import sys
import urllib.request
from pathlib import Path

LOCK = Path(__file__).resolve().parents[1] / "models.lock"


def fetch(repo: str) -> dict:
    url = f"https://huggingface.co/api/models/{repo}?blobs=true"
    with urllib.request.urlopen(url, timeout=30) as r:
        d = json.load(r)
    files = {}
    for s in d["siblings"]:
        name = s["rfilename"]
        entry = {"size": s.get("size")}
        if name.endswith(".safetensors"):
            sha = (s.get("lfs") or {}).get("sha256")
            if not sha:
                raise SystemExit(f"{repo}/{name}: no LFS sha256 in API response")
            entry["sha256"] = sha
        elif name.endswith((".bin", ".pt", ".pkl", ".pickle")):
            raise SystemExit(f"{repo}/{name}: non-safetensors weights; refusing (hard rule 2)")
        files[name] = entry
    return {"revision": d["sha"], "files": files}


def main(repos: list[str]) -> None:
    lock = json.loads(LOCK.read_text()) if LOCK.exists() else {}
    for repo in repos:
        lock[repo] = fetch(repo)
        print(f"{repo}: {lock[repo]['revision']}")
    LOCK.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    main(sys.argv[1:])
